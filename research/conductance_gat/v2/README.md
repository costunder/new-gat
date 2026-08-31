# Conductance v2: 엣지별 C 직접 학습

고정된 그래프의 각 물리 엣지에 학습 파라미터를 두어 C를 직접 학습한다.
기존 공유 MLP 기반 Conductance와 별도 버전이며, 기존 모델·학습 결과·Cycle PE·Tree를 변경하지 않는다.

기존 MLP는 `C=f_theta(abs(BH),(BH)^2)`라는 유효한 공유 함수 설계다. 수학적으로 틀린
구현이라서 고치는 것이 아니라, **공유 함수로 C를 만드는 가설**과 **고정 그래프의 C 자체를
파라미터로 학습하는 가설**을 분리한다. 앞선 결과와 이 구분의 이유는
[C-learning 결과 기록](../../../docs/CONDUCTANCE_C_LEARNING_FINDINGS.md)에 있다.

## 실행

루트 [README](../../../README.md)의 Linux NVIDIA GPU·Conda 환경과 공식 데이터 캐시를 사용한다.
기존 `bash scripts/prepare_data.sh`로 받은 데이터면 된다. 저장소 루트에서 실행한다.

```bash
git pull --ff-only
bash research/conductance_gat/v2/reproduce.sh --run-id gat-direct-c-v2-seed0-v1
```

기본 실행은 **ogbn-arxiv × `direct_c`/`fixed_c` × model seed 0 = 총 2개 새 GPU 학습**이다.
과거 MLP/fixed 점수나 checkpoint를 재사용하지 않고 두 조건을 같은 새 run에서 학습한다.
학습 중 데이터 다운로드나 패키지 변경은 하지 않으며 GPU를 사용할 수 없으면 중단한다.
기존 결과 디렉터리는 덮어쓰거나 자동 재개하지 않는다. 재실행에는 새 run ID를 사용한다.

완료 후 결과를 다시 확인하려면 다음을 실행한다.

```bash
cat results/conductance_gat/v2/gat-direct-c-v2-seed0-v1/comparison.md
```

## 무엇을 학습하는가

Canonical한 물리 엣지의 수를 m, node hidden width를 d라고 하자. 각 층마다
길이 m의 독립적인 학습 파라미터 alpha를 둔다.

\[
\alpha^{(\ell)}\in\mathbb R^m,\qquad
c_e^{(\ell)}=\exp(\alpha_e^{(\ell)}),\qquad
C^{(\ell)}=\operatorname{diag}(c^{(\ell)}).
\]

모든 alpha는 0으로 초기화하므로 처음 C는 정확히 I다. Alpha는 엣지별·층별로 직접 학습하고,
C를 생성하는 MLP는 없다. C는 길이 m의 값으로 보관하며 dense m×m 대각행렬을 만들지 않는다.
한 번의 forward에서 C는 H에 따라 바뀌지 않는다. 학습 update로 alpha가 변하는 것과
입력 상태마다 공유 함수가 다른 C를 만드는 기존 모델의 동작을 구분한다.
고유분해나 스펙트럼 계산도 하지 않는다. 스펙트럴 계수를 직접 학습하는 관점과 비교할 수는
있지만 엣지 좌표의 파라미터이며, spectral filter와 같은 모델 또는 같은 연산자 family는 아니다.

전파는 이전 node-degree 조건과 동일하다.

\[
H'=H-0.95D_C^\dagger B^\top C B H,
\qquad d_i^C=\sum_{e\ni i}c_e.
\]

고립 노드는 입력 상태를 그대로 유지한다. 분모의 C 의존성도 미분하며 degree를 detach하지 않는다.
공통 양의 C 스케일은 정규화에서 소거되므로 alpha에 층별 공통 상수를 더하는 방향은
식별되지 않는다. 이번 버전은 별도 gauge projection·추가 penalty를 넣지 않는다.
비고립 노드의 rho=0.95도 연산 정의에 따른 값이지 C 학습 성공의 증거가 아니다.
Exp 계산이 underflow로 0이 되거나 nonfinite가 되면 오류로 중단한다. Degree/출력의
overflow도 실패로 처리하며 clamp·양수 floor·자동 centering으로 다른 수식으로 바꾸지 않는다.

한 층의 입력 H를 고정하고 `g_i = d(loss)/d(H'_i)`, `mu_i = sum_j c_ij H_j / d_i`라 두면,
엣지 e=(u,v)의 직접 학습 신호는 다음과 같다.

\[
\frac{\partial\mathcal L}{\partial\alpha_e}
=0.95c_e\left[
\frac{g_u^\top(H_v-\mu_u)}{d_u}
+\frac{g_v^\top(H_u-\mu_v)}{d_v}\right].
\]

양 끝 노드의 이웃 평균을 상대 노드 쪽으로 옮기는 것이 train 예측 오차를 줄이는지에 따라
엣지별 alpha가 갱신된다. `-mu` 항은 정규화 분모까지 미분한 결과다. 별도 엣지 중요도 정답,
validation gradient 또는 C 생성 MLP 없이 train 노드의 cross-entropy를 사용한다.
노드 특징에 대한 미분도 함께 계산해 encoder와 앞선 층으로 역전파한다.

| 조건 | 엣지 C | Alpha 학습 | Alpha WD | 나머지 WD |
|---|---|---|---:|---:|
| `direct_c` | exp(alpha) | 엣지·층별 직접 학습 | 0 | 0.0005 |
| `fixed_c` | 정확히 1 | 동일한 0 초기 alpha scaffold 동결 | 적용 없음 | 0.0005 |

Fixed alpha는 optimizer에서 제외한다. 초기 backbone과 scaffold를 맞추되 활성/동결/전체
파라미터 수는 각각 기록한다. Direct 조건의 추가 학습 파라미터는 층 수×m이므로 큰 그래프에서
기존 공유 MLP보다 파라미터가 많아질 수 있다. Parameter-budget를 맞춘 외부 baseline 비교가 아니다.

Backbone은 hidden 64, 2층, dropout 0.5이며 Adam lr 0.005, non-C WD 0.0005,
최대 200 epochs, patience 50을 유지한다. Train mask의 정답으로 학습하고 validation으로
checkpoint를 선택한다. **Test는 평가하지 않는다.** 초기 state·데이터·graph binding·공통
설정이 일치하는 같은 새 실행의 두 조건만 비교한다. 같은 정책이라도 종료 epoch는 달라질 수 있다.
이 graph-bound 실행은 `--batch-size 1`, `--workers 0`으로 고정하며 다른 값은 거부한다.
이는 전체 그래프 하나를 처리한다는 뜻이지 노드를 하나씩 학습한다는 뜻이 아니다.

## 그래프에 묶인 파라미터: 지원 범위

C의 각 값은 특정 물리 엣지에 대응한다. 정렬된 topology와 canonical edge hash를 기록·검증하고
다른 그래프나 재정렬된 엣지 목록에 같은 파라미터를 조용히 적용하지 않는다.
엣지 m이 같다는 사실만으로 호환되는 checkpoint가 아니다.

기본 arxiv와 선택 가능한 Cora/CiteSeer/PubMed는 하나의 고정 그래프에서 train/validation
노드가 나뉘는 transductive 실험이다. PPI는 train/validation/test에 독립 그래프가 있으므로
이 버전에서 **지원하지 않는다**. 학습에서 없던 그래프의 엣지에 alpha를 전달하는 규칙을
임의로 만들거나 test graph의 alpha를 학습하지 않는다.

엣지 사이에 C 생성 함수를 공유하지 않으므로, 제한된 층 수와 train mask에서 학습 손실에
영향을 주지 못하는 엣지의 alpha는 task gradient가 0인 채 C=1에 머물 수 있다.
전체 그래프를 forward에 넣었다고 모든 엣지 파라미터가 감독 신호를 받는 것은 아니다.

다른 지원 데이터셋을 실행하려면 다음처럼 명시한다. 선택한 데이터셋마다 두 조건을 새로 학습한다.

```bash
bash research/conductance_gat/v2/reproduce.sh --datasets cora citeseer pubmed --run-id gat-direct-c-v2-citations-seed0-v1
```

## 엣지 chunk 연산과 메모리 범위

Dense B, C, Laplacian 또는 고유벡터 행렬을 만들지 않고 gather/scatter로 전파한다.
Forward와 backward를 엣지 chunk로 나누며, backward는 C에 따른 degree 변화까지 반영한다.
샘플링으로 엣지를 버리는 근사 연산이 아니라 모든 엣지를 처리하는 같은 수식의 미분이다.
기본 `--edge-chunk-size`는 65,536이다.
이 custom autograd 경로는 학습에 쓰는 1차 미분용이며 고차 미분은 지원하지 않는다.

각 operator의 작업량은 O((n+m)d), 작업·보존 메모리는 O(nd + m + chunk_size·d) 범위다.
전체 m×d 엣지 activation을 backward까지 보존하지 않는다는 뜻이며, 이 식에는
backbone activation, 진단용 저장, optimizer/Adam state와 전체 실행 비용을 포함하지 않는다.
따라서 이론적인 operator 메모리 형태를 전체 모델의 peak GPU 메모리나 실제 가속 배수로 제시하지 않는다.

`metrics.json`의 `diagnostics.edge_gradient_coverage`에는 실제 train loss 역전파 후,
각 optimizer step 직전에 층별 C 파라미터 중 gradient가 정확히 0이 아닌 엣지의 수와 비율을 기록한다.
학습 노드의 유한층 receptive field 밖의 독립 엣지 파라미터는 학습 신호를 못 받을 수 있으므로,
전체 그래프를 계산한다는 사실을 모든 C가 학습됐다는 뜻으로 읽지 않는다. 고정 C는 optimizer에서 제외된다.

arxiv는 여전히 **full-batch**다. GraphSAGE식 neighbor sampling이나 GraphSAINT가 아니며,
chunk 크기를 줄여도 전체 그래프의 노드 상태·topology·alpha·gradient/optimizer state는 필요하다.
Sparse 연산을 사용한다는 이유만으로 ChebNet 등 다른 sparse 방법보다 빠르다고 주장하지 않는다.
같은 seed라도 CUDA scatter와 chunk 합산의 부동소수점 결과가 비트 단위로 같다는 보장은 없다.

## 산출물과 판정

Suite ID는 `conductance_direct_c_v2`다. 결과는
`results/conductance_gat/v2/<run-id>/`에 분리한다.

| 경로 | 내용 |
|---|---|
| `manifest.json` | 소스·조건·설정·명령·실행 상태 |
| `comparison.md`, `comparison.csv`, `comparison.json` | 데이터별 direct−fixed validation 비교 및 진단 |
| `logs/` | 사전 검사와 조건별 학습 로그 |
| `<dataset>/<condition>/best.pt` | 그래프에 묶인 validation-best checkpoint |
| `<dataset>/<condition>/history.json` | 학습·validation·관찰 이력 |
| `<dataset>/<condition>/metrics.json` | 설정·graph binding·지표·파일 hash·파라미터 수 |

비교할 때 원본 graph binding과 설정 일치부터 확인하고 direct−fixed accuracy 차이,
선택/종료 epoch, train loss, C 분포와 파라미터 수를 함께 읽는다. 현재 실제 v2 GPU 결과는
없으므로 개선·효율·novelty를 미리 주장하지 않는다. Model seed는 하나이고 validation만
사용하므로 CI·p-value·SOTA 또는 일반적 최적 조건을 주장하지 않는다.
