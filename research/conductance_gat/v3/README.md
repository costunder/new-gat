# Conductance v3: 상대 C 생성기와 전파 강도 분리

v2를 변경하지 않는 별도 실험이다. v2는 고정 그래프의 엣지마다 직접 파라미터를 두고,
v3는 노드 상태와 구조적 특징에서 **공유 함수로 상대적인 C를 생성**한다.
C 자체를 self-attention이라고 정의하지 않는다. C는 incidence의 엣지 좌표에서 작동하는
양의 대각 metric이며, train loss를 줄이는 방향으로 생성 함수와 전파 강도를 학습한다.
Cycle PE 및 Tree Augmentation은 결합하지 않는다.

## 실행

루트 [README](../../../README.md)의 Linux NVIDIA GPU·Conda 환경과 공식 데이터 캐시를
사용한다. 이미 데이터 준비를 끝냈다면 패키지 설치나 다운로드를 반복할 필요 없다.
저장소 루트에서 실행한다.

```bash
git pull --ff-only
bash research/conductance_gat/v3/reproduce.sh --run-id gat-relative-c-v3-seed0-v1
```

기본은 **ogbn-arxiv × `relative_c`/`fixed_c` × model seed 0 = 두 번의 새 GPU 학습**이다.
두 조건 모두 train mask의 cross-entropy로 학습하고 validation으로 checkpoint를 선택한다.
Test label은 학습·선택·진단에 사용하지 않으며 test 성능을 평가하지 않는다.
GPU가 없거나 공식 캐시가 누락·손상되면 중단한다. CPU 학습·대체 데이터·자동 설치는 없다.
기존 run을 덮어쓰거나 자동 재개하지 않으므로 재실행에는 새 run ID를 사용한다.

결과 확인:

```bash
cat results/conductance_gat/v3/gat-relative-c-v3-seed0-v1/comparison.md
```

다른 인용 그래프를 명시적으로 선택할 수 있다.

```bash
bash research/conductance_gat/v3/reproduce.sh --datasets cora citeseer pubmed --run-id gat-relative-c-v3-citations-seed0-v1
```

각 데이터마다 두 조건을 학습한다. 이번 실행 규약은 v2와 같은 transductive 데이터로 한정한다.
공유 생성기는 엣지 ID에 묶이지 않지만, **이 실행 파일에는 PPI 전이 학습을 추가하지 않았다.**
새 그래프로 전이할 수 있는 파라미터화와 실제 inductive 성능 검증을 구분한다.

## C를 어떻게 학습하는가

발생행렬 B의 행은 물리 엣지에 대응한다. C는 길이 m인 벡터로 보관하고, 수식상
`C=diag(c)`로 해석한다. `R=diag(sqrt(c))`이면

\[
B^\top C B=(RB)^\top(RB).
\]

따라서 B의 행을 scaling하는 값은 sqrt(c)이며, c는 그 제곱에 해당한다.
구현은 dense B/C/R이나 고유벡터 행렬을 만들지 않는다.

각 층의 엣지 e=(u,v)에서 다음 특징을 만든다. d는 C 적용 전 물리 그래프의 degree다.

\[
z_e=[|h_v-h_u|,(h_v-h_u)^2,h_u+h_v,h_u\odot h_v,
\log(1+d_u)+\log(1+d_v),
|\log(1+d_u)-\log(1+d_v)|].
\]

특징의 LayerNorm과 공유 MLP로 score `s_e`를 얻는다. 두 degree를 순서대로 붙이면 방향
불변이 아니므로 합·절대차를 사용했다. 입력이 없는 edge attribute를 임의로 생성하지 않는다.
양 끝점을 바꾸어도 같은 score를 만들며, 엣지 순서 변경에는 같은 순서로 출력이 바뀐다.
출력층의 공통 bias는 뒤의 중심화로 소거되므로 두지 않는다.

그래프 전체의 엣지를 기준으로 중심화하고 상대 C를 만든다.

\[
\bar s_g=|E_g|^{-1}\sum_{e\in E_g}s_e,\qquad
r_e=\tau\tanh(s_e-\bar s_g),\qquad
\tilde c_e=\frac{\exp(r_e)}{|E_g|^{-1}\sum_{f\in E_g}\exp(r_f)},
\]

\[
c_e=(1-\gamma)+\gamma\tilde c_e,\qquad
\tau=2\sigma(t),\quad\gamma=\sigma(g).
\]

각 층마다 t,g와 MLP를 독립 학습한다. 초기 tau=1, gamma=0.5이며 마지막 score 출력층을
0으로 초기화하여 C=1에서 시작한다. 전체 MLP를 모두 0으로 만드는 초기화는 아니다.
초기 forward에서는 C가 상수라 gamma/tau의 gradient가 0일 수 있다. 마지막 score 층의
gradient가 먼저 학습 신호를 받아 이후 표현이 달라진다.

비어 있지 않은 그래프에서 mean(C)=1이고 C는 양수다. 중심화·평균 정규화는 chunk별이 아니라
**그래프 전체 엣지**에 적용하며 gradient도 그 전체 의존성을 포함한다. 공통 score shift의
자유도를 제거하지만, 상수 C로 수렴하지 않는다는 보장은 없다. Gamma만으로 C의 유용성을
판정하지 않는다. Gamma·tau·score 사이에도 표현의 상호 보상이 가능하다.
또한 그래프 전체 통계 때문에 최종 C에는 비국소 의존성이 있다. MLP의 입력 특징이 엣지
양 끝점에서 나왔다는 이유로 전체 계산이 국소 이웃 계산만으로 닫힌다고 주장하지 않는다.

## 전파 강도와 대칭 정규화

\[
d_i^C=\sum_{e\ni i}c_e,\qquad
L_{\mathrm{sym},C}=D_C^{-1/2}B^\top C B D_C^{-1/2},
\qquad H'=H-\alpha L_{\mathrm{sym},C}H,
\qquad\alpha=\sigma(a).
\]

Alpha도 층마다 학습하며 초기값은 0.5다. Degree가 0인 노드의 inverse square root는 0으로
정의하므로 고립 노드는 operator에서 입력을 그대로 유지한다. Degree의 C 의존성을 detach하지
않고 양쪽 정규화 인자까지 미분한다. LayerNorm·ELU·dropout은 operator 다음에 적용된다.

비고립 노드에서는 `(1-alpha)H + alpha D^-1/2 A_C D^-1/2 H`와 같다. 이는 row 평균이 아니다.
불규칙 그래프에서 이웃 계수의 행 합은 alpha와 다를 수 있으며, 상수 H나 노드 상태의 단순 합을
보존하지 않는다. 고정된 양의 C의 선형 operator는 alpha가 [0,1]일 때 고유값 절댓값이 1 이하지만,
**C(H)를 포함하는 비선형 전체 층의 Jacobian 안정성까지 보장하지 않는다.**

## 두 조건과 optimizer

| 조건 | C | 전파 alpha | C 생성기/제어 파라미터 |
|---|---|---|---|
| `relative_c` | 공유 score에서 상대 C 생성 | 학습 | MLP·입력 norm·gamma·tau 학습 |
| `fixed_c` | 정확히 1 | 학습 | 같은 초기 scaffold를 동결하고 optimizer에서 제외 |

입력 encoder·출력 decoder와 각 층 LayerNorm, hidden 64, 두 층, dropout 0.5를 공통으로 쓴다.
동일 seed의 전체 초기 state hash를 비교하며 활성·동결·전체 파라미터 수를 각각 기록한다.
Fixed 조건도 alpha를 학습하므로 이 대조는 **v3 안에서 적응적 C 생성의 기여**를 질문한다.

| AdamW 그룹 | Learning rate | Weight decay |
|---|---:|---:|
| Backbone | 0.005 | 0.0005 |
| C 생성 MLP와 입력 norm | 0.010 | 0 |
| Alpha·gamma·tau의 raw scalar | 0.005 | 0 |

Train loss 외에 C를 일정하게 만들거나 다양성을 강제하는 보조 손실은 없다.
최대 200 epochs, validation patience 50, FP32, AMP/compile/TF32 비활성이다.
Full graph 학습이므로 `--batch-size 1`, `--workers 0`만 받으며 노드를 하나씩 학습하는 뜻은 아니다.

## 진단과 checkpoint 개입

매 epoch의 실제 train forward/gradient와 선택된 best checkpoint의 validation을 기록한다.
층별 score mean/std, C CV와 log(C) std, alpha/gamma/tau, gate MLP·입력 norm의
parameter/gradient norm, weighted degree 및 전파 변화량을 함께 본다.
V1/v2의 고정 rho=0.95를 v3에 적용하지 않는다.
고정 scaffold의 norm은 활성 gate 학습량이 아니다.

Best checkpoint를 고정한 추가 validation forward로 다음 네 가지 전체 층 개입을 검사한다.

- 그래프·층별 평균 C로 교체.
- 그래프 내 C를 정해진 seed로 섞기.
- C=1로 교체.
- 전파 operator를 identity로 교체.

각 개입에서는 바뀐 C로 weighted degree를 다시 계산한다. 양의 그래프 상수 C는 대칭
정규화에서 소거되므로 평균 C와 C=1은 수학적으로 동등하며 둘의 차이는 수치 검산 대상이다.
전파 제거는 전체 모델 제거가 아니다. Encoder, norm, 활성화, decoder는 유지한다.
후속 층은 개입으로 바뀐 노드 상태에서 C를 다시 계산한다. 따라서 개입은 고정 C 목록에 대한
독립 효과나 fresh training 이득과 같지 않다.

이 개입들은 **선택된 checkpoint에서 한 번씩** 실행하며 매 epoch 반복하지 않는다.
재학습·optimizer step·test 평가는 없다. Validation accuracy 변화(pp), 바뀐 노드 예측 비율,
평균 절대 logit 변화량을 기록한다. 단일 shuffled-C 표본은 shuffle 불확실성 추정이 아니다.

## 계산량과 확장성의 경계

전파에는 sparse gather/scatter와 정확한 chunked 1차 backward를 사용한다. C 생성은
chunk별 activation checkpointing으로 중간 특징을 재계산하며 전체 그래프의 score/C 벡터는
유지한다. 기본 `--edge-chunk-size 65536`이며 모든 엣지를 처리한다. Degree와 graph mean을
chunk 안에서 따로 정규화하는 근사 계산이나 이웃 샘플링은 아니다.

전파만 보면 O((n+m)d)지만, v3의 공유 MLP 비용은 별도로 포함해야 한다. Hidden width가
d에 비례하면 생성기의 작업량은 O(md²)다. Graph 전체 노드 상태와 O(m) score/C/인덱스,
backbone·optimizer·진단 상태도 필요하다. Chunking은 전체 메모리를 상수로 만들지 않으며
checkpoint 재계산은 추가 작업을 요구한다. 1차 미분을 검증하며 고차 미분은 지원하지 않는다.

Dense 고유분해가 없다는 것은 구현 사실이고, spectral GNN 전부보다 빠르다는 결론은 아니다.
여전히 full-graph 실행으로 GraphSAGE/GraphSAINT sampling은 구현하지 않았다.
실제 GPU 시간·peak memory·가속률은 서버 측 측정 전까지 미확인이다.

## v2와 어떻게 비교할 것인가

| 구분 | v2 | v3 |
|---|---|---|
| C 파라미터화 | 고정 그래프의 엣지별 log C | 노드 상태의 공유 상대 C 생성기 |
| 정규화 | Row node-degree | Symmetric weighted-degree |
| 전파 강도 | 고정 0.95 | 층별 학습, 초기 0.5 |
| Optimizer | Adam | 그룹별 AdamW |
| 자체 대조군 | 직접 C vs C=1 | 상대 C vs C=1, 둘 다 alpha 학습 |
| 기본 데이터/seed | ogbn-arxiv / 0 | ogbn-arxiv / 0 |

같은 데이터·seed라는 이유로 v2↔v3 전체 차이를 한 요인의 효과로 읽지 않는다.
먼저 각각의 learned−fixed를 비교하고, 점수와 C 분포·개입·epoch/시간/메모리를 함께 확인한다.
비교표는 이전 MLP나 v2 checkpoint/점수를 재사용하지 않는다. CI·p-value·SOTA·novelty나
일반적인 우월성은 이 단일 seed validation 실험으로 입증되지 않는다.
반복적으로 validation 결과를 보고 설계를 고르면 validation 자체에 과적합할 수 있다.

첨부 제안의 dmax/작은 rho 설명은 예전 global-max v1에 관한 것이다. V2에는 이미 row
node-degree 및 C weight decay 0이 적용되어 있어, v3를 그 버그 수정으로 설명하지 않는다.
Multi-head, implicit solve, 추가 여섯 조건 factorial 및 세 연구의 결합은 이번 구현에 없다.

## 산출물

Suite ID는 `conductance_relative_c_v3`이며 `results/conductance_gat/v3/<run-id>/`를 사용한다.

| 경로 | 내용 |
|---|---|
| `manifest.json` | 소스 hash·실행 명령·설정·조건·상태 |
| `comparison.md/csv/json` | 두 조건의 validation 비교, 진단과 개입 |
| `logs/` | 환경 검사 및 조건별 실행 로그 |
| `<dataset>/<condition>/best.pt` | Validation-best checkpoint와 출처 |
| `<dataset>/<condition>/history.json` | Train/validation 학습 이력과 진단 |
| `<dataset>/<condition>/metrics.json` | 결과·설정·무결성 hash·개입 |

두 조건의 초기 state, 데이터 계약, 소스와 설정을 검사한다. 누락·혼합·변조된 산출물을
성공 비교로 표시하지 않는다. 아직 실제 v2/v3 GPU 결과는 수령하지 않았으며 로컬 검증 범위는
[실험 상태](../../../docs/EXPERIMENT_STATUS.md)에 별도로 기록한다.
