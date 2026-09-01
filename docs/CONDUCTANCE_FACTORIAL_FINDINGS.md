# Conductance: 지금까지의 결과와 다음 검증

기준일: 2026-08-31. 이 기록은 사용자가 제공한 실제 GPU 실행 로그에 근거한다.
서버의 원본 checkpoint·manifest·history 전체를 이 작업 공간으로 받아 독립 재검증한 것은 아니다.
아래 측정값과 이번에 추가한 후속 코드의 실행 상태를 구분한다.

2026-09-01 갱신: 후속 C-learning의 네 조건 보고서를 수령했다. 이 문서의 2×2 수치는
그대로 보존하고, 새 결과와 새 learned checkpoint의 평균-C 검사는
[C-learning 결과 문서](CONDUCTANCE_C_LEARNING_FINDINGS.md)에서 별도로 다룬다.

## 1. 질문이 어떻게 좁혀졌는가

1. 기존 benchmark의 PPI/arxiv checkpoint에서 관측한 C가 두 층 모두 상수였고, arxiv의
   graph-local 최대 가중 차수 정규화는 대부분 노드의 이웃 전달량을 매우 작게 만들었다.
2. 기존 checkpoint의 C를 그래프·층별 평균이나 셔플 값으로 바꿔도 PPI/arxiv validation
   예측이 바뀌지 않았다. 전파 전체를 끄면 PPI 성능은 하락했다. 따라서 **그래프 연결의 기여**와
   **엣지별 C 차별화의 기여**는 같은 질문이 아니었다.
3. 이에 gate weight decay와 정규화를 한 번에 바꾸지 않고 2×2로 분리하여 새로 학습했다.
   이번 결과는 이 세 번째 단계다. 이전 5-seed test 표나 checkpoint 개입 값과 합치지 않는다.

이전 결과와 full audit의 수치·근거는 [실험 상태](EXPERIMENT_STATUS.md)에 보존되어 있다.

## 2. 실행과 근거

| 항목 | 확인된 값 |
|---|---|
| Run ID | `gat-factorial-seed0-v1` |
| 실행 소스 | `43afd632b97a4285dfeae26847b4f12a8fd1a1e4` |
| 실행 명령 | `bash research/conductance_gat/ablation/reproduce.sh --run-id gat-factorial-seed0-v1` |
| 완료 범위 | PPI / ogbn-arxiv × 4조건 × model seed 0, 8개 학습과 비교표 모두 `passed` |
| GPU / OS ABI | NVIDIA RTX A6000 / Linux glibc 2.35 |
| Python / Torch / PyG | 3.11.16 / 2.7.1+cu118 / 2.7.0 |
| 사용자 첨부 식별자 | `20b4a93d-06ed-4cff-9fe5-530eacf39766` |
| 첨부 텍스트 SHA-256 | `2C78D02BB210BF00865AB7207DF651B02B2081EE4FAE6E8A6A83665A5D331161` |

첨부 hash는 제공된 **텍스트 파일**의 hash다. 데이터나 checkpoint hash로 제시하지 않는다.
개인 계정·호스트 경로와 전체 원본 로그는 공개 문서에 복제하지 않았다.
GPU 사전 검사만 성공한 상태가 아니라, 여덟 조건의 epoch 기록과 최종 비교표까지 확인했다.

공통 설정은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005, non-gate WD 0.0005,
최대 200 epochs, patience 50, PPI batch size 2, workers 0이다. FP32로 실행하며
AMP·TF32·compile은 끈다. arxiv는 full-batch다. 공식 train 정답으로 loss를 계산하고
validation으로 best checkpoint를 선택했다. **Test split은 평가하지 않았다.**
비교 실행기는 초기 state, 데이터/cache protocol, 공통 설정의 일치를 검사한다. 같은 초기 상태가
CUDA scatter 연산의 비트 단위 동일성이나 모든 조건의 동일 종료 epoch까지 보장하지는 않는다.

| 조건 | 정규화 | Gate WD | 나머지 WD |
|---|---|---:|---:|
| `baseline` | graph-local global maximum weighted degree | 0.0005 | 0.0005 |
| `gate_no_wd` | graph-local global maximum weighted degree | 0 | 0.0005 |
| `node_degree` | node-local weighted degree | 0.0005 | 0.0005 |
| `node_degree_gate_no_wd` | node-local weighted degree | 0 | 0.0005 |

## 3. Validation 결과

퍼센트 점수는 사용자 로그의 여섯 자리 출력을 그대로 옮겼다. Δ 단위는 상대 개선율이 아니라
**percentage points(pp)**다. PPI는 global micro-F1, arxiv는 accuracy이며 둘을 평균내지 않는다.

| 조건 | PPI micro-F1 (%) | Best / 실행 epoch | arxiv accuracy (%) | Best / 실행 epoch |
|---|---:|---:|---:|---:|
| `baseline` | 48.986770 | 113 / 163 | 50.927883 | 199 / 200 |
| `gate_no_wd` | 49.378028 | 134 / 184 | 50.565451 | 172 / 200 |
| `node_degree` | 52.465469 | 64 / 114 | 68.317723 | 195 / 200 |
| `node_degree_gate_no_wd` | 50.340520 | 41 / 91 | 67.995566 | 196 / 200 |

| 조건부 비교 | PPI Δ(pp) | arxiv Δ(pp) |
|---|---:|---:|
| Global-max에서 gate WD 제거 | +0.391258 | −0.362432 |
| Gate WD 유지하고 node-degree로 변경 | +3.478699 | +17.389840 |
| Node-degree에서 gate WD 제거 | −2.124949 | −0.322157 |
| Gate WD 없이 node-degree로 변경 | +0.962492 | +17.430115 |
| 상호작용: both − gate-only − normalization-only + baseline | −2.516208 | +0.040275 |

각 Δ는 반올림 전 원 지표에서 계산된 보고서 값을 옮겼다. 표의 반올림된 퍼센트 값끼리 계산하면
마지막 자리가 다를 수 있다. 상호작용은 네 점수의 대수적 대비이지 통계적 유의성 검정이 아니다.

## 4. 선택된 checkpoint의 C와 전파

다음은 validation 그래프별 통계를 낸 뒤 그래프에 같은 가중치를 준 평균이다.
PPI는 validation 그래프 2개, arxiv는 전체 transductive 그래프 1개다. C CV는 각 그래프 내부의
`std(C)/mean(C)`로, 그래프 간 평균 C 차이를 섞은 pooled CV가 아니다. 층 번호는 0부터다.
전파 변화량은 LayerNorm/ELU 전 `||H_after_conv−H_before_conv||/||H_before_conv||`다.

| 데이터 | 조건 | 층 | C CV | rho 평균 | 전파 상대 변화량 | Gate parameter L2 |
|---|---|---:|---:|---:|---:|---:|
| PPI | baseline | 0 | 0 | 0.0522017 | 0.0990779 | 9.54558e-5 |
| PPI | baseline | 1 | 0 | 0.0522018 | 0.0929463 | 1.26455e-6 |
| PPI | gate_no_wd | 0 | 0.637470 | 0.0606669 | 0.144769 | 26.9633 |
| PPI | gate_no_wd | 1 | 0.584932 | 0.105537 | 0.124904 | 20.3282 |
| PPI | node_degree | 0 | 0.534122 | 0.946912 | 1.02783 | 2.10246 |
| PPI | node_degree | 1 | 0.123135 | 0.946912 | 0.783196 | 1.41693 |
| PPI | node_degree_gate_no_wd | 0 | 0.840766 | 0.946912 | 1.08141 | 12.2539 |
| PPI | node_degree_gate_no_wd | 1 | 0.399742 | 0.946912 | 0.786316 | 14.4391 |
| arxiv | baseline | 0 | 0 | 0.000986925 | 0.00304104 | 0.000247152 |
| arxiv | baseline | 1 | 0 | 0.000986923 | 0.00210088 | 0.000244609 |
| arxiv | gate_no_wd | 0 | 1.58085 | 0.00231003 | 0.00757714 | 11.5340 |
| arxiv | gate_no_wd | 1 | 6.03303 | 0.00244190 | 0.0107650 | 10.3196 |
| arxiv | node_degree | 0 | 0 | 0.950000 | 0.729080 | 0.000225269 |
| arxiv | node_degree | 1 | 0.00948423 | 0.950000 | 0.443829 | 1.06375 |
| arxiv | node_degree_gate_no_wd | 0 | 0.0742367 | 0.950000 | 0.723815 | 12.3619 |
| arxiv | node_degree_gate_no_wd | 1 | 0.137906 | 0.950000 | 0.446793 | 10.1775 |

Node-degree 정규화에서 비고립 노드의 rho는 정의상 0.95이고 고립 노드는 0이다.
PPI의 평균 0.946912와 arxiv의 0.95는 gate 학습 성공의 독립적인 증거가 아니다.
단, 상태 자체가 바뀐 정도와 validation 점수가 같이 증가했다는 관측은 구분해 기록할 수 있다.

## 5. 현재 허용되는 해석

- **이 seed와 설정에서는 node-degree 정규화가 주요 개선 요인이다.** Gate WD를 유지한
  `node_degree`가 두 데이터 모두 최고이고, 특히 arxiv는 baseline보다 +17.389840pp다.
- Gate WD를 제거하면 선택된 모델의 비상수 C와 gate norm이 커졌다. 그러나 arxiv 점수는
  개선되지 않았고, PPI도 node-degree 아래에서는 −2.124949pp다. 따라서 “C 붕괴 제거”와
  “성능 개선”은 동치가 아니며 WD를 무조건 없애야 한다는 결론도 성립하지 않는다.
- **학습한 C의 순수 기여는 아직 입증되지 않았다.** arxiv의 최고 조건은 첫 층 C CV가 0이고
  둘째 층도 약 0.00948이다. 정규화만으로 상당한 개선이 가능한지 분리해야 한다.
  반대로 PPI의 최고 조건에는 비상수 C가 남으므로 모든 데이터에서 C가 쓸모없다고 할 수도 없다.
- 다음 비교의 기준은 `node_degree + gate WD 0.0005`다. 이는 후속 내부 실험의 기준 선택이며,
  기존 `benchmark.py`나 기본 재현 명령의 모델을 조용히 교체한 것이 아니다.

여전히 model seed는 하나다. seed 평균·표준편차·CI·p-value, SOTA, 일반적인 최적 조건,
novelty 또는 아직 평가하지 않은 test 성능을 주장하지 않는다. 반복적인 validation 기반
선택 자체가 validation에 과적합할 수 있다. arxiv는 여러 조건의 best가 200-epoch 예산 끝에
가까우므로 학습 예산 밖에서 순위가 같다는 보장도 없다.

## 6. 2026-08-31 당시 설계한 후속 분석: 두 질문을 분리

### A. 현재 checkpoint가 엣지별 C 차이에 의존하는가?

기존 `gat-factorial-seed0-v1`의 `node_degree` checkpoint를 읽기만 한다.
원래 validation을 먼저 재현한 뒤 C를 **그래프·층별 평균**으로 바꾸고, 바뀐 C로 weighted
degree를 다시 계산한다. 전체 층 개입과 한 층씩 개입을 분리한다. 재학습·optimizer update·
test 평가는 없고, 보고서만 별도 경로에 저장한다.

원 checkpoint·cache·소스와 validation 재현이 맞지 않으면 유효한 개입 대비를 만들지 않는다.
원래 global-max용 `diagnose_conductance.sh`에 이 checkpoint를 넣지 않는다.
평균 C 개입 후 점수가 거의 같으면 현재 선택된 checkpoint가 엣지별 C 변동을 쓰는 정도가
작다는 근거다. 학습 과정 전체에서 gate가 아무 역할도 하지 않았다는 증명은 아니다.

### B. C를 학습하는 것이 동일 설정의 고정 C보다 나은가?

별도 `c_learning` 실험에서 PPI/arxiv 각각 **`learned_c`와 `fixed_c`를 처음부터 학습**한다.
두 조건 모두 node-degree 정규화와 같은 non-gate WD 0.0005, 초기 backbone, seed 0,
데이터·학습·validation 선택 정책을 사용한다. `learned_c`의 gate WD도 0.0005다.
`fixed_c`는 모든 물리 엣지의 C를 정확히 1로 고정하며 gate 파라미터를 학습하지 않는다.
Fixed 조건은 동일한 초기 state/RNG를 맞추려고 gate scaffold를 보존하지만,
이를 평가하거나 optimizer에 넣지 않는다. 전체·활성 학습·동결 파라미터 수는 따로 기록한다.
따라서 fixed 조건에서 출력하는 동결 gate norm을 실제 C를 생성하는 gate의 norm으로 읽지 않는다.

고정 C에서 비고립 노드의 전파는 다음과 같다.

\[
H'_i=0.05H_i+0.95\frac{1}{d_i}\sum_{j\in\mathcal N(i)}H_j.
\]

고립 노드는 그대로 둔다. 이는 같은 모델 내부에서 **적응적 C 학습을 제거한 대조군**이지,
외부 논문의 GCN/GAT/GraphSAGE를 재현한 모델이 아니다. Gate를 제거하면 실제 학습 가능한
파라미터 수는 줄어드므로 동일 parameter-budget 경쟁 모델 비교로도 부르지 않는다.

총 학습은 **2데이터 × 2조건 × 1seed = 4개**다. 이미 받은 2×2의 learned 점수를 가져와
새 fixed 점수와 대신 짝짓지 않고 두 조건을 같은 새 실행에서 비교한다. A의 개입 효과와
B의 재학습 차이는 별도 보고서에 남긴다. 이후 B의 네 조건은 완료 보고서를 수령했으며,
그 새 run의 learned checkpoint를 검사하는 평균-C 개입도 이후 `passed` GPU 출력을 수령했다.
PPI는 전체 층 평균화에서 −6.649440pp, arxiv는 −0.033558pp였지만 이는 선택 checkpoint의
민감도이며 fresh learned/fixed 학습 이득과 같지 않다. 정확한 층별 값과 근거 한계는
[C-learning 결과 문서](CONDUCTANCE_C_LEARNING_FINDINGS.md)를 따른다.

설치·실행·결과 확인은 [C 학습 기여 실험 안내](../research/conductance_gat/c_learning/README.md)를 따른다.

## 7. 당시 정한 결과 판단 순서

1. Manifest 상태와 source/cache/checkpoint 무결성, 원 validation 재현부터 확인한다.
2. A의 all-layer 및 layer별 metric/logit/flip 차이로 현재 C 변동 의존도를 확인한다.
3. B의 데이터셋별 `learned_c−fixed_c` validation 차이, best/실행 epoch, train loss,
   C 분포와 전파 변화량을 함께 본다. PPI/arxiv를 합산하지 않는다.
4. A에서 민감하지만 B의 차이가 작으면 “현재 checkpoint의 사용”과 “재학습 시 필요성”이
   다를 수 있다. A에서 둔감하지만 B가 다르면 학습 경로의 영향 가능성이 남는다.
5. 같은 결과를 본 뒤 새로운 WD/dropout/epochs를 동시에 바꾸지 않는다. 추가 가설은 별도 run과
   대조 조건으로 분리한다. C-learning 결과와 후속 개입에도 Cycle PE/Tree를 섞지 않는다.
