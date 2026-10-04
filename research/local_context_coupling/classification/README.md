# 로컬 문맥 결합: 실제 노드 분류

이번 실험은 **copy 공간의 교차 연산을 넣고 새로 학습하면 분류가 좋아지는가**, 그리고
**그 교차 강도를 CE로 학습하는 것이 고정 강도보다 유익한가**를 확인합니다.
첫 fixed audit의 메커니즘 검사를 실제 classification 학습으로 연결하는 별도 실험입니다.

각 macro 층은

\[
Z\xrightarrow{R}Y_0\xrightarrow{I-\eta A}Y_1
\xrightarrow{I-\gamma\rho K}Y_2\xrightarrow{I-\eta A}Y_3\xrightarrow{M}H'
\]

로 작동합니다. A는 모든 1홉 유도 로컬 그래프의 가중 라플라시안 block,
K는 인접한 중심의 로컬 그래프에서 **같은 물리 노드의 copy**를 연결하는 라플라시안입니다.
Cross를 끈 대조도 같은 내부 전파 두 번을 유지합니다.

## 여섯 조건

| 내부 C | off | fixed | learned |
| --- | --- | --- | --- |
| unit | ρ=0 | ρ=1 | ρ=sigmoid(θ) |
| local_degree | ρ=0 | ρ=1 | ρ=sigmoid(θ) |

Learned는 **독립적인 seed 모델마다 θ 하나를 두 macro 층에서 공유**합니다. 초기 θ=0, gain=.5입니다.
C와 개별 cross 엣지 가중치는 고정입니다. Learned 조건에서 배우는 것은 교차 연산의 강도입니다.
Off/fixed에는 사용하지 않는 θ 파라미터가 없습니다.

## 전체 본학습 계약

- Cora·CiteSeer·PubMed의 전체 그래프, 특징, 엣지, public mask를 사용합니다.
- 2개 macro 층, hidden dimension 64, dropout .5입니다. 첫 층 뒤에 ReLU를 적용하고 둘째 층은 logits를 출력합니다.
- 매 학습은 500 epoch입니다. Adam, LR .001/.003/.01을 비교하며 projection matrix에만 weight decay 5e−4를 적용합니다.
- Tuning seed 101/202/303에서 validation으로 LR를 선택합니다. Final seed는 11/23/37/53/71입니다.
- **Tuning 162회와 final 90회, 총 252회 학습과 126,000회 독립 모델 갱신**입니다.
- 전체 final 선택을 고정한 뒤 test를 읽습니다. 이전 연구에서 같은 public test를 확인한 후속 탐색 연구입니다.
- Source의 그래프 201개를 검증하고 보존합니다. 분류 학습에는 전체 citation 그래프 3개를 사용하며, 합성 그래프 198개는 분류 학습 대상이 아닙니다.
- Graph/model/data 규모를 줄이지 않고 seed packing·정적 cache·정확한 edge chunking·symmetric backward·activation checkpointing을 사용합니다.

기존 scalar E/J 실험과 wedge 결과는 보존합니다. 이번 모델은 scalar energy를 학습 벡터로 더하는 경로가 없습니다.
통합 copy 에너지가 전파를 정의하고, 분류 CE가 projection과 learned gain을 학습합니다.

## 결과에서 구분할 것

1. **새로 학습한 fixed−off/learned−off/learned−fixed:** 해당 교차 구조로 학습한 효과.
2. **같은 checkpoint의 gain0/gain1 개입:** 학습한 모델이 최종 예측에서 교차를 사용하는 효과.
3. **Gain, gradient, 파라미터 갱신과 실제 층 출력 차이:** 파라미터가 학습된다는 사실과 분류에 도움이 된다는 것을 구분합니다.

완료 후 실제 CSV·summary·completion으로 결과를 확인합니다. DEBUG 검사 통과를 본학습 성능으로 보고하지 않습니다.

- [서버 실행과 결과 확인](RUN.md)
- [수식과 실제 모델](MODEL_MATH.md)
- [실험 설계와 판단 기준](EXPERIMENT_DESIGN.md)
- [전체 본학습 설정](config_full.json)
- [로컬 검증 기록](VERIFICATION.md)
- [제출된 서버 FULL 결과와 해석](../../../docs/LOCAL_CONTEXT_CLASSIFICATION_SERVER_FINDINGS_20261004.md)
