# 실제 계산과 검사 대상

## 1. 공유 규칙으로 양수 C 생성

층 입력 H의 shape는 N×256이다. Q/K는 서로 다른 `Linear(256,256)`이며,
각각 N×8×32로 나눈다. bias를 포함한다. 여기의 Q는 attention 투영이고,
발생행렬 flow를 뜻하는 소문자 q와 구분한다.

\[
Q_i=H_iW_Q+b_Q,\quad K_i=H_iW_K+b_K,
\qquad
s_{uv}^{(h)}=\frac{Q_u^{(h)}\cdot K_v^{(h)}+Q_v^{(h)}\cdot K_u^{(h)}}{2r},\ r=32,
\qquad c_{uv}^{(h)}=\exp(s_{uv}^{(h)}).
\]

`conductance.py::BracketConductance`가 이 식을 직접 계산한다.
실제 물리 엣지 목록만 조회하며, N×N attention 행렬을 만들지 않는다.
양방향 평균으로 엣지 방향을 뒤집어도 C는 같다. Q/K는 독립적으로 초기화하고
CE로 학습한다. head 사이에는 점수를 평균하지 않는다.

Q/K 투영은 노드당 한 번 수행하고, 엣지 내적은 `edge_chunk_size` 단위로 계산한다.
모든 엣지를 처리한다. 엣지 내적의 backward 저장량은 선택 가능한 checkpoint로 줄인다.
계산량은 O(N D H_a r + |E| H_a r)이다. 실측 속도는 별도 자료에 기록한다.

AMP에서도 Q/K·score·exp는 FP32로 계산한다. 수식 검사 시 FP64도 지원한다.
score 또는 C가 nonfinite이거나 C가 0으로 underflow하면 오류로 중단한다.
임의의 clamp, 최대값 빼기, graphwise centering, 다른 양수 함수로의 fallback은 없다.

## 2. 기존 발생행렬 전파에 연결

B의 행마다 tail=-1, head=+1로 둔다. 실제 코드에서는 엣지 목록과 sparse
scatter 연산을 사용한다. B와 C의 dense 행렬은 단위검사 외에는 만들지 않는다.

\[
V^{(h)}=HW_v^{(h)},\quad \widetilde c_e^{(h)}=a_e c_e^{(h)},\quad
L^{(h)}=B^T\operatorname{diag}(\widetilde c^{(h)})B.
\]

기존 sampling correction a는 전파에 한 번 곱한다. 기존 샘플러의
`edge_normalization_weight`는 반복 solver가 없어진 새 C 생성기에서는 사용하지 않는다.
그 필드를 다시 C에 곱해 보정을 중복 적용하지 않는다.

\[
\alpha_{i\leftarrow j}^{(h)}=
\frac{\widetilde c_{ij}^{(h)}}{\sum_{k\in N(i)}\widetilde c_{ik}^{(h)}},\qquad
U^{(h)}=V^{(h)}-\beta^{(h)}D_{\widetilde c}^{-1}L^{(h)}V^{(h)}.
\]

이웃이 있는 노드에서는
\(U_i^{(h)}=(1-\beta^{(h)})V_i^{(h)}+\beta^{(h)}\sum_j\alpha_{i\leftarrow j}^{(h)}V_j^{(h)}\)다.
고립 노드는 U=V로 유지한다. `diffusion.py::row_diffusion`은 기존
`shared_head_diffusion`의 row 전파 순서와 `_ChunkedHeadPropagation` 커널을 사용한다.
실제 weighted degree와 계수의 유한성을 확인한 뒤 전파한다. 유한한 개별 C들을
더하다 degree가 overflow하는 경우도 오류로 중단한다. 다른 수식으로 대체하지 않는다.
degree와 alpha를 만드는 C의 미분을 끊지 않는다.

같은 엣지의 C는 양쪽에서 같지만 각 수신 노드의 다른 이웃 때문에 alpha는 달라질 수 있다.
양수 보정 a에 대해 위 식은 수신 노드의 이웃 안에서
\(\alpha_{i\leftarrow j}=\operatorname{softmax}_{j\in N(i)}(s_{ij}+\log a_{ij})\)와 같다.
즉, 학습한 연결 점수가 실제 이웃 반영 비중을 바꾼다. 같은 수신 노드의 모든
가중치가 동일 비율로 커지는 변화는 정규화에서 사라진다. C의 크기 변화와
alpha의 변화 모두를 기록하는 이유다. 엣지 점수의 대칭성은 유지한다.
출력 head 결합·출력 투영·ReLU·dropout·encoder·decoder는 기존 구조다.
beta의 `GraphConditionedBeta`와 문맥 특징 계산도 유지한다.

## 3. 초기화와 두 조건

value의 Xavier, 출력 투영의 기존 Linear 초기화, beta의 구성·초기화 알고리즘을
유지한다. 새 직접 구성은 불필요한 이전 모듈을 만들지 않으므로, 같은 seed라도
이전 v2.0의 모든 공통 파라미터 실현값까지 같다고 주장하지 않는다.
**새 learned/fixed 두 조건 안에서는 같은 초기 모델을 복사하고 공통 hash를 확인한다.**

fixed에서는 Q/K 생성기 자체를 파라미터 없는 ones 모듈로 교체한다.
두 조건 모두 value·beta·출력 투영·encoder·decoder를 학습한다.
learned에서는 이 경로를 통해 Q/K도 CE gradient를 받는다.

학습 후 C=1 개입은 다른 파라미터를 고정하고 C만 교체한다.
sampling correction은 유지하며 degree·alpha·전체 forward를 다시 계산한다.
독립적으로 학습한 fixed 모델과 이 개입 결과를 구분한다.

## 4. 관측 기록

`inspect_c.py`는 매 epoch 첫 학습 배치의 각 층 입력을 저장하고, optimizer 갱신
전후에 같은 입력으로 생성기를 비교한다. 필드는 **score**다. 이전의 cost와 달리
점수가 커지면 C가 커진다. 실제 전파의 C는 detach하지 않고 관측 복사본만 detach한다.

점수 → C → alpha 전후, live C gradient, Q/K 파라미터 gradient·갱신량을 기록한다.
alpha의 갱신 전 값은 실제 forward가 사용한 계수에서 가져온다. 재계산도 같은
edge chunk 크기를 사용한다. 갱신 없이 다시 계산한 차이를 함께 기록하고,
같은 C의 CPU FP64 기준 alpha도 비교한다. C가 그대로면 학습으로 alpha가
바뀌었다고 표시하지 않는다. 수치적으로 구별되는 변화도 유용한 학습을 뜻하지 않는다.

같은 실제 입력에서 H/Q/K의 head별 RMS, 문맥 안에서 노드 평균을 뺀 RMS,
엣지 양 끝 차이의 RMS를 기록한다. 입력 크기 축소와 노드 특징의 유사화를 구분하기
위한 관측이다. forward에 정규화나 새 loss를 추가하지 않는다.
AdamW의 이상적인 decay 감소분과 실제 갱신에서 이를 뺀 잔차도 기록한다.
이 잔차에는 과거 gradient의 optimizer 상태와 부동소수점 반올림이 포함되므로
현재 CE의 순수 기여라고 부르지 않는다.
모든 층·head·해당 배치 전체 엣지를 요약한다. 이웃 표는 미리 정한 규칙으로
선택한 수신 노드의 모든 이웃을 보여준다. 표 크기 제한은 그래프 축소가 아니다.

동일한 입력의 raw C에 대해, 독립적인 다른 그래프를 배치에 더하거나 해당 엣지의
양 끝이 아닌 노드 특징을 바꿔도 C가 같아야 한다. 이전 층에서 전파되어 들어온 H가
달라지면 C가 달라질 수 있으므로, 이 검사는 H를 고정해서 수행한다.

### 층 안에서 특징 크기가 변하는 위치

`signal_stages_before_update`는 같은 실제 forward에서 다음 여섯 값을 읽는다.

| JSON 필드 | 실제 텐서 |
| --- | --- |
| `input_h` | 층 입력 H |
| `value_projection` | V = H W_v, N×heads×head_width |
| `neighbor_mixing` | 위 발생행렬 전파의 U |
| `output_projection` | M = concat(U) W_o^T |
| `relu` | R = max(M, 0) |
| `dropout_next_h` | H_next = dropout(R), 다음 층에 실제 전달한 값 |

전체 크기는 RMS(Z) = sqrt(sum(Z²)/(N·D))다. 모든 노드·채널을 사용하며,
관측 복사본을 CPU FP64로 환산하여 합산한다. 모델 forward 정밀도는 그대로다.
`rms_by_channel_group`은 연속된 32채널 묶음별 RMS다. V/U에서는 실제 head에
해당하지만, 출력 투영이 채널들을 혼합하므로 H/M/R의 묶음을 같은 attention
head가 보존된 것으로 해석하면 안 된다. 단계 간 전체 RMS가 기본 비교 기준이다.

ReLU 전후와 dropout 전후도 분리한다. 학습 시 dropout은 살아남은 값에
1/(1-p)를 곱하므로 RMS를 늘릴 수 있다. 전체 RMS 감소를 모든 노드 표현의
동일화나 정보 손실량으로 등치하지 않는다. H/Q/K의 문맥 내 차이 관측은 유지한다.

추가 연산은 상세 관측 배치에서만 실행한다. 같은 forward의 실제 dropout 결과를
사용하며 dropout을 다시 실행하지 않는다. checkpoint backward 재계산에서는
수집을 꺼서 기록을 덮거나 중간 텐서를 계속 붙잡지 않는다. 일반 validation의
C=1 메시지 비교 observer에는 이 상세 RMS 관측을 적용하지 않는다.

## 5. 범위

양의 C가 정하는 이차형식 \(\operatorname{tr}(V^TB^T\widetilde CBV)\)는 정의되지만,
이번 경로에서는 이를 별도 특징이나 loss로 추가하지 않는다. 노드/엣지의 별도 상태,
ODE 적분, 층간 교차항, cycle 복원도 없다. 논문 전체 동역학의 보존·안정성 성질이나
GATv2 대비 우수성은 이번 생성식 채택만으로 주장하지 않는다.
