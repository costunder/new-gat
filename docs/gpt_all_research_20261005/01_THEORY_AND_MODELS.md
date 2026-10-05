# 이론과 모델: 발생행렬에서 로컬 문맥 결합까지

작성 기준: **2026-10-05**. 이 문서는 연구 아이디어가 어떤 모델로 구현됐는지 설명한다.
실험 수치와 완료 여부는 이 묶음의 결과 문서와 원래 서버 증거를 함께 확인한다.
과거 문서에는 당시의 최신 상태가 남아 있으므로 날짜와 namespace를 함께 읽어야 한다.

**현재 연구는 각 1홉 로컬 그래프의 내부 전파와, 겹치는 로컬의 같은 물리 노드 copy 사이 전파를 결합하는 모델이다.**
최신 구현은 research/local_context_coupling/normalization/이다.
이전 여섯 조건에 이어 최신 정규화 20조건의 서버 FULL 완료 보고와 결과 요약 원문을 수령했다.
최신 실행은 840회 학습·420,000회 독립 모델 갱신을 보고한다.
큰 accuracy 변화는 내부 정규화에서 나타났고, 교차 연산의 추가 accuracy 개선은 확인되지 않았다.
일부 조건에서 CE 개선이 관측됐다. 상세 수치와 증거 범위는 [정규화 FULL 결과](08_NORMALIZATION_FULL_RESULTS.md)를 따른다.
이전 Conductance, 정보 흐름 v2, wedge, scalar E/J 결과를 최신 모델의 성능으로 옮겨 적지 않는다.

## 1. 연구가 이동한 순서와 버전 이름

처음 질문은 **발생행렬 사이에 넣을 양의 엣지 가중치 C를 공유 규칙으로 학습하고, 다른 부분구조에서도 재사용할 수 있는가**였다.
그 뒤에는 집계에서 무엇이 보이지 않는지, 에너지와 교차 관계를 별도로 남기면 도움이 되는지를 검토했다.
Wedge는 길이 2 경로의 이차 차분을 쓰는 별도 모델이다.
최근 로컬 연구는 내부 에너지 E와 관계 J를 먼저 감사하고, scalar 특징을 더하는 후보를 거쳐,
copy 공간의 결합 에너지가 실제 전파를 정의하는 모델로 이동했다.

| 이름·경로 | C 또는 교차 가중치 | 실제 학습 대상 | 구분 |
| --- | --- | --- | --- |
| Conductance V1, research/conductance_gat/ | 공유 생성기가 엣지 차이에서 C 산출 | 생성기와 예측 모델 | 초기 독립 conductance 연구 |
| Conductance V2, research/conductance_gat/v2/ | 물리 엣지·층마다 exp(alpha_e) | 해당 그래프의 엣지 파라미터 | 새 그래프용 공유 규칙이 아님 |
| Conductance V3, research/conductance_gat/v3/ | 공유 score에서 상대 C 산출 | 생성기·혼합·전파 강도 | V2를 보존한 별도 모델 |
| Conductance V4, research/conductance_gat/v4/ | V3의 C와 spatial W | C 생성기·W·전파 강도 | C × W의 2×2 비교 |
| Conductance V5, research/conductance_gat/v5/ | 학습된 cost의 유한 반복 solver로 C 산출 | cost 규칙·W·beta·backbone | 초기 MLP/staged와 corrected optimization/joint 구분 |
| 정보 흐름 v2, experiments/information_flow_v2/ | 기존 incidence 생성기 또는 고정 C | 예측 모델·요약값 readout | Conductance V2와 다른 버전 |
| C 학습 진단, experiments/c_learning_only/ 및 c_learning_bracket/ | solver 또는 대칭 Q/K score | 공유 생성기·예측 모델 | 정보 보완 전체를 구현한 모델은 아님 |
| Wedge, research/wedge_propagation/ | 길이 2 경로의 C2 | 단계별 고정 또는 공유 gate·예측 모델 | 물리 엣지 C와 경로 C2 구분 |
| Scalar E/J, research/local_energy_relations/ | unit/local_degree 고정 C | 분류 projection·scalar readout | 결합 에너지의 전파 모델과 구분 |
| Copy 결합, research/local_context_coupling/ 및 classification/ | 내부 C와 개별 copy 연결은 고정 | 분류 projection·일부 조건의 교차 gain | 고정 감사와 분류 학습을 별도 실행 |
| 최신 copy 정규화, research/local_context_coupling/normalization/ | 같은 연결에 대칭 S/G 적용 | 분류 projection·일부 조건의 교차 gain | 20조건 FULL 완료 보고 수령: 840회 학습·420,000회 모델 갱신 |
| Cycle PE·Tree augmentation | cycle 기저·정적 구조 표현 | 별도 그래프 예측 모델 | 최신 copy 모델에 결합하지 않음 |

v2라는 이름이 같아도 경로가 다르면 모델·설정·checkpoint가 다르다.
원래 구분은 [Conductance 인계](../../gpt_handoff/HANDOFF.md),
[독립 연구 개요](../RESEARCH_OVERVIEW.md), [최신 copy 수식](../../research/local_context_coupling/normalization/MODEL_MATH.md)을 따른다.

## 2. 발생행렬, 가중 라플라시안, 이차형식

단순 무방향 물리 그래프에 노드 N개, 엣지 E개, 특징 F개가 있다고 하자.
물리 엣지는 한 번씩 세고 계산용 방향만 임의로 정한다.

| 기호 | 크기 | 의미 |
| --- | --- | --- |
| X | N × F | 노드 특징 또는 현재 hidden state |
| B | E × N | signed incidence; 엣지 행마다 한 끝점 −1, 다른 끝점 +1 |
| c, C=diag(c) | E, E × E | 양의 엣지 가중치; 실제 코드는 c 벡터로 보관 |
| BX | E × F | 연결된 노드의 특징 차이 |
| q=CBX | E × F | 가중 엣지 메시지 또는 flow |
| d=Bᵀq | N × F | 노드별 signed 집계 |
| L_C=BᵀCB | N × N | 대칭 PSD 가중 라플라시안 |

\[
L_C=B^\top CB=(C^{1/2}B)^\top(C^{1/2}B),\qquad d=L_CX.
\]

**L_C는 이차형식을 나타내는 행렬이고, 현재 특징을 대입한 값은 그래프의 이차 에너지다.**

\[
E_C(X)=\operatorname{tr}(X^\top L_CX)
=\sum_{e=(u,v)}c_e\|X_v-X_u\|^2.
\]

스칼라 특징이면 xᵀL_Cx다. 여러 특징 채널이면 채널별 이차형식을 합한 trace를 쓴다.
가중 라플라시안만으로 이 Dirichlet 에너지를 계산할 수 있다. 추가 energy 모듈은 필수 조건이 아니다.
다만 **에너지를 계산하는 것**, **scalar 특징으로 예측에 넣는 것**, **에너지 연산자를 전파에 쓰는 것**은 서로 다른 구현이다.

### 2.1 에너지와 실제 전파의 연결

C가 고정이면 ½ E_C(X)의 X에 대한 gradient는 L_CX다.
따라서 X−eta L_CX는 이 고정 에너지의 gradient step으로 해석할 수 있다.
일부 문서는 에너지를 ½ 없이, copy 문서는 ½를 넣어 정의한다. 이 차이는 gradient의 2배 계수와 step에 영향을 준다.

정규화와 identity가 들어간 모델은 raw L_CX만 출력하지 않는다. 예를 들어

\[
P_CX=X-\beta D_C^{-1}L_CX
=(1-\beta)X+\beta D_C^{-1}A_CX
\]

는 비고립 노드에서 자기 상태와 가중 이웃 평균을 섞는다.
D_C는 weighted degree, A_C는 weighted adjacency다.
고립 노드는 inverse degree를 0으로 정의해 자기 상태를 유지한다.
L_C는 연결 성분의 상수 입력을 0으로 보내지만 이 전파는 상수 입력을 유지한다.

### 2.2 입력에 따라 C가 달라지는 경우

공유 생성기가 C=C_theta(X)를 만들면 한 forward에서 C를 고정한 연산자는 여전히 PSD다.
하지만 전체 함수 X ↦ L_C(X)X는 일반적으로 비선형이다.
입력 의존 에너지를 미분하면

\[
\nabla_X\left[\tfrac12\sum_e c_e(X)\|(BX)_e\|^2\right]
=L_{C(X)}X+\tfrac12\sum_e\|(BX)_e\|^2\nabla_Xc_e(X).
\]

따라서 L_C(X)X를 메시지로 쓰는 것과 위 전체 energy gradient를 쓰는 것은 다르다.
분류 loss의 backward가 C 생성기까지 전달된다는 사실은 이 구분과 모순되지 않는다.
Task backward는 실제 forward에 존재하는 C의 입력·파라미터 의존성을 미분한다.

근거: [V5 수학 검토](../../gpt_handoff/CONDUCTANCE_V5.md#c-learning-audit-20260908),
[wedge 학습 수식](../../research/wedge_propagation/learned/MODEL_MATH.md).

## 3. C의 역할, 합동변환, spectral과 spatial

### 3.1 C는 엣지 좌표의 가중 metric이다

BᵀCB는 엣지 metric을 노드 좌표로 가져온 Gram 행렬이다.
C^(1/2)B는 발생행렬의 각 행을 sqrt(c_e)배 한다.
B는 보통 직사각형이므로 가역적인 노드 좌표 변환으로 보지 않는다.

이차형식의 노드 기저를 가역 P로 바꾸는 합동변환은 L ↦ PᵀLP다.
C와 이 P는 좌표·크기·역할이 다르다.
같은 rank의 PSD 행렬 사이에 어떤 합동변환이 존재한다는 일반 사실만으로,
현재 C 생성기를 **기존 L의 빠른 합동변환을 구하는 알고리즘**이라고 결론낼 수 없다.
현재 구현은 고정 지지집합의 엣지 weight와 그 결과의 라플라시안을 만든다.

### 3.2 Spectral 해석을 sparse message passing으로 계산할 수 있다

고정 C의 정규화 라플라시안을 L_sym,C=U Lambda Uᵀ라고 쓰면

\[
(I-\beta L_{sym,C})V=U(I-\beta\Lambda)U^\top V.
\]

선형 전파 부분은 고유값에 1−beta lambda를 적용하는 1차 spectral polynomial이다.
계산할 때 U나 고유값을 실제로 구할 필요는 없다.
코드는 엣지 차이를 gather하고 양 끝점으로 scatter한다.
따라서 **spectrum으로 해석하는 연산자를 sparse message passing으로 계산**할 수 있다.
다항식 해석의 기존 원 논문 링크는 [Defferrard et al., 2016](https://arxiv.org/abs/1606.09375)이다.

입력에서 C를 다시 생성하면 spectrum과 eigenbasis도 입력에 따라 바뀐다.
C·비선형·dropout·projection을 포함한 전체 모델을 하나의 고정 spectral filter라고 부르지는 않는다.

Spatial W는 노드 채널을 섞는 학습 행렬이다.
W가 모든 노드에서 같고 C가 고정이면 L_C(XW)=(L_CX)W다.
그러나 **C를 W 적용 전 상태에서 생성할지 후 상태에서 생성할지**는 입력 의존 모델에서 다른 설계다.
V4는 W 이전 상태에서 C를 만들고 P_C(HW)를 전파한다.
두 단계는 계산 역할의 구분이며 1단계를 먼저 완전히 학습한 뒤 고정한다는 뜻은 아니다.

## 4. 사이클 소거와 실제 메시지 복원

### 4.1 임의의 엣지 메시지는 cycle 성분을 구분할 수 없다

임의의 q에 대해 Bᵀz=0인 cycle flow z를 더해도

\[
B^\top(q+z)=B^\top q
\]

다. 전체 노드 집계만 관측하면 자유로운 cycle 성분을 구분할 수 없다.
연결 성분 수가 k인 그래프의 cycle 공간 차원은 E−N+k다.
이는 **관측 연산자 Bᵀ의 영공간**에 대한 명제다.

### 4.2 메시지가 q=CBX로 제한되면 다른 문제다

알려진 양의 C를 고정하고 모든 노드의 d=BᵀCBX를 관측하면

\[
\boxed{q=CB(B^\top CB)^\dagger d}.
\]

X는 연결 성분별 상수까지 복원되지만 B가 그 상수를 없애므로 q는 정확히 복원된다.
L_C^dagger L_CX가 X에서 연결 성분별 상수를 제거한 값이라는 점으로 확인할 수 있다.

따라서 **cycle이 있다 → 현재 q가 반드시 복구 불가능하다**는 결론은 틀리다.
비균일 C에서 q의 Euclidean cycle projection이 0이 아닐 수 있어도,
알려진 C에 따른 제한된 q는 위 식으로 복원된다.
Cycle projection의 norm과 실제 허용 메시지의 복원오차는 서로 다른 지표다.

다음에는 이 보장을 그대로 사용할 수 없다.

- C를 모르거나 C와 X를 함께 식별해야 하는 경우.
- 일부 수신 노드의 집계만 관측하는 경우.
- 샘플 밖 엣지와 노드의 메시지가 빠진 경우.
- 메시지가 독립적인 자유 엣지 상태인 경우.
- 이후 비선형·projection·압축으로 관측값을 바꾼 경우.

Partial 관측은 선택 행렬 P를 넣어 A_obs=P Bᵀ로 정의한다.
정보 흐름 v2는 W A_obsᵀ(A_obs W A_obsᵀ)^dagger A_obs q 형태의 probe를 쓴다.
W=I의 전체 노드 probe는 Euclidean cycle 성분을 제외한 흐름을,
W=C의 전체 노드 probe는 제한된 q의 복원을 검사한다.
Partial 수신 집합의 probe는 그 관측 범위 안에서의 복원 가능성을 검사한다.
Degree 보정이나 원래 노드 ID 보관 자체가 누락 메시지를 복원하지는 않는다.

근거: [정보 흐름 v2 audit](../../experiments/information_flow_v2/audit.py),
[로컬 에너지 감사 수식](../../research/local_energy_relations/MODEL_MATH.md),
[수신 집계 수식](../../research/local_energy_relations/receiver_aggregation/MODEL_MATH.md).

## 5. Conductance V1–V5의 실제 학습 문제

### 5.1 V1: 공유 엣지 규칙

V1은 현재 특징 차이·엣지 입력에서 공유 규칙으로 양의 C를 만들고 BᵀCBH를 전파에 사용한다.
공유 theta를 배우므로 엣지 수가 다른 그래프에도 같은 생성 규칙을 적용할 수 있다.
새로운 구조에서 실제 예측이 개선되는지는 별도 평가가 필요하다.
초기 global degree bound, node-degree 정규화, gate weight decay 등을 바꾼 대조는 각 문서의 독립 조건으로 읽는다.

원래 모델은 encoder, conductance 층, LayerNorm/ELU/dropout, decoder를 포함한다.
나중에 외부 wrapper를 제거한 8층 비교 모델과 같은 backbone이 아니다.
근거: [V1 설명](../CONDUCTANCE_GAT.md), [정규화 × decay 대조](../CONDUCTANCE_FACTORIAL.md),
[C 학습 대조](../CONDUCTANCE_C_LEARNING.md).

### 5.2 V2: 해당 그래프의 엣지별 파라미터

\[
c_e^{(\ell)}=\exp(\alpha_e^{(\ell)}),\qquad
H'=H-.95D_C^\dagger B^\top CBH.
\]

Alpha는 엣지·층에 직접 묶이고 초기 alpha=0이므로 C=I다.
입력 H가 바뀌었다고 같은 forward의 C가 바뀌지는 않는다.
특정 topology의 checkpoint를 다른 그래프에 적용할 공유 규칙이 없다.
미관측 PPI 그래프용 C를 임의로 만들지 않는다.
공통 alpha shift는 row 정규화에서 소거된다.
이 식별되지 않는 공통 배율과 상대 엣지 가중치의 학습을 구분한다.
근거: [V2 수식과 graph binding](../../gpt_handoff/CONDUCTANCE_V2.md).

### 5.3 V3: 상대 C와 전파 강도 분리

끝점 순서에 불변인 특징의 공유 MLP score s_e를 그래프 전체에서 중심화한다.

\[
r_e=\tau\tanh(s_e-\bar s_g),\quad
\widetilde c_e=\frac{e^{r_e}}{\operatorname{mean}_{f\in E_g}e^{r_f}},\quad
c_e=(1-\gamma)+\gamma\widetilde c_e.
\]

전파는 H−alpha D_C^(-1/2)L_CD_C^(-1/2)H다.
Alpha, gamma, tau와 생성기를 학습하고, 최종 score 층을 0으로 초기화해 C=1에서 시작한다.
평균 C=1은 공통 배율을 고정하지만 상대 C가 반드시 다양해지거나 유용해진다는 보장은 아니다.
끝점 score 입력이 국소적이어도 전체 엣지 평균을 쓰므로 최종 C에는 그래프 전체 통계의 의존성이 있다.
근거: [V3 수식](../../gpt_handoff/CONDUCTANCE_V3.md).

### 5.4 V4: C 생성과 spatial W의 2×2 비교

\[
H'=(1-\alpha)H+\alpha D_C^{-1/2}A_CD_C^{-1/2}(HW)
\]

를 비고립 노드에 적용하고 고립 노드는 H를 유지한다.
W는 square·biasless·identity 초기화다.
고정/상대 C × identity/학습 W를 새로 학습해 각각의 효과와 상호작용을 나눈다.
자기 경로는 H, 이웃 경로는 HW이므로 단순히 (I−alpha L_sym,C)HW와 같다고 쓰지 않는다.
근거: [V4 수식과 대조 계약](../../gpt_handoff/CONDUCTANCE_V4.md).

### 5.5 Corrected V5: 학습된 cost에서 C를 반복 계산

V5에는 역사적 MLP backend와 staged 학습도 남아 있다.
Corrected 결과의 설정은 **optimization / joint / width_scaled / beta_initial=.5**다.
현재 특징·문맥·구조에서 학습된 cost delta_theta,e를 만들고,
각 forward에서 C=1부터 8번 내부 반복한다. 공유 theta까지 매번 초기화하지 않는다.

\[
\mathcal F_g(c;\delta_\theta)=
\frac1{\Omega_g}\sum_e\omega_e
[c_e\delta_{\theta,e}+\tau(c_e\log c_e-c_e+1)]
-\frac{\lambda_d}{n_+}\sum_{i:d_i^{ref}>0}\log\frac{d_i(c)}{d_i^{ref}},
\]
\[
c_e>0,\qquad \Omega_g=\sum_e\omega_e,\qquad
\frac{\sum_e\omega_ec_e}{\Omega_g}=1.
\]

Omega는 sampling correction omega의 합, d_i(c)는 omega c의 weighted degree다.
Entropy와 degree barrier는 내부 C 문제의 목적함수다.
최종 모델은 task loss로 공유 cost 규칙·W·beta·backbone을 학습한다.
8-step unroll의 gradient이며 완전히 수렴한 argmin의 implicit gradient는 아니다.
유한값·양수·내부 에너지 검사와 실제 K8이 task에 충분한지에 대한 검증을 나눈다.

각 head는 고정 C 해석에서 (I−beta_h L_sym,C)HW_h를 쓴다.
C를 특징 head 간 공유하는 설정과 head별 확장은 실제 run metadata를 따른다.
C에 의존하는 정규화 degree도 task backward에 포함한다.
공통 C 배율은 degree 정규화에서 상쇄되므로 평균 C만으로 attention 변화를 측정할 수 없다.

근거: [V5 최신 수학 검토와 역사](../../gpt_handoff/CONDUCTANCE_V5.md),
[solver](../../research/conductance_gat/v5/optimization.py),
[operator](../../research/conductance_gat/v5/operator.py).

### 5.6 후속 Gram/lift 비교

Experiments/aggregation_comparison/에는 같은 현재 value projection을 과거 depth state에 적용하고

\[
\Gamma_i^{(k,m)}=\tfrac12\sum_{j\sim i}c_{ij}
\langle V_i^{(k)}-V_j^{(k)},V_i^{(m)}-V_j^{(m)}\rangle
\]

를 readout하는 조건이 있다.
Diagonal은 자기 에너지, off-diagonal은 같은 edge 좌표의 교차 관계다.
과거 depth는 정확한 거리별 shell이 아니다.
Pre-lift [V,V²]는 별도 조건이다. 차원이나 rank가 늘 수 있다는 사실만으로 전체 네트워크의 역변환을 보장하지 않는다.
Zero 초기 energy readout은 첫 step에서 readout 자신이 gradient를 받는다.
처음부터 그 추가 경로로 C gradient가 늘어나는 것은 아니다. 기본 diffusion의 C gradient는 별도로 존재한다.

근거: [상세 구현 검토](../DEEP_IMPLEMENTATION_REVIEW_20260927.md),
[현재 arxiv 비교 계약](../ARXIV_BASELINE_COMPARISON.md).

## 6. 정보 흐름 v2와 대칭 Q/K 후보

### 6.1 정보 흐름 v2는 세 가지 요약을 예측에 더했다

하나의 sampled context 안에서 깊이에 따라 송신·수신 노드 집합을 예측 대상 쪽으로 좁힌다.
매층 독립적인 부분그래프를 무작위로 다시 뽑는 설계가 아니다.
공유 C 규칙은 서로 다른 sampled context에 적용한다.

실제 branch는 다음을 scalar로 집계하고 학습 readout 벡터를 곱해 기본 전파에 더한다.

| branch | 계산하는 정보 | 보존하지 않는 것 |
| --- | --- | --- |
| Within | mean_channels(q²)를 양 끝점에 집계 | 개별 q의 부호·전체 채널 내용 |
| Boundary | 다음 수신 집합에서 빠지는 노드와의 C_eff(BV)² | sampled context 밖에서 처음부터 못 본 메시지 |
| Layer cross | 두 단계에 남는 같은 edge의 mean(past_q * q) | 과거에 사라진 모든 edge 상태 |

Within의 q²에는 C², Dirichlet energy의 C(BV)²에는 C가 곱해진다. 같은 양이 아니다.
Branch는 실제 예측에 연결되지만 **요약 통계를 추가하는 방식이며 원래 flow를 복원해 유지하는 구현은 아니다**.
분류 CE로 학습하며 복원오차를 학습 loss로 쓰지 않는다.
전체 문맥과의 비교 audit는 진단용이다. 누락 특징을 sampled forward에 제공하지 않는다.

근거: [v2 model](../../experiments/information_flow_v2/model.py),
[transfer plan](../../experiments/information_flow_v2/blocks.py),
[audit](../../experiments/information_flow_v2/audit.py),
[실행 설명](../../experiments/information_flow_v2/README.md).

### 6.2 C 학습 진단과 bracket 논문에서 채택한 생성식

C_learning_only는 cost/solver의 C와 실제 row attention alpha를 같은 H에서 비교하는 후보다.
C_learning_bracket은 독립 Q/K projection의 대칭 score를 채택한다.
아래 r은 head의 Q/K 차원이며 이 후보의 기본값은 32다.

\[
Q_i=H_iW_Q+b_Q,\qquad K_i=H_iW_K+b_K,
\]
\[
s_{uv}^{(h)}=
\frac{Q_u^{(h)}\cdot K_v^{(h)}+Q_v^{(h)}\cdot K_u^{(h)}}{2r},
\qquad c_{uv}^{(h)}=e^{s_{uv}^{(h)}}.
\]

Sampling 보정 a가 있으면 실효 weight는 a_uv c_uv이며 한 번만 곱한다.

\[
\alpha_{i\leftarrow j}^{(h)}
=\frac{a_{ij}e^{s_{ij}^{(h)}}}{\sum_{k\sim i}a_{ik}e^{s_{ik}^{(h)}}}
=\operatorname{softmax}_{j\sim i}(s_{ij}^{(h)}+\log a_{ij}).
\]

정규화 전 c는 대칭이지만 수신 노드마다 분모가 다른 alpha는 일반적으로 비대칭이다.
Q/K를 노드마다 한 번 계산하고 실제 edge에서만 내적을 계산한다.
Dense N×N attention을 만들지 않는다.
이 사실만으로 GATv2보다 빠르거나 같은 attention family라고 결론낼 수 없다.

생성식 출처는 Gruber·Lee·Trask의
[Reversible and irreversible bracket-based dynamics for deep graph neural networks](https://arxiv.org/html/2305.15616v3)와
[저자 attention 코드](https://github.com/natrask/BracketGraphs/blob/f108848fac22516623fe3371869aad6c303fc389/src/attention.py)다.
채택한 것은 양의 대칭 edge score를 쓰는 부분이며 논문의 bracket dynamics 전체 재현은 아니다.
Head별 C, 1/r scaling, 초기화 등의 차이는 [후보 수식](../../experiments/c_learning_bracket/MODEL_MATH.md)에 기록되어 있다.

Signal audit는 projection·mixing·ReLU·dropout 위치별 norm을 관측한다.
RMS 감소만으로 정보가 전부 사라졌다거나 장시간 학습 실패의 원인이 확정됐다고 하지 않는다.
Output initialization 실험도 정보 보완의 효과와 기본 학습의 성립을 구분하기 위한 대조다.

근거: [solver 후보](../../experiments/c_learning_only/MODEL_MATH.md),
[output initialization 실험](../../experiments/output_init_ablation/README.md).

## 7. Wedge: 발생행렬에서 길이 2 경로를 만든다

Physical B는 E×N이다. 별도 행렬 A_w를 만든다.
중심 j의 서로 다른 이웃 i,k를 unordered pair로 모두 한 번씩 열거하며 triangle도 제외하지 않는다.
경로 수를 P라 하면 A_w∈R^(P×N)의 행은 e_i−2e_j+e_k다.

\[
(A_wX)_{(i,j,k)}=X_i-2X_j+X_k=g_2-g_1,\qquad
g_1=X_j-X_i,\quad g_2=X_k-X_j.
\]

Physical edge의 방향 부호를 맞춰 두 차이를 조합하는 P_w를 만들면 A_w=P_wB다.
P_w와 뒤의 copy replication R은 별도 행렬이다.
P_w는 부호 있는 경로 차이의 조합이며 일반 line graph incidence와 자동으로 같아지는 것은 아니다.

\[
Q=A_w^\top A_w,\qquad
\operatorname{tr}(X^\top QX)=\sum_{(i,j,k)}\|X_i-2X_j+X_k\|^2.
\]

Q는 PSD지만 distance-2의 양의 off-diagonal을 가질 수 있다.
따라서 일반적인 양의 edge-weight Laplacian과 같은 family가 아니다.

### 7.1 이름과 계수의 중요한 교정

모든 unordered wedge를 unit weight로 세고 physical graph가 단순 무방향이면 실제 구현의 항등식은

\[
\boxed{Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B},\qquad L=B^\top B.
\]

이전 제안의 −2 식은 **Q 자체가 아니라 Q+2L**다.

\[
\boxed{Q+2L=L^2+B^\top\operatorname{diag}(d_u+d_v-2)B}.
\]

Fixed study의 열 이름 L2는 L@L, 즉 L²를 뜻한다.
다른 설계 문서의 L2(C2)=A_wᵀC2A_w와 혼동하지 않도록 이 요약에서는 후자를 T_C2로 쓴다.
이 교정으로 새 Q+2L 성능 실험을 했다는 뜻은 아니다. 원래 코드의 Q와 비교 조건은 그대로다.

모든 edge의 degree 합이 s이면 Q=L²+(s−4)L이다.
d-regular graph이면 Q=L²+2(d−2)L, cycle graph이면 Q=L²다.
불규칙 그래프에서 span{L,L²}에 대한 fit residual이 남는 것만으로 모든 spectral 방법 대비 우월성이 입증되지는 않는다.

근거: [고정 wedge 수식](../../research/wedge_propagation/MODEL_MATH.md),
[항등식 구현](../../research/wedge_propagation/operators.py),
[fixed study의 열 이름](../../research/wedge_propagation/study.py).

### 7.2 Learned C2와 synthetic teacher/student

합성 teacher는 각 스칼라 경로의 g1,g2에서 다음 score와 상대 weight를 만든다.

\[
r_p^*=\frac{g_1g_2}{|g_1||g_2|+\epsilon}
+\frac{|g_2-g_1|}{|g_1|+|g_2|+\epsilon},\qquad
c_p^*=\frac{\exp(\tanh r_p^*)}{\operatorname{mean}_{q\in P_g}\exp(\tanh r_q^*)},
\quad\epsilon=10^{-8}.
\]

이 weight를 써서

\[
M^*=A_w^\top\operatorname{diag}(c^*)A_wX
\]

를 목표 메시지로 삼는다.
Student는 |g1|+|g2|, g1g2, |g2−g1|, (|g1|−|g2|)²에서 공유 MLP score를 만들고,
exp(tanh(score))를 그래프의 경로 평균으로 정규화한다.

\[
\widehat M=\beta A_w^\top C_{2,\theta}(X)A_wX.
\]

학습은 정규화 node-message MSE이며 C 정답을 직접 supervise하지 않는다.
평균 C2=1과 beta는 상대 weight와 전체 scale을 나누지만 C의 유일한 식별까지 보장하지 않는다.
초기 합성 특징 draw는 독립 스칼라 실현이다. 그 개수를 하나의 공동 다채널 벡터 차원으로 읽지 않는다.

RMS gate는 sigma=sqrt(mean_edges (BX)²)로 **gate 입력만** 정규화한다.
실제 A_wX를 같은 scale로 줄이지 않는다.
양의 특징 배율 a에서 이상적으로 C2(aX)=C2(X), M(aX)=aM(X)가 된다.
이 동변성과 teacher fit, 분류 test 성능, 독립 그래프 일반화는 별개의 관측이다.

### 7.3 Wedge 분류와 두 정규화

분류 층은 대체로

\[
Z=\operatorname{Dropout}(H)W,\qquad
U=Z-\alpha\bar LZ-\beta\bar T_{C2}Z
\]

를 쓴다.
Energy scalar readout을 추가하는 것이 아니라 operator가 직접 전파한다.
Global scale은 고정 Q의 대각과 최대 대각 비율 kappa를, node scale은 현재 C2의 대각을 쓴다.

\[
\bar T_{global}
=\frac1{3\kappa}D_Q^{-1/2}A_w^\top C2 A_wD_Q^{-1/2},
\qquad
\bar T_{node}
=\frac13D_{C2}^{-1/2}A_w^\top C2 A_wD_{C2}^{-1/2}.
\]

D_Q=diag(diag(Q)), D_C2=diag(diag(A_wᵀC2A_w))이며 대각이 0인 inverse는 0이다.
Node scale은 단순히 kappa를 빼는 것이 아니라 양쪽 노드별 scale도 바꾼다.
현재 C2의 정규화 의존성까지 학습에서 미분한다. C2=I면 두 정규화가 일치한다.
고정 C2의 norm bound를 입력 의존 gate를 포함한 전체 Jacobian이나 학습 수렴의 보장으로 확대하지 않는다.

근거: [teacher/student](../../research/wedge_propagation/learned/MODEL_MATH.md),
[RMS gate](../../research/wedge_propagation/scale_normalization/MODEL_MATH.md),
[wedge 분류 수식·실험 설계](../../research/wedge_propagation/classification/EXPERIMENT_DESIGN.md),
[node normalization](../../research/wedge_propagation/node_normalization/MODEL_MATH.md).

## 8. 로컬 E/J가 실제로 측정한 관계

중심 v의 1홉 induced local은 S_v={v}∪N(v)이며 그 집합 안의 모든 물리 edge를 포함한다.
중심의 star edge뿐 아니라 이웃끼리의 edge도 포함한다.
R_v를 노드 선택 행렬로 두면

\[
g_v=B_vR_vH,\quad q_v=C_vg_v,\quad d_v=B_v^\top q_v=L_vR_vH,
\]
\[
E_v=\operatorname{tr}((R_vH)^\top L_v(R_vH)).
\]

C_v는 I 또는 c_v,ab=2/(d_v(a)+d_v(b))다.
여기서 d_v는 로컬의 무가중 degree다.
같은 물리 edge도 어느 local에 속하느냐에 따라 local_degree C가 달라질 수 있다.

Primary J는 공통 물리 노드에서 **signed 집계끼리의 내적**이다.

\[
\boxed{J_{vu}=\sum_{a\in S_v\cap S_u}\langle d_v[a],d_u[a]\rangle}.
\]

전체 노드 좌표로 확장한 발생행렬 barB를 쓰면

\[
J_{vu}=\operatorname{tr}(q_v^\top\bar B_v\bar B_u^\top q_u).
\]

이는 서로 다른 edge 집합 사이의 직사각형 연산자를 쓰는 쌍선형 관계다.
그러나 물리 union의 경계 edge에서 만든 L_vu 자체는 아니다.
고정 C와 공통 특징 좌표에서 bilinear라고 해석할 수 있고 J의 부호는 음수가 될 수 있다.
같은 물리 edge의 항을 J_shared로 따로 계산하면 J_distinct=J−2J_shared다.
2는 incidence 행의 자기 내적이 2이기 때문이다.

||q_v||²=Σc²||g||²와 E_v=Σc||g||²도 다르다.
고정 감사의 H0/H1/H2는 물리 그래프에서 만든 고정 확산 reference 입력이며 새로운 학습층이 아니다.

### 8.1 E_v+2J_vu+E_u가 자동으로 넓은 그래프 에너지인가

전체 노드 좌표의 로컬 operator를 A_v=R_vᵀL_vR_v라 하면

\[
E_v=\operatorname{tr}(H^\top A_vH),\qquad J_{vu}=\langle A_vH,A_uH\rangle_F.
\]

이 J를 교차항으로 갖는 제곱은

\[
\|(A_v+A_u)H\|_F^2
=\operatorname{tr}(H^\top A_v^2H)
+2J_{vu}+\operatorname{tr}(H^\top A_u^2H).
\]

여기의 내부항은 A_v²이며 현재 E_v의 A_v와 다르다.
현재 E와 J를 더해도 H의 이차식으로 쓸 수 있지만,
같은 큰 라플라시안의 block에 대응하는지와 PSD인지는 별도 조건이다.

**같은 큰 이차형식의 내부 block과 교차 block을 맞춰 더하면 넓은 에너지다.
현재의 E와 현재의 J를 그대로 더하는 것만으로 그 조건이 충족되지는 않는다.**

근거: [고정 E/J 수식](../../research/local_energy_relations/MODEL_MATH.md),
[E/J kernels](../../research/local_energy_relations/prediction/operators.py).

### 8.2 Receiver aggregation의 복원 감사

Source local v에서 receiver local u의 공통 노드 a에 r_(v,u,a)=d_v[a]를 보낸다.
같은 receiver/tag에서 합친 Y는 임의 receipt를 입력하면 집계 영공간을 가진다.
그러나 실제 receipt는 모두 **공통 물리 특징 H와 알려진 양의 로컬 Laplacian**에서 생성된다는 제약이 있다.

모든 receiver의 중심 행에는

\[
Y_{(u,u)}=\sum_{b\sim u}w_{ub}(H_u-H_b),\qquad w_{ub}>0
\]

가 남는다.
w가 대칭일 필요는 없지만 연결 성분의 최대값 원리로 이 연산의 kernel은 상수뿐이다.
따라서 전체 Y는 H를 성분별 상수까지 정하고 q·receipt·E·J를 다시 계산할 수 있다.
이 결과는 임의 receipt의 복원과 다른 제한된 조건의 결과다.
Partial receiver, 자유 flow, 미지 C, 서로 다른 source 특징, 비선형 압축 이후로 자동 확장하지 않는다.

Decoder는 **Y만**으로 복원한다. E/J 추가로 복원율을 높인 실험은 아니다.
임의 집계 공간에서 없어지는 성분과 실제 허용 입력에서 식별 불가능한 성분을 나눈 감사다.
근거: [Receiver 수식과 조건](../../research/local_energy_relations/receiver_aggregation/MODEL_MATH.md).

### 8.3 이후 scalar E/J 분류 후보

이전 prediction/placement는

\[
U_{base}=(1-\alpha)Z+\alpha P_C^2Z,\qquad
U=U_{base}+e\,w_E^\top+j\,w_J^\top
\]

를 쓴다.
e/j는 E/J를 topology·채널 scale로 정규화한 **노드별 scalar**다.
Unit이어도 P_C의 물리 edge weight는 1+공통 이웃 수이며 표준 GCN이 아니다.
같은 현재 projection Z에서 J를 계산하므로 무관한 층별 channel 좌표끼리의 내적은 아니다.
첫 층만·출력층만·양층의 placement는 별도로 재학습한다.
Readout 출력 폭도 달라 완전한 parameter-matched 대조는 아니다.
이 모델의 결과는 다음 copy energy 전파 모델의 결과가 아니다.

근거: [prediction 수식](../../research/local_energy_relations/prediction/MODEL_MATH.md),
[placement 수식](../../research/local_energy_relations/placement/MODEL_MATH.md).

## 9. 공통 노드, 중복 edge, union energy

겹치는 노드를 제거할 필요는 없다.
물리 union에서 각 노드를 한 번씩 쓰려면 겹치지 않는 세 집합으로 나눈다.

\[
P=S_v\setminus S_u,\qquad O=S_v\cap S_u,\qquad T=S_u\setminus S_v.
\]

O를 한 번만 좌표에 두고 정확한 union incidence와 weight를 정하면

\[
L_{union}=B_{union}^\top C_{union}B_{union},\qquad
E_{union}=\operatorname{tr}(X_{union}^\top L_{union}X_{union})
\]

를 P/O/T의 3×3 block으로 전개할 수 있다.
각 diagonal block은 **집합 사이 edge의 degree도 포함한다**.
Off-diagonal block은 대응 weighted adjacency의 음수다.
독립된 각 집합의 Laplacian을 diagonal에 두고 교차 내적만 붙이면 경계 degree가 빠진다.

E(S_v)∪E(S_u)와 S_v∪S_u의 induced edge set도 다를 수 있다.
P–T edge는 두 기존 local 어느 쪽에도 없지만 induced union에는 들어갈 수 있다.
어떤 edge를 포함하는 넓은 그래프인지 먼저 정의해야 한다.

Copy 표현에서는 O를 O_v/O_u로 나누어 P/O_v/O_u/T의 네 집단을 둘 수 있다.
이는 물리 좌표 세 집단 표현과 역할이 다르다.
현재 모델은 같은 물리 노드 copy를 연결하는 **copy graph**를 쓴다.
같은 노드를 공유하는 것, 물리 edge를 여러 local에 중복 포함하는 것,
copy 사이의 결합 edge를 추가하는 것을 구분한다.

### 9.1 물리 edge의 중복 계산

물리 edge e가 m_e개의 local에 있으면

\[
\sum_vE_v
=\sum_e\left(\sum_{v:e\in E_v}c_{v,e}\right)\|(BH)_e\|^2.
\]

물리 edge를 한 번만 계산하려면 local occurrence별 계수를 조정해야 한다.
예를 들어 c_v,e/m_e로 합치면 물리 weight는 local C의 평균이다.
Unit이면 원래 unit physical energy다.
Local-specific C면 평균 가중 에너지이며, 모든 local에서 같은 physical C를 쓸 때만 그 C의 에너지를 복원한다.

현재 copy 모델은 모든 local 내부 edge를 유지하고 1/m_e 보정을 넣지 않는다.
정의된 copy graph energy이며 물리 union energy와 같다고 주장하지 않는다.
다음 M의 평균은 **노드 copy 개수 보정**이다. 물리 edge의 중복 횟수 m_e 보정과 다르다.

## 10. Copy 공간의 결합 에너지와 새 전파

### 10.1 로컬마다 별도 상태를 둔다

Copy (v,a)는 중심 v의 local에 속한 물리 노드 a의 상태다.
총 copy 수를 m이라 하면

\[
A=\operatorname{blockdiag}_vL_v\in\mathbb R^{m\times m},\qquad
R\in\mathbb R^{m\times N},\qquad (RH)_{(v,a)}=H_a,
\]
\[
D=R^\top R=\operatorname{diag}(\text{노드별 copy 수}),\qquad
M=D^{-1}R^\top\in\mathbb R^{N\times m},\qquad MR=I.
\]

각 노드는 자기 중심 local에 포함되므로 D는 양의 대각이다.
각 local은 A에 한 번씩 넣는다.
중심 pair마다 같은 내부 block을 만들어 중복 합산하지 않는다.

### 10.2 K는 같은 물리 노드의 서로 다른 문맥을 연결한다

인접 중심의 unordered pair v<u에서 모든 공통 노드 a에 대해 (v,a)와 (u,a)를 한 번씩 연결한다.
Copy incidence를 J_copy라 하면

\[
K=J_{copy}^\top WJ_{copy},\qquad W=I.
\]

J_copy와 이전 scalar J_vu는 다른 기호다.
K는 물리적으로 다른 a,b를 새로 연결하는 것이 아니라
**같은 a가 두 local에서 얻은 문맥별 상태**를 연결한다.
중심 pair와 공통 노드의 같은 연결을 방향별로 두 번 세지 않는다.

\[
KR=0,\qquad R^\top K=0,\qquad MK=0.
\]

처음 RH에서는 같은 물리 노드의 copy가 모두 같으므로 cross energy는 0이다.
Local A 이후에는 copy가 문맥별로 달라져 cross가 작용할 수 있다.

\[
\mathcal E(Y)=\tfrac12\operatorname{tr}(Y^\top(A+\lambda K)Y)
\]

는 λ≥0일 때의 결합 copy 에너지다.
두 local block에서 K는 음의 off-diagonal과 양의 degree diagonal을 함께 가진다.
내부·교차의 모든 block이 같은 operator에서 나오므로 PSD인 큰 **copy energy**다.
이전 scalar E/J readout을 이 식으로 바꿔 읽는 것은 아니다.

### 10.3 Cross 직후 평균하면 증가분이 사라진다

이전 fixed audit는 graph별 eta=.5/max weighted intra degree,
gamma=.5/max unit cross degree를 사용했다.

\[
M(I-\gamma\rho K)(I-\eta A)RH=M(I-\eta A)RH.
\]

MK=0이므로 cross가 copy 상태를 바꿔도 바로 평균하면 증가분이 사라진다.
이는 copy 상태를 유지하는 순서가 왜 필요한지 보여 주는 대조다.

### 10.4 Sandwich는 cross 이후 다시 local을 적용한다

\[
T_\rho=M(I-\eta A)(I-\gamma\rho K)(I-\eta A)R,\qquad
T_0=M(I-\eta A)^2R.
\]

Cross off도 내부 전파 두 번을 유지한다.
동일 입력 H의 정확한 증가분은

\[
\boxed{(T_\rho-T_0)H=-\rho\gamma\eta^2MAKARH}.
\]

첫 A가 문맥 차이를 만들고, K가 이를 재분배하고, 둘째 A가 평균 후에도 남는 변화를 만든다.
Scalar energy를 벡터로 바꿔 더하는 방식이 아니라 energy operator를 copy 상태에 직접 쓴다.

동일한 앞뒤 A, 대칭 PSD K, 양의 step/gain, 정확한 R/M에서
첫 A 이후 K action이 비영이면 마지막 on/off 차이도 비영이다.
이는 같은 입력에 대한 한 macro의 보장이다.
분류 개선, 원 flow 복원, 모든 특정 노드쌍 derivative의 비영을 보장하지 않는다.

### 10.5 Persistent 대조는 세 번째 step에서 달라진다

같은 s를 써서

\[
M(I-s(A+\rho K))^tRH
\]

를 K off와 비교하면 t=1,2의 차이는 0이고 t=3의 차이는

\[
-s^3\rho MAKARH
\]

다.
Copy를 유지하더라도 처음 물리 입력에서 출발해 한두 step만 진행하고 평균하면 cross가 물리 출력에 남지 않는다.
고정 A/K의 전체는 하나의 물리 operator T로 접을 수 있다.
Copy를 도입했다는 사실만으로 모든 선형 node operator보다 표현력이 높다고 주장하지 않는다.

근거: [fixed audit 수식](../../research/local_context_coupling/MODEL_MATH.md),
[sparse operators](../../research/local_context_coupling/operators.py),
[audit study](../../research/local_context_coupling/study.py).

## 11. 분류 연결과 최신 정규화

### 11.1 이전 여섯 조건에서 배운 것은 무엇인가

여섯 조건은 unit/local_degree × off/fixed/learned다.
C와 unit copy K는 고정이며 rho는 off=0, fixed=1, learned=sigmoid(theta)다.
Theta는 seed 모델마다 하나이며 두 macro에서 공유한다. 초기 theta=0, rho=.5다.
Off/fixed에 사용하지 않는 theta는 등록하지 않는다.

\[
Z_0=\operatorname{Dropout}_{.5}(X)W_0,\qquad
H_1=\operatorname{ReLU}(T_\rho Z_0),
\]
\[
Z_1=\operatorname{Dropout}_{.5}(H_1)W_1,\qquad logits=T_\rho Z_1.
\]

W0는 F_in×64, W1은 64×classes다. Bias와 추가 출력 projection은 없다.
Train mask의 평균 CE로 projection과 learned theta를 Adam에서 갱신한다.
Projection에만 weight decay 5e−4를 적용하고 theta에는 적용하지 않는다.
Energy·복원·teacher 보조 loss는 없다.
**Learned는 교차 강도를 배운다는 뜻이며 C 학습은 아니다.**

본학습은 252개의 독립 모델을 각각 500 epoch 학습하는 계약이며 서버 FULL 완료 결과를 수령했다.
같은 C에서 on/off 차이가 작은 결과와 unit/local_degree 사이의 큰 성능 차이를 나누어야 한다.
후자에는 C에 따라 달라지는 graph eta도 포함된다. C policy 차이를 K의 효과로 옮겨 해석하지 않는다.
수치는 [여섯 조건 서버 결과](../LOCAL_CONTEXT_CLASSIFICATION_SERVER_FINDINGS_20261004.md)를 따른다.

### 11.2 최신 모델은 step을 대칭 S/G에 포함한다

Graph g의 최대 intra weighted degree를 d_max,g,
local v의 최대값을 d_max,v라 하면

\[
S_{graph}=\eta_gA,\qquad \eta_g=.5/d_{max,g},
\]
\[
S_{local}=\operatorname{blockdiag}_v(\eta_vL_v),\qquad \eta_v=.5/d_{max,v}.
\]

최대 degree가 0이면 해당 step은 0이다.
Local block 안의 eta는 scalar이므로 대칭성을 유지한다.
노드별 row scaling의 비대칭 연산자로 바꾸지 않는다. 앞뒤 pass는 같은 S다.

원래 unit K에서 copy i의 degree를 k_i라 하면

\[
G_{graph}=\gamma_gK,\qquad \gamma_g=.5/\max_{i\in g}k_i,
\]
\[
\boxed{G_{edge}=J_{copy}^\top\operatorname{diag}
\left(\frac{.5}{\max(k_i,k_j)}\right)J_{copy}}.
\]

Edge policy는 원래 unit degree를 분모로 쓴다.
정규화 이후 degree로 다시 나누지 않는다. 모든 canonical link를 유지한다.
둘 다 같은 물리 노드 copy끼리만 연결하므로 GR=MG=0이다.

\[
\boxed{T_\rho=M(I-S)(I-\rho G)(I-S)R},\qquad
\boxed{T_\rho-T_0=-\rho MSGSR}.
\]

Graph/graph는 이전 여섯 조건과 같은 수식이다.
다만 step을 edge weight에 미리 곱하므로 float32 합산 순서는 달라질 수 있다.
Float64 action·gradient 일치와 과거 학습의 bitwise 재현을 구분한다.
이전 여섯 조건도 새 study 안에서는 새로 학습한다.

### 11.3 비영 보장과 D metric에서의 안정성

\[
\operatorname{tr}[Z^\top D(T_0-T_\rho)Z]
=\rho\|G^{1/2}SRZ\|_F^2\ge0.
\]

정확한 R/M, 같은 앞뒤 대칭 S, PSD G에서 rho>0이고 G(I−S)RZ가 비영이면 오른쪽은 양수다.
따라서 matched output 증가분도 비영이다.
그 이후 projection·ReLU·dropout·분류가 이 차이를 유용하게 쓰는지는 보장하지 않는다.
재학습으로 hidden이 서로 달라진 조건에 동일 Z의 식을 그대로 적용하지 않는다.

S/G의 최대 weighted degree는 각각 .5 이하이며 eigenvalue는 [0,1] 안에 있다.
Rho∈[0,1]이면 copy 공간의 sandwich는 PSD contraction이다.
Q_c=RD^(-1/2)는 Q_cᵀQ_c=I이며

\[
D^{1/2}T_\rho D^{-1/2}
=Q_c^\top(I-S)(I-\rho G)(I-S)Q_c
\]

도 symmetric PSD contraction이다.
물리 노드의 보장 norm은

\[
\|H\|_D^2=\operatorname{tr}(H^\top DH),\qquad
\|T_\rho H\|_D\le\|H\|_D
\]

다.
Copy 수가 노드마다 달라 일반 Euclidean node norm으로 바꿔 주장하지 않는다.
W를 포함한 전체 네트워크 norm이 반드시 감소하는 것도 아니다.
각 pass에서 내부와 교차 에너지가 모두 항상 감소한다고 가정하지 않는다.

### 11.4 C 이름은 다르지만 적용 S가 같은 경우

한 local의 모든 weight를 c배 하면 L_v와 최대 weighted degree가 함께 c배 된다.
Local eta에서는 그 공통 c가 상쇄된다.
삼각형이 없는 물리 그래프의 모든 1홉 induced local은 star 또는 isolate이며,
local_degree C는 local 안에서 공통 scalar가 된다.
따라서 **local policy에서 S_unit=S_local_degree**다.
Graph policy는 다른 local의 degree bound도 사용하므로 모든 구조에서 상쇄되는 것은 아니다.

Complete clique, 상수 입력, empty graph에서는 문맥 차이가 없거나 operator가 영이므로 cross 차이도 0이다.
비영 보장의 전제가 성립하지 않는 입력에서 0이 나오는 것은 오류가 아니다.

### 11.5 20조건과 최신 상태

조건 ID는 weight__intra__cross__variant다.

- Weight: unit / local_degree.
- Intra: graph / local.
- Cross/variant: none/off, graph/fixed, graph/learned, edge/fixed, edge/learned.

2×2×5=20개다. Off를 graph/edge별로 중복 등록하지 않는다.
Off의 진단용 G는 graph policy지만 실제 gain은 0이다.
두 macro, hidden64, dropout.5, 전체 topology·citation 특징·public mask를 유지한다.
바뀐 연구 요인은 고정 대칭 정규화다.

2026-10-05에 제출된 서버 원문은 FULL 완료를 보고했다.
Tuning 540회와 final 300회, 총 840회이며 각각 500 epoch를 수행한 계약상 모델 갱신은 420,000회다.
원문은 새 갱신 횟수도 420,000회로 표시한다. 이전 여섯 조건은 이 실행 안에서 새로 튜닝·학습했으며 과거 수치를 섞지 않았다.
결과 요약은 metric 900행, frozen 4,320행, branch 3,480행의 범위와 hash 검증을 보고한다.
이것은 사용자가 제출한 completion·summary 원문의 보고이며 서버의 모든 CSV·checkpoint를 독립 수령해 검사한 것과 구분한다.

관측된 결과는 다음처럼 나누어 읽는다.

- **내부 정규화:** Unit C·cross off에서 graph→local의 test accuracy 차이는 Cora +21.46pp, CiteSeer +12.74pp, PubMed +4.64pp다.
  Cross를 끈 상태에서도 나타나는 변화이므로 이 큰 차이를 교차항의 효과로 돌리지 않는다.
- **교차 연산의 작용:** Local+edge fixed 첫 층의 matched Δ/off는 약 1.65–2.59%다.
  같은 local 내부에서 graph G보다 약 49–86배 큰 출력 변화를 만들었다.
  이는 연산이 현재 입력에 작용했다는 관측이며 정보 복원이나 분류 개선 자체는 아니다.
- **분류:** Cross-on/off accuracy 비교 48개는 모두 paired 95% 구간에 0을 포함한다.
  추가 accuracy 개선을 확인하지 못했으며 동등성을 증명한 것도 아니다.
  CE 비교 48개 중 18개는 개선 방향으로 0을 제외한다. 같은 public split의 5개 seed와 다중 비교 보정 없는 탐색적 결과다.
- **학습 대상과 자료 범위:** Learned gain 평균은 초기 .5에서 조건별로 달라졌지만 C는 계속 고정이다.
  제출 원문에 epoch별 gradient/update, frozen 개입의 상세 ΔCE·flip, formula residual·degree max 값은 없다.
  코드의 학습 연결과 이번 서버에서 관측된 상세 수치를 구분한다.

자세한 결과와 누락된 증거는 [새 FULL 결과 정리](08_NORMALIZATION_FULL_RESULTS.md)와
[서버 결과 판정](../LOCAL_CONTEXT_NORMALIZATION_SERVER_FINDINGS_20261005.md)을 따른다.
기존 2026-10-04 묶음과 과거 결과는 당시 상태로 보존한다.

근거: [최신 수식](../../research/local_context_coupling/normalization/MODEL_MATH.md),
[operators](../../research/local_context_coupling/normalization/operators.py),
[model](../../research/local_context_coupling/normalization/model.py),
[FULL 설정](../../research/local_context_coupling/normalization/config_full.json).

## 12. 진단, frozen 개입, 재학습 비교

최신 branch energy는 **applied S/G**의 ½tr(YᵀSY), ½tr(YᵀGY)다.
이전 raw A/K energy와 같은 값으로 비교하지 않는다.
Context norm은 ||GY1||, raw_context_norm은 ||KY1||이다.
Matched delta는 같은 Z에서 ||T_rho Z−T0 Z||이며 delta/off의 분모는 ||T0 Z||다.
분모가 0인 비율은 undefined/null로 남기고 임의 숫자를 넣지 않는다.

| 비교 | 확인하는 질문 | 그 결과만으로 확인하지 못하는 것 |
| --- | --- | --- |
| 새로 학습한 fixed/learned와 off | 해당 구조로 최적화하면 분류가 좋아지는가 | 선택 checkpoint의 현재 cross 의존성 |
| 고정 checkpoint의 gain0/gain1 | 다른 학습 weight를 유지하면 cross가 출력·CE를 바꾸는가 | cross 없는 구조로 재학습한 성능 |
| Theta/rho의 변화 | gain이 초기값에서 바뀌었는가 | 엣지 C 학습 또는 분류 개선 |
| 첫 pass 이후 cross energy | 같은 노드 copy가 문맥별로 달라졌는가 | 그 차이가 누락 원 flow 자체인가 |
| Matched delta와 exact formula residual | 의도한 operator가 입력에 작용하는가 | 일반화·우월성·정보 복원 |

Fixed/learned checkpoint는 gain0/gain1 × layer_0/layer_1/both의 6개 probe를 모두 평가한다.
첫 층 개입이면 뒤 hidden도 다시 계산한다.
Fixed gain1은 예상 no-op다.
Off는 original metric/branch만 출력하며 불필요한 off probe를 추가하지 않는다.
Checkpoint hash를 보존하고 optimizer update는 0회다.

학습에서 보지 않은 sample context, 같은 graph의 큰 문맥, 독립 새 graph는 다른 일반화 축이다.
Citation public mask 분류만으로 독립 그래프 일반화를 입증하지 않는다.
Public test를 이미 본 이후의 후속 탐색 연구라는 점도 해석에 남긴다.

## 13. GCN, GraphSAGE, GATv2와의 관계

Unit C이면 L=BᵀB=D−A_adj다. 그러나 표준 GCN은 raw L을 그대로 곱하지 않는다.

\[
\widehat A=A_{adj}+I,\qquad \widehat D=D+I,
\]
\[
P_{GCN}=\widehat D^{-1/2}\widehat A\widehat D^{-1/2}
=I-\widehat D^{-1/2}L\widehat D^{-1/2},\qquad
H'=\sigma(P_{GCN}HW).
\]

Self-loop, 양쪽 degree 정규화, projection, 비선형이 들어간다.
GraphSAGE mean도 BᵀB만 곱하는 방식이 아니며 mean neighbor와 별도 root transform을 쓰는 설정이 있다.
Unit C의 row diffusion이 mean aggregation과 가깝다는 것과 GraphSAGE 전체가 같다는 것은 다르다.

GATv2는 projection/attention score와 수신 이웃 softmax를 쓴다.
대칭 c의 row alpha는 국소 attention으로 해석할 수 있지만,
GATv2와 같은 score family·self-loop·head 결합·초기화는 아니다.
Residual 여부만 맞춰도 내부 signal path가 같아지지는 않는다.
Residual 부재만으로 실패 원인을 확정하지 않고 projection·정규화·학습 등의 대조를 확인한다.

이전 wedge 분류에는 GCN 대조가 있지만 현재 six/twenty copy 분류에는 동일 조건의 직접 GCN 대조가 없다.
별도 protocol의 과거 GCN 수치를 새 모델의 승패로 사용하지 않는다.
근거: [GCN 원 논문](https://arxiv.org/abs/1609.02907),
[GATv2 원 논문](https://arxiv.org/abs/2105.14491),
[저장된 baseline 구현 계약](../ARXIV_BASELINE_COMPARISON.md).

## 14. 별도 연구: Cycle PE, tree chart, 보류된 flow completion

### 14.1 Cycle PE와 실제 flow의 cycle 성분

Topology의 cycle basis F_T∈R^(E×beta), beta=E−N+k는 BᵀF_T=0을 만족한다.
임의 cycle flow를 이 기저로 표현할 수 있지만,
기저에서 positional encoding을 만든 것만으로 현재 q의 cycle 값을 복원하지는 않는다.

이전 cycle_set은 BFS fundamental cycle의 길이·membership 등의 6가지 통계를 쓴다.
Cycle 열 부호·순서의 불변성과 spanning tree 자체를 바꾼 기저의 불변성은 다르다.

\[
P_{cycle}=F_T(F_T^\top F_T)^\dagger F_T^\top
\]

는 같은 cycle 공간의 기저 변경에 불변인 projector다.
E×E dense 표현은 큰 비용이 들 수 있다.
이전 projector 후보와 현행 sparse DFS v2의 구현·cache·결과를 섞지 않는다.

### 14.2 현행 Cycle PE V2: sparse DFS SE와 상대 위치 PE

DFS의 각 chord에서 beta개 전체 fundamental cycle을 만든다.
각 cycle에 고유 chord가 있어 전체 기저가 되며 QR/SVD로 rank를 보충하지 않는다.
Unsigned membership A_c=|F_T|를 이용한 edge→cycle→edge SE와,
cycle 순서의 cos/sin을 이용한 상대 PE를 비교한다.

\[
R_{rel}=D_r^{-1}
[F_cD_\ell^{-1}F_c^\top V+G_cD_\ell^{-1}G_c^\top V].
\]

F_c=A_c cos(theta), G_c=A_c sin(theta)이며 D_ell은 cycle 길이,
D_r은 edge별 membership 수다.
상대 kernel은 cycle 위치 차이의 cos에 해당한다.
Cycle 시작점·방향 변경에 불변이지만 임의 기저 변경이나 다른 DFS tree에 불변인 projector는 아니다.
DFS 탐색이 O(N+E)여도 기저 출력은 O(nnz F_T), 집계는 O(nnz F_T × width)다.
Nnz는 커질 수 있으므로 전체 PE 비용이 무조건 선형이라고 하지 않는다.

근거: [Cycle PE V1](../CYCLE_PE.md),
[현재 V2 수식과 역사](../../gpt_handoff/CYCLE_PE_V2.md).
저장소의 관련 원 논문 링크는 [SignNet](https://arxiv.org/abs/2202.13013),
[PEARL](https://arxiv.org/abs/2502.01122) 등이다.
현재 모델이 그 논문들의 정확한 재현이거나 새로운 방법이라고 단정하는 근거는 아니다.

### 14.3 Full-beta tree chart와 tree augmentation

두 tree chart T,T'의 전체 기저는 같은 cycle 공간을 생성한다.
좌표 전이 Q가 존재해

\[
F_{T'}Q_{T'\leftarrow T}=F_T,\qquad
a_{T'}=Q_{T'\leftarrow T}a_T
\]

가 된다.
전체 beta 좌표를 사용하면 flow를 유지하며 chart를 바꿀 수 있다.
세 chart에서는 Q_(T''←T')Q_(T'←T)=Q_(T''←T)의 cocycle 법칙을 검사한다.
이 Q는 tree 좌표 전이이며 wedge Gram Q, copy 정규화 Q_c와 다르다.
k<beta 절단은 lossless가 아니며 현행 lossless 실험에서는 비활성화되어 있다.

Tree augmentation은 물리 graph와 label을 유지하고 학습 중 BFS/DFS chart를 바꿔 예측의 강건성을 확인한다.
Wilson UST를 학습하지 않은 sampler family로 평가하는 protocol이 있다.
새 draw, sampler family 변경, 물리 graph OOD를 나눈다.
Sampler family가 달라도 같은 물리 tree가 나올 수 있으며 이를 허용하고 보고한다.
Chart 좌표의 선형 전이가 lossless여도 비선형 set encoder의 임의 chart 불변성을 보장하지 않는다.
Tree 연구에 conductance C나 flow completion을 결합하지 않았다.

근거: [Tree 연구 설명](../TREE_AUGMENTATION.md),
[augmentation 구현](../../research/tree_augmentation/augmentation.py),
[공통 선형대수](../../src/chartgat/algebra.py).

### 14.4 Combined prototype는 보류 상태다

이전 prototype는 conductance, persistent cycle 좌표, hard-constrained flow completion,
chart-equivariant update를 결합했다.
현재 독립 track의 증거로 사용하지 않으며 root smoke pipeline에서 제외한다.

고정 C와 알려진 노드 집계 d에 대응하는 particular flow는 q_part=CBL_C^dagger d다.
일반 flow의 해는

\[
q=q_{part}+F_Ta
\]

이지만 자유 cycle 좌표 a는 d만으로 결정할 수 없다.
관측 edge flow, 추가 조건, loss 중 무엇으로 a를 결정할지가 별도 문제다.
이것과 제4절의 제한된 q=CBX 복원을 구분한다.
저장된 prototype 결과는 과거 합성 관측이며 최신 C/copy/분류 모델에서 flow completion이 완료된 증거는 아니다.

근거: [보류의 공식 설명](../COMBINED_LATER.md),
[역사를 포함한 전체 인계](../../gpt_handoff/HANDOFF.md).

## 15. 이 자료의 판단 범위

자료는 실제 state·operator·energy·학습 대상과 조건부 수학 항등식을 설명한다.
최신 copy 모델은 같은 물리 노드의 문맥별 상태를 유지하고 내부·교차 PSD 연산자를 사용하는 message passing이다.
고정 audit와 matched formula는 연산의 작용을, 분류 FULL은 실제 예측 결과를 각각 보여 준다.

다음은 이 자료만으로 확정하지 않는다.

- 모든 정보가 보존·복원된다는 것.
- Energy가 비영이면 분류가 개선된다는 것.
- Scalar E/J, wedge, copy 결합이 같은 모델이나 같은 가설이라는 것.
- C=I 초기값이나 작은 cross 증가분만으로 학습 실패 원인이 정해진다는 것.
- Citation transductive test가 독립 graph 일반화를 보여 준다는 것.
- 기존 연구와 완전히 같거나 연구의 신규성이 성립한다는 것.

선행연구와의 관계는 실제 채택 부분과 변경 부분을 원자료에 대응시켜 판단한다.
기존 링크 [Kalofolias, 2016](https://proceedings.mlr.press/v51/kalofolias16.html)은 graph weight 학습의 선행 예지만,
현재 signed cost·finite unroll·task loss 모델의 정확한 구현은 아니다.
이름이 비슷한 iterative GNN 논문과 현재 copy 구조의 동일성은 이 자료에서 확립하지 않았다.

두 ChatGPT share 링크는 가져오지 못해 대화 전체를 복원했다고 주장하지 않는다.
대화에서 비롯한 설명은 사용자가 session에 제공한 본문·첨부와 검증 가능한 저장 코드·문서를 근거로 한다.
최신 정규화 FULL 완료 보고와 요약 수치는 수령했다.
**같은 C·intra에서의 cross 비교**와 **내부·교차 정규화 자체의 비교**를 나누면,
현재의 큰 accuracy 변화는 내부 정규화에서 확인되고 교차항은 일부 CE 개선을 보인다.
교차 작용의 증가, gain 학습, 분류 이득, 정보 복원은 계속 별개의 판단 대상이다.
이 자료가 gradient/update나 frozen 개입의 미수령 원시 수치까지 확인한 것으로 해석하지 않는다.
