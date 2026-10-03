# Experiment 4.1 계산 설명

## 저장된 모델

각 wedge p=(i,j,k)의 행은 A[p,:]=e_iᵀ−2e_jᵀ+e_kᵀ이다.
Q=AᵀA이며 q_v=Q[v,v]다. S=diag(q_v^−1/2)이고 q_v=0인 노드의 S값은 0이다.
물리 엣지 발생행렬 B로 L=BᵀB, L̄=(1/2)D^−1/2 L D^−1/2를 만든다.

현재 층의 투영 특징 Z=HW에서 기존 gate로 양의 diagonal C를 생성한다.
모든 경로의 C 평균은 1이다. Gate 파라미터는 Experiment 4에서 CE로 학습된 그대로다.

\[
R_C(Z)=\tfrac13 S A^\top C(Z) A S Z,\qquad
\kappa(C)=\max_{v:q_v>0}\frac{(A^\top C A)_{vv}}{q_v},\qquad
M_C(Z)=R_C(Z)/\kappa(C).
\]

\[
U=Z-\alpha\bar L Z-\beta M_C(Z).
\]

첫 층 U에 ReLU를 적용하고 두 번째 층으로 보낸다. 두 번째 U는 분류 logits다.
평가에서 dropout은 꺼지며 W, gate, α, β 모두 고정된다.
Q의 quadratic energy는 tr(ZᵀQZ)=||AZ||²지만 이번 진단의 변경 대상은 실제 분류에 들어가는 M이다.

## 바꾸는 메시지

현재 Z에서 참조 C와 κ를 먼저 계산한다. R_I=(1/3)S AᵀA S Z다.
기존 manifest의 permutation π로 C를 경로 사이에서 섞은 값을 C_π라 한다.

| 처리 | 분류에 들어갈 M |
| --- | --- |
| baseline | R_C/κ |
| true_c_unit_denominator | R_C |
| identity_hold_reference | R_I/κ |
| identity_unit_denominator | R_I |
| identity_norm_matched | R_I · ||M_C||_F / ||R_I||_F |
| branch_off | 0 |
| shuffle_hold_reference | R_Cπ/κ |
| shuffle_norm_matched | (R_Cπ/κ) · ||M_C||_F / ||R_Cπ/κ||_F |

모든 norm은 seed마다 현재 그래프의 전체 노드·channel을 사용한다. Label을 사용하지 않는다.
β가 고정되므로 norm match는 βM의 norm도 일치시킨다.
αL̄Z와의 각도나 상쇄, U의 norm, 다음 층의 norm까지 같게 만들지는 않는다.
Candidate와 참조 norm이 둘 다 0이면 gain=1로 실제 0을 유지한다.
Candidate norm=0이고 참조 norm>0이면 명시적인 오류를 발생시킨다.
0 분모 비율·cosine은 undefined flag와 null로 기록한다.

## 두 종류의 관측

**Fixed Z:** 원래 forward의 Z를 각 층에서 고정하고 위 메시지만 비교한다.
이 관측의 message norm·각도 차이는 다음 층의 입력 변경과 섞이지 않는다.

**전체 forward:** layer_0, layer_1, both에 각각 적용하고 train/validation/test metric을 계산한다.
첫 층을 바꾸면 두 번째 Z도 달라진다. 그때 두 번째 층은 변경된 현재 Z에서
참조 C와 κ를 새로 계산한다. 원래 clean forward의 κ를 빌려 쓰지 않는다.

## 기록하는 수치

- ||Z||_F, ||αL̄Z||_F, ||βM||_F 및 입력 대비 비율.
- 일차/이차 메시지와 Z의 cosine, 참조/변경 메시지의 cosine과 차이 norm.
- 학습된 α, β, 실제 사용 κ, norm match gain.
- Active node의 diagonal ratio 평균·p50·p90·p99·최댓값·max node ID·최댓값 1% 이내 비율.
- 고정 checkpoint의 실제 CE/accuracy, 원래 값과의 seed별 paired 차이.

Standard GCN은 별도 adjacency 전파이므로 αL̄/βM 분기 관측의 대상이 아니다.
원래 분류 성능은 모든 8조건에서 재현한다.
분모 1이나 norm match 개입에 원래 operator norm 상한을 그대로 주장하지 않는다.
이미 test를 본 뒤의 진단이므로 최고 개입값을 새 모델의 최종 성능으로 선택하지 않는다.
