# Experiment 3의 실제 계산과 해석

## 모델은 무엇을 하는가

중심 j의 두 이웃 i,k를 잇는 경로에

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,
\quad g_1=x_j-x_i,\quad g_2=x_k-x_j,
\quad (AX)_p=g_2-g_1
\]

을 사용한다. 고정 연산은 Q=A.T A, 학습 연산은 A.T C2(X) A다.
모든 경로에서 차분을 만들고, C2로 가중한 뒤 노드로 다시 집계한다.
Student는 두 차분의 관계에서 같은 공유 MLP로 경로 가중치를 만든다.

\[
z_p=[|g_1|+|g_2|,\ g_1g_2,\ |g_2-g_1|,\ (|g_1|-|g_2|)^2],
\quad s_p=\mathrm{MLP}_\theta(z_p),
\]

\[
c_{\theta,p}(X)=
\frac{\exp(\tau\tanh s_p)}{P^{-1}\sum_{q\in G}\exp(\tau\tanh s_q)},
\qquad \hat M(X)=\beta A^\top C_{2,\theta}(X)AX.
\]

한 입력에서 C2를 고정하면 A.T C2 A는 PSD이며 에너지는
X.T A.T C2 A X = sum_p c_p(AX)_p²다. C2가 입력에 의존하므로 전체 메시지 함수는 비선형이다.
에너지 값 자체를 학습 loss나 추가 분류 branch로 넣는 구현은 아니다.

이번 실행에서는 Experiment 2에서 선택한 θ와 β를 고정한다.
나머지 baseline의 u,v,β와 random-pair 모델도 고정한다.
원래 학습식·선택 규칙·개입 정의는 [Experiment 2 수식](../learned/MODEL_MATH.md)에 있다.

## 원본 재현과 새로운 입력

G의 연결·경로·random pair는 원본 NPZ에서 읽는다. 원본 X로 selected 모델의 출력 지표를
다시 계산해 원본 CSV의 각 graph/target/condition/seed에 맞춰 검증한다.

새로운 독립 표준정규 입력 \(\tilde X_G\)를 만든 뒤

\[
X_{G,\alpha}=\alpha\tilde X_G,
\qquad \alpha\in\{0.25,0.5,1,2,4\}
\]

를 평가한다. 그래프 ID와 master seed 20261003으로 독립 feature seed를 만들며,
원본 feature seed와 값의 중복을 거부한다. Worker 수가 달라도 동일한 특징을 생성한다.
배율마다 같은 \(\tilde X_G\)를 사용한다.

Target은 LX, L²X, 경로 teacher 메시지다. 경로 teacher의 실제 규칙은

\[
r_p^*(X)=\frac{g_1g_2}{|g_1||g_2|+\epsilon}
+\frac{|g_2-g_1|}{|g_1|+|g_2|+\epsilon},
\quad
c_p^*(X)=\frac{\exp(\tanh r_p^*)}
{P^{-1}\sum_{q\in G}\exp(\tanh r_q^*)},
\]

\[
M^*(X)=A^\top\operatorname{diag}(c^*(X))AX,
\qquad \epsilon=10^{-8}.
\]

Teacher의 비율은 epsilon=0인 경우 양의 배율에 불변이다.
실제 epsilon이 0이 아니므로 완전한 불변성을 가정하지 않고 모든 배율에서 teacher와
목표를 다시 계산한다. Student z에는 일차·이차 scale 항이 함께 있어 자동적인 배율 불변성이 없다.
이 차이를 새 모델 구조나 재학습으로 보정하지 않고 먼저 측정한다.

## 지표 1: 목표 메시지를 얼마나 맞추는가

그래프 G, scalar 실현 r마다

\[
\mathrm{RelErr}_{G,r}(\alpha)=
\frac{\|\hat M(X_{G,\alpha})-M^*(X_{G,\alpha})\|_2}
{\|M^*(X_{G,\alpha})\|_2+\epsilon_{metric}},
\quad\epsilon_{metric}=10^{-8}.
\]

LX/L²X target에서는 해당 메시지가 M*다. Absolute RMSE도 저장한다.
Graph 내 16개 실현을 평균하고, split의 graph macro를 계산한 뒤
독립 초기화 seed 5개의 평균·표본 표준편차를 보고한다.
결정론적 scalar baseline은 한 번의 fitting 결과이며 독립 seed 반복으로 세지 않는다.

원본↔fresh 비교는 같은 graph ID·모델 seed를 대응시킨 오차 차이다.
Fresh amplitude1이 특징 교체의 효과, 같은 fresh X의 다른 배율이 입력 크기 변화의 효과다.
Train 원본과 train fresh, unseen graph의 각 split을 분리해서 읽는다.

## 지표 2: 입력 크기 변화에 모델이 어떻게 반응하는가

기준 입력은 같은 fresh X의 amplitude1이다. 실현별로

\[
\Delta_C(\alpha)=
\frac{\|c_\theta(\alpha\tilde X)-c_\theta(\tilde X)\|_2}
{\|c_\theta(\tilde X)\|_2+\epsilon_{metric}},
\]

\[
\Delta_M(\alpha)=
\frac{\|\hat M(\alpha\tilde X)/\alpha-\hat M(\tilde X)\|_2}
{\|\hat M(\tilde X)\|_2+\epsilon_{metric}}.
\]

Teacher에도 같은 비교를 적용한다. LX/L²X의 teacher 메시지는 해당 선형 target을 사용하고,
path target에서만 teacher C2의 변화를 보고한다.
Random-pair의 C 변화는 해당 모델의 pair 위에서 비교하며, 실제 경로의 C*와 동일하게 해석하지 않는다.
Scalar baseline에는 학습한 C가 없으므로 C 변화는 undefined다. 경로가 없는 그래프도 같다.
정의되지 않은 값을 0으로 채우지 않는다. C의 평균이 1이어도 분포는 달라질 수 있다.

\(\Delta_M\)은 목표 메시지에 대한 오차와 별도의 진단이다. 잘못된 메시지를 만들더라도
입력에 비례하는 모델이면 작을 수 있다. \(\Delta_C\)가 작다는 사실만으로 teacher의 C를
회수했다고 판단할 수 없다. 두 진단을 RelErr와 함께 읽는다.

## 원본의 C2·개입·teacher 진단은 어떻게 사용하는가

원본 CSV에서 C2 오차·상관과 identity/mean/shuffle/other graph pattern/
correspondence randomization 개입 결과를 읽어 다시 집계한다.
Teacher 행렬 T=A.T diag(c*) A의 span{L,L²,Q}에 대한 Frobenius 잔차와
입력에 따른 연산 변화도 원본에서 읽는다.
새로운 개입 학습을 수행하거나 원본 CSV에 없는 진단값을 만들어 보고하지 않는다.

목표 노드 메시지만으로 개별 C2가 유일하게 결정되는 것은 아니다.
메시지 회수, C2 진단, 고정한 모델의 개입, 각 일반화 split의 결과를 함께 해석한다.
이 실험은 합성 teacher가 정한 규칙을 재현하는지 검사한다. 일반적인 GNN 성능이나 연구의
신규성을 이 결과만으로 판정하지 않는다. 실제 분류기와 loss의 연결은 다음 단계의 계약에서 정의한다.
