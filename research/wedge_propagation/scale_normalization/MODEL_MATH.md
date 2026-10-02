# Experiment 3.1 수식과 실제 계산

## 1. 실제 차분과 C 생성기 입력을 구분한다

그래프 G의 물리 엣지 발생행렬을 B라고 하자. 각 scalar 실현 r에 대해

\[
\sigma_{G,r}
=\sqrt{\frac{1}{|E_G|}\sum_{e\in E_G}(BX_r)_e^2}.
\]

엣지가 없거나 모든 차분이 0이면 σ=1을 사용한다. 양수인 σ에는 epsilon을 더하지 않는다.
다른 그래프·scalar 실현·모델 seed의 값으로 σ를 계산하지 않는다.

중심 j를 공유하는 경로 i–j–k에는

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,
\quad g_{1,p,r}=x_{j,r}-x_{i,r},
\quad g_{2,p,r}=x_{k,r}-x_{j,r},
\]

\[
(AX_r)_p=g_{2,p,r}-g_{1,p,r}
\]

를 사용한다. normalized 모델은 C를 만들 때만

\[
\tilde g_{1,p,r}=g_{1,p,r}/\sigma_{G,r},
\qquad \tilde g_{2,p,r}=g_{2,p,r}/\sigma_{G,r}
\]

로 바꾼다. 실제 메시지에 들어가는 AX는 원래 입력의 차분이다.

## 2. 같은 공유 생성기를 정규화 입력으로 학습한다

\[
z_{p,r}=
[|\tilde g_1|+|\tilde g_2|,
\tilde g_1\tilde g_2,
|\tilde g_2-\tilde g_1|,
(|\tilde g_1|-|\tilde g_2|)^2],
\quad s_{p,r}=\mathrm{MLP}_\theta(z_{p,r}),
\]

\[
c_{\theta,p,r}(X)=
\frac{\exp(\tau\tanh s_{p,r})}
{P_G^{-1}\sum_{q\in G}\exp(\tau\tanh s_{q,r})},
\quad
\hat M_r(X)=\beta A^\top C_{2,\theta,r}(X)AX_r.
\]

τ=1이며 C의 각 경로 가중치는 양수이고 그래프·scalar 실현별 평균은 1이다.
경로가 없는 그래프의 메시지는 0이고 경로 가중치 진단은 undefined다.
에너지 표현은 \(X_r^\top A^\top C_{2,r}AX_r\)지만, 에너지를 별도 학습 branch에 넣는 것은 아니다.
C 값을 고정하면 이 행렬은 양의 준정부호이며 이차형식을 정의한다.
전체 모델에서는 C도 X에 따라 바뀌므로 X에서 메시지로 가는 함수는 일반적으로 비선형이다.
실제 메시지를 C(X)의 미분항까지 포함한 전체 에너지 gradient라고 주장하지 않는다.

raw는 원래 g₁,g₂를 입력으로 사용한 Experiment 2 모델이다. 파라미터를 다시 학습하지 않는다.
normalized는 같은 원본 train 데이터·초기화 seed·hidden 64·500 epoch·physical graph batch 240으로
새로 학습한다. validation만으로 seed별 checkpoint를 선택한 뒤 모든 평가에서 고정한다.
정규화가 만드는 효과와 파라미터 변화가 함께 있는 재학습 비교이며, 동일 checkpoint의 입력만 바꾼 검사는 아니다.

## 3. 양의 배율에서 보장되는 성질

α>0이고 σ>0이면

\[
\sigma(\alpha X)=\alpha\sigma(X),
\quad \tilde g(\alpha X)=\tilde g(X),
\]

\[
C_{2,\theta}(\alpha X)=C_{2,\theta}(X),
\qquad \hat M(\alpha X)=\alpha\hat M(X).
\]

모든 차분이 0인 경우 정규화 입력도 0이므로 같은 성질이 성립한다.
이는 이상적인 산술에서의 구조적 성질이다. 실제 부동소수점 계산의 오차는 따로 측정한다.
이 식만으로 목표 메시지를 정확하게 맞추거나 C*를 회수한다는 결론은 나오지 않는다.

`random_pair`는 원래 저장된 서로 다른 물리 엣지 pair와 부호·계수를 사용한다.
Gate 차분을 나눌 때 σ는 해당 pair들의 분포가 아니라 같은 그래프의 전체 물리 엣지에서 구한다.
실제 random-pair 메시지 \(\beta A_r^\top C_r(X)A_rX\)는 정규화하지 않는다.

## 4. Teacher와 학습 목표는 유지한다

원본 teacher는 raw 차분으로

\[
r_p^*(X)=\frac{g_1g_2}{|g_1||g_2|+\epsilon}
+\frac{|g_2-g_1|}{|g_1|+|g_2|+\epsilon},
\quad
c_p^*(X)=\frac{\exp(\tanh r_p^*)}
{P_G^{-1}\sum_{q\in G}\exp(\tanh r_q^*)},
\]

\[
M^*(X)=A^\top\operatorname{diag}(c^*(X))AX,
\qquad\epsilon=10^{-8}
\]

를 만든다. 같은 그래프와 독립적인 새 특징을 배율 0.25/0.5/1/2/4로 바꿀 때 teacher와
target을 매번 다시 계산한다. Teacher의 epsilon 때문에 완전한 스케일 불변성을 가정하지 않는다.

각 target은 LX, L²X, M*(X)다. 학습은 원래의 그래프·scalar 실현별 정규화 메시지 제곱오차를 사용한다.
C*를 직접 맞추는 보조 loss와 배율 augmentation은 없다. first/polynomial/fixed의 원래 scalar fit은
재추정하지 않고 동일한 frozen 대조군으로 두 variant에 표시한다.

## 5. 정확도·스케일 반응·가중치 진단을 따로 읽는다

메시지 상대오차는

\[
\mathrm{RelErr}_{G,r}
=\frac{\|\hat M_r-M_r^*\|_2}{\|M_r^*\|_2+10^{-8}}
\]

이며 absolute RMSE도 저장한다. 스케일 진단은 같은 fresh X의 배율 1을 기준으로

\[
\Delta_C(\alpha)=
\frac{\|c_\theta(\alpha X)-c_\theta(X)\|_2}{\|c_\theta(X)\|_2+10^{-8}},
\quad
\Delta_M(\alpha)=
\frac{\|\hat M(\alpha X)/\alpha-\hat M(X)\|_2}{\|\hat M(X)\|_2+10^{-8}}
\]

를 측정한다. Teacher도 같은 검사를 수행한다. 스케일 진단이 작은 것과 예측 오차가 작은 것은 별개다.

C 오차·상관은 원본과 fresh 배율 1을 분리해 보고한다. 평균 1을 맞춰도 노드 메시지만으로 C를
유일하게 식별할 수 없다. 상관이 undefined일 때 값을 0으로 대체하지 않는다.
identity/mean/shuffle/other graph pattern/correspondence randomization 개입은 각 checkpoint와 입력을
고정한 채 수행한다. 개입이 오차를 줄인 경우도 보고하며 identity와 mean을 독립적인 두 효과로 세지 않는다.

특징 실현을 먼저 그래프 안에서 평균하고, 모든 그래프를 같은 비중으로 모은 뒤 seed 통계를 구한다.
raw↔normalized와 original↔fresh는 동일 graph ID·seed를 대응시켜 오차 차이를 먼저 계산한다.
train 원본·train 새 특징·ID·각 OOD를 합치지 않는다. 같은 fresh X의 여러 배율도 독립 표본으로 세지 않는다.
L/first와 L²/polynomial의 양성 대조가 맞아도 일반적인 spectral 함수의 한계나 실제 GNN 분류 우위를
입증한 것은 아니다. 이번 결과는 지정한 합성 메시지 규칙의 재현·일반화 범위에 한정된다.
