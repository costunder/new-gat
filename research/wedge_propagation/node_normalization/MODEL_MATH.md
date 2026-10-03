# Experiment 4.2 모델 수식과 실제 작동 경로

## 1. 각 층이 계산하는 것

입력 특징을 투영한 `Z`에서 각 wedge의 C를 만든다.
일차 메시지와 이차 메시지를 계산하고, 학습한 α·β로 빼서 다음 층에 전달한다.
첫 층 뒤에는 ReLU, 두 번째 층 뒤에는 분류 logits가 나온다.

\[
Z^{(\ell)}=\operatorname{Dropout}(H^{(\ell)})W^{(\ell)},
\quad U^{(\ell)}=Z^{(\ell)}-\alpha_\ell\bar LZ^{(\ell)}-\beta_\ell M_\ell,
\]

\[
H^{(1)}=\operatorname{ReLU}(U^{(0)}),\qquad \text{logits}=U^{(1)}.
\]

두 층의 출력 channel은 각각 64와 클래스 수다. 투영 bias·추가 output projection은 없다.
서로 다른 seed의 모델은 `[seed,node,channel]` 차원으로 함께 계산하며 파라미터를 공유하지 않는다.

## 2. 그래프 연산자의 정의

`B`는 물리 엣지를 행으로 둔 부호 있는 발생행렬이다.
`A`는 중심 j의 서로 다른 이웃 i,k로 만든 모든 unordered wedge를 행으로 둔다.

\[
(BZ)_{uv}=Z_v-Z_u,\qquad (AZ)_{ijk}=Z_i-2Z_j+Z_k.
\]

\[
L=B^\top B,\quad Q=A^\top A,\quad
\bar L=\frac12S_dLS_d,\quad S_d=\operatorname{diag}(d)^{-1/2}.
\]

`C=diag(c)`는 wedge별 양의 scalar이고 모든 channel에 같은 값을 사용한다.
한 path의 계수가 `(1,−2,1)`이므로 대각은 다음과 같다.

\[
(D_C)_v=\sum_{p\ni v}c_p a_{pv}^2,\qquad
q_v=(D_I)_v=\operatorname{diag}(Q)_v.
\]

구현은 각 path 끝점에 c, 중심에 4c를 scatter하여 D_C를 계산한다.
고립 노드와 어떤 wedge에도 참여하지 않는 노드의 역제곱근은 0이다.

## 3. 같은 gate로 C 생성

각 층의 현재 투영 Z로부터 다음 두 물리 엣지 차이를 계산한다.

\[
g_1=(Z_j-Z_i)/\sigma,\qquad g_2=(Z_k-Z_j)/\sigma.
\]

\[
\phi_p=[|g_1|+|g_2|,\;g_1g_2,\;|g_2-g_1|,\;(|g_1|-|g_2|)^2].
\]

\[
s_p=\operatorname{MLP}_{4F\to64\to1}(\phi_p),\quad
\tilde c_p=e^{\tanh s_p},\quad
c_p=\tilde c_p/\operatorname{mean}_{\text{all paths}}\tilde c_p.
\]

MLP는 첫 linear에 bias와 ReLU, 두 번째 linear에는 bias가 없다.
Path 평균 1은 모든 C가 같은 값이라는 뜻이 아니다. C std·min·max를 별도로 기록한다.
Raw 조건은 σ=1이고, RMS 조건은 물리 엣지·모든 channel에서 다음 값을 계산한다.

\[
\sigma=\sqrt{\frac{\|BZ\|_F^2}{E F}}.
\]

물리 엣지가 없거나 차이 신호가 정확히 0이면 σ=1이라는 기존 규칙을 유지한다.
C의 입력은 정규화 이전의 Z이며, local 정규화로 gate 입력을 다시 만들지 않는다.

## 4. Global와 node의 차이

Global은 topology에서 계산한 S_Q를 사용하고 하나의 κ로 전체 분기를 나눈다.

\[
S_Q=\operatorname{diag}(q)^{-1/2},\quad
\kappa(C)=\max_{q_v>0}(D_C)_v/q_v,
\]

\[
\boxed{M_{\text{global}}=\frac{S_QA^\top C A S_Q Z}{3\kappa(C)}}.
\]

Node는 현재 C로 계산한 노드별 대각 D_C를 양쪽에 적용한다.

\[
S_C=D_C^{-1/2},\qquad
\boxed{M_{\text{node}}=\frac13S_CA^\top C A S_C Z}.
\]

이 차이는 κ를 제거한 global 연산자와도 다르다. Node의 양쪽 좌표는 현재 C에 따라 바뀐다.
학습 중 C·D_C·왼쪽 S_C·오른쪽 S_C 전체에 gradient가 전달된다.
`detach`는 평가용 기록에만 적용하고 학습 메시지 경로에는 적용하지 않는다.
C=I이면 D_C=diag(Q), κ=1이므로 두 연산자가 같고 fixed 대조군은 하나다.

두 조건 모두 α·β는 기존 방식으로 학습한다.

\[
t=\operatorname{sigmoid}(u),\quad r=\operatorname{sigmoid}(v),\quad
\alpha=t(1-r),\quad\beta=tr.
\]

초기값은 α=0.5, β=0.25이며 CE backward와 Adam update에 함께 연결되어 있다.

## 5. 이차형식과 spectral norm 상한

현재 C를 고정하면 node 연산자 T_C는 대칭 PSD이고 다음 에너지를 정의한다.

\[
\operatorname{tr}(Z^\top T_C Z)
=\frac13\|C^{1/2}AS_C Z\|_F^2,
\qquad T_C=\frac13S_CA^\top C A S_C.
\]

각 path의 세 항에 Cauchy–Schwarz를 적용하면

\[
(x_i-2x_j+x_k)^2\le3(x_i^2+4x_j^2+x_k^2),
\quad A^\top CA\preceq3D_C.
\]

따라서 `0 ≼ T_C ≼ I`이며 spectral norm 상한은 1이다.
Global도 D_C≼κD_I를 사용하여 같은 상한을 갖는다.
이 상한은 고정된 현재 C의 선형 연산자에 대한 성질이다.
`C=C(Z)`인 전체 함수의 Jacobian 상한이나 학습 성공을 보장하지 않는다.
마찬가지로 M은 C를 고정한 에너지의 선형 작용이며,
C와 S_C의 Z 의존성까지 미분한 전체 에너지 gradient와 동일하다고 주장하지 않는다.

## 6. 학습 loss와 선택

각 전체 그래프의 train mask에서 CE를 계산하고 500 epoch를 모두 학습한다.
정규화용 auxiliary loss·teacher·C target loss를 추가하지 않는다.
기존 weight 행렬의 L2 regularization 규칙을 유지하며 평가 CE에는 이 항을 포함하지 않는다.

Tuning seed 101·202·303과 learning rate 0.001·0.003·0.01로 평균 validation CE를 비교한다.
각 run의 checkpoint는 최소 validation CE, 최대 validation accuracy, 가장 이른 epoch 순으로 선택한다.
학습률을 잠근 뒤 final seed 11·23·37·53·71을 새로 학습한다.
모든 final 선택을 잠근 뒤 train·validation·test와 frozen C 진단을 평가한다.
210 runs·105,000 optimizer updates가 full 계약이다.

## 7. Frozen 진단의 정확한 의미

처리는 C=1, C 위치 shuffle, 각각의 norm match, 이차 분기 제거다.
각각 layer_0·layer_1·both를 평가하고 원래 모델의 α·β·파라미터를 고정한다.

Native C 교체는 현재 C로 D_C와 양쪽 S_C 또는 global κ를 다시 계산한다.
Norm match는 현재 층의 true C 참조 메시지 M_ref와 후보 M_alt에 대해 seed마다

\[
g=\frac{\|M_{\mathrm{ref}}\|_F}{\|M_{\mathrm{alt}}\|_F},\qquad
M_{\mathrm{matched}}=gM_{\mathrm{alt}}
\]

를 사용한다. β가 같으므로 실제 βM norm도 같다.
이는 전체 update와 일차 분기와의 상쇄, ReLU 이후 상태까지 같게 하지 않는다.
원래 Z가 아닌 **개입 후 해당 층에 도달한 현재 Z**에서 참조를 계산한다.
참조·후보 둘 다 0이면 g=1, 후보만 0이면 정확한 오류로 중단한다.
정의되지 않은 비율·cosine은 None과 false defined flag로 남긴다.
사후 norm matching의 g가 1을 넘을 수 있으므로 native spectral norm 상한을 그대로 주장하지 않는다.

Norm을 맞춘 C 배치가 분류에 유익한지, 분기를 제거하면 달라지는지,
재학습한 node/global·learned/fixed 모델 차이가 같은 seed에서 일관적인지 함께 읽는다.
Shuffle 10개는 seed 안에서 평균한다. 모든 seed·split·층 위치의 값을 보존하며 최적 개입을 선택하지 않는다.
