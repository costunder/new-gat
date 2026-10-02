# 실제 Experiment 2 모델의 작동 방식

## 그래프와 경로 연산

단순 무방향 그래프의 발생행렬 B는 엣지별 노드 차분을 만든다. L=B.T B다.
중심 j의 서로 다른 이웃 {i,k}마다 한 경로를 만들고 다음 행을 사용한다.

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,\qquad
Q=A^\top A,\qquad C_2(X)=\operatorname{diag}(c_p(X)).
\]

AX가 경로별 이차 차분을 만들고, C2가 이를 가중하며, A.T가 노드로 다시 집계한다.
각 입력에서 C2를 고정하면 A.T C2 A는 대칭 양의 준정부호 행렬이다.
그 입력의 에너지는 X.T A.T C2 A X = sum_p c_p (AX)_p²로 표현된다.
전체 메시지 M(X)는 C2가 X에 의존하므로 비선형이다.

## 두 엣지에서 C2 생성

\[
g_1=x_j-x_i,\quad g_2=x_k-x_j,\quad q_p=(AX)_p=g_2-g_1.
\]

이번 입력은 scalar다. 16개 특징 열은 독립적인 scalar 실현을 병렬로 처리하는 축이다.
일반 vector cosine 법칙으로 확장한 모델을 이미 구현했다고 해석하지 않는다.

\[
r_p^*=\theta_1\frac{g_1g_2}{|g_1||g_2|+\epsilon}
+\theta_2\frac{|g_2-g_1|}{|g_1|+|g_2|+\epsilon},\qquad
\tilde c_p^*=\exp(\tau\tanh r_p^*).
\]

\[
c_p^*=\frac{\tilde c_p^*}{P^{-1}\sum_{q\in G}\tilde c_q^*},
\qquad M^*=A^\top C_2^*(X)AX.
\]

Teacher는 theta1=theta2=tau=1, epsilon=1e-8이다.
정규화는 해당 그래프의 모든 경로에서 계산한다.
P=0이면 이차 연산의 메시지는 정확히 0이며 경로 가중치 진단은 정의되지 않는다.

Student의 입력과 출력은 다음과 같다.

\[
z_p=[|g_1|+|g_2|,\ g_1g_2,\ |g_2-g_1|,\ (|g_1|-|g_2|)^2],
\quad s_p=\operatorname{MLP}_\theta(z_p),
\quad c_p=\frac{\exp(\tau\tanh s_p)}{P^{-1}\sum_{q\in G}\exp(\tau\tanh s_q)}.
\]

경로를 뒤집으면 (g1,g2)→(−g2,−g1)이므로 네 통계가 모두 동일하다.
Score 생성기는 공유 local 함수이고, 최종 C2에는 graph mean 정규화가 추가된다.
그래프 ID, node ID, degree, PE와 teacher C2는 MLP 입력에 넣지 않는다.

\[
\hat M=\beta A^\top C_{2,\theta}(X)AX.
\]

Beta는 자유롭게 학습하는 scalar다. C2의 평균은 1이라 전체 scale을 담당하는 beta와
경로별 상대 가중치를 구분할 수 있다. 이 gauge 고정이 개별 C2의 유일한 식별을 보장하지는 않는다.
가중치 label·weight loss·correlation loss는 학습에 제공하지 않는다.

## 학습 loss와 평가 지표

각 그래프 G와 독립 scalar 실현 r에 대해 학습 loss는

\[
\ell_{G,r}=\frac{N_G^{-1}\sum_i(\hat M_{i,r}-M^*_{i,r})^2}
{N_G^{-1}\sum_i(M^*_{i,r})^2+\epsilon_{loss}}.
\]

실현을 평균하고 graph를 평균한다. 다섯 seed 모델의 loss를 합해 backward한다.
각 seed의 파라미터가 분리되어 있으므로 각자의 graph macro objective gradient가 전달된다.
Adam 상태도 seed 축에서 독립이다. Seed 축 전체를 묶는 gradient clipping은 사용하지 않는다.

주 평가 지표는

\[
\mathrm{RelErr}_{G,r}=\frac{\|\hat M_{G,r}-M^*_{G,r}\|_2}
{\|M^*_{G,r}\|_2+\epsilon_{metric}}.
\]

Graph 내 실현을 평균한 뒤 graph macro, 마지막에 독립 초기화 seed 평균·표본 표준편차를 계산한다.
Zero target에도 epsilon을 그대로 사용하며 absolute RMSE를 함께 저장한다.
Correlation 분모가 0인 경우는 undefined로 남긴다.
True path 가중치 회수는 path target의 learned 조건에서만 보고한다.
Random-pair의 가중치는 true path와 대응이 다르므로 그대로 C2*와 비교하지 않는다.

## Random-pair와 고정 checkpoint 개입

정적 random control은 그래프의 서로 다른 물리 엣지 두 개를 unordered pair로 선택한다.
True path 수와 같은 P개 pair를 중복 없이 선택하고 부호를 manifest에 고정한다.
Traversal 계수 s1,s2로 g1=s1 B_eX, g2=s2 B_fX를 정의한다.
`ArX=g2−g1`이며 각 Ar 행의 norm을 sqrt(6)으로 맞춘다.
같은 MLP 입력식·양의 mean1 정규화·파라미터 수로 random-pair 모델을 별도로 학습한다.

Learned checkpoint의 beta와 모든 파라미터를 고정해 다음을 평가한다.

1. Identity: C2=1.
2. Mean: 실제 graph mean C2로 모든 path를 채운다. Mean1 규약에서 identity와 거의 같아야 한다.
3. Weight shuffle: 같은 A를 유지하고 C2의 path 대응만 바꾼다.
4. Other graph pattern: 같은 split·size의 다른 graph C2를 path 순서에 선형 보간하고 평균 1로 다시 맞춘다.
5. Correspondence randomization: true path에서 생성한 C2를 고정하고 A를 저장된 Ar로 바꾼다.

4번은 단순한 weight shuffle과 달리 가중치 분포도 바뀔 수 있다.
5번은 파라미터가 동일한 별도 random-pair 학습 모델과 다른 개입이다.
Random pair는 공유 중심 비율, node support, hop 거리와 edge 사용 빈도도 바꿀 수 있다.
그 통계를 저장하므로 악화를 연속성 하나의 인과 효과로 단정하지 않는다.

## Teacher가 만드는 입력 의존 연산의 검사

모든 graph와 특징 실현에서 T_r=A.T diag(c*_r) A를 직접 구성한다.
T_r을 span{L,L²,Q}에 최적으로 맞춘 상대 Frobenius 잔차를 계산한다.
첫 입력의 T_0와 다른 입력의 T_r 차이도 계산한다.
이는 teacher의 실제 입력 의존성과 세 가지 고정 basis 밖의 성분을 확인하는 진단이다.
학습 정확도나 임의 spectral 함수 전체와의 차이를 증명하는 지표가 아니다.

## 평가 이후 주장할 수 있는 범위

LX/L²X 대조가 작동하고, path target에서 learned 출력 오차가 낮으며 frozen 개입과 OOD 결과가
이를 뒷받침하는지 확인한다. 결과가 나오기 전에 성공을 확정하지 않는다.
이 단계는 설정한 synthetic law의 학습·재사용 가능성을 검사한다.
실제 분류 문제에서 유리한지는 다음 GNN 단계에서 별도로 검증한다.
