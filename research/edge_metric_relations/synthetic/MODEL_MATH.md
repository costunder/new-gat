# 실험 B 수식과 해석 범위

## 1. 입력과 학생

각 그래프의 물리 노드 특징은 scalar field `X∈R^(N×1)`이다. 그래프마다 독립 realization 16개를 생성한다. 구현의 입력 형상은 `[S,R,N,1]`이며, S는 독립 seed 모델, R은 독립 scalar 입력이다. gate의 RMS와 내적은 마지막 feature 축에서만 계산한다. R개의 입력을 하나의 벡터로 섞지 않는다.

모든 induced 1-hop local의 엣지 occurrence를 사용한다. 물리 발생행렬 B의 엣지 e가 r_e번 나타나면, occurrence 복제 행렬의 비영 원소를 `1/√r_e`로 둔다. 이 행렬을 U라고 하면

\[
\mathcal B=UB,\qquad U^\top U=I.
\]

unit recipe의 occurrence 기본 가중치는 1이다. local_degree recipe는 해당 induced local에서

\[
c^0_{v,e}=\frac{2}{d_v(i)+d_v(j)}.
\]

물리 엣지별 기본 가중치 `c̄⁰_e`는 occurrence 가중치의 평균이다. 따라서 고정 diagonal 연산자는

\[
L_d=\mathcal B^\top\operatorname{diag}(c^0)\mathcal B
=B^\top\operatorname{diag}(\bar c^0)B.
\]

학생은 공유 gate로 대각 배수 a와 서로 다른 물리 엣지 occurrence 사이의 K를 만들고

\[
D_\theta=\operatorname{diag}(c^0_{v,e}a_e),\qquad
\mathcal C_\theta=D_\theta^{1/2}(I+0.5K_\theta)D_\theta^{1/2},
\]

\[
Q_\theta(X)=\mathcal B^\top\mathcal C_\theta(X)\mathcal B,
\qquad \widehat Y=Q_\theta(X)X
\]

를 계산한다. 대각 MLP는 4→64→1, pair MLP는 8→64→1이며 SiLU를 사용한다. 마지막 층은 0으로 초기화한다. 대각 배수는 `exp(log(2) tanh(ξ))`, pair raw 계수는 `tanh(φ)`이다. geometry와 orientation 및 pair-degree normalization은 [공통 구현](../geometry.py), [gate](../gates.py)에 정의되어 있다. RMS epsilon은 1e-4로 고정한다.

| 조건 | 대각 배수 | off-diagonal pair |
| --- | --- | --- |
| D0 | 1 | 없음 |
| D1 | 학습 | 없음 |
| F0 | 1 | 고정 raw 계수 1 |
| F1 | 1 | 학습 |
| F2 | 학습 | 학습 |
| DA | ξ와 pair gate의 물리 엣지 요약으로 학습 | 별도 off-diagonal 적용 없음 |

DA는 F2와 같은 gate parameter 수를 사용한다. pair gate의 sign-free raw 계수에 pair normalization을 곱한 뒤 occurrence에 더하고 물리 엣지 occurrence 수로 평균한다. 이 값이 ξ에 추가된다. pair가 없는 occurrence도 평균 분모에 포함한다. DA와 F2의 함수 공간이 동일하다고 주장하지 않는다.

## 2. 세 고정 교사

교사 recipe는 unit/local_degree이고 학생 recipe와 독립해서 전부 비교한다.

\[
Y_1=L_dX,\qquad Y_2=L_d(L_dX).
\]

세 번째 교사는 pair feature의 정규화 내적 t와 local 집합 overlap o를 사용한다.

\[
t=\frac{\langle\delta_e/s,\delta_f/s\rangle}{F},\qquad
s^2=\frac{\|\delta_e\|^2+\|\delta_f\|^2}{2F}+10^{-8},
\]

\[
o=\frac{|S_v\cap S_u|}{\sqrt{|S_v||S_u|}},\qquad
k_{\rm raw}=\tanh\bigl(2t+0.5(2o-1)\bigr).
\]

여기서 δ는 shared physical endpoint에서 각 엣지의 다른 endpoint로 향하는 특징 차이다. k_raw에 공통 orientation sign과 pair-degree normalization을 적용하고, `a=1`, `ρ=0.5`로 고정한 Q를 만든다.

\[
Y_3=Q_{\rm analytic}(X)X.
\]

교사의 가중치나 메시지는 학생의 입력 특징으로 제공하지 않는다. 학습 loss는 출력 메시지의 오차다. float64 교사 메시지는 학습 전 생성해 별도 저장한다.

## 3. Loss와 학습 예산

각 seed에 대해 그래프와 realization을 동일하게 평균한다.

\[
\ell_s=\frac1{G_{\rm train}R}
\sum_g\sum_r
\frac{\|\widehat Y_{s,g,r}-Y_{g,r}\|^2}
{\max(\|Y_{g,r}\|^2,10^{-8})}.
\]

seed 모델들의 loss를 합해 backward한다. seed 수로 다시 나누면 독립 모델의 gradient 스케일이 바뀌므로 그렇게 하지 않는다. 물리 graph batch의 loss에 `batch 그래프 수 / 전체 train 그래프 수`를 곱해 누적하고, 전체 train 자료를 사용한 뒤 Adam을 한 번 갱신한다.

교사 recipe 2 × target 3 × 학생 recipe 2 × 학습 조건 4 × seed 5 = **240개 학습 run**이다. 각 500 epoch로 총 **120,000회 seed optimizer 갱신**이다. packed tensor로 묶인 optimizer 호출 수는 24,000회다. 고정 조건과 analytic least-squares 기준선에는 optimizer가 없다.

validation NMSE가 가장 낮은 상태를 seed별로 선택한다. 초기 epoch 0도 후보이며, 이후 10 epoch마다와 마지막 epoch에서 검사한다. 동률이면 먼저 관측한 epoch를 유지한다. 선택과 관계없이 500 epoch를 전부 수행한다. test split에서는 parameter를 갱신하지 않는다.

## 4. Train 자료만 사용하는 수식 기준선

교사와 **같은 Ld**를 사용해 다음 basis의 계수를 float64 weighted least squares로 적합한다.

1. `α LdX`.
2. `β Ld²X`.
3. `α₀X+α₁LdX+α₂Ld²X`.
4. `α Q_wedge X` — 기존 고정 wedge 연산자.

회귀의 row weight는 각 graph·realization에서 `1/max(||Y||²,epsilon)`이므로 gate 학습과 같은 NMSE 목적을 푼다. rank가 부족하면 minimum-norm Moore-Penrose 해를 사용한다. 계수는 train 자료만으로 정하고 held-out 자료에서는 고정한다. 분류용 normalized polynomial을 raw Ld²의 정확한 oracle로 취급하지 않는다.

## 5. 결과로 말할 수 있는 범위

- t1과 t2의 해당 SAME-teacher oracle은 target와 구현의 정확성을 확인하는 양성 대조다.
- bounded 학생은 `0.25 Q0 ⪯ Qθ ⪯ 3 Q0`이다. 같은 recipe에서 Ld²은 고유값 λ>3인 방향에 대해 이 범위를 넘을 수 있다. 따라서 t2는 알려진 크기 제약을 포함한 스트레스 검사다. t2의 실패만으로 off-diagonal 관계가 쓸모없다고 결론낼 수 없다.
- t3는 고정한 비선형 규칙의 메시지 회복과 새로운 graph 구조에서의 재사용을 측정한다. 가중치의 유일한 식별이나 실제 데이터 분류 개선을 입증하지 않는다.
- 그래프마다 16개 입력은 같은 그래프의 여러 realization이다. 독립 그래프 수를 16배로 보고하지 않는다.
- size OOD, family OOD 및 결합 OOD를 따로 보고한다. 정확한 labeled topology 중복을 배제하지만 graph isomorphism 차원의 disjointness는 보장하지 않는다.
- 여섯 사전 지정 NMSE contrast를 held-out split마다 비교한다. 총 360개 two-sided paired t 비교에 하나의 Holm family(α=0.05)를 사용한다. seed 5개의 개별 t 신뢰구간과 보정 p-value를 구분한다.
- zero target의 epsilon 처리와 개수를 실제 per-draw 표에 남긴다.

로컬의 dense 수식 검사, 실제 MSE 한 번의 갱신 및 DEBUG 학습은 FULL 성능 결과와 구분한다.
