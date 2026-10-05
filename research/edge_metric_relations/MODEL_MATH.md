# 실제 코드의 수식

## 1. 상태와 발생행렬

물리 그래프의 노드는 n개, 엣지는 m개다. 방향을 임의로 고정한 발생행렬 B∈R^(m×n)의 엣지 행은 한 끝점 −1, 다른 끝점 +1이다. 노드 특징 X∈R^(n×f)에 대해 BX는 엣지 끝점의 특징 차이다.

모든 중심 v의 induced 1홉 local을 유지한다. 엣지 e의 local 등장 횟수를 r_e라 하고, 등장 행렬 U의 각 행을 해당 물리 엣지에 대한 1/sqrt(r_e)로 정의한다.

    Bcal = U B,       UᵀU = I_m,       BcalᵀBcal = BᵀB.

이는 **여러 local의 엣지 행으로 물리 노드 상태를 표현하는 factorization**이다. 현재 모델이 별도 node-copy 상태를 계속 진화시키는 것은 아니다. 중복 보정은 같은 엣지를 여러 번 센 것만으로 에너지가 증가하는 문제를 막는다.

## 2. 같은 local의 이차항과 local 사이의 쌍선형항

등장 i=(v,e)의 기준 가중치 c_i는 unit이면 1, local_degree이면 2/(d_v(u)+d_v(w))다. d_v는 해당 induced local 내부 degree다. 학습 배율 a_e는 같은 물리 엣지의 모든 등장에 공유한다. D_ii=c_i a_e다.

서로 인접한 중심의 local에서 서로 다른 물리 엣지 e/f가 끝점 h를 공유할 때 등장 쌍 i/j를 만든다. 각 undirected 등장 쌍은 한 번 세며 같은 물리 엣지끼리는 결합하지 않는다. 모든 eligible 쌍을 유지한다.

eligible pair graph의 degree를 δ_i, 공유 끝점에서의 발생 부호를 σ_i, orientation-invariant 계수를 t_ij∈[−1,1]이라 하면

    K_ij = σ_i σ_j t_ij / sqrt(δ_i δ_j),   K_ii=0,
    Ccal = sqrt(D) (I + ρ K) sqrt(D),      ρ=1/2,
    Qθ(X) = Bcalᵀ Ccalθ(X) Bcal.

Y=Bcal X라 두면 실제 에너지는

    Etotal(X) = 1/2 tr(XᵀQθ(X)X)
              = 1/2 Σ_i D_ii ||Y_i||²
                + ρ Σ_{i<j eligible} sqrt(D_ii D_jj) K_ij <Y_i,Y_j>.

첫 합은 같은 local의 이차 에너지다. 둘째 합은 다른 local의 엣지 차이 사이 쌍선형 관계다. 둘째 합은 음수가 될 수 있다. 전체 에너지는 아래 양의 계량 조건으로 음수가 되지 않는다. **국소 scalar E와 scalar J를 임의로 더해 전체 그래프 에너지라고 해석하는 방식과 다르다.**

## 3. 학습 입력과 네 수정

대각 생성기는 엣지 양 끝의 노드 특징, pair 생성기는 공유 끝점에서 바깥 끝점으로 향하는 두 특징 차이를 사용한다. 각각 a,b에 대해

    s = sqrt((mean(a²)+mean(b²))/2 + ε²),  ε=1e−4,
    a'=a/s, b'=b/s,
    dynamic = [mean(a'²)+mean(b'²),
               |mean(a'²)−mean(b'²)|,
               mean(a'b'), mean((a'−b')²)].

pair 입력에는 local degree의 log1p 합/차, local size의 log1p 합, 공유 노드 비율을 추가해 8차원이 된다. 대각 입력은 위 4차원이다. MLP는 4→64→1 또는 8→64→1, SiLU, bias 포함이다. hidden은 Xavier, 출력 weight/bias는 0으로 초기화한다.

    a_e = exp(log(2) tanh(fξ(edge_input))) ∈ (1/2,2),
    t_ij = tanh(φθ(pair_input)).

두 분류 층의 생성기는 서로 다른 파라미터를 쓴다. 각 층의 규칙은 모든 엣지·그래프에 공유된다. ε는 영점에서의 불연속을 제거한다. ε 때문에 정확한 scale invariance를 주장하지 않는다. 절댓값을 사용하는 기존 특징은 그 인수가 0인 경계에서 비매끄러울 수 있다.

DA 대조는 동일 pair 입력/MLP의 계수를 **등장 행마다 합산한 뒤**, 물리 엣지의 r_e개 등장에 대해 평균한다.

    v_i = Σ_{j eligible} t_ij / sqrt(δ_i δ_j),
    b_e = fξ(edge_input) + (1/r_e) Σ_{i:edge(i)=e} v_i,
    a_e^DA = exp(log(2) tanh(b_e)),   K^DA=0.

eligible 쌍 개수로 한 번 더 나누지 않는다. 빈 행의 합은 0이며 그 행도 등장 평균에 포함한다. DA/F2는 입력과 활성 파라미터 수가 맞지만 함수 집합이나 표현력이 같다는 보장은 없다.

## 4. 합성 모델 B의 출력과 목표

B는 projection·분류기 없이 raw 메시지

    output_B = Qθ(X) X

를 학습한다. 16개의 독립 scalar 실현은 16차원 특징으로 RMS 평균하지 않고 독립 실현 축으로 유지한다. 목표는 같은 raw 기준 L_d X, L_d²X, 그리고 고정된 analytic pair 규칙의 Q_teacher(X)X다. Teacher3의 규칙은 t_ij=tanh(2 dynamic[2]+0.5(2 overlap_ratio−1)), a_e=1이다. 이는 학생과 같은 family를 사용하는 명시적인 positive control이다.

동일 raw L_d와 L_d²의 정확 기준선을 별도 제공한다. 다른 정규화의 다항식이 같은 teacher를 정확히 재현한다고 주장하지 않는다. train 자료로 fit한 계수는 validation/held-out에 고정한다.

**대각 제곱 목표의 크기 한계**: 같은 recipe에서 학생은 Qθ≼3Q0로 제한된다. L_d² 목표에서 큰 고유값 λ>3인 성분은 이 학생 family가 정확히 표현할 수 없다. 그 목표를 임의로 정규화하지 않고 크기·구조 스트레스로 기록한다. 높은 제곱 목표 오차만으로 비대각 결합이 쓸모없다고 결론내리지 않는다. 동일 L_d² oracle은 이 제한을 갖지 않는 정확 대조다.

## 5. 분류 모델 C의 두 층

N0=diag(1+degree(Q0))^(−1/2)이며 Q0는 해당 recipe의 기준 대각 연산자다. 각 층은

    Z_l = Dropout(H_l) W_l,
    U_l = Z_l − (1/3) N0 Qθ_l(Z_l) N0 Z_l,
    H_(l+1) = ReLU(U_l)  (첫 층), logits=U_l (둘째 층).

생성기의 입력은 Z_l, 연산자가 적용되는 값은 N0 Z_l다. CE는 생성기와 RMS 경로까지 미분한다. 위 식은 **특징에 따라 바뀌는 에너지의 전체 gradient descent**를 구현한 것이 아니다. Q를 고정하면 그 이차 에너지의 gradient는 QX이지만, Qθ(X)가 X에 의존하면 에너지의 전체 미분에는 Q의 미분항도 있다.

## 6. 성립하는 보장과 성립하지 않는 주장

대칭 K는 |t|≤1이고 topology degree로 정규화했으므로 ||K||₂≤1이다. D>0이고 ρ=1/2이므로 Ccal은 양의 정부호다. 따라서 Q는 양의 준정부호이며 raw 좌표에서 ker Q=ker B다. 방향 반전은 발생 부호와 K의 부호가 함께 변해 Q를 유지한다.

    (1/4) Q0 ≼ Q ≼ 3 Q0,
    λmax(N0 Q N0) ≤ 6.

그러므로 **고정된 계수**에서 I−N0QN0/3의 스펙트럼은 [−1,1]에 있다. 이는 전체 adaptive 모델의 Jacobian, 학습 안정성, 분류 우월성 보장이 아니다. raw Q는 상수 성분을 소거하지만 정규화한 N0 Q N0의 영공간은 N0^(−1)ker B다. residual 경로와 raw 메시지의 영공간도 구분해야 한다.

고정 Q의 value action은 엣지 쌍의 공통 끝점 때문에 최대 2홉 범위다. 생성기 입력과 반복 층을 포함한 adaptive 의존 범위를 이 보장으로 대신 설명하지 않는다.

이 결합은 임의 엣지 메시지의 cycle 성분을 원래 노드 집계에서 완전히 복원한다는 보장이 없다. 양의 정부호인 알려진 C에서 q=CBX, 전체 divergence d=Bᵀq를 관측하는 제한된 raw 경우는 q=CB(BᵀCB)†d로 복원 가능하다. 부분 노드 관측·local 분해·압축·비선형·sampling은 각각 별도 조건이며 A에서 나눠 검사한다.

A의 잡음 복원은 min_x ||OFx−y||²를 푼다. 작은 합성 그래프는 tolerance를 명시한 SVD pseudoinverse, 큰 citation은 zero-start primal CG로 (AᵀA)x=Aᵀy, A=OF/scale를 푼다. Normal-equation 잔차는 계산 수렴 검사이고 관측 잔차는 잡음·관측의 불일치를 포함한다. 수렴하지 않은 finite iterate는 성공 복원값으로 집계하지 않는다. 이는 선택된 원래 입력에 대한 수치 진단이며 citation의 exact rank나 원래 메시지의 무조건 복원을 증명하지 않는다.

코드 대응: `geometry.py`, `gates.py::_features/_diagonal/apply`, `operators.py::apply_metric`, `classification/model.py::_metric`.
