# 분류 모델: 수식·학습·고정 개입

Copy 공간의 정의와 nonzero 보장의 조건은 [부모 수식 문서](../MODEL_MATH.md)를 따른다.
여기서는 실제 2층 분류기와 여섯 조건의 차이를 설명한다.

## 1. 고정 graph 연산

모든 중심 v에 S_v={v}∪N(v)의 유도 로컬 그래프를 만들고 B_v를 발생행렬로 둔다.

\[
A=\operatorname{blockdiag}_v(B_v^\top C_vB_v),\qquad K=J^\top J.
\]

C_v의 두 조건은 unit의 1과 local_degree의 2/(d_v(a)+d_v(b))다.
J는 인접 중심 {v,u}의 모든 shared physical-node copy (v,a)/(u,a)를 한 번씩 연결한다.
방향 중복은 canonical 중심 v<u로 제거한다. A/K/R/M은 입력 특징과 학습 파라미터에 의존하지 않는다.

\[
(RH)_{(v,a)}=H_a,\quad D=R^\top R,\quad M=D^{-1}R^\top,
\quad KR=0,\quad MK=0.
\]

그래프마다 η=.5/max weighted intra degree, γ=.5/max cross degree를 고정한다.
최대 degree가 0인 연산의 step은 0이다.
같은 η/γ와 앞뒤 같은 A를 모든 variant에 쓴다.

\[
T_\rho=M(I-\eta A)(I-\gamma\rho K)(I-\eta A)R,
\]
\[
T_0=M(I-\eta A)^2R,\qquad
\boxed{T_\rho-T_0=-\eta^2\gamma\rho\,MAKAR}.
\]

Copy joint energy는 ½tr(Yᵀ(A+λK)Y)다. Cross는 첫 내부 계산으로 생긴 문맥 차이를 재분배하고,
둘째 내부 계산이 이를 최종 물리 노드에서 관측되는 차이로 바꾼다.
고정 Tρ는 하나의 전역 node 연산으로도 쓸 수 있으므로 copy만으로 표현력 우위를 주장하지 않는다.

## 2. 두 macro 층

입력 X∈R^(N×F_in)의 행은 물리 노드이고 열은 하나의 특징 벡터의 channel이다.
Channel을 독립적인 학습 반복이나 통계 표본으로 세지 않는다.

\[
Z_0=\operatorname{Dropout}_{.5}(X)W_0,\quad
H_1=\operatorname{ReLU}(T_\rho Z_0),
\]
\[
Z_1=\operatorname{Dropout}_{.5}(H_1)W_1,\quad
\operatorname{logits}=T_\rho Z_1.
\]

W₀의 shape는 F_in×64, W₁은 64×K다. Bias와 추가 출력 projection은 없다.
동일 seed의 초기 projection과 counter dropout stream을 조건 간 맞춘다.
전파는 2개 macro 층이며 각 층에서 내부 A를 두 번 적용한다. Off도 이 계산 깊이를 유지한다.

## 3. 고정/학습 gain

| variant | ρ | 학습 가능한 θ |
| --- | --- | --- |
| off | 0 | 없음 |
| fixed | 1 | 없음 |
| learned | sigmoid(θ) | seed당 1개, 두 macro 층 공유 |

θ 초기값은 0이고 초기 ρ=.5다. 공유한다는 뜻은 한 독립 모델의 두 층이 같은 θ를 쓴다는 것이다.
함께 계산하는 seed 모델들은 서로 다른 θ, projection, Adam 상태를 갖는다. 조건 간 θ를 공유하지 않는다.
C_v/J/개별 cross 엣지 가중치/η/γ는 고정이며, 특징에서 attention 점수를 생성하는 모델이 아니다.

Seed당 활성 파라미터 수는 F_in×64+64×K+1[learned]다.

| 데이터 | off/fixed | learned |
| --- | ---: | ---: |
| Cora | 92,160 | 92,161 |
| CiteSeer | 237,376 | 237,377 |
| PubMed | 32,192 | 32,193 |

## 4. Loss와 미분

\[
\mathcal L_{CE}=\frac1{|V_{train}|}\sum_{v\in V_{train}}
-\log\operatorname{softmax}(\operatorname{logits}_v)_{y_v}.
\]

전체 그래프를 전파하고 train node만 loss에 사용한다. Energy, 복원, teacher C 보조 loss는 없다.
Adam의 matrix weight decay는 projection에만 5e−4를 적용하고 θ에는 적용하지 않는다.
평가 CE는 L2 항을 제외한다.

θ derivative는 ρ(1−ρ)와 실제 cross correction을 통해 CE에 연결된다.
다만 ρ가 양수거나 gradient가 존재한다는 사실만으로 task 성능 개선을 보장하지 않는다.
매 epoch의 θ/gain/gradient/update, 선택된 모델의 실제 context·matched 출력 변화를 따로 기록한다.

A/K는 고정 대칭이다. Exact edge chunk의 backward는 같은 A/K 작용을 gradient에 적용한다.
Feature-sized edge activation을 모두 유지하지 않아도 모든 edge/link의 기여와 projection/θ gradient를 보존한다.
Macro activation checkpointing도 같은 전파를 다시 계산한다. 일부 node/edge/channel을 버리는 sampling이 아니다.

## 5. 선택된 checkpoint의 frozen 개입

Fixed와 learned 모델에서 gain0/gain1을 layer_0/layer_1/both에 각각 적용한다.
선택 층의 effective ρ만 0/1로 바꾸며 학습 파라미터를 편집하지 않는다.
첫 층 개입이면 바뀐 hidden으로 둘째 층 전체를 다시 계산한다.

Frozen forward 후 parameter와 buffer의 hash가 원래와 같아야 하고 optimizer 갱신은 0회다.
Fixed gain1은 예상 no-op control로 표시하고 coverage에 남긴다. Off에는 별도 개입을 만들지 않는다.
새로 off로 학습한 모델과 learned checkpoint의 gain0 예측을 동일한 대조로 해석하지 않는다.
앞 층 개입 효과와 뒤 층 개입 효과가 단순히 더해진다고 가정하지 않는다.

## 6. 실제 작용 지표

현재 층의 동일 투영 특징 Z에서

\[
\delta=\|T_\rho Z-T_0Z\|_F,\quad
r=\delta/\|T_0Z\|_F,\quad c=\|KY_1\|_F
\]

를 계산한다. 분모가 0이면 r은 undefined다. 에너지는 cross 전후와 intra 전후의 ½trace로 기록한다.
다른 variant/layer의 Z까지 동일하다고 가정하지 않는다.
이 지표는 실제 전파의 사용 정도이며 정보 복원률이나 분류 기여율이 아니다.

## 7. 주장 범위

대칭/PSD, 균등 merge, 같은 A, 양의 step 조건에서 KY₁≠0이면 fixed gain의 출력 차이는 0이 아니다.
Task 성능은 별도 재학습/평가로 확인한다. Citation public split은 고정 graph의 transductive 연구다.
선행 실험의 test를 본 후 설계한 이번 비교를 독립 새 graph 일반화나 완전히 새로운 test 검증이라고 주장하지 않는다.
