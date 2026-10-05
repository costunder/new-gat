# 로컬 이차 에너지와 엣지 간 쌍선형 결합: 실험 재설계안

**상태: 제안. 본 모델 구현·공식 데이터 학습·새 성능 평가는 하지 않았다.**
기준 자료: `GPT_ALL_RESEARCH_20261005.zip`. 현재 copy 모델을 고친 완료 버전이라고 부르지 않는다.
이 문서의 수학 참조 검사는 별도 작은 FP64 검사이며 연구용 FULL 결과가 아니다.

## 0. 먼저 바로잡을 정의

- B ∈ R^(m×n): 물리 엣지×노드 signed incidence.
- L0 = BᵀB: 무가중 라플라시안.
- Ld = BᵀCdB, Cd=diag(c_e)>0: 양의 대각 conductance 라플라시안.
- Q_C = BᵀC_edge B: 비대각 원소까지 허용한 엣지 공간 metric의 노드 연산자.
- F_l: 특징 채널 변환. C 또는 cross K와 다른 변수.

L0² = Bᵀ(BBᵀ)B이고 L0^k = Bᵀ(BBᵀ)^(k−1)B이다.
그러나 임의의 BᵀCdB가 L0²라는 뜻이 아니다. Ld² = Bᵀ[Cd BBᵀ Cd]B이다.
과거 설명에서 올바른 Ld=BᵀCdB 정의를 오류라고 한 것은 취소한다.
비대각 C_edge 학습은 이번의 **새 가설**이지 기존 diagonal C 코드의 설명이 아니다.

C_edge가 대칭 PSD이면 Q_C는 PSD지만, 일반적인 비음수 물리 엣지 가중 라플라시안일 필요는 없다.
C_edge가 비대칭이면 에너지는 대칭 부분만 보므로 에너지와 메시지의 일치를 주장하려면 별도 처리가 필요하다.
본 후보는 대칭 PSD에 한정한다. 모든 가능한 PSD C를 표현한다는 주장은 하지 않는다.

## 1. 연구 질문과 독립적인 판정

Q1. 정해진 관측 범위와 전파에서 q/E/J의 어떤 차이가 실제로 구별되지 않는가?
Q2. 같은 특징 접근 범위에서 비대각 엣지 결합이 대각 가중치 또는 고정 다항식보다 유용한 관계를 계산하는가?
Q3. 그 차이가 합성 법칙 재사용 및 실제 citation 분류의 이득으로 이어지는가?

연산자가 다름, 메시지가 바뀜, gradient가 전달됨, E/J 재현이 쉬움, CE 감소, accuracy 우위,
전역 복원 가능성, 신규성은 별도 결론이다. 결론을 먼저 정한 채 유리한 teacher만 만들지 않는다.

## 2. 정보 소실 주장의 이론적 경계

### 2.1 전체 관측

고정 양의 Cd와 전체 d=LdX를 알면 q=Cd B Ld^†d이다.
대칭 PSD Ld에 대해 ker(Ld^k)=ker(Ld)이다. 일반 SPD C_edge도
ker(BᵀC_edgeB)=ker(B)이다. 따라서 이 조건에서 대각→비대각으로 바꾸는 것만으로
전역 정확 식별성이 자동으로 증가하지 않는다.

### 2.2 기존 copy 모델의 추가 경계: 직접 유도

A=I−S, Dcopy=RᵀR, M=Dcopy^−1Rᵀ,
T0=MA²R, Tρ=MA(I−ρG)AR라고 하자. S는 대칭이고 G는 고정이다.

T0 z=0이면
zᵀDcopy T0 z=||ARz||²=0이므로 ARz=0, 따라서 Tρz=0이다.
즉 ker(T0) ⊆ ker(Tρ). 0≤ρ<1, 0≤G≤I이면 두 kernel은 같다.

이는 **동일한 고정 S/G, 선형 macro, 전체 노드 출력**에서의 명제다.
다른 parameter로 재학습한 모델, 부분 관측, macro 밖의 ReLU, 입력 의존 C를 이 명제로 단정하지 않는다.
기존 cross가 J-off의 전역 exact collision을 풀었다고 주장할 수는 없다.

### 2.3 관측 범위를 지정한 검사

고정된 propagation F와 관측 선택 O에서 y=OFx를 정의한다.
관측은 (a) 전체 노드, (b) 각 target node, (c) 각 target의 1홉 집합으로 구분한다.
전체 그래프 forward를 유지한 상태에서 읽는 범위를 바꾸는 것이며 데이터 샘플링이 아니다.

Aobs=OF, N0는 Aobs의 orthonormal null basis라 하자.
- 선형 목표 q=Tq x가 관측에서 결정되려면 Tq N0=0이어야 한다.
- 이차 목표 ψ(x)=xᵀKψx, Kψ 대칭이 모든 입력에서 결정되려면 KψN0=0이어야 한다.

KψN0≠0이면 n∈ker(Aobs), Kψn≠0를 택하고
x+=x0+εn, x−=x0−εn, x0=Kψn으로 두면 관측은 같고
ψ(x+)−ψ(x−)=4ε||Kψn||²≠0이다.
이것은 의도적으로 만든 구별 불가능성 사례이며 자연 데이터의 빈도를 증명하지 않는다.

반드시 역방향도 검사한다. 새 연산자의 관측이 구별하지 못하지만 baseline은 구별하는 사례가 있다면,
전체 정보 보존 증가가 아니라 구별하는 방향이 달라진 것이다.
입력 의존 연산에는 고정 행렬 nullspace 검사를 그대로 적용하지 않는다.
Jacobian은 해당 입력에서의 국소 민감도이며 전역 injectivity 증명이 아니다.

## 3. 새 후보 하나: 로컬 에너지 블록의 비대각 결합

### 3.1 그래프와 중복 처리

모든 induced 1-hop ego S_v={v}∪N(v)를 유지한다.
Pv는 물리 노드 특징의 local 선택, Bv는 해당 ego의 물리 엣지 incidence다.
물리 엣지 방향은 하나의 원래 ID convention을 공유한다.
r_e는 그 물리 엣지가 들어가는 모든 ego의 수다.

Bcal_v = diag(r_e^−1/2 : e∈E_v) Bv Pv,
Bcal = vertical_stack_v Bcal_v.

Bcal은 m_occ×n이고 node state를 복제해 별도로 진화시키는 current copy 모델과 다르다.
여기서는 같은 전역 X에서 local gradient를 만드는 중복 보정 factorization이다.
Uocc_(v,e),e=1/sqrt(r_e)라 하면 Bcal=Uocc B, UoccᵀUocc=I_m이다.

로컬 기준 conductance c0_(v,e)는 두 recipe를 별도로 유지한다.
- unit: 1.
- local_degree: 2/(deg_v(i)+deg_v(j)).

D0_occ=diag(c0_(v,e))이고 coupling이 꺼졌을 때
Q0=BcalᵀD0_occ Bcal = Bᵀdiag(cbar0_e)B,
cbar0_e = (1/r_e) Σ_{v:e∈E_v}c0_(v,e).

원래 raw E_v를 그대로 더한 에너지와 이 보정 에너지는 다르다.
raw E_v와 corrected E_v를 둘 다 기록한다. correction은 명시적인 새 설계 선택이다.
공유 엣지의 C를 로컬별로 보존하면서 physical reference는 평균 C가 된다는 것을 숨기지 않는다.

### 3.2 무엇을 cross pair로 볼 것인가

첫 계약은 기존 현재 자료처럼 **인접 중심의 로컬 쌍 전체**를 사용한다.
그 안에서 다음을 만족하는 모든 엣지 occurrence 쌍을 한 번씩 포함한다.
1. 서로 다른 로컬 v,u에 속한다.
2. 서로 다른 물리 엣지 e,f이다.
3. 같은 물리 노드 a를 endpoint로 공유한다.

같은 물리 엣지의 두 occurrence는 diagonal 성분의 재가중과 혼동되므로 cross 후보에서 제외한다.
동일 물리 엣지들의 기여는 위 cbar0 및 diagonal factor가 담당한다.
비인접 중심의 모든 overlap pair 또는 임의 멀리 떨어진 엣지 결합은 첫 버전에 포함하지 않는다.
따라서 이것은 일반 bilinear family의 한정된 local support다. 고정 상태에서 추가 node stencil은 최대 2홉이다.

전체 후보 목록과 수를 manifest에 보존한다. top-k, degree cap, edge dropout, sampling으로 줄이지 않는다.

### 3.3 방향과 순서에 불변인 shared pair generator

δe=−B[e,a](BX)_e=X_other(e,a)−X_a, δf도 같다.
계산용 엣지 방향을 뒤집어도 δe와 δf는 그대로다.

입력 특징은 다음 8개 scalar다. 이 scalar는 C 생성기의 입력일 뿐, 메시지 자체를 두 scalar로 압축하는 것이 아니다.
전체 F채널 gradient는 이후 연산에 그대로 남는다.

s²=(||δe||²+||δf||²)/(2F).
s>0이면 δe/s, δf/s를 아래에 사용하며 s=0이면 처음 네 특징을0으로 둔다.
1. 두 차이의 제곱 norm 합/F
2. 두 차이의 제곱 norm 차이 절댓값/F
3. 두 차이의 내적/F
4. 두 차이의 차이 제곱 norm/F
5. log(1+deg_v(a))+log(1+deg_u(a))
6. 두 log degree의 차이 절댓값
7. log(1+|S_v|)+log(1+|S_u|)
8. |S_v∩S_u|/sqrt(|S_v||S_u|)

모든 항은 (v,e)↔(u,f) 교환에 불변이다. 첫 네 특징에는 중복이 있으나 명시적 고정 입력이다.
실험 중 입력을 추가·삭제하지 않는다. Node/edge ID는 feature로 주지 않는다.

a_rs=−B[e,a]·(−B[f,a])·tanh(φθ(z_pair)), φ:8→64→1, SiLU.
a_sr=a_rs, 그 외0. 마지막 출력층은0 초기화한다.
고정 cross 대조에서는 tanh(φ)를 학습하지 않고1로 둔다.
이 상수는 topology/orientation에만 의존하는 비교 기준이다.

nr는 occurrence r의 정해진 eligible pair 수, nr=0일 때 정규화 분모만1로 둔다.
Npair=diag(max(1,nr)), Kθ=Npair^−1/2 Aθ Npair^−1/2.
후보 adjacency의 각 edge coefficient 절댓값≤1이므로 ||Kθ||₂≤1이다.
이 분모는 **topology-only**이며 특징 전체 평균이나 graph-global maximum을 사용하지 않는다.
재정향 부호 행렬 Qocc에 대해 K′=Qocc K Qocc를 검사한다.

### 3.4 Diagonal conductance 학습

고정 condition은 a_e=1.
학습 condition은 physical edge마다 공유 endpoint gate ξ를 적용해
a_e=exp(log(2)·tanh(fξ(z_edge))) ∈ (1/2,2).

z_edge는 endpoint 두 벡터의 norm 제곱 합, norm 제곱 차이 절댓값, 내적, 차이 norm 제곱의 네 scalar이며
공통 endpoint RMS로 먼저 정규화한다.0 RMS는0 입력으로 처리한다.
fξ:4→64→1, SiLU, 마지막 출력층0 초기화.

Dθ_occ=diag(c0_(v,e) a_e). 같은 physical edge의 multiplicative a_e는 모든 occurrence에서 공유한다.
새 graph에 edge-ID parameter를 추가하지 않는다. Feature normalization과 가중치 생성의 미분은 유지한다.

### 3.5 전체 metric, 내부 에너지, 교차 에너지

rho*=0.5를 이번 coupling family의 고정값으로 사전 지정한다. learned rho 실험을 동시에 추가하지 않는다.

Ccalθ = Dθ_occ^1/2 (I + rho* Kθ) Dθ_occ^1/2,
Qθ = Bcalᵀ Ccalθ Bcal,
Eθ(X) = 1/2 tr(Xᵀ Qθ X).

K의 같은-local block과 대각은0이다. 따라서
Eθ = 1/2 Σ_v tr(ghat_vᵀDθ_v ghat_v)
     + rho* Σ_{v<u} tr(ghat_vᵀ Dθ_v^1/2 K_vu Dθ_u^1/2 ghat_u),
ghat_v=Bcal_vX.

이것이 실제 하나의 에너지에서 나온 내부 이차항과 cross bilinear 항이다.
교차항을 따로 scalar로 만든 뒤 학습 vector를 곱해 분류기에 추가하지 않는다.
현재의 node-copy consensus G나 기존 (+1,−2,+1) wedge Q로 대체하지 않는다.

노드 공간에서는 Qθ=BᵀCeffθB, Ceffθ=UoccᵀCcalθUocc다.
Ceffθ의 diagonal은 cbar0_e a_e와 같고, 다른 물리 엣지 사이의 비대각이 새 결합이다.
서로 다른 Ccal이 같은 Ceff/Q를 만들 수 있으므로 Ccal 회수의 유일성은 주장하지 않는다.

Ccalθ≫0이며
(1−rho*) Qdiagθ ≤ Qθ ≤ (1+rho*)Qdiagθ,
Qdiagθ=BcalᵀDθ_occ Bcal.
따라서 일반 PSD 조건뿐 아니라 교차항과 내부항의 크기 관계도 명시된다.
이 bounded sparse family가 모든 C_edge 또는 모든 L0^k를 포함한다는 주장은 하지 않는다.

## 4. 실제 메시지와 분류 forward

각 층에서 Z=Dropout(H)F_l, F_l은 shared channel projection이다.
C/K 생성기는 **현재 Z**를 입력받는다. X나 projection 전 H와 혼용하지 않는다.

Q0의 physical weighted degree d0_i=(Q0)_ii,
Dtilde0=diag(1+d0_i), N0=Dtilde0^−1/2,
a_max=2, rho_max=.5, tau=1/[a_max(1+rho_max)]=1/3.

모든 proposed-family condition에서 동일한 topology-only N0와 tau를 사용한다.
단순히 off가 강하고 on이 약해지도록 각각 다른 graph maximum으로 재정규화하지 않는다.

U = Z − tau N0 Qθ(Z) N0 Z.
1층 후 ReLU, 2층은 logits. C 생성기 입력과 메시지에 적용되는 N0Z를 구분한다.
이 수식은 가중치 고정 시 1/2 tr(ZᵀN0QθN0Z)의 gradient step이다.
입력 의존 Qθ(Z)의 전체 에너지 gradient와는 다르다.
CE backward는 forward의 C/K/RMS 의존성을 모두 미분한다. detach하지 않는다.

안정성: Qθ≤a_max(1+rho_max) Q0,
N0Q0N0≤2I이므로 0≤tau N0QθN0≤2I.
가중치를 고정한 Pθ=I−tauN0QθN0는 Euclidean norm에서 비팽창이다.
Pθ가 PSD라는 보장은 하지 않는다. 입력 의존 Jacobian, F_l, dropout, ReLU를 포함한 전체 모델의 비팽창도 주장하지 않는다.

N0는 physical self-loop1을 사용하는 reference degree다. 현재 copy 모델의 Dcopy나 S_local을 가져온 것이 아니다.
이 family를 표준 GCN이라고 부르지 않는다. 표준 GCN은 별도 직접 대조한다.

## 5. 실험 A: 학습 전 구조와 관측가능성 감사

입력은 기존 고정 감사의198 synthetic+3 citation=201 graphs 전체, 합성3,168 scalar 입력,
두 C recipe와 원래 vector citation 특징이다. 데이터를 작게 대체하지 않는다.

기본 고정 operators: L0, Ld, L0², Ld², 현재 copy T0/Tρ, 제안 Qθ의 사전 지정 고정 fixture.
출력 비교와 실제 propagation 비교를 구분한다. L^k와 (I−etaL)^k를 같은 실험 행에 합치지 않는다.
반복 횟수 k=1,2,4,8,16. 이는 무학습 operator 반복 감사이며2층 본학습 예산을 바꾸지 않는다.

A0 algebra:
- weighted/unweighted power identities, known-C 복원.
- local block energy 전개와 Qθ 에너지 일치.
- occurrence 보정, edge orientation/순열/node 재번호 불변성.
- PSD 및 정규화 bound, isolated/forest/overlap 없음/동일 physical edge 반복.
- coupling off equality, dense/sparse forward와 미분, FP64 gradcheck.
- C/K가 달라도 BᵀC B가 같아지는 invisible parameter 방향.

A1 정보 검사:
- 같은 X를 전체/target/target1홉으로 관측한 경우를 따로 저장.
- 기존 q_v, E_v, J_shared/J_node/J_distinct의 개별 target을 사용. J 정의와 matrix order를 정확히 기록.
- 전역 이론 복원 oracle과 제한된 readout을 분리.
- exact collision proof, numeric near-collision, noisy reconstruction, 학습 readout 성능을 다른 열에 저장.
- 분모0 사례를 null로 기록. numerical rank tolerance를 보고하며 작은 singular value를 정확한0으로 해석하지 않는다.
- 합성≤100node에서는 FP64 직접 분해. 대형 citation은 sparse matvec/반복해법과 수렴 residual을 기록.
  대형 dense global SVD를 강제하지 않고 근사 spectrum을 exact rank로 보고하지 않는다.
- 에너지 감소율이나 ||operator difference||를 정보 손실률로 부르지 않는다.

A2 관측가능성이 부족한 경우의 pair diagnostic:
- analytic collision pair뿐 아니라 원래 feature draws 전부를 보존한다.
- 새 model 유리한 nullspace pair와 역방향 pair를 함께 보고한다.
- 행렬 차원, 관측 노드 수, 실제 feature Jacobian support를 맞춘다.
- finite-depth nonlinear classifier에는 global matrix argument를 그대로 쓰지 않는다.

A3 operator family 차이:
고정 Qθ에 대해 min_{a,b}||Qθ−aLd−bLd²||F/||Qθ||F와 commutator를 기록할 수 있다.
이는 fixed polynomial family와의 차이이지 정보 소실 또는 모든 GNN의 표현 한계가 아니다.
공통 norm으로 맞춘 작용 차이도 별도로 기록한다. 이것은 진단일 뿐 학습 forward의 normalization을 바꾸지 않는다.

## 6. 실험 B: 공유 생성 규칙과 새 그래프 일반화

기존531 graph 규모 및 split cardinality 유지:
train240 / validation60 / ID120 / sizeOOD90 / familyOOD12 / family+sizeOOD9.
train/val/ID n=20,30,40,50; sizeOOD n=60,80,100.
ER/tree/tree+chord 및 cycle/star/grid 구분 유지. graph당16열은 독립 scalar 실현이며 F=1이다.
vector16채널로 해석하지 않는다.

기존 파일을 보존한 채 새 target namespace를 만든다. Dataset master seed는20261002,
새 feature-realization stream은 원본 생성 stream과 분리해 manifest에 hash를 기록한다.
5 model seeds11/23/37/53/71,500epoch,Adam lr=.003,hidden64 유지.
고정 scalar fit은 train-only least squares이며 optimizer run으로 세지 않는다.

Teacher target 세 종류를 모두 비교:
1. 알려진 diagonal LdX.
2. 같은 Ld의 squared message Ld²X.
3. 사전에 고정한 analytic pair rule로 생성한 Q*(X)X.

Teacher3은 위8-feature 입력의3번째 alignment 값을 t, 8번째 overlap 값을 o라 하여
pair raw scalar=tanh(2t + .5(2o−1))로 고정한다.
Student MLP parameter를 teacher로 복사하지 않는다. rho=.5, diagonal a_e=1.
Teacher가 노출된 같은 pair topology/feature를 사용하는 positive control이라는 한계를 명시한다.
Teacher C나 pair weights는 loss에 주지 않고 target message만 감독한다.

각 조건의 message MSE는 realization별 target norm으로 정규화하되0 target은 따로 기록한다.
Graph 안 realization 평균→graph 동일 가중 평균→seed 집계.
보고: message error, effective Q 행동 오차, c_e 및 pair coefficient 진단, unseen feature/size/family 성능.
C matrix/각 occurrence coefficient 자체의 유일한 recovery는 가정하지 않는다.
정답이 Ld나 Ld²일 때 polynomial 대조가 정확히 맞추는 것을 먼저 검산한다.

같은 parameter budget의 readout probe를 붙여 E/J 접근성을 검사할 경우,
raw feature oracle / propagated feature / proposed feature를 구분한다.
probe는 동일1-layer linear와 동일2-layer hidden64 MLP로 사전 고정한다.
새 graph의 정보는 fit에 주지 않는다. teacher3 우위만으로 real-task 효용을 주장하지 않는다.

## 7. 실험 C: 실제 citation 분류

### 7.1 비교군

아래6조건을 unit/local_degree 각각 수행한다:12개.
D0: 고정 diagonal, cross off.
D1: learned diagonal fξ, cross off.
F0: 고정 diagonal + topology-fixed cross.
F1: 고정 diagonal + learned pair cross φθ.
F2: learned diagonal + learned pair cross.
DA: pair-informed diagonal, cross off. F2와 동일 fξ/φθ를 가지며 같은 pair 입력을 읽는다.

DA는 normalized pair coefficient에서 orientation sign을 제거한 scalar를 physical edge별로 평균한다.
그 평균을 fξ의 logit에 더해 a_e=exp(log2*tanh(fξ+pair_mean_e))를 만든다.
Kθ는 node operator에 비대각으로 사용하지 않는다. F2와 active parameter수·pair feature 접근을 맞춘다.
함수 공간과 계산 결과까지 같은 대조는 아니며 실제 FLOPs와 Jacobian support를 기록한다.

독립 baseline3개를 동일 run에 추가: 총15조건.
G1: 표준2-layer GCN, Pgc=Dtilde^−1/2(Aadj+I)Dtilde^−1/2.
G2: 같은2개 projection/ReLU 위치에서 각 macro의 Pgc를 Pgc²로 교체한2-hop-support 대조.
P2: normalized Ltilde=I−Pgc의 degree2 polynomial p(Ltilde).

P2 p(t)=a0+a1t+a2t², t∈[0,2]. κ=max(1,max_[0,2]|p(t)|).
구간 양 끝과 내부 도함수0점에서 κ를 정확히 계산해 p(Ltilde)/κ를 적용한다.
초기 p(t)=1−tau*t. 각층3개 coefficient를 학습하고 κ 의존성도 미분한다.
ChebNet 원 논문 전체를 재현했다고 부르지 않는다. degree2 polynomial의 별도 통제 모델이다.
G1은 외부 기준이고 G2/P2는 새 2hop action과 depth를 비교하기 위한 대조다.

이번 family의 feature-dependent coefficient가 참조하는 특징은 실제 공유-node pair의 endpoints뿐이다.
구조 통계는 고정이며 coefficient 정규화도 topology-only다.
따라서 한 macro의 feature support는 최대2hop이다. 그래도 실제 Jacobian으로 검사하고 문서와 다른 경우 중단한다.

### 7.2 학습 계약

원래 전체 Cora/CiteSeer/PubMed, public masks, row-sum-normalized features, sampling_ratio=1.
모든 physical edge, induced ego, 선언한 모든 eligible pair 사용.
2층 Dinput→64→K, bias없는 Xavier projection, 각 projection 직전dropout.5,
첫 update 후 ReLU, 마지막 logits. 모델 선택에 보조 에너지 loss 없음.

Adam, projection weight decay5e−4, gate/polynomial scalar WD0,
lr .001/.003/.01, tuning101/202/303, final11/23/37/53/71,
각500epoch, float32, TF32off, AMPoff, early stopping/scheduler/clipping 없음.
같은공통 parameter 초기화·dropout stream을 맞춘다. fixed-cross는 처음부터 연산이 다르므로 초기 출력 동등성을 강요하지 않는다.
최종 선택은 validation CE최소→validation accuracy최대→가장 이른epoch.
LR는 tuning-selected validation CE평균최소→작은LR. Final 선택 잠금 뒤 test.

15조건×3datasets×(3LR×3tuningseed+5finalseed)=630새학습.
315,000 optimizer updates. 기존20조건840회의 실험은 그대로 보존하며 이 실험으로 재명명하지 않는다.
비교 질문이 다른 새15조건이므로 기존 normalization 조건 개수를 복사하지 않는다.
실험 중 모델/graph/seed/epoch를 줄이는 자동 fallback은 금지한다.

### 7.3 무엇을 주 비교로 볼 것인가

Primary metric은 기존대로 accuracy. CE는 secondary이며 이득이 나온 뒤 주지표로 바꾸지 않는다.
주 비교3개×2C×3datasets=18 accuracy contrasts:
- F2−D1: 비대각 coupling이 일반 learned diagonal보다 나은가?
- F2−DA: 동일 pair 정보/parameter를 diagonal에만 쓰는 것보다 나은가?
- F2−P2: degree2 polynomial보다 나은가?

F1−D0, F1−F0, F2−F1은 학습·고정 coupling·diagonal 학습의 역할을 분리하는 보조 대비.
G1/G2 대비도 전부 보고하며 최고의 C한 개만 골라 결과를 숨기지 않는다.
개별seed, 평균/std, paired 차이, unadjusted95%tCI,18primary의Holm 보정 결과를 함께 저장한다.
동일 public split을 이미 연구 과정에서 봤으므로 전부 후속 탐색이다. 독립 최종 검증이라고 부르지 않는다.
5seed의 불확실성은 graph/split 일반화 불확실성이 아니다.

실용적 차이 기준으로0.5pp를 **제안값**으로 둔다. 시작 전 승인된 값으로 계약을 잠그고 이후 변경하지 않는다.
0포함 CI는 미결이다. 동등성을 주장하려면 사전 허용범위와 별도 equivalence test가 필요하다.
새 구조를 유지/폐기하는 판단은 실제 uncertainty와 실용적 효과를 함께 보며 통계 미결을 실패증명으로 바꾸지 않는다.

## 8. Frozen 개입과 진단

선택 checkpoint 고정 후:
- 비대각 K만0, diagonal과 N0/tau/projection 유지.
- diagonal a_e만1, pair gate와 N0/tau 유지.
- offdiag pattern의 topology-preserving intervention: 단순 임의 permutation으로 allowed support를 깨지 않는다.
  순열이 support와 symmetry/방향을 보존하는지 검증할 수 없는 graph에서는 해당 셔플을 N/A로 둔다.
- 실제 pair topology를 random edge로 바꾸는 것은 별도 support 변경 실험이며 placement-only 효과라고 하지 않는다.

matched-Z에서 E_intra, E_cross, E_total, ||message||, ||Δ_on/off||,
Ceff−diag(Ceff), BᵀΔCeffB의 norm, constant/kernel 반응, feature support를 기록한다.
weight parameter 변화와 실제 node operator 변화를 나눠 본다.
원래 learned-message norm을 써서 맞추는 교체는 pure removal이 아니므로 기본 on/off와 별도 열에 기록한다.

Every actual training step: projection/gate gradient norm, nonzero/finite count, optimizer update norm.
초기 zero output 때문에 일부 hidden-gate parameter gradient가0인 것은 정상일 수 있다.
모든 파라미터가 매 step 비영이어야 한다는 조건은 두지 않는다. 설계된 gradient fixture에서는 nonzero를 검사한다.

## 9. 실행·자료·실패 처리

신규 namespace 제안: repository/research/edge_metric_relations/ (현재 존재한다고 주장하지 않음).
기존 Conductance/wedge/local/copy 코드 및 결과 덮어쓰기 금지.
재사용은 검증된 data/masks/seed-runner/report 계약과 단순 gather/scatter 유틸리티로 제한.
과거 solver, scalar lift, copy-consensus macro를 새 forward에 암묵적으로 넣지 않는다.

모든 relation을 처리하는 exact chunking과 activation checkpointing을 사용한다.
비용은 O(m_occ·F + P_rel·F + P_rel·MLP비용). 단순 O(m)이라고 하지 않는다.
Pair 수, 메모리, 실측 calibration을 먼저 기록한다. GPU 부족 시 중단/정확chunk 재설정이며 relation/data를 버리지 않는다.
Full training은 server-only. 로컬 fixture 검증은 별도 result namespace.

필수 산출물:
contract.json, source_manifest.json, data_manifest.json,
physical_edges.*, local_membership.*, edge_occurrence.*, cross_pair_index.*,
math_checks.json, observability.csv, operator_actions.csv,
train_history.csv, gate_gradients.csv, validation_selection.json,
metrics_per_seed.csv, comparisons.csv, frozen.csv, energy_terms.csv,
selected.pt, completion.json, resource_report.json.

기존 trained-model loss/관측가능성 감사에는 기존 selected.pt/graph cache/source/contract가 필요하다.
미제공인 현재 서버 원시 파일을 이 설계에서 이미 재분석한 것으로 쓰지 않는다.

## 10. 사전 판정표

- identities/방향/중복/PSD/gradient 실패: 본학습 금지, 구현 수정.
- full known-C에서 q복원: 기존 이론과 일치, '전역 비가역 손실' 주장 철회.
- 일부 관측에서만 collision: 그 관측계약의 결과로 제한.
- polynomial residual nonzero: 다른 연산 확인, 정보 보존/성능은 미확인.
- teacher3회수: 공유 rule 학습 positive control, real task 증명 아님.
- F2가 DA보다 이득 없음: 비대각 작용의 추가 효용은 미확인. pair정보와 parameter 증가를 결합효과로 오인하지 않음.
- CE만 개선: 확률 예측의 손실 개선, accuracy 향상으로 변경하지 않음.
- 정규화나 baseline변경에서만 큰차이: cross성과로 집계하지 않음.
- 여러 graph에 frozen rule 재사용 성공: 지정한 held-graph/size/family 범위의 일반화.
- 실제 비교의 부정적 증거: 이 후보를 계속 복잡하게 만들기 전에 가설/작용범위를 재평가.

## 11. 근거 지도

자료에서 확인한 내용:
- `00_READ_FIRST.md`: 최신 copy의 정의, 고정 C/learned rho, 연구 의도.
- `01_THEORY_AND_MODELS.md` §§2,4: weighted L, adaptive energy gradient, known-C q복원.
- `08_NORMALIZATION_FULL_RESULTS.md`: 내부정규화 효과와 cross accuracy/CE 분리.
- `repository/research/local_context_coupling/normalization/config_full.json`: 실제 최신 전체 학습계약.
- `repository/research/wedge_propagation/learned/config_full.json`: 실제531graph합성 규모·500epoch·5seed.
- `05_EVIDENCE_AND_OPEN_QUESTIONS.md`: raw server artifacts 공백과 결과 제한.

외부 기준 원문:
- Kipf & Welling, arXiv:1609.02907, 식2: 표준GCN normalization.
- Defferrard et al., arXiv:1606.09375, §2.1: polynomial localized spectral filters.
이 설계를 위 논문과 동일하거나 새로운 연구라고 판정하지 않는다.

새로 유도한 부분:
- current fixed-copy ker(T0)⊆ker(Tρ).
- 이차 target Kψ의 관측가능성 조건.
- occurrence-corrected bounded block metric, Ceff 및 forward bound.
위 내용은 이 문서의 제안·대수적 유도이며 기존 완료 실험으로 소급하지 않는다.
