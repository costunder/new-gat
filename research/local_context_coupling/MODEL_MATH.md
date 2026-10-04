# 로컬 문맥 결합 모델: 실제 수식과 주장 범위

이 문서는 `research/local_context_coupling/`의 새 후보를 설명한다.
기존 `local_energy_relations/prediction`·`placement`의 scalar E/J 추가 실험은 보존한다.
새 후보는 그 통계를 학습 벡터로 더하는 구조와 다르며, **로컬 copy 상태 자체를 내부→교차→내부 순서로 전파**한다.

## 1. 상태 공간과 동일 물리 노드의 copy

원래 물리 그래프 G=(V,E)는 단순 무방향 그래프다. 모든 노드·물리 엣지를 사용한다.
각 중심 v의 1홉 집합 S_v={v}∪N(v)에 유도된 모든 물리 엣지를 포함한 로컬 그래프를 만든다.

로컬마다 같은 물리 노드 a의 별도 copy (v,a)를 둔다. 모든 로컬 노드를 펼친 공간은
\(\widetilde V=\{(v,a):a\in S_v\}\)이고, 그 크기를 m이라 하자.
\(H\in\mathbb R^{N\times F}\)를 copy 공간으로 옮기는 연산은

\[
R\in\{0,1\}^{m\times N},\quad (RH)_{(v,a)}=H_a.
\]

물리 노드별 copy 개수의 대각행렬과 균등 merge는

\[
D=R^\top R=\operatorname{diag}(|\{v:a\in S_v\}|),\qquad
M=D^{-1}R^\top,\qquad MR=I.
\]

전체 1홉 로컬을 유지하므로 copy 개수는 원래 physical degree+1이다. 고립 노드도 자기 copy 하나가 있다.
Copy를 즉시 합치지 않고 계산 중 유지하는 것이 이 모델의 핵심이다.

## 2. 로컬 내부 연산 A

로컬 v의 발생행렬 B_v는 해당 로컬에 유도된 물리 엣지의 차분을 만든다.
양의 대각 가중치 C_v는 다음 두 고정 조건이다.

| 조건 | 물리 엣지 {a,b}의 로컬 가중치 |
| --- | --- |
| unit | c_{v,ab}=1 |
| local_degree | c_{v,ab}=2/(d_v(a)+d_v(b)) |

여기서 d_v는 원래 graph의 degree가 아니라 **해당 유도 로컬 graph의 degree**다.
실제 로컬 엣지 양 끝의 degree 합은 양수다. 입력 특징이나 label로 C를 학습하지 않는다.

\[
L_v=B_v^\top C_vB_v,\qquad A=\operatorname{blockdiag}_{v\in V}(L_v).
\]

따라서 A=Aᵀ⪰0이다. 로컬 copy 특징 Y에서 내부 에너지와 gradient는

\[
E_I(Y)=\tfrac12\sum_v\operatorname{tr}(Y_v^\top L_vY_v),\qquad
\nabla_Y E_I=AY.
\]

현재 구현은 앞뒤 두 내부 단계에서 **동일한 A**를 사용한다.
기존 wedge 트랙의 A=e_i−2e_j+e_k와 기호가 겹치지만, 여기의 A는 block 내부 라플라시안이다.

## 3. 로컬 사이 연산 K

원래 물리 엣지로 인접한 중심 {v,u}마다 한 번씩 로컬 pair를 만든다.
S_v∩S_u의 모든 동일 물리 노드 a에 대해 copy (v,a)와 (u,a)를 연결한다.
서로 다른 물리 노드를 직접 연결하는 cross 엣지는 아니다.
기존 topology는 양방향 pair를 저장하므로 **center v<u인 방향만 선택**해 중복을 제거한다.
Disjoint-union batch에서는 graph마다 방향 순서가 다시 시작하므로 전체 pair의 앞 절반을 고르는 방식은 쓰지 않는다.

이 copy 연결의 발생행렬을 J, 양의 cross 가중치를 W라 하면

\[
K=J^\top WJ,\qquad E_J(Y)=\tfrac12\operatorname{tr}(Y^\top KY)
=\tfrac12\sum_{\{v,u\}\in E}\sum_{a\in S_v\cap S_u}w_{vu,a}\|Y_{v,a}-Y_{u,a}\|^2.
\]

본 fixed audit의 W는 unit이다. K=Kᵀ⪰0이며, 처음 동일한 copy에는

\[
KR=0,\qquad R^\top K=0,\qquad MK=0
\]

가 성립한다. 초기 copy의 cross 에너지는0이다. 이를 처음부터 유용한 신호가 있는 것처럼 처리하지 않는다.

### 내부와 교차를 같은 큰 이차형식으로 쓰기

Copy 공간에서 두 에너지는 정확히

\[
E_{joint}(Y)=E_I(Y)+\lambda E_J(Y)
=\tfrac12\operatorname{tr}\bigl(Y^\top(A+\lambda K)Y\bigr),\qquad \lambda\ge0
\]

로 합쳐진다. 두 로컬만 표시하면 교차 matching 행렬 W_vu에 대해

\[
\begin{bmatrix}
L_v+\lambda D_{vu}&-\lambda W_{vu}\\
-\lambda W_{vu}^\top&L_u+\lambda D_{uv}
\end{bmatrix}
\]

가 된다. D_vu/uv는 cross 연결의 양쪽 weighted degree 대각행렬이다.
Cross 에너지의 전개에는 두 내부 copy 제곱항과 −2Y_vᵀW_vuY_u 교차항이 함께 들어간다.
여러 로컬을 연결할 때 각 L_v는 전체 A에 한 번만 넣고, 각 canonical cross pair의 K 기여를 더한다.
이 두 로컬 표시를 pair마다 반복하면서 L_v까지 중복해서 더하지 않는다.

이는 더 큰 **copy 그래프**의 에너지다. 처음 Y=RH에서는 cross가0이므로
처음부터 원래 물리 graph에 새 에너지가 생긴다고 주장할 수 없다.
기존 scalar J=⟨L_vX_v,L_uX_u⟩를 여기의 copy 차이 cross 에너지와 동일시하지 않는다.

## 4. 주 후보: 내부→교차→내부→merge

물리 graph마다 최대 내부 weighted degree와 최대 cross weighted degree를 측정하고

\[
\eta_G=\frac{.5}{\max d_I},\qquad \gamma_G=\frac{.5}{\max d_J}
\]

를 사용한다. 해당 연산의 degree 최대가0이면 작용도0이며, 구현의 유한한 zero-degree 규칙을 사용한다.
Zero 최대 degree에서는 해당 step을0으로 둔다.
이 값은 해당 graph의 모든 로컬 block에 같은 scalar로 적용한다. Batch 전체의 최대값으로 바꾸지 않는다.
이 step은 안정적인 fixed 연산 비교를 위한 규칙이며 모델/graph/data를 줄이지 않는다.

\[
Y_0=RH,\qquad Y_1=(I-\eta A)Y_0,
\]
\[
Y_2=(I-\gamma\rho K)Y_1,\qquad
Y_3=(I-\eta A)Y_2,\qquad H_{on}=MY_3.
\]

Fixed audit는 ρ=1이다. Matched cross-off는 cross 단계만 identity로 하고 **같은 내부 연산을 두 번** 유지한다.

\[
H_{off}=M(I-\eta A)^2RH.
\]

Cross 이후 바로 merge하면 MK=0으로 변화가 소거된다. 두 번째 내부 연산은 copy 사이 재분배를
최종 물리 노드에서 관측되는 변화로 바꾼다.

## 5. 정확한 차이식과 nonzero 보장

Graph마다 η/γ가 같은 scalar이고 앞뒤 A가 같으면

\[
\boxed{H_{on}-H_{off}=-\eta^2\gamma\rho\,MAKARH}.
\]

보장에 필요한 조건은 KR=0, M=D⁻¹Rᵀ, 같은 대칭 A, 대칭 PSD K, η/γ/ρ>0이다.
첫 내부 갱신 후 KY₁≠0이면 KARH≠0이다. 또한

\[
F=R^\top AKAR\succeq0,\qquad
\operatorname{tr}(H^\top FH)=\operatorname{tr}((ARH)^\top K(ARH))>0.
\]

따라서 FH≠0이고 D가 가역이므로 MAKARH≠0이다. 즉

\[
\boxed{KY_1\ne0\ \Longrightarrow\ H_{on}\ne H_{off}}.
\]

반대로 KY₁=0이면 cross가 작용하지 않아 on/off가 같다. Constant 특징·clique의 동일 문맥·고립 노드는 이를 검사하는 control이다.
실제 부동소수점 검사에서는 norm과 absolute/relative 오차 허용 범위를 함께 기록한다.

### 전역 node 연산으로 환원되는 한계

고정 조건에서는 전체 forward가 H′=T_seqH인 선형 node 연산이다.
Copy를 쓴다는 사실만으로 모든 node-level 선형 연산보다 표현력이 강해지지는 않는다.
Folded F는 PSD이지만 M의 D⁻¹ 때문에 최종 T가 보통 Euclidean 좌표에서 대칭인 라플라시안인 것도 아니다.
검증할 질문은 이 연산의 구조적 차이와 task 기여다. 사이클 정보 복원이나 신규성은 별도 검증이 필요하다.

## 6. 추가 fixed control: copy를 유지한 joint 반복

보조 연산은 copy를 매번 merge하지 않고

\[
Y_t=(I-s(A+\lambda K))^tRH,\qquad H_t=MY_t
\]

를 계산한다. Cross-off에서도 같은 t와 s를 유지한다.
KR=MK=0 때문에 t=1/2에서는 cross-on/off 차이가0이고 t=3에서는

\[
H_3^{on}-H_3^{off}=-s^3\lambda MAKARH
\]

가 처음 남는다. Sandwich와 joint 반복을 같은 연산으로 설명하지 않는다.
기본 s는 같은 A/K degree bound에서 정하고 on/off에 공통으로 쓴다.

## 7. 입력 의존성 검사

2홉 입력 의존성이 있는지만 검사하면 부족하다. Cross-off에도 내부 전파를 두 번 하므로
\(\partial H_{off}[v]/\partial H[w]\)가 이미0이 아닐 수 있다.
측정하는 양은 같은 계산 깊이에서

\[
\Delta^J_{vw}=\frac{\partial H_{on}[v]}{\partial H[w]}
-\frac{\partial H_{off}[v]}{\partial H[w]}.
\]

Path0–1–2, η=.1, γ=.35에서는 on/off의 endpoint derivative가 둘 다 .005다.
반면 별도 비정규5노드 control은 비인접3/4에 대한 추가 derivative를 보여준다.
한 endpoint의 차이가0이라는 것과 전체 on/off 차이가0이라는 것을 구분한다.

## 8. 두 층 분류기와 학습 연결: DEBUG 검사만

후속 학습 연결을 확인하는 실제 분류기 API는

\[
Z^{(\ell)}=\operatorname{Dropout}(H^{(\ell)})W_\ell,\qquad
\widehat H^{(\ell+1)}=T_{seq}(\rho)Z^{(\ell)},
\]

첫 층 뒤 ReLU, 둘째 층 logits를 사용하는 2층·hidden64·bias 없는 projection 구조다.
Cross-on은 seed마다 θ_J 하나를 두 층에서 공유하고 ρ=sigmoid(θ_J)로 cross step의 강도를 학습한다.
θ_J 초기값은0이며 초기 gain은.5다. Fixed audit의 gain1과 구분한다.
Cross-off는 θ_J를 등록하지 않으며, 두 내부 pass는 유지한다. C/K topology는 고정이다.
실제 CE→backward→optimizer update에서 θ_J와 projection gradient/update를 검사한다.
이 CE 검사는 명시된 DEBUG fixture이며 citation 분류 본학습 결과가 아니다.

본 `study --profile full`은 아래 모든 입력에서 **fixed audit**를 수행한다. Classifier optimizer/checkpoint는 만들지 않는다.
향후 본학습을 하려면 별도 데이터·조건·LR·seed·epoch·평가 계약을 확정해야 한다.

## 9. FULL fixed audit 계약

- Source: 기존 완료 `local-energy-20261004-002815`의 manifest·원래 데이터와 topology. 원본 보존.
- 합성198 graph·16 scalar 실현씩과 Cora1433/CiteSeer3703/PubMed500 실제 전체 feature channel: 총201 graph·8,804 feature 열.
- H0/H1/H2는 기존 audit의 **고정 physical reference 상태**다. 물리 unit L과 τ=.5/max physical degree로 만든 기존 상태이며, 새 copy 모델의 중간 단계 이름이 아니다.
- 각 입력 상태를 새로운 immediate/sandwich/persistent1/2/3에 각각 제공한다. 2 C조건·3상태의 모든 입력을 유지한다.
- 모든 로컬·물리 엣지·canonical cross correspondence·channel을 계산한다. 샘플링/숨겨진 cap 없음.
- 독립 graph는 disjoint-union batching, 모든 feature 열은 tensor 연산으로 처리한다. Exact chunking은 값을 버리지 않는 메모리 처리다.
- Channel metric은52,824행(8,804×2×3), graph summary는1,206행(201×2×3)이다.
  Graph norm 비율은 channel별 비율을 평균하지 않고 전체 channel squared norm을 합친 뒤 계산한다.
- Hardware/CPU 준비/GPU batch 계측과 실제 선택을 기록한다. 물리 graph batch와 독립 feature/seed 축은 별개다.

실행·산출물은 [RUN.md](RUN.md), 당시 검증 범위는 [VERIFICATION.md](VERIFICATION.md)를 따른다.
