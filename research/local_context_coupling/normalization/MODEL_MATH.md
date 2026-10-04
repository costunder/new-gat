# 내부·교차 정규화 모델의 수식

물리 그래프, 발생행렬, canonical copy 연결은 [원래 copy 모델](../MODEL_MATH.md)과 같다.
이번 변경은 그 연결을 유지하며 내부와 cross의 **고정 대칭 가중치 및 step**을 바꾼다.

## 1. 물리 노드와 로컬 copy

단순 무방향 물리 그래프의 각 중심 v에서 S_v={v}∪N(v)의 유도 로컬 그래프를 만든다.
그 발생행렬을 B_v, 내부 가중치를 C_v로 두면

\[
L_v=B_v^\top C_v B_v,\qquad A=\operatorname{blockdiag}_v L_v.
\]

Unit은 c_ab=1, local_degree는 c_ab=2/(d_v(a)+d_v(b))다. d_v는 해당 유도 로컬 그래프의 degree다.
두 C는 특징이나 학습 파라미터로 생성하지 않는다.

Copy (v,a)는 중심 v의 로컬 그래프에 속한 물리 노드 a다. 인접 중심 v<u에 대해
교집합 S_v∩S_u의 모든 물리 노드 a를 한 번씩 연결한다. 발생행렬 J의 한 행이 이 copy 쌍의 차이를 나타낸다.
Unit cross 라플라시안은 K=JᵀJ다. 인접 중심 쌍과 모든 shared copy 연결을 유지하고 방향 중복만 제거한다.

\[
(RX)_{(v,a)}=X_a,\quad D=R^\top R=\operatorname{diag}(\text{copy counts}),
\quad M=D^{-1}R^\top.
\]

각 물리 노드는 자신의 copy를 항상 가지므로 D의 대각은 양수다.
두 cross 정책 모두 같은 물리 노드의 copy만 연결하므로

\[
GR=0,\qquad R^\top G=0,\qquad MG=0,\qquad MR=I.
\]

## 2. 내부 정책: graph와 local

가중 degree는 d_v^C(a)=Σ_b c_ab다. 물리 그래프 g에 대해 다음 step을 정의한다.

\[
\eta_g=\frac{.5}{\max_{v\in g,a\in S_v}d_v^C(a)},\qquad
\eta_v=\frac{.5}{\max_{a\in S_v}d_v^C(a)}.
\]

분모가 0이면 해당 step은 0이다. Graph 정책은 같은 물리 그래프의 모든 로컬 block에 η_g를 쓰고,
local 정책은 각 로컬 block에 η_v를 쓴다.

\[
S_{graph}=\eta_g A,\qquad
S_{local}=\operatorname{blockdiag}_v(\eta_v L_v).
\]

각 block 안의 step은 scalar다. 노드별 row scaling을 한 비대칭 라플라시안으로 바꾸지 않는다.
앞뒤 내부 pass는 정확히 같은 S를 쓴다. 서로 다른 물리 그래프를 batch하면 η_g도 그래프별로 계산한다.

## 3. Cross 정책: graph와 edge

Unit K에서 copy i의 원래 degree를 k_i라 하자. Graph 정책은 기존 step을 유지한다.

\[
\gamma_g=\frac{.5}{\max_{i\in g} k_i},\qquad G_{graph}=\gamma_g K.
\]

최대 degree가 0이면 γ_g=0이다. Edge 정책은 원래 unit degree로 각 canonical copy 엣지의 가중치를 정한다.

\[
\boxed{G_{edge}=J^\top\operatorname{diag}\!\left(
\frac{.5}{\max(k_i,k_j)}\right)J}.
\]

모든 실제 cross 엣지의 양 끝 degree는 양수다. 정규화한 후의 degree를 다시 분모로 쓰지 않는다.
양 끝에 같은 가중치를 쓰므로 G_edge도 대칭이다. 각 copy의 적용된 가중 degree는

\[
\sum_{j\sim i}\frac{.5}{\max(k_i,k_j)}\le .5
\]

다. Graph G의 최대 가중 degree와 두 S 정책의 최대 가중 degree도 .5 이하다.
Off의 cross_policy는 none이다. 진단용 G는 graph 정책을 참고하며 실제 cross gain은 0이다.

Graph/graph는 이전 분류 모델과 같은 수식이다. 이번 구현은 step을 엣지 가중치에 미리 곱한다.
Float64 값과 gradient의 일치를 검사했으며, float32의 합산 순서 차이 때문에 과거 학습 결과의
bitwise 재현을 주장하지 않는다. 기존 여섯 조건도 이번 코드로 새로 학습한다.

## 4. 전파와 정확한 차이

\[
Y_0=RZ,\quad Y_1=(I-S)Y_0,\quad Y_2=(I-\rho G)Y_1,\quad
Y_3=(I-S)Y_2,\quad H'=MY_3.
\]

\[
T_\rho=M(I-S)(I-\rho G)(I-S)R,\qquad T_0=M(I-S)^2R.
\]

GR=MG=0을 대입하면 step이 서로 달라도

\[
\boxed{T_\rho-T_0=-\rho MSGSR}
\]

다. 첫 pass가 문맥 차이를 만들고 G가 이를 재분배한 뒤, 둘째 S가 merge 후에도 관측되는 변화를 만든다.
즉시 cross 후 merge하는 M(I−ρG)R은 원래 입력과 같다.

Folded F=RᵀSGSR은 대칭 PSD다. 같은 Z에서

\[
\operatorname{tr}\big[Z^\top D(T_0-T_\rho)Z\big]
=\rho\|G^{1/2}SRZ\|_F^2\ge0.
\]

ρ>0이고 GY₁≠0이면 이 값이 양수여서 최종 cross-on/off 출력 차이도 0이 아니다.
이 보장은 대칭 PSD S/G, 같은 앞뒤 S, 균등 M과 정확한 copy 대응을 전제로 한다.
특정 노드 간 derivative 하나가 0일 수 있고, 분류 개선이나 정보 복원을 보장하지는 않는다.

## 5. 안정성은 D metric에서 해석한다

S와 G는 대칭 PSD이며 최대 가중 degree가 .5 이하다. 따라서 각각의 spectrum은 [0,1] 안에 있다.
ρ∈[0,1]이면 I−S와 I−ρG는 copy 공간에서 contraction이다.
Q=RD^(−1/2)는 QᵀQ=I를 만족하며

\[
D^{1/2}T_\rho D^{-1/2}=Q^\top(I-S)(I-\rho G)(I-S)Q
\]

도 대칭 PSD contraction이다. 물리 노드에서 보장하는 norm은

\[
\|H\|_D^2=\operatorname{tr}(H^\top DH),\qquad
\|T_\rho H\|_D\le\|H\|_D
\]

다. 모든 노드의 copy 수가 같지 않으므로 일반 Euclidean node norm의 contraction으로 바꾸어 말하지 않는다.
Projection W는 이 보장의 대상이 아니므로 전체 분류 모델의 norm이 항상 줄어든다고 주장하지 않는다.

## 6. C와 step이 상쇄되는 경우

로컬 block의 모든 엣지 가중치가 공통 scalar c로 바뀌면 L_v와 최대 가중 degree가 함께 c배가 된다.
Local 정책의 η_vL_v에서는 이 c가 상쇄된다.
예를 들어 물리 star의 중심 ego는 local_degree 가중치가 모두 2/(m+1)이고, 잎 ego는 단일 엣지다.
이 구조에서는 local 정책의 S가 unit/local_degree 사이에서 같다.
더 일반적으로 삼각형이 없는 물리 그래프의 모든 1홉 유도 ego는 star 또는 고립 노드이므로,
local 정책에서 두 C 조건은 같은 S를 만든다. 이 경우 C 비교는 적용된 내부 연산이 같은 대조가 된다.
Graph step은 다른 ego의 degree bound도 함께 보므로 같은 상쇄를 모든 구조와 정책에 적용하지 않는다.
이번 실험은 C 이름의 차이와 실제 적용된 S 차이를 구분한다.

## 7. 두 macro 층과 분류 loss

\[
Z_0=\operatorname{Dropout}_{.5}(X)W_0,\quad H_1=\operatorname{ReLU}(T_\rho Z_0),
\quad Z_1=\operatorname{Dropout}_{.5}(H_1)W_1,\quad logits=T_\rho Z_1.
\]

W₀는 F_in×64, W₁은 64×클래스 수이며 bias나 추가 출력 projection은 없다.
Off는 ρ=0, fixed는 ρ=1, learned는 ρ=sigmoid(θ)다. θ는 seed 모델마다 1개이며 두 층에서 공유한다.
초기 θ=0이고 초기 gain은 .5다. Off와 fixed에는 사용하지 않는 θ 파라미터가 없다.

전체 그래프를 전파하고 train mask의 평균 분류 CE를 최소화한다.
Adam weight decay 5e−4는 projection에만 적용한다. θ에는 weight decay를 적용하지 않는다.
C·S·G·R·M은 고정이다. 에너지·복원·teacher 보조 loss는 없다.
고정 대칭 S/G의 custom backward는 같은 action을 gradient에 적용한다. Exact edge chunking은 연결을 버리지 않는다.
Seed packing은 독립 파라미터·dropout stream·Adam 상태를 병렬 계산하며 모델의 batch를 바꾸지 않는다.

## 8. 진단과 frozen 개입

Branch의 energy_operator는 applied_S_G다. 에너지는 각각 ½tr(YᵀSY), ½tr(YᵀGY)이며 raw A/K와 다르다.
Context norm은 ||GY₁||, raw_context_norm은 ||KY₁||다. 적용된 정규화의 변화와 원래 unit cross 관계를 구분한다.
Matched Δ는 현재 층의 동일 Z에서 ||TρZ−T₀Z||다. 위 정확한 식으로 예측한 Δ와의 residual도 기록한다.
Δ/off의 분모는 ||T₀Z||, formula relative error의 분모는 ||Z||이며 0이면 undefined다.

고정 checkpoint에서 gain0/gain1을 layer_0/layer_1/both에 적용한다. 첫 층 개입이면 후속 hidden도 다시 계산한다.
Fixed gain1은 예상 no-op 대조다. 모델 hash를 보존하고 optimizer 갱신은 0회다.
이 개입과 처음부터 off로 다시 학습한 조건은 다른 실험이다.
에너지는 전파의 정의와 진단이며, 모든 pass에서 모든 종류의 에너지가 단조 감소한다고 가정하지 않는다.

