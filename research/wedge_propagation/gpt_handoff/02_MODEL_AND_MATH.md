# 현재 모델의 작동 방식과 수식

## 먼저 읽을 설명

현재 연구는 `research/wedge_propagation`의 독립 wedge 트랙이다.
기본 단위는 **중심 노드와 그 노드에 붙은 서로 다른 두 이웃**이다.
i–j–k에서 두 엣지의 차이가 얼마나 다른지 계산하고,
그 차이에 경로별 가중치 C를 곱한 다음 세 노드에 다시 모은다.

실험은 고정 경로 연산 확인 → 합성 규칙 학습 → 새 그래프·특징 평가 →
RMS 입력 정규화 → 실제 노드 분류 → C 개입·분기 크기 분석 → 노드별 정규화 재학습으로 이어진다.
**에너지를 정의할 수 있다는 것, 에너지를 학습 loss로 쓴다는 것,
C의 배치가 분류에 도움이 된다는 것은 각각 다른 주장이다.**
아래는 실제 계산과 학습 목표의 설명이다. 성능 결과와 증거의 범위는 결과 문서를 함께 읽는다.

## 1. B, L, A, Q

N은 노드 수, E는 물리 엣지 수, P는 wedge 수다.
물리 그래프는 self-loop와 중복 방향 엣지를 제거한 단순 무방향 그래프다.
각 물리 엣지를 방향을 정해 한 번 기록한다.

\[
B\in\mathbb R^{E\times N},\quad B_{uv,:}=-e_u^\top+e_v^\top,
\qquad L=B^\top B.
\]

BX는 엣지별 노드 차분이다. 양의 엣지 가중치 W를 쓰면 BᵀWB가 가중 라플라시안이다.
현재 분류기의 일차 분기는 고정 L을 사용한다. 학습하는 C는 아래의 **wedge 행**에 붙는다.
중심 j의 서로 다른 두 이웃 {i,k}마다 한 행을 만든다. 이웃 쌍은 unordered이고
삼각형에서도 중심을 공유하는 두 엣지가 있으면 wedge를 포함한다.

\[
A\in\mathbb R^{P\times N},\quad A_{ijk,:}=e_i^\top-2e_j^\top+e_k^\top,
\qquad Q=A^\top A.
\]
\[
g_1=x_j-x_i,\quad g_2=x_k-x_j,\qquad (AX)_{ijk}=g_2-g_1.
\]

A는 두 차분의 차이를 만들고 Aᵀ는 `(1,−2,1)`로 노드에 다시 집계한다.
Wedge의 두 엣지를 e,f라 할 때 R의 두 계수를 −B[e,j], −B[f,j]로 두면 A=RB다.
물리 엣지 방향을 바꾸면 대응하는 R의 부호도 바뀐다. R은 여기서 정의한 엣지 간 결합 행렬이다.

### 고정 C=I의 정확한 빠른 식

\[
\boxed{Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B}.
\]

d_v는 물리 degree다. 엣지 보정 계수는 음수일 수 있으며 코드가 이를 자르지 않는다.
전체 Q는 AᵀA이므로 PSD다. Fixed QX는 이 식으로 O(EF)에 정확히 계산한다.
경로마다 C가 다른 learned 연산은 모든 wedge를 계산해야 한다.
엣지마다 degree sum이 같은 s이면 Q=L²+(s−4)L이다.
Cycle에서는 Q=L², d-regular에서는 Q=L²+2(d−2)L이다.
Q가 항상 L·L²와 독립이라는 주장은 성립하지 않는다.
일반 Q에는 거리 2의 양의 비대각 항도 생길 수 있다.

구현: [operators.py](../operators.py), [algebra.py](../algebra.py).

## 2. 이차형식과 쌍선형 교차항

현재 C를 양의 대각행렬로 고정하면

\[
T_C=A^\top CA,\qquad E_C(X)=\operatorname{tr}(X^\top T_CX)
=\sum_p c_p\|(AX)_p\|^2
\]

는 이차형식이다. 한 경로에서

\[
c_p\|g_2-g_1\|^2
=c_p\|g_1\|^2+c_p\|g_2\|^2-2c_p g_1^\top g_2.
\]

마지막 항은 두 엣지 차분의 쌍선형 교차항이다. 이 관계가 현재 A 안에 들어 있다.
현재 모델은 wedge별 **scalar C**를 만들고 모든 channel에 공유한다.
임의의 직사각 K를 학습하는 집합 간 쌍선형 모듈은 구현되어 있지 않다.
C는 경로 좌표에서 차분을 가중한다. 기존 라플라시안의 일반 합동변환을
추정하는 별도 알고리즘은 없다.

C를 고정하면 E_C의 X gradient는 2T_CX다.
모델에서는 C=C(X)이므로 전체 에너지 gradient에는 C의 미분항도 들어간다.
실제 메시지는 T_C(X)X로 정의한다. 이를 C(X)까지 미분한 전체 에너지 gradient와
동일하다고 해석하면 안 된다. 실제 loss backward는 메시지의 C 의존성을 그대로 미분한다.

## 3. 합성 실험의 scalar 입력과 규칙 학습

고정 실험은 198개 그래프·3,168개 scalar 입력이며 노드 크기는 20/30/40/60/80/100이다.
각 X의 shape는 `[N,16]`이다. **16개 열은 독립 scalar 실현 16개이고 물리 channel은 1개다.**
학습 파라미터·분류기·loss 없이 LX·L²X·QX, 에너지, 고유값, 다항식 적합 잔차를 측정한다.

Learned 실험은 531개 그래프·8,496개 scalar 실현을 사용한다.

| Split | 그래프 수 | 노드 크기 | Family |
| --- | ---: | --- | --- |
| train | 240 | 20/30/40/50 | ER, tree, tree_chord |
| validation | 60 | 20/30/40/50 | ER, tree, tree_chord |
| ID | 120 | 20/30/40/50 | ER, tree, tree_chord |
| size OOD | 90 | 60/80/100 | ER, tree, tree_chord |
| family OOD | 12 | 20/30/40/50 | cycle, star, grid |
| family+size OOD | 9 | 60/80/100 | cycle, star, grid |

모든 wedge와 scalar 실현을 사용한다. 서버 학습 physical graph batch는 240개였다.
Seed 축 S=5는 독립 모델들의 병렬 계산 축이며 한 모델의 batch를 5배로 만들지 않는다.

### Teacher와 student

Teacher는 각 scalar 실현에서 다음 규칙을 사용한다.

\[
r_p^*=\frac{g_1g_2}{|g_1||g_2|+10^{-8}}
+\frac{|g_2-g_1|}{|g_1|+|g_2|+10^{-8}},
\]
\[
c_p^*=\frac{e^{\tanh r_p^*}}{\operatorname{mean}_{q\in G}e^{\tanh r_q^*}},
\qquad Y^*=A^\top\operatorname{diag}(c^*)AX.
\]

C*는 target 생성과 진단에 쓴다. Student 입력이나 C target loss로 제공하지 않는다.
Student는 같은 네 scalar 통계를 공유 MLP에 넣는다.

\[
\phi_p=[|g_1|+|g_2|,\ g_1g_2,\ |g_2-g_1|,\ (|g_1|-|g_2|)^2],
\]
\[
s_p=\mathrm{MLP}_{4\to64\to1}(\phi_p),\quad
c_p=\frac{e^{\tanh s_p}}{\operatorname{mean}_{q\in G}e^{\tanh s_q}},\quad
\hat Y=\beta A^\top\operatorname{diag}(c)AX.
\]

합성 gate의 두 linear 모두 bias가 있고 중간에 ReLU가 있다.
Gate 385개와 자유 scalar β 1개를 합해 seed당 386개 파라미터다.
C는 `[S,P,16]`, 출력은 `[S,N,16]`이며 실현마다 C를 따로 생성한다.
경로를 뒤집으면 (g₁,g₂)→(−g₂,−g₁)이므로 네 통계가 동일하다.

### 목표와 loss

Targets는 LX, L²X, 위 경로 teacher Y*의 세 가지다.
그래프 G·실현 r의 loss는

\[
\ell_{G,r}=\frac{\operatorname{mean}_i(\hat Y_{i,r}-Y^*_{i,r})^2}
{\operatorname{mean}_i(Y^*_{i,r})^2+10^{-8}}.
\]

실현 평균·graph 평균을 구하고 독립 seed들의 loss를 합해 backward한다.
Learned/random-pair를 각각 500 epoch 학습하며 validation 메시지 상대오차로 선택한다.
First `uLX`, polynomial `uLX+vL²X`, fixed `βQX`는 train만 사용한 가중 최소제곱 해다.
이 결정적 scalar 대조군을 5개의 독립 학습 결과로 세지 않는다.

Random-pair는 동일한 수 P의 서로 다른 물리 엣지 쌍으로 A_r을 만들고 별도로 학습한다.
행 norm은 √6으로 맞추지만 공유 중심·support·엣지 사용 빈도도 달라질 수 있다.
결과 차이를 경로 연속성 하나의 인과 효과로 단정하면 안 된다.

구현: [learned/model.py](../learned/model.py), [learned/data.py](../learned/data.py),
[learned/train.py](../learned/train.py), [learned/config_full.json](../learned/config_full.json).

## 4. 새 특징 평가와 RMS 입력 정규화

Experiment 3은 선택한 θ·β와 scalar 대조군을 고정한다.
독립 표준정규 특징과 배율 0.25/0.5/1/2/4에서 teacher·target도 새로 계산한다.
Optimizer update는 0이다.

Experiment 3.1은 gate 입력을 다음처럼 정규화해 같은 합성 목표로 재학습한다.

\[
\sigma_{G,r}=\sqrt{\operatorname{mean}_{e\in E_G}(BX_r)_e^2},\quad
\tilde g_1=g_1/\sigma_{G,r},\quad\tilde g_2=g_2/\sigma_{G,r}.
\]

엣지가 없거나 차이가 정확히 0이면 σ=1이다. 양수 σ에 epsilon을 더하지 않는다.
정규화 차분은 C 입력에만 사용하고 메시지의 AX는 원래 입력을 유지한다.
이상적인 산술의 양의 배율 a에서 C(aX)=C(X), Ŷ(aX)=aŶ(X)다.
이 성질이 목표 복원이나 분류 성능을 보장하지는 않는다.
실제 teacher에는 epsilon이 있으므로 모든 배율의 정답을 다시 계산한다.

구현: [generalization/study.py](../generalization/study.py),
[generalization/frozen.py](../generalization/frozen.py),
[scale_normalization/model.py](../scale_normalization/model.py),
[scale_normalization/study.py](../scale_normalization/study.py).

## 5. 실제 노드 분류의 두 층 vector 모델

합성 teacher의 checkpoint를 옮기지 않는다.
분류기는 현재 투영 vector Z에서 자체 C를 만들고 train label의 CE로 새로 학습한다.

| 데이터 | N | 입력 D | 클래스 K | 물리 E | wedge P | train/val/test |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 2708 | 1433 | 7 | 5278 | 52301 | 140/500/1000 |
| CiteSeer | 3327 | 3703 | 6 | 4552 | 26918 | 120/500/1000 |
| PubMed | 19717 | 500 | 3 | 44324 | 699342 | 60/500/1000 |

공개 transductive split의 전체 그래프·wedge를 사용한다.
모든 노드의 특징·연결을 사용하고 학습 CE는 train mask의 label로 계산한다.
두 층은 같은 노드·엣지·wedge 집합을 사용한다.

\[
Z^{(\ell)}=\operatorname{Dropout}_{0.5}(H^{(\ell)})W^{(\ell)},\qquad
U^{(\ell)}=Z^{(\ell)}-\alpha_\ell\bar LZ^{(\ell)}-\beta_\ell M_\ell.
\]

첫 투영은 D→64, 둘째는 64→K다. Projection bias와 추가 output projection은 없다.
첫 U 뒤에는 ReLU, 둘째 U는 logits다.
Packed shape는 `[S,N,F]`이고 각 seed의 파라미터·Adam 상태는 독립이다.

\[
\bar L=\tfrac12S_dLS_d,\quad S_d=\operatorname{diag}(d)^{-1/2},\qquad
t=\operatorname{sigmoid}(u),\ r=\operatorname{sigmoid}(v),\quad
\alpha=t(1-r),\ \beta=tr.
\]

초기 α=0.5, β=0.25이고 α+β=t<1이다. Degree 0 노드의 S_d는 0이다.

### 분류 gate의 입력과 출력

g₁,g₂는 F차원 vector다. 네 통계를 channel별로 계산해 길이 4F로 이어 붙인다.
Gate는 **4F→64→1**이며 첫 linear에만 bias와 ReLU가 있다.
첫 층 F=64, 둘째 F=K다. C는 `[S,P]`이며 scalar 하나를 모든 channel에 공유한다.
Exp(tanh(score))를 전체 P개 경로 평균으로 나눈다. Mean=1이 C 전부 1을 뜻하지 않는다.

Raw는 차분을 그대로 쓰고 RMS는 gate 입력을 다음 값으로 나눈다.

\[
\sigma=\sqrt{\|BZ\|_F^2/(EF)}.
\]

분류 RMS는 graph·layer·seed마다 모든 엣지·channel을 함께 평균한다.
최종 C mean도 전체 경로를 사용하므로 최종 가중치에 graph 전체 통계가 들어간다.
현재 구현에 노드별 incoming-edge softmax는 없다.

## 6. Global κ와 node 대각 정규화

\[
q_v=(A^\top A)_{vv}=4\binom{d_v}{2}+\sum_{j\sim v}(d_j-1),
\qquad S_Q=\operatorname{diag}(q)^{-1/2}.
\]

q와 S_Q는 topology에서 계산해 cache한다. 현재 C의 가중 대각은

\[
(D_C)_v=\sum_p c_p a_{pv}^2.
\]

코드는 wedge의 끝점에 c, 중심에 4c를 scatter한다.
Wedge에 속하지 않는 노드의 역제곱근은 0이다.

| 모델 | 이차 메시지 M | 특징 |
| --- | --- | --- |
| Fixed C=1 | S_QAᵀAS_QZ/3 | C를 학습하지 않음 |
| Global/raw·RMS | S_QAᵀCA S_QZ/(3κ) | 하나의 κ가 전체 분모 |
| Node/raw·RMS | S_CAᵀCA S_CZ/3 | S_C=D_C^(-1/2), 현재 C의 노드별 대각 |

\[
\kappa(C)=\max_{q_v>0}(D_C)_v/q_v.
\]

경로가 없으면 global κ=1이고 이차 메시지는 0이다.
Node는 C·D_C·왼쪽 S_C·오른쪽 S_C 모두 CE 미분 경로에 남긴다.
Gate의 C는 S_CZ가 아니라 **현재 원래 투영 Z**에서 만든다.
Node 전환은 κ만 없애는 변경보다 크다. 양쪽 노드 좌표와 메시지 방향도 달라진다.
C=I이면 S_C=S_Q, κ=1이므로 두 정규화가 같다. 따라서 fixed 대조군은 하나다.

### 현재 C를 고정했을 때의 상한

세 항의 Cauchy–Schwarz로 AᵀCA≼3D_C이고

\[
0\preceq\tfrac13S_CA^\top CA S_C\preceq I.
\]

Global도 D_C≼κ diag(q)이므로 PSD·norm≤1 상한을 갖는다.
일차 분기도 0≼L̄≼I이므로 현재 C의 조건부 전파 행렬 고유값은 1−α−β와 1 사이다.
이는 **현재 C를 고정한 선형 행렬**에 대한 성질이다.
C=C(Z)인 전체 함수의 Jacobian 상한이나 학습 성공을 보장하지 않는다.
Node의 조건부 에너지는 `(1/3)||C^(1/2) A S_C Z||²_F`다.
에너지 값을 별도 출력·loss로 추가하는 구현은 없다.

구현: [classification/model.py](../classification/model.py),
[node_normalization/model.py](../node_normalization/model.py).

## 7. CE 학습과 선택 계약

Train-node 평균 CE를 학습하며 packed seed loss를 합해 backward한다.
Adam의 coupled L2는 projection/gate weight에만 적용한다.
Bias·u·v에는 decay가 없고 평가 CE에는 L2 항을 포함하지 않는다.
Teacher loss·C label loss·복원 loss는 없다.

모든 run은 500 epoch다. Tuning seed는 101/202/303,
LR 후보는 0.001/0.003/0.01, final seed는 11/23/37/53/71이다.
Checkpoint는 최소 validation CE→최대 validation accuracy→가장 이른 epoch 순으로 선택한다.
LR은 tuning seed 평균 selected validation CE→작은 LR 순으로 선택한다.
Final checkpoint를 모두 확정한 뒤 test를 평가한다.

Experiment 4의 8조건은 MLP, first_order, polynomial_2, fixed_wedge,
learned raw/RMS, fixed_wedge_node_mlp, standard_gcn이며 336 run·168,000 update다.
Experiment 4.2는 fixed/global raw·RMS/node raw·RMS의 5조건을 새로 학습하며
210 run·105,000 update다. 기존 run을 새 학습 결과로 재사용하지 않는다.

Experiment 4의 각 층에서 비교하는 계산은 다음과 같다.

| 조건 | 투영 후 계산 |
| --- | --- |
| MLP | Z만 전달 |
| First order | Z−sigmoid(u)L̄Z |
| Polynomial 2 | Z−αL̄Z−βL̄²Z |
| Fixed wedge | Z−αL̄Z−βQ̄Z |
| Learned raw/RMS | Z−αL̄Z−βM_global |
| Fixed wedge + node MLP | Fixed 계산 + β N(Z), N은 F→128→F ReLU MLP |
| Standard GCN | Self-loop를 포함한 D̃^(-1/2)(Adj+I)D̃^(-1/2)Z |

GCN의 D̃는 self-loop를 더한 degree다. Gate 모델과 node MLP 대조군은
같은 활성 파라미터 수를 갖지만 공간 연산과 입력 정보는 다르다.

| 데이터 | Fixed 파라미터/seed | Learned global·node 파라미터/seed |
| --- | ---: | ---: |
| Cora | 92164 | 110596 |
| CiteSeer | 237380 | 255556 |
| PubMed | 32196 | 49604 |

Normalization 자체에 추가 파라미터는 없다.
구현: [classification/training.py](../classification/training.py),
[classification/study.py](../classification/study.py),
[node_normalization/training.py](../node_normalization/training.py),
[node_normalization/study.py](../node_normalization/study.py).

## 8. Frozen C 개입과 분기 크기

Frozen 검사는 θ·projection·α·β를 고정하며 추가 update는 0이다.
각 개입을 layer_0, layer_1, both에 적용한다.
뒤층 참조 C는 앞층 개입 후 **그 층에 도달한 현재 Z**에서 다시 만든다.
Global의 참조 메시지를 R/κ라 쓰면 실제 분기 크기는 seed마다 정확히

\[
\frac{\|\beta M\|_F}{\|Z\|_F}
=\frac{\beta}{\kappa}\frac{\|R\|_F}{\|Z\|_F}
\]

다. 평균들의 곱으로 대체하지 않는다.

Experiment 4.1은 true C의 κ→1, C=1의 참조 κ 유지·κ→1,
norm matching, shuffle, 분기 제거를 기존 global checkpoint에서 비교한다.
Experiment 4.1.1은 저장된 scalar CSV 분석이며 새 forward·학습이 없다.

Experiment 4.2의 native C=1/shuffle은 해당 C로 global κ 또는 local D_C·양쪽 S_C를
다시 계산한다. C를 바꾸면 자연스러운 메시지 크기도 달라질 수 있다.
Norm-matched 후보는 seed·현재 층 Z마다

\[
M_{\rm matched}=\frac{\|M_{\rm reference}\|_F}{\|M_{\rm candidate}\|_F}
M_{\rm candidate}
\]

로 바꾼다. **C=1 후보도 참조 learned-C 메시지의 전체 norm을 유지한다.**
이것은 native C=1과 별도의 개입이다. βM의 norm도 맞지만 노드별 방향,
일차 분기와의 상쇄, ReLU 결과는 다를 수 있다.
사후 gain이 1을 넘을 수 있어 native norm≤1 상한을 그대로 적용하지 않는다.
두 메시지가 모두 0이면 gain=1, 참조는 양수인데 후보만 0이면 오류로 중단한다.
정의되지 않은 비율·cosine은 defined flag로 남긴다.
Shuffle 10개는 seed 안에서 평균하며 독립 학습 seed 50개로 세지 않는다.
두 층 동시 개입은 효과가 상쇄될 수 있으므로 저장한 층별 행도 함께 읽는다.

구현: [branch_strength/core.py](../branch_strength/core.py),
[branch_analysis/core.py](../branch_analysis/core.py),
[node_normalization/evaluation.py](../node_normalization/evaluation.py).

## 9. 계산 방식과 구현 범위

### 코드를 읽을 때의 주요 API

| 파일·API | 입력 → 출력 |
| --- | --- |
| `operators.fixed_wedge_fast_apply(edges, x)` | 물리 edge index와 X → 고정 QX |
| `learned.model.SeedBatchedModel(kind, seeds, hidden=64, tau=1)` | 합성 batch → `[S,N,16]` 메시지와 C 또는 None |
| `classification.model.PackedClassifier(condition, input_dim, classes, seeds, ...)` | graph → `[S,N,K]` logits와 두 층 detail |
| `node_normalization.model.PackedClassifier(...)` | 같은 생성자·forward 인터페이스의 5조건 모델 |
| `probe_layer(graph, layer_index, z, ...)` | 현재 `[S,N,F]` Z → 이차 메시지와 detail |
| `apply_node_branch(graph, z, c)` | 주어진 양의 C로 D_C를 계산 → node 정규화 메시지 |

분류 `forward(graph, epoch=0, ..., diagnostics=False)`에서 epoch는 공통 dropout stream을 정한다.
학습에 쓰는 C는 미분이 연결된 tensor다. Detail의 C·Z·메시지는 진단용으로 detach한다.
새 모델은 `experiment_condition`에 5조건의 실제 이름을 기록한다.
내부 `condition`은 기존 raw/RMS gate·forward를 재사용하기 위해 canonical 이름을 유지한다.
학습·checkpoint metadata에는 외부 실험 이름을 사용한다.

- B/A와 degree/q는 topology에서 고정된다. Projection·gate·α·β는 학습된다.
  C와 node D_C는 입력·layer·seed마다 새로 계산된다.
- Dense B/A 대신 edge/path index와 scatter로 적용한다.
  Fixed Q의 빠른 항등식과 learned C의 전 경로 계산을 구분한다.
- Packing·path chunk는 실제 처리량·VRAM으로 선택한다.
  Chunk는 모든 경로를 유지하는 계산 분할이며 subset이 아니다.
- Source/config/graph/checkpoint hash와 resume identity를 검증한다.
  기존 결과를 덮어쓰지 않고 새 디렉터리에 복원한다.
- Gradient, 입력→logits→CE→Adam 연결, packed와 독립 seed의 동일성은 DEBUG로 검사한다.
  DEBUG 성공과 서버 전체 학습 결과를 구분한다.
- 합성의 새 graph·family·size 평가와 citation의 고정 public split 평가는 구분한다.
  기존 test를 본 뒤 정한 후속 비교를 독립 일반화 검증으로 해석하지 않는다.

현재 wedge 트랙에는 명시적인 사이클 잔차 분리·복원이나 별도 엣지 상태 보존이 없다.
홉마다 다른 송신·수신 집합을 만들고 경계 누락을 기록하는 경로도 없다.
각 GNN 층에서 일차·이차 연산을 함께 계산한다.
고유벡터 기반 spectral 단계 뒤에 별도 spatial 단계를 연결한 모델로 보고하지 않는다.
이 설명은 앞선 대화의 구상과 현재 구현의 차이를 명확히 하기 위한 것이다.

최종 설정과 source 계약은
[node_normalization/config_full.json](../node_normalization/config_full.json),
[node_normalization/common.py](../node_normalization/common.py),
[node_normalization/MODEL_MATH.md](../node_normalization/MODEL_MATH.md)에서 확인한다.
