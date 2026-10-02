# 고정 경로 연산부터 검증하는 독립 실험 계획

2026년 10월 2일. 사용자 첨부의 수정 요청을 반영한 현재 계획이다. 이전 계획은 archive에 보존했으며 그 설정과 실행 수를 계승하지 않는다. Experiment 0/1 코드는 구현했으며, 실행과 검증 상태는 [README](README.md)와 [검증 기록](VERIFICATION.md)에 구분해 적었다. 이 문서의 후속 학습 단계는 설계다.

연구 질문은 **연속된 두 엣지의 특징 변화 관계를 이용한 경로 차분 연산이 무엇을 추가하며, 이후 공유 생성기로 그 경로 가중치를 학습할 수 있는가**이다.

실행 순서는 고정한다. 먼저 대수 검증과 고정 연산자 비교만 수행한다. 그 결과를 확인한 뒤 새 경로 가중치 학습, 독립 그래프 평가, 실제 분류를 진행한다. 초기 단계에 학습 C2 생성기를 넣지 않는다.

## 독립 실험의 범위

- 기존 Conductance 모델, C 생성기, 정규화, checkpoint, optimizer 규약, synthetic generator를 가져오지 않는다.
- 앞 계획의 degree 입력, graph mean C 정규화, softmax alpha/beta, hidden 256, AdamW, 10seed와 698run은 이번 실험의 기본 설정에서 제거한다.
- Adaptive edge, blind gate, repeated first-order, PPI, arxiv는 초기 실험 범위에서 제외한다. GraphSAGE, GAT, GATv2도 이번 비교에 넣지 않는다.
- 프로젝트 안전 규칙, 전체 경로 보존, 실제 자원 계측, debug와 본학습의 구분은 그대로 적용한다.
- 새 code/data/results/config를 이 트랙 안에 분리한다. 이전 코드와 실험 결과는 보존한다.

## 새 연산의 정의

단순 무방향 그래프의 물리 엣지마다 B의 행을 하나 만든다. 자기 연결과 reciprocal duplicate는 제외한다. 각 중심 j와 서로 다른 이웃의 unordered pair {i,k}를 한 번씩 사용한다. 삼각형 안의 경로도 포함한다.

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,
\quad L=B^\top B,
\quad Q=A^\top A,
\quad L_2(C_2)=A^\top C_2A.
\]

\[
g_1=h_j-h_i,\qquad g_2=h_k-h_j,
\qquad (AH)_p=g_2-g_1.
\]

R의 두 계수를 `R[p,e]=-B[e,j]`, `R[p,f]=-B[f,j]`로 두면 A=RB가 된다. 엣지 방향을 바꾸면 R도 함께 바꿔 동일한 A를 유지한다.

\[
E_2(H)=\sum_p c_p\|g_2-g_1\|^2
=\operatorname{tr}(H^\top A^\top C_2AH).
\]

이 식에는 두 차분의 내적이 포함된다. 양의 c는 경로의 이차 차분에 penalty를 부여한다. 이 연산을 반전을 적극적으로 유도하는 signed metric이나 임의 그래프의 물리적 곡률로 해석하지 않는다.

## Experiment 0 대수와 구현 검사

학습 없는 debug 검사에 path, cycle, star, clique, 비정규 그래프를 사용한다.

1. A=RB, 경로 뒤집기, 엣지 방향 변경, 노드 재번호 부여를 검사한다.
2. 에너지 항등식, PSD, row sum 0, 경로 중복 제거를 검사한다.
3. A를 직접 구성한 구현과 독립적으로 계산한 식을 비교해 다음 항등식을 검사한다.

\[
\boxed{Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B.}
\]

Float64 행렬 참조와 tensor gather/scatter 구현을 비교한다. 연산 단위 테스트에는 출력과 입력 gradient의 일치 검사를 포함한다. Debug 통과를 학습 완료로 표현하지 않는다.

## Experiment 1 고정 연산자만 비교

**이 단계에는 학습 C2, optimizer, classifier를 넣지 않는다.** 단일 block에서 입력 특징을 그대로 사용해 raw action LX, L²X, QX를 비교한다. Raw operator를 적용한 결과와 I−tau T 형태의 diffusion을 구분한다.

고정 연산자 측정을 위한 새로운 데이터 계약을 다음과 같이 제안한다. 측정 범위는 첨부의 그래프 크기와 구조를 기준으로 정한다.

- 노드 수: 20/30/40/60/80/100. 첨부에 나온 작은 그래프와 큰 그래프의 범위를 여러 측정점으로 나눈다.
- 고정 구조: cycle, star, rectangular grid. 각 크기에서 하나씩 사용한다.
- 확률 구조: ER, Prüfer 열로 생성한 tree, 해당 tree에 연결되지 않은 pair의 chord를 floor(n/4)개 추가한 graph. 각 크기에서 독립 graph draw 10개를 사용한다.
- ER은 각 pair를 확률 4/(n−1)로 연결한다. 고립 노드와 비연결 graph도 유지하며, 그 수를 기록한다.
- Grid의 열 수는 크기 순서대로 4/5/5/6/8/10이며, n개 노드를 모두 사용한다.
- 합계는 198graph다. Cycle과 star의 동형 복사본을 독립 graph draw로 세어 표본 수를 늘리지 않는다.
- 각 graph에 독립적인 scalar X 16개를 표준정규분포에서 생성한다. 전체 입력 수는 3168이다. Graph seed와 feature seed를 따로 고정하고 manifest에 저장한다. Graph draw와 feature draw의 master seed는 20261002로 기록하고 두 난수 stream을 분리한다.

다음 다섯 항목을 측정한다.

| 측정 | 판단할 내용 |
| --- | --- |
| LX, L²X, QX | 동일한 특징에 대한 작용의 차이 |
| X.T L X, X.T L² X, X.T Q X | 에너지의 차이 |
| 고유값·nullspace·operator norm | 강도와 소거되는 방향의 차이 |
| 엣지별 d_u+d_v의 분포 | Degree 보정이 일정한지, 위치에 따라 달라지는지 |
| Q를 span{L,L²}에 맞춘 잔차 | 고정된 이차 다항식으로 설명할 수 있는 범위 |

주표에는 raw scale을 그대로 기록한다. 보조표에서는 0이 아닌 각 operator를 해당 spectral norm으로 나눠, 강도를 맞춘 상태의 차이도 확인한다. 작은 graph의 참조 계산이므로 고유값을 직접 계산할 수 있다. 이 측정용 정규화는 모델의 forward 규약과 구분한다.

Polynomial 잔차는 다음과 같이 정의한다.

\[
r_G=\frac{\min_{u,v}\|Q-uL-vL^2\|_F}{\|Q\|_F}.
\]

이는 평가 graph의 전체 행렬을 사용하는 진단이다. Graph마다 u,v를 다시 추정하는 학습 모델이나 일반화 성능으로 해석하지 않는다. Q=0이면 상대값은 undefined로 표시하고, absolute residual을 별도로 기록한다.

Regular graph에서는 Q=L²+2(d−2)L, cycle에서는 Q=L²다. 또한 모든 edge의 degree sum이 s로 일정하면 Q=L²+(s−4)L이며, star 등의 biregular graph도 여기에 해당한다. 따라서 비정규 그래프라는 이유만으로 차이가 생긴다고 가정하지 않는다. Edge degree sum이 달라지는 graph에서 잔차와 action 차이를 측정한다.

이 단계의 결과물은 graph/feature manifest, raw scale과 강도를 맞춘 결과표, 항등식 오차, operator action과 spectrum의 그림, 모든 case의 시간·메모리 측정값이다. 학습된 checkpoint는 만들지 않는다.

## Experiment 2 새 경로 규칙의 학습

Experiment 0/1의 검사 결과를 확인한 뒤 시작한다. 새 생성기의 입력은 두 엣지 차분뿐이다.

\[
\psi_p=[|g_1|+|g_2|,\ g_1^2+g_2^2,
\ g_1\odot g_2,\ |g_2-g_1|],
\qquad c_p=\operatorname{softplus}(\operatorname{MLP}(\psi_p))+\epsilon.
\]

입력을 이어 붙이면 4F차원이다. 경로를 뒤집어 (g1,g2)가 (-g2,-g1)이 되어도 입력은 변하지 않는다. Degree, PE, cycle, 기존 C, attention은 입력하지 않는다. 처음에는 scalar feature에 대한 `4 → m → 1` MLP를 사용하며 hidden layer는 하나다.

같은 graph/feature split에서 다음 task를 나누어 평가한다.

| Task | Target | 필요한 대조 |
| --- | --- | --- |
| A | Y=LX | First-order가 충분한 양성 대조 |
| B | Y=L²X | Polynomial이 충분한 양성 대조 |
| Fixed 확인 | Y=QX | Fixed-wedge의 양성 대조 |
| C | Y=A.T C2*(X) A X | 공유 path rule을 학습하는 대상 |

Task C의 첫 teacher는 scalar feature에서 `c*=sigmoid(lambda g1 g2)+epsilon`을 생성한다. 특징 분포는 표준정규로 고정하고 lambda와 epsilon을 독립 계약에 기록한다. Student가 degree를 입력받지 않으므로 teacher에도 degree를 사용하지 않는다. Indicator로 정의한 반전 가중치는 이후 stress test로 남기고, 처음에는 매끄러운 규칙을 사용한다.

연산 회수 실험은 단일 block, identity 특징 변환, 활성화 없는 구조를 사용한다. Pure teacher를 학생 모델로 표현할 수 있도록 일차 항의 계수를 항상 1로 고정한 혼합식은 쓰지 않는다. Gate의 원래 scale을 확인하는 실험에서는 beta=1로 두어 C2와 beta의 scale 보상을 분리한다. 필요한 scalar readout은 조건별로 실제 사용하는 항에만 두고 계수 0을 허용한다.

출력 Y의 오차를 주지표로 사용한다. Target C2와 path flux는 loss에 직접 제공하지 않고 진단에만 사용한다. 개별 C2는 유일하게 식별되지 않을 수 있다. q=0인 path는 가중치를 바꿔도 출력에 영향을 주지 않으므로 가중치 상관만으로 회수 성공을 판정하지 않는다.

Graph 수, train/validation/test 비율, m, lambda, epsilon, loss의 scale, optimizer, epoch, seed 수는 이 단계 시작 전에 독립 config로 확정한다. 이전 계획의 42/9/9, 노드 16〜32, AdamW, 200epoch 등으로 자동 설정하지 않는다. Target이 0일 때의 relative error 처리도 계약에 포함한다.

## Experiment 3 같은 규칙을 새 부분구조에 적용

Train은 20〜40노드의 sparse random graph다. Checkpoint를 고정하고 다음 차이를 구분한다.

- 같은 graph의 새로운 특징.
- 같은 생성 규칙의 새로운 graph.
- 같은 규칙으로 크기를 60〜100노드로 확대.
- 같은 크기에서 cycle, grid, tree+chord로 구조 변경.
- 구조와 크기를 모두 변경.

특징의 scale은 teacher와 train에 맞추고, amplitude 변화는 별도 stress test로 다룬다. 동일 graph의 특징 실현을 unseen graph test에 섞지 않는다. Graph를 먼저 생성·분할하고 seed와 content 중복을 검사한다.

## Experiment 4 공통 backbone과 표준 GCN

Synthetic를 검증한 뒤 Cora, CiteSeer, PubMed의 공식 split에서 평가한다. 독립 loader를 만들며 공식 raw cache는 checksum을 확인한 뒤 사용할 수 있다. 기존 Conductance adapter나 전처리 규약을 import하지 않는다.

통제된 backbone은 모든 조건에서 Z=HW, H'=sigma(PZ)를 공유하고 P만 바꾼다.

| 주 모델 | 사용하는 graph operator |
| --- | --- |
| MLP | Graph propagation 없음 |
| First-order | L |
| Polynomial-2 | L과 L² |
| Fixed-wedge | L과 Q |
| Learned-wedge | L과 A.T C2(H) A |

추가 파라미터 수를 맞춘 node MLP와 표준 이층 GCN을 추가한다. 표준 GCN은 자기 연결을 포함한 대칭 정규화 인접행렬을 사용하는 외부 sanity baseline이다. Combinatorial L을 사용하는 통제된 모델을 표준 GCN과 동일하다고 부르지 않는다.

Public task의 P를 raw T로 할지 I−tau T로 할지, 부호·분기 계수·정규화를 사전에 수식으로 확정한다. 첨부의 GCN+L²라는 이름만으로 forward를 결정하지 않는다. 이층 이차 연산은 최대 4hop, 이층 GCN은 최대 2hop이므로 연산의 추가 효과는 polynomial 대 wedge를 주로 비교한다.

Depth는 새 이층 backbone을 기본안으로 둔다. Width/dropout/optimizer 등은 원 GCN의 저자 코드에서 독립적으로 참조할 수 있다. 이를 이전 설정의 계승이나 원논문의 완전 재현으로 설명하지 않는다. 최종 설정과 공통 tuning 예산을 benchmark 시작 전에 독립 계약으로 확정하며, 설정이 없으면 실행하지 않는다.

용량 대조는 dummy parameter 없이 정확히 맞출 수 있다. Gate는 4F→m→1, node MLP는 F→2m→F로 만들고 두 모델 모두 첫 Linear에만 bias를 사용한다. 그러면 추가 파라미터 수가 각각 m(4F+2)로 같다. Node MLP 출력을 실제 특징 업데이트에 연결하고 추가 학습 scalar 수도 맞춘다. 파라미터 수가 같다는 것을 전체 표현력이 동일하다는 증명으로 해석하지 않는다.

Validation만으로 설정과 checkpoint를 선택하고 모든 조건을 고정한 뒤 test를 평가한다. Seed별 값과 paired 차를 보고한다. 반복 수와 학습 기간도 새 계약에서 근거와 함께 결정한다. 이전 698run 예산은 철회한다.

## 학습 모델의 다섯 가지 개입

1. C2=1.
2. C2=mean(C2).
3. Path 사이에서 C2 shuffle.
4. True wedge를 random distinct edge-pair로 치환.
5. 이차 분기 전체 제거.

Raw scale에서 1과 mean의 차이를 기록한다. 이후 mean 정규화나 operator 정규화를 채택하면 두 개입이 동일해질 수 있다. 그 경우 중복을 명시하고 원래 reference norm을 고정한 scale 진단과 구분한다.

Random pair는 signed pair operator T를 만들어 Arand=TB로 정의한다. B의 방향을 S로 바꾸면 T도 TS로 바꾸고, gate에 제공하는 두 차분의 부호도 함께 바꾼다. Pair와 sign을 manifest에 고정한다. Pair를 바꾸면 row norm도 달라지므로 true wedge의 sqrt(6)으로 row norm을 맞춘 대조도 함께 기록한다.

Pair 수, edge 사용 빈도, 공유 중심의 비율, node support, hop 거리, trace, operator norm을 기록한다. Random pair의 악화에는 이러한 변화도 포함되므로 연속성만의 인과 효과라고 단정하지 않는다. True gate를 고정하고 operator만 바꾸는 개입과, random pair에 맞춰 gate 입력도 바꾸는 개입을 구분한다. Random 대조군을 다시 학습한다면 고정 checkpoint 개입과 별도 표로 보고한다.

## 판정과 실행 조건

Fixed 단계는 학습 정확도의 우위를 판정하는 단계가 아니다. 정확한 항등식, polynomial로 환원되는 범위, 남는 연산 작용의 차이를 보고한다.

Learned 단계에서는 task A/B의 양성 대조가 작동하고, task C에서 fixed/polynomial보다 출력 오차가 작아지며 새 graph에서도 유지되는지 본다. Teacher로 만든 task의 회수 결과를 실제 분류 성능과 혼동하지 않는다. Public task의 성능 차, 용량 대조, 개입 결과는 각각의 근거로 보고한다.

본학습은 서버에서만 진행한다. GPU/MIG, RAM, CPU, 전체 N/E/P를 실측한다. 모든 path를 유지하며 tensor 연산, cache, exact chunking을 사용한다. 독립 graph는 disjoint-union으로 batch 처리한다. Physical batch는 후보들을 측정해 결정하며, 같은 학습 비교에서는 조건 간 batch와 update 수를 맞춘다.

실행한 같은 terminal에 현재 phase, case 또는 epoch, loss, metric, 처리시간을 표시하고 log에도 저장한다. 새 결과 directory에 실제 config/source/data hash와 완전한 재개 checkpoint를 보관한다. Experiment 0/1은 학습 checkpoint를 만들지 않는다. 현재 Experiment 0/1 구현과 별도 debug 검증을 완료했고, 서버의 full 고정 실험과 Experiment 2–4 학습은 아직 실행하지 않았다.

## 출처

- 이번 방향의 근거: 사용자 첨부 `626327d7-5b90-4965-a6c8-667344d1359b`의 본문.
- [원 GCN 논문](https://arxiv.org/abs/1609.02907).
- [GCN 저자의 학습 코드](https://github.com/tkipf/gcn/blob/master/gcn/train.py).
- [표준 GCNConv 정의](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
