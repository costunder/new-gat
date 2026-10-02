# 이전 경로 차분 전파 실험 계획

**이 문서는 더 이상 현재 계획이 아니다.** 사용자 첨부에 따라 [고정 연산부터 검증하는 새 계획](EXPERIMENT_PLAN_FIXED_FIRST.md)으로 대체했다. 아래의 기존 설정, 비교군, 698run 예산을 새 실험에 적용하지 않는다. 수정 전 원본은 `archive/EXPERIMENT_PLAN_20261002_before_fixed_first.md`에 보존했다.

2026년 10월 2일. 이 문서는 새 연구의 실험 설계다. 구현이나 학습 결과를 뜻하지 않는다. 본학습은 서버에서 수행한다.

질문은 **연속된 두 엣지의 특징 변화 관계를 보고 경로별 전파 강도를 학습하면, 고정 경로 연산이나 일반적인 두 홉 전파보다 예측과 새 그래프 일반화가 좋아지는가**이다. 고정 연산의 효과, 가중치 학습의 효과, 관계 정보를 입력한 효과를 구분한다.

트랙 이름은 `wedge_propagation`이다. 기존 Conductance 버전들과 output initialization 실험은 보존한다. 기존 ogbn-arxiv의 8층, hidden 256, 8 heads, 200 epoch 계약을 변경하지 않는다. 여기의 두 층은 첨부 설계에서 제안한 독립 연구의 두 특징 변환 층이다.

## 모델과 경로 정의

단순 무방향 그래프의 물리 엣지를 한 번씩 저장하고 발생행렬에서 자기 연결을 제외한다. 각 중심 j에서 서로 다른 두 이웃의 unordered pair를 한 번씩 사용한다. 삼각형 안의 경로도 포함한다. i와 k 사이에 엣지가 없어야 한다는 조건을 추가하지 않는다.

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,\qquad
q_p=(AZ)_p=z_i-2z_j+z_k.
\]

\[
g_1=z_j-z_i,\quad g_2=z_k-z_j,\quad q_p=g_2-g_1,
\qquad L_0=B^\top B,\quad L_2(C_2)=A^\top C_2A=B^\top R^\top C_2RB.
\]

R의 계수는 `R[p,e] = -B[e,j]`, `R[p,f] = -B[f,j]`이다. B의 방향을 바꾸면 R도 함께 바꾼다. (i,j,k)와 (k,j,i)는 동일한 경로다.

각 층은 다음과 같다.

\[
Z^{(\ell)}=H^{(\ell)}W^{(\ell)}+b^{(\ell)},\qquad
U^{(\ell)}=Z^{(\ell)}-\alpha_\ell\bar L_0Z^{(\ell)}
-\beta_\ell\bar L_2(C_2(Z^{(\ell)}))Z^{(\ell)}.
\]

첫 층 뒤에는 ReLU와 dropout을 적용하고 둘째 층의 U는 logits로 사용한다. 별도의 encoder, decoder, 출력 투영, LayerNorm 또는 skip module을 붙이지 않는다. 기본 식의 Z 항은 해당 대조군들이 공유한다. 주모델의 C1은 I로 고정한다. C2 생성기와 분류기는 분류 loss로 함께 학습한다.

양의 C2는 경로 차분에 penalty를 부여한다. Signed metric을 통한 반전 선호는 후속 연구로 남긴다. 이 연산을 임의 그래프의 물리적 곡률이나 사이클 정보의 완전한 복원이라고 설명하지 않는다.

## 경로 가중치 생성

각 층의 생성기는 모든 경로와 그래프에 공유되는 `6 → 256 → 1` MLP이며 첫 Linear 뒤에는 ReLU를 사용한다. Graph ID, node ID, path ID를 학습 입력으로 제공하지 않는다.

F를 현재 Z의 채널 수라고 하고 다음을 계산한다.

\[
a=\|g_1\|^2/F,\quad b=\|g_2\|^2/F,\quad
t=\frac{g_1^\top g_2}{\max(\|g_1\|\|g_2\|,10^{-8})}.
\]

공통 입력 다섯 개는 다음과 같다.

\[
[a+b,\ |a-b|,\ \log(1+d_j),\
\log(1+d_i)+\log(1+d_k),\
|\log(1+d_i)-\log(1+d_k)|].
\]

- `learned_wedge`는 여섯 번째 입력으로 방향 관계 t를 사용한다.
- `blind_wedge`는 여섯 번째 입력으로 `ab/(1+a+b)`를 사용한다. 크기의 함수이며 방향 관계를 제공하지 않는다.

입력 차원과 파라미터 수가 같다. 0 입력으로 사용되지 않는 파라미터를 만들지 않는다. Blind 생성기에 원래 노드 특징, 끝점 간 거리, q의 크기를 주면 관계를 다시 계산할 수 있으므로 제공하지 않는다. 기본 메시지 경로는 두 모델 모두 Z를 사용한다. Blind는 전체 모델에서 관계 정보를 제거한 모델이 아니라 C2 생성기 입력에 대한 대조다.

경로를 뒤집으면 (g1,g2)가 (-g2,-g1)로 변한다. 위 입력은 이에 불변이므로 `c_ijk = c_kji`가 보장된다.

그래프별 경로 수 P에 대해 양의 상대 가중치를 만든다.

\[
r_p=\operatorname{softplus}(s_p)+10^{-6},\qquad
c_p=\frac{P r_p}{\sum_{q\in G}r_q}.
\]

Mean(C2)=1이고 trace(L2)=6P이다. 전체 scale과 상대 경로 가중치를 구분하기 위한 선택이다. 서로 다른 C2가 동일한 연산자를 만들 수 있으므로 개별 가중치의 유일한 복구를 주장하지 않는다. 생성기의 출력 Linear는 Xavier gain .01로 초기화해 거의 균일한 가중치에서 시작한다.

## 정규화와 안정성

다음 안전한 상계를 사용한다.

\[
\rho_1=2\max_v d_v,\qquad
\rho_2=\max_v\left[
8\sum_{p:\mathrm{center}=v}c_p+
4\sum_{p:\mathrm{endpoint}=v}c_p\right],
\qquad \bar L_0=L_0/\rho_1,\quad \bar L_2=L_2/\rho_2.
\]

대칭행렬의 절대 행합에서 얻은 상계이며 lambda_max(Lk)≤rho_k를 보장한다. Dense 행렬이나 매 forward의 eigenvalue 계산이 필요 없다. Power iteration의 추정값을 보장된 상계로 취급하지 않는다. C2와 rho2 모두 autograd에 연결하고, 미분 검사는 max의 동점이 없는 입력에서 수행한다.

두 학습 scalar로 `(alpha,beta,self_budget)=softmax(u,v,0)`을 만든다. 초기값은 .25/.25/.5이다. Alpha와 beta는 층별 scalar이며 alpha+beta<1을 만족한다. 노드별 계수는 사용하지 않는다. 일차 전파만 사용하는 조건은 초기 alpha=.5인 sigmoid scalar 하나를 사용한다. 사용하지 않는 beta는 만들지 않는다.

P=0이면 이차 연산은 수학적으로 0이므로 해당 분기를 생략하고 그래프 수를 기록한다. 엣지가 없는 그래프의 일차 연산도 0이다. 오류를 임의의 0으로 대체하는 fallback은 금지한다.

보장은 현재 C2를 고정한 선형 부분에 적용된다. 입력 의존 C2와 W, 활성화를 포함한 전체 네트워크의 Lipschitz 보장이나 입력 의존 에너지의 단조 감소를 주장하지 않는다.

Mean(C2)와 rho2는 그래프 전체에 의존한다. 아래의 최대 4홉은 가중치를 고정한 희소 메시지 연산의 도달 범위다. 입력 의존 정규화 scalar를 통한 먼 특징의 의존성까지 4홉으로 제한한다는 뜻이 아니다. Blind와 adaptive_edge에도 그래프별 상대 가중치 정규화를 사용하고, 개입에서 정규화와 scale 변화를 함께 기록한다.

## 비교하는 여덟 조건

| 조건 | 각 특징 변환 층에서 하는 전파 | 판정 대상 |
| --- | --- | --- |
| first_order | I − alpha Lbar0 | 일차 전달의 기준 |
| repeated_first | (I − b Lbar0)(I − a Lbar0) | 동일한 W와 비선형 수로 두 단계 전달하는 효과 |
| polynomial2 | I − alpha Lbar0 − beta L0²/rho1² | 일반적인 이차 필터로 충분한가 |
| fixed_wedge | I − alpha Lbar0 − beta Lbar2(I) | 고정 경로 구조의 효과 |
| blind_wedge | I − alpha Lbar0 − beta Lbar2(C2) | 크기와 차수를 이용한 가중치 학습 |
| learned_wedge | 위와 같고 C2에 방향 관계 입력 | 관계 정보가 더하는 효과 |
| adaptive_edge | 학습 C1로 일차 전달을 두 단계 적용 | 적응적 엣지 가중치로 충분한가 |
| standard_gcn | 표준 GCNConv 두 층 | 표준 분류 모델과의 위치 비교 |

Repeated와 adaptive_edge는 층당 W 하나를 사용하며 두 전달 사이에 W나 활성화를 추가하지 않는다. a,b는 초기값 .25인 별도 sigmoid scalar다. Adaptive_edge는 해당 층의 Z에서 C1을 한 번 생성해 두 단계에서 공유한다. 생성기의 입력은 차분 제곱평균, 양 끝 특징 제곱평균의 합과 절댓값 차, log degree의 합과 절댓값 차, 차분 norm의 여섯 가지다. 좌우 교환에 불변인 `6 → 256 → 1` 양의 생성기를 쓰고 edge mean을 1로 정규화한다. Rho1은 2×최대 가중 차수다.

Wedge, polynomial, 두 단계 일차 전달은 두 층에서 최대 4홉이다. First_order와 표준 GCN은 최대 2홉이다. 주 비교는 learned 대 polynomial, fixed, blind다. 이층 GCN만 이겼다는 결과로 관계 정보의 효과를 결론내지 않는다.

고정 모델과 학습 모델에는 gate 파라미터 수 차이가 있다. 모든 조건을 parameter matched라고 부르지 않는다. Blind와 learned는 생성기 용량이 정확히 같은 대조다. 총 파라미터 수, trainable 수, 연산량을 조건별로 출력한다.

표준 GCN은 자기 연결을 포함한 대칭 정규화 인접행렬을 쓴다. Custom diffusion을 GCN으로 부르지 않으며, 이 공통 설정을 원논문 성능 재현이라고 주장하지 않는다.

## 수학과 구현 검사

Debug 전용 path, cycle, star, clique, 비정규 그래프에서 float64 dense 참조와 희소 구현을 비교한다.

1. A=RB, 엣지 방향 반전, 경로 뒤집기, 노드 재번호 부여의 불변성 또는 동변성을 확인한다.
2. trace(Z.T L2 Z)=sum c_p||q_p||², PSD, row sum0을 확인한다.
3. Fixed의 `L2=L0²+B.T diag(d_u+d_v−4)B`를 확인한다. Explicit와 고속 구현의 output 및 gradient를 비교한다.
4. d정규 그래프의 `L2=L0²+2(d−2)L0`, cycle의 `L2=L0²`를 확인한다. 이 일치를 새로운 표현력의 증거로 쓰지 않는다.
5. Mean(C2)=1, 양수 조건, 실제 lambda_max≤rho, alpha+beta<1, zero-operator 처리를 확인한다.
6. Chunk 크기를 바꿔도 output, loss, gradient가 일치하는지 확인한다.
7. 실제 분류 loss에서 gate가 optimizer에 포함되고 finite gradient와 parameter update가 생기는지 검사한다. C2 개입에 따른 logit 변화도 검사한다.

상대오차 기준은 float64 1e−10, float32 1e−5이며 0 근처에는 해당 절대오차도 사용한다. CUDA 병렬 합산 순서의 차이는 기록하고 구조적 오류와 구분한다. Debug 통과를 본학습이나 최종 성능 검증으로 표현하지 않는다.

## 합성 데이터에서 학습 규칙 확인

합성 실험은 의도한 연산을 학습할 수 있는지 보는 보조 실험이다. 실제 데이터의 우위 증거로 대신하지 않는다.

기존 S1 규모를 기준으로 ER형/RGG형 그래프를 train 42개, validation 9개, test 9개 사용한다. 노드 수는 16〜32이며, 그래프당 특징 실현 수는 6/3/3이다. Train 그래프에서는 새로운 특징 실현을 각 2개씩 만들어 `seen_graph_test`로 별도 보관한다. Data seed는 1729다. 기존 edge conductance target 대신 아래의 새로운 teacher를 사용한다.

- 일차 teacher: Y=H−.5 Lbar0 H.
- 고정 경로 teacher: Y=H−.25 Lbar0 H−.25 Lbar2(I) H.
- 적응 경로 teacher: Y=H−.25 Lbar0 H−.25 Lbar2(C2*) H. Raw weight는 softplus(2t)+1e−6이며, 그래프별 평균이 1이 되도록 정규화한다.

이 식별 실험의 student는 단일 propagation block이며, 특징 변환은 identity로 고정한다. ReLU와 dropout도 사용하지 않고 gate와 전달 scalar만 학습한다. 이는 연산 회수를 따로 확인하는 실험이다. Fixed/blind/learned 세 조건에서 seeds 0〜9, 200 epoch, AdamW lr .001, decay .01을 사용한다. Loss는 예측 업데이트와 target 업데이트의 MSE를 target 업데이트 에너지로 정규화한 값이다. Identity만 반환해 좋은 점수가 나오는 것을 방지한다.

일차 teacher의 beta=0은 strict softmax 학생의 경계값이므로, 정확한 scalar 회수보다 beta가 충분히 작아지고 업데이트 오차가 감소하는지 본다. 연산자들이 비례하는 그래프에서는 alpha/beta도 서로 보상할 수 있어, scalar 일치를 모든 그래프의 성공 기준으로 쓰지 않는다. Target 업데이트의 제곱 norm이 1e−12 이하인 예제는 학습에서 비정규화 absolute MSE를 사용하며, relative metric은 undefined로 기록한다. 해당 예제 수와 absolute error를 별도로 보고한다.

그래프별 업데이트의 상대 L2 오차를 평균한 값, 별도 특징에 대한 operator action error, alpha/beta를 주로 측정한다. 가중치 상관은 보조 지표다. 개별 C2는 유일하게 식별되지 않을 수 있으므로, target C2와의 불일치만으로 실패를 판정하지 않는다. Target C2나 edge/path flux를 loss에 직접 제공하지 않는다.

동일 checkpoint로 다음 OOD 조건을 따로 평가한다. 각 조건에는 독립 그래프 16개를 사용하며, 그래프마다 특징 실현 3개를 만든다.

| Cell | Family | 노드 수 | 달라지는 조건 |
| --- | --- | --- | --- |
| size | ER형/RGG형 | 48〜96 | 크기 |
| family | grid/barbell | 16〜32 | 생성 규칙 |
| family_size | grid/barbell | 48〜96 | 둘 모두 |

Graph/feature seed와 content hash를 manifest에 기록해 고정한다. 동일 그래프의 특징 실현을 `unseen_graph_test`에 나누어 넣지 않는다. Canonical edge content가 완전히 중복되는지 검사한다. 독립 seed나 내용 중복 검사를 그래프 동형성까지 배제한 증거로 해석하지 않는다.

## 실제 데이터와 학습 계약

기존 공식 데이터 adapter를 사용하며 split을 새로 만들지 않는다. 현재 로컬 data 폴더에서 실제 cache를 확인하지 못했으므로, 서버에서 cache, checksum, N/E/P를 확인한다. 자료가 없으면 정해진 데이터 준비 절차를 따른다.

| Dataset | Train / validation / test | Loss와 주지표 | 평가 의미 |
| --- | --- | --- | --- |
| Cora | 140 / 500 / 1000 nodes | CE, accuracy | 동일 그래프 내 분류 |
| CiteSeer | 120 / 500 / 1000 nodes | CE, accuracy | 동일 그래프 내 분류 |
| PubMed | 60 / 500 / 1000 nodes | CE, accuracy | 동일 그래프 내 분류 |
| PPI | 20 / 2 / 2 graphs | BCE, global micro-F1 | 독립 그래프에 고정 모델 적용 |

Citation은 전체 노드·엣지·경로를 관측하고 train mask에만 loss를 적용한다. PPI는 그래프별 정규화를 사용하며, test graph를 학습 batch에 섞지 않는다. Micro-F1은 logit>0을 기준으로 전체 node-label의 TP/FP/FN을 합산해 계산한다. Cora와 CiteSeer는 입력 차원과 라벨이 다르므로, 두 데이터 사이의 직접 이전을 동일 checkpoint의 일반화로 취급하지 않는다.

| 항목 | 값과 근거 |
| --- | --- |
| 특징 변환 | 두 층 input→256→classes. 첨부 설계에 따른 독립적인 두 층 비교 |
| hidden | 256. 기존 실험의 폭을 계승 |
| heads | Scalar path metric. 기존 8-head 모델과 별도로 정의한 연산 |
| dropout | .2, input과 첫 hidden에 적용. 기존 공통 설정을 계승 |
| 초기화 | W1 Xavier gain sqrt(2), W2 gain 1, bias 0 |
| optimizer | AdamW, decay .01, clip 5, fixed learning rate |
| epoch | 200. 매 epoch 전체 train을 처리하며 early stopping은 사용하지 않음 |
| precision | FP32, TF32 off |
| 최종 seeds | 0〜9, 조건 간 공통 |
| tuning seeds | 100〜102 |
| tuning lr | .0005/.001/.005, 모든 조건에 동일 예산 |
| 선택 | Best validation metric. 동률이면 낮은 validation loss, 다시 동률이면 이른 epoch |

Backbone/gate RNG를 분리해 gate 추가가 공통 W 초기화나 dropout stream을 바꾸지 않도록 한다. PPI batch 순서도 seed별로 대응시킨다. Tuning은 dataset·조건별로 validation만 사용한다. 모든 최종 설정을 고정한 뒤 test를 평가한다. 사전에 정의한 test 개입은 진단에만 사용하며 설정 선택에는 쓰지 않는다.

최종 실험은 8조건×4dataset×10seed=320run, tuning은 8×4×3lr×3seed=288run이다. 합성 식별은 3teacher×3조건×10seed=90run이다. 합계는 698run이며 debug·자원 계측은 별도다. Cora부터 진행해 로그·checkpoint·평가의 연결을 확인하고 나머지 dataset으로 이어간다. 올바른 구현으로 얻은 불리한 결과도 보존한다.

## 학습된 모델에 하는 개입

최종 checkpoint를 고정하고 eval mode에서 다음 개입을 적용한다.

1. Beta=0: 경로 분기 전체를 제거한다.
2. C2=I: 상대 가중치를 균일하게 만든다. Mean=1이므로 mean 치환과 같은 개입이다.
3. 중심별 C2 shuffle: 유효 경로를 유지하면서 가중치와 경로의 대응을 깨뜨린다.
4. 중심별 gate t 입력 shuffle: 방향 관계와 가중치 생성의 대응을 깨뜨린다.

Shuffle seeds는 1000〜1009다. 한 중심의 경로가 하나이면 shuffle로 바뀌지 않으므로, 변경된 경로의 비율을 기록한다. Rho2, 이차 message RMS, alpha/beta, logits, metric의 변화도 함께 기록한다.

재정규화한 표준 개입과 함께, 원래 message의 RMS를 맞춘 진단도 출력한다. Zero message는 RMS 맞춤 대상에서 제외하고 이를 명시한다. 이 진단은 scale 변화와 대응 파괴를 구분하기 위한 것이다. RMS를 맞춘 뒤에는 원래의 안정성 보장을 주장하지 않는다.

Random edge pair는 hop 거리·차수·연결성도 바꾸므로 첫 주대조군으로 쓰지 않는다. 먼저 valid wedge에서 대응을 깨뜨린다. 다른 구조 대응을 검사할 때는 보존할 통계와 변경할 통계를 먼저 정의한다. 재학습 대조와 frozen 개입의 결과는 별도로 해석한다.

## 판정과 통계

Finite loss/gradient와 optimizer update는 학습 연결의 증거다. C2의 분산이나 파라미터 변화만으로 연구의 성공을 선언하지 않는다.

주 비교는 learned 대 fixed, blind, polynomial2, adaptive_edge의 네 개다. Seed별 원값, 평균±표준편차, 같은 seed의 성능 차, seed 단위 bootstrap 95% CI를 출력한다. Bootstrap은 20000회, seed 20261002로 고정한다. 다중 비교를 추론할 때는 paired 차의 모든 2^10 sign permutation에 의한 양측 p값과 4비교×4dataset의 Holm 보정을 보고한다. 검정통계량은 paired 차의 평균 절댓값이다. 귀무가설 아래 차이의 부호가 교환 가능하다는 가정을 명시하며, 평균 동일성만으로 정확한 검정이라고 주장하지 않는다. Public mask의 10seed는 10개의 독립 데이터 분할이 아니다. PPI test graph가 둘이므로 seed CI를 graph 모집단의 불확실성이라고 부르지 않는다.

Learned 대 fixed와 blind의 개선이 복수 dataset에서 재현되고 polynomial/adaptive_edge 대조에서도 차이가 확인되면 경로 관계의 유용성을 지지한다. CI가 0을 가로지르는 차이는 우위 미확인으로 보고한다. 새 그래프 일반화는 PPI나 합성 held-graph 평가의 결과로 판정한다.

- Fixed만 개선: 고정 경로 필터의 효과.
- Learned와 blind가 비슷함: 크기·차수의 적응화 효과 가능성, 방향 관계의 추가 효과 미확인.
- Polynomial/두 단계 일차 전달로 같은 결과: 새 경로 metric의 추가 가치 미확인.
- C2 개입이 거의 영향 없음: 학습 연결 문제와 해당 task에서 분기를 쓰지 않는 상황을 구분.
- 합성에서만 개선: Teacher 연산의 학습 가능성에 한정.

## 자원과 진행 표시

실행 전에 현재 할당 GPU/MIG UUID·실제 VRAM·SM·CPU quota·RAM·cache와 N/E/P·degree 분포를 기록한다. 과거 서버는 약 9.5GiB A100 MIG였으나 현재 할당은 미확인이다. 로컬은 RTX 5070 Ti 16303MiB, 16 logical CPU, RAM 약 63.93GiB를 확인했다. 로컬 측정을 서버 ETA로 사용하지 않는다.

P=sum_j d_j(d_j−1)/2이며 전체 경로를 유지한다. Degree/path cap, 숨은 subset, 임의 sampling은 추가하지 않는다.

- Citation은 단일 전체 그래프의 노드를 병렬 처리한다. 독립 그래프를 하나씩 GPU에 보내는 직렬화와 구분한다.
- PPI는 disjoint-union과 size bucket으로 graph batch 2/4/8/20을 실측한다.
- Synthetic은 graph-example batch 8/16/32/64를 실측한다.
- Worker 2/4/8, pinned memory, persistent worker, prefetch를 측정한다. 데이터 전체가 GPU에 상주하면 DataLoader가 필요 없는 이유를 기록한다.
- Static edge/path index, degree, 고정 정규화를 cache한다. Cache의 RAM 점유를 측정하며 iteration마다 재구축하지 않는다.
- Tensor gather/scatter, exact chunking, activation checkpointing을 사용한다. Chunking 후에도 모든 중간 activation을 쌓아두지 않는다. 모든 path의 forward/backward를 유지한다.
- Fixed는 항등식의 O(EF) 고속 구현을 사용하고 explicit 참조와 비교한다. 일반적인 learned C2에는 이 복잡도를 주장하지 않는다.
- 여러 GPU가 실제로 할당되어 있으면 독립 run을 분배한다. 다른 사용자의 GPU를 빈 자원이라는 이유로 사용하지 않는다.

성능 비교에서는 모든 조건이 통과한 후보 중 처리량과 메모리 여유가 좋은 공통 physical batch를 선택한다. PPI와 합성의 조건 간 batch 구성, 순서, optimizer update 수를 맞춘다. 조건별 최적 batch를 사용하는 별도 처리량 표와 학습 성능 표를 혼동하지 않는다. Gradient accumulation은 기본 1이며, 실제 메모리 제약으로 필요해지면 physical/effective batch와 그 근거를 모두 기록한다.

자원 교정은 debug 출력에 독립 저장한다. Full model/full graph로 warmup 3step, 계측 10step, validation 1회를 측정한다. Physical/effective batch와 optimizer update 수를 명시한다. OOM에는 중간 tensor·재계산·streaming·분산을 검토하고 모델/그래프/epoch를 줄이지 않는다.

Data 준비, H2D, gate, 일차/이차 연산, backward, optimizer, validation, checkpoint 시간과 peak VRAM, CPU/RAM, 처리량을 측정한다. Plain step과 진단을 포함한 step을 분리한다. 각 run의 200epoch와 validation/checkpoint를 반영한 종료 예측은 이 계측 후에 출력한다.

같은 터미널에 dataset/condition/seed, epoch, train/validation loss, metric, epoch당 초, 측정에 근거한 ETA를 표시한다. 긴 전처리나 validation의 단계와 경과 시간도 표시한다. Child stdout/stderr를 log에 저장하면서 화면에도 출력한다. 별도의 monitor terminal을 필수로 하지 않는다.

가벼운 통계는 매 epoch, 상세 gate/branch 진단은 epoch 0/1/10/50/100/200과 최종 checkpoint에서 수행한다. 매 batch의 무거운 replay·dense 복원은 하지 않는다. 진단으로 추가된 시간을 따로 보고한다.

## 저장물과 실행 순서

Run마다 새 directory를 사용하고 실제 contract, source hash/Git commit/library version/data checksum, metrics.jsonl, terminal log, 자원 계측을 보관한다. Best와 last checkpoint에는 model뿐 아니라 optimizer, epoch/선택 상태, Python/NumPy/torch/CUDA RNG, sampler 위치를 저장한다. Scheduler를 사용하지 않는 계약도 명시한다. 재개 전에 source/data/config 일치를 검사한다. 실행 중인 서버 checkout에 git pull하지 않는다.

C2 전체 tensor를 매 epoch 복제하지 않는다. 전체 분포의 streaming 집계와 고정 probe의 완전한 값을 분리하고 probe를 전체 통계로 설명하지 않는다. Review package에는 수식 설명, 구현 source, test 결과, 실행 명령, 모든 seed의 결과표와 개입·paired 통계를 담는다.

실행 순서는 ①새 모델과 debug 검사 구현 ②서버 자원 교정 ③합성 식별/OOD ④Cora tuning/최종 10seed/개입 ⑤CiteSeer·PubMed·PPI ⑥성능과 비용 종합이다.

Arxiv는 다음 규모 검증에서 원그래프 전체 P와 exact 계산의 시간·메모리를 먼저 측정한다. 새 arxiv depth/head/sampling 계약을 명시하기 전에는 이층 citation 결과를 기존 8층 연구와 합치지 않는다. Faster local attention이라는 주장도 측정 전에는 하지 않는다.

이번에 완료한 것은 이 설계 문서다. 새 모델 구현, 단위테스트, 실제 데이터 smoke test, 전체 학습/평가는 미실행이다.

## 근거

- [사용자 공유 설계](https://chatgpt.com/share/6abe9556-89dc-83e8-b878-d65822319b26)와 첨부 리뷰.
- 기존계약: `experiments/output_init_ablation/README.md`.
- Split과 adapter: `research/conductance_gat/datasets.yaml`, `benchmark_data.py`.
- 합성 규모 기준: `research/conductance_gat/paper_data.py`의 S1/S2. 이번 teacher와 OOD cell은 별도로 정의했다.
- [Planetoid 공식 설명](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html).
- [PPI 공식 설명](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.PPI.html).
- [표준 GCNConv 공식 정의](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
