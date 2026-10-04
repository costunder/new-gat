# 로컬 에너지·로컬 사이 관계: 전체 실험 이력

작성 기준: 2026-10-04. 이 문서는 `research/local_energy_relations/` 계열의 설계,
실제 코드, 사용자가 제출한 서버 원문을 대조한 기록이다.
초기 고정 감사 → 수신 합·복원 감사 → E/J 분류 → 주입 위치별 재학습 순서다.
이전 Conductance·wedge 계열의 버전 이름과 혼동하지 않는다.

**현재 모델은 로컬 내부 에너지와 로컬 사이 관계를 스칼라 특징으로 만들고,
학습 벡터를 곱해 기본 노드 전파에 더하는 후보 모델이다.**
큰 가중 라플라시안의 대각 블록과 교차 블록을 직접 조립해 전파하는 모델을 구현한 것은 아니다.
고정 감사의 계산 성공, 분류기의 학습 성공, 기본 모델 대비 개선을 각각 구분한다.

## 1. 전체 범위와 서버 증거

| 단계 | 실제 질문 | 코드 | FULL 계약 | 제출받은 서버 증거 |
| --- | --- | --- | --- | --- |
| 1. 초기 고정 감사 | 내부 에너지·로컬 사이 관계·전달 범위가 실제 발생행렬 계산과 맞는가? | `local_energy_relations/` 최상위 | 합성 198 + citation 3 = 201 graphs, 합성 scalar 입력 3,168개, 두 C·세 상태, 학습 0 | completion JSON + 전체 요약 원문 |
| 2. 수신 집계·복원 | 여러 송신 값을 합한 뒤 실제 메시지와 E/J를 결정할 수 있는가? | `receiver_aggregation/` | 같은 201개 입력, 두 C·세 상태·다섯 관측 조건, 학습 0 | completion JSON + 전체 요약 원문 |
| 3. E/J 분류 | 같은 기본 전파에 E·J·E+J를 더해 재학습하면 예측이 좋아지는가? | `prediction/` | citation 3 × 8조건, 336 runs × 500 epoch = 168,000 updates | completion JSON + 전체 요약 원문, 별도 층 제거 12행 |
| 4. 위치별 재학습 | E/J를 첫 층·출력층·두 층에 넣으면 무엇이 달라지는가? | `placement/` | citation 3 × 20조건, 840 runs × 500 epoch = 420,000 updates | 전체 요약 원문만 제출됨 |

단계 1–3은 **제출받은 완료 기록과 요약을 확인**했다.
서버의 모든 원시 CSV·입력 snapshot·checkpoint를 가져와 hash와 평가를 독립적으로 재실행한 상태는 아니다.
단계 4는 요약이 540 tuning / 300 final runs와 계약 420,000 updates를 보고한다.
계획한 표의 행 수와 수치는 확인했으나 `completion.json`·실제 결과 경로·source digest·원시 CSV·checkpoint가 없다.
따라서 단계 4의 840회 완료를 원시 학습 파일로 다시 검증했다고 쓰지 않는다.

각 구현의 DEBUG 검사와 서버 FULL 결과는 별도다. DEBUG fixture의 정확도·시간·VRAM을
실제 citation 성능이나 A6000 본학습 자원 수치로 제출하지 않는다.

## 2. 모든 단계에 공통인 그래프와 수식

원래 단순 무방향 그래프의 중심 v에 대해 `S_v={v}∪N(v)`를 선택한다.
S_v 안에 양 끝이 있는 **모든 원래 엣지**를 포함하므로 이웃끼리 엣지도 들어간다.
물리 엣지는 원래 node ID 순서로 방향을 통일한다.

| citation | 전체 노드 N | 물리 엣지 M | 입력 특징 D | 클래스 K | train / validation / test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 5,278 | 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 4,552 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 44,324 | 500 | 3 | 60 / 500 / 1,000 |

입력은 row-sum 정규화한 citation 특징이다. 감사에서는 float32 loader 입력을 float64로 승격했다.
분류는 float32·전체 public split의 transductive 조건이다.
Validation/test의 특징과 연결은 전체 그래프 forward에 포함하지만 분류 CE는 train mask에만 적용한다.
모든 노드·엣지·로컬·원래 인접 중심 쌍의 양방향·전체 특징 채널을 사용한다. Sampling ratio는 1이다.

B_v는 엣지×로컬 노드 발생행렬, R_v는 전역 특징 Z에서 로컬 행을 고르는 행렬이다.

\[
Z_v=R_vZ,\quad g_v=B_vZ_v,\quad q_v=C_vg_v,\quad
d_v=B_v^\top q_v=L_vZ_v,\quad L_v=B_v^\top C_vB_v.
\]

C_v는 현재 이 계열에서 **학습하지 않는 양의 대각 행렬**이다.

- `unit`: c_e=1.
- `local_degree`: 로컬 엣지 (a,b)에서 c_e=2/(deg_v(a)+deg_v(b)). degree는 해당 induced local에서 계산한다.

### 내부 에너지 E

\[
E_v=\operatorname{tr}(Z_v^\top L_vZ_v)
   =\sum_{e\in E_v}c_{v,e}\|g_{v,e}\|_2^2.
\]

E는 c가 한 번 들어가는 가중 라플라시안 이차 에너지다.
메시지 norm 제곱은 \(\|q_v\|_F^2=\sum_e c_{v,e}^2\|g_{v,e}\|^2\)이므로 일반적인 C에서는 다른 양이다.
현재 코드에 고유벡터 분해·eigendecomposition을 실행하는 별도 spectral 층은 없다.
발생행렬로 계산한 에너지가 정확한 이차형식이라는 사실과 그 구현 여부를 구분한다.

### 서로 다른 로컬의 관계 J

B_v의 열을 원래 전체 노드 좌표로 확장한 발생행렬을 \(\bar B_v\)라 하면

\[
J_{vu}=\operatorname{tr}(q_v^\top\bar B_v\bar B_u^\top q_u)
      =\sum_{a\in S_v\cap S_u}\langle d_v[a],d_u[a]\rangle.
\]

두 로컬에서 같은 원래 노드의 행을 연결하는 행렬을 M_vu라 쓰면
\(J_{vu}=\operatorname{tr}(Z_v^\top L_v M_{vu}L_uZ_u)\)다.
즉 현재 J는 **각 로컬에서 라플라시안으로 얻은 d끼리 공통 노드에서 내적**하는 관계다.
두 로컬 입력을 별도로 보면 쌍선형식이며, 둘 다 같은 전역 Z의 선택이라는 제약 아래에서는 전역 Z의 이차 함수다.

큰 행렬 Q를 집합별 블록으로 나누면 내부 항과 교차항으로 전개할 수 있다는 일반 항등식은 맞다.
그러나 현재 J를 원래 큰 가중 라플라시안의 off-diagonal 블록이라고 놓고,
E와 J로 그 라플라시안을 정확히 재조립했다고 주장할 근거는 없다.
실제 모델은 아래 5절의 **E/J 스칼라 lift 추가**를 사용한다.
J는 signed 관계이며 항상 양의 에너지나 전역 PSD 연산자라고 강제하지 않는다.

## 3. 실험 1 — 초기 고정 감사

### 질문과 실제 계산

내부 이차 에너지, 같은 물리 엣지 내적 `J_shared`, 공통 노드 내적 `J_node`,
서로 다른 incident 엣지 기여 `J_distinct`를 확인한다.

\[
J_{\rm node}=2J_{\rm shared}+J_{\rm distinct}.
\]

같은 물리 엣지의 발생행렬 행끼리 내적이 2이므로 위 분해가 나온다.
세 항을 모두 기록하지만 독립 정보 세 가지가 아니다.
공통 노드로 d를 복사할 때 남는/빠지는 부분과 retained/boundary/omitted flow를 기록한다.
하나의 물리 엣지가 여러 로컬에 포함되면서 생기는 에너지 중복도 별도로 검사한다.

상태는 학습 층이 아니라 원래 물리 unit Laplacian의 고정 reference다.

\[
H^{(0)}=X,\quad H^{(t+1)}=H^{(t)}-\tau L_{\rm physical}H^{(t)},
\quad \tau=1/(2d_{\max}),\quad t=0,1.
\]

모든 로컬이 같은 H0/H1/H2에서 자기 행을 선택한다.
같은 상태의 인접 로컬 관계와 H0→H1/H1→H2 관계, 동일 로컬의 시간 관계를 따로 기록한다.
전달 bookkeeping으로 복사한 d를 H_next에 넣지는 않는다.

실제 q의 Euclidean cycle 성분과, 알려진 양의 C·전체 로컬 d를 쓴 제한된 복원을 구분한다.

\[
q_{\rm cycle}=q_v-B_v(B_v^\top B_v)^\dagger d_v,
\quad B_v^\top q_{\rm cycle}=0,
\qquad \hat q_v=C_vB_vL_v^\dagger d_v.
\]

현재 q=CBZ 제약에서는 \(\hat q=q\)다. cycle 성분이 있다는 것만으로 복구 불가능한 손실이라고 하지 않는다.
부분 노드 관측이나 비선형 압축 이후의 복원을 검사한 식이 아니다.

### 규모와 완료 기록

합성 그래프는 cycle/star/grid/ER/tree/tree+chord, N=20/30/40/60/80/100의 기존 198개다.
모든 16 scalar realization을 사용해 3,168개 입력을 계산했고 실제 citation 3개를 추가했다.
Trainable parameter·optimizer update는 0이며 분류기는 없다.

서버 결과 경로는 `/home/aicompetition07/new-gat/results/local-energy-20261004-002815`다.
제출된 completion은 FULL201·실제 citation3·전체 입력 보존을 보고하며 시간은 563.826초다.
Raw records는 local 1,199,952 / relation 17,961,240 / transfer 3,592,248 / temporal 799,968행이다.
Source digest는 `6342eb5da8d8ca2e54557d3058605924db63946290e2b70b8d394f81429b0265`다.

| H0 실제 citation | unit cycle 제곱 비율 | local_degree cycle 제곱 비율 | local_degree의 q 복원 상대오차 최대 |
| --- | ---: | ---: | ---: |
| Cora | 1.40e−19 | 1.2969% | 1.6750e−9 |
| CiteSeer | 6.70e−20 | 0.5706% | 6.3795e−10 |
| PubMed | 2.56e−19 | 1.4268% | 1.0845e−9 |

Unit에서 q=BZ이므로 Euclidean cycle 성분은 수치 오차 수준이다.
Local_degree에서는 실제 cycle 투영이 양수지만 알려진 C·전체 d로 q를 복원했다.
Citation 전체 18행의 q 상대 복원오차 최대는 4.39145e−9다.
따라서 이 결과는 현재 constrained q의 정확한 복원 가능성과 일치한다.

로컬 에너지를 단순 합하면 물리 엣지를 반복 계산한다.
예를 들어 Cora unit H0의 raw local 합은 1,891.0446, 물리 에너지는 649.61352다.
각 occurrence를 물리 엣지 등장 횟수로 나눈 보정은 unit 물리 에너지와 일치했다.
Local_degree 보정은 물리 엣지별 평균 C의 에너지이며 unit 에너지와 같지 않다.

이 단계는 관측·전달 범위와 항등식을 검사했다. 분류 개선이나 학습된 C의 일반화를 검사하지 않았다.
최상위 `VERIFICATION.md`의 “FULL 아직 실행하지 않음”은 서버 원문 제출 이전 구현 기록이다.
이번 이력은 이후 제출된 FULL 완료 기록을 함께 반영한다.

## 4. 실험 2 — 실제 수신 합과 복원

송신 v→수신 u의 공통 노드 a에 대해

\[
R_{(v,u,a)}=d_v[a],\qquad
Y_{(u,a)}=\sum_{v\sim u:\,a\in S_v}R_{(v,u,a)}.
\]

R은 송신 식별자가 있는 receipt, Y는 수신 local·원래 노드별 합이다.
관측 조건은 `tagged / sum / sum_within / sum_between / sum_both`다.
같은 전체 201개 snapshot·두 C·세 H 상태를 쓰며 학습은 없다.

임의의 receipt 공간에서는 합 S의 kernel이 있고, 직접 보이지 않는 성분은

\[
R_{\rm hidden}=R-S^\top\operatorname{diag}(m)^\dagger Y
\]

다. 하지만 실제 R은 알려진 양의 C와 공통 전역 H에서 생성된다.
전체 Y의 수신 중심 행들이 양의 directed Laplacian을 포함하므로
H를 원래 연결성분별 상수까지 결정한다. 그 상수는 q/R/E/J에 영향을 주지 않는다.
따라서 이 실제 제약에서는 채널당 rank=N−연결성분 수다.
이는 수학적 결과이며 citation 전체 SVD를 실측한 값이 아니다.

| H0 | C | 직접 숨는 receipt 제곱 norm 비율 | Y에서 실제 q의 상대 복원오차 |
| --- | --- | ---: | ---: |
| Cora | unit | 34.35% | 2.70996e−7 |
| Cora | local_degree | 10.33% | 6.796091e−8 |
| CiteSeer | unit | 28.16% | 3.305544e−7 |
| CiteSeer | local_degree | 9.48% | 8.550849e−8 |
| PubMed | unit | 22.69% | 6.815293e−7 |
| PubMed | local_degree | 12.61% | 1.197799e−7 |

18개 citation 요약에서 실제 수신 합 상대 잔차 최대 9.992807e−9,
q 상대오차 최대 6.815293e−7, centered H 최대 8.026027e−6이었다.
Y로 복원한 H에서 E/J를 재계산하면 E 최대 상대오차 9.961601e−9,
동일·연속 단계의 개별 관계를 포함한 J 최대 상대오차 1.831518e−8이었다.
Decoder는 E/J를 입력에 쓰지 않고 **Y만으로** 복원했다.
E/J 추가에 따른 복원 개선 실험으로 읽지 않는다.

Completion은 FULL201, 3,168 scalar 입력, raw 입력 지표 19,026행,
복원/조건/관계 요약 1,206 / 6,030 / 6,030행, elapsed 3,464.85초를 보고한다.
원문과 상세 분석은 [수신 집계 서버 결과](../RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md)에 있다.

**관측 합에서 직접 숨는 비율과 영구 정보 손실률은 다르다.**
현재 모든 수신을 전역적으로 이용한 반복 역복원 성공은 제한된 깊이의 국소 GNN 효과를 보장하지 않는다.
E/J가 전역 Y의 함수라도 제한된 모델에서 유용한 비선형 특징일 가능성은 별도 분류로 확인할 수 있다.

## 5. 실험 3 — 두 층에 E/J를 넣는 실제 분류

### 구현한 모델

기본 전파는 receiver-center 행에서 만든 positive neighbor transition P_C를 두 번 적용한다.
Unit의 엣지 가중치는 `1+공통 이웃 수`이며 삼각형이 없으면 일반 random-walk 평균과 같다.
Local_degree는 송신 local의 C를 합하므로 방향에 따라 가중치가 달라질 수 있다.
고립 행은 자기 특징을 유지한다.

\[
U_{\rm base}=(1-\alpha)Z+\alpha P_C^2Z,
\quad\alpha=\operatorname{sigmoid}(a),\quad\alpha_{\rm init}=0.5.
\]

현재 F채널에서 내부 에너지와 관계를 topology 상수만으로 정규화한다.

\[
e_v=\frac{E_v}{2F\sum_e c_{v,e}},\quad
\widehat J_{vu}=\frac{J_{vu}}{F\sqrt{\operatorname{tr}(L_v^2)\operatorname{tr}(L_u^2)}},
\quad j_u=\frac1{|N(u)|}\sum_{v\sim u}\widehat J_{vu}.
\]

Feature RMS는 쓰지 않으며 J 부호를 유지한다. 고립 local의 e/j는 0이다.
각 활성 층에서 현재 Z로 E/J를 새로 계산한 뒤

\[
U=U_{\rm base}+e\,w_E^\top+j\,w_J^\top
\]

로 더한다. **채널·엣지별 정보를 각각 다음 층에 보존하는 구현이 아니라 두 scalar의 lift다.**
W·α·활성 w_E/w_J를 train CE로 학습하고 C·topology 분모·P는 고정한다.
Lift는 0으로 초기화하며 비활성 parameter는 만들지 않는다.
Auxiliary reconstruction loss·teacher·전역 역복원·learned C는 없다.

두 층은 `D→64→K`, projection 직전 dropout 0.5, 첫 층 뒤 ReLU, 둘째 층 logits다.
한 층의 공통 기본 support는 최대 두 홉이고 두 층은 최대 네 홉이다.
현재 예측의 J는 **같은 층의 서로 다른 local** 관계다.
초기 감사의 H0→H1 같은 서로 다른 단계의 관계를 학습 forward에 추가한 모델이 아니다.

### 학습·평가 계약

두 C × `base / within(E) / between(J) / both(E+J)`의 8조건을 새로 학습했다.
위 표의 citation 3개 전체 public 그래프를 사용했다.
Adam·500 epoch/run·float32·TF32 off, projection에만 weight decay 5e−4를 적용한다.
Early stopping·scheduler·clipping은 없다.
LR 후보 0.001/0.003/0.01, tuning seeds 101/202/303, final seeds 11/23/37/53/71이다.

- Tuning: 3데이터×8조건×3LR×3seed=216 runs.
- Final: 3데이터×8조건×5seed=120 runs.
- 합계: 336 runs, 168,000 독립 optimizer updates.

Checkpoint는 validation CE 최소 → validation accuracy 최대 → 가장 이른 epoch 순이다.
LR은 tuning seed의 selected validation CE 평균 최소 → 작은 LR 순이다.
모든 final checkpoint 선택을 잠근 뒤 test를 평가하며 tuning에서 test metric은 쓰지 않는다.
같은 고정 split·동일 final seed의 paired t 95% 구간이며 다중 비교 보정은 없다.

### 실제 FULL 결과

제출된 completion은 실제 데이터 사용, 336회·168,000 updates,
split metric 360행 / 제거 metric 1,350행 / branch 진단 1,140행,
frozen update 0, elapsed 2,263.15초를 보고한다.
[서버 원문](../evidence/local_prediction_server_full_20261004.txt)과
[분석](../LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md)에 보존했다.

| 데이터 | C | base 정확도 % | E−base pp | J−base pp | E+J−base pp |
| --- | --- | ---: | ---: | ---: | ---: |
| Cora | unit | 76.22 | −0.30 | −0.60 | −0.18 |
| Cora | local_degree | 74.34 | +0.10 | −0.86 | +0.22 |
| CiteSeer | unit | 67.24 | −1.24 | −2.84 | −2.18 |
| CiteSeer | local_degree | 64.90 | −0.26 | −2.12 | −1.28 |
| PubMed | unit | 79.60 | −0.32 | −0.26 | −0.16 |
| PubMed | local_degree | 78.94 | +0.10 | +0.18 | +0.10 |

같은 C base 대비 18개 accuracy 구간 전체가 양수인 비교는 없었다.
모든 활성 lift의 checkpoint norm은 양수이며 제거 개입으로 CE/accuracy가 달라졌다.
즉 모델은 학습·사용했지만 기본 모델 대비 정확도 개선을 확인하지 못했다.
PubMed J의 CE는 두 C에서 약 −0.0025 개선 구간을 보였으나 accuracy 구간은 0을 포함했다.
CiteSeer J는 두 C에서 accuracy와 CE 모두 악화 구간이었다.

별도 제출된 E+J 층 제거에서는 출력층 제거의 평균 test CE 변화가 첫 층보다 컸다.
Cora·CiteSeer의 출력층 제거는 악화, PubMed의 두 층 제거는 각각 개선했다.
첫 층 제거 후 다음 층 특징과 E/J를 다시 계산하므로 층별 값을 더해 독립 기여도를 만들지 않는다.
Frozen 제거와 처음부터 그 분기 없이 재학습한 base는 다른 모델이다.

## 6. 실험 4 — E/J 주입 위치별 재학습

각 C에서 base를 한 번 학습하고, `E / J / E+J × hidden / output / all`을 비교했다.
Hidden은 첫 층만, output은 출력층만, all은 두 층이다. 두 backbone 층은 모든 조건에서 유지한다.
기존 base/all도 새 실험 안에서 다시 학습해 공통 계약으로 비교한다.
각 위치의 현재 projected Z와 F에 맞춰 같은 E/J 수식을 재계산한다.

| 단계 | 계산 | FULL budget |
| --- | --- | ---: |
| Tuning | 3데이터×20조건×3LR×3seed | 540 runs / 270,000 updates |
| Final | 3데이터×20조건×5seed | 300 runs / 150,000 updates |
| 합계 | run당 500 epoch | 840 runs / 420,000 updates |

데이터·깊이 2·hidden 64·LR·seed·선택 규칙은 앞선 예측 실험과 같다.
활성 layer에만 벡터를 등록한다. 한 항의 추가 parameter 수는 hidden=64, output=K, all=64+K다.
따라서 위치 비교에는 특징 공간과 추가 용량 변화가 함께 들어간다.
활성 층의 E/J 제거만 평가하며 비활성 층의 no-op 제거를 중복 기록하지 않는다.

### 받은 요약에서 확인한 결과

전송된 표는 조건별 요약 60행, 상호작용 18행, paired 비교 174행,
두 층 branch 관측 120행, 활성 제거 범위 150행으로 계획된 집계 범위와 맞았다.
원문은 모든 coverage 검증과 계약 budget을 보고하지만 completion·원시 파일은 제출되지 않았다.
[전체 원문](../evidence/local_placement_server_full_20261004.txt)과
[검토 범위·분석](../LOCAL_PLACEMENT_SERVER_FINDINGS_20261004.md)에 있다.

| 주요 재학습 비교: 조건−같은 C base | Δaccuracy pp [95% 구간] | ΔCE [95% 구간] |
| --- | --- | --- |
| Cora unit, 첫 층 E | +0.0800 [−0.35369, +0.51370] | −0.0027974 [−0.0040867, −0.0015082] |
| Cora local_degree, 첫 층 E | +0.0600 [−0.14777, +0.26777] | −0.0023179 [−0.0034673, −0.0011684] |
| CiteSeer unit, 출력층 J | −2.78 [−3.5914, −1.9686] | +0.017584 [+0.0011707, +0.033997] |
| CiteSeer local_degree, 출력층 J | −2.08 [−3.9761, −0.18387] | +0.021422 [+0.012266, +0.030577] |
| PubMed local_degree, 출력층 J | +0.24 [+0.051693, +0.42831] | −0.0026657 [−0.0046963, −0.00063507] |

같은 C base 대비 54개 accuracy 비교에서 구간이 완전히 양수인 것은 PubMed 출력층 J 한 개다.
이 후보의 정확도 79.18%는 unit base의 79.60%보다 낮다.
무보정 다중 비교 중 한 후보이며 전체 조건의 최종 우승 모델로 선택한 결과가 아니다.
첫 층 E+J 여섯 조건의 accuracy 구간은 모두 0을 포함한다.
위치만 바꾸면 세 데이터에서 보편적으로 좋아진다는 결론은 나오지 않았다.

60개 조건의 선택 LR은 0.01, train accuracy는 100%다.
96개 활성 lift의 checkpoint 평균 norm은 모두 양수다.
Branch/base norm은 대략 첫 층 E 0.9–3.3%·J 0.33–1.68%,
출력층 E 5.2–12.0%·J 3.2–16%다. 이 수치는 정보 보존율이나 정확도 기여율이 아니다.

PubMed local_degree 출력층 J 모델에서 frozen J 제거는 CE를 −0.0058299 낮췄다.
그 모델을 별도로 재학습한 결과는 base보다 좋았다는 표와 모순되지 않는다.
J를 사용하며 학습한 projection을 고정하는 개입과 J가 없는 모델의 전체 재학습은 다른 비교다.

Resource 후보 시간 범위 0.0064513–2.6678초는 모든 조건·packing·chunk 후보의 epoch 시간이다.
전체 elapsed·선택 GPU·VRAM·실제 packing은 첨부되지 않아 대표 학습 속도로 요약하지 않는다.

## 7. 코드 검증과 현재 한계

현재 위치 실험의 로컬 검증은 신규 202개 + 기존 prediction 111개, 합계 313개 테스트 통과 기록이 있다.
CUDA DEBUG는 360 runs / 1,080 updates, fixture 3개, actual_data=false로 완료했다.
완료 기록·source/입력 보존·active frozen coverage·zero-update resume·그림 표시를 확인했다.
이 검증은 구현 연결과 계약 검증이며 서버 FULL 통계의 원시 재검증을 대신하지 않는다.
세부 기록은 [구현 검증](../LOCAL_PLACEMENT_VERIFICATION_20261004.md)에 있다.

현재 자료에서 아직 확인하지 않은 항목은 다음과 같다.

- Learned C·GATv2 형태의 학습 attention 효과. 현재 C는 고정이다.
- 독립 그래프·새 split·부분구조 sampling 일반화. 현재 분류는 한 public split의 전체 그래프다.
- 일부 수신만 남기는 경우, source별 독립 상태, 비선형 압축 후의 실제 복원 가능성.
- 원래 메시지 전체를 다음 층으로 보존하는 모델이나 cycle 성분의 예측 활용.
- 큰 가중 라플라시안의 실제 대각·교차 블록을 조립한 연산자의 PSD·안정성·예측 효과.
- E/J 용량을 맞춘 대조 및 무보정 54개 비교의 독립 재확인.
- 기존 GNN/GATv2와 같은 학습 계약의 최종 성능·처리시간 대조 및 논문 신규성 검토.
- 최신 placement의 completion·seed별 CSV·checkpoint·source digest를 이용한 서버 원시 검증.

후속 후보로 **첫 층에서 채널별 E/J를 보존하는 대조**를 검토할 수 있다.
현재는 채널 평균 e/j 한 숫자에 벡터를 곱한다.
같은 64채널·64개 branch parameter를 유지한 `e_f w_E,f`, `j_f w_J,f` 대조는
요약 방식의 효과를 검사하는 설계 제안이다. 현재 이 문서가 설명하는 완료 실험에는 구현·학습하지 않았다.
스칼라 요약이 부진의 원인이라고 확정한 결과도 없다.

## 8. 코드와 원문을 읽는 위치

- [초기 고정 감사 수식](../../research/local_energy_relations/MODEL_MATH.md),
  [설계](../../research/local_energy_relations/EXPERIMENT_DESIGN.md), `operators.py`, `topology.py`, `study.py`.
- [수신 합·복원 수식](../../research/local_energy_relations/receiver_aggregation/MODEL_MATH.md),
  [서버 분석·원문](../RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md).
- [예측 수식](../../research/local_energy_relations/prediction/MODEL_MATH.md),
  `prediction/operators.py`, `prediction/model.py`, `prediction/training.py`,
  [336회 서버 분석·원문](../LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md).
- [현재 위치 모델 수식](../../research/local_energy_relations/placement/MODEL_MATH.md),
  [실험 계약](../../research/local_energy_relations/placement/EXPERIMENT_DESIGN.md),
  `placement/model.py`, `placement/study.py`, `placement/report.py`,
  [840회 제출 요약의 확인 범위](../LOCAL_PLACEMENT_SERVER_FINDINGS_20261004.md).

초기 서버 [완료·요약 원문](../evidence/local_energy_server_full_20261004.txt)은
사용자가 첨부한 `af5804c5-a6ba-4cae-88da-7b0d0734be14/붙여넣은 텍스트.txt`에서 보존했다.
수신·예측·위치별 서버 원문도 `docs/evidence/`에 있다.
원시 결과 디렉터리 전체와 원문 요약은 별개의 자료다.
