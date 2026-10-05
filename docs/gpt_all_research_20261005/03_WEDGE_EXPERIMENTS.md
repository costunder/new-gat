# Wedge 경로 연산: 전체 실험·수식·결과

작성 기준: 2026-10-05. 대상은 `research/wedge_propagation/`의 독립 경로 연산 트랙이다.
이후의 로컬 그래프 E/J 실험은 별도 문서를 따른다. 여기의 학습 C는 물리 엣지 가중치가 아니라
**두 엣지가 이어진 wedge 행에 붙는 가중치**다. 서로 다른 트랙의 C, 정규화, 정확도를 같은 모델의 버전처럼 합치지 않는다.

## 증거와 연구 범위

이 문서는 [이전 전체 정리](../gpt_experiments_20261004/02_WEDGE_HISTORY.md)를 기반으로,
실험별 SERVER 원문·현재 코드·전체 설정을 함께 확인한 새 전달 문서다.
**합성 FULL 학습은 지정된 전체 합성 데이터의 실제 과학 실험**이다.
별도 DEBUG fixture나 공식 citation 분류 FULL과 구분한다.
동일한 서버 run을 여러 발췌에서 받았다고 독립 반복 수를 늘리지 않는다.

## 1. 이 트랙에서 확인한 것

1. 고정 경로 연산 Q는 일부 그래프에서 L과 L²의 조합으로 환원되고, 다른 그래프에서는 잔차가 남았다.
2. 공유 경로 C 생성기는 지정한 합성 teacher의 메시지를 학습했고, 새 그래프·특징에서도 낮은 회수 오차를 유지했다.
3. C 입력의 RMS 정규화는 입력 배율에 대한 안정성을 개선했지만, 평균 teacher 회수 오차를 개선하지 않았다.
4. 실제 노드 분류에서는 학습 C가 고정 C=1보다 유리하다는 근거를 얻지 못했다. 표준 GCN이 실험 4의 모든 조건 중 가장 높은 평균 정확도를 보였다.
5. C 의존 노드별 정규화는 전역 정규화보다 개선됐지만, 학습 C 자체의 추가 이득으로 이어지지는 않았다.

이 결론은 **사용자가 제공한 서버 출력과 저장소의 결과 기록**에 근거한다.
서버 원본 checkpoint·전체 CSV·NPZ를 이 PC로 가져와 다시 평가한 결과는 아니다.
`completed=true`, hash·coverage·replay·원본 보존은 해당 출력의 보고이고, 로컬 구현 검사와 구분한다.

## 2. 실제 순서와 학습 예산

| 실제 순서 | 질문 | 처리 범위 | 새 학습 | 서버 증거 |
| --- | --- | --- | --- | --- |
| 0 | A/B 연산과 대수식이 맞는가? | 7종 대수 검사, 출력·미분 참조 | 없음 | 고정 실험 완료 로그에 검사 통과 |
| 1 | 고정 Q가 L·L²와 어떻게 다른가? | 198 graph·3,168 scalar 입력 | 없음 | FULL 완료 로그·결과 발췌 |
| 2 | 공유 C가 합성 메시지 규칙을 배우는가? | 531 graph·3 target·5 조건 | gate 30개 seed 모델×500 epoch = 15,000 독립 update | FULL 평가 75줄·완료 로그 |
| 3 | 고정 모델이 새 특징·배율에도 통하는가? | 같은 531 topology·fresh 특징·5배율 | 없음 | FULL 완료·요약 |
| 3.1 | C 입력 RMS가 배율 문제를 해결하는가? | 같은 데이터·normalized gate, raw 고정 | gate 30개 seed 모델×500 epoch = 15,000 독립 update | FULL 요약 |
| 4 | Teacher 없이 CE로 학습한 C가 분류에 유익한가? | 3 citation·8조건·336 run | 168,000 update | completion·분류 요약 |
| 4.1 | C 배치 효과와 분기 크기 효과가 다른가? | 기존 final 모델 120개 재현·learned 30개 개입 | 없음 | completion·frozen 개입 요약 |
| 4.1 CSV 분석 | 작은 분기를 어떤 인자가 만드는가? | 기존 scalar CSV 분석 | 없음, 모델 forward도 없음 | completion·인자 분석 |
| 4.2 | C 의존 노드별 정규화로 재학습하면 개선되는가? | 3 citation·5조건·210 run | 105,000 update | completion·분류/개입 요약 |
| 추가 CSV 확인 | 둘째 층 C 배치가 같은 크기의 C=1보다 유익한가? | 4.2의 저장된 개입 12행 | 없음 | 사용자 제공 평균·95% 구간 |

Gate 30개는 3 target×learned/random-pair×5 seed다. 독립 seed를 함께 계산한 packed optimizer 호출과
독립 모델 update 수를 구분한다. 합성 first/polynomial/fixed의 9개 train-only scalar fit은 최소제곱이며 optimizer 학습이 아니다.

**초기 계획은 실제 실행과 다르다.** 기존 698-run 계획, hidden 256·10 seed·AdamW 등은 계승하지 않았다.
고정 연산부터 검증하는 새 계약으로 전환했고, 이후 각 패키지의 실제 `config_full.json`을 따랐다.
최초 softplus/sigmoid teacher·β=1 제안도 실제 실험 2의 exp(tanh)/graph mean·자유 β 규칙으로 갱신됐다.
과거 계획과 실제 결과를 합쳐 가상의 실험을 만들지 않는다.
변경 이력: [고정 연산부터 시작한 계획](../../research/wedge_propagation/EXPERIMENT_PLAN_FIXED_FIRST.md),
[초기 계획 보관본](../../research/wedge_propagation/archive/EXPERIMENT_PLAN_20261002_before_fixed_first.md).

## 3. 공통 수식: 무엇을 실제로 계산했는가?

단순 무방향 그래프의 물리 엣지를 방향을 정해 한 번 저장하고 B의 행으로 만든다.
중심 j의 서로 다른 이웃 {i,k}를 unordered pair로 한 번씩 사용한다. 삼각형 내부 wedge도 포함한다.

\[
L=B^\top B,\quad A_{ijk,:}=e_i^\top-2e_j^\top+e_k^\top,\quad Q=A^\top A,
\]
\[
g_1=x_j-x_i,\quad g_2=x_k-x_j,\quad (AX)_{ijk}=g_2-g_1.
\]

A는 두 엣지 차분의 차이를 계산하고 Aᵀ는 (1,−2,1)로 노드에 다시 집계한다. 대응하는 엣지 결합 R을 쓰면 A=RB다.
양의 대각 C를 **현재 입력에서 고정하면**

\[
T_C=A^\top CA,\qquad E_C(X)=\operatorname{tr}(X^\top T_CX)=\sum_p c_p\|(AX)_p\|^2,
\]
\[
c_p\|g_2-g_1\|^2=c_p\|g_1\|^2+c_p\|g_2\|^2-2c_p g_1^\top g_2.
\]

즉 두 차분의 쌍선형 교차항이 A 안에 들어 있다. 임의의 집합 간 직사각 K를 학습하는 별도 모듈은 없다.
C가 고정이면 에너지 gradient는 2T_CX다. 실제 adaptive 모델의 메시지는 T_C(X)X이며,
C(X)까지 미분한 전체 에너지 gradient와 같지는 않다. 실제 loss backward에서는 C 의존성을 미분한다.
에너지의 존재, 에너지를 학습 loss로 사용함, 사이클 소거 정보 복원은 각각 별도 주장이다. 이 트랙은 사이클 복원을 입증하지 않았다.

정확한 고정 연산식은

\[
\boxed{Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B}.
\]

보정 계수는 음수일 수 있으며 자르지 않는다. Q 전체는 AᵀA이므로 PSD다.
엣지 degree sum이 일정한 s이면 Q=L²+(s−4)L이고, cycle은 Q=L²다.
고정 QX는 이 식으로 O(EF)에 계산하며, 경로별 C가 다른 learned 연산은 모든 wedge를 계산한다.
상세: [모델 수식](../../research/wedge_propagation/gpt_handoff/02_MODEL_AND_MATH.md),
[operators.py](../../research/wedge_propagation/operators.py).

## 4. 실험 0/1: 고정 연산자

**질문:** Q가 L·L²와 다른 작용을 하는가? 학습/분류 없이 차이를 측정했다.

- FULL float64, N=20/30/40/60/80/100.
- cycle/star/grid는 크기당 하나, ER/tree/tree+chord는 크기당 독립 draw 10개. 총 198 graph.
- graph당 독립 Gaussian scalar 특징 16개, 총 3,168 입력. 독립 16실현을 vector-valued 채널로 해석하지 않는다.
- 10,890 node·15,325 edge·53,039 wedge. 학습 파라미터·optimizer·classifier 없음.
- A=RB, 방향/재번호 불변성, PSD·row sum·energy·출력/미분 참조 및 7종 대수 검사를 수행했다.

\[
r_G=\min_{a,b}\|Q-aL-bL^2\|_F/\|Q\|_F.
\]

| Family | graph 수 | r_G 중앙값 | spectral norm을 맞춘 Q/L² 출력 상대차 중앙값 |
| --- | ---: | ---: | ---: |
| cycle | 6 | 1.84459e−14 | 0 |
| star | 6 | 1.65675e−13 | 0.123054 |
| grid | 6 | 0.0950461 | 0.130971 |
| ER | 60 | 0.100550 | 0.119721 |
| tree | 60 | 0.142190 | 0.198462 |
| tree+chord | 60 | 0.129991 | 0.149200 |

서버 전체 graph에서 위 항등식 absolute error 0이 보고됐다. Cycle/star는 두 basis의 조합으로 환원된다.
Star의 Q/L² 작용 차이는 L 항으로 설명되므로 r_G≈0과 모순되지 않는다. 나머지 네 family에는 최적 적합 후 잔차가 남았다.
Graph마다 전체 행렬을 보고 맞춘 r_G는 학습된 생성기의 일반화 지표가 아니다.
정보 손실률·분류 정확도·C 학습 성과를 측정한 실험도 아니다.

자원 기록: A6000 1개, CPU worker 후보 실측 후 2개·graph batch 33개 선택. 마지막 연산 진행 elapsed 29.4초이며 그림 작성 시간은 이후다.
원문/기록: `e5e7dcf0-1e82-45c8-b7f9-efb9b59e52c5`,
[SERVER_FIXED_RESULTS.md](../../research/wedge_propagation/SERVER_FIXED_RESULTS.md).

## 5. 실험 2: 합성 teacher 메시지 학습

**질문:** 서로 다른 graph에서도 재사용하는 C 생성 규칙을 출력 메시지만으로 학습하는가?

전체 531 graph·8,496 scalar 실현이다. Split은 train/validation/ID/size OOD/family OOD/family+size OOD
=240/60/120/90/12/9 graph. Train·validation·ID 크기는 20/30/40/50, size OOD는 60/80/100.
Train family는 ER/tree/tree+chord이고 새 family는 cycle/star/grid다.

Teacher는

\[
r_p^*=\frac{g_1g_2}{|g_1||g_2|+10^{-8}}+\frac{|g_2-g_1|}{|g_1|+|g_2|+10^{-8}},\quad
c_p^*=\frac{e^{\tanh r_p^*}}{\operatorname{mean}_{q\in G}e^{\tanh r_q^*}}.
\]

Student는 [|g₁|+|g₂|, g₁g₂, |g₂−g₁|, (|g₁|−|g₂|)²]를 공유 4→64→1 MLP에 넣어
동일한 exp(tanh)/graph mean 규칙으로 양의 C를 만들고 βAᵀCθ(X)AX를 출력한다.
Gate와 자유 β는 seed당 총 386개 파라미터다. Teacher C label/loss는 주지 않고
graph·실현별 target-normalized 메시지 MSE를 최소화했다.

3 target은 LX/L²X/path teacher다. 조건은 first aLX, polynomial aLX+bL²X,
fixed βQX, learned, random-pair다. 앞의 3조건은 train-only scalar fit,
뒤의 2조건은 각 5 seed×500 epoch, Adam lr=.003·weight decay 없음으로 새로 학습했다.
Graph 실현 평균→graph 동일 비중 평균→seed 평균±표본 std 순서로 아래 상대오차를 집계했다.

| Path target split | Polynomial | Fixed Q | Learned | Random-pair |
| --- | ---: | ---: | ---: | ---: |
| validation | .247994 | .168908 | .046531±.013085 | .549527±.000347 |
| ID | .250175 | .171868 | .047140±.013491 | .555487±.000533 |
| size OOD | .246178 | .166230 | .046147±.012647 | .554482±.000441 |
| family OOD | .317892 | .193912 | .049000±.013141 | .616995±.005943 |
| family+size OOD | .288272 | .173095 | .042535±.011215 | .582529±.003682 |

LX target의 first/polynomial, L²X target의 polynomial은 모든 split에서 오차 0이었다.
Learned가 모든 target에서 우세하지 않았다. LX ID learned 오차는 .932349이고 family+size OOD는 10.928360이다.
L²X ID learned .211503, fixed .208187로 learned 우위도 없다.
Path teacher에서 얻은 메시지 회수를 실제 분류나 범용 연산 표현 우위로 확대하지 않는다.
Random-pair는 gate 용량은 같지만 support·hop·엣지 사용 빈도가 달라 연속성 하나의 인과 대조가 아니다.

서버 physical graph batch 240, accumulation 1, 16실현과 5개 독립 seed 병렬 처리.
한 모델의 batch를 seed 수로 곱하지 않는다. A6000, 학생 float32/teacher float64,
CPU 후보 측정 후 worker 1개(531 graph 준비 1.936초), calibration peak 약2.05GB였다.
원문: `0412f129-7f5b-4c0f-9212-29dfb745cfe4`.
[전체 target 결과](../../research/wedge_propagation/SERVER_LEARNED_RESULTS.md),
[실제 full 설정](../../research/wedge_propagation/learned/config_full.json).

## 6. 실험 3: 새 특징과 입력 배율, 학습 0

**질문:** 실험 2의 고정된 규칙이 새 입력에도 통하고, 배율에 안정적인가?
같은 저장된 531 topology에서 fresh Gaussian 특징을 만들고 .25/.5/1/2/4배를 적용했다.
모든 배율은 같은 fresh 특징을 공유한다. Teacher C/target은 매 배율 다시 계산했다.
6 gate checkpoint의 각 5 seed와 9 scalar fit을 고정했으며 새 graph·학습·checkpoint 선택은 없다.

| Fresh path target, 배율 1 | Learned 오차 | Fixed Q 오차 |
| --- | ---: | ---: |
| ID | .0475536 | .174729 |
| size OOD | .0465101 | .166116 |
| family OOD | .0512217 | .203409 |
| family+size OOD | .0473821 | .179715 |

Original ID teacher C 상대오차 .123585±.0286, 상관 .947717±.0231.
같은 learned 모델의 메시지 오차는 원래 .0471402, C=1 .383920,
shuffle .439846, 다른 graph 패턴 .449091, 연결 대응 무작위화 .708137이었다.
개입은 학습 β를 고정하므로 별도로 β를 적합한 fixed Q와 다르다. Cmean=1과 높은 상관은 유일한 C 회수를 보장하지 않는다.

Fresh ID를 4배로 키우면 student C 변화 .369839, 메시지 비례성 오차 .306899,
예측 상대오차 .302559로 커졌다. Teacher의 해당 변화는 6.82005e−7/1.20792e−7이었다.
새 특징의 기본 배율에서는 회수가 유지됐으나 raw gate는 크기에 민감했다.
메시지/scale 124,254/103,545행, update 0, original replay·보존 통과가 보고됐다.
이 행 수는 독립 graph 수가 아니다.
원문: `43094fe2-6bfc-4a48-9974-f30776c53783`, `e7640614-a320-4e01-88b7-8e88a447d102`(같은 run).
[서버 기록](../../research/wedge_propagation/SERVER_GENERALIZATION_RESULTS.md).

## 7. 실험 3.1: C 입력 RMS 정규화, normalized만 새 학습

C 생성 입력에서만 g₁·g₂를 physical-edge 차분 RMS로 나눴다. 실제 AX 메시지는 원래 차분을 쓴다.
\(\sigma_{G,r}=\sqrt{\|BX_r\|^2/|E_G|}\). 비영 σ에 epsilon을 더하지 않고 zero/no-edge일 때 1을 쓴다.
같은 데이터·hidden dimension 64·seed 5개·500 epoch·graph batch 240으로 normalized learned/random-pair를 새로 학습했다.
Raw checkpoint·scalar fit은 고정하고 실험 3의 특징/배율을 재사용했다.

| Fresh path target 배율 1 | Raw | Normalized | Fixed Q |
| --- | ---: | ---: | ---: |
| ID | .0475536±.0139 | .0515799±.00357 | .174729 |
| size OOD | .0465101±.0128 | .0525378±.00365 | .166116 |
| family OOD | .0512217±.0143 | .0532518±.00297 | .203409 |
| family+size OOD | .0473821±.0130 | .0508475±.00274 | .179715 |

Fresh ID의 graph/seed별 **다섯 배율 중 최대 변화**를 집계하면 raw/normalized C 변화는
.371238±.135 / 1.95587e−7±1.58e−8, 메시지 비례성 오차는
.306902±.0969 / 2.06874e−7±1.06e−8이었다.
배율 안정성과 seed 변동은 개선됐지만 모든 표의 평균 회수 오차는 raw보다 높다.

Normalized original ID C 상대오차 .130976±.00870, 상관 .944367±.00686.
Normalized original ID 개입의 오차 증가는 C=1 .348761, shuffle .402727,
다른 graph 패턴 .412882, 대응 무작위화 .665389였다. Identity와 mean은 Cmean=1 제약에서 같은 개입이다.
Original replay 11,349행 verified가 보고됐지만 최대 metric 차이 .002053013486,
target-normalized RMSE 차이 8.21676e−5이므로 완전한 수치 동일성은 아니다.
메시지/scale/개입 248,508/207,090/159,300행, raw/test update 0.
첨부에는 정확한 output 폴더명·GPU/VRAM/batch 실측·전체 소요시간이 없어 추정하지 않는다.
원문: `da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63`.
[서버 기록](../../research/wedge_propagation/SERVER_SCALE_NORMALIZATION_RESULTS.md).

## 8. 실험 4: 실제 노드 분류

**질문:** 합성 teacher 없이 분류 CE로 학습한 C가 도움이 되는가?
전체 public citation graph를 사용하며 synthetic checkpoint를 가져오지 않았다.

| 데이터 | N×F, K | 물리 E | wedge P | train/validation/test |
| --- | --- | ---: | ---: | --- |
| Cora | 2708×1433, 7 | 5,278 | 52,301 | 140/500/1000 |
| CiteSeer | 3327×3703, 6 | 4,552 | 26,918 | 120/500/1000 |
| PubMed | 19717×500, 3 | 44,324 | 699,342 | 60/500/1000 |

2층 입력→64→K, bias 없는 Xavier projection, 각 projection 전 dropout .5,
첫 층 뒤 ReLU, 둘째 층 logits. C gate는 현재 투영 Z의 4F→64→1이며 path C 하나를 모든 channel에 공유한다.
Raw/RMS를 각각 처음부터 학습한다. 여기의 RMS는 graph·seed·layer별 물리 엣지와 전체 channel을 묶은 RMS다.
합성 실험처럼 scalar 실현마다 따로 정규화하는 것과 구분한다.
Projection·gate·α·β는 train-node CE로 학습하며 teacher/에너지/복원 보조 loss는 없다.

\[
\bar L=\tfrac12S_dLS_d,\quad S_Q=\operatorname{diag}(Q)^{-1/2},\quad
\kappa(C)=\max_{q_v>0}\frac{\operatorname{diag}(A^\top CA)_v}{q_v},
\]
\[
M_C=\frac{S_QA^\top CA S_QZ}{3\kappa(C)},\quad U=Z-\alpha\bar LZ-\beta M_C.
\]

α=t(1−r), β=tr, t/r는 sigmoid이며 초기 α=.5, β=.25다. Polynomial의 둘째 분기는 L̄²Z,
fixed는 C=1이다. Fixed+nodeMLP는 실제 추가 MLP로 gate와 파라미터 수를 맞췄다.
GCN 대조는 자기 연결을 포함한 대칭 정규화 인접행렬 집계이며 BᵀB를 그대로 곱하는 모델은 아니다.
대각 정규화의 inverse sqrt는 양의 대각에서만 계산하고 zero 대각은0이다. 표의 accuracy는 %, 차이는 pp,
CE는 L2 항을 제외한 평가 분류 손실이다.

각 조건 LR .001/.003/.01×tuning seed101/202/303, final seed11/23/37/53/71,
각 500 epoch다. Adam·matrix weight decay 5e−4·float32·TF32 off를 사용하며 early stopping/scheduler/clipping은 없다.
Validation CE로 checkpoint/LR를 선택하고 모든 final 선택을 고정한 뒤 test 평가했다.
Tuning 216회 + final 120회 = 총 336회 학습·168,000회 갱신이다. 모델당 전체 그래프 batch는 1이며 독립 seed 모델은 측정 후 병렬 pack한다.

| Test accuracy 평균 % | Cora | CiteSeer | PubMed |
| --- | ---: | ---: | ---: |
| MLP | 57.14 | 56.22 | 72.28 |
| First order | 70.78 | 63.78 | 75.10 |
| Polynomial 2 | 72.54 | 66.02 | 75.72 |
| Fixed wedge | 73.56 | 64.94 | 75.46 |
| Learned raw | 70.74 | 63.74 | 75.06 |
| Learned RMS | 70.52 | 63.86 | 74.68 |
| Fixed wedge+nodeMLP | 63.70 | 55.92 | 71.18 |
| Standard GCN | 81.88 | 71.12 | 79.06 |

Learned raw/RMS의 평균 정확도는 fixed/polynomial보다 낮고 표준 GCN이 가장 높다.
C std는 raw 약 .058–.413, RMS 약 .319–.688이므로 전부 C=1인 것은 아니다.
True κ 유지 C=1/shuffle 개입은 정확도 변화가 작고, C=1에서 κ를 재계산하면 +.40~1.32pp였다.
κ 유지와 실제 메시지 norm 유지는 달라 다음 frozen 진단으로 분리했다.
Original/개입/scale/층 진단 360/2,970/2,100/2,220행, completed/actual_data/보존 보고,
전체 3,970.394초. 원문: `a0770e1e-506a-43e1-b656-bbd3b045b108`.
[서버 기록](../../research/wedge_propagation/SERVER_CLASSIFICATION_RESULTS.md),
[실제 설계](../../research/wedge_propagation/classification/EXPERIMENT_DESIGN.md).

## 9. 실험 4.1: C 배치와 분기 크기, frozen 평가

기존 120 final 모델의 원래 metric을 재현하고 learned raw/RMS 30개에 개입했다.
C를 1/shuffle로 바꿀 때 실제 메시지 norm을 맞추는 처리와,
true C를 유지하면서 κ 분모를 1로 바꾸는 처리를 layer_0/layer_1/both에서 비교했다.
Fixed-Z 별도 진단은 원래 깨끗한 Z를 고정했다. 모델·학습·checkpoint 선택은 고정, update 0이다.

| Both 개입 | 평균 test accuracy 차이 범위 | CE/구간 |
| --- | ---: | --- |
| C=1, 실제 메시지 norm 일치 | −.020~+.120pp | 6 paired accuracy 구간 모두 0 포함 |
| C shuffle, 실제 메시지 norm 일치 | −.034~+.090pp | 6 paired accuracy 구간 모두 0 포함 |
| True C, κ 분모 1 | +.320~+1.260pp | 6조건 CE +.00493~+.01537 악화 |
| C=1, 분모 1 | +.400~+1.320pp | 6조건 평균 CE 악화 |

Cora/RMS fixed-Z C=1 cosine은 첫/둘째 층 .848/.889, shuffle은 .831/.861이었다.
C는 방향을 바꾸지만 같은 norm에서 배치의 분류 이득은 뚜렷하지 않았다. 0 포함 구간은 동등성 증명이 아니다.
이차 분기 제거는 Cora raw/RMS accuracy −.58/−.30pp였다.
Accuracy 상승과 CE 악화가 함께 있어 κ 제거를 성능 개선/실패 원인으로 확정하지 않는다.
Baseline ||βM||/||Z||는 약1.45–3.24%이며 노드별 영향이나 gradient 기여율이 아니다.

Original/개입/층/fixed-Z 360/6,750/4,740/1,560행, 2,370 seed-forward 경우,
102.1296초였고 보고된 최대 allocated peak는 약 8.89 GiB다. PubMed raw는 seed 4+1개, RMS는 seed 5개를 pack했다.
이는 진단을 포함한 비용이며 순수 표준 모델 forward 비용과 비교하지 않는다.
원문: `975a1691-1a87-4c34-a672-f9aa8cf682ec`.
[서버 기록](../../research/wedge_propagation/SERVER_BRANCH_STRENGTH_RESULTS.md).

## 10. 실험 4.1 CSV 분석: 작은 분기 인자의 분해

모델을 다시 돌리지 않고 저장된 CSV를 분석했다.

\[
R=\tfrac13S_QA^\top CA S_QZ,\qquad
\|\beta M\|/\|Z\|=(\beta/\kappa)(\|R\|/\|Z\|).
\]

Seed별 인자를 계산한 뒤 집계했다. 평균 인자의 곱은 실제 평균 분기 비율과 같지 않을 수 있다.
β 평균 .270–.304, κ 약4–7. 노드 대각 비율 평균은 약1인데 드문 최대값이 전체 메시지의 분모가 됐다.
이 관측이 C 의존 노드별 정규화 재학습의 근거다. κ 하나가 실패 원인이라는 증명은 아니다.
인자/강도/paired/fixed-Z 60/204/756/384행, forward0·update0·원본보존·분석.904초.
실행 commit `3f8b24f`, 원문 `76feaf64-c752-4d8b-bb09-8aaf929648b2`.
[서버 기록](../../research/wedge_propagation/SERVER_BRANCH_ANALYSIS_RESULTS.md).

## 11. 실험 4.2: C 의존 노드별 정규화 재학습

Fixed C=1/global raw·RMS/node raw·RMS의 5조건을 모두 처음부터 학습했다.
같은 데이터·2층·hidden dimension 64·LR·seed·500 epoch를 유지했다. Tuning 135회 + final 75회 = 총 210회 학습·105,000회 갱신이다.
Global도 이번에 재학습했으므로 실험4의 과거 global 수치와 섞지 않는다.

\[
D_C=\operatorname{diag}(A^\top CA),\quad S_C=D_C^{-1/2},\quad
M_{node}=\tfrac13S_CA^\top CA S_CZ.
\]

C·D_C·양쪽 S_C 미분을 유지했다. C=1에서는 global/node fixed가 같다.
현재 C 고정 native 연산자는 PSD·norm≤1이지만 adaptive 전체 Jacobian/학습 성공 보장은 아니다.

| Test accuracy 평균 % | Fixed C=1 | Global/raw | Global/RMS | Node/raw | Node/RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 73.56 | 70.76 | 70.52 | 72.72 | 72.08 |
| CiteSeer | 64.94 | 63.74 | 63.84 | 64.92 | 64.90 |
| PubMed | 75.46 | 75.06 | 74.74 | 75.56 | 75.72 |

모든 조건 train accuracy 평균100%. Node−global accuracy는 raw/RMS 순서로
Cora +1.96/+1.56pp, CiteSeer +1.18/+1.06pp, PubMed +.50/+.98pp다.
Cora 두 accuracy 구간만 0을 제외했고, CE는 PubMed/RMS 외 5비교가 음의 구간이었다.
Node 정규화는 전역 정규화보다 개선됐다. 정규화·좌표·메시지 방향·전체 파라미터의 재학습을 포함하므로 κ만의 인과 효과는 아니다.
Cora/raw 첫 층 β는 .2697→.04251, βM/Z는 .02395→.01387로 낮아져 모든 분기의 증폭도 아니다.

Node−fixed accuracy는 Cora −.84/−1.48pp, CiteSeer −.02/−.04pp, PubMed +.10/+.26pp다.
PubMed 양의 구간은 0을 포함한다. Node/RMS CE는 세 데이터 모두 fixed보다 높고 구간도 양수다.
학습 C가 fixed보다 유익하다는 근거를 얻지 못했다.

Node/raw 첫 층 C std는 Cora7.19e−7/CiteSeer6.926e−7/PubMed1.055e−5로 거의 균일하나,
둘째 층은 .6141/.4788/.6555로 비균일하다. RMS는 두 층 모두 비균일하다.
현재 checkpoint의 첫 층 균일성을 학습 전체·모든 C의 미학습으로 확대하지 않는다.
Both norm-matched C=1/shuffle accuracy 차이는 −.042~+.18pp, C=1의 평균 CE는 6조건 모두 감소했다.
이차 분기 제거는 CiteSeer/node raw·RMS accuracy를1.42/1.16pp 낮췄으나 CE는 감소했다.
분기 자체의 기여와 학습 C 배치의 기여를 구분해야 한다.

Primary/frozen/층 225/12,420/8,430행, actual_data·전체coverage·보존 보고,
6,946.80초(약1시간55분47초), 구현 기준 `2422568`.
원문: `5bdcdc63-17da-4c09-90bb-30b3139d7704`.
[서버 기록](../../research/wedge_propagation/SERVER_NODE_NORMALIZATION_RESULTS.md).

### 추가 확인: 둘째 층만 같은 norm의 C=1로 교체

사용자가 저장된 `layer_1/c_identity_norm_matched` CSV에서 직접 제공한 12행이다.
추가 학습은 없으며 아래 차이는 **교체 후−원래 모델**이다.

| 데이터 | Gate | Δaccuracy pp [95% 구간] | ΔCE [95% 구간] |
| --- | --- | ---: | ---: |
| Cora | raw | 0 [−.087798,+.087798] | −.00015439 [−.00041737,+.00010859] |
| Cora | RMS | +.14 [+.071995,+.20801] | −.0005968 [−.00085903,−.00033457] |
| CiteSeer | raw | −.02 [−.075528,+.035528] | −.00011344 [−.00041775,+.00019087] |
| CiteSeer | RMS | +.02 [−.21884,+.25884] | −.00076339 [−.0012577,−.00026909] |
| PubMed | raw | +.18 [+.018106,+.34189] | −.0012165 [−.0014868,−.00094613] |
| PubMed | RMS | +.079999 [−.26455,+.42455] | −.0013966 [−.001688,−.0011052] |

4 CE 구간은 전부 음수, 2 accuracy 구간은 전부 양수다. 둘째 층 배치의 이득은 확인되지 않았고 일부 교체는 개선됐다.
원래 C의 메시지 norm이 gain에 남고 첫 층 C·projection·α·β도 유지되므로 C 전체 제거/재학습 동등성은 아니다.
원시 seed 값은 이 발췌에 없다.

## 12. 로컬 검증과 서버 FULL의 경계

각 단계의 [VERIFICATION.md](../../research/wedge_propagation/VERIFICATION.md),
[learned](../../research/wedge_propagation/learned/VERIFICATION.md),
[generalization](../../research/wedge_propagation/generalization/VERIFICATION.md),
[scale](../../research/wedge_propagation/scale_normalization/VERIFICATION.md),
[classification](../../research/wedge_propagation/classification/VERIFICATION.md),
[branch strength](../../research/wedge_propagation/branch_strength/VERIFICATION.md),
[CSV 분석](../../research/wedge_propagation/branch_analysis/VERIFICATION.md),
[node normalization](../../research/wedge_propagation/node_normalization/VERIFICATION.md)에 당시 구현 검사가 있다.
이번 전달 작업에서 그 단위 검사/학습을 다시 실행한 것은 아니다.

로컬 DEBUG는 별도 작은 fixture의 연산·pipeline 검사다. 예컨대4.2는206개 새 테스트와
CUDA DEBUG는 90회 fixture 학습·270회 갱신으로, N=24/30/36·F=12·K=3·hidden 8·3 epoch를 사용했다.
실제 citation 전체·2층·hidden dimension 64·seed 5개·모든 wedge의 별도 무갱신 forward/backward에서는
optimizer update0, aggregate gradient 유한, local peak 약1.63GiB였다.
이 값을 서버 Adam 장기 학습의 peak나 A6000 처리량으로 대체하지 않는다.
Server FULL 결과는 위 사용자 제공 완료/요약의 별도 증거다.

## 13. 원문과 서버 폴더

서버 상대경로의 기준은 `/home/aicompetition07/new-gat`다. 확인되지 않은 timestamp/commit은 추정하지 않았다.

| 단계 | 원문 첨부 ID | 확인된 서버 결과 폴더 |
| --- | --- | --- |
| 0/1 | e5e7dcf0-1e82-45c8-b7f9-efb9b59e52c5 및 대화 완료 로그 | results/wedge-fixed-20261002-144119 |
| 2 | 0412f129-7f5b-4c0f-9212-29dfb745cfe4 | results/wedge-learned-20261002-172132 |
| 3 | 43094fe2-6bfc-4a48-9974-f30776c53783, e7640614-a320-4e01-88b7-8e88a447d102 | results/wedge-feature-20261003-042123 |
| 3.1 | da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63 | 요약에서 정확한 이름 미확인 |
| 4 | a0770e1e-506a-43e1-b656-bbd3b045b108 | 완료 발췌에서 원본 이름 미기록 |
| 4.1 | 975a1691-1a87-4c34-a672-f9aa8cf682ec | results/wedge-branch-strength-20261003-104349 |
| 4.1 CSV | 76feaf64-c752-4d8b-bb09-8aaf929648b2 | results/wedge-branch-analysis-20261003-133432 |
| 4.2 | 5bdcdc63-17da-4c09-90bb-30b3139d7704 및 대화 CSV12행 | ece-a6gpu6, 정확한 폴더명 미확인 |

첨부의 원래 로컬 경로는 `C:/Users/user/.codex/attachments/<ID>/붙여넣은 텍스트.txt`다.
전달 ZIP의 evidence 원문을 수령한 bytes와 함께 읽고, 요약 문서만으로 실행 증거를 대체하지 않는다.

## 14. GPT가 검토할 때 지켜야 할 해석 범위

- Synthetic 메시지 상대오차 .05는 분류 정확도95%가 아니다. 합성 teacher 회수와 실제 task를 구분한다.
- 고정 Q의 잔차는 임의 spectral 모델의 표현 한계 증명이 아니다. 그래프에 따라 환원되기도 한다.
- RMS 안정성은 설계가 작동했다는 증거이고, task 성능/정보 복원 성공은 별도 문제다.
- C의 비균일성, 분기가 사용됨, C 배치가 유익함은 같은 주장이 아니다.
- Frozen 교체와 해당 구조를 처음부터 재학습한 비교는 다르다. Norm match는 원래 C의 크기 효과를 남긴다.
- Citation paired95% t 구간은 같은 public split의5초기화 seed 변동이다. 독립 split/graph·다중 비교 불확실성을 포함하지 않는다.
- 이미 test를 본 뒤의 후속 실험을 완전히 보지 않은 test에 대한 독립 검증이라고 주장하지 않는다.
- 정확한 통합 에너지 전파, 사이클 복원, 학습 C의 속도 이득이나 선행연구 대비 신규성은 이 결과로 입증하지 않았다.

전체 수치 재집계에는 서버 completion/contract, source manifest, per-seed metric·개입 CSV,
checkpoint와 graph/data manifest가 필요하다. 현재 묶음은 수령한 결과와 실제 코드의 대응을 검토할 수 있게 정리한 자료다.

## 15. 구현·수식·검증·결과 대응

| 단계 | 실제 코드 | 수식·설계와 검사 | 서버 결과 원문을 정리한 문서 |
| --- | --- | --- | --- |
| 0/1 고정 Q | [algebra](../../research/wedge_propagation/algebra.py), [operators](../../research/wedge_propagation/operators.py), [study](../../research/wedge_propagation/study.py) | [수식](../../research/wedge_propagation/MODEL_MATH.md), [검증](../../research/wedge_propagation/VERIFICATION.md) | [Fixed](../../research/wedge_propagation/SERVER_FIXED_RESULTS.md) |
| 2 learned path C | [model](../../research/wedge_propagation/learned/model.py), [train](../../research/wedge_propagation/learned/train.py) | [수식](../../research/wedge_propagation/learned/MODEL_MATH.md), [FULL 설정](../../research/wedge_propagation/learned/config_full.json) | [Learned](../../research/wedge_propagation/SERVER_LEARNED_RESULTS.md) |
| 3 fresh 특징·배율 | [frozen](../../research/wedge_propagation/generalization/frozen.py), [study](../../research/wedge_propagation/generalization/study.py) | [수식](../../research/wedge_propagation/generalization/MODEL_MATH.md), [FULL 설정](../../research/wedge_propagation/generalization/config_full.json) | [Generalization](../../research/wedge_propagation/SERVER_GENERALIZATION_RESULTS.md) |
| 3.1 RMS gate | [model](../../research/wedge_propagation/scale_normalization/model.py), [study](../../research/wedge_propagation/scale_normalization/study.py) | [수식](../../research/wedge_propagation/scale_normalization/MODEL_MATH.md), [FULL 설정](../../research/wedge_propagation/scale_normalization/config_full.json) | [Scale normalization](../../research/wedge_propagation/SERVER_SCALE_NORMALIZATION_RESULTS.md) |
| 4 citation 분류 | [model](../../research/wedge_propagation/classification/model.py), [training](../../research/wedge_propagation/classification/training.py) | [실험 계약](../../research/wedge_propagation/classification/EXPERIMENT_DESIGN.md), [FULL 설정](../../research/wedge_propagation/classification/config_full.json) | [Classification](../../research/wedge_propagation/SERVER_CLASSIFICATION_RESULTS.md) |
| 4.1 frozen 강도 | [core](../../research/wedge_propagation/branch_strength/core.py), [study](../../research/wedge_propagation/branch_strength/study.py) | [수식](../../research/wedge_propagation/branch_strength/MODEL_MATH.md), [FULL 설정](../../research/wedge_propagation/branch_strength/config_full.json) | [Branch strength](../../research/wedge_propagation/SERVER_BRANCH_STRENGTH_RESULTS.md) |
| CSV 인자 분석 | [core](../../research/wedge_propagation/branch_analysis/core.py), [study](../../research/wedge_propagation/branch_analysis/study.py) | [분석 계약](../../research/wedge_propagation/branch_analysis/README.md), [검증](../../research/wedge_propagation/branch_analysis/VERIFICATION.md) | [Branch analysis](../../research/wedge_propagation/SERVER_BRANCH_ANALYSIS_RESULTS.md) |
| 4.2 노드 정규화 | [model](../../research/wedge_propagation/node_normalization/model.py), [training](../../research/wedge_propagation/node_normalization/training.py) | [수식](../../research/wedge_propagation/node_normalization/MODEL_MATH.md), [FULL 설정](../../research/wedge_propagation/node_normalization/config_full.json) | [Node normalization 및 layer_1 CSV](../../research/wedge_propagation/SERVER_NODE_NORMALIZATION_RESULTS.md) |

이 트랙의 새 gradient 학습 예산은 실험 2의 15,000 + 3.1의 15,000 + 4의 168,000 + 4.2의 105,000
= **303,000회 독립 모델 갱신**이다. 여러 seed를 pack한 optimizer 호출 횟수와 다르다.
고정 실험·scalar 최소제곱 fit·frozen forward·CSV 분석·disposable calibration은 이 수에 합치지 않는다.
같은 topology·split을 재사용한 단계를 새 독립 그래프 데이터로 세지 않는다.

## 16. 현재 연구와의 연결

이 트랙은 길이 2 경로의 AᵀCA다. 다음 [로컬 그래프 연구](../../research/local_energy_relations/README.md)는
각 ego의 내부 이차 에너지와 로컬 그래프 사이 관계를 명시적으로 기록한다.
그 후 [copy 결합](../../research/local_context_coupling/MODEL_MATH.md)은
같은 물리 노드의 문맥별 copy를 유지한 채 intra→cross→intra→merge를 실제 전파에 사용한다.

현재 [Normalization](../../research/local_context_coupling/normalization/MODEL_MATH.md)은
기존 copy 모델의 C=unit/local_degree와 연결은 유지하며, graph/local 내부 step 및 graph/edge cross 정규화를 비교한다.
500 epoch·2개 macro 층·hidden 64·20조건·840회/420,000회 갱신은 **새 본학습 계약**이다.
[현재 검증](../../research/local_context_coupling/normalization/VERIFICATION.md)은
222개 단위 테스트, DEBUG 360회 fixture 학습/1,080회 갱신 및 resume 0회 갱신을 기록한다.
20조건·840회 학습·420,000회 갱신의 normalization 서버 FULL 분류 결과를 수령했다.
로컬 본학습은 미실행이며, 새 결과는 [정규화 FULL 결과](08_NORMALIZATION_FULL_RESULTS.md)와
[서버 원문](../evidence/local_context_normalization_server_full_20261005.txt)을 따른다.
Wedge의 4.2 수치나 이전 copy 분류 수치를 그 새 FULL 결과로 바꾸어 쓰지 않는다.
