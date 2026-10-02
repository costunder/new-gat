# Experiment 4 — 실제 노드 분류에서 경로 가중치의 기여

2026년 10월 3일 설계 계약. **서버 실행 코드를 구현했으며 전체 분류 학습은 서버에서 실행한다.**
실행 명령은 [README.md](README.md), 구현 검증과 미실행 범위는 [VERIFICATION.md](VERIFICATION.md)에 있다.
설정은 [design_contract.json](design_contract.json), 선행 결과는
[Experiment 3.1 서버 기록](../SERVER_SCALE_NORMALIZATION_RESULTS.md)에 있다.

## 1. 확인할 질문

**합성 teacher 없이 분류 loss만으로 배운 C가 실제 분류에 도움이 되는가?**

- Fixed Q와 비교해 경로마다 다른 C의 효과를 확인한다.
- 직접 메시지가 최대 두 홉 안에서 전달되는 polynomial과 비교한다.
- 같은 추가 파라미터를 사용하는 노드 MLP로 용량을 대조한다.
- 학습 뒤 C를 1로 바꾸거나 위치를 섞어 실제 사용 여부를 확인한다.
- Raw와 RMS gate의 분류 성능·배율 반응·계산 비용을 비교한다.

3.1 fresh ID 회수 오차는 raw 0.0475536, normalized 0.0515799였다.
RMS는 배율 반응을 수치 오차 수준으로 줄였지만 평균 회수 오차는 증가했다.
따라서 두 조건을 모두 유지한다. Synthetic checkpoint와 teacher C를 분류 학습에 사용하지 않는다.

## 2. 전체 그래프와 공식 split

Cora, CiteSeer, PubMed의 **public fixed split**을 사용한다.
전체 그래프를 forward하고 train mask의 노드에만 CE를 적용한다.
Validation/test의 특징·연결도 사용하는 transductive 실험이다.
데이터셋마다 새로 학습하므로 독립 새 graph의 zero-shot 일반화를 입증하지는 않는다.

| 데이터 | 노드 | 공식 edge_index 수 | 입력 특징 | 클래스 | train / validation / test |
| --- | ---: | ---: | ---: | ---: | --- |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 9,104 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 88,648 | 500 | 3 | 60 / 500 / 1,000 |

규모·split 정의는 [공식 Planetoid 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)를 따른다.
표의 연결 수에는 양방향 표현이 포함된다. 자기 연결·reciprocal duplicate를 제거한 물리 엣지 수 E와
모든 unordered wedge 수 P=Σ_v d_v(d_v−1)/2는 실제 raw를 읽고 측정한다.
고립 노드·삼각형의 wedge·공식 mask를 유지하며 `split="full"`로 바꾸지 않는다.

독립 loader는 raw SHA256, node ID 순서, 특징·label·mask hash를 저장하고 CiteSeer의 빠진 test index도 복원·검증한다.
각 노드 특징을 feature sum으로 나누고 영 행은 유지한다. 기존 Conductance adapter·model·전처리를 import하지 않는다.

## 3. 공통 두 층 모델과 연산 정규화

입력 D → hidden 64 → 클래스 K. 각 층은 dropout 0.5를 적용한 입력을 bias 없이 투영한다.

\[
Z_\ell=\operatorname{Dropout}_{0.5}(H_\ell)W_\ell,\qquad
U_\ell=(I-\alpha_\ell\bar L-\beta_\ell\bar T_\ell)Z_\ell.
\]

H₀=X, H₁=ReLU(U₀), U₁은 logits다. 두 번째 출력에 ReLU를 적용하지 않는다.
추가 output projection·normalization layer·attention head는 사용하지 않는다.
두 연산 조건은 층마다 t=sigmoid(u), r=sigmoid(v), α=t(1−r), β=tr로 학습한다.
초기 α=0.5, β=0.25이므로 u=log(3), v=−log(2)다.
First-order는 α=sigmoid(u), 초기 0.5이며 β parameter를 만들지 않는다.
MLP/GCN에도 사용하지 않는 분기 scalar를 만들지 않는다. Synthetic의 자유 beta와 별도인 분류용 규약이다.

### 고정 topology 좌표와 크기 상한

L=BᵀB, Q=AᵀA, T_C=AᵀCA를 유지하며 다음 분류용 정규화를 새로 적용한다.

\[
\bar L=\tfrac12 S_dLS_d,\quad(S_d)_{vv}=d_v^{-1/2},\qquad
q_v=Q_{vv}=4\binom{d_v}{2}+\sum_{j\sim v}(d_j-1),\quad(S_Q)_{vv}=q_v^{-1/2}.
\]

대각값 0의 역제곱근은 0이다. S_d,S_Q는 topology에서 한 번 계산하고 고정한다.
현재 층의 C로 d_C=diag(T_C)를 계산해 graph·층·seed별 scalar κ를 얻는다.

\[
\kappa(C)=\max_{v:q_v>0}\frac{(d_C)_v}{q_v},\qquad
\bar Q=\tfrac13 S_QQS_Q,\qquad
\bar T_C=\frac{S_QA^\top CA S_Q}{3\kappa(C)}.
\]

Wedge가 없으면 Q,T_C=0, κ=1이다. C=1이면 κ=1이고 T̄_C=Q̄다.
S_Q는 C와 함께 바뀌지 않으므로 learned/fixed가 같은 노드 좌표를 사용한다.
κ는 강도 조절이며 gradient를 유지하고 branch norm과 함께 기록한다.
κ는 전체 path의 d_C를 합산한 뒤 노드 축에서 torch.amax로 계산한다.
동률 최대 노드에 gradient를 균등 분배하며 chunk별 max를 다시 합치는 gradient 규약을 사용하지 않는다.

Incidence 행의 support는 2개, wedge 행은 3개다. Cauchy–Schwarz로
L≼2diag(L), T_C≼3diag(T_C)≼3κdiag(Q)가 성립한다.
따라서 L̄, L̄², Q̄, T̄_C는 PSD이고 최대 고유값이 1 이하다.
현재 C를 고정하면 (1−α−β)I≼P≼I다. 전체 비선형 network의 Jacobian·학습 수렴 보장으로 확대하지 않는다.
이는 norm의 상한을 맞추는 규칙이며 실제 spectral norm을 같게 만들지는 않는다.

실제 learned branch는 S_Q AᵀC A(S_Q Z)/(3κ)다.
3.1의 원래 AZ 메시지와 다른 분류용 좌표를 쓰며 raw/RMS는 이 규약을 공유한다.
정규화된 Q의 nullspace도 좌표가 다르므로 raw Q의 상수 특징 소거를 그대로 적용하지 않는다.
Polynomial은 L̄²를 사용한다. Raw Q=L²+degree correction 항등식은 선행 대수 결과다.
Q̄와 L̄²의 분류 차이를 이 항등식만으로 설명하거나 새 spectral 표현력의 증명으로 주장하지 않는다.

## 4. 다채널 C 생성

현재 Z에서 g₁=Z_j−Z_i, g₂=Z_k−Z_j를 만든다. 첫 층 F=64, 둘째 층 F=K다.
경로마다 scalar c_p 하나를 생성해 메시지의 모든 channel에서 공유한다.
Raw는 g를 그대로 gate에 넣고 RMS는 다음 scalar를 **gate 입력에만** 적용한다.

\[
\sigma(Z)=\sqrt{\frac{\|BZ\|_F^2}{EF}},\qquad
\widetilde g_1=g_1/\sigma,\quad\widetilde g_2=g_2/\sigma.
\]

비영 σ에는 epsilon을 더하지 않는다. E=0 또는 σ=0이면 scale 1을 사용한다.
모든 channel에 같은 σ를 적용해 상대적인 channel 크기를 유지한다.
학습한 Z의 σ는 매 forward에서 미분 가능하게 계산한다.
기존 16개 독립 scalar 실현과 공동 hidden channel gate를 혼동하지 않는다.

\[
\phi_p=\operatorname{concat}[|\widetilde g_1|+|\widetilde g_2|,
\widetilde g_1\odot\widetilde g_2,|\widetilde g_2-\widetilde g_1|,
(|\widetilde g_1|-|\widetilde g_2|)^2],\quad
s_p=\operatorname{MLP}_{4F\to64\to1}(\phi_p),\quad
c_p=\frac{\exp(\tanh(s_p))}{P^{-1}\sum_a\exp(\tanh(s_a))}.
\]

첫 Linear에만 bias, 중간 활성화는 ReLU다. Source scalar gate의 마지막 bias까지 옮긴 구조가 아니다.
경로 뒤집기에 φ가 불변이며 C는 양수·graph mean 1이다.
C·RMS·mean·κ의 gradient를 끊지 않으며 분류 CE로 투영·gate·분기 계수를 공동 학습한다.
직접 메시지 support는 두 층에서 최대 4홉이다. Graph 전체 RMS·κ 때문에 전체 특징 의존은 전역이다.
이 정규화에서는 C의 공통 mean 인자가 T_C/κ(C)에서 상쇄된다.
Mean 1은 C를 비교하는 gauge이고 전역 메시지 의존은 RMS·κ에서 생긴다.
에너지는 C와 좌표를 고정했을 때 이차형식이다. C(Z)를 포함한 전체 함수나 전파를 에너지의 완전한 gradient라고 부르지 않는다.

## 5. 여덟 비교 조건

| 조건 | 전파 | 목적 |
| --- | --- | --- |
| MLP | U=Z | Graph 전파 기준 |
| First-order | U=(I−αL̄)Z | 일차 전파 |
| Polynomial-2 | T̄=L̄² | 최대 두 홉의 직접 메시지 범위 대조 |
| Fixed-wedge | T̄=Q̄ | 고정 경로 연산 |
| Learned-wedge raw | T̄=T̄_C, raw gate | 학습 가중치 |
| Learned-wedge RMS | T̄=T̄_C, RMS gate | 배율 안정성과 분류 효과 |
| Fixed-wedge + node MLP | U=(I−αL̄−βQ̄)Z+βf(Z) | 추가 용량 대조 |
| Standard GCN | U=D̃⁻¹/²(Adj+I)D̃⁻¹/²Z | 표준 GCN 전파 기준 |

GCN도 hidden 64·같은 dropout·Adam·epoch·seed·튜닝 예산을 쓴다.
[GCN 저자 코드](https://raw.githubusercontent.com/tkipf/gcn/master/gcn/train.py)의 lr 0.01·dropout 0.5·L2 5e-4를 참고했다.
저자 설정 hidden 16·200 epoch·early stopping을 완전히 재현한 모델이라고 부르지 않는다.
Hidden 64는 citation 입력의 공통 투영 크기로 정하며 gate hidden 64와 분리된 설정이다.
500 epoch는 gate·투영·분기 계수의 공동 학습 기간을 충분히 두고 모든 조건의 update 수를 맞추는 예산이다.

f는 노드별 F→128→F MLP이며 첫 Linear에만 bias, 중간 ReLU다.
βf(Z)를 실제 예측 경로에 더하며 별도 scalar를 추가하지 않는다.
Gate와 f의 추가 parameter는 층마다 동일한 64(4F+2)개다.

| 전체 trainable 수 | Cora | CiteSeer | PubMed |
| --- | ---: | ---: | ---: |
| MLP / GCN | 92,160 | 237,376 | 32,192 |
| First-order | 92,162 | 237,378 | 32,194 |
| Polynomial / Fixed | 92,164 | 237,380 | 32,196 |
| Learned raw / RMS / Fixed + node MLP | 110,596 | 255,556 | 49,604 |

Projection D×64+64×K, gate/node MLP 합계 64(4·64+2)+64(4K+2),
이차 조건의 분기 scalar는 두 층 합계 4개다. Dummy parameter를 넣지 않는다.
Parameter 수 일치가 함수의 표현력 일치를 뜻하지는 않는다.
Node MLP를 더한 전체 update에는 graph operator의 contraction bound를 적용하지 않는다.

## 6. 학습 예산·선택 절차

**본학습은 서버에서만 실행한다.** 각 run은 전체 그래프로 500 epoch·500 update를 수행한다.
Adam, lr {0.001, 0.003, 0.01}, dropout 0.5, weight decay 5e-4다.
Adam coupled L2는 projection·gate·node MLP의 weight matrix에만 적용하고 bias·분기 scalar는 제외한다.
Scheduler·gradient accumulation·early stopping·gradient clipping은 사용하지 않는다.
Float32, TF32 off. Xavier uniform, bias 0, 분기 scalar는 위 초기화 규약을 따른다.

| 단계 | seed | run 수 | update 수 |
| --- | --- | ---: | ---: |
| Validation 학습률 선택 | 101, 202, 303 | 3데이터×8조건×3lr×3seed = 216 | 108,000 |
| 최종 반복 | 11, 23, 37, 53, 71 | 3데이터×8조건×5seed = 120 | 60,000 |
| 합계 | 단계 간 독립 | **336** | **168,000** |

Checkpoint는 epoch별 validation CE 최소, 동률이면 validation accuracy 최대, 다시 동률이면 이른 epoch를 고른다.
Validation CE에는 L2 penalty를 더하지 않는다.
조건·데이터별 lr는 튜닝 3seed의 선택 checkpoint validation CE 평균으로 고르며 동률이면 작은 lr다.
선택 목록·hash를 저장하고 별도 final seed로 새 학습한다.
모든 final checkpoint를 validation으로 확정한 뒤 test label을 평가기에 제공한다. 튜닝 중 test metric을 계산하지 않는다.

Raw/RMS는 seed별 동일 초기 projection·gate를 쓰며 다른 조건도 같은 shape의 backbone 초기값을 공유한다.
Dropout mask는 dataset·seed·epoch·층의 stream으로 고정해 조건·lr 사이에서 공유한다.
최종 표에는 선택 lr를 표시하고, 튜닝 단계의 같은 lr validation 대응 결과도 보존한다.
두 모델의 조건별 최적 lr가 다를 수 있으므로 final 대응 차이는 동일 튜닝 예산으로 선택한 전체 학습 절차의 비교다.
계획한 run/update coverage가 다르면 완료로 표시하지 않는다.

## 7. 평가와 고정 개입

주지표 test accuracy, 보조지표 CE. 각 seed·평균·표본 std와 같은 seed의
learned−fixed / learned−polynomial / learned−capacity / RMS−raw 차이를 저장한다.
Paired 95% t 구간은 고정 split의 초기화 변동만 나타낸다. 새 graph/split의 불확실성을 뜻하지 않는다.
세 데이터의 노드·seed를 한 표본 집합으로 합치지 않는다.

Epoch별 loss·validation, 실제 gate gradient·parameter update, 층별 Cmean/std/min/max,
κ·α·β와 branch norm을 기록한다. C 분산만으로 유용성을 판단하지 않는다.
C 정답이 없는 실제 데이터에 C 회수 오차·teacher 상관을 만들지 않는다.

선택 raw/RMS checkpoint는 update 0으로 고정하고 두 층에 다음을 적용한다.

1. **C=1**.
2. **C 위치 shuffle**: histogram을 유지한다.
3. **이차 분기 제거**: βT̄만 0으로 하고 남은 branch 비율을 다시 맞추지 않는다.
4. **경로 대응 무작위화**: 같은 수의 distinct physical edge-pair와 row norm √6인 A_rand를 고정 manifest로 만든다.
   Gate는 현재 Z의 true wedge에서 C를 계산하고 row 순서에 대응해 A_rand에 적용한다.

Cmean=1이므로 mean 개입은 C=1과 같아 중복 효과로 세지 않는다.
Shuffle/random은 dataset별 미리 고정한 10개 manifest를 사용해 seed 안에서 평균낸다.
10개를 학습 seed 50개로 세지 않는다. Random의 edge 사용 빈도·support·거리·norm 변화도 기록한다.
악화를 경로 연속성만의 인과 효과로 단정하지 않는다.

C=1/shuffle은 (a) 개입 C로 κ를 다시 계산한 모델 규칙,
(b) 각 층의 개입 직전 C로 구한 κ를 유지한 강도 대조를 모두 평가한다.
Random은 true-wedge S_Q·개입 직전 κ를 유지하는 진단으로 실행하며 원래 3-support PSD bound를 적용하지 않는다.
κ를 고정한 shuffle도 norm≤1을 보장하지 않는다. κ 고정 진단의 실제 operator norm·branch 강도를 기록한다.
다음 층의 C는 변경된 현재 Z에서 계산한다. Clean forward의 두 층 C를 통째로 재사용하지 않는다.
Random-pair 재학습은 이번 336 run에 포함되지 않는다.

배율 0.25/0.5/1/2/4는 eval mode의 고정된 층 입력 Z에서 C 변화·κ 변화·메시지 비례성 오차를 검사한다.
기준 norm이 0이면 relative 값은 undefined다. 최종 accuracy·CE 배율 반응은 별도 관측값이며 확률 불변성을 주장하지 않는다.
최종 배율 평가의 입력은 **전처리 완료된 X**를 aX로 바꾼 것이며 row-sum 정규화를 다시 하지 않는다. Dropout은 끈다.

## 8. 판정

- Fixed보다 개선하며 C 개입에도 반응하면 학습 가중치를 유용하게 사용한 근거가 된다.
- Polynomial·용량 대조와의 차이로 직접 support·추가 용량으로 설명되는 범위를 확인한다.
- Fixed와 동률이며 개입도 무영향이면 C의 유효한 분류 기여 근거가 부족하다.
- 개입으로 악화돼도 baseline보다 성능이 나쁘면 유용한 모델이라고 결론내릴 수 없다.
- RMS가 배율 검사만 통과하고 분류가 같거나 나쁘면 안정성과 분류 효과가 분리된 결과다.
- 데이터마다 방향이 다르거나 차이가 seed 변동보다 작으면 그 범위를 그대로 보고한다.

불리한 seed/데이터를 제거하거나 test 결과로 추가 설정을 탐색하지 않는다.
임의 spectral 함수 전부의 우위나 사이클에서 잃은 메시지 복원을 주장하지 않는다.

## 9. A6000 구현·실행 순서

1. 공식 전체 데이터·split·hash를 검증하고 N/E/P·degree·고립 노드·RAM을 측정한다.
2. 별도 DEBUG로 독립 dense 참조 출력·gradient, C=1 동일성, 방향·경로 뒤집기, RMS 성질,
   실제 train-mask CE→gate/node MLP/scalar gradient→optimizer update를 검사한다.
3. 서버 GPU/MIG/free VRAM·CPU affinity/quota·RAM·storage를 출력한다.
   전체 graph forward/backward에서 tuning의 packed run 후보 1/2/3, final의 1/2/4/5를 측정한다.
   같은 dataset·condition·lr의 실제 seed 수 안에서 처리량과 peak VRAM으로 고른다.
4. 216 tuning run을 수행하고 validation 선택 목록을 고정한다.
5. 120 final run을 수행하고 checkpoint를 고정한다.
6. Test·고정 개입·배율·전체 coverage·hash를 확인해 보고서를 만든다.

전체 graph 1개가 데이터 단위인 full-batch 학습이다. 각 run의 physical/effective graph batch=1,
gradient accumulation=1, data-parallel replica=1이며 packed 동시 run 수는 별도로 기록한다.
독립 seed의 parameter·Adam 상태를 앞 축으로 pack하고 각 seed의 train-node 평균 CE를 **합산**한다.
독립 실행과 packed 실행의 한 step 출력·gradient·optimizer state 일치를 검사한다.

Raw 특징·CSR edge·모든 path index·S_d·S_Q는 cache한다.
모든 wedge를 유지하며 필요하면 exact chunking·activation checkpointing을 적용한다.
σ·Cmean·κ는 모든 chunk를 합쳐 계산하고 chunk별 정규화를 하지 않는다. Chunk는 sampling이 아니며 sampling ratio=1이다.
독립 seed·노드·channel을 Python으로 하나씩 GPU에 전달하지 않는다.
CPU 전처리 worker 1/2/4/8을 계측한다. 상주 GPU epoch에 DataLoader가 필요하지 않으면 N/A로 기록한다.
여러 할당 GPU는 dataset·condition·lr group을 분배한다.
전체 소요시간은 실제 epoch 처리량으로 산출한다. 같은 터미널에 phase·run·epoch·loss·validation·초/epoch·VRAM을 출력하고 log에도 저장한다.
새 output directory만 허용하고 resume는 동일 계약·hash를 확인한 새 폴더에 저장한다.

구현한 산출물은 full/debug config, 독립 loader/model/study/evaluation/report, data manifest·자원 calibration,
epoch history·validation 선택 목록·final checkpoint, metric/paired/C/개입/배율 CSV·scientific PNG/PDF,
contract/coverage/completion/failure JSON과 서버 실행 문서다.

`design_contract.json`은 설계 계약이고 `config_full.json`은 서버 실행 설정이다.
수학 DEBUG는 [DESIGN_VERIFICATION.md](DESIGN_VERIFICATION.md), 구현·데이터·실행 검증은
[VERIFICATION.md](VERIFICATION.md)에 구분해 기록한다. 서버 명령은 [README.md](README.md)에 있다.
전체 336 run citation 학습·최종 평가는 서버에서 실행할 대상으로 남아 있다.
