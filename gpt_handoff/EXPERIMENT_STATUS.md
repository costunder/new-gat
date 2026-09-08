# 실험 결과와 구현 상태

기준일: 2026-09-08 (Asia/Seoul).

### 최신 수령: 기존 corrected checkpoint의 20조건 validation 감사

사용자가 전달한 전체 감사 출력은 실행 20/20 성공이며 새 학습/test가 아니다.
아래는 해당 감사에서 다시 평가한 validation %, PPI는 micro-F1이고 나머지는 accuracy다.
과거 저장된 best 점수와 재평가값의 작은 수치 변동을 섞지 않는다.

| Dataset | Profile | Learned C | 같은 checkpoint C=1 | 별도 학습 fixed-C |
| --- | --- | ---: | ---: | ---: |
| Cora | reference | 75.6000 | 74.6000 | 75.2000 |
| Cora | large | 76.0000 | 75.6000 | 75.2000 |
| CiteSeer | reference | 65.4000 | 65.8000 | 65.8000 |
| CiteSeer | large | 66.0000 | 66.0000 | 68.0000 |
| PubMed | reference | 74.2000 | 74.0000 | 75.6000 |
| PubMed | large | 74.0000 | 73.6000 | 73.6000 |
| ogbn-arxiv | reference | 71.7071 | 71.0829 | 71.5427 |
| ogbn-arxiv | large | 71.9017 | 70.9957 | 71.8279 |
| PPI | reference | 98.5311 | 76.0396 | 98.4048 |
| PPI | large | 98.6500 | 72.2789 | 98.6123 |

PPI C 제거 개입의 약 22.49/26.37%p 하락은 checkpoint의 C 의존성이지, 독립 fixed-C보다
그만큼 개선됐다는 뜻이 아니다. 독립 fixed 대비 차이는 약 +0.1263/+0.0377%p다.
arxiv도 C는 사용하지만 별도 fixed 대비 작은 차이이고 citation 결과는 혼재한다.

K8의 층×그래프 잔차 120개는 모두 1e-4 미달이다. K64에서는 citation/arxiv 80개 충족,
PPI 40개 미달(약 0.000347~0.001974)이다. 평가 K만 늘린 10조건 점수는 모두 소폭
하락했지만 K64로 공동 학습한 대조군이 아니므로 solver 정확도가 불필요하다는 증명이 아니다.

구형 감사에는 **max-abs>0만으로 contribution=observed를 표시하는 판정 오류**가 있다.
fixed-C 10조건도 모두 observed였고, 동일 C 개입에서 PubMed large는 0.2%p 변동했다.
새 감사는 같은 checkpoint의 무개입 반복 변동과 fixed not-applicable을 명시한다.
raw C 분위수/구간 비율을 당시 출력하지 않았으므로 CV/베타만으로 C>=0.7 비율을
추론하지 않는다. graph 평균 C=1은 제약이지 개별 엣지가 전부 1이라는 증거가 아니다.

arxiv reference의 기록에는 최대 169,343 노드로 전체 그래프까지 포화된 배치가 있다.
모든 조건의 overlap은 미기록이므로 새 disjoint 학습의 다양성 검증으로 취급하지 않는다.
GPU SM utilization은 pynvml/안전한 장치 매핑 부재로 unavailable였다. CUDA 사용 자체는
allocator 측정으로 확인되지만 해당 출력으로 GPU 사용률을 확정하지 않는다.

이후 승인된 구현: [멀티 C·정규화·분포·실행 계약](CONDUCTANCE_V5.md#multi-c-mechanisms-20260908).
기존 결과를 보존하고 별도 mechanism namespace로 12조건 비교를 구성했다. 새 구조의
실제 GPU 실측·전체 학습·test 결과는 아직 없다. 아래 이전 상태 문구는 당시 시점의 기록이다.

### 후속 코드 수정 상태

아래 결과 문서화 이후 피드백을 반영해 독립 부분 그래프 배치, sampling overlap 기록,
기존 checkpoint의 C=1 개입 및 frozen-H K 참조 검사를 구현했다.
구현 범위·검사 명령·새 실험과 기존 checkpoint의 경계는
[V5 피드백 반영 절](CONDUCTANCE_V5.md#feedback-implementation-20260908)에 모았다.
아래 학습/test 결과는 그대로 보존하며, 새 sampling의 성능 결과로 재분류하지 않는다.
추가 전체 학습과 실제 서버 checkpoint 감사는 아직 실행하지 않았다.

<a id="v5-audit-20260908"></a>

## 2026-09-08 최신: corrected V5 전체 결과와 C 학습 검증 감사

### 결론과 증거의 경계

**20조건의 학습 절차는 완료됐지만, 학습 C의 효과와 sampled-Laplacian 연구 가설이
검증 완료된 상태는 아니다.** PPI에서는 학습·일반화가 뚜렷하게 개선되었다.
Cora/CiteSeer/PubMed/arxiv에는 train loss가 계속 낮아지는 동안 validation이 정체·악화되는
양상이 있다. Fixed-C도 같은 문제를 보이므로 낮은 성적의 원인을 C 하나로 단정할 수 없다.
Dynamic-C의 paired best-validation 차이는 작거나 음수이며, test는 dynamic 10조건만 수령했다.

이번 기록은 **문서화**다. 모델·설정·실험 결과·checkpoint를 수정하지 않았고,
새 학습·GPU 평가·추가 seed 실험·Git push를 수행하지 않았다.
아래 2026-09-07 이하의 ‘GPU 결과 미검증/미수령’은 당시 기록이다.
과거 선택적 전환 run의 `historical_reference=2 / pending_extra_budget=1 / passed=17`과
이번 corrected run의 **20 passed**를 섞지 않는다. Cycle/Tree/V1–V4의 새 결과를 수령한 것도 아니다.

증거는 다음과 같이 구분한다.

- 결과 run: `corrected-c-v5-a6000-gpu3-seed0-v1-conductance`.
- 서버 결과 root:
  `/home/aicompetition07/new-gat/results/conductance_gat/scaling/corrected-c-v5-a6000-gpu3-seed0-v1-conductance`.
  조건별 경로는 `v5/{profile}/model-seed-0/{dataset}/{condition}/`.
- 검토한 로컬 코드 기준: `8da06cacec59515d84c08d892315e4c8ecfd5b5b`.
  이것은 모든 epoch가 이 한 소스로 처음부터 생성됐다는 뜻이 아니다. 특히 arxiv reference
  dynamic history에는 성능 수정 전후의 연속 기록이 포함된다.
- 사용자 첨부 history: `06191601-f6e4-4c6d-a0e2-2ee42788626d/pasted-text.txt`.
  SHA256: `97DEBC3D99DDD76A90F99F73926B7EB4B84222D3D9ECE6C8CADBCE154A10074D`.
- 사용자 첨부 test: `5c28841b-710a-4c42-9f2a-c0e90fe02492/pasted-text.txt`.
  SHA256: `4B1CE80FB7E9126F158ADFADBBD78E413E83805F1BEB3F2918685E96960521DB`.
  원본 위치는 각 ID 아래의 `C:/Users/lock1/.codex/attachments/`다.
- 두 첨부를 독립적으로 교차 파싱했다. history는 **20조건, 3,832행**, epoch 1부터
  각 last epoch까지 누락 없이 연속이다. 전체 원문을 [에포크 부록](#v5-epoch-archive-20260908)에
  보존하고, test 결과 10개의 원시 수치·checkpoint SHA를 [test 부록](#v5-test-archive-20260908)에 둔다.
- 수치는 사용자가 제공한 **서버 출력**에서 확인했다. 서버의 원본 best.pt/last.pt,
  manifest/metrics/history 전체를 로컬에 받아 실제 데이터로 독립 재실행한 검증은 아니다.
  Test 명령은 `test_only`, `optimizer_steps=0`, `files_written=false`를 보고했다.
  이번 test 점수가 서버 metrics.json에 자동 저장됐다고 보고하면 안 된다.

### 실제 설정과 데이터 범위

| 항목 | reference | large |
|---|---|---|
| Hidden / layers / heads / FFN multiplier | 256 / 8 / 8 / 4 | 384 / 12 / 8 / 4 |
| Dropout / optimizer | 0.2 / AdamW | 0.2 / AdamW |
| C backend / training schedule | optimization / joint | optimization / joint |
| C cost / solver steps | width_scaled / 8 | width_scaled / 8 |
| Solver step upper bound / entropy / degree barrier / cost bound | 0.25 / 1.0 / 0.1 / 2.0 | 동일 |
| W/backbone/C/beta LR | 0.0005(C/beta multiplier 1) | 동일 |
| W/backbone decay / C decay / beta decay | 0.01 / 0 / 0 | 동일 |
| Beta / initialization | sigmoid / 0.5(강제 margin 없음) | 동일 |
| Gradient clipping / precision | global norm 5 / BF16 dense·FP32 C geometry, TF32 on | 동일 |
| Edge chunk / block activation checkpoint | 131072 / false | 동일 |
| Model seed / requested epochs / patience | 0 / 200 / 50、reference_updates | 동일 |
| arxiv physical supervised seed batch / batches per epoch | 8192 / 12 | 2048 / 45 |
| PPI physical train graph batch / batches per epoch | 20 / 1 | 20 / 1 |

표의 '동일'는 왼쪽과 동일함을 뜻한다. C 내부 chunk/unroll checkpoint는
block `activation_checkpoint=false`와 별도로 사용된다.
현재 C는 feature head마다 나누지 않고 공유하며, W와 beta는 head별 구조를 갖는다.
모델별 실제 parameter 수는 test 표에 기록했다. 대략 reference 955만–1,047만,
large 3,147만–3,284만으로, 과거 64채널/2층 mechanism probe의 성적이 아니다.

Cora/CiteSeer/PubMed는 Planetoid public split(각 train 140/120/60, val 500,
test 1000), NormalizeFeatures, full graph다. arxiv는 공식 time split과 기본 128차원
특징을 사용하고 train에서는 cluster, validation/test에서는 전체 그래프를 사용한다.
PPI는 공식 train/val/test 20/2/2 graphs이고 graph mini-batch이지 각 그래프 내부 B 샘플링이 아니다.
별도 PE나 외부 텍스트 모델 특징을 이 corrected Conductance 실행에 추가한 것은 아니다.

저장 설정에서 PPI는 workers 8 / persistent_workers true / prefetch_factor 2,
pin_memory true다. Transductive full/sampled graph는 일반 DataLoader worker를 쓰지 않아
workers 0으로 기록되며, sampled path의 별도 prefetch 및 정적 graph cache와 구분한다.
workers 0 또는 PPI Batches=1만으로 직렬 graph 학습이라고 판정하지 않는다.
단, 이번 붙여넣기만으로 calibration의 모든 후보·peak train VRAM·구간별 처리량을
재검증한 것은 아니다. 새 GPU 자원 최적화 인증으로 사용하지 않는다.

### 전체 20조건 학습 요약

모두 `passed`, seed 0, phase `joint`다. PPI는 micro-F1, 나머지는 accuracy.
점수 단위는 **%**, loss는 원래 단위다. 아래 수치는 출력 정밀도를 따른다.
`0.0000`은 반올림 결과이지 정확히 0이라는 뜻이 아니다.
시간 합계는 표시된 epoch 시간의 합이며 setup/calibration/checkpoint 등 모든 비용을 포함한
run 전체 wall time이 아니다. 특히 arxiv reference dynamic 합계는 느린/빠른 구간이 섞였다.

| Dataset | Profile | C | Best / last epoch | Best / last val (%) | Loss at best / last | Batches/epoch | Final updates | 기록 epoch 시간 합계 (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cora | reference | fixed | 12 / 62 | 75.200 / 74.400 | 0.0029 / 0.0006 | 1 | 62 | 5.68 |
| Cora | reference | dynamic | 11 / 61 | 75.600 / 73.600 | 0.0033 / 0.0006 | 1 | 61 | 20.70 |
| Cora | large | fixed | 6 / 56 | 75.200 / 75.000 | 0.0139 / 0.0002 | 1 | 56 | 6.67 |
| Cora | large | dynamic | 111 / 161 | 76.000 / 75.600 | 0.0001 / 0.0001 | 1 | 161 | 76.64 |
| CiteSeer | reference | fixed | 25 / 75 | 65.800 / 65.600 | 0.0010 / 0.0004 | 1 | 75 | 6.88 |
| CiteSeer | reference | dynamic | 68 / 118 | 65.400 / 65.000 | 0.0005 / 0.0003 | 1 | 118 | 39.61 |
| CiteSeer | large | fixed | 15 / 65 | 68.000 / 64.000 | 0.0013 / 0.0002 | 1 | 65 | 7.71 |
| CiteSeer | large | dynamic | 4 / 54 | 66.000 / 63.600 | 0.1045 / 0.0002 | 1 | 54 | 25.67 |
| PubMed | reference | fixed | 47 / 97 | 75.600 / 74.000 | 0.0001 / 0.0001 | 1 | 97 | 12.21 |
| PubMed | reference | dynamic | 10 / 60 | 74.200 / 73.600 | 0.0003 / 0.0001 | 1 | 60 | 21.05 |
| PubMed | large | fixed | 7 / 57 | 73.600 / 70.000 | 0.0008 / 0.0000 | 1 | 57 | 14.32 |
| PubMed | large | dynamic | 6 / 56 | 74.000 / 70.000 | 0.0016 / 0.0000 | 1 | 56 | 29.62 |
| PPI | reference | fixed | 599 / 600 | 98.408 / 98.404 | 0.0265 / 0.0264 | 1 | 600 | 260.88 |
| PPI | reference | dynamic | 597 / 600 | 98.531 / 98.526 | 0.0232 / 0.0231 | 1 | 600 | 2780.30 |
| PPI | large | fixed | 586 / 600 | 98.612 / 98.597 | 0.0127 / 0.0121 | 1 | 600 | 509.77 |
| PPI | large | dynamic | 600 / 600 | 98.650 / 98.650 | 0.0111 / 0.0111 | 1 | 600 | 4525.29 |
| ogbn-arxiv | reference | fixed | 13 / 201 | 71.549 / 67.888 | 0.7605 / 0.0294 | 12 | 2412 | 2672.78 |
| ogbn-arxiv | reference | dynamic | 13 / 201 | 71.700 / 68.442 | 0.7284 / 0.0293 | 12 | 2412 | 54992.98 |
| ogbn-arxiv | large | fixed | 4 / 54 | 71.838 / 67.043 | 0.9381 / 0.0431 | 45 | 2430 | 1459.36 |
| ogbn-arxiv | large | dynamic | 4 / 54 | 71.898 / 67.378 | 0.9488 / 0.0426 | 45 | 2430 | 4145.47 |

### Fixed-C 대비 추가 효과: validation만의 비교

| Dataset | reference: fixed → dynamic | Δ (%p) | large: fixed → dynamic | Δ (%p) |
|---|---:|---:|---:|---:|
| Cora | 75.200 → 75.600 | +0.400 | 75.200 → 76.000 | +0.800 |
| CiteSeer | 65.800 → 65.400 | -0.400 | 68.000 → 66.000 | -2.000 |
| PubMed | 75.600 → 74.200 | -1.400 | 73.600 → 74.000 | +0.400 |
| ogbn-arxiv | 71.549 → 71.700 | +0.151 | 71.838 → 71.898 | +0.060 |
| PPI | 98.408 → 98.531 | +0.123 | 98.612 → 98.650 | +0.038 |

이 표는 같은 데이터셋/profile의 관측 비교다. 단일 seed에서 작은 차이의 유의성이나
C의 일반적인 우수성을 주장하지 않는다. Fixed-C는 V5의 C=1 대조군이지
일반 논문의 vanilla GCN과 동일한 모델이 아니다. Cross-profile arxiv는 seed batch도 달라
폭·깊이 하나만의 효과로 해석할 수 없다.

### Dynamic-C 공식 test 10조건

모두 validation으로 선택한 best checkpoint에 대한 test이며 last epoch의 test가 아니다.
PPI는 micro-F1, 나머지는 accuracy. 표는 %로 환산했고 원시 비율은 부록에 보존한다.

| Dataset | Profile | Best epoch | Selected val (%) | Test (%) | Parameters |
|---|---|---:|---:|---:|---:|
| Cora | reference | 11 | 75.6000 | 76.8000 | 9,880,377 |
| Cora | large | 111 | 76.0000 | 75.4000 | 31,963,961 |
| CiteSeer | reference | 68 | 65.4000 | 67.0000 | 10,465,780 |
| CiteSeer | large | 4 | 66.0000 | 61.4000 | 32,839,796 |
| PubMed | reference | 10 | 74.2000 | 73.6000 | 9,638,635 |
| PubMed | large | 6 | 74.0000 | 70.0000 | 31,602,283 |
| ogbn-arxiv | reference | 13 | 71.7004 | 70.4010 | 9,552,168 |
| ogbn-arxiv | large | 4 | 71.8984 | 71.1849 | 31,472,936 |
| PPI | reference | 597 | 98.5313 | 99.0947 | 9,552,861 |
| PPI | large | 600 | 98.6498 | 99.2076 | 31,474,013 |

- **Fixed-C test 값은 미수령**이다. 위 validation 차이를 test 개선량으로 옮기지 않는다.
- PPI test 99.0947/99.2076%는 Cora accuracy와 서로 다른 과제/metric이다.
  PPI의 높은 절대 점수만으로 C가 기여했다고 말할 수도 없다.
- 앞서 구두로 제시된 CiteSeer 0.68은 large **fixed-C validation**,
  PubMed 0.756은 reference **fixed-C validation**이다. Dynamic-C/test와 혼동하지 않는다.
- Cora/CiteSeer/PubMed는 test node 1000개, arxiv는 48,603개를 평가했다.
  PPI는 공식 test graph 2개, 5,524 nodes × 121 labels = 668,404 node-label entries다.
  출력의 `official_graphs` 24개 통계는 전체 split 목록이며 24개를 test에 사용했다는 뜻이 아니다.
- Test-only 경로는 저장된 checkpoint identity/SHA/selected epoch·validation 및
  source/data/split compatibility를 검사하고 optimizer 없이 평가하도록 작성됐다.
  이 설명은 코드 경로와 제공된 성공 출력의 해석이며, 로컬에서 서버 checkpoint의 SHA를
  새로 검증했다는 주장이 아니다.
- 단일 seed·서로 다른 레시피의 논문 수치와 단순 순위 비교하여 SOTA라고 선언하지 않는다.
  향후 모델 선택/튜닝에는 validation을 사용하고 이미 확인한 test에 맞춰 조정하지 않는다.

### 에포크별 진행의 해석과 예산

1. **Citation 데이터: 훈련 적합은 매우 강하지만 validation 일반화가 따라오지 않는다.**
   예를 들어 CiteSeer large dynamic은 epoch 4의 loss 0.1045 / val 66.000%에서
   epoch 54의 loss 0.0002 / val 63.600%로 간다. PubMed large dynamic은
   epoch 6의 val 74.000%에서 epoch 56의 70.000%로 떨어진다.
   이는 과적합/일반화 악화 양상이다. C collapse나 oversmoothing의 원인 확정은 아니다.
   Citation 모든 조건은 selected epoch 이후 50 epochs에서 끝난다.
2. **arxiv도 train loss 감소와 validation 하락이 분리된다.**
   reference dynamic은 epoch 13의 loss 0.7284 / val 71.700%에서
   epoch 201의 0.0293 / 68.442%로 간다. Large dynamic은 epoch 4의
   0.9488 / 71.898%에서 epoch 54의 0.0426 / 67.378%로 간다.
   Fixed-C에서도 유사한 악화가 있다. 더 오래 돌리기만 하면 해결된다는 근거는 없다.
3. **PPI는 개선 추세가 다르다.**
   reference dynamic validation은 epoch 200/300/400/500에서
   81.182/94.462/97.705/98.340%, best 597에서 98.531%다.
   large dynamic은 같은 지점에서 82.565/96.771/98.424/98.610%,
   epoch 600에서 98.650%다. Loss와 validation이 함께 개선된다.
   large의 best가 마지막 epoch라는 사실만으로 완전 수렴을 선언하지 않는다.
4. **600 epochs는 임의의 재학습이 아니라 reference update 예산이다.**
   PPI 20 train graphs에서 기준 batch 8은 3 updates/epoch × 요청 200 = 600 updates다.
   선택 physical batch 20은 1 update/epoch이므로 같은 최대 update 수에 600 epochs가 필요하다.
   `Batches=1`은 **20개 graph를 한 번에 학습**한다는 뜻이다.
   Effective batch는 physical × accumulation × data-parallel workers이며, 기록된 실행은
   optimizer update를 매 batch 수행하는 단일 CUDA-device 경로다.
5. **arxiv의 best epoch 13 대 4를 그대로 속도/수렴 비교하지 않는다.**
   reference는 13×12=156 updates, large는 4×45=180 updates에서 best다.
   Stale updates는 reference (201−13)×12=2256, large (54−4)×45=2250으로,
   기준 50×45=2250 updates의 patience 및 epoch 경계 검사와 맞는다.
   reference의 계획상 최대 750 epochs/9000 updates는 조기종료를 금지한다는 뜻이 아니다.
6. 에포크 1은 한 optimizer step 직전이 아니라 **그 epoch의 모든 train batches 뒤**의
   validation이다. arxiv reference는 이미 12 updates를 거친 값이다.
   큰 epoch-1 점수만으로 pretrained model 사용이나 label leakage를 단정하지 않는다.
   전체 graph의 unlabeled node 특징/연결을 사용하는 transductive 설정과 test label 학습은 다르다.

### 실행 시간과 자원: 해결된 부분과 남은 경계

| arxiv reference dynamic epoch | Batches | 누적 updates | 기록 시간 (s) |
|---|---:|---:|---:|
| 77 | 12 | 924 | 667.54 |
| 78 | 12 | 936 | 30.86 |
| 79 | 12 | 948 | 29.13 |
| 80 | 12 | 960 | 29.00 |
| 201 | 12 | 2412 | 29.10 |

약 23배의 **관측된 epoch 시간** 개선이 있고, epoch/update가 끊기거나 batch가 줄어든 흔적은
이 표에 없다. 성능 패치 후 기록과 부합하지만 통제된 GPU A/B profiler 결과는 아니다.
전체 201 epochs 평균을 현재 속도로 쓰거나 667초를 12로 나눠 순수 GPU batch 시간이라고
보고하지 않는다. 해당 epoch 시간은 train·validation·diagnostics를 포함하며 checkpoint commit
등의 범위는 별도다. Speed 개선 후에도 best validation 71.700%@13을 갱신하지 못했다.

PPI의 600 epochs 기록 평균은 reference fixed 0.435초 / dynamic 4.634초(약 10.66배),
large fixed 0.850초 / dynamic 7.542초(약 8.88배)다.
그에 비해 best-validation 추가 이득은 +0.123/+0.038%p다.
이것은 총 epoch 비용 대비 추가 효과의 문제이며 C kernel만 분리한 실행 시간은 아니다.

Test 로그의 condition별 peak allocated VRAM 최대는 arxiv large **3.137251 GiB**다.
맨 끝 resource summary의 **0.319221 GiB**는 condition마다 peak를 reset한 뒤 마지막
PubMed reference에 해당한다. 이를 전체 10조건 최고치로 보고하는 것은 잘못이다.
추론 VRAM을 training의 backward/optimizer 메모리 요구량과 동일시하지 않는다.
GPU utilization null은 pynvml 사용 불가와 숫자 CUDA ordinal→nvidia-smi UUID 안전 매핑 불가다.
0% 사용이나 test 실패가 아니며, 측정하지 못한 utilization을 임의로 채우지 않는다.

서버 test 출력의 논리/할당 CPU는 96, system RAM은 1,081,814,663,168 bytes,
available RAM은 1,000,269,946,880 bytes, visible CUDA device total은
47,839,313,920 bytes다. A6000 식별은 앞서 수령한 nvidia-smi 로그와 연결되는 맥락이며,
로컬 CPU-only 환경을 서버 GPU로 오인하지 않는다. 이 snapshot은 전체 학습의 평균 CPU/I/O나
배치 최적화 검증을 대신하지 않는다.

### 문제 목록: 관측, 코드 사실, 가설을 분리

| ID | 판정 | 문제와 근거 | 아직 단정할 수 없는 것 |
|---|---|---|---|
| V5-R01 | 관측 | Citation/arxiv에서 train loss 감소와 validation 악화; fixed도 부진 | C만이 원인, 반드시 oversmoothing/누수라는 진단 |
| V5-R02 | 관측 | Dynamic−fixed best val −2.000~+0.800%p, PPI 이득은 +0.123/+0.038%p | C가 일관되게 유용하거나 전혀 학습되지 않았다는 주장 |
| V5-R03 | 관측 | PPI dynamic epoch 비용 약 9~11배; arxiv 성능 패치 후에도 정확도 개선 미관측 | wall time 전부가 solver의 특정 한 연산 때문이라는 주장 |
| V5-R04 | 코드 사실·미검증 | entropy 1/cost bound 2/평균 C=1/finite K8의 표현력과 최적화 제약 | 이것들이 실제 C 붕괴의 원인이라는 확정 |
| V5-R05 | 코드 사실·미검증 | 목적함수 감소/finite residual을 검사하지만 실제 K8 잔차의 충분성 통과 기준 없음 | K8이 실제 데이터에서 수렴했거나 실패했다는 확정 |
| V5-R06 | 코드 사실·미검증 | C/β 통계·마지막 train batch gradient·validation 개입은 기록 경로가 있지만 제공 표에는 없음 | C=1 붕괴, beta=0, 장기 gradient 소실을 이미 측정했다는 주장 |
| V5-R07 | 범위 불일치 | auto B 샘플링은 arxiv만; citation full graph/PPI 원그래프 batch | 모든 데이터셋에서 다양한 부분 B를 학습했다는 주장 |
| V5-R08 | 코드 사실·미검증 | arxiv ref seed 8192×26=212992>N=169343으로 full batch의 cluster 예산 포화 | 충분한 국소 구조 다양성이 확보됐거나 이것이 낮은 정확도의 단독 원인이라는 주장 |
| V5-R09 | 미검증 | degree-ratio 보정은 근사; sampled/full C·정규화 연산·gradient·주파수 응답 비교 없음 | unbiased estimator나 spectral approximation 보증 |
| V5-R10 | 코드 사실·미검증 | C/W 동일 LR, 고정 optimizer recipe; residual/FFN·beta가 전파 기여를 약하게 만들 가능성 | 동일 LR 자체가 버그라거나 bypass가 실제 지배한다는 단정 |
| V5-R11 | 증거 한계 | seed 0, dynamic-only test, cross-profile batch 차이, 수령 출력 중심 | paired test 우월성, 통계적 유의성, SOTA, 폭 하나의 인과효과 |
| V5-R12 | 보고/계측 한계 | test stdout-only, 마지막 condition의 peak가 전체 summary처럼 보임, GPU utilization unavailable | test artifact 저장 완료/전체 peak/배치 최적화 실측 완료 |

핵심 수학 연결과 현재 C 목적함수, 기존 테스트 및 새 CPU 수치 점검의 정확한 범위는
[CONDUCTANCE_V5.md의 최신 감사](CONDUCTANCE_V5.md#c-learning-audit-20260908)에 기록한다.
'gradient가 연결됐다'와 'task에 유용한 C를 실제로 학습했다'는 서로 다른 판정이다.
검증이 없다는 이유로 C=1과 정확히 같다고 가정하거나, 분산을 억지로 키워 성공으로 표시하지 않는다.

### 다음 확인 순서 — 아직 실행한 실험이 아님

1. **기존 artifact부터 읽기**: selected_checkpoint_interventions의 learned/ones/mean/shuffle
   validation, layer별 C CV·범위, beta, train C/W/backbone/beta gradient,
   solver initial/final energy·projected residual·step 범위를 확인한다.
   학습 전 기간의 gradient를 한 번의 first-active 검사로 대신하지 않는다.
   서버 원본이 없거나 필드가 빠졌으면 unavailable로 남긴다. 재학습이나 test 재튜닝이 필요 없는 단계다.
2. **K8 품질을 실제 고정 입력에서 확인**: 같은 checkpoint/그래프/hidden state를 고정한
   진단용 고정밀 참조 최적화와 C·operator response·prediction/residual을 비교한다.
   참조 solver를 production 설정으로 조용히 바꾸지 않는다. 더 낮은 내부 에너지가
   무조건 더 나은 최종 task 성적이라는 가정도 금지한다.
3. **샘플링 목표를 검증**: actual Bs 크기/반복/overlap/coverage를 읽고, 부분 그래프와
   full graph의 weighted/normalized operator response 및 task gradient 오차를 조사한다.
   C가 context-dependent이므로 서로 다른 Bs에서 C가 반드시 같아야 한다고 요구하지 않는다.
   다양성 확대 설계는 별도 제안 후 승인받으며 fanout/배치/데이터를 몰래 축소하지 않는다.
4. **진단 후 원인별 통제 실험**: C/W optimizer 균형, entropy·비용 대비,
   beta·residual/FFN 기여를 분리한다. 모든 recipe를 한 번에 변경하지 않는다.
   현재 seed 0 시간 제약을 유지하고, 요청하지 않은 5-seed 마무리 실험으로 확대하지 않는다.
5. **최종 성능 주장 전**에는 fixed-C test와 protocol/selection/데이터 동일성을 확인한다.
   지금 문서화 요청은 새 평가나 재학습을 승인한 것이 아니므로 실행하지 않았다.

### 검증 완료 범위

| 구분 | 상태 |
|---|---|
| 이번 문서화 | 수령 20조건/3,832행과 10개 test 출력 교차 대조; 원문·원시 점수 보존 |
| 모델 구현 변경 | 이번 작업에서는 없음 |
| 정적/전체 회귀 | 소스 8da06ca의 기존 기록 2704 passed / 103 skipped / 10 warnings / 217.21s; 이번 문서 작업에서 재실행하지 않음 |
| CPU 수치 점검 | 앞선 읽기 전용 synthetic float64 C→BᵀCB→sparse/eigen filter 및 gradient 대조; 상세는 V5 문서 |
| GPU 전체 학습 | 사용자가 제공한 실제 데이터 서버 완료 출력 20조건; 로컬 독립 재학습 없음 |
| GPU test | 사용자가 제공한 dynamic 10조건 성공 출력; fixed test 미수령; 로컬 독립 평가 없음 |
| 연구 가설 검증 | 실제 C 기여·K8 충분성·sample/full 및 새 구조 일반화 검증 미완료 |
| 보존 | 기존 run/원본 첨부/checkpoint/학습 설정 유지, V1–V4/Cycle/Tree 기록 유지 |

<a id="v5-test-archive-20260908"></a>

### 부록 A — 수령한 test 결과 10개 원시 레코드

아래는 원본 출력에서 condition별 결과 레코드만 추출했다. 점수는 0~1 원값이며,
checkpoint SHA·평가 seconds·condition별 peak를 보존한다. 거대한 자원 snapshot과
반복 configuration JSON은 위 공통 설정/자원 절로 요약했다. 새로 생성한 평가 결과가 아니다.

```jsonl
{"status":"passed","condition":"v5/reference/model-seed-0/cora/shared_dynamic_c","metric":"accuracy","best_epoch":11,"validation":0.7560000419616699,"TEST":0.7680000066757202,"seconds":0.05193502362817526,"peak_vram_gib":0.10314416885375977,"checkpoint_sha256":"5a36d5e66cbb440e66201a4e78c71b49f508f3cb92dd9326dbbe71ae3fb204eb","files_written":false}
{"status":"passed","condition":"v5/large/model-seed-0/cora/shared_dynamic_c","metric":"accuracy","best_epoch":111,"validation":0.7600000500679016,"TEST":0.7540000081062317,"seconds":0.07621018402278423,"peak_vram_gib":0.24117803573608398,"checkpoint_sha256":"abf16df086904d23afabf8eaedfe45b48956988d497297b66a9651663b0fcbdf","files_written":false}
{"status":"passed","condition":"v5/reference/model-seed-0/citeseer/shared_dynamic_c","metric":"accuracy","best_epoch":68,"validation":0.6540000438690186,"TEST":0.6700000166893005,"seconds":0.0547667071223259,"peak_vram_gib":0.1670093536376953,"checkpoint_sha256":"c03a693daae5b1ab56d2bc8f51891c8f347e9f882c21ee80051a9777bea75245","files_written":false}
{"status":"passed","condition":"v5/large/model-seed-0/citeseer/shared_dynamic_c","metric":"accuracy","best_epoch":4,"validation":0.6600000262260437,"TEST":0.6140000224113464,"seconds":0.27215679828077555,"peak_vram_gib":0.2854652404785156,"checkpoint_sha256":"879343f1c73c1fb9537d53805e62b26a94fdaf3527a5dfe970a59668c2e820bf","files_written":false}
{"status":"passed","condition":"v5/reference/model-seed-0/pubmed/shared_dynamic_c","metric":"accuracy","best_epoch":10,"validation":0.7420000433921814,"TEST":0.7360000610351562,"seconds":0.057120177894830704,"peak_vram_gib":0.31922149658203125,"checkpoint_sha256":"d0d7dafe1bc2502e2bbda599b95c5869fc8dd393bf8c009baf6f6779c41b005d","files_written":false}
{"status":"passed","condition":"v5/large/model-seed-0/pubmed/shared_dynamic_c","metric":"accuracy","best_epoch":6,"validation":0.7400000095367432,"TEST":0.7000000476837158,"seconds":0.1058869268745184,"peak_vram_gib":0.5447773933410645,"checkpoint_sha256":"8040ba377da1c37a7e6c46d20424d5646635d2869369b59939fa9797d2ed01e0","files_written":false}
{"status":"passed","condition":"v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c","metric":"accuracy","best_epoch":13,"validation":0.7170038819313049,"TEST":0.7040100693702698,"seconds":0.618370602838695,"peak_vram_gib":2.0741987228393555,"checkpoint_sha256":"6b6df1ef8951fe2df43d4ff60170ac89c74f8c6a159b0a116f18546833cb4820","files_written":false}
{"status":"passed","condition":"v5/large/model-seed-0/ogbn-arxiv/shared_dynamic_c","metric":"accuracy","best_epoch":4,"validation":0.7189838290214539,"TEST":0.7118490934371948,"seconds":1.370499774813652,"peak_vram_gib":3.1372509002685547,"checkpoint_sha256":"e0fde800bed923576fcfa2e3d3d9e3890e3c4a690774fa97b2977f3f992bf862","files_written":false}
{"status":"passed","condition":"v5/reference/model-seed-0/ppi/shared_dynamic_c","metric":"micro_f1","best_epoch":597,"validation":0.9853127089940241,"TEST":0.9909467615530132,"seconds":0.15271377936005592,"peak_vram_gib":0.4738197326660156,"checkpoint_sha256":"b937b90dacb32036af90ff9c3af777d742fe6239fca28794b69c15c4bcb41949","files_written":false}
{"status":"passed","condition":"v5/large/model-seed-0/ppi/shared_dynamic_c","metric":"micro_f1","best_epoch":600,"validation":0.9864977029063463,"TEST":0.9920755783277445,"seconds":0.593765078112483,"peak_vram_gib":0.798861026763916,"checkpoint_sha256":"603a19fbfcf00afb4864bf4918949dac033d9341dba3b36bb3b30cd4fe49dc4b","files_written":false}
```

<a id="v5-epoch-archive-20260908"></a>

### 부록 B — 20조건, 3,832개 전체 에포크 원문

요약으로 생략하지 않은 수령 원문이다. 조건별로 접어 두었으며 순서·표시 정밀도·
selected 표시를 유지한다. 원본 첨부의 바깥쪽 공백과 줄바꿈 형식만 정리했다.
부록의 실험 수치와 아래 과거 run의 수치를 합산하지 않는다.

<details>
<summary>cora | reference | fixed_c — 전체 에포크</summary>

```text
[cora | reference | fixed_c]
status=passed | last_epoch=62 | selected_epoch=12
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.1131    61.600    61.600        1        1      0.27
  2  joint                   0.8138    67.600    67.600        1        2      0.09
  3  joint                   0.3105    71.400    71.400        1        3      0.08
  4  joint                   0.1016    73.000    73.000        1        4      0.09
  5  joint                   0.0408    73.400    73.400        1        5      0.09
  6  joint                   0.0214    73.400    73.400        1        6      0.06
  7  joint                   0.0113    74.000    74.000        1        7      0.09
  8  joint                   0.0075    74.000    74.000        1        8      0.06
  9  joint                   0.0055    74.200    74.200        1        9      0.10
 10  joint                   0.0042    74.600    74.600        1       10      0.07
 11  joint                   0.0035    74.800    74.800        1       11      0.10
 12  joint                   0.0029    75.200    75.200        1       12      0.09 *
 13  joint                   0.0025    74.600    75.200        1       13      0.06
 14  joint                   0.0022    74.600    75.200        1       14      0.09
 15  joint                   0.0020    74.400    75.200        1       15      0.09
 16  joint                   0.0018    73.800    75.200        1       16      0.09
 17  joint                   0.0017    73.800    75.200        1       17      0.10
 18  joint                   0.0015    73.600    75.200        1       18      0.10
 19  joint                   0.0014    73.800    75.200        1       19      0.10
 20  joint                   0.0013    73.800    75.200        1       20      0.09
 21  joint                   0.0013    73.800    75.200        1       21      0.09
 22  joint                   0.0012    74.000    75.200        1       22      0.09
 23  joint                   0.0012    74.200    75.200        1       23      0.10
 24  joint                   0.0011    74.200    75.200        1       24      0.06
 25  joint                   0.0011    74.000    75.200        1       25      0.09
 26  joint                   0.0010    74.200    75.200        1       26      0.09
 27  joint                   0.0010    74.000    75.200        1       27      0.09
 28  joint                   0.0010    74.000    75.200        1       28      0.10
 29  joint                   0.0009    74.000    75.200        1       29      0.10
 30  joint                   0.0009    74.000    75.200        1       30      0.09
 31  joint                   0.0009    74.000    75.200        1       31      0.09
 32  joint                   0.0009    74.000    75.200        1       32      0.10
 33  joint                   0.0009    74.000    75.200        1       33      0.10
 34  joint                   0.0008    74.000    75.200        1       34      0.09
 35  joint                   0.0008    74.000    75.200        1       35      0.10
 36  joint                   0.0008    74.000    75.200        1       36      0.09
 37  joint                   0.0008    74.000    75.200        1       37      0.09
 38  joint                   0.0008    74.000    75.200        1       38      0.09
 39  joint                   0.0008    74.200    75.200        1       39      0.09
 40  joint                   0.0008    74.400    75.200        1       40      0.09
 41  joint                   0.0007    74.200    75.200        1       41      0.07
 42  joint                   0.0007    74.200    75.200        1       42      0.09
 43  joint                   0.0007    74.200    75.200        1       43      0.09
 44  joint                   0.0007    74.200    75.200        1       44      0.09
 45  joint                   0.0007    74.200    75.200        1       45      0.09
 46  joint                   0.0007    74.200    75.200        1       46      0.09
 47  joint                   0.0007    74.200    75.200        1       47      0.10
 48  joint                   0.0007    74.200    75.200        1       48      0.09
 49  joint                   0.0007    74.200    75.200        1       49      0.09
 50  joint                   0.0007    74.200    75.200        1       50      0.07
 51  joint                   0.0006    74.200    75.200        1       51      0.09
 52  joint                   0.0006    74.200    75.200        1       52      0.09
 53  joint                   0.0006    74.200    75.200        1       53      0.09
 54  joint                   0.0006    74.400    75.200        1       54      0.09
 55  joint                   0.0006    74.400    75.200        1       55      0.09
 56  joint                   0.0006    74.400    75.200        1       56      0.09
 57  joint                   0.0006    74.400    75.200        1       57      0.08
 58  joint                   0.0006    74.400    75.200        1       58      0.09
 59  joint                   0.0006    74.400    75.200        1       59      0.09
 60  joint                   0.0006    74.400    75.200        1       60      0.09
 61  joint                   0.0006    74.400    75.200        1       61      0.09
 62  joint                   0.0006    74.400    75.200        1       62      0.09
```

</details>

<details>
<summary>cora | reference | shared_dynamic_c — 전체 에포크</summary>

```text
[cora | reference | shared_dynamic_c]
status=passed | last_epoch=61 | selected_epoch=11
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.1164    60.600    60.600        1        1      0.66
  2  joint                   0.8073    67.400    67.400        1        2      0.33
  3  joint                   0.3080    71.400    71.400        1        3      0.32
  4  joint                   0.0958    73.800    73.800        1        4      0.32
  5  joint                   0.0379    74.000    74.000        1        5      0.43
  6  joint                   0.0188    74.800    74.800        1        6      0.33
  7  joint                   0.0105    74.600    74.800        1        7      0.32
  8  joint                   0.0070    74.800    74.800        1        8      0.32
  9  joint                   0.0053    74.800    74.800        1        9      0.32
 10  joint                   0.0040    75.000    75.000        1       10      0.33
 11  joint                   0.0033    75.600    75.600        1       11      0.42 *
 12  joint                   0.0027    75.000    75.600        1       12      0.32
 13  joint                   0.0024    75.000    75.600        1       13      0.33
 14  joint                   0.0021    74.800    75.600        1       14      0.30
 15  joint                   0.0018    74.200    75.600        1       15      0.33
 16  joint                   0.0018    73.800    75.600        1       16      0.33
 17  joint                   0.0016    73.600    75.600        1       17      0.33
 18  joint                   0.0014    73.600    75.600        1       18      0.33
 19  joint                   0.0014    73.600    75.600        1       19      0.33
 20  joint                   0.0013    73.400    75.600        1       20      0.32
 21  joint                   0.0012    73.200    75.600        1       21      0.33
 22  joint                   0.0012    73.400    75.600        1       22      0.42
 23  joint                   0.0011    73.600    75.600        1       23      0.33
 24  joint                   0.0011    73.200    75.600        1       24      0.32
 25  joint                   0.0010    73.200    75.600        1       25      0.33
 26  joint                   0.0010    73.000    75.600        1       26      0.33
 27  joint                   0.0009    73.200    75.600        1       27      0.33
 28  joint                   0.0009    73.000    75.600        1       28      0.32
 29  joint                   0.0009    73.200    75.600        1       29      0.33
 30  joint                   0.0009    73.200    75.600        1       30      0.32
 31  joint                   0.0008    73.400    75.600        1       31      0.32
 32  joint                   0.0008    73.200    75.600        1       32      0.33
 33  joint                   0.0008    73.200    75.600        1       33      0.43
 34  joint                   0.0008    73.200    75.600        1       34      0.33
 35  joint                   0.0008    73.200    75.600        1       35      0.32
 36  joint                   0.0008    73.000    75.600        1       36      0.33
 37  joint                   0.0008    73.200    75.600        1       37      0.32
 38  joint                   0.0007    73.200    75.600        1       38      0.32
 39  joint                   0.0007    73.400    75.600        1       39      0.33
 40  joint                   0.0007    73.400    75.600        1       40      0.33
 41  joint                   0.0007    73.800    75.600        1       41      0.30
 42  joint                   0.0007    73.600    75.600        1       42      0.33
 43  joint                   0.0007    73.600    75.600        1       43      0.32
 44  joint                   0.0007    73.800    75.600        1       44      0.43
 45  joint                   0.0007    73.800    75.600        1       45      0.30
 46  joint                   0.0007    73.800    75.600        1       46      0.33
 47  joint                   0.0006    73.600    75.600        1       47      0.33
 48  joint                   0.0007    73.800    75.600        1       48      0.30
 49  joint                   0.0006    73.600    75.600        1       49      0.33
 50  joint                   0.0006    73.600    75.600        1       50      0.33
 51  joint                   0.0006    73.600    75.600        1       51      0.31
 52  joint                   0.0006    73.800    75.600        1       52      0.30
 53  joint                   0.0006    73.800    75.600        1       53      0.32
 54  joint                   0.0006    73.800    75.600        1       54      0.43
 55  joint                   0.0006    73.800    75.600        1       55      0.33
 56  joint                   0.0006    73.600    75.600        1       56      0.33
 57  joint                   0.0006    73.600    75.600        1       57      0.33
 58  joint                   0.0006    73.600    75.600        1       58      0.32
 59  joint                   0.0006    73.600    75.600        1       59      0.33
 60  joint                   0.0006    73.600    75.600        1       60      0.33
 61  joint                   0.0006    73.600    75.600        1       61      0.33
```

</details>

<details>
<summary>cora | large | fixed_c — 전체 에포크</summary>

```text
[cora | large | fixed_c]
status=passed | last_epoch=56 | selected_epoch=6
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.0779    60.200    60.200        1        1      0.29
  2  joint                   0.5322    57.200    60.200        1        2      0.21
  3  joint                   0.3154    72.200    72.200        1        3      0.09
  4  joint                   0.0489    73.400    73.400        1        4      0.09
  5  joint                   0.0336    73.800    73.800        1        5      0.10
  6  joint                   0.0139    75.200    75.200        1        6      0.10 *
  7  joint                   0.0136    74.400    75.200        1        7      0.10
  8  joint                   0.0063    74.800    75.200        1        8      0.11
  9  joint                   0.0033    73.000    75.200        1        9      0.10
 10  joint                   0.0026    72.000    75.200        1       10      0.08
 11  joint                   0.0024    71.000    75.200        1       11      0.11
 12  joint                   0.0023    70.400    75.200        1       12      0.09
 13  joint                   0.0023    70.000    75.200        1       13      0.10
 14  joint                   0.0023    69.800    75.200        1       14      0.12
 15  joint                   0.0018    69.400    75.200        1       15      0.10
 16  joint                   0.0017    69.800    75.200        1       16      0.11
 17  joint                   0.0013    70.800    75.200        1       17      0.11
 18  joint                   0.0011    71.000    75.200        1       18      0.10
 19  joint                   0.0009    71.600    75.200        1       19      0.10
 20  joint                   0.0008    71.800    75.200        1       20      0.12
 21  joint                   0.0007    72.200    75.200        1       21      0.13
 22  joint                   0.0007    72.200    75.200        1       22      0.12
 23  joint                   0.0006    72.600    75.200        1       23      0.10
 24  joint                   0.0005    72.400    75.200        1       24      0.10
 25  joint                   0.0005    72.800    75.200        1       25      0.13
 26  joint                   0.0005    72.800    75.200        1       26      0.09
 27  joint                   0.0004    72.800    75.200        1       27      0.12
 28  joint                   0.0004    73.200    75.200        1       28      0.12
 29  joint                   0.0004    73.200    75.200        1       29      0.13
 30  joint                   0.0004    73.400    75.200        1       30      0.13
 31  joint                   0.0004    73.200    75.200        1       31      0.13
 32  joint                   0.0004    73.200    75.200        1       32      0.13
 33  joint                   0.0004    74.000    75.200        1       33      0.13
 34  joint                   0.0003    74.200    75.200        1       34      0.13
 35  joint                   0.0003    74.200    75.200        1       35      0.12
 36  joint                   0.0003    74.200    75.200        1       36      0.13
 37  joint                   0.0003    74.400    75.200        1       37      0.13
 38  joint                   0.0003    74.400    75.200        1       38      0.13
 39  joint                   0.0003    74.400    75.200        1       39      0.13
 40  joint                   0.0003    74.600    75.200        1       40      0.13
 41  joint                   0.0003    74.600    75.200        1       41      0.13
 42  joint                   0.0003    74.600    75.200        1       42      0.09
 43  joint                   0.0003    74.600    75.200        1       43      0.13
 44  joint                   0.0003    74.600    75.200        1       44      0.13
 45  joint                   0.0003    74.600    75.200        1       45      0.11
 46  joint                   0.0003    74.600    75.200        1       46      0.13
 47  joint                   0.0002    74.800    75.200        1       47      0.13
 48  joint                   0.0002    74.800    75.200        1       48      0.13
 49  joint                   0.0002    74.800    75.200        1       49      0.09
 50  joint                   0.0002    75.000    75.200        1       50      0.13
 51  joint                   0.0002    75.000    75.200        1       51      0.11
 52  joint                   0.0002    75.000    75.200        1       52      0.13
 53  joint                   0.0002    75.000    75.200        1       53      0.13
 54  joint                   0.0002    75.000    75.200        1       54      0.09
 55  joint                   0.0002    75.000    75.200        1       55      0.10
 56  joint                   0.0002    75.000    75.200        1       56      0.12
```

</details>

<details>
<summary>cora | large | shared_dynamic_c — 전체 에포크</summary>

```text
[cora | large | shared_dynamic_c]
status=passed | last_epoch=161 | selected_epoch=111
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.0826    59.400    59.400        1        1      0.78
  2  joint                   0.5251    57.800    59.400        1        2      0.48
  3  joint                   0.2695    71.800    71.800        1        3      0.47
  4  joint                   0.0411    73.000    73.000        1        4      0.47
  5  joint                   0.0254    74.200    74.200        1        5      0.48
  6  joint                   0.0103    74.000    74.200        1        6      0.47
  7  joint                   0.0090    73.400    74.200        1        7      0.56
  8  joint                   0.0047    73.200    74.200        1        8      0.47
  9  joint                   0.0030    72.400    74.200        1        9      0.47
 10  joint                   0.0025    72.200    74.200        1       10      0.47
 11  joint                   0.0021    71.800    74.200        1       11      0.46
 12  joint                   0.0017    70.800    74.200        1       12      0.57
 13  joint                   0.0016    71.000    74.200        1       13      0.45
 14  joint                   0.0015    70.600    74.200        1       14      0.46
 15  joint                   0.0014    70.600    74.200        1       15      0.46
 16  joint                   0.0013    70.400    74.200        1       16      0.46
 17  joint                   0.0012    70.400    74.200        1       17      0.46
 18  joint                   0.0010    71.000    74.200        1       18      0.56
 19  joint                   0.0009    71.600    74.200        1       19      0.47
 20  joint                   0.0007    72.400    74.200        1       20      0.47
 21  joint                   0.0007    72.600    74.200        1       21      0.46
 22  joint                   0.0006    72.800    74.200        1       22      0.46
 23  joint                   0.0005    73.000    74.200        1       23      0.47
 24  joint                   0.0005    73.200    74.200        1       24      0.57
 25  joint                   0.0005    73.200    74.200        1       25      0.47
 26  joint                   0.0004    73.000    74.200        1       26      0.46
 27  joint                   0.0004    73.200    74.200        1       27      0.47
 28  joint                   0.0004    73.400    74.200        1       28      0.47
 29  joint                   0.0004    73.600    74.200        1       29      0.58
 30  joint                   0.0003    73.600    74.200        1       30      0.46
 31  joint                   0.0003    73.600    74.200        1       31      0.47
 32  joint                   0.0003    74.000    74.200        1       32      0.46
 33  joint                   0.0003    74.200    74.200        1       33      0.47
 34  joint                   0.0003    74.600    74.600        1       34      0.46
 35  joint                   0.0003    74.800    74.800        1       35      0.56
 36  joint                   0.0003    74.600    74.800        1       36      0.46
 37  joint                   0.0003    75.000    75.000        1       37      0.46
 38  joint                   0.0003    75.000    75.000        1       38      0.45
 39  joint                   0.0003    75.000    75.000        1       39      0.47
 40  joint                   0.0003    75.000    75.000        1       40      0.47
 41  joint                   0.0003    75.200    75.200        1       41      0.56
 42  joint                   0.0002    75.200    75.200        1       42      0.46
 43  joint                   0.0002    75.200    75.200        1       43      0.46
 44  joint                   0.0002    75.200    75.200        1       44      0.46
 45  joint                   0.0002    75.200    75.200        1       45      0.47
 46  joint                   0.0002    75.200    75.200        1       46      0.55
 47  joint                   0.0002    75.200    75.200        1       47      0.46
 48  joint                   0.0002    75.200    75.200        1       48      0.47
 49  joint                   0.0002    75.200    75.200        1       49      0.47
 50  joint                   0.0002    75.200    75.200        1       50      0.46
 51  joint                   0.0002    75.200    75.200        1       51      0.46
 52  joint                   0.0002    75.200    75.200        1       52      0.57
 53  joint                   0.0002    75.000    75.200        1       53      0.46
 54  joint                   0.0002    75.000    75.200        1       54      0.46
 55  joint                   0.0002    75.000    75.200        1       55      0.47
 56  joint                   0.0002    75.000    75.200        1       56      0.46
 57  joint                   0.0002    75.200    75.200        1       57      0.46
 58  joint                   0.0002    75.200    75.200        1       58      0.55
 59  joint                   0.0002    75.200    75.200        1       59      0.45
 60  joint                   0.0002    75.200    75.200        1       60      0.47
 61  joint                   0.0002    75.400    75.400        1       61      0.47
 62  joint                   0.0002    75.200    75.400        1       62      0.47
 63  joint                   0.0002    75.400    75.400        1       63      0.46
 64  joint                   0.0002    75.400    75.400        1       64      0.56
 65  joint                   0.0002    75.400    75.400        1       65      0.46
 66  joint                   0.0002    75.600    75.600        1       66      0.46
 67  joint                   0.0002    75.600    75.600        1       67      0.46
 68  joint                   0.0002    75.600    75.600        1       68      0.46
 69  joint                   0.0002    75.600    75.600        1       69      0.56
 70  joint                   0.0002    75.400    75.600        1       70      0.45
 71  joint                   0.0002    75.600    75.600        1       71      0.47
 72  joint                   0.0002    75.600    75.600        1       72      0.46
 73  joint                   0.0002    75.600    75.600        1       73      0.47
 74  joint                   0.0002    75.600    75.600        1       74      0.48
 75  joint                   0.0002    75.600    75.600        1       75      0.55
 76  joint                   0.0002    75.400    75.600        1       76      0.46
 77  joint                   0.0002    75.600    75.600        1       77      0.46
 78  joint                   0.0002    75.400    75.600        1       78      0.46
 79  joint                   0.0002    75.600    75.600        1       79      0.46
 80  joint                   0.0002    75.600    75.600        1       80      0.54
 81  joint                   0.0002    75.400    75.600        1       81      0.47
 82  joint                   0.0002    75.400    75.600        1       82      0.47
 83  joint                   0.0002    75.600    75.600        1       83      0.47
 84  joint                   0.0002    75.600    75.600        1       84      0.46
 85  joint                   0.0002    75.400    75.600        1       85      0.47
 86  joint                   0.0002    75.400    75.600        1       86      0.57
 87  joint                   0.0002    75.600    75.600        1       87      0.47
 88  joint                   0.0002    75.400    75.600        1       88      0.47
 89  joint                   0.0002    75.600    75.600        1       89      0.46
 90  joint                   0.0002    75.600    75.600        1       90      0.47
 91  joint                   0.0002    75.600    75.600        1       91      0.47
 92  joint                   0.0001    75.600    75.600        1       92      0.57
 93  joint                   0.0001    75.400    75.600        1       93      0.46
 94  joint                   0.0001    75.400    75.600        1       94      0.47
 95  joint                   0.0002    75.400    75.600        1       95      0.47
 96  joint                   0.0001    75.600    75.600        1       96      0.47
 97  joint                   0.0001    75.600    75.600        1       97      0.46
 98  joint                   0.0001    75.400    75.600        1       98      0.55
 99  joint                   0.0001    75.600    75.600        1       99      0.46
100  joint                   0.0001    75.800    75.800        1      100      0.46
101  joint                   0.0001    75.800    75.800        1      101      0.46
102  joint                   0.0001    75.800    75.800        1      102      0.47
103  joint                   0.0001    75.800    75.800        1      103      0.47
104  joint                   0.0001    75.800    75.800        1      104      0.55
105  joint                   0.0001    75.800    75.800        1      105      0.47
106  joint                   0.0001    75.800    75.800        1      106      0.46
107  joint                   0.0001    75.800    75.800        1      107      0.47
108  joint                   0.0001    75.800    75.800        1      108      0.47
109  joint                   0.0001    75.800    75.800        1      109      0.56
110  joint                   0.0001    75.800    75.800        1      110      0.45
111  joint                   0.0001    76.000    76.000        1      111      0.46 *
112  joint                   0.0001    75.800    76.000        1      112      0.47
113  joint                   0.0001    75.800    76.000        1      113      0.46
114  joint                   0.0001    76.000    76.000        1      114      0.47
115  joint                   0.0001    76.000    76.000        1      115      0.56
116  joint                   0.0001    76.000    76.000        1      116      0.47
117  joint                   0.0001    75.800    76.000        1      117      0.46
118  joint                   0.0001    75.800    76.000        1      118      0.47
119  joint                   0.0001    76.000    76.000        1      119      0.46
120  joint                   0.0001    76.000    76.000        1      120      0.44
121  joint                   0.0001    76.000    76.000        1      121      0.56
122  joint                   0.0001    76.000    76.000        1      122      0.44
123  joint                   0.0001    76.000    76.000        1      123      0.44
124  joint                   0.0001    76.000    76.000        1      124      0.44
125  joint                   0.0001    75.800    76.000        1      125      0.44
126  joint                   0.0001    76.000    76.000        1      126      0.44
127  joint                   0.0001    75.800    76.000        1      127      0.53
128  joint                   0.0001    75.600    76.000        1      128      0.44
129  joint                   0.0001    75.800    76.000        1      129      0.44
130  joint                   0.0001    75.800    76.000        1      130      0.44
131  joint                   0.0001    75.800    76.000        1      131      0.44
132  joint                   0.0001    75.800    76.000        1      132      0.54
133  joint                   0.0001    75.800    76.000        1      133      0.44
134  joint                   0.0001    75.800    76.000        1      134      0.44
135  joint                   0.0001    75.800    76.000        1      135      0.44
136  joint                   0.0001    75.600    76.000        1      136      0.44
137  joint                   0.0001    75.600    76.000        1      137      0.44
138  joint                   0.0001    75.600    76.000        1      138      0.54
139  joint                   0.0001    75.600    76.000        1      139      0.44
140  joint                   0.0001    75.600    76.000        1      140      0.44
141  joint                   0.0001    75.600    76.000        1      141      0.44
142  joint                   0.0001    75.600    76.000        1      142      0.44
143  joint                   0.0001    75.600    76.000        1      143      0.44
144  joint                   0.0001    75.800    76.000        1      144      0.54
145  joint                   0.0001    75.600    76.000        1      145      0.44
146  joint                   0.0001    75.600    76.000        1      146      0.44
147  joint                   0.0001    75.800    76.000        1      147      0.44
148  joint                   0.0001    75.800    76.000        1      148      0.44
149  joint                   0.0001    75.600    76.000        1      149      0.44
150  joint                   0.0001    75.600    76.000        1      150      0.54
151  joint                   0.0001    75.600    76.000        1      151      0.44
152  joint                   0.0001    75.600    76.000        1      152      0.44
153  joint                   0.0001    75.600    76.000        1      153      0.44
154  joint                   0.0001    75.600    76.000        1      154      0.45
155  joint                   0.0001    75.600    76.000        1      155      0.44
156  joint                   0.0001    75.600    76.000        1      156      0.44
157  joint                   0.0001    75.600    76.000        1      157      0.44
158  joint                   0.0001    75.600    76.000        1      158      0.44
159  joint                   0.0001    75.600    76.000        1      159      0.45
160  joint                   0.0001    75.600    76.000        1      160      0.44
161  joint                   0.0001    75.600    76.000        1      161      0.45
```

</details>

<details>
<summary>citeseer | reference | fixed_c — 전체 에포크</summary>

```text
[citeseer | reference | fixed_c]
status=passed | last_epoch=75 | selected_epoch=25
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.8928    52.600    52.600        1        1      0.28
  2  joint                   0.6751    57.400    57.400        1        2      0.08
  3  joint                   0.2681    64.400    64.400        1        3      0.08
  4  joint                   0.0884    61.800    64.400        1        4      0.08
  5  joint                   0.0589    62.000    64.400        1        5      0.08
  6  joint                   0.0232    63.000    64.400        1        6      0.08
  7  joint                   0.0126    63.600    64.400        1        7      0.10
  8  joint                   0.0085    63.800    64.400        1        8      0.09
  9  joint                   0.0077    65.000    65.000        1        9      0.09
 10  joint                   0.0058    64.400    65.000        1       10      0.06
 11  joint                   0.0038    64.400    65.000        1       11      0.06
 12  joint                   0.0031    64.600    65.000        1       12      0.09
 13  joint                   0.0025    64.600    65.000        1       13      0.10
 14  joint                   0.0023    64.800    65.000        1       14      0.09
 15  joint                   0.0019    64.600    65.000        1       15      0.09
 16  joint                   0.0018    65.000    65.000        1       16      0.10
 17  joint                   0.0016    64.800    65.000        1       17      0.09
 18  joint                   0.0015    65.000    65.000        1       18      0.10
 19  joint                   0.0014    65.200    65.200        1       19      0.06
 20  joint                   0.0013    65.000    65.200        1       20      0.10
 21  joint                   0.0012    65.000    65.200        1       21      0.10
 22  joint                   0.0011    65.000    65.200        1       22      0.10
 23  joint                   0.0011    65.200    65.200        1       23      0.10
 24  joint                   0.0010    65.600    65.600        1       24      0.07
 25  joint                   0.0010    65.800    65.800        1       25      0.10 *
 26  joint                   0.0010    65.400    65.800        1       26      0.10
 27  joint                   0.0009    65.400    65.800        1       27      0.09
 28  joint                   0.0009    65.400    65.800        1       28      0.11
 29  joint                   0.0009    65.400    65.800        1       29      0.11
 30  joint                   0.0008    65.400    65.800        1       30      0.09
 31  joint                   0.0008    65.400    65.800        1       31      0.10
 32  joint                   0.0008    65.600    65.800        1       32      0.10
 33  joint                   0.0008    65.800    65.800        1       33      0.08
 34  joint                   0.0008    65.600    65.800        1       34      0.09
 35  joint                   0.0007    65.600    65.800        1       35      0.08
 36  joint                   0.0007    65.600    65.800        1       36      0.10
 37  joint                   0.0007    65.600    65.800        1       37      0.09
 38  joint                   0.0007    65.600    65.800        1       38      0.10
 39  joint                   0.0007    65.600    65.800        1       39      0.10
 40  joint                   0.0007    65.600    65.800        1       40      0.10
 41  joint                   0.0007    65.600    65.800        1       41      0.10
 42  joint                   0.0006    65.400    65.800        1       42      0.09
 43  joint                   0.0006    65.400    65.800        1       43      0.09
 44  joint                   0.0006    65.400    65.800        1       44      0.09
 45  joint                   0.0006    65.800    65.800        1       45      0.10
 46  joint                   0.0006    65.800    65.800        1       46      0.09
 47  joint                   0.0006    65.800    65.800        1       47      0.10
 48  joint                   0.0006    65.800    65.800        1       48      0.10
 49  joint                   0.0006    65.800    65.800        1       49      0.09
 50  joint                   0.0006    65.800    65.800        1       50      0.11
 51  joint                   0.0006    65.800    65.800        1       51      0.09
 52  joint                   0.0006    65.800    65.800        1       52      0.06
 53  joint                   0.0005    65.800    65.800        1       53      0.09
 54  joint                   0.0005    65.800    65.800        1       54      0.05
 55  joint                   0.0005    65.800    65.800        1       55      0.10
 56  joint                   0.0005    65.400    65.800        1       56      0.10
 57  joint                   0.0005    65.600    65.800        1       57      0.09
 58  joint                   0.0005    65.400    65.800        1       58      0.10
 59  joint                   0.0005    65.400    65.800        1       59      0.09
 60  joint                   0.0005    65.600    65.800        1       60      0.09
 61  joint                   0.0005    65.600    65.800        1       61      0.08
 62  joint                   0.0005    65.600    65.800        1       62      0.05
 63  joint                   0.0005    65.600    65.800        1       63      0.09
 64  joint                   0.0005    65.600    65.800        1       64      0.09
 65  joint                   0.0005    65.600    65.800        1       65      0.09
 66  joint                   0.0005    65.600    65.800        1       66      0.09
 67  joint                   0.0005    65.600    65.800        1       67      0.09
 68  joint                   0.0005    65.600    65.800        1       68      0.05
 69  joint                   0.0005    65.600    65.800        1       69      0.09
 70  joint                   0.0005    65.600    65.800        1       70      0.09
 71  joint                   0.0005    65.600    65.800        1       71      0.09
 72  joint                   0.0005    65.600    65.800        1       72      0.08
 73  joint                   0.0004    65.600    65.800        1       73      0.09
 74  joint                   0.0004    65.600    65.800        1       74      0.09
 75  joint                   0.0004    65.600    65.800        1       75      0.09
```

</details>

<details>
<summary>citeseer | reference | shared_dynamic_c — 전체 에포크</summary>

```text
[citeseer | reference | shared_dynamic_c]
status=passed | last_epoch=118 | selected_epoch=68
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.8983    54.000    54.000        1        1      0.67
  2  joint                   0.6580    56.400    56.400        1        2      0.33
  3  joint                   0.2598    62.800    62.800        1        3      0.33
  4  joint                   0.0776    62.000    62.800        1        4      0.33
  5  joint                   0.0497    61.400    62.800        1        5      0.33
  6  joint                   0.0179    62.000    62.800        1        6      0.33
  7  joint                   0.0100    62.400    62.800        1        7      0.34
  8  joint                   0.0064    63.200    63.200        1        8      0.33
  9  joint                   0.0049    63.800    63.800        1        9      0.33
 10  joint                   0.0034    63.600    63.800        1       10      0.33
 11  joint                   0.0029    63.800    63.800        1       11      0.42
 12  joint                   0.0026    64.200    64.200        1       12      0.32
 13  joint                   0.0023    64.600    64.600        1       13      0.32
 14  joint                   0.0023    64.800    64.800        1       14      0.32
 15  joint                   0.0018    65.000    65.000        1       15      0.30
 16  joint                   0.0017    65.000    65.000        1       16      0.29
 17  joint                   0.0016    65.000    65.000        1       17      0.32
 18  joint                   0.0014    65.200    65.200        1       18      0.32
 19  joint                   0.0014    65.200    65.200        1       19      0.32
 20  joint                   0.0013    64.800    65.200        1       20      0.32
 21  joint                   0.0012    65.000    65.200        1       21      0.41
 22  joint                   0.0012    65.000    65.200        1       22      0.32
 23  joint                   0.0011    65.200    65.200        1       23      0.29
 24  joint                   0.0010    65.000    65.200        1       24      0.32
 25  joint                   0.0010    65.000    65.200        1       25      0.32
 26  joint                   0.0010    65.000    65.200        1       26      0.31
 27  joint                   0.0009    65.200    65.200        1       27      0.32
 28  joint                   0.0009    65.000    65.200        1       28      0.32
 29  joint                   0.0009    65.000    65.200        1       29      0.32
 30  joint                   0.0008    64.800    65.200        1       30      0.31
 31  joint                   0.0008    65.200    65.200        1       31      0.42
 32  joint                   0.0008    65.000    65.200        1       32      0.32
 33  joint                   0.0008    65.000    65.200        1       33      0.32
 34  joint                   0.0007    65.000    65.200        1       34      0.32
 35  joint                   0.0007    65.000    65.200        1       35      0.33
 36  joint                   0.0007    65.000    65.200        1       36      0.32
 37  joint                   0.0007    65.000    65.200        1       37      0.33
 38  joint                   0.0007    65.200    65.200        1       38      0.32
 39  joint                   0.0006    65.200    65.200        1       39      0.32
 40  joint                   0.0006    65.200    65.200        1       40      0.31
 41  joint                   0.0006    65.000    65.200        1       41      0.43
 42  joint                   0.0006    65.000    65.200        1       42      0.32
 43  joint                   0.0006    65.000    65.200        1       43      0.32
 44  joint                   0.0006    64.800    65.200        1       44      0.32
 45  joint                   0.0006    64.800    65.200        1       45      0.32
 46  joint                   0.0006    64.600    65.200        1       46      0.32
 47  joint                   0.0006    64.600    65.200        1       47      0.32
 48  joint                   0.0006    64.600    65.200        1       48      0.31
 49  joint                   0.0006    64.600    65.200        1       49      0.32
 50  joint                   0.0005    64.600    65.200        1       50      0.32
 51  joint                   0.0005    64.600    65.200        1       51      0.42
 52  joint                   0.0005    64.800    65.200        1       52      0.32
 53  joint                   0.0005    65.000    65.200        1       53      0.32
 54  joint                   0.0005    65.000    65.200        1       54      0.31
 55  joint                   0.0005    65.000    65.200        1       55      0.32
 56  joint                   0.0005    65.000    65.200        1       56      0.32
 57  joint                   0.0005    65.000    65.200        1       57      0.32
 58  joint                   0.0005    65.000    65.200        1       58      0.32
 59  joint                   0.0005    65.000    65.200        1       59      0.31
 60  joint                   0.0005    65.000    65.200        1       60      0.32
 61  joint                   0.0005    65.200    65.200        1       61      0.41
 62  joint                   0.0005    64.800    65.200        1       62      0.32
 63  joint                   0.0005    65.000    65.200        1       63      0.31
 64  joint                   0.0005    65.000    65.200        1       64      0.31
 65  joint                   0.0005    65.200    65.200        1       65      0.31
 66  joint                   0.0005    65.000    65.200        1       66      0.33
 67  joint                   0.0005    65.200    65.200        1       67      0.33
 68  joint                   0.0005    65.400    65.400        1       68      0.33 *
 69  joint                   0.0005    65.400    65.400        1       69      0.34
 70  joint                   0.0004    65.200    65.400        1       70      0.33
 71  joint                   0.0004    65.200    65.400        1       71      0.33
 72  joint                   0.0004    65.200    65.400        1       72      0.43
 73  joint                   0.0004    65.200    65.400        1       73      0.32
 74  joint                   0.0004    65.200    65.400        1       74      0.33
 75  joint                   0.0004    65.200    65.400        1       75      0.33
 76  joint                   0.0004    65.200    65.400        1       76      0.33
 77  joint                   0.0004    65.200    65.400        1       77      0.33
 78  joint                   0.0004    65.000    65.400        1       78      0.33
 79  joint                   0.0004    65.000    65.400        1       79      0.33
 80  joint                   0.0004    65.000    65.400        1       80      0.33
 81  joint                   0.0004    65.200    65.400        1       81      0.33
 82  joint                   0.0004    65.200    65.400        1       82      0.40
 83  joint                   0.0004    65.200    65.400        1       83      0.33
 84  joint                   0.0004    65.200    65.400        1       84      0.33
 85  joint                   0.0004    65.200    65.400        1       85      0.30
 86  joint                   0.0004    65.200    65.400        1       86      0.33
 87  joint                   0.0004    65.200    65.400        1       87      0.33
 88  joint                   0.0004    65.200    65.400        1       88      0.34
 89  joint                   0.0004    65.200    65.400        1       89      0.33
 90  joint                   0.0004    65.200    65.400        1       90      0.33
 91  joint                   0.0004    65.200    65.400        1       91      0.33
 92  joint                   0.0004    65.200    65.400        1       92      0.33
 93  joint                   0.0004    65.200    65.400        1       93      0.42
 94  joint                   0.0004    65.200    65.400        1       94      0.33
 95  joint                   0.0004    65.200    65.400        1       95      0.34
 96  joint                   0.0004    65.200    65.400        1       96      0.33
 97  joint                   0.0004    65.200    65.400        1       97      0.33
 98  joint                   0.0004    65.200    65.400        1       98      0.33
 99  joint                   0.0004    65.200    65.400        1       99      0.33
100  joint                   0.0004    65.200    65.400        1      100      0.32
101  joint                   0.0004    65.200    65.400        1      101      0.33
102  joint                   0.0004    65.200    65.400        1      102      0.33
103  joint                   0.0004    65.200    65.400        1      103      0.33
104  joint                   0.0004    65.200    65.400        1      104      0.43
105  joint                   0.0003    65.200    65.400        1      105      0.33
106  joint                   0.0004    65.200    65.400        1      106      0.33
107  joint                   0.0003    65.200    65.400        1      107      0.33
108  joint                   0.0003    65.200    65.400        1      108      0.33
109  joint                   0.0003    65.200    65.400        1      109      0.33
110  joint                   0.0003    65.200    65.400        1      110      0.33
111  joint                   0.0003    65.200    65.400        1      111      0.33
112  joint                   0.0003    65.200    65.400        1      112      0.33
113  joint                   0.0003    65.200    65.400        1      113      0.33
114  joint                   0.0003    65.200    65.400        1      114      0.33
115  joint                   0.0003    65.000    65.400        1      115      0.42
116  joint                   0.0003    65.000    65.400        1      116      0.33
117  joint                   0.0003    65.000    65.400        1      117      0.33
118  joint                   0.0003    65.000    65.400        1      118      0.33
```

</details>

<details>
<summary>citeseer | large | fixed_c — 전체 에포크</summary>

```text
[citeseer | large | fixed_c]
status=passed | last_epoch=65 | selected_epoch=15
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.9247    64.000    64.000        1        1      0.29
  2  joint                   0.5201    42.600    64.000        1        2      0.12
  3  joint                   0.5196    60.000    64.000        1        3      0.11
  4  joint                   0.1101    64.800    64.800        1        4      0.09
  5  joint                   0.0772    65.200    65.200        1        5      0.09
  6  joint                   0.0272    64.200    65.200        1        6      0.12
  7  joint                   0.0240    65.400    65.400        1        7      0.10
  8  joint                   0.0212    65.000    65.400        1        8      0.11
  9  joint                   0.0101    66.400    66.400        1        9      0.12
 10  joint                   0.0045    67.200    67.200        1       10      0.12
 11  joint                   0.0054    67.400    67.400        1       11      0.12
 12  joint                   0.0053    67.600    67.600        1       12      0.13
 13  joint                   0.0046    67.800    67.800        1       13      0.13
 14  joint                   0.0020    67.800    67.800        1       14      0.11
 15  joint                   0.0013    68.000    68.000        1       15      0.11 *
 16  joint                   0.0011    67.000    68.000        1       16      0.11
 17  joint                   0.0010    66.800    68.000        1       17      0.12
 18  joint                   0.0009    66.600    68.000        1       18      0.12
 19  joint                   0.0008    66.600    68.000        1       19      0.12
 20  joint                   0.0008    66.800    68.000        1       20      0.10
 21  joint                   0.0007    66.800    68.000        1       21      0.12
 22  joint                   0.0006    66.400    68.000        1       22      0.13
 23  joint                   0.0006    66.400    68.000        1       23      0.13
 24  joint                   0.0005    66.200    68.000        1       24      0.11
 25  joint                   0.0005    66.400    68.000        1       25      0.13
 26  joint                   0.0005    66.600    68.000        1       26      0.12
 27  joint                   0.0004    66.200    68.000        1       27      0.13
 28  joint                   0.0004    66.000    68.000        1       28      0.13
 29  joint                   0.0004    65.400    68.000        1       29      0.13
 30  joint                   0.0004    65.200    68.000        1       30      0.12
 31  joint                   0.0003    64.600    68.000        1       31      0.08
 32  joint                   0.0003    64.600    68.000        1       32      0.07
 33  joint                   0.0003    64.600    68.000        1       33      0.11
 34  joint                   0.0003    64.400    68.000        1       34      0.11
 35  joint                   0.0003    64.400    68.000        1       35      0.11
 36  joint                   0.0003    64.600    68.000        1       36      0.11
 37  joint                   0.0003    64.400    68.000        1       37      0.13
 38  joint                   0.0003    64.200    68.000        1       38      0.11
 39  joint                   0.0003    64.000    68.000        1       39      0.13
 40  joint                   0.0002    64.000    68.000        1       40      0.13
 41  joint                   0.0002    64.000    68.000        1       41      0.12
 42  joint                   0.0002    64.200    68.000        1       42      0.11
 43  joint                   0.0002    64.200    68.000        1       43      0.12
 44  joint                   0.0002    64.400    68.000        1       44      0.13
 45  joint                   0.0002    64.400    68.000        1       45      0.10
 46  joint                   0.0002    64.400    68.000        1       46      0.13
 47  joint                   0.0002    64.400    68.000        1       47      0.13
 48  joint                   0.0002    64.400    68.000        1       48      0.11
 49  joint                   0.0002    64.400    68.000        1       49      0.12
 50  joint                   0.0002    64.200    68.000        1       50      0.13
 51  joint                   0.0002    64.400    68.000        1       51      0.11
 52  joint                   0.0002    64.400    68.000        1       52      0.11
 53  joint                   0.0002    64.200    68.000        1       53      0.11
 54  joint                   0.0002    64.200    68.000        1       54      0.13
 55  joint                   0.0002    64.200    68.000        1       55      0.11
 56  joint                   0.0002    64.200    68.000        1       56      0.11
 57  joint                   0.0002    64.200    68.000        1       57      0.09
 58  joint                   0.0002    64.200    68.000        1       58      0.13
 59  joint                   0.0002    64.200    68.000        1       59      0.13
 60  joint                   0.0002    64.200    68.000        1       60      0.11
 61  joint                   0.0002    64.200    68.000        1       61      0.10
 62  joint                   0.0002    64.000    68.000        1       62      0.12
 63  joint                   0.0002    64.000    68.000        1       63      0.13
 64  joint                   0.0002    64.000    68.000        1       64      0.13
 65  joint                   0.0002    64.000    68.000        1       65      0.11
```

</details>

<details>
<summary>citeseer | large | shared_dynamic_c — 전체 에포크</summary>

```text
[citeseer | large | shared_dynamic_c]
status=passed | last_epoch=54 | selected_epoch=4
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.9233    65.800    65.800        1        1      0.79
  2  joint                   0.5103    42.200    65.800        1        2      0.46
  3  joint                   0.4816    61.200    65.800        1        3      0.46
  4  joint                   0.1045    66.000    66.000        1        4      0.46 *
  5  joint                   0.0590    65.800    66.000        1        5      0.47
  6  joint                   0.0230    65.400    66.000        1        6      0.47
  7  joint                   0.0168    65.800    66.000        1        7      0.47
  8  joint                   0.0094    64.600    66.000        1        8      0.45
  9  joint                   0.0040    65.600    66.000        1        9      0.45
 10  joint                   0.0022    65.800    66.000        1       10      0.45
 11  joint                   0.0020    65.400    66.000        1       11      0.45
 12  joint                   0.0022    65.600    66.000        1       12      0.54
 13  joint                   0.0020    65.000    66.000        1       13      0.45
 14  joint                   0.0013    64.400    66.000        1       14      0.45
 15  joint                   0.0010    64.400    66.000        1       15      0.46
 16  joint                   0.0009    64.200    66.000        1       16      0.46
 17  joint                   0.0007    63.800    66.000        1       17      0.46
 18  joint                   0.0007    63.200    66.000        1       18      0.55
 19  joint                   0.0006    63.200    66.000        1       19      0.46
 20  joint                   0.0006    63.000    66.000        1       20      0.46
 21  joint                   0.0005    63.200    66.000        1       21      0.45
 22  joint                   0.0005    63.200    66.000        1       22      0.45
 23  joint                   0.0005    63.000    66.000        1       23      0.55
 24  joint                   0.0004    63.200    66.000        1       24      0.46
 25  joint                   0.0004    63.000    66.000        1       25      0.45
 26  joint                   0.0004    62.800    66.000        1       26      0.45
 27  joint                   0.0003    63.000    66.000        1       27      0.45
 28  joint                   0.0003    63.200    66.000        1       28      0.55
 29  joint                   0.0003    63.200    66.000        1       29      0.47
 30  joint                   0.0003    63.200    66.000        1       30      0.46
 31  joint                   0.0003    63.000    66.000        1       31      0.46
 32  joint                   0.0003    62.800    66.000        1       32      0.44
 33  joint                   0.0003    62.600    66.000        1       33      0.46
 34  joint                   0.0003    62.600    66.000        1       34      0.56
 35  joint                   0.0003    63.000    66.000        1       35      0.45
 36  joint                   0.0002    63.000    66.000        1       36      0.45
 37  joint                   0.0002    63.000    66.000        1       37      0.45
 38  joint                   0.0002    63.200    66.000        1       38      0.44
 39  joint                   0.0002    63.200    66.000        1       39      0.45
 40  joint                   0.0002    63.200    66.000        1       40      0.55
 41  joint                   0.0002    63.200    66.000        1       41      0.45
 42  joint                   0.0002    63.200    66.000        1       42      0.45
 43  joint                   0.0002    63.200    66.000        1       43      0.44
 44  joint                   0.0002    63.200    66.000        1       44      0.46
 45  joint                   0.0002    63.600    66.000        1       45      0.54
 46  joint                   0.0002    63.600    66.000        1       46      0.46
 47  joint                   0.0002    63.600    66.000        1       47      0.47
 48  joint                   0.0002    63.600    66.000        1       48      0.45
 49  joint                   0.0002    63.600    66.000        1       49      0.46
 50  joint                   0.0002    63.600    66.000        1       50      0.45
 51  joint                   0.0002    63.600    66.000        1       51      0.56
 52  joint                   0.0002    63.600    66.000        1       52      0.45
 53  joint                   0.0002    63.600    66.000        1       53      0.46
 54  joint                   0.0002    63.600    66.000        1       54      0.45
```

</details>

<details>
<summary>pubmed | reference | fixed_c — 전체 에포크</summary>

```text
[pubmed | reference | fixed_c]
status=passed | last_epoch=97 | selected_epoch=47
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.1493    65.200    65.200        1        1      0.75
  2  joint                   0.1960    68.400    68.400        1        2      0.12
  3  joint                   0.0390    70.000    70.000        1        3      0.12
  4  joint                   0.0122    72.600    72.600        1        4      0.12
  5  joint                   0.0032    73.000    73.000        1        5      0.12
  6  joint                   0.0016    73.400    73.400        1        6      0.12
  7  joint                   0.0008    74.000    74.000        1        7      0.12
  8  joint                   0.0006    74.400    74.400        1        8      0.12
  9  joint                   0.0004    74.200    74.400        1        9      0.12
 10  joint                   0.0003    74.200    74.400        1       10      0.12
 11  joint                   0.0003    73.800    74.400        1       11      0.12
 12  joint                   0.0002    74.000    74.400        1       12      0.12
 13  joint                   0.0002    74.400    74.400        1       13      0.12
 14  joint                   0.0002    74.400    74.400        1       14      0.12
 15  joint                   0.0002    74.200    74.400        1       15      0.12
 16  joint                   0.0002    74.200    74.400        1       16      0.12
 17  joint                   0.0001    74.400    74.400        1       17      0.12
 18  joint                   0.0001    74.400    74.400        1       18      0.12
 19  joint                   0.0001    74.400    74.400        1       19      0.12
 20  joint                   0.0001    75.000    75.000        1       20      0.12
 21  joint                   0.0001    74.600    75.000        1       21      0.11
 22  joint                   0.0001    75.200    75.200        1       22      0.12
 23  joint                   0.0001    74.800    75.200        1       23      0.12
 24  joint                   0.0001    75.000    75.200        1       24      0.12
 25  joint                   0.0001    75.000    75.200        1       25      0.12
 26  joint                   0.0001    75.000    75.200        1       26      0.12
 27  joint                   0.0001    75.200    75.200        1       27      0.12
 28  joint                   0.0001    75.200    75.200        1       28      0.12
 29  joint                   0.0001    75.200    75.200        1       29      0.12
 30  joint                   0.0001    75.200    75.200        1       30      0.12
 31  joint                   0.0001    75.000    75.200        1       31      0.12
 32  joint                   0.0001    75.200    75.200        1       32      0.12
 33  joint                   0.0001    75.200    75.200        1       33      0.12
 34  joint                   0.0001    75.200    75.200        1       34      0.12
 35  joint                   0.0001    75.200    75.200        1       35      0.12
 36  joint                   0.0001    75.000    75.200        1       36      0.12
 37  joint                   0.0001    75.000    75.200        1       37      0.12
 38  joint                   0.0001    75.200    75.200        1       38      0.12
 39  joint                   0.0001    75.400    75.400        1       39      0.12
 40  joint                   0.0001    75.200    75.400        1       40      0.12
 41  joint                   0.0001    75.200    75.400        1       41      0.12
 42  joint                   0.0001    75.200    75.400        1       42      0.12
 43  joint                   0.0001    75.200    75.400        1       43      0.12
 44  joint                   0.0001    75.200    75.400        1       44      0.12
 45  joint                   0.0001    75.000    75.400        1       45      0.12
 46  joint                   0.0001    75.200    75.400        1       46      0.12
 47  joint                   0.0001    75.600    75.600        1       47      0.12 *
 48  joint                   0.0001    75.200    75.600        1       48      0.12
 49  joint                   0.0001    75.400    75.600        1       49      0.12
 50  joint                   0.0001    75.200    75.600        1       50      0.12
 51  joint                   0.0001    75.200    75.600        1       51      0.12
 52  joint                   0.0001    75.000    75.600        1       52      0.12
 53  joint                   0.0001    75.000    75.600        1       53      0.12
 54  joint                   0.0001    75.200    75.600        1       54      0.12
 55  joint                   0.0001    75.000    75.600        1       55      0.12
 56  joint                   0.0001    75.000    75.600        1       56      0.12
 57  joint                   0.0001    75.000    75.600        1       57      0.12
 58  joint                   0.0001    75.000    75.600        1       58      0.12
 59  joint                   0.0001    75.000    75.600        1       59      0.12
 60  joint                   0.0001    75.000    75.600        1       60      0.12
 61  joint                   0.0001    75.000    75.600        1       61      0.12
 62  joint                   0.0001    74.800    75.600        1       62      0.12
 63  joint                   0.0001    74.800    75.600        1       63      0.11
 64  joint                   0.0001    75.000    75.600        1       64      0.12
 65  joint                   0.0001    74.800    75.600        1       65      0.12
 66  joint                   0.0001    74.800    75.600        1       66      0.12
 67  joint                   0.0001    74.800    75.600        1       67      0.12
 68  joint                   0.0001    74.600    75.600        1       68      0.11
 69  joint                   0.0001    74.600    75.600        1       69      0.12
 70  joint                   0.0001    74.600    75.600        1       70      0.12
 71  joint                   0.0001    74.600    75.600        1       71      0.12
 72  joint                   0.0001    74.600    75.600        1       72      0.11
 73  joint                   0.0001    74.600    75.600        1       73      0.12
 74  joint                   0.0001    74.600    75.600        1       74      0.12
 75  joint                   0.0001    74.600    75.600        1       75      0.12
 76  joint                   0.0001    74.600    75.600        1       76      0.12
 77  joint                   0.0001    74.200    75.600        1       77      0.12
 78  joint                   0.0001    74.400    75.600        1       78      0.12
 79  joint                   0.0001    74.200    75.600        1       79      0.12
 80  joint                   0.0001    74.200    75.600        1       80      0.12
 81  joint                   0.0001    74.200    75.600        1       81      0.12
 82  joint                   0.0001    74.200    75.600        1       82      0.12
 83  joint                   0.0001    74.200    75.600        1       83      0.12
 84  joint                   0.0001    74.200    75.600        1       84      0.12
 85  joint                   0.0001    74.200    75.600        1       85      0.12
 86  joint                   0.0001    74.200    75.600        1       86      0.11
 87  joint                   0.0001    74.200    75.600        1       87      0.12
 88  joint                   0.0001    74.200    75.600        1       88      0.12
 89  joint                   0.0001    74.200    75.600        1       89      0.12
 90  joint                   0.0001    74.200    75.600        1       90      0.12
 91  joint                   0.0001    74.000    75.600        1       91      0.12
 92  joint                   0.0001    74.000    75.600        1       92      0.12
 93  joint                   0.0001    74.000    75.600        1       93      0.12
 94  joint                   0.0001    74.000    75.600        1       94      0.12
 95  joint                   0.0001    74.000    75.600        1       95      0.12
 96  joint                   0.0001    74.000    75.600        1       96      0.11
 97  joint                   0.0001    74.000    75.600        1       97      0.12
```

</details>

<details>
<summary>pubmed | reference | shared_dynamic_c — 전체 에포크</summary>

```text
[pubmed | reference | shared_dynamic_c]
status=passed | last_epoch=60 | selected_epoch=10
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.1396    64.800    64.800        1        1      0.68
  2  joint                   0.1893    69.000    69.000        1        2      0.34
  3  joint                   0.0345    70.400    70.400        1        3      0.34
  4  joint                   0.0105    70.600    70.600        1        4      0.34
  5  joint                   0.0030    71.200    71.200        1        5      0.34
  6  joint                   0.0015    72.200    72.200        1        6      0.50
  7  joint                   0.0008    72.400    72.400        1        7      0.33
  8  joint                   0.0005    73.000    73.000        1        8      0.34
  9  joint                   0.0004    74.000    74.000        1        9      0.34
 10  joint                   0.0003    74.200    74.200        1       10      0.34 *
 11  joint                   0.0003    74.000    74.200        1       11      0.43
 12  joint                   0.0002    73.600    74.200        1       12      0.34
 13  joint                   0.0002    73.200    74.200        1       13      0.34
 14  joint                   0.0002    73.400    74.200        1       14      0.34
 15  joint                   0.0002    73.600    74.200        1       15      0.33
 16  joint                   0.0001    73.600    74.200        1       16      0.34
 17  joint                   0.0001    73.600    74.200        1       17      0.35
 18  joint                   0.0001    73.600    74.200        1       18      0.33
 19  joint                   0.0001    73.600    74.200        1       19      0.33
 20  joint                   0.0001    73.600    74.200        1       20      0.45
 21  joint                   0.0001    73.600    74.200        1       21      0.33
 22  joint                   0.0001    73.600    74.200        1       22      0.33
 23  joint                   0.0001    73.600    74.200        1       23      0.33
 24  joint                   0.0001    73.600    74.200        1       24      0.33
 25  joint                   0.0001    73.600    74.200        1       25      0.33
 26  joint                   0.0001    73.600    74.200        1       26      0.33
 27  joint                   0.0001    73.600    74.200        1       27      0.33
 28  joint                   0.0001    73.600    74.200        1       28      0.33
 29  joint                   0.0001    73.600    74.200        1       29      0.33
 30  joint                   0.0001    73.600    74.200        1       30      0.43
 31  joint                   0.0001    73.600    74.200        1       31      0.33
 32  joint                   0.0001    73.600    74.200        1       32      0.33
 33  joint                   0.0001    73.600    74.200        1       33      0.33
 34  joint                   0.0001    73.600    74.200        1       34      0.33
 35  joint                   0.0001    73.600    74.200        1       35      0.33
 36  joint                   0.0001    73.600    74.200        1       36      0.33
 37  joint                   0.0001    73.800    74.200        1       37      0.34
 38  joint                   0.0001    73.800    74.200        1       38      0.33
 39  joint                   0.0001    73.800    74.200        1       39      0.33
 40  joint                   0.0001    73.600    74.200        1       40      0.43
 41  joint                   0.0001    73.800    74.200        1       41      0.33
 42  joint                   0.0001    73.800    74.200        1       42      0.33
 43  joint                   0.0001    73.800    74.200        1       43      0.33
 44  joint                   0.0001    73.800    74.200        1       44      0.31
 45  joint                   0.0001    73.800    74.200        1       45      0.33
 46  joint                   0.0001    73.800    74.200        1       46      0.34
 47  joint                   0.0001    73.800    74.200        1       47      0.33
 48  joint                   0.0001    73.800    74.200        1       48      0.33
 49  joint                   0.0001    73.800    74.200        1       49      0.33
 50  joint                   0.0001    73.800    74.200        1       50      0.43
 51  joint                   0.0001    73.800    74.200        1       51      0.33
 52  joint                   0.0001    73.800    74.200        1       52      0.33
 53  joint                   0.0001    73.800    74.200        1       53      0.33
 54  joint                   0.0001    73.600    74.200        1       54      0.33
 55  joint                   0.0001    73.800    74.200        1       55      0.33
 56  joint                   0.0001    73.600    74.200        1       56      0.33
 57  joint                   0.0001    73.800    74.200        1       57      0.33
 58  joint                   0.0001    73.600    74.200        1       58      0.33
 59  joint                   0.0001    73.600    74.200        1       59      0.33
 60  joint                   0.0001    73.600    74.200        1       60      0.41
```

</details>

<details>
<summary>pubmed | large | fixed_c — 전체 에포크</summary>

```text
[pubmed | large | fixed_c]
status=passed | last_epoch=57 | selected_epoch=7
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.2252    69.200    69.200        1        1      0.45
  2  joint                   0.1844    65.000    69.200        1        2      0.37
  3  joint                   0.1423    67.400    69.200        1        3      0.24
  4  joint                   0.0181    71.600    71.600        1        4      0.25
  5  joint                   0.0185    72.800    72.800        1        5      0.24
  6  joint                   0.0021    73.200    73.200        1        6      0.25
  7  joint                   0.0008    73.600    73.600        1        7      0.25 *
  8  joint                   0.0006    73.000    73.600        1        8      0.25
  9  joint                   0.0006    71.800    73.600        1        9      0.24
 10  joint                   0.0004    70.800    73.600        1       10      0.25
 11  joint                   0.0006    70.000    73.600        1       11      0.24
 12  joint                   0.0006    69.200    73.600        1       12      0.25
 13  joint                   0.0004    69.400    73.600        1       13      0.25
 14  joint                   0.0006    68.800    73.600        1       14      0.25
 15  joint                   0.0003    68.200    73.600        1       15      0.25
 16  joint                   0.0004    68.000    73.600        1       16      0.24
 17  joint                   0.0002    68.200    73.600        1       17      0.24
 18  joint                   0.0003    68.200    73.600        1       18      0.25
 19  joint                   0.0002    68.200    73.600        1       19      0.24
 20  joint                   0.0002    68.600    73.600        1       20      0.25
 21  joint                   0.0001    68.600    73.600        1       21      0.24
 22  joint                   0.0001    68.600    73.600        1       22      0.25
 23  joint                   0.0001    68.600    73.600        1       23      0.24
 24  joint                   0.0001    68.800    73.600        1       24      0.25
 25  joint                   0.0001    68.800    73.600        1       25      0.24
 26  joint                   0.0001    68.800    73.600        1       26      0.25
 27  joint                   0.0001    68.800    73.600        1       27      0.25
 28  joint                   0.0001    69.000    73.600        1       28      0.24
 29  joint                   0.0001    69.200    73.600        1       29      0.24
 30  joint                   0.0001    69.200    73.600        1       30      0.24
 31  joint                   0.0001    69.200    73.600        1       31      0.24
 32  joint                   0.0001    69.200    73.600        1       32      0.24
 33  joint                   0.0001    69.200    73.600        1       33      0.24
 34  joint                   0.0000    69.000    73.600        1       34      0.25
 35  joint                   0.0000    69.000    73.600        1       35      0.25
 36  joint                   0.0000    69.000    73.600        1       36      0.25
 37  joint                   0.0000    69.200    73.600        1       37      0.24
 38  joint                   0.0000    69.200    73.600        1       38      0.25
 39  joint                   0.0000    69.200    73.600        1       39      0.24
 40  joint                   0.0000    69.400    73.600        1       40      0.25
 41  joint                   0.0000    69.200    73.600        1       41      0.25
 42  joint                   0.0000    69.200    73.600        1       42      0.24
 43  joint                   0.0000    69.200    73.600        1       43      0.24
 44  joint                   0.0000    69.200    73.600        1       44      0.25
 45  joint                   0.0000    69.200    73.600        1       45      0.25
 46  joint                   0.0000    69.200    73.600        1       46      0.25
 47  joint                   0.0000    69.200    73.600        1       47      0.24
 48  joint                   0.0000    69.200    73.600        1       48      0.24
 49  joint                   0.0000    69.400    73.600        1       49      0.24
 50  joint                   0.0000    69.200    73.600        1       50      0.25
 51  joint                   0.0000    69.600    73.600        1       51      0.25
 52  joint                   0.0000    69.400    73.600        1       52      0.24
 53  joint                   0.0000    69.400    73.600        1       53      0.25
 54  joint                   0.0000    69.600    73.600        1       54      0.24
 55  joint                   0.0000    69.600    73.600        1       55      0.25
 56  joint                   0.0000    69.800    73.600        1       56      0.25
 57  joint                   0.0000    70.000    73.600        1       57      0.25
```

</details>

<details>
<summary>pubmed | large | shared_dynamic_c — 전체 에포크</summary>

```text
[pubmed | large | shared_dynamic_c]
status=passed | last_epoch=56 | selected_epoch=6
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   1.2379    69.000    69.000        1        1      0.86
  2  joint                   0.1770    63.800    69.000        1        2      0.58
  3  joint                   0.1297    66.400    69.000        1        3      0.49
  4  joint                   0.0185    69.800    69.800        1        4      0.51
  5  joint                   0.0119    71.400    71.400        1        5      0.51
  6  joint                   0.0016    74.000    74.000        1        6      0.50 *
  7  joint                   0.0007    73.400    74.000        1        7      0.61
  8  joint                   0.0006    72.000    74.000        1        8      0.50
  9  joint                   0.0005    72.400    74.000        1        9      0.51
 10  joint                   0.0004    71.600    74.000        1       10      0.51
 11  joint                   0.0004    71.400    74.000        1       11      0.51
 12  joint                   0.0005    70.600    74.000        1       12      0.51
 13  joint                   0.0003    70.200    74.000        1       13      0.62
 14  joint                   0.0004    70.000    74.000        1       14      0.51
 15  joint                   0.0002    70.000    74.000        1       15      0.51
 16  joint                   0.0002    69.600    74.000        1       16      0.51
 17  joint                   0.0001    69.600    74.000        1       17      0.51
 18  joint                   0.0002    69.400    74.000        1       18      0.60
 19  joint                   0.0001    69.200    74.000        1       19      0.55
 20  joint                   0.0001    69.400    74.000        1       20      0.51
 21  joint                   0.0001    69.400    74.000        1       21      0.51
 22  joint                   0.0001    69.400    74.000        1       22      0.50
 23  joint                   0.0001    69.400    74.000        1       23      0.49
 24  joint                   0.0001    69.800    74.000        1       24      0.58
 25  joint                   0.0001    69.800    74.000        1       25      0.49
 26  joint                   0.0001    70.000    74.000        1       26      0.50
 27  joint                   0.0001    70.200    74.000        1       27      0.50
 28  joint                   0.0001    70.400    74.000        1       28      0.51
 29  joint                   0.0001    70.400    74.000        1       29      0.60
 30  joint                   0.0001    70.400    74.000        1       30      0.51
 31  joint                   0.0001    70.200    74.000        1       31      0.51
 32  joint                   0.0001    70.000    74.000        1       32      0.50
 33  joint                   0.0001    70.000    74.000        1       33      0.51
 34  joint                   0.0001    70.000    74.000        1       34      0.50
 35  joint                   0.0001    70.000    74.000        1       35      0.59
 36  joint                   0.0001    70.000    74.000        1       36      0.50
 37  joint                   0.0001    70.000    74.000        1       37      0.51
 38  joint                   0.0001    70.000    74.000        1       38      0.50
 39  joint                   0.0000    70.000    74.000        1       39      0.50
 40  joint                   0.0000    70.000    74.000        1       40      0.60
 41  joint                   0.0000    70.000    74.000        1       41      0.49
 42  joint                   0.0000    70.000    74.000        1       42      0.49
 43  joint                   0.0000    70.000    74.000        1       43      0.51
 44  joint                   0.0000    70.000    74.000        1       44      0.51
 45  joint                   0.0000    70.000    74.000        1       45      0.51
 46  joint                   0.0000    70.000    74.000        1       46      0.63
 47  joint                   0.0000    70.000    74.000        1       47      0.50
 48  joint                   0.0000    70.000    74.000        1       48      0.50
 49  joint                   0.0000    70.000    74.000        1       49      0.51
 50  joint                   0.0000    70.000    74.000        1       50      0.51
 51  joint                   0.0000    70.000    74.000        1       51      0.60
 52  joint                   0.0000    70.000    74.000        1       52      0.51
 53  joint                   0.0000    70.000    74.000        1       53      0.50
 54  joint                   0.0000    70.000    74.000        1       54      0.51
 55  joint                   0.0000    70.000    74.000        1       55      0.50
 56  joint                   0.0000    70.000    74.000        1       56      0.51
```

</details>

<details>
<summary>ppi | reference | fixed_c — 전체 에포크</summary>

```text
[ppi | reference | fixed_c]
status=passed | last_epoch=600 | selected_epoch=599
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   0.7404    41.996    41.996        1        1      1.12
  2  joint                   0.6544    41.939    41.996        1        2      0.41
  3  joint                   0.6138    41.239    41.996        1        3      0.42
  4  joint                   0.5895    41.272    41.996        1        4      0.43
  5  joint                   0.5745    40.347    41.996        1        5      0.43
  6  joint                   0.5652    39.686    41.996        1        6      0.43
  7  joint                   0.5594    39.589    41.996        1        7      0.43
  8  joint                   0.5556    40.006    41.996        1        8      0.43
  9  joint                   0.5532    40.682    41.996        1        9      0.43
 10  joint                   0.5516    40.724    41.996        1       10      0.42
 11  joint                   0.5506    40.498    41.996        1       11      0.43
 12  joint                   0.5500    40.171    41.996        1       12      0.43
 13  joint                   0.5496    39.953    41.996        1       13      0.43
 14  joint                   0.5492    39.937    41.996        1       14      0.43
 15  joint                   0.5489    40.050    41.996        1       15      0.43
 16  joint                   0.5485    40.376    41.996        1       16      0.43
 17  joint                   0.5479    40.777    41.996        1       17      0.43
 18  joint                   0.5472    41.316    41.996        1       18      0.42
 19  joint                   0.5463    41.957    41.996        1       19      0.43
 20  joint                   0.5451    42.582    42.582        1       20      0.43
 21  joint                   0.5439    43.179    43.179        1       21      0.44
 22  joint                   0.5426    43.549    43.549        1       22      0.44
 23  joint                   0.5412    43.703    43.703        1       23      0.42
 24  joint                   0.5395    44.622    44.622        1       24      0.45
 25  joint                   0.5377    45.783    45.783        1       25      0.44
 26  joint                   0.5360    45.529    45.783        1       26      0.43
 27  joint                   0.5345    48.120    48.120        1       27      0.44
 28  joint                   0.5336    42.689    48.120        1       28      0.44
 29  joint                   0.5346    49.032    49.032        1       29      0.43
 30  joint                   0.5309    47.925    49.032        1       30      0.44
 31  joint                   0.5289    45.423    49.032        1       31      0.44
 32  joint                   0.5286    50.942    50.942        1       32      0.43
 33  joint                   0.5281    42.440    50.942        1       33      0.44
 34  joint                   0.5302    46.383    50.942        1       34      0.44
 35  joint                   0.5245    52.743    52.743        1       35      0.43
 36  joint                   0.5336    42.403    52.743        1       36      0.44
 37  joint                   0.5283    40.961    52.743        1       37      0.44
 38  joint                   0.5329    44.483    52.743        1       38      0.44
 39  joint                   0.5267    49.320    52.743        1       39      0.43
 40  joint                   0.5250    51.098    52.743        1       40      0.42
 41  joint                   0.5291    49.833    52.743        1       41      0.44
 42  joint                   0.5262    46.511    52.743        1       42      0.43
 43  joint                   0.5227    43.269    52.743        1       43      0.43
 44  joint                   0.5240    42.724    52.743        1       44      0.42
 45  joint                   0.5244    45.660    52.743        1       45      0.43
 46  joint                   0.5208    49.598    52.743        1       46      0.43
 47  joint                   0.5189    51.533    52.743        1       47      0.43
 48  joint                   0.5202    50.519    52.743        1       48      0.43
 49  joint                   0.5168    47.361    52.743        1       49      0.43
 50  joint                   0.5152    46.302    52.743        1       50      0.43
 51  joint                   0.5160    50.384    52.743        1       51      0.44
 52  joint                   0.5116    52.847    52.847        1       52      0.44
 53  joint                   0.5137    47.953    52.847        1       53      0.44
 54  joint                   0.5102    49.021    52.847        1       54      0.43
 55  joint                   0.5076    53.741    53.741        1       55      0.43
 56  joint                   0.5100    49.660    53.741        1       56      0.44
 57  joint                   0.5042    49.953    53.741        1       57      0.44
 58  joint                   0.5040    52.218    53.741        1       58      0.44
 59  joint                   0.5056    52.992    53.741        1       59      0.44
 60  joint                   0.5026    41.917    53.741        1       60      0.44
 61  joint                   0.5212    55.950    55.950        1       61      0.44
 62  joint                   0.5144    53.107    55.950        1       62      0.44
 63  joint                   0.5050    48.696    55.950        1       63      0.43
 64  joint                   0.5035    49.502    55.950        1       64      0.43
 65  joint                   0.5046    51.826    55.950        1       65      0.43
 66  joint                   0.5054    51.712    55.950        1       66      0.43
 67  joint                   0.4987    49.363    55.950        1       67      0.44
 68  joint                   0.4994    51.647    55.950        1       68      0.43
 69  joint                   0.4974    54.331    55.950        1       69      0.43
 70  joint                   0.4995    51.061    55.950        1       70      0.43
 71  joint                   0.4951    51.216    55.950        1       71      0.43
 72  joint                   0.4933    54.745    55.950        1       72      0.43
 73  joint                   0.4944    51.836    55.950        1       73      0.43
 74  joint                   0.4905    53.122    55.950        1       74      0.43
 75  joint                   0.4878    55.475    55.950        1       75      0.43
 76  joint                   0.4882    52.479    55.950        1       76      0.43
 77  joint                   0.4864    54.112    55.950        1       77      0.43
 78  joint                   0.4830    55.796    55.950        1       78      0.42
 79  joint                   0.4825    53.249    55.950        1       79      0.44
 80  joint                   0.4822    56.982    56.982        1       80      0.44
 81  joint                   0.4806    51.923    56.982        1       81      0.44
 82  joint                   0.4812    57.343    57.343        1       82      0.44
 83  joint                   0.4777    54.166    57.343        1       83      0.45
 84  joint                   0.4769    56.781    57.343        1       84      0.43
 85  joint                   0.4749    56.497    57.343        1       85      0.43
 86  joint                   0.4711    56.970    57.343        1       86      0.43
 87  joint                   0.4704    56.365    57.343        1       87      0.44
 88  joint                   0.4700    53.960    57.343        1       88      0.44
 89  joint                   0.4698    60.023    60.023        1       89      0.44
 90  joint                   0.4693    56.169    60.023        1       90      0.44
 91  joint                   0.4627    56.037    60.023        1       91      0.43
 92  joint                   0.4625    60.003    60.023        1       92      0.44
 93  joint                   0.4608    58.056    60.023        1       93      0.44
 94  joint                   0.4619    58.824    60.023        1       94      0.43
 95  joint                   0.4610    55.700    60.023        1       95      0.44
 96  joint                   0.4677    59.546    60.023        1       96      0.43
 97  joint                   0.4614    54.092    60.023        1       97      0.43
 98  joint                   0.4700    59.623    60.023        1       98      0.43
 99  joint                   0.4572    60.691    60.691        1       99      0.43
100  joint                   0.4583    57.676    60.691        1      100      0.44
101  joint                   0.4532    56.398    60.691        1      101      0.44
102  joint                   0.4549    61.147    61.147        1      102      0.42
103  joint                   0.4478    61.838    61.838        1      103      0.44
104  joint                   0.4505    61.063    61.838        1      104      0.44
105  joint                   0.4451    59.737    61.838        1      105      0.44
106  joint                   0.4446    58.776    61.838        1      106      0.44
107  joint                   0.4432    61.617    61.838        1      107      0.44
108  joint                   0.4397    61.933    61.933        1      108      0.44
109  joint                   0.4398    62.834    62.834        1      109      0.44
110  joint                   0.4383    61.507    62.834        1      110      0.44
111  joint                   0.4386    60.111    62.834        1      111      0.43
112  joint                   0.4376    61.781    62.834        1      112      0.43
113  joint                   0.4460    57.002    62.834        1      113      0.44
114  joint                   0.4494    60.791    62.834        1      114      0.44
115  joint                   0.4340    62.880    62.880        1      115      0.44
116  joint                   0.4384    63.703    63.703        1      116      0.42
117  joint                   0.4371    60.729    63.703        1      117      0.45
118  joint                   0.4290    58.939    63.703        1      118      0.43
119  joint                   0.4311    61.602    63.703        1      119      0.43
120  joint                   0.4246    63.665    63.703        1      120      0.43
121  joint                   0.4245    64.883    64.883        1      121      0.43
122  joint                   0.4206    64.501    64.883        1      122      0.44
123  joint                   0.4190    63.360    64.883        1      123      0.43
124  joint                   0.4162    62.892    64.883        1      124      0.43
125  joint                   0.4131    64.304    64.883        1      125      0.43
126  joint                   0.4113    65.188    65.188        1      126      0.43
127  joint                   0.4096    65.707    65.707        1      127      0.43
128  joint                   0.4077    65.370    65.707        1      128      0.44
129  joint                   0.4048    64.826    65.707        1      129      0.44
130  joint                   0.4062    64.287    65.707        1      130      0.42
131  joint                   0.4076    65.982    65.982        1      131      0.44
132  joint                   0.4052    66.647    66.647        1      132      0.44
133  joint                   0.4003    65.319    66.647        1      133      0.44
134  joint                   0.4043    64.211    66.647        1      134      0.43
135  joint                   0.4082    67.118    67.118        1      135      0.44
136  joint                   0.3983    66.241    67.118        1      136      0.44
137  joint                   0.3944    66.004    67.118        1      137      0.44
138  joint                   0.3927    67.622    67.622        1      138      0.43
139  joint                   0.3892    68.619    68.619        1      139      0.44
140  joint                   0.3866    68.235    68.619        1      140      0.44
141  joint                   0.3840    67.831    68.619        1      141      0.43
142  joint                   0.3818    68.692    68.692        1      142      0.42
143  joint                   0.3799    69.457    69.457        1      143      0.44
144  joint                   0.3768    68.460    69.457        1      144      0.44
145  joint                   0.3762    69.844    69.844        1      145      0.44
146  joint                   0.3715    70.446    70.446        1      146      0.44
147  joint                   0.3690    69.241    70.446        1      147      0.44
148  joint                   0.3710    68.723    70.446        1      148      0.43
149  joint                   0.3746    70.011    70.446        1      149      0.43
150  joint                   0.3706    71.070    71.070        1      150      0.43
151  joint                   0.3665    70.956    71.070        1      151      0.44
152  joint                   0.3602    69.995    71.070        1      152      0.43
153  joint                   0.3623    70.217    71.070        1      153      0.43
154  joint                   0.3578    72.171    72.171        1      154      0.43
155  joint                   0.3582    71.224    72.171        1      155      0.44
156  joint                   0.3544    71.777    72.171        1      156      0.44
157  joint                   0.3510    72.641    72.641        1      157      0.44
158  joint                   0.3490    72.772    72.772        1      158      0.44
159  joint                   0.3445    72.509    72.772        1      159      0.44
160  joint                   0.3427    73.185    73.185        1      160      0.43
161  joint                   0.3403    73.969    73.969        1      161      0.44
162  joint                   0.3377    74.458    74.458        1      162      0.44
163  joint                   0.3378    69.942    74.458        1      163      0.45
164  joint                   0.3478    71.646    74.458        1      164      0.43
165  joint                   0.3672    74.071    74.458        1      165      0.43
166  joint                   0.3410    70.524    74.458        1      166      0.43
167  joint                   0.3461    72.219    74.458        1      167      0.43
168  joint                   0.3405    73.639    74.458        1      168      0.43
169  joint                   0.3371    74.389    74.458        1      169      0.43
170  joint                   0.3341    74.516    74.516        1      170      0.43
171  joint                   0.3306    74.301    74.516        1      171      0.44
172  joint                   0.3309    74.305    74.516        1      172      0.43
173  joint                   0.3278    74.911    74.911        1      173      0.43
174  joint                   0.3277    75.054    75.054        1      174      0.44
175  joint                   0.3226    75.688    75.688        1      175      0.44
176  joint                   0.3200    75.616    75.688        1      176      0.44
177  joint                   0.3184    76.497    76.497        1      177      0.43
178  joint                   0.3140    76.240    76.497        1      178      0.44
179  joint                   0.3111    77.026    77.026        1      179      0.43
180  joint                   0.3095    76.980    77.026        1      180      0.43
181  joint                   0.3060    76.734    77.026        1      181      0.43
182  joint                   0.3049    77.724    77.724        1      182      0.43
183  joint                   0.3009    78.265    78.265        1      183      0.44
184  joint                   0.2996    77.536    78.265        1      184      0.44
185  joint                   0.2995    76.960    78.265        1      185      0.44
186  joint                   0.3055    78.171    78.265        1      186      0.43
187  joint                   0.2993    78.159    78.265        1      187      0.44
188  joint                   0.2946    78.827    78.827        1      188      0.44
189  joint                   0.2903    78.907    78.907        1      189      0.44
190  joint                   0.2907    79.165    79.165        1      190      0.44
191  joint                   0.2864    79.182    79.182        1      191      0.44
192  joint                   0.2827    79.697    79.697        1      192      0.43
193  joint                   0.2815    80.339    80.339        1      193      0.45
194  joint                   0.2777    80.271    80.339        1      194      0.43
195  joint                   0.2759    80.508    80.508        1      195      0.43
196  joint                   0.2749    80.393    80.508        1      196      0.43
197  joint                   0.2729    80.598    80.598        1      197      0.43
198  joint                   0.2724    81.014    81.014        1      198      0.44
199  joint                   0.2737    79.958    81.014        1      199      0.43
200  joint                   0.2714    80.504    81.014        1      200      0.43
201  joint                   0.2742    80.513    81.014        1      201      0.43
202  joint                   0.2706    81.351    81.351        1      202      0.43
203  joint                   0.2685    80.819    81.351        1      203      0.44
204  joint                   0.2647    81.991    81.991        1      204      0.43
205  joint                   0.2649    80.675    81.991        1      205      0.44
206  joint                   0.2710    79.838    81.991        1      206      0.43
207  joint                   0.2722    80.142    81.991        1      207      0.43
208  joint                   0.2748    80.030    81.991        1      208      0.43
209  joint                   0.2799    80.678    81.991        1      209      0.43
210  joint                   0.2691    81.025    81.991        1      210      0.43
211  joint                   0.2655    81.675    81.991        1      211      0.43
212  joint                   0.2586    82.080    82.080        1      212      0.43
213  joint                   0.2576    82.196    82.196        1      213      0.43
214  joint                   0.2521    82.523    82.523        1      214      0.44
215  joint                   0.2512    83.142    83.142        1      215      0.44
216  joint                   0.2482    83.513    83.513        1      216      0.44
217  joint                   0.2436    83.335    83.513        1      217      0.44
218  joint                   0.2427    83.614    83.614        1      218      0.44
219  joint                   0.2383    83.864    83.864        1      219      0.44
220  joint                   0.2373    84.289    84.289        1      220      0.44
221  joint                   0.2343    84.626    84.626        1      221      0.44
222  joint                   0.2321    84.825    84.825        1      222      0.44
223  joint                   0.2290    84.957    84.957        1      223      0.44
224  joint                   0.2270    85.193    85.193        1      224      0.44
225  joint                   0.2247    85.403    85.403        1      225      0.45
226  joint                   0.2225    85.704    85.704        1      226      0.44
227  joint                   0.2202    85.936    85.936        1      227      0.44
228  joint                   0.2183    85.917    85.936        1      228      0.44
229  joint                   0.2162    86.217    86.217        1      229      0.44
230  joint                   0.2138    86.551    86.551        1      230      0.44
231  joint                   0.2116    86.651    86.651        1      231      0.44
232  joint                   0.2100    86.923    86.923        1      232      0.44
233  joint                   0.2076    87.055    87.055        1      233      0.44
234  joint                   0.2057    87.252    87.252        1      234      0.44
235  joint                   0.2034    87.491    87.491        1      235      0.44
236  joint                   0.2021    87.672    87.672        1      236      0.44
237  joint                   0.2000    87.726    87.726        1      237      0.44
238  joint                   0.1984    87.789    87.789        1      238      0.44
239  joint                   0.1981    87.065    87.789        1      239      0.44
240  joint                   0.2026    87.025    87.789        1      240      0.44
241  joint                   0.2059    87.747    87.789        1      241      0.44
242  joint                   0.1986    86.764    87.789        1      242      0.44
243  joint                   0.2024    87.493    87.789        1      243      0.44
244  joint                   0.2058    85.109    87.789        1      244      0.43
245  joint                   0.2191    80.637    87.789        1      245      0.43
246  joint                   0.2696    80.981    87.789        1      246      0.43
247  joint                   0.2651    84.721    87.789        1      247      0.43
248  joint                   0.2333    83.232    87.789        1      248      0.43
249  joint                   0.2374    83.546    87.789        1      249      0.44
250  joint                   0.2448    83.492    87.789        1      250      0.44
251  joint                   0.2387    84.505    87.789        1      251      0.44
252  joint                   0.2308    84.862    87.789        1      252      0.43
253  joint                   0.2246    85.104    87.789        1      253      0.43
254  joint                   0.2184    85.975    87.789        1      254      0.43
255  joint                   0.2134    86.122    87.789        1      255      0.41
256  joint                   0.2126    86.783    87.789        1      256      0.43
257  joint                   0.2047    87.040    87.789        1      257      0.43
258  joint                   0.2012    87.137    87.789        1      258      0.43
259  joint                   0.1999    87.434    87.789        1      259      0.44
260  joint                   0.1957    87.657    87.789        1      260      0.43
261  joint                   0.1941    88.197    88.197        1      261      0.43
262  joint                   0.1899    88.539    88.539        1      262      0.44
263  joint                   0.1873    88.665    88.665        1      263      0.44
264  joint                   0.1855    88.830    88.830        1      264      0.44
265  joint                   0.1830    88.935    88.935        1      265      0.44
266  joint                   0.1808    89.104    89.104        1      266      0.44
267  joint                   0.1790    89.300    89.300        1      267      0.44
268  joint                   0.1770    89.502    89.502        1      268      0.44
269  joint                   0.1749    89.732    89.732        1      269      0.44
270  joint                   0.1726    89.891    89.891        1      270      0.44
271  joint                   0.1706    89.931    89.931        1      271      0.44
272  joint                   0.1690    90.065    90.065        1      272      0.44
273  joint                   0.1676    90.286    90.286        1      273      0.44
274  joint                   0.1659    90.414    90.414        1      274      0.44
275  joint                   0.1641    90.464    90.464        1      275      0.44
276  joint                   0.1624    90.550    90.550        1      276      0.44
277  joint                   0.1611    90.742    90.742        1      277      0.44
278  joint                   0.1596    90.904    90.904        1      278      0.44
279  joint                   0.1581    90.972    90.972        1      279      0.44
280  joint                   0.1566    91.064    91.064        1      280      0.44
281  joint                   0.1553    91.185    91.185        1      281      0.44
282  joint                   0.1539    91.362    91.362        1      282      0.44
283  joint                   0.1523    91.463    91.463        1      283      0.44
284  joint                   0.1514    91.530    91.530        1      284      0.44
285  joint                   0.1499    91.611    91.611        1      285      0.44
286  joint                   0.1487    91.758    91.758        1      286      0.44
287  joint                   0.1472    91.871    91.871        1      287      0.44
288  joint                   0.1461    91.946    91.946        1      288      0.44
289  joint                   0.1447    92.021    92.021        1      289      0.44
290  joint                   0.1436    92.120    92.120        1      290      0.44
291  joint                   0.1424    92.227    92.227        1      291      0.44
292  joint                   0.1413    92.303    92.303        1      292      0.43
293  joint                   0.1406    92.379    92.379        1      293      0.43
294  joint                   0.1391    92.471    92.471        1      294      0.43
295  joint                   0.1381    92.565    92.565        1      295      0.43
296  joint                   0.1371    92.631    92.631        1      296      0.44
297  joint                   0.1360    92.699    92.699        1      297      0.44
298  joint                   0.1349    92.775    92.775        1      298      0.43
299  joint                   0.1339    92.860    92.860        1      299      0.43
300  joint                   0.1329    92.925    92.925        1      300      0.43
301  joint                   0.1320    92.989    92.989        1      301      0.43
302  joint                   0.1309    93.101    93.101        1      302      0.43
303  joint                   0.1297    93.152    93.152        1      303      0.43
304  joint                   0.1290    93.214    93.214        1      304      0.43
305  joint                   0.1277    93.319    93.319        1      305      0.43
306  joint                   0.1268    93.403    93.403        1      306      0.43
307  joint                   0.1262    93.461    93.461        1      307      0.42
308  joint                   0.1250    93.515    93.515        1      308      0.43
309  joint                   0.1238    93.583    93.583        1      309      0.43
310  joint                   0.1232    93.656    93.656        1      310      0.43
311  joint                   0.1222    93.725    93.725        1      311      0.43
312  joint                   0.1212    93.757    93.757        1      312      0.44
313  joint                   0.1204    93.830    93.830        1      313      0.44
314  joint                   0.1196    93.924    93.924        1      314      0.43
315  joint                   0.1188    93.983    93.983        1      315      0.44
316  joint                   0.1180    94.006    94.006        1      316      0.44
317  joint                   0.1170    94.103    94.103        1      317      0.44
318  joint                   0.1160    94.177    94.177        1      318      0.44
319  joint                   0.1152    94.216    94.216        1      319      0.44
320  joint                   0.1144    94.278    94.278        1      320      0.43
321  joint                   0.1137    94.332    94.332        1      321      0.43
322  joint                   0.1129    94.381    94.381        1      322      0.43
323  joint                   0.1121    94.457    94.457        1      323      0.43
324  joint                   0.1111    94.518    94.518        1      324      0.43
325  joint                   0.1105    94.591    94.591        1      325      0.43
326  joint                   0.1097    94.627    94.627        1      326      0.44
327  joint                   0.1089    94.688    94.688        1      327      0.43
328  joint                   0.1081    94.738    94.738        1      328      0.43
329  joint                   0.1074    94.788    94.788        1      329      0.43
330  joint                   0.1066    94.860    94.860        1      330      0.42
331  joint                   0.1059    94.906    94.906        1      331      0.43
332  joint                   0.1051    94.948    94.948        1      332      0.44
333  joint                   0.1044    95.009    95.009        1      333      0.44
334  joint                   0.1036    95.058    95.058        1      334      0.43
335  joint                   0.1030    95.101    95.101        1      335      0.44
336  joint                   0.1021    95.136    95.136        1      336      0.43
337  joint                   0.1013    95.182    95.182        1      337      0.43
338  joint                   0.1005    95.218    95.218        1      338      0.43
339  joint                   0.1000    95.265    95.265        1      339      0.44
340  joint                   0.0992    95.310    95.310        1      340      0.40
341  joint                   0.0986    95.365    95.365        1      341      0.43
342  joint                   0.0979    95.402    95.402        1      342      0.44
343  joint                   0.0973    95.453    95.453        1      343      0.43
344  joint                   0.0966    95.483    95.483        1      344      0.43
345  joint                   0.0959    95.532    95.532        1      345      0.44
346  joint                   0.0951    95.593    95.593        1      346      0.43
347  joint                   0.0946    95.629    95.629        1      347      0.43
348  joint                   0.0941    95.660    95.660        1      348      0.44
349  joint                   0.0934    95.699    95.699        1      349      0.44
350  joint                   0.0926    95.728    95.728        1      350      0.44
351  joint                   0.0921    95.770    95.770        1      351      0.43
352  joint                   0.0914    95.801    95.801        1      352      0.43
353  joint                   0.0909    95.827    95.827        1      353      0.43
354  joint                   0.0903    95.860    95.860        1      354      0.43
355  joint                   0.0897    95.917    95.917        1      355      0.43
356  joint                   0.0891    95.949    95.949        1      356      0.43
357  joint                   0.0881    95.995    95.995        1      357      0.43
358  joint                   0.0879    96.048    96.048        1      358      0.43
359  joint                   0.0872    96.073    96.073        1      359      0.44
360  joint                   0.0866    96.071    96.073        1      360      0.43
361  joint                   0.0861    96.127    96.127        1      361      0.42
362  joint                   0.0856    96.168    96.168        1      362      0.44
363  joint                   0.0848    96.196    96.196        1      363      0.43
364  joint                   0.0845    96.240    96.240        1      364      0.44
365  joint                   0.0837    96.253    96.253        1      365      0.43
366  joint                   0.0832    96.291    96.291        1      366      0.43
367  joint                   0.0827    96.322    96.322        1      367      0.44
368  joint                   0.0822    96.348    96.348        1      368      0.44
369  joint                   0.0817    96.384    96.384        1      369      0.43
370  joint                   0.0811    96.433    96.433        1      370      0.43
371  joint                   0.0805    96.454    96.454        1      371      0.43
372  joint                   0.0803    96.465    96.465        1      372      0.43
373  joint                   0.0796    96.500    96.500        1      373      0.43
374  joint                   0.0792    96.531    96.531        1      374      0.43
375  joint                   0.0786    96.562    96.562        1      375      0.43
376  joint                   0.0778    96.581    96.581        1      376      0.43
377  joint                   0.0774    96.591    96.591        1      377      0.43
378  joint                   0.0772    96.640    96.640        1      378      0.43
379  joint                   0.0767    96.618    96.640        1      379      0.43
380  joint                   0.0763    96.659    96.659        1      380      0.42
381  joint                   0.0754    96.699    96.699        1      381      0.42
382  joint                   0.0751    96.676    96.699        1      382      0.43
383  joint                   0.0749    96.731    96.731        1      383      0.43
384  joint                   0.0742    96.747    96.747        1      384      0.43
385  joint                   0.0737    96.789    96.789        1      385      0.42
386  joint                   0.0731    96.819    96.819        1      386      0.43
387  joint                   0.0729    96.828    96.828        1      387      0.43
388  joint                   0.0725    96.854    96.854        1      388      0.43
389  joint                   0.0720    96.912    96.912        1      389      0.43
390  joint                   0.0717    96.919    96.919        1      390      0.43
391  joint                   0.0711    96.932    96.932        1      391      0.43
392  joint                   0.0705    96.969    96.969        1      392      0.43
393  joint                   0.0702    96.992    96.992        1      393      0.42
394  joint                   0.0697    97.003    97.003        1      394      0.43
395  joint                   0.0693    97.025    97.025        1      395      0.43
396  joint                   0.0688    97.020    97.025        1      396      0.44
397  joint                   0.0684    97.058    97.058        1      397      0.42
398  joint                   0.0681    97.096    97.096        1      398      0.42
399  joint                   0.0677    97.088    97.096        1      399      0.43
400  joint                   0.0671    97.106    97.106        1      400      0.44
401  joint                   0.0670    97.150    97.150        1      401      0.44
402  joint                   0.0664    97.141    97.150        1      402      0.44
403  joint                   0.0661    97.176    97.176        1      403      0.43
404  joint                   0.0656    97.208    97.208        1      404      0.43
405  joint                   0.0653    97.207    97.208        1      405      0.44
406  joint                   0.0649    97.217    97.217        1      406      0.42
407  joint                   0.0646    97.252    97.252        1      407      0.44
408  joint                   0.0641    97.282    97.282        1      408      0.43
409  joint                   0.0637    97.258    97.282        1      409      0.44
410  joint                   0.0634    97.267    97.282        1      410      0.44
411  joint                   0.0630    97.318    97.318        1      411      0.43
412  joint                   0.0627    97.307    97.318        1      412      0.44
413  joint                   0.0624    97.320    97.320        1      413      0.44
414  joint                   0.0617    97.370    97.370        1      414      0.43
415  joint                   0.0616    97.355    97.370        1      415      0.43
416  joint                   0.0612    97.344    97.370        1      416      0.44
417  joint                   0.0609    97.400    97.400        1      417      0.43
418  joint                   0.0605    97.394    97.400        1      418      0.43
419  joint                   0.0602    97.396    97.400        1      419      0.43
420  joint                   0.0598    97.424    97.424        1      420      0.44
421  joint                   0.0596    97.431    97.431        1      421      0.44
422  joint                   0.0594    97.483    97.483        1      422      0.44
423  joint                   0.0588    97.493    97.493        1      423      0.44
424  joint                   0.0586    97.446    97.493        1      424      0.44
425  joint                   0.0583    97.500    97.500        1      425      0.43
426  joint                   0.0578    97.525    97.525        1      426      0.43
427  joint                   0.0575    97.503    97.525        1      427      0.43
428  joint                   0.0573    97.540    97.540        1      428      0.44
429  joint                   0.0568    97.564    97.564        1      429      0.44
430  joint                   0.0565    97.548    97.564        1      430      0.43
431  joint                   0.0562    97.570    97.570        1      431      0.43
432  joint                   0.0559    97.596    97.596        1      432      0.42
433  joint                   0.0556    97.589    97.596        1      433      0.43
434  joint                   0.0553    97.593    97.596        1      434      0.43
435  joint                   0.0550    97.623    97.623        1      435      0.43
436  joint                   0.0547    97.625    97.625        1      436      0.42
437  joint                   0.0543    97.634    97.634        1      437      0.44
438  joint                   0.0541    97.660    97.660        1      438      0.44
439  joint                   0.0538    97.677    97.677        1      439      0.43
440  joint                   0.0534    97.675    97.677        1      440      0.42
441  joint                   0.0532    97.677    97.677        1      441      0.42
442  joint                   0.0528    97.700    97.700        1      442      0.44
443  joint                   0.0526    97.709    97.709        1      443      0.43
444  joint                   0.0523    97.710    97.710        1      444      0.43
445  joint                   0.0520    97.711    97.711        1      445      0.43
446  joint                   0.0519    97.729    97.729        1      446      0.43
447  joint                   0.0515    97.746    97.746        1      447      0.40
448  joint                   0.0512    97.755    97.755        1      448      0.43
449  joint                   0.0511    97.757    97.757        1      449      0.44
450  joint                   0.0507    97.766    97.766        1      450      0.43
451  joint                   0.0504    97.761    97.766        1      451      0.43
452  joint                   0.0502    97.777    97.777        1      452      0.42
453  joint                   0.0500    97.791    97.791        1      453      0.44
454  joint                   0.0497    97.798    97.798        1      454      0.43
455  joint                   0.0495    97.796    97.798        1      455      0.43
456  joint                   0.0492    97.826    97.826        1      456      0.44
457  joint                   0.0491    97.845    97.845        1      457      0.44
458  joint                   0.0488    97.819    97.845        1      458      0.44
459  joint                   0.0484    97.851    97.851        1      459      0.43
460  joint                   0.0481    97.846    97.851        1      460      0.44
461  joint                   0.0479    97.846    97.851        1      461      0.40
462  joint                   0.0476    97.859    97.859        1      462      0.43
463  joint                   0.0475    97.868    97.868        1      463      0.43
464  joint                   0.0473    97.859    97.868        1      464      0.44
465  joint                   0.0471    97.875    97.875        1      465      0.44
466  joint                   0.0466    97.900    97.900        1      466      0.43
467  joint                   0.0464    97.905    97.905        1      467      0.44
468  joint                   0.0462    97.911    97.911        1      468      0.44
469  joint                   0.0460    97.921    97.921        1      469      0.44
470  joint                   0.0457    97.924    97.924        1      470      0.44
471  joint                   0.0456    97.935    97.935        1      471      0.44
472  joint                   0.0453    97.950    97.950        1      472      0.44
473  joint                   0.0451    97.947    97.950        1      473      0.44
474  joint                   0.0449    97.955    97.955        1      474      0.44
475  joint                   0.0448    97.962    97.962        1      475      0.44
476  joint                   0.0445    97.967    97.967        1      476      0.44
477  joint                   0.0441    97.968    97.968        1      477      0.44
478  joint                   0.0440    97.969    97.969        1      478      0.44
479  joint                   0.0439    97.983    97.983        1      479      0.44
480  joint                   0.0437    97.993    97.993        1      480      0.43
481  joint                   0.0433    97.989    97.993        1      481      0.44
482  joint                   0.0432    97.990    97.993        1      482      0.44
483  joint                   0.0429    98.006    98.006        1      483      0.44
484  joint                   0.0427    98.011    98.011        1      484      0.44
485  joint                   0.0426    98.014    98.014        1      485      0.44
486  joint                   0.0425    98.031    98.031        1      486      0.44
487  joint                   0.0422    98.024    98.031        1      487      0.44
488  joint                   0.0419    98.022    98.031        1      488      0.44
489  joint                   0.0418    98.037    98.037        1      489      0.44
490  joint                   0.0415    98.049    98.049        1      490      0.44
491  joint                   0.0414    98.057    98.057        1      491      0.44
492  joint                   0.0412    98.053    98.057        1      492      0.43
493  joint                   0.0410    98.059    98.059        1      493      0.43
494  joint                   0.0409    98.065    98.065        1      494      0.43
495  joint                   0.0406    98.076    98.076        1      495      0.43
496  joint                   0.0405    98.083    98.083        1      496      0.43
497  joint                   0.0403    98.093    98.093        1      497      0.44
498  joint                   0.0400    98.102    98.102        1      498      0.44
499  joint                   0.0398    98.100    98.102        1      499      0.44
500  joint                   0.0397    98.116    98.116        1      500      0.44
501  joint                   0.0394    98.113    98.116        1      501      0.43
502  joint                   0.0393    98.107    98.116        1      502      0.42
503  joint                   0.0390    98.127    98.127        1      503      0.44
504  joint                   0.0390    98.113    98.127        1      504      0.43
505  joint                   0.0388    98.117    98.127        1      505      0.43
506  joint                   0.0386    98.128    98.128        1      506      0.43
507  joint                   0.0383    98.120    98.128        1      507      0.44
508  joint                   0.0383    98.138    98.138        1      508      0.43
509  joint                   0.0383    98.141    98.141        1      509      0.44
510  joint                   0.0380    98.142    98.142        1      510      0.43
511  joint                   0.0378    98.160    98.160        1      511      0.44
512  joint                   0.0375    98.165    98.165        1      512      0.43
513  joint                   0.0374    98.162    98.165        1      513      0.44
514  joint                   0.0373    98.172    98.172        1      514      0.42
515  joint                   0.0372    98.175    98.175        1      515      0.44
516  joint                   0.0369    98.168    98.175        1      516      0.43
517  joint                   0.0367    98.185    98.185        1      517      0.43
518  joint                   0.0366    98.193    98.193        1      518      0.44
519  joint                   0.0365    98.169    98.193        1      519      0.43
520  joint                   0.0364    98.199    98.199        1      520      0.44
521  joint                   0.0361    98.204    98.204        1      521      0.43
522  joint                   0.0360    98.188    98.204        1      522      0.44
523  joint                   0.0359    98.203    98.204        1      523      0.42
524  joint                   0.0357    98.202    98.204        1      524      0.42
525  joint                   0.0357    98.196    98.204        1      525      0.42
526  joint                   0.0354    98.205    98.205        1      526      0.43
527  joint                   0.0351    98.222    98.222        1      527      0.44
528  joint                   0.0350    98.215    98.222        1      528      0.44
529  joint                   0.0349    98.219    98.222        1      529      0.43
530  joint                   0.0348    98.230    98.230        1      530      0.43
531  joint                   0.0347    98.232    98.232        1      531      0.43
532  joint                   0.0345    98.228    98.232        1      532      0.43
533  joint                   0.0344    98.228    98.232        1      533      0.43
534  joint                   0.0342    98.236    98.236        1      534      0.44
535  joint                   0.0340    98.257    98.257        1      535      0.43
536  joint                   0.0339    98.256    98.257        1      536      0.43
537  joint                   0.0338    98.257    98.257        1      537      0.44
538  joint                   0.0336    98.259    98.259        1      538      0.43
539  joint                   0.0335    98.247    98.259        1      539      0.43
540  joint                   0.0333    98.254    98.259        1      540      0.43
541  joint                   0.0332    98.262    98.262        1      541      0.42
542  joint                   0.0331    98.265    98.265        1      542      0.42
543  joint                   0.0329    98.273    98.273        1      543      0.43
544  joint                   0.0328    98.279    98.279        1      544      0.43
545  joint                   0.0327    98.276    98.279        1      545      0.43
546  joint                   0.0325    98.276    98.279        1      546      0.43
547  joint                   0.0324    98.273    98.279        1      547      0.43
548  joint                   0.0323    98.277    98.279        1      548      0.42
549  joint                   0.0322    98.282    98.282        1      549      0.43
550  joint                   0.0320    98.289    98.289        1      550      0.44
551  joint                   0.0318    98.275    98.289        1      551      0.43
552  joint                   0.0317    98.284    98.289        1      552      0.43
553  joint                   0.0316    98.309    98.309        1      553      0.44
554  joint                   0.0315    98.306    98.309        1      554      0.43
555  joint                   0.0314    98.289    98.309        1      555      0.44
556  joint                   0.0313    98.306    98.309        1      556      0.44
557  joint                   0.0312    98.311    98.311        1      557      0.43
558  joint                   0.0309    98.303    98.311        1      558      0.44
559  joint                   0.0309    98.318    98.318        1      559      0.43
560  joint                   0.0306    98.322    98.322        1      560      0.43
561  joint                   0.0307    98.314    98.322        1      561      0.43
562  joint                   0.0305    98.323    98.323        1      562      0.44
563  joint                   0.0303    98.329    98.329        1      563      0.44
564  joint                   0.0303    98.329    98.329        1      564      0.43
565  joint                   0.0301    98.331    98.331        1      565      0.43
566  joint                   0.0301    98.337    98.337        1      566      0.44
567  joint                   0.0299    98.329    98.337        1      567      0.43
568  joint                   0.0298    98.324    98.337        1      568      0.43
569  joint                   0.0297    98.337    98.337        1      569      0.44
570  joint                   0.0295    98.340    98.340        1      570      0.43
571  joint                   0.0295    98.322    98.340        1      571      0.43
572  joint                   0.0293    98.339    98.340        1      572      0.43
573  joint                   0.0293    98.340    98.340        1      573      0.44
574  joint                   0.0291    98.338    98.340        1      574      0.42
575  joint                   0.0291    98.355    98.355        1      575      0.43
576  joint                   0.0289    98.357    98.357        1      576      0.43
577  joint                   0.0287    98.349    98.357        1      577      0.43
578  joint                   0.0287    98.354    98.357        1      578      0.43
579  joint                   0.0284    98.366    98.366        1      579      0.43
580  joint                   0.0285    98.366    98.366        1      580      0.43
581  joint                   0.0283    98.370    98.370        1      581      0.44
582  joint                   0.0282    98.366    98.370        1      582      0.44
583  joint                   0.0282    98.374    98.374        1      583      0.42
584  joint                   0.0280    98.375    98.375        1      584      0.43
585  joint                   0.0279    98.381    98.381        1      585      0.43
586  joint                   0.0277    98.381    98.381        1      586      0.44
587  joint                   0.0278    98.377    98.381        1      587      0.44
588  joint                   0.0276    98.373    98.381        1      588      0.43
589  joint                   0.0275    98.380    98.381        1      589      0.42
590  joint                   0.0274    98.387    98.387        1      590      0.42
591  joint                   0.0273    98.382    98.387        1      591      0.42
592  joint                   0.0273    98.388    98.388        1      592      0.42
593  joint                   0.0271    98.384    98.388        1      593      0.44
594  joint                   0.0270    98.391    98.391        1      594      0.42
595  joint                   0.0269    98.406    98.406        1      595      0.43
596  joint                   0.0268    98.397    98.406        1      596      0.44
597  joint                   0.0267    98.392    98.406        1      597      0.42
598  joint                   0.0268    98.399    98.406        1      598      0.44
599  joint                   0.0265    98.408    98.408        1      599      0.42 *
600  joint                   0.0264    98.404    98.408        1      600      0.44
```

</details>

<details>
<summary>ppi | reference | shared_dynamic_c — 전체 에포크</summary>

```text
[ppi | reference | shared_dynamic_c]
status=passed | last_epoch=600 | selected_epoch=597
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   0.7406    41.897    41.897        1        1      5.82
  2  joint                   0.6560    41.776    41.897        1        2      4.53
  3  joint                   0.6132    41.104    41.897        1        3      4.65
  4  joint                   0.5883    41.234    41.897        1        4      4.68
  5  joint                   0.5740    40.301    41.897        1        5      4.62
  6  joint                   0.5652    39.650    41.897        1        6      4.62
  7  joint                   0.5595    39.524    41.897        1        7      4.72
  8  joint                   0.5558    39.925    41.897        1        8      4.60
  9  joint                   0.5534    40.572    41.897        1        9      4.70
 10  joint                   0.5518    40.699    41.897        1       10      4.55
 11  joint                   0.5508    40.382    41.897        1       11      4.66
 12  joint                   0.5503    40.104    41.897        1       12      4.60
 13  joint                   0.5498    39.936    41.897        1       13      4.63
 14  joint                   0.5495    39.940    41.897        1       14      4.60
 15  joint                   0.5492    40.117    41.897        1       15      4.59
 16  joint                   0.5489    40.366    41.897        1       16      4.70
 17  joint                   0.5484    40.664    41.897        1       17      4.72
 18  joint                   0.5477    41.174    41.897        1       18      4.72
 19  joint                   0.5468    41.823    41.897        1       19      4.56
 20  joint                   0.5457    42.442    42.442        1       20      4.55
 21  joint                   0.5444    43.046    43.046        1       21      4.77
 22  joint                   0.5432    43.390    43.390        1       22      4.59
 23  joint                   0.5417    43.631    43.631        1       23      4.65
 24  joint                   0.5399    44.836    44.836        1       24      4.74
 25  joint                   0.5380    45.590    45.590        1       25      4.68
 26  joint                   0.5360    46.105    46.105        1       26      4.68
 27  joint                   0.5342    47.856    47.856        1       27      4.67
 28  joint                   0.5327    41.577    47.856        1       28      4.52
 29  joint                   0.5361    52.830    52.830        1       29      4.67
 30  joint                   0.5391    41.022    52.830        1       30      4.65
 31  joint                   0.5364    40.095    52.830        1       31      4.58
 32  joint                   0.5397    42.628    52.830        1       32      4.58
 33  joint                   0.5346    46.498    52.830        1       33      4.58
 34  joint                   0.5309    49.595    52.830        1       34      4.70
 35  joint                   0.5333    50.471    52.830        1       35      4.60
 36  joint                   0.5342    49.118    52.830        1       36      4.74
 37  joint                   0.5305    46.080    52.830        1       37      4.54
 38  joint                   0.5280    42.790    52.830        1       38      4.69
 39  joint                   0.5286    41.298    52.830        1       39      4.67
 40  joint                   0.5287    42.567    52.830        1       40      4.65
 41  joint                   0.5264    45.897    52.830        1       41      4.63
 42  joint                   0.5240    48.829    52.830        1       42      4.67
 43  joint                   0.5243    50.060    52.830        1       43      4.60
 44  joint                   0.5229    49.301    52.830        1       44      4.67
 45  joint                   0.5200    47.651    52.830        1       45      4.67
 46  joint                   0.5202    47.795    52.830        1       46      4.65
 47  joint                   0.5189    50.139    52.830        1       47      4.67
 48  joint                   0.5159    52.115    52.830        1       48      4.64
 49  joint                   0.5167    50.496    52.830        1       49      4.71
 50  joint                   0.5132    48.068    52.830        1       50      4.73
 51  joint                   0.5137    50.750    52.830        1       51      4.73
 52  joint                   0.5098    53.814    53.814        1       52      4.61
 53  joint                   0.5113    48.953    53.814        1       53      4.71
 54  joint                   0.5090    50.837    53.814        1       54      4.70
 55  joint                   0.5049    53.977    53.977        1       55      4.62
 56  joint                   0.5087    46.057    53.977        1       56      4.66
 57  joint                   0.5100    51.727    53.977        1       57      4.67
 58  joint                   0.5012    55.555    55.555        1       58      4.64
 59  joint                   0.5077    50.966    55.555        1       59      4.68
 60  joint                   0.4996    48.232    55.555        1       60      4.65
 61  joint                   0.5026    54.031    55.555        1       61      4.60
 62  joint                   0.4965    55.774    55.774        1       62      4.56
 63  joint                   0.4989    51.297    55.774        1       63      4.59
 64  joint                   0.4953    51.118    55.774        1       64      4.68
 65  joint                   0.4940    55.562    55.774        1       65      4.64
 66  joint                   0.4934    55.024    55.774        1       66      4.68
 67  joint                   0.4901    51.768    55.774        1       67      4.69
 68  joint                   0.4895    54.470    55.774        1       68      4.70
 69  joint                   0.4865    56.354    56.354        1       69      4.72
 70  joint                   0.4871    53.824    56.354        1       70      4.56
 71  joint                   0.4831    55.785    56.354        1       71      4.56
 72  joint                   0.4809    56.005    56.354        1       72      4.70
 73  joint                   0.4818    54.727    56.354        1       73      4.59
 74  joint                   0.4781    56.737    56.737        1       74      4.69
 75  joint                   0.4760    53.386    56.737        1       75      4.60
 76  joint                   0.4799    58.034    58.034        1       76      4.70
 77  joint                   0.4852    48.572    58.034        1       77      4.77
 78  joint                   0.4962    55.692    58.034        1       78      4.63
 79  joint                   0.4821    57.751    58.034        1       79      4.53
 80  joint                   0.4822    56.418    58.034        1       80      4.61
 81  joint                   0.4773    54.469    58.034        1       81      4.69
 82  joint                   0.4799    54.877    58.034        1       82      4.69
 83  joint                   0.4733    55.990    58.034        1       83      4.65
 84  joint                   0.4730    57.319    58.034        1       84      4.71
 85  joint                   0.4722    57.270    58.034        1       85      4.57
 86  joint                   0.4677    56.113    58.034        1       86      4.78
 87  joint                   0.4673    56.102    58.034        1       87      4.65
 88  joint                   0.4651    57.561    58.034        1       88      4.61
 89  joint                   0.4629    57.898    58.034        1       89      4.63
 90  joint                   0.4608    58.052    58.052        1       90      4.53
 91  joint                   0.4587    58.940    58.940        1       91      4.64
 92  joint                   0.4573    58.641    58.940        1       92      4.66
 93  joint                   0.4547    58.590    58.940        1       93      4.68
 94  joint                   0.4519    60.224    60.224        1       94      4.66
 95  joint                   0.4515    60.074    60.224        1       95      4.70
 96  joint                   0.4482    59.384    60.224        1       96      4.55
 97  joint                   0.4469    60.081    60.224        1       97      4.69
 98  joint                   0.4445    61.757    61.757        1       98      4.54
 99  joint                   0.4422    60.976    61.757        1       99      4.66
100  joint                   0.4404    60.247    61.757        1      100      4.53
101  joint                   0.4414    61.464    61.757        1      101      4.54
102  joint                   0.4551    52.791    61.757        1      102      4.64
103  joint                   0.4783    60.348    61.757        1      103      4.65
104  joint                   0.4664    57.455    61.757        1      104      4.68
105  joint                   0.4528    57.990    61.757        1      105      4.72
106  joint                   0.4539    60.705    61.757        1      106      4.58
107  joint                   0.4528    59.955    61.757        1      107      4.70
108  joint                   0.4481    59.286    61.757        1      108      4.54
109  joint                   0.4458    59.882    61.757        1      109      4.69
110  joint                   0.4452    61.630    61.757        1      110      4.56
111  joint                   0.4390    61.948    61.948        1      111      4.66
112  joint                   0.4375    61.328    61.948        1      112      4.71
113  joint                   0.4344    60.464    61.948        1      113      4.60
114  joint                   0.4336    61.442    61.948        1      114      4.66
115  joint                   0.4298    62.485    62.485        1      115      4.63
116  joint                   0.4277    62.824    62.824        1      116      4.72
117  joint                   0.4243    62.945    62.945        1      117      4.58
118  joint                   0.4242    63.623    63.623        1      118      4.66
119  joint                   0.4204    63.947    63.947        1      119      4.59
120  joint                   0.4192    64.546    64.546        1      120      4.60
121  joint                   0.4159    64.258    64.546        1      121      4.56
122  joint                   0.4142    63.988    64.546        1      122      4.60
123  joint                   0.4127    65.667    65.667        1      123      4.65
124  joint                   0.4115    63.032    65.667        1      124      4.53
125  joint                   0.4178    63.064    65.667        1      125      4.69
126  joint                   0.4259    64.420    65.667        1      126      4.58
127  joint                   0.4059    65.258    65.667        1      127      4.60
128  joint                   0.4132    65.369    65.667        1      128      4.52
129  joint                   0.4048    65.251    65.667        1      129      4.52
130  joint                   0.4029    66.075    66.075        1      130      4.59
131  joint                   0.4036    65.566    66.075        1      131      4.60
132  joint                   0.4013    66.326    66.326        1      132      4.59
133  joint                   0.3968    67.107    67.107        1      133      4.56
134  joint                   0.3961    66.419    67.107        1      134      4.68
135  joint                   0.3948    67.448    67.448        1      135      4.62
136  joint                   0.3918    66.941    67.448        1      136      4.76
137  joint                   0.3892    67.669    67.669        1      137      4.67
138  joint                   0.3877    67.497    67.669        1      138      4.61
139  joint                   0.3914    67.092    67.669        1      139      4.69
140  joint                   0.3966    65.115    67.669        1      140      4.64
141  joint                   0.3990    68.717    68.717        1      141      4.56
142  joint                   0.3915    65.066    68.717        1      142      4.72
143  joint                   0.3942    67.511    68.717        1      143      4.64
144  joint                   0.3850    68.066    68.717        1      144      4.69
145  joint                   0.3838    68.284    68.717        1      145      4.74
146  joint                   0.3835    69.052    69.052        1      146      4.57
147  joint                   0.3766    67.743    69.052        1      147      4.62
148  joint                   0.3775    68.901    69.052        1      148      4.69
149  joint                   0.3749    69.810    69.810        1      149      4.58
150  joint                   0.3701    70.351    70.351        1      150      4.60
151  joint                   0.3692    70.849    70.849        1      151      4.66
152  joint                   0.3658    70.267    70.849        1      152      4.65
153  joint                   0.3639    70.706    70.849        1      153      4.68
154  joint                   0.3606    71.139    71.139        1      154      4.59
155  joint                   0.3589    71.998    71.998        1      155      4.69
156  joint                   0.3563    72.305    72.305        1      156      4.71
157  joint                   0.3532    71.525    72.305        1      157      4.53
158  joint                   0.3528    72.551    72.551        1      158      4.57
159  joint                   0.3511    70.742    72.551        1      159      4.61
160  joint                   0.3556    73.113    73.113        1      160      4.64
161  joint                   0.3594    68.642    73.113        1      161      4.65
162  joint                   0.3606    71.526    73.113        1      162      4.70
163  joint                   0.3637    71.995    73.113        1      163      4.59
164  joint                   0.3562    71.205    73.113        1      164      4.64
165  joint                   0.3542    72.072    73.113        1      165      4.67
166  joint                   0.3495    73.658    73.658        1      166      4.57
167  joint                   0.3434    73.085    73.658        1      167      4.57
168  joint                   0.3446    72.799    73.658        1      168      4.60
169  joint                   0.3394    73.036    73.658        1      169      4.62
170  joint                   0.3362    73.338    73.658        1      170      4.65
171  joint                   0.3364    74.433    74.433        1      171      4.60
172  joint                   0.3304    74.529    74.529        1      172      4.64
173  joint                   0.3297    74.375    74.529        1      173      4.66
174  joint                   0.3275    74.783    74.783        1      174      4.54
175  joint                   0.3239    75.376    75.376        1      175      4.61
176  joint                   0.3219    75.416    75.416        1      176      4.66
177  joint                   0.3192    75.664    75.664        1      177      4.44
178  joint                   0.3179    76.211    76.211        1      178      4.69
179  joint                   0.3145    76.497    76.497        1      179      4.64
180  joint                   0.3129    76.096    76.497        1      180      4.56
181  joint                   0.3123    76.155    76.497        1      181      4.64
182  joint                   0.3127    75.420    76.497        1      182      4.52
183  joint                   0.3196    76.919    76.919        1      183      4.62
184  joint                   0.3072    76.080    76.919        1      184      4.64
185  joint                   0.3096    77.178    77.178        1      185      4.64
186  joint                   0.3043    77.714    77.714        1      186      4.65
187  joint                   0.3031    77.430    77.714        1      187      4.60
188  joint                   0.2989    78.163    78.163        1      188      4.61
189  joint                   0.2961    78.507    78.507        1      189      4.72
190  joint                   0.2944    78.598    78.598        1      190      4.66
191  joint                   0.2917    79.116    79.116        1      191      4.62
192  joint                   0.2888    79.025    79.116        1      192      4.66
193  joint                   0.2871    79.391    79.391        1      193      4.63
194  joint                   0.2849    79.904    79.904        1      194      4.57
195  joint                   0.2818    80.010    80.010        1      195      4.62
196  joint                   0.2794    80.105    80.105        1      196      4.64
197  joint                   0.2773    80.503    80.503        1      197      4.73
198  joint                   0.2760    80.700    80.700        1      198      4.63
199  joint                   0.2737    80.502    80.700        1      199      4.56
200  joint                   0.2722    81.182    81.182        1      200      4.50
201  joint                   0.2706    80.233    81.182        1      201      4.60
202  joint                   0.2708    81.501    81.501        1      202      4.63
203  joint                   0.2691    81.401    81.501        1      203      4.56
204  joint                   0.2632    81.540    81.540        1      204      4.72
205  joint                   0.2630    82.316    82.316        1      205      4.60
206  joint                   0.2585    82.403    82.403        1      206      4.58
207  joint                   0.2575    82.074    82.403        1      207      4.66
208  joint                   0.2544    82.983    82.983        1      208      4.54
209  joint                   0.2550    81.260    82.983        1      209      4.72
210  joint                   0.2579    82.777    82.983        1      210      4.54
211  joint                   0.2591    83.053    83.053        1      211      4.65
212  joint                   0.2490    82.546    83.053        1      212      4.57
213  joint                   0.2505    83.137    83.137        1      213      4.60
214  joint                   0.2462    83.782    83.782        1      214      4.66
215  joint                   0.2432    83.744    83.782        1      215      4.67
216  joint                   0.2435    84.066    84.066        1      216      4.66
217  joint                   0.2416    83.806    84.066        1      217      4.63
218  joint                   0.2408    83.777    84.066        1      218      4.55
219  joint                   0.2413    84.352    84.352        1      219      4.54
220  joint                   0.2370    83.784    84.352        1      220      4.66
221  joint                   0.2366    84.496    84.496        1      221      4.66
222  joint                   0.2381    83.749    84.496        1      222      4.61
223  joint                   0.2385    84.517    84.517        1      223      4.69
224  joint                   0.2324    85.067    85.067        1      224      4.56
225  joint                   0.2274    85.469    85.469        1      225      4.66
226  joint                   0.2255    85.131    85.469        1      226      4.68
227  joint                   0.2235    85.867    85.867        1      227      4.64
228  joint                   0.2220    85.816    85.867        1      228      4.61
229  joint                   0.2188    85.948    85.948        1      229      4.69
230  joint                   0.2171    86.178    86.178        1      230      4.73
231  joint                   0.2149    86.567    86.567        1      231      4.60
232  joint                   0.2123    86.651    86.651        1      232      4.66
233  joint                   0.2103    86.809    86.809        1      233      4.53
234  joint                   0.2082    87.150    87.150        1      234      4.63
235  joint                   0.2064    87.352    87.352        1      235      4.64
236  joint                   0.2039    87.294    87.352        1      236      4.63
237  joint                   0.2029    87.494    87.494        1      237      4.68
238  joint                   0.2004    87.832    87.832        1      238      4.66
239  joint                   0.1989    87.981    87.981        1      239      4.62
240  joint                   0.1968    88.001    88.001        1      240      4.66
241  joint                   0.1953    88.393    88.393        1      241      4.61
242  joint                   0.1939    88.309    88.393        1      242      4.72
243  joint                   0.1923    88.473    88.473        1      243      4.71
244  joint                   0.1904    88.707    88.707        1      244      4.67
245  joint                   0.1885    88.925    88.925        1      245      4.58
246  joint                   0.1865    89.011    89.011        1      246      4.73
247  joint                   0.1850    89.096    89.096        1      247      4.65
248  joint                   0.1830    89.342    89.342        1      248      4.64
249  joint                   0.1817    89.452    89.452        1      249      4.52
250  joint                   0.1797    89.575    89.575        1      250      4.61
251  joint                   0.1782    89.742    89.742        1      251      4.55
252  joint                   0.1766    89.862    89.862        1      252      4.72
253  joint                   0.1753    89.953    89.953        1      253      4.66
254  joint                   0.1736    90.095    90.095        1      254      4.72
255  joint                   0.1721    90.167    90.167        1      255      4.62
256  joint                   0.1706    90.489    90.489        1      256      4.64
257  joint                   0.1694    90.122    90.489        1      257      4.57
258  joint                   0.1685    90.746    90.746        1      258      4.60
259  joint                   0.1678    90.383    90.746        1      259      4.64
260  joint                   0.1666    90.742    90.746        1      260      4.69
261  joint                   0.1640    91.021    91.021        1      261      4.60
262  joint                   0.1630    90.804    91.021        1      262      4.56
263  joint                   0.1622    91.105    91.105        1      263      4.59
264  joint                   0.1602    91.292    91.292        1      264      4.64
265  joint                   0.1589    91.407    91.407        1      265      4.55
266  joint                   0.1566    91.299    91.407        1      266      4.72
267  joint                   0.1563    91.641    91.641        1      267      4.66
268  joint                   0.1549    91.741    91.741        1      268      4.64
269  joint                   0.1537    91.706    91.741        1      269      4.58
270  joint                   0.1514    91.896    91.896        1      270      4.61
271  joint                   0.1504    92.120    92.120        1      271      4.86
272  joint                   0.1490    92.105    92.120        1      272      4.62
273  joint                   0.1481    92.293    92.293        1      273      4.62
274  joint                   0.1463    92.235    92.293        1      274      4.60
275  joint                   0.1455    92.361    92.361        1      275      4.63
276  joint                   0.1439    92.615    92.615        1      276      4.60
277  joint                   0.1429    92.635    92.635        1      277      4.67
278  joint                   0.1415    92.624    92.635        1      278      4.59
279  joint                   0.1406    92.708    92.708        1      279      4.69
280  joint                   0.1398    92.771    92.771        1      280      4.60
281  joint                   0.1392    92.912    92.912        1      281      4.53
282  joint                   0.1370    93.000    93.000        1      282      4.53
283  joint                   0.1356    93.105    93.105        1      283      4.70
284  joint                   0.1349    93.271    93.271        1      284      4.51
285  joint                   0.1333    93.255    93.271        1      285      4.65
286  joint                   0.1326    93.286    93.286        1      286      4.61
287  joint                   0.1310    93.489    93.489        1      287      4.63
288  joint                   0.1303    93.552    93.552        1      288      4.73
289  joint                   0.1289    93.560    93.560        1      289      4.64
290  joint                   0.1283    93.734    93.734        1      290      4.58
291  joint                   0.1271    93.777    93.777        1      291      4.63
292  joint                   0.1260    93.856    93.856        1      292      4.62
293  joint                   0.1246    93.970    93.970        1      293      4.53
294  joint                   0.1236    93.996    93.996        1      294      4.56
295  joint                   0.1227    94.137    94.137        1      295      4.71
296  joint                   0.1216    94.176    94.176        1      296      4.71
297  joint                   0.1209    94.212    94.212        1      297      4.61
298  joint                   0.1197    94.343    94.343        1      298      4.59
299  joint                   0.1186    94.367    94.367        1      299      4.56
300  joint                   0.1175    94.462    94.462        1      300      4.67
301  joint                   0.1166    94.476    94.476        1      301      4.58
302  joint                   0.1158    94.549    94.549        1      302      4.70
303  joint                   0.1148    94.588    94.588        1      303      4.69
304  joint                   0.1139    94.669    94.669        1      304      4.61
305  joint                   0.1130    94.760    94.760        1      305      4.66
306  joint                   0.1121    94.775    94.775        1      306      4.65
307  joint                   0.1112    94.830    94.830        1      307      4.68
308  joint                   0.1102    94.967    94.967        1      308      4.73
309  joint                   0.1094    94.948    94.967        1      309      4.60
310  joint                   0.1085    95.037    95.037        1      310      4.57
311  joint                   0.1077    95.060    95.060        1      311      4.61
312  joint                   0.1073    95.139    95.139        1      312      4.60
313  joint                   0.1063    95.186    95.186        1      313      4.58
314  joint                   0.1054    95.118    95.186        1      314      4.58
315  joint                   0.1050    95.313    95.313        1      315      4.62
316  joint                   0.1048    95.028    95.313        1      316      4.71
317  joint                   0.1043    95.400    95.400        1      317      4.62
318  joint                   0.1033    95.372    95.400        1      318      4.57
319  joint                   0.1016    95.309    95.400        1      319      4.55
320  joint                   0.1016    95.451    95.451        1      320      4.68
321  joint                   0.1010    95.614    95.614        1      321      4.64
322  joint                   0.0992    95.480    95.614        1      322      4.62
323  joint                   0.0988    95.624    95.624        1      323      4.58
324  joint                   0.0976    95.737    95.737        1      324      4.55
325  joint                   0.0977    95.616    95.737        1      325      4.65
326  joint                   0.0970    95.787    95.787        1      326      4.67
327  joint                   0.0956    95.846    95.846        1      327      4.56
328  joint                   0.0948    95.836    95.846        1      328      4.68
329  joint                   0.0944    95.954    95.954        1      329      4.59
330  joint                   0.0935    95.921    95.954        1      330      4.65
331  joint                   0.0925    95.978    95.978        1      331      4.70
332  joint                   0.0917    96.072    96.072        1      332      4.57
333  joint                   0.0913    96.093    96.093        1      333      4.62
334  joint                   0.0905    96.127    96.127        1      334      4.56
335  joint                   0.0898    96.195    96.195        1      335      4.64
336  joint                   0.0890    96.251    96.251        1      336      4.68
337  joint                   0.0882    96.287    96.287        1      337      4.61
338  joint                   0.0876    96.285    96.287        1      338      4.67
339  joint                   0.0873    96.318    96.318        1      339      4.53
340  joint                   0.0865    96.406    96.406        1      340      4.64
341  joint                   0.0855    96.413    96.413        1      341      4.73
342  joint                   0.0850    96.460    96.460        1      342      4.67
343  joint                   0.0844    96.502    96.502        1      343      4.58
344  joint                   0.0838    96.501    96.502        1      344      4.56
345  joint                   0.0833    96.543    96.543        1      345      4.52
346  joint                   0.0829    96.572    96.572        1      346      4.55
347  joint                   0.0823    96.619    96.619        1      347      4.56
348  joint                   0.0815    96.639    96.639        1      348      4.66
349  joint                   0.0808    96.685    96.685        1      349      4.69
350  joint                   0.0800    96.741    96.741        1      350      4.71
351  joint                   0.0796    96.756    96.756        1      351      4.47
352  joint                   0.0790    96.790    96.790        1      352      4.63
353  joint                   0.0786    96.800    96.800        1      353      4.57
354  joint                   0.0780    96.806    96.806        1      354      4.70
355  joint                   0.0776    96.849    96.849        1      355      4.69
356  joint                   0.0769    96.892    96.892        1      356      4.70
357  joint                   0.0764    96.906    96.906        1      357      4.60
358  joint                   0.0762    96.940    96.940        1      358      4.60
359  joint                   0.0753    96.978    96.978        1      359      4.59
360  joint                   0.0747    96.986    96.986        1      360      4.57
361  joint                   0.0741    97.015    97.015        1      361      4.61
362  joint                   0.0738    97.049    97.049        1      362      4.62
363  joint                   0.0733    97.029    97.049        1      363      4.71
364  joint                   0.0728    97.119    97.119        1      364      4.65
365  joint                   0.0722    97.118    97.119        1      365      4.61
366  joint                   0.0717    97.106    97.119        1      366      4.74
367  joint                   0.0716    97.159    97.159        1      367      4.65
368  joint                   0.0707    97.150    97.159        1      368      4.66
369  joint                   0.0703    97.216    97.216        1      369      4.61
370  joint                   0.0697    97.221    97.221        1      370      4.55
371  joint                   0.0693    97.219    97.221        1      371      4.58
372  joint                   0.0692    97.274    97.274        1      372      4.67
373  joint                   0.0686    97.270    97.274        1      373      4.66
374  joint                   0.0679    97.297    97.297        1      374      4.59
375  joint                   0.0676    97.342    97.342        1      375      4.55
376  joint                   0.0671    97.348    97.348        1      376      4.56
377  joint                   0.0665    97.368    97.368        1      377      4.69
378  joint                   0.0663    97.375    97.375        1      378      4.59
379  joint                   0.0657    97.412    97.412        1      379      4.69
380  joint                   0.0653    97.438    97.438        1      380      4.71
381  joint                   0.0650    97.421    97.438        1      381      4.70
382  joint                   0.0645    97.473    97.473        1      382      4.62
383  joint                   0.0641    97.470    97.473        1      383      4.61
384  joint                   0.0636    97.450    97.473        1      384      4.71
385  joint                   0.0634    97.487    97.487        1      385      4.66
386  joint                   0.0629    97.497    97.497        1      386      4.79
387  joint                   0.0623    97.513    97.513        1      387      4.62
388  joint                   0.0620    97.541    97.541        1      388      4.62
389  joint                   0.0617    97.542    97.542        1      389      4.68
390  joint                   0.0614    97.562    97.562        1      390      4.61
391  joint                   0.0609    97.569    97.569        1      391      4.67
392  joint                   0.0604    97.579    97.579        1      392      4.54
393  joint                   0.0601    97.599    97.599        1      393      4.67
394  joint                   0.0599    97.590    97.599        1      394      4.61
395  joint                   0.0593    97.635    97.635        1      395      4.56
396  joint                   0.0590    97.644    97.644        1      396      4.56
397  joint                   0.0586    97.669    97.669        1      397      4.59
398  joint                   0.0582    97.685    97.685        1      398      4.74
399  joint                   0.0580    97.703    97.703        1      399      4.64
400  joint                   0.0575    97.705    97.705        1      400      4.56
401  joint                   0.0572    97.715    97.715        1      401      4.73
402  joint                   0.0570    97.722    97.722        1      402      4.65
403  joint                   0.0565    97.729    97.729        1      403      4.53
404  joint                   0.0562    97.758    97.758        1      404      4.73
405  joint                   0.0558    97.765    97.765        1      405      4.64
406  joint                   0.0556    97.778    97.778        1      406      4.70
407  joint                   0.0552    97.810    97.810        1      407      4.60
408  joint                   0.0550    97.744    97.810        1      408      4.59
409  joint                   0.0548    97.810    97.810        1      409      4.65
410  joint                   0.0545    97.818    97.818        1      410      4.69
411  joint                   0.0540    97.818    97.818        1      411      4.50
412  joint                   0.0536    97.825    97.825        1      412      4.60
413  joint                   0.0535    97.828    97.828        1      413      4.52
414  joint                   0.0531    97.851    97.851        1      414      4.62
415  joint                   0.0530    97.825    97.851        1      415      4.66
416  joint                   0.0527    97.869    97.869        1      416      4.63
417  joint                   0.0520    97.888    97.888        1      417      4.71
418  joint                   0.0519    97.861    97.888        1      418      4.60
419  joint                   0.0518    97.922    97.922        1      419      4.56
420  joint                   0.0513    97.928    97.928        1      420      4.61
421  joint                   0.0511    97.917    97.928        1      421      4.56
422  joint                   0.0508    97.946    97.946        1      422      4.59
423  joint                   0.0506    97.929    97.946        1      423      4.63
424  joint                   0.0501    97.941    97.946        1      424      4.68
425  joint                   0.0498    97.984    97.984        1      425      4.53
426  joint                   0.0496    97.949    97.984        1      426      4.63
427  joint                   0.0491    97.963    97.984        1      427      4.64
428  joint                   0.0490    97.993    97.993        1      428      4.55
429  joint                   0.0487    97.997    97.997        1      429      4.60
430  joint                   0.0485    98.006    98.006        1      430      4.73
431  joint                   0.0481    98.015    98.015        1      431      4.67
432  joint                   0.0479    98.025    98.025        1      432      4.71
433  joint                   0.0477    98.033    98.033        1      433      4.77
434  joint                   0.0474    98.040    98.040        1      434      4.66
435  joint                   0.0470    98.059    98.059        1      435      4.69
436  joint                   0.0469    98.062    98.062        1      436      4.69
437  joint                   0.0466    98.069    98.069        1      437      4.68
438  joint                   0.0463    98.078    98.078        1      438      4.59
439  joint                   0.0461    98.085    98.085        1      439      4.69
440  joint                   0.0458    98.076    98.085        1      440      4.63
441  joint                   0.0456    98.086    98.086        1      441      4.66
442  joint                   0.0455    98.096    98.096        1      442      4.64
443  joint                   0.0451    98.090    98.096        1      443      4.66
444  joint                   0.0449    98.080    98.096        1      444      4.58
445  joint                   0.0448    98.120    98.120        1      445      4.63
446  joint                   0.0444    98.113    98.120        1      446      4.63
447  joint                   0.0443    98.119    98.120        1      447      4.71
448  joint                   0.0439    98.132    98.132        1      448      4.69
449  joint                   0.0439    98.130    98.132        1      449      4.68
450  joint                   0.0435    98.139    98.139        1      450      4.77
451  joint                   0.0432    98.160    98.160        1      451      4.62
452  joint                   0.0430    98.153    98.160        1      452      4.71
453  joint                   0.0427    98.154    98.160        1      453      4.59
454  joint                   0.0426    98.147    98.160        1      454      4.68
455  joint                   0.0425    98.130    98.160        1      455      4.50
456  joint                   0.0421    98.147    98.160        1      456      4.67
457  joint                   0.0421    98.148    98.160        1      457      4.55
458  joint                   0.0418    98.163    98.163        1      458      4.52
459  joint                   0.0415    98.168    98.168        1      459      4.63
460  joint                   0.0413    98.178    98.178        1      460      4.73
461  joint                   0.0412    98.188    98.188        1      461      4.50
462  joint                   0.0409    98.202    98.202        1      462      4.63
463  joint                   0.0407    98.187    98.202        1      463      4.62
464  joint                   0.0406    98.207    98.207        1      464      4.59
465  joint                   0.0404    98.186    98.207        1      465      4.58
466  joint                   0.0401    98.199    98.207        1      466      4.66
467  joint                   0.0400    98.210    98.210        1      467      4.66
468  joint                   0.0397    98.199    98.210        1      468      4.71
469  joint                   0.0396    98.205    98.210        1      469      4.75
470  joint                   0.0393    98.205    98.210        1      470      4.58
471  joint                   0.0392    98.204    98.210        1      471      4.51
472  joint                   0.0390    98.215    98.215        1      472      4.58
473  joint                   0.0388    98.219    98.219        1      473      4.67
474  joint                   0.0386    98.221    98.221        1      474      4.66
475  joint                   0.0384    98.241    98.241        1      475      4.65
476  joint                   0.0383    98.247    98.247        1      476      4.65
477  joint                   0.0380    98.243    98.247        1      477      4.72
478  joint                   0.0379    98.251    98.251        1      478      4.66
479  joint                   0.0378    98.240    98.251        1      479      4.58
480  joint                   0.0376    98.257    98.257        1      480      4.69
481  joint                   0.0374    98.263    98.263        1      481      4.50
482  joint                   0.0371    98.260    98.263        1      482      4.60
483  joint                   0.0370    98.270    98.270        1      483      4.70
484  joint                   0.0368    98.271    98.271        1      484      4.55
485  joint                   0.0367    98.268    98.271        1      485      4.66
486  joint                   0.0365    98.279    98.279        1      486      4.67
487  joint                   0.0362    98.292    98.292        1      487      4.65
488  joint                   0.0361    98.298    98.298        1      488      4.69
489  joint                   0.0360    98.294    98.298        1      489      4.58
490  joint                   0.0359    98.297    98.298        1      490      4.57
491  joint                   0.0356    98.307    98.307        1      491      4.59
492  joint                   0.0354    98.303    98.307        1      492      4.59
493  joint                   0.0353    98.319    98.319        1      493      4.57
494  joint                   0.0352    98.323    98.323        1      494      4.56
495  joint                   0.0350    98.322    98.323        1      495      4.65
496  joint                   0.0348    98.339    98.339        1      496      4.62
497  joint                   0.0346    98.331    98.339        1      497      4.56
498  joint                   0.0344    98.340    98.340        1      498      4.63
499  joint                   0.0344    98.342    98.342        1      499      4.67
500  joint                   0.0341    98.340    98.342        1      500      4.60
501  joint                   0.0340    98.352    98.352        1      501      4.66
502  joint                   0.0339    98.349    98.352        1      502      4.58
503  joint                   0.0337    98.351    98.352        1      503      4.58
504  joint                   0.0335    98.366    98.366        1      504      4.67
505  joint                   0.0334    98.352    98.366        1      505      4.57
506  joint                   0.0333    98.353    98.366        1      506      4.58
507  joint                   0.0331    98.355    98.366        1      507      4.56
508  joint                   0.0329    98.360    98.366        1      508      4.41
509  joint                   0.0329    98.356    98.366        1      509      4.69
510  joint                   0.0328    98.364    98.366        1      510      4.65
511  joint                   0.0326    98.365    98.366        1      511      4.65
512  joint                   0.0324    98.371    98.371        1      512      4.62
513  joint                   0.0323    98.372    98.372        1      513      4.55
514  joint                   0.0322    98.364    98.372        1      514      4.61
515  joint                   0.0320    98.363    98.372        1      515      4.61
516  joint                   0.0319    98.379    98.379        1      516      4.65
517  joint                   0.0317    98.376    98.379        1      517      4.73
518  joint                   0.0316    98.369    98.379        1      518      4.65
519  joint                   0.0315    98.398    98.398        1      519      4.56
520  joint                   0.0314    98.388    98.398        1      520      4.60
521  joint                   0.0312    98.370    98.398        1      521      4.54
522  joint                   0.0310    98.387    98.398        1      522      4.58
523  joint                   0.0309    98.385    98.398        1      523      4.59
524  joint                   0.0308    98.395    98.398        1      524      4.60
525  joint                   0.0307    98.380    98.398        1      525      4.57
526  joint                   0.0306    98.401    98.401        1      526      4.66
527  joint                   0.0304    98.412    98.412        1      527      4.66
528  joint                   0.0303    98.399    98.412        1      528      4.71
529  joint                   0.0303    98.408    98.412        1      529      4.67
530  joint                   0.0300    98.420    98.420        1      530      4.63
531  joint                   0.0300    98.415    98.420        1      531      4.80
532  joint                   0.0298    98.404    98.420        1      532      4.60
533  joint                   0.0298    98.415    98.420        1      533      4.58
534  joint                   0.0296    98.426    98.426        1      534      4.70
535  joint                   0.0294    98.426    98.426        1      535      4.58
536  joint                   0.0293    98.422    98.426        1      536      4.55
537  joint                   0.0291    98.436    98.436        1      537      4.75
538  joint                   0.0290    98.432    98.436        1      538      4.64
539  joint                   0.0289    98.419    98.436        1      539      4.73
540  joint                   0.0288    98.429    98.436        1      540      4.59
541  joint                   0.0287    98.436    98.436        1      541      4.71
542  joint                   0.0287    98.426    98.436        1      542      4.65
543  joint                   0.0284    98.429    98.436        1      543      4.58
544  joint                   0.0282    98.429    98.436        1      544      4.66
545  joint                   0.0283    98.430    98.436        1      545      4.60
546  joint                   0.0281    98.432    98.436        1      546      4.61
547  joint                   0.0280    98.442    98.442        1      547      4.65
548  joint                   0.0279    98.426    98.442        1      548      4.58
549  joint                   0.0278    98.442    98.442        1      549      4.61
550  joint                   0.0276    98.446    98.446        1      550      4.75
551  joint                   0.0276    98.446    98.446        1      551      4.62
552  joint                   0.0274    98.458    98.458        1      552      4.63
553  joint                   0.0273    98.457    98.458        1      553      4.69
554  joint                   0.0273    98.449    98.458        1      554      4.71
555  joint                   0.0270    98.462    98.462        1      555      4.55
556  joint                   0.0270    98.469    98.469        1      556      4.53
557  joint                   0.0270    98.464    98.469        1      557      4.72
558  joint                   0.0268    98.479    98.479        1      558      4.66
559  joint                   0.0267    98.479    98.479        1      559      4.53
560  joint                   0.0266    98.470    98.479        1      560      4.67
561  joint                   0.0265    98.469    98.479        1      561      4.58
562  joint                   0.0264    98.470    98.479        1      562      4.58
563  joint                   0.0263    98.472    98.479        1      563      4.63
564  joint                   0.0262    98.467    98.479        1      564      4.61
565  joint                   0.0261    98.472    98.479        1      565      4.72
566  joint                   0.0261    98.477    98.479        1      566      4.56
567  joint                   0.0259    98.479    98.479        1      567      4.64
568  joint                   0.0258    98.480    98.480        1      568      4.64
569  joint                   0.0256    98.482    98.482        1      569      4.64
570  joint                   0.0256    98.489    98.489        1      570      4.66
571  joint                   0.0256    98.485    98.489        1      571      4.71
572  joint                   0.0253    98.487    98.489        1      572      4.64
573  joint                   0.0254    98.493    98.493        1      573      4.62
574  joint                   0.0251    98.509    98.509        1      574      4.60
575  joint                   0.0251    98.494    98.509        1      575      4.53
576  joint                   0.0251    98.498    98.509        1      576      4.62
577  joint                   0.0249    98.508    98.509        1      577      4.69
578  joint                   0.0249    98.497    98.509        1      578      4.69
579  joint                   0.0248    98.502    98.509        1      579      4.74
580  joint                   0.0247    98.513    98.513        1      580      4.55
581  joint                   0.0246    98.507    98.513        1      581      4.58
582  joint                   0.0246    98.509    98.513        1      582      4.66
583  joint                   0.0244    98.513    98.513        1      583      4.55
584  joint                   0.0244    98.501    98.513        1      584      4.54
585  joint                   0.0242    98.509    98.513        1      585      4.68
586  joint                   0.0241    98.507    98.513        1      586      4.71
587  joint                   0.0241    98.504    98.513        1      587      4.61
588  joint                   0.0240    98.518    98.518        1      588      4.72
589  joint                   0.0238    98.511    98.518        1      589      4.73
590  joint                   0.0238    98.511    98.518        1      590      4.76
591  joint                   0.0237    98.518    98.518        1      591      4.65
592  joint                   0.0237    98.514    98.518        1      592      4.75
593  joint                   0.0236    98.514    98.518        1      593      4.62
594  joint                   0.0236    98.523    98.523        1      594      4.61
595  joint                   0.0234    98.524    98.524        1      595      4.59
596  joint                   0.0232    98.531    98.531        1      596      4.62
597  joint                   0.0232    98.531    98.531        1      597      4.59 *
598  joint                   0.0232    98.525    98.531        1      598      4.61
599  joint                   0.0231    98.507    98.531        1      599      4.70
600  joint                   0.0231    98.526    98.531        1      600      4.66
```

</details>

<details>
<summary>ppi | large | fixed_c — 전체 에포크</summary>

```text
[ppi | large | fixed_c]
status=passed | last_epoch=600 | selected_epoch=586
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   0.7392    44.803    44.803        1        1      1.46
  2  joint                   0.6291    42.273    44.803        1        2      0.99
  3  joint                   0.5805    41.505    44.803        1        3      0.84
  4  joint                   0.5629    40.264    44.803        1        4      0.85
  5  joint                   0.5557    39.263    44.803        1        5      0.85
  6  joint                   0.5526    39.358    44.803        1        6      0.85
  7  joint                   0.5512    39.370    44.803        1        7      0.85
  8  joint                   0.5506    40.054    44.803        1        8      0.85
  9  joint                   0.5503    40.012    44.803        1        9      0.85
 10  joint                   0.5501    39.784    44.803        1       10      0.84
 11  joint                   0.5499    40.185    44.803        1       11      0.85
 12  joint                   0.5494    40.204    44.803        1       12      0.85
 13  joint                   0.5486    39.986    44.803        1       13      0.85
 14  joint                   0.5475    40.423    44.803        1       14      0.85
 15  joint                   0.5460    41.448    44.803        1       15      0.85
 16  joint                   0.5445    42.529    44.803        1       16      0.85
 17  joint                   0.5431    43.352    44.803        1       17      0.85
 18  joint                   0.5414    43.718    44.803        1       18      0.85
 19  joint                   0.5393    45.953    45.953        1       19      0.85
 20  joint                   0.5373    44.247    45.953        1       20      0.85
 21  joint                   0.5356    50.153    50.153        1       21      0.85
 22  joint                   0.5362    40.158    50.153        1       22      0.85
 23  joint                   0.5406    40.969    50.153        1       23      0.85
 24  joint                   0.5385    45.593    50.153        1       24      0.85
 25  joint                   0.5330    50.453    50.453        1       25      0.85
 26  joint                   0.5371    49.442    50.453        1       26      0.85
 27  joint                   0.5341    44.702    50.453        1       27      0.85
 28  joint                   0.5308    41.139    50.453        1       28      0.85
 29  joint                   0.5327    41.406    50.453        1       29      0.85
 30  joint                   0.5314    45.403    50.453        1       30      0.85
 31  joint                   0.5275    49.721    50.453        1       31      0.85
 32  joint                   0.5295    48.895    50.453        1       32      0.85
 33  joint                   0.5269    45.180    50.453        1       33      0.85
 34  joint                   0.5254    43.884    50.453        1       34      0.85
 35  joint                   0.5264    46.501    50.453        1       35      0.85
 36  joint                   0.5226    50.995    50.995        1       36      0.83
 37  joint                   0.5238    48.688    50.995        1       37      0.86
 38  joint                   0.5195    46.335    50.995        1       38      0.86
 39  joint                   0.5198    49.987    50.995        1       39      0.85
 40  joint                   0.5190    45.497    50.995        1       40      0.83
 41  joint                   0.5208    50.211    50.995        1       41      0.85
 42  joint                   0.5158    52.769    52.769        1       42      0.85
 43  joint                   0.5191    43.415    52.769        1       43      0.85
 44  joint                   0.5184    46.025    52.769        1       44      0.85
 45  joint                   0.5141    53.679    53.679        1       45      0.85
 46  joint                   0.5149    52.241    53.679        1       46      0.85
 47  joint                   0.5104    45.831    53.679        1       47      0.85
 48  joint                   0.5103    48.566    53.679        1       48      0.85
 49  joint                   0.5075    51.501    53.679        1       49      0.86
 50  joint                   0.5041    46.666    53.679        1       50      0.84
 51  joint                   0.5123    55.364    55.364        1       51      0.85
 52  joint                   0.5166    44.790    55.364        1       52      0.85
 53  joint                   0.5130    45.799    55.364        1       53      0.85
 54  joint                   0.5105    51.223    55.364        1       54      0.85
 55  joint                   0.5059    53.393    55.364        1       55      0.85
 56  joint                   0.5111    52.883    55.364        1       56      0.85
 57  joint                   0.5060    49.528    55.364        1       57      0.85
 58  joint                   0.5012    45.679    55.364        1       58      0.86
 59  joint                   0.5038    47.069    55.364        1       59      0.85
 60  joint                   0.5006    52.519    55.364        1       60      0.85
 61  joint                   0.4960    55.009    55.364        1       61      0.85
 62  joint                   0.4971    53.738    55.364        1       62      0.85
 63  joint                   0.4906    50.394    55.364        1       63      0.84
 64  joint                   0.4908    52.284    55.364        1       64      0.85
 65  joint                   0.4868    54.354    55.364        1       65      0.85
 66  joint                   0.4935    54.126    55.364        1       66      0.83
 67  joint                   0.5219    49.014    55.364        1       67      0.85
 68  joint                   0.5047    50.952    55.364        1       68      0.82
 69  joint                   0.4900    54.848    55.364        1       69      0.85
 70  joint                   0.5061    49.695    55.364        1       70      0.83
 71  joint                   0.5004    52.928    55.364        1       71      0.84
 72  joint                   0.4914    55.349    55.364        1       72      0.85
 73  joint                   0.4901    52.523    55.364        1       73      0.85
 74  joint                   0.4872    48.563    55.364        1       74      0.85
 75  joint                   0.4885    52.260    55.364        1       75      0.85
 76  joint                   0.4841    53.877    55.364        1       76      0.85
 77  joint                   0.4847    51.816    55.364        1       77      0.85
 78  joint                   0.4818    51.611    55.364        1       78      0.85
 79  joint                   0.4800    55.281    55.364        1       79      0.85
 80  joint                   0.4792    52.791    55.364        1       80      0.82
 81  joint                   0.4758    55.196    55.364        1       81      0.84
 82  joint                   0.4728    55.503    55.503        1       82      0.83
 83  joint                   0.4720    54.200    55.503        1       83      0.85
 84  joint                   0.4710    57.760    57.760        1       84      0.85
 85  joint                   0.4720    52.360    57.760        1       85      0.85
 86  joint                   0.4790    58.733    58.733        1       86      0.85
 87  joint                   0.4806    55.364    58.733        1       87      0.85
 88  joint                   0.4666    52.308    58.733        1       88      0.85
 89  joint                   0.4713    55.720    58.733        1       89      0.85
 90  joint                   0.4629    57.667    58.733        1       90      0.85
 91  joint                   0.4652    56.825    58.733        1       91      0.86
 92  joint                   0.4596    55.005    58.733        1       92      0.87
 93  joint                   0.4583    57.387    58.733        1       93      0.88
 94  joint                   0.4574    57.678    58.733        1       94      0.86
 95  joint                   0.4521    57.516    58.733        1       95      0.87
 96  joint                   0.4516    58.856    58.856        1       96      0.85
 97  joint                   0.4490    58.359    58.856        1       97      0.85
 98  joint                   0.4455    58.607    58.856        1       98      0.85
 99  joint                   0.4438    60.542    60.542        1       99      0.84
100  joint                   0.4432    58.865    60.542        1      100      0.85
101  joint                   0.4484    50.099    60.542        1      101      0.85
102  joint                   0.4840    60.037    60.542        1      102      0.84
103  joint                   0.4561    55.551    60.542        1      103      0.84
104  joint                   0.4626    60.095    60.542        1      104      0.85
105  joint                   0.4626    59.470    60.542        1      105      0.85
106  joint                   0.4543    57.939    60.542        1      106      0.84
107  joint                   0.4485    59.737    60.542        1      107      0.85
108  joint                   0.4450    61.192    61.192        1      108      0.85
109  joint                   0.4431    58.008    61.192        1      109      0.85
110  joint                   0.4431    58.979    61.192        1      110      0.83
111  joint                   0.4381    59.719    61.192        1      111      0.84
112  joint                   0.4378    60.108    61.192        1      112      0.84
113  joint                   0.4353    61.465    61.465        1      113      0.84
114  joint                   0.4325    60.579    61.465        1      114      0.85
115  joint                   0.4288    61.858    61.858        1      115      0.85
116  joint                   0.4261    63.459    63.459        1      116      0.85
117  joint                   0.4256    61.852    63.459        1      117      0.85
118  joint                   0.4248    62.539    63.459        1      118      0.85
119  joint                   0.4226    63.003    63.459        1      119      0.85
120  joint                   0.4179    63.077    63.459        1      120      0.85
121  joint                   0.4149    64.327    64.327        1      121      0.85
122  joint                   0.4144    60.215    64.327        1      122      0.85
123  joint                   0.4214    66.011    66.011        1      123      0.85
124  joint                   0.4183    63.142    66.011        1      124      0.82
125  joint                   0.4140    60.728    66.011        1      125      0.85
126  joint                   0.4169    64.755    66.011        1      126      0.85
127  joint                   0.4213    64.691    66.011        1      127      0.86
128  joint                   0.4139    64.473    66.011        1      128      0.85
129  joint                   0.4031    64.041    66.011        1      129      0.85
130  joint                   0.4044    64.418    66.011        1      130      0.85
131  joint                   0.4000    65.207    66.011        1      131      0.85
132  joint                   0.3965    66.583    66.583        1      132      0.85
133  joint                   0.3953    66.878    66.878        1      133      0.85
134  joint                   0.3907    66.018    66.878        1      134      0.85
135  joint                   0.3879    66.191    66.878        1      135      0.91
136  joint                   0.3854    67.928    67.928        1      136      0.85
137  joint                   0.3823    68.548    68.548        1      137      0.87
138  joint                   0.3796    68.111    68.548        1      138      0.85
139  joint                   0.3771    68.048    68.548        1      139      0.85
140  joint                   0.3816    65.378    68.548        1      140      0.85
141  joint                   0.4068    64.326    68.548        1      141      0.85
142  joint                   0.3918    67.810    68.548        1      142      0.84
143  joint                   0.4021    63.553    68.548        1      143      0.85
144  joint                   0.4123    65.963    68.548        1      144      0.85
145  joint                   0.3953    64.646    68.548        1      145      0.85
146  joint                   0.3940    65.832    68.548        1      146      0.85
147  joint                   0.3909    66.606    68.548        1      147      0.85
148  joint                   0.3852    67.995    68.548        1      148      0.85
149  joint                   0.3830    68.483    68.548        1      149      0.85
150  joint                   0.3798    67.345    68.548        1      150      0.85
151  joint                   0.3752    68.025    68.548        1      151      0.85
152  joint                   0.3726    68.675    68.675        1      152      0.85
153  joint                   0.3692    69.139    69.139        1      153      0.85
154  joint                   0.3658    70.537    70.537        1      154      0.85
155  joint                   0.3635    70.150    70.537        1      155      0.85
156  joint                   0.3592    70.702    70.702        1      156      0.85
157  joint                   0.3562    70.814    70.814        1      157      0.85
158  joint                   0.3530    71.677    71.677        1      158      0.85
159  joint                   0.3498    71.555    71.677        1      159      0.85
160  joint                   0.3471    72.338    72.338        1      160      0.85
161  joint                   0.3453    70.854    72.338        1      161      0.85
162  joint                   0.3465    71.773    72.338        1      162      0.84
163  joint                   0.3627    67.890    72.338        1      163      0.86
164  joint                   0.3685    68.574    72.338        1      164      0.85
165  joint                   0.3729    68.779    72.338        1      165      0.85
166  joint                   0.4001    67.383    72.338        1      166      0.85
167  joint                   0.3693    68.065    72.338        1      167      0.85
168  joint                   0.3676    70.238    72.338        1      168      0.85
169  joint                   0.3596    69.856    72.338        1      169      0.85
170  joint                   0.3563    69.980    72.338        1      170      0.85
171  joint                   0.3525    71.016    72.338        1      171      0.85
172  joint                   0.3485    71.871    72.338        1      172      0.85
173  joint                   0.3456    72.479    72.479        1      173      0.84
174  joint                   0.3424    72.593    72.593        1      174      0.85
175  joint                   0.3378    72.278    72.593        1      175      0.85
176  joint                   0.3348    72.647    72.647        1      176      0.85
177  joint                   0.3320    73.589    73.589        1      177      0.84
178  joint                   0.3281    73.948    73.948        1      178      0.85
179  joint                   0.3240    74.339    74.339        1      179      0.84
180  joint                   0.3209    74.934    74.934        1      180      0.84
181  joint                   0.3175    75.035    75.035        1      181      0.85
182  joint                   0.3150    75.499    75.499        1      182      0.85
183  joint                   0.3144    74.610    75.499        1      183      0.84
184  joint                   0.3171    74.653    75.499        1      184      0.85
185  joint                   0.3238    76.059    76.059        1      185      0.83
186  joint                   0.3066    75.530    76.059        1      186      0.85
187  joint                   0.3076    76.702    76.702        1      187      0.85
188  joint                   0.3014    76.622    76.702        1      188      0.85
189  joint                   0.3016    76.833    76.833        1      189      0.83
190  joint                   0.2986    77.020    77.020        1      190      0.85
191  joint                   0.2953    77.317    77.317        1      191      0.84
192  joint                   0.2950    77.383    77.383        1      192      0.82
193  joint                   0.2918    77.657    77.657        1      193      0.85
194  joint                   0.2871    78.634    78.634        1      194      0.85
195  joint                   0.2836    78.681    78.681        1      195      0.84
196  joint                   0.2818    78.561    78.681        1      196      0.84
197  joint                   0.2790    79.527    79.527        1      197      0.84
198  joint                   0.2752    79.712    79.712        1      198      0.85
199  joint                   0.2730    78.686    79.712        1      199      0.85
200  joint                   0.2751    79.033    79.712        1      200      0.85
201  joint                   0.2856    78.038    79.712        1      201      0.84
202  joint                   0.2855    77.413    79.712        1      202      0.85
203  joint                   0.2819    78.610    79.712        1      203      0.85
204  joint                   0.2816    79.149    79.712        1      204      0.85
205  joint                   0.2805    79.519    79.712        1      205      0.85
206  joint                   0.2697    79.315    79.712        1      206      0.84
207  joint                   0.2699    80.301    80.301        1      207      0.84
208  joint                   0.2638    80.478    80.478        1      208      0.85
209  joint                   0.2611    81.097    81.097        1      209      0.84
210  joint                   0.2579    81.352    81.352        1      210      0.85
211  joint                   0.2552    81.548    81.548        1      211      0.85
212  joint                   0.2532    80.975    81.548        1      212      0.84
213  joint                   0.2530    81.414    81.548        1      213      0.84
214  joint                   0.2610    78.542    81.548        1      214      0.85
215  joint                   0.2721    78.482    81.548        1      215      0.84
216  joint                   0.2964    81.095    81.548        1      216      0.84
217  joint                   0.2533    80.484    81.548        1      217      0.85
218  joint                   0.2575    81.146    81.548        1      218      0.84
219  joint                   0.2545    81.916    81.916        1      219      0.85
220  joint                   0.2492    82.136    82.136        1      220      0.84
221  joint                   0.2454    82.325    82.325        1      221      0.85
222  joint                   0.2447    82.873    82.873        1      222      0.85
223  joint                   0.2407    83.106    83.106        1      223      0.85
224  joint                   0.2381    83.212    83.212        1      224      0.85
225  joint                   0.2337    83.506    83.506        1      225      0.86
226  joint                   0.2311    83.841    83.841        1      226      0.85
227  joint                   0.2282    84.023    84.023        1      227      0.84
228  joint                   0.2258    84.358    84.358        1      228      0.85
229  joint                   0.2223    84.582    84.582        1      229      0.85
230  joint                   0.2200    84.914    84.914        1      230      0.85
231  joint                   0.2169    85.281    85.281        1      231      0.85
232  joint                   0.2141    85.530    85.530        1      232      0.84
233  joint                   0.2108    85.613    85.613        1      233      0.84
234  joint                   0.2086    85.951    85.951        1      234      0.85
235  joint                   0.2056    85.955    85.955        1      235      0.85
236  joint                   0.2041    85.775    85.955        1      236      0.85
237  joint                   0.2060    84.841    85.955        1      237      0.85
238  joint                   0.2166    82.594    85.955        1      238      0.85
239  joint                   0.2421    81.171    85.955        1      239      0.85
240  joint                   0.2563    83.441    85.955        1      240      0.85
241  joint                   0.2350    81.108    85.955        1      241      0.83
242  joint                   0.2461    82.451    85.955        1      242      0.83
243  joint                   0.2329    83.483    85.955        1      243      0.85
244  joint                   0.2312    84.302    85.955        1      244      0.84
245  joint                   0.2223    84.817    85.955        1      245      0.85
246  joint                   0.2124    85.078    85.955        1      246      0.84
247  joint                   0.2120    85.521    85.955        1      247      0.84
248  joint                   0.2084    85.587    85.955        1      248      0.84
249  joint                   0.2044    86.087    86.087        1      249      0.84
250  joint                   0.2010    86.259    86.259        1      250      0.84
251  joint                   0.1977    86.528    86.528        1      251      0.84
252  joint                   0.1950    86.875    86.875        1      252      0.85
253  joint                   0.1906    87.188    87.188        1      253      0.83
254  joint                   0.1889    87.254    87.254        1      254      0.83
255  joint                   0.1869    86.626    87.254        1      255      0.83
256  joint                   0.1922    85.856    87.254        1      256      0.85
257  joint                   0.2008    86.916    87.254        1      257      0.83
258  joint                   0.1927    86.146    87.254        1      258      0.83
259  joint                   0.1943    87.236    87.254        1      259      0.83
260  joint                   0.1866    87.483    87.483        1      260      0.83
261  joint                   0.1818    87.968    87.968        1      261      0.84
262  joint                   0.1785    88.273    88.273        1      262      0.84
263  joint                   0.1755    88.539    88.539        1      263      0.86
264  joint                   0.1713    88.744    88.744        1      264      0.85
265  joint                   0.1686    89.036    89.036        1      265      0.85
266  joint                   0.1661    89.299    89.299        1      266      0.84
267  joint                   0.1634    89.525    89.525        1      267      0.86
268  joint                   0.1600    89.721    89.721        1      268      0.84
269  joint                   0.1577    90.018    90.018        1      269      0.85
270  joint                   0.1553    90.264    90.264        1      270      0.85
271  joint                   0.1528    90.499    90.499        1      271      0.85
272  joint                   0.1500    90.750    90.750        1      272      0.86
273  joint                   0.1473    90.907    90.907        1      273      0.85
274  joint                   0.1454    91.071    91.071        1      274      0.85
275  joint                   0.1436    91.231    91.231        1      275      0.84
276  joint                   0.1418    91.442    91.442        1      276      0.85
277  joint                   0.1394    91.482    91.482        1      277      0.86
278  joint                   0.1379    91.827    91.827        1      278      0.85
279  joint                   0.1360    91.753    91.827        1      279      0.84
280  joint                   0.1342    92.073    92.073        1      280      0.85
281  joint                   0.1318    92.264    92.264        1      281      0.85
282  joint                   0.1293    92.304    92.304        1      282      0.85
283  joint                   0.1281    92.614    92.614        1      283      0.86
284  joint                   0.1250    92.726    92.726        1      284      0.85
285  joint                   0.1236    92.926    92.926        1      285      0.85
286  joint                   0.1208    93.097    93.097        1      286      0.85
287  joint                   0.1189    93.225    93.225        1      287      0.85
288  joint                   0.1174    93.461    93.461        1      288      0.85
289  joint                   0.1152    93.536    93.536        1      289      0.85
290  joint                   0.1133    93.717    93.717        1      290      0.85
291  joint                   0.1116    93.776    93.776        1      291      0.85
292  joint                   0.1104    93.876    93.876        1      292      0.85
293  joint                   0.1094    93.879    93.879        1      293      0.85
294  joint                   0.1087    93.829    93.879        1      294      0.85
295  joint                   0.1092    94.129    94.129        1      295      0.85
296  joint                   0.1064    94.049    94.129        1      296      0.85
297  joint                   0.1050    94.297    94.297        1      297      0.85
298  joint                   0.1038    94.135    94.297        1      298      0.85
299  joint                   0.1040    94.337    94.337        1      299      0.86
300  joint                   0.1012    94.590    94.590        1      300      0.85
301  joint                   0.0983    94.758    94.758        1      301      0.85
302  joint                   0.0969    94.815    94.815        1      302      0.85
303  joint                   0.0954    94.984    94.984        1      303      0.85
304  joint                   0.0940    95.029    95.029        1      304      0.85
305  joint                   0.0924    95.221    95.221        1      305      0.85
306  joint                   0.0908    95.306    95.306        1      306      0.85
307  joint                   0.0891    95.376    95.376        1      307      0.85
308  joint                   0.0876    95.526    95.526        1      308      0.85
309  joint                   0.0863    95.612    95.612        1      309      0.85
310  joint                   0.0852    95.703    95.703        1      310      0.85
311  joint                   0.0838    95.801    95.801        1      311      0.85
312  joint                   0.0825    95.857    95.857        1      312      0.85
313  joint                   0.0813    95.917    95.917        1      313      0.85
314  joint                   0.0801    96.020    96.020        1      314      0.85
315  joint                   0.0788    96.101    96.101        1      315      0.85
316  joint                   0.0778    96.179    96.179        1      316      0.98
317  joint                   0.0766    96.236    96.236        1      317      0.85
318  joint                   0.0755    96.317    96.317        1      318      0.85
319  joint                   0.0745    96.386    96.386        1      319      0.85
320  joint                   0.0734    96.439    96.439        1      320      0.85
321  joint                   0.0723    96.469    96.469        1      321      0.84
322  joint                   0.0715    96.594    96.594        1      322      0.85
323  joint                   0.0703    96.623    96.623        1      323      0.85
324  joint                   0.0696    96.627    96.627        1      324      0.84
325  joint                   0.0688    96.687    96.687        1      325      0.85
326  joint                   0.0676    96.735    96.735        1      326      0.85
327  joint                   0.0670    96.794    96.794        1      327      0.85
328  joint                   0.0660    96.851    96.851        1      328      0.81
329  joint                   0.0651    96.884    96.884        1      329      0.85
330  joint                   0.0645    96.946    96.946        1      330      0.85
331  joint                   0.0636    96.995    96.995        1      331      0.85
332  joint                   0.0629    97.041    97.041        1      332      0.85
333  joint                   0.0621    97.071    97.071        1      333      0.85
334  joint                   0.0614    97.122    97.122        1      334      0.85
335  joint                   0.0605    97.168    97.168        1      335      0.85
336  joint                   0.0599    97.208    97.208        1      336      0.85
337  joint                   0.0590    97.235    97.235        1      337      0.84
338  joint                   0.0583    97.286    97.286        1      338      0.84
339  joint                   0.0577    97.326    97.326        1      339      0.84
340  joint                   0.0570    97.333    97.333        1      340      0.85
341  joint                   0.0563    97.376    97.376        1      341      0.85
342  joint                   0.0557    97.410    97.410        1      342      0.84
343  joint                   0.0550    97.449    97.449        1      343      0.85
344  joint                   0.0545    97.478    97.478        1      344      0.85
345  joint                   0.0538    97.504    97.504        1      345      0.85
346  joint                   0.0533    97.548    97.548        1      346      0.85
347  joint                   0.0527    97.552    97.552        1      347      0.85
348  joint                   0.0521    97.574    97.574        1      348      0.86
349  joint                   0.0515    97.575    97.575        1      349      0.83
350  joint                   0.0509    97.591    97.591        1      350      0.85
351  joint                   0.0503    97.625    97.625        1      351      0.85
352  joint                   0.0498    97.644    97.644        1      352      0.84
353  joint                   0.0493    97.673    97.673        1      353      0.85
354  joint                   0.0488    97.709    97.709        1      354      0.85
355  joint                   0.0484    97.731    97.731        1      355      0.83
356  joint                   0.0478    97.749    97.749        1      356      0.85
357  joint                   0.0474    97.765    97.765        1      357      0.85
358  joint                   0.0470    97.785    97.785        1      358      0.85
359  joint                   0.0464    97.804    97.804        1      359      0.85
360  joint                   0.0460    97.816    97.816        1      360      0.86
361  joint                   0.0455    97.832    97.832        1      361      0.85
362  joint                   0.0451    97.854    97.854        1      362      0.85
363  joint                   0.0446    97.857    97.857        1      363      0.86
364  joint                   0.0442    97.883    97.883        1      364      0.85
365  joint                   0.0438    97.897    97.897        1      365      0.85
366  joint                   0.0434    97.909    97.909        1      366      0.86
367  joint                   0.0430    97.934    97.934        1      367      0.86
368  joint                   0.0425    97.948    97.948        1      368      0.86
369  joint                   0.0422    97.966    97.966        1      369      0.86
370  joint                   0.0419    97.971    97.971        1      370      0.86
371  joint                   0.0414    97.982    97.982        1      371      0.85
372  joint                   0.0410    98.002    98.002        1      372      0.85
373  joint                   0.0408    98.015    98.015        1      373      0.85
374  joint                   0.0404    98.027    98.027        1      374      0.83
375  joint                   0.0400    98.032    98.032        1      375      0.85
376  joint                   0.0397    98.042    98.042        1      376      0.86
377  joint                   0.0393    98.062    98.062        1      377      0.83
378  joint                   0.0389    98.068    98.068        1      378      0.86
379  joint                   0.0386    98.085    98.085        1      379      0.85
380  joint                   0.0382    98.098    98.098        1      380      0.85
381  joint                   0.0380    98.098    98.098        1      381      0.85
382  joint                   0.0377    98.102    98.102        1      382      0.85
383  joint                   0.0373    98.123    98.123        1      383      0.85
384  joint                   0.0371    98.118    98.123        1      384      0.85
385  joint                   0.0367    98.126    98.126        1      385      0.85
386  joint                   0.0365    98.147    98.147        1      386      0.85
387  joint                   0.0361    98.146    98.147        1      387      0.84
388  joint                   0.0358    98.143    98.147        1      388      0.84
389  joint                   0.0355    98.165    98.165        1      389      0.86
390  joint                   0.0352    98.181    98.181        1      390      0.85
391  joint                   0.0349    98.177    98.181        1      391      0.85
392  joint                   0.0346    98.193    98.193        1      392      0.84
393  joint                   0.0344    98.203    98.203        1      393      0.85
394  joint                   0.0341    98.220    98.220        1      394      0.85
395  joint                   0.0338    98.214    98.220        1      395      0.85
396  joint                   0.0336    98.221    98.221        1      396      0.85
397  joint                   0.0333    98.230    98.230        1      397      0.85
398  joint                   0.0331    98.240    98.240        1      398      0.85
399  joint                   0.0328    98.237    98.240        1      399      0.85
400  joint                   0.0327    98.249    98.249        1      400      0.85
401  joint                   0.0324    98.258    98.258        1      401      0.86
402  joint                   0.0321    98.259    98.259        1      402      0.85
403  joint                   0.0319    98.257    98.259        1      403      0.85
404  joint                   0.0315    98.264    98.264        1      404      0.85
405  joint                   0.0314    98.272    98.272        1      405      0.85
406  joint                   0.0312    98.269    98.272        1      406      0.85
407  joint                   0.0309    98.283    98.283        1      407      0.85
408  joint                   0.0308    98.290    98.290        1      408      0.85
409  joint                   0.0305    98.282    98.290        1      409      0.84
410  joint                   0.0303    98.298    98.298        1      410      0.85
411  joint                   0.0301    98.291    98.298        1      411      0.85
412  joint                   0.0298    98.283    98.298        1      412      0.85
413  joint                   0.0296    98.300    98.300        1      413      0.85
414  joint                   0.0294    98.293    98.300        1      414      0.85
415  joint                   0.0292    98.308    98.308        1      415      0.85
416  joint                   0.0290    98.324    98.324        1      416      0.85
417  joint                   0.0289    98.310    98.324        1      417      0.85
418  joint                   0.0286    98.321    98.324        1      418      0.85
419  joint                   0.0284    98.331    98.331        1      419      0.85
420  joint                   0.0283    98.336    98.336        1      420      0.85
421  joint                   0.0281    98.358    98.358        1      421      0.83
422  joint                   0.0279    98.348    98.358        1      422      0.85
423  joint                   0.0277    98.345    98.358        1      423      0.85
424  joint                   0.0276    98.362    98.362        1      424      0.84
425  joint                   0.0273    98.367    98.367        1      425      0.85
426  joint                   0.0270    98.354    98.367        1      426      0.85
427  joint                   0.0270    98.364    98.367        1      427      0.85
428  joint                   0.0268    98.385    98.385        1      428      0.84
429  joint                   0.0266    98.389    98.389        1      429      0.85
430  joint                   0.0265    98.387    98.389        1      430      0.83
431  joint                   0.0263    98.387    98.389        1      431      0.85
432  joint                   0.0261    98.394    98.394        1      432      0.85
433  joint                   0.0259    98.402    98.402        1      433      0.85
434  joint                   0.0257    98.393    98.402        1      434      0.85
435  joint                   0.0256    98.386    98.402        1      435      0.85
436  joint                   0.0255    98.398    98.402        1      436      0.85
437  joint                   0.0254    98.419    98.419        1      437      0.85
438  joint                   0.0252    98.412    98.419        1      438      0.85
439  joint                   0.0250    98.406    98.419        1      439      0.85
440  joint                   0.0248    98.408    98.419        1      440      0.85
441  joint                   0.0247    98.412    98.419        1      441      0.85
442  joint                   0.0245    98.401    98.419        1      442      0.84
443  joint                   0.0244    98.413    98.419        1      443      0.85
444  joint                   0.0242    98.413    98.419        1      444      0.83
445  joint                   0.0241    98.412    98.419        1      445      0.85
446  joint                   0.0239    98.417    98.419        1      446      0.85
447  joint                   0.0238    98.430    98.430        1      447      0.86
448  joint                   0.0237    98.429    98.430        1      448      0.85
449  joint                   0.0235    98.433    98.433        1      449      0.85
450  joint                   0.0234    98.443    98.443        1      450      0.85
451  joint                   0.0232    98.446    98.446        1      451      0.84
452  joint                   0.0231    98.448    98.448        1      452      0.85
453  joint                   0.0230    98.446    98.448        1      453      0.85
454  joint                   0.0228    98.443    98.448        1      454      0.83
455  joint                   0.0227    98.447    98.448        1      455      0.86
456  joint                   0.0226    98.447    98.448        1      456      0.85
457  joint                   0.0225    98.453    98.453        1      457      0.83
458  joint                   0.0224    98.453    98.453        1      458      0.85
459  joint                   0.0223    98.451    98.453        1      459      0.85
460  joint                   0.0222    98.443    98.453        1      460      0.85
461  joint                   0.0220    98.440    98.453        1      461      0.84
462  joint                   0.0219    98.454    98.454        1      462      0.85
463  joint                   0.0217    98.451    98.454        1      463      0.85
464  joint                   0.0216    98.454    98.454        1      464      0.85
465  joint                   0.0215    98.462    98.462        1      465      0.84
466  joint                   0.0214    98.476    98.476        1      466      0.85
467  joint                   0.0213    98.484    98.484        1      467      0.85
468  joint                   0.0211    98.475    98.484        1      468      0.85
469  joint                   0.0210    98.471    98.484        1      469      0.85
470  joint                   0.0210    98.470    98.484        1      470      0.86
471  joint                   0.0208    98.472    98.484        1      471      0.85
472  joint                   0.0207    98.476    98.484        1      472      0.85
473  joint                   0.0206    98.483    98.484        1      473      0.85
474  joint                   0.0205    98.484    98.484        1      474      0.84
475  joint                   0.0203    98.487    98.487        1      475      0.85
476  joint                   0.0202    98.490    98.490        1      476      0.85
477  joint                   0.0201    98.492    98.492        1      477      0.85
478  joint                   0.0201    98.501    98.501        1      478      0.85
479  joint                   0.0200    98.492    98.501        1      479      0.83
480  joint                   0.0198    98.485    98.501        1      480      0.86
481  joint                   0.0198    98.493    98.501        1      481      0.85
482  joint                   0.0196    98.493    98.501        1      482      0.85
483  joint                   0.0195    98.495    98.501        1      483      0.85
484  joint                   0.0195    98.500    98.501        1      484      0.85
485  joint                   0.0193    98.506    98.506        1      485      0.85
486  joint                   0.0193    98.508    98.508        1      486      0.86
487  joint                   0.0192    98.504    98.508        1      487      0.83
488  joint                   0.0190    98.517    98.517        1      488      0.85
489  joint                   0.0189    98.511    98.517        1      489      0.85
490  joint                   0.0189    98.506    98.517        1      490      0.86
491  joint                   0.0188    98.514    98.517        1      491      0.85
492  joint                   0.0187    98.521    98.521        1      492      0.84
493  joint                   0.0187    98.505    98.521        1      493      0.85
494  joint                   0.0185    98.512    98.521        1      494      0.85
495  joint                   0.0184    98.523    98.523        1      495      0.85
496  joint                   0.0183    98.518    98.523        1      496      0.85
497  joint                   0.0183    98.519    98.523        1      497      0.85
498  joint                   0.0182    98.528    98.528        1      498      0.84
499  joint                   0.0181    98.537    98.537        1      499      0.85
500  joint                   0.0179    98.529    98.537        1      500      0.85
501  joint                   0.0179    98.527    98.537        1      501      0.86
502  joint                   0.0178    98.530    98.537        1      502      0.85
503  joint                   0.0177    98.536    98.537        1      503      0.85
504  joint                   0.0177    98.533    98.537        1      504      0.85
505  joint                   0.0176    98.534    98.537        1      505      0.85
506  joint                   0.0175    98.527    98.537        1      506      0.85
507  joint                   0.0175    98.529    98.537        1      507      0.85
508  joint                   0.0173    98.545    98.545        1      508      0.85
509  joint                   0.0172    98.552    98.552        1      509      0.85
510  joint                   0.0172    98.550    98.552        1      510      0.82
511  joint                   0.0171    98.551    98.552        1      511      0.83
512  joint                   0.0170    98.555    98.555        1      512      0.85
513  joint                   0.0169    98.561    98.561        1      513      0.83
514  joint                   0.0169    98.555    98.561        1      514      0.85
515  joint                   0.0168    98.548    98.561        1      515      0.85
516  joint                   0.0168    98.555    98.561        1      516      0.85
517  joint                   0.0166    98.551    98.561        1      517      0.84
518  joint                   0.0166    98.554    98.561        1      518      0.85
519  joint                   0.0165    98.553    98.561        1      519      0.85
520  joint                   0.0164    98.545    98.561        1      520      0.85
521  joint                   0.0164    98.542    98.561        1      521      0.84
522  joint                   0.0163    98.551    98.561        1      522      0.85
523  joint                   0.0162    98.555    98.561        1      523      0.85
524  joint                   0.0161    98.559    98.561        1      524      0.85
525  joint                   0.0160    98.547    98.561        1      525      0.85
526  joint                   0.0160    98.553    98.561        1      526      0.85
527  joint                   0.0159    98.559    98.561        1      527      0.85
528  joint                   0.0159    98.561    98.561        1      528      0.85
529  joint                   0.0158    98.557    98.561        1      529      0.86
530  joint                   0.0157    98.555    98.561        1      530      0.83
531  joint                   0.0157    98.558    98.561        1      531      0.84
532  joint                   0.0156    98.560    98.561        1      532      0.85
533  joint                   0.0155    98.559    98.561        1      533      0.85
534  joint                   0.0155    98.563    98.563        1      534      0.84
535  joint                   0.0154    98.565    98.565        1      535      0.85
536  joint                   0.0154    98.566    98.566        1      536      0.85
537  joint                   0.0153    98.566    98.566        1      537      0.86
538  joint                   0.0153    98.568    98.568        1      538      0.85
539  joint                   0.0152    98.562    98.568        1      539      0.85
540  joint                   0.0152    98.555    98.568        1      540      0.85
541  joint                   0.0150    98.555    98.568        1      541      0.85
542  joint                   0.0150    98.569    98.569        1      542      0.85
543  joint                   0.0149    98.574    98.574        1      543      0.84
544  joint                   0.0149    98.568    98.574        1      544      0.85
545  joint                   0.0148    98.568    98.574        1      545      0.85
546  joint                   0.0148    98.576    98.576        1      546      0.85
547  joint                   0.0146    98.575    98.576        1      547      0.83
548  joint                   0.0146    98.575    98.576        1      548      0.85
549  joint                   0.0146    98.573    98.576        1      549      0.85
550  joint                   0.0145    98.574    98.576        1      550      0.85
551  joint                   0.0144    98.583    98.583        1      551      0.85
552  joint                   0.0144    98.586    98.586        1      552      0.85
553  joint                   0.0144    98.583    98.586        1      553      0.85
554  joint                   0.0143    98.590    98.590        1      554      0.85
555  joint                   0.0142    98.586    98.590        1      555      0.85
556  joint                   0.0142    98.583    98.590        1      556      0.85
557  joint                   0.0142    98.580    98.590        1      557      0.85
558  joint                   0.0141    98.580    98.590        1      558      0.85
559  joint                   0.0140    98.586    98.590        1      559      0.85
560  joint                   0.0139    98.589    98.590        1      560      0.86
561  joint                   0.0139    98.585    98.590        1      561      0.85
562  joint                   0.0139    98.584    98.590        1      562      0.85
563  joint                   0.0138    98.587    98.590        1      563      0.85
564  joint                   0.0138    98.576    98.590        1      564      0.85
565  joint                   0.0137    98.578    98.590        1      565      0.85
566  joint                   0.0137    98.585    98.590        1      566      0.85
567  joint                   0.0136    98.590    98.590        1      567      0.84
568  joint                   0.0135    98.591    98.591        1      568      0.85
569  joint                   0.0135    98.594    98.594        1      569      0.85
570  joint                   0.0134    98.605    98.605        1      570      0.85
571  joint                   0.0134    98.599    98.605        1      571      0.85
572  joint                   0.0134    98.597    98.605        1      572      0.85
573  joint                   0.0133    98.591    98.605        1      573      0.85
574  joint                   0.0133    98.592    98.605        1      574      0.85
575  joint                   0.0132    98.599    98.605        1      575      0.86
576  joint                   0.0132    98.602    98.605        1      576      0.85
577  joint                   0.0131    98.599    98.605        1      577      0.85
578  joint                   0.0130    98.598    98.605        1      578      0.85
579  joint                   0.0130    98.598    98.605        1      579      0.85
580  joint                   0.0129    98.604    98.605        1      580      0.85
581  joint                   0.0129    98.606    98.606        1      581      0.85
582  joint                   0.0129    98.602    98.606        1      582      0.85
583  joint                   0.0128    98.600    98.606        1      583      0.85
584  joint                   0.0128    98.600    98.606        1      584      0.85
585  joint                   0.0128    98.605    98.606        1      585      0.85
586  joint                   0.0127    98.612    98.612        1      586      0.84 *
587  joint                   0.0127    98.607    98.612        1      587      0.85
588  joint                   0.0126    98.610    98.612        1      588      0.85
589  joint                   0.0126    98.610    98.612        1      589      0.85
590  joint                   0.0126    98.608    98.612        1      590      0.85
591  joint                   0.0125    98.604    98.612        1      591      0.85
592  joint                   0.0125    98.610    98.612        1      592      0.84
593  joint                   0.0124    98.605    98.612        1      593      0.85
594  joint                   0.0124    98.610    98.612        1      594      0.85
595  joint                   0.0123    98.605    98.612        1      595      0.85
596  joint                   0.0123    98.608    98.612        1      596      0.84
597  joint                   0.0122    98.605    98.612        1      597      0.85
598  joint                   0.0122    98.602    98.612        1      598      0.84
599  joint                   0.0122    98.602    98.612        1      599      0.84
600  joint                   0.0121    98.597    98.612        1      600      0.85
```

</details>

<details>
<summary>ppi | large | shared_dynamic_c — 전체 에포크</summary>

```text
[ppi | large | shared_dynamic_c]
status=passed | last_epoch=600 | selected_epoch=600
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   0.7391    44.620    44.620        1        1      8.77
  2  joint                   0.6293    42.075    44.620        1        2      7.34
  3  joint                   0.5794    41.354    44.620        1        3      7.54
  4  joint                   0.5626    40.267    44.620        1        4      7.65
  5  joint                   0.5558    39.303    44.620        1        5      7.52
  6  joint                   0.5527    39.344    44.620        1        6      7.45
  7  joint                   0.5514    39.368    44.620        1        7      7.72
  8  joint                   0.5508    40.156    44.620        1        8      7.54
  9  joint                   0.5506    39.970    44.620        1        9      7.69
 10  joint                   0.5504    39.692    44.620        1       10      7.47
 11  joint                   0.5502    40.074    44.620        1       11      7.61
 12  joint                   0.5498    39.978    44.620        1       12      7.48
 13  joint                   0.5489    39.462    44.620        1       13      7.53
 14  joint                   0.5479    40.181    44.620        1       14      7.49
 15  joint                   0.5464    41.374    44.620        1       15      7.54
 16  joint                   0.5448    42.656    44.620        1       16      7.64
 17  joint                   0.5433    43.534    44.620        1       17      7.68
 18  joint                   0.5418    43.518    44.620        1       18      7.65
 19  joint                   0.5399    45.516    45.516        1       19      7.49
 20  joint                   0.5378    44.492    45.516        1       20      7.44
 21  joint                   0.5356    48.896    48.896        1       21      7.72
 22  joint                   0.5345    40.877    48.896        1       22      7.48
 23  joint                   0.5379    44.713    48.896        1       23      7.57
 24  joint                   0.5314    53.319    53.319        1       24      7.66
 25  joint                   0.5431    42.308    53.319        1       25      7.64
 26  joint                   0.5340    39.819    53.319        1       26      7.67
 27  joint                   0.5403    40.851    53.319        1       27      7.63
 28  joint                   0.5385    43.485    53.319        1       28      7.32
 29  joint                   0.5343    46.297    53.319        1       29      7.59
 30  joint                   0.5328    48.225    53.319        1       30      7.57
 31  joint                   0.5340    48.555    53.319        1       31      7.46
 32  joint                   0.5343    47.326    53.319        1       32      7.49
 33  joint                   0.5316    44.763    53.319        1       33      7.47
 34  joint                   0.5286    41.942    53.319        1       34      7.67
 35  joint                   0.5281    41.205    53.319        1       35      7.54
 36  joint                   0.5280    43.453    53.319        1       36      7.70
 37  joint                   0.5255    47.288    53.319        1       37      7.45
 38  joint                   0.5229    49.916    53.319        1       38      7.64
 39  joint                   0.5233    50.454    53.319        1       39      7.61
 40  joint                   0.5217    48.720    53.319        1       40      7.56
 41  joint                   0.5195    47.297    53.319        1       41      7.49
 42  joint                   0.5191    48.536    53.319        1       42      7.59
 43  joint                   0.5166    50.331    53.319        1       43      7.47
 44  joint                   0.5154    50.464    53.319        1       44      7.56
 45  joint                   0.5129    49.291    53.319        1       45      7.59
 46  joint                   0.5120    50.691    53.319        1       46      7.53
 47  joint                   0.5093    50.815    53.319        1       47      7.51
 48  joint                   0.5078    48.205    53.319        1       48      7.51
 49  joint                   0.5071    54.118    54.118        1       49      7.65
 50  joint                   0.5091    43.887    54.118        1       50      7.64
 51  joint                   0.5192    52.136    54.118        1       51      7.65
 52  joint                   0.5010    54.612    54.612        1       52      7.48
 53  joint                   0.5121    49.529    54.612        1       53      7.63
 54  joint                   0.5022    46.937    54.612        1       54      7.64
 55  joint                   0.5060    50.750    54.612        1       55      7.53
 56  joint                   0.5004    54.421    54.612        1       56      7.59
 57  joint                   0.4988    53.690    54.612        1       57      7.63
 58  joint                   0.4975    49.852    54.612        1       58      7.54
 59  joint                   0.4951    50.611    54.612        1       59      7.59
 60  joint                   0.4925    54.319    54.612        1       60      7.58
 61  joint                   0.4922    54.021    54.612        1       61      7.50
 62  joint                   0.4870    50.868    54.612        1       62      7.43
 63  joint                   0.4890    56.374    56.374        1       63      7.45
 64  joint                   0.4845    53.657    56.374        1       64      7.57
 65  joint                   0.4845    55.846    56.374        1       65      7.51
 66  joint                   0.4805    50.149    56.374        1       66      7.58
 67  joint                   0.4850    58.118    58.118        1       67      7.58
 68  joint                   0.4925    50.254    58.118        1       68      7.57
 69  joint                   0.4927    52.422    58.118        1       69      7.62
 70  joint                   0.4826    55.422    58.118        1       70      7.39
 71  joint                   0.4791    56.242    58.118        1       71      7.35
 72  joint                   0.4818    55.504    58.118        1       72      7.60
 73  joint                   0.4761    54.221    58.118        1       73      7.41
 74  joint                   0.4750    55.105    58.118        1       74      7.61
 75  joint                   0.4728    55.932    58.118        1       75      7.47
 76  joint                   0.4689    56.049    58.118        1       76      7.63
 77  joint                   0.4660    57.247    58.118        1       77      7.71
 78  joint                   0.4636    57.806    58.118        1       78      7.56
 79  joint                   0.4618    58.530    58.530        1       79      7.41
 80  joint                   0.4582    57.489    58.530        1       80      7.50
 81  joint                   0.4597    57.654    58.530        1       81      7.60
 82  joint                   0.5175    49.969    58.530        1       82      7.56
 83  joint                   0.4786    55.636    58.530        1       83      7.50
 84  joint                   0.4713    59.178    59.178        1       84      7.62
 85  joint                   0.4723    57.557    59.178        1       85      7.43
 86  joint                   0.4677    55.255    59.178        1       86      7.74
 87  joint                   0.4703    55.147    59.178        1       87      7.52
 88  joint                   0.4668    56.771    59.178        1       88      7.47
 89  joint                   0.4640    57.481    59.178        1       89      7.49
 90  joint                   0.4607    56.425    59.178        1       90      7.32
 91  joint                   0.4596    56.590    59.178        1       91      7.50
 92  joint                   0.4562    57.600    59.178        1       92      7.60
 93  joint                   0.4544    57.936    59.178        1       93      7.58
 94  joint                   0.4505    58.102    59.178        1       94      7.56
 95  joint                   0.4498    59.236    59.236        1       95      7.61
 96  joint                   0.4468    59.370    59.370        1       96      7.46
 97  joint                   0.4451    59.107    59.370        1       97      7.64
 98  joint                   0.4424    60.804    60.804        1       98      7.39
 99  joint                   0.4394    60.084    60.804        1       99      7.59
100  joint                   0.4364    60.325    60.804        1      100      7.39
101  joint                   0.4336    61.590    61.590        1      101      7.39
102  joint                   0.4315    60.097    61.590        1      102      7.56
103  joint                   0.4306    61.756    61.756        1      103      7.56
104  joint                   0.4341    56.897    61.756        1      104      7.63
105  joint                   0.4507    62.653    62.653        1      105      7.70
106  joint                   0.4277    62.104    62.653        1      106      7.47
107  joint                   0.4373    59.870    62.653        1      107      7.57
108  joint                   0.4262    62.813    62.813        1      108      7.43
109  joint                   0.4269    62.818    62.818        1      109      7.60
110  joint                   0.4241    61.124    62.818        1      110      7.40
111  joint                   0.4190    63.295    63.295        1      111      7.55
112  joint                   0.4163    64.081    64.081        1      112      7.64
113  joint                   0.4160    62.169    64.081        1      113      7.49
114  joint                   0.4116    62.095    64.081        1      114      7.60
115  joint                   0.4114    65.781    65.781        1      115      7.55
116  joint                   0.4199    61.675    65.781        1      116      7.65
117  joint                   0.4154    63.840    65.781        1      117      7.46
118  joint                   0.4050    65.549    65.781        1      118      7.50
119  joint                   0.4053    66.353    66.353        1      119      7.48
120  joint                   0.4006    63.870    66.353        1      120      7.51
121  joint                   0.3988    63.689    66.353        1      121      7.44
122  joint                   0.3958    67.337    67.337        1      122      7.52
123  joint                   0.3930    66.216    67.337        1      123      7.59
124  joint                   0.3900    66.975    67.337        1      124      7.43
125  joint                   0.3901    62.938    67.337        1      125      7.62
126  joint                   0.4168    64.039    67.337        1      126      7.50
127  joint                   0.4337    62.004    67.337        1      127      7.51
128  joint                   0.4144    63.583    67.337        1      128      7.42
129  joint                   0.3986    66.525    67.337        1      129      7.42
130  joint                   0.4055    66.623    67.337        1      130      7.45
131  joint                   0.3982    64.636    67.337        1      131      7.58
132  joint                   0.3977    64.784    67.337        1      132      7.53
133  joint                   0.3925    66.174    67.337        1      133      7.41
134  joint                   0.3894    66.573    67.337        1      134      7.64
135  joint                   0.3873    66.353    67.337        1      135      7.53
136  joint                   0.3845    66.450    67.337        1      136      7.70
137  joint                   0.3805    66.996    67.337        1      137      7.62
138  joint                   0.3772    67.652    67.652        1      138      7.52
139  joint                   0.3752    68.914    68.914        1      139      7.62
140  joint                   0.3720    69.396    69.396        1      140      7.51
141  joint                   0.3694    69.128    69.396        1      141      7.41
142  joint                   0.3654    69.054    69.396        1      142      7.58
143  joint                   0.3638    69.865    69.865        1      143      7.56
144  joint                   0.3594    70.488    70.488        1      144      7.55
145  joint                   0.3582    70.746    70.746        1      145      7.65
146  joint                   0.3544    70.570    70.746        1      146      7.47
147  joint                   0.3521    71.083    71.083        1      147      7.46
148  joint                   0.3491    71.933    71.933        1      148      7.54
149  joint                   0.3465    72.138    72.138        1      149      7.42
150  joint                   0.3436    72.151    72.151        1      150      7.49
151  joint                   0.3409    71.807    72.151        1      151      7.56
152  joint                   0.3394    73.193    73.193        1      152      7.52
153  joint                   0.3520    65.795    73.193        1      153      7.54
154  joint                   0.3901    70.807    73.193        1      154      7.47
155  joint                   0.3579    72.324    73.193        1      155      7.64
156  joint                   0.3509    71.087    73.193        1      156      7.65
157  joint                   0.3547    71.808    73.193        1      157      7.37
158  joint                   0.3471    72.330    73.193        1      158      7.47
159  joint                   0.3415    71.439    73.193        1      159      7.52
160  joint                   0.3417    71.727    73.193        1      160      7.51
161  joint                   0.3371    72.824    73.193        1      161      7.59
162  joint                   0.3339    73.591    73.591        1      162      7.67
163  joint                   0.3290    73.711    73.711        1      163      7.47
164  joint                   0.3282    75.057    75.057        1      164      7.55
165  joint                   0.3236    74.800    75.057        1      165      7.57
166  joint                   0.3188    74.569    75.057        1      166      7.40
167  joint                   0.3160    75.602    75.602        1      167      7.43
168  joint                   0.3124    76.249    76.249        1      168      7.51
169  joint                   0.3098    76.002    76.249        1      169      7.55
170  joint                   0.3057    76.096    76.249        1      170      7.51
171  joint                   0.3036    76.863    76.863        1      171      7.45
172  joint                   0.3002    76.190    76.863        1      172      7.47
173  joint                   0.3044    74.124    76.863        1      173      7.60
174  joint                   0.3434    72.741    76.863        1      174      7.42
175  joint                   0.3256    73.052    76.863        1      175      7.49
176  joint                   0.3340    75.308    76.863        1      176      7.56
177  joint                   0.3136    75.598    76.863        1      177      7.29
178  joint                   0.3131    75.116    76.863        1      178      7.56
179  joint                   0.3104    75.370    76.863        1      179      7.55
180  joint                   0.3068    76.248    76.863        1      180      7.43
181  joint                   0.3042    76.768    76.863        1      181      7.59
182  joint                   0.2994    76.732    76.863        1      182      7.46
183  joint                   0.2968    76.992    76.992        1      183      7.53
184  joint                   0.2942    77.902    77.902        1      184      7.55
185  joint                   0.2890    77.950    77.950        1      185      7.48
186  joint                   0.2871    78.051    78.051        1      186      7.54
187  joint                   0.2832    78.702    78.702        1      187      7.48
188  joint                   0.2808    79.281    79.281        1      188      7.55
189  joint                   0.2775    79.187    79.281        1      189      7.67
190  joint                   0.2745    79.775    79.775        1      190      7.54
191  joint                   0.2706    80.340    80.340        1      191      7.58
192  joint                   0.2687    80.195    80.340        1      192      7.57
193  joint                   0.2656    80.513    80.513        1      193      7.60
194  joint                   0.2627    81.139    81.139        1      194      7.44
195  joint                   0.2596    81.320    81.320        1      195      7.50
196  joint                   0.2565    81.460    81.460        1      196      7.53
197  joint                   0.2541    81.804    81.804        1      197      7.56
198  joint                   0.2523    82.045    82.045        1      198      7.50
199  joint                   0.2515    81.655    82.045        1      199      7.41
200  joint                   0.2507    82.565    82.565        1      200      7.32
201  joint                   0.2487    82.423    82.565        1      201      7.53
202  joint                   0.2417    82.859    82.859        1      202      7.56
203  joint                   0.2421    82.784    82.859        1      203      7.43
204  joint                   0.2399    82.692    82.859        1      204      7.66
205  joint                   0.2435    82.881    82.881        1      205      7.51
206  joint                   0.2426    82.471    82.881        1      206      7.45
207  joint                   0.2384    82.914    82.914        1      207      7.57
208  joint                   0.2378    83.378    83.378        1      208      7.41
209  joint                   0.2384    83.465    83.465        1      209      7.64
210  joint                   0.2312    84.281    84.281        1      210      7.41
211  joint                   0.2263    83.667    84.281        1      211      7.56
212  joint                   0.2274    84.858    84.858        1      212      7.45
213  joint                   0.2206    84.652    84.858        1      213      7.49
214  joint                   0.2206    85.272    85.272        1      214      7.64
215  joint                   0.2158    85.491    85.491        1      215      7.53
216  joint                   0.2123    85.650    85.650        1      216      7.64
217  joint                   0.2107    85.856    85.856        1      217      7.55
218  joint                   0.2072    86.316    86.316        1      218      7.48
219  joint                   0.2038    86.424    86.424        1      219      7.45
220  joint                   0.2025    86.952    86.952        1      220      7.55
221  joint                   0.1989    86.872    86.952        1      221      7.60
222  joint                   0.1966    87.037    87.037        1      222      7.54
223  joint                   0.1974    85.900    87.037        1      223      7.61
224  joint                   0.2019    86.988    87.037        1      224      7.40
225  joint                   0.1984    86.825    87.037        1      225      7.61
226  joint                   0.1962    86.371    87.037        1      226      7.61
227  joint                   0.2035    84.973    87.037        1      227      7.55
228  joint                   0.2092    86.453    87.037        1      228      7.48
229  joint                   0.2023    85.083    87.037        1      229      7.62
230  joint                   0.2140    84.940    87.037        1      230      7.69
231  joint                   0.2067    85.861    87.037        1      231      7.53
232  joint                   0.2068    86.675    87.037        1      232      7.58
233  joint                   0.1979    86.392    87.037        1      233      7.36
234  joint                   0.1956    87.028    87.037        1      234      7.55
235  joint                   0.1901    87.616    87.616        1      235      7.57
236  joint                   0.1878    87.996    87.996        1      236      7.53
237  joint                   0.1836    88.062    88.062        1      237      7.59
238  joint                   0.1807    88.529    88.529        1      238      7.61
239  joint                   0.1775    88.821    88.821        1      239      7.49
240  joint                   0.1740    88.940    88.940        1      240      7.58
241  joint                   0.1708    89.187    89.187        1      241      7.53
242  joint                   0.1678    89.480    89.480        1      242      7.68
243  joint                   0.1654    89.708    89.708        1      243      7.61
244  joint                   0.1629    89.816    89.816        1      244      7.60
245  joint                   0.1607    90.168    90.168        1      245      7.46
246  joint                   0.1580    90.216    90.216        1      246      7.64
247  joint                   0.1557    90.353    90.353        1      247      7.57
248  joint                   0.1546    90.402    90.402        1      248      7.55
249  joint                   0.1522    90.826    90.826        1      249      7.37
250  joint                   0.1495    91.105    91.105        1      250      7.58
251  joint                   0.1461    91.292    91.292        1      251      7.43
252  joint                   0.1440    91.375    91.375        1      252      7.64
253  joint                   0.1422    91.656    91.656        1      253      7.59
254  joint                   0.1390    91.855    91.855        1      254      7.67
255  joint                   0.1375    92.103    92.103        1      255      7.50
256  joint                   0.1349    92.213    92.213        1      256      7.55
257  joint                   0.1332    92.501    92.501        1      257      7.46
258  joint                   0.1311    92.499    92.501        1      258      7.52
259  joint                   0.1294    92.683    92.683        1      259      7.56
260  joint                   0.1274    92.710    92.710        1      260      7.67
261  joint                   0.1258    92.965    92.965        1      261      7.47
262  joint                   0.1245    92.965    92.965        1      262      7.46
263  joint                   0.1232    93.245    93.245        1      263      7.51
264  joint                   0.1215    93.413    93.413        1      264      7.55
265  joint                   0.1191    93.342    93.413        1      265      7.39
266  joint                   0.1176    93.754    93.754        1      266      7.69
267  joint                   0.1151    93.854    93.854        1      267      7.59
268  joint                   0.1137    93.940    93.940        1      268      7.55
269  joint                   0.1113    94.094    94.094        1      269      7.47
270  joint                   0.1095    94.176    94.176        1      270      7.52
271  joint                   0.1082    94.382    94.382        1      271      7.76
272  joint                   0.1066    94.431    94.431        1      272      7.52
273  joint                   0.1048    94.595    94.595        1      273      7.52
274  joint                   0.1034    94.726    94.726        1      274      7.54
275  joint                   0.1017    94.734    94.734        1      275      7.53
276  joint                   0.1003    94.890    94.890        1      276      7.49
277  joint                   0.0985    95.052    95.052        1      277      7.52
278  joint                   0.0979    95.012    95.052        1      278      7.54
279  joint                   0.0969    95.127    95.127        1      279      7.62
280  joint                   0.0957    95.285    95.285        1      280      7.45
281  joint                   0.0937    95.385    95.385        1      281      7.40
282  joint                   0.0919    95.490    95.490        1      282      7.43
283  joint                   0.0910    95.578    95.578        1      283      7.58
284  joint                   0.0894    95.583    95.583        1      284      7.34
285  joint                   0.0883    95.746    95.746        1      285      7.53
286  joint                   0.0869    95.896    95.896        1      286      7.51
287  joint                   0.0857    95.882    95.896        1      287      7.54
288  joint                   0.0844    95.945    95.945        1      288      7.69
289  joint                   0.0835    96.053    96.053        1      289      7.46
290  joint                   0.0820    96.140    96.140        1      290      7.44
291  joint                   0.0811    96.223    96.223        1      291      7.52
292  joint                   0.0798    96.298    96.298        1      292      7.53
293  joint                   0.0789    96.360    96.360        1      293      7.40
294  joint                   0.0777    96.413    96.413        1      294      7.47
295  joint                   0.0765    96.487    96.487        1      295      7.68
296  joint                   0.0756    96.520    96.520        1      296      7.63
297  joint                   0.0748    96.629    96.629        1      297      7.53
298  joint                   0.0737    96.629    96.629        1      298      7.49
299  joint                   0.0728    96.693    96.693        1      299      7.45
300  joint                   0.0716    96.771    96.771        1      300      7.56
301  joint                   0.0707    96.826    96.826        1      301      7.46
302  joint                   0.0700    96.856    96.856        1      302      7.57
303  joint                   0.0690    96.918    96.918        1      303      7.69
304  joint                   0.0682    96.939    96.939        1      304      7.52
305  joint                   0.0674    97.006    97.006        1      305      7.60
306  joint                   0.0668    96.980    97.006        1      306      7.53
307  joint                   0.0662    97.052    97.052        1      307      7.65
308  joint                   0.0652    97.096    97.096        1      308      7.64
309  joint                   0.0642    97.142    97.142        1      309      7.45
310  joint                   0.0634    97.218    97.218        1      310      7.47
311  joint                   0.0627    97.231    97.231        1      311      7.52
312  joint                   0.0619    97.287    97.287        1      312      7.50
313  joint                   0.0612    97.319    97.319        1      313      7.47
314  joint                   0.0604    97.335    97.335        1      314      7.47
315  joint                   0.0597    97.411    97.411        1      315      7.52
316  joint                   0.0591    97.427    97.427        1      316      7.66
317  joint                   0.0584    97.445    97.445        1      317      7.54
318  joint                   0.0577    97.478    97.478        1      318      7.44
319  joint                   0.0570    97.529    97.529        1      319      7.44
320  joint                   0.0564    97.541    97.541        1      320      7.65
321  joint                   0.0557    97.565    97.565        1      321      7.49
322  joint                   0.0552    97.601    97.601        1      322      7.57
323  joint                   0.0544    97.625    97.625        1      323      7.45
324  joint                   0.0540    97.645    97.645        1      324      7.44
325  joint                   0.0532    97.672    97.672        1      325      7.56
326  joint                   0.0527    97.692    97.692        1      326      7.61
327  joint                   0.0522    97.714    97.714        1      327      7.42
328  joint                   0.0516    97.716    97.716        1      328      7.64
329  joint                   0.0511    97.728    97.728        1      329      7.47
330  joint                   0.0506    97.763    97.763        1      330      7.59
331  joint                   0.0500    97.780    97.780        1      331      7.66
332  joint                   0.0494    97.807    97.807        1      332      7.47
333  joint                   0.0488    97.824    97.824        1      333      7.55
334  joint                   0.0483    97.855    97.855        1      334      7.37
335  joint                   0.0479    97.861    97.861        1      335      7.57
336  joint                   0.0475    97.890    97.890        1      336      7.64
337  joint                   0.0470    97.907    97.907        1      337      7.51
338  joint                   0.0464    97.931    97.931        1      338      7.59
339  joint                   0.0462    97.928    97.931        1      339      7.43
340  joint                   0.0457    97.964    97.964        1      340      7.49
341  joint                   0.0451    97.971    97.971        1      341      7.70
342  joint                   0.0447    97.985    97.985        1      342      7.61
343  joint                   0.0443    98.009    98.009        1      343      7.46
344  joint                   0.0439    98.019    98.019        1      344      7.42
345  joint                   0.0434    98.027    98.027        1      345      7.39
346  joint                   0.0430    98.041    98.041        1      346      7.32
347  joint                   0.0426    98.051    98.051        1      347      7.49
348  joint                   0.0422    98.082    98.082        1      348      7.55
349  joint                   0.0418    98.103    98.103        1      349      7.65
350  joint                   0.0414    98.108    98.108        1      350      7.66
351  joint                   0.0411    98.110    98.110        1      351      7.30
352  joint                   0.0406    98.138    98.138        1      352      7.48
353  joint                   0.0403    98.141    98.141        1      353      7.52
354  joint                   0.0399    98.151    98.151        1      354      7.69
355  joint                   0.0396    98.155    98.155        1      355      7.63
356  joint                   0.0391    98.156    98.156        1      356      7.62
357  joint                   0.0389    98.176    98.176        1      357      7.57
358  joint                   0.0386    98.185    98.185        1      358      7.48
359  joint                   0.0382    98.192    98.192        1      359      7.45
360  joint                   0.0377    98.196    98.196        1      360      7.47
361  joint                   0.0376    98.208    98.208        1      361      7.52
362  joint                   0.0372    98.222    98.222        1      362      7.55
363  joint                   0.0370    98.238    98.238        1      363      7.65
364  joint                   0.0366    98.247    98.247        1      364      7.58
365  joint                   0.0364    98.257    98.257        1      365      7.53
366  joint                   0.0359    98.259    98.259        1      366      7.69
367  joint                   0.0357    98.265    98.265        1      367      7.59
368  joint                   0.0354    98.279    98.279        1      368      7.55
369  joint                   0.0352    98.291    98.291        1      369      7.54
370  joint                   0.0348    98.288    98.291        1      370      7.45
371  joint                   0.0345    98.299    98.299        1      371      7.51
372  joint                   0.0343    98.295    98.299        1      372      7.60
373  joint                   0.0341    98.295    98.299        1      373      7.59
374  joint                   0.0337    98.298    98.299        1      374      7.61
375  joint                   0.0336    98.317    98.317        1      375      7.43
376  joint                   0.0332    98.320    98.320        1      376      7.44
377  joint                   0.0330    98.328    98.328        1      377      7.61
378  joint                   0.0328    98.333    98.333        1      378      7.45
379  joint                   0.0325    98.332    98.333        1      379      7.61
380  joint                   0.0322    98.336    98.336        1      380      7.65
381  joint                   0.0321    98.340    98.340        1      381      7.64
382  joint                   0.0317    98.353    98.353        1      382      7.59
383  joint                   0.0315    98.368    98.368        1      383      7.59
384  joint                   0.0313    98.355    98.368        1      384      7.72
385  joint                   0.0311    98.371    98.371        1      385      7.58
386  joint                   0.0309    98.377    98.377        1      386      7.77
387  joint                   0.0307    98.386    98.386        1      387      7.52
388  joint                   0.0303    98.382    98.386        1      388      7.59
389  joint                   0.0301    98.386    98.386        1      389      7.60
390  joint                   0.0300    98.395    98.395        1      390      7.55
391  joint                   0.0298    98.383    98.395        1      391      7.57
392  joint                   0.0295    98.400    98.400        1      392      7.48
393  joint                   0.0293    98.416    98.416        1      393      7.65
394  joint                   0.0291    98.402    98.416        1      394      7.51
395  joint                   0.0289    98.410    98.416        1      395      7.50
396  joint                   0.0287    98.410    98.416        1      396      7.50
397  joint                   0.0284    98.421    98.421        1      397      7.52
398  joint                   0.0283    98.440    98.440        1      398      7.57
399  joint                   0.0281    98.435    98.440        1      399      7.56
400  joint                   0.0279    98.424    98.440        1      400      7.49
401  joint                   0.0276    98.442    98.442        1      401      7.69
402  joint                   0.0275    98.441    98.442        1      402      7.58
403  joint                   0.0273    98.440    98.442        1      403      7.41
404  joint                   0.0271    98.457    98.457        1      404      7.63
405  joint                   0.0269    98.457    98.457        1      405      7.59
406  joint                   0.0267    98.450    98.457        1      406      7.67
407  joint                   0.0266    98.461    98.461        1      407      7.50
408  joint                   0.0264    98.466    98.466        1      408      7.50
409  joint                   0.0262    98.455    98.466        1      409      7.62
410  joint                   0.0261    98.461    98.466        1      410      7.57
411  joint                   0.0260    98.466    98.466        1      411      7.36
412  joint                   0.0257    98.477    98.477        1      412      7.47
413  joint                   0.0255    98.475    98.477        1      413      7.37
414  joint                   0.0254    98.474    98.477        1      414      7.58
415  joint                   0.0253    98.479    98.479        1      415      7.63
416  joint                   0.0251    98.482    98.482        1      416      7.58
417  joint                   0.0249    98.483    98.483        1      417      7.60
418  joint                   0.0247    98.484    98.484        1      418      7.53
419  joint                   0.0246    98.488    98.488        1      419      7.43
420  joint                   0.0245    98.490    98.490        1      420      7.51
421  joint                   0.0243    98.493    98.493        1      421      7.43
422  joint                   0.0241    98.492    98.493        1      422      7.50
423  joint                   0.0239    98.493    98.493        1      423      7.47
424  joint                   0.0238    98.494    98.494        1      424      7.60
425  joint                   0.0237    98.497    98.497        1      425      7.38
426  joint                   0.0236    98.505    98.505        1      426      7.59
427  joint                   0.0234    98.505    98.505        1      427      7.55
428  joint                   0.0232    98.509    98.509        1      428      7.41
429  joint                   0.0231    98.512    98.512        1      429      7.49
430  joint                   0.0230    98.513    98.513        1      430      7.61
431  joint                   0.0228    98.514    98.514        1      431      7.57
432  joint                   0.0227    98.518    98.518        1      432      7.66
433  joint                   0.0227    98.518    98.518        1      433      7.73
434  joint                   0.0225    98.519    98.519        1      434      7.59
435  joint                   0.0223    98.524    98.524        1      435      7.61
436  joint                   0.0222    98.522    98.524        1      436      7.57
437  joint                   0.0221    98.519    98.524        1      437      7.60
438  joint                   0.0220    98.517    98.524        1      438      7.47
439  joint                   0.0218    98.521    98.524        1      439      7.63
440  joint                   0.0218    98.526    98.526        1      440      7.56
441  joint                   0.0216    98.529    98.529        1      441      7.57
442  joint                   0.0215    98.536    98.536        1      442      7.51
443  joint                   0.0214    98.546    98.546        1      443      7.53
444  joint                   0.0212    98.551    98.551        1      444      7.48
445  joint                   0.0211    98.549    98.551        1      445      7.52
446  joint                   0.0210    98.554    98.554        1      446      7.53
447  joint                   0.0209    98.555    98.555        1      447      7.62
448  joint                   0.0208    98.548    98.555        1      448      7.62
449  joint                   0.0207    98.559    98.559        1      449      7.54
450  joint                   0.0206    98.554    98.559        1      450      7.73
451  joint                   0.0204    98.558    98.559        1      451      7.50
452  joint                   0.0204    98.556    98.559        1      452      7.65
453  joint                   0.0201    98.558    98.559        1      453      7.44
454  joint                   0.0201    98.554    98.559        1      454      7.60
455  joint                   0.0201    98.561    98.561        1      455      7.34
456  joint                   0.0198    98.560    98.561        1      456      7.54
457  joint                   0.0198    98.549    98.561        1      457      7.44
458  joint                   0.0196    98.549    98.561        1      458      7.33
459  joint                   0.0196    98.553    98.561        1      459      7.51
460  joint                   0.0195    98.555    98.561        1      460      7.67
461  joint                   0.0194    98.554    98.561        1      461      7.41
462  joint                   0.0193    98.558    98.561        1      462      7.55
463  joint                   0.0192    98.564    98.564        1      463      7.48
464  joint                   0.0191    98.574    98.574        1      464      7.49
465  joint                   0.0190    98.570    98.574        1      465      7.44
466  joint                   0.0189    98.570    98.574        1      466      7.61
467  joint                   0.0189    98.576    98.576        1      467      7.63
468  joint                   0.0187    98.578    98.578        1      468      7.65
469  joint                   0.0186    98.575    98.578        1      469      7.63
470  joint                   0.0185    98.569    98.578        1      470      7.50
471  joint                   0.0184    98.569    98.578        1      471      7.33
472  joint                   0.0183    98.573    98.578        1      472      7.47
473  joint                   0.0183    98.567    98.578        1      473      7.63
474  joint                   0.0182    98.567    98.578        1      474      7.66
475  joint                   0.0180    98.569    98.578        1      475      7.62
476  joint                   0.0180    98.577    98.578        1      476      7.46
477  joint                   0.0179    98.575    98.578        1      477      7.66
478  joint                   0.0178    98.579    98.579        1      478      7.59
479  joint                   0.0178    98.569    98.579        1      479      7.55
480  joint                   0.0176    98.570    98.579        1      480      7.68
481  joint                   0.0176    98.582    98.582        1      481      7.38
482  joint                   0.0174    98.575    98.582        1      482      7.47
483  joint                   0.0174    98.579    98.582        1      483      7.58
484  joint                   0.0174    98.589    98.589        1      484      7.44
485  joint                   0.0172    98.586    98.589        1      485      7.59
486  joint                   0.0171    98.588    98.589        1      486      7.60
487  joint                   0.0171    98.595    98.595        1      487      7.61
488  joint                   0.0170    98.595    98.595        1      488      7.73
489  joint                   0.0169    98.601    98.601        1      489      7.44
490  joint                   0.0168    98.608    98.608        1      490      7.44
491  joint                   0.0167    98.605    98.608        1      491      7.54
492  joint                   0.0167    98.599    98.608        1      492      7.54
493  joint                   0.0166    98.609    98.609        1      493      7.52
494  joint                   0.0165    98.610    98.610        1      494      7.44
495  joint                   0.0165    98.599    98.610        1      495      7.56
496  joint                   0.0164    98.602    98.610        1      496      7.43
497  joint                   0.0163    98.611    98.611        1      497      7.55
498  joint                   0.0163    98.614    98.614        1      498      7.58
499  joint                   0.0162    98.609    98.614        1      499      7.58
500  joint                   0.0161    98.610    98.614        1      500      7.50
501  joint                   0.0160    98.612    98.614        1      501      7.61
502  joint                   0.0159    98.608    98.614        1      502      7.52
503  joint                   0.0158    98.609    98.614        1      503      7.42
504  joint                   0.0158    98.606    98.614        1      504      7.66
505  joint                   0.0157    98.604    98.614        1      505      7.49
506  joint                   0.0157    98.612    98.614        1      506      7.56
507  joint                   0.0156    98.614    98.614        1      507      7.49
508  joint                   0.0155    98.611    98.614        1      508      7.21
509  joint                   0.0155    98.608    98.614        1      509      7.62
510  joint                   0.0154    98.614    98.614        1      510      7.55
511  joint                   0.0153    98.622    98.622        1      511      7.68
512  joint                   0.0153    98.623    98.623        1      512      7.57
513  joint                   0.0152    98.621    98.623        1      513      7.47
514  joint                   0.0153    98.616    98.623        1      514      7.55
515  joint                   0.0150    98.612    98.623        1      515      7.54
516  joint                   0.0150    98.619    98.623        1      516      7.64
517  joint                   0.0150    98.620    98.623        1      517      7.63
518  joint                   0.0150    98.618    98.623        1      518      7.61
519  joint                   0.0148    98.615    98.623        1      519      7.44
520  joint                   0.0148    98.622    98.623        1      520      7.59
521  joint                   0.0148    98.623    98.623        1      521      7.41
522  joint                   0.0147    98.628    98.628        1      522      7.46
523  joint                   0.0146    98.625    98.628        1      523      7.48
524  joint                   0.0146    98.618    98.628        1      524      7.44
525  joint                   0.0145    98.621    98.628        1      525      7.52
526  joint                   0.0144    98.622    98.628        1      526      7.65
527  joint                   0.0144    98.615    98.628        1      527      7.58
528  joint                   0.0143    98.610    98.628        1      528      7.69
529  joint                   0.0143    98.614    98.628        1      529      7.55
530  joint                   0.0142    98.616    98.628        1      530      7.58
531  joint                   0.0141    98.622    98.628        1      531      7.69
532  joint                   0.0142    98.628    98.628        1      532      7.50
533  joint                   0.0140    98.622    98.628        1      533      7.53
534  joint                   0.0140    98.622    98.628        1      534      7.67
535  joint                   0.0139    98.633    98.633        1      535      7.49
536  joint                   0.0139    98.635    98.635        1      536      7.44
537  joint                   0.0138    98.634    98.635        1      537      7.66
538  joint                   0.0137    98.630    98.635        1      538      7.53
539  joint                   0.0137    98.628    98.635        1      539      7.72
540  joint                   0.0137    98.628    98.635        1      540      7.46
541  joint                   0.0136    98.633    98.635        1      541      7.65
542  joint                   0.0136    98.628    98.635        1      542      7.55
543  joint                   0.0135    98.620    98.635        1      543      7.46
544  joint                   0.0135    98.630    98.635        1      544      7.56
545  joint                   0.0134    98.638    98.638        1      545      7.47
546  joint                   0.0134    98.635    98.638        1      546      7.47
547  joint                   0.0134    98.629    98.638        1      547      7.55
548  joint                   0.0132    98.629    98.638        1      548      7.50
549  joint                   0.0132    98.633    98.638        1      549      7.50
550  joint                   0.0132    98.643    98.643        1      550      7.64
551  joint                   0.0131    98.646    98.646        1      551      7.54
552  joint                   0.0131    98.643    98.646        1      552      7.60
553  joint                   0.0130    98.632    98.646        1      553      7.59
554  joint                   0.0130    98.641    98.646        1      554      7.64
555  joint                   0.0129    98.649    98.649        1      555      7.45
556  joint                   0.0129    98.648    98.649        1      556      7.36
557  joint                   0.0128    98.632    98.649        1      557      7.65
558  joint                   0.0128    98.633    98.649        1      558      7.55
559  joint                   0.0127    98.631    98.649        1      559      7.36
560  joint                   0.0127    98.642    98.649        1      560      7.59
561  joint                   0.0126    98.635    98.649        1      561      7.43
562  joint                   0.0126    98.625    98.649        1      562      7.54
563  joint                   0.0126    98.627    98.649        1      563      7.51
564  joint                   0.0126    98.629    98.649        1      564      7.43
565  joint                   0.0125    98.621    98.649        1      565      7.69
566  joint                   0.0124    98.614    98.649        1      566      7.46
567  joint                   0.0124    98.621    98.649        1      567      7.66
568  joint                   0.0123    98.631    98.649        1      568      7.60
569  joint                   0.0123    98.638    98.649        1      569      7.58
570  joint                   0.0122    98.641    98.649        1      570      7.58
571  joint                   0.0122    98.639    98.649        1      571      7.60
572  joint                   0.0122    98.630    98.649        1      572      7.58
573  joint                   0.0121    98.631    98.649        1      573      7.45
574  joint                   0.0121    98.644    98.649        1      574      7.52
575  joint                   0.0120    98.643    98.649        1      575      7.39
576  joint                   0.0120    98.633    98.649        1      576      7.57
577  joint                   0.0119    98.629    98.649        1      577      7.65
578  joint                   0.0119    98.644    98.649        1      578      7.56
579  joint                   0.0119    98.647    98.649        1      579      7.74
580  joint                   0.0118    98.642    98.649        1      580      7.43
581  joint                   0.0118    98.641    98.649        1      581      7.56
582  joint                   0.0118    98.638    98.649        1      582      7.57
583  joint                   0.0117    98.632    98.649        1      583      7.43
584  joint                   0.0117    98.637    98.649        1      584      7.39
585  joint                   0.0117    98.638    98.649        1      585      7.54
586  joint                   0.0116    98.633    98.649        1      586      7.73
587  joint                   0.0116    98.635    98.649        1      587      7.50
588  joint                   0.0115    98.649    98.649        1      588      7.65
589  joint                   0.0115    98.650    98.650        1      589      7.68
590  joint                   0.0115    98.642    98.650        1      590      7.77
591  joint                   0.0114    98.640    98.650        1      591      7.57
592  joint                   0.0114    98.642    98.650        1      592      7.67
593  joint                   0.0114    98.645    98.650        1      593      7.53
594  joint                   0.0113    98.645    98.650        1      594      7.52
595  joint                   0.0113    98.646    98.650        1      595      7.50
596  joint                   0.0113    98.641    98.650        1      596      7.53
597  joint                   0.0112    98.637    98.650        1      597      7.48
598  joint                   0.0112    98.645    98.650        1      598      7.51
599  joint                   0.0111    98.648    98.650        1      599      7.57
600  joint                   0.0111    98.650    98.650        1      600      7.64 *
```

</details>

<details>
<summary>ogbn-arxiv | reference | fixed_c — 전체 에포크</summary>

```text
[ogbn-arxiv | reference | fixed_c]
status=passed | last_epoch=201 | selected_epoch=13
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.8439    49.757    49.757       12       12     14.02
  2  joint                   1.7609    60.794    60.794       12       24     13.18
  3  joint                   1.3106    66.331    66.331       12       36     13.25
  4  joint                   1.1339    68.301    68.301       12       48     13.25
  5  joint                   1.0507    67.230    68.301       12       60     13.30
  6  joint                   0.9962    69.043    69.043       12       72     13.28
  7  joint                   0.9412    69.533    69.533       12       84     13.26
  8  joint                   0.9108    70.559    70.559       12       96     13.29
  9  joint                   0.8735    70.231    70.559       12      108     13.32
 10  joint                   0.8459    70.415    70.559       12      120     13.29
 11  joint                   0.8196    70.992    70.992       12      132     13.31
 12  joint                   0.7916    71.254    71.254       12      144     13.27
 13  joint                   0.7605    71.549    71.549       12      156     13.25 *
 14  joint                   0.7309    71.049    71.549       12      168     13.31
 15  joint                   0.7105    70.707    71.549       12      180     13.25
 16  joint                   0.6743    70.875    71.549       12      192     13.27
 17  joint                   0.6506    70.922    71.549       12      204     13.29
 18  joint                   0.6194    69.653    71.549       12      216     13.27
 19  joint                   0.5883    69.925    71.549       12      228     13.29
 20  joint                   0.5584    69.257    71.549       12      240     13.32
 21  joint                   0.5341    68.321    71.549       12      252     13.42
 22  joint                   0.5020    70.116    71.549       12      264     13.34
 23  joint                   0.4652    70.371    71.549       12      276     13.30
 24  joint                   0.4461    69.100    71.549       12      288     13.29
 25  joint                   0.4309    68.949    71.549       12      300     13.27
 26  joint                   0.4130    68.865    71.549       12      312     13.30
 27  joint                   0.3841    68.526    71.549       12      324     13.32
 28  joint                   0.3640    68.499    71.549       12      336     13.29
 29  joint                   0.3510    69.016    71.549       12      348     13.30
 30  joint                   0.3401    69.026    71.549       12      360     13.29
 31  joint                   0.3256    68.616    71.549       12      372     13.28
 32  joint                   0.2999    67.841    71.549       12      384     13.28
 33  joint                   0.2787    68.200    71.549       12      396     13.28
 34  joint                   0.2651    67.952    71.549       12      408     13.29
 35  joint                   0.2506    68.952    71.549       12      420     13.30
 36  joint                   0.2413    68.345    71.549       12      432     13.28
 37  joint                   0.2413    66.808    71.549       12      444     13.27
 38  joint                   0.2294    68.680    71.549       12      456     13.27
 39  joint                   0.2145    68.482    71.549       12      468     13.27
 40  joint                   0.2075    67.492    71.549       12      480     13.29
 41  joint                   0.1997    68.204    71.549       12      492     13.30
 42  joint                   0.1902    68.361    71.549       12      504     13.27
 43  joint                   0.1763    67.979    71.549       12      516     13.27
 44  joint                   0.1750    67.912    71.549       12      528     13.28
 45  joint                   0.1706    67.989    71.549       12      540     13.30
 46  joint                   0.1665    68.288    71.549       12      552     13.30
 47  joint                   0.1568    67.767    71.549       12      564     13.29
 48  joint                   0.1516    67.566    71.549       12      576     13.26
 49  joint                   0.1474    68.328    71.549       12      588     13.29
 50  joint                   0.1490    67.697    71.549       12      600     13.27
 51  joint                   0.1381    67.509    71.549       12      612     13.33
 52  joint                   0.1285    68.079    71.549       12      624     13.27
 53  joint                   0.1270    67.922    71.549       12      636     13.31
 54  joint                   0.1243    67.496    71.549       12      648     13.27
 55  joint                   0.1198    67.455    71.549       12      660     13.28
 56  joint                   0.1214    67.539    71.549       12      672     13.30
 57  joint                   0.1235    67.720    71.549       12      684     13.33
 58  joint                   0.1186    67.532    71.549       12      696     13.26
 59  joint                   0.1156    66.908    71.549       12      708     13.30
 60  joint                   0.1120    67.677    71.549       12      720     13.29
 61  joint                   0.1091    67.586    71.549       12      732     13.26
 62  joint                   0.1066    67.449    71.549       12      744     13.26
 63  joint                   0.1021    67.331    71.549       12      756     13.29
 64  joint                   0.1005    67.747    71.549       12      768     13.26
 65  joint                   0.0945    67.741    71.549       12      780     13.27
 66  joint                   0.0935    68.053    71.549       12      792     13.28
 67  joint                   0.0917    67.751    71.549       12      804     13.27
 68  joint                   0.0870    67.834    71.549       12      816     13.25
 69  joint                   0.0917    67.734    71.549       12      828     13.30
 70  joint                   0.0888    67.626    71.549       12      840     13.30
 71  joint                   0.0901    67.892    71.549       12      852     13.27
 72  joint                   0.0848    68.093    71.549       12      864     13.28
 73  joint                   0.0848    68.009    71.549       12      876     13.27
 74  joint                   0.0827    67.371    71.549       12      888     13.28
 75  joint                   0.0821    67.871    71.549       12      900     13.30
 76  joint                   0.0835    68.106    71.549       12      912     13.30
 77  joint                   0.0845    68.090    71.549       12      924     13.30
 78  joint                   0.0772    67.720    71.549       12      936     13.34
 79  joint                   0.0769    67.214    71.549       12      948     13.28
 80  joint                   0.0831    67.566    71.549       12      960     13.42
 81  joint                   0.0761    67.016    71.549       12      972     13.32
 82  joint                   0.0701    67.126    71.549       12      984     13.28
 83  joint                   0.0778    67.378    71.549       12      996     13.25
 84  joint                   0.0717    67.475    71.549       12     1008     13.38
 85  joint                   0.0747    68.157    71.549       12     1020     13.28
 86  joint                   0.0710    67.858    71.549       12     1032     13.28
 87  joint                   0.0681    68.016    71.549       12     1044     13.30
 88  joint                   0.0685    67.922    71.549       12     1056     13.28
 89  joint                   0.0636    67.828    71.549       12     1068     13.42
 90  joint                   0.0725    68.331    71.549       12     1080     13.30
 91  joint                   0.0751    67.700    71.549       12     1092     13.26
 92  joint                   0.0649    67.962    71.549       12     1104     13.30
 93  joint                   0.0664    67.529    71.549       12     1116     13.25
 94  joint                   0.0637    67.647    71.549       12     1128     13.31
 95  joint                   0.0643    68.079    71.549       12     1140     13.27
 96  joint                   0.0628    67.412    71.549       12     1152     13.32
 97  joint                   0.0620    67.939    71.549       12     1164     13.30
 98  joint                   0.0592    67.254    71.549       12     1176     13.27
 99  joint                   0.0595    67.324    71.549       12     1188     13.29
100  joint                   0.0588    67.647    71.549       12     1200     13.28
101  joint                   0.0572    67.808    71.549       12     1212     13.33
102  joint                   0.0642    67.784    71.549       12     1224     13.30
103  joint                   0.0611    67.600    71.549       12     1236     13.28
104  joint                   0.0568    67.881    71.549       12     1248     13.26
105  joint                   0.0566    67.546    71.549       12     1260     13.30
106  joint                   0.0557    67.613    71.549       12     1272     13.36
107  joint                   0.0549    68.019    71.549       12     1284     13.26
108  joint                   0.0583    67.955    71.549       12     1296     13.27
109  joint                   0.0526    67.680    71.549       12     1308     13.29
110  joint                   0.0552    67.704    71.549       12     1320     13.29
111  joint                   0.0569    67.892    71.549       12     1332     13.27
112  joint                   0.0579    68.012    71.549       12     1344     13.27
113  joint                   0.0480    68.274    71.549       12     1356     13.30
114  joint                   0.0442    68.183    71.549       12     1368     13.29
115  joint                   0.0477    68.079    71.549       12     1380     13.29
116  joint                   0.0441    68.137    71.549       12     1392     13.28
117  joint                   0.0479    67.519    71.549       12     1404     13.27
118  joint                   0.0511    68.073    71.549       12     1416     13.29
119  joint                   0.0515    67.868    71.549       12     1428     13.30
120  joint                   0.0512    67.908    71.549       12     1440     13.27
121  joint                   0.0485    68.012    71.549       12     1452     13.29
122  joint                   0.0479    67.422    71.549       12     1464     13.25
123  joint                   0.0502    67.647    71.549       12     1476     13.44
124  joint                   0.0517    68.026    71.549       12     1488     13.32
125  joint                   0.0478    68.254    71.549       12     1500     13.28
126  joint                   0.0495    68.251    71.549       12     1512     13.29
127  joint                   0.0409    67.408    71.549       12     1524     13.42
128  joint                   0.0409    67.428    71.549       12     1536     13.25
129  joint                   0.0467    68.187    71.549       12     1548     13.29
130  joint                   0.0455    68.439    71.549       12     1560     13.31
131  joint                   0.0405    67.808    71.549       12     1572     13.29
132  joint                   0.0437    67.848    71.549       12     1584     13.27
133  joint                   0.0446    68.224    71.549       12     1596     13.27
134  joint                   0.0434    68.492    71.549       12     1608     13.35
135  joint                   0.0426    68.103    71.549       12     1620     13.43
136  joint                   0.0412    68.328    71.549       12     1632     13.30
137  joint                   0.0449    67.939    71.549       12     1644     13.25
138  joint                   0.0467    68.110    71.549       12     1656     13.27
139  joint                   0.0409    68.016    71.549       12     1668     13.27
140  joint                   0.0408    68.120    71.549       12     1680     13.27
141  joint                   0.0392    68.049    71.549       12     1692     13.31
142  joint                   0.0454    68.180    71.549       12     1704     13.27
143  joint                   0.0432    68.016    71.549       12     1716     13.30
144  joint                   0.0434    68.039    71.549       12     1728     13.29
145  joint                   0.0393    68.707    71.549       12     1740     13.28
146  joint                   0.0375    68.116    71.549       12     1752     13.28
147  joint                   0.0402    67.959    71.549       12     1764     13.45
148  joint                   0.0432    67.858    71.549       12     1776     13.28
149  joint                   0.0438    67.791    71.549       12     1788     13.30
150  joint                   0.0407    68.475    71.549       12     1800     13.43
151  joint                   0.0443    68.116    71.549       12     1812     13.30
152  joint                   0.0421    67.287    71.549       12     1824     13.28
153  joint                   0.0419    68.288    71.549       12     1836     13.42
154  joint                   0.0407    68.157    71.549       12     1848     13.27
155  joint                   0.0355    68.492    71.549       12     1860     13.30
156  joint                   0.0376    68.600    71.549       12     1872     13.28
157  joint                   0.0407    68.093    71.549       12     1884     13.27
158  joint                   0.0405    68.271    71.549       12     1896     13.27
159  joint                   0.0391    67.945    71.549       12     1908     13.28
160  joint                   0.0374    67.499    71.549       12     1920     13.26
161  joint                   0.0371    68.160    71.549       12     1932     13.25
162  joint                   0.0389    67.996    71.549       12     1944     13.30
163  joint                   0.0330    67.861    71.549       12     1956     13.31
164  joint                   0.0352    68.022    71.549       12     1968     13.29
165  joint                   0.0364    67.932    71.549       12     1980     13.30
166  joint                   0.0361    68.428    71.549       12     1992     13.31
167  joint                   0.0380    67.858    71.549       12     2004     13.27
168  joint                   0.0382    68.167    71.549       12     2016     13.32
169  joint                   0.0411    68.100    71.549       12     2028     13.26
170  joint                   0.0388    68.318    71.549       12     2040     13.25
171  joint                   0.0369    67.962    71.549       12     2052     13.30
172  joint                   0.0392    68.314    71.549       12     2064     13.26
173  joint                   0.0361    67.915    71.549       12     2076     13.29
174  joint                   0.0383    68.267    71.549       12     2088     13.28
175  joint                   0.0376    67.690    71.549       12     2100     13.32
176  joint                   0.0339    68.167    71.549       12     2112     13.28
177  joint                   0.0354    68.405    71.549       12     2124     13.27
178  joint                   0.0356    68.147    71.549       12     2136     13.29
179  joint                   0.0339    67.606    71.549       12     2148     13.25
180  joint                   0.0317    67.959    71.549       12     2160     13.26
181  joint                   0.0330    68.603    71.549       12     2172     13.33
182  joint                   0.0332    68.465    71.549       12     2184     13.38
183  joint                   0.0316    68.311    71.549       12     2196     13.27
184  joint                   0.0331    68.475    71.549       12     2208     13.32
185  joint                   0.0349    67.714    71.549       12     2220     13.35
186  joint                   0.0338    67.878    71.549       12     2232     13.25
187  joint                   0.0333    67.794    71.549       12     2244     13.30
188  joint                   0.0342    68.073    71.549       12     2256     13.27
189  joint                   0.0318    68.126    71.549       12     2268     13.27
190  joint                   0.0319    67.865    71.549       12     2280     13.27
191  joint                   0.0327    67.586    71.549       12     2292     13.29
192  joint                   0.0340    67.895    71.549       12     2304     13.38
193  joint                   0.0342    67.539    71.549       12     2316     13.29
194  joint                   0.0333    67.794    71.549       12     2328     13.38
195  joint                   0.0317    68.257    71.549       12     2340     13.30
196  joint                   0.0340    68.029    71.549       12     2352     13.27
197  joint                   0.0315    67.989    71.549       12     2364     13.31
198  joint                   0.0325    68.100    71.549       12     2376     13.37
199  joint                   0.0321    68.059    71.549       12     2388     13.27
200  joint                   0.0311    67.996    71.549       12     2400     13.30
201  joint                   0.0294    67.888    71.549       12     2412     13.26
```

</details>

<details>
<summary>ogbn-arxiv | reference | shared_dynamic_c — 전체 에포크</summary>

```text
[ogbn-arxiv | reference | shared_dynamic_c]
status=passed | last_epoch=201 | selected_epoch=13
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.8409    49.723    49.723       12       12    667.93
  2  joint                   1.7352    61.777    61.777       12       24    667.62
  3  joint                   1.2890    66.039    66.039       12       36    667.84
  4  joint                   1.1181    68.992    68.992       12       48    667.86
  5  joint                   1.0320    68.616    68.992       12       60    667.59
  6  joint                   0.9720    68.637    68.992       12       72    668.00
  7  joint                   0.9223    70.489    70.489       12       84    668.28
  8  joint                   0.8908    70.996    70.996       12       96    667.44
  9  joint                   0.8579    71.076    71.076       12      108    670.73
 10  joint                   0.8218    71.214    71.214       12      120    667.47
 11  joint                   0.7895    71.190    71.214       12      132    667.90
 12  joint                   0.7640    71.076    71.214       12      144    667.16
 13  joint                   0.7284    71.700    71.700       12      156    667.03 *
 14  joint                   0.6948    70.274    71.700       12      168    667.14
 15  joint                   0.6742    69.502    71.700       12      180    666.82
 16  joint                   0.6371    70.506    71.700       12      192    667.76
 17  joint                   0.6122    70.200    71.700       12      204    667.98
 18  joint                   0.5719    69.016    71.700       12      216    667.36
 19  joint                   0.5368    69.190    71.700       12      228    667.20
 20  joint                   0.5176    69.321    71.700       12      240    667.64
 21  joint                   0.4844    68.818    71.700       12      252    668.96
 22  joint                   0.4555    69.519    71.700       12      264    666.91
 23  joint                   0.4303    69.969    71.700       12      276    667.14
 24  joint                   0.4087    68.277    71.700       12      288    667.33
 25  joint                   0.3833    68.932    71.700       12      300    668.20
 26  joint                   0.3629    68.509    71.700       12      312    668.21
 27  joint                   0.3425    68.210    71.700       12      324    666.76
 28  joint                   0.3250    68.637    71.700       12      336    667.69
 29  joint                   0.3074    68.559    71.700       12      348    667.21
 30  joint                   0.2930    68.261    71.700       12      360    667.08
 31  joint                   0.2794    68.143    71.700       12      372    667.21
 32  joint                   0.2591    68.432    71.700       12      384    666.34
 33  joint                   0.2465    68.630    71.700       12      396    664.80
 34  joint                   0.2382    67.892    71.700       12      408    664.50
 35  joint                   0.2326    68.734    71.700       12      420    667.09
 36  joint                   0.2243    68.465    71.700       12      432    667.64
 37  joint                   0.2115    67.784    71.700       12      444    667.77
 38  joint                   0.2022    68.673    71.700       12      456    666.88
 39  joint                   0.1938    68.969    71.700       12      468    667.13
 40  joint                   0.1851    68.244    71.700       12      480    666.96
 41  joint                   0.1747    67.932    71.700       12      492    667.66
 42  joint                   0.1631    67.670    71.700       12      504    667.42
 43  joint                   0.1586    68.126    71.700       12      516    665.65
 44  joint                   0.1544    67.677    71.700       12      528    667.93
 45  joint                   0.1593    68.465    71.700       12      540    667.51
 46  joint                   0.1505    68.294    71.700       12      552    668.04
 47  joint                   0.1443    68.311    71.700       12      564    668.03
 48  joint                   0.1414    68.428    71.700       12      576    667.00
 49  joint                   0.1303    68.509    71.700       12      588    667.65
 50  joint                   0.1253    68.626    71.700       12      600    667.03
 51  joint                   0.1205    68.251    71.700       12      612    667.14
 52  joint                   0.1226    68.371    71.700       12      624    665.91
 53  joint                   0.1175    68.194    71.700       12      636    665.10
 54  joint                   0.1143    68.684    71.700       12      648    664.51
 55  joint                   0.1100    67.143    71.700       12      660    665.16
 56  joint                   0.1125    68.637    71.700       12      672    667.53
 57  joint                   0.1102    67.734    71.700       12      684    667.72
 58  joint                   0.1100    67.962    71.700       12      696    666.91
 59  joint                   0.1095    68.680    71.700       12      708    667.35
 60  joint                   0.1034    67.261    71.700       12      720    667.37
 61  joint                   0.1063    68.204    71.700       12      732    667.11
 62  joint                   0.0936    68.583    71.700       12      744    666.91
 63  joint                   0.0920    68.730    71.700       12      756    667.76
 64  joint                   0.0896    68.157    71.700       12      768    666.97
 65  joint                   0.0923    68.328    71.700       12      780    667.92
 66  joint                   0.0909    67.925    71.700       12      792    667.24
 67  joint                   0.0899    68.600    71.700       12      804    667.02
 68  joint                   0.0847    68.516    71.700       12      816    667.26
 69  joint                   0.0834    67.432    71.700       12      828    668.49
 70  joint                   0.0905    68.687    71.700       12      840    670.89
 71  joint                   0.0835    68.153    71.700       12      852    667.20
 72  joint                   0.0878    68.432    71.700       12      864    667.31
 73  joint                   0.0866    68.224    71.700       12      876    668.24
 74  joint                   0.0815    67.771    71.700       12      888    668.04
 75  joint                   0.0830    68.053    71.700       12      900    667.44
 76  joint                   0.0803    67.724    71.700       12      912    667.54
 77  joint                   0.0820    68.972    71.700       12      924    667.54
 78  joint                   0.0753    68.392    71.700       12      936     30.86
 79  joint                   0.0753    67.878    71.700       12      948     29.13
 80  joint                   0.0761    68.660    71.700       12      960     29.00
 81  joint                   0.0759    68.214    71.700       12      972     28.93
 82  joint                   0.0752    68.311    71.700       12      984     28.98
 83  joint                   0.0761    68.626    71.700       12      996     29.01
 84  joint                   0.0743    68.311    71.700       12     1008     29.30
 85  joint                   0.0744    68.553    71.700       12     1020     29.10
 86  joint                   0.0713    68.788    71.700       12     1032     28.96
 87  joint                   0.0640    68.281    71.700       12     1044     29.09
 88  joint                   0.0690    68.690    71.700       12     1056     29.10
 89  joint                   0.0673    68.583    71.700       12     1068     28.94
 90  joint                   0.0636    68.502    71.700       12     1080     28.99
 91  joint                   0.0690    69.096    71.700       12     1092     29.20
 92  joint                   0.0717    68.328    71.700       12     1104     29.03
 93  joint                   0.0659    68.871    71.700       12     1116     28.98
 94  joint                   0.0639    67.730    71.700       12     1128     28.95
 95  joint                   0.0615    68.714    71.700       12     1140     28.99
 96  joint                   0.0612    68.791    71.700       12     1152     29.10
 97  joint                   0.0618    68.351    71.700       12     1164     28.95
 98  joint                   0.0665    68.026    71.700       12     1176     29.02
 99  joint                   0.0640    68.167    71.700       12     1188     28.97
100  joint                   0.0637    68.355    71.700       12     1200     29.01
101  joint                   0.0624    68.747    71.700       12     1212     29.24
102  joint                   0.0631    68.230    71.700       12     1224     28.96
103  joint                   0.0627    68.328    71.700       12     1236     29.04
104  joint                   0.0568    68.553    71.700       12     1248     28.95
105  joint                   0.0554    68.741    71.700       12     1260     28.99
106  joint                   0.0516    67.865    71.700       12     1272     28.98
107  joint                   0.0535    68.512    71.700       12     1284     28.98
108  joint                   0.0494    68.553    71.700       12     1296     29.07
109  joint                   0.0465    68.942    71.700       12     1308     28.96
110  joint                   0.0504    68.767    71.700       12     1320     29.07
111  joint                   0.0524    67.969    71.700       12     1332     28.97
112  joint                   0.0568    68.465    71.700       12     1344     29.02
113  joint                   0.0531    68.791    71.700       12     1356     29.00
114  joint                   0.0475    68.808    71.700       12     1368     29.03
115  joint                   0.0493    67.798    71.700       12     1380     29.00
116  joint                   0.0503    68.314    71.700       12     1392     29.42
117  joint                   0.0481    68.170    71.700       12     1404     29.15
118  joint                   0.0475    68.687    71.700       12     1416     29.09
119  joint                   0.0524    68.093    71.700       12     1428     29.08
120  joint                   0.0503    68.227    71.700       12     1440     29.31
121  joint                   0.0538    67.902    71.700       12     1452     29.27
122  joint                   0.0550    67.788    71.700       12     1464     29.04
123  joint                   0.0545    68.526    71.700       12     1476     29.00
124  joint                   0.0526    68.710    71.700       12     1488     29.09
125  joint                   0.0532    68.556    71.700       12     1500     29.05
126  joint                   0.0533    68.603    71.700       12     1512     29.06
127  joint                   0.0493    68.388    71.700       12     1524     29.22
128  joint                   0.0460    68.355    71.700       12     1536     29.18
129  joint                   0.0405    68.439    71.700       12     1548     29.15
130  joint                   0.0390    68.788    71.700       12     1560     29.00
131  joint                   0.0410    68.194    71.700       12     1572     29.06
132  joint                   0.0395    68.110    71.700       12     1584     29.01
133  joint                   0.0403    68.361    71.700       12     1596     29.03
134  joint                   0.0381    68.422    71.700       12     1608     29.02
135  joint                   0.0402    68.130    71.700       12     1620     28.99
136  joint                   0.0459    68.036    71.700       12     1632     29.11
137  joint                   0.0462    68.657    71.700       12     1644     29.01
138  joint                   0.0484    67.643    71.700       12     1656     29.10
139  joint                   0.0433    67.996    71.700       12     1668     29.30
140  joint                   0.0443    68.835    71.700       12     1680     29.38
141  joint                   0.0469    68.375    71.700       12     1692     29.09
142  joint                   0.0436    68.428    71.700       12     1704     29.00
143  joint                   0.0416    68.177    71.700       12     1716     29.05
144  joint                   0.0445    68.479    71.700       12     1728     29.11
145  joint                   0.0436    68.304    71.700       12     1740     29.09
146  joint                   0.0379    68.747    71.700       12     1752     29.00
147  joint                   0.0387    68.663    71.700       12     1764     28.97
148  joint                   0.0374    67.949    71.700       12     1776     29.35
149  joint                   0.0367    68.855    71.700       12     1788     28.99
150  joint                   0.0386    68.475    71.700       12     1800     29.19
151  joint                   0.0410    68.355    71.700       12     1812     28.94
152  joint                   0.0410    68.277    71.700       12     1824     28.99
153  joint                   0.0434    68.556    71.700       12     1836     28.91
154  joint                   0.0393    68.173    71.700       12     1848     29.06
155  joint                   0.0371    68.667    71.700       12     1860     28.97
156  joint                   0.0383    68.083    71.700       12     1872     29.00
157  joint                   0.0422    68.100    71.700       12     1884     29.16
158  joint                   0.0471    68.083    71.700       12     1896     28.95
159  joint                   0.0386    67.979    71.700       12     1908     28.98
160  joint                   0.0349    68.096    71.700       12     1920     28.98
161  joint                   0.0366    68.643    71.700       12     1932     29.01
162  joint                   0.0393    68.398    71.700       12     1944     28.96
163  joint                   0.0322    68.529    71.700       12     1956     29.09
164  joint                   0.0348    68.553    71.700       12     1968     28.99
165  joint                   0.0409    68.314    71.700       12     1980     29.01
166  joint                   0.0403    68.392    71.700       12     1992     28.94
167  joint                   0.0385    68.237    71.700       12     2004     29.02
168  joint                   0.0369    68.435    71.700       12     2016     28.96
169  joint                   0.0330    68.428    71.700       12     2028     29.02
170  joint                   0.0333    68.707    71.700       12     2040     28.99
171  joint                   0.0327    68.516    71.700       12     2052     29.07
172  joint                   0.0366    67.663    71.700       12     2064     28.95
173  joint                   0.0340    68.704    71.700       12     2076     29.00
174  joint                   0.0334    68.663    71.700       12     2088     29.22
175  joint                   0.0409    69.147    71.700       12     2100     29.37
176  joint                   0.0395    68.496    71.700       12     2112     29.52
177  joint                   0.0393    68.284    71.700       12     2124     29.15
178  joint                   0.0375    68.106    71.700       12     2136     29.12
179  joint                   0.0370    68.519    71.700       12     2148     28.99
180  joint                   0.0383    68.173    71.700       12     2160     29.09
181  joint                   0.0357    68.942    71.700       12     2172     29.03
182  joint                   0.0355    68.375    71.700       12     2184     29.10
183  joint                   0.0323    68.519    71.700       12     2196     28.98
184  joint                   0.0311    68.637    71.700       12     2208     29.04
185  joint                   0.0273    68.640    71.700       12     2220     29.01
186  joint                   0.0274    68.835    71.700       12     2232     29.04
187  joint                   0.0325    68.801    71.700       12     2244     29.02
188  joint                   0.0368    68.653    71.700       12     2256     29.27
189  joint                   0.0323    68.794    71.700       12     2268     29.32
190  joint                   0.0317    68.767    71.700       12     2280     29.19
191  joint                   0.0319    68.318    71.700       12     2292     29.20
192  joint                   0.0335    68.230    71.700       12     2304     29.00
193  joint                   0.0359    68.381    71.700       12     2316     29.33
194  joint                   0.0371    68.445    71.700       12     2328     29.01
195  joint                   0.0323    69.039    71.700       12     2340     29.62
196  joint                   0.0299    68.784    71.700       12     2352     29.08
197  joint                   0.0305    68.667    71.700       12     2364     29.20
198  joint                   0.0318    68.472    71.700       12     2376     29.05
199  joint                   0.0347    68.147    71.700       12     2388     29.09
200  joint                   0.0366    68.532    71.700       12     2400     29.02
201  joint                   0.0293    68.442    71.700       12     2412     29.10
```

</details>

<details>
<summary>ogbn-arxiv | large | fixed_c — 전체 에포크</summary>

```text
[ogbn-arxiv | large | fixed_c]
status=passed | last_epoch=54 | selected_epoch=4
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.2369    63.301    63.301       45       45     27.86
  2  joint                   1.1746    68.479    68.479       45       90     27.00
  3  joint                   1.0212    68.945    68.945       45      135     27.02
  4  joint                   0.9381    71.838    71.838       45      180     26.99 *
  5  joint                   0.8710    69.751    71.838       45      225     26.99
  6  joint                   0.8163    71.049    71.838       45      270     26.98
  7  joint                   0.7531    71.704    71.838       45      315     27.03
  8  joint                   0.6970    69.392    71.838       45      360     27.01
  9  joint                   0.6422    69.039    71.838       45      405     27.00
 10  joint                   0.5759    69.392    71.838       45      450     27.03
 11  joint                   0.5223    69.429    71.838       45      495     26.98
 12  joint                   0.4635    67.502    71.838       45      540     27.01
 13  joint                   0.4095    68.452    71.838       45      585     26.99
 14  joint                   0.3600    68.637    71.838       45      630     26.98
 15  joint                   0.3179    66.787    71.838       45      675     27.04
 16  joint                   0.2795    66.006    71.838       45      720     27.00
 17  joint                   0.2471    66.875    71.838       45      765     27.09
 18  joint                   0.2215    67.063    71.838       45      810     27.01
 19  joint                   0.1993    67.284    71.838       45      855     27.18
 20  joint                   0.1754    66.405    71.838       45      900     26.99
 21  joint                   0.1580    66.633    71.838       45      945     26.99
 22  joint                   0.1469    67.214    71.838       45      990     26.97
 23  joint                   0.1351    66.301    71.838       45     1035     26.98
 24  joint                   0.1272    66.277    71.838       45     1080     26.99
 25  joint                   0.1148    66.892    71.838       45     1125     26.99
 26  joint                   0.1081    67.241    71.838       45     1170     27.01
 27  joint                   0.1053    66.935    71.838       45     1215     26.97
 28  joint                   0.0968    66.002    71.838       45     1260     27.02
 29  joint                   0.0826    66.449    71.838       45     1305     27.01
 30  joint                   0.0863    67.237    71.838       45     1350     27.04
 31  joint                   0.0790    67.150    71.838       45     1395     27.05
 32  joint                   0.0735    66.895    71.838       45     1440     27.02
 33  joint                   0.0758    67.452    71.838       45     1485     27.03
 34  joint                   0.0651    66.553    71.838       45     1530     27.00
 35  joint                   0.0710    66.341    71.838       45     1575     26.99
 36  joint                   0.0710    67.163    71.838       45     1620     26.97
 37  joint                   0.0626    67.066    71.838       45     1665     26.97
 38  joint                   0.0594    67.841    71.838       45     1710     26.97
 39  joint                   0.0569    67.281    71.838       45     1755     26.99
 40  joint                   0.0556    67.298    71.838       45     1800     27.02
 41  joint                   0.0595    66.892    71.838       45     1845     26.98
 42  joint                   0.0554    67.173    71.838       45     1890     26.96
 43  joint                   0.0501    66.744    71.838       45     1935     26.99
 44  joint                   0.0472    66.918    71.838       45     1980     26.97
 45  joint                   0.0509    67.365    71.838       45     2025     27.03
 46  joint                   0.0480    67.140    71.838       45     2070     27.05
 47  joint                   0.0457    67.059    71.838       45     2115     27.03
 48  joint                   0.0472    67.499    71.838       45     2160     26.99
 49  joint                   0.0479    67.133    71.838       45     2205     26.97
 50  joint                   0.0468    67.136    71.838       45     2250     27.01
 51  joint                   0.0435    66.848    71.838       45     2295     26.97
 52  joint                   0.0454    66.821    71.838       45     2340     27.05
 53  joint                   0.0447    67.234    71.838       45     2385     27.22
 54  joint                   0.0431    67.043    71.838       45     2430     26.98
```

</details>

<details>
<summary>ogbn-arxiv | large | shared_dynamic_c — 전체 에포크</summary>

```text
[ogbn-arxiv | large | shared_dynamic_c]
status=passed | last_epoch=54 | selected_epoch=4
 Ep  Phase                TrainLoss    Val(%)   Peak(%)  Batches    Steps  Epoch(s)  Selected
  1  joint                   2.2911    60.026    60.026       45       45     78.18
  2  joint                   1.2130    67.412    67.412       45       90     77.00
  3  joint                   1.0416    70.267    70.267       45      135     77.14
  4  joint                   0.9488    71.898    71.898       45      180     76.67 *
  5  joint                   0.8827    70.637    71.898       45      225     77.08
  6  joint                   0.8251    71.130    71.898       45      270     76.69
  7  joint                   0.7611    70.771    71.898       45      315     76.65
  8  joint                   0.7016    68.445    71.898       45      360     76.64
  9  joint                   0.6469    69.653    71.898       45      405     76.77
 10  joint                   0.5844    69.090    71.898       45      450     76.78
 11  joint                   0.5263    69.197    71.898       45      495     76.54
 12  joint                   0.4643    67.277    71.898       45      540     76.63
 13  joint                   0.4106    68.204    71.898       45      585     76.70
 14  joint                   0.3673    68.133    71.898       45      630     76.66
 15  joint                   0.3243    66.650    71.898       45      675     76.62
 16  joint                   0.2820    67.707    71.898       45      720     76.67
 17  joint                   0.2522    67.120    71.898       45      765     76.73
 18  joint                   0.2308    66.687    71.898       45      810     76.64
 19  joint                   0.2105    67.479    71.898       45      855     76.69
 20  joint                   0.1813    67.100    71.898       45      900     76.87
 21  joint                   0.1671    67.469    71.898       45      945     76.68
 22  joint                   0.1507    67.861    71.898       45      990     76.82
 23  joint                   0.1295    66.781    71.898       45     1035     76.64
 24  joint                   0.1276    66.955    71.898       45     1080     76.56
 25  joint                   0.1201    66.781    71.898       45     1125     76.86
 26  joint                   0.1148    67.126    71.898       45     1170     76.70
 27  joint                   0.1069    67.036    71.898       45     1215     76.70
 28  joint                   0.1018    65.985    71.898       45     1260     76.84
 29  joint                   0.0887    67.334    71.898       45     1305     76.57
 30  joint                   0.0840    67.301    71.898       45     1350     76.68
 31  joint                   0.0825    66.868    71.898       45     1395     76.79
 32  joint                   0.0769    66.959    71.898       45     1440     76.69
 33  joint                   0.0712    67.452    71.898       45     1485     76.86
 34  joint                   0.0691    67.351    71.898       45     1530     76.57
 35  joint                   0.0729    67.039    71.898       45     1575     76.78
 36  joint                   0.0704    67.402    71.898       45     1620     76.79
 37  joint                   0.0626    66.740    71.898       45     1665     76.82
 38  joint                   0.0575    67.838    71.898       45     1710     76.56
 39  joint                   0.0594    67.301    71.898       45     1755     76.76
 40  joint                   0.0595    67.845    71.898       45     1800     76.88
 41  joint                   0.0583    66.536    71.898       45     1845     76.75
 42  joint                   0.0570    67.633    71.898       45     1890     76.66
 43  joint                   0.0551    67.509    71.898       45     1935     76.84
 44  joint                   0.0556    66.700    71.898       45     1980     76.52
 45  joint                   0.0515    67.932    71.898       45     2025     76.85
 46  joint                   0.0498    67.942    71.898       45     2070     77.03
 47  joint                   0.0461    67.271    71.898       45     2115     76.82
 48  joint                   0.0483    67.781    71.898       45     2160     76.61
 49  joint                   0.0438    67.908    71.898       45     2205     76.65
 50  joint                   0.0467    67.848    71.898       45     2250     76.66
 51  joint                   0.0443    67.385    71.898       45     2295     76.68
 52  joint                   0.0413    67.039    71.898       45     2340     76.79
 53  joint                   0.0408    67.613    71.898       45     2385     76.60
 54  joint                   0.0426    67.378    71.898       45     2430     77.11
```

</details>

## 과거 기록 — 아래 절은 각 날짜 당시 상태

## 2026-09-07 당시: corrected V5 실행 시간과 병목 수정

수령 경로: corrected-c-v5-a6000-gpu3-seed0-v1-conductance /
v5/reference/model-seed-0/ogbn-arxiv/shared_dynamic_c/history.json.
epoch 62–66은 각각 12 train batches, 666.9104/667.7595/666.9653/667.9185/667.2440초,
optimizer_steps 744/756/768/780/792다. 평균은 667.3595초/epoch다.
해당 필드는 train+validation을 포함하므로 순수 batch time이나 GPU kernel time이 아니다.
같은 시점 제공된 RTX A6000 관측은 GPU 100%, 27,143/46,068 MiB, 112.77 W다.
validation은 0.685828/0.687305/0.681566/0.683278/0.679251이고 이전 best는 0.717004다.
새 test 성적이나 수렴 완료 결과는 아니다.

코드에서는 단일 graph 집계 병목, 정적 구조 재계산, 포화 cluster BFS 반복,
validation 입력 재복제/재전송, 중복 diagnostics를 수정했다.
batch 선택에는 reference_updates 예산을 반영하고 이미 저장된 자원 계획은 보존한다.
실제 C solver 수식/모든 gradient 비교와 CPU profiler에서 graph 목적지
index_add 54회 및 scatter_reduce 24회가 sum/amax로 대체됨을 확인했다.
이 연산 횟수는 CPU ATen 관측이며 A6000 가속 배수나 667초 중 기여 시간은 아니다.

수정 후 phase별 CPU/CUDA 시간과 checkpoint 시간, 실제 Bs 크기를 수집한다.
모델 크기·K=8·샘플링 법칙·기존 배치·epoch/update 예산을 축소하지 않았다.
검증 범위: 로컬 정적 검사, 단위/회귀 검사, synthetic CPU 학습·재개 검사.
현재 로컬 torch는 CPU-only여서 수정 후 실제 데이터 A6000 학습·VRAM·처리량·정확도는
미검증이다. 서버의 기존 프로세스/결과는 변경하지 않았다.
상세 구현과 재개 계약은 CONDUCTANCE_V5.md 및 RICH_SCALING_EXPERIMENTS.md에 있다.

최종 로컬 회귀: 2,704 passed / 103 skipped / 10 warnings (217.21초).
생략 사유는 실제 CUDA·PyG 부재, Linux 전용 셸/파일시스템 검사, Windows symlink 권한,
별도 opt-in 대형 IPC stress다. 10 warnings는 의도적으로 드러낸 cluster 포화 경고다.
Ruff 정적·format 검사와 git diff --check 통과, CODE_SUMMARY.md 291개 소스 일치 확인.
실제 51da819 Git 소스→현행의 rich/conductance/resource 세 검사 및 synthetic CPU의
fixed/dynamic checkpoint 모델·AdamW·RNG·epoch 보존 후 업데이트를 검증했다.
이 검사는 과거 GPU 커널의 재실행이나 서버 원본 checkpoint의 독립 검증을 뜻하지 않는다.

이 문서는 사용자가 제공한 **서버 결과 출력**과 **현재 소스 버전의 구현**을 구분한 기록이다.
문서 작성 자체가 새 학습을 실행했다는 뜻은 아니다. 수치는 사용자 로그에서 확인했으며,
서버의 전체 원본 checkpoint/manifest/history 파일을 로컬로 받아 독립 재검증한 것은 아니다.

## 1. 소스 버전과 측정 범위

### 이전 수령: 선택적 전환 후 20조건 validation 결과와 교정 구현

사용자 제공 요약: 전체 `pending_extra_budget`, historical_reference 2 / pending_extra_budget 1 /
passed 17. 이는 20개의 신형 모델 학습 완료나 test 평가 완료라는 뜻이 아니다.
ogbn-arxiv reference fixed 0.725427 및 large fixed 0.710997은 역사적 참조다.
reference dynamic 0.727944는 구형 C의 완료 결과로 새 학습 예산이 없어 보류돼 있다.
large dynamic 0.675895는 원본 epoch 186에서 새 C로 200까지 14 epochs 전환한 결과다.

나머지 16개는 source epoch 0에서 시작한 결과이며 아래 값은 모두 validation이다.
PPI는 micro-F1, 다른 데이터셋은 accuracy다. 차이는 기술적 비교이며 동일 학습 궤적이나
C의 단일 요인 인과효과를 증명하지 않는다.

| Dataset | Profile | Fixed C | Dynamic C | Dynamic−fixed (percentage points) |
|---|---|---:|---:|---:|
| PPI | reference | 0.710244 | 0.707290 | -0.2954 |
| PPI | large | 0.566501 | 0.634512 | +6.8011 |
| PubMed | reference | 0.712000 | 0.712000 | 0.0 |
| PubMed | large | 0.700000 | 0.698000 | -0.2 |
| CiteSeer | reference | 0.610000 | 0.616000 | +0.6 |
| CiteSeer | large | 0.602000 | 0.602000 | 0.0 |
| Cora | reference | 0.688000 | 0.694000 | +0.6 |
| Cora | large | 0.638000 | 0.648000 | +1.0 |

확인 가능한 해석: 큰 모델의 fixed도 악화되어 C만의 문제라고 할 수 없다. 같은 accuracy도
C가 동일하거나 gradient가 끊겼음을 뜻하지 않는다. arxiv의 14-epoch 전환으로 나머지
fresh 조건의 부진을 설명해서는 안 된다. 실제 학습 C/beta·train loss·선택 batch·update 수는
요약표에 없으므로 서버 원본 진단 JSON 없이는 원인을 확정하지 않는다.

코드 교정은 CONDUCTANCE_V5.md 첫 절에 정의한다. 분산 0의 NaN을 고치고 명시적인
width-scaled 비용·beta 0.5 초기화·reference-update 예산을 제공한다. CPU 수치/실제
forward-loss-backward-AdamW 및 중단/완료 재개 테스트와 실제 A6000 전체 학습을 구분한다.
이번 교정 설정의 실제 데이터 학습·GPU 처리량·성능 회복은 아직 검증하지 않았다.
원본 결과를 읽는 stdout 분석 CLI도 추가했다. 과거 검증 기록은 아래에 보존한다.

교정 후 전체 로컬 회귀: **2,560 passed / 103 skipped**, 183.31초. 변경한 Python 18개
파일의 Ruff check/format 검사 및 CODE_SUMMARY 284개 source 일치 검사도 통과했다.
이 검증에는 실제 CPU forward/loss/backward/AdamW, 0분산 NaN 회귀, width-scaled C의
수치 미분·대칭·chunk/batch 불변성, 업데이트 예산의 실제 trainer 적용, 중간 checkpoint 재개
동일성, 완료 checkpoint의 재학습 금지, 결과 예산 변조 거부와 읽기 전용 분석을 포함한다.
로컬 환경은 torch 2.13.0+cpu, CUDA 없음, 논리 CPU 16개다. 생략 103개는 CUDA/BF16,
미설치 PyG, Linux/Bash/native Linux, Windows symlink 권한 및 opt-in IPC stress 제약이다.
디버그 합성 입력을 실제 데이터나 성능 결과로 제출하지 않는다. GitHub push와 서버 실행은
이번 코드 수정 작업에서 수행하지 않았다.

### 이전: V5 기존 학습 상태를 유지하는 선택적 전환 구현

추가 사용자 로그 `204d5998-9283-424e-a79b-c7b8f94aaf0f/pasted-text.txt`에는 reference/arxiv의
fixed·dynamic과 large/arxiv fixed, 총 3개 완료 조건을 검증 후 건너뛴 기록이 있다.
4번째 large/arxiv dynamic은 RNG 복구 예외를 적용한 뒤 누적 epoch 160까지 진행했다.
384 channels/12 layers/8 heads, 총 29,917,064 parameters 중 공통 backbone/W/beta
26,122,952개가 유지 대상이고 기존 MLP C 3,794,112개가 교체 대상이다.
이는 로그에 나타난 상태이며 실제 서버 last.pt의 최신 epoch·SHA를 확보한 것은 아니다.

전환 기능은 공통 가중치·AdamW moment/step·누적 epoch를 유지하고 C만 초기화한다.
완료 fixed 결과는 읽기 전용 검증 후 보존하며, 완료 dynamic에 새 C 학습 예산이 없으면
추가 예산 필요 상태로 구분한다. 새 초기값의 비교 실험이나 정확한 동일 모델 재개로 위장하지 않는다.
구현·정적 검사·로컬 CPU 회귀를 완료했다. 전체 검사는 **2,419 passed / 103 skipped**
(186.61초)이며, 전환 관련 177개 검사를 포함한다. 공통 tensor/AdamW 상태 유지, 실제 CPU
forward/backward/update, source epoch 이후 재개, 초기화 5개 중단 지점 복구, 원본 bytes 보존,
완료 결과 재사용, 조건별 추가 예산과 mixed-source journal 검증을 포함한다.
생략 항목은 CUDA/BF16, 미설치 PyG, Linux/Bash/native Linux 기능, Windows symlink 권한,
별도 opt-in IPC stress다. GPU 측정 증명서 테스트는 명시적인 CPU fixture이며 실제 측정이 아니다.
실제 서버 checkpoint 이식·A6000 실측·실제 데이터 전체 학습/평가는 미실행이다.
아래 2,242개 회귀는 전환 기능 추가 이전 소스의 검증 기록이다.

### 이전: V5 C 최적화 계층으로 구조 변경

기본 V5를 입력 그래프별 K회 C 최적화 후 가중 라플라시안으로 전파하는 구조로 변경했다.
기존 MLP-C는 명시적 비교 옵션으로 보존한다. Reference/large 모델 규모, 공식 데이터셋,
seed 0, 전체 epoch와 실험 범위는 유지하며 기본 학습은 첫 epoch부터 joint다.
이 변경은 이전 RNG 복구 패치와 달리 모델/학습 계약 변경이다. 과거 MLP checkpoint와
새 optimization checkpoint를 섞지 않으며 기존 호환 registry도 확장하지 않는다.
새 solver의 설정·에너지·잔차와 실제 반복 수를 기록하고 자원 calibration에도 같은 모델을 사용한다.
구현과 로컬 CPU 회귀 검증을 완료했다. 최종 전체 회귀는 **2,242 passed / 103 skipped**
(153.82초)다. 새 C solver 44개 및 학습/재개/calibration 연결 13개 시험을 포함한다.
과제 loss에서 C 계층의 모든 학습 파라미터로 전달되는 gradient와 실제 optimizer update,
미분 수치 검사, 라벨 없는 추론, C 양수·가중 평균 1, CPU 다음 학습 step의 정확한 재개,
이전 구조 checkpoint 거부와 기존 artifact 보존을 검사했다. Ruff 정적 검사와 코드 스냅샷
일치 검사도 통과했다. 생략 항목은 CUDA/BF16, 미설치 PyG, Linux/Bash 및 native Linux 기능,
Windows symlink 권한, 별도 opt-in IPC stress 등이다.
A6000 실측·실제 데이터 전체 학습/평가는 미실행이다. CPU debug/단위 테스트는 GPU smoke
test나 실제 데이터 성능 증거가 아니다. K=8은 유한 반복 설계 기본값이며 수렴 완료나
SOTA 성능을 주장하지 않는다. 아래 회귀 개수는 이전 소스의 역사 기록이다.

### 이전: measured run 재개 오류 수령과 복구

사용자 첨부 `d735747a-bea6-4e6b-9a8e-c5b2b14b537d/pasted-text.txt`에서
`measured-v5-cycle-se-pe-a6000-gpu3-seed0-v1`의 다음 상태를 확인했다.

- 첫 3개 V5 조건은 `verified, skipping`으로 기존 완료 결과를 재검증했다.
- 네 번째 large/ogbn-arxiv/shared_dynamic_c는 `last.pt` RNG 복원 중
  `TypeError: RNG state must be a torch.ByteTensor`로 학습 재개 전에 실패했다.
- Cycle은 GPU preflight를 통과했으나 resource plan의 데이터 식별 비교에서 실패했다.
  코드상 최적 worker 변경을 데이터 변경으로 오인하는 경로를 재현했다. 서버 실측 plan
  원문은 수령하지 않아 실제로 달라진 worker 값 자체는 아직 확인하지 못했다.

후속 수정은 CPU checkpoint/RNG 복원과 Cycle worker 메타데이터 비교 분리다.
기존 결과·체크포인트·실측 계획을 재사용하도록 정확한 source hash 쌍만 허용하는
`v5-rng-cycle-workers-v1` 호환 목록을 추가했다. 데이터·recipe·runtime·artifact 검증과
실행 중 소스 변경 검출은 유지한다. 이전 best checkpoint 및 old fixed/new dynamic 비교도
별도로 검사한다. 이는 모델 구조 변경이나 과거 모든 source에 대한 포괄적 호환이 아니다.

최종 전체 로컬 회귀는 **2,148 passed / 103 skipped** (211.73초)다. 앞선 타깃 회귀
299 passed / 10 skipped, V5 호환 26개 및 scaling 호환 13개 시험도 통과했다.
CPU 실제 dropout/AdamW 다음 step 일치와 Cycle 실제 loader의 worker 8→2/4/16을 검사했다.
CUDA와 Windows symlink 미지원 시험은 구분해 생략했고, 수정판 원격 GPU 학습·전체 평가는
아직 실행하지 않았다. 전체 원본 실험 artifact의 독립 재검증도 미실시다.

### 이전: 실제 학습 자원 calibration 도입

사용자 A6000 GPU 3 화면은 9,311/46,068MiB와 utilization 100%였다. GPU 실행 중이라는
관측이며 최적 batch·처리량·전체 학습 완료 근거는 아니다. 이전 `a6000-48gb` profile에는
여러 후보의 실측 선택이 연결되지 않았으며 그 누락을 수정한다.

현재 rich 실행은 V5/Cycle V2 본 학습 전 실제 optimizer update/state를 포함한 batch/worker
probe를 실행하고 paired 조건에 같은 선택을 적용한다. 모델/데이터 축소 없이 최소 요청
batch부터 탐색하며 실패를 작은 값으로 숨기지 않는다. Probe와 본 학습·평가를 분리하고
source/GPU/runtime/data identity에 묶인 계획 및 진행 상태를 보존한다. 자세한 정책과 명령은
[RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md) 첫 절에 있다.
실제 A6000 calibration 결과·최적 선택값·가속률·전체 새 학습 결과는 아직 수령하지 않았다.

이번 calibration 변경 후 전체 로컬 회귀는 **2,053 passed / 101 skipped** (156.76초)다.
Ruff 정적 검사, diff 공백 검사와 코드 스냅샷 일치 검사도 통과했다. 생략 항목은 실제
CUDA/BF16, 미설치 PyG, Linux/Bash 및 native Linux 기능, Windows symlink 권한,
별도 opt-in IPC stress 등이다. CPU 단위 테스트와 GPU 호출 모의 검증을 실제 GPU smoke
test로 간주하지 않는다. 이번 작업에서 실제 A6000 측정·전체 실제 데이터 학습·전체 평가는
실행하지 않았다. 모델·연산자·데이터 설정은 축소하지 않았고 기존 서버 작업·결과는 보존했다.
아래 역사적 회귀 수치는 이번 검증 수와 구분한다.

### 역사적 로컬 검증

throughput/IPC/격리 교정 당시 전체 로컬 CPU 회귀는 **1,940 passed / 99 skipped**
(189.74초)다. `CYCLE_V2_IPC_STRESS_GRAPHS=10000`으로 실제 2-worker의 1만 합성
그래프 준비·캐시 검증까지 이 전체 실행에 포함했다. Ruff·코드 스냅샷 일치 검사도 통과했다.
생략은 Linux/Bash, native Linux renameat2, Windows symlink 권한, 미설치 PyG, CUDA/BF16
조건이다. 실제 서버의 mmap/FD 한도와 GPU 전체 학습·평가를 검증한 결과는 아니다.
실제 이전 결과의 격리/삭제도 실행하지 않았으며 격리 테스트는 합성 디렉터리만 사용했다.

2026-09-05 SE/PE 분리 완료 당시 로컬 CPU 전체 회귀는 **1,852 passed / 98 skipped /
1 warning** (164.78초)다. Ruff 및 코드 스냅샷 일치 검사도 통과했다. 생략은 Linux/Bash,
Windows symlink 권한, 미설치 PyG, CUDA/BF16 환경 조건이며 경고는 PyTorch multiprocessing의
sparse tensor 재구성 경고다. 이는 실제 데이터 전체 학습·평가나 A6000 성능 검증이 아니다.
아래 이전 구현 시점의 테스트 수치는 역사 기록으로 보존한다.

### ad041e2 이후 수령한 실제 실패와 후속 수정

사용자 첨부 `9e790812-518b-4369-9745-88f1fc46d76e/pasted-text.txt`의 SHA-256은
`BE7433E5C4C5570B26C83F301845EA0DF557829C0DC45F77163C9307B5EC3E63`다.
run ID는 `v5-cycle-se-pe-a6000-gpu3-seed0-v1`, A6000 GPU 3,
Torch 2.7.1+cu118, model seed 0, 계획은 V5 20학습과 Cycle SE/PE 8학습이었다.
전체 checkpoint/manifest를 로컬로 수령한 독립 재검증은 아니다.

- V5 첫 reference/ogbn-arxiv/fixed-C가 54 epochs·2,430 steps를 완료하고
  최고 validation 0.719621, allocator peak 7,505,515,008 bytes를 기록했다.
  child 저장 후 `passed`를 출력했지만 집계의 `throughput.scope` 검사에서 실패했다.
  나머지 19개 V5는 미실행이며 이 값으로 fixed/dynamic 또는 reference/large 비교를 하지 않는다.
- Cycle 8개 조건은 모두 전처리 중 공유 storage mmap 실패로 학습을 시작하지 못했다.
  약 7천 그래프 이후 몇 bytes 매핑도 ENOMEM인 현상은 mapping-count 누적과 맞지만
  서버의 실제 `vm.max_map_count`/cgroup/주소공간 한도는 확인하지 못했다.
- GPU utilization은 `pynvml` 미설치와 UUID 없는 numeric-device mapping 거부로 null이다.
  이를 GPU 사용률 0%로 해석하지 않는다. 이 계측 부재가 위 두 실패를 일으킨 원인은 아니다.

후속 수정은 V5 telemetry writer/consumer 계약 정합화와 Cycle 양방향 Tensor-free IPC다.
V5/Cycle 모델 구조·파라미터·데이터·batch는 ad041e2 대비 바뀌지 않는다. 이전 테스트는
실제 writer→집계 연결과 Linux 대량 IPC 누적을 놓쳤으므로 그 테스트 수를 전체 GPU 검증으로
인용하지 않는다. 이전 실패 결과는 삭제/수정하지 않고 선택적으로 whole-run 격리하며
새 source run과 혼합하거나 hash 검사를 우회하지 않는다.

현재 최상위 구조는 [Conductance V5](CONDUCTANCE_V5.md)와 새
[Cycle PE V2](CYCLE_PE_V2.md)다. V5는 shared graph-conditioned dynamic C와 multi-head W,
graph-conditioned beta를 분리했고, `fixed_c/shared_dynamic_c` 두 arm을 연구급 architecture에서
비교한다. Cycle의 과거 `cycle_basis_v2`와 QR 기반 `cycle_projector_pe_v2`는 현재 경로에서
제외했다. 구조 요약을 PE라고 부르던 `cycle_dfs_sparse_pe_v2`와도 결과를 분리한다.
현재 SE는 `cycle_dfs_se_v2`, PE는 `cycle_dfs_relative_pe_v2`다. PE는 동일 SE/backbone에
cycle 내부 상대 위치 residual을 더하며 추가 trainable parameter는 0개다. 유일한
`dfs_fundamental` backend의 전체 기저와 실제 cycle 순서를 sparse로 사용하고
QR/SVD/Gram inverse를 하지 않는다. 선택된 DFS tree에 의존하며 일반 기저변환 불변성은
주장하지 않는다. 두 조건의 실제 GPU 전체 학습·성능·가속 결과는 아직 없다.

현재 [전체 scaling](RICH_SCALING_EXPERIMENTS.md)은 `reference/large` 두 architecture
profile을 사용한다. Conductance V1–V5는 106 child/model trainings, Cycle V1/V2는 12
child/model trainings, Tree V1/V2는 4 child 안에서 8 models를 학습한다. 따라서 합계는
**122 child runs / 126 fresh model trainings**다. Cycle은 V1 4학습과 V2 SE/PE 8학습이다.
동일 run-id는 검증된 완료 child를 skip하며
같은 새 source/config/schema의 V5와 Cycle V2 `last.pt`가 있으면 epoch 상태를 복원한다.
현재 Cycle cache는 `cycle_pe_v2_ordered_dfs_benchmark`이며 SE/PE가 공유한다. 두 encoding의
checkpoint는 교환할 수 없다. 구 cache/checkpoint와 수정 전 V5 source checkpoint는 호환되지 않으며 hash 검사를
우회하지 않는다. 완료한 전체 matrix를 다시 실행하라는 의미는 아니다. 과거의
`base/wide/deep/large` 204-training 계획은 폐기됐고 현재 실행 계약이 아니다.

실행 hardware profile은 보수적인 `portable`과 opt-in `a6000-48gb`로 분리된다. A6000
profile은 실제 minibatch/sample 크기와 수치 실행을 바꾸므로 portable 결과와 점수나 실행 시간을
직접 대응시켜 모델 효과 또는 GPU 효과로 해석할 수 없다. 2026-09-04 첫 A6000 실행은 아래
메모리·수치 오류로 중단됐으며 성공한 scaling 성능이나 가속 실측으로 사용하지 않는다.

현재 통합 runner는 단일 `--device`에서 track을 순차 실행하고, 서로 다른 indexed 장치를
`--devices`로 명시했을 때만 independent track을 GPU별로 병렬 배정한다. 같은 GPU의 최상위 track
concurrency는 1이며, Tree A6000 child 내부의 명시적 candidate concurrency 2는 별도 계층이다.
새 resource monitor는 실제 GPU 서버에서 Conductance V5, Cycle PE V1/V2와 Tree V1/V2의 GPU
SM/메모리 utilization, CUDA allocator, process CPU·RSS/HWM과 system available RAM을
주기적으로 기록하고, 미지원 측정은 `null`과 원인을 남긴다. 그러나 이 계측이 적용된 수정판의
완료 GPU artifact는 아직 수령하지 않았으므로 현재 문서에 새 utilization 수치를 제시하지 않는다.
당시 Rich runner는 여러 physical batch 후보를 자동 튜닝하지 않았고 manifest에도
`throughput_candidate_sweep=false`를 기록했다. 현재는 첫 절의 optimizer-inclusive calibration이
이 누락을 보완하며 본 학습 중에는 선택값을 고정한다. 별도 CUDA microbenchmark는
Conductance V1/V5, Cycle PE V1/V2, Tree V1/V2의 명시 후보마다 device-wide GPU
utilization·peak VRAM·CPU/RAM·처리량과 수치/gradient integrity를 측정하고 10% memory headroom
기준의 권고를 낸다. 다만 optimizer state·전체 epoch·validation/checkpoint가 없는 고정 real-batch
측정이고 profile 기본값을 자동 변경하지 않으므로, 이를 새 calibration의 대체 근거로
주장할 수는 없다.

후속 사용자 요청으로 현재 기본 실행은 model seed **0 하나**다. 기존 5-seed 측정값은 아래에
그대로 보존하며, 기본값 변경이 과거 결과나 source revision을 바꾸지는 않는다.
단일 seed의 std/CI는 null로 기록한다. 새 read-only `--full-audit`는 C 평균/셔플/전파 제거와
train-label gradient를 검사하며 **5e801c3 실행의 새 GPU 로그를 수령했다.** 아래 확장 검사 절에
기록했다. 이후 **43afd63의 2×2 GPU 재학습 결과도 수령했으며 여덟 조건 모두 passed**다.
이어 **C-learning의 네 조건 비교도 모두 passed인 보고서를 수령했다.** 새 run의
`learned_c` checkpoint의 평균-C 검사도 이후 `passed` GPU 출력을 수령했다.
다음은 엣지별 직접 C의 Conductance v2, 공유 상대 C 생성기의 v3, 그리고 상대 C graph
operator와 spatial message transform을 함께 학습하는 v4다. 세 버전은 별도 구현·실행 경로를
사용한다. 2026-09-02 사용자 보고상 **과거 arxiv-only** Conductance v2/v3 runner와 Cycle PE v2
runner는 각각 `passed`다. 성능 수치와 전체 원본 artifact는 수령하지 않아 이 문서에 창작해
넣지 않는다. 과거 arxiv-only V4는 첫 arm의 200 epochs·child exit 0 뒤 구 report gate에서
중단되어 정식 결과가 아니다. 현재 확대된 V2/V3/V4 8/10/20-job 전체 결과는 수령하지 않았다.

이후 단일 `hidden/layer` 설정만으로는 큰 모델에서의 적합도를 확인할 수 없다는 사용자 요청에
따라 [전체 모델 규모 확장 실험](RICH_SCALING_EXPERIMENTS.md)을 연구급 크기로 재설계했다.
Conductance V1–V5, Cycle PE V1/새 V2, Tree fixed/multi를 dataset-aware `reference`/`large`로
실행한다. 기본 model seed 0 계획은 Conductance 106, Cycle 12, Tree 8 trainings로 총
**122 child runs / 126 model trainings**이다. Tree child 하나가 fixed/multi 두 모델을
학습하므로 두 수가 다르다. Cycle/Tree 후보는 validation-only로 비교한 뒤 선택 checkpoint만
test-only로 평가한다. Cycle은 encoding×dataset별 선택이므로 V1 포함 6회, V2-only 4회 test다.
현재 확인된 것은 runner·manifest·무결성 검사와 로컬 테스트이며 이 GPU
학습 결과는 아직 실행·수령하지 않았다. 같은 인수와 run ID 재실행은 완료 child를 검증·skip하고,
V5와 새 Cycle V2는 `last.pt`부터 이어지며 legacy 미완료 child만 처음부터 재시도한다.

### 2026-09-02 사용자 서버 실행 보고

사용자 terminal 보고의 공통 source/pull revision은 `7b4cd32`다. GPU preflight에는
`NVIDIA A100-SXM4-80GB MIG 1g.10gb`가 기록됐고, 서버 할당은
`CUDA_VISIBLE_DEVICES=6`, 프로세스 내부 논리 장치는 `cuda:0`이었다. 아래 상태는 사용자
출력 기준이며 성능 수치와 전체 원본 artifact를 수령해 독립 검증한 결과가 아니다.

| 트랙 | Run ID | 사용자 보고 상태 |
|---|---|---|
| Conductance v2 | `gat-direct-c-v2-gpu6-seed0-v1` | 과거 arxiv-only 2-job run `passed`; 현재 8-job 결과 아님 |
| Conductance v3 | `gat-relative-c-v3-gpu6-seed0-v1` | 과거 arxiv-only 2-job run `passed`; 현재 10-job 결과 아님 |
| Conductance v4 | `gat-hybrid-c-spatial-v4-gpu6-seed0-v1` | 과거 arxiv-only 4-arm run의 `fixed_c_identity_w`만 200 epochs·child exit 0 뒤 구 report gate 중단; 나머지 3개 pending; 현재 20-job 결과 아님 |
| 폐기된 구 Cycle PE v2 | `cycle-pe-v2-gpu6-seed0-v1` | 당시 runner `passed`; 현재 sparse DFS 모델 결과가 아님 |
| 당시 전체 scaling 계획 | 미실행 | 구 118-child/122-training 계획의 기록이며 GPU 결과 없음; 현재 SE/PE 분리 계획은 위 122-child/126-training 계약 |

### 2026-09-04 A6000 V5·Cycle V2 partial 실행과 old-source r2 재현

Run `new-v5-cyclev2-a6000-gpu3-seed0-r1`, GPU 3 RTX A6000, model seed 0의 사용자
terminal 로그를 수령했다. 첨부 텍스트 SHA-256은
`D6394C813EDB52DD2CE6746B531973A750A1F0AE807EC5A3410B938FAFF6F93E`다. 전체
manifest/checkpoint/history를 로컬로 받은 독립 재검증은 아니다.

- Conductance child는 20개를 계획했다. 첫 `v5/reference/ogbn-arxiv/fixed_c`만 200 epochs와
  child `passed`를 완료했다. 전체 로그 최고 validation은 epoch 10의 0.692775였고, 구
  joint-only primary는 0.680392였다. Train loss가 0.579221에서 0.019955로 내려가는 동안
  validation은 대체로 0.67대로 하락했다.
- 두 번째 `shared_dynamic_c`는 epoch 20 warm-up 뒤 최초 conductance calibration backward에서
  44.47/44.55GiB를 사용하고 추가 104MiB 할당에 실패했다. 나머지 18개 Conductance job은
  미실행이다.
- Cycle V2는 Peptides-struct와 ZINC-12K의 projector cache 준비를 완료했지만 reference/large
  네 학습 모두 첫 epoch 전에 non-finite gradient로 실패해 성공 metric이 없다.
- 원인은 각각 dynamic edge-score autograd activation 누적과 A6000 FP16+GradScaler 초기
  scale overflow로 확인됐다. 당시 수정은 edge-score chunk checkpoint, BF16/no-scaler,
  condition-aware V5 checkpoint selection이었다. 이후 r3의 diffusion OOM 재발과 현재 수정은
  아래에 별도로 기록한다.

구 V5 fixed run은 실제 global-best model state를 저장하지 않아 corrected checkpoint로 복구할
수 없고, 구 partial `last.pt`도 새 selection schema/source와 이어 붙이지 않는다. 수정판은 새
run ID가 필요하다. 당시 동일 projector 모델의 cache 재사용과 달리 현재 sparse DFS 모델은
`cycle_pe_v2_ordered_dfs_benchmark`를 새로 준비하며 projector 및 구 support-only cache를
재사용하지 않는다. 이 partial run의 수치를
fixed-vs-dynamic 또는 V5/Cycle V2 성능 결과로 인용하지 않는다.

후속 run `new-v5-cyclev2-a6000-gpu3-seed0-r2`의 사용자 terminal 로그도 수령했다. 첨부
`bd63fc9a-60da-4daf-9ab9-da49db7cbbe1/pasted-text.txt`의 SHA-256은
`F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6`다. 그러나 새 run ID만
사용했을 뿐 수정 commit `214265c0dcc09a67ae0c9c3d09d9e2cbc26c63bb`는 서버에 적용되지
않았다. Conductance traceback의 `train.py:785`와 `joint_best=` 단독 출력, Cycle traceback의
`benchmark.py:589`는 모두 직전 `08d8ed6` 소스와 일치한다.

r2도 r1과 같은 실패를 재현했다. Conductance는 fixed-C 한 job만 완료한 뒤 dynamic-C 최초
calibration backward에서 다시 44.47/44.55GiB OOM이 발생했고 18개가 미실행이다. Cycle V2
네 job도 모두 첫 epoch 전에 같은 non-finite gradient로 실패했다. 따라서 r2는 edge-score
checkpoint, condition-aware selection, BF16/no-scaler 수정의 검증 결과가 아니며 그 수치나
artifact를 수정판 결과로 사용하지 않는다. 이것은 r2의 역사적 기록이며 r3까지 old source였다는
뜻이 아니다.

### r3 large OOM 재발과 2026-09-05 QR-free 구현

사용자가 추가로 제공한 r3 로그는 4/20번째 `v5/large/ogbn-arxiv/shared_dynamic_c`에서
warm-up 20 epoch 이후 diffusion forward가 192MiB 할당에 실패한 기록이다. GPU 44.55GiB 중
해당 프로세스 44.54GiB, PyTorch allocated 41.70GiB였다. Traceback은 `214265c`와 정확히
일치하므로 score checkpoint 수정판을 실행한 뒤에도 OOM이 남았음이 확인됐다. 192MiB는
131072-edge chunk × hidden384 × FP32 4bytes다. 나머지 job의 완료 여부를 이 로그로 추정하지 않는다.

현재는 V5 diffusion custom backward가 node message/scalar edge weights만 저장하고 edge
features를 chunk별 재계산한다. 기존 규모와 sampling은 유지했다. 관련 CPU 검사 28개 통과,
동일 합성 입력의 고유 autograd 저장량 563,840→60,544 bytes(약89.3% 감소)를 확인했으나
실제 GPU peak나 전체 학습 성공을 검증한 것이 아니다. 상세는 [V5 문서](CONDUCTANCE_V5.md)를 따른다.

Cycle 현재 조건은 QR-free sparse DFS 구조 SE와 동일 SE+cycle 상대 PE이며 backbone 크기는
유지했다. 기본 ZINC parameter는 두 조건 모두 7,262,785개(구 projector보다 +16,704)다.
DFS 탐색 O(V+E), 기저·실제 cycle 위치 출력 O(nnz Z), 각 sparse 집계 O(nnz Z*d)를 구분한다.
SE는 sparse product 2회, PE는 6회이므로 PE가 SE보다 빠르다고 주장하지 않는다.
새 V2만 실행하는 `cycle-se-pe-a6000-gpu3-seed0-v1`은 `--encodings se pe`, 두 dataset·두
profile·seed 0의 8학습과 선택 checkpoint 4회 test다. 명령과 checkpoint/cache 비호환성은
[CYCLE_PE_V2.md](CYCLE_PE_V2.md)에 있다. 완료한 다른 트랙이나 전체 matrix를 다시 실행하지 않는다.

V3는 graph-centered score → bounded relative C → isotropic mixture와 학습 alpha의
대칭 정규화를 사용한다. AdamW backbone/생성기/scalar 그룹을 분리했다. 현재 기본은
Cora/CiteSeer/PubMed/PPI/ogbn-arxiv × `relative_c`/`fixed_c` × seed 0의 10회 학습이며
v2나 이전 MLP의 checkpoint·점수를 재사용하지 않는다.
PPI는 공식 20/2/2 독립 graph split, whole-graph minibatch 2와 BCEWithLogitsLoss를 사용하고
`logit > 0`의 모든 validation node-label 결정을 합친 global micro-F1로 checkpoint를 선택한다.
Test graph는 train/validation loader·forward·loss·metric·선택·진단에는 들어가지 않지만 full
cache의 test tensor와 metadata는 공식 split·shape·checksum 무결성 검사로 load/validate된다.
선택된 checkpoint에서 평균 C·셔플 C·C=1·전파 제거 validation 검사도 별도 forward로 수행한다.
평균 C와 C=1은 대칭 정규화에서 동등하므로 서로 일치해야 하는 수치 검산이다.
실행과 수식·비교 경계는 [전체 인수인계](HANDOFF.md)와 [코드 스냅샷](CODE_SUMMARY.md)을 따른다.
초기 V3 구현 게시 revision은 `6f9d3b0981e8cfa8feb76e59fb348e26cc6909d6`이다.
사용자 보고상 과거 arxiv-only v3 runner는 `passed`지만 validation 수치와 전체 artifact는
수령하지 않았고, 현재 10-job 전체 기본 결과도 없다.

V4는 v3의 상대 C 생성기와 대칭 weighted-degree 정규화를 유지하고, 각 층에 bias 없는
identity-initialized `W`를 추가한다. 한 층은 `C(H)`를 먼저 만들고 비고립 노드에서
`(1-alpha)H + alpha P_C(HW)`를 계산한다. `W=I`이면 v3 전파와 일치한다. 현재 기본은
v1의 5개 데이터 × 고정/상대 C × identity/학습 W × seed 0의 20개 새 CUDA 학습이다. Alpha는
모든 조건에서 학습하고, inactive C/W scaffold는 동결하여 optimizer에서 제외한다. V3 결과를
재사용하지 않으며 네 cell의 조건부 주효과와 interaction만 V4 내부의 기술적 대조로 보고한다.
PPI의 split·batch·loss·`logit > 0` global micro-F1과 test 미평가/cache 무결성 경계는 위 V3와 같다.
선택 checkpoint의 C/W 개입은 재학습 효과와 구분한다. Mean-C와 C=1은 대칭 정규화에서
대수적으로 중복된다. 별도 CUDA forward의 logit 차이와 `allclose_rtol=1e-5`,
`allclose_atol=1e-6`, `within_declared_tolerance`는 informational non-gating 진단이며
arm·report·run의 성공 조건이 아니다. 사용자 보고상 과거 arxiv-only run의
`fixed_c_identity_w`는 200 epochs와
child exit 0까지 완료됐지만 구 numeric hard gate에서 중단되어 나머지 세 arm은 pending이다.
성능 수치는 수령하지 않았고 이 partial arm은 재사용하지 않으므로, 정식 V4 결과에는 새
run의 5개 데이터 × 네 fresh arm, 총 20 jobs가 모두 필요하다.

이전 전체 scaling runner 추가 시점의 구현은 `PYTHONUTF8=1` 전체 로컬 회귀에서
**1418 passed / 77 skipped** (80.24 s, exit 0)를 통과했다. V2/V3/V4 전용 결과는 각각
**118 passed**, **141 passed / 2 skipped**, **131 passed**다. Ruff·compileall과 재생성한
`code_summary --check`도 통과했다. 생략은 Linux/Bash·Windows symlink 권한·로컬 PyG 미설치·
실제 CUDA RNG처럼 이 호스트에서 충족되지 않은 환경 조건이다. 이 검증은 공개 데이터 GPU 학습
결과나 성능 측정이 아니다.

아래 검증 숫자는 5-dataset/PPI 확장 이전 구현 시점의 역사 기록이며 현재 숫자로 재해석하지
않는다. 당시 CUDA 수치검사 교정 후 전체 로컬 회귀는 **1301 passed / 65 skipped** (135.13 s, exit 0),
V4 전용 **122개**와 Ruff·compileall·`code_summary --check`가 모두 통과했다. 생략은 기존
환경별 검사이며 공개 데이터 학습이나 GPU 성능 측정은 로컬에서 실행하지 않았다. Windows의
한국어 작업 경로에서는 기존 V3 fixture의 UTF-8 JSON을 기본 CP949로 읽지 않도록
`PYTHONUTF8=1`로 전체 회귀를 실행했다.

직전 상대 C v3 추가 시점의 전체 로컬 회귀는 **1176 passed / 65 skipped** (44.45 s, exit 0)였다.
V3 전용 134개가 통과했고 실제 CUDA RNG 보존 검사 1개는 로컬 GPU가 없어 생략됐다.

직전 직접 C v2 구현 시점의 로컬 회귀는 **1042 passed / 64 skipped** (47.41 s, exit 0)였다.
직전 평균-C 검사 확장의 924개에 v2 전용 118개를 추가했다. 직접 C의 FP64 미분·chunk 연산,
graph binding과 학습 루프→checkpoint→비교표 연결 및 실제 C gradient coverage를 검사했다.
이전 shared-MLP와 그 평균-C 검사도 전체 회귀에 포함했다. 공개 데이터 학습은 실행하지 않았다.
당시 65개 생략은 Linux/Bash 전용 62개, Windows 실제 symlink 권한 1개, 로컬 PyG 미설치
1개와 실제 CUDA RNG 검사 1개다. 이 로컬 회귀는 Linux/CUDA 실행 또는 GPU 가속의 증거로
제시하지 않는다.

| 구분 | 확인된 상태 |
|---|---|
| 이전 진단 전용 게시 commit | `ebf8cd19b80e6cd6c742b132e2bb1dadb97b019c` |
| 이전 commit의 추가 내용 | Conductance 진단 Python/Bash, 전용 테스트, 안내 문서, 트랙 README의 5개 파일 |
| 기존 학습 코드 | 위 진단 commit은 기존 benchmark의 모델·학습 수식을 변경하지 않음 |
| 현재 Cycle V2 SE/상대 PE | `cycle_dfs_se_v2`와 `cycle_dfs_relative_pe_v2` 구현·CPU 계약 검증; 두 조건의 새 GPU 전체 학습·성능·가속 결과 없음. 과거 projector V2의 네 FP16 실패는 별도 역사 기록 |
| 실행 최적화·선택적 compile·속도 도구 | 이 소스 버전에 포함, 로컬 단위 검증 완료. GPU 가속 실측 미수령 |
| 단일 seed 기본값·확장 checkpoint 검사 | 5e801c3 GPU full audit 수령, seed 0 다섯 데이터셋 passed |
| Gate WD × normalization 2×2 | 43afd63 실제 GPU 결과 수령. PPI/arxiv × 4조건 × seed 0 모두 passed |
| Node-degree의 learned C vs fixed C | `gat-c-learning-seed0-v1`, 2데이터 × 2조건 × seed 0, 모두 passed 보고서 수령 |
| Node-degree checkpoint mean-C 개입 | 새 c_learning/learned_c의 PPI/arxiv GPU 출력 수령, passed. 기존 factorial도 별도 지원 |
| Conductance 직접 C v2 | 과거 arxiv-only `gat-direct-c-v2-gpu6-seed0-v1` 사용자 보고상 `passed`; 현재 4-dataset 기본 결과는 미수령 |
| Conductance 상대 C v3 | 과거 arxiv-only `gat-relative-c-v3-gpu6-seed0-v1` 사용자 보고상 `passed`; 현재 5-dataset 기본 결과는 미수령 |
| [Conductance C × spatial W v4](CONDUCTANCE_V4.md) | 과거 arxiv run은 첫 arm 뒤 구 report gate에서 중단. 현재 5-dataset × 4-condition = 20-arm 정식 결과 없음; 새 전체 run 필요 |
| [Conductance graph-conditioned v5](CONDUCTANCE_V5.md) | r1/r2 partial 및 214265c 적용 r3 large diffusion OOM 수령. 현재 custom backward CPU 검증, 새 GPU 전체 성공·유효 fixed/dynamic 비교 없음 |
| [전체 큰 모델 scaling](RICH_SCALING_EXPERIMENTS.md) | Conductance V1–V5 106 + Cycle V1/V2 12 + Tree fixed/multi 8 = 126 trainings / 122 child 계약. V5/구 Cycle V2 partial 실패 로그만 수령, 전체 GPU 결과 없음 |
| [CODE_SUMMARY.md](CODE_SUMMARY.md) | 이 버전의 source/test/config/script 전체를 파일별로 보존한 스냅샷 |

`ebf8cd1`까지만 받은 서버에는 새 기능이 없으므로 업데이트 후 `git rev-parse HEAD`로
실행 revision을 확인한다. 소스 업데이트가 서버에서의 실행 완료를 뜻하지는 않는다.
아래 기존 학습 결과를 Conductance v2/v3/v4·최적화·새 2×2/C-learning 결과로 재분류하면 안 된다.

### 수령한 C-learning 비교: 학습 C의 성능 이득은 관측하지 못함

2026-09-01 수령한 `gat-c-learning-seed0-v1` 보고서는 model seed 0이며 네 조건과
전체 비교가 `passed`다. Validation만 평가했고 test는 읽지 않았다.

| 데이터 | Learned C (%) | Fixed C=1 (%) | Learned − fixed (pp) | Best / 실행 epoch (learned; fixed) |
|---|---:|---:|---:|---|
| PPI micro-F1 | 52.564966 | 52.705738 | −0.140772 | 64 / 114; 90 / 140 |
| ogbn-arxiv accuracy | 68.317723 | 68.324435 | −0.006711 | 195 / 200; 195 / 200 |

PPI learned에는 비상수 C가 남아 있지만 성능 이득은 없었고, arxiv learned는 C 변동과
점수 차이가 모두 작다. 이는 이 seed·설정에서 이득을 관측하지 못했다는 결과이지
일반적 동등성이나 C의 보편적 무용성을 증명한 것이 아니다. 동결 gate scaffold를 보존하므로
약 69%의 **활성 학습 파라미터 감소**를 저장 공간·GPU 메모리·속도 개선으로 바꾸어 말하지 않는다.

C-learning 구현 게시본은 `25ca328`이지만 제공된 비교표에는 실제 서버 source revision이
없다. 해당 manifest revision을 독립 확인하지 않았으므로 실행 commit으로 단정하지 않는다.
이번 근거는 inline 붙여넣기라 별도 첨부 파일/SHA-256도 없다. 정확한 파라미터 수·층별
진단과 이전 PPI 점수와의 구분은 아래 C-learning 결과 절과 [전체 인수인계](HANDOFF.md)를 따른다.

이어 **같은 새 C-learning run의 learned checkpoint**에 대한 읽기 전용 평균-C 검사도
수령했다. 원 validation 및 원본 무결성 확인 후 그래프·층별 C 변동에 대한 현재 의존도를
검사했으며 재학습·optimizer step·test 평가는 없다. 구체적인 결과는 바로 다음 절을 따른다.

### 수령한 평균-C 검사: PPI checkpoint 의존도와 재학습 이득은 다름

사용자가 제공한 inline terminal 출력의 revision 표시는 `8f6b4da`, 검사 상태는 `passed`다.
대상은 `gat-c-learning-seed0-v1`의 seed 0 `learned_c`다. 원 validation은 PPI
52.564966% 그대로 재현됐고 arxiv는 저장 68.317723%, 재계산 68.317729%다.

| 데이터 | 전체 층 평균 C 후 validation (%) | 원 재계산 값 대비 Δ(pp) | 바뀐 예측 (%) |
|---|---:|---:|---:|
| PPI micro-F1 | 45.915526 | −6.649440 | 7.619317 |
| ogbn-arxiv accuracy | 68.284171 | −0.033558 | 0.184570 |

PPI에서 layer 0만 평균화한 차이는 −6.198266pp, layer 1만 평균화한 차이는 −0.238916pp다.
따라서 그 learned checkpoint는 엣지별 C 패턴에 의존한다. 그러나 처음부터 학습한 fixed C
모델은 52.705738%였다. **고정된 checkpoint의 개입 민감도와 다른 모델을 새로 학습한
성능 차이는 다른 질문**이다. 층별 효과도 더해서 전체 효과로 만들지 않는다.
arxiv는 layer 0 prediction flip이 0이며 layer 1 개입이 전체 개입과 같은 작은 점수 차이를 냈다.
6개 개입의 정확한 점수·flip 단위·logit 변화는 이 문서의 C-learning 결과 절에 있다.

이 근거는 사용자 출력이며 서버의 원본 artifact 전체를 독립 검사한 것은 아니다.
단일 model seed의 validation 결과이므로 test·유의성·보편적 동등성 주장은 하지 않는다.

### 과거 arxiv-only v2/v3 보고와 현재 확대된 전체 실행 필요

[Conductance v2](CONDUCTANCE_V2.md)는 canonical 물리 엣지마다
층별 alpha를 두고 `c_e=exp(alpha_e)`를 직접 학습한다. Alpha=0에서 C=1로 시작하며
C 생성 MLP·고유분해 없이 implicit diagonal C와 기존 node-degree 전파를 사용한다.
직접 alpha의 WD는 0, 나머지 파라미터의 WD는 0.0005다. 같은 초기 상태의 direct/fixed를
별도 새 run에서 학습하므로 기존 MLP 결과를 v2의 점수로 가져오지 않는다.

현재 기본은 **Cora/CiteSeer/PubMed/ogbn-arxiv × direct_c/fixed_c × seed 0 = 8개 CUDA
학습**이다. Unseen 독립 그래프에 엣지 파라미터를 전달하는 규칙이 없으므로 PPI는 V2에서
N/A다. 네 데이터는 모두 full-batch다.
Chunk 연산은 전체 엣지의 forward/backward를 처리하는 메모리 제어이며 GraphSAGE식 sampling이 아니다.
기존 공유 MLP 설계도 유효하다. V2는 별도 직접 파라미터화 가설이다. 사용자 보고상 과거
arxiv-only runner는 `passed`했지만 GPU 성능 수치와 전체 artifact는 수령하지 않았다.

[Conductance v3](CONDUCTANCE_V3.md)는 공유 MLP가 방향 불변인 엣지
특징에서 score를 생성한다. 그래프별 중심화 및 mean(C)=1 상대화, 학습 gamma/tau,
학습 alpha와 symmetric normalization을 조합한다. 현재는 v1의 5개 데이터/seed 0을 사용하고
각 dataset에서 C=1 모델도 새로 학습한다. V3의 fixed C도 alpha는 학습한다.
정규화·파라미터화·optimizer가 여러 가지 바뀌므로 버전 간 점수 차이는 단일 요인 효과가 아니다.
제안의 dmax/작은 rho 지적은 과거 global-max v1에 해당하며 현재 v2의 오류라고 기록하지 않는다.
Gamma 값만으로 C의 중요도를 판정하거나 full-graph chunking을 neighbor sampling으로
설명하지 않는다. PPI는 공식 20/2/2 graph split, whole-graph minibatch 2,
BCEWithLogitsLoss와 `logit > 0`의 global node-label micro-F1을 쓰는 inductive 실험이다.
Test graph는 계산·선택·진단에는 미사용이고 full cache 무결성 검사에만 포함된다.
Multi-head·implicit solve는 이번 v3 실행 범위에 없다. 사용자 보고상 과거 arxiv-only runner도
`passed`했지만 GPU 성능 수치와 전체 artifact는 수령하지 않았다.

[V4 통합 문서](CONDUCTANCE_V4.md)의 실험은 v3의 `C(H)`를 학습 가능한
graph operator/metric 경로로 유지하면서, 이웃 feature message에 층별 `W`를 적용한다.
비고립 노드의 식은 `(1-alpha)H + alpha P_C(HW)`이고 고립 노드는 `H`를 유지한다.
`W=I` 조건은 v3 전파와 일치한다. 현재 기본은 v1의 5개 데이터/seed 0에서 고정/상대 C ×
identity/학습 W의 20개 fresh training이며 모든 cell에서 alpha를 학습한다. 보고서는 `C|W off`, `C|W on`,
`W|C fixed`, `W|C relative`, interaction을 V4 내부에서만 계산한다. V3 checkpoint/점수는
재사용하지 않고, 선택 checkpoint의 C/W 제거 개입을 fresh-training 차이로 해석하지 않는다.
V4의 PPI도 V3와 같은 20/2/2, whole-graph minibatch 2, BCEWithLogitsLoss, `logit > 0` global
node-label micro-F1과 test 미평가/cache 무결성 검사 경계를 사용한다.
Mean-C/C=1 별도 CUDA forward의 logit 차이와 선언된 allclose 허용오차 판정은 informational
non-gating이다. 사용자 보고상 첫 arm만 200 epochs·child exit 0까지 완료된 뒤 구 report
numeric gate에서 중단됐고 나머지 세 arm은 pending이다. 이 과거 arxiv partial 실행을 2×2
결과로 재사용하지 않으며 새 run에서 전체 20개 arm을 모두 fresh 완료해야 한다.

### 수령한 2×2 GPU 재학습: 정규화 효과와 C 학습은 별개

Run `gat-factorial-seed0-v1`, 소스 `43afd632b97a4285dfeae26847b4f12a8fd1a1e4`,
model seed 0. NVIDIA RTX A6000, Python 3.11.16, Torch 2.7.1+cu118, PyG 2.7.0,
Linux glibc 2.35에서 여덟 fresh training과 최종 비교표가 모두 `passed`다.
Train 정답으로 학습하고 validation으로 선택했으며 **test는 평가하지 않았다**.

| 조건 | PPI validation micro-F1 (%) | arxiv validation accuracy (%) |
|---|---:|---:|
| baseline | 48.986770 | 50.927883 |
| gate_no_wd | 49.378028 | 50.565451 |
| node_degree | 52.465469 | 68.317723 |
| node_degree_gate_no_wd | 50.340520 | 67.995566 |

두 데이터 모두 **node_degree + gate WD 0.0005**가 최고다. 기존 정규화 대비 PPI
+3.478699pp, arxiv +17.389840pp다. WD를 제거하면 C 변동은 커지지만 성능은 일관되게
좋아지지 않았다. 특히 arxiv 최고 조건의 두 층 C CV는 0과 약 0.00948이므로
이 결과만으로 학습 C의 기여를 입증할 수 없다. PPI에는 비상수 C가 남아 있어
반대로 C가 항상 불필요하다는 결론도 성립하지 않는다.

정확한 epoch·5개 대비·층별 C/rho/전파량·해석 경계와 다음 분석은
아래 Conductance 실험 정리 절과 [전체 인수인계](HANDOFF.md)에 보존했다.
단일 seed의 탐색적 validation 결과이며 유의성·일반적 최적값·SOTA 주장은 하지 않는다.

### 수령한 확장 검사: 기존 checkpoint에 대한 개입

학습 run `paper-20260830T150244764889Z`, model seed 0. 진단 실행 소스는 `5e801c3`이며
보고서 폴더 suffix는 `20260831T120740120251Z`다. 사용자 첨부 로그 SHA-256:
`CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F`.

| 데이터 (validation) | learned C | mean C | shuffled C | graph off |
|---|---:|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.646000 | 0.646000 | 0.632000 |
| CiteSeer accuracy | 0.644000 | 0.642000 | 0.642000 | 0.626000 |
| PubMed accuracy | 0.724000 | 0.726000 | 0.726000 | 0.718000 |
| PPI micro-F1 | 0.487508 | 0.487508 | 0.487508 | 0.452144 |
| arxiv accuracy | 0.509279 | 0.509279 | 0.509279 | 0.508876 |

다섯 dataset 모두 완료, 최종 `Diagnostic status: passed`다. Validation 재검증 오차는
약 0~5e-8이다. 각 조건은 같은 checkpoint에 대한 개입이며 재학습한 baseline의 성능이 아니다.
PPI/arxiv는 두 층 모두 관찰한 C가 상수이고 mean/shuffle의 prediction flip이 0이다.
PPI에서는 graph-off 시 3.5364 percentage points 하락해 연결 구조의 기여와 C 차별화 실패가
구분된다. arxiv에서는 0.0403 points 하락하며 rho 중앙값 약 .000433으로 전파 영향이 매우 작다.

Gradient 검사는 eval/dropout-off, PPI는 첫 train batch 하나다. Gate 파라미터 norm과 task
gradient가 극소지만 학습 당시 Adam update 이력을 복원한 것은 아니므로 WD 원인 확정은 아니다.
기존 full audit의 극소 ratio는 분모 하한 1e-12 적용 여부를 확인해야 한다. 새 2×2의 train-mode
관찰은 raw task/decay norm을 따로 저장하고 0분모 비율만 null로 기록한다.

## 2. 사용자가 제공한 5-seed test 집계

모든 항목의 model seeds는 `0,1,2,3,4`다. `±`는 seed 사이 **표본 표준편차**이며,
표준오차·신뢰구간·5-fold 교차검증이 아니다. 모델·데이터·평가 조건이 다른 트랙끼리
수치를 직접 차감해 기여도를 추정하지 않는다.

### Conductance GAT

Run: `paper-20260830T150244764889Z`.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| Cora | accuracy | 0.661600 ± 0.023639 |
| CiteSeer | accuracy | 0.626800 ± 0.007918 |
| PubMed | accuracy | 0.721200 ± 0.005630 |
| ogbn-arxiv | accuracy | 0.486089 ± 0.002813 |
| PPI | global micro-F1 | 0.500051 ± 0.004877 |

이 값은 실행·집계 결과이지 경쟁력 또는 novelty의 증명이 아니다. 외부 논문 표와
비교하려면 split·입력 전처리·모델 크기·학습 설정·모델 선택 절차를 별도로 확인해야 한다.

### Cycle PE v1

Run: `paper-20260831T015711388279Z`. 모델 키는 **`cycle_set`**이다.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| ZINC-12K | MAE, 낮을수록 좋음 | 0.189090 ± 0.016624 |
| Peptides-struct | MAE, 낮을수록 좋음 | 0.259728 ± 0.002816 |

이것은 cycle 기저를 여섯 통계로 요약하는 v1의 결과다. 좌영공간 기저벡터 전체를 입력하는
**`cycle_basis_v2` 결과가 아니다.** `schema_version=2` 같은 저장 형식 버전과 모델 v2도
구분한다. 같은 backbone의 PE 제외 ablation이 없으므로 이 표만으로 PE의 순수 효과를
분리할 수 없다. 두 데이터셋이 회귀 과제이므로 MAE 사용 자체는 맞다.

### Tree Augmentation

Run: `paper-20260831T060149709584Z`.

| 데이터·평가 조건 | 지표 | Fixed BFS | Multi-chart |
|---|---|---:|---:|
| CSL / seen family | accuracy | 0.400000 ± 0.028260 | 0.768333 ± 0.029404 |
| CSL / unseen family | accuracy | 0.059167 ± 0.009501 | 0.137500 ± 0.002946 |
| ZINC / seen family | MAE | 0.730009 ± 0.081641 | 0.753210 ± 0.156958 |
| ZINC / unseen family | MAE | 0.727205 ± 0.083463 | 0.749209 ± 0.155026 |

- `seen/unseen family`는 원본 그래프 종류가 아니라 **spanning-tree 생성 방식**이다.
  Fixed는 root-0 BFS, multi는 random-root BFS/DFS로 학습한다. 같은 test 그래프에
  fresh random-root BFS(seen)와 학습에서 제외한 Wilson UST(unseen)를 적용한다.
- CSL은 단일 고정 label-stratified 90/30/30 분할이다. 5개 모델 seed는 5-fold가 아니다.
  정확도는 chart별 정확도의 평균이며 여러 chart의 logits을 앙상블한 값이 아니다.
- CSL seen에서는 큰 개선이 있지만 unseen의 절대 성능은 낮다. unseen prediction flip rate도
  0.200000 → 0.319167로 증가하여 모든 chart 변화에 더 안정적이라고 주장할 수 없다.
- ZINC의 평균 MAE는 두 조건 모두 개선되지 않았다. Chart prediction std는 seen에서
  0.053970 → 0.041503, unseen에서 0.053139 → 0.041492로 줄었지만 이는 **한 그래프의
  chart 변경에 따른 흔들림**이지 model-seed 사이 성능 변동 감소나 MAE 개선이 아니다.
- `rounded_exact_vector_accuracy=0`은 정수 count용 일치 지표를 연속값 ZINC에도 노출한
  부적절한 보조 지표다. 코드가 반올림한 예측과 연속 정답의 완전 일치를 검사하므로,
  이를 회귀 학습 실패 또는 정확도 0%로 해석하지 않는다. 아직 코드에서는 제거하지 않았다.
- 현재 Tree 기본 구현은 고정 800 optimizer updates 후 모델을 평가하며 validation-best
  checkpoint 선택을 하지 않는다. 이 내부 fixed-vs-multi 실험을 표준 논문 표와 동일한
  재현이라고 제시하지 않는다. 표의 차이에 대한 paired 유의성 검정은 수행하지 않았다.

## 3. Conductance의 실제 GPU checkpoint 진단

사용자가 `ebf8cd1`을 pull한 뒤 아래 명령을 실행했고,
`Diagnostic status: passed (stdout only)` 출력을 제공했다.

```bash
bash scripts/diagnose_conductance.sh --run-id paper-20260830T150244764889Z --ablate-graph
```

대상은 **당시 기본값**인 model seed 0, Cora/PPI/ogbn-arxiv다. FP32 추론이며 AMP/TF32는 껐다.
이후 CLI 기본 데이터 목록에 CiteSeer/PubMed를 추가했지만 이 과거 로그에는 포함되지 않는다.
새로 계산한 지표는 train/validation뿐이고, test는 기존 저장값만 출력했다.
원래 validation과 재계산값의 차이는 약 `0~5e-8`이다. 소스 불일치 경고는 제공된 출력에 없다.
이 로그는 실제 GPU 추론 완료의 근거지만 전체 seed의 gate 상태나 GPU 가속의 근거는 아니다.

저장된 학습 설정은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005,
weight decay 0.0005, 최대 200 epochs, patience 50, PPI batch size 2다.

### 선택된 checkpoint의 성능과 학습 기록

| 데이터 | Best epoch / 실행 epoch | Eval train 지표 | Eval validation 지표 | Eval train / validation loss |
|---|---:|---:|---:|---:|
| Cora | 23 / 73 | accuracy 1.000000 | accuracy 0.658000 | CE 0.083452 / 1.163915 |
| PPI | 69 / 119 | micro-F1 0.496796 | micro-F1 0.487508 | BCE 0.541717 / 0.536941 |
| ogbn-arxiv | 199 / 200 | accuracy 0.495695 | accuracy 0.509279 | CE 1.879628 / 1.826401 |

Train-mode loss의 first → selected → last는 Cora `2.118194 → 0.720689 → 0.172052`,
PPI `0.699084 → 0.541241 → 0.541466`, arxiv `3.944757 → 2.252732 → 2.251006`이다.
진단의 train도 **dropout을 끈 eval 추론**이다. 학습 중 dropout을 켜고 기록한 loss와
시점·모드가 다르므로 두 값을 직접 비교해 checkpoint 오류라고 판단하지 않는다.

- Cora: 선택된 모델도 train 정확도 100%인 반면 validation은 65.8%다. 훈련 노드 적합은
  충분하며 큰 일반화 차이가 확인됐다. 단순 epoch 부족이라고 보기 어렵다.
- PPI: train에서도 F1이 낮아 현재 표현·정규화·최적화의 적합 부족을 의심한다. validation의
  예측 양성 비율은 0.202331, 실제 양성 비율은 0.294568이다. 제공된 micro-F1과 양성 비율에서
  계산한 precision은 약 0.598629, recall은 약 0.411182다. 원 로그가 직접 출력한 지표와
  이 후처리 계산값을 구분한다. 낮은 recall의 원인은 아직 확정하지 않았다.
- arxiv: 마지막 loss가 최저이고 best epoch가 199/200이라 추가 학습 여지가 있다.
  그러나 아래의 극도로 약한 이웃 전달 문제도 병존한다.
- Cora `23+50=73`, PPI `69+50=119`로 patience 설정과 종료 시점은 일치한다.

### 층별 엣지 가중치와 이웃 혼합량

층 번호는 코드의 `layer 0/1`을 따른다. Cora/arxiv 통계는 전체 transductive 그래프,
PPI 표는 **validation 그래프 두 개**의 node/edge-pooled 통계다. 차수 최대값은
그래프별로 계산한 뒤 통계를 모았다. `rho`는 0~1의 비율이며 퍼센트가 아니다.

| 데이터 / 층 | C 평균 | C의 CV | rho 중앙값 | rho < 0.01 노드 비율 | 전파 상대 변화량 |
|---|---:|---:|---:|---:|---:|
| Cora / 0 | 0.6984 | 6.623e-7 | 0.01696 | 17.91% | 0.02989 |
| Cora / 1 | 2.266 | 0.7224 | 0.02779 | 14.96% | 0.05120 |
| PPI / 0 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09871 |
| PPI / 1 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09498 |
| arxiv / 0 | 0.6933 | 0 | 0.0004331 | 99.35% | 0.003041 |
| arxiv / 1 | 0.6933 | 0 | 0.0004330 | 99.35% | 0.002101 |

CV는 `std(C)/mean(C)`이고, 전파 상대 변화량은 LayerNorm/ELU 이전
`||H_after_conv - H_before_conv|| / ||H_before_conv||`이다. PPI train 그래프 20개에서도
두 층의 C CV는 0이며 rho 중앙값은 0.03308이다.

PPI/arxiv의 **관측한 FP32 eval 입력**에서 C가 상수인 현상이 확인됐다. 표시는 작은 양수도
지수 표기로 출력하므로 `CV=0`은 단순 표시 반올림 설명으로 지울 수 없다. C는 FP32로
chunk 재계산하고 분산은 float64로 집계했다. 전체-batch GEMM과의 bitwise 동일성이나
모든 가능한 입력에서 함수가 상수라는 결론까지 보장하지는 않는다.

Cora의 layer 1에는 비상수 C가 있으므로 구현이 항상 C를 상수로 강제한다는 설명은 틀린다.
또 진단은 학습 중 C 궤적을 기록한 것이 아니므로 학습 내내 C가 고정돼 있었다고 할 수 없다.

이 과거 checkpoint와 변경하지 않은 기본 benchmark의 전파식은 다음과 같다.

\[
H'=H-\frac{0.95}{d_{\max}^C}B^\top C B H,\qquad
\rho_i=0.95\frac{d_i^C}{d_{\max}^C}.
\]

`C=cI`이면 공통 c가 상쇄되어 `H'=H-(0.95/d_max)L_unweighted H`가 된다.
따라서 관측한 PPI/arxiv 입력에서는 적응적 엣지 가중치가 아니라 동일 가중치의 전파로
작동한다. arxiv rho 중앙값 `0.0004331`은 이웃 총가중치 **0.04331%**다.
이는 안정성용 스텝 선택의 효과이며 `L=B^T B`라는 라플라시안 정의 자체의 필수 조건이 아니다.

### 같은 checkpoint에서 전파만 우회한 validation 결과

| 데이터 / 지표 | 원래 값 | 전파 우회 값 | 우회 − 원래 |
|---|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.632000 | -0.026000 = -2.6000%p |
| PPI micro-F1 | 0.487508 | 0.452144 | -0.035364 |
| arxiv accuracy | 0.509279 | 0.508876 | -0.000403 ≈ -0.04027%p |

전파 우회 값은 원 로그의 원래 지표와 delta를 합산했다. arxiv의 차이는 validation
29,799개 노드에서 정답 수 순감소 12개에 해당하며, 예측이 12개만 바뀌었다는 뜻은 아니다.
우회는 encoder/LN/ELU/decoder를 남기는 추론 개입이다. 별도 MLP를 재학습한 baseline이나
저성능 원인을 단독으로 증명하는 실험이 아니다. Cora/PPI에는 분명한 지표 하락이 있으므로
모든 데이터에서 그래프 연산을 전혀 사용하지 않는다고 말하면 안 된다.

## 4. 미확정 원인과 다음 검증

기존 checkpoint에서 관측된 상수 C와 약한 전달량은 사실이다. 이후 2×2 새 학습에서
node-degree 정규화가 개선을 이끈 결과를 확보했지만, 모든 데이터·seed에 대한 원인 설명이나
학습 C의 필요성까지 확정된 것은 아니다.
`softplus(0)+1e-5 ≈ 0.693157`이므로 PPI의 C 평균은 softplus 이전 gate raw logit이
0 부근인 상황과 맞는다.
이것만으로 모든 gate 파라미터가 0이라고 증명되지는 않는다. `softplus'(0)=0.5`이므로
이를 softplus의 출력 포화라고 설명하는 것도 부정확하다.

확장 audit에는 gate 입력·raw logit·gradient 검사가 있고, 2×2 학습에는 실제 train-mode
첫 batch의 task gradient와 decay 관찰값이 있다. Gate WD 제거 후 선택된 C의 변동이
커졌다는 결과와 WD 제거가 성능을 높인다는 주장은 분리한다.

`learned_c`/`fixed_c=1`의 네 새 학습과 **그 새 run의 learned checkpoint** 평균-C 개입은
모두 사용자 GPU 보고서를 수령했다. PPI의 현재 checkpoint 의존도가 크다는 결과와
fresh-training 이득을 관측하지 못했다는 결과를 함께 보존한다. 이전 2×2 `node_degree` 검사도
지원하지만 다른 source run의 결과로 분리한다.

사용자 보고상 [직접 C v2](CONDUCTANCE_V2.md)와
[상대 C v3](CONDUCTANCE_V3.md)의 **과거 arxiv-only runner**는 `passed`다. V2는 같은 그래프에
묶인 direct/fixed C이고, v3는 공유 상대-C 생성기의 relative/fixed C이며 fixed v3도 alpha는
학습한다. 둘 다 기존 MLP의 수학 오류 수정이 아니다. 각 버전 내부 비교를 먼저 보고,
파라미터화·정규화·전파 강도·optimizer가 함께 다른 버전 간 차이를 단일 요인으로 해석하지 않는다.
다만 v2/v3 성능 수치와 전체 artifact는 미수령이고, 과거 arxiv-only V4는 partial 첫 arm 뒤
중단됐다. 현재 확대된 V2/V3/V4 8/10/20-job 정식 결과는 없다.

노드별 정규화는 기존 대칭성·보존성의 의미를 바꾸는 실험이므로 단순 속도 최적화나
버그 수정으로 부르지 않는다. 기존 기본 benchmark는 유지한다. 다른 model seed의
일반화, 확대된 v2/v3/V4 전체 행렬과 Cycle PE v2의 수치·전체 artifact 독립 검증,
GPU 가속 실측은 여전히 별도 검증 대상이다.

## 5. 근거와 검증 범위

| 근거 | 식별자 / SHA-256 |
|---|---|
| 사용자 제공 5-seed 집계 출력 | 첨부 `d4acb1eb-bd9d-40ef-af24-e5f7ba34f138`; `CEF76E8494C462E8302AF2811CCCD19BBB6D8DC8266DB852866237ED95DD5CEC` |
| 사용자 제공 GPU 진단 stdout | 첨부 `5db2e997-c8ab-495e-b762-c32fa620c02c`; `C0E89FC76A438D1707FE90C889923390FDF8277F05780B2811FF4D444DD01A21` |
| 사용자 제공 5e801c3 full-audit stdout | 첨부 `c4abbad1-654a-4f5e-a774-f84f7e88e4dd`; `CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F` |
| 사용자 제공 43afd63 2×2 GPU 학습·비교 stdout | 첨부 `20b4a93d-06ed-4cff-9fe5-530eacf39766`; `2C78D02BB210BF00865AB7207DF651B02B2081EE4FAE6E8A6A83665A5D331161` |
| 사용자 제공 C-learning GPU 비교 | 2026-09-01 inline 보고서, `gat-c-learning-seed0-v1`; 별도 첨부 파일/SHA 없음, 실행 revision 미확인 |
| 사용자 제공 C-learning 평균-C GPU 검사 | 2026-09-01 inline terminal 출력, revision 표시 `8f6b4da`, `gat-c-learning-seed0-v1`의 각 데이터셋 `learned_c`; 별도 첨부 파일/SHA 없음 |
| 사용자 제공 A6000 V5·Cycle V2 r1 실패 stdout | 첨부 `606865d2-9e2f-4818-a7dd-b4599ab165ae`; `D6394C813EDB52DD2CE6746B531973A750A1F0AE807EC5A3410B938FAFF6F93E` |
| 사용자 제공 A6000 V5·Cycle V2 r2 old-source 재현 stdout | 첨부 `bd63fc9a-60da-4daf-9ab9-da49db7cbbe1`; `F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6` |

이 hash는 **제공된 텍스트 파일의 hash**이며 서버의 checkpoint/원본 데이터 hash가 아니다.
개인 서버 계정·호스트 경로와 원본 로그 전체는 이 문서에 복제하지 않았다.

확장 검사 구현 전 문서 갱신의 로컬 회귀는 619 passed / 63 skipped (21.84 s, exit 0),
당시 진단 전용은 42 passed였다. Ruff/diff 및 당시 문서 로컬 링크 34개 검사도 통과했다.
당시 확장 검사 결과는 handoff의 해당 역사 검증 항목을 따른다.
단일 seed·확장 진단 구현 후 전체 회귀는 **680 passed / 63 skipped**, 진단 전용은
**89 passed**다. 이는 당시 로컬 단위 검증이며, 이후 수령한 실제 GPU full-audit 로그는 위에
별도로 기록했다. 후속 2×2 구현 후 전체 검사는 **794 passed / 64 skipped** (31.83 s, exit 0),
Ruff 통과다. 기존 생략 사유에 Windows 실제 symlink 권한 1개가 추가됐으며 차단 로직은 별도
mock 검사로 확인했다. 이 수치는 2×2 코드 구현 당시의 로컬 검사이며, 이후 수령한
43afd63의 실제 GPU 재학습·성능 비교는 위에 별도 기록했다.
기존 게시 학습 코드 `a64c235`와의 진단 호환성도 메모리 로딩을 통한 42개 단위 검사로 확인했다.
생략된 63개는 Linux/Bash 전용 62개와 로컬 PyG 미설치 1개다. 이번에도 Windows faulthandler의
`access violation` 메시지가 있었으나 pytest는 위 결과와 exit 0까지 실행됐다.
이 호스트 경고를 Linux 성공의 근거로 해석하지 않는다. 로컬 단위 검사는 GPU 학습
또는 가속 배수의 검증을 대신하지 않는다. 현재 소스 스냅샷 checksum은
[HANDOFF.md](HANDOFF.md)의 코드 스냅샷 항목을 따른다.
