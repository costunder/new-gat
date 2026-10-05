# 초기 연구: 전체 실험 이력과 증거

기준일: 2026-10-05. 이 문서는 **현재 wedge/local-energy 연구로 방향을 바꾸기 전의 기록**이다.
아래 실험의 성공·실패를 현재 E/J·copy 결합·normalization 모델의 성능 원인으로 옮겨 붙이지 않는다.
실험을 다시 실행하는 안내가 아니라 GPT가 이전 근거와 현재 연구를 구분하기 위한 목록이다.

## 증거를 읽는 기준

- **실제 과학 실험인 작은 합성 연구:** 정해진 입력·목표를 학습하거나 대수 성질을 측정한 결과다. 작은 그래프라고 모두 DEBUG는 아니다.
- **DEBUG·smoke·단위 검사:** 모델 경로와 수식의 구현 검사다. 공식 데이터 성능 결과로 세지 않는다.
- **실제 데이터 calibration:** 전체 입력 규칙을 유지한 자원·gradient 측정이다. 최종 학습이나 공식 test를 대신하지 않는다.
- **서버 완료 출력:** 수령한 콘솔에 실제 학습·평가 완료가 기록된 범위다. 서버 원본 파일의 독립 재평가와 다르다.
- **구현·계획·실패:** 전체 결과가 없으면 그대로 표시한다. 과거 계획표의 조건 수를 완료 횟수로 합산하지 않는다.

이 문서는 [이전 종합 정리](../gpt_experiments_20261004/03_EARLIER_HISTORY.md)를 기반으로
직접 읽은 작은 합성 산출물과 원본 상태 문서를 추가한 새 전달 문서다.
기존 문서·코드·결과는 변경하지 않았다.

## 1. 버전 이름부터 구분

| 이름 | 뜻 |
| --- | --- |
| Conductance V1–V5 | `research/conductance_gat/`의 서로 다른 C 생성·전파 모델 |
| CGAT 연구 v1/v2 | 이후 `experiments/`에서 기본 집계, 정보 흐름, C 학습을 점검한 연구 단계 |
| Cycle PE v1/v2 | 정적 cycle-space 특징 연구. Conductance와 별도 트랙 |
| aggregation 구현 v3/v4 | 집계 비교 구현의 수정 버전. Conductance V3/V4와 다름 |
| wedge / local-energy | 이번 묶음의 현재 연구. 위 모델의 결과를 동일 실험으로 합산하지 않음 |

기본 경계는 [RESEARCH_OVERVIEW](../../docs/RESEARCH_OVERVIEW.md)를 따른다.
작은 합성 결과, 공식 데이터 본학습, checkpoint 개입, 테스트 실행 성공을 서로 구분한다.
대부분의 아래 서버 결과는 사용자 제공 콘솔·보고서를 기록한 것이다.
서버 원본 checkpoint·manifest·history를 현재 로컬에서 직접 읽고 재평가한 증거는 아니다.

## 2. 최초 독립 트랙: 5-seed test 집계

모델 seed는 0, 1, 2, 3, 4다. ±는 표본 표준편차이며 신뢰구간이나 5-fold 결과가 아니다.
근거는 [EXPERIMENT_STATUS의 5-seed 집계](../../gpt_handoff/EXPERIMENT_STATUS.md)에 있다.

| 트랙 / run | 데이터 | Test 평균 ± 표본 std |
| --- | --- | ---: |
| Conductance V1 / `paper-20260830T150244764889Z` | Cora accuracy (%) | 66.1600 ± 2.3639 |
| 동일 | CiteSeer accuracy (%) | 62.6800 ± 0.7918 |
| 동일 | PubMed accuracy (%) | 72.1200 ± 0.5630 |
| 동일 | ogbn-arxiv accuracy (%) | 48.6089 ± 0.2813 |
| 동일 | PPI global micro-F1 (%) | 50.0051 ± 0.4877 |
| Cycle PE v1 / `paper-20260831T015711388279Z` | ZINC-12K MAE | 0.189090 ± 0.016624 |
| 동일 | Peptides-struct MAE | 0.259728 ± 0.002816 |

Cycle v1의 모델 키는 `cycle_set`이다. cycle 기저를 여섯 통계로 요약한 모델이며
기저 전체를 입력한 구 `cycle_basis_v2`나 현재 sparse DFS v2 결과가 아니다.
같은 backbone에서 PE를 뺀 대조군이 없어 위 MAE만으로 PE의 순수 기여를 분리할 수 없다.

Tree Augmentation의 run은 `paper-20260831T060149709584Z`다.

| 데이터 / 평가 tree family | 지표 | Fixed BFS | Multi-chart |
| --- | --- | ---: | ---: |
| CSL / seen | accuracy (%) | 40.0000 ± 2.8260 | 76.8333 ± 2.9404 |
| CSL / unseen | accuracy (%) | 5.9167 ± 0.9501 | 13.7500 ± 0.2946 |
| ZINC / seen | MAE | 0.730009 ± 0.081641 | 0.753210 ± 0.156958 |
| ZINC / unseen | MAE | 0.727205 ± 0.083463 | 0.749209 ± 0.155026 |

seen/unseen은 같은 test 그래프에 적용하는 **spanning-tree 생성 방식**이다.
Fixed는 root-0 BFS, multi는 random-root BFS/DFS로 학습하고, 새 BFS 및 학습에서 제외한
Wilson UST로 평가했다. CSL은 고정 90/30/30 label-stratified split이다.
CSL seen에서는 개선됐지만 unseen의 절대 성능은 낮았고 ZINC MAE는 개선되지 않았다.
고정 800 optimizer update 후 평가한 내부 대조이며 validation-best 선택 실험이 아니다.
Tree의 정확한 chart 전환 수학과 downstream 조건은 [TREE_AUGMENTATION](../../docs/TREE_AUGMENTATION.md)를 참조한다.

## 3. Conductance V1의 원인 분리

초기 checkpoint에서는 PPI/arxiv의 C가 상수였고, global-max weighted-degree 정규화가
특히 arxiv의 이웃 전달량을 작게 만들었다. 평균·셔플 C 개입과 전파 제거를 별도로 검사한 뒤
gate weight decay와 degree 정규화를 2×2로 바꿔 새로 학습했다.

`gat-factorial-seed0-v1`은 PPI/arxiv × 4조건 = 8학습이 모두 `passed`로 보고됐다.
Seed 0, hidden dimension 64, 2층, 최대 200 epoch, patience 50, **validation only**다.

| 조건 | PPI micro-F1 (%) | arxiv accuracy (%) |
| --- | ---: | ---: |
| baseline: global-max / gate WD 유지 | 48.986770 | 50.927883 |
| global-max / gate WD 제거 | 49.378028 | 50.565451 |
| node-degree / gate WD 유지 | 52.465469 | 68.317723 |
| node-degree / gate WD 제거 | 50.340520 | 67.995566 |

이 실행에서는 정규화 변경이 주요 개선 요인이었다. 비상수 C를 만드는 것 자체가
점수 개선과 같지는 않았다. 원문·설정은 [CONDUCTANCE_FACTORIAL_FINDINGS](../../docs/CONDUCTANCE_FACTORIAL_FINDINGS.md)에 있다.

후속 `gat-c-learning-seed0-v1`은 같은 node-degree 설정에서 learned/fixed C를 새로 학습했다.
PPI/arxiv × 2조건 = 4학습이 모두 `passed`로 보고됐다. 과거 factorial의 learned 점수를 재사용하지 않았다.

| 데이터 | learned C validation (%) | fixed C=1 validation (%) | learned − fixed (pp) |
| --- | ---: | ---: | ---: |
| PPI | 52.564966 | 52.705738 | −0.140772 |
| arxiv | 68.317723 | 68.324435 | −0.006711 |

같은 learned checkpoint의 C를 그래프·층별 평균으로 바꾸는 후속 검사도 `passed`로 보고됐다.
전체 층 평균화에서 PPI는 −6.649440pp, arxiv는 −0.033558pp였다.
이는 **학습된 checkpoint의 C 의존성**이며, 별도 fixed 모델보다 그만큼 좋아졌다는 뜻이 아니다.
한 seed·validation 결과로 C의 보편적 무용성·동등성·test 우위를 주장하지 않는다.
세부 기록은 [CONDUCTANCE_C_LEARNING_FINDINGS](../../docs/CONDUCTANCE_C_LEARNING_FINDINGS.md)를 따른다.

## 4. Conductance V2–V5와 후속 구조 비교

| 실험 | 모델·질문 | 확인된 실행 상태와 한계 |
| --- | --- | --- |
| V2 | 고정 그래프의 물리 엣지마다 C 파라미터 직접 학습 | 과거 arxiv-only 2-job run `gat-direct-c-v2-gpu6-seed0-v1`의 `passed` 보고만 수령. 점수·전체 artifact 없음. 현재 4데이터 × 2조건 = 8job 완료 결과로 확대하지 않음 |
| V3 | 공유 상대 C 생성기와 전파 강도 분리 | 과거 arxiv-only 2-job `gat-relative-c-v3-gpu6-seed0-v1`의 `passed` 보고만 수령. 현재 5데이터 × 2조건 = 10job의 점수·전체 결과 없음 |
| V4 | 상대 C × identity/학습 spatial W, 2×2 | 과거 arxiv-only 첫 `fixed_c_identity_w` arm만 200epoch·child exit 0 뒤 report gate 중단. 나머지 3arm pending. 현재 5데이터 × 4조건 = 20job 정식 결과 없음 |
| 초기 V5 r1/r2 | graph-conditioned C / projector Cycle v2와 함께 실행 | fixed-C 1job 완료 뒤 dynamic 최초 calibration의 44.47/44.55GiB OOM. 나머지 Conductance 18job 미실행. Cycle 4job 첫 epoch 전 non-finite gradient 실패. r2는 수정 소스 미반영으로 동일 실패 재현 |
| V5 r3 | score checkpoint 수정 후 large dynamic | 수정 소스 실행 중 diffusion 192MiB 할당 OOM 재발. 이 일부 로그로 나머지 조건 완료를 추정하지 않음 |
| corrected V5 | optimization C, reference/large × 5데이터 × fixed/dynamic | `corrected-c-v5-a6000-gpu3-seed0-v1-conductance` 20조건 학습 `passed`, 3,832개 연속 epoch 출력 및 dynamic 10조건 공식 test 수령. fixed-C test는 미수령 |
| corrected V5 checkpoint 감사 | C=1/mean/shuffle 및 frozen-H solver 참조 | validation 감사 20/20 성공 출력 수령. K8의 수렴 완료나 충분성을 입증한 것은 아님. C 사용과 별도 fixed 재학습 이득을 구분 |
| V5 독립 부분그래프 배치 후속 | sampling overlap 기록, `auto_disjoint`/`cluster_disjoint` | 구현·로컬 계약 검사. corrected V5 점수를 새 sampling의 성능으로 재명명하지 않음 |
| V5 multi-C mechanisms | per-head C 등 12조건, 총 120학습 계획 | 구현·로컬 검사·dry-run. 실제 전체 GPU 학습/test 결과 미수령 |
| V5 edge selection | zero gate, forest/chord, 구조 11 + corruption 2조건, 총 130학습 계획 | `edge-selection-v5-gpu1-seed0-v1`의 full/reference/arxiv 1조건 56epoch 학습 종료·best val 72.1098% 뒤 quantile 감사 실패 수령. 전체 130조건 완료 자료 없음. 분위수·GPU 재할당 복구는 코드/로컬 검사이며 서버 성공으로 합산하지 않음 |

V2/V3의 과거 보고 source는 `7b4cd32`, GPU는 A100 MIG 1g.10gb다.
물리 A100 80GB 이름을 해당 프로세스의 실제 할당 VRAM으로 사용하지 않는다.
설계 근거: [V2](../../gpt_handoff/CONDUCTANCE_V2.md), [V3](../../gpt_handoff/CONDUCTANCE_V3.md),
[V4](../../gpt_handoff/CONDUCTANCE_V4.md), [V5](../../gpt_handoff/CONDUCTANCE_V5.md).
날짜별 실행·실패·감사 원문은 [EXPERIMENT_STATUS](../../gpt_handoff/EXPERIMENT_STATUS.md)에 보존돼 있다.

corrected V5의 핵심 수치만 다음에 남긴다. validation Δ는 dynamic − fixed, test는 dynamic만이다.
Seed 0, reference는 8층/hidden 256/8 head, large는 12층/hidden 384/8 head다. 현재 2층 E/J 모델과 다르다.

| 데이터 | validation Δ reference / large (pp) | dynamic test reference / large (%) |
| --- | ---: | ---: |
| Cora | +0.400 / +0.800 | 76.8000 / 75.4000 |
| CiteSeer | −0.400 / −2.000 | 67.0000 / 61.4000 |
| PubMed | −1.400 / +0.400 | 73.6000 / 70.0000 |
| arxiv | +0.151 / +0.060 | 70.4010 / 71.1849 |
| PPI micro-F1 | +0.123 / +0.038 | 99.0947 / 99.2076 |

PPI C=1 checkpoint 개입은 약 22.49/26.37pp 하락했으나 fresh fixed 대비 이득은 작았다.
arxiv reference dynamic의 epoch 시간에는 성능 수정 전후 구간이 섞여 있다.
전체 [rich scaling](../../gpt_handoff/RICH_SCALING_EXPERIMENTS.md)의 126학습/122child 계획이
완료됐다는 근거는 아니다. V5·Cycle·Tree 등 서로 다른 레시피의 시간·점수를 합쳐 우위를 계산하지 않는다.

## 5. Cycle v2, Tree, 보류된 결합 prototype

| 대상 | 실제 내용 | 상태 |
| --- | --- | --- |
| 구 Cycle `cycle_basis_v2` | 기저 raw-column 모델 | 폐기된 역사적 모델. 과거 runner `passed` 보고를 현재 DFS 모델 결과로 재사용하지 않음 |
| 구 Cycle projector v2 | 기저 변환 불변 projector, QR 사용 | 공식 cache 준비 및 FP16 gradient 실패 기록. 유효 성능 결과와 현재 QR-free 모델을 구분 |
| 현재 Cycle sparse DFS v2 | 같은 DFS 구조 SE 대 SE+cycle 상대 PE | 구현·CPU 수학/계약 검사. 두 데이터·두 profile·SE/PE 총 8학습 및 선택 checkpoint 4test의 새 전체 GPU 결과 없음 |
| Tree core/paper | full cycle-rank chart 전환, fixed BFS 대 multi-chart, sampler-family OOD | 정확한 전환 수치 검증과 2절의 공식 데이터 test 집계를 구분. 새 scaling 전체 완료 자료 없음 |
| `combined_later` | learned C, persistent cycle, flow completion, tree-equivariant prototype 결합 | 보류. 작은 합성 `certification`, `fixed_c`, `identifiability` 산출물 보존. 현재 두 독립 기여의 benchmark 증거로 사용하지 않음 |

Cycle v2의 현재 모델 ID는 `cycle_dfs_se_v2` / `cycle_dfs_relative_pe_v2`다.
기저 방향·열 순서 불변성과 임의 spanning-tree 변경 불변성은 같은 보장이 아니다.
근거: [Cycle v1](../../docs/CYCLE_PE.md), [Cycle v2](../../gpt_handoff/CYCLE_PE_V2.md),
[Tree](../../docs/TREE_AUGMENTATION.md), [보류 prototype](../../docs/COMBINED_LATER.md).
Git에 보존된 `research/{conductance_gat,cycle_pe,tree_augmentation}/results/`의 작은 합성 결과는
위 공식 데이터 점수·새 GPU 실험을 대체하지 않는다. cycle-membership 100%는 구 진단 목표를
특징이 직접 드러내는 사례여서 PE 일반화 증거가 아니다.

## 6. incidence 집계·샘플링과 CGAT 연구 v1/v2

| 경로 | 실제 질문·모델 | 확인한 범위 |
| --- | --- | --- |
| `experiments/incidence_ablation/` | residual/FFN을 포함한 기존 backbone에서 lift·bilinear 8조건 | PPI reference seed0 run `incidence-ppi-reference-mig10gb-gpu4-seed0-v2`, 학습/감사 8/8 `passed` 콘솔 수령. baseline val F1 98.8320%, post-lift 98.9503%(+0.1183pp). bilinear 결합의 일관된 추가 이득 없음. test·다중 seed 결과 아님 |
| `experiments/aggregation_comparison/` | incidence 내부 16조건 + GCN/SAGE/GATv2/DUALFormer/no-skip, arxiv 21조건 | 구현·합성 CUDA 검사. 실제 arxiv full/sampled × fixed/learned 4조건 calibration 완료, 최종 200epoch/공식 test 비교는 미완료 |
| `experiments/sampled_inductive/` | PPI 전용 full/sampled × fixed/learned, graph-held-out 구현 | 사용자가 요청한 arxiv 범위와 달랐던 구현. production launcher 폐기. 구현 존재를 원래 연구 완료로 세지 않음 |
| CGAT v1 보존 | 동일 sampled 노드·엣지에서 각 층 전파·국소 층간 에너지 | 기준 commit `f071bcddb92b9253628c3087bf7931be37205e2b`, tracked 462파일 manifest. ignored/untracked 결과까지 완전 보존 목록이라는 뜻은 아님 |
| `experiments/information_flow_v2/` | 송신/수신 집합을 명시하고 수신 집합 축소. q² 요약, 경계 에너지, 동일 엣지 층간 내적 | 25개 합성 단위/CUDA 검사와 실제 arxiv 8층/256/8head의 짧은 calibration. base/within/between/both 본학습·독립 그래프 일반화 미완료 |
| `experiments/c_learning_only/` | 기존 C 생성기의 학습·C=1 개입·fresh learned/fixed 비교 | 12개 DEBUG 검사. 당시 실제 데이터 200epoch 본학습 결과 없음. 기존 생성기의 내부 8회 최적화·문맥 beta 유지 |
| `experiments/c_learning_bracket/` | Q/K 대칭 내적 → exp로 head별 양의 C | 실제 calibration과 signal audit, 평가 대상 ID/CE 설명 수정 후 33개 DEBUG 검사. standalone 200epoch learned/fixed 비교는 미완료 |
| `experiments/output_init_ablation/` | 출력 Linear baseline/Kaiming × learned/fixed C | 아래 raw-exp 실패 및 log-row 복구의 일부 완료 기록. 최종 4조건 비교표 미수령 |

`information_flow_v2`는 매 층 부분그래프를 새로 무작위 추출하는 모델이 아니다.
요약 분기는 예측·CE에 연결돼 있으나 원래 모든 메시지를 복원·보관하거나 샘플 밖 누락을 채우는 모델도 아니다.
알려진 양의 C·모든 노드 집계·`q=CBV` 제약에서는 q가 복원될 수 있다는 조건을 따로 검사했다.
cycle 공간의 존재를 복구 불가능한 실제 정보 손실과 동일시하지 않는다.

aggregation calibration은 로컬 RTX 5070 Ti에서 7GiB allocator 상한으로 4/4조건을 통과했다.
전체 진단 포함 peak reserved는 6.936/6.811/6.902/6.576GiB였다.
해당 후보에서 sampled learned train pass 450.574초, full 20.378초로 sampling 속도 우위가 없었다.
이는 자원 측정이며 최종 분류 성능·서버 MIG 적합성·전체 학습 완료가 아니다.
근거: [ARXIV_MEMORY_REVIEW](../../docs/ARXIV_MEMORY_REVIEW_20260927.md),
[AGGREGATION_COMPARISON](../../docs/AGGREGATION_COMPARISON.md), [SAMPLED_INDUCTIVE](../../docs/SAMPLED_INDUCTIVE.md),
[이전 CGAT 상세 기록](../../research/wedge_propagation/gpt_handoff/06_PRIOR_TRACK_HISTORY.md).

bracket 후보의 score는 `(Q_u·K_v + Q_v·K_u)/(2r)`, C는 `exp(score)`다.
bracket 논문의 양의 대각 C 생성 부분을 참조한 적용이며 논문의 전체 dynamics 재현이 아니다.
Q/K는 로컬 엣지 특징을 쓰지만 beta에는 기존 그래프 문맥이 남는다.
초기 깊은 층 C≈1이나 RMS 감소만으로 학습 실패 원인을 확정하지 않았다.

## 7. 출력 초기화 실험의 완료·실패를 분리

8층/256채널/8 head, ogbn-arxiv, seed 0, 200 epoch로 네 조건을 학습하는 실험이다. 공식 test는 사용하지 않는다.

- raw-exp `output-init-server-20260929-163340`: baseline learned epoch1 완료,
  240.2초/epoch, validation 7.6278%. epoch2의 `exp produced nonfinite/zero C`와 code −6 기록.
  네 조건 전체 완료나 현재 E/J 모델의 학습 실패로 분류하지 않는다.
- 후속 log-row: 정규화 exp 비율을 안정적으로 계산하는 별도 소스/run이다.
  `output-init-log-row-server-20261001-172250`의 Kaiming learned 200/200,
  마지막 CE 0.62667·validation 72.5897% 뒤 소스 계약 재검사 오류가 기록됐다.
  이 값은 마지막 epoch이며 best/test 점수가 아니다.
- 복구 원문 `28537dca-db0f-49a6-9c49-afc35f80d2e8`은 baseline learned/fixed 및
  Kaiming learned 세 조건 완료·CUDA 재검증을 보고했다. Kaiming fixed는 새로 시작해
  epoch4/200(CE 1.43837, validation 64.3210%)까지 기록되고 epoch5 시작 뒤 출력이 끝난다.
  **마지막 조건의 200 epoch 완료·최종 `comparison.json`은 제공된 자료에서 확인하지 못했다.**

근거: [출력 초기화 README](../../experiments/output_init_ablation/README.md),
[수치 안정성](../../experiments/output_init_ablation/NUMERICAL_STABILITY.md),
[RECOVERY](../../experiments/output_init_ablation/RECOVERY.md), [이전 CGAT 상세 기록](../../research/wedge_propagation/gpt_handoff/06_PRIOR_TRACK_HISTORY.md).
서버 터미널 진행 표시, Git 소스 반영, 단일-GPU 지정, 수치 계산 및 계약 복구는 각각 실행 문제다.
이를 새로운 과학적 ablation이나 모델 정확도 개선 결과로 집계하지 않는다.

## 8. 추가 원문이 있어야 확정할 수 있는 것

| 부족한 자료 | 확정하지 못하는 항목 |
| --- | --- |
| V2/V3의 `comparison.*`, manifest, history | 과거 `passed`의 성능 수치, 현재 확대된 전체 matrix 완료 |
| V4 나머지 arm 및 전체 보고서 | 2×2 C/W 효과와 20job 정식 결과 |
| corrected V5의 fixed-C 공식 test | 학습 C의 paired test 개선량 |
| edge-selection 전체 manifest/audit | 130조건 완료 및 source/GPU 재할당 수정 후 서버 재개 성공 |
| Cycle sparse DFS v2 전체 보고서 | SE 대비 상대 PE의 공식 데이터 효과 |
| output-init 복구 최종 `comparison.json`/`recovery-completed.json` | 네 조건 모두 완료 및 최종 초기화·C 효과 |
| 과거 서버 원본 checkpoint·manifest·history | 사용자 보고를 넘어서는 artifact 독립 재검증 |

옛 첨부 일부는 문서에 `C:/Users/lock1/.codex/attachments/` 경로로 남아 있다.
이번 묶음은 해당 파일의 현존이나 추가 재실행 성공을 추정하지 않는다.
보존된 인용·hash·epoch/test 부록의 출처는 각 기존 문서를 따른다.

**현재 연구에 가져오는 교훈은 세 가지다.** 값이 변하는지, checkpoint가 그 값을 사용하는지,
동일 조건에서 새로 학습했을 때 좋아지는지를 나눠 확인해야 한다.
이전 깊은 CGAT의 수치 오류·신호 감소가 현재 2층 local-energy 모델의 원인이라는 근거는 없다.

## 9. 저장된 작은 합성 연구: DEBUG와 공식 benchmark 사이를 구분

아래 산출물은 현재 저장소에서 직접 읽었다. 이번 전달 작업에서 새로 학습하거나 재평가하지 않았다.
공식 데이터의 결과가 아니며, 해당 연구 목표에서 관측한 값만 기록한다.
원본 상태 문서의 2026-09-16 보존 목록은 JSON 6개·CSV 4개·checkpoint 1개·PNG 2개,
총 13개/346,743 bytes다. 아래 여섯 연구 요약이 그 JSON 6개에 대응한다.
반복 smoke·재감사·tiny cache는 별도 개발 검증이며 새로운 과학 학습 결과로 합산하지 않는다.

| 연구·원문 | 실제 입력·예산 | 핵심 관측 | 해석 범위 |
| --- | --- | --- | --- |
| [Conductance summary](../../research/conductance_gat/results/summary.json), [학습 history](../../research/conductance_gat/results/learned_history.csv) | CPU, seed 17, 고정 그래프 N=28/E=49, train excitation 144개/test 48개, history 마지막 epoch 500, step .03 | Learned/isotropic node-update RMSE .00104489/.04645564, flux RMSE .0180034/.876096, C RMSE .00642642/.623936, excited-edge C 상관 .999947 | 특정 고정 그래프·excitation의 teacher 회수. 다른 그래프의 규칙 일반화·분류 성능이 아님 |
| [Cycle PE summary](../../research/cycle_pe/results/summary.json) | Train graph 16개(201 edge), held-out family test 16개(257 edge), cycle-membership 목표 | Test accuracy: degree-only .260700, raw .377432, cycle-set/projector leverage 1.0 | Cycle support 특징이 목표 자체를 직접 드러내는 진단. 100%를 다른 downstream task나 새 PE의 우위로 옮기지 않음 |
| [Tree summary](../../research/tree_augmentation/results/summary.json) | N=12/E=18, cycle rank 7 전체 보존, training tree 16개, unseen chart probe | Lossless 전환 오차 9.93014e−16, cocycle 오차 0. Fixed/multi unseen MSE .0289956/.0283638, 비율 .978213 | 정확한 chart 전환과 같은 그래프의 새 chart probe. 공식 CSL/ZINC 결과와 별도 |
| [Combined E0](../../results/combined_later/certification.json) | Seed 2026, graph 5개, N=9, extra edge 7개, depth 4, tolerance 1e−9 | passed=true, reconstruction 최대 9.99201e−15, orientation-layer 오차 0 | Archived prototype의 대수 certification. 현재 copy 모델이나 benchmark 학습 결과가 아님 |
| [Combined fixed-C](../../results/combined_later/fixed_c/summary.json) | CPU, N=14/E=22/rank 9, sample 192개, 관측 edge 6개, 40 epoch | Physical missing RMSE .0766073, raw coordinate .135110, unseen raw .179200. Physical chart variation 1.77636e−15, conservation 오차 1.77636e−15 | 알려진 C·hard observation 조건의 작은 completion 연구. 임의 샘플링에서 미관측 메시지 복원을 입증하지 않음 |
| [Combined identifiability](../../results/combined_later/identifiability/summary.json) | N=16/E=25/rank 10, noise std .02, ridge .01 | 동일 divergence 쌍 오차 3.77476e−15. Full rank가 된 관측 수: chord-first 10, random 11, rank-greedy 10, tree-first 15. Noiseless relative error 최대 1.51682e−14 | 관측 rank·cycle ambiguity의 조건 검사. 새 GNN의 정확도·일반화 근거가 아님 |

Combined 결과는 [보류 상태 문서](../../docs/COMBINED_LATER.md)에 따라 독립 트랙의 성능 증거에서 제외한다.
원래 산출물의 과거 로컬 경로 metadata는 역사적 기록이다. 지금 경로에서 새 실행됐다는 의미가 아니다.

## 10. 실제 예산과 계획을 대조할 때 필요한 보충

### Corrected V5의 요청 epoch와 실제 update

[원본 20조건 표](../../gpt_handoff/EXPERIMENT_STATUS.md)에 실제 best/last epoch와 update가 있다.
요청 200 epoch와 reference-update 예산을 같은 숫자로 읽지 않는다.

- PPI는 기준 graph batch 8에서 3 update/epoch × 200 = 600 update였다.
  선택된 physical batch 20에서는 1 update/epoch이므로 네 PPI 조건이 각각 600 epoch를 실행했다.
- Arxiv reference는 12 batch/epoch, fixed/dynamic 각각 마지막 epoch 201, 2,412 update다.
  Large는 45 batch/epoch, fixed/dynamic 각각 마지막 epoch 54, 2,430 update다.
- Citation은 각각 1 update/epoch이며 validation 선택 후 50 epoch에서 끝난 조건들이다.
  전체 20조건의 실제 epoch 출력은 3,832행이다. 이를 20×200이나 3,832 optimizer update라고 세지 않는다.
- Early stopping과 batch별 update 계약은 당시 실행의 일부다. 현재 500 epoch 고정 normalization 계약과 같지 않다.

### 새 scaling 계획은 완료 결과가 아니다

[Rich scaling 원문](../../gpt_handoff/RICH_SCALING_EXPERIMENTS.md)의 계약은
Conductance 106학습 + Cycle 12학습 + Tree 8모델 학습 = **126 fresh model training / 122 child run**이다.
이 전체 matrix를 완료한 자료는 수령하지 않았다.

| 트랙 | Reference / large 계획 | 현재 확인한 범위 |
| --- | --- | --- |
| Conductance V1–V5 | 8층/256/8head 대 12층/384/8head(버전별 지원 요소 적용) | Corrected V5 20조건·dynamic test 10개를 다른 V1–V4 완료로 확대하지 않음 |
| Cycle v1/v2 | ZINC 128/64/10 대 192/96/12, Peptides 256/64/6 대 320/96/8(hidden/PE/layer) | 초기 Cycle v1 5-seed test와 새 sparse DFS v2의 미수령 상태를 구분 |
| Tree | 8층/128 대 12층/256 | 기존 fixed/multi 공식 test 집계와 새 scaling 전체 matrix를 구분 |

V5 multi-C의 120학습 계획, edge-selection의 130학습 계획도 실제 받은 일부 출력과 다르다.
조건별 full/reference/sample/profile·source hash·epoch 완료가 없으면 전체 완료로 표시하지 않는다.

## 11. 코드·원문 위치와 다음 트랙

| 연구 | 코드 | 설계·상태·원문 |
| --- | --- | --- |
| Conductance V1–V5 | [research/conductance_gat](../../research/conductance_gat/) | [원본 전체 상태와 epoch/test 부록](../../gpt_handoff/EXPERIMENT_STATUS.md), [V5 수식과 한계](../../gpt_handoff/CONDUCTANCE_V5.md) |
| Cycle PE | [research/cycle_pe](../../research/cycle_pe/) | [v1](../../docs/CYCLE_PE.md), [v2](../../gpt_handoff/CYCLE_PE_V2.md) |
| Tree | [research/tree_augmentation](../../research/tree_augmentation/) | [정확한 chart 전환과 데이터 계약](../../docs/TREE_AUGMENTATION.md) |
| 보류 combined | [research/combined_later](../../research/combined_later/) | [보류 근거](../../docs/COMBINED_LATER.md), [보존 산출물](../../results/combined_later/) |
| Incidence 내부 ablation | [experiments/incidence_ablation](../../experiments/incidence_ablation/) | [PPI 8조건 수령 표](../../gpt_handoff/EXPERIMENT_STATUS.md) |
| Aggregation·샘플링 | [experiments/aggregation_comparison](../../experiments/aggregation_comparison/), [sampled_inductive](../../experiments/sampled_inductive/) | [집계 비교](../../docs/AGGREGATION_COMPARISON.md), [실제 arxiv calibration](../../docs/ARXIV_MEMORY_REVIEW_20260927.md) |
| Information flow v2 | [experiments/information_flow_v2](../../experiments/information_flow_v2/) | [수식](../../gpt_handoff_v2_20260928/01_MODEL_MATH.md), [당시 검증](../../experiments/information_flow_v2/VERIFICATION.md) |
| C-learning only/bracket | [only](../../experiments/c_learning_only/), [bracket](../../experiments/c_learning_bracket/) | [Only 수식](../../experiments/c_learning_only/MODEL_MATH.md), [Bracket 수식](../../experiments/c_learning_bracket/MODEL_MATH.md), [수정 검증](../../experiments/c_learning_bracket/REVIEW_FIXES.md) |
| Output-init | [experiments/output_init_ablation](../../experiments/output_init_ablation/) | [수치 안정성](../../experiments/output_init_ablation/NUMERICAL_STABILITY.md), [복구](../../experiments/output_init_ablation/RECOVERY.md) |

다음 독립 트랙의 전체 이력은 [Wedge 실험](03_WEDGE_EXPERIMENTS.md)을 따른다.
현재 로컬 에너지·copy 결합·정규화 수식은 각 전용 문서와 이 묶음의 현재 연구 설명을 따른다.
Normalization의 20조건·840회 학습·420,000회 갱신에 대한 서버 FULL 결과를 수령했다.
로컬 본학습은 미실행이며, 새 결과의 근거와 해석은 [정규화 FULL 결과](08_NORMALIZATION_FULL_RESULTS.md)를 따른다.
서로 다른 모델의 같은 버전 번호·C=1·정규화라는 단어를 근거로 결과를 한 ablation처럼 합치지 않는다.

## 12. PPI incidence 내부 ablation: 수령한 여덟 조건 전체

원문은 [EXPERIMENT_STATUS의 2026-09-26 추가 수령 절](../../gpt_handoff/EXPERIMENT_STATUS.md)에 있다.
`incidence-ppi-reference-mig10gb-gpu4-seed0-v2`의 학습·감사 8/8 passed가 보고됐다.
아래는 선택된 validation checkpoint의 micro-F1이며, 공식 test와 다중 seed 결과가 아니다.

| 조건 | Validation micro-F1 (%) | Best epoch | Baseline 대비 pp |
| --- | ---: | ---: | ---: |
| baseline | 98.8320 | 169 | 0 |
| linear_lift | 98.8655 | 160 | +0.0335 |
| pre_lift | 98.8825 | 165 | +0.0505 |
| post_lift | 98.9503 | 193 | +0.1183 |
| bilinear | 98.8370 | 127 | +0.0050 |
| bilinear_linear_lift | 98.8458 | 200 | +0.0138 |
| bilinear_pre_lift | 98.8611 | 196 | +0.0291 |
| bilinear_post_lift | 98.8917 | 184 | +0.0597 |

Best epoch는 각 arm의 총 학습 epoch 수가 아니다. Bilinear 결합은 대응 lift 단독보다 낮았다.
기존 backbone의 외부 residual/FFN을 포함한 내부 대조이며, 이후 이를 제거한 aggregation 비교와 다른 모델이다.
함께 나온 이전 v1 run은 GPU 할당이 바뀐 뒤 calibration을 섞지 않도록 사전 중단되어 학습·감사 0/8이었다.
이 실패와 위 v2의 완료를 하나의 실행으로 합치지 않는다.
