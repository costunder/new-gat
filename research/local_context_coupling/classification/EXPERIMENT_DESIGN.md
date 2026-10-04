# 분류 실험 계약과 판단 기준

## 1. 연구 질문

**동일한 내부 전파 깊이에서 로컬 문맥 사이의 cross 결합이 분류를 개선하는가?**
그 다음으로 **한 모델의 모든 로컬 그래프와 두 층에서 공유하는 cross 강도를 학습하면 고정 강도 1보다 유익한가?**를 본다.
기존 scalar E/J 추가 실험과 고정 연산 audit를 이 모델의 분류 성과로 합치지 않는다.

## 2. 데이터와 source

| 데이터 | 전체 N×F | 클래스 | Train/validation/test |
| --- | --- | ---: | --- |
| Cora | 2708×1433 | 7 | 140/500/1000 |
| CiteSeer | 3327×3703 | 6 | 120/500/1000 |
| PubMed | 19717×500 | 3 | 60/500/1000 |

완료된 원본 audit `local-energy-20261004-002815`의 그래프 201개와 입력 hash를 확인하고 보존한다.
분류는 이 source의 전체 citation 그래프 3개에서 원래 특징·물리 엣지·모든 1홉 로컬 그래프를 그대로 재사용한다.
Label과 public mask는 원본 hash를 고정한 raw 데이터에서 읽고 원래 graph ID에 대응시킨다.
합성 그래프 198개는 분류 학습 대상이 아니다. 모든 노드·물리 엣지·유도 로컬 그래프·canonical cross link를 유지한다.

## 3. 조건과 학습 예산

각 C 조건인 unit과 local_degree에서 off/fixed/learned 모델을 독립적으로 처음부터 학습한다.
2개 macro 층, hidden dimension 64, dropout .5, Xavier projection 초기화, θ 초기값 0을 [config_full.json](config_full.json)에 고정했다.
Off와 fixed의 projection 파라미터 수는 같다. Learned는 seed 모델마다 θ 1개만 추가한다.

| 단계 | 독립 학습 횟수 | 조건 |
| --- | ---: | --- |
| Tuning | 162 | 데이터셋 3개 × 조건 6개 × LR 3개 × tuning seed 3개 |
| Final | 90 | 데이터셋 3개 × 조건 6개 × final seed 5개 |
| 합계 | 252 | 학습마다 500 epoch, epoch마다 1회 갱신: 총 126,000회 갱신 |

LR 후보는 .001/.003/.01, tuning seed는 101/202/303, final seed는 11/23/37/53/71이다.
Adam의 LR는 학습 동안 고정한다. Projection weight decay는 5e−4, θ weight decay는 0이며 float32와 TF32 off를 사용한다.
Early stopping, scheduler, gradient clipping은 사용하지 않는다. Checkpoint는 validation CE가 가장 낮은 것을 선택한다. 동률이면 validation accuracy가 높은 것, 그다음 이른 epoch를 선택한다.
Tuning seed들의 선택된 validation CE 평균으로 LR를 선택한다. 모든 final checkpoint 선택을 고정한 후 test를 평가한다.

한 독립 모델의 physical batch는 전체 그래프 1개다. Seed 축은 별도 모델들을 병렬 계산하는 축이며 한 모델의 effective batch에 곱하지 않는다.
Gradient accumulation과 data-parallel replica 수가 각각 1이므로 모델당 effective graph batch도 1이다.

## 4. 구현과 자원 계약

본학습은 할당받은 GPU를 명시한 서버 CUDA에서 실행한다. GPU를 여러 개 할당받았다면 데이터셋·조건·LR별 작업을 분배한다.
동시에 계산하는 독립 seed 수는 tuning에서 1/2/3개, final에서 1/2/4/5개를 실측한다. Exact edge chunk 후보는 256/4096/16384/65536이다.
정적 그래프·normalization·copy 연결을 cache하고 모든 연결을 처리하는 sparse gather/scatter를 사용한다.
고정 대칭 A/K의 custom backward와 macro activation checkpointing으로 메모리를 줄인다.
CPU 준비 worker 수는 1/2/4/8 후보를 측정한다. CPU thread·RAM·GPU/VRAM·자원 할당 제한과 실제 처리시간을 기록한다.
Calibration은 별도 초기 상태에서 수행하여 본학습의 시작 상태와 갱신 예산을 보존한다.
OOM이 발생한 후보는 실패로 기록한다. 모델·그래프·노드·특징·epoch 수를 조용히 줄이지 않는다.

## 5. 평가와 출력 범위

주요 지표는 test accuracy, 보조 지표는 CE(L2 항 제외)다. Train/validation/test의 final seed별 원시 값·평균·표본 std를 출력한다.
같은 seed에서 조건 간 차이를 먼저 계산하여 paired 95% t 구간을 보고한다.

- Within-C fixed−off: 교차 연산을 사용해 새로 학습한 기여.
- Within-C learned−off: 학습 gain 모델의 기여.
- Within-C learned−fixed: gain 학습의 추가 기여.
- 같은 variant의 local_degree−unit: 고정 내부 C 조건 비교, 탐색적으로 표시.

Frozen 개입은 fixed와 learned 모델에서 gain0/gain1 × layer_0/layer_1/both의 6개 조합을 평가한다.
Fixed 모델의 gain1 개입은 출력이 같아야 하는 no-op 대조로 표시한다. Off 모델은 원래 평가만 한다.
Frozen 개입은 같은 checkpoint를 사용하며, 조건을 바꾸고 처음부터 학습한 ablation과 구분한다.
선택된 모델의 θ·gain·에너지·현재 Z에 대한 matched 출력 차이와 매 epoch의 gain gradient·파라미터 갱신을 함께 읽는다.

FULL 원시 결과는 분류 지표 270행, frozen 개입 1,080행, branch 진단 900행이다.
파생 요약은 metric 54행, paired 비교 162행, frozen 차이 432행, branch 요약 180행이다.
이 행 수를 독립 그래프 수나 학습 반복 수로 세지 않는다.

## 6. 결과 판정

기본 모델 대비 accuracy/CE, seed별 paired 구간, frozen 결과와 실제 cross 작용을 함께 보고한다.
ρ가 1에 가깝다는 사실만으로 fixed가 최선이라고 판단하거나, ρ가 0에 가깝다는 사실만으로 학습 실패라고 판단하지 않는다.
Gain이 학습되는지, 실제 출력이 변하는지, 분류에 도움이 되는지를 각각 설명한다. 좋은 test 개입 하나를 골라 최종 결과로 바꾸지 않는다.
Accuracy와 CE의 방향이 다르면 각각 기록하고 개선을 한 지표로 뭉개지 않는다.

같은 public test를 이미 여러 선행 실험에서 확인했다. 이번 연구는 그 뒤의 탐색적 비교다.
Paired 구간은 같은 split에서 초기화 seed 5개에 따른 변동을 나타낸다. 독립 split·그래프에 대한 일반화나 다중 비교 보정은 포함하지 않는다.
0을 포함하는 구간은 동등성 증명이 아니다. 구조 차이·활성 파라미터·energy 정의만으로 분류 개선·정보 복원·연구 신규성을 입증하지 않는다.

## 7. DEBUG의 분리

별도 DEBUG-Cora/CiteSeer/PubMed fixture를 사용한다. 노드 수는 각각 24/30/36, 특징 수는 12, 클래스 수는 3, hidden dimension은 8, 학습은 3 epoch다.
여섯 조건, LR 2개, tuning seed 2개, final seed 2개로 총 108회 학습과 324회 갱신을 수행한다.
원본 DEBUG source의 그래프 21개 계약을 사용하며 actual_data=false로 출력한다.
이 pipeline 검사는 FULL 설정·데이터·예산에 덮어쓰지 않고 citation 분류 성능으로 제출하지 않는다.

현재 실제 실행 검증과 서버 FULL 실행 여부는 [VERIFICATION.md](VERIFICATION.md)에 따로 기록한다.
