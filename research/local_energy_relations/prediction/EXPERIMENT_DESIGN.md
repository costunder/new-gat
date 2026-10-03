# 다음 실험: 로컬 이차항·쌍선형항의 실제 분류 기여

## 연구 질문

**같은 두 홉 기본 전파·두 층 모델에서 E/J를 실제 노드 업데이트에 더하면 예측에 도움이 되는가?**
앞선 고정 연산 감사와 전체 수신 역복원 결과는 보존한다. 이번에는 분류 CE로 학습하며,
E/J 계산→lift→logits→CE→backward→optimizer→선택 checkpoint→metric을 연결한다.

## 전체 데이터와 공통 모델

Cora·CiteSeer·PubMed의 기존 public split 전체 그래프를 사용한다.
특징은 row-sum 정규화, 영 행 유지, 원래 노드 ID·공식 mask 유지다.
전체 노드·물리 엣지·모든 1홉 induced locals·연결된 모든 중심 쌍을 사용한다. Sampling ratio는 1이다.
Validation/test 특징·연결도 forward에 포함되는 transductive 설정이며, loss는 train mask에만 적용한다.

| 데이터 | 노드 | 특징 | 클래스 | train / validation / test |
| --- | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 500 | 3 | 60 / 500 / 1,000 |

원본·processed data SHA256, 노드·mask·label 순서와 N/E/local occurrence/중심 쌍 수를 확인한다.
기존 public citation loader의 pinned raw 검증을 재사용하며 synthetic teacher나 기존 checkpoint를 옮기지 않는다.
DEBUG는 별도 명시된 fixture와 별도 예산이며, 실제 citation 결과라고 제출하지 않는다.

모든 조건은 2층 hidden 64, 각 투영 직전 dropout 0.5, 첫 층 후 ReLU,
둘째 층 logits, bias 없는 Xavier projection, 동일 α=sigmoid(a) 초기 0.5를 사용한다.
공통 기본 전파는 `(1−α)Z+αP_C²Z`다. 한 층 최대 두 홉의 support를 모든 조건에 제공한다.

## 여덟 조건

| 고정 C | 네 가지 조건 | 실제 차이 |
| --- | --- | --- |
| unit | unit__base / within / between / both | E lift·J lift의 유무 |
| local_degree | local_degree__base / within / between / both | E lift·J lift의 유무 |

조건 이름은 예를 들어 `unit__within`, `local_degree__both`다.
C는 학습하지 않는다. E는 `E/(2FΣc)`, J는
`J_node/(F sqrt(tr(L_v²)tr(L_u²)))`를 incoming 이웃 평균으로 사용한다.
분모는 topology에만 의존하며 raw 이차·쌍선형 성질과 부호를 보존한다.
활성 lift 벡터는 0 초기화한다. 같은 C 안에서 projection/dropout stream과 초기 forward가 같다.
세부 수식·공간 범위·영 분모 규칙은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.

이번 primary 비교는 **추가 특징과 추가 파라미터를 함께 넣은 총 기여**다.
별도 용량 대조 모델을 재학습하지 않으므로 유일한 기하 구조 효과를 분리했다고 주장하지 않는다.
학습한 벡터의 norm·분기 norm·고정 제거 개입을 보아 실제 사용 여부를 함께 확인한다.

## 기존 학습 예산 유지

본학습은 서버에서 실행한다. Adam, lr 0.001/0.003/0.01, 500 epoch,
매 epoch 전체 그래프 CE 한 번과 optimizer update 한 번을 수행한다.
Early stopping·scheduler·gradient clipping·gradient accumulation은 없다.
Float32, TF32 off. Projection 행렬에만 weight decay 5e−4, α·lift 벡터에는 0이다.

| 단계 | seed | run 수 | update 수 |
| --- | --- | ---: | ---: |
| Validation LR 선택 | 101 / 202 / 303 | 3데이터×8조건×3LR×3seed = 216 | 108,000 |
| 선택 LR로 최종 새 학습 | 11 / 23 / 37 / 53 / 71 | 3데이터×8조건×5seed = 120 | 60,000 |
| 합계 | 단계 간 독립 | **336** | **168,000** |

각 run의 checkpoint는 validation CE 최소→accuracy 최대→가장 이른 epoch 순으로 선택한다.
조건·데이터별 LR은 tuning 3seed의 선택 checkpoint validation CE 평균 최소→작은 LR 순이다.
평가 CE에 L2를 더하지 않는다. 모든 final 선택을 고정한 뒤 test 평가를 연다.
Tuning에서는 test metric을 계산하지 않는다. Seed별 checkpoint·선택 목록·hash를 보존한다.

## 평가와 고정 개입

모든 최종 seed의 train/validation/test accuracy·CE와 parameter 수를 남긴다.
같은 C 안에서 `within−base`, `between−base`, `both−base`,
`both−within`, `both−between`을 같은 seed로 대응시킨다.
같은 variant의 `local_degree−unit`도 보존한다.
또한 같은 seed의 `both−within−between+base`를 각 C·split·metric에서 계산한다.
이는 따로 학습한 네 모델의 성능 차이에 대한 상호작용 비교이며,
raw 이차식의 교차항을 직접 측정했다는 뜻은 아니다.
Seed별 차이, 평균, 표본 std, 양측 95% t 구간을 출력한다.
Accuracy 차이는 양수, CE 차이는 음수가 개선이다.

같은 checkpoint에서 활성 E만 제거, J만 제거, 둘 다 제거한다.
각 제거는 layer_0 / layer_1 / both에 적용한다. 비활성 항을 제거하는 중복 처리는 만들지 않는다.
α·projection·lift·optimizer를 고정하고 모델 state SHA256 불변을 확인한다.
다음 층의 특징과 E/J는 개입 후 현재 Z로 다시 계산한다. Clean 두 층 값을 재사용하지 않는다.
모든 split·개입·층의 결과를 보존하며 좋은 개입만 고르지 않는다.

분기 진단은 projected/base norm, E/J feature norm, lift vector norm,
실제 추가 branch norm과 base 대비 비율, α, raw E/J 평균, J 음수 비율이다.
분모가 0인 비율은 undefined와 그 flag를 기록한다.
E/J는 특징 크기의 제곱이므로, 분기가 작게 남으면 신호 규모와 사용 여부를 함께 읽는다.

## 자원·검증·결과의 범위

A6000 실제 할당 VRAM/GPU·CPU affinity/quota·RAM·storage를 기록한다.
Full graph당 physical/effective batch는 1이며, 독립 seed 모델을 앞 축에 pack해 동시 계산한다.
이는 데이터 병렬 replica나 gradient accumulation이 아니다. 각 replica의 train-node 평균 CE를 합산한다.
Parameter·dropout·Adam 상태는 seed별 독립이며 packed/독립 한 update 일치를 검증한다.

전체 topology/P/static 분모를 캐시한다. 모든 local/shared-node correspondence는 유지하고,
필요하면 정확한 chunking을 사용한다. CPU workers·packed seed 수·chunk size 후보를 실제로 계측한다.
OOM 시 데이터·레이어·hidden·epoch를 줄이거나 CPU로 조용히 바꾸지 않는다.
여러 할당 GPU는 독립 dataset/condition/LR job을 나누고, 한 터미널에 phase/epoch/CE/validation/ETA를 출력한다.

정적 검사, dense 독립 수식·gradient, support·P² 범위, topology 정규화,
0 초기화 forward parity, 첫 lift CE gradient/업데이트 뒤 feature gradient,
packed Adam·고정 개입 state 불변·coverage/누락 실패를 DEBUG에서 검사한다.
DEBUG 성공은 서버 FULL 336 run의 완료나 정확도 검증을 뜻하지 않는다.

이 public test는 앞선 실험에서도 관찰했다. 이번은 같은 fixed split의 후속 탐색이며,
95% 구간은 초기화 seed 변동을 나타낸다. 독립 새 split/graph 일반화·다중 비교 보정을 포함하지 않는다.
성능 개선을 복구 불가능한 사이클 메시지의 복원, learned C 성공, 전체 spectral 함수의 우위라고 부르지 않는다.
