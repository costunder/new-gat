# E/J 위치별 재학습 설계

## 이번 질문

**같은 기본 모델에서 E/J를 첫 층만, 출력층만, 두 층에 넣고 학습했을 때 무엇이 달라지는가?**
앞선 두 층 모델의 출력층 제거 반응이 컸다는 관찰은 새 위치를 비교할 근거다.
기존 checkpoint에서 항을 제거하는 것과 그 위치에 처음부터 항 없이 학습하는 것은 다르다.
이번에는 모든 비교 조건을 같은 계약으로 새로 학습한다.

## 20조건

고정 C는 `unit` / `local_degree`다. 각 C에서 아래 열 조건을 쓴다.

| 조건 | 첫 층 | 출력층 |
| --- | --- | --- |
| base | 기본 전파 | 기본 전파 |
| within__hidden | 기본 + E | 기본 |
| within__output | 기본 | 기본 + E |
| within__all | 기본 + E | 기본 + E |
| between__hidden | 기본 + J | 기본 |
| between__output | 기본 | 기본 + J |
| between__all | 기본 + J | 기본 + J |
| both__hidden | 기본 + E+J | 기본 |
| both__output | 기본 | 기본 + E+J |
| both__all | 기본 + E+J | 기본 + E+J |

실제 이름은 `unit__base`, `unit__both__hidden`, `local_degree__within__output` 등이다.
base를 세 번 복제하지 않는다. 두 고정 C별로 한 번씩 학습한다.
앞선 base/두 층 모델도 새 실험 안에서 다시 학습하므로 공통 source·data·hardware 조건으로 비교된다.
과거 결과는 별도 기록으로 보존한다.

모든 조건의 기본 전파는 `(1−α)Z+αP_C²Z`다. 첫 층은 64채널, 출력층은 클래스 수 채널이다.
E/J의 정의, topology 분모, 부호, 입력 support를 바꾸지 않는다.
활성 층에만 lift를 만든다. 비활성 층의 미사용 parameter나 계산은 없다.
0으로 초기화한 lift 덕분에 같은 C·seed의 모든 조건은 같은 초기 projection/dropout/forward를 갖는다.

## 전체 데이터와 모델 유지

| 데이터 | 전체 노드 | 입력 특징 | 클래스 | train / validation / test |
| --- | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 500 | 3 | 60 / 500 / 1,000 |

전체 public split, row-sum 정규화, 원래 노드 ID·mask·label 순서를 유지한다.
전체 물리 엣지, 모든 1홉 induced local, 인접한 모든 중심 쌍과 공통 노드 대응을 쓴다.
Sampling ratio는 1이다. Validation/test의 특징·연결을 포함하는 transductive forward이며,
CE는 train mask에만 적용한다. 앞선 원본 local-energy audit의 hash 검증을 유지한다.

모델은 두 층, hidden 64, projection bias 없음, Xavier 초기화,
projection 직전 dropout 0.5, 첫 층만 ReLU다. 둘째 층은 logits다.
α=sigmoid(a)는 각 층마다 학습하고 초기 α=0.5다.

Adam, float32, TF32 off, lr 후보 0.001/0.003/0.01을 유지한다.
Projection에만 weight decay 5e−4를 적용하고 α·lift에는 0이다.
모든 run은 500 epoch, epoch마다 전체 그래프 optimizer update 한 번이다.
Early stopping, scheduler, clipping, accumulation은 없다.
평가 CE에는 L2를 더하지 않는다.

## 전체 예산과 선택

| 단계 | 독립 seed | runs | updates |
| --- | --- | ---: | ---: |
| 조건별 validation LR 선택 | 101 / 202 / 303 | 3데이터×20조건×3LR×3seed = 540 | 270,000 |
| 선택 LR로 최종 새 학습 | 11 / 23 / 37 / 53 / 71 | 3데이터×20조건×5seed = 300 | 150,000 |
| 합계 | 단계 간 독립 | **840** | **420,000** |

각 run의 checkpoint는 validation CE 최소 → accuracy 최대 → 가장 이른 epoch 순이다.
조건·데이터별 LR은 tuning seed의 선택 checkpoint validation CE 평균 최소 → 작은 LR 순이다.
모든 final checkpoint 선택을 고정한 뒤 test 평가를 연다. Tuning에서 test metric은 계산하지 않는다.
위치를 test 성능으로 선택하거나 가장 좋은 test 값만 최종 결과로 제출하지 않는다.

DEBUG는 별도 fixture 3개, hidden 8, 3 epoch, 2 LR, tuning/final 각 2seed의
360 runs / 1,080 updates다. 실제 데이터 성능이 아니며 FULL 설정에 덮어쓰지 않는다.

## 대응 비교와 고정 제거

같은 seed에서 모든 split의 accuracy·CE를 비교한다.

- 각 C·위치의 `within−base`, `between−base`, `both−base`.
- 각 C·variant의 `hidden−output`, `hidden−all`, `output−all`.
- 각 C·위치의 `both−within`, `both−between`, `both−within−between+base`.
- 같은 variant·위치의 `local_degree−unit`, 그리고 base의 C 차이.

평균·표본 std·양측 95% t 구간과 모든 seed의 값을 보존한다.
Accuracy 차이가 양수, CE 차이가 음수면 개선이다.
네 모델의 성능 상호작용은 원래 이차형식의 교차항을 직접 측정한 값이 아니다.

같은 checkpoint에서 E / J / E+J의 활성 항만 제거한다.
hidden은 `layer_0`, output은 `layer_1`, all은 `layer_0` / `layer_1` / `both`를 평가한다.
비활성 층을 제거하는 중복 평가를 만들지 않는다.
모든 개입은 train/validation/test에서 수행하고, model state SHA256 불변과 업데이트 0을 확인한다.
첫 층 제거 시 다음 층 특징·E/J를 현재 상태에서 다시 계산한다.
따라서 층별 제거 효과를 독립적이고 가산적인 기여로 읽지 않는다.

원래 모델과 모든 개입의 두 층 diagnostics를 남긴다. Projected/base/feature/branch/lift norm,
α, branch/base 비율, raw E/J 평균, J 음수 비율을 기록한다.
base norm이 0이면 비율을 undefined로 표시한다. 비활성 층의 분기·벡터·특징 진단은 0이다.

## 자원·검증·해석

전체 topology·P·분모를 캐시하며 정확한 local/relation chunking을 사용한다.
CPU worker, 독립 seed packing, chunk 후보를 실제로 측정한다.
Full graph당 physical/effective batch는 1이고, 독립 seed model을 앞 축에 pack한다.
각 seed의 train-node 평균 CE를 합산한다. Seed parameter·dropout·Adam 상태는 독립이다.
여러 할당 GPU는 독립 jobs를 분배한다. 동일 터미널에서 epoch 진행을 표시한다.
모델·노드·엣지·데이터·epoch를 축소하는 fallback은 없다.

이 위치 비교는 **항의 위치와 활성 lift 용량의 총 효과**다.
추가 parameter는 hidden의 E/J 한 항당 64개, output은 클래스 수 K개, all은 64+K개다.
별도 용량 대조가 없으므로 위치 효과만 유일하게 분리했다고 주장하지 않는다.
앞선 public test 관찰 뒤 설계한 후속 탐색이며, 구간은 같은 split의 초기화 변동이다.
새 split/graph 일반화, 다중 비교 보정, learned C 성공, 사이클 메시지 복원을 입증하지 않는다.
