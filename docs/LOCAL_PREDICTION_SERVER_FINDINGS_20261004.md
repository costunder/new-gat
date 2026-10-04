# 로컬 E/J 예측 실험 서버 FULL 결과 — 2026-10-04

## 결론

**이번 모델은 학습했고 E/J를 실제로 사용했다. 그러나 기본 모델 대비 test accuracy 개선은 확인되지 않았다.**
현재의 두 scalar 특징을 학습 벡터로 노드 상태에 더하는 후보 설계에 대한 결과다.
이 결과를 이차형식·쌍선형 관계 전반의 실패나 learned C 학습 결과로 확대하지 않는다.

## 근거와 완료 범위

사용자가 전달한 completion과 전체 요약의 원문을
[서버 증거](evidence/local_prediction_server_full_20261004.txt)에 그대로 보존했다.
SHA256은 `a2955bbe563a49b2ba96f565799f6224b7ee4c6404221f600472d46e36683760`이다.
서버의 원본 seed별 CSV·checkpoint·자원 JSON을 가져와 다시 평가한 결과와 구분한다.
표의 평균 차이와 대응 구간, 성능 상호작용은 표시된 평균·std와 반올림 범위 안에서 일치했다.

| 항목 | 첨부된 완료 기록 |
| --- | --- |
| 완료 / profile / 실제 데이터 | completed=true / full / actual_data=true |
| 데이터 / 조건 | citation 3개 / 4가지 분기 × 두 고정 C = 8 |
| 학습 구성 | 2층 hidden 64, 각 run 500 epoch |
| Tuning / final / 합계 | 216 / 120 / 336 runs |
| 독립 optimizer update | 168,000 |
| 기본 metric / 제거 metric / 층 진단 | 360 / 1,350 / 1,140행 |
| 고정 평가 update | 0 |
| 시간 | 2,263.15초 = 37분 43초 |
| 기존 코드·그래프 보존 | code_and_graphs_preserved=true |

C는 unit과 local_degree 두 고정 규칙이다. C 생성 파라미터를 학습하지 않았다.
실제 결과 폴더명·실행 Git SHA·source digest·GPU/VRAM·raw seed 값은 이번 첨부에 없다.
630개 resource 관측의 seconds/epoch 범위 0.0057994–2.6735는 서로 다른 조건·packing·chunk를
포함하므로 이를 대표 epoch 시간이라고 쓰지 않는다.

## 기본 모델 대비 정확도

각 수치는 같은 C의 base 대비 test accuracy 차이 **퍼센트포인트(pp)**다.

| 데이터 | C | E 추가 | J 추가 | E+J 추가 |
| --- | --- | ---: | ---: | ---: |
| Cora | unit | −0.30 | −0.60 | −0.18 |
| Cora | local_degree | +0.10 | −0.86 | +0.22 |
| CiteSeer | unit | −1.24 | −2.84 | −2.18 |
| CiteSeer | local_degree | −0.26 | −2.12 | −1.28 |
| PubMed | unit | −0.32 | −0.26 | −0.16 |
| PubMed | local_degree | +0.10 | +0.18 | +0.10 |

18개 비교에서 95% 구간 전체가 양수인 경우는 없다. 양수인 평균도 모두 0을 포함한다.
세 데이터 모두 unit base가 여덟 조건 중 가장 높은 평균 test accuracy를 보였다.
Unit base의 정확도는 Cora 76.22%, CiteSeer 67.24%, PubMed 79.60%다.
구간은 고정 public split의 초기화 5seed 변동이며 다중 비교를 보정하지 않았다.

특히 CiteSeer의 J 단독은 unit −2.84pp [−3.9175, −1.7625],
local_degree −2.12pp [−3.8075, −0.43253]이다. CE도 두 조건 모두 악화됐다.
Unit의 E+J는 −2.18pp [−3.6736, −0.68638]다.
PubMed unit의 E 단독 역시 −0.32pp [−0.54212, −0.097883]다.

PubMed J 단독의 test CE는 unit −0.0024997 [−0.0043978, −0.00060169],
local_degree −0.0025772 [−0.0046444, −0.00050992]로 개선됐다.
해당 accuracy 구간은 0을 포함한다. CE 개선과 분류 정확도 개선을 구분한다.

고정 C를 local_degree로 바꾼 base는 unit base보다 Cora −1.88pp,
CiteSeer −2.34pp, PubMed −0.66pp이며 각각 표시된 95% 구간이 음수다.
이 C 비교에는 기본 전파 P_C의 변화도 포함된다.

## 실제 분기 학습과 사용

선택된 모든 checkpoint의 train accuracy는 100%이고, 모든 활성 lift vector norm은 양수다.
Lift는 0에서 초기화했으므로 실제로 파라미터가 바뀌었다.
활성 분기의 평균 `branch norm / base norm` 범위는 다음과 같다.

| 위치 | E / base | J / base |
| --- | ---: | ---: |
| 첫 층 | 0.92–3.21% | 0.33–1.64% |
| 출력 층 | 5.22–12.03% | 3.20–15.99% |

이는 표현 크기의 비율이다. 정보 복원률이나 정확도 기여율이 아니다.
고정 분기 제거로 CE/accuracy도 달라졌다. 따라서 항이 항상 0이거나 예측 경로에
연결되지 않았다는 설명과 맞지 않는다. 학습·사용 여부와 기본 모델 대비 유용성은 별개다.

## 고정 제거 개입과 상호작용

Cora·CiteSeer는 요약에 나온 모든 활성 분기 제거에서 평균 test CE가 나빠졌다.
하지만 별도로 학습한 base보다 성능이 좋았다는 뜻은 아니다.

PubMed의 두 C에서 표시된 10가지 두 층 제거 개입은 모두 평균 test accuracy가 올라가고
CE가 내려갔다. 예를 들어 unit both의 E/J 전체 제거는 +0.20pp / CE −0.0060285,
local_degree both의 전체 제거는 +0.28pp / CE −0.0067084다.
이 요약에는 제거 개입의 구간이 없으므로 유의성을 확정하지 않는다.
파라미터를 고정하고 분기만 제거한 모델은 별도로 학습한 base와 동일한 모델이 아니다.

CiteSeer unit의 `both−within−between+base` 상호작용은 +1.90pp지만,
both는 base보다 여전히 −2.18pp다. 단일 분기의 악화를 함께 사용하며 일부 줄였다는 결과다.
이를 base 개선이나 raw 쌍선형식의 교차항 측정으로 설명하지 않는다.

## 후속 확인 완료: 첫 층과 출력층의 제거

사용자가 기존 `intervention_changes.csv`에서 전달한
[층별 제거 12행](evidence/local_prediction_layer_removal_server_20261004.txt)을 보존했다.
각 값은 학습된 both 모델의 해당 층에서 **E와 J를 함께 제거**했을 때
test CE 변화다. 파라미터는 고정했다. 양수는 제거 시 악화, 음수는 제거 시 개선이다.

| 데이터 | C | 첫 층 제거 ΔCE | 출력층 제거 ΔCE |
| --- | --- | ---: | ---: |
| Cora | unit | +0.002446 | +0.012675 |
| Cora | local_degree | +0.001549 | +0.009127 |
| CiteSeer | unit | +0.000176 | +0.012594 |
| CiteSeer | local_degree | +0.000190 | +0.011077 |
| PubMed | unit | −0.000458 | −0.005809 |
| PubMed | local_degree | −0.000482 | −0.006491 |

모든 경우 출력층 제거의 평균 CE 변화가 첫 층보다 컸다.
Cora·CiteSeer의 출력층 제거 구간은 모두 양수다. 현재 학습된 모델이 출력층 E/J를
활용하는 결과지만, 별도로 학습한 base보다 좋다는 근거로 쓰지 않는다.
PubMed는 두 C·두 층의 제거 구간이 모두 음수다. 특히 출력층 제거의 평균 개선이 크다.
현재 checkpoint에서 직접 logits에 더한 항은 PubMed test CE에 불리하게 작용했다.

첫 층은 Cora unit의 구간이 양수이고, Cora local_degree와 CiteSeer 두 C의 구간은
0을 포함한다. 첫 층 효과가 모두 없었다고 설명하지 않는다.
95% 구간은 기존 고정 split·초기화 seed 비교이며 다중 비교 보정은 없다.

첫 층 제거는 뒤의 특징과 E/J를 다시 계산한다. 두 층의 제거값을 합산해 독립 기여도나
책임 비율을 만들지 않는다. E/J를 함께 제거했으므로 E와 J 각각의 원인을 분리한 표도 아니다.
이 추가 표는 test CE만 포함하므로 train/validation 반응이나 정확도 변화는 확인되지 않았다.

## 다음 비교의 연구 질문

다음 재학습 비교는 **E/J를 넣는 위치**를 구분하는 것이 타당하다.
현재 기본 모델과 두 층 모두 주입을 보존하고, 첫 층에만 주입·출력층에만 주입을 별도 조건으로
학습해 비교한다. E·J·E+J의 비교와 두 C·전체 데이터·깊이·hidden·500 epoch·seed 계약을 유지한다.
위치 선택과 설정 비교는 validation으로 수행하고 test는 선택 후 평가한다.
이번 test 관측을 바탕으로 정한 후속 탐색임을 기록한다.

첫 층만 남기면 좋아진다는 보장은 없다. Cora·CiteSeer는 현재 출력층 항의 제거가 오히려
불리하므로, 출력층을 무조건 폐기할 근거도 없다. 고정 개입은 위치 대조의 근거이며
별도 재학습의 성능 결과를 대신하지 않는다. 이후 이 위치 대조를 구현했으며,
2026-10-04 제출된 서버 요약은 [위치별 재학습 결과](LOCAL_PLACEMENT_SERVER_FINDINGS_20261004.md)에 기록했다.

이 자료로 독립 그래프 일반화·사이클 정보 복원·learned C 성공을 주장하지 않는다.
이번 비교는 추가 E/J 특징과 작은 추가 파라미터의 총 기여이며 별도 용량 대조는 없다.
