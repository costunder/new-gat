# Experiment 4 서버 완료 결과

사용자가 제공한 `completion.json`과 `CLASSIFICATION_SUMMARY.md`의 terminal 출력에 근거한다.
첨부 ID는 `a0770e1e-506a-43e1-b656-bbd3b045b108`이다.
서버 원본 checkpoint·CSV를 로컬로 옮겨 독립 검증한 결과는 아니다.
Experiment 4.1 실행기는 서버의 원본을 읽고 hash·학습 이력·전체 범위·원래 metric을 검증한다.

## 실행 범위

- Full public split: Cora, CiteSeer, PubMed의 모든 노드·물리 엣지·wedge.
- 8조건, 두 층, hidden 64, 각 run 500 epoch.
- Tuning 216 + final 120 = 336 run, 새 optimizer update 168,000개.
- 원래 metric 360행, 개입 2,970행, 배율 2,100행, 층 진단 2,220행.
- `completed=true`, `actual_data=true`, frozen update 0, code/graph 보존.
- 총 실행 3,970.394초, 약 66분 10초. 다른 작업의 실행시간 예측에 사용하지 않는다.

## Test 정확도

5개 최종 seed의 평균이며 단위는 %다. 전체 표준편차·CE·paired 구간은 원본 summary에 있다.

| 조건 | Cora | CiteSeer | PubMed |
| --- | ---: | ---: | ---: |
| MLP | 57.14 | 56.22 | 72.28 |
| First order | 70.78 | 63.78 | 75.10 |
| Polynomial 2 | 72.54 | 66.02 | 75.72 |
| Fixed wedge | 73.56 | 64.94 | 75.46 |
| Learned raw | 70.74 | 63.74 | 75.06 |
| Learned RMS | 70.52 | 63.86 | 74.68 |
| Fixed wedge + node MLP | 63.70 | 55.92 | 71.18 |
| Standard GCN | 81.88 | 71.12 | 79.06 |

Learned raw/RMS는 이번 설정에서 fixed wedge와 polynomial보다 평균 정확도가 낮았다.
모든 조건 중 standard GCN의 평균 정확도가 가장 높았다.
고정 public split의 seed 변동을 독립 그래프 일반화로 해석하지 않는다.

## C와 분기 강도

C mean=1은 생성식의 정규화다. 원래 층의 C population std를 seed 평균한 값은
raw 약 0.058–0.413, RMS 약 0.319–0.688이므로 C가 전부 1인 모델은 아니다.
κ의 seed 평균은 raw 약 3.99–5.89, RMS 약 5.61–7.09였다.

현재 층 true C의 κ를 유지한 C=1/shuffle 개입의 accuracy 변화는 대체로 ±0.04pp 안이었다.
C=1에서 κ를 재계산하면 여섯 dataset/condition 모두 +0.40–1.32pp였다.
그러나 κ 유지가 실제 메시지 norm 유지와 같지는 않다. 이 관측만으로 C 위치가 무용하거나
κ가 실패의 원인이라고 확정할 수 없다.

RMS 모델은 고정 Z 배율 검사에서 C 최대 변화가 0이고 branch 등변 오차가 약 1e-7–4e-7이었다.
이 설계상의 배율 안정성이 이번 분류 정확도 개선으로 이어지지는 않았다.

## 다음 단계

[Experiment 4.1](branch_strength/README.md)은 저장된 모델을 그대로 사용한다.
C 위치 변경의 효과를 같은 메시지 norm에서 확인하고,
true C 메시지의 방향을 유지한 강도 변경과 구분한다.
각 층의 αL̄Z, βM, κ 분포와 test CE/accuracy를 함께 보고하며 새 학습은 하지 않는다.
