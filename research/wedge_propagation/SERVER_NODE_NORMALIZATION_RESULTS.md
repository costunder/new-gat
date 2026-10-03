# Experiment 4.2 — 서버 전체 결과

사용자가 제공한 서버 `ece-a6gpu6` 완료 출력
(`5bdcdc63-17da-4c09-90bb-30b3139d7704`)의 completion과 summary를 기록한다.
구현 기준은 `2422568`이다. 아래 수치는 첨부된 출력에서 확인했으며,
원본 서버 CSV·checkpoint를 로컬에서 다시 평가한 결과는 아니다.

## 실행 범위

- Full, 실제 Cora·CiteSeer·PubMed, 다섯 조건, 각 500 epoch.
- Tuning 135 + final 75 = 210 runs, 105,000 optimizer updates.
- Primary metric 225행, frozen intervention metric 12,420행, layer diagnostic 8,430행.
- `completed=true`, 모든 dataset/condition/seed/split 포함, 실제 데이터 사용.
- Frozen model update 0, code/graph 보존 완료가 보고됐다.
- 6,946.80초, 약 1시간 55분 47초.

## 분류 결과

Test accuracy 평균(%). 모든 조건의 train accuracy 평균은 100%다.

| 데이터 | 고정 C=1 | Global/raw | Global/RMS | Node/raw | Node/RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 73.56 | 70.76 | 70.52 | 72.72 | 72.08 |
| CiteSeer | 64.94 | 63.74 | 63.84 | 64.92 | 64.90 |
| PubMed | 75.46 | 75.06 | 74.74 | 75.56 | 75.72 |

### Node와 global 비교

| 데이터 | Raw Δaccuracy pp | RMS Δaccuracy pp | Raw ΔCE | RMS ΔCE |
| --- | ---: | ---: | ---: | ---: |
| Cora | +1.96 | +1.56 | −0.07738 | −0.07254 |
| CiteSeer | +1.18 | +1.06 | −0.01985 | −0.01903 |
| PubMed | +0.50 | +0.98 | −0.005381 | +0.0007282 |

평균 정확도는 여섯 비교 모두 개선됐다. Accuracy paired 95% t 구간이 0을
제외한 비교는 Cora raw/RMS다. CE는 PubMed RMS를 제외한 다섯 비교에서
음의 paired 구간이 관측됐다. PubMed RMS의 CE 차이 구간은 0을 포함한다.

이 결과는 **C 의존 노드별 정규화로 재학습한 모델이 기존 global 모델의 성능을 개선했다**는
근거다. 전역 κ 하나가 원인이라는 증명은 아니다. 정규화가 노드별 좌표와 메시지 방향을
바꾸고 projection·gate·α·β도 재학습됐다.

예를 들어 Cora raw 첫 층의 β는 global 0.2697에서 node 0.04251로 낮아졌고,
실제 βM/Z도 0.02395에서 0.01387로 낮아졌다. 모든 분기가 일괄 증폭됐다고 해석하지 않는다.

### Node와 고정 C=1 비교

| 데이터 | Raw Δaccuracy pp | RMS Δaccuracy pp | Raw ΔCE | RMS ΔCE |
| --- | ---: | ---: | ---: | ---: |
| Cora | −0.84 | −1.48 | +0.01573 | +0.03074 |
| CiteSeer | −0.02 | −0.04 | +0.0005341 | +0.002582 |
| PubMed | +0.10 | +0.26 | +0.003659 | +0.008511 |

**학습한 C가 고정 C=1보다 분류에 유리하다는 근거는 아직 없다.**
PubMed의 양의 accuracy 차이는 paired 구간에 0이 포함된다.
Node/RMS의 CE는 세 데이터 모두 고정 C=1보다 높으며 해당 구간도 양수다.

## C와 실제 분기

선택된 node/raw checkpoint의 평가 Z에서 첫 층 C는 거의 균일하다.
C의 path 평균은 생성식에 의해 1이다.

| 데이터 | Raw 첫 층 C std | Raw 두 번째 층 C std |
| --- | ---: | ---: |
| Cora | 7.19e−7 | 0.6141 |
| CiteSeer | 6.926e−7 | 0.4788 |
| PubMed | 1.055e−5 | 0.6555 |

RMS의 C는 양쪽 층에서 비균일하다. 이 관측은 raw 첫 층의 현재 작용이 거의 C=1이라는
뜻이다. 학습 내내 C가 상수였거나 모든 층의 C가 미학습이라는 뜻은 아니다.

두 층을 함께 개입해 메시지 norm을 맞춘 C=1/shuffle의 평균 accuracy 변화는
node 조건에서 −0.042~+0.18%p다. 두 층 norm-matched C=1 교체의 평균 CE는
여섯 node 조건에서 모두 감소했다. 비균일 가중치의 생성과 그 배치의 분류 이득을 구분한다.

이차 분기를 없애면 CiteSeer node의 평균 accuracy가 1.42/1.16%p 낮아지지만
CE는 감소한다. 이차 분기의 top-1 기여와 학습 C의 배치 효과, 확률 예측의 품질은
같은 주장이 아니다. 개입의 구간은 원본 CSV에서 확인해야 한다.

## 다음 확인

**두 번째 층의 비균일 C가 같은 메시지 크기의 C=1보다 실제로 도움이 되는가?**

이미 생성된 `intervention_changes.csv`의 `layer_1`에서 node/raw·RMS,
norm-matched C=1/shuffle, test accuracy·CE와 paired 구간을 먼저 읽는다.
`layer_0`과 `both`를 함께 보면 각 층 효과와 두 층 동시 개입의 차이를 확인할 수 있다.
이번 출력은 both 요약만 포함하므로 각 층의 필요성을 확정하지 않는다.
이 확인을 위해 새 500-epoch 학습을 실행할 필요는 없다.

구간은 같은 public split의 초기화 seed 변동이며 다중 비교 보정이나 독립 split/graph의
불확실성을 포함하지 않는다. 같은 test를 본 뒤 설계한 후속 연구라는 범위를 유지한다.
