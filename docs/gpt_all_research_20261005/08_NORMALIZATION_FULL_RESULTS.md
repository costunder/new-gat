# 최신 추가: 정규화 FULL 완료 — 2026-10-05

이 장은 이전2026-10-04 전달본의 정규화 FULL 미수령 상태를 갱신한다. 이전 자료는 당시 기록으로 보존했다.

**실제 citation3개·20조건·840학습·420,000갱신**의 FULL 완료 원문을 수령했다.
2개 macro·hidden64·각500epoch, metric900/frozen4,320/branch3,480행, frozen 갱신0이다.
전체6,820.536초, 약1시간53분41초다. 표시된 집계60/312/72/120행도 모두 확인했다.

근거는 [새 서버 원문](../evidence/local_context_normalization_server_full_20261005.txt),
[전체 분석](../LOCAL_CONTEXT_NORMALIZATION_SERVER_FINDINGS_20261005.md),
[파싱한 전체 집계](../evidence/local_context_normalization_summary_20261005.json)다.
원시 seed CSV·checkpoint를 다시 평가한 결과는 아니다.

## 큰 정확도 변화는 내부 graph→local

C=unit, cross-off의 동일 비교다.

| 데이터 | graph acc % | local acc % | 차이 pp [95% 구간] |
| --- | ---: | ---: | --- |
| Cora | 58.66 | 80.12 | +21.46 [20.3861,22.5339] |
| CiteSeer | 57.62 | 70.36 | +12.74 [12.0343,13.4457] |
| PubMed | 73.16 | 77.80 | +4.64 [3.37802,5.90199] |

Local_degree/off의 변경은 Cora+0.78pp, CiteSeer−0.18pp, PubMed+0.24pp이며 Cora만 구간이0을 제외한다.
Local 내부 정책에서 C 두 조건의 accuracy15개 비교는 모두0을 포함한다.
이전의 큰 C 차이를 가중치 자체의 성과로만 설명할 수 없다.

## 교차 경로의 판정

- 같은 C·intra의 fixed−off24개와 learned−off24개 **accuracy 구간은 모두0 포함**이다.
- CE는 fixed−off12개와 learned−off6개에서 개선 구간이다.
- Learned−fixed CE는10개에서 악화 구간이며 개선 구간은 없다.
- Edge−graph accuracy의 좁은 양수 구간 하나는 edge−off의 정확도 우위가 아니다.
- 상호작용도 cross의 상대 기여 변화와 cross−off 자체의 이득을 구분한다.

Local+edge fixed 첫 층의 실제 Δ/off는 약1.65–2.59%다.
각 모델의 현재 Z에서 첫 층 fixed의 작용은 local+graph 조건보다 약49–86배 컸다.
교차 작용이 커졌다는 관측과 추가 정확도 개선은 별개다. 일부 CE 개선도 실제로 존재한다.

현재 C·개별 연결 가중치는 고정이며 learned는 θ 하나의 공유 강도다.
Epoch gradient/update와 frozen ΔCE/flip 원문은 없으므로 그 상세 효과까지 주장하지 않는다.
같은 public split의5seed·다중 비교 미보정·후속 탐색이라는 범위를 유지한다.

**현재 결론은 내부 정규화의 큰 기여와 일부 교차 CE 기여다.**
정보 복원·새 그래프 일반화·현재 구조의 GCN/GATv2 우위·신규성을 대신 입증하지 않는다.
