# 로컬 문맥 결합 분류 FULL 결과 — 2026-10-04

**분류 본학습은 완료됐다. 교차 강도는 변했지만, 교차 연결을 추가한 정확도 개선은 확인되지 않았다.**
큰 정확도 차이는 `unit`과 `local_degree`의 내부 전파 조건 사이에서 나타났다.
교차를 끈 모델에서도 이 차이가 있으므로 교차 결합의 성과로 해석하지 않는다.

## 제출 자료와 확인 범위

사용자가 제출한 `completion.json`과 `LOCAL_CONTEXT_CLASSIFICATION_SUMMARY.md`의 터미널 출력을
[원문](evidence/local_context_classification_server_full_20261004.txt)에 byte 그대로 보존했다.
SHA-256은 `889e5f356f896f1d85051df302ccc60ef22d13dfa8533a1e51241503844bf6b9`다.
서버 결과 폴더의 실제 경로, raw CSV, checkpoint, source/data hash와 hardware 파일은 첨부에 없다.
여기서는 제출된 완료 기록·수치와 구현 계약을 대조했다. 서버 원시 파일을 독립 재검증했다고 표현하지 않는다.

## 1. 전체 학습과 평가 완료 기록

| 항목 | 제출 값 | 계약과의 대응 |
| --- | ---: | --- |
| Tuning / final / 총 학습 | 162 / 90 / 252 | 세 데이터 × 여섯 조건 × LR·seed 계약 |
| 새 optimizer update | 126,000 | 252개 독립 모델 × 500 epoch |
| Primary metric | 270행 | 최종 모델 90개 × train/validation/test |
| Frozen 개입 metric | 1,080행 | 3개 데이터 × 활성 4조건 × 5seed × 6개입 × 3split |
| 층 진단 | 900행 | 원래 모델 180행 + 개입 720행 |
| 전체 소요시간 | 2,067.5805초 | 약34분28초, 준비·calibration·학습·평가·보고서 포함 |

`completed=true`, `profile=full`, `actual_data=true`, code/graph 보존 여부가 보고됐다.
동결 평가의 optimizer update는0이다. 전체 시간을 나눠 개별 모델의 epoch 시간으로 주장하지 않는다.

## 2. 같은 C에서의 교차 결합 비교

Test accuracy는 최종 seed 5개의 평균이며 단위는%다.

| 데이터 | local_degree off | local_degree fixed | local_degree learned | fixed−off (pp) | learned−off (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 79.38 | 79.46 | 79.38 | +0.08 | 약0 |
| CiteSeer | 70.64 | 70.60 | 70.60 | −0.04 | −0.04 |
| PubMed | 77.66 | 77.66 | 77.66 | 0 | 약0 |

동일 C의 fixed−off, learned−off, learned−fixed accuracy 차이의95% paired 구간은
unit과 local_degree 모두0을 포함한다. 정확도 우위나 동등성을 입증한 결과가 아니다.

Fixed의 일부 CE 비교는 작지만0을 벗어나는 구간을 보고했다.

| 데이터 / C | fixed−off ΔCE | 95% 구간 |
| --- | ---: | --- |
| CiteSeer / unit | −0.000135732 | [−0.000265591, −0.00000587282] |
| Cora / local_degree | −0.000224400 | [−0.000384059, −0.0000647399] |
| PubMed / local_degree | −0.0000701785 | [−0.000110865, −0.0000294916] |

Learned가 fixed보다 일관되게 좋은 결과는 없다.
Cora / local_degree의 learned−fixed ΔCE는+0.000115347이며 구간도 양수다.
이 비교에서는 learned의 CE가 조금 나빴다. 모든 조건의 train accuracy 평균은100%였다.

## 3. 큰 차이는 내부 C와 그에 따른 전파 조건

교차 off에서 local_degree−unit의 test accuracy 차이는
Cora **+20.72pp**, CiteSeer **+13.02pp**, PubMed **+4.50pp**다.
세 paired 구간 모두 양수다.

이는 고정 내부 가중치뿐 아니라 그 가중치로 정한 η와 실제 전파까지 함께 바뀐 비교다.
가중치의 어느 부분이 효과를 냈는지, step 차이 때문인지 각각 분리하지 않았다.
Learned C, attention 생성 규칙, 교차 결합의 기여로 해석하지 않는다.
Cross-off도 두 내부 A pass와 copy/merge를 사용하는 현재 모델이며 표준 GCN과 동일하다고 두지 않는다.

## 4. 교차 강도 학습과 실제 작용

Learned 조건은 seed마다 θ 하나를 두 macro 층에서 공유한다.
초기 θ=0, 초기ρ=0.5이며 선택된 모델에서 다음 평균을 보고했다.

| 데이터 | unit ρ | local_degree ρ |
| --- | ---: | ---: |
| Cora | 0.857757 | 0.371410 |
| CiteSeer | 0.902937 | 0.0659821 |
| PubMed | 0.469102 | 0.266711 |

초기값에서의 변화는 강도 파라미터가 갱신됐음을 뒷받침한다.
첨부에는 epoch별 CE gradient와 θ update CSV가 없어 그 크기와 학습 추세는 직접 검토하지 않았다.
이번 모델의 C와 개별 copy 연결 가중치는 고정이다.

선택된 모델의 현재 층 Z에서 실제 상대 변화는
`||TρZ−T0Z|| / ||T0Z||`로 측정했다. 아래는 각 데이터/층의 seed 평균 값 범위다.

- Fixed / unit: **0.00110–0.00444%**.
- Fixed / local_degree: **0.01289–0.01584%**.
- Learned 두 C: **0.000518–0.00531%**.

Copy 사이 에너지는 줄고 실제 노드 출력도 바뀌지만, 그 출력 변화는 off 표현에 비해 작다.
Off의 상대 변화 약1e−8과 fixed/gain1 CE 차이 약1e−8은 수치 잔차로 읽는다.
약한 출력 변화가 성능 효과의 유일한 원인이라고 확정하지 않는다.

## 5. 동결 개입에서 확인한 것

현재 checkpoint에서 gain을0으로 바꾸면 CE가 미세하게 악화됐다.
Learned checkpoint에서 gain을1로 바꾸면 CE가 미세하게 개선됐다.
Accuracy는 대부분 그대로이며 일부 차이도95% 구간이0을 포함한다.

예를 들어 local_degree learned의 두 층 gain1 개입 ΔCE는
Cora−0.0000486016, CiteSeer−0.0000303984, PubMed−0.0000319481이다.
이는 현재 checkpoint가 교차 연산을 사용하는 효과다.
처음부터 fixed로 학습한 모델의 성과나 더 좋은 학습 설정을 확정하는 결과는 아니다.
θ는 train CE로 학습했으므로, test에서 gain1이 조금 나은 것과 학습된ρ가 작은 것은 모순이 아니다.

## 6. 다음 설계에서 구분할 질문

현재까지는 메커니즘의 작동과 학습 파라미터의 변화가 확인됐고 추가 정확도 효과는 작았다.
다음 질문은 **안전한 고정 step이 교차 작용을 너무 약하게 만든 것인지,
현재 copy 관계의 분류 기여 자체가 작은 것인지**다.

현재 차이식은 `Tρ−T0=−γη²ρMAKAR`다.
η·γ의 분포와 실제 mixed action을 확인하고 내부 정규화·교차 연산을 구분한 비교를 설계해야 한다.
η/γ의 크기만으로 실패 원인을 확정하거나 강도를 무작정 키우지 않는다.
후속 정규화는 대칭 A, PSD K, copy 대응과 merge의 소거 관계를 확인하고
같은 내부 전파 깊이·전체 데이터·학습 예산을 유지하는 별도 계약으로 정의한다.
현재 모델·원래 결과에 새 설정을 섞지 않는다.

이번 결과의 구간은 같은 public split의 초기화 seed 5개에 관한 탐색적 비교다.
새 그래프 일반화, 정보 복원, learned edge C의 성능은 검증하지 않았다.
