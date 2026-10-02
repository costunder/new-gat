# Experiment 3.1 서버 전체 학습·평가 결과

2026년 10월 3일 사용자 첨부의 `SCALE_NORMALIZATION_SUMMARY.md` 출력 내용을 기록했다.
출처는 첨부 ID `da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63`이다.
**서버 원본 CSV·checkpoint·contract.json을 직접 읽어 검증한 결과는 아니다.**
첨부에서 결과 폴더의 정확한 이름과 run timestamp는 확인할 수 없어 기록하지 않는다.

## 확인된 실행 범위

- FULL, raw source와 normalized 모두 500 epoch·5 seed로 표시됐다.
- 전체 531개 그래프를 사용했으며 `data_fraction=1.0`이다.
  Train/validation/ID/size OOD/family OOD/family+size OOD는 각각 240/60/120/90/12/9개다.
- 원본 학습 source는 `/home/aicompetition07/new-gat/results/wedge-learned-20261002-172132`,
  새 특징 source는 `/home/aicompetition07/new-gat/results/wedge-feature-20261003-042123`이다.
- Raw checkpoint와 first/polynomial/fixed 대조군은 고정했다.
  Normalized는 세 target × 두 gate 조건의 6개 job을 새로 학습했다.
- L/L²/path × 다섯 조건 × raw/normalized × original 및 다섯 fresh 배율을 평가했다.
  Fresh 배율 0.25/0.5/1/2/4는 같은 새 특징을 공유하는 대응 처리다.
- 메시지 248,508행, 스케일 207,090행, 개입 159,300행이 보고됐다.
  행 수는 독립 그래프 표본 수가 아니다.
- `raw_optimizer_updates=0`, `test_updates=0`이며,
  학습 후 평가 구간의 모델 불변성과 source·feature source·코드 보존이 모두 true로 보고됐다.
- Original metric 재현은 11,349 graph/seed 행에 대해 `verified=true`로 보고됐다.
  최대 absolute metric 차이는 0.0020530134860337057,
  최대 target-normalized RMSE 차이는 8.216763834559611e-5다.
  재현 결과를 오차 0이나 파일의 수치상 완전 일치로 설명하지 않는다.

첨부 요약에는 서버 GPU·VRAM·배치·전체 소요시간의 상세 계측이 포함되지 않았다.
이 항목은 원본 contract와 실행 로그를 확인해야 한다.

## 새 특징의 경로 목표 상대오차

배율 1의 learned/path 결과다. 특징 실현 평균 → 동일 비중 그래프 평균 →
학습 seed 평균·표본 std 순서로 집계했다. Fixed Q는 같은 frozen 대조군으로 두 variant에 재사용했다.

| Split | Raw learned | Normalized learned | Fixed Q |
| --- | ---: | ---: | ---: |
| ID | 0.0475536 ± 0.0139 | 0.0515799 ± 0.00357 | 0.174729 |
| size OOD | 0.0465101 ± 0.0128 | 0.0525378 ± 0.00365 | 0.166116 |
| family OOD | 0.0512217 ± 0.0143 | 0.0532518 ± 0.00297 | 0.203409 |
| family + size OOD | 0.0473821 ± 0.0130 | 0.0508475 ± 0.00274 | 0.179715 |

Normalized의 fresh ID 평균 오차는 raw보다 0.0040263, 약 8.47% 높고 seed 표준편차는 작다.
표의 모든 OOD split에서도 normalized 평균 오차가 더 높다.
양쪽 모두 fixed Q보다 작은 오차를 보였지만, 정규화가 회수 정확도를 개선했다는 결과는 아니다.
이 표는 합성 teacher 메시지의 회수 결과이며 실제 분류 정확도를 뜻하지 않는다.

## 배율 안정성

아래는 fresh ID의 learned/path다. 각 graph·seed에서 다섯 관측 배율의 최대 변화를 구한 뒤
graph macro와 seed 통계를 계산했다. 특정 배율 4만의 결과나 전체 관측 중 단일 최댓값이 아니다.

\[
\Delta_C(a)=\frac{\|C(aX)-C(X)\|_2}{\|C(X)\|_2+\epsilon},\qquad
\Delta_M(a)=\frac{\|M(aX)/a-M(X)\|_2}{\|M(X)\|_2+\epsilon}.
\]

| Variant | Student C 변화 | Student 메시지 비례성 오차 |
| --- | ---: | ---: |
| raw | 0.371238 ± 0.135 | 0.306902 ± 0.0969 |
| normalized | 1.95587e-7 ± 1.58e-8 | 2.06874e-7 ± 1.06e-8 |

같은 teacher의 ID 변화는 C 1.08028e-5, 메시지 1.78726e-6이다.
Teacher는 원래 epsilon=1e-8 규칙으로 매 배율에서 다시 계산됐다.
정규화 학생의 배율 반응은 수치 오차 수준으로 줄었다.
이는 설계한 C 불변성·메시지 비례성이 동작한 결과이며 목표 메시지 정확도 개선과는 별개다.

## C 회수와 고정 개입

Path target의 ID 보조 진단이다. Cmean=1은 전역 scale 자유도를 제한하지만 C의 유일한 회수를 보장하지 않는다.

| Scenario | Variant | C 상대오차 | Teacher C와 상관 |
| --- | --- | ---: | ---: |
| original | raw | 0.123585 ± 0.0286 | 0.947717 ± 0.0231 |
| original | normalized | 0.130976 ± 0.00870 | 0.944367 ± 0.00686 |
| fresh 1 | raw | 0.122813 ± 0.0288 | 0.948524 ± 0.0233 |
| fresh 1 | normalized | 0.129194 ± 0.00874 | 0.945999 ± 0.00682 |

아래 개입 값은 **개입 메시지 오차 − 원래 메시지 오차**의 ID 평균이다.
개입 중 모델과 beta는 고정했다. 양수는 개입 후 오차 증가다.

| Scenario | Variant | C=I (=mean) | 가중치 위치 섞기 | 다른 그래프 패턴 | 연결 대응 무작위화 |
| --- | --- | ---: | ---: | ---: | ---: |
| original | raw | 0.336780 | 0.392706 | 0.401951 | 0.660997 |
| original | normalized | 0.348761 | 0.402727 | 0.412882 | 0.665389 |
| fresh 1 | raw | 0.340129 | 0.391498 | 0.403708 | 0.670791 |
| fresh 1 | normalized | 0.352626 | 0.402349 | 0.415597 | 0.676112 |

Identity와 mean은 평균 1 제약에서 동일하므로 두 독립 효과로 세지 않는다.
연결 대응 무작위화는 연산 support도 바꾸므로 경로 연속성만의 인과 효과로 해석하지 않는다.
이 개입은 합성 teacher 회수에서 가중치 값과 배치의 기여를 보여 준다.
실제 분류에서도 그 기여가 유지되는지는 별도 실험으로 확인해야 한다.

## 다음 단계에 주는 결론

**입력 RMS 정규화는 배율 안정성을 개선했지만 평균 메시지 회수 오차는 개선하지 않았다.**
따라서 Experiment 4에서는 raw와 normalized를 함께 유지하고, teacher 없이 분류 CE로 새로 학습한다.
분류 성능·고정 가중치 개입·추가 파라미터 용량·계산 비용을 각각 비교한다.
Synthetic 성과로 실제 분류 우위나 범용 spectral 연산의 우위를 단정하지 않는다.

계산과 실행 계약은 [scale_normalization/README.md](scale_normalization/README.md),
구현 검증은 [scale_normalization/VERIFICATION.md](scale_normalization/VERIFICATION.md)에 있다.
