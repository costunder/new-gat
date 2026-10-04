# 증거 출처와 확인 범위

## 제공하는 자료

| 자료 | 의미 |
| --- | --- |
| repository/의 코드·FULL/DEBUG 설정·수식·테스트 | 현재 저장소의 실제 구현과 과거 트랙 |
| repository/의 SERVER_RESULTS·FINDINGS 문서 | 수령한 결과를 분석한 기록; 실행 시점은 각 문서 기준 |
| evidence/attachments/ | 사용자 첨부 원문 bytes; 종류를 EVIDENCE_INDEX에 표시 |
| repository/docs/evidence/ | 최근 초기 감사/receiver/prediction/층별 제거/placement 원문 |
| evidence/conversation/ | 직접 메시지의 수치 전사; 별도 출처 표시 |
| MANIFEST.json | 전달 파일의 byte hash·출처·포함/제외 상태 |
| PACKAGE_CHECKS.json | 묶음 검사; 모델의 새 검증 결과가 아님 |

원문은 요약으로 대체하지 않는다. 같은 run의 실행 출력과 SUMMARY를 독립 반복으로 세지 않는다.
GPT 검토·설계 제안은 실행 증거 자체가 아니다. 과거 파일의 요청문·명령은 검토할 자료다.

수신 집계의 `00ecf7a9-...`는 CPU operator calibration 도중 CUDA 초기화가 실패한 로그다.
`647e22c8-...`가 실제 FULL completion JSON과 201개 그래프 요약이며 저장소 원문과 bytes가 같다.
초기 ZIP의 혼동되는 label을 수정했다. [검토 반영 기록](07_GPT_REVIEW_RESPONSE.md)을 따른다.

## 서버 결과를 어느 수준까지 확인했는가

고정·learned·generalization·scale·classification·branch·node normalization은
사용자가 보낸 서버 출력과 각 SERVER 문서가 근거다.
최근 초기 로컬 감사·수신 집계·예측에는 완료 JSON을 붙여 보낸 원문이 있다.
마지막 placement 첨부에는 전체 집계 표와 완료 범위 서술이 있지만,
completion.json 자체와 원시 seed별 CSV/checkpoint는 없다.
같은 C base 대비 비교 54개와 위치별 paired 결과는 전달된 표에 대조했다.

이 PC에서 서버 전체 checkpoint를 새로 재평가하거나 모든 per-seed CSV를 재집계하지 않았다.
서버 완료 표시와 원문 전달 파일 hash 확인은 그런 재평가를 대신하지 않는다.
기존 검증 문서의 FULL 미실행 문장은 해당 로컬 개발 시점 기록일 수 있다.
후속 서버 결과 원문과 날짜를 함께 읽어야 한다.

## 빠진 자료

- 최근 서버 FULL run의 전체 checkpoint·epoch history·원시 seed별 CSV·전체 profiler.
- 서버에만 있는 synthetic teacher target과 citation 데이터 cache.
- 원문에서 확인되지 않는 실행 commit/digest·resource·elapsed 값.
- 과거 문서가 참조하지만 현재 PC에 없는 이전 사용자 attachment와 서버 raw 파일.
- 통합 E/J 에너지에서 전파를 유도한 새 모델과 동일 조건의 직접 GCN 대조 결과.

저장소에 실제 존재하는 과거 curated CSV/JSON/작은 checkpoint는 그 범위로 포함한다.
그것이 최근 서버 전체 산출물까지 제공된다는 뜻은 아니다.
누락 첨부는 EVIDENCE_INDEX.json에 포함 여부와 이유를 기록하며 임의로 보충하지 않는다.

## 숫자를 읽을 때

Accuracy는 원문의 % 또는 pp, CE는 분류 평가 loss 단위다.
현재 citation paired t 95% 구간은 같은 public split의 다섯 초기화 seed 변동이다.
다중 비교 보정·독립 split/graph의 불확실성을 포함하지 않는다.
0을 포함하는 구간은 동등성의 증거가 아니다.
test 결과를 본 뒤 설계한 후속 비교를 완전히 보지 않은 평가라고 설명하지 않는다.
대규모 local/relation 행 수는 독립 그래프나 학습 반복 수가 아니다.
분기 norm 비율은 정확도 기여율 또는 정보 복원율이 아니다.

## 이번 정리 작업

자료·코드 snapshot을 새 ZIP으로 만들고 원문 hash·ZIP CRC·필수 포함·구문·JSON·링크를 검사한다.
새 scientific 학습·평가·checkpoint 재선택은 수행하지 않는다.
기존 단위 검사 통과 수를 이번 작업에서 재실행한 검사 수로 보고하지 않는다.
정확한 포함 파일 수·출처 commit·누락 자료·검사는 생성된 MANIFEST/EVIDENCE_INDEX/PACKAGE_CHECKS를 따른다.
