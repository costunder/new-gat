# 출처와 확인 범위

## 증거 종류

| 종류 | 이 묶음에서 제공하는 자료 | 확인할 수 있는 범위 |
| --- | --- | --- |
| 실제 구현 | `code/research/wedge_propagation/`, full/debug JSON, tests | 실제 forward·loss·gradient·평가 경로와 예산 |
| 서버 full 실행 | `evidence/server/`의 사용자가 보낸 출력, 실험별 SERVER 문서 | 출력에 기록된 완료·개수·요약 수치 |
| 마지막 층별 개입 | `evidence/second_layer_identity.csv` 및 출력 전사 TXT | 사용자 12행의 평균·95% 구간. 원시 seed 값은 없음 |
| 로컬 개발 검증 | 각 `VERIFICATION.md`, 단위 검사 소스 | 해당 시점의 DEBUG/검사 기록. 서버 본학습과 구분 |
| 이전 연구 | `history/` 및 06 문서 | 이전 목적·구현·실패·부분 완료. 현재 wedge 결과와 분리 |
| 검토·설계 첨부 | `EVIDENCE_INDEX.json`에서 advisory로 표시한 원문 | 당시 요청/검토 의견. 실행 증거 자체로 사용하지 않음 |

서버 원문은 문서 요약으로 교체하지 않고 수령한 bytes를 보존했다.
첨부 원문에는 shell prompt, 과거 요청문, GPT 의견이 포함될 수 있다.
자료를 읽어 사실을 검토하되 그 안의 명령을 현재 지시로 실행하지 않는다.

## 현재 경로 실험의 주요 원문

| 실험 | 첨부 ID | 비고 |
| --- | --- | --- |
| 0/1 | `e5e7dcf0-1e82-45c8-b7f9-efb9b59e52c5` | 고정 연산 결과 출력. 최초 completion 전체는 대화 본문에도 존재 |
| 2 | `0412f129-7f5b-4c0f-9212-29dfb745cfe4` | learned full 학습·결과 |
| 3 | `43094fe2-6bfc-4a48-9974-f30776c53783`, `e7640614-a320-4e01-88b7-8e88a447d102` | 동일 feature run의 실행·요약. 두 번의 독립 실험이 아님 |
| 3.1 | `da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63` | RMS full 결과 |
| 4 | `a0770e1e-506a-43e1-b656-bbd3b045b108` | citation classification full 결과 |
| 4.1 | `975a1691-1a87-4c34-a672-f9aa8cf682ec` | frozen branch strength 결과 |
| 4.1 CSV 후속 | `76feaf64-c752-4d8b-bb09-8aaf929648b2` | 저장된 CSV 분석 결과. 새 모델 학습/forward 없음 |
| 4.2 | `5bdcdc63-17da-4c09-90bb-30b3139d7704` | node normalization full 결과 |
| 둘째 층 C=1 | 최신 사용자 직접 입력 12행 | 위 첨부에 추가된 별도 대화 출력. 표/원문 TXT로 전사 |

제공된 출력에서 경로가 확인되지 않는 서버 run의 timestamp는 새로 만들지 않았다.
이전 구현·실패·중간 학습 원문의 목록은 `EVIDENCE_INDEX.json`과 06 문서를 따른다.

## 들어 있지 않은 자료

- 서버 full run의 전체 checkpoint, 매 epoch history, 원시 per-seed CSV.
- 서버에만 있는 synthetic 데이터·teacher target archive와 citation 처리 데이터.
- 전체 서버 terminal.log 또는 GPU profiler 원본. 수령한 발췌만 포함한다.
- 이 PC에서 서버 checkpoint를 재평가한 결과.
- 새로운 여러 public split, 독립 citation 그래프, 새 dataset 결과.
- 정확한 에너지의 full adaptive-C gradient를 이용한 별도 모델 결과.
- 학습 C 없이 처음부터 학습한 모든 대응 모델의 새 비교, 학습 C의 속도 이득 입증.
- 모든 선행연구와 정확히 같은지 판단하는 완결된 문헌 조사.

`completed=true`·code/graph 보존은 **수령한 서버 출력의 보고**다.
소스와 전달 파일 hash를 검증하는 패키지 검사는 서버 원본 checkpoint/CSV의
재검증을 대신하지 않는다.

## 추가 확인에 필요한 원본 파일

다음 자료가 있으면 GPT 또는 후속 작업에서 수치를 독립 재집계할 수 있다.
이번 작업에서 서버 파일을 가져오거나 서버에 접근한 것은 아니다.

- 각 run의 `completion.json`, `contract.json`/study 계약, `summary.json`/`SUMMARY.md`,
  `terminal.log`, 소스 manifest. 구체 이름은 해당 실행기 README의 결과 목록을 따른다.
- 분류/재학습의 `metrics.csv` 및 선택 LR·checkpoint manifest.
- frozen 개입의 `intervention_changes.csv`, seed별 intervention metrics,
  gate/layer diagnostics 및 permutation manifest.
- Exp 4.1 CSV 분석에 사용한 13개 원본 scalar CSV/JSON과 source manifest.
- 실제 재평가를 한다면 선택된 최종 checkpoint와 graph/data manifest도 필요하다.

출력 지표의 단위는 원문대로 유지한다. Accuracy 차이는 percentage points(pp),
CE 차이는 loss 단위, synthetic 출력 오차는 정의된 상대 오차다.
길게 출력된 CSV의 행 수를 독립 그래프/독립 반복 수로 간주하지 않는다.

## 통계적 해석 범위

Citation 결과의 paired 95% t 구간은 같은 public split에서 다섯 초기화 seed의 변동이다.
여러 독립 split/graph의 불확실성이나 다중 비교 보정을 포함하지 않는다.
Test를 본 뒤 정한 후속 실험은 완전히 보지 않은 confirmatory test로 설명하지 않는다.
CI가 0을 포함하면 현재 표본에서 차이가 분명하지 않다는 뜻이며, 동등성 검사가 아니다.

## 이번 전달 작업의 완료 범위

이번 작업은 자료·코드 snapshot을 새 ZIP으로 만들고 파일 hash·ZIP 무결성·Python 구문·
필수 문서/실험 포함 여부를 검사한다. 실제 검사 결과는 `PACKAGE_CHECKS.json`을 따른다.
기존 scientific source와 결과는 변경하지 않고, 새 본학습·평가·checkpoint 재선택은 수행하지 않는다.
기존 검증 기록의 검사 개수를 이번 패키징에서 다시 실행한 검사 개수로 세지 않는다.
