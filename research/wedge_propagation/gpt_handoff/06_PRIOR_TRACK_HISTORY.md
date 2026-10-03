# 이전 CGAT 연구와 학습 점검 기록

기준일: 2026-10-03. 이 문서는 현재 wedge 실험으로 방향을 바꾸기 전의 기록이다.
**구현, 로컬 검사, 서버의 일부 조건 완료, 전체 비교 완료를 구분한다.**
현재 연구의 판정은 [연구 상태](01_RESEARCH_STATUS.md)와 [실험·결과](03_EXPERIMENTS_AND_RESULTS.md)를 따른다.

## 1. 이름과 범위

여기서 CGAT 연구 v1/v2는 연구 질문의 버전이다. 저장소의 `Conductance v2`,
`Cycle PE v2`, aggregation 구현의 v3/v4와 같은 번호를 뜻하지 않는다.
이전 소스가 Conductance 모듈을 import하는 것은 기술적 의존성이다.
그 의존성을 현재 wedge의 선행 실험이나 완료 결과로 합산하지 않는다.

이전 관심사는 **여러 샘플 문맥에서 C 생성 규칙을 학습하고, 집계와 집합 전환에서
구별되지 않는 정보를 진단·보완하는 것**이었다. 이어서 C 자체가 배우는지와
깊은 층에서 신호가 약해지는지를 먼저 확인하는 별도 실험으로 범위를 좁혔다.
현재 wedge는 길이 2 경로의 이차 차분 `A=RB`와 `Aᵀ C A`를 연구한다.
이전의 보완 분기를 그대로 발전시킨 동일 모델이라고 설명하면 안 된다.

## 2. 방향 변경 전 단계별 상태

| 단계 | 실제 모델·질문 | 확인한 범위 | 남은 한계 |
| --- | --- | --- | --- |
| CGAT v1 / aggregation | 같은 샘플의 노드·엣지 집합에서 층별 전파. 기본 집계와 국소 층간 에너지 등의 대조 | 코드·수학 문서와 보존 기준이 있음 | 이 자료에 해당 전체 benchmark의 최종 비교 결과가 없음. 서로 다른 홉 집합을 명시한 전달도 아님 |
| CGAT v2 / information flow | 단계별로 수신 집합을 좁히고 `base/within/between/both` 후보 비교 | 25개 합성 단위/CUDA 검사, 실제 ogbn-arxiv의 짧은 자원 측정 | 네 조건의 본학습·정보 보완 효과·독립 그래프 일반화를 입증한 결과는 아님 |
| C-learning-only v2.0 | 이전 보완 분기를 빼고 기존 C 생성기의 CE 미분·업데이트·C=1 대조부터 검사 | 12개 DEBUG 검사 | 당시 검증 문서에서 실제 데이터 200epoch 본학습과 유용한 C 효과는 미확인 |
| C-learning-bracket | 양 끝 노드의 대칭 Q/K 점수로 양의 대각 C 생성 | 실제 데이터 calibration, 신호 관측, 후속 DEBUG 33개 검사 | standalone 200epoch 비교는 당시 미완료. 초기 C≈1만으로 학습 실패라고 판정할 수 없음 |
| 출력 초기화 대조 / raw-exp | 기존 출력 초기화와 Kaiming 초기화 × learned/fixed C | 서버 calibration과 baseline learned epoch 1 완료 | epoch 2에서 C의 finite/positive 검사 실패. 네 조건 완료 아님 |
| 출력 초기화 대조 / log_row 및 복구 | 네 조건의 정규화된 exp 가중치를 안정적으로 계산. 완료 조건 검증 후 나머지 실행 | 서버 복구 기록에서 세 조건의 완료와 CUDA 재검증을 보고함 | 마지막 Kaiming fixed는 제공된 출력에서 epoch 4까지만 완료. 최종 네 조건 비교 자료는 없음 |

이 표의 미확인은 **전달받은 자료에서 확인하지 못했다**는 뜻이다.
현재 서버에 결과가 없다고 단정하는 것이 아니다.

## 3. CGAT v1에서 v2 후보로 바꾼 내용

v1 보존 기준은 `f071bcddb92b9253628c3087bf7931be37205e2b`다.
`V1_PRESERVATION.json`에는 당시 tracked 파일 462개의 SHA-256이 있다.
ignored/untracked 결과는 이 manifest에 포함되지 않으므로 전체 과거 결과의 목록은 아니다.

v2 후보는 하나의 샘플 안에서 감독 노드 쪽으로 수신 집합을 좁혔다.
층마다 그래프를 새로 무작위 추출한 구현은 아니다. 송신·수신 노드 ID,
경계 엣지, 이전 단계와 대응하는 엣지를 기록한다.
실제 메시지는 `q=C_eff B V`, 노드 집계는 `Bᵀq`다.

- `within`: 실제 q의 제곱 크기를 별도 특징으로 집계한다.
- `between`: 수신 집합 경계의 가중 차이 제곱과, 두 단계에 남는 동일 엣지의 메시지 내적을 쓴다.
- `both`: 두 분기를 함께 쓴다. 학습 readout을 거쳐 예측과 CE에 연결했다.

이 통계는 **메시지 전체를 복원해 다음 층에 보관하는 구현이 아니다.**
샘플링 때문에 처음부터 관측하지 못한 메시지를 채워 넣는 모델도 아니다.
복원 진단 코드와 예측에 쓰는 요약 특징의 역할을 구분해야 한다.
알려진 양의 C와 모든 노드 집계를 관측할 때 `q=CBV`가 복원될 수 있다는 조건도 검사했다.
Euclidean cycle 성분이 있다는 것만으로 복구 불가능한 손실이 있다고 주장하지 않는다.

실제 ogbn-arxiv 측정은 전체 그래프와 원래 8층/256채널/8헤드를 유지한 calibration이다.
물리 seed batch 2048/4096, worker 2/4 조합에서 세 step을 측정했으며 최종 학습이 아니다.
최신 v2 소스보다 앞선 calibration을 최신 네 조건의 최종 성능 검증으로 재사용하지 않는다.

## 4. C 생성기만 먼저 점검한 실험

### 기존 C 생성기를 유지한 C-learning-only

이 경로는 모든 층에서 같은 샘플 노드·엣지를 쓰며 다음 batch에서 문맥이 달라진다.
기존 C 비용·내부 최적화·문맥 기반 β와 출력 투영을 유지했다.
learned C와 C=1 고정 조건 모두 나머지 backbone을 학습하는 대조다.
within/boundary/cross 보완이나 cycle 복원 실험으로 읽으면 안 된다.

12개 DEBUG 검사는 작은 합성 입력과 2epoch를 썼다. 모델 크기는 8/256/8을 유지했다.
C의 gradient, optimizer 포함 여부, 같은 입력에서 업데이트 전후 변화와 frozen 개입을 확인했다.
이것은 실제 데이터 분류 이득이나 GATv2보다 빠른 attention을 증명하지 않는다.

### 대칭 Q/K 점수의 C-learning-bracket

엣지 e=(u,v), head h의 score와 C는 다음과 같다. r은 head 좌표 수다.

` s_eh = (Q_u·K_v + Q_v·K_u)/(2r),    C_eh = exp(s_eh) > 0 `

기존 value, 문맥 기반 β, incidence 전파와 sampled degree 보정을 유지하고 C 생성기만 바꿨다.
β에 그래프 문맥이 남아 있으므로 전체 모델이 순수한 로컬 attention이라고 주장하지 않는다.
기존 사용자 링크의 [bracket 논문](https://arxiv.org/html/2305.15616v3)과 저자 코드에서
연결별 점수를 양의 대각 C로 만드는 부분을 참조했다. 논문의 전체 dynamics 재현은 아니다.

실제 데이터 calibration에서 깊은 층의 초기 C가 거의 1이고 신호 RMS가 줄어드는 것이 관측됐다.
후속 검토는 평가 대상 ID와 CE 계산의 연결을 고쳤으며 33개 DEBUG 검사를 보고했다.
그 뒤 최신 calibration을 다시 실행한 결과와 standalone 200epoch 최종 비교는 당시 문서에 없다.
초기 RMS 감소는 위험 신호이며, 잔차 연결 부재나 출력 초기화 하나를 실패 원인으로 확정한 증거는 아니다.

## 5. 출력 초기화 실험: 실패 기록과 이후 완료 범위를 분리

네 조건은 baseline/Kaiming 출력 초기화 × learned/fixed C다.
Kaiming 조건은 새 난수를 뽑지 않고 기존 출력 weight를 `sqrt(6)`배해 초기 방향을 대응시켰다.
각 조건은 ogbn-arxiv 8층/256채널/8헤드, 200epoch, model seed 0,
AdamW lr 0.0005, dropout 0.2였다. 공식 test는 이 실험에 사용하지 않는다.

### 최초 raw-exp 실행

원문 `d159052b-a84f-457f-a992-5ef3c7a5e16b`에는
`output-init-server-20260929-163340`의 baseline learned epoch 1/200 완료가 있다.
한 epoch는 240.2초였고 validation accuracy는 7.6278%였다.
epoch 2에서 `exp produced nonfinite/zero C` assertion과 종료 코드 -6이 기록됐다.
이는 그 실행의 실패 증거다. score NaN과 exp 표현 범위 문제 중 정확한 원인은 이 출력만으로 확정하지 않는다.

더 앞선 `ef2fa6c7-06cf-425c-8f09-e661f0765c1a` 출력은 실행 중 상태와 MIG 자원을 보여 준다.
물리 A100 80GB 이름을 실제 할당 VRAM으로 쓰면 안 된다. 당시 보이는 MIG slice는 약 9.7GB였다.

### log_row와 복구 실행

후속 log_row는 정규화된 exp 비율을 logsumexp/행 최대값으로 계산한다.
유한 score의 수학적 비율은 유지하지만 raw-exp와 실행 소스·관측 형식이 달라 별도 run이다.
모든 수치 오류를 해결했다거나 C의 분류 유용성이 입증됐다고 설명하지 않는다.

`RECOVERY.md`에는 `output-init-log-row-server-20261001-172250`의 Kaiming learned
200/200 기록이 있다. 마지막 epoch CE는 0.62667, validation accuracy는 72.5897%다.
이 수치는 best checkpoint나 공식 test 점수가 아니다.
학습 저장 후 소스 계약 검사에 걸린 문제와 미완료 fixed 조건을 구분했다.

이후 서버 원문 `28537dca-db0f-49a6-9c49-afc35f80d2e8`은 다음을 보고한다.

- baseline learned/fixed, Kaiming learned 세 조건의 완료 상태와 CUDA 재검증.
- 변경 내역은 독립 wedge 파일 추가이며 blocked 목록은 비어 있음.
- 완료 모델은 재학습하지 않고, Kaiming fixed만 원래 200epoch로 새로 시작.
- Kaiming fixed epoch 4/200 완료: CE 1.43837, validation accuracy 64.3210%. 출력은 epoch 5 시작 뒤 끝남.

**제공된 자료에서 네 조건 전체 완료와 `comparison.json`은 확인하지 못했다.**
세 조건이 완료됐다는 서버 보고를 전부 미실행으로 바꾸거나, 마지막 조건의 진행 출력을
200epoch 완료로 확대해 쓰면 둘 다 잘못이다. 로컬에서 서버 checkpoint를 다시 읽은 것도 아니다.

## 6. 원문과 코드 위치

ZIP의 `history/`에는 아래 이전 소스와 문서를, `evidence/server/`에는 관련 원문을 보관한다.
경로의 README·검증 문서는 작성 당시 상태이므로 위의 나중 서버 증거와 함께 읽는다.

| 검토 대상 | 저장소 출처 | 증거 성격 |
| --- | --- | --- |
| v1 aggregation | `experiments/aggregation_comparison/`, `docs/AGGREGATION_COMPARISON.md` | 구현·과거 설계. 현재 wedge 결과 아님 |
| v2 정보 흐름 | `experiments/information_flow_v2/`, `gpt_handoff_v2_20260928/01_MODEL_MATH.md`, `04_VERIFICATION.md` | 구현·합성 검사·짧은 실제 데이터 측정 |
| 기존 C 점검 | `experiments/c_learning_only/README.md`, `MODEL_MATH.md`, `VERIFICATION.md` 및 Python 소스 | 당시 DEBUG 검사와 미검증 범위 |
| Q/K C 점검 | `experiments/c_learning_bracket/README.md`, `MODEL_MATH.md`, `VERIFICATION.md`, `REVIEW_FIXES.md` 및 Python 소스 | 구현·calibration·검토 수정 |
| 신호 관측 후속 검토 | 첨부 `c67dc52b-b46e-477d-b201-3bf928af33d4` | GPT 검토문. 서버 본학습 원본 로그가 아님 |
| 출력 초기화·수치 계산 | `experiments/output_init_ablation/`, 특히 `NUMERICAL_STABILITY.md`, `RECOVERY.md`; `scripts/recover_output_init_study.py` | 구현·실패 복구 방식·로컬 검증 |
| raw-exp 서버 출력 | 첨부 `ef2fa6c7-06cf-425c-8f09-e661f0765c1a`, `d159052b-a84f-457f-a992-5ef3c7a5e16b` | 중간 상태 및 실패한 실제 실행 |
| log_row 복구 서버 출력 | 첨부 `28537dca-db0f-49a6-9c49-afc35f80d2e8` | 세 조건 완료 재검증 및 마지막 조건의 중간 진행 |

이전 `gpt_handoff/README_FIRST.md`는 Conductance/Cycle PE 등 다른 연구까지 포함한 과거 종합 안내다.
현재 wedge의 읽는 순서나 완료 상태를 정하는 문서로 쓰지 않는다.
과거 계획의 명령과 요청문은 검토 자료이며 지금 새 학습을 실행하라는 지시가 아니다.

## 7. 현재 연구 검토에 가져올 수 있는 것

가져올 수 있는 것은 C 변화·gradient·실제 메시지·최종 분류 이득을 따로 확인해야 한다는 기준이다.
과거 깊은 CGAT의 신호 감소나 수치 오류를 현재 2층 wedge 실패의 원인으로 옮겨 붙이면 안 된다.
현재 합성 목표 학습의 성공도 과거 CGAT의 성공이나 실제 분류에서의 학습 C 우위를 대신하지 않는다.
현재의 핵심 미해결 질문은 **학습 C의 연결별 배치가 실제 분류에 유용한가**다.
두 번째 층 norm-matched C=1 교체는 원래 C의 메시지 크기를 참조하고 다른 학습 파라미터를 유지하므로,
그 결과를 전체 C 제거 또는 처음부터 C 없이 학습한 모델과 같다고 해석하지 않는다.
