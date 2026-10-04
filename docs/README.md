# NEW GAT 문서 모음

최신 수정: [실제 arxiv CUDA 메모리 측정·진단 projection·allocator 예산](ARXIV_MEMORY_REVIEW_20260927.md).

이전 수정: [energy 정밀도·실제 BF16 검사·reference 순서 재계산](ENERGY_PRECISION_REVIEW_20260927.md).

이전 수정: [fused caller·수동적 진단·validation 회귀](FUSED_REVIEW_FIXES_20260927.md).

최신 코드 보완과 실행 계약: [네 검수 문서 후속 구현](REVIEW_REMEDIATION_20260927.md).

프로젝트의 일반 사용자 안내, 연구 설계와 개별 실험 문서는 이 `docs/` 폴더에서 관리한다.
외부 GPT에 줄 전체 프로젝트 검토 묶음은 별도의 **[`gpt_handoff/`](../gpt_handoff/README_FIRST.md)**
폴더에 있으며, 루트 `README.md`는 두 위치로 들어오는 짧은 입구만 제공한다.

## 처음 실행할 때

| 문서 | 내용 |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 설치, 데이터 준비, 전체 재현과 트랙별 실행 명령 |
| [ENVIRONMENT.md](ENVIRONMENT.md) | CUDA·Conda·glibc 환경별 설치와 문제 해결 |
| [DATASETS.md](DATASETS.md) | 데이터 원본, split, cache, metric 계약 |

## 현재 상태와 성능 범위

| 문서 | 내용 |
|---|---|
| [EXPERIMENT_STATUS.md](../gpt_handoff/EXPERIMENT_STATUS.md) | 완료된 결과, 로컬 검증, 아직 실행하지 않은 실험 |
| [RICH_SCALING_EXPERIMENTS.md](../gpt_handoff/RICH_SCALING_EXPERIMENTS.md) | Conductance V1–V5, Cycle PE V1/V2, Tree의 reference/large 전체 실험(118 child / 122 model trainings) |
| [PERFORMANCE.md](PERFORMANCE.md) | 구현 최적화, 시간·메모리 측정법, 미측정 범위 |

## Conductance 연구

| 문서 | 내용 |
|---|---|
| [CONDUCTANCE_GAT.md](CONDUCTANCE_GAT.md) | 기본 Conductance 모델과 benchmark |
| [CONDUCTANCE_FACTORIAL.md](CONDUCTANCE_FACTORIAL.md) | Gate weight decay × normalization 2×2 실행법 |
| [CONDUCTANCE_FACTORIAL_FINDINGS.md](CONDUCTANCE_FACTORIAL_FINDINGS.md) | 2×2 GPU 결과와 해석 |
| [CONDUCTANCE_C_LEARNING.md](CONDUCTANCE_C_LEARNING.md) | learned C × fixed C 실행·checkpoint 검사법 |
| [CONDUCTANCE_C_LEARNING_FINDINGS.md](CONDUCTANCE_C_LEARNING_FINDINGS.md) | C-learning GPU 결과와 해석 |
| [CONDUCTANCE_DIAGNOSTICS.md](CONDUCTANCE_DIAGNOSTICS.md) | 기존 checkpoint의 읽기 전용 진단 |
| [CONDUCTANCE_V2.md](../gpt_handoff/CONDUCTANCE_V2.md) | 엣지별 C 직접 학습 |
| [CONDUCTANCE_V3.md](../gpt_handoff/CONDUCTANCE_V3.md) | 상대 C graph operator 학습 |
| [CONDUCTANCE_V4.md](../gpt_handoff/CONDUCTANCE_V4.md) | 상대 C graph operator × spatial W 2×2 통합 문서 |
| [CONDUCTANCE_V5.md](../gpt_handoff/CONDUCTANCE_V5.md) | 입력별 C 최적화·가중 라플라시안·multi-head W와 기존 학습 상태를 보존하는 전환 |
| [INCIDENCE_ABLATION.md](INCIDENCE_ABLATION.md) | 기존 backbone의 독립 8조건 lift/cross-depth 내부 ablation |
| [AGGREGATION_COMPARISON.md](AGGREGATION_COMPARISON.md) | 외부 residual/FFN 없는 incidence 4조건과 DUALFormer 2조건, CUDA 검증·미측정 범위 |

## Cycle PE와 Tree Augmentation

| 문서 | 내용 |
|---|---|
| [CYCLE_PE.md](CYCLE_PE.md) | 기본 Cycle PE 모델과 benchmark |
| [CYCLE_PE_V2.md](../gpt_handoff/CYCLE_PE_V2.md) | 구 V2를 폐기하고 교체한 cycle-space projector PE V2 |
| [TREE_AUGMENTATION.md](TREE_AUGMENTATION.md) | 고정 tree·다중 tree augmentation 실험 |

## 연구 구조와 후속 아이디어

| 문서 | 내용 |
|---|---|
| [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) | 세 독립 연구 트랙의 경계와 코드 위치 |
| [COMBINED_LATER.md](COMBINED_LATER.md) | 트랙 결합을 현재 결과와 분리해 둔 후속 아이디어 |
| [로컬 이차 에너지·집합 사이 쌍선형 관계](../research/local_energy_relations/README.md) | 원래 로컬 E/J 연구의 첫 고정 진단: 전체 induced ego, 집계·복원·전달, 서버 FULL 실행 |
| [수신 측 집계와 복원](../research/local_energy_relations/receiver_aggregation/README.md) | 송신별 보존·수신 합·E/J 추가의 다섯 조건, 전체 201개 입력 재사용과 서버 실행 |
| [수신 집계 구현 검증](RECEIVER_AGGREGATION_VERIFICATION_20261004.md) | 단위 검사, GPU DEBUG 범위와 실제 수치, 서버 worker 오류 수정 |
| [수신 집계 서버 FULL 결과](RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md) | 201개 완료 원문, 실제 q/E/J 재구성 수치와 다음 국소 예측 질문 |
| [로컬 E/J 실제 예측 실험](../research/local_energy_relations/prediction/README.md) | 같은 두 홉 기본 전파의 base/E/J/both × 두 고정 C, 전체 citation 3개·336회 학습·서버 실행 |
| [로컬 E/J 예측 구현 검증](LOCAL_PREDICTION_VERIFICATION_20261004.md) | 독립 수식·미분·학습·이어하기·평가 검사와 CUDA DEBUG 실행 |
| [로컬 E/J 예측 서버 FULL 결과](LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md) | 336회 완료 원문, 기본 대비 정확도와 첫 층·출력층 제거 결과, 주입 위치 대조 제안 |
| [E/J 위치별 재학습](../research/local_energy_relations/placement/README.md) | 첫 층만·출력층만·두 층 모두 × E/J/both, 두 고정 C와 base의 20조건·840회 서버 학습 |
| [위치별 실험 구현 검증](LOCAL_PLACEMENT_VERIFICATION_20261004.md) | 이전 대조군과 계산 일치, 활성 파라미터·학습·평가·이어하기 검사와 CUDA DEBUG 범위 |
| [E/J 위치별 서버 결과](LOCAL_PLACEMENT_SERVER_FINDINGS_20261004.md) | 840회 계획의 제출 요약, 54개 base 대비 비교, 층별 재학습·제거 반응과 채널별 특징 대조 제안 |

## GPT 전체 프로젝트 전달 묶음

| 문서 | 내용 |
|---|---|
| [2026-10-04 전체 실험 전달 안내](gpt_experiments_20261004/00_READ_FIRST.md) | 이전 Conductance·CGAT, wedge 전 단계, 최근 로컬 E/J와 위치별 결과, 원래 통합 에너지 구상과 구현 차이 |
| [현재 연구 의도와 실제 수식](gpt_experiments_20261004/01_RESEARCH_AND_MATH.md) | E/J 결합의 조건, 현재 J와 L² 내부항, 기본 전파와 GCN의 차이, 아직 구현하지 않은 것 |
| [GPT 검토 반영](gpt_experiments_20261004/07_GPT_REVIEW_RESPONSE.md) | 수신 실패/완료 색인 수정, 통합 에너지의 수학적 조건과 성능 원인 미확정 |
| [README_FIRST.md](../gpt_handoff/README_FIRST.md) | GPT에 전달할 정확한 10개 파일, 읽는 순서와 요청문 |
| [HANDOFF.md](../gpt_handoff/HANDOFF.md) | 전체 구현 이력, 검증 근거, 남은 작업과 외부 검토 질문 |
| [CODE_SUMMARY.md](../gpt_handoff/CODE_SUMMARY.md) | 현재 source/test/config/script의 파일별 원문 스냅샷 |

새 문서를 추가할 때는 본문을 다른 폴더의 README에 만들지 않고 이 폴더에 두며, 이 목록에도
같이 추가한다.
