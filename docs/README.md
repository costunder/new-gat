# NEW GAT 문서 모음

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
| [RICH_SCALING_EXPERIMENTS.md](../gpt_handoff/RICH_SCALING_EXPERIMENTS.md) | V1을 포함한 전체 트랙의 큰·깊은 모델 확장 실험 |
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

## Cycle PE와 Tree Augmentation

| 문서 | 내용 |
|---|---|
| [CYCLE_PE.md](CYCLE_PE.md) | 기본 Cycle PE 모델과 benchmark |
| [CYCLE_PE_V2.md](../gpt_handoff/CYCLE_PE_V2.md) | 좌영공간 기저벡터 전체를 입력하는 별도 v2 |
| [TREE_AUGMENTATION.md](TREE_AUGMENTATION.md) | 고정 tree·다중 tree augmentation 실험 |

## 연구 구조와 후속 아이디어

| 문서 | 내용 |
|---|---|
| [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) | 세 독립 연구 트랙의 경계와 코드 위치 |
| [COMBINED_LATER.md](COMBINED_LATER.md) | 트랙 결합을 현재 결과와 분리해 둔 후속 아이디어 |

## GPT 전체 프로젝트 전달 묶음

| 문서 | 내용 |
|---|---|
| [README_FIRST.md](../gpt_handoff/README_FIRST.md) | GPT에 전달할 정확한 9개 파일, 읽는 순서와 요청문 |
| [HANDOFF.md](../gpt_handoff/HANDOFF.md) | 전체 구현 이력, 검증 근거, 남은 작업과 외부 검토 질문 |
| [CODE_SUMMARY.md](../gpt_handoff/CODE_SUMMARY.md) | 현재 source/test/config/script의 파일별 원문 스냅샷 |

새 문서를 추가할 때는 본문을 다른 폴더의 README에 만들지 않고 이 폴더에 두며, 이 목록에도
같이 추가한다.
