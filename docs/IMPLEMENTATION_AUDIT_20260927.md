# 2026-09-27 요구사항·구현 재검수

## 판정

`08521c8`을 원래 요청 전체의 완료본이라고 보고한 판단은 잘못됐다.
PPI 선택뿐 아니라 학습 계약 변경, 주 실험 통합 누락, 평가 잠금 결함과
검증 범위 과장이 있었다. 이 문서는 그 커밋 및 앞선 검수 ZIP의 완료 주장을
정정한다. 테스트 개수는 아래 미구현 항목을 충족했다는 증거가 아니다.

## 확인된 결함과 누락

| 항목 | 코드에서 확인한 사실 | 처리/남은 작업 |
| --- | --- | --- |
| 데이터셋 변경 | 기존 주 비교는 `ogbn-arxiv`; 새 `sampled_inductive`는 PPI로 고정했다. | PPI를 주 실험으로 권한 문서를 정정했다. PPI 코드를 주 실험 완료로 세지 않는다. |
| 주 4조건 비교 미완성 | aggregation runner의 incidence sampling 제한만 해제했다. PPI 전용 실행기는 ogbn-arxiv 4조건 controller/report를 제공하지 않는다. | ogbn-arxiv의 fixed/learned C × full/sampled 통합 비교는 여전히 미완성이다. |
| 학습 계약 임의 변경 | 새 runner는 `learning_budget_policy=epochs`, patience=epochs, 자체 200-epoch 루프를 사용한다. 기존 비교는 reference_updates 및 patience=50이다. | 기존 계약을 유지했다고 보고할 수 없다. PPI 구현을 기존 주 실험에 대입하지 않는다. |
| 업데이트 수의 교란 | 기존 reference_updates도 full/sample 업데이트 수를 서로 맞추는 정책이 아니다. 각 모드의 기준 배치보다 업데이트를 줄이지 않는 정책이다. | 합성 직접 검사에서 full은 계획 200회, sampled는 600회였다. 실제 arxiv 수치가 아니다. 조건부 효과를 순수 sampling 효과/동일 update 비교로 주장하면 안 된다. |
| test 잠금의 실제 결함 | 조건 수만 확인한 뒤 한 조건씩 검사·평가했다. 뒤쪽 checkpoint가 잘못되면 앞쪽 test 경로가 이미 실행될 수 있었다. | 실패하는 반례를 직접 재현했다. 정확한 조건 집합 및 모든 디스크 증거를 먼저 검증하고, 그 뒤 전체 hash를 잠그도록 수정했다. |
| 샘플링 검증 공백 | 앞서 새로 추가한 arxiv 관련 검사는 CLI 계획 검증이었다. PPI GPU 성공이 arxiv 데이터 경로 검증을 대신하지 못한다. | 실제 PreparedInputs/sampler→CUDA loss/backward/optimizer→full validation을 합성 단일 그래프로 새로 검사했다. 실제 데이터 성능 검사는 아니다. |
| 일반화 주장 범위 | ogbn-arxiv 공식 split에서는 전체 그래프 구조/특징을 context로 사용한다. | 미관측 노드 label 평가와 독립된 새 그래프 일반화를 구분한다. 새 독립 그래프 평가 protocol은 아직 별도 정의·구현이 필요하다. PPI로 임의 대체하지 않는다. |

## 코드 경로와 기존 증거로 확인한 부분

- **C 학습:** `GraphOptimizedConductance`는 특징/연결/구조 통계로 비용을 만들고
  유한 K-step을 계산한다. task 경로의 c에는 detach가 없다. last_c 등 진단용
  사본만 detach한다. 평가 시 정답을 solver에 전달하지 않는다. C는 입력 그래프가
  아니라 입력으로부터 계산하는 전도도이며, 공유 theta는 새 그래프마다 초기화하지 않는다.
- **새 backbone:** 공통 linear encoder/decoder와 incidence operators를 사용한다.
  역사적 classifier의 외부 residual·SwiGLU FFN·입출력 LayerNorm을 forward에 넣지 않는다.
  `I-beta D^-1 L`의 I 항은 diffusion 정의이다. C 생성기 내부의 특징 정규화는 별개이다.
- **Energy/lift:** 3종 energy × 4종 lift가 같은 backbone에 있다. fixed/shared C
  controls를 포함해 incidence 16조건이다. 대각·교차항은 분리돼 있으며 Gram은
  lift 전의 공통 projected value 좌표를 쓴다. depth state를 정확한 거리별 hop shell로
  바꾸어 설명하면 안 된다. energy readout 초기값 0일 때 C로 가는 추가 energy gradient는
  첫 step에 0이고, 기존 diffusion 경로의 C 학습은 존재한다.
- **Gram backward:** 새 analytic first derivative는 history/C gradient를 한 번씩
  할당한다. 이전 CUDA FP64 dense/finite-difference와 nested checkpoint 검사를
  다시 근거와 대조했다. 이번에 이 계산 코드는 수정하지 않았다. double backward와
  전체 네트워크 가역성은 지원/입증한 기능이 아니다.
- **외부 비교:** GATv2는 실제 GATv2Conv이며 외부 skip이 없다. DUALFormer의
  원형 내부 residual/LayerNorm은 유지하고 no-skip 변형을 별도 표기한다.
  공통 레시피 비교이며 파라미터 수 일치나 각 논문의 최적 점수 재현이 아니다.
  단순히 이 baseline들이 있다는 이유로 SOTA 비교 완료라고 판단할 수 없다.
- **선택 점수 재현:** aggregation의 best reload, integrity, audit에서 선택 당시
  정수 count와 점수의 재현을 검사한다. 이번에 발견한 새 PPI matrix 선검증 결함과
  이 기존 per-cell 재현 검사를 혼동하지 않는다.
- **기존 결과 보존:** research/src/scripts 및 기존 incidence_ablation 코드는 이번
  재검수에서 수정하지 않았다. 이전 checkpoint와 실제 실험 결과를 가져와 새 조건의
  성능 증거로 사용하지 않았다.

## 이번 직접 실행의 범위

로컬 RTX 5070 Ti 16GB에서 forward/backward/optimizer는 CUDA로 실행했다.
CPU 검사는 sampler/metadata 제어이며 CPU 모델 학습은 실행하지 않았다.

`tests/test_aggregation_sampling_path_cuda.py`:

- full, cluster, cluster_disjoint × fixed C, shared C, energy+pre-lift의 9조건.
- reference 크기 8 layers / 256 hidden / 8 heads 유지.
- **명시적 합성 입력**: 160 nodes, 학습 seed 96개, validation 32개.
- 실제 sampler에서 각 학습 seed가 정확히 한 번 등장하고, disjoint context가
  물리적으로 묶이는지 검사했다.
- 실제 CUDA loss/backward와 optimizer 변경, 모든 trainable parameter의 CUDA
  gradient 연결, full-graph validation의 label count를 검사했다.
- labels만 바꿔도 logits와 C가 동일한지 검사했다.
- 한 합성 epoch 검사는 본학습/실제 ogbn-arxiv 데이터/학습 수렴 검증이 아니다.

`tests/test_sampled_inductive_matrix_integrity.py`:
잘못된 뒤쪽 checkpoint가 있을 때 test 입력 경로 호출보다 먼저 실패해야 한다.
수정 전 실패를 재현했고 수정 후 통과했다. mock metadata 제어 검사이며
가짜 학습 실적이나 checkpoint를 만들지 않는다.

증거: `results/scope-audit-20260927-01.xml` (10 passed: CUDA 9 + control 1).
정상 matrix 평가 경로 회귀: `results/scope-audit-20260927-02.xml`.
기존 130개 기록과 이번 재검수를 더해 연구가 완성됐다고 주장하지 않는다.

## 아직 완료가 아닌 것

1. 주 데이터셋 ogbn-arxiv에서 예산·coverage·비용의 비교 기준까지 고정한 4조건 통합 실험.
2. 별도 그래프를 실제로 홀드아웃하는 새 그래프 일반화 protocol.
3. 실제 데이터 본학습, 공식 test 성능, 통계적 우위, end-to-end 속도 개선.
4. A100 MIG 10GB에서 원래 크기를 유지한 실측·calibration.
5. 정확한 거리별 B_r, bipartite boundary solver, 새 backbone의 역복원/Jacobian,
   cycle/heterogeneous 관계 확장. 이전 논의에 나왔다는 사실을 현재 구현 완료로 세지 않는다.

주 4조건 실험이 완성되기 전에는 이전 PPI 실행 명령을 다음 본실험 명령으로 권하지 않는다.
