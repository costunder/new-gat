# 실험 결과와 구현 상태

기준일: 2026-09-05 (Asia/Seoul).

이 문서는 사용자가 제공한 **서버 결과 출력**과 **현재 소스 버전의 구현**을 구분한 기록이다.
문서 작성 자체가 새 학습을 실행했다는 뜻은 아니다. 수치는 사용자 로그에서 확인했으며,
서버의 전체 원본 checkpoint/manifest/history 파일을 로컬로 받아 독립 재검증한 것은 아니다.

## 1. 소스 버전과 측정 범위

### 현재: 실제 학습 자원 calibration 도입, 서버 측정 결과는 아직 없음

사용자 A6000 GPU 3 화면은 9,311/46,068MiB와 utilization 100%였다. GPU 실행 중이라는
관측이며 최적 batch·처리량·전체 학습 완료 근거는 아니다. 이전 `a6000-48gb` profile에는
여러 후보의 실측 선택이 연결되지 않았으며 그 누락을 수정한다.

현재 rich 실행은 V5/Cycle V2 본 학습 전 실제 optimizer update/state를 포함한 batch/worker
probe를 실행하고 paired 조건에 같은 선택을 적용한다. 모델/데이터 축소 없이 최소 요청
batch부터 탐색하며 실패를 작은 값으로 숨기지 않는다. Probe와 본 학습·평가를 분리하고
source/GPU/runtime/data identity에 묶인 계획 및 진행 상태를 보존한다. 자세한 정책과 명령은
[RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md) 첫 절에 있다.
실제 A6000 calibration 결과·최적 선택값·가속률·전체 새 학습 결과는 아직 수령하지 않았다.

이번 calibration 변경 후 전체 로컬 회귀는 **2,053 passed / 101 skipped** (156.76초)다.
Ruff 정적 검사, diff 공백 검사와 코드 스냅샷 일치 검사도 통과했다. 생략 항목은 실제
CUDA/BF16, 미설치 PyG, Linux/Bash 및 native Linux 기능, Windows symlink 권한,
별도 opt-in IPC stress 등이다. CPU 단위 테스트와 GPU 호출 모의 검증을 실제 GPU smoke
test로 간주하지 않는다. 이번 작업에서 실제 A6000 측정·전체 실제 데이터 학습·전체 평가는
실행하지 않았다. 모델·연산자·데이터 설정은 축소하지 않았고 기존 서버 작업·결과는 보존했다.
아래 역사적 회귀 수치는 이번 검증 수와 구분한다.

### 역사적 로컬 검증

throughput/IPC/격리 교정 당시 전체 로컬 CPU 회귀는 **1,940 passed / 99 skipped**
(189.74초)다. `CYCLE_V2_IPC_STRESS_GRAPHS=10000`으로 실제 2-worker의 1만 합성
그래프 준비·캐시 검증까지 이 전체 실행에 포함했다. Ruff·코드 스냅샷 일치 검사도 통과했다.
생략은 Linux/Bash, native Linux renameat2, Windows symlink 권한, 미설치 PyG, CUDA/BF16
조건이다. 실제 서버의 mmap/FD 한도와 GPU 전체 학습·평가를 검증한 결과는 아니다.
실제 이전 결과의 격리/삭제도 실행하지 않았으며 격리 테스트는 합성 디렉터리만 사용했다.

2026-09-05 SE/PE 분리 완료 당시 로컬 CPU 전체 회귀는 **1,852 passed / 98 skipped /
1 warning** (164.78초)다. Ruff 및 코드 스냅샷 일치 검사도 통과했다. 생략은 Linux/Bash,
Windows symlink 권한, 미설치 PyG, CUDA/BF16 환경 조건이며 경고는 PyTorch multiprocessing의
sparse tensor 재구성 경고다. 이는 실제 데이터 전체 학습·평가나 A6000 성능 검증이 아니다.
아래 이전 구현 시점의 테스트 수치는 역사 기록으로 보존한다.

### ad041e2 이후 수령한 실제 실패와 후속 수정

사용자 첨부 `9e790812-518b-4369-9745-88f1fc46d76e/pasted-text.txt`의 SHA-256은
`BE7433E5C4C5570B26C83F301845EA0DF557829C0DC45F77163C9307B5EC3E63`다.
run ID는 `v5-cycle-se-pe-a6000-gpu3-seed0-v1`, A6000 GPU 3,
Torch 2.7.1+cu118, model seed 0, 계획은 V5 20학습과 Cycle SE/PE 8학습이었다.
전체 checkpoint/manifest를 로컬로 수령한 독립 재검증은 아니다.

- V5 첫 reference/ogbn-arxiv/fixed-C가 54 epochs·2,430 steps를 완료하고
  최고 validation 0.719621, allocator peak 7,505,515,008 bytes를 기록했다.
  child 저장 후 `passed`를 출력했지만 집계의 `throughput.scope` 검사에서 실패했다.
  나머지 19개 V5는 미실행이며 이 값으로 fixed/dynamic 또는 reference/large 비교를 하지 않는다.
- Cycle 8개 조건은 모두 전처리 중 공유 storage mmap 실패로 학습을 시작하지 못했다.
  약 7천 그래프 이후 몇 bytes 매핑도 ENOMEM인 현상은 mapping-count 누적과 맞지만
  서버의 실제 `vm.max_map_count`/cgroup/주소공간 한도는 확인하지 못했다.
- GPU utilization은 `pynvml` 미설치와 UUID 없는 numeric-device mapping 거부로 null이다.
  이를 GPU 사용률 0%로 해석하지 않는다. 이 계측 부재가 위 두 실패를 일으킨 원인은 아니다.

후속 수정은 V5 telemetry writer/consumer 계약 정합화와 Cycle 양방향 Tensor-free IPC다.
V5/Cycle 모델 구조·파라미터·데이터·batch는 ad041e2 대비 바뀌지 않는다. 이전 테스트는
실제 writer→집계 연결과 Linux 대량 IPC 누적을 놓쳤으므로 그 테스트 수를 전체 GPU 검증으로
인용하지 않는다. 이전 실패 결과는 삭제/수정하지 않고 선택적으로 whole-run 격리하며
새 source run과 혼합하거나 hash 검사를 우회하지 않는다.

현재 최상위 구조는 [Conductance V5](CONDUCTANCE_V5.md)와 새
[Cycle PE V2](CYCLE_PE_V2.md)다. V5는 shared graph-conditioned dynamic C와 multi-head W,
graph-conditioned beta를 분리했고, `fixed_c/shared_dynamic_c` 두 arm을 연구급 architecture에서
비교한다. Cycle의 과거 `cycle_basis_v2`와 QR 기반 `cycle_projector_pe_v2`는 현재 경로에서
제외했다. 구조 요약을 PE라고 부르던 `cycle_dfs_sparse_pe_v2`와도 결과를 분리한다.
현재 SE는 `cycle_dfs_se_v2`, PE는 `cycle_dfs_relative_pe_v2`다. PE는 동일 SE/backbone에
cycle 내부 상대 위치 residual을 더하며 추가 trainable parameter는 0개다. 유일한
`dfs_fundamental` backend의 전체 기저와 실제 cycle 순서를 sparse로 사용하고
QR/SVD/Gram inverse를 하지 않는다. 선택된 DFS tree에 의존하며 일반 기저변환 불변성은
주장하지 않는다. 두 조건의 실제 GPU 전체 학습·성능·가속 결과는 아직 없다.

현재 [전체 scaling](RICH_SCALING_EXPERIMENTS.md)은 `reference/large` 두 architecture
profile을 사용한다. Conductance V1–V5는 106 child/model trainings, Cycle V1/V2는 12
child/model trainings, Tree V1/V2는 4 child 안에서 8 models를 학습한다. 따라서 합계는
**122 child runs / 126 fresh model trainings**다. Cycle은 V1 4학습과 V2 SE/PE 8학습이다.
동일 run-id는 검증된 완료 child를 skip하며
같은 새 source/config/schema의 V5와 Cycle V2 `last.pt`가 있으면 epoch 상태를 복원한다.
현재 Cycle cache는 `cycle_pe_v2_ordered_dfs_benchmark`이며 SE/PE가 공유한다. 두 encoding의
checkpoint는 교환할 수 없다. 구 cache/checkpoint와 수정 전 V5 source checkpoint는 호환되지 않으며 hash 검사를
우회하지 않는다. 완료한 전체 matrix를 다시 실행하라는 의미는 아니다. 과거의
`base/wide/deep/large` 204-training 계획은 폐기됐고 현재 실행 계약이 아니다.

실행 hardware profile은 보수적인 `portable`과 opt-in `a6000-48gb`로 분리된다. A6000
profile은 실제 minibatch/sample 크기와 수치 실행을 바꾸므로 portable 결과와 점수나 실행 시간을
직접 대응시켜 모델 효과 또는 GPU 효과로 해석할 수 없다. 2026-09-04 첫 A6000 실행은 아래
메모리·수치 오류로 중단됐으며 성공한 scaling 성능이나 가속 실측으로 사용하지 않는다.

현재 통합 runner는 단일 `--device`에서 track을 순차 실행하고, 서로 다른 indexed 장치를
`--devices`로 명시했을 때만 independent track을 GPU별로 병렬 배정한다. 같은 GPU의 최상위 track
concurrency는 1이며, Tree A6000 child 내부의 명시적 candidate concurrency 2는 별도 계층이다.
새 resource monitor는 실제 GPU 서버에서 Conductance V5, Cycle PE V1/V2와 Tree V1/V2의 GPU
SM/메모리 utilization, CUDA allocator, process CPU·RSS/HWM과 system available RAM을
주기적으로 기록하고, 미지원 측정은 `null`과 원인을 남긴다. 그러나 이 계측이 적용된 수정판의
완료 GPU artifact는 아직 수령하지 않았으므로 현재 문서에 새 utilization 수치를 제시하지 않는다.
당시 Rich runner는 여러 physical batch 후보를 자동 튜닝하지 않았고 manifest에도
`throughput_candidate_sweep=false`를 기록했다. 현재는 첫 절의 optimizer-inclusive calibration이
이 누락을 보완하며 본 학습 중에는 선택값을 고정한다. 별도 CUDA microbenchmark는
Conductance V1/V5, Cycle PE V1/V2, Tree V1/V2의 명시 후보마다 device-wide GPU
utilization·peak VRAM·CPU/RAM·처리량과 수치/gradient integrity를 측정하고 10% memory headroom
기준의 권고를 낸다. 다만 optimizer state·전체 epoch·validation/checkpoint가 없는 고정 real-batch
측정이고 profile 기본값을 자동 변경하지 않으므로, 이를 새 calibration의 대체 근거로
주장할 수는 없다.

후속 사용자 요청으로 현재 기본 실행은 model seed **0 하나**다. 기존 5-seed 측정값은 아래에
그대로 보존하며, 기본값 변경이 과거 결과나 source revision을 바꾸지는 않는다.
단일 seed의 std/CI는 null로 기록한다. 새 read-only `--full-audit`는 C 평균/셔플/전파 제거와
train-label gradient를 검사하며 **5e801c3 실행의 새 GPU 로그를 수령했다.** 아래 확장 검사 절에
기록했다. 이후 **43afd63의 2×2 GPU 재학습 결과도 수령했으며 여덟 조건 모두 passed**다.
이어 **C-learning의 네 조건 비교도 모두 passed인 보고서를 수령했다.** 새 run의
`learned_c` checkpoint의 평균-C 검사도 이후 `passed` GPU 출력을 수령했다.
다음은 엣지별 직접 C의 Conductance v2, 공유 상대 C 생성기의 v3, 그리고 상대 C graph
operator와 spatial message transform을 함께 학습하는 v4다. 세 버전은 별도 구현·실행 경로를
사용한다. 2026-09-02 사용자 보고상 **과거 arxiv-only** Conductance v2/v3 runner와 Cycle PE v2
runner는 각각 `passed`다. 성능 수치와 전체 원본 artifact는 수령하지 않아 이 문서에 창작해
넣지 않는다. 과거 arxiv-only V4는 첫 arm의 200 epochs·child exit 0 뒤 구 report gate에서
중단되어 정식 결과가 아니다. 현재 확대된 V2/V3/V4 8/10/20-job 전체 결과는 수령하지 않았다.

이후 단일 `hidden/layer` 설정만으로는 큰 모델에서의 적합도를 확인할 수 없다는 사용자 요청에
따라 [전체 모델 규모 확장 실험](RICH_SCALING_EXPERIMENTS.md)을 연구급 크기로 재설계했다.
Conductance V1–V5, Cycle PE V1/새 V2, Tree fixed/multi를 dataset-aware `reference`/`large`로
실행한다. 기본 model seed 0 계획은 Conductance 106, Cycle 12, Tree 8 trainings로 총
**122 child runs / 126 model trainings**이다. Tree child 하나가 fixed/multi 두 모델을
학습하므로 두 수가 다르다. Cycle/Tree 후보는 validation-only로 비교한 뒤 선택 checkpoint만
test-only로 평가한다. Cycle은 encoding×dataset별 선택이므로 V1 포함 6회, V2-only 4회 test다.
현재 확인된 것은 runner·manifest·무결성 검사와 로컬 테스트이며 이 GPU
학습 결과는 아직 실행·수령하지 않았다. 같은 인수와 run ID 재실행은 완료 child를 검증·skip하고,
V5와 새 Cycle V2는 `last.pt`부터 이어지며 legacy 미완료 child만 처음부터 재시도한다.

### 2026-09-02 사용자 서버 실행 보고

사용자 terminal 보고의 공통 source/pull revision은 `7b4cd32`다. GPU preflight에는
`NVIDIA A100-SXM4-80GB MIG 1g.10gb`가 기록됐고, 서버 할당은
`CUDA_VISIBLE_DEVICES=6`, 프로세스 내부 논리 장치는 `cuda:0`이었다. 아래 상태는 사용자
출력 기준이며 성능 수치와 전체 원본 artifact를 수령해 독립 검증한 결과가 아니다.

| 트랙 | Run ID | 사용자 보고 상태 |
|---|---|---|
| Conductance v2 | `gat-direct-c-v2-gpu6-seed0-v1` | 과거 arxiv-only 2-job run `passed`; 현재 8-job 결과 아님 |
| Conductance v3 | `gat-relative-c-v3-gpu6-seed0-v1` | 과거 arxiv-only 2-job run `passed`; 현재 10-job 결과 아님 |
| Conductance v4 | `gat-hybrid-c-spatial-v4-gpu6-seed0-v1` | 과거 arxiv-only 4-arm run의 `fixed_c_identity_w`만 200 epochs·child exit 0 뒤 구 report gate 중단; 나머지 3개 pending; 현재 20-job 결과 아님 |
| 폐기된 구 Cycle PE v2 | `cycle-pe-v2-gpu6-seed0-v1` | 당시 runner `passed`; 현재 sparse DFS 모델 결과가 아님 |
| 당시 전체 scaling 계획 | 미실행 | 구 118-child/122-training 계획의 기록이며 GPU 결과 없음; 현재 SE/PE 분리 계획은 위 122-child/126-training 계약 |

### 2026-09-04 A6000 V5·Cycle V2 partial 실행과 old-source r2 재현

Run `new-v5-cyclev2-a6000-gpu3-seed0-r1`, GPU 3 RTX A6000, model seed 0의 사용자
terminal 로그를 수령했다. 첨부 텍스트 SHA-256은
`D6394C813EDB52DD2CE6746B531973A750A1F0AE807EC5A3410B938FAFF6F93E`다. 전체
manifest/checkpoint/history를 로컬로 받은 독립 재검증은 아니다.

- Conductance child는 20개를 계획했다. 첫 `v5/reference/ogbn-arxiv/fixed_c`만 200 epochs와
  child `passed`를 완료했다. 전체 로그 최고 validation은 epoch 10의 0.692775였고, 구
  joint-only primary는 0.680392였다. Train loss가 0.579221에서 0.019955로 내려가는 동안
  validation은 대체로 0.67대로 하락했다.
- 두 번째 `shared_dynamic_c`는 epoch 20 warm-up 뒤 최초 conductance calibration backward에서
  44.47/44.55GiB를 사용하고 추가 104MiB 할당에 실패했다. 나머지 18개 Conductance job은
  미실행이다.
- Cycle V2는 Peptides-struct와 ZINC-12K의 projector cache 준비를 완료했지만 reference/large
  네 학습 모두 첫 epoch 전에 non-finite gradient로 실패해 성공 metric이 없다.
- 원인은 각각 dynamic edge-score autograd activation 누적과 A6000 FP16+GradScaler 초기
  scale overflow로 확인됐다. 당시 수정은 edge-score chunk checkpoint, BF16/no-scaler,
  condition-aware V5 checkpoint selection이었다. 이후 r3의 diffusion OOM 재발과 현재 수정은
  아래에 별도로 기록한다.

구 V5 fixed run은 실제 global-best model state를 저장하지 않아 corrected checkpoint로 복구할
수 없고, 구 partial `last.pt`도 새 selection schema/source와 이어 붙이지 않는다. 수정판은 새
run ID가 필요하다. 당시 동일 projector 모델의 cache 재사용과 달리 현재 sparse DFS 모델은
`cycle_pe_v2_ordered_dfs_benchmark`를 새로 준비하며 projector 및 구 support-only cache를
재사용하지 않는다. 이 partial run의 수치를
fixed-vs-dynamic 또는 V5/Cycle V2 성능 결과로 인용하지 않는다.

후속 run `new-v5-cyclev2-a6000-gpu3-seed0-r2`의 사용자 terminal 로그도 수령했다. 첨부
`bd63fc9a-60da-4daf-9ab9-da49db7cbbe1/pasted-text.txt`의 SHA-256은
`F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6`다. 그러나 새 run ID만
사용했을 뿐 수정 commit `214265c0dcc09a67ae0c9c3d09d9e2cbc26c63bb`는 서버에 적용되지
않았다. Conductance traceback의 `train.py:785`와 `joint_best=` 단독 출력, Cycle traceback의
`benchmark.py:589`는 모두 직전 `08d8ed6` 소스와 일치한다.

r2도 r1과 같은 실패를 재현했다. Conductance는 fixed-C 한 job만 완료한 뒤 dynamic-C 최초
calibration backward에서 다시 44.47/44.55GiB OOM이 발생했고 18개가 미실행이다. Cycle V2
네 job도 모두 첫 epoch 전에 같은 non-finite gradient로 실패했다. 따라서 r2는 edge-score
checkpoint, condition-aware selection, BF16/no-scaler 수정의 검증 결과가 아니며 그 수치나
artifact를 수정판 결과로 사용하지 않는다. 이것은 r2의 역사적 기록이며 r3까지 old source였다는
뜻이 아니다.

### r3 large OOM 재발과 2026-09-05 QR-free 구현

사용자가 추가로 제공한 r3 로그는 4/20번째 `v5/large/ogbn-arxiv/shared_dynamic_c`에서
warm-up 20 epoch 이후 diffusion forward가 192MiB 할당에 실패한 기록이다. GPU 44.55GiB 중
해당 프로세스 44.54GiB, PyTorch allocated 41.70GiB였다. Traceback은 `214265c`와 정확히
일치하므로 score checkpoint 수정판을 실행한 뒤에도 OOM이 남았음이 확인됐다. 192MiB는
131072-edge chunk × hidden384 × FP32 4bytes다. 나머지 job의 완료 여부를 이 로그로 추정하지 않는다.

현재는 V5 diffusion custom backward가 node message/scalar edge weights만 저장하고 edge
features를 chunk별 재계산한다. 기존 규모와 sampling은 유지했다. 관련 CPU 검사 28개 통과,
동일 합성 입력의 고유 autograd 저장량 563,840→60,544 bytes(약89.3% 감소)를 확인했으나
실제 GPU peak나 전체 학습 성공을 검증한 것이 아니다. 상세는 [V5 문서](CONDUCTANCE_V5.md)를 따른다.

Cycle 현재 조건은 QR-free sparse DFS 구조 SE와 동일 SE+cycle 상대 PE이며 backbone 크기는
유지했다. 기본 ZINC parameter는 두 조건 모두 7,262,785개(구 projector보다 +16,704)다.
DFS 탐색 O(V+E), 기저·실제 cycle 위치 출력 O(nnz Z), 각 sparse 집계 O(nnz Z*d)를 구분한다.
SE는 sparse product 2회, PE는 6회이므로 PE가 SE보다 빠르다고 주장하지 않는다.
새 V2만 실행하는 `cycle-se-pe-a6000-gpu3-seed0-v1`은 `--encodings se pe`, 두 dataset·두
profile·seed 0의 8학습과 선택 checkpoint 4회 test다. 명령과 checkpoint/cache 비호환성은
[CYCLE_PE_V2.md](CYCLE_PE_V2.md)에 있다. 완료한 다른 트랙이나 전체 matrix를 다시 실행하지 않는다.

V3는 graph-centered score → bounded relative C → isotropic mixture와 학습 alpha의
대칭 정규화를 사용한다. AdamW backbone/생성기/scalar 그룹을 분리했다. 현재 기본은
Cora/CiteSeer/PubMed/PPI/ogbn-arxiv × `relative_c`/`fixed_c` × seed 0의 10회 학습이며
v2나 이전 MLP의 checkpoint·점수를 재사용하지 않는다.
PPI는 공식 20/2/2 독립 graph split, whole-graph minibatch 2와 BCEWithLogitsLoss를 사용하고
`logit > 0`의 모든 validation node-label 결정을 합친 global micro-F1로 checkpoint를 선택한다.
Test graph는 train/validation loader·forward·loss·metric·선택·진단에는 들어가지 않지만 full
cache의 test tensor와 metadata는 공식 split·shape·checksum 무결성 검사로 load/validate된다.
선택된 checkpoint에서 평균 C·셔플 C·C=1·전파 제거 validation 검사도 별도 forward로 수행한다.
평균 C와 C=1은 대칭 정규화에서 동등하므로 서로 일치해야 하는 수치 검산이다.
실행과 수식·비교 경계는 [전체 인수인계](HANDOFF.md)와 [코드 스냅샷](CODE_SUMMARY.md)을 따른다.
초기 V3 구현 게시 revision은 `6f9d3b0981e8cfa8feb76e59fb348e26cc6909d6`이다.
사용자 보고상 과거 arxiv-only v3 runner는 `passed`지만 validation 수치와 전체 artifact는
수령하지 않았고, 현재 10-job 전체 기본 결과도 없다.

V4는 v3의 상대 C 생성기와 대칭 weighted-degree 정규화를 유지하고, 각 층에 bias 없는
identity-initialized `W`를 추가한다. 한 층은 `C(H)`를 먼저 만들고 비고립 노드에서
`(1-alpha)H + alpha P_C(HW)`를 계산한다. `W=I`이면 v3 전파와 일치한다. 현재 기본은
v1의 5개 데이터 × 고정/상대 C × identity/학습 W × seed 0의 20개 새 CUDA 학습이다. Alpha는
모든 조건에서 학습하고, inactive C/W scaffold는 동결하여 optimizer에서 제외한다. V3 결과를
재사용하지 않으며 네 cell의 조건부 주효과와 interaction만 V4 내부의 기술적 대조로 보고한다.
PPI의 split·batch·loss·`logit > 0` global micro-F1과 test 미평가/cache 무결성 경계는 위 V3와 같다.
선택 checkpoint의 C/W 개입은 재학습 효과와 구분한다. Mean-C와 C=1은 대칭 정규화에서
대수적으로 중복된다. 별도 CUDA forward의 logit 차이와 `allclose_rtol=1e-5`,
`allclose_atol=1e-6`, `within_declared_tolerance`는 informational non-gating 진단이며
arm·report·run의 성공 조건이 아니다. 사용자 보고상 과거 arxiv-only run의
`fixed_c_identity_w`는 200 epochs와
child exit 0까지 완료됐지만 구 numeric hard gate에서 중단되어 나머지 세 arm은 pending이다.
성능 수치는 수령하지 않았고 이 partial arm은 재사용하지 않으므로, 정식 V4 결과에는 새
run의 5개 데이터 × 네 fresh arm, 총 20 jobs가 모두 필요하다.

이전 전체 scaling runner 추가 시점의 구현은 `PYTHONUTF8=1` 전체 로컬 회귀에서
**1418 passed / 77 skipped** (80.24 s, exit 0)를 통과했다. V2/V3/V4 전용 결과는 각각
**118 passed**, **141 passed / 2 skipped**, **131 passed**다. Ruff·compileall과 재생성한
`code_summary --check`도 통과했다. 생략은 Linux/Bash·Windows symlink 권한·로컬 PyG 미설치·
실제 CUDA RNG처럼 이 호스트에서 충족되지 않은 환경 조건이다. 이 검증은 공개 데이터 GPU 학습
결과나 성능 측정이 아니다.

아래 검증 숫자는 5-dataset/PPI 확장 이전 구현 시점의 역사 기록이며 현재 숫자로 재해석하지
않는다. 당시 CUDA 수치검사 교정 후 전체 로컬 회귀는 **1301 passed / 65 skipped** (135.13 s, exit 0),
V4 전용 **122개**와 Ruff·compileall·`code_summary --check`가 모두 통과했다. 생략은 기존
환경별 검사이며 공개 데이터 학습이나 GPU 성능 측정은 로컬에서 실행하지 않았다. Windows의
한국어 작업 경로에서는 기존 V3 fixture의 UTF-8 JSON을 기본 CP949로 읽지 않도록
`PYTHONUTF8=1`로 전체 회귀를 실행했다.

직전 상대 C v3 추가 시점의 전체 로컬 회귀는 **1176 passed / 65 skipped** (44.45 s, exit 0)였다.
V3 전용 134개가 통과했고 실제 CUDA RNG 보존 검사 1개는 로컬 GPU가 없어 생략됐다.

직전 직접 C v2 구현 시점의 로컬 회귀는 **1042 passed / 64 skipped** (47.41 s, exit 0)였다.
직전 평균-C 검사 확장의 924개에 v2 전용 118개를 추가했다. 직접 C의 FP64 미분·chunk 연산,
graph binding과 학습 루프→checkpoint→비교표 연결 및 실제 C gradient coverage를 검사했다.
이전 shared-MLP와 그 평균-C 검사도 전체 회귀에 포함했다. 공개 데이터 학습은 실행하지 않았다.
당시 65개 생략은 Linux/Bash 전용 62개, Windows 실제 symlink 권한 1개, 로컬 PyG 미설치
1개와 실제 CUDA RNG 검사 1개다. 이 로컬 회귀는 Linux/CUDA 실행 또는 GPU 가속의 증거로
제시하지 않는다.

| 구분 | 확인된 상태 |
|---|---|
| 이전 진단 전용 게시 commit | `ebf8cd19b80e6cd6c742b132e2bb1dadb97b019c` |
| 이전 commit의 추가 내용 | Conductance 진단 Python/Bash, 전용 테스트, 안내 문서, 트랙 README의 5개 파일 |
| 기존 학습 코드 | 위 진단 commit은 기존 benchmark의 모델·학습 수식을 변경하지 않음 |
| 현재 Cycle V2 SE/상대 PE | `cycle_dfs_se_v2`와 `cycle_dfs_relative_pe_v2` 구현·CPU 계약 검증; 두 조건의 새 GPU 전체 학습·성능·가속 결과 없음. 과거 projector V2의 네 FP16 실패는 별도 역사 기록 |
| 실행 최적화·선택적 compile·속도 도구 | 이 소스 버전에 포함, 로컬 단위 검증 완료. GPU 가속 실측 미수령 |
| 단일 seed 기본값·확장 checkpoint 검사 | 5e801c3 GPU full audit 수령, seed 0 다섯 데이터셋 passed |
| Gate WD × normalization 2×2 | 43afd63 실제 GPU 결과 수령. PPI/arxiv × 4조건 × seed 0 모두 passed |
| Node-degree의 learned C vs fixed C | `gat-c-learning-seed0-v1`, 2데이터 × 2조건 × seed 0, 모두 passed 보고서 수령 |
| Node-degree checkpoint mean-C 개입 | 새 c_learning/learned_c의 PPI/arxiv GPU 출력 수령, passed. 기존 factorial도 별도 지원 |
| Conductance 직접 C v2 | 과거 arxiv-only `gat-direct-c-v2-gpu6-seed0-v1` 사용자 보고상 `passed`; 현재 4-dataset 기본 결과는 미수령 |
| Conductance 상대 C v3 | 과거 arxiv-only `gat-relative-c-v3-gpu6-seed0-v1` 사용자 보고상 `passed`; 현재 5-dataset 기본 결과는 미수령 |
| [Conductance C × spatial W v4](CONDUCTANCE_V4.md) | 과거 arxiv run은 첫 arm 뒤 구 report gate에서 중단. 현재 5-dataset × 4-condition = 20-arm 정식 결과 없음; 새 전체 run 필요 |
| [Conductance graph-conditioned v5](CONDUCTANCE_V5.md) | r1/r2 partial 및 214265c 적용 r3 large diffusion OOM 수령. 현재 custom backward CPU 검증, 새 GPU 전체 성공·유효 fixed/dynamic 비교 없음 |
| [전체 큰 모델 scaling](RICH_SCALING_EXPERIMENTS.md) | Conductance V1–V5 106 + Cycle V1/V2 12 + Tree fixed/multi 8 = 126 trainings / 122 child 계약. V5/구 Cycle V2 partial 실패 로그만 수령, 전체 GPU 결과 없음 |
| [CODE_SUMMARY.md](CODE_SUMMARY.md) | 이 버전의 source/test/config/script 전체를 파일별로 보존한 스냅샷 |

`ebf8cd1`까지만 받은 서버에는 새 기능이 없으므로 업데이트 후 `git rev-parse HEAD`로
실행 revision을 확인한다. 소스 업데이트가 서버에서의 실행 완료를 뜻하지는 않는다.
아래 기존 학습 결과를 Conductance v2/v3/v4·최적화·새 2×2/C-learning 결과로 재분류하면 안 된다.

### 수령한 C-learning 비교: 학습 C의 성능 이득은 관측하지 못함

2026-09-01 수령한 `gat-c-learning-seed0-v1` 보고서는 model seed 0이며 네 조건과
전체 비교가 `passed`다. Validation만 평가했고 test는 읽지 않았다.

| 데이터 | Learned C (%) | Fixed C=1 (%) | Learned − fixed (pp) | Best / 실행 epoch (learned; fixed) |
|---|---:|---:|---:|---|
| PPI micro-F1 | 52.564966 | 52.705738 | −0.140772 | 64 / 114; 90 / 140 |
| ogbn-arxiv accuracy | 68.317723 | 68.324435 | −0.006711 | 195 / 200; 195 / 200 |

PPI learned에는 비상수 C가 남아 있지만 성능 이득은 없었고, arxiv learned는 C 변동과
점수 차이가 모두 작다. 이는 이 seed·설정에서 이득을 관측하지 못했다는 결과이지
일반적 동등성이나 C의 보편적 무용성을 증명한 것이 아니다. 동결 gate scaffold를 보존하므로
약 69%의 **활성 학습 파라미터 감소**를 저장 공간·GPU 메모리·속도 개선으로 바꾸어 말하지 않는다.

C-learning 구현 게시본은 `25ca328`이지만 제공된 비교표에는 실제 서버 source revision이
없다. 해당 manifest revision을 독립 확인하지 않았으므로 실행 commit으로 단정하지 않는다.
이번 근거는 inline 붙여넣기라 별도 첨부 파일/SHA-256도 없다. 정확한 파라미터 수·층별
진단과 이전 PPI 점수와의 구분은 아래 C-learning 결과 절과 [전체 인수인계](HANDOFF.md)를 따른다.

이어 **같은 새 C-learning run의 learned checkpoint**에 대한 읽기 전용 평균-C 검사도
수령했다. 원 validation 및 원본 무결성 확인 후 그래프·층별 C 변동에 대한 현재 의존도를
검사했으며 재학습·optimizer step·test 평가는 없다. 구체적인 결과는 바로 다음 절을 따른다.

### 수령한 평균-C 검사: PPI checkpoint 의존도와 재학습 이득은 다름

사용자가 제공한 inline terminal 출력의 revision 표시는 `8f6b4da`, 검사 상태는 `passed`다.
대상은 `gat-c-learning-seed0-v1`의 seed 0 `learned_c`다. 원 validation은 PPI
52.564966% 그대로 재현됐고 arxiv는 저장 68.317723%, 재계산 68.317729%다.

| 데이터 | 전체 층 평균 C 후 validation (%) | 원 재계산 값 대비 Δ(pp) | 바뀐 예측 (%) |
|---|---:|---:|---:|
| PPI micro-F1 | 45.915526 | −6.649440 | 7.619317 |
| ogbn-arxiv accuracy | 68.284171 | −0.033558 | 0.184570 |

PPI에서 layer 0만 평균화한 차이는 −6.198266pp, layer 1만 평균화한 차이는 −0.238916pp다.
따라서 그 learned checkpoint는 엣지별 C 패턴에 의존한다. 그러나 처음부터 학습한 fixed C
모델은 52.705738%였다. **고정된 checkpoint의 개입 민감도와 다른 모델을 새로 학습한
성능 차이는 다른 질문**이다. 층별 효과도 더해서 전체 효과로 만들지 않는다.
arxiv는 layer 0 prediction flip이 0이며 layer 1 개입이 전체 개입과 같은 작은 점수 차이를 냈다.
6개 개입의 정확한 점수·flip 단위·logit 변화는 이 문서의 C-learning 결과 절에 있다.

이 근거는 사용자 출력이며 서버의 원본 artifact 전체를 독립 검사한 것은 아니다.
단일 model seed의 validation 결과이므로 test·유의성·보편적 동등성 주장은 하지 않는다.

### 과거 arxiv-only v2/v3 보고와 현재 확대된 전체 실행 필요

[Conductance v2](CONDUCTANCE_V2.md)는 canonical 물리 엣지마다
층별 alpha를 두고 `c_e=exp(alpha_e)`를 직접 학습한다. Alpha=0에서 C=1로 시작하며
C 생성 MLP·고유분해 없이 implicit diagonal C와 기존 node-degree 전파를 사용한다.
직접 alpha의 WD는 0, 나머지 파라미터의 WD는 0.0005다. 같은 초기 상태의 direct/fixed를
별도 새 run에서 학습하므로 기존 MLP 결과를 v2의 점수로 가져오지 않는다.

현재 기본은 **Cora/CiteSeer/PubMed/ogbn-arxiv × direct_c/fixed_c × seed 0 = 8개 CUDA
학습**이다. Unseen 독립 그래프에 엣지 파라미터를 전달하는 규칙이 없으므로 PPI는 V2에서
N/A다. 네 데이터는 모두 full-batch다.
Chunk 연산은 전체 엣지의 forward/backward를 처리하는 메모리 제어이며 GraphSAGE식 sampling이 아니다.
기존 공유 MLP 설계도 유효하다. V2는 별도 직접 파라미터화 가설이다. 사용자 보고상 과거
arxiv-only runner는 `passed`했지만 GPU 성능 수치와 전체 artifact는 수령하지 않았다.

[Conductance v3](CONDUCTANCE_V3.md)는 공유 MLP가 방향 불변인 엣지
특징에서 score를 생성한다. 그래프별 중심화 및 mean(C)=1 상대화, 학습 gamma/tau,
학습 alpha와 symmetric normalization을 조합한다. 현재는 v1의 5개 데이터/seed 0을 사용하고
각 dataset에서 C=1 모델도 새로 학습한다. V3의 fixed C도 alpha는 학습한다.
정규화·파라미터화·optimizer가 여러 가지 바뀌므로 버전 간 점수 차이는 단일 요인 효과가 아니다.
제안의 dmax/작은 rho 지적은 과거 global-max v1에 해당하며 현재 v2의 오류라고 기록하지 않는다.
Gamma 값만으로 C의 중요도를 판정하거나 full-graph chunking을 neighbor sampling으로
설명하지 않는다. PPI는 공식 20/2/2 graph split, whole-graph minibatch 2,
BCEWithLogitsLoss와 `logit > 0`의 global node-label micro-F1을 쓰는 inductive 실험이다.
Test graph는 계산·선택·진단에는 미사용이고 full cache 무결성 검사에만 포함된다.
Multi-head·implicit solve는 이번 v3 실행 범위에 없다. 사용자 보고상 과거 arxiv-only runner도
`passed`했지만 GPU 성능 수치와 전체 artifact는 수령하지 않았다.

[V4 통합 문서](CONDUCTANCE_V4.md)의 실험은 v3의 `C(H)`를 학습 가능한
graph operator/metric 경로로 유지하면서, 이웃 feature message에 층별 `W`를 적용한다.
비고립 노드의 식은 `(1-alpha)H + alpha P_C(HW)`이고 고립 노드는 `H`를 유지한다.
`W=I` 조건은 v3 전파와 일치한다. 현재 기본은 v1의 5개 데이터/seed 0에서 고정/상대 C ×
identity/학습 W의 20개 fresh training이며 모든 cell에서 alpha를 학습한다. 보고서는 `C|W off`, `C|W on`,
`W|C fixed`, `W|C relative`, interaction을 V4 내부에서만 계산한다. V3 checkpoint/점수는
재사용하지 않고, 선택 checkpoint의 C/W 제거 개입을 fresh-training 차이로 해석하지 않는다.
V4의 PPI도 V3와 같은 20/2/2, whole-graph minibatch 2, BCEWithLogitsLoss, `logit > 0` global
node-label micro-F1과 test 미평가/cache 무결성 검사 경계를 사용한다.
Mean-C/C=1 별도 CUDA forward의 logit 차이와 선언된 allclose 허용오차 판정은 informational
non-gating이다. 사용자 보고상 첫 arm만 200 epochs·child exit 0까지 완료된 뒤 구 report
numeric gate에서 중단됐고 나머지 세 arm은 pending이다. 이 과거 arxiv partial 실행을 2×2
결과로 재사용하지 않으며 새 run에서 전체 20개 arm을 모두 fresh 완료해야 한다.

### 수령한 2×2 GPU 재학습: 정규화 효과와 C 학습은 별개

Run `gat-factorial-seed0-v1`, 소스 `43afd632b97a4285dfeae26847b4f12a8fd1a1e4`,
model seed 0. NVIDIA RTX A6000, Python 3.11.16, Torch 2.7.1+cu118, PyG 2.7.0,
Linux glibc 2.35에서 여덟 fresh training과 최종 비교표가 모두 `passed`다.
Train 정답으로 학습하고 validation으로 선택했으며 **test는 평가하지 않았다**.

| 조건 | PPI validation micro-F1 (%) | arxiv validation accuracy (%) |
|---|---:|---:|
| baseline | 48.986770 | 50.927883 |
| gate_no_wd | 49.378028 | 50.565451 |
| node_degree | 52.465469 | 68.317723 |
| node_degree_gate_no_wd | 50.340520 | 67.995566 |

두 데이터 모두 **node_degree + gate WD 0.0005**가 최고다. 기존 정규화 대비 PPI
+3.478699pp, arxiv +17.389840pp다. WD를 제거하면 C 변동은 커지지만 성능은 일관되게
좋아지지 않았다. 특히 arxiv 최고 조건의 두 층 C CV는 0과 약 0.00948이므로
이 결과만으로 학습 C의 기여를 입증할 수 없다. PPI에는 비상수 C가 남아 있어
반대로 C가 항상 불필요하다는 결론도 성립하지 않는다.

정확한 epoch·5개 대비·층별 C/rho/전파량·해석 경계와 다음 분석은
아래 Conductance 실험 정리 절과 [전체 인수인계](HANDOFF.md)에 보존했다.
단일 seed의 탐색적 validation 결과이며 유의성·일반적 최적값·SOTA 주장은 하지 않는다.

### 수령한 확장 검사: 기존 checkpoint에 대한 개입

학습 run `paper-20260830T150244764889Z`, model seed 0. 진단 실행 소스는 `5e801c3`이며
보고서 폴더 suffix는 `20260831T120740120251Z`다. 사용자 첨부 로그 SHA-256:
`CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F`.

| 데이터 (validation) | learned C | mean C | shuffled C | graph off |
|---|---:|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.646000 | 0.646000 | 0.632000 |
| CiteSeer accuracy | 0.644000 | 0.642000 | 0.642000 | 0.626000 |
| PubMed accuracy | 0.724000 | 0.726000 | 0.726000 | 0.718000 |
| PPI micro-F1 | 0.487508 | 0.487508 | 0.487508 | 0.452144 |
| arxiv accuracy | 0.509279 | 0.509279 | 0.509279 | 0.508876 |

다섯 dataset 모두 완료, 최종 `Diagnostic status: passed`다. Validation 재검증 오차는
약 0~5e-8이다. 각 조건은 같은 checkpoint에 대한 개입이며 재학습한 baseline의 성능이 아니다.
PPI/arxiv는 두 층 모두 관찰한 C가 상수이고 mean/shuffle의 prediction flip이 0이다.
PPI에서는 graph-off 시 3.5364 percentage points 하락해 연결 구조의 기여와 C 차별화 실패가
구분된다. arxiv에서는 0.0403 points 하락하며 rho 중앙값 약 .000433으로 전파 영향이 매우 작다.

Gradient 검사는 eval/dropout-off, PPI는 첫 train batch 하나다. Gate 파라미터 norm과 task
gradient가 극소지만 학습 당시 Adam update 이력을 복원한 것은 아니므로 WD 원인 확정은 아니다.
기존 full audit의 극소 ratio는 분모 하한 1e-12 적용 여부를 확인해야 한다. 새 2×2의 train-mode
관찰은 raw task/decay norm을 따로 저장하고 0분모 비율만 null로 기록한다.

## 2. 사용자가 제공한 5-seed test 집계

모든 항목의 model seeds는 `0,1,2,3,4`다. `±`는 seed 사이 **표본 표준편차**이며,
표준오차·신뢰구간·5-fold 교차검증이 아니다. 모델·데이터·평가 조건이 다른 트랙끼리
수치를 직접 차감해 기여도를 추정하지 않는다.

### Conductance GAT

Run: `paper-20260830T150244764889Z`.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| Cora | accuracy | 0.661600 ± 0.023639 |
| CiteSeer | accuracy | 0.626800 ± 0.007918 |
| PubMed | accuracy | 0.721200 ± 0.005630 |
| ogbn-arxiv | accuracy | 0.486089 ± 0.002813 |
| PPI | global micro-F1 | 0.500051 ± 0.004877 |

이 값은 실행·집계 결과이지 경쟁력 또는 novelty의 증명이 아니다. 외부 논문 표와
비교하려면 split·입력 전처리·모델 크기·학습 설정·모델 선택 절차를 별도로 확인해야 한다.

### Cycle PE v1

Run: `paper-20260831T015711388279Z`. 모델 키는 **`cycle_set`**이다.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| ZINC-12K | MAE, 낮을수록 좋음 | 0.189090 ± 0.016624 |
| Peptides-struct | MAE, 낮을수록 좋음 | 0.259728 ± 0.002816 |

이것은 cycle 기저를 여섯 통계로 요약하는 v1의 결과다. 좌영공간 기저벡터 전체를 입력하는
**`cycle_basis_v2` 결과가 아니다.** `schema_version=2` 같은 저장 형식 버전과 모델 v2도
구분한다. 같은 backbone의 PE 제외 ablation이 없으므로 이 표만으로 PE의 순수 효과를
분리할 수 없다. 두 데이터셋이 회귀 과제이므로 MAE 사용 자체는 맞다.

### Tree Augmentation

Run: `paper-20260831T060149709584Z`.

| 데이터·평가 조건 | 지표 | Fixed BFS | Multi-chart |
|---|---|---:|---:|
| CSL / seen family | accuracy | 0.400000 ± 0.028260 | 0.768333 ± 0.029404 |
| CSL / unseen family | accuracy | 0.059167 ± 0.009501 | 0.137500 ± 0.002946 |
| ZINC / seen family | MAE | 0.730009 ± 0.081641 | 0.753210 ± 0.156958 |
| ZINC / unseen family | MAE | 0.727205 ± 0.083463 | 0.749209 ± 0.155026 |

- `seen/unseen family`는 원본 그래프 종류가 아니라 **spanning-tree 생성 방식**이다.
  Fixed는 root-0 BFS, multi는 random-root BFS/DFS로 학습한다. 같은 test 그래프에
  fresh random-root BFS(seen)와 학습에서 제외한 Wilson UST(unseen)를 적용한다.
- CSL은 단일 고정 label-stratified 90/30/30 분할이다. 5개 모델 seed는 5-fold가 아니다.
  정확도는 chart별 정확도의 평균이며 여러 chart의 logits을 앙상블한 값이 아니다.
- CSL seen에서는 큰 개선이 있지만 unseen의 절대 성능은 낮다. unseen prediction flip rate도
  0.200000 → 0.319167로 증가하여 모든 chart 변화에 더 안정적이라고 주장할 수 없다.
- ZINC의 평균 MAE는 두 조건 모두 개선되지 않았다. Chart prediction std는 seen에서
  0.053970 → 0.041503, unseen에서 0.053139 → 0.041492로 줄었지만 이는 **한 그래프의
  chart 변경에 따른 흔들림**이지 model-seed 사이 성능 변동 감소나 MAE 개선이 아니다.
- `rounded_exact_vector_accuracy=0`은 정수 count용 일치 지표를 연속값 ZINC에도 노출한
  부적절한 보조 지표다. 코드가 반올림한 예측과 연속 정답의 완전 일치를 검사하므로,
  이를 회귀 학습 실패 또는 정확도 0%로 해석하지 않는다. 아직 코드에서는 제거하지 않았다.
- 현재 Tree 기본 구현은 고정 800 optimizer updates 후 모델을 평가하며 validation-best
  checkpoint 선택을 하지 않는다. 이 내부 fixed-vs-multi 실험을 표준 논문 표와 동일한
  재현이라고 제시하지 않는다. 표의 차이에 대한 paired 유의성 검정은 수행하지 않았다.

## 3. Conductance의 실제 GPU checkpoint 진단

사용자가 `ebf8cd1`을 pull한 뒤 아래 명령을 실행했고,
`Diagnostic status: passed (stdout only)` 출력을 제공했다.

```bash
bash scripts/diagnose_conductance.sh --run-id paper-20260830T150244764889Z --ablate-graph
```

대상은 **당시 기본값**인 model seed 0, Cora/PPI/ogbn-arxiv다. FP32 추론이며 AMP/TF32는 껐다.
이후 CLI 기본 데이터 목록에 CiteSeer/PubMed를 추가했지만 이 과거 로그에는 포함되지 않는다.
새로 계산한 지표는 train/validation뿐이고, test는 기존 저장값만 출력했다.
원래 validation과 재계산값의 차이는 약 `0~5e-8`이다. 소스 불일치 경고는 제공된 출력에 없다.
이 로그는 실제 GPU 추론 완료의 근거지만 전체 seed의 gate 상태나 GPU 가속의 근거는 아니다.

저장된 학습 설정은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005,
weight decay 0.0005, 최대 200 epochs, patience 50, PPI batch size 2다.

### 선택된 checkpoint의 성능과 학습 기록

| 데이터 | Best epoch / 실행 epoch | Eval train 지표 | Eval validation 지표 | Eval train / validation loss |
|---|---:|---:|---:|---:|
| Cora | 23 / 73 | accuracy 1.000000 | accuracy 0.658000 | CE 0.083452 / 1.163915 |
| PPI | 69 / 119 | micro-F1 0.496796 | micro-F1 0.487508 | BCE 0.541717 / 0.536941 |
| ogbn-arxiv | 199 / 200 | accuracy 0.495695 | accuracy 0.509279 | CE 1.879628 / 1.826401 |

Train-mode loss의 first → selected → last는 Cora `2.118194 → 0.720689 → 0.172052`,
PPI `0.699084 → 0.541241 → 0.541466`, arxiv `3.944757 → 2.252732 → 2.251006`이다.
진단의 train도 **dropout을 끈 eval 추론**이다. 학습 중 dropout을 켜고 기록한 loss와
시점·모드가 다르므로 두 값을 직접 비교해 checkpoint 오류라고 판단하지 않는다.

- Cora: 선택된 모델도 train 정확도 100%인 반면 validation은 65.8%다. 훈련 노드 적합은
  충분하며 큰 일반화 차이가 확인됐다. 단순 epoch 부족이라고 보기 어렵다.
- PPI: train에서도 F1이 낮아 현재 표현·정규화·최적화의 적합 부족을 의심한다. validation의
  예측 양성 비율은 0.202331, 실제 양성 비율은 0.294568이다. 제공된 micro-F1과 양성 비율에서
  계산한 precision은 약 0.598629, recall은 약 0.411182다. 원 로그가 직접 출력한 지표와
  이 후처리 계산값을 구분한다. 낮은 recall의 원인은 아직 확정하지 않았다.
- arxiv: 마지막 loss가 최저이고 best epoch가 199/200이라 추가 학습 여지가 있다.
  그러나 아래의 극도로 약한 이웃 전달 문제도 병존한다.
- Cora `23+50=73`, PPI `69+50=119`로 patience 설정과 종료 시점은 일치한다.

### 층별 엣지 가중치와 이웃 혼합량

층 번호는 코드의 `layer 0/1`을 따른다. Cora/arxiv 통계는 전체 transductive 그래프,
PPI 표는 **validation 그래프 두 개**의 node/edge-pooled 통계다. 차수 최대값은
그래프별로 계산한 뒤 통계를 모았다. `rho`는 0~1의 비율이며 퍼센트가 아니다.

| 데이터 / 층 | C 평균 | C의 CV | rho 중앙값 | rho < 0.01 노드 비율 | 전파 상대 변화량 |
|---|---:|---:|---:|---:|---:|
| Cora / 0 | 0.6984 | 6.623e-7 | 0.01696 | 17.91% | 0.02989 |
| Cora / 1 | 2.266 | 0.7224 | 0.02779 | 14.96% | 0.05120 |
| PPI / 0 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09871 |
| PPI / 1 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09498 |
| arxiv / 0 | 0.6933 | 0 | 0.0004331 | 99.35% | 0.003041 |
| arxiv / 1 | 0.6933 | 0 | 0.0004330 | 99.35% | 0.002101 |

CV는 `std(C)/mean(C)`이고, 전파 상대 변화량은 LayerNorm/ELU 이전
`||H_after_conv - H_before_conv|| / ||H_before_conv||`이다. PPI train 그래프 20개에서도
두 층의 C CV는 0이며 rho 중앙값은 0.03308이다.

PPI/arxiv의 **관측한 FP32 eval 입력**에서 C가 상수인 현상이 확인됐다. 표시는 작은 양수도
지수 표기로 출력하므로 `CV=0`은 단순 표시 반올림 설명으로 지울 수 없다. C는 FP32로
chunk 재계산하고 분산은 float64로 집계했다. 전체-batch GEMM과의 bitwise 동일성이나
모든 가능한 입력에서 함수가 상수라는 결론까지 보장하지는 않는다.

Cora의 layer 1에는 비상수 C가 있으므로 구현이 항상 C를 상수로 강제한다는 설명은 틀린다.
또 진단은 학습 중 C 궤적을 기록한 것이 아니므로 학습 내내 C가 고정돼 있었다고 할 수 없다.

이 과거 checkpoint와 변경하지 않은 기본 benchmark의 전파식은 다음과 같다.

\[
H'=H-\frac{0.95}{d_{\max}^C}B^\top C B H,\qquad
\rho_i=0.95\frac{d_i^C}{d_{\max}^C}.
\]

`C=cI`이면 공통 c가 상쇄되어 `H'=H-(0.95/d_max)L_unweighted H`가 된다.
따라서 관측한 PPI/arxiv 입력에서는 적응적 엣지 가중치가 아니라 동일 가중치의 전파로
작동한다. arxiv rho 중앙값 `0.0004331`은 이웃 총가중치 **0.04331%**다.
이는 안정성용 스텝 선택의 효과이며 `L=B^T B`라는 라플라시안 정의 자체의 필수 조건이 아니다.

### 같은 checkpoint에서 전파만 우회한 validation 결과

| 데이터 / 지표 | 원래 값 | 전파 우회 값 | 우회 − 원래 |
|---|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.632000 | -0.026000 = -2.6000%p |
| PPI micro-F1 | 0.487508 | 0.452144 | -0.035364 |
| arxiv accuracy | 0.509279 | 0.508876 | -0.000403 ≈ -0.04027%p |

전파 우회 값은 원 로그의 원래 지표와 delta를 합산했다. arxiv의 차이는 validation
29,799개 노드에서 정답 수 순감소 12개에 해당하며, 예측이 12개만 바뀌었다는 뜻은 아니다.
우회는 encoder/LN/ELU/decoder를 남기는 추론 개입이다. 별도 MLP를 재학습한 baseline이나
저성능 원인을 단독으로 증명하는 실험이 아니다. Cora/PPI에는 분명한 지표 하락이 있으므로
모든 데이터에서 그래프 연산을 전혀 사용하지 않는다고 말하면 안 된다.

## 4. 미확정 원인과 다음 검증

기존 checkpoint에서 관측된 상수 C와 약한 전달량은 사실이다. 이후 2×2 새 학습에서
node-degree 정규화가 개선을 이끈 결과를 확보했지만, 모든 데이터·seed에 대한 원인 설명이나
학습 C의 필요성까지 확정된 것은 아니다.
`softplus(0)+1e-5 ≈ 0.693157`이므로 PPI의 C 평균은 softplus 이전 gate raw logit이
0 부근인 상황과 맞는다.
이것만으로 모든 gate 파라미터가 0이라고 증명되지는 않는다. `softplus'(0)=0.5`이므로
이를 softplus의 출력 포화라고 설명하는 것도 부정확하다.

확장 audit에는 gate 입력·raw logit·gradient 검사가 있고, 2×2 학습에는 실제 train-mode
첫 batch의 task gradient와 decay 관찰값이 있다. Gate WD 제거 후 선택된 C의 변동이
커졌다는 결과와 WD 제거가 성능을 높인다는 주장은 분리한다.

`learned_c`/`fixed_c=1`의 네 새 학습과 **그 새 run의 learned checkpoint** 평균-C 개입은
모두 사용자 GPU 보고서를 수령했다. PPI의 현재 checkpoint 의존도가 크다는 결과와
fresh-training 이득을 관측하지 못했다는 결과를 함께 보존한다. 이전 2×2 `node_degree` 검사도
지원하지만 다른 source run의 결과로 분리한다.

사용자 보고상 [직접 C v2](CONDUCTANCE_V2.md)와
[상대 C v3](CONDUCTANCE_V3.md)의 **과거 arxiv-only runner**는 `passed`다. V2는 같은 그래프에
묶인 direct/fixed C이고, v3는 공유 상대-C 생성기의 relative/fixed C이며 fixed v3도 alpha는
학습한다. 둘 다 기존 MLP의 수학 오류 수정이 아니다. 각 버전 내부 비교를 먼저 보고,
파라미터화·정규화·전파 강도·optimizer가 함께 다른 버전 간 차이를 단일 요인으로 해석하지 않는다.
다만 v2/v3 성능 수치와 전체 artifact는 미수령이고, 과거 arxiv-only V4는 partial 첫 arm 뒤
중단됐다. 현재 확대된 V2/V3/V4 8/10/20-job 정식 결과는 없다.

노드별 정규화는 기존 대칭성·보존성의 의미를 바꾸는 실험이므로 단순 속도 최적화나
버그 수정으로 부르지 않는다. 기존 기본 benchmark는 유지한다. 다른 model seed의
일반화, 확대된 v2/v3/V4 전체 행렬과 Cycle PE v2의 수치·전체 artifact 독립 검증,
GPU 가속 실측은 여전히 별도 검증 대상이다.

## 5. 근거와 검증 범위

| 근거 | 식별자 / SHA-256 |
|---|---|
| 사용자 제공 5-seed 집계 출력 | 첨부 `d4acb1eb-bd9d-40ef-af24-e5f7ba34f138`; `CEF76E8494C462E8302AF2811CCCD19BBB6D8DC8266DB852866237ED95DD5CEC` |
| 사용자 제공 GPU 진단 stdout | 첨부 `5db2e997-c8ab-495e-b762-c32fa620c02c`; `C0E89FC76A438D1707FE90C889923390FDF8277F05780B2811FF4D444DD01A21` |
| 사용자 제공 5e801c3 full-audit stdout | 첨부 `c4abbad1-654a-4f5e-a774-f84f7e88e4dd`; `CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F` |
| 사용자 제공 43afd63 2×2 GPU 학습·비교 stdout | 첨부 `20b4a93d-06ed-4cff-9fe5-530eacf39766`; `2C78D02BB210BF00865AB7207DF651B02B2081EE4FAE6E8A6A83665A5D331161` |
| 사용자 제공 C-learning GPU 비교 | 2026-09-01 inline 보고서, `gat-c-learning-seed0-v1`; 별도 첨부 파일/SHA 없음, 실행 revision 미확인 |
| 사용자 제공 C-learning 평균-C GPU 검사 | 2026-09-01 inline terminal 출력, revision 표시 `8f6b4da`, `gat-c-learning-seed0-v1`의 각 데이터셋 `learned_c`; 별도 첨부 파일/SHA 없음 |
| 사용자 제공 A6000 V5·Cycle V2 r1 실패 stdout | 첨부 `606865d2-9e2f-4818-a7dd-b4599ab165ae`; `D6394C813EDB52DD2CE6746B531973A750A1F0AE807EC5A3410B938FAFF6F93E` |
| 사용자 제공 A6000 V5·Cycle V2 r2 old-source 재현 stdout | 첨부 `bd63fc9a-60da-4daf-9ab9-da49db7cbbe1`; `F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6` |

이 hash는 **제공된 텍스트 파일의 hash**이며 서버의 checkpoint/원본 데이터 hash가 아니다.
개인 서버 계정·호스트 경로와 원본 로그 전체는 이 문서에 복제하지 않았다.

확장 검사 구현 전 문서 갱신의 로컬 회귀는 619 passed / 63 skipped (21.84 s, exit 0),
당시 진단 전용은 42 passed였다. Ruff/diff 및 당시 문서 로컬 링크 34개 검사도 통과했다.
당시 확장 검사 결과는 handoff의 해당 역사 검증 항목을 따른다.
단일 seed·확장 진단 구현 후 전체 회귀는 **680 passed / 63 skipped**, 진단 전용은
**89 passed**다. 이는 당시 로컬 단위 검증이며, 이후 수령한 실제 GPU full-audit 로그는 위에
별도로 기록했다. 후속 2×2 구현 후 전체 검사는 **794 passed / 64 skipped** (31.83 s, exit 0),
Ruff 통과다. 기존 생략 사유에 Windows 실제 symlink 권한 1개가 추가됐으며 차단 로직은 별도
mock 검사로 확인했다. 이 수치는 2×2 코드 구현 당시의 로컬 검사이며, 이후 수령한
43afd63의 실제 GPU 재학습·성능 비교는 위에 별도 기록했다.
기존 게시 학습 코드 `a64c235`와의 진단 호환성도 메모리 로딩을 통한 42개 단위 검사로 확인했다.
생략된 63개는 Linux/Bash 전용 62개와 로컬 PyG 미설치 1개다. 이번에도 Windows faulthandler의
`access violation` 메시지가 있었으나 pytest는 위 결과와 exit 0까지 실행됐다.
이 호스트 경고를 Linux 성공의 근거로 해석하지 않는다. 로컬 단위 검사는 GPU 학습
또는 가속 배수의 검증을 대신하지 않는다. 현재 소스 스냅샷 checksum은
[HANDOFF.md](HANDOFF.md)의 코드 스냅샷 항목을 따른다.
