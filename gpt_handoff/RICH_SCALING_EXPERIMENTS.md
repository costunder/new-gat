# Conductance V1–V5·Cycle PE V1/V2·Tree reference/large 전체 scaling

과거 `64/128 × 2/4층` grid는 mechanism probe로만 남기고 현재 scaling 기본 계획에서는
폐기한다. 새 계획은 파라미터를 버전 사이에 강제로 일치시키지 않으며, 각 구조가 실제 연구급
capacity에서 성능을 낼 수 있는지를 본다. 기본 seed는 시간 제약 때문에 0 하나다.

이 문서의 전체 matrix는 실험 계약 설명이며 완료한 V1–V4/Cycle V1/Tree를 다시 돌리라는
안내가 아니다. 이번 QR-free 교체의 새 Cycle V2만 실행하는 명령은 마지막 절과
[CYCLE_PE_V2.md](CYCLE_PE_V2.md)를 따른다. 구 source/checkpoint의 strict resume 검사는 우회하지 않는다.

## ad041e2 실패 실행 격리와 수정판 재실행

`v5-cycle-se-pe-a6000-gpu3-seed0-v1`은 첫 V5 fixed-C 학습 후 throughput 집계 오류,
Cycle 8개는 CPU IPC mmap 누적으로 학습 전 실패했다. 이번 수정은 기록/전달 방식 수정이지
ad041e2 대비 V5 또는 SE/PE 모델 구조 교체가 아니다. 과거 214265c의 unused fixed-C scorer와
공통 초기화 계약은 달랐으므로 그 더 오래된 결과까지 동일 모델 실행으로 묶지 않는다.

소스 무결성 계약은 유지한다. 이전 실패 run의 hash/checkpoint/manifest를 고쳐 새 run으로
가장하지 않는다. 첫 fixed-C 54-epoch 결과는 역사적 partial artifact로 남고, 아래 새 source
run은 별도 학습이다. 이전 결과를 반드시 지워야 새 모델이 작동하는 것은 아니며, **격리는
선택 사항**이다. 새 run ID만으로도 기존 결과와 분리된다.

사용자가 이전 V5 결과를 현재 결과 폴더에서 치우려는 경우, 서버에서 해당 run이 완전히
끝난 뒤 저장소 루트의 `new-gat` 환경에서 먼저 **읽기 전용 계획**을 확인한다.

```bash
python -B scripts/archive_failed_rich_run.py --run-id v5-cycle-se-pe-a6000-gpu3-seed0-v1
```

출력에 나오는 원본은 아래 **지정한 실행의 세 폴더만**이어야 한다. 참조되지 않은 경로나
다른 run, 원본 dataset/cache, V1–V4/Cycle V1/Tree를 자동 탐색해 이동하지 않는다.

- `results/rich_scaling/v5-cycle-se-pe-a6000-gpu3-seed0-v1`
- `results/conductance_gat/scaling/v5-cycle-se-pe-a6000-gpu3-seed0-v1-conductance`
- `results/cycle_pe/scaling/v5-cycle-se-pe-a6000-gpu3-seed0-v1-cycle`

계획이 맞을 때만 별도로 적용한다. 이 명령은 **영구 삭제가 아니라 같은 filesystem의
rename**이다. 적용 시 Linux `/proc`에서 현재 사용자 소유 활성 작업을 확인하며, 동일
run/output을 사용하는 작업이나 불명확한 상태·링크·경로 이탈·기존 대상이 있으면 거부한다.
같은 run을 다른 터미널에서 새로 실행하면서 동시에 격리하지 않는다.

```bash
python -B scripts/archive_failed_rich_run.py --run-id v5-cycle-se-pe-a6000-gpu3-seed0-v1 --apply
```

원본 manifest/checkpoint 내용은 고치지 않고
`results/_archived_failed_runs/<run-id>-<UTC>-<unique>/`에 보관하며 이동 기록에
원래 위치와 격리 위치를 남긴다. 중간 이동 실패도 숨기지 않으며 기록된 경로로 복구할 수 있다.
이미 passed인 전체 실행, legacy version이 섞인 실행, 실행 중인 작업은 이 도구의 대상이 아니다.
이 문서는 사용자의 서버에서 실제로 격리를 실행했다는 보고가 아니다.

수정판 source가 게시되고 서버에서 동기화된 뒤, **새 run ID**로 V5와 Cycle V2 SE/PE만
실행하는 GPU 3/A6000 명령은 다음과 같다. V5 20학습 + Cycle 8학습 = **28학습**이며
Cycle의 encoding×dataset별 선택 checkpoint test 평가4회는 추가 학습이 아니다.
이후 동일 source/config/run ID의 재실행만 epoch checkpoint부터 resume한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_rich_scaling.py \
  --run-id v5-cycle-se-pe-a6000-gpu3-seed0-v2 \
  --tracks conductance cycle \
  --conductance-versions v5 --cycle-versions v2 \
  --cycle-v2-encodings se pe \
  --profiles reference large --model-seeds 0 \
  --hardware-profile a6000-48gb --min-free-gb 40 \
  --cycle-v2-basis-backend dfs_fundamental --device cuda:0
```

### CPU 대량 IPC 회귀와 실제 GPU 학습의 구분

기존 7천 그래프 실패 구간을 넘는 1만 합성 그래프의 실제 process-pool 준비와 cache 검증은
별도 opt-in debug 검사로 수행한다. Linux에서는 `/proc/self/maps` 및 fd 수가 그래프 수에
비례해 누적되지 않는지도 검사한다. Windows 로컬 검사에서는 이 Linux 관측을 unavailable로
기록하며 CUDA/실제 데이터 성능을 검증한 것처럼 보고하지 않는다.

```bash
CYCLE_V2_IPC_STRESS_GRAPHS=10000 python -B -m pytest -q -s research/cycle_pe/tests/test_v2_data.py::test_debug_large_synthetic_preparation_and_cache_ipc_have_bounded_os_handles
```

이 환경변수는 위 테스트 전용이며 실제 dataset/worker/batch 설정을 변경하지 않는다.

## Architecture profiles

### Conductance V1–V5

| Profile | Hidden | Layers | V5 heads | V5 FFN | Dropout |
|---|---:|---:|---:|---:|---:|
| `reference` | 256 | 8 | 8 | 4 | 0.2 |
| `large` | 384 | 12 | 8 | 4 | 0.2 |

V1–V4에는 지원하는 hidden/layers/dropout만 전달한다. V5는 heads/FFN과 activation
checkpointing을 추가로 쓴다. V2 direct-C만 PPI가 N/A이고 나머지 버전은 V1과 같은 다섯
dataset을 사용한다. 한 profile/seed당 V1 5, V2 8, V3 10, V4 20, V5 10으로 53회이며
두 profile에서는 106 fresh validation-only trainings다.

V5의 두 arm은 architecture, seed와 초기화는 같지만 학습 recipe까지 같은 단일-factor
ablation은 아니다. fixed-C는 C coordinate 예산을 spatial 그룹에 쓰는 강한 baseline이고,
dynamic-C는 그 예산을 C calibration/alternation에 쓴다. 따라서 phase별
`effective_optimizer_steps_by_group`이 다른 end-to-end recipe 비교로 해석한다.

### Cycle PE V1/V2

| Profile | ZINC-12K | Peptides-struct |
|---|---|---|
| `reference` | hidden/PE/layers 128/64/10 | 256/64/6 |
| `large` | 192/96/12 | 320/96/8 |

V2는 FFN×4, dropout 0.1, layer scale 0.1을 사용한다. V1은 4학습,
V2는 SE/PE 두 조건×두 datasets×두 profiles로 8학습, Cycle 전체는 12학습이다.
V2 ID는 SE `cycle_dfs_se_v2`, PE `cycle_dfs_relative_pe_v2`다.
PE는 동일 SE에 cycle 내부 상대 위치 residual만 추가하며 추가 trainable parameter는 없다.
Backbone 크기는 유지했고 두 조건의 기본 ZINC 모델은 각각 7,262,785 parameters다.
과거 raw/projector/이름만 PE였던 sparse ID의 결과와 checkpoint는 별도로 보존한다.

### Tree V1/V2

| Profile | Hidden | Message layers |
|---|---:|---:|
| `reference` | 128 | 8 |
| `large` | 256 | 12 |

CSL/ZINC × fixed-BFS/multi-chart × 두 profiles = 8 model trainings다.

## 전체 계획

- Conductance: 106 child/model trainings.
- Cycle: 12 child/model trainings.
- Tree: 4 child runs, 각 두 모델 = 8 model trainings.
- 합계: **122 child runs, 126 fresh model trainings**.
- profile 선택 후보는 train/validation만 사용한다. 기존 Cycle/Tree 계약의 selected-checkpoint
  test-only 단계는 validation 선택 이후 별도 평가이며 재학습하지 않는다.

## 실행 장치와 동시성 계약

최상위 `run_rich_scaling.py`의 `--device`와 `--devices`는 상호 배타적이다.

- `--device cuda:0`처럼 장치 하나를 지정하면 Conductance, Cycle, Tree **track runner를
  순차 실행**한다. 한 track의 전체 child matrix가 끝난 뒤 다음 track을 시작하며, 같은 GPU에
  여러 최상위 track을 겹쳐 실행하지 않는다.
- `--devices cuda:0 cuda:1 ...`처럼 서로 다른 indexed CUDA 장치를 명시한 경우에만 독립
  track들을 병렬 실행한다. Track은 요청 순서대로 장치에 round-robin 배정되고, 장치 수보다
  track 수가 많으면 bounded wave로 나뉜다. 한 wave에서 동일 GPU에 배정되는 최상위 track은
  하나뿐이므로 이 계층의 same-GPU concurrency는 1이다.
- `--devices`의 중복 장치와 다중 장치 목록의 unindexed `cuda`는 거부한다. 각 track child는
  배정된 단일 장치만 사용하며 DDP로 한 모델을 여러 GPU에 복제하는 구현은 아니다.
- 위 제한은 **최상위 cross-track 계층**의 계약이다. A6000 Tree runner 내부의
  `job_concurrency=2`는 disjoint output을 가진 두 candidate/selected-test child를 같은 GPU에서
  실행하는 별도 명시적 profile 설정이다. 이를 최상위 same-GPU track 병렬화로 세지 않는다.

예를 들어 scheduler 또는 컨테이너가 물리 GPU 3, 4, 5를 노출했다면, 프로세스에서 보이는
논리 장치 0, 1, 2를 다음처럼 세 track에 배정한다. 실제 할당받지 않은 GPU를 목록에 넣지 않는다.

```bash
CUDA_VISIBLE_DEVICES=3,4,5 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-multigpu-seed0-v1 \
  --profiles reference large \
  --model-seeds 0 \
  --devices cuda:0 cuda:1 cuda:2 \
  --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --cycle-v2-basis-backend dfs_fundamental
```

GPU가 하나만 할당된 아래 예시들은 계속 `--device cuda:0`을 사용하며 최상위 track을 순차
실행한다.

다음은 과거에 사용한 물리 GPU 6의 10GB MIG slice를 그대로 지정하는 **portable 실행
예시**다. GPU 번호는 실제 할당에 맞춰 바꾸되 프로세스 내부 장치는 항상 `cuda:0`을 쓴다.

```bash
git pull --ff-only

CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-portable-gpu6-seed0-v1 \
  --profiles reference large \
  --model-seeds 0 \
  --device cuda:0 \
  --hardware-profile portable \
  --cycle-v2-basis-backend dfs_fundamental
```

별도 shell fail-fast 설정 없이 각 Python runner가 subprocess return code, artifact, source
hash와 manifest를 직접 검증한다. 같은 명령/run-id를 다시 실행하면 완료된 child를 검증해
건너뛰며, V5 Conductance와 새 Cycle V2는 `last.pt`가 있으면 epoch 단위로 이어간다.
저장 model/optimizer/RNG에서 epoch-boundary continuation을 수행하지만 CUDA bitwise 재현을
주장하지 않는다. 다른 config/source/job matrix를 같은 run-id와 섞으면 중단한다.

## RTX A6000 48GB throughput profile

GPU 3 한 장이 실제로 할당된 서버에서는 다음처럼 물리 GPU 3을 프로세스 안의 `cuda:0`으로
매핑한다. `a6000-48gb`의 공통 장치 계약은 **보이는 VRAM 40GiB 이상과 compute capability
8.0 이상**이다. 아래 통합 명령은 여기에 `--min-free-gb 40`을 명시해 세 트랙의 일반
preflight에서 시작 시 free VRAM도 40GiB 이상 요구한다. profile 자체의 내장 free-memory
계약은 Conductance와 Tree가 32GiB 이상이고, Cycle은 전달받은 `--min-free-gb`와 V2의
worst-case pre-epoch forward/backward capacity probe를 사용한다. 어느 경로도 작은 MIG 장치로
자동 fallback하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-a6000-gpu3-seed0-v1 \
  --profiles reference large \
  --model-seeds 0 \
  --device cuda:0 \
  --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --cycle-v2-basis-backend dfs_fundamental
```

최상위 `--cycle-v2-basis-backend`는 Cycle child의 V2에 전달되고 통합 manifest와 재개
configuration에 결속된다. 유일한 기본값은 `dfs_fundamental`이다. DFS forest와 parent-path로
만든 전체 기저를 sparse block-diagonal edge→cycle→edge PE에 사용한다. 준비/cache/forward에
QR/SVD/Gram inverse가 없으며 `thin_q`와 이전 rank/column tuning 옵션은 거부된다.
DFS 탐색 O(V+E)와 기저 출력 O(nnz Z), sparse 집계 O(nnz Z*d)를 구분한다. DFS tree 선택에
의존하므로 일반 Z→ZR 불변성이나 graph 크기에 대한 엄밀한 전체 선형시간을 주장하지 않는다.

### 완료된 구 Conductance scaling을 다시 돌리지 않는 실행

이미 `base/wide/deep/large`에서 Conductance V1–V4 172회를 완료한 경우 그 결과를 별도
artifact로 보존하고, 현재 통합 실행에서는 Conductance V5만 선택할 수 있다. Cycle V1/V2와
Tree까지 아직 남았다면 다음 명령은 **36 child runs / 40 fresh model trainings**만 계획한다.

```bash
CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_rich_scaling.py \
  --run-id remaining-v5-cycle-tree-a6000-gpu3-seed0-r1 \
  --tracks conductance cycle tree \
  --conductance-versions v5 \
  --cycle-versions v1 v2 \
  --profiles reference large \
  --model-seeds 0 \
  --device cuda:0 \
  --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --v5-beta-parameterization sigmoid \
  --v5-beta-initial 0.1 \
  --cycle-v2-basis-backend dfs_fundamental \
  --allow-download
```

V5와 폐기·재구현된 Cycle V2만 새로 실행할 때는 `--tracks conductance cycle
--conductance-versions v5 --cycle-versions v2`를 사용한다. reference/large와 seed 0에서
**28 child runs / 28 fresh model trainings**다. 기본 `--cycle-v2-encodings se pe`를
포함한 선택 버전·조건 목록은 통합 manifest와 재개
identity에 들어가므로, 중단 후에는 같은 run ID와 같은 목록을 그대로 사용한다.

architecture profile(`reference/large`)과 hardware profile(`portable/a6000-48gb`)은 서로 다른
축이다. A6000 profile의 실제 실행 차이는 다음과 같다.

| 트랙 | `portable` | `a6000-48gb` |
|---|---|---|
| Conductance V5 | FP32, TF32 off, block checkpoint on, dynamic-C score-chunk checkpoint on, edge chunk 65,536, arxiv seed-node batch 1,024, PPI whole-graph batch 2 | dense BF16 autocast·TF32, conductance geometry FP32, block checkpoint off, dynamic-C score-chunk checkpoint on, edge chunk 131,072, arxiv seed-node batch 2,048, PPI whole-graph batch 8, sample prefetch/pinned transfer |
| Conductance V1–V4 | 기존 FP32 계약. PPI는 V1/V3/V4 batch 2이고 V2는 PPI N/A | 같은 legacy FP32·batch 계약을 그대로 유지 |
| Cycle V1/V2 | 모든 dataset/profile batch 32, workers 4, prefetch 2, AMP off; V2 sparse COO block-diagonal 집계 | reference ZINC/Peptides batch 512/128, large 256/64, workers 8; V2 backbone BF16·sparse 집계 FP32, GradScaler off, prefetch 4; V1은 기존 AMP/loader 계약 유지 |
| Tree V1/V2 | batch 16, workers 4, prefetch 2, suite config의 FP16 AMP, child concurrency 1 | batch 64, workers 4, prefetch 2, 명시적 FP16 AMP, 독립 child concurrency 2 |

각 **track runner 내부**에서 Conductance와 Cycle child는 순차 실행한다. Tree만 서로 다른
output/log를 갖는 candidate와 selected-test child를 최대 2개 병렬화하며 coordinator 하나만 공용
manifest를 원자적으로 쓴다. 이 Tree 내부 동시성은 위의 최상위 `run_rich_scaling.py`가 서로 다른
GPU에 independent track을 배분하는 cross-track 동시성과 별개다. A6000 작업 순서는 큰 workload를
먼저 검증하도록 deterministic heavy-first다. 한 Tree child가 실패해도 같은 wave에서 통과한 peer
artifact는 보존되며 같은 run ID 재실행에서는 실패한 작업만 새 attempt 경로에서 실행한다.

이 설정은 메모리만 더 쓰는 동치 실행 스위치가 아니다. V5는 real sample/PPI batch와 numeric
execution이 바뀌고, Cycle은 batch와 AMP가 바뀌며, Tree는 800 optimizer updates마다 보는 graph
수가 달라진다. 따라서 fixed/dynamic C, Cycle V1/V2, Tree fixed/multi 비교는 같은 hardware
profile 안에서만 해석한다. 특히 legacy Conductance V1/V3/V4 PPI는 batch 2/FP32이고 V5 A6000
PPI는 batch 8/BF16이므로 V1–V5 PPI 차이는 descriptive scaling 결과일 뿐 단일 구조 요인의
인과 비교가 아니다. portable와 A6000 사이의 점수나 wall time 차이를 모델 또는 GPU 하나의
효과로 직접 해석하지 않는다.

## 자원 실측과 보고 범위

GPU가 없는 이 Windows 작업 공간에서는 서버 GPU utilization을 측정할 수 없다. 대신 실제 Linux
GPU child가 시작될 때 preflight와 학습 프로세스 내부 monitor가 그 서버의 값을 측정해 artifact에
기록한다. 로컬에서 GPU 수치를 추정하거나 0으로 채우지 않는다.

Preflight report는 선택 GPU의 이름, 논리 index, free/total VRAM, compute capability와 이름 기반
MIG 감지, 보이는 모든 GPU의 inventory, logical/affinity CPU 수, available RAM과 측정 방법,
`CUDA_VISIBLE_DEVICES` 및 scheduler resource 환경을 기록한다. 이는 **시작 시점의 availability
snapshot**이며 학습 전체의 utilization이나 처리량을 인증하지 않는다.

현재 Conductance V5, Cycle PE V1/V2와 Tree V1/V2 경로는 `RuntimeResourceMonitor`를 직접
연결한다. Monitor는 각 학습 또는 명시적 selected-test 실행 경계의 start/end와 기본 1초 주기
표본을 분리해 다음을 child 결과에 기록한다.

- GPU SM utilization, memory-controller utilization, CUDA allocator allocated/reserved bytes의
  sample count와 min/mean/max.
- 프로세스 RSS/HWM과 system available RAM의 sample count와 min/mean/max.
- 관측 wall time과 process CPU time으로 계산한 한 logical core 대비 평균 CPU 사용률 및 할당된
  CPU capacity 대비 평균 사용률.
- 호출자가 정한 학습 경계의 peak CUDA allocated/reserved bytes.

GPU utilization counter는 PyTorch가 제공하는 NVML-backed query를 사용하며, 프로세스 전용이
아닌 선택된 장치 전체의 순간 사용률이다. 같은 GPU를 외부 프로세스와 공유하면 그 부하도 포함될
수 있다. NVML, `/proc`, CPU affinity 또는 특정 counter를 읽을 수 없으면 해당 값은 `null`과
구체적인 `reason`으로 남고 0이나
추정값으로 대체되지 않는다. 이 시계열 원문은 각 child metrics/summary artifact의
`resource_observability`에 있으며 Tree는 `runtime.resource_observability`에 둔다. V5와 Cycle
V2의 `pre_run_observability`, Cycle V1/Tree의 pre-run JSON 및 구조화된 data/batch/parameter/step
필드는 모델·파라미터·데이터·실제 batch·계획 optimization step을 가능한 실행 경계에 맞춰
기록한다. 상위 scaling summary의 elapsed/peak 집계만 보고 원문 시계열이 있다고 간주하지 말고
child artifact와 hash를 함께 보존한다.

현재 CPU 평균과 process RSS/HWM은 주 학습 프로세스 기준이므로 DataLoader worker 각각의 CPU
time/RSS를 합산한 값이 아니다. System available RAM은 전체 시스템 관측이고 다른 작업의 영향도
포함한다. 1초 표본은 짧은 kernel burst를 놓칠 수 있으며 storage throughput, I/O wait,
data-loading/H2D/forward/backward/optimizer 각 단계의 분해가 세 track에 공통으로 구현된 것은
아니다. Cycle supervised 경로는 loader wait, packed H2D, forward/loss, backward, optimizer
시간을 나눠 기록하지만, 별도 batch 후보 microbenchmark는 이미 준비된 한 real batch의
forward/loss/backward만 측정하고 loader·optimizer·validation·checkpoint를 의도적으로 제외한다.
따라서 resource 시계열만으로 전체 병목 분석이 끝났다고 해석하지 않는다.

`nvidia-smi` 한 번의 390MiB/13% 화면은 CUDA context 생성, CPU 전처리, validation 또는 작은
커널 사이의 순간일 수 있어 전체 GPU 활용률의 증거가 아니다. 내장 시계열과 별개의 외부
교차검사로 학습 중 다음 2초 시계열을 함께 볼 수 있다.

```bash
nvidia-smi --id=3 \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 2
```

최종 판단에는 내장 시계열, 외부 교차검사, elapsed time, CUDA peak, 처리량을 함께 사용한다. GPU
utilization이 낮으면서 CPU가 포화되면 loader/기저 준비 병목을 조사하고, VRAM peak와 GPU
utilization이 모두 낮으면 별도의 batch 후보 측정 run을 설계한다. 한 장면만으로 batch나
concurrency를 바꾸지 않는다.

### Rich runner의 고정 recipe와 별도 batch 후보 microbenchmark

현재 runner의 physical batch는 hardware/dataset/profile별로 사전 등록된 값이다.
`run_rich_scaling.py`는 학습 중 batch를 탐색하거나 recipe를 바꾸지 않고 manifest에도
`throughput_candidate_sweep=false`를 기록한다. A6000 profile의 큰 batch는 명시적 연구
recipe이지 rich runner가 자동 선택한 결과가 아니다.

별도 `scripts/benchmark_speed.py --batch-sizes ...`는 이와 구분되는 CUDA
**fixed-real-batch microbenchmark**다. Conductance V1/V5, Cycle PE V1/V2와 Tree V1/V2에서
공식 train cache를 검증한 뒤 각 명시 후보를 동일 초기 state로 독립 실행하며, device-wide GPU
SM·memory-controller utilization, CUDA peak allocated/reserved, process CPU/RAM과
graphs/seed-nodes/chart-views 처리량을 기록한다. OOM/error 후보를 더 작은 값으로 조용히
대체하지 않고 실패로 보존한다. 10% 이상 projected device-memory headroom을 남긴 후보 중
처리량이 가장 높은 값을 report의 **microbenchmark recommendation**으로 표시한다.

이 benchmark는 optimizer step·parameter update가 0이고 전체 epoch, DataLoader 처리량,
validation/test, checkpoint와 optimizer-state memory를 포함하지 않는다. Legacy Conductance와
동일 연산의 독립 execution variant가 있는 경우에만 출력/loss/모든 gradient 수치 동등성을
`passed`로 기록하고, V5/Cycle V1/Tree의 current-only 경로는 자기 비교를 하지 않고 production
path identity, finite loss/모든 trainable gradient와 parameter 불변성을 검사한다. 권고는
profile 기본값을 자동 변경하지 않으며, 실제 최종 학습 recipe로 채택하려면 새 run ID의 전체
학습에서 optimizer lifetime memory·epoch throughput·validation을 다시 검증해야 한다.

GPU 3의 A6000에서 V5/ogbn-arxiv cluster 후보를 측정하는 독립 명령은 다음과 같다. Output
directory는 기존 결과를 덮어쓰지 않는 새 이름이어야 한다.

```bash
CUDA_VISIBLE_DEVICES=3 python -B scripts/benchmark_speed.py \
  --track conductance_v5 \
  --dataset ogbn-arxiv \
  --v5-scale-profile reference \
  --v5-hardware-profile a6000-48gb \
  --v5-sampling cluster \
  --batch-sizes 1024 2048 4096 \
  --device cuda:0 \
  --resource-sample-interval-seconds 0.1 \
  --output-dir runs/performance/v5-arxiv-a6000-batch-sweep-r1
```

`report.json`의 후보별 원문은
`batch_candidates[*].variants[*].resource_observability.interval_series` 아래
`gpu_sm_utilization_percent`, `gpu_memory_controller_utilization_percent`,
`gpu_allocator_allocated_bytes`, `gpu_allocator_reserved_bytes`,
`gpu_device_free_bytes`, `gpu_device_used_bytes`, `process_cpu_seconds`,
`allocated_cpu_busy_seconds`, `allocated_cpu_total_seconds`, `process_resident_bytes`,
`process_peak_resident_bytes`, `system_available_bytes`에 있다. coordinator-process/allocated-CPU
평균과 caller-defined CUDA peak는 같은 resource 객체의 `summary`, 후보별 순위 적격/배제 이유는
최상위 `batch_candidate_analysis`에 있다. 이 분석은 optimizer 상태와 update가 없는
microbenchmark 순위만 기록하고 최종 학습 batch를 선택하지 않는다. 모든 관측치는
`value/unit/reason` 계약이며,
전체 track은 같은 `scripts/benchmark_speed.py` 진입점에서 `--track`
(`conductance_gat`, `conductance_v5`, `cycle_pe_v1`, `cycle_pe_v2`,
`tree_augmentation`)만 바꿔 실행한다. 생성되는 CSV에는 run/track/profile,
후보 physical/effective batch size, 반복별 처리량과 지연시간, GPU·CPU·RAM 관측치,
microbenchmark 순위 적격 여부와 배제 이유가 기록된다.

Cycle V2의 pre-epoch capacity probe는 등록된 worst-case batch가 가능한지 확인하고 불가능하면
명시적으로 실패시키는 검사다. 더 작은 batch로 자동 변경하지 않으며 throughput sweep도 아니다.
또한 고정 batch로 완료된 학습 run의 `graphs/sec` 또는 `samples/sec`는 그 한 설정의 사후
처리량일 뿐, 다른 후보보다 최적이라는 증거가 아니다. 별도 microbenchmark 권고와 전체 학습
검증을 거쳐 선택 결과를 새 hardware profile과 새 run ID에 명시하기 전에는 현재 batch가
하드웨어 최적이라고 주장하지 않는다.

## 10GB MIG 메모리 정책

- V5 activation checkpointing 기본 on.
- ogbn-arxiv V5 train은 기본 cluster sampling, seed-node batch 1024, edge chunk 65536.
- PPI는 공식 inductive graph split 때문에 full graph batch 2.
- 모든 validation은 완전한 공식 graph/split에서 수행한다.
- Cycle V2는 sparse cycle membership·AMP/batch를 기록하며 parameter ceiling 50M을 적용한다.
- 실제 peak memory가 없는 상태에서 large가 10GB에 적합하다고 주장하지 않는다. reference부터
  실행하고 manifest의 peak allocation으로 large 실행 가능성을 판단한다.

2026-09-04 run `new-v5-cyclev2-a6000-gpu3-seed0-r1`의 GPU 로그를 수령했다. Conductance는
20개 중 reference/ogbn-arxiv/fixed-C 하나만 완료했고, dynamic-C는 최초 C calibration에서
44.47/44.55GiB OOM, 나머지 18개는 미실행이다. 완료된 fixed-C도 구 joint-only 선택이 실제
epoch-10 global best 0.692775 대신 0.680392를 골라 corrected 비교에 재사용할 수 없다. Cycle V2
네 학습은 기존 FP16 GradScaler overflow로 모두 첫 epoch 전에 실패했다. 따라서 이 run에는
당시 V5 또는 projector V2의 유효한 전체 성능 결과가 없다. 수정판은 source/resume identity가
달라 새 run ID를 사용하되 V1–V4, Cycle V1과 Tree를 다시 계획할 필요는 없다.

후속 `new-v5-cyclev2-a6000-gpu3-seed0-r2`도 동일하게 중단됐다. 첨부
`bd63fc9a-60da-4daf-9ab9-da49db7cbbe1/pasted-text.txt`의 SHA-256은
`F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6`다. 이 로그의
Conductance `train.py:785`·`joint_best=` 형식과 Cycle `benchmark.py:589`는 수정 전
`08d8ed6`에 해당한다. 즉 r2에는 `214265c`의 score checkpoint/selection/BF16 수정이 적용되지
않았고, r1의 dynamic-C OOM과 Cycle 네 non-finite gradient 실패를 old source로 다시 재현했을
뿐이다. r2를 수정 검증으로 간주하거나 같은 output에서 수정판을 resume하지 않는다.

이후 r3의 `v5/large/ogbn-arxiv/shared_dynamic_c`는 `214265c` 수정판으로 실행됐지만
diffusion에서 192MiB 할당 OOM이 재발했다. 현재 custom backward는 edge-feature 저장 누적을
제거했고 관련 CPU 검사28개와 합성 저장량 약89.3% 감소를 확인했다. 이것은 새 A6000 전체 학습
성공이나 VRAM 측정이 아니다. 상세 로그와 수정 범위는 [CONDUCTANCE_V5.md](CONDUCTANCE_V5.md)를 따른다.

### 이번 QR-free Cycle V2만 새로 실행

아래는 두 dataset×두 profile×SE/PE 두 조건×seed0, **8개 학습**과
조건×dataset별 validation 선택 후 checkpoint test-only 평가4개다.
SE와 PE를 섞어서 하나만 선택하지 않는다. 완료한 Conductance/Cycle V1/Tree를 반복하지 않는다.
유효한 source가 게시·동기화된 후 사용하며, 단순히 과거 `214265c`가 존재하는 것은 이번
diffusion/sparse DFS 수정판을 확보했다는 증거가 아니다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_cycle_scaling.py \
  --versions v2 --profiles reference large \
  --encodings se pe \
  --datasets zinc12k peptides_struct --model-seeds 0 \
  --device cuda:0 --hardware-profile a6000-48gb --min-free-gb 40 \
  --basis-backend dfs_fundamental \
  --run-id cycle-se-pe-a6000-gpu3-seed0-v1
```

새 공유 cache namespace는 `cycle_pe_v2_ordered_dfs_benchmark`다. 구 cache/checkpoint는
호환되지 않는다. 같은 새 source/config/schema에서 위 명령을 반복하면 완료 child를
hash 검증 후 건너뛰고 encoding별 모델 폴더의 `last.pt`에서 epoch 상태를 복원한다.
두 조건은 모델 shape가 같아도 checkpoint를 교환할 수 없다.
V5도 operator source가 바뀌었으므로 구 r3 source checkpoint를 현재 구현에 강제로 연결하지
않는다. 실행 대상과 새 run ID를 명시적으로 선택하며 기존 artifact는 보존한다.
Block checkpoint override나 다른 설정 변경 역시 manifest/config/job command에 결속된다.
