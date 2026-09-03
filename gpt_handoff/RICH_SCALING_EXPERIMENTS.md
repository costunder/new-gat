# Conductance V1–V5·Cycle PE V1/V2·Tree reference/large 전체 scaling

과거 `64/128 × 2/4층` grid는 mechanism probe로만 남기고 현재 scaling 기본 계획에서는
폐기한다. 새 계획은 파라미터를 버전 사이에 강제로 일치시키지 않으며, 각 구조가 실제 연구급
capacity에서 성능을 낼 수 있는지를 본다. 기본 seed는 시간 제약 때문에 0 하나다.

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

V2는 FFN×4, dropout 0.1, layer scale 0.1을 사용한다. 두 versions × 두 datasets × 두
profiles = 8 trainings다. 현재 V2 identity는 `cycle_projector_pe_v2`이며 폐기된
`cycle_basis_v2` 결과나 checkpoint를 혼용하지 않는다.

### Tree V1/V2

| Profile | Hidden | Message layers |
|---|---:|---:|
| `reference` | 128 | 8 |
| `large` | 256 | 12 |

CSL/ZINC × fixed-BFS/multi-chart × 두 profiles = 8 model trainings다.

## 전체 계획

- Conductance: 106 child/model trainings.
- Cycle: 8 child/model trainings.
- Tree: 4 child runs, 각 두 모델 = 8 model trainings.
- 합계: **118 child runs, 122 fresh model trainings**.
- profile 선택 후보는 train/validation만 사용한다. 기존 Cycle/Tree 계약의 selected-checkpoint
  test-only 단계는 validation 선택 이후 별도 평가이며 재학습하지 않는다.

다음은 과거에 사용한 물리 GPU 6의 10GB MIG slice를 그대로 지정하는 **portable 실행
예시**다. GPU 번호는 실제 할당에 맞춰 바꾸되 프로세스 내부 장치는 항상 `cuda:0`을 쓴다.

```bash
git pull --ff-only

env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-portable-gpu6-seed0-v1 \
  --profiles reference large \
  --model-seeds 0 \
  --device cuda:0 \
  --hardware-profile portable \
  --cycle-v2-basis-backend thin_q
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
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-a6000-gpu3-seed0-v1 \
  --profiles reference large \
  --model-seeds 0 \
  --device cuda:0 \
  --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --cycle-v2-basis-backend thin_q
```

최상위 `--cycle-v2-basis-backend`는 Cycle child의 V2에 전달되고 통합 manifest와 재개
configuration에 결속된다. `thin_q`가 전체 학습용 기본값이다. DFS forest와 parent-path
역추적 경로를 진단하려면 새 run ID에서 `dfs_fundamental`을 선택한다. 이 경로는 runtime QR을
반복하므로 속도 개선용 설정이 아니다.

### 완료된 구 Conductance scaling을 다시 돌리지 않는 실행

이미 `base/wide/deep/large`에서 Conductance V1–V4 172회를 완료한 경우 그 결과를 별도
artifact로 보존하고, 현재 통합 실행에서는 Conductance V5만 선택할 수 있다. Cycle V1/V2와
Tree까지 아직 남았다면 다음 명령은 **32 child runs / 36 fresh model trainings**만 계획한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
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
  --cycle-v2-basis-backend thin_q \
  --allow-download
```

V5와 폐기·재구현된 Cycle V2만 새로 실행할 때는 `--tracks conductance cycle
--conductance-versions v5 --cycle-versions v2`를 사용한다. reference/large와 seed 0에서
**24 child runs / 24 fresh model trainings**다. 선택한 버전 목록은 통합 manifest와 재개
identity에 들어가므로, 중단 후에는 같은 run ID와 같은 목록을 그대로 사용한다.

architecture profile(`reference/large`)과 hardware profile(`portable/a6000-48gb`)은 서로 다른
축이다. A6000 profile의 실제 실행 차이는 다음과 같다.

| 트랙 | `portable` | `a6000-48gb` |
|---|---|---|
| Conductance V5 | FP32, TF32 off, activation checkpoint on, edge chunk 65,536, arxiv seed-node batch 1,024, PPI whole-graph batch 2 | dense BF16 autocast·TF32, conductance geometry FP32, checkpoint off, edge chunk 131,072, arxiv seed-node batch 2,048, PPI whole-graph batch 8, sample prefetch/pinned transfer |
| Conductance V1–V4 | 기존 FP32 계약. PPI는 V1/V3/V4 batch 2이고 V2는 PPI N/A | 같은 legacy FP32·batch 계약을 그대로 유지 |
| Cycle V1/V2 | 모든 dataset/profile batch 32, workers 4, prefetch 2, AMP off; V2 column chunk 16, pair budget 32,768 | reference ZINC/Peptides batch 512/128, large 256/64, workers 8, FP16 AMP; V1은 실제 loader prefetch 2, V2는 prefetch 4·FP32 projector·column chunk 32·pair budget 4,194,304 |
| Tree V1/V2 | batch 16, workers 0, suite config의 FP16 AMP, child concurrency 1 | batch 64, workers 4, 명시적 FP16 AMP, 독립 child concurrency 2 |

Conductance와 Cycle child는 순차 실행한다. Tree만 서로 다른 output/log를 갖는 candidate와
selected-test child를 최대 2개 병렬화하며 coordinator 하나만 공용 manifest를 원자적으로 쓴다.
A6000 작업 순서는 큰 workload를 먼저 검증하도록 deterministic heavy-first다. 한 Tree child가
실패해도 같은 wave에서 통과한 peer artifact는 보존되며 같은 run ID 재실행에서는 실패한 작업만
새 attempt 경로에서 실행한다.

이 설정은 메모리만 더 쓰는 동치 실행 스위치가 아니다. V5는 real sample/PPI batch와 numeric
execution이 바뀌고, Cycle은 batch와 AMP가 바뀌며, Tree는 800 optimizer updates마다 보는 graph
수가 달라진다. 따라서 fixed/dynamic C, Cycle V1/V2, Tree fixed/multi 비교는 같은 hardware
profile 안에서만 해석한다. 특히 legacy Conductance V1/V3/V4 PPI는 batch 2/FP32이고 V5 A6000
PPI는 batch 8/BF16이므로 V1–V5 PPI 차이는 descriptive scaling 결과일 뿐 단일 구조 요인의
인과 비교가 아니다. portable와 A6000 사이의 점수나 wall time 차이를 모델 또는 GPU 하나의
효과로 직접 해석하지 않는다.

`nvidia-smi` 한 번의 390MiB/13% 화면은 CUDA context 생성, CPU 전처리, validation 또는 작은
커널 사이의 순간일 수 있어 전체 GPU 활용률의 증거가 아니다. 학습 중 별도 터미널에서 다음처럼
2초 시계열을 확인한다.

```bash
nvidia-smi --id=3 \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 2
```

최종 판단에는 이 시계열과 각 child summary에 상위 aggregate까지 보존되는 `runtime`,
`elapsed_seconds`, `peak_gpu_allocated_bytes`, `peak_gpu_reserved_bytes`를 함께 사용한다. GPU
utilization이 낮으면서 CPU core가 포화되면 loader/기저 준비 병목이고, VRAM peak가 충분히 낮고
GPU utilization도 낮으면 다음 run ID에서 batch/concurrency 조정을 검토한다. 현재 한 장면만으로
batch나 concurrency를 더 올리지는 않는다.

## 10GB MIG 메모리 정책

- V5 activation checkpointing 기본 on.
- ogbn-arxiv V5 train은 기본 cluster sampling, seed-node batch 1024, edge chunk 65536.
- PPI는 공식 inductive graph split 때문에 full graph batch 2.
- 모든 validation은 완전한 공식 graph/split에서 수행한다.
- Cycle은 projector row pair budget과 AMP/batch를 기록하며 parameter ceiling 50M을 적용한다.
- 실제 peak memory가 없는 상태에서 large가 10GB에 적합하다고 주장하지 않는다. reference부터
  실행하고 manifest의 peak allocation으로 large 실행 가능성을 판단한다.

현재 이 문서는 실행 계약이다. 새 V5와 projector V2의 GPU 성능 결과는 아직 수령하지 않았다.
