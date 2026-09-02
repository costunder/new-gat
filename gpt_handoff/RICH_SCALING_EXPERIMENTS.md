# 전체 모델 규모 확장 실험

이 문서는 파라미터 수를 억지로 같게 맞추는 실험이 아니다. 각 방법의 원래 수식과
파라미터화를 유지한 채 모델을 더 넓게 또는 더 깊게 만들어 성능이 어떻게 변하는지 확인하는
사전 정의 scaling 실험이다. 기존 단일 크기 benchmark는 `base`로 그대로 보존하고, 새로운
결과는 모두 별도의 `results/*/scaling/` 경로에 저장한다.

## 실험 프로필

### Conductance V1/V2/V3/V4

| Profile | Hidden | Conductance layers | Dropout |
|---|---:|---:|---:|
| `base` | 64 | 2 | 0.5 |
| `wide` | 128 | 2 | 0.5 |
| `deep` | 64 | 4 | 0.5 |
| `large` | 128 | 4 | 0.5 |

V1도 제외하지 않는다. V1은 Cora, CiteSeer, PubMed, PPI, ogbn-arxiv에서 실행한다. V2는
물리 edge ID에 묶인 직접 C이므로 Cora, CiteSeer, PubMed, ogbn-arxiv에서만 실행하고 PPI는
명시적인 N/A다. V3/V4는 V1의 다섯 데이터를 모두 사용한다.

한 profile/seed당 V1 5회, V2 8회, V3 10회, V4 20회로 총 43회다. 시간 제약을 반영한 기본값은
model seed 0 하나이므로 `43 × 4 × 1 = 172`개의 validation-only fresh training이다. V1
scaling은 전용 경로에서 test split을 만들거나 평가하지 않는다. V2/V3/V4도 기존
validation-only 계약을 유지한다. 추가 seed는 `--model-seeds`에 명시한 경우에만 실행한다.

### Cycle PE V1/V2

| Profile | Hidden | PE width | Message layers |
|---|---:|---:|---:|
| `base` | 64 | 32 | 3 |
| `wide` | 128 | 64 | 3 |
| `deep` | 64 | 32 | 6 |
| `large` | 128 | 64 | 6 |

V1 `cycle_set`과 V2 `cycle_basis_v2`를 모두 ZINC-12K와 Peptides-struct에서 실행한다. 기본값은
model seed 0 하나이므로 `2 versions × 2 datasets × 4 profiles × 1 seed = 16`개의 fresh
training이다. 이 16개 후보 학습은 train/validation split만 로드한다. `version × dataset`별로
요청된 seed의 평균 validation MAE가 가장 낮은 공통 profile 하나를 고정한 다음, 기본 실행에서는
선택 checkpoint 4개만 test-only 단계에서 각각 한 번 평가한다. 후보 학습에는 test loader나
test metric이 없다.

### Tree fixed/multi

| Profile | Hidden | Message layers | Optimizer updates | Train/eval charts |
|---|---:|---:|---:|---:|
| `base` | 64 | 2 | 800 | 8/8 |
| `wide` | 128 | 2 | 800 | 8/8 |
| `deep` | 64 | 4 | 800 | 8/8 |
| `large` | 128 | 4 | 800 | 8/8 |

CSL과 ZINC 각각에서 모든 profile과 기본 model seed 0을 실행한다. 한 child가 `fixed_bfs`와
`multi_chart`를 모두 별도 초기화·학습하므로 `2 datasets × 4 profiles × 1 seed × 2 models
= 16`개의 fresh model training이다. 후보 학습은 공식 validation split만 평가한다. 이후
`dataset × condition`별로 요청된 seed의 평균 validation 목적값을 사용해 공통 profile 하나를
선택하고, 기본 실행에서는 선택 checkpoint 4개만 test-only 단계에서 평가한다. 네 profile의
update 수와 chart 수는 같으므로 `deep/large`의 차이는
더 오래 학습하거나 chart를 더 본 효과가 아니라 실제 message-layer 증가다. 실제 적용 설정,
전체/학습 가능 파라미터 수, chart family, checkpoint와 artifact hash를 기록한다.

Tree의 공식 데이터는 모든 split을 담은 단일 검증 cache이므로 candidate도 cache 전체의 형식과
hash를 읽어 검증한다. Manifest는 이 cache 무결성 접근과 모델의 split 사용을 분리한다. 모델
fit에는 train, profile 목적값에는 validation만 사용하고, 선택 전에는 test metric을 계산하거나
profile 선택에 사용하지 않는다. 출력 차원도 전체 record label이 아니라 선언된 target metadata로
정한다.

## 전체 범위와 실행

기본 전체 실행은 Conductance 172 + Cycle 8 + Tree 8 = **188 training child runs**다.
Cycle과 Tree의 child 하나가 모델 두 개씩을 학습하므로 실제 학습 수는 Conductance 172회,
Cycle 16회, Tree 16회, 총 **204 fresh model trainings**이다. 학습 완료 후 Cycle 4회와
Tree 4회의 선택-checkpoint test-only 평가가 추가되지만, 이는 optimizer나 재학습을 만들지
않으므로 188 training child/204 학습 횟수에 넣지 않는다. 작은
점검용 프로필이 아니라 실제 공식 데이터 학습이므로 상당한 GPU 시간이 필요하다. 실행계획만
먼저 확인하려면 다음 명령을 사용한다. `--dry-run`은 결과 폴더를 만들지 않는다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-all-gpu6-v1 \
  --profiles base wide deep large \
  --model-seeds 0 \
  --device cuda:0 \
  --dry-run
```

전체 실행은 마지막 `--dry-run`만 제거한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-all-gpu6-v1 \
  --profiles base wide deep large \
  --model-seeds 0 \
  --device cuda:0
```

더 짧은 사전 점검이 꼭 필요할 때만 base/large 두 profile로 줄일 수 있다. 이 결과는 네 profile
전체 결과로 가장하면 안 된다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_rich_scaling.py \
  --run-id rich-screen-gpu6-v1 \
  --profiles base large \
  --model-seeds 0 \
  --device cuda:0
```

통합 runner는 Conductance, Cycle, Tree를 순서대로 실행한다. 한 트랙이 실패해도 기본값은
다음 트랙을 계속 실행하고 전체 상태를 `failed`로 남긴다. 첫 실패에서 멈춰야 할 때만
`--fail-fast`를 명시한다. 어떤 명령에도 `set -euo pipefail`은 필요하지 않다.

중단 후에는 **인수와 `--run-id`를 바꾸지 않고 같은 명령을 다시 실행**한다. 통합 runner와 세
하위 runner는 기존 manifest의 설정·소스 hash·작업 행렬을 먼저 대조하고, `passed` 작업의
artifact와 summary hash를 다시 검증한 뒤 건너뛴다. `pending`, `running`, `failed`였던 작업만
다시 실행한다. 하위 run 전체가 이미 완료된 경우에는 dependency·source·candidate·선택
checkpoint·집계 summary를 재검증하고 GPU preflight나 child를 다시 띄우지 않는다.

기존 `passed` artifact가 손상됐으면 Conductance는 fail-closed로 중단한다. Cycle은 손상된
candidate/test 출력을 `resume-orphans`에 보존하고 깨끗한 원래 binding에서 재시도한다. 후보
재학습으로 선택 checkpoint/hash가 달라지면 이전 test 결과도 보존한 뒤 새 selection에
재바인딩하여 test-only를 한 번 다시 실행한다. Tree는 손상 attempt를 보존하고
`resume-attempts`의 새 경로에서 실행한다. Manifest·summary·preflight·output·log가 run 밖이나
간접 symlink를 가리키면 subprocess 전에 거부한다. 어느 경우도 조용히 덮어쓰지 않는다.
인수나 소스가 달라졌으면 섞어서 재개하지 않고 새 run ID를 요구한다. 현재 실행 중이던 단일
child의 epoch 내부 상태를 잇는 방식은 아니므로 그 child만 처음부터 다시 시작하지만, 이전에
완료·검증된 child는 재학습하지 않는다.

## 결과와 주장 범위

- 통합 상태: `results/rich_scaling/<run-id>/manifest.json`
- Conductance: `results/conductance_gat/scaling/<run-id>-conductance/`
- Cycle: `results/cycle_pe/scaling/<run-id>-cycle/`
- Tree: `results/tree_augmentation/scaling/<run-id>-tree/`

각 후보 child의 최초 학습은 기존 다른 run의 checkpoint를 재사용하지 않는다. 같은 run ID의
재개에서는 해당 run에서 이미 통과한 checkpoint와 artifact만 검증 후 유지한다. Profile, seed,
실제 hidden/layer/update/chart 설정, trainable parameter 수, validation, 실행시간과 peak GPU
memory를 가능한 범위에서 검증한다. Cycle/Tree의 후보 행에는 test 값이 허용되지 않고, validation
선택이 완료된 checkpoint만 별도 test-evaluation 행에 들어간다. 이 scaling curve는 큰 모델에서
각 방법이 어떻게 변하는지 보는 실험이다. 서로 다른 버전의 파라미터 수가 같다고 주장하지 않으며,
버전 간 정규화·optimizer·파라미터화 차이가 사라졌다고도 주장하지 않는다.

현재 저장소에 추가된 것은 실행 코드와 계약 테스트다. 204개 GPU 학습 결과가 완료됐다는 뜻은
아니며, 서버 manifest와 summary를 수령하기 전에는 성능 결론을 기록하지 않는다.
