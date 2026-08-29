# Static Cycle PE 연구 트랙

이 폴더는 그래프 topology에서 한 번 계산하는 정적 cycle positional encoding(PE)을
독립적으로 검증한다. Linux/CUDA 논문 경로는 `research.cycle_pe.paper`이며,
CycleCount-OOD, BREC v3, ZINC-12K를 같은 batch-safe backbone에서 실행한다. 기존
`research.cycle_pe.run`은 작은 structural smoke probe로 남아 있고 논문 결과 경로가 아니다.

## 구현 경계와 PE 비교군

모든 PE는 edge-by-node incidence matrix와 BFS spanning tree로 만든 fundamental cycle
basis에서 학습 전에 계산된다. 논문 CLI는 다음 네 비교군을 기본으로 모두 실행한다.

- `no_pe`: topology PE를 주지 않는 backbone
- `raw`: fundamental basis 좌표를 직접 투영하는 진단용 baseline
- `set`: cycle-column 부호와 순열에 불변인 edge별 cycle-set 통계
- `projector`: cycle-space projector를 사용하는 basis-change 불변 baseline

`raw`는 intrinsic 표현이 아니다. 폭은 **train split의 최대 cycle rank로만** 정하고, 더 큰
rank가 나온 validation/test/OOD split에는
`not_applicable_train_fitted_width_overflow`를 기록한다. 좌표 절단이나 test-fit은 하지 않는다.
`set`은 column 부호/순열에는 불변이지만 spanning-tree 변경 불변성까지 보장하지 않는다.
`projector`는 invertible basis change에 불변이며 prior-style baseline이지 이 트랙의 novelty
주장이 아니다.
모든 variant에서 incidence, BFS tree와 full fundamental basis `F_T`/`raw_basis`를 계산한다.
요청 여부에 따라 조건부로 생략되는 것은 set 통계와 dense projector다. 따라서
`no_pe/raw/set` 실행에는 dense `m×m` projector를 만들지 않지만 `no_pe`도 basis 전처리 비용을
없애지는 않으며, projector variant 자체의 O(m²) 메모리/시간 한계는 남는다.

learned conductance, node potential, sample-dependent circulation, layer 간 circulation state,
flow completion은 이 트랙에 포함하지 않는다. spanning-tree augmentation도 별도
`research/tree_augmentation` 트랙이다.

## Suite와 데이터

### `core`: CycleCount-OOD

외부 다운로드가 없는 deterministic generator다. non-tiny protocol은 총 20,000개다.

| split | 그래프 수 | 역할 |
|---|---:|---|
| train | 10,000 | training family/size |
| validation | 2,000 | 같은 family/size의 held-out seed |
| id_test | 2,000 | ID 최종 평가 |
| size_ood | 3,000 | 더 큰 node-count 범위 |
| family_ood | 3,000 | unseen small-world/local-chord family |

target은 leakage를 막기 위해 한 모델에서 함께 최적화하지 않는다. `--core-targets`로 선택한
각 level마다 별도 model/head/checkpoint가 생긴다.

- edge: C3, C4, C5, C6 참여 횟수, shortest-cycle length, cycle congestion
- node: C3, C4, C5, C6 참여 횟수
- graph: C3, C4, C5, C6 총개수

MAE, RMSE, train-normalized MAE, graph-macro MAE, rounded exact accuracy를 기록한다. 현재
generator는 size-OOD와 family-OOD를 구현하지만 degree-sequence-matched counterfactual은
구현하지 않았으며 그런 coverage를 주장하지 않는다.

### `brec`: BREC v3 official RPC

full run은 공식 `brec_v3.npy`의 400개 pair와 category(Basic, Regular, Extension, CFI,
4-Vertex-Condition, Distance-Regular)를 사용한다. 기본 q=32와 threshold=72.34에서
[공식 `base/test_BREC.py`](https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py)의
계산을 그대로 따른다.

```text
D = left_embeddings - right_embeddings
T² = D_mean.T @ torch.linalg.pinv(torch.cov(D)) @ D_mean
```

q 배수는 곱하지 않는다. distinction은 `train_T² > 72.34`이면서 train/reliability T²가
공식 `torch.isclose(..., atol=1e-6)` 조건에서 같지 않을 때이고, reliability 통과는 별도로
`reliability_T² < 72.34`이다. Full run의 기본 `--brec-protocol official`은 batch 16,
20 epochs, LR/weight decay `1e-4`, float32, no AMP, no clipping, no shuffle를 강제한다.
각 variant와 공식 seed `100,200,...,1000` 조합을 한 번 seed한 뒤 400 pair를 순서대로
평가한다. Upstream 호환 산출물은 seed별 `Correct`, `Fail`, `Real_correct`다. 열 seed 전체가
complete하고 reliability failure가 없을 때만 참인 `global_valid`는 **저장소가 추가한 보수적
gate**이지 upstream BREC metric이 아니다. 공식 search가 정의하지 않은 any-seed union은
만들지 않는다. 상수와 제어 흐름은 reference에 정적으로 맞췄지만 upstream runner와의
golden-output/differential numerical parity는 아직 검증하지 않았다.

`--brec-protocol custom`은 flexible q/batch, AMP, clipping, pair shuffle과 derived per-pair
seed를 허용하며, 기존 집계는 `custom_pairwise_union`으로만 기록한다. `--tiny`가 protocol을
생략하면 offline fixture에 맞춰 custom으로 해석된다.

기본 데이터 위치는 `<data-root>/BREC/Data/raw/brec_v3.npy`다. 파일이 없으면 fail-closed이며,
`--allow-download`를 명시했을 때만 GraphPKU Release ZIP을 HTTPS로 받아 경로, symlink,
압축/추출 크기를 검사하고 `brec_v3.npy` 하나만 추출한다. Official mode는 q=32,
400 pairs, 51,200 records를 강제한다. ZIP과 NPY SHA-256은 provenance로 기록하지만,
GraphPKU가 canonical SHA-256 pin을 배포했다고 주장하지 않는다. `--tiny`는 네트워크를
사용하지 않고 로컬 RPC fixture를 생성하므로 pipeline smoke test에만 사용한다.

### `zinc`: ZINC-12K

PyTorch Geometric의 `ZINC(subset=True)` official train/val/test 10,000/1,000/1,000 split과
atom/bond feature를 사용한다. local cache가 없으면 fail-closed이고 `--allow-download`를
명시해야 PyG download를 허용한다. `--tiny`도 official split의 앞부분만 읽으므로 PyG와
cache(또는 명시적 download)는 여전히 필요하다. PyG import가 없거나 CUDA/PyTorch 조합이
맞지 않으면 설치 문서를 포함한 actionable error로 종료한다.

### 현재 과학적·scaling 한계

- `B^Tq`는 circulation component를 잃지만, simple unweighted graph의 full `L=D-A`는 adjacency와
  topology를 결정한다. 이 PE는 sample circulation 복구가 아니라 topology inductive bias다.
- Degree/`(n,m,beta)`-matched 또는 1-WL-indistinguishable known-contrast가 없고, raw/set의
  isomorphic relabeling·spanning-tree shift robustness도 core/ZINC에서 평가하지 않았다.
- No-PE 외 LapPE/sign-invariant LapPE, RWSE, GIN/GINE, GPS/Graphormer, CycleNet/기존 cycle-space
  PE baseline이 없다.
- Suite-level preparation time은 variant별 비용을 분리하지 않고 CPU RSS도 기록하지 않는다.
  Projector는 모든 split graph의 dense `m×m` tensor를 보관하므로 large-graph scaling이 미검증이다.
- Official BREC는 pair/seed 단위 incremental checkpoint/resume가 없고 완료 뒤 결과를 기록한다.
  중단 시 현재 child의 장시간 진행분을 같은 run-id로 재개할 수 없다.

## 독립 seed 축

CLI는 `--data-seed`, `--split-seed`, `--chart-seed`, `--model-seed`를 독립적으로 받는다.
기존 `--seed`는 지정하지 않은 축의 호환 fallback일 뿐이다.

- CycleCount 생성과 content-addressed cache identity에는 `data_seed`만 사용한다.
- Supervised model 초기화와 DataLoader shuffle에는 `model_seed`만 사용한다.
- CycleCount split은 generator가 family/size regime별로 직접 만들므로 현재 `split_seed`는
  `not_applicable`이다.
- Static Cycle PE는 deterministic BFS fundamental basis를 사용하며 chart sampling이 없으므로
  `chart_seed`도 `not_applicable`이다.
- ZINC는 fixed public data와 PyG official split을 그대로 사용하므로 data/split seed가
  `not_applicable`이고 model seed만 학습에 사용된다.
- BREC official mode는 바깥 `model_seed`를 사용하지 않는다. 내부 official search seed
  `100,200,...,1000`이 별도 protocol axis다.

각 suite manifest의 `seed_axes`는 resolve된 네 값을, `seed_axis_policy`는 실제 사용 여부와
`not_applicable` 사유를 기록한다.

## Linux/CUDA 설치

서버 CUDA driver와 맞는 PyTorch wheel을 먼저 설치한 뒤 저장소 root에서 실행한다.

```bash
python -m pip install --upgrade pip
# CUDA 버전에 맞는 torch wheel 설치 후:
python -m pip install --no-build-isolation -e ".[dev,paper]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

MobaXterm은 SSH terminal일 뿐이므로 같은 Linux 명령을 사용한다. 결과 재현성 때문에 실제
GPU 서버의 Python, CUDA, torch, GPU 이름은 각 manifest에 자동 기록된다.

## 정확한 실행 명령

각 실행의 `--output-dir`는 존재하지 않거나 비어 있어야 한다. 기존 artifact가 있으면
덮어쓰지 않고 종료한다.

CycleCount-OOD 전체 CUDA run:

```bash
python -m research.cycle_pe.paper \
  --suite core \
  --data-root ./data \
  --output-dir ./paper_runs/cycle-core-seed2025 \
  --device cuda \
  --seed 2025 --data-seed 2025 --model-seed 2025 \
  --workers 8 \
  --batch-size 64 \
  --variants no_pe,raw,set,projector \
  --core-targets edge,node,graph \
  --amp --pin-memory --non-blocking
```

BREC v3 전체 CUDA run(공식 열 seed가 기본값):

```bash
python -m research.cycle_pe.paper \
  --suite brec \
  --data-root ./data \
  --output-dir ./paper_runs/cycle-brec \
  --device cuda \
  --seed 2025 \
  --brec-protocol official \
  --workers 0 \
  --batch-size 16 \
  --variants no_pe,raw,set,projector \
  --brec-num-relabel 32 \
  --brec-seeds 100,200,300,400,500,600,700,800,900,1000 \
  --allow-download \
  --no-amp --no-pin-memory --no-non-blocking
```

이미 공식 BREC 파일을 배치했다면 `--allow-download`를 빼는 편이 더 엄격하다.

ZINC-12K 전체 CUDA run:

```bash
python -m research.cycle_pe.paper \
  --suite zinc \
  --data-root ./data \
  --output-dir ./paper_runs/cycle-zinc-seed2025 \
  --device cuda \
  --seed 2025 --model-seed 2025 \
  --workers 8 \
  --batch-size 64 \
  --variants no_pe,raw,set,projector \
  --allow-download \
  --amp --pin-memory --non-blocking
```

모든 suite의 cache/graph/PE만 검증하고 학습을 생략하려면 다음처럼 실행한다. `all`은
core → BREC → ZINC 순서이며 어느 suite라도 실패하면 그 실패 suite의 불완전 artifact만
정리한다. 이미 완료된 suite는 보존하고 최상위 `run_manifest.json`에 완료 suite hash와 실패
원인을 남긴다.

```bash
python -m research.cycle_pe.paper \
  --suite all \
  --data-root ./data \
  --output-dir ./paper_runs/cycle-prepare \
  --device cuda \
  --seed 2025 \
  --workers 8 \
  --prepare-only \
  --allow-download
```

다운로드 없는 CPU core smoke와 BREC offline fixture 확인:

```bash
python -m research.cycle_pe.paper \
  --suite core --data-root ./data --output-dir ./paper_runs/core-tiny \
  --device cpu --seed 7 --workers 0 --tiny --prepare-only --no-amp

python -m research.cycle_pe.paper \
  --suite brec --data-root ./data --output-dir ./paper_runs/brec-tiny \
  --device cpu --seed 7 --workers 0 --tiny --prepare-only --no-amp
```

`--epochs`, `--learning-rate`, `--weight-decay`, `--hidden-dim`, `--pe-dim`, `--layers`로
학습 설정을 override할 수 있다. CPU에서는 AMP, pinned memory, non-blocking transfer가
자동으로 비활성화된다.

## Artifact 구조

```text
<output-dir>/
├── run_manifest.json
└── core/
    ├── manifest.json
    ├── edge/<variant>/{model.pt,metrics.json,history.json,runtime.json}
    ├── node/<variant>/{model.pt,metrics.json,history.json,runtime.json}
    └── graph/<variant>/{model.pt,metrics.json,history.json,runtime.json}
```

BREC는 `<variant>/pairs.json`과 `<variant>/metrics.json`, ZINC는
`graph/<variant>/...`를 쓴다. manifest에는 CLI 전체, seed, split 크기, target 분리 정책,
raw overflow, code/cache hash, runtime, CUDA AMP 설정, peak GPU memory가 포함된다.

## 검증

```bash
python -m ruff check research/cycle_pe
python -m pytest research/cycle_pe/tests -q
python scripts/check_datasets.py --profile paper --json
```

기존 structural smoke probe는 별도로 실행한다.

```bash
python -m research.cycle_pe.run
```

smoke용 `config.yaml`의 `max_cycles`는 legacy linear-probe padding 설정이다. 논문
`research.cycle_pe.paper` 경로는 이 값을 읽지 않으며 variable-beta graph를 고정 cap 없이
처리한다.
