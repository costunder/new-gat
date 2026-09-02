# Static Cycle PE 연구 트랙

좌영공간 기저벡터 자체를 입력하는 새 버전은 [Cycle PE v2](../gpt_handoff/CYCLE_PE_V2.md)에 있다.
아래 기본 실행은 기존 6개 통계형 v1이며, v2와 캐시·모델·결과를 공유하지 않는다.

`research/cycle_pe/`의 코드는 그래프 topology에서 계산하는 **우리 Cycle PE 모델만** 학습·평가한다.
기본 실행 경로는 `research.cycle_pe.benchmark`다. SignNet·PEARL이 사용한 ZINC-12K와
PEARL이 사용한 Peptides-struct의 공식 데이터·split을 쓰되, **해당 논문 모델을
재구현하거나 같이 실행하지 않는다.** 다른 논문의 성능은 출처를 명시한 외부 비교표에서 다룬다.
다른 트랙의 conductance 연산이나 tree 증강도 결합하지 않는다.

## 이 트랙 재현

[시작 안내](GETTING_STARTED.md)의 환경 설치와 데이터 준비를 완료한 뒤,
프로젝트 Conda 환경이 활성화된 저장소 최상위 폴더에서 실행한다.

```bash
bash research/cycle_pe/reproduce.sh
```

ZINC-12K와 Peptides-struct에서 우리 `cycle_set` 모델을 CUDA model seed `0`으로 실행한다.
여러 seed 반복은 `--model-seeds 0,1,2,3,4`처럼 명시한다. 학습/검증/시험은 공식 split
그대로이며 seed로 다시 나누지 않는다.
학습에 필요한 공개 데이터가 없거나 손상되면 자동 다운로드나 대체 없이 오류로 종료한다.
데이터·결과 경로, run ID와 공통 옵션은 시작 안내를 따른다.

## 기본 데이터와 우리 모델

| 데이터 | 동일 데이터를 사용하는 논문 | 공식 train / validation / test | 입력과 정답 |
|---|---|---|---|
| ZINC-12K | SignNet, PEARL | 10,000 / 1,000 / 1,000 | 원자·결합 범주 → penalized logP, MAE |
| Peptides-struct | PEARL Appendix K.2 | 10,873 / 2,331 / 2,331 | OGB 9원자/3결합 범주 → 공식 11개 구조 특성, MAE |

Peptides의 3D 좌표·거리·정답은 PE 입력으로 사용하지 않는다. 공식 배포 y와 그 정규화는
그대로 보존한다. 같은 데이터셋을 쓴다는 사실이 원 논문의 모든 학습 조건을 재현했다는
뜻은 아니므로, 외부 논문 수치와 비교할 때 입력·모델 크기·학습 조건 차이를 함께 기록한다.

기본 모델 `cycle_set`은 다음 구성이다.

- **PE:** 기존 BFS fundamental basis와 `cycle_set_statistics`의 여섯 통계,
  GELU MLP를 재사용한다. 기저 column 부호/순열에는 불변이지만 BFS chart 자체의 변경에는
  불변이 아니다. 전체 raw basis를 손실 없이 전달하는 codec이라고 주장하지 않는다.
- **메시지 전달:** 이 트랙에 이미 있던 `paper_model._MessageLayer`를 그대로 재사용한다.
  cycle PE를 원래 결합 특징의 embedding에 붙여 엣지에 넣고, 대칭 엣지 업데이트·양방향
  메시지·degree-normalized 집계·LayerNorm을 적용한다. 별도의 GINE 경쟁 모델을
  이름만 바꾼 실행이 아니다.
- **예측:** 노드 mean/max와 엣지 mean/max를 모은 뒤 graph MLP로 목표값을 예측한다.
  PE만으로 회귀값이 나오는 것이 아니므로 이 downstream predictor는 우리 실험의 일부다.
- **기본 규모:** hidden 64, PE 32차원, 메시지 전달 3 layers. 전체 trainable parameter
  500,000 상한을 검사하고 실제 수를 기록한다.

Adam(lr=0.001), MAE loss, 최대 300 epochs, validation plateau LR 감소,
50-epoch early stopping을 사용한다. test는 best-validation checkpoint를 선택한 뒤
한 번만 평가한다. AMP 기본값은 off다. `--amp`를 켜도 메시지 집계는 FP32로 유지한다.
기본 CLI에 경쟁 모델 선택 옵션이나 경쟁 모델 학습 loop는 없다.

데이터 준비는 공식 원본과 **우리 cycle PE만** 계산하여 저장한다. 불필요한 Laplacian
고유벡터·random-walk PE·random probe 계산은 하지 않는다. dense `m×m` projector도
기본 경로에 없다. 원래 incidence/basis 계산 비용은 남는다.
새 전처리 cache는 `data/paper/cycle_pe_benchmark` 아래 별도 version으로 저장하여,
이전 비교용 cache와 혼용하지 않는다.

결과는 `manifest.json`, `metrics.json`,
`<dataset>/cycle_set/history.json`, `<dataset>/cycle_set/best.pt`에 저장된다.
metric schema는 v2이며 `datasets.<dataset>.models.cycle_set`에 우리 결과만 들어간다.
공식 split 내용 SHA256, 구현 SHA256, 실제 parameter 수, GPU peak memory와 runtime을
기록한다. 준비만 한 경우 status는 `prepared`이며 학습 성공으로 처리하지 않는다.

데이터 연결 근거: [SignNet 원 논문](https://arxiv.org/abs/2202.13013),
[PEARL 원 논문](https://arxiv.org/abs/2502.01122),
[LRGB](https://github.com/vijaydwivedi75/lrgb).

Alchemy는 기본에 넣지 않았다. 조사한 공식 SignNet Alchemy index 파일에 중복 및 split 간
겹침이 있어, 임의로 split을 재생성하고 동일 프로토콜이라고 부를 수 없다. 공식 출처와
정규화/중복 처리 정책을 별도 확정하기 전에는 blocked optional로 남긴다.

## 선택적 보조 실험: 기존 raw/set/projector 및 구조 진단

아래 `research.cycle_pe.paper`의 `core`, `brec`, `zinc` suite는 명시적으로 요청할 때만
사용한다. 우리 모델의 내부 표현 및 구조 진단용이며 기본 공개 데이터 실험을 대체하지 않는다.

### 기존 구현 경계와 PE 비교군

모든 PE는 edge-by-node incidence matrix와 BFS spanning tree로 만든 fundamental cycle
basis에서 학습 전에 계산된다. 보조 CLI 기본값은 `raw,set,projector`다.

- `no_pe`: 명시적으로 요청할 때만 실행하는 topology PE 제거 ablation
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

외부 다운로드가 없는 deterministic scientific generator이며 총 20,000개 graph를 사용한다.

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
seed를 허용하며, 집계는 `custom_pairwise_union`으로만 기록한다. Custom은 명시적으로
요청해야 하며 실제로 제공한 BREC artifact를 사용한다. 기본은 항상 official이다.

기본 데이터 위치는 `<data-root>/BREC/Data/raw/brec_v3.npy`다. 파일이 없으면 fail-closed이며,
`--allow-download`를 명시했을 때만 GraphPKU Release ZIP을 HTTPS로 받아 경로, symlink,
압축/추출 크기를 검사하고 `brec_v3.npy` 하나만 추출한다. Official mode는 q=32,
400 pairs, 51,200 records를 강제한다. ZIP과 NPY SHA-256은 provenance로 기록하지만,
GraphPKU가 canonical SHA-256 pin을 배포했다고 주장하지 않는다. 누락된 BREC artifact를
가짜 graph로 대체하거나 자동 생성하는 경로는 없다.

### `zinc`: ZINC-12K

PyTorch Geometric의 `ZINC(subset=True)` official train/val/test 10,000/1,000/1,000 split과
atom/bond feature를 사용한다. local cache가 없으면 fail-closed이고 `--allow-download`를
명시해야 PyG download를 허용한다. Official split별 graph 수를 검증하고 전체를 읽는다.
PyG import가 없거나 CUDA/PyTorch 조합이
맞지 않으면 설치 문서를 포함한 actionable error로 종료한다.

### 현재 과학적·scaling 한계

- `B^Tq`는 circulation component를 잃지만, simple unweighted graph의 full `L=D-A`는 adjacency와
  topology를 결정한다. 이 PE는 sample circulation 복구가 아니라 topology inductive bias다.
- Degree/`(n,m,beta)`-matched 또는 1-WL-indistinguishable known-contrast가 없고, raw/set의
  isomorphic relabeling·spanning-tree shift robustness도 core/ZINC에서 평가하지 않았다.
- 다른 논문의 결과는 원 출처와 데이터 split·입력·모델 규모·학습 조건 차이를 표시한
  외부 비교표에서 다룬다. 이 코드에서 경쟁 모델을 별도로 학습하지 않는다.
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
  `100,200,...,1000`이 별도 protocol axis이며, 바깥 model seed 기본값 변경의 영향을 받지 않는다.

각 suite manifest의 `seed_axes`는 resolve된 네 값을, `seed_axis_policy`는 실제 사용 여부와
`not_applicable` 사유를 기록한다.

## 개별 프로토콜 설정

재현 스크립트의 `--cycle-variants`, `--cycle-core-targets`로 비교군과 target을 선택할 수 있다.
`--cycle-epochs`, `--cycle-learning-rate`는 CycleCount와 ZINC에만 적용되며 공식 BREC의
고정 학습 설정을 바꾸지 않는다. 기본값을 변경한 결과는 기준 재현 실험과 구분해서 기록한다.

하위 `research.cycle_pe.paper` 모듈은 `core`, `brec`, `zinc`, `all` suite와
`--epochs`, `--learning-rate`, `--weight-decay`, `--hidden-dim`, `--pe-dim`, `--layers`를
제공한다. 이 단일 실행 모듈을 직접 호출하는 것은 재현 스크립트의 seed sweep과 다르다.
각 `--output-dir`는 새 경로이거나 비어 있어야 하며 기존 artifact를 덮어쓰지 않는다.
모듈의 `all` 실행은 core → BREC → ZINC 순서로 진행하고, 실패한 suite의 불완전 artifact만
정리한다. 완료된 suite는 보존하고 `run_manifest.json`에 완료 suite hash와 실패 원인을 남긴다.

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

`research.cycle_pe.paper`는 variable-beta graph를 고정 cap 없이 처리한다.
