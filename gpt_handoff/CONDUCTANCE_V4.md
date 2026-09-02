# Conductance V4 — 통합 문서

> 이 문서는 [GPT 전체 프로젝트 전달 안내](README_FIRST.md)에 포함된 V4 설계·실행 문서다.
> V4만이 아니라 다른 모든 트랙과 전체 실험 상태는 같은 전달 폴더의 `HANDOFF.md`와
> `EXPERIMENT_STATUS.md`에서 함께 검토한다.

> V1~V4를 모두 포함해 더 넓고 깊은 모델을 실행하는 별도 계약은
> [RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md)를 따른다. 아래 20-job 기본
> 실험을 큰 모델 결과로 재분류하지 않는다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 구현 | 완료 |
| 로컬 V4 전용 검사 | **131 passed** |
| 저장소 전체 회귀 | **1418 passed / 77 skipped** (80.24 s, exit 0) |
| Ruff·compileall·코드 스냅샷 검사 | 통과 |
| 공식 데이터 CUDA 학습 | **정식 결과 없음** — 과거 arxiv-only 4-arm run에서 첫 arm만 200 epochs·child exit 0 후 구 report gate 중단, 나머지 3개 pending |
| 기본 실행 대상 | Cora/CiteSeer/PubMed/PPI/ogbn-arxiv, model seed 0, 20번의 fresh training |

로컬 검사는 작은 CPU fixture로 수학·미분·runner·report 무결성을 확인한 것이다. V2 전용은
118 passed, V3 전용은 141 passed / 2 skipped이며 전체 검사는 `PYTHONUTF8=1`에서 실행했다.
이는 공식 데이터 GPU 학습 결과가 아니다. 2026-09-02
사용자 보고의 source/pull revision은 `7b4cd32`, 과거 arxiv-only run은
`gat-hybrid-c-spatial-v4-gpu6-seed0-v1`이다. Preflight GPU는
`NVIDIA A100-SXM4-80GB MIG 1g.10gb`였고 `CUDA_VISIBLE_DEVICES=6`을 프로세스 내부
`cuda:0`으로 사용했다. 이 partial run에서 `fixed_c_identity_w`가 200 epochs를 마치고 child
exit 0을 반환했지만, 당시 mean-C/C=1 수치 검사를 hard gate로 쓰던 report가 실행을 중단해
나머지 세 arm은 pending으로 남았다. 점수와 전체 artifact는 수령하지 않았으며, 이 한 arm을
정식 V4 결과로 보지 않는다. 확대된 기본 실험의 실제 성능은 새 run에서 20개 CUDA 학습이
모두 끝난 뒤에만 기록한다.

## 정확한 V4 아이디어

V3 자체를 spectral GNN이라고 분류하는 실험이 아니다. 여기서 말하는 spectral-like 학습은
노드 상태에서 만든 양의 conductance `C(H)`가 weighted adjacency와 degree, 즉 그래프 연산자
자체를 정한다는 뜻이다. V4는 이 경로에 spatial GNN처럼 이웃 message의 feature channel을
변환하는 층별 공유 행렬 `W`를 추가한다.

- `C`: 어떤 이웃 연결을 얼마나 강하게 전달할지 학습한다.
- `W`: 이웃에게 전달할 feature channel을 어떻게 섞을지 학습한다.
- `alpha`: 원래 상태와 이웃 전파를 얼마나 섞을지 학습한다.

중요하게도 `C`는 `W` 적용 전 상태에서 계산한다. 두 메커니즘을 섞어 정의하지 않는다.

## 층 수식

물리 엣지의 conductance를 `C`, weighted adjacency를 `A_C`, weighted degree를 `D_C`라 하면

\[
P_C=D_C^{-1/2}A_CD_C^{-1/2},\qquad \alpha=\sigma(a).
\]

층 입력을 `H`, spatial message를 `M=HW`라 할 때 비고립 노드는

\[
H'=(1-\alpha)H+\alpha P_C(HW)
\]

로 갱신한다. 고립 노드는 정확히 `H`를 유지한다. `W`는 bias 없는 층별 정사각 행렬이며
identity로 초기화한다. 따라서 `W=I`이면

\[
H'=(1-\alpha)H+\alpha P_C(H)
\]

가 되어 V3의 대칭 conductance 전파와 정확히 같아진다.

상대 C 생성기는 V3와 같은 graph-wise score 중심화, 양수화와 `mean(C)=1` 정규화를 사용한다.
weighted degree의 C 의존성까지 미분하며, dense adjacency·incidence·Laplacian·고유벡터 행렬을
만들지 않고 모든 물리 엣지를 정확히 chunk 처리한다.

## 네 학습 조건

| 조건 | C 경로 | Spatial W 경로 |
|---|---|---|
| `fixed_c_identity_w` | 정확히 C=1, 생성기 동결 | 정확히 W=I, 동결 |
| `relative_c_identity_w` | 상대 C 생성기 학습 | 정확히 W=I, 동결 |
| `fixed_c_spatial_w` | 정확히 C=1, 생성기 동결 | W=I에서 시작해 학습 |
| `relative_c_spatial_w` | 상대 C 생성기 학습 | W=I에서 시작해 학습 |

네 조건 모두 다음 계약을 지킨다.

- 전체 model state의 이름·shape·초기값 hash가 같다.
- `alpha`는 모든 층과 모든 조건에서 학습한다.
- 비활성 C/W scaffold는 동결하고 optimizer에서 제외한다.
- 같은 공식 cache·split·topology와 model seed 0을 사용한다.
- 네 조건을 각각 처음부터 새로 학습한다.
- V3 checkpoint나 점수를 재사용하지 않는다.
- train label로 학습하고 validation으로 checkpoint를 선택한다.
- test label과 test metric은 학습·선택·진단에서 사용하지 않는다.

공통 모델 설정은 hidden 64, 2층, dropout 0.5, 최대 200 epochs, patience 50, FP32다.
AMP·compile·TF32는 비활성이다.

## 실행

지원 환경은 Linux, NVIDIA GPU, 활성화된 전용 Conda 환경이다. 공식 데이터 cache가 아직
없다면 먼저 저장소 루트에서 준비한다.

```bash
conda activate new-gat
bash scripts/setup_gpu.sh
bash scripts/prepare_data.sh
```

기본 V4 실험은 다음 한 명령으로 실행한다.

```bash
bash research/conductance_gat/v4/reproduce.sh --run-id gat-hybrid-c-spatial-v4-seed0-v1
```

이 명령은 **Cora/CiteSeer/PubMed/PPI/ogbn-arxiv × 네 조건 × seed 0 = 20번의 새 CUDA 학습**을
순서대로 수행한다.
학습 중 패키지를 설치하거나 데이터를 다운로드하지 않는다. CUDA가 없거나 cache가 검증되지
않으면 결과를 만들기 전에 중단한다. 같은 run ID를 덮어쓰거나 자동 재개하지 않는다.

실행 계획만 확인하려면 다음 dry-run을 사용한다.

```bash
python -B scripts/run_conductance_v4.py \
  --dry-run \
  --run-id gat-hybrid-c-spatial-v4-seed0-v1
```

기본 범위 중 Cora, CiteSeer, PubMed만 명시적으로 선택할 수도 있다.

```bash
bash research/conductance_gat/v4/reproduce.sh \
  --datasets cora citeseer pubmed \
  --run-id gat-hybrid-c-spatial-v4-citations-seed0-v1
```

Cora/CiteSeer/PubMed/ogbn-arxiv는 하나의 고정 그래프에서 train/validation 노드를 나누는
transductive full-graph 실험이다. PPI는 v1과 같은 공식 20/2/2 inductive graph split,
batch size 2와 BCEWithLogitsLoss를 사용한다. `logit > 0`을 양성 예측으로 하여 validation graph
두 개 전체의 node-label 결정을 합친 global micro-F1로 checkpoint를 선택한다. 모든 dataset의
workers는 0이며 PPI도 neighbor sampling이 아니라 각 minibatch의 graph 전체를 처리한다.
Test graph는 train/validation loader, forward, loss, metric, checkpoint 선택과 진단에 들어가지
않는다. 다만 full cache의 test tensor와 metadata는 공식 20/2/2 분할·shape·checksum 무결성
검사를 위해 load/validate된다.

## 결과 확인

기본 실행 결과는 다음 파일 하나에서 먼저 확인한다.

```bash
cat results/conductance_gat/v4/gat-hybrid-c-spatial-v4-seed0-v1/comparison.md
```

전체 결과 폴더는 `results/conductance_gat/v4/<run-id>/`이며 구조는 다음과 같다.

| 경로 | 내용 |
|---|---|
| `comparison.md` | 사람이 읽는 최종 네 조건 비교·진단·개입 보고서 |
| `comparison.csv` | 네 조건과 factorial 대조의 표 형식 결과 |
| `comparison.json` | 검증 가능한 전체 구조화 결과 |
| `manifest.json` | source hash, 실행 설정, 선택 dataset × 네 조건의 job 상태; 기본은 20 jobs |
| `logs/` | GPU 사전검사와 조건별 학습 로그 |
| `<dataset>/<condition>/best.pt` | validation-best checkpoint |
| `<dataset>/<condition>/history.json` | epoch별 train/validation과 실제 task-gradient 진단 |
| `<dataset>/<condition>/metrics.json` | 설정·graph binding·지표·hash·시간·메모리 |

선택한 모든 dataset의 네 fresh 학습, 즉 기본 20 jobs가 모두 통과하고
source/cache/topology/configuration/초기-state/metrics hash가
일치하기 전에는 최종 factorial 대조를 공개하지 않는다. checkpoint 개입의 informational
mean-C/C=1 수치 차이는 이 성공 조건에 포함하지 않는다. standalone report 재생성도 현재
source hash를 다시 확인한다.

## 비교값 해석

위 표 순서의 dataset별 validation 지표(일반 데이터 accuracy, PPI global micro-F1)를
`y00`, `y10`, `y01`, `y11`이라 두면 V4 내부에서만
다음 다섯 값을 계산한다.

\[
\begin{aligned}
C\mid W_{off}&=y10-y00, & C\mid W_{on}&=y11-y01,\\
W\mid C_{fixed}&=y01-y00, & W\mid C_{relative}&=y11-y10,\\
interaction&=y11-y10-y01+y00.
\end{aligned}
\]

이 값은 seed 0의 validation 기반 기술적 대조다. CI·p-value·일반적 인과효과·SOTA 또는
최적 설정을 뜻하지 않는다. V3↔V4 전체 점수 차이도 여러 변경이 함께 들어가므로 한 요인의
효과라고 해석하지 않는다.

## 기록되는 진단과 개입

매 epoch의 실제 train forward/backward에서 다음을 기록한다.

- 층별 score/C/log-C 분포와 weighted degree
- `alpha`, `gamma`, `tau`
- C 생성기와 W의 parameter norm 및 실제 task-gradient norm
- `W-I` Frobenius norm과 W singular values
- epoch 시간, 전체 선택 시간, 후처리 시간
- peak CUDA allocated/reserved memory

선택된 best checkpoint에서는 재학습 없이 validation에서 다음 개입을 모든 층에 동시에 적용한다.

- 그래프별 평균 C
- 그래프 내부에서 고정 seed로 섞은 C
- C=1
- W=I
- C=1과 W=I 동시 적용
- conductance propagation off

C를 바꾸는 개입은 바뀐 C로 weighted degree를 다시 계산한다. 대칭 정규화에서는 양의
graph-constant C가 정확히 소거되므로 mean-C와 C=1은 **대수적으로 중복된 같은 개입**이며,
독립 효과나 별도 성공 조건이 아니다. 구현은 두 개입을 CUDA에서 각각 forward하여 logit 차이와
`allclose_rtol=1e-5`, `allclose_atol=1e-6`, `within_declared_tolerance`를 기록하지만, 이는
CUDA scatter·부동소수점 합산 차이를 관찰하기 위한 **informational, non-gating** 진단이다.
허용오차 초과만으로 arm·report·run을 실패시키지 않는다. 이 개입들은 선택 checkpoint가 각
메커니즘에 얼마나 의존하는지 볼 뿐, 반드시 모두 완료해야 하는 네 fresh-training 조건의
차이를 대신하지 않는다.

## 구현 파일

| 파일 | 역할 |
|---|---|
| [`research/conductance_gat/v4/model.py`](CODE_SUMMARY.md) | `C(H_pre-W)` 생성기, identity 초기 W, V4 모델 |
| [`research/conductance_gat/v4/operator.py`](CODE_SUMMARY.md) | `P_C(HW)`의 정확한 chunked forward/backward |
| [`research/conductance_gat/v4/train.py`](CODE_SUMMARY.md) | fresh CUDA 학습·checkpoint 선택 |
| [`research/conductance_gat/v4/diagnostics.py`](CODE_SUMMARY.md) | 실제 task-gradient 진단과 checkpoint 개입 |
| [`research/conductance_gat/v4/report.py`](CODE_SUMMARY.md) | fail-closed 네 조건 비교 보고서 |
| [`research/conductance_gat/v4/protocol.py`](CODE_SUMMARY.md) | 고정 프로토콜과 네 조건 정의 |
| [`scripts/run_conductance_v4.py`](CODE_SUMMARY.md) | GPU 사전검사와 dataset별 네 조건, 기본 20-job orchestration |
| [`research/conductance_gat/v4/reproduce.sh`](CODE_SUMMARY.md) | 최상위 재현 명령 |
| [`tests/test_conductance_v4_core.py`](CODE_SUMMARY.md) | 수식·미분·W=I 동치 검사 |
| [`tests/test_conductance_v4_runner.py`](CODE_SUMMARY.md) | dataset별 네 조건·기본 20-job·source·실행 무결성 검사 |
| [`tests/test_conductance_v4_report.py`](CODE_SUMMARY.md) | report·contrast·개입 fail-closed 검사 |

## 현재 남은 한 가지 작업

지원되는 Linux NVIDIA GPU 환경에서 **새 run ID로 5개 데이터의 네 조건, 총 20개 arm을 모두
fresh 실행**하고 생성된 `comparison.md/csv/json`, `manifest.json`, 각 조건의 `metrics.json`을
보존하는 것이다. 구 report gate에서 끝난 arxiv partial run의 첫 arm은 새 2×2 대조에 재사용하지
않는다. 전체 20개 arm이 모두 완료되기 전까지 확대된 기본 V4의 정식 성능 수치는 없다.
