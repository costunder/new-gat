# Conductance GAT V5 — graph-specific C optimization and weighted-Laplacian propagation

## 2026-09-08 추가: 실제 zero gate / forest–chord 선택 실험

사용자의 추가 제안을 `research/conductance_gat/edge_selection/` 및
`scripts/run_v5_edge_selection.py`에 **별도 실험군으로 구현**했다. 기존 V5/Cycle 소스와
체크포인트를 대체하지 않는다. 아래 멀티-C 120회 계획과 이 130회 계획은 서로 다른 실행기다.

각 층에서 기존 양의 C 최적화 출력은 amplitude `r[e,h]>0`로 유지한다.
새 변수 `z[e]`는 feature head들이 공유하는 물리 엣지 선택이고,
`c_eff[e,h]=z[e]*r[e,h]`, `L_h=B_s^T diag(omega*z*r_h) B_s`로 실제 전파된다.
`r`의 보정 가중평균 1은 유지하지만 **z 또는 z*r를 평균 1로 재정규화하지 않는다.**
행 정규화 attention 계수는 활성 이웃에서 합이 1이고 고립 노드의 이웃 질량은 0이다.
최종 task loss가 z 생성기, 양의 r 최적화 파라미터, W, beta까지 함께 업데이트한다.
단, r의 내부 solver는 후보 연결 전체의 기존 양의 목적함수다. z가 포함된 새 내부 최적화
문제를 풀었다고 주장하지 않는다. z는 외부 task/보조 loss로 학습한다.

### 서로 혼동하지 않는 두 비교

| 실험군 | 조건 | 정확한 의미 |
| --- | --- | --- |
| 구조 11조건 | full, forest-only, random/learned/cycle 각각 chord 25/50/75% | 원본 그래프에서 label-free 고정 DFS forest를 보호하며, 각 disjoint graph의 나머지 chord 중 정확히 floor(q*chords)개 선택 |
| corruption 2조건 | task+L0, task+L0+negative auxiliary | 원본 성분 안에 원본 엣지 수의 10%인 nonedge를 추가하고 hard-concrete로 선택; forest 보호나 exact-k 제약은 없음 |

구조 비교의 train/eval forward는 동일한 정확한 k의 binary mask다. 역전파에는
합 제약 logistic relaxation의 **편향된 straight-through gradient**를 사용한다.
hard-concrete를 사용하면서 train에서 정확한 k를 보장한다고 설명하지 않는다.
corruption은 hard-concrete의 reparameterized stochastic gate와 expected L0를 사용한다.
기본 온도 2/3, stretch [-0.1,1.1], L0 계수 1e-4, negative 계수 0 대 1이다.
L0는 그래프별 활성확률 합 → 그래프 평균 → 층 평균이고, negative loss는 그래프별
원본/추가 클래스 균형 BCE다. BCE의 logit은 raw gate log-alpha가 아니라 Pr(z>0)에 대응한다.
이 계수들은 명시적인 시작 실험 설정이지 최적 성능을 확인한 값이 아니다.

추가 엣지는 실제 signed negative conductance가 아니다. 원본=1/추가=0의 정답은
모델 입력에서 분리해 학습 보조 loss와 사후 진단에만 전달한다. train/eval의 추가 엣지는
서로 겹치지 않으며, 원래 다른 연결 성분이나 PPI의 다른 그래프를 연결하지 않는다.
정확한 수를 만들 수 없으면 축소·중복·거짓 샘플로 대체하지 않고 명시적으로 실패한다.
원본·후보·추가 엣지와 출처는 immutable metadata/SHA로 검증한다.

cycle 조건은 DFS fundamental-cycle의 **unsigned edge→cycle→edge scalar context**를
gate score에 연결한다. QR/SVD/고유값 분해, 명시적인 전체 cycle 경로 행렬 없이
tree prefix/subtree 누적으로 처리한다. signed implicit basis는 B^T Z=0 및 adjoint
테스트로 별도 검증한다. 이것은 cycle 선택 문맥이지 새로운 positional encoding이라고
주장하지 않으며 DFS 기저 선택에 의존한다. 효과 없는 B^T Za 출력으로 연결하지 않는다.
DFS와 암시적 cycle 연산은 O(N+E)지만 후보 canonical 정렬과 exact top-k 정렬은
별도 비용이다. 전체 파이프라인을 선형 시간이라고 주장하지 않는다. 긴 경로의 누적 오차를
막기 위해 cycle prefix/subtree는 FP64로 누적한 뒤 입력 dtype으로 복귀한다.

### 규모·실측·재개 및 결과 판정

- 5개 official V1 데이터셋(Cora/CiteSeer/PubMed/PPI/ogbn-arxiv), seed 0 한 개,
  reference(256 hidden/8 layers/8 heads), large(384/12/8)를 그대로 사용한다.
  13조건×5데이터셋×2규모=130회이며 학습 epoch/patience 기준 200/50과
  `reference_updates` 계약을 유지한다. 물리 배치가 달라지면 필요한 전체 epoch 수가
  업데이트 예산에 맞춰 늘 수 있다. 샘플링 법칙·fanout·기존 solver K를 축소하지 않았다.
- 모든 조건은 per-head r, row 정규화, optimized solver, width-scaled 비용,
  joint 학습, beta 초기값 0.5로 맞춘다. random/learned 선택을 서로 다른 backbone
  초기값으로 비교하지 않도록 공통 파라미터 초기 SHA를 확인한다.
- 기본 auto sampling은 arxiv만 기존 cluster 방식이며, citation은 full graph,
  PPI는 cached topology를 결합하는 disjoint graph minibatch다. forest의 보장은 실제
  공급된 B_s 안의 연결성이지 샘플링으로 빠진 전체 그래프 경로의 복구가 아니다.
- 새 실행기는 실제 할당 GPU에서 여러 physical batch/worker 후보의 전체 epoch
  forward/backward/optimizer, 전체 validation, graph/cycle 준비와 자원을 측정한다.
  PPI는 큰 그래프들의 동시 배치도 추가 stress 측정한다. 모든 조건에 안전한 공통값을
  전체 학습 예산의 예상 시간 기준으로 고른다. 교정 모델은 폐기하고 본 학습은 원래 seed로
  새로 시작한다. 로컬 CPU 테스트를 A6000 배치 최적화 실측으로 표시하지 않는다.
- 새 run의 재호출은 검증된 완료 학습을 skip하고, 미완료 last.pt에서 model/optimizer/
  Python·NumPy·CPU·CUDA RNG를 epoch 경계로 복원한다. 소스/설정/데이터/학습 예산
  차이는 거부한다. 기존 run을 이 새 실험으로 resume하거나 기존 점수를 재명명하지 않는다.
  신규 파일도 기존 mechanism 실행기의 전체 소스 스냅샷에 잡히므로 구 실행기의 strict
  resume는 소스 변경으로 거부될 수 있다. 기존 결과 보존과 구 실행기의 resume 허용은 다르다.
- 각 완료 checkpoint에서 전체 validation을 최소 5회 반복해 수치 변동을 기록한다.
  z/r/z*r/alpha/beta 분포, 정확한 0 비율, 원본/추가 선택률, 활성 연결 성분·고립 노드·
  cycle rank, degree별 entropy/effective neighbor/max-alpha를 기록한다. 경로 길이는
  명시적인 seed 고정 32개 landmark→전체 노드 진단이며 all-pairs라고 주장하지 않는다.
- 같은 checkpoint에서 all-gates-open, baseline gate를 고정한 amplitude r=1,
  구조군의 random same-k 개입을 전체 validation labels로 비교한다. corruption 모델은
  추가 엣지 없는 clean validation도 평가한다. 감사는 optimizer나 checkpoint를 수정하지
  않으며 실패해도 완료 학습은 보존한다. test set은 이 실행기에서 평가하지 않는다.
- 정확한 zero mask가 있어도 학습 gradient 경로를 위해 후보 엣지는 계속 계산한다.
  실제 sparse kernel의 엣지 제거/속도 향상을 구현·측정했다고 주장하지 않는다.
  effective-resistance sparsifier는 향후 비교 대상이며 이 실행에 포함하지 않았다.

구현·CPU 검증과 실제 데이터에서의 성능 향상은 별개다. 신규 GPU 교정·전체 학습·전체
평가 결과는 아직 없으며, 최종 로컬 회귀검사 수는 `EXPERIMENT_STATUS.md`에 기록한다.

<a id="multi-c-mechanisms-20260908"></a>

## 2026-09-08: 멀티 C / attention 정규화 / 원인 분리 실험

이 절은 아래 단일 shared-C 설계의 **명시적 확장**이다. 구형 checkpoint와 결과는
삭제하거나 신형 구조로 재명명하지 않는다. 기존 shared/symmetric/linear 기본 경로는
보존했으며, 새 구조는 새 run에서 학습한다. 이 절의 구현은 성능 향상 실험 결과가 아니다.

### 수학과 실제 연결

`--conductance-heads per_head`는 한 층의 각 feature head에 독립적인 비용 metric과
C 최적화 변수를 둔다. C는 `E × heads`, degree는 `N × heads`로 처리한다. 독립 head의
C나 degree를 평균 내지 않으며, GPU에서 Python head 루프 없이 일괄 계산한다.
메시지 함수는 edge-feature 중간곱을 backward에서 재계산하는 sparse/chunked 경로다.
모델 너비·깊이·head 수·학습 K·fanout·전체 데이터를 줄이지 않았다.

각 head의 유효 conductance는 `a_e^h = omega_e c_e^h`이며
`L_h = B^T diag(a^h) B = D_h - A_h`다.

- `symmetric`: 기존 `P_h = D_h^(-1/2) A_h D_h^(-1/2)`.
- `row`: `P_h = D_h^(-1) A_h`, 즉 `alpha_ij^h = a_ij^h / d_i^h`.
  비고립 노드의 이웃 합은 1이며, 각 head가 자기 weighted degree를 쓴다.
- 기본 전파는 `(1-beta_h) H W_h + beta_h P_h H W_h`.
  고립 노드는 자기 메시지를 그대로 유지한다. C와 degree에서 gradient를 끊지 않는다.
- `polynomial3`: 기존 전파에 `a2_h(P_h^2-I) + a3_h(P_h^3-I)`를 더한다.
  `a2=a3=0`으로 시작해 기존 전파와 같고, 두 계수는 실제 optimizer로 학습한다.
  계수 합 1인 3차 필터이며 임의의 모든 다항식 계수를 독립 학습한다고 주장하지 않는다.

최적화 C 원값은 양수이고 1보다 클 수 있다. 그래프·C head별 보정 가중평균을 1로
맞추는 조건은 유지한다. **C를 [0,1]로 자르지 않는다.** 평균 1 제약과 [0,1] 제한을
동시에 두면 모든 C=1만 가능하기 때문이다. 사용자가 확인할 [0,1] 이웃 비중은 row
전파의 alpha이며, raw C·C CV·beta와 별도로 기록한다. 다양성을 강제로 만들도록 loss를
추가하지 않았다. 균일성/집중도는 실제 분포와 task 효과로 판정한다.

`num_relations > 0`인 모델 API는 **실제** `edge_relation_id`(물리 엣지 순서에 대응하는
int64 E개)를 요구하고 관계·head별 quadratic metric을 비용에 연결한다. 샘플링에서도
원본 물리 엣지 ID로 관계를 보존한다. 관계 ID를 라벨/특징으로 임의 생성하지 않는다.
현재 5개 official 데이터셋은 untyped graph이므로 head별 C 비교만 실행한다.
이는 공통 feature 공간을 가진 **무방향 관계 타입 그래프** 지원이지 HGT 전체 구현이나
실제 heterogeneous benchmark 완료가 아니다. 방향성을 요구하는 관계는 명시적으로 거부한다.

### 원인 분리 실험 행렬

새 실행기 `scripts/run_v5_mechanism_experiments.py`의 suite는 다음과 같다.

| Suite | 비교 | 다른 suite와 중복 대조군 처리 |
| --- | --- | --- |
| core | fixed / shared optimized / per-head optimized × symmetric / row | 6조건 |
| generators | fixed / degree-only / shared MLP / shared optimized, symmetric | 추가 2조건 |
| solvers | 기존 K8 / barrier=0 K8 / barrier=0 entropy 정확해 | 추가 2조건 |
| filters | fixed / shared optimized × linear / polynomial3, symmetric | 추가 2조건 |

전체 합집합은 12조건이다. 동일 조건을 suite마다 재학습하지 않는다. reference(256폭,
8층, 8heads)와 large(384폭, 12층, 8heads), Cora/CiteSeer/PubMed/PPI/ogbn-arxiv,
seed 0을 모두 선택하면 **120회 학습**이다. core+generators 기본은 80회다.
3~5개 학습 seed를 자동 추가하지 않는다. 5회 반복은 선택 checkpoint의 **평가 변동 측정**이다.

degree-only는 delta=0으로 현재 entropy·degree barrier·K를 유지하며 C 비용 파라미터를
두지 않는다. W·beta·backbone은 정상 학습한다. entropy-exact는 rho=0에서
`c*=exp(-delta/tau)/mean_omega(exp(-delta/tau))`를 학습과 평가 모두에 사용한다.
MLP는 현재 명시적 shared/untyped 대조군이며 optimization-C를 MLP로 몰래 대체하지 않는다.

모든 비교는 joint, beta 초기값 0.5, 기존 corrected optimizer/dropout과 같은 sampler를
사용한다. 학습 예산은 기본 200 epochs/patience 50의 reference-updates 계약이다.
실측에서 physical batch가 증가하면 기존 update 예산을 보존하도록 epoch 상한과 patience가
명시적으로 증가할 수 있다. 선택 epoch/실제 update 수는 별도로 보고한다.
기본 sampler auto는 arxiv의 기존 cluster, 나머지 full/원래 PPI graph batch다.
새 disjoint sampler는 명시 선택해야 하며 C 비교 중 일부 조건만 sampler를 바꾸지 않는다.

### 공통 GPU 실측과 재개

profile×dataset마다 **요청한 모든 variant**를 같은 physical batch/worker 후보에서 측정한다.
optimizer update·validation을 포함한 측정에서 안전한 공통 후보를 고른 뒤 모든 팔에 적용한다.
모델별로 다른 배치를 골라 비교를 오염시키지 않는다. 초기값은 C 생성기를 제외한 shared
해시와 polynomial 확장까지 제외한 common-backbone 해시로 검증한다. 숫자상 VRAM 점유율을
목표로 삼지 않으며 측정되지 않은 GPU utilization은 원인과 함께 미확인으로 남긴다.

새 결과 경로는 `results/conductance_gat/mechanisms/<run-id>`다. 실행 명령을 그대로
재실행하면 정확히 같은 설정·소스·자원 계획을 검증하고 완료 학습을 건너뛴다.
미완료 학습은 해당 새 run의 last.pt에서 epoch 경계로 재개한다. **구형 shared-C의
optimizer를 신형 per-head 모델에 끼워 넣지 않는다.** 과거 fixed/shared 결과는 참고로
보존하며, 공통 실측/예산/초기화 이력이 다른 결과를 새 paired 대조군으로 자동 재사용하지 않는다.

학습 완료와 감사 완료 상태는 별도다. 감사 실패로 완료 학습을 다시 돌리지 않는다.
재개 시 누락/실패한 감사만 다시 수행한다. 학습 로그·감사 로그·history.json·checkpoint를
각 condition 폴더/manifest에서 추적하며 원본 로그를 덮어쓰지 않는다.

### 분포와 학습 역할 검사

각 선택 checkpoint에 전체 official validation 감사를 실행한다.

- raw C: 정확한 분위수·히스토그램·C>=0.7·abs(C-1)<=0.1·C>1 비율.
- 노드별: alpha 최대 비중·정규화 entropy·유효 이웃 수 `1/sum(alpha^2)`.
  고립 노드와 degree 1을 분리하고 degree 구간별·graph/layer/head별 관측 범위를 표시한다.
- 동일 sampling correction의 C=1 대조, head 간 상대 비중 TV/JS 차이.
  대칭 전파의 실제 kernel 계수와 진단용 행 정규화 비중을 구분한다.
- 같은 checkpoint 무개입 5회로 수치 변동을 측정한다. fixed-C는 학습 C 기여 판정의
  대상이 아니며 단순 max-abs>0으로 `observed`를 표시하지 않는다.
- beta를 head 평균으로 바꾼 평가와 K64 참조 검사를 분리한다. MLP의 solver 비교는
  not-applicable이며, 정확해와 유한 반복을 혼동하지 않는다. K64도 수렴을 보장하지 않는다.
- 선택적 `--head-gradient-conflict`: 실제 validation task loss의 C-space head별
  민감도(VJP)를 측정한다. optimizer를 갱신하지 않으며 theta gradient 충돌과 동일하다고
  주장하지 않는다. 계산 비용 때문에 명시 선택 옵션이다.

새 학습 실행 예시(기존 결과와 다른 run, GPU 3만 노출):

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
/home/aicompetition07/.conda/envs/new-gat/bin/python -B scripts/run_v5_mechanism_experiments.py \
  --run-id multic-v5-a6000-gpu3-seed0-v1 \
  --suites core generators solvers filters \
  --profiles reference large --model-seeds 0 \
  --datasets cora citeseer pubmed ppi ogbn-arxiv \
  --device cuda:0 --hardware-profile a6000-48gb \
  --activation-checkpoint
```

activation checkpointing은 전체 비교에 공통 적용하는 메모리 전략이며 모델/데이터 축소가
아니다. `--dry-run` 추가 시 파일/프로세스/GPU 측정 없이 계획만 출력한다.
이 명령은 구형 학습 재개가 아니라 **승인된 새 구조의 비교 학습**이며, 같은 새 명령의
재실행만 신규 run 재개다. 기존 V1–V4/Cycle/Tree는 수정하거나 재실행하지 않는다.

검증 구분: 최종 전체 pytest **2,953 passed / 107 skipped / 0 failed**(470.06초),
수정 Python 파일 Ruff 통과, Git staged 원문의 학습 소스 25개 지문과 감사 계약 일치,
위 실행 옵션의 dry-run 120개 계획 검증을 완료했다. 로컬 CPU 수치·gradient·optimizer
update·재개·실행 계약 검증이며 작은 합성 입력은 명시적 debug 테스트에만 사용했다.
로컬 PyTorch는 CPU 전용이고 PyG가 없어 실제 PyG 통합 일부는 skip이다. CUDA·Linux 전용
검사도 skip이다. 원격 A6000 실측·실제 데이터의 새 전체 학습·새 test 평가는 미실행이다.
서버 실행기가 수행할 실측을 이미 측정한 것으로 보고하지 않는다. 자동 생성 CODE_SUMMARY.md는
별도 덮어쓰기 승인 요구로 미갱신이며, README_FIRST.md에 해당 스냅샷의 범위를 명시했다.

아래 절들은 변경 전 shared-C 실험과 그 당시 검사 기록이다.

<a id="feedback-implementation-20260908"></a>

## 2026-09-08: 피드백 반영 구현 — 독립 부분 그래프 배치와 C 역할 검사

아래 기존 결과 감사 이후의 **코드 변경**이다. 기존 20조건 결과와 checkpoint는 보존했다.
이를 새 실험 결과로 재해석하거나 성능이 개선됐다고 보고하지 않는다.

### 변경한 학습 경로

`cluster_disjoint`는 physical seed batch와 개별 부분 그래프의 seed context를 분리한다.
한 physical batch 안의 context마다 독립적으로 기존 cluster BFS 규칙을 적용해 B_s를
만든 뒤, 노드·엣지의 독립 복사본들을 PyG disjoint-union batch로 합친다.
각 context의 C·degree·graph 통계·beta는 분리되고, C→가중 라플라시안→메시지 패싱을
한 batched forward/backward로 처리한다. 겹치는 원본 노드라도 context 간 메시지는 섞이지 않는다.
감독 seed는 epoch당 정확히 한 번 사용하며 physical batch당 optimizer update는 한 번이다.
context RNG는 seed/epoch/context 순번에 고정되어 CPU thread 순서에 영향받지 않는다.

- `--v5-sampling auto_disjoint`: arxiv만 새 방식, Cora/CiteSeer/PubMed full graph,
  PPI 원래 그래프 batch를 유지한다. **모든 데이터셋의 부분 B 샘플링 구현으로 주장하지 않는다.**
- `--v5-sampling cluster_disjoint`: 명시한 transductive 데이터에 적용한다. PPI에는 허용하지 않는다.
- `--v5-sample-context-seed-batch-size`는 새 방식에서 필수다. 2,048은 기존 A6000
  기준 seed context를 보존하는 제안값이다. physical 8,192라면 독립 context 4개로 합친다.
  기존 8,192개 seed를 하나의 큰 B로 확장한 계산과는 **다른 샘플링 실험**이다.
- physical batch는 context 크기의 정수배여야 한다. 마지막 epoch 꼬리와 train split 전체를
  담는 자연 경계만 예외다. 샘플 수·fanout·학습 예산을 줄이는 자동 fallback은 없다.
- context 자체가 그래프 크기까지 포화될 수는 있다. 새 모드라는 이유만으로 다양성을
  보장하지 않으며, 실제 context 노드/엣지 수·비율·hash·Jaccard overlap으로 판정한다.

`history.json`과 `performance.json`의 `batch_observations`에 disjoint graph 수와
`sampling_observation`을 함께 기록한다. context별 크기, context 쌍 및 직전 physical batch와의
노드/엣지 overlap을 구분한다. 예전 기록에 overlap이 없으면 unavailable로 표시한다.

Rich 실행기의 optimizer-inclusive GPU 실측에도 새 방식을 연결했다. physical seed batch
후보와 **CPU sampling context worker** 후보를 별도로 측정하며, loader workers=0과 혼동하지
않는다. 선택한 context worker는 자원 계획에 고정하고 양팔에 공통 적용한다.
같은 context/fanout으로 physical batch를 묶는 정도만 탐색한다. 기존 규모보다 작은 physical
batch로 자동 전환하지 않으며, 안전한 후보가 없으면 실패 원인과 측정을 남긴다.

### 재학습 없는 기존 checkpoint 검사

`scripts/audit_v5_stages.py`는 corrected `optimization/joint/width_scaled/beta_initial=0.5`
checkpoint의 **전체 official validation**을 검사한다. 파일을 쓰거나 optimizer를 호출하지 않는다.

1. learned C / C=1 / mean C / shuffle C의 validation·logit·예측 변화를 비교한다.
   개입한 C로 실제 degree를 다시 계산한다. fresh fixed-C 학습과는 다른 실험이다.
2. 각 층 baseline H/B/ω/W/β를 고정한 채 학습 K와 더 큰 K의 C 및 operator 오차를 비교한다.
   별도로 전체 모델의 K를 늘린 validation도 출력하되, 그때 downstream H는 바뀐다고 명시한다.
3. 참조 계산의 projected-gradient residual·energy·사용자 지정 tolerance 도달 여부,
   기존 history의 C/β/gradient와 sampling 관측을 출력한다.

더 큰 K도 정확해/충분한 수렴으로 가정하지 않는다. execution=passed는 검사 실행 성공이며,
contribution=observed는 C 개입에 출력이 민감하다는 뜻일 뿐 이득·일반화 증명이 아니다.
β=0처럼 C가 달라도 출력이 같으면 inconclusive로 표시한다. β/residual/FFN을 강제 변경하거나
C 분산만 커지도록 loss를 추가하지 않았다. 모델 규모·학습 K=8·τ·ρ·joint gradient 경로는 유지했다.

아래는 기존 서버 결과를 읽는 명령이다. 64회는 **진단용 비교 예산**이며 수렴 보장이나 학습 K 변경이 아니다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/audit_v5_stages.py \
  --root results/conductance_gat/scaling/corrected-c-v5-a6000-gpu3-seed0-v1-conductance \
  --device cuda:0 \
  --reference-steps 64 \
  --reference-tolerance 0.0001
```

기본은 조건·층별 사람이 읽는 stdout이고 `--json`은 전체 수치를 출력한다. 특정 조건만
검사하려면 `--root`를 그 조건 디렉터리로 지정한다. checksum/selected epoch/데이터 split과
정확한 소스 allowlist를 검증한다. 이 읽기 전용 허용은 training resume 권한이 아니다.
test 예측/선택은 하지 않으며 데이터 cache 무결성 확인은 전체 split 메타데이터를 검사한다.
진단 forward 전체를 기존 자원 관측 모듈로 감싸 CPU·RAM·UUID 확인 GPU utilization을
주기적으로 기록한다. GPU utilization은 장치 전체 값이지 이 프로세스만의 값이 아니며,
미확인 값은 원인과 함께 null로 남긴다. 조건별 CUDA peak를 분리하고 실패 시에도
관측 thread를 정리한다. 이 기능을 구현한 것과 실제 서버 값을 수령한 것은 구분한다.

### 기존 실험과 새 설정의 경계

기존 `auto` 기본값은 바꾸지 않았다. 새 B 샘플링을 기존 run-id/checkpoint에 조용히
이어 붙이지 않는다. 동일한 새 설정으로 시작한 run은 기존 epoch checkpoint 규칙으로
재개할 수 있지만 구형 sampling run과의 동등성은 주장하지 않는다.
구형 결과는 위 읽기 전용 검사에 재사용한다. 변경 전 미완료 학습은 원래 소스의 재개 계약을
따르며, 이번 변경을 포괄적인 구소스 training resume 허용으로 등록하지 않았다.

신규 설정의 **무실행 계획 확인** 예시다. 현재 Rich의 V5 전체 grid인 20조건을 출력한다.
완료 조건의 재학습 지시가 아니며, V1–V4/Cycle/Tree는 이 계획에 포함하지 않는다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_rich_scaling.py \
  --run-id disjoint-c-v5-a6000-gpu3-seed0-v1 \
  --tracks conductance --conductance-versions v5 \
  --profiles reference large --model-seeds 0 \
  --device cuda:0 --hardware-profile a6000-48gb --min-free-gb 40 \
  --v5-solver-cost-scaling width_scaled --v5-beta-initial 0.5 \
  --v5-learning-budget-policy reference_updates \
  --v5-sampling auto_disjoint --v5-sample-context-seed-batch-size 2048 \
  --dry-run
```

이 예시에서 dry-run을 제거하면 V5 20조건의 새 학습이 시작되므로, 기존 checkpoint
감사와 새 실험 범위 결정 전에 제거해 실행하지 않는다. 실제 실행 경로에는 먼저 GPU
자원 후보를 실측하는 단계가 연결되어 있다.
검증 범위: 로컬 CPU 수치·회귀·파서 검사를 수행했다. 로컬 CUDA와 PyG가 없어 실제 PyG
integration 일부는 skip이고 A6000 처리량·VRAM, 실제 checkpoint 감사 및 새 전체 학습은
미실행이다. 이 제약을 GPU 검증 완료나 모델 성능 개선으로 바꿔 보고하지 않는다.

최종 로컬 검증(2026-09-08): 전체 pytest **2,836 passed / 106 skipped / 10 warnings**,
215.71초. skip은 CUDA/PyG 미설치, Linux/Bash·symlink 권한 등 해당 환경에서 실행할 수 없는
검사와 기존 opt-in stress 검사다. 경고 10개는 기존 cluster 포화 경고의 회귀 검사다.
변경 Python의 Ruff와 `git diff --check`가 통과했고, `CODE_SUMMARY.md`는 298개 소스와
일치하도록 재생성·확인했다. 기존 재개 registry는 넓히지 않았다. 역사 51da→8da source-map
계약 검사와 실제 현행 소스 거부·checkpoint 바이트 보존 검사를 분리했다.

<a id="c-learning-audit-20260908"></a>

## 2026-09-08: C 학습 구조·spectral 해석·실제 검증 범위

이 절은 소스 `8da06ca`와 corrected V5 수령 결과를 검토한 최신 판정이다.
[전체 결과·문제 목록](EXPERIMENT_STATUS.md#v5-audit-20260908)에 20조건 학습,
3,832개 에포크 원문, dynamic 10조건 test, 비용·일반화 문제와 증거 출처를 모았다.
아래 날짜별 교정·구형 MLP 기록은 유지하지만 현재 결과와 혼합하지 않는다.
**배선/수치 검증은 존재한다. 그러나 실제 데이터에서 C의 유용성·K8 최적화 충분성·
샘플링 확장의 타당성까지 검증 완료라고 말한 것은 과장이었다.**

### 1. 사용자 요구와 실제 두 단계

B는 edge×node의 signed incidence다. 무방향 physical edge마다 방향은 임의로 한 번만 잡는다.
레이어 ℓ의 현재 hidden state를 H라고 쓰면,

\[
B_s=\operatorname{Sample}(B),\qquad
c_{\theta,s}=\operatorname{Solve}_{K=8}(B_s,H_s;\theta),\qquad
C_{\theta,s}=\operatorname{diag}(c_{\theta,s}),
\]
\[
L_{\theta,s}
=B_s^\top\operatorname{diag}(\omega_s\odot c_{\theta,s})B_s,\quad
\mathcal L_{\theta,s}=D_{\theta,s}^{-1/2}L_{\theta,s}D_{\theta,s}^{-1/2},
\]
\[
M_h=(I-\beta_{g,h}\mathcal L_{\theta,s})H_sW_h,\qquad
\widehat Y_s=\operatorname{GNN}_{W}(H_s,\mathcal L_{\theta,s}).
\]

D는 실제 유효 가중치 ωc의 degree다. ω는 sampling 보정이고 full graph에서는 1이다.
고립 노드의 D^{-1/2}는 0, 정규화 L의 해당 행/열은 0이며 메시지의 identity 항은 유지한다.
복수 그래프 batch에서는 graph별 beta와 block-diagonal operator로 해석한다.
각 residual/FFN block이 C 계산→전파를 반복하고, C는 모든 feature head가 공유한다.

\[
\min_{\theta,W,\psi}\;
\mathbb E_s\left[
\mathcal L_{\rm task}\bigl(\widehat Y_s,y_s\bigr)
\right],
\quad
\mathcal L_{\rm task}\to M\to\mathcal L\to L\to c\to\theta.
\]

ψ는 beta/backbone 등 나머지 학습 파라미터를 뜻한다.
두 단계는 계산 역할의 구분이다. 1단계를 완전히 학습한 뒤 고정하는 방식이나
별도 spatial encoder를 새로 추가하라는 요구가 아니다. 샘플러 자체는 고정 규칙이고,
샘플에서 계산되는 C와 degree를 통한 task gradient는 끊지 않는다.

구현 근거(행 번호는 8da06ca 기준):

- `model.py:508–527`: 같은 incidence와 샘플 metadata로 C를 계산하고 실제 diffusion에 전달.
- `operator.py:213–228`: ωc, 그 가중치의 live degree, 양방향 sparse 전파와 beta 혼합.
- `model.py:566–571`: residual/FFN을 포함한 전체 block.
- `train.py:356–384, 825–827, 1644–1676`: C/W/backbone/beta optimizer 그룹,
  joint 활성화 및 task-loss backward/optimizer update.
- `optimization.py:624–627`: 진단 복사본만 detach하고 forward에는 live C 반환.
- 코드 파일들은 `research/conductance_gat/v5/` 아래이며, 원문 스냅샷은
  `gpt_handoff/CODE_SUMMARY.md`에서 확인할 수 있다.

현재 corrected run은 **optimization / joint / width_scaled / beta_initial=0.5**다.
아래 역사적 staged warmup/MLP 또는 beta 0.1 설명을 이 run에 적용하지 않는다.
Fixed-C는 의도적으로 C=1이며 잔여 V5 구조는 유지한다. 따라서 vanilla GCN과 동일하지 않다.

### 2. C는 무엇을 학습하며 왜 입력마다 다시 계산하는가

학습되는 공유 θ는 node projection, graph-context metric, structure metric이다.
기존 edge ID마다 영구 파라미터 하나를 저장하는 형태는 아니다.
매 forward에서 c=1로 내부 반복을 시작하지만 학습된 θ까지 초기화하지 않는다.
현재 입력·연결구조에 맞는 c를 산출하도록 **내부 최적화 과정을 통해 θ를 학습**한다.
이는 endpoint MLP가 C를 한 번 바로 출력하던 구형 backend와 다르다.
다만 현행에도 비용을 만드는 학습 가능한 선형 projection/metric은 존재한다.

그래프 하나에 대해 S=Σ_e ω_e, n₊는 degree 양수 노드 수,
d_i(c)=Σ_{e∋i}ω_ec_e, d_i(ref)=Σ_{e∋i}ω_e로 놓으면,

\[
E_g(c;\delta_\theta)=
\frac1S\sum_e\omega_e
\left[c_e\delta_{\theta,e}
+\tau(c_e\log c_e-c_e+1)\right]
-\frac{\rho}{n_+}\sum_{i:d_i(ref)>0}
\log\frac{d_i(c)}{d_i(ref)},
\]
\[
c_e>0,\qquad \frac1S\sum_e\omega_ec_e=1.
\]

실제 cost는 projection p_i=L2Normalize(AθH_i), signed graph metric
m_g=tanh(Mθ LayerNorm(z_g)), 8차원 edge structure s_e를 사용한다.
s_e는 sample/full degree의 log1p 합·절대차 4개, 끝점별 sample/full coverage 2개,
full degree 역수 2개를 모아 tanh한 벡터다. 끝점 쌍의 coverage/역수는 정렬해 방향에
의존하지 않도록 한다. 아래 aθᵀs_e는 이 tanh 이후의 구조 특징에 대한 선형 비용이다.

\[
r_e=\sqrt{d}\sum_k m_{g,k}(p_{u,k}-p_{v,k})^2+a_\theta^\top s_e,
\qquad
\delta_{\theta,e}=2\tanh\left(\frac{r_e-\overline r_g^{\,\omega}}2\right).
\]

이는 positive-distance smoothness만 강제하는 cost가 아니다. Signed metric은
feature 관계에 따라 연결을 선호/억제할 수 있다. Graph 전체의 weighted mean으로
중심화하며 memory chunk별로 서로 다른 중심을 쓰지 않는다.

현재 τ=1, ρ=0.1, cost bound=2, K=8이다.
KL-prox 반복의 step upper bound는 0.25이고 curvature/displacement에 따라 줄어든다.
내부 에너지는 C 문제의 목적함수이며, 외부 θ/W는 **최종 task loss**로 갱신된다.
K-step unroll의 gradient이지 완전히 수렴한 argmin의 implicit gradient라고 주장하지 않는다.

평균 C=1은 공통 배율의 gauge를 고정한다. 정규화 전파에서 C 전체를 같은 양수로
배율 변경하면 degree 정규화가 상쇄하므로, 이 gauge 자체를 C 학습을 막는 버그라고 할 수 없다.
반면 entropy τ=1과 bounded cost/metric, finite K는 상대 C 대비와 표현력에 영향을 주는
실제 설계 제약이다. 이것들이 **실제 성능 부진의 원인인지는 아직 확인되지 않았다.**

코드는 C 양수/유한, energy/residual 유한 및 final energy가 initial energy보다
허용 오차 밖으로 증가하지 않는지 검사한다. **실제 K8 residual이 task에 충분히 작은지**,
C 개입 효과가 충분한지에 대한 통과 기준은 없다.
고정 δ, τ>0의 내부 문제는 convex 구조지만 K=8 실행만으로 정확 최적해를 보장하지 않는다.

### 3. Spectral과 spatial의 관계: 현재 무엇이 같은가

한 forward의 C를 고정해 해석하면 정규화 L은
\(\mathcal L_C=U\Lambda U^\top\)이고 각 head 전파는

\[
(I-\beta_h\mathcal L_C)V_h
=U(I-\beta_h\Lambda)U^\top V_h,\qquad V_h=HW_h.
\]

따라서 현재의 **선형 전파 부분**은 1차 spectral polynomial filter를 sparse message passing으로
계산한 것과 같다. 전체 모델은 C가 H에 의존하고 beta·비선형·residual/FFN을 포함하므로
하나의 고정 spectral filter라고 부르면 부정확하다.
C 학습은 L의 edge weight와 그에 따른 spectrum/eigenbasis를 바꾸는 것이며,
고정 L에서 필터 계수만 학습하는 방식과도 구별한다.

고유값분해를 매번 실행해야만 spectral 필터인 것은 아니다.
다항식 기반 localized spectral filtering의 근거는
[Defferrard et al., 2016](https://arxiv.org/abs/1606.09375)을 참고한다.
그래프 edge weight 자체를 신호로 학습하는 선행 예는
[Kalofolias, 2016](https://proceedings.mlr.press/v51/kalofolias16.html)이다.
현행의 signed learned cost/entropy/task unroll이 해당 논문의 정확한 재구현이라는 뜻은 아니다.
논문 존재가 현재 C의 학습 효과나 설계의 신규성을 입증하지도 않는다.
이 문서화에서 production에 dense eigendecomposition/QR/SVD를 추가하지 않았다.

### 4. 이미 존재하는 검증과 실제 범위

아래는 테스트 본문 확인이다. 이번 문서 작업에서 전체 suite를 새로 실행한 것은 아니다.
8da06ca의 기존 로컬 기록은 2704 passed / 103 skipped / 10 warnings다.
CPU synthetic/수치/재개 검증을 실제 데이터 A6000 학습 성공으로 보고하지 않는다.

| 검증 | 코드 근거 | 확인하는 것 / 한계 |
|---|---|---|
| 독립 dense adjacency와 sparse 전파 | tests/test_conductance_v5_diffusion_memory.py:25,72 | 출력 및 message/C/beta/correction gradient; disjoint graph·isolate·chunk 포함 |
| 1·2차 수치 미분 및 AMP geometry | 같은 파일 :100,132 | gradcheck/gradgradcheck, BF16 아래 FP32 geometry; 실제 학습 성적 검증 아님 |
| 내부 E와 analytic C gradient | tests/test_conductance_v5_optimization.py:167 | dense unsigned-incidence/autograd 기준식과 일치 |
| Task loss→C 파라미터→update | optimization.py 테스트 :204; optimization_integration.py 테스트 :170 | 실제 optimizer 경로; 해당 기본 integration fixture는 legacy_unit |
| 현재 width_scaled/K8 task gradient | tests/test_conductance_v5_cost_scaling.py:176 | 모든 C 파라미터의 finite nonzero task gradient |
| 방향·순열·graph 독립·ω 공통 배율 | 같은 파일 :125 | 현재 width_scaled의 equivariance/invariance; unseen-topology 학습 효과는 아님 |
| 비용 스케일의 수치 미분 | 같은 파일 :201 | K3 gradcheck, K8 실제 데이터 수렴 검사가 아님 |
| Iteration 증가 시 E 감소 | tests/test_conductance_v5_optimization.py:268 | synthetic legacy_unit, K1/2/4/8/16/32, 마지막 residual 감소; production K8 충분성 아님 |
| Analytic entropy 최적해 | optimization.py 테스트 :402; cost_scaling.py 테스트 :214 | ρ=0/K128에서 closed-form과 일치; 현재 ρ=.1/K8의 보증 아님 |
| Width 256/384 초기 cost 대비 | tests/test_conductance_v5_cost_scaling.py:250 | 초기 synthetic 대비 교정; 학습 완료 C의 대비/정확도 검증 아님 |
| 성능 패치 전후 수식·gradient | tests/test_conductance_v5_single_graph_reductions.py:180 | 두 cost mode/K8/여러 barrier·dtype의 C 및 모든 활성 gradient 동등성 |
| 샘플 구조·seed/RNG 보존 | tests/test_v5_sampling_performance.py:45; tests/test_v5_static_graph_runtime.py:140 | 유효 induced physical edges 및 성능 수정 전후 샘플 보존; full-graph 근사 보증 아님 |

경로가 축약된 optimization/integration/cost_scaling 테스트는 모두
`tests/test_conductance_v5_*.py`를 뜻한다.
기존 수렴 테스트가 있으므로 '수렴 검증을 전혀 안 했다'는 설명도 잘못이다.
정확한 한계는 **현재 설정·실제 그래프에서 유용성까지 입증한 검증이 부족하다**는 것이다.

### 5. 추가로 실행한 읽기 전용 CPU 수치 점검

앞선 감사에서 synthetic CPU float64로 한 번 실행했다. 신규 학습/실제 데이터 실험이나
지속적인 regression test 추가는 아니다. Random seed 741, CPU threads 2,
8 nodes(삼각형·경로·분리 edge·isolate), 6 edges,
width=256, optimization/width_scaled/K8/τ1/ρ.1,
비균일 ω=[1,1.4,.8,1.2,1,1.7], beta=[.3,.7] 조건이다.

Signed B를 직접 만들고 \(B^\top\operatorname{diag}(\omega c)B\),
정규화 dense 연산, 실제 sparse 전파, eigendecomposition 기준 전파를 비교했다.
C를 생성하는 파라미터까지 포함한 gradient 비교는 **dense BᵀCB 경로 대 실제 sparse 경로**다.
Eigendecomposition은 detach한 forward 기준이므로 EVD 자체의 gradient를 검사한 것이 아니다.

| 관측 | 값 |
|---|---:|
| Sparse 대 dense BᵀCB 전파 최대 절대 오차 | 3.3306690738754696e-16 |
| Sparse 대 eigen-filter forward 최대 절대 오차 | 1.3322676295501878e-15 |
| Dense/sparse 모든 활성 gradient 최대 절대 오차 | 4.718447854656915e-16 |
| L 대칭 오차 / L·1 오차 | 0 / 0 |
| 정규화 L 최소 / 최대 고유값 | −4.163336342344337e-17 / 2 |
| C 파라미터의 task gradient | 모두 finite·nonzero |
| Projected residual 초기 → K8 후 | 0.5954043678306257 → 0.07066408603804239 |

최소 고유값의 미소 음수는 float64 반올림 수준이다.
이 결과는 해당 입력의 수학 연결과 역전파를 뒷받침한다.
**실제 데이터의 K8 수렴 완료, C의 성능 이득, sample/full 일치, A6000 성능 측정은 아니다.**
이번 문서에만 기존 점검 결과를 보존하며 원본 checkpoint나 학습 코드를 바꾸지 않았다.

### 6. Spatial 샘플링 확장의 구현 범위와 검증 공백

실제 sampler는 원래 physical edge 중 양 끝 노드가 샘플에 들어온 induced edge만 선택한다.
동일한 B_s가 C solver와 전파를 함께 구동한다. 없는 연결을 임의로 만드는 방식은 아니다.

\[
\omega_{uv}=
\operatorname{clip}_{[1,64]}
\sqrt{\frac{d_u^{full}}{\max(d_u^s,1)}
      \frac{d_v^{full}}{\max(d_v^s,1)}}.
\]

`sampling.py:283–311`의 같은 ω가 C 목적함수/gauge와 실제 연산자에 쓰인다.
앞의 사용은 **최적화 문제를 정의**하고 뒤의 사용은 **전파 가중치를 정의**하므로
우연히 ω²을 메시지에 곱하는 코드라는 뜻은 아니다.
하지만 edge inclusion probability의 역수가 아니며 `sampling.py:341`도 근사라고 명시한다.
정규화 및 context-dependent C가 비선형이므로 raw edge sum에 관한 가정만으로
normalized operator나 task gradient의 unbiasedness가 따라오지 않는다.

현재 `auto`는 arxiv만 cluster다. Citation은 full graph이고 PPI는 원래 그래프 batch다.
전체 train seeds는 epoch마다 사용되지만 **여러 종류의 부분 연결구조를 모든 데이터셋에서
학습했다는 요구 충족으로 확대할 수 없다.**

특히 arxiv reference의 저장 seed batch는 8192, fanouts=[15,10]이고 cluster budget은
8192×(1+15+10)=212992 > 원래 169343 nodes다.
`sampling.py:207–211`의 포화 경로에서는 supervised seeds가 속한 연결 성분 전체를 선택한다.
따라서 대부분 full batch가 거대한 성분을 반복 사용하며 seed mask만 달라질 수 있다.
구조 다양성 부족의 우려는 있지만, 실제 Bs 크기·overlap를 읽지 않고 모든 batch가
완전히 동일하다고 단정하지 않는다. 완전 성분을 쓰는 경우 경계 근사 오차는 오히려 없어지므로
'포화 때문에 반드시 sample/full mismatch가 커진다'는 설명도 잘못이다.

아직 확인하지 않은 항목:

- 실제 학습된 C의 구조별 분포 및 새 연결구조에서의 task 일반화.
- Sampling 평균의 full-graph operator/gradient/주파수 응답 근사 품질.
- Sample 크기·coverage·overlap 변화에 대한 C와 prediction의 안정성.
- 현재 K8/.1의 residual 충분성과 수렴 참조 대비 task 수준 차이.
- 학습 후 C=1/mean/shuffle validation 개입, beta 및 장기 gradient의 실제 수치.

샘플 기반 학습이 biased라는 이유만으로 무효인 것은 아니다.
C가 Bs와 hidden context에 의존하므로 C_s가 full C의 제한과 항상 같아야 하는 것도 아니다.
검증 목표는 사용자가 의도한 연결구조 학습과 downstream task에 이 근사가 적합한지 확인하는 것이다.

### 7. C가 학습됐다는 판정 기준을 구분한다

- **현재 입증된 것:** C 계산/정규화/전파/optimizer 경로 연결, 독립 수식과 gradient의
  synthetic 일치, 실제 run의 loss 감소·PPI 일반화 개선, 일부 synthetic solver 수렴 특성.
- **실제 run에서 아직 수치가 없는 것:** 장기 C/β 변화, task gradient 균형,
  K8 잔차 품질, learned 대 ones/shuffle validation 차이.
  기록 코드가 있다는 사실을 그 값을 읽어 효과를 검증한 것처럼 표현하지 않는다.
- **결과에서 관측한 것:** fixed 대비 C의 추가 validation 이득은 작거나 음수이고
  계산 비용은 증가한다. 이것만으로 C gradient가 끊겼다고 역추론하지 않는다.
- **다음 진단:** 기존 metrics/history를 먼저 읽고 원인을 분리한다.
  동일 LR 자체를 버그로 단정하지 않으며, entropy/K/폭/깊이/배치/샘플링 범위를
  승인 없이 바꾸거나 C variance를 강제로 키워 검증을 대체하지 않는다.
- **보존:** 완료된 fixed/dynamic checkpoint와 전체 history를 유지한다.
  이번 문서화가 기존 결과 폐기·재시작·새 C 구조 도입 또는 Git push를 뜻하지 않는다.

## 2026-09-07: 모델·샘플링 계약을 보존한 실행 병목 수정

사용자 서버 로그의 reference/arxiv dynamic-C는 epoch 62–66에 12 train batches,
epoch당 평균 667.36초였다. 이 시간은 train forward/backward만이 아니라 validation 등을
포함한다. GPU 100%/27,143 MiB/112.77 W 한 번의 관측만으로 계산 효율을 판정하지 않는다.

- 단일 그래프 C solver의 graph-ID scatter를 sum/amax/expand로 바꾼다. K=8 forward에서
  graph 목적지 index_add 54회와 scatter_reduce 24회를 없애고, 같은 degree 계산을
  14회에서 9회로 재사용한다. 메시지 패싱의 실제 node/edge reduction은 생략하지 않는다.
  다중 그래프의 분리 집계, max 동률 gradient, 양수 C·가중 평균 1과 task-loss 역전파를 유지한다.
- topology-only degree/coverage/구조 context는 forward당 한 번 계산한다.
  layer별 hidden-dependent context는 매번 계산하고 AMP 아래 geometry는 FP32를 유지한다.
- 현재 arxiv auto sampler는 neighbor가 아니라 cluster다. seed 수×26의 기존 확장 예산이
  전체 노드 수 이상이면 seed가 속한 연결 성분 전체까지 탐색하는 동일한 규칙을 유지하되,
  무방향 그래프의 connected components를 한 번 cache하여 Python BFS 반복을 제거한다.
  방향성이 비대칭이거나 SciPy가 없으면 이유를 명시하고 원래 탐색을 사용한다.
  포화 경고와 실제 nodes/edges/seed 수를 기록한다. 새로운 샘플링 법칙이나 cap은 추가하지 않는다.
  따라서 기존 실행의 큰 배치에서 B_s가 반복적으로 거의 전체 그래프가 되는 문제 자체가
  새로 다양한 국소 샘플로 바뀌었다고 주장하지 않는다.
- 고정 validation 입력은 학습 invocation당 한 번 clone/device transfer한다.
  모델 출력·activation은 cache하지 않고 입력 변조를 검출한다. 이 상주 메모리는
  새 physical-batch 후보 실측에도 포함하며, validation 자체의 forward 시간은 별도 계측한다.
- epoch마다 중복하던 layer diagnostics를 한 번으로 통합하고 sampled train mask의 nonzero는
  GPU 전송 전에 수행한다. training/validation/diagnostics의 CUDA event와 CPU wall time,
  checkpoint commit 시간 및 actual batch shapes를 history.json과 performance.json에 기록한다.
  CPU/GPU 시간은 overlap 가능하므로 더해서 전체 시간으로 해석하지 않는다.
- 새 calibration은 reference_updates 예산에서 samples/sec만 최대화하지 않고 같은 실제
  update 예산을 완료하는 예상 training 시간을 비교한다. full measurement epoch 수와
  마지막 불완전 batch까지 반영한다. validation/checkpoint 비용은 선택 목적함수에서 제외되며
  따라서 end-to-end 최적 배치라고 주장하지 않는다. 이미 완료된 자원 계획은 재선택하지 않는다.

폭·깊이·heads·K=8·데이터·샘플링 범위·physical batch·optimization 예산은 임의 축소하지 않았다.
기존 corrected run의 12 batches/epoch와 reference_updates를 그대로 유지하면 코드상
기준 45 batches/epoch × 200 epochs = 9,000 updates, 최대 750 epochs다
(실제 저장된 자원/예산 계획이 우선). 이번 수정이 이를 200 epochs로 줄이지 않는다.

정확한 51da819 소스에서 이번 성능 수정으로만 한 방향 재개를 허용한다.
기존 모델/AdamW/RNG/epoch/recipe 검사는 그대로이며 기존 결과를 삭제하지 않는다.
reduction 순서가 바뀌므로 이후 부동소수점 궤적의 bitwise 동일성은 주장하지 않는다.
기존 자원 계획은 과거 실측임을 유지하며 새 코드의 처리량 인증으로 재해석하지 않는다.
로컬 CPU 수식·gradient·sampler·재개 검증과 실제 A6000 전체 학습/성능 검증은 구분한다.

## 2026-09-06: 저성능 결과 이후의 명시적 교정 설정

사용자가 올린 20조건 validation 요약은 `historical_reference` 2개,
`pending_extra_budget` 1개, `passed` 17개다. 세부 수치와 혼합 provenance는
`EXPERIMENT_STATUS.md`에 기록한다. 원본 서버 metrics/history/checkpoint는 아직 로컬에 없으므로
학습된 C가 붕괴했다거나 특정 변경으로 성능이 회복됐다고 주장하지 않는다.

확정한 두 단계와 joint 역전파는 유지한다. 모델 폭/깊이/heads/FFN, 전체 공식 데이터,
seed 0, C 공유 채널, 8-step solver, C 양수/가중 평균 1은 줄이거나 바꾸지 않는다.

- 수치 수정: graph-context의 분산 0에서 sqrt backward가 NaN이 되지 않도록 sqrt 입력을 먼저
  안전하게 분기한다. forward의 `sqrt(clamp(var,0))` 값은 유지하며 가짜 분산을 더하지 않는다.
- 명시적 `solver_cost_scaling=width_scaled`: 단위 길이 특징의 quadratic 비용 q에 대해
  `r_e=sqrt(hidden_channels)*q_e+structure_e`,
  `delta_e=b*tanh((r_e-weighted_mean_graph(r))/b)`를 사용한다. 전체 그래프별 중심화이며
  chunk별 중심화가 아니다. 기존 bound b=2와 entropy=1을 유지하고 C 분산 목표를 강요하지 않는다.
  기본 `legacy_unit`은 과거 비용/gradient를 보존하는 비교 설정이다.
- 교정 실행 예시는 `beta_initial=0.5`로 자기/이웃 계수를 균형 있게 시작한다. beta는 여전히
  sigmoid로 학습되며 0.05 같은 강제 하한을 추가하지 않는다. 이는 0.1 초기화와 다른 학습 설정이다.
- 명시적 `learning_budget_policy=reference_updates`: GPU 실측으로 physical batch가 커져도
  기준 batch 대비 최대 업데이트 및 patience 예산을 줄이지 않는다. A6000 기준은 기존
  PPI graph batch 8, sampled seed batch 2048, full graph 1이다. 기준은 별도 CLI로 지정할 수도 있다.
  요청 E epochs, 기준 R batches/epoch, 실측 A batches/epoch에 대해
  `planned_epochs=max(E,ceil(E*R/A))`다. patience는 실제 저장된 optimizer-step 차이로 판단한다.
  PPI 20개/200 epochs에서 batch 8이면 600 updates/200 epochs, batch 20이면
  같은 600 updates를 계획하므로 최대 600 epochs다. 조기종료는 여전히 가능하며,
  큰 배치가 동일한 학습 궤적이나 충분한 총 학습량을 보장한다는 주장은 하지 않는다.
  기존 `epochs` 정책은 자동 연장하지 않는다. 새 정책은 현재 joint fresh 학습 및 그 정확한
  재개에 연결되며, 구형 MLP 전환의 누적 예산을 임의로 변경하는 데 사용하지 못한다.
- 매 epoch 마지막 실제 train batch의 C/W/backbone/beta gradient norm을 기록한다.
  validation C/beta 통계와 train gradient의 관측 시점은 명확히 분리한다.

CPU 초기값 진단(학습/실제 데이터 성능 아님)에서 width 256/384의 비용 SD는 각각
0.08074/0.06488에서 0.98884/0.97724로 바뀌었다. 이것은 비용의 차원 스케일 교정 증거이며
정확도 향상 증거가 아니다. 실제 검증에서는 기존 `selected_checkpoint_interventions`의
learned/C=1/shuffle 성적, layer별 C/beta, train loss와 실제 update 수를 함께 확인한다.

`scripts/analyze_v5_results.py --root <결과폴더 또는 manifest.json>`는 위 진단을 stdout으로
모아서 읽는다. JSON만 읽고 GPU/torch/checkpoint 로딩·파일 생성·원본 변경을 하지 않는다.
누락은 unavailable로, 구형 참조·전환·fresh·추가 예산 대기는 각각 분리해서 표시한다.

이 교정 설정을 기존 run ID/source hash에 조용히 덮어씌우지 않는다. 과거 결과와 checkpoint는
삭제하지 않는다. 구형 MLP 전용 전환 실행기가 이미 optimization으로 학습된 checkpoint를
이번 교정 설정으로 자동 이식한다고 주장하지 않는다. 새 설정의 정확한 중단/완료 재개는
검증하되, 과거 모델을 새 설정으로 이어 학습하는 별도 전환은 현재 지원 범위가 아니다.
실행 예시는 `RICH_SCALING_EXPERIMENTS.md`의 교정 설정 절을 따른다.

## 사용자 확정 기준: 샘플 B의 C 학습과 후속 GNN의 공동 최적화

이 절은 이후 설계·구현 검토에서 유지할 사용자 요구다. 현재 코드의 구현 범위와 혼동하지 않는다.

1. 전체 그래프에서 유효한 노드–엣지 부분 연결구조를 샘플링해 발생행렬 B_s를 구성한다.
2. 1단계는 B_s와 입력 특징에서 C_theta,s를 학습하고 L_theta,s = B_s^T C_theta,s B_s를 만든다.
3. 2단계는 그 학습된 가중 라플라시안을 사용하는 메시지 패싱 GNN으로 최종 예측을 만든다.
4. 2단계의 task loss는 L과 C 계산을 거쳐 1단계의 학습 파라미터 theta까지 역전파되어야 한다.
   2단계 W와 1단계 theta를 최종 과제에 맞게 함께 조정한다. 1단계의 자체 에너지 감소나
   정렬만으로 최종 예측에 유용한 라플라시안을 학습했다고 판단하지 않는다.

여기서 두 단계는 기능적 계산 순서다. 1단계를 먼저 완전히 학습한 뒤 영구 고정한다는 뜻도,
별도의 spatial GNN encoder를 반드시 추가한다는 뜻도 아니다. B 샘플링은 전체 그래프의
연결구조를 국소적으로 학습하기 위한 수단이다. 현재 고정된 샘플링 규칙 자체를 미분 가능하게
만들 필요는 없지만, 해당 샘플의 C→라플라시안→예측 경로를 detach해서는 안 된다.

현재 optimization/joint dynamic-C 경로에는 이 task-loss 피드백이 연결되어 있다. 내부 K회
C 최적화와 외부 task-loss 파라미터 갱신은 구분한다. 매 입력에서 C=1로 반복 계산을 시작해도
학습된 theta를 초기화하지는 않는다. Fixed-C 대조군은 의도적으로 C를 학습하지 않는다.
실제 전파는 sampling 보정·degree 정규화를 포함하며 블록마다 C 계산과 전파를 반복한다.
현재 기본 실행의 B 샘플링은 ogbn-arxiv에만 적용된다. 다른 transductive 데이터셋은 full graph,
PPI는 원래 그래프들의 미니배치다. 이를 모든 데이터셋의 B 샘플링 구현 완료로 보고하지 않는다.

## 현재 기본 구조: 2026-09-06 C 최적화 계층

현재 V5의 기본 `conductance_backend`는 `optimization`이다. 이전 endpoint MLP가 C를
한 번 출력하던 방식은 `--conductance-backend mlp`로 명시하는 비교 옵션으로 남긴다.
이는 C 모듈과 학습 방식의 변경이지 backbone 전체의 폐기가 아니다. Backbone, multi-head W,
beta, residual/FFN, 출력층, 전파 연산자와 샘플러는 유지했다. 이전 checkpoint를 일반 resume로
읽는 것은 허용하지 않지만, 아래의 명시적 전환으로 공통 가중치와 AdamW 상태를 재사용한다.
기존 결과를 삭제하거나 source hash를 고쳐 새 구조의 결과인 것처럼 이어 붙이지 않는다.

### 진행 중인 MLP V5를 보존하는 선택적 전환

`scripts/run_v5_transition.py`는 기존 scaling/rich manifest를 읽고 조건별 처리를 나눈다.
원본은 읽기 전용이고 결과는 별도 전환 디렉터리에 기록한다. 일반 resume guard나 이전
numerical-repair 호환 registry를 완화하지 않는다.

- 완료 fixed C: metrics/last/best/history와 데이터·설정·소스 해시를 검증하고 역사적 대조군으로
  보존한다. 다시 학습하지 않으며 새 solver로 새로 학습한 결과라고 표시하지 않는다.
- 미완료 fixed C: 모든 모델·optimizer·RNG·history/선택 상태와 기존 단계별 학습 설정을 유지한다.
- Dynamic C: 공통 backbone/W/beta/FFN/출력 가중치를 이름·shape·dtype 단위로 모두 이식하고,
  해당 AdamW moment/step도 보존한다. C estimator namespace만 교체하고 새 C optimizer 상태만
  초기화한다. 이전 C의 최고 점수를 새 C best/early-stopping 기준으로 가져오지 않는다.
- 학습 누적 epoch와 원래 총 예산을 유지한다. 예를 들어 실제 last.pt가 160/200이면 다음은
  누적 161이며 기본적으로 200까지 남은 40 epochs다. 새 C가 200 epochs 학습됐다고 주장하지 않는다.
  추가 학습은 명시적인 `--extra-epochs` 또는 조건별 `--extra-epochs-for JOB_ID=N`으로만 승인한다.
- 완료 dynamic에서 남은 예산이 없다면 이전 결과는 보존하고 새 C 조건을 `pending_extra_budget`로
  남긴다. 다른 진행 가능한 조건까지 버리거나 이미 학습한 모델을 조용히 처음부터 돌리지 않는다.
- 시작하지 않은 조건만 새 초기화한다. V1–V4/Cycle/Tree는 전환 실행기의 대상이 아니다.

전환 후 첫 update 전에 source-epoch boundary checkpoint(schema 4)를 저장한다. 새 C의
history는 전환 이후만 기록하며 `epoch_offset`으로 누적 epoch와 연결하고, 원본 history는 별도
보존한다. 같은 전환 명령의 재개는 model/optimizer/RNG와 전환 provenance를 함께 검증한다.
초기 boundary 저장 자체가 중단된 경우에도 동일 요청에 묶인 초기화 마커를 검증해 재시도한다.
검증된 source history와 fixed best는 덮어쓰지 않으며, 무관한 파일이나 이미 학습된 결과가
섞인 디렉터리는 초기화 재시도 대상으로 받아들이지 않는다.
공통 부분의 새 초기값도 실제로 이식된 state의 hash이며 무작위 초기화를 학습 재사용으로 속이지 않는다.

신형 C의 physical batch 후보는 원래 값보다 작게 줄이지 않고 실제 optimizer-inclusive
probe로 다시 측정한다. 같은 수치 학습을 유지하는 fixed continuation은 원래 실행 구성을
보존하고 그 구성에서 probe한다. 변경된 실행 값과 실제 GPU/data/runtime/source에 묶인
인증서를 기록한다. 원본 resource plan을 새 C의 실측인 것처럼 재사용하지 않는다.

이전 fixed/reference 결과와 사전학습 상태에서 시작한 새 C 결과는 같은 초기값의 fresh paired
실험이 아니다. 전환 전후 학습량·원본 artifact·재사용/초기화 내역을 분리한 보고서로 비교한다.
CPU 전환 검증과 실제 서버 checkpoint 이식·A6000 측정·전체 학습은 구분한다.

### 학습되는 대상과 forward

각 레이어에서 C는 모든 feature head에 공유되는 양수 엣지 변수다. 입력 그래프별로 C=1에서
시작하여 정확히 K회의 미분 가능한 KL-proximal mirror update를 수행한다. 학습되는 공유
파라미터 phi는 대칭 signed quadratic compatibility와 구조 비용을 정의한다. 전체 채널의
노드 projection, 그래프 문맥에 따른 signed metric, endpoint 교환 불변 degree/coverage
특징을 쓰며, edge-output MLP 또는 고정된 엣지별 파라미터 테이블은 기본 경로에 없다.
Signed metric은 같은 특징끼리만 연결해야 한다는 homophily 제약을 강제하지 않는다.

그래프별 샘플링 보정 omega, 가중 degree d(c), 기준 degree d(1)에 대해 최적화 에너지는

\[
E_\phi(c)=\operatorname{mean}_\omega[c\delta_\phi]
+\tau\operatorname{mean}_\omega[c\log c-c+1]
-\rho\operatorname{mean}_{i:d_i(1)>0}\log\frac{d_i(c)}{d_i(1)},
\qquad c>0,\quad\operatorname{mean}_\omega(c)=1.
\]

Entropy는 집중을 제어하고 degree 항은 가중 degree 붕괴를 억제한다. 계수는 task에 대해
최적이라고 검증된 값이 아니다. 그래프의 상대 곡률과 log-C 변위에 근거한 adaptive step을
사용하며 K나 엣지 수를 조용히 줄이지 않는다. 학습 loss는 K회 계산 전체를 거쳐 compatibility
파라미터와 W에 역전파된다. 평가에서도 G와 X만 사용하며 validation/test 정답으로 C를
최적화하지 않는다. 유한 K 결과를 최적해 또는 수렴 완료라고 표현하지 않는다.

\[
L_C=B^\top\operatorname{diag}(c)B,\quad
\mathcal L_C=D_C^{-1/2}L_CD_C^{-1/2},\quad
M_h=(I-\beta_h\mathcal L_C)HW_h.
\]

샘플 경로에서는 실제 edge weight가 omega*c다. C뿐 아니라 D_C도 미분 경로에 남긴다.
구현은 edge difference, scalar-C multiplication, node aggregation으로 이 연산을 계산한다.
Dense 발생/라플라시안 행렬, QR/SVD/EVD는 사용하지 않는다. C=1은 같은 backbone의
고정 엣지 전파 대조군이며, self-loop 처리와 beta/residual/FFN이 달라 표준 GCN 그 자체는 아니다.

### 설정·학습·자원 계약

- 기본 K=8, step 상한=0.25, entropy=1.0, degree barrier=0.1. 각각 `--solver-steps`,
  `--solver-step-size`, `--solver-entropy`, `--solver-degree-barrier`로 명시 변경할 수 있다.
  K=8은 유한 반복 연구 설정이며 수렴/최적 처리량을 보증하지 않는다. 초기/최종 에너지,
  projected-gradient RMS, 마지막 log-C update, 실제 adaptive step 범위를 레이어별 기록한다.
- 기존 `max_log_conductance=2.0`은 optimization backend에서 compatibility cost의 tanh
  범위로 사용된다. 최종 log-C의 고정 범위라는 뜻이 아니다. MLP backend에서는 과거 의미를 유지한다.
- 기본 `training_schedule=joint`: 첫 epoch부터 C/W/backbone/beta 공동 학습. 기존 staged
  warmup/calibration/alternating/joint는 명시적 `--training-schedule staged` 비교 옵션이다.
  총 200 epochs / patience 50 계약, 별도 optimizer parameter group은 유지한다.
- Reference 256 channels x 8 layers x 8 heads, large 384 x 12 x 8, FFN multiplier 4,
  dropout 0.2, model seed 0과 전체 공식 데이터/split을 유지한다. C=1은 C 파라미터가 없고
  두 조건의 backbone/W/beta 초기 state는 같은 seed로 정렬한다.
- 호환되지 않는 모델/solver/schedule은 다른 run ID로 실행한다. 같은 새 설정에서는
  CPU-staged checkpoint와 model/optimizer/RNG를 사용한 epoch-boundary resume를 유지한다.
- 실제 batch calibration과 benchmark_speed도 요청된 backend/K/계수를 적용한 joint 모델을
  사용한다. 모델이나 데이터를 줄이지 않고 실제 optimizer state까지 포함해 측정한다.
- C 기하/정규화는 FP32, dense 부분은 기존 hardware profile의 precision을 유지한다.
  모든 edge/head를 처리하며 chunk/recompute/checkpoint는 정확한 메모리 절약 수단이다.

### 샘플링과 미검증 범위

현재 sampler의 degree-ratio boundary correction은 휴리스틱이며 inclusion probability의
역수가 아니다. 샘플 그래프에서 C와 degree를 다시 계산하므로 전체 그래프 연산의 정확하거나
불편인 추정량이라고 주장하지 않는다. 전체 공식 그래프 validation을 유지하고, sample/full
연산 차이는 별도 진단 대상으로 둔다. 새 최적화 계층의 서버 A6000 성능, 실제 데이터 전체
학습/평가 및 기존 MLP-C 대비 정확도 개선은 아직 검증하지 않았다.

원리 참고: [Laplacian edge-weight optimization](https://proceedings.mlr.press/v51/kalofolias16.html),
[GRAND neural diffusion](https://proceedings.mlr.press/v139/chamberlain21a.html).
위 task-trained 유한 반복 계층은 이 프로젝트의 설계 후보이며 두 논문의 그대로인 구현은 아니다.

## 아래 내용은 2026-09-05까지의 MLP V5 및 실행 수정 기록

아래의 이전 현재/후속/재개 표현은 당시 소스를 가리킨다. 새 optimization V5의 구조와
호환 규칙은 위 절과 RICH_SCALING_EXPERIMENTS.md의 2026-09-06 절을 우선한다.

## 판정

V5는 V4 점수의 후속 반복이 아니라, V3/V4에서 상대 C가 거의 `C=1`로 퇴화한 원인을
수정하는 새 구조 실험이다. 2026-09-04 A6000 partial 실행 로그는 수령했지만 유효한
fixed/dynamic 전체 비교는 아직 없으며 SOTA 주장을 하지 않는다.
외부 suite identity는 `conductance_graph_conditioned_v5`다.

### 2026-09-05 ad041e2 실행의 집계 오류 수정

사용자 run `v5-cycle-se-pe-a6000-gpu3-seed0-v1`에서 첫
`reference/ogbn-arxiv/fixed_c`는 54 epochs·2,430 optimizer steps를 완료하고
최고 validation 0.719621을 저장한 뒤 child `passed`를 출력했다. 이후 scaling 집계가
`throughput.scope` 누락으로 실패해 나머지 V5 19개는 시작하지 않았다.
첫 fixed-C의 allocator peak는 7,505,515,008 bytes였다. 이는 dynamic-C/large의
OOM 해결 또는 전체 V5 비교 성공을 증명하지 않는다.

원인은 V5 writer와 공통 telemetry validator의 계약 불일치다. scope 누락뿐 아니라
`*_per_elapsed_second` 이름과 raw boolean도 validator가 요구하는 형식과 달랐다.
현재 writer는 명시적 scope, `*_per_second` 관측값 및 누적 history/elapsed 분모를
기록하며 실제 writer 결과를 scaling 소비 경로에 전달하는 회귀 검사를 추가했다.
분모에 validation/checkpoint/intervention 등이 포함된 기존 타이머 경계를 보존하며,
이를 순수 GPU forward 처리량 또는 배치 최적값으로 바꾸어 해석하지 않는다.

**이 수정은 V5 모델 구조 변경이 아니다.** V5 소스는 같은 경로에서 갱신되므로
이전 V5 소스 폴더를 따로 지울 필요도 없다. ad041e2의 C/W/beta, reference/large 크기,
optimizer, sampling, physical batch, 학습/선택 규칙과 checkpoint state 구조를 유지한다.
다만 train.py의 source fingerprint가 바뀌므로 이전 실패 run에 새 소스를 강제로 resume하지
않는다. 기존 checkpoint를 삭제하거나 hash를 바꿔 끼우지 않고, 실패 run 전체를 복구 가능한
격리 폴더에 보존하는 선택지를 [실행 문서](RICH_SCALING_EXPERIMENTS.md)에 제공한다.

더 오래된 214265c까지 전부 동일하다는 뜻은 아니다. 이후 사용되지 않는 fixed-C scorer 제거와
공통 파라미터 초기화 정렬이 있었으므로 구 parameter key/초기 state와 현재 결과를 혼합하지 않는다.

## 모델 계약

그래프 \(G=(V,E,X)\), layer \(l\)에서 하나의 shared conductance field를 만든다.

\[
s_e^{(l)}=f_{\theta_C}^{(l)}
\left(h_u+h_v,\ |h_u-h_v|,\ h_u\odot h_v,\ p_u,p_v,p_e,z_G\right),
\qquad e=\{u,v\}.
\]

입력은 endpoint 교환에 불변이고, \((p,z_G)\)는 원본 그래프의 local/global 구조 문맥이다.
학습되는 것은 고정 edge table이 아니라 모든 그래프와 sample에 공유되는 함수
\(f_{\theta_C}\)다. graph \(g\)의 edge 보정값을 \(\omega_e\)라 하면 실제 구현의 scale 고정은

\[
\bar s_g=\frac{\sum_{e\in E_g}\omega_e s_e}{\sum_{e\in E_g}\omega_e},\qquad
\widetilde c_e=\exp\!\left(a\tanh\frac{s_e-\bar s_g}{a}\right),\qquad
c_e=\frac{\widetilde c_e}
{\left(\sum_{j\in E_g}\omega_j\widetilde c_j\right)/\left(\sum_{j\in E_g}\omega_j\right)}.
\]

full graph에서는 \(\omega_e=1\)이고 sampled graph에서는 명시된 boundary correction을 쓴다.
따라서 C는 항상 양수이고 graph별 가중 산술평균이 정확히 1이며, log-score 범위도 \(a=2\)로
제한된다. C의 전역 배율은 normalized Laplacian에서 소거되므로 별도 graph-conditioned
head scale \(\beta_h(G)\)가 실제 diffusion 크기를 맡는다.

\[
L_C=B^\top\operatorname{diag}(c)B,\quad
\mathcal L_C=D_C^{-1/2}L_CD_C^{-1/2},\quad
M_h=(I-\beta_h(G)\mathcal L_C)\,H W_h.
\]

C는 head마다 따로 만들지 않고 한 layer에서 공유한다. 여러 head는 동일한 graph geometry 위에서
각자의 \(W_h\)와 \(\beta_h\)만 학습한다. C generator는 작은 nonzero 초기화로 첫 step부터
전체가 gradient를 받으며, C/W 우회 문제를 줄이기 위해 spatial warm-up, C calibration,
alternating, joint phase를 사용한다.

기본 beta parameterization은 hard margin이 없는

\[
\beta_h(G)=\operatorname{sigmoid}(r_h(G)),\qquad
b_{\beta,0}=\operatorname{logit}(0.1)
\]

이다. 마지막 beta weight는 작은 nonzero 값으로 초기화하므로 최초 출력은 정확한 상수가 아니라
명시된 nominal `beta_initial=0.1` 근방이며, 첫 step부터 upstream beta network에도 gradient가
흐른다. 기본 config/checkpoint identity에는 의미 없는 `beta_min`/`beta_max`를 기록하지 않는다.

이전 bounded-margin 식은 삭제하지 않고 명시적 ablation으로만 남긴다.

\[
\beta_h(G)=\beta_{\min}+(\beta_{\max}-\beta_{\min})
\operatorname{sigmoid}(r_h(G)).
\]

과거 설정을 재현하려면 direct V5 runner에
`--beta-parameterization margin_sigmoid --beta-initial 0.5 --beta-min 0.05 --beta-max 0.95`를
함께 지정한다. 이 모드는 `0 <= beta_min < beta_initial < beta_max <= 1`을 강제하고,
`beta_initial`이 실제 parameterization의 nominal 출력이 되도록 정규화된 위치의 역로짓을
마지막 bias에 넣는다. 선택한 방식과 초기값·margin은 manifest와 child resume identity에 고정된다.
Conductance scaling과 rich scaling에서 같은 ablation을 선택할 때는 각 옵션에 `--v5-` 접두사를
붙인 `--v5-beta-parameterization`, `--v5-beta-initial`, `--v5-beta-min`, `--v5-beta-max`를 쓴다.

## 비교 조건

- `fixed_c`: 정확히 C=1. W, beta, FFN과 classifier에 C calibration/alternation 예산까지
  배정해 강한 spatial baseline으로 학습한다.
- `shared_dynamic_c`: 위 shared graph-conditioned C를 coordinate phase에서 학습하고 나머지
  phase에서 W, beta, FFN과 classifier를 학습한다.

두 arm은 동일한 데이터/split/seed/sampling과 공통 backbone·W·beta·FFN·classifier 구조 및
초기화를 쓴다. `fixed_c`의 C=1은 parameter-free이며, `shared_dynamic_c`만 실제 forward에 쓰는
C score network를 추가한다. 공통 state hash는 일치해야 하지만 전체 parameter 수와 전체 state
hash를 억지로 맞추지 않으며 그 차이를 보고한다. 의도적으로 phase별 parameter-group update
배분도 다르다. 따라서 이것은 **fixed-C strong recipe 대 dynamic-C
coordinate recipe의 end-to-end 비교**이며 C 하나만 치환한 인과효과가 아니다. manifest에는
`effective_optimizer_steps_by_group`을 반드시 기록한다. 같은 checkpoint의 `C=1`, mean-C,
shuffled-C intervention과 C gradient/CV를 함께 읽고, fixed-C와의 점수 차이만으로 C의 순수
기여를 판정하지 않는다.

이 parameter-free control 변경 전 source로 만든 partial V5 checkpoint는 새 비교에 resume하거나
재사용하지 않는다. source/resume hash가 이를 거부하므로 새 run-id로 두 arm을 모두 fresh 실행한다.

Checkpoint 선택도 condition별 역할을 구분한다. `fixed_c`에는 기다려야 할 C mechanism이
없으므로 모든 epoch 중 validation 최고점을 primary checkpoint로 고르고 그 기준으로 early
stopping한다. `shared_dynamic_c`의 warm-up은 C를 강제로 1로 우회하므로, 전체 epoch 최고점은
auxiliary prediction score로만 기록하고 primary checkpoint는 C가 실제로 활성화된 calibration,
alternating 또는 joint epoch 중에서 선택한다. Dynamic arm의 early stopping은 별도의 joint-phase
best를 감시해 warm-up 최고점 때문에 C 학습이 시작되기도 전에 중단되지 않게 한다. 최종 비교표의
primary 차이는 fixed all-epoch best 대 dynamic C-active best이고, dynamic all-epoch prediction
best와 joint monitor best도 별도 열로 함께 보고한다. Test label은 어느 선택에도 사용하지 않는다.

## 실제 규모와 execution profile

- `reference`: hidden 256, 8 layers, 8 heads, FFN multiplier 4, dropout 0.2.
- `large`: hidden 384, 12 layers, 8 heads, FFN multiplier 4, dropout 0.2.
- `auto` sampling은 ogbn-arxiv train에 cluster sampling, 나머지는 full graph를 쓴다.
- validation은 항상 완전한 공식 graph/split이다. PPI는 공식 20/2/2 inductive graph split이라
  neighbor/cluster sampling을 허용하지 않는다.

Architecture profile과 hardware profile은 별도 축이다. Cora/CiteSeer/PubMed는 두 hardware
profile 모두 single full-graph batch 1이라 48GB GPU를 가득 채울 minibatch 축이 없다.

| 설정 | `portable` | `a6000-48gb` |
|---|---|---|
| dense numeric path | FP32, TF32 off | BF16 autocast, TF32 on |
| conductance score/centering/degree/diffusion | FP32 | FP32 |
| block activation checkpoint | on | off |
| dynamic-C edge-score chunk checkpoint | gradient가 있을 때 on | gradient가 있을 때 on |
| edge chunk | 65,536 | 131,072 |
| ogbn-arxiv sampled seed-node batch | 1,024 | 2,048 |
| PPI whole-graph batch | 2 | 8 |
| sample pipeline | pinned transfer, synchronous construction | pinned transfer와 sample prefetch |

`a6000-48gb`는 보이는 VRAM 40GiB 이상, 시작 시 free VRAM 32GiB 이상, compute capability
8.0 이상과 BF16 지원을 child 시작 시 검사하고 조건이 맞지 않으면 자동 fallback 없이 중단한다.
아래 명령은 더 엄격하게 `--min-free-gb 40`을 지정한다.

과거 `214265c`는 score-network chunk의 activation을 재계산하도록 고쳤지만, diffusion의
edge-feature tensor를 backward까지 전부 보관하는 경로는 남아 있었다. **그 수정판이 적용된
r3의 large/arxiv에서도 OOM이 재발했다.** 이는 구 source가 실행됐다는 이유로 설명할 수 없다.

현재 `shared_head_diffusion`은 정규화된 scalar edge weight의 대칭 propagation을 custom
autograd로 수행한다. Node message·scalar edge weight·incidence만 저장하고 backward에서
각 edge chunk의 gather/곱셈을 재계산한다. Degree normalization은 미분 가능한 상태로 유지한다.
일반 first-order 저장량은 `O(N*heads*width+E)`, 임시 tensor는 chunk 크기에 제한되며 모든
edge·layer·channel·batch·sampling을 그대로 처리한다. `create_graph=True`의 2차 미분도
지원하지만 추가 derivative graph 저장은 first-order 메모리 보장에 포함하지 않는다.

합성 CPU 진단(N=128,E=1024,heads=4,width=8; frozen message, C gradient on)에서 고유
autograd 저장량은 563,840→60,544 bytes로 약 89.3% 줄었다. Dense 독립식과의 출력 및
C/message/beta/correction gradient, double gradcheck/gradgradcheck, BF16 FP32 geometry,
isolates·빈 그래프와 저장량 경계 등 관련 CPU 검사 28개가 통과했다. **실제 A6000 peak VRAM이나
전체 학습 성공·속도 개선을 검증한 결과는 아니다.**

Block checkpoint 기본값은 표처럼 유지한다. 필요하면 `--v5-activation-checkpoint`를 새 실행
계약에서 명시할 수 있지만 현재 source/config hash와 다른 partial checkpoint를 같은 run ID에
억지로 연결하지 않는다. 직접 V5/scaling과 통합 `run_rich_scaling.py` 모두 override를
Conductance child에만 전달하고 manifest에 기록한다.

V5 child는 실제 학습 경계에서 기본 1초 주기로 GPU SM·memory-controller utilization, CUDA
allocator allocated/reserved, process CPU·RSS/HWM과 system available RAM을 측정해
`resource_observability`에 저장한다. 지원되지 않는 counter는 0이 아니라 `null`과 원인을
기록한다. 이 계측이 적용된 전체 GPU run은 아직 수령하지 않았으므로 utilization 수치를
성능 결과처럼 인용하지 않는다. 1,024/2,048 seed batch와 PPI 2/8 graph batch는 등록된
profile recipe이며, 현재 rich 경로에서는 실측 탐색의 요청 하한이다. 본 학습 전에 별도 joint-phase
모델의 실제 loss/backward/clipping/AdamW update와 optimizer state를 포함해 여러 batch와
worker 후보를 측정한다. V5 probe는 전체 train epoch를 완료하며 fixed/dynamic 조건에 공통으로
안전한 값을 선택해 본 학습 동안 고정한다. 모델 크기·fanout·데이터·epoch를 줄이지 않는다.
Full-graph citation 데이터는 별도 batch 축이 없는 사유를 기록하며 그래프를 복제하지 않는다.
현재 통합 실행과 재개 계약은 [RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md)
첫 절을 따른다. 실제 A6000 후보 측정 결과는 아직 없다. 이 경로와 별개로
`scripts/benchmark_speed.py --track conductance_v5 --batch-sizes ...`는 공식 train 입력에서
명시한 seed-node/PPI graph batch 후보를 각각 측정하고, 10% 이상 projected device-memory
headroom을 남긴 후보 중 처리량이 가장 높은 값을 **microbenchmark 권고**로 기록한다. 이 측정은
optimizer state·전체 epoch·validation/checkpoint를 포함하지 않고 training profile 기본값도
바꾸지 않으므로, 권고값을 최종 학습 최적값으로 해석하려면 별도 전체-run 검증이 필요하다.

한 architecture profile에서 다섯 datasets × 두 arms는 10 fresh trainings다. 두
architecture profiles의 V5만 실행하면 20 trainings이고, V1–V5 전체 reference/large
Conductance scaling은 106 child/model trainings다.

### Portable/10GB MIG 예시

아래 GPU 6은 과거 사용한 10GB MIG slice의 물리 번호를 보존한 portable 예시다. 실제 할당
번호가 다르면 `CUDA_VISIBLE_DEVICES`만 바꾸고 프로세스 내부에서는 `cuda:0`을 쓴다.

```bash
CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_conductance_v5.py \
  --datasets cora citeseer pubmed ppi ogbn-arxiv \
  --profile reference --model-seed 0 --device cuda:0 \
  --sampling auto --sample-seed-batch-size 1024 \
  --hardware-profile portable \
  --run-id conductance-v5-portable-gpu6-seed0
```

### RTX A6000 GPU 3 예시

```bash
CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_conductance_v5.py \
  --datasets cora citeseer pubmed ppi ogbn-arxiv \
  --profile reference --model-seed 0 --device cuda:0 \
  --sampling auto --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --run-id conductance-v5-a6000-gpu3-seed0
```

동일한 명령과 run-id를 다시 실행하면 passed artifact를 hash 검증 후 건너뛰고,
`<child>/last.pt`가 있는 미완료 V5 child는 삭제하지 않고 model/optimizer/phase/history/RNG를
epoch 경계에서 복원한다. 저장 RNG/state 기준의 deterministic continuation을 목표로 하지만 CUDA
kernel까지 bitwise 동일하다고 주장하지 않는다. 다른 config/source/job matrix로 같은 run-id를
재사용하면 fail closed한다. 재개할 때는 해당 hardware profile과 모든 인수 및 run-id를 그대로
유지해야 한다.

이 보장은 **같은 implementation hash**에만 적용된다. 아래 r1/r2는 checkpoint-selection
수정 전 source이며 r3는 그 수정 이후다. 이번 diffusion backward 변경 때문에 r3 source도 현재와
다르므로 구 partial을 같은 run ID에 억지로 resume하지 않는다. 기존 artifact를 보존하고 필요한
대상만 새 run ID로 선택한다. r1/r2 fixed-C의 epoch 10 global-best model state는 구 코드가
저장하지 않았기 때문에 수정된 primary checkpoint로 복구할 수 없다.

## 2026-09-04 A6000 partial 실행과 old-source r2 재현

Run `new-v5-cyclev2-a6000-gpu3-seed0-r1-conductance`, model seed 0에서 첫 job인
`v5/reference/ogbn-arxiv/fixed_c`는 200 epochs를 완료했다. 로그상 전체 최고 validation은
epoch 10의 0.692775였지만 구 joint-only 선택은 0.680392를 골랐다. Train loss는 0.579221에서
0.019955까지 내려가는 동안 validation이 대체로 0.67대로 하락해 과적합 또는 sampled-train/
full-graph-eval 불일치 신호가 있다. 이 결과는 corrected fixed primary 결과로 쓰지 않는다.

두 번째 `shared_dynamic_c`는 warm-up epoch 20까지 실행한 뒤 최초 C calibration backward에서
CUDA OOM으로 중단됐다. 따라서 dynamic C 점수, fixed-vs-dynamic 비교 및 나머지 18개 V5 job은
미완료다. GPU preflight 통과는 장치 가용성 검사였고 이 모델의 peak-memory 적합성을 인증한
것이 아니었다. 위 partial 수치는 실패 원인과 수정 필요성의 근거이지 V5 성능 결론이 아니다.

후속 `new-v5-cyclev2-a6000-gpu3-seed0-r2-conductance`도 fixed-C 하나만 완료한 뒤 dynamic-C
최초 calibration backward에서 같은 44.47/44.55GiB OOM으로 중단되어 18개가 미실행이다.
첨부 `bd63fc9a-60da-4daf-9ab9-da49db7cbbe1/pasted-text.txt`의 SHA-256은
`F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6`다. 로그의
`train.py:785`와 `joint_best=` 단독 출력은 수정 전 `08d8ed6` 코드와 정확히 일치한다. 따라서
r2는 `214265c`의 dynamic edge-score checkpoint나 condition-aware checkpoint selection을
검증하지 않았다. 이는 r2의 역사적 판정이며 아래 r3 재발에 적용하지 않는다.

## r3 large OOM 재발과 현재 메모리 수정

사용자 제공 `new-v5-cyclev2-a6000-gpu3-seed0-r3-conductance` 로그의 4/20번째 job은
`v5/large/model-seed-0/ogbn-arxiv/shared_dynamic_c`다. Warm-up validation은 epoch 1
0.599550, epoch 10 0.699822, epoch 20 0.666331, global best 0.715494였고 이후 forward의
`shared_head_diffusion`에서 추가 192MiB 할당에 실패했다. 이는 최종 성능이나 C-active best가 아니다.

Traceback의 `model.py:406/455/570`, `train.py:832/1192/1203`은 `214265c`와 정확히
일치한다. `primary_best=pending` 출력도 해당 selection 수정판의 증거다. GPU 총 44.55GiB 중
free 9.62MiB, 해당 프로세스 44.54GiB, PyTorch allocated 41.70GiB와 reserved-but-unallocated
2.52GiB였다. 실패한 192MiB는 `131072 edges * 384 hidden * 4 FP32 bytes`와 일치한다.
단순 환경변수·fragmentation 문제로만 보거나 이미 해결된 로그로 취급하지 않는다.

위 custom backward가 이번 diffusion 저장 누적을 수정한 구현이다. 아직 서버의 새 full-run
성공 결과는 없고 r3의 나머지 job 상태도 이 일부 로그로 추정하지 않는다. V5 operator source가
변경됐으므로 r3 partial의 strict source-hash resume 거부를 우회하지 않는다. 기존 결과는
보존하며 현재 소스로 실행할 대상만 새 run ID에서 명시적으로 선택한다. 새 Cycle V2는 별도
[QR-free sparse DFS 실행](CYCLE_PE_V2.md)을 사용하고 완료한 V1–V4/Cycle V1/Tree를 반복하지 않는다.

## V1–V5 reference/large 비교

다음은 GPU 6 portable 예시다. A6000 전체 실행은
[전체 scaling 문서](RICH_SCALING_EXPERIMENTS.md)의 GPU 3 명령을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_conductance_scaling.py \
  --versions v1 v2 v3 v4 v5 --profiles reference large \
  --model-seeds 0 --device cuda:0 --hardware-profile portable \
  --run-id conductance-v1-v5-portable-gpu6
```

Conductance scaling의 A6000 profile은 V5의 실제 sample/PPI batch와 numeric recipe만 바꾸고
V1–V4의 legacy FP32·batch 계약은 바꾸지 않는다. 특히 PPI에서 V1/V3/V4는 batch 2/FP32,
V5 A6000은 batch 8/BF16이므로 cross-version PPI 차이는 descriptive scaling 결과다.
fixed/dynamic C는 같은 hardware profile 안에서만 비교하고, portable와 A6000 사이의 점수나
wall time 차이를 C·모델·GPU 하나의 효과로 직접 해석하지 않는다.
