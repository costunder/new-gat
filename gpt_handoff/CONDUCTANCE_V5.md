# Conductance GAT V5 — graph-conditioned shared dynamic C

## 판정

V5는 V4 점수의 후속 반복이 아니라, V3/V4에서 상대 C가 거의 `C=1`로 퇴화한 원인을
수정하는 새 구조 실험이다. 2026-09-04 A6000 partial 실행 로그는 수령했지만 유효한
fixed/dynamic 전체 비교는 아직 없으며 SOTA 주장을 하지 않는다.
외부 suite identity는 `conductance_graph_conditioned_v5`다.

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
성능 결과처럼 인용하지 않는다. 현재 1,024/2,048 seed batch와 PPI 2/8 graph batch는 등록된
profile recipe이며, rich/V5 학습 runner가 자동 튜닝해 고른 값이 아니다. 별도
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
