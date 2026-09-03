# Conductance GAT V5 — graph-conditioned shared dynamic C

## 판정

V5는 V4 점수의 후속 반복이 아니라, V3/V4에서 상대 C가 거의 `C=1`로 퇴화한 원인을
수정하는 새 구조 실험이다. 아직 GPU 성능 결과는 없으며 SOTA 주장을 하지 않는다.
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

두 arm은 동일한 데이터/split/seed/architecture/초기화/sampling을 쓰지만, 의도적으로 phase별
parameter-group update 배분이 다르다. 따라서 이것은 **fixed-C strong recipe 대 dynamic-C
coordinate recipe의 end-to-end 비교**이며 C 하나만 치환한 인과효과가 아니다. manifest에는
`effective_optimizer_steps_by_group`을 반드시 기록한다. 같은 checkpoint의 `C=1`, mean-C,
shuffled-C intervention과 C gradient/CV를 함께 읽고, fixed-C와의 점수 차이만으로 C의 순수
기여를 판정하지 않는다.

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
| activation checkpoint | on | off |
| edge chunk | 65,536 | 131,072 |
| ogbn-arxiv sampled seed-node batch | 1,024 | 2,048 |
| PPI whole-graph batch | 2 | 8 |
| sample pipeline | pinned transfer, synchronous construction | pinned transfer와 sample prefetch |

`a6000-48gb`는 보이는 VRAM 40GiB 이상, 시작 시 free VRAM 32GiB 이상, compute capability
8.0 이상과 BF16 지원을 child 시작 시 검사하고 조건이 맞지 않으면 자동 fallback 없이 중단한다.
아래 명령은 더 엄격하게 `--min-free-gb 40`을 지정한다.

한 architecture profile에서 다섯 datasets × 두 arms는 10 fresh trainings다. 두
architecture profiles의 V5만 실행하면 20 trainings이고, V1–V5 전체 reference/large
Conductance scaling은 106 child/model trainings다.

### Portable/10GB MIG 예시

아래 GPU 6은 과거 사용한 10GB MIG slice의 물리 번호를 보존한 portable 예시다. 실제 할당
번호가 다르면 `CUDA_VISIBLE_DEVICES`만 바꾸고 프로세스 내부에서는 `cuda:0`을 쓴다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_conductance_v5.py \
  --datasets cora citeseer pubmed ppi ogbn-arxiv \
  --profile reference --model-seed 0 --device cuda:0 \
  --sampling auto --sample-seed-batch-size 1024 \
  --hardware-profile portable \
  --run-id conductance-v5-portable-gpu6-seed0
```

### RTX A6000 GPU 3 예시

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
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

## V1–V5 reference/large 비교

다음은 GPU 6 portable 예시다. A6000 전체 실행은
[전체 scaling 문서](RICH_SCALING_EXPERIMENTS.md)의 GPU 3 명령을 사용한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=6 \
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
