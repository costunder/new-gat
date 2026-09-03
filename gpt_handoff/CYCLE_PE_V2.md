# Cycle PE V2 — coordinate-free left-nullspace projector PE

## 구 V2 폐기

과거 `cycle_basis_v2`는 좌영공간 기저의 임의 column을 학습기에 직접 보여 주었고 실제 GPU
결과도 실패했다. 그 구현과 checkpoint/result identity는 폐기되었으며 새 V2와 호환되지 않는다.
현재 유일한 V2 model identity는 `cycle_projector_pe_v2`다.

## 수학 계약

무방향 그래프의 공급 orientation으로 edge-by-node incidence를

\[
B\in\mathbb R^{m\times n},\qquad B_{e,u}=-1,\ B_{e,v}=+1
\]

로 두면 cycle space는

\[
\mathcal C(G)=\ker(B^\top),\qquad \beta=m-n+k
\]

이다. 구현은 SVD/eigendecomposition을 매번 수행하지 않는다. Union-find spanning forest의
각 chord가 닫는 fundamental cycle로 sparse full-rank

\[
Z\in\mathbb R^{m\times\beta},\qquad B^\top Z=0
\]

를 만들고 데이터 준비 시 orthonormal \(Q\)로 변환·검증해 cache한다. 학습기에는 arbitrary
basis column이 아니라 coordinate-free projector만 제공한다.

\[
P_{\mathcal C}=Z(Z^\top Z)^{-1}Z^\top=QQ^\top.
\]

이는 모든 invertible basis change \(Z\mapsto ZR\)에 불변이다. Edge orientation을 뒤집으면
\(P\mapsto SPS\), \(S_{ee}\in\{-1,+1\}\)이므로 다음 orientation-free kernel을 사용한다.

\[
K_{\mathcal C}=P_{\mathcal C}\odot P_{\mathcal C}.
\]

Bond embedding \(X_E\)에 대한 핵심 cycle mixing과 leverage는

\[
M_E=K_{\mathcal C}\phi(X_E),\qquad
\ell_e=(P_{\mathcal C})_{ee}=\|Q_{e,:}\|_2^2
\]

다. PE encoder는 local value, (M_E/\max(\ell_e,\epsilon)), leverage와 rank fraction을
결합한다. Bridge와 acyclic component의 모든 edge는 \(\ell_e=0\)이므로 최종 PE가 정확히 0이다.
Production forward는 \(P\)나 \(K\)의 \(m\times m\) 행렬을 만들거나 저장하지 않는다. 대신

\[
(K\phi(X_E))_{i,d}
=q_i^\top\!\left[Q^\top\operatorname{diag}(\phi(X_E)_{:,d})Q\right]q_i
\]

를 feature/rank block의 pair-free low-rank contraction으로 계산한다. 시간 복잡도는
\(O(m\beta^2d)\)이고 `--basis-pair-budget`는 임시 feature-by-rank-by-rank core allocation의
원소 수를 제한한다.

## 선택 가능한 basis backend

`--basis-backend`는 다음 두 경로를 선택한다.

- `thin_q`(기본): union-find spanning forest에서 sparse fundamental \(Z\)를 만들고 데이터 준비
  시 reduced thin QR을 한 번 수행해 model-ready \(Q\)를 cache한다. 학습 forward에는 QR이 없다.
- `dfs_fundamental`: iterative DFS spanning forest를 만들고, 각 non-tree edge(chord)에 대해
  DFS parent pointer를 역추적해 유일한 tree path와 chord를 합친 signed fundamental cycle 열을
  만든다. raw \(Z\)를 cache하며 projector에 넣기 직전 graph별 reduced QR로 \(Q\)를 만든다.

DFS가 선택하는 tree edge가 \(n-k\)개이고 나머지 chord가 \(m-n+k=\beta\)개이므로 정확히
cycle-space 차원만큼의 독립 열을 만든다. 각 열은

\[
z_j=\pm e_{\text{chord }j}+\text{signed parent path},\qquad B^\top z_j=0
\]

을 만족한다. 이는 모든 simple cycle을 backtracking으로 열거하는 알고리즘이 아니다. 모든 simple
cycle 열거는 출력 개수 자체가 지수적으로 커질 수 있으므로 선형 시간이라고 할 수 없다.

복잡도 주장은 다음처럼 분리한다.

- DFS forest 발견: \(O(|V|+|E|)\)
- explicit sparse fundamental basis 복원: \(O(|V|+|E|+\operatorname{nnz}(Z))\)
- 현재 dense cache materialization: 최소 \(\Omega(m\beta)\) storage/write
- raw backend의 graph별 reduced QR: 통상 \(O(m\beta^2)\), 매 model forward 반복
- 이후 projector-kernel contraction: \(O(m\beta^2d)\)

따라서 `dfs_fundamental`은 DFS로 선택된 실제 cycle basis를 검사·비교하기 위한 진단 backend이지,
전체 projector V2 학습을 선형 시간으로 만드는 가속 옵션은 아니다. 두 backend는 최종적으로 같은
cycle-space projector를 사용하므로 모델 identity는 `cycle_projector_pe_v2`로 유지하지만, cache
signature, CLI arguments, manifest와 resume identity는 backend를 구분하고 혼용을 거부한다.

## 모델과 실제 규모

현재 V2는 projector bond PE를 쓰는 residual molecular GNN이다. model identity는
`cycle_projector_pe_v2`이고 구 `cycle_basis_v2` artifact를 읽지 않는다.

| Profile | ZINC-12K | Peptides-struct |
|---|---|---|
| `reference` | hidden 128, PE 64, 10 layers | hidden 256, PE 64, 6 layers |
| `large` | hidden 192, PE 96, 12 layers | hidden 320, PE 96, 8 layers |

모두 FFN multiplier 4, dropout 0.1, layer scale 0.1이다. fail-closed parameter ceiling은
50M이다. 이는 실행 architecture 계약이지 성능 결과가 아니다.

## 실행과 재개

Architecture profile과 hardware profile은 서로 다른 축이다. Scaling runner가 V1/V2에 적용하는
실행 자원은 다음과 같다.

| 설정 | `portable` | `a6000-48gb` |
|---|---|---|
| graph batch | 모든 dataset/profile 32 | `reference`: ZINC 512, Peptides 128; `large`: ZINC 256, Peptides 64 |
| loader | workers 4, prefetch 2 | workers 8; V1 prefetch 2, V2 prefetch 4 |
| numeric path | AMP off, FP32 | backbone FP16 AMP; V2 projector contraction FP32 |
| V2 contraction | column chunk 16, pair budget 32,768 | column chunk 32, pair budget 4,194,304, packed cycle-basis H2D |

`a6000-48gb`는 보이는 VRAM 40GiB 이상과 compute capability 8.0 이상을 요구한다. V2는 epoch
1 전에 큰 graph부터 구성한 요청 batch의 training-mode forward/backward를 실행해 capacity를
검사한다. OOM이면 batch를 자동으로 줄이거나 중간 epoch에서 protocol을 바꾸지 않고 실패하며,
명시적으로 더 작은 `--batch-size`와 새 run-id를 요구한다.

다음 GPU 6 명령은 과거 10GB MIG 할당 번호를 보존한 portable 예시다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=6 \
python -B scripts/run_cycle_scaling.py \
  --versions v1 v2 --profiles reference large \
  --datasets zinc12k peptides_struct --model-seeds 0 \
  --device cuda:0 --hardware-profile portable \
  --run-id cycle-v1-v2-portable-gpu6
```

물리 GPU 3의 RTX A6000을 프로세스 내부 `cuda:0`으로 매핑하는 실행은 다음과 같다. 이 명령은
profile의 장치 계약에 더해 모든 child의 일반 preflight에 free VRAM 40GiB를 요구한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
python -B scripts/run_cycle_scaling.py \
  --versions v1 v2 --profiles reference large \
  --datasets zinc12k peptides_struct --model-seeds 0 \
  --device cuda:0 --hardware-profile a6000-48gb \
  --min-free-gb 40 \
  --run-id cycle-v1-v2-a6000-gpu3
```

V2만 실행하려면 `--versions v2 --profiles reference`를 쓴다. 각 dataset/profile/seed는
서로 다른 output을 가진다. 동일 run-id 재실행 시 완결 artifact는 검증 후 건너뛰고, 새 V2의
`<output>/<dataset>/cycle_projector_pe_v2/last.pt`가 있으면 output을 삭제하지 않고
`--resume`으로 model/optimizer/epoch/history/RNG를 복원한다.

재개 계약은 동일한 config/source/artifact hash 결속과 epoch boundary의 model·optimizer·
scheduler·AMP scaler·history·RNG/DataLoader 상태 복원이다. CUDA scatter/reduction처럼
비결정적일 수 있는 kernel까지 중단 없는 실행과 bitwise 동일하다고 보장하지는 않는다.

DFS fundamental backend를 명시적으로 검사하려면 같은 scaling 명령에 다음 인수를 추가한다.

```bash
--basis-backend dfs_fundamental
```

기본 `thin_q`가 전체 학습용 경로다. DFS backend는 runtime QR 비용 때문에 wall time 개선을 기대하는
옵션이 아니며, 두 backend의 결과와 timing은 manifest의 `basis_backend`를 확인해 구분한다.

seed 0에서 V1/V2 × 두 datasets × reference/large는 8 fresh candidate trainings이고,
validation profile 선택 뒤 4 selected-checkpoint test-only evaluations가 추가된다. 후자는 optimizer
step이나 재학습이 아니다. 같은 hardware profile 안에서 V1/V2를 비교한다. A6000은 batch와
numeric execution이 실제로 달라지므로 portable와 A6000 사이의 점수나 wall time 차이를 PE 또는
GPU 하나의 효과로 직접 해석하지 않는다.

현재 새 V2 GPU 결과는 없다. 구 V2 실패값을 새 projector V2 결과로 재사용하거나 SOTA와
비교해서는 안 된다.
