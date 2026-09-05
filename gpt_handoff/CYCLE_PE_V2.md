# Cycle V2 — SE 대 SE+상대 PE 실험

기준일: 2026-09-05. 현재 V2는 같은 DFS 기저와 backbone을 사용하는 두 조건이다.

현재 rich 통합 실행은 본 학습 전에 실제 공식 train graph로 batch/worker 후보를 측정한다.
Sparse collate·전송·loss/backward·clipping·Adam update와 optimizer state 및 큰 실제 그래프
batch의 메모리를 포함하며 SE/PE에 공통으로 안전한 측정 설정을 적용한다. 기존 단일 capacity
probe만으로 batch 선택을 끝내지 않는다. 이 변경은 DFS 기저·PE 수학·모델 크기를 바꾸지 않으며
QR/SVD를 추가하지 않는다. 실행·재개 계약은
[RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md) 첫 절에 모았다.
실제 A6000 측정·전체 새 학습 결과는 아직 없으며 CPU 회귀와 구분한다.

| 조건 | 모델 ID | 역할 |
|---|---|---|
| `se` | `cycle_dfs_se_v2` | 기존 cycle 소속·길이·결합 문맥의 구조 인코딩 |
| `pe` | `cycle_dfs_relative_pe_v2` | 동일 SE에 cycle 내부 상대 위치 residual을 추가 |

`pe`는 SE를 없앤 pure-PE 조건이 아니다. 동일한 학습 파라미터와 backbone에서
상대 위치 항을 추가했을 때의 효과를 비교한다. QR, SVD, 고유값분해, Gram inverse,
dense edge-by-cycle 또는 cycle별 all-pairs 행렬을 만들지 않는다.

## 이전 V2와 분리

과거 `cycle_basis_v2`, `cycle_projector_pe_v2`는 모두 현재 모델과 다른 설계다.
전자는 폐기된 raw-column 모델이고, 후자는 `P=Z(Z^T Z)^{-1}Z^T`의 방향 불변 kernel을
사용한 projector 모델이다. 후자의 일반 기저 변환 불변성을 보존하려고 QR을 넣었으나,
사용자가 요청한 DFS 기반 속도 개선과 맞지 않아 현재 학습 경로에서 제거했다.
기존 결과·checkpoint·projector cache를 새 DFS 모델에 재사용하지 않는다. 역사적 실패 기록은
아래에 보존하며 현재 성능으로 재분류하지 않는다.

그 다음 `cycle_dfs_sparse_pe_v2`는 이번 SE와 같은 특징 요약 연산이지만 당시
PE라고만 명명한 것은 부정확했다. 현재는 구조 요약과 상대 위치 항의 비교를 명시적으로
나누며 구 ID 결과를 새 조건으로 섞지 않는다.

## 발생행렬과 DFS 기저

edge-by-node 발생행렬의 각 edge 방향을 tail에서 head로 두면

\[
B\in\mathbb R^{m\times n},\qquad B_{e,tail}=-1,\quad B_{e,head}=+1,
\qquad \mathcal C(G)=\ker(B^\top).
\]

연결 성분 수가 \(k\)이면 \(\beta=m-n+k\)다. 유일한 backend `dfs_fundamental`은 iterative
DFS spanning forest를 만든 뒤 각 non-tree edge(chord)에서 parent pointer를 역추적하여
해당 chord와 tree path가 닫는 signed fundamental cycle을 만든다.

\[
Z=[z_1,\ldots,z_\beta]\in\mathbb R^{m\times\beta},\qquad B^\top Z=0.
\]

각 cycle에는 자기 chord가 있고 다른 chord는 없다. 따라서 chord 행이 독립성의 증거가 되며,
전체 \(\beta\)개의 열이 좌영공간 기저를 이룬다. 이 검증은 dense Gram/Cholesky/rank 분해로
대체하지 않는다. 기저 열·그래프·edge를 잘라내지 않는다. 모든 simple cycle을 열거하는
알고리즘은 아니며, 그런 열거는 출력 개수 자체가 지수적으로 커질 수 있다.

## SE 수식과 학습 경로

\(A=|Z|\)를 unsigned sparse cycle membership으로 사용하고,
\(L_j=\sum_e A_{ej}\), \(r_e=\sum_j A_{ej}\)로 cycle 길이와 edge별 cycle 수를 정의한다.
Bond embedding \(x_e\)에 대해 학습 경로는 다음과 같다.

\[
v_e=\phi(x_e),\qquad
u_j=\gamma\!\left[\frac{\sum_e A_{ej}v_e}{\max(L_j,1)},\log(1+L_j)\right],
\qquad
w_e=\frac{\sum_j A_{ej}u_j}{\max(r_e,1)}.
\]

SE 조건의 최종 인코딩은 \(v_e,w_e,\log(1+r_e)\), 그리고 포함 cycle의 평균
\(\log(1+L_j)\)·\(1/L_j\)를 MLP에 넣고 \(1[r_e>0]\)를 곱한다.
Bridge와 forest edge의 PE는 정확히 0이다. Empty-cycle batch도 autograd 연결을 보존하며
그 경우 PE parameter gradient가 끊기는 것이 아니라 0이다.

Signed 기저는 cache에서 보존·검증하고, 모델 입력은 sparse COO block-diagonal membership이다.
한 physical batch의 모든 그래프를 `edge → cycle → edge` sparse multiplication으로 처리한다.
그래프별 GPU forward loop나 dense \(m\times\beta\), \(m\times m\) projector를 만들지 않는다.
이 PE는 bond embedding과 결합되어 기존 residual edge-aware GNN, graph pooling, 예측,
loss, backward, optimizer update로 연결된다. PE를 계산만 하고 버리는 경로가 아니다.

## PE 조건: 동일 SE + cycle 상대 위치 residual

DFS에서 cycle을 구성할 때 실제 순서를 chord + head→LCA + reverse(tail→LCA)로
기록한다. CSR 행 번호를 위치로 착각하지 않는다. Cycle \(j\)의 순서에 따라
\(t_{ej}\in\{0,\ldots,L_j-1\}\), \(\theta_{ej}=2\pi t_{ej}/L_j\)로 두고
다음 sparse factor를 캐시한다.

\[
F_{ej}=A_{ej}\cos\theta_{ej},\qquad
G_{ej}=A_{ej}\sin\theta_{ej}.
\]

\(D_L=\operatorname{diag}(\max(L_j,1))\),
\(D_r=\operatorname{diag}(\max(r_e,1))\)일 때

\[
R=D_r^{-1}\left(FD_L^{-1}F^\top V+GD_L^{-1}G^\top V\right),\qquad
W_{\mathrm{SE}}=D_r^{-1}AU,\qquad
W_{\mathrm{PE}}=W_{\mathrm{SE}}+R.
\]

두 조건 모두 위 \(W\)를 **같은 edge MLP 입력 위치**에 넣는다. SE의 column_phi,
cycle_mlp, edge_psi, output 및 backbone은 그대로이며 추가 trainable parameter는 없다.
\(R=0\)이면 동일 state에서 SE와 정확히 같은 연산이 된다.

Cycle별 residual kernel은 \(\cos(2\pi(t_e-t_f)/L)\)다. 균등 평균과 합친 선형 kernel
\((1+\cos(2\pi(t_e-t_f)/L))/L\)은 cycle 내부 상대 간격에 따라 달라지며 정규화된다.
단, 실제 SE 항에는 비선형 cycle_mlp가 있으므로 전체 모델을 이 선형 kernel 하나라고
주장하지 않는다. Sin/cos moment에 따로 비선형 MLP를 적용하지 않고 재결합한 뒤 MLP에 넣는다.
Cycle 시작점을 옮기거나 방향을 뒤집어도 cos 차이는 같으므로 이 선택에는 불변이다.

이것은 **선택한 cycle 안의 상대 PE**다. 다른 chord를 통한 원 그래프 최단거리와 같다고
보장하지 않으며, 모든 노드·엣지에 고유 좌표를 주는 global PE도 아니다. 동일 특징만 있는
대칭적 단일 고리에서는 출력이 동일할 수 있지만, 결합 하나에 표식이 있으면 그 결합까지의
cycle 간격에 따라 위치 항이 달라진다. SE 평균만으로는 이 거리 차이를 전달하지 못한다.
Forest/bridge의 위치 항은 0이며 기존 backbone은 계속 작동한다.

## 보장하는 대칭성과 보장하지 않는 것

Cycle 방향·edge orientation·cycle 열 순서는 \(A=|Z|\)와 공유 집계 때문에 결과를 바꾸지 않는다.
선택된 기저와 edge/node 인덱스를 함께 운반하는 permutation에도 대응한다. 그러나 **DFS tree
선택에 의존하며 임의의 가역 \(Z\mapsto ZR\) 변환에 불변인 projector가 아니다.**
노드 번호를 바꾼 뒤 DFS를 다시 수행하면 다른 tree가 선택될 수 있으므로, 이를 완전한
chart-independent/isomorphism-invariant PE라고 주장하지 않는다. 이는 QR을 제거하고 선택된
cycle의 구조를 학습하는 현재 설계의 명시적 경계다.

## 복잡도

- DFS forest 탐색: \(O(|V|+|E|)\).
- Signed cycle 출력·희소 저장: \(O(|V|+|E|+\operatorname{nnz}(Z))\).
- 각 sparse edge-cycle pass: \(O(\operatorname{nnz}(Z)d)\), 별도로 edge/cycle MLP와 기존 GNN 비용.
- SE는 batch 전체 sparse product 2회, PE는 6회다. PE가 SE보다 빠르다는 주장이 아니며
  위치 항의 추가 시간·메모리와 성능을 별도로 비교한다.
- QR/SVD·Gram inverse·\(O(m\beta^2d)\) projector contraction은 없다.

\(\operatorname{nnz}(Z)\)는 총 cycle 길이이며 일부 그래프에서 이차적으로 커질 수 있다.
따라서 DFS 탐색이 선형이라는 사실을 전체 기저 출력·PE·학습의 엄밀한 선형 시간 보장으로
확대하지 않는다. GPU wall-time 가속 배수도 실제 측정 전에는 주장하지 않는다.

## 모델 크기와 데이터

| Profile | ZINC-12K | Peptides-struct |
|---|---|---|
| `reference` | hidden 128, PE 64, 10 layers | hidden 256, PE 64, 6 layers |
| `large` | hidden 192, PE 96, 12 layers | hidden 320, PE 96, 8 layers |

모두 FFN multiplier 4, dropout 0.1, layer scale 0.1이며 backbone은 축소하지 않았다.
기본 ZINC 모델은 **두 조건 모두 7,262,785 parameters**다. 동일 seed에서 초기 state도 같으며
상대 위치 항은 학습 파라미터를 추가하지 않는다. 이전 projector 모델보다 16,704개 늘었다.
50M parameter ceiling은 초과 시 실패하는 계약이지 작은 모델로 자동 전환하는 기능이 아니다.

ZINC-12K는 공식 train/validation/test 10000/1000/1000, Peptides-struct는 공식
10873/2331/2331과 11개 target을 그대로 사용한다. 새 cache namespace는
`cycle_pe_v2_ordered_dfs_benchmark`다. 원본 공식 데이터는 그대로 사용하지만 구 projector
cache를 이름만 바꾸거나 새 membership으로 간주하지 않는다.

## 실행과 재개

| 설정 | `portable` | `a6000-48gb` |
|---|---|---|
| graph batch | 모든 dataset/profile 32 | reference ZINC/Peptides 512/128, large 256/64 |
| loader | workers 4, prefetch 2 | workers 8, V2 prefetch 4 |
| numeric path | 기본 FP32 | backbone BF16, sparse 집계 FP32, GradScaler off |
| cycle layout | sparse COO block diagonal | sparse COO block diagonal |

A6000 profile은 보이는 VRAM 40GiB 이상과 compute capability 8.0 이상 등 명시된 장치
계약을 검사한다. 아래 실행은 추가로 free VRAM 40GiB를 요구한다. 일반 AMP 요청에서는
BF16 미지원 시 FP32 정책을 기록할 수 있지만 A6000 계약을 만족하지 않는 장치를 묵인하지 않는다.
Capacity probe는 큰 공식 graph batch의 forward/backward 가능성을 확인할 뿐 최적 batch를
증명하지 않는다. OOM이면 조용히 batch·모델·데이터를 줄이지 않는다. 병목 및 메모리 측정 후
변경이 필요하면 사용자 승인과 새 실행 계약을 사용한다.

완료한 V1–V4, Cycle V1, Tree를 재실행하지 않고 **새 Cycle V2만** 실행하는 GPU 3 명령이다.
전체 두 dataset·두 profile·SE/PE 두 조건·seed 0의 **8개 학습**과
각 encoding×dataset 안에서 validation으로 선택한 checkpoint의 test-only 평가 **4개**다.
SE와 PE를 한 profile 경쟁으로 섞지 않는다. 한 조건만 원하면 `--encodings se` 또는
`--encodings pe`를 명시하며, 이때 4학습·2test다. seed를 임의로 늘리지 않는다.
위치 항의 직접 비교는 같은 dataset/profile/seed/hardware 행끼리 한다. 각 조건별 최종
선택값은 서로 다른 profile일 수 있으므로, 그 두 최종값 차이를 위치 항 하나의 효과라고
해석하지 않는다. 두 조건 모두 baseline 대비 이득이 있는지는 별도 no-encoding 실험 없이는
판정하지 않는다. 이번 기본 실행에 no-encoding이나 추가 seed를 몰래 추가하지 않았다.

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

`thin_q`, `--column-chunk-size`, `--basis-pair-budget`, `--basis-execution`은 제거되어 거부된다.
같은 **새 ordered-cycle schema**에서 source/config/artifact hash가 일치하면 동일 run ID의 완료
child를 검증 후 건너뛴다. `<child>/<dataset>/<encoding별 모델 ID>/last.pt`부터 복원한다.
Resume schema는 `cycle-dfs-se-relative-pe-v2-epoch-resume-1`이다.
SE↔PE checkpoint 교환은 state_dict shape가 같아도 거부한다. Model·optimizer·scheduler·
history·RNG 등의 저장 상태를 복원하지만 CUDA kernel의 bitwise 결정성까지 보장하지 않는다.

구 projector checkpoint/cache는 호환되지 않으며 r1/r2/r3를 새 모델에 강제로 resume하지 않는다.
V5도 diffusion source가 변경되었으므로 구 source checkpoint의 resume 거부를 우회하지 않는다.
새 모델만 실행하는 것과 같은 모델의 중단 후 재개를 구분한다. 통합 전체 실험은
[RICH_SCALING_EXPERIMENTS.md](RICH_SCALING_EXPERIMENTS.md)를 참고하되 완료한 트랙을
불필요하게 다시 계획하지 않는다.

## 검증 범위와 역사적 GPU 실패

IPC 수정 후 전체 로컬 CPU 회귀는 **1,940 passed / 99 skipped** (189.74초)다.
이 실행에는 1만 합성 그래프의 실제 병렬 준비와 cache 재검증도 포함됐다.
Windows 호스트에서 private storage·값·순서 보존은 확인했지만 Linux mmap/FD 수 실측과
native renameat2, 실제 PyG 데이터 학습·GPU 성능은 확인하지 못했다.

2026-09-05 SE/PE 분리 당시 전체 로컬 CPU 회귀는 **1,852 passed / 98 skipped / 1 warning**
(164.78초)이며 Ruff와 코드 스냅샷 일치 검사도 통과했다. 생략은 Linux/Bash, Windows
symlink 권한, 미설치 PyG 및 CUDA/BF16 환경 조건 때문이다. 경고는 PyTorch multiprocessing의
sparse tensor 재구성 시 내부 invariant-check 정책 경고이며, 저장 데이터의 불변식은 별도로 검사한다.

그 뒤 ad041e2 서버 실행은 GPU 학습 전에 두 dataset 모두 약 7천 그래프에서
`rebuild_storage_fd -> _new_shared_fd_cpu -> mmap ENOMEM`으로 실패했다.
작업자가 반환한 Graph의 작은 Tensor storage들을 부모가 split 전체 동안 유지해
공유 mmap/FD가 누적된 경로였다. 정확한 서버의 OS 한도는 측정하지 않았지만
4~732 bytes 매핑도 실패하는 양상은 `vm.max_map_count` 고갈과 잘 맞는다.
이전 소규모 process-pool smoke는 이 누적을 검증하지 못했다.

현재는 pool에 보내는 공식 graph/cache 입력과 돌아오는 결과 모두 Tensor-free owned
NumPy byte payload로 전달하고 부모의 일반 CPU storage로 복원한다. Sparse indices/values,
dtype, shape, 모든 cycle/위치/샘플 및 순서를 유지하며 cache 검증에도 같은 경로를 적용한다.
Pool workers, bounded chunks, 모델, 기저 알고리즘, batch 및 dataset을 줄이지 않는다.
공식 InMemoryDataset의 큰 backing storage를 샘플마다 복제하지 않도록 필요한 graph slice만
보낸다. Serial/parallel 값 동치, 양방향 IPC에서 torch storage reducer가 호출되지 않는 계약,
복원 결과가 shared storage가 아닌지 검사하며 10,000-graph 합성 stress는 별도 opt-in debug다.
이는 실제 Linux kernel 한도나 서버 GPU 전체 학습 검증을 대신하지 않는다.

현재도 cache는 split 전체를 완성한 뒤 저장하므로 전처리 중 중단된 미완성 split은 다시 준비한다.
epoch checkpoint 재개와 전처리 중간 재개를 혼동하지 않는다. 모델/PE/cache 내용 형식은
ad041e2와 같지만 실행 source identity는 달라 이전 실패 run에 강제로 이어 붙이지 않는다.

현재 구현은 CPU에서 좌영공간·기저 독립성·실제 cycle 순서·sparse SE/PE·gradient·배치·재개
계약을 검증한다. 위치 항의 explicit cosine-kernel 동치, 시작점/방향 불변성, 표식 결합까지의
cycle 상대 간격 구별과 R=0일 때 SE 동치를 별도로 검사한다.
**두 조건의 실제 GPU 전체 학습·성능·VRAM·정확도 개선 결과는 아직 없다.**
실제 Linux child는 주기적 GPU/CPU/RAM·throughput과 지원되지 않는 counter의 원인을 기록한다.
등록 batch나 capacity probe, optimizer/validation/checkpoint를 제외한 microbenchmark 결과를
전체 학습 최적화나 최종 성능으로 표현하지 않는다.

아래는 현재 모델과 분리된 역사 기록이다.

- `new-v5-cyclev2-a6000-gpu3-seed0-r1-cycle`: 구 projector cache 준비 이후
  reference/large 네 학습이 FP16 GradScaler 초기 배율 65,536의 gradient overflow로 실패했다.
- `new-v5-cyclev2-a6000-gpu3-seed0-r2-cycle`: 네 학습이 다시 첫 epoch 전에 실패했다.
  첨부 `bd63fc9a-60da-4daf-9ab9-da49db7cbbe1/pasted-text.txt` SHA-256은
  `F797F10F2D81BF23ED269DB698817EEEA99DB3F70DEBD3D0D68119C2917431D6`다.
  `benchmark.py:589`는 `08d8ed6`의 FP16+GradScaler 경로와 일치하므로 `214265c`의
  BF16/no-scaler 수정을 검증한 결과가 아니었다.
- 위 실패 checkpoint·cache를 현재 sparse 모델로 재사용하거나 SOTA 비교 값으로 인용하지 않는다.
  r3에서 수령한 V5 large OOM은 [CONDUCTANCE_V5.md](CONDUCTANCE_V5.md)에 별도로 기록한다.
