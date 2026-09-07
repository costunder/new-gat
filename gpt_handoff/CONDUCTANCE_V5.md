# Conductance GAT V5 — graph-specific C optimization and weighted-Laplacian propagation

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
