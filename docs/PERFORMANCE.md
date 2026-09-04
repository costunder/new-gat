# GPU 실행 최적화와 속도 확인

이 기능은 학습 수식·파라미터·공식 split·loss·validation 체크포인트 선택을 바꾸지 않는다.
PyTorch의 사전 빌드된 C++/CUDA 연산을 사용한다. 프로젝트 전용 C++ 확장을 설치하거나,
시스템 CUDA·glibc·드라이버를 변경하지 않는다. 실제 성능 향상은 아래 GPU 측정으로 확인한다.

이 최적화는 현재 소스 버전에 포함된다. 이전 진단 전용 commit `ebf8cd1`에는 없었으며,
GPU 가속 실측은 아직 받지 않았다. 기존 benchmark 결과·checkpoint 진단과의
구분은 [실험 상태](../gpt_handoff/EXPERIMENT_STATUS.md)를 따른다.

별도 Conductance [v2](../gpt_handoff/CONDUCTANCE_V2.md),
[v3](../gpt_handoff/CONDUCTANCE_V3.md), [V4 통합 문서](../gpt_handoff/CONDUCTANCE_V4.md)는 아래 기존 benchmark compile/속도 도구의
대상이 아니다. V2는 직접 C 전파의 정확한 chunked 1차 backward를, v3는 대칭 전파의
정확한 chunked 1차 backward와 공유 MLP activation checkpointing을 자체 실행 경로에서 쓴다.
V4는 잔차 상태와 `HW` 메시지를 분리한 대칭 전파의 정확한 chunked 1차 backward를 사용하고,
상대 C 조건에서는 v3와 같은 shared generator checkpointing을 사용한다.
세 경로 모두 neighbor sampling이나 `torch.compile` 옵션을 제공하지 않는다. V2와 V3/V4의
transductive 데이터는 full-graph batch 1이며, V3/V4의 PPI는 graph를 자르지 않는 whole-graph
minibatch 2를 사용한다.
특히 v3의 전파 자체는 O((n+m)d)이지만 width가 d에 비례하는 C 생성 MLP는 O(md²) 작업을
요구한다. Chunking/checkpointing을 전체 모델 메모리 상수화나 실측 가속으로 표현하지 않는다.
현재 확대된 V2/V3/V4 8/10/20개 기본 실행의 GPU 시간·peak memory는 수령하지 않았다.
과거 arxiv-only runner 상태를 전체 비교의 실측으로 재사용하지 않으며, 각각의 완전한
`comparison.md`가 보존되기 전에는 확정값으로 기록하지 않는다.

## 기본 적용

| 경로 | 최적화 |
|---|---|
| 기본 Conductance benchmark | 그래프 수를 CPU 메타데이터에서 읽어 층별 GPU scalar 조회 제거; split 인덱스 1회 준비; loss/PPI 지표 device 누적 |
| Cycle PE v1/v2 | categorical embedding 전체 stack 제거; pooling의 크기가 고정된 count 집계; message stack의 연결·차수 정보 1회 계산 |
| Cycle PE v2 | 그래프·기저 열의 독립 context를 유지하면서 여러 그래프의 기저 MLP 연산을 묶는 batched 경로 |
| 학습 기록 | train loss/지표와 `epoch_seconds` 출력·저장; 로그 파일 즉시 flush |

기본 재현 명령을 그대로 사용한다. legacy Conductance와 Cycle PE는 기본 AMP OFF를
유지한다. Conductance V5는 선택한 hardware profile을, Tree Augmentation은
`config.yaml`의 현행 AMP 설정을 그대로 쓴다. 정확성 보호를 위한 nonfinite 검사도
유지한다. 모든 GPU 동기화를 제거한 것은 아니다.

Cycle PE v2는 전체 signed 기저를 그대로 사용한다. `f(u)`/`f(-u)` 평균, 열별 전체 엣지 context,
모든 열의 합과 `sqrt(beta)` 정규화를 보존한다. 부동소수점 합산 순서가 달라질 수 있으므로
bitwise 동일한 학습 궤적은 보장하지 않는다. 파라미터 이름과 checkpoint 형식은 유지한다.

`--basis-pair-budget`의 기본값은 `32768`이다. 한 MLP 호출에 들어가는 엣지×기저열 pair
수를 제한하며, 기저 rank를 자르지 않는다. 모든 기저·역전파 activation의 총 메모리를
이 값으로 제한한다는 뜻은 아니다. 원래 Cycle PE v2 실행 경로와 비교하려면 다음처럼 실행한다.

```bash
bash research/cycle_pe/v2/reproduce.sh --basis-execution reference
```

이것은 같은 모델의 실행 방식 비교이며 다른 PE 가설이나 경쟁 모델이 아니다.
SVD는 여전히 최초 데이터 준비 시 CPU에서 수행하고 캐시한다. 전처리 GPU화는 포함하지 않는다.

## 선택적 컴파일

```bash
bash research/conductance_gat/reproduce.sh --compile
```

```bash
bash research/cycle_pe/v2/reproduce.sh --compile
```

Cycle PE v1도 `--compile`을 지원한다. 이 옵션은 반복 호출되는 tensor MLP 블록만
`torch.compile`의 Inductor backend로 컴파일하며 기본값은 OFF다. 가변 기저의 Python
작업 분할 루프 전체를 추적하지 않는다. Tree Augmentation 및 보조 core/all suite에는 적용하지 않는다.
첫 호출은 컴파일 시간이 포함되고, 가변 그래프 크기는 재컴파일을 일으킬 수 있다.
따라서 작은 데이터나 실행 환경에 따라 느려질 수도 있다.

PyTorch가 요구하는 compiler/runtime이 없는 환경에서는 컴파일이 실패할 수 있다.
특히 기존 Ubuntu 18.04 Singularity/legacy-cu118 환경에서 실제 호환성은 별도 확인이 필요하다.
자동으로 컴파일러·패키지를 설치하거나 compiler 오류를 잡아 eager 성공으로 숨기지 않는다.
PyTorch 자체의 shape guard/cache 경고는 로그에서 확인한다. `dynamic=True`도 모든 재컴파일을
없애지는 않는다. 컴파일 대상 블록 목록은 `execution.compiled_modules`에 기록한다.
기존 CUDA 실행을 사용하려면 새 run에서 `--compile`을 빼면 된다.

## 실제 데이터 GPU 속도 비교

데이터 준비와 환경 설치를 끝낸 뒤 실행한다. 다운로드·자체 생성 데이터·CPU 학습 fallback은 없다.

```bash
bash scripts/benchmark_speed.sh --track conductance_gat
```

```bash
bash scripts/benchmark_speed.sh --track cycle_pe_v2
```

현행 학습 경로의 physical mini-batch 후보를 실제로 비교하려면 후보를 명시한다.

```bash
bash scripts/benchmark_speed.sh --track cycle_pe_v1 --dataset zinc12k --batch-sizes 16 32 64
```

```bash
bash scripts/benchmark_speed.sh --track cycle_pe_v2 --dataset zinc12k --batch-sizes 16 32 64
```

```bash
bash scripts/benchmark_speed.sh --track tree_augmentation --dataset csl --tree-arm fixed_bfs --batch-sizes 8 16 32
```

```bash
bash scripts/benchmark_speed.sh --track tree_augmentation --dataset csl --tree-arm multi_chart --batch-sizes 8 16 32
```

```bash
bash scripts/benchmark_speed.sh --track tree_augmentation --dataset zinc --tree-arm multi_chart --batch-sizes 8 16 32
```

Tree의 기본 cache 위치는 `research/tree_augmentation/data`다. 다른 위치에 검증된
전체 cache를 준비했다면 `--tree-data-root`로 지정한다. CSL/ZINC cache가 없을 때
benchmark가 다운로드하거나 작은 대체 데이터로 바꾸지 않고 해당 run을 실패시킨다.
Tree의 두 arm은 입력 chart view 자체가 다르므로 한 run 안에서 서로를 속도 variant로
취급하지 않는다. `--tree-arm fixed_bfs`와 `--tree-arm multi_chart`를 별도로 실행해야
각각의 현행 학습 batch를 측정한다.

V5 A6000/ogbn-arxiv의 실제 cluster seed batch 후보는 다음처럼 측정한다.

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/benchmark_speed.sh \
  --track conductance_v5 \
  --dataset ogbn-arxiv \
  --v5-scale-profile reference \
  --v5-hardware-profile a6000-48gb \
  --v5-sampling cluster \
  --batch-sizes 1024 2048 4096 \
  --device cuda:0 \
  --resource-sample-interval-seconds 0.1 \
  --output-dir runs/performance/v5-arxiv-a6000-batch-sweep-r1
```

`--output-dir`는 기존 결과를 덮어쓰지 않는 새 경로여야 한다. 같은 이름이 이미 있으면 기존
artifact를 수정하지 않고 실패하므로 다음 측정에는 새 run suffix를 쓴다.

컴파일까지 비교하려면 다음 명령을 사용한다.

```bash
bash scripts/benchmark_speed.sh --track cycle_pe_v2 --include-compile
```

기본 데이터는 legacy Conductance와 Conductance V5의 Cora, Cycle PE V1/V2의
ZINC-12K, Tree Augmentation의 CSL이다. `--dataset`으로 해당 트랙의 다른 지원
데이터를 선택한다. 공식 전체 cache를 한 번 검증·적재한 뒤, 각 후보에서 현행 학습
sampler가 만드는 첫 번째 deterministic real training batch를 사용한다. 후보를 만들기
위해 train split을 자르거나 작은 데이터로 바꾸지는 않는다.

Cycle PE V1은 현행 seeded/shuffled DataLoader, `CyclePEModel(64, 32, 3)`과 MAE를
그대로 쓴다. Tree는 full/reference `config.yaml`의 128 hidden, 8 message layers,
AMP 정책과 padded `PaddedChartBatch`를 사용한다. CSL은 cross-entropy, ZINC는 선택한
arm의 전체 unique training graph target으로 계산한 정규화 MSE를 그대로 사용한다.
Tree sampler는 chart view를 replacement sampling하므로 Tree의 physical batch 단위는
`chart_views`이며, 같은 물리 graph가 한 batch에 여러 번 나타날 수 있다. report에는
unique physical graph 수도 별도로 남긴다.
Tree 정식 direct/scaling 실행의 DataLoader 기본은 workers 4, worker당 prefetch factor
2다. 각 seeded loader를 한 번 완전히 소비한 뒤 폐기하므로 `persistent_workers=false`이며,
이 세 값은 실제 runtime telemetry에 함께 기록되고 scaling coordinator가 검증한다.

- 기본 warmup 5회 후 forward+backward 20회를 측정한다. 첫 호출·warmup 비용은 별도 기록한다.
- optimizer update, 데이터 읽기/전송, validation/test, checkpoint 저장을 제외한 **고정 batch
  연산 측정**이다. 전체 학습 시간이나 최종 정확도의 개선으로 해석하지 않는다.
- 각 measured variant는 최소 2초 동안 CUDA-synchronized forward/loss/backward를
  반복한다. 이는 background resource sampler가 실제 GPU 작업 구간을 관측하게 하려는
  하한이며, 요청한 step 수를 줄이지 않는다.
- Conductance reference는 층별 graph-count scalar 조회가 있던 기본 benchmark 경로다.
  모든 이전 코드와의 end-to-end 비교는 아니다. Cycle PE v2는 reference/batched 기저
  인코더를 비교한다. Cycle PE V1은 current/optional compiled, Tree는 compile을 지원하지
  않는 exact current 경로만 측정한다.
- 같은 후보의 execution variant는 동일한 초기 state에서 eval prediction, loss와 모든
  trainable parameter gradient를 검사한다. 모든 track에서 finite prediction/loss/gradient,
  loss에 연결되지 않은 trainable parameter 부재, 측정 전후 parameter 불변성을 확인한다.
  optimizer step과 parameter update는 모두 0이다.
- 첫 current/reference row에는 독립 oracle이 아직 없으므로 자기 자신과 비교해
  equivalence passed로 표시하지 않고 `not_applicable`로 기록한다. legacy Conductance와
  Cycle PE V2의 실제 optimized row, 또는 명시적으로 요청한 compiled row처럼 독립 실행
  variant가 있을 때만 수치 equivalence를 `passed`로 기록한다. V5/Cycle V1/Tree의 기본
  current-only run은 대신 정확한 production import path와 finite loss/gradient/no-update
  integrity를 별도 필드로 남긴다.
- 결과는 새로운 `runs/performance/` 하위 폴더의 `summary.csv`, `report.json`에 기록한다.
  논문 성능 집계인 `runs/paper/`와 분리한다. 실패하면 실패 기록과 비정상 종료로 알린다.
- 각 후보/variant는 GPU SM utilization mean/max, memory-controller utilization mean/max,
  CUDA peak allocated/reserved VRAM, process CPU/RAM, system available RAM, step/s와
  physical-batch item/s를 기록한다. GPU utilization은 device-wide 값이므로 공유 GPU의
  다른 process가 섞일 수 있으며, NVML/driver가 제공하지 않는 값은 임의의 0이 아니라
  `null`과 측정 불가 이유로 남는다.
- `report.json`의 원문 시계열은
  `batch_candidates[*].variants[*].resource_observability.interval_series` 아래
  `gpu_sm_utilization_percent`, `gpu_memory_controller_utilization_percent`,
  `gpu_allocator_allocated_bytes`, `gpu_allocator_reserved_bytes`,
  `gpu_device_free_bytes`, `gpu_device_used_bytes`, `process_cpu_seconds`,
  `allocated_cpu_busy_seconds`, `allocated_cpu_total_seconds`, `process_resident_bytes`,
  `process_peak_resident_bytes`, `system_available_bytes`에 있다. Caller-defined workload peak와
  coordinator-process/allocated-CPU 평균은 같은 객체의 `summary`에 있으며, 후보별 배제 이유와
  optimizer 없는 microbenchmark 순위는 최상위
  `batch_candidate_analysis`를 확인한다. 이 분석은 최종 학습 batch를 선택하지 않는다. 각 관측치는
  `value/unit/reason` 계약을 사용한다.
- OOM 또는 어떤 실행 오류도 더 작은 후보로 조용히 축소하거나 CPU/model fallback하지
  않는다. 실패한 후보를 `failed`와 정확한 reason으로 기록하고 나머지 명시 후보를
  독립적으로 계속 측정한 뒤 전체 run을 실패 상태로 남긴다.
- Cora/Citeseer/PubMed/ogbn-arxiv의 full-graph mode에는 physical mini-batch가 없으므로
  batch 후보 sweep을 거부하고 compatibility 값 1회만 받는다. V5 sampled ogbn-arxiv의
  단위는 seed nodes, PPI와 Cycle의 단위는 graphs, Tree의 단위는 chart views다.
- 속도 비율은 해당 GPU·데이터·batch의 실측값이다. warmup으로 compiler cache가 준비된
  상태와 실제 전체 실험의 cold start 비용을 구분한다.

정식 학습의 `history.json`에 있는 `epoch_seconds`는 CUDA 동기화 후 측정한 train+validation
시간이다. checkpoint/history 파일 쓰기는 제외하며 첫 epoch에는 컴파일 준비 비용이 포함될 수 있다.
실행 옵션은 manifest와 모델별 `metrics.json`의 `execution`에 남긴다.

## 검증 범위

출력·입력/파라미터 gradient, 빈 엣지·forest·가변 rank·작은 pair budget·부호/열 순서
불변성의 개발 단위 검사를 수행한다. 이 검사 통과를 서버 GPU 속도 향상이나 논문 성능
개선으로 간주하지 않는다. 구현 환경에서 GPU 실측을 하지 않았다면 가속 배수를 제시하지 않는다.
