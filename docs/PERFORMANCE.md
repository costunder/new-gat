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

기본 재현 명령을 그대로 사용한다. AMP는 여전히 기본 OFF이며, 정확성 보호를 위한
nonfinite 검사도 유지한다. 모든 GPU 동기화를 제거한 것은 아니다.

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

컴파일까지 비교하려면 다음 명령을 사용한다.

```bash
bash scripts/benchmark_speed.sh --track cycle_pe_v2 --include-compile
```

기본 데이터는 기본 Conductance benchmark의 Cora, Cycle PE v2의 ZINC-12K다.
`--dataset`으로 해당 트랙의 다른
지원 데이터를 선택한다. 공식 train split의 동일 입력 batch와 동일 초기 파라미터에서
출력·모든 parameter gradient를 먼저 비교하고, 통과한 실행을 GPU에서 측정한다.

- 기본 warmup 5회 후 forward+backward 20회를 측정한다. 첫 호출·warmup 비용은 별도 기록한다.
- optimizer update, 데이터 읽기/전송, validation/test, checkpoint 저장을 제외한 **고정 batch
  연산 측정**이다. 전체 학습 시간이나 최종 정확도의 개선으로 해석하지 않는다.
- Conductance reference는 층별 graph-count scalar 조회가 있던 기본 benchmark 경로다.
  모든 이전 코드와의 end-to-end 비교는 아니다. Cycle PE v2는 reference/batched 기저
  인코더를 비교한다.
- 결과는 새로운 `runs/performance/` 하위 폴더의 `summary.csv`, `report.json`에 기록한다.
  논문 성능 집계인 `runs/paper/`와 분리한다. 실패하면 실패 기록과 비정상 종료로 알린다.
- 속도 비율은 해당 GPU·데이터·batch의 실측값이다. warmup으로 compiler cache가 준비된
  상태와 실제 전체 실험의 cold start 비용을 구분한다.

정식 학습의 `history.json`에 있는 `epoch_seconds`는 CUDA 동기화 후 측정한 train+validation
시간이다. checkpoint/history 파일 쓰기는 제외하며 첫 epoch에는 컴파일 준비 비용이 포함될 수 있다.
실행 옵션은 manifest와 모델별 `metrics.json`의 `execution`에 남긴다.

## 검증 범위

출력·입력/파라미터 gradient, 빈 엣지·forest·가변 rank·작은 pair budget·부호/열 순서
불변성의 개발 단위 검사를 수행한다. 이 검사 통과를 서버 GPU 속도 향상이나 논문 성능
개선으로 간주하지 않는다. 구현 환경에서 GPU 실측을 하지 않았다면 가속 배수를 제시하지 않는다.
