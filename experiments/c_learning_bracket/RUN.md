# 실행 계약과 결과 읽기

## 이번 후속 수정의 증거 범위

초기·epoch별 validation, 선택 checkpoint의 전체 validation·새 문맥 평가에도
`expected_seed_ids`를 전달한다. `*-initial.json`, `*-epoch-*.json`,
`*-trained.json`, `evaluation.json`에 같은 `seed_coverage`가 남는다.
최종 새 문맥 평가의 parameter hash도 선택 checkpoint의 전체 평가와 대조한다.

이번 패키지의 실제 데이터 측정은 이전 signal_audit_2의 기록이다.
현재 소스 hash와 다르므로 현재 본학습용 calibration으로 사용할 수 없다.
기존 엄격한 소스 검사도 유지했다. 아래 calibration 명령을 새 경로에서 실행한 뒤
그 결과로 본학습한다. 이번 작업에서는 평가 연결 테스트만 다시 실행했다.

## 본학습 설정

ogbn-arxiv 전체 원본 그래프와 공식 train/validation split, 기존 cluster-disjoint
sampling을 유지한다. 8층·hidden 256·8 heads, dropout 0.2, AdamW lr 0.0005,
weight decay 0.01, gradient clipping norm 5, 200epoch 이상 전체 train seed pass다.
매 epoch 각 train seed를 정확히 한 번 감독하며 두 조건의 샘플 순서 hash를 확인한다.

physical batch는 동시에 계산하는 감독 seed 수이고, 여러 문맥은 disjoint union으로
묶는다. context seed 수는 문맥 크기를 정한다. 둘을 구분한다.
accumulation=1, 단일 할당 GPU 경로다. 다른 GPU 수는 명시적 분산 설계 없이 실행하지 않는다.
transductive 경로는 DataLoader를 사용하지 않으며, 문맥 생성 thread pool과
선택한 prefetch·pinning, 정적 그래프 cache를 사용한다.
이번 입력 경로는 전송 직전 CPU 텐서별 `is_pinned()`를 검사해 로그에 남긴다.
기존 `portable` 실행 profile이 적용되어 실제 설정은 FP32, TF32 off,
pin_memory on, sample_prefetch off다. 공유 hardware validator가 이 네 값을
profile 값으로 확정하므로, 같은 이름의 CLI flag보다 profile이 우선한다.
실제 사용값은 `contract.json.configuration`으로 확인한다. context worker의
병렬 문맥 생성은 prefetch와 별도로 작동한다.

## Calibration

```powershell
.\.venv-gpu\Scripts\python.exe -m experiments.c_learning_bracket.train `
  --action calibrate --output-dir results/bracket-calibration-new `
  --sample-seed-batch-size 2048 --sample-context-seed-batch-size 2048 `
  --sample-context-workers 2 --edge-chunk-size 16384 `
  --physical-seed-candidates 2048 4096 --context-worker-candidates 2 4 `
  --eval-context-seeds 2048 4096
```

후보값은 측정 대상이며, 항상 작은 batch를 택한다는 뜻이 아니다.
각 조합에서 learned/fixed를 모두 측정한다. 모델과 데이터를 축소하지 않는다.
일반 step 시간과 메모리 여유를 비교한다. 짧은 calibration의 처리량을
200epoch 전체 실행의 평균 처리량으로 보장하지 않는다.

각 조건에서 warmup 후 최소 세 개의 실제 배치를 사용한다. 각 배치에서 모델·optimizer
상태·dropout 난수를 복원하여 일반 step과 관측 step을 짝지어 측정한다.
관측 추가 비용은 `observed.step_wall_seconds - plain.step_wall_seconds`로 기록한다.
복사·복원 시간은 그 측정에서 제외한다. 일회성 수치에는 측정 변동이 포함된다.

`plain.timing`에는 forward+CE, backward, gradient 검사·clipping, optimizer를 따로 기록한다.
`observed.timing`의 forward/backward는 관측을 포함한다고 명시하고, 같은 입력 재실행도
별도로 기록한다. CUDA event 시간과 CPU 제출 시간을 합산하지 않는다.
`supervised_nodes_per_second`는 측정된 sampling/transfer 대기와 일반 step의
합으로 계산한다. 실제 portable profile은 prefetch가 꺼져 있으므로 관측 step 중
다음 배치를 미리 만드는 효과는 없다. 이후 prefetch를 켜는 별도 profile을
추가한다면 관측 step이 다음 배치 준비를 가리는 효과를 따로 측정해야 한다.
관측 시간까지 GPU 기본 모델 비용에 더하지 않는다.
전체 validation, C=1 개입, 선언된 새 문맥 평가도 실행해 메모리에 들어가는지 확인한다.
calibration 중 계산한 정확도나 몇 번 갱신한 모델을 본학습 결과로 해석하지 않는다.

## 본학습

위 명령에서 `--action train`, 새 `--output-dir`, 측정된 physical batch/worker를 사용하고
`--calibration-report`에 이 경로의 calibration.json을 지정한다. 나머지 과학 설정은 유지한다.
기존 v2.0 calibration은 재사용할 수 없다. 소스 hash나 설정이 다르면 오류로 중단한다.
두 조건 모두에서 관측·평가를 포함해 VRAM 여유 10% 이상을 확인해야 한다.

2048 seed·worker 4로 본학습할 때의 명령은 다음과 같다.
수정판 측정 결과와 메모리 여유를 먼저 확인한다.
**이 명령의 200epoch 본학습은 아직 실행하지 않았다.**

```powershell
.\.venv-gpu\Scripts\python.exe -m experiments.c_learning_bracket.train `
  --action train --output-dir results/bracket-train-new `
  --sample-seed-batch-size 2048 --sample-context-seed-batch-size 2048 `
  --sample-context-workers 4 --edge-chunk-size 16384 `
  --eval-context-seeds 2048 4096 `
  --calibration-report results/bracket-calibration-new/calibration.json
```

기존 저장소에서 실행하는 명령이다. 검토 ZIP의 코드만 별도 환경에서 실행한다면
실제 데이터 경로와 의존성을 준비하고 새 환경에서 calibration을 먼저 수행한다.

본학습은 두 조건을 순서대로 실행한다. 각 모델 내부의 문맥들은 배치로 계산한다.
모든 epoch를 끝내며 early stopping은 하지 않는다. validation으로 checkpoint를 선택하고,
재로딩해 정답 수와 parameter hash를 확인한 뒤 두 checkpoint를 고정한다.
동일 그래프의 새 sampler 문맥을 평가하지만 독립 그래프 일반화라고 부르지 않는다.
공식 test split 평가와 중단된 본학습 resume은 이번 실행기에 없다.

## 결과 파일

- `contract.json`: 실제 설정·source hash·자원. backend는 bracket으로 기록한다.
- `timing-*.json`: 같은 배치·상태에서의 일반/관측 step 시간.
- `calibration-*.json`, `calibration.json`: 조합별 처리량·peak VRAM·전체 평가 반환값.
- `inspection-*.json`: calibration 관측. 본학습 관측과 구분해서 표시한다.
- `learned-initial.json`, `fixed-initial.json`: 공통 초기 hash·데이터·초기 예측.
- `*-inspection-epoch.json`: 학습 중 score·C·alpha 전후·gradient·갱신량.
- `*-epoch-*.json`: CE, validation, 전체 train seed coverage, 자원.
- `*-best-*.pt`, `frozen-checkpoints.json`, `evaluation.json`: 선택·고정·개입 평가.
- `failure.json`: 실행 중 예외·traceback. 수식 변경이나 fallback 없이 실패를 기록한다.

수정판의 calibration `inspection-*.json`은 `scope=real-data calibration only`로
기록한다. 초기판의 일반 관측 문구가 이를 덮어쓰던 것도 수정했다.

signal_audit_2부터 각 calibration 조합의 `evaluation.full_validation`과
`evaluation.new_sampling_contexts[context]`에 정답 수, 전체 수, CE, accuracy,
parameter SHA256, C=1 개입 결과를 보존한다. 세 평가의 파라미터 hash는 같아야 한다.
`seed_coverage`는 실제 평가한 원래 노드 ID와 공식 validation ID를 정렬 대조한다.
중복·누락·다른 ID가 있으면 개수가 같더라도 실패한다. 기대/관측 ID hash도 저장한다.
새 문맥 결과의 `sampling_evidence`는 sampler가 전체 pass 종료 후 확인한 기록이다.
이 내부 기록의 `every_train_seed_exactly_once`는 재사용한 sampler의 필드 이름이며,
여기서는 validation ID를 순회한다. CE 값은 평가 지표로 계산하지만,
backward와 optimizer 갱신은 수행하지 않는다.
평가용 physical seed batch는 context 크기의 배수로 올림한 실제 값을 함께 저장한다.

`inspection-*.json`의 `layers[].signal_stages_before_update`는 입력 H부터
value·혼합·출력 투영·ReLU·dropout까지 실제 forward의 RMS를 기록한다.
이번 모델에서도 상세 관측은 본학습 epoch 첫 배치에만 수행한다.

고정 C=1에서도 sampling correction이 비균일하면 alpha는 균등하지 않다.
같은 입력에서 alpha가 바뀌었는지, 개입 시 예측이 바뀌는지, 별도 fixed보다 성능이
나은지를 구분한다. `run_completed=true`를 연구 성공 판정으로 대체하지 않는다.

## 검사

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest experiments/c_learning_bracket/test_debug.py -q -p no:cacheprovider
```

작은 그래프는 합성 단위검사 전용이다. CUDA 파이프라인 smoke는 모델 8/256/8을
유지한 합성 2epoch이며 저장 결과를 debug로 표시한다. production CLI는 이를 허용하지 않는다.
실제 수행 범위와 수치는 `REVIEW_FIXES.md`를 확인한다.
