# 실행과 결과 읽기

## 유지하는 본학습 계약

- ogbn-arxiv의 기존 전체 데이터와 공식 train/validation split.
- 8층, hidden 256, 8 heads, dropout 0.2, 기존 초기화·출력 투영.
- AdamW, lr 0.0005, weight decay 0.01, gradient clipping norm 5.
- 200 epoch 이상. 매 epoch 모든 train seed를 정확히 한 번 감독한다.
- 기존 cluster-disjoint sampler. context 크기와 physical batch 크기는 구분한다.
- physical batch는 동시에 계산하는 감독 seed 수다. 여러 문맥을 disjoint union으로 합친다.
- accumulation=1, 단일 할당 GPU. 여러 GPU 환경은 별도 분산 실행 설계가 필요하며 조용히 한 장으로 대체하지 않는다.
- 기존 그래프 cache·문맥 thread pool을 재사용한다. DataLoader workers=0은 이 transductive 경로에 DataLoader가 없기 때문이다.
- calibration에서 physical batch 후보와 context worker 후보를 각각 두 개 이상 측정한다.
- 검사 관측을 포함한 optimizer step과 전체 validation/C=1 평가 메모리도 측정한다.

아래 후보값은 **측정할 값**이다. 최종 batch 선택이나 이미 측정된 권장값이 아니다.
context 2048은 기존 문맥 조건을 유지하는 예이며, 다른 문맥 계약을 택하면 별도 calibration이 필요하다.
전체 입력·샘플 그래프를 줄여 OOM을 숨기는 fallback은 없다.

## 1. 새 경로에서 calibration

저장소 루트에서 PowerShell로 실행한다. 기존 데이터 cache가 필요하며 자동 다운로드하지 않는다.

```powershell
.\.venv-gpu\Scripts\python.exe -m experiments.c_learning_only.train `
  --action calibrate --output-dir results/c_learning_only/calibration-001 `
  --sample-seed-batch-size 4096 --sample-context-seed-batch-size 2048 `
  --sample-context-workers 2 --edge-chunk-size 16384 `
  --physical-seed-candidates 4096 8192 --context-worker-candidates 2 4 `
  --eval-context-seeds 2048 4096 --sample-prefetch
```

CLI는 두 조건을 모두 측정한다. 첫 step은 warmup, 중간 두 step은 일반 학습,
마지막 step은 상세 관측을 포함한다. `calibration-repeats`를 늘릴 수 있다.
이 측정은 최종 학습이나 정확도 평가가 아니다. 새 실행 경로의 실제 측정은 아직 하지 않았다.

## 2. 측정 결과로 본학습 실행

`calibration.json`의 두 조건 모두에서 메모리 여유 10% 이상인 physical/worker 조합을
찾고 처리량을 비교한다. 같은 조합을 양쪽 학습에 사용한다. 모형·문맥 크기는 바꾸지 않는다.

같은 명령의 `--action`을 `train`으로, `--output-dir`을 새로운 run 경로로 바꾸고,
측정으로 선택한 `--sample-seed-batch-size`와 `--sample-context-workers`를 넣는다.
다음을 추가한다.

```text
--calibration-report results/c_learning_only/calibration-001/calibration.json
```

두 조건을 순서대로 학습해 GPU 메모리에 동시에 두 학습 모델을 올리지 않는다.
각 모델 내부의 여러 샘플 문맥은 병렬로 계산한다. 공통 초기 hash, epoch별 전체
샘플/감독 순서 hash가 다르면 오류로 중단한다. 데이터·source·정밀도·문맥 등
calibration 계약이 달라도 오류로 중단한다. 기존 출력 디렉터리는 재사용하지 않는다.

## 3. 파일 읽는 순서

1. `contract.json`: 실제 적용 설정, 소스 hash, 자원, 실행 범위.
2. `learned-initial.json`, `fixed-initial.json`: 공통 초기값 일치, 초기 예측, 데이터 계약.
3. `learned-inspection-0001.json` 등: 비용·C·alpha 전후, live C gradient,
   생성기 gradient·갱신량, 원래 노드 ID가 있는 이웃 표. 배열 원소는 head 순서다.
4. `*-epoch-*.json`: 매 epoch CE, validation, 전체 train seed coverage, 샘플 순서 hash, 자원.
5. `*-best-*.pt`, `frozen-checkpoints.json`: validation으로 선택하고 고정한 checkpoint.
6. `evaluation.json`: 두 조건의 전체 validation, C=1 개입, 새 문맥에서의 frozen 평가.

선택된 checkpoint는 재로딩한 뒤 validation 정답 수와 파라미터 hash를 확인한다.
새 문맥 평가는 학습 epoch 범위 밖의 sampler seed를 사용한다. 모든 validation seed를
사용하며, 양쪽 조건에서 동일한 문맥 순서를 확인한다. 독립 그래프 일반화 주장은 하지 않는다.
공식 test split은 이번 실행기에서 평가하지 않는다.

## 검사와 제한

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest experiments/c_learning_only/test_debug.py -q -p no:cacheprovider
```

작은 합성 입력은 단위 검사 전용이다. CUDA 파이프라인 smoke는 8/256/8 모델로
합성 128노드, 2epoch를 실행하며 모든 저장 결과·checkpoint를 debug로 표시한다.
production CLI는 200epoch 미만을 거부한다.

현재 실행기는 fresh run만 지원한다. 중단된 본학습의 resume은 구현하지 않았다.
세부 관측과 개입 평가는 추가 시간·RAM을 사용하므로 calibration에 포함한다.
첫 배치의 CPU 스냅샷은 검사용이며 본학습의 일반 step에서는 수행하지 않는다.
기본 recipe를 바꾸는 실험은 이 실행기에 옵션을 덧붙이기 전에 다음 버전의 범위를 정한다.
