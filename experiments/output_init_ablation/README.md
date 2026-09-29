# 출력 투영 초기화 대조 실험

검토를 통과한 `c_learning_bracket`을 기준으로, 출력 투영의 초기 크기만 비교한다.
기준 모델 파일은 수정하지 않는다. 새 실험은 같은 모델의 forward를 상속하고
별도 runner에서 초기화 조건과 소스 hash를 기록한다.

## 네 조건

| 출력 초기화 | C 학습 | C=1 고정 |
| --- | --- | --- |
| 기존 Linear 기본 초기화 | baseline / learned | baseline / fixed |
| ReLU용 Kaiming uniform | kaiming_relu / learned | kaiming_relu / fixed |

각 조건은 8층·hidden 256·8 heads, dropout .2, AdamW lr .0005·decay .01,
gradient clip 5, **200epoch 전체 train pass**다. model seed는 기존의 0 하나다.
ogbn-arxiv 원본 169,343 nodes·1,157,799 물리 엣지, 공식 train 90,941,
validation 29,799, test 48,603 구분을 유지한다. 공식 test는 이 실험에 사용하지 않는다.
학습은 physical/effective seed batch 2048, 문맥 seed 2048, worker 4를 사용한다.
FP32·TF32 off·pinning on·prefetch off, accumulation=1, GPU worker=1이다.

## 정확히 무엇을 바꾸는가

출력 투영은 bias 없는 D×D Linear다. D=256일 때 기존 초기 분포는
W ~ Uniform[-1/sqrt(D), 1/sqrt(D)]이며 분산은 1/(3D)다.
ReLU용 Kaiming uniform은 W ~ Uniform[-sqrt(6/D), sqrt(6/D)], 분산 2/D다.

새 난수를 다시 뽑지 않고 **기존 W를 sqrt(6)배**한다. 그러면 원소의 부호와
상대 방향을 유지하며 목표 초기 분포에 맞출 수 있다. 이는 초기 파라미터의 설정이다.
forward에 gain이나 정규화 층을 넣는 것이 아니며 학습 중 재스케일하지 않는다.

같은 seed의 encoder·decoder·Q/K·value·beta 초기값과 난수 상태는 동일하다.
초기 출력 투영 weight 8개만 다르다. 각 초기화 안에서 learned/fixed 공통 파라미터도
일치한다. 최종 비교에서 두 초기화의 출력 외 초기 파라미터 hash와 네 조건의
모든 epoch sample sequence·전체 seed coverage를 확인한다.

## 비교할 것

1. 첫 층부터 마지막 층까지 H → value → 혼합 → 출력 → ReLU → dropout의 RMS.
2. 깊은 층 C의 분포, 같은 입력에서 optimizer 전후 C·alpha 변화와 실제 CE 미분.
3. validation으로 선택한 checkpoint의 전체 그래프·새 문맥 성능과 C=1 개입 영향.
4. 각 초기화에서 learned-minus-fixed 성능 차이가 얼마나 달라졌는지.

초기 RMS가 커지거나 C가 1과 달라지는 것만으로 성공을 판정하지 않는다.
수치 변화와 분류 유용성을 함께 본다. 이 비교는 seed 0 한 번의 결과이며
통계적 유의성·독립 그래프 일반화·공식 test 성능을 주장하지 않는다.

## 실행 — 서버 전용 본학습

본학습은 서버의 할당된 GPU 세션에서 실행한다. 로컬 PC에서는 본학습을 실행하지 않는다.
서버에서 아래 저장소 경로와 Python 환경이 존재하고 실제 GPU 할당이 활성화되어 있어야 한다.
현재 서버 접속·GPU 상태는 이 전달본에서 검증하지 않았다. 기존 CUDA_VISIBLE_DEVICES와
스케줄러 할당을 유지하며 임의로 GPU 번호를 고르지 않는다. runner는 보이는 CUDA 장치가
정확히 하나인지 확인한다. 여러 GPU가 할당되었다면 실행 배치를 먼저 검토한다.

서버 저장소의 main 브랜치에서 Git으로 코드를 받은 뒤 실행한다.
`git pull --ff-only`가 실패하면 아래 학습 명령으로 넘어가지 않는다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only origin main &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m experiments.output_init_ablation.study \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "results/output-init-server-$(date +%Y%m%d-%H%M%S)"
```

서버에서 calibration부터 새로 측정한다. 로컬 calibration과 소요시간은 서버 학습의
근거로 재사용하지 않는다. 이 명령은 두 calibration을 통과하면 본학습으로 이어진다.

Kaiming과 기준 초기화 각각 physical seed 2048/4096 × worker 2/4 × learned/fixed,
총 16개 조합을 calibration한다. 각 조합은 warmup 뒤 실제 배치 3개에서
일반/관측 step을 짝지어 측정한다. 두 초기화의 calibration을 완료한 뒤
기준 learned/fixed, Kaiming learned/fixed 순서로 각 200epoch를 수행한다.
단일 GPU이므로 무거운 학습 작업을 겹쳐 실행하지 않는다. 각 모델 안의 문맥은
disjoint union으로 병렬 계산하고 문맥 준비에 worker 4개를 사용한다.

학습 시작 시 현재 소스·설정과 calibration 계약이 일치해야 한다.
예외가 발생하면 로그와 failure JSON을 남기고 다음 실험으로 넘어가지 않는다.
수식·정밀도·배치 크기를 조용히 바꾸지 않으며, 다른 GPU 작업을 종료하지 않는다.
기존 runner처럼 중단된 epoch부터의 resume은 제공하지 않는다.
checkpoint와 완료된 결과는 보존되며 중단 시 새 실행 계획을 별도로 정한다.

## 결과 위치

- `study_contract.json`, `execution_sources.zip`: 시작 시 소스와 전체 설정.
- `status.json`, `*-started.json`, `*-completed.json`: 단계·PID·명령·완료 상태.
- `calibrate-*/`: 각 초기화의 실제 calibration·단계별 관측·문맥 평가.
- `train-*/`: 초기·매 epoch 관측/평가, best checkpoint, 최종 평가.
- `comparison.json`: 네 조건이 모두 끝나고 초기값·샘플 대응 검사도 통과한 최종 비교.
- `signal_trajectory.json`: 4조건 × 200epoch의 첫 배치, 모든 층의 신호·C 변화 요약.
- `study_failure.json` 또는 하위 `failure.json`: 실패 원인. 정상 완료로 표시하지 않는다.

## 검증

합성 debug 검사는 기준군과 검토 완료 모델의 출력·미분 일치, 출력 weight만의
변경, 난수 상태 일치, 실제 4조건 CUDA 학습·checkpoint·평가·최종 대조를 검사한다.
작은 합성 그래프와 2epoch는 별도 테스트에만 사용하며 모델은 8/256/8을 유지한다.
이 테스트 통과는 실제 데이터 200epoch 실험 완료를 뜻하지 않는다.
