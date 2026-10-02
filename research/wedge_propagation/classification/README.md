# Experiment 4 — 실제 노드 분류

**분류 CE로 학습한 경로 가중치 C가 고정 Q보다 도움이 되는지 확인하는 서버 실험이다.**
수식과 비교 이유는 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md), 실제 실행 설정은
[config_full.json](config_full.json), 검증 기록은 [VERIFICATION.md](VERIFICATION.md)에 있다.
기존 합성 실험 checkpoint를 불러오지 않고 각 조건을 새로 학습한다.

## 전체 실행 범위

| 항목 | full 설정 |
| --- | --- |
| 데이터 | Cora, CiteSeer, PubMed 전체 그래프와 public fixed split |
| 모델 | 입력 → hidden 64 → 클래스, 두 층, dropout 0.5 |
| 조건 | MLP, first-order, polynomial L̄², fixed Q, learned raw/RMS, fixed Q+용량 대조 MLP, GCN |
| 경로 | 모든 unordered wedge. 총 778,561개, sampling 1.0 |
| 학습률 | 0.001, 0.003, 0.01 |
| 튜닝 | 조건·데이터·학습률마다 3개 seed, 총 216 run |
| 최종 학습 | 선택한 학습률마다 별도 5개 seed, 총 120 run |
| 예산 | 매 run 500 epoch, 총 336 run·168,000 독립 Adam update |
| 정밀도 | FP32, TF32 끔 |
| 선택 | validation CE로 checkpoint와 학습률 선택 |
| 평가 | 모든 최종 checkpoint 선택 후 train/validation/test 및 frozen 개입·배율 검사 |

각 학습 update는 전체 그래프 하나를 사용한다. Physical/effective graph batch는 1이고
gradient accumulation은 1이다. 동시에 계산하는 독립 seed 모델 수는 별도로 측정한다.
튜닝은 1/2/3, 최종 학습은 1/2/4/5개 모델 후보를 실제로 실행해 처리량·peak VRAM을 비교한다.
파라미터·Adam 상태는 seed별로 독립이며 CE 평균을 seed 축에서 합산해 각 gradient 크기를 유지한다.

## A6000 서버 실행

저장소와 Python 환경이 `/home/aicompetition07` 아래에 있는 서버 기준이다.
GPU 입력에는 **실제 할당받은 번호 또는 UUID**를 넣는다. 여러 개를 할당받았다면 `0,1`처럼 입력한다.
모든 visible GPU에 데이터·조건별 작업을 나누고 각 GPU 안에서는 측정한 seed pack을 사용한다.
필수 패키지 목록은 기존 CUDA PyTorch를 교체하지 않는다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
/home/aicompetition07/.conda/envs/new-gat/bin/python -m pip install \
  -r research/wedge_propagation/classification/requirements.txt &&
read -r -p "할당받은 A6000 GPU 번호 또는 UUID(여러 개는 쉼표): " CLASSIFICATION_GPU &&
test -n "$CLASSIFICATION_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$CLASSIFICATION_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.wedge_propagation.classification.study \
  --profile full --device cuda \
  --data-root /home/aicompetition07/new-gat/data/wedge-citation \
  --output-dir "results/wedge-classification-$(date +%Y%m%d-%H%M%S)"
```

실행한 터미널에서 `[data]`, `[calibration]`, `[job]`, **매 epoch의 CE·validation accuracy·시간**,
`[evaluation progress]`, `[complete]`를 확인할 수 있다. 같은 출력은 `terminal.log`에 저장된다.
별도 모니터 터미널은 필요하지 않다. CUDA가 없거나 할당 GPU가 지정되지 않으면 full 실행을 거부한다.

첫 실행은 고정된 공식 원본 24개를 다운로드하고 SHA256을 검사한다. 이후 verified raw/NPZ cache를 재사용한다.
다운로드가 막힌 서버에는 검증된 `data/wedge-citation`을 옮긴 뒤 같은 명령에 `--offline`을 추가한다.
기존 파일의 SHA가 달라지면 덮어쓰지 않고 오류를 낸다.

## 계산과 메모리

고정 Q는 정확한 엣지 항등식으로 계산한다. Learned C는 모든 경로를 exact chunking으로 처리한다.
Activation checkpointing을 사용하며 chunk 크기도 전체 forward/backward 처리량으로 측정한다.
OOM 후보는 기록하고 다른 chunk/독립 seed pack 후보를 시험한다. 모델·노드·경로·epoch는 유지한다.
자원 기록에는 CPU affinity·RAM·GPU·peak VRAM·CPU 사용률·측정 가능한 GPU 사용률이 들어간다.
GPU 사용률을 읽을 수 없으면 사유와 null을 기록한다.

Frozen 평가에서는 dataset마다 shuffle/random manifest 10개를 한 번 생성해 모든 모델이 공유한다.
C=1과 C 위치 섞기는 κ 재계산/기존 κ 유지 두 방식으로 비교한다.
두 번째 분기 제거와 무작위 physical edge pair 대응도 검사한다.
각 layer의 현재 표현에서 C를 다시 계산하며 모델 hash가 평가 전후 같은지 확인한다.
배율 0.25/0.5/1/2/4는 전처리 완료된 입력에 적용한다.

## 중단 후 재개

50 epoch마다 모델·Adam·seed별 최적 checkpoint·history를 hash와 함께 저장한다.
원래 폴더를 보존하고 **새 output 폴더**에서 이어 간다.
위 실행 명령에 다음 인자를 추가한다.

```bash
--resume-from /home/aicompetition07/new-gat/results/원래실험폴더
```

코드·설정·데이터 hash와 원래 seed pack/chunk가 같아야 한다.
완료한 run은 재학습하지 않으며 실행한 새 update 수를 별도로 기록한다.
미완성 snapshot은 보존하고 이전에 hash 검증된 snapshot부터 재개한다.
원래 packing이 현재 자원에서 불가능하면 오류로 중단하며 임의로 규모를 바꾸지 않는다.

## 결과 확인

- `CLASSIFICATION_SUMMARY.md`: 실제 계산한 요약·동일 seed의 paired 표.
- `classification_performance`, `frozen_interventions`, `scale_response`, `branches_and_resources`의 PNG/PDF: 성능·개입·배율·자원 그림.
- `metrics.csv`: 최종 5 seed의 train/validation/test CE·accuracy.
- `tuning_validation.csv`, `learning_rate_selection.json`: validation 전용 학습률 선택 근거.
- `final_validation_selection.csv`, `test_evaluation_unlocked.json`: 최종 checkpoint 선택과 test 잠금 해제.
- `interventions.csv`, `scale.csv`, `gate_diagnostics.csv`: 고정 모델 개입·배율·C/κ/분기 진단.
- `resources.csv`, `calibration/`, `jobs/`: 처리량·메모리 측정과 모든 epoch·checkpoint.
- `config.json`, `source.json`, `data_manifest.json`, `contract.json`, `coverage.json`: 설정·hash·전체 범위 확인.
- `completion.json`: 해당 profile 완료 여부. 실패하면 `failure.json`과 원래 산출물을 보존한다.

DEBUG는 별도 fixture·예산을 쓰며 결과에 `actual_data=False`를 표시한다.
DEBUG 통과를 실제 citation 학습 성능으로 해석하지 않는다.
