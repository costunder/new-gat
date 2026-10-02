# Experiment 3.1: C 생성기 입력의 스케일 정규화

같은 그래프·입력·teacher를 사용해 **raw C 생성기와 정규화 C 생성기**를 비교한다.
변경은 C를 만드는 차분 입력뿐이다. 메시지를 만드는 실제 `AX`는 유지한다.
양의 입력 배율에서 C가 유지되도록 설계한 뒤, 목표 메시지 정확도와 새 특징 일반화도
실제로 좋아지는지 측정한다.

## 두 모델에서 바뀌는 것

| 구분 | raw | normalized |
| --- | --- | --- |
| Gate 입력 | 원래 차분 g₁, g₂ | g₁/σ, g₂/σ |
| σ | 사용하지 않음 | 그래프·scalar 실현별 전체 물리 엣지 차분 RMS |
| 실제 메시지 | βAᵀC(X)AX | βAᵀC(g₁/σ,g₂/σ)AX |
| 파라미터 | Experiment 2 checkpoint 고정 | 원본 학습 입력에서 새로 학습 |
| Teacher·목표 | 원본 규칙 | 같은 원본 규칙 |
| 배율을 사용한 학습 | 없음 | 없음 |

σ가 양수면 epsilon을 더하지 않고 나눈다. 엣지가 없거나 모든 엣지 차분이 0이면 σ=1을
사용한다. 정규화 범위는 각 입력 그래프 전체이며, 서로 다른 그래프나 scalar 실현을 섞지 않는다.
`random_pair`도 같은 물리 엣지 σ를 사용한다. 수식은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.

## full 계약

| 항목 | 범위 |
| --- | --- |
| 그래프 | 원본 531개, 모든 노드·엣지·경로·random pair 유지 |
| Target·조건 | L/L²/path × first/polynomial/fixed/learned/random_pair |
| 모델 | Gate hidden 64, 원래 구성 유지 |
| 학습 seed | 11, 23, 37, 53, 71 |
| 새 학습 | normalized의 3 target × 2 gate 조건만 500 epoch |
| 학습 입력 | 원본 train 240개 그래프의 scalar 실현 16개 |
| 학습 physical batch | 원본에서 측정한 240개 그래프 유지 |
| 모델 선택 | 원본 validation만 사용해 seed별 checkpoint 선택 |
| Raw·고정 대조군 | 원본 파라미터 고정, 추가 학습·scalar 재적합 없음 |
| 새 특징 | Experiment 3의 전체 입력·seed·hash를 재사용 |
| 평가 배율 | original 1 + 같은 fresh X의 0.25/0.5/1/2/4 |
| Split | train/validation/ID/size OOD/family OOD/family+size OOD |
| 메시지·스케일 측정 | 248,508행 / 207,090행 |
| 고정 개입 | original과 fresh 1, 두 variant의 learned 모델, 전체 3 target·6 split: 159,300행 |

`first`, `polynomial`, `fixed`는 원본의 같은 대조군을 두 variant에 표시한다.
두 번 학습하거나 독립적인 두 결과로 세지 않는다. 새 모델도 직접 C*를 맞추는 보조 loss는
사용하지 않으며, 원래 메시지 loss로 학습한다. Teacher와 목표는 매 배율에서 다시 계산한다.

## A6000 서버 실행

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
WEDGE_PYTHON=/home/aicompetition07/.conda/envs/new-gat/bin/python &&
read -r -p "할당받은 A6000 GPU 번호 또는 UUID: " WEDGE_GPU &&
test -n "$WEDGE_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$WEDGE_GPU" \
  "$WEDGE_PYTHON" -u -m research.wedge_propagation.scale_normalization.study \
  --profile full --device cuda \
  --source-dir results/wedge-learned-20261002-172132 \
  --feature-source-dir results/wedge-feature-20261003-042123 \
  --output-dir "results/wedge-scale-$(date +%Y%m%d-%H%M%S)"
```

원본 학습 결과와 Experiment 3 입력 폴더를 읽기만 한다. 새 출력 폴더에 normalized 학습 결과와
두 모델의 평가를 저장한다. 원본 재현·입력 hash·모델 고정·파일 불변성 검사를 통과해야 한다.
기존 Experiment 2·3 환경을 재사용하며 추가 패키지는 없다.
학습과 평가 physical batch는 각각 Experiment 2·3의 기록을 유지한다.
대안 batch와 CPU worker 처리량도 측정하고, 대응 비교를 위해 원래 batch를 유지한 이유를 기록한다.
학습 진행과 평가 결과는 실행 터미널과 로그에 표시한다.

## 결과 읽기

- `SCALE_NORMALIZATION_SUMMARY.md`: 동일 입력의 normalized−raw 차이, fresh−original 차이,
  스케일 진단, C 회수와 개입을 구분한 한국어 요약.
- `metrics.csv`: 모든 variant·target·조건·seed·graph·시나리오·배율의 실제 메시지 오차.
- `scale_checks.csv`: 같은 fresh X의 기준 배율 1에 대한 C 변화와 메시지 비례성 오차.
- `interventions.csv`: original과 fresh 1에서 모델을 고정한 다섯 개입. 원래 출력은 metrics에 있다.
- `training.csv`: normalized 모델의 실제 학습·validation 기록.
- `contract.json`: 학습 구성·입력 hash·배치·자원·불변성·원본 재현 계약.
- PNG/PDF 4종: 메시지 오차, 배율별 오차, 비례성 진단, C·개입 진단.

스케일 비례성은 구조의 성질이고 정확도는 경험적 결과다. C가 입력 배율에 안정적이어도
잘못된 메시지를 만들 수 있다. Cmean=1과 높은 상관만으로 C를 유일하게 회수했다고 판단하지 않는다.
개입의 개선·악화·동률과 undefined 값을 그대로 기록한다. 실제 GNN 분류 효과는 별도 단계다.

## 개발 검증

DEBUG는 원본 DEBUG 입력과 별도 설정을 사용한다. 원본 DEBUG의 학습 기간·그래프·scalar 실현·seed를
상속하고, 그림과 요약에 DEBUG임을 표시한다. DEBUG 결과를 full 성능으로 보고하지 않는다.
실제 완료된 검증과 미실행 범위는 [VERIFICATION.md](VERIFICATION.md)에 기록한다.

중단된 normalized job은 같은 revision의 `checkpoints/*-epoch*.pt`를
`--resume-from`에 넣고 새 출력 폴더에서 재개할 수 있다. 해당 job의 optimizer와 epoch를 복원하며,
다른 다섯 job은 새 폴더에서 처음부터 실행한다. raw checkpoint는 재개 대상으로 받지 않는다.
