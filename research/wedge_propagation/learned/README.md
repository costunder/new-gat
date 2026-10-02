# Experiment 2 — 입력에서 경로 가중치를 배우는 실험

고정 연산 검증 다음 단계다. **세 가지 목표 메시지와 다섯 비교군**을 같은 데이터에서 평가한다.
공유 MLP가 두 엣지의 특징 변화 관계를 보고 C2를 생성하며, 학습 loss에는 목표 노드 메시지만 제공한다.
Teacher의 C2는 평가 진단에만 사용한다. GNN backbone과 분류기는 이번 단계에 포함하지 않는다.

## 실제 비교군

| 조건 | 출력 | 학습 파라미터 수, 모델 하나 기준 |
| --- | --- | ---: |
| first | u LX | 1 |
| polynomial | u LX + v L²X | 2 |
| fixed | β QX, Q=A.T A | 1 |
| learned | β A.T C2(X) AX | 386 |
| random_pair | β Ar.T C2(X) ArX | 386 |

세 target은 LX, L²X, A.T C2*(X) AX다. LX와 L²X는 해당 연산만으로 충분한 대조다.
Fixed·learned·random-pair 출력에는 추가 일차 항을 넣지 않는다.

Scalar baseline은 학습 그래프의 메시지 loss를 최소화하는 계수를 직접 구한다.
검증·평가 label은 fitting에 사용하지 않는다. 학습 초기화 실패에 좌우되지 않는 강한 대조군이다.
결정론적 fitting 한 번을 seed 여러 개의 독립 실험으로 세지 않는다.
Learned와 random-pair는 각각 초기화 seed 5개로 학습한다.

## 수식과 설정

Teacher와 student는 모두 `exp(tau*tanh(score))`를 사용하고 그래프별 경로 평균을 1로 맞춘다.
각 특징 열은 독립 scalar 입력이므로 **그래프와 특징 실현마다** 정규화한다.
전체 생성식·loss·개입의 정확한 정의는 [MODEL_MATH.md](MODEL_MATH.md)에 있다.

| 설정 | full 값과 선택 근거 |
| --- | --- |
| 특징 | Scalar, 그래프당 독립 입력 16개. 고정 단계와 같은 입력 분포를 사용한다. |
| Teacher | theta1=theta2=tau=1, epsilon=1e-8. 두 관계를 같은 계수로 사용하고 exp 범위를 제한한다. |
| Gate | 4→64→1, ReLU, 두 bias, Xavier 초기화. 4개 대칭 통계에서 비선형 비율 규칙을 근사하는 단일 hidden MLP다. |
| Epoch | 500, early stop 없음. 이 작은 생성기의 학습 경향을 확인할 고정 예산이다. 수렴을 보장하는 값은 아니다. |
| Seed | 11/23/37/53/71. 초기화 변동을 5회 측정하며 같은 데이터에서 조건별로 대응시킨다. |
| Optimizer | Adam, lr=0.003, weight decay 없음. 매개변수 수가 작은 생성기의 독립 시작 설정이다. |
| Precision | CPU teacher/참조 float64, 학습 float32, TF32 사용 안 함. |
| Validation | 초기 상태(epoch 0), 첫 epoch와 매 10 epoch, 마지막 epoch. Seed별 validation 메시지 오차로 상태를 선택한다. 선택 epoch가 0이면 초기 모델이 선택됐다는 뜻이다. |

폭·학습률·epoch는 이전 Conductance 실험에서 계승하지 않았다.
새 full 계약은 [config_full.json](config_full.json)에 저장한다. 전부 명시된 값이며 조용한 축소·fallback은 없다.
Debug는 [config_debug.json](config_debug.json)의 별도 4 epoch/hidden 16/2 seed 계약이다.

## 데이터와 분할

| Split | Family | 노드 수 | 그래프 | 입력 |
| --- | --- | --- | ---: | ---: |
| Train | ER/tree/tree+chord | 20/30/40/50 | 240 | 3,840 |
| Validation | 동일 family, 새 그래프 | 20/30/40/50 | 60 | 960 |
| ID | 동일 family, 새 그래프 | 20/30/40/50 | 120 | 1,920 |
| Size OOD | 동일 family | 60/80/100 | 90 | 1,440 |
| Family OOD | cycle/star/grid | 20/30/40/50 | 12 | 192 |
| Family + size OOD | cycle/star/grid | 60/80/100 | 9 | 144 |
| 합계 | | | **531** | **8,496** |

Train은 size/family마다 20개 graph draw, validation 5개, ID와 size OOD는 10개다.
학습·평가 그래프와 특징 seed를 분리하고 정확히 같은 labeled topology의 재사용을 검사한다.
동형인 다른 그래프를 모두 배제하는 split은 아니다. 해당 범위를 manifest에 기록한다.
중복이 발견되면 기록된 retry seed로 다시 생성한다. 256회 안에 유일한 그래프를 생성할 수 없으면
전체 준비를 실패 처리하며 그래프를 버리지 않는다. 원래 난수 분포에 중복 배제 조건이 추가됨을 기록한다.
Tree+chord는 해당 split·크기·draw의 tree에 floor(n/4)개의 chord를 추가한다.
모든 노드, 엣지, unordered wedge와 16개 입력을 유지한다.

## A6000 서버 실행

기존 CUDA PyTorch 환경을 사용한다. 추가 requirements는 PyTorch를 교체하지 않는다.

```bash
cd ~/new-gat &&
git pull --ff-only &&
WEDGE_PYTHON=/home/aicompetition07/.conda/envs/new-gat/bin/python &&
"$WEDGE_PYTHON" -m pip install -r research/wedge_propagation/learned/requirements.txt &&
read -r -p "할당받은 A6000 GPU 번호 또는 UUID: " WEDGE_GPU &&
test -n "$WEDGE_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$WEDGE_GPU" \
  "$WEDGE_PYTHON" -u -m research.wedge_propagation.learned.train \
  --profile full --device cuda \
  --output-dir "results/wedge-learned-$(date +%Y%m%d-%H%M%S)"
```

CPU worker 후보를 전체 데이터 생성으로 측정하고, 실제 forward/backward/optimizer로 graph batch 후보를 측정한다.
정적 데이터는 한 번 준비해 cache하며 epoch마다 읽거나 graph를 다시 만들지 않는다.
다섯 seed의 파라미터는 독립이고 같은 seed 축에서 병렬 계산한다.
Seed 수를 effective batch의 DDP worker 수로 세지 않는다.
선택한 batch는 두 학습 조건·세 target에서 공통으로 사용한다.
크기마다 GPU 연산을 하나씩 실행하는 방식 대신 disjoint-union batch를 사용한다.

실행한 터미널에 **epoch·loss·validation 오차·처리시간·checkpoint 경로**가 출력되고 `terminal.log`에 저장된다.
최종 결과가 좋다고 미리 판정하지 않는다. 실제 비교가 동률이거나 나빠도 보고서에 그대로 나온다.

## 결과

- `LEARNED_SUMMARY.md`: target·split·condition별 graph macro와 seed 통계, 개입 결과.
- `metrics.csv`: 모든 평가 그래프의 메시지 오차, 가능한 경우 C2 오차·상관.
- `interventions.csv`: 선택한 checkpoint를 고정한 다섯 개입.
- `training.csv`: seed별 학습 loss, validation 오차와 best epoch.
- `teacher_operator_audit.csv`: teacher 연산이 span{L,L²,Q}로 설명되는 잔차와 입력에 따른 변화.
- `dataset.npz`, `data_manifest.json`: 실제 생성 그래프·입력·목표·teacher C2·random pair·seed·hash.
- `intervention_manifest.json`: shuffle과 donor 대응 규칙.
- `contract.json`: 실제 자원·cache·physical batch·전체 update 수·코드 hash·검증 범위.
- `checkpoints/`: immutable epoch checkpoint와 seed별 validation 선택을 합친 selected checkpoint.
- PNG/PDF 4종: 학습곡선, split 메시지 오차, 개입 오차, 가중치 진단.

메시지 오차는 graph마다 16개 scalar 실현을 평균한 뒤 graph macro를 계산한다.
ID/size/family/family+size 결과를 합치지 않는다.
C2 평균을 1로 맞춰도 출력만으로 각 path weight가 유일하게 결정되지는 않는다.
따라서 메시지 회수가 주지표이며 C2 회수는 진단이다.

## 재개와 결과 보호

결과 폴더는 새 경로만 허용한다. 기존 checkpoint·표·그림을 덮어쓰지 않는다.
`--resume-from checkpoints/...-epoch....pt`를 추가하면 **그 checkpoint의 한 target/condition job**을 이어간다.
새 결과 폴더에서 나머지 job은 다시 실행해 전체 비교 결과를 만든다.
저장된 config와 코드 hash가 일치해야 하며 저장한 physical batch를 유지한다.
Optimizer state·seed별 best 상태·완료 epoch·학습 기록·RNG 상태를 복원한다.
이는 이전 결과 폴더를 수정하며 전체 job을 건너뛰는 재개가 아니다.

## 개발 검사

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_wedge_learned_model.py tests/test_wedge_learned_data.py \
  tests/test_wedge_learned_report.py tests/test_wedge_learned_pipeline.py \
  tests/test_wedge_learned_diagnostics.py
python -u -m research.wedge_propagation.learned.train --profile debug --device cuda \
  --output-dir "results/wedge-learned-debug-$(date +%Y%m%d-%H%M%S)"
```

Debug는 **36개 그래프·144개 입력**이다. 본학습과 최종 성능 결과로 사용하지 않는다.
현재 검증 상태는 [VERIFICATION.md](VERIFICATION.md)에 따로 기록한다.
