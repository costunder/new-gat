# 실험 B — raw 메시지 생성 규칙 학습

이 실험은 **새 엣지 metric이 입력에서 메시지 생성 규칙을 학습하고, 다른 그래프에서도 재사용하는지** 검사한다. 학생의 출력은 `Qθ(X)X`다. 분류기와 분류용 smoothing은 포함하지 않는다.

## FULL 계약

| 항목 | 실제 실행 범위 |
| --- | --- |
| 그래프 | 531개: train 240 / validation 60 / ID 120 / size OOD 90 / family OOD 12 / family+size OOD 9 |
| 입력 | 그래프마다 독립 scalar 입력 16개. feature 차원은 1이다. |
| 교사 | diagonal / diagonal_squared / analytic_pair, 각각 unit 및 local_degree |
| 학생 | D0 / D1 / F0 / F1 / F2 / DA, 각각 unit 및 local_degree |
| 학습 | seed 11, 23, 37, 53, 71 / 500 epoch / Adam lr 0.003 / gate hidden 64 |
| 실제 학습 예산 | 240개 seed 모델, 120,000회 seed optimizer 갱신 |
| 고정 대조 | D0/F0의 24개 고정 조건. optimizer와 사용하지 않는 parameter가 없다. |
| 수식 기준선 | 같은 교사의 Ld, Ld², I/Ld/Ld² polynomial 및 기존 고정 wedge Q. train 자료만으로 총 24회 적합한다. |

한 epoch마다 train 그래프 240개와 모든 scalar 입력을 사용한다. 물리 batch가 여러 개이면 그래프 수로 가중한 gradient를 합친 뒤 Adam을 한 번 갱신한다. seed 모델은 tensor의 별도 축에서 병렬로 학습한다. 모든 occurrence와 eligible pair를 유지하며 chunking은 계산만 나눈다.

여러 GPU가 할당되면 72개 독립 packed job을 실제 GPU worker에 분배한다. parent가 한 번 준비한 입력·교사·topology cache를 공유하고, 모든 worker 결과를 합쳐 하나의 전체 coverage와 보고서를 만든다. GPU별 측정 batch와 checkpoint는 `workers/gpu-XX/`에 저장한다.

## 문서와 산출물

- [수식과 해석 범위](MODEL_MATH.md)
- [실행 및 재개](RUN.md)
- [검증 상태](VERIFICATION.md)
- `completion.json`: 완료·예산·수식 검사 상태.
- `per_graph.csv`, `per_realization.csv`: 모든 그래프·seed·scalar 입력의 오차.
- `training.csv`, `checkpoints/`: loss, gradient, optimizer 변화와 validation 선택 상태.
- `graphs/`, `geometry/`, `teacher_messages.pt`: 실제 입력, 모든 연결 대응, 고정 target.
- `data_manifest.json`, `source_manifest.json`, `coverage.json`, `resources.json`: 데이터·소스 해시와 자원 측정.

FULL은 서버에서 실행한다. 로컬 DEBUG 결과는 전체 성능 결과로 쓰지 않는다. B는 A의 201개 그래프를 대신 사용하지 않고, 선언한 531개 합성 그래프를 생성한다.
