# 실험 B 검증 상태

## 실행 시 자동 검사

`study`는 본학습 전에 다음 검사를 수행하고 `math_checks.json`에 기록한다.

- path / triangle / star / irregular / isolate, 두 recipe: 총 10개 조건.
- 각 조건에서 세 raw teacher를 독립 dense 행렬식과 비교: 총 30개 수식 검사.
- F2가 실제 raw 메시지 MSE에 연결되는지 확인하는 별도 DEBUG fixture의 backward 및 Adam 갱신.

이 fixture의 두 seed 갱신과 calibration의 임시 갱신은 본학습 예산에 포함하지 않는다. gate parameter를 학습한 것처럼 만든 가짜 checkpoint는 사용하지 않는다.

## 전용 단위 테스트

[tests/test_edge_metric_synthetic.py](../../../tests/test_edge_metric_synthetic.py)는 다음을 검사한다.

- FULL 531개 graph membership, 16개 scalar 입력, 240개 학습 run과 120,000회 갱신 계약.
- DEBUG 전체 조건 축과 임의 축소 거부.
- 기존 graph seed stream 유지와 새로운 feature stream 분리, 데이터 해시 재현.
- 세 teacher의 dense 식, disjoint graph batching과 scalar realization 독립성.
- 여섯 조건의 실제 parameter·forward 연결, seed 병렬 모델의 독립 실행과 동일성.
- 전체 graph batch와 물리 batch로 나눈 exact gradient accumulation의 loss·gradient·Adam 상태 동일성.
- SAME-teacher raw Ld/Ld² oracle의 held-out 정확성.
- zero target 처리, 완료한 job의 재개 시 새 optimizer 갱신 0회, 데이터 해시 불일치 거부.
- CUDA가 있으면 CUDA dense 검사와 실제 scalar loss backward.
- 여러 GPU의 72개 packed job 분배가 중복·누락 없이 유지되는지와 학습 job 균형.
- 공유 tensor cache의 전체 입력·topology 해시 재현 및 실제 단일 CUDA worker의 DEBUG 학습·평가.

CPU와 CUDA의 독립 30개 teacher 수식 검사를 실제 수행했다. 최대 절대 오차는 각각 약 5.33e-15와 1.78e-15였다. DEBUG MSE의 최대 gradient 약 0.2880, Adam parameter 변화 약 0.0030을 관측했다.

전용 테스트 **49개가 통과**했다. 여기에는 FULL 서버·명시적 GPU 할당 보호, CUDA dense 검사와 30개 DEBUG 그래프 전부를 사용하는 실제 single GPU worker의 한 조건 학습·평가가 포함된다. 여러 물리 GPU의 동시 실행은 로컬에 GPU 한 개만 있어 아직 측정하지 않았다.

## 전체 DEBUG 파이프라인

`results/edge-metric-pipeline-DEBUG-20261005-02/B/`에서 모든 선언 조건을 연결한 CPU DEBUG 실행이 완료됐다. 결과는 다음과 같다.

- 30개 그래프, scalar 입력 120개, 학습 run 96회와 고정 조건 24회, train-only 적합 24회.
- 새 seed optimizer 갱신 288회, packed optimizer 호출 144회, 총 13.03초.
- `per_graph.csv` 4,320행, `per_realization.csv` 17,280행, `training.csv` 288행, paired 비교 360행. 평가와 학습의 key 중복이 없다.
- 선택 checkpoint 48개, 입력 graph NPZ 30개, geometry NPZ 60개.
- 저장한 source digest `c3445bcea0a0e8cbaabb4fa7d44213cddfd724dd04d807a0e1f9e32e53876be5`와 QA 시점의 과학 코드 해시가 일치했다.
- 30개 dense teacher 검사가 통과했고 최대 절대 오차는 5.33e-15였다.
- gradient와 optimizer 변화가 양수인 학습 행은 240개다. 나머지 48개는 교사와 학생 recipe가 같은 diagonal target이며, 초기 출력이 이미 정확해 train NMSE와 gradient가 0이었다. 사용하지 않는 parameter가 있다는 판정으로 해석하지 않는다.
- SAME-teacher linear/diagonal 및 squared/diagonal_squared oracle의 평가 행 120개에서 최대 NMSE는 약 3.51e-32였다. 이는 DEBUG 양성 대조의 수식 일치를 확인한다.

PNG를 직접 확인했으며 축·제목·legend가 겹치지 않고 읽을 수 있다. 로그 그림에서 0 또는 1e-16보다 작은 값은 1e-16에 표시한다. 실제 오차는 CSV에 저장되어 있다. 그림을 단독으로 전달할 때에는 **DEBUG, seed 2개, 3 epoch, 로그 표시 하한 1e-16**을 함께 적는다.

이 실행은 구현·coverage·보고 경로의 DEBUG 검증이다. **서버 FULL 본학습과 전체 성능 평가는 아직 실행하지 않았다.**
