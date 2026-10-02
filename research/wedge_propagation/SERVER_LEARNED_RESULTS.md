# 서버에서 확인한 경로 규칙 학습 결과

2026년 10월 2일 서버에서 **Experiment 2 full 학습과 평가가 완료됐다.**
사용자가 제공한 첨부 `0412f129-7f5b-4c0f-9212-29dfb745cfe4`의 터미널 로그를 직접 읽고
세 target × 다섯 condition × 다섯 평가 split의 `[evaluate]` 75줄과 최종 `[complete]`를 확인했다.

결과 폴더는 `/home/aicompetition07/new-gat/results/wedge-learned-20261002-172132`다.
이 문서는 그 로그에 나온 출력 오차를 정리한다. 서버의 원본 `metrics.csv`, `interventions.csv`,
`teacher_operator_audit.csv`, NPZ와 checkpoint를 로컬로 가져와 재검증한 결과는 아니다.
일부 붙여넣은 hardware/data JSON의 문자가 잘려 있으므로, 손상된 free VRAM과 일부 경로 수 범위는 추정하지 않았다.

## 실제 실행 범위

| 항목 | 로그에서 확인한 값 |
| --- | --- |
| GPU | NVIDIA RTX A6000 1개, 할당 GPU 0 |
| 환경 | Python 3.11.16, PyTorch 2.7.1+cu118, CUDA 11.8 |
| CPU | affinity 96개, 전처리 worker 후보 1/2/4/8/16 실측 후 1개 선택 |
| 입력 | 531개 합성 그래프, 그래프당 독립 scalar 특징 16개, 총 8,496개 |
| Split | train 240 / validation 60 / ID 120 / size OOD 90 / family OOD 12 / family+size OOD 9 |
| 노드 크기 | train·validation·ID 20/30/40/50, size OOD 60/80/100 |
| 학습 구조 | `4 → 64 → 1` gate, learned/random-pair 각각 seed당 386개 파라미터 |
| 학습 | gate job마다 500 epoch, seed 11/23/37/53/71 병렬 처리 |
| Physical batch | 학습 그래프 240개, accumulation 1, 그래프당 16개 scalar 실현 병렬 처리 |
| 정밀도 | 학습·평가 float32, 저장된 teacher·target 참조는 float64 |
| 비교군 | first / polynomial / fixed / learned / random-pair |
| Target | `LX`, `L²X`, `Aᵀ C₂*(X) A X` |

독립 seed 축은 서로 다른 모델을 함께 계산하는 축이다. 한 모델의 effective batch를 seed 수 5로 곱하지 않는다.
First/polynomial/fixed는 학습 데이터만 이용한 결정적 scalar 최소제곱 해이며, seed 반복이 없다.
CPU 전처리 1개가 선택된 이유는 전체 531개 처리 시간이 1.936초로 후보 중 가장 짧았기 때문이다.
240개 batch의 calibration 처리량은 learned 16,283.6 graphs/s, random-pair 16,241.5 graphs/s였다.
그때 보고된 peak는 각각 2,053,353,472 / 2,053,361,664 bytes다. 이것은 calibration 측정값이다.

## 출력 오차

지표는 각 그래프에서 16개 실현의 메시지 상대 오차를 평균한 뒤, split의 그래프를 동일 비중으로 평균한 값이다.
아래 learned/random-pair 값은 로그의 다섯 seed 값에서 계산한 **평균 ± 표본 표준편차**이며 소수 여섯 자리로 표시한다.
Scalar 비교군은 한 번의 결정적 fit 값이다. 오차 0.05를 분류 정확도 95%로 해석하지 않는다.

### 경로 target: `Y=Aᵀ C₂*(X) A X`

| Split | First | Polynomial | Fixed | Learned | Random-pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.499753 | 0.247994 | 0.168908 | 0.046531 ± 0.013085 | 0.549527 ± 0.000347 |
| ID | 0.501094 | 0.250175 | 0.171868 | 0.047140 ± 0.013491 | 0.555487 ± 0.000533 |
| size OOD | 0.507761 | 0.246178 | 0.166230 | 0.046147 ± 0.012647 | 0.554482 ± 0.000441 |
| family OOD | 0.706794 | 0.317892 | 0.193912 | 0.049000 ± 0.013141 | 0.616995 ± 0.005943 |
| family+size OOD | 0.721565 | 0.288272 | 0.173095 | 0.042535 ± 0.011215 | 0.582529 ± 0.003682 |

ID에서 learned의 다섯 seed 원값은 seed 순서 11/23/37/53/71로
`0.04671841724726414, 0.04378812087583202, 0.032428742737840896, 0.06921504282356282, 0.043550622531047425`다.
평균 오차는 fixed보다 약 72.57% 작다. 크기와 family를 바꾼 평가에서도 낮은 출력 오차가 유지됐다.
이는 정해진 synthetic teacher가 만든 **경로 출력 규칙**을 학습한 결과다.

### 양성 대조 target: `Y=LX`

| Split | First | Polynomial | Fixed | Learned | Random-pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.000000 | 0.000000 | 0.476891 | 0.962120 ± 0.218819 | 0.874664 ± 0.140299 |
| ID | 0.000000 | 0.000000 | 0.472840 | 0.932349 ± 0.209324 | 0.867237 ± 0.137230 |
| size OOD | 0.000000 | 0.000000 | 0.483981 | 0.979741 ± 0.224664 | 0.882304 ± 0.144231 |
| family OOD | 0.000000 | 0.000000 | 1.881546 | 4.959894 ± 0.949028 | 2.811330 ± 0.429773 |
| family+size OOD | 0.000000 | 0.000000 | 4.594379 | 10.928360 ± 2.060463 | 5.939069 ± 0.892878 |

### 양성 대조 target: `Y=L²X`

| Split | First | Polynomial | Fixed | Learned | Random-pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.370400 | 0.000000 | 0.206254 | 0.209423 ± 0.003263 | 0.527221 ± 0.000989 |
| ID | 0.368480 | 0.000000 | 0.208187 | 0.211503 ± 0.003240 | 0.531642 ± 0.001080 |
| size OOD | 0.378128 | 0.000000 | 0.203145 | 0.206987 ± 0.002925 | 0.525265 ± 0.000733 |
| family OOD | 0.550931 | 0.000000 | 0.367143 | 0.379748 ± 0.008982 | 0.787745 ± 0.004140 |
| family+size OOD | 0.553731 | 0.000000 | 0.268101 | 0.278057 ± 0.006655 | 0.555757 ± 0.003616 |

L target의 first/polynomial과 L² target의 polynomial이 모든 split에서 오차 0으로 양성 대조를 통과했다.
Learned-wedge가 모든 target에서 우세한 것은 아니다. 특히 L target은 경로 연산만으로 잘 맞추지 못했다.
경로 teacher target의 우위를 모든 그래프 task의 우위로 확대하지 않는다.
Random-pair는 gate 파라미터 수를 맞췄지만 연결 구조·엣지 사용 빈도·support도 달라진다.
따라서 이 표만으로 경로 연속성 하나의 인과 효과를 확정할 수 없다.

## 이 로그에서 확인되지 않은 항목

- 경로 C₂의 실제 분포·teacher 가중치와의 오차·상관과 seed별 beta.
- 다섯 고정 checkpoint 개입의 출력 변화와 geometry 차이.
- Teacher 연산의 `span{L,L²,Q}` 잔차와 입력에 따른 변화.
- 원본 checkpoint/config/source/data hash의 상호 일치 및 CSV의 전체 graph ID coverage.

완료 로그는 이 단계의 full 실행이 끝났음을 보여 준다. 위 항목은 서버의 원본 파일을 읽어 확인해야 한다.
출력 회수가 좋아도 개별 C₂가 유일하게 회수됐다는 결론은 성립하지 않는다.
Experiment 3가 원본 입력과 선택된 모델을 읽어 재현 검사를 수행하며,
서버의 개입 CSV와 teacher audit CSV를 원본 진단으로 함께 보고한다.

## 다음 단계: 고정 모델의 새 특징 평가

[Experiment 3 패키지](generalization/README.md)는 이 완료 폴더를 입력으로 사용한다.
새 학습과 checkpoint 재선택 없이 9개 scalar fit과 6개 선택 checkpoint의 5-seed 모델을 그대로 읽는다.
원본 NPZ의 531개 그래프와 모든 wedge/random-pair를 유지한다.

1. 원본 X에서 Experiment 2 평가를 재현한다.
2. 같은 저장된 그래프에서 독립적인 새 Gaussian 특징 X₀를 생성한다.
3. 같은 X₀에 amplitude `0.25 / 0.5 / 1 / 2 / 4`를 적용하고 split별로 평가한다.
4. 각 amplitude의 teacher C₂와 세 target을 다시 계산해 epsilon의 영향을 유지한다.

새 graph·size·family OOD는 이미 Experiment 2에 들어 있었다.
Experiment 3에서 추가로 확인하는 것은 **같은 저장된 topology의 새 특징과 특징 크기 변화**다.
Experiment 3의 로컬 debug 검증과 서버 full 평가 여부는 [검증 문서](generalization/VERIFICATION.md)에 구분해 기록한다.
Cora/CiteSeer/PubMed 분류는 아직 별도 Experiment 4다.
