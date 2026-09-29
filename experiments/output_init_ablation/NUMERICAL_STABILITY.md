# log-row 계산: 정의와 수치 표현의 구분

## 확인된 실패와 수정 범위

서버 raw-exp 실행은 baseline learned 조건의 1epoch를 240.2초에 완료했다.
train CE 3.14324, validation accuracy 7.6278%였으며, 2epoch 중 conductance의
유한성·양수 검사에서 CUDA assertion이 발생했다. 그 검사에는 score 유한성도 포함된다.
원래 로그만으로 score 자체가 비유한지 exp의 표현 범위를 넘었는지 구분되지 않는다.

새 계산은 **유한한 score에서 exp 절댓값의 overflow/underflow로 정규화가 실패하는 경우**를
해결한다. score 자체나 출력 특징이 NaN/Inf이면 오류로 중단한다. 이 경우 모델·optimizer의
수치 문제까지 해결됐다고 주장하지 않는다. 오류 검사는 CUDA device assertion 대신
Python 예외를 발생시켜 해당 실패 JSON을 기록할 수 있게 한다.

## 유지하는 모델과 수식

각 물리 엣지 e=(u,v), head h에서 s_eh=(Q_u·K_v+Q_v·K_u)/(2r), C_eh=exp(s_eh)다.
Q/K, value, beta, output projection, dropout, loss, optimizer, 층·너비·head 수는 동일하다.
샘플링 보정 a_e를 포함한 전파는 기존 U=V-beta D^-1 B^T diag(a*C) B V다.

실제 수신 계수는

    alpha_(i<-j) = exp(s_ij + log(a_ij)) / sum_k exp(s_ik + log(a_ik))

이므로, z_e=s_e+log(a_e)와 수신 노드별 m_i=max_(e incident to i) z_e를 사용해

    alpha_(i<-j) = exp(z_ij-m_i) / sum_k exp(z_ik-m_i)

로 계산한다. m_i는 분자와 분모에서 상쇄된다. 수치용 최대값의 미분을 끊어도
원래 score 및 보정 계수에 대한 alpha의 미분은 같다. 분모 전체의 미분은 유지한다.
물리 엣지의 s와 C 정의는 양방향에서 공유된다. 양쪽 수신 계수만 각 이웃 집합에서 정규화한다.
고립 노드는 topology로 판정해 기존 V를 유지한다. 어떤 엣지도 제거하거나 제한하지 않는다.

수식은 실수 연산에서 같다. FP32에서는 연산 순서가 달라 반올림 차이가 있으며,
아주 작은 alpha 자체의 underflow까지 없애는 것은 아니다. 원래 exp 절댓값이 표현 불가능한
경우에도 정규화 비율은 계산할 수 있다. 최대값 이동은 C의 정의나 가중 라플라시안의
절댓값을 변경하는 연산으로 사용하지 않는다. 이 항등식은 현재의 row-normalized 전파에
적용된다. 나중에 비정규화 에너지 X^T L X를 계산할 때 raw C의 절댓값 대신 alpha를 쓰면 안 된다.

## 관측과 개입

모델 내부 엣지 상태는 명시적으로 **log C**를 전달한다. 이 경로의 OperatorOutput의
conductance/effective_weight 필드 역시 각각 log C/log(a*C)이며 model contract에 명시한다.
raw C를 1로 바꾸거나 clip하는 방식이 아니다. 기존 raw-exp 클래스·실험 파일은 보존한다.
직접 train CLI의 `--conductance-evaluation raw_exp`는 과거 경로 검사용으로 남겨 둔다.
study의 네 조건은 모두 `log_row`를 사용한다.

- 관측 파일의 `log_c`는 score 분포와 같은 입력에서 optimizer 전후 변화다.
- 실제 forward에서 사용한 alpha와 무갱신 재계산 오차를 그대로 비교한다.
- CPU FP64 기준도 log 좌표에서 계산하며 수치 변화와 유용한 학습의 결론을 구분한다.
- 기록되는 live gradient는 d(CE)/d(log C)다. d(CE)/dC로 표시하지 않는다.
- alpha를 통한 gradient만 있는 현재 전파에서는 d(CE)/d(log C)=C*d(CE)/dC다.
- raw C 분포·raw C 미분을 유한한 텐서로 얻었다고 주장하지 않는다.
- C=1 대조 및 개입은 log C=0으로 평가한다. sampling correction은 유지한다.
- score/H/Q/K와 층별 RMS, parameter 미분·갱신, AdamW decay 비교를 유지한다.

최종 signal_trajectory에는 각 기록의 `conductance_coordinate`가 c/log_c를 구분한다.
기존 관측 파일과 새 관측 파일을 같은 절댓값 통계로 혼합하지 않는다.

## 검증과 실행 한계

별도 synthetic debug 검사로 수신 계수와 score·sampling 보정 미분, 전체 8/256/8 모델의
출력·모든 parameter 미분, 동일 초기값·RNG, 두 초기화와 두 C 조건, C=1 개입,
activation checkpoint 켜기/끄기, 고립 노드와 빈 엣지 집합을 확인한다.
±1000의 유한 점수와 1000 수준의 log C를 가진 모델에서 optimizer·관측 경로도 검사한다.
FP64 수학 검사는 테스트에서 static topology의 dtype을 양쪽 모델에 동일하게 FP64로 맞춘다.
실제 기본 학습은 FP32이며 full production을 FP64로 전환하지 않는다.

유한성 검사는 층별 host 확인이 필요하며 처리시간에 비용이 있다. 불필요한 개별 검사들은
합쳐서 수행한다. 모델/데이터/physical batch/200epoch 목표를 축소하지 않는다.
새 구현의 서버 MIG 처리시간·VRAM·전체 학습은 아직 확인되지 않았다. 서버 calibration은
이 실행 경로로 다시 측정해야 하며, 이전 raw-exp calibration을 그대로 승인 자료로 쓰지 않는다.
기존 model-only best checkpoint에는 optimizer 재개 상태가 없어 정확한 중간 재개를 제공하지 않는다.

240.2초가 유지될 때 baseline learned 200epoch는 약 13시간 21분이다.
이 수치는 이전 구현 한 조건의 첫 epoch 관측이며 새 구현이나 전체 네 조건의 ETA가 아니다.
