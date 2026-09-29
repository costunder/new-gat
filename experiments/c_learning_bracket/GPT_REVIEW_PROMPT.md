# GPT 검토 요청

**최신 후속 수정:** `train_one`의 초기·epoch별 validation 및
`evaluate_selected`의 전체·새 문맥 평가에도 `expected_seed_ids`를 연결했다.
최종 문맥 평가 사이 parameter hash를 대조하며, RUN.md의 CE 설명을 고쳤다.
이번 CUDA 통합 테스트는 실제 학습·checkpoint 재로딩·평가 호출 12개와
저장 JSON의 coverage를 검사한다. 총 33개 테스트가 다시 통과했다.

실제 데이터 측정은 `evidence/prior_signal_audit_2_calibration`에 보존한 이전
판의 자료다. 현재 소스로 재측정하지 않았으며 hash를 바꿔 새 증거로 만들지 않았다.
`CHANGES_FROM_SIGNAL_AUDIT_2.json`으로 현재와 이전 소스 차이를 확인해 달라.
모델·관측 계산은 그대로이며 출력 초기화 대조 실험은 아직 수행하지 않았다.

아래는 직전 signal_audit_2의 관측 설계 설명이다.

이번 전달본은 **signal_audit_2 관측 보강판**이다. 직전 review_fixes_1의 여섯 수정은
수용된 상태다. 이번에는 모델을 바꾸지 않고 다음 두 가지를 구현·검사했다.

1. 실제 층 입력 → value → 혼합 → 출력 투영 → ReLU → dropout의 전체 RMS 기록.
2. calibration의 전체 validation·문맥별 평가 반환값과 원래 seed ID coverage 보존.

먼저 `REVIEW_FIXES.md`의 이번 측정 결과를 읽고 어느 연산에서 특징 크기가
얼마나 바뀌었는지 쉽게 설명해 달라. 초기 관측을 200epoch 후 학습 결과로
해석하거나, 관측만으로 초기화·정규화·점수 스케일 변경을 확정하지 말아 달라.
출력 투영 이후의 32채널 묶음은 기존 attention head와 동일한 좌표가 아니다.
전체 RMS 비율로 비교하고, 노드 특징의 동일화와 절대 크기 감소를 구분해 달라.
문맥별 정답 수·CE·parameter hash와 validation ID hash를 실제 원자료에서 대조해 달라.

이번 후보는 기존 C-only 실험에서 C 생성기만 로컬 Q/K 대칭 내적·exp로 교체했다.
구현 사실을 먼저 쉽게 설명하고, 첨부 수식과 실행 경로가 맞는지 확인해 달라.

실제 시작점은 `experiments/c_learning_bracket/train.py`다.
모델 경로는 `make_model → BracketClassifier → BracketOperator`이며,
C는 `BracketConductance`, 전파는 `diffusion.py::row_diffusion`이 계산한다.
이 함수는 기존 row 연산·sparse 커널을 유지하면서 실제 degree/계수를 검사한다.
패키지의 다른 실험 파일은 공용 import와 기존 코드 대조를 위한 의존성이다.
파일에 클래스가 정의되어 있다는 이유만으로 이번 forward가 그 클래스를
사용한다고 해석하지 말고 실제 생성·호출 경로를 확인해 달라.

확인할 점:

1. score=(Q_u·K_v+Q_v·K_u)/(2r), C=exp(score), head별 C인가?
2. 실제 엣지만 처리하고 N×N attention·반복 solver·C 생성용 그래프 통계가 없는가?
3. 초기 모델부터 새 생성기를 넣으며 이전 optimized backend를 실행하지 않는가?
4. 기존 value·beta·sampling correction·row-normalized 발생행렬 전파를 유지하는가?
5. beta에는 그래프 문맥이 남는다는 점과 C의 로컬성을 구분했는가?
6. Q/K와 degree 정규화까지 CE 미분이 연결되는가? score 기록의 부호 의미가 맞는가?
7. learned/fixed 공통 초기값·샘플 순서와 학습 후 C=1 개입을 구분하는가?
8. 시간 측정에서 관측 비용을 기본 모델 시간과 구분하는가?
9. `BracketInputs`가 같은 샘플을 유지하며 불필요한 forest/cycle 계획을 만들지 않는가?
10. 실제 pinning·degree overflow 보호·무갱신 재계산·H/Q/K 크기·decay 기록이
    이번 검토의 문제를 해결하는가? 수치적으로 구별됨을 학습 성공으로 오해하지 않았는가?

논문은 arXiv:2305.15616v3이다. 저자 코드와 달리 per-head C, 1/r 스케일,
독립 Xavier Q/K 초기화를 명시적으로 선택했다. 논문 전체 ODE나 보존 성질의 재현을
주장하지 않는다. 이번에 ODE·보완·cycle 복원 기능을 추가하라고 해석하지 말아 달라.

검토 답변은 **현재 코드 설명 → 확인된 불일치와 근거 → 미검증 항목** 순서로 해 달라.
합성 검사, 실제 데이터 calibration, 200epoch 본학습 결과를 구분해 달라.
추가 제안은 제안이라고 표시하고 현재 구현처럼 설명하지 말아 달라.
