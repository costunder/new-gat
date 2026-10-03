# Experiment 4.2 검토 요청

검토 대상은 경로 이차 차분 분기의 **C 의존 노드별 대각 정규화 재학습 비교**다.
README, MODEL_MATH, 두 config와 실행 코드, VERIFICATION을 함께 읽어 달라.

## 핵심 질문

현재 층 Z에서 생성한 양의 경로 가중치 C를 사용한다.

- 기존: `S_Q Aᵀ C A S_Q Z / (3κ)`.
- 새 모델: `S_C Aᵀ C A S_C Z / 3`, `S_C=diag(Aᵀ C A)^(-1/2)`.
- C=1에서는 두 정규화가 같아 고정 대조군 하나를 둔다.
- C 생성식, 두 층, hidden 64, dropout, α/β 파라미터화는 유지한다.
- C, 가중 대각, 양쪽 S_C를 모두 미분한다.

다섯 조건 모두 새로 학습한다. 각 조건의 전체 그래프·500 epoch·LR 후보·tuning/final seed를
기존 Experiment 4와 같게 유지한다. 전체 135 tuning + 75 final = 210 run이다.
모든 final checkpoint가 validation으로 확정된 뒤 test와 frozen 개입을 평가한다.

## 검토할 것

1. 수식과 실제 forward/backward가 일치하는가? 양쪽 S_C의 미분, 고립 노드, 모든 경로와
   chunk 경계, packed seed의 독립 Adam을 확인해 달라.
2. 고정 C에서 PSD·norm 상한을 증명한 범위와 전체 비선형 모델의 안정성 주장이 구분돼 있는가?
3. 전역·노드별 정규화의 fresh training 비교가 공정한가? 모델 선택에 test를 사용하지 않는가?
4. C=1/shuffle의 자연스러운 재정규화와 같은 norm으로 맞춘 frozen 개입은 각각 무엇을 측정하는가?
   첫 층 개입 뒤 다음 층의 C와 기준 메시지를 현재 Z에서 다시 계산하는가?
5. 분류 정확도·CE·실제 αL/βM 크기·C 변동·seed paired 차이를 함께 보고하는가?
6. 전체 서버 학습, 실제 데이터의 무갱신 gradient 검사, DEBUG pipeline 검증을 정확히 구분하는가?

파일에 구현된 사실, 테스트가 확인한 범위, 아직 실행하지 않은 실험과 새 제안을 구분해 달라.
실패를 단정하거나 설계 밖의 기능을 구현됐다고 설명하지 말고, 결함을 지적할 때는 파일·위치·근거를 제시해 달라.

기존 public test 결과를 본 뒤 설계한 후속 비교다. 같은 split/초기화 seed의 차이를
독립적인 새 그래프 일반화 성능으로 해석할 수 없다.
