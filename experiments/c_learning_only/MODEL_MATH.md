# v2.0의 실제 계산

이 문서는 코드가 실행하는 범위를 설명한다. 새로운 보완 설계를 제안하는 문서가 아니다.

## 1. 부분그래프와 발생행렬

샘플 문맥마다 노드 집합 S와 물리 엣지 집합 E를 만든다. B의 각 행은 한 엣지를
나타내며 tail에 -1, head에 +1을 둔다. 여러 문맥은 disjoint union으로 한 번에 계산한다.
서로 다른 문맥의 특징 통계를 섞지 않는다. 한 forward의 모든 층에서 B는 같다.

## 2. 현재 C 생성기는 그대로 유지한다

층 입력을 H, 공유 생성기 파라미터를 phi라 쓰면 다음과 같다.

\[
\delta_\phi=\operatorname{cost}_\phi(H,E,\operatorname{context}(H,S,E)),\qquad
C_\phi=\operatorname{diag}(\operatorname{Solver}_8(\delta_\phi)).
\]

실제 구현은 엣지×head 배열을 사용한다. 대각행렬을 메모리에 만들지 않는다.
문맥은 노드 특징의 평균·표준편차와 degree·coverage 등의 구조 통계를 포함한다.
비용에는 정규화한 양 끝 특징의 차이 제곱과 학습된 문맥별 metric,
구조 항이 들어간다. 그래프별 중심 조정과 tanh 제한을 거친 비용이 solver에 들어간다.

solver는 매 forward에서 C=1로 시작해 8회 계산한다. 학습되는 것은 비용 생성기의
파라미터이며, 엣지마다 영구 보관하는 C 파라미터가 아니다.
내부 목적에는 비용, entropy, degree barrier가 있다. 양의 C를 만들며
문맥·head마다 omega 가중평균 C=1로 정규화한다.
omega는 기존 `edge_normalization_weight`로, 이후 집계의 sampling correction과
역할이 다르다. 현재 샘플러는 두 필드에 같은 보정 배열을 줄 수 있다.

관측하는 cost는 실제 solver에 들어간 `last_scores` 값이다. C의 학습 신호는
실제 forward의 live C에 hook을 걸어 측정한다. `last_c`의 gradient를 읽지 않는다.

## 3. C를 실제 가중 집계에 사용한다

head별 value는 \(V^{(h)}=HW_v^{(h)}\)다. 샘플링 보정을 a라 하면

\[
\widetilde c_{ij}^{(h)}=a_{ij}c_{ij}^{(h)},\quad
\alpha_{i\leftarrow j}^{(h)}=
\frac{\widetilde c_{ij}^{(h)}}{\sum_{k\in N(i)}\widetilde c_{ik}^{(h)}}.
\]

이웃이 있는 노드의 전파는

\[
U_i^{(h)}=(1-\beta_g^{(h)})V_i^{(h)}+
\beta_g^{(h)}\sum_{j\in N(i)}\alpha_{i\leftarrow j}^{(h)}V_j^{(h)}.
\]

beta도 기존 문맥 조건부 학습 경로를 유지한다. 고립 노드는 U=V다.
head를 결합하고 기존 출력 투영, ReLU, dropout을 적용해 다음 층으로 보낸다.
encoder·value·출력 투영·beta·decoder·C 생성기는 분류 CE로 함께 학습한다.

가중 라플라시안으로 쓰면 \(L=B^T\widetilde C B\)이고, 활성 노드에서
전파는 \(U=V-\beta D^{-1}LV\)와 같다. 기본 전파는 이 연산을 sparse 방식으로
계산한다. \(\operatorname{tr}(V^TLV)\)는 해당 고정 metric의 이차 에너지지만,
이번 v2.0에서는 그 scalar 에너지를 별도 특징이나 loss로 추가하지 않는다.

## 4. 두 종류의 C=1 비교

**독립 학습 대조:** fixed 모델은 생성기 파라미터가 없다. value·beta·출력 투영 등은
learned 조건과 같은 초기값에서 독립적으로 학습한다.

**학습 후 개입:** learned checkpoint의 나머지 파라미터를 고정하고 생성 C만 1로
바꾼다. a는 유지하고 degree·alpha·전파·예측을 다시 계산한다.
후속 층 특징도 그 전파에 따라 바뀐다. 생성기만 다르게 쓰는 전체 forward 개입이다.

C=1이어도 a가 비균일하면 alpha는 단순 균등 비중이 아니다.
모든 C를 같은 상수로 곱하면 row 정규화에서 소거될 수 있으므로 C와 alpha를 함께 본다.

## 5. 학습 전후 관측의 정확한 의미

epoch 첫 실제 학습 배치에서 층 입력 H를 저장한다. 평소대로 CE backward,
gradient clipping, AdamW 갱신을 수행한다. 이후 저장했던 H와 같은 연결·보정·문맥으로
갱신된 생성기를 실행한다. 앞선 층의 파라미터 갱신 때문에 H가 변하는 효과를 제외하고,
생성 규칙 phi의 갱신이 cost·C·alpha를 어떻게 바꿨는지 비교한다.

매 epoch의 관측 문맥은 달라질 수 있다. 서로 다른 epoch의 표를 동일 입력의
장기 추적으로 해석하지 않는다. 각 표 안의 갱신 직전·직후가 동일 입력 비교다.
전체 층·head·해당 배치 전체 엣지를 요약하며, 작은 표에는 미리 정한 수의 수신 노드와
그 모든 이웃을 기록한다. 이 표시 제한은 학습 노드·엣지를 자르는 제한이 아니다.
