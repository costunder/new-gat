# E/J를 넣는 층의 재학습 실험

**E/J를 첫 층에만 넣거나, 출력층에만 넣거나, 두 층에 모두 넣어 다시 학습한다.**
앞선 frozen 제거는 이미 학습한 모델의 반응을 보았다. 이번에는 각 위치가
학습 과정에서 필요한 역할을 스스로 배우도록 별도의 모델을 학습한다.

고정 `unit` / `local_degree` C마다 base 한 개와
E만 / J만 / E+J × 첫 층 / 출력층 / 두 층의 열 조건을 비교한다.
총 20조건이다. 모든 조건의 기본 전파, 전체 그래프, 두 층, hidden 64,
dropout, seed, 500 epoch 예산을 유지한다.

- [실험 설계와 전체 예산](EXPERIMENT_DESIGN.md)
- [모델 수식과 실제 forward](MODEL_MATH.md)
- [서버 실행·진행·결과 확인·재개](RUN.md)
- [검증 기록](../../../docs/LOCAL_PLACEMENT_VERIFICATION_20261004.md)
- [앞선 분류 결과와 후속 실험의 근거](../../../docs/LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md)

FULL은 840 runs / 420,000 updates다. 본학습은 서버에서 실행한다.
로컬 DEBUG는 별도 fixture의 360 runs / 1,080 updates이며 실제 citation 성능이 아니다.
기존 `prediction/` 코드와 결과는 보존한다. 이번 20조건은 모두 새로 학습하고,
기존 checkpoint를 새로운 위치 모델의 초기값으로 옮기지 않는다.

E는 각 1홉 induced local의 이차 에너지, J는 인접한 서로 다른 local의
공통 원래 노드 대응에서 계산하는 쌍선형 관계다. 두 스칼라를 활성 층의
학습 벡터로 노드 상태에 더한다. C는 이번 실험에서 학습하지 않는다.
이 모델을 엣지 메시지의 완전한 복원이나 사이클 정보 보존이라고 설명하지 않는다.
