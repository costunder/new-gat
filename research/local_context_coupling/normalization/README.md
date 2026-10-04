# 로컬 문맥 결합: 내부·교차 정규화 실험

이번 실험은 **같은 copy 결합 구조에서 정규화만 바꾸면 교차 경로의 분류 기여가 달라지는가**를 확인합니다.

1. 내부 전파의 step을 그래프 전체에서 하나로 정하는 대신, 각 1홉 로컬 그래프에서 정합니다.
2. 교차 전파의 step을 그래프 전체에서 하나로 정하는 대신, 각 cross 엣지 양 끝 copy의 연결 수로 정합니다.
3. 각 경우를 cross off, 고정 강도 1, 학습 강도로 비교합니다.

각 macro 층은 다음 연산을 사용합니다.

\[
\boxed{T_\rho=M(I-S)(I-\rho G)(I-S)R},\qquad
T_\rho-T_0=-\rho MSGSR.
\]

S는 step까지 포함한 내부 가중 라플라시안 block이고, G는 step까지 포함한 cross 가중 라플라시안입니다.
R은 물리 노드의 특징을 로컬 copy로 복제하고, M은 같은 물리 노드의 copy를 균등 평균합니다.
Cross는 인접한 로컬 그래프에서 **같은 물리 노드의 copy끼리** 연결합니다.

## 20개 조건

C는 unit/local_degree, 내부 정책은 graph/local입니다. 각 조합에 아래 다섯 조건을 둡니다.

| cross 정책 | 강도 | 조건 |
| --- | --- | --- |
| none | 0 | off |
| graph | 1 | fixed |
| graph | sigmoid(θ) | learned |
| edge | 1 | fixed |
| edge | sigmoid(θ) | learned |

총 2 × 2 × 5 = **20개 고유 조건**입니다. Off는 cross 정책별로 중복 학습하지 않습니다.
기존 6조건은 graph 내부 정책의 off와 graph cross fixed/learned로 포함하며 이번 계약에서 다시 학습합니다.
과거 결과 파일을 새 성능 값에 합치지 않습니다.

Learned는 seed 모델마다 θ 하나를 두 macro 층에서 공유합니다. 초기 θ=0, gain=.5입니다.
내부 C와 각 cross 엣지 가중치는 고정입니다. 학습하는 것은 projection과 cross의 전체 강도입니다.

## 본학습 규모

- Cora·CiteSeer·PubMed의 전체 노드·특징·물리 엣지·유도 로컬 그래프·cross 연결을 사용합니다.
- 2개 macro 층, hidden dimension 64, dropout .5, 학습마다 500 epoch입니다.
- LR .001/.003/.01과 tuning seed 101/202/303으로 validation에서 선택합니다.
- Final seed는 11/23/37/53/71입니다. 모든 final 선택을 고정한 후 test를 평가합니다.
- Tuning 540회와 final 300회, **총 840회 학습과 420,000회 독립 모델 갱신**입니다.
- 완료된 원본 `local-energy-20261004-002815`의 그래프 201개를 검증하고 보존합니다. 분류 학습은 그중 전체 citation 그래프 3개를 사용합니다.
- GPU/CPU 계측, 독립 seed packing, 정적 cache, 모든 연결을 처리하는 exact edge chunking과 activation checkpointing을 유지합니다.

## 결과에서 확인할 것

보고서는 cross 효과 24개, cross 정규화 비교 8개, 내부 정규화 비교 10개, C 비교 10개를 기록합니다.
내부·교차 정책의 상호작용 12개도 같은 seed에서 네 조건을 먼저 비교해 계산합니다.
같은 checkpoint의 gain0/gain1 개입은 새로 학습한 조건 비교와 별도로 기록합니다.

Branch 에너지는 적용된 S/G 기준입니다. Off의 G 에너지는 graph G에 대한 참고 진단입니다.
Gain의 학습, 실제 출력 변화, 분류 개선을 각각 확인합니다. 에너지나 출력 차이가 있다는 사실만으로 정보 복원을 주장하지 않습니다.

서버 FULL 본학습의 성능은 실행 후 CSV와 완료 파일로 판단합니다. DEBUG는 별도 fixture의 구현 검사입니다.

- [서버 실행과 결과 확인](RUN.md)
- [수식과 정확한 정규화](MODEL_MATH.md)
- [실험 설계와 판단 기준](EXPERIMENT_DESIGN.md)
- [전체 본학습 설정](config_full.json)
- [실행 검증 기록](VERIFICATION.md)

