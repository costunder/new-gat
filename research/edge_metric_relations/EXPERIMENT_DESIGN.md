# A/B/C 실행 계약

## 제안에서 수정한 네 항목

1. **RMS**: 영점 예외 처리를 없애고 sqrt(mean_square+1e−8)를 사용한다. ε=1e−4를 고정하고 영점/작은 입력/일반 입력의 값과 gradient를 검사한다.
2. **DA**: 정규화 pair 계수를 등장 행별 SUM → 물리 엣지별 등장 평균으로 바꾼다. F2와 입력/파라미터 수를 맞추되 동일 표현력이라고 부르지 않는다. scale·메시지 변화·generator gradient/update를 기록한다.
3. **B 기준선**: teacher를 만든 **동일 raw L_d**의 선형·제곱·다항식 기준선을 제공한다. C의 self-loop GCN 기준선은 별도 정의다.
4. **B 실험을 고정**: raw Qθ(X)X를 출력한다. teacher recipe 2개 × 목표 3개 × student recipe 2개 × 조건 6개를 전부 실행한다. learned 조건 4개에는 5seed, 500epoch를 유지한다.

원문 제안의 practical accuracy 0.5pp는 승인된 판정 기준으로 간주하지 않는다. 코드의 practical threshold는 null이며 결과의 성공/실패에 사용하지 않는다.

## 공통 조건

| 조건 | 대각 배율 | 비대각 결합 | 학습 generator |
|---|---|---|---|
| D0 | 1 | 0 | 없음 |
| D1 | learned | 0 | 대각 |
| F0 | 1 | 고정 t=1 | 없음 |
| F1 | 1 | learned | pair |
| F2 | learned | learned | 대각+pair |
| DA | pair-informed learned | 0 | 대각+pair |

unit/local_degree 두 recipe를 유지한다. F1/F2는 0 출력 초기화 때문에 처음에 pair 결합이 0이다. D1/F2/DA의 최초 대각 배율은 1이다. hidden generator는 출력 weight가 0인 첫 update에서 gradient가 0일 수 있으므로 여러 DEBUG update와 본학습의 매 epoch 기록으로 활성화를 확인한다. generator의 gradient가 존재한다는 것과 유용한 규칙을 배웠다는 것은 다르다.

## A: 식별과 수치 복원

기존 완료된 local-energy audit의 198 synthetic + 3 citation을 읽는다. 모든 original scalar/feature 채널을 사용한다. 고정 연산자, 반복 횟수 1/2/4/8/16, 전체/target/target 1홉 관측과 개별 q_v/E_v/J targets를 기록한다. 작은 전체 synthetic 그래프에서는 정확 nullspace·충돌을 검사한다. citation은 희소 수치 복원과 residual/수렴 상태를 기록하며 exact rank 결과라고 부르지 않는다.

이 첫 A는 고정 연산 감사다. 이전 학습 checkpoint가 없으면 학습된 연산을 검사한 것처럼 대체하지 않는다. 새 C의 완료된 checkpoint에서는 frozen 개입으로 실제 모델의 사용 여부를 별도 확인한다.

A의 operator view는21개다: L0(1회), Ld(1회/제곱), L0/Ld smoothing(각1/2/4/8/16회), 기존 copy off/on(각1 sandwich), 새로운 raw Q와 normalized residual P의 diagonal/F0/reference(각1회). noise0/1e−6/1e−3는 수치 안정성 probe이며 새로운 실제 학습 표본으로 세지 않는다.

## B: 규칙 학습과 구조 일반화

531그래프 = train240 + validation60 + ID120 + sizeOOD90 + familyOOD12 + familySizeOOD9다. 각 그래프의 16개 scalar 실현을 보존한다. 지정된 master seed와 새로운 명명 stream을 사용한다. topology 중복 재시도는 기록하고 그래프를 버리지 않는다. labelled topology 중복 제거를 isomorphism-level 독립성으로 해석하지 않는다.

대각, 대각 제곱, analytic pair 목표마다 unit/local_degree teacher를 제공한다. student recipe 2개와 조건 6개를 모두 비교한다. D1/F1/F2/DA는 seed11/23/37/53/71, Adam lr0.003, 500epoch다. 매 epoch 전체240 train 그래프의 gradient를 반영한다. disjoint batch와 exact accumulation은 계산 배치만 조정한다. validation으로 checkpoint를 선택하고 held-out에서 파라미터를 고정한다.

240 learned seed run, 120000 seed optimizer update. D0/F0의 결정적24조건과 train-only fit 기준선24조건에는 optimizer가 없다. train graph 수·16실현·모든 held-out·500epoch를 줄이는 fallback은 없다.

Teacher3는 새 family를 학생이 표현할 수 있는지 확인하는 positive control이다. 그 결과 하나로 자연 데이터에서의 우월성을 결론내리지 않는다. normalized message MSE와 고정 상태의 gradient/weight 변화, coefficient·message scale을 확인한다.

## C: 실제 분류

Cora/CiteSeer/PubMed의 public split, 기존 pinned 원본과 feature 전처리를 그대로 사용한다. 전체 그래프 두 층, hidden64, projection bias 없음, dropout0.5, 첫 층 ReLU, float32/TF32 off, Adam500epoch, projection weight decay0.0005, generator weight decay0이다. early stopping은 없다.

12 recipe/variant 조건에 G1/G2/P2를 추가한다. G1은 self-loop symmetric GCN, G2는 층마다 그 전파를 두 번 적용한다. P2는 같은 정규화 Ltilde=I−P_GCN의 2차 다항식을 학습하며 [0,2]에서의 정확 supremum과 1 중 큰 값으로 나눈다. 모든 조건의 projection과 dropout stream을 맞춘다.

- LR0.001/0.003/0.01 × tuning seed101/202/303: 405회.
- validation 선택 LR × final seed11/23/37/53/71: 225회.
- 총630회 × 500epoch = 315000 seed update.
- primary18 비교: F2−D1, F2−DA, F2−P2 × dataset3 × recipe2.
- primary accuracy: 양측 paired t, Holm18, α0.05. 원래 단위는 동일 split의 initialization seed다.
- CE·F1 비교·추가 개입은 exploratory로 표시한다. CI가 0을 포함한다고 equivalence를 주장하지 않는다.

public test는 앞선 연구에서 이미 관찰했으므로 이 단계는 탐색적이다. 새로운 독립 그래프 일반화는 C 결과만으로 입증되지 않는다.

FULL final metric675행, frozen intervention3915행, original+intervention layer diagnostic3060행을 검증한다. 개입은 적용 가능한 조건에 대해 no-op/offdiag_zero/diagonal_one/pair_zero를 첫 층/둘째 층/둘 다에 적용한다. 파라미터 갱신은0이며 checkpoint hash를 유지한다. 개입 뒤 hidden이 바뀌면 다음 층 계수는 재계산한다. 같은 Z와 같은 대각에서의 비대각 제거는 별도 진단이다.

## 자원과 실패 처리

FULL은 Linux 서버의 명시적으로 할당된 CUDA에서만 실행한다. 여러 GPU가 보이면 독립 job을 분배한다. 정적 구조 cache, seed packing, 모든 pair를 유지하는 exact chunk, checkpointing을 사용한다. graph/seed batch와 chunk 후보를 실제 throughput/peak VRAM으로 고른다. calibration update는 본학습으로 세지 않는다.

터미널에 phase/조건/epoch/손실/진행시간/측정 ETA를 출력하고 같은 로그를 저장한다. 실패 시 traceback과 failure.json을 남기며 기존 결과와 원격 세션을 보존한다. resume는 이전 결과를 읽고 새 디렉터리에 쓴다. DEBUG는 별도 profile이며 최종 결과가 아니다.
