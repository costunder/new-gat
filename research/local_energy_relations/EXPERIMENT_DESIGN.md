# 첫 실험 설계 — 로컬 에너지·관계·집계 감사

## 목적

**각 노드의 1홉 induced graph에서 이차 에너지를 계산하고, 서로 연결된 두 중심의
서로 다른 로컬 집합 사이에서 실제 연결을 따른 쌍선형 관계를 계산한다.**
첫 단계에서는 그 계산과 집계가 수식대로 되는지, 어떤 메시지를 관측·전달했는지를 확인한다.
아직 학습한 C/W의 분류 효과나 새로운 GNN의 성능을 평가하는 단계는 아니다.

## 고정 계약

| 항목 | full |
| --- | --- |
| 그래프 | 기존 fixed generator의 합성 198개 + 실제 Cora/CiteSeer/PubMed 3개, 총 201개 |
| 합성 그래프 | cycle/star/grid/ER/tree/tree+chord, 노드 수 20/30/40/60/80/100; 기존 seed·draw 계약 유지 |
| 합성 입력 | 모든 16개 독립 scalar feature columns |
| 실제 입력 | 기존 검증 loader의 모든 노드·전체 특징 채널 |
| 실제 입력 전처리 | row-sum-normalized float32 citation 특징을 float64로 승격; 원본 raw BOW라고 하지 않음 |
| 로컬 집합 | 모든 중심의 `{중심}∪N(중심)` 및 이웃끼리 엣지를 포함한 induced graph |
| 방향·ID | 원래 node ID로 물리 엣지 방향 통일, 로컬 occurrence와 물리 ID를 구분 |
| 고정 C | unit, local_degree=2/(로컬 endpoint degree 합) |
| 상태 | 같은 global unit Laplacian에서 두 번 확산한 H0/H1/H2 |
| 비교할 중심 쌍 | 원래 연결된 모든 중심 쌍의 두 방향 |
| 집합 사이 관계 | 공통 물리 엣지, 공통 노드, 서로 다른 incident 엣지의 기여 |
| 단계 비교 | 같은 단계 및 이웃 중심의 t→t+1; 같은 ego의 t→t+1도 별도 기록 |
| 계산 정밀도 | float64; 모든 노드·엣지·특징 사용, sampling ratio=1 |
| 학습 | trainable parameter 0, optimizer/epoch/update 없음 |

DEBUG는 따로 표시한 합성 18개와 DEBUG citation fixtures 3개, 총 21개다.
DEBUG 성공을 full의 실제 citation 평가 완료로 제출하지 않는다.
실제 citation labels/splits가 loader에 존재해도 이번 계산은 CE·accuracy를 만들지 않는다.

## 측정과 판정

1. **내부 에너지**: `tr(HᵀBᵀCBH)`와 엣지별 `Σc||BH||²`의 일치, `||q||²`와의 구분.
2. **실제 소거 성분**: Euclidean cycle 성분 norm 및 `Bᵀq_cycle` 잔차. cycle 개수만으로 손실량을 대신하지 않음.
3. **제한된 복원**: 알려진 양의 C와 전체 로컬 d에서 실제 q 복원 및 선형계 잔차. 부분 관측 복원 주장 없음.
4. **집합 사이 관계**: 공통 노드 d 내적과 `q_vᵀ B_v_global B_u_globalᵀ q_u`의 일치.
5. **관계의 분해**: `Jnode=2Jshared+Jdistinct`; 같은 단계/교차 단계와 방향을 따로 기록.
6. **전달 범위**: 전체·남는·빠지는 d의 norm, 공통/누락/경계 엣지 메시지의 제곱 norm.
7. **중복 집계**: 모든 ego의 raw 합과 물리 unique 에너지의 차이; inverse-occurrence 보정과 평균 C 에너지의 일치.

이 항등식들은 float64의 정해진 오차 허용 범위에서 검사한다.
전달 연산은 bookkeeping·진단이다. 복사한 d를 H_next에 입력하는 모델은 이번 단계에 없다.
H1/H2는 전체 물리 라플라시안에서 얻은 고정 reference states다.
유한성·coverage·수치해법 검증에 실패하면 실패 기록을 남기며 결과를 0으로 대체하지 않는다.
서로 다른 incident 엣지의 항이 음수라고 오류로 처리하지 않는다.
0 norm에서 정의되지 않는 상대 비율은 정의 상태를 함께 기록한다.

## 자원과 실행

FULL은 서버의 실제 할당 CUDA 자원으로 실행한다. 로컬에서는 단위/DEBUG만 실행한다.
프로그램은 GPU/VRAM·CPU/RAM과 실제 적용 입력 규모를 기록한다.
로컬 topology와 incidence 대응은 cache하고 독립 ego/관계는 physical batch로 계산한다.
여러 batch 후보를 실제 측정해 처리량과 peak memory를 기준으로 선택한다.
메모리를 줄이기 위해 ego·채널·그래프를 생략하거나 중심 몇 개만 사용하지 않는다.
chunking은 전체 데이터 계약을 유지하는 작업 메모리 조절이며 sampling이 아니다.

결과는 그래프 종류와 C, reference 단계로 구분한다. citation 특징은 하나의 vector trace이고
합성 scalar draws는 생성된 입력의 반복이다. 이들을 같은 독립 표본 수로 합치지 않는다.
같은 graph의 방향·ego·stage를 독립 학습 seed로 취급하거나 임의의 유의확률을 만들지 않는다.

## 이 단계 뒤에 결정할 일

이번 감사로 각 내부 항과 관계 항이 실제 메시지·집계에서 무엇을 보여 주는지 먼저 읽는다.
그 다음에 사용자가 C 학습, W 학습, 관계 항을 prediction에 연결할지 결정한다.
새 학습을 미리 넣거나 고정 연산의 검사 성공을 학습 모델의 유용성으로 바꾸지 않는다.
