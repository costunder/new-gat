# Conductance C-learning: 학습 비교와 checkpoint 검사 결과

기준일: 2026-09-01. 사용자가 대화에 붙여 넣은 `gat-c-learning-seed0-v1` 비교 보고서에
근거한 기록이다. 전체 보고서와 네 조건 모두 `passed`, model seed는 0이다.
**현재 결과에서는 학습 C가 고정 C=1보다 나은 validation 성능을 보이지 않았다.**
두 방법의 일반적인 동등성이나 C가 항상 불필요하다는 증명은 아니다.
이후 같은 learned checkpoint의 평균-C 검사도 `passed` 출력을 수령했다.
PPI에서는 checkpoint가 엣지별 C 변동에 의존했지만 fresh-training 이득과는 별개였다.

## 1. 근거와 실행 범위

| 항목 | 확인한 범위 |
|---|---|
| Run ID | `gat-c-learning-seed0-v1` |
| 조건 | PPI / ogbn-arxiv × `learned_c` / `fixed_c` |
| 상태 | 비교 보고서와 네 조건 모두 `passed` |
| Seed / 평가 | model seed 0, validation only, test 미평가 |
| 구현 게시본 | C-learning 구현은 `25ca328`에 게시됨 |
| 실제 서버 실행 revision | 붙여 넣은 비교표에 없음. 서버 manifest의 revision을 독립 확인하지 않음 |
| 근거 형태 | 사용자 inline 보고서. 별도 첨부 파일이나 그 파일의 SHA-256은 없음 |

구현 게시 commit을 실제 서버에서 실행한 commit이라고 단정하지 않는다. 서버의 checkpoint,
manifest, history 전체도 로컬로 받아 독립 재검증하지 않았다. 이후 사용자 terminal 출력에서
원 validation 재현과 평균-C 검사의 완료를 확인했으며 그 범위는 6절에 별도 기록했다.

이 비교는 node-degree 정규화 아래에서 adaptive C만 분리한 내부 실험이다.
구현의 공통 학습 정책은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005,
non-gate WD 0.0005, 최대 200 epochs, patience 50, PPI batch size 2, workers 0,
FP32/AMP OFF/TF32 OFF/compile OFF다. Learned gate의 WD는 0.0005다.
Fixed 조건은 모든 물리 엣지의 C를 1로 고정하고 gate를 학습하거나 실행하지 않는다.
두 조건을 같은 새 run에서 처음부터 학습하며 과거 2×2의 learned 점수를 재사용하지 않는다.

## 2. Validation 점수와 학습 길이

| 데이터 / 지표 | 조건 | Validation (%) | Best epoch | 실행 epoch | 활성 학습 파라미터 | 동결 파라미터 |
|---|---|---:|---:|---:|---:|---:|
| PPI micro-F1 | learned_c | 52.564966 | 64 | 114 | 36,347 | 0 |
| PPI micro-F1 | fixed_c | 52.705738 | 90 | 140 | 11,385 | 24,962 |
| ogbn-arxiv accuracy | learned_c | 68.317723 | 195 | 200 | 36,074 | 0 |
| ogbn-arxiv accuracy | fixed_c | 68.324435 | 195 | 200 | 11,112 | 24,962 |

| 데이터 | Learned − fixed (percentage points) |
|---|---:|
| PPI | −0.140772 |
| ogbn-arxiv | −0.006711 |

원 보고서의 대비를 그대로 보존했다. arxiv의 반올림된 퍼센트 점수끼리 빼면 마지막 자리가
다를 수 있으므로 보고서 대비를 반올림된 표에서 다시 계산해 덮어쓰지 않는다.
PPI micro-F1과 arxiv accuracy를 하나로 평균내지 않는다.

Fixed 조건은 활성 학습 파라미터가 약 69% 적다. 다만 동일 초기 state와 RNG를 맞추려고
동결 gate scaffold 24,962개를 보존하므로 **전체 저장 파라미터 수 감소가 아니다**.
이 표만으로 VRAM·checkpoint 크기·시간·속도 개선을 주장하지 않는다. 서로 다른 실행 epoch도
있으므로 학습 시간 비교가 필요하면 별도의 동등한 예산 측정이 필요하다.

## 3. 선택된 checkpoint의 C와 전파

C CV는 그래프 내부 `std(C)/mean(C)`를 먼저 계산한 뒤 그래프를 동일 가중 평균한 값이다.
PPI는 validation 그래프 2개(2/2), arxiv는 전체 transductive 그래프 1개(1/1)다.
층은 0부터다. 상대 Conv 변화량은 LayerNorm/ELU 전의
`||H_after_conv−H_before_conv||/||H_before_conv||`다.

| 데이터 | 조건 | 층 | C CV 평균 | rho 평균 | 상대 Conv 변화량 | Gate parameter L2 |
|---|---|---:|---:|---:|---:|---:|
| PPI | learned_c | 0 | 0.541124 | 0.946912 | 1.02837 | 2.08729 |
| PPI | learned_c | 1 | 0.127339 | 0.946912 | 0.780309 | 1.43187 |
| PPI | fixed_c | 0 | 0 | 0.946912 | 1.00178 | 6.59735 |
| PPI | fixed_c | 1 | 0 | 0.946912 | 0.786742 | 6.55908 |
| arxiv | learned_c | 0 | 0 | 0.95 | 0.72908 | 0.000238633 |
| arxiv | learned_c | 1 | 0.00948437 | 0.95 | 0.443828 | 1.06375 |
| arxiv | fixed_c | 0 | 0 | 0.95 | 0.727876 | 6.59124 |
| arxiv | fixed_c | 1 | 0 | 0.95 | 0.444553 | 6.56015 |

Fixed 행의 gate L2는 **실행하지 않는 동결 초기 scaffold**의 norm이다. 이 gate가 C를
생성하거나 학습에 기여했다는 뜻이 아니다. 실제 fixed C는 정확히 1이다.
또한 두 조건의 비고립 노드 rho는 연산 정의상 0.95다. PPI 평균이 조금 낮은 것은 고립 노드의
rho가 0인 집계를 포함하기 때문이며, rho가 같다는 사실은 예측 함수가 같다는 증명이 아니다.

PPI learned 모델은 비상수 C를 학습했지만 fixed보다 validation 점수가 높지는 않았다.
arxiv learned 모델은 첫 층 C가 상수이고 둘째 층 변동도 작으며 fixed와 점수 차이가 작다.
**엣지별 C가 변한다는 사실, 현재 모델이 그 변동을 사용하는지, 새로 학습했을 때 성능이
개선되는지는 서로 다른 질문**이다.

## 4. 이전 2×2와 합치지 않는 이유

이전 run `gat-factorial-seed0-v1`의 PPI `node_degree`는 52.465469%이고,
이번 run의 PPI `learned_c`는 52.564966%다. 이번 fixed와 비교할 learned 값은
**이번 run의 52.564966%**다. 두 실행의 차이를 새로운 요인의 효과로 해석하지 않는다.
현재 자료만으로 실행 간 차이의 원인을 특정하지 않으며 같은 seed가 CUDA의 비트 단위
동일성을 보장한다고 주장하지 않는다.

이전 2×2에서 관측한 정규화 개선은
[당시 결과 문서](CONDUCTANCE_FACTORIAL_FINDINGS.md)에 그대로 보존한다.
그 결과를 C 학습의 기여로 재명명하거나 이번 fixed 조건으로 소급 대체하지 않는다.

## 5. 지금 말할 수 있는 것과 없는 것

- 이 seed·데이터·설정·validation 선택에서는 learned C의 성능 이득을 확인하지 못했다.
- 현재 후속 실험의 간단한 기준은 fixed C=1로 둘 수 있지만, 이를 모든 데이터/seed의
  최적 조건이나 외부 GCN/GAT/GraphSAGE의 재현으로 부르지 않는다.
- 차이가 작다고 두 모델이 동등하다는 통계적 결론은 내리지 않는다. n=1이고
  표준편차·CI·p-value·동등성 검정이 없다. 특히 arxiv의 best epoch가 예산 끝에 가까워
  더 긴 학습에서도 같은 관계라고 보장하지 않는다.
- 반복적인 validation 기반 설계 선택은 selection bias를 만들 수 있다. Test를 아직
  평가하지 않았으므로 최종 test 성능, 논문 경쟁력 또는 novelty 주장을 붙이지 않는다.
- 이번 표는 환경 호환성 전반이나 실행 최적화의 가속 실측이 아니다.

## 6. 수령한 같은 run의 learned checkpoint 평균-C 검사

추가 학습 없이 **`gat-c-learning-seed0-v1`의 각 데이터셋 `learned_c`** 원 validation을 먼저 재현하고,
C를 그래프·층별 평균으로 교체해 검증한다. 전체 층 교체와 한 층씩 교체를 분리하며
교체한 C로 weighted node degree를 다시 계산한다. 다른 학습 파라미터는 고정한다.
이는 새 fixed 모델의 성능을 다시 측정하는 단계가 아니라 **선택된 learned checkpoint가
엣지별 C 차이에 현재 얼마나 의존하는지**를 확인하는 읽기 전용 개입이다.

```bash
git pull --ff-only
bash research/conductance_gat/c_learning/audit.sh --source-run results/conductance_gat/c_learning/gat-c-learning-seed0-v1 --output-dir results/conductance_gat/c_learning_audits/gat-c-learning-seed0-v1
cat results/conductance_gat/c_learning_audits/gat-c-learning-seed0-v1/report.md
```

검사기는 manifest의 suite를 읽어 기존 `conductance_factorial/node_degree`와
새 `conductance_c_learning/learned_c`를 구분한다. 이번 명령은 **새 C-learning run**을 쓴다.
원본 model/source/cache/checkpoint와 validation 재현 검사에 실패하면 유효한 대비를 만들지
않는다. 새 보고서만 별도 폴더에 기록하고 원본 checkpoint·학습 결과는 덮어쓰지 않는다.
이미 같은 보고서 폴더가 있으면 `--output-dir`에 새 이름을 지정한다. 재학습·optimizer step·
test 평가·다운로드·더미 데이터 생성은 없다. 이 확장 검사의 실제 GPU terminal 출력을
이후 수령했으며 상태는 `passed`다. 사용자 terminal 출력에 표시된 short revision은 `8f6b4da`다.
원본 서버를 직접 조사한 것은 아니고 inline 근거이므로 첨부 파일 hash를 만들지 않는다.

원 run은 `gat-c-learning-seed0-v1`, condition은 `learned_c`, model seed는 0이다.
PPI 저장/재계산 validation은 모두 52.564966%이고, arxiv는 저장 68.317723%,
재계산 68.317729%다. 아래 delta는 저장된 반올림 값이 아니라 **검사에서 재계산한 원래 값**에 대한 차이다.

| 데이터 / 지표 | 평균 C로 바꾼 층 | Validation (%) | Δ 원래 (pp) | 바뀐 예측 (%) | Logit 평균 절대 차이 |
|---|---|---:|---:|---:|---:|
| PPI micro-F1 | 전체 층 | 45.915526 | −6.649440 | 7.619317 | 0.340052 |
| PPI micro-F1 | layer 0만 | 46.366699 | −6.198266 | 7.279934 | 0.332338 |
| PPI micro-F1 | layer 1만 | 52.326050 | −0.238916 | 1.572709 | 0.0758783 |
| arxiv accuracy | 전체 층 | 68.284171 | −0.033558 | 0.184570 | 0.00299709 |
| arxiv accuracy | layer 0만 | 68.317729 | 0 | 0 | 1.66161e-7 |
| arxiv accuracy | layer 1만 | 68.284171 | −0.033558 | 0.184570 | 0.00299709 |

PPI의 바뀐 예측 비율은 node-label 이진 판정 전체에서의 flip 비율이다. arxiv는 validation
노드의 argmax class flip 비율이다. 둘을 동일한 표본 단위로 해석하거나 합산하지 않는다.
표는 사용자 출력의 반올림 값을 그대로 보존했다.

PPI에서 전체 층 평균화 시 −6.649440pp이며 큰 변화는 layer 0 개입에서 나타났다.
즉 **선택된 learned checkpoint는 학습한 엣지별 C 패턴을 실제로 사용한다.** 하지만 앞서
새로 학습한 fixed 모델은 52.705738%였고 learned보다 낮지 않았다. 다른 파라미터를 고정한
추론 개입과 처음부터 고정 C에 적응하도록 학습한 모델은 다른 질문이므로 모순이 아니다.
층별 개입 효과는 더해서 전체 층 효과로 만들 수 없다. 비선형 층과 이후 층의 입력이 함께 변한다.

arxiv는 layer 0 개입의 예측 flip이 0이고 수치 수준의 작은 logit 변화만 있다.
Layer 1/전체 층 개입도 −0.033558pp로 작다. 이를 일반적인 함수 동등성이나 다른 seed의
결과로 확대하지 않는다. 모든 값은 validation이고 새로운 test 평가는 없다.

해석할 때는 PPI와 arxiv 각각 원 validation 일치 여부, all-layer/층별 점수 차이,
logit 변화와 prediction flip을 확인한다. 점수가 유지돼도 logits가 바뀔 수 있으므로 점수 하나만
보지 않는다. 개입 민감도가 있어도 fresh-training 이득이 없을 수 있으며 두 결과는 모순이 아니다.

## 7. 다음 독립 가설: 직접 C v2와 상대 C 생성기 v3

첫 번째 별도 실험은 [Conductance v2](../research/conductance_gat/v2/README.md)다.
고정 그래프의 canonical 물리 엣지·층마다 alpha를 두고 `c_e=exp(alpha_e)`를 직접 학습한다.
Alpha는 0으로 초기화해 C=1에서 시작한다. C 생성 MLP와 고유분해는 없고,
node-degree 전파·backbone은 유지한 채 직접 C와 fixed C=1의 두 조건을 새로 비교한다.

기존 MLP는 공유 edge 함수를 학습하는 올바른 설계다. V2는 수학 오류를 고치는 것이 아니라
**graph-bound 직접 파라미터화**라는 다른 가설이다. 특정 그래프의 엣지에 파라미터가 묶이므로
PPI의 unseen 독립 그래프에는 적용하지 않는다. 기본은 arxiv × direct/fixed × seed 0의
2개 CUDA 학습이고 arxiv는 여전히 full-batch다. Cora/CiteSeer/PubMed는 명시적으로 선택할 수 있다.
이 v2의 실제 GPU 결과는 아직 없으며 이전 결과를 v2 성능으로 재사용하지 않는다.

추가 [Conductance v3](../research/conductance_gat/v3/README.md)는 v2를 대체하지 않는다.
방향 불변인 노드상태·degree 특징의 공유 MLP score를 그래프별 중심화해 상대 C를 만들고,
등방성 혼합 gamma와 bounded tau, 대칭 정규화의 전파 강도 alpha를 별도로 학습한다.
같은 arxiv와 seed 0에서 `relative_c`/`fixed_c`를 새로 학습하며 fixed 조건도 alpha는 학습한다.
선택된 checkpoint에는 평균 C·shuffled C·C=1·전파 제거 validation 개입을 실행한다.
대칭 정규화에서 평균 C와 C=1은 동등하므로 둘은 독립 대비가 아니라 수치 검산이다.
이 개입은 fresh fixed-C 재학습과 다른 질문이다. V2와 v3 사이에는 파라미터화·정규화·alpha·
optimizer가 함께 달라 전체 점수 차이를 C 한 요인의 효과로 해석하지 않는다. 실제 v3 GPU 결과도
아직 없으며, 단일 seed validation 결과를 유의성·test·SOTA 주장으로 확대하지 않는다.
