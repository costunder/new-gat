# 실험 결과와 구현 상태

기준일: 2026-08-31 (Asia/Seoul).

이 문서는 사용자가 제공한 **서버 결과 출력**과 **현재 소스 버전의 구현**을 구분한 기록이다.
문서 작성 자체가 새 학습을 실행했다는 뜻은 아니다. 수치는 사용자 로그에서 확인했으며,
서버의 전체 원본 checkpoint/manifest/history 파일을 로컬로 받아 독립 재검증한 것은 아니다.

## 1. 소스 버전과 측정 범위

후속 사용자 요청으로 현재 기본 실행은 model seed **0 하나**다. 기존 5-seed 측정값은 아래에
그대로 보존하며, 기본값 변경이 과거 결과나 source revision을 바꾸지는 않는다.
단일 seed의 std/CI는 null로 기록한다. 새 read-only `--full-audit`는 C 평균/셔플/전파 제거와
train-label gradient를 검사하며 **5e801c3 실행의 새 GPU 로그를 수령했다.** 아래 확장 검사 절에
기록했다. 이후 **43afd63의 2×2 GPU 재학습 결과도 수령했으며 여덟 조건 모두 passed**다.
이 결과를 근거로 추가한 C-learning 비교와 평균-C 검사는 아직 새 GPU 결과가 없다.

후속 코드의 로컬 회귀는 **890 passed / 64 skipped** (30.76 s, exit 0), Ruff 통과다.
새 학습 두 조건과 평균-C 검사는 작은 단위 fixture로 검증했고 공개 데이터 학습은 실행하지 않았다.
생략은 Linux/Bash 62개, 로컬 PyG 미설치 1개, Windows 실제 symlink 권한 1개다.

| 구분 | 확인된 상태 |
|---|---|
| 이전 진단 전용 게시 commit | `ebf8cd19b80e6cd6c742b132e2bb1dadb97b019c` |
| 이전 commit의 추가 내용 | Conductance 진단 Python/Bash, 전용 테스트, 안내 문서, 트랙 README의 5개 파일 |
| 기존 학습 코드 | 위 진단 commit은 기존 benchmark의 모델·학습 수식을 변경하지 않음 |
| Cycle PE 기저벡터 v2 | 이 소스 버전에 포함, 로컬 단위 검증 완료. 실제 GPU 결과 미수령 |
| 실행 최적화·선택적 compile·속도 도구 | 이 소스 버전에 포함, 로컬 단위 검증 완료. GPU 가속 실측 미수령 |
| 단일 seed 기본값·확장 checkpoint 검사 | 5e801c3 GPU full audit 수령, seed 0 다섯 데이터셋 passed |
| Gate WD × normalization 2×2 | 43afd63 실제 GPU 결과 수령. PPI/arxiv × 4조건 × seed 0 모두 passed |
| Node-degree의 learned C vs fixed C | 별도 `c_learning` 코드 추가. 2데이터 × 2조건 × seed 0, 새 GPU 결과 미수령 |
| Node-degree checkpoint mean-C 개입 | 별도 읽기 전용 검사 추가. 새 GPU 결과 미수령 |
| `code_summary.md` | 이 버전의 source/test/config/script 전체를 파일별로 보존한 스냅샷 |

`ebf8cd1`까지만 받은 서버에는 새 기능이 없으므로 업데이트 후 `git rev-parse HEAD`로
실행 revision을 확인한다. 소스 업데이트가 서버에서의 실행 완료를 뜻하지는 않는다.
아래 기존 학습 결과를 v2·최적화·새 2×2/C-learning 결과로 재분류하면 안 된다.

### 수령한 2×2 GPU 재학습: 정규화 효과와 C 학습은 별개

Run `gat-factorial-seed0-v1`, 소스 `43afd632b97a4285dfeae26847b4f12a8fd1a1e4`,
model seed 0. NVIDIA RTX A6000, Python 3.11.16, Torch 2.7.1+cu118, PyG 2.7.0,
Linux glibc 2.35에서 여덟 fresh training과 최종 비교표가 모두 `passed`다.
Train 정답으로 학습하고 validation으로 선택했으며 **test는 평가하지 않았다**.

| 조건 | PPI validation micro-F1 (%) | arxiv validation accuracy (%) |
|---|---:|---:|
| baseline | 48.986770 | 50.927883 |
| gate_no_wd | 49.378028 | 50.565451 |
| node_degree | 52.465469 | 68.317723 |
| node_degree_gate_no_wd | 50.340520 | 67.995566 |

두 데이터 모두 **node_degree + gate WD 0.0005**가 최고다. 기존 정규화 대비 PPI
+3.478699pp, arxiv +17.389840pp다. WD를 제거하면 C 변동은 커지지만 성능은 일관되게
좋아지지 않았다. 특히 arxiv 최고 조건의 두 층 C CV는 0과 약 0.00948이므로
이 결과만으로 학습 C의 기여를 입증할 수 없다. PPI에는 비상수 C가 남아 있어
반대로 C가 항상 불필요하다는 결론도 성립하지 않는다.

정확한 epoch·5개 대비·층별 C/rho/전파량·해석 경계와 다음 분석은
[Conductance 실험 정리](CONDUCTANCE_FACTORIAL_FINDINGS.md)에 보존했다.
단일 seed의 탐색적 validation 결과이며 유의성·일반적 최적값·SOTA 주장은 하지 않는다.

### 수령한 확장 검사: 기존 checkpoint에 대한 개입

학습 run `paper-20260830T150244764889Z`, model seed 0. 진단 실행 소스는 `5e801c3`이며
보고서 폴더 suffix는 `20260831T120740120251Z`다. 사용자 첨부 로그 SHA-256:
`CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F`.

| 데이터 (validation) | learned C | mean C | shuffled C | graph off |
|---|---:|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.646000 | 0.646000 | 0.632000 |
| CiteSeer accuracy | 0.644000 | 0.642000 | 0.642000 | 0.626000 |
| PubMed accuracy | 0.724000 | 0.726000 | 0.726000 | 0.718000 |
| PPI micro-F1 | 0.487508 | 0.487508 | 0.487508 | 0.452144 |
| arxiv accuracy | 0.509279 | 0.509279 | 0.509279 | 0.508876 |

다섯 dataset 모두 완료, 최종 `Diagnostic status: passed`다. Validation 재검증 오차는
약 0~5e-8이다. 각 조건은 같은 checkpoint에 대한 개입이며 재학습한 baseline의 성능이 아니다.
PPI/arxiv는 두 층 모두 관찰한 C가 상수이고 mean/shuffle의 prediction flip이 0이다.
PPI에서는 graph-off 시 3.5364 percentage points 하락해 연결 구조의 기여와 C 차별화 실패가
구분된다. arxiv에서는 0.0403 points 하락하며 rho 중앙값 약 .000433으로 전파 영향이 매우 작다.

Gradient 검사는 eval/dropout-off, PPI는 첫 train batch 하나다. Gate 파라미터 norm과 task
gradient가 극소지만 학습 당시 Adam update 이력을 복원한 것은 아니므로 WD 원인 확정은 아니다.
기존 full audit의 극소 ratio는 분모 하한 1e-12 적용 여부를 확인해야 한다. 새 2×2의 train-mode
관찰은 raw task/decay norm을 따로 저장하고 0분모 비율만 null로 기록한다.

## 2. 사용자가 제공한 5-seed test 집계

모든 항목의 model seeds는 `0,1,2,3,4`다. `±`는 seed 사이 **표본 표준편차**이며,
표준오차·신뢰구간·5-fold 교차검증이 아니다. 모델·데이터·평가 조건이 다른 트랙끼리
수치를 직접 차감해 기여도를 추정하지 않는다.

### Conductance GAT

Run: `paper-20260830T150244764889Z`.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| Cora | accuracy | 0.661600 ± 0.023639 |
| CiteSeer | accuracy | 0.626800 ± 0.007918 |
| PubMed | accuracy | 0.721200 ± 0.005630 |
| ogbn-arxiv | accuracy | 0.486089 ± 0.002813 |
| PPI | global micro-F1 | 0.500051 ± 0.004877 |

이 값은 실행·집계 결과이지 경쟁력 또는 novelty의 증명이 아니다. 외부 논문 표와
비교하려면 split·입력 전처리·모델 크기·학습 설정·모델 선택 절차를 별도로 확인해야 한다.

### Cycle PE v1

Run: `paper-20260831T015711388279Z`. 모델 키는 **`cycle_set`**이다.

| 데이터셋 | 지표 | Test 평균 ± 표준편차 |
|---|---|---:|
| ZINC-12K | MAE, 낮을수록 좋음 | 0.189090 ± 0.016624 |
| Peptides-struct | MAE, 낮을수록 좋음 | 0.259728 ± 0.002816 |

이것은 cycle 기저를 여섯 통계로 요약하는 v1의 결과다. 좌영공간 기저벡터 전체를 입력하는
**`cycle_basis_v2` 결과가 아니다.** `schema_version=2` 같은 저장 형식 버전과 모델 v2도
구분한다. 같은 backbone의 PE 제외 ablation이 없으므로 이 표만으로 PE의 순수 효과를
분리할 수 없다. 두 데이터셋이 회귀 과제이므로 MAE 사용 자체는 맞다.

### Tree Augmentation

Run: `paper-20260831T060149709584Z`.

| 데이터·평가 조건 | 지표 | Fixed BFS | Multi-chart |
|---|---|---:|---:|
| CSL / seen family | accuracy | 0.400000 ± 0.028260 | 0.768333 ± 0.029404 |
| CSL / unseen family | accuracy | 0.059167 ± 0.009501 | 0.137500 ± 0.002946 |
| ZINC / seen family | MAE | 0.730009 ± 0.081641 | 0.753210 ± 0.156958 |
| ZINC / unseen family | MAE | 0.727205 ± 0.083463 | 0.749209 ± 0.155026 |

- `seen/unseen family`는 원본 그래프 종류가 아니라 **spanning-tree 생성 방식**이다.
  Fixed는 root-0 BFS, multi는 random-root BFS/DFS로 학습한다. 같은 test 그래프에
  fresh random-root BFS(seen)와 학습에서 제외한 Wilson UST(unseen)를 적용한다.
- CSL은 단일 고정 label-stratified 90/30/30 분할이다. 5개 모델 seed는 5-fold가 아니다.
  정확도는 chart별 정확도의 평균이며 여러 chart의 logits을 앙상블한 값이 아니다.
- CSL seen에서는 큰 개선이 있지만 unseen의 절대 성능은 낮다. unseen prediction flip rate도
  0.200000 → 0.319167로 증가하여 모든 chart 변화에 더 안정적이라고 주장할 수 없다.
- ZINC의 평균 MAE는 두 조건 모두 개선되지 않았다. Chart prediction std는 seen에서
  0.053970 → 0.041503, unseen에서 0.053139 → 0.041492로 줄었지만 이는 **한 그래프의
  chart 변경에 따른 흔들림**이지 model-seed 사이 성능 변동 감소나 MAE 개선이 아니다.
- `rounded_exact_vector_accuracy=0`은 정수 count용 일치 지표를 연속값 ZINC에도 노출한
  부적절한 보조 지표다. 코드가 반올림한 예측과 연속 정답의 완전 일치를 검사하므로,
  이를 회귀 학습 실패 또는 정확도 0%로 해석하지 않는다. 아직 코드에서는 제거하지 않았다.
- 현재 Tree 기본 구현은 고정 800 optimizer updates 후 모델을 평가하며 validation-best
  checkpoint 선택을 하지 않는다. 이 내부 fixed-vs-multi 실험을 표준 논문 표와 동일한
  재현이라고 제시하지 않는다. 표의 차이에 대한 paired 유의성 검정은 수행하지 않았다.

## 3. Conductance의 실제 GPU checkpoint 진단

사용자가 `ebf8cd1`을 pull한 뒤 아래 명령을 실행했고,
`Diagnostic status: passed (stdout only)` 출력을 제공했다.

```bash
bash scripts/diagnose_conductance.sh --run-id paper-20260830T150244764889Z --ablate-graph
```

대상은 **당시 기본값**인 model seed 0, Cora/PPI/ogbn-arxiv다. FP32 추론이며 AMP/TF32는 껐다.
이후 CLI 기본 데이터 목록에 CiteSeer/PubMed를 추가했지만 이 과거 로그에는 포함되지 않는다.
새로 계산한 지표는 train/validation뿐이고, test는 기존 저장값만 출력했다.
원래 validation과 재계산값의 차이는 약 `0~5e-8`이다. 소스 불일치 경고는 제공된 출력에 없다.
이 로그는 실제 GPU 추론 완료의 근거지만 전체 seed의 gate 상태나 GPU 가속의 근거는 아니다.

저장된 학습 설정은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005,
weight decay 0.0005, 최대 200 epochs, patience 50, PPI batch size 2다.

### 선택된 checkpoint의 성능과 학습 기록

| 데이터 | Best epoch / 실행 epoch | Eval train 지표 | Eval validation 지표 | Eval train / validation loss |
|---|---:|---:|---:|---:|
| Cora | 23 / 73 | accuracy 1.000000 | accuracy 0.658000 | CE 0.083452 / 1.163915 |
| PPI | 69 / 119 | micro-F1 0.496796 | micro-F1 0.487508 | BCE 0.541717 / 0.536941 |
| ogbn-arxiv | 199 / 200 | accuracy 0.495695 | accuracy 0.509279 | CE 1.879628 / 1.826401 |

Train-mode loss의 first → selected → last는 Cora `2.118194 → 0.720689 → 0.172052`,
PPI `0.699084 → 0.541241 → 0.541466`, arxiv `3.944757 → 2.252732 → 2.251006`이다.
진단의 train도 **dropout을 끈 eval 추론**이다. 학습 중 dropout을 켜고 기록한 loss와
시점·모드가 다르므로 두 값을 직접 비교해 checkpoint 오류라고 판단하지 않는다.

- Cora: 선택된 모델도 train 정확도 100%인 반면 validation은 65.8%다. 훈련 노드 적합은
  충분하며 큰 일반화 차이가 확인됐다. 단순 epoch 부족이라고 보기 어렵다.
- PPI: train에서도 F1이 낮아 현재 표현·정규화·최적화의 적합 부족을 의심한다. validation의
  예측 양성 비율은 0.202331, 실제 양성 비율은 0.294568이다. 제공된 micro-F1과 양성 비율에서
  계산한 precision은 약 0.598629, recall은 약 0.411182다. 원 로그가 직접 출력한 지표와
  이 후처리 계산값을 구분한다. 낮은 recall의 원인은 아직 확정하지 않았다.
- arxiv: 마지막 loss가 최저이고 best epoch가 199/200이라 추가 학습 여지가 있다.
  그러나 아래의 극도로 약한 이웃 전달 문제도 병존한다.
- Cora `23+50=73`, PPI `69+50=119`로 patience 설정과 종료 시점은 일치한다.

### 층별 엣지 가중치와 이웃 혼합량

층 번호는 코드의 `layer 0/1`을 따른다. Cora/arxiv 통계는 전체 transductive 그래프,
PPI 표는 **validation 그래프 두 개**의 node/edge-pooled 통계다. 차수 최대값은
그래프별로 계산한 뒤 통계를 모았다. `rho`는 0~1의 비율이며 퍼센트가 아니다.

| 데이터 / 층 | C 평균 | C의 CV | rho 중앙값 | rho < 0.01 노드 비율 | 전파 상대 변화량 |
|---|---:|---:|---:|---:|---:|
| Cora / 0 | 0.6984 | 6.623e-7 | 0.01696 | 17.91% | 0.02989 |
| Cora / 1 | 2.266 | 0.7224 | 0.02779 | 14.96% | 0.05120 |
| PPI / 0 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09871 |
| PPI / 1 | 0.6932 | 0 | 0.02653 | 23.03% | 0.09498 |
| arxiv / 0 | 0.6933 | 0 | 0.0004331 | 99.35% | 0.003041 |
| arxiv / 1 | 0.6933 | 0 | 0.0004330 | 99.35% | 0.002101 |

CV는 `std(C)/mean(C)`이고, 전파 상대 변화량은 LayerNorm/ELU 이전
`||H_after_conv - H_before_conv|| / ||H_before_conv||`이다. PPI train 그래프 20개에서도
두 층의 C CV는 0이며 rho 중앙값은 0.03308이다.

PPI/arxiv의 **관측한 FP32 eval 입력**에서 C가 상수인 현상이 확인됐다. 표시는 작은 양수도
지수 표기로 출력하므로 `CV=0`은 단순 표시 반올림 설명으로 지울 수 없다. C는 FP32로
chunk 재계산하고 분산은 float64로 집계했다. 전체-batch GEMM과의 bitwise 동일성이나
모든 가능한 입력에서 함수가 상수라는 결론까지 보장하지는 않는다.

Cora의 layer 1에는 비상수 C가 있으므로 구현이 항상 C를 상수로 강제한다는 설명은 틀린다.
또 진단은 학습 중 C 궤적을 기록한 것이 아니므로 학습 내내 C가 고정돼 있었다고 할 수 없다.

이 과거 checkpoint와 변경하지 않은 기본 benchmark의 전파식은 다음과 같다.

\[
H'=H-\frac{0.95}{d_{\max}^C}B^\top C B H,\qquad
\rho_i=0.95\frac{d_i^C}{d_{\max}^C}.
\]

`C=cI`이면 공통 c가 상쇄되어 `H'=H-(0.95/d_max)L_unweighted H`가 된다.
따라서 관측한 PPI/arxiv 입력에서는 적응적 엣지 가중치가 아니라 동일 가중치의 전파로
작동한다. arxiv rho 중앙값 `0.0004331`은 이웃 총가중치 **0.04331%**다.
이는 안정성용 스텝 선택의 효과이며 `L=B^T B`라는 라플라시안 정의 자체의 필수 조건이 아니다.

### 같은 checkpoint에서 전파만 우회한 validation 결과

| 데이터 / 지표 | 원래 값 | 전파 우회 값 | 우회 − 원래 |
|---|---:|---:|---:|
| Cora accuracy | 0.658000 | 0.632000 | -0.026000 = -2.6000%p |
| PPI micro-F1 | 0.487508 | 0.452144 | -0.035364 |
| arxiv accuracy | 0.509279 | 0.508876 | -0.000403 ≈ -0.04027%p |

전파 우회 값은 원 로그의 원래 지표와 delta를 합산했다. arxiv의 차이는 validation
29,799개 노드에서 정답 수 순감소 12개에 해당하며, 예측이 12개만 바뀌었다는 뜻은 아니다.
우회는 encoder/LN/ELU/decoder를 남기는 추론 개입이다. 별도 MLP를 재학습한 baseline이나
저성능 원인을 단독으로 증명하는 실험이 아니다. Cora/PPI에는 분명한 지표 하락이 있으므로
모든 데이터에서 그래프 연산을 전혀 사용하지 않는다고 말하면 안 된다.

## 4. 미확정 원인과 다음 검증

기존 checkpoint에서 관측된 상수 C와 약한 전달량은 사실이다. 이후 2×2 새 학습에서
node-degree 정규화가 개선을 이끈 결과를 확보했지만, 모든 데이터·seed에 대한 원인 설명이나
학습 C의 필요성까지 확정된 것은 아니다.
`softplus(0)+1e-5 ≈ 0.693157`이므로 PPI의 C 평균은 softplus 이전 gate raw logit이
0 부근인 상황과 맞는다.
이것만으로 모든 gate 파라미터가 0이라고 증명되지는 않는다. `softplus'(0)=0.5`이므로
이를 softplus의 출력 포화라고 설명하는 것도 부정확하다.

확장 audit에는 gate 입력·raw logit·gradient 검사가 있고, 2×2 학습에는 실제 train-mode
첫 batch의 task gradient와 decay 관찰값이 있다. Gate WD 제거 후 선택된 C의 변동이
커졌다는 결과와 WD 제거가 성능을 높인다는 주장은 분리한다.

다음은 [C-learning 전용 실행](../research/conductance_gat/c_learning/README.md)이다.
기존 `node_degree` checkpoint의 C를 평균으로 바꿔 현재 의존도를 검사하는 **재학습 없는
개입**과, 같은 정규화에서 `learned_c`/`fixed_c=1`을 처음부터 학습하는 **4개 새 학습**을
분리한다. 두 데이터와 seed 0을 유지하고 새 learned 점수도 같은 run에서 얻는다.
이 두 후속 분석은 구현 상태이며 GPU 결과를 아직 수령하지 않았다.

노드별 정규화는 기존 대칭성·보존성의 의미를 바꾸는 실험이므로 단순 속도 최적화나
버그 수정으로 부르지 않는다. 기존 기본 benchmark는 유지한다. 다른 model seed의
일반화, v2 학습 결과, GPU 가속 실측은 여전히 별도 검증 대상이다.

## 5. 근거와 검증 범위

| 근거 | 식별자 / SHA-256 |
|---|---|
| 사용자 제공 5-seed 집계 출력 | 첨부 `d4acb1eb-bd9d-40ef-af24-e5f7ba34f138`; `CEF76E8494C462E8302AF2811CCCD19BBB6D8DC8266DB852866237ED95DD5CEC` |
| 사용자 제공 GPU 진단 stdout | 첨부 `5db2e997-c8ab-495e-b762-c32fa620c02c`; `C0E89FC76A438D1707FE90C889923390FDF8277F05780B2811FF4D444DD01A21` |
| 사용자 제공 5e801c3 full-audit stdout | 첨부 `c4abbad1-654a-4f5e-a774-f84f7e88e4dd`; `CFA2118D4B9257CA8772FC16BE9834D1D0FB402FA375DDCB4E652D6FB37D564F` |
| 사용자 제공 43afd63 2×2 GPU 학습·비교 stdout | 첨부 `20b4a93d-06ed-4cff-9fe5-530eacf39766`; `2C78D02BB210BF00865AB7207DF651B02B2081EE4FAE6E8A6A83665A5D331161` |

이 hash는 **제공된 텍스트 파일의 hash**이며 서버의 checkpoint/원본 데이터 hash가 아니다.
개인 서버 계정·호스트 경로와 원본 로그 전체는 이 문서에 복제하지 않았다.

확장 검사 구현 전 문서 갱신의 로컬 회귀는 619 passed / 63 skipped (21.84 s, exit 0),
당시 진단 전용은 42 passed였다. Ruff/diff 및 당시 문서 로컬 링크 34개 검사도 통과했다.
최신 확장 검사 결과는 handoff의 최신 검증 항목을 따른다.
단일 seed·확장 진단 구현 후 전체 회귀는 **680 passed / 63 skipped**, 진단 전용은
**89 passed**다. 이는 당시 로컬 단위 검증이며, 이후 수령한 실제 GPU full-audit 로그는 위에
별도로 기록했다. 후속 2×2 구현 후 전체 검사는 **794 passed / 64 skipped** (31.83 s, exit 0),
Ruff 통과다. 기존 생략 사유에 Windows 실제 symlink 권한 1개가 추가됐으며 차단 로직은 별도
mock 검사로 확인했다. 이 수치는 2×2 코드 구현 당시의 로컬 검사이며, 이후 수령한
43afd63의 실제 GPU 재학습·성능 비교는 위에 별도 기록했다.
기존 게시 학습 코드 `a64c235`와의 진단 호환성도 메모리 로딩을 통한 42개 단위 검사로 확인했다.
생략된 63개는 Linux/Bash 전용 62개와 로컬 PyG 미설치 1개다. 이번에도 Windows faulthandler의
`access violation` 메시지가 있었으나 pytest는 위 결과와 exit 0까지 실행됐다.
이 호스트 경고를 Linux 성공의 근거로 해석하지 않는다. 로컬 단위 검사는 GPU 학습
또는 가속 배수의 검증을 대신하지 않는다. 현재 소스 스냅샷 checksum은
[hand_off.md](../hand_off.md)의 코드 스냅샷 항목을 따른다.
