# Conductance 원인 분리 실험

기존 Conductance GAT의 낮은 성능을 **gate weight decay**와 **그래프 정규화**의
2×2 실험으로 분리한다. 기본 benchmark, Cycle PE, Tree Augmentation은 변경하지 않는다.
외부 비교 모델을 추가하는 실험도 아니다.

## 실행

저장소 README에 따라 GPU Conda 환경과 공식 데이터 캐시를 준비한 뒤 저장소 루트에서 실행한다.
기존 `bash scripts/prepare_data.sh`로 받은 데이터를 그대로 사용한다. 새 다운로드나 더미 데이터는 없다.

```bash
bash research/conductance_gat/ablation/reproduce.sh --run-id gat-factorial-seed0-v1
```

이 한 명령은 **PPI와 ogbn-arxiv × 4조건 × model seed 0 = 총 8개 학습**을 순서대로 실행한다.
5-seed 반복이 아니다. 각 조건은 별도 프로세스에서 처음부터 학습하므로 이전 조건의 모델·optimizer·
CUDA 메모리를 이어받지 않는다. 과거 baseline 점수를 대신 사용하는 것도 아니다.
GPU 전용이며 CPU fallback이나 자동 패키지 설치는 하지 않는다.

실패하면 그 시점에서 멈추고 완료된 조건·로그·부분 비교표를 보존한다. 같은 run ID를 덮어쓰거나
자동 resume하지 않는다. 재실행은 `gat-factorial-seed0-v2`처럼 새 이름을 쓴다.

## 고정 조건과 변경 조건

| 저장 폴더 이름 | 정규화 | Gate WD | Encoder/decoder/LayerNorm WD |
|---|---|---:|---:|
| `baseline` | 기존 graph-local 최대차수 | 0.0005 | 0.0005 |
| `gate_no_wd` | 기존 graph-local 최대차수 | 0 | 0.0005 |
| `node_degree` | 노드별 weighted degree | 0.0005 | 0.0005 |
| `node_degree_gate_no_wd` | 노드별 weighted degree | 0 | 0.0005 |

Gate는 `operators.*.estimator.*`의 weight와 bias 전체다. 다른 파라미터에는 기존 Adam의
coupled weight decay를 유지한다. AdamW로 바꾸는 실험이 아니다.

공통 설정은 hidden 64, 2층, dropout 0.5, Adam lr 0.005, 최대 200 epochs, patience 50,
PPI batch size 2, workers 0이다. arxiv는 기존과 동일한 full-batch다.
FP32이며 AMP·TF32·compile은 끈다. 매 조건의 초기 state hash와 데이터 캐시 hash가 같고,
공통 설정이 일치해야 유효한 비교로 표시한다. 같은 seed라도 CUDA scatter의 bitwise 동일성까지
보장하지는 않는다. 동일 early stopping 정책에도 실제 epoch·optimizer step 수는 달라질 수 있다.

현재 원인 탐색은 **train 정답으로 학습하고 validation으로 checkpoint를 선택**한다.
**Test는 평가하지 않는다.** 논문 최종 test 성능이나 여러 seed의 평균·유의성을 주장하지 않는다.

## 정규화 수식

\[
L_C=B^\top CB,\qquad d_i^C=\sum_j c_{ij}.
\]

- 기존: \(H'=H-0.95L_CH/\max_i d_i^C\). 최대값은 그래프별이다.
- 변경: \(H'=H-0.95(D_C)^\dagger L_CH\). 고립 노드의 출력은 그대로 둔다.

Gate 입력 `abs(BH)`, `(BH)^2`, softplus, encoder/decoder/LN/ELU/dropout은 동일하다.
분모를 detach하지 않는다. 변경 연산은 일반적으로 Euclidean 공간에서 비대칭이므로
원래 대칭 연산과 같은 수축·에너지 보장을 주장하지 않는다. 고정 C에서는 degree-weighted
평균을 보존하지만 일반적인 node 평균 보존은 달라질 수 있다.

두 정규화 모두 `C → kC`의 공통 스케일은 소거된다. C의 절대 크기 식별을 해결하는 실험이 아니라
**허브의 최대차수가 다른 노드의 전달량을 제한하는 효과**를 검사한다. 변경군의 비고립 노드
`rho=0.95`는 정의상 정해지는 값이다. rho 증가만으로 성공이라 판단하지 않고 validation·엣지 C의
변동·실제 상태 변화·학습 곡선을 함께 본다.

## 결과 확인

학습이 끝나면 4조건 비교표와 차이가 터미널에 자동 출력된다. 나중에 다시 확인하려면:

```bash
cat results/conductance_gat/ablations/gat-factorial-seed0-v1/comparison.md
```

비교표에는 best-checkpoint의 층별 **그래프 내부 C CV·rho·실제 Conv 변화량**의 그래프별
단순평균과 gate 파라미터 norm도 함께 표시된다. 그래프 간 평균 차이를 C CV로 섞지 않으며,
없는 관찰값은 0 대신 `—`로 표시한다.

| 경로 (run 폴더 아래) | 내용 |
|---|---|
| `manifest.json` | 조건, 상태, 명령, 고정 설정, 실행 소스 hash |
| `comparison.md` | 데이터셋별 4조건 점수·효과·미완료 목록 |
| `comparison.csv`, `comparison.json` | 동일 비교의 구조화된 결과 |
| `logs/` | GPU 사전 검사와 조건별 학습 로그 |
| `<dataset>/<condition>/best.pt` | 해당 조건의 validation-best checkpoint |
| `<dataset>/<condition>/history.json` | epoch별 train loss·validation·gate 관찰값 |
| `<dataset>/<condition>/metrics.json` | 지표, 초기/선택/종료 상태 관찰, hash |

비교표는 `gate_no_wd − baseline`, `node_degree − baseline`, 다른 요인 아래의 각각의 효과와
`both − gate_no_wd − node_degree + baseline` 상호작용을 데이터셋별로 계산한다.
PPI micro-F1과 arxiv accuracy를 섞어서 평균내지 않는다. 1-seed 표준편차·CI·p-value는 만들지 않는다.
파일 누락·실패·초기 상태/캐시/공통 설정 불일치가 있으면 완성된 비교로 표시하지 않는다.

Gradient는 실제 첫 training batch의 backward 이후, Adam step 이전에 읽는다.
추가 training forward나 loader iteration을 소비하지 않는다. PPI에서는 첫 batch의 관찰이며
전체 train gradient가 아니다. validation 관찰은 epoch 종료 후라 시점을 구분해야 한다.
Gate별 task-gradient norm과 `lambda*parameter` norm을 따로 기록하며 0분모 비율은 null이다.
Scalar moment는 관찰한 모든 엣지/노드에서 계산하고, 큰 배열의 quantile은 명시적인 결정적 표본이다.

새 checkpoint는 `conductance_factorial` 식별자와 normalization 정보를 갖는다. 기존
`diagnose_conductance.sh`에 넣어 원래 모델로 잘못 읽지 말고 이 실험의 내장 관찰 결과를 사용한다.

## 선택 옵션

데이터셋 하나만 먼저 실행할 수 있다. 선택한 데이터셋 안의 4조건은 항상 유지한다.

```bash
bash research/conductance_gat/ablation/reproduce.sh --datasets ppi --run-id ppi-factorial-seed0-v1
bash research/conductance_gat/ablation/reproduce.sh --datasets ogbn-arxiv --run-id arxiv-factorial-seed0-v1
```

`--epochs`, `--patience`, `--batch-size`, `--workers`는 네 조건 모두에 동일하게 적용된다.
다른 단일 seed는 `--model-seed 1`, 다른 디스크는 `--data-root`·`--results-root`로 지정한다.
`--dry-run`은 명령 목록만 출력하고 데이터/모델/결과를 만들지 않는 개발용 옵션이다.
단위 검증과 실제 GPU 성능 비교는 별개다. 새 4조건의 실측값은 서버 실행 후에만 얻어진다.
