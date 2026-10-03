# Experiment 4.2 — C로 계산한 노드별 정규화

이번 질문은 **같은 C 생성 규칙을 노드별 정규화와 함께 다시 학습하면,
기존 전체 그래프 κ 정규화보다 분류에 도움이 되는가**다.
학습한 C로 `D_C = diag(AᵀCA)`를 계산하고, 이차 연산자 양쪽에 `D_C^(-1/2)`를 적용한다.
기존 모델과 결과를 보존하며 새 독립 실험 폴더에서 학습한다.
서버 전체 210-run 학습·평가 완료 결과는 [서버 기록](../SERVER_NODE_NORMALIZATION_RESULTS.md)에 있다.

## 비교할 다섯 조건

| 이름 | C 생성 | 이차 메시지 |
| --- | --- | --- |
| `fixed_wedge` | 모든 wedge의 C=1 | `S_Q AᵀA S_Q Z / 3` |
| `learned_wedge_raw` | 원래 raw gate | `S_Q AᵀCA S_Q Z / (3κ)` |
| `learned_wedge_rms` | 원래 RMS gate | `S_Q AᵀCA S_Q Z / (3κ)` |
| `learned_wedge_local_raw` | 같은 raw gate | `S_C AᵀCA S_C Z / 3` |
| `learned_wedge_local_rms` | 같은 RMS gate | `S_C AᵀCA S_C Z / 3` |

`S_Q = diag(AᵀA)^(-1/2)`, `S_C = D_C^(-1/2)`다. Wedge 대각이 0인 노드의 역제곱근 값은 0이다.
**C=1이면 node와 global의 fixed 연산자가 같으므로 fixed 대조군은 하나다.**
원래 C 생성·초기화·dropout·일차 전파·α·β·분류 loss를 유지한다.
Node 조건은 C와 D_C, 양쪽 S_C를 모두 미분하며 분류 CE로 학습한다.
고정한 C에 대한 이차 연산자는 PSD이고 spectral norm 상한 1을 유지한다.
이는 입력에 따라 C가 변하는 전체 신경망의 Jacobian이나 학습 성공에 대한 보장은 아니다.

## 전체 학습 계약

| 항목 | Full |
| --- | --- |
| 데이터 | Cora·CiteSeer·PubMed, public fixed split, 전체 그래프 |
| 그래프 | 모든 물리 엣지·모든 unordered wedge, sampling ratio 1 |
| Backbone | 두 층 `입력 → hidden 64 → 클래스`, projection bias 없음 |
| Gate | 각 층 `4F → 64 → 1`, raw/RMS 생성 규칙 동일 |
| Activation/dropout | 첫 전파 뒤 ReLU, 각 projection 전 dropout 0.5 |
| 초기 계수 | α=0.5, β=0.25, 두 조건 모두 같은 학습식 |
| Optimizer | Adam, weight 행렬에 weight decay 0.0005 |
| 학습률 | 0.001·0.003·0.01, tuning seed 101·202·303 |
| 선택 | tuning의 평균 validation CE, 각 run의 validation-selected checkpoint |
| Final seed | 11·23·37·53·71, 모든 모델 선택을 잠근 뒤 test 평가 |
| 예산 | 조건마다 500 epoch, early stopping 없음 |
| 전체 범위 | tuning 135 + final 75 = **210 runs·105,000 updates** |
| 새 파라미터 | 정규화용 추가 학습 파라미터 없음 |

각 replica는 하나의 전체 citation 그래프에서 노드 분류를 하므로 graph batch는 1이다.
독립 모델은 seed 축으로 벡터화하며 모든 노드·엣지·wedge를 동시에 계산한다.
Tuning은 packed seed 1·2·3, final은 1·2·4·5 후보의 처리량과 peak VRAM을 측정해 선택한다.
그래프 전체를 GPU에 cache하므로 epoch마다 DataLoader로 그래프를 다시 읽지 않는다.

Public split을 이미 확인한 뒤의 후속 연구다.
같은 split의 성능은 이번 비교의 근거이며 새 split·새 그래프의 확인 실험을 대신하지 않는다.
원래 Experiment 4의 여덟 조건과 결과는 별도 대조 자료로 보존한다.

## 최종 checkpoint의 C 배치 검사

네 learned 조건에 대해 layer_0·layer_1·both를 모두 평가한다.

- **Native C=1/shuffle:** 해당 C로 node의 D_C·양쪽 S_C 또는 global의 κ를 다시 계산한다.
- **같은 norm의 C=1/shuffle:** 현재 층 원래 메시지와 전체 노드·channel norm을 seed별로 맞춘다.
- **이차 분기 제거:** 그 층의 βM만 제거한다.

Shuffle은 데이터별 공통 고정 permutation 10개를 재사용한다. 모든 wedge를 유지한다.
첫 층 개입 뒤 다음 층에서는 변경된 현재 Z로 참조 C·정규화를 다시 계산한다.
Norm match는 β가 같으므로 실제 βM의 norm도 맞춘다. 전체 update·상쇄·ReLU 결과까지 같아지지는 않는다.
원래 메시지와 후보 메시지 모두 0이면 gain=1이다.
후보가 0이고 참조가 양수면 norm match가 불가능하므로 정확한 오류를 출력하고 중단한다.
정의되지 않은 norm 비율·cosine을 0으로 채우지 않는다.
Norm을 맞추는 사후 개입에는 native 연산자의 spectral norm 상한을 그대로 주장하지 않는다.

Primary metric 225행, frozen intervention metric 12,420행, 층별 진단 8,430행을 저장한다.
Shuffle manifest는 seed 안에서 먼저 평균하고 독립 학습 반복 수로 세지 않는다.
정확도와 CE, C std, 실제 αL/Z·βM/Z를 함께 보고 C의 유익성을 판단한다.

## A6000 서버 실행

실제 할당받은 GPU 번호 또는 UUID를 입력한다. 같은 터미널에 epoch·평가 진행 상태가 나온다.
모델·그래프·seed·500 epoch를 유지하면서 physical seed packing과 path chunk를 실제 측정한다.
Chunk는 메모리에 함께 올리는 중간 tensor만 나누며 모든 wedge를 계산한다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
read -r -p "할당받은 GPU 번호 또는 UUID: " NODE_GPU &&
test -n "$NODE_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$NODE_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.wedge_propagation.node_normalization.study \
  --profile full --device cuda \
  --data-root /home/aicompetition07/new-gat/data/wedge-citation \
  --output-dir "results/wedge-node-normalization-$(date +%Y%m%d-%H%M%S)"
```

GPU가 여러 개 실제 할당되었다면 `0,1`처럼 입력해 독립 job을 분배할 수 있다.
기존 실험 폴더를 output으로 지정하지 않는다. DEBUG는 별도 fixture·예산의 연결 검사다.

## 결과 확인

```bash
NODE_RUN=$(ls -dt /home/aicompetition07/new-gat/results/wedge-node-normalization-[0-9]*/ | head -n 1) &&
test -n "$NODE_RUN" &&
cat "${NODE_RUN}completion.json" &&
cat "${NODE_RUN}NODE_NORMALIZATION_SUMMARY.md"
```

| 파일 | 확인할 내용 |
| --- | --- |
| `NODE_NORMALIZATION_SUMMARY.md` | 다섯 조건의 전체 split, paired 차이, 실제 분기, 같은 norm의 C 배치 진단 |
| `metrics.csv` | 모든 final seed의 train·validation·test 정확도와 CE |
| `paired_comparisons.csv` | node−global, learned−fixed, RMS−raw의 각 split 차이·std·95% t 구간 |
| `interventions.csv`, `intervention_changes.csv` | 모든 처리·세 위치·고정 permutation의 수치와 paired 변화 |
| `gate_diagnostics.csv`, `branch_strength_estimates.csv` | C·α·β·작동 κ·참조 global κ·실제 메시지 크기와 방향 |
| `jobs/*/pack-*/epoch_history.csv`, `learning_rate_selection.json`, `resources.csv` | 전체 학습 진행·validation 선택·실측 자원 |
| `coverage.json`, `contract.json`, `completion.json` | 전체 범위·실제 데이터·source 보존·완료 증거 |

수식은 [MODEL_MATH.md](MODEL_MATH.md), 구현·검사·미검증 범위는 [VERIFICATION.md](VERIFICATION.md)에 있다.
