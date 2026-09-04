# Node-degree 정규화에서 C 학습의 기여

앞선 [2×2 GPU 실험](CONDUCTANCE_FACTORIAL_FINDINGS.md)에서는
`node_degree + gate WD 0.0005`가 PPI/arxiv 모두 최고였다. 하지만 arxiv의 C가 거의
상수인 상태에서도 점수가 높았으므로 **정규화의 효과와 C 학습의 효과를 분리**한다.
기존 benchmark, 2×2 결과, Cycle PE와 Tree Augmentation은 변경하거나 합치지 않는다.

`gat-c-learning-seed0-v1`의 네 학습은 완료됐다. PPI learned/fixed는
52.564966% / 52.705738%, arxiv는 68.317723% / 68.324435%로,
이 seed의 validation에서는 학습 C의 이득을 확인하지 못했다.
정확한 값·파라미터 수·진단·근거의 한계는
[C-learning 결과](CONDUCTANCE_C_LEARNING_FINDINGS.md)에 기록했다.

**새 C-learning run의 learned checkpoint 검사도 완료 보고서를 수령했다.**
PPI의 전체 층 평균-C 교체는 -6.649440 pp, arxiv는 -0.033558 pp였다.
선택된 checkpoint의 의존성과 fresh training의 이득은 별개다.
`research/conductance_gat/c_learning/`의 코드는 기존 MLP 실험과 검사 재현용으로 유지한다. 다음 직접 엣지 C 학습은
[v2](../gpt_handoff/CONDUCTANCE_V2.md), 공유 상대-C 생성기 학습은 [v3](../gpt_handoff/CONDUCTANCE_V3.md)의 별도 명령으로
실행하며 과거 결과에 덮어쓰지 않는다. 두 버전은 서로도 별도 가설이다.

## 준비

[시작 안내](GETTING_STARTED.md)에 따라 Linux/NVIDIA GPU와 Conda 환경을 준비한다.
기존 `bash scripts/prepare_data.sh`의 PPI·ogbn-arxiv 공식 캐시를 그대로 사용한다.
학습·검사 중 다운로드, 더미 데이터 생성, 자동 패키지 교체나 CPU fallback은 없다.
모든 명령은 저장소 루트에서 실행한다.

```bash
git pull --ff-only
```

## 1. 완료된 learned checkpoint 검사를 다시 재현할 때

이미 `gat-c-learning-seed0-v1`을 완료한 서버에서 실행한다.

```bash
bash research/conductance_gat/c_learning/audit.sh --source-run results/conductance_gat/c_learning/gat-c-learning-seed0-v1 --output-dir results/conductance_gat/c_learning_audits/gat-c-learning-seed0-v1
cat results/conductance_gat/c_learning_audits/gat-c-learning-seed0-v1/report.md
```

이 명령은 source run의 **PPI/arxiv `learned_c` best checkpoint**를 사용한다.
원 validation을 재현한 뒤, 각 그래프와 각 층에서 C를 해당 평균으로 바꿔 추론한다.
전체 층 교체와 한 층씩 교체를 따로 기록하며 **교체한 C로 node degree를 다시 계산**한다.
원래 모델의 다른 가중치는 고정한다. 재학습·optimizer step·test 평가는 하지 않는다.

원본 checkpoint·결과·캐시는 수정하지 않는다. 별도 보고서 위치를 터미널에 출력하고,
위 명령은 `results/conductance_gat/c_learning_audits/gat-c-learning-seed0-v1/`에 저장한다.
그 폴더의 `report.md`는 읽기용, `audit.json`은 전체 검사 결과다.
기존 경로는 덮어쓰지 않으므로 재검사는 `--output-dir`의 끝을 `gat-c-learning-seed0-v2`처럼
바꾼다. `--output-dir`을 생략하면 `<source-run-name>-<UTC timestamp>`를 자동으로 붙인다.

Checkpoint/소스/cache protocol 무결성과 원 validation 재현이 맞지 않으면 실패로 처리한다.
실패하거나 누락된 조건의 점수 차이를 성공한 대비로 제시하지 않는다.
기존 global-max용 `scripts/diagnose_conductance.sh`로 node-degree checkpoint를 읽지 않는다.

해석: C를 평균으로 바꿨을 때의 하락은 **현재 선택된 checkpoint가 엣지별 차이에 의존하는 정도**다.
별도 fixed-C 모델을 학습한 성능이 아니며, 학습 중 gate의 역할까지 제거한 실험도 아니다.

### 과거 2×2 checkpoint도 별도로 검사할 때

검사기는 manifest의 suite를 읽어 다음 두 종류만 엄격히 구분한다.

| Source suite | 선택할 condition | 의미 |
|---|---|---|
| `conductance_c_learning` | `learned_c` | 이번 새 학습의 learned checkpoint |
| `conductance_factorial` | `node_degree` | 이전 2×2의 node-degree checkpoint |

이전 결과를 다시 보려는 경우에만 다음 명령을 쓴다. 이번 learned/fixed 비교의 후속 검사와
과거 checkpoint 결과를 합치지 않는다.

```bash
bash research/conductance_gat/c_learning/audit.sh --source-run results/conductance_gat/ablations/gat-factorial-seed0-v1
```

## 2. 비교 학습 자체를 새로 재현할 때만 실행

이미 완료한 `gat-c-learning-seed0-v1`의 다음 검사에 이 명령은 필요 없다.
별도 새 학습을 원할 때는 기존 run을 덮어쓰지 않도록 새 이름을 쓴다.

```bash
bash research/conductance_gat/c_learning/reproduce.sh --run-id gat-c-learning-seed0-v2
```

**PPI/arxiv × 2조건 × model seed 0 = 총 4개 학습**을 순서대로 실행한다. 5-seed가 아니다.
앞선 2×2에서 얻은 learned 점수나 checkpoint를 가져와 재사용하지 않는다.
두 조건 모두 새로 초기화하고 같은 새 run 안에서 비교한다.

| 조건 | 물리 엣지의 C | Gate 학습 | 정규화 | 나머지 WD |
|---|---|---|---|---:|
| `learned_c` | 기존 positive C estimator | Adam WD 0.0005 | node-degree | 0.0005 |
| `fixed_c` | 정확히 1 | 하지 않음 | node-degree | 0.0005 |

공통 설정은 hidden 64, conductance 2층, dropout 0.5, Adam lr 0.005,
최대 200 epochs, patience 50, PPI batch size 2다. PPI DataLoader는 기본 workers 4,
prefetch factor 2, persistent workers와 pinned/non-blocking transfer를 사용한다.
arxiv는 DataLoader가 없는 full-batch라 workers 0이고 FP32/AMP OFF/TF32 OFF/compile OFF다.
비교 가능한 backbone 초기 상태·캐시·공통 학습 설정을 검증한다. 동일한 early stopping
정책이어도 best epoch와 실제 실행 epoch는 달라질 수 있고 결과에 각각 기록한다.

Train 정답으로 loss를 계산하고 validation으로 checkpoint를 선택한다. **Test는 평가하지 않는다.**
PPI는 global micro-F1, arxiv는 accuracy다. 둘을 합산한 단일 점수를 만들지 않는다.
고정 C는 우리 모델의 내부 대조군이며 외부 GCN/GAT 재구현이 아니다. Gate를 학습하지 않는
조건은 활성 파라미터 수가 다르므로 동일 parameter-budget 비교라고 주장하지 않는다.
Fixed 조건도 동일 초기 state와 RNG를 맞추기 위한 gate scaffold는 보존하지만, 이를 평가하거나
optimizer에 넣지 않는다. 활성/동결/전체 파라미터 수를 구분하며 동결 gate norm을 학습 성과로 읽지 않는다.

## 결과 확인

학습이 끝나면 비교표가 터미널에 자동 출력된다. 이미 완료한 첫 run은 다음으로 확인한다.

```bash
cat results/conductance_gat/c_learning/gat-c-learning-seed0-v1/comparison.md
```

| 경로 (학습 run 폴더 아래) | 내용 |
|---|---|
| `manifest.json` | 실행 상태, source hash, 조건, 명령과 설정 |
| `comparison.md` | 데이터별 learned/fixed validation 점수와 차이 |
| `comparison.csv`, `comparison.json` | 구조화된 비교와 무결성 검사 결과 |
| `logs/` | GPU 사전 검사와 조건별 학습 로그 |
| `<dataset>/<condition>/best.pt` | 조건별 validation-best checkpoint |
| `<dataset>/<condition>/history.json` | 학습·validation 이력과 관찰값 |
| `<dataset>/<condition>/metrics.json` | 설정, 지표, 파일 hash와 진단 |

실패하면 그 시점에서 중단하고 이미 완료된 결과를 보존한다. 같은 run ID의 덮어쓰기나
자동 resume는 하지 않는다. 재실행은 `gat-c-learning-seed0-v2`처럼 새 이름을 쓴다.
다른 데이터/결과 디스크는 `--data-root`/`--results-root`로 지정하며 데이터 준비 때와 같아야 한다.

## 결론을 내리는 범위

평균-C 검사와 fresh-training 비교는 별도 보고서로 읽는다. 첫 번째는 현 checkpoint의
민감도, 두 번째는 해당 학습 예산에서 adaptive C를 학습한 모델과 고정 모델의 차이다.
둘이 다르면 오류라고 단정하지 말고 학습 경로·층별 의존성·선택 epoch를 확인한다.

Model seed는 하나이므로 표준편차·CI·p-value나 SOTA를 주장하지 않는다.
반복적인 validation 분석으로 선택 편향이 생길 수 있으며 아직 보지 않은 test 성능도 아니다.
학습 비교와 `conductance_c_learning/learned_c` 평균-C 검사 모두 GPU 보고서를 수령했으며,
위 결과 문서에 별도로 보존했다. 로컬에서 서버의 전체 원본 artifact를 독립 검증한 것은 아니다.
