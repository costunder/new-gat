# Conductance GAT 성능 진단

완료된 benchmark의 학습 기록과 `best.pt`를 읽고 **model seed 0 하나**를 검사한다.
GPU 추론과 선택적 train-label gradient 계산을 수행하지만 재학습, optimizer update,
파라미터 변경, 데이터 다운로드는 하지 않는다. 기존 checkpoint·학습 기록·캐시는 덮어쓰지 않는다.

## 실행

학습에 사용한 Conda 환경을 활성화하고 저장소 루트에서 실행한다. `RUN_ID`는 학습 종료 때
출력된 기존 실행 ID로 바꾼다. 새 실험 ID가 아니다.

```bash
bash scripts/diagnose_conductance.sh --run-id RUN_ID --full-audit
```

기본 대상은 **seed 0의 Cora, CiteSeer, PubMed, PPI, ogbn-arxiv**다. 5개 모델 seed를
반복하지 않는다. `--model-seed 1`을 명시하면 seed 1 하나로 바뀔 뿐 seed 0과 함께 돌지 않는다.
`--full-audit`는 아래 C 개입과 gate/gradient 검사를 모두 포함한다.

사용자가 이미 완료한 Conductance run은 다음처럼 검사한다.

```bash
bash scripts/diagnose_conductance.sh --run-id paper-20260830T150244764889Z --full-audit
```

한 데이터셋만 먼저 검사하려면 `--datasets cora`를 추가한다. 이는 모델 seed·데이터셋 선택이지
공식 그래프를 축소하거나 더미 데이터로 바꾸는 옵션이 아니다.

기존의 기본 진단과 단순 graph-off 비교만 실행하는 명령도 유지한다.

```bash
bash scripts/diagnose_conductance.sh --run-id RUN_ID --datasets cora ppi --ablate-graph
```

이 옵션은 **validation에서만**, 동일 checkpoint의 conductance 전파를 항등 연산으로
우회한다. encoder, LayerNorm, ELU, decoder는 남긴다. 별도의 MLP를 학습하는 비교가 아니며,
학습 때 없던 입력 분포를 만들 수 있으므로 성능 차이를 원인 확정으로 해석하지 않는다.

전체/확장 진단은 새 폴더를 자동 생성하고 종료 시 위치를 출력한다.

```text
runs/diagnostics/conductance-RUN_ID-model-seed-0-TIMESTAMP/
├── report.md       사람이 읽는 지표·개입·gradient 요약
└── report.json     분포·파라미터·checksum을 포함한 전체 기록
```

직접 위치를 지정하려면 `--output-dir runs/diagnostics/conductance-seed0-v2`를 추가한다.
지정한 폴더는 새 경로여야 하며 기존 데이터·학습 결과 아래 경로는 거부한다.
확장 옵션 없는 기본 진단은 기존처럼 stdout만 출력하고 `--output-dir`을 선택할 수 있다.

학습 때 사용자 지정 경로를 썼다면 `--results-root`와 필요할 경우 `--data-root`로
실제 경로를 지정한다. 기본 결과 위치는 다음 구조다.

```text
research/conductance_gat/results/paper/RUN_ID/model-seed-0/benchmark/
```

다른 GPU를 할당받았다면 `--device cuda:0`처럼 **현재 프로세스에 보이는 GPU 번호**를
지정한다. CPU 추론 fallback이나 자동 의존성 설치는 없다. CLI 도움말은 다음과 같다.

```bash
bash scripts/diagnose_conductance.sh --help
```

## 확인하는 내용

| 진단 | 의미 |
|---|---|
| 학습 기록과 best epoch | 학습 손실의 감소·정체, validation 선택 시점, 최대 epoch까지 학습했는지 확인 |
| checkpoint의 train / validation 성능 | dropout을 끈 동일 모델에서 성능 차이를 확인. 당시 dropout을 켜고 기록한 train loss와 수치가 같을 필요는 없음 |
| 층별 가중 차수와 `rho` 분포 | 그래프 최대 가중 차수 대비 이웃 정보가 얼마나 섞이는지 확인 |
| 층별 conductance 분포·변동계수 | 엣지별 C가 거의 상수인지, 상대 가중치 차이가 생겼는지 확인 |
| 전파 전후 상대 변화량 | LayerNorm·ELU 이전 conductance 연산 자체가 표현을 얼마나 바꾸는지 확인 |
| PPI 정답 / 예측 양성 비율 | 낮은 micro-F1이 과소·과다 양성 예측과 함께 나타나는지 확인 |
| 선택적 validation 전파 우회 | 기존 checkpoint가 이웃 전파에 얼마나 민감한지 확인 |

## C 개입: 같은 checkpoint의 validation 비교

`--full-audit` 또는 `--interventions`로 실행한다.

| 이름 | 동작 |
|---|---|
| `learned_C` | 변경 없는 기존 모델 |
| `mean_C` | 각 그래프·층의 C를 해당 평균 상수로 치환 |
| `shuffled_C` | 각 그래프·층 내부에서 C를 고정 permutation으로 재배치 |
| `graph_off` | conductance 전파만 identity로 우회; encoder/LN/ELU/decoder는 유지 |

각 개입은 모든 층 동시 적용과 한 층씩 적용을 분리한다. 현재 2층 모델에서는 원본을 포함해
10개 validation 조건이며, **10개 모델을 재학습하는 것이 아니다.** 빠르게 전체 층 비교만
하려면 `--no-layerwise-interventions`로 4개 조건만 검사할 수 있다.

C를 변경하면 weighted degree와 `0.95/dmax`도 다시 계산한다. 따라서 shuffle은 엣지 배치뿐
아니라 `rho`도 바꿀 수 있다. 각 조건의 C/CV·rho·전파량, loss·metric·logit 변화·prediction
flip을 함께 저장하며 차이를 순수한 attention 기여나 학습 원인으로 확정하지 않는다.
Shuffle은 model seed 반복과 별개인 `--shuffle-seed 0` **하나**를 사용한다. graph-off의
통계상 effective C=0은 전파 우회를 표현할 뿐 학습 gate가 0이라는 뜻이 아니며 CV는 null이다.

## Gate 파라미터·입력·gradient 검사

`--full-audit` 또는 `--gate-audit`로 실행한다. 기본은 **eval mode + autograd ON**이다.
Dropout을 끄고 기존 checkpoint에서 train loss를 미분하며 optimizer는 생성하지 않는다.

- Cora/CiteSeer/PubMed/arxiv: 기존 full graph를 forward하되 train mask의 정답만 사용.
- PPI: 공식 train split 순서에서 **첫 1개 batch**만 사용. batch size는 저장된 학습 설정
  (현재 기본 2개 그래프)을 따르고, 실제 그래프 ID·라벨 수·loss reduction을 기록.
- 각 gate의 `abs(BH)`, `(BH)^2`, Linear/SiLU 중간 값, raw logit, C와 raw-logit gradient.
- gate뿐 아니라 encoder/decoder/LN의 파라미터 norm, 0/near-zero 비율, task-gradient norm,
  `lambda * parameter` norm, 비율·cosine. 영벡터의 cosine은 null로 기록.

`--gradient-mode train`은 고정 RNG의 dropout-on 국소 검사다. 학습 재개가 아니며 eval 검사와
구분해서 해석한다. 추가 PPI batch가 필요할 때만 `--gradient-batches`를 명시한다.
PPI 여러 batch를 선택하면 label 수로 가중한 합산 objective이며 과거 optimizer step 재현이 아니다.

모든 원소의 mean/std/norm은 float64 블록 집계로 계산한다. 큰 activation의 quantile은
기본 최대 4,096개 위치에서 얻은 **결정적 표본 quantile**이며 전체 정확 quantile로 표기하지 않는다.
`--gradient-sample-limit`과 `--near-zero-threshold`는 보고서에 기록된다. 실제 forward/autograd
경로를 잘라서 일부 edge의 gradient를 전체 gradient라고 표시하지 않는다.

이는 현재 checkpoint의 국소 task-gradient와 L2 항 비교다. Adam moment·과거 update·붕괴
발생 과정을 복원하지 않으므로 WD가 원인이라는 확정 실험이 아니다. `dmax.detach()`, WD 변경,
다른 normalization 및 재학습 baseline은 이번 진단 기능에 포함되지 않는다.

현재 연산의 노드별 총 이웃 가중치는 다음과 같다.

\[
\rho_i=0.95\frac{d_i^C}{\max_j d_j^C},\qquad d_i^C=\sum_j c_{ij}.
\]

최대값은 **각 그래프별**이다. PPI의 여러 그래프를 합쳐 하나의 최대값을 쓰면 안 된다.
`rho`가 대부분 작으면 약한 이웃 전달 가설을 지지한다. 그러나 이 값만으로 낮은 정확도의
원인을 확정할 수는 없다. C 전체의 공통 크기는 스텝 크기에서 상쇄되므로, C 평균의 크기보다
상대 분포·`rho`·실제 전파 변화량을 함께 본다. C가 거의 상수여도 그래프 전파 자체가 사라지는
것은 아니다. 1%/5%/10% 같은 분포 구간은 설명용이며 검증된 합격·실패 기준이 아니다.

## 평가와 재현 경계

- 진단은 FP32이며 AMP/TF32를 끈다. 원래 run이 AMP/TF32를 사용했다면 저장된 validation과
  차이가 날 수 있다. 원래 저장값과 재계산값의 차이를 함께 확인한다.
- 새로 계산하는 성능은 train과 validation뿐이다. test 점수는 저장된 결과만 표시하며
  test 기준으로 새 ablation·튜닝을 하지 않는다.
- Cora/CiteSeer/PubMed/arxiv는 기존 학습처럼 전체 그래프 특징을 사용하는 transductive
  forward다. test 노드의 특징과 연결까지 제거하는 것이 아니라 test label로 평가하지 않는 것이다.
- PPI는 공식 train / validation 그래프를 모두 읽는다. micro-F1은 그래프별 F1의 평균이 아니라
  전체 노드·라벨의 TP/예측 양성/정답 양성 수를 합쳐 계산한다.
- 추론 당시 소스와 학습 당시 소스 checksum이 다르면 경고를 확인한다. state_dict가 로드된다고
  전파 수식까지 동일하다는 보장은 없다. 소스 차이를 확인한 뒤 수치를 해석한다.
- 확장 검사에서 validation 재계산이 저장값과 `1e-4`보다 크게 다르면 개입 전에 중단한다.
  개입의 변경 없는 reference도 기본 재검사의 metric/loss와 대조한다.
- 종료 후 파라미터·기존 gradient·module mode·hooks·RNG를 복원하고 checkpoint hash를 확인한다.
- Gradient 검사는 실제 backward 메모리가 필요하다. arxiv를 축소하거나 CPU로 우회하지 않는다.
  OOM이면 실패로 기록하고 이미 완료된 데이터의 진단을 보고서에 남긴다. `--edge-chunk-size`는
  보조 통계/치환 경로의 chunk이지 원래 모델의 backward 메모리 상한이 아니다.
- 이 진단은 모델의 학습 속도나 GPU 가속 배수를 측정하지 않는다. 새로운 seed의 학습 결과나
  연구 가설의 최종 검증으로 집계하지 않는다.

## 수령한 실제 진단 결과

2026-08-31 사용자 제공 로그에서 기존 run `paper-20260830T150244764889Z`의 seed 0,
Cora/PPI/ogbn-arxiv 진단이 `passed`로 종료됨을 확인했다. PPI/arxiv의 관측한 FP32 eval
입력에서 두 층 C의 CV는 0이었고, arxiv의 이웃 혼합량 중앙값은 약 0.0433%였다.
Cora의 두 번째 층은 비상수 C이며, 모든 데이터에서 전파가 무효인 것은 아니다.

정확한 train/validation 지표, 층별 통계, validation 전파 우회 차이와 해석 한계는
[실험 상태](EXPERIMENT_STATUS.md)에 있다. 원인이 weight decay라는 설명은 아직 가설이다.
이 결과는 전체 seed 진단·모델 수정·재학습 또는 최적화 가속 측정이 아니다.
새 mean/shuffle·gradient 검사 결과는 아직 실제 서버에서 받지 않았다. 위 명령은 그 추가
검사를 실행하는 코드이며 기존에 제공된 seed 0 로그에 없던 측정값을 만들어 적지 않는다.
