# 주 벤치마크: ogbn-arxiv × GCN / GraphSAGE / GATv2 / incidence

최신 보완: [REVIEW_REMEDIATION_20260927.md](REVIEW_REMEDIATION_20260927.md).
기존 21조건 비교와 별도로 4조건 core controller, 원본 test 봉인, 시간순 custom protocol,
선택적 sampled GCN/SAGE가 추가됐다. 아래 공식 transductive 규약과 시간순 규약을 섞지 않는다.

PPI는 새 벤치마크에서 제외한다. 새 aggregation 실행기는 ogbn-arxiv만 허용하며,
PPI-only sampled_inductive의 실행 진입점은 중단된다. 이전 결과/소스의 재현 자료는
보존하지만 새 실험의 데이터나 성능 근거로 가져오지 않는다.

## 데이터와 평가 규약

공식 [OGB node prediction 규약](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)에
따른 ogbn-arxiv: 논문 169,343개, 원본 directed citation 1,166,243개,
128차원 입력, 40개 클래스. 공식 시간순 train/validation/test 분할을 사용한다.
2017년까지 train, 2018년 validation, 2019년 이후 test다.
저장소의 검증된 캐시는 train 90,941 / validation 29,799 / test 48,603개를 요구한다.

모든 모델은 같은 캐시·특징·label·mask와 원래 연결을 공유한다. 기존 공통 전처리에
따라 citation을 undirected physical support로 정규화하고, 메시지 연산에서는
양방향으로 사용한다. 원본 directed edge 수와 정규화 후 physical edge 수는
같은 숫자로 가정하지 않고 실행 로그에 실제 수를 기록한다.

학습은 train mask의 cross-entropy, 선택은 validation Accuracy, 최종 평가는
공식 test Accuracy다. micro-F1/BCE를 사용하지 않는다. 공식 transductive 평가이며
독립된 새 그래프 일반화 실험으로 부르지 않는다.

## 모델: 사전학습 가중치 없이 처음부터 학습

모든 모델은 무작위 초기화에서 시작한다. 모델/가중치 다운로드나 외부 checkpoint
가져오기가 없다. 설치된 PyTorch Geometric의 연산 구현을 사용한다.

| 이름 | 실제 구현 및 연산 |
| --- | --- |
| gcn | GCNConv. self-loop 포함 symmetric normalized adjacency. 정규화는 입력 그래프별 cache를 공유한다. |
| graphsage | SAGEConv(mean). 이웃 평균의 학습 변환 + 별도의 root 변환. project=False, normalize=False. |
| gatv2 | GATv2Conv. 별도 source/target 변환, self-loop, residual=False. |
| incidence 계열 | 기존 공유 C 규칙과 weighted Laplacian, 동일 backbone의 16개 내부 대조군. |

공식 연산 정의:
[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html),
[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html),
[GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html).

reference 8 message-passing layers/256 hidden, large 12/384를 유지한다.
attention/incidence는 8 heads이며 GCN/GraphSAGE는 multi-head 모델이 아니다.
공통 linear encoder/decoder, ReLU, feature dropout, loss, optimizer 및 학습 예산을
적용한다. GCN/GraphSAGE에도 외부 residual/FFN/LayerNorm을 추가하지 않는다.
GraphSAGE의 root 변환은 그 모델의 정의이며 외부 wrapper skip을 추가한 것이 아니다.

기존 16개 내부 조건과 DUALFormer 두 조건을 줄이지 않았다. GCN/GraphSAGE 추가 후
전체 기본 matrix는 **21조건**이다. DUALFormer는 자체 normalization과 residual을
가진 외부 비교 모델이며 no-skip 변형과 따로 표기한다.

이는 공통 구조·레시피에서의 비교다. 파라미터 수 일치, 각 논문의 최적 hyperparameter
재현, SOTA 달성을 뜻하지 않는다. baseline을 약하게 만들기 위한 별도 축소도 하지 않는다.

## 선택·최종 test 분리

기존 validation 학습·감사 완료 후 `--evaluate-test`로 최종 평가를 수행한다.

1. 요청된 dataset/profile/seed/arm의 정확한 전체 matrix를 검사한다.
2. 모든 cell의 디스크 학습 증거, audit, source/checkpoint/log hash를 확인한다.
3. 하나라도 미완료/손상이면 test 데이터 경로에 들어가기 전에 거부한다.
4. 전체 best checkpoint hash를 manifest에 먼저 고정한다.
5. 새 모델에 저장 best를 읽고 선택 당시 validation count를 재현한다.
6. 전체 그래프를 사용하되 공식 test mask의 노드만 평가한다. 가중치는 갱신하지 않는다.
7. 모델 state·checkpoint·source 불변성을 확인하고 test 결과를 저장한다.

test 결과로 epoch/model을 재선택하지 않는다. 완료된 test 재실행은 저장 증거를
검사하고 inference를 반복하지 않는다. 결과는 manifest의 official_test와
comparison.md에 validation과 구분해 표시한다. 여러 seed면 평균과 sample std,
한 seed면 std unavailable로 기록한다.

## 자원과 실행

기본 hardware profile은 사용자 장치에 맞춰 portable이다. 이 profile 선택이
A100 MIG 10GB 적합성 실측을 대신하지 않는다. 실제 calibration에서 full graph,
backward/optimizer, validation 및 기전 감사까지 측정한다. 부족하면 원래 규모를
유지한 채 실패를 보고하며 PPI/작은 모델/CPU로 자동 전환하지 않는다.

GCN/GraphSAGE/GATv2는 현재 이 공통 비교에서 full-graph 학습을 사용한다.
GCN은 static normalization을 cache하지만 PyG message tensor 자체를 edge_chunk_size로
나누는 구현은 아니다. GATv2도 마찬가지다. fit 여부는 실제 측정으로 결정한다.

검증된 ogbn-arxiv 캐시가 있는 서버에서 사용할 명령이다. 다음 dry-run은 모델이나
데이터를 다운로드하지 않고 본학습도 하지 않는다.

```bash
cd /home/aicompetition07/new-gat &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.aggregation_comparison \
  --run-id arxiv-comparison-mig10gb-seed0-v1 \
  --datasets ogbn-arxiv --profiles reference --model-seeds 0 \
  --hardware-profile portable --device cuda:0 \
  --edge-chunk-size 4096 --activation-checkpoint --min-free-gb 8 \
  --evaluate-test --dry-run
```

`--dry-run`을 제거하면 calibration 후 전체 21조건과 최종 평가를 실행한다.
calibration만 필요하면 `--dry-run` 대신 `--calibration-only`를 사용한다.
데이터 캐시가 없으면 자동 다운로드하지 않고 오류를 낸다. 이미 사용한 run ID나
다른 source/configuration의 checkpoint를 이 변경에 이어 붙이지 않는다.
여러 seed는 명시적 --model-seeds 목록으로 지정하며, 기본 seed 0만의 결과는
통계적 우위의 근거로 확대하지 않는다.

## 이번 로컬 검증 범위

새 검사는 GCN/GraphSAGE의 dense 기준식 대비 출력·gradient, 그래프 cache 교체,
GCN/GraphSAGE/GATv2/incidence의 실제 CUDA 학습·저장·validation audit·test mask,
test 전 전체 matrix 잠금 및 완료 test 재실행 경로를 확인한다.
수학/실행 검사는 명시적 합성 debug다. 실제 arxiv 본학습·공식 성능·MIG 적합성은
로컬에서 실행하지 않았다. 증거는 별도 JUnit에 기록한다.

RTX 5070 Ti 16GB에서 `results/arxiv-baselines-debug-20260927-01.xml`의 105개 검사가
통과했다. 21개 모델의 FP32/BF16 학습·optimizer 재개 검사와 새 baseline/test
파이프라인이 포함된다. 최종 CLI/기록/test 재사용 검사를 보강한 뒤 영향 경로
13개를 `results/arxiv-baselines-debug-20260927-02.xml`로 재검사해 통과했다.
두 실행 모두 failure/error/skip 0이며, 반복 검사를 더해 118개라고 세지 않는다.
Torch/PyG의 upstream deprecation 경고는 있었다. Ruff와 Git whitespace 검사도
통과했다. 현재 명령의 21조건 arxiv dry-run 기록은
`results/arxiv-baselines-dry-run-20260927.txt`다.
