# 네 검수 문서 이후 실제 보완 사항

**이 문서의 152개 검증 기록은 `4c2d7f4` 시점이다. 이후 확인된 fused shape/진단 오류와
최신 수정·검증은 [FUSED_REVIEW_FIXES_20260927.md](FUSED_REVIEW_FIXES_20260927.md)를 따른다.**

이 문서는 `FOUR_DOCUMENT_REVIEW_20260927.md`의 **수정 전** 반례 검토와 구분되는 구현 기록이다.
현재 소스가 기준이며, 이전 4b4df52/2602d90 검수 묶음으로 새 실험을 실행하지 않는다.

## 적용한 변경

| 요구 | 실제 구현 | 연구상 해석 |
| --- | --- | --- |
| R0-A | `IncidenceOperator.forward_with_state`가 message, C, 실제 전파 weight, beta를 반환. energy 모델의 객체를 캡처하던 hook 제거 | C 재계산 없이 같은 `omega * gate * C` 사용. deepcopy 시 다른 모델의 C를 참조하지 않음 |
| R0-B | 최초 test의 원래 노드 ID·예측 class·checkpoint/data/source/visibility를 `test_result.json`에 봉인. 재사용·표 생성 전 원본과 대조 | 정확도는 예측으로 재계산. loss는 최초 저장값의 해시 무결성만 확인. 외부 서명/악의적 전면 재작성 방지는 아님 |
| R0-C | immutable identity부터 configuration, metrics, audit, test까지 debug 출처 전달 | synthetic 검증은 별도 in-process scope. production CLI의 완료 승인기는 debug 결과 거부 |
| R1 | 별도 `aggregation_comparison.core` 실행기, F0/F1/S0/S1, 전체 matrix test barrier | complete supervised pass/validation 횟수 일치, optimizer step 수는 별도 보고. 동일 update budget 비교라고 주장하지 않음 |
| R1 샘플 통제 | `StudyInputs`가 매 pass의 실제 감독 ID coverage 및 부분 그래프 순서를 해시로 기록 | S0/S1이 동일 context/physical-batch 시퀀스를 사용하지 않으면 보고 거부. context 크기와 physical batch는 별개 |
| R2 | 선택 가능한 `--gram-implementation fused`, Gram/readout 결합 및 1차 gradient buffer 재사용 | node×head×pair 통계 생성을 생략. projected history는 아직 유지. reference 기본값 유지, BF16 및 전체 성능 별도 검증 필요 |
| R3 | 별도 `arxiv_node_year_views_v1` 규약, 검증된 node-year sidecar, view-local degree/context/topology/cache | train-only → train+val → 연도별 test. 공식 OGB 성적이나 독립 그래프 일반화로 부르지 않음 |
| R4 | 기존 21조건 유지. 명시적 sampled GCN/SAGE 옵션 추가 | 설치된 PyG의 기존 모델 사용. induced-context 국소 정규화, omega 보정 없음. GraphSAINT라고 부르지 않음 |

reference 8층/256 hidden/8 heads, large 12층/384 hidden/8 heads, C solver K=8, 전체
공식 split과 생산용 200 epoch를 줄이지 않았다. 모델/사전학습 가중치/데이터를 다운로드하지 않았다.
C는 입력 그래프가 아니라 입력 그래프와 특징에서 공유 학습 규칙으로 계산하는 conductance다.

## 핵심 실행 계약

새 실행기:

```text
python -B -m experiments.aggregation_comparison.core
  --run-id <새 ID> --profiles reference --model-seeds 0
  --device cuda:0 --hardware-profile portable
  --sample-context-seed-batch-size <명시적으로 선택한 연구 context seed 수>
  --sample-seed-batch-size <physical seed batch 탐색 시작값>
  --sample-context-workers <할당 CPU 안에서 사용할 수>
  --num-neighbors 15 10 --edge-chunk-size 4096
  --activation-checkpoint --min-free-gb 8 --evaluate-test
```

위 `<...>`는 설계 변수이며 임의의 작은 생산 기본값을 넣지 않았다. context size는 단순 VRAM
튜닝 변수가 아니라 연산자가 보는 그래프의 범위를 바꾼다. physical seed batch는 같은 context를
여러 개 disjoint-union으로 묶는 축이며 두 C 조건에 공통인 실제 측정으로 선택한다.
단일 full graph의 batch=1은 원본 그래프를 임의로 복제하지 않는 구조적 예외다.
할당 GPU/MIG 식별과 메모리는 서버에서 확인해야 하며 로컬 RTX 통과를 MIG 적합성으로 쓰지 않는다.

`--include-sampled-baselines`를 추가하면 full/sampled 각각 GCN·GraphSAGE를 함께 학습한다.
기존 GATv2 등을 포함한 21조건 실행은 기존 `python -m experiments.aggregation_comparison`
진입점을 계속 사용한다. global attention 조건을 sampled 경로로 자동 변환하지 않는다.

두 child 그룹은 각각 공통 calibration, 실제 학습, 기존 5회 validation audit를 끝낸다.
이후 상위 실행기가 네 핵심 조건의 노출량·공통 초기값·데이터·샘플 순서를 검사하고 **전부의
checkpoint를 동결한 다음에만** test를 허용한다. 중간 실패는 기존 결과를 보존한다.
공식 test에서는 공식 전체 graph 지원을 유지하며, 시간순 프로토콜에서는 해당 시점의 view만 쓴다.

complete pass 계약은 calibration의 비용 추정에도 적용된다. 기존 `reference_updates` 비용
모델을 그대로 이용해 잘못된 epoch/validation 횟수를 추정하지 않는다. child wall time에는
학습 checkpoint I/O가 포함되며 전체 invocation 시간에는 calibration/audit/I/O도 포함된다.
stage timing과 validation/audit 시간도 함께 보존한다. C solver 독립 시간 및 실제 arxiv profiler
결과는 아직 없으며 end-to-end 가속을 주장하지 않는다.

## 시간순 입력 준비와 제한

기존 arxiv cache에는 node_year가 없다. 이미 보관된 원본 OGB raw 폴더를 대상으로:

```text
python -B -m experiments.aggregation_comparison.temporal
  --data-root <기존 데이터 루트> --raw-dir <기존 ogbn_arxiv/raw 폴더>
```

`node-feat.csv.gz`, `node-label.csv.gz`, `edge.csv.gz`, `node-year.csv.gz`가 모두 필요하다.
특징·라벨·canonical edges가 기존 cache의 노드 순서와 정확히 일치해야 sidecar를 만든다.
기존 cache/sidecar는 덮어쓰지 않는다. 해당 원본이 없으면 명확히 실패하며 다운로드하거나
split에서 연도를 임의 복원하지 않는다. 생성 위치는 `<data-root>/ogbn-arxiv/node_year_v1.json`.

이후 core 또는 기존 비교 실행기에 `--visibility-protocol arxiv_node_year_views_v1`를 추가한다.
공식 transductive 결과와 새로운 run ID 및 결과 디렉터리로 분리한다.
학습된 파라미터는 test에서 갱신하지 않고 C만 각 view의 특징·구조에서 다시 계산한다.
test 노드는 자신의 연도에서 정확히 한 번 점수에 포함된다.

이는 **노드 연도에 따른 구조 가시성 실험**이다. 원본 엣지 생성 시각을 알 수 없고,
제공 특징이 시점별로 생성됐다는 보장도 없다. 독립된 미관측 그래프 실험이나 완전한
시간 인과성 실험으로 확대 해석하지 않는다.

## 검증 범위 및 남은 실측

최종 소스 회귀: **152 passed, 0 failed, 0 errors, 0 skipped**, 461.88초.
`results/revision-final-debug-20260927-01.xml`에 기록했다.
CUDA 모델·독립 수학·sampler·최초 test/재사용·시간순 통합 검사 118개와,
CPU에서 모델 계산을 하지 않는 metadata/실행 차단 검사 34개다.
라이브러리 deprecation 및 JUnit record_property 관련 warning은 2,654개로 남아 있다.
중간 R0 검사 45개와 새 경로 검사 31개는 최종 152개에 중복되므로 더하지 않는다.
정적 Ruff 검사 및 `git diff --check`도 통과했다.

core 통합 검사는 실제 reference 8층/256 hidden 모델로 두 visibility 규약 각각
F0/F1/S0/S1의 학습 → 저장 → 5회 감사 → 전체 freeze → test → 재사용을 수행했다.
patience=1인 명시적 4 epoch debug에서도 네 조건 모두 4 pass/동일 감독 노드 노출량을
완주하고, full 4 updates/sample 12 updates를 별도로 기록했음을 확인했다.
전파 value/output/beta 및 encoder/decoder 초기값의 해시도 네 조건에서 일치했다.
생산용 전체 arxiv calibration이나 본학습을 실행했다는 뜻은 아니다.

선택적 fused Gram의 독립 CUDA 합성 microbenchmark
(`results/revision-gram-profile-20260927-02.json`, RTX 5070 Ti, FP32,
history 8×4096×8×32, 16384 edges, chunk 512, warmup 2회+측정 5회):

| 경로 | forward+backward 중앙값 | peak allocated bytes |
| --- | ---: | ---: |
| reference | 31.950ms | 216,032,256 |
| fused | 51.225ms | 210,723,840 |

이 작업량에서 fused는 약 2.5% 적은 할당 메모리를 사용했지만 **더 느렸다**.
따라서 reference 기본값을 유지한다. kernel/전체 모델/arxiv/MIG 가속 근거로 사용하지 않는다.
최초 측정의 보조 스크립트가 마지막 reference 중간 tensor를 보유하던 부분을 제거한 뒤
별도 `-02.json`에 다시 기록했다. 최초 결과를 덮어쓰지 않았다.
FP64/FP32 대수 검산 및 BF16로 양자화된 입력 검산과 실제 BF16 autocast 전체 모델 검사를
구분했다. 연산 재배치가 BF16 bitwise 동등함을 보장하지 않으므로 동일 run에서 전환할 수 없다.

GPU 합성 회귀의 최종 결과와 묶음 manifest를 함께 확인한다. 합성 graph의 작고 명시적인
4 epoch 실행은 debug 통합 검사이며 생산 설정을 바꾸거나 연구 성능으로 보고하지 않는다.
실제 arxiv 본학습·공식 test·시간순 test·A100 MIG 10GB 자원 실측은 아직 수행하지 않았다.
matched-update 보조 연구, projected-history까지 제거하는 후속 융합, C solver 개별 profiler,
독립 그래프 일반화는 이 변경으로 완료했다고 주장하지 않는다.
