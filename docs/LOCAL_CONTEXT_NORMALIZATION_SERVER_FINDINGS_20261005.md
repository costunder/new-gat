# 로컬 문맥 결합 정규화 FULL 결과 — 2026-10-05

**전체 본학습 완료 원문을 수령했다. 큰 정확도 개선은 로컬별 내부 정규화에서 나왔고, 교차 경로의 실제 작용과 일부 CE 기여도 커졌다. 같은 C·내부 정책에서 cross를 추가한 정확도 우위는 확인되지 않았다.**

## 원문·완료 범위

[보존한 콘솔 원문](evidence/local_context_normalization_server_full_20261005.txt)은86,652bytes다.
SHA-256: `32e7237f8ee421f49f090c5e0dad49e73a2b4fbfc218c1b82881d3fa6dcaed33`.
첨부 ID: `44329a2d-caa6-402f-8a85-068711c44298`.
[집계 JSON](evidence/local_context_normalization_summary_20261005.json)은 표시된 수치를 옮긴 자료이며 원시 seed CSV가 아니다.
[파싱 스크립트](../scripts/inspect_context_normalization_report_20261005.py)는 전체 행 수·key 중복을 확인한다. CI·모델을 재계산하지 않았다.

| 항목 | 수령 값 |
| --- | ---: |
| Tuning / final / 전체 학습 | 540 / 300 / 840 |
| 계약 / 새 optimizer 갱신 | 420,000 / 420,000 |
| Primary metric / frozen / branch | 900 / 4,320 / 3,480행 |
| 전체 시간 | 6,820.5363초, 약1시간53분41초 |
| 실제 데이터 / frozen 갱신 | true / 0 |

2개 macro 층·hidden64·각500epoch와20조건 FULL 계약에 맞는다.
`completed=true`, 전체 coverage와 code/graph 보존이 보고됐다.
콘솔의 분류60행·직접 비교312행·상호작용72행·원래 모델 branch120행이 모두 존재한다.
서버 결과 폴더의 실제 경로, source/data/hardware 파일, 원시 CSV·checkpoint는 첨부되지 않았다.
서버 원시 파일 전체를 독립 재검증한 것으로 표현하지 않는다. 전체 elapsed를 모델 수로 나눠 개별 epoch 시간으로 쓰지 않는다.

## 1. 무엇을 바꿨나

각 macro는 `Tρ=M(I−S)(I−ρG)(I−S)R`다. C와 copy 연결은 유지했다.
S는 graph 전체 bound 또는 local별 bound, G는 graph 전체 bound 또는 대칭 edge degree 가중치다.
Gain은 off0 / fixed1 / learned sigmoid(θ)이며 θ는 모델마다 하나를 두 층에서 공유한다.
**C와 개별 연결 가중치는 고정이다. Learned C·로컬 attention 생성기의 학습 실험은 아니다.**

이전6조건도 새로 학습했다. 해당18개 데이터/조건 집계의 LR와 train/validation accuracy 평균은 같았다.
이전 결과와 test accuracy 평균 차이는 최대0.04pp, 표시된 test CE 평균 차이는 최대 약5×10⁻⁵다.
Float32 합산 순서가 달라질 수 있는 계약의 근사 재현이며 bitwise 재현은 아니다.

## 2. 큰 정확도 변화는 내부 정규화

아래는 모두 cross-off다. Accuracy는 final5seed의 test 평균%, 차이는 같은 seed에서 계산한 pp다.

| C | 데이터 | graph 내부 | local 내부 | local−graph pp [95% 구간] |
| --- | --- | ---: | ---: | --- |
| unit | Cora | 58.66 | 80.12 | +21.46 [20.3861,22.5339] |
| unit | CiteSeer | 57.62 | 70.36 | +12.74 [12.0343,13.4457] |
| unit | PubMed | 73.16 | 77.80 | +4.64 [3.37802,5.90199] |
| local_degree | Cora | 79.40 | 80.18 | +0.780 [0.150534,1.40947] |
| local_degree | CiteSeer | 70.66 | 70.48 | −0.180 [−0.501397,0.141395] |
| local_degree | PubMed | 77.66 | 77.90 | +0.240 [−0.291169,0.771167] |

Graph→local accuracy30개 비교 중20개가 양수 구간이며 unit15개와 Cora/local_degree5개다. CE는28개가 개선 구간이다.
Local 내부 정책에서는 local_degree−unit accuracy15개 비교가 모두0을 포함한다.
이전의 큰 C 차이를 가중치 배치 자체의 성과로만 설명할 수 없다. 이 조건에서는 내부 step 선택의 기여가 컸다.

## 3. 같은 C·내부 정책에서 cross를 추가한 효과

| 비교 | Accuracy95% 구간 | CE95% 구간 |
| --- | --- | --- |
| Fixed−off | 24개 모두0 포함 | 12개 개선, 12개0 포함 |
| Learned−off | 24개 모두0 포함 | 6개 개선, 18개0 포함 |
| Learned−fixed | 24개 모두0 포함 | 10개 악화, 14개0 포함; 개선 구간0 |

일부 CE는 개선됐지만 추가 정확도 이득은 확인되지 않았다. Learned가 fixed보다 일관되게 좋은 결과도 없다.
0 포함 구간은 동등성이나 무효과의 증명이 아니다.

Local 내부에서 edge fixed−off의 주요 ΔCE는 다음과 같다.

| C | 데이터 | ΔCE [95% 구간] |
| --- | --- | --- |
| unit | Cora | −0.00831550 [−0.00935171,−0.00727930] |
| local_degree | Cora | −0.00685143 [−0.00755190,−0.00615097] |
| unit | CiteSeer | −0.00265635 [−0.00405564,−0.00125706] |
| local_degree | CiteSeer | −0.00236418 [−0.00352083,−0.00120754] |

Cora/unit/local의 accuracy는 off80.12%, edge fixed80.10%다. CE가 개선돼도 정답 개수는 늘지 않을 수 있다.
CE는 정답 클래스에 준 확률을, accuracy는 최댓값 클래스가 맞는지를 평가한다.
PubMed/local_degree/local은 off77.90%, edge fixed77.36%지만 그 차이 구간은0을 포함한다.

## 4. Cross 정규화와 상호작용

Edge−graph accuracy24개 중 양수 구간은 Cora/local_degree/local/fixed의 **+0.20pp [0.00367451,0.396325]** 하나다.
Edge−off accuracy 우위가 아니다. Edge−graph CE24개 중11개는 개선 구간이다.

상호작용 accuracy36개 중0을 제외하는 것은 CiteSeer/unit/graph-cross fixed의
`(local cross−local off)−(graph cross−graph off)` **+0.16pp [0.0184294,0.301571]** 하나다.
Graph에서 cross 효과−0.14pp가 local에서+0.02pp로 바뀐 상대 비교다. Local cross−off 자체의 이득을 증명하지 않는다.
상호작용 CE36개 중10개가 개선 구간이며 Cora8개·CiteSeer2개다. PubMed는 모두0을 포함한다.
모두 같은 public split의5seed·다중 비교 미보정·이전 test를 본 후의 탐색이다.

## 5. 실제 작용과 gain 학습

아래는 각 학습 모델의 현재 Z에서 측정한 첫 층 matched `Δ/off`를 %로 변환한 평균이다.
서로 다른 학습 조건의 표는 동일 Z를 고정한 순수 operator 비교는 아니다.

| C | 데이터 | local + graph fixed | local + edge fixed |
| --- | --- | ---: | ---: |
| unit | Cora | 0.0311971% | 2.03626% |
| unit | CiteSeer | 0.0345046% | 1.69182% |
| unit | PubMed | 0.0304056% | 2.57782% |
| local_degree | Cora | 0.0287987% | 1.92699% |
| local_degree | CiteSeer | 0.0333820% | 1.65370% |
| local_degree | PubMed | 0.0300021% | 2.59057% |

Local의 첫 층 fixed에서 edge 조건의 작용은 graph 조건보다 약49–86배 컸다.
Fixed local+edge의 첫 층에서 적용된 cross energy는 약82–91% 줄었다.
같은 원래 노드 copy의 문맥 불일치를 줄인 관측이며, flow 복원이나 정보 보존율이 아니다.
Off ratio 약1.4–3.1×10⁻⁸은 float32 재계산 잔차 수준이다.
Learned gain 평균은0.0618136–0.902937이며24개 데이터/조건 중18개가 초기0.5보다 작다.
Epoch별 θ gradient/update 원문이 없어 모든 step의 비영 gradient나 갱신 크기는 주장하지 않는다.

## 현재 판단과 남은 자료

Frozen4,320행과 hash 보존은 완료 기록에 보고됐다. 이번 콘솔에 개입 ΔCE·flip·no-op 잔차,
formula residual·적용 S/G degree max의 원시값은 없다. 자원1,976행도 선언됐지만 실제 GPU/VRAM·pack/chunk·epoch 시간을 직접 읽지 못했다.

**로컬 내부 정규화의 큰 기여와 일부 교차 CE 기여는 관측됐다.**
교차의 추가 accuracy 우위, learned C, 정보 복원, 독립 그래프 일반화와 신규성은 아직 입증되지 않았다.
이번 결과를 정리하면서 새 모델·학습 규모·loss를 변경하거나 학습을 실행하지 않았다.
