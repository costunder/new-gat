# 증거의 수준, 누락 자료와 아직 답하지 못한 질문

기준일: 2026-10-05. 이 자료는 지금까지의 연구를 **현재 코드·확인된 실제 실행·서버에서 받은 원문·완료 근거가 없는 설계**로 구분해 GPT에 전달하기 위한 것이다. 최신 정규화 FULL completion·요약을 반영했으며 자료 정리 작업에서 새 학습을 하거나 과거 checkpoint를 재선택하지 않았다.

## 1. 무엇이 어떤 증거인가

| 증거 종류 | 직접 확인할 수 있는 것 | 그것만으로 확인할 수 없는 것 |
| --- | --- | --- |
| 실제 코드·FULL/DEBUG 설정·수식·단위 테스트 | 구현된 연산, 학습 경로, 현재 계약과 검사 내용 | 서버의 전체 학습이 해당 코드로 완료됐다는 사실 |
| 로컬 단위 테스트·DEBUG 결과 파일 | 명시된 fixture의 algebra/gradient/optimizer/resume/coverage | 실제 citation 성능, A6000 FULL 실행 시간, 독립 그래프 일반화 |
| 사용자가 보낸 서버 completion·SUMMARY·로그 | 보고된 실행 범위, 완료 상태, 표시된 수치 | 첨부되지 않은 per-seed CSV·checkpoint·source/data 파일의 재검증 |
| SERVER_RESULTS / SERVER_FINDINGS 문서 | 수령한 원문을 계약과 대조한 해석 | 새로운 독립 실험 또는 서버 모델의 재평가 |
| GPT 검토·설계 첨부 | 논의한 수식, 지적, 다음 설계 제안 | 제안이 이미 구현·학습돼 검증됐다는 사실 |
| 실패 로그 | 해당 명령·단계가 실패했다는 사실 | 후속 정상 실행의 실패 또는 성공, 미실행 조건의 성능 |
| 전달 ZIP manifest·hash·CRC 검사 | 포함 파일의 출처·bytes·무결성·포함 여부 | 모델의 성능, 과학적 신규성, 미제공 서버 결과의 진위 |

같은 run의 콘솔·completion·SUMMARY를 여러 독립 반복으로 세지 않는다. Graph/channel/ego/relation 행 수와 model seed 수도 서로 다르다. 과거 VERIFICATION의 “FULL 미실행”은 로컬 개발 당시 상태일 수 있으므로 뒤에 수령한 서버 원문과 날짜를 함께 읽는다.

## 2. 최근 서버 원문: 수령·보존한 파일

아래는 실제 저장소에 있는 원문 파일이다. Hash는 이 보존 파일의 bytes를 확인하는 값이다. 원문 안에 이름만 언급된 CSV·그림·checkpoint까지 받아 검증했다는 뜻은 아니다.

| 실험·자료 | 보존 원문 | bytes | SHA-256 |
| --- | --- | ---: | --- |
| 로컬 에너지 fixed FULL completion+SUMMARY | [local_energy_server_full](../evidence/local_energy_server_full_20261004.txt) | 8,425 | `af232396c6194b644b16d0913d5989df1bc9c6a72799744132996ac35a663dcc` |
| Receiver fixed FULL completion+SUMMARY | [receiver_aggregation_server_full](../evidence/receiver_aggregation_server_full_20261004.txt) | 14,522 | `820122a9109ce0fdc5b0781f901547c1c30d851491f91bfd93956b71082d3df8` |
| Scalar E/J prediction FULL completion+SUMMARY | [local_prediction_server_full](../evidence/local_prediction_server_full_20261004.txt) | 24,552 | `a2955bbe563a49b2ba96f565799f6224b7ee4c6404221f600472d46e36683760` |
| Prediction 층별 제거 test CE12행 | [local_prediction_layer_removal](../evidence/local_prediction_layer_removal_server_20261004.txt) | 864 | `d7276674fde6b7fdff1afd0330215ef3c2de8e0a3f878da242508cffc779630c` |
| E/J placement FULL 전체 요약 | [local_placement_server_full](../evidence/local_placement_server_full_20261004.txt) | 84,818 | `a5b7e6db68d60f7cd5e13b3864d7d33d2c4ff021d1d4e6ec1fc2c1e5875cf408` |
| Copy coupling fixed FULL 요약 | [local_context_coupling_server_full](../evidence/local_context_coupling_server_full_20261004.txt) | 6,130 | `b156f830ee8d349b687ff992959f6d3444282b8135cf327af83c5fbb145632fe` |
| Copy classification FULL completion+SUMMARY | [local_context_classification_server_full](../evidence/local_context_classification_server_full_20261004.txt) | 30,827 | `889e5f356f896f1d85051df302ccc60ef22d13dfa8533a1e51241503844bf6b9` |
| Copy normalization FULL completion+SUMMARY | [local_context_normalization_server_full](../evidence/local_context_normalization_server_full_20261005.txt) | 86,652 | `32e7237f8ee421f49f090c5e0dad49e73a2b4fbfc218c1b82881d3fa6dcaed33` |

[정규화 추출 JSON](../evidence/local_context_normalization_summary_20261005.json)은 반올림된 콘솔 요약에서 추출한60개 평균 표·312개 직접 비교·72개 상호작용·120개 원래 분기 평균과 completion이다. 원시 seed CSV가 아니며 구간을 독립 재계산하지 않았다. [전체 결과 해설](08_NORMALIZATION_FULL_RESULTS.md)과 [서버 결과 분석](../LOCAL_CONTEXT_NORMALIZATION_SERVER_FINDINGS_20261005.md)도 같은 수령 원문을 해석한 자료다.

### 완료와 자료 부족을 같이 기록해야 한다

- Local energy와 receiver에는 FULL completion JSON과 요약이 있다. 원래 local source 경로는 `/home/aicompetition07/new-gat/results/local-energy-20261004-002815`로 확인됐다.
- Scalar prediction에는 FULL completion JSON·요약·후속 제거12행이 있다. GPU/VRAM·실제 run 폴더·전체 raw seed CSV/checkpoint는 이 첨부에 없다.
- Placement는 전체 집계 표·예산/coverage 서술을 수령했다. Completion.json 자체, 실제 결과 경로, source digest·checkpoint·seed별 CSV를 받지 않았다. 요약의840회 완료 보고를 전체 원시 파일의 독립 재검증이라고 표현하지 않는다.
- Copy fixed audit는201 graph·parameter0·update0·정확한 차이식을 보고한 SUMMARY를 수령했다. 서버 결과 경로·completion/source digest·resource 원본은 해당 메시지에 없다.
- Copy classification은252회·126,000 update·2,067.5805초의 completion과 요약을 수령했다. 서버 결과 경로·전체 epoch history·raw CSV/checkpoint·hardware/source/data 원본은 첨부되지 않았다.
- Normalization은20조건·840회·420,000 새 update·6,820.536256초의 FULL completion·요약을 수령했다. Actual_data=true, 동결 평가 update0이다. Completion은 primary900/frozen4,320/branch3,480행을 보고하지만 개입별 수치·전체 branch/epoch gradient·raw seed CSV/checkpoint·hardware/source/data 원본은 첨부되지 않았다. 로컬 본학습은 하지 않았다. 이전222개 검사와 DEBUG360회·1,080 update·완료 checkpoint 재사용0 update는 별도 구현 검증 기록이다.

최신 실제 상태·수치는 [04_LOCAL_AND_COPY_EXPERIMENTS.md](04_LOCAL_AND_COPY_EXPERIMENTS.md)에 정리했다. 과거 wedge 원문 해석은 [wedge 실험·결과](../../research/wedge_propagation/gpt_handoff/03_EXPERIMENTS_AND_RESULTS.md), 더 이전 Conductance/CGAT/Cycle/Tree의 완료·부분완료·실패·계획 구분은 [이전 연구 기록](../gpt_experiments_20261004/03_EARLIER_HISTORY.md)을 함께 읽는다.

## 3. 실패·검토·역사 기록을 잘못 읽지 않기

Receiver 첨부 `00ecf7a9-5c65-4eb3-8370-ed8ee4b864ad`는 CPU operator calibration 중 CUDA initialization error로 중단한 **실패 로그**다. 실제 FULL completion 첨부는 `647e22c8-116d-4f15-a7ef-c376fca5db99`다. 초기 전달 ZIP의 잘못된 label을 [검토 반영 기록](../gpt_experiments_20261004/07_GPT_REVIEW_RESPONSE.md)에서 바로잡았다. 실패 첨부를 완료 증거로 세지 않는다.

이전 output initialization·Conductance V5에는 import/GPU 할당/abort/OOM/nonfinite gradient/감사 오류가 있었다. 정확한 run·단계·후속 복구 여부를 각 원문과 함께 읽는다. 이 로그들을 지금 copy 모델의 본학습 실패 원인으로 옮겨 붙이지 않는다.

사용자가 전달한 GPT 검토는 설계·비평 자료다. GPT가 보고한 검사 수나 추천안을 이 작업의 새 실행 결과로 세지 않는다. Attachment가 과거 경로에서 없으면 패키지 색인에 누락으로 표시하고 요약으로 원문을 재작성해 채우지 않는다.

## 4. 실제로 입증된 범위와 아직 입증되지 않은 범위

| 질문 | 지금의 증거가 말하는 것 | 남은 범위 |
| --- | --- | --- |
| 발생행렬·이차/쌍선형 계산이 코드와 맞는가 | Fixed 감사·독립 dense·sparse/gradient 참조에서 항등식이 일치 | 이 대수 일치만으로 유용성·신규성을 판정하지 못함 |
| Cycle나 수신 합 상쇄가 곧 영구 정보 손실인가 | 알려진 양의 C·공통 H·전체 관측의 q는 수치상 복원됨 | 임의 q, unknown/learned C의 미관측 상태, 부분 수신, 경계 누락, source별 독립 상태, 비선형 압축은 별도 문제 |
| Scalar E/J가 예측에 실제 쓰였나 | Active lift가 갱신됐고 분기 제거 시 예측·CE가 변함 | Base보다 보편적으로 좋은 정확도 효과는 확인되지 않음 |
| E/J 위치만 바꾸면 좋아지나 | 일부 CE/accuracy 후보가 있으나 위치의 보편적 우세 없음 | 다중 비교를 고려한 후보 확인, 다른 split/graph 평가가 필요 |
| Copy cross가 merge 뒤에 남나 | 같은 S/A 앞뒤 전달의 정확한 mixed-action 차이, actual input에서 nonzero 확인 | 모든 입력의 nonzero 또는 추가 표현력/복원 능력은 보장하지 않음 |
| Cross 강도 θ를 학습하나 | 로컬 CE→gradient→Adam 연결과 서버 선택된ρ 변화 확인 | 서버 전체 epoch별 gradient/update 원본은 아직 미수령 |
| Cross가 실제 citation accuracy를 개선했나 | 첫6조건과 후속 정규화 FULL에서 같은 C·intra의 cross−off accuracy 구간은 모두0 포함; 후속 비교는48개 | 우위도 동등성도 입증되지 않음; 일부 CE 개선·cross 정책끼리 비교와 구분 |
| 정규화 변경이 도움이 되나 | FULL에서는 unit의 graph→local 내부 변경이 cross off에서도 Cora+21.46pp·CiteSeer+12.74pp·PubMed+4.64pp로 구간이 양수. Edge cross는 일부 CE 비교에서 개선 | 추가 cross의 accuracy 개선은 확인되지 않음; 원시 seed·개입·학습 중 gradient 검토는 남음 |
| C 생성 규칙을 실제 새 모델에서 학습했나 | 현재 local/copy C는 unit/local_degree로 고정 | Learnedρ는 learned C가 아님. Wedge teacher의 C 학습 성과도 새 모델의 성과가 아님 |
| 새 부분구조/문맥/독립 graph에 일반화하나 | Wedge의 지정한 합성 teacher·feature 평가는 그 계약 안에서 수행 | 현재 local/copy public split 결과는 새 graph 일반화가 아님 |
| 표준 GCN/GATv2보다 좋은가 | 이전 wedge 트랙에는 자체 GCN/GATv2 대조가 있음 | 최신 local/copy에 동일 조건의 직접 GCN/GATv2 대조 결과는 없음 |
| 기존 연구와 정확히 다른가 | 에너지·copy 상태·연산자의 실제 정의를 기록함 | 체계적 선행연구 대조와 신규성 증명이 필요 |

## 5. 수식과 해석에서 계속 지켜야 할 구분

### 5.1 현재 q의 복원과 arbitrary cycle ambiguity

`Bᵀ(q+z)=Bᵀq`, `Bᵀz=0`는 임의 엣지 메시지의 구별 불가능성을 나타낸다. 그러나 q가 알려진 양의 C 아래 `q=CBH`로 제한되고 전체 d=Bᵀq를 관측하면

\[
q=CB(B^\top CB)^\dagger d
\]

로 제한된 실제 q를 복원할 수 있다. H는 연결성분 상수만큼 식별되지 않지만 q는 그것에 영향받지 않는다. 실제 q의 Euclidean cycle projection이 nonzero라는 관측과 제한된 q의 복원 가능성은 양립한다.

E/J 재현오차가 작은 것 또한 known-C·전체 Y에서의 결정 가능성에 관한 결과다. 실제 GNN이 한 local에서 한 층에 전역 역산을 했다는 뜻은 아니다.

### 5.2 큰 이차형식과 실제 모델의 공간

Scalar 실험의 내부 E와 `J=⟨d_v,d_u⟩`를 임의로 더하면 원래 물리 union graph의 양의 weighted Laplacian이 자동으로 나오지 않는다. 내부·교차 블록의 차수항, 중복, 겹치는 원래 ID와 계수를 함께 정해야 한다. 현재 copy 모델은 별도 공간의 `A+λK`에서 내부·교차 에너지를 명시했다.

Copy cross는 같은 물리 노드의 로컬 표현 차이를 줄이는 consensus 작용이다. 차이를 그대로 보존하거나 원래 cycle 메시지를 복원하는 기능이라고 부르지 않는다. 처음 Y=RH에서는 cross energy0이며 내부 전달이 다른 로컬 문맥을 만든 뒤 작용한다.

### 5.3 효과가 남는 것과 표현력이 늘어나는 것

현재 고정 S/G/R/M의 macro는 하나의 전역 node 선형 연산 T로도 쓸 수 있다. `GR=MG=0`과 같은 S의 앞뒤 적용으로

\[
T_\rho-T_0=-\rho MSGSR
\]

가 남는다. 이는 구현한 cross 경로의 실제 작용을 정한다. Copy를 유지했다는 이유만으로 모든 node 선형 모델보다 강한 표현력이나 새로운 복원 능력을 주장할 수 없다.

2홉 의존성도 cross만의 성과가 아니다. Off가 이미 내부2회를 사용하므로 같은 깊이의 on/off Jacobian 차이를 비교해야 한다. Unit3노드 path 예에서 endpoint 미분은 둘 다.005였고, 별도 비정규5노드 control에서는 추가 cross 의존성이 관측됐다. 모든 2홉 입력에 cross가 추가로 작용한다고 일반화하지 않는다.

### 5.4 정규화·에너지·gradient의 현재 의미

새 정규화는 S와 G를 대칭 PSD로 유지하고 weighted degree를.5 이하로 정한다. 안정성 주장은 copy 공간 norm 및 `D=RᵀR`의 물리 weighted norm에 관한 것이다. D는 copy 개수이고 전체1홉에서는 physical degree+1이다. 일반 Euclidean node norm이나 projection/ReLU/dropout까지 포함한 전체 분류기의 모든 에너지 감소를 보장하지 않는다.

학습·평가의 `intra/cross_energy`는 실제 적용한 S/G의 **½trace**다. 이전 fixed local E의 trace, 원시 A/K energy와 숫자를 섞지 않는다. `context_norm=||GY1||`, `raw_context_norm=||KY1||`도 따로 기록한다. Off의 graph-G energy/norm은 참고 진단이며 실제 gain은0이다.

Local η_v는 해당 block의 C 전체 배율을 상쇄한다. 특히 triangle-free ego가 star이면 unit/local_degree의 적용 S가 같아질 수 있다. 20개의 설정 ID가 있다는 이유만으로 모든 graph에서20개의 다른 연산이 된다고 주장하지 않는다.

## 6. 숫자·통계·시간을 읽는 기준

- Accuracy 평균은%, 차이는pp다. CE는 L2 항을 제외한 평가 분류 손실이다. Accuracy양수와 CE음수가 개선이다.
- ±는 표본 std이고 95% 구간과 다르다. 현재 citation paired t95% 구간은 같은 public split의 final5seed 변동이다.
- 독립 split·새 graph·새 문맥의 불확실성을 포함하지 않는다. 이전 test를 본 뒤 정한 다음 설계는 후속 탐색이다.
- 모든 지정 비교를 보존한다. 다중 비교 보정은 없고, 0포함 구간은 동등성 증명이 아니다. 한 양수 후보를 전체 연구의 성공으로 바꾸지 않는다.
- Frozen 제거/gain 개입은 동일 checkpoint의 의존성을 본다. 그 파라미터를 재학습한 off/fixed 모델과 같은 실험이 아니다.
- Branch/base norm 비율과 matched Δ/off는 표현의 크기다. 정보 보존율이나 accuracy의 책임 비율이 아니다.
- 전체 study elapsed, calibration 후보 epoch 시간, 실제 선택 pack의 epoch 시간은 다르다. 총 elapsed를 모든 독립 update 수로 나눠 개별 epoch 시간으로 부르지 않는다.
- Independent seed packing은 서로 다른 모델을 동시에 계산한다. 모델 하나의 effective graph batch는1이며 seed 수를 곱해 적지 않는다. Chunking은 전체 엣지를 계산하는 작업 메모리 조절이지 sampling이 아니다.
- DEBUG 성능·RTX5070Ti 시간은 A6000 FULL 성능·실행 시간의 예측값이 아니다.

## 7. 지금 이어서 확인할 질문

1. **Normalization FULL의 해석:** 52개 직접 비교와12개 paired 상호작용을 데이터3개·accuracy/CE에서 모두 보존했다. 내부 정규화의 큰 accuracy 차이, 일부 cross CE 개선, cross−off accuracy 구간의0포함을 구분한다. 5seed·고정 public split·보정 없는 다중 비교를 넘어 효과가 유지되는지는 아직 확인하지 않았다.
2. **실제 작용:** 정규화 뒤 `||GY1||`, matched Δ/off, formula residual, θ의 CE gradient와 실제 update가 어떻게 달라지는가? 선택된 원래 분기의 평균120행은 수령했지만 전체 frozen 개입과 epoch gradient/update 원문은 없다. 크기가 커지는 것만으로 분류 개선을 판정하지 않는다.
3. **C와 step 분리:** 내부 local 정규화에서는 local_degree−unit accuracy15개 구간이 모두0을 포함한다. Triangle-free에서 S가 같은 조건과 CE 차이를 함께 읽어 기존 큰 C 차이가 C 배치·η·전파 중 어떤 효과였는지 검토한다. C가 불필요하다는 결론이나 단일 원인의 확정은 현재 증거를 넘는다.
4. **Learned C:** 로컬 attention에 해당하는 공유 C 생성 규칙을 실제 새 구조에 연결했을 때 무엇이 달라지는가? 이는 아직 현재 고정 C 실험의 완료 결과가 아니다.
5. **부분 관측과 정보:** 경계 누락·서로 독립된 local 상태·비선형 압축에서 실제 구별 불가능성이 생기는가? 무엇을 관측하고 복원 대상으로 둘지 먼저 정해야 한다.
6. **일반화와 기준선:** 동일 조건 GCN/GATv2, 새로운 문맥 크기·sampling·독립 graph에서 규칙이 재사용되는가? 이전 다른 구조의 baseline 점수를 공정한 현재 대조로 재사용하지 않는다.
7. **신규성:** 기존 에너지 기반 GNN·copy/overlap consensus·다중 차트·라플라시안 필터 연구와 연산자/상태/학습 목표 수준에서 무엇이 다른가? 제목·키워드 일치만으로 동일 연구 또는 신규라고 확정하지 않는다.

Normalization FULL의 완료·요약은 이미 수령했다. 이를 원시 서버 산출물 전체의 독립 재평가로 바꾸어 설명하지 않는다. 나머지 질문에는 현재 완료 근거가 없으므로 GPT가 임의로 구현 완료·실험 성공으로 설명해서는 안 된다.

## 8. 패키지에 없는 서버 자료

필요한 독립 재검증 자료는 서버에 있는 전체 checkpoint·epoch history·seed별 metric/intervention/branch CSV·source/data manifest·hardware/resource JSON·원본 topology/feature/teacher cache다. 저장소에 이미 있는 curated 수치·작은 checkpoint·DEBUG 결과의 포함은 패키지 manifest에 표시한다. 이것이 최근 서버 FULL 산출물 전체를 받았다는 뜻은 아니다.

전달한 사용자 attachment·직접 메시지 전사·과거 GPT 검토의 포함 여부와 누락 원인은 패키지 색인을 따른다. 오래된 경로의 실제 파일이 없으면 없다고 기록한다. 원문 링크만 있었고 내용을 확보하지 못했다면 확보한 것처럼 인용하지 않는다.

이번 ZIP 작업의 검사는 파일 무결성·필수 포함·구문·JSON·링크·출처 기록이다. 기존222개 단위 검사와 DEBUG 결과는 당시 실행 기록으로 인용하며, ZIP을 만들며 다시 본학습을 했다고 표현하지 않는다.
