# 로컬 에너지부터 copy 문맥 결합까지: 실제 실험과 결과

기준일: 2026-10-05. **Copy 문맥 결합의 첫 분류와 후속 정규화 비교의 서버 FULL completion·요약을 수령했다. 정규화는 20조건·840학습·420,000 update의 완료를 보고한다. 로컬에서는 본학습을 하지 않았다.**

이 문서의 FULL 수치는 사용자가 전달한 서버 원문에서 읽었다. 서버의 모든 CSV·checkpoint를 이 PC로 가져와 다시 평가한 결과는 아니다. 상세한 증거 구분은 [05_EVIDENCE_AND_OPEN_QUESTIONS.md](05_EVIDENCE_AND_OPEN_QUESTIONS.md)를 따른다.

## 1. 실험 순서와 완료 상태

| 순서 | 실제 코드 | 무엇을 확인했나 | 완료 상태·범위 |
| --- | --- | --- | --- |
| L1 | [local_energy_relations](../../research/local_energy_relations/README.md) | 모든 1홉 induced local의 내부 에너지, 관계 J, cycle 성분, 중복·전달 범위 | 서버 FULL 고정 감사: 합성198 + 실제 citation3 = 201 graph, 학습0 |
| L2 | [receiver_aggregation](../../research/local_energy_relations/receiver_aggregation/README.md) | 여러 로컬의 메시지를 같은 수신 행에 합친 뒤 실제 q/H/E/J 복원이 가능한가 | 서버 FULL 고정 감사: 동일201 graph, 두 C·세 상태·다섯 관측 조건, 학습0 |
| L3 | [prediction](../../research/local_energy_relations/prediction/README.md) | 스칼라 E/J를 학습 벡터로 노드 특징에 더하면 분류에 도움이 되는가 | 서버 FULL 분류: 8조건, 336학습·168,000 update |
| L4 | [placement](../../research/local_energy_relations/placement/README.md) | E/J를 첫 층·출력층·두 층 중 어디에 넣는가 | 서버 FULL 요약 수령: 20조건, 840학습·420,000 update 보고 |
| C1 | [local_context_coupling](../../research/local_context_coupling/README.md) | 스칼라 통계 대신 로컬 copy 상태를 내부→교차→내부로 전달하면 교차 작용이 merge 뒤에 남는가 | 서버 FULL 고정 감사: 201 graph, parameter0·update0 |
| C2 | [classification](../../research/local_context_coupling/classification/README.md) | 같은 내부 두 번을 유지한 cross off/fixed/learned의 실제 분류 기여 | 서버 FULL 완료 원문 수령: 6조건, 252학습·126,000 update |
| C3 | [normalization](../../research/local_context_coupling/normalization/README.md) | graph/local 내부 step과 graph/edge 교차 정규화를 분리하면 추가 cross 기여가 달라지는가 | 서버 FULL completion·요약 수령: 20조건, 840학습·420,000 update. 선행 구현222검사·전체 DEBUG 완료, 로컬 본학습 미실행 |

L1/L2/C1의 H0/H1/H2는 공통 전역 reference 확산으로 만든 상태다. 학습된 encoder의 세 층이 아니다. 이 고정 감사에 epoch나 분류 성능을 붙이지 않는다.

### 분류 FULL에 공통인 데이터와 학습 기준

| 데이터 | 전체 노드 × 특징 | 클래스 | train / validation / test 노드 |
| --- | --- | ---: | --- |
| Cora | 2,708 × 1,433 | 7 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 × 3,703 | 6 | 120 / 500 / 1,000 |
| PubMed | 19,717 × 500 | 3 | 60 / 500 / 1,000 |

원래 public split, row-sum-normalized 특징, 전체 물리 노드·엣지와 모든 induced 1홉 로컬을 사용한다. 합성198 graph는 고정 감사에 쓰며 citation 분류에 가짜 label을 붙여 넣지 않는다. Sampling ratio는1이다.

분류는 2층·hidden64·dropout.5, run당500 epoch, Adam, LR 후보 .001/.003/.01, tuning seed101/202/303, final seed11/23/37/53/71을 쓴다. Checkpoint와 LR는 validation으로 선택하고, final 선택을 모두 고정한 뒤 test를 평가한다. 각 실험의 기본 전파가 달라 과거 점수와 새 점수를 직접 같은 대조군으로 합치지 않는다.

## 2. L1: 로컬 내부 에너지·집합 사이 관계의 고정 감사

근거: [서버 completion·전체 요약 원문](../evidence/local_energy_server_full_20261004.txt), [고정 설계](../../research/local_energy_relations/EXPERIMENT_DESIGN.md), [수식](../../research/local_energy_relations/MODEL_MATH.md).

각 중심 v의 집합은 `S_v={v}∪N(v)`이고 이웃끼리의 엣지도 포함한다. `q_v=C_v B_v H_v`, `d_v=B_vᵀq_v`이며 내부 에너지는 `E_v=tr(H_vᵀB_vᵀC_vB_vH_v)`다. `||q_v||²`은 C가 두 번 들어가므로 일반적으로 E_v와 다르다.

여기서 고정 C는 `unit`과 `local_degree=2/(로컬 양 끝 degree 합)`이다. C 생성 규칙을 학습하지 않는다. J는 공통 엣지 관계·공통 노드 d 내적·서로 다른 incident 엣지 관계를 기록하며 `Jnode=2Jshared+Jdistinct`를 검사한다. Jdistinct의 부호를 없애지 않는다.

서버 완료 기록은 201 graph·합성 scalar 입력3,168개·전체 노드/엣지/특징·parameter0·update0을 보고한다. 563.8261초에 완료됐고 원시 행 수는 local1,199,952 / relation17,961,240 / transfer3,592,248 / temporal799,968이다. 이 occurrence 수는 독립 학습 반복 수가 아니다.

### H0에서 실제로 측정한 것

| 데이터 | C | 전체 local E 합 | inverse-occurrence 보정 E 합 | 실제 q의 cycle 제곱 비율 | q 상대 복원오차 최대 |
| --- | --- | ---: | ---: | ---: | ---: |
| Cora | unit | 1,891.0446 | 649.61352 | 1.4049e−19 | 2.1088e−9 |
| Cora | local_degree | 621.90103 | 230.05348 | 0.0129690 = 1.29690% | 1.6750e−9 |
| CiteSeer | unit | 640.72285 | 236.65637 | 6.7015e−20 | 1.6508e−9 |
| CiteSeer | local_degree | 259.23156 | 108.08671 | 0.00570604 = 0.570604% | 6.3795e−10 |
| PubMed | unit | 7,253.1956 | 2,585.8757 | 2.5599e−19 | 4.3915e−9 |
| PubMed | local_degree | 1,957.2545 | 792.35083 | 0.0142675 = 1.42675% | 1.0845e−9 |

같은 물리 엣지가 여러 ego에 들어가므로 raw local E 합은 중복을 센다. Inverse-occurrence 보정은 물리 엣지별 평균 C의 에너지와 일치했다. Unit에서만 이 보정 값이 원래 unit 물리 에너지와 같다. 보정 항등식의 citation 표 최대 절대오차는3.5016e−11이다.

Cycle 성분이 있다고 해서 이 제한된 q가 복원 불가능한 것은 아니다. **알려진 양의 C와 모든 local d를 관측한 q=CBH**는 복원됐다. H0/H1/H2 citation 표의 q 상대오차 최대는4.3915e−9였다. 부분 관측·경계 누락·임의 q의 복원 결과로 확대하지 않는다.

전달 표는 공통 노드에 남긴 d와 빠진 d, retained/boundary/omitted 엣지를 구분한 진단이다. 이 d를 학습 층의 다음 상태로 쓰는 모델은 L1에 없다.

## 3. L2: 수신 합에서 숨는 성분과 제한된 실제 복원

근거: [서버 원문](../evidence/receiver_aggregation_server_full_20261004.txt), [결과 해석](../RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md), [설계](../../research/local_energy_relations/receiver_aggregation/EXPERIMENT_DESIGN.md).

송신 local별 receipt를 구분한 상태에서 수신 local·원래 노드별 합 Y를 만든다. `tagged/sum/sum_within/sum_between/sum_both`를 기록한다. 알려진 C와 하나의 공통 전역 H에서 만든 실제 메시지라는 제약을 유지한다.

서버 완료 기록: 201 graph, 입력 지표19,026행, 복원 요약1,206행, 조건 요약6,030행, 관계 요약6,030행, parameter0·update0. 준비·계산 시간3,464.85초다.

| 데이터 | C | H0 receipt에서 직접 숨는 제곱 norm 비율 | 수신 합 Y에서 실제 q 상대 복원오차 |
| --- | --- | ---: | ---: |
| Cora | unit | 34.35% | 2.70996e−7 |
| Cora | local_degree | 10.33% | 6.796091e−8 |
| CiteSeer | unit | 28.16% | 3.305544e−7 |
| CiteSeer | local_degree | 9.48% | 8.550849e−8 |
| PubMed | unit | 22.69% | 6.815293e−7 |
| PubMed | local_degree | 12.61% | 1.197799e−7 |

전체 citation18개 데이터/C/state 요약에서 Y 상대잔차 최대9.992807e−9, q 복원오차 최대6.815293e−7, 연결성분 평균을 제거한 H 오차 최대8.026027e−6, receipt 재구성 오차 최대4.773164e−8이다. 숨는 receipt 비율을 영구 정보 손실률로 부르지 않는다.

Y만으로 복원한 H에서 E/J를 재계산했다. E 상대 재현오차 최대9.961601e−9, 개별·연속 단계 J까지 포함한 표의 상대오차 최대1.831518e−8이었다. `additional_rank=0`은 known-C·공통 H·전체 관측의 정리에서 정한 값이지 citation 전체 행렬의 SVD 측정값이 아니다. E/J를 복원 loss에 넣어 decoder를 개선한 실험도 아니다.

전체 Y를 사용하는 반복 역산과 한 local의 한 층 업데이트는 다르다. 이 전역 결정 가능성은 E/J의 제한된 깊이에서의 분류 유용성을 판정하지 않는다.

## 4. L3: E/J를 스칼라 특징으로 넣은 분류

근거: [서버 완료·요약 원문](../evidence/local_prediction_server_full_20261004.txt), [분석](../LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md), [모델 수식](../../research/local_energy_relations/prediction/MODEL_MATH.md).

각 고정 C에서 `base/within(E)/between(J)/both`를 새로 학습했다. E/J를 채널 평균 스칼라로 만들고 학습 벡터를 곱해 두 층의 기본 전파에 더했다. 분기는 실제 CE→gradient→optimizer 경로에 연결됐다. 이것은 메시지 전체를 복원하거나 통합 물리 에너지의 gradient를 직접 전파하는 모델과 다르다.

서버 완료 기록은 tuning216 + final120 = 336학습, 168,000 update, metric360 / 제거metric1,350 / 층진단1,140행, 2,263.15초다. 동결 평가 update는0이다.

### 같은 C base 대비 test accuracy 차이

단위는 pp다. 양수가 개선이며 final5seed 평균 차이다.

| 데이터 | C | base accuracy % | E−base | J−base | E+J−base |
| --- | --- | ---: | ---: | ---: | ---: |
| Cora | unit | 76.22 | −0.30 | −0.60 | −0.18 |
| Cora | local_degree | 74.34 | +0.10 | −0.86 | +0.22 |
| CiteSeer | unit | 67.24 | −1.24 | −2.84 | −2.18 |
| CiteSeer | local_degree | 64.90 | −0.26 | −2.12 | −1.28 |
| PubMed | unit | 79.60 | −0.32 | −0.26 | −0.16 |
| PubMed | local_degree | 78.94 | +0.10 | +0.18 | +0.10 |

18개 base 비교에서 accuracy95% 구간 전체가 양수인 경우는 없다. CiteSeer J의 구간은 unit −2.84pp [−3.9175,−1.7625], local_degree −2.12pp [−3.8075,−0.43253]으로 악화됐다. PubMed J의 CE는 unit−0.0024997, local_degree−0.0025772로 구간이 모두 음수였지만 accuracy 개선 구간은 확인되지 않았다.

활성 lift norm은 모두 양수이고 train accuracy는100%였다. 분기/base norm은 첫 층 E0.92–3.21%·J0.33–1.64%, 출력층 E5.22–12.03%·J3.20–15.99%였다. 이는 표현 크기이며 정보 복원율이나 정확도 기여율이 아니다.

### 같은 checkpoint에서 층별 E+J 제거

근거: [사용자가 추가 전달한 test CE12행](../evidence/local_prediction_layer_removal_server_20261004.txt). 양수는 제거 시 악화, 음수는 제거 시 개선이다.

| 데이터 | C | 첫 층 제거 ΔCE | 출력층 제거 ΔCE |
| --- | --- | ---: | ---: |
| Cora | unit | +0.002446 | +0.012675 |
| Cora | local_degree | +0.001549 | +0.009127 |
| CiteSeer | unit | +0.000176 | +0.012594 |
| CiteSeer | local_degree | +0.000190 | +0.011077 |
| PubMed | unit | −0.000458 | −0.005809 |
| PubMed | local_degree | −0.000482 | −0.006491 |

Cora/CiteSeer 출력층 제거 구간은 모두 양수, PubMed는 두 C·두 층 제거 구간이 모두 음수다. 이 개입은 projection을 고정한 제거다. 별도 재학습한 base 대비 성능과 같지 않으며, 첫 층 제거는 후속 특징·E/J도 다시 계산한다.

## 5. L4: E/J 주입 위치를 바꿔 재학습

근거: [서버 전체 요약 원문](../evidence/local_placement_server_full_20261004.txt), [분석](../LOCAL_PLACEMENT_SERVER_FINDINGS_20261004.md), [수식](../../research/local_energy_relations/placement/MODEL_MATH.md).

고정 C마다 base1개 + E/J/E+J × hidden/output/all = 10조건, 두 C로20조건이다. 같은 2층·hidden64·500epoch를 유지했다. Tuning540 + final300 = 840학습·420,000 update를 보고한 전체 요약을 수령했다. 이 첨부에는 completion.json 자체와 전체 원시 CSV/checkpoint가 없다.

| 데이터 | C | base % | 첫 층 E+J % | 출력층 E+J % | 두 층 E+J % |
| --- | --- | ---: | ---: | ---: | ---: |
| Cora | unit | 76.22 | 76.34 | 75.96 | 76.04 |
| Cora | local_degree | 74.34 | 74.42 | 74.38 | 74.56 |
| CiteSeer | unit | 67.24 | 67.14 | 64.94 | 65.06 |
| CiteSeer | local_degree | 64.90 | 64.84 | 63.70 | 63.62 |
| PubMed | unit | 79.60 | 79.54 | 79.32 | 79.44 |
| PubMed | local_degree | 78.94 | 78.90 | 78.96 | 79.04 |

첫 층 E+J 여섯 비교의 accuracy95% 구간은 모두0을 포함한다. 같은 C base 대비54개 비교 중 accuracy 구간 전체가 양수인 것은 PubMed `local_degree__between__output` 하나: +0.24pp [+0.051693,+0.42831], CE−0.0026657 [−0.0046963,−0.00063507]이다. 절대 accuracy79.18%는 unit/base79.60%보다 낮다. 전체 후보를 본 뒤 이 하나를 최종 우승 모델로 선택하지 않는다.

반대로 CiteSeer 출력층 J는 unit−2.78pp, local_degree−2.08pp로 구간이 모두 음수다. Cora 첫 층 E는 unit/local_degree에서 CE−0.0027974/−0.0023179로 구간이 음수지만 accuracy 구간은0을 포함한다. 위치 변경의 보편적 개선은 확인되지 않았다.

채널별 E/J 대조는 당시 문서의 **제안**으로 남았다. 구현·본학습 완료로 세지 않는다. 후속 논의는 scalar lift보다 실제 copy 상태 전달과 큰 copy 에너지의 정의를 먼저 바꾸는 방향으로 진행됐다.

## 6. C1: copy 상태와 큰 이차 에너지에서 유도한 전달

근거: [서버 fixed 요약 원문](../evidence/local_context_coupling_server_full_20261004.txt), [분석](../LOCAL_CONTEXT_COUPLING_SERVER_FINDINGS_20261004.md), [실제 수식](../../research/local_context_coupling/MODEL_MATH.md).

같은 물리 노드라도 local마다 별도 copy `(v,a)`를 둔다. `R`은 물리 특징을 copy에 복제하고 `M=(RᵀR)⁻¹Rᵀ`는 같은 물리 ID의 copy를 균등 평균한다. 내부 `A=blockdiag(B_vᵀC_vB_v)`, 교차 `K=JᵀWJ`는 인접 중심 pair의 공통 물리 노드 copy를 연결한다. Fixed W=1이며 C도 두 고정 규칙이다.

이 모델의 큰 에너지는 **copy 공간**의 `½tr(Yᵀ(A+λK)Y)`다. Scalar prediction의 `J=⟨d_v,d_u⟩`와 여기의 copy 차이 에너지는 다른 정의다. 원래 물리 노드 ID를 하나로 합친 union graph의 라플라시안과 자동으로 같아지지도 않는다.

`KR=MK=0`이므로 cross 직후 평균하면 cross 영향이 사라진다. 주 후보는 같은 내부 A를 앞뒤에 적용한다:

\[
T_\rho H=M(I-\eta A)(I-\gamma\rho K)(I-\eta A)RH,
\qquad
T_\rho H-T_0H=-\gamma\eta^2\rho MAKARH.
\]

Off도 같은 내부2회를 유지한다. 보조 persistent joint 반복은 copy를 유지하며 `M(I−s(A+λK))^kRH`를 계산한다. k1/2에는 cross 차이가0이고 k3에 `−s³λMAKARH`가 처음 남는다. Sandwich와 동일한 연산이라고 설명하지 않는다.

서버 고정 감사는201 graph·parameter0·update0을 보고했다. Citation 분류는 수행하지 않았다.

| 실제 검사 | 원문 최대값 |
| --- | ---: |
| 정확한 sandwich 차이식의 상대 residual | 6.99393e−17 |
| cross 직후 즉시 merge의 on/off 차이 norm | 4.37073e−15 |
| persistent1 / persistent2 on/off 차이 norm | 4.14771e−15 / 4.53098e−15 |
| context·예측 차이·실측 차이가 threshold를 넘은 graph/C/state 셀 | 각각1,206 = 201×2×3 |

Citation H0/H1/H2의 `||T1H−T0H||/||T0H||` 범위는 unit0.00111344–0.00523755%, local_degree0.0274956–0.0387885%였다. 수치 오차보다 크지만 표현에 비해 작았다. Unit에서 persistent3 변화는 sandwich의1/8이며, 같은 mixed action의 계수 차이로 설명된다.

이는 메커니즘의 작동 증거다. 모든 입력에서 nonzero, 분류 개선, 독립 표현력, 사이클 복원 또는 신규성의 증거는 아니다.

## 7. C2: copy coupling의 첫 실제 분류 FULL

근거: [completion·전체 서버 요약](../evidence/local_context_classification_server_full_20261004.txt), [결과 분석](../LOCAL_CONTEXT_CLASSIFICATION_SERVER_FINDINGS_20261004.md), [설계](../../research/local_context_coupling/classification/EXPERIMENT_DESIGN.md).

두 고정 C × off/fixed/learned의6조건이다. Fixedρ=1, offρ=0에는 θ가 없다. Learned는 seed마다 θ 하나를 두 macro 층에서 공유하며 `ρ=sigmoid(θ)`, 초기ρ=.5다. **학습된 C, attention edge score 또는 learned 개별 W의 실험이 아니다.**

완료 기록: tuning162 + final90 = 252학습, 126,000 update, metric270 / frozen1,080 / branch900행, 전체2,067.5805초. Actual_data=true, 동결 평가 update0. 이 시간을 전체 update 수로 나눠 서버의 개별 epoch 시간으로 사용하지 않는다.

### 실제 test 평균: 모든6조건

Final5seed의 평균 accuracy(%)다. 원문의 std·CE·paired 구간은 위 원문에 그대로 있다. 모든 조건의 선택 LR는.01, train accuracy 평균은100%였다.

| 데이터 | unit off | unit fixed | unit learned | local_degree off | local_degree fixed | local_degree learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 58.66 | 58.72 | 58.60 | 79.38 | 79.46 | 79.38 |
| CiteSeer | 57.62 | 57.46 | 57.32 | 70.64 | 70.60 | 70.60 |
| PubMed | 73.16 | 73.16 | 73.16 | 77.66 | 77.66 | 77.66 |

같은 C의 fixed−off / learned−off / learned−fixed accuracy95% 구간은 모두0을 포함한다. 일부 CE 비교는 작은 개선을 보인다: CiteSeer/unit fixed−off−0.000135732, Cora/local_degree−0.000224400, PubMed/local_degree−0.0000701785이며 해당 구간은 음수다. Learned가 fixed보다 일관되게 좋지는 않았다.

교차 off에서도 local_degree−unit accuracy 차이가 Cora+20.72pp, CiteSeer+13.02pp, PubMed+4.50pp다. 이 차이는 C와 그 C가 정한 η·기본 전파를 함께 바꾼 결과다. Cross 성과나 learned C의 성과로 설명하지 않는다. Copy off 모델을 표준 GCN이라고 두지도 않는다.

| 데이터 | 선택된 unit learned ρ 평균 | 선택된 local_degree learned ρ 평균 |
| --- | ---: | ---: |
| Cora | 0.857757 | 0.371410 |
| CiteSeer | 0.902937 | 0.0659821 |
| PubMed | 0.469102 | 0.266711 |

ρ는 초기값에서 변했다. 하지만 서버 epoch별 θ gradient/update CSV는 수령하지 않아 크기·추세를 독립 검토하지 않았다. 선택된 실제 hidden의 matched 상대 변화는 fixed/unit0.00110–0.00444%, fixed/local_degree0.01289–0.01584%, learned 두 C0.000518–0.00531%로 작았다.

같은 learned checkpoint에서 두 층을 gain1로 바꾼 local_degree ΔCE는 Cora−0.0000486016, CiteSeer−0.0000303984, PubMed−0.0000319481이었다. 고정 checkpoint의 사용 반응이며 fixed로 처음부터 학습한 모델의 이득을 대신하지 않는다. 작동·학습·추가 정확도 효과를 구분한다.

## 8. C3: 정규화 비교의 서버 FULL 결과

근거: [서버 completion·전체 요약 원문](../evidence/local_context_normalization_server_full_20261005.txt), [전체 결과 해설](08_NORMALIZATION_FULL_RESULTS.md), [서버 결과 분석](../LOCAL_CONTEXT_NORMALIZATION_SERVER_FINDINGS_20261005.md), [설계](../../research/local_context_coupling/normalization/EXPERIMENT_DESIGN.md), [수식](../../research/local_context_coupling/normalization/MODEL_MATH.md), [검증 기록](../../research/local_context_coupling/normalization/VERIFICATION.md).

약한 cross 작용이 graph 전체 bound의 영향인지, 관계 자체의 task 기여가 작은지 분리하기 위해 기존 결과를 수정하지 않고 별도20조건을 모두 다시 학습했다. 원인은 결과를 읽기 전에 확정하지 않았다.

\[
T_\rho=M(I-S)(I-\rho G)(I-S)R,
\qquad T_\rho-T_0=-\rho MSGSR.
\]

S는 graph ηA 또는 각 ego의 local η_vL_v, G는 graph γK 또는 unit K endpoint degree를 기준으로 한 대칭 edge weight다. 같은 S를 앞뒤에 사용한다. G는 같은 물리 ID의 copy끼리만 연결하여 `GR=MG=0`을 유지한다. Applied S/G weighted degree는.5 이하다. C와 개별 W는 학습하지 않는다.

조건 ID는 `C__intra__cross__variant`. C2 × intra2 × `[none/off,graph/fixed,graph/learned,edge/fixed,edge/learned]` =20이다. Off를 cross 정책별로 중복 세지 않는다. 52개 직접 비교와12개 paired 상호작용, 전체 frozen 개입을 사전에 정했다.

### 수령한 서버 FULL 완료 범위

| 항목 | completion 보고 |
| --- | --- |
| 학습 범위 | tuning540 + final300 =840학습, run당500epoch |
| 실제 update | 새 optimizer update420,000 |
| 평가·진단 행 | primary metric900 / frozen4,320 / branch3,480 |
| 완료·데이터 | completed=true, profile=full, actual_data=true, code_and_graphs_preserved=true |
| 동결 평가 update | 0 |
| 전체 elapsed | 6,820.536256초: 약1시간53분41초 |

보존 원문은86,652bytes, SHA-256 `32e7237f8ee421f49f090c5e0dad49e73a2b4fbfc218c1b82881d3fa6dcaed33`이다. [추출 JSON](../evidence/local_context_normalization_summary_20261005.json)은 표시된 요약 수치를 구조화한 자료이며, 원시5seed CSV에서 구간을 다시 계산한 자료가 아니다. 이번 요약의 이전6조건도 새 학습 결과다. C2 수치와 섞거나 독립 반복으로 추가 집계하지 않는다.

원문에는 전체60개 dataset/condition의 평균·std, test 직접 비교312행과 상호작용72행, 선택된 원래 분기의 층별 평균120행이 있다. Completion의4,320 frozen 행과3,480 branch 행이 전부 이 첨부에 실린 것은 아니다. 개입별 결과·epoch gradient/update·원시 seed CSV/checkpoint·hardware/source/data manifest는 수령하지 않았다.

### 내부 정규화: cross off만 비교해도 큰 차이가 난다

아래는 다른 조건을 고정한 `intra local−graph`다. Accuracy 차이는pp이며 CE에는 L2를 넣지 않는다.

| C | 데이터 | off Δaccuracy와95% 구간 | off ΔCE |
| --- | --- | ---: | ---: |
| unit | Cora | +21.46 [+20.3861,+22.5339] | −0.532157 |
| unit | CiteSeer | +12.74 [+12.0343,+13.4457] | −0.247475 |
| unit | PubMed | +4.64 [+3.37802,+5.90199] | −0.0901897 |
| local_degree | Cora | +0.780 [+0.150534,+1.40947] | −0.0412238 |
| local_degree | CiteSeer | −0.180 [−0.501397,+0.141395] | −0.0171181 |
| local_degree | PubMed | +0.240 [−0.291169,+0.771167] | −0.00410881 |

위6개 CE 구간은 모두 음수다. 전체 `intra local−graph`의30개 accuracy 비교에서는20개 구간이 양수다: unit의15개와 Cora/local_degree의5개다. 따라서 unit graph 조건의 낮은 accuracy를 cross 부재의 결과로 설명할 수 없다. Local 정규화에서는 `local_degree−unit` accuracy15개 구간이 모두0을 포함한다. C2의 큰 C 차이가 그대로 유지되지 않았으며 C 자체의 효과와 적용 step의 효과를 구분해야 한다.

### 교차 작용: CE의 일부 개선과 accuracy 결과를 구분한다

- **같은 C·intra의 fixed−off24개와 learned−off24개 accuracy 구간은 모두0을 포함한다.** 추가 cross의 정확도 개선이나 동등성을 입증하지 않았다.
- Fixed−off CE는24개 중12개, learned−off CE는24개 중6개 구간이 음수다. Cora의 local/edge/fixed는 unit−0.00831550 [−0.00935171,−0.00727930], local_degree−0.00685143 [−0.00755190,−0.00615097]다. CiteSeer의 같은 비교는 unit−0.00265635, local_degree−0.00236418로 구간이 음수다. PubMed/local의 cross−off CE 구간은 모두0을 포함한다.
- `edge−graph` CE는24개 중11개 구간이 음수다. Accuracy는 Cora/local_degree/local/fixed의 +0.20pp [+0.00367451,+0.396325] 한 구간만 양수다. 이는 두 cross 정책끼리의 비교이며 같은 off 대비 accuracy 개선의 증거는 아니다.
- Learned−fixed CE는24개 중10개 구간이 양수이고 음수인 구간은 없다. Learned 강도가 fixed보다 일관되게 좋은 test CE를 내지는 않았다. 이 결과만으로 학습 경로가 끊겼다고 판단하지 않는다.
- 상호작용36개 accuracy 구간에서는 CiteSeer/unit/graph/fixed의 내부 정규화에 따른 cross 기여 변화 +0.16pp [+0.0184294,+0.301571] 하나만 양수다. Graph의 cross−off−0.14pp가 local의+0.02pp로 달라진 비교다. Local cross−off 자체의 구간은0을 포함한다. CE 상호작용은10개 구간이 음수이며 PubMed는12개 모두0을 포함한다.

모든 구간은 final5seed paired t95% 구간이며 다중 비교 보정이 없다. 단일한 얇은 양수 구간을 전체 후보의 검증된 우위로 읽지 않는다. 선택된ρ·표현 norm은 원문의120개 평균 행으로 확인할 수 있지만, 전체 개입 효과와 학습 중 gradient 추세를 대신하지 않는다.

### 서버 FULL 이전의 구현·DEBUG 검증

| 항목 | 실제 확인 |
| --- | --- |
| 구현·정적 검사 | sparse 정규화, 분류기, 학습/선택/동결 평가/보고서/엄격 resume, AST·CLI import/help |
| 단위 테스트 | 222개 통과: 모델113 / 학습38 / 계약·입력·평가 잠금31 / 보고서40 |
| 전체 DEBUG | fixture3개, 20조건, tuning240 + final120 =360학습·1,080 update, 221.4062초 |
| DEBUG 평가 | metric360 / frozen1,728 / branch1,392행 |
| DEBUG learned 기록 | 432개 epoch/seed 행 모두 실제 CE의 θ gradient norm과 실제 optimizer update norm이0보다 큼 |
| DEBUG algebra | 최대 frozen formula relative error9.2034121e−8, S/G 최대 degree각.5 |
| 완료 checkpoint 재사용 | 새 폴더, 같은 coverage, 추가 학습0, 29.0130초 |
| 재사용 비교 | 모델 hash 동일, accuracy차이0, float32 CE차이 최대2.3841858e−7 |

로컬 원본: [DEBUG 완료](../../results/local-context-normalization-DEBUG-20261004-01/completion.json), [DEBUG 보고서](../../results/local-context-normalization-DEBUG-20261004-01/LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md), [재사용 완료](../../results/local-context-normalization-resume-DEBUG-20261004-01/completion.json). ZIP에 포함된 구체 파일은 패키지 manifest를 따른다.

DEBUG는 Windows RTX5070Ti16GiB, 노드24/30/36·특징12·클래스3·hidden8·3epoch의 별도 fixture다. FULL hidden64·500epoch를 바꾼 값이 아니다. 이 DEBUG accuracy와 시간은 실제 citation 성능이나 A6000 FULL 처리량의 추정치가 아니다.

## 9. 현재 결론과 남은 판단

1. 고정 known-C·전체 관측에서는 실제 constrained q를 복원할 수 있었다. Cycle나 receipt 상쇄량만으로 영구 정보 손실을 주장할 수 없다.
2. 스칼라 E/J lift는 학습·사용됐지만 보편적 정확도 개선이 확인되지 않았다. 이것이 이차 에너지·쌍선형 관계 전체의 실패를 뜻하지 않는다.
3. Copy sandwich는 교차 작용을 merge 뒤에 남기는 정확한 구조이며 첫 실제 분류도 완료됐다. 동일 C의 추가 accuracy 효과는 확인되지 않았다.
4. 정규화 FULL에서는 cross off 상태에서도 unit의 내부 graph→local 변경이 큰 accuracy 차이를 만들었다. 교차 정규화의 일부 CE 개선은 확인했지만 같은 C·intra의 cross 추가 accuracy 개선은 확인되지 않았다.
5. 정규화 완료·요약은 수령했고 서버 원시 산출물 전체의 독립 재평가는 하지 않았다. Learned C·새 그래프 일반화·정보 복원·신규성은 여전히 입증되지 않았다. [남은 질문과 증거 범위](05_EVIDENCE_AND_OPEN_QUESTIONS.md)를 함께 검토한다.
