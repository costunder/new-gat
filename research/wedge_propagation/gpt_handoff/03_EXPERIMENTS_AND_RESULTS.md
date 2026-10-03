# 현재까지의 실험과 결과

## 1. 현재 결론부터

**경로 가중치 생성기는 지정한 합성 teacher의 메시지 규칙을 학습하고 새 입력에 재사용했다.
하지만 실제 노드 분류에서 학습한 C가 고정 C=1보다 유리하다는 근거는 아직 없다.**

노드별 정규화로 다시 학습한 모델은 기존 global 정규화 모델보다 개선됐다.
그 개선을 C 가중치 배치의 이득으로 바로 해석할 수는 없다.
고정 C=1 대조와 같은 메시지 크기의 C 교체 결과를 함께 읽어야 한다.
둘째 층의 학습 C를 같은 크기의 C=1 메시지로 바꾼 추가 확인에서도 가중치 배치의 이득은 확인되지 않았다.

이 문서는 현재 독립 `wedge_propagation` 트랙만 다룬다.
이전 Conductance/CGAT/Cycle PE/output initialization 실험의 구조·예산·결과를 여기에 합치지 않는다.
초기 폐기된 698-run 계획 대신 실제 실행한 각 패키지의 full config를 기준으로 정리했다.

## 2. 결과의 출처와 검증 범위

서버 결과는 사용자가 붙여넣은 `completion.json`, 요약 Markdown, 터미널 로그와 일부 CSV 행에 근거한다.
첨부된 원문과 저장소의 `SERVER_*_RESULTS.md`를 함께 읽는다.
**서버 원본 CSV·NPZ·checkpoint 전체를 로컬로 가져와 독립 재평가한 상태는 아니다.**
서버 실행기가 수행했다고 출력한 hash·coverage·replay 검사와 로컬에서 직접 수행한 검사를 구분한다.
Local 단위 테스트·DEBUG pipeline·실제 데이터의 무갱신 gradient 검사는 구현 연결의 증거다.
별도 DEBUG 결과를 citation 본학습 성능으로 사용하지 않는다.

| 자료 | 확인 가능한 증거 |
| --- | --- |
| 사용자 제공 원문 | 서버 완료 상태, 출력된 수치, 해당 출력에 포함된 범위·보존 검사 |
| `SERVER_*_RESULTS.md` | 원문 수치의 정리와 해석 범위 |
| 각 `config_full.json`·수식·코드 | 실제 모델·데이터·학습·평가 계약 |
| 각 `VERIFICATION.md`·단위 테스트·DEBUG | 로컬에서 직접 확인한 구현과 실행 연결 |
| 서버 원본 CSV·NPZ·checkpoint | 세부 재집계·모델 재평가에 필요한 자료이며 서버에 보관됨 |

## 3. 전체 순서와 예산

| 단계 | 질문 | 실제 처리 | 현재 상태 |
| --- | --- | --- | --- |
| 0 | 경로 연산 수식과 구현이 일치하는가? | 7종 대수 검사, dense·희소 출력/미분 참조 | 검사 완료 |
| 1 | 고정 Q가 L·L²와 어떻게 다른가? | 198개 그래프·3,168개 입력, 학습 0 | 서버 full 완료 |
| 2 | 공유 C 생성 규칙이 합성 메시지를 학습하는가? | 531개 그래프, 3 target·5 조건, gate 30 seed 모델 × 500 epoch | 서버 full 완료 |
| 3 | 고정한 모델이 새 특징과 특징 배율에 통하는가? | 같은 531 topology, fresh 특징과 5개 배율, 학습 0 | 서버 full 평가 완료 |
| 3.1 | C 입력 RMS 정규화가 배율 문제와 회수 오차를 바꾸는가? | Raw 고정, normalized gate 30 seed 모델 × 500 epoch | 서버 full 완료 |
| 4 | Teacher 없이 실제 분류 CE로 학습한 C가 유익한가? | 3 citation·8조건·336 runs·168,000 updates | 서버 full 완료 |
| 4.1 | C 배치 효과와 분기 크기 효과를 구분할 수 있는가? | 120 final 모델 재현, learned 30개 개입, 학습 0 | 서버 full 완료 |
| 4.1 CSV 분석 | 작은 분기 크기는 어떤 인자로 구성되는가? | 기존 CSV 전체 분석, 모델 forward 0·학습 0 | 서버 full 완료 |
| 4.2 | C 의존 노드별 정규화로 재학습하면 개선되는가? | 3 citation·5조건·210 runs·105,000 updates | 서버 full 완료 |
| 둘째 층 C=1 확인 | 현재 둘째 층 C 배치가 같은 크기의 C=1보다 유익한가? | 기존 4.2의 저장된 paired 개입 12행 확인 | 추가 학습 없이 확인 |

Experiment 2/3.1의 각 30개는 3 target × learned/random-pair × 5 seed다.
Seed별 500 update이므로 각 새 학습의 독립 모델 update는 15,000개다.
다섯 모델을 한 번에 계산하는 packed optimizer 호출과 독립 모델 update 수는 다르다.
First/polynomial/fixed의 9개 train-only scalar fit은 결정적 최소제곱이며 optimizer 학습을 하지 않는다.
Experiment 3.1은 이 scalar fit과 raw checkpoint를 재학습하지 않았다.

## 4. 공통 그래프 정의

단순 무방향 그래프의 물리 엣지를 한 번씩 저장한다. 각 중심 j의 서로 다른 이웃 i,k를
unordered pair로 한 번씩 사용하며 삼각형 안의 wedge도 포함한다.

\[
L=B^\top B,\qquad
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,\qquad
Q=A^\top A.
\]

\[
g_1=x_j-x_i,\quad g_2=x_k-x_j,\quad
(AX)_p=x_i-2x_j+x_k=g_2-g_1.
\]

현재 C를 고정하면 `AᵀCA`는 대칭 PSD이고 `XᵀAᵀCAX=Σ c_p(AX)_p²`라는 이차형식을 정의한다.
그 고정 C에 대해 메시지 `AᵀCAX`는 이 에너지 gradient의 절반이다.
**C=C(X)인 실제 모델의 메시지는 C의 입력 미분항까지 포함한 전체 에너지 gradient와 동일하지 않다.**
이차형식이 있다는 사실만으로 소거된 엣지 메시지나 사이클 정보를 복원했다고 주장하지 않는다.

## 5. Experiment 0/1 — 고정 연산자의 차이

학습 C·optimizer·classifier 없이 LX·L²X·QX, 에너지·스펙트럼·nullspace·연산 잔차를 측정했다.
Float64 full은 노드 수 20/30/40/60/80/100, cycle/star/grid/ER/tree/tree+chord,
총 198개 그래프와 그래프당 독립 Gaussian scalar 특징 16개다.

\[
\boxed{Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B.}
\]

7종 검사와 전체 full 그래프에서 이 항등식의 absolute error가 0으로 보고됐다.
다음 잔차는 그래프마다 Q를 두 고정 basis에 최적으로 맞춘 진단이다.

\[
r_G=\min_{a,b}\|Q-aL-bL^2\|_F/\|Q\|_F.
\]

| Family | 그래프 수 | r_G 중앙값 | Spectral norm을 맞춘 Q/L² 출력 상대차 중앙값 |
| --- | ---: | ---: | ---: |
| cycle | 6 | 1.84459e−14 | 0 |
| star | 6 | 1.65675e−13 | 0.123054 |
| grid | 6 | 0.0950461 | 0.130971 |
| ER | 60 | 0.10055 | 0.119721 |
| tree | 60 | 0.14219 | 0.198462 |
| tree+chord | 60 | 0.129991 | 0.1492 |

Cycle에서는 Q=L²다. Star도 L과 L²의 조합으로 환원된다.
나머지 네 family에는 두 basis에 맞추고도 잔차가 남았다.
이는 고정 연산의 차이이며 정보 손실률·예측 정확도·학습된 C의 이득을 측정한 값이 아니다.
7종 검사 통과도 전체 학습 완료를 뜻하지 않는다.

출처: [서버 고정 결과](../SERVER_FIXED_RESULTS.md), [고정 full README](../README.md).

## 6. Experiment 2 — 합성 teacher 메시지 학습

531개 그래프의 split은 train/validation/ID/size OOD/family OOD/family+size OOD
각각 240/60/120/90/12/9개다. 그래프당 독립 scalar 특징 16개, 총 8,496개다.
Train·validation·ID 크기는 20/30/40/50, size OOD는 60/80/100이다.
Train family는 ER/tree/tree+chord이고 family OOD는 cycle/star/grid다.

세 target은 LX, L²X, `AᵀC*(X)AX`다. Path teacher는 다음 규칙을 사용한다.

\[
r_p^*=\frac{g_1g_2}{|g_1||g_2|+10^{-8}}
+\frac{|g_2-g_1|}{|g_1|+|g_2|+10^{-8}},\quad
c_p^*=\frac{e^{\tanh r_p^*}}{\operatorname{mean}_{q\in G}e^{\tanh r_q^*}}.
\]

Student는 `[|g₁|+|g₂|, g₁g₂, |g₂−g₁|, (|g₁|−|g₂|)²]`를
공유 `4→64→1` MLP에 넣고 exp(tanh)/graph mean으로 양의 C를 만든다.
출력은 `βAᵀCθ(X)AX`이고 자유 scalar β와 gate를 메시지 오차로 학습한다.
Teacher C의 label·상관·weight loss를 학습에 제공하지 않는다.
16개 열은 독립 scalar 실현이며 vector-valued 분류 channel과 구분한다.

First는 aLX, polynomial은 aLX+bL²X, fixed는 βQX를 train에서 fit한다.
Learned/random-pair는 각 5 seed·500 epoch·Adam lr 0.003으로 학습했다.
학습 physical graph batch는 전체 train 240개, accumulation 1이며 모든 scalar 실현을 함께 계산했다.
Student는 float32, 저장한 teacher 참조는 float64다.

### Path target의 메시지 상대오차

실현 평균 → 동일 비중 그래프 평균 → 학습 seed 평균±표본 std 순서다.
이 오차를 `1−분류 정확도`로 바꾸어 읽지 않는다.

| Split | Polynomial | Fixed Q | Learned | Random-pair |
| --- | ---: | ---: | ---: | ---: |
| ID | 0.250175 | 0.171868 | 0.047140±0.013491 | 0.555487±0.000533 |
| size OOD | 0.246178 | 0.166230 | 0.046147±0.012647 | 0.554482±0.000441 |
| family OOD | 0.317892 | 0.193912 | 0.049000±0.013141 | 0.616995±0.005943 |
| family+size OOD | 0.288272 | 0.173095 | 0.042535±0.011215 | 0.582529±0.003682 |

LX target의 first/polynomial과 L²X target의 polynomial은 모든 split에서 오차 0이었다.
Learned-wedge는 모든 target에서 우세하지 않았으며 LX target을 잘 맞추지 못했다.
Path teacher에서는 learned의 출력 회수가 fixed와 polynomial보다 좋았고 새 그래프에서도 유지됐다.
Random-pair는 gate 용량을 맞췄지만 support·hop·엣지 사용 빈도도 바뀌므로
이 악화를 경로 연속성 하나의 인과 효과로 확정하지 않는다.

출처: [서버 learned 결과](../SERVER_LEARNED_RESULTS.md), [실제 teacher/student 수식](../learned/MODEL_MATH.md).

## 7. Experiment 3 — 같은 topology의 새 특징과 배율

Experiment 2의 선택된 gate checkpoint 6개×5seed와 scalar fit 9개를 고정했다.
동일한 531개 topology에서 독립적인 fresh Gaussian 특징을 만들고
배율 0.25/0.5/1/2/4를 같은 fresh 특징에 적용했다.
각 배율에서 teacher C와 세 target을 다시 계산했다. 새로운 그래프를 추가하거나 재학습하지 않았다.

| Path target, fresh 배율 1 | Learned 오차 | Fixed Q 오차 |
| --- | ---: | ---: |
| ID | 0.0475536 | 0.174729 |
| size OOD | 0.0465101 | 0.166116 |
| family OOD | 0.0512217 | 0.203409 |
| family+size OOD | 0.0473821 | 0.179715 |

Original ID에서 teacher C와의 상대오차는 0.123585±0.0286, 상관은 0.947717±0.0231이었다.
같은 learned 모델의 original ID 메시지 오차는 원래 0.0471402,
C=1 0.383920, C 위치 shuffle 0.439846이었다.
이 frozen 개입의 β는 고정한 값이며 β를 별도로 적합한 fixed Q와 다른 조건이다.
C 평균 1과 높은 상관만으로 개별 C를 유일하게 복원했다고 판단하지 않는다.

Fresh ID 특징을 4배로 키우면 student C 변화 0.369839,
메시지 비례성 오차 0.306899, 예측 상대오차 0.302559로 커졌다.
Teacher의 해당 변화는 약 6.82e−7/1.21e−7이었다.
새 특징의 배율 1에서는 회수가 유지됐지만 입력 크기 변화에는 raw gate가 민감했다.

새 학습·optimizer update·checkpoint 선택은 0이고, 메시지 124,254행·scale 103,545행이다.
이 행 수는 같은 그래프에 대한 대응 처리 수이며 독립 그래프 표본 수가 아니다.
출처: [서버 generalization 결과](../SERVER_GENERALIZATION_RESULTS.md).

## 8. Experiment 3.1 — C 입력 RMS 정규화의 재학습

C를 만들 때만 물리 엣지 차분의 graph/field별 RMS로 g₁·g₂를 나눈다.
실제 AX 메시지는 원래 입력의 차분을 사용한다.
동일 531개 그래프·16실현·hidden64·500epoch·5seed·batch240을 유지한 normalized gate를 새로 학습했다.
Raw checkpoint와 scalar 대조는 고정하고 Experiment 3의 fresh 특징·다섯 배율을 재사용했다.

\[
\sigma_{G,r}=\sqrt{\|BX_r\|_2^2/|E_G|},\qquad
\tilde g_1=g_1/\sigma_{G,r},\quad\tilde g_2=g_2/\sigma_{G,r}.
\]

양의 비영 RMS에는 epsilon을 추가하지 않는다. 차분이 모두 0이거나 엣지가 없으면 σ=1이다.
이상적인 산술에서 양의 배율 a에 대해 `C(aX)=C(X)`, `M(aX)=aM(X)`다.
이는 설계한 성질이며 회수 정확도 개선과 별도로 확인했다.

| Fresh path target 배율 1 | Raw 오차 | Normalized 오차 | Fixed Q |
| --- | ---: | ---: | ---: |
| ID | 0.0475536±0.0139 | 0.0515799±0.00357 | 0.174729 |
| size OOD | 0.0465101±0.0128 | 0.0525378±0.00365 | 0.166116 |
| family OOD | 0.0512217±0.0143 | 0.0532518±0.00297 | 0.203409 |
| family+size OOD | 0.0473821±0.0130 | 0.0508475±0.00274 | 0.179715 |

다섯 관측 배율의 최대 변화를 graph/seed별로 계산한 fresh ID 진단은 다음과 같다.

| Variant | C 변화 | 메시지 비례성 오차 |
| --- | ---: | ---: |
| Raw | 0.371238±0.135 | 0.306902±0.0969 |
| Normalized | 1.95587e−7±1.58e−8 | 2.06874e−7±1.06e−8 |

배율 안정성과 seed 변동은 개선됐지만 평균 메시지 회수 오차는 raw보다 높았다.
이를 정규화가 학습 정확도를 개선했다고 설명하지 않는다.
Original metric replay는 11,349행에서 verified=true로 보고됐지만 최대 absolute metric 차이는
0.002053013486, target-normalized RMSE 차이는 8.21676e−5였다. 수치가 완전히 동일하다고 기록하지 않는다.
메시지/scale/개입 행은 각각 248,508/207,090/159,300개이며 raw update와 test update는 0이다.
출처: [서버 scale-normalization 결과](../SERVER_SCALE_NORMALIZATION_RESULTS.md).

## 9. Experiment 4 — 실제 노드 분류

Synthetic teacher를 제거하고 train node의 분류 CE로 projection·C gate·α·β를 새로 학습했다.
Public fixed split을 사용하며 모든 노드·물리 엣지·wedge를 유지한다.

| 데이터 | N | F | 클래스 | 물리 E | Wedge P | Train/validation/test |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 2,708 | 1,433 | 7 | 5,278 | 52,301 | 140/500/1,000 |
| CiteSeer | 3,327 | 3,703 | 6 | 4,552 | 26,918 | 120/500/1,000 |
| PubMed | 19,717 | 500 | 3 | 44,324 | 699,342 | 60/500/1,000 |

두 층 `입력→64→클래스`, projection bias 없음, 각 projection 전 dropout0.5,
첫 전파 뒤 ReLU를 사용한다. C는 현재 vector-valued 투영 Z의 `4F→64→1` gate로 생성한다.
한 path C를 모든 channel에 공유한다. Raw σ=1, RMS는 해당 graph/seed/layer의 물리 엣지·전체 channel RMS다.

\[
\bar L=\tfrac12S_dLS_d,\quad
\bar Q=\tfrac13S_QQS_Q,\quad S_Q=\operatorname{diag}(Q)^{-1/2},
\]

\[
\kappa(C)=\max_{q_v>0}\frac{\operatorname{diag}(A^\top CA)_v}{q_v},\quad
M_C=\frac{S_QA^\top CA S_QZ}{3\kappa(C)},\quad
U=Z-\alpha\bar LZ-\beta M_C.
\]

α=t(1−r), β=tr, t=sigmoid(u), r=sigmoid(v)이며 초기 α=.5, β=.25다.
Polynomial 조건은 이차 분기를 L̄²Z로 바꾼다.
Fixed+nodeMLP는 실제 node MLP를 더하여 learned gate와 파라미터 수를 맞춘 용량 대조다.
Standard GCN은 자기 연결을 포함한 대칭 정규화 인접행렬 전파를 사용한다.

8조건마다 LR .001/.003/.01×tuning seed101/202/303에서 validation으로 선택하고,
독립 final seed11/23/37/53/71을 학습했다. 각 500epoch, tuning216+final120=336run·168,000update다.
Tuning에서 test를 읽지 않고 모든 final 선택을 잠근 뒤 평가했다.

### Test 정확도 평균(%)

| 조건 | Cora | CiteSeer | PubMed |
| --- | ---: | ---: | ---: |
| MLP | 57.14 | 56.22 | 72.28 |
| First order | 70.78 | 63.78 | 75.10 |
| Polynomial 2 | 72.54 | 66.02 | 75.72 |
| Fixed wedge | 73.56 | 64.94 | 75.46 |
| Learned raw | 70.74 | 63.74 | 75.06 |
| Learned RMS | 70.52 | 63.86 | 74.68 |
| Fixed wedge + node MLP | 63.70 | 55.92 | 71.18 |
| Standard GCN | 81.88 | 71.12 | 79.06 |

Learned raw/RMS는 fixed와 polynomial보다 평균 정확도가 낮았고 GCN이 가장 높았다.
C mean=1은 생성식이며 원래 C std는 raw 약 .058–.413, RMS 약 .319–.688이었다.
따라서 모든 C가 1이라는 결과는 아니지만 C의 비균일성이 분류 이득을 입증하지는 않는다.
κ를 유지한 C=1/shuffle의 정확도 변화가 작았고, C=1에서 κ를 재계산하면 +.40–1.32pp였다.
κ 유지와 실제 메시지 norm 유지는 다르므로 이를 구분하는 다음 진단을 수행했다.

원래 metric360행·개입2,970행·scale2,100행·층 진단2,220행, 완료 시간3,970.394초다.
RMS의 배율 안정성은 작동했지만 분류 개선으로 이어지지 않았다.
출처: [서버 classification 결과](../SERVER_CLASSIFICATION_RESULTS.md).

## 10. Experiment 4.1 — C 배치와 메시지 크기의 고정 진단

기존 120 final 모델의 원래 metric을 재현한 뒤 learned raw/RMS 30개에 개입했다.
기존 두 층·hidden64·전체 그래프·경로·10 manifest를 유지하고 새로운 학습은 하지 않았다.
C 배치를 바꿀 때 실제 메시지 norm을 맞추는 처리와,
true C를 유지한 채 κ 분모를 1로 바꾸는 처리를 layer_0/layer_1/both에서 비교했다.
별도 fixed-Z 진단은 원래 깨끗한 Z를 고정했다.

| Both 개입 | 평균 test accuracy 변화 범위 | 함께 확인한 결과 |
| --- | ---: | --- |
| C=1, 실제 메시지 norm 일치 | −.020~+.120pp | 여섯 paired 95% 구간 모두 0 포함 |
| C shuffle, 실제 메시지 norm 일치 | −.034~+.090pp | 여섯 paired 95% 구간 모두 0 포함 |
| True C, κ 분모 1 | +.320~+1.260pp | 여섯 평균 CE 모두 +.00493~+.01537 악화 |
| C=1, 분모 1 | +.400~+1.320pp | 여섯 평균 CE 모두 악화 |

C는 메시지 방향을 실제로 바꿨다. 예를 들어 Cora/RMS의 fixed-Z C=1 cosine은
첫/둘째 층 약 .848/.889, shuffle은 .831/.861이었다.
하지만 같은 norm에서 learned C 배치의 분류 이득은 뚜렷하지 않았다.
95% 구간에 0이 포함된다고 정확한 동등성을 증명한 것은 아니다.
Accuracy 상승과 CE 악화가 함께 나왔으므로 κ 제거를 전체 성능 개선으로 판단하지 않는다.

이차 분기 제거는 Cora raw/RMS accuracy를 −.58/−.30pp 바꿨다.
Baseline 전체 norm 비율 `||βM||/||Z||`는 약1.45–3.24%였다.
이 값은 개별 노드 영향이나 학습 gradient를 직접 나타내지 않는다.
원래 metric360·개입6,750·층 진단4,740·fixed-Z1,560행, 2,370seed forward 경우,
optimizer update0, 전체102.1296초가 보고됐다.
출처: [서버 branch-strength 결과](../SERVER_BRANCH_STRENGTH_RESULTS.md).

## 11. 4.1 CSV 분석 — 분기 크기 인자

기존 CSV에서 각 seed·층의 값을 분해했다. 모델 forward와 optimizer update는 0이다.

\[
R=\tfrac13S_QA^\top CA S_QZ,\qquad
\frac{\|\beta M\|_F}{\|Z\|_F}
=\frac{\beta}{\kappa}\frac{\|R\|_F}{\|Z\|_F}.
\]

각 seed에서 계산한 후 집계하므로 평균 인자끼리 곱한 값과 실제 평균 βM/Z는 다를 수 있다.
β 평균은 .270–.304, κ는 약4–7이었다.
노드별 대각 비율 평균은 약1인데 드문 최대값으로 그래프 전체 이차 메시지를 나누는 구조가 관측됐다.
이 관측은 노드별 정규화 재학습의 근거였고 κ 하나가 학습 실패의 원인이라는 증명은 아니다.

Seed/층 인자60행·강도 추정204행·paired 개입756행·fixed-Z추정384행,
원본 보존·모델 forward0·update0·분석.904초가 보고됐다.
출처: [서버 CSV 분석 결과](../SERVER_BRANCH_ANALYSIS_RESULTS.md).

## 12. Experiment 4.2 — C 의존 노드별 정규화 재학습

5조건 fixed C=1/global raw·RMS/node raw·RMS를 모두 처음부터 학습했다.
동일한 citation 데이터·두 층·hidden64·LR 후보·tuning/final seed·500epoch를 유지했다.
135 tuning+75final=210run·105,000update다.
Global 결과는 이번에 재학습한 값이며 Experiment 4의 과거 결과와 섞지 않는다.

\[
D_C=\operatorname{diag}(A^\top CA),\quad S_C=D_C^{-1/2},\qquad
M_{\text{node}}=\tfrac13S_CA^\top CA S_CZ.
\]

C·D_C·양쪽 S_C의 미분을 유지했다. C=1이면 D_C=diag(Q)이고 global/node fixed가 같다.
현재 C를 고정한 native 연산자는 대칭 PSD이고 norm 상한1이다.
이는 입력 의존 전체 신경망의 Jacobian이나 학습 성공 보장이 아니다.
Norm-match 사후 개입에는 그 상한을 그대로 적용하지 않는다.

### Test 정확도 평균(%)

| 데이터 | Fixed C=1 | Global/raw | Global/RMS | Node/raw | Node/RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 73.56 | 70.76 | 70.52 | 72.72 | 72.08 |
| CiteSeer | 64.94 | 63.74 | 63.84 | 64.92 | 64.90 |
| PubMed | 75.46 | 75.06 | 74.74 | 75.56 | 75.72 |

모든 조건의 train accuracy 평균은100%였다. Train fit과 일반화 성능을 구분한다.

| 데이터 | Node−global raw Δacc pp / ΔCE | Node−global RMS Δacc pp / ΔCE |
| --- | ---: | ---: |
| Cora | +1.96 / −.07738 | +1.56 / −.07254 |
| CiteSeer | +1.18 / −.01985 | +1.06 / −.01903 |
| PubMed | +.50 / −.005381 | +.98 / +.0007282 |

여섯 평균 accuracy가 개선됐고 Cora raw/RMS의 paired accuracy 구간은0을 제외했다.
PubMed/RMS를 제외한 다섯 CE 비교의 paired 구간은 음수였고 PubMed/RMS는0을 포함했다.
재학습은 노드별 좌표·메시지 방향과 projection·gate·α·β를 모두 바꾼다.
예를 들어 Cora/raw 첫 층 β는 global .2697→node .04251,
βM/Z는 .02395→.01387로 낮아졌다. 모든 분기가 커져서 개선됐다는 설명은 맞지 않는다.

| 데이터 | Node−fixed raw Δacc pp / ΔCE | Node−fixed RMS Δacc pp / ΔCE |
| --- | ---: | ---: |
| Cora | −.84 / +.01573 | −1.48 / +.03074 |
| CiteSeer | −.02 / +.0005341 | −.04 / +.002582 |
| PubMed | +.10 / +.003659 | +.26 / +.008511 |

학습 C의 fixed C=1 대비 추가 분류 이득은 입증되지 않았다.
PubMed의 양의 accuracy 차이는 구간에0을 포함하며 Node/RMS의 CE는 세 데이터 모두 fixed보다 높고 구간도 양수다.

### C 비균일성과 실제 분기의 구분

| 데이터 | Node/raw 첫 층 C std | Node/raw 둘째 층 C std |
| --- | ---: | ---: |
| Cora | 7.19e−7 | .6141 |
| CiteSeer | 6.926e−7 | .4788 |
| PubMed | 1.055e−5 | .6555 |

현재 checkpoint의 raw 첫 층 C는 거의 균일하지만 둘째 층은 비균일하다.
RMS는 두 층 모두 비균일하다. 이 수치만으로 학습 내내 C가 상수였거나 모든 C가 미학습이라고 말하지 않는다.
Both norm-matched C=1/shuffle의 node accuracy 변화 범위는 −.042~+.18pp였다.
Both norm-matched C=1의 평균 CE는 여섯 node 조건 모두 감소했다.
이차 분기 제거는 CiteSeer/node raw·RMS accuracy를1.42/1.16pp 낮추면서 CE는 감소시켰다.
분기 자체의 top-1 기여와 learned C의 배치 효과는 다른 문제다.

Primary225·frozen12,420·layer8,430행, 실제 데이터·전체coverage·보존 완료,
6,946.80초(약1시간55분47초)가 보고됐다.
출처: [서버 node-normalization 결과](../SERVER_NODE_NORMALIZATION_RESULTS.md).

## 13. 추가 확인 — 둘째 층만 같은 크기의 C=1로 교체

사용자가 기존 4.2 `intervention_changes.csv`에서 추출해 제공한
node 조건·test·layer_1·`c_identity_norm_matched` 12행을 정리했다.
추가 학습이나 모델 변경을 수행한 결과가 아니다.
차이는 교체 후−원래 모델이며 accuracy 양수, CE 음수가 개선이다.

| 데이터 | Gate | Δaccuracy pp [95% 구간] | ΔCE [95% 구간] |
| --- | --- | ---: | ---: |
| Cora | raw | 0 [−.087798,+.087798] | −.00015439 [−.00041737,+.00010859] |
| Cora | RMS | +.14 [+.071995,+.20801] | −.0005968 [−.00085903,−.00033457] |
| CiteSeer | raw | −.02 [−.075528,+.035528] | −.00011344 [−.00041775,+.00019087] |
| CiteSeer | RMS | +.02 [−.21884,+.25884] | −.00076339 [−.0012577,−.00026909] |
| PubMed | raw | +.18 [+.018106,+.34189] | −.0012165 [−.0014868,−.00094613] |
| PubMed | RMS | +.079999 [−.26455,+.42455] | −.0013966 [−.001688,−.0011052] |

네 비교의 CE 구간은 전체가 음수이고 두 비교의 accuracy 구간은 전체가 양수다.
Accuracy 하락이 확인된 비교는 없으며, 나머지 구간의0 포함을 동등성 증거로 보지 않는다.
현재 둘째 층 C의 배치가 같은 크기의 C=1보다 유익하다는 근거는 없고 일부 조건은 교체하면 개선된다.

**Norm match의 gain에는 원래 C의 메시지 norm이 남는다.** 첫 층 C와 projection·α·β도 유지한다.
따라서 C 전체를 제거한 검사나 처음부터 C 없이 학습한 모델의 동등성으로 해석하지 않는다.
첫 층 기여는 기존 layer_0/both의 paired 구간을 별도로 확인해야 한다.
출처는 [4.2 결과 기록의 추가 CSV 절](../SERVER_NODE_NORMALIZATION_RESULTS.md)이다.

## 14. 로컬 구현 검증과 서버 결과를 분리해서 읽기

4.2의 로컬 검증은 새 테스트206개와 CUDA DEBUG90run·270update의 전체 pipeline이다.
DEBUG는 N24/30/36·F12·K3·hidden8·3epoch의 별도 fixture이며 primary90·개입1,512·층진단1,068행이다.
실제 citation 데이터의 별도 무갱신 검사는 새 node raw/RMS, hidden64·두 층·5seed·모든 wedge로
forward/backward를 수행했고 optimizer update0, 모든 파라미터/seed의 유한한 aggregate gradient를 확인했다.
로컬 RTX5070Ti에서 그 검사 peak는약1.63GiB였다.
이 측정값을 Adam 포함 장기 학습이나 A6000 처리량으로 대체하지 않는다.

서버 full210run 완료는 사용자 제공 서버 completion/summary의 별도 증거다.
세부 원값과 재현 검증을 확대하려면 서버의 원본을 받아 읽어야 한다.
관련 기록: [4.2 검증](../node_normalization/VERIFICATION.md),
[4 검증](../classification/VERIFICATION.md), [4.1 검증](../branch_strength/VERIFICATION.md).

## 15. 증거 파일의 위치

| 단계 | 사용자 원문 식별자 | 확인된 서버 결과 위치 |
| --- | --- | --- |
| 0/1 | full 완료 로그·e5e7dcf0-1e82-45c8-b7f9-efb9b59e52c5 | `results/wedge-fixed-20261002-144119` |
| 2 | 0412f129-7f5b-4c0f-9212-29dfb745cfe4 | `results/wedge-learned-20261002-172132` |
| 3 | 43094fe2-6bfc-4a48-9974-f30776c53783; e7640614-a320-4e01-88b7-8e88a447d102 | `results/wedge-feature-20261003-042123` |
| 3.1 | da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63 | 첨부 요약에서 정확한 폴더명 미확인 |
| 4 | a0770e1e-506a-43e1-b656-bbd3b045b108 | 완료 출력에서 원본 폴더명 미기록 |
| 4.1 | 975a1691-1a87-4c34-a672-f9aa8cf682ec | `results/wedge-branch-strength-20261003-104349` |
| CSV 분석 | 76feaf64-c752-4d8b-bb09-8aaf929648b2 | `results/wedge-branch-analysis-20261003-133432` |
| 4.2 | 5bdcdc63-17da-4c09-90bb-30b3139d7704; 추가 CSV 발췌 | ece-a6gpu6, 정확한 결과 폴더명은 원본 확인 필요 |

서버 상대 경로의 기준은 `/home/aicompetition07/new-gat`다.
4.2 구현 기준은2422568, CSV 분석의 출력된 실행 commit은3f8b24f다.
첨부에 없는 timestamp·source commit·raw CSV 내용은 추정해서 채우지 않았다.

## 16. 검토 전에 확정해 둘 주장 범위

1. 고정 Q의 연산 차이와 합성 teacher 회수는 확인했다. 범용 spectral 함수의 표현 한계 증명은 아니다.
2. RMS는 배율 안정성을 개선했다. Teacher 회수와 분류 성능까지 좋아졌다는 결과는 아니다.
3. Node 정규화 재학습은 global보다 개선됐다. κ 하나의 인과 원인 증명이나 learned C의 추가 이득은 아니다.
4. C가 비균일하다는 것, 이차 분기가 쓰인다는 것, C 배치가 유익하다는 것을 각각 검증해야 한다.
5. 현재 학습 C의 fixed 대비 이득은 미확인이다. 둘째 층 배치도 같은 norm의 C=1보다 이득이 확인되지 않았다.
6. Citation은 고정 그래프의 public split이다. 합성 OOD와 달리 독립 새 그래프 일반화를 확인한 결과가 아니다.
7. 고정 checkpoint의 사후 개입에서 좋은 test 처리 하나를 고르지 않는다. 기존 test를 본 뒤의 후속 비교라는 범위를 유지한다.

다음 변경이나 추가 학습을 확정하기 전에, 이 코드가 연구 질문을 실제로 표현하는지와
현재 결과로 주장할 수 있는 범위를 먼저 검토한다.
