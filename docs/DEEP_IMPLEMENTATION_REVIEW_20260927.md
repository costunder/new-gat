# 2026-09-27 첨부 검수 통합 재검토

이 문서는 cc173ed까지의 검수 기록이다. 이후 사용자 지시에 따라 PPI 실행을 차단하고
GCN/GraphSAGE 및 arxiv 공식 test 경로를 추가한 현재 계약은
[ARXIV_BASELINE_COMPARISON.md](ARXIV_BASELINE_COMPARISON.md)에 있다.
아래 당시 측정값을 후속 코드에서 새로 실행한 결과로 읽지 않는다.

## 판정과 범위

**현재 상태를 원래 연구 요구사항의 완료본으로 판정할 수 없다.** 주 데이터셋을
벗어난 PPI 전용 설계와 주 4조건 실행기 부재가 남아 있다. 별도로 저장본 검증
결함을 실제 CUDA 모델에서 재현해 수정했다. 수학은 아래 독립 계산·미분 검사의
범위에서 일치했다. 국소 연산의 정확성이 전체 연구 설계의 합격을 뜻하지 않는다.

검토 시작 HEAD는 `cd411cffee409cd9f3a93ee99205362e7b27cec9`이다. 최신 첨부는
이전 `08521c8` 검수이므로 이미 수정된 것과 현재도 남은 것을 구분했다.
최종 커밋과 파일별 SHA256은 이번 검수 ZIP의 `MANIFEST.json`에 있다.

최신 8ef518d9 첨부 및 앞선 7b646e85/e3be3519/36fc7ebd 검수 첨부를 현재 코드와
대조했다. 첨부에 연결된 외부 sandbox의 원본 증거까지 내려받아 재현했다고
주장하지 않는다. 첨부의 측정값과 이번 직접 실행값은 서로 다른 증거다.
주 검토 대상은 aggregation_comparison, sampled_inductive와 실제 호출되는
incidence operator, V5 C solver·전파·샘플러·학습 예산이다. 저장소의 모든 별도
연구 트랙을 전수 실행한 검토는 아니다.

## 1. 요구사항 대응표

| 요구·논의 | 현재 확인한 구현 | 판정 범위 |
| --- | --- | --- |
| C의 의미 | B,H에서 공유 θ와 유한 K-step으로 c를 계산하며 C=diag(c)이다. | C를 입력 그래프라고 한 설명은 잘못이다. |
| 공유 C 규칙 학습 | loss에서 live C 및 차수 정규화를 거쳐 θ로 미분된다. 영구 edge-ID 파라미터가 아니다. | 구현됨. 모든 분포에서 성능을 보장하지 않는다. |
| 고유분해 없는 전파 | edge difference·가중치 곱·scatter로 Bᵀdiag(ωc)B의 작용을 계산한다. | 구현됨. 전체 실행 속도 우위는 미측정이다. |
| 새 backbone 샘플링 | aggregation의 incidence-only full/cluster/cluster_disjoint를 허용한다. | 이전의 무조건 full-only 차단은 해제됨. |
| 주 데이터셋 | 기존 주 비교는 ogbn-arxiv이다. | PPI로 바꿀 근거가 없다. |
| 주 4조건 | fixed/learned C × full/sampled를 통합 관리·검증·보고하는 arxiv controller가 없다. | 미완성. PPI runner로 대체할 수 없다. |
| 독립 새 그래프 평가 | arxiv 공식 node split은 독립 graph split이 아니다. | 별도 protocol 미구현. PPI로 임의 대체하지 않는다. |
| 같은 backbone의 energy/lift | none/diagonal/diagonal+cross × none/linear/pre/post의 12조건. | 구현됨. |
| C 대조군 | no-lift에서 fixed/shared/per-head × energy off/on. 중복 제외 4조건 추가. | incidence 총 16조건. 전체 36조건은 아니다. |
| 외부 비교 | GATv2, DUALFormer, 명시적 DUALFormer no-skip 변형. | 공통 레시피 비교. 같은 용량·공식 최적 점수 재현은 아니다. |
| 기전 진단·개입·효과 보고 | aggregation의 mechanisms/audit/effects에 연결돼 있다. | sampled_inductive에는 동일한 전체 기전 감사가 없다. |
| 거리별 B_r·블록 경계값 해법 | 새 forward에 없다. | 미구현. depth-history로 대체됐다고 할 수 없다. |
| 역복원·Jacobian·작은 특이값 | 새 모델 전체에 연결된 진단이 없다. | 옛 incidence 진단이나 특징 rank와 다르다. |
| forest/negative/cycle/관계 타입 | 별도 트랙의 일부 구현은 있으나 새 comparison은 full support·untyped로 제한한다. | 새 모델 통합 완료로 세지 않는다. |
| 실제 성능·비용·MIG | 로컬 실제 데이터 캐시가 없다. | 본학습·공식 test·10GB 적합성 미검증. |

PPI는 단순 기본값이 아니다. runner에 dataset 옵션이 없고 child arguments 및
load가 PPI로 고정돼 있다. data.validate_splits는 다른 dataset을 거부하고
loss/metric도 BCE/micro-F1이다. 이름만 arxiv로 바꾸는 것은 해결책이 아니다.

## 2. 실제 CUDA로 재현한 저장본 검증 결함과 수정

기존 sampled trainer는 best_state를 저장했지만 종료 audit에서는 메모리에 남은
best_state를 사용했다. 파일 hash/history/count 상호 일치는 검사하면서도
**실제 저장된 가중치의 예측**은 검증하지 않았다.

CPU mock 예측 대신 reference CUDA 모델로 반례를 만들었다. 합성 그래프의 label을
1로 만들고 저장 함수에만 오류를 주입했다. 직렬화되는 best_state만
decoder.weight=0, decoder.bias=-50으로 변경하고 메모리 best와 이력은 유지했다.

| 항목 | 이번 직접 측정 |
| --- | ---: |
| 기존 실행의 완료 상태 | passed |
| 기록된 validation F1 | 0.938034188034188 |
| 기록된 TP/FP/FN/total | 439/0/58/497 |
| 파일 best_state를 새 모델에 로드한 F1 | 0.0 |
| 파일에서 재평가한 TP/FP/FN/total | 0/0/497/497 |

증거: `results/deep-audit-disk-fault-proof-20260927.json`, 수정 전 실패한
`results/deep-audit-disk-fault-before-20260927.xml`. 수정 전 테스트는 예외가 나야
하는데 나지 않아 실패했다. 이 기록을 최종 통과 수에 합산하지 않는다.
**의도적인 합성 오류 주입이며 사용자 체크포인트 손상을 관측한 것이 아니다.**

이번 수정 후 종료 순서는 다음과 같다.

1. 마지막 atomic checkpoint 저장을 마친다.
2. 학습 모델·optimizer·메모리 best_state를 해제한다.
3. 실제 last.pt를 다시 읽고 identity/history/best_epoch/선택 count를 확인한다.
4. 새 CUDA 모델에 파일의 best_state를 strict load한다.
5. 전체 validation 5회에서 선택 당시 정수 count가 정확히 재현돼야 한다.
6. 모델 state/source/checkpoint hash가 audit 도중 변하지 않았는지 검사한다.
7. 모두 통과한 경우에만 passed와 persisted_best_audit를 기록한다.

두 모델을 동시에 VRAM에 유지하지 않는다. 정상 경로 검사에서는 weak reference로
기존 모델 해제를 확인하고 실제 새 모델이 생성되는지도 확인했다. 오류를 주입한
저장본은 이제 거부되며 통과 metrics.json을 만들지 않는다.

completed()는 checkpoint hash/저장 best_state hash/선택 epoch와 감사 증거를
대조한다. 감사 증거 삭제·변조와 반복 평가 graph ID 변경도 거부한다.
**completed()가 호출될 때마다 CUDA 추론을 새로 하는 구조는 아니다.** 완료 시
수행한 저장본 CUDA 감사의 증거를 이후 검사한다. 모든 파일을 함께 위조하는 것까지
방어하는 외부 서명 체계라는 뜻도 아니다.

앞선 cd411cf의 **모든 조건 선검증 후 test matrix 잠금**은 별개의 수정이다.
이번에도 정상 matrix 평가와 잘못된 뒤쪽 조건을 test 접근 전에 거부하는 검사를
통과했다. 기존 aggregation에는 disk best reload와 선택 count 검사가 이미 있어,
이번 결함을 모든 체크포인트 코드의 공통 결함으로 확대하지 않는다.

## 3. 추가로 발견하고 수정한 설정 기록 불일치

aggregation.make_optimizer는 모든 파라미터에 같은 AdamW group과 weight decay
0.01을 사용한다. 그러나 configuration에는 역사적 trainer의 C/scalar decay
0.0이 남아 있었다. 설정을 읽는 사람이 실제와 다른 최적화 계약으로 이해하게 된다.
conductance/scalar decay를 실제 공통 decay와 맞추고 공통 lr multiplier=1을
명시했다. **optimizer 동작은 바꾸지 않았다.** 실제 CUDA 모델의 optimizer group과
기록을 비교하는 검사도 추가했다.

sampled_inductive는 aggregation 설정을 재사용하며 수행하지 않는 layer/head
statistics and interventions 감사 설명까지 상속했다. 별도 configuration에서
mechanism_audit=null, 실제 저장본 재현 검사를 validation_audit로 기록하도록 했다.
dry-run 각 cell의 full/sample support도 같은 실제 설정을 사용한다.
이는 기전 감사를 새로 구현한 것이 아니라, 수행하지 않는 기능의 기록을 고친 것이다.

## 4. 독립 수학·미분 검사

아래 모델 및 미분 계산은 전부 CUDA다. 소규모 합성 행렬은 명시적 수학 debug이며
최종 모델/데이터 설정을 바꾸지 않는다.

### C objective gradient: 6조건

signed incidence B와 unsigned endpoint 행렬을 별도로 구성해 다음 목적함수를
직접 계산하고 autograd로 미분했다. shared/3-head × rho=0/0.1/10이다.

E(c) = mean_omega[c delta + tau(c log(c) - c + 1)]
       - rho mean_active[log(d_c / d_reference)].

실제 _scaled_gradient에 omega/graph_mass를 곱한 값과 일치했다.
FP64 rtol=1e-10, atol=1e-11을 통과했다. 비균일 omega, isolated node가 있는
disjoint graphs를 포함했다.

### 유한 K-step C 전체 gradient: 2조건

실제 K=8의 shared/3-head solver에서 feature, graph context, omega, theta의
방향 미분을 중심 유한차분과 비교했다. 검사 방향 중 최대 절대오차는 **9.5705e-10**.
positive/finite C, 그래프별 weighted mean C=1, edge orientation 반전 불변성,
독립 graph를 배치/단독 실행할 때의 일치도 확인했다.
선택한 입력·방향의 1차 미분 검증이다. 모든 입력의 안정성 증명, 다른 solver의
전체 재구현, 유한 K에서 최적해에 도달한다는 증명은 아니다.

### 희소 전파와 dense Laplacian: 4조건

L=Bᵀdiag(omega*c)B를 직접 만들어 row/symmetric × shared/per-head에서 출력과
value/C/omega/beta gradient를 비교했다. 현재 comparison은 row를 사용한다.
출력 FP64 rtol=1e-11, atol=1e-12, gradient rtol=1e-10, atol=1e-11을 통과했다.
isolated node 보존도 확인했다. diffusion과 Gram 각각 omega*c를 한 번 사용하며,
correction을 두 번 곱하는 오류는 발견하지 않았다.

### 16 incidence 조건의 dense 출력·gradient

전파/Gram/lift/readout을 독립 dense 식으로 계산했다. 추가 분기가 꺼져 우연히
일치하지 않도록 energy readout과 lift projection을 중립 초기값에서 변경했다.
실제 경로는 activation checkpoint를 사용한다.

| 최대 절대오차 | 이번 CUDA 측정 |
| --- | ---: |
| 출력 | 5.9605e-8 |
| 입력 gradient | 9.6858e-8 |
| 파라미터 gradient | 4.7684e-7 |

hidden=12/layers=3/heads=3의 수학 debug다. C·beta 생성기/context는 실제 모듈을
양쪽에서 공통 사용하고 C 내부 미분은 앞의 별도 검사로 보완했다.
첨부의 CPU 수치는 입력·정밀도·실행이 달라 이번 수치와 동일해야 하는 것이 아니다.
각 조건별 증거는 `results/deep-audit-math-summary-20260927.json`에 있다.

## 5. 해석에서 넘으면 안 되는 경계

현재 Gamma는 omega*c로 가중한 edge difference 내적을 양 endpoint에 절반씩
누적한다. node 합은 tr(V_kᵀ L V_l)이다. k=l은 이차형식, k!=l은 쌍선형형식이다.
과거 state들은 현재 층의 같은 W로 투영된다. **층의 깊이는 정확한 거리별 hop
shell이 아니다.** Gram은 lift 전 V 좌표이며 [V,V²] 전체의 Gram이 아니다.
현재 Gamma를 같은 층 C solver에 명시적 입력으로 주지도 않는다. readout으로
메시지에 더해 다음 층 H/C에 영향을 준다. readout=0 초기에는 energy→C 추가
gradient가 0이고 기존 diffusion→C gradient는 존재한다.

양의 무방향 가중 Laplacian L의 영공간은 연결성분별 상수다. Bx/Lx만 관측하면
성분별 상수 오프셋은 보이지 않는다. B를 알면 손실 공간을 분석할 수 있지만,
사라진 값을 추가 관측 없이 복원할 수 있다는 뜻은 아니다.

그러나 **P=I-beta D^-1 L을 L 자체와 혼동하면 안 된다.** P는 상수 mode를 보존한다.
고정 C 및 graph/head별 scalar beta에서 양의 차수 부분은
D^(1/2) P D^(-1/2)=I-beta D^(-1/2)L D^(-1/2)이며 spectral 응답은 1-beta*lambda다.
역가능성은 이 값이 0인지에 달린다. beta=1/2, 정규화 Laplacian lambda=2인
mode가 있으면 그 mode는 사라진다. 전체 모델의 C(H), projection, ReLU, dropout,
readout까지 하나의 고정 spectral operator로 부를 수는 없다.

rank(XW)는 rank(X)를 넘지 않지만 [V,V²]는 일부 입력의 특징 rank를 높일 수 있다.
그 뒤 projection·전파·ReLU가 injective해지는 것은 아니며, 원래 head width로
되돌리는 projection의 출력 rank도 그 폭으로 제한된다. 유사역이 역변환처럼
작동한다는 주장에는 관측 map/복원 대상/Jacobian/nullspace/anchor/잡음 안정성의
정의와 검증이 필요하다. 새 comparison에는 이 실험이 없다.
옛 B[X,X²] 진단은 새 신경망 최종 출력의 역변환이 아니다.

custom Gram은 once_differentiable인 1차 미분 구현이다. differentiable double
backward/Hessian 지원으로 소개하면 안 된다. 현재 1차 task loss에서 C unroll로
역전파하는 요구와는 별개다.

## 6. 비교 공정성·학습 예산·자원

공통 linear encoder/decoder는 차원을 맞추는 공통 구조다. 이를 넣었다는 이유만으로
비교가 무효가 되지는 않지만, 전체 모델 용량까지 통제됐다는 뜻도 아니다.
incidence는 역사적 외부 residual/FFN/LayerNorm을 제거했다. GATv2도 residual=False다.
DUALFormer는 자체 SA residual/LayerNorm을 유지하고 no-skip은 별도 변형이다.
**19조건 전부가 내부 normalization/residual까지 같은 단일 operator 교체 실험은
아니다.** 외부 구조 보존 비교와 같은-backbone 내부 ablation을 구분해야 한다.
파라미터 수 일치나 공식 논문의 tuned score 재현은 보장하지 않는다.

reference 8/256/8, large 12/384/8을 이번에 변경하지 않았다.
shared_initial hash는 공통 encoder/decoder를 확인하며 서로 다른 모델의 모든
내부 파라미터가 같다는 증거가 아니다.

reference_updates는 full/sample의 step 수를 일치시키는 정책이 아니다. 각 모드의
기준 업데이트 최소 용량과 지정 epoch를 유지하는 정책이라 batches/epoch가 다르면
계획·실제 step도 달라진다. 별도 PPI runner는 200 complete epochs/no early stopping으로
기존 patience=50/reference_updates와도 다르다. 차이를 문서화했다고 사용자 승인이
생기지는 않는다. 이 경로를 주 실험으로 대입하지 않는다.

sampling correction은 full/sample degree 비 기반 근사이며 정확한 edge 포함확률의
역수가 아니다. sample마다 C와 정규화도 바뀐다. 샘플 평균=full 연산이라는 불편
추정 주장을 할 수 없다. full/sample 점수 차이에는 optimization schedule도 섞인다.

Gram forward의 buffer 누적과 backward의 전체 history/C gradient 1회 할당은
이미 구현돼 있다. 이전 CUDA profiler의 500→1, 32→1은 특정 합성 backward에서의
할당 횟수다. 이번에 그 profiler를 재실행하지 않았다. 누적 logical bytes를 peak
VRAM이나 전체 모델 속도 개선율로 바꿔 말할 수 없다. O(K²) 깊이쌍과 전체 node
통계의 비용 자체가 없어진 것도 아니다.

생산 경로는 CPU sampler/topology/cache/loader와 GPU 모델/C/loss/backward/optimizer를
분리하고 CPU fallback을 허용하지 않는다. 저장소의 과거 CPU 모델 단위검사 파일은
여전히 존재하므로 전체 pytest가 전부 GPU라는 설명은 틀리다. 이번에는 GPU 모델
검사와 CPU 제어/샘플러 검사를 명시적으로 선택했다.

PPI 경로의 batch/worker calibration은 구현돼 있으나 실제 데이터로 이번에 수행하지
않았다. nested C timer는 forward/backward와 겹치므로 합산하면 이중 계산이다.
worker CPU 시간도 wall에 더하지 않는다. 속도 주장은 sampling/transfer/C/학습/
checkpoint/eval 및 별도 loading/calibration을 포함하는 비용 범위를 고정한 뒤
실제 데이터와 할당 장치에서 해야 한다.

## 7. 최종 검증 기록

로컬 RTX 5070 Ti 16,303 MiB, PyTorch 2.13.0+cu130, CUDA runtime 13.0,
PyG 2.8.0.post1에서 실행했다. logical CPU/affinity 16개, RAM 68,640,653,312 bytes다.
사용자 A100 MIG 10GB 측정은 아니다. data에는 .gitkeep 외 실제 benchmark 캐시가 없다.

| 최종 JUnit | 수 | 범위 |
| --- | ---: | --- |
| deep-audit-verified-20260927-02.xml | 47 | CUDA 모듈 38 + sampler/metadata/control 9 |
| deep-audit-control-20260927-03.xml | 46 | aggregation integrity/validation/effects 제어 |

**중복 없는 93개, 실패·오류·skip 0.** 중간 29개 실행과 합산하지 않는다.
CUDA 모듈은 수학 28개, 저장본/설정 3개, 정상 학습·matrix·probe·재개 7개다.
설정 검사는 CUDA 모델/optimizer 대조이며 자체 forward 학습 검사는 아니다.
학습·재개는 reference 크기와 명시적 4-epoch 합성 debug다. 중단 후 재개와
연속 실행의 최종 파라미터 bitwise 동일성도 통과했다. 업스트림 torch.jit.script
deprecation 경고 2개가 있었다. Ruff 및 Git whitespace도 확인한다.

이전 cd411cf의 실제 transductive sampler→CUDA→full validation 합성 9조건은
역사적 증거로 유지한다. 실제 arxiv 실행으로 바꾸어 말하지 않는다.
첨부 검수자의 108 CPU pass/1 failure/3 deselected도 이번 93개와 다른 실행이다.

## 8. 변경 영향과 남은 완료 기준

이번 변경은 저장본 검증, 잘못된 설정 기록, 회귀·수학 검사와 문서다.
모델/생산 epoch/데이터/sampling 규모를 축소하지 않았다. historical research/src/
scripts/incidence_ablation 및 사용자 결과는 변경하지 않았다. 작업 시작 전 dirty였던
gpt_handoff 3개 파일은 이번 커밋·ZIP에서 제외한다.

source/config identity가 달라 이전 comparison/sampled checkpoint를 이 버전의
증거로 조용히 이어 붙일 수 없다. 기존 기록은 보존하며 변경 경로의 실행에는 새
run ID가 필요하다. 서버 실행·본학습·공식 test·원격 push는 이번에 하지 않았다.

원래 요구사항 완료에는 다음이 남아 있다.

1. arxiv 주 4조건의 통합 controller/manifest/report 및 각 cell의 실제 coverage,
   업데이트 수, 학습 예산 비교 기준을 구현·검증한다.
2. official node split과 독립 새 그래프 protocol을 구분한다. 후자는 graph 구성,
   feature/label 규약, split/누수 방지, 최종 test 기준이 필요하다.
3. C·sampling의 조건부 차이와 최적화 일정 차이를 구분하고 내부/외부 비교 목적을
   고정한다. 테스트 숫자나 기존 PPI 점수로 이 설계를 대신하지 않는다.
4. 실제 할당된 MIG 10GB에서 원래 규모의 calibration, 본학습, 저장본 감사,
   최종 평가 및 전체 비용 측정을 한다.
5. 역복원/exact-hop/cycle/heterogeneous를 주장하려면 별도 관측·연산·대조군을
   구현한다. 다른 트랙의 파일 존재는 새 모델의 완료 근거가 아니다.

이번 산출물은 구체적인 결함 수정과 검증 범위를 명시한 재검수본이다.
원래 실험 전체의 완료 보고가 아니다.
