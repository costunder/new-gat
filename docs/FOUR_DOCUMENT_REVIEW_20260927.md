# 네 첨부 문서의 코드·수학·연구 설계 대조 검토

**보존된 수정 전 검토 기록이다. 이후 적용한 수정은
[REVIEW_REMEDIATION_20260927.md](REVIEW_REMEDIATION_20260927.md)를 따른다.**

검토일: 2026-09-27. 소스 기준: `2602d907f32893706af5a848be8c05b88d3996d1`
(모델 구현 `4b4df52`, 전달 문서/패키징 추가 `2602d90`).
**이 문서는 검토 및 반례 재현 결과다. 아래 R0–R4 수정안은 아직 production 코드에 적용하지 않았다.**
모델/데이터/학습 규모와 기존 실험 파일을 변경하지 않았다.

## 1. 검토 대상과 근거 구분

네 첨부의 검수 본문을 읽고 서로 중복되는 판단과 실제 제안을 분리했다.

| ID | 첨부 | 성격 |
| --- | --- | --- |
| D1 | 77c1d710… / 이번 버전은 arxiv 중심 실험… | 독립 검수 요약, 반례, 수학 검사 주장 |
| D2 | e97609b4… / 핵심 수식은 유지하고… | 연구 우선순위 및 수정 설계 제안 |
| D3 | b930b36d… / 코드 수정 및 연구 실행 방향 | R0–R4 구현 위치·완료 기준·융합 대수 prototype |
| D4 | 1baa25db… / 독립 요구사항·수학·구현 검수 | 상세 검수, 실행 범위, 한계 |

D4의 실제 보고서는 1–204행이다. 뒤에 붙은 브라우저 UI/직렬화 상태는 검수 내용이
아니므로 보고서 근거 및 전달 파일에서 제외했다. 원본 첨부는 수정하지 않았다.
첨부의 `sandbox:/mnt/data/...` 링크에 있는 ZIP/스크립트는 이 로컬 환경에 제공되지
않았다. 링크 파일을 직접 검사했다고 주장하지 않는다. 문서가 제시한 수치와 이번에
직접 재현한 수치를 구분한다. D1/D4의 267개 CPU 검사도 이번 실행 개수에 포함하지 않는다.

현재 검토 산출물은 `results/review-four-documents-20260927/`에 있다.

- `reproduce.py`, `reproduced.json`: CUDA 모델 복제, 실제 저장 체크포인트 test 재사용,
  debug 출처, reference 공통 초기값 확인.
- `fusion_and_sampling.py`, `fusion-sampling.json`: 새 융합 제안의 독립 CUDA 대수 검산,
  sampler RNG·감독 노드 coverage 확인. 최적화 커널 구현이 아니다.
- `cuda-math-sampling.xml`, `.log`: 현재 수학·16조건·sampler CUDA 회귀.
- 중간 실행 로그도 보존했다. 실패를 최종 통과로 바꾸거나 합산하지 않는다.

## 2. 판정과 우선순위

**세 가지 R0 결함 지적은 타당하며 현재 소스에서 재현됐다. 핵심 수식을 폐기하거나
PPI로 돌아갈 근거는 없다. 연구 설계 제안은 대체로 적절하지만 아래의 예산·샘플 크기·
정밀도·시간순 데이터 규약을 보완해야 바로 구현할 수 있다.**

| 항목 | 판단 | 이번 근거 |
| --- | --- | --- |
| 저장 test score/count 동시 변경 | P1: 재사용 무결성 결함 | 실제 최초 CUDA test 후 메모리 사본의 동시 변경 수락 |
| 완료 cell 삭제 | P1: 완료/중단 상태 구분 결함 | passed 상태인데 test 준비에 재진입 |
| energy 모델 deepcopy | P1: 모델 객체 격리 결함, 사용 경로 한정 | CUDA 예외·stale C·원본 cache 오염·출력/gradient 불일치 |
| debug 출처 불일치 | P1: 연구 결과 승인 경계 결함 | configuration=true, protocol=true, metrics=false이며 승인기 수락 |
| C/전파/Gram/lift 핵심 | 검사한 범위에서 유지 가능 | 관련 CUDA 37검사 통과 |
| incidence-only sampler | 이미 구현·연결됨 | full/cluster/cluster_disjoint 실제 CUDA 경로 재검사 |
| 4조건 통합 연구 | 아직 미완료 | 실행 모드·예산·보고·전체 matrix의 통제 통합이 없음 |
| 연도별 temporal induction | 새 custom protocol로 설계 가능 | node_year 캐시·view identity·통계 제한 등 추가 구현 필요 |
| 융합 Gram/readout | 실수 대수상 타당, 아직 production 최적화 아님 | FP64/FP32 검산; BF16 재배치 차이 확인 |
| 실제 정확도·가속·MIG 10GB | 판단 보류 | 실제 arxiv 데이터/서버 학습·프로파일 미실행 |

P1 표기는 이번 검토에서 먼저 해결해야 한다는 뜻이다. 실제 arxiv 성적이나 기존
사용자 PPI 결과가 손상됐다는 관측이 아니다.

## 3. R0-B: test 결과 재사용 결함의 범위와 수정 계약

위치: `experiments/aggregation_comparison/final_test.py:37–183`, 특히 저장 결과 분기.

최초 test는 전체 matrix의 학습/audit를 검사하고 checkpoint를 먼저 고정하며,
디스크 best를 새 모델에 읽고 validation count를 재현한다. 이 경로가 있다는
D1/D4의 판단이 맞다. 옛 disk-best 결함을 현재 최초 평가에 그대로 적용하면 안 된다.

이번에는 새 CUDA 합성 GCN(reference 8층/256 hidden, 명시적 160노드/4epoch debug)을
실제 학습·저장·5회 audit한 뒤 최초 test를 실행했다. production checker와
checkpoint 로더를 mock하지 않았다. 데이터 로더만 명시적 synthetic payload로
교체했다. 이후 manifest의 메모리 사본만 바꾸고 재사용을 검사했다.

| 입력 변경 | 현재 결과 |
| --- | --- |
| 최초 test | correct=5/32, Accuracy=0.15625 |
| correct=22/32, Accuracy=0.6875로 함께 변경 | 수락, 재추론 없음 |
| evaluation_seconds=-100 | 수락 |
| evaluation_seconds=NaN | 수락 |
| status를 unrecognized로 변경 | 수락 후 passed로 덮음 |
| passed 결과에서 유일한 cell 삭제 | PreparedInputs에 재진입; 재추론 전 sentinel로 중단 |

검수에 사용한 디스크 best/last/metrics/history/configuration/audit 파일은 반례 전후
SHA-256이 같았다. 마지막 반례는 실제 test 추론을 끝까지 다시 수행한 사례가 아니다.

필요한 수정은 단순히 report 전체에 hash 하나를 추가하는 것보다 구체적이어야 한다.

1. 각 cell에 최초 평가 artifact를 독립 저장한다. node ID 순서, 예측 class,
   checkpoint/data/split/source/protocol/arm/seed/profile 식별자 및 평가 count·시간을
   결합한다. 보고서 row는 그 artifact를 읽어 생성한다.
2. Accuracy는 공식 test ID 집합과 **순서·중복·범위·정확한 coverage**를 확인하고,
   고정된 정답에서 재산출한다. 단순히 total 값만 맞는 것으로는 부족하다.
3. class prediction만으로 Accuracy는 재계산할 수 있지만 cross-entropy loss는
   재계산할 수 없다. Loss까지 재산출하려면 정답 class log probability 등 충분한
   값을 별도로 저장해야 한다. 그렇지 않으면 loss는 hash로 고정한 최초 기록으로 구분한다.
4. runtime은 bool이 아닌 유한한 비음수 숫자여야 한다. schema와 상태 enum을 검사한다.
5. passed는 정확한 expected cell 집합과 모든 artifact·hash를 요구한다. 누락/추가/
   중복/다른 run 혼입을 거부하며 `test_evaluated`와 완료 상태의 불일치도 거부한다.
6. running만 미평가 cell을 재개한다. 이미 완료로 기록된 cell의 파일이 없으면
   손상이다. 조용히 재추론하지 않는다. 아직 report에 반영되지 않은 artifact는
   crash 시점 규약에 따라 검증 후 연결하거나 명시적으로 거부한다.
7. artifact 임시 기록→원자적 publish→manifest 연결 순서를 고정하고 그 사이
   중단에 대한 검사를 둔다. 모델 재학습·test로 epoch 재선택을 허용하지 않는다.

이는 실험 산출물의 손상·혼합·변경 검출이다. 모든 파일과 manifest를 함께 수정하는
공격에 대한 인증 체계라고 주장하지 않는다. 이전 형식의 완료 결과에 새 신뢰 표시를
자동 부여하거나, 자료가 없다는 이유로 test를 자동 재실행하는 migration은 피한다.

## 4. R0-A: 명시적 C 반환과 모델 복제

위치: `model.py:303–312`, `model.py:539–560`,
`experiments/incidence_ablation/model.py:113–212`.

현재 hook의 `op=operator`는 원본 객체를 캡처한다. `deepcopy`가 함수를 새 operator에
다시 묶어주지 않으므로 D1/D4의 원인 설명이 맞다.

이번 CUDA 재현은 9노드/3층/12폭/3head의 명시적 수학 debug에서 nonzero energy
readout을 사용했다. 원본과 다른 입력을 복제본에 주고 fresh+state_dict 모델과 비교했다.

- 첫 forward 이전 deepcopy: `live_comparison_c` 부재로 AttributeError.
- no_grad forward 후 deepcopy: 복제본 C는 이전 값에 고정되고 원본 cache가 바뀜.
- stale C와 해당 입력에서 실제 계산된 C 차이: 최대 **1.3910904**.
- 출력 차이: 최대 **0.0842526**.
- 파라미터 gradient 차이: 최대 **3.2842207**.

현재 학습/audit/최초 test는 fresh 생성+state_dict 방식이다. 따라서 이 반례는
정상 학습 전체가 stale C로 수행됐다는 증거가 아니다. EMA/teacher/복제 기반 분석에서
실제로 잘못된 결과를 만들 수 있는 모델 인터페이스 결함이다.

D3의 `forward_with_state` 방향을 채택하는 것이 적절하다. 다만 다음 의미를 고정해야 한다.

| 반환값 | 정확한 의미 |
| --- | --- |
| message | 기존 operator가 반환하던 전파·lift·출력 projection 결과 |
| conductance | 같은 호출에서 solver가 계산한 live c |
| effective_weight | 실제 전파에 적용된 a=omega*c; gate가 있는 일반 연산자는 omega*gate*c |
| beta | 해당 graph/head의 live beta |

현재 comparison은 full selector라 gate=1이다. 미래의 gate 존재까지 암묵적으로
가정하지 말고 인터페이스에서 명확히 다룬다. 기존 shared_head_diffusion은 correction을
내부에서 곱하므로 이미 곱한 a와 같은 correction을 다시 넘기면 이중 적용이다.

Solver 한 번만 호출, live tensor detach 금지, 같은 forward의 전파와 Gram이 같은
가중치를 사용한다는 조건을 검사한다. 반환 state는 지역 변수로 소비하고 장기 cache에
autograd graph를 남기지 않는다. 진단 복사본만 detach한다. 기존 parameter 이름/shape와
공통 초기값은 보존한다. 이전 실험의 source guard를 완화하며 이어 붙이는 수정은 아니다.

완료 검사는 clone 첫 실행, 서로 다른 입력의 교차 실행, 원본 cache 불변성,
nonzero readout의 output/gradient, checkpoint on/off, 고립 노드 및 disjoint graph,
shared/per-head C와 비단위 omega를 포함해야 한다. checkpoint 재계산에서 전달 state를
다른 forward의 side effect로 대체하지 않는다. 진단은 eval/no_grad의 별도 경로이며
진단 RNG 사용이 이후 학습 RNG를 바꾸지 않아야 한다.

## 5. R0-C: 합성 출처와 연구 결과 승인

위치: `engine.py:556,711`, `integrity.py:190–197`.

정적 발견에 그치지 않고 새 CUDA debug training 산출물에서 다음을 확인했다.

| 필드/동작 | 관측값 |
| --- | --- |
| configuration.json.debug | true |
| metrics.protocol.explicit_synthetic_debug | true |
| metrics.debug | false |
| engine.inspect_completed | accepted |

공식 cache 로더가 가짜 arxiv를 수락했다는 의미는 아니다. 이번 debug는 로더를
명시적으로 바꿨다. 그러나 그 결과를 연구용 완료 승인기가 수락하는 모순은 실제다.

단순히 최종 debug 값을 바꾸면 현재 승인기를 호출하는 합성 검사가 실패한다.
그러므로 provenance를 identity에 고정하는 수정과 **합성 artifact 검증/연구 결과 승인
분리**를 함께 해야 한다. production CLI에 임의의 bypass 플래그를 추가하지 않는다.
debug=true 자료도 checkpoint/hash/학습 경로 검사를 받을 수 있어야 하지만,
official benchmark report로 내보내는 승인 단계에서는 거부해야 한다.
configuration/checkpoint/metrics/audit/final-test에서 같은 출처를 읽고 교차 검증한다.

Train-only temporal view는 합성 debug도 임의 subset도 아니다. 전체 데이터 universe,
시점별 허용 view, 실제 감독 집합의 식별자를 별도로 기록해야 한다.

## 6. 유지할 수학과 과장하면 안 되는 해석

기본 모델은 다음이다.

\[
c_s=\operatorname{Solve}_K(B_s,H_s;\theta),\quad
a_s=\omega_s\odot c_s,\quad
L_s=B_s^\top\operatorname{diag}(a_s)B_s,\quad
P_s=I-\beta D_s^\dagger L_s.
\]

C는 입력 그래프나 영구 edge-ID 파라미터가 아니다. 학습되는 공유 규칙 theta를
고정해도 관측 graph/features가 바뀌면 C를 다시 계산한다. C 계산과 차수 정규화의
C 의존성 모두 task gradient에 포함된다. K8 함수 미분의 검산은 argmin 수렴 증명이 아니다.

\[
\Gamma_{i,h}^{k\ell}=\frac12\sum_{e\ni i}a_{e,h}
\langle b_e^\top V_h^{(k)},b_e^\top V_h^{(\ell)}\rangle.
\]

고정된 a와 value 좌표에 대한 조건부 쌍선형형식이며 k=l은 이차형식이다.
전체 모델에서는 a가 입력에 의존하므로 전역적으로 고정된 쌍선형 함수가 아니다.
대각항은 미분 기여가 두 번 들어가야 하며 shared C의 head별 gradient는 합해야 한다.
현재 Gram backward의 두 scatter를 하나로 줄이면 오류다.

깊이 history는 정확한 거리별 hop shell이 아니다. energy+pre-lift에서도 Gram은
pre-lift value 좌표에서 계산된다. 행렬의 nullspace와 실제 P의 정보 보존은 구분한다.
L이 component-constant를 없애도 P는 그 상수 모드를 보존한다. projection/ReLU를
포함한 전체 모델의 역함수가 lift rank 증가만으로 보장되지 않는다.

이번 현재 소스 CUDA 검사:

- solver objective 독립 식·유한 K 방향 미분 및 row/symmetric 전파: 12개.
- nonzero energy/readout/lift를 사용한 내부 16조건의 dense 출력·gradient: 16개.
- full/cluster/cluster_disjoint 실제 준비→CUDA 학습→optimizer→평가·label isolation: 9개.
- 합계 **37 passed, failures/errors/skips 0**, 52.09초. 이전 105/13 또는 reviewer 267과 합산하지 않는다.

전체 모델 dense 기준은 C/beta 생성기를 공유한다. 이를 보완하는 solver 방향 미분은
선택한 방향/입력에서의 검사이며 모든 입력의 안정성·고차 미분 증명이 아니다.
기존 custom Gram은 first-order 계약이다. JVP/2차 미분을 별도 보장하지 않는다.

## 7. R1: 4조건 study의 타당성과 추가 통제

F0=full/fixed C, F1=full/dynamic C, S0=sampled/fixed C, S1=sampled/dynamic C.
현재 per-head/K8/backbone을 유지하고 core에서는 energy/lift off로 동일하게 둔다는
제안은 연구 질문을 분리한다. 기존 16/21조건을 삭제하거나 규모를 축소하는 제안이 아니다.
F0의 C 고정은 beta와 backbone도 고정한다는 뜻이 아니다.

현재 incidence-only sampling은 허용돼 있다(`runner.py:107–108`). 부족한 것은
sampling 구현 전부가 아니라 네 cell의 통제된 실행·중단/재개·감사·대비식·보고 통합이다.

이미 충족한 부분을 다시 결함으로 만들지 않는다.

- sampler는 private torch.Generator를 사용한다. 같은 seed/epoch/law에서 전역 RNG를
  서로 다르게 소비시켜도 S0/S1의 node IDs/edges/seeds가 같았다.
- cluster 및 cluster_disjoint 합성 입력 각각에서 96개 train seed가 정확히 한 번씩
  사용됐다. 실제 arxiv 전체 coverage 증거는 아니다.
- 같은 seed=613에서 reference fixed/dynamic의 encoder/decoder와 모든 공통
  value/output projection/beta 파라미터가 동일했다. V5의 fork_rng가 dynamic-only
  초기화를 격리한다. 공통 초기값이 깨졌다고 추측할 근거가 없다.
- 실제 arxiv 차원에서 fixed 1,641,960 / per-head dynamic 10,702,824 파라미터를 확인했다.
  공통 초기화와 parameter matching은 다른 계약이다.

반드시 보완할 통제:

1. 일반 cluster는 `budget=seed_count*(1+sum(fanouts))`이므로 physical seed batch를
   바꾸면 context도 바뀐다. 이를 실행 자원만의 변수로 취급하면 안 된다.
   cluster_disjoint에서는 고정 context seed 수와 독립 context 묶음 수를 분리할 수 있다.
   두 C 조건에서 동일한 context sequence를 hash로 연결하고 calibration이 법칙을
   바꾸지 않게 한다. graph context가 서로 섞이지 않는 disjoint 구성도 검사한다.
2. 현재 `reference_updates`는 update 수의 최소 보존/연장 정책이지 full/sample
   update equality가 아니다. 이번 합성 계획에서도 full은 200, sampled는 600 updates다.
3. 같은 planned epoch여도 early stopping이 다르면 실제 label 노출량과 validation
   횟수가 다르다. 엄격한 exposure-matched 비교는 고정 감독 pass/관찰 시점 계약을
   별도로 명시해야 한다. 조기 종료 실용 비교는 실제 exposure·update·관찰 횟수를
   기록하고 동일 예산이라고 표현하지 않는다. 기존 production 정책을 몰래 바꾸지 않는다.
4. matched-update 진단은 한 pass의 CE sum / 전체 감독 seed 수를 누적하고 clipping과
   optimizer/weight decay를 pass 끝에 한 번 적용한다. batch별 mean 합산, batch별
   clipping/step이면 같은 진단이 아니다. dropout·context 의존 함수가 full과 같아지는
   것은 아니며 이 진단의 처리량을 일반 minibatch 학습 성능과 합치지 않는다.
5. 모든 train seed의 횟수·tail batch·중복을 원 ID로 검증한다. context 동반 노드 label은
   supervision에 넣지 않는다. model seed·sampler seed·graph-view·법칙·실제 예산을 고정한다.

효과식은 D2/D3와 같이 ΔC_full=F1−F0, ΔC_sampled=S1−S0,
Δsampling_dynamic=S1−F1, interaction=S1−S0−F1+F0로 정의할 수 있다.
이것은 선택한 학습 전략/용량의 조건부 성능 대비이며 C 구조만의 순수 원인 효과나
parameter-matched 우위를 자동 보장하지 않는다. 여러 seed의 짝지은 차이를 보고하고,
표준편차를 표준오차/유의성으로 바꾸지 않는다.

현재 omega는 degree-ratio 보정이며 inclusion probability 역수가 아니다. sample마다
C와 degree/context도 재계산되므로 full 연산/gradient의 불편추정량으로 부르지 않는다.
같은 hidden state에서 단층 alpha=a/d, beta, 누락 이웃 질량을 비교하는 진단과
실제 다층 prediction 변동을 분리하는 제안은 타당하다. C를 context 간 강제로 같게
하는 새 loss는 진단 없이 추가하지 않는다.

## 8. R2: 정확한 계산 재배치와 실측 비용

Gram 뒤가 선형 readout이면 합의 순서를 바꾼 D3 식은 타당하다.

\[
E_{ihr}=\frac12\sum_{e\ni i}a_{eh}
\sum_{k\le\ell}R_{h,k\ell,r}\langle\Delta_{eh}^{k},\Delta_{eh}^{\ell}\rangle.
\]

현재 value projection에 bias가 없으므로 delta=(H_v−H_u)W도 실수 대수상 같다.
Gram 이후 nonlinear normalization이 들어가면 이 융합을 그대로 적용할 수 없다.

별도 CUDA prototype에서 shared/per-head × diagonal/full × edge chunk 1/3/8을
FP64, FP32, BF16 autocast 각각 검사했다. production 모델/커널에는 적용하지 않았다.

| 정밀도 | 출력 최대 절대차 | H/W/C/omega/readout 미분 최대 절대차 | 출력 상대 L2 최대 |
| --- | ---: | ---: | ---: |
| FP64 | 2.22e-16 | 1.33e-15 | 2.76e-16 |
| FP32 | 2.38e-7 | 7.15e-7 | 1.76e-7 |
| BF16 autocast | 0.0034233 | 0.03125 | 0.0050845 |

FP64/FP32 24경우는 정한 출력·gradient 허용오차를 통과했다. BF16 12경우는 차이를
측정한 것이며 production 동등성 승인으로 세지 않는다. projection 전후 차분,
readout과 weight 곱의 순서, 누적 dtype이 반올림 지점을 바꾸기 때문이다.
관측한 차이가 task 정확도를 해친다는 증거도 아니다. 기존 autocast 계약과 비교할
허용오차/학습 안정성 기준을 먼저 정하고 nonzero readout·checkpoint 재계산까지 검사한다.

메모리를 줄였다고 속도가 자동으로 빨라지는 것은 아니다. node projection 재사용을
edge projection으로 바꾸면 E/N에 따라 연산이 늘고, node readout을 edge readout으로
바꾸면 readout 반복도 늘 수 있다. autograd prototype의 합치만으로 full gradient
buffer를 한 번만 할당하는 최적화된 backward가 구현됐다고 말하지 않는다.

모든 pair와 깊이를 유지하고 history/W/C/omega/readout gradient를 검산한 뒤,
warmup 후 동일한 실제 graph/precision에서 forward+backward+optimizer를 비교한다.
peak allocated/reserved, steady state, wall time을 함께 측정한다. CUDA event time과
CPU wait는 겹치므로 합산하지 않는다. 현재 StageTimer는 sampling/forest/transfer를
묶고 forward에는 C와 Gram이 함께 들어간다. epoch elapsed는 checkpoint I/O 이전에
기록된다. 원래 목표에 필요한 cost 분해에는 이 부분과 validation/audit/I/O를 추가로
구분해야 한다. 감사 비용을 빼서 end-to-end speedup처럼 표시하지 않는다.

Arxiv 마지막 층 projected history 약 1.29 GiB, Gram 약 186 MiB라는 shape 계산은
타당하지만 전체 peak 추정이 아니다. MIG10GB의 fit/불가능 어느 쪽도 이번 자료로 확정하지 않는다.
C cost/finite-K/전파/Gram을 측정하기 전에 K·폭·head 수를 줄이지 않는다.
현재 node_projection은 이미 head 공통이므로 그것을 새 공유 최적화처럼 제안하지 않는다.
context metric 저랭크화는 가설 공간 변경이며 별도 모델/버전 실험이다.

## 9. R3: temporal view는 별도 protocol이며 독립 그래프 실험은 아니다

Train induced view → train+validation view → 연도별 관측 view의 제안은
공식 transductive 평가와 분리하면 의미가 있다. 다음 전제가 더 필요하다.

1. 현재 `_graph()`는 x/y/edge_index/incidence만 저장한다
   (`research/conductance_gat/benchmark_data.py:76–88`). **node_year가 없다.**
   official split만으로 test 내 연도 순서를 복구하지 않는다. 공식 원본의 year와
   node mapping을 provenance/hash에 묶은 별도 schema/version 또는 검증된 sidecar가
   필요하다. 기존 데이터 캐시를 덮어쓰거나 연도를 추정하지 않는다.
2. 학습 view를 만든 뒤 full_degree, graph_structure, pooled feature context, sampler
   correction, forest, normalization, cache를 모두 그 view에서 생성해야 한다.
   여기서 full_degree의 full은 전체 미래 graph가 아니라 **허용 parent view**다.
3. cache key에는 단순 train/val 이름뿐 아니라 protocol/view/content identity가 필요하다.
   global↔local node/edge mapping과 cutoff, 포함 ID 집합을 검증한다.
4. G[V_≤t]는 노드 연도에 따른 관측 규약이다. edge의 실제 최초 관측 시각을 재현한
   것인지, 양 끝점 연도로 포함시키는 proxy인지 명시해야 한다. 같은 연도 노드를
   모두 함께 보는 연말 batch 평가와 순차 online 도착 평가도 구분한다.
5. 제공 특징을 그대로 쓰는 조건부 모델 평가이며 특징 생성의 역사적 순수성을
   보장하는 실험으로 확대하지 않는다. train 통계로 학습하는 추가 정규화만 제한하는
   것과 외부 제공 특징 전체의 출처를 새로 만드는 것은 다르다.
6. 미래 노드 topology/features/labels 변화가 train sample/output/gradient에 영향을
   주지 않는 negative test를 둔다. 평가 시 theta뿐 아니라 backbone, buffer, optimizer,
   persistent cache 의미가 변하지 않는지 검사한다. C/degree의 입력별 재계산은 허용한다.
7. official full-graph checkpoint를 custom inductive 학습 결과로 재분류하지 않는다.
   별도 run ID와 표가 필요하다. 연도별 test 결과를 보고 설계를 재선택하지 않도록
   전체 cutoff/평가 정책을 먼저 고정한다.

이 프로토콜은 확장된 동일 그래프의 새 노드에 대한 temporal induction이다.
서로 독립적으로 수집된 새 그래프 일반화를 대신하지 않는다. 독립 graph 주장은
별도 데이터/프로토콜이 충족돼야 하며 이를 명분으로 PPI를 다시 채택하지 않는다.

## 10. R4: 메커니즘 및 외부 비교

16조건은 energy none/diag/full × lift none/linear/pre/post의 12개와 fixed/shared C
energy off/on 4개다. 21조건에 GCN/SAGE/GATv2/DUAL/no-skip이 포함된다.
기존 비교는 유지하고 core study와 목적·보고를 구분하는 제안이 맞다.

- Fixed/shared/per-head는 용량도 달라진다. diag/full도 readout 파라미터가 늘어난다.
  모듈의 총 효과와 동일 용량의 구조 효과를 혼동하지 않는다.
- Linear-lift는 중복 채널을 통한 재파라미터화이므로 no-lift와 최적화 경로까지 같은
  대조가 아니다. inference intervention은 fresh-training ablation을 대체하지 않는다.
- 동일 8/12층 common recipe는 각 family의 최고 튜닝 성능이 아니다. 각 모델 tuning은
  validation만 사용하고 시도 수/시간/parameter/compute budget을 공개한 별도 연구다.
- Sampled C를 full-only baseline과 비교한 속도 차이는 sampling 기여와 섞인다.
  같은 declared sampler의 GCN/SAGE 대조를 별도 추가하는 방향이 적절하다.
  단순 CLI 차단 해제만으로 끝낼 수 없다. SAGE mean/GCN normalization에 omega를
  적용할지 여부 자체가 새로운 operator 규약이므로 이름·수식·cost를 명시한다.
- custom cluster sampler를 GraphSAINT/Cluster-GCN 재현으로 개명하지 않는다.
  EVD-free propagation 자체가 최초 novelty라는 주장도 이 검수로 도출되지 않는다.

## 11. 실행 순서와 완료 기준

| 단계 | 완료를 판정할 근거 |
| --- | --- |
| R0 | 세 반례 및 상태/시간 추가 반례 차단, nonzero branch 출력·gradient 보존, debug/production 승인 분리 |
| R1 | 4 cell plan/calibration/resume/audit/report 연결, 실제 원 ID exposure와 context 순서, 예산 계약 및 대비식 |
| R2 | 수식·미분·dtype 계약, nonzero readout, 전체 규모의 실제 CUDA 비용/메모리 비교 |
| R3 | year provenance와 view identity, 미래 정보 negative tests, frozen 평가; official/custom 분리 |
| R4 | 기존 ablation/external 연구를 명시적 seed/tuning/cost 규약으로 수행 |

R1의 실행기 구현과 실제 arxiv 4 run 완료를 같은 항목으로 체크하지 않는다.
R2 대수 검산도 최적화 코드와 실측 가속을 각각 구분한다. 현재 연구 가설에 직접 필요하지
않은 cycle/forest/negative-edge/exact-distance-shell/inverse loss를 기본 모델에 동시에
추가하는 제안에는 동의하지 않는다. 기존 별도 연구 코드는 보존한다.

## 12. 이번 실행 한계와 보존 확인

- 로컬 GPU: RTX 5070 Ti 16GB, 관측 free 14,102 MiB, Torch 2.13.0+cu130.
  CPU logical 16, 관측 가용 RAM 약 46.45 GB. 서버 A100 MIG 실측이 아니다.
- 모든 새 모델 forward/backward는 CUDA였다. CPU는 파일 검증, sampling/topology,
  metadata 및 결과 기록에 사용했다. 모델·가중치·데이터 다운로드는 없었다.
- 초기 기존 fixture 재사용은 새 packaging script가 source hash 집합에 추가돼
  거부됐다. guard를 우회하지 않고 현재 소스로 새 합성 debug checkpoint를 만들었다.
- 반례 스크립트의 reference 모델 생성에 edge_chunk_size를 빠뜨린 중간 실행도
  실패 로그에 남겼다. 인자를 보완한 최종 실행은 완료됐다. 이것은 모델 결함이 아니다.
- 검산 prototype의 cuBLAS 초기 context/텐서 scalar 변환 경고와 pytest 경고는
  로그에 보존했다. 최적화된 production code라고 포장하지 않는다.
- 실제 arxiv 본학습·공식 성적·다중 seed 통계·A100 MIG fit·전체 가속은 미실행이다.
- production 소스와 기존 결과는 바뀌지 않았다. 이 검토로 알려진 R0 결함이
  수정됐다고 주장하지 않는다. 수정 작업은 이 보고서의 반례와 완료 기준을 따라야 한다.
