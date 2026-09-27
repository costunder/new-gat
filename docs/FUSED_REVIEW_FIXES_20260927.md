# 4c2d7f4 재검수 후 fused 연결·진단 수정

**이 문서는 3eb0eb6의 과거 기록이다. 이후 BF16 테스트 helper의 FP32 덮어쓰기가 발견됐다.
아래 BF16 모델 검사 표시는 실제 AMP 검증으로 인정하지 않는다. 최신 수정과 정정은
[energy 정밀도 후속 검수](ENERGY_PRECISION_REVIEW_20260927.md)를 따른다.**

검수 첨부 `79a74552-0239-428b-a1a2-970b87f072ab`의 본문 전체를 읽고 실제 소스와 대조했다.
첨부가 링크한 외부 sandbox ZIP/스크립트는 로컬에 제공되지 않았으므로 읽었다고 주장하지 않는다.
이 문서는 이전 `REVIEW_REMEDIATION_20260927.md` 이후의 추가 보완 기록이다.

## 확인된 결함과 수정

1. **shared/fixed energy의 fused 호출 shape 오류**
   실제 estimator는 C를 `(E,)`로 반환하지만 kernel은 `(E,1)` 또는 `(E,H)`를 받는다.
   kernel 테스트만으로 놓쳤던 실제 caller 연결 오류다. custom autograd 바깥에서
   `metric[:, None]`으로 통일하여 shared estimator의 원래 gradient shape도 보존했다.
2. **진단 수집이 예측 경로를 바꾸는 오류**
   collector 유무가 fused/reference 분기를 결정하지 않도록 바꿨다. 진단용 Gram은
   별도로 계산하고, 예측 branch는 항상 선언한 구현을 사용한다. fused의 energy-off,
   cross-off, diagonal-off 개입은 같은 kernel에 readout mask를 전달한다.
   reference는 기존 Gram-mask/readout 순서를 유지한다.
3. **기존 validation fixture의 누락 필드**
   `test_aggregation_validation.py`의 명시적 mock에 `visibility_protocol`을 추가했다.
   실제 parser 검증을 느슨하게 하거나 production에서 누락 필드를 조용히 대체하지 않았다.
   이 모듈을 이번 필수 회귀 목록에 포함했다.

정답 개수의 정확한 재현 기준은 바꾸지 않았다. 모델/조건/수식/깊이/너비/solver K/학습 예산을
추가하거나 줄이지 않았다. reference와 energy-off F0/F1/S0/S1은 계속 기본 경로다.
소스 identity는 변경된다. 기존 실행 결과와 ZIP을 보존하며, 이 커밋으로 새 학습을 시작할 때는
새 run ID를 사용한다. 이전 소스의 checkpoint를 변경된 소스와 같은 실험으로 자동 승인하지 않는다.

## 수정 전 직접 재현

RTX 5070 Ti에서 reference 크기 8층/256 hidden/8 heads 및 명시적 합성 그래프를 사용했다.
모델이나 데이터를 다운로드하지 않았다.

- fixed/shared energy 두 모델: 실제 fused forward에서 동일한 `IndexError` 재현.
- per-head energy-pre-lift: collector 전후 FP32 logit 최대 절대차 `1.7881393432617188e-7`.
  정확한 tensor 동일성 검사가 실패했다. 실제 arxiv 성능 변화의 관측값은 아니다.
- 선택 실행 결과 **3 failed, 1 passed, 44 deselected**. 실패가 수정 전 결함 증거다.
  원본 기록은 `results/fused-review-before-20260927.xml` 및 `.log`에 보존했다.

## 이번 회귀의 검사 내용

`tests/test_fused_integration_cuda.py`:

- 실제 16개 incidence caller × FP32/BF16, reference와 fused의 출력·입력 gradient·모든
  파라미터 gradient 비교. 중립값에 숨지 않도록 energy readout을 비영으로 설정.
- per-head/shared/fixed/diagonal energy × FP32/BF16 × reference/fused에서 collector 전후
  **logit을 rtol=0, atol=0으로 비교**. RNG 및 모델 state 불변도 검사.
- energy-off/cross-off/diagonal-off에서도 동일 검사. fused kernel이 매 레이어 실제로
  호출되는지 계측하고, 전달된 readout이 의도한 mask와 정확히 일치하는지 확인.

기존 raw kernel 수학·전체 모델·deepcopy/checkpoint·dense 기준식·원래 validation 제어 검사,
공식/시간순 4조건의 학습/감사/test 저장·재사용 경로도 영향 회귀에 포함했다.
출력/gradient의 fused-vs-reference 근사 비교와 observer 전후 **정확한 동일성**은 별도 조건이다.
BF16 연산 재배치가 reference와 bitwise 동등하다는 주장은 하지 않는다.

최종 결과는 이번 ZIP의 `VERIFICATION.json`의 `current_affected_regression`과
`results/fused-review-after-20260927.xml`을 기준으로 확인한다.
**110 passed, 0 failed, 0 errors, 0 skipped**, 366.05초. CUDA 모델/수학/통합 검사
100개와 명시적 mock을 이용한 validation 제어 검사 10개다. 모델 계산을 CPU로 옮기지 않았다.
기존 라이브러리 deprecation/JUnit warning 1,418개는 로그에 남아 있다.
Ruff와 `git diff --check`도 통과했다.
이전 152개 기록은 `4c2d7f4`의 과거 검사로 따로 보존하며 새 검사에 합산하지 않는다.

## 그대로 남는 실측 경계

이 변경은 fused를 속도 개선판으로 바꾸지 않는다. 기존 합성 microbenchmark의 느린 측정값을
유지하고 reference 기본값도 유지했다. diagnostic Gram의 추가 시간·메모리도 사라진 것이 아니다.
실제 arxiv calibration/본학습/정확도/전체 시간·VRAM, A100 MIG 10GB 적합성은 미실행이다.
시간순 view 실험을 독립 그래프 일반화나 완전한 시간 인과성 실험으로 부르지 않는다.
