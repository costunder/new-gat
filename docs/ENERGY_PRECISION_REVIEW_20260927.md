# 3eb0eb6 검수 후 energy readout 정밀도 보완

검수 첨부 `9169bb9c-7b47-4c7d-8738-f48d0eb854c1` 본문을 실제 코드와 대조했다.
외부 검수의 CPU Torch 2.10 결과 및 sandbox 파일을 여기서 직접 재현·열람한 것으로
표현하지 않는다. 아래는 로컬 CUDA에서 별도로 수행한 검사다.

## 구현 변경과 실험 계약

reference energy readout의 `einsum("nhp,hpd->nhd", ...)`을 명시적인
`torch.autocast(..., enabled=False)` 안에서 FP32로 계산한다. `.float()`만으로는
autocast의 BF16 변환을 막을 수 없다. fused readout은 이미 같은 FP32 정책이었다.
history projection, propagation, output projection 등의 바깥 AMP 정책은 유지한다.

**dtype만 맞추는 것으로 충분하지 않았다.** 실제 BF16을 켠 첫 영향 회귀는
`energy-precision-focused-20260927.xml`에 **92 passed / 20 failed**로 보존했다.
실패는 energy 조건의 checkpoint off/on fused/reference gradient 비교다.
이는 최종 수정 전 중간 검사이며 최종 통과 개수에 합산하지 않는다.

기존 fused는 엣지별 readout을 먼저 적용한 뒤 노드로 합산했다. reference와 수학적으로
같아도 FP32 합산 순서가 달라지고, 다음 BF16 cast와 ReLU를 지나며 오차가 커졌다.
따라서 optional fused의 구현을 `reference_order_gram_recompute_v2`로 변경했다.
두 구현 모두 **노드별 Gram 집계 후 readout** 순서를 사용하며, fused는 forward에서
생성한 Gram을 backward까지 저장하지 않고 입력 history/C/readout에서 재계산한다.
backward도 reference의 autograd 순서와 LocalGram의 1회 gradient buffer 할당을 쓴다.
지원하는 1차 미분 범위는 유지하며 고차 미분을 새로 지원한다고 주장하지 않는다.

**현재 fused는 transient node×head×pair Gram을 생성한다.** 이전 버전의 “그 tensor를
생성하지 않는다”는 설명은 현재 구현에 해당하지 않는다. 같은 CLI 이름은 유지하지만
configuration/model contract의 실행 정책이 달라져 변경 사실이 기록된다. 기존 결과로
메모리 감소나 가속을 주장할 수 없으며 재계산 비용과 peak VRAM은 다시 측정해야 한다.

정책 이름 `fp32_autocast_disabled_v1`을 configuration의 `comparison_contract`와
energy 모델의 `model_contract`에 기록했다. Energy 없는 모델의 model contract는
이 항목을 `None`으로 표시한다. 수식·모델 크기·데이터·학습 예산의 변경은 없다.
기본 구현은 reference이고 fused는 선택 사항이다. F0/F1/S0/S1의 energy-off 계산은
이 변경의 적용 대상이 아니지만 소스 identity는 갱신된다.

이는 **AMP 수치 계약 변경**이다. 기존 run, checkpoint, 결과, 검수 ZIP을 보존한다.
새 소스에는 **새 run ID**가 필요하며 이전 소스 결과를 같은 실험에 혼합하지 않는다.
소스 호환성 검사와 validation의 정답 개수 재현 기준을 완화하지 않았다.

## 추가로 발견한 기존 테스트 오류 — BF16 표시 정정

`tests/test_aggregation_comparison_cuda.py::reference_arguments`는 precision을 먼저
설정한 뒤 `engine.validate_args()`를 호출했다. 검증 과정의 hardware profile 해석이
portable precision을 FP32로 다시 설정했다. 따라서 이 helper를 사용하는 기존 모델
검사의 BF16 parameter label은 **실제 CUDA BF16 실행을 증명하지 않는다**.

이 helper를 사용하는 aggregation 모델 학습/복원 검사, fused integration 검사,
revision paths의 전체 모델 검사가 해당한다. 과거 110/152개 XML을 삭제하거나 숫자를
새 결과에 합산하지 않는다. 독립 raw-kernel의 BF16-quantized 입력 검사와 실제 AMP
모델 검사는 다른 검증이다. 외부 검수자가 별도 실행한 CPU BF16 기록도 우리 CUDA
기록으로 바꾸어 해석하지 않는다.

합성 테스트 helper는 production 설정 검증 **후** 테스트 행렬의 precision을 적용하고,
실제 CUDA autocast enabled 여부와 autocast dtype을 assert하도록 수정했다. 이는
명시적 합성 검증만의 override다. production hardware profile은 수정하지 않았다.

수정 전 조사 기록:

- `energy-precision-before-20260927.xml`: 기존 helper로 1 passed. FP32 대조 기록이며
  BF16 통과 기록이 아니다.
- `energy-precision-before-actual-20260927.xml`: 실제 BF16 활성화 후 shared-energy
  reference caller에서 `energy readout must disable AMP` 검사가 **1 failed**.
  외부 CPU 반례의 logit/gradient 수치를 직접 재현한 결과라고 주장하지 않는다.

## 검증 범위

모델 검사는 RTX 5070 Ti 16GB, CUDA PyTorch 환경에서 실행한다. reference 8층,
hidden 256, 8 heads를 유지하며 4개의 64-node 합성 그래프를 하나의 disjoint batch로
처리한다. synthetic debug 입력은 단위/통합 검증용이고 arxiv 본학습 데이터가 아니다.
energy readout은 0이 아닌 값으로, lift projection은 중립 초기값에 비영 perturbation을
더한 상태로 검사한다. CPU는 메타데이터/제어 검사에만 사용한다.

검사 환경은 PyTorch 2.13.0+cu130, PyG 2.8.0.post1, 논리 CPU 16개, RAM 약 64 GiB다.
실행 전 GPU 전체 16,303 MiB 중 14,335 MiB가 비어 있었다. 검사 fixture는 deterministic
algorithms를 켜고 TF32를 끈다. 이 조건에서의 정확한 재현 검사를 production CUDA의
항상 bitwise 같은 실행 보장으로 확대하지 않는다. production recipe는 변경하지 않았다.

- 실제 per-head/shared/fixed/diagonal energy caller, FP32/BF16, reference/fused,
  checkpoint off/on에서 readout 시 autocast 비활성화와 FP32 입출력을 직접 계측한다.
  backward에서 입력·모든 파라미터 gradient의 존재 및 유한성을 검사한다.
- 16개 incidence 조건 × FP32/BF16 × checkpoint off/on에서 reference/fused 출력,
  입력 및 모든 파라미터 gradient를 비교한다. 기존 절대 오차 비교에 더해 energy
  readout gradient의 상대 L2 오차를 별도로 제한한다(FP32 0.0005, BF16 0.02).
- 실제 ReLU 직전 값 중 절댓값 0.001 미만의 양수·음수가 모두 존재하는지 확인하고,
  checkpoint off/on의 preactivation, logit, 입력·파라미터 gradient를 정확히 비교한다.
  모든 입력에서 fused/reference bitwise 일치나 ReLU 경계 안정성을 보장하는 것은 아니다.
- backward까지 저장하는 tensor가 입력 history/C/readout/edges 네 개뿐인지 검사하고,
  세 부동소수 입력의 requires-grad 조합 7개에서 출력과 필요한 gradient를 reference와
  정확히 비교한다. fixed-C처럼 일부 입력에 gradient가 없는 경로도 포함한다.
- 진단 collector 및 energy/cross/diagonal 개입의 수동성은 같은 구현 내에서 logit
  `rtol=0, atol=0`으로 확인한다. fused/reference 근사 동등성과 구분한다.
- 기존 raw kernel 수학, dense 기준식, deepcopy, validation 제어, 공식/시간순 핵심
  4조건 및 GCN/SAGE 학습·감사·test 보존 경로의 영향 회귀를 함께 실행한다.

최종 실행 결과와 실행 환경은 이 문서 하단 및 ZIP의 `VERIFICATION.json`에서 확인한다.

## 최종 실행 결과

`results/energy-precision-after-20260927.xml`: **230 passed / 0 failed / 0 errors /
0 skipped**, 696.57초. CUDA 모델·수학·통합 검사 220개와 validation 제어 검사 10개다.
정적 검사(Ruff, compileall, git diff --check)도 통과했다. 2,743개의 라이브러리
deprecation/JUnit 등의 warning은 로그에 남겼으며 실행 실패나 skip으로 숨기지 않았다.

| 검사 모듈 | 통과 수 |
|---|---:|
| test_energy_precision_cuda | 43 |
| test_fused_integration_cuda | 80 |
| test_revision_paths_cuda | 29 |
| test_revision_integrity_cuda | 2 |
| test_aggregation_validation (제어 검사) | 10 |
| test_all_arms_dense_cuda | 16 |
| test_core_conductance_cuda | 2 |
| test_arxiv_baselines_cuda | 3 |
| test_aggregation_comparison_cuda | 45 |

16조건 × 2 precision × checkpoint off/on의 64개 비교에서 기록한 energy-readout
gradient 최대 상대 L2 오차는 **0.0**이었다. Energy 없는 조건은 이 통계에 대해 0을
기록하므로, energy 조건의 개별 property도 함께 확인한다. 실제 AMP가 켜진 비교는
32개다. ReLU 검사는 각 FP32 모델에서 경계 0.001 이내의 비영 입력 993개, BF16에서
865개를 포함했고 양수·음수가 모두 있었다. checkpoint off/on preactivation·출력·
gradient의 정확한 동일성 검사가 통과했다. 이는 지정한 합성 입력·seed·환경의 결과다.

비교 모델 학습 smoke의 기록된 peak allocated VRAM은 79,369,216–284,659,712 bytes다.
작은 합성 그래프의 검사 메모리이며 arxiv full graph 또는 MIG 적합성 수치가 아니다.
실제 arxiv 데이터로 calibration/전체 학습/전체 평가를 수행한 것이 아니다.

최종 실행한 모듈 목록은 다음과 같다. CUDA가 없으면 실패하며 CPU 모델 fallback은 없다.

```text
python -m pytest tests/test_energy_precision_cuda.py tests/test_fused_integration_cuda.py
  tests/test_revision_paths_cuda.py tests/test_revision_integrity_cuda.py
  tests/test_aggregation_validation.py tests/test_all_arms_dense_cuda.py
  tests/test_core_conductance_cuda.py tests/test_arxiv_baselines_cuda.py
  tests/test_aggregation_comparison_cuda.py -q --tb=line --show-capture=no
  --junitxml=results/energy-precision-after-20260927.xml
```

위 목록은 가독성을 위한 줄바꿈이며 실행 시 한 명령으로 전달한다. 과거 110/152개,
중간 112개, 개별 재현 실행을 이 230개에 합산하지 않는다. 기존 결과 파일을 보존하며
재실행 시 XML/log 출력도 새 경로를 사용한다.

## 여전히 실측하지 않은 항목

실제 ogbn-arxiv calibration, 본학습, 공식/시간순 test, 전체 시간·VRAM·샘플 포화율,
A100 MIG 10GB 적합성은 미실행이다. 로컬에 arxiv 데이터 캐시가 없고 원격 서버 연결을
확인하지 못했다. 모델·가중치·데이터를 다운로드하지 않았다.

기존 합성 microbenchmark에서 이전 fused가 reference보다 느렸다는 기록은 유지한다.
그 측정은 재계산 방식으로 바뀐 현재 fused의 성능 측정이 아니다.
이번 정밀도 보완을 속도 향상, SOTA 또는 독립 그래프 일반화의 증거로 제시하지 않는다.
다음 연구 실행의 중심은 arxiv full/sampled × fixed/learned C의 실제 정확도와 전체
비용 측정이다. energy/lift는 추가 표현력 축이며 핵심 네 조건을 대체하지 않는다.
