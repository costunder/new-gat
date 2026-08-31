# NEW GAT 연구 프로젝트 Hand-off

작성 기준일: 2026-09-01 (Asia/Seoul)

이 문서는 외부 ChatGPT 또는 연구 리뷰어가 저장소를 처음 받아도 수학적 가설, 구현 경계,
데이터 계약, 실행법, 검증 범위와 미완료 항목을 혼동하지 않도록 만든 인수인계 문서다.
원문 코드는 같은 폴더의 `code_summary.md`에 파일별로 들어 있다.
처음 설치·실행하는 사용자는 [README.md](README.md)의 순서를 따른다.
이 문서는 실행 입문서가 아니라 연구·구현 교차검토용이다.

## 0. 리뷰어가 먼저 알아야 할 판정

### 2026-09-01 C-learning 결과 수령과 같은 run의 읽기 전용 검사

`gat-c-learning-seed0-v1`의 PPI/arxiv × learned/fixed C × seed 0 보고서를 수령했다.
비교표와 네 조건은 모두 `passed`, 평가는 validation뿐이며 test는 평가하지 않았다.

| 데이터 | Learned C (%) | Fixed C=1 (%) | Learned − fixed (pp) | Best / 실행 epoch (learned; fixed) |
|---|---:|---:|---:|---|
| PPI micro-F1 | 52.564966 | 52.705738 | −0.140772 | 64 / 114; 90 / 140 |
| ogbn-arxiv accuracy | 68.317723 | 68.324435 | −0.006711 | 195 / 200; 195 / 200 |

이 seed·설정에서는 **학습 C의 validation 이득을 관측하지 못했다.** 일반적인 동등성,
C의 보편적 무용성, 유의성·SOTA를 입증한 것은 아니다. PPI의 learned C CV는
0.541124 / 0.127339로 비상수지만 점수 이득이 없고, arxiv는 0 / 0.00948437로 작다.
새 PPI learned 점수 52.564966%를 과거 2×2 node-degree의 52.465469%와 합치거나
대체하지 않는다. 실행 간 차이의 원인은 현재 자료만으로 특정하지 않는다.

활성 학습 파라미터는 PPI 36,347→11,385, arxiv 36,074→11,112로 약 69% 적다.
Fixed 조건은 gate scaffold 24,962개를 동결 보존하므로 전체 저장 파라미터·메모리·속도
감소를 뜻하지 않는다. Fixed 행의 gate L2는 실행하지 않는 scaffold의 norm이다.

근거는 사용자 inline 비교표이며 별도 첨부 파일/SHA-256이 없다. C-learning 구현은
`25ca328`에 게시됐지만 표에는 실제 서버 source revision이 없으므로 실행 commit으로
단정하지 않는다. 원격 manifest·checkpoint 전체를 로컬로 받아 재검증한 것도 아니다.
정확한 8개 진단 행과 해석은
[C-learning 결과 문서](docs/CONDUCTANCE_C_LEARNING_FINDINGS.md)에 보존했다.

다음은 **그 새 run의 learned checkpoint**를 대상으로 하는 평균-C 검사다.
`c_learning/audit.sh`는 source manifest의 suite를 읽어
`conductance_c_learning/learned_c`와 과거 `conductance_factorial/node_degree`를 구분한다.
이번 검사에는 `results/conductance_gat/c_learning/gat-c-learning-seed0-v1`을 넣는다.
원 validation·source/cache/checkpoint 무결성을 확인한 뒤 C를 그래프·층별 평균으로 바꾸고
weighted degree를 다시 계산한다. 전체 층과 한 층씩 개입을 분리한다.
원본 결과를 수정하지 않고 `results/conductance_gat/c_learning_audits/`의 새 폴더에
`report.md`/`audit.json`만 기록한다. **새 학습·optimizer step·test 평가는 없다.**
이 확장 검사의 실제 GPU 출력은 아직 수령하지 않았다. 실행은
[C-learning README](research/conductance_gat/c_learning/README.md)의 첫 번째 명령을 따른다.

이번 확장 후 로컬 전체 회귀는 **924 passed / 64 skipped** (32.11 s, exit 0),
Ruff 통과다. 기존 890개에서 34개 검사를 추가했다. 새 run의 두 실제 단위 학습 루프에서
만든 checkpoint를 검사기로 읽는 통합 경로, learned/fixed 및 suite 혼동 차단,
원 validation 재현 실패 차단, 원본 바이트 보존을 확인했다. 평균-C 개입은 학습된 다른
가중치를 그대로 둔 C=1 교체와도 독립 대조했다. 이는 작은 fixture의 CPU 단위 검증이며
연구용 CPU 학습·공식 데이터 다운로드·실제 GPU 실행이 아니다. 생략된 검사는 Linux/Bash
62개, 로컬 PyG 미설치 1개, Windows 실제 symlink 권한 1개다.

### 2026-08-31 2×2 GPU 결과 수령과 다음 C-learning 실험

사용자 서버에서 `gat-factorial-seed0-v1`의 **PPI/arxiv × 4조건 × seed 0** 학습과 비교표가
모두 `passed`로 끝났다. 실행 소스는 `43afd632b97a4285dfeae26847b4f12a8fd1a1e4`다.
NVIDIA RTX A6000, Python 3.11.16, Torch 2.7.1+cu118, PyG 2.7.0, Linux glibc 2.35다.
첨부 텍스트 SHA-256은 `2C78D02BB210BF00865AB7207DF651B02B2081EE4FAE6E8A6A83665A5D331161`이다.
이 hash는 checkpoint hash가 아니다. 서버 원본 artifact 전체를 직접 가져와 재검증한 것은 아니다.

| 조건 | PPI validation micro-F1 (%) | arxiv validation accuracy (%) |
|---|---:|---:|
| baseline | 48.986770 | 50.927883 |
| gate_no_wd | 49.378028 | 50.565451 |
| node_degree | 52.465469 | 68.317723 |
| node_degree_gate_no_wd | 50.340520 | 67.995566 |

두 데이터 모두 `node_degree + gate WD 0.0005`가 최고다. Baseline 대비 각각
**+3.478699pp / +17.389840pp**이며, node-degree 아래 WD를 제거하면 각각
−2.124949pp / −0.322157pp다. Gate WD 제거로 C 변동과 gate norm이 커졌지만
그 자체가 성능 개선은 아니었다. arxiv 최고 조건의 C CV가 0 / 0.00948423이므로
학습 C의 순수 기여는 여전히 미확정이다. PPI 최고 조건에는 비상수 C가 남아 있다.
정확한 epoch, 대비, 층별 진단과 해석은
[CONDUCTANCE_FACTORIAL_FINDINGS.md](docs/CONDUCTANCE_FACTORIAL_FINDINGS.md)에 보존했다.
단일 seed의 validation 탐색이며 test·CI·p-value·SOTA·일반적 최적값을 보고하지 않는다.

이번 후속 코드는 `research/conductance_gat/c_learning/`에 분리했다.

- **현재 checkpoint의 의존도:** 기존 2×2 `node_degree` checkpoint에서 원 validation을
  재현한 뒤 C를 그래프·층별 평균으로 교체한다. 전체 층 및 한 층씩 개입을 분리하고
  교체된 C로 node degree를 다시 계산한다. 학습·optimizer step·test 평가가 없는 읽기 전용
  검사이며 보고서만 `results/conductance_gat/c_learning_audits/`에 따로 기록한다.
- **학습 C의 기여:** PPI/arxiv × `learned_c`/`fixed_c` × seed 0의 총 4개 fresh training이다.
  Node-degree 정규화, 같은 backbone 초기화·데이터·non-gate WD 0.0005·학습·selection 정책을
  유지한다. Learned gate WD도 0.0005이며 fixed 조건의 물리 엣지 C는 정확히 1이다.
  기존 2×2의 learned 점수를 가져와 새 fixed 점수와 짝짓지 않고 둘 다 새 run에서 학습한다.
- Fixed C는 우리 모델 내부에서 적응적 가중치 학습을 없앤 대조군이지 외부 GCN/GAT 재구현이
  아니다. Gate를 학습하지 않으므로 활성 parameter 수 차이는 따로 공개해야 한다.
- 새 학습 결과는 `results/conductance_gat/c_learning/<run-id>/comparison.md/csv/json`으로
  분리한다. 기존 benchmark/2×2·Cycle PE·Tree 모델과 결과는 변경하지 않는다.
- 이 구현 당시에는 두 후속 경로의 GPU 결과가 없었다. 이후 C-learning 학습 보고서를
  수령하여 위 2026-09-01 절에 기록했다. 개입 대비와 재학습 차이는 계속 분리한다.

실행법과 결과 경로는 [C-learning README](research/conductance_gat/c_learning/README.md)를 따른다.

이번 후속 구현의 전체 로컬 회귀는 **890 passed / 64 skipped** (30.76 s, exit 0),
Ruff 전체 검사 통과다. 기존 794개에서 96개 검사가 추가됐다. 새 두 조건의 실제 학습 루프→
checkpoint/history→runner/report 연결은 작은 단위 fixture와 모킹한 CUDA API로 확인했다.
평균-C 검사도 원본 보호, 그래프 경계, degree 재계산, validation 불일치 차단을 검증했다.
생략된 64개는 Linux/Bash 62개, 로컬 PyG 미설치 1개, Windows 실제 symlink 권한 1개다.
공식 데이터 다운로드·연구용 CPU 학습·실제 GPU 학습을 로컬에서 실행한 것은 아니다.

### 2026-08-31 2×2 원인 분리 실험의 구현 기록

`research/conductance_gat/ablation/`에 기존 모델과 분리된 새 학습 경로를 추가했다.
기본은 PPI/arxiv × baseline/gate_no_wd/node_degree/both × seed 0, 총 8개의 fresh training이다.
실행은 `bash research/conductance_gat/ablation/reproduce.sh --run-id gat-factorial-seed0-v1`이다.
기존 `benchmark.py` 모델·학습 함수와 기본 reproduce 경로는 변경하지 않았다.

- WD 요인은 gate estimator의 weight/bias만 0.0005→0으로 바꾼다. 다른 파라미터는 기존 Adam
  coupled decay 0.0005를 유지한다. normalization 요인은 global-max 대신 row node-degree
  preconditioning `H-.95 D_C^dagger B^T C B H`를 쓴다. 분모 detach는 없다.
- 변경 연산은 일반적으로 비대칭이며 보존 성질이 다르다. 두 정규화 모두 C의 공통 스케일을
  소거한다. 변경군 rho=.95는 조작 확인값이지 gate 학습 성공의 증거가 아니다.
- 같은 seed와 초기 state hash, cache hash, fixed config를 검증한다. 동일 early-stop 정책을
  적용하되 실제 epoch/optimizer step 수가 다를 수 있으므로 모두 기록한다.
- Train 정답으로 학습하고 validation으로 선택한다. Test 평가·외부 비교 모델·PE/트리 결합은 없다.
  1-seed 차이에 CI, 표준편차, p-value 또는 seed 일반화 주장을 붙이지 않는다.
- 각 조건을 별도 프로세스에서 순차 실행한다. 초깃값/최적/종료 validation 관찰과 실제 매 epoch의
  첫 train batch에서 backward 후 Adam step 전 gradient·parameter·decay를 저장한다.
  관찰용 추가 training forward/backward나 train loader iteration은 없다.
- `scripts/run_conductance_factorial.py`는 CUDA/의존성 사전 검사, fresh path, 실행 중 소스
  hash 변동 검사, 실패 중단·부분 보고서를 제공한다. 완료 시 `comparison.md/csv/json`에
  4조건, 두 요인의 조건부 차이, 상호작용을 데이터셋별로 표시한다.
- 모든 산출물은 `results/conductance_gat/ablations/<run-id>/`로 분리한다. 기존 checkpoint와
  혼동하지 않도록 새 checkpoint model/research_suite은 `conductance_factorial`이다.
  상세 실행·해석은 [실험 README](research/conductance_gat/ablation/README.md)를 따른다.

이 2×2의 실제 GPU 학습 결과는 위 절에 별도 기록했다. 아래의 5e801c3 full-audit 로그는
기존 checkpoint의 읽기 전용 검사이므로 43afd63의 재학습 결과와 합치지 않는다.

구현 후 전체 회귀: **794 passed / 64 skipped** (31.83 s, exit 0), Ruff 통과.
새 실험 관련 114개 검사가 포함된다. CUDA API를 모킹한 4노드 단위 fixture에서 실제 runner →
4조건 train loop → checkpoint/history → SHA 검증 → 비교표의 연결도 확인했다. 이것은 공개
데이터의 CPU 실험이나 GPU 학습이 아니다. 생략은 Linux/Bash 62개, 로컬 PyG 1개, Windows
실제 symlink 권한 1개다. symlink 차단은 별도 mock 검사로 확인했다. 기존 Windows faulthandler
`access violation` 출력은 있었으나 전체 검사는 exit 0으로 끝났다.

### 2026-08-31 단일 seed 기본값과 읽기 전용 확장 검사

사용자 요청에 따라 향후 기본 model seed는 **0 하나**다. 기본 benchmark는 GAT 1개,
Cycle v1 1개, Tree CSL/ZINC 2개의 총 4개 child만 실행한다. 데이터 준비는 기존 4개 child다.
명시적 `--model-seeds 0,1,2,3,4`는 선택적으로 유지하지만 자동 반복하지 않는다. 보조 `all`의
공식 BREC 내부 10-search-seed는 별도 protocol이며 기본 benchmark에는 실행되지 않는다.
과거에 이미 받은 5-seed 집계는 삭제하거나 1-seed 결과로 재분류하지 않았다.

`scripts/diagnose_conductance.py --full-audit`도 선택한 model seed 하나만 사용한다.
기본 대상은 seed 0의 Cora/CiteSeer/PubMed/PPI/arxiv다. 주요 추가 내용은 다음과 같다.

- `scripts/conductance_interventions.py`: validation의 learned/mean/shuffled/off C, 전체 층과
  층별 개입. 변경한 C로 degree/dmax/rho를 재계산하고 metric/loss/logit/flip/전파량을 저장한다.
- `scripts/conductance_gate_audit.py`: 기존 checkpoint의 train-label gradient만 계산한다.
  기본 eval+autograd ON, PPI 첫 1 batch, citation/arxiv full graph의 train mask를 사용한다.
  실제 gate 입력·Linear/SiLU·raw logit·C·raw-logit gradient와 parameter/gradient/decay 항을 기록한다.
- 정량 moment는 전체 원소의 float64 블록 집계이며 큰 activation quantile만 명시적인 유한
  deterministic sample이다. 전체 quantile 또는 전체 PPI train gradient로 과장하지 않는다.
- `report.md`와 `report.json`을 `runs/diagnostics/`의 고유한 새 폴더에 자동 저장한다.
  재학습·optimizer step·다운로드·새 test 평가·원본 덮어쓰기·CPU fallback은 없다.
- hook/forward/mode/gradient/RNG 복구, 기존 checkpoint hash 보호와 finite JSON을 검사한다.
  validation 재현 차이가 크면 확장 검사 전에 중단하고 실패 시 앞선 진단은 별도 보고서에 남긴다.
- C 상수화와 dmax 전파 감쇠를 구분한다. WD 원인은 가설이며 `detach(dmax)`나 정규화 변경,
  외부 모델·내부 모델 재학습을 이 진단에 조용히 추가하지 않았다.
- 집계 artifact schema는 3, metric registry는 5, efficiency registry는 4다. 1-seed std/CI는
  `null`(CSV 빈칸)과 `uncertainty_status=insufficient_samples`로 표시한다. 숫자 0의 불확실성으로
  보고하지 않는다. 명시적 여러 seed의 기존 통계는 유지한다.

실행은 [진단 안내](docs/CONDUCTANCE_DIAGNOSTICS.md)를 따른다. 확장 검사와 단일 seed 변경은
현재 소스 버전에 포함되며 이전 진단 전용 게시 commit에는 없었다. 이후 사용자가 제공한
5e801c3 full-audit 로그에서 seed 0의 다섯 데이터셋 모두 passed를 확인했다. 실측값과 범위는
[실험 상태](docs/EXPERIMENT_STATUS.md)의 확장 검사 절을 따른다.
로컬 전체 회귀는 **680 passed / 63 skipped** (30.54 s, exit 0), 확장 진단 관련 3개 파일은
**89 passed** (4.28 s)다. Windows faulthandler의 기존 `access violation` 경고가 출력됐으나
검사는 위 결과와 exit 0으로 완료됐다. Linux/Bash 62개와 로컬 PyG 미설치 1개는 생략했으며,
이 수치는 실제 서버 GPU 학습·진단 완료의 증거가 아니다.
기존 게시 commit `ebf8cd1`의 모델 소스를 메모리에 로드한 호환성 검사도 **89 passed**
(2.47 s)다. 소스를 checkout하거나 기존 checkpoint를 변경하지 않고 3-인자 Conv API와
새 검사 모듈의 호환성을 확인했다.

### 현재 상태와 읽는 순서

사용자가 제공한 **세 트랙의 기존 5-model-seed 집계**, **Conductance seed 0의 실제 GPU
checkpoint 진단**, **2×2 및 C-learning 재학습 결과**가 있다. 결과 수치, run ID, 평가 범위,
근거 식별자와 미확정 원인은
[EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md)에 정리했다. 이 문서의 과거 감사 기록을
현재의 결과 미수집 상태로 해석하면 안 된다.

- 이전 진단 전용 게시 commit은 `ebf8cd19b80e6cd6c742b132e2bb1dadb97b019c`다.
  해당 commit은 진단 Python/Bash, 테스트, 안내, 트랙 README의 **5개 파일만** 추가·갱신했다.
- 이번 소스 버전에는 기저벡터 Cycle PE v2, 실행 최적화·속도 도구, 단일 seed 기본값,
  확장 진단·2×2·후속 C-learning 코드가 포함된다. `code_summary.md`는 이 버전의 스냅샷이다.
- 제공된 Cycle 결과는 `cycle_set` v1이다. 이를 `cycle_basis_v2`의 학습 결과로 쓰지 않는다.
  기존 benchmark 결과와 당시 진단은 이번 최적화의 가속 실측도 아니다.
- 원격 서버의 전체 checkpoint/manifest를 직접 내려받아 검사한 것은 아니다. 사용자 로그로
  확인한 사실, 로컬 단위 검증, 추가 검증이 필요한 가설을 분리한다.
- 소스 게시·서버 pull과 모델 재학습·GPU 진단 완료는 다르다. 실행 revision과 결과를 별도로 보존한다.

처음 실행할 때는 README, 실제 측정값은 실험 상태 문서, 수학·코드 감사는 이 handoff와
code summary를 읽는다. 세 연구는 계속 독립이며 결합 모델을 추가하지 않았다.

### 2026-08-31 Conductance checkpoint 성능 진단

`scripts/diagnose_conductance.py`와 active-Conda wrapper를 추가했다. 완료된 공식 benchmark의
기록과 checkpoint를 읽어 GPU에서 train/validation 추론만 수행하며 재학습·다운로드·optimizer
update·원래 산출물 덮어쓰기는 없다. 기본 출력은 터미널이고 새 별도 경로의 보고서는 선택 사항이다.
사용법과 해석은 [CONDUCTANCE_DIAGNOSTICS.md](docs/CONDUCTANCE_DIAGNOSTICS.md)에 있다.

- 층별 C 분포, 가중 차수, `rho_i = .95 d_i^C / d_max^C`, 전파 전후 상대 변화량을 확인한다.
  최대 차수는 각 그래프 안에서 계산하며 PPI 그래프를 합친 최대값으로 대체하지 않는다.
- 같은 checkpoint의 train/validation 지표와 PPI 양성 비율을 확인한다. 기존 test 점수는
  저장된 값만 읽으며 test 대상의 신규 평가·ablation은 수행하지 않는다.
- 선택적 `--ablate-graph`는 validation에서만 conductance 전파를 항등 연산으로 우회한다.
  재학습한 MLP baseline이나 원인 확정 실험이 아니며, 해당 checkpoint의 개입 민감도다.
- 전역 최대 가중 차수 스텝은 C 전체의 공통 스케일을 상쇄한다. 작은 `rho`는 이웃 전달 제한의
  증거일 수 있지만 낮은 정확도의 주원인이라는 결론은 성능·학습 곡선과 함께 검토해야 한다.
- 모델·학습 설정은 수정하지 않았다. 사용자가 제공한 실행 로그에서 seed 0의 Cora/PPI/arxiv
  GPU 진단이 `passed`로 끝났고 validation 재계산 오차는 약 `0~5e-8`임을 확인했다.
- PPI/arxiv의 관측한 FP32 eval 입력에서 두 층 C의 변동계수는 0이다. Cora는 첫 층만
  거의 상수이고 두 번째 층은 비상수다. arxiv의 이웃 혼합량 중앙값은 약 0.0433%다.
  상수화의 원인이 weight decay라는 주장은 아직 가설이며 다른 seed까지 일반화하지 않는다.

확장 전 진단 전용 검사는 **42 passed**였다. CUDA-only CLI, 의존성 없는 help, checkpoint/학습 기록
일치, graph-local rho와 pooled 통계, PPI global micro-F1, test 평가 배제, 원본 파일
SHA/mtime 불변과 저장 경로 보호를 검증했다. 로컬 tensor/helper 검증이며 실제 GPU 실행은 아니다.
확장 전 문서 갱신의 전체 회귀는 **619 passed, 63 skipped** (21.84 s, exit 0),
Ruff/diff 검사와 갱신 문서의 로컬 링크 34개 검사 통과다.
생략은 기존 Linux/Bash 전용 62개 및 로컬 PyG 미설치 1개다.
이번 실행에서도 Windows faulthandler의 `access violation` 메시지가 출력됐으나 pytest는
끝까지 실행되어 위 결과와 exit 0을 반환했다. 이 호스트 진단을 숨기거나 실제 Linux/CUDA
성공의 근거로 사용하지 않는다.

### 2026-08-31 실행 최적화 추가

모델의 수식·parameter/state_dict·공식 split·loss·checkpoint 선택은 유지하고 실행 경로를
최적화했다. 변경 전후의 부동소수점 합산 순서는 일부 다르므로 bitwise 학습 재현을 주장하지 않는다.

- Conductance: CPU graph-count metadata 전달로 층별 scalar GPU sync 제거, split indices
  사전 준비, loss와 PPI global micro-F1 count의 device 누적. nonfinite fail-closed 검사는 유지.
- Cycle PE 공통: categorical field stack 제거, 고정 크기 pooling count, FP32 benchmark의
  message topology를 forward당 1회 준비해 layer stack에서 재사용.
- v2: `basis_execution=batched` 기본. 모든 signed U_c entry를 `(graph,column)`별로 묶어
  two-pass context/edge encoding 수행. reference 경로와 checkpoint 호환. pair_budget은
  MLP 호출당 pair 수만 제한하며 전체 basis/autograd 메모리의 상한을 뜻하지 않는다.
- `src/chartgat/execution.py`: 선택적 `--compile`, Inductor/dynamic shapes, in-place
  tensor MLP 블록의 bound forward를 컴파일해 checkpoint 키 유지. 가변 기저 Python 루프 전체를
  compile했을 때 반복 재컴파일이 재현되어 해당 scheduling은 eager로 유지한다.
  기본 OFF/AMP 기본 OFF, compiler 오류를 잡아 성공으로 숨기지 않는다.
- 정식 runner는 train 지표와 `epoch_seconds`를 기록하고 로그를 즉시 flush한다.
  `scripts/benchmark_speed.sh`는 공식 train 입력만 사용하는 CUDA forward/backward 비교이며
  warmup·steady-state·동등성 검사·peak memory를 `runs/performance/`에 분리 기록한다.
  optimizer update 및 전체 학습 속도/정확도 실험과 구분한다.
- GPU 실측·전체 학습 완료·가속 배수는 확인하지 않았다. 사용법과 측정 경계는
  [PERFORMANCE.md](docs/PERFORMANCE.md)에 있다. 구형 Singularity에서 opt-in compiler
  호환성을 보장하지 않으며 환경/드라이버/공용 라이브러리를 자동 변경하지 않는다.

이후의 과거 감사 수치와 한계는 해당 시점의 기록이며, 현재 최적화의 GPU 검증 결과가 아니다.

진단 도구 추가 전, 최적화 구현 시점의 전체 회귀는 **577 passed, 63 skipped** (19.96 s, exit 0),
Ruff 및 diff 검사 통과다. 생략은 Linux/Bash 전용 62개와 로컬 PyG 미설치 1개다.
추가된 Dynamo 검사는 CPU tensor/counting backend에서 10개 ragged shape와 실제 블록
컴파일 경로 실행, 출력·입력/전체 parameter gradient 및 checkpoint 호환성을 확인한다.
이는 GPU Inductor 코드 생성·구형 Singularity compiler 호환성·실제 GPU 가속 검증이 아니다.

### 2026-08-31 Cycle PE v2: 좌영공간 기저벡터 자체를 입력

사용자는 기존 기본 Cycle PE가 기저벡터 자체를 받는다고 이해했으나, 실제 v1 기본 경로는
기저를 계산한 뒤 여섯 수작업 통계로 요약한 `cycle_set`이었다. 두 표현을 동일시하지 않는다.
요청에 따라 `research/cycle_pe/v2/`에 **기저벡터 입력 버전**을 독립 구현했다. 연구 트랙은
여전히 세 개이고, Cycle PE에 별도 버전을 추가한 것이며 다른 트랙과 결합하지 않았다.

- `basis.py`: canonical `u<v` incidence `B[m,n]`에서 float64 full SVD로 전체 좌영공간
  `U_c[m,beta]`를 계산한다. `beta=m-n+c`, `B.T @ U_c≈0`, `U_c.T @ U_c≈I`를 검사하고
  float32로 저장한다. 모든 열을 보존하며 원핫·6개 통계·projector·상위 k개로 대체하지 않는다.
- `data.py`: 기존 공식 ZINC-12K/Peptides-struct 원본·split adapter만 재사용한다. v1의
  통계 전처리는 호출하지 않는다. 기저마다 edge-row를 일치시키고 가변 `[m_i,beta_i]` 행렬을
  배치 내에서도 별도로 보존한다. forest/disconnected/isolates/edgeless 입력을 처리한다.
  namespace는 `cycle_pe_v2_benchmark`; 구현/NumPy/원본 checksum과 수학 조건을 검사한다.
- `model.py`: 각 기저벡터의 signed coefficient와 bond embedding을 먼저 학습층에 넣고,
  열별 전체 엣지 문맥을 학습한 뒤 엣지 PE로 전달한다. `f(u)+f(-u)` 비선형 대칭화와 모든 열의
  집계로 부호/열 순서 불변성을 구현한다. 입력 기저는 topology-only이나 encoder는 bond에도
  조건화된다. 가변 rank를 자르거나 train 최대 폭을 test에 강제하지 않는다.
- 임의의 `U_c Q` 직교 회전이나 엣지 재정향 불변성은 없다. SVD를 재계산하는 graph relabeling은
  기저 회전을 일으킬 수 있으므로 graph-isomorphism invariance를 주장하지 않는다. 고정 폭의
  학습 PE는 무손실 codec이 아니다. Dense SVD 전처리와 모든 열을 통한 학습의 비용도 남는다.
- `benchmark.py`: CUDA-only, float32 기본, 기존 message backbone과 공식 입력/target 유지,
  validation checkpoint 선택 후 test 1회. 모델 이름은 `cycle_basis_v2`, track은 `cycle_pe`,
  version은 `v2`다. 기본 데이터·batch·epochs·optimizer는 v1과 같고 파라미터 수는 따로 기록한다.
- 전용 `prepare_data.sh`/`reproduce.sh`는 root runner의 `--cycle-pe-version v2`를 사용한다.
  v2는 `cycle_pe` 단독 benchmark만 허용한다. 기본 전체 실행과 v1 경로는 그대로다.
  결과는 `research/cycle_pe/v2/results/paper/`, custom root는 `cycle_pe_v2/` 아래에 분리한다.
- 집계 schema는 paper 5 / efficiency 4이며 `cycle_basis_v2.test` 및 효율 지표만 명시적으로
  추가했다. v1·v2는 별도 model metric path로 집계하며 외부 논문 수치·validation과 합치지 않는다.
- `v2/datasets.yaml`은 v2 run에 보존하는 데이터/표현 계약이다. 기존 일반 `check_datasets.py`의
  v1 registry 검증이 v2 기저를 검증했다고 해석하지 않는다. v2 로더 자체가 수학·내용을 검사한다.

설치 완료 후 실행 명령은 [v2 README](research/cycle_pe/v2/README.md)에 있다. 이번 작업에서
공식 데이터 다운로드나 CPU/GPU 연구 학습은 실행하지 않았다. 작은 개발 fixture의 수학·배치·미분
검사는 실제 공개 데이터의 학습 완료나 성능/novelty 입증과 구분한다.

최적화·진단 추가 전, v2 구현 시점의 전체 회귀는 **482 passed, 63 skipped**,
Ruff 및 diff 검사 통과다. 생략된 63개는
기존 Linux/Bash 전용 62개와 로컬 PyG 미설치 검사 1개다. 새 검사는 전체 기저의 nullity/
orthonormality/큰 rank 허용오차, cache 무결성·원본 정렬, 가변 rank 배치, 실제 signed 계수의
학습층 입력, 모든 열의 gradient, sign/order symmetry, chunk 결과 일치, forest/빈 엣지,
CUDA 학습 guard, val→test 순서, v2 CLI/manifest/결과 분리와 집계 배제를 포함한다.
이 검증은 실제 Linux shell 또는 GPU 실행 성공을 대신하지 않는다.

### 2026-08-31 Ubuntu 18.04 Singularity의 glibc 호환 경로

사용자가 제공한 실제 실행 환경은 Ubuntu 18.04.5 / glibc 2.27이다. Torch import는
`GLIBC_2.28 not found`로 실패했다. `/tools/scripts/ssu_a6gpu`는 stripped ELF 실행기이고
Conda는 `/tools/anaconda3`의 공용 설치본이다. 이미지 선택 옵션은 확인되지 않았으므로
실행기를 수정하거나 임의 플래그를 추측하지 않았다. 이전의 CUDA 12.2→cu118 교정만으로는
이 libc 충돌이 해결되지 않는다. 아래 기록의 기본 cu118은 여전히 Torch 2.7.1이다.

- 명시적 `legacy-cu118` profile을 추가했다. Python 3.11 / Torch 2.6.0+cu118 / PyG 2.7.0이며
  별도의 `requirements-legacy-cu118-lock.txt`와 `constraints-legacy-cu118.txt`를 사용한다.
  기본 `auto` 및 기존 cu118/cu126/cu130/cu132 선택은 바꾸지 않았다.
- 사용자는 새 `new-gat-legacy` Conda 환경에서 `setup_gpu.sh --profile legacy-cu118`을 실행한다.
  설치 전 보호 검사는 `new-gat` 환경 자체와 다른 Torch 버전이 이미 설치된 환경을 거부한다.
  기존 환경/진행 중인 학습, 공용 Conda base, 컨테이너 이미지, glibc, 드라이버는 변경하지 않는다.
- Torch 2.6 cu118의 Manylinux2014 배포는 glibc 2.17부터지만, NumPy/SciPy 등의 직접 pin을
  포함한 조합은 glibc 2.27 이상으로 제한한다. Python 3.11/x86_64용 직접 의존성 wheel 제공
  여부를 공식 PyPI/PyTorch 메타데이터로 확인했다. 전이 의존성을 전부 사전 잠근 것은 아니다.
- `check_dependencies.py`는 runtime import 전에 호스트 ABI를 검사한다. 호스트 오류는 별도
  종료 코드 `3`이며 `paper.sh`와 `run_paper.py`가 이를 보존한다. 데이터 준비가 동일한
  비호환 패키지 설치를 자동 반복하지 않는다. 일반 의존성 누락만 기존 1회 보완 경로를 쓴다.
  보완 시 metadata로 인식한 기존 profile을 유지하고, 미등록/CPU/custom Torch는 자동 교체하지 않는다.
- 동일한 `cu118` runtime이라도 정확한 Torch 버전으로 두 profile을 구분한다. 설치 verifier와
  실행 manifest에 `profile_id`를 추가했다. CPU/custom Torch, 다른 pin과 runtime은 계속 거부한다.
  legacy 설치 기록은 활성 Conda prefix의 `.new-gat-environment/`에 별도로 저장한다.
- 모델, 데이터셋, split, 학습·평가 설정은 변경하지 않았다. 새 profile의 결과는 기존 profile의
  seed 반복으로 합치지 않는다. 기존 run을 자동 재개하거나 덮어쓰지 않는다.
- 구버전 Torch에는 공개된 체크포인트 로딩 취약점이 있다. Torch 2.6뿐 아니라 2.7도 해당하며
  `weights_only=True`를 보안 보장으로 취급하지 않는다. 공식 출처가 확인된 데이터와 직접 만든
  체크포인트만 사용한다. 별도 Conda 환경은 보안 sandbox가 아니다. 최신 보안 패치가 필요하면
  더 새 컨테이너가 필요하다. [환경 안내](docs/ENVIRONMENT.md)에 설치 절차와 제약을 정리했다.

근거: [PyTorch Linux wheel 플랫폼 공지](https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581),
[공식 cu118 wheel 목록](https://download.pytorch.org/whl/cu118/torch/),
[PyG 2.7 지원 목록](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.7.0),
[체크포인트 로딩 보안 공지](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).

전체 pytest **387 passed, 63 skipped**, Ruff 및 diff 검사 통과. 생략은 Linux/Bash 동적
검사 62개와 로컬 PyG 미설치에 따른 batching 검사 1개다. profile 선택/정확한 pin/CUDA
runtime, glibc 경계, 기존 환경 보호, 의존성 보완 시 profile 유지, manifest, README 실행
계약을 검증했다. `code_summary.md`도 현재 105개 소스/설정 파일에서 다시 생성했다.
이 호스트에서 실제 Ubuntu 18.04 설치·CUDA 학습은 실행하지 않았다. 아래 이전 날짜의
pytest 수치는 해당 시점의 이력이며 최신 회귀 결과와 구분한다.

### 2026-08-30 CUDA 12.2 서버의 설치 차단 교정

사용자 서버는 RTX A6000 4개이며 `nvidia-smi`가 CUDA 12.2를 표시했다. 이전 setup은
CUDA 12.6 이상을 요구하여 **첫 pip 설치 전에 종료**했다. 따라서 이어진 학습의 전체
의존성 누락은 설치가 성공하지 않은 데 따른 결과이며, Conda를 다시 만드는 문제는 아니다.

- `scripts/gpu_profiles.py`는 stdlib만으로 고정 조합을 선택한다. 기본 `auto`는 CUDA 표시값
  11.8 이상/12.6 미만에서 **Torch 2.7.1+cu118 / PyG 2.7.0**, 12.6 이상에서 기존
  **Torch 2.13.0+cu126 / PyG 2.8.0.post1**을 선택한다. 다른 직접 의존성 pin은 같다.
- `requirements-cu118-lock.txt`와 `constraints-cu118.txt`를 함께 추가했다. PyG 2.8은
  Torch 2.7 지원 대상이 아니므로 Torch만 낮추지 않았다. 근거는
  [PyTorch 공식 wheel](https://download.pytorch.org/whl/cu118/torch/) 및
  [PyG 2.7 지원 조합](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.7.0)이다.
- 명시적 `CUDA_WHEEL_TAG`는 임의로 대체하지 않는다. 설치 중 드라이버/시스템 CUDA를
  변경하지 않으며 CPU fallback도 없다. glibc >=2.28 및 cu118의 x86_64/Python 3.11–3.13
  wheel 조건을 첫 pip 전에 검사한다. 재현용 Python은 기존 `environment.yml`의 3.11이다.
- NVIDIA minor-version compatibility 때문에 cu126이 이 드라이버에서 어떤 경우에도
  불가능하다고 판정한 것은 아니다. 설치 정책은 그 기능·PTX 제한에 의존하지 않는 보수적 선택이다.
- 데이터 준비·학습 검사는 GPU를 조회하지 않고 **설치된 wheel tag의 동일한 lock**을 사용한다.
  cu118 설치를 다시 2.13 요구로 거부하거나 재설치하지 않는다. 명시적 선택은 검사에도 적용한다.
  설치와 검사는 `torch==버전+CUDA tag`까지 요구하므로 CPU/custom build를 인정하지 않는다.
- 설치 후 NumPy↔Torch 빈 배열 변환을 검사해 import만 통과한 ABI 문제도 보고한다.
  데이터나 학습 결과를 만들지 않는다. 실제 NumPy 2.4.6/구버전 Torch 조합의 서버 검증은
  이 설치 검사가 담당하며, 이 작업 공간에서 그 조합을 설치해 확인했다고 주장하지 않는다.
- 실행 manifest의 `research_environment`에 CUDA tag, lock SHA-256, 직접 의존성 버전을
  추가 기록한다. 기존 전체 패키지 snapshot도 유지한다. 서로 다른 조합을 동일 환경의 seed
  반복으로 합치지 않는다. 모델/데이터/평가 protocol 변경은 없다.

전체 pytest **314 passed, 32 skipped**, Ruff 통과. 생략은 Linux/Bash 동적 검사 31개와
로컬 PyG 미설치에 따른 batching 검사 1개다. CUDA 12.2→cu118 선택, 고정 pin/runtime 검사,
custom/CPU wheel 거부, Python/glibc 조건, manifest 기록을 검증했다. 새 Linux 설치 스텁
8건은 추가했지만 이 Windows 호스트에서는 실행하지 못했다. 실제 서버 설치·CUDA 학습은
수행하지 않았으며 CPU 학습 결과를 만들지 않았다.

### 2026-08-30 서버의 NumPy 누락 오류에 대한 설치 경로 교정

서버의 활성 `new-gat` Python에서 데이터 준비 중 `ModuleNotFoundError: numpy`가 보고됐다.
`environment.yml`은 Python/pip만 만드는 bootstrap이며, 기존 `paper.sh`는 Conda interpreter만
확인한 뒤 의존성 검사 없이 runner를 시작했다. `chartgat.cache` import가 package initializer의
eager algebra import를 통해 NumPy부터 요구하여 실제 준비 전 traceback으로 종료됐다.
서버의 이전 설치 로그가 없으므로 setup을 건너뛰었는지, 설치가 중간에 실패했는지는 단정하지 않는다.

- `scripts/check_dependencies.py`는 표준 라이브러리만으로 시작해 전체 lock의 누락/버전 불일치를
  한 번에 보고하고, runtime import와 CUDA wheel 종류도 검사한다. GPU allocation 자체는 요구하지 않는다.
- `paper.sh --prepare-only`는 검증된 Conda Python으로 먼저 검사하고, 불완전하면 전체 setup을
  한 번 실행한 뒤 재검사한다. 어느 단계든 실패하면 데이터 준비로 넘어가지 않는다.
- `setup_gpu.sh`의 `SKIP_DEPS` 분기를 제거했다. 기본 설치와 자동 보완 모두 전체 고정 의존성을
  설치한다. 설치 자체의 Linux/NVIDIA 드라이버·GPU 할당·네트워크 요건은 그대로다.
- 학습 진입점은 설치 상태를 읽기 전용으로 확인하며 누락되면 run 폴더 생성 전에 설치 명령을 안내한다.
  `chartgat`의 algebra export는 lazy loading으로 바꿔 cache/도움말이 NumPy에 의존하지 않게 했다.
  `--help`/`--dry-run`은 설치·다운로드 없이 실행되며 자동 보완 대상이 아니다.
- 패키지를 볼 수 없는 `python -S` subprocess로 cache import, CLI 도움말/dry-run, 누락 오류 안내를
  검증했다. 기존 public algebra API도 유지된다. 전체 pytest **256 passed, 24 skipped**, Ruff 통과.
  생략은 Linux/Bash 동적 검사 23개와 PyG batching 검사 1개다. 기존 Windows faulthandler 진단이
  출력됐지만 pytest 프로세스는 종료 코드 0이었다. 실제 서버 설치와 CUDA 학습은 여기서 실행하지 않았다.

### 2026-08-30 최종 실행 범위: 트랙별 데이터에서 우리 모델만 실행

기본 실행은 `--suite benchmark`로 변경했다. Conductance GAT는 GAT/GATv2 논문에
나오는 Cora/CiteSeer/PubMed/PPI/ogbn-arxiv에서 우리 conductance 모델만 학습한다.
Cycle PE는 SignNet/PEARL 공통 ZINC-12K 및 PEARL의 Peptides-struct에서 우리 cycle-set
PE 모델만 학습한다. 외부 비교 모델의 구현·실행은 제거했다. 트리 증강의 CSL/ZINC
fixed-BFS vs multi-chart는 같은 우리 모델의 증강 ablation이므로 독립적으로 유지한다.

기본 실행에서 S1–S4/CycleCount 생성과 BREC는 제외했다. 기존 코드는 명시적인 `core`/`all`
보조 suite에 남아 있다. 아래의 이전 감사 내용에서 '현재 paper core'는 이 교정 전 경로를
가리키며, 새 benchmark의 실행 목록으로 읽으면 안 된다. 이 보조 suite에서도 외부
MLP/GCN/GAT/GINE 모델은 실행하지 않는다. 자체 연산의 ablation/해석적 진단은 별개다.

외부 비교 점수는 **논문 표에서 출처를 밝혀 인용**한다. 저장소가 직접 재현한 값이라고
표기하지 않으며 우리 seed별 결과와 paired 통계로 합치지 않는다. 데이터 이름뿐 아니라
버전/split/지표/입력/파라미터 예산/학습 조건도 확인하고 다른 조건을 표 주석에 명시한다.
전체 공개 데이터 다운로드/GPU 학습은 이 Windows 작업 공간에서 수행하지 않았다.
Alchemy는 upstream index의 중복·split 겹침 때문에 기본 데이터에 추가하지 않았다.

#### 새 기본 실행의 구현 경계

| 트랙 | 새 실행 모듈 | 비교·데이터 계약 |
|---|---|---|
| Conductance GAT | `research/conductance_gat/benchmark.py` | Cora/CiteSeer/PubMed public masks, PPI 20/2/2, ogbn-arxiv official split; 우리 conductance |
| Cycle PE | `research/cycle_pe/benchmark.py` | ZINC 10k/1k/1k, Peptides-struct 10873/2331/2331; 우리 cycle-set PE |
| Tree augmentation | 기존 `research/tree_augmentation/paper.py`의 `csl`·`zinc` | fixed-BFS vs multi-chart; CSL은 한 개의 stratified 90/30/30 split, ZINC는 official split |

- GAT의 `benchmark_data.py`는 원본/분할 checksum과 전처리 규칙을 저장한다.
  우리 모델은 기존 positive scalar estimator와 sparse `H - eta B^T C B H`를 사용한다.
  표준 GAT/GATv2/GCN/SAGE 생성 분기와 `--baselines` 선택 옵션은 없다.
  ogbn-arxiv 학습은 full-batch로, GATv2 논문의 GraphSAINT 설정과 다르다.
- PE의 `benchmark_data.py`는 공식 atom/bond feature와 target을 그대로 보존한다.
  `benchmark_models.py`는 기존 Cycle PE의 edge-aware message layer와 task head를 사용한다.
  BFS fundamental basis, 여섯 set statistic, GELU encoder 및 edge-PE 주입을 재사용한다.
  별도 GINE 경쟁 모델이나 LapPE/RWSE/SignNet/PEARL 구현·전처리는 없다.
  PE를 downstream 예측으로 읽는 neural layer는 우리 모델의 구성요소이며 외부 비교 모델이 아니다.
- Cycle-set은 cycle-column sign/order에는 불변이지만 chart 교체에는 불변이 아니다.
  원 논문과 같은 데이터셋을 쓰는 것과 논문 모델의 수치를 재현하는 것은 구분한다.
- 학습은 CUDA 전용이다. 기본 float32, PPI batch 2, 분자/tree batch 32다. 당시 model seeds는
  0–4였으며 이후 사용자 요청으로 현재 기본값은 0 하나로 변경했다.
  GAT/PE는 validation으로 checkpoint를 선택한 뒤 test를 한 번 평가한다. GAT는 accuracy,
  PPI는 전체 node-label micro-F1, PE는 MAE를 사용한다. 시간/메모리/파라미터도 따로 저장한다.
- Root `scripts/run_paper.py` 기본값과 다섯 Bash wrapper를 새 `benchmark`로 연결했다.
  준비는 GAT/PE/CSL/ZINC 네 child만 한 번씩 수행한다. 기본 재현은 preflight 이후
  당시 GAT 5개 + PE 5개 + tree 10개 child였고 현재는 1+1+2개다. 세 트랙 결과 폴더는 분리한다.
- Benchmark schema v2는 `datasets.<dataset>.models.conductance` 또는 `.models.cycle_set`를
  사용한다. `scripts/aggregate_paper.py`는 이 경로의 test만 성능으로 집계하고
  validation/history/외부 모델 점수/인용 수치를 제외한다. Paired 비교는 우리 모델의
  내부 ablation에만 적용하며 benchmark의 단일 모델로 외부 모델과의 통계를 만들지 않는다.
- 보조 Cycle PE 기본 variant는 raw/set/projector이고 No-PE는 명시적 옵션의 내부 ablation이다.

#### 검증 범위

- 외부 모델 제거 후 root runner/집계/registry 및 세 트랙 관련 검사: **174 passed, 1 skipped**.
  생략된 한 검사는 로컬 PyG 미설치로 실행하지 못한 데이터 batching 검사다.
- 수정 Python의 Ruff 검사 통과. 당시 기본 재현의 20개 학습 child와 준비의 4개 child 명령은
  실제 각 트랙의 CLI parser로 검증했다. 외부 모델 선택 옵션은 거부되고, 논문 인용 점수는
  우리 결과 집계와 paired 통계에 포함되지 않는 것을 테스트했다.
- Windows 검사 중 `Windows fatal exception: access violation` 진단 출력이 있었으나
  검사 프로세스는 계속 실행되어 위 pytest 결과와 종료 코드 0을 반환했다. 이 호스트 진단을
  Linux/CUDA 검증 통과로 해석하지 않는다.
- 이 작업에서는 실제 데이터 다운로드, PyG/OGB 실제 cache 로딩, Linux Bash 실행 또는
  GPU benchmark 학습을 수행하지 않았다. CPU 학습 결과를 연구 결과로 생성하지 않았다.

### 기존 연구 및 감사 기록

1. 활성 연구는 세 개이며 서로 독립이다.
   - `research/conductance_gat`: positive scalar edge conductance를 학습하는 sparse incidence
     operator.
   - `research/cycle_pe`: topology에서 미리 계산한 static cycle-space edge PE.
   - `research/tree_augmentation`: 같은 graph의 full-cycle-rank fundamental basis를 여러
     spanning-tree chart로 바꾸는 augmentation.
2. 위 세 연구를 결합한 모델은 아직 없다. `research/combined_later`는 격리된 과거
   prototype이며 paper runner가 import하거나 실행하지 않는다.
3. 구현과 가설 입증은 다르다. 로컬 코드·CLI·fixture·artifact 회귀 검사와 사용자가 제공한
   benchmark 5-seed 집계는 별도 근거다. 보조 `core/all` 전체 결과, v2 GPU 결과,
   최적화의 GPU 가속 실측을 모두 확보한 상태는 아니다.
4. 실험 CLI의 `--tiny`, 공개 데이터 대체용 가짜 데이터 생성, legacy smoke 실행기는 제거했다.
   테스트 내부의 작은 입력과 실제 연구용 S1–S4/CycleCount 합성 벤치마크는 별개다.
5. dataset registry의 `implemented/code_ready`는 adapter와 runner가 있다는 뜻이다. 현재 로컬에
   official public cache가 있다는 뜻은 아니다.
6. novelty는 코드만으로 확정할 수 없다. 특히 conductance operator는 learned symmetric
   diffusion 계열, projector PE는 기존 cycle-space projector 계열, tree chart augmentation은
   gauge/frame augmentation 계열과 문헌 대조가 필요하다.
7. 2026-08-29 두 차례 외부 교차감사의 코드 gate를 반영했다. Strict cache, BREC official
   mode, Wilson exposure, tree orientation gauge, 독립 seed 축에 이어 두 번째 감사의 exact GPU
   constraints, suite-aware preflight, closed paper-metric registry도 구현했다. 반면 “통합 모델을
   active로 승격”하라는 항목은 결함 수정으로 받아들이지 않았다. 사용자가 요구한 현재 범위는
   세 독립 연구이고 결합은 이후 단계이기 때문이다.
8. 2026-08-30 GPU 실행 환경을 전용 Conda 생성·활성화 방식으로 변경했다. 두 Bash
   entrypoint는 활성 non-base Conda Python을 검증한 뒤 설치·실행하며 venv를 생성하지 않는다.
   전체 연구 모델·데이터·평가 protocol은 유지하고, 축소 데이터 실행 경로는 제거했다.
9. 공개 재현 안내는 `environment.yml` → `setup_gpu.sh` → 데이터 준비 → 트랙별 실험으로 정리했다.
   설치 기본값은 위의 두 고정 조합을 고르는 `auto`이며 전체 pytest는 `RUN_TESTS=1`일 때만 실행한다.

### 코드 스냅샷

- 파일: `code_summary.md`
- 포함 파일: 162개
- 크기: 1,574,571 bytes, 40,118 lines (`str.splitlines()` 기준)
- SHA-256: `C38A75284506CE7CDDB32153C9A6745F24C0AC93D591C36E60AE3C2352CFF700`
- 포함: 모든 Python source/test, TOML/YAML, Bash/PowerShell script, requirements, `.gitignore`, `.gitattributes`
- 제외: `.venv*`, data/cache, run artifact, `egg-info`, README류 설명 문서
- 범위: 이 버전의 전체 source/test/config/script. 생성기는 작업본 변경도 포함하므로 게시 전
  `--check`로 같은 revision의 소스와 맞는지 확인한다.
- 형식: `# 파일경로` 다음에 해당 파일의 코드 전체를 넣으며 코드 내용을 요약·생략하지 않는다.

재생성과 원본 일치 검사는 저장소 루트에서 실행한다.

```bash
python scripts/generate_code_summary.py
python scripts/generate_code_summary.py --check
```

이 디렉터리는 Git repository로 초기화되어 있으며 원격은
`https://github.com/costunder/new-gat.git`이다. 실행 환경에서는 먼저 `git rev-parse HEAD`를
확인한다. Paper runner도 source revision과 dirty 상태를 manifest에 기록한다.

## 1. 공통 수학 관례와 원래 문제의식

### 1.1 Incidence convention

연결된 무방향 graph에 임의의 edge orientation을 고정한다. 코드의 incidence matrix는
edge-by-node convention이다.

\[
B\in\mathbb{R}^{m\times n},\qquad
B_{e,u}=-1,\quad B_{e,v}=+1
\]

edge `e=(u→v)`에 대해 node signal `p`의 edge gradient는

\[
(Bp)_e=p_v-p_u
\]

이고 edge flow `q`의 node aggregation/divergence는 `B^Tq`다. Unweighted node Laplacian은

\[
L=B^TB=D-A.
\]

연결 graph에서

\[
\operatorname{rank}(B)=n-1,\qquad
\dim\ker(B^T)=m-n+1=\beta.
\]

따라서 사용자가 말한 “B의 좌영공간”은 정확히 `ker(B^T)`이며 edge circulation/cycle
space다. `F_T∈R^{m×β}`는 spanning tree `T`에 대한 full-column-rank fundamental cycle
basis이고 코드가 `B^TF_T=0`을 검사한다.

### 1.2 비가역성과 pseudoinverse에 대한 정확한 해석

`B^Tq=b`가 consistent라면 일반해는

\[
q=q_{\min}+F_Ta,
\]

이다. `q_min=(B^T)^+b`는 최소노름 particular solution일 뿐 원래 flow의 임의 cycle
component `F_Ta`를 복구하지 못한다. 이것은 “pseudoinverse가 완전한 원래 해를 복구한다”는
뜻이 아니다.

다만 rank deficient라는 사실만으로 방정식이 모순인 것은 아니다. `b`가 `im(B^T)`에 있으면
해가 여러 개이고, 밖에 있으면 least-squares residual이 남는다. `Lp=b`도 connected graph에서
상수 gauge 때문에 singular하지만 `sum(b)=0`이면 consistent하고 `L^+b`가 gauge-fixed
minimum-norm node solution을 준다.

즉 핵심 정보 손실은 다음과 같다.

\[
B^T(q+F_Ta)=B^Tq.
\]

node aggregation만 보면 cycle flow를 구별할 수 없다. 하지만 이것만으로 sample의 cycle
coefficient를 static topology PE에서 자동 복구할 수 있는 것은 아니다. 추가 관측, prior 또는
supervision이 필요하다.

여기서 잃는 것은 **주어진 edge flow `q`의 circulation component**이지 graph topology 자체가
아니다. Simple unweighted graph에서는 full matrix `L=D-A`의 off-diagonal entry가 adjacency를
결정하므로 cycle topology는 `L`에서 정보이론적으로 사라지지 않는다. 따라서 Static Cycle PE는
`L`이 지운 topology를 복원하는 codec이 아니라, cycle 구조를 모델이 쓰기 쉬운 edge
representation으로 제공하는 inductive bias다.

### 1.3 세 연구를 분리해야 하는 이유

| 연구 | 학습/입력 대상 | 검증하려는 것 | 검증하지 않는 것 |
|---|---|---|---|
| Conductance | `c_e>0`인 diagonal `Cθ` | node→edge→node transport law | cycle coefficient 복구, cycle PE |
| Static Cycle PE | topology-only `F_T`/cycle statistics/projector | cycle 구조를 edge representation에 주는 효용 | sample flow, learned C |
| Tree augmentation | 동일 graph의 여러 full-β `F_T` chart | chart shift에 대한 robustness | adaptive MST, sparsification, learned C |

`q=CθBp+F_Ta`를 하나의 state codec처럼 학습하는 연구는 현재 active pipeline에 없다. 결합은
각 독립 가설이 먼저 지지된 뒤 별도 연구로 해야 한다.

## 2. 저장소 지도

### Root와 공통 계층

- `README.md`: Linux NVIDIA GPU 환경의 설치부터 전체 재현까지의 실행 명령.
- `DATASETS.md`: 사람이 읽는 데이터·split·metric 계약.
- `docs/EXPERIMENT_STATUS.md`: 기존 5-seed 결과, 실제 seed 0 GPU 진단·2×2·C-learning과 미확정 원인.
- `docs/CONDUCTANCE_FACTORIAL_FINDINGS.md`: 2×2의 정확한 점수·대비·층별 진단·근거와 다음 C-learning의 해석 경계.
- `docs/CONDUCTANCE_C_LEARNING_FINDINGS.md`: 새 learned/fixed 결과·활성/동결 파라미터·진단과 같은 run의 후속 검사.
- `docs/CONDUCTANCE_DIAGNOSTICS.md`: 기존 checkpoint의 읽기 전용 GPU 진단 실행·해석.
- `docs/PERFORMANCE.md`: 실행 최적화 옵션과 미측정 GPU 가속의 검증 경계.
- `pyproject.toml`: Python 3.11+, core/dev/paper dependency와 pytest/Ruff 설정.
- `requirements-lock.txt`, `requirements-cu118-lock.txt`, `requirements-legacy-cu118-lock.txt`,
  `constraints-*.txt`: Python 3.11 호환 exact top-level 연구 stack과 profile별 official Torch 계약.
- `scripts/gpu_profiles.py`: 보수적 드라이버/호스트 조건, 정확한 설치 버전으로 고정 조합 선택.
- `scripts/check_dependencies.py`: 연구 import 전 profile/pin/ABI 검사, 별도 호스트 오류 종료 코드.
- `requirements-paper.txt`: portable paper dependency가 같은 lock을 사용하게 하는 진입점.
- `scripts/setup_gpu.sh`, `scripts/verify_gpu_lock.py`: 활성 프로젝트 전용 Conda 환경에 exact
  package 설치, ABI/CUDA runtime 검증과 transitive freeze snapshot.
- `scripts/conda_env.sh`, `scripts/verify_conda_env.py`: Linux Bash 진입점이 공유하는
  Conda/Python 검사. 비활성 환경, base 환경과 중첩된 별도 Python 환경을 거부한다.
- `scripts/gpu_preflight.py`: CUDA device, 여유 메모리, dependency import 검사. 데이터 생성·학습 없음.
- `scripts/paper.sh`: 활성 Conda 환경의 Python으로 master runner 실행.
- `scripts/run_paper.py`: 세 독립 트랙을 model-seed별 subprocess로 dispatch하고 중앙 manifest 작성.
- `scripts/prepare_data.sh`: 전체 데이터 준비 명령을 담은 실행 파일.
- `scripts/reproduce.sh`, `research/<track>/reproduce.sh`: 전체 또는 트랙별 정식 실험 실행 파일.
- `scripts/aggregate_paper.py`: 폐쇄형 paper metric/efficiency registry와 seed-aligned 통계.
- `scripts/conductance_interventions.py`, `scripts/conductance_gate_audit.py`: 단일 checkpoint의
  validation C 개입 및 train-label 국소 gradient 검사. production 학습 경로에는 import하지 않는다.
- `scripts/run_conductance_factorial.py`: 원래 2×2의 CUDA subprocess·source 무결성·별도 결과 runner.
- `scripts/run_conductance_c_learning.py`: node-degree learned/fixed C의 별도 4-training runner.
- `scripts/generate_code_summary.py`: 외부 교차검증용 exact source snapshot 생성/검사.
- `scripts/check_datasets.py`: 세 `datasets.yaml`의 code/cache readiness 검사.
- `src/chartgat/algebra.py`: incidence, fundamental cycle basis, chart transition 등 공통 저수준 수학.
- `src/chartgat/cache.py`: same-directory temporary, fsync, validation, atomic replace cache writer.
- `src/chartgat/graphs.py`: connected graph와 tree helper.
- `src/chartgat/seeds.py`: data/split/chart/model seed 축과 legacy fallback 해석.
- `tests/`: root runner, GPU preflight, registry, algebra, cross-track import boundary 테스트.

### 활성 연구 폴더

- `research/conductance_gat/`
  - `benchmark_data.py`, `benchmark.py`: 기본 5개 공개 데이터의 cache·우리 모델 학습/평가.
  - `ablation/`: gate WD × normalization 2×2. `train.py`의 명시적인 training definition으로
    후속 C-learning에 같은 학습 루프를 제공하되 기본 4조건의 모델·optimizer 동작은 유지한다.
  - `c_learning/model.py`, `protocol.py`, `train.py`: node-degree learned/fixed C만 바꾸는 별도 suite.
    Fixed 조건은 RNG/state 일치를 위해 gate scaffold를 동결 보존하지만 실행·optimizer에서는 제외한다.
  - `c_learning/report.py`: 같은 새 run의 initialization/cache/config/파일 hash를 확인한 learned−fixed 비교.
  - `c_learning/intervene.py`, `audit.sh`: suite를 구분해 기존 factorial/node_degree 또는
    새 c_learning/learned_c checkpoint의 GPU mean-C 개입. 원 validation·source/cache/checkpoint
    무결성 확인, 별도 보고서. 재학습 없음.
  - `sparse.py`: paper headline sparse operator와 packed variable-graph batch.
  - `paper_data.py`: S1–S4 generated protocols와 deterministic cache.
  - `public_data.py`: PascalVOC-SP와 ogbg-molhiv adapter.
  - `paper.py`: 보조 `core/all` runner, 우리 모델의 내부 ablation/metrics/artifacts.
  - `model.py`: 저수준 연산 및 수학 단위 검증용 유틸리티. legacy 실행기와 production synthetic generator는 제거했다.
- `research/cycle_pe/`
  - `benchmark_data.py`, `benchmark_models.py`, `benchmark.py`: 기본 ZINC/Peptides v1 학습/평가.
  - `v2/`: 전체 좌영공간 기저, 별도 cache·encoder·benchmark·실행 파일.
  - `features.py`: fundamental basis, set statistics, projector 수학.
  - `paper_data.py`: CycleCount-OOD generator와 exact cycle labels.
  - `paper_adapters.py`: BREC/ZINC adapter와 안전한 download/cache 처리.
  - `paper_model.py`: 네 PE variant와 공통 graph backbone.
  - `paper_train.py`: supervised train/eval/runtime.
  - `paper.py`: core/BREC/ZINC paper runner.
- `research/tree_augmentation/`
  - `augmentation.py`: full-β chart, transition, algebra certification과 legacy probe.
  - `paper_data.py`: core/CSL/ZINC graph data와 BFS/DFS/Wilson samplers.
  - `paper_model.py`: variable-edge/variable-β chart encoder와 training/evaluation.
  - `paper.py`: fixed-vs-multi independent paper runner.

### 격리된 결합 prototype

`research/combined_later/`에는 과거 flow completion, hard observation preservation, edge
residual 실험이 남아 있다. `pyproject.toml` package discovery와 active pytest 경로에서
제외되며 master paper runner도 실행하지 않는다. 외부 리뷰어는 이 코드를 active contribution과
섞어 평가하면 안 된다.

## 3. Linux NVIDIA GPU 재현 파이프라인

지원 실행 환경은 Linux와 NVIDIA GPU가 있는 워크스테이션 또는 서버다. 로컬 Linux
터미널에서 직접 실행하거나 임의의 SSH client로 해당 환경에 접속해 같은 명령을 실행한다.
MobaXterm은 선택 가능한 SSH client일 뿐 의존성이 아니며, tmux도 필수 도구가 아니다.
Cluster의 login node에 GPU가 없다면 해당 cluster의 scheduler로 GPU allocation을 먼저 받는다.

### 3.1 설치

일반 사용자용 실행 순서는 [README.md](README.md), 환경별 조정은
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)에 있다. 저장소 root에서 실행한다.

```bash
conda env create -f environment.yml
conda activate new-gat
bash scripts/setup_gpu.sh
```

Setup은 활성 non-base Conda의 Python만 사용한다. 기본 `auto`는 위의 cu118/cu126 조합을
선택하며 `CUDA_WHEEL_TAG`로 명시적 선택도 가능하다. 이후 준비/학습은 설치된 조합을 유지한다.
직접 의존성은 선택한 requirements lock과 해당 constraints 파일의 exact pin으로 설치한다.
전이 의존성 전체를 사전에 잠근 것은 아니며 실제 설치 결과는
`.gpu-environment.json`, `.gpu-environment.freeze.txt`에 기록한다.
Version/import ABI/CUDA 검증은 유지하고 전체 pytest는 `RUN_TESTS=1`일 때만 실행한다.
Ubuntu 18.04/glibc 2.27은 위 기본 설치 대신 README의 별도 `new-gat-legacy` 생성과
`--profile legacy-cu118` 설치 경로를 사용한다. 설치 기록도 해당 Conda 환경 안에 분리한다.

삭제된 entrypoint는 `setup.sh`, `setup.ps1`, `smoke.sh`, `smoke.ps1`,
`run_all.py`와 세 트랙의 legacy `run.py`다. 설치·실험 안내는 위 단일 경로를 사용한다.
삭제된 소스는 Git 이력으로 복원 가능하다. 사용자 데이터·결과·로컬 환경은 삭제하지 않았다.

### 3.2 데이터 준비와 cache 확인

```bash
bash scripts/prepare_data.sh
```

기본 데이터 경로는 `data/paper/`다. 준비 단계는 모델 학습이나 CPU 시험 학습을 하지 않는다.
위 wrapper는 기본 `benchmark`에 `--prepare-only --allow-download`를 넘기며 공식 원본과
검증된 cache를 준비한다. 기본 실행은 S1–S4/CycleCount를 생성하지 않는다. GAT/PE benchmark
loader는 원본·전처리·분할 checksum을 검사하며, 학습 중 다운로드나 가짜 데이터 fallback은 없다.
v2는 `bash research/cycle_pe/v2/prepare_data.sh`로 별도 기저 cache를 준비한다.

아래 checker는 **보조 `core/all` 데이터 레지스트리 전용**이다. 위 기본 준비 다음에
모든 보조 cache까지 준비됐다고 인증하는 명령이 아니며 v2 기저 검증도 대신하지 않는다.

```bash
python scripts/check_datasets.py --profile paper --data-root data/paper --require-cache
```

보조 checker는 request/schema, split cardinality, graph IDs, tensor/target shape, finite 값,
content/artifact hash를 읽기 전용으로 검증한다. 상태는
`valid/missing/incomplete/corrupt/wrong_request`로 구분한다.
Data와 split seed가 다르면 `--data-seeds`, `--split-seeds`를 각각 지정한다.
기존 축소·가짜 public cache는 전체 benchmark cache로 통과시키지 않는다.

### 3.3 독립 실행

```bash
bash research/conductance_gat/reproduce.sh
bash research/cycle_pe/reproduce.sh
bash research/tree_augmentation/reproduce.sh
```

위 세 명령을 순서대로 실행하는 대안은
`bash scripts/reproduce.sh`다. 두 방식을 중복 실행할 필요는 없다.

기본값은 `benchmark`, CUDA FP32/AMP OFF, model seed `0` 하나, data/split/chart seed `0`,
workers 4다. PPI batch는 2, 분자/tree batch는 32이고 Cora/CiteSeer/PubMed/arxiv는 full-batch다.
Run ID는 실행마다 자동 생성하며 같은 ID를 덮어쓰거나 자동 resume하지 않는다.
기본 data/와 트랙별 results/는 clone에 포함되고 하위 run 디렉터리는 자동 생성된다.
트랙 실패 시 기본적으로 다른 독립 run은
계속하며 `--fail-fast`로 전체 중단을 선택할 수 있다.
공통 GPU 검사 실패는 child 학습 전에 전체를 중단한다.

GPU 사전검사는 CUDA 사용 가능 여부, device index, 현재 여유 메모리와 package import만 확인한다.
가짜 그래프 생성, tensor 학습 입력 생성, 모델 forward/backward는 수행하지 않는다.
이 검사는 실제 데이터의 메모리 적합성이나 학습 성공을 보장하지 않는다.
데이터 준비에는 GPU 검사를 실행하지 않는다.
기본 실행은 Conductance 1개, Cycle v1 1개, Tree CSL/ZINC 2개 child를 실행한다.
v2는 전용 `reproduce.sh`로 Cycle PE만 독립 실행한다.
보조 `all`을 명시한 경우에만 BREC는 batch16/workers0/no-AMP, 내부 seed 10개의 단일
child로 실행한다. 보조 Cycle PE의 `core`는 CycleCount만 실행하고 `all`의 CycleCount/ZINC는
외부 model seeds마다 반복한다.
공식 BREC에는 master의 cycle optimizer override를 적용하지 않는다.

### 3.4 중앙 및 트랙별 산출물

```text
runs/paper/<run-id>/
├── manifest.json
├── environment.txt
├── gpu-preflight.json
├── dataset-registries/
├── aggregate/
│   ├── aggregate.json
│   ├── samples.csv
│   ├── metrics.csv
│   ├── paired.csv
│   ├── efficiency.csv
│   └── failures.csv
└── logs/
```

`--results-root`를 주면 실제 모델 결과는
`<results-root>/<track>/<run-id>/...`에 분리된다. 중앙 runner는 모든 JSON이 parse 가능하고
NaN/Inf가 없는지 검사하고 dependency/source/registry hash를 기록한다.

Root 집계는 JSON의 모든 숫자를 metric으로 취급하지 않는다. Track별 폐쇄형 registry에 등록된
test metric만 data/split/chart seed를 grouping key로 유지하고 model seed에 대해 mean, sample
std, median, min/max, deterministic bootstrap 95% CI로 요약한다. Rule이 `pairable`이고 같은
model seed가 있는 condition만 right-minus-left difference와 paired Cohen's dz를 계산한다.
Seed/epoch/batch/config/history/count는 제외된다. Elapsed time, peak memory와
`trainable_active_parameters_only` count는 raw `efficiency.csv`로 분리하고 bootstrap/paired test를
하지 않는다. `ignored_numeric_fields`가 제외 수를 감사 가능하게 남긴다. 이 집계도 논문별
multiple-comparison correction이나 task-specific significance test를 대신하지 않는다.
단일 seed에서는 표본 std·bootstrap 구간·paired effect size를 추정하지 않고 null로 남긴다.
Aggregate schema 3의 `uncertainty_status`/`uncertainty_policy`를 확인하며, 명시적으로 bootstrap을
끈 경우의 과거 mean placeholder도 신뢰구간으로 해석하지 않는다.

## 4. 연구 트랙 A — Sparse Positive Conductance Operator

**기본 실행 범위:** `benchmark.py`에서 Cora/CiteSeer/PubMed/PPI/ogbn-arxiv의 노드 예측을
학습한다. hidden 64의 encoder, conductance 2층, LayerNorm/ELU/dropout, decoder를 사용하며
gate는 엣지 속성 없이 `abs(BH)`와 `(BH)^2`를 읽는다. CE/PPI BCE 학습, validation 선택 후
test 평가다. 아래 `node_only`/flux supervision 표는 이 노드 분류 loss가 아니라 보조
S1–S4 operator-identification 실험의 계약이다. 기본 결과·진단은 실험 상태 문서를 따른다.

### 4.1 실제 가설

핵심 paper layer는 dense incidence matrix를 만들지 않고 다음을 계산한다.

\[
g_e=H_v-H_u,\qquad
c_e=\operatorname{softplus}(f_\theta(\cdot))+c_{\min},\qquad
q_e=c_eg_e,
\]

\[
H'=H-\eta B^Tq
   =H-\eta B^T\operatorname{diag}(c_\theta)BH.
\]

학습하는 것은 arbitrary `B^TCB` matrix가 아니라 edge별 positive diagonal scalar `c_e`다.
하나의 scalar conductance가 모든 hidden/state channel에 공통 적용된다. Directed/signed 또는
channel-coupled matrix conductance는 구현하지 않았다.

`full`의 conductance input은 edge feature와 `|BH|,(BH)^2`; `edge_only`는 edge feature;
실제 `gradient_only` 구현은 `|BH|`만 사용한다. Node update는 oriented gather와 두 번의
`index_add_`로 수행되며 graph 크기가 다른 sample을 packed batch로 처리한다.

Orientation을 뒤집으면 gradient/flux 부호는 바뀌지만 undirected orientation-invariant edge
feature라는 전제에서 conductance와 node update는 불변이다. `sum(B^Tq)=0`이므로 node-state
총합을 보존한다. 이 보존성은 conductance 전파 블록에 대한 것이지 encoder/LN/ELU/decoder를
포함한 분류기 전체의 보존성은 아니다. 기본 benchmark의 step은 그래프별
`0.95 / max_i(d_i^C)`이며, 보조 연산은 요청 step에 graph별 stability cap을 적용한다.

이 구조는 original softmax-neighbor GAT라기보다 positive symmetric diffusion/transport
operator에 가깝다. 논문 명칭과 novelty claim은 GRAND류 diffusion, anisotropic diffusion,
edge-conditioned convolution, graph neural PDE 문헌과 다시 대조해야 한다.

### 4.2 보조 S1–S4의 학습 objective와 내부 비교군

| 이름 | 입력/구조 | supervision | 해석 |
|---|---|---|---|
| `isotropic` | learned scalar `C=cI` | node message | diffusion baseline |
| `edge_only` | `c=f(x_E)` | node message | static edge law |
| `gradient_only` | `c=f(|BH|)` | node message | state-only law |
| `full` | edge + gradient | node message only | headline |
| `full_flux_supervised` | full | per-edge flux only | supervised neural ceiling |
| `full_joint` | full | node + flux | objective ablation |
| `flux_ls` | analytic | evaluation flux | transductive ceiling |
| `node_message_nnls` | projected NNLS | evaluation node message | identifiability ceiling |
| `oracle` | true C | none | data/operator oracle |

보조 S1–S4 headline `full`의 loss path는 flux target을 읽지 않는다. Regression test가 flux label을
삭제하거나 바꿔도 node-only loss가 동일함을 확인한다. LS/NNLS는 inductive learned
baseline으로 해석하면 안 된다.

### 4.3 Generated core S1–S4

| Suite | Full split/생성 | 핵심 평가 |
|---|---|---|
| S1 | graph 42/9/9, excitation 6/3/3; train graph의 new excitation 84개 별도 | held-graph 및 seen-graph excitation generalization, flux/node-message relative L2, log-C recovery |
| S2 | train/val: ER-like/RGG-like n=16–32, test: grid/barbell n=48–96 | topology/size OOD graph-macro error |
| S3 | graph 12/3/5, graph당 trajectory 1, horizon 50 | rollout 1/5/10/50, norm growth, dissipation/stability violation |
| S4 | contrast 1/10/100 × active fraction 1/0.25 × SNR inf/40/20 = 18 cells | known-contrast conditional recovery/error curve와 excitation coverage |

Cache spec에는 generator version, seed, full protocol, graph IDs, content/file SHA-256가 들어가며
deterministic reload 때 모두 검증한다.

S4의 edge feature에는 `log10(contrast)/2`가 포함된다. 따라서 이것은 관측 가능한 operating
condition을 준 조건부 복원 실험이지 blind contrast identification이 아니며, unknown
conductivity-range recovery의 근거로 사용하면 안 된다. 18개 factor cell이 모두 train/validation/
test에 있으므로 unseen-contrast OOD가 아니라 held-graph-ID empirical recovery다. 또한 truth
conductance는 graph별 edge-feature min/max로 정규화되어 graph-global context에 의존하지만
headline estimator는 edge-local 입력만 보므로, S4 오차에는 identifiability뿐 아니라
function-class misspecification도 섞인다.

### 4.4 보조 Public benchmarks (`all`)

- PascalVOC-SP: PyG LRGB official train/val/test, node classification, macro-F1.
- ogbg-molhiv: OGB official scaffold split, graph classification, OGB ROC-AUC evaluator.

이 보조 public 경로도 우리 conductance model만 실행한다. 이전 no-message MLP, sparse GCN,
custom single-head edge-aware GAT, GINE 경쟁 모델 구현은 제거했다. Parameter count를 기록하고
MolHIV의 OGB AtomEncoder/BondEncoder는 원자·결합 입력을 읽는 구성요소로 유지한다.
외부 점수는 논문 표를 인용하며 모델·학습 조건 차이를 표시한다.

### 4.5 보조 suite 산출물과 테스트

Standalone:

```bash
python -m research.conductance_gat.paper \
  --suite all --data-root ./data \
  --output-dir ./results/conductance-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --batch-size 32 --workers 4 --amp
```

성공 시 `summary.json`, `metrics.csv`, `history.csv`, CPU-portable `models.pt`가 생기며 runtime,
CUDA/AMP, peak memory, elapsed time을 기록한다. `summary.json`은 resolved
`data/split/chart/model` seed axes와 suite별 applicability도 기록한다. Generated core cache와
graph/excitation/trajectory 생성에는 `data_seed`만, model 초기화와 training shuffle에는
`model_seed`만 적용한다. 현재 conductance core의 split assignment는 data generation에
포함되어 별도 `split_seed`가 적용되지 않으며 chart 축도 없다. Official PascalVOC-SP와
MolHIV의 split/chart seed는 명시적으로 `not_applicable`이다. 기존 `--seed`는 명시하지 않은
축의 standalone fallback으로만 유지한다.

초기 보조 프로토콜의 Conductance test 20개는 sparse-vs-dense algebra, variable graph isolation, positivity,
orientation, mass conservation, objective leakage, S1/S2/S3/S4 split, deterministic cache,
S2 full cardinality contract, real public adapter, collision refusal와 가짜 데이터 옵션 거부를
검사한다.

### 4.6 반드시 재검토할 gap

1. 보조 S1–S4 및 PascalVOC-SP/MolHIV 결과는 제공받지 않았다. 기본 benchmark 5개
   데이터의 5-seed 집계, seed 0 GPU/full audit, PPI/arxiv의 2×2 새 학습은 확보했다.
   Node-degree 정규화 개선은 관측했지만 학습 C의 순수 기여, 다른 seed의 재현 여부,
   외부 논문과의 조건 차이는 아직 검토 대상이다.
2. S1/S2는 graph ID를 분리하지만 cross-split exact isomorphism/feature/C-law content hash
   guard가 없다. Registry는 더 이상 구현되지 않은 dedup을 claim하지 않는다.
3. generator 내부 명칭 `er`와 `rgg`는 엄밀한 G(n,p)/radius RGG가 아니다. 각각 connected
   recursive-tree+random pairs와 Euclidean MST+shortest pairs 형태이므로 논문에는 ER-like,
   RGG-like로 쓰거나 표준 generator로 교체해야 한다.
4. S3은 graph당 trajectory가 하나라 trajectory split이 graph split에 종속되고, unseen graph와
   unseen initial-condition 효과를 분리하지 못한다.
5. Core neural ablation은 input과 capacity가 함께 바뀌며 active parameter count를 기록하지 않는다.
   따라서 edge-only/gradient-only/full 차이를 순수 input contribution으로 해석하면 안 된다.
6. 보조 public 경로는 우리 conductance 1-layer 모델만 사용한다. 기본 benchmark는 2-layer
   모델이며 외부 GCN/GAT/GATv2/SAGE 구현은 없다. 외부 논문 표와의 파라미터 예산·depth·
   dropout·scheduler·tuning 조건을 동일하게 맞췄다고 주장할 수 없다.
7. `suite=all` public 단계가 실패하면 그 child 안에서 이미 끝난 core를 독립 partial artifact로
    보존하지 않고 중앙 log만 남을 수 있다.
8. Real sensor conductance recovery dataset과 signed/directed/channel-matrix negative control이
    없다. Roman-empire는 planned, PGLib/MATPOWER는 현재 core가 아니다.

이미 교정된 항목: `gradient_only` 문서는 실제 `|BH|` 입력과 일치한다. Reciprocal
categorical conflict는 거부하고 continuous attribute는 평균하며, PascalVOC mean CE는 node
label 수로 train/validation 합산한다.

## 5. 연구 트랙 B — Static Cycle-space PE

**기본 실행 범위:** `benchmark.py`는 ZINC-12K/Peptides-struct에서 여섯 통계형 `cycle_set`
v1만 학습한다. 좌영공간 기저벡터 전체를 입력하는 모델은 별도 `v2/`의 `cycle_basis_v2`이며
아직 GPU 결과를 받지 않았다. 아래 네 variant, CycleCount/BREC/보조 ZINC 설명은 명시적으로
선택하는 `core/all` suite 계약이다. 기본 benchmark가 이 네 모델을 모두 실행하는 것은 아니다.

### 5.1 실제 가설

Root-0 BFS tree로 topology에서 `F_T`를 학습 전에 한 번 계산하고 edge PE로 전달했을 때,
ordinary degree/topology feature만 쓰는 같은 backbone보다 cycle-composition/expressivity/분자
task에 도움이 되는지 검증한다. Learned conductance, node potential, sample circulation
coefficient, layer-to-layer cycle state는 없다.

### 5.2 보조 suite의 네 PE variant

| Variant | 실제 입력 | 불변성/주의 |
|---|---|---|
| `no_pe` | zero PE | 공통 backbone control |
| `raw` | `F_T[e,:]` train-width zero padding | sign/order/tree/orientation dependent diagnostic |
| `set` | edge별 fundamental-cycle set 6 통계 | column sign/order 및 row orientation flip 불변, tree change 불변 아님 |
| `projector` | full `P=F(F^TF)^{-1}F^T` row DeepSets | invertible basis change와 orientation sign에 불변, O(m²) |

Paper projector encoder는 단순 `diag(P)`만 쓰지 않는다. 각 row의 full dense pair feature
`|P_ij|, |P_ij|², P_jj`를 encode하고 row mean/max, diagonal/row magnitude를 합친다. 반면
삭제된 legacy bridge-vs-cycle probe는 `diag(P)` leverage로 target을 직접 드러냈으므로 headline
evidence에서 제외된다.

전처리의 lazy 범위는 제한적이다. 모든 variant에서 incidence, root-0 BFS tree와 full
fundamental basis `F_T`/`raw_basis`를 계산한다. 요청 여부에 따라 생략되는 것은 set 통계와 dense
projector뿐이다. 따라서 `no_pe`는 PE를 model input으로 사용하지 않지만 cycle-basis 전처리 비용까지
없애는 순수 end-to-end timing control은 아니다.

Raw width는 train split의 최대 `β`만으로 정한다. OOD graph의 `β`가 더 크면 절단하거나
test-fit하지 않고 해당 split을 N/A로 기록하며 다른 PE variant 평가는 계속한다. Validation
일부가 overflow면 compatible subset만 early stopping에 사용하고 full validation raw metric은
N/A다.
여기의 `raw`는 `cycle_basis_v2`가 아니다. Train-width/overflow 제약은 전체 가변 기저를
보존하는 v2에 적용되지 않는다.

공통 backbone은 variable-edge/variable-β graph list batch, node/edge encoder, symmetric endpoint
edge update, 양방향 mean message passing, residual LayerNorm, node/edge mean-max graph pooling을
사용한다. Edge/node/graph head는 독립 task마다 새 모델로 생성된다.

### 5.3 CycleCount-OOD v4

Full split:

| split | graphs | family/size |
|---|---:|---|
| train | 10,000 | cubic regular 및 tree+chords, n=14–22 |
| validation | 2,000 | train regime held-out |
| ID test | 2,000 | train regime held-out |
| size OOD | 3,000 | regular/tree+chords, n=28–38 |
| family OOD | 3,000 | unseen small-world/local-chords |

Exact target:

- edge: C3–C6 participation, shortest containing cycle length, short-cycle congestion.
- node: C3–C6 participation.
- graph: C3–C6 total count.

Edge/node/graph target은 별도 model/head/checkpoint로 학습해 graph target이 node/edge auxiliary
label에서 직접 유도되는 leakage를 막는다. Metric은 MAE, RMSE, train-standard-deviation
normalized MAE, graph-macro MAE, rounded exact accuracy다.

Degree sequence 또는 `(n,m,β)` matched counterfactual은 아직 없다. 따라서 현재 protocol로
degree를 완전히 통제한 cycle-composition claim을 하면 안 된다.

Cycle PE도 data/split/chart/model seed 축을 분리한다. CycleCount 생성과 cache identity는
`data_seed`, supervised model 초기화와 DataLoader shuffle은 `model_seed`만 사용한다.
CycleCount의 split regime는 generator-defined라 `split_seed`는 `not_applicable`이고, static
BFS fundamental-basis PE에는 sampled chart가 없어 `chart_seed`도 `not_applicable`이다.
기존 `--seed`는 명시되지 않은 축의 호환 fallback이며 manifest가 resolve 값과 실제 axis 사용
정책을 함께 기록한다.

### 5.4 BREC v3

- Official 400 non-isomorphic pair와 6 category.
- 기본 relabel `q=32`, threshold `72.34`.
- 내부 search seed `100,200,...,1000`.
- Embedding difference `D=L-R`.
- Hotelling statistic:

\[
T^2=\bar D^T\operatorname{pinv}(\operatorname{cov}(D))\bar D
\]

q multiplier는 없다. Train distinguishability와 isomorphic reliability를 별도로 계산한다.
Official mode는 batch 16, 20 epochs, LR/weight decay `1e-4`, float32, no AMP,
no clipping, no shuffle를 강제한다. Variant와 search seed별로 한 번 seed한 뒤 400 pair를
순서대로 실행하고, upstream과 호환되는 seed별 `Correct`, `Fail`, `Real_correct`를 따로
보존한다. `global_valid=true`는 모든 seed가 complete하고 reliability failure가 0인지 보는
**저장소 자체의 보수적 gate**이며 upstream BREC metric이 아니다. Upstream search가 정의하지
않은 official any-seed union은 없다.

위 상수와 제어 흐름은 upstream reference에 정적으로 맞췄지만, 같은 model을 양쪽 runner에서
실행해 golden output을 비교하는 differential parity test는 아직 없다. 따라서
“official-protocol compatible”이라고는 할 수 있어도 bytewise/numerically identical한 upstream
execution이라고 주장하면 안 된다.
기존 union은 custom mode의 `custom_pairwise_union`으로 격리했다.
바깥 `model_seed`는 BREC official 실행에 섞지 않으며, 내부 search seed 열 개가 manifest의
별도 protocol axis다.

Adapter는 official mode에서 q=32, 400 pairs, 51,200 records를 강제하고 pair 단위 lazy
decode한다. Full download는
opt-in이며 HTTPS host, archive path traversal, Windows path, symlink, member/size/ratio와 NPY magic을
검사하고 ZIP/NPY SHA-256을 provenance로 저장한다. Upstream이 canonical SHA-256 pin을
공개했다고 주장하지 않는다. 2-pair offline custom 입력은 tests 내부의 단위검사 fixture이며,
실험 CLI의 축소 BREC 옵션이 아니다.

중요: full `--prepare-only`는 400 pair 전체가 아니라 first/last pair만 decode/PE sanity check한다.

### 5.5 보조 suite의 ZINC-12K

PyG `ZINC(subset=True)` official train/validation/test adapter를 사용한다. Atom은 28-way,
bond는 4-way categorical one-hot으로 변환하고 reciprocal bond type 불일치를 거부한다.
Graph regression target은 constrained solubility다. 네 PE variant를 같은 backbone에서 비교한다.

코드는 PyG adapter가 official split을 준다고 신뢰하며 loader length가 정확히
10,000/1,000/1,000인지 별도 assert하지 않는다. 축소 데이터 실행 옵션은 제거했으며
공식 전체 split과 실제 cache만 사용한다. 기본 `benchmark`의 별도 split 검증과 구분한다.

### 5.6 보조 suite 산출물과 테스트

Standalone 예시:

```bash
python -m research.cycle_pe.paper \
  --suite core --data-root ./data \
  --output-dir ./results/cycle-core-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --variants raw,set,projector \
  --core-targets edge,node,graph --batch-size 64 --workers 8 --amp
```

Supervised artifact는 `<suite>/<level>/<variant>/` 아래 `model.pt`, `metrics.json`,
`history.json`, `runtime.json`을 저장한다. BREC는 `<variant>/pairs.json`과 `metrics.json`.

Master runner에서는 child output container에 suite 이름이 한 번 더 생긴다. 예를 들어 실제
core manifest는 `.../model-seed-0/core/core/manifest.json`, BREC는
`.../brec-official-10-seed/brec/...` 구조다.

초기 보조 프로토콜의 Cycle test 43개는 PE invariance/non-invariance, β=0, exact labels, 20k split specification,
deterministic cache, train-only raw width, variable batch, BREC layout/download/T²/reliability/seed
aggregation, variant-lazy projector, ZINC fixture와 CLI collision/partial preservation을 검사한다.

### 5.7 반드시 재검토할 gap

1. 보조 CycleCount 20k/BREC 400-pair 전체 결과는 제공받지 않았다. 기본 benchmark의
   ZINC-12K/Peptides-struct `cycle_set` v1 5-seed 결과는 확보했지만, 기저벡터 v2의
   GPU 학습 결과와 같은 backbone의 PE 제외 효과 분리는 아직 없다.
2. 모든 variant가 `F_T`를 계산한다. Set 통계와 projector만 요청에 따라 생략되며,
   `no_pe/raw/set`은 dense projector를 만들지 않는다. Projector variant 자체는 dense `m×m`,
   O(m²)이고 대형 graph scaling 검증이 없다.
3. Suite-level `preparation_seconds`는 여러 variant 비용이 섞이고 per-variant efficiency table은
   전처리 CPU time/RSS를 포함하지 않는다. 모든 split의 dense projector를 동시에 보관하는 비용도
   별도로 계측하지 않아 현재 표로 projector overhead를 공정하게 비교할 수 없다.
4. 네 variant의 parameter allocation은 거의 같게 생성되지만 실제 active path가 다르고 manifest에
   exact parameter count를 기록하지 않는다.
5. Auxiliary label perturbation을 통한 직접 leakage negative test는 없다. 구조적으로 target level별
   별도 모델인 것만 테스트한다.
6. Degree/`(n,m,β)` matched 또는 1-WL-indistinguishable이면서 C3–C6 target이 다른 known-contrast
   pair가 없다. 현재 CycleCount만으로 degree/size/family confound와 cycle signal을 완전히 분리할
   수 없다.
7. Raw/set은 root-0 BFS tree와 node labeling에 의존한다. Core/ZINC에는 isomorphic relabeling이나
   spanning-tree shift robustness 평가가 없어 graph-isomorphism-invariant PE라고 주장할 수 없다.
8. Raw는 큰-β OOD split 전체가 N/A가 될 수 있고, validation overflow 시 raw만 compatible subset으로
   early stopping한다. 따라서 hardest OOD에서 완전한 4-way 비교와 동일 validation-distribution
   비교가 성립하지 않을 수 있다.
9. Upstream 호환 field는 seed별 `Correct/Fail/Real_correct`다. `global_valid`는 저장소 자체 gate이고
   `custom_pairwise_union`은 custom metric이므로 둘 다 upstream official score로 표현하면 안 된다.
10. Official BREC의 상수/제어 흐름은 정적으로 맞췄지만 upstream differential/golden-output parity는
    검증하지 않았다.
11. Official BREC는 최대 4×10×400 pair training 결과를 메모리에 모은 뒤 마지막에 기록한다.
    Pair/seed 단위 incremental checkpoint와 resume가 없어 중단 시 현재 child 진행분을 같은
    run-id로 재개할 수 없다. Pair decode/PE 전처리도 variant/seed마다 반복된다.
12. 외부 모델은 사용자가 정한 실행 범위에서 제외한다. 관련 논문의 표를 인용하고
    split/입력/예산/평가 조건이 일치하는지 기록한다. 인용 점수에 대해 우리 seed와의
    paired significance를 주장하지 않는다.

## 6. 연구 트랙 C — Full-β Spanning-tree Chart Augmentation

### 6.1 실제 가설

같은 physical graph와 downstream label을 유지하고 spanning tree만 바꿔 full fundamental-cycle
chart `F_T∈R^{m×β}`를 여러 개 제공하면, root-0 fixed BFS chart 하나만 본 모델보다 held-out
chart와 topology OOD에 강해지는지 검증한다.

Full chart 사이에는

\[
F_{T_2}M=F_{T_1}
\]

인 invertible coordinate transform이 있고 구현은 수치적으로 integer unimodular structure와
cocycle을 인증한다. `k<β` truncation은 lossless라고 부르지 않고 명시적으로 비활성화한다.

### 6.2 Sampler와 training fairness

- fixed condition: graph당 root-0 BFS chart 1개.
- multi condition: graph당 random-root BFS/DFS만 섞은 finite chart bank. Full 기본 8개,
  축소 실행 profile은 제거했다. Wilson은 sampler-family OOD 평가를 위해 training에서 제외한다.
- `random_priority_kruskal`: Wilson과 다른 non-uniform legacy sampler이며 headline multi
  condition에는 들어가지 않는다.

Multi-chart는 매 update마다 새 tree를 online resample하는 것이 아니라 시작 시 생성한 finite
bank에서 minibatch sampling한다. Fixed와 multi는 architecture, model seed, optimizer와 optimizer
update 수(800)를 동일하게 맞춘다. Data/split/chart/model seed는 독립 축으로
manifest에 기록하며 chart bank는 chart seed, Torch 초기화와 minibatch는 model seed만 사용한다.
Distinct view 수는 1 대 K로 다르며 graph에
unique spanning tree가 충분하지 않으면 duplicate chart가 반복될 수 있다.

평가 quadrant:

| graph regime | `fresh_chart_seen_family` | `fresh_chart_unseen_family` |
|---|---|---|
| ID graph | fresh random-root BFS | fresh held-out Wilson |
| OOD graph | fresh random-root BFS | fresh held-out Wilson |

평가 BFS family는 fixed와 multi가 모두 training에서 보았고, Wilson family는 둘 다 보지 않았다.
따라서 model별로 의미가 달랐던 기존 seen/unseen 표기를 제거하고 fresh chart와 sampler-family
shift를 명시적으로 분리한다.
Wilson output이 BFS output과 같은 tree라는 이유로 reject하지 않으므로 UST를 BFS 결과에
조건부로 만들지 않는다. Exact-tree overlap은 별도 diagnostic으로 기록하며, held-out은 sampler
family exposure에 관한 용어다.

### 6.3 Model

`GraphChartView`는 physical graph, dense `F_T`, target, tree key와 optional chemistry를 보존한다.
Batch는 `[batch,max_edges,max_beta]`로 dense padding하고 masks로 variable edge/β와 β=0을
지원한다.

`VariableBetaCycleEncoder`는 각 `F[e,c]`에서 `|F[e,c]|`, `F[e,c]²`, normalized cycle
support를 encode하고 valid cycle columns에 sum/mean/max set pooling한다. 여기에 endpoint degree,
`1/n`, `1/m`, ZINC일 때 symmetric endpoint atom embedding과 bond embedding을 합쳐 edge
representation을 만들고 graph-level sum/mean/max pooling 후 예측한다.

이 sign-even 입력과 set pooling은 같은 physical tree의 edge orientation, cycle-column sign,
aligned row/column ordering, chemistry를 함께 옮긴 node relabeling에 invariant하다. 하지만
arbitrary spanning-tree basis change에는 본질적으로 invariant하지 않다. 특히 node relabeling으로
label-dependent BFS/DFS가 다른 physical tree를 선택하면 별도 chart shift다. 바로 그 chart
dependence를 multi-chart augmentation으로 줄이는 것이 가설이다. Dense padded `m×β`이므로 대형
graph scaling 결과는 없다.

### 6.4 Datasets

#### Core CycleCount-OOD v2

- Full: train 128, validation 24, ID test 40, OOD test 40.
- ID regime: n=8–12 recursive-tree+chords.
- OOD: C3–C6 여러 개를 bridge로 이은 unseen cactus cycle-chain family.
- Target: physical graph에서 chart sampling 전에 계산한 exact graph-level C3–C6 count 4-vector.
- `(n,m)` bucket 내 graph isomorphism dedup으로 split 중복 방지.

#### CSL

- PyG `GNNBenchmarkDataset(name='CSL')`.
- Label별 deterministic permutation 후 5 folds.
- fold 0–2 train, 3 validation, 4 test.
- 10-class accuracy.

#### ZINC-12K

- PyG official train/validation/test.
- Atom integer `x`와 canonical undirected bond category를 cache에 lossless 보존.
- Reciprocal category conflict, duplicate directed arc, self-loop, range error를 거부.
- 모든 chart에서 chemistry tensor는 고정되고 model prediction에 실제 영향을 주는 regression
  test가 있다.

### 6.5 Metrics, artifacts, tests

Regression은 MAE, normalized MAE, RMSE, graph-macro MAE, graph별 worst-chart MAE 평균,
prediction spread/std, rounded flip rate/exact vector accuracy를 기록한다. Classification은 accuracy,
graph-macro accuracy, worst-chart accuracy, probability std와 flip rate를 기록한다.

Standalone:

```bash
python -m research.tree_augmentation.paper \
  --suite all --data-root ./data \
  --output-dir ./results/tree-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --batch-size 16 --workers 4 --amp
```

성공 artifact는 `summary.json`, `manifest.json`, `fixed_bfs_model.pt`,
`multi_chart_model.pt`. Checkpoint는 CPU state dict와 target normalization/settings를 저장한다.

초기 Tree protocol test 21개는 full-β lossless basis, chart transition/cocycle, Wilson UST 분포,
training/held-out sampler 분리, chart-independent target, β=0/1/2 masks, orientation/column-sign/
edge-order/same-tree-node-relabel gauge invariance, deterministic cache/split, ZINC chemistry
roundtrip/invariance/sensitivity, collision과 suite partial failure를 검사한다.

### 6.6 반드시 재검토할 gap

1. 기본 CSL/ZINC 5-seed 집계는 확보했다. CSL 정확도는 개선되지만 unseen 절대 성능은
   낮고 flip rate는 악화한다. ZINC 평균 MAE도 개선되지 않았다. 보조 core 결과와
   표준 논문 protocol 재현, 유의성 검정은 확보한 근거에 포함되지 않는다.
2. Adaptive/learned MST, trainable spanning tree, MST sparsification은 구현하지 않았다.
3. Multi-chart는 finite offline bank이지 online resampling algorithm이 아니다.
4. Validation split은 cache에는 있으나 현재 fixed 800-update training의 early stopping 또는
   hyperparameter selection에 사용하지 않는다.
5. Fixed-vs-multi 같은 encoder 비교만 있고 conventional GNN/GAT/no-PE, sampler별 ablation,
   tuning 및 CI/significance가 없다.
6. `paper_headline_eligible=True`는 전체 protocol run의 artifact flag일 뿐 성능이나
   재현성 통과 판정이 아니다.
7. Full dense `[batch,max_edges,max_beta]` padding의 memory/scaling study가 없다.
8. Degree-matched OOD와 BREC chart stress는 없다.
9. CSL은 fixed-beta sanity에 가깝고 ZINC도 standard SOTA reference configuration과 비교하지
   않는다.
10. Fresh-seen은 exact train chart 재사용이 아니라 양쪽 model이 본 BFS family의 새 chart이고,
    fresh-unseen은 양쪽 model이 training에서 제외한 Wilson family다.
11. Lossy chord selection과 `k<β` compression은 의도적으로 비활성화되어 있다.

## 7. 데이터 레지스트리 상태

기본 `prepare_data.sh`/`reproduce.sh`는 `benchmark`다. 해당 데이터와 이번 결과 범위는 다음과 같다.

| Track / 버전 | 기본 benchmark | 제공된 결과 |
|---|---|---|
| Conductance | Cora, CiteSeer, PubMed, PPI, ogbn-arxiv | 5개 데이터 5-seed 집계; seed 0 GPU/full audit; PPI/arxiv 2×2·C-learning seed 0 재학습 |
| Cycle PE v1 | ZINC-12K, Peptides-struct | `cycle_set` 5-seed 집계 |
| Cycle PE v2 | 위와 같은 공식 원본·split, 별도 기저 cache | 구현·단위 검증; GPU 결과 미수령 |
| Tree augmentation | CSL, ZINC-12K | fixed-BFS/multi-chart 5-seed 집계 |

아래 12개는 `scripts/check_datasets.py --profile paper --json`이 확인하는 **보조 `core/all`
계약**이다. 기본 benchmark나 v2 기저 cache 검증으로 대체해서 해석하지 않는다.

| Track | Generated/core | Public/all |
|---|---|---|
| Conductance | S1, S2, S3, S4 | PascalVOC-SP, ogbg-molhiv |
| Cycle PE | CycleCount-OOD v4 | BREC v3, ZINC-12K |
| Tree augmentation | CycleCount-OOD v2 multi-chart | CSL, ZINC-12K multi-chart |

현재 로컬 public cache는 없으므로 로컬에서 `--require-cache`를 통과했다고 기록하면 안 된다.
사용자 서버의 benchmark 실행 결과가 있다는 사실과 이 작업 공간의 cache 준비 여부는 다르다.
새 환경에서는 해당 버전의 준비 명령과 loader의 checksum·split 검증을 수행한다.

## 8. 자동 검증 상태

검증 수치는 구현 시점별로 0절에 보존한다. 현재 새 learned_c checkpoint 검사 확장 후 전체
회귀는 **924 passed / 64 skipped** (32.11 s, exit 0), Ruff 통과다. 이전 C-learning 구현은
**890 passed / 64 skipped** (30.76 s, exit 0)였다. 2×2 구현 당시에는
**794 passed / 64 skipped**, 그 이전 확장 진단 구현 당시 전용 검사는 **89 passed**였다.
로컬 단위 검증과 사용자 제공 실제 GPU 진단·재학습의 범위를 혼동하지 않는다.

### 과거 기록 — 2026-08-30

아래는 더미 실행 경로 제거 및 실제 재현 실행 파일 추가 **당시**의 로컬 검증이다.
테스트·소스 파일 수는 현재 값이 아니며, Windows 진단 메시지도 해당 실행의 기록이다.

```text
pytest -q                     204 passed, 12 skipped (11.61 s, exit 0)
ruff check .                 All checks passed
ruff format --check .        77 files already formatted
README command contract      5 script entrypoints resolve to full protocols
code_summary --check         current, 89 source files
git diff --check             passed
```

트랙별 단위시험은 Conductance 27개, Cycle PE 46개, Tree augmentation 28개다.
나머지 103개는 공통 수학, cache, runner, 집계, 환경·문서 계약 검사다.
Linux/Bash 전용 12개는 현재 Windows 환경에 Bash가 없어 skip했다.
이 중 5개는 새 실행 파일의 인자·종료 코드 전달을 검증하는 Linux shell 검사다.
이 결과는 실제 Linux Conda/Bash/CUDA 실행 성공을 인증하지 않는다.

이번 검증은 실험 CLI의 더미 옵션 거부, 공개 데이터 부재·다운로드 실패의 오류 처리,
가짜/축소 cache 거부, 전체 scientific protocol 크기·seed 유지와 README 인자 계약을 확인한다.
GPU 사전검사가 sample tensor나 모델을 생성하지 않는지도 mocked hardware 단위시험으로 검사한다.
테스트용 작은 graph/adapter 입력은 tests 내부에만 있으며 공개 experiment profile로 제공하지 않는다.

전체 pytest는 exit 0으로 끝났지만 실행 중 기존 Windows faulthandler
`access violation` 진단이 다시 출력됐다. 이를 숨기거나 Linux GPU 성공의 증거로 해석하지 않는다.
당시에는 지원 Linux GPU 환경에서 실제 의존성 설치, cache 준비, 학습·평가 확인이 남아 있었다.
이후 사용자 benchmark 결과와 seed 0 진단을 받았으며 범위는 실험 상태 문서를 따른다.
과거 삭제된 더미 실행기의 숫자는 현재 검증·논문 결과로 이 문서에 재사용하지 않는다.

Read-only protocol 교차검토에서는 CycleCount full specification/hash가 이전 full protocol과
동일하고, BREC의 통계·reliability·학습·집계 함수 및 공식 설정도 유지됨을 확인했다.
전체 공개 데이터 cache나 GPU 학습 결과를 이 작업에서 생성하지 않았다.

## 9. 논문 claim 전 우선순위

### P0 — 이미 나온 결과의 검증과 Conductance 실패 원인 분리

1. 기존 run ID, source revision/dirty 상태, lock, data checksum, best checkpoint, history와
   seed별 원본 지표를 보존한다. 제공된 집계 텍스트와 서버 원본 artifact의 검증을 혼동하지 않는다.
2. 완료된 **C-learning의 learned checkpoint**에서 C를 그래프·층별 평균으로 바꿔 현재
   의존도를 확인한다. 원 validation 재현과 source/cache/checkpoint 무결성 검사 후에만
   대비를 해석하며, 과거 factorial checkpoint와 대상 run을 혼동하지 않는다.
3. 이미 나온 learned−fixed 차이가 PPI −0.140772pp / arxiv −0.006711pp인 사실과
   후속 개입의 점수·logit·prediction flip을 나란히 해석한다. 현재 checkpoint의 사용과
   재학습 이득은 다를 수 있다. 지금은 새 WD/dropout/epochs나 트랙 결합을 동시에 추가하지 않는다.
4. v2는 별도 코드·cache·run으로 GPU 검증한다. 기존 `cycle_set` 결과를 기저벡터 실적으로
   재분류하지 않는다. 실행 최적화 역시 동등성·peak memory·GPU 속도를 별도로 측정한다.
5. Tree의 chart-family OOD, validation 미사용, 연속 target에 부적절한 rounded 지표를
   반영해 claim을 제한한다. 지표/학습 변경은 문서 수정과 별도 작업으로 다룬다.

코드 수준 P0 교정은 완료됐다: semantic strict cache와 atomic publish, BREC official/custom
분리와 global reliability gate, Wilson train-family 제거, tree orientation gauge test, 네 seed 축,
closed root metric/efficiency 집계, exact CUDA constraints/verification,
cycle candidate CLI, stale S2 full-cache cardinality(112/24/48) 계약 교정을 반영했다.
과거 shape-stress는 더미 모델 실행 제거에 맞춰 hardware/import 검사로 교체했다.
위 코드 gate의 완료는 모든 scientific gap 해소를 의미하지 않는다. 기본 benchmark 집계와
seed 0 진단·2×2·C-learning 재학습은 있지만 새 learned checkpoint의 평균-C 검사,
보조 suite 전체, v2 결과, 가속 실측까지 완료된 것은 아니다.

### P1 — 강한 scientific claim 전에

1. Conductance: 관련 논문 표를 인용한 비교의 조건 확인, real physical/sensor conductance data
   또는 명확한 synthetic-only claim. 외부 모델을 저장소에 추가하는 작업은 현재 범위 밖이다.
2. Cycle PE: v1/v2 및 같은 backbone의 PE 제외 효과 분리, degree/`(n,m,β)` matched
   counterfactual, 기존 논문과의 수학적 차이 설명. v2의 임의 기저 회전/재정향 비불변성과
   dense SVD·전체 기저 메모리 비용을 확인하고, 보조 projector 비용은 별도로 측정한다.
3. Tree augmentation: 우리 모델의 BFS-only/DFS-only/Wilson-only ablation, validation 기반 selection,
   large-β scaling.
4. Leakage negative tests와 graph/canonical hash guard 강화.
5. 독립 reviewer가 BREC threshold/statistic/reliability와 category aggregation을 official source와
   다시 대조.

### P2 — 독립 가설 후 확장

1. Learned/adaptive spanning-tree 또는 variable tree distribution.
2. MST sparsification과 cycle coverage를 함께 다루는 별도 연구.
3. Conductance와 static cycle PE 결합.
4. 관측 edge flow가 있을 때만 particular solution + explicit cycle coefficient completion 연구.

이 확장들은 현재 구현된 contribution으로 쓰면 안 된다.

## 10. 외부 ChatGPT에게 권장하는 교차검증 질문

`code_summary.md`, 이 파일과 `docs/EXPERIMENT_STATUS.md`를 함께 주고 다음을 요청한다.
먼저 기본 benchmark/기저벡터 v2/보조 `core/all` 범위를 구분한다.

1. `B∈R^{m×n}` convention에서 `ker(B^T)`, `L=B^TB`, fundamental basis와 pseudoinverse 설명이
   수학적으로 정확한가?
2. 세 트랙 사이에 import, label, cache 또는 artifact를 통한 숨은 결합이 있는가?
3. Conductance `full node_only`가 실제로 flux target을 전혀 읽지 않는가?
4. S1–S4의 split이 graph/excitation/trajectory leakage를 충분히 막는가?
5. Conductance 5개 데이터셋 결과를 외부 논문 표와 비교할 때 split·전처리·모델 크기·학습
   조건 차이를 충분히 공개하는가? 이 저장소는 외부 5-model 비교를 실행하지 않는다.
6. CycleCount edge/node/graph task가 auxiliary-label leakage 없이 완전히 독립적인가?
7. Raw/set/projector의 invariance claim이 코드와 정확히 일치하는가?
8. BREC T², no-q convention, `isclose`, reliability와 10-seed aggregation이 official protocol과
   일치하는가? `custom_pairwise_union`이 official 결과와 완전히 분리됐는가?
9. Tree fixed-vs-multi가 optimizer exposure, distinct views와 sampler shift 측면에서 공정한가?
10. ZINC/CSL/PascalVOC/MolHIV adapter가 official split과 chemistry/feature 정보를 보존하는가?
11. CUDA/AMP/DataLoader에서 device mismatch, hidden CPU tensor, OOM 또는 nondeterminism 위험이
    있는가?
12. Existing tests가 놓친 failure mode를 우선순위와 함께 제시할 수 있는가?
13. 각 아이디어의 closest prior work와 실제 novelty boundary는 무엇인가?
14. 현재 artifact만으로 허용되는 claim과 금지해야 할 claim을 분리해 달라.
15. seed 0의 C 상수화·rho·전파 우회 결과에서 관측과 원인 가설을 정확히 구분하는가?
    그래프별 최대 차수 스텝과 C 공통 스케일 상쇄가 어떤 최적화 한계를 만드는가?
16. Cycle `cycle_set` 결과와 `cycle_basis_v2` 구현, GitHub 게시본과 로컬 소스 스냅샷을
    혼동하지 않는가? v2의 sign/order 불변성을 arbitrary basis-rotation 불변성으로 과장하는가?
17. Tree의 CSL 5-seed/단일 split과 chart-sampler OOD, ZINC MAE 및 부적절한 rounded
    보조 지표를 실제 의미에 맞게 해석하는가?
18. 2×2의 node-degree 개선과 learned C의 순수 효과를 혼동하지 않는가? Mean-C 추론 개입과
    fresh learned/fixed 학습 비교를 구분하고, 초기 backbone·학습 예산·parameter 수 차이를
    정확히 공개하는가? 1-seed 대비를 통계적 유의성이나 최종 test 성능으로 과장하지 않는가?

## 11. 공식 데이터/프로토콜 출처

- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch Geometric ZINC: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
- OGB graph property prediction: https://ogb.stanford.edu/docs/graphprop/
- LRGB PascalVOC-SP: https://github.com/vijaydwivedi75/lrgb
- BREC: https://github.com/GraphPKU/BREC
- BREC reference runner: https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py
- Benchmarking GNNs/CSL/ZINC: https://www.jmlr.org/papers/v24/22-0567.html
- CycleNet: https://proceedings.mlr.press/v231/yan24b.html

## 12. 최종 인수인계 문장

현재 저장소에는 세 독립 연구의 실행·artifact pipeline이 있고 사용자 제공 기존 benchmark
5-seed 결과, Conductance seed 0 GPU 진단과 PPI/arxiv 2×2·C-learning 재학습도 있다.
Node-degree 개선은 관측했지만 같은 정규화에서 learned C의 validation 이득은 관측하지
못했다. 이것이 보편적 동등성·novelty·경쟁력의 판정은 아니다. 다음 작업자는 같은 새 run의
learned checkpoint 평균-C 개입으로 현재 의존도를 확인하고, v2/최적화의 GPU 검증과
Tree protocol 한계를 독립적으로 다뤄야 한다.
Adaptive MST나 세 트랙 결합은 이후 별도 실험이며 기존 결과를 덮어쓰지 않는다.
