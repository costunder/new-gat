# GPT 전달용 전체 프로젝트 묶음

## 현재 검수 기준 — 2026-09-27, 구현 커밋 `4b4df52`

**이번 ZIP의 `MANIFEST.json`에 문서까지 포함한 정확한 커밋과 파일별 SHA-256이 있다.**
이 절이 현재 상태이며 아래 2026-09-26 및 그 이전 내용은 역사적 기록이다.
당시의 6조건/19조건, GATv2 미포함, validation-only 안내를 현재 계약으로 읽지 않는다.
이전부터 수정 중이던 세 전달 문서의 내용도 삭제하지 않고 아래에 보존했다.

- 주 벤치마크는 **ogbn-arxiv**다. 새 aggregation CLI와 하위 학습기는 PPI를 거부한다.
  PPI-only sampled_inductive 공개 실행 진입점도 중단했다. 과거 결과는 보존 자료다.
- 현재 기본 matrix는 **21조건**: incidence 내부 대조군 16개 + GCN + GraphSAGE +
  GATv2 + DUALFormer + DUALFormer no-skip control이다.
- 사전학습 모델/가중치 다운로드 없이 무작위 초기화부터 학습한다. 설치된 PyG의
  GCNConv, SAGEConv, GATv2Conv를 사용한다. 자기 실험의 저장 checkpoint 복원과 구분한다.
- 공통 reference 8층/256 hidden, large 12층/384 hidden 및 기존 학습 예산을 유지한다.
  공통 encoder/decoder·데이터·split·레시피 비교이며 parameter-matched/SOTA 재현은 아니다.
- validation으로 선택한 전체 matrix의 best checkpoint를 먼저 고정한 뒤
  `--evaluate-test`로 공식 test mask를 평가한다. 실제 데이터 본학습·test 평가는 미실행이다.
- RTX 5070 Ti의 합성 debug/무결성 회귀 검사 105개 통과 후 최종 변경 영향 검사 13개를
  재실행해 통과했다. 중복 실행을 118개로 합산하지 않는다. MIG 10GB 적합성은 미측정이다.
- **연구 전체 완성은 아니다.** arxiv의 고정/학습 C × full/sampled 4조건 통합 실행·보고,
  학습 예산 해석, 독립된 미관측 그래프 평가 규약, 실제 데이터 성능·총비용 측정은 남아 있다.
  arxiv 공식 transductive test를 독립 그래프 inductive 검증으로 취급하지 않는다.

읽는 순서: 이 절 → `HANDOFF.md` 최신 절 → `EXPERIMENT_STATUS.md` 최신 절 →
`docs/ARXIV_BASELINE_COMPARISON.md` → `docs/DEEP_IMPLEMENTATION_REVIEW_20260927.md`
(직전 감사의 역사적 근거) → 실제 소스 및 `CODE_SUMMARY.md`.
ZIP에는 기존 전달 문서 10개, 현재 코드·설정·테스트·실행 문서, 최신 및 이전 감사의
구분된 증거를 넣는다. 데이터, 가상환경, 모델 가중치 및 대형 학습 산출물은 제외한다.

### GPT에 줄 최신 검수 요청문

> README_FIRST.md의 2026-09-27 절과 MANIFEST.json부터 읽고 현재 코드의 요구사항 충족을
> 독립 검수해 줘. 아래 과거 문서는 날짜별 역사로 다루고, 현재 21조건 및 arxiv 계약은
> docs/ARXIV_BASELINE_COMPARISON.md와 실제 소스를 우선 대조해 줘.
> C는 입력 그래프가 아니라 공유 규칙으로 계산되는 conductance다. C 계산→가중
> 라플라시안→예측→loss→gradient→optimizer 경로, 에너지·cross-depth Gram·lift와
> 각 대조군의 식/구현 일치 및 비교 공정성을 확인해 줘. nonlinear rank와 가역성을
> 동일시하거나 depth history를 정확한 거리별 hop으로 해석하면 안 된다.
> GCN/GraphSAGE/GATv2의 실제 연산, root/self-loop/정규화/외부 residual의 구분,
> 공통 backbone과 모델별 parameter 수 차이, baseline 튜닝 한계를 검토해 줘.
> final_test.py의 전체 matrix 검증·디스크 증거·checkpoint 사전 고정·validation 재현·
> test mask 및 완료 결과 재사용의 무결성을 확인해 줘. test로 모델을 재선택하지 않는지 봐 줘.
> PPI 차단이 원래의 샘플링/inductive 연구 목표 완성을 의미하지 않는다는 점을 유지해 줘.
> 테스트는 합성 debug이며 실제 benchmark 성능·속도·MIG 적합성을 입증하지 않는다.
> 결과를 심각도순 오류(파일/함수/근거), 요구사항 누락, 비교 공정성, 허용/금지할 주장,
> 필요한 수정과 재검증으로 정리해 줘. 실행하지 못한 것은 정적 검수로 명시해 줘.

## 아래는 2026-09-26 전달본의 보존 기록

**최신화 기준: 2026-09-26 / 구현 기준 로컬 커밋 `9103fad`.**
기존 10개 문서를 유지하며 최신 우선 검수 대상은
`experiments/aggregation_comparison/`의 독립 비교 실험이다.
`CODE_SUMMARY.md`는 현재 source/test/config/script 원문으로 다시 생성했다.
아래 9월 중순 이전 안내는 역사적 기록이며 현재 상태는 이 절을 우선한다.

## 현재 상태

| 항목 | 확인된 상태 |
| --- | --- |
| 기존 incidence ablation | 사용자 제공 PPI v2 콘솔에서 학습·감사 8/8 passed. 최고 post_lift 0.989503. residual/FFN backbone의 내부 ablation |
| 새 외부 모델 비교 | `experiments/aggregation_comparison/`, 6조건, 새 모델·결과 폴더·run ID |
| 우리 모델 | 외부 residual/FFN 제거; 기본, pre-lift, 국소 에너지, 에너지+pre-lift |
| 외부 비교 대상 | DUALFormer(ICLR 2025) 공식 수식 기반 공통 레시피 구현과 별도 no-skip 변형 |
| 미포함 비교 | 원래 GAT, GATv2, M3Dphormer는 새 6조건에 없음. 2026 최신 SOTA 비교 완료 주장은 금지 |
| GPU 검증 | RTX 5070 Ti 16GB, 6개 reference 모델 FP32/BF16 포함 15개 CUDA 검사 통과 |
| 실제 benchmark | 새 6조건의 전체 데이터 학습·test 평가·다중 seed 비교는 미실행 |
| A100 MIG 10GB | 실제 적합성·처리량·전체 그래프 calibration 미측정 |
| 배포 | 로컬 구현/커밋 기준. 원격 서버 동기화·실행 완료로 간주하지 않음 |

GPU 검사에는 실패 이력도 있다. 기본 환경 2.14 CUDA의 최초 비결정적 실행에서
FP32 에너지 모델 재개 일치 검사 1개가 실패했다(14 passed / 1 failed).
테스트에만 결정적 CUDA 연산을 적용한 뒤 같은 허용 오차로 15개가 통과했다.
정식 학습 레시피까지 결정적으로 바꾼 것은 아니다. 근거와 범위는
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)의 최신 절에 있다.

## 이번 검수의 읽는 순서

1. `README_FIRST.md`: 이 최신 절과 아래 검수 요청문.
2. `HANDOFF.md`: 최신 요구사항 반영표, 코드 지도, 미구현 범위와 검수 질문.
3. `EXPERIMENT_STATUS.md`: CUDA 증거와 실패 이력, 과거 연구 결과의 구분.
4. `CONDUCTANCE_V5.md`: 최신 local Gram/lift/DUALFormer 설계와 과거 V5 계약.
5. `CODE_SUMMARY.md`: 현재 전체 원문. `# experiments/aggregation_comparison/model.py`부터
   engine, runner, calibration, integrity, audit, CUDA tests 및 기존 의존 구현을 확인한다.
6. `CONDUCTANCE_V2.md`, `CONDUCTANCE_V3.md`, `CONDUCTANCE_V4.md`, `CYCLE_PE_V2.md`,
   `RICH_SCALING_EXPERIMENTS.md`: 다른 연구 트랙을 각 날짜·계약별로 구분해서 검토한다.

소스 스냅샷은 파일별 원문이며 코드가 생략된 요약문이 아니다. LF 줄바꿈으로 정규화한다.
검수 ZIP에는 문서 10개, 실제 소스, 실행 문서, 해시 manifest와 CUDA XML/JSON 기록을
포함한다. `MANIFEST.json`으로 파일 무결성을 확인한다. 데이터 cache·가상환경·대형
checkpoint는 포함하지 않는다. 한 파일씩 전달할 때는 적어도 문서 10개를 함께 제공한다.

## 이번에 GPT에 그대로 줄 검수 요청문

> 첨부 자료를 읽고 NEW GAT의 현재 코드와 연구 설계를 독립적으로 검수해 줘.
> 먼저 README_FIRST.md의 2026-09-26 상태와 HANDOFF.md 최신 반영표를 읽어라.
> ZIP 접근이 가능하면 압축을 풀고 MANIFEST.json 및 실제 소스를 확인해라.
> 코드 실행 환경이 없다면 정적 검수와 실행으로 확인한 사실을 명확히 구분해라.
>
> 우선 `experiments/aggregation_comparison/` 6조건을 검토해라. 기존 8조건 내부
> ablation 및 역사적 PPI 점수를 새 외부 비교의 결과로 취급하지 마라.
> 우리 모델의 외부 residual/FFN 제거, 내부 diffusion의 identity 항,
> diagonal/cross-depth local Gram, live C gradient, pre-lift의 실제 forward/loss/
> backward/optimizer 연결을 코드로 확인해라. depth state를 exact-distance hop으로,
> feature rank 증가를 전체 Jacobian의 가역성으로 오해한 주장이 있는지 확인해라.
>
> DUALFormer 공식 commit 68fbdaf007af2f7d409cd435c4c48dd0e3155510과 수식을 대조해라.
> intrinsic residual/LayerNorm을 유지한 행과 제거한 control을 구분하고,
> 공통 width/head/depth/AdamW가 정말 공정한 비교인지 비판적으로 판단해라.
> 파라미터 수가 같지 않고 원 논문의 튜닝 레시피 재현도 아니라는 점을 반영해라.
> 원형 DUALFormer의 graph step 수와 새 reference의 8단계 차이도 검토해라.
>
> 공식 split·full graph·validation-only checkpoint 선택·모든 arm의 동일 자원
> calibration·update budget·source hash·optimizer/RNG 재개 계약을 확인해라.
> CUDA 15개 통과가 synthetic graph 검증임을 유지하고, 실패한 최초 재개 검사와
> 테스트에만 적용한 결정적 연산의 의미를 검토해라. MIG 적합성이나 SOTA를 추정하지 마라.
>
> 이어 다른 연구 트랙은 각 날짜와 결과 출처를 유지해서 검토해라.
> 결과는 (1) 심각도순 오류와 파일/함수/근거, (2) 비교 공정성 문제,
> (3) 구현 누락, (4) 현재 증거로 허용/금지할 주장, (5) 필요한 다음 실험으로 정리해라.
> 변경 제안에는 원래 규모를 유지하는 수정안과 재검증 방법을 적고,
> 검증 없이 모델·데이터·학습 예산을 축소하거나 과거 결과를 재해석하지 마라.

## 최신 근거의 경계

- 기본 `.venv`는 CPU 전용 설치를 수정하여 2.14.0+cu130을 사용한다.
  `.venv-gpu`는 2.13.0+cu130과 고정 의존성을 사용하는 별도 검증 환경이다.
  Windows CUDA smoke와 Linux production 설치 계약은 다르다.
- 과거 CPU 49개 통과는 수식·무결성 검사 기록이며 이번 전체 학습 완료가 아니다.
- 사용자 제공 PPI v2 콘솔의 8개 점수를 `EXPERIMENT_STATUS.md` 최신 절에 반영했다.
  최고 post_lift 0.989503은 새 6조건의 test/SOTA 점수가 아니다. 학습·감사 완료 상태는
  콘솔로 확인했으며 원본 manifest/checkpoint/audit 내용의 독립 검증은 아직이다.
- 최신 모델 선택을 모두 해결하지 못했다. 현재 외부 comparator는 2025 DUALFormer 하나다.
- 서버의 기존 checkpoint와 전체 학습 결과는 별도 원본 자료가 있어야 재검증할 수 있다.

## 이전 전달 안내의 역사적 기록 — 현재 상태로 읽지 않음

최신 보존(2026-09-16): 작은 기존 연구 산출물 13개(338.6 KiB)를 Git에 포함했다.
초기 합성 conductance/cycle/tree 검증과 보류된 combined prototype의 기록이며,
최근 서버 GPU 결과가 아니다. 원본 파일을 바꾸지 않고 byte-preserving 속성으로 보존한다.
목록·출처·제외 범위는 `EXPERIMENT_STATUS.md` 맨 위에 있다. 임시/smoke/cache는 계속 제외한다.

최신 수정(2026-09-12): 동일 사양 GPU를 재할당받아 물리 번호가 1→4로 바뀐 경우
같은 edge-selection run의 기존 학습/교정 기록을 보존하고 새 할당을 별도로 재검증한다.
미완료 학습은 다음 epoch와 optimizer/RNG부터 복원한다. 실제 다른 GPU 사양이나
런타임 변경을 무조건 허용하지 않는다. 지원 범위와 CPU/GPU 검증 구분은
`EXPERIMENT_STATUS.md` 맨 위에 기록했다. 기존 세부 문서를 최신화하며 새 인계 파일은 늘리지 않는다.

최신 수령·수리(2026-09-09): arxiv full/reference는 학습 후 best validation 0.721098을
저장했으나 전체 분포의 `torch.quantile()` 크기 제한으로 감사가 실패했다. 모델 재학습이
아닌 전체값 분위수 계산과 정확한 소스 해시 기반 감사 재개 수리이며, 원본 학습 결과는
보존한다. 구체적 실패 로그 해석과 검증 범위는 `EXPERIMENT_STATUS.md` 맨 위에 있다.

최신 추가 구현(2026-09-08): `CONDUCTANCE_V5.md` 맨 위에 zero gate×양의 amplitude,
보호 forest와 exact chord budget, 별도 corruption/hard-concrete/negative loss 실험을
추가했다. 새 실행기는 `scripts/run_v5_edge_selection.py`, 실제 구현은
`research/conductance_gat/edge_selection/`이다. 기본 130회 계획이며 기존 V5/Cycle
결과는 보존한다. 신규 GPU 학습 결과가 나온 것은 아니다. 아래 멀티-C 실행기 120회와
혼동하지 않는다. 이 전달 폴더에 새 문서를 늘리지 않고 기존 문서를 업데이트했다.

최신 구현 추가(2026-09-08): `CONDUCTANCE_V5.md` 첫 절에 head별/실제 관계별 C,
행합 1 attention 전파, C 생성기·solver·다항 필터 대조군, 실제 C/alpha/beta 분포
진단과 공통 GPU 배치 실측 실행기를 기록했다. 전체 확장 선택은 12조건×5데이터셋×
2규모×seed0=120회다. 구형 결과는 보존하며, 새로운 구조의 GPU 학습 결과가 나온 것은 아니다.
아래 이전 날짜별 결과와 새 실험 계획을 구분한다.

당시 스냅샷 상태(2026-09-08, 현재는 재생성 완료): `CODE_SUMMARY.md`는 멀티 C 구현 전의 스냅샷이었다. 기존 자동 생성
파일 재생성이 별도 덮어쓰기 승인 요구로 차단되어 갱신하지 않았다. 현재 구현 검토에는
저장소의 `research/conductance_gat/v5/`와 `scripts/run_v5_mechanism_experiments.py`,
해당 tests의 실제 소스를 사용해야 한다. 아래 과거의 스냅샷 갱신 기록과 구분한다.

최신 갱신(2026-09-08): corrected V5의 20조건 학습 결과, 3,832개 에포크 원문과
dynamic-C 10조건 test 결과를 `EXPERIMENT_STATUS.md` 첫 절 및 부록에 기록했다.
reference/arxiv dynamic은 기록상 약 667초에서 29초/epoch로 바뀌었지만, C의 추가 성능
이득은 작거나 음수이고 샘플링 확장의 유효성은 아직 검증되지 않았다. `CONDUCTANCE_V5.md`
첫 절은 C→가중 라플라시안→메시지 패싱의 수학·gradient 검증과 실제 학습 효과의 미검증을
구분한다. 아래 과거 기록의 ‘GPU 결과 미수령’은 해당 날짜의 상태이지 최신 판정이 아니다.
이후 피드백 반영 코드 변경은 `CONDUCTANCE_V5.md` 맨 위 절에 있다. 독립 부분 그래프
배치(`cluster_disjoint`)와 기존 checkpoint의 C 기여·K 참조 검사를 추가했다. 기존 결과와
checkpoint는 보존하고 소스 스냅샷은 갱신했다. GPU 실측·실제 checkpoint 감사·새 학습은
아직 하지 않았다. 이전 절의 결과와 새 sampling 실험을 섞지 않는다.

이 폴더는 **V5만이 아니라 NEW GAT 전체 프로젝트를 외부 GPT에게 검토시키기 위한 전달 묶음**이다.
GPT에는 파일을 따로 고르지 말고 이 폴더의 **10개 파일을 전부** 전달한다.

## 읽는 순서

1. `README_FIRST.md`: 검토 범위와 요청문
2. `HANDOFF.md`: 전체 연구 트랙, 수학, 구현 경계, 위험 요소와 검토 질문
3. `EXPERIMENT_STATUS.md`: 실제 수령 결과, 로컬 검증, 미실행 실험과 주장 가능한 범위
4. `CONDUCTANCE_V2.md`: 고정 그래프의 엣지별 C 직접 학습 계약
5. `CONDUCTANCE_V3.md`: 공유 상대 C graph operator 학습 계약
6. `CONDUCTANCE_V4.md`: C graph operator × spatial W 2×2 실험의 정확한 계약
7. `CONDUCTANCE_V5.md`: C 최적화 계층·가중 라플라시안, 기존 가중치/optimizer/진행분 유지 전환 계약
8. `CYCLE_PE_V2.md`: QR-free DFS 기저의 구조 SE 대 SE+cycle 상대 PE 비교 계약
9. `RICH_SCALING_EXPERIMENTS.md`: Conductance V1–V5, Cycle PE V1/V2, Tree의
   reference/large 전체 scaling 계약(122 child / 126 model trainings)
10. `CODE_SUMMARY.md`: 현재 Python·test·config·script 원문 스냅샷(2026-09-26 재생성)

Conductance v2/v3/v4/v5와 Cycle PE v2는 각각의 원문 문서를 직접 제공한다. 이 문서만 보는
것도 아니며 Conductance v1, Cycle PE v1, Tree Augmentation, 전체 scaling 실험, 데이터·평가
계약과 과거 GPU 결과는 `RICH_SCALING_EXPERIMENTS.md`, `HANDOFF.md`,
`EXPERIMENT_STATUS.md`에서 함께 검토한다.

## GPT에 그대로 줄 요청문

> 첨부한 `gpt_handoff` 폴더의 10개 파일을 모두 읽고 NEW GAT 전체 프로젝트를 교차검증해 줘.
> V5만 검토하지 말고 Conductance v1/v2/v3/v4/v5, Cycle PE v1/v2, Tree Augmentation과 공통
> scaling profile, 데이터·평가·실행 계약을 모두 범위에 포함해라. 문서의 주장과
> `CODE_SUMMARY.md`의 실제 구현이
> 일치하는지 확인하고, 수학 오류·데이터 누수·비교 불공정·재현성·artifact 무결성·검증 누락을
> 심각도순으로 보고해라. 단일 `--device` 순차 실행과 명시적 distinct `--devices` 병렬 실행,
> same-GPU 동시성 계층, GPU/CPU/RAM 실측 필드와 `null+reason` 처리도 코드와 대조해라. Rich
> runner가 V5/Cycle V2의 본 학습 전 실제 optimizer-inclusive batch/worker 후보를 측정하고
> paired arm에 공통 자원 계획을 고정하는지 확인해라.
> 별도 V5 전환 실행기는 완료 fixed 결과를 보존하고 공통 가중치·AdamW state·누적 epoch를
> 유지하며 C만 초기화하는지, 신형 C 실측 인증서와 전환 전후 결과가 정확히 구분되는지도 확인해라.
> 전환 결과를 같은 초기값의 fresh paired 결과로 취급해서는 안 된다. 이전 profiler의
> optimizer/전체 epoch 제외 범위를 이 새 측정과 혼동하지 마라. 실제 수령한 GPU 결과와
> 로컬 CPU fixture, 아직 실행하지 않은 실험을 반드시 구분하고, 현재 근거로 허용되는 주장과
> 금지해야 할 주장을 나눠라.

## 근거 범위

- 최신 corrected run의 20조건은 모두 passed이고, 과거 선택적 전환 run의
  historical_reference 2 / pending_extra_budget 1 / passed 17과 별개다.
  `EXPERIMENT_STATUS.md`에 validation/test 구분, fixed/dynamic 비교, 전체 에포크 원문,
  test 원시 점수·checkpoint SHA, 비용·일반화 문제와 미검증 목록이 있다.
  `CONDUCTANCE_V5.md`에는 현재 width_scaled/K=8 C 목적함수와 검증 범위가 있다.
  핵심 배선·수치 검증 통과를 C의 학습 효과·K=8 수렴·sampling/full-graph 일치 증명으로
  확대하지 않는다. 원본 결과·checkpoint는 보존한다.

- 2026-09-06 V5 기본은 `optimization`/`joint`다. MLP-C는 명시적 비교 옵션이다.
  새 구조의 C 반복 최적화와 task loss 역전파를 검토하고, 과거 MLP checkpoint의 재개 허용을
  새 구조까지 확대하지 않았는지 확인한다. K회 수행은 수렴 보증이 아니다.
- 최신 실행·calibration·재개 명령은 `RICH_SCALING_EXPERIMENTS.md` 첫 절을 따른다.
  A6000 48GB의 약 9GB 사용/100% utilization 화면을 batch 최적화 완료 근거로 쓰지 않는다.
- ad041e2 서버 실행의 V5 집계 실패와 Cycle IPC 실패, 후속 교정 및 기존 결과 격리는
  `EXPERIMENT_STATUS.md`와 `RICH_SCALING_EXPERIMENTS.md`에 함께 기록한다.
  로컬 테스트 통과를 Linux 전체 데이터 전처리나 GPU 전체 학습 성공으로 해석하지 않는다.
- `CODE_SUMMARY.md`는 source/test/config/script를 담지만 데이터와 실제 run artifact는 포함하지 않는다.
- GPU 성능을 독립 재검증하려면 서버의 manifest, checkpoint, history와 결과 파일을 별도로 첨부해야 한다.
- 폴더 밖의 `docs/`는 설치·실행과 개별 과거 실험을 사람이 찾아보는 프로젝트 문서이며 기본 GPT
  전달 대상이 아니다.
