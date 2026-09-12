# GPT 전달용 전체 프로젝트 묶음

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

스냅샷 상태: `CODE_SUMMARY.md`는 이번 멀티 C 구현 전의 스냅샷이다. 기존 자동 생성
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
10. `CODE_SUMMARY.md`: 멀티 C 확장 이전 Python·test·config·script 원문 스냅샷(위 주의 참고)

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
