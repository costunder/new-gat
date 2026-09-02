# GPT 전달용 전체 프로젝트 묶음

이 폴더는 **V4만이 아니라 NEW GAT 전체 프로젝트를 외부 GPT에게 검토시키기 위한 전달 묶음**이다.
GPT에는 파일을 따로 고르지 말고 이 폴더의 **9개 파일을 전부** 전달한다.

## 읽는 순서

1. `README_FIRST.md`: 검토 범위와 요청문
2. `HANDOFF.md`: 전체 연구 트랙, 수학, 구현 경계, 위험 요소와 검토 질문
3. `EXPERIMENT_STATUS.md`: 실제 수령 결과, 로컬 검증, 미실행 실험과 주장 가능한 범위
4. `CONDUCTANCE_V2.md`: 고정 그래프의 엣지별 C 직접 학습 계약
5. `CONDUCTANCE_V3.md`: 공유 상대 C graph operator 학습 계약
6. `CONDUCTANCE_V4.md`: C graph operator × spatial W 2×2 실험의 정확한 계약
7. `CYCLE_PE_V2.md`: 좌영공간 기저벡터 전체를 입력하는 Cycle PE v2 계약
8. `RICH_SCALING_EXPERIMENTS.md`: V1을 포함한 전체 트랙의 큰·깊은 모델 scaling 계약
9. `CODE_SUMMARY.md`: 현재 Python·test·config·script 전체의 원문 스냅샷

Conductance v2/v3/v4와 Cycle PE v2는 각각의 원문 문서를 직접 제공한다. 이 네 문서만 보는
것도 아니며 Conductance v1, Cycle PE v1, Tree Augmentation, 전체 scaling 실험, 데이터·평가
계약과 과거 GPU 결과는 `RICH_SCALING_EXPERIMENTS.md`, `HANDOFF.md`,
`EXPERIMENT_STATUS.md`에서 함께 검토한다.

## GPT에 그대로 줄 요청문

> 첨부한 `gpt_handoff` 폴더의 9개 파일을 모두 읽고 NEW GAT 전체 프로젝트를 교차검증해 줘.
> V4만 검토하지 말고 Conductance v1/v2/v3/v4, Cycle PE v1/v2, Tree Augmentation과 공통
> scaling profile, 데이터·평가·실행 계약을 모두 범위에 포함해라. 문서의 주장과
> `CODE_SUMMARY.md`의 실제 구현이
> 일치하는지 확인하고, 수학 오류·데이터 누수·비교 불공정·재현성·artifact 무결성·검증 누락을
> 심각도순으로 보고해라. 실제 수령한 GPU 결과와 로컬 CPU fixture, 아직 실행하지 않은 실험을
> 반드시 구분하고, 현재 근거로 허용되는 주장과 금지해야 할 주장을 나눠라.

## 근거 범위

- `CODE_SUMMARY.md`는 source/test/config/script를 담지만 데이터와 실제 run artifact는 포함하지 않는다.
- GPU 성능을 독립 재검증하려면 서버의 manifest, checkpoint, history와 결과 파일을 별도로 첨부해야 한다.
- 폴더 밖의 `docs/`는 설치·실행과 개별 과거 실험을 사람이 찾아보는 프로젝트 문서이며 기본 GPT
  전달 대상이 아니다.
