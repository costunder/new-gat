# 원본 자료·인용·코드 찾기

기준일: 2026-10-05. 이 문서는 논문 신규성 판정이나 새 문헌 조사 결과가 아니라, 이번 묶음의 근거를 찾는 지도다.

## 1. 사용자가 제공한 원본 링크

- [초기 연구 대화](https://chatgpt.com/share/6ab95663-72e4-83ee-b434-201a1773977d)
- [wedge 방향 전환 대화](https://chatgpt.com/share/6abe9556-89dc-83e8-b878-d65822319b26)

이번 정리에서 위 공유 URL의 전체 대화 본문을 읽으려 했으나 가져오지 못했다. ZIP에는 이 대화에 붙여 넣은 텍스트, 첨부 원문과 저장소의 보존 문서가 들어 있다. 공유 대화 전체를 포함했다고 주장하지 않는다.

Bracket C 생성의 출처는 Anthony Gruber, Kookjin Lee, Nathaniel Trask의 **Reversible and irreversible bracket-based dynamics for deep graph neural networks**다. 사용자가 지정한 [arXiv 2305.15616v3 원문](https://arxiv.org/html/2305.15616v3)에서 제목·저자·저자 코드 [BracketGraphs](https://github.com/natrask/BracketGraphs)를 확인했다. 이 논문은 attention을 양의 대각 edge 내적과 node 정규화에 연결한다. 이 저장소의 bracket 실험은 대칭 Q/K 점수로 양의 C를 생성하는 부분을 채택했다. 논문의 전체 bracket 동역학이나 모든 에너지 보장을 그대로 구현한 것으로 읽지 않는다. 구체식은 [이론 정리](01_THEORY_AND_MODELS.md)와 실제 [bracket 코드](../../experiments/c_learning_bracket/model.py)를 따른다.

과거 GATv2·GCN·GraphSAGE·iterative-algorithm GNN에 관한 대화는 배경 질문이다. 저장소의 직접 baseline 코드·실행 조건과 구분한다. 연구 아이디어가 선행연구와 정확히 같은지의 답은 별도 원문 대조가 필요하다.

## 2. 실제 코드·이론·실행 근거

| 범위 | 입구와 구현 | 상태·원문 |
| --- | --- | --- |
| 초기 트랙 전체 | [연구 경계](../RESEARCH_OVERVIEW.md), [기존 전달 안내](../../gpt_handoff/README_FIRST.md), [src/chartgat](../../src/chartgat/) | [초기 상태](../../gpt_handoff/EXPERIMENT_STATUS.md), [새 실험 확대 계획](../../gpt_handoff/RICH_SCALING_EXPERIMENTS.md) |
| Conductance V1–V5 | [research/conductance_gat](../../research/conductance_gat/), [수식·요약](../../gpt_handoff/HANDOFF.md) | [factorial](../CONDUCTANCE_FACTORIAL_FINDINGS.md), [C 학습](../CONDUCTANCE_C_LEARNING_FINDINGS.md), [초기 전체 이력](02_EARLY_EXPERIMENTS.md) |
| Cycle PE / Tree | [Cycle PE](../CYCLE_PE.md), [Cycle v2](../../gpt_handoff/CYCLE_PE_V2.md), [Tree](../TREE_AUGMENTATION.md) | [EXPERIMENT_STATUS](../../gpt_handoff/EXPERIMENT_STATUS.md), 추적된 과거 `results/` 산출물 |
| incidence / aggregation | [집계 비교](../AGGREGATION_COMPARISON.md), [내부 ablation](../INCIDENCE_ABLATION.md), [experiments](../../experiments/) | [초기 전체 이력](02_EARLY_EXPERIMENTS.md), 보존된 검증·메모리·실패 로그 |
| CGAT information-flow v2 | [로컬 역사 코드](../../experiments/information_flow_v2/), [최초 수학본](../../gpt_handoff_v2_20260928/01_MODEL_MATH.md) | [역사 문서](../../gpt_handoff_v2_20260928/), 첨부 원문의 설계·검토 |
| C-only / bracket | [C-only](../../experiments/c_learning_only/), [bracket](../../experiments/c_learning_bracket/) | 각 `VERIFICATION.md`, 실제 arxiv calibration·signal audit, 첨부 원문 |
| 출력 초기화 | [output_init_ablation](../../experiments/output_init_ablation/), [README](../../experiments/output_init_ablation/README.md) | 원문의 calibration·abort·recovery와 부분학습, [초기 이력](02_EARLY_EXPERIMENTS.md) |
| Wedge 전 단계 | [wedge_propagation](../../research/wedge_propagation/), [원본 모델 수식](../../research/wedge_propagation/MODEL_MATH.md) | 각 `SERVER_*RESULTS.md`, [전체 wedge 이력](03_WEDGE_EXPERIMENTS.md), 원문 첨부 |
| Local E/J 감사·수신·예측·위치 | [local_energy_relations](../../research/local_energy_relations/), 각각의 `MODEL_MATH.md`·`config_full.json` | [docs/evidence](../evidence/), [전체 local 이력](04_LOCAL_AND_COPY_EXPERIMENTS.md) |
| Copy 고정 연산·분류 | [local_context_coupling](../../research/local_context_coupling/), [분류](../../research/local_context_coupling/classification/) | [고정 결과](../LOCAL_CONTEXT_COUPLING_SERVER_FINDINGS_20261004.md), [분류 결과](../LOCAL_CONTEXT_CLASSIFICATION_SERVER_FINDINGS_20261004.md) |
| 최신 정규화 | [normalization](../../research/local_context_coupling/normalization/), [MODEL_MATH](../../research/local_context_coupling/normalization/MODEL_MATH.md), [FULL 설정](../../research/local_context_coupling/normalization/config_full.json) | 20조건·840회 학습·420,000회 갱신의 서버 FULL 결과 수령. [새 결과 요약](08_NORMALIZATION_FULL_RESULTS.md), [서버 해석](../LOCAL_CONTEXT_NORMALIZATION_SERVER_FINDINGS_20261005.md), [원문](../evidence/local_context_normalization_server_full_20261005.txt), [원문 집계](../evidence/local_context_normalization_summary_20261005.json), [구현 검증](../../research/local_context_coupling/normalization/VERIFICATION.md), [서버 명령](../../research/local_context_coupling/normalization/RUN.md) |

테스트 파일은 [tests/](../../tests/)에 있다. 조건별 실행 계약은 해당 연구의 `EXPERIMENT_DESIGN.md`, `design_contract.json`, `config_full.json`, `RUN.md`를 읽는다. DEBUG 설정을 FULL 설정으로 대체하지 않는다.

## 3. 자료의 원본과 새 요약을 구분

- `repository/`의 파일은 현재 추적 파일의 원본 bytes다. 과거 문서의 상태는 그 문서 작성 시점에 적용된다.
- 새 최상위 00–08 문서는 2026-10-05 기준 종합 정리이며 최신 정규화 FULL 수령 결과를 포함한다. 로컬 본학습을 실행하거나 원문에 없는 새로운 실행 결과를 만들지 않았다.
- `EVIDENCE_INDEX.json`은 이 대화에서 특정한 첨부 원문의 포함/누락, 종류, SHA-256을 기록한다.
- `LOCAL_RUN_INDEX.json`은 보존된 로컬 실행 폴더의 포함/제외 파일과 completion 존재 여부를 기록한다. 이름에 DEBUG가 없어도 폴더명만 보고 실제 데이터 FULL로 분류하지 않는다.
- `MANIFEST.json`은 각 포함 파일의 크기·출처·hash를 기록한다. 서버의 원래 디스크 전체나 과거 모든 Git revision의 manifest가 아니다.
- `PACKAGE_CHECKS.json`의 검사는 ZIP 포함·문법·링크·무결성 검사다. 새로운 본학습·평가 결과로 세지 않는다.

과거 리뷰 ZIP과 종합 문서도 저장소 snapshot에 보존된 범위에서 함께 들어 있다. 최신 상태 판단의 입구는 새 `00_READ_FIRST.md`다.

최신 정규화 증거는 사용자가 전달한 completion·결과 요약 원문과 그 텍스트의 집계다.
서버의 모든 per-seed CSV·checkpoint·hardware/hash 원본을 직접 받은 것으로 읽지 않는다.
기존 여섯 조건의 FULL과 이번 20조건의 FULL은 각각 새로 학습한 별도 결과이며, 로컬 DEBUG는 본학습 성능에 합치지 않는다.
