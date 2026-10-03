# 수신 집계 실험 구현 검증 — 2026-10-04

## 구현 범위

새 구현은 `research/local_energy_relations/receiver_aggregation/`에 있다.
이전 실험의 과학 코드·설정·결과는 보존했다. 전체 실행은 완료된 이전 서버 결과의
198개 합성 그래프와 실제 Cora·CiteSeer·PubMed 전체 3개 입력·토폴로지를 재사용한다.

실제 forward 관측은 다음 다섯 가지다.

- `tagged`: 송신 로컬을 구분한 receipt.
- `sum`: 수신 로컬·원래 노드별 합.
- `sum_within`: 합과 내부 이차 에너지 E.
- `sum_between`: 합과 로컬 사이 쌍선형 관계 J.
- `sum_both`: 합·E·J.

E/J는 원래 메시지에서 계산한 tensor로 관측에 연결된다. 복원에는 Y만 사용하고,
복원한 H에서 E/J를 다시 계산해 관측값과 비교한다. E/J로 역문제를 개선한 결과로 보고하지 않는다.
E_norm_sq와 J_norm_sq는 복원 대상 E/J 배열의 제곱 norm이다. 내부 에너지의 총합과 다르다.

알려진 양의 C와 공통 전역 H, 모든 이웃 송신과 수신 행이라는 현재 계약에서는
수신 합이 H를 연결성분별 상수까지 결정한다. 송신별 상쇄 크기와 실제 복원오차를 함께 기록한다.
정확한 희소 A/Aᵀ를 cache하고, dense AᵀA를 만들지 않는다.

## 단위 검사

핵심 연산·입력 계약·보고서·runner 통합의 76개 검사가 통과했다. 검사에는 CUDA, 실제 incidence 대응,
정확한 adjoint와 normal diagonal, dense 기준 복원, 원본 checksum·전체 coverage·계약 변경 거부,
보고서의 지표와 E/J 조건 연결이 포함된다.

runner 통합 검사에서는 dense 기준과 실제 compute/collect/CSV를 대조하고,
전체 채널과 채널 분할의 vector trace가 일치하는지 확인했다. 이전 DEBUG 입력 21개를
전부 읽어 CPU에서 450개 원시 입력 지표와 모든 요약 행을 생성하는 경로도 통과했다.

Ruff 정적 검사도 통과했다.

## GPU DEBUG 실행

출력: `results/receiver-aggregation-DEBUG-20261004-01`.
원본: `results/local-energy-DEBUG-20261004-02`의 완료된 입력·토폴로지.

| 항목 | 실제 결과 |
|---|---|
| GPU | RTX 5070 Ti 16GB |
| precision | float64, TF32 끔 |
| 그래프 | DEBUG 21개: 합성 18개, citation 형태 vector fixture 3개 |
| 실제 citation | 0개 |
| C·상태·조건 | 2종 × H0/H1/H2 × 5종 |
| 합성 입력 | 독립 scalar 72개 |
| 원시 입력 지표 | 450행 |
| 요약 | 복원 126행, 조건 630행, 관계 630행 |
| 합성 physical batch | 후보 4/8/16/18 측정, 18 선택 |
| vector fixture 채널 | 각각 원래 12개 전부 처리 |
| GPU 실행 peak allocation | 최대 877,568 bytes; GPU 전체 점유량과 구분 |
| CG 최대 실제 반복 | 37 |
| 최대 실제 Y 상대 잔차 | 6.12197e−9 |
| 최대 q 상대 복원오차 | 6.18696e−9 |
| 최대 성분 평균 제거 H 상대 오차 | 5.87279e−9 |
| 최대 E 상대 재현오차 | 2.11625e−9 |
| 최대 같은 단계 J 상대 재현오차 | 2.40120e−9 |
| 원본·과학 코드 보존 | 실행 전후 checksum 검사 통과 |

모든 DEBUG 그래프·노드·엣지·원래 특징 채널을 처리했다. 두 PNG와 두 PDF를 생성했고,
PNG에서 축·범례·정의되지 않은 비율 표기를 직접 확인했다.

completion에 기록된 계산·준비 시간은 약 11.67초다. 이 값은 서버 FULL 소요시간의 추정치가 아니다.
CPU source workers 1/2/4/8과 sparse operator preparation workers 1/2/4/8을 실제 측정했다.
DEBUG에서는 각각 1이 선택됐으며, 서버에서는 같은 후보를 전체 citation 입력으로 다시 측정한다.

## 완료와 미검증 상태

- 구현·정적 검사·단위 검사·전체 DEBUG 경로: 완료.
- 초기 로컬 구현 검증에서는 서버 FULL을 실행하지 않았다. 이후 사용자가 전달한 서버 결과는
  201개와 실제 citation 3개 완료를 기록한다. [서버 결과](RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md) 참조.
- 분류기·C/W·E/J decoder 학습: 이번 고정 진단에는 없음.
- 예측 개선과 학습 일반화: 이번 진단의 결과로 주장하지 않음.

서버 실행과 결과 확인은 [RUN.md](../research/local_energy_relations/receiver_aggregation/RUN.md),
수학과 판정 범위는 [MODEL_MATH.md](../research/local_energy_relations/receiver_aggregation/MODEL_MATH.md)에 있다.

## 서버 CPU worker 초기화 오류 수정

서버 FULL은 입력 201개를 읽은 뒤 CPU sparse operator calibration의 worker 2개 단계에서
`CUDA error: initialization error`로 실패했다. GPU 연산과 복원 검사는 시작되지 않았다.
CUDA hardware discovery 뒤에 Linux 기본 fork로 자식 프로세스를 만든 것이 실패 지점이다.

`ProcessPoolExecutor`에 명시적인 `multiprocessing.get_context("spawn")`을 전달하고,
준비하는 CSR tensor와 graph index의 device를 CPU로 명시했다. 자식은 새 interpreter에서
시작하며 부모의 CUDA runtime을 물려받지 않는다. CUDA와 fork에 대한 제한은
[서버 버전 PyTorch 2.7 문서](https://docs.pytorch.org/docs/2.7/notes/multiprocessing.html#cuda-in-multiprocessing)에도 있다.
worker 수·그래프·채널·정밀도·solver 계약은 유지했다. CPU 후보 1/2/4/8은 계속 측정한다.

수정 후 관련 core·study 검사 31개와 GPU DEBUG 전체 21개가 통과했다.
DEBUG 출력은 `results/receiver-aggregation-DEBUG-20261004-02`이며 이전 결과를 보존했다.
새 회귀 검사는 context가 spawn인지 확인하고, 실제 CUDA를 초기화한 부모에서 CPU worker
두 개로 CSR을 준비했다. 자식의 CUDA 상태는 준비 전후 모두 초기화되지 않았고,
반환한 operator의 GPU forward는 직렬 준비와 1e−12 허용오차로 일치했다.
이 회귀 검사는 로컬 Windows에서 실행했다. 후속 Linux 서버 FULL 결과는
[서버 결과](RECEIVER_AGGREGATION_SERVER_FINDINGS_20261004.md)에 원문과 함께 기록했다.
