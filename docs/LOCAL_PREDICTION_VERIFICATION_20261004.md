# 로컬 E/J 예측 모델 구현·검증 — 2026-10-04

이번 후속은 로컬 이차 에너지 E와 집합 사이 쌍선형 관계 J를 **현재 층의 특징에서
계산해 노드 갱신과 분류 CE에 연결**한다. 기존 고정 audit·수신 역복원 소스와 결과는 보존한다.

## 구현과 본실험 계약

- 새 코드: `research/local_energy_relations/prediction/`.
- 공통 기본 전파: `(1−α)Z+αP_C²Z`. 한 층 최대 2홉, 2층 hidden 64.
- 비교: base / E / J / E+J, 두 고정 C인 unit / local_degree.
- 입력: 이전 local-energy FULL audit의 citation 특징·엣지·전체 induced 1홉 topology.
  공식 raw에서 label·public mask를 읽고 원래 특징·엣지를 정확히 대조한다.
  기존 198개 synthetic 그래프는 provenance로 보존하며 분류 loss에는 쓰지 않는다.
- 학습: 전체 Cora·CiteSeer·PubMed, 500 epoch, 튜닝 216회와 최종 120회,
  총 336회·168,000 독립 optimizer update. Test는 모든 최종 checkpoint 선택을 잠근 뒤 평가한다.
- 활성 E/J lift만 parameter로 만들고, train CE로 projection·α·lift를 학습한다.
  C는 고정하며 전역 역복원을 forward에 넣지 않는다.
- 자원: 모든 할당 GPU에 독립 job을 분배하고, 독립 seed 모델을 앞 축에서 병렬 계산한다.
  CPU 준비·seed packing·정확한 chunk 후보는 실행 자원에서 계측한다.
- 같은 터미널에 phase·epoch·CE·validation·초당 처리·현재 pack ETA·VRAM을 출력한다.
  신규 결과 폴더·hash 검증·동일 계약 resume으로 기존 결과를 보존한다.

수식과 선택 근거는 [MODEL_MATH.md](../research/local_energy_relations/prediction/MODEL_MATH.md),
전체 예산·평가 범위는 [EXPERIMENT_DESIGN.md](../research/local_energy_relations/prediction/EXPERIMENT_DESIGN.md),
서버 실행과 결과 확인은 [RUN.md](../research/local_energy_relations/prediction/RUN.md)에 있다.

## 검증 범위

별도 DEBUG 단위 fixture에서 다음을 검사한다.

- 독립 dense 발생행렬로 E·J·P·P²를 대조하며 두 C·고립 노드·정확한 chunk를 포함한다.
- topology만 쓰는 분모와 특징 배율 제곱, signed J, chunk별 gradient 일치를 확인한다.
- 네 조건의 동일 초기 forward·dropout·projection을 확인한다.
- CE → 실제 모든 활성 parameter gradient → Adam update → 분기 제거 시 출력 변화를 확인한다.
  첫 lift update 이후 E/J 특징을 통한 미분도 확인한다.
- packed seed 학습과 seed별 독립 Adam update를 대조한다.
- 원본 전체 topology·raw identity, 캐시 roundtrip, 데이터·mask 변경 거절을 검사한다.
- validation 선택·완료 coverage·checkpoint hash·resume identity·고정 개입 state 불변을 검사한다.
- 보고서가 누락 조건·seed·split·층·개입·비유한 측정을 거절하는지 확인한다.

2026-10-04 로컬 실행: 새 패키지와 6개 테스트 파일의 Ruff 검사 통과,
단위 검사 **111개 통과**. PyTorch의 sparse CSR beta 알림 1건이 있었으며 실패는 없었다.

```powershell
.venv-gpu\Scripts\python.exe -B -X utf8 -m pytest -q -p no:cacheprovider tests/test_local_prediction_data.py tests/test_local_prediction_model.py tests/test_local_prediction_training.py tests/test_local_prediction_study.py tests/test_local_prediction_evaluation.py tests/test_local_prediction_report.py --basetemp work/local-prediction-all-tests-20261004-01
```

## CUDA DEBUG 통합 실행

로컬 RTX 5070 Ti 16GB, CPU 16개, RAM 64GB에서 별도 DEBUG 설정을 실행했다.
입력은 기존 `results/local-energy-DEBUG-20261004-02`의 명시된 fixture이며 실제 citation 데이터가 아니다.

| 관측 | 실제 결과 |
| --- | ---: |
| 모델 조건 | 8 |
| DEBUG 그래프 | 3개, 24 / 30 / 36 노드 |
| 튜닝 / 최종 run | 96 / 48 |
| Epoch / run | 3 |
| 독립 optimizer update | 432 |
| 모든 split의 기본 metric 행 | 144 |
| 고정 E/J 제거 metric 행 | 540 |
| 층별 branch 진단 행 | 456 |
| 동일 seed 상호작용 통계 행 | 36 |
| 선택된 동시 seed 수 | 2 |
| 최대 측정 VRAM allocation | 67,527,168 bytes |
| 전체 DEBUG 실행 시간 | 109.59초 |

출력은 `results/local-prediction-DEBUG-20261004-01`이다.
완료 계약·입력/소스 보존·모든 조건/seed/split/개입/층 coverage를 확인했다.
E/J 분기 norm이 선택 checkpoint에서 실제로 0보다 커졌고, CE 미분과 업데이트는 단위 검사에서도 확인했다.
PNG/PDF 두 종류와 요약·원시 측정·검증된 checkpoint가 생성됐다.
이 시간과 VRAM은 DEBUG fixture 관측이며 서버 FULL 소요시간의 추정값으로 쓰지 않는다.

같은 완료 결과를 새 `results/local-prediction-resume-DEBUG-20261004-01`에 재개한 실행은
14.52초, **추가 optimizer update 0**으로 완료했다.
선택 모델의 parameter/buffer SHA256과 accuracy가 같았다.
Float32 CUDA 누적 순서에 따른 평가 CE 차이는 최대 1.192093e−7이었다.
기존 결과와 checkpoint를 덮어쓰지 않았다.

통합 실행 뒤 그림의 긴 제목을 두 줄로 바꿨다. 보고서 관련 17개 검사를 다시 통과했고
새 PNG의 제목·legend·축 배치를 확인했다. 모델·학습·평가 수식은 변경하지 않았다.

## 남아 있는 범위

위 구현·로컬 검증을 작성한 시점에는 서버의 실제 citation FULL 학습·평가를 실행하지 않았다.
이후 사용자가 336회 FULL 완료 기록을 전달했다. [서버 결과](LOCAL_PREDICTION_SERVER_FINDINGS_20261004.md)에
원문과 그 자료에서 확인한 범위를 별도로 기록했다.
DEBUG fixture 학습은 코드 연결 검사이며 본실험 정확도 결과가 아니다.
할당 GPU 여러 개를 사용하는 실제 서버 실행은 아직 확인하지 않았다. Job 분배 코드는 단위 검사로 확인했다.
추가 특징과 작은 추가 parameter의 총 기여를 비교한다. 별도 용량 대조는 이번 계약에 없다.
이 public split의 후속 탐색이며 독립 graph 일반화·learned C·복구 불가능한 정보 복원을 입증하지 않는다.
