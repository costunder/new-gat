# v2.0 검증 기록 — 2026-09-28

## 완료한 것

- C-only 모델, 파라미터 없는 fixed C 조건, 독립 실행기 구현.
- 같은 입력의 비용·C·alpha 전후 비교, live C gradient, 생성기 gradient·갱신량 기록.
- 두 조건의 공통 초기값·epoch별 샘플 순서 검사.
- checkpoint 저장·재로딩, frozen validation, C=1 개입과 새 sampler 문맥 평가 연결.
- 실제 데이터 calibration 계약과 production 설정 검증 코드 구현.
- Python 정적 검사와 합성 단위/CUDA smoke 검사.

## 최종 테스트

`results/c-learning-debug-20260928-02.xml`: **12 passed, 실패·skip 0, 19.73초**.
중간 실행의 테스트 수를 합산하지 않았다.

검사 내용:

1. C=1·2·7 → 수신 비중 10%·20%·70%, C=1 대조, sampling correction 유지.
2. 보완 파라미터 부재, fixed 생성기 파라미터·optimizer 항목 부재.
3. 관측 on/off의 loss·gradient·갱신·RNG 동일성: checkpoint on/off 각각 검사.
4. optimizer 갱신이 없을 때 동일 층 입력 재실행의 비용·C·alpha 일치.
5. 실제 생성기 파라미터 방향의 유한차분과 CE 역전파 일치.
6. 실제 엣지 C 한 값을 섭동하고 정규화부터 다시 계산한 유한차분 검사.
7. 8층·256차원·8헤드 forward/CE/backward/optimizer CPU smoke.
8. 축소 production 설정과 보완 옵션 거부.
9. 같은 부분그래프에서 기존 기본 전파와 새 모델의 정확한 출력 일치.
10. CUDA에서 두 조건 학습·샘플 순서 일치·checkpoint·개입·문맥 평가 전체 연결 검사.

CUDA smoke는 합성 128노드, train 96/validation 16/test 16,
2epoch, 모델 8/256/8을 사용했다. 네 epoch 기록과 checkpoint는 모두 `debug=true`다.
test split은 사용하지 않았다. 이 작은 그래프에서는 일부 context가 전체 그래프에
도달했으며 sampler가 경고했다. 따라서 이 검사는 새 부분구조 일반화의 성능 증거가 아니다.

최종 출력 경고 4개: PyTorch JIT deprecation 2개, 위 context 포화 경고 2개.
앞선 debug-01 실행의 import 출력에는 Windows native access-violation traceback이
있었으나 프로세스는 exit code 0과 12 passed로 끝났다. 최종 debug-02 재실행에서는
그 native 예외 출력 없이 위 경고만 발생했다. 환경이나 시스템 설정을 변경하지 않았다.

## 자원 확인

- GPU: RTX 5070 Ti, 16,303 MiB, 한 장. 작업 시작 시 사용 메모리 6,660 MiB.
- CPU: 논리 16개, 사용 가능한 affinity 0–15.
- RAM: 전체 68,640,653,312 bytes, 확인 시 사용 가능 약 41.15 GB.
- 합성 CUDA smoke 기록의 peak allocated: 281,207,296 bytes.
- 같은 smoke의 peak reserved: 327,155,712 bytes.

이는 작은 합성 입력의 수치다. 실제 arxiv batch 크기를 정할 근거로 사용하지 않는다.
다른 실행 중인 작업이나 서버·터미널 세션을 종료하지 않았다.

## 아직 하지 않은 것

- 새 v2.0 경로의 실제 데이터 physical batch/worker calibration.
- 실제 데이터의 200epoch 두 조건 본학습.
- 학습된 실제 checkpoint의 전체 validation·C=1 개입·새 문맥 평가.
- C의 유용성, GATv2 대비 속도, 독립 그래프 일반화 확인.

기존 v2의 calibration을 새 v2.0 calibration으로 대체 사용하지 않는다.
기존 생성기의 수식·초기화·학습률을 바꾸는 수정은 하지 않았다.
실행 후 학습이 막히는 위치가 확인되면 그 위치를 근거와 함께 수정한다.

## 보존

기존 tracked 소스 462개와 이전 `information_flow_v2` 후보, 기존 전달 ZIP을 보존한다.
새 패키지의 `PACKAGE_CHECKS.json`은 원본 manifest 및 이전 전달본과의 실제 byte/hash
대조 결과를 기록한다. 새 결과는 별도 디렉터리에만 쓴다.
