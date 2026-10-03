# 구현과 검증 상태 — 2026-10-04

## 구현한 범위

- 모든 중심의 완전한 induced 1홉 로컬 집합과 원래 node/edge ID 대응.
- 두 고정 양의 대각 C, 내부 이차 에너지, 공통 엣지/공통 노드/서로 다른 incident 엣지의 쌍선형 관계.
- 실제 weighted flow의 Euclidean cycle 투영과 별도 known-C 전체 집계 복원.
- 공통 노드로 d를 복사할 때 남는/빠지는 부분 및 retained/boundary/omitted 엣지의 정확한 norm 분할.
- 중복된 로컬 에너지 합과 물리 평균-C 에너지로의 보정.
- 입력·대응 snapshot/checksum, 전체 coverage 검사, 진행 출력, 자원 측정, CSV·요약·PNG/PDF.

이번 단계에는 trainable parameter, optimizer, encoder, 분류기 또는 복원 loss가 없다.
핵심 경로는 입력 → 정확한 연산 → 측정/항등식/복원 검사 → 보고서다.
복사한 d를 H_next로 전달하는 학습 모델은 아직 구현한 범위에 포함하지 않는다.

## 검사

다섯 테스트 파일의 총 **117개 검사 통과**:

- core 29: dense 식·직사각형 K·orientation/relabel·disjoint batch·빈 그래프·실패·CUDA 비교.
- data 37: FULL 입력 예산·엄격 설정·전체 DEBUG21 snapshot·원본 보존·변조 검출.
- report 31: 전체 macro coverage·에너지/관계 항등식·scalar/vector 구분·raw 보존·PNG/PDF·JSON.
- study 10: 완전한 DEBUG21 실행, 원래 center/pair ID, 특징 chunk 동등성, 출력 보존·source guard.
- calibration 10: 후보 OOM/실측 peak 예산/cache 정리/전체 후보 실패/수치 오류 전파.

107개 core/data/report/study 통합 실행과 10개 calibration 실행을 각각 확인했다.
OOM/VRAM 단위 검사의 자원값은 모의 조건이다. 실제 FULL 자원 측정 결과로 사용하지 않는다.
새 source와 테스트의 Ruff 검사도 통과했다.

별도 로컬 RTX 5070 Ti에서 DEBUG21 전체 GPU 경로를 실행했다.
입력·source·topology 보존, raw row coverage, 보고서/그림/completion을 확인했다.
그림 3종도 실제로 열어 축·범례·표시를 확인했다.
DEBUG fixture는 실제 Cora/CiteSeer/PubMed 성능 결과가 아니다.

최종 로컬 DEBUG 기록: `results/local-energy-DEBUG-20261004-02/completion.json`.
raw records는 local=3,348, relation=46,620, transfer=9,324, temporal=2,232이며
전체 계약에서 계산한 기대 개수와 일치했다.
테스트한 source content digest는
`5384fb1496b92b32cd780e804efc181ea5aaf204df73230d895ccc8bcf6cf1af`다.

## FULL 상태와 제한

서버 FULL201 실행은 아직 하지 않았다. [RUN.md](RUN.md)의 명령으로 서버에서 실행한다.
모든 원래 특징/노드/엣지를 사용하며 runtime chunk는 계산만 분할한다.
단위 테스트는 서버 PyTorch 2.7.1/A6000의 실제 전체 데이터 실행을 대신하지 않는다.
학습된 C/W의 효과, 새로운 분류 성능, 부분 관측 복원, 임의 층의 좌표 정렬,
일반적인 W 또는 전역 PSD 메시지 패싱의 효과는 이 결과로 주장하지 않는다.
