# 구현과 검증 기록

## 검증 범위

이번 코드는 `EDGE_METRIC_REDESIGN.zip`의 A/B/C 제안을 실행하는 새 연구 트랙이다. 기존 wedge, scalar E/J, node-copy 문맥 결합의 코드·checkpoint·결과는 변경하지 않았다. 원문 제안과 현재 구현을 구분해 보존했다.

**로컬에서 수행한 것은 DEBUG 검증이다. 서버 FULL 본학습과 실제 citation 성능 평가는 아직 실행하지 않았다.** DEBUG 결과로 분류 우월성이나 독립 그래프 일반화를 주장하지 않는다.

## 단위 테스트와 독립 CPU/CUDA 검사

`results/edge-metric-tests-DEBUG-20261005-02/unit-tests.xml`과 `unit-tests.txt`에 전체 **232개 테스트 통과, 46.55초**를 기록했다.

| 영역 | 통과한 테스트 |
|---|---:|
| 발생행렬·중복 보정·계량·gradient·방향 불변성 | 70 |
| A의 관측·충돌·잡음 최소제곱·원본 coverage | 50 |
| B의 teacher·규칙 학습·배치·평가·재개 | 49 |
| C의 모델·CE·생성기·checkpoint·배치 | 24 |
| C의 frozen 평가·통계·보고 | 32 |
| A→B→C 순서와 source/coverage 보호 | 7 |

독립 검사 결과는 `results/edge-metric-independent-DEBUG-20261005-01/`에 저장했다.

- CPU와 CUDA에서 각각 96개 수식·RMS·입력 미분·dense 참조·CE 연결 검사가 통과했다.
- CUDA에서 15조건, 2층·hidden64의 독립 classifier fixture를 각각 3 update 실행했다. 모든 활성 parameter가 바뀌었고 no-op 출력과 frozen parameter가 보존됐다.
- A 테스트에는 CPU/CUDA의 잡음·rank-deficient packed 복원을 dense pseudoinverse와 비교한 검사가 있다. 관측 잔차와 normal stationarity 잔차, 짧은 iteration의 미수렴, 영점 RHS를 구분한다.
- 한 채널이라도 미수렴하면 전체 원래 벡터를 성공 복원 요약에서 제외한다. 원문 오류값·두 잔차·미수렴 상태는 그대로 기록한다.

이 검사들은 별도 DEBUG fixture다. 그 optimizer update와 자원 calibration update는 B/C FULL 학습 예산에 포함하지 않는다.

## 실제 학습 경로의 DEBUG 확인

첫 전체 DEBUG 실행 `results/edge-metric-pipeline-DEBUG-20261005-02/`에서 B/C의 전체 선언 조건을 확인했다. A의 잡음 dual CG 문제를 발견해 결과를 보존하고 알고리즘을 수정했다. 현재 A는 zero-start primal CG로 `AᵀA x=Aᵀy`를 풀며, 직접 SVD와 같은 최소제곱 목적을 사용한다.

- B: 30개 DEBUG 그래프와 scalar 실현 120개, 학습 96회, 고정 조건 24회, train-only 적합 24회, seed update 288회. 모든 평가 key와 checkpoint 수가 맞았다.
- C: 15조건, tuning 180회와 final 90회, seed update 810회. 최종 metric 270행, frozen intervention 1,566행, branch 1,224행이 계약과 일치했다.
- C 두 층의 모든 adaptive parameter seed slice 1,728개에 CE에서 온 비영 Adam moment가 있었다. 게이트의 weight decay는 0이다. 모든 final checkpoint를 잠근 뒤 test를 평가했다.
- 45개의 frozen 평가 pack에서 checkpoint hash가 유지됐고 추가 optimizer update는 0이었다. no-op의 810개 metric 행 변화는 0이었다.

상세한 실제 기록은 [A 검증](audit/VERIFICATION.md), [B 검증](synthetic/VERIFICATION.md), [C 검증](classification/VERIFICATION.md)에 있다. B의 DEBUG 그림은 3epoch/2seed이며 로그 표시 하한은 1e−16이다. 원래 CSV 수치는 바꾸지 않았다.

## 수정 후 전체 DEBUG 재실행

`results/edge-metric-pipeline-DEBUG-20261005-03/`에서 A→B→C가 **597.445초**에 완료됐다. 실행 당시 source digest는 `86a98b7601d6dab9f1be575c25307cae9fe529ebd1b34a8e09b26cf4648f1c60`이다. 모든 선언 조건과 입력 coverage를 유지했다.

| 단계 | 실제 실행 기록 |
|---|---|
| A | 21그래프·42 graph/recipe 조합·21 operator view, 302.899초, optimizer 0 |
| B | 30그래프·96 learned run·288 seed update, 14.356초 |
| C | 270 run·810 seed update·270/1,566/1,224 결과 행, 270.893초 |

A의 CG 미수렴과 breakdown은 0건이다. Deep smoothing의 직접 SVD에서 stationarity 기준을 못 맞춘 73건은 수치 정확도·conditioning 문제로 표시했다. 그 채널을 포함한 전체 벡터 4,474행을 성공 요약에서 제외하고 원문에는 보존했다. 성공 요약 2,179,610행을 원문에서 재계산해 일치를 확인했다. 관측 잔차 기준 미충족 5,718건은 잡음·불일치의 별도 진단이며 solver 미수렴과 합치지 않는다.

C의 학습 연결·checkpoint 잠금·frozen hash·no-op을 새 실행에서도 다시 확인했다. 서버 FULL 성능 검증으로 해석하지 않는다.

실행 완료 뒤 `package_review.py`만 보완해 C의 `epoch_history.csv`, inherited adapter의 보조 파일, ZIP 증거 경로 안내를 포함했다. 모델·입력·학습·복원 파일은 실행 당시 바이트와 같다. 실행 당시 패키징 도구도 `results/edge-metric-packaging-source-DEBUG-20261005-01/`에 보존했다. ZIP의 `MANIFEST.json`은 실행 당시와 전달 당시 source identity 및 패키징 파일 하나의 차이를 명시한다. 실행 근거의 해시를 새 해시로 바꿔 적지 않았다.

## 자원과 미검증 범위

로컬 장치는 RTX 5070 Ti 한 개, VRAM 약 17GB, 논리 CPU 16개, RAM 약 68.6GB다. 독립 CUDA 검사는 이 장치에서 수행했고 전체 A/B/C DEBUG는 CPU에서 실행했다. 모든 실행은 별도 새 출력 폴더를 사용했다.

FULL은 Linux 서버의 실제 할당 GPU에서만 실행한다. A6000의 실제 VRAM·CPU·RAM·worker·chunk·physical batch 후보를 측정해 선택한다. 관계 chunking과 seed packing은 모든 원래 노드·엣지·특징·eligible pair와 epoch를 유지한다. 여러 GPU가 할당되면 독립 job을 분배한다.

여러 물리 GPU의 동시 실행과 실제 A6000 FULL 처리량은 아직 측정하지 않았다. A의 citation 복원은 수치 진단이며 exact rank를 주장하지 않는다. B의 제곱 teacher에는 명시된 학생 계량의 크기 제한이 있고, C의 public test는 이전 연구에서 이미 관찰한 탐색적 split이다.

FULL 예산은 A 201그래프·optimizer 0, B 531그래프·학습 240회·120,000 update, C 실제 citation 3개·학습 630회·315,000 update다. 서버 실행과 결과 확인은 [RUN.md](RUN.md)에 정리했다.
