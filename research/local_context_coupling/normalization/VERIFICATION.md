# 구현과 실행 검증 — 2026-10-04

## 완료 범위

| 항목 | 확인 결과 |
| --- | --- |
| 구현 | 20조건의 정규화 연산·2층 분류기·독립 seed 학습·고정 개입·보고서 완료 |
| 정적 검사 | Python AST 검사와 서버 CLI import/help 성공 |
| 단위 테스트 | 총 222개 통과: 모델 113, 학습 38, 계약/입력/평가 잠금 31, 보고서 40 |
| CUDA 검사 | 8개 정규화 조합에서 값·gradient·chunk/checkpoint 일치 검사 포함 |
| 시작 검사 | CPU/CUDA 각각 독립 dense 수식 40조건, DEBUG CE/gradient/Adam 연결 8조건 통과 |
| DEBUG 전체 실행 | 20조건 모두 360회 fixture 학습, 1,080회 독립 모델 갱신 완료 |
| DEBUG 전체 평가 | metric 360행, frozen 1,728행, branch 1,392행과 모든 보고서 출력 완료 |
| 재시작 | 새 폴더에서 완료 checkpoint 재사용, 추가 학습 갱신 0회, 모델 hash 보존 |
| 실제 citation 본학습 | 이번 normalization FULL 실험은 아직 실행하지 않음; 서버에서 실행 |
| 실제 citation 전체 평가 | normalization FULL 결과 없음 |

기존 classification과 원본 operator·입력 reader의 과학 코드 및 설정은 변경하지 않았다.
기존 scientific code digest는 `333d1531b5f047c2f7c075768be1ae93ea44db0cdddf52e14941676094283c28`로 유지됐다.
기존 6조건을 이번 20조건 안에서도 새로 학습하며, 과거 FULL 수치는 새 결과에 섞지 않는다.

## 수식과 실제 학습 연결

- 독립 dense 구현과 sparse S/G의 action·energy·gradient·degree/eigenvalue bound를 대조했다.
- `GR=MG=0`, `Δ=−ρMSGSRZ`, 즉시 merge의 소거, 같은 S의 앞뒤 적용을 검사했다.
- 상수 입력·clique·고립 노드·빈 그래프, disjoint graph batch를 확인했다.
- Triangle-free 그래프에서 local 정규화의 unit/local_degree S가 같은 것을 검사했다.
- 기존 graph/graph macro와 2층 모델의 float64 값·모든 파라미터 gradient를 대조했다.
- 모든 20조건의 실제 CE→gradient→Adam 갱신과 packed seed의 독립 Adam 상태를 검사했다.
- 실제 dropout·chunk·activation checkpoint와 frozen gain0/gain1의 후속 층 재계산을 확인했다.
- 완성/중간 checkpoint의 엄격한 resume, source/graph/policy 변경 거부, test 선택 잠금을 검사했다.
- 보고서의 52개 직접 비교와 12개 상호작용은 각 seed의 차이를 먼저 계산하며 모든 범위를 검증한다.

## DEBUG 전체 실행 기록

실행 환경은 Windows, RTX 5070 Ti 16 GiB, CPU 16개, RAM 약 64 GiB다.
Fixture는 DEBUG-Cora/CiteSeer/PubMed의 전체 노드 24/30/36, 특징 12개, 클래스 3개다.
각 모델은 2층·hidden8·3 epoch이며 FULL 설정 파일과 별도로 관리한다.
Source의 DEBUG 그래프 21개를 모두 검증하고 actual_data=false를 기록했다.

```powershell
.venv-gpu/Scripts/python.exe -B -X utf8 -m research.local_context_coupling.normalization.study `
  --profile debug --device cuda --workers 2 --offline `
  --source-dir results/local-energy-DEBUG-20261004-02 `
  --data-root results/local-context-normalization-data-DEBUG-20261004-01 `
  --output-dir results/local-context-normalization-DEBUG-20261004-01
```

전체 DEBUG 실행은 221.41초였다. 3개 입력의 8개 정규화 geometry를 CPU에서 만드는 시간은
0.007초였고 이후 GPU worker의 정적 cache를 사용했다. Seed pack 1/2와 exact chunk 16/64/전체 엣지를
실측했으며 모든 조건에서 pack2를 선택했다. CPU source 준비는 명시한 worker2를 사용했다.
이 DEBUG 자원 측정은 A6000의 FULL 처리량이나 본학습 시간 추정치가 아니다.

전체 epoch history는 1,080행이다. Learned history 432행 모두에서 실제 교차 CE gradient와
파라미터 갱신이 양수였다. 이는 fixture에서 연결됐다는 검사이며 실제 데이터의 이익을 입증하지 않는다.
Frozen branch의 최대 formula relative error는 `9.2034121e−08`,
적용된 S와 G의 최대 가중 degree는 모두 `0.5`였다.
시작 검사 8회와 disposable calibration의 갱신은 1,080회 학습 예산과 따로 기록했다.

재시작 출력은 `results/local-context-normalization-resume-DEBUG-20261004-01`이다.
29.01초에 같은 coverage를 재평가했고 새 학습 갱신은 0회였다.
모든 원래/재시작 metric 행의 model hash가 같았고 accuracy 차이는 0,
float32 CE 차이의 최대값은 `2.3841858e−07`이었다.
새 폴더를 사용해 원래 결과를 보존했다.

생성된 과학 그림의 label과 CE 눈금도 직접 확인했다.
DEBUG accuracy는 실제 citation 성능으로 제출하지 않는다.

## 서버에서 확인할 것

[RUN.md](RUN.md)의 명령은 전체 노드·엣지·특징·로컬·cross 연결, 2층·hidden64,
500 epoch와 LR/seed 범위를 보존한다. 540회 tuning + 300회 final,
총 840회 학습·420,000회 독립 모델 갱신이다.
실제 할당 GPU에서 pack·chunk 후보와 peak VRAM·처리시간을 다시 측정한다.
Linux 서버 CUDA와 명시적 GPU 할당이 없으면 FULL을 거부한다.

학습 후 `completion.json`과 `LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md`로 결과를 확인한다.
핵심 판단은 같은 내부 정책에서 cross를 더한 분류 차이, 정규화 간 차이,
실제 출력·gain·gradient이며, 정보 복원이나 새 그래프 일반화는 이번 실험의 검증 범위가 아니다.
