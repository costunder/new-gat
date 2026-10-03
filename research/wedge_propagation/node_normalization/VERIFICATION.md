# Experiment 4.2 구현과 검증 범위

## 구현 완료

- 기존 Experiment 4/4.1의 Python·JSON은 변경하지 않았다.
- Fixed, global/raw·RMS, node/raw·RMS 다섯 조건을 독립 패키지로 구현했다.
- Node 메시지는 `S_C Aᵀ C A S_C Z / 3`이며 C·D_C·양쪽 S_C의 미분을 유지한다.
- 실제 train CE → backward → Adam → validation 선택 → final 평가에 연결된다.
- 전체 설정은 3개 citation 그래프, 두 층, hidden 64, 각 500 epoch,
  기존 LR 후보·tuning/final seed로 210 runs·105,000 updates다.
- 실측 seed packing/path chunk, 전체 경로 처리, resident graph cache, 할당 GPU별 job 분배,
  같은 터미널 진행 출력, immutable checkpoint와 새 폴더 재개를 유지한다.
- 모든 final checkpoint가 validation으로 확정된 뒤 test와 frozen 개입을 평가한다.

## 정적 검사와 테스트

Ruff 검사 통과. 새 테스트 **206개 통과**.

| 대상 | 수 | 확인한 내용 |
| --- | ---: | --- |
| 모델 | 26 | float32/64 dense 수식·전체 CE 미분, D_C 미분, 고정 C 상한, 고립 노드·빈 경로, C1 동일성 |
| 계약 | 20 | 기존 데이터·모델·seed·학습량 보존, 축소·test 누출·숨은 cap 거부, dependency hash |
| 학습/실행 | 69 | validation 선택, 전부 완료 뒤 test 잠금, 실제 2CPU worker, immutable resume, CUDA 실행 계약 |
| 평가/보고서 | 91 | 4 learned 모델·3개 층 위치·5개 개입의 dense 참조, current Z, norm match, coverage와 paired 통계 |

Packed 모델의 CE/Adam/dropout은 seed별 독립 실행과 비교했다.
기존 fixed/global 모델은 새 wrapper와 원래 클래스의 출력·미분이 동일하다.
Chunk와 checkpoint의 forward/gradient는 모든 경로를 한 번에 계산한 참조와 비교했다.
Checkpoint를 새 폴더에 재개하는 실제 DEBUG 학습 검사에서는 완료 epoch를 다시 갱신하지 않았다.

## 최종 CUDA DEBUG 전체 흐름

`results/wedge-node-normalization-DEBUG-20261003-02`에서 완료했다.

- 별도 fixture: N=24/30/36, F=12, K=3, hidden 8, 각 3 epoch.
- Tuning 60 + final 30 = 90 runs, 270 optimizer updates.
- Primary metric 90행, frozen metric 1,512행, layer diagnostic 1,068행.
- Paired 비교 144행, 모든 seed·split·층·manifest 포함.
- 완료 시간 약 80.47초. 같은 터미널·`terminal.log`에 진행을 출력했다.
- `actual_data=false`, frozen model update 0, source/graph 보존 확인.

이는 별도 DEBUG pipeline 검사다. Citation 분류 성능이나 서버 본학습 결과가 아니다.

## 전체 실제 데이터의 무갱신 검사

`results/wedge-node-fullscope-preflight-DEBUG-20261003-01`에서 실제 데이터의
forward/backward를 검사했다. Optimizer는 생성하지 않았고 update는 0이다.

| 그래프 | N | E | 전체 wedge P |
| --- | ---: | ---: | ---: |
| Cora | 2,708 | 5,278 | 52,301 |
| CiteSeer | 3,327 | 4,552 | 26,918 |
| PubMed | 19,717 | 44,324 | 699,342 |

- 양쪽 새 node/raw·RMS, 두 층·hidden 64, final seed 5개를 함께 계산했다.
- 모든 파라미터에서 모든 seed의 유한하고 0이 아닌 aggregate CE gradient를 확인했다.
- 원본 모델 state hash가 forward/backward 전후 같다.
- 검사 모델의 소스 hash가 최종 모델과 같다(`model_source_verified.json`).
- GPU는 RTX 5070 Ti 16GB, PyTorch 2.13.0+cu130, float32, TF32 off였다.
- 전체 경로를 chunk 16,384로 처리했으며 그래프와 특징을 줄이지 않았다.
- 그래프 한 개와 5 seed의 gradient 검사 최대 VRAM은 약 1.63GiB였다.

이 VRAM은 Adam state·장기 학습·A6000 실행의 측정값을 대신하지 않는다.
서버 실행기는 실제 allocation과 batch/chunk 후보의 처리량·peak VRAM을 다시 측정한다.

## 서버 본학습과 평가

**사용자가 제공한 출력에서 Experiment 4.2 서버 210-run 학습·전체 평가 완료를 확인했다.**
전체 105,000 updates, primary 225행, 개입 12,420행, 층별 진단 8,430행,
실제 데이터·전체 coverage·source/graph 보존이 보고됐으며 약 1시간 56분이 걸렸다.
결과와 해석 범위는 [서버 기록](../SERVER_NODE_NORMALIZATION_RESULTS.md)에 있다.
원본 서버 CSV·checkpoint를 로컬에서 재평가한 것은 아니다.
서버 실행 명령은 [README](README.md), 수식은 [MODEL_MATH](MODEL_MATH.md)에 있다.
`completion.json`과 `NODE_NORMALIZATION_SUMMARY.md`에서 완료 범위·정확도·CE·C 변동·실제 분기 크기와 paired 차이를 확인했다.

고정 C의 PSD/norm 상한은 수학적·수치적으로 확인했다.
서버 결과에서 node 정규화는 global 대비 성능을 개선했지만 학습한 C의 고정 C=1 대비 이득은 입증되지 않았다.
추가 제공된 둘째 층 norm-matched C=1 교체 결과에서도 가중치 배치의 이득은 확인되지 않았다.
CE는 네 조건에서 음의 paired 구간, 정확도는 두 조건에서 양의 구간을 보였다.
이 검사는 원래 C의 norm과 첫 층 C를 유지하므로 C 전체 제거의 검증은 아니다.
전체 비선형 신경망의 안정성을 증명한 것은 아니다.
기존 public test 관측 뒤의 후속 비교이며 독립적인 새 그래프 일반화 검증은 아니다.
