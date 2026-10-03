# Experiment 4 구현 검증

2026년 10월 3일 요청에 대한 구현 기록. 로컬 산출물에는 실행 시각에 따라 `20261004` 이름이 붙었다.
**서버 실행 코드·수학·연결·DEBUG를 검증했고, 이후 사용자가 서버 full 완료 결과를 제공했다.**
336 run·168,000 새 update·모든 고정 모델 평가 완료는
[서버 결과 기록](../SERVER_CLASSIFICATION_RESULTS.md)에 정리했다.
원본 서버 checkpoint/CSV의 독립 검증은 Experiment 4.1 실행 시 수행한다.

## 구현한 경로

공식 raw 전체 → SHA/split/ID 검사 → 전체 물리 엣지·wedge·고정 정규화 cache →
8조건 공통 분류 모델 → train-mask CE 합산 → backward → seed별 독립 Adam →
validation checkpoint 선택 → 3개 튜닝 seed의 학습률 선택 → 별도 5개 seed 최종 학습 →
모든 최종 checkpoint 잠금 → test/frozen 개입/배율/층 진단 → 전체 범위 검사 → 표·PNG/PDF.

Gate·node MLP·전파 계수는 실제 forward·CE·gradient·optimizer 경로에 연결되어 있다.
학습은 test CE/accuracy를 계산하지 않으며 L2는 선택용 validation CE에 포함하지 않는다.
자동 epoch 축소·early stop·경로 sampling·모델 축소·CPU fallback은 없다.

## 검사

- Ruff 정적 검사 통과.
- 데이터 20개: raw 순서·CiteSeer 누락 ID·공식 mask·전체 wedge·NPZ/hash·경로 이탈 보호.
- 모델 36개: 독립 dense 출력/입력·C gradient, 모든 비교 조건,
  같은 초기화·dropout, packed/독립 Adam, exact chunk/checkpoint, 개입과 zero-RMS.
- 평가·보고서 21개: 모델/manifest 보존, split metrics, 정적 index cache,
  manifest 내부 평균·seed paired 통계, 실제 그림·보고서 파일.
- 실행기 61개: 전체 336/168000 계약, validation 선택, 전체 노드/split/seed/LR coverage,
  2 CPU worker 실제 subprocess 실행, 공유 manifest 보호, 모든 frozen table coverage,
  atomic checkpoint·중단 재개·기존 산출물 보호.

최종 고유 테스트는 총 138개다. DEBUG와 단위 테스트 결과를 최종 분류 성능으로 제출하지 않는다.

## 실제 전체 데이터 전처리

공식 고정 commit의 원본 24개 SHA를 확인해 전체 데이터를 읽었다.
CPU worker 후보 1/2/4/8을 측정했고 실제 입력에 따라 선택한다.
다시 offline cache를 읽을 때 원본과 processed 파일 hash가 유지됨을 확인했다.

| 데이터 | N | 물리 E | 전체 P | train / validation / test |
| --- | ---: | ---: | ---: | --- |
| Cora | 2,708 | 5,278 | 52,301 | 140 / 500 / 1,000 |
| CiteSeer | 3,327 | 4,552 | 26,918 | 120 / 500 / 1,000 |
| PubMed | 19,717 | 44,324 | 699,342 | 60 / 500 / 1,000 |

P 합계는 778,561이다. Citation 전처리 검증은 전체 학습 완료를 뜻하지 않는다.

### 실제 전체 크기의 forward/backward DEBUG

`work/classification-fullsize-preflight-20261003-complete/completion.json`에 결과를 저장했다.
실제 전체 3개 데이터 × 8조건 × 최종 seed 5개를 사용해 24개 packed case,
120개 독립 seed의 train CE/backward를 각각 한 번 검사했다.
두 층·hidden/gate 64·모든 노드/엣지/wedge를 유지했다.

- 모든 parameter tensor의 seed별 gradient가 finite이며 nonzero였다.
- Adam update=0, weight/buffer/source 변경=0. 실제 전체 epoch 학습은 실행하지 않았다.
- Local RTX 5070 Ti에서 최대 allocated peak는 1,748,901,888 bytes(약 1.63GiB)였다.
- PubMed raw/RMS의 한 번 forward+CE+backward는 2.471/2.765초였다.
  Validation·Adam·checkpoint 저장을 포함한 서버 epoch 시간으로 사용하지 않는다.

이 검사는 전체 입력의 수치·메모리·gradient 연결 확인이다. 본학습 완료나 분류 효과의 증거가 아니다.

## GPU 전체 DEBUG 실행

산출물: `results/wedge-classification-debug-20261004-01`.
RTX 5070 Ti 16GB에서 별도 fixture·hidden 8·3 epoch DEBUG 설정을 사용했다.
전체 본학습의 hidden 64·500 epoch 설정을 바꾸지 않았다.

- 3개 DEBUG graph, 모든 8조건, 96 tuning + 48 final = 144 run.
- 432개 독립 Adam update 완료, 새 update도 432개.
- 최종 split metric 144행, frozen 개입 324행, 배율 840행, 두 층 진단 312행.
- 모델 hash·전체 graph hash·코드 hash 보존, frozen update 0개.
- 같은 터미널의 매 epoch·평가 진행 출력과 log 저장 확인.
- `CLASSIFICATION_SUMMARY.md`와 네 PNG/PDF 생성, 실제 그림의 축·범례·DEBUG 표기 확인.
- `completion.json`: `completed=true`, `actual_data=false`, scope `DEBUG_pipeline_only`.

새 폴더 `results/wedge-classification-debug-resume-20261004-01`에서 전체 재개도 완료했다.
원래 결과를 보존하고 새 Adam update=0, 모든 평가 행 coverage를 유지했다.
최종 모델 hash는 동일하고 모든 primary accuracy도 동일했다.
GPU FP32 재평가에서 primary CE의 최대 차이는 1.1921e-7이었다. CSV의 bitwise 동일성을 주장하지 않는다.
완료 전 일부 epoch부터 이어 가는 모델/Adam 일치와 미완성 selected snapshot 처리도 단위 테스트했다.

독립적으로 2개 CPU worker에서 DEBUG 전체 96 tuning run·288 update를 실행했다.
worker 분배·실시간 출력·source/graph 보존을 확인했으며 test 잠금은 유지했다.

## 서버 완료 출력과 남은 확인 범위

- A6000의 전체 run 처리량·peak VRAM 및 할당 GPU 수에 따른 분배.
- 사용자가 제공한 full 완료 출력: 500 epoch × 336 run, 전체 분류·고정 모델 평가 완료.
- 원본 checkpoint/CSV·학습 이력·hash의 독립 검증: Experiment 4.1 source validator가 수행한다.

로컬 CPU subprocess 분배와 한 GPU 계산은 검증했다. 실제 여러 GPU의 동시 실행과 A6000 처리시간은
서버에서 계측한다. 실행 명령은 [README.md](README.md)에 있다.
