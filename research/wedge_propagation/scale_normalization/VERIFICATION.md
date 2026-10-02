# Experiment 3.1 구현·검증 기록

2026년 10월 3일. C 생성기의 입력 차분만 그래프·scalar 실현별 물리 엣지 RMS로 정규화하고,
raw 원본과 normalized 재학습 모델을 대응해 비교하는 실험이다.

## 검증해야 할 연결

- σ 계산이 전체 물리 엣지와 graph·scalar 축을 유지하는지 확인한다.
- 비영 σ에는 epsilon을 더하지 않고, 영 σ만 1로 처리하는지 확인한다.
- 실제 AX·teacher·목표·원본 그래프와 pair는 유지하는지 확인한다.
- normalized의 파라미터가 메시지 loss·gradient·optimizer update에 연결되는지 확인한다.
- 원본 train batch·구성·seed·epoch를 유지하고 validation만 모델 선택에 사용하는지 확인한다.
- raw checkpoint와 scalar fit을 재사용하며 원본 출력을 재현하는지 확인한다.
- Experiment 3의 새 특징·배율·전체 hash를 재사용하는지 확인한다.
- original/fresh·raw/normalized·3 target·5 condition·6 split의 대응과 전체 범위를 확인한다.
- 학습 뒤 고정한 모델의 스케일 반응·C 진단·다섯 개입을 실제 값으로 평가하는지 확인한다.
- graph macro → seed 통계, shared control의 단일 fit, undefined 값·악화·동률을 확인한다.
- CPU/GPU 자원·batch·처리량을 측정하고 core 연산의 병렬화를 유지하는지 확인한다.
- 원본과 기존 결과를 보존하고 새 폴더에만 쓰는지 확인한다.

## 완료 상태

새 모델·데이터·보고서·실행기의 통합 검사 92개가 통과했다(98.05초).
이후 보고서 보존 검사와 worker provenance 기록을 반영한 보고서·실행기 검사 29개도
통과했다(81.49초). Random-pair 출력·입력 gradient의 독립 dense 참조 검사도 추가 통과했다.
Ruff 정적 검사를 통과했고 CLI 도움말과 서버 실행 인자의 연결을 확인했다.
모델 검사에는 CPU float64 배율 성질과 실제 CUDA FP32 비교, zero 입력의 유한 gradient,
모든 gate 파라미터의 실제 optimizer update, cache 무효화 검사가 포함돼 있다.
데이터 검사는 실제로 완료한 별도 Experiment 2·3 DEBUG 산출물로 원본 읽기와 손상 거부를 확인했다.

### 로컬 CUDA DEBUG 전체 연결

- 결과: `results/wedge-scale-debug-20261003-01`.
- RTX 5070 Ti 16 GiB, torch 2.13.0+cu130, 학생 float32, teacher float64, TF32 off.
- 별도 DEBUG 전체 36 graph·4 scalar 실현·hidden 16·4 epoch·2 seed.
- 세 목표 × 두 normalized gate의 학습, 두 variant의 여섯 시나리오 평가·개입·보고서 완료.
- 메시지 9,072행, scale 7,560행, 개입 4,320행. 완료 시간 34.304초.
- 원본 학습 batch 12, Experiment 3 평가 batch 36을 유지했다. 대안 처리량과 worker를 계측했다.
- original과 다섯 fresh 배율의 raw 수치 재현 검사 전부 통과했다.
  재현 검사는 원 source에서 평가한 다섯 nontrain split을 비교한다.
  train은 별도로 실제 평가하며 전체 coverage에 포함한다.
- source·feature source·고정 모델·코드 보존 검사 통과.
- 평가 단계 최대 할당 VRAM 69,044,224 bytes. 이는 DEBUG 데이터의 값이다.
- normalized learned/path의 모든 DEBUG graph·seed·배율에서 관측한 최대 C 변화는
  2.51930e-7, 최대 메시지 등변성 오차는 2.56654e-7이다. 정확도 개선의 증거로 해석하지 않는다.
- 실제 출력 PNG를 확인했다. 그림에는 DEBUG와 학습 epoch·seed·집계 규약을 표시한다.

### 서버 FULL 결과 기록

**서버 full 학습·평가 완료 결과가 사용자 첨부로 제공됐다.**
첨부 ID는 `da9ba5a2-ee1e-4ea1-9964-09d9c29ceb63`이며,
로컬에서 서버 원본 CSV·checkpoint·contract.json을 직접 읽어 검증한 것은 아니다.
정확한 결과 폴더 이름과 run timestamp는 첨부에서 확인되지 않았다.
Full 계약은 531 graph·16 실현·hidden 64·500 epoch·5 seed다.

- 첨부 보고서의 FULL 표시는 500 epoch·5 seed이고 전체 사용 그래프는 531개다.
- Normalized의 6개 gate job 학습과 raw/normalized 평가·개입 결과가 보고됐다.
- 메시지 248,508행, scale 207,090행, 개입 159,300행이다.
- Raw 추가 optimizer update와 test update는 0이며,
  학습 후 평가 모델·source·feature source·코드 보존은 모두 true로 보고됐다.
- Original metric 재현 `verified=true`가 보고됐다. 수치상 완전 일치로 해석하지 않는다.
- Fresh ID learned/path 평균 오차는 raw 0.0475536, normalized 0.0515799다.
  정규화 후 배율 변화는 줄었지만 평균 회수 오차는 커졌다.

실제 첨부 수치와 해석 범위는
[서버 전체 결과](../SERVER_SCALE_NORMALIZATION_RESULTS.md)에 기록했다.
서버 자원·배치·소요시간의 상세는 첨부 요약만으로 확인하지 못했다.
DEBUG 결과와 서버 FULL 결과는 구분하며, 실제 데이터 분류는 별도 Experiment 4다.

실행 명령과 결과 설명은 [README.md](README.md), 계산과 해석은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.
