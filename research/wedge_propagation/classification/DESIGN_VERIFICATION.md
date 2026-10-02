# Experiment 4 설계 확인

2026년 10월 3일. **분류기 구현·실제 데이터 학습의 완료 기록이 아니다.**

## 문서·계약 확인

- 독립 검토로 공통 forward, static S_Q·κ 정규화의 PSD/norm 상한, C=1 동일성을 확인했다.
- 두 층 gate와 실제 node MLP의 parameter 수를 비교했다. 문서의 세 데이터별 전체 수와 일치했다.
- 216 tuning + 120 final = 336 run, 500 epoch/run = 168,000 독립 optimizer update를 검산했다.
- Tuning/final seed를 분리하고 validation으로만 lr·checkpoint를 고르는 규약을 확인했다.
- 비교 대상의 직접 메시지 범위는 최대 두 홉이지만 support가 항상 동일하지 않다.
  예를 들어 두 노드 component는 Q=0이고 L̄²는 0이 아니다.
- 전체 Cmean이 T_C/κ에서 상쇄되는 점, 전역 RMS·κ의 특징 의존을 구분했다.
- Fixed-strength shuffle/random 진단의 unit-norm bound가 사라질 수 있음을 명시했다.
- 같은 condition/lr의 실제 seed 수에 맞춰 packed 후보를 tuning 1/2/3, final 1/2/4/5로 정했다.
- 배율 평가는 전처리 완료된 X에서 수행하며 다시 row-normalize하지 않는다.

## 독립 CPU float64 수학 DEBUG

별도 로컬 파일 `work/experiment4-design-math-debug-20261003-01.py`로 실행했다.
모든 path를 구성한 path/cycle/star/clique/irregular/isolates/empty 7종 작은 graph 검사다.
Citation 데이터와 classifier 학습을 사용하지 않았다.

| 검사 | 최대 absolute 차이 |
| --- | ---: |
| Topology 공식의 diag(Q) vs 직접 AᵀA | 0 |
| Dense normalized operator vs 독립 gather/scatter | 2.220446049250313e-16 |
| 입력 Z gradient | 1.6653345369377348e-16 |
| C gradient, κ 미분 포함 | 4.440892098500626e-16 |
| RMS gate의 관측 양의 배율 C 변화 | 0 |
| 정규화 branch의 관측 양의 배율 비례성 차이 | 0 |

C=1의 learned/fixed 동일성, L̄/L̄²/Q̄/T̄_C의 PSD·최대고유값≤1,
초기 α=0.5·β=0.25의 P 고유값 범위 [0.25,1], zero RMS의 유한 입력 gradient 검사도 통과했다.
이는 유한한 DEBUG 입력의 수치 확인이며 모든 graph/학습 조건의 경험적 검증을 뜻하지 않는다.
JSON 예산과 실제 nn.Linear의 parameter 수를 함께 확인했다.

## 후속 구현 상태

독립 loader·분류 forward·CE/Adam·calibration·전체 실행기·frozen 평가·보고서를 이후 구현했다.
실제 테스트·DEBUG 상태는 [VERIFICATION.md](VERIFICATION.md)에 별도로 기록한다.
이 문서의 수학 DEBUG는 전체 citation 학습 결과가 아니다. 본학습은 서버 실행 대상으로 유지한다.
