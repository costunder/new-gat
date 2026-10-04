# E/J 위치별 재학습 구현 검증

## 구현과 전체 실행 계약

새 코드는 [`research/local_energy_relations/placement/`](../research/local_energy_relations/placement/README.md)에 있다.
기존 prediction의 수식·연산·설정·checkpoint·결과를 보존하고, README에 후속 실험 링크만 추가했다.

각 고정 C에서 base 한 개와 E만·J만·E+J × hidden/output/all을 비교한다.
hidden은 첫 층, output은 출력층, all은 두 층이다. 전체 20조건을 새로 학습한다.

FULL은 Cora/CiteSeer/PubMed 전체 public 그래프, sampling ratio 1,
2층·hidden 64·각 run 500 epoch다. 기존 LR 3개, tuning 3seed, final 5seed를 유지한다.
Tuning 540회 + final 300회 = **840회·420,000 update**다.
C는 unit/local_degree로 고정하며, 모든 조건의 기본 전파와 E/J 분모는 같다.

활성 층에만 lift 벡터를 등록하고 해당 층의 특징을 계산한다. 두 backbone 층은 항상 유지한다.
이전 데이터 loader와 geometry를 그대로 사용하는 adapter는 데이터·source·backbone·operators·resources와
run당 학습 설정의 일치를 검사한다. 원래 manifest와 새 placement provenance를 함께 보존한다.

## 완료한 정적·단위·회귀 검사

- 새 Python 모듈 8개와 테스트 7개의 AST 문법 검사 완료.
- FULL config/design 일치와 60개 데이터·조건 cell의 실제 parameter 수 일치 확인.
- **313개 테스트 통과:** 새 위치별 검사 202개 + 기존 prediction 회귀 111개.
- 이후 그림 눈금·축 제목 표시만 수정하고 보고서 검사 **19개를 다시 통과**했다.
- Git whitespace 검사 완료. 이전 prediction의 scientific Python/JSON 파일 변경 없음.

검사는 다음 실제 동작을 확인한다.

- 같은 C·seed의 모든 위치에서 초기 projection·dropout·forward가 동일하다.
- 새 base/all 대조군은 기존 모델과 CPU float64 출력·gradient·3회 Adam 업데이트가 정확히 같다.
- 20조건 모두 등록된 parameter가 CE→gradient→optimizer에 연결된다. 활성 lift가 실제로 바뀐다.
- 업데이트된 lift를 통해 E/J 특징이 입력과 projection의 미분에 기여한다.
- 첫 층·출력층만 활성화한 모델은 나머지 층을 제거한 기존 모델의 실제 비영 lift forward와 같다.
- CPU/CUDA forward·gradient, 독립 seed packing, 정상 종료한 학습의 이어하기가 일치한다.
- Tuning에서 test metric을 금지하고, 모든 final checkpoint를 확정한 뒤 test를 평가한다.
- Frozen 개입은 활성 층에만 적용하며 state hash와 optimizer update 0을 검증한다.
- Source·graph·config·checkpoint 변경, 잘못된 placement·coverage, 기존 output 중복을 거부한다.
- FULL의 일반 진입점과 worker 진입점 모두 Linux 서버·CUDA·명시적 GPU 할당을 요구한다.

## 실제 CUDA DEBUG 전체 실행

완료 폴더: `results/local-placement-DEBUG-20261004-02`.
기존 source: `results/local-energy-DEBUG-20261004-02`의 전체 21개 감사 입력.
학습에는 DEBUG citation fixture 3개만 사용하며 실제 citation 데이터 사용은 **false**다.

RTX 5070 Ti 16GB, CPU 16개, RAM 약 64GiB, Python 3.13.2,
PyTorch 2.13.0+cu130에서 실행했다. DEBUG는 hidden 8·각 run 3 epoch의 별도 계약이다.

| 항목 | 실제 완료 |
| --- | ---: |
| Tuning / final runs | 240 / 120 |
| 전체 optimizer updates | 1,080 |
| 원래 모델 split metric rows | 360 |
| 활성 층 frozen metric rows | 900 |
| 원래·개입의 두 층 branch rows | 840 |
| 전체 소요 | 214.697초 |
| 측정 peak CUDA allocation | 67,529,216 bytes |

CPU 준비 후보 1/2/4/8을 측정했고 이 fixture에서는 1을 선택했다.
GPU 독립 seed pack 1/2와 정확한 chunk 후보를 측정했으며 전체 경로를 처리했다.
작은 fixture의 VRAM·시간을 A6000 본학습의 자원 수치로 해석하지 않는다.

CUDA 실행과 재개 때 기록된 source digest는
`42a92d742d04a336a832f458e0d909b8962d403e1b3abbc51555f9308e6e6c1f`다.
실행 이후 변경은 report.py의 그림 x축 표시뿐이다. 모델·학습·평가·설정은 같다.
같은 DEBUG 측정값의 그림을 별도 `results/local-placement-report-style-DEBUG-20261004-02`에 생성했으며,
최종 표시의 테스트와 세 PNG의 시각 검토를 수행했다. 기존 결과는 덮어쓰지 않았다.

## 기존 폴더를 보존한 실제 재개

`results/local-placement-resume-DEBUG-20261004-01`로 재개해 19.896초에 완료했다.
추가 optimizer updates는 **0**, 원본/재개 final model 60개 pack의 hash는 모두 같다.
원본 selected checkpoint 180개의 checksum도 일치한다.

평가 CE의 최대 차이는 `1.1920928955078125e-07`이다.
CUDA scatter의 수치 변동 범위이며 모델 state 변화와 구분한다.
개별 run의 원본 파일 byte 보존과 이어하기 거부 조건도 단위 검사했다.

## 아직 실행하지 않은 것

이 새 840회 FULL 본학습과 실제 citation 성능 평가는 **서버에서 실행해야 한다**.
로컬 DEBUG 결과를 최종 분류 성능·새 그래프 일반화·learned C·사이클 정보 복원의 증거로 제출하지 않는다.
모델의 위치별 lift 폭은 hidden 64 / output K / all 64+K여서 용량도 함께 변한다.

[서버 실행·결과 확인](../research/local_energy_relations/placement/RUN.md),
[전체 설계](../research/local_energy_relations/placement/EXPERIMENT_DESIGN.md),
[수식과 실제 forward](../research/local_energy_relations/placement/MODEL_MATH.md).
