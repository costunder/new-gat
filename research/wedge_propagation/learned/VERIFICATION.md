# Experiment 2 구현과 검증 기록

2026년 10월 2일 구현, 10월 3일 상태 갱신. 사용자 첨부 `654482c4-8516-4ff8-a8c3-792dc49ac796`의
teacher·student 생성식, 세 목표와 다섯 비교군을 구현했다.
**A6000 서버 full 학습·평가가 완료됐다.** 제공된 terminal 로그의 결과와 확인 범위는
[SERVER_LEARNED_RESULTS.md](../SERVER_LEARNED_RESULTS.md)에 기록했다.

| 구분 | 실제 확인 상태 |
| --- | --- |
| 코드·수식·실행 문서 | 구현 완료 |
| 정적 검사 | 새 패키지와 다섯 테스트 파일 Ruff 통과 |
| 단위·통합 테스트 | **88개 통과** |
| 실제 CUDA 검사 | 행렬 참조와 출력·gradient 일치, optimizer update 확인 |
| 전체 debug 파이프라인 | **36개 그래프·144개 scalar 입력**, 4 epoch·2 seed, 완료 |
| Debug 평가 범위 | 세 target × 다섯 condition × 다섯 평가 split, **504행** |
| 고정 checkpoint 개입 | 다섯 개입 × 세 target × 두 seed × 24개 평가 그래프, **720행** |
| Teacher 연산 진단 | Debug의 36개 그래프 모두 측정 |
| 본실험 학습·평가 | **서버 full 완료**. 531개 graph·8,496개 scalar 실현, hidden 64·500 epoch·5 seed |
| 서버 결과 확인 범위 | Terminal의 75개 target/condition/split 출력과 `[complete]` 확인. 원본 CSV/NPZ/checkpoint는 아직 로컬 재분석하지 않음 |
| 실제 데이터 학습·분류 | 미실행. 이번 실험의 입력은 명시된 합성 그래프와 특징 |

## 무엇을 검사했는가

- Teacher 수식, 경로 반전 대칭, 양의 C2와 graph/realization별 mean1.
- 다섯 출력식과 사용되는 파라미터 수. Fixed·learned·random-pair에 일차 항이 섞이지 않음.
- Teacher C2와 target을 바꿔도 forward 출력이 바뀌지 않음.
- Dense 참조와 gather/scatter의 출력·입력/파라미터 gradient 일치.
- Gate 파라미터가 optimizer에 포함되고 실제로 업데이트됨. 병렬 seed의 업데이트 독립성.
- 학습 데이터만 사용하는 scalar 최소제곱, seed별 validation checkpoint 선택.
- Optimizer·RNG·best 상태를 복원한 checkpoint 재개와 직접 실행의 일치.
- Full **531개 그래프·8,496개 입력** 계약과 debug 계약의 분리.
- 전체 unordered wedge 보존, split seed·content 중복 검사, disjoint-union 대응.
- Random-pair의 행 norm·방향 처리, 고정 개입의 weight/geometry 구분.
- Teacher 행렬의 span{L,L²,Q} 잔차와 입력에 따른 변화가 dense 참조와 일치.
- Undefined 가중치 진단과 실제 동률·악화 보고, graph macro/seed 통계.
- 기존 결과 보호와 잘못된 graph ID 대응 거부.

88개는 model 35개, data 16개, report 13개, pipeline 8개, diagnostics 16개다.
CPU에서 full 데이터 생성 계약을 확인한 테스트를 full 학습 완료로 세지 않는다.
개입 설정의 누락·오타를 학습 전에 거부하고, 보고서·그림에서 debug 실행이 명시되는 것도 검사한다.

## 로컬 GPU debug 실행

최종 코드의 결과 폴더: `results/wedge-learned-debug-20261002-02`.
앞 실행 `results/wedge-learned-debug-20261002-01`도 그대로 보존했다.
RTX 5070 Ti 16GB, PyTorch 2.13.0+cu130에서 실행했다.
CPU worker 1/2/4/8/16과 graph batch 4/12를 실제로 측정했고, 최종 실행에서 worker 1과 batch 12를 선택했다.
전체 static GPU cache를 포함한 관측 peak는 약 67.9 MB였다.
학습·평가·개입·CSV/NPZ·checkpoint·보고서·PNG/PDF 4종 생성까지 완료했다.
네 그림을 직접 열어 debug·epoch·seed 표시와 범례가 읽히는지 확인했다.

LX target의 first/polynomial, L²X target의 polynomial은 모든 평가 split에서 오차 0이었다.
4 epoch의 learned 결과는 최종 성능이나 연구 성공 근거로 사용하지 않는다.
해당 debug에서 일부 seed와 비교군의 오차가 더 컸으며, 보고서는 이 값을 그대로 기록한다.

## 서버 완료 결과와 후속 확인

서버 폴더는 `/home/aicompetition07/new-gat/results/wedge-learned-20261002-172132`다.
사용자 첨부 `0412f129-7f5b-4c0f-9212-29dfb745cfe4`의 완료 로그에서
[config_full.json](config_full.json)의 500 epoch·hidden 64·5 seed와 모든 출력 평가 조합을 확인했다.
전처리 worker 1과 physical graph batch 240은 서버에서 후보를 실측해 선택했다.

Path target ID의 graph macro 메시지 상대 오차는 fixed 0.171868081,
learned 다섯 seed 평균 0.047140189(표본 SD 0.013490970), random-pair 평균 0.555486887이다.
Learned의 path 출력 오차는 size/family/family+size OOD에서도 낮게 유지됐다.
L target의 first/polynomial과 L² target의 polynomial은 모든 split에서 오차 0이다.
Learned는 L/L² target에서도 항상 우세한 모델이 아니며, synthetic 출력 회수 결과는 실제 분류 성능과 구분한다.

Full 계약의 평가 행은 11,349개, 고정 개입 행은 21,825개, teacher audit은 531개 그래프다.
이 숫자는 계약에서 정해진 coverage다. 현재 제공된 terminal 발췌만으로 원본 CSV의 전체 ID와
C₂ 회수·개입·teacher 진단 값을 직접 재검증했다고 표현하지 않는다.

[Experiment 3](../generalization/README.md)가 서버의 원본 완료 폴더를 읽어
config/source/data/checkpoint hash와 전체 coverage를 검증하고, 고정된 모델의 원본 X 평가를 재현한다.
그 뒤 같은 graph 위 새 scalar 특징과 amplitude 0.25/0.5/1/2/4를 평가한다.
새 학습과 checkpoint 재선택은 수행하지 않는다. 후속 패키지의 테스트·debug·서버 full 평가 여부는
[generalization/VERIFICATION.md](../generalization/VERIFICATION.md)에 기록한다.
개별 C₂의 유일한 회수, 실제 분류 성능과 연구의 새로움은 이 메시지 오차만으로 입증하지 않는다.

실행 명령은 [README.md](README.md), 실제 모델 수식은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.
