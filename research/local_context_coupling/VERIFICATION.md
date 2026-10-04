# 구현과 검증 범위

이 기록은 새 로컬 copy 결합 후보의 검증을 구분하기 위한 문서다.
기존 scalar E/J·wedge·분류 결과를 이 후보의 성과로 재사용하지 않는다.

## 구현 계약

- 모든1홉 induced local graph와 동일 physical-node copy를 유지한다.
- 내부 C는 unit/local_degree 두 고정 조건이다.
- Cross는 인접 중심의 canonical pair마다 모든 shared physical-node copy를 한 번 연결한다.
- M=D⁻¹Rᵀ 균등 merge, 앞뒤 동일 대칭 내부 A, PSD cross K.
- 주 후보는 intra→cross→intra→merge이며 off도 같은 두 intra를 쓴다.
- 고정 η/γ는 graph마다 weighted degree에서 정해 block 전체에 공유한다.
- 두 층 분류기의 cross-on은 seed당 θ_J 하나를 두 층에서 공유하고 CE backward에 연결한다.
- Full study는 기존 source의201 graph·8,804 feature 열·3reference 상태·2C를 쓰는 fixed audit다.

## 추가한 DEBUG 검사

`tests/test_local_context_coupling.py`의 작은 물리 graph는 실제 source 데이터를 대체하지 않는다.
독립 dense 참조는 production geometry 값을 복사하지 않고 원래 물리 엣지에서 직접 만든다.

1. R/A/K/M, copy 개수, canonical cross 연결 수와 독립 dense 일치.
2. 내부+cross 에너지 gradient=(A+λK)Y, PSD와 채널/seed batch.
3. Initial KR=0·cross energy0·즉시 cross gain gradient0, MK=0.
4. Sandwich on/off와 −η²γMAKARH 정확한 차이식.
5. KY₁≠0인 control의 최종 차이, folded RᵀAKAR PSD.
6. Clique/constant/isolate에서 context disagreement0이면 최종 차이0.
7. Path3의 두 endpoint derivative가 둘 다 .005인 control과 비정규5노드의 추가 입력 의존성.
8. Copy를 유지한 joint 반복1/2의 소거와3의 −s³λMAKARH.
9. 두 graph disjoint batch의 graph별 step과 forward/reverse canonical 중복 방지.
10. Positive cross weight와 입력 gradient의 dense 참조.
11. 실제2층·hidden64 forward→CE→backward→Adam에서 cross/projection gradient와 update.
12. Cross-off에 사용하지 않는 θ_J가 없음. CUDA float64의 출력/입력 gradient 참조.

## 실행 결과: 로컬 DEBUG

2026-10-04, `.venv-gpu/Scripts/python.exe -B -X utf8`로 위 단위 검사 파일을 실행했다.
**26 passed, 3.42초**이며 CUDA 사용이 가능해 CUDA 두 조건도 실제로 실행됐다.
작은 DEBUG graph의 대수·dense·gradient·CE 연결을 확인한 결과다.

별도 `verify.run_checks("cpu")` 기록은
`results/local-context-verification-DEBUG-20261004-01/cpu.json`에 저장했다.

- Path/비정규 shared-neighbor/clique/component+isolate/empty × unit/local_degree의10 dense 조건 통과.
- Sparse/dense 최대 absolute 차이3.3306690738754696e−16.
- Initial cross null·mean-cross null·즉시 merge 소거와 정확한 sandwich 차이식 확인.
- 실제2층·hidden64·3개 독립 seed DEBUG 분류기에서 CE backward→Adam1회 update.
- Cross gradient 최대 absolute 값7.512427829929255e−5, 파라미터 변화.009998669049570136.
- DEBUG 특징과 label을 썼으며 dataset 본학습이나 checkpoint 성능 결과가 아니다.

실행기·계약·보고서 검사 `tests/test_local_context_coupling_study.py`도 **32 passed, 7.11초**다.
전체 입력 범위 고정, 기존 snapshot reader 연결, 모든 reference/channel의 dense 참조,
channel chunking 뒤 벡터 집계, 영입력·nonfinite 오류, source hash, worker 설정과 결과 파일을 검사했다.
두 검사 파일의 통과 수는 합계58개다.

## 실제 기존 입력을 읽은 CUDA DEBUG 실행

2026-10-04, 기존 `results/local-energy-DEBUG-20261004-02`를 그대로 읽어
`results/local-context-DEBUG-20261004-01`에 새 결과를 저장했다.
`study --profile debug --device cuda`가 RTX5070Ti에서 완료됐다.
이 입력은 합성18개와 citation 이름의 명시된 DEBUG fixture3개이며 실제 citation 데이터는0개다.

- 전체21 graph·108 feature 열·두 C·H0/H1/H2 포함: channel648행, graph summary126행.
- Source와 numeric/topology snapshot hash의 실행 전후 일치 확인.
- 실제 sequential 차이와 직접 계산한 `−γη²MAKARH`의 최대 상대 residual8.992165852012639e−17.
- 즉시 merge 대조의 차이 norm 최대6.77599954797753e−16.
- Joint 반복1/2 대조의 차이 norm 최대6.323251485722694e−16 / 1.0434444622550422e−15.
- 126 graph/C/state 셀 중 context·예측 차이·실측 차이가 수치상 구별된 셀은 모두114개.
  이 개수의 일치는 이 DEBUG 입력의 결과이며 모든 입력의 성능 개선을 뜻하지 않는다.
- 합성 physical batch 후보4/8/16/18을 계측해18을 선택했다. 모든4개 scalar 열을 함께 계산했다.
  세 citation DEBUG fixture는 모든12개 열을 함께 처리했다.
- 기록된 실행 시간8.1784초. Source 읽기·계측·preflight·보고서 작성 포함이며 Python 시작 시간은 제외한다.
- 별도 CUDA preflight의 실제2층·hidden64·3 seed 분류기에 CE gradient와 Adam1회 update 연결 확인.
  본 fixed audit 자체의 optimizer update는0이고 classifier training은 false다.

완료 상태, raw CSV, 수식 비교 수치, 자원 계측, PNG/PDF 그림을 실제 생성해 확인했다.

GPU별 이름·UUID·실제 할당 VRAM 기록을 보강한 최종 실행기도 별도로 확인했다.
Study 검사32개를 다시 실행해6.74초에 통과했고, 기존 같은 입력을 새
`results/local-context-DEBUG-20261004-02`에 처리해7.8683초에 완료했다.
최종 source digest는 `a069a2e8d8140f359d26dac710640d11e0e8a46230db82eef7bc01a4229849ec`다.
최대 수식 상대 residual9.870444832880725e−17, channel648행·summary126행·resolved114개를 확인했다.
실제 다중 GPU 병렬 실행은 아직 수행하지 않았으며, 로컬 검증은 CUDA GPU1개를 사용했다.
서버 FULL fixed audit의 사용자 제출 요약은 수령했다.
[서버 결과 정리](../../docs/LOCAL_CONTEXT_COUPLING_SERVER_FINDINGS_20261004.md)에 수치·수식 대조와 확인 범위를 기록했다.
Classifier 본학습 결과는 아직 없다.
이번 full fixed audit가 끝나도 분류 본학습 완료나 task 개선을 뜻하지 않는다.

## 아직 입증하지 않은 것

- Citation classification 정확도·CE의 개선.
- Learned C 생성의 이득(본 후보의 C는 고정).
- 원래 엣지/사이클 정보의 완전 복원.
- 모든 node-level 모델 대비 표현력 우위 또는 선행연구 대비 신규성.
- 원본 서버 결과를 가져온 독립 재평가. 이번에는 기존 완료 source를 보존하며 새 fixed 연산을 비교한다.

수식의 nonzero 보장은 [MODEL_MATH.md](MODEL_MATH.md)의 대칭/PSD/균등 merge/양의 step 조건에서만 주장한다.
