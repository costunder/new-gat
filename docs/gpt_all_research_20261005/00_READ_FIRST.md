# 전체 연구·실험 GPT 전달본 — 2026-10-05

이 묶음은 **초기 Conductance·Cycle PE·Tree부터 CGAT 정보 흐름, wedge, 로컬 E/J, 현재 copy 문맥 결합과 정규화 비교까지**의 이론·구현·실행 근거를 모았다. 새 실험을 실행하지 않고 현재 보존 자료를 정리했다.

**최신 상태:** 정규화 비교의 서버 FULL 완료 결과를 수령했다. 실제 citation3개·20조건·840학습·420,000갱신, 전체6,820.536초다. 큰 정확도 변화는 내부 graph→local 정책에서 나왔고, 교차의 실제 작용과 일부 CE 기여도 커졌다. 같은 C·내부 정책의 cross−off 정확도 우위는 확인되지 않았다. 새 결과는 [08_NORMALIZATION_FULL_RESULTS.md](08_NORMALIZATION_FULL_RESULTS.md)를 먼저 읽는다.

## 1. 처음 읽을 핵심

연구의 중심은 다음과 같다.

> 각 노드 주변의 로컬 그래프에서 발생행렬과 가중 라플라시안으로 내부 이차 에너지를 만들고, 서로 다른 로컬·홉·층의 연결을 쌍선형 관계로 다룬다. 이 관계가 실제 메시지 전달·집계에 어떻게 작용하고, 정보의 구별과 분류에 도움이 되는지 검증한다. 공유 C 생성 규칙을 서로 다른 부분구조에서 학습·재사용한다는 초기 목적도 이 역사에 포함한다.

현재 모델은 같은 원래 노드를 **로컬별 copy**로 유지한다. 내부 연산 S가 서로 다른 문맥 표현을 만든 뒤, 같은 원래 노드의 copy를 잇는 G가 교차 작용을 하고, 다시 S를 적용한 후 원래 노드별 평균으로 합친다.

\[
\boxed{T_\rho=M(I-S)(I-\rho G)(I-S)R},\qquad
\boxed{T_\rho-T_0=-\rho MSGSR}.
\]

R은 원래 특징을 로컬 copy로 복사하고 M은 copy 평균이다. 같은 원래 노드 copy 연결 때문에 `GR=0`, `MG=0`이다. 따라서 내부 전달 없이 cross를 적용하거나 cross 직후 바로 평균하면 효과가 사라진다. 내부→교차→내부 순서는 이 소거를 피하는 실제 구현이다. 두 macro 층을 거친 CE로 특징 투영과 learned 조건의 교차 강도ρ를 학습한다.

**현재 local/copy의 C는 unit 또는 local_degree로 고정되어 있다.** `learned`는 공유 교차 강도 ρ를 학습한다는 뜻이다. 이를 학습된 연결별 C나 attention으로 설명하면 안 된다. 초기 C 학습, wedge의 C 학습, 현재 copy 분류는 각각의 실제 코드와 실험으로 읽어야 한다.

## 2. 읽는 순서

| 파일 | 내용 |
| --- | --- |
| [08_NORMALIZATION_FULL_RESULTS.md](08_NORMALIZATION_FULL_RESULTS.md) | 새 정규화 FULL 완료, 내부·교차 기여와 실제 작용의 핵심 결과 |
| [01_THEORY_AND_MODELS.md](01_THEORY_AND_MODELS.md) | 모든 이론의 발전 과정, 행렬 차원, 실제 모델 수식, 복원·소거·안정성 조건 |
| [02_EARLY_EXPERIMENTS.md](02_EARLY_EXPERIMENTS.md) | Conductance V1–V5, Cycle PE, Tree, incidence/CGAT, C-only/bracket, 초기화 실험과 실패 |
| [03_WEDGE_EXPERIMENTS.md](03_WEDGE_EXPERIMENTS.md) | 고정 연산부터 teacher C 학습·새 특징·정규화·분류·분기 개입까지 전 단계 |
| [04_LOCAL_AND_COPY_EXPERIMENTS.md](04_LOCAL_AND_COPY_EXPERIMENTS.md) | 로컬 고정 감사, 수신 집계, E/J 예측·위치, copy 고정·분류, 최신 정규화 상태 |
| [05_EVIDENCE_AND_OPEN_QUESTIONS.md](05_EVIDENCE_AND_OPEN_QUESTIONS.md) | 원문과 해석의 증거 수준, hash, 자료 공백, 아직 답하지 못한 질문 |
| [06_GPT_REVIEW_PROMPT.md](06_GPT_REVIEW_PROMPT.md) | GPT에 붙여 넣을 요청문 |
| [07_REFERENCES_AND_SOURCE_MAP.md](07_REFERENCES_AND_SOURCE_MAP.md) | 원본 자료·인용·코드·설정·결과를 찾는 위치 |

`REVIEW_ALL.md`는 위 안내와 여덟 문서를 한 파일로 합친 읽기용 문서다. `GPT_REVIEW_PROMPT.md`는 요청문의 별도 사본이다. ZIP 안의 링크는 ZIP 내부 위치로 연결된다. 저장소의 원본 문서는 `repository/`에 원래 bytes 그대로 남긴다.

## 3. 전체 실험 지도

조건 수와 실행 횟수의 정확한 예산·수치·예외는 각 상세 문서와 원문을 따른다.

| 단계 | 연구 질문·실제 연산 | 실행 근거·현재 상태 |
| --- | --- | --- |
| 초기 Conductance V1 | 공유 C 생성과 weighted Laplacian 전파 | 실제 5-seed test, gate·정규화 factorial, learned/fixed C 및 checkpoint 진단 |
| Conductance V2–V5 | 엣지별 C, 상대 operator, spatial W, 입력별 동적 C 최적화 | 작은 합성 연구·서버 학습·검사·중단·부분 평가를 버전별로 구분 |
| Cycle PE·Tree | cycle 기저/투영 특징, tree chart와 augmentation | 초기 공식 데이터 test, 작은 합성 연구, 이후 구현 검사; 새 scaling 계획은 결과 아님 |
| incidence/CGAT v1 | 기본 메시지, 층간 에너지와 sampling | PPI 8조건·arxiv 학습·메모리/미분 검사, 일부 실패·미완료 |
| 정보 흐름 v2 | 층별 송신·수신 집합, 메시지 제곱·경계·연속층 통계 | 구현·DEBUG, C-only와 bracket 검사, 실제 arxiv calibration |
| 출력 초기화 | 출력 투영과 신호 축소의 원인 분리 | 서버 calibration·abort·복구, 최종 네 조건 학습 결과는 불완전 |
| Wedge 1 | AᵀA와 L/L²의 관계 | FULL 198 graph·3,168 입력, 학습0 |
| Wedge 2 | 공유 wedge C가 합성 teacher 메시지를 배우는가 | FULL 30 seed 모델·15,000 update |
| Wedge 3 | 고정 파라미터로 새 특징·배율에 일반화하는가 | FULL frozen 평가, 학습0 |
| Wedge 3.1 | C 입력 RMS 정규화 | FULL normalized 모델30개·15,000 update |
| Wedge 4 | 실제 citation의 CE 학습·GCN/GATv2 대조 | FULL 336학습·168,000 update |
| Wedge 4.1 | C의 배치와 분기 크기 효과 | FULL frozen 개입과 저장 CSV 분석, 새 학습0 |
| Wedge 4.2 | C 의존 노드별 정규화 | FULL 210학습·105,000 update, 둘째 층 개입12행 |
| Local L1 | 내부 E·관계 J·cycle·경계/누락 | FULL 201 graph 고정 감사, 학습0 |
| Local L2 | 수신 합의 상쇄와 제약된 q/H/E/J 복원 | FULL 같은201 graph 고정 감사, 학습0 |
| Local L3 | scalar E/J를 실제 예측에 추가 | FULL 336학습·168,000 update |
| Local L4 | 첫 층/출력층/두 층의 E/J 위치 | 서버 집계 요약 수령, 840학습·420,000 update 보고; 원시 completion/CSV/checkpoint 미수령 |
| Copy C1 | copy 공간의 내부→교차→내부 전달 | FULL 201 graph 고정 감사 원문, 학습0 |
| Copy C2 | 같은 내부 전달을 둔 cross off/fixed/learned | FULL 252학습·126,000 update 완료 원문 |
| Copy C3 | graph/local S × graph/edge G 정규화 | 서버 FULL840학습·420,000update 완료 원문 수령; 별도 구현222검사·DEBUG360학습/1,080update·resume0update |

표의 update는 독립 seed 모델의 갱신 수다. Seed packing의 optimizer 호출 수, least-squares scalar fit, calibration, frozen 평가와 다르다. 표에 있는 서로 다른 연구의 정확도를 하나의 같은 모델 성능 곡선으로 합치지 않는다.

## 4. 지금의 결과를 짧게 읽으면

1. Wedge C는 지정한 합성 teacher 규칙을 학습했다. 실제 citation에서 learned C가 고정 C보다 유리하다는 결과는 얻지 못했다.
2. 알려진 양의 C와 전체 관측 아래 실제 `q=CBH`는 복원 가능했다. 임의 edge flow의 cycle 소거, 부분 관측·경계 누락과는 조건이 다르다.
3. Scalar E/J는 예측에 실제 연결됐지만 그 두 요약을 더하는 것만으로 넓은 물리 그래프의 에너지 전달을 구현한 것은 아니었다.
4. 이후 copy 모델은 내부와 교차 에너지를 copy 공간의 연산에 직접 사용했다. Fixed 감사는 정확한 차이식을 확인했고 현재 입력에서 교차 작용이 merge 뒤에도 남는 것을 측정했다.
5. 첫 copy 분류의 같은 C 대비 accuracy 차이는 모든 95% 구간에0이 들어갔다. 작은 CE 변화와 ρ 학습은 관측했지만 accuracy 우위나 동등성은 입증하지 못했다.
6. 정규화 FULL은 완료됐다. Unit cross-off의 내부 graph→local 변경은 Cora+21.46pp, CiteSeer+12.74pp, PubMed+4.64pp였다. Cross의 일부 CE 기여와 실제 작용도 커졌지만 추가 정확도 우위는 확인되지 않았다.

## 5. ZIP 구성과 범위

- 최상위: 새 종합 문서00–08, 한 파일 읽기본, GPT 요청문, `MANIFEST.json`, `PACKAGE_CHECKS.json`, `EVIDENCE_INDEX.json`, `LOCAL_RUN_INDEX.json`.
- `repository/`: 현재 Git이 추적하는 코드·설정·테스트·수식·설계·원문 기록·보존 산출물 전체. 과거 상태 문서도 원본으로 포함한다.
- `local_history/`: Git 밖에 보존된 초기 v2/bracket/output-init 코드·수식·검토 자료의 지정된 역사 폴더.
- `evidence/attachments/`: 이 대화에 첨부된 설계·GPT 검토·서버 완료·실패 원문. 자료의 종류를 색인에 표시한다.
- `evidence/conversation/`: 첨부 파일 없이 직접 보낸 수치의 기존 보존 전사본. 전사본이라는 출처를 명시한다.
- `evidence/local_runs/`: 보존된 로컬 개발 실행의 최상위 텍스트 산출물과 하위 job의 학습 이력·완료·자원·로그 등 지정한 근거 파일. DEBUG/calibration을 실제 citation FULL 성능으로 읽지 않는다.

새 ZIP에 또 다른 ZIP이나 환경·다운로드 데이터·Git 내부·pytest 임시 폴더를 재귀로 넣지 않는다. Git 밖 로컬 checkpoint/NPZ/그림 등은 전부 복제하지 않으며 제외 범위는 `LOCAL_RUN_INDEX.json`에 기록한다. 최근 서버의 per-seed CSV·checkpoint가 없다는 공백도 유지한다. 실제 실험의 입력·학습·평가 규모를 바꾸는 작업은 하지 않았다.

과거 정리본의 “통합 에너지 모델 미구현”은 당시 scalar E/J 단계의 상태다. **현재 copy 공간 모델은 구현·고정 감사·첫 분류까지 진행됐다.** 원래 물리 union 그래프의 에너지와 copy 공간 에너지의 동일성, 완전한 정보 복원, 성능 우위는 각기 별도의 질문이다.

현재 코드 전체와 역사 자료를 포함했어도 과거 모든 Git revision이나 접근하지 못한 ChatGPT 공유 대화 전체를 복구한 것은 아니다. 수령·보존 자료가 없는 실행은 목록에서 그 한계를 명시한다.
