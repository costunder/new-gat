# 로컬 문맥 결합: 첫 fixed audit

각 1홉 로컬 그래프에 같은 물리 노드의 별도 copy를 둡니다.
**내부 계산 → 동일 노드 copy 사이 교환 → 내부 계산 → 균등 merge** 순서로 전파합니다.
처음 copy는 같아 교차 에너지가0이지만, 로컬 내부 계산이 만든 문맥 차이에 교차 연산이 작용합니다.

현재 주 후보는

\[
H_{on}=M(I-\eta A)(I-\gamma K)(I-\eta A)RH,
\qquad H_{off}=M(I-\eta A)^2RH.
\]

대칭 A·PSD K·같은 앞뒤 A·균등 merge 등 조건에서
\(H_{on}-H_{off}=-\eta^2\gamma MAKARH\)이고,
첫 내부 계산 후 KY₁≠0이면 최종 on/off 차이도0이 아닙니다.
Cross-off에도 내부 연산 두 번을 유지합니다.

## 이번에 실행하는 범위

- 기존 완료 audit source의201 graph와8,804 feature 열 전체, H0/H1/H2 세 입력 상태.
- 고정 C=unit/local_degree, canonical copy 연결의 W=1.
- Initial/immediate 소거 control, sandwich의 중간/최종 변화와 정확한 차이식, copy 유지 joint1/2/3 비교.
- Source와 데이터 보존, full coverage, CPU/GPU batch 후보 계측, tensor/batch 계산.
- 작은 DEBUG의 dense/gradient·2층 CE 연결 검사와 서버 FULL fixed audit를 구분합니다.

**FULL 실행은 고정 연산 진단입니다. 분류기 본학습은 실행하지 않습니다.**
실제 2층·hidden64 분류기와 cross 강도 파라미터의 CE 연결은 구현/DEBUG 검사 대상입니다.
본학습 성능을 보고하려면 다음 단계의 별도 실험 계약이 필요합니다.

## 이전 E/J와의 차이

이전 `local_energy_relations/prediction`·`placement`는 scalar 내부 에너지와 divergence 내적을
학습 벡터로 펼쳐 노드 특징에 더했습니다. 새 모델은 copy 공간에서
\(\tfrac12Y^\top(A+\lambda K)Y\)라는 실제 통합 이차형식을 정의하고 그 연산으로 전파합니다.
기존 코드·수치·결과는 보존합니다.

Copy를 사용한다고 선형 표현력의 우위나 정보 복원이 자동으로 보장되지는 않습니다.
고정 연산은 하나의 전역 node 연산자로도 쓸 수 있습니다. 이번에는 그 연산 차이가 실제로 나타나는지부터 확인합니다.

- [전체 수식과 보장 조건](MODEL_MATH.md)
- [서버 실행·결과 확인](RUN.md)
- [검증 기록](VERIFICATION.md)
