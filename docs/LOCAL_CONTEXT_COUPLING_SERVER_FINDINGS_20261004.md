# 로컬 문맥 결합 FULL 결과 — 2026-10-04

**제출된 서버 요약은 설계한 교차 전달이 최종 노드 출력까지 남는다는 점을 확인했다.**
즉시 평균하는 대조에서는 교차 효과가 소거됐고, 내부→교차→내부→평균에서는
정확한 차이식에 맞는 변화가 관측됐다. Citation 분류 성능은 이번 실행의 측정 대상이 아니다.

## 제출 자료와 확인 범위

사용자가 메시지로 제출한 `LOCAL_CONTEXT_SUMMARY.md` 내용을
[서버 요약 원문](evidence/local_context_coupling_server_full_20261004.txt)에 UTF-8로 보존했다.
보존 파일의 SHA-256은 `b156f830ee8d349b687ff992959f6d3444282b8135cf327af83c5fbb145632fe`다.
요약은 FULL 201 graph·실제 citation3개·고정 parameter0·optimizer update0을 보고한다.
서버의 결과 경로·completion/source digest·전체 CSV·hardware/resource 파일은 이번 메시지에 없다.
여기서는 제출된 수치와 실제 코드의 수식을 대조했으며, 서버 원시 파일을 독립 재검증한 결과로 표현하지 않는다.

## 1. 예상한 세 현상이 확인됨

| 검사 | 제출된 최대값 | 해석 |
| --- | ---: | --- |
| Sandwich 정확한 차이식의 상대 residual | 6.99393e−17 | 실제 on/off 차이와 직접 계산한 −γη²MAKARH가 일치 |
| Cross 직후 즉시 평균한 on/off 차이 norm | 4.37073e−15 | MK=0에 따른 소거, float64 오차 수준 |
| Joint 반복1/2의 on/off 차이 norm | 4.14771e−15 / 4.53098e−15 | 첫 두 단계에서 예상한 소거 |
| Context / 예측 차이 / 실측 차이 resolved 셀 | 각1,206개 | 전체201×두 C×H0/H1/H2에서 각 지표가 threshold를 넘음 |

1,206은 서로 관련된 graph/C/reference 셀의 개수다. 독립 학습 반복 수나 일반화 성공 횟수가 아니다.
이번 입력에서의 nonzero 관측과 모든 가능한 입력에 관한 주장을 구분한다.
상수·영공간·같은 local 문맥 control의 차이가0인 것은 정상이며 기존 DEBUG 검사에서 다룬다.

## 2. 교차 효과의 크기

분모는 같은 두 내부 pass를 유지한 cross-off 최종 출력의 전체 벡터 norm이다.
Feature별 비율을 평균한 값이 아니라 모든 channel squared norm을 합친 뒤 만든 비율이다.

| C | Citation H0/H1/H2 전체의 상대 변화 범위 | H0 CiteSeer | H0 Cora | H0 PubMed |
| --- | ---: | ---: | ---: | ---: |
| unit | 0.00111344–0.00523755% | 0.00523755% | 0.00339371% | 0.00238411% |
| local_degree | 0.0274956–0.0387885% | 0.0387885% | 0.0293122% | 0.0326407% |

현재 고정 step에서 교차 효과는 수치 오차보다 크지만 off 출력에 비해 작다.
이 크기로 분류 성능의 개선이나 실패를 확정할 수 없다. 분류기의 projection·ReLU와 학습된 hidden 상태는
이번 raw/reference 입력과 다르므로, 학습 중 실제 층별 상대 변화와 task gradient를 측정해야 한다.
local_degree의 더 큰 에너지·차이를 정보가 더 유용하다는 순위로 해석하지 않는다.
C와 그 C에서 정한 내부 step, 출력의 분모까지 함께 바뀌었다.

## 3. Persistent3이 더 작은 이유

각 물리 graph에서 같은 A/K/R/M을 사용하므로 두 변화는 정확히

\[
\Delta_{sandwich}=-\gamma\eta^2 MAKARH,\qquad
\Delta_{persistent,3}=-s^3 MAKARH,
\]
\[
\Delta_{persistent,3}=\frac{s^3}{\gamma\eta^2}\Delta_{sandwich}
\]

관계다. 같은 방향의 mixed action에 서로 다른 계수를 곱한 것이다.
이 비교만으로 서로 다른 정보를 얻었거나 잃었다고 판단하지 않는다.

Unit에서는 최대 내부 degree와 최대 cross degree가 모두 물리 최대 degree d_max다.
중심의 자기 copy가 두 최대값을 동시에 달성하므로 joint 최대 degree는2d_max다.
따라서 η=γ=1/(2d_max), s=1/(4d_max)이고 persistent 변화는 sandwich의 정확히1/8이다.
제출된 citation 세 데이터와 세 reference 상태의 값도 표시 정밀도 안에서 이 관계와 일치한다.

local_degree의 H0 persistent/sandwich norm 비율은 CiteSeer 약0.0003650,
Cora0.0001323, PubMed0.0001292다. 그 비율은 각각 약0.03650%·0.01323%·0.01292%다.
이 차이는 joint bound가 정한 공통 s와 η/γ의 계수 관계로 설명된다.
이를 K의 정보가 본질적으로 약하다는 증거로 사용하지 않는다.

## 4. 다음 비교: 같은 모델의 분류 본학습

권고하는 조건은 cross-off / fixed cross gain1 / learned cross gain 세 가지다.
각 macro layer에서 앞뒤 동일한 두 내부 pass와 동일한 merge를 유지한다.
같은 C 안에서 projection 초기화·dropout·데이터 분할·학습 예산과 validation 선택 규칙을 맞춘다.
기존 citation 학습 계약의2층·hidden64·dropout.5·500epoch·LR/seed 범위를 보존하는 설계가 기준이다.
이 문서는 후속 학습 완료나 새 실행기 구현 완료를 뜻하지 않는다.

Fixed gain1 조건은 추가 gate parameter 없이 이번 연산의 실제 task 기여를 확인한다.
Learned 조건은 현재 API의 seed별 공유 θ와 ρ=sigmoid(θ)를 사용한다.
ρ 초기값은.5이고 범위는0–1이다. 동일한 입력에 대해 gate만 바꿀 때에는
이번 fixed gain1보다 교차 항을 증폭할 수 없다. 학습된 hidden 특징의 상대 변화가
이번 citation raw 특징의 상대 변화 범위 안에 갇힌다는 뜻은 아니다.
한 층에서 입력을 고정했을 때 초기 θ 미분은 full-gain 출력 차이의.25배다.
두 층 공유 θ의 전체 loss gradient는 두 층의 chain rule을 포함하므로 직접 기록한다.

본학습에서는 accuracy·CE와 함께 층별 on/off 변화 norm, gain, CE의 θ gradient,
optimizer의 실제 θ 변화량을 기록한다. 재학습한 on/off 비교와 고정 checkpoint에서 gain을 바꾸는 검사를 구분한다.
효과를 크게 보이게 하려고 현재 step·gain 범위·정규화를 조용히 바꾸지 않는다.

수식과 구현 조건은 [MODEL_MATH.md](../research/local_context_coupling/MODEL_MATH.md)를 따른다.
