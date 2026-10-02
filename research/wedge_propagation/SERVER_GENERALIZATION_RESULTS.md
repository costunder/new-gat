# Experiment 3 서버 전체 평가 결과

2026-10-03 사용자 제공 완료 로그와 `SYNTHETIC_GENERALIZATION_SUMMARY.md`를 기록했다.
완료 폴더는 `results/wedge-feature-20261003-042123`, 학습 source는
`results/wedge-learned-20261002-172132`이다. 첨부 ID는
`43094fe2-6bfc-4a48-9974-f30776c53783`와 `e7640614-a320-4e01-88b7-8e88a447d102`다.
로컬에서 서버 원본 CSV를 직접 읽은 결과로 설명하지 않는다.

## 확인된 실행 범위

- A6000, torch 2.7.1+cu118, 학생 float32, teacher float64.
- 전체 531개 그래프와 그래프별 독립 scalar 특징 16개. 모든 엣지·wedge·random pair 유지.
- 세 목표 × 다섯 조건 × original 및 다섯 fresh 배율.
- 기존 gate 모델 5개 seed 고정, 새 epoch·optimizer update·checkpoint 선택 0.
- 메시지 행 124,254개, scale 행 103,545개. 대응 처리 수이며 독립 그래프 표본 수가 아니다.
- 완료 및 모델·원본 artifact·코드 보존, original metric replay 통과가 보고됐다.

## 새 특징의 경로 목표 상대오차

배율 1, graph macro 후 학습 seed 평균이다.

| Split | Learned | Fixed Q |
| --- | ---: | ---: |
| ID | 0.0475536 | 0.174729 |
| size OOD | 0.0465101 | 0.166116 |
| family OOD | 0.0512217 | 0.203409 |
| family + size OOD | 0.0473821 | 0.179715 |

이는 합성 경로 teacher 메시지의 회수 결과다. 분류 정확도나 범용 연산자 우위를 뜻하지 않는다.
L 목표의 first와 L² 목표의 polynomial 대조는 상대오차 0을 유지했다.

## C 회수와 고정 개입

아래 값은 기존 학습 source의 original ID 결과다. 새 특징의 개입으로 설명하지 않는다.
C 상대오차 0.123585 ± 0.0286, teacher C와 상관 0.947717 ± 0.0231이다.

| 동일한 학습 모델의 C 처리 | 메시지 상대오차 |
| --- | ---: |
| 원래 학습 C | 0.0471402 |
| C=I | 0.383920 |
| 가중치 위치 섞기 | 0.439846 |
| 다른 그래프의 가중치 패턴 | 0.449091 |
| 경로 연결 대응 무작위화 | 0.708137 |

개입에서는 학습 beta를 고정한다. 별도로 beta를 적합한 fixed Q와 다른 조건이다.
평균 C=1 제약은 C의 유일한 복원을 보장하지 않는다.
연결 대응 무작위화는 support도 바꾸므로 경로 연속성만의 효과로 해석하지 않는다.

## 배율 민감도

같은 fresh ID 특징을 4배로 키우면 student C 변화 0.369839,
student 메시지 등변성 오차 0.306899다. Teacher는 각각 6.82005e-7, 1.20792e-7이다.
예측 상대오차는 배율 1에서 0.0475536, 배율 4에서 0.302559이고,
배율 4의 fixed Q 오차는 약 0.174729다.

이 증거에 따라 Experiment 3.1은 C 입력의 graph/field별 physical-edge RMS 정규화를
같은 데이터·학습 규모로 검증한다. 기존 checkpoint에 새 전처리를 끼워 평가하지 않고,
새 normalized gate를 처음부터 학습한다. 모델과 실행은 [별도 패키지](scale_normalization/README.md)에 있다.
