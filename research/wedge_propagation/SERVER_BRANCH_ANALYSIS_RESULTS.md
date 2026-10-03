# Experiment 4.1 후속 분석 — 서버 결과

사용자가 제공한 서버 완료 출력(`76feaf64-c752-4d8b-bb09-8aaf929648b2`)을 기록한다.

- 실행 코드: `3f8b24f`, 서버 결과: `results/wedge-branch-analysis-20261003-133432`.
- 원본: `results/wedge-branch-strength-20261003-104349`의 full 결과.
- Seed/층 인자 60행, 강도 추정 204행, paired 개입 비교 756행, fixed-Z 추정 384행.
- `completed=true`, 원본 보존, 모델 forward 0, optimizer update 0, 분석 시간 0.904초.

## 다음 실험의 근거

학습된 β 평균은 0.270–0.304다. 전역 κ는 약 4–7이고, 실제 βM/Z는
약 1.45–3.24%다. 대각 비율의 평균은 약 1인데 드문 최대값으로 모든 노드의
이차 메시지를 나누는 구조가 관측됐다.

Frozen checkpoint에서 κ를 1로 바꾸면 두 층 개입의 정확도는 0.32–1.26%p
증가하지만 여섯 조건 모두 CE가 증가한다. 이 결과로 학습 실패의 원인이나
κ 제거의 타당성을 확정할 수 없다.

[Experiment 4.2](node_normalization/README.md)는 연산자 상한을 유지하는
C 의존 노드별 대각 정규화로 재학습한다. 기존 전역 정규화 raw/RMS와 고정 C=1을
같은 모델·데이터·학습량으로 새로 학습해 비교한다.

이 기록은 첨부된 완료 출력에 근거한다. 원본 서버 CSV와 checkpoint는 서버에 있다.
