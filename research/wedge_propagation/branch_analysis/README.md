# Experiment 4.1 후속 분석 — β·κ·실제 분기 크기

완료된 Experiment 4.1의 CSV를 읽어 **이차 분기가 작은 위치를 수치로 분해한다.**
모든 기존 데이터·seed·층·개입·split의 관측을 사용한다.
모델을 다시 불러오거나 forward·학습·GPU 평가를 실행하지 않는다.
CPU에서 기존 CSV를 한 번씩 읽으며, 결과는 새 폴더에 저장한다.

## 무엇을 확인하나

원래 층의 전파식과 이차 분기 분해는 다음과 같다.

\[
R=\frac13S_Q A^\top C A S_Q Z,
\qquad M=\frac{R}{\kappa},
\qquad U=Z-\alpha\bar LZ-\beta M.
\]

\[
\frac{\|\beta M\|_F}{\|Z\|_F}
=\frac{\beta}{\kappa}\frac{\|R\|_F}{\|Z\|_F}.
\]

- **β/κ**: 학습한 β와 전체 그래프 κ 분모가 만드는 스칼라 크기.
- **R/Z**: 현재 입력 Z에 대한, κ를 적용하기 전 이차 연산자의 반응.
- **βM/Z와 αL̄Z/Z**: 실제 이차·일차 분기가 입력 대비 얼마나 큰지.
- **일차/이차 cosine**: 두 분기가 같은 방향인지, 서로 상쇄되는 방향인지.
- **κ 분포**: 노드별 대각 비율의 평균·분위수·최대·최대 근접 비율.
- **층별 효과**: κ 분모 1과 분기 제거의 layer_0/layer_1/both 개입에서 정확도와 CE 변화.

각 비율은 seed마다 계산한 뒤 평균한다. 평균 β/κ와 평균 R/Z를 곱하면
실제 평균 βM/Z와 달라질 수 있다. `unscaled_branch_to_input`은 **β를 곱하기 전 M/Z**이며
이미 κ로 나눈 값이다. `raw_branch_to_input`은 **κ를 적용하기 전 R/Z**다.
분모 norm이 0이면 정의되지 않은 값으로 남기며 0으로 채우지 않는다.
κ 최대 노드 ID는 seed별 CSV에 보존하고 평균하지 않는다.

이 관측만으로 학습 실패의 원인을 확정하지 않는다.
전역 norm은 개별 노드의 영향과 다르고, 첫 층 개입은 다음 층 입력도 바꾼다.
정확도와 CE를 함께 읽고, test를 이미 본 뒤의 진단 결과로 최적 개입을 선택하지 않는다.

## 서버 실행

완료된 `wedge-branch-strength-*` 결과 폴더가 필요하다. 추가 패키지 설치나 GPU 지정은 없다.
원본 결과의 hash·완료 상태·전체 행 범위를 검증하고, 원본 파일을 변경하지 않는다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
STRENGTH_SOURCE=$(ls -dt /home/aicompetition07/new-gat/results/wedge-branch-strength-[0-9]*/ | head -n 1) &&
test -n "$STRENGTH_SOURCE" &&
/home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.wedge_propagation.branch_analysis.study \
  --source-dir "$STRENGTH_SOURCE" \
  --output-dir "results/wedge-branch-analysis-$(date +%Y%m%d-%H%M%S)"
```

별도의 전체 학습 또는 4.1 GPU 평가를 시작하는 명령이 아니다.
Source의 full/debug profile을 보존하며 DEBUG 결과는 실제 citation 성능으로 제출하지 않는다.

## 결과 확인

```bash
STRENGTH_RUN=$(ls -dt /home/aicompetition07/new-gat/results/wedge-branch-analysis-[0-9]*/ | head -n 1) &&
test -n "$STRENGTH_RUN" &&
cat "${STRENGTH_RUN}completion.json" &&
cat "${STRENGTH_RUN}STRENGTH_ANALYSIS_SUMMARY.md"
```

| 파일 | 내용 |
| --- | --- |
| `STRENGTH_ANALYSIS_SUMMARY.md` | β·κ·연산 반응·실제 분기 크기, κ 분포, 층별 정확도/CE 변화 |
| `baseline_strength.csv` | Learned raw/RMS의 모든 seed·층별 원래 수치와 분해된 인자 |
| `strength_estimates.csv` | 각 인자의 seed 평균·표본 std·95% t 구간·undefined 수 |
| `layer_changes.csv` | 모든 기존 개입 × 세 위치 × 모든 split의 paired 정확도/CE 변화 |
| `fixed_estimates.csv` | 원래 Z를 고정한 모든 C 배치의 크기·방향·변화량 |
| `coverage.json`, `contract.json`, `source.json` | 전체 범위와 기존 파일 보존, 분석 소스 hash |
| `completion.json` | CSV 분석 완료, 새 학습·forward·optimizer update 0회 |

Shuffle manifest는 seed 안에서 평균하고 독립 학습 반복 수로 세지 않는다.
95% 구간은 기존 고정 split에서 초기화 seed의 변동만 나타내며
새 그래프·새 split의 불확실성이나 다중 비교 보정을 포함하지 않는다.

테스트·실제 DEBUG 분석·서버 미실행 범위는 [VERIFICATION.md](VERIFICATION.md)에 있다.
