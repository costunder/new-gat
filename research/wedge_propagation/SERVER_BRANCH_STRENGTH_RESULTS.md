# Experiment 4.1 서버 완료 결과

사용자가 제공한 terminal의 `completion.json`과 `BRANCH_STRENGTH_SUMMARY.md`에 근거한다.
첨부 ID는 `975a1691-1a87-4c34-a672-f9aa8cf682ec`이다.
이 기록은 서버 원본 CSV/checkpoint를 로컬로 옮겨 독립 검증한 결과가 아니다.
추가 분석 실행기는 서버에 저장된 CSV의 전체 범위와 실행 기록을 직접 확인한다.

## 완료 범위

- Full, 실제 Cora/CiteSeer/PubMed 전체 그래프와 고정 public split.
- 원래 8조건 × 3데이터 × 5seed = 120개 모델, learned raw/RMS 30개에 개입.
- 원래 metric 360행, 개입 metric 6,750행, 층 진단 4,740행, fixed-Z 1,560행.
- 전체 forward 2,370 seed 경우. 원래 두 층·hidden 64·모든 경로·10개 manifest 유지.
- `completed=true`, `actual_data=true`, 원본 파일·모델 보존, optimizer update 0.
- 전체 102.1296초. 평가 wall에는 진단·metric·fixed-Z가 포함된다.
- PubMed learned raw는 4seed+1seed, RMS는 5seed 동시 처리.
  최대 보고 allocated peak는 약 8.89GiB다. 기준 모델과 같은 순수 forward 비용으로 해석하지 않는다.

## 두 층을 함께 바꾼 관측

| 개입 | 평균 test accuracy 변화 범위 | 관측 |
| --- | ---: | --- |
| C=1, 실제 메시지 norm 일치 | −0.020~+0.120pp | 6개 조건의 paired 95% 구간 모두 0 포함 |
| C shuffle, 실제 메시지 norm 일치 | −0.034~+0.090pp | 6개 조건의 paired 95% 구간 모두 0 포함 |
| True C 유지, κ 분모를 1로 변경 | +0.320~+1.260pp | 6개 조건 모두 평균 CE가 +0.00493~+0.01537 악화 |
| C=1, 분모 1 | +0.400~+1.320pp | 6개 조건 모두 평균 CE 악화 |

Learned C의 메시지 방향은 실제로 바뀐다. Cora RMS의 fixed-Z cosine은
C=1에서 약 0.848/0.889, shuffle에서 약 0.831/0.861이다.
그러나 같은 메시지 norm에서 학습한 C 배치가 test accuracy에 주는 이득은 뚜렷하지 않았다.
구간에 0이 포함된다는 사실을 두 모델의 동등성 증명으로 사용하지 않는다.

κ 분모 제거는 맞히는 노드 수와 정답 log probability 평균에 다른 효과를 줬다.
Accuracy 상승만으로 전체 성능 개선이나 κ가 학습 실패의 원인임을 확정하지 않는다.
CE 악화만으로 확률 보정 실패나 과신의 원인을 확정하지도 않는다.

이차 분기 제거에서 Cora는 raw −0.58pp, RMS −0.30pp였다.
CiteSeer/PubMed의 accuracy 구간은 0을 포함했고 평균 CE는 감소했다.
모든 데이터에서 분기가 유익하거나 분기 기여가 완전히 0이라고 주장할 수 없다.

## 메시지 크기와 다음 확인

Baseline ||βM||/||Z||의 seed 평균은 약 1.45~3.24%였다.
이는 전체 노드·channel의 Frobenius norm 비율이다.
일차 메시지 대비 비율, 개별 노드의 분류 영향, 학습 중 gradient를 직접 나타내지 않는다.

[저장된 결과 분석](branch_analysis/README.md)은 같은 CSV에서 β·κ·정규화 전 메시지 반응을
seed마다 분해하고 αL/Z 및 이차/일차 메시지 비율과 함께 표시한다.
각 층 개별 개입의 accuracy·CE도 요약한다. 모델 forward와 새 학습은 모두 0이다.
