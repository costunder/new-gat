# 구현과 검증 상태

2026년 10월 2일. 이 기록은 로컬 개발 검증이며 서버 본실험 결과가 아니다.

| 구분 | 상태 |
| --- | --- |
| Experiment 0/1 구현 | 완료 |
| 정적 검사 | 새 코드·테스트 Ruff 통과 |
| 단위 테스트 | 63개 통과 |
| CUDA 수치 검사 | Explicit/빠른 항등식/행렬 참조 출력·입력 gradient 일치 |
| 별도 GPU debug 실행 | 18개 그래프·72개 입력, 완료 |
| Full 고정 연산 비교 | 서버에서 실행할 예정, 로컬 미실행 |
| Learned C2와 분류기 | 이번 단계 미구현·미학습 |
| 실제 데이터 학습·평가 | 미실행 |

단위 검사는 독립 NumPy 최소제곱과 다항식 잔차를 비교한다.
Cycle/star 환원, 빈 그래프, 상수 특징과 nullspace, 경로·엣지 방향·노드 재번호,
seed 분리, 전체 데이터 계약 198/3168, 기존 결과 보호도 검사한다.
허용오차는 absolute와 relative를 함께 사용하며, 작은 분모의 상대값을 undefined로 남긴다.

로컬 GPU는 RTX 5070 Ti 16GB이며 서버의 A6000 속도와 최적 batch는 여기서 판정하지 않았다.
Debug 실행은 CPU worker 1/2/4/8과 graph batch 4/8/9를 실제로 측정했다.
그 실행에서는 worker 1과 batch 9가 선택됐으며, 서버 full profile에서 후보를 다시 측정한다.
Debug 입력 수와 경로 수는 본실험 profile의 기본값에 반영되지 않는다.

Debug 결과의 CSV/NPZ, summary와 PNG/PDF 생성까지 확인했다.
이 파일들은 Git에 포함하지 않고, 각 실행의 새 결과 폴더에 보관한다.
첫 debug 실행에서 Windows psutil의 Path 인자 오류를 확인했으며 str로 변환해 수정했다.
실패 폴더에는 traceback과 failure.json을 보존했고 새 폴더에서 재실행을 완료했다.

## 재현 명령

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_wedge_fixed_operators.py tests/test_wedge_fixed_data.py \
  tests/test_wedge_fixed_report.py tests/test_wedge_fixed_study.py
python -u -m research.wedge_propagation.study --profile debug --device cuda \
  --output-dir "results/wedge-fixed-debug-$(date +%Y%m%d-%H%M%S)"
```

본실험 실행 명령과 결과 파일 설명은 [README](README.md)에 있다.
