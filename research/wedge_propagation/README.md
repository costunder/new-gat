# 독립 경로 이차 차분 실험

현재 구현은 **Experiment 0 대수 검사와 Experiment 1 고정 연산 비교**다.
`C2=I`로 고정하고 같은 입력에 `LX`, `L²X`, `QX`를 적용한다.
이후 경로 가중치 학습과 분류 실험은 [현재 계획](EXPERIMENT_PLAN_FIXED_FIRST.md)에 정리되어 있다.

## 무엇을 계산하는가

중심 노드 j의 서로 다른 이웃 i,k를 unordered pair로 한 번씩 센다.
각 경로의 차분은 `x_i − 2x_j + x_k`다. 삼각형 안의 경로도 포함한다.

\[
A_{p,:}=e_i^\top-2e_j^\top+e_k^\top,\qquad
L=B^\top B,\qquad Q=A^\top A.
\]

\[
X^\top QX=\|AX\|^2,\qquad
QX=A^\top(AX),\qquad
Q=L^2+B^\top\operatorname{diag}(d_u+d_v-4)B.
\]

코드는 모든 경로를 사용하는 gather/scatter 구현, 엣지만 사용하는 정확한 항등식 구현,
독립 행렬 참조의 결과를 비교한다. 실제 측정은 동일 크기의 여러 그래프와 모든 특징을 함께 batch 처리한다.
행렬 고유값을 직접 계산해 norm과 nullspace 차원을 측정한다.
Dense 행렬은 이 작은 그래프의 참조·스펙트럼 검사에 사용하며, 큰 그래프용 learned 모델의 구현은 후속 단계다.

주요 질문은 **Q가 L과 L²의 조합으로 설명되는 범위와 남는 작용의 차이**다.
Cycle에서는 Q=L²이고, star도 L과 L²의 조합으로 환원된다.
정확한 판정식과 주의점은 [실제 계산 설명](MODEL_MATH.md)에 있다.

## 본실험 데이터

| 항목 | full profile |
| --- | --- |
| 노드 수 | 20, 30, 40, 60, 80, 100 |
| 그래프 종류 | cycle, star, grid, ER, tree, tree+chord |
| 고정 구조 | 크기마다 종류별 1개 |
| 확률 구조 | 크기마다 종류별 10개 |
| 합계 | 198개 그래프 |
| 입력 | 그래프마다 독립 표준정규 scalar 특징 16개, 총 3,168개 |
| 경로 | 모든 unordered wedge, sampling 없음 |
| 수치 정밀도 | float64, TF32 사용 안 함 |
| 학습 | optimizer 없음, trainable parameter 0개 |

ER의 고립 노드와 비연결 그래프도 그대로 포함한다.
그래프와 특징의 난수 stream을 분리하고 seed·content hash·원본 입력을 저장한다.
`debug`는 별도 테스트 profile로 18개 그래프·72개 입력을 사용한다. `full`이 기본값이다.

## 서버 실행 — A6000

기존 `new-gat` CUDA 환경에서 실행한다. 아래 명령은 할당받은 GPU만 사용한다.
`WEDGE_PYTHON`은 기존 서버 환경의 경로이며, 환경 위치가 다른 서버에서는 그 경로를 바꾼다.

```bash
cd ~/new-gat &&
git pull --ff-only &&
WEDGE_PYTHON=/home/aicompetition07/.conda/envs/new-gat/bin/python &&
"$WEDGE_PYTHON" -m pip install -r research/wedge_propagation/requirements.txt &&
read -r -p "할당받은 A6000 GPU 번호 또는 UUID: " WEDGE_GPU &&
test -n "$WEDGE_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$WEDGE_GPU" \
  "$WEDGE_PYTHON" -u -m research.wedge_propagation.study \
  --profile full --device cuda \
  --output-dir "results/wedge-fixed-$(date +%Y%m%d-%H%M%S)"
```

GPU 모델과 실제 free VRAM, CPU affinity/quota, RAM, 저장 공간을 먼저 출력한다.
CPU preprocessing worker 후보와 physical graph batch 후보를 측정해 처리량이 가장 좋은 값을 선택한다.
그래프 크기는 유지하고, 모든 경로와 입력을 처리한다. 이 작은 고정 연산 검사에서는 46GB 전체를 채울 필요가 없다.
각 크기의 최대 33개 그래프를 동시에 처리하며 추가 그래프나 모델을 임의로 생성하지 않는다.

실행한 터미널에 `[algebra]`, `[cpu calibration]`, `[batch calibration]`, `[fixed]`, `[complete]`가 표시된다.
같은 내용은 `terminal.log`에 저장된다. `--output-dir`은 새 폴더여야 한다.
오류가 나면 traceback과 `failure.json`을 남기고 예외를 반환한다.
같은 결과 폴더를 다시 사용하면 덮어쓰기를 거부한다.

## 결과 파일

- `SUMMARY.md`: 측정 결과와 해석 범위.
- `operators.csv`: 그래프별 항등식 오차, 다항식 잔차, norm, nullspace 차원.
- `actions.csv`: 3,168개 입력별 raw/정규화 출력 norm과 에너지, Q–L² 차이.
- `action_values.npz`: 각 입력의 실제 LX/L²X/QX와 spectral norm으로 나눈 출력.
- `spectra.npz`: 모든 그래프의 L/L²/Q 전체 고유값.
- `dataset.npz`, `data_manifest.json`: 모든 엣지·경로·특징·degree·edge degree sum, seed와 hash.
- `contract.json`: 실제 설정, 코드 hash와 Git commit, 자원, batch/worker 계측, 처리량.
- `algebra.json`: 7종 debug 대수 검사 결과.
- PNG/PDF 4종: 다항식 잔차, 강도를 맞춘 작용 차이, 스펙트럼, raw 출력과 에너지.
- `completion.json`: 전체 해당 profile 완료 여부와 처리 개수.

정의되지 않는 상대 지표는 CSV 빈 칸/JSON null로 표시하고 absolute 값과 구분한다.
원래 크기의 차이와 정규화 이후 차이를 함께 읽는다.
고정 연산 비교로 학습 성능이나 새로운 그래프에서의 일반화를 판정하지 않는다.

## 개발 검증

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_wedge_fixed_operators.py tests/test_wedge_fixed_data.py \
  tests/test_wedge_fixed_report.py tests/test_wedge_fixed_study.py
python -u -m research.wedge_propagation.study --profile debug --device cuda \
  --output-dir "results/wedge-fixed-debug-$(date +%Y%m%d-%H%M%S)"
```

디버그 검사는 본실험과 별도로 기록한다. 전체 fixed 실험과 후속 학습의 완료 여부는
각 서버 실행의 결과로 판단한다.
