# Q/K 점수 역전파 계산 개선

## 변경

`log_row` 경로에서 `checkpoint_edges=True`일 때 대칭 점수의 미분을 직접 누적한다.
기존 경로는 엣지 청크마다 gather의 역전파가 전체 노드 크기의 Q/K gradient를
만들고 합쳤다. 새 경로는 Q와 K의 gradient buffer를 한 번씩 만들고 각 청크의
기여를 `index_add_`로 더한다. 청크 내부 엣지·헤드·특징 계산은 GPU tensor 연산이다.
중간 엣지 특징은 저장하지 않고 backward에서 필요한 Q/K를 다시 읽는다.

점수는 기존과 동일하다. 헤드 차원을 d, 엣지를 (u,v)라 하면

\[
s_{uv}=\frac{Q_u^T K_v+Q_v^T K_u}{2d},\qquad C_{uv}=\exp(s_{uv}).
\]

상위 미분을 g라 할 때 `dQ_u += g K_v/(2d)`,
`dQ_v += g K_u/(2d)`이며 K의 미분도 대칭으로 누적한다.
부동소수점 덧셈 순서가 달라 비트 단위의 미분 일치는 보장하지 않는다.
점수, 정규화, 전파, loss, 초기화, 8층/256차원/8헤드, sampling, epoch 계약은 유지한다.
`checkpoint_edges=False`는 기존 autograd 비교 경로를 유지한다.

## 측정 범위

2026-09-29 로컬 RTX 5070 Ti, PyTorch 2.13.0+cu130, FP32.
실제 로컬 calibration에서 기록한 크기 N=53,248, E=379,398, H=8, d=32,
chunk=16,384를 사용했다. **Q/K와 연결은 합성이며 실제 데이터 학습이 아니다.**
동일 입력에서 각 구현을 warmup한 뒤 순서를 교대하며 6회 측정했다.
각 forward/backward 끝에서 CUDA를 동기화한 wall time의 중앙값이다.

| 점수 생성 연산만 측정 | 기존 청크 checkpoint | gradient 직접 누적 |
| --- | ---: | ---: |
| forward | 13.56 ms | 7.87 ms |
| backward | 51.45 ms | 11.41 ms |
| 합계 | 64.90 ms | 19.49 ms |
| peak allocated (입력 포함) | 496,715,776 B | 391,587,328 B |

출력은 비트 단위로 같았고 Q/K 미분 최대 절대차는 각각 8.94e-8이었다.
측정 코드: `benchmark_scores_debug.py`.
로컬 원본 결과: `results/score-backward-component-debug-20260929-01.json`.
실제 데이터 로딩, Q/K 투영, 나머지 전파, 8층 전체 backward, 관측, 평가 시간은
이 표에 포함되지 않는다. **서버 에폭 240초에 3.33배를 적용할 수 없다.**
서버 MIG에서의 전체 처리량은 새 소스의 기존 calibration으로 측정해야 한다.
실행 중인 서버 checkout을 갱신하면 source contract가 달라지므로 갱신하지 않는다.

## 검증

- FP64 점수, Q/K 미분, gradcheck와 gradgradcheck; 여러 청크 크기와 고립 노드.
- 빈 엣지와 한쪽 입력만 gradient를 요구하는 경우.
- 기존 raw-exp와 log-row의 전체 8층 출력·parameter 미분 비교.
- 4조건 합성 학습·저장·재로딩·평가 smoke 검사. 실제 데이터 200epoch와 구분한다.

관련 3개 테스트 파일에서 41개 통과, Ruff 검사 통과.
검사 기록: `results/score-backward-regression-debug-20260929-01.xml`.
이번 작업에서는 실제 데이터 전체 학습·전체 평가를 실행하지 않았다.

## 원논문 시간과의 관계

[원논문 부록 B.3](https://arxiv.org/html/2305.15616v3#A2.SS3)의 Table 10은
Cora의 깊이별 wall-clock 비교다. 본문·표에는 이것이 학습 1epoch를 측정한 값인지
명시돼 있지 않으며, 우리의 ogbn-arxiv/8층/256차원/샘플링 조건과도 다르다.
Table 7의 Cora는 2,485노드와 5,069엣지다. 원논문의 전체 node/edge ODE 모델과
대칭 Q/K 점수를 채택한 이 실험도 다르다. 논문을 근거로 우리 서버의
240초/epoch가 정상 또는 불가피하다고 판단할 수 없다.
