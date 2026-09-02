# Cycle PE v2 — 좌영공간 기저벡터 입력

발생행렬 `B`의 좌영공간 `ker(B.T)`을 이루는 **기저벡터 전체**를 모델에 넣는 버전이다.
기존 통계형 `cycle_set`(v1)은 수정하거나 대체하지 않는다. 다른 연구 트랙과 결합하지 않으며,
외부 비교 모델도 실행하지 않는다.

이 소스 버전에는 v2 구현·단위 검사가 포함되어 있다. 이전 진단 전용 commit `ebf8cd1`에는
없었다. 기존 `cycle_set` 5-seed 결과는 v1 결과이고 v2의 GPU 학습 결과는 아직 없다.
게시·실험 범위는 [실험 상태](EXPERIMENT_STATUS.md)에 기록한다.

## 실행

저장소 최상위 폴더에서 실행한다. [전체 인수인계](HANDOFF.md)의 환경 계약에 따라 GPU 의존성
설치를 끝내고 해당 Conda 환경을 활성화해야 한다. Ubuntu 18.04/glibc 2.27에서는
전용 `new-gat-legacy` 환경과 `setup_gpu.sh --profile legacy-cu118`을 사용한다.
환경 생성·활성화만으로 연구 패키지 설치가 완료되는 것은 아니다.

먼저 공식 데이터와 v2 기저 캐시를 준비한다. 이미 받은 공식 원본은 재사용하며, v1의
6개 통계 캐시는 읽지 않는다. 기저 계산은 CPU 전처리이며 모델 학습은 하지 않는다.

```bash
bash research/cycle_pe/v2/prepare_data.sh
```

준비가 성공한 후 v2만 학습·평가한다.

```bash
bash research/cycle_pe/v2/reproduce.sh
```

기본값은 ZINC-12K와 Peptides-struct, CUDA float32, model seed `0`, batch size `32`,
workers `4`, 최대 300 epochs, validation patience `50`이다. 여러 seed를 반복하려면
`--model-seeds 0,1,2,3,4`처럼 명시한다. 기본 명령도 축소 데이터가 아니라 전체 공식
데이터에서 seed 0을 학습한다.

공통 옵션 `--run-id`, `--data-root`, `--results-root`, `--batch-size`, `--workers`, `--fail-fast`를
사용할 수 있다. 준비와 학습의 data root는 같아야 한다. 기존 run을 덮어쓰거나 자동 재개하지 않는다.
`--cycle-epochs`와 `--cycle-learning-rate`는 v2의 epoch 수와 Adam learning rate를 변경한다.
기본 root `scripts/reproduce.sh`와 기존 `research/cycle_pe/reproduce.sh`는 여전히 v1을 실행한다.
v2 wrapper는 다른 track/version 선택 옵션을 전달해도 Cycle PE v2 benchmark만 선택한다.

## 데이터와 평가

| 데이터 | 공식 train / validation / test | 예측 대상·지표 |
|---|---|---|
| ZINC-12K | 10,000 / 1,000 / 1,000 | supplied graph target, MAE |
| Peptides-struct | 10,873 / 2,331 / 2,331 | supplied 11 graph targets, MAE |

원본 atom/bond 범주형 입력과 target을 그대로 사용한다. target 재정규화, split 재추출,
3D target의 입력 사용은 하지 않는다. validation으로 checkpoint를 선택한 후 test를 한 번 평가한다.
v1과 데이터·backbone·기본 학습 조건을 공유하지만 encoder와 파라미터 수는 다르므로 결과에 기록한다.

## 실제로 전달하는 기저

노드 `n`개, 무방향 엣지 `m`개, 연결 성분 `c`개인 그래프에서 다음을 사용한다.

\[
B\in\mathbb R^{m\times n},\quad
\beta=m-n+c,\quad
U_c\in\mathbb R^{m\times\beta},\quad
B^\top U_c=0,\quad U_c^\top U_c=I_\beta.
\]

엣지 방향은 노드 index가 작은 쪽에서 큰 쪽으로 고정하고, 엣지를 정렬할 때 bond feature도
같이 이동한다. float64 full SVD `B = U S V.T`에서 `U[:, rank(B):]`를 취하고 float32로 저장한다.
각 **열은 좌영공간의 기저벡터**, 각 **행은 해당 엣지의 기저 좌표**다. 원핫 ID가 아니다.

- 모든 `beta`개 열을 저장·입력한다. 고정된 상위 k개 선택, train 최대 폭에 의한 자르기,
  6개 수작업 통계, dense cycle projector로 대체하지 않는다.
- 서로 다른 그래프의 기저 좌표를 섞지 않는다. 배치 안에서도 `[m_i, beta_i]` 행렬을
  각각 보존하고 그래프별로 인코딩한다.
- tree/forest는 `[m, 0]`, edgeless graph는 `[0, 0]` 기저와 0 PE를 사용한다.
- 전처리와 캐시 로딩에서 전체 열 수, `B.T @ U_c`, `U_c.T @ U_c`, 유한값과 엣지 정렬을 검사한다.

## 가변 기저를 받는 학습 인코더

열 개수가 그래프마다 다르므로 동일한 학습 함수를 모든 기저 열에 적용한다. 기저 열 `u_k`와
엣지 bond embedding `h_e`에 대해 먼저 `phi([h_e, u_k[e]])`를 계산하고 엣지 방향으로 평균하여
**그 기저벡터 전체의 문맥**을 만든다. 이어 `psi([h_e, u_k[e], context_k])`를 계산한다.

`u_k`와 `-u_k`를 각각 비선형 인코딩한 결과를 평균하고, 모든 열의 출력을 합쳐
`sqrt(beta)`로 나눈 뒤 PE MLP에 넣는다. 입력을 먼저 절댓값이나 통계로 바꾸지 않는다.
최종 학습 PE를 bond embedding과 결합해 기존 Cycle PE의 edge-aware message layer로 전달한다.

기저 **입력**은 topology-only지만, 학습 인코더는 bond feature에도 조건화된다.
최종 고정 차원 PE는 학습된 압축 표현이며, 기저의 손실 없는 복원이나 단사성을 보장하지 않는다.
`--column-chunk-size`(직접 benchmark CLI, 기본 16)는 임시 열 처리 크기이며 기저 개수 제한이 아니다.
역전파는 모든 열의 연산 그래프를 유지하므로 chunking이 전체 학습 메모리를 일정하게 만들지는 않는다.

기본 `--basis-execution batched`는 여러 그래프의 엣지×기저 열 연산을 묶는다.
각 `(graph, column)`의 context는 서로 분리하며 모든 signed 계수를 보존한다.
`--basis-pair-budget 32768`은 MLP 호출당 pair 수이며 기저 열 개수 제한이 아니다.
`--basis-execution reference`로 같은 파라미터를 쓰는 기존 그래프별 경로를 선택할 수 있다.
선택적 컴파일 및 실제 train 데이터의 속도 비교 범위는 [실험 상태](EXPERIMENT_STATUS.md)를 따른다.

## 보장하지 않는 것

기저 열의 부호 반전과 순서 변경에는 불변이다. 하지만 같은 좌영공간의 다른 직교기저
`U_c Q`에 대한 **임의 회전 불변성은 없다**. SVD의 영특이값 공간은 다차원일 수 있으므로,
노드 재번호화 후 SVD를 다시 계산하거나 수치 라이브러리를 바꾸면 부호·순서 이상의 기저 변화가
생길 수 있다. 재계산 후 graph-isomorphism invariance나 독립적인 엣지 방향 변경 불변성을
주장하지 않는다. 이 한계는 실험 결과 해석 시 별도로 다뤄야 한다.

Dense SVD는 그래프별 CPU 전처리 비용이 있고 큰 그래프용 확장성을 보장하지 않는다.
캐시 signature에 구현 hash와 NumPy 버전, 원본 split fingerprint를 기록한다. 동일 캐시를
보존하는 것이 재현에 중요하다. 서로 다른 기저/코드/환경의 결과를 같은 seed 반복으로 합치지 않는다.
이 버전의 도입이나 단위 검사 통과는 novelty 또는 성능 향상의 입증이 아니다.

## 산출물과 검증 범위

| 위치 | 내용 |
|---|---|
| `data/paper/cycle_pe_v2_benchmark/<dataset>/<signature>/` | 전체 기저와 공식 입력·target 캐시 |
| `research/cycle_pe/v2/results/paper/<run-id>/model-seed-<seed>/benchmark/` | v2 학습 결과 |
| 위 경로의 `<dataset>/cycle_basis_v2/` | `best.pt`, `history.json` |
| `runs/paper/<run-id>/` | manifest, 로그, 환경·데이터 계약 기록, 집계 |

`--results-root`를 지정하면 결과는 `<results-root>/cycle_pe_v2/<run-id>/` 아래에 저장된다.
집계는 `cycle_basis_v2.test`만 성능으로 읽으며, v1 `cycle_set`·validation·외부 논문 수치와
합치지 않는다. 각 버전의 checkpoint와 결과 디렉터리는 독립이다.
기존 일반 `check_datasets.py`의 v1 검사 결과는 v2 기저 검증을 대신하지 않는다.
v2 데이터 준비/로딩 경로 자체가 캐시 checksum·공식 내용·기저 수학 조건을 검증한다.

검사는 작은 수학/배치 fixture를 사용하는 개발 단위 검사다. 실험 CLI에는 가짜 데이터나
CPU 학습 fallback이 없다. 실제 공식 데이터의 전체 GPU 학습 성공 여부는 별도로 확인해야 한다.
구버전 Torch의 체크포인트 보안 제약은 [전체 인수인계](HANDOFF.md)의 환경 절을 따른다.
