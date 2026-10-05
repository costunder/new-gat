# Verification — 2026-10-05

구현 완료. 새 모델 학습/optimizer update 없음. 기존 선택 checkpoint 분석만 수행한다.
원본 classification 코드·설정·checkpoint·데이터는 변경하지 않았다.

## 실제 실행한 검사

- Ruff: 새 package 및 해당 테스트 모두 통과.
- 테스트 **23개 통과**: 독립 dense-incidence CUDA 검산 6개, 기존 selected 모델
  로드/재현 CUDA 검사 13개, 결과 통계 검사 2개, CUDA 전체 분석/gating 검사 2개.
- 마지막 GPU 참조 해제 수정 뒤 CUDA 전체 분석/gating 2개도 다시 통과했다.
- GPU: RTX 5070 Ti 16 GiB, Torch 2.13.0+cu130, Python 3.13, CPU affinity 16.
  모델 FP32 / TF32 off, 에너지 FP64. CUDA가 없는 CPU 대체 실행은 하지 않았다.

독립 CUDA oracle은 전체 induced local incidence matrix와 직접 distinct-edge 교차항을
작은 **별도 검산 그래프**에서 계산해 희소 구현과 비교했다. packed stage/feature chunk,
방향 정합성, 배율, translation, degree-zero, isolated/empty graph, 음수 J를 검증했다.
local/global/augmented degree normalization과 TopoOOD의 global-degree control도 검산했다.
정규화 J가 ±1을 넘는 예제를 포함했다. 작은 검산 그래프는 최종 실험 데이터가 아니다.

전체 경로 검사는 **이미 완료돼 있던 DEBUG checkpoint**를 이용했다. DEBUG 모델은
hidden 8·3 epochs의 기존 fixture이며 새로 학습하지 않았다. 9 checkpoint × seed 11에서
99 stage / 99 transition, original metrics 재현, optimizer 생성 금지, 원본 artifact
불변, no final ReLU, MLP actual aggregation=I, GCN replay P와 실제 aggregation의
수치 일치를 확인했다. 이 fixture의 성능 수치를 연구 결과로 제출하지 않는다.

CUDA scatter 집계는 같은 연산도 비트 단위로 달라질 수 있다. 예를 들어 DEBUG-Cora
GCN logits 재현 최대 차이는 `1.4901161193847656e-08`이었다. 각 모델별 최대 오차와
`atol=rtol=2e-6` 통과 여부를 기록한다. 저장된 CE는 같은 허용오차, accuracy는
`atol=1e-7`로 확인한다. 원본 metric 재현 실패 시 다른 checkpoint를 고르지 않고 중단한다.

## 원래 크기의 실제 citation 입력 검사

이미 존재하는 SHA256 검증 cache에서 **모든 노드·엣지·특징**을 사용했다.
가짜 모델/예측을 만들지 않았다. 각각 input E/J의 서로 다른 feature chunk 결과와
15개 동일 input을 묶은 stage 축 결과를 비교했다. 이 15개 copy는 seed 반복이나
학습 모델이 아니라 vectorized 계산·메모리 검사용이다.

| Dataset | Nodes | Physical edges | Full features | Induced local edges | Directed center pairs | Max input reserved MiB | Max packed reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2,708 | 5,278 | 1,433 | 15,446 | 10,556 | 934 | 890 |
| CiteSeer | 3,327 | 4,552 | 3,703 | 12,605 | 9,104 | 782 | 804 |
| PubMed | 19,717 | 44,324 | 500 | 126,208 | 88,648 | 7,234 | 7,018 |

여기서 max는 실제 측정한 chunk 후보들의 최대 **allocator reserved** 값이다.
한 input의 선택 chunk는 Cora/CiteSeer 512, PubMed 128이었다. packed 15-axis 후보
8/32 중 선택 값은 32/32/8이었다. PubMed packed 최대 allocated는 3,358.69 MiB였다.
이 수치는 실제 local RTX 실행 측정이며 A100 MIG에서 측정했다고 표현하지 않는다.
서버에서는 실제 할당 VRAM과 처리시간으로 다시 정확한 chunk를 선택한다.
구조/feature/graph/stage 수는 줄이지 않는다.

전체 입력 검사는 `results/frozen-energy-full-inputs-COMPONENT-20261005-01`에 기록했다.
정적 topology 구성은 각각 0.342 / 0.344 / 3.664초, 구성·calibration·packing 검사
전체는 각각 3.943 / 6.925 / 10.163초였다. 모델 forward/결과 저장을 포함한 최종
checkpoint 분석 시간의 추정치로 사용하지 않는다.

## 남아 있는 실행

서버의 **full 규모로 학습된 seed-11 selected checkpoint 9개**에 대한 실제 가설 분석은
아직 실행하지 않았다. 로컬에 그 full checkpoint가 없으므로 다시 학습하거나
DEBUG weight를 full graph 결과처럼 사용하지 않았다. 따라서 현재 기록은 구현·수학·
원래 크기 입력·DEBUG 전체 파이프라인 검증이며, 가설을 지지/기각하는 최종 결과가 아니다.
한 seed 분석만으로 여러 초기화에 대한 안정성이나 통계적 유의성을 주장하지 않는다.

검수용 묶음에는 현재 코드/설계/테스트와 이름에 DEBUG 또는 COMPONENT scope가 표시된
실행 증거만 넣는다. 서버 selected checkpoint와 원본 데이터를 새로 배포하지 않는다.
