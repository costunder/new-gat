# Conductance GAT 성능 진단

완료된 benchmark의 학습 기록과 `best.pt`를 읽고, 같은 모델로 **GPU 추론만** 수행한다.
재학습, optimizer update, 모델 변경, 데이터 다운로드는 하지 않는다. 기존 checkpoint,
학습 기록, 데이터 캐시는 덮어쓰지 않는다. 기본 출력은 터미널이며 파일 저장은 선택 사항이다.

## 실행

학습에 사용한 Conda 환경을 활성화하고 저장소 루트에서 실행한다. `RUN_ID`는 학습 종료 때
출력된 기존 실행 ID로 바꾼다. 새 실험 ID가 아니다.

```bash
bash scripts/diagnose_conductance.sh --run-id RUN_ID --datasets cora ppi
```

처음에는 Cora와 PPI의 model seed 0부터 확인한다. arxiv 및 다른 seed는 별도로 선택한다.

```bash
bash scripts/diagnose_conductance.sh --run-id RUN_ID --datasets ogbn-arxiv --model-seed 0
```

기존 모델에서 이웃 전파만 끄는 비교를 추가하려면 다음처럼 실행한다.

```bash
bash scripts/diagnose_conductance.sh --run-id RUN_ID --datasets cora ppi --ablate-graph
```

이 옵션은 **validation에서만**, 동일 checkpoint의 conductance 전파를 항등 연산으로
우회한다. encoder, LayerNorm, ELU, decoder는 남긴다. 별도의 MLP를 학습하는 비교가 아니며,
학습 때 없던 입력 분포를 만들 수 있으므로 성능 차이를 원인 확정으로 해석하지 않는다.

별도 JSON 보고서도 보존하려면 `--output-dir runs/diagnostics/conductance-seed0`를 추가한다.
기존 폴더를 재사용하지 말고 매번 새 경로를 쓴다. 학습 결과 폴더 안에는 저장하지 않는다.

학습 때 사용자 지정 경로를 썼다면 `--results-root`와 필요할 경우 `--data-root`로
실제 경로를 지정한다. 기본 결과 위치는 다음 구조다.

```text
research/conductance_gat/results/paper/RUN_ID/model-seed-0/benchmark/
```

다른 GPU를 할당받았다면 `--device cuda:0`처럼 **현재 프로세스에 보이는 GPU 번호**를
지정한다. CPU 추론 fallback이나 자동 의존성 설치는 없다. CLI 도움말은 다음과 같다.

```bash
bash scripts/diagnose_conductance.sh --help
```

## 확인하는 내용

| 진단 | 의미 |
|---|---|
| 학습 기록과 best epoch | 학습 손실의 감소·정체, validation 선택 시점, 최대 epoch까지 학습했는지 확인 |
| checkpoint의 train / validation 성능 | dropout을 끈 동일 모델에서 성능 차이를 확인. 당시 dropout을 켜고 기록한 train loss와 수치가 같을 필요는 없음 |
| 층별 가중 차수와 `rho` 분포 | 그래프 최대 가중 차수 대비 이웃 정보가 얼마나 섞이는지 확인 |
| 층별 conductance 분포·변동계수 | 엣지별 C가 거의 상수인지, 상대 가중치 차이가 생겼는지 확인 |
| 전파 전후 상대 변화량 | LayerNorm·ELU 이전 conductance 연산 자체가 표현을 얼마나 바꾸는지 확인 |
| PPI 정답 / 예측 양성 비율 | 낮은 micro-F1이 과소·과다 양성 예측과 함께 나타나는지 확인 |
| 선택적 validation 전파 우회 | 기존 checkpoint가 이웃 전파에 얼마나 민감한지 확인 |

현재 연산의 노드별 총 이웃 가중치는 다음과 같다.

\[
\rho_i=0.95\frac{d_i^C}{\max_j d_j^C},\qquad d_i^C=\sum_j c_{ij}.
\]

최대값은 **각 그래프별**이다. PPI의 여러 그래프를 합쳐 하나의 최대값을 쓰면 안 된다.
`rho`가 대부분 작으면 약한 이웃 전달 가설을 지지한다. 그러나 이 값만으로 낮은 정확도의
원인을 확정할 수는 없다. C 전체의 공통 크기는 스텝 크기에서 상쇄되므로, C 평균의 크기보다
상대 분포·`rho`·실제 전파 변화량을 함께 본다. C가 거의 상수여도 그래프 전파 자체가 사라지는
것은 아니다. 1%/5%/10% 같은 분포 구간은 설명용이며 검증된 합격·실패 기준이 아니다.

## 평가와 재현 경계

- 진단은 FP32이며 AMP/TF32를 끈다. 원래 run이 AMP/TF32를 사용했다면 저장된 validation과
  차이가 날 수 있다. 원래 저장값과 재계산값의 차이를 함께 확인한다.
- 새로 계산하는 성능은 train과 validation뿐이다. test 점수는 저장된 결과만 표시하며
  test 기준으로 새 ablation·튜닝을 하지 않는다.
- Cora/CiteSeer/PubMed/arxiv는 기존 학습처럼 전체 그래프 특징을 사용하는 transductive
  forward다. test 노드의 특징과 연결까지 제거하는 것이 아니라 test label로 평가하지 않는 것이다.
- PPI는 공식 train / validation 그래프를 모두 읽는다. micro-F1은 그래프별 F1의 평균이 아니라
  전체 노드·라벨의 TP/예측 양성/정답 양성 수를 합쳐 계산한다.
- 추론 당시 소스와 학습 당시 소스 checksum이 다르면 경고를 확인한다. state_dict가 로드된다고
  전파 수식까지 동일하다는 보장은 없다. 소스 차이를 확인한 뒤 수치를 해석한다.
- 이 진단은 모델의 학습 속도나 GPU 가속 배수를 측정하지 않는다. 새로운 seed의 학습 결과나
  연구 가설의 최종 검증으로 집계하지 않는다.
