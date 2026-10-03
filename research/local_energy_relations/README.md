# 1홉 로컬 에너지와 로컬 그래프 사이 관계

**각 노드 주변의 실제 1홉 그래프 안에서 에너지를 재고, 서로 연결된 두 노드의
로컬 그래프 사이에서 메시지 관계를 잰다.** 이웃끼리 연결된 엣지까지 포함한다.
겹치는 노드로 무엇이 전달되고, 여러 로컬 그래프를 합칠 때 같은 엣지를 몇 번 세는지도 기록한다.

이 첫 단계는 고정 연산의 감사다. C는 `1` 또는 로컬 degree로 정한 양수이며 학습하지 않는다.
분류기와 CE 학습도 없다. 현재 wedge 실험의 완료 결과나 새로운 모델 성능으로 합산하지 않는다.

## 실제 계산 순서

```text
전체 원래 그래프·전체 특징
    → global Laplacian 확산으로 공통 H0/H1/H2 생성
    → 모든 중심의 1홉 induced graph와 원래 ID 대응 준비
    → 두 고정 C로 엣지 메시지 q=C BH, 노드 집계 d=Bᵀq 계산
    → 각 로컬 그래프의 이차 에너지와 실제 cycle 성분·known-C 복원 검사
    → 연결된 모든 중심 쌍의 같은 단계/교차 단계 쌍선형 관계
    → 공통 노드로 d 전달 시 남는/빠지는 부분 감사
    → raw 중복 합과 물리 엣지 단위의 보정 합 비교
    → CSV·요약·PNG/PDF 과학 그림
```

에너지 `Σc||BH||²`와 메시지 크기 `Σc²||BH||²`는 구분한다.
집합 사이 관계에는 실제 incidence 대응 `B_v_global B_u_globalᵀ`를 쓴다.
서로 다른 incident 엣지의 항은 부호가 있으며 항상 양의 에너지라고 하지 않는다.
q의 cycle 성분이 있어도 알려진 양의 C와 전체 노드 집계 아래의 `q=CBH`는 복원될 수 있다.
부분 관측에서 원래 메시지를 복원하는 모델은 이번 범위에 없다.

공통 노드로 d를 복사하는 연산은 **전달 범위를 확인하는 bookkeeping·진단**이다.
그 d를 H_next 업데이트에 넣지 않는다. H1/H2는 전체 물리 라플라시안의 reference 확산에서 온다.
새 메시지 패싱 모델의 학습·예측 연결은 이 진단을 읽은 뒤 결정할 두 번째 단계다.

## 입력 규모

- FULL: 합성 198개와 실제 Cora/CiteSeer/PubMed 3개, 총 201개. 모든 노드·엣지·특징, float64, sampling 없음.
- DEBUG: 별도 fixture 21개. 실험 결과와 분리한 기능 검사.
- citation 특징은 기존 loader의 row-sum-normalized float32 입력을 float64로 승격한다.
- trainable parameter 0, epoch/optimizer step 해당 없음. H0/H1/H2는 공통 reference 좌표의 고정 확산 상태다.

[수식](MODEL_MATH.md)과 [실험 계약](EXPERIMENT_DESIGN.md)에 범위·관측·한계를 적었다.
구현·단위/DEBUG 검사와 서버 FULL 실행 여부는 실행 완료 기록에서 각각 확인한다.
이 문서를 작성했다는 사실이 FULL 완료를 의미하지 않는다.

## 서버 실행

GPU 선택부터 실행·완료 확인까지 한 터미널에서 진행하는 명령은 [RUN.md](RUN.md)에 있다.

기존 할당 GPU의 `CUDA_VISIBLE_DEVICES`를 유지한다. FULL은 서버 CUDA 실행이다.
다음 명령은 새 결과 폴더를 사용하며 기존 결과를 덮어쓰지 않는다.

```bash
test -n "${CUDA_VISIBLE_DEVICES:-}" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.study \
  --profile full --device cuda \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "results/local-energy-$(date +%Y%m%d-%H%M%S)"
```

A6000 48GB에서 실제 사용 가능한 VRAM과 batch별 처리량을 runner가 측정한다.
CPU 전처리는 가장 큰 합성 그래프와 전체 citation 그래프로 측정하고 선택된 결과를 cache한다.
GPU 후보는 실제 peak VRAM이 가용 메모리 안전 예산을 만족하는 경우에만 선택한다.
GPU 번호를 임의로 바꾸거나 실행 중인 다른 작업을 중단하지 않는다.
출력에는 그래프별 진행, 현재 계산 단계, 실제 전체 coverage와 resource 측정이 포함된다.
데이터·해법·계약 오류가 발생하면 해당 실패와 traceback을 남기고 완료로 표시하지 않는다.

## 결과 확인

`completion.json`에서 `status=complete`, `profile=full`, `graphs=201`,
`actual_citation_graphs=3`, 전체 입력·source 보존을 먼저 확인한다.
실행이 실패했거나 DEBUG면 이 조건을 FULL 완료라고 읽지 않는다.

- `LOCAL_ENERGY_RELATIONS_SUMMARY.md`: 실제 에너지·cycle·복원·중복 집계 표와 해석 범위.
- `graph_summary.csv`, `local_summary.csv`, `relation_summary.csv`, `transfer_summary.csv`, `assembly_summary.csv`: 모든 그래프의 요약 측정.
- `local_states.csv`, `relations.csv`, `transfers.csv`, `local_temporal.csv`: 모든 중심·방향·입력의 개별 측정.
- `resources.json`: 실제 graph batch/channel chunk calibration과 실행 시간·메모리.
- `figures/`: 세 과학 그림의 PNG와 PDF. 그래프 family·고정 C·reference 단계별로 표시한다.

전체 로그를 복사할 필요는 없다. 먼저 completion과 요약을 전달하고,
특정 관계나 중심을 검토할 때 해당 raw CSV를 추가한다.

## 다음 단계: 여러 송신 로컬의 수신 합

첫 감사의 입력·토폴로지를 그대로 읽는 [수신 집계 실험](receiver_aggregation/README.md)을
추가했다. 송신별 메시지 보존, 수신 행별 합, 내부 E·집합 사이 J 추가의 다섯 관측을 비교한다.
합에서 직접 숨는 송신별 차이와 실제 메시지 복원 가능성을 따로 측정한다.
서버 명령은 [receiver_aggregation/RUN.md](receiver_aggregation/RUN.md)에 있다.
