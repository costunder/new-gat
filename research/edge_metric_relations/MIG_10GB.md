# A100 MIG 10GB 실행 정책

## 유지하는 실험

이 옵션은 실행 자원 설정이다. `--profile full`의 과학적 계약을 줄이지 않는다.

| 단계 | 전체 실험 계약 |
| --- | --- |
| A | 201개 그래프, 모든 원래 feature, 21개 관측, 두 C recipe, float64 감사 |
| B | 531개 그래프, 16개 feature draw, 학습 240회 × 500 epoch = 120,000 seed update, float32 |
| C | 전체 Cora/CiteSeer/PubMed, 15조건, 2층·hidden 64, 학습 630회 × 500 epoch = 315,000 seed update, float32 |

모든 eligible pair와 induced local을 유지한다. Chunk는 계산을 분할하며 정보를 제거하지 않는다.
TF32·mixed precision·모델 축소를 적용하지 않는다. 기존 activation checkpointing을 사용한다.
기존 scientific config의 `resources.gpu`에는 최초 A6000 계획이 남아 있다.
실제 GPU·할당 메모리·실행 정책은 새 `hardware.json`, worker resource와 calibration 기록을 기준으로 읽는다.

## 메모리와 병렬 처리

- CUDA의 `mem_get_info`와 device property에서 **할당된 instance**의 실제 용량을 읽는다.
- 이미 GPU에 올라간 정적 자료를 제외한 free memory의 70%와 `free − 1GiB` 중
  작은 값을 추가 할당 예산으로 사용한다. Allocated·reserved peak를 모두 검사한다.
- 기존 후보에 더 작은 exact chunk를 추가하고 후보별 처리량·peak memory를 측정한다.
  Independent seed의 동시 배치 수는 측정해서 고른다. Seed 수나 전체 학습 예산은 유지한다.
- B 후보의 측정 시간에는 전체 train epoch와 OOD를 포함한 모든 split의 forward가 함께 들어간다.
  이 calibration 시간과 실제 학습 epoch 시간을 구분한다.
- C는 전체 그래프를 CPU에 cache하고 현재 dataset의 전체 그래프만 GPU에 유지한다.
  다른 dataset으로 옮길 때 GPU 자료를 해제한다. 노드·엣지·feature를 제외하지 않는다.
- 큰 후보는 중간 텐서 메모리 추정치로 먼저 검사한다. 추정 때문에 제외한 후보와
  실제 실행 후 제외한 후보를 구분해 기록한다. 실제 측정에 통과한 후보만 선택한다.
- 안전한 후보가 없으면 원인과 남은 메모리를 출력하고 중단한다. 실험 규모를 자동으로 줄이지 않는다.

CPU에서 전체 citation geometry를 따로 생성해 확인한 원래 pair 수와 공유 정적 저장량이다.
GPU 학습 peak 측정값은 아니다. Feature 폭과 모델의 중간 텐서는 별도로 더해진다.

| 데이터 | 전체 eligible pair | 두 recipe 공유 geometry float32 | 원래 입력 X float32 |
| --- | ---: | ---: | ---: |
| Cora | 297,364 | 29,052,784 B | 15,522,256 B |
| CiteSeer | 189,870 | 18,708,324 B | 49,279,524 B |
| PubMed | 4,695,369 | 424,859,424 B | 39,434,000 B |

PubMed의 `5 seeds × 4,695,369 pairs × 64 features × 4 bytes`는 텐서 하나만
약 6.01GB다. 전체 pair를 처리하되 작은 chunk로 중간 메모리를 제한해야 하는 이유다.

## 할당과 로그

할당받은 단일 `MIG-…` UUID를 `CUDA_VISIBLE_DEVICES`로 지정한다.
여러 MIG instance의 CUDA 열거 방식은 driver/CUDA 버전에 따라 다르므로 이 명령은 단일 instance를 사용한다.
MIG를 생성하거나 서버 설정을 변경하지 않는다.
실제 nominal 10GB 용량과 SM 수는 런타임 기록을 읽는다. MIG 이름만으로 A6000 대비 속도를 예측하지 않는다.
[NVIDIA MIG 장치 이름 설명](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/mig-device-names.html),
[MIG 실행 안내](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html).

- Pipeline: `hardware_policy.json`, `plan.json`, 각 단계 `terminal.log`.
- A/B: 메모리·batch·chunk 후보와 실제 선택 결과를 각 단계 resource/calibration 기록에 저장한다.
- C: `hardware.json`, `calibration/*.json`, `resources.csv`,
  `workers/*/resources.json`, `workers/*/dataset_residency.json`.
- C의 checkpoint 재개는 같은 code/config/seed packing을 요구한다.
  허용된 재개에서도 현재 GPU에서 기존 packing을 다시 검사한다.
  A6000 결과를 조용히 재배치하거나 중간 epoch를 버리지 않는다.
- 코드 hash가 바뀐 이전 A/B 결과의 재사용은 허용되지 않는다.
  새 출력 경로에서 시작하며 기존 결과와 원래 local-energy 입력은 보존한다.

실행 명령과 결과 확인 방법: [RUN.md](RUN.md).
