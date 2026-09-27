# 실제 arxiv CUDA 메모리 검증 및 후속 보완

이 문서는 `4ddead7` 이후의 메모리 검증 기록이다. 공식 ogbn-arxiv를 로컬 GPU에서
직접 실행한다. 합성 smoke test, 실제 데이터 calibration, 최종 학습, 실제 MIG 측정을
혼동하지 않는다. 최종 검증 수치는 이 문서 하단과 묶음의 JSON/XML 증거를 따른다.

## 검증 환경과 보존한 연구 계약

- 실제 장치: NVIDIA GeForce RTX 5070 Ti, 16GB급. A100/MIG 장치가 아니다.
- PyTorch 2.13.0+cu130, PyG 2.8.0.post1, Windows, CPU 16 logical / RAM 64GiB.
- 공식 ogbn-arxiv: 169,343노드, 1,157,799 무방향 physical edges,
  2,315,598 방향성 arcs, 128 features, 40 classes.
- 공식 train 90,941 / validation 29,799 / test 48,603. Test label 평가는 실행하지 않는다.
- reference 8 layers / hidden 256 / 8 heads, portable FP32, checkpoint 활성화,
  edge chunk 4096. 원래 최종 epochs=200 계약은 변경하지 않는다.
- full calibration은 warmup 2 complete passes + measurement 5 complete passes,
  전체 validation, 전체 mechanism audit 및 intervention restoration을 수행한다.
- sampled는 context seeds=2048, fanouts=15/10, physical seed floor=2048 그대로다.
  각 calibration pass는 모든 공식 supervised train seeds를 포함한다.
- 이 로컬 메모리 검증은 지정된 physical floor를 측정한다. 최적 batch/worker를 찾았다는
  주장이 아니다. 서버의 기존 common calibration은 더 큰 physical batch와 worker 후보를
  계속 실측한다. full은 단일 공식 그래프가 이미 GPU에 상주하므로 physical=1/workers=0이고,
  sampled context CPU workers는 기존 규칙으로 4가 선택된다. GPU 모델 연산의 CPU fallback은 없다.
- 데이터 준비 과정에서 공식 데이터만 다운로드했다. 모델·사전학습 가중치는 받지 않았다.

## 수정 전 실제 측정

| 조건 | PyTorch 상한 | peak allocated | peak reserved | 결과 |
|---|---:|---:|---:|---|
| full fixed C | 10GiB | 4.824GiB | 9.201GiB | 실행 완료, 10GiB 기준 2GiB 여유 불충족 |
| full learned C | 10GiB | 5.930GiB | 9.723GiB | 실행 완료, 10GiB 기준 2GiB 여유 불충족 |
| full learned C | 7.5GiB | JSON 참조 | JSON 참조 | 학습 완료 후 진단 projection에서 OOM |

원래 오류 문구만으로 서버 실패가 실제 OOM인지 headroom rejection인지 구분할 수
없었다. 위 로컬 측정은 실제로 headroom rejection을 재현한다. 사용자의 서버에서
정확히 어느 경로로 실패했는지를 원본 calibration JSON 없이 단정하지 않는다.

`expandable_segments:True`는 이 Windows 빌드가 지원하지 않았다. `max_split_size_mb:128`
단독 변경도 해결하지 못했다. 두 시도는 성공 증거로 사용하지 않는다. 메모리 상한만
낮춘 시도도 실패했으므로, allocator 제한만으로 해결했다고 주장하지 않는다.

첫 수정 후 실측(`*-fp32-after-cap7.json`)에서는 fixed/learned 모두 7GiB 상한에서
전체 calibration을 완료했다. peak reserved는 각각 6.936 / 6.811GiB다. 수정 전과
공식 데이터 protocol, 초기 model state hash, parameter 수, 실제 처리 supervised units가
각각 일치한다. 최종 소스 고정 후 재측정 결과는 `*-final-cap7.json`으로 별도 보존한다.

## 변경 내용

1. Energy가 없는 core 두 조건의 **진단 전용** projection에서 전체 hop 입력을 동시에
   stack하지 않고 노드 chunk로 projection한다. 모든 노드·hop·head를 계산하고 전체
   Gram을 유지한다. 예측·학습의 contraction 순서와 energy branch는 변경하지 않는다.
   관측 projection의 GPU GEMM shape는 달라질 수 있으므로 FP32/BF16 수치 대조를 수행한다.
2. 진단 Gram 생성 후 불필요한 projection 참조를 해제한 뒤 FP64 통계를 계산한다.
3. `--cuda-allocator-limit-gib`를 추가한다. 모델·데이터·physical batch 축소가 아니라
   PyTorch native allocator의 예약/할당 예산이다. 부족하면 명시적인 CUDA OOM으로
   중단하며 CPU나 작은 모델로 전환하지 않는다. CUDA context 등 비-PyTorch 메모리는
   이 상한 밖에 있으므로 기존 2GiB/10% 안전 여유 검사도 유지한다.
4. 이 옵션을 core→comparison→calibration/train으로 전달하고 immutable configuration과
   저장 training arguments에 포함한다. 감사와 최종 test가 복원한 인자로 모델을 만들 때도
   같은 상한을 적용한다. 새 source identity/설정에는 새 run ID를 사용한다.
5. 실패 출력에 조건·seed·physical batch·workers·peak allocated/reserved·free before·
   안전 여유 부족량을 표시한다. 실제 OOM과 headroom rejection을 구분하고 OOM의
   traceback 및 자원 관측도 보존한다.

## 증거 해석

최종 영향 회귀: **160 passed / 0 failed / 0 errors / 0 skipped**, 674.28초.
CUDA 검증 143개와 metadata/control 검증 17개를 구분한다.
`results/arxiv-memory-regression-20260927.xml`이 최종 실행 증거다. 앞선 22개 focused
실행 및 과거 230개는 별도 기록이며 합산하지 않는다. 기존 PyG annotation deprecation과
pytest `record_property`/xunit2 경고가 있었고, 실패나 CUDA 미실행 skip은 없었다.
정적 Ruff 검사와 diff whitespace 검사도 통과했다.

### 최종 소스를 고정한 실제 데이터 측정

| 조건 | 전체 경로 peak allocated (GiB) | 전체 경로 peak reserved (GiB) | train pass 시간 (초) |
|---|---:|---:|---:|
| full / fixed C | 3.772 | 6.936 | 1.840 |
| full / learned C | 5.214 | 6.811 | 20.378 |
| sampled / fixed C | 3.528 | 6.902 | 136.078 |
| sampled / learned C | 4.631 | 6.576 | 450.574 |

**네 조건 모두 통과했다.** 네 JSON의 시작/종료 source hash가 일치하며, 검증 종료 시
현재 source snapshot과도 일치함을 `verification-summary.json`으로 대조했다.
전체 경로 peak reserved는 모두 7GiB 미만이다. 기존 2GiB 여유 기준에 필요한 free VRAM은
각각 8.936 / 8.811 / 8.902 / 8.576GiB다. 실제 MIG에서는 장치의 실제 free VRAM과
실제 측정치를 사용한 기존 safety check를 다시 통과해야 한다.

메모리 peak에는 학습뿐 아니라 full-graph validation, 진단, intervention/restoration이
포함된다. train pass 시간은 모든 train seeds를 사용하는 한 pass의 시간이며 데이터 생성,
전송, forward/backward, optimizer를 포함한다. full은 측정 5 pass 평균, sampled는 측정
1 complete pass(45 optimizer updates)다. 별도의 warmup complete pass는 시간 평균에서 제외한다.
이는 로컬 단일 seed·단일 physical floor의 자원 calibration이며 최종 정확도나 평균 benchmark가 아니다.

현재 floor에서 sampled 두 조건 모두 full보다 느렸다. 특히 learned C의 train pass는
full 20.378초 대 sampled 450.574초다. 이것은 해당 로컬 후보의 실측이며 다른 hardware나
batch 선택으로 일반화하지 않는다. sampled fixed의 topology 준비
누적 시간은 206.54초(초기 준비 및 warmup/measurement 모두 포함)다. sampling이 빠르다는
주장은 이 결과로 뒷받침되지 않는다. 실제 MIG에서 더 큰 physical batch/worker 후보를
선택한 결과와도 구분한다. 이 수정은 메모리 실행 경로를 보완하며 연구의 속도 가설을
자동으로 성립시키지 않는다.

최종 상태: 구현·정적 검사·GPU/제어 회귀·공식 전체 데이터 네 조건 자원 calibration 완료.
200 epoch 최종 학습 및 공식 test는 미실행이다. 실제 A100 MIG 장치에서의 실행은 미검증이다.

`scripts/profile_arxiv_memory.py`는 기존 실제 calibration 함수를 사용한다. 공식 캐시가
없으면 오류를 내며, subset이나 가짜 그래프를 만들지 않는다. 결과 경로가 이미 있으면
덮어쓰지 않는다. GPU와 전체 물리 메모리를 실제 값으로 기록하고, PyTorch 상한을 별도로
기록한다. 이 상한은 A100 MIG 에뮬레이션이 아니다. Local RTX 시간으로 MIG 속도를 추정하지 않는다.

수정 전 실험은 각각 시작 당시 source snapshot으로 기록되며 최종 소스 검증과 분리한다.
초기 `learned-fp32-cap7p5.json` 탐색 중 파일 편집이 겹쳐 traceback의 일부 표시 줄이
실행 당시 파일과 일치하지 않는다. 그 기록은 OOM 관측 자료로만 보존하며, 최종 소스의
검증 증거로 사용하지 않는다. `*-final-cap7.json`은 실행 전후 전체 source hash가
동일해야 성공으로 기록하도록 보완한 뒤 소스를 고정하여 측정한다.
최종 200 epoch 학습, 공식 test 점수, multi-seed 우위, 실제 A100 MIG 성능은 이 작업의
calibration 결과로 입증되지 않는다.

## 서버 실행 계약

아래는 기존 reference/200 epochs/context 2048/physical floor 2048/fanouts 15,10을
보존하고 CUDA allocator 예산만 7GiB로 명시하는 실행이다. 전체 안전 여유 검사와
physical batch/worker 후보 실측은 그대로 수행한다. 모델이나 그래프를 자동 축소하지 않는다.
실제 MIG에서의 판단은 이 장치에서 측정하는 calibration 결과에 따른다.
기존 실패 run은 보존하고 새 ID를 쓴다. calibration-only가 네 조건을 확인한 뒤 학습한다.

```bash
cd /home/aicompetition07/new-gat && git pull --ff-only && {
  CORE_ARGS=(
    --run-id arxiv-core-reference-mig10gb-gpu4-seed0-v2
    --datasets ogbn-arxiv --profiles reference --model-seeds 0
    --device cuda:0 --hardware-profile portable --epochs 200
    --gram-implementation reference
    --sample-context-seed-batch-size 2048 --sample-seed-batch-size 2048
    --num-neighbors 15 10 --edge-chunk-size 4096
    --cuda-allocator-limit-gib 7
    --activation-checkpoint --min-free-gb 8 --evaluate-test
  )
  env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
    /home/aicompetition07/.conda/envs/new-gat/bin/python -B \
    -m experiments.aggregation_comparison.core "${CORE_ARGS[@]}" --calibration-only &&
  env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
    /home/aicompetition07/.conda/envs/new-gat/bin/python -B \
    -m experiments.aggregation_comparison.core "${CORE_ARGS[@]}"
}
```

이 core 실행은 F0/F1/S0/S1 네 조건이다. 21조건 외부 모델 비교를 실행했다는 의미가 아니다.
