# 실제 수행한 검증

**이 문서는 초기 bracket 판의 보존 기록이다. review_fixes_1의 최신 검증은
`REVIEW_FIXES.md`를 읽는다. 아래 초기판 calibration은 수정판 학습에 재사용하지 않는다.**

## 현재 판단

새 생성기가 실제 전파·분류 CE·역전파·optimizer에 연결된 것은 확인했다.
**유용한 C를 학습했는지와 fixed C=1보다 좋은지는 아직 확인하지 않았다.**
200epoch 본학습 및 학습된 두 모델의 최종 성능 비교는 수행하지 않았다.

## 정적 검사와 단위검사

- Ruff 검사 통과.
- `test_debug.py`: **19 passed**, 11.33초, skip 없음.
- 결과 원본: `results/bracket-debug-20260929-01.xml`.
- 대칭 점수와 exp, head별 출력, chunk 크기별 출력·gradient 일치를 확인했다.
- 명시적인 dense 발생행렬 계산과 sparse 전파를 FP64로 비교했다.
  고립 노드와 엣지 방향 반전도 검사했다.
- 분류 CE의 Q/K weight·bias 수치미분을 검사했다. C로 계산하는 degree도
  미분 경로에 포함한다. 실제 모델의 파라미터·엣지 가중치 섭동 검사도 통과했다.
- 고정 입력에서 독립 그래프를 배치에 추가해도 기존 엣지의 raw C가 같았다.
- 기존 optimized 생성기를 만들거나 호출하면 실패하도록 막은 구성 검사를 통과했다.
- fixed의 공통 파라미터를 기존 모델과 맞추면 forward 출력이 정확히 같았다.
- 관측 on/off의 출력·gradient·갱신·RNG 일치와 checkpoint 경로를 검사했다.
- CUDA BF16 관측/시간 계측을 검사했다. 실제 calibration은 FP32다.
- nonfinite exp는 clamp 없이 오류를 내는지 CPU에서 검사했다.

CUDA 파이프라인 smoke는 8층·256·8 heads를 유지한 **합성 데이터 2epoch**다.
learned/fixed 공통 초기값·같은 샘플 순서, checkpoint 선택·재로딩, C=1 개입,
새 문맥 평가의 연결을 확인했다. 합성 checkpoint에는 debug 표시가 있다.
이 smoke는 실제 데이터 학습 성능의 증거가 아니다.
4개 warning은 외부 torch JIT API deprecation 2개와 합성 그래프의 문맥 포화 2개다.

## 실제 데이터와 자원

ogbn-arxiv 원본 전체 169,343 nodes, 중복 제거한 무방향 물리 엣지 1,157,799개를
사용한다. 공식 train 90,941 / validation 29,799 / test 48,603 split을 유지한다.
calibration은 일부 학습 배치의 시간·미분 관측이며 전체 학습 pass가 아니다.
각 측정 조합에서 **전체 validation 29,799개**와 C=1 개입을 실행한다.
새 문맥 seed 크기 2048/4096에서도 validation을 모두 평가한다.

장치: RTX 5070 Ti 한 개, VRAM 17,094,344,704 bytes, 논리 CPU 16개,
RAM 68,640,653,312 bytes. 시작 시 가용 RAM 약 38GB였으며 결과 파일에
각 조합의 peak GPU 메모리·RSS·가용 RAM·CPU 사용량을 기록한다.
CPU/GPU 시스템 설정이나 원격 세션은 변경하지 않았다.

실제 모델은 8층·hidden 256·8 heads다. learned 2,694,632개,
fixed 1,641,960개의 학습 파라미터를 갖는다. accumulation=1, GPU worker=1이다.
physical seed 2048/4096은 effective seed batch와 같고, 문맥당 seed는 2048이다.
서로 다른 문맥은 disjoint union으로 동시에 처리한다. context worker 2/4를 측정한다.

기존 portable profile의 실제 적용값은 FP32, TF32 off, pinning on, prefetch off다.
실행 명령에 주었던 prefetch flag보다 공유 profile 설정이 우선했다.
RUN.md의 재현 명령에서는 오해를 막기 위해 해당 flag를 생략했다.
모델·문맥 크기·엣지 수를 OOM 회피용으로 줄인 적은 없다.

## 초기 C 관측

`inspection-learned-2048-2.json`은 calibration 초기 갱신에서 같은 층 입력을
고정해 기록한 결과다. 모든 32개 Q/K 파라미터 tensor의 갱신과 live C gradient가
확인됐다. AdamW weight decay도 파라미터를 바꾸므로 갱신량만으로 CE 효과를
판단하지 않는다. CE 미분은 별도 수치미분 및 live C gradient로 검사했다.

해당 배치의 head별 C 표준편차 중 최댓값은 첫 층 약 2.246e-3,
마지막 층 약 2.864e-9였다. **깊은 층의 초기 C는 여전히 1에 매우 가깝다.**
이 측정은 초기 상태이며 학습 종료 후의 C나 학습 실패 판정이 아니다.
이 현상을 보고 초기화·출력 투영·기본 전파를 임의로 변경하지 않았다.

같은 입력의 alpha 갱신, 학습된 모델의 C=1 개입 영향, 별도 fixed 대비
정확도 차이는 각각 다른 증거다. 최종 유용성 판단에는 본학습 결과가 필요하다.

## 완료 범위

구현, 정적 검사, 단위검사, 합성 CUDA smoke를 완료했다.
실제 데이터 calibration도 8개 조합 모두 완료했다. 종료 시 251개 소스의 hash와
과학 설정이 시작 시점과 같음을 확인했다. 오류나 OOM, 수식 대체는 없었다.
200epoch 본학습, 학습된 모델의 전체 validation 비교, 공식 test 평가는 미실행이다.

## 실제 calibration 결과

실행 시각: 2026-09-29 00:57–01:27 KST.
원본: `results/bracket-arxiv-calibration-20260929-01/calibration.json`.
검토 ZIP에는 같은 자료를 `evidence/real_data_calibration`에 넣었다.
각 조합에서 warmup 후 실제 배치 3개의 평균이다. 두 step은 같은 배치·파라미터·
optimizer 상태·dropout 난수로 짝지어 비교했다. 복사/복원 시간은 제외했다.

| 조건 | physical seed | worker | 일반 step 초 | 관측 포함 step 초 | 준비·전송 초 | peak reserved GiB | seed/초 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | 2048 | 2 | 1.103 | 7.073 | 2.949 | 3.527 | 505.5 |
| fixed | 2048 | 2 | 0.225 | 6.145 | 2.939 | 3.422 | 647.2 |
| learned | 2048 | 4 | 1.137 | 6.971 | 2.770 | 3.527 | 524.3 |
| fixed | 2048 | 4 | 0.232 | 6.066 | 2.743 | 3.422 | 688.4 |
| learned | 4096 | 2 | 2.966 | 17.602 | 6.238 | 5.475 | 445.1 |
| fixed | 4096 | 2 | 0.668 | 14.146 | 5.906 | 5.182 | 623.0 |
| learned | 4096 | 4 | 2.940 | 16.043 | 5.812 | 5.475 | 468.0 |
| fixed | 4096 | 4 | 0.702 | 13.690 | 5.714 | 5.182 | 638.5 |

seed/초는 준비·전송과 일반 step을 합친 시간으로 계산했다. 상세 관측 시간은
제외한다. 전체 실행 시간을 이 열로 대체해서는 안 된다. 특히 초기화·관측·
validation·checkpoint I/O는 별도다. 준비·전송에는 샘플 생성과 CPU 측
샘플 동일성 기록도 포함하므로, 이 시간을 순수 디스크 I/O로 해석하지 않는다.

일반 step 안의 CUDA event 평균은 다음과 같다. CPU 시간과 더하지 않는다.

| 조건 | seed | worker | forward+CE 초 | backward 초 | optimizer 초 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned | 2048 | 2 | 0.2109 | 0.8857 | 0.0022 |
| fixed | 2048 | 2 | 0.0827 | 0.1383 | 0.0015 |
| learned | 2048 | 4 | 0.2017 | 0.9291 | 0.0021 |
| fixed | 2048 | 4 | 0.0875 | 0.1403 | 0.0013 |
| learned | 4096 | 2 | 0.4528 | 2.5067 | 0.0017 |
| fixed | 4096 | 2 | 0.3070 | 0.3572 | 0.0018 |
| learned | 4096 | 4 | 0.5037 | 2.4289 | 0.0024 |
| fixed | 4096 | 4 | 0.3510 | 0.3466 | 0.0018 |

마지막 측정 배치의 입력은 2048 조건에서 `[53248,128]`, 물리 엣지 379,398개,
4096 조건에서 `[106496,128]`, 두 문맥의 물리 엣지 합 769,884개였다.
실제 edge 수는 배치마다 달라진다. 전체 validation과 C=1 개입,
2048/4096 문맥 평가를 모든 조합에서 완료했다. peak reserved는 이 평가까지
포함하며 모든 조합이 총 VRAM의 90% 이하였다. 측정 종료 시 RSS의 최댓값은
약 3.717GiB다. CPU/RAM 값은 스냅샷이며 전체 실행의 peak RAM이라고 부르지 않는다.

**이번에 측정한 후보 중 본학습 시작값은 physical seed 2048, worker 4로 선택한다.**
두 조건 모두 이 조합의 측정 처리량이 가장 높았다. 4096이 메모리에 들어갔지만
이번 실행에서는 처리량이 증가하지 않았다. 이 선택은 작은 batch를 임의로 고른
것이 아니며, 전체 가능한 batch/worker 중 최적이라고 입증한 것도 아니다.
단일 짧은 측정이므로 worker 2와 4의 작은 차이에 강한 인과 해석은 하지 않는다.
모델과 문맥 크기·학습 기간·데이터 계약은 유지한다.

GATv2나 이전 반복 최적화 생성기의 동일 조건 시간을 측정하지 않았다.
따라서 이 자료로 그 모델들보다 빠르다거나 정확하다고 주장하지 않는다.

## 보존 확인

기존 원본 462개 파일과 직전 C-only v2.0 파일 12개를 해시/내용으로 대조했고
변경이 없었다. 검토 패키지 생성기는 이전 information-flow 후보도 별도로 비교한다.
기존 checkpoint·실험 결과·ZIP을 덮어쓰지 않는다.
