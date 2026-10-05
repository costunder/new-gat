# Frozen energy trace — single seed 11

2026-10-05. 기존 완료된 wedge classification의 selected checkpoint 9개만 읽는다.
Cora / CiteSeer / PubMed × MLP / Standard GCN / Polynomial-2 × seed **11**.
사용자가 시간을 이유로 seed 하나를 승인했으므로 원래 final seed 목록의 첫 값을
성적과 무관하게 고정한다. 다른 seed를 고르거나 평균·표준편차·신뢰구간을 만들지 않는다.

새 모델 설계, 학습, optimizer update, LR 재선택, checkpoint 재선택은 없다.
원래 full 계약은 2 layers, hidden 64, dropout 0.5, 500 epochs이며, 이 분석에서는
그 결과의 validation-selected `best_state`를 `eval()`로 읽는다. dropout은 꺼지고
마지막 출력에 ReLU를 추가하지 않는다. 기존 결과나 checkpoint를 수정하지 않는다.
checkpoint가 없거나 완료·해시·설정 검증이 실패하면 명시적으로 중단한다.

## 차용한 방법과 추가한 진단

| 근거 | 차용 범위 | 이 실험에서의 확장 / 구분 |
|---|---|---|
| [Cai & Wang, 2020](https://arxiv.org/abs/2006.13318) | layer별 DE, 전파/가중치/활성화 구분, 고정 특징의 반복 전파 | 전체 그래프의 augmented normalized DE와 각 stage를 기록한다. 지역 에너지와 J는 추가 진단이다. |
| [G², ICLR 2023](https://arxiv.org/abs/2210.00513) | layerwise energy dynamics를 비교하는 관점 | G² 모델·깊이 실험을 재현하지 않는다. 첨부 문서의 2006.13318은 G²가 아니라 Cai & Wang이다. |
| [TINED, ICML 2025, Def. 4.2–4.3](https://arxiv.org/html/2412.11180v2) | FT/GP 구분과 DE ratio | 문헌의 DE는 `(1/n) tr(Hᵀ(D−A)H)`이다. `global_E_per_node`와 operation ratio를 기록한다. 증류·teacher injection은 하지 않는다. |
| [LEReg, CIKM 2021](https://arxiv.org/abs/2203.10565) | 지역별 smoothing 차이를 측정하는 관점 | 문헌의 class-based subgraph와 달리 모든 induced closed 1-hop ego를 쓴다. 같은 local 정의를 재현했다고 주장하지 않는다. |
| [TopoOOD, ICML 2024, Eq. 3](https://proceedings.mlr.press/v235/bao24c.html) | node-centered k-hop 에너지 | `k=1`로 고정하며 원본 전체 그래프의 `d+1`을 쓰는 Eq. 3 에너지도 별도 계산한다. 가중 GDE·OOD 학습/검출은 하지 않는다. |
| [Di Giovanni et al., TMLR 2023](https://arxiv.org/abs/2206.10991) | energy dynamics와 Rayleigh quotient 해석 | DE 감소만으로 정보 손실을 단정하지 않는다. 특징 norm으로 나눈 값과 관계 패턴을 함께 본다. |

신규 진단은 기존 프로젝트 감사의 local-incidence 관계 분해를 각 모델 stage에
적용하는 것이다. J의 operation ratio와 label separation은 문헌의 표준 metric이라고
표기하지 않는다. 이 실험은 frozen operator의 진단이며, 학습된 C의 효과나
inductive 일반화, SOTA, 깊은 네트워크의 oversmoothing을 검증하는 실험은 아니다.

## 수학과 실제 측정

원본 무향 물리 엣지는 한 번만 세고, 모든 로컬에서 같은 전역 방향을 쓴다.
`S_v={v}∪N(v)`의 **induced** 그래프이므로 이웃끼리 연결된 엣지도 포함한다.
`C_v=I`, `q_v=B_v H_v`, `r_v=B_vᵀq_v`를 고정한다.

\[
E_v=\|q_v\|_F^2,\quad
J_{\rm shared}(v,u)=\sum_{e\in E_v\cap E_u}\langle q_{v,e},q_{u,e}\rangle,
\]

\[
J_{\rm node}(v,u)=\sum_{i\in S_v\cap S_u}\langle r_{v,i},r_{u,i}\rangle,
\quad J_{\rm distinct}=J_{\rm node}-2J_{\rm shared}.
\]

인접 중심 `(v,u)` 양방향 모두 측정한다. 양방향 중복은 명시되며 독립 표본으로
사용하지 않는다. `J_distinct`는 signed 값이다. 0으로 clipping하거나 cosine처럼
[-1,1]로 제한하지 않는다. `J_node=2J_shared+J_distinct`는 항등식이지 가설 검증 결과가 아니다.

각 J에 `J/(sqrt(E_v E_u)+1e-12)`를 기록한다. 이것도 일반적으로 cosine이 아니다.
raw E와 J는 특징 배율 `a`에 `a²`만큼 변한다. 따라서 다음을 함께 기록한다.

- `E_sym`: **로컬 차수**로 정규화한 에너지. 차수 0의 역제곱근은 0.
- `E_sym_augmented`: **로컬 차수+1**로 정규화한 에너지.
- `E_topoood_1hop`: 같은 로컬 엣지에 **원본 전체 차수+1**을 적용한 TopoOOD Eq. 3.
- `E_scale_free`: raw E / `(로컬 평균을 뺀 특징 norm² + 1e-12)`.
- 나머지 `*_scale_free`: 해당 에너지 / `(로컬 특징 norm² + 1e-12)`.
- `global_*`: 전체 그래프의 raw/normalized/augmented energy, `raw/n`, 특징 norm,
  명시적 norm quotient. 지역 에너지의 합은 엣지가 중복되므로 global E와 같지 않다.

**차수 정규화만으로 특징 크기가 통제되는 것은 아니다.** 명시적인 norm quotient도
epsilon 부근에서는 완전한 배율 불변량이 아니므로 0 분모와 norm을 함께 보존한다.

원래 forward를 그대로 쪼갠다: 입력 → `HW` → 실제 aggregation → 첫 layer ReLU
→ 두 번째 `HW` → 실제 aggregation/logits. 총 7개 이름이며 마지막 aggregated와
logits는 같은 값의 alias다. FT/GP/ACT의 실제 transition은 5개다.

각 layer의 **동일한 projected Z**에 공통 GCN propagation
`P=(D+I)^(-1/2)(A+I)(D+I)^(-1/2)`를 0/1/2회 적용한다. 사이에 W·ReLU·dropout을
넣지 않는다. 세 모델의 Z에 모두 같은 P를 적용하므로 모델에 따른 Z 차이도 비교할
수 있다. MLP의 실제 aggregation은 I이며 negative control이다.
Polynomial-2의 학습된 `I−α L_bar−β L_bar²`와 공통 replay의 `P²`는 별개다.
`P²`와 `L²`를 같은 연산이라고 부르지 않는다.

한 checkpoint당 실제 7 + replay 4 = **11 stage**, 실제 5 + replay 6 = **11 transition**.
전체 9 checkpoint에서 각 99 rows다. I replay는 projected stage를 재사용한다.

E operation ratio는 노드별로 계산해 median/IQR/p10/p90을 기록한다. 원래 E=0인
노드는 분모 미정으로 제외하고 개수·mask를 보존한다. signed J를 직접 나누지 않고
`mean(abs(J_normalized_after))/(mean(abs(J_normalized_before))+1e-12)`를 쓴다.
L2 크기비와 같은 노드/쌍 순서의 pattern cosine도 함께 보존한다.

마지막 secondary 분석에서만 알려진 label의 same/different 쌍별
`J_distinct_normalized` 분포와 평균 차이를 계산한다. `y=-1`은 이 분석에서만
제외한다. test label이 들어가는 post-hoc 진단이며 모델 선택이나 OOD 평가가 아니다.

## 판정과 산출물

`ENERGY_TRACE.md`에서 실제 GP와 공통 I/P/P² replay를 우선 비교한다.
FT·ReLU·MLP에서도 같은 변화가 있으면 aggregation 고유 현상이라는 해석이 약해진다.
E가 줄어도 normalized J의 크기와 패턴이 유지되면 관계가 파괴됐다고 결론 내리지
않는다. 반복 P가 관계를 더 바꾸는지 P→P²로 확인한다. 임의 수치 threshold나
자동 “가설 입증” 판정은 없다. 한 seed의 관측을 일반적인 성능 주장으로 확장하지 않는다.

`stages.{json,csv}`, `transitions.{json,csv}`, 노드/쌍별 FP64 fields NPZ,
각 transition의 ratio/delta NPZ와 undefined mask, 원본 선택 metadata,
checkpoint/graph/source/산출물 SHA256, 숫자 parity evidence, resource calibration을 저장한다.
ratio NPZ의 undefined 값은 NaN이며 대응 mask=false다. JSON의 undefined 통계는 null이다.

## 자원과 실행

모델 forward 및 E/J는 CUDA에서 실행한다. CPU는 기존 파일 검증, 정적 topology
생성/cache, 결과 통계와 저장을 담당한다. 세 데이터셋 topology를 병렬 준비하고,
같은 폭의 모델/stage들을 leading axis로 묶어 GPU 계산한다. 모든 노드·엣지·특징을
유지하고 **특징 축만 정확히 chunk**한다. 후보들의 실제 처리시간과 peak VRAM을
측정해 10% VRAM 여유를 만족하는 가장 빠른 후보를 고른다. OOM 시 모델·그래프·
stage 수를 줄이거나 CPU로 fallback하지 않는다. epoch DataLoader는 필요 없다.

기본 feature chunk 후보 32/128/512는 작은 working set부터 큰 vectorized set까지
측정하기 위한 메모리 단위다. 원본 feature 수가 이보다 작으면 전체 feature 수를
사용하며, packed stage에는 8도 측정한다. calibration 반복 2회는 학습 seed 반복이
아니다. CPU worker는 실제 affinity/quota 범위에서 auto로 정한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.frozen_energy_trace.study --seed 11 \
  --results-root /home/aicompetition07/new-gat/results \
  --output-dir "results/frozen-energy-trace-seed11-$(date +%Y%m%d-%H%M%S)"
```

완료된 full classification이 정확히 하나면 자동 발견한다. 여러 개면 후보를 출력하고
중단하므로 원래 완료 run을 `--classification-run /원래/완료/run`으로 지정한다.
성적·mtime으로 자동 선택하지 않는다. 분석 출력은 새 디렉터리만 허용한다.

별도 `--profile debug`는 이미 완료된 DEBUG fixture checkpoint를 검사할 때만 쓴다.
축소 fixture를 full graph 검증이나 최종 결과로 보고하지 않는다. 실제 전체 모델의
학습이 이 분석으로 완료됐다고 주장하지 않는다.
