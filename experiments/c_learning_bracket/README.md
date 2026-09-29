# CGAT — C-learning bracket 후보

**C 생성기만 연결 양 끝의 Q/K 대칭 내적으로 교체한 실험이다.**
기존 `c_learning_only` v2.0과 이전 결과는 보존한다.

이번 전달본은 **signal_audit_2의 평가 검증 후속 수정**이다.
calibration에 적용했던 원래 노드 ID 검증을 초기·epoch별 validation과
학습 후 전체·새 문맥 평가에도 연결했다. CE는 평가 지표로 계산하며
backward·optimizer 갱신을 하지 않는다는 문서 설명도 바로잡았다.
확장한 CUDA 통합검사를 포함해 33개 테스트가 통과했다.
C 생성식, 전파, 초기화, 관측 방식과 학습 설정은 직전 판과 같다.
이번에 실제 데이터 calibration을 재실행하지 않았다. 첨부된 실제 측정은
`evidence/prior_signal_audit_2_calibration`의 이전 판 증거다.
**200epoch 본학습은 아직 수행하지 않았으며 C의 학습 효과는 미검증이다.**
이번 결과와 이전 측정의 구분은 `REVIEW_FIXES.md` 앞부분을 읽는다.

## 현재 모델

```text
층 입력 H → Q/K 변환 → 엣지별 대칭 내적 점수 → exp → head별 C
층 입력 H → 기존 value 변환 → C로 가중한 발생행렬 전파 → 출력 변환·ReLU·dropout
```

한 forward의 모든 층에서 같은 부분그래프를 사용한다. 다음 배치에서는 다른
부분그래프를 받는다. 동일한 Q/K 생성 규칙을 모든 노드·부분그래프에 재사용한다.

| 항목 | 이번 구현 |
| --- | --- |
| C 생성 | 독립 Q/K, 대칭 내적 평균, exp, head별 양수 C |
| C 계산의 문맥 | 연결 양 끝의 현재 특징만 사용 |
| beta | 기존 그래프 문맥 조건부 경로 유지 |
| 기본 전파·sampling correction | 기존 수식 유지 |
| 학습 | 기존 분류 CE, learned/fixed 두 조건 |
| 모델·기간 | 8층·hidden 256·8 heads, 200epoch 이상 전체 train pass |
| 보완·ODE·노드/엣지 별도 동역학 | 추가하지 않음 |

따라서 **C 생성은 로컬하지만 모델 전체가 로컬한 것은 아니다.** beta는 그래프
통계를 사용한다. 새 생성기에는 그래프 평균·분산, 내부 최적화, C 평균=1 정규화,
score clipping, exp 이외의 양수 변환이 없다.

## 출처와 적용상의 선택

출처는 Gruber, Lee, Trask의
[Reversible and irreversible bracket-based dynamics for deep graph neural networks](https://arxiv.org/html/2305.15616v3)이다.
논문 부록 A.4/B.3의 차원으로 나눈 내적과 exp 설명을 확인했다.
저자 [SparseNodeEdgeAttentionLayer 코드](https://github.com/natrask/BracketGraphs/blob/f108848fac22516623fe3371869aad6c303fc389/src/attention.py)의
대칭 내적, head 평균, 초기화도 대조했다.

이번 적용은 head 평균을 없애고 head별 C를 유지한다. 점수에는 1/head_dim을
명시적으로 적용하고 Q/K를 독립 Xavier, bias 0으로 초기화한다. 저자 코드의
`scaled_dot` 분기에는 해당 차원 나눗셈이 없고, Q/K weight는 1e-5, bias는 0.01이다.
**논문 전체 모델의 재현이나 보존 성질을 입증한 구현이라고 부르지 않는다.**

## 읽는 순서

1. `MODEL_MATH.md`: 실제 수식, 구현 위치, C와 수신 비중 alpha의 차이.
2. `RUN.md`: 설정 계약, 측정·학습·평가 실행, 결과 파일 해석.
3. `REVIEW_FIXES.md`: 이번 수정, 새 검사·실측 결과와 미검증 항목.
4. `GPT_REVIEW_PROMPT.md`: 다음 검토에서 확인할 범위.

핵심 실행 파일은 `conductance.py`, `operator.py`, `model.py`, `train.py`다.
새 모델 구성은 기존 optimization 생성기나 aggregation model factory를 호출하지 않는다.
기존 파일의 기본값도 변경하지 않는다.
