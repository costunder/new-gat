# 로컬 이차 에너지·쌍선형 관계를 사용하는 분류 모델

이 모델은 **두 홉을 전달하는 동일한 기본 전파에 내부 에너지 E와 집합 사이 관계 J를 더해,
실제 분류 CE로 학습하는 네 가지 조건**을 비교한다. C는 이번 단계에서 고정한다.
앞선 전체 수신 역복원 성공이 이 모델의 분류 기여를 보장하지는 않는다.

## 1. 같은 물리 그래프 위의 서로 다른 로컬 집합

원래 단순 무방향 그래프의 노드 수를 N, 물리 엣지 수를 M이라 하자.
중심 v의 집합은 S_v={v}∪N(v)이고, 양끝이 S_v에 있는 모든 원래 엣지를 포함한다.
이웃끼리 연결도 포함하며, 없는 엣지를 만들거나 로컬 밖의 특징을 로컬 안에 있다고 가정하지 않는다.

B_v는 엣지×노드 발생행렬이다. 모든 로컬에서 같은 물리 엣지는 원래 노드 ID 순서에 따른
같은 방향을 사용한다. R_v는 전체 특징의 S_v 행을 선택한다. 각 층의 현재 투영 특징 Z에서

\[
g_v=B_vR_vZ,\qquad q_v=C_vg_v,\qquad
d_v=B_v^\top q_v,\qquad L_v=B_v^\top C_vB_v
\]

를 계산한다. C_v의 두 고정 규칙은 c_e=1인 `unit`과
c_(a,b)=2/(deg_v(a)+deg_v(b))인 `local_degree`다. degree는 해당 induced local에서 계산한다.
이 C를 attention으로 학습했다거나 선행 synthetic checkpoint를 옮긴 모델이라고 설명하지 않는다.

## 2. 네 조건 모두 같은 두 홉 기본 전파

원래 엣지로 연결된 중심 v→u마다 공통 노드의 d_v를 수신 로컬 S_u의 해당 행에 더한다.
전체 수신 합을 Y라 하면 수신 중심의 행은

\[
Y_u[u]=\sum_{v\sim u}d_v[u]
       =\sum_{b\sim u}w_{ub}(Z_u-Z_b),
\]

\[
w_{ub}=\sum_{v\in N(u)\cap(\{b\}\cup N(b))}c_{v,(u,b)}>0.
\]

이 행들을 모은 행렬을 A_0, D_0=diag(A_0)라 하자. D_0가 양수인 행에서
P=I−D_0^(-1)A_0이며, 고립 노드에서는 P의 해당 행이 자기 특징을 그대로 돌려준다.
P는 비음수 행 확률 행렬이다. `unit`이면 w_ub=1+공통 이웃 수이고,
삼각형이 없는 그래프에서는 일반적인 random-walk 전파와 같다.
`local_degree`는 일반적으로 방향에 따라 w_ub가 다를 수 있다.

\[
\boxed{U_{\rm base}=(1-\alpha)Z+\alpha P^2Z},
\qquad\alpha=\operatorname{sigmoid}(a),\quad a_{\rm init}=0.
\]

센터 수신 행 하나만 쓰면 기본 전파의 특징 support는 한 홉이다.
이번에는 모든 조건이 P를 두 번 적용해 한 층에서 최대 두 홉의 특징을 사용한다.
추가 J가 한 층에서 두 홉을 참조한다는 이유만으로 좋아지는 경우와 비교하기 위한 공통 기본 모델이다.
P·P²를 dense하게 만들 필요는 없고, 캐시한 sparse P를 두 번 적용하면 된다.
행 확률 성질을 Euclidean contraction이나 전체 비선형 학습 수렴 보장으로 확대하지 않는다.

## 3. 내부 이차 에너지 E

\[
E_v=\operatorname{tr}\bigl(Z_v^\top L_vZ_v\bigr)
   =\sum_{e\in E_v}c_{v,e}\|g_{v,e}\|_2^2\ge0.
\]

F는 현재 층의 채널 수다. topology에서 한 번 구한 s_v=Σ_e c_(v,e)를 사용해

\[
\boxed{e_v=\frac{E_v}{2Fs_v}}
\]

를 예측 특징으로 사용한다. 엣지가 없어 s_v=0인 로컬은 E_v=e_v=0이다.
분모는 특징에 의존하지 않으므로 e_v도 정확한 이차형식이다.
q의 제곱 norm은 Σ c²||g||²이고, 일반적인 C에서 E와 다르다.

## 4. 집합 사이 쌍선형 관계 J

각 발생행렬의 열을 원래 N개 노드 좌표로 확장한 것을 \(\bar B_v\)라 하자.
서로 연결된 중심 v,u에 대해

\[
K_{vu}=\bar B_v\bar B_u^\top,\qquad
J_{vu}=\operatorname{tr}(q_v^\top K_{vu}q_u)
      =\sum_{a\in S_v\cap S_u}\langle d_v[a],d_u[a]\rangle.
\]

K는 |E_v|×|E_u| 직사각형이다. 양쪽의 서로 다른 발생행렬을 실제 공통 노드로 대응시킨다.
같은 단계의 두 로컬은 같은 projected Z 채널 좌표를 사용하며 J_vu=J_uv다.
특징을 양의 행렬로 바꾸거나 내적의 음수 성분을 버리지 않는다.

로컬 weighted degree를 δ_(v,a)=Σ_(e incident a)c_(v,e)라 하면

\[
t_v=\operatorname{tr}(L_v^2)
   =\sum_{a\in S_v}\delta_{v,a}^2+2\sum_{e\in E_v}c_{v,e}^2.
\]

이 topology 상수로 정규화하고 수신 중심에서 이웃 관계를 평균한다.

\[
\widehat J_{vu}=\frac{J_{vu}}{F\sqrt{t_vt_u}},\qquad
\boxed{j_u=\frac1{|N(u)|}\sum_{v\sim u}\widehat J_{vu}}.
\]

고립 중심의 j는 0이다. 실제 엣지로 연결된 중심은 양쪽 t가 양수다.
J는 부호 있는 쌍선형 관계이며 항상 양의 에너지라고 설명하지 않는다.
J_shared와 J_distinct를 별도로 진단할 수 있지만 J_distinct=J−2J_shared이므로
세 종속 통계를 모두 예측 입력으로 넣지 않는다. primary 입력은 J 하나다.

## 5. 에너지·관계를 실제 노드 상태에 연결

각 층에 학습 벡터 w_E,w_J∈R^F를 두고 활성화한 항만 만든다.

\[
\boxed{U=U_{\rm base}+e\,w_E^\top+j\,w_J^\top}.
\]

| 조건 | 실제 추가 노드 특징 | 층별 추가 파라미터 |
| --- | --- | ---: |
| base | 없음 | 0 |
| within | e w_Eᵀ | F |
| between | j w_Jᵀ | F |
| both | e w_Eᵀ+j w_Jᵀ | 2F |

비활성 벡터는 parameter로 만들지 않는다. 활성 벡터는 0으로 초기화하므로
같은 C·projection·dropout을 가진 네 조건의 초기 forward는 동일하다.
첫 CE backward에서 벡터의 gradient가 흐르고, 벡터 업데이트 뒤 에너지·관계를 통한
projection gradient도 흐르는지를 비퇴화 DEBUG fixture로 확인한다.
실제 gradient가 작은지, 벡터가 실제로 바뀌고 예측에 사용되는지는 학습 기록과 고정 개입으로 확인한다.

Z→aZ이면 기본 전파는 a배, e/j는 a²배다. Feature RMS나 graph-wide feature 정규화는 넣지 않는다.
따라서 작은 projected 특징에서 에너지·관계 분기가 작게 남을 수 있다.
분기 norm·base 대비 비율·벡터 norm을 기록하며, 음성 결과를 이차형식 원리의 일반적인 실패로 해석하지 않는다.

## 6. 두 층 분류와 학습 대상

\[
Z_\ell=\operatorname{Dropout}_{0.5}(H_\ell)W_\ell,\qquad
H_0=X,\quad H_1=\operatorname{ReLU}(U_0),\quad\text{logits}=U_1.
\]

입력 D→hidden 64→클래스 K이며, projection bias·추가 output projection은 없다.
두 층의 channel은 64와 K, 직접 특징 support는 최대 네 홉이다.
각 층의 현재 Z에서 E/J를 새로 계산한다. 이전 고정 reference H0/H1/H2를 학습 상태로 재사용하지 않는다.

Train mask의 CE로 W·α·활성 lift 벡터만 학습한다. C·P·topology 분모는 고정한다.
Adam, 500 epoch, projection weight에만 L2 5e−4를 적용하고 α·lift 벡터에는 weight decay를 넣지 않는다.
평가 CE에는 L2가 포함되지 않는다. Auxiliary reconstruction loss·teacher·learned C는 없다.
총 파라미터는 D·64+64·K+2+b(64+K), b=0/1/1/2다.

앞선 전체 Y에서의 전역 반복 역복원과 이 제한된 깊이의 로컬 분류는 서로 다른 검사다.
E/J가 전역 Y의 함수라는 사실만으로 이 모델의 기여가 0이라고 하거나,
반대로 분류 개선을 복구 불가능한 메시지의 복원이라고 설명하지 않는다.
