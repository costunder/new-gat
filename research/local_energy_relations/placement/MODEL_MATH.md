# 위치별 E/J 분류 모델의 수식과 forward

이번 변경은 **동일한 이차 에너지 E와 쌍선형 관계 J를 어느 층에 더할지**다.
기본 전파·발생행렬·C·분모를 바꾸지 않는다. 기존 계산 코드는
[`../prediction/operators.py`](../prediction/operators.py)를 그대로 재사용한다.

## 1. 서로 다른 1홉 local

원래 그래프의 중심 v에 대해 S_v={v}∪N(v)다. S_v 안의 모든 원래 엣지를 포함하는
induced local을 쓴다. 이웃끼리 연결도 포함한다. B_v는 엣지×local 노드 발생행렬이며,
같은 물리 엣지는 모든 local에서 원래 노드 ID에 따른 같은 방향을 쓴다.
R_v는 전체 현재 특징 Z에서 S_v의 행을 선택한다.

\[
g_v=B_vR_vZ,\qquad q_v=C_vg_v,\qquad
d_v=B_v^\top q_v,\qquad L_v=B_v^\top C_vB_v.
\]

C_v는 고정 양의 대각 행렬이다. `unit`은 c_(a,b)=1,
`local_degree`는 c_(a,b)=2/(deg_v(a)+deg_v(b))다.
degree는 해당 induced local의 degree다. 이번 실험에서는 C를 학습하지 않는다.

## 2. 모든 조건이 쓰는 기본 전파

실제 엣지로 연결된 중심 v→u마다 공통 원래 노드의 d_v를 수신 local에 더한다.
수신 중심 행의 합은

\[
Y_u[u]=\sum_{v\sim u}d_v[u]
      =\sum_{b\sim u}w_{ub}(Z_u-Z_b),\qquad
w_{ub}=\sum_{v\in N(u)\cap(\{b\}\cup N(b))}c_{v,(u,b)}.
\]

이 행들의 연산자를 A_0, 대각을 D_0라 하면 양의 degree 행에서
P_C=I−D_0^(-1)A_0다. 고립 행은 identity다. P_C는 비음수 행 확률 행렬이며,
`unit`의 w_ub는 1+공통 이웃 수다. `local_degree`의 w는 방향에 따라 달라질 수 있다.
Sparse P_C를 두 번 적용해 모든 조건에 같은 최대 두 홉 support를 제공한다.

\[
\boxed{U_{\rm base,\ell}=(1-\alpha_\ell)Z_\ell+
          \alpha_\ell P_C^2Z_\ell},\qquad
\alpha_\ell=\operatorname{sigmoid}(a_\ell),\quad a_{\ell,0}=0.
\]

## 3. 내부 이차 에너지 E

\[
E_v=\operatorname{tr}((R_vZ)^\top L_v(R_vZ))
   =\sum_{e\in E_v}c_{v,e}\|g_{v,e}\|_2^2\ge0,
\qquad
\boxed{e_v=\frac{E_v}{2F\sum_e c_{v,e}}}.
\]

F는 현재 층 채널 수다. 엣지 없는 local은 E_v=e_v=0이다.
분모가 topology에만 의존하므로 e_v도 정확한 이차형식이다.
q의 제곱 norm Σc²||g||²는 일반적인 C에서 이 E와 다르다.

## 4. 서로 다른 local 사이 쌍선형 관계 J

B_v의 열을 원래 전체 N개 노드 좌표로 확장한 것을 \(\bar B_v\)라 하자.
서로 연결된 중심 v,u에 대해

\[
K_{vu}=\bar B_v\bar B_u^\top,\qquad
J_{vu}=\operatorname{tr}(q_v^\top K_{vu}q_u)
      =\sum_{a\in S_v\cap S_u}\langle d_v[a],d_u[a]\rangle.
\]

K는 |E_v|×|E_u|이며, 서로 다른 발생행렬을 공통 원래 노드로 대응시킨다.
같은 층의 local들이 같은 projected 채널 좌표를 쓰므로 J_vu=J_uv다.
J의 부호를 유지하며 항상 양의 에너지라고 부르지 않는다.

\[
t_v=\operatorname{tr}(L_v^2)=\sum_{a\in S_v}\delta_{v,a}^2
                       +2\sum_{e\in E_v}c_{v,e}^2,
\qquad \delta_{v,a}=\sum_{e\ni a}c_{v,e},
\]
\[
\widehat J_{vu}=\frac{J_{vu}}{F\sqrt{t_vt_u}},\qquad
\boxed{j_u=\frac1{|N(u)|}\sum_{v\sim u}\widehat J_{vu}}.
\]

고립 중심의 j는 0이다. Feature RMS·절댓값·ReLU를 J 계산에 넣지 않는다.
이번 J는 같은 층의 서로 다른 local 관계다. 서로 다른 층의 채널을 직접 내적하는 항을 추가하지 않는다.

## 5. 위치별로 활성화하는 항

Placement p의 활성 층은 hidden={0}, output={1}, all={0,1}다.
Variant가 E를 사용하는지 나타내는 \(m_E\), J를 사용하는지 나타내는 \(m_J\)를 두면

\[
\boxed{U_\ell=U_{\rm base,\ell}
 +\mathbf1_{\ell\in p}m_Ee_\ell w_{E,\ell}^\top
 +\mathbf1_{\ell\in p}m_Jj_\ell w_{J,\ell}^\top}.
\]

Base는 m_E=m_J=0, within은 (1,0), between은 (0,1), both는 (1,1)이다.
활성 층의 벡터만 parameter로 등록하며 해당 층의 E/J만 계산한다.
각 벡터는 0으로 초기화한다. 같은 C·seed의 모든 초기 forward가 동일하다.
첫 CE backward에서 활성 벡터에 gradient가 흐르고, 벡터 업데이트 뒤 E/J를 통한
projection gradient도 흐르는지를 별도 DEBUG 검사한다.

## 6. 실제 두 층 forward

\[
H_0=X,\quad Z_\ell=\operatorname{Dropout}_{0.5}(H_\ell)W_\ell,
\quad H_1=\operatorname{ReLU}(U_0),\quad\text{logits}=U_1.
\]

입력 D→hidden 64→클래스 K다. W는 bias 없는 Xavier projection이다.
예를 들어 `both__hidden`은 첫 projected 64채널에서 E/J를 계산해 노드 상태에 더하고,
ReLU·두 번째 projection·기본 전파를 거쳐 logits를 만든다.
`both__output`은 첫 층을 기본 전파로 보내고, 둘째 projected K채널에서 E/J를 계산해 logits에 더한다.
`both__all`은 두 층의 현재 Z에서 각각 새로 계산한다.

Train-mask CE가 W·α·활성 lift를 학습한다. Adam은 500 epoch, epoch마다 전체 그래프 update 한 번이다.
Projection weight decay만 5e−4다. 평가 CE에 L2를 더하지 않는다.
Auxiliary reconstruction loss·teacher·learned C는 없다.

각 seed의 기본 parameter 수는 \(D H+H K+2\), H=64다.
활성 항 수 b=0/1/1/2와 위치 폭 f_p=0/H/K/(H+K)에 대해 총수는

\[
\boxed{\#\theta=D H+H K+2+b f_p}.
\]

따라서 위치 비교는 활성 lift 용량도 함께 바뀐다. Z→aZ에서 base는 a배,
E/J는 a²배다. 분기의 신호 규모와 사용 여부를 실제 norm·lift·고정 제거로 확인한다.
두 스칼라 lift는 원래 메시지 전체를 보존하는 표현이 아니다.

고정 제거는 같은 checkpoint의 활성 항을 0으로 만든다. 첫 층 제거 후 둘째 층 상태와 E/J를 재계산한다.
재학습 결과와 frozen 제거 반응을 구분하며, 분류 개선을 사이클 성분 복원으로 해석하지 않는다.
