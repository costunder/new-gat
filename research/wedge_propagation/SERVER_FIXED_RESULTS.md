# 서버에서 확인한 고정 연산 결과

사용자가 2026년 10월 2일 제공한 완료 로그와 SUMMARY.md 발췌에 근거한다.
결과 경로는 `/home/aicompetition07/new-gat/results/wedge-fixed-20261002-144119`다.
전체 NPZ/CSV를 로컬에 복사해 재분석한 결과는 아니다. 서버의 Git commit은 제공된 발췌에 없다.

- GPU: NVIDIA RTX A6000, torch 2.7.1+cu118, Python 3.11.16.
- Full: 198개 그래프, 3,168개 입력, 10,890개 노드, 15,325개 엣지, 53,039개 unordered wedge.
- CPU worker: 계측 후 2개. Graph batch: 계측 후 33개.
- 대수 검사 7종 통과. 모든 그래프의 행렬 항등식 absolute error는 0.
- 연산 비교 완료 로그 elapsed 29.4초. 그림 생성은 이후 완료됐다.

| Family | Graph 수 | span{L,L²} 상대 잔차 중앙값 | spectral norm을 맞춘 Q/L² 출력 상대차 중앙값 |
| --- | ---: | ---: | ---: |
| cycle | 6 | 1.84459e-14 | 0 |
| star | 6 | 1.65675e-13 | 0.123054 |
| grid | 6 | 0.0950461 | 0.130971 |
| ER | 60 | 0.10055 | 0.119721 |
| tree | 60 | 0.14219 | 0.198462 |
| tree+chord | 60 | 0.129991 | 0.1492 |

Cycle과 star는 예상대로 L과 L²의 조합으로 환원된다.
Star의 Q/L² 출력 차이는 추가 L 항으로 설명되므로 상대 잔차가 0에 가까운 것과 모순되지 않는다.
나머지 네 family에서는 이 두 고정 basis에 최적으로 맞춰도 잔차가 남는다.
고정 Q의 차이는 정확한 degree 보정항에서 온다.
이 표는 연산 차이이며 정보 손실률·예측 정확도·C2 학습 결과가 아니다.
