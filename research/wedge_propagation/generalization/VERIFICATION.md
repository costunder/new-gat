# Experiment 3 구현·검증 기록

2026년 10월 3일. 완료된 Experiment 2를 입력으로 받아 모든 모델을 고정한 상태에서
새 특징과 입력의 0.25/0.5/1/2/4배를 평가하는 파이프라인을 구현했다.

## 검증 대상

- Source의 full/DEBUG 계약, 완료 상태, 입력 NPZ·수식 코드·checkpoint의 hash와 대응.
- 전체 531개 그래프·모든 경로·16개 특징 실현·원래 모델 구성과 seed의 유지. 숨겨진 subset 없음.
- 원본 특징에서의 평가 재현과 실행 전후 모델 상태·source 파일의 불변성.
- 새 특징 seed, worker 수에 관계없는 재현, 원래 연결·pair·순서의 유지.
- 같은 fresh X에 배율을 적용한 대응 입력과 각 입력의 teacher·target 재계산.
- 3 target × 5 condition의 모든 모델에서 재학습·scalar 재추정·checkpoint 재선택 금지.
- Dense 참조와 출력·스케일 진단의 일치, 경로가 없는 입력·undefined 값·변조 검출.
- Source CSV의 전체 범위, 그래프별 동일 비중 평균 → seed 통계, 악화·동률의 표시.
- 모든 split·3 target·5 condition·6 scenario의 행 수와 graph ID 대응.
- CPU worker·GPU batch의 실제 측정, disjoint-union과 전체 특징 실현·seed의 병렬 평가.
- 실제 측정값으로 Markdown·CSV·PNG/PDF를 생성하고 DEBUG 실행을 명시.

본 평가는 A6000 서버에서 실행한다. 로컬 DEBUG를 full 완료나 최종 성능으로 보고하지 않는다.

## 실제 검사 결과

| 검사 | 결과 |
| --- | --- |
| Ruff | 새 패키지·공통 fixture·네 테스트 파일 통과 |
| 단위·통합 테스트 | **87개 통과**: data 31, frozen 25, report 21, study 10 |
| CUDA 단위 검사 | Dense 출력·paired scale 진단·고정 모델 검증 통과 |
| CUDA DEBUG 전체 평가 | **36개 그래프, 864개 graph/feature/amplitude 조합**, 완료 |
| DEBUG 산출물 | 메시지 4,536행, scale 3,780행, 새 입력 NPZ 5개, Markdown·그림 4쌍 |
| 원본 재현 | 기존 평가 504행 재현. atol=rtol=1e-5 기준 통과 |
| 실행 전후 불변성 | 원본 artifact·모델 상태·수식 의존 코드 hash 동일 |
| 그림 검수 | 실제 PNG 4종과 표·DEBUG 표시·범례 확인 |

87개 전체 검사는 69.41초에 통과했다. 이후 보고서의 teacher 진단 설명을
행렬 span{L,L²,Q}에 대한 Frobenius 잔차로 수정하고, 모델 구성 로그·contract를 보강했다.
영향받는 report 21개와 study 10개를 다시 검사했으며 모두 통과했다.
Source 그림은 미측정 train 위치에 점을 생성하지 않는 것도 실제 좌표로 확인했다.

## 로컬 CUDA DEBUG 측정

최종 결과는 `results/wedge-feature-debug-20261003-02`에 있다.
앞 실행 `results/wedge-feature-debug-20261003-01`과 원본
`results/wedge-learned-debug-20261002-02`도 그대로 보존했다.
RTX 5070 Ti 16GB, PyTorch 2.13.0+cu130, CPU 16개와 RAM 64GiB에서 실행했다.
Teacher 참조는 float64, 모델 평가는 float32이며 TF32는 껐다.

Worker 1/2/4/8/16 및 physical graph batch 12/24/36을 실제 측정했고,
worker 1과 batch 36을 선택했다. 네 scalar 실현과 두 모델 seed를 병렬 처리했다.
시나리오 peak VRAM은 최대 35,231,232 bytes, 약 33.6MiB였다.
원본 파일·모델 상태를 유지한 채 평가·NPZ·CSV·보고서 생성이 22.34초에 완료됐다.
이는 DEBUG 크기의 관측 시간이며 A6000 full 실행 시간의 보장은 아니다.

원본 재현에서 모든 비교 필드의 최대 절대 차이는 3.46e-5였다.
비교 값 크기에 따른 상대 허용값도 포함하므로 각 필드의 atol+rtol 판정을 모두 통과했다.
오차 재현과 별도로 모델 상태·입력 파일은 정확한 hash 일치로 검사했다.
DEBUG의 4-epoch 모델 성능을 최종 연구 결과로 해석하지 않는다.

## 완료 범위

### 서버에서 발견한 재현 검사 문제와 수정

사용자 첨부 `88eb7672-9661-455c-b2de-91f4cc7b35ca`에서 A6000의 원본 재현 검사가
중단된 것을 확인했다. 새 특징 시나리오는 실행되지 않았다.
`path/learned/seed11/family_ood/star-n40`의 raw RMSE는 원본 5.948675171168588,
재평가 5.948534762838067이었다. 차이는 0.000140408이다.

원본은 split별로 최대 240개를 평가했지만 초기 Experiment 3는 원본 531개를 한 배치로
평가했다. 원본 재현에는 원본 split·순서·physical batch·input shape를 복구하고,
각 구성의 일치를 검증하도록 수정했다. 새 특징의 batch 계측과 전체 평가 범위는 유지한다.

추가 CUDA 수치 검사에서 같은 배치라도 경로 합산의 반올림 차이가 있었다.
잔차 RMSE 자체에 대한 상대 허용값은 teacher에 가까운 모델에서 지나치게 작아질 수 있다.
따라서 RMSE를 원본 정답 RMS+epsilon으로 정규화해 기존 atol=rtol=1e-5로 비교한다.
Scalar·메시지 상대 오차·원본 파일/모델 hash 검사는 유지한다.
재현 측정 CSV와 원본 배치 JSON을 실패 전에도 저장한다.

학습 없는 수치 진단에서 실제 full family의 21개 그래프와 기존 teacher 함수를 사용하고,
beta=1.001로 고정한 probe를 100회 평가했다. 정답 RMS로 나눈 RMSE 변화는
star40에서 8.43e-7, star100에서 1.44e-6이었다. 이는 실제 서버 checkpoint의 성능 결과가
아니며, 재현 검사의 단위를 정하는 개발 진단이다. 측정 GPU는 RTX 5070 Ti였다.
수정본의 A6000 full 재실행 결과는 아직 확보하지 않았다.

수정한 study 회귀·통합 테스트 15개와 report 테스트 21개, 총 **36개가 53.49초에 통과**했다.
관측된 RMSE 차이의 정답 크기 기준 허용, 실제 메시지·계수 변조 거부, zero target의 비영점 오차
거부, 원본 batch count/shape 불일치 거부, 전체 평가·보고서·원본 불변성을 검사했다.
Ruff도 통과했다. 실제 CUDA DEBUG 전체 실행은 `results/wedge-feature-debug-20261003-03`에서
완료됐으며 기존 두 결과 폴더를 보존했다. 원본 재현은 batch 12의 split별 구성, 새 특징은
계측으로 선택한 batch로 처리했다. 이 DEBUG 결과를 A6000 full 완료로 표현하지 않는다.

| 항목 | 상태 |
| --- | --- |
| 코드·실행 계약·수식 설명 | 구현 완료 |
| 기존 Experiment 2 변경 | 원본 수식·학습 코드·설정 유지 |
| Experiment 3 추가 학습 | 0 epoch, optimizer update 0 |
| A6000의 full 새 특징 평가 | 미실행. 사용자의 서버 실행으로 결과 확보 |
| 실제 GNN 분류기 학습·평가 | 미실행. Experiment 4 후속 작업 |

실행 명령과 파일 설명은 [README.md](README.md)에 있다.
