# Experiment 4.1 검증 기록

## 구현

완료 Experiment 4의 전체 source 검증 → 최종 checkpoint 재구성 → 원래 metric 재현 →
고정 Z 메시지 비교 → 각 층/두 층의 실제 forward 개입 → 전체 범위 검사 → CSV·수식·PNG/PDF.

새 optimizer를 생성하지 않고 모든 파라미터의 gradient를 비활성화한다.
Source의 원래 노드·엣지·경로·split·최종 seed·manifest를 모두 사용한다.
기존 classification의 Python/JSON 소스와 결과를 바꾸지 않았다.
Source 파일과 모델 state의 hash를 실행 전후에 검사한다.

## 회귀 검사

- Core 70개: 독립 dense 계산, 8처리·3개입 위치, 두 층의 변경된 현재 Z,
  동일 norm·각도·0 신호·empty graph·CUDA·κ 분포·cache 보호.
- Source 36개: 완료 상태·원래 전체 계약·모든 final seed·selected validation/history,
  실제 그래프·공식 raw SHA·manifest·checkpoint·복사본 경로·변조·실행 후 hash.
- Report 19개: paired seed 통계·manifest 평균·undefined·범위·새 학습 0·실제 PNG/PDF.
- Study 61개: 전체 계약·정확한 coverage·scalar 타입·nonfinite 차단·원본 metric 재현,
  실제 완료 DEBUG checkpoint 재배치·전체 learned pack·두 CPU subprocess 분산 실행.

고유 테스트 합계 186개. Ruff 통과. 새 package 6개 모듈의 Python 3.11 문법 검사 통과.
로컬 실행 환경은 Python 3.13.2/PyTorch 2.13.0+cu130이며 서버 Python 3.11/PyTorch 2.7.1에서의
실제 실행 완료와 문법 검사를 구분한다.

## GPU DEBUG 전체 실행

최종 코드 산출물: `results/wedge-branch-strength-debug-20261004-02`.
RTX 5070 Ti 16GB, 원래 완료한 classification DEBUG fixture checkpoint를 사용했다.
실제 citation 본학습 설정을 바꾸거나 DEBUG를 실제 분류 결과로 제출하지 않았다.

- 전체 3 DEBUG graph, 8조건, 2 최종 seed = 48개 원래 모델 상태.
- Learned 12개 상태에 모든 처리·3위치·2manifest 적용, 전체 forward 372 seed 경우.
- Baseline 144행, treatment 972행, layer diagnostic 744행, fixed-Z 240행.
- 원래 metric 144개 모두 재현. 최대 CE 차이 1.1921e-7, accuracy 차이 0.
- 원래 source 151개 파일과 모델 state 보존. Trainable parameter 0, optimizer update 0.
- 최대 allocated peak VRAM 34,043,904 bytes. DEBUG 전체 실행 24.18초.
  이 작은 fixture의 처리시간·VRAM을 서버 full 성능 예측에 사용하지 않는다.
- 같은 터미널 진행 출력, source/config/Git commit/hash/자원 기록, CSV와 summary 생성.
- PNG/PDF 네 쌍 생성. 실제 그림의 축·범례·DEBUG/source epoch 표시를 확인했다.
- `completed=true`, `actual_data=false`, scope `frozen_branch_strength_diagnostic`.

실제 CPU subprocess 두 개로도 모든 24개 작업·48개 상태·372개 경우를 처리했다.
동일 coverage·metric 재현·모델/원본 보존·같은 터미널의 두 worker 출력과 각 worker 로그를 검사했다.

## 서버 실행 범위

Experiment 4 서버 full 완료 출력은 [기록](../SERVER_CLASSIFICATION_RESULTS.md)에 있다.
사용자가 서버 full 완료 출력을 제공했다. [기록](../SERVER_BRANCH_STRENGTH_RESULTS.md)에
전체 범위·처리시간·실제 데이터·원본 보존·관측 결과를 정리했다.
로컬에서 원본 서버 CSV/checkpoint를 직접 읽은 것으로 표현하지 않는다.
서버 실행기는 전체 3개 citation graph·120개 원래 모델·30개 learned 모델,
최종 5seed·원래 10manifest·모든 경로를 검사한다.
총 2,370개 전체 forward seed 경우와 fixed-Z 관측을 수행한다.
Baseline 360행, treatment 6,750행, layer diagnostic 4,740행, fixed-Z 1,560행이 필수다.

할당 GPU에서 seed pack과 정확한 path chunk 후보를 실제로 측정하고 메모리 여유·처리량으로 선택한다.
모델·그래프·개입을 줄이지 않는다. 여러 GPU가 명시적으로 할당되면 독립 작업을 분배한다.
실제 A6000 처리량·peak VRAM·여러 GPU 동시 실행·서버 전체 평가 완료는 서버에서 확인한다.
실행 명령은 [README.md](README.md), 계산은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.
