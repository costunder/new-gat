# 저장된 Experiment 4.1 결과 분석 검증

## 구현 범위

완료한 4.1의 13개 CSV/JSON 파일을 읽고 전체 범위·원래 코드·설정·replay·모델 보존 기록을
검증한다. 원래 학습 폴더의 절대 경로나 checkpoint·그래프·특징을 열지 않는다.
Source의 full/debug profile과 모든 seed·manifest·층·개입·split을 유지한다.

각 seed의 원래 층에서 β/κ와 정규화 전 R/Z를 계산하고 저장된 βM/Z와 일치하는지 검사한다.
일차/이차 크기, 각도, κ 분포, 각 층 개입의 accuracy·CE를 표로 만든다.
최적 개입·학습률·checkpoint를 선택하지 않는다.
Source의 모든 입력 파일 hash를 실행 전후에 비교하며 결과는 새 폴더만 허용한다.

기존 classification과 branch_strength의 Python/JSON 소스 및 결과를 바꾸지 않았다.
모델 forward·checkpoint 로딩·optimizer update·GPU 사용은 0이다.

## 검사

- Core 33개: seed별 분해와 평균의 곱 차이, undefined 보존, 분모 1 관계,
  manifest를 seed 안에서 평균, 모든 split·층 위치의 756개 paired 그룹, 잘못된 수치 차단.
- Report 40개: 모든 인자·κ 분포·개입 위치의 범위, undefined·표·seed 구간,
  원래 epoch 표기, 새 파일 쓰기와 덮어쓰기 거부.
- Source 86개: 실제 완료 DEBUG의 13개 scalar artifact만 로딩,
  full 구조 계약·전체 범위·공식 node/split 수·replay·frozen provenance·변조·변경 검사.
- Study 4개: 실제 저장된 DEBUG 전체 실행, 모델/optimizer/checkpoint/CUDA 호출 금지,
  같은 터미널·로그 출력, 원본/기존 출력 보존, 실패 보고.

합계 163개 테스트 통과. Ruff 통과. 새 모듈 5개 Python 3.11 문법 검사 통과.
로컬 실제 실행은 Python 3.13.2 환경이며 서버 Python 3.11 실제 실행과 구분한다.

## 실제 저장된 DEBUG 전체 분석

Input: `results/wedge-branch-strength-debug-20261004-02`.
Output: `results/wedge-branch-analysis-debug-20261003-01`.

- 원래 baseline 144행, treatment 972행, 층 진단 744행, fixed-Z 240행 전체 사용.
- Learned 12개 모델의 두 층, 모든 2seed·2manifest·3개입 위치 유지.
- 분석: seed/층 인자 24행, 강도 추정 204행, paired 층 변화 756행, fixed-Z 추정 384행.
- 저장된 실제 비율과 seed별 분해식의 최대 절대 차이 약 4.78e-9.
- `completed=true`, `actual_data=false`, 원본 보존, forward 0, optimizer update 0.
- 요약·CSV·coverage·source/Git commit·contract·CPU/RAM·같은 터미널 진행 로그 생성.

이 결과는 분석기 연결 검사다. 서버의 분류 성능이나 학습 실패 원인 검증으로 제출하지 않는다.

## 서버 범위

사용자가 제공한 4.1 서버 full 완료 summary는 [기록](../SERVER_BRANCH_STRENGTH_RESULTS.md)에 있다.
새 분석기의 서버 full 실행은 아직 수행하지 않았다.
서버에서는 기존 360/6,750/4,740/1,560행과 120개 모델 보존 기록을 모두 읽는다.
분석 결과는 seed/층 인자 60행, 강도 추정 204행, paired 층 변화 756행, fixed-Z 추정 384행이다.
공식 graph·split을 축소하지 않으며 원래 4.1 평가나 본학습을 다시 시작하지 않는다.
실행 명령과 인자 정의는 [README.md](README.md)에 있다.
