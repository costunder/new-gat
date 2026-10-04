# 구현·검증 기록 — 2026-10-04

## 완료 범위

새 classification의 데이터 연결, 6조건의 실제 분류 forward,
CE backward, 독립 seed의 Adam update, validation 선택,
동결한 checkpoint의 층별 개입, 보고서, 새 폴더로의 재개를 구현했다.
기존 fixed audit와 E/J 실험의 코드·결과는 수정하지 않았다.

2026-10-04 사용자가 서버 FULL 본학습·평가 완료 기록을 제출했다.
162회 tuning과90회 final 학습, 새 update126,000개, 실제 데이터 사용을 보고했다.
[제출 결과와 해석](../../../docs/LOCAL_CONTEXT_CLASSIFICATION_SERVER_FINDINGS_20261004.md)을 기록했다.
서버 raw CSV·checkpoint를 이 컴퓨터에서 독립 재평가하지는 않았다.
아래 로컬 실행은 명시된 DEBUG fixture 검사이며 실제 데이터 성능을 뜻하지 않는다.

## 테스트

- 모델·동결 평가: 42개 통과, CUDA 검사2개 포함.
- 학습·Adam 독립성·완전/부분 재개: 18개 통과, CUDA 검사1개 포함.
- 보고서·전체 행 coverage·입력 검증: 31개 통과.
- 전체 계약·원본 데이터 보존·평가 잠금·실행 조정: 24개 통과.
- 합계115개. DEBUG fixture는 별도 계약이며 FULL 설정을 바꾸지 않았다.

## 전체 DEBUG 실행

명령:

```powershell
.venv-gpu\Scripts\python.exe -B -X utf8 -m research.local_context_coupling.classification.study --profile debug --device cuda --data-root data/wedge-citation --source-dir results/local-energy-DEBUG-20261004-02 --output-dir results/local-context-classification-DEBUG-20261004-01
```

- RTX5070Ti16GB, PyTorch2.13.0+cu130, float32, TF32 끔.
- 원본 DEBUG audit21개를 검증하고 citation 이름의 DEBUG 입력3개만 분류에 사용.
- 모든 원래 fixture 노드·엣지·특징 열·로컬 copy 연결을 유지.
- 총108회, 새 optimizer update324개, 최종 모델36개.
- primary metric108행, 개입 metric432행, 층 진단360행.
- 요약54행, paired 비교162행, 개입 요약432행, 층 요약180행.
- code·graph 보존 검사 통과, 동결 평가 optimizer update0.
- 전체 파이프라인62.5083초. 실제 데이터 여부 `actual_data=false`.
- 학습 조건의108 epoch/seed 기록 모두 CE gradient와 실제 θ update가0보다 컸다.
  절대 gradient 범위1.84428e-6–1.97939e-4, 최대 절대 θ update0.00999949.
  이 수치는 fixture의 미분 연결 검사다. 서버에서도 같은 로그를 기록한다.
- PNG/PDF3종 생성. gain·층별 변화 그림을 직접 열어 표시를 확인했다.

CE 축의 겹치는 눈금을 수정하고 보고서 테스트31개를 다시 통과했다.
최종 코드는 `results/local-context-classification-DEBUG-20261004-02`에서
전체 DEBUG를 다시 완료했다(64.8516초, 108회·324 update).
원시·파생 행 coverage와 source 보존 검사가 모두 통과했다.
학습 조건의108개 기록 모두 gradient·실제 update가0보다 컸다.
수정한 그림도 직접 열어 눈금이 읽히는 것을 확인했다.

## 전체 DEBUG 재개

완료된 위 결과를 `--resume-from`으로 읽고
`results/local-context-classification-resume-DEBUG-20261004-01`에 새로 저장했다.
기존 결과는 보존됐고 추가 optimizer update는0이었다. 전체 coverage도 동일했다.
학습률 선택과 동결 평가의 parameter hash/provenance는 파일 내용까지 동일했다.
재평가 accuracy·gain·θ는 동일했고 float32 CE의 최대 절대 차이는1.19209e-7,
층 진단 숫자의 최대 절대 차이는2.38419e-7이었다. 숫자 CSV의 byte 단위 동일성을 주장하지 않는다.
부분 재개의 Adam 상태와 epoch 예산 검사는 별도 학습 테스트에 포함된다.

## FULL 계약

- 실제 Cora·CiteSeer·PubMed 전체 특징·노드·엣지·public split.
- 같은 두 macro 층, hidden64, dropout0.5.
- `unit/local_degree × off/fixed/learned`의6조건.
- 학습률3개×tuning seed3개×18 dataset/condition =162회.
- 별도 final seed5개×18 dataset/condition =90회.
- 매회500 epoch, 총252회·126,000 seed별 optimizer update.
- GPU의 seed packing과 edge chunk는 실제 처리량·VRAM을 측정해 선택한다.
  그래프와 모델 규모를 바꾸지 않는다.
- 모든 최종 checkpoint를 validation으로 선택한 뒤 test를 연다.

## 제출된 서버 FULL 결과

완료 기록은 전체2,067.5805초(약34분28초), primary270행,
동결 개입1,080행, 층 진단900행과 code/graph 보존을 보고했다.
강도ρ는 초기0.5에서 변했고, 같은 C에서 추가 accuracy 효과의95% 구간은 모두0을 포함했다.
원문과 주요 수치·검토 한계는 위 결과 문서에 보존했다.
