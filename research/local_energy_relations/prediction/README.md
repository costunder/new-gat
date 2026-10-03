# 로컬 에너지·집합 사이 관계의 실제 노드 분류

기본 전파 / E 추가 / J 추가 / 둘 다 추가를 두 고정 C에서 비교한다.
모든 조건에 같은 2층 hidden 64와 한 층 최대 두 홉의 기본 전파를 제공한다.
E/J는 실제 logits와 분류 CE·gradient·optimizer update에 연결한다.

- [MODEL_MATH.md](MODEL_MATH.md): 무엇을 계산하고 노드 상태에 어떻게 더하는가.
- [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md): 8조건·전체 데이터·336 run 계약과 해석 범위.
- [RUN.md](RUN.md): 서버 실행과 결과 확인.
- [VERIFICATION.md](VERIFICATION.md): 이번 구현에서 실제 완료한 검사.

앞선 fixed audit와 수신 역복원 실험은 그대로 보존한다.
그 역복원 성공은 제한된 깊이에서 E/J가 분류에 도움이 되는지에 답하지 않는다.
이번에는 C를 새로 학습하지 않는다. 이 단계의 개선은 추가 특징·파라미터의 총 기여이며,
별도 용량 대조나 독립 그래프 일반화의 증거로 설명하지 않는다.

## 서버 실행

[RUN.md](RUN.md)의 단일 터미널 명령을 사용한다.
FULL은 Cora/CiteSeer/PubMed public 전체 그래프, sampling ratio 1,
2층 hidden 64, 각 run 500 epoch, LR 3개×tuning 3seed와 final 5seed다.
Full 본학습은 서버에서 실행한다. 진행 상황과 로그 파일을 동시에 기록한다.
새 출력 폴더를 사용하며 기존 결과를 덮어쓰지 않는다.

## 완료 뒤 보내줄 자료

`completion.json`과 `LOCAL_PREDICTION_SUMMARY.md`만 먼저 보내면 된다.
요약에는 실제 최종 seed 성능·대응 차이·분기 사용·고정 제거·전체 coverage가 들어간다.
원시 전체 metric/개입/branch/resource CSV와 checkpoint·hash는 결과 폴더에 남는다.
