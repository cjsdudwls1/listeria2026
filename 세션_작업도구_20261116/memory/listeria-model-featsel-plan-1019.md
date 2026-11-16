---
name: listeria-model-featsel-plan-1019
description: "2026-10-19 사용자 요청: 시계열 모델 추가(TCN·Prophet·Informer/PatchTST·N-BEATS/N-HiTS)와 최적 변수 조합 탐색(SFS·Optuna·PyCaret·AutoGluon)의 현황·내 제안 순서·결정 대기 5개, .venv-exp2 설치 패키지 실측"
metadata:
  node_type: memory
  type: project
  originSessionId: 06e0faba-94f2-42b3-9a9d-818a4afb5351
  modified: 2026-11-06T05:54:24.438Z
---

2026-10-19 [실험 기록 최고 결과 조회] 세션에서 사용자가 "모델은 또 실험해야 해(TCN, Prophet, Informer/PatchTST, N-BEATS/N-HiTS 등), 최적 변수 조합도 찾아야 함(SequentialFeatureSelector, Optuna, PyCaret, AutoGluon 등), 현황 파악 후 어떻게 할지 알려줘" → 계획만 설명함, 실행·설치는 안 함.

**제안(사용자 답 대기)**: 순서 = ⓪ 평가 틀(검증 한 해·검출 122건 대신 시간순 접기 여러 개 평균, 같은 접기·시드끼리 짝 비교, 위약판보다 나을 때만 채택, 시험은 맨 끝 1회) → ① 변수 조합(CatBoost 고정, 묶음 30~40개 단위: 위약 묶음과 함께 순열 중요도 선별 → 전진·후진 확인) → ② 알고리즘 비교(같은 변수·접기: CatBoost·LGBM·XGB·로지스틱·AutoGluon 참고선 / LSTM·트랜스포머·TCN·PatchTST 에 A3 등 정적 변수 결합 / 스태킹) → ③ 하이퍼파라미터(Optuna, 사용자 10-19 규칙대로 마지막) → ④ 시험 (가) 판 1회·시드 10.
- 모델 판단: TCN·PatchTST 포함, Informer 제외(창 50일이라 이미 돌린 트랜스포머와 사실상 같음), N-BEATS/N-HiTS·Prophet 제외(한 계열 미래값 예측용, 우리 정답은 검사한 날만 있는 0/1 — 방식 B 정답 칸 87.8% 빈칸; 쓰려면 정답 재정의 필요).
- 도구 판단: SFS 는 칸 단위라 비용·구조 안 맞음 → 같은 알고리즘을 묶음 단위로 직접 구현, Optuna 는 HPO 에, PyCaret 미사용(미설치·무작위 CV), AutoGluon 은 알고리즘 비교에.
- 근거: 시드 표준편차 0.003~0.009, 위약판 P1p 가 진짜 P1 보다 검증·시험 모두 높음, 검증·시험 순위 엇갈림, 9-29 옛 실험 2(버린 데이터) Optuna 259시도 검증 0.022~0.033 → 시험 0.010~0.014. 기존 딥러닝 4개는 A3 이력 없이 기상+범주만 받아 A3 와 공정 비교 아님. 1차 XGB·LGBM 은 1~3라운드 조기 종료.
- 결정 대기: ① 순서 ② 선정 기준을 접기 평균+위약 초과로 바꿀지 ③ 모델 범위 ④ 제품명 N1·N2 를 후보에 넣을지 ⑤ 실행 주체 = exp/2_고도화 담당 [1번 베이스라인 실험 다시] 세션에 계획 전달(추천).
- **사용자 답(10-19 오후)**: "하이퍼파라미터는 조정 안 해도 됨" → ③ HPO 단계 삭제(순서 = 평가 틀 → 변수 → 알고리즘 → 시험 1회). "모델 범위 동의 일단 하지 말고 대기" → 모델 범위 보류, 진행 금지. "선정 기준은 지금까지 검증 한 해로 한 거 아님?" → 맞다고 답함: 2차 선정 = 검증 2024-07-01~2025-06-30 한 해의 구간 가중 PR-AUC 시드 3개 평균, 같은 해를 CatBoost 조기 종료(e2models.py eval_set=pva)·판정 기준값(검증 60개 h F2 최대, e2eval.py)에도 씀. 아직 미답: 선정 기준 바꿀지, N1·N2, 실행 세션.

- **10-19 15시께 사용자 지시([1번 베이스라인 실험 다시] 세션, [종합]이 확인)**: "cat 만 쓰지 말고 트리 3종·BiLSTM·TCN·PatchTST 까지, 묶음 단위 변수 선택 + AutoGluon 비교·앙상블·스태킹, 하나로 고르지 말고 상위권 제시" + AutoGluon "연결 시험부터"(서버에서 A3 입력 1회) → 위 '모델 범위 보류'를 대체. [종합] 결정: 순서는 AE·VAE 실행 → 1차 보고서 → 모델 넓히기(2차 단계). **→ 같은 날 사용자 "너무 복잡하게 시켰다… 고도화 전략 10절 기록표처럼, 모델은 xgb든 cat이든 선정근거만"으로 모델 넓히기·AutoGluon 취소([[listeria-reports-1019]]).**

**환경 실측(10-19)**: `리스테리아ML/.venv-exp2`(Python 3.12.14, uv 로 만들어 pip 없음 → importlib.metadata 로 확인) 있음 = torch 2.13.0+cu126·optuna 5.0.0·autogluon.tabular 1.6.3·catboost 1.2.10·lightgbm 4.7.0·xgboost 3.3.0·scikit-learn 1.9.1·mlflow 3.16.1; 없음 = neuralforecast·darts·prophet·pycaret·tsai·transformers·sktime·mlxtend·shap·autogluon.timeseries·statsforecast. 아플라톡신 팀 코드(toxin_2026/01_아플라톡신/train)에 Optuna 탐색(시드 평균 목적함수·sqlite 이어하기·MLflow)·InceptionTime·MiniRocket·BiLSTM 있음, **TCN 은 registry.py 설명 예시뿐 구현 없음**. 실행 시간(이 PC): CatBoost 1회 약 1분, 딥러닝 1회 약 1.5분.

**How to apply:** 사용자 답을 받으면 그 내용대로 [1번 베이스라인 실험 다시] 세션에 전달하거나 사용자 지시를 따름. 관련 [[listeria-task2-model-window-exp]], [[listeria-1017-professor-call]], [[listeria-test-T-basis-scoring]], [[gpu-server-usage]].

**결말(10-19 15시 반, [종합] 전달)**: 사용자가 모델 넓히기(BiLSTM·TCN·PatchTST·AutoGluon) 취소, 선정 기준 변경·표준편차 기반 재비교 안 함("점수만 보고") → 위 제안(평가 틀·변수 탐색·모델 추가)은 채택 안 됨. 틀은 교수님 「1차 고도화 전략」 10절 기록표로 정리([[listeria-reports-1019]]). 다시 제안하지 말 것.

**11-06 갱신:** 이 취소는 10-19 그때 실험 한정이었음. 최종 계획(최종실험_계획_20261102)에서는 사용자가 "TCN 넣고"로 3-0 딥러닝에 TCN 을 넣음(근거 = 교수님 「1차 고도화 전략」 readme 6·7절 "TCN 또는 LSTM"). 옛 취소를 새 계획의 '담당자 결정'으로 넘겨 적지 말 것 — 계획마다 사용자에게 다시 물을 것.
