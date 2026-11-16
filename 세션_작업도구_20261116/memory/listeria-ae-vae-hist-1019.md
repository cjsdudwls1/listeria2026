---
name: listeria-ae-vae-hist-1019
description: "10-19 사용자 결정: AE·VAE 이상 탐지(표 입력·기상 시퀀스 둘 다, A3 CatBoost 에 점수 1칸) + 이력 추가(1년 log 검사 수 3칸·불확실성 3칸), AutoGluon 은 나중; 실험 세션에 사양 전달"
metadata:
  node_type: memory
  type: project
  originSessionId: af234a63-7c5d-4cfa-a1a7-e2ca98d874ba
  modified: 2026-10-19T05:05:15.959Z
---

2026-10-19 gpu-server-usage-memo 세션(af234a63) 사용자 결정(AskUserQuestion):
- AutoGluon·Optuna 로 변수·모델 자동 선택은 "나중에". 이유 설명함: 검증 순위가 시험과 안 맞음(P3 검증 1위→시험 14위, 위약 P1p 검증 3위), 후보 차이 0.005 < 시드 표준편차 0.004~0.009.
- **AE·VAE 이상 탐지**(교수님 `Listeria/1차 고도화 전략/readme.md` 8절 E09·E10, 10-12 통화 "정상만 학습, FN 보완"): 입력 = **둘 다 따로**(표 = A3 입력 200칸+범주 4 / 기상 시퀀스 50일×38채널), 붙일 모델 = **A3 CatBoost**(+점수 1칸).
- **이력 추가**: **1년 log(1+검사 수) 3칸 + 불확실성 3칸**. 과거 검사 건수 log 는 이미 A2(5년, 3수준 품목군 3단계·시군구·단위)·스무딩 (s+k·p0)/(n+k) 로 들어가 있음 — 사용자가 잊었을 수 있으니 물으면 A2 부터 설명.
- 내 제안(사용자 반대 없음, 실험 세션에 전달): 정상 = 학습 구간 불검출 행, 학습 행 점수 5겹 교차 적합, AE=MSE·VAE=복원+KL, 시퀀스 AE 1D 합성곱, 점수 단독·A3+1칸·위약판; 불확실성 = 1년 스무딩 검출률 베타 사후 표준편차 sqrt(m(1−m)/(n+k+1)), A3 k*; 1년 검사 수 연도별 추세(날짜 시계) 점검.
- **10-19 실험 세션 수락**: 구현·실행 모두 exp/2_고도화 src 에서(이 PC RTX 2060). ID 안: 이력 A3_L1·A3_U1·A3_L1U1 / 점수 단독 S_AEt·S_VAEt·S_AEs·S_VAEs / A3_AEt·A3_VAEt·A3_AEs·A3_VAEs + 위약 *_p 4개(점수를 달력 연도 안에서 섞음), 시드 3, 정규화도 학습 구간 불검출 행에서만, 하이퍼파라미터 기본값. 배치 9 뒤 구현·독립 검증 → 돌리기 전 알림 → 끝나면 이쪽에서 rescore_T.py 로 (가) 재채점.

- 10-19 15시께 사용자가 실험 세션에서 직접 'AutoGluon 연결 시험부터'를 골라 서버에서 A3 입력 파일럿 1회만 돌림(본 비교 아님). AE·이력은 이 PC 에서 14:15 부터 구현 중 — 자원 겹침 없음. 이 세션의 '나중에'와 엇갈렸으나 사용자 본인 선택이라 그대로.

관련 [[listeria-1017-professor-call]], [[listeria-1012-professor-call]], [[listeria-test-T-basis-scoring]], [[listeria-name-external-features-1019]].
