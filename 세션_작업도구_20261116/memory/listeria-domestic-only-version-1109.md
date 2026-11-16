---
name: listeria-domestic-only-version-1109
description: "국내만(src_route=domestic) 버전 — 11-09 병행 시작했으나 11-12 사용자가 \"멈추고 다 버려. 국내판은 쓸모없음\"으로 폐기(서버 결과·로그·코드 사본 삭제, MLflow -국내 실험 휴지통, 로컬 결과 휴지통)"
metadata:
  node_type: memory
  type: project
  originSessionId: a19471fc-8fc4-4365-8d63-c14cd9e46718
  modified: 2026-11-12T03:02:50.092Z
---

2026-11-09 사용자 지시로 국내만 버전을 처음부터(1차 window 고르기부터) 지금 판과 병행했다(결과 뿌리 results_dom, MLflow listeria-준비/1차/2차-국내, 코드 scope 모드 lst/scope.py·configs_dom·run_dom). 국내판 진행: 1차 베이스라인 1-CAT-W100-sum(CV 0.0352), 3-0 1위 CatBoost 0.0382, 3-2 로지스틱+이상 점수 0.0406 채택, 3-0b 중요도 35시간째에서 멈춤.

**2026-11-12 폐기**: 식품 특성(t_aw_cl) DB 갱신으로 다시 학습해야 하는 상황에서 사용자 "멈추고 다 버려. 국내판은 쓸모없음" → 서버 프로세스 멈춤, results_dom(5.5GB)·logs_dom·code_frozen/dom_* 삭제, MLflow '-국내' 실험 3개 삭제(휴지통), 로컬 final_results_dom(30.9GB) 휴지통.

**Why:** 사용자 판단으로 국내판 비교가 필요 없어짐.
**How to apply:** 국내판을 다시 제안하지 말 것. 코드의 국내만 모드(scope)는 기본 꺼짐이라 남겨 둠(지금 판 동작 무관). 계획서 '국내만 버전' 절은 지움. 관련 [[listeria-1102-final-integrated-plan]].
