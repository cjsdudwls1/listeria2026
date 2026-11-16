---
name: feedback-always-sido-sgg
description: "사용자 절대 원칙(2026-10-20): 모든 모델의 지역 입력은 시도+시군구 둘 다(sido = sgg_code 앞 2자리, sgg 5자리 범주) — 어떤 실험·변수 선택에서도 빼거나 바꾸지 않음"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2af334fa-b578-4913-a2a5-05aedbdda88a
  modified: 2026-10-20T02:36:21.275Z
---

사용자 '절대 원칙'([종합] 전달, 2026-10-20): 모든 모델의 지역 입력은 **시도 + 시군구 둘 다**(sido = sgg_code 앞 2자리, sgg = 5자리 범주) — 1차·2차와 같게. 바닐라 비교·Optuna 튜닝·변수 선택(중요도로 빼기)·딥러닝·고위험 품목·이상치 탐지 어디서도 시도/시군구 칸을 빼거나 바꾸지 말 것.

**Why:** 예측 단위가 시군구이고 지역 정보는 기본 입력이라 실험마다 달라지면 비교가 무너짐. 사용자가 '절대'라고 못 박음.

**How to apply:** 변수 선택·묶음 빼기 후보에서 범주 4칸(sido·sgg·p2·p3)과 h 는 항상 제외. 범주를 숫자로 바꿔야 하는 모델(로지스틱·랜덤 포레스트)도 sido·sgg 를 둘 다 넣음. 실행 점검에 '모든 실행 config 의 범주 입력 = [sido, sgg, p2, p3]' 확인을 넣는다. 에이전트에 실험을 맡길 때 지시에 그대로 넣는다. 관련 [[listeria-1017-professor-call]], [[listeria-exp1-baseline-decisions]].
