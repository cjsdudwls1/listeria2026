---
name: feedback-report-f2-figures
description: "보고서 작성 규칙(2026-10-19): 점수는 F2 중심 + Recall·Precision·혼동행렬, PR-AUC 는 주 점수 아님(전체 요약표에만), 표·그림 적극·그림은 출력한 시각화 이미지 그대로, 시스템 구현 부분 안 씀, 1차 보고서는 모델을 골라 주지 않고 표로 다 보여 주는 '골라 주세요' 보고서"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 79b84d9c-2d6f-4801-81de-e968c256423d
  modified: 2026-10-19T06:23:55.502Z
---

2026-10-19 사용자([종합] 세션, 보고서 작성 지휘 중):
- "보고서 작성시 표 그림 적극 활용하고, 그림 활용 시 출력된 시각화(seaborn 등) 그대로 복붙하고, 시스템구현 부분은 작성안해도되고"
- "생각해보니 알고리즘 선별은 안하니까 prauc가 쓰일일은 없을듯. 근데 보여주긴해야하니까 암튼 점수는 f2를 보여주면될듯. 그이외에 recall, precision, 혼동행렬등 당연히 보여줘야하는건 보여줘야하고"
- 10/22 발표 모델 고르는 법을 묻자: "골라주세요~하는 보고서를 작성하는거임. 표로 다 보여주면 됨"

**Why:** 보고서 독자(원장님·교수님)가 모델을 고르는 보고서이고, 알고리즘 선별을 하지 않으니 순위용 PR-AUC 보다 판정 결과(F2·재현율·정밀도·혼동행렬)가 중요.

**How to apply:** 비교표 주 점수 = F2(h 1~60 평균; 판정 기준값은 2026-10-20 부터 **검증 Recall ≥ 0.8 중 Precision 최대** — 그 전엔 검증 F2 최대), 같이 Recall·Precision·TP·FN·FP·TN. PR-AUC·ROC-AUC 는 '모델 성과 발표 예시.docx' 식 전체 요약표에만. 그림은 matplotlib 등 출력 이미지를 그대로 붙임. 원본 양식의 시스템 구현 장은 생략. 점수 표현은 [[feedback-no-stddev-caveats]] 를 따름. 관련 [[listeria-reports-1019]].

**2026-10-20 갱신([종합] 전달)**: 판정 기준값 = 검증 Recall ≥ 0.8 인 기준값 중 Precision 최대(F2 최대를 대체), 모델·창 선정 = 검증 PR-AUC. 실험 방식은 3차(시간순 교차검증)로 바뀜 — [[listeria-1020-professor-call]].
