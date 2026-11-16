---
name: feedback-no-ap-anywhere
description: "실험 결과에 AP(average_precision_score)는 보고 표뿐 아니라 결과 파일에도 적지 않음, 지표는 trapz PR-AUC 하나 (2026-11-04 지시)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: eef9de9c-6d63-42dc-b5a4-262af470cc8d
  modified: 2026-11-03T23:28:47.607Z
---

리스테리아 실험의 점수 기록·보고에 AP(`average_precision_score`)를 어디에도 적지 않는다 — 보고 표, 결과 파일, 동점 판 보조 지표 모두.

**Why:** 2026-11-04 계획 검토에서 결정 12 권고 'AP 는 결과 파일에만'을 보고 사용자가 "AP는 파일에만 적음->그 어디에도 적지마". 지난 실험이 AP 를 PR-AUC 라 부르는 실수를 해 지표가 헷갈렸고(11-02), 주 지표는 하나여야 한다는 10-30 지시("헷갈리게하지")의 연장.

**How to apply:** 새 실험 코드·결과 파일·보고서에는 trapz PR-AUC(구간 가중)와 F2·Recall·Precision 등 정해진 칼럼만. 동점 많은 판은 AP 대신 '동점 많음 — trapz 부풂' 표시(결정 46). 지난 실험 수치가 AP 였다는 사실 설명은 계획서 근거로만 남김. 관련 [[listeria-1102-final-integrated-plan]], [[feedback-report-f2-figures]].
