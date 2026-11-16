---
name: feedback-per-h-metrics-always
description: 모든 실험 결과는 h 1~60 별 × 평가지표별 점수를 바로 볼 수 있어야 함(보고서용) — 판마다 h_metrics 표 저장·내보내기 명령 (2026-11-07)
metadata:
  node_type: memory
  type: feedback
  originSessionId: a19471fc-8fc4-4365-8d63-c14cd9e46718
  modified: 2026-11-07T06:52:10.492Z
---

모든 판(실험 결과)에서 h 1~60 별로 각 평가지표(PR-AUC·Recall·Precision·F1·F2·혼동행렬·상위 q%·배수 등) 점수를 사용자가 원하면 바로 볼 수 있어야 한다 — 사용자 "모든결과에서 h 1~60 별 각 평가지표 별 점수도 내가 보고싶으면 바로 볼수있어야해 … 보고서에 써야할수도있어서".

**Why:** 보고서에 h 별 표·그림을 넣어야 할 수 있음.
**How to apply:** 판마다 `h_metrics.csv`(CV 는 판 끝날 때, test 는 보고 허가 때만)와 MLflow artifact 를 남기고, 저장된 예측만으로 다시 만들 수 있게(학습 없이), 여러 판을 한 파일로 내보내는 명령(`scripts/export_h_metrics.py`)을 유지. 지표는 trapz PR-AUC 중심, AP 금지([[feedback-no-ap-anywhere]]). 관련 [[feedback-report-f2-figures]].
