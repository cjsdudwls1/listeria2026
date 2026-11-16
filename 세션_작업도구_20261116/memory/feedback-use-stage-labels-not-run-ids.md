---
name: feedback-use-stage-labels-not-run-ids
description: "결과 표를 보여 준 뒤 설명할 때는 그 표의 '단계' 칸 이름으로만 말하고 판 이름(3-0-LSTM·3-0b-LSTM-end 등 내부 ID)은 쓰지 말 것 (2026-11-13 지적)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: a19471fc-8fc4-4365-8d63-c14cd9e46718
  modified: 2026-11-13T00:57:00.754Z
---

결과 표(단계별 val·test 표 등)를 보여 준 뒤 그 내용을 설명할 때는, 그 표의 '단계' 칸 값(예: '2차: 13구조 1위', '3차: 변수 빼기', '결합 스태킹')으로만 말한다. 판 이름(3-0-LSTM, 3-0b-LSTM-end, 5-LSTM-food2 같은 내부 ID)은 쓰지 않는다.

**Why:** 2026-11-13 사용자가 "왜 3차 변수빼기를 선정했지?"라고 물은 데 내가 판 이름(3-0-LSTM·3-0b-LSTM-end)을 섞어 답하자 "니가 준 표에서 … '단계'열 값으로만 얘기해"라고 지적. 판 이름은 사용자가 표에서 바로 찾을 수 없어 헷갈림.

**How to apply:** 표를 줄 때 '단계' 칸 이름을 사람이 읽을 수 있게 짓고, 뒤이은 설명·답에서는 그 이름을 그대로 따옴표로 씀. 판 이름이 꼭 필요하면 표에 이미 있는 경우에만, 단계 이름 뒤 괄호로. 관련 [[feedback-report-reader-first]], [[feedback-exact-table-names]].
