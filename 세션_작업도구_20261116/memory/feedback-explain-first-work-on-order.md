---
name: feedback-explain-first-work-on-order
description: "질문(\"~가 더 맞지 않아?\")에는 아는 기록으로 상황 설명부터, 재측정·DB 조회 같은 작업은 사용자가 지시할 때만 (2026-10-14 지적)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b092be94-75f3-4ae6-a16e-77e0d3bcdeb4
  modified: 2026-10-14T05:43:44.904Z
---

질문을 받으면 이미 있는 기록(메모·노트북·파일)으로 지금 상황과 그렇게 된 이유를 먼저 설명한다. 다시 재기·DB 조회·시뮬레이션 같은 작업은 사용자가 시킬 때만 하고, 필요하면 "다시 재 볼까요?"로 제안만 한다.

**Why:** 2026-10-14 `식품.parquet` 조인 키 순서 질문("train_prdlst_code_2 부터 찾는게 더 맞지 않아?")에 내가 설명 없이 DB 접속 준비·재측정에 들어가자 "상황 설명부터해. 작업지시는 내가 내릴때만." 이라고 함.

**How to apply:** 질문형 메시지 → 기록 기반 설명 + 출처 표시 + (필요 시) 할 수 있는 작업 제안으로 끝낸다. 파일 읽기 수준의 확인도 최소로. 관련: [[feedback-brainstorm-before-download]], [[feedback-no-rerun-without-asking]], [[feedback-ask-consent-for-new-changes]].
