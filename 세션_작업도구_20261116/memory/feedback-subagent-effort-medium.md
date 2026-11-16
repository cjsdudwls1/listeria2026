---
name: feedback-subagent-effort-medium
description: 에이전트(서브에이전트·병렬 탐색)를 띄울 때 effort 는 모두 medium 으로 (2026-10-26 사용자 지시)
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1fb5c799-232e-410c-8d2d-08899f55f659
  modified: 2026-10-26T04:55:03.669Z
---

2026-10-26 사용자: "effort는 모두 middle로" — 친환경 자료 탐색 에이전트 3개를 기본 effort 로 띄우자 지시. 기존 에이전트를 멈추고 effort: medium 으로 다시 띄움.

**Why:** 사용자가 에이전트 비용·속도를 중간 수준으로 맞추길 원함.
**How to apply:** Agent 도구로 에이전트를 띄울 때 effort 를 지정 — 작은 작업은 "medium", 좀 복잡한 작업(검수·설계·여러 파일 대조 등)은 "high"(2026-10-26 사용자 보충: "작은 작업이면 mid, 좀 복잡하면 high"). 효율적이면 병렬로. Workflow 의 agent() 도 같은 기준.
