---
name: feedback-own-ideas-not-user-memo
description: "제안을 요청받으면 사용자 메모(memo.md 등)에 이미 적힌 안을 되돌려 말하지 말고, 데이터를 먼저 재고 출처를 확인한 Claude 자신의 안을 낼 것"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 22fbad70-c9b3-4e78-b656-9d1f9a1ba699
  modified: 2026-10-06T01:51:23.780Z
---

"어떤 데이터를 쓰면 좋을까" 같은 제안 요청에 `리스테리아ML/notebook/memo.md` 에 사용자가 이미 적어 둔 안(품목 특성, 업체 HACCP·행정처분)을 1·2순위로 그대로 되돌려 답했다가 지적받았다 (2026-10-06, 사용자: "니가 말한거 1,2,는 내 메모잖아 니가 생각해서 또 말해봐").

**Why:** 사용자는 자기 메모를 요약받으려는 게 아니라 새 후보를 원한다. 메모·작업기록에 있는 안은 이미 검토 중인 것이다.

**How to apply:** 제안 전에 `memo.md` 와 `작업기록.md` 의 최근 항목을 읽어 이미 있는 안을 걸러 낸다(겹치면 "메모에 있는 안"이라고 밝히고 순위에서 뺀다). 제안은 배경지식 나열이 아니라 ① 검출 행이 어디에 몰려 있고 어떤 키로 붙는지 데이터에서 먼저 재고 ② 외부 자료는 제공 페이지에서 받을 수 있는지 확인한 뒤 낸다. (뒤 두 가지는 이 지적에 대한 Claude 의 해석이다.) 관련: [[feedback-cite-sources-exactly]], [[listeria-external-data-candidates]]
