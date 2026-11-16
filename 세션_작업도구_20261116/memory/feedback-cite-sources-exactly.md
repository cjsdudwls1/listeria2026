---
name: feedback-cite-sources-exactly
description: "수치·사실의 출처를 댈 때 실제 출처(CLAUDE.md/메모리/노트북/DB)를 정확히 — 메모리 속 옛 수치를 CLAUDE.md 라고 하지 말고 다시 재서 쓰기, 표 구조는 조회 후 말하기 (2026-09-28 지적)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 212a77c0-0802-4f6c-99a5-646ec16ab768
  modified: 2026-09-28T08:45:50.361Z
---

어떤 수치·사실을 "어디에 적혀 있다"고 말할 때는 실제 출처를 정확히 댄다. 메모리에서 온 옛 수치는 "09-14 노트북 3 결과(메모)"처럼 출처와 시점을 밝히고, 가능하면 지금 데이터로 다시 재서 쓴다.

**Why:** 2026-09-28 "CLAUDE.md 에 적힌 대로 검출 행에서 10~20%p 덜 붙는다"고 답했는데, CLAUDE.md 에는 "결측률이 달라진다"까지만 있고 10~20%p 는 메모리([[listeria-nb3-db-join-findings]])의 09-14 수치였다. 사용자가 "이게 뭔소리노? 클로드.md에 뭐라고 써져있는데?"라고 되물음. 같은 답에서 `api_data_info.soil_ph_info` 를 "시간이 없는 값"이라고 했으나 실제로는 `any_year`(시료채취년도)·`exam_day`(토양검정일) 칼럼이 있었다.

**How to apply:** 규칙·문서 인용은 원문을 다시 열어 확인한 뒤 인용한다. 표 구조(날짜·좌표 칼럼 유무 등)는 DB 칼럼·코멘트를 조회한 뒤 말한다. CLAUDE.md 의 "추정하지 않는다" 규칙과 같은 맥락이다.
