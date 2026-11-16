---
name: feedback-raw-data-no-preprocessing
description: DB 수집 데이터는 원자료(raw) 그대로 받아야 함 — 결측 채움·단위 변환·범위 필터·집계 등 전처리는 사용자 승인 없이 하지 말 것
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d9b9b418-a320-4356-99d8-fbe2d77c9f46
  modified: 2026-09-14T07:03:28.349Z
---

DB 에서 받는 모든 변수는 원자료 형태로 저장해야 한다. 결측→0 채움, 이상값 제거, 단위 변환(분→시간 등), text→숫자 캐스팅, 조건 필터 후 집계, 창 집계 같은 가공은 사용자가 요청하거나 승인한 경우에만 한다.

**Why:** 2026-09-14 3_DB수집_조인.ipynb 에서 임의로 QC·단위 변환·NULL→0 채움을 했더니 사용자가 "모든 변수는 raw형태로 받아져있어야하는데 니가 뭔데 자꾸 전처리함? 식정원 관계자가 물으면 니가 대답하고 책임질거야?" 라고 지적했다. 가공 판단은 식정원에 설명할 책임이 있는 사용자 몫이다. 실제로 ASOS 강수 NULL→0 채움은 근거가 틀렸다(값 중 0 도 22.6% 기록됨).

**How to apply:** 수집 단계는 SELECT * 수준 원형 저장(`data/collected/raw/`)과 상태 측정(결측률·연도·지점·의심값 개수)만. 가공이 필요해 보이면 무엇을 왜 바꿀지 제안만 하고 사용자 결정을 기다린다. 관련: [[listeria-nb3-db-join-findings]], [[toxin-db-read-only-rule]]
