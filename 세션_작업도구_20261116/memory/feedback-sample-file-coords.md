---
name: feedback-sample-file-coords
description: 좌표로 매칭하는 모든 조인(관측소 최근접·해외 제조소 좌표 표·거리)은 항상 시료 파일 listeria_body.parquet 의 lat·lon — DB 좌표(db_lat·h_lat 등) 금지 (2026-09-17)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d4e71e82-3176-414c-86cf-127d35e207ad
  modified: 2026-09-17T07:39:08.135Z
---

좌표로 매칭할 때는 항상 시료 파일(`리스테리아ML/data/listeria_body.parquet`)의 `lat`·`lon` 을 쓴다. DB 전처리 시료표(`ready_data.hazard_preprocessed_data`) 좌표 — 조인 데이터셋 `db_lat`·`db_lon`, 추출본 `h_lat`·`h_lon` — 는 참고만.

**Why:** 2026-09-17 공유용 데이터셋 첫 판이 DB 좌표를 써서 지적받았고, 같은 날 해외 표 점검에서 제조소 좌표 표를 DB 좌표로 맞춘 사실이 드러나자 사용자가 "매칭할땐 항상 시료 파일 좌표를 써야함"이라고 일반 규칙으로 못박고 CLAUDE.md 기록을 지시함.

**How to apply:** 시료 파일 좌표는 float32 → double 표 좌표와 그대로 등호 비교하면 0건. float32 로 맞춰 비교하거나 최근접, 방법은 사용자 확인. 국내 296행은 두 좌표가 다름, 해외 제조소 좌표는 float32 로 3,072→3,071곳(호주 1곳 31행 표 좌표 2개). 검증: LIMS조인_데이터셋 asos/nas/air 43칼럼은 시료 파일 좌표로 만든 것 확인(국내 628,895행 전부 재계산 일치). 관련 [[listeria-share-dataset]], [[listeria-global-country-tables-join]].
