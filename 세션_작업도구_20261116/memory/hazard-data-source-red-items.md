---
name: hazard-data-source-red-items
description: 리서치팀문서/위해요소별_중요인자_데이터출처.xlsx — red text in column D marks the factors the user must collect per hazard; how to parse it (openpyxl rich_text=True) and known quirks
metadata: 
  node_type: memory
  type: project
  originSessionId: 30e02f3c-1d9c-4156-8d4b-85f5fc038035
  modified: 2026-09-10T02:49:42.900Z
---

The user is building food-hazard prediction ML (e.g. 리스테리아ML/) and collects external data listed in
`리서치팀문서/위해요소별_중요인자_데이터출처.xlsx`. Originally 10 sheets (one per hazard); on 2026-09-10 the user had all
sheets except '리스테리아 모노사이토제네스' deleted from this file and from `[2026] 위해요소별 데이터 현황_0707.xlsx`
(untouched originals are in `리서치팀문서/원본백업/`). The 현황_0707 file carries the same red marks as 데이터출처.
Columns: A 데이터구분, B 데이터명, C 제공기관, D 데이터 제공 항목, E 관리부서명, F 현재상태, G 생산 주기, H URL, I 비고.
In column D, red font (RGB FFFF0000, partly as rich-text runs, partly whole-cell) marks the "주요 인자" the user
must collect; blue (FF0070C0) text is the remaining, non-target items. A whole row in blue = non-target source.

**Why:** The meaning of the colors is not written anywhere in the file; the user explained it on 2026-09-10.

**How to apply:** The earlier parser `리서치팀문서/parse_red_items.py` was deleted by 2026-09-10 (only a .pyc
remains) — do not cite it. Re-derive with `openpyxl.load_workbook(path, rich_text=True)`, iterate `CellRichText`
runs and keep `TextBlock`s whose `font.color.rgb == 'FFFF0000'`, plus whole cells whose `cell.font.color.rgb` is red.
Known quirks: unbalanced parentheses ('산도(pH)유효인산(mg/kg'), red text cut mid-word (리스테리아 r21 '…시설 데이'),
duplicated items within one cell (팜맵 row), merged cell D23:D24 in 노로바이러스/패류독소 sheets.
Listeria sheet result (2026-09-10) is saved in [[listeria-variable-mapping]]. Run with `py` per [[python-env-use-py-launcher]].
