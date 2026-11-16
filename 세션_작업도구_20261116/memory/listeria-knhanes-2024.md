---
name: listeria-knhanes-2024
description: "국민건강영양조사 2024 SAS(사용자 제공, Downloads\\ㅇ) → data/collected/knhanes_2024 parquet·컬럼 선정 xlsx; 시료와 공통 키·조사 날짜 없음, 식품코드별 성분 고정 (2026-09-28)"
metadata:
  node_type: memory
  type: project
  originSessionId: 8d475110-b329-4f55-8b15-dc954cb3b250
  modified: 2026-09-28T05:04:45.185Z
---

2026-09-28 사용자가 리스테리아용으로 국민건강영양조사 2024 SAS 2개(`C:\Users\NT751\Downloads\ㅇ\HN24_ALL(SAS)\hn24_all.sas7bdat`, `HN24_24RC(SAS)\hn24_24rc.sas7bdat`)를 줌.

- 변환: `리스테리아ML/scripts/knhanes_to_parquet.py` → `data/collected/knhanes_2024/hn24_all.parquet`(1명 1행 6,997×813), `hn24_24rc.parquet`(재료 1개 1행 543,782×110, 6,802명). 문자열 bytes→cp949, 안 풀린 39칸은 끝 글자 잘림(�).
- 컬럼 설명은 SAS 변수 라벨(pandas 가 latin-1 str 로 줌 → encode('latin-1').decode('cp949')). 코드값 뜻은 파일에 없음 → 이용지침서 필요(사용자에게 아직 없음).
- 선정 35개·제외 이유: `docs/02_데이터수집/국민건강영양조사2024_컬럼목록.xlsx`, 근거 수치 `scripts/knhanes_columns.py`, 작업기록 2026-09-28 항목.
- 핵심 사실: 조사 날짜·월 없음(N_DAY 요일만), 2024 한 해; 식품코드 N_FCODE 5자리 ≠ 식품공전 train_prdlst_code 14자리; 제품명·회사명 원문 일치해도 리스테리아 검출 2·7행; NF_*÷NF_INTK 가 식품코드마다 일정(식품코드별 수분·나트륨 함량 계산 가능); N_FNAME '생것' = 조리 전 재료 상태(된장찌개에도 들어감).

**Why:** 사용자는 품목 단위 속성(섭취 형태·RTE 등)을 외부 자료로 얻으려 함 — 질문초안.md "국민건강영양조사 계열 자료가 식약처 코드로 매칭되는지".
**How to apply:** 시료 행 직접 조인 불가 → 품목 대응표(식품코드↔식품공전 코드) 경유 고정값만 가능하다는 전제로 다룸. 대응표 존재·이용지침서·수분/나트륨 대용 여부는 미결. 관련: [[listeria-item-code-tables]], [[feedback-raw-data-no-preprocessing]].
