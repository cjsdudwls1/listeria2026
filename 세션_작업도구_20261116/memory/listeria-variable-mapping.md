---
name: listeria-variable-mapping
description: "리스테리아 예측변수 ↔ DB(테이블정의서) 매핑 결과 파일 위치, 테이블정의서 색상의 실제 의미, 스키마명 출처, DB에 없는 변수 목록"
metadata: 
  node_type: memory
  type: project
  originSessionId: 30e02f3c-1d9c-4156-8d4b-85f5fc038035
  modified: 2026-09-15T02:55:17.281Z
---

Deliverable (2026-09-10): `리스테리아ML/docs/리스테리아_수집변수_매핑.xlsx` — 6 sheets: 1 빨간항목→DB 매핑,
2 PPT 추가변수(수분활성도·pH·염도·토양온도…) 매핑, 3 미수집(외부 수집 필요), 4 추가 발굴 제안, 5 빨간항목 원문,
6 조인키 요약. Sheet 4 keeps only rows whose 근거 유형 is A(문서 직접) or B(문서 응용) judged against the three research docs
(데이터출처.xlsx, 데이터현황_0707.xlsx, 예측인자 정리 ppt.jpg). On 2026-09-10 the user had every C(문서 근거 없음) row and the
microbiology-knowledge groups deleted (sheet 7 removed too) and also rejects rationale borrowed from other hazards (e.g. the 납
note on a_org_collect_insp.anals_se). The companion md 리스테리아_추가발굴_제안.md was deleted by the user; do not recreate unasked.
Also removed on the user's order: PM10/PM2.5/오존 rows (red only on 살모넬라·대장균·납 sheets, not Listeria), the
아플라톡신-only columns, and everything from ba_data.t_aw_cl (테이블목록 calls it 곰팡이 독소 전용) — so 수분활성도·식품pH·염도 now
count as 미수집 (sheet 3). Every (schema, table, column) in it was validated against 테이블정의서.xlsx by script.

Facts about `테이블명세서(모델팀)/테이블정의서.xlsx` (4 sheets: api_data_info, 'ba_data 테이블', 'ba_data 뷰', ready_data;
columns 테이블/순번/컬럼/타입/길이/NULL/설명; table header rows are merged A:G like `name (TABLE · n컬럼 · rows · size)`):
- The only yellow-ish fill (FFF3D6) is on `judge_result` rows (★ 검출/불검출 판정) in the 'ba_data 뷰' sheet = target.
  No predictor columns are yellow. Some rows carry an explicit white (theme 0) fill — likely cleared highlights
  (sum_rn, sum_sshr, sum_gsr, srqty, soil_wt, any_year, acid, most of ready_data) — not a reliable marker.
- Schema names come from `테이블목록.xlsx` (same folder): api_data_info / ba_data / ready_data.

Separate file `리스테리아ML/docs/20260901.테이블정의서 (1).xlsx` (3 sheets: api_data_info, 'ba_data 테이블', 'ba_data 뷰'; no ready_data)
carries manual highlights on the 설명 cell (col G) — checked 2026-09-15: yellow FFFF00 = keco pm10_value, ASOS sum_sshr·sum_gsr,
NAS temp·hum·rn, NAS station instl_la·instl_lo, t_aw_cl aw·ph·na; green 92D050 = kma_weather_predict_info.precip_mm, NAS sun_time·srqty;
red = t_aw_cl.mc; cream FFF3D6 = judge_result rows. Their status was appended as section 5 (rows 135~164) of 0_문자보고 in the
DB수집 결과보고 xlsx ([[listeria-nb3-db-join-findings]]). That 정의서 lists t_aw_cl as 8 cols·1,262 rows; the DB on 2026-09-14 had 9 cols (aw_source added)·1,316 rows.

Variables NOT in the DB (need external collection): depth-resolved 토양수분·기온·풍향, 유역별/흙토람/수자원공사 토양수분,
SSP/RCP 월별 미래기후, 해수 염분·해상풍(수과원), 지하수 오염예측등급·DRASTIC, CO2 농도(keco co_value is CO),
수질(농업저수지·수질오염원). Closest DB substitutes: 저장·유통온도 → m_org.cnvync_sttus_tp + t_prdlst_mnf_ledg.prsrv_distb_stdr_cn;
포장유형 → t_prdlst_mnf_ledg.frmlc_mtrqlt(재질만); 식품 aw/pH/염도 → ba_data.t_aw_cl (text ranges; aw_grade is 아플라톡신 전용).

**Why:** Saves re-deriving the color semantics and the DB gap list next session; the user will collect the DB part via
DataGrip and the rest from 기상청 etc.

**How to apply:** Start from the xlsx above; re-validate columns against 테이블정의서.xlsx if that file's mtime changes
(was 2025-08-13). See [[hazard-data-source-red-items]] for the source-file color logic.
