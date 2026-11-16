---
name: listeria-db-structure-findings
description: "리스테리아 DB 구조 확인 사실 (2026-09-14, 2.ipynb) — 검사목적 코드 IM02·'기타' 82%, 코드분류는 IM*(ZIM* 아님), Z27 코드=팽이버섯, parquet=DB−검사기관주소행, 날짜 컬럼 실체, 조인키, ba_data_info 원재료 테이블, 제품명 기준 원물 8.6%, export 3열(result_value_num·judge_stdr·judge_stdr_num) 정의=staging_data.build_export_qual (09-22)"
metadata: 
  node_type: memory
  type: project
  originSessionId: a1448cd6-c4bb-4222-9534-942acfa01019
  modified: 2026-09-14T07:41:17.908Z
---

2026-09-14 에 `리스테리아ML/notebook/2.ipynb` 를 만들며 DB(TOXIN_MODEL_PRED)를 읽기 전용으로 확인한 사실.
보조 추출 파일은 `리스테리아ML/data/collected/nb2_db_*` (attr·s_org·ledg·instt·codes + SQL), 재추출은 `scripts/nb2_extract_db.py`.

- **검사목적**: 원본 LIMS `m_org_lims_data_info.inspct_purps_code`(공통코드 IM02)가 parquet 행의 98.9%에 있음. 검출 1,969건 중 '기타' 1,619(82.2%), 자가품질위탁 127, 수입검사 96, 수거검사 66. '기타'는 99.9% 자체의뢰(CI07000001). ready_data 에는 이 컬럼이 없음.
- **공통코드 분류명은 `IM02/IM03/IM08/IM11/IM40`, `CI07`**. `collect_all.py` 가 쓰는 `ZIM08/ZIM11/ZIM40` 은 0행이라 이름 조인이 비어버림.
- **품목코드**: Z2700400000000('정액,수정란,종란') 행의 실제 제품명은 수출용 팽이버섯·재배장 환경검체(검출 640). Z3800100000000('기타기준규격외')은 HACCP 공정시료가 다수. 코드 첫 글자 A/B/C/Z/G/E/D = ready_data `statistic_prdlst_nm_1`.
- **parquet vs DB**: parquet(676,537)의 키는 전부 DB(765,891)에 있고 판정·기준일 100% 일치. DB에만 있는 89,354행 중 86,696행이 `addr_source='검사기관'`.
- **날짜**: parquet `mnfctur_de` = 기준일(원본 제조일은 43.8%만). 국내 `pick_de`(채취일) 0%, `rcept_de`=의뢰일 당일, 검사완료−접수 중앙값 9일, 제조→의뢰 중앙값 2일. `m_org.distb_de` 는 유통기한 날짜로 보임(대장 유통기한과 상관 0.80). `valid_tmlmt` 는 날짜 아닌 작은 숫자(2.5%만).
- **수입**: `ready_data.cntc_sys_key2 = s_org_lims_data_info.rcept_no` (99.6% 매칭). date_basis '선적일' = `s_org.shipng_de` 100%. `s_org.distb_bgn_de` 는 선적 후 중앙값 180일이라 제조일이 아니라 유통기한으로 보임(모델팀 미확인).
- **품목제조대장**: `prdlst_mnfctur_sttemnt_no`(원문 또는 숫자만) → `t_prdlst_mnf_ledg.prdlst_report_no`. 인덱스 없음 → `= ANY(%s::text[])` 로 한 번에 조회하면 1초. 유통기한은 `pog_daycnt` 문구("제조일로부터 12개월"), `prsrv_distb_stdr_cn` 은 거의 비어 있음. 국내 행 1/3 연결, 검출은 11%만.
- **DB 전체 탐색(2.ipynb 1-2절, `nb2_db_explore_*.csv`)**: 스키마 12·테이블/뷰 253·컬럼 10,688. 테이블목록·정의서는 api_data_info·ba_data·ready_data 만 다룸. ready_data 는 54열 중 parquet 에 없는 36열(통계용 품목분류 1~4 100%, 품목제조번호 국내 28.7% 등). 정의서 밖 `ba_data_info` 에 원재료 테이블 있음: `lcns_ledg_hist_rawmtrl`(원료명·원산지·배합비, 5,148만행 14GB, PK 인덱스 (rawmtrl_sno, prdlst_report_ledg_no, hist_ordno) → 조인키가 둘째 컬럼이라 무거움), `mnf_step_exam`/`distb_step_exam`(제조·유통단계 조사, 서술형), `t_cmmn_spec`(기준규격 n·c·m). 미정의 스키마라 **리스테리아 행과 조인 안 함, 사용 여부 모델팀 질문 중**. 구조화된 원재료 입고일·수확일 컬럼은 DB 어디에도 없음.
- **8칸 품목별 양성률(2.ipynb 3-6-1절)**: 품목 = ready_data `prdlst_nm`(추출 파일 `prdlst_nm_self`, 코드당 이름 1개). 단 **품목명↔품목코드는 1:1 아님** — 8칸 품목명 626 ↔ 코드 1,192 (14자리 vs 20자리=뒤에 000000, 가공식품 C03 현행·C02·C01 혼재; 수입 축·수산물 20자리 코드는 어종·냉동/냉장·부위·횟감 세부코드). `prdlst_nm_org` 는 표기 변형(소고기/쇠고기) 있어 비권장. **수입 LIMS 리스테리아 행에 농산물(A) 코드 0행** → 해외×농산물 칸은 원천부터 없음. parquet 의 '검사기관' 주소 행 제외가 국내 원물 양성률을 올림(국내 수산물 DB 4,334행 0.115% → parquet 469행 0.640%). 주요 값: 국내 포장육 101/2,668(3.8%), 수입 소고기 9/165, 수입 치즈 17/4,296 vs 국내 1/3,543, 알가열제품 0/19,818. 품목 단위·기준 행은 모델팀 질문 중.
- **원물 식별(2.ipynb 3-6-2절, 제품명 정규식 휴리스틱)**: 품목코드 A·B 기준 국내 원물 3,533행(검출 27) → 기준규격외(Z) 코드 속 원물 이름 행 50,251행(검출 661) 추가 시 53,784행(국내 8.6%, 검출 688). 검출은 팽이버섯 418·원료육 112·생선 115에 몰리고 채소 17,359행 5건·과일 2,743행 0건. Z 속 원물은 검사목적 '기타' 92%·HACCP 7%, 채소는 김치공장 세척 전후 원료가 대부분 → 산지 원물이 아님.

**Why:** 사용자가 나박사님·모델팀에 질문을 준비 중이고, 이후 수집 스크립트 수정·모델 범위 결정이 이 사실들에 달려 있음. 모델팀 답변으로 바뀔 수 있으니 확인 여부를 먼저 볼 것.

**How to apply:** DB 작업 전 이 조인키·코드분류를 재사용하고, 값이 달라 보이면 추출일(2026-09-14) 이후 DB 갱신을 의심. 규칙은 [[toxin-db-read-only-rule]], 변수 매핑은 [[listeria-variable-mapping]].

**2026-09-22 추가 — parquet 원천과 정의서 미기재 열 3개 (정의 확정)**: `listeria_body.parquet`(676,537행·23열)은 `staging_data.export_body_listeria`(모델팀 export 표, 코멘트 없음)와 동일. 세 열 `result_value_num`·`judge_stdr`·`judge_stdr_num` 은 명세서 3개 파일 어디에도 없고 `staging_data.export_body_*` 4개 표에만 있으며, 정의는 DB 함수 **`staging_data.build_export_qual`**(정성 항목용 export 생성 함수; 정량용은 `build_export_quant`) 소스에 있다: `result_value_num = round(ba_data.jz_num(result_value,'micro'),4)`(문자열 정규식 정리 후 남는 숫자 최댓값 — 리스테리아는 0 이 3,775행·'2건이상…'→2 가 2행뿐인 파싱 부산물), `judge_stdr = staging_data.hazard_preprocessed_data.mxmm_value`(정의서 "규격 기준값"; 국내 64행=`m_org.mxmm_value`"최대 값", 수입 7,203행=`s_org.stdr_value`"기준값"), `judge_stdr_num = round(ba_data.jz_stdr(mxmm_value),4)`(상한 숫자 파싱, 68행). 세 식 모두 전수 재계산과 100% 일치. 같은 함수의 WHERE(`addr_source <> '검사기관'` 등)가 parquet 에서 검사기관 주소 행이 빠진 이유. 미해결: 수입 40,073행은 `s_org.stdr_value` 가 있는데 `mxmm_value` 가 NULL, 국내는 `m_org.stndrd_value`(774,620행)가 export 에 안 실림. 확인 스크립트 `리스테리아ML/script/db_export_3cols_정의확인.py`, 상세는 작업기록 2026-09-22 16:30 항목. 이름만 보고 "검사 결과값·판정기준"이라 부르지 말 것(CLAUDE.md "추정하지 않는다").
