---
name: listeria-nb3-db-join-findings
description: "3_DB수집_조인.ipynb(2026-09-14) — DB 수집·조인 산출물 위치, 원천 데이터 품질 함정(ASOS 강수 NULL=무강수, NAS 일조=분·±50 경계값, 해외 일조=초, 토양 pH 이상값), t_aw_cl=리서치팀 파일, 매핑 엑셀 오류 목록"
metadata: 
  node_type: memory
  type: project
  originSessionId: d9b9b418-a320-4356-99d8-fbe2d77c9f46
  modified: 2026-11-12T08:46:47.520Z
---

2026-09-14 `리스테리아ML/notebook/3_DB수집_조인.ipynb` 를 만들어 매핑 엑셀의 DB 컬럼을 수집하고 parquet(676,537행)에 조인했다.
산출물: `리스테리아ML/data/collected/` 의 `00_body_db`·`02a~02f`·`31~34`·`41/42_*_qc`·`43`·`50`·`60/61`·`70`·`80` parquet 과
`listeria_joined.parquet`(676,537행 × 약 230열), `coverage_summary.csv`. 노트북 생성기는 worktree 브랜치 `worktree-listeria-db-collect` 의 `리스테리아ML/scripts/build_nb3.py`.

사용자 지적(2026-09-14): 노트북 3 의 QC·단위 변환·NULL 채움은 승인 없는 전처리였음 → [[feedback-raw-data-no-preprocessing]]. 원형 파일은 `data/collected/raw/raw_<table>.parquet`(ASOS·NAS 일자료/관측소, 에어코리아 일자료/측정소, 예보 테이블; SELECT *).

해외 기상 원천: `api_data_info.global_factory_weather`(ERA5, 해외 제조소 좌표×일, 2.3억 행, 테이블목록·정의서에 없음) 의 lat_g/lon_g 가 hazard_preprocessed_data 해외 lat/lon 과 3,072/3,072 일치 → 수입 시료 97.4% 당일 값. PM10 은 `global_factory_fine_dust`(같은 좌표). `weather_data.overseas_weather` 는 그 부분집합·단위 변환본. 노트북 3 은 국가당 1점인 global_weather_info 를 썼음.

교수님 보고용 결과보고(2026-09-14, v3 2026-09-15): `리스테리아ML/docs/리스테리아_위해인자_DB수집_결과보고.xlsx`(시트 0_문자보고·1_수집가능여부·2_조인방법·3_데이터상태(raw)·4_수집불가_수집방법), 생성기 `scripts/report/build_report3.py`, 근거 JSON `scripts/report/results/`(브랜치 worktree-listeria-db-collect 커밋 4b9e6c2). 작성 규칙은 [[feedback-report-plain-language]].
매핑 이후 DB 전 스키마 재탐색으로 새로 확인(테이블목록·정의서 미기재):
- `api_data_info.water_pollution` = 출처 r21 수질오염원설치시설(기타) 64,048개소(좌표 TM 계열·좌표계 미상, 2026-08-20 적재). 뷰 water_pollution_alive 는 가공 뷰.
- `ba_data_info.food_moisture_info` 수분활성도·수분함량 문헌값 565품목(aw 96), 코드 없음(몸통 prdlst_nm 정확 일치 국내 43.6%).
- `ba_data_info.distb_step_exam` 유통단계조사 보관_온도·포장_형태 텍스트, LIMS 연결 키 없음.
- `t_aw_cl` 은 리서치팀 파일 적재 후 aw 빈칸 128코드 채움('original' 표기), 값 변경 8코드, 우유류 6코드 부등호 반전(0.98< → <0.98), aw 0.15 동일값 615행.
- NAS `gr_temp` DB 코멘트 '초상온도'(정의서 '지면온도'), `soil_wt` '토양수분보정값(%)'. soil_ph_info pnu_cd 는 2023년 이후 10자리 법정동코드 621,359행 섞임.
- hazard_preprocessed_data 품목 컬럼(train_/statistic_/mapped_)은 가공 컬럼 → 원자료는 m_org.prdlst_code/prdlst_nm, s_org.prdlst_cd/prdlst_nm, t_prdlst_cl.kor_nm.

원천 데이터 특성 (측정 사실, 처리는 사용자 결정):
- ASOS `kma_asos_daily_info.sum_rn`: 2026-09-15 근거 확보 → NULL=무강수, 0.0=0.1mm 미만 강수로 판단. NULL 날(243,843일)은 hr_1_max_rn·mi_10_max_rn·sum_rn_dur 가 40·26·84일뿐, 0.0 날은 55.2%·31.1% 기록 / 5km 이내 NAS 34쌍 비교 NULL 날 NAS 0mm 97.2% / NULL 비율 7월 47% 최저. 기상청 공식 문서는 못 찾음. **11-03 사용자 확인("ASOS 강수는 원래 0이면 nan으로 두는 기관") → 빈 값 = 0mm 로 채움 확정(최종 계획 결정 61).**
- 최근접 관측소를 날짜 무관(일자료 있는 전 관측소)으로 고르면 개설 전·폐쇄 후 날짜가 잡혀 행 없음: ASOS 2.8%(대구(기) 176번 2015-06-11 종료 → 12,861행), NAS 2.4%, 에어코리아 13.4%(2019~2025 신설). 기준일에 자료가 있는 관측소 중 최근접이면 ASOS 기온 99.9%·에어코리아 98.3%. 방식은 사용자 결정 대기.
- NAS 관측소표에 시험용 '동방로거테스트'(561034A001, 정읍 업체) 일자료 444일 → 국내 시료 3,135행이 최근접으로 잡음. NAS 코드 7번째 글자 F 32곳은 일조·토양수분·지중온도·초상온도 DB 전체 0건.
- `kma_weather_predict_info` 는 2026-07-01 발표 12일 예보 1,068행뿐 → 연구기간 행 0.
- NAS `sun_time` 은 분 단위(중앙값 422), 기온·토양수분에 ±50 경계값, soil_wt 음수 → 범위 밖 결측.
- `global_weather_info.sunshine_duration` 은 초 단위, 풍속은 km/h 로 보임.
- `soil_ph_info.acid` 에 3 미만 51·10 초과 138 필지 → pH 3~10 만 집계.
- `keco_airkorea_obs_station.dm_x/dm_y` 는 TM 이 아니라 위도/경도 그대로.
- 국가코드: 몸통 alpha-2 → `ba_data_info.nation_code` 로 alpha-3 변환(정의서 밖 코드표).
- `ba_data.t_aw_cl`(1,316행) = 리서치팀 `통합_수분활성도_수분함량_식품pH_염도_최종.xlsx`(653행) 적재본. pH·염도 100% 일치, 수분활성도 빈칸은 상위 분류 중앙값으로 채우고 상위 분류 코드 663개 추가. **2026-11-12 사용자 "리서치팀 원본 엑셀은 쓰레기임. 삭제" → 그 엑셀은 휴지통, 근거·비교로 쓰지 말고 DB 표만 볼 것.** 11-12 DB 갱신(2,475행, aw_source 'xlsx_20261111' 1,365행) + `t_aw_cl_full`(8,864행, 값은 t_aw_cl 과 같음).

매핑 엑셀(리스테리아_수집변수_매핑.xlsx) 확인된 오류 — 사용자 수정 여부 미정:
- 시트3 에 '수집방안' 컬럼 없음(기관·확보형식·주기·URL 만). r9 팜맵 농업기상(높이별·깊이별) 행 누락.
- 시트3 PPT 행 순위 표기 'pH 2순위·염도 3순위' 는 PPT 상 3·4순위. 시트2 우선순위 번호도 pH 이하 1씩 작음.
- 시트3 r17 농업저수지는 빨간 글자 아님(근거 PPT '수질'), PPT 3행은 데이터출처에 근거 행 없음.
- 시트1·6 의 'dm_x/dm_y TM→위경도 변환' 은 틀림. 시트6 의 ZIM* 코드분류는 실제 IM*.

누수 주의: 품목제조대장·aw 파일 조인 값은 검출 행에서 10~20%p 덜 붙는다(09-14 수치, CLAUDE.md 에는 "결측률이 달라진다"까지만 있음).
2026-09-28 X.parquet 로 다시 잰 t_aw_cl 정확 일치 연결률(검출 / 불검출): `prdlst_code` 키 aw 13.15 / 37.03%, ph 12.39 / 31.42%, na 12.85 / 33.00% · `train_prdlst_code` 키 aw 27.32 / 51.11%, ph 2.69 / 16.84%, na 2.69 / 20.24%.
2026-09-28 현재 t_aw_cl.`aw_source`(코멘트 없음) = 'original' 1,251행 + 그 밖 65행(derived_median 9·recomputed 10·mapped_from·assigned); ph·na 값은 original 행(+recomputed 1)에만 → 위 '128코드 채움' 서술과 다름, 최신 값 기준으로 볼 것.

**Why:** 다음 작업(EDA·모델링·매핑 엑셀 수정)이 이 파일과 품질 규칙을 전제로 한다.
**How to apply:** 재수집 시 노트북의 `FORCE=True` 또는 해당 parquet 삭제. 규칙은 [[toxin-db-read-only-rule]], 앞선 사실은 [[listeria-db-structure-findings]], 매핑 원본은 [[listeria-variable-mapping]].
