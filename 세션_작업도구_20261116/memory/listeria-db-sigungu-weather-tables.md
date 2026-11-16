---
name: listeria-db-sigungu-weather-tables
description: "DB(TOXIN_MODEL_PRED)의 시군구 단위 기상 테이블 조사(2026-10-15) — 전부 2026년 아플라톡신 예측 실행분(기준일 전 100일+11일 예보)뿐, 학습 기간 전체 시군구 기상은 없음; api_data_info.kma_sigungu_point(시군구 대표 좌표 251개)가 대표 지점 후보, 우리 국내 행 98.61%·검출 98.86% 연결"
metadata:
  node_type: memory
  type: project
  originSessionId: 2af334fa-b578-4913-a2a5-05aedbdda88a
  modified: 2026-10-16T03:03:46.390Z
---

2026-10-15 사용자 "시군구 단위로 미리 조인된 기상데이터가 있다는데 db에서 찾아봐" → 읽기 전용 카탈로그 전수 조회(세션 [1번 베이스라인 실험 다시], 스크립트는 job tmp `db_catalog.py`·`db_cand.py`·`db_stats.py`·`sgg_cov.py`).

**있는 것 (전부 짧은 기간)**
- `predict_data.p0830_past_long`: 시군구 251 × 과거 99일(2026-05-23~08-29), 변수 11개(NAS식 이름 `tmprt_150`·`hd_150`·`arvlty_300`·`afp`·`일조`·`solrad_Qy`·`soil_Mitr_10`·`미세먼지` 등). `p0830_sigungu` 에 시군구마다 관측소 1곳(`wx_key`, `wx_dist_km`).
- `predict_data.p0830_fcst_long`: 시군구 251 × 11일 예보(2026-08-30~09-09). `p0830_meta` = 시군구 251 × 3단계 481 예측 격자(120,982행, 독소_생성 칼럼 — 아플라톡신용). `8_31_data` 는 hazard_label=아플라톡신.
- `predict_data.p0713/p0726/p0803/p0831_wx`: 232행(키 `P_<시군구 10자리>`) × 11변수 × 100일 가로형, 각 기준일용. `predict_data.sigungu`(233)·`sigungu_station`(가장 가까운 관측소 1곳).
- `api_data_info.kma_sigungu_point`: 시군구 대표 좌표 251개, 코멘트 "시군구 안 측정소(농업기상·에어코리아·ASOS) 좌표의 평균. 예보 수집 대상 좌표", 2026-10-14 생성, n_points 1~22, 103곳은 에어코리아 측정소만. `api_data_info.kma_predict_raw_hf`(GDPS 예보 2026-08-30 발표 1회, hf 1~288)·`p0830_sigungu` 좌표가 이 251개와 전부 일치.
- 예보 보관: `api_data_info.kma_weather_predict_info`(기상청 11일 예측) 발표일 2026-07-01·08-30 두 번뿐, `kr_pm_forecast_om` 2026-08-30~09-09 → 과거 예보로 학습 불가.
- `weather_data.weather_daily_eupmyeondong`: 읍면동 6,488 × 이틀(2016-07-02 N512, 2026-03-25 N128)뿐 — 쓸모 없음.
- `weather_data.interpolation_weather`: 관측소 218곳 2015-01-01~2026-09-03(플래그 OBS 등) — 시군구 아님.

**우리 시료와 대조**: 국내 시군구 코드(법정동 앞 5자리) 289 = 실제 275 + 미상 XX000 14. kma_sigungu_point 251개는 전부 우리 데이터에 있음 → 연결 620,143행(98.61%)·검출 1,823/1,844(98.86%). 안 맞는 24개(7,247행, 검출 8) = 옛·바뀐 코드: 부천시 41190(2,556행)·군위군 47720(907)·고양시 41280(884)·인천 남구 28170(811)·화성시 일반구 41591~41597·청원군 43710·수원 팔달구 41115 등.

**`ba_data_info.korea_region_code`(2026-10-15 사용자 제안 "여기 시군구 대표 코드 쓰면 될 듯")**: 칼럼 legal_dong_code·legal_dong_nm·molit_use_yn(국토교통부_폐지여부 존재/폐지)·level(sido 25·sigungu 512·eupmyeondong 13,606·ri 39,244)·kw_query·kw_matched_name·addr_matched_name·lon·lat·status·use_at (좌표·status 칼럼 코멘트 없음). 값으로 보면 시군구 좌표 = 시군구청 카카오 키워드 검색(`kw_query`="서울특별시 종로구청", status OK_KAKAO 222), 구청 없는 곳은 지역명 주소 검색(OK_KAKAO_ADDR 43, 화성시 일반구·수원 팔달구 등), 군위군 47720 은 27720 에서 채움(CODE_MAP_FILL), 폐지 코드는 좌표 없음. `predict_data.sigungu` 좌표와 같은 값. 우리 실제 시군구 275 중 266 좌표 있음 → 국내 625,862행(99.52%)·검출 1,831(99.30%); 좌표 없는 9개 = 폐지 코드(인천 남구 811·청원군 452·여주군 133·41012 97·창원시 48110 24 등, 1,528행, 검출 0); XX000 14개는 sido 좌표 전부 있음. kma_sigungu_point 와 거리 중앙값 2.24km·90% 7.89km·최대 94.0km(옹진군 — 군청이 섬 밖, 신안군 64.4km).

**How to apply:** 시군구 대표 지점은 korea_region_code 시군구청 좌표 사용이 유력(사용자 제안, 10-15 실측 지지) — 폐지 코드 9개·XX000·옹진군 처리 결정 필요. kma_sigungu_point 는 예보 수집 좌표라 예보를 쓸 때 참고. 옛 코드 24개 대응·미상 1,488행 처리는 사용자 결정 필요. 관련 [[listeria-production-unit-inputs]], [[listeria-exp1-baseline-decisions]], [[toxin-db-read-only-rule]].
