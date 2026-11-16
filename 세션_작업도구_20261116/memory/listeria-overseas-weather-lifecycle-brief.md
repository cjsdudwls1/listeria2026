---
name: listeria-overseas-weather-lifecycle-brief
description: "2026-09-15 브리핑 사실 — 해외 기상 DB 표 현황, 기준일 '선적일' 행 좌표 검증(한국 좌표 20행·일본 시설 좌표 4,194행), 제조~섭취(기간·온도·공정·섭취형태·유통기한·포장) 데이터 유무와 연결률"
metadata: 
  node_type: memory
  type: project
  originSessionId: d9b9b418-a320-4356-99d8-fbe2d77c9f46
  modified: 2026-09-15T05:26:04.667Z
---

2026-09-15 사용자 요청 브리핑(채팅으로 전달, 엑셀 미반영). 스크립트: worktree `리스테리아ML/scripts/report/brief/`.

해외 기상:
- 원본 `api_data_info.global_factory_weather`(표 설명 "Open-Meteo Archive(ERA5) 해외 제조업소 좌표 일별 기상"): 13변수(기온·습도 각 평균/최고/최저, 강수, 풍속 평균/최고/최저 km/h, 일사 MJ/m², 일조 초, 0~7cm 토양수분 ERA5-Land), 2015-01-01~2026-08-08, 수입 시료 좌표 3,072곳 전 기간(표본 30곳 4,238일 빠짐 없음), 날짜 UTC. 인덱스 (lat_g, lon_g, obsr_de).
- `global_weather_info` 나라당 1점 249개국 ~2026-07-07 / `weather_data.overseas_weather` = 단위 변환 + PM2.5 가공본 / `ba_data.t_global_risk_60d_weather` 아플라톡신 사건 전용.
- 해외에 없는 인자: 지중온도, 7cm 넘는 토양수분, 풍향, CO2·NO2·CO·SO2, 해수·수질. Open-Meteo 문서상 시간 단위 soil_temperature/soil_moisture 0-7/7-28/28-100/100-255cm, 일 wind_direction_10m_dominant 제공(ERA5 0.25°≈25km, ERA5-Land 0.1°≈11km).

선적일 행(47,072, 주소 출처 100% 해외제조업소):
- 한국 땅 좌표 7곳 20행 = 국내 경로 9행(평택 한글 주소, 해외 기상 없음) + 해외 경로 11행(부산·김포·청도 참치 업소).
- 좌표 나라 ≠ 수입 LIMS 제조국가코드 4,194행(좌표 210곳, 검출 26): 대부분 일본 시즈오카·요코하마 냉동창고/가공공장 좌표 + 제조국 TW·TN·TR·IT·ES 등(냉동 참치 횟감) → 시설 위치로 보임(추정).
- `ready_data.hazard_preprocessed_data.geocode_status` 원값 ok 62.4% / low_precision 37.6%. 칼럼 설명 없음. 스테이징 표에서 match_type city·district·state 등 = 전부 low_precision, house·street = 대부분 ok.
- 해외 경로 47,063행 모두 제조소 좌표 기상 전 기간·선적일 당일 100%.

제조~섭취: 시료별 공정·섭취형태·포장방식·실제 보관온도 칸은 DB에 없음. 대체 정보는 시료상태(수입 84.4%/국내 2.6%), 수입 품명 글자('냉동' 7,995·'횟감' 8,003행), 식품유형 정의(즉석섭취·신선편의 국내 6.0%), 대장 유통기한 문구(국내 30.9%)·포장재질 속 '진공' 4,194행, 수입 물류 날짜(선적→의뢰 중앙값 17일). distb_step_exam·mnf_step_exam 은 문서번호뿐, lpsms 포장재 칸 전체 NULL, 대장 haccp_appn_no 전체 NULL. 리서치팀 문서에 이 항목 출처 없음.

수입 시료 추가 확인(2026-09-15, `scripts/report/brief/imp_*.py`):
- 구성: 신고제품구분 축산물 38,002(검출 82, 대부분 치즈·아이스크림·햄 등 가공품)·수산물 8,004(43)·가공식품 1,421(0), 농산물 0행. HS 기준 원물(0201~0209 고기 767행 검출 19, 0302~0304·0306~0308 어패류 7,160행 검출 40).
- 도축·어획·수확 날짜·장소 칸은 DB에 없음(국내 표에만). 수입 LIMS 유통시작일: 수산물 99%·가공식품 99%가 선적 전(중앙값 37·41일) → 생산일 후보, 축산물은 선적 후 중앙값 289일 → 유통기한. 후보 8,647행 중 2015 이전 2행.
- `ba_data_info.lims_org_info` 는 splore_reqest_no 만으로 붙이면 다른 시료 제조일이 섞임(41.9%로 잘못 나옴) → cntc_sys_key2 까지 맞추면 수입 제조일 0%, 선적일은 s_org 와 100% 같음.
- B/L 번호 11자리 숫자(항공화물번호 형태) 7,261행 선적→입항 중앙값 1일, 영문 시작 28,141행 15일. 검사반입장소 이름에 냉동·냉장 22.1%. 해외제조업소 대장 연결은 수산물·가공식품 100%·축산물 0%.

**Why:** 수입 기상·유통 변수 설계와 모델팀 질문의 근거.
**How to apply:** 수입 기상 연결 전 한국 좌표 20행·일본 시설 4,194행 처리 방침을 먼저 사용자에게 확인. 관련 [[listeria-nb3-db-join-findings]], [[listeria-db-structure-findings]], [[feedback-raw-data-no-preprocessing]].
