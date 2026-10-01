# 리스테리아ML/data

각 파일이 언제·무엇으로·어떻게 만들어졌는지 간단히 적는다. 자세한 수치와 남은 의문은 `리스테리아ML/docs/00_보고서/작업기록.md` 의 해당 날짜 항목을 본다.
데이터를 추가·재생성하면 이 문서도 함께 고친다.

| 파일 | 행 × 열 | 만든 날 | 만든 방법 |
|---|---|---|---|
| `listeria_body.parquet` | 676,537 × 23 | 2026-09-08 추출 | DB 에서 내려받은 원자료 |
| `listeria_df_joined.parquet` | 676,537 × 41 | 2026-09-22~28, **2026-10-01 재생성**(검출 0% 품목을 지우지 않음 — 이전 600,040행) | 노트북 6 |
| `listeria_df_joined_full.parquet` | 676,537 × 40 | 2026-09-28, **2026-10-01 재생성**(의뢰 시스템 코드 3열 추가) | 노트북 6-1 |
| `X.parquet` | 676,537 × 11 | 2026-09-28, **2026-10-01 재생성**(검사기관 코드 제거, 78행 `src_route` 변경) | `EXP/1/1_데이터정제.ipynb` |
| `y.parquet` | 676,537 × 2 | 2026-09-28, 2026-10-01 재생성(값 같음) | `EXP/1/1_데이터정제.ipynb` |
| `collected/knhanes_2024/hn24_all.parquet` | 6,997 × 813 | 2026-09-28 | `scripts/knhanes_to_parquet.py` |
| `collected/knhanes_2024/hn24_24rc.parquet` | 543,782 × 110 | 2026-09-28 | `scripts/knhanes_to_parquet.py` |
| `collected/raw/raw_<테이블>.parquet` (8개) | 아래 표 | 2026-09-29 | `EXP/1/3_기상조인_후_학습.ipynb` 3절 — DB `SELECT *` 그대로 |
| `collected/exp1_wx/wx_<변수>.parquet` (17개) | 각 676,537 × 101 | 2026-09-29, **2026-10-01 재생성** | `EXP/1/3_기상조인_후_학습.ipynb` 4절 |
| `collected/exp1_wx/food.parquet` | 676,537 × 17 | 2026-09-29, **2026-10-01 재생성**(값 표기 3열 추가) | `EXP/1/3_기상조인_후_학습.ipynb` 5절 |
| `collected/exp1_wx/join_summary.csv` | 17 × 10 | 2026-09-29, 2026-10-01 재생성 | `EXP/1/3_기상조인_후_학습.ipynb` 4-5절 (조인 점검표) |
| `exp2/tabular.parquet` | 600,040 × 368 | 2026-09-29, **2026-10-01 18:06 재생성**(다운사이징 기준 변경 — 이전 600,075행) | `EXP/2/1_데이터준비.ipynb` (다운사이징 + 변수 제거 + 기상 창 집계) |
| `exp2/seq_std.npy` | 600,040 × 100일 × 17변수 (float16, 2.0GB) | 2026-09-29, 2026-10-01 18:06 재생성 | `EXP/2/1_데이터준비.ipynb` (딥러닝 시계열 입력, train 통계 표준화) |
| `exp2/meta.json` | — | 2026-09-29, 2026-10-01 18:06 재생성 | `EXP/2/1_데이터준비.ipynb` (칼럼 목록·삭제 품목·표준화 통계) |

## listeria_body.parquet — 리스테리아 시료 원자료

- **원천**: `staging_data.export_body_listeria`(모델팀 작업 테이블). 모델팀 DB 함수 `staging_data.build_export_qual` 이 `staging_data.hazard_preprocessed_data`(위해요소 통합 전처리 결과)에서 아래 조건으로 만든 테이블이다.
  - `hazard_label = '리스테리아'`, `lat IS NOT NULL`, (`geo_route='overseas'` 또는 `addr_source <> '검사기관'`), (`anals_se IS NULL` 또는 농산물), `mnfctur_de_or_fallback` 2015-03-02~2026-06-30
  - 그래서 `ready_data.hazard_preprocessed_data` 리스테리아 765,891행보다 적고, 주소 출처가 '검사기관'인 국내 행이 빠져 있다.
- **추출 시점**: 2026-09-08. 누가 어떤 코드로 내려받았는지는 기록이 없다(확인 필요). git 첫 커밋(2026-09-10) 이후 바뀌지 않았다.
- **구성**: 국내(`src_route`=domestic) 628,895행, 수입(overseas) 47,642행. 타깃 `judge_result` 검출 1,969건(0.291%). 기준일 `mnfctur_de` 2015-04-07~2026-06-30. 행 키 `lims_data_key`.
- **주의**: `train_prdlst_code`(모델팀 학습용 품목코드)는 추출 시점 값이라 현재 DB 와 9,236행 다르다(옛 C01·C02·A12 코드 행, 작업기록 2026-09-22). 재추출 여부는 미결.
- 이 폴더에서 git 이 추적하는 유일한 데이터 파일이다.

## listeria_df_joined.parquet — 전체 행 + 가공식품 판정 열 + 품목 계층 조인

- **만든 곳**: `notebook/6_교수님과 미팅 후 품목 코드까지 조인.ipynb` 의 마지막 저장 셀(노트북을 처음부터 실행하면 저장된다).
- **2026-10-01 재생성 (사용자 결정)**: 노트북 6 의 2절이 검출 0% 품목을 **지우지 않게** 바뀌었다. 이전 저장본은 검출이 한 건도 없는 `train_prdlst_nm` 38개(76,497행)를 지운 600,040행이었고, 지금은 `listeria_body.parquet` 676,537행 그대로다.
- **과정**: `listeria_body.parquet` 676,537행(행 삭제 없음) → 노트북 5절의 가공식품 판정 열 4개(`is_processed`·`가공식품_여부`·`train_prdlst_code_str`·`has_C_prefix`) 추가 → `train_prdlst_code` = `ba_data.t_prdlst_cl.prdlst_cd`(DB 테이블 코멘트 "식품코드") 왼쪽 조인으로 `cl_kor_nm`·`cl_lv`·`lv1_cd`~`lv6_nm` 14열 추가. DB 는 읽기 전용 SELECT.
- 원자료 23열은 값·행 순서·형식 그대로다(재생성 뒤 `listeria_body.parquet` 와 대조해 같음을 확인). `train_prdlst_code` 는 원자료 형식(category)이고, 이전 저장본은 문자열이었다.
- `train_prdlst_code` 가 없는 '기타' 170행은 계층 14열이 비어 있고(`cl_lv` 는 Int64 `<NA>`), `train_prdlst_code_str` 은 빈 문자열이다.
- `is_processed`(코드가 C 로 시작) 332,934행과 `lv1_cd` 가 `C0000000000000` 인 333,276행은 `X0100027500000` 342행에서 갈린다.
- 검출 1,969건. 이전 600,040행 저장본은 이 파일에서 검출이 한 건이라도 있는 `train_prdlst_nm` 15개의 행만 고르면 다시 만들 수 있다.
- `listeria_df_joined_full.parquet` 와의 차이: 이 파일에는 가공식품 판정 열 4개가 있고 의뢰 시스템 코드 3열이 없다. 행은 같다.

## listeria_df_joined_full.parquet — 전체 행 유지 + 품목 계층 조인

- **만든 곳**: `notebook/6-1_품목코드 라벨링_전체행 유지.ipynb` 마지막 셀(혼자 실행해도 같은 파일을 다시 만든다). 2026-09-28 사용자 요청.
- **과정**: `listeria_body.parquet` 676,537행을 그대로 두고 노트북 6 과 같은 방식으로 `ba_data.t_prdlst_cl` 계층을 조인(DB 는 읽기 전용 SELECT). 원자료 23열 + 계층 14열. 행 필터·결측 채움 없음.
- `train_prdlst_code` 가 없는 '기타' 170행은 계층 열이 비어 있다.
- 노트북 6 의 가공식품 판정 열 4개는 넣지 않았다(코드 첫 글자 기준과 `lv1_cd` 기준이 `X0100027500000` 342행에서 갈리기 때문).
- 2026-09-28 대조: 이 파일에서 검출이 있는 15개 품목만 고르면 당시 `listeria_df_joined.parquet`(600,040행 저장본)와 공통 37열 값이 같았다. 2026-10-01 에 `listeria_df_joined.parquet` 도 676,537행이 되어 이 대조는 옛 저장본 기준이다.
- **2026-10-01 추가 3열** (노트북 6-1 의 7-1절, `lims_data_key` 로 왼쪽 조인, DB 읽기 전용 SELECT) → 40열
  - `reqest_sys_code`: `ba_data.m_org_lims_data_info.reqest_sys_code` (DB 코멘트 "ZCI07 의뢰 시스템 코드"). `ba_data.m_org_lims_data_info` 에 없는 7,203행은 비어 있다.
  - `reqest_sys_nm`: 그 코드의 이름(`ba_data.t_cm_cmmn_code.cmmn_code_nm`, `cmmn_cl_code`='CI07'). 자체의뢰 593,955 · 통합망 35,141 · 수입축산물 31,583 · 수입수산물 7,279 · 수입식품 1,376 · 비어 있음 7,203.
  - `import_lims_linked`(bool): `ready_data.hazard_preprocessed_data.cntc_sys_key2` 가 `ba_data.s_org_lims_data_info.rcept_no`(수입식품 LIMS 검사 결과, `label`='리스테리아')에 있으면 True. True 47,441행(검출 125) / False 629,096행(검출 1,844).
  - 의뢰 시스템 코드 이름이 '수입…' 이거나 비어 있는 행 = `import_lims_linked` True 행(예외 0). `src_route` 와는 221행이 어긋난다(overseas 인데 False 211행, domestic 인데 True 10행). 이 열들은 "어느 의뢰 시스템으로 들어온 검사인가"를 가르며, 주소의 나라나 제조국을 뜻하지 않는다.

## X.parquet · y.parquet — 베이스라인 실험 1 입력(정제만 한 원자료)

- **만든 곳**: `EXP/1/1_데이터정제.ipynb` 의 저장 셀(`X.to_parquet`/`y.to_parquet`). 2026-09-28 사용자 실행. 실험 목표는 `EXP/1/readme.md`(튜닝·불균형 처리·기상 조인 없이 성능 확인).
- **과정**: `listeria_body.parquet` 676,537행 × 23열 → 행은 그대로(필터·결측 채움 없음), 열만 삭제 (2026-10-01 부터 0번 단계 추가)
  0. **(2026-10-01, 사용자 결정) 로마자로 적힌 한국 주소 78행의 `src_route` 를 overseas → domestic 으로 바꾼다.** `src_route` 는 DB `geo_route`(정의서: "domestic(한글주소→법정동) / overseas(비한글→지오코딩)")와 같은 값이라 한국 주소가 로마자로 적힌 행이 overseas 에 들어가 있었다. 찾는 방법은 `addr` 낱말(KOREA·도·광역시 이름·-EUP·-MYEON)과 좌표(domestic 행 좌표에서 0.5도 이내) 두 가지이고, 두 방법이 같은 78행을 가리킬 때만 바꾸도록 노트북에 assert 가 있다. 78행 전부 불검출(제조일·제조사소재지 67 + 선적일·해외제조업소 11). 원자료 `listeria_body.parquet` 는 그대로다.
  1. 결측 98% 이상 3열 삭제: `result_value_num`(99.44%), `judge_stdr`(98.92%), `judge_stdr_num`(99.99%)
  2. 모델 입력으로 쓰지 않는 열 삭제: `hazard_label`·`data_source`(값 1개), `expr_iem_nm`, `prduct_nm`·`addr`·`mnfctur_cmpny_nm`(자유 텍스트), **`inspct_instt_code`(검사기관 코드 — 예측 시점에 알 수 없는 값. 2026-10-01 교수님 지적, 사용자 결정으로 제거)**. 행 키 `lims_data_key` 는 조인용으로 남긴다(2026-09-28 사용자 결정).
  3. `result_value`(검사 결과 문자열, `judge_result` 와 1:1 대응하는 정답 누수) 삭제 — X 에도 y 에도 없다.
  4. `judge_result` 를 y 로 분리하고 검출=1, 불검출=0 (int8) 으로 변환. y 에는 행 키 `lims_data_key` 도 함께 넣는다(2026-09-28 사용자 결정 — 노트북 [9] 셀, X·y 키 순서가 같은지 assert 로 확인)
- **X 11열**: `lims_data_key`(string) · `src_route`, `prdlst_code`, `train_prdlst_code`, `train_prdlst_nm`, `mnfctur_de`, `date_basis`, `addr_source`, `legal_dong_code`(category) · `lat`, `lon`(float32). `lims_data_key`(행 키)와 `mnfctur_de`(datetime, 2015-04-07~2026-06-30, 시간순 분할 기준)는 보조용이며 모델 피처에서 뺀다. `src_route` 는 domestic 628,973 · overseas 47,564(원자료는 628,895 · 47,642). 결측: `train_prdlst_code` 170행, `legal_dong_code` 47,642행(overseas 47,564행 + domestic 으로 옮긴 78행 — 옮긴 행의 법정동 코드는 원자료에 없어 비워 둠).
- **y 2열**: `lims_data_key`(string, X 와 같은 행 순서) · `judge_result`(int8) 1=1,969 / 0=674,568 (0.291%).
- **주의**
  - X·y 는 `lims_data_key` 로 맞춘다. 조인·정렬·필터 뒤에는 키로 다시 맞추거나 키 순서가 같은지 확인한다(`lims_data_key` 는 676,537개 모두 고유).
  - 2026-09-28 18:05 재생성 전의 `y.parquet` 는 `judge_result` 1열 버전이었다(X 와 행 순서로만 대응). 재생성 때 X 는 이전 파일과 값·형식이 완전히 같고, y 의 `judge_result` 도 이전과 같다(대조 확인). `EXP/1/2_바로학습.ipynb` 1절 로드 셀은 2026-09-29 현재 `lims_data_key` 로 맞추도록 바뀌어 있다.
  - 2026-09-28 16:04 에 처음 저장한 파일은 `lims_data_key` 가 없는 11열 버전이다. 노트북을 처음부터 다시 실행해야 위 12열 버전으로 바뀐다.
  - `train_prdlst_code`·`train_prdlst_nm` 은 2026-09-08 추출 값(현재 DB 와 9,236행 다름, 위 `listeria_body.parquet` 주의 참고).
  - `date_basis` 가 예측 시점에 알 수 있는 정보인지는 확인 전이다. `inspct_instt_code` 는 2026-10-01 에 뺐다.
  - 2026-10-01 재생성 전 파일과 대조: 행 키 순서 같음, `src_route` 78행만 다름, 나머지 10열 값 같음, y 값 같음. 이전 12열 파일로 만든 MLflow 기록은 `리스테리아ML/mlflow_backup_20261001/` 로 옮겼다.

## collected/knhanes_2024/ — 국민건강영양조사 2024

- **원천**: 사용자가 받은 SAS 파일 `hn24_all.sas7bdat`(1명 1행), `hn24_24rc.sas7bdat`(24시간 회상 — 한 사람이 먹은 음식의 재료 1개가 1행).
- **만든 방법**: `py 리스테리아ML/scripts/knhanes_to_parquet.py` 로 parquet 변환(2026-09-28). 값은 바꾸지 않고 문자열을 cp949 로 풀기만 했다. 풀리지 않는 39칸(문자열 끝 글자가 잘린 것)은 �로 바뀌어 있다.
- 리스테리아 시료와 공통 키·조사 날짜가 없어 행 단위로 붙일 수 없다. 쓸 만한 컬럼 35개 선정 결과는 `docs/02_데이터수집/국민건강영양조사2024_컬럼목록.xlsx`.
- `collected/` 아래는 git 추적에서 제외된다.

## collected/raw/ — 실험 1 기상 조인용 DB 원자료 (2026-09-29)

- **만든 곳**: `EXP/1/3_기상조인_후_학습.ipynb` 3절. 읽기 전용 세션(`default_transaction_read_only=on`)에서 SELECT 만, 값·단위 변환 없이 저장(DB numeric 은 float64 로만 바꿈). 파일이 있으면 노트북이 DB 에 다시 가지 않는다(`FORCE_DB=True` 면 다시 받음).

| 파일 | DB 테이블 | 행 × 열 | 범위 |
|---|---|---|---|
| `raw_kma_asos_daily_info.parquet` | `api_data_info.kma_asos_daily_info` (ASOS 일자료) | 406,909 × 61 | 전체 (2015-01-01~2026-09-03, 99지점) |
| `raw_kma_ground_obs_station_info.parquet` | `api_data_info.kma_ground_obs_station_info` (ASOS 지점, 위치 이력 포함) | 243 × 21 | 전체 |
| `raw_nas_agri_weather_daily_info.parquet` | `api_data_info.nas_agri_weather_daily_info` (NAS 일자료) | 984,648 × 17 | 전체 (2008-01-01~, 219지점) |
| `raw_nas_agri_weather_obs_station.parquet` | `api_data_info.nas_agri_weather_obs_station` (NAS 지점) | 219 × 16 | 전체 |
| `raw_keco_airkorea_fine_dust_info.parquet` | `api_data_info.keco_airkorea_fine_dust_info` (에어코리아 일평균) | 2,150,630 × 8 | 전체 (2015-01-01~, 691측정소) |
| `raw_keco_airkorea_obs_station.parquet` | `api_data_info.keco_airkorea_obs_station` (에어코리아 측정소) | 753 × 12 | 전체 |
| `raw_t_aw_cl.parquet` | `ba_data.t_aw_cl` (품목별 수분활성도·식품pH·염도) | 1,316 × 9 | 전체 |
| `raw_global_factory_weather_listeria.parquet` | `api_data_info.global_factory_weather` (해외 제조업소 좌표 일별 기상, ERA5) | 12,900,454 × 18 | 해외 시료 좌표 3,071곳 × 2014-12-28~2026-06-29 만 (테이블 약 2.3억 행) |

- 7개 작은 테이블은 2026-09-29 11:39, 해외 기상은 13:24 에 받았다.

## collected/exp1_wx/ — 실험 1 기상·대기·식품 조인 결과 (2026-09-29, 2026-10-01 재생성)

- **만든 곳**: `EXP/1/3_기상조인_후_학습.ipynb` 4~5절. 조인 규칙과 코드는 노트북에 있다(계획 셀·4절 표). `FORCE_JOIN=True` 면 다시 만든다.
- **행**: 모든 파일이 `X.parquet` 와 같은 676,537행·같은 행 순서, 첫 칼럼 `lims_data_key`.
- **`wx_<변수>.parquet` 17개**: 변수마다 `<변수>_d001`(기준일 `mnfctur_de` 전날) ~ `<변수>_d100`(100일 전) float32. 변수 = `avg_temp`·`max_temp`·`min_temp`·`rain`·`avg_wind`·`max_wind`·`sunshine`·`solar_rad`·`soil_t5`·`soil_t10`·`soil_t20`·`soil_t30`·`soil_t_nas`·`soil_moist`·`no2`·`co`·`so2`.
  - 국내 행: 시료 `lat`·`lon` 에서 위경도 유클리드 거리로 가까운 관측소 3곳 평균(그날 값 있는 관측소 중, ASOS 는 그날 위치). 기본 칼럼이 없으면 대체 칼럼(기온·풍속·일사량만 NAS 대체).
  - 해외 행: `global_factory_weather` 같은 좌표의 그날 값(기온 3개·강수·일사량). 단위가 다르거나 자료가 없는 변수는 결측.
  - 결측 채움·단위 변환 없음.
  - **2026-10-01 재생성**: `X.parquet` 의 `src_route` 가 바뀌어 로마자 한국 주소 78행에 국내 관측소 값이 붙었다(이전에는 `global_factory_weather` 값 5변수만, 12변수 결측). 나머지 행의 값은 이전과 같다(재생성 뒤 대조 결과는 작업기록 2026-10-01).
  - overseas 47,564행은 12변수(`avg_wind`·`max_wind`·`sunshine`·`soil_t5`·`soil_t10`·`soil_t20`·`soil_t30`·`soil_t_nas`·`soil_moist`·`no2`·`co`·`so2`)가 100일 전부 결측이다.
- **`food.parquet`**: `aw`·`ph`·`na`(float32) + 칸마다 쓴 조인 키(`*_key`: `prdlst_code`/`train_prdlst_code`), DB 원문(`*_raw`), 숫자 변환 규칙(`*_rule`: 숫자 그대로 / 범위 → 가운데 값 / 부등호 → 경계값), **값 표기(`*_label`: 단일값 / 중앙 / 이상 / 이하 / 초과 / 미만, 값이 없으면 빈 문자열 — 2026-10-01 사용자 지시로 추가)**, `aw_source`(t_aw_cl 원값). 범위·부등호 변환은 2026-09-29 사용자 결정. 값 표기는 원문 기호에 따른다: 숫자 하나 = 단일값, `a~b` = 중앙, `≥` = 이상, `≤` = 이하, `>` = 초과, `<` = 미만.
- **`join_summary.csv`**: 변수별 국내·해외 값 있는 비율, 대체 칼럼 사용 비율, 고른 관측소까지 거리.

## exp2/ — 실험 2 입력 (2026-09-29, 2026-10-01 재생성, git 제외)

- **만든 곳**: `EXP/2/1_데이터준비.ipynb` (Python 3.12 가상환경 `.venv-exp2`, 커널 `listeria-exp2`). 입력은 `X.parquet`·`y.parquet`·`collected/exp1_wx/`(기상 17변수 × D-1~D-100, 식품 3). 사용자 지시 2026-09-29.
- **행 600,040** = `X.parquet` 676,537행에서 **전체 기간(train·val·test)에 검출이 0 인 `train_prdlst_nm` 38종의 행 76,497개를 삭제**(다운사이징, 2026-10-01 18:06 재생성 — 사용자 지시). 삭제 목록은 `meta.json` 의 `downsizing.zero_items`(품목명)·`zero_item_codes`(`train_prdlst_code`). train 440,116(검출 1,491) · val 78,420(191) · test 81,504(287).
  - 이전 기준(2026-09-29 ~ 2026-10-01 17:27 저장본): train(~2024-06-30)에 검출이 0 인 `train_prdlst_code` 37종·76,462행 삭제 → 600,075행. 바뀐 것은 train 에 행이 없는 `환경검체`(`Z4100000000000`) 35행(val 21·test 14, 검출 0)이 더 지워진 것 하나다.
  - 새 기준은 품목 목록을 정할 때 val·test 의 검출 여부도 본다.
- **`tabular.parquet` 368열** = `lims_data_key`, `split`, `y`, `mnfctur_de` + 피처 364: 범주형 8(LIMS 5 `src_route`·`prdlst_code`·`train_prdlst_code`·`date_basis`·`addr_source` + 식품 값 표기 3 `aw_label`·`ph_label`·`na_label`, category) + 식품 3(`aw`·`ph`·`na`, float32) + 기상 창 집계 353(float32). 2026-10-01 재생성: `inspct_instt_code` 제거(이전 366열·피처 362), 값 표기 3 추가, 78행 국내 기상 반영. 뺀 칼럼: `lat`·`lon`·`legal_dong_code`·`train_prdlst_nm`(성능향상_방안 §3), 기상 하루값 1,700칼럼(창 집계로 대체). 값은 원자료 그대로(결측 유지, 표준화 없음).
  - 창 집계: 변수 17개 × 창 7·14·30·100일(D-1~D-w) × 평균·최고·최저·표준편차(`{변수}_mean7` 등) + 강수 `rain_sum{w}`·`rain_wet{w}`(>0mm 일수)·`rain_maxdry{w}`(최장 무강수, 결측=0mm) + 생육 범위 일수 `avg_temp_ge30_{w}`(≥30℃)·`avg_temp_opt_{w}`(30~37℃)·`min_temp_gem2_{w}`(≥−2℃)·지중온도 5개의 `_opt_{w}`(8~15℃)·`_lim_{w}`(8~30℃) + `{변수}_nmiss100`(100일 중 결측 일수). 기준값은 리서치팀 예측인자 PPT(성능향상_방안 §2-11). 해외 행은 원래 없는 12개 변수의 집계가 NaN.
- **`seq_std.npy`**: (600,040, 100, 17) float16. 시간 순서 0 = D-100 … 99 = D-1(하루값 파일과 반대 방향). 변수마다 **train 행의 평균·표준편차로 표준화**하고 결측은 0(= 평균). 통계는 `meta.json` 의 `seq.stats`. 행 순서는 `tabular.parquet` 과 같다.
- **`meta.json`**: 만든 시각, 입력 파일·수정 시각, 분할 건수, 삭제 품목, 칼럼 목록(`cat_cols`·`food_cols`·`agg_cols`·`features`), 창·기준값, 시계열 통계.

## collected/probe/ — 성능 향상 후보 점검용 LIMS 원본 코드 (2026-09-29)

- **`morg_listeria_codes.parquet`** (758,574 × 14): `ready_data.hazard_preprocessed_data`(위해요소 통합 전처리 결과) 의 `hazard_label='리스테리아'` 행에 `ba_data.m_org_lims_data_info`(식약처 LIMS 원본)를 `lims_data_key` 로 조인해 코드·날짜 칼럼만 SELECT 한 것. 값 변환 없음. 읽기 전용 세션(`default_transaction_read_only=on`).
  - 칼럼: `lims_data_key`, `inspct_purps_code`(검사 목적 코드, IM02), `inspct_knd_code`(검사 종류 코드, IM03), `reqest_sys_code`(의뢰 시스템 코드, CI07), `jdgmnt_fom_code`(판정 형식 코드, IM15), `cnvync_mth_code`(운반 방법 코드, IM08), `splore_sttus_code`(시료 상태 코드, IM11), `inspct_splore_cnt`(검사 시료 카운트), `reqest_de`(의뢰 일), `inspct_compt_de`(검사 완료 일), `pick_de`(채취 일), `m_mnfctur_de`(m_org 의 제조 일 원본), `distb_de`(유통 일), `valid_tmlmt`(유효 기한). 코드 뜻은 `ba_data.t_cm_cmmn_code`(`cmmn_cl_code`·`cmmn_code`·`cmmn_code_nm`)에서 확인.
  - `listeria_body.parquet`(676,537행)보다 많은 이유: DB 리스테리아 행 전체(주소 출처 '검사기관' 행 포함)를 받았기 때문. 사용할 때 `lims_data_key` 로 왼쪽 조인한다.
  - 쓰는 곳: `scripts/probe_feature_candidates.py`(탐색용, MLflow 기록 없음). 정식 실험에 쓰려면 `collected/raw/raw_m_org_lims_data_info_listeria.parquet`(SELECT *) 로 다시 받는 것이 원칙에 맞다.
