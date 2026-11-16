# 리스테리아 최종 실험 — 구현 공통 규칙 (모든 구현·검수 에이전트가 따름)

## 기준 문서
- 실험 계획서(유일한 사양): `C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\최종실험_계획_20261102.md` (UTF-8). 계획서에 없는 설정은 새로 만들지 말고, 꼭 필요한 구현 세부는 코드 주석과 반환 보고에 '구현 세부(계획서에 없음)'로 적어 담당자 승인을 받는다(계획서 §8 "문서에 없는 구현 세부는 단계 0 에서 정해 … 고르기·점수에 영향이 있으면 담당자 확인 뒤 적용").
- 교수님 지시·계획서 내용을 바꾸지 말 것. 계획서와 다르게 해야 할 이유가 보이면 코드를 쓰지 말고 보고.

## 코드 위치
- 로컬(작성): `C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\` — 파이썬 패키지 `lst/`(모듈), `scripts/`(실행 진입점, 단계 번호 이름 예: `s0_1_samples.py`), `tests/`(pytest), `configs/`(JSON 설정).
- 서버(실행): `~/cyj_listeria_final/code/`(코드 사본), 자료 뿌리 `~/cyj_listeria_final/data/`(로컬 `C:\Users\NT751\Desktop\천영진\` 아래 상대 경로 그대로), 결과 `~/cyj_listeria_final/results/`.
- 경로: 코드·설정에 절대 경로 금지. 자료 뿌리는 실행 인자 `--root`(필수), 결과 뿌리는 `--out`(필수). 스크립트 맨 앞에서 경로를 덮는 환경변수(PYTHONPATH 외 경로 관련) unset(계획서 B-2).
- 설정: JSON, 모르는 칸이면 오류로 멈춤, 계획서에 없는 칼럼이면 멈춤(B-2).
- 결과마다 `code_md5`·입력 md5·스레드 수·설정 덤프를 같이 저장(B-2).

## 지표 (가장 중요 — 지난번 'PR-AUC 라 해 놓고 AP 코드' 사고를 되풀이하지 말 것)
- PR-AUC = sklearn `precision_recall_curve` → `auc`(trapz), `drop_intermediate=False`. `average_precision_score` 는 어디에도 쓰지 않음(import 도 금지).
- 동점은 u_tie(부록 A-1)로 풀어 쌍마다 한 점(부록 A-12). sklearn 기본(같은 값 묶음) PR-AUC 는 따로 기록만.
- 구간 가중 = 0.5·0.3·0.2(h 1~20·21~40·41~60), NaN 규칙은 계획서 §1-2 표 그대로.

## 서버 자원(공용 — 파튤린 담당이 같은 계정으로 GPU 를 쓰는 중)
- 내 폴더(`~/cyj_listeria_final`) 밖에는 쓰지 않음. `patulin_optuna`·`norovirus` 와 남의 프로세스는 읽지도 건드리지도 않음. 프로세스 종료는 내가 띄운 PID 만.
- 스레드: OMP/MKL/OPENBLAS/NUMEXPR 스레드 8, 알고리즘 n_jobs ≤ 8.
- GPU: `torch.cuda.set_per_process_memory_fraction(0.4)`(16GB) 이하.
- RAM: 가용 메모리 15GB 아래면 내 작업만 멈춤. 디스크: 여유 10GB 아래면 내 작업만 멈춤(감시 함수 공통 모듈에).
- 학습이 멈추면(메모리 오류 등) 건너뛰거나 실패로 넘기지 않음 — 원인을 없앤 뒤 그 판부터 다시(계획서 B-1).
- 입력 캐시를 디스크에 쌓지 말 것(B-1), 꼭 필요하면 md5 이름표.

## DB (TOXIN_MODEL_PRED, 공용 [권한 가림] 계정)
- SELECT 만. 서버·계정 설정 변경, TEMP 테이블, 쓰기 금지. 접속 정보는 `C:\Users\NT751\Desktop\천영진\리스테리아ML\.env`(값을 출력·로그·파일에 남기지 말 것).
- DB 는 로컬 PC 에서만 닿음(서버에서는 안 닿음). 추출 결과는 parquet 로 저장해 서버에 올림.

## 자료 원칙
- 원자료 그대로. 결측 채움·단위 변환·필터·집계는 계획서에 적힌 것만.
- 좌표 매칭은 시료 파일의 lat·lon(DB 좌표 아님). float32 등호 비교 금지.
- 제품명(`prduct_nm`)은 단계 4 줄이기 칼럼 고르기·행 고르기에만. 모델 입력·보고·추정 근거에 쓰지 않음.
- 기존 데이터 파일을 덮어쓰지 말 것(새 파일로).

## 파이썬
- 로컬 시험: `C:\Users\NT751\Desktop\천영진\리스테리아ML\.venv-exp2\Scripts\python.exe`(3.12, numpy·pandas·pyarrow·sklearn·scipy). 로컬 anaconda python·`py` 는 쓰지 말 것.
- 서버 실행: `~/cyj_listeria_final/venv/bin/python`.
- 작성 중 시험은 합성 자료나 작은 조각으로만, 실제 결과 파일은 만들지 않음. 실제 실행(결과를 만드는 실행)은 검수 에이전트 검수를 통과한 뒤 지휘자가 함.

## 반환 보고(한국어, 짧게)
- 만든 파일 목록, 실행 방법(명령), 계획서 절과의 대응, '구현 세부(계획서에 없음)' 목록, 시험 결과, 남은 문제.

## 추가(11-07) — 이미 있는 것
- 핵심 라이브러리 `리스테리아ML2\exp\final\lst\`: paths(`parse_paths` --root/--out, `resolve`), config(엄격 JSON·`save_run_meta`·md5), resources(`set_thread_env`·`check_resources`·`limit_gpu_memory`), hassign(`assign_h`·`u_tie`), metrics, folds(`SPLITS`·`check_split_counts`·`make_train_rows`·`scoring_pairs`·덜 찬 달 틀), windows(`window_bounds`·`DailyCube`·`window_features`·`sequences`·`anomaly_*`·`rr8_mask`). 새 코드는 이 모듈을 import 해서 쓰고, 같은 기능을 다시 만들지 말 것. 모듈 API 를 바꿔야 하면 바꾸지 말고 보고.
- DB 추출본(읽기 전용, 2026-11-07): `리스테리아ML2\data\추출_20261107\` — s_org_lims_data_info_listeria(53,176행, rcept_no 중복 8,554행이나 나라 코드 다른 중복 0), hazard_preprocessed_data_nation(765,891행, 시료 lims_data_key 100% 연결, 해외 47,642행 중 cntc_sys_key2 없는 행 211), nation_code(249), global_weather_info(1,068,459행, 249국, 2015-01-01~2026-09-30, 나라마다 lat·lon 1곳 고정 — country_lat·lon 과 다름), global_soil_ph(235), kma_sigungu_point(251), p0830_fcst_long(2,761), kma_weather_predict_info(7,092). 표마다 옆 json 에 쿼리·md5.
- 서버 실행 모양: `~/cyj_listeria_final/venv/bin/python scripts/<스크립트> --root ~/cyj_listeria_final/data --out ~/cyj_listeria_final/results/s0/<단계>` (자료 뿌리 아래는 로컬 `천영진\` 과 같은 상대 경로, 위 추출본도 올려 둠). 서버는 인터넷이 됨(Open-Meteo·pypi), 로컬 venv 파이썬은 방화벽으로 인터넷·DB 가 막힘.
- 작성 중 실제 자료는 구조 확인용으로 일부만 읽기(읽기 전용, 결과 파일 안 만듦). 전체 실행은 검수 뒤 지휘자가.

## 추가(11-07 오후) — 계획서 v74(PR #12) 반영·단계 0 결과 위치
- 계획서가 바뀜(같은 파일): 해외 기상 = DB `global_weather_info` 값만(Open-Meteo 안 씀), DB 에 없는 변수(soil_t5~t30·soil_t_nas·no2·co·so2·풍향 sin·cos·이슬점)·날짜 = NaN, 1차 기상 = 2차 기상, 해외 행에서 늘 결측인 그 변수들의 결측 표시는 '해외 구조적 결측' 1칼럼(§4-2). 해외 시료 나라 95곳 모두 DB 에 있음.
- 단계 0 결과(서버 `~/cyj_listeria_final/results/s0/…`, 로컬 사본 `C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror\s0\…` — 읽기만):
  - `0-1/apply/samples.parquet`(676,150행): lims_data_key·y(int8)·src_route·sgg(해외 = 'OV'+나라 alpha-3)·sido(해외 'OV')·p2·p3·d(date)·h(int16, 부록 A-1)·nation_a3·nation_src·rr1_moved·rr3_p2_nocode·rr3_p3_nocode·rr5_coord_differs·rr6_overseas_only_p3·rr7_jul01. 원 시료 파일 칼럼(제품명 등)은 `리스테리아ML\data\listeria_data.parquet` 를 lims_data_key 로.
  - `0-4_v2/시군구_일별_기상_0-4_합본.parquet`(sgg_code·date·19변수 + rh·td, 256 시군구 × 4,573일), `0-4_v2/시군구_대표좌표_0-4.parquet`(sgg_code·sgg_nm·lat·lon·source), 습도이슬점·옹진군·2/29 채움 파일.
  - `0-4_weather_overseas/해외_나라_일별_기상_1차.parquet`(= 2차, alpha_3·sgg('OV'+alpha_3)·date 2015-01-01~2026-06-30·19변수·avg_rhm·avg_td·src_<변수>) — 국내는 습도·이슬점 이름이 rh·td, 해외는 avg_rhm·avg_td(이름 맞추기는 쓰는 쪽에서).
  - `0-3/`: 예보 준비(운영용, 실험 입력 아님).
- 단계 0 결과를 입력으로 읽는 새 스크립트는 `--inputs`(결과 뿌리, 서버 `~/cyj_listeria_final/results`, 로컬 시험 `…\tmp\results_mirror`)를 필수 인자로 받고, 설정 JSON 에 그 아래 상대 경로(예: `s0/0-1/apply/samples.parquet`)를 적음. `--root`(자료 뿌리)·`--out` 규칙은 그대로.
- 인터넷 받기가 필요하면(면적 통계 등) 로컬 `py -I`(3.14, pyarrow·openpyxl, pandas 깨짐) 또는 서버(인터넷 됨)로, 받은 파일은 새 빈 폴더에 따로, 출처 URL·받은 날·기준 연도 기록. DB 다시 조회 금지(받아 둔 parquet 만).
- 환경 확인(11-07): 서버 scipy 1.18.1·keras 3.15.1(KERAS_BACKEND=torch)·autokeras 3.0.0·tensorflow 2.21.0(autokeras import 용)·mlflow 3.16.1(로컬 .venv-exp2 와 같은 판으로 맞춤). CatBoost Focal(`loss_function='Focal:focal_alpha=…;focal_gamma=…'`) 동작, CatBoost has_time 기본 False.
  - AutoKeras 3.0.0 + Keras 3.15.1: 출력층 Dense 에 units 가 numpy.int64 로 넘어와 멈춤 → AutoKeras 를 쓰기 전 `keras.layers.Dense.__init__` 에서 units 를 int 로 바꾸는 패치(값 그대로)를 공통 모듈에 두고 씀(구현 세부). `AutoModel.predict` 는 분류 라벨(0/1)을 돌려주므로 확률은 `export_model().predict` 로.
  - MLflow: 서버 `mlflow server --backend-store-uri sqlite:///<서버>/cyj_listeria_final/mlflow/mlflow.db --artifacts-destination <서버>/cyj_listeria_final/mlflow/artifacts --host [IP 가림] --port 5055`, 기록은 `http://[IP 가림]:5055`. 로컬 복사 = mlflow.db(sqlite3 backup 으로 뜬 사본) + artifacts 폴더를 `리스테리아ML2\exp\mlflow\` 로, 로컬 확인 = 같은 두 경로로 `mlflow ui`(시험: 서버 기록 run·metric·tag·artifact 를 로컬에서 읽음 확인).
  - 로컬(Windows)에서 `mlflow server/ui` 를 끌 때는 자식 uvicorn 까지 끔(프로세스 트리 — `taskkill /T /PID <띄운 PID>`), 안 그러면 db 파일이 잠김.

## 추가(11-07 저녁) — 단계 0 결과(더)·학습 틀 인터페이스 약속
- 더 생긴 단계 0 결과(서버 `results/s0/…`, 로컬 사본 `…\tmp\results_mirror\s0\…` — 0-4_spi·0-5·0-6_history·0-10 은 서버에만 있을 수 있음):
  - `0-4_spi/`: SPI_국내_관측_0-4.parquet(sgg_code·date·spi3·spi6 = 관측 칸), SPI_국내_예보_0-4.parquet(같은 칼럼 = 예보 칸, `windows` 의 forecast_cube), SPI_해외_0-4.parquet(sgg·alpha_3·date·spi3·spi6).
  - `0-6_static/`: 식품_고정표.parquet(p3·food_aw·food_ph·food_na·aw_label·…·품목 rg 4), 축사_L1.parquet(unit·L1·…). 조회 함수 = `lst/static_feats.py`(SO1 `SoilLookup`, X1 `eco_indicator`, X2 `gw_indicator`, season, `food_features`, `livestock_lookup`).
  - 기상 파생 = `lst/wxfeat.py`(`combined_cube`, `wx_derived` — WX3~WX6), 이력·인코딩·7-a = `lst/history.py`(`history_features`, `category_rates`, `te7a_grid`), SPI = `lst/spi.py`.
- 학습 틀 인터페이스(단계 0-6 입력 조립·0-7·0-9·단계 1~8 공통, 바꿔야 하면 지휘자에게 보고):
  - 행 표(meta): DataFrame, 행 = (시료, h) — lims_data_key·src_route·sido·sgg·p2·p3·d·h·T·E·S·g·R(+ y, test 쌍은 y 없음). train = 시료당 h 1개(`folds.make_train_rows`), 평가 쌍 = `folds.scoring_pairs`/`iter_scoring_pairs_by_h` 로 h 마다 만들어 바로 예측(디스크 캐시 금지, B-1).
  - 트리 입력 X: DataFrame — 숫자 칼럼 float32(NaN 그대로), 범주 칼럼 = 원 문자열. 알고리즘별 범주 변환(부록 A-12 '범주 처리')은 트리 모델 객체가 fit 때 train 값으로 정해 보관하고 predict 때 같게 씀.
  - 딥러닝 입력: dict(x_seq = float32 (N, W, C), x_static = float32 (N, S), x_cat = int64 (N, 4) — sido·sgg·p2·p3 번호). 표준화·결측 0·결측 표시 채널은 입력 조립 쪽 전처리기(그 모델 train 으로 fit, 객체 보관)가 만들고, 딥러닝 모델은 받은 배열만 씀.
  - 모델 객체: `fit(X, y, *, seed, n_jobs, eval_set=None) → self`, `predict_proba(X, n_iter=None) → float64 (N,)`(양성 확률), 부스팅은 `predict_proba(X, n_iter=k)` 로 56 시점 채점 가능, `save(path)`/`load(path)`.
  - 채점은 `lst/metrics.py` 만(trapz PR-AUC·구간 가중·threshold), 예측 저장 = parquet(lims_data_key·h·fold·pred), test 정답은 단계 보고 때만(§1-3 test 막기).
- 담당자 결정(11-07): AutoKeras 블록 대응 = 가까운 블록(`lst/tuning.py` AK_PROPOSAL — LSTM·GRU·LSTMA·GRUA·CMB = RNNBlock, TCN = ConvBlock, TRF·CMBV·⑦-D = 튜닝 안 함), 우리 기본 설정 모델은 20번째 후보로 남음. 화성시 만세·효행·병점구 면적 = 경기일보 2026-01-20 기사 값(0-6_static 설정에 출처 기록).
