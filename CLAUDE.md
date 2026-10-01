# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 저장소 성격

식품 위해요소 예측모델 과제 중 **리스테리아 검출(judge_result) 예측**을 맡은 개인 작업 폴더다. 빌드·린트·테스트 체계는 없고, Jupyter 노트북 + DB 추출용 Python 스크립트 + 참고 문서(xlsx/hwp/pptx)로 구성된다. 사용자는 한국어로 소통하며 산출물(노트북 설명, 보고서, 질문 문서)도 한국어로 작성한다.

- `리스테리아ML/` — 실제 작업 공간 (아래 설명)
- `리서치팀문서/` — 위해요소별 중요인자·데이터출처 xlsx, 예측인자 PPT 캡처, 수분활성도·pH·염도 통합 파일. `원본백업/` 은 수정 전 원본이므로 건드리지 않는다.
- `테이블명세서(모델팀)/` — DB 테이블목록·정의서. `database계정.txt` 는 .gitignore 대상(커밋 금지).
- `개별보고서_양식/`, `센터_발표자료_사업보고서/` — 보고서 양식·참고 자료 (hwp/pptx, 코드와 무관)
- `가이드.txt` — 과제 안내. **접속 계정 정보가 들어 있으므로 내용을 다른 파일·출력·커밋 메시지에 옮기지 않는다.**

## 실행 환경

- Python 은 `py` 런처(Python 3.14, pandas·openpyxl·psycopg 설치)로 실행한다. PATH 의 `python`(anaconda)은 셸 도구에서 DLL 오류로 죽는다.
- 한글이 들어간 코드는 `py -c` 한 줄 대신 UTF-8 .py 파일로 저장해 실행한다. 스크립트는 `sys.stdout.reconfigure(encoding="utf-8")` 로 시작하는 관례를 따른다.
- 사용자는 노트북을 Jupyter(ipykernel)에서 직접 실행·편집한다. 노트북을 덮어쓰기 전에 사용자가 고친 셀이 있는지 커밋본과 비교한다.
- **실험 2(`EXP/2`)는 별도 Python 3.12 가상환경 `리스테리아ML/.venv-exp2`** 로 돌린다(AutoGluon 이 Python 3.14 를 지원하지 않음. `py -m uv python install 3.12` + `uv venv` 로 만듦, 2026-09-29). autogluon.tabular 1.6.3 · torch 2.13+cu126 · optuna 5 · mlflow 3.16 · lightgbm 4.7 · catboost 1.2.10 · xgboost 3.3(GPU 빌드, AutoGluon 기본 xgboost-cpu 를 교체) · pandas 2.3.3. Jupyter 커널 이름 `listeria-exp2`. 실행은 `.venv-exp2\Scripts\python.exe`(nbclient 로 노트북 실행: `EXP/2/run_all.py`). 이 PC 에서 **Ray 가 시작되지 않아**(raylet 시작 시간 초과) AutoGluon 내장 HPO(`hyperparameter_tune_kwargs`)는 못 쓴다 — Optuna TPE 가 설정을 제안하고 AutoGluon `fit_extra` 로 하나씩 학습해 대신한다.

```
py 리스테리아ML/scripts/test_conn.py        # DB 접속·스키마 확인 (read-only 여부 출력)
py 리스테리아ML/scripts/run_checks.py       # sql/listeria_collect_queries.sql 의 1-x 검증 쿼리 실행
py 리스테리아ML/scripts/nb2_extract_db.py   # 2.ipynb 용 보조 데이터 → data/collected/nb2_db_*
py 리스테리아ML/scripts/check_col_comments.py ba_data.t_aw_cl   # 테이블·칼럼 DB 코멘트·자료형 출력 (읽기 전용, 테이블 여러 개 가능)
py 리스테리아ML/scripts/find_in_table_spec.py t_aw_cl           # 테이블정의서 xlsx 2개에서 낱말이 든 행 출력
```

스크립트 일부(`collect_all.py`, `run_checks.py`)는 `scripts/` 를 cwd 로 두고 `from db_utils import ...` 하거나 절대경로(`C:\Users\NT751\Desktop\천영진\...`)를 하드코딩하고 있다.

## DB 규칙 (가장 중요)

데이터 원천은 PostgreSQL `TOXIN_MODEL_PRED` (host 10.31.72.23). **회사 공용 superuser 계정**이라 모든 쓰기가 실제로 실행된다.

- 접속 정보는 `리스테리아ML/.env` 의 `DB_HOST`·`DB_PORT`·`DB_NAME`·`DB_USER`·`DB_PASSWORD` 키에서 읽는다. 값은 코드·로그·출력·커밋 메시지에 쓰지 않고, `.env` 는 커밋하지 않는다(.gitignore 대상). 접속할 때는 항상 세션 옵션 `default_transaction_read_only=on` 을 켠다(psycopg `options="-c default_transaction_read_only=on"`). 옛 방식(`scripts/db_utils.py` 의 `connect()` + `%APPDATA%\postgresql\pgpass.conf`)의 `db_utils.py` 는 2026-09-28 현재 작업 폴더에서 삭제된 상태다.
- SELECT·다운로드만 한다. 테이블/인덱스/뷰 생성, INSERT/UPDATE/DELETE, `ALTER ROLE/SYSTEM/DATABASE` 금지. 세션 TEMP 테이블도 사용자 확인 없이는 쓰지 않는다.
- `collect_all.py` 는 `connect(readonly=False)` 로 TEMP `body` 테이블과 `SET work_mem` 을 쓰는 옛 방식이다. 규칙 위반이므로 그대로 실행하지 말고 CTE/서브쿼리 방식으로 바꾼 뒤 사용한다(`nb2_extract_db.py` 가 규칙을 따르는 예).
- 큰 결과는 서버측 named cursor(트랜잭션 블록 안)로 청크 fetch 후 Decimal/date 를 변환해 `리스테리아ML/data/collected/` 에 parquet 로 저장하는 패턴을 따른다.

## 데이터 원칙

- `리스테리아ML/data/` 에 데이터 파일을 추가·재생성·삭제하면 `리스테리아ML/data/README.md`(파일별 생성 시점·방법)도 함께 최신화한다.
- **원자료 그대로 수집한다.** 결측 채움(NULL→0 등), 단위 변환, 범위 필터, QC, 집계는 사용자 승인 후에만 한다. 원형 테이블은 `data/collected/raw/raw_<table>.parquet` (SELECT *) 로 따로 둔다.
- 사용자에게 보여줄 분석·질문 문서에서는 추정을 사실처럼 단정하지 않는다. 확인한 사실(DB·조인 → 작업 → 결과)과 의문을 구분해 질문형으로 쓴다.
- **추정하지 않는다.** 컬럼·코드의 뜻, 결측 이유, 수치의 출처는 테이블정의서·DB 코멘트(`col_description`)·실제 값으로 확인한 것만 쓴다. 확인하지 못했으면 컬럼명을 그대로 적고 "정의서 설명 없음 — 확인 필요"라고 표시한 뒤 사용자에게 묻는다. 이름을 보고 한글 이름을 지어 붙이지 않는다. (2026-09-22 사용자 지적: `result_value_num`·`judge_stdr`·`judge_stdr_num` 을 "검사 결과값·판정기준"으로 부른 것)
- **컬럼 뜻·분류는 말하기 전에 직접 확인한다.** 컬럼이 무엇인지, 어떤 종류의 데이터인지(식품·기상·토양 등), 단위가 무엇인지 답하거나 문서·메일 문구에 쓰기 전에 (1) DB 코멘트와 (2) 테이블정의서 두 파일(`테이블명세서(모델팀)/테이블정의서.xlsx`, `리스테리아ML/docs/참고자료/20260901.테이블정의서 (1).xlsx`)을 **그 자리에서 조회**하고, 답에는 어디서 확인했는지(DB 코멘트 원문·정의서 시트와 문구)를 함께 적는다. 테이블 구조·칼럼 이름·노트북 설명·메모리·작업기록의 옛 기록만 보고 말하지 않는다 — 그것밖에 없으면 "구조로 본 추정 — DB 코멘트·정의서 미확인"이라고 밝힌다. 코멘트·정의서에 없는 것(단위 등)은 "설명 없음 — 확인 필요"로 둔다. 조회 스크립트: `scripts/check_col_comments.py`, `scripts/find_in_table_spec.py`. (2026-10-01 사용자 지적: `ba_data.t_aw_cl` 의 `aw`·`ph`·`na` 를 DB 코멘트 확인 없이 테이블 구조만 보고 "식품 데이터"라고 답한 것)
- **배경지식으로 표현을 붙이지 않는다.** DB 코멘트·테이블정의서·실제 값·이 폴더의 문서에 없는 말(자료의 성격, 만든 방법, 일반적인 구조 설명 등)을 Claude 의 배경지식으로 덧붙여 설명하지 않는다. 출처에 적힌 문구를 그대로 쓰고(예: DB 코멘트 "Open-Meteo Archive(ERA5) 해외 제조업소 좌표 일별 기상"), 그 이상은 "설명 없음 — 확인 필요"로 둔다. "외국에서 만든 제품"처럼 값에서 넘겨짚은 해석도 쓰지 않고 확인된 사실("`import_nation_code` 가 KR 이 아님")만 적는다. (2026-10-01 사용자 지적: `api_data_info.global_factory_weather` 값을 DB 에 없는 표현 "재분석 격자 값"으로 부른 것, 법정동 코드 10자리 구조를 기억으로 설명한 것)
- **DB 객체는 정확한 이름으로 부른다.** 대화·문서·작업기록에서 "몸통", "표", "계층표", "코드표" 같은 별칭을 쓰지 않는다. 항상 스키마.테이블(`ready_data.hazard_preprocessed_data`, `ba_data.t_prdlst_cl`, `staging_data.prdlst_hierarchy`)·컬럼 이름·파일 경로(`리스테리아ML/data/listeria_body.parquet`)로 말한다. 첫 언급에는 이름 뒤에 한 줄 뜻을 붙이고(예: `ready_data.hazard_preprocessed_data`(위해요소 통합 전처리 결과)), 이후에도 이름을 반복한다. 단계·레벨·코드를 말할 때는 어느 테이블의 어느 컬럼 값인지 함께 적는다. (2026-09-22 사용자 지적)
- 누수 주의: 품목제조대장·수분활성도 조인 값은 검출 행에서 결측률이 달라진다. 수거·의뢰·검사 이후 날짜는 예측 시점 이후 정보일 수 있다.

## 최종 보고서를 위한 중간 정리

최종 산출물은 모델링까지 끝낸 뒤 전체 과정과 결과를 담은 **리스테리아 개별 보고서**다(목차: `docs/00_보고서/보고서 목차.md` — 데이터 구축 → EDA → 예측변수 → 전처리 → 모델 → 검증·XAI → 부록 코드). 끝에 한꺼번에 되짚지 않도록 작업할 때마다 정리한다.

- 의미 있는 작업 단위(수집·조인, EDA 한 주제, 전처리 결정, 변수 생성, 학습·검증 실험)를 마치면 `C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\00_보고서\작업기록.md` 에 날짜별로 추가한다(없으면 만든다). 항목마다 **보고서 목차의 해당 절**, 한 일, 사용한 데이터·테이블·조인 키·필터 조건, 결과 수치(행 수·비율·성능), 산출물 경로(노트북 셀/스크립트/parquet/그림), 남은 의문을 적는다. 아주 소상히 적는다
- **결정은 근거와 함께 남긴다.** 무엇을 왜 넣고 뺐는지, 누가 승인했는지(사용자·전문가·모델팀 답변), 검토했지만 버린 선택지까지 적는다. 전처리·대상 범위 결정은 사용자 승인 사실을 기록한다.
- 보고서에 들어갈 표·그림은 노트북 출력에만 두지 말고 파일로도 저장하고, 수치를 만든 코드 위치를 함께 적어 재현할 수 있게 한다. 모델 실험은 데이터 버전·분할 방식·변수 목록·하이퍼파라미터·지표를 실험마다 남긴다.
- 문구는 처음 보는 사람(교수님·식정원 관계자)이 바로 이해하도록 쓴다: 내부 약어·행번호("출처 r15") 대신 풀어 쓰고, "부분·대체" 같은 모호어 대신 뜻이 보이는 말을 쓰고, 수치마다 산출 기준(0 포함 여부 등)과 판단 근거를, 결측은 컬럼별로 비어 있는 이유를 적는다. 추정은 "추정 — 확인 필요"로 표시한다.
- 기존 정리 문서(`docs/02_데이터수집/리스테리아_위해인자_DB수집_결과보고.xlsx`, `docs/03_질문리스트/`, 2.ipynb 결론 절)와 내용이 겹치면 새로 쓰지 말고 위치를 가리킨다.

## 데이터 구조 (여러 파일을 읽어야 알 수 있는 부분)

- **기준 테이블 `ready_data.hazard_preprocessed_data`** (`WHERE hazard_label = '리스테리아'`, 약 765,891행). 행 키는 `lims_data_key`, 타깃은 `judge_result`(양성 약 0.29%). 파일 `data/listeria_body.parquet`(676,537행, domestic 628,895 / overseas 47,642)는 이 테이블에서 주소 출처 '검사기관' 행이 빠진 것으로 보인다 — 기준 선택은 사용자 몫. 이 파일의 `train_prdlst_code` 는 2026-09-08 추출 시점 값이라 현재 DB 값과 9,236행 다르다(옛 C01/C02 코드 행 재배정, 작업기록 2026-09-22).
- **기준일**: `mnfctur_de_or_fallback` (제조일이 없으면 다른 날짜로 대체, 출처는 `date_basis`). `geo_route` 로 국내/해외를 나눈다.
- **조인 대상** (모두 1행/lims_data_key 가 되도록 `DISTINCT ON` 으로 중복 제거):
  - `ba_data.m_org_lims_data_info` — 식약처 LIMS 원본 (`lims_data_key`)
  - `ba_data.s_org_lims_data_info` — 수입식품 LIMS (`rcept_no` ↔ `cntc_sys_key2` 또는 `lims_data_key` 두 번째 토큰; 매칭 수로 선택)
  - `ba_data.t_prdlst_mnf_ledg` — 품목제조대장 (`prdlst_mnfctur_sttemnt_no` ↔ `prdlst_report_no`/`prdlst_report_ledg_no`)
  - `ba_data.t_bsn_lcns_ledg`, `ba_data.t_prdlst_cl`(식품공전 분류), `ba_data.t_cm_cmmn_code`(공통코드 — 실제 분류코드는 `IM02/IM08/IM11/IM40` 등 `IM*`; 매핑 엑셀·`collect_all.py` 의 `ZIM*` 는 틀림)
  - 기상·환경 `api_data_info.*`: ASOS/NAS 일자료(국내 좌표 최근접 관측소, haversine), `soil_ph_info`, 에어코리아, 해외는 `global_factory_weather`(해외 제조소 좌표×일) 또는 국가 단위 `global_weather_info`
- 국내 기상은 `geo_route='domestic'` 에만 붙인다. 원천 단위 함정: ASOS `sum_rn` NULL 이 무강수라고 단정 불가, NAS 일조는 분, 해외 일조는 초.
- `hazard_preprocessed_data` 의 `train_/statistic_/mapped_` 품목 컬럼은 가공 컬럼이다. 원자료 품목은 m_org/s_org/t_prdlst_cl 쪽에 있다.
- 테이블정의서의 노란색 표시는 타깃(`judge_result`)만 뜻한다.

## 노트북·문서 흐름

- `notebook/1.ipynb` — 사용자의 초기 탐색 (parquet 로드, 컬럼 이해, 타깃 분포)
- `notebook/2.ipynb` — 비즈니스·데이터 이해 점검 (검사목적·기관·날짜·주소·도메인 대조, 전문가/모델팀 질문 목록). 보조 데이터는 `nb2_extract_db.py` 산출물.
- `notebook/3_DB수집_조인.ipynb` — 매핑 엑셀 기준 DB 수집 → 커버리지·누수 점검 → `data/collected/listeria_joined.parquet`. 재수집은 노트북의 `FORCE=True` 또는 해당 parquet 삭제.
- `notebook/4_기상_가공식품_타겟범위.ipynb` — 기상 결합 대상·가공식품 제거·타깃 범위 판단용 집계와 질문(결정은 하지 않음)
- `sql/listeria_collect_queries.sql` — DataGrip 용 섹션별 쿼리. 헤더 `-- 1-1.` 형식을 `db_utils.split_sql_sections()` 가 파싱하므로 형식을 유지한다. 실행 순서 1(검증) → 0(몸통) → 2 → 3 → … → 7.
- `docs/` 는 주제별 폴더로 나눈다: `00_보고서`(보고서 목차·작업기록), `01_메모`(사용자 메모 — info.md·질문.md), `02_데이터수집`(매핑 엑셀·DB수집 결과보고), `03_질문리스트`(최신판만 바로 아래, 옛 버전은 `이전버전/`), `참고자료`(받은 자료 — 20260901 테이블정의서). 새 문서도 맞는 폴더에 넣고 docs 바로 아래에는 두지 않는다. 파일을 옮기면 노트북·스크립트·SQL 의 경로 참조도 함께 고친다.
- `docs/02_데이터수집/리스테리아_수집변수_매핑.xlsx` — 수집 변수 기준표(알려진 오류 있음: 시트3 수집방안 컬럼 없음, 순위 번호 어긋남 등). `docs/01_메모/info.md` 는 사용자의 리스테리아 도메인 메모, `docs/00_보고서/보고서 목차.md` 는 최종 개별보고서 목차.
- `리서치팀문서/위해요소별_중요인자_데이터출처.xlsx` 의 D열 **빨간 글자**가 수집 대상 인자다. 셀 단위가 아니라 openpyxl rich_text run 단위로 파싱해야 한다.

## Git·작업 위치

- 원격은 GitHub 비공개 저장소 `origin` = `cjsdudwls1/listeria2026` 이다(2026-09-17 의 로컬 전용·원격 삭제 결정은 2026-09-22 사용자 결정으로 해제, 원격 복구). push 는 사용자가 요청할 때만 하고, 그 전에 `git status` 로 `가이드.txt`·`database계정.txt`·`리스테리아ML/.env` 같은 계정 정보 파일이 스테이징되지 않았는지 확인한다.
- **worktree 를 쓰지 않는다.** 백그라운드 작업도 이 폴더에서 바로 작업한다(`.claude/settings.json` 의 `worktree.bgIsolation: none`). EnterWorktree·`git worktree add` 로 작업 폴더를 새로 만들지 않는다.
- 격리가 없으므로 여러 작업이 같은 파일(특히 사용자가 여는 노트북)을 동시에 고치지 않게 하고, 커밋할 때는 사용자 변경이 섞이지 않았는지 `git status` 로 확인한다.
- `리스테리아ML/data/collected/` 와 `공유용/` 아래 parquet 는 git 추적에서 제외한다(DB 원자료·대용량). 파일은 이 폴더에만 둔다.

## 데이터 조인 기준
- 'C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet'의 'mnfctur_de','lat','lon'

## ML 실험에서 숙지할 것

**분할과 누수**
- 기준일로 정렬한 뒤 시간순 분할. 랜덤 분할은 금지
- 최근 1년 홀드아웃은 실험 끝날 때까지 건드리지 않기 (test 기간 - 2025.07.01 ~ 2026.06.30 (고정))
- TimeSeriesSplit 시 확장 윈도우, fold마다 양성 수 확인. 양성 0인 fold가 나오면 지표가 정의되지 않음
- 기상 lookback이 기준일 이후를 포함하지 않게. D0을 넣을지 D-1부터 할지 먼저 정하고 고정

**결측과 불균형**
- NaN 유지하고 XGBoost·랜덤포레스트. 로지스틱은 못 돌아감
- 양성 비율이 낮으니 accuracy는 무의미. PR 곡선 기준
- 임계값 0~1 스윕해서 precision·recall·F1 곡선을 모델별로 그리기. 어떤 지표를 기준으로 고를지 먼저 정해야 함. 식품안전은 보통 놓치는 비용이 커서 recall 우선

**베이스라인**
- 품목만으로 예측, 계절(월)만으로 예측, 기상만으로 예측 세 개를 먼저 돌려. 이걸 못 이기면 모델을 쓸 이유가 없음
- 비교는 같은 분할, 같은 지표로
- 구현(2026-09-29): 품목만 `EXP/1/5_품목만.ipynb`, 계절(월)만 `EXP/1/4_월만.ipynb`. ML 없이 **train 기간 값별 검출 비율(검출 수 ÷ 행 수)을 예측 확률로 쓴다**(입력이 칼럼 하나라 어떤 모델이든 이 표를 배우는 것과 같음). 분할·임계값 규칙(0.5·val F1최대·val F2최대)·지표·MLflow 기록 구조는 `EXP/1/2_바로학습.ipynb` 와 같다(MLflow 실험 `리스테리아_EXP1`, run `4_월만`·`5_품목만`).
  - 월 = `mnfctur_de` 의 월(1~12, 범주로 다룸). `mnfctur_de` 는 행마다 제조일·의뢰일·선적일 중 하나(`date_basis`)라 월의 뜻도 행마다 다르다.
  - 품목 = `train_prdlst_code`(52종). 결측 170행은 `(결측)` 한 그룹, train 에 없는 값은 train 전체 검출 비율. 원본 `prdlst_code`(1,251종)는 test 에 train 에 없는 코드가 66종 473행이라 쓰지 않음.
  - 기상만 = `EXP/1/6_기상만.ipynb`(2026-09-29, run `6_기상만`). 입력이 연속값 1,700칼럼(`3_기상조인_후_학습` 의 기상·대기 17변수 × D-1~D-100, `data/collected/exp1_wx/`)이라 비율표 대신 `2_바로학습` 과 같은 모델 3개 기본값(CatBoost GPU). test roc_auc 0.48~0.54 로 무작위 수준(2026-10-01 재실행).

**2026-10-01 입력 변경 (실험 1·2 공통, 사용자 결정)**
- `data/X.parquet` 은 11열이다: 검사기관 코드 `inspct_instt_code` 를 뺐다(예측 시점에 어느 검사기관으로 갈지 알 수 없음 — 교수님 지적). 실험 1 의 LIMS 피처는 9개(`src_route`·`prdlst_code`·`train_prdlst_code`·`train_prdlst_nm`·`date_basis`·`addr_source`·`legal_dong_code`·`lat`·`lon`). 검사기관 코드를 피처로 다시 넣지 않는다.
- `X.parquet` 의 `src_route` 는 로마자로 적힌 한국 주소 78행을 domestic 으로 옮긴 값이다(`EXP/1/1_데이터정제.ipynb` — `addr` 낱말과 좌표 두 방법이 같은 행을 가리킬 때만 옮김). 원자료 `listeria_body.parquet` 의 `src_route` 는 그대로(DB `geo_route` 와 같은 값, 한글 주소 여부). 기상은 `X.parquet` 의 값 기준으로 붙는다.
- 수입 검사 행 여부는 `src_route` 가 아니라 `data/listeria_df_joined_full.parquet` 의 `reqest_sys_nm`(의뢰 시스템 코드 이름)·`import_lims_linked`(수입식품 LIMS 연결 여부)로 본다(노트북 6-1 의 7-1절). `src_route` 와 221행 어긋난다.
- 식품 값(`aw`·`ph`·`na`)이 원문에서 어떤 표기였는지는 `aw_label`·`ph_label`·`na_label`(단일값·중앙·이상·이하·초과·미만)에 있고 범주형 피처로 넣는다(`EXP/1/3_기상조인_후_학습.ipynb` 5절).
- overseas 행에서 100일 전부 결측인 기상·대기 변수는 12개(풍속 2·일조·지중온도 5·토양수분·대기 3)다.
- 2026-10-01 이전 MLflow 기록(검사기관 포함)은 `리스테리아ML/mlflow_backup_20261001/`(mlflow.db·mlruns)로 옮겼다. 지금 `리스테리아ML/mlflow.db` 에는 2026-10-01 재실행 기록만 있다.

**실험 2 (`EXP/2`, MLflow 실험 `리스테리아_EXP2`, 2026-09-29 사용자 지시)**
- 입력 `data/exp2/`(`1_데이터준비.ipynb`): 다운사이징(**전체 기간 검출 0 인 `train_prdlst_nm` 38종·76,497행 삭제** — 2026-10-01 사용자 지시로 "train 기간만 보고 정한 37종·76,462행"에서 바꿈, 차이는 train 에 행이 없는 `환경검체` 35행) + `lat`·`lon`·`legal_dong_code`·`train_prdlst_nm` 제거 + 기상 하루값 → 창 집계 353칼럼 = 364피처(범주형 8 = LIMS 5 + 식품 값 표기 3, 식품 3, 집계 353 — 2026-10-01 에 검사기관 코드 제거·값 표기 추가로 362 → 364), 600,040행(2026-10-01 18:06 재생성, 이전 600,075행). 노트북 6(`notebook/6_교수님과 미팅 후 품목 코드까지 조인.ipynb`)은 2026-10-01 부터 검출 0% 품목을 지우지 않는다(`data/listeria_df_joined.parquet` 676,537행 × 41열). 딥러닝용 시계열 `seq_std.npy`(17변수 × 100일, train 통계 표준화, 시간 순서 D-100→D-1).
- `2_autogluon.ipynb`: A단계 5가족(LightGBM·CatBoost·XGBoost·RF·XT — 신경망은 3_딥러닝으로만)을 **Optuna TPE 로 탐색**(Optuna `ask` → AutoGluon `fit_extra` 로 모델 1개 학습 → val AP 를 `tell`; 가족별 시간 예산, storage `output/optuna_exp2_ag.db`) → 가족별 최적 5개 + 가중 앙상블(가중치는 val). **배깅·스태킹 없음** — OOF 분할이 시간 순서를 어긴다는 이유로 사용자가 뺌(2026-09-29). 넣어야 한다면 `groups`(기준일 순 시간 블록) + `use_bag_holdout=True` + `ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}`(병렬 fold 는 Ray 필요)로 되고 tuning_data 에 groups 칼럼은 넣지 않는다; `validation_structure(temporal_forward_only)` 는 스태킹과 같이 못 씀. CatBoost·XGBoost GPU(xgboost-cpu → xgboost GPU 빌드로 교체), 나머지 CPU. AutoGluon 은 `memory_limit − 노트북 프로세스 사용량`을 가용 메모리로 보고 모자라 보이면 모델을 **조용히 건너뛴다**(RF·XT·XGBoost 가 빠졌던 적 있음, 2026-09-29) → `memory_limit=14` + 모델별 `ag.max_memory_usage_ratio`(RF·XT 3.0, 나머지 1.5). 학습 로그 `predictor_log.txt` 에서 "Skipping this model" 을 확인한다. RF·XT 도 Optuna 로 튜닝(AutoGluon 내장은 RF·XT 탐색 없음). 탐색 범위는 논문에 묶지 않음(2026-09-29 사용자 지시).
- `3_딥러닝.ipynb`: 논문(`docs/참고자료/0808Manuscript.2.docx` Table 2) Conv1D(1~3층)–BiLSTM(64/32)–additive attention + 표 형식 가지(범주형 임베딩 8 + 수치 359) → PyTorch. 층 순서 Conv/Dense→BN→ReLU→Dropout 로 고정. `DL_논문설정`(논문 값) + `DL_튜닝`(Optuna 최대 25회/6h, MedianPruner, storage `output/optuna_exp2_dl.db`). 양성 ×3 증강·가중 BCE 는 논문대로(탐색에서는 선택지).
- 각 노트북 설정 셀의 `SMOKE_TEST`(작은 표본 점검, MLflow 기록 안 함)·`LOG_TO_MLFLOW` 스위치. 본 실행은 `.venv-exp2\Scripts\python.exe EXP\2\run_all.py`(순서 실행, 로그 `output/exp2/run_all_*.log`).

**피처**
- 정적 피처(품목 코드, 카테고리)와 시계열 피처는 딥러닝에서 입력을 분리

**운영**
- seed 고정하고 실험 로그(데이터 구성, 피처 세트, fold별 지표) 남기기. 나중에 보고서 쓸 때 이게 근거가 됨.
- 실험 기록은 **MLflow** 로 남긴다. run 마다 입력 데이터 파일·수정 시각, 분할 기간·건수, 피처 목록, seed·장치, 하이퍼파라미터 전체, 임계값 규칙과 값, 지표(`EXP/1/readme.md` 목록), 그림을 기록한다. tracking URI 는 `리스테리아ML/.env` 의 `AFLA_MLFLOW_URI`(아플라톡신 템플릿에서 온 키 이름)에서 읽는다. 이 값이 상대경로(`sqlite:///mlflow.db`)면 노트북을 실행한 폴더마다 DB 가 따로 생기니 주의한다(EXP/1 노트북들은 `.env` 가 있는 `리스테리아ML/` 기준으로 고정 → `리스테리아ML/mlflow.db`, 그림·파일 `리스테리아ML/mlruns/`, 둘 다 .gitignore). 결과는 파일로 따로 저장하지 않고 MLflow 에만 남긴다(2026-09-29 사용자 결정). 기록 구조: 부모 run = 노트북 이름, 자식 run 이름 = `노트북 / 모델 / 임계값 규칙`(예: `2_바로학습 / CatBoost / F2최대` — 섞어 정렬해도 어느 실행인지 보이게), 부모·자식 모두 param `input` 에 입력 데이터 한 줄 설명(예: `LIMS 10칼럼`, `월 (mnfctur_de 의 월)`). 보는 법: `py -m mlflow ui --backend-store-uri "sqlite:///C:/Users/NT751/Desktop/천영진/리스테리아ML/mlflow.db"` → http://127.0.0.1:5000 → 왼쪽 위 **Model training**(GenAI 화면에는 학습 run 이 안 보임). C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\00_보고서\작업기록.md에는 mlflow의 어느 실험은 이거다 라는 식으로 간결히 작성.


### ML/DL 시 공통 사항

1. 검출/불검출 여부 확인 (전체 데이터/검출(개수,%))
    1. 불균형 문제 어떻게 해결했는지
2. 위해요소 하나당, base 모델 여러개 돌리고 성능 기록
    
    ex) ML(XGBoost, CatBoost, LightGBM), DL(LSTM)
    
3. 과거 기상 기후는 최대 100일까지 사용 가능(window = 100)
4. 데이터 분할 
   - train ~2024-06-30 / val 2024-07-01~2025-06-30 / test 2025-07-01~2026-06-30

** 기상 데이터 조인 시**
1. 주소지와 좌표 둘 다 있는데 기상 관측 지점과 좌표가 안 맞음
2. 가까운 관측소 3개(부족하면 5개) 평균값 사용. 한국은 좁아서 지구 곡률 무시하고 직선 거리로 충분.
