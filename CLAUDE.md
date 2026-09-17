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

```
py 리스테리아ML/scripts/test_conn.py        # DB 접속·스키마 확인 (read-only 여부 출력)
py 리스테리아ML/scripts/run_checks.py       # sql/listeria_collect_queries.sql 의 1-x 검증 쿼리 실행
py 리스테리아ML/scripts/nb2_extract_db.py   # 2.ipynb 용 보조 데이터 → data/collected/nb2_db_*
```

스크립트 일부(`collect_all.py`, `run_checks.py`)는 `scripts/` 를 cwd 로 두고 `from db_utils import ...` 하거나 절대경로(`C:\Users\NT751\Desktop\천영진\...`)를 하드코딩하고 있다. worktree 에서 실행하면 출력이 메인 체크아웃 경로로 간다는 점에 주의.

## DB 규칙 (가장 중요)

데이터 원천은 PostgreSQL `TOXIN_MODEL_PRED` (host 10.31.72.23). **회사 공용 superuser 계정**이라 모든 쓰기가 실제로 실행된다.

- 접속은 항상 `scripts/db_utils.py` 의 `connect()` (기본 `readonly=True` → 세션 옵션 `default_transaction_read_only=on`). 비밀번호는 `%APPDATA%\postgresql\pgpass.conf` 에서 libpq 가 읽으며 코드·로그에 쓰지 않는다.
- SELECT·다운로드만 한다. 테이블/인덱스/뷰 생성, INSERT/UPDATE/DELETE, `ALTER ROLE/SYSTEM/DATABASE` 금지. 세션 TEMP 테이블도 사용자 확인 없이는 쓰지 않는다.
- `collect_all.py` 는 `connect(readonly=False)` 로 TEMP `body` 테이블과 `SET work_mem` 을 쓰는 옛 방식이다. 규칙 위반이므로 그대로 실행하지 말고 CTE/서브쿼리 방식으로 바꾼 뒤 사용한다(`nb2_extract_db.py` 가 규칙을 따르는 예).
- 큰 결과는 서버측 named cursor(트랜잭션 블록 안)로 청크 fetch 후 Decimal/date 를 변환해 `리스테리아ML/data/collected/` 에 parquet 로 저장하는 패턴을 따른다.

## 데이터 원칙

- **원자료 그대로 수집한다.** 결측 채움(NULL→0 등), 단위 변환, 범위 필터, QC, 집계는 사용자 승인 후에만 한다. 원형 테이블은 `data/collected/raw/raw_<table>.parquet` (SELECT *) 로 따로 둔다.
- 사용자에게 보여줄 분석·질문 문서에서는 추정을 사실처럼 단정하지 않는다. 확인한 사실(DB·조인 → 작업 → 결과)과 의문을 구분해 질문형으로 쓴다.
- 누수 주의: 품목제조대장·수분활성도 조인 값은 검출 행에서 결측률이 달라진다. 수거·의뢰·검사 이후 날짜는 예측 시점 이후 정보일 수 있다.

## 최종 보고서를 위한 중간 정리

최종 산출물은 모델링까지 끝낸 뒤 전체 과정과 결과를 담은 **리스테리아 개별 보고서**다(목차: `docs/00_보고서/보고서 목차.md` — 데이터 구축 → EDA → 예측변수 → 전처리 → 모델 → 검증·XAI → 부록 코드). 끝에 한꺼번에 되짚지 않도록 작업할 때마다 정리한다.

- 의미 있는 작업 단위(수집·조인, EDA 한 주제, 전처리 결정, 변수 생성, 학습·검증 실험)를 마치면 `C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\00_보고서\작업기록.md` 에 날짜별로 추가한다(없으면 만든다). 항목마다 **보고서 목차의 해당 절**, 한 일, 사용한 데이터·테이블·조인 키·필터 조건, 결과 수치(행 수·비율·성능), 산출물 경로(노트북 셀/스크립트/parquet/그림), 남은 의문을 적는다. 아주 소상히 적는다
- **결정은 근거와 함께 남긴다.** 무엇을 왜 넣고 뺐는지, 누가 승인했는지(사용자·전문가·모델팀 답변), 검토했지만 버린 선택지까지 적는다. 전처리·대상 범위 결정은 사용자 승인 사실을 기록한다.
- 보고서에 들어갈 표·그림은 노트북 출력에만 두지 말고 파일로도 저장하고, 수치를 만든 코드 위치를 함께 적어 재현할 수 있게 한다. 모델 실험은 데이터 버전·분할 방식·변수 목록·하이퍼파라미터·지표를 실험마다 남긴다.
- 문구는 처음 보는 사람(교수님·식정원 관계자)이 바로 이해하도록 쓴다: 내부 약어·행번호("출처 r15") 대신 풀어 쓰고, "부분·대체" 같은 모호어 대신 뜻이 보이는 말을 쓰고, 수치마다 산출 기준(0 포함 여부 등)과 판단 근거를, 결측은 컬럼별로 비어 있는 이유를 적는다. 추정은 "추정 — 확인 필요"로 표시한다.
- 기존 정리 문서(`docs/02_데이터수집/리스테리아_위해인자_DB수집_결과보고.xlsx`, `docs/03_질문리스트/`, 2.ipynb 결론 절)와 내용이 겹치면 새로 쓰지 말고 위치를 가리킨다.

## 데이터 구조 (여러 파일을 읽어야 알 수 있는 부분)

- **몸통**: `ready_data.hazard_preprocessed_data WHERE hazard_label = '리스테리아'`. 행 키는 `lims_data_key`, 타깃은 `judge_result`(양성 약 0.29%). `data/listeria_body.parquet`(676,537행, domestic 628,895 / overseas 47,642)는 DB 몸통(약 765,891행)에서 주소 출처 '검사기관' 행이 빠진 것으로 보인다 — 기준 선택은 사용자 몫.
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
