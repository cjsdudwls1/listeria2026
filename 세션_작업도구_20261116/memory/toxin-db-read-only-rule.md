---
name: toxin-db-read-only-rule
description: "TOXIN_MODEL_PRED(PostgreSQL, [IP 가림]:5432) 접속 규칙 — 회사 공용 [권한 가림] 계정이라 읽기·다운로드만 허용, DB/계정/서버 설정 변경 금지, 접속 정보는 리스테리아ML/.env(09-28~, 값 출력 금지)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 01abded3-7c52-4a0a-b759-ccb02cdf2da3
  modified: 2026-10-14T05:48:59.771Z
---

DB `TOXIN_MODEL_PRED` (PostgreSQL 18.4, host [IP 가림], port 5432) 에 접속할 때는 **읽기와 다운로드만** 한다.
사용자가 2026-09-10 에 명시: "읽고 다운만 가능하게, db 수정은 불가. 설정은 따로 해놓지 말고 기억만 해놔"
(회사 공용 계정이라 다른 사람도 쓰므로 계정·서버 쪽 설정을 건드리면 안 됨).

**Why:** 계정이 [권한 가림] 권한(rolsuper=true)이라 어떤 쓰기도 실제로 실행되어 버린다. 공용 계정이므로
`ALTER ROLE ... SET`, 권한 변경, 영구 테이블·인덱스·뷰 생성, INSERT/UPDATE/DELETE 등은 다른 사용자에게 영향을 준다.

**How to apply:**
- (2026-09-28 갱신) 접속 정보는 `리스테리아ML/.env` 의 `DB_HOST`·`DB_PORT`·`DB_NAME`·`DB_USER`·`DB_PASSWORD`
  (사용자 지정, CLAUDE.md 반영). psycopg `options="-c default_transaction_read_only=on"` 로 연다(내 세션에만 적용).
  `.env` 는 아플라톡신 템플릿 복사본이라 키 이름에 `AFLA_` 가 붙은 것도 있음. 값은 절대 출력하지 않는다(키 이름·채워짐 여부만).
  옛 `scripts/db_utils.py` 는 작업 폴더에서 삭제됨. SELECT 만 실행한다.
- 서버 측에는 아무 객체도 만들지 않는다. 세션 전용 TEMP 테이블도 사용자 확인 없이는 쓰지 않는다
  (2026-09-10 시점 `collect_all.py` 는 TEMP body 테이블을 쓰도록 되어 있으므로, 실행 전에 CTE/서브쿼리 방식으로 바꿀 것).
- 세션 한정 `SET work_mem` 같은 설정도 꼭 필요할 때만, 영구 설정(ALTER SYSTEM/ALTER ROLE/ALTER DATABASE)은 절대 금지.
- 비밀번호는 `%APPDATA%\postgresql\pgpass.conf` (host:port:db:user:password 한 줄) 에서 libpq 가 읽는다.
  코드·로그·대화에 비밀번호를 출력하지 않는다. `테이블명세서(모델팀)/database계정.txt` 는 안전 필터가 읽기를 막는다.
- 큰 결과는 서버측 커서(트랜잭션 블록 필요)로 청크 fetch 해 `리스테리아ML/data/collected/` parquet 로 저장.
- 정의서에 없는 스키마도 존재: weather_data(interpolation_weather 등 QC 된 관측소 일자료, overseas_weather 138M행),
  predict_data, staging_*, train_data(wx160/wx60 아플라톡신 산출물). 사용은 사용자와 상의 후.
- DB 리스테리아 행 수는 765,891 (2026-09-10) 로 노트북 parquet(676,537) 보다 많다. 기준 선택은 사용자 몫.
  차이의 대부분은 주소 출처 '검사기관' 행(86,696)이며, 조인키·코드분류 등 확인 사실은 [[listeria-db-structure-findings]].
- (2026-10-14) 자동 모드 백그라운드 세션에서 pgpass 로 접속하는 SELECT 스크립트(`ba_data.t_prdlst_cl` 조회)가
  Claude Code 자동 모드 판정기에 이유 없이 거부됨. 같은 날 `c` 함수에 `--dangerously-skip-permissions` 를 넣는 프로필 수정도
  "Create Unsafe Agents" 로 거부 → 사용자가 직접 고치도록 안내. 거부되면 우회하지 말고 사용자에게 직접 실행(`! 명령`)을 제안.
  같은 날 전체 허용 모드 세션에서는 같은 SELECT(pgpass 읽기 전용 세션)가 정상 실행됨. py 는 `-I` 로 돌리면 pandas 가 dateutil 을 못 찾음.
관련: [[listeria-variable-mapping]], [[python-env-use-py-launcher]]
