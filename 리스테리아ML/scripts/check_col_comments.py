import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import psycopg
from dotenv import dotenv_values

# DB 테이블 코멘트·칼럼 코멘트(col_description)·자료형을 그대로 출력한다. 읽기 전용 세션, SELECT 만.
# 사용: py check_col_comments.py ba_data.t_aw_cl [스키마.테이블 ...]
ENV_PATH = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\.env")
SQL = """
select a.attnum, a.attname, format_type(a.atttypid, a.atttypmod), col_description(a.attrelid, a.attnum)
from pg_attribute a
where a.attrelid = %s::regclass and a.attnum > 0 and not a.attisdropped
order by a.attnum
"""

e = dotenv_values(ENV_PATH)
with psycopg.connect(host=e["DB_HOST"], port=int(e.get("DB_PORT") or 5432), dbname=e["DB_NAME"],
                     user=e["DB_USER"], password=e.get("DB_PASSWORD") or None,
                     options="-c default_transaction_read_only=on",
                     application_name="listeria_check_col_comments", connect_timeout=15) as conn:
    with conn.cursor() as cur:
        cur.execute("show default_transaction_read_only")
        assert cur.fetchone()[0] == "on", "읽기 전용 세션이 아님"
        for tbl in sys.argv[1:]:
            cur.execute("select obj_description(%s::regclass, 'pg_class')", (tbl,))
            print(f"\n== {tbl} | 테이블 코멘트: {cur.fetchone()[0]!r}")
            cur.execute(SQL, (tbl,))
            for n, name, typ, cmt in cur.fetchall():
                print(f"{n:>3} {name:<28} {typ:<28} {cmt!r}")
