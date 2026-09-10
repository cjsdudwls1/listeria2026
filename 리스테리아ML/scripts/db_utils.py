# -*- coding: utf-8 -*-
"""TOXIN_MODEL_PRED(PostgreSQL) 접속 유틸.
비밀번호는 %APPDATA%\\postgresql\\pgpass.conf 에서 libpq 가 직접 읽는다 (코드/로그에 노출되지 않음).
"""
import os
import sys
import time
import psycopg
import pandas as pd

PGPASS = os.path.join(os.environ.get("APPDATA", ""), "postgresql", "pgpass.conf")
DEFAULT_HOST = "10.31.72.23"


def pgpass_entry(host: str = DEFAULT_HOST) -> dict:
    """pgpass.conf 에서 host 에 해당하는 항목의 host/port/dbname/user 만 돌려준다 (password 는 읽지 않음)."""
    with open(PGPASS, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\\:", "\x00").split(":")
            if len(parts) < 5:
                continue
            h, p, d, u = (x.replace("\x00", ":") for x in parts[:4])
            if h in (host, "*"):
                return {"host": host if h == "*" else h,
                        "port": 5432 if p == "*" else int(p),
                        "dbname": d, "user": u}
    raise RuntimeError(f"pgpass.conf 에 {host} 항목이 없습니다: {PGPASS}")


def connect(readonly: bool = True, host: str = DEFAULT_HOST, dbname: str | None = None) -> psycopg.Connection:
    e = pgpass_entry(host)
    opts = "-c default_transaction_read_only=on" if readonly else ""
    conn = psycopg.connect(host=e["host"], port=e["port"], dbname=dbname or e["dbname"], user=e["user"],
                           options=opts, application_name="listeria_collect", connect_timeout=15)
    conn.autocommit = True
    return conn


def read_sql(sql: str, conn=None, params=None, label: str = "") -> pd.DataFrame:
    """SELECT 결과를 DataFrame 으로. 큰 결과는 서버측 커서로 청크 fetch."""
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    try:
        with conn.cursor(name=f"c_{int(t0)}") as cur:  # server-side cursor
            cur.itersize = 50_000
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            chunks, n = [], 0
            while True:
                rows = cur.fetchmany(50_000)
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=cols))
                n += len(rows)
                print(f"  [{label}] fetched {n:,} rows ({time.time() - t0:,.0f}s)", file=sys.stderr, flush=True)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    finally:
        if own:
            conn.close()
    print(f"  [{label}] done: {len(df):,} rows x {df.shape[1]} cols in {time.time() - t0:,.0f}s", file=sys.stderr, flush=True)
    return df


def split_sql_sections(path: str) -> dict:
    """listeria_collect_queries.sql 을 '-- 섹션 헤더' 기준으로 잘라 {섹션ID: [sql, ...]} 로 돌려준다."""
    import re
    text = open(path, encoding="utf-8").read()
    sections, cur_id = {}, None
    buf = []
    for line in text.splitlines():
        m = re.match(r"^--\s*(\d+(?:-\d+)?)\.\s", line)
        if m:
            if cur_id is not None:
                sections.setdefault(cur_id, []).append("\n".join(buf))
            cur_id, buf = m.group(1), []
            continue
        if line.startswith("-- ====") or line.startswith("-- (선택)") or line.startswith("-- CREATE") or line.startswith("-- SELECT * FROM ready"):
            continue
        buf.append(line)
    if cur_id is not None:
        sections.setdefault(cur_id, []).append("\n".join(buf))
    # 각 섹션 안의 개별 statement 분리 (';' 기준)
    out = {}
    for k, blocks in sections.items():
        stmts = []
        for b in blocks:
            for s in b.split(";"):
                s2 = "\n".join(l for l in s.splitlines() if not l.strip().startswith("--")).strip()
                if s2:
                    stmts.append(s2)
        out[k] = stmts
    return out
