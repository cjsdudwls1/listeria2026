# -*- coding: utf-8 -*-
"""db_conn.py — 식정원 DB(TOXIN_MODEL_PRED) 읽기 전용 접속.

- 접속 정보(주소·포트·DB 이름·사용자·비밀번호)는 이 파일에 적지 않는다.
  사용자 PC 의 pgpass 파일(%APPDATA%\\postgresql\\pgpass.conf, 또는 환경변수 PGPASSFILE)에서
  DB 이름이 TOXIN_MODEL_PRED 인 줄을 찾아 주소·포트·DB 이름·사용자를 읽는다.
  비밀번호는 넘기지 않는다 — psycopg(libpq)가 같은 pgpass 파일에서 스스로 읽는다.
- 세션은 읽기 전용: options "-c default_transaction_read_only=on -c statement_timeout=<ms>",
  connect_timeout, autocommit=True.
- 회사 공용 superuser 계정이다. SELECT 만 하고, 서버에 아무 객체도 만들지 않는다(TEMP 테이블 포함).
  접속 정보는 출력하지 않는다.

사용: from db_conn import connect  →  with connect() as conn: ...
"""
import os

import psycopg

DB_NAME = "TOXIN_MODEL_PRED"
STATEMENT_TIMEOUT_MS = 120_000
CONNECT_TIMEOUT_S = 15
APPLICATION_NAME = "listeria_ml2_notebook"


def _pgpass_path():
    p = os.environ.get("PGPASSFILE")
    if p:
        return p
    return os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")


def _split_pgpass_line(line):
    """pgpass 한 줄을 ':' 로 나눈다. '\\:' 는 글자 ':', '\\\\' 는 글자 '\\'."""
    parts, cur, i = [], [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur.append(line[i + 1])
            i += 2
            continue
        if ch == ":":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _pgpass_entry(db_name=DB_NAME):
    """pgpass 에서 DB 이름이 db_name 인 첫 줄의 (host, port, dbname, user). 값은 돌려주기만 하고 출력하지 않는다."""
    with open(_pgpass_path(), encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = _split_pgpass_line(line)
            if len(parts) < 5:
                continue
            host, port, dbname, user = parts[:4]
            if dbname == db_name and host != "*" and user != "*":
                return host, (5432 if port == "*" else int(port)), dbname, user
    raise RuntimeError(f"pgpass 에 DB 이름 {db_name} 줄이 없음")


def connect(statement_timeout_ms=STATEMENT_TIMEOUT_MS, application_name=APPLICATION_NAME):
    """읽기 전용 세션(default_transaction_read_only=on)으로 접속한 psycopg 연결(autocommit=True)."""
    host, port, dbname, user = _pgpass_entry()
    conn = psycopg.connect(
        host=host, port=port, dbname=dbname, user=user,
        options=f"-c default_transaction_read_only=on -c statement_timeout={int(statement_timeout_ms)}",
        application_name=application_name, connect_timeout=CONNECT_TIMEOUT_S)
    conn.autocommit = True
    return conn
