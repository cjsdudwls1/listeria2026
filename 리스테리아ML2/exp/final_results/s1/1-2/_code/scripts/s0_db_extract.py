# -*- coding: utf-8 -*-
"""s0_db_extract.py — 최종 실험 단계 0 에 필요한 식정원 DB(TOXIN_MODEL_PRED) 자료를 읽기 전용으로 받는다.

받는 표·칼럼·조건 = configs/s0_db_extract.json (표마다 계획서 절은 "plan" 칸).
- SELECT 만. 세션 읽기 전용(접속 옵션 default_transaction_read_only=on + 트랜잭션 READ ONLY, 둘 다 이 세션에만 적용).
  서버·계정 설정 변경 없음, TEMP 테이블·뷰 등 서버 객체 만들지 않음.
- 접속 정보 = <root>/리스테리아ML/.env 의 DB_HOST·DB_PORT·DB_NAME·DB_USER·DB_PASSWORD (dotenv). 값은 출력·로그·파일에 남기지 않음.
- 표마다 한 트랜잭션(REPEATABLE READ, READ ONLY) 안에서 count(*)(같은 WHERE) 와 데이터 받기 → 같은 시점 행 수 대조.
- 값은 받은 그대로(결측 채움·단위 변환·집계 없음). 리스테리아 범위 조건은 설정의 "where" 그대로.
- 행 순서 = DB 에서 정렬하지 않고(공용 DB 부담·statement_timeout 회피) 받은 뒤 로컬 pyarrow 로 모든 받는 칼럼 오름차순 정렬(null 은 끝) — 다시 받아도 같은 파일이 나오게.
- 출력: <out>/<name>.parquet + <name>.json(행 수·DB count·md5·쿼리문·칼럼·자료형·받은 시각·code_md5·config_md5).
  이미 있는 파일은 덮지 않고 멈춤. 받는 중에는 <name>.parquet.part 로 쓰고 다 받으면 이름을 바꿈.

구현 세부(계획서에 없음 — 담당자 확인 필요):
- DB numeric → float64(값은 float() 그대로, 기존 notebook/01_원자료_받기.py 규칙과 같음). 그 밖은 DB 자료형대로.
- ready_data.hazard_preprocessed_data 를 hazard_label = '리스테리아' 로 거름(계획서는 s_org 의 label 조건만 적음).
  시료 파일 hazard_label 은 전부 '리스테리아'라 시료 범위를 넘지 않음. 시료와는 lims_data_key 로 로컬에서 이음.
- 드라이버: psycopg2 우선, 없으면 psycopg(3). 같은 SQL·같은 자료형 규칙.

실행(지휘자, 검수 뒤):
  <python> -I 리스테리아ML2/exp/final/scripts/s0_db_extract.py --root C:/Users/NT751/Desktop/천영진 \
      --out C:/Users/NT751/Desktop/천영진/리스테리아ML2/data/추출_20261107 [--only name1 name2 ...] [--dry-run]
  --dry-run: DB 에 information_schema.columns 만 물어 칼럼 확인 + 보낼 SQL 출력(자료 SELECT 안 함)
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# 경로·접속을 덮는 환경변수 정리(부록 B-2) — libpq 가 읽는 PG* 변수가 .env 값을 덮거나 세션 옵션을 끼워 넣지 않게
for _k in ['PGHOST', 'PGHOSTADDR', 'PGPORT', 'PGDATABASE', 'PGUSER', 'PGPASSWORD', 'PGPASSFILE', 'PGSERVICE',
           'PGSERVICEFILE', 'PGSYSCONFDIR', 'PGOPTIONS', 'PGTARGETSESSIONATTRS']:
    os.environ.pop(_k, None)

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import dotenv_values

try:
    import psycopg2 as _drv
    from psycopg2 import sql as pgsql
    DRIVER = 'psycopg2'
except ImportError:  # psycopg2 가 없을 때만
    import psycopg as _drv
    from psycopg import sql as pgsql
    DRIVER = 'psycopg'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CONFIG_KEYS = {'env_file', 'chunk_rows', 'statement_timeout_ms', 'tables'}
TABLE_KEYS = {'name', 'table', 'columns', 'where', 'plan'}

# PostgreSQL 자료형 oid → (parquet 자료형, 변환). 표에 없는 자료형이면 멈춤
PG_TYPES = {
    25: (pa.string(), None), 1043: (pa.string(), None), 1042: (pa.string(), None),   # text·varchar·bpchar
    1082: (pa.date32(), None),                                                       # date
    701: (pa.float64(), None), 700: (pa.float64(), None),                            # double·real
    1700: (pa.float64(), 'float'),                                                   # numeric → float64
    20: (pa.int64(), None), 21: (pa.int64(), None), 23: (pa.int64(), None),          # bigint·smallint·integer
    16: (pa.bool_(), None),                                                          # boolean
    1114: (pa.timestamp('us'), None),                                                # timestamp
    1184: (pa.timestamp('us', tz='UTC'), None),                                      # timestamptz
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    unknown = set(cfg) - CONFIG_KEYS
    missing = CONFIG_KEYS - set(cfg)
    assert not unknown and not missing, f'설정 칸 이상: 모르는 칸 {unknown}, 빠진 칸 {missing}'
    names = [t['name'] for t in cfg['tables']]
    assert len(names) == len(set(names)), '설정 표 이름 중복'
    for t in cfg['tables']:
        assert set(t) == TABLE_KEYS, f'{t.get("name")}: 표 설정 칸 이상 {set(t) ^ TABLE_KEYS}'
        assert t['where'] is None or (';' not in t['where'] and '(' not in t['where']), f'{t["name"]}: where 에 ; ( 금지(칼럼 = 리터럴만)'
    return cfg


def connect(env):
    keys = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    empty = [k for k in keys if not env.get(k)]
    assert not empty, f'.env 에 빈 키: {empty}'   # 키 이름만 출력, 값은 출력 안 함
    try:   # 실패 메시지에 호스트·포트·계정이 들어가므로 원 메시지를 끊음(값 출력 금지)
        conn = _drv.connect(host=env['DB_HOST'], port=int(env['DB_PORT']), dbname=env['DB_NAME'], user=env['DB_USER'],
                            password=env['DB_PASSWORD'], connect_timeout=15, application_name='listeria_final_s0_extract',
                            options=f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}")
    except Exception as e:
        raise SystemExit(f'DB 접속 실패({type(e).__name__}) — 메시지는 접속 정보가 들어 있어 출력하지 않음') from None
    if DRIVER == 'psycopg2':
        conn.set_session(isolation_level='REPEATABLE READ', readonly=True, autocommit=False)
    else:
        conn.isolation_level = _drv.IsolationLevel.REPEATABLE_READ
        conn.read_only = True
        conn.autocommit = False
    cur = conn.cursor()
    cur.execute('show default_transaction_read_only')
    assert cur.fetchone()[0] == 'on', '읽기 전용 세션이 아님'
    cur.execute('show transaction_read_only')
    assert cur.fetchone()[0] == 'on', '트랜잭션이 읽기 전용이 아님'
    cur.close()
    conn.rollback()
    return conn


def db_columns(conn, table):
    s, t = table.split('.')
    cur = conn.cursor()
    cur.execute('select column_name, data_type from information_schema.columns '
                'where table_schema = %s and table_name = %s order by ordinal_position', (s, t))
    rows = cur.fetchall()
    cur.close()
    conn.rollback()
    assert rows, f'{table}: information_schema 에 칼럼 없음(표 없음?)'
    return rows


def build_sql(spec, cols):
    s, t = spec['table'].split('.')
    ident = pgsql.Identifier(s, t)
    col_sql = pgsql.SQL(', ').join(pgsql.Identifier(c) for c in cols)
    where = pgsql.SQL(' where ' + spec['where']) if spec['where'] else pgsql.SQL('')
    data_sql = pgsql.SQL('select {} from {}{}').format(col_sql, ident, where)   # 정렬은 로컬에서
    count_sql = pgsql.SQL('select count(*) from {}{}').format(ident, where)
    return data_sql, count_sql


def fetch_table(conn, spec, cols, out_dir, chunk):
    name = spec['name']
    out_pq, out_js, tmp = out_dir / f'{name}.parquet', out_dir / f'{name}.json', out_dir / f'{name}.parquet.part'
    for p in (out_pq, out_js, tmp):
        assert not p.exists(), f'이미 있는 파일 — 덮지 않음: {p}'
    data_sql, count_sql = build_sql(spec, cols)
    data_txt, count_txt = data_sql.as_string(conn), count_sql.as_string(conn)
    t0 = time.time()
    started = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    writer, n = None, 0
    try:
        cur = conn.cursor()
        cur.execute(count_txt)
        db_count = cur.fetchone()[0]
        cur.close()
        cur = conn.cursor(name='s0_fetch')          # 서버측 커서 — 같은 트랜잭션(같은 시점)
        cur.execute(data_txt)
        first = cur.fetchmany(chunk)
        desc = cur.description
        fields, convs = [], []
        for d in desc:
            if d[1] not in PG_TYPES:
                raise TypeError(f'{spec["table"]}.{d[0]}: 표에 없는 DB 자료형 oid {d[1]}')
            typ, conv = PG_TYPES[d[1]]
            fields.append(pa.field(d[0], typ))
            convs.append(conv)
        schema = pa.schema(fields, metadata={'source_table': spec['table'], 'sql': data_txt})
        parts = []
        rows = first
        while rows:
            colvals = list(zip(*rows))
            arrays = []
            for i, f in enumerate(fields):
                v = colvals[i]
                if convs[i] == 'float':
                    v = [None if x is None else float(x) for x in v]
                arrays.append(pa.array(v, type=f.type))
            parts.append(pa.Table.from_arrays(arrays, schema=schema))
            n += len(rows)
            rows = cur.fetchmany(chunk)
        cur.close()
        table = pa.concat_tables(parts) if parts else schema.empty_table()
        if n > 0:   # 받은 뒤 로컬에서 모든 칼럼 오름차순(null 끝) — 같은 자료면 같은 파일
            table = table.sort_by([(f.name, 'ascending') for f in fields])
        writer = pq.ParquetWriter(tmp, schema)
        writer.write_table(table)
        writer.close()
        writer = None
        conn.rollback()      # 읽기 전용 트랜잭션 끝(바뀐 것 없음)
        os.replace(tmp, out_pq)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
    meta = {
        'name': name, 'table': spec['table'], 'where': spec['where'], 'plan': spec['plan'],
        'sql': data_txt, 'count_sql': count_txt, 'rows': n, 'db_count': db_count, 'rows_match': n == db_count,
        'columns': [{'name': f.name, 'parquet_type': str(f.type)} for f in fields],
        'bpchar_columns(끝 공백 그대로)': [d[0] for d in desc if d[1] == 1042],
        'numeric_to_float64_columns': [d[0] for d in desc if d[1] == 1700],
        'parquet_md5': md5_file(out_pq), 'parquet_bytes': out_pq.stat().st_size,
        'started_at': started, 'seconds': round(time.time() - t0, 1),
        'transaction': 'REPEATABLE READ, READ ONLY (count 와 데이터 같은 트랜잭션)',
        'driver': f'{DRIVER} {_drv.__version__}', 'python': sys.version.split()[0], 'pyarrow': pa.__version__,
        'code_md5': CODE_MD5, 'config_md5': CONFIG_MD5,
    }
    out_js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='자료 뿌리(천영진 폴더)')
    ap.add_argument('--out', required=True, help='추출 결과 폴더(예: <root>/리스테리아ML2/data/추출_20261107)')
    ap.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'configs' / 's0_db_extract.json'))
    ap.add_argument('--only', nargs='*', default=None, help='받을 표 name 만(없으면 설정 전부)')
    ap.add_argument('--dry-run', action='store_true', help='information_schema 칼럼 확인 + SQL 출력만')
    a = ap.parse_args()

    global CODE_MD5, CONFIG_MD5, STATEMENT_TIMEOUT_MS
    CODE_MD5 = md5_file(__file__)
    CONFIG_MD5 = md5_file(a.config)
    cfg = load_config(a.config)
    STATEMENT_TIMEOUT_MS = int(cfg['statement_timeout_ms'])
    root, out_dir = Path(a.root), Path(a.out)
    specs = cfg['tables']
    if a.only:
        bad = set(a.only) - {t['name'] for t in specs}
        assert not bad, f'설정에 없는 name: {bad}'
        specs = [t for t in specs if t['name'] in a.only]

    env = dotenv_values(root / cfg['env_file'])
    conn = connect(env)
    print(f'접속(읽기 전용 확인) — 드라이버 {DRIVER}')
    try:
        plan = []
        for spec in specs:
            dbinfo = db_columns(conn, spec['table'])
            dbcols = [c for c, _ in dbinfo]
            cols = dbcols if spec['columns'] is None else spec['columns']
            miss = [c for c in cols if c not in dbcols]
            assert not miss, f'{spec["table"]}: DB 에 없는 칼럼 {miss}'
            plan.append((spec, cols))
            data_sql, count_sql = build_sql(spec, cols)
            print(f'- {spec["name"]} ← {spec["table"]} ({len(cols)}칼럼) | {data_sql.as_string(conn)}')
            print('    자료형: ' + ', '.join(f'{c}={ty}' for c, ty in dbinfo if c in cols))
        if a.dry_run:
            print('dry-run: 자료 SELECT 안 함')
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        for spec, _ in plan:   # 받기 전에 모든 출력 파일을 한 번에 검사(덮지 않음)
            for suf in ('.parquet', '.json', '.parquet.part'):
                assert not (out_dir / f'{spec["name"]}{suf}').exists(), f'이미 있는 파일 — 덮지 않음: {spec["name"]}{suf}'
        for spec, cols in plan:
            m = fetch_table(conn, spec, cols, out_dir, int(cfg['chunk_rows']))
            print(f'받음 {m["name"]}: {m["rows"]:,}행 · DB count {m["db_count"]:,} · '
                  f'{"일치" if m["rows_match"] else "불일치"} · md5 {m["parquet_md5"]} · {m["seconds"]}초')
            assert m['rows_match'], f'{m["name"]}: 행 수가 DB count 와 다름'
    finally:
        conn.rollback()
        conn.close()


if __name__ == '__main__':
    main()
