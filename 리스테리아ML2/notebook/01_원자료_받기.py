"""01_원자료_받기.py — 식정원 DB(TOXIN_MODEL_PRED) 원자료 9개 테이블을 받아 리스테리아ML2/data/raw/ 에 parquet 로 저장한다.
(2026-10-16 토양 표 api_data_info.soil_ph_info → raw_soil_ph_info 추가. 나머지 7개 테이블 받는 방법은 바뀌지 않았다)
(2026-10-16 식품 값 표 ba_data.t_aw_cl → raw_t_aw_cl 추가. 받는 방법은 다른 테이블과 같다)

받는 방법
- 테이블마다 SELECT * (모든 칼럼, 모든 행). 값은 받은 그대로 둔다 — 결측 채움·단위 변환·이상값 처리·필터 없음.
- 자료형: DB numeric 만 float64 로 바꾼다(값은 float() 한 그대로). 그 밖의 칼럼은 DB 자료형에 맞는 parquet 자료형으로 적는다
  (text·varchar → string, date → date32, double precision → float64). 칼럼 자료형은 값이 아니라 DB 자료형으로 정한다
  (참고 구현 final_기상조인코드_사본.ipynb 3절 fetch_to_parquet 은 "값이 전부 빈 칼럼"만 DB 자료형으로 정했다.
   여기서는 모든 칼럼을 DB 자료형으로 정해, 나눠 받는 크기와 상관없이 같은 파일이 나오게 했다).
- 서버측 커서(트랜잭션 블록 안의 named cursor)로 CHUNK 행씩 나눠 받아 바로 파일에 쓴다.
- 다시 돌려도 같은 결과가 나오도록 행 순서를 "모든 칼럼을 칼럼 순서대로 오름차순(ORDER BY 1, 2, …, n)"으로 고정한다.
  정렬은 행 순서만 정하고 값·행 수는 바꾸지 않는다. (ORDER BY 가 없으면 DB 가 돌려주는 행 순서는 보장되지 않는다)
- 테이블마다 한 트랜잭션(REPEATABLE READ, READ ONLY) 안에서 count(*) 와 데이터 받기를 함께 해, 같은 시점의 행 수와 비교한다.
- 파일은 <이름>.parquet.part 로 쓴 뒤 다 받으면 <이름>.parquet 로 바꾼다. 이미 있는 파일은 덮어쓴다(돌릴 때마다 DB 에서 다시 받는다).

DB 규칙 — 회사 공용 superuser 계정이다. 읽기 전용 세션(default_transaction_read_only=on)으로 SELECT 만 한다.
서버에 아무 객체도 만들지 않는다(TEMP 테이블 포함). 접속 정보는 출력하지 않는다(접속은 같은 폴더 db_conn.py 의 connect — pgpass 사용).

실행: py "리스테리아ML2\\notebook\\01_원자료_받기.py"   (py 3.14 — pyarrow·psycopg 사용, pandas 쓰지 않음)
      → 9개 테이블 전부 받는다.
      py "리스테리아ML2\\notebook\\01_원자료_받기.py" raw_soil_ph_info   → 이름을 준 파일만 받는다(다른 파일은 건드리지 않음).
      py "리스테리아ML2\\notebook\\01_원자료_받기.py" raw_t_aw_cl        → 식품 값 표만 받는다.
"""
import os
import sys
import time
from pathlib import Path

import psycopg
from psycopg import sql as pgsql
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_conn import connect  # noqa: E402  같은 폴더의 db_conn.py — 읽기 전용 세션 자동 설정, pgpass 사용

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = Path(r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw')
CHUNK = 100_000          # 한 번에 받는 행 수 (메모리 때문에). 값·행 순서·자료형에는 영향이 없다

RAW_TABLES = {   # 파일 이름 → DB 테이블 (스키마.테이블)
    'raw_kma_asos_daily_info':          'api_data_info.kma_asos_daily_info',
    'raw_kma_ground_obs_station_info':  'api_data_info.kma_ground_obs_station_info',
    'raw_nas_agri_weather_daily_info':  'api_data_info.nas_agri_weather_daily_info',
    'raw_nas_agri_weather_obs_station': 'api_data_info.nas_agri_weather_obs_station',
    'raw_keco_airkorea_fine_dust_info': 'api_data_info.keco_airkorea_fine_dust_info',
    'raw_keco_airkorea_obs_station':    'api_data_info.keco_airkorea_obs_station',
    'raw_korea_region_code':            'ba_data_info.korea_region_code',
    'raw_soil_ph_info':                 'api_data_info.soil_ph_info',      # 토양검정 결과 (2026-10-16 추가)
    'raw_t_aw_cl':                      'ba_data.t_aw_cl',                 # 품목코드별 수분활성도·식품pH·염도 (2026-10-16 추가, 06_품목_식품속성.py 가 씀)
}
DATE_COLS = {   # 날짜 칼럼 (보고용 최소·최대) — 일자료 3개 + 토양검정일
    'raw_kma_asos_daily_info': 'meas_date',
    'raw_nas_agri_weather_daily_info': 'meas_date',
    'raw_keco_airkorea_fine_dust_info': 'msur_dt',
    'raw_soil_ph_info': 'exam_day',
}

# PostgreSQL 자료형 번호(oid) → (parquet 자료형, 값 변환). 표에 없는 자료형이 나오면 멈춘다(모르는 변환을 하지 않으려고)
PG_TYPES = {
    25:   (pa.string(), None),        # text
    1043: (pa.string(), None),        # varchar
    1042: (pa.string(), None),        # bpchar
    1082: (pa.date32(), None),        # date
    701:  (pa.float64(), None),       # double precision
    700:  (pa.float64(), None),       # real
    1700: (pa.float64(), 'float'),    # numeric → float64 (값은 float() 그대로)
    20:   (pa.int64(), None),         # bigint
    21:   (pa.int64(), None),         # smallint
    23:   (pa.int64(), None),         # integer
    16:   (pa.bool_(), None),         # boolean
    1114: (pa.timestamp('us'), None),  # timestamp without time zone
}


def to_array(values, typ, conv):
    if conv == 'float':
        values = [None if v is None else float(v) for v in values]
    return pa.array(values, type=typ)


def fetch_to_parquet(conn, name, table):
    """한 테이블을 서버측 커서로 나눠 받아 RAW_DIR/<name>.parquet 로 저장. (받은 행 수, DB count(*), 칼럼 수) 를 돌려준다"""
    out_path = RAW_DIR / f'{name}.parquet'
    tmp = RAW_DIR / f'{name}.parquet.part'
    schema_name, table_name = table.split('.')
    ident = pgsql.Identifier(schema_name, table_name)
    writer, n = None, 0
    try:
        with conn.transaction():        # REPEATABLE READ · READ ONLY — count(*) 와 데이터가 같은 시점을 본다
            with conn.cursor() as cur:
                cur.execute(pgsql.SQL('select count(*) from {}').format(ident))
                db_count = cur.fetchone()[0]
                cur.execute(pgsql.SQL('select * from {} limit 0').format(ident))
                ncol = len(cur.description)
            order_by = pgsql.SQL(', ').join(pgsql.Literal(i) for i in range(1, ncol + 1))
            sql = pgsql.SQL('select * from {} order by {}').format(ident, order_by)
            with conn.cursor(name='raw_fetch') as cur:      # 서버측 커서 (named cursor)
                cur.execute(sql)
                fields, convs = [], []
                for d in cur.description:
                    if d.type_code not in PG_TYPES:
                        raise TypeError(f'{table}.{d.name}: 표에 없는 DB 자료형 oid {d.type_code}')
                    typ, conv = PG_TYPES[d.type_code]
                    fields.append(pa.field(d.name, typ))
                    convs.append(conv)
                schema = pa.schema(fields, metadata={'source_table': table, 'sql': sql.as_string(conn)})
                writer = pq.ParquetWriter(tmp, schema)
                while rows := cur.fetchmany(CHUNK):
                    cols = list(zip(*rows))
                    arrays = [to_array(cols[i], f.type, convs[i]) for i, f in enumerate(fields)]
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    n += len(rows)
                    del rows, cols, arrays
        writer.close()
        writer = None
        os.replace(tmp, out_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
    return n, db_count, len(fields)


def main(names=None):
    """names: 받을 파일 이름 리스트 (None 이면 RAW_TABLES 전부)"""
    names = list(RAW_TABLES) if not names else names
    unknown = [n for n in names if n not in RAW_TABLES]
    assert not unknown, f'RAW_TABLES 에 없는 이름: {unknown}'
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print('결과 폴더:', RAW_DIR)
    print('받을 파일:', ', '.join(names))
    result = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute('show default_transaction_read_only')
            assert cur.fetchone()[0] == 'on', '읽기 전용 세션이 아님'
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        for name in names:
            table = RAW_TABLES[name]
            t0 = time.time()
            n, db_count, ncol = fetch_to_parquet(conn, name, table)
            result.append((name, table, n, db_count, ncol, time.time() - t0))
            print(f'받음 {name} ← {table}: {n:,}행 · {ncol}칼럼 · DB count(*) {db_count:,} · {"일치" if n == db_count else "불일치"} ({time.time() - t0:.0f}초)')

    print('\n== 파일별 결과 ==')
    print(f'{"파일":36s} {"행 수":>11s} {"칼럼":>5s} {"DB count(*)":>12s} {"일치":>4s} {"파일 크기(MB)":>13s}')
    for name, table, n, db_count, ncol, sec in result:
        p = RAW_DIR / f'{name}.parquet'
        pf = pq.ParquetFile(p)
        assert pf.metadata.num_rows == n and pf.metadata.num_columns == ncol
        print(f'{name + ".parquet":36s} {n:>11,} {ncol:>5} {db_count:>12,} {"O" if n == db_count else "X":>4s} {p.stat().st_size / 1e6:>13.1f}')

    print('\n== 날짜 칼럼 최소·최대 ==')
    for name, col in DATE_COLS.items():
        if name not in names:
            continue
        a = pq.read_table(RAW_DIR / f'{name}.parquet', columns=[col])[col]
        mm = pc.min_max(a)
        print(f'{name}.{col}: {mm["min"].as_py()} ~ {mm["max"].as_py()} (결측 {a.null_count:,}행, 날짜 수 {pc.count_distinct(a).as_py():,})')

    bad = [r[0] for r in result if r[2] != r[3]]
    assert not bad, f'DB count(*) 와 행 수가 다른 파일: {bad}'
    print('\n모든 파일의 행 수가 DB count(*) 와 같다.')


if __name__ == '__main__':
    main(sys.argv[1:])
