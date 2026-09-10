import sys
sys.stdout.reconfigure(encoding='utf-8')
from db_utils import connect
conn = connect()
with conn.cursor() as cur:
    cur.execute("""
        SELECT n.nspname, c.relname, c.relkind, c.reltuples::bigint,
               pg_size_pretty(pg_total_relation_size(c.oid)),
               (SELECT string_agg(a.attname, ', ' ORDER BY a.attnum) FROM pg_attribute a
                 WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped) AS cols
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname IN ('weather_data','predict_data') AND c.relkind IN ('r','v','m')
        ORDER BY 1,2""")
    for ns, rel, kind, rows, size, cols in cur.fetchall():
        print(f"[{ns}.{rel}] {kind} rows~{rows:,} {size}\n    {cols[:600]}")
conn.close()
