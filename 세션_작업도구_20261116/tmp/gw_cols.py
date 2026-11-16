import sys
sys.path.insert(0, r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook')
from db_conn import connect

out = []
with connect() as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT a.attname, format_type(a.atttypid, a.atttypmod), col_description(a.attrelid, a.attnum)
        FROM pg_attribute a
        WHERE a.attrelid = 'api_data_info.global_weather_info'::regclass AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum""")
    for r in cur.fetchall():
        out.append(f'{r[0]} | {r[1]} | {r[2]}')
    cur.execute("SELECT obj_description('api_data_info.global_weather_info'::regclass, 'pg_class')")
    out.append('TABLE COMMENT: ' + str(cur.fetchone()[0]))
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(out))
