import sys
sys.path.insert(0, r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook')
from db_conn import connect

out = []
with connect() as conn, conn.cursor() as cur:
    cur.execute("SELECT MIN(meas_date)::text, MAX(meas_date)::text, COUNT(DISTINCT stn_id) FROM api_data_info.kma_asos_daily_info")
    out.append('전체: ' + str(cur.fetchone()))
    cur.execute("SELECT substring(meas_date::text, 1, 4) AS y, COUNT(*), COUNT(DISTINCT stn_id) "
                "FROM api_data_info.kma_asos_daily_info WHERE meas_date::text < '2016' GROUP BY 1 ORDER BY 1")
    for r in cur.fetchall():
        out.append(f'{r[0]}: 행 {r[1]}, 지점 {r[2]}')
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(out))
