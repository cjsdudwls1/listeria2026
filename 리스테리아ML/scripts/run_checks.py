import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from db_utils import connect, split_sql_sections
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 40)
secs = split_sql_sections(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\sql\listeria_collect_queries.sql")
conn = connect()
for sid in ["1-1","1-2","1-3","1-4","1-5","1-6","1-7","1-8","1-9"]:
    for i, sql in enumerate(secs.get(sid, [])):
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute(sql)
            df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
        print(f"\n===== {sid} (#{i+1}) {time.time()-t0:.0f}s =====")
        print(df.to_string(index=False))
conn.close()
