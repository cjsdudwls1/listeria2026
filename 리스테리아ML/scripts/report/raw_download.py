# -*- coding: utf-8 -*-
"""11개 변수가 들어 있는 테이블을 DB 원형 그대로 받는다 (SELECT *, 필터·캐스팅·결측 처리 없음).
text 는 text 그대로, date 는 date 그대로. numeric 은 parquet 저장을 위해 float64 로만 담는다(값 동일)."""
import sys, time, os, decimal
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML\scripts")
import pandas as pd
from db_utils import connect

OUT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\raw"
os.makedirs(OUT, exist_ok=True)
TABLES = ['kma_weather_predict_info', 'kma_ground_obs_station_info', 'nas_agri_weather_obs_station', 'keco_airkorea_obs_station',
          'kma_asos_daily_info', 'nas_agri_weather_daily_info', 'keco_airkorea_fine_dust_info']
conn = connect()
for t in TABLES:
    path = os.path.join(OUT, f"raw_{t}.parquet")
    if os.path.exists(path):
        print("skip", t); continue
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='api_data_info' AND table_name=%s ORDER BY ordinal_position", (t,))
        types = dict(cur.fetchall())
    with conn.transaction():
        with conn.cursor(name=f"raw_{t}") as cur:
            cur.execute(f"SELECT * FROM api_data_info.{t}")
            cols = [d.name for d in cur.description]
            chunks = []
            while True:
                rows = cur.fetchmany(200_000)
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=cols))
    df = pd.concat(chunks, ignore_index=True)
    for c in cols:
        if types.get(c) == 'numeric':
            df[c] = pd.to_numeric(df[c], errors='raise').astype('float64')   # Decimal -> float (값 동일)
        elif types.get(c) == 'date':
            df[c] = pd.to_datetime(df[c]).astype('datetime64[ms]')
    df.to_parquet(path, index=False)
    print(f"{t}: {len(df):,} rows x {df.shape[1]} cols ({time.time() - t0:.0f}s)", flush=True)
    del df, chunks
conn.close()
print("DONE")
