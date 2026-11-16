# -*- coding: utf-8 -*-
import pyarrow.parquet as pq, pyarrow.compute as pc, datetime as dt
t = pq.read_table(r"C:/Users/NT751/Desktop/천영진/리스테리아ML/data/listeria_data.parquet",
                  columns=["lims_data_key","judge_result","src_route","mnfctur_de"])
out = open("src/chk_rr8.txt","w",encoding="utf-8")
out.write(f"rows {t.num_rows}\n")
jr = t.column("judge_result").to_pylist()
from collections import Counter
out.write(str(Counter(jr))+"\n")
sr = t.column("src_route").to_pylist()
out.write(str(Counter(zip(sr,jr)))+"\n")
d = t.column("mnfctur_de").to_pylist()
out.write(f"type {type(d[0])} {d[0]}\n")
# 국내 d <= 2015-05-29 행
def todate(x):
    if isinstance(x, dt.datetime): return x.date()
    if isinstance(x, dt.date): return x
    return dt.date.fromisoformat(str(x)[:10])
rows = [(k, j, s, todate(x)) for k, j, s, x in zip(t.column("lims_data_key").to_pylist(), jr, sr, d)]
early = [r for r in rows if r[2]=="domestic" and r[3] <= dt.date(2015,5,29)]
out.write(f"domestic d<=2015-05-29: {len(early)} {Counter(r[1] for r in early)} {sorted(set(r[3] for r in early))}\n")
early2 = [r for r in rows if r[2]=="domestic" and r[3] <= dt.date(2015,5,27)]
out.write(f"domestic d<=2015-05-27: {len(early2)} {Counter(r[1] for r in early2)}\n")
out.close()
