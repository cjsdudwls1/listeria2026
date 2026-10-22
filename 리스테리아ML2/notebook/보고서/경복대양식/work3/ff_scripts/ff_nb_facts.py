# -*- coding: utf-8 -*-
"""제품명 뭉치: 뭉치별 첫 시료 날짜(학습 행 N1 스무딩 검출률 결측 이유), 그날 유형 N2 가 하나라도 1 인 행 비율(행 묶음별). 읽기만."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
DS = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋")
NULL = "코드없음"
S = pq.read_table(DS / "시료_시군구.parquet", columns=["lims_data_key", "sgg_code", "train_prdlst_code", "train_prdlst_code_2", "mnfctur_de"]).to_pandas()
nb = pq.read_table(DS / "제품명" / "시료별_뭉치.parquet").to_pandas()
assert (nb["lims_data_key"].astype(str).to_numpy() == S["lims_data_key"].astype(str).to_numpy()).all()
CL = [c for c in nb.columns if c.startswith("nb_")]
d = S["mnfctur_de"].values.astype("datetime64[D]")
print("뭉치별 첫 시료 날짜 · 전체 1 인 시료")
for c in CL:
    m = nb[c].to_numpy() == 1
    print(f"  {c}: 첫 {d[m].min()}  마지막 {d[m].max()}  시료 {int(m.sum()):,}")
# N2: 단위 x 날짜 행에서 하나라도 1
key = pd.DataFrame({"sgg": S["sgg_code"].astype(str), "p2": S["train_prdlst_code"].astype(object).where(S["train_prdlst_code"].notna(), NULL).astype(str),
                    "p3": S["train_prdlst_code_2"].astype(object).where(S["train_prdlst_code_2"].notna(), NULL).astype(str), "d": d})
anyf = nb[CL].to_numpy().max(1)
key["any"] = anyf
g = key.groupby(["sgg", "p2", "p3", "d"], sort=False)["any"].max().reset_index()
g["d"] = g["d"].values.astype("datetime64[D]")
print("단위 x 날짜 행", len(g), "N2 하나라도 1 인 행", int(g["any"].sum()), f"{100 * g['any'].mean():.2f}%")
for nm, lo, hi in (("~2025-06-30", "2015-01-01", "2025-06-30"), ("2025-07-01~", "2025-07-01", "2026-06-30")):
    m = (g["d"] >= np.datetime64(lo)) & (g["d"] <= np.datetime64(hi))
    print(f"  기준일 {nm}: 행 {int(m.sum()):,}, 하나라도 1 {100 * g.loc[m, 'any'].mean():.2f}%")
