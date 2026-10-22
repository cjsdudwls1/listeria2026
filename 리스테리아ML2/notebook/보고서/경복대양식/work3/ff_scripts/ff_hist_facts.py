# -*- coding: utf-8 -*-
"""과거 이력: 학습 끝 날(겹·최종)마다 전국 5년·1년 창 시료·검출 수와 p0, 품목 3단계·시군구·단위 수(읽기만).
학습 끝 값 고정 창 = 기준일 ∈ [TE − P + 1, TE] (cv.PAST_RULE_DESC)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
DS = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋")
S = pq.read_table(DS / "시료_시군구.parquet", columns=["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "mnfctur_de", "judge_result"]).to_pandas()
d = S["mnfctur_de"].values.astype("datetime64[D]")
pos = (S["judge_result"].astype(object) == "검출").to_numpy()
print("시료", len(S), "검출", int(pos.sum()), "기준일", d.min(), "~", d.max())
for te in ["2022-06-30", "2023-06-30", "2024-06-30", "2025-06-30"]:
    TE = np.datetime64(te)
    for P, nm in ((1826, "5년"), (365, "1년")):
        m = (d >= TE - np.timedelta64(P - 1, "D")) & (d <= TE)
        n, s = int(m.sum()), int(pos[m].sum())
        print(f"  학습 끝 {te} {nm} 창 [{TE - np.timedelta64(P - 1, 'D')}, {te}]: 시료 {n:,} 검출 {s:,} p0 {100 * s / n:.4f}%")
# 연도별 시료·검출 (기준일)
y = pd.Series(d.astype("datetime64[Y]").astype(int) + 1970)
t = pd.DataFrame({"y": y, "pos": pos}).groupby("y").agg(n=("pos", "size"), s=("pos", "sum"))
print(t.to_string())
