# -*- coding: utf-8 -*-
"""검수용(읽기만): 4.2.3 '검사 수는 0건부터 18만여 건까지' 문구 확인.
과거 검출 이력의 검사 수 n = 창 안 그 무리(품목 3단계 전국·시군구·단위) 시료 수.
창 끝(맞혀 보는 날 T 전날 또는 학습 끝 날)이 2025-06-30 이하인 5년(1,826일)·1년(365일) 창에서 n 의 최댓값을 센다.
자료: 리스테리아ML2/data/데이터셋/시료_시군구.parquet (읽기만)
"""
import sys
import pandas as pd
import numpy as np

SRC = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시료_시군구.parquet"
END = pd.Timestamp("2025-06-30")

df = pd.read_parquet(SRC, columns=["mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_code_2"])
df["d"] = pd.to_datetime(df["mnfctur_de"]).dt.normalize()
df["sgg_code"] = df["sgg_code"].astype(str)
df["p2"] = df["train_prdlst_code"].astype(object).fillna("코드없음").astype(str)
df["p3"] = df["train_prdlst_code_2"].astype(object).fillna("코드없음").astype(str)
df["unit"] = df["sgg_code"].astype(str) + "|" + df["p2"] + "|" + df["p3"]
print("시료", len(df), "기준일", df["d"].min().date(), "~", df["d"].max().date())

days = pd.date_range(df["d"].min(), END, freq="D")
for P in (1826, 365):
    for lvl in ("p3", "sgg_code", "unit"):
        g = df[df["d"] <= END].groupby([lvl, "d"]).size()
        best = (0, None, None)
        for key, s in g.groupby(level=0):
            s = s.droplevel(0).reindex(days, fill_value=0)
            r = s.rolling(P, min_periods=1).sum()
            i = int(np.argmax(r.values))
            if r.values[i] > best[0]:
                best = (int(r.values[i]), key, r.index[i].date())
        print("창 %d일 %s: 최대 %s건 (%s, 창 끝 %s)" % (P, lvl, format(best[0], ","), best[1], best[2]))
