# -*- coding: utf-8 -*-
"""TF-IDF idf 가 어느 문서 집합(기간)에서 나왔는지 시험: 문서 = 시료 1건의 제품명(prduct_nm)이라 가정.
글자 어휘(char 2~4그램)마다 df = 그 글자열을 포함하는 문서 수 -> sklearn smooth idf = ln((1+N)/(1+df))+1 과 어휘.csv idf 비교.
정리 방식 2가지(소문자+공백 하나 / NFKC+소문자+공백 하나) x 문서 집합(전체·학습 기간·학습+검증)."""
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\c4nb_verify"
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋"
t = pq.read_table(D + r"\시료_시군구.parquet", columns=["prduct_nm", "mnfctur_de"]).to_pandas()
voc = pd.read_csv(D + r"\제품명\단위날짜_tfidf_어휘.csv", encoding="utf-8-sig", keep_default_na=False)
d = t["mnfctur_de"].to_numpy().astype("datetime64[D]")
nm = t["prduct_nm"]
REP = {"prduct_nm_null": int(nm.isna().sum())}
ws = re.compile(r"\s\s+")
sets = {"all": np.ones(len(t), bool), "train": d <= np.datetime64("2024-06-30"), "train_val": d <= np.datetime64("2025-06-30")}
chars = voc[voc["종류"].str.startswith("char")]
words = voc[voc["종류"].str.startswith("word")]
for norm in ("lower", "nfkc_lower"):
    s = nm.fillna("").astype(str)
    if norm == "nfkc_lower":
        s = s.map(lambda x: unicodedata.normalize("NFKC", x))
    s = s.str.lower().map(lambda x: ws.sub(" ", x))
    arr = s.to_numpy(dtype=object)
    for setname, m in sets.items():
        sub = pd.Series(arr[m])
        N = len(sub)
        diffs = []
        for term, idf in zip(chars["어휘"], chars["idf"]):
            df = int(sub.str.contains(term, regex=False).sum())
            diffs.append(abs(np.log((1 + N) / (1 + df)) + 1 - idf))
        diffs = np.array(diffs)
        REP[f"{norm}|{setname}"] = {"N": N, "median_abs_diff": float(np.median(diffs)), "max_abs_diff": float(diffs.max()),
                                     "n_lt_1e-3": int((diffs < 1e-3).sum()), "n_terms": len(diffs)}
        print(norm, setname, REP[f"{norm}|{setname}"], flush=True)
json.dump(REP, open(OUT + r"\v4_idf_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
