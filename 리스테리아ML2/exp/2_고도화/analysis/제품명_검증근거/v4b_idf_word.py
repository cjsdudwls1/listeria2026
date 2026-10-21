# -*- coding: utf-8 -*-
"""낱말 어휘(word 1~2) idf 기간 시험: 문서 = 시료 제품명, 낱말 = NFKC·소문자 뒤 한글·영문·숫자 덩어리(추정 규칙). 1낱말 어휘만."""
import json, re, unicodedata
import numpy as np, pandas as pd, pyarrow.parquet as pq
OUT = r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\c4nb_verify"
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋"
t = pq.read_table(D + r"\시료_시군구.parquet", columns=["prduct_nm", "mnfctur_de"]).to_pandas()
voc = pd.read_csv(D + r"\제품명\단위날짜_tfidf_어휘.csv", encoding="utf-8-sig", keep_default_na=False)
w = voc[voc["종류"].str.startswith("word")]
uni = w[~w["어휘"].str.contains(" ")]
d = t["mnfctur_de"].to_numpy().astype("datetime64[D]")
tok = re.compile(r"[가-힣a-z0-9]+")
toks = t["prduct_nm"].fillna("").astype(str).map(lambda x: set(tok.findall(unicodedata.normalize("NFKC", x).lower()))).to_numpy(dtype=object)
REP = {"n_word_terms": len(w), "n_unigram": len(uni)}
for setname, m in {"all": np.ones(len(t), bool), "train": d <= np.datetime64("2024-06-30"), "train_val": d <= np.datetime64("2025-06-30")}.items():
    sub = toks[m]; N = len(sub)
    from collections import Counter
    c = Counter()
    for s in sub: c.update(s)
    diffs = np.array([abs(np.log((1 + N) / (1 + c.get(term, 0))) + 1 - idf) for term, idf in zip(uni["어휘"], uni["idf"])])
    REP[setname] = {"N": N, "median_abs_diff": float(np.median(diffs)), "n_lt_1e-3": int((diffs < 1e-3).sum()), "n": len(diffs)}
    print(setname, REP[setname], flush=True)
json.dump(REP, open(OUT + r"\v4b_idf_word_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
