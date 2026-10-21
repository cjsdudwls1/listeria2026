# -*- coding: utf-8 -*-
"""독립 검증 1단계(만든 쪽 src 를 import 하지 않음). 원자료 parquet 만 읽어 직접 계산.
- 시료 -> 단위x날짜 행, C4 목록·C4 밖 검출, N2(단위x날짜 뭉치 OR, pandas groupby max), N0(tfidf 키 결합),
- (행, h) 쌍 선택(무작위 + 학습 초기 + h=60 + 검출 행 + 코드없음 행 + T 당일 시료 있는 쌍), N1 을 반복문(쌍마다 원시 마스크)으로 계산.
- 전체 행 N1(h=1, h=60)은 별도 벡터 계산(누적합)으로도 만들어 쌍 결과와 맞춤.
출력: 이 폴더의 my_*.parquet / *.npy / v1_report.json
"""
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\c4nb_verify"
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋"
SAMPLE = D + r"\시료_시군구.parquet"
NBS = D + r"\제품명\시료별_뭉치.parquet"
NBDEF = D + r"\제품명\뭉치_정의.csv"
TFIDF = D + r"\제품명\단위날짜_tfidf.parquet"
VOCAB = D + r"\제품명\단위날짜_tfidf_어휘.csv"
NULL = "코드없음"
TRAIN_END = np.datetime64("2024-06-30")
VAL = (np.datetime64("2024-07-01"), np.datetime64("2025-06-30"))
TEST = (np.datetime64("2025-07-01"), np.datetime64("2026-06-30"))
K = 10.0
PERIODS = [(1826, "5y"), (365, "1y")]
REP = {}
t00 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    with open(OUT + r"\v1_log.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def fill(col):
    s = col.astype(object)
    return s.where(s.notna(), NULL).astype(str).to_numpy(dtype=object)


# ---------------------------------------------------------------- 시료
t = pq.read_table(SAMPLE, columns=["lims_data_key", "sgg_code", "train_prdlst_code", "train_prdlst_code_2",
                                   "mnfctur_de", "judge_result"]).to_pandas()
S = pd.DataFrame({"key": t["lims_data_key"].astype(str).to_numpy(),
                  "sgg": t["sgg_code"].astype(str).to_numpy(dtype=object),
                  "p2": fill(t["train_prdlst_code"]), "p3": fill(t["train_prdlst_code_2"]),
                  "d": t["mnfctur_de"].to_numpy().astype("datetime64[D]")})
jr = t["judge_result"].astype(str).to_numpy()
assert set(np.unique(jr)) == {"검출", "불검출"}, np.unique(jr)
S["pos"] = (jr == "검출").astype(np.int8)
REP["samples"] = {"n": len(S), "pos": int(S["pos"].sum()), "p2_null": int(t["train_prdlst_code"].isna().sum()),
                  "p3_null": int(t["train_prdlst_code_2"].isna().sum()), "d_min": str(S["d"].min()), "d_max": str(S["d"].max())}
dd = S["d"].to_numpy()
split = np.where(dd <= TRAIN_END, "train", np.where((dd >= VAL[0]) & (dd <= VAL[1]), "val",
                                                   np.where((dd >= TEST[0]) & (dd <= TEST[1]), "test", "")))
assert (split != "").all()
S["split"] = split
log("시료", REP["samples"])

# ---------------------------------------------------------------- C4 (학습 구간 시료만)
tr = S[S["split"] == "train"]
g = tr.groupby("p3").agg(n=("pos", "size"), s=("pos", "sum")).reset_index()
c4 = g[g["s"] >= 1].sort_values(["s", "p3"], ascending=[False, True]).reset_index(drop=True)
C4 = c4["p3"].tolist()
REP["C4"] = {"n_codes": len(C4), "codes": C4, "pos_by_code": dict(zip(c4["p3"], c4["s"].astype(int).tolist())),
             "train_samples_in_C4": int(c4["n"].sum()), "train_samples_total": int(len(tr)),
             "train_pos_in_C4": int(c4["s"].sum()), "train_pos_total": int(tr["pos"].sum()),
             "null_code_in_C4": NULL in C4}
# 비교: 전 기간 시료로 골랐다면(누수 대조) 목록이 달라지는가
gall = S.groupby("p3")["pos"].sum()
REP["C4"]["codes_if_all_periods"] = sorted(gall[gall >= 1].index.tolist())
REP["C4"]["all_period_extra_codes"] = sorted(set(REP["C4"]["codes_if_all_periods"]) - set(C4))
PRODUCER_C4 = ["Z2700400000000", "Z3800100000000", "C0316050000000", "C0316080000000", "C0316010000000", "C0322020000000",
               "C0319060000000", "C0316020000000", "A0700600000000", "B0101000000000", "C0318090000000", "C0316030000000",
               "C0316040000000", "C0323020000000", "A1217900000000", "C0319010000000", "A0400100000000", "B0203000000000",
               "B0205000000000", "B0209000000000", "C0309010000000", "C0316060000000", "C0316070000000", "C0318010000000",
               "C0319030000000", "C0322030000000"]
REP["C4"]["same_as_producer_report_list_and_order"] = C4 == PRODUCER_C4
log("C4", REP["C4"])

# ---------------------------------------------------------------- 단위x날짜 행
KEYS = ["sgg", "p2", "p3", "d"]
S["rid"] = S.groupby(KEYS, sort=True).ngroup().to_numpy()
R = S.groupby(KEYS, sort=True).agg(n_samples=("pos", "size"), n_pos=("pos", "sum")).reset_index()
assert (S["rid"].max() + 1) == len(R)
R["y"] = (R["n_pos"] > 0).astype(np.int8)
rd = R["d"].to_numpy().astype("datetime64[D]")
R["split"] = np.where(rd <= TRAIN_END, "train", np.where((rd >= VAL[0]) & (rd <= VAL[1]), "val", "test"))
R["inC4"] = R["p3"].isin(C4)
REP["rows"] = {sp: {"rows": int((R["split"] == sp).sum()), "pos": int(R.loc[R["split"] == sp, "y"].sum())} for sp in ("train", "val", "test")}
REP["rows"]["total"] = len(R)
REP["C4_rows"] = {sp: {"rows": int(((R["split"] == sp) & R["inC4"]).sum()), "pos": int(R.loc[(R["split"] == sp) & R["inC4"], "y"].sum())}
                  for sp in ("train", "val", "test")}
out = {}
for sp in ("val", "test"):
    m = R["split"] == sp
    ms = S["split"] == sp
    out[sp] = {"rows": int(m.sum()), "rows_outside": int((m & ~R["inC4"]).sum()),
               "pos_rows": int(R.loc[m, "y"].sum()), "pos_rows_outside": int(R.loc[m & ~R["inC4"], "y"].sum()),
               "samples": int(ms.sum()), "samples_outside": int((ms & ~S["p3"].isin(C4)).sum()),
               "pos_samples": int(S.loc[ms, "pos"].sum()), "pos_samples_outside": int(S.loc[ms & ~S["p3"].isin(C4), "pos"].sum()),
               "pos_rows_outside_codes": R.loc[m & ~R["inC4"] & (R["y"] == 1), "p3"].value_counts().to_dict()}
    out[sp]["rows_outside_rate"] = out[sp]["rows_outside"] / out[sp]["rows"]
REP["C4_outside"] = out
log("rows", REP["rows"], "C4 rows", REP["C4_rows"], "outside", out)

# ---------------------------------------------------------------- 뭉치 플래그
nbt = pq.read_table(NBS).to_pandas()
defs = pd.read_csv(NBDEF, encoding="utf-8-sig")
CL = defs["칼럼명"].astype(str).tolist()
nbcols = [c for c in nbt.columns if c.startswith("nb_")]
REP["nb_file"] = {"rows": len(nbt), "cols_match_def_order": nbcols == CL, "n_clusters": len(CL),
                  "key_same_order": bool(np.array_equal(nbt["lims_data_key"].astype(str).to_numpy(), S["key"].to_numpy())),
                  "dtypes": sorted(set(str(nbt[c].dtype) for c in nbcols)),
                  "values": sorted(set(np.unique(nbt[nbcols].to_numpy()).tolist())),
                  "selection_period_text": defs["선정 기간"].astype(str).unique().tolist(),
                  "def_has_val_stats_cols": [c for c in defs.columns if "검증" in c]}
FL = nbt[CL].to_numpy().astype(np.int8)
log("nb_file", REP["nb_file"])

# ---------------------------------------------------------------- N2: 단위x날짜 OR (pandas groupby max)
n2 = pd.DataFrame(FL, columns=CL).groupby(S["rid"].to_numpy()).max()
assert len(n2) == len(R) and (n2.index.to_numpy() == np.arange(len(R))).all()
N2 = n2.to_numpy().astype(np.int8)
np.save(OUT + r"\my_n2.npy", N2)
REP["N2"] = {"rows": len(N2), "rows_with_any": int(N2.any(1).sum()), "col_sums": N2.sum(0).astype(int).tolist()}

# ---------------------------------------------------------------- N0: tfidf 결합
pf = pq.ParquetFile(TFIDF)
names = pf.schema_arrow.names
tcols = [c for c in names if c.startswith("tf_")]
kt = pq.read_table(TFIDF, columns=["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d", "n_rows"]).to_pandas()
TK = pd.DataFrame({"sgg": kt["sgg_code"].astype(str).to_numpy(dtype=object), "p2": fill(kt["train_prdlst_code"]),
                   "p3": fill(kt["train_prdlst_code_2"]), "d": pd.to_datetime(kt["d"]).to_numpy().astype("datetime64[D]")})
TK["ti"] = np.arange(len(TK))
REP["N0_keys"] = {"rows": len(TK), "p2_null": int(kt["train_prdlst_code"].isna().sum()), "p3_null": int(kt["train_prdlst_code_2"].isna().sum()),
                  "sgg_null": int(kt["sgg_code"].isna().sum()), "dup_keys": int(TK.duplicated(KEYS).sum()),
                  "n_tf_cols": len(tcols), "other_cols": [c for c in names if c not in tcols]}
Rk = R[KEYS].copy()
Rk["rid"] = np.arange(len(R))
mg = Rk.merge(TK, on=KEYS, how="outer", indicator=True)
REP["N0_keys"]["merge"] = mg["_merge"].value_counts().to_dict()
mg = mg[mg["_merge"] == "both"]
mg = mg.sort_values("rid")
assert len(mg) == len(R) and (mg["rid"].to_numpy() == np.arange(len(R))).all()
ti_of_row = mg["ti"].to_numpy().astype(np.int64)
REP["N0_keys"]["n_rows_eq_n_samples"] = bool(np.array_equal(kt["n_rows"].to_numpy()[ti_of_row], R["n_samples"].to_numpy()))
null_rows = np.flatnonzero(kt["train_prdlst_code"].isna().to_numpy()[ti_of_row])     # 내 행 번호(2단계 null 이던 tfidf 행)
REP["N0_keys"]["null_p2_rows_in_my_rows"] = len(null_rows)
REP["N0_keys"]["null_p2_rows_p2_value"] = sorted(set(R["p2"].to_numpy()[null_rows]))
voc = pd.read_csv(VOCAB, encoding="utf-8-sig")
REP["N0_keys"]["vocab_cols_same_order"] = voc["열이름"].astype(str).tolist() == tcols
X0 = np.empty((len(R), len(tcols)), dtype=np.float32)
for a in range(0, len(tcols), 100):
    cs = tcols[a:a + 100]
    tb = pq.read_table(TFIDF, columns=cs)
    for j, c in enumerate(cs):
        col = tb.column(c).to_numpy(zero_copy_only=False)
        assert col.dtype == np.float32
        X0[:, a + j] = col[ti_of_row]
    del tb
nb = np.ascontiguousarray(X0).view(np.uint8).reshape(len(R), -1)
h0 = np.array([int.from_bytes(hashlib.blake2b(nb[i].tobytes(), digest_size=8).digest(), "little") for i in range(len(R))], dtype=np.uint64)
np.save(OUT + r"\my_n0_rowhash.npy", h0)
# 건수 대리 점검: 행 L2 크기·0 아닌 칸 수와 n_samples
c_norm = np.sqrt((X0[:, :300].astype(np.float64) ** 2).sum(1))
w_norm = np.sqrt((X0[:, 300:].astype(np.float64) ** 2).sum(1))
nnz = (X0 != 0).sum(1)
ns = R["n_samples"].to_numpy()
y = R["y"].to_numpy()
n2cnt = N2.sum(1)


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


REP["count_proxy"] = {
    "n0_char_norm_quantiles": np.quantile(c_norm, [0, .01, .5, .99, 1]).round(6).tolist(),
    "n0_word_norm_quantiles": np.quantile(w_norm, [0, .01, .5, .99, 1]).round(6).tolist(),
    "n0_nnz_quantiles": np.quantile(nnz, [0, .5, .9, .99, 1]).tolist(),
    "spearman_nnz_vs_n_samples": spearman(nnz, ns),
    "spearman_n2count_vs_n_samples": spearman(n2cnt, ns),
    "spearman_n_samples_vs_y": spearman(ns, y),
    "y_rate_by_n_samples": {str(k): [int(v[0]), float(v[1])] for k, v in
                            pd.DataFrame({"b": np.minimum(ns, 6), "y": y}).groupby("b")["y"].agg(["size", "mean"]).iterrows()},
    "y_rate_by_n2count": {str(k): [int(v[0]), float(v[1])] for k, v in
                          pd.DataFrame({"b": np.minimum(n2cnt, 4), "y": y}).groupby("b")["y"].agg(["size", "mean"]).iterrows()},
}
# TF-IDF idf 가 어느 문서 집합에서 나왔나: 문서 = 단위x날짜 행이라 가정, df = 0 아닌 행 수, sklearn smooth idf
trm = (R["split"] == "train").to_numpy()
res = {}
for lab, m in (("all_rows", np.ones(len(R), bool)), ("train_rows", trm)):
    df = (X0[m] != 0).sum(0).astype(np.float64)
    n = m.sum()
    idf = np.log((1 + n) / (1 + df)) + 1
    diff = np.abs(idf - voc["idf"].to_numpy())
    res[lab] = {"n_docs": int(n), "max_abs_diff": float(diff.max()), "median_abs_diff": float(np.median(diff)),
                "n_within_1e-4": int((diff < 1e-4).sum())}
REP["idf_test"] = res
log("N0", REP["N0_keys"], REP["count_proxy"], res)

# ---------------------------------------------------------------- 쌍 선택
rng = np.random.default_rng(20261019)
base = S["d"].min()
S["day"] = (S["d"].to_numpy().astype("datetime64[D]") - np.datetime64(base, "D")).astype(np.int64)
R["day"] = (rd - np.datetime64(base, "D")).astype(np.int64)
ukey = (R["sgg"] + "|" + R["p2"] + "|" + R["p3"]).to_numpy()
skey = (S["sgg"] + "|" + S["p2"] + "|" + S["p3"]).to_numpy()
u_codes, u_uni = pd.factorize(np.concatenate([ukey, skey]), sort=True)
R["unit"] = u_codes[:len(R)]
S["unit"] = u_codes[len(R):]
rows_sp = {sp: np.flatnonzero((R["split"] == sp).to_numpy()) for sp in ("train", "val", "test")}
pairs = []


def add(rows, hs, cat):
    for r, h in zip(rows, hs):
        pairs.append((int(r), int(h), cat))


add(rng.choice(rows_sp["train"], 120, replace=False), rng.integers(1, 61, 120), "train_random")
early = np.flatnonzero((R["day"].to_numpy() <= 730) & (R["split"] == "train").to_numpy())
add(rng.choice(early, 60, replace=False), rng.integers(1, 61, 60), "train_early_2y")
earliest = np.argsort(R["day"].to_numpy(), kind="mergesort")[:15]
add(earliest, rng.integers(1, 61, 15), "train_earliest")
add(rng.choice(np.arange(len(R)), 60, replace=False), np.full(60, 60), "h60_any")
add(rng.choice(rows_sp["train"], 20, replace=False), np.full(20, 60), "h60_train")
for sp in ("train", "val", "test"):
    p = np.flatnonzero((R["y"] == 1).to_numpy() & (R["split"] == sp).to_numpy())
    add(rng.choice(p, 30, replace=False), rng.integers(1, 61, 30), f"pos_{sp}")
add(rng.choice(rows_sp["val"], 40, replace=False), rng.integers(1, 61, 40), "val_random")
add(rng.choice(rows_sp["test"], 40, replace=False), rng.integers(1, 61, 40), "test_random")
add(null_rows, rng.integers(1, 61, len(null_rows)), "p2_null_code")
# T 당일에 같은 단위 시료가 있는 쌍(당일 제외가 실제로 효력 있는지)
udays = S.groupby("unit")["day"].apply(lambda x: np.unique(x.to_numpy()))
cand = rng.permutation(np.arange(len(R)))
got = 0
for r in cand:
    u, dday = R["unit"].iat[r], R["day"].iat[r]
    ds = udays.loc[u]
    hs = dday - ds
    hs = hs[(hs >= 1) & (hs <= 60)]
    if len(hs):
        pairs.append((int(r), int(rng.choice(hs)), "T_day_sample_in_unit"))
        got += 1
        if got >= 60:
            break
PR = pd.DataFrame(pairs, columns=["rid", "h", "cat"])
PR["sgg"] = R["sgg"].to_numpy()[PR["rid"]]
PR["p2"] = R["p2"].to_numpy()[PR["rid"]]
PR["p3"] = R["p3"].to_numpy()[PR["rid"]]
PR["d"] = rd[PR["rid"]]
PR["split"] = R["split"].to_numpy()[PR["rid"]]
PR["y"] = R["y"].to_numpy()[PR["rid"]]
PR.to_parquet(OUT + r"\pairs.parquet", index=False)
REP["pairs"] = {"n": len(PR), "by_cat": PR["cat"].value_counts().to_dict(), "by_split": PR["split"].value_counts().to_dict(),
                "pos": int(PR["y"].sum()), "h60": int((PR["h"] == 60).sum())}
log("pairs", REP["pairs"])

# ---------------------------------------------------------------- N1 반복문(쌍마다 원시 마스크)
sday = S["day"].to_numpy()
spos = S["pos"].to_numpy().astype(np.int64)
FLi = FL.astype(np.int64)
FP = FLi * spos[:, None]
unit_idx = S.groupby("unit").indices
TRAIN_DAY = int((TRAIN_END - np.datetime64(base, "D")).astype(np.int64))
DAYF = pd.DataFrame(FLi).groupby(sday).sum().reindex(range(int(sday.max()) + 1), fill_value=0).to_numpy()
DAYP = pd.DataFrame(FP).groupby(sday).sum().reindex(range(int(sday.max()) + 1), fill_value=0).to_numpy()
# 날짜별 표 점검: 시료 단위 원시 마스크 합과 몇 날짜 대조
for _dchk in (0, 100, 2000, int(sday.max())):
    _m = sday == _dchk
    assert np.array_equal(DAYF[_dchk], FLi[_m].sum(0)) and np.array_equal(DAYP[_dchk], FP[_m].sum(0))


def feat(n, nc, sc, Nc, Sc):
    share = nc / n if n > 0 else np.full(len(nc), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = np.where(Nc > 0, Sc / np.where(Nc > 0, Nc, 1), np.nan)
    rate = (sc + K * p0) / (nc + K)
    return share, rate


def window_feats(r, h, mode="excl", p0mode="window"):
    u = R["unit"].iat[r]
    T = R["day"].iat[r] - h
    ii = unit_idx[u]
    parts = []
    for P, _ in PERIODS:
        if mode == "excl":
            lo, hi = T - P, T - 1
        elif mode == "incl_shift":
            lo, hi = T - P + 1, T
        elif mode == "incl_extra":
            lo, hi = T - P, T
        m = (sday[ii] >= lo) & (sday[ii] <= hi)
        n = float(m.sum())
        nc = FLi[ii][m].sum(0).astype(np.float64)
        sc = FP[ii][m].sum(0).astype(np.float64)
        # 전국: 날짜별 표(pandas groupby 합)를 창 안 날짜만 그대로 더함(누적합 아님)
        if p0mode == "window":
            a, b = max(lo, 0), max(hi + 1, 0)
        elif p0mode == "allperiod":
            a, b = 0, len(DAYF)
        elif p0mode == "train_only":
            a, b = 0, TRAIN_DAY + 1
        Nc = DAYF[a:b].sum(0).astype(np.float64)
        Sc = DAYP[a:b].sum(0).astype(np.float64)
        sh, ra = feat(n, nc, sc, Nc, Sc)
        parts += [sh, ra]
    return np.concatenate(parts)


t1 = time.time()
MY = np.stack([window_feats(r, h) for r, h in zip(PR["rid"], PR["h"])])
ALT = {}
for nm, kw in (("incl_shift", dict(mode="incl_shift")), ("incl_extra", dict(mode="incl_extra")),
               ("p0_allperiod", dict(p0mode="allperiod")), ("p0_train_only", dict(p0mode="train_only"))):
    ALT[nm] = np.stack([window_feats(r, h, **kw) for r, h in zip(PR["rid"], PR["h"])])
np.save(OUT + r"\my_n1_pairs.npy", MY)
for k, v in ALT.items():
    np.save(OUT + rf"\my_n1_pairs_{k}.npy", v)
N1NAMES = [f"{kind}{pl}_{c[3:]}" for _, pl in PERIODS for kind in ("share", "rate") for c in CL]
json.dump(N1NAMES, open(OUT + r"\my_n1_names.json", "w", encoding="utf-8"), ensure_ascii=False)
REP["n1_pairs"] = {"seconds": time.time() - t1, "nan_cells": int(np.isnan(MY).sum()), "cells": int(MY.size),
                   "pairs_with_any_nan": int(np.isnan(MY).any(1).sum()),
                   "pairs_unit_window_empty_5y": int(np.isnan(MY[:, 0]).sum()),
                   "pairs_unit_window_empty_1y": int(np.isnan(MY[:, 42]).sum())}
log("n1 pairs", REP["n1_pairs"])

# ---------------------------------------------------------------- N1 전체 행(h=1, h=60) 벡터 계산(누적합, 쌍 계산과 다른 방식)
t2 = time.time()
Dh = int(sday.max()) + 1
order = np.lexsort((sday, S["unit"].to_numpy()))
uo = S["unit"].to_numpy()[order].astype(np.int64)
do = sday[order]
comp = uo * (Dh + 2) + do
cumN = np.concatenate([[0], np.cumsum(np.ones(len(order), np.int64))])
cumF = np.vstack([np.zeros((1, len(CL)), np.int64), np.cumsum(FLi[order], 0)])
cumP = np.vstack([np.zeros((1, len(CL)), np.int64), np.cumsum(FP[order], 0)])
natF = np.zeros((Dh + 1, len(CL)), np.int64)
natP = np.zeros((Dh + 1, len(CL)), np.int64)
dayF = pd.DataFrame(FLi).groupby(sday).sum().reindex(range(Dh), fill_value=0).to_numpy()
dayP = pd.DataFrame(FP).groupby(sday).sum().reindex(range(Dh), fill_value=0).to_numpy()
natF[1:] = np.cumsum(dayF, 0)
natP[1:] = np.cumsum(dayP, 0)
ru = R["unit"].to_numpy().astype(np.int64)
rday = R["day"].to_numpy().astype(np.int64)


def full_n1(h):
    T = rday - h
    parts = []
    for P, _ in PERIODS:
        lo = np.maximum(T - P, 0)
        hi = np.maximum(T, 0)          # [lo, hi) = 날짜 lo..hi-1
        a = np.searchsorted(comp, ru * (Dh + 2) + lo, "left")
        b = np.searchsorted(comp, ru * (Dh + 2) + hi, "left")
        n = (cumN[b] - cumN[a]).astype(np.float64)
        nc = (cumF[b] - cumF[a]).astype(np.float64)
        sc = (cumP[b] - cumP[a]).astype(np.float64)
        lo2, hi2 = np.clip(lo, 0, Dh), np.clip(hi, 0, Dh)
        Nc = (natF[hi2] - natF[lo2]).astype(np.float64)
        Sc = (natP[hi2] - natP[lo2]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(n[:, None] > 0, nc / np.where(n > 0, n, 1)[:, None], np.nan)
            p0 = np.where(Nc > 0, Sc / np.where(Nc > 0, Nc, 1), np.nan)
            rate = (sc + K * p0) / (nc + K)
        parts += [share, rate]
    return np.hstack(parts)


for h in (1, 60):
    np.save(OUT + rf"\my_n1_full_h{h}.npy", full_n1(h).astype(np.float32))
# 벡터 계산 = 반복문 계산(쌍 중 h=1·60 이 아닌 것도 같은 함수로 확인)
# 쌍 전부를 한 번에: h 배열 버전
def full_n1_rows(rows, hs):
    global rday, ru
    sv_rd, sv_ru = rday, ru
    rday, ru = sv_rd[rows], sv_ru[rows]
    try:
        return full_n1(np.asarray(hs, np.int64))
    finally:
        rday, ru = sv_rd, sv_ru


V = full_n1_rows(PR["rid"].to_numpy(), PR["h"].to_numpy())
REP["vector_vs_loop"] = {"nan_pattern_same": bool(np.array_equal(np.isnan(V), np.isnan(MY))),
                         "max_abs_diff": float(np.nanmax(np.abs(V - MY))), "bit_equal_f64": bool(np.array_equal(V, MY, equal_nan=True))}
REP["n1_full_seconds"] = time.time() - t2
R[["sgg", "p2", "p3", "d", "n_samples", "n_pos", "y", "split", "inC4", "unit", "day"]].to_parquet(OUT + r"\my_rows.parquet", index=False)
REP["seconds"] = time.time() - t00
json.dump(REP, open(OUT + r"\v1_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log("끝", REP["vector_vs_loop"], REP["seconds"])
