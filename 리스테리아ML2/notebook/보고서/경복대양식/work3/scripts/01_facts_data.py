# -*- coding: utf-8 -*-
"""3차 연구 보고서 'I. 데이터 전처리 부문' 사실 모으기 (2026-10-22).

읽기만 한다(원자료·데이터셋·실험 코드). 쓰는 곳은 work3/out_facts/ 뿐.
- 1차 실험 코드(data.py·features.py)를 import 해서 단위x날짜 행·기상 요약 칸을 실험과 똑같이 다시 만든다
  (sys.dont_write_bytecode 로 __pycache__ 를 남기지 않음).
- 3차 규칙(cfg3.py BOUNDS·cv.py CV._h_assign·_t_rows)은 여기서 같은 식으로 다시 계산해 서버 로그 값과 맞춰 본다.
실행: <.venv-exp2 python> -B 01_facts_data.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

T0 = time.time()
BASE = Path(r"C:\Users\NT751\Desktop\천영진")
ML2 = BASE / "리스테리아ML2"
RAW_SAMPLE = BASE / "리스테리아ML" / "data" / "listeria_data.parquet"
DS = ML2 / "data" / "데이터셋"
SAMPLE = DS / "시료_시군구.parquet"
COORD = DS / "시군구_대표좌표.parquet"
WX = DS / "식정원_db_기상기후" / "시군구_일별_기상.parquet"
WX_CHECK = ML2 / "data" / "중간" / "시군구_기상_점검.csv"
DROP_SUMMARY = ML2 / "data" / "중간" / "시군구_미배정_삭제행_요약.csv"
EXP1_SRC = ML2 / "exp" / "1_베이스라인_lims기상" / "src"
SEL_WX = ML2 / "exp" / "3_시간순교차검증" / "results" / "selection_wx.json"
OUT = ML2 / "notebook" / "보고서" / "경복대양식" / "work3" / "out_facts"
OUT.mkdir(parents=True, exist_ok=True)

# 3차 날짜 경계(cfg3.compute_bounds 와 같은 값 — 서버 로그 '경계' 줄과 대조)
DATA_END = np.datetime64("2026-06-30")
LEARN_END = np.datetime64("2025-06-30")
FOLDS = [(1, "2022-06-30", "2022-07-01", "2023-05-01"),
         (2, "2023-06-30", "2023-07-01", "2024-05-01"),
         (3, "2024-06-30", "2024-07-01", "2025-05-01")]
TEST = ("2025-07-01", "2026-05-01")
H_MAX = 60
SEED = 42
L_BASE = 80

F = {}          # 모든 사실(JSON)


def save_csv(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def d64(x):
    return np.asarray(x).astype("datetime64[D]")


# ---------------------------------------------------------------- 1. 원 시료 파일(국내·해외 전체)
raw = pq.read_table(RAW_SAMPLE, columns=["src_route", "judge_result", "mnfctur_de", "lims_data_key"])
raw_df = pd.DataFrame({"route": raw["src_route"].to_pandas().astype(str),
                       "judge": raw["judge_result"].to_pandas().astype(str),
                       "d": d64(raw["mnfctur_de"].to_numpy())})
pf_raw = pq.ParquetFile(RAW_SAMPLE)
F["raw"] = {
    "file": str(RAW_SAMPLE), "rows": int(pf_raw.metadata.num_rows), "cols": int(pf_raw.metadata.num_columns),
    "columns": pf_raw.schema_arrow.names,
    "lims_data_key_unique": int(pc.count_distinct(raw["lims_data_key"]).as_py()),
    "d_min": str(raw_df["d"].min())[:10], "d_max": str(raw_df["d"].max())[:10],
    "by_route": {r: {"rows": int((g := raw_df[raw_df.route == r]).shape[0]),
                     "det": int((g.judge == "검출").sum()),
                     "d_min": str(g["d"].min())[:10], "d_max": str(g["d"].max())[:10]}
                 for r in sorted(raw_df.route.unique())},
    "judge_values": raw_df["judge"].value_counts().to_dict(),
}
print("[1] 원 시료", F["raw"]["rows"], F["raw"]["by_route"])

# 삭제 행(시군구 못 정함) 요약
drop = pd.read_csv(DROP_SUMMARY, encoding="utf-8-sig")
F["sgg_drop"] = {"rows": int(drop["행수"].sum()), "det": int(drop["검출수"].sum()),
                 "by_reason": drop.groupby("사유코드")[["행수", "검출수"]].sum().astype(int).to_dict("index")}

# ---------------------------------------------------------------- 2. 시료_시군구.parquet (3차 실험 입력)
t = pq.read_table(SAMPLE)
pf = pq.ParquetFile(SAMPLE)
F["sample"] = {"file": str(SAMPLE), "rows": t.num_rows, "cols": t.num_columns, "columns": t.column_names}
nulls = []
for c in t.column_names:
    col = t.column(c)
    n_null = int(col.null_count)
    n_empty = None
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type) or pa.types.is_dictionary(col.type):
        s = pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col
        n_empty = int(pc.sum(pc.cast(pc.equal(pc.utf8_trim_whitespace(s), ""), pa.int64())).as_py() or 0)
    col_u = pc.cast(col, col.type.value_type) if pa.types.is_dictionary(col.type) else col
    n_unique = int(pc.count_distinct(col_u).as_py())
    nulls.append({"칼럼": c, "자료형": str(col.type), "빈 값(null)": n_null, "빈 글자": n_empty, "서로 다른 값 수": n_unique})
null_df = pd.DataFrame(nulls)
save_csv(null_df, "sample_columns_nulls.csv")
F["sample"]["nulls"] = {r["칼럼"]: {"null": r["빈 값(null)"], "empty": r["빈 글자"], "unique": r["서로 다른 값 수"]} for r in nulls}

S = t.select(["judge_result", "mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_nm", "train_prdlst_code_2",
              "train_prdlst_nm_2", "date_basis", "addr_source", "data_source", "hazard_label", "expr_iem_nm", "src_route",
              "inspct_instt_code"]).to_pandas()
for c in S.columns:
    if c != "mnfctur_de":
        S[c] = S[c].astype(object)
S["pos"] = (S["judge_result"] == "검출").astype(int)
S["d"] = d64(S["mnfctur_de"].to_numpy())
F["sample"]["det"] = int(S["pos"].sum())
F["sample"]["d_min"], F["sample"]["d_max"] = str(S["d"].min())[:10], str(S["d"].max())[:10]
F["sample"]["d_has_time"] = int((pd.to_datetime(S["mnfctur_de"]) != pd.to_datetime(S["mnfctur_de"]).dt.normalize()).sum())


def vc(col):
    g = S.groupby(col, dropna=False)["pos"].agg(["size", "sum"]).reset_index()
    g.columns = [col, "rows", "det"]
    g[col] = g[col].astype(str)
    return {r[col]: {"rows": int(r["rows"]), "det": int(r["det"])} for _, r in g.iterrows()}


for c in ["judge_result", "src_route", "hazard_label", "data_source", "expr_iem_nm", "date_basis", "addr_source"]:
    F["sample"][f"vc_{c}"] = vc(c)
F["sample"]["n_inspct_instt"] = int(S["inspct_instt_code"].nunique())
F["sample"]["n_sgg"] = int(S["sgg_code"].nunique())
F["sample"]["n_sido"] = int(S["sgg_code"].str[:2].nunique())
F["sample"]["n_p2"] = int(S["train_prdlst_code"].nunique())
F["sample"]["n_p3"] = int(S["train_prdlst_code_2"].nunique())
F["sample"]["p2_null_rows"] = int(S["train_prdlst_code"].isna().sum())
F["sample"]["p3_null_rows"] = int(S["train_prdlst_code_2"].isna().sum())
F["sample"]["p2_or_p3_null_rows"] = int((S["train_prdlst_code"].isna() | S["train_prdlst_code_2"].isna()).sum())
F["sample"]["p2_or_p3_null_det"] = int(S.loc[S["train_prdlst_code"].isna() | S["train_prdlst_code_2"].isna(), "pos"].sum())
F["sample"]["null_code_names"] = (S.loc[S["train_prdlst_code_2"].isna(), ["train_prdlst_nm", "train_prdlst_nm_2"]]
                                  .astype(str).value_counts(dropna=False).head(10).reset_index().astype(str).values.tolist())
# 이름 쌍과 코드의 1:1 여부(보고 표에 이름을 쓰기 위함)
pair = S[["train_prdlst_code_2", "train_prdlst_nm_2"]].astype(str).drop_duplicates()
F["sample"]["p3_code_name_pairs"] = int(len(pair))
F["sample"]["p3_code_multi_name"] = int(pair.groupby("train_prdlst_code_2").size().gt(1).sum())
pair2 = S[["train_prdlst_code", "train_prdlst_nm"]].astype(str).drop_duplicates()
F["sample"]["p2_code_name_pairs"] = int(len(pair2))
F["sample"]["p2_code_multi_name"] = int(pair2.groupby("train_prdlst_code").size().gt(1).sum())
F["sample"]["p3_equals_p2_rows"] = int((S["train_prdlst_code"].astype(str) == S["train_prdlst_code_2"].astype(str)).sum())
# 시료 기준 연도별
S["year"] = pd.to_datetime(S["d"]).dt.year
yr = S.groupby("year")["pos"].agg(["size", "sum"]).reset_index().rename(columns={"size": "검사", "sum": "검출"})
yr["검출률(%)"] = (yr["검출"] / yr["검사"] * 100).round(4)
save_csv(yr, "sample_by_year.csv")
F["sample"]["by_year"] = yr.to_dict("records")
# 기간 구간별(교차검증 해) 시료 검출
bins = [("~2022-06-30", None, "2022-06-30"), ("2022-07-01~2023-06-30", "2022-07-01", "2023-06-30"),
        ("2023-07-01~2024-06-30", "2023-07-01", "2024-06-30"), ("2024-07-01~2025-06-30", "2024-07-01", "2025-06-30"),
        ("2025-07-01~2026-06-30", "2025-07-01", "2026-06-30")]
per = []
for nm, a, b in bins:
    m = (S["d"] <= np.datetime64(b)) & ((S["d"] >= np.datetime64(a)) if a else True)
    per.append({"구간(기준일 d)": nm, "시료": int(m.sum()), "검출 시료": int(S.loc[m, "pos"].sum())})
F["sample"]["by_period"] = per
print("[2] 시료_시군구", F["sample"]["rows"], "검출", F["sample"]["det"], F["sample"]["by_period"])

# ---------------------------------------------------------------- 3. 단위 x 날짜 행 (1차 data.py 그대로)
sys.path.insert(0, str(EXP1_SRC))
import config as C1  # noqa: E402
import data as D1  # noqa: E402
import features as F1  # noqa: E402

assert Path(C1.SAMPLE_PATH) == SAMPLE and Path(C1.WEATHER_PATH) == WX, (C1.SAMPLE_PATH, C1.WEATHER_PATH)
samples = D1.load_samples()
rows = D1.build_unit_rows(samples)
levels = D1.category_levels(rows)
rd = d64(rows["d"].to_numpy())
y = rows["y"].to_numpy().astype(np.int64)
ns = rows["n_samples"].to_numpy()
npos = rows["n_pos_samples"].to_numpy()
unit_key = (rows["sgg_code"].astype(str) + "|" + rows["train_prdlst_code"].astype(str) + "|" + rows["train_prdlst_code_2"].astype(str)).to_numpy()
learn = rd <= LEARN_END
testp = rd > LEARN_END
F["rows"] = {
    "rows": int(len(rows)), "pos_rows": int(y.sum()), "samples_sum": int(ns.sum()), "pos_samples_sum": int(npos.sum()),
    "units": int(len(np.unique(unit_key))), "d_min": str(rd.min()), "d_max": str(rd.max()),
    "n_samples_per_row": {"mean": float(ns.mean()), "median": float(np.median(ns)), "max": int(ns.max()),
                          "rows_with_1": int((ns == 1).sum()), "rows_with_2plus": int((ns >= 2).sum())},
    "pos_rows_with_2plus_pos_samples": int((npos >= 2).sum()), "max_pos_samples_in_row": int(npos.max()),
    "pos_rows_n_samples_mean": float(ns[y == 1].mean()),
    "learn": {"rows": int(learn.sum()), "pos": int(y[learn].sum()), "units": int(len(np.unique(unit_key[learn]))),
              "pos_samples": int(npos[learn].sum())},
    "test_period_d": {"rows": int(testp.sum()), "pos": int(y[testp].sum()), "units": int(len(np.unique(unit_key[testp]))),
                      "pos_samples": int(npos[testp].sum())},
    "null_code_rows": int(((rows["train_prdlst_code"] == C1.NULL_CODE) | (rows["train_prdlst_code_2"] == C1.NULL_CODE)).sum()),
    "null_code_rows_learn": int((((rows["train_prdlst_code"] == C1.NULL_CODE) | (rows["train_prdlst_code_2"] == C1.NULL_CODE)) & learn).sum()),
    "null_code_pos_rows": int(y[((rows["train_prdlst_code"] == C1.NULL_CODE) | (rows["train_prdlst_code_2"] == C1.NULL_CODE)).to_numpy()].sum()),
    "levels_all_rows": {c: len(v) for c, v in levels.items()},
    "levels_learn_rows": {c: int(rows.loc[learn, c].nunique()) for c in C1.CAT_COLS},
}
u_learn = set(unit_key[learn])
new_units = np.array([u not in u_learn for u in unit_key])
F["rows"]["test_period_rows_new_unit"] = int((testp & new_units).sum())
F["rows"]["test_period_pos_rows_new_unit"] = int(y[testp & new_units].sum())
for c in ("sgg", "p3"):
    seen = set(rows.loc[learn, c])
    nm = rows[c].map(lambda v: v not in seen).to_numpy()
    F["rows"][f"test_period_rows_new_{c}"] = int((testp & nm).sum())
print("[3] 단위x날짜", F["rows"]["rows"], F["rows"]["pos_rows"], "학습", F["rows"]["learn"], "시험기간", F["rows"]["test_period_d"])

# 학습 행 h 배정(cv.CV._h_assign 과 같은 식: 행 순서 = 단위 키 + d 정렬 = build_unit_rows 순서)
last_te = np.datetime64(FOLDS[-1][1])
h = np.zeros(len(rows), dtype=np.int16)
ma = rd <= last_te
mb = (rd > last_te) & (rd <= LEARN_END)
h[ma] = np.random.default_rng(SEED).integers(1, H_MAX + 1, size=int(ma.sum()))
h[mb] = np.random.default_rng(SEED + 1).integers(1, H_MAX + 1, size=int(mb.sum()))
hc = np.bincount(h[learn], minlength=H_MAX + 1)[1:]
F["h_assign"] = {"rule": "d <= 2024-06-30 행 = default_rng(42).integers(1, 61), 2024-07-01~2025-06-30 행 = default_rng(43).integers(1, 61), "
                         "행 순서 = 단위 키(sgg_code, 2단계, 3단계) + d 정렬",
                 "rows_rng42": int(ma.sum()), "rows_rng43": int(mb.sum()),
                 "h_mean_learn": float(h[learn].mean()), "rows_per_h_min": int(hc.min()), "rows_per_h_max": int(hc.max()),
                 "pos_rows_h_mean": float(h[learn & (y == 1)].mean())}

# 겹·시험 행 수(cv.CV.train_rows·_t_rows 와 같은 식)
fold_tab = []
for k, te, ts, tt in FOLDS:
    mtr = rd <= np.datetime64(te)
    n_pair = n_pair_pos = 0
    per_h_n = []
    for hh in range(1, H_MAX + 1):
        T = rd - np.timedelta64(hh, "D")
        m = (T >= np.datetime64(ts)) & (T <= np.datetime64(tt)) & (rd <= DATA_END)
        n_pair += int(m.sum())
        n_pair_pos += int(y[m].sum())
        per_h_n.append(int(m.sum()))
    # 맞혀 보는 해(T 범위)에 대응하는 기준일 d 범위의 서로 다른 행
    md = (rd >= np.datetime64(ts) + np.timedelta64(1, "D")) & (rd <= np.datetime64(tt) + np.timedelta64(H_MAX, "D"))
    fold_tab.append({"구분": f"겹 {k}", "학습 행 기준일 d": f"~{te}", "학습 행": int(mtr.sum()), "학습 행 검출": int(y[mtr].sum()),
                     "학습 시료": int(ns[mtr].sum()), "학습 검출 시료": int(npos[mtr].sum()),
                     "예측하는 날 T": f"{ts}~{tt}", "맞혀 보는 (행,h)": n_pair, "맞혀 보는 (행,h) 검출": n_pair_pos,
                     "h 하나당 평균 행": round(n_pair / H_MAX, 2), "h 하나당 평균 검출": round(n_pair_pos / H_MAX, 2),
                     "맞혀 보는 해 기준일 d 범위의 행": int(md.sum()), "그 행 중 검출": int(y[md].sum())})
mtr = learn
n_pair = n_pair_pos = 0
for hh in range(1, H_MAX + 1):
    T = rd - np.timedelta64(hh, "D")
    m = (T >= np.datetime64(TEST[0])) & (T <= np.datetime64(TEST[1])) & (rd <= DATA_END)
    n_pair += int(m.sum())
    n_pair_pos += int(y[m].sum())
md = (rd >= np.datetime64(TEST[0]) + np.timedelta64(1, "D")) & (rd <= np.datetime64(TEST[1]) + np.timedelta64(H_MAX, "D"))
fold_tab.append({"구분": "최종(시험)", "학습 행 기준일 d": f"~{LEARN_END}", "학습 행": int(mtr.sum()), "학습 행 검출": int(y[mtr].sum()),
                 "학습 시료": int(ns[mtr].sum()), "학습 검출 시료": int(npos[mtr].sum()),
                 "예측하는 날 T": f"{TEST[0]}~{TEST[1]}", "맞혀 보는 (행,h)": n_pair, "맞혀 보는 (행,h) 검출": n_pair_pos,
                 "h 하나당 평균 행": round(n_pair / H_MAX, 2), "h 하나당 평균 검출": round(n_pair_pos / H_MAX, 2),
                 "맞혀 보는 해 기준일 d 범위의 행": int(md.sum()), "그 행 중 검출": int(y[md].sum())})
fold_df = pd.DataFrame(fold_tab)
save_csv(fold_df, "folds.csv")
F["folds"] = fold_tab
print("[3b] 겹\n" + fold_df.to_string())

# ---------------------------------------------------------------- 4. 기상표
pw = pq.ParquetFile(WX)
md_w = {k.decode(): v.decode("utf-8") for k, v in (pw.schema_arrow.metadata or {}).items() if k != b"pandas"}
wvars_meta = json.loads(md_w["variables"])
wt = pq.read_table(WX)
wd = d64(wt["date"].to_numpy())
F["weather"] = {"file": str(WX), "rows": wt.num_rows, "n_sgg": int(pc.count_distinct(wt["sgg_code"]).as_py()),
                "date_min": str(wd.min()), "date_max": str(wd.max()), "n_dates": int(len(np.unique(wd))),
                "metadata": {k: v for k, v in md_w.items() if k != "variables"}, "variables": wvars_meta}
chk = pd.read_csv(WX_CHECK, encoding="utf-8-sig")
F["weather"]["check_csv"] = chk.to_dict("records")
# 2015-01-01 뒤에 모든 시군구가 결측인 날
alln = {}
dates_u = np.unique(wd)
for v in [m["var"] for m in wvars_meta]:
    arr = wt[v].to_numpy(zero_copy_only=False)
    isn = np.isnan(arr.astype(np.float64))
    df_ = pd.DataFrame({"d": wd, "n": isn})
    g = df_.groupby("d")["n"].all()
    days = g[g].index.values.astype("datetime64[D]")
    after = days[days >= np.datetime64("2015-01-01")]
    alln[v] = {"all_missing_days": int(len(days)), "first": str(days.min()) if len(days) else None,
               "last_before_2015": str(days[days < np.datetime64("2015-01-01")].max()) if (days < np.datetime64("2015-01-01")).any() else None,
               "after_2015": [str(x) for x in after], "missing_cells": int(isn.sum()), "cells": int(len(isn)),
               "missing_cells_after_2015": int(isn[wd >= np.datetime64("2015-01-01")].sum()),
               "cells_after_2015": int((wd >= np.datetime64("2015-01-01")).sum())}
F["weather"]["all_missing"] = alln
c = pq.read_table(COORD).to_pandas()
F["coord"] = {"rows": int(len(c)), "source_counts": c["source"].value_counts().to_dict(),
              "ongjin": c[c.sgg_code == "28720"].iloc[0][["lat", "lon", "source"]].to_dict(),
              "sido": c.assign(s2=c.sgg_code.str[:2], s=c.sgg_nm.str.split().str[0]).groupby(["s2", "s"]).size().reset_index().values.tolist()}
smp_sgg = set(S["sgg_code"].astype(str))
F["coord"]["sample_sgg_in_coord"] = int(len(smp_sgg & set(c.sgg_code)))
F["coord"]["sample_sgg_not_in_coord"] = sorted(smp_sgg - set(c.sgg_code))
F["coord"]["coord_sgg_without_sample"] = int(len(set(c.sgg_code) - smp_sgg))
print("[4] 기상", F["weather"]["rows"], F["weather"]["n_sgg"], F["weather"]["date_min"], F["weather"]["date_max"])

# ---------------------------------------------------------------- 5. 1차 베이스라인 입력(CatBoost, 기상 요약 80일, 기상 18개)
sel = json.loads(SEL_WX.read_text(encoding="utf-8"))
wvars18 = list(sel["wvars"])
dropped = [x for x in C1.WEATHER_VARS if x not in wvars18]
names19 = F1.feature_names("summary", L_BASE)
per_var = len(names19) // len(C1.WEATHER_VARS)
names18 = [n for n in names19 if any(n == f"{v}_{s}" for v in wvars18
                                     for s in [f"b{k:02d}" for k in range(1, L_BASE // C1.BLOCK + 1)] + ["mean", "std", "min", "max", "slope"])]
F["baseline_input"] = {"model": sel["model"], "input": sel["input"], "L": sel["L"], "wvars": wvars18, "dropped": dropped,
                       "per_var_cols": per_var, "per_var_names_example": [n for n in names19 if n.startswith("avg_temp_")],
                       "weather_cols_19": len(names19), "weather_cols_18": len(names18),
                       "numeric_cols": 1 + len(names18), "cat_cols": list(C1.CAT_COLS), "total_cols": 1 + len(names18) + len(C1.CAT_COLS),
                       "cv_score_start_19": sel["start_score"], "cv_score_final_18": sel["final_score"]}
# 학습 행(최종 재학습 행 = d <= 2025-06-30, 행마다 h 하나) 기상 요약 칸의 결측
W = D1.Weather()
lr = np.flatnonzero(learn)
s_idx = W.s_idx(rows["sgg_code"].to_numpy()[lr])
t_idx = W.d_idx(rd[lr]) - h[lr].astype(np.int64)
W.check_windows(t_idx, L_BASE, "학습 행")
X = F1.build_tree_matrix(W.W, s_idx, t_idx, h[lr], L_BASE, "summary")
cols = ["h"] + names19
assert X.shape[1] == len(cols)
col_idx = {n: i for i, n in enumerate(cols)}
miss_var = []
for v in C1.WEATHER_VARS:
    ci = [col_idx[n] for n in names19 if n.startswith(v + "_") and n[len(v) + 1:] in
          [f"b{k:02d}" for k in range(1, L_BASE // C1.BLOCK + 1)] + ["mean", "std", "min", "max", "slope"]]
    sub = X[:, ci]
    miss_var.append({"변수": v, "칸 수": len(ci), "결측 칸 비율(%)": round(float(np.isnan(sub).mean()) * 100, 4),
                     "결측 칸이 하나라도 있는 행": int(np.isnan(sub).any(1).sum()), "모든 칸 결측 행": int(np.isnan(sub).all(1).sum())})
mv = pd.DataFrame(miss_var)
save_csv(mv, "baseline_learn_rows_weather_missing.csv")
X18 = X[:, [col_idx[n] for n in names18]]
T_dates = rd[lr] - h[lr].astype("timedelta64[D]")
F["baseline_input"]["learn_rows"] = int(len(lr))
F["baseline_input"]["learn_rows_any_nan_18"] = int(np.isnan(X18).any(1).sum())
F["baseline_input"]["learn_nan_cell_share_18"] = float(np.isnan(X18).mean())
F["baseline_input"]["learn_rows_window_before_2015"] = int(((T_dates - np.timedelta64(L_BASE, "D")) < np.datetime64("2015-01-01")).sum())
F["baseline_input"]["learn_pos_rows_window_before_2015"] = int(y[lr][((T_dates - np.timedelta64(L_BASE, "D")) < np.datetime64("2015-01-01"))].sum())
F["baseline_input"]["missing_by_var"] = miss_var
print("[5] 베이스라인 입력", {k: v for k, v in F["baseline_input"].items() if k not in ("missing_by_var", "per_var_names_example")})

F["_meta"] = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - T0, 1), "script": str(Path(__file__).resolve())}
(OUT / "facts.json").write_text(json.dumps(F, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("끝", round(time.time() - T0, 1), "초 ->", OUT / "facts.json")
