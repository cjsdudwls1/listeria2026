# -*- coding: utf-8 -*-
"""3차 보고서 I 부 보충 사실(2026-10-22): 검출이 몰린 단위·품목, 정액,수정란,종란 > 기타(Z2700400000000) 행의 구성,
같은 단위·같은 날 여러 검출, 겹·시험 T 날 수. 읽기만 함. 출력 work3/out_facts/extra.json·extra_top_units.csv
실행: <.venv-exp2 python> -B 03_extra_facts.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
DS = BASE / "data" / "데이터셋"
OUT = BASE / "notebook" / "보고서" / "경복대양식" / "work3" / "out_facts"
t = pq.read_table(DS / "시료_시군구.parquet", columns=["judge_result", "mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_nm",
                                                     "train_prdlst_code_2", "train_prdlst_nm_2", "inspct_instt_code", "prduct_nm",
                                                     "date_basis"]).to_pandas()
for c in t.columns:
    if c != "mnfctur_de":
        t[c] = t[c].astype(object)
t["pos"] = (t["judge_result"] == "검출").astype(int)
t["d"] = pd.to_datetime(t["mnfctur_de"]).dt.normalize()
coord = pq.read_table(DS / "시군구_대표좌표.parquet").to_pandas()
SGG_NM = dict(zip(coord["sgg_code"], coord["sgg_nm"]))
X = {}
P = int(t["pos"].sum())

# 단위(시군구 x 2단계 x 3단계)별 검출
t["p2"] = t["train_prdlst_code"].fillna("코드없음")
t["p3"] = t["train_prdlst_code_2"].fillna("코드없음")
u = t.groupby(["sgg_code", "p2", "p3"]).agg(검사=("pos", "size"), 검출=("pos", "sum"),
                                              품목2=("train_prdlst_nm", "first"), 품목3=("train_prdlst_nm_2", "first")).reset_index()
u["시군구"] = u["sgg_code"].map(SGG_NM)
u = u.sort_values(["검출", "검사"], ascending=[False, False]).reset_index(drop=True)
u["누적 검출 비중(%)"] = u["검출"].cumsum() / P * 100
X["units_total"] = int(len(u))
X["units_with_det"] = int((u["검출"] > 0).sum())
X["units_top_n_for_50pct"] = int((u["누적 검출 비중(%)"] < 50).sum() + 1)
X["top1_unit"] = u.iloc[0][["시군구", "품목2", "품목3", "검사", "검출"]].to_dict()
u.head(20)[["시군구", "sgg_code", "품목2", "품목3", "p3", "검사", "검출", "누적 검출 비중(%)"]].to_csv(
    OUT / "extra_top_units.csv", index=False, encoding="utf-8-sig")

# 정액,수정란,종란 > 기타 (Z2700400000000)
z = t[t["p3"] == "Z2700400000000"]
X["z27"] = {"rows": int(len(z)), "det": int(z["pos"].sum()),
            "top_sgg_det": {f"{SGG_NM.get(k)}({k})": int(v) for k, v in z.groupby("sgg_code")["pos"].sum().sort_values(ascending=False).head(3).items()},
            "top_instt_rows_share": {str(k): round(float(v) * 100, 2) for k, v in z["inspct_instt_code"].value_counts(normalize=True).head(2).items()},
            "top_instt_det_share": {str(k): round(float(v) / max(1, int(z["pos"].sum())) * 100, 2)
                                    for k, v in z.groupby("inspct_instt_code")["pos"].sum().sort_values(ascending=False).head(2).items()},
            "det_prduct_nm_top": z.loc[z["pos"] == 1, "prduct_nm"].astype(str).value_counts().head(8).to_dict(),
            "rows_prduct_nm_has_팽이": int(z["prduct_nm"].astype(str).str.contains("팽이").sum()),
            "det_prduct_nm_has_팽이": int(z.loc[z["pos"] == 1, "prduct_nm"].astype(str).str.contains("팽이").sum())}

# 기타기준규격외 (Z3800100000000)
z38 = t[t["p3"] == "Z3800100000000"]
X["z38"] = {"rows": int(len(z38)), "det": int(z38["pos"].sum()),
            "det_prduct_nm_top": z38.loc[z38["pos"] == 1, "prduct_nm"].astype(str).value_counts().head(8).to_dict()}

# 같은 단위·같은 날 여러 검출
g = t.groupby(["sgg_code", "p2", "p3", "d"])["pos"].agg(["size", "sum"])
pos_rows = g[g["sum"] > 0]
multi = pos_rows[pos_rows["sum"] >= 2]
X["unitday_pos_rows"] = int(len(pos_rows))
X["unitday_multi_pos_rows"] = int(len(multi))
X["det_samples_in_multi_rows"] = int(multi["sum"].sum())
X["det_samples_in_multi_rows_share"] = round(float(multi["sum"].sum()) / P * 100, 2)

# 겹·시험 예측하는 날 T 날 수
for nm, a, b in (("겹1", "2022-07-01", "2023-05-01"), ("겹2", "2023-07-01", "2024-05-01"), ("겹3", "2024-07-01", "2025-05-01"),
                 ("시험", "2025-07-01", "2026-05-01")):
    X[f"T_days_{nm}"] = int((np.datetime64(b) - np.datetime64(a)).astype(int)) + 1

# 기준일 근거(date_basis)별 연도
X["date_basis_by_year_det"] = t[t["pos"] == 1].groupby([t["d"].dt.year, "date_basis"])["pos"].sum().unstack(fill_value=0).to_dict("index")
(OUT / "extra.json").write_text(json.dumps(X, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in X.items() if k != "date_basis_by_year_det"}, ensure_ascii=False, indent=1, default=str))
print(u.head(12)[["시군구", "품목2", "품목3", "검사", "검출", "누적 검출 비중(%)"]].to_string())
