# -*- coding: utf-8 -*-
"""최종 CatBoost(model_final.cbm)의 내장 중요도·나무 갈림 확인(읽기 전용) — 3차 보고서 7장 그림·문장 근거.

1) get_feature_importance()(기본 = PredictionValuesChange)를 칸별로 다시 내어 묶음별 합이 변수중요도_묶음별.csv 와 같은지
2) 나무(127그루, 깊이 4)의 갈림(split)에 쓰인 칸: JSON 으로 내보내 숫자 칸 갈림과 범주(CTR) 갈림의 원래 칸을 셈
   -> 중요도 0 인 묶음(h·시도·시군구·품목 2단계 등)이 갈림에 한 번도 안 쓰였는지
결과: work3/calc/cat_importance_check.json (+ 화면)
실행: <.venv-exp2 python> -B check_cat_importance.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from catboost import CatBoost

sys.stdout.reconfigure(encoding="utf-8")
FC = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\3_시간순교차검증\results\final_cat"
OUT = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\경복대양식\work3\calc\cat_importance_check.json"

m = CatBoost()
m.load_model(os.path.join(FC, "model_final.cbm"))
cols = pd.read_csv(os.path.join(FC, "model_columns.csv"), encoding="utf-8-sig")
bund = pd.read_csv(os.path.join(FC, "변수중요도_묶음별.csv"), encoding="utf-8-sig")
names = list(m.feature_names_)
assert names == list(cols["칸 이름"]), "모델 칸 이름·순서가 model_columns.csv 와 다름"
imp = np.asarray(m.get_feature_importance(), dtype=float)
cols["imp"] = imp
g = cols.groupby("소속 묶음")["imp"].sum()
cmp = bund.set_index("묶음 ID")["중요도 합"]
diff = float(max(abs(g.get(k, 0.0) - v) for k, v in cmp.items()))

tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
tmp.close()
m.save_model(tmp.name, format="json")
js = json.load(open(tmp.name, encoding="utf-8"))
os.remove(tmp.name)
fi = js["features_info"]
float_flat = {f["feature_index"]: f["flat_feature_index"] for f in fi.get("float_features", [])}
cat_flat = {f["feature_index"]: f["flat_feature_index"] for f in fi.get("categorical_features", [])}
trees = js["oblivious_trees"]
used = {}
for t in trees:
    for s in t["splits"]:
        if s["split_type"] == "FloatFeature":
            k = names[float_flat[s["float_feature_index"]]]
            used[k] = used.get(k, 0) + 1
        elif s["split_type"] == "OneHotFeature":
            k = names[cat_flat[s["cat_feature_index"]]]
            used[k] = used.get(k, 0) + 1
ctrs = []
for c in fi.get("ctrs", []):
    el = c.get("elements", [])
    ctrs.append({"ctr_type": c.get("ctr_type"), "elements": el,
                 "cat_names": [names[cat_flat[e["cat_feature_index"]]] for e in el if "cat_feature_index" in e]})
n_ctr_splits = sum(1 for t in trees for s in t["splits"] if s["split_type"] == "OnlineCtr")
for c in ctrs:
    for nm in c["cat_names"]:
        used[nm] = used.get(nm, 0) + n_ctr_splits     # CTR 하나뿐일 때만 정확(아래 assert)
assert len(ctrs) <= 1, "CTR 이 둘 이상이면 갈림별 CTR 대응을 따로 셈"
cols["갈림 수"] = cols["칸 이름"].map(used).fillna(0).astype(int)
gb = cols.groupby(["소속 묶음", "소속 묶음 이름"]).agg(칸수=("칸 이름", "size"), 갈림=("갈림 수", "sum"), 중요도=("imp", "sum"),
                                                쓰인칸=("갈림 수", lambda s: int((s > 0).sum()))).reset_index()
gb = gb.sort_values("중요도", ascending=False)
top5 = gb.head(5)
res = {
    "n_cols": len(names), "n_cat": int((cols["범주 여부"] == "예").sum()), "imp_sum": float(imp.sum()),
    "bundle_vs_csv_maxdiff": diff, "n_trees": len(trees), "depth": len(trees[0]["splits"]),
    "n_splits": sum(len(t["splits"]) for t in trees), "n_ctr_splits": n_ctr_splits, "ctrs": ctrs,
    "n_cols_used_in_splits": int((cols["갈림 수"] > 0).sum()),
    "top5_sum_pct": float(top5["중요도"].sum()), "top5": top5[["소속 묶음 이름", "중요도"]].values.tolist(),
    "bundles": gb.to_dict("records"),
    "zero_imp_bundles": gb[gb["중요도"] == 0][["소속 묶음", "소속 묶음 이름", "칸수", "갈림"]].to_dict("records"),
}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("칸", res["n_cols"], "범주", res["n_cat"], "중요도 합", round(res["imp_sum"], 6), "csv 차이", diff)
print("나무", res["n_trees"], "깊이", res["depth"], "갈림", res["n_splits"], "CTR 갈림", n_ctr_splits, "CTR", ctrs)
print("갈림에 쓰인 칸 수", res["n_cols_used_in_splits"], "/", len(names))
print("상위 5 묶음 합 %.2f%%" % res["top5_sum_pct"])
print(gb.to_string(index=False))
print("저장", OUT)
