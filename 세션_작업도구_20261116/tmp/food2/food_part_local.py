# 분석용(결과 파일 안 씀): s0_6_static 의 식품 부분을 실제 입력으로 메모리에서 — 출처 세기·이전 대비·결측 목록 대조
import sys, json
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts")
import pandas as pd
import s0_6_static as m
from lst import paths as ph
ROOT = r"C:\Users\NT751\Desktop\천영진"
M = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror"
cfg = m.load_cfg(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\configs\s0_6_static_food2.json")
P = {k: ph.resolve(ROOT, v) for k, v in cfg["inputs"].items() if k in ("prdlst_cl", "aw_cl", "prev_food_table")}
samples = pd.read_parquet(M + r"\s0\0-1\apply\samples.parquet", columns=["lims_data_key", "y", "src_route", "sgg", "p2", "p3", "d", "h"])
chk = {}
prev = pd.read_parquet(M + r"\s0\0-6_static\식품_고정표.parquet")
t = m.run_food(P, samples, chk, cfg, prev)
print(m.source_counts(t).pivot(index="값 출처", columns="항목", values="3단계 코드 수").to_string())
f = chk["FD1"]
print("표에 코드 없음", len(f["codes_not_in_t_aw_cl"]), f["codes_not_in_t_aw_cl"])
print("자기값 aw_source", json.dumps(f["self_value_aw_source"], ensure_ascii=False))
print("parse", json.dumps(f["parse_rules"], ensure_ascii=False)); print("unparsed", f["unparsed"])
print("vs prev static", json.dumps(f["vs_prev_static"], ensure_ascii=False))
print("aw_label codes", f["aw_label_codes"])
print("missing rows food_aw", f["missing_rows"]["food_aw"], "food_ph", f["missing_rows"]["food_ph"]["nan"], "food_na", f["missing_rows"]["food_na"]["nan"])
lists = json.load(open(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\food2\missing_lists_server.json", encoding="utf-8"))
r = m.missing_list_check(t, samples, lists, "2025-06-30")
print("missing check same:", r["same"], r["changed"], {k: v.get("nan_rows_d_le_cut") for k, v in r["variables"].items() if "nan_rows_d_le_cut" in v})
cl = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "kor_nm"])
nm = dict(zip(cl.prdlst_cd, cl.kor_nm))
miss = t.loc[t["food_aw"].isna(), ["p3", "aw_rule", "ph_rule", "na_rule"]]
cnt = samples["p3"].value_counts()
miss = miss.assign(rows=miss["p3"].map(cnt).fillna(0).astype(int), name=miss["p3"].map(nm)).sort_values("rows", ascending=False)
print(miss.head(12).to_string(index=False)); print("코드없음 rows", int(cnt.get("코드없음", 0)), "합", int(miss["rows"].sum()))
