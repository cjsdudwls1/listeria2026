# 분석용(결과 파일 안 만듦): t_aw_cl_full vs t_aw_cl 비교 + 식품 고정표 v2(t_aw_cl) vs v3(full) 메모리 비교
import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
M = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror"
full = S._to_plain(pq.read_table(D + r"\추출_20261112_full\t_aw_cl_full.parquet").to_pandas())
cur = S._to_plain(pq.read_table(D + r"\추출_20261112_full\t_aw_cl.parquet").to_pandas())
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "hrnk_prdlst_cd"])
print("full 행", len(full), "고유 코드", full.prdlst_cd.nunique(), "| cur 행", len(cur), "고유", cur.prdlst_cd.nunique())
print("분류표 코드", cl.prdlst_cd.nunique(), "| full 에 없는 분류표 코드", len(set(cl.prdlst_cd) - set(full.prdlst_cd)),
      "| 분류표에 없는 full 코드", len(set(full.prdlst_cd) - set(cl.prdlst_cd)))
print("full lv", dict(collections.Counter(full.lv)))
print("full aw_source", dict(collections.Counter(full.aw_source.fillna("(null)"))))
def has(v): return v.notna() & (v.astype(str).str.strip() != "")
new = full[~full.prdlst_cd.isin(cur.prdlst_cd)]
print("full 에만 있는 행", len(new), "그중 값 있음:", {c: int(has(new[c]).sum()) for c in ("aw", "mc", "ph", "na", "aflatoxin_aw", "aw_source")})
print("full 에만 있는 행 lv", dict(collections.Counter(new.lv)))
for c in ("aw", "mc", "ph", "na", "aflatoxin_aw"):
    print(f"  값 있음 {c}: full {int(has(full[c]).sum())} / cur {int(has(cur[c]).sum())}")
both = sorted(set(full.prdlst_cd) & set(cur.prdlst_cd))
f = full.drop_duplicates("prdlst_cd").set_index("prdlst_cd"); g = cur.drop_duplicates("prdlst_cd").set_index("prdlst_cd")
print("두 표 공통 코드", len(both), "| cur 에만", len(set(cur.prdlst_cd) - set(full.prdlst_cd)))
for c in ("kor_nm", "lv", "aw", "mc", "ph", "na", "aflatoxin_aw", "aw_source"):
    norm = lambda v: v.astype(object).where(v.notna(), "").astype(str).str.strip()
    a = norm(f.loc[both, c]); b = norm(g.loc[both, c])
    d = a != b
    ex = list(zip(np.array(both)[d.to_numpy()][:4], b[d].head(4), a[d].head(4)))
    print(f"  공통 코드 값 다름 {c}: {int(d.sum())}  예(코드, cur, full): {ex}")
# 시료 코드 기준
s = pd.read_parquet(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\awfull\inp\samples5.parquet", columns=["src_route", "p2", "p3", "d", "y"])
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object)
dom = (s["src_route"] == "domestic").to_numpy(); real = p3 != S.NO_CODE
codes3 = sorted(set(p3[real])); p2_of = sf.pair_p2(p3[real], p2[real])
for nm, t in (("cur", g), ("full", f)):
    inn = [k for k in codes3 if k in t.index]
    print(f"시료 3단계 코드 {len(codes3)} 중 {nm} 에 행 있음 {len(inn)} | 바로 값: " +
          ", ".join(f"{c} {int(has(t.loc[inn, c]).sum())}" for c in ("aw", "ph", "na")))
res = {}
for nm, tab in (("v2_cur", cur), ("v3_full", full)):
    ft = sf.build_food_table(cl, tab, codes3, p2_of, aw_source_filter="all")
    t = ft.table; res[nm] = t.set_index("p3")
    print("==", nm, {c: dict(collections.Counter(t[f"{c}_rule"])) for c in sf.FOOD_COLS})
    print("   parse", ft.parse_rules, "unparsed", ft.unparsed)
    cut = pd.Timestamp("2025-06-30"); m = (pd.to_datetime(s["d"]) <= cut).to_numpy()
    ff = sf.food_features(t, p3[m])
    for c in ("food_aw", "food_ph", "food_na"):
        v = ff[c].to_numpy(); print("   d<=cut NaN", c, int(np.isnan(v).sum()), "/", int(m.sum()))
a, b = res["v2_cur"], res["v3_full"]
for c in sf.FOOD_COLS:
    x = a[f"food_{c}"]; y = b.loc[a.index, f"food_{c}"]
    same = (x.isna() & y.isna()) | np.isclose(x.fillna(-9), y.fillna(-9))
    diff = a.index[~same.to_numpy()]
    nrow = int(np.isin(p3, diff).sum()); npos = int((np.isin(p3, diff) & (s["y"].to_numpy() == 1)).sum())
    print(f"v2 대비 v3 값 다른 3단계 코드 {c}: {len(diff)} (시료 {nrow:,}행·검출 {npos})  예: " +
          str([(k, a.loc[k, f'food_{c}'], b.loc[k, f'food_{c}'], a.loc[k, f'{c}_rule'], b.loc[k, f'{c}_rule']) for k in diff[:5]]))
