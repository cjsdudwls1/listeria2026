import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
full = S._to_plain(pq.read_table(D + r"\추출_20261112_full\t_aw_cl_full.parquet").to_pandas())
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet")
nm = dict(zip(cl.prdlst_cd, cl.kor_nm))
s = pd.read_parquet(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\awfull\inp\samples5.parquet")
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object); y = s["y"].to_numpy()
real = p3 != S.NO_CODE
codes3 = sorted(set(p3[real])); p2_of = sf.pair_p2(p3[real], p2[real])
t = sf.build_food_table(cl[["prdlst_cd", "hrnk_prdlst_cd"]], full, codes3, p2_of, aw_source_filter="all").table.set_index("p3")
print("1단계 이름:", {c: nm.get(c) for c in sorted(cl.loc[cl.prdlst_cd.str.endswith("0000000000000"), "prdlst_cd"])[:12]})
f = full.set_index("prdlst_cd")
def blank(v): return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""
for col in ("aw", "ph", "na"):
    rule = pd.Series(p3).map(t[f"{col}_rule"]).to_numpy(dtype=object)
    val = pd.Series(p3).map(t[f"food_{col}"]).to_numpy(dtype=float)
    miss = np.isnan(val)
    why = np.full(len(p3), "", dtype=object)
    why[miss & ~real] = "품목 코드 없음(코드없음)"
    for c in set(p3[miss & real]):
        m = miss & (p3 == c)
        own = f.loc[c, col] if c in f.index else None
        if c not in f.index: r = "표에 그 코드 행 없음"
        elif not blank(own): r = f"원문 해석 불가('{own}')"
        else: r = "원문 빈칸(자기·하위·2단계 모두 값 없음)"
        why[m] = ("Z " if c.startswith("Z") else "Z 아님 ") + r
    print(f"== {col}: 결측 {int(miss.sum()):,} / {len(p3):,}행 ({miss.mean()*100:.1f}%), 검출 {int(y[miss].sum())} / {int(y.sum())}")
    for k, v in collections.Counter(why[miss]).most_common():
        mm = why == k
        print(f"   {k}: {v:,}행 · 검출 {int(y[mm].sum())} · 코드 {len(set(p3[mm]))}개")
    if col == "aw":
        top = collections.Counter(p3[miss & real]).most_common(8)
        print("   큰 결측 코드:", [(c, nm.get(c), n, f.loc[c, 'aw_source'] if c in f.index else None) for c, n in top])
zrows = np.array([str(c).startswith("Z") for c in p3])
print("Z 행", int(zrows.sum()), "검출", int(y[zrows].sum()), "| Z 코드(행 많은 순):", [(c, nm.get(c), n) for c, n in collections.Counter(p3[zrows]).most_common(6)])
print("Z 코드 표 값:", f.loc[[c for c in set(p3[zrows]) if c in f.index], ["kor_nm", "aw", "ph", "na", "aw_source"]].head(10).to_string())
