# 분석용(결과 파일 안 만듦): 새 t_aw_cl 로 식품 고정표를 메모리에서 만들어 출처 세기·결측 확인
import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
M = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror"
s = pd.read_parquet(M + r"\s0\0-1\apply\samples.parquet", columns=["src_route", "p2", "p3", "d", "y"])
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object)
dom = (s["src_route"] == "domestic").to_numpy(); real = p3 != S.NO_CODE
cd = set(p3[dom & real]); co = set(p3[~dom & real]) - cd
codes3 = sorted(cd | co); p2_of = sf.pair_p2(p3[real], p2[real])
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "hrnk_prdlst_cd"])
aw_new = S._to_plain(pq.read_table(D + r"\추출_20261112\t_aw_cl.parquet").to_pandas())
print("codes3", len(codes3), "dom", len(cd), "ov_only", sorted(co))
in_tab = set(aw_new["prdlst_cd"])
print("표에 코드 없음:", len([c for c in codes3 if c not in in_tab]))
for mode in ("original", "all"):
    ft = sf.build_food_table(cl, aw_new, codes3, p2_of, aw_source_filter=mode)
    t = ft.table
    print("==", mode, {c: dict(collections.Counter(t[f"{c}_rule"])) for c in sf.FOOD_COLS})
    print("   parse", ft.parse_rules); print("   unparsed", ft.unparsed)
    if mode == "all":
        cut = pd.Timestamp("2025-06-30")
        m = (pd.to_datetime(s["d"]) <= cut).to_numpy()
        f = sf.food_features(t, p3[m])
        for c in ("food_aw", "food_ph", "food_na"):
            v = f[c].to_numpy(); print("   d<=cut NaN", c, int(np.isnan(v).sum()), "dom", int(np.isnan(v[dom[m]]).sum()), "ov", int(np.isnan(v[~dom[m]]).sum()))
        print("   d<=cut 코드없음 행", int((p3[m] == S.NO_CODE).sum()))
        print("   aw_label", dict(collections.Counter(t["aw_label"].fillna("(null)"))))
        old = pd.read_parquet(M + r"\s0\0-6_static\식품_고정표.parquet").set_index("p3")
        tt = t.set_index("p3")
        for c in sf.FOOD_COLS:
            a = tt[f"food_{c}"]; b = old.loc[tt.index, f"food_{c}"]
            same = (a.isna() & b.isna()) | np.isclose(a.fillna(-9), b.fillna(-9))
            print("   옛 고정표 대비 값 다른 코드", c, int((~same).sum()))
