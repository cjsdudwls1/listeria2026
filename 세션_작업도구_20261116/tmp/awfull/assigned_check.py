import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
a = S._to_plain(pq.read_table(D + r"\추출_20261112\t_aw_cl.parquet").to_pandas())
m = a["aw_source"].astype(str).str.startswith("assigned")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(a.loc[m, ["prdlst_cd", "kor_nm", "lv", "aw", "mc", "ph", "na", "aflatoxin_aw", "aw_source"]].to_string(index=False))
print("aflatoxin_aw 값 분포:", collections.Counter(a["aflatoxin_aw"].astype(str)).most_common(8))
print("aw 원문이 0·1·2 (정수 글자)인 행:", collections.Counter((a.loc[a["aw"].astype(str).str.strip().isin(["0", "1", "2"]), "aw_source"]).astype(str)))
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "hrnk_prdlst_cd"])
s = pd.read_parquet(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\awfull\inp\samples5.parquet")
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object); y = s["y"].to_numpy(); real = p3 != S.NO_CODE
codes3 = sorted(set(p3[real])); p2_of = sf.pair_p2(p3[real], p2[real])
old = pd.read_parquet(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\awfull\inp\식품_고정표.parquet").set_index("p3")
new = sf.build_food_table(cl, a, codes3, p2_of, aw_source_filter="all").table.set_index("p3")
d = (old["food_aw"].fillna(-9) - new.loc[old.index, "food_aw"].fillna(-9)).abs() > 1e-12
print(f"v1→v2 aw 바뀐 3단계 코드 {int(d.sum())}개 · 시료 {int(np.isin(p3, old.index[d]).sum()):,}행 · 검출 {int((np.isin(p3, old.index[d]) & (y == 1)).sum())}")
for c in ["E0300000000000", "E0318000000000", "E0326400000000"]:
    mm = p3 == c
    print(c, "행", int(mm.sum()), "검출", int(y[mm].sum()), "| v1", old.loc[c, "food_aw"] if c in old.index else None, old.loc[c, "aw_rule"] if c in old.index else None,
          "| v2", new.loc[c, "food_aw"], new.loc[c, "aw_rule"])
# assigned 값이 들어간 3단계 코드(자기값·하위중앙·2단계값 경로 포함) 찾기: assigned 행 aw 만 지운 표로 다시 만들어 비교
b = a.copy(); b.loc[m, "aw"] = None
alt = sf.build_food_table(cl, b, codes3, p2_of, aw_source_filter="all").table.set_index("p3")
dd = (alt["food_aw"].fillna(-9) - new["food_aw"].fillna(-9)).abs() > 1e-12
print("assigned 행 aw 때문에 값이 달라지는 3단계 코드:", [(c, new.loc[c, "food_aw"], alt.loc[c, "food_aw"], int((p3 == c).sum()), int(((p3 == c) & (y == 1)).sum())) for c in new.index[dd]])
