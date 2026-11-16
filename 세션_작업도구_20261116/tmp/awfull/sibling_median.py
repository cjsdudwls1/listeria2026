# 분석용: 지금 고정표에서 결측인 3단계 코드에 '같은 상위 코드를 공유하는 다른 코드 값의 중앙값'을 쓰면 얼마나 채워지나
import sys, collections
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import static_feats as sf, samples as S
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
a = S._to_plain(pq.read_table(D + r"\추출_20261112\t_aw_cl.parquet").to_pandas())
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "hrnk_prdlst_cd", "kor_nm", "lv"])
nm = dict(zip(cl.prdlst_cd, cl.kor_nm)); up = dict(zip(cl.prdlst_cd, cl.hrnk_prdlst_cd))
children = collections.defaultdict(list)
for c, u in up.items():
    if u is not None and u == u and u != c: children[u].append(c)
s = pd.read_parquet(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\awfull\inp\samples5.parquet")
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object); y = s["y"].to_numpy(); real = p3 != S.NO_CODE
codes3 = sorted(set(p3[real])); p2_of = sf.pair_p2(p3[real], p2[real])
t = sf.build_food_table(cl[["prdlst_cd", "hrnk_prdlst_cd"]], a, codes3, p2_of, aw_source_filter="all").table.set_index("p3")
val = {c: {} for c in ("aw", "ph", "na")}
for r in a.itertuples(index=False):
    for c in val:
        x = sf.parse_db(getattr(r, c))[0]
        if not np.isnan(x): val[c][r.prdlst_cd] = x
for c in ("aw", "ph", "na"):
    miss = [k for k in t.index if np.isnan(t.loc[k, f"food_{c}"])]
    rows_tot = int(np.isin(p3, miss).sum()); pos_tot = int((np.isin(p3, miss) & (y == 1)).sum())
    filled_rows = filled_pos = 0; det = []
    for k in miss:
        par = up.get(k)                       # 분류표 상위 코드
        sib_sub = [d for d in sf.descendants(children, par) if d != k and d in val[c]] if par else []
        n_rows = int((p3 == k).sum()); n_pos = int(((p3 == k) & (y == 1)).sum())
        if sib_sub:
            filled_rows += n_rows; filled_pos += n_pos
        det.append((n_rows, k, nm.get(k), par, nm.get(par), len(sib_sub), float(np.median([val[c][d] for d in sib_sub])) if sib_sub else None, n_pos))
    print(f"== {c}: 지금 결측 코드 {len(miss)}개 · {rows_tot:,}행 · 검출 {pos_tot} → 형제(상위 코드 아래 다른 코드) 중앙값으로 채워지는 것: {filled_rows:,}행 · 검출 {filled_pos}")
    if c == "aw":
        for d in sorted(det, reverse=True)[:8]:
            print("   행 %s | %s %s | 상위 %s %s | 값 있는 형제 %s개 | 중앙값 %s | 검출 %s" % d)
