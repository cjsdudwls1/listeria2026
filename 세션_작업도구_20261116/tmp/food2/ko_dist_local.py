# 분석용(로컬 사본 자료, 결과 폴더에 안 씀): 식품 v2 를 메모리에서 만들고 KO 6칼럼 시료 기준 분포
import sys, time
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from lst import inputs as ip, static_feats as sf, samples as S
import s_ko_dist as kd
D = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data"
M = r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror"
cfg = {"results": {"samples": "s0/0-1/apply/samples.parquet",
                   "domestic_weather": "s0/0-4_v2/시군구_일별_기상_0-4_합본.parquet",
                   "overseas_weather": "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet"},
       "data": {}, "y_cut": "2025-06-30"}
ctx = ip.load_context(cfg, D, M, components=("weather",))
s = ctx.samples
p3 = s["p3"].to_numpy(dtype=object); p2 = s["p2"].to_numpy(dtype=object)
dom = (s["src_route"] == "domestic").to_numpy(); real = p3 != S.NO_CODE
codes3 = sorted(set(p3[dom & real]) | (set(p3[~dom & real]) - set(p3[dom & real])))
cl = pd.read_parquet(D + r"\raw\raw_t_prdlst_cl.parquet", columns=["prdlst_cd", "hrnk_prdlst_cd"])
aw = S._to_plain(pq.read_table(D + r"\추출_20261112\t_aw_cl.parquet").to_pandas())
ctx.food = sf.build_food_table(cl, aw, codes3, sf.pair_p2(p3[real], p2[real]), aw_source_filter="all").table
print("aw range", ctx.food["food_aw"].min(), ctx.food["food_aw"].max(), "ph", ctx.food["food_ph"].min(), ctx.food["food_ph"].max())
t = time.time()
v = kd.ko_values(ctx, s.reset_index(drop=True), 60, 50000, log=lambda m: None)
tab = kd.dist_table(v, s.reset_index(drop=True), "2025-06-30")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab[tab["묶음"].isin(["전체", "국내", "해외"])].to_string(index=False))
print(f"{time.time()-t:.0f}s")
