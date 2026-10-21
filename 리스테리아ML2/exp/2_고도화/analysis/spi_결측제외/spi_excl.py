# SPI 결측 구간을 뺀 평가(재학습 없음)
# 검증·시험 (행, h) 중 SPI 창 T-50 .. T-1 에 그 시군구의 SPI3 또는 SPI6 결측 날이 하루라도 있으면 뺀다.
# 구간 가중 PR-AUC = 0.5 x (H1~20 평균) + 0.3 x (H21~40 평균) + 0.2 x (H41~60 평균)
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

BASE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2"
R = os.path.join(BASE, "exp", "2_고도화", "results")
OUT = os.path.join(BASE, "exp", "2_고도화", "analysis", "spi_결측제외")

spi = pq.read_table(os.path.join(BASE, "data", "데이터셋", "외부_수집", "시군구_일별_SPI.parquet")).to_pandas()
dates = pd.to_datetime(spi["date"]).values.astype("datetime64[D]")
d0 = dates.min()
di = (dates - d0).astype(np.int64)
miss = (spi["spi3"].isna() | spi["spi6"].isna()).to_numpy().astype(np.int64)
sggs = sorted(spi["sgg_code"].astype(str).unique())
si = {s: i for i, s in enumerate(sggs)}
gi = spi["sgg_code"].astype(str).map(si).to_numpy()
M = np.zeros((len(sggs), int(di.max()) + 1), np.int64)
M[gi, di] = miss
C = np.concatenate([np.zeros((len(sggs), 1), np.int64), np.cumsum(M, axis=1)], axis=1)  # C[:, k] = 결측 날 수(날짜 번호 < k)

BANDS = [(1, 20, 0.5), (21, 40, 0.3), (41, 60, 0.2)]
RUNS = ["E00", "A3", "F1", "F4", "A3_F4"]
rows = []
for rid in RUNS:
    for s in (42, 43, 44):
        for sp in ("val", "test"):
            t = pq.read_table(os.path.join(R, f"{rid}__s{s}", f"pred_{sp}.parquet")).to_pandas()
            d = pd.to_datetime(t["d"]).values.astype("datetime64[D]")
            h = t["h"].astype(int).to_numpy()
            T = (d - d0).astype(np.int64) - h
            g = t["sgg_code"].astype(str).map(si).to_numpy()
            nmiss = C[g, T] - C[g, T - 50]          # 창 T-50 .. T-1 의 결측 날 수
            y = t["y"].to_numpy(); p = t["p"].to_numpy()
            for lab, mask in (("전체", np.ones(len(t), bool)), ("SPI결측창제외", nmiss == 0)):
                ap = {}
                for hh in range(1, 61):
                    mh = mask & (h == hh)
                    ap[hh] = average_precision_score(y[mh], p[mh]) if y[mh].sum() > 0 else np.nan
                m60 = float(np.nanmean(list(ap.values())))
                wt = float(sum(w * np.nanmean([ap[x] for x in range(a, b + 1)]) for a, b, w in BANDS))
                rows.append(dict(실험=rid, 시드=s, 구간=sp, 평가행=lab, 구간가중PR=wt, 평균60PR=m60,
                                 남은비율=float(mask.mean()), 남은검출_h당=float(y[mask].sum()) / 60))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "spi_결측제외_시드별.csv"), index=False, encoding="utf-8-sig")
gsum = df.groupby(["실험", "구간", "평가행"]).agg(
    구간가중PR=("구간가중PR", "mean"), 표준편차=("구간가중PR", "std"), 평균60PR=("평균60PR", "mean"),
    남은비율=("남은비율", "mean"), 남은검출_h당=("남은검출_h당", "mean")).reset_index()
gsum.to_csv(os.path.join(OUT, "spi_결측제외_요약.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "spi_결측제외_요약.txt"), "w", encoding="utf-8") as f:
    f.write(gsum.round(4).to_string(index=False))
