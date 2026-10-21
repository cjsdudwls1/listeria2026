import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
BASE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2"
R = BASE + r"\exp\1_베이스라인_lims기상\results"
OUT = BASE + r"\exp\1_베이스라인_lims기상\analysis\h별_예측확률"
cur = pd.read_csv(r"C:\Users\NT751\.claude\jobs\2af334fa\tmp\hprob\mean_curves.csv", index_col=0, encoding="utf-8-sig")
cur.to_csv(OUT + r"\h별_평균확률.csv", encoding="utf-8-sig")

rep = pq.read_table(BASE + r"\data\데이터셋\시군구_대표좌표.parquet").to_pandas()
SG = dict(zip(rep["sgg_code"].astype(str), rep["sgg_nm"].astype(str)))
smp = pq.read_table(BASE + r"\data\데이터셋\시료_시군구.parquet",
                    columns=["train_prdlst_code_2", "train_prdlst_nm", "train_prdlst_nm_2"]).to_pandas()
smp = smp.drop_duplicates("train_prdlst_code_2")
PN = {str(r.train_prdlst_code_2): f"{r.train_prdlst_nm} > {r.train_prdlst_nm_2}" for r in smp.itertuples()}

def nice(k):
    a, b, c = k.split("|")
    return f"{SG.get(a, a)} × {PN.get(b, b)} × {c}"

t = pq.read_table(R + r"\catboost__summary__L050\pred_test.parquet").to_pandas()
t["key"] = t["sgg_code"].astype(str) + "|" + t["train_prdlst_code_2"].astype(str) + "|" + t["d"].astype(str)
P = t.pivot(index="key", columns="h", values="p")
m = P.mean(1).sort_values()
pick = [m.index[-1], m.index[-50], m.index[int(len(m) * 0.9)], m.index[len(m) // 2]]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
lab = {"catboost__summary__L050": "CatBoost 요약 50일(베이스라인)", "catboost__summary__L020": "CatBoost 요약 20일",
       "catboost__summary__L030": "CatBoost 요약 30일", "catboost__summary__L060": "CatBoost 요약 60일",
       "catboost__summary__L100": "CatBoost 요약 100일", "lstm__seq__L100": "LSTM 100일"}
for c, l in lab.items():
    ax[0].plot(cur.index, cur[c], lw=2.5 if c == "catboost__summary__L050" else 1.2, label=l)
ax[0].set_title("시험 구간 전체 행의 h별 평균 예측 확률")
ax[0].set_xlabel("예측 거리 h(일)"); ax[0].set_ylabel("평균 확률")
ax[0].set_ylim(0, 0.008); ax[0].axhline(0.00483, color="gray", ls="--", lw=1)
ax[0].text(2, 0.0051, "시험 검출 행 비율 0.483%", color="gray", fontsize=9)
ax[0].legend(fontsize=8)
for k in pick:
    ax[1].plot(P.columns, P.loc[k].values, label=nice(k))
ax[1].set_yscale("log")
ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax[1].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax[1].set_title("베이스라인: 행(단위 × 목표일)마다 h에 따른 확률 예시")
ax[1].set_xlabel("예측 거리 h(일)"); ax[1].set_ylabel("예측 확률(로그 눈금)")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT + r"\h별_예측확률.png", dpi=130)
print([nice(k) for k in pick])
