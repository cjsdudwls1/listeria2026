# -*- coding: utf-8 -*-
"""3차 연구 보고서 Ⅰ부 2장 EDA 보강(2026-10-22 rev2): 품목 3단계별 검사 건수 분포 표 + 검사 건수-검출률 산점도.

자료(읽기만): 리스테리아ML2/data/데이터셋/시료_시군구.parquet (3차 실험이 쓴 국내 시료 628,517행, 검출 1,844).
기간: 기준일 mnfctur_de 2015-04-07 ~ 2026-06-30 전체(시험 해 포함, 자료 설명용 — 모델·변수 고르기에 쓰지 않음).
단위: 시료(행) 하나 = 검사 1건. 검출 = judge_result '검출'.
품목 3단계 = train_prdlst_code_2(빈 값 = '코드없음' 한 품목 — 실험의 과거 검출 이력 '품목 3단계' 무리와 같음).
예측 단위 = 시군구(sgg_code) × 품목 2단계(train_prdlst_code) × 3단계(train_prdlst_code_2), 빈 코드 = '코드없음'.
검사 건수 구간: 1~9, 10~99, 100~999, 1,000~9,999, 10,000 이상(건).
  품목 비율(%) = 그 구간 품목 수 ÷ 전체 품목 수 × 100
  품목별 검출률(%) = 그 품목의 검출 ÷ 그 품목의 검사 × 100 → 구간 안 최소~최대
그림: 품목마다 점 하나(가로 = 검사 건수 로그 눈금, 세로 = 검출률), 가로선 = 전체 검출률(1,844 ÷ 628,517).
출력: figs3/I_5_item3_count_rate.png (Malgun Gothic, 폭 6.3인치·200dpi — 다른 EDA 그림과 같은 모양),
      work3/out_facts/eda_item3_count_bins.csv·eda_unit_count_bins.csv·eda_item3_counts.csv·eda_item_count_summary.json
실행: <리스테리아ML/.venv-exp2 python> -B 06_eda_item_counts.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
SAMPLE = BASE / "data" / "데이터셋" / "시료_시군구.parquet"
REP = BASE / "notebook" / "보고서" / "경복대양식"
FIG = REP / "figs3"
OUT = REP / "work3" / "out_facts"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 모양(02_eda_figs.py 와 같음)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#6f6e69"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FS = 9.5
W = 6.3
DPI = 200
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FS,
    "axes.labelsize": FS, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": FS,
})

# ---------------------------------------------------------------- 자료
t = pq.read_table(SAMPLE, columns=["judge_result", "sgg_code", "train_prdlst_code", "train_prdlst_nm",
                                   "train_prdlst_code_2", "train_prdlst_nm_2", "addr_source"]).to_pandas()
for c in t.columns:
    t[c] = t[c].astype(object)
t["pos"] = (t["judge_result"] == "검출").astype(int)
N_ALL, P_ALL = len(t), int(t["pos"].sum())
assert (N_ALL, P_ALL) == (628517, 1844), (N_ALL, P_ALL)
RATE_ALL = P_ALL / N_ALL * 100
NOCODE = "코드없음"
t["p2"] = t["train_prdlst_code"].where(t["train_prdlst_code"].notna(), NOCODE)
t["p3"] = t["train_prdlst_code_2"].where(t["train_prdlst_code_2"].notna(), NOCODE)

# 3단계 코드 하나가 2단계 코드 하나에만 속하는지(그렇다면 '품목 3단계' = (2단계, 3단계) 쌍과 같은 무리)
p3_p2 = t.groupby("p3")["p2"].nunique()
print("3단계 코드 중 2단계가 둘 이상인 코드 수:", int((p3_p2 > 1).sum()))

BINS = [1, 10, 100, 1000, 10000, np.inf]
BIN_LB = ["1 ~ 9", "10 ~ 99", "100 ~ 999", "1,000 ~ 9,999", "10,000 이상"]


def group_counts(keys):
    g = t.groupby(keys)["pos"].agg(["size", "sum"]).reset_index()
    g = g.rename(columns={"size": "검사", "sum": "검출"})
    g["검출률(%)"] = g["검출"] / g["검사"] * 100
    g["구간"] = pd.cut(g["검사"], BINS, right=False, labels=BIN_LB)
    return g


def bin_table(g, what):
    rows = []
    n = len(g)
    for lb in BIN_LB:
        s = g[g["구간"] == lb]
        rows.append({
            "검사 건수 구간(건)": lb,
            f"{what} 수": len(s),
            f"{what} 비율(%)": len(s) / n * 100,
            "검사 수 합": int(s["검사"].sum()),
            "검출 수": int(s["검출"].sum()),
            f"{what}별 검출률 최소(%)": float(s["검출률(%)"].min()) if len(s) else np.nan,
            f"{what}별 검출률 최대(%)": float(s["검출률(%)"].max()) if len(s) else np.nan,
            f"검출 0건 {what} 수": int((s["검출"] == 0).sum()),
        })
    b = pd.DataFrame(rows)
    assert b[f"{what} 수"].sum() == n and b["검사 수 합"].sum() == N_ALL and b["검출 수"].sum() == P_ALL
    return b


def summary(g, what):
    return {f"{what} 수": int(len(g)), "검사 건수 중앙값": float(g["검사"].median()),
            "검사 건수 평균": float(g["검사"].mean()), "검사 건수 최대": int(g["검사"].max()),
            "검사 건수 최소": int(g["검사"].min()),
            f"검출 있는 {what} 수": int((g["검출"] > 0).sum())}


# ---------------------------------------------------------------- 품목 3단계
item = group_counts(["p3"])
names = t.groupby("p3")[["train_prdlst_nm", "train_prdlst_nm_2"]].first()
item = item.join(names, on="p3")
item = item.sort_values("검사", ascending=False).reset_index(drop=True)
item.to_csv(OUT / "eda_item3_counts.csv", index=False, encoding="utf-8-sig")
ib = bin_table(item, "품목")
ib.to_csv(OUT / "eda_item3_count_bins.csv", index=False, encoding="utf-8-sig")
isum = summary(item, "품목")

# ---------------------------------------------------------------- 예측 단위(시군구 × 2단계 × 3단계)
unit = group_counts(["sgg_code", "p2", "p3"])
ub = bin_table(unit, "단위")
ub.to_csv(OUT / "eda_unit_count_bins.csv", index=False, encoding="utf-8-sig")
usum = summary(unit, "단위")

# ---------------------------------------------------------------- 참고: 주소 출처(addr_source) 값별 시료 수(Ⅰ부 1.2 글)
addr = t.groupby("addr_source", dropna=False)["pos"].agg(["size", "sum"]).sort_values("size", ascending=False)
addr.columns = ["시료", "검출"]

# ---------------------------------------------------------------- 그림: 검사 건수(로그) × 검출률 — (가) 품목 3단계, (나) 예측 단위
# 점 하나 = 품목(단위) 하나, 점선 = 전체 검출률. 두 판은 세로 눈금이 다름(각자 0 ~ 최대).
fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"wspace": 0.24})
TICK_KR = {1: "1", 10: "10", 100: "100", 1000: "1천", 10000: "1만", 100000: "10만", 1000000: "100만"}


def panel(ax, g, title, size, alpha, xlabel):
    x = g["검사"].to_numpy(dtype=float)
    y = g["검출률(%)"].to_numpy(dtype=float)
    ax.scatter(x, y, s=size, color=BLUE, alpha=alpha, edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(RATE_ALL, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.7, g["검사"].max() * 1.8)
    ymax = float(np.nanmax(y))
    ax.set_ylim(-ymax * 0.05, ymax * 1.13)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: TICK_KR.get(int(round(v)), "")))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="both", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.text(0.985, 0.985, title, transform=ax.transAxes, ha="right", va="top", fontsize=FS, color=INK)
    return ymax


a1, a2 = axes
ym1 = panel(a1, item, f"(가) 품목 3단계 {len(item):,}개", 20, 0.8, "검사 건수(건, 로그 눈금)")
a1.set_ylabel("검출률(%)")
# 직접 표시: 검사가 적은데 검출률이 높은 품목 2개(점 오른쪽 글), 검사가 가장 많은 품목 1개(화살표로 점을 가리킴)
for code, dx, dy in (("A0700600000000", 1.5, 0.0), ("A1217900000000", 1.5, 0.0)):
    r = item[item["p3"] == code].iloc[0]
    lab = f"{r['train_prdlst_nm_2']} {int(r['검출'])}/{int(r['검사']):,}건"
    a1.annotate(lab, (r["검사"], r["검출률(%)"]), xytext=(r["검사"] * dx, r["검출률(%)"] + dy * ym1),
                fontsize=FS - 1, color=INK2, ha="left", va="center")
r = item[item["p3"] == "Z3800100000000"].iloc[0]
a1.annotate(f"{r['train_prdlst_nm_2']}\n{int(r['검출'])}/{int(r['검사']):,}건", (r["검사"], r["검출률(%)"]),
            xytext=(r["검사"] / 1.3, ym1 * 0.42), fontsize=FS - 1, color=INK2, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=1, shrinkB=3,
                            connectionstyle="angle,angleA=0,angleB=90,rad=0"))
a1.text(0.9, RATE_ALL + ym1 * 0.035, f"전체 {RATE_ALL:.2f}%", ha="left", va="bottom", fontsize=FS - 1, color=INK2)
ym2 = panel(a2, unit, f"(나) 예측 단위 {len(unit):,}개", 9, 0.45, "검사 건수(건, 로그 눈금)")
a2.text(0.9, RATE_ALL + ym2 * 0.035, f"전체 {RATE_ALL:.2f}%", ha="left", va="bottom", fontsize=FS - 1, color=INK2)
fig.subplots_adjust(left=0.075, right=0.99, top=0.975, bottom=0.17)
fp = FIG / "I_5_item3_count_rate.png"
fig.savefig(fp, dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- 출력
res = {"자료": f"국내 시료 {N_ALL:,}건 · 검출 {P_ALL:,}건(기준일 전체 기간)", "전체 검출률(%)": RATE_ALL,
       "품목 요약": isum, "단위 요약": usum, "그림": str(fp)}
(OUT / "eda_item_count_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(f"전체 검출률 {RATE_ALL:.4f}%")
print("== 품목 3단계 구간표 ==")
print(ib.to_string(index=False))
print("품목 요약:", isum)
print("== 예측 단위 구간표 ==")
print(ub.to_string(index=False))
print("단위 요약:", usum)
print("== 검사 많은 품목 상위 8 ==")
print(item.head(8)[["p3", "train_prdlst_nm", "train_prdlst_nm_2", "검사", "검출", "검출률(%)"]].to_string(index=False))
print("== 검출률 높은 품목 상위 10 ==")
print(item.sort_values("검출률(%)", ascending=False).head(10)[["p3", "train_prdlst_nm", "train_prdlst_nm_2", "검사", "검출", "검출률(%)"]].to_string(index=False))
print("== 검출 있는 품목의 검사 건수 ==")
print(item[item["검출"] > 0][["train_prdlst_nm", "train_prdlst_nm_2", "검사", "검출", "검출률(%)"]].sort_values("검사").to_string(index=False))
print("== 주소 출처(addr_source) ==")
print(addr.to_string())
print("그림", fp)
