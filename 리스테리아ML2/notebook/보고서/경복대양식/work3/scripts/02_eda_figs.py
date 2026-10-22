# -*- coding: utf-8 -*-
"""3차 연구 보고서 'I. 데이터 전처리 부문' EDA 그림 3개 + 그림 자료 표 (2026-10-22, rev1).

자료(읽기만): 리스테리아ML2/data/데이터셋/시료_시군구.parquet (3차 실험이 쓴 국내 시료 628,517행),
             시군구_대표좌표.parquet(시도 이름), 식정원_db_기상기후/시군구_일별_기상.parquet(평균기온 avg_temp).
단위: 시료(행) 하나. 검출 = judge_result '검출'.
  검출 비중(%) = 그 값의 검출 시료 ÷ 전체 검출 시료(1,844) × 100   (사용자 기본 지표)
  검사 비중(%) = 그 값의 시료 ÷ 전체 시료(628,517) × 100           (참고 막대)
  검출률(%)   = 그 구간의 검출 시료 ÷ 그 구간의 시료 × 100          (기온 표만)
기간: 기준일 mnfctur_de 2015-04-07 ~ 2026-06-30 전체(시험 해 포함, 자료 설명용 — 모델·변수 고르기에 쓰지 않음).
rev1(사용자 결정 10-22): 그림 5개 -> 3개(연도별 검출, 품목 3단계 검출 비중 상위 10, 월별 검출 비중). 시도별·기온 구간별은
  그림을 그리지 않고 표(eda_sido.csv·eda_temp_*.csv)만 만들어 본문 글 한두 줄 수치로 씀(계산은 전과 같음).
  그림은 본문 폭(161mm)에 줄이지 않고 들어가게 폭 6.3인치·글자 9.5pt 이상, 그림 안 제목은 빼고(그림 아래 '[그림 n]' 제목이
  대신함) 200dpi 로 저장 — 한 쪽에 그림 두 개가 들어가는 높이.
출력: figs3/I_1_year_detections.png·I_2_item3_share_top10.png·I_4_month_share.png (Malgun Gothic), work3/out_facts/eda_*.csv
실행: <.venv-exp2 python> -B 02_eda_figs.py
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
DS = BASE / "data" / "데이터셋"
SAMPLE = DS / "시료_시군구.parquet"
COORD = DS / "시군구_대표좌표.parquet"
WX = DS / "식정원_db_기상기후" / "시군구_일별_기상.parquet"
REP = BASE / "notebook" / "보고서" / "경복대양식"
FIG = REP / "figs3"
OUT = REP / "work3" / "out_facts"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 모양(밝은 바탕 인쇄용, dataviz 기준 팔레트)
BLUE = "#2a78d6"        # 주 계열(검출)
REF = "#c9c7bf"         # 참고 계열(검사 비중) - 회색
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#6f6e69"       # 참고 숫자(인쇄에서 읽히게 조금 진하게)
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FS = 9.5                # 기본 글자(pt) = 인쇄 크기(그림을 줄이지 않고 넣음)
W = 6.3                 # 그림 폭(인치) = 본문 폭 161mm 안
DPI = 200
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FS,
    "axes.labelsize": FS, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": FS,
})


def style(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("그림", p)
    return str(p)


# ---------------------------------------------------------------- 자료
t = pq.read_table(SAMPLE, columns=["judge_result", "mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_nm",
                                   "train_prdlst_code_2", "train_prdlst_nm_2"]).to_pandas()
for c in t.columns:
    if c != "mnfctur_de":
        t[c] = t[c].astype(object)
t["pos"] = (t["judge_result"] == "검출").astype(int)
t["d"] = pd.to_datetime(t["mnfctur_de"]).dt.normalize()
N_ALL, P_ALL = len(t), int(t["pos"].sum())
assert (N_ALL, P_ALL) == (628517, 1844), (N_ALL, P_ALL)
PERIOD = f"{t['d'].min().date()}~{t['d'].max().date()}"
SUB = f"국내 시료 {N_ALL:,}건 · 검출 {P_ALL:,}건 · 기준일 {PERIOD}"


def share_table(key):
    g = t.groupby(key, dropna=False)["pos"].agg(["size", "sum"]).reset_index()
    g.columns = list(g.columns[:-2]) + ["검사", "검출"]
    g["검출 비중(%)"] = g["검출"] / P_ALL * 100
    g["검사 비중(%)"] = g["검사"] / N_ALL * 100
    g["검출률(%)"] = g["검출"] / g["검사"] * 100
    return g


FIGS = {}

# ---------------------------------------------------------------- 그림 1. 연도별 검출 수(위) + 검사 수(아래, 참고)
t["year"] = t["d"].dt.year
yr = share_table("year")
yr.to_csv(OUT / "eda_year.csv", index=False, encoding="utf-8-sig")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.18})
x = np.arange(len(yr))
labels = [str(v) for v in yr["year"]]
a1.bar(x, yr["검출"], width=0.62, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
for xi, v in zip(x, yr["검출"]):
    a1.text(xi, v + max(yr["검출"]) * 0.03, f"{v:,}", ha="center", va="bottom", fontsize=FS, color=INK)
a1.set_ylabel("검출 시료(건)")
a1.set_ylim(0, max(yr["검출"]) * 1.22)
style(a1)
a2.bar(x, yr["검사"] / 1000, width=0.62, color=REF, edgecolor="white", linewidth=1.0, zorder=3)
for xi, v in zip(x, yr["검사"]):
    a2.text(xi, v / 1000 + max(yr["검사"]) / 1000 * 0.03, f"{v / 1000:.1f}" if v >= 1000 else f"{v}건", ha="center", va="bottom",
            fontsize=FS, color=MUTED)
a2.set_ylabel("검사 시료\n(천 건)")
a2.set_ylim(0, max(yr["검사"]) / 1000 * 1.3)
a2.set_xticks(x, labels)
style(a2)
fig.align_ylabels([a1, a2])
fig.subplots_adjust(left=0.115, right=0.99, top=0.98, bottom=0.09)
FIGS["I_1"] = save(fig, "I_1_year_detections.png")

# ---------------------------------------------------------------- 그림 2. 품목 3단계별 검출 비중 상위 10
p3 = share_table(["train_prdlst_code_2", "train_prdlst_nm", "train_prdlst_nm_2"])
p3 = p3.sort_values(["검출", "train_prdlst_code_2"], ascending=[False, True], na_position="last").reset_index(drop=True)
p3["순위"] = np.arange(1, len(p3) + 1)
p3.to_csv(OUT / "eda_item3.csv", index=False, encoding="utf-8-sig")
top = p3.head(10).copy()
top_share = float(top["검출 비중(%)"].sum())
n_items_pos = int((p3["검출"] > 0).sum())
top["label"] = top["train_prdlst_nm"].astype(str) + " > " + top["train_prdlst_nm_2"].astype(str)
top["label"] = top["label"].str.replace("기타기준규격외 > 기타기준규격외", "기타기준규격외", regex=False)
fig, ax = plt.subplots(figsize=(W, 3.5))
yy = np.arange(len(top))[::-1]
hgt = 0.38
ax.barh(yy + hgt / 2, top["검출 비중(%)"], height=hgt, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label="검출 비중(전체 검출 중 %)")
ax.barh(yy - hgt / 2, top["검사 비중(%)"], height=hgt, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label="검사 비중(전체 검사 중 %)")
xmax = max(top["검출 비중(%)"].max(), top["검사 비중(%)"].max())
for yi, v, n in zip(yy, top["검출 비중(%)"], top["검출"]):
    ax.text(v + xmax * 0.012, yi + hgt / 2, f"{v:.1f}% ({n:,}건)", va="center", fontsize=FS, color=INK)
for yi, v in zip(yy, top["검사 비중(%)"]):
    ax.text(v + xmax * 0.012, yi - hgt / 2, f"{v:.2f}%" if v < 0.1 else f"{v:.1f}%", va="center", fontsize=FS, color=MUTED)
ax.set_yticks(yy, top["label"])
ax.set_xlim(0, xmax * 1.3)
ax.set_xlabel("비중(%)")
ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
style(ax, "x")
fig.subplots_adjust(left=0.375, right=0.99, top=0.99, bottom=0.125)
FIGS["I_2"] = save(fig, "I_2_item3_share_top10.png")

# ---------------------------------------------------------------- 시도별 검출 비중(표만 — rev1 에서 그림은 뺌)
coord = pq.read_table(COORD).to_pandas()
SIDO_NM = {c[:2]: n.split()[0] for c, n in zip(coord["sgg_code"], coord["sgg_nm"])}
SHORT = {"서울특별시": "서울", "전남광주통합특별시": "전남광주", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
         "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "충청북도": "충북", "충청남도": "충남",
         "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주", "강원특별자치도": "강원", "전북특별자치도": "전북"}
t["sido"] = t["sgg_code"].astype(str).str[:2]
sd = share_table("sido")
sd["시도"] = sd["sido"].map(SIDO_NM)
sd["약칭"] = sd["시도"].map(SHORT)
assert sd["약칭"].notna().all(), sd
sd = sd.sort_values(["검출", "검사"], ascending=[False, False]).reset_index(drop=True)
sd.to_csv(OUT / "eda_sido.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 그림 3. 월별 검출 비중
t["month"] = t["d"].dt.month
mo = share_table("month")
# 달력상 그 달이 기준일 기간(2015-04-07 ~ 2026-06-30)에 든 해 수: 4~6월 12개 해(2015~2026), 나머지 11개 해
mo["기간에 든 해 수(달력)"] = [12 if m in (4, 5, 6) else 11 for m in mo["month"]]
mo.to_csv(OUT / "eda_month.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(W, 2.35))
x = np.arange(12)
w = 0.38
ax.bar(x - w / 2, mo["검출 비중(%)"], width=w, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label="검출 비중(전체 검출 중 %)")
ax.bar(x + w / 2, mo["검사 비중(%)"], width=w, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label="검사 비중(전체 검사 중 %)")
ymax = max(mo["검출 비중(%)"].max(), mo["검사 비중(%)"].max())
for xi, v in zip(x, mo["검출 비중(%)"]):
    ax.text(xi - w / 2, v + ymax * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=FS, color=INK)
ax.set_xticks(x, [f"{m}월" for m in range(1, 13)])
ax.set_ylim(0, ymax * 1.32)
ax.set_ylabel("비중(%)")
ax.legend(loc="upper left", ncol=2, handlelength=1.2, borderaxespad=0.1)
style(ax)
fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.125)
FIGS["I_4"] = save(fig, "I_4_month_share.png")

# ---------------------------------------------------------------- 월평균 기온 구간별 검출률(표만 — rev1 에서 그림은 뺌)
w_ = pq.read_table(WX, columns=["sgg_code", "date", "avg_temp"]).to_pandas()
w_["date"] = pd.to_datetime(w_["date"])
w_["ym"] = w_["date"].dt.to_period("M")
mt = w_.groupby(["sgg_code", "ym"])["avg_temp"].mean().rename("t_month").reset_index()
t["ym"] = t["d"].dt.to_period("M")
t = t.merge(mt, left_on=["sgg_code", "ym"], right_on=["sgg_code", "ym"], how="left")
assert t["t_month"].notna().all(), int(t["t_month"].isna().sum())
# 참고(민감도): 기준일 전 30일(d−30 ~ d−1) 시군구 평균기온
sgg_list = sorted(w_["sgg_code"].unique())
dates = np.array(sorted(w_["date"].unique()), dtype="datetime64[D]")
w_ = w_.sort_values(["sgg_code", "date"])
A = w_["avg_temp"].to_numpy(dtype=np.float64).reshape(len(sgg_list), len(dates))
ok = ~np.isnan(A)
cs = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(np.where(ok, A, 0.0), axis=1)], axis=1)
cn = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(ok, axis=1)], axis=1)
si = t["sgg_code"].map({s: i for i, s in enumerate(sgg_list)}).to_numpy()
di = (t["d"].to_numpy().astype("datetime64[D]") - dates[0]).astype(np.int64)
assert (di >= 30).all()
with np.errstate(invalid="ignore", divide="ignore"):
    t["t_prev30"] = (cs[si, di] - cs[si, di - 30]) / (cn[si, di] - cn[si, di - 30])
EDGES = [-np.inf, 0, 5, 10, 15, 20, 25, np.inf]
BIN_LB = ["0 미만", "0~5", "5~10", "10~15", "15~20", "20~25", "25 이상"]


def bin_table(col):
    b = pd.cut(t[col], EDGES, right=False, labels=BIN_LB)
    g = t.assign(bin=b).groupby("bin", observed=False)["pos"].agg(["size", "sum"]).reset_index()
    g.columns = ["기온 구간(°C, 이상~미만)", "검사", "검출"]
    g["검출률(%)"] = g["검출"] / g["검사"] * 100
    g["검출 비중(%)"] = g["검출"] / P_ALL * 100
    g["검사 비중(%)"] = g["검사"] / N_ALL * 100
    return g


tb = bin_table("t_month")
tb.to_csv(OUT / "eda_temp_month_bins.csv", index=False, encoding="utf-8-sig")
tb30 = bin_table("t_prev30")
tb30.to_csv(OUT / "eda_temp_prev30_bins.csv", index=False, encoding="utf-8-sig")
# 참고: 품목 구성 차이 확인용 — 2단계 코드가 Z 로 시작하는 품목(기준규격외·정액 등) / 그 밖
zmask = t["train_prdlst_code"].astype(str).str.startswith("Z")
sub_rows = []
for nm, m in (("Z로 시작하는 2단계 코드", zmask), ("그 밖의 2단계 코드(코드없음 포함)", ~zmask)):
    b = pd.cut(t.loc[m, "t_month"], EDGES, right=False, labels=BIN_LB)
    g = t.loc[m].assign(bin=b).groupby("bin", observed=False)["pos"].agg(["size", "sum"]).reset_index()
    for r in g.itertuples(index=False):
        sub_rows.append({"품목 묶음": nm, "기온 구간(°C)": r[0], "검사": int(r[1]), "검출": int(r[2]),
                         "검출률(%)": (r[2] / r[1] * 100) if r[1] else np.nan})
pd.DataFrame(sub_rows).to_csv(OUT / "eda_temp_month_bins_by_Z.csv", index=False, encoding="utf-8-sig")

summary = {"sub": SUB, "figs": FIGS, "top10_share": top_share, "n_items_pos": n_items_pos,
           "temp_month_bins": tb.to_dict("records"), "temp_prev30_bins": tb30.to_dict("records")}
(OUT / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(yr.to_string())
print(top[["순위", "label", "검사", "검출", "검출 비중(%)", "검사 비중(%)", "검출률(%)"]].to_string())
print(sd[["약칭", "검사", "검출", "검출 비중(%)", "검사 비중(%)", "검출률(%)"]].to_string())
print(mo.to_string())
print(tb.to_string())
print(tb30.to_string())
print(pd.DataFrame(sub_rows).to_string())
