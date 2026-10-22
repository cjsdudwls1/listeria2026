# -*- coding: utf-8 -*-
"""3차 연구 보고서 Ⅰ부 2장 EDA 그림: 품목 3단계의 검사 건수 구간별 품목 수(왼쪽)와 검출·검사 비중(오른쪽) (2026-10-22).

사용자 지시(10-22): 표 '품목 3단계별 검사 건수 분포'가 이해되지 않으니 그림으로 바꿈. 06_eda_item_counts.py 의 산점도
  (I_5_item3_count_rate.png)는 이해가 안 된다고 뺀 그림이라 쓰지 않음.
자료(읽기만): work3/out_facts/eda_item3_count_bins.csv (06_eda_item_counts.py 결과 — 검사 건수 구간별 품목 수·검사 수 합·검출 수,
  국내 시료 628,517건 · 검출 1,844건 · 품목 3단계 218개, 기준일 전체 기간).
  품목 비율(%) = 그 구간 품목 수 ÷ 전체 품목 수(218) × 100          (왼쪽 판 꼬리표 괄호)
  검출 비중(%) = 그 구간 품목들의 검출 ÷ 전체 검출(1,844) × 100      (오른쪽 판 파란 막대)
  검사 비중(%) = 그 구간 품목들의 검사 ÷ 전체 검사(628,517) × 100    (오른쪽 판 회색 막대)
모양: 02_eda_figs.py·06_eda_item_counts.py 와 같음(Malgun Gothic, 폭 6.3인치, 200dpi, 글자 9.5pt, 같은 rcParams,
  BLUE = 검출, REF = 검사). 왼쪽 판 막대는 검출·검사와 헷갈리지 않는 중립색(진한 회청색) 하나. 그림 안 제목 없음
  (캡션은 원고에 있음). 꼬리표 = 소수 한 자리(0.1 미만은 둘째 자리, 0 은 '0').
10-22 검수 반영: 왼쪽 판 괄호 % 가 무엇 중 비율인지(전체 218개 중) 그림에 없어, 세로축 이름을 '품목 수(개)' →
  '품목 수 / (개, 괄호 = 전체 218개 중 %)' 두 줄로 바꾸고 왼쪽 여백을 0.075 → 0.10 으로 넓힘(자료·숫자는 그대로).
출력: figs3/I_5_item3_test_bins.png
실행: <리스테리아ML/.venv-exp2 python> -B 11_eda_item_test_bins.py
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REP = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\경복대양식")
SRC = REP / "work3" / "out_facts" / "eda_item3_count_bins.csv"
FIG = REP / "figs3"

# ---------------------------------------------------------------- 모양(02_eda_figs.py 와 같음)
BLUE = "#2a78d6"        # 검출
REF = "#c9c7bf"         # 검사(회색)
NEUTRAL = "#4f5d6e"     # 품목 수(진한 회청색 — BLUE 와 OKLab ΔE 16 이상, REF 와 35 이상)
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


def style(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fmt(v):
    """비중 꼬리표: 0 은 '0', 0.1 미만은 소수 둘째 자리, 나머지는 첫째 자리."""
    if v == 0:
        return "0"
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


# ---------------------------------------------------------------- 자료
b = pd.read_csv(SRC, encoding="utf-8-sig")
BINS = ["1 ~ 9", "10 ~ 99", "100 ~ 999", "1,000 ~ 9,999", "10,000 이상"]
assert list(b["검사 건수 구간(건)"]) == BINS, list(b["검사 건수 구간(건)"])
N_ITEM, N_ALL, P_ALL = int(b["품목 수"].sum()), int(b["검사 수 합"].sum()), int(b["검출 수"].sum())
assert (N_ITEM, N_ALL, P_ALL) == (218, 628517, 1844), (N_ITEM, N_ALL, P_ALL)
n_item = b["품목 수"].to_numpy(dtype=int)
item_pct = n_item / N_ITEM * 100
assert np.allclose(item_pct, b["품목 비율(%)"].to_numpy(dtype=float))
det = b["검출 수"].to_numpy(dtype=float) / P_ALL * 100
insp = b["검사 수 합"].to_numpy(dtype=float) / N_ALL * 100
# 원고 수치와 같은지 확인
assert [fmt(v) for v in det] == ["0", "0.8", "1.0", "8.1", "90.1"], [fmt(v) for v in det]
assert [fmt(v) for v in insp] == ["0.03", "0.4", "2.6", "12.5", "84.5"], [fmt(v) for v in insp]

# 가로축 눈금 글자(긴 글자는 두 줄)
TICK = ["1 ~ 9", "10 ~ 99", "100 ~\n999", "1,000 ~\n9,999", "10,000\n이상"]
XLABEL = "품목의 검사 건수(건)"

# ---------------------------------------------------------------- 그림
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.22})
x = np.arange(len(BINS))

# 왼쪽: 구간별 품목 수(괄호 = 218개 중 비율)
a1.bar(x, n_item, width=0.62, color=NEUTRAL, edgecolor="white", linewidth=0.8, zorder=3)
ymax1 = float(n_item.max())
for xi, n, p in zip(x, n_item, item_pct):
    a1.text(xi, n + ymax1 * 0.03, f"{n}개\n({p:.1f}%)", ha="center", va="bottom", fontsize=FS, color=INK)
a1.set_ylim(0, ymax1 * 1.3)
a1.set_xlim(-0.6, len(BINS) - 0.4)
a1.set_xticks(x, TICK)
# 세로축 이름에 괄호 % 의 기준(전체 품목 수)을 적음(10-22 검수). 한 줄로는 그림 높이를 넘어 02_eda_figs.py 의
# '검사 시료\n(천 건)' 처럼 괄호 앞에서 줄을 바꿈
a1.set_ylabel(f"품목 수\n(개, 괄호 = 전체 {N_ITEM}개 중 %)")
a1.set_xlabel(XLABEL)
style(a1)

# 오른쪽: 구간별 검출 비중(파랑)·검사 비중(회색)
w = 0.4
a2.bar(x - w / 2, det, width=w, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label="검출 비중(전체 검출 중 %)")
a2.bar(x + w / 2, insp, width=w, color=REF, edgecolor="white", linewidth=0.8, zorder=3, label="검사 비중(전체 검사 중 %)")
ymax2 = float(max(det.max(), insp.max()))
lab_d = [a2.annotate(fmt(v), (xi - w / 2, v + ymax2 * 0.02), xytext=(0, 0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS, color=INK) for xi, v in zip(x, det)]
lab_i = [a2.annotate(fmt(v), (xi + w / 2, v + ymax2 * 0.02), xytext=(0, 0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS, color=MUTED) for xi, v in zip(x, insp)]
a2.set_ylim(0, ymax2 * 1.16)
a2.set_xlim(-0.6, len(BINS) - 0.4)
a2.set_xticks(x, TICK)
a2.set_ylabel("비중(%)")
a2.set_xlabel(XLABEL)
a2.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
style(a2)

fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.255)  # 왼쪽 여백: 두 줄 세로축 이름이 들어가게 0.075 → 0.10

# 짝 막대의 두 꼬리표가 붙지 않게: 두 글자 상자가 같은 높이에 걸치고 가로 틈이 MIN_GAP_PT 보다 좁으면
# 모자란 만큼 반씩 파랑은 왼쪽·회색은 오른쪽으로 비킴(90.1·84.5, 8.1·12.5 처럼 높이가 비슷한 짝). 저장 해상도에서 잼.
MIN_GAP_PT = 3.5
fig.set_dpi(DPI)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
px_per_pt = DPI / 72
for td, ti in zip(lab_d, lab_i):
    b1, b2 = td.get_window_extent(rend), ti.get_window_extent(rend)
    if min(b1.y1, b2.y1) - max(b1.y0, b2.y0) <= 0:
        continue
    gap_pt = (b2.x0 - b1.x1) / px_per_pt
    if gap_pt < MIN_GAP_PT:
        s = (MIN_GAP_PT - gap_pt) / 2
        td.xyann = (-s, 0)
        ti.xyann = (s, 0)

out = FIG / "I_5_item3_test_bins.png"
fig.savefig(out, dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- 출력
print("그림", out)
print(f"품목 {N_ITEM}개 · 검사 {N_ALL:,} · 검출 {P_ALL:,}")
for lb, n, p, d_, i_ in zip(BINS, n_item, item_pct, det, insp):
    print(f"{lb:>14} | 품목 {n:3d}개({p:4.1f}%) | 검출 비중 {d_:7.3f}% ({fmt(d_)}) | 검사 비중 {i_:7.3f}% ({fmt(i_)})")
