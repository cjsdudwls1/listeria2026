# -*- coding: utf-8 -*-
"""3차 연구 보고서 'I. 데이터 전처리 부문' 1.2 그림: 예측하는 날 T · 며칠 뒤 h · 맞히려는 날 d 와 기상 창.

자료 없음(날짜 규칙 그림). 규칙 출처(사실 모음 work3/facts_data.md 2.4·5):
  T = 예측하는 날, h = 1~60(1일 단위), d = 맞히려는 날(시료 기준일 mnfctur_de) = T + h
  기상 창 = T−80 ~ T−1 (80일, T 당일 제외), 기상 요약 = 10일 구간 평균 8개(가장 최근 구간 = T−10 ~ T−1) + 창 전체 통계
  보기 날짜 = 시험 첫 예측일 T = 2025-07-01 (h = 60 → d = 2025-08-30)
2026-10-22 rev1: 본문 폭(161mm)에 줄이지 않고 들어가게 6.3 x 2.35인치, 글자 9pt 이상(인쇄 크기 그대로), 그림 안 제목은 빼고
  (그림 아래 '[그림 n]' 제목이 대신함) 200dpi 로 저장.
2026-10-22 rev2(사용자 지시 — 억지 번역 대신 일반 용어): 그림 안 글자 '기상 창' → '기상 window'. 자료·모양·크기·dpi 는 그대로.
  글자가 길어져 T 선에 붙지 않게 그 꼬리표의 '(T-80 ~ T-1)' 만 '(T-80~T-1)' 로 띄어쓰기를 줄임.
출력: figs3/I_0_T_h_d.png (Malgun Gothic)
실행: <.venv-exp2 python> -B 05_fig_T_h_d.py
"""
import sys

sys.dont_write_bytecode = True

import datetime as dt  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIG = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\경복대양식\figs3")
FIG.mkdir(parents=True, exist_ok=True)

# 모양(02_eda_figs.py 와 같은 값, dataviz 기준 팔레트 1·2번)
BLUE = "#2a78d6"
BLUE_FILL = "#cde2fb"     # 파랑 단계 100
ORANGE = "#eb6834"
ORANGE_FILL = "#fbe0d5"   # 주황 옅은 면
INK = "#0b0b0b"
INK2 = "#52514e"
AXIS = "#c3c2b7"
FS = 9.5                  # 기본 글자(pt) — 그림을 줄이지 않고 넣으므로 인쇄 크기와 같음
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": FS,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

L = 80            # 기상 창(일)
H_MAX = 60        # 예측 거리 최대(일)
T0 = dt.date(2025, 7, 1)


def day(k):
    return (T0 + dt.timedelta(days=k)).isoformat()


# 날짜 계산 확인(사실 모음과 같은지)
assert day(-L) == "2025-04-12" and day(-1) == "2025-06-30"
assert day(1) == "2025-07-02" and day(H_MAX) == "2025-08-30"

fig, ax = plt.subplots(figsize=(6.3, 2.35))
fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.25)
ax.set_xlim(-L - 5, H_MAX + 5)
ax.set_ylim(0.12, 2.62)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_yticks([])
ax.tick_params(axis="x", length=3, color=AXIS, labelsize=FS, labelcolor=INK2)

# 하루 = [k-0.5, k+0.5] 칸
# 1) 기상 창 T-80 ~ T-1 (10일 구간 8개)
y_w, hh = 1.88, 0.40
ax.add_patch(Rectangle((-L - 0.5, y_w - hh / 2), L, hh, facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=1.2, zorder=2))
for k in range(1, L // 10):
    xb = -10 * k - 0.5
    ax.plot([xb, xb], [y_w - hh / 2, y_w + hh / 2], color=BLUE, linewidth=0.7, zorder=3)
ax.text(-L - 0.5, y_w + hh / 2 + 0.07, "입력: 기상 window 80일(T-80~T-1), 10일 구간 8개", ha="left", va="bottom", fontsize=FS,
        color=INK, fontweight="bold")

# 2) 맞히려는 날 범위 T+1 ~ T+60
y_d = 1.02
ax.add_patch(Rectangle((0.5, y_d - hh / 2), H_MAX, hh, facecolor=ORANGE_FILL, edgecolor=ORANGE, linewidth=1.2, zorder=2))
ax.text(H_MAX + 0.5, y_d + hh / 2 + 0.07, "정답: 맞히려는 날 d = T + h", ha="right", va="bottom", fontsize=FS, color=INK,
        fontweight="bold")
for h in (1, 30, 60):
    ax.plot([h], [y_d], marker="o", markersize=6.5, markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.5, zorder=5)
for h in (30, 60):
    ax.text(h, y_d - hh / 2 - 0.05, f"h={h}", ha="center", va="top", fontsize=FS, color=INK2)
ax.text(1 + 1.5, y_d - hh / 2 - 0.05, "h=1", ha="left", va="top", fontsize=FS, color=INK2)

# 3) h 화살표(T → d) — 기상 창 아래 왼쪽 빈 자리에 설명
y_a = 0.32
ax.add_patch(FancyArrowPatch((0, y_a), (H_MAX, y_a), arrowstyle="-|>", mutation_scale=10, color=INK2, linewidth=1.1, zorder=4))
ax.text(-2, y_a, "h = 며칠 뒤를 맞히는지(1 ~ 60일)", ha="right", va="center", fontsize=FS, color=INK2)

# 4) T 선
ax.plot([0, 0], [0.22, 2.45], color=INK, linewidth=1.4, zorder=6)
ax.text(0, 2.48, "T = 예측하는 날", ha="center", va="bottom", fontsize=FS, color=INK, fontweight="bold")

# 5) 축 눈금: 상대 날짜 + 보기 날짜(시험 첫 예측일)
ticks = [-L, -40, 0, 30, H_MAX]
names = {-L: "T-80", -40: "T-40", 0: "T", 30: "T+30", H_MAX: "T+60"}
ax.set_xticks(ticks)
ax.set_xticklabels([f"{names[k]}\n{day(k)}" for k in ticks])

out = FIG / "I_0_T_h_d.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print("그림", out)
