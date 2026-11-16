# -*- coding: utf-8 -*-
"""보고서 그림(PNG) — 숫자는 facts.json(원 파일에서 읽은 값)만 씀."""
import datetime as dt
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

W = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(W, "figs")
os.makedirs(OUT, exist_ok=True)
F = json.load(open(os.path.join(W, "facts.json"), encoding="utf-8"))
R = F["runs"]

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "font.size": 9.5,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e4e3de",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "axes.axisbelow": True,
})
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, LGRAY = "#a9a8a2", "#d9d8d2"
FIG_W = 6.3  # inch = 160 mm (본문 폭 안)


def val(run):
    return R[run]["PR-AUC 구간 가중 val"]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ 1. T·h·d·window
def fig_window():
    d = dt.date(2025, 11, 1)
    Wd = 60
    cases = [5, 30]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.25))
    ax.grid(False)
    for row, h in enumerate(cases):
        y = 1 - row
        T = d - dt.timedelta(days=h)
        E = min(d, T + dt.timedelta(days=10))
        S = E - dt.timedelta(days=Wd - 1)
        obs_days = (T - S).days
        fc_days = (E - T).days + 1
        g = (d - E).days
        ax.barh(y, obs_days, left=mdates.date2num(S), height=0.36, color=BLUE, edgecolor="white", linewidth=1.5)
        ax.barh(y, fc_days, left=mdates.date2num(T), height=0.36, color=ORANGE, edgecolor="white", linewidth=1.5)
        if g > 0:
            ax.barh(y, g, left=mdates.date2num(E + dt.timedelta(days=1)), height=0.36, color="white",
                    edgecolor=GRAY, linewidth=1.0, hatch="///")
        ax.text(mdates.date2num(S) + obs_days / 2, y, f"관측 {obs_days}일", ha="center", va="center", color="white", fontsize=8.5)
        ax.text(mdates.date2num(T) + fc_days / 2, y + 0.30, f"예보 {fc_days}일", ha="center", va="bottom", color=INK, fontsize=8)
        if g > 0:
            ax.text(mdates.date2num(E + dt.timedelta(days=1)) + g / 2, y, f"window 밖 g = {g}일", ha="center",
                    va="center", color=INK, fontsize=8, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        close = (d - T).days < 15
        for dd, lab in ((T, "T"), (d, "d")):
            x = mdates.date2num(dd) + (0 if lab == "T" else 1)
            ax.plot([x, x], [y - 0.28, y + 0.28], color=INK, lw=1.2)
            ha = "center"
            if close:
                ha = "right" if lab == "T" else "left"
            ax.text(x, y - 0.30, f"{lab} {dd.strftime('%m-%d')}", ha=ha, va="top", fontsize=8, color=INK)
        ax.text(mdates.date2num(S) - 1.5, y, f"h = {h}", ha="right", va="center", fontsize=9, color=INK)
    ax.set_ylim(-0.75, 1.65)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(mdates.date2num(dt.date(2025, 8, 2)), mdates.date2num(dt.date(2025, 11, 6)))
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=BLUE, label="관측 칸(T 전날까지)"),
               Patch(color=ORANGE, label="예보 칸(T ~ T+10, 학습·검증·평가에는 관측값)"),
               Patch(facecolor="white", edgecolor=GRAY, hatch="///", label="window 밖(g일, 기상 입력 없음)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=8, handlelength=1.6)
    save(fig, "fig_window.png")


# ------------------------------------------------------------------ 2. 나누기
def fig_split():
    fs = F["final_scores"]
    folds = fs["cv"]["folds"]
    test = fs["test"]["report"]
    rows = [
        ("fold 1", dt.date(2022, 6, 30), (dt.date(2022, 7, 1), dt.date(2023, 5, 1)), folds["fold1"], "검증"),
        ("fold 2", dt.date(2023, 6, 30), (dt.date(2023, 7, 1), dt.date(2024, 5, 1)), folds["fold2"], "검증"),
        ("fold 3", dt.date(2024, 6, 30), (dt.date(2024, 7, 1), dt.date(2025, 5, 1)), folds["fold3"], "검증"),
        ("최종·평가", dt.date(2025, 6, 30), (dt.date(2025, 7, 1), dt.date(2026, 5, 1)), test, "평가"),
    ]
    start = dt.date(2015, 4, 7)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    ax.grid(axis="y", visible=False)
    for i, (lab, tr_end, (s, e), sc, kind) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, (tr_end - start).days + 1, left=mdates.date2num(start), height=0.42, color=BLUE,
                edgecolor="white", linewidth=1.5)
        col = ORANGE if kind == "검증" else AQUA
        ax.barh(y, (e - s).days + 1, left=mdates.date2num(s), height=0.42, color=col, edgecolor="white", linewidth=1.5)
        ax.text(mdates.date2num(start) + 40, y, f"학습 ~ {tr_end.isoformat()}", va="center", ha="left",
                color="white", fontsize=8)
        ax.text(mdates.date2num(e) + 25, y, f"{kind} {sc['n_pairs']:,}쌍 · 검출 {sc['n_pos']:,}쌍", va="center",
                ha="left", color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(mdates.date2num(dt.date(2015, 1, 1)), mdates.date2num(dt.date(2029, 6, 30)))
    ax.set_xticks([mdates.date2num(dt.date(yy, 1, 1)) for yy in range(2015, 2027)])
    handles = [Patch(color=BLUE, label="학습(시료 기준일 d)"), Patch(color=ORANGE, label="검증(예측하는 날 T)"),
               Patch(color=AQUA, label="평가(예측하는 날 T)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, fontsize=8)
    save(fig, "fig_split.png")


# ------------------------------------------------------------------ 3. 1차 70개 설정
def fig_stage1():
    Ws = list(range(10, 101, 10))
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.9), sharey=True)
    algos = [("CAT", "CatBoost", BLUE), ("XGB", "XGBoost", ORANGE), ("LGB", "LightGBM", AQUA)]
    best = max(((val(n), n) for n in R if n.startswith("1-")), key=lambda t: t[0])
    for k, (mode, title) in enumerate((("sum", "(가) 기상 요약(트리)·기상 시퀀스(LSTM)"), ("raw", "(나) 기상 일별 원값(트리)"))):
        ax = axes[k]
        for code, name, col in algos:
            ys = [val(f"1-{code}-W{w:03d}-{mode}") for w in Ws]
            ax.plot(Ws, ys, color=col, lw=1.6, marker="o", ms=3.2, label=name)
        if mode == "sum":
            ys = [val(f"1-LSTM-W{w:03d}") for w in Ws]
            ax.plot(Ws, ys, color=YELLOW, lw=1.6, marker="o", ms=3.2, label="LSTM")
        if best[1].endswith(mode) or (mode == "sum" and "LSTM" in best[1]):
            w_best = int(best[1].split("-W")[1][:3])
            ax.plot([w_best], [best[0]], marker="o", ms=9, mfc="none", mec=INK, mew=1.1)
            ax.annotate(f"1위 {best[0]:.4f}", (w_best, best[0]), xytext=(9, -3), textcoords="offset points",
                        fontsize=8.5, color=INK, va="top")
        ax.set_title(title, fontsize=9, color=INK, pad=4)
        ax.set_xticks(Ws)
        ax.set_xlabel("window 길이(일)")
    axes[0].set_ylabel("검증 구간 가중 PR-AUC")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_stage1.png")


# ------------------------------------------------------------------ 4. 최종 h 별
def fig_final_h():
    fs = F["final_scores"]
    folds = fs["cv"]["folds"]
    hs = list(range(1, 61))
    cv = [sum(folds[f]["pr_auc_by_h"][i] for f in ("fold1", "fold2", "fold3")) / 3 for i in range(60)]
    te = fs["test"]["report"]["pr_auc_by_h"]
    hm = {int(r["h"]): r for r in F["final_h"] if r["split"] == "test"}
    rec = [float(hm[h]["recall"]) for h in hs]
    pre = [float(hm[h]["precision"]) for h in hs]
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.35, 1, 1]})
    for ax in axes:
        for a, b, c in ((0.5, 20.5, "#f3f2ee"), (40.5, 60.5, "#f3f2ee")):
            ax.axvspan(a, b, color=c, zorder=0, lw=0)
    axes[0].plot(hs, cv, color=ORANGE, lw=1.6, label="검증(3 fold 평균)")
    axes[0].plot(hs, te, color=BLUE, lw=1.6, label="평가")
    axes[0].set_ylabel("PR-AUC")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    for txt, x in (("h 1 ~ 20 (가중 0.5)", 10.5), ("h 21 ~ 40 (가중 0.3)", 30.5), ("h 41 ~ 60 (가중 0.2)", 50.5)):
        axes[0].text(x, 1.02, txt, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                     color=INK2)
    axes[1].plot(hs, rec, color=BLUE, lw=1.6)
    axes[1].axhline(0.8, color=INK2, lw=0.9, ls="--")
    axes[1].text(0.995, 0.8, "Recall 0.80", transform=axes[1].get_yaxis_transform(), va="bottom", ha="right",
                 fontsize=7.5, color=INK2)
    axes[1].set_ylabel("Recall(평가)")
    axes[2].plot(hs, pre, color=BLUE, lw=1.6)
    axes[2].set_ylabel("Precision(평가)")
    axes[2].set_xlabel("h(며칠 뒤를 맞히는지)")
    axes[2].set_xticks([1, 10, 20, 30, 40, 50, 60])
    axes[2].set_xlim(0.5, 60.5)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig_final_h.png")


GROUP_NAME = {
    "hist_rate5y_p3": "5년 검출률(품목)", "hist_rate5y_sgg": "5년 검출률(지역)", "hist_rate5y_u": "5년 검출률(단위)",
    "hist_rate1y_p3": "1년 검출률(품목)", "hist_rate1y_sgg": "1년 검출률(지역)", "hist_rate1y_u": "1년 검출률(단위)",
    "hist_logn5y_p3": "5년 로그 검사 수(품목)", "hist_logn5y_sgg": "5년 로그 검사 수(지역)", "hist_logn5y_u": "5년 로그 검사 수(단위)",
    "hist_logn1y_p3": "1년 로그 검사 수(품목)", "hist_logn1y_sgg": "1년 로그 검사 수(지역)", "hist_logn1y_u": "1년 로그 검사 수(단위)",
    "woe_sido": "WOE(시도)", "woe_sgg": "WOE(지역)", "woe_p2": "WOE(품목 2단계)", "woe_p3": "WOE(품목 3단계)",
    "hist_unc1y_p3": "1년 검출률 불확실성(품목)", "hist_unc1y_sgg": "1년 검출률 불확실성(지역)",
    "hist_unc1y_u": "1년 검출률 불확실성(단위)",
    "food_aw": "수분활성도", "food_ph": "식품 pH", "food_na": "염도",
    "rg_aw_lim": "수분활성도 생장 한계 안", "rg_aw_opt_dist": "수분활성도 최적값 거리",
    "rg_ph_lim": "식품 pH 생장 한계 안", "rg_ph_opt_dist": "식품 pH 최적값 거리",
    "soil_ph": "토양 pH", "soil_om": "토양 유기물", "soil_vldpha": "토양 유효인산",
    "rg_soilph_opt": "토양 pH 최적 범위 안", "rg_soilph_lim": "토양 pH 한계 범위 안",
    "rg_soilph_opt_dist": "토양 pH 최적값 거리",
    "eco_ratio": "친환경 인증 비율", "gw_pass_rate": "지하수 수질검사 적합률", "L1": "축사 밀도",
    "생육확률(Koutsoumanis)": "Koutsoumanis 생장식",
}
WEATHER = {"기온", "강수", "바람", "햇빛", "지중온도", "토양수분", "습도", "대기질", "계절"}


def group_cat(g):
    if g in WEATHER:
        return "기상"
    if g.startswith("hist_") or g.startswith("woe_"):
        return "검출 이력"
    if g in ("eco_ratio", "gw_pass_rate", "L1"):
        return "지역 지표·축사"
    if g == "생육확률(Koutsoumanis)":
        return "Koutsoumanis 생장식"
    return "식품·토양"


CAT_COL = {"기상": BLUE, "검출 이력": ORANGE, "식품·토양": AQUA, "지역 지표·축사": VIOLET, "Koutsoumanis 생장식": MAGENTA}


def fig_importance(top=20):
    rows = [r for r in F["importance"] if r["target_kind"] == "group"]
    rows = sorted(rows, key=lambda r: float(r["mean_I"]), reverse=True)[:top]
    names = [GROUP_NAME.get(r["target"], r["target"]) for r in rows]
    vals = [float(r["mean_I"]) for r in rows]
    cats = [group_cat(r["target"]) for r in rows]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.9))
    ax.grid(axis="y", visible=False)
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, vals, color=[CAT_COL[c] for c in cats], height=0.68, edgecolor="white", linewidth=1.0)
    for y, v in zip(ys, vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.4f}", va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("섞었을 때 검증 구간 가중 PR-AUC 평균 하락")
    ax.set_xlim(0, max(vals) * 1.13)
    seen = []
    for c in cats:
        if c not in seen:
            seen.append(c)
    order = [c for c in CAT_COL if c in seen]
    ax.legend(handles=[Patch(color=CAT_COL[c], label=c) for c in order], loc="lower right", fontsize=8)
    save(fig, "fig_importance.png")


# ------------------------------------------------------------------ 6. 전체 흐름
def fig_overview():
    S = F["summary"]
    items = [
        ("1차 베이스라인(CatBoost·기상 요약·60일)", "1-CAT-W060-sum", "채택"),
        ("2차: 13개 구조 1위(LSTM)", "3-0-LSTM", "채택"),
        ("결합: 시계열만 LSTM + attention", "x6-LSTMA", "버림"),
        ("결합: 시계열만 트랜스포머", "x7-D", "버림"),
        ("결합: 랜덤포레스트 + 이상 점수", "3-1-RF-VAE", "버림"),
        ("결합: CatBoost + 이상 점수", "3-2-CAT", "버림"),
        ("결합: LIMS 만 스태킹", "x4-L6", "버림"),
        ("결합: 트리 앙상블", "3-3-T3", "버림"),
        ("3차: 변수 빼기", "3-0b-LSTM", "버림"),
        ("비교: 정형만(기상 뺌)", "x8-RF", "비교"),
        ("다운사이징: Theil's U 기반", "4-a0-LSTM", "버림"),
        ("다운사이징: 60칸 출력 신경망", "x2-NN60", "버림"),
        ("다운사이징: 검출 칸 음성 샘플링", "6-LSTM", "버림"),
        ("최종 실험: 연도별 무작위 언더샘플링\n+ 식품 특성 갱신", "5-LSTM-food2", "버림"),
        ("최종: 위 + Koutsoumanis 생장식", "5-LSTM-KO", "채택"),
    ]
    col = {"채택": BLUE, "버림": GRAY, "비교": LGRAY}
    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    ax.grid(axis="y", visible=False)
    n = len(items)
    ys = list(range(n))[::-1]
    vals = [val(run) for _, run, _ in items]
    ax.barh(ys, vals, color=[col[k] for _, _, k in items], height=0.66, edgecolor="white", linewidth=1.0)
    ref = val("3-0-LSTM")
    for y, v, (_, _, k) in zip(ys, vals, items):
        if v >= ref - 0.0002:
            ax.text(v + 0.0004, y, f"{v:.4f}", va="center", ha="left", fontsize=7.5, color=INK)
        elif v > 0.012:
            ax.text(v - 0.0004, y, f"{v:.4f}", va="center", ha="right", fontsize=7.5,
                    color=(INK if k == "비교" else "white"))
        else:
            ax.text(v + 0.0004, y, f"{v:.4f}", va="center", ha="left", fontsize=7.5, color=INK)
    ref = val("3-0-LSTM")
    ax.axvline(ref, color=INK2, lw=0.9, ls="--", zorder=3)
    ax.text(ref + 0.0003, n - 0.45, f"2차 모델 {ref:.4f}", ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _, _ in items], fontsize=8)
    ax.set_xlabel("검증 구간 가중 PR-AUC")
    ax.set_xlim(0, 0.041)
    ax.set_ylim(-0.7, n + 0.3)
    handles = [Patch(color=BLUE, label="채택"), Patch(color=GRAY, label="채택 안 함"),
               Patch(color=LGRAY, label="비교만")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.36, 1.0), ncol=3, fontsize=8)
    save(fig, "fig_overview.png")


if __name__ == "__main__":
    fig_window()
    fig_split()
    fig_stage1()
    fig_final_h()
    fig_importance()
    fig_overview()
