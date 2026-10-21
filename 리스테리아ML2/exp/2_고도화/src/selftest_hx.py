# -*- coding: utf-8 -*-
"""HX 과거 이력 추가 칸(logn1·sd1, 실험 A3_L1·A3_U1·A3_L1U1) 자체 점검 (전체 자료). 결과: logs/selftest_hx.txt (UTF-8),
날짜 시계 확인표 logs/selftest_hx_dateclock.csv. 실패가 있으면 종료 코드 1.
results/·tables/·cache/ 에는 쓰지 않는다(기상 요약 캐시는 메타가 같을 때만 읽기, k* 는 results/selection_k.json 읽기만).

(0) 설정: 실험 3개 단계 'hx'·시드 42·43·44, 입력 = A3 + 묶음(logn1·sd1), 칸 이름, STAGES 에서 'hx' 바로 뒤 'ae', 둘 다 'tables'(끝) 앞(2026-10-19 xs·stk 추가 뒤),
    run_exp2.ps1 ValidateSet 에 'hx'
(a) 무작위 (행, h) 300쌍(분할마다 100) + 자료 시작 직후 쌍: 시료 parquet 원본을 따로 읽어 반복문으로 창(기준일 ∈ [T−365, T−1]) 안
    그 수준 시료 수 n·검출 s, 전국 N·S 를 세어 logn1 = log(1+n), sd1 = sqrt(m(1−m)/(n+k+1)), m = (s + k·S/N)/(n + k) 를 구해 대조(상대 1e-6).
    rate1(A3 칸)도 같은 m 인지 대조. 대조군: 창에 T 당일까지 넣으면 달라지는 쌍이 있음. 한 쌍씩(h 정수) = 한꺼번에(h 배열) 비트 단위
(b) 기존 칸 그대로: 고치기 전 e2hist(scratch/ae/e2hist_before_ae.py.bak) hist_features 와 기존 묶음(rate5·logn5·rate1) 값 비트 단위 같음
    (분할마다 3,000행, h 배열·정수), A3_L1·A3_U1·A3_L1U1 특성 행렬 = A3 특성 행렬(비트 단위) + 새 칸(= hist_features 새 묶음 값)
(c) 날짜 시계 확인: logn1·sd1(3수준, 참고로 A3 의 logn5 도)의 예측일 T 달력 연도별 평균 - 학습 = 시드 42 h, 검증·시험 = h 1~60 전부
    -> logs/selftest_hx_dateclock.csv. 학습 연도 평균이 해마다 늘어나는지 적음(기록, 실패 조건 아님)
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import config as C1  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2hist import HX_DATECLOCK_CSV, hist_features, hist_names  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_hx.txt"
SCR = E.EXP2_DIR / "scratch" / "ae"
OLD_E2HIST = SCR / "e2hist_before_ae.py.bak"
HX = {"A3_L1": ["logn1"], "A3_U1": ["sd1"], "A3_L1U1": ["logn1", "sd1"]}
lines, fails = [], []


def say(msg):
    lines.append(msg)
    try:
        print(msg, flush=True)
    except Exception:
        pass


def check(name, ok, detail=""):
    say(f"[{'통과' if ok else '실패'}] {name} {detail}")
    if not ok:
        fails.append(name)


def same(a, b):
    return (np.isnan(a) and np.isnan(b)) or (not np.isnan(a) and not np.isnan(b) and abs(a - b) <= 1e-6 * max(1.0, abs(b)))


def k_star():
    return json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]


def t_config():
    a3 = E.EXP_BY_ID["A3"]["feats"]
    ok = all(E.EXP_BY_ID[i]["stage"] == "hx" and E.EXP_BY_ID[i]["model"] == "cat"
             and E.EXP_BY_ID[i]["feats"] == dict(a3, hist=a3["hist"] + b) and E.exp_seeds(E.EXP_BY_ID[i], False) == [42, 43, 44]
             for i, b in HX.items())
    check("(0) A3_L1·A3_U1·A3_L1U1: 단계 'hx', 입력 = A3 + logn1 / sd1 / 둘 다, 시드 42·43·44", ok)
    names = hist_names(["logn1", "sd1"])
    check("(0) 칸 이름 hist_logn1y_<p3|sgg|unit>·hist_sd1y_<p3|sgg|unit>",
          names == [f"hist_{a}1y_{lv}" for a in ("logn", "sd") for lv in E.HIST_LEVELS], str(names))
    st = E.STAGES
    check("(0) STAGES: 'hx' -> 'ae' 이어서, 'tables' 가 끝(사이에 xs·stk 추가됨)",
          st.index("ae") == st.index("hx") + 1 and st.index("ae") < st.index("tables") == len(st) - 1, str(st[-6:]))
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'hx'·'ae'", bool(vs) and "'hx'" in vs[0] and "'ae'" in vs[0])


class Raw:
    """시료 parquet 원본(반복문 대조용): 1차 load_samples 를 쓰지 않고 따로 읽음(빈 코드 = '코드없음')."""

    def __init__(self):
        df = pq.read_table(C1.SAMPLE_PATH, columns=C1.KEY_COLS + ["mnfctur_de", "judge_result"]).to_pandas()
        self.sgg = df["sgg_code"].astype(str).to_numpy()
        self.p2 = df["train_prdlst_code"].astype(object).where(df["train_prdlst_code"].notna(), C1.NULL_CODE).astype(str).to_numpy()
        self.p3 = df["train_prdlst_code_2"].astype(object).where(df["train_prdlst_code_2"].notna(), C1.NULL_CODE).astype(str).to_numpy()
        self.d = df["mnfctur_de"].values.astype("datetime64[D]")
        self.pos = (df["judge_result"].astype(str) == "검출").to_numpy()

    def values(self, row, T, k, incl_today=False):
        """{lv: (logn1, sd1, rate1)} - 창 기준일 ∈ [T−365, T−1] (incl_today 면 [T−365, T])."""
        hi = T if incl_today else T - np.timedelta64(1, "D")
        w = (self.d >= T - np.timedelta64(E.P1, "D")) & (self.d <= hi)
        N, S = int(w.sum()), int((w & self.pos).sum())
        p0 = S / N if N > 0 else float("nan")
        m_lv = {"p3": self.p3 == row["train_prdlst_code_2"], "sgg": self.sgg == row["sgg_code"],
                "unit": (self.sgg == row["sgg_code"]) & (self.p2 == row["train_prdlst_code"]) & (self.p3 == row["train_prdlst_code_2"])}
        out = {}
        for lv in E.HIST_LEVELS:
            g = w & m_lv[lv]
            n, s = int(g.sum()), int((g & self.pos).sum())
            m = (s + k * p0) / (n + k)
            out[lv] = (float(np.log1p(n)), float(np.sqrt(m * (1 - m) / (n + k + 1))), float(m))
        return out


def t_values(ctx, rng, raw, k):
    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        picks += [(sp, int(j), int(rng.integers(1, E.H_MAX + 1))) for j in rng.choice(n, 100, replace=False)]
    hd = ctx.sp["train"]["hd"]                                   # 자료 시작 직후(T 가 이력 날짜 0~400 번째) 쌍 - p0 결측·창이 잘리는 곳
    for j in np.flatnonzero(hd < 460)[:: max(1, int((hd < 460).sum()) // 30)][:30]:
        picks.append(("train", int(j), int(rng.integers(1, E.H_MAX + 1))))
    sets = ["rate1", "logn1", "sd1"]
    names = hist_names(sets)
    col = {nm: i for i, nm in enumerate(names)}
    by = {}
    for sp, j, h in picks:
        by.setdefault(sp, []).append((j, h))
    batch = {}
    for sp, v in by.items():
        pos = np.array([j for j, _ in v])
        grp = {lv: g[pos] for lv, g in ctx.sp[sp]["grp"].items()}
        batch[sp] = hist_features(ctx.H, grp, ctx.sp[sp]["hd"][pos], np.array([h for _, h in v]), sets, k)
    cnt = {sp: 0 for sp in by}
    bad, bad_batch, diff_today, n_nan = 0, 0, 0, 0
    first = None
    for sp, j, h in picks:
        r = ctx.rows.iloc[ctx.sp[sp]["idx"][j]]
        T = np.datetime64(r["d"], "D") - np.timedelta64(h, "D")
        grp = {lv: g[[j]] for lv, g in ctx.sp[sp]["grp"].items()}
        got = hist_features(ctx.H, grp, ctx.sp[sp]["hd"][[j]], h, sets, k)[0]
        bad_batch += int(not np.array_equal(got, batch[sp][cnt[sp]], equal_nan=True))
        cnt[sp] += 1
        ref = raw.values(r, T, k)
        ref2 = raw.values(r, T, k, incl_today=True)
        for lv in E.HIST_LEVELS:
            g_l, g_s, g_r = (float(got[col[f"hist_{a}1y_{lv}"]]) for a in ("logn", "sd", "rate"))
            ok = same(g_l, ref[lv][0]) and same(g_s, ref[lv][1]) and same(g_r, ref[lv][2])
            n_nan += int(np.isnan(g_s))
            if not ok:
                bad += 1
                first = first or (sp, j, h, lv, str(T), (g_l, g_s, g_r), ref[lv])
            diff_today += int(not (same(g_l, ref2[lv][0]) and same(g_s, ref2[lv][1])))
    n = len(picks) * len(E.HIST_LEVELS)
    check("(a) logn1·sd1·rate1 = 시료 원본 반복문(창 기준일 ∈ [T−365, T−1], m = (s + k·S/N)/(n + k))", bad == 0,
          f"불일치 {bad}/{n}(쌍 {len(picks)} x 3수준), sd1 결측 {n_nan}(전국 창 시료 0 - p0 결측), k = {k}")
    check("(a) 대조군: 창에 T 당일까지 넣으면 달라지는 (쌍, 수준)이 있음", diff_today > 0, f"{diff_today}/{n}")
    check("(a) 한 쌍씩(h 정수) = 한꺼번에(h 배열) 비트 단위", bad_batch == 0, f"불일치 {bad_batch}/{len(picks)}")
    if first:
        say(f"    [첫 불일치] {first}")


def _load_old():
    ld = importlib.machinery.SourceFileLoader("e2hist_old_ae", str(OLD_E2HIST))
    spec = importlib.util.spec_from_loader("e2hist_old_ae", ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


def t_old_same(ctx, rng, k):
    O = _load_old()
    old_sets = ["rate5", "logn5", "rate1"]
    bad, tot = [], 0
    for sp in C1.SPLITS:
        pos = rng.choice(len(ctx.sp[sp]["idx"]), 3000, replace=False)
        grp = {lv: g[pos] for lv, g in ctx.sp[sp]["grp"].items()}
        hd = ctx.sp[sp]["hd"][pos]
        hh = rng.integers(1, E.H_MAX + 1, size=len(pos))
        for h in (hh, int(hh[0])):
            for kk in (k, 30):
                tot += 1
                a = O.hist_features(ctx.H, grp, hd, h, old_sets, kk)
                b = hist_features(ctx.H, grp, hd, h, old_sets, kk)
                if not np.array_equal(a, b, equal_nan=True):
                    bad.append((sp, "정수" if np.isscalar(h) else "배열", kk))
        ow = O.woe_features(ctx.H, grp, hd, hh)
        nw = __import__("e2hist").woe_features(ctx.H, grp, hd, hh)
        if not np.array_equal(ow, nw, equal_nan=True):
            bad.append((sp, "WOE"))
    check("(b) 기존 이력 묶음(rate5·logn5·rate1)·WOE 값 = 고치기 전 e2hist (비트 단위)", not bad, f"불일치 {bad} / {tot}")
    check("(b) 기존 칸 이름 = 고치기 전", O.hist_names(old_sets) == hist_names(old_sets) and O.woe_names() == __import__("e2hist").woe_names())
    from e2models import FeatureBuilder
    b0 = FeatureBuilder(ctx, dict(E.EXP_BY_ID["A3"]["feats"], k=k))
    for eid, blk in HX.items():
        b1 = FeatureBuilder(ctx, dict(E.EXP_BY_ID[eid]["feats"], k=k), seed=42)
        bad = []
        for sp in C1.SPLITS:
            pos = rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False)
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            grp = {lv: g[pos] for lv, g in ctx.sp[sp]["grp"].items()}
            for h in (hh, int(hh[0])):
                X0, X1 = b0.num(sp, pos, h), b1.num(sp, pos, h)
                add = hist_features(ctx.H, grp, ctx.sp[sp]["hd"][pos], h, blk, k)
                if not (X1.shape[1] == X0.shape[1] + add.shape[1] and np.array_equal(X1[:, :X0.shape[1]], X0, equal_nan=True)
                        and np.array_equal(X1[:, X0.shape[1]:], add, equal_nan=True)):
                    bad.append((sp, "정수" if np.isscalar(h) else "배열"))
        check(f"(b) {eid} 특성 행렬 = A3 행렬(비트 단위) + {blk} {3 * len(blk)}칸",
              b1.names == b0.names + hist_names(blk) and not bad, f"불일치 {bad}, 칸 {len(b1.names)}")


def t_dateclock(ctx, k):
    sets = ["logn1", "sd1", "logn5"]
    names = hist_names(sets)
    recs = []
    H = ctx.H
    for sp in C1.SPLITS:
        v = ctx.sp[sp]
        n = len(v["idx"])
        hs = [ctx.h_fit(42)["train"].astype(np.int64)] if sp == "train" else [np.full(n, h, dtype=np.int64) for h in range(1, E.H_MAX + 1)]
        acc = {}
        for hh in hs:
            X = hist_features(H, v["grp"], v["hd"], hh, sets, k).astype(np.float64)
            yr = (H.base + (v["hd"] - hh).astype("timedelta64[D]")).astype("datetime64[Y]").astype(int) + 1970
            for y_ in np.unique(yr):
                m = yr == y_
                a = acc.setdefault(int(y_), [0, np.zeros(len(names)), np.zeros(len(names))])
                ok = ~np.isnan(X[m])
                a[0] += int(m.sum())
                a[1] += np.where(ok, X[m], 0.0).sum(0)
                a[2] += ok.sum(0)
        for y_ in sorted(acc):
            n_, s_, c_ = acc[y_]
            recs.append({"split": sp, "year": y_, "n": n_, **{nm: s_[i] / c_[i] if c_[i] else np.nan for i, nm in enumerate(names)}})
    df = pd.DataFrame(recs)
    HX_DATECLOCK_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HX_DATECLOCK_CSV, index=False, encoding="utf-8-sig")
    say(f"    [날짜 시계 확인표] 예측일 T 달력 연도별 평균 (학습 = 시드 42 h, 검증·시험 = h 1~60 전부의 (행, h)) -> {HX_DATECLOCK_CSV}")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4f}"):
        say(df.to_string(index=False))
    tr = df[df["split"] == "train"].sort_values("year")
    for nm in names:
        v = tr[nm].to_numpy()
        up = int(np.sum(np.diff(v) > 0))
        say(f"    [기록] {nm}: 학습 {int(tr['year'].iloc[0])} {v[0]:.4f} -> {int(tr['year'].iloc[-1])} {v[-1]:.4f}, 해마다 늘어난 해 {up}/{len(v) - 1}"
            + (" - 해마다 늘어남(날짜 시계 의심)" if up == len(v) - 1 else ""))
    check("(c) 날짜 시계 확인표 작성(학습·검증·시험 모두 있음)", set(df["split"]) == set(C1.SPLITS) and HX_DATECLOCK_CSV.exists())


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_hx.log")
    rng = np.random.default_rng(20261019)
    t_config()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    k = k_star()
    raw = Raw()
    t_values(ctx, rng, raw, k)
    t_old_same(ctx, rng, k)
    t_dateclock(ctx, k)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
