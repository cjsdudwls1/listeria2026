# -*- coding: utf-8 -*-
"""2차 자체 점검 (전체 자료). 결과: logs/selftest.txt (UTF-8). 하나라도 실패하면 종료 코드 1.

(1) h 배정: 시드 42 = 1차 data.assign_h (학습 42, 조기 종료 검증 43), 시드 43·44 재현·서로 다름, H10 (중복 없는 10개, 재현)
(2) 이력 변수: 무작위 (행, h) 와 'T 당일에 같은 단위 시료가 있는' (행, h) 에서 단순 계산(무리별 시료 날짜를 직접 거름)과 대조.
    T 당일·이후 자료 제외 확인('T 포함' 으로 세면 값이 달라지는 경우가 실제로 있는지도 확인), p0 없는 창(NaN)
(3) WOE: 같은 방식으로 대조
(4) 검증·시험 기상 요약 캐시 = 그 자리에서 1차 build_tree_matrix 로 만든 값
(5) 고위험 품목 묶음 = 사양의 예상 코드
(6) 구간 가중 계산 + 1차 summary 의 60일 평균과 같은지
(7) SMOTE('시도 안 이웃 + 기준 행 범주', 사용자 결정 2026-10-19) - 작은 가짜 자료와 실제 학습 자료(D2 시드 42 입력) 둘 다:
    만들 개수, (a) 시도 안 이웃: 합성 행마다 기준 행·이웃이 같은 시도(전국 대체 시도 제외), 이웃 = 그 시도 검출 행 중
    반복문 거리로 가장 가까운 5개(전국 대체 시도는 전국에서), (b) 합성 행 범주 = 기준 행 범주(100%), (c) 숫자 칸 보간식,
    난수 순서(기준 행·이웃 칸·비율)가 이전 방식과 같은지(scratch 백업 e2smote_before_smote_fix.py.bak 이 있으면)
(12) 고위험품목비교 추가 줄(전체 모델, 기준값 = 그 품목 검증 행에서 다시 구함): 기준값·지표를 저장 예측에서 직접 다시 계산해 대조
     (전체 결과 results/ 를 읽기만 함), 기존 줄은 tables/고위험품목비교.csv 와 같은지
(8) H60 방식(메모리맵 열 우선 + quantize)과 보통 방식(메모리 행 우선)의 예측 비교 (작은 부분집합)
(9) E00 시드 42 를 끝까지 실행해 1차 catboost__summary__L050 예측·지표와 같은지 (results/_selftest, 매번 지우고 새로 실행)
    + config.json 에 코드 지문(code_md5)이 남는지, 코드가 바뀐 뒤 건너뛰면 [주의] 로그가 나오는지
(10) 여러 h 임시 파일: 양자화 중 오류가 나면 파일을 지우는지, 죽은 pid 가 남긴 파일만 지우는지
(11) 잠금(run.lock): 살아 있는 run.py 가 쥐면 종료 코드 3, 죽은 잠금은 넘겨받고, 끝나면 지움
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C1  # noqa: E402
import data as D  # noqa: E402
import features as F  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem, read_preds, summarize, weighted  # noqa: E402
from e2hist import hist_features, woe_features  # noqa: E402
from e2smote import smotenc  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest.txt"
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


def t_h(ctx):
    rows1 = D.assign_h(D.build_unit_rows(D.load_samples()))
    for sp in ("train", "val"):
        h1 = rows1.loc[rows1["split"] == sp, "h_fit"].to_numpy()
        h2 = ctx.h_fit(42)[sp]
        check(f"(1) 시드 42 {sp} h = 1차 h_fit", np.array_equal(h1, h2), f"({len(h1):,}행)")
    a, b = ctx.h_fit(43), ctx.h_fit(43)
    c = ctx.h_fit(44)
    check("(1) 시드 43 재현", all(np.array_equal(a[s], b[s]) for s in a))
    check("(1) 시드 42·43·44 학습 h 서로 다름", not np.array_equal(ctx.h_fit(42)["train"], a["train"])
          and not np.array_equal(a["train"], c["train"]))
    check("(1) 조기 종료 검증 h: 시드 s 는 default_rng(s+1)",
          np.array_equal(a["val"], np.random.default_rng(44).integers(1, 61, size=len(a["val"])).astype(np.int16)))
    for sd in (42, 43):
        M = ctx.h_multi(sd, 10)
        distinct = np.all(np.sort(M, axis=1)[:, 1:] != np.sort(M, axis=1)[:, :-1])
        check(f"(1) H10 시드 {sd}: (학습 행, 10), 1..60, 행마다 중복 없음",
              M.shape == (len(ctx.sp["train"]["idx"]), 10) and M.min() >= 1 and M.max() <= 60 and bool(distinct),
              f"shape {M.shape}")
    check("(1) H10 재현·시드별 다름", np.array_equal(ctx.h_multi(42, 10), ctx.h_multi(42, 10))
          and not np.array_equal(ctx.h_multi(42, 10), ctx.h_multi(43, 10)))
    M60 = ctx.h_multi(42, 60)
    check("(1) H60 = 1..60 전부", bool((M60 == np.arange(1, 61)).all()))


def t_hist(ctx, rng):
    s = ctx.samples.copy()
    s["sido"] = s["sgg_code"].str[:2]
    s["unit"] = s["sgg_code"] + "|" + s["train_prdlst_code"].astype(str) + "|" + s["train_prdlst_code_2"].astype(str)
    s["dd"] = s["d"].to_numpy().astype("datetime64[D]")
    col = {"p3": "train_prdlst_code_2", "sgg": "sgg_code", "unit": "unit", "sido": "sido", "p2": "train_prdlst_code"}
    groups = {lv: {k: (g["dd"].to_numpy(), g["pos"].to_numpy()) for k, g in s.groupby(c, observed=True)} for lv, c in col.items()}
    all_d, all_p = s["dd"].to_numpy(), s["pos"].to_numpy()

    def cnt(dates, pos, T, P, incl_T=False):
        lo = T - np.timedelta64(P, "D")
        hi = T if incl_T else T - np.timedelta64(1, "D")
        m = (dates >= lo) & (dates <= hi)
        return int(m.sum()), int(pos[m].sum())

    picks = []
    for sp in C1.SPLITS:
        n = len(ctx.sp[sp]["idx"])
        for i in rng.choice(n, 70, replace=False):
            picks.append((sp, int(i), int(rng.integers(1, 61)), "무작위"))
    # T 당일에 같은 단위 시료가 있는 경우: 같은 단위의 이전 날짜 d' (d-d' = 1..60) -> h = d-d'
    rows = ctx.rows
    ukey = (rows["sgg_code"] + "|" + rows["train_prdlst_code"].astype(str) + "|" + rows["train_prdlst_code_2"].astype(str)).to_numpy()
    dd = rows["d"].to_numpy().astype("datetime64[D]")
    prev_same = np.r_[False, ukey[1:] == ukey[:-1]]
    gap = np.r_[0, (dd[1:] - dd[:-1]).astype(int)]
    cand = np.flatnonzero(prev_same & (gap >= 1) & (gap <= 60))
    pos_in_sp = {}
    for sp in C1.SPLITS:
        for j, ix in enumerate(ctx.sp[sp]["idx"]):
            pos_in_sp[ix] = (sp, j)
    for ix in rng.choice(cand, 120, replace=False):
        sp, j = pos_in_sp[ix]
        picks.append((sp, j, int(gap[ix]), "T당일 같은 단위 시료 있음"))
    # p0 가 없는 창(첫 시료 날짜 이전 T)
    tr = ctx.sp["train"]
    early = np.argsort(tr["hd"])[:5]
    for j in early:
        picks.append(("train", int(j), 60, "창에 시료 없음 가능"))
    nbad, n_diff_incl, n_nan, maxrel = 0, 0, 0, 0.0
    ks = [10, 200]
    for sp, j, h, kind in picks:
        v = ctx.sp[sp]
        r = rows.iloc[v["idx"][j]]
        T = np.datetime64(r["d"], "D") - np.timedelta64(h, "D")
        keys = {"p3": str(r["train_prdlst_code_2"]), "sgg": str(r["sgg_code"]), "unit": ukey[v["idx"][j]],
                "sido": str(r["sgg_code"])[:2], "p2": str(r["train_prdlst_code"])}
        grp = {lv: v["grp"][lv][j:j + 1] for lv in v["grp"]}
        hd = v["hd"][j:j + 1]
        exp_vals, got_vals = [], []
        for k in ks:
            got_vals += list(hist_features(ctx.H, grp, hd, h, ["rate5", "logn5", "rate1"], k)[0])
            for st, P in (("rate5", E.P5), ("logn5", E.P5), ("rate1", E.P1)):
                N, S = cnt(all_d, all_p, T, P)
                p0 = S / N if N > 0 else np.nan
                for lv in E.HIST_LEVELS:
                    n, sc = cnt(*groups[lv][keys[lv]], T, P)
                    exp_vals.append((sc + k * p0) / (n + k) if st != "logn5" else np.log1p(n))
        got_vals += list(woe_features(ctx.H, grp, hd, h)[0])
        N, S = cnt(all_d, all_p, T, E.P5)
        for lv in E.WOE_LEVELS:
            n, sc = cnt(*groups[lv][keys[lv]], T, E.P5)
            exp_vals.append(np.log((sc + .5) / (S + .5)) - np.log((n - sc + .5) / (N - S + .5)))
        e, g = np.array(exp_vals, dtype=np.float64), np.array(got_vals, dtype=np.float64)
        same_nan = np.array_equal(np.isnan(e), np.isnan(g))
        fin = np.isfinite(e)
        n_nan += int(np.isnan(e).any())
        rel = np.abs(e[fin] - g[fin]) / np.maximum(1e-12, np.abs(e[fin])) if fin.any() else np.array([0.0])
        abs_ = np.abs(e[fin] - g[fin]) if fin.any() else np.array([0.0])
        maxrel = max(maxrel, float(np.max(np.minimum(rel, abs_ * 1e3))))
        if not (same_nan and np.allclose(g[fin], e[fin], rtol=1e-5, atol=1e-7)):
            nbad += 1
        if kind.startswith("T당일"):
            n1, _ = cnt(*groups["unit"][keys["unit"]], T, E.P5)
            n2, _ = cnt(*groups["unit"][keys["unit"]], T, E.P5, incl_T=True)
            n_diff_incl += int(n2 != n1)
    check("(2)(3) 이력 변수 9개(k=10, 200) + WOE 4개 = 단순 계산", nbad == 0,
          f"불일치 {nbad}/{len(picks)} (무작위 {sum(p[3] == '무작위' for p in picks)}, T당일 같은 단위 시료 "
          f"{sum(p[3].startswith('T당일') for p in picks)}, 이른 행 5), 최대 상대오차 {maxrel:.1e}, NaN 포함 경우 {n_nan}")
    check("(2) T 당일 제외가 실제로 값에 영향: 'T 포함'으로 세면 단위 검사 수가 달라지는 경우 있음(그래도 위 대조는 일치)",
          n_diff_incl > 0, f"T당일 경우 120개 중 {n_diff_incl}개가 달라짐")
    # 이후 자료 영향 없음: 마지막 시료 날짜 뒤에 가짜 시료를 더해도 창 합계가 같아야 함(누적합 구조 확인)
    j = 0
    v = ctx.sp["test"]
    grp = {lv: v["grp"][lv][j:j + 1] for lv in v["grp"]}
    base = hist_features(ctx.H, grp, v["hd"][j:j + 1], 1, ["rate5", "logn5", "rate1"], 30)
    hi = ctx.H.Cn["unit"][grp["unit"][0]]
    t = int(v["hd"][j] - 1)
    check("(2) 누적합 창: 날짜 번호 t 의 창 합계는 t 보다 뒤 칸을 쓰지 않음",
          int(hi[min(t, ctx.H.Dh)] - hi[max(t - E.P5, 0)]) == int(np.round(np.expm1(base[0, 5]))))


def t_cache(ctx):
    nb = 0
    for sp in ("val", "test"):
        v = ctx.sp[sp]
        pos = np.arange(len(v["idx"]))
        for h in (1, 30, 60):
            a = ctx.weather_block(sp, pos, h)
            hh = np.full(len(pos), h, dtype=np.int16)
            b = F.build_tree_matrix(ctx.weather.W, v["s"], v["d"] - h, hh, E.L, "summary")
            nb += int(not np.array_equal(a, b, equal_nan=True))
    check("(4) 검증·시험 기상 요약 캐시 = 1차 build_tree_matrix (h=1,30,60)", nb == 0, f"불일치 {nb}/6")


def t_groups(ctx):
    for g in ("C1", "C2", "C3"):
        G = ctx.groups[g]
        check(f"(5) {g} {G['desc']} = 예상", G["matches_expected"],
              f"코드 {G['codes']}, 학습 검사 {G['train_samples']:,}, 검출 {G['train_pos']} ({G['train_pos_share']:.4f})")


def t_weighted():
    v = np.arange(1, 61, dtype=float)
    check("(6) 구간 가중: 1..60 -> 0.5x10.5 + 0.3x30.5 + 0.2x50.5 = 24.5", abs(weighted(v) - 24.5) < 1e-12)
    d = E.EXP1_RESULTS / "catboost__summary__L050"
    s = summarize(pd.read_csv(d / "metrics_by_h.csv"))
    import json
    s1 = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    check("(6) 1차 L050 60일 평균 PR-AUC·ROC-AUC 다시 계산 = 1차 summary",
          all(abs(s[f"{sp}_{m}_mean60"] - s1[f"{sp}_{m}_mean60"]) < 1e-12 for sp in ("val", "test") for m in ("pr_auc", "roc_auc")),
          f"검증 구간가중 PR-AUC {s['val_pr_auc_w']:.6f}, 시험 {s['test_pr_auc_w']:.6f}")


def _loop_dist(Xm, Cm, med, n_oh, i, j):
    """반복문용 nan-유클리드 거리(sklearn 을 쓰지 않음): 숫자 칸은 둘 다 있는 칸만, 범주 칸이 다르면 원-핫 두 칸이
    (중앙값 표준편차/2) 만큼 다름, 칸 수 비율(전체 칸 / 둘 다 있는 칸)로 키움. 원-핫 칸은 결측이 없다."""
    a, b = Xm[i], Xm[j]
    m = ~(np.isnan(a) | np.isnan(b))
    sq = float(np.sum((a[m] - b[m]) ** 2)) + int(np.sum(Cm[i] != Cm[j])) * 2 * (med / 2) ** 2
    return float(np.sqrt(sq * (Xm.shape[1] + n_oh) / (int(m.sum()) + n_oh)))


def _smote_loop_checks(tag, X, codes, y, group, Xn, Cn, info, plan, k=5):
    """(a)(b)(c) 를 반복문으로 대조. group = 행마다 시도(이웃 찾기 묶음)."""
    mi = plan["minority_index"]
    check(f"(7){tag} 검출 행 번호 = y==1 인 행", np.array_equal(mi, np.flatnonzero(np.asarray(y) == 1)))
    Xm = X[mi].astype(np.float64)
    Cm = codes[mi]
    gm = np.asarray(group)[mi].astype(str)
    n = len(mi)
    with np.errstate(invalid="ignore"):
        med = float(np.nanmedian(np.nanstd(Xm, axis=0)))
    n_oh = sum(len(np.unique(Cm[:, q])) for q in range(Cm.shape[1]))
    cnt = {}
    for g_ in gm:
        cnt[g_] = cnt.get(g_, 0) + 1
    small = sorted(g_ for g_, c in cnt.items() if c < k + 1)
    check(f"(7){tag} 시도별 검출 행 수·전국 대체 시도(검출 행 {k + 1}개 미만) = 기록",
          info["minority_by_group"] == dict(sorted(cnt.items())) and sorted(info["fallback_groups"]) == small,
          f"전국 대체 {[(g_, cnt[g_]) for g_ in small]}")
    exact = valid = 0
    for i in range(n):
        if gm[i] in small:
            cand = [j for j in range(n) if j != i]
        else:
            cand = [j for j in range(n) if j != i and gm[j] == gm[i]]
        d = np.array([_loop_dist(Xm, Cm, med, n_oh, i, j) for j in cand])
        o = np.argsort(d, kind="stable")[:k]
        got = set(int(x) for x in plan["nn"][i])
        exact += int(set(np.array(cand)[o].tolist()) == got)
        dmap = dict(zip(cand, d))
        kth = float(np.sort(d)[k - 1])
        valid += int(all(j in dmap and dmap[j] <= kth * (1 + 1e-9) + 1e-12 for j in got))   # 거리 동점이면 어느 쪽이든 인정
    check(f"(7a){tag} 이웃 {k}개 = 그 시도 검출 행 중 반복문 거리로 가장 가까운 {k}개(전국 대체 시도는 전국에서)", valid == n,
          f"정확히 같은 행 {exact}/{n}, 동점 허용 {valid}/{n}")
    n_same = n_cross = n_fb = nb_c = nb_b = 0
    for t in range(len(Xn)):
        r, c, st = int(plan["rows"][t]), int(plan["cols"][t]), float(plan["steps"][t])
        nb = int(plan["nn"][r, c])
        if gm[r] in small:
            n_fb += 1
        elif gm[nb] == gm[r]:
            n_same += 1
        else:
            n_cross += 1
        e = Xm[r] + st * (Xm[nb] - Xm[r])
        nb_c += int(not np.allclose(Xn[t], e.astype(np.float32), rtol=1e-6, atol=1e-6, equal_nan=True))
        nb_b += int(not np.array_equal(Cn[t], Cm[r]))
    check(f"(7a){tag} 합성 행마다 기준 행과 이웃이 같은 시도(전국 대체 시도 기준 행 제외)", n_cross == 0 and n_same + n_fb == len(Xn),
          f"같은 시도 {n_same:,}, 다른 시도 {n_cross}, 전국 대체 시도 기준 행 {n_fb:,} (기록 {info['n_new_from_fallback']})")
    check(f"(7b){tag} 합성 행 범주 = 기준 검출 행 범주 (100%)", nb_b == 0, f"다른 행 {nb_b}/{len(Xn):,}")
    check(f"(7c){tag} 숫자 칸 = 기준 행 + u x (이웃 - 기준 행) (결측은 그대로 결측)", nb_c == 0,
          f"불일치 {nb_c}/{len(Xn):,}, 결측 비율 원래 {info['nan_rate_minority']:.3f} -> 새 {info['nan_rate_synthetic']:.3f}")
    return Xm, Cm, gm


def _old_smote():
    """scratch 의 이전 e2smote 백업(있으면) - 난수 순서가 이전과 같은지 대조용."""
    import importlib.util
    f = E.EXP2_DIR / "scratch" / "e2smote_before_smote_fix.py.bak"
    if not f.exists():
        return None
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("e2smote_old", str(f))
    spec = importlib.util.spec_from_loader("e2smote_old", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.smotenc


def t_smote():
    """작은 가짜 자료: 시도 5개(검출 45·40·24·6·5 -> 5개인 'D' 는 전국 대체, 6개인 'F' 는 시도 안), 결측 15%."""
    g = np.random.default_rng(5)
    sizes = {"A": 45, "B": 40, "C": 24, "F": 6, "D": 5}
    n_min = sum(sizes.values())
    n_maj, p = 3000, 8
    X = g.normal(size=(n_maj + n_min, p)).astype(np.float32) * np.array([1, 10, 0.1, 5, 1, 1, 100, 2], dtype=np.float32)
    X[g.random(X.shape) < 0.15] = np.nan
    codes = np.stack([g.integers(0, kk, n_maj + n_min) for kk in (3, 7, 4, 9)], 1)
    y = np.r_[np.zeros(n_maj, int), np.ones(n_min, int)]
    group = np.r_[g.choice(list(sizes), n_maj), np.concatenate([[k_] * v for k_, v in sizes.items()])]
    perm = g.permutation(len(y))                                  # 검출 행이 섞여 있게
    X, codes, y, group = X[perm], codes[perm], y[perm], group[perm]
    Xn, Cn, info, plan = smotenc(X, codes, y, 0.1, 5, 42, group=group)
    check("(7) SMOTE 만들 개수 = int(불검출 x 0.1 - 검출)", info["n_new"] == int(n_maj * 0.1 - n_min) == len(Xn), f"{info['n_new']}")
    check("(7) 기록: 방식 = '시도 안 이웃 + 기준 행 범주', 전국 대체 = {'D': 5}",
          info["method"] == "시도 안 이웃 + 기준 행 범주" and info["fallback_groups"] == {"D": 5}, str(info["fallback_groups"]))
    _smote_loop_checks("[가짜]", X, codes, y, group, Xn, Cn, info, plan)
    old = _old_smote()
    if old is None:
        say("    [참고] (7) 이전 방식 대조 건너뜀: scratch/e2smote_before_smote_fix.py.bak 없음")
        return
    Xo, Co, io, po = old(X, codes, y, 0.1, 5, 42)
    check("(7)[가짜] 난수 순서 이전과 같음: 기준 행·이웃 칸·비율이 이전 방식과 동일",
          np.array_equal(po["rows"], plan["rows"]) and np.array_equal(po["cols"], plan["cols"])
          and np.array_equal(po["steps"], plan["steps"]))
    Xa, Ca, ia, pa_ = smotenc(X, codes, y, 0.1, 5, 42, group=np.full(len(y), "전국"))
    check("(7)[가짜] 시도가 하나뿐이면 이웃·숫자 칸이 이전 방식(전국 이웃)과 같음(거리 계산이 이전과 같다는 확인)",
          np.array_equal(pa_["nn"], po["nn"]) and np.array_equal(Xa, Xo, equal_nan=True))


def t_smote_real(ctx):
    """실제 학습 자료: D2 시드 42 와 같은 입력(E00 특성 + 시드 42 h, 범주 번호, sgg 앞 2자리)으로 smotenc 를 돌려 대조."""
    from e2models import FeatureBuilder, smote_cat_consistency
    t0 = time.time()
    tr = ctx.sp["train"]
    pos = ctx.positions("train")
    fb = FeatureBuilder(ctx, dict(E.BASE_FEATS))
    X = fb.num("train", pos, ctx.h_fit(42)["train"][pos])
    y = tr["y"][pos]
    codes = tr["cats"][pos]
    cats = fb.cats("train", pos)
    jsgg, jsido = C1.CAT_COLS.index("sgg"), C1.CAT_COLS.index("sido")
    group = np.array([str(s)[:2] for s in cats[:, jsgg]], dtype=object)
    check("(7)[실제] 이웃 묶음(sgg 앞 2자리) = 범주 sido 칸", bool(np.all(group == cats[:, jsido].astype(str))))
    Xn, Cn, info, plan = smotenc(X, codes, y, E.SMOTE_RATIO, E.SMOTE_K, 42, group=group)
    say(f"    (7)[실제] 학습 행 {len(y):,}, 검출 {info['n_minority']:,}, 불검출 {info['n_majority']:,}, 새 표본 {info['n_new']:,}")
    say(f"    (7)[실제] 시도별 학습 검출 행: {info['minority_by_group']}")
    say(f"    (7)[실제] 전국 대체 시도(검출 행 {E.SMOTE_K + 1}개 미만): {info['fallback_groups']}, 그 시도 기준 새 표본 {info['n_new_from_fallback']}")
    check("(7)[실제] 만들 개수 = int(불검출 x 0.1 - 검출)", info["n_new"] == int(info["n_majority"] * E.SMOTE_RATIO - info["n_minority"]) == len(Xn))
    _smote_loop_checks("[실제]", X, codes, y, group, Xn, Cn, info, plan, k=E.SMOTE_K)
    cats_new = np.empty((len(Cn), len(C1.CAT_COLS)), dtype=object)
    for j, c in enumerate(C1.CAT_COLS):
        cats_new[:, j] = np.array(ctx.levels[c], dtype=object)[Cn[:, j]]
    cc = smote_cat_consistency(cats_new, cats[plan["minority_index"]], plan["rows"])
    check("(7b)[실제] 합성 행 문자열 범주: sido = sgg 앞 2자리 100%, 기준 검출 행과 같은 단위 100%, 학습 검출 단위에 있는 조합 100%",
          cc["sido_eq_sgg2_rate"] == 1.0 and cc["same_as_base_rate"] == 1.0 and cc["in_train_pos_units_rate"] == 1.0,
          f"{cc['sido_eq_sgg2_rate']:.4f} / {cc['same_as_base_rate']:.4f} / {cc['in_train_pos_units_rate']:.4f}, "
          f"합성 행 단위 {cc['n_distinct_units_synthetic']}종 / 학습 검출 단위 {cc['n_distinct_units_train_pos']}종")
    check("(7)[실제] 기록된 시도 안 이웃 비율 = 1", info["neighbor_same_group_rate"] == 1.0, str(info["neighbor_same_group_rate"]))
    old = _old_smote()
    if old is not None:
        Xo, Co, io, po = old(X, codes, y, E.SMOTE_RATIO, E.SMOTE_K, 42)
        same_rng = (np.array_equal(po["rows"], plan["rows"]) and np.array_equal(po["cols"], plan["cols"])
                    and np.array_equal(po["steps"], plan["steps"]))
        n_nn_changed = int(np.sum(np.any(np.sort(po["nn"], 1) != np.sort(plan["nn"], 1), axis=1)))
        check("(7)[실제] 난수 순서 이전과 같음(기준 행·이웃 칸·비율 동일)", same_rng,
              f"이웃 목록이 바뀐 검출 행 {n_nn_changed}/{info['n_minority']}, 이전 방식 sido ≠ sgg 앞 2자리 "
              f"{float(np.mean(Co[:, jsido] != np.array([ctx.levels['sido'].index(ctx.levels['sgg'][s][:2]) for s in Co[:, jsgg]]))):.3f}")
    say(f"    (7)[실제] {time.time() - t0:.0f}s")


def t_highrisk():
    """(12) 고위험 비교 추가 줄을 저장 예측에서 직접 다시 계산(sklearn·1차 metrics 를 쓰지 않음)해 대조. 전체 results/ 읽기만."""
    import json
    import pyarrow.parquet as pq
    from e2tables import high_risk_comparison
    P = E.paths(False)
    gpath, apath = P["results"] / "groups.json", P["results"] / "selection_A.json"
    if not (gpath.exists() and apath.exists()):
        check("(12) 고위험 비교 대조: 전체 결과(groups.json, selection_A.json) 있음", False)
        return
    groups = json.loads(gpath.read_text(encoding="utf-8"))
    selA = json.loads(apath.read_text(encoding="utf-8"))          # select_A 는 파일을 다시 쓰므로 부르지 않음
    t0 = time.time()
    comp = high_risk_comparison(P, False, selA)
    tag_re = "기준값 = 그 품목 검증 행에서 다시 구함"
    new = comp[comp["모델"].astype(str).str.contains(tag_re, regex=False)]
    check("(12) 묶음마다 추가 줄 2개(E00, A 최고)", len(new) == 6 and all((new["묶음"] == g).sum() == 2 for g in ("C1", "C2", "C3")),
          f"{len(new)}줄: {list(new['모델'].unique())}")

    def load(eid, sd, sp, codes):
        t = pq.read_table(P["results"] / E.run_dir_name(eid, sd) / f"pred_{sp}.parquet").to_pandas()
        t = t[t["train_prdlst_code_2"].astype(str).isin(codes)]
        w = t.pivot_table(index=["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d"], columns="h", values="p",
                          aggfunc="first", observed=True)
        yy = t[t["h"] == 1].set_index(["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d"])["y"].reindex(w.index)
        return w.to_numpy(np.float64).T, yy.to_numpy().astype(int)          # (60, n), (n,)

    def f2max_thr(p, y):
        """모든 서로 다른 예측값 t 에 대해 (p >= t) 의 F2 = 5TP/(5TP+4FN+FP), 최대, 같으면 가장 낮은 t."""
        p, y = p.reshape(-1), np.tile(y, E.H_MAX)
        u = np.unique(p)
        pp, pn = np.sort(p[y == 1]), np.sort(p[y == 0])
        tp = len(pp) - np.searchsorted(pp, u, side="left")
        fp = len(pn) - np.searchsorted(pn, u, side="left")
        fn = len(pp) - tp
        f2 = np.where(tp > 0, 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1), 0.0)
        return float(u[int(np.argmax(f2))])

    def wsum(v):
        v = np.asarray(v, float)
        return 0.5 * v[:20].mean() + 0.3 * v[20:40].mean() + 0.2 * v[40:60].mean()

    nbad, nchk, detail = 0, 0, []
    whole = [("E00", "전체 모델 E00"), (selA["best"], f"전체 모델 A 최고({selA['best']})")]
    for g in ("C1", "C2", "C3"):
        codes = groups[g]["codes"]
        for eid, lab in whole:
            acc = {k_: [] for k_ in ("thr", "rec_w", "prec_w", "f2_w", "rec_m", "prec_m", "f2_m", "tp", "fn", "fp", "tn")}
            for sd in E.SEEDS:
                pv, yv = load(eid, sd, "val", codes)
                pt, yt = load(eid, sd, "test", codes)
                thr = f2max_thr(pv, yv)
                yb = yt.astype(bool)
                per = {k_: [] for k_ in ("rec", "prec", "f2", "tp", "fn", "fp", "tn")}
                for h in range(E.H_MAX):
                    pr = pt[h] >= thr
                    tp_, fn_, fp_, tn_ = int((pr & yb).sum()), int((~pr & yb).sum()), int((pr & ~yb).sum()), int((~pr & ~yb).sum())
                    rc = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
                    pc = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
                    per["rec"].append(rc)
                    per["prec"].append(pc)
                    per["f2"].append(5 * pc * rc / (4 * pc + rc) if pc + rc else 0.0)
                    for k_, v_ in (("tp", tp_), ("fn", fn_), ("fp", fp_), ("tn", tn_)):
                        per[k_].append(v_)
                acc["thr"].append(thr)
                for k_ in ("rec", "prec", "f2"):
                    acc[k_ + "_w"].append(wsum(per[k_]))
                    acc[k_ + "_m"].append(float(np.mean(per[k_])))
                for k_ in ("tp", "fn", "fp", "tn"):
                    acc[k_].append(float(np.mean(per[k_])))
            r = new[(new["묶음"] == g) & new["모델"].astype(str).str.startswith(lab.rstrip(")"))]
            o = comp[(comp["묶음"] == g) & (comp["모델"] == lab)]
            if len(r) != 1 or len(o) != 1:
                nbad += 1
                detail.append(f"{g} {lab}: 줄 {len(r)}/{len(o)}")
                continue
            r, o = r.iloc[0], o.iloc[0]
            pairs = [("판정 기준값", np.mean(acc["thr"])), ("시험 구간가중 Recall", np.mean(acc["rec_w"])),
                     ("시험 구간가중 Precision", np.mean(acc["prec_w"])), ("시험 구간가중 F2", np.mean(acc["f2_w"])),
                     ("시험 60일평균 Recall", np.mean(acc["rec_m"])), ("시험 60일평균 Precision", np.mean(acc["prec_m"])),
                     ("시험 60일평균 F2", np.mean(acc["f2_m"]))] + \
                    [(f"시험 {k_.upper()} (h당 평균)", np.mean(acc[k_])) for k_ in ("tp", "fn", "fp", "tn")]
            for col, v in pairs:
                nchk += 1
                if not np.isclose(float(r[col]), float(v), rtol=1e-9, atol=1e-12):
                    nbad += 1
                    detail.append(f"{g} {lab} {col}: 표 {r[col]} vs 직접 {v}")
            for col in ("검증 구간가중 PR-AUC", "시험 구간가중 PR-AUC", "시험 60일평균 PR-AUC", "시험 60일평균 ROC-AUC",
                        "시험 구간가중 ROC-AUC", "시험 행 수", "시험 검출 행 수", "검증 행 수", "검증 검출 행 수"):
                nchk += 1
                if not (float(r[col]) == float(o[col])):
                    nbad += 1
                    detail.append(f"{g} {lab} {col}: 추가 줄 {r[col]} vs 기존 줄 {o[col]}")
            say(f"    (12) {g} {lab}: 기준값(시드 평균) 그 품목 검증 {np.mean(acc['thr']):.5f} vs 전체 행 {float(o['판정 기준값']):.5f}, "
                f"시험 구간가중 Recall {float(r['시험 구간가중 Recall']):.4f} (기존 {float(o['시험 구간가중 Recall']):.4f}), "
                f"F2 {float(r['시험 구간가중 F2']):.4f} (기존 {float(o['시험 구간가중 F2']):.4f})")
    check("(12) 추가 줄 기준값·Recall·Precision·F2·TP~TN = 직접 계산, PR-AUC·ROC-AUC·행 수 = 기존 줄", nbad == 0,
          f"불일치 {nbad}/{nchk} {detail[:3]} ({time.time() - t0:.0f}s)")
    old_csv = E.EXP2_DIR / "tables" / "고위험품목비교.csv"
    if old_csv.exists():
        prev = pd.read_csv(old_csv)
        keep = comp[~comp["모델"].astype(str).str.contains(tag_re, regex=False)].reset_index(drop=True)
        num = [c for c in prev.columns if c in keep.columns and pd.api.types.is_numeric_dtype(prev[c])]
        same = (len(prev) == len(keep) and prev[["묶음", "모델", "실험 ID"]].astype(str).equals(keep[["묶음", "모델", "실험 ID"]].astype(str))
                and np.allclose(prev[num].to_numpy(float), keep[num].to_numpy(float), rtol=1e-12, atol=0, equal_nan=True))
        check("(12) 기존 줄(추가 줄 제외)은 이전 표 tables/고위험품목비교.csv 와 같음", same, f"{len(prev)}줄 vs {len(keep)}줄, 숫자 칸 {len(num)}개")


def t_h60_equiv(ctx, log):
    """작은 부분집합(학습 행 1500 x h 60 = 9만 행)으로 세 방식 예측 비교."""
    from e2models import Cat2, FeatureBuilder, build_h60_pool
    rng = np.random.default_rng(3)
    tr = ctx.sp["train"]
    pos_tr = np.sort(np.r_[rng.choice(np.flatnonzero(tr["y"] == 0), 1400, replace=False), np.flatnonzero(tr["y"] == 1)[:100]])
    va = ctx.sp["val"]
    pos_va = np.sort(np.r_[rng.choice(np.flatnonzero(va["y"] == 0), 3000, replace=False), np.flatnonzero(va["y"] == 1)])
    fb = FeatureBuilder(ctx, dict(E.BASE_FEATS))
    hv = ctx.h_fit(42)["val"][pos_va]
    Xva, cva, yva = fb.num("val", pos_va, hv), fb.cats("val", pos_va), va["y"][pos_va]
    rep = np.tile(pos_tr, 60)
    hh = np.repeat(np.arange(1, 61, dtype=np.int16), len(pos_tr))
    Xtr, ctr, ytr = fb.num("train", rep, hh), fb.cats("train", rep), tr["y"][rep]
    preds = {}
    for name in ("A 메모리 행우선(내부 양자화)", "B 메모리 행우선 + quantize()", "C H60 방식(파일 열우선 + quantize() + 전체 행 slice)"):
        m = Cat2(42, log, 60, 1000)
        if name.startswith("A"):
            m.fit(Xtr, ytr, lambda: (Xva, cva), yva, fb.names, cats_tr=ctr)
        elif name.startswith("B"):
            m.num_names = fb.names
            pool = m.pool(Xtr, ctr, ytr)
            pool.quantize(random_seed=42, task_type="CPU")
            m.fit(pool, None, lambda: (Xva, cva), yva, fb.names)
        else:
            pool, info = build_h60_pool(ctx, fb, pos_tr, 42, E.EXP2_DIR / "cache", lambda *_: None)
            check("(8) 여러 h 파일: 양자화 뒤 원값 놓고 파일 지움", info.get("memmap_removed") is True, str(info.get("memmap_removed")))
            m.fit(pool, None, lambda: (Xva, cva), yva, fb.names)
            del pool
        preds[name] = m.predict(Xva, cva)
    ks = list(preds)
    dAB = float(np.max(np.abs(preds[ks[0]] - preds[ks[1]])))
    dBC = float(np.max(np.abs(preds[ks[1]] - preds[ks[2]])))
    say(f"    (8) 학습 {len(ytr):,}행(H60 순서 같음), 60라운드: A-B 최대 차이 {dAB:.2e}, B-C 최대 차이 {dBC:.2e}")
    check("(8) 여러 h 방식(파일 열우선 + quantize + slice) = 메모리 행우선 + quantize (예측 같음)", dBC == 0.0, f"최대 차이 {dBC:.2e}")
    return dAB, dBC


def t_e00(ctx, log):
    import shutil
    from run import check_windows, run_one
    P = E.paths(False)
    P["results"] = E.EXP2_DIR / "results" / "_selftest"
    P["tables"] = E.EXP2_DIR / "tables" / "_selftest"
    if (P["results"] / "E00__s42").exists():
        shutil.rmtree(P["results"] / "E00__s42")          # 이전 점검 결과를 건너뛰지 않고 지금 코드로 새로 실행
    P["results"].mkdir(parents=True, exist_ok=True)
    mem = Mem()
    mem.start()
    win = check_windows(ctx, log)
    t0 = time.time()
    s = run_one(E.EXP_BY_ID["E00"], 42, ctx, P, log, mem, False, win)
    mem.stop()
    d1 = E.EXP1_RESULTS / "catboost__summary__L050"
    d2 = P["results"] / "E00__s42"
    for sp in ("val", "test"):
        p1, y1, i1 = read_preds(d1 / f"pred_{sp}.parquet")
        p2, y2, i2 = read_preds(d2 / f"pred_{sp}.parquet")
        same_rows = i1.astype(str).equals(i2.astype(str)) and np.array_equal(y1, y2)
        check(f"(9) E00 시드 42 {sp} 예측 = 1차 catboost__summary__L050 (행·y·p 모두 비트 단위)",
              same_rows and np.array_equal(p1, p2), f"{p1.shape}, 최대 차이 {float(np.max(np.abs(p1.astype(np.float64) - p2))):.1e}")
    m1 = pd.read_csv(d1 / "metrics_by_h.csv")
    m2 = pd.read_csv(d2 / "metrics_by_h.csv")
    cols = ["pr_auc", "roc_auc", "threshold", "tp", "fn", "fp", "tn", "precision", "recall", "f1", "f2"]
    check("(9) E00 시드 42 h 별 지표 = 1차", np.allclose(m1[cols].to_numpy(float), m2[cols].to_numpy(float), rtol=0, atol=1e-12))
    import json
    s1 = json.loads((d1 / "summary.json").read_text(encoding="utf-8"))
    check("(9) E00 시드 42 최적 라운드·학습 라운드 = 1차", s["best_rounds_or_epoch"] == s1["best_rounds_or_epoch"]
          and s["n_rounds_trained"] == s1["n_rounds_trained"],
          f"{s['best_rounds_or_epoch']}/{s['n_rounds_trained']} vs {s1['best_rounds_or_epoch']}/{s1['n_rounds_trained']}, "
          f"총 {time.time() - t0:.0f}s")
    import run as R
    cfg = json.loads((d2 / "config.json").read_text(encoding="utf-8"))
    md = cfg.get("code_md5") or {}
    check("(9) config.json 에 코드 지문(code_md5) 기록", "run.py" in md and "e2models.py" in md and "1차/models.py" in md,
          f"{len(md)}개 파일")
    msgs = []
    R.check_code_changed(d2, msgs.append, "E00__s42")
    saved = dict(R.code_md5())
    try:
        R._MD5 = dict(saved, **{"e2models.py": "0" * 32})
        R.check_code_changed(d2, msgs.append, "E00__s42")
    finally:
        R._MD5 = saved
    check("(9) 건너뛸 때 코드가 같으면 조용, 바뀌면 [주의] 로그", len(msgs) == 1 and "[주의]" in msgs[0] and "e2models.py" in msgs[0],
          msgs[0][:80] if msgs else "로그 없음")


def t_cleanup(ctx, log):
    import catboost
    import psutil
    import subprocess
    from e2models import FeatureBuilder, build_multi_h_pool
    cdir = E.EXP2_DIR / "scratch" / "_selftest_cache"
    cdir.mkdir(parents=True, exist_ok=True)
    for f in cdir.glob("*"):
        f.unlink()
    # 죽은 pid 찾기: 내가 띄운 짧은 프로세스가 끝난 뒤의 pid (재사용 가능성은 거의 없음)
    pr = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-c", "pass"])
    pr.wait()
    dead = pr.pid
    alive = os.getppid() if psutil.pid_exists(os.getppid()) else os.getpid()
    f_dead = cdir / f"multih30_train_full_s42_{dead}.f32"
    f_alive = cdir / f"multih30_train_full_s42_{alive}.f32"
    f_dead.write_bytes(b"x" * 1024)
    f_alive.write_bytes(b"x" * 1024)
    pos = np.arange(500)
    Hm = ctx.h_multi(42, 30)[pos]
    fb = FeatureBuilder(ctx, dict(E.BASE_FEATS))
    orig = catboost.Pool.quantize

    def boom(self, *a, **k):
        raise RuntimeError("시험용 양자화 오류")
    catboost.Pool.quantize = boom
    err = None
    try:
        build_multi_h_pool(ctx, fb, pos, Hm, 42, cdir, lambda *_: None)
    except RuntimeError as e:
        err = str(e)
    finally:
        catboost.Pool.quantize = orig
    left = sorted(f.name for f in cdir.glob("*.f32"))
    mine = [x for x in left if x.endswith(f"_{os.getpid()}.f32")]
    check("(10) 양자화 중 오류 -> 오류는 그대로 올라오고 이 프로세스의 임시 파일은 지워짐", err == "시험용 양자화 오류" and not mine,
          f"오류 {err}, 남은 파일 {left}")
    check("(10) 죽은 pid 의 임시 파일만 지우고 살아 있는 pid 의 파일은 둠", (not f_dead.exists()) and f_alive.exists(),
          f"dead pid {dead} 파일 있음 {f_dead.exists()}, alive pid {alive} 파일 있음 {f_alive.exists()}")
    pool, info = build_multi_h_pool(ctx, fb, pos, Hm, 42, cdir, lambda *_: None)
    check("(10) 정상 실행은 그대로(파일 지움, 행 수 500 x 30)", info.get("memmap_removed") is True and pool.shape[0] == 500 * 30,
          f"{info.get('memmap_removed')}, {pool.shape}")
    del pool
    for f in cdir.glob("*"):
        f.unlink()
    cdir.rmdir()


def t_lock():
    import subprocess
    import run as R
    ldir = E.EXP2_DIR / "scratch" / "_selftest_lock"
    ldir.mkdir(parents=True, exist_ok=True)
    P = {"logs": ldir}
    lock = ldir / "run.lock"
    lock.unlink(missing_ok=True)
    # 명령줄에 run.py 가 들어 있는 내 프로세스(잠자기)를 띄워 그 pid 로 잠금을 만든다
    dummy = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-c", "import time; time.sleep(120)", "run.py"])
    try:
        time.sleep(1.0)
        lock.write_text(f"pid={dummy.pid}\nstarted=시험\nargs=시험\n", encoding="utf-8")
        code = None
        try:
            R.acquire_lock(P)
        except SystemExit as e:
            code = e.code
        check("(11) 살아 있는 run.py 가 잠금을 쥐면 종료 코드 3, 잠금 파일은 그대로", code == R.LOCK_EXIT_CODE
              and f"pid={dummy.pid}" in lock.read_text(encoding="utf-8"), f"종료 코드 {code}")
    finally:
        dummy.terminate()                                   # 내가 띄운 프로세스만 끝냄
        dummy.wait(timeout=30)
    got = R.acquire_lock(P)
    check("(11) 죽은 잠금은 넘겨받음(내 pid 로 바뀜)", f"pid={os.getpid()}\n" in got.read_text(encoding="utf-8"))
    R.release_lock(got)
    check("(11) 끝나면 잠금 파일 지움", not lock.exists())
    ldir.rmdir()


def main():
    t0 = time.time()
    log = Logger(E.EXP2_DIR / "logs" / "progress_selftest.log")
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    rng = np.random.default_rng(2026)
    t_h(ctx)
    t_hist(ctx, rng)
    t_cache(ctx)
    t_groups(ctx)
    t_weighted()
    t_smote()
    t_smote_real(ctx)
    t_highrisk()
    t_h60_equiv(ctx, log)
    t_cleanup(ctx, log)
    t_lock()
    if "--no-e00" not in sys.argv:
        t_e00(ctx, log)
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
