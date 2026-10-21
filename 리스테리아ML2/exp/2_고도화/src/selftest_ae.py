# -*- coding: utf-8 -*-
"""AE·VAE 이상 탐지(e2ae.py, 실험 S_*·A3_AE*·A3_*_p, 단계 'ae') 자체 점검 (전체 자료, 시드 42). 결과: logs/selftest_ae.txt (UTF-8).
실패가 있으면 종료 코드 1. results/·tables/·cache/ 에는 쓰지 않는다(기상 요약 캐시는 메타가 같을 때만 읽기, k* 는 results/selection_k.json 읽기만,
표 다시 만들기는 scratch/ae/selftest_tables/ 에 쓰고 선정 파일 쓰기도 그 아래로 돌림).
먼저 src/timing_ae.py 를 돌려 두면 (c) 재현·(g) 예측 파일 형식을 그 결과(scratch/ae/timing_root, 다른 프로세스에서 계산한 점수)와 대조한다.

(0) 설정: 실험 12개 단계 'ae'·시드 42·43·44·입력(A3 + 묶음), 칸 이름, 조합 ID 와 겹치지 않음, run_exp2.ps1 ValidateSet 'ae'
(a) 정상 행만: 모델 4종 x 6모델(겹 0~4·전체) 행 번호가 학습 구간 불검출 행 안, 학습 ∪ 조기 종료 = 모델 행, 서로 겹치지 않음, 겹 모델은 그 겹 제외,
    전체 모델 = 불검출 전부. 정규화 통계(표: 칸별 평균·표준편차, 시퀀스: 창 겹침 가중 평균·표준편차)·임베딩 어휘를 그 행만으로 따로 계산해 대조
    (시퀀스는 창을 직접 모아 계산 - 다른 방법). build_seq 재사용: 학습 기간 통계를 넣으면 build_seq(ctx) 와 비트 단위 같음
(b) 교차 적합: 겹 배정 = default_rng([42, 1]) 규칙·크기 차이 1 이하, 모든 학습 행이 한 번씩 점수, 저장 가중치로 다시 계산한 겹 점수 = 저장 점수(비트 단위),
    그 겹 점수 ≠ 전체 모델 점수(겹 밖), 검출 행·검증·시험(h 1·30·60) 점수 = 전체 모델 재계산
(c) 재현: 같은 시드 두 번 같은 값(timing_ae 가 다른 프로세스에서 계산해 저장한 점수와 비트 단위 비교, 없으면 이 프로세스에서 한 번 더 계산)
(d) 시퀀스 창 = T−50 … T−1: 배치 입력 = build_seq 표의 그 구간(비트 단위), 기상표 칸을 바꿔 보면 T−1·T−50 은 점수가 바뀌고 T(당일)·T−51 은 안 바뀜
(e) 위약판: 두 번 만들어도 같음, (분할, h, 예측일 T 연도) 묶음마다 값 모음 같음, 대부분 자리가 바뀜, A3_*_p 칸 = 위약 행렬 값
(f) A3 + 점수 행렬 = A3 행렬(비트 단위) + 1칸(학습 = 그 시드 h 점수, 검증 조기 종료용 h·평가 h), 학습 행을 다른 h 로 부르면 오류
(g) 점수 단독 예측 파일 형식 = 다른 실험(results/E00__s42)과 같음(칸 이름·형식·행 순서·키·h·y), p = 저장 점수, metrics_by_h.csv 칸·summary 키
(h) 기존 실험 그대로: 고치기 전 e2models(scratch/ae/e2models_before_ae.py.bak)와 기존 실험 특성 행렬·범주·칸 이름 비트 단위 같음,
    기존 결과(results/*__s42, CatBoost·로지스틱) config.json 입력 칸·칸 수·설명 = 지금 코드,
    같은 results/ 로 고치기 전 e2tables(실험 목록도 고치기 전)·지금 e2tables 로 각각 만든 표에서 기존 실험 줄(실험기록·시드별·고위험품목비교) 같음,
    설정 시트 기존 줄 그대로 + 새 줄
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import config as C1  # noqa: E402
import e2ae as A  # noqa: E402
from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = E.EXP2_DIR / "logs" / "selftest_ae.txt"
SCR = E.EXP2_DIR / "scratch" / "ae"
TIMING_ROOT = SCR / "timing_root"
OLD_E2MODELS = SCR / "e2models_before_ae.py.bak"
SEED = 42
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


def k_star():
    return json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(spec)
    ld.exec_module(mod)
    return mod


# ---------------------------------------------------------------- (0)
def t_config():
    a3 = E.EXP_BY_ID["A3"]["feats"]
    ok = True
    for kd in A.KINDS:
        s, c, p = E.EXP_BY_ID[f"S_{kd}"], E.EXP_BY_ID[f"A3_{kd}"], E.EXP_BY_ID[f"A3_{kd}_p"]
        ok &= (s["stage"] == c["stage"] == p["stage"] == "ae" and s["model"] == "ae" and c["model"] == p["model"] == "cat"
               and s["feats"] == dict(a3, ae=[kd]) and c["feats"] == dict(a3, ae=[kd]) and p["feats"] == dict(a3, ae=[f"{kd}_p"])
               and all(E.exp_seeds(e, False) == [42, 43, 44] for e in (s, c, p)))
    ids = [e["id"] for e in E.EXPERIMENTS if e["stage"] == "ae"]
    check("(0) 실험 12개(S_·A3_·A3_*_p x AEt·VAEt·AEs·VAEs): 단계 'ae', 모델·입력(A3 + 묶음)·시드 42·43·44", ok and len(ids) == 12, str(ids))
    import e2fg
    check("(0) 조합 ID 정규식과 겹치지 않음", not any(e2fg.is_combo_id(i) for i in ids))
    check("(0) 칸 이름", A.ae_names(["AEt", "VAEs_p"]) == ["ae_score_AEt", "ae_score_VAEs_placebo"])
    ps1 = (E.EXP2_DIR / "run_exp2.ps1").read_text(encoding="utf-8-sig")
    vs = [ln for ln in ps1.splitlines() if "ValidateSet" in ln]
    check("(0) run_exp2.ps1 ValidateSet 에 'ae'", bool(vs) and "'ae'" in vs[0])


# ---------------------------------------------------------------- (a)(b)
def tab_X(ctx, k):
    from e2models import FeatureBuilder
    fb = FeatureBuilder(ctx, A.a3_feats(k))
    return fb.num("train", np.arange(len(ctx.sp["train"]["idx"])), ctx.h_fit(SEED)["train"])


def seq_stats_direct(ctx, rows, chunk=4096):
    """창 T−50 … T−1 의 기상 값을 직접 모아 변수별 평균·표준편차(NaN 무시, ddof=0) - window_weight 와 다른 방법."""
    W = ctx.weather.W
    v = ctx.sp["train"]
    s = v["s"][rows].astype(np.int64)
    T = v["d"][rows].astype(np.int64) - ctx.h_fit(SEED)["train"][rows].astype(np.int64)
    L = E.L
    ar = np.arange(L)
    V = W.shape[2]
    sm, ct = np.zeros(V), np.zeros(V)
    for a in range(0, len(rows), chunk):
        x = W[s[a:a + chunk, None], T[a:a + chunk, None] - L + ar[None, :]].astype(np.float64).reshape(-1, V)
        sm += np.nansum(x, 0)
        ct += (~np.isnan(x)).sum(0)
    mu = sm / ct
    sq = np.zeros(V)
    for a in range(0, len(rows), chunk):
        x = W[s[a:a + chunk, None], T[a:a + chunk, None] - L + ar[None, :]].astype(np.float64).reshape(-1, V)
        sq += np.nansum((x - mu) ** 2, 0)
    return mu, np.sqrt(sq / ct)


def t_rows_prep(ctx, k, kind, S, dbg, Xtab, seq_ref):
    y = ctx.sp["train"]["y"]
    ntr = len(y)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    fold = dbg["fold"]
    f_ref = np.empty(len(neg), dtype=np.int64)
    f_ref[np.random.default_rng([SEED, 1]).permutation(len(neg))] = np.arange(len(neg)) % E.AE["n_folds"]
    bc = np.bincount(fold, minlength=E.AE["n_folds"])
    check(f"(b) {kind} 겹 배정 = default_rng([42, 1]) 순열 위치 % 5, 크기 차이 1 이하",
          np.array_equal(fold, f_ref) and bc.max() - bc.min() <= 1, f"겹 크기 {bc.tolist()}")
    bad_rows, bad_prep, cover = [], [], np.zeros(ntr, dtype=np.int64)
    for m, d in dbg["models"].items():
        rows, fit, es, tg = d["rows"], d["fit"], d["es"], d["target"]
        exp_rows = neg if m == "full" else neg[fold != m]
        exp_tg = pos if m == "full" else neg[fold == m]
        ok = (np.array_equal(rows, exp_rows) and np.array_equal(tg, exp_tg) and np.all(y[rows] == 0) and rows.max() < ntr
              and np.array_equal(np.sort(np.concatenate([fit, es])), rows) and np.intersect1d(fit, es).size == 0
              and abs(len(es) - round(len(rows) * E.AE["es_frac"])) <= 0
              and (m == "full" or np.intersect1d(rows, neg[fold == m]).size == 0))
        if not ok:
            bad_rows.append(m)
        cover[tg] += 1
        pr = d["prep"]
        if A.is_tab(kind):
            Xd = pd.DataFrame(Xtab[rows].astype(np.float64))
            mu_ref = Xd.mean(skipna=True).to_numpy()
            sd_ref = Xd.std(ddof=0, skipna=True).to_numpy()
            sd_ref = np.where((sd_ref == 0) | np.isnan(sd_ref), 1.0, sd_ref)
            voc_ok = all(np.array_equal(pr["vocab"][j], np.unique(ctx.sp["train"]["cats"][rows, j])) for j in range(4))
            if not (np.allclose(pr["mu"], mu_ref, rtol=1e-9, atol=1e-12) and np.allclose(pr["sd"], sd_ref, rtol=1e-9, atol=1e-12) and voc_ok):
                bad_prep.append(m)
        elif m in seq_ref:
            mu_ref, sd_ref = seq_ref[m]
            if not (np.allclose(pr["mu64"], mu_ref, rtol=1e-9) and np.allclose(pr["sd64"], sd_ref, rtol=1e-9)
                    and np.array_equal(pr["mu"], pr["mu64"].astype(np.float32))):
                bad_prep.append(m)
    check(f"(a) {kind} 모델 6개 행: 학습 구간 불검출 행만, 학습 ∪ 조기 종료(10%) = 모델 행·겹치지 않음, 겹 모델은 그 겹 제외, 전체 모델 = 불검출 전부",
          not bad_rows, f"이상 {bad_rows}, 불검출 {len(neg):,}·검출 {len(pos):,}")
    check(f"(b) {kind} 모든 학습 행이 정확히 한 번 점수(겹 정상 행 = 그 겹 모델, 검출 행 = 전체 모델)", bool(np.all(cover == 1)))
    what = "칸별 평균·표준편차(pandas 로 따로)·임베딩 어휘" if A.is_tab(kind) else f"창 겹침 가중 평균·표준편차(창 직접 모아 계산, 모델 {[str(m) for m in seq_ref]})"
    check(f"(a) {kind} 정규화 통계 = 그 모델 정상 행만으로 따로 계산한 {what}", not bad_prep, f"다름 {bad_prep}")


def _rebuild(kind, d, dev):
    net = A.make_net(kind, *d["net_args"]).to(dev)
    net.load_state_dict(d["state"])
    net.eval()
    return net


def t_oof(ctx, k, kind, S, dbg):
    dev = torch.device("cuda")
    htr = ctx.h_fit(SEED)["train"].astype(np.int64)
    inp = A.TabInput(ctx, k, htr) if A.is_tab(kind) else A.SeqInput(ctx, htr)
    bad, diff_full, ins_gap = [], [], []
    with A.deterministic_torch():
        dfull = dbg["models"]["full"]
        net_full = _rebuild(kind, dfull, dev)
        for m in range(E.AE["n_folds"]):
            d = dbg["models"][m]
            dd = inp.device_train(d["prep"], dev)
            net = _rebuild(kind, d, dev)
            sc = A.score_rows(net, inp.batch, dd, d["target"], dev)
            if not np.array_equal(sc, S["train"][d["target"]]):
                bad.append(m)
            ddf = inp.device_train(dfull["prep"], dev)
            scf = A.score_rows(net_full, inp.batch, ddf, d["target"], dev)
            diff_full.append(float(np.mean(sc != scf)))
            g = (m + 1) % E.AE["n_folds"]                         # 이 겹 행으로 학습한 다른 겹 모델(겹 안 점수, 기록용)
            dg = dbg["models"][g]
            scg = A.score_rows(_rebuild(kind, dg, dev), inp.batch, inp.device_train(dg["prep"], dev), d["target"], dev)
            ins_gap.append(float(np.mean(sc) - np.mean(scg)))
            del dd, ddf
        ddf = inp.device_train(dfull["prep"], dev)
        ok_pos = np.array_equal(A.score_rows(net_full, inp.batch, ddf, dfull["target"], dev), S["train"][dfull["target"]])
        ok_ev = True
        C_cache = {}
        for sp in ("val", "test"):
            n = len(ctx.sp[sp]["idx"])
            for h in (1, 30, 60):
                de = inp.device_eval(dfull["prep"], dev, sp, h, C_cache) if A.is_tab(kind) else inp.device_eval(ddf, sp, h)
                ok_ev &= bool(np.array_equal(A.score_rows(net_full, inp.batch, de, np.arange(n), dev), S[sp][h - 1]))
    torch.cuda.empty_cache()
    check(f"(b) {kind} 저장 가중치로 다시 계산한 겹 점수 = 저장 점수(비트 단위)", not bad, f"다른 겹 {bad}")
    check(f"(b) {kind} 겹 점수 ≠ 전체 모델 점수(겹 밖 모델로 매김)", min(diff_full) > 0.99, f"다른 행 비율 겹별 {[round(x, 4) for x in diff_full]}")
    say(f"    [기록] {kind} 겹 밖 점수 평균 − 그 행으로 학습한 다른 겹 모델 점수 평균(겹 안): {[round(x, 6) for x in ins_gap]}")
    check(f"(b) {kind} 검출 행·검증·시험(h 1·30·60) 점수 = 전체 정상 행 모델 재계산", ok_pos and ok_ev)


def t_repro(ctx, k, kind, S):
    f = TIMING_ROOT / "results" / A.SCORE_DIR / f"{kind}_s{SEED}_full.npz"
    mf = f.with_suffix(".json")
    if f.exists() and mf.exists():
        j = json.loads(mf.read_text(encoding="utf-8"))
        same_meta = j["meta"] == json.loads(json.dumps(A.meta(ctx, kind, SEED, k), default=str))
        z = np.load(f)
        ok = all(np.array_equal(z[sp], S[sp]) for sp in ("train", "val", "test"))
        mx = max(float(np.nanmax(np.abs(z[sp].astype(np.float64) - S[sp]))) for sp in ("train", "val", "test"))
        check(f"(c) {kind} 같은 시드 두 번 같은 값: timing_ae(다른 프로세스) 저장 점수 = 이 점검 계산(비트 단위)", ok and same_meta,
              f"메타 같음 {same_meta}, 최대 차이 {mx:.3g}")
        return
    S2, _ = A.compute_scores(ctx, kind, SEED, k, None)
    ok = all(np.array_equal(S2[sp], S[sp]) for sp in ("train", "val", "test"))
    check(f"(c) {kind} 같은 시드 두 번 같은 값(이 프로세스에서 두 번 계산, timing_ae 저장 점수 없음)", ok)


def t_window(ctx, k, kind, dbg):
    dev = torch.device("cuda")
    d = dbg["models"]["full"]
    W = ctx.weather.W
    seq_np = A.seq_table(W, d["prep"]["mu"], d["prep"]["sd"])
    v = ctx.sp["val"]
    h = 7
    _, first = np.unique(v["s"], return_index=True)              # 시군구가 서로 다른 행(칸 바꾸기가 서로 섞이지 않게)
    rows = np.sort(first)[:50]
    s = v["s"][rows].astype(np.int64)
    T = v["d"][rows].astype(np.int64) - h
    tb = torch.from_numpy(seq_np).to(dev)
    st, Tt = torch.from_numpy(s).to(dev), torch.from_numpy(T).to(dev)
    dd = {"tb": tb, "s": st, "T": Tt, "ar": torch.arange(E.L, device=dev)}
    x = A.SeqInput.batch(dd, torch.arange(len(rows), device=dev))[0].cpu().numpy()
    ref = np.stack([seq_np[s[i], T[i] - E.L:T[i], :].T for i in range(len(rows))])
    check(f"(d) {kind} 배치 입력 (B, 38, 50) = build_seq 표의 [T−50, T−1] 구간(위치 0 = T−50, 49 = T−1, 비트 단위)", np.array_equal(x, ref))
    res = {}
    with A.deterministic_torch():
        net = _rebuild(kind, d, dev)
        base = A.score_rows(net, A.SeqInput.batch, dd, np.arange(len(rows)), dev)
        for off in (0, 1, E.L, E.L + 1):
            tb2 = tb.clone()
            tb2[st, Tt - off, :] += 50.0
            sc = A.score_rows(net, A.SeqInput.batch, dict(dd, tb=tb2), np.arange(len(rows)), dev)
            res[off] = float(np.mean(sc != base))
    check(f"(d) {kind} 창 = T−50 … T−1: 칸을 바꾸면 T−1·T−50 은 점수가 바뀌고 T(당일)·T−51 은 그대로",
          res[0] == 0 and res[E.L + 1] == 0 and res[1] == 1 and res[E.L] == 1,
          f"바뀐 행 비율 T {res[0]}, T−1 {res[1]}, T−50 {res[E.L]}, T−51 {res[E.L + 1]} (val 50행, h={h})")


def t_placebo(ctx, k, kind, S):
    P1 = A.placebo_scores(ctx, S, SEED)
    P2 = A.placebo_scores(ctx, S, SEED)
    ok_rep = all(np.array_equal(P1[sp], P2[sp]) for sp in P1)
    ht = ctx.h_fit(SEED)["train"].astype(np.int64)
    ok_set, n_grp, moved, tot = True, 0, 0, 0
    yr = A.calendar_year(ctx, ctx.sp["train"]["d"].astype(np.int64) - ht)
    for g in np.unique(yr):
        m = yr == g
        ok_set &= bool(np.array_equal(np.sort(P1["train"][m]), np.sort(S["train"][m])))
        n_grp += 1
    moved += int(np.sum(P1["train"] != S["train"]))
    tot += len(S["train"])
    for sp in ("val", "test"):
        d = ctx.sp[sp]["d"].astype(np.int64)
        for h in range(1, E.H_MAX + 1):
            yr = A.calendar_year(ctx, d - h)
            for g in np.unique(yr):
                m = yr == g
                ok_set &= bool(np.array_equal(np.sort(P1[sp][h - 1][m]), np.sort(S[sp][h - 1][m])))
                n_grp += 1
        moved += int(np.sum(P1[sp] != S[sp]))
        tot += S[sp].size
    check(f"(e) {kind} 위약판: 두 번 만들어도 같음, (분할, h, 예측일 T 연도) 묶음 {n_grp}개마다 값 모음 같음", ok_rep and ok_set)
    check(f"(e) {kind} 위약판: 대부분 자리가 바뀜", moved / tot > 0.95, f"바뀐 자리 {moved / tot:.4f}")
    from e2models import FeatureBuilder
    fb = FeatureBuilder(ctx, dict(E.EXP_BY_ID[f"A3_{kind}_p"]["feats"], k=k), seed=SEED)
    hf = ctx.h_fit(SEED)
    n_tr, n_va = len(ht), len(hf["val"])
    a = fb.num("train", np.arange(n_tr), hf["train"])[:, -1]
    b = fb.num("val", np.arange(n_va), hf["val"])[:, -1]
    c = fb.num("test", np.arange(len(ctx.sp["test"]["idx"])), 17)[:, -1]
    ok = (np.array_equal(a, P1["train"]) and np.array_equal(b, P1["val"][hf["val"].astype(np.int64) - 1, np.arange(n_va)])
          and np.array_equal(c, P1["test"][16]))
    check(f"(e) A3_{kind}_p 칸 = 위약 행렬 값(학습 = 그 시드 h, 검증 = 조기 종료용 h, 시험 h=17)", ok)


def t_matrix(ctx, k, kind, S, rng):
    from e2models import FeatureBuilder
    b0 = FeatureBuilder(ctx, dict(E.EXP_BY_ID["A3"]["feats"], k=k))
    b1 = FeatureBuilder(ctx, dict(E.EXP_BY_ID[f"A3_{kind}"]["feats"], k=k), seed=SEED)
    hf = ctx.h_fit(SEED)
    bad = []
    pos = np.sort(rng.choice(len(hf["train"]), 5000, replace=False))
    X0, X1 = b0.num("train", pos, hf["train"][pos]), b1.num("train", pos, hf["train"][pos])
    if not (X1.shape[1] == X0.shape[1] + 1 and np.array_equal(X1[:, :-1], X0, equal_nan=True) and np.array_equal(X1[:, -1], S["train"][pos])):
        bad.append("train")
    for sp, h in (("val", "es"), ("val", 13), ("test", 60)):
        n = len(ctx.sp[sp]["idx"])
        pp = np.sort(rng.choice(n, 3000, replace=False))
        hh = hf["val"][pp] if h == "es" else h
        X0, X1 = b0.num(sp, pp, hh), b1.num(sp, pp, hh)
        col = S[sp][np.asarray(hh, dtype=np.int64) - 1, pp] if h == "es" else S[sp][h - 1, pp]
        if not (X1.shape[1] == X0.shape[1] + 1 and np.array_equal(X1[:, :-1], X0, equal_nan=True) and np.array_equal(X1[:, -1], col)):
            bad.append(f"{sp} h={h}")
    check(f"(f) A3_{kind} 특성 행렬 = A3 행렬(비트 단위) + 점수 1칸(학습 = 그 시드 h, 검증 조기 종료용 h·h=13, 시험 h=60)",
          not bad and b1.names == b0.names + A.ae_names([kind]), f"다름 {bad}, 칸 {len(b1.names)}")
    try:
        b1.num("train", pos[:10], 5)
        raised = False
    except ValueError:
        raised = True
    check(f"(f) A3_{kind} 학습 행을 그 시드 h 가 아닌 h 로 부르면 오류", raised)


def t_pred_format(ctx, kind, S):
    d = TIMING_ROOT / "results" / E.run_dir_name(f"S_{kind}", SEED)
    ref = E.EXP2_DIR / "results" / E.run_dir_name("E00", SEED)
    if not (d / "summary.json").exists():
        check(f"(g) S_{kind} 예측 파일 형식(timing_ae 결과 필요)", False, f"없음: {d}")
        return
    ok_all, notes = True, []
    for sp in ("val", "test"):
        a, b = pq.read_table(d / f"pred_{sp}.parquet"), pq.read_table(ref / f"pred_{sp}.parquet")
        same_schema = a.schema.names == b.schema.names and [str(t) for t in a.schema.types] == [str(t) for t in b.schema.types]
        ta, tb = a.to_pandas(), b.to_pandas()
        keys = ["sgg_code", "train_prdlst_code", "train_prdlst_code_2", "d", "h", "y"]
        same_keys = len(ta) == len(tb) and all(np.array_equal(ta[c].astype(str).to_numpy(), tb[c].astype(str).to_numpy()) for c in keys)
        same_p = np.array_equal(ta["p"].to_numpy(), S[sp].reshape(-1))
        ok_all &= same_schema and same_keys and same_p
        notes.append(f"{sp}: 형식 {same_schema}, 키·h·y·순서 {same_keys}, p = 점수 {same_p}, {len(ta):,}행")
    ma = pd.read_csv(d / "metrics_by_h.csv").columns.tolist()
    mb = pd.read_csv(ref / "metrics_by_h.csv").columns.tolist()
    sa = set(json.loads((d / "summary.json").read_text(encoding="utf-8")))
    sb = set(json.loads((ref / "summary.json").read_text(encoding="utf-8")))
    check(f"(g) S_{kind} 예측 파일 = E00 과 같은 형식(칸·형식·행 순서·키·h·y), p = 저장 점수", ok_all, "; ".join(notes))
    check(f"(g) S_{kind} metrics_by_h.csv 칸 = E00, summary.json 키 = E00", ma == mb and sa == sb,
          f"summary 키 차이 {sorted(sa ^ sb)}")


# ---------------------------------------------------------------- (h)
H_IDS = ["E00", "A1", "A3", "B2", "D3", "C4_hist", "F4", "G2", "M1", "Q4", "P3", "P4p", "A3_N1N2"]


def t_existing(ctx, k, rng):
    O = _load("e2models_old_ae", OLD_E2MODELS)
    from e2models import CatRunner, FeatureBuilder, LRRunner
    a_best = json.loads((E.EXP2_DIR / "results" / "selection_A.json").read_text(encoding="utf-8"))["best"]
    bad = []
    for eid in H_IDS:
        f = E.EXP_BY_ID[eid]["feats"]
        f = dict(E.EXP_BY_ID[a_best]["feats"] if f == "A*" else f)
        if f.get("k") == "k*":
            f["k"] = k
        fo, fn = O.FeatureBuilder(ctx, f), FeatureBuilder(ctx, f, seed=SEED)
        ok = fo.names == fn.names and fo.cat_names == fn.cat_names
        for sp in C1.SPLITS:
            pos = np.sort(rng.choice(len(ctx.sp[sp]["idx"]), 500, replace=False))
            hh = rng.integers(1, E.H_MAX + 1, size=500)
            for h in (hh, int(hh[0])):
                ok &= bool(np.array_equal(fo.num(sp, pos, h), fn.num(sp, pos, h), equal_nan=True))
            co, cn = fo.cats(sp, pos), fn.cats(sp, pos)
            ok &= (co is None and cn is None) or bool(np.array_equal(co, cn))
        if not ok:
            bad.append(eid)
    check("(h) 기존 실험 특성 행렬·범주·칸 이름 = 고치기 전 e2models (비트 단위, 분할마다 500행, h 배열·정수)", not bad,
          f"{len(H_IDS)}개 {H_IDS}, 다름 {bad}")
    rows, nb = [], 0
    for d in sorted((E.EXP2_DIR / "results").glob("*__s42")):
        cf = d / "config.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if c.get("model") not in ("cat", "lr"):
            continue
        exp = E.EXP_BY_ID.get(c["exp"]) or dict(id=c["exp"], model=c["model"], feats=c["feats"])
        R = CatRunner if c["model"] == "cat" else LRRunner
        r = R(ctx, exp, 42, c["feats"], lambda *_: None, rounds=1, patience=1) if c["model"] == "cat" else R(ctx, exp, 42, c["feats"], lambda *_: None)
        Ro = (O.CatRunner(ctx, exp, 42, c["feats"], lambda *_: None, 1, 1) if c["model"] == "cat"
              else O.LRRunner(ctx, exp, 42, c["feats"], lambda *_: None))
        ok = (r.cols == c["feature_columns"] and r.n_features_desc == c["n_features_desc"] and r.n_features == c["n_features"]
              and Ro.cols == r.cols and Ro.n_features_desc == r.n_features_desc)
        nb += int(not ok)
        rows.append(c["exp"] + ("" if ok else "(다름)"))
    check("(h) 기존 결과(results/*__s42 CatBoost·로지스틱) config.json 입력 칸·칸 수·설명 = 지금 코드 = 고치기 전 코드", nb == 0 and len(rows) > 40,
          f"{len(rows)}개, 다름 {nb}")


def t_tables():
    """고치기 전 e2tables(scratch/ae/e2tables_before_ae.py.bak, 실험 목록도 고치기 전 것만)와 지금 e2tables 로 같은 results/ 에서 표를 각각
    scratch/ae/selftest_tables/{old,new}/ 에 만들어 비교(tables/실험기록.csv 는 그 뒤 다른 세션이 results/ 에 H60 을 넣어 비교 기준으로 쓰지 않음).
    results/ 의 선정 파일 쓰기(selection_*.json)는 scratch 로 돌림."""
    import e2tables as T
    O = _load("e2tables_old_ae", SCR / "e2tables_before_ae.py.bak")
    real = (E.EXP2_DIR / "results").resolve()
    out = SCR / "selftest_tables"
    redir = out / "selection_redirected"
    orig = {m: m.jdump for m in (T, O)}

    def jd(obj, p):                                              # results/ 의 선정 파일 쓰기를 scratch 로 돌림(results/ 에 쓰지 않음)
        p = Path(p)
        if p.resolve().parent == real:
            redir.mkdir(parents=True, exist_ok=True)
            p = redir / p.name
        return orig[T](obj, p)

    old_exps = [e for e in E.EXPERIMENTS if e["stage"] not in ("hx", "ae")]
    full_exps = list(E.EXPERIMENTS)
    for m in (T, O):
        m.jdump = jd
    try:
        for name, mod, exps in (("old", O, old_exps), ("new", T, full_exps)):
            P = {"results": real, "tables": out / name, "logs": out / "logs", "cache": E.EXP2_DIR / "cache"}
            P["tables"].mkdir(parents=True, exist_ok=True)
            E.EXPERIMENTS = exps
            mod.build_tables(P, False, None)
    finally:
        E.EXPERIMENTS = full_exps
        for m, f in orig.items():
            m.jdump = f
    rd = lambda n, f: pd.read_csv(out / n / f, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    old, new = rd("old", "실험기록.csv"), rd("new", "실험기록.csv")
    new_old = new[new["실험 ID"].isin(old["실험 ID"])].reset_index(drop=True)
    same_cols = list(new.columns) == list(old.columns)
    diff, fmt_only = [], []
    if same_cols and len(new_old) == len(old):
        for c in old.columns:
            m = (old[c] != new_old[c]).to_numpy()
            if not m.any():
                continue
            # 결과 없는 줄(새 실험 15개)이 있으면 pandas 가 정수 칸을 실수로 읽어 '222793' -> '222793.0' 처럼 쓰는 것만 다른 경우는 값이 같음
            a = pd.to_numeric(old.loc[m, c], errors="coerce").to_numpy()
            b = pd.to_numeric(new_old.loc[m, c], errors="coerce").to_numpy()
            if not np.isnan(a).any() and np.array_equal(a, b):
                fmt_only.append(c)
            else:
                diff.append((c, int(m.sum()), list(old.loc[m, "실험 ID"])[:5]))
    added = [i for i in new["실험 ID"] if i not in set(old["실험 ID"])]
    want = [e["id"] for e in E.EXPERIMENTS if e["stage"] in ("hx", "ae")]
    if fmt_only:
        say(f"    [기록] 표기만 다른 칸(값 같음 - 결과 없는 새 줄 때문에 정수가 '.0' 붙은 실수로 써짐, 전체 실행 뒤엔 사라짐): {fmt_only}")
    check("(h) 실험기록 표(같은 results/ 에서 고치기 전·지금 코드로 각각): 기존 실험 줄 칸·값·순서 모두 같음, 새 줄 = hx·ae 실험 15개",
          same_cols and len(new_old) == len(old) and list(new_old["실험 ID"]) == list(old["실험 ID"]) and not diff and added == want,
          f"기존 {len(old)}줄, 다른 칸 {diff}, 새 줄 {added}")
    so_, sn_ = rd("old", "시드별.csv"), rd("new", "시드별.csv")
    sn_old = sn_[sn_["실험 ID"].isin(set(so_["실험 ID"]))].reset_index(drop=True)
    check("(h) 시드별·고위험품목비교 표: 기존 줄 같음", list(so_.columns) == list(sn_.columns) and so_.equals(sn_old)
          and rd("old", "고위험품목비교.csv").equals(rd("new", "고위험품목비교.csv")), f"시드별 {len(so_)}줄")
    so = pd.read_excel(out / "old" / "실험기록.xlsx", sheet_name="설정", dtype=str).fillna("")
    sn = pd.read_excel(out / "new" / "실험기록.xlsx", sheet_name="설정", dtype=str).fillna("")
    old_pairs = list(zip(so["항목"], so["값"]))
    new_pairs = list(zip(sn["항목"], sn["값"]))
    missing = [p[0] for p in old_pairs if p not in new_pairs]
    add_keys = [p[0] for p in new_pairs if p not in old_pairs]
    check("(h) 설정 시트: 기존 줄 그대로 + 새 줄(HX·AE)", not missing and new_pairs[:len(old_pairs)] == old_pairs
          and any(x.startswith("HX") for x in add_keys) and any(x.startswith("AE") for x in add_keys), f"빠진 줄 {missing}, 새 줄 {add_keys}")


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    log = Logger(SCR / "progress_selftest_ae.log")
    rng = np.random.default_rng(20261019)
    t_config()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    from selftest_fg import _cache_meta_ok
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        say("    기상 요약 캐시 메타가 다름 -> 캐시를 쓰지 않음")
    k = k_star()
    from e2dl_impl import build_seq
    mu, sd = ctx.weather.train_period_stats()
    check("(a) build_seq 재사용: 학습 기간 통계를 넣은 seq_table = build_seq(ctx) (비트 단위)",
          np.array_equal(A.seq_table(ctx.weather.W, mu, sd), build_seq(ctx)[0]))
    Xtab = tab_X(ctx, k)
    seq_ref, seq_prep = {}, None
    for kind in A.KINDS:
        tk = time.time()
        S, info, dbg = A.compute_scores(ctx, kind, SEED, k, log, debug=True)
        say(f"    [{kind}] 점수 계산 {time.time() - tk:.0f}s, 기록용 {info['record_only']}")
        if not A.is_tab(kind) and not seq_ref:
            y = ctx.sp["train"]["y"]
            neg = np.flatnonzero(y == 0)
            for m in ("full", 0):
                rows = neg if m == "full" else neg[dbg["fold"] != 0]
                seq_ref[m] = seq_stats_direct(ctx, rows)
        if not A.is_tab(kind):                                   # AEs·VAEs 는 같은 행 -> 같은 정규화
            pk = {m: (d["prep"]["mu64"], d["prep"]["sd64"]) for m, d in dbg["models"].items()}
            if seq_prep is None:
                seq_prep = pk
            else:
                check("(a) AEs·VAEs 정규화 통계 같음(같은 시드 = 같은 행)", all(np.array_equal(seq_prep[m][0], pk[m][0])
                                                                         and np.array_equal(seq_prep[m][1], pk[m][1]) for m in pk))
        t_rows_prep(ctx, k, kind, S, dbg, Xtab, seq_ref)
        t_oof(ctx, k, kind, S, dbg)
        t_repro(ctx, k, kind, S)
        if not A.is_tab(kind):
            t_window(ctx, k, kind, dbg)
        ctx.__dict__.setdefault("_ae_scores", {})[(kind, SEED, k)] = (S, info, "selftest")   # (e)(f) 가 다시 계산하지 않게
        t_placebo(ctx, k, kind, S)
        t_matrix(ctx, k, kind, S, rng)
        t_pred_format(ctx, kind, S)
        ctx._ae_scores.pop((kind, SEED, k), None)
        ctx.__dict__.get("_ae_placebo", {}).clear()
        del dbg
        torch.cuda.empty_cache()
    del Xtab
    t_existing(ctx, k, rng)
    t_tables()
    say(f"\n결과: 실패 {len(fails)}개 {fails} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)
