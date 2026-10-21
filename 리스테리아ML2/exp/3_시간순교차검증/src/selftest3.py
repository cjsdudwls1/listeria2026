# -*- coding: utf-8 -*-
"""3차 틀 점검(사양 4절 1번, 실행 전 실패 0).

python selftest3.py --root <결과 루트> [--parts all|bounds,rows,cache,score,ap,params,past,guard,perm,cat,rawes] [--scan-root <다른 결과 루트> ...]
(기상 창은 설정/환경변수 EXP2_L, 기본 100)

bounds : 날짜 경계가 사양 1절 표대로인지(겹 학습 끝 = 맞혀 보는 해 시작 전날 06-30, 최종 2025-06-30, 결과 시차 앞당김 없음) - 기본·가짜 경계·직접 지정
rows   : 겹별 학습 행·맞혀 보는 행·시험 행이 경계대로인지(기본 + 가짜 경계), 겹 행 수·검출 수, 시험 행이 어떤 겹의 학습·맞혀 보는 행에도 없음,
         창이 기상표 안
past   : 과거 검사 결과 칸(과거 이력·WOE·N1, 전국 p0) - 학습 행 = T 전날까지 굴러가는 창(T 당일·이후 시료를 바꿔도 그대로, 대조: T 전날을 바꾸면 달라짐),
         맞혀 보는 행 = 그 겹 학습 끝 값 고정(학습 끝 뒤 시료를 바꿔도 그대로, 대조: 학습 끝 이전 시료를 바꾸면 달라짐, 같은 단위면 h·T 가 달라도 같은 값,
         2차 계산에 예측일 = 학습 끝 + 1일을 넘긴 값과 같음)
guard  : 고르는 단계에서 시험 행을 만들면 오류로 멈춤
cache  : 맞혀 보는 행 입력 캐시 = FeatureBuilder 직접 계산(비트 단위; 기본 입력 h 4개 블록 전체 + '변수 전부'(이상치 점수 제외) 표본 행)
score  : 검사축소율·상위 n%·판정 기준값(Recall≥0.8 중 Precision 최대, h 합친 행)·구간 가중 PR-AUC 손계산
ap     : 조기 종료용 빠른 AP = sklearn, 원값 단계별 AP·조기 종료 재현·묶음 이어 붙인 곡선
perm   : 묶음 섞기·기상 변수 섞기 중요도 손계산, 기상 변수 칸 찾기, 원값 칸·BiLSTM 채널 고르기
params : 트리 3종 - 파라미터 저장(get_params JSON) 뒤 불러와(set_params) 다시 학습하면 예측이 비트 단위로 같음
cat    : 범주 입력 = [sido, sgg, p2, p3] (+ 식품 묶음이면 aw_label) - 결과 폴더 config.json 전부(--scan-root)
rawes  : 원값 입력 조기 종료 재현(raw3) = 실제 조기 종료(models3) - 최적 반복·예측 대조(작은 겹, 기상 변수 일부 + 덧붙인 칸 판 포함)
결과: <root>/results/_점검/selftest3_L<창>.json
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONUTF8", "1")

import cfg3 as G   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import cv as CV3      # noqa: E402
import engine3 as EN  # noqa: E402
import models3 as M3  # noqa: E402
import metrics as M1  # noqa: E402
from common import Logger  # noqa: E402

E00_BEST = 31
BASE4 = ["sido", "sgg", "p2", "p3"]


def bits_equal(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def cmp_pred(p, q):
    d = np.abs(p.astype(np.float64) - q.astype(np.float64))
    return {"bit_equal": bool(bits_equal(p, q)), "n": int(len(p)), "n_diff": int((p.view(np.uint32) != q.view(np.uint32)).sum()),
            "max_abs_diff": float(d.max()) if len(d) else 0.0}


class Checker:
    def __init__(self, log):
        self.log, self.items = log, []

    def __call__(self, name, cond, detail=None):
        self.items.append({"name": name, "ok": bool(cond), "detail": detail})
        self.log(f"  [{'통과' if cond else '실패'}] {name}" + (f" | {detail}" if detail is not None else ""))
        return bool(cond)


def swap_bounds(b):
    G.BOUNDS, G.FOLDS = b, b["folds"]
    G.FOLD_BY_K = {f["fold"]: f for f in b["folds"]}


def part_bounds(chk, rep):
    """손으로 적은 기댓값(사양 1절 표)."""
    cfg = {"dates": {"data_end": "2026-06-30", "n_folds": 3, "folds": None, "test": None}}
    b = G.compute_bounds(cfg)
    exp = [("2022-06-30", "2022-07-01", "2023-05-01"), ("2023-06-30", "2023-07-01", "2024-05-01"), ("2024-06-30", "2024-07-01", "2025-05-01")]
    ok = [(f["train_end"], f["t_start"], f["t_end"]) for f in b["folds"]] == exp and b["test"]["t_start"] == "2025-07-01" \
        and b["test"]["t_end"] == "2026-05-01" and b["learn_end"] == "2025-06-30" and b["cv_end"] == "2025-06-30" and "result_lag_days" not in b
    chk("경계(데이터 끝 2026-06-30): 사양 1절 표와 같음(겹 학습 끝 2022-06-30·2023-06-30·2024-06-30, 맞혀 보는 T 07-01~05-01, 최종 2025-06-30, "
        "시험 2025-07-01~2026-05-01, 결과 시차 앞당김 없음)", ok, b)
    cfg2 = {"dates": {"data_end": "2025-12-31", "n_folds": 3, "folds": None, "test": None}}
    b2 = G.compute_bounds(cfg2)
    exp2 = [("2021-12-31", "2022-01-01", "2022-11-01"), ("2022-12-31", "2023-01-01", "2023-11-01"), ("2023-12-31", "2024-01-01", "2024-11-01")]
    ok2 = [(f["train_end"], f["t_start"], f["t_end"]) for f in b2["folds"]] == exp2 and b2["test"]["t_start"] == "2025-01-01" \
        and b2["test"]["t_end"] == "2025-11-01" and b2["learn_end"] == "2024-12-31"
    chk("가짜 경계(데이터 끝 2025-12-31): 시험 2025-01-01~2025-11-01, 겹 학습 끝 2021-12-31·2022-12-31·2023-12-31, 최종 2024-12-31", ok2, b2)
    cfg3 = {"dates": {"data_end": "2026-06-30", "folds": [{"train_end": "2020-06-30", "t_start": "2020-07-01", "t_end": "2021-05-01"}],
                      "test": {"t_start": "2025-07-01", "t_end": "2026-05-01"}}}
    b3 = G.compute_bounds(cfg3)
    chk("겹·시험 직접 지정이 그대로 쓰임", len(b3["folds"]) == 1 and b3["folds"][0]["t_end"] == "2021-05-01", b3["folds"])
    try:
        G.compute_bounds({"dates": {"data_end": "2026-06-30", "folds": [{"train_end": "2024-06-30", "t_start": "2024-07-01",
                                                                           "t_end": "2025-06-01"}], "test": None}})
        bad = False
    except ValueError:
        bad = True
    chk("겹 맞혀 보는 행이 시험 시작 전날을 넘는 지정은 오류로 멈춤", bad)
    chk("지금 쓰는 경계 = 기본 경계(데이터 끝 2026-06-30)", G.BOUNDS["folds"] == b["folds"] and G.BOUNDS["learn_end"] == b["learn_end"], G.BOUNDS["folds"])
    rep["bounds"] = {"default": b, "fake_2025_12_31": b2}
    rep["bounds_in_use"] = G.BOUNDS


def fold_row_checks(chk, cvo, label):
    out = {}
    D = np.datetime64(G.BOUNDS["data_end"], "D")
    for f in G.FOLDS:
        k = f["fold"]
        rtr, rev = cvo.fold_train(k), cvo.eval_rows(k)
        ts, te = np.datetime64(f["t_start"], "D"), np.datetime64(f["t_end"], "D")
        days = int((te - ts).astype(int)) + 1
        offs = rev.h_offsets()
        tdays = sorted({len(np.unique(rev.T[a:b])) for a, b in offs})
        ok = (rtr.d.max() <= np.datetime64(f["train_end"], "D") and rev.T.min() >= ts and rev.T.max() <= te
              and rev.d.min() > np.datetime64(f["train_end"], "D") and rev.d.max() <= np.datetime64(G.BOUNDS["cv_end"], "D")
              and (cvo.smoke.get("enabled") or tdays == [days]))
        chk(f"[{label}] (나) 겹 {k}: 학습 행 d 최댓값 {rtr.d.max()} <= 학습 끝 {f['train_end']}, 맞혀 보는 T ∈ [{f['t_start']}, {f['t_end']}], 모든 h 같은 {days}일", ok,
            {"train_d_max": str(rtr.d.max()), "eval_T": [str(rev.T.min()), str(rev.T.max())], "T_days_per_h": tdays})
        out[k] = (rtr, rev)
    lr0 = cvo.learn_rows()
    chk(f"[{label}] (나) 최종 학습 행 d 최댓값 {lr0.d.max()} <= 최종 학습 끝 {G.BOUNDS['learn_end']}",
        lr0.d.max() <= np.datetime64(G.BOUNDS["learn_end"], "D"))
    te_ = cvo.test_rowset("selftest 점검(위치만)", allow=True)
    t = G.BOUNDS["test"]
    ok = te_.T.min() >= np.datetime64(t["t_start"]) and te_.T.max() <= np.datetime64(t["t_end"]) and te_.d.max() <= D
    keys = set(te_.pos.tolist())
    inter = {k: (len(keys & set(a.pos.tolist())), len(keys & set(b.pos.tolist()))) for k, (a, b) in out.items()}
    lr = cvo.learn_rows()
    chk(f"[{label}] 시험 T ∈ [{t['t_start']}, {t['t_end']}], d <= 데이터 끝 {G.BOUNDS['data_end']}", ok,
        [str(te_.T.min()), str(te_.T.max()), str(te_.d.max())])
    chk(f"[{label}] 시험 행이 어떤 겹의 학습·맞혀 보는 행·최종 학습 행에도 없음", all(v == (0, 0) for v in inter.values())
        and not (keys & set(lr.pos.tolist())) and lr.d.max() < te_.d.min(), {"겹": inter, "learn_d_max": str(lr.d.max()), "test_d_min": str(te_.d.min())})
    CV3.TEST_ACCESS.clear()
    return out, te_


def part_rows(chk, rep, cvo, args):
    S = cvo.ctx.samples
    sd = S["d"].to_numpy().astype("datetime64[D]")
    spos = S["pos"].to_numpy().astype(np.int64)
    pairs, te_ = fold_row_checks(chk, cvo, "기본 경계")
    folds = {}
    for f in G.FOLDS:
        k = f["fold"]
        rtr, rev = pairs[k]
        ts, te = np.datetime64(f["t_start"], "D"), np.datetime64(f["t_end"], "D")
        y1 = np.datetime64(str(int(f["t_start"][:4]) + 1) + f["t_start"][4:], "D") - np.timedelta64(1, "D")
        offs = rev.h_offsets()
        ph = [(int(b - a), int(rev.y[a:b].sum())) for a, b in offs]
        u = pd.DataFrame({"pos": rev.pos, "y": rev.y}).drop_duplicates("pos")
        folds[k] = {"train_rows": int(rtr.n), "train_pos_rows": int(rtr.y.sum()), "eval_pairs": int(rev.n), "eval_pos_pairs": int(rev.y.sum()),
                    "eval_rows_per_h_min_max": [min(x for x, _ in ph), max(x for x, _ in ph)],
                    "eval_pos_per_h_min_max_mean": [min(x for _, x in ph), max(x for _, x in ph), float(np.mean([x for _, x in ph]))],
                    "eval_unique_unit_day_rows": int(len(u)), "eval_unique_pos_rows": int(u["y"].sum()),
                    "samples_pos_year_d": int(spos[(sd >= ts) & (sd <= y1)].sum()),
                    "samples_pos_eval_d_range": int(spos[(sd > ts) & (sd <= te + 60)].sum()),
                    "samples_pos_d_in_T_range": int(spos[(sd >= ts) & (sd <= te)].sum()),
                    "spec_pos_samples": {1: 230, 2: 196, 3: 164}.get(k)}
        cvo.log(f"  겹 {k}: {folds[k]}")
        chk(f"겹 {k}: 모든 h 에 검출·불검출 있음", all(0 < p < n for n, p in ph))
    rep["folds"] = folds
    rep["test_rows"] = {"pairs": int(te_.n), "unit_day_rows": int(len(np.unique(te_.pos))), "d_range": [str(te_.d.min()), str(te_.d.max())],
                        "note": "시험은 위치·날짜만 셈(입력·예측·채점 없음)"}
    # 겹 3 = 2차
    rtr3, rev3 = pairs[3]
    n_a = int((cvo.date <= np.datetime64("2024-06-30")).sum())
    h2 = np.random.default_rng(cvo.seed).integers(1, 61, size=n_a)
    # 2차 대조는 옛 자료·기본 경계에서만(--e00-pred 를 줄 때) - 새 자료에서는 건너뜀
    ok = (rtr3.n == 222793 and int(rtr3.y.sum()) == 851 and np.array_equal(rtr3.h, h2[: rtr3.n]))         if (args.e00_pred and not cvo.smoke.get("enabled")) else None
    if ok is not None:
        chk("겹 3 학습 행 = 2차 학습 분할(222,793행/검출 851), h = 2차 규칙 default_rng(42)", ok)
    if args.e00_pred and not cvo.smoke.get("enabled"):
        t = pq.read_table(args.e00_pred).to_pandas()
        d = t["d"].to_numpy().astype("datetime64[D]")
        h = t["h"].to_numpy().astype(np.int64)
        T = d - h.astype("timedelta64[D]")
        e = t[(T >= np.datetime64("2024-07-01")) & (T <= np.datetime64("2025-05-01"))].reset_index(drop=True)
        kf = rev3.keys_frame()
        same = (len(e) == rev3.n and all(np.array_equal(e[c].astype(str).to_numpy(), np.asarray(kf[c].to_pandas()).astype(str))
                                         for c in ("sgg_code", "train_prdlst_code", "train_prdlst_code_2"))
                and np.array_equal(e["d"].to_numpy().astype("datetime64[D]"), rev3.d)
                and np.array_equal(e["h"].to_numpy().astype(np.int64), rev3.h.astype(np.int64))
                and np.array_equal(e["y"].to_numpy().astype(np.int8), rev3.y))
        chk("겹 3 맞혀 보는 행 = 2차 E00 pred_val 의 T 기준 행(키·d·h·y, 같은 순서)", same, {"2차": len(e), "3차": rev3.n})
        rep["_e00_p"] = e["p"].to_numpy().astype(np.float32) if same else None
    if args.hcount and not cvo.smoke.get("enabled"):
        hc = pd.read_csv(args.hcount, encoding="utf-8-sig")
        hc = hc[hc.iloc[:, 0].astype(str).str.startswith("통합")]
        hc = hc[pd.to_numeric(hc["h"], errors="coerce").notna()].copy()
        hc["h"] = hc["h"].astype(int)
        hc = hc.set_index("h").loc[range(1, 61)]
        offs = rev3.h_offsets()
        ok = (np.array_equal(hc["검증 행 수"].to_numpy().astype(int), offs[:, 1] - offs[:, 0])
              and np.array_equal(hc["검출 행 수"].to_numpy().astype(int), [int(rev3.y[a:b].sum()) for a, b in offs]))
        chk("겹 3 h 별 행 수·검출 수 = 2차 analysis/T기준_통합_L100/h별_행수_검증.csv '통합(T 기준)'", ok)
    win = cvo.check_windows([r for pr in pairs.values() for r in pr] + [cvo.learn_rows()])
    chk(f"모든 창(T−{G.L} .. T−1)이 기상표 안", True, win)
    # 가짜 경계
    orig = G.BOUNDS
    fake = G.compute_bounds({"dates": {"data_end": "2025-12-31", "n_folds": 3, "folds": None, "test": None}})
    swap_bounds(fake)
    try:
        cvf = CV3.CV(cvo.ctx, cvo.log, cvo.cache_dir, seed=cvo.seed, smoke=cvo.smoke)
        fold_row_checks(chk, cvf, "가짜 경계 2025-12-31")
        after = int((cvf.date > np.datetime64("2025-12-31")).sum())
        used = set()
        for f in fake["folds"]:
            used |= set(cvf.fold_train(f["fold"]).pos.tolist()) | set(cvf.eval_rows(f["fold"]).pos.tolist())
        used |= set(cvf.learn_rows().pos.tolist()) | set(cvf.test_rowset("selftest 가짜 경계", allow=True).pos.tolist())
        CV3.TEST_ACCESS.clear()
        late = set(np.flatnonzero(cvf.date > np.datetime64("2025-12-31")).tolist())
        chk("[가짜 경계] 데이터 끝(2025-12-31) 뒤 행은 어디에도 안 쓰임", not (used & late), {"끝 뒤 행": after})
    finally:
        swap_bounds(orig)


def part_cache(chk, rep, cvo):
    fb = cvo.builder(["wx"], G.K_SMOOTH_FIXED)
    chk(f"기본 입력 숫자 칸 = {1 + G.E.NW}(h 1 + 기상 {G.E.NW})", len(fb.names) == 1 + G.E.NW, len(fb.names))
    res = {}
    for f in G.FOLDS:
        k = f["fold"]
        rev = cvo.eval_rows(k)
        X, _ = cvo.assemble_X(rev, ["wx"], G.K_SMOOTH_FIXED, f"{'smoke_' if cvo.smoke.get('enabled') else ''}eval_f{k}")
        offs = rev.h_offsets()
        r = []
        for hh in (1, 17, 33, 60):
            a, b = offs[hh - 1]
            r.append(bool(bits_equal(np.asarray(X[a:b]), fb.num(CV3.SP, rev.pos[a:b], hh))))
        res[k] = r
    chk("맞혀 보는 행 입력 캐시(기본 입력, 묶음별 캐시를 모은 행렬) = FeatureBuilder 직접 계산(비트 단위, h 1·17·33·60 블록 전체)", all(all(v) for v in res.values()), res)
    lr = cvo.learn_rows()
    Xl, _ = cvo.assemble_X(lr, ["wx"], G.K_SMOOTH_FIXED, f"{'smoke_' if cvo.smoke.get('enabled') else ''}learn_s{cvo.seed}")
    idx = np.sort(np.random.default_rng(0).choice(lr.n, size=min(20000, lr.n), replace=False))
    chk("학습 행 입력 캐시 = 직접 계산(무작위 2만 행)", bits_equal(np.asarray(Xl[idx]), fb.num(CV3.SP, lr.pos[idx], lr.h[idx])))
    Xlh, fblh = cvo.assemble_X(lr, ["wx", "hist_core", "woe"], G.K_SMOOTH_FIXED, f"{'smoke_' if cvo.smoke.get('enabled') else ''}learn_s{cvo.seed}")
    sub_l = lr.subset(idx[:3000])
    Xdl, _ = cvo.compute_X(sub_l, ["wx", "hist_core", "woe"], G.K_SMOOTH_FIXED)
    fb_h2 = cvo.builder(["wx", "hist_core", "woe"], G.K_SMOOTH_FIXED)
    chk("학습 행 입력 캐시(기상 + 과거 이력 + WOE, 묶음별 캐시) = 직접 계산 = 2차 FeatureBuilder 에 h 그대로(굴러가는 창, 표본 3천 행)",
        bits_equal(np.asarray(Xlh[idx[:3000]]), Xdl) and bits_equal(Xdl, fb_h2.num(CV3.SP, sub_l.pos, sub_l.h)))
    allb = [b for b in CV3.resolve_featset("all") if b != "ae4"]
    fba = cvo.builder(allb, G.K_SMOOTH_FIXED)
    rev = cvo.eval_rows(G.FOLDS[0]["fold"])
    sub = rev.subset(np.sort(np.random.default_rng(1).choice(rev.n, size=min(3000, rev.n), replace=False)))
    Xs, fbs = cvo.assemble_X(sub, allb, G.K_SMOOTH_FIXED, "selftest_allsub")
    Xd, fbd = cvo.compute_X(sub, allb, G.K_SMOOTH_FIXED)
    ok = bits_equal(Xs, Xd) and fbs.names == fba.names == fbd.names
    # 과거 칸 규칙 확인: 맞혀 보는 행 과거 이력 칸 = FeatureBuilder 에 예측일 = 학습 끝 + 1일(h = d − 학습 끝 − 1)을 넘긴 값
    fbh = cvo.builder(["hist_core"], G.K_SMOOTH_FIXED)
    a0, b0 = 0, min(500, sub.n)
    he = (sub.d[a0:b0] - (np.datetime64(G.FOLDS[0]["train_end"], "D") + np.timedelta64(1, "D"))).astype(np.int64)
    ref = fbh.num(CV3.SP, sub.pos[a0:b0], he)[:, 1:]
    cols = [fbs.names.index(n) for n in fbh.names[1:]]
    chk(f"캐시 입력의 과거 이력 칸 = 2차 계산에 예측일 = 겹 학습 끝 + 1일을 넘긴 값(학습 끝 값 고정, 표본 {b0}행)", bits_equal(Xs[a0:b0][:, cols], ref))
    chk(f"'변수 전부'(이상치 점수 제외) 묶음별 캐시를 모은 입력 = 한꺼번에 직접 계산(표본 3,000 (행,h), 숫자 {len(fba.names)}칸, 칸 순서 같음)", ok)
    for q in cvo.cache_dir.glob("selftest_allsub_*"):
        q.unlink()
    nb, cb = CV3.column_bundles(fba.names, fba.cat_names, allb)
    rep["all_columns"] = {"n_num": len(fba.names), "cat": fba.cat_names, "by_bundle": pd.Series(nb).value_counts().to_dict()}


def part_score(chk, rep):
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], dtype=np.float32)
    df = CV3.per_h(y, p, np.array([[0, 10]]), 0.65)
    r = df.iloc[0]
    # 손계산: 기준값 0.65 -> 판정 = p >= 0.65 인 행 0,1,2 -> TP 2(행 0·2), FP 1(행 1), FN 1(행 9), TN 6, N 10
    # 검사축소율 = 1 - (2+1+1)/10 = 0.6 / 상위 10% = ceil(1) 행(행 0, 검출) -> 1/3 / 상위 20% = 2행(행 0·1) -> 1/3 / 상위 30% = 3행 -> 2/3
    ok = (r.tp, r.fp, r.fn, r.tn) == (2, 1, 1, 6) and abs(r.inspect_reduction - 0.6) < 1e-12 and abs(r.top10 - 1 / 3) < 1e-12 \
        and abs(r.top20 - 1 / 3) < 1e-12 and abs(CV3.topk_recall(y, p, 30) - 2 / 3) < 1e-12 and abs(r.top1 - 1 / 3) < 1e-12 \
        and abs(r.top5 - 1 / 3) < 1e-12
    chk("검사축소율·상위 n% 손계산 대조(10행 예: TP 2·FP 1·FN 1·TN 6, 검사축소율 0.6, 상위 1·5·10·20% 1/3, 30% 2/3)", ok, r.to_dict())
    # 기준값 규칙(Recall >= 0.80 중 Precision 최대) 손계산 ①: 확률 내림차순 정답 [1,1,0,1,0,1,0,0,1,0] (검출 5)
    #   기준값 0.6(위 6행): TP 4·FP 2 -> P 4/6 = 0.6667, R 0.8 / 0.3(위 9행): TP 5·FP 4 -> P 0.5556, R 1 / 그 밖 Recall 0.8 이상은 P 더 낮음 -> 0.6
    y1 = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int8)
    p1 = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    t1 = CV3.threshold_at_recall(y1, p1, 0.8)
    ok1 = abs(t1["threshold"] - 0.6) < 1e-7 and abs(t1["precision"] - 4 / 6) < 1e-12 and abs(t1["recall"] - 0.8) < 1e-12
    chk("기준값 손계산 ①(Recall ≥ 0.8 중 Precision 최대): 기준값 0.6, Precision 0.6667, Recall 0.8", ok1, t1)
    # ② 같은 Precision 이면 Recall 큰 쪽: 정답 [0,0,0,0,1,1,1,1,0,1] -> 위 8행 TP 4·FP 4(P 0.5, R 0.8) 와 위 10행 TP 5·FP 5(P 0.5, R 1) 동률 -> 10행(기준값 0.1)
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int8)
    p2 = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    t2 = CV3.threshold_at_recall(y2, p2, 0.8)
    ok2 = abs(t2["threshold"] - 0.1) < 1e-7 and abs(t2["precision"] - 0.5) < 1e-12 and abs(t2["recall"] - 1.0) < 1e-12
    chk("기준값 손계산 ②(같은 Precision 이면 Recall 큰 쪽): 기준값 0.1, Precision 0.5, Recall 1.0", ok2, t2)
    # ③ Recall 목표를 못 지키는 경우가 없는지: 무작위 예 300개(동점 포함)에서 늘 후보가 있고 고른 기준값의 Recall >= 0.8
    rng = np.random.default_rng(9)
    worst = 1.0
    for i in range(300):
        n = int(rng.integers(20, 3000))
        y = (rng.random(n) < rng.uniform(0.002, 0.3)).astype(np.int8)
        y[int(rng.integers(0, n))] = 1
        y[int(rng.integers(0, n))] = 0 if y.sum() < n else y[0]
        if y.sum() in (0, n):
            continue
        p = np.round(rng.random(n), int(rng.integers(1, 5))).astype(np.float32)
        t = CV3.threshold_at_recall(y, p, 0.8)
        cm = M1.confusion(y, p, t["threshold"])
        worst = min(worst, cm["recall"])
        assert t["n_candidates"] >= 1 and abs(cm["precision"] - t["precision"]) < 1e-12
    chk("Recall 목표 0.8 을 못 지키는 경우 없음(무작위 300개 예, 고른 기준값의 최소 Recall >= 0.8, 혼동행렬 Precision = 곡선 값)", worst >= 0.8 - 1e-12, worst)
    # ④ 판정 기준값은 60개 h 를 합친 예측에서: h 2개 예(h 마다 따로 정하면 다른 값) -> 합친 행에서 정한 값 = 손계산
    #   합친 정답(확률 내림차순) [1(0.9),0(0.85),1(0.8),0(0.7),1(0.6),0(0.5),0(0.4),1(0.3),0(0.2),0(0.1)] 검출 4
    #   Recall ≥ 0.8 -> 위 8행(기준값 0.3): TP 4·FP 4 -> P 0.5 가 유일(위 5행은 R 0.75) -> 기준값 0.3
    yh = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    ph = np.array([0.9, 0.6, 0.5, 0.2, 0.1, 0.85, 0.8, 0.7, 0.3, 0.4], dtype=np.float32)
    th = CV3.threshold_at_recall(yh, ph, 0.8)
    chk("기준값 손계산 ④(h 를 합친 행에서 하나): 기준값 0.3, Precision 0.5, Recall 1.0", abs(th["threshold"] - 0.3) < 1e-7 and abs(th["precision"] - 0.5) < 1e-12
        and abs(th["recall"] - 1.0) < 1e-12, th)
    # 구간 가중 손계산 ①: h 값 = h/100 -> 0.5x10.5 + 0.3x30.5 + 0.2x50.5 = 24.5 -> 0.245, h1 결측이면 구간 1 평균 = 11 -> 0.2475
    v = np.arange(1, 61) / 100.0
    w1 = CV3.band_weighted(v)
    v2 = v.copy()
    v2[0] = np.nan
    w2 = CV3.band_weighted(v2)
    chk("구간 가중 손계산 ①: h/100 -> 0.245, h1 결측 -> 0.2475(구간 평균에서 뺌)", abs(w1 - 0.245) < 1e-12 and abs(w2 - 0.2475) < 1e-12, [w1, w2])
    # 구간 가중 PR-AUC 손계산 ②: h 마다 4행, 정답 [1,0,0,1]
    #   h 1~20: 확률 [0.9,0.8,0.7,0.6] -> 순위 1·4 번째 검출 -> AP = (1/1 + 2/4)/2 = 0.75
    #   h 21~40: [0.6,0.9,0.8,0.7] -> 검출이 3·4 번째 -> AP = (1/3 + 2/4)/2 = 5/12
    #   h 41~60: [0.9,0.1,0.2,0.8] -> 검출이 1·2 번째 -> AP = 1
    #   구간 가중 = 0.5x0.75 + 0.3x5/12 + 0.2x1 = 0.7, 60일 평균 = (0.75 + 5/12 + 1)/3
    yb = np.tile(np.array([1, 0, 0, 1], dtype=np.int8), 60)
    pb = np.concatenate([np.tile([0.9, 0.8, 0.7, 0.6], 20), np.tile([0.6, 0.9, 0.8, 0.7], 20), np.tile([0.9, 0.1, 0.2, 0.8], 20)]).astype(np.float32)
    offs = np.array([[4 * i, 4 * i + 4] for i in range(60)])
    wpr = CV3.wpr(yb, pb, offs)
    s_, df_ = CV3.score_fold(yb, pb, offs)
    chk("구간 가중 PR-AUC 손계산 ②: 0.7 (score_fold 의 wpr_auc 도 같음, 60일 평균 = 0.72222)", abs(wpr - 0.7) < 1e-12 and abs(s_["wpr_auc"] - 0.7) < 1e-12
        and abs(s_["pr_auc"] - (0.75 + 5 / 12 + 1) / 3) < 1e-12, {"wpr": wpr, "score_fold": s_["wpr_auc"], "pr_auc": s_["pr_auc"]})
    rep["score_example"] = {"per_h": r.to_dict(), "thr": t1, "band": [w1, w2], "wpr": wpr}


def part_ap(chk, rep):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(3)
    mx = 0.0
    for i in range(50):
        n = int(rng.integers(50, 5000))
        y = (rng.random(n) < 0.05).astype(np.int8)
        y[0] = 1
        p = np.round(rng.random(n), int(rng.integers(1, 6))).astype(np.float32)   # 같은 값(동점) 포함
        mx = max(mx, abs(CV3.ap_fast(y, p) - average_precision_score(y, p)))
    chk("빠른 AP(조기 종료용) = sklearn average_precision (동점 포함 50개 예, 최대 차이 <= 1e-12)", mx <= 1e-12, mx)
    import raw3
    mx2 = 0.0
    for i in range(10):
        n, T = int(rng.integers(200, 3000)), int(rng.integers(5, 300))
        y = (rng.random(n) < 0.05).astype(np.int8)
        y[0] = 1
        M = np.round(rng.normal(size=(n, T)).cumsum(axis=1), int(rng.integers(0, 3)))
        a = raw3.staged_ap(y, M)
        b = np.array([average_precision_score(y, M[:, t]) for t in range(T)])
        mx2 = max(mx2, float(np.max(np.abs(a - b))))
    chk("원값 단계별 AP(열 단위 계산) = sklearn average_precision (동점 포함 10개 행렬, 최대 차이 <= 1e-12)", mx2 <= 1e-12, mx2)
    cur = [0.1, 0.3, 0.2, 0.3, 0.25] + [0.29] * 100 + [0.5]
    b1 = raw3.simulate_es(cur, 100)
    cur2 = [0.1, 0.2, 0.3] + [0.1] * 50
    b2 = raw3.simulate_es(cur2, 100)
    chk("조기 종료 재현(인내 100): 처음 나온 최고(같은 값은 앞쪽)에서 멈춤 규칙", b1 == (2, 102) and b2 == (3, 53), [b1, b2])
    # 묶음 단위로 곡선을 이어 붙여도(StagedES) 한꺼번에 만든 곡선과 같고, 멈춤 판단 시점이 맞음
    n, T = 600, 450
    y = (rng.random(n) < 0.08).astype(np.int8)
    y[0] = 1
    offs = np.array([[10 * i, 10 * i + 10] for i in range(60)])
    for a_, b_ in offs:
        y[a_] = 1
        y[a_ + 1] = 0
    C = rng.normal(size=(n, T)) * 0.1
    full = np.nanmean(np.stack([raw3.staged_ap(y[a_:b_], np.cumsum(C[a_:b_], axis=1)) for a_, b_ in offs]), axis=0)
    es = raw3.StagedES(y, offs, 100)
    stops = []
    for k0 in range(0, T, 100):
        new = min(100, T - k0)
        stops.append(es.add(lambda a_, b_: C[a_:b_, k0:k0 + new], new))
        if stops[-1]:
            break
    m_ = len(es.curve)
    stop_ok = (stops[-1] and raw3.es_stopped(full[:m_], 100) and not any(raw3.es_stopped(full[:e], 100) for e in range(100, m_, 100)))         or ((not stops[-1]) and m_ == T and not raw3.es_stopped(full, 100))
    chk("묶음 이어 붙인 조기 종료 곡선(StagedES) = 한꺼번에 만든 곡선(1e-12), 최적 반복 같음, 멈춤 판단 시점 맞음",
        float(np.max(np.abs(np.array(es.curve) - full[:m_]))) <= 1e-12 and raw3.simulate_es(es.curve, 100)[0] == raw3.simulate_es(full, 100)[0] and stop_ok,
        {"rounds": m_, "best": raw3.simulate_es(es.curve, 100), "full": raw3.simulate_es(full, 100), "stopped": bool(stops[-1])})


def small_fold(cvo, bundles, k_smooth, log, n_tr=4000, per_h=40):
    """점검용 작은 겹 자료(겹 1 학습 행 일부 + 맞혀 보는 행 h 마다 일부)."""
    f = G.FOLDS[0]["fold"]
    rtr = cvo.fold_train(f)
    rng = np.random.default_rng(5)
    pos1 = np.flatnonzero(rtr.y == 1)
    neg = np.flatnonzero(rtr.y == 0)
    i_tr = np.sort(np.concatenate([pos1, rng.choice(neg, size=min(n_tr, len(neg)), replace=False)]))
    rtr = rtr.subset(i_tr)
    rev = cvo.eval_rows(f)
    keep = []
    for a, b in rev.h_offsets():
        idx = np.arange(a, b)
        pp = idx[rev.y[a:b] == 1][:5]
        nn = rng.choice(idx[rev.y[a:b] == 0], size=min(per_h, int((rev.y[a:b] == 0).sum())), replace=False)
        keep.append(np.sort(np.concatenate([pp, nn])))
    rev = rev.subset(np.concatenate(keep))
    fb = cvo.builder(bundles, k_smooth)
    levels = cvo.cat_levels(fb.cat_names, fb)
    cs_tr, cd_tr = cvo.cats(rtr, fb, levels)
    cs_ev, cd_ev = cvo.cats(rev, fb, levels)
    return EN.FoldData(f, f, rtr, rev, cvo.compute_X(rtr, bundles, k_smooth)[0], cvo.compute_X(rev, bundles, k_smooth)[0], fb.names, fb.cat_names, levels,
                       cs_tr, cd_tr, cs_ev, cd_ev, bundles)


def part_params(chk, rep, cvo, log, tmp):
    bundles = ["wx", "food"]
    fd = small_fold(cvo, bundles, G.K_SMOOTH_FIXED, log)
    tr, ev = fd.view(bundles)
    import optuna
    res = {}
    for m in M3.MODELS:
        space = G.CFG["optuna"]["spaces"][m]
        mid = {}
        for kname, spec in space.items():
            if spec[0] in ("float", "float_log"):
                mid[kname] = float(np.sqrt(spec[1] * spec[2])) if spec[0] == "float_log" else (spec[1] + spec[2]) / 2
            elif spec[0] in ("int", "int_log"):
                mid[kname] = int((spec[1] + spec[2]) // 2)
            elif spec[0] == "int_or_none":
                mid[f"{kname}_none"] = False
                mid[kname] = int((spec[1] + spec[2]) // 2)
        params = M3.suggest(optuna.trial.FixedTrial(mid), space)
        if m == "rf":
            params["n_estimators"] = 30
        a = M3.make_model(m, cvo.seed, "cpu", log, params)
        a.fit(tr, ev if m in M3.TREE_BOOST else None, fd.offs if m in M3.TREE_BOOST else None)
        pa_ = a.predict(ev)
        f = Path(tmp) / f"params_{m}.json"
        M3.save_params(f, a)
        b = M3.make_model(m, cvo.seed, "cpu", log, M3.load_params(f))
        b.fit(tr, ev if m in M3.TREE_BOOST else None, fd.offs if m in M3.TREE_BOOST else None)
        pb_ = b.predict(ev)
        res[m] = {"bit_equal": bool(bits_equal(pa_, pb_)), "best_rounds": [a.best_rounds, b.best_rounds],
                  "params_saved": len(M3.load_params(f)), "cat_columns": tr.cat_names}
        f.unlink()
    chk("모델 5개: 파라미터 저장(get_params JSON) -> 불러오기(set_params) -> 다시 학습한 예측이 비트 단위로 같음", all(v["bit_equal"] for v in res.values()), res)
    chk("모델 입력 범주 = [sido, sgg, p2, p3] + 식품 묶음 aw_label (작은 겹)", tr.cat_names[:4] == BASE4 and tr.cat_names[4:] == ["aw_label"], tr.cat_names)
    rep["params_roundtrip"] = res


def part_past(chk, rep, cvo, log):
    """과거 검사 결과 칸(과거 이력·WOE·N1, 전국 p0 포함) 규칙(사양 2절):
    학습 행 = 예측일 T 전날까지 굴러가는 창 / 맞혀 보는 행 = 그 겹 학습 끝 값 고정(h·T 와 무관)."""
    from e2hist import History
    import e2nb
    ctx = cvo.ctx
    bl = ["hist_core", "logn1", "sd1", "woe", "n1"]
    S0, H0 = ctx.samples, ctx.H
    nbd = e2nb.get_nb(ctx, ["n1"])
    base_hist = nbd.hist
    sd = S0["d"].to_numpy().astype("datetime64[D]")

    def recompute(rs, m):
        """기준일 조건 m 인 시료: 정답 뒤집기 + 같은 시료 하나씩 더하기 뒤 다시 계산."""
        S1 = S0.copy()
        S1.loc[m, "pos"] = (1 - S1.loc[m, "pos"].to_numpy()).astype(S1["pos"].dtype)
        S1 = pd.concat([S1, S1[m]], ignore_index=True)
        try:
            ctx.samples, ctx.H = S1, History(S1)
            nbd.hist = e2nb.NbHist(np.concatenate([nbd.s_unit, nbd.s_unit[m]]), np.concatenate([nbd.s_day, nbd.s_day[m]]), ctx.H.Dh,
                                   np.vstack([nbd.flags, nbd.flags[m]]), S1["pos"].to_numpy())
            X1, _ = cvo.compute_X(rs, bl, G.K_SMOOTH_FIXED)
        finally:
            ctx.samples, ctx.H = S0, H0
            nbd.hist = base_hist
        return X1

    # (1) 학습 행: 같은 T0 를 가진 학습 행 묶음
    f3 = G.FOLDS[-1]
    tr = cvo.fold_train(f3["fold"])
    T0 = np.datetime64("2023-03-15", "D")
    r0 = tr.subset(np.flatnonzero(tr.T == T0))
    X0, fb = cvo.compute_X(r0, bl, G.K_SMOOTH_FIXED)
    Xa = recompute(r0, sd == T0)                                       # T 당일
    Xb = recompute(r0, sd > T0)                                        # T 당일·이후 전부
    Xc = recompute(r0, sd == T0 - np.timedelta64(1, "D"))              # 대조: T 전날
    nb2, _ = CV3.column_bundles(fb.names, fb.cat_names, bl)
    diff_c = sorted({nb2[j] for j in range(X0.shape[1]) if not np.array_equal(X0[:, j].view(np.uint32), Xc[:, j].view(np.uint32))})
    chk(f"과거 칸 - 학습 행(T={T0}, {r0.n}행): T 당일 시료({int((sd == T0).sum()):,}개)를 바꿔도(정답 뒤집기·시료 더하기) 비트 단위로 그대로", bits_equal(X0, Xa))
    chk(f"과거 칸 - 학습 행: T 당일·이후 시료 전부({int((sd > T0).sum() + (sd == T0).sum()):,}개 중 T 이후)를 바꿔도 그대로", bits_equal(X0, Xb))
    chk(f"과거 칸 - 학습 행 대조군: T 전날 시료를 바꾸면 달라짐 {diff_c}", (not bits_equal(X0, Xc)) and set(diff_c) == set(bl), diff_c)
    # (2) 맞혀 보는 행: 겹 1 (학습 끝 TE)
    f1 = G.FOLDS[0]
    TE = np.datetime64(f1["train_end"], "D")
    ev = cvo.eval_rows(f1["fold"])
    rng = np.random.default_rng(21)
    idx = np.sort(rng.choice(ev.n, size=min(4000, ev.n), replace=False))
    re_ = ev.subset(idx)
    Y0, _ = cvo.compute_X(re_, bl, G.K_SMOOTH_FIXED)
    Ya = recompute(re_, sd > TE)                                       # 학습 끝 뒤 시료 전부
    Yb = recompute(re_, sd == TE)                                      # 대조: 학습 끝 당일
    Yc = recompute(re_, (sd > TE - np.timedelta64(30, "D")) & (sd <= TE))   # 대조: 학습 끝 직전 30일
    diff_b = sorted({nb2[j] for j in range(Y0.shape[1]) if not np.array_equal(Y0[:, j].view(np.uint32), Yb[:, j].view(np.uint32))})
    chk(f"과거 칸 - 맞혀 보는 행(겹 1, 학습 끝 {TE}, 표본 {re_.n}행): 학습 끝 뒤 시료 {int((sd > TE).sum()):,}개를 바꿔도 비트 단위로 그대로", bits_equal(Y0, Ya))
    chk(f"과거 칸 - 맞혀 보는 행 대조군: 학습 끝 당일 시료를 바꾸면 달라짐 {diff_b}", (not bits_equal(Y0, Yb)) and set(diff_b) == set(bl), diff_b)
    chk("과거 칸 - 맞혀 보는 행 대조군: 학습 끝 직전 30일 시료를 바꾸면 달라짐", not bits_equal(Y0, Yc))
    # (3) 같은 단위면 h·T 가 달라도 같은 값(학습 끝 고정)
    unit = ctx.sp[CV3.SP]["grp"]["unit"][re_.pos]
    nt = 0
    same = True
    for u in np.unique(unit):
        ii = np.flatnonzero(unit == u)
        if len(ii) > 1:
            nt += 1
            same &= bool(np.all(Y0[ii, 1:].view(np.uint32) == Y0[ii[0], 1:].view(np.uint32)))
    chk(f"과거 칸 - 맞혀 보는 행: 같은 단위의 행은 h·T·d 가 달라도 값이 같음(행 2개 넘는 단위 {nt}개)", same and nt > 10, nt)
    # (4) 직접 대조: 학습 끝 값 = 2차 계산에 '예측일 = 학습 끝 + 1일' 을 넘긴 값
    fbh = cvo.builder(["hist_core"], G.K_SMOOTH_FIXED)
    he = (re_.d - (TE + np.timedelta64(1, "D"))).astype(np.int64)
    ref = fbh.num(CV3.SP, re_.pos, he)[:, 1:]
    cols = [fb.names.index(n) for n in fbh.names[1:]]
    chk("과거 칸 - 맞혀 보는 행 과거 이력 = 2차 e2hist 에 예측일 = 학습 끝 + 1일(창 [TE−P+1, TE])을 넘긴 값", bits_equal(Y0[:, cols], ref))
    rep["past_check"] = {"train_T0": str(T0), "train_rows": int(r0.n), "eval_TE": str(TE), "eval_rows": int(re_.n), "units_multi": nt,
                         "control_changed_train": diff_c, "control_changed_eval": diff_b, "rule": CV3.PAST_RULE_DESC}


def part_cat_scan(chk, rep, roots, fds_cats=None):
    bad, n = [], 0
    for r in roots:
        for f in Path(r).glob("results/**/config.json"):
            js = CV3.jload(f)
            cc = js.get("cat_columns") or js.get("config", {}).get("cat_columns")
            if cc is None:
                continue
            n += 1
            if cc[:4] != BASE4 or any(c != "aw_label" for c in cc[4:]):
                bad.append((str(f), cc))
    chk(f"결과 폴더 config.json {n}개의 범주 입력 = [sido, sgg, p2, p3](+ 식품 묶음 aw_label)", not bad and n > 0, bad[:5] if bad else n)
    rep["cat_scan"] = {"n": n, "bad": bad}


def part_catboost(chk, rep, cvo, log):
    p_e00 = rep.pop("_e00_p", None)
    if p_e00 is None:
        chk("CatBoost 대조: 2차 E00 예측을 못 읽어 건너뜀", False)
        return
    fd = EN.build(cvo, 3, ["wx"], G.K_SMOOTH_FIXED, log)
    tr, ev = fd.view(["wx"])
    out = {}
    ma = M3.make_model("cat", cvo.seed, "cpu", log, {"iterations": E00_BEST})
    ma.fit(tr, rounds=E00_BEST)
    out["a_iter31_no_eval_default_lr"] = {**cmp_pred(ma.predict(ev), p_e00), "learning_rate": ma.notes["learning_rate_used"]}
    from catboost import CatBoostClassifier
    mq = CatBoostClassifier(iterations=2000, random_seed=cvo.seed, od_type="Iter", od_wait=1, use_best_model=True, task_type="CPU",
                            verbose=0, allow_writing_files=False)
    mq.fit(ma.pool(tr), eval_set=ma.pool(ev.rows(0, 20000)))
    lr_auto = float(mq.learning_rate_)
    mb = M3.make_model("cat", cvo.seed, "cpu", log, {"iterations": E00_BEST, "learning_rate": lr_auto})
    mb.fit(tr, rounds=E00_BEST)
    out["b_iter31_no_eval_lr_E00_condition"] = {**cmp_pred(mb.predict(ev), p_e00), "learning_rate": lr_auto}
    mc = M3.make_model("cat", cvo.seed, "cpu", log)
    mc.fit(tr, ev, fd.offs)
    pc = mc.predict(ev, ntree_end=E00_BEST) if mc.best_rounds >= E00_BEST else None
    out["c_vanilla_fold3_first31"] = {**(cmp_pred(pc, p_e00) if pc is not None else {"note": "최적 반복 < 31"}),
                                      "vanilla_best_rounds": mc.best_rounds, "vanilla_n_trained": mc.n_trained,
                                      "learning_rate": mc.notes["learning_rate_used"]}
    rep["catboost_compare"] = out
    log(f"  CatBoost 대조: {out}")
    log(f"  [참고] CatBoost (b) 31그루 설정·학습률 = E00 조건 값 -> 비트 단위로 같음 여부 {out['b_iter31_no_eval_lr_E00_condition']['bit_equal']} "
        "(다르면 원인 = iterations 설정값이 나무에 영향 - catboost2 의 (d)·(e) 로 확인. 3차 최종 재학습은 그래서 최대 반복 학습 후 앞 R그루만 남김)")
    if pc is not None:
        chk("CatBoost (c): 바닐라 겹 3(조기 종료) 모델의 앞 31그루 -> 2차 E00 예측과 비트 단위로 같음", out["c_vanilla_fold3_first31"]["bit_equal"])


def part_catboost2(chk, rep, cvo, log):
    """(b) 가 다를 때 원인 가리기: (d) 반복 2000·학습률 고정·eval set 없음 -> 앞 31그루, (e) 2차 E00 과 같은 조건 완전 재현
    (eval set = 2차 조기 종료용 검증 행 = 2024-07-01~2025-06-30 행, h = default_rng(43), 지표 = 1차 _APMetric, od Iter 100, use_best_model)."""
    p_e00 = rep.pop("_e00_p", None)
    if p_e00 is None:
        chk("CatBoost 추가 대조: 2차 E00 예측을 못 읽어 건너뜀", False)
        return
    from catboost import CatBoostClassifier
    import models as MD1
    fd = EN.build(cvo, 3, ["wx"], G.K_SMOOTH_FIXED, log)
    tr, ev = fd.view(["wx"])
    lr_rows = cvo.learn_rows()
    Xl, fb = cvo.assemble_X(lr_rows, ["wx"], G.K_SMOOTH_FIXED, f"learn_s{cvo.seed}")
    m = lr_rows.d > np.datetime64("2024-06-30")
    es = lr_rows.subset(np.flatnonzero(m))
    lv = cvo.cat_levels(fb.cat_names, fb)
    cs, cd = cvo.cats(es, fb, lv)
    esI = M3.Inputs(np.ascontiguousarray(Xl[m]), cd, cs, es.y, fb.names, fb.cat_names, lv)
    h2 = np.random.default_rng(cvo.seed + 1).integers(1, 61, size=es.n)
    out = {"es_rows": int(es.n), "es_pos": int(es.y.sum()), "es_h_eq_2차_default_rng43": bool(np.array_equal(es.h, h2))}
    helper = M3.make_model("cat", cvo.seed, "cpu", log)
    base = dict(random_seed=cvo.seed, task_type="CPU", verbose=0, allow_writing_files=False)
    me = CatBoostClassifier(iterations=2000, od_type="Iter", od_wait=100, use_best_model=True, eval_metric=MD1._APMetric(), **base)
    me.fit(helper.pool(tr), eval_set=helper.pool(esI))
    pe = me.predict(helper.pool(ev, with_label=False), prediction_type="Probability")[:, 1].astype(np.float32)
    lr_e = float(me.learning_rate_)
    out["e_full_reproduction"] = {**cmp_pred(pe, p_e00), "best_rounds": int(me.get_best_iteration()) + 1, "learning_rate": lr_e,
                                  "tree_count": int(me.tree_count_)}
    log(f"  CatBoost (e) 2차 조건 완전 재현: {out['e_full_reproduction']}")
    md = CatBoostClassifier(iterations=2000, learning_rate=lr_e, **base)
    md.fit(helper.pool(tr))
    pdd = md.predict(helper.pool(ev, with_label=False), prediction_type="Probability", ntree_end=E00_BEST)[:, 1].astype(np.float32)
    out["d_iter2000_no_eval_lr_fixed_first31"] = {**cmp_pred(pdd, p_e00), "learning_rate": float(md.learning_rate_)}
    log(f"  CatBoost (d) 반복 2000·eval set 없음·학습률 {lr_e} -> 앞 31그루: {out['d_iter2000_no_eval_lr_fixed_first31']}")
    rep["catboost_compare2"] = out
    chk("CatBoost (e): 2차 E00 과 같은 조건(eval set·지표·조기 종료)으로 다시 학습 -> 2차 E00 예측과 비트 단위로 같음(3차 학습 행·입력이 2차와 같음)",
        out["e_full_reproduction"]["bit_equal"], out["e_full_reproduction"])
    chk("CatBoost (d): 반복 2000·eval set 없음(학습률 고정)의 앞 31그루 = 2차 E00 (같으면 eval set 은 나무에 영향 없음, 반복 수 설정만 영향)",
        out["d_iter2000_no_eval_lr_fixed_first31"]["bit_equal"], out["d_iter2000_no_eval_lr_fixed_first31"])


def part_rawes(chk, rep, cvo, log):
    """원값 입력: '최대 반복 학습 + 단계별 점수로 조기 종료 재현'(raw3) = 실제 조기 종료(models3, 맞혀 보는 행 eval set) - 최적 반복·예측 대조(작은 겹)."""
    import raw3
    f = G.FOLDS[0]["fold"]
    rng = np.random.default_rng(5)
    rtr = cvo.fold_train(f)
    i_tr = np.sort(np.concatenate([np.flatnonzero(rtr.y == 1), rng.choice(np.flatnonzero(rtr.y == 0), size=3000, replace=False)]))
    rtr = rtr.subset(i_tr)
    rev = cvo.eval_rows(f)
    keep = []
    for a, b in rev.h_offsets():
        idx = np.arange(a, b)
        keep.append(np.sort(np.concatenate([idx[rev.y[a:b] == 1][:5], rng.choice(idx[rev.y[a:b] == 0], size=40, replace=False)])))
    rev = rev.subset(np.concatenate(keep))
    fb = cvo.builder(["wx"], G.K_SMOOTH_FIXED)
    levels = cvo.cat_levels(list(CV3.BASE_CATS), fb)
    cats_tr, cats_ev = cvo.cats(rtr, fb, levels), cvo.cats(rev, fb, levels)
    X_tr = raw3.raw_X(cvo.ctx, rtr, G.L)
    out = raw3.run_fold(cvo.ctx, rtr, rev, X_tr, cats_tr, cats_ev, levels, ["xgb", "lgbm", "cat"], cvo.seed, log)
    names = raw3.raw_names(G.L)
    trI = M3.Inputs(X_tr, cats_tr[1], cats_tr[0], rtr.y, names, list(CV3.BASE_CATS), levels)
    evI = M3.Inputs(raw3.raw_X(cvo.ctx, rev, G.L), cats_ev[1], cats_ev[0], rev.y, names, list(CV3.BASE_CATS), levels)
    res = {}
    for m in ("xgb", "lgbm", "cat"):
        live = M3.make_model(m, cvo.seed, "cpu", log)
        live.fit(trI, evI, rev.h_offsets())
        pl = live.predict(evI)
        ps, inf = out[m]
        res[m] = {"best_live": live.best_rounds, "best_staged": inf["best_rounds"], "pred_max_abs_diff": float(np.max(np.abs(pl.astype(np.float64) - ps))),
                  "staged_check_maxdiff": inf.get("staged_check_maxdiff")}
    ok = all(v["best_live"] == v["best_staged"] and v["pred_max_abs_diff"] < 1e-6 for v in res.values())
    chk("원값: 묶음 단위 조기 종료 재현(raw3) = 실제 조기 종료(최적 반복 같음, 예측 차이 < 1e-6) - XGBoost·LightGBM·CatBoost (작은 겹)", ok, res)
    # 기상 변수 일부 + 덧붙인 칸(과거 이력 핵심, 맞혀 보는 행 = 학습 끝 값 고정) + 변수 섞기 중요도
    wl = ["avg_temp", "rain", "soil_moist"]
    cols = raw3.raw_cols(wl)
    fbh = cvo.builder(["hist_core"], G.K_SMOOTH_FIXED)
    Xh_tr = cvo.compute_X(rtr, ["hist_core"], G.K_SMOOTH_FIXED)[0][:, 1:]
    Xh_ev = cvo.compute_X(rev, ["hist_core"], G.K_SMOOTH_FIXED)[0][:, 1:]
    X2_tr = np.ascontiguousarray(np.hstack([X_tr[:, cols], Xh_tr]), dtype=np.float32)
    ex = {"names": list(fbh.names[1:]), "ev": lambda a, b: Xh_ev[a:b]}
    out2 = raw3.run_fold(cvo.ctx, rtr, rev, X2_tr, cats_tr, cats_ev, levels, ["xgb", "lgbm", "cat"], cvo.seed, log, extra=ex, wvars=wl,
                         perm_wvars=True, perm_fold=f)
    names2 = raw3.raw_names(G.L, wl) + ex["names"]
    X2_ev = np.ascontiguousarray(np.hstack([raw3.raw_X(cvo.ctx, rev, G.L)[:, cols], Xh_ev]), dtype=np.float32)
    tr2 = M3.Inputs(X2_tr, cats_tr[1], cats_tr[0], rtr.y, names2, list(CV3.BASE_CATS), levels)
    ev2 = M3.Inputs(X2_ev, cats_ev[1], cats_ev[0], rev.y, names2, list(CV3.BASE_CATS), levels)
    offs = rev.h_offsets()
    pseed, pscope = EN.perm_cfg()
    res2 = {}
    for m in ("xgb", "lgbm", "cat"):
        live = M3.make_model(m, cvo.seed, "cpu", log)
        live.fit(tr2, ev2, offs)
        pl = live.predict(ev2)
        ps, inf = out2[m]
        pdiff = {}
        for v in wl:
            src = EN.perm_source(rev.n, offs, np.random.default_rng([int(pseed), int(f), EN.wvar_seed_index(v)]), pscope)
            gi = [i for i, n_ in enumerate(names2) if EN.wvar_of(n_) == v]
            Xs = X2_ev.copy()
            Xs[:, gi] = X2_ev[src][:, gi]
            man = CV3.wpr(rev.y, live.predict(M3.Inputs(Xs, cats_ev[1], cats_ev[0], rev.y, names2, list(CV3.BASE_CATS), levels)), offs)
            pdiff[v] = abs(man - inf["perm_wvar_score"][v])
        res2[m] = {"best_live": live.best_rounds, "best_staged": inf["best_rounds"], "pred_max_abs_diff": float(np.max(np.abs(pl.astype(np.float64) - ps))),
                   "perm_score_max_diff": float(max(pdiff.values())), "n_num": len(names2)}
    ok2 = all(v["best_live"] == v["best_staged"] and v["pred_max_abs_diff"] < 1e-6 and v["perm_score_max_diff"] < 1e-6 for v in res2.values())
    chk(f"원값(기상 변수 {len(wl)}개 + 과거 이력 핵심 칸): 조기 종료 재현 = 실제 조기 종료, 변수 섞기 점수 = 직접 섞어 다시 예측한 값(차이 < 1e-6)", ok2, res2)
    rep["raw_es_reproduction"] = {"all_vars": res, "subset_plus_hist": res2}


def part_guard(chk, rep, cvo):
    try:
        cvo.test_rowset("selftest: 고르는 단계 흉내")
        raised = False
    except RuntimeError:
        raised = True
    chk("(다) 고르는 단계(ALLOW_TEST 없음)에서 시험 행을 만들면 오류로 멈춤", raised and not CV3.ALLOW_TEST)


def part_perm(chk, rep):
    """섞기 중요도 손계산: (가) 묶음 단위 - 모델 = spi 묶음 칸 두 개 합, food 묶음은 안 씀. (나) 기상 변수 단위 - 모델 = avg_temp 칸만 씀."""
    rng = np.random.default_rng(4)
    per = 10
    n = 60 * per
    offs = np.array([[per * i, per * i + per] for i in range(60)])
    y = (rng.random(n) < 0.15).astype(np.int8)
    for a_, b_ in offs:
        y[a_], y[a_ + 1] = 1, 0
    hcol = np.repeat(np.arange(1, 61), per)
    X = np.column_stack([hcol, rng.normal(size=n) + 2 * y, rng.normal(size=n) + y, rng.normal(size=n)]).astype(np.float32)
    ev = M3.Inputs(X, np.zeros((n, 0), np.int32), np.zeros((n, 0), object), y, ["h", "a1", "a2", "b1"], [], {})
    groups = {"spi": ([1, 2], []), "food": ([3], [])}
    fn = lambda x: x.X[:, 1] + x.X[:, 2]
    base = CV3.wpr(y, fn(ev), offs)
    imp1 = EN.perm_importance(fn, ev, offs, groups, base, 99, 1, "h_block", chunk=150)
    imp2 = EN.perm_importance(fn, ev, offs, groups, base, 99, 1, "h_block", chunk=1000)
    r = np.random.default_rng([99, 1, CV3.BUNDLE_ORDER.index("spi")])
    src = np.concatenate([a_ + r.permutation(b_ - a_) for a_, b_ in offs])
    Xp = X.copy()
    Xp[:, [1, 2]] = X[src][:, [1, 2]]
    hand = base - CV3.wpr(y, Xp[:, 1] + Xp[:, 2], offs)
    ok = imp1["food"] == 0.0 and abs(imp1["spi"] - hand) < 1e-12 and imp1 == imp2 and hand > 0
    chk("묶음 섞기 중요도 손계산: 안 쓰는 묶음 = 0, 쓰는 묶음 = 같은 씨앗으로 h 마다 직접 섞은 구간 가중 PR-AUC 감소, 씨앗 같으면 같은 값(조각 크기 달라도)",
        ok, {"imp": imp1, "hand_spi": hand})
    names_w = ["h", "avg_temp_b01", "avg_temp_mean", "rain_b01", "rain_mean"]
    gw = EN.wvar_groups(names_w, None)
    chk("기상 변수 단위 = 그 변수 칸 전부(요약 이름 규칙): avg_temp -> 칸 1·2, rain -> 칸 3·4, 나머지 변수 칸 없음", gw == {"avg_temp": ([1, 2], []), "rain": ([3, 4], [])}, gw)
    Xw = np.column_stack([hcol, rng.normal(size=n) + 2 * y, rng.normal(size=n) + y, rng.normal(size=n), rng.normal(size=n)]).astype(np.float32)
    evw = M3.Inputs(Xw, np.zeros((n, 0), np.int32), np.zeros((n, 0), object), y, names_w, [], {})
    fw = lambda x: x.X[:, 1] + x.X[:, 2]
    bw = CV3.wpr(y, fw(evw), offs)
    impw = EN.perm_importance(fw, evw, offs, gw, bw, 99, 2, "h_block", seed_index=EN.wvar_seed_index)
    rr = np.random.default_rng([99, 2, 1000 + EN.WVARS.index("avg_temp")])
    srw = np.concatenate([a_ + rr.permutation(b_ - a_) for a_, b_ in offs])
    Xq = Xw.copy()
    Xq[:, [1, 2]] = Xw[srw][:, [1, 2]]
    handw = bw - CV3.wpr(y, Xq[:, 1] + Xq[:, 2], offs)
    chk("기상 변수 섞기 중요도 손계산: 안 쓰는 변수(rain) = 0, 쓰는 변수(avg_temp) = 씨앗 [99, 겹, 1000 + 변수 번호]로 직접 섞은 감소",
        impw["rain"] == 0.0 and abs(impw["avg_temp"] - handw) < 1e-12 and handw > 0, {"imp": impw, "hand": handw})
    import raw3
    nm_ok = (EN.wvar_of("soil_t5_lag001") == "soil_t5" and EN.wvar_of("wind_dir_sin_slope") == "wind_dir_sin" and EN.wvar_of("max_temp_b10") == "max_temp"
             and EN.wvar_of("woe_sgg") is None and EN.wvar_of("hist_rate5y_p3") is None and EN.wvar_of("spi3_mean") is None)
    rc = raw3.raw_cols(["max_temp"], 10)
    rn = raw3.raw_names(10, ["max_temp"])
    chk("기상 변수 칸 찾기(요약·원값 이름) 와 원값 칸 고르기(L=10, max_temp -> h + 원값 칸 11~20)", nm_ok and rc == [0] + list(range(11, 21))
        and rn == ["h"] + [f"max_temp_lag{j:03d}" for j in range(1, 11)] and raw3.raw_cols(None, 10) == list(range(191)), {"cols": rc, "names": rn[:3]})
    try:
        import dl3
        sc = dl3.seq_channels(["max_temp", "rain"])
        chk("BiLSTM 기상 변수 채널: max_temp·rain -> 값 채널 1·3 + 결측 표시 채널 20·22, 전부면 38채널 그대로", sc == [1, 3, 20, 22] and dl3.seq_channels(None) == list(range(38)), sc)
    except ImportError as e:
        log_ = rep.setdefault("skipped", [])
        log_.append(f"dl3 채널 점검 건너뜀(torch 없음): {e}")
    rep["perm_example"] = {"imp": imp1, "hand_spi": hand, "imp_wvar": impw, "hand_wvar": handw}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--e00-pred", default=None)
    ap.add_argument("--hcount", default=None)
    ap.add_argument("--parts", default="all")
    ap.add_argument("--scan-root", action="append", default=[])
    args = ap.parse_args()
    R = Path(args.root).resolve()
    out_dir = R / "results" / "_점검"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(R / "logs" / "selftest3.log")
    parts = {"bounds", "rows", "cache", "score", "ap", "params", "past", "guard", "perm", "cat", "rawes"} if args.parts == "all" else set(args.parts.split(","))
    chk = Checker(log)
    rep = {"L": G.L, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "parts": sorted(parts), "smoke": dict(G.SMOKE)}
    t0 = time.time()
    log(f"===== selftest3 {sorted(parts)} | L={G.L} | 경계 {G.BOUNDS['data_end']} ({G.BOUNDS['data_end_source']})")
    if "bounds" in parts:
        part_bounds(chk, rep)
    if "score" in parts:
        part_score(chk, rep)
    if "ap" in parts:
        part_ap(chk, rep)
    if "perm" in parts:
        part_perm(chk, rep)
    need_ctx = parts & {"rows", "cache", "params", "past", "catboost", "catboost2", "rawes", "guard"}
    if need_ctx:
        ctx = CV3.load_ctx(log)
        cvo = CV3.CV(ctx, log, R / "cache")
        if "rows" in parts or "catboost" in parts or "catboost2" in parts:
            part_rows(chk, rep, cvo, args)
        if "cache" in parts:
            part_cache(chk, rep, cvo)
        if "params" in parts:
            part_params(chk, rep, cvo, log, out_dir)
        if "past" in parts:
            part_past(chk, rep, cvo, log)
        if "rawes" in parts:
            part_rawes(chk, rep, cvo, log)
        if "guard" in parts:
            part_guard(chk, rep, cvo)
        if "catboost" in parts and G.L == 100 and not cvo.smoke.get("enabled"):
            part_catboost(chk, rep, cvo, log)
        if "catboost2" in parts and G.L == 100 and not cvo.smoke.get("enabled"):
            part_catboost2(chk, rep, cvo, log)
        chk("점검 중 시험 행 접근 기록이 남지 않음(점검용 위치 세기는 지운 뒤 확인)", not CV3.TEST_ACCESS, CV3.TEST_ACCESS)
    if "cat" in parts:
        part_cat_scan(chk, rep, [R] + list(args.scan_root))
    rep.pop("_e00_p", None)
    rep["checks"] = chk.items
    rep["all_ok"] = all(c["ok"] for c in chk.items)
    rep["seconds"] = round(time.time() - t0, 1)
    name = f"selftest3_L{G.L:03d}" + ("_smoke" if G.SMOKE.get("enabled") else "") + ("" if args.parts == "all" else "_" + "_".join(sorted(parts)))
    CV3.jdump(rep, out_dir / f"{name}.json")
    log(f"===== selftest3 끝: 전체 통과 {rep['all_ok']} ({sum(c['ok'] for c in chk.items)}/{len(chk.items)}, {rep['seconds']}s) -> {name}.json")


if __name__ == "__main__":
    main()
