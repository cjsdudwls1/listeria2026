# -*- coding: utf-8 -*-
"""3차 틀 점검.

python selftest3.py --root <결과 루트> [--e00-pred <2차 E00__s42 pred_val.parquet> --hcount <2차 h별_행수_검증.csv>]
                    [--parts all|bounds,rows,cache,score,params,leak,cat,ap,scan] [--scan-root <다른 결과 루트> ...]
(기상 창은 설정/환경변수 EXP2_L, 기본 100)

bounds : 날짜 경계가 설정 규칙대로인지 - 기본(시료 파일 끝 날짜)·가짜 경계(데이터 끝 2025-12-31)·겹 직접 지정 세 가지를 손으로 적은 기댓값과 대조
rows   : 겹별 학습 행·맞혀 보는 행·시험 행이 경계대로인지(기본 + 가짜 경계), 겹 행 수·검출 수(단위x날짜 행 + 시료 기준 여러 정의 -> 사양 230/196/164),
         시험 행이 어떤 겹의 학습·맞혀 보는 행에도 없음 + 겹 자료를 만드는 동안 시험 행 접근 0회,
         (기본 경계) 겹 3 학습 행 = 2차 학습 분할(222,793행/검출 851, h = 2차 Ctx2.h_fit(42) 규칙), 겹 3 맞혀 보는 행 = 2차 E00 pred_val 의 T 기준 행,
         h 별 행 수 = analysis/T기준_통합_L100/h별_행수_검증.csv, 창이 기상표 안
cache  : 맞혀 보는 행 입력 캐시 = FeatureBuilder 직접 계산(비트 단위; 기본 입력 h 4개 블록 전체 + '변수 전부'(이상치 점수 제외) 표본 행)
score  : 검사축소율·상위 n%·F2 기준값 손계산 대조(작은 예)
ap     : 조기 종료용 빠른 AP(cv.ap_fast) = sklearn average_precision (1e-12)
params : 모델 5개 - 파라미터 저장(get_params JSON) 뒤 불러와(set_params) 다시 학습하면 예측이 비트 단위로 같음
leak   : 겹 k 입력을 만들 때 그 겹 학습 끝 날짜 뒤 정답이 쓰이지 않음 - 학습 끝 뒤 시료 정답을 뒤집은 자료로 다시 만든 학습 입력이 비트 단위로 같음
         (대조군: 같은 뒤집기에서 맞혀 보는 행 입력(그 행의 T 전날까지 이력)은 달라짐 = 점검이 정답 변화를 잡을 수 있음)
cat    : 범주 입력 = [sido, sgg, p2, p3] (+ 식품 묶음이면 aw_label) - 겹 자료·결과 폴더 config.json 전부(--scan-root)
catboost: (기본 경계, 창 100) CatBoost 겹 3 을 2차 E00 시드 42 최적 반복 31 로 조기 종료 없이 학습 -> 2차 E00 예측(T 기준 행)과 비트 단위 비교 (a)(b)(c)
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
    """손으로 적은 기댓값."""
    cfg = {"dates": {"data_end": "2026-06-30", "n_folds": 3, "folds": None, "test": None}}
    b = G.compute_bounds(cfg)
    exp = [("2022-06-30", "2022-07-01", "2023-05-01"), ("2023-06-30", "2023-07-01", "2024-05-01"), ("2024-06-30", "2024-07-01", "2025-05-01")]
    ok = [(f["train_end"], f["t_start"], f["t_end"]) for f in b["folds"]] == exp and b["test"]["t_start"] == "2025-07-01" \
        and b["test"]["t_end"] == "2026-05-01" and b["learn_end"] == "2025-06-30"
    chk("경계(데이터 끝 2026-06-30): 사양 표와 같음(겹 1~3·시험 2025-07-01~2026-05-01·학습 자료 끝 2025-06-30)", ok, b)
    cfg2 = {"dates": {"data_end": "2025-12-31", "n_folds": 3, "folds": None, "test": None}}
    b2 = G.compute_bounds(cfg2)
    exp2 = [("2021-12-31", "2022-01-01", "2022-11-01"), ("2022-12-31", "2023-01-01", "2023-11-01"), ("2023-12-31", "2024-01-01", "2024-11-01")]
    ok2 = [(f["train_end"], f["t_start"], f["t_end"]) for f in b2["folds"]] == exp2 and b2["test"]["t_start"] == "2025-01-01" \
        and b2["test"]["t_end"] == "2025-11-01" and b2["learn_end"] == "2024-12-31"
    chk("가짜 경계(데이터 끝 2025-12-31): 시험 2025-01-01~2025-11-01, 겹 2022·2023·2024 년(1~11-01), 학습 자료 끝 2024-12-31", ok2, b2)
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
    chk("겹 맞혀 보는 행이 학습 자료 끝을 넘는 지정은 오류로 멈춤", bad)
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
              and rev.d.min() > np.datetime64(f["train_end"], "D") and rev.d.max() <= np.datetime64(G.BOUNDS["learn_end"], "D")
              and (cvo.smoke.get("enabled") or tdays == [days]))
        chk(f"[{label}] 겹 {k}: 학습 d <= {f['train_end']}, 맞혀 보는 T ∈ [{f['t_start']}, {f['t_end']}], 모든 h 같은 {days}일", ok,
            {"train_d_max": str(rtr.d.max()), "eval_T": [str(rev.T.min()), str(rev.T.max())], "T_days_per_h": tdays})
        out[k] = (rtr, rev)
    te_ = cvo.test_rowset("selftest 점검(위치만)")
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
        used |= set(cvf.learn_rows().pos.tolist()) | set(cvf.test_rowset("selftest 가짜 경계").pos.tolist())
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
    allb = [b for b in CV3.resolve_featset("all") if b != "ae4"]
    fba = cvo.builder(allb, G.K_SMOOTH_FIXED)
    rev = cvo.eval_rows(G.FOLDS[0]["fold"])
    sub = rev.subset(np.sort(np.random.default_rng(1).choice(rev.n, size=min(3000, rev.n), replace=False)))
    Xs, fbs = cvo.assemble_X(sub, allb, G.K_SMOOTH_FIXED, "selftest_allsub")
    ok = bits_equal(Xs, cvo.compute_X(sub, fba)) and fbs.names == fba.names
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
    thr = t1
    rep["score_example"] = {"per_h": r.to_dict(), "thr": thr}


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
    return EN.FoldData(f, f, rtr, rev, cvo.compute_X(rtr, fb), cvo.compute_X(rev, fb), fb.names, fb.cat_names, levels,
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


def part_leak(chk, rep, cvo, log):
    """학습 끝 뒤 시료 정답을 뒤집어도 겹 학습 입력이 같은지(대조군: 맞혀 보는 행 입력은 달라짐)."""
    from e2hist import History
    ctx = cvo.ctx
    bundles = [b for b in CV3.resolve_featset("all") if b != "ae4"]
    res = {}
    S0, H0, nb0 = ctx.samples, ctx.H, getattr(ctx, "_nb", None)
    for f in G.FOLDS[:2]:
        k = f["fold"]
        te = np.datetime64(f["train_end"], "D")
        rtr = cvo.fold_train(k)
        rev = cvo.eval_rows(k)
        rng = np.random.default_rng(11)
        i_tr = np.sort(rng.choice(rtr.n, size=min(3000, rtr.n), replace=False))
        a, b = rev.h_offsets()[0]
        i_ev = np.arange(a, min(b, a + 3000))
        str_, sev = rtr.subset(i_tr), rev.subset(i_ev)
        fb = cvo.builder(bundles, G.K_SMOOTH_FIXED)
        X0_tr, X0_ev = cvo.compute_X(str_, fb), cvo.compute_X(sev, fb)
        S1 = S0.copy()
        late = S1["d"].to_numpy().astype("datetime64[D]") > te
        S1.loc[late, "pos"] = (1 - S1.loc[late, "pos"].to_numpy()).astype(S1["pos"].dtype)   # 학습 끝 뒤 정답 모두 뒤집기
        try:
            ctx.samples, ctx.H, ctx._nb = S1, History(S1), None
            fb1 = cvo.builder(bundles, G.K_SMOOTH_FIXED)
            X1_tr, X1_ev = cvo.compute_X(str_, fb1), cvo.compute_X(sev, fb1)
        finally:
            ctx.samples, ctx.H, ctx._nb = S0, H0, nb0
        nb_, _ = CV3.column_bundles(fb.names, fb.cat_names, bundles)
        diff_cols = sorted({nb_[j] for j in range(X0_ev.shape[1])
                            if not np.array_equal(X0_ev[:, j].view(np.uint32), X1_ev[:, j].view(np.uint32))})
        res[k] = {"train_rows_checked": int(str_.n), "train_bit_equal": bool(bits_equal(X0_tr, X1_tr)), "flipped_samples": int(late.sum()),
                  "control_eval_rows_checked": int(sev.n), "control_eval_changed": not bits_equal(X0_ev, X1_ev),
                  "control_changed_bundles": diff_cols, "train_y_d_max": str(rtr.d.max())}
        chk(f"겹 {k}: 학습 끝({f['train_end']}) 뒤 시료 정답 {int(late.sum()):,}개를 뒤집어도 학습 입력(변수 전부, 이상치 점수 제외) 비트 단위로 같음",
            res[k]["train_bit_equal"], {kk: v for kk, v in res[k].items() if kk != "control_changed_bundles"})
        chk(f"겹 {k} 대조군: 같은 뒤집기에서 맞혀 보는 행(h=1, 그 행 T 전날까지 이력) 입력은 달라짐 -> 점검이 정답 변화를 잡음",
            res[k]["control_eval_changed"], diff_cols)
    rep["leak"] = res
    rep["leak_note"] = ("과거 이력·WOE·제품명 과거 유형은 e2hist·e2nb 가 '기준일 < T' 시료로 그때그때 계산(고정 표 아님). 학습 행은 T < d <= 학습 끝 이므로 "
                        "학습 끝 뒤 정답이 안 들어감. 맞혀 보는 행 입력은 그 행의 T 전날까지 정답을 씀(운영과 같은 전제). 이상치 점수는 겹 학습 행(정답 = 겹 학습 행 y)으로만 학습.")


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
    chk("원값: 최대 반복 학습 + 단계별 점수 조기 종료 재현 = 실제 조기 종료(최적 반복 같음, 예측 차이 < 1e-6) - XGBoost·LightGBM·CatBoost (작은 겹)", ok, res)
    rep["raw_es_reproduction"] = res


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
    parts = {"bounds", "rows", "cache", "score", "ap", "params", "leak", "cat", "catboost", "rawes"} if args.parts == "all" else set(args.parts.split(","))
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
    need_ctx = parts & {"rows", "cache", "params", "leak", "catboost", "catboost2", "rawes"}
    if need_ctx:
        ctx = CV3.load_ctx(log)
        cvo = CV3.CV(ctx, log, R / "cache")
        if "rows" in parts or "catboost" in parts or "catboost2" in parts:
            part_rows(chk, rep, cvo, args)
        if "cache" in parts:
            part_cache(chk, rep, cvo)
        if "params" in parts:
            part_params(chk, rep, cvo, log, out_dir)
        if "leak" in parts:
            part_leak(chk, rep, cvo, log)
        if "rawes" in parts:
            part_rawes(chk, rep, cvo, log)
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
