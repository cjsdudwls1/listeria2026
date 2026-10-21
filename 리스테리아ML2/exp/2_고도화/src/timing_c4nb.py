# -*- coding: utf-8 -*-
"""C4_base·C4_hist(C4 다운사이징)·A3_N1·A3_N2·A3_N1N2·A3_N0(제품명) 시간·메모리 측정(전체 자료, 시드 42, 본 설정 조기 종료 그대로 - 본 결과 아님).
결과 = scratch/c4nb/timing_c4nb.json, 실행 결과 폴더 = scratch/c4nb/timing_root/results (results/·tables/·logs/ 는 건드리지 않음).
기상 요약 캐시(cache/)는 메타가 같을 때만 읽기(다르면 쓰지 않고 그 자리에서 계산). k*·A 최고는 results/selection_k.json·selection_A.json 을 읽기만 함.

python src/timing_c4nb.py [실험ID ...]   (기본: C4_base C4_hist A3_N1 A3_N2 A3_N1N2 A3_N0)
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import psutil  # noqa: E402

from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem  # noqa: E402
from run import check_windows, run_one  # noqa: E402
from selftest_fg import _cache_meta_ok  # noqa: E402


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or ["C4_base", "C4_hist", "A3_N1", "A3_N2", "A3_N1N2", "A3_N0"]
    scr = E.EXP2_DIR / "scratch" / "c4nb"
    root = scr / "timing_root"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    for k in ("results", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    log = Logger(scr / "progress_timing_c4nb.log")
    mem = Mem(gpu_every=0)
    mem.start()
    t0 = time.time()
    res = {"system_available_gb_at_start": round(psutil.virtual_memory().available / 1e9, 2)}
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        log("기상 요약 캐시 메타가 다름 -> 캐시 안 씀")
    res.update({"ctx_seconds": time.time() - t0, "runs": {}})
    win = check_windows(ctx, log)
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    a_best = json.loads((E.EXP2_DIR / "results" / "selection_A.json").read_text(encoding="utf-8"))["best"]
    res["k_star"], res["A_best"] = k_star, a_best
    for eid in ids:
        exp = E.EXP_BY_ID[eid]
        f = exp["feats"]
        if f == "A*":
            f = E.EXP_BY_ID[a_best]["feats"]
        f = dict(f)
        if f.get("k") == "k*":
            f["k"] = k_star
        exp = dict(exp, feats=f)
        d = P["results"] / E.run_dir_name(exp["id"], 42)
        if d.exists():
            shutil.rmtree(d)
        t = time.time()
        s = run_one(exp, 42, ctx, P, log, mem, False, win)
        res["runs"][eid] = {k: s.get(k) for k in ("total_seconds", "fit_seconds", "feature_seconds", "eval_seconds", "n_features",
                                                  "n_train_rows", "n_train_pos", "best_rounds_or_epoch", "n_rounds_trained",
                                                  "peak_private_gb", "peak_rss_gb", "system_min_available_gb", "val_pr_auc_w", "test_pr_auc_w")}
        res["runs"][eid]["wall_seconds"] = time.time() - t
        (scr / "timing_c4nb.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    (scr / "timing_c4nb.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("timing_c4nb 끝: " + json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
