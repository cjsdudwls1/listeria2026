# -*- coding: utf-8 -*-
"""M1·A3_M1(이동평균 차감 기상) 시간·메모리 측정(전체 자료, 시드 42, 본 설정 조기 종료 그대로 - 본 결과 아님).
결과 = scratch/wxanom/timing_wxanom.json, 실행 결과 폴더 = scratch/wxanom/timing_root/results (results/·tables/·logs/ 는 건드리지 않음).
기상 요약 캐시(cache/)는 메타가 같을 때만 읽기(다르면 쓰지 않고 그 자리에서 계산). k* 는 results/selection_k.json 을 읽기만 함.

python src/timing_wxanom.py [실험ID ...]   (기본: M1 A3_M1)
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem  # noqa: E402
from run import check_windows, run_one  # noqa: E402
from selftest_fg import _cache_meta_ok  # noqa: E402


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or ["M1", "A3_M1"]
    scr = E.EXP2_DIR / "scratch" / "wxanom"
    root = scr / "timing_root"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    for k in ("results", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    log = Logger(scr / "progress_timing_wxanom.log")
    mem = Mem(gpu_every=0)
    mem.start()
    t0 = time.time()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        log("기상 요약 캐시 메타가 다름 -> 캐시 안 씀")
    res = {"ctx_seconds": time.time() - t0, "runs": {}}
    win = check_windows(ctx, log)
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    res["k_star"] = k_star
    for eid in ids:
        exp = E.EXP_BY_ID[eid]
        if exp["feats"].get("k") == "k*":
            exp = dict(exp, feats=dict(exp["feats"], k=k_star))
        d = P["results"] / E.run_dir_name(exp["id"], 42)
        if d.exists():
            shutil.rmtree(d)
        t = time.time()
        s = run_one(exp, 42, ctx, P, log, mem, False, win)
        res["runs"][eid] = {k: s.get(k) for k in ("total_seconds", "fit_seconds", "feature_seconds", "eval_seconds", "n_features",
                                                  "best_rounds_or_epoch", "n_rounds_trained", "peak_private_gb", "peak_rss_gb",
                                                  "system_min_available_gb", "val_pr_auc_w", "test_pr_auc_w")}
        res["runs"][eid]["wall_seconds"] = time.time() - t
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    (scr / "timing_wxanom.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("timing_wxanom 끝: " + json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
