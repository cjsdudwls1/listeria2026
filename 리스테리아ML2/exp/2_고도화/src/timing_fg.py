# -*- coding: utf-8 -*-
"""F·G 시간·메모리 측정(전체 자료, 시드 42, 본 설정 조기 종료 그대로 - 본 결과 아님).
결과 = logs/timing_fg.json, 실행 결과 폴더 = scratch/fg/timing_root/results (results/ 는 건드리지 않음).
기상 요약 캐시(cache/)는 메타가 같을 때만 읽기(다르면 쓰지 않고 그 자리에서 계산).

python src/timing_fg.py [실험ID ...] [--out=파일이름.json]   (기본: F4 G2 A3_ALL, logs/timing_fg.json)  A3_ALL = A3 + F·G 블록 전부(가장 큰 조합, 측정용)
--out 을 주면 logs/<파일이름> 에 씀(기존 timing_fg.json 을 덮지 않기 위해).
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
import e2fg as G  # noqa: E402
from run import check_windows, run_one  # noqa: E402
from selftest_fg import _cache_meta_ok  # noqa: E402


def main():
    outs = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")]
    out_name = outs[0] if outs else "timing_fg.json"
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or ["F4", "G2", "A3_ALL"]
    root = E.EXP2_DIR / "scratch" / "fg" / "timing_root"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    for k in ("results", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    log = Logger(E.EXP2_DIR / "logs" / "progress_timing_fg.log")
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
    for eid in ids:
        if eid == "A3_ALL":
            feats = dict(E.EXP_BY_ID["A3"]["feats"], k=k_star, ext=list(G.BLOCK_ORDER))
            exp = dict(id="A3_ALL", stage="timing", plan="측정용", model="cat", feats=feats, desc="측정용: A3 + F·G 블록 전부")
        else:
            exp = E.EXP_BY_ID[eid]
        d = P["results"] / E.run_dir_name(exp["id"], 42)
        if d.exists():
            shutil.rmtree(d)
        t = time.time()
        if exp["feats"].get("k") == "k*":
            exp = dict(exp, feats=dict(exp["feats"], k=k_star))
        s = run_one(exp, 42, ctx, P, log, mem, False, win)
        res["runs"][eid] = {k: s.get(k) for k in ("total_seconds", "fit_seconds", "feature_seconds", "eval_seconds", "n_features",
                                                  "best_rounds_or_epoch", "n_rounds_trained", "peak_private_gb", "peak_rss_gb",
                                                  "system_min_available_gb", "val_pr_auc_w")}
        res["runs"][eid]["wall_seconds"] = time.time() - t
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    (E.EXP2_DIR / "logs" / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("timing_fg 끝: " + json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
