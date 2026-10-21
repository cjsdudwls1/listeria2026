# -*- coding: utf-8 -*-
"""시간·메모리 측정(전체 자료, 라운드 제한 - 본 결과 아님). 결과 = logs/timing_<what>.json. results/ 는 건드리지 않음.

python src/timing.py --what cat_e00|cat_a3|cat_d2|cat_d3|cat_h10|cat_h30|lr_b2 [--rounds N]
  cat_*: 조기 종료를 끄고(인내 무한) N 라운드만 학습 -> 라운드당 시간, 학습 특성 시간, 평가(검증·시험 x h 60) 시간, 최대 메모리
  lr_b2: B2 로지스틱 1회 학습 전체 + 평가
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402

from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem  # noqa: E402
from e2models import CatRunner, LRRunner  # noqa: E402

WHAT = {"cat_e00": "E00", "cat_a3": "A3", "cat_d2": "D2", "cat_d3": "D3", "cat_h10": "H10", "cat_h30": "H30", "lr_b2": "B2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=list(WHAT))
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--k", type=int, default=50, help="이력 변수 k (측정용, 결과와 무관)")
    a = ap.parse_args()
    log = Logger(E.EXP2_DIR / "logs" / "progress_timing.log")
    log(f"===== timing {a.what} rounds={a.rounds} pid {os.getpid()}")
    mem = Mem(gpu_every=0)
    mem.start()
    t0 = time.time()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    res = {"what": a.what, "rounds_limit": a.rounds, "ctx_seconds": time.time() - t0, "mem_after_ctx": mem.snapshot()}
    exp = dict(E.EXP_BY_ID[WHAT[a.what]])
    feats = exp["feats"]
    feats = dict(feats)
    if feats.get("k") == "k*":
        feats["k"] = a.k
    # 캐시 생성 시간(처음 한 번)
    t = time.time()
    ctx._eval_cache("val")
    ctx._eval_cache("test")
    res["eval_cache_seconds(이미 있으면 읽기만)"] = time.time() - t
    if a.what.startswith("cat"):
        r = CatRunner(ctx, exp, 42, feats, log, a.rounds, 10 ** 6, cache_dir=E.EXP2_DIR / "cache", mem=mem)
    else:
        r = LRRunner(ctx, exp, 42, feats, log)
    t = time.time()
    r.fit()
    res["fit_total_seconds(특성+학습)"] = time.time() - t
    res["fit_seconds(학습만)"] = r.fit_seconds
    res["n_train_rows"] = r.n_train_rows
    res["rounds_trained"] = r.n_rounds_trained
    res["seconds_per_round"] = r.fit_seconds / max(1, r.n_rounds_trained)
    res["feature_seconds_train"] = r.feature_seconds
    res["mem_after_fit"] = mem.snapshot()
    t = time.time()
    fs0 = r.feature_seconds
    for sp in ("val", "test"):
        for h in range(1, E.H_MAX + 1):
            r.predict(sp, h)
    res["eval_seconds(검증·시험 x h60)"] = time.time() - t
    res["eval_feature_seconds"] = r.feature_seconds - fs0
    inf = r.info()
    res["extra"] = inf.get("extra") if a.what.startswith("cat") else None
    res["mem_train_phase"] = inf.get("mem_train_phase")
    res["lr_info"] = {k: inf[k] for k in ("n_iter", "converged", "nan_rate_train", "n_allnan_cols", "n_zero_sd_cols") if k in inf}
    r.close()
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    out = E.EXP2_DIR / "logs" / f"timing_{a.what}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"timing {a.what} 끝: {json.dumps({k: v for k, v in res.items() if k != 'extra'}, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()
