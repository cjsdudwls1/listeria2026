# -*- coding: utf-8 -*-
"""시간·메모리 측정(전체 데이터, 라운드 수 제한). 결과 = logs/timing_<what>.json. 본 결과(results/)는 건드리지 않음.

python src/timing.py --what xgb_summary|xgb_raw|lgb_raw|cat_summary|cat_raw|lstm [--rounds N]
- 트리: 조기 종료를 끄고(인내 무한) N 라운드만 학습 -> 라운드당 시간, 특성 생성 시간, (분할, h) 1회 평가 시간, 최대 메모리
- lstm: 1 에폭 학습 + 조기종료 val 평가 + (분할, h) 평가 2회
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config as C
from common import Ctx, Logger, MemSampler, make_runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--L", type=int, default=100)
    a = ap.parse_args()
    log = Logger(C.EXP_DIR / "logs" / "progress_timing.log")
    log(f"===== timing {a.what} rounds={a.rounds} L={a.L} pid {os.getpid()}")
    C.MAX_ROUNDS = a.rounds
    C.PATIENCE = 10 ** 6
    mem = MemSampler()
    mem.start()
    t0 = time.time()
    ctx = Ctx(False, log)
    res = {"what": a.what, "L": a.L, "data_seconds": time.time() - t0}
    if a.what == "lstm":
        r = make_runner(ctx, "lstm", "seq", a.L, log)
        t = time.time()
        r.r.fit(max_epochs=a.rounds)
        res["epoch_seconds_incl_val_eval"] = time.time() - t
        res["history"] = r.r.history
        ev = []
        for sp in ("val", "test"):
            t = time.time()
            r.predict(sp, 30)
            ev.append(time.time() - t)
        res["eval_pass_seconds"] = ev
        res["gpu_peak_allocated_gb"] = r.r.gpu_peak_gb
        r.close()
    else:
        lib, inp = a.what.split("_")
        model = {"xgb": "xgboost", "lgb": "lightgbm", "cat": "catboost"}[lib]
        r = make_runner(ctx, model, inp, a.L, log)
        t = time.time()
        Xtr = r._X("train")
        Xva = r._X("val")
        res["feature_train_val_seconds"] = time.time() - t
        res["train_matrix_gb"] = Xtr.nbytes / 1e9
        res["n_columns"] = Xtr.shape[1]
        tr, va = ctx.sp["train"], ctx.sp["val"]
        t = time.time()
        r.m.fit(Xtr, tr["y"], Xva, va["y"], r.cols, cats_tr=tr["cat_str"], cats_va=va["cat_str"])
        res["fit_seconds"] = time.time() - t
        res["rounds"] = r.m.n_rounds_trained
        res["seconds_per_round"] = res["fit_seconds"] / max(1, r.m.n_rounds_trained)
        res["val_ap_best_in_rounds"] = r.m.best_val_ap
        res["best_round_in_rounds"] = r.m.best_rounds
        del Xtr, Xva
        ev = []
        for sp in ("val", "test"):
            t = time.time()
            r.predict(sp, 30)
            ev.append(time.time() - t)
        res["eval_pass_seconds(feature+predict, best_rounds trees)"] = ev
        res["feature_seconds_total"] = r.feature_seconds
    res["total_seconds"] = time.time() - t0
    res.update(mem.stop())
    out = C.EXP_DIR / "logs" / f"timing_{a.what}_L{a.L:03d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    log(f"timing {a.what} 끝: {json.dumps({k: v for k, v in res.items() if k != 'history'}, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()