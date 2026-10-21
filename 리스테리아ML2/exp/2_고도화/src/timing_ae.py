# -*- coding: utf-8 -*-
"""HX(A3_L1·A3_U1·A3_L1U1)·AE(S_*·A3_AE*·A3_*_p) 시간·메모리 측정(전체 자료, 시드 42 한 번, 본 설정 그대로 - 본 결과 아님).
결과 = scratch/ae/timing.json, 실행 결과 폴더 = scratch/ae/timing_root/results (AE 점수 저장 = 그 아래 _ae_scores/, 시작할 때 지움).
results/·tables/·logs/·cache/ 에는 쓰지 않음. 기상 요약 캐시(cache/)는 메타가 같을 때만 읽기(다르면 쓰지 않고 그 자리에서 계산).
k* 는 results/selection_k.json 을 읽기만 함. AE 점수 계산 시간은 처음 쓰는 실험(S_*)에 들어감 -> 모델별로 따로 적음(ae_compute).

python src/timing_ae.py [실험ID ...]   (기본: 단계 'hx'·'ae' 실험 전부, 단계 순서)
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

SCR = E.EXP2_DIR / "scratch" / "ae"


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or [e["id"] for e in E.EXPERIMENTS if e["stage"] in ("hx", "ae")]
    root = SCR / "timing_root"
    P = {"results": root / "results", "tables": root / "tables", "logs": root / "logs", "cache": E.EXP2_DIR / "cache",
         "progress": root / "logs" / "progress.log"}
    for k in ("results", "logs"):
        P[k].mkdir(parents=True, exist_ok=True)
    if (P["results"] / "_ae_scores").exists():                  # AE 점수 계산 시간을 재려고 이전 저장분을 지움(측정용 폴더만)
        shutil.rmtree(P["results"] / "_ae_scores")
    log = Logger(SCR / "progress_timing_ae.log")
    mem = Mem()
    mem.start()
    t0 = time.time()
    res = {"system_available_gb_at_start": round(psutil.virtual_memory().available / 1e9, 2), "note":
           "전체 자료·시드 42 한 번. S_* 의 총 시간에 그 모델 AE 점수 계산(교차 적합 6모델 + 검증·시험 h 1~60)이 들어감, "
           "A3_* · A3_*_p 는 같은 시드 점수를 메모리에서 읽음. peak_gpu_used_mb_device = nvidia-smi 장치 전체 사용량(다른 프로그램 포함)"}
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    if not _cache_meta_ok(ctx):
        ctx.use_cache = False
        log("기상 요약 캐시 메타가 다름 -> 캐시 안 씀")
    res.update({"ctx_seconds": time.time() - t0, "runs": {}, "ae_compute": {}})
    win = check_windows(ctx, log)
    k_star = json.loads((E.EXP2_DIR / "results" / "selection_k.json").read_text(encoding="utf-8"))["k_star"]
    res["k_star"] = k_star
    for eid in ids:
        exp = E.EXP_BY_ID[eid]
        f = dict(exp["feats"])
        if f.get("k") == "k*":
            f["k"] = k_star
        exp = dict(exp, feats=f)
        d = P["results"] / E.run_dir_name(exp["id"], 42)
        if d.exists():
            shutil.rmtree(d)
        t = time.time()
        s = run_one(exp, 42, ctx, P, log, mem, False, win)
        r = {k: s.get(k) for k in ("total_seconds", "fit_seconds", "feature_seconds", "eval_seconds", "n_features",
                                   "n_train_rows", "n_train_pos", "best_rounds_or_epoch", "n_rounds_trained", "peak_private_gb",
                                   "peak_rss_gb", "peak_gpu_used_mb_device", "system_min_available_gb", "val_pr_auc_w", "test_pr_auc_w",
                                   "val_n_pos", "val_n_rows", "test_n_pos", "test_n_rows")}
        r["wall_seconds"] = time.time() - t
        if exp["model"] == "ae":
            ai = json.loads((d / "config.json").read_text(encoding="utf-8"))["model_info"]["ae"]
            kind = f["ae"][0]
            res["ae_compute"][kind] = {"seconds_total": ai["seconds_total"], "input_seconds": ai["input_seconds"],
                                       "eval_seconds": ai["eval_seconds"], "gpu_peak_allocated_gb": ai["gpu_peak_allocated_gb"],
                                       "gpu_peak_reserved_gb": ai["gpu_peak_reserved_gb"],
                                       "models": {m: {"best_epoch": v["best_epoch"], "epochs": v["epochs"], "seconds": v["seconds"],
                                                      "n_fit": v["n_fit"]} for m, v in ai["models"].items()},
                                       "record_only": ai["record_only"]}
        res["runs"][eid] = r
        (SCR / "timing.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    per_seed = sum(r["wall_seconds"] for r in res["runs"].values())
    res["estimate"] = {"one_seed_seconds": per_seed, "three_seeds_seconds": 3 * per_seed,
                       "rule": "시드 42 한 번의 실험별 실제 시간 합 x 3 (시드마다 AE 점수 4개 새로 계산, CatBoost 라운드 수는 시드마다 다를 수 있음)"}
    (SCR / "timing.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("timing_ae 끝: " + json.dumps({k: v for k, v in res.items() if k != "runs"}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
