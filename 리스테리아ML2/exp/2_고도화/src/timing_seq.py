# -*- coding: utf-8 -*-
"""--stage all 처럼 한 프로세스에서 여러 실행을 차례로 돌릴 때의 시간·메모리 측정(전체 자료, 라운드·에폭 제한 - 본 결과 아님).

run.py 의 run_one 을 그대로 쓰되(특성·학습·평가 h60·예측 저장 전부) 라운드·에폭만 줄인다(조기 종료 끔).
결과 폴더는 scratch/_timing_seq (results/ 는 건드리지 않음), 기록은 logs/timing_seq.json.
목적: (1) 실행 1회의 고정 시간(특성·평가·저장)과 라운드당 시간, (2) 앞 실행들 뒤에 같은 프로세스에서
H10 을 두 번, H30 을 돌릴 때 개인 메모리(private)가 새 프로세스 측정값보다 커지는지.

python src/timing_seq.py [--cat-rounds 100] [--h10-rounds 30] [--h30-rounds 5] [--dl-epochs 2] [--skip-h30]
"""
import argparse
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
import run as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-rounds", type=int, default=100)
    ap.add_argument("--h10-rounds", type=int, default=30)
    ap.add_argument("--h30-rounds", type=int, default=5)
    ap.add_argument("--dl-epochs", type=int, default=2)
    ap.add_argument("--skip-h30", action="store_true")
    ap.add_argument("--plan", default="full", choices=["full", "h30"],
                    help="h30: E00(평가로 캐시 페이지를 올림) 다음 H30 만 - 평가 캐시를 놓는 효과 확인용")
    a = ap.parse_args()
    out_name = "timing_seq.json" if a.plan == "full" else f"timing_seq_{a.plan}.json"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out_dir = E.EXP2_DIR / "scratch" / "_timing_seq"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    P = dict(E.paths(False))
    P["results"] = out_dir
    log = Logger(E.EXP2_DIR / "logs" / "progress_timing.log")
    log(f"===== timing_seq {vars(a)} pid {os.getpid()}")
    E.PATIENCE = 10 ** 6                 # 조기 종료 끔: 정해진 라운드만
    E.DL["patience"] = 10 ** 6
    E.DL["max_epochs"] = a.dl_epochs
    mem = Mem(gpu_every=4)
    mem.start()
    t0 = time.time()
    ctx = Ctx2(False, log, cache_dir=P["cache"])
    win = R.check_windows(ctx, log)
    res = {"args": vars(a), "ctx_seconds": time.time() - t0, "mem_after_ctx": mem.snapshot(),
           "system_available_gb_at_start": round(psutil.virtual_memory().available / 1e9, 3), "runs": []}
    k = 50                                # 측정용(결과와 무관)
    A3 = dict(E.EXP_BY_ID["A3"]["feats"], k=k)
    plan = [("E00", 42, None, a.cat_rounds), ("A3", 42, A3, a.cat_rounds), ("B2", 42, dict(E.EXP_BY_ID["B2"]["feats"], k=k), None),
            ("C3_hist", 42, A3, a.cat_rounds), ("D1", 42, None, a.cat_rounds), ("D2", 42, None, a.cat_rounds),
            ("D3", 42, None, a.cat_rounds), ("H10", 42, None, a.h10_rounds), ("H10", 43, None, a.h10_rounds)]
    if not a.skip_h30:
        plan.append(("H30", 42, None, a.h30_rounds))
    plan += [("DL_lstmA", 42, None, None), ("DL_trfB", 42, None, None)]
    if a.plan == "h30":
        plan = [("E00", 42, None, a.cat_rounds), ("H30", 42, None, a.h30_rounds)]
    for eid, seed, feats, rounds in plan:
        exp = dict(E.EXP_BY_ID[eid])
        if feats is not None:
            exp["feats"] = feats
        if rounds is not None:
            E.MAX_ROUNDS = rounds
        before = mem.snapshot()
        avail0 = psutil.virtual_memory().available / 1e9
        t = time.time()
        s = R.run_one(exp, seed, ctx, P, log, mem, False, win)
        cfg = json.loads((out_dir / E.run_dir_name(eid, seed) / "config.json").read_text(encoding="utf-8"))
        mi = cfg.get("model_info") or {}
        rec = {"run": f"{eid}__s{seed}", "rounds_limit": rounds, "seconds": round(time.time() - t, 1),
               "fit_seconds": s.get("fit_seconds"), "feature_seconds": s.get("feature_seconds"),
               "eval_seconds": s.get("eval_seconds"), "n_rounds_trained": s.get("n_rounds_trained"),
               "sec_per_round": (s["fit_seconds"] / s["n_rounds_trained"]) if s.get("n_rounds_trained") else None,
               "n_train_rows": s.get("n_train_rows"),
               "private_before_gb": before["private_gb"], "rss_before_gb": before["rss_gb"],
               "system_available_before_gb": round(avail0, 3),
               "peak_private_gb": s.get("peak_private_gb"), "peak_rss_gb": s.get("peak_rss_gb"),
               "system_min_available_gb": s.get("system_min_available_gb"),
               "peak_gpu_used_mb_device": s.get("peak_gpu_used_mb_device"),
               "mem_train_phase": mi.get("mem_train_phase"),
               "h60_info": {kk: vv for kk, vv in ((mi.get("extra") or {}).get("h60") or {}).items()
                            if kk.endswith("seconds") or kk.startswith("mem_")}}
        rec["private_after_gb"] = mem.snapshot()["private_gb"]
        res["runs"].append(rec)
        log(f"  측정 {rec['run']}: {rec['seconds']}s, private 앞 {rec['private_before_gb']} -> 최대 {rec['peak_private_gb']} -> 뒤 {rec['private_after_gb']} GB, "
            f"사용 가능 최소 {rec['system_min_available_gb']} GB")
        (E.EXP2_DIR / "logs" / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    res["total_seconds"] = time.time() - t0
    res["process_peak"] = mem.stop()
    (E.EXP2_DIR / "logs" / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"timing_seq 끝: {res['total_seconds']:.0f}s, 프로세스 최대 {res['process_peak']}")
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
