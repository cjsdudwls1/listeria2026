# -*- coding: utf-8 -*-
"""DL 시간·메모리 측정(전체 자료, 에폭 제한 - 본 결과 아님). 결과 = logs/timing_dl_<id>.json. results/ 는 건드리지 않음.

python src/timing_dl.py --what DL_lstmA|DL_trfA|DL_lstmB|DL_trfB [--epochs 2]
  에폭 N 개만 학습(조기 종료 없음) -> 에폭당 학습 시간·조기 종료용 검증 시간, 평가(검증·시험 x h 60) 시간,
  GPU 메모리(torch 할당·예약 최대, 단계별) + nvidia-smi 장치 전체 최대, 프로세스 메모리 최대(rss·private)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem  # noqa: E402
import e2dl  # noqa: E402,F401
from e2dl_impl import DLRunner  # noqa: E402

ARCH = {"DL_lstmA": ("lstm", "A"), "DL_trfA": ("trf", "A"), "DL_lstmB": ("lstm", "B"), "DL_trfB": ("trf", "B")}
GB = 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=list(ARCH))
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    log = Logger(E.EXP2_DIR / "logs" / "progress_timing.log")
    log(f"===== timing_dl {a.what} epochs={a.epochs} pid {os.getpid()}")
    mem = Mem(gpu_every=2)
    mem.start()
    t0 = time.time()
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    res = {"what": a.what, "epochs_limit": a.epochs, "ctx_seconds": time.time() - t0, "mem_after_ctx": mem.snapshot()}
    arch, mode = ARCH[a.what]
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.time()
    r = DLRunner(ctx, {"id": a.what}, 42, log, False, arch, mode, max_epochs=a.epochs, patience=10 ** 6)
    res["init_seconds(입력 표준화·GPU 올리기·B 정답)"] = time.time() - t
    res["gpu_data_allocated_gb(기상 38채널·행 텐서·B 정답)"] = (torch.cuda.memory_allocated() - base) / GB
    res["gpu_W_seq_gb"] = r.Wg.numel() * 4 / GB
    if mode == "B":
        res["gpu_B_labels_gb"] = (r.T["train"]["Y"].numel() * 4 + r.T["train"]["M"].numel() * 4) / GB
        res["mask_stats"] = r.mask_info
    res["mem_after_init"] = mem.snapshot()
    t = time.time()
    r.fit()
    res["fit_seconds"] = r.fit_seconds
    res["epochs"] = [{k: v for k, v in h.items()} for h in r.history]
    res["seconds_per_epoch_train"] = float(np.mean([h["train_seconds"] for h in r.history]))
    res["seconds_per_epoch_val_es"] = float(np.mean([h["val_seconds"] for h in r.history]))
    res["batches_per_epoch"] = r.history[0]["batches"]
    res["n_params"] = r.n_params
    res["gpu_train_peak_allocated_gb"] = r.gpu_train_peak_alloc_gb
    res["gpu_train_peak_reserved_gb"] = r.gpu_train_peak_reserved_gb
    res["mem_after_fit"] = mem.snapshot()
    t = time.time()
    for sp in ("val", "test"):
        for h in range(1, E.H_MAX + 1):
            r.predict(sp, h)
    res["eval_seconds(검증·시험 x h60)"] = time.time() - t
    res["gpu_eval_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / GB
    res["gpu_eval_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / GB
    r.close()
    res["estimate_full_run_seconds(30에폭 상한)"] = res["init_seconds(입력 표준화·GPU 올리기·B 정답)"] + 30 * (
        res["seconds_per_epoch_train"] + res["seconds_per_epoch_val_es"]) + res["eval_seconds(검증·시험 x h60)"]
    res["total_seconds"] = time.time() - t0
    res["peak"] = mem.stop()
    out = E.EXP2_DIR / "logs" / f"timing_dl_{a.what}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"timing_dl {a.what} 끝: 에폭당 학습 {res['seconds_per_epoch_train']:.2f}s + 검증 {res['seconds_per_epoch_val_es']:.2f}s, "
        f"평가 {res['eval_seconds(검증·시험 x h60)']:.1f}s, GPU 학습 최대 할당 {r.gpu_train_peak_alloc_gb:.3f}GB/예약 {r.gpu_train_peak_reserved_gb:.3f}GB, "
        f"장치 {res['peak']['peak_gpu_used_mb_device']}MB, private {res['peak']['peak_private_gb']}GB")


if __name__ == "__main__":
    main()
