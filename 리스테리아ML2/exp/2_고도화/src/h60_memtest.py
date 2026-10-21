# -*- coding: utf-8 -*-
"""H60 메모리 방법 시험: 전체 학습 행 x h 60개(약 1,337만 행) E00 입력으로 학습 Pool 을 만들고 quantize 까지만 한다
(학습은 하지 않음 - 지시). 결과: logs/h60_memtest.json.

방법: 학습 행 x h 를 디스크 메모리맵(float32, 열 우선 F-order, h 우선 행 순서)에 h 하나씩 쓰고,
FeaturesData(num_feature_data=메모리맵, cat_feature_data=범주 문자열) 로 Pool 을 만든 뒤 Pool.quantize().
- CatBoost 는 열 우선 float32 숫자 입력을 복사하지 않고 참조한다(행 우선은 복사 - scratch 시험에서 확인).
- 파일로 써서 catboost.utils.quantize 로 블록 단위 양자화하는 방법은 범주 변수가 있으면 안 됨
  ('Categorical features are not supported in block quantization' 오류 - scratch 시험에서 확인).
측정: 단계별 작업 집합(rss, 메모리맵 파일 페이지 포함)과 커밋된 개인 메모리(private) 최대값.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2cfg as E  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from common import Logger  # noqa: E402
from e2data import Ctx2  # noqa: E402
from e2eval import Mem  # noqa: E402
from e2models import FeatureBuilder, build_h60_pool  # noqa: E402


def main():
    log = Logger(E.EXP2_DIR / "logs" / "progress_h60_memtest.log")
    log(f"===== H60 메모리 시험 pid {os.getpid()}")
    mem = Mem(gpu_every=0)
    mem.start()
    t0 = time.time()
    vm0 = psutil.virtual_memory()
    res = {"system_total_gb": round(vm0.total / 1e9, 2), "system_available_gb_at_start": round(vm0.available / 1e9, 2)}
    ctx = Ctx2(False, log, cache_dir=E.EXP2_DIR / "cache")
    res["ctx_seconds"] = round(time.time() - t0, 1)
    res["mem_after_ctx"] = mem.snapshot()
    fb = FeatureBuilder(ctx, dict(E.BASE_FEATS))
    pos = np.arange(len(ctx.sp["train"]["idx"]))
    min_avail = [psutil.virtual_memory().available]

    def watch():
        import threading

        def loop():
            while not stop[0]:
                min_avail[0] = min(min_avail[0], psutil.virtual_memory().available)
                time.sleep(0.5)
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th
    stop = [False]
    th = watch()
    t1 = time.time()
    pool, info = build_h60_pool(ctx, fb, pos, 42, E.EXP2_DIR / "cache", log, mem)
    res["h60_total_seconds"] = round(time.time() - t1, 1)
    res.update(info)
    res["quantized_pool_shape"] = list(pool.shape)
    res["is_quantized"] = bool(pool.is_quantized())
    res["mem_after_quantize_final"] = mem.snapshot()
    del pool
    import gc
    gc.collect()
    res["mem_after_free"] = mem.snapshot()
    stop[0] = True
    th.join(timeout=5)
    res["system_min_available_gb"] = round(min_avail[0] / 1e9, 2)
    res["peak"] = mem.stop()
    res["total_seconds"] = round(time.time() - t0, 1)
    res["limit_gb"] = E.H60_RAM_LIMIT_GB
    res["private_peak_within_limit"] = res["peak"]["peak_private_gb"] <= E.H60_RAM_LIMIT_GB
    res["note"] = ("rss = 작업 집합(메모리맵 파일 페이지 포함, OS 가 필요하면 내보냄), private = 커밋된 개인 메모리. "
                   "양자화 뒤 학습 단계 메모리는 재지 않음(지시: quantize 단계까지만)")
    out = E.EXP2_DIR / "logs" / "h60_memtest.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"H60 메모리 시험 끝: {json.dumps(res, ensure_ascii=False, default=str)}")


if __name__ == "__main__":
    main()
