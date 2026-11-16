# -*- coding: utf-8 -*-
"""GPU 메모리 상한(0.4)만 걸고 스크립트를 그대로 실행하는 감싸개(검증관 도구, 패키지 밖 — 계산은 바꾸지 않음).
lst/resources.py 와 같은 순서: 스레드 환경변수 8 → torch → set_per_process_memory_fraction(0.4) → 스크립트.
사용: python -X utf8 gpu_cap_run.py <스크립트> [인자 …]
"""
import os
import runpy
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "8"
import torch  # noqa: E402

if torch.cuda.is_available():
    for d in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(0.4, d)
    print(f"[gpu_cap_run] GPU 메모리 상한 0.4 · {torch.cuda.get_device_name(0)}", flush=True)
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
