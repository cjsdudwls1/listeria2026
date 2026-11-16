"""서버 자원 규칙 — 공통 규칙 '서버 자원', 계획서 부록 B-1.

- 스레드 8(OMP/MKL/OPENBLAS/NUMEXPR), 알고리즘 n_jobs ≤ 8.
- GPU 메모리 `torch.cuda.set_per_process_memory_fraction(0.4)` 이하.
- 가용 RAM 15GB·디스크 여유 10GB 아래면 내 작업만 멈춤(예외).

구현 세부(계획서에 없음): GB = 2**30 바이트(GiB — 더 보수적).
구현 세부(계획서에 없음): 감시는 호출 시점 검사 함수(`check_resources`) — 학습·fold·h 묶음
시작 전에 부르는 것으로 씀(백그라운드 감시 스레드는 두지 않음).
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings

THREADS = 8
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
N_JOBS_MAX = 8
GPU_MEMORY_FRACTION = 0.4
MIN_AVAILABLE_RAM_GB = 15
MIN_FREE_DISK_GB = 10
GB = 1024 ** 3


class ResourceFloorError(RuntimeError):
    """가용 RAM·디스크 여유가 바닥값 아래 — 내 작업만 멈춤(B-1: 원인을 없앤 뒤 그 판부터 다시)."""


def set_thread_env(n: int = THREADS) -> dict[str, str]:
    """스레드 환경변수 설정. numpy 등 BLAS 를 불러오기 전에 불러야 효과가 확실함."""
    if not (1 <= n <= THREADS):
        raise ValueError(f"스레드 수는 1~{THREADS}: {n}")
    if any(m in sys.modules for m in ("numpy", "torch", "sklearn")):
        warnings.warn("numpy/torch/sklearn 을 이미 불러옴 — 스레드 환경변수는 import 전에 설정해야 확실함",
                      RuntimeWarning, stacklevel=2)
    for v in THREAD_ENV_VARS:
        os.environ[v] = str(n)
    return {v: os.environ[v] for v in THREAD_ENV_VARS}


def check_n_jobs(n_jobs: int) -> int:
    """알고리즘 n_jobs ≤ 8(-1·None 금지)."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not (1 <= n_jobs <= N_JOBS_MAX):
        raise ValueError(f"n_jobs 는 1~{N_JOBS_MAX} 정수: {n_jobs!r}")
    return n_jobs


def limit_gpu_memory(fraction: float = GPU_MEMORY_FRACTION, device: int | None = None) -> bool:
    """GPU 메모리 상한. CUDA 가 없으면 아무것도 안 하고 False."""
    if not (0 < fraction <= GPU_MEMORY_FRACTION):
        raise ValueError(f"GPU 메모리 비율은 0 초과 {GPU_MEMORY_FRACTION} 이하: {fraction}")
    import torch  # 늦게 불러옴(트리 작업엔 torch 불필요)

    if not torch.cuda.is_available():
        return False
    devices = range(torch.cuda.device_count()) if device is None else [device]
    for d in devices:
        torch.cuda.set_per_process_memory_fraction(fraction, d)
    return True


def available_ram_bytes() -> int:
    try:
        import psutil  # 있으면 씀

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("/proc/meminfo 에 MemAvailable 없음")
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            raise RuntimeError("GlobalMemoryStatusEx 실패")
        return int(st.ullAvailPhys)
    raise RuntimeError(f"가용 메모리를 못 읽는 플랫폼: {sys.platform}")


def free_disk_bytes(path: str | os.PathLike) -> int:
    return int(shutil.disk_usage(path).free)


def check_resources(path: str | os.PathLike) -> dict[str, float]:
    """가용 RAM·디스크 여유(path 가 있는 디스크) 검사. 바닥값 아래면 ResourceFloorError."""
    ram = available_ram_bytes() / GB
    disk = free_disk_bytes(path) / GB
    problems = []
    if ram < MIN_AVAILABLE_RAM_GB:
        problems.append(f"가용 RAM {ram:.1f}GB < {MIN_AVAILABLE_RAM_GB}GB")
    if disk < MIN_FREE_DISK_GB:
        problems.append(f"디스크 여유 {disk:.1f}GB < {MIN_FREE_DISK_GB}GB ({path})")
    if problems:
        raise ResourceFloorError("; ".join(problems))
    return {"available_ram_gb": ram, "free_disk_gb": disk}
