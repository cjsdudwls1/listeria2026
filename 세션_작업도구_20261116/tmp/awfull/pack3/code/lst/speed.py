"""속도 개선 스위치(결과 불변 — 사용자 승인 2026-11-12, 구현 세부(계획서에 없음)).

설정 칸 "speed"(모든 단계 설정·s_final 설정에 넣을 수 있음, 없으면 전부 꺼짐 = 지금과 바이트 단위로 같은 동작):
  dl_keyed_inputs  (bool)  딥러닝 입력 transform: 시퀀스(x_seq)를 (sgg, E, T, src_route) 고유값 표에서 표준화·결측 0·
                           결측 표시까지 끝낸 뒤 행으로 한 번만 펼침(원소별 계산이라 값 같음 — lst/inputs 구현 세부 (h)).
                           트리 입력은 블록을 키 표에서 X 칸으로 바로 펼침(중간 배열 한 번 덜 만듦).
  fast_predict     (bool)  딥러닝 예측·이상 점수: 행 번호 배열 a[np.arange] 대신 연속 구간 자르기 + 고정 메모리(pinned)로
                           GPU 에 보냄(배치 경계·크기·순서 그대로 — models_dl 구현 세부 18).
  gpu_importance   (bool)  순열 중요도 딥러닝 잎: h 마다 그 h 입력을 GPU 에 한 번 올리고 섞기·배치 만들기를 GPU 색인으로
                           (섞기 순열(난수)·배치 순서·배치 크기 그대로 — importance 구현 세부 (i)). 트리 잎은 CPU 그대로.
  parallel_groups  (int)   단계 기본 실행에서 판 묶음(입력 묶음 group) 하위 프로세스를 동시에 몇 개(1 = 지금처럼 차례로).
                           2차 사슬 부스팅 판이 든 묶음은 서로 차례대로(최대 반복 기록을 앞 판이 정하고 뒤 판이 읽음).
  cpu_cores        (int)   동시 실행 하위 프로세스(판 묶음·중요도 부분)가 나눠 쓰는 코어 수 합(≤ 12). 프로세스마다
                           min(8, cpu_cores // 동시 수)개 코어에 묶음(리눅스 sched_setaffinity). 스레드 수(n_jobs·
                           OMP 등)는 바꾸지 않음 — 스레드 수가 바뀌면 일부 알고리즘 결과가 달라질 수 있어서.
  gpu_fraction_cap (float) 이 프로세스·하위 프로세스의 GPU 메모리 상한(0 초과 0.4 이하, 예 0.25 = 10GB) — 설정의
                           gpu_memory_fraction 이 더 크면 이 값으로 낮춤(resources.limit_gpu_memory).
"""
from __future__ import annotations

from typing import Any, Mapping

from . import config as cf

MAX_CORES = 12
FLAGS = ("dl_keyed_inputs", "fast_predict", "gpu_importance")

FIELD = {
    "dl_keyed_inputs": cf.Field(bool, required=False),
    "fast_predict": cf.Field(bool, required=False),
    "gpu_importance": cf.Field(bool, required=False),
    "parallel_groups": cf.Field(int, required=False),
    "cpu_cores": cf.Field(int, required=False),
    "gpu_fraction_cap": cf.Field(float, required=False),
}

_STATE: dict[str, Any] = {}


def _defaults() -> dict:
    return {"dl_keyed_inputs": False, "fast_predict": False, "gpu_importance": False, "parallel_groups": 1,
            "cpu_cores": None, "gpu_fraction_cap": None}


_STATE.update(_defaults())


def check(section: Mapping | None) -> dict:
    """설정 칸 점검 → 채운 값(빠진 칸 = 꺼짐)."""
    out = _defaults()
    if section is None:
        return out
    unknown = set(section) - set(FIELD)
    if unknown:
        raise cf.ConfigError(f"speed 에 모르는 칸: {sorted(unknown)}")
    out.update({k: v for k, v in section.items()})
    for k in FLAGS:
        if not isinstance(out[k], bool):
            raise cf.ConfigError(f"speed.{k} 는 true/false")
    pg = out["parallel_groups"]
    if isinstance(pg, bool) or not isinstance(pg, int) or pg < 1:
        raise cf.ConfigError(f"speed.parallel_groups ≥ 1 정수: {pg!r}")
    cc = out["cpu_cores"]
    if cc is not None and (isinstance(cc, bool) or not isinstance(cc, int) or not 1 <= cc <= MAX_CORES):
        raise cf.ConfigError(f"speed.cpu_cores 는 1~{MAX_CORES}: {cc!r}")
    if pg > 1 and cc is None:
        raise cf.ConfigError("speed.parallel_groups > 1 이면 cpu_cores 도(코어 합 ≤ 12)")
    if cc is not None and pg > cc:
        raise cf.ConfigError(f"speed.parallel_groups {pg} > cpu_cores {cc}")
    cap = out["gpu_fraction_cap"]
    if cap is not None:
        from .resources import GPU_MEMORY_FRACTION

        if not (0 < float(cap) <= GPU_MEMORY_FRACTION):
            raise cf.ConfigError(f"speed.gpu_fraction_cap 은 0 초과 {GPU_MEMORY_FRACTION} 이하: {cap}")
        out["gpu_fraction_cap"] = float(cap)
    return out


def configure(section: Mapping | None) -> dict:
    """설정 칸 → 이 프로세스 전역 스위치(설정을 읽는 곳 — stage_cli.load_cfg·s_final.load_cfg — 이 부름)."""
    st = check(section)
    _STATE.clear()
    _STATE.update(st)
    from . import resources as rs

    rs.set_gpu_cap(st["gpu_fraction_cap"])
    return dict(st)


def on(flag: str) -> bool:
    if flag not in FLAGS:
        raise KeyError(flag)
    return bool(_STATE[flag])


def get(name: str):
    return _STATE[name]


def state() -> dict:
    return dict(_STATE)


def reset_for_tests() -> None:
    _STATE.clear()
    _STATE.update(_defaults())
    from . import resources as rs

    rs.set_gpu_cap(None)


# ---------------------------------------------------------------- 코어 묶기(동시 실행 하위 프로세스)
def core_slots(n_procs: int, cpu_cores: int | None) -> list[list[int]] | None:
    """동시 n_procs 개 프로세스의 코어 묶음(리눅스만 — 다른 OS·cpu_cores 없음이면 None = 묶지 않음).
    프로세스마다 min(8, cpu_cores // n_procs)개(최소 1), 이 프로세스가 쓸 수 있는 코어의 앞 cpu_cores 개에서 나눔."""
    import os

    if cpu_cores is None or not hasattr(os, "sched_getaffinity"):
        return None
    avail = sorted(os.sched_getaffinity(0))[:cpu_cores]
    n = max(1, int(n_procs))
    k = max(1, min(8, len(avail) // n))
    return [[avail[(i * k + j) % len(avail)] for j in range(k)] for i in range(n)]


def affinity_preexec(cores: list[int] | None):
    """subprocess.Popen preexec_fn(리눅스) — 하위 프로세스를 그 코어에 묶음."""
    if not cores:
        return None
    import os

    cs = set(int(c) for c in cores)

    def _f():
        os.sched_setaffinity(0, cs)
    return _f
