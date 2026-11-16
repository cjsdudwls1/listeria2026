"""순열 중요도 나눠 돌리기 공통(설정 importance_parts·`--importance --part`) — scripts/s3_0b·scripts/s_final 이 씀.
계산 규칙(재생·h 마다 상태 파일·서명·합치기)은 lst/importance 머리 구현 세부 (h).

  parse_part(s)·parse_parts(parts)·part_name(p): 부분 = 'fold1'(그 fold h 1~60) 또는 'fold3:1-30'(그 fold h 구간).
      세 fold 모두, fold 마다 h 1~60 을 겹침·빈틈 없이 덮어야 함(아니면 설정 오류).
  launch_parts(cmds, logs): 부분 하위 프로세스를 한꺼번에 띄우고 모두 기다림(하나가 멈춰도 나머지는 끝까지),
      출력은 부분마다 로그 파일(이어 씀). 본 프로세스가 Ctrl-C 등으로 멈추면 띄운 부분 프로세스만 끝냄.
  run_parts(...): 남은 부분만 띄움 → 멈춘 부분은 실험 로그 'stopped' + RunStopped(다시 하면 남은 h 만) →
      그래도 남은 h 가 있으면 멈춤.

구현 세부(계획서에 없음 — 담당자 확인): 위 전부.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import config as cf
from . import evaluate as ev
from .hassign import H_MAX


def parse_part(s: str) -> tuple[str, int, int]:
    """'fold1' → (fold1, 1, 60), 'fold3:1-30' → (fold3, 1, 30)."""
    if not isinstance(s, str):
        raise cf.ConfigError(f"importance_parts 원소는 문자열: {s!r}")
    fold, sep, rng = s.partition(":")
    if fold not in ev.CV_FOLDS:
        raise cf.ConfigError(f"importance_parts: fold 이름은 {ev.CV_FOLDS} — {s!r}")
    if not sep:
        return fold, 1, H_MAX
    a, dash, b = rng.partition("-")
    if not dash or not a.isdigit() or not b.isdigit():
        raise cf.ConfigError(f"importance_parts: 'fold:처음-끝' 꼴 — {s!r}")
    lo, hi = int(a), int(b)
    if not (1 <= lo <= hi <= H_MAX):
        raise cf.ConfigError(f"importance_parts: h 구간 1~{H_MAX} — {s!r}")
    return fold, lo, hi


def parse_parts(parts) -> list[tuple[str, int, int]]:
    """부분 목록 점검: 세 fold 모두, fold 마다 h 1~60 을 겹침·빈틈 없이."""
    if not parts:
        raise cf.ConfigError("importance_parts 가 비었음(나누지 않으려면 칸을 뺄 것)")
    out = [parse_part(s) for s in parts]
    if len(set(out)) != len(out):
        raise cf.ConfigError(f"importance_parts 중복: {parts}")
    for f in ev.CV_FOLDS:
        rs = sorted((lo, hi) for ff, lo, hi in out if ff == f)
        nxt = 1
        for lo, hi in rs:
            if lo != nxt:
                raise cf.ConfigError(f"importance_parts: {f} h 구간이 겹치거나 빔(h {nxt} 부터여야 함): {parts}")
            nxt = hi + 1
        if nxt != H_MAX + 1:
            raise cf.ConfigError(f"importance_parts: {f} 가 h 1~{H_MAX} 를 다 덮지 않음: {parts}")
    return out


def part_name(p: tuple[str, int, int]) -> str:
    f, lo, hi = p
    return f"{f}:{lo}-{hi}"


def check_part_arg(part: str, parts) -> tuple[str, int, int]:
    """--part 값이 설정 importance_parts 의 한 부분인지."""
    p = parse_part(part)
    if p not in parse_parts(parts):
        raise cf.ConfigError(f"--part {part} 가 설정 importance_parts 에 없음")
    return p


def log_path(log_dir: Path, part: str) -> Path:
    return Path(log_dir) / f"{part.replace(':', '_h')}.log"


def launch_parts(cmds: list[list[str]], logs: list[Path]) -> list[int]:
    """부분 하위 프로세스를 한꺼번에 띄우고 모두 기다림(하나가 멈춰도 나머지는 끝까지). 본 프로세스가 Ctrl-C 등으로
    멈추면 띄운 부분 프로세스만 끝냄. 설정 speed.cpu_cores 가 있으면(lst/speed — 리눅스) 부분마다 코어 묶음
    (min(8, cpu_cores // 부분 수)개 — 스레드 수는 그대로)."""
    from . import speed

    slots = speed.core_slots(len(cmds), speed.get("cpu_cores")) if cmds else None
    procs, fhs = [], []
    try:
        for i, (c, lp) in enumerate(zip(cmds, logs)):
            lp.parent.mkdir(parents=True, exist_ok=True)
            fh = open(lp, "ab")
            fhs.append(fh)
            kw = {}
            pre = speed.affinity_preexec(slots[i] if slots else None)
            if pre is not None:
                kw["preexec_fn"] = pre
            procs.append(subprocess.Popen(c, stdout=fh, stderr=subprocess.STDOUT, **kw))
            print(f"부분 프로세스 PID {procs[-1].pid}: {c[-1]} → {lp}"
                  + (f" (코어 {slots[i]})" if slots else ""), flush=True)
        return [p.wait() for p in procs]
    except BaseException:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for fh in fhs:
            fh.close()


def run_parts(*, label: str, stage: str, parts: Sequence[str], pending_fn: Callable[[], list[str]], script: Path,
              argv: Sequence[str], log_dir: Path, exp_log, launch: Callable = launch_parts,
              run_in_process: Callable[[str], object] | None = None) -> None:
    """남은 부분을 한꺼번에(run_in_process 가 있으면 시험 — 같은 프로세스에서 차례로) → 멈춘 부분이 있으면 멈춤.
    끝나면 돌아옴(합치기는 부른 쪽)."""
    from . import runner as rn

    pend = pending_fn()
    exp_log.append("note", stage=stage, message=f"{label} 나눠 돌리기", parts=list(parts), pending=pend)
    print(f"{label} 나눠 돌리기: 부분 {list(parts)}, 남은 부분 {pend}")
    if pend and run_in_process is not None:
        for part in pend:
            run_in_process(part)
    elif pend:
        cmds = [[sys.executable, str(Path(script).resolve()), *argv, "--importance", "--part", part] for part in pend]
        logs = [log_path(log_dir, part) for part in pend]
        rcs = launch(cmds, logs)
        bad = [(part, rc) for part, rc in zip(pend, rcs) if rc != 0]
        for part, rc in bad:
            exp_log.append("stopped", stage=stage, name=f"{label} 부분 {part}", returncode=rc, reason=rn._rc_reason(rc))
        if bad:
            raise rn.RunStopped(f"{label} 부분 멈춤 — {[(p, rn._rc_reason(rc)) for p, rc in bad]}. 원인을 없앤 뒤 같은 "
                                f"명령을 다시(끝난 h 는 건너뜀 — B-1). 로그: {log_dir}")
    left = pending_fn()
    if left:
        raise rn.RunStopped(f"{label} 부분이 끝났는데 남은 h 가 있음: {left}")
