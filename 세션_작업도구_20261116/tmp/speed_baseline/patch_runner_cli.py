import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel, pairs):
    p = root / rel
    s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:80])
        s = s.replace(old, new)
    p.write_bytes(s.encode('utf-8'))


# ------------------------------------------------------------------ runner
patch('lst/runner.py', [(
'''def run_variants_isolated(script, groups: Sequence[Sequence[str] | str], base_argv: Sequence[str],
                          exp_log: ExperimentLog, stage: str, python: str | None = None) -> list[str]:
    """묶음(판 이름 목록 — 문자열 하나면 판 하나)마다 `python <script> <base_argv> --variant <판,판,…>` 하위
    프로세스(B-1). 하나라도 멈추면 전체 멈춤."""
    py = python or sys.executable
    finished = []
''',
'''def run_variants_isolated(script, groups: Sequence[Sequence[str] | str], base_argv: Sequence[str],
                          exp_log: ExperimentLog, stage: str, python: str | None = None, *, processes: int = 1,
                          cpu_cores: int | None = None, serial: Iterable[str] = (), log_dir: Path | None = None,
                          launch: Callable | None = None) -> list[str]:
    """묶음(판 이름 목록 — 문자열 하나면 판 하나)마다 `python <script> <base_argv> --variant <판,판,…>` 하위
    프로세스(B-1). 하나라도 멈추면 전체 멈춤.
    processes > 1(설정 speed.parallel_groups — lst/speed, 구현 세부(계획서에 없음)) = 묶음을 그 수만큼 동시에
    (`_run_parallel`). 1 이면 예전 그대로(차례로)."""
    if processes > 1:
        return _run_parallel(script, groups, base_argv, exp_log, stage, python, processes=processes,
                             cpu_cores=cpu_cores, serial=serial, log_dir=log_dir, launch=launch)
    py = python or sys.executable
    finished = []
'''), (
'''# ================================================================ 단계 표·보고
def load_done(out_dir, name: str) -> tuple[dict, dict, dict]:''',
'''PARALLEL_LOCK = "_parallel.lock"
PARALLEL_LOG_DIR = "_parallel_logs"


class _StageLock:
    """동시 실행 모드: 같은 단계 폴더를 두 부모 프로세스가 같이 돌리지 않게(잡히면 멈춤 — 같은 판 두 번 방지)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise RunStopped(f"이 단계 폴더를 다른 프로세스가 돌리는 중(잠금 {self.path}) — 같은 판을 두 번 돌리지 않음")
        return self

    def __exit__(self, *exc):
        try:
            if sys.platform == "win32":
                import msvcrt

                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()
        return False


def _popen_group(cmd: list[str], log_path: Path, cores: list[int] | None):
    from . import speed

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "ab")
    kw = {"stdout": fh, "stderr": subprocess.STDOUT}
    pre = speed.affinity_preexec(cores)
    if pre is not None:
        kw["preexec_fn"] = pre
    return subprocess.Popen(cmd, **kw), fh


def _run_parallel(script, groups, base_argv, exp_log: ExperimentLog, stage: str, python: str | None, *,
                  processes: int, cpu_cores: int | None, serial: Iterable[str], log_dir: Path | None,
                  launch: Callable | None) -> list[str]:
    """동시 실행(구현 세부(계획서에 없음) — speed.parallel_groups): 묶음 하위 프로세스를 최대 processes 개 동시에.
    - 묶음 목록 차례대로 빈 자리에 띄움. serial 판(2차 사슬 부스팅 — 최대 반복 기록을 앞 판이 정하고 뒤 판이 읽음)이 든
      묶음끼리는 차례대로(앞 serial 묶음이 끝난 뒤 다음 serial 묶음). 묶음 하나는 한 프로세스만(같은 판 두 번 없음).
    - 프로세스마다 코어 묶음(speed.core_slots — 리눅스 sched_setaffinity, 스레드 수는 그대로), 출력은
      <log_dir>/<묶음 첫 판>.log(이어 씀).
    - 하나가 멈추면 새로 띄우지 않고 돌던 것은 끝까지 기다린 뒤 실험 로그에 남기고 멈춤(B-1 — 다시 하면 남은 판만).
    - launch(cmd, log, cores) → Popen 꼴 객체(시험용 주입)."""
    import time as _time

    from . import speed

    py = python or sys.executable
    gl = [[g] if isinstance(g, str) else list(g) for g in groups]
    gl = [g for g in gl if g]
    ser = set(serial)
    log_dir = Path(log_dir) if log_dir is not None else None
    slots = speed.core_slots(processes, cpu_cores)
    free = list(range(processes))
    queue = list(range(len(gl)))
    running: dict[int, tuple] = {}          # 묶음 번호 → (proc, fh, slot)
    serial_busy = False
    finished, bad = [], []
    print(f"묶음 동시 실행: 묶음 {len(gl)}개, 동시 {processes}, 코어 묶음 {slots}", flush=True)
    try:
        while queue or running:
            started = False
            if not bad:
                for i in list(queue):
                    if not free:
                        break
                    is_ser = any(n in ser for n in gl[i])
                    if is_ser and serial_busy:
                        continue
                    if is_ser and any(any(n in ser for n in gl[j]) for j in queue if j < i):
                        continue            # 앞 serial 묶음이 아직 시작 안 함 — 차례 지킴
                    slot = free.pop(0)
                    cores = slots[slot] if slots else None
                    arg = ",".join(gl[i])
                    cmd = [py, str(script), *base_argv, "--variant", arg]
                    lp = (log_dir or Path(".")) / f"{gl[i][0]}.log"
                    if launch is not None:
                        proc, fh = launch(cmd, lp, cores), None
                    else:
                        proc, fh = _popen_group(cmd, lp, cores)
                    print(f"묶음 시작 PID {getattr(proc, 'pid', '?')}: {arg} (코어 {cores}) → {lp}", flush=True)
                    running[i] = (proc, fh, slot, is_ser)
                    serial_busy = serial_busy or is_ser
                    queue.remove(i)
                    started = True
            elif not running:
                break
            done_any = False
            for i, (proc, fh, slot, is_ser) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                done_any = True
                if fh is not None:
                    fh.close()
                del running[i]
                free.append(slot)
                free.sort()
                if is_ser:
                    serial_busy = False
                arg = ",".join(gl[i])
                if rc != 0:
                    reason = _rc_reason(rc)
                    exp_log.append("stopped", stage=stage, name=arg, returncode=rc, reason=reason)
                    bad.append((arg, reason))
                    print(f"묶음 멈춤: {arg} — {reason}", flush=True)
                else:
                    finished.extend(gl[i])
                    print(f"묶음 끝: {arg}", flush=True)
            if not started and not done_any:
                _time.sleep(2.0)
    except BaseException:
        for proc, fh, _, _ in running.values():
            if proc.poll() is None:
                proc.terminate()
        raise
    if bad:
        raise RunStopped(f"판 묶음 멈춤 {bad} — 원인을 없앤 뒤 같은 명령을 다시 실행하면 남은 판만(B-1). 로그: {log_dir}")
    return finished


# ================================================================ 단계 표·보고
def load_done(out_dir, name: str) -> tuple[dict, dict, dict]:'''),
])

# ------------------------------------------------------------------ stage_cli
patch('lst/stage_cli.py', [(
'''    "scope": scope_.FIELD,                    # 국내만 버전(configs_dom — lst/scope.py), 없으면 지금 판
''',
'''    "scope": scope_.FIELD,                    # 국내만 버전(configs_dom — lst/scope.py), 없으면 지금 판
    "speed": cf.Field(dict, required=False),  # 속도 개선 스위치(lst/speed — 없으면 전부 꺼짐, 결과 불변)
'''), (
'''    scope_.activate(cfg.get("scope"))         # 범위·결과 뿌리 표지 대조(실행 인자는 parse_paths·_pre 가 받아 둠)
''',
'''    scope_.activate(cfg.get("scope"))         # 범위·결과 뿌리 표지 대조(실행 인자는 parse_paths·_pre 가 받아 둠)
    from . import speed as speed_

    speed_.configure(cfg.get("speed"))        # 이 프로세스 전역 스위치(하위 프로세스도 같은 설정을 읽어 같게)
'''), (
'''    todo = [[s.name for s in g if s.name in pend] for g in groups]
    rn.run_variants_isolated(script.resolve(), [t for t in todo if t], argv_for(rp.out), exp_log, cfg["stage"])
''',
'''    todo = [[s.name for s in g if s.name in pend] for g in groups]
    from . import speed as speed_

    npar = speed_.get("parallel_groups")
    if npar > 1:   # 동시 실행(lst/speed) — 같은 단계 폴더 잠금 안에서
        serial = [s.name for s in specs if rn.is_chain_boost(s)]
        with rn._StageLock(rp.out / rn.PARALLEL_LOCK):
            rn.run_variants_isolated(script.resolve(), [t for t in todo if t], argv_for(rp.out), exp_log,
                                     cfg["stage"], processes=npar, cpu_cores=speed_.get("cpu_cores"), serial=serial,
                                     log_dir=rp.out / rn.PARALLEL_LOG_DIR)
    else:
        rn.run_variants_isolated(script.resolve(), [t for t in todo if t], argv_for(rp.out), exp_log, cfg["stage"])
'''),
])

# ------------------------------------------------------------------ imp_parts (중요도 부분 동시 — 코어 묶음)
patch('lst/imp_parts.py', [(
'''def launch_parts(cmds: list[list[str]], logs: list[Path]) -> list[int]:
    """부분 하위 프로세스를 한꺼번에 띄우고 모두 기다림(하나가 멈춰도 나머지는 끝까지). 본 프로세스가 Ctrl-C 등으로
    멈추면 띄운 부분 프로세스만 끝냄."""
    procs, fhs = [], []
    try:
        for c, lp in zip(cmds, logs):
            lp.parent.mkdir(parents=True, exist_ok=True)
            fh = open(lp, "ab")
            fhs.append(fh)
            procs.append(subprocess.Popen(c, stdout=fh, stderr=subprocess.STDOUT))
            print(f"부분 프로세스 PID {procs[-1].pid}: {c[-1]} → {lp}", flush=True)
''',
'''def launch_parts(cmds: list[list[str]], logs: list[Path]) -> list[int]:
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
'''),
])
print("ok")
