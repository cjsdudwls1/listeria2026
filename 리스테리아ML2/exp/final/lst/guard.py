"""학습 멈춤 처리 — 계획서 부록 B-1('학습이 멈추면 … 시도·학습을 하위 프로세스로 돌려 비정상 종료·메모리 관련 예외로
앎, 세부는 단계 0 구현 세부 … 건너뛰거나 실패로 넘기지 않고, 결과를 바꾸지 않는 방법 … 으로 원인을 없앤 뒤 그
알고리즘·그 판(시도)부터만 다시 돌려 … 다시 돌린 판·이유는 실험 로그에').

    run_in_child(func_ref, kwargs, *, timeout=None) -> 반환값        # 실패면 ChildFailure
    run_guarded(func_ref, kwargs, *, log_path, run_id, n_jobs_ladder, retries_per_level=1,
                before_retry=None, timeout=None) -> 반환값             # 멈추면 TrainingHalted

func_ref = 'package.module:function'(하위 프로세스가 import — 테스트·스크립트 모듈 최상위 함수), kwargs = pickle 되는
작은 값(경로·설정 등). 큰 자료는 하위 프로세스가 직접 읽음(자료를 넘기면 메모리가 두 배).

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 원인 분류: 'memory' = 하위 프로세스 안 MemoryError 또는 메시지에 bad_alloc·out of memory·cannot allocate memory·
     unable to allocate(XGBoost·LightGBM·CatBoost C++ 예외), 'killed' = 신호로 죽음(exitcode < 0 — 리눅스 OOM killer
     SIGKILL 등), 'crash' = 메시지 없이 끝남(exitcode ≠ 0), 'timeout' = timeout 초과(내가 띄운 하위 프로세스만 끔),
     'error' = 그 밖 예외(코드 문제로 봄).
 (2) 다시 돌리기: 'memory'·'killed'·'crash' 만. 같은 n_jobs 로 retries_per_level 번까지 다시(그 전에 before_retry —
     기본 = resources.check_resources(로그 폴더): 가용 RAM 15GB·디스크 10GB 아래면 ResourceFloorError 로 멈춤),
     그래도 멈추면 n_jobs_ladder 다음 값으로. 사다리 끝이면 TrainingHalted(건너뛰지 않음 — 원인을 없앤 뒤 담당자가
     그 판부터 다시). 'error'·'timeout' 은 다시 돌리지 않고 TrainingHalted.
 (3) n_jobs_ladder 는 결과를 바꾸지 않는 값만(트리 = models_tree.retry_n_jobs_ladder). 하위 함수 kwargs 의 'n_jobs' 를
     사다리 값으로 덮어씀(kwargs 에 'n_jobs' 가 없으면 사다리 길이 1 만 허용).
 (4) 실험 로그 = log_path 에 시도마다 JSON 한 줄 덧붙임(run_id·시도 번호·n_jobs·원인·exitcode·예외형·메시지·시각·pid).
 (5) 하위 프로세스 = multiprocessing 'spawn'(리눅스·윈도 같게, 부모 메모리 복사 없음).
"""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import resources as rs

MEMORY_PATTERNS = ("bad_alloc", "out of memory", "cannot allocate memory", "unable to allocate", "memoryerror",
                   "can't allocate", "cannot allocate", "failed to allocate", "std::bad_alloc")  # CatBoost·LightGBM 문구 포함
RETRYABLE = ("memory", "killed", "crash")


class ChildFailure(RuntimeError):
    def __init__(self, cause: str, *, exitcode: int | None, exc_type: str | None = None,
                 message: str = "", tb: str = ""):
        self.cause, self.exitcode, self.exc_type, self.message, self.tb = cause, exitcode, exc_type, message, tb
        super().__init__(f"[{cause}] exitcode={exitcode} {exc_type or ''}: {message[:500]}")


class TrainingHalted(RuntimeError):
    """다시 돌려도 안 됨 — 건너뛰지 않고 멈춤(B-1). attempts = 시도 기록 목록."""

    def __init__(self, msg: str, attempts: list[dict]):
        self.attempts = attempts
        super().__init__(msg)


def _resolve(func_ref: str) -> Callable:
    mod, _, name = func_ref.partition(":")
    if not mod or not name:
        raise ValueError(f"func_ref 는 'module:function': {func_ref!r}")
    obj: Any = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def is_memory_error(e: BaseException) -> bool:
    seen = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, MemoryError):
            return True
        text = f"{type(cur).__name__} {cur}".lower()
        if any(p in text for p in MEMORY_PATTERNS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _child_main(conn, func_ref: str, kwargs: Mapping[str, Any]) -> None:
    try:
        res = _resolve(func_ref)(**kwargs)
        conn.send(("ok", res))
    except BaseException as e:  # noqa: BLE001 — 하위 프로세스는 모든 예외를 부모에게 알림
        try:
            conn.send(("err", f"{type(e).__module__}.{type(e).__qualname__}", str(e)[:4000],
                       traceback.format_exc()[-8000:], is_memory_error(e)))
        except BaseException:  # noqa: BLE001
            pass
        conn.close()
        os._exit(1)
    conn.close()


def run_in_child(func_ref: str, kwargs: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
    """func_ref(**kwargs) 를 spawn 하위 프로세스에서 실행, 반환값을 돌려줌. 실패면 ChildFailure(원인 분류)."""
    _resolve(func_ref)  # 부모에서 먼저 이름 확인(오타면 바로 멈춤)
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_main, args=(send, func_ref, dict(kwargs or {})), daemon=False)
    p.start()
    send.close()
    t0 = time.monotonic()
    msg = None
    try:
        while True:
            if recv.poll(0.2):
                try:
                    msg = recv.recv()
                except EOFError:
                    msg = None
                break
            if not p.is_alive():
                if recv.poll(0):
                    try:
                        msg = recv.recv()
                    except EOFError:
                        msg = None
                break
            if timeout is not None and time.monotonic() - t0 > timeout:
                p.terminate()  # 내가 띄운 하위 프로세스만
                p.join(30)
                raise ChildFailure("timeout", exitcode=p.exitcode, message=f"{timeout}s 초과")
        p.join()
    finally:
        recv.close()
        if p.is_alive():
            p.terminate()
            p.join(30)
    code = p.exitcode
    if msg is not None and msg[0] == "ok":
        if code != 0:
            raise ChildFailure("crash", exitcode=code, message="결과를 보낸 뒤 비정상 종료")
        return msg[1]
    if msg is not None and msg[0] == "err":
        _, et, text, tb, is_mem = msg
        raise ChildFailure("memory" if is_mem else "error", exitcode=code, exc_type=et, message=text, tb=tb)
    if code is not None and code < 0:
        raise ChildFailure("killed", exitcode=code, message=f"신호 {-code} 로 죽음")
    raise ChildFailure("crash", exitcode=code, message="메시지 없이 끝남")


def _log(log_path: Path, rec: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _default_before_retry(log_path: Path) -> Callable[[dict], None]:
    def hook(_rec: dict) -> None:
        rs.check_resources(log_path.parent)  # 바닥값 아래면 ResourceFloorError — 내 작업만 멈춤
    return hook


def run_guarded(func_ref: str, kwargs: Mapping[str, Any], *, log_path: str | os.PathLike, run_id: str,
                n_jobs_ladder: Sequence[int], retries_per_level: int = 1,
                before_retry: Callable[[dict], None] | None = None, timeout: float | None = None) -> Any:
    """B-1 멈춤 처리. 성공하면 반환값, 다시 돌려도 안 되면 TrainingHalted(시도 기록 포함)."""
    log_path = Path(log_path)
    ladder = [rs.check_n_jobs(int(n)) for n in n_jobs_ladder]
    if not ladder or any(b > a for a, b in zip(ladder, ladder[1:])):
        raise ValueError(f"n_jobs_ladder 는 비지 않은 내림차순: {n_jobs_ladder!r}")
    if "n_jobs" not in kwargs and len(ladder) > 1:
        raise ValueError("kwargs 에 n_jobs 가 없으면 n_jobs_ladder 길이 1 만")
    if retries_per_level < 0:
        raise ValueError("retries_per_level ≥ 0")
    hook = before_retry if before_retry is not None else _default_before_retry(log_path)
    attempts: list[dict] = []
    attempt = 0
    for n_jobs in ladder:
        for _ in range(retries_per_level + 1):
            attempt += 1
            kw = dict(kwargs)
            if "n_jobs" in kw:
                kw["n_jobs"] = n_jobs
            rec = {"run_id": run_id, "func_ref": func_ref, "attempt": attempt, "n_jobs": n_jobs,
                   "start": _dt.datetime.now().isoformat(timespec="seconds"), "parent_pid": os.getpid()}
            if attempt > 1:
                hook(rec)
            try:
                res = run_in_child(func_ref, kw, timeout=timeout)
            except ChildFailure as e:
                rec.update(status="failed", cause=e.cause, exitcode=e.exitcode, exc_type=e.exc_type,
                           message=e.message[:2000], traceback=e.tb[-4000:],
                           end=_dt.datetime.now().isoformat(timespec="seconds"))
                _log(log_path, rec)
                attempts.append(rec)
                if e.cause not in RETRYABLE:
                    raise TrainingHalted(f"{run_id}: 다시 돌리지 않는 원인 '{e.cause}' — 원인을 고친 뒤 이 판부터 다시",
                                         attempts) from e
                continue
            rec.update(status="ok", end=_dt.datetime.now().isoformat(timespec="seconds"))
            _log(log_path, rec)
            attempts.append(rec)
            return res
    raise TrainingHalted(f"{run_id}: n_jobs {ladder} 로 다시 돌려도 멈춤 — 메모리를 확보한 뒤 이 판부터 다시", attempts)
