"""guard 시험용 하위 프로세스 함수(spawn 이 import 함)."""
import os


def ok(x, n_jobs=1):
    return {"x": x * 2, "n_jobs": n_jobs, "pid": os.getpid()}


def memory_unless_one(n_jobs):
    if n_jobs > 1:
        raise MemoryError("Unable to allocate 40.0 GiB")
    return n_jobs


def bad_alloc(n_jobs):
    raise RuntimeError("[12:00:00] xgboost/src/c_api.cc: std::bad_alloc")


def crash(n_jobs=1):
    os._exit(3)


def value_error(n_jobs=1):
    raise ValueError("코드 문제")
