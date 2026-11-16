"""멈춤 처리(부록 B-1) 시험 — 하위 프로세스 성공·메모리 예외·비정상 종료·코드 오류 분류, 다시 돌리기·로그."""
import json

import pytest

from lst import guard


def test_child_ok():
    r = guard.run_in_child("_guard_targets:ok", {"x": 21})
    assert r["x"] == 42


def test_child_memory_and_crash_and_error():
    with pytest.raises(guard.ChildFailure) as e:
        guard.run_in_child("_guard_targets:memory_unless_one", {"n_jobs": 4})
    assert e.value.cause == "memory" and e.value.exc_type.endswith("MemoryError")
    with pytest.raises(guard.ChildFailure) as e:
        guard.run_in_child("_guard_targets:bad_alloc", {"n_jobs": 1})
    assert e.value.cause == "memory"
    with pytest.raises(guard.ChildFailure) as e:
        guard.run_in_child("_guard_targets:crash", {})
    assert e.value.cause == "crash" and e.value.exitcode == 3
    with pytest.raises(guard.ChildFailure) as e:
        guard.run_in_child("_guard_targets:value_error", {})
    assert e.value.cause == "error" and "코드 문제" in e.value.message
    with pytest.raises(AttributeError):
        guard.run_in_child("_guard_targets:없는함수", {})


def test_guarded_retry_lower_n_jobs(tmp_path):
    log = tmp_path / "log" / "retry.jsonl"
    seen = []
    r = guard.run_guarded("_guard_targets:memory_unless_one", {"n_jobs": 8}, log_path=log, run_id="t1",
                          n_jobs_ladder=(4, 2, 1), retries_per_level=0, before_retry=seen.append)
    assert r == 1
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["n_jobs"] for x in recs] == [4, 2, 1]
    assert [x.get("cause") for x in recs] == ["memory", "memory", None]
    assert recs[-1]["status"] == "ok" and len(seen) == 2


def test_guarded_halts_not_skips(tmp_path):
    log = tmp_path / "halt.jsonl"
    with pytest.raises(guard.TrainingHalted) as e:
        guard.run_guarded("_guard_targets:memory_unless_one", {"n_jobs": 8}, log_path=log, run_id="t2",
                          n_jobs_ladder=(8,), retries_per_level=1, before_retry=lambda rec: None)
    assert len(e.value.attempts) == 2 and all(a["cause"] == "memory" for a in e.value.attempts)
    with pytest.raises(guard.TrainingHalted) as e:
        guard.run_guarded("_guard_targets:value_error", {"n_jobs": 8}, log_path=log, run_id="t3",
                          n_jobs_ladder=(8, 4), before_retry=lambda rec: None)
    assert len(e.value.attempts) == 1 and e.value.attempts[0]["cause"] == "error"  # 코드 오류는 다시 안 돌림
    with pytest.raises(ValueError):
        guard.run_guarded("_guard_targets:ok", {"x": 1}, log_path=log, run_id="t4", n_jobs_ladder=(4, 2))
