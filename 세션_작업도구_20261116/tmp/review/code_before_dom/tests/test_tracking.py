"""lst/tracking.py 시험 — 로컬 임시 sqlite 저장소로 run 계층·tag·판정 한 번·중단 run·사본(sqlite backup)."""
import os
import sqlite3

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from lst import evaluate as ev  # noqa: E402
from lst import tracking as tk  # noqa: E402


def make_tracker(tmp_path):
    return tk.Tracker(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", artifact_root=(tmp_path / "artifacts").as_uri())


@pytest.fixture()
def tracker(tmp_path):
    return make_tracker(tmp_path)


def test_hierarchy_tags(tracker, tmp_path):
    sr = tracker.stage_run("listeria-1차", "1-1")
    assert tracker.stage_run("listeria-1차", "1-1") == sr          # 있으면 그대로
    rid = tracker.start_variant("listeria-1차", sr, "1-CAT-W060-sum", {"단계": "1-1", "역할": "후보"},
                                {"algorithm": "CAT", "W": 60, "model_params": {}})
    tracker.set_judgement(rid, "해당 없음")
    with pytest.raises(tk.TrackingError):
        tracker.set_judgement(rid, "채택")                          # 판정은 한 번만
    with pytest.raises(tk.TrackingError):
        tracker.set_tags(rid, {"없는 tag": "x"})
    with pytest.raises(tk.TrackingError):
        tracker.experiment_id("listeria-기타")
    tracker.log_metrics(rid, {"cv_band_weighted": 0.05, "nan_metric": float("nan")})
    tracker.log_series(rid, "fold1_pr_auc_h", [0.1] * 59 + [float("nan")])
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n", encoding="utf-8")
    tracker.log_artifact(rid, f, "tables")
    tracker.finish(rid)
    r = tracker.client.get_run(rid)
    assert r.data.tags["판 이름"] == "1-CAT-W060-sum" and r.data.tags["판정"] == "해당 없음"
    assert r.data.tags["mlflow.parentRunId"] == sr and r.data.tags["차수"] == "1차"
    assert r.data.params["W"] == "60"
    assert [a.path for a in tracker.client.list_artifacts(rid, "tables")] == ["tables/a.csv"]
    with pytest.raises(tk.TrackingError):                           # 끝난 판은 다시 안 만듦
        tracker.start_variant("listeria-1차", sr, "1-CAT-W060-sum", {"역할": "후보"})
    # 멈춘 판: 끝나지 않은 run 은 '중단됨' 으로 돌리고 새로
    r1 = tracker.start_variant("listeria-1차", sr, "1-XGB-W010-raw", {"역할": "후보"})
    r2 = tracker.start_variant("listeria-1차", sr, "1-XGB-W010-raw", {"역할": "후보"})
    old = tracker.client.get_run(r1)
    assert old.data.tags["중단됨"] and old.info.status == "FAILED" and old.info.run_name.endswith("-중단1")
    assert tracker.find_run("listeria-1차", "1-XGB-W010-raw", sr).info.run_id == r2


def test_test_metrics_need_permit(tracker):
    sr = tracker.stage_run("listeria-1차", "1-1")
    rid = tracker.start_variant("listeria-1차", sr, "v", {"역할": "후보"})
    with pytest.raises(ev.TestPeekError):
        tracker.log_test(rid, {}, permit=None)


def test_snapshot_and_local_copy(tracker, tmp_path):
    sr = tracker.stage_run("listeria-준비", "0-10")
    rid = tracker.start_variant("listeria-준비", sr, "7-a-5y-ku050-kp050", {"역할": "기준 줄"})
    tracker.log_metrics(rid, {"cv_band_weighted": 0.04})
    f = tmp_path / "g.csv"
    f.write_text("a\n", encoding="utf-8")
    tracker.log_artifact(rid, f)
    tracker.finish(rid)
    # 서버 쪽 폴더 꼴로: mlflow.db + artifacts
    snap = tmp_path / "snap"
    info = tk.snapshot_store(tmp_path, snap)
    assert info["n_artifact_files"] >= 1
    with pytest.raises(FileExistsError):
        tk.snapshot_store(tmp_path, snap)
    local = tmp_path / "local_mlflow"
    tk.install_local_copy(snap, local)
    con = sqlite3.connect(local / "mlflow.db")
    assert con.execute("select count(*) from runs").fetchone()[0] == 2
    con.close()
    t2 = tk.Tracker(f"sqlite:///{(local / 'mlflow.db').as_posix()}")
    r = t2.client.get_run(rid)
    assert r.data.metrics["cv_band_weighted"] == 0.04 and r.data.tags["역할"] == "기준 줄"
    assert (local / "artifacts").is_dir()
    cmd = tk.ui_command(local)
    assert cmd[1] == "ui" and "--artifacts-destination" in cmd and cmd[cmd.index("--port") + 1] == "5000"


def test_stop_server_refuses_foreign_pid(tmp_path):
    import json
    import os as _os

    pf = tmp_path / "server_5055.pid.json"
    pf.write_text(json.dumps({"pid": _os.getpid(), "start_time": "다른값", "cmd": []}), encoding="utf-8")
    with pytest.raises(tk.TrackingError):
        tk.stop_server(pf)   # 시작 시각이 다르면(남의 프로세스) 끄지 않음
    assert pf.exists()
