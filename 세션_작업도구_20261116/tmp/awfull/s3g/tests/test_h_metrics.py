"""h 별 표 시험 — 판 끝날 때 표 = 저장 예측으로 다시 만든 표, test 행은 보고 허가 때만, 내보내기 명령."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import runner as rn
from tests.test_runner import CODE_ROOT, ctx_of, env, spec_of  # noqa: F401  (env = fixture)


def _run(env):
    out = env["inputs"] / "s1" / "1-1"
    s = spec_of()
    rn.run_variant(s, ctx_of(env, out))
    return out, s, out / s.name


def test_h_metrics_at_done_equals_rebuilt(env):
    out, s, v = _run(env)
    t = rn.read_h_metrics(v / rn.H_METRICS_FILE)
    assert len(t) == 180 and set(t["split"]) == set(ev.CV_FOLDS) and list(t.columns) == list(ev.H_METRIC_COLS)
    cv = rn._load_scores(v / "cv_scores.json")
    thr = float(cv["threshold"]["threshold"])
    re = ev.h_metrics_cv(rn.load_preds(v, ev.CV_FOLDS), env["st"], thr)
    pd.testing.assert_frame_equal(t, re.astype(t.dtypes.to_dict()), check_exact=False, rtol=0, atol=1e-12)
    # h 별 PR-AUC = CV 채점의 fold 별 h 값, 혼동행렬 합 = 쌍 수
    for f in ev.CV_FOLDS:
        a = t[t["split"] == f]["pr_auc"].to_numpy()
        b = np.asarray(cv["folds"][f]["pr_auc_by_h"], dtype=float)
        assert np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12)
    assert (t["tp"] + t["fn"] == t["n_pos"]).all() and (t["tp"] + t["fn"] + t["fp"] + t["tn"] == t["n_pairs"]).all()
    assert (t["threshold"] == thr).all()
    with pytest.raises(ev.TestPeekError):
        ev.h_metrics_test(pd.read_parquet(v / "pred_refit.parquet"), env["st"], type("T", (), {"permit": None})(), thr)


def test_h_metrics_test_rows_only_with_permit(env):
    out, s, v = _run(env)
    (v / rn.H_METRICS_FILE).unlink()          # 이전 코드로 끝난 판(표 없음) 흉내 — 보고 때 저장 예측으로 만듦
    permit = ev.report_permit("1-1", "시험 보고")
    rn.report_stage(out_dir=out, stage="1-1", names=[s.name], samples=env["st"], samples_path=env["samples_path"],
                    permit=permit, incomplete={("overseas", "2026-06")}, exp_log=env["log"])
    t = rn.read_h_metrics(v / rn.H_METRICS_FILE)
    assert len(t) == 240 and (t["split"] == "test").sum() == 60
    ts = rn._load_scores(v / "test_scores.json")
    a = t[t["split"] == "test"]["pr_auc"].to_numpy()
    assert np.allclose(a, np.asarray(ts["report"]["pr_auc_by_h"], float), equal_nan=True, rtol=0, atol=1e-12)
    assert t[t["split"] == "test"]["n_pairs"].sum() == ts["report"]["n_pairs"]   # 덜 찬 달 쌍 뺌


def _export():
    spec = importlib.util.spec_from_file_location("export_h_metrics", CODE_ROOT / "scripts" / "export_h_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_export_script(env, tmp_path):
    out, s, v = _run(env)
    m = _export()
    base = ["--root", str(tmp_path), "--inputs", str(env["inputs"]), "--stage", "1-1", "--out"]
    p = m.main(base + [str(tmp_path / "e1"), "--only", s.name])[0]
    long = pd.read_excel(p, sheet_name="long")
    wide = pd.read_excel(p, sheet_name="pr_auc_wide")
    assert set(long["split"]) == set(ev.CV_FOLDS) and len(long) == 180
    assert len(wide) == 3 and "h60" in wide.columns
    with pytest.raises(SystemExit):
        m.main(base + [str(tmp_path / "e2"), "--with-test"])           # 이유 없이 test 안 됨
    p2 = m.main(base + [str(tmp_path / "e3"), "--with-test", "--reason", "시험"])[0]
    long2 = pd.read_excel(p2, sheet_name="long")
    assert (long2["split"] == "test").sum() == 60
    seen = [e for e in env["log"].entries() if e["event"] == "test_viewed"]
    assert len(seen) == 1 and seen[0]["names"] == [s.name] and seen[0]["reason"] == "시험"
    with pytest.raises(FileExistsError):
        m.main(base + [str(tmp_path / "e1"), "--only", s.name])
