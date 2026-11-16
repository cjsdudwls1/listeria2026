"""설정·경로(부록 B-2)·자원(부록 B-1) 시험."""
import json
import os

import pytest

from lst import config as cf
from lst import paths as ph
from lst import resources as rs


# ---------------------------------------------------------------- paths
def test_to_rel_and_resolve(tmp_path):
    assert ph.to_rel("리스테리아ML\\data\\listeria_data.parquet").as_posix() == "리스테리아ML/data/listeria_data.parquet"
    for bad in ["C:\\x\\y", "C:x", "/home/a", "\\\\srv\\share", "~/a", "a/../../b"]:
        with pytest.raises(ValueError):
            ph.to_rel(bad)
    assert ph.resolve(tmp_path, "a\\b.txt") == tmp_path / "a" / "b.txt"


def test_parse_paths_and_unset_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LST_ROOT", "x")
    monkeypatch.setenv("PYTHONPATH", "keep")
    rp, _ = ph.parse_paths(["--root", str(tmp_path), "--out", str(tmp_path / "out")])
    assert "LST_ROOT" not in os.environ and os.environ["PYTHONPATH"] == "keep"
    assert rp.out.is_dir() and rp.data("a/b") == tmp_path.resolve() / "a" / "b"
    with pytest.raises(SystemExit):
        ph.parse_paths(["--root", str(tmp_path)])  # --out 필수
    with pytest.raises(FileNotFoundError):
        ph.parse_paths(["--root", str(tmp_path / "없음"), "--out", str(tmp_path)])


# ---------------------------------------------------------------- config
SCHEMA = {
    "W": cf.Field(int, choices=tuple(range(10, 101, 10))),
    "mode": cf.Field(str, choices=("sum", "raw")),
    "inputs": cf.Field(list, item_types=str),
    "opt": cf.Field(float, required=False),
    "sub": {"k": cf.Field(int)},
}


def _write(tmp_path, text):
    p = tmp_path / "c.json"
    p.write_text(text, encoding="utf-8")
    return p


def test_config_ok(tmp_path):
    p = _write(tmp_path, json.dumps({"W": 60, "mode": "raw", "inputs": ["a/b.parquet"], "sub": {"k": 3}}))
    cfg = cf.load_config(p, SCHEMA)
    assert cfg["W"] == 60 and "opt" not in cfg  # 기본값을 채워 넣지 않음


@pytest.mark.parametrize("text", [
    '{"W": 60, "mode": "raw", "inputs": [], "sub": {"k": 3}, "extra": 1}',      # 모르는 칸
    '{"W": 60, "mode": "raw", "inputs": [], "sub": {"k": 3, "z": 1}}',          # 하위 모르는 칸
    '{"W": 60, "W": 70, "mode": "raw", "inputs": [], "sub": {"k": 3}}',         # 같은 칸 두 번
    '{"W": 65, "mode": "raw", "inputs": [], "sub": {"k": 3}}',                  # 허용 값 아님
    '{"W": true, "mode": "raw", "inputs": [], "sub": {"k": 3}}',                # bool ≠ int
    '{"W": 60, "mode": "raw", "inputs": [1], "sub": {"k": 3}}',                 # 원소 타입
    '{"W": 60, "mode": "raw", "inputs": [], "sub": {"k": 3}, "opt": NaN}',      # NaN
    '{"mode": "raw", "inputs": [], "sub": {"k": 3}}',                           # 필수 칸 없음
    '{"W": 60, "mode": "raw", "inputs": ["C:\\\\data\\\\x.parquet"], "sub": {"k": 3}}',  # 절대 경로
])
def test_config_errors(tmp_path, text):
    with pytest.raises(cf.ConfigError):
        cf.load_config(_write(tmp_path, text), SCHEMA)


def test_check_columns():
    cf.check_columns(["a", "b"], ["a", "b", "c"])
    with pytest.raises(cf.ConfigError):
        cf.check_columns(["a", "zz"], ["a"])


def test_code_md5_line_endings_and_meta(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, nl in ((a, b"\n"), (b, b"\r\n")):
        (root / "lst").mkdir(parents=True)
        (root / "lst" / "m.py").write_bytes(b"x = 1" + nl + b"y = 2" + nl)
    assert cf.code_md5(a) == cf.code_md5(b)
    (b / "lst" / "m.py").write_bytes(b"x = 3\n")
    assert cf.code_md5(a) != cf.code_md5(b)
    (a / "in.txt").write_text("hello", encoding="utf-8")
    md5s = cf.inputs_md5(a, ["in.txt"])
    assert md5s == {"in.txt": "5d41402abc4b2a76b9719d911017c592"}
    out = cf.save_run_meta(tmp_path / "res" / "meta.json", {"W": 60}, md5s, a, n_jobs=8)
    meta = json.loads(out.read_text(encoding="utf-8"))
    assert {"code_md5", "inputs_md5", "threads", "config"} <= set(meta)
    assert meta["threads"]["n_jobs"] == 8
    with pytest.raises(FileExistsError):
        cf.save_run_meta(out, {"W": 60}, md5s, a)


# ---------------------------------------------------------------- resources
def test_thread_env(monkeypatch):
    for v in rs.THREAD_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    with pytest.warns(RuntimeWarning):  # numpy 는 이미 불러옴
        env = rs.set_thread_env()
    assert all(os.environ[v] == "8" for v in rs.THREAD_ENV_VARS) and len(env) == 4
    with pytest.raises(ValueError):
        rs.set_thread_env(9)


def test_n_jobs():
    assert rs.check_n_jobs(8) == 8
    for bad in (-1, 0, 9, None, True, 2.0):
        with pytest.raises(ValueError):
            rs.check_n_jobs(bad)


def test_check_resources(monkeypatch, tmp_path):
    info = rs.check_resources(tmp_path) if rs.available_ram_bytes() >= 15 * rs.GB and \
        rs.free_disk_bytes(tmp_path) >= 10 * rs.GB else None
    if info is not None:
        assert info["available_ram_gb"] >= 15
    monkeypatch.setattr(rs, "available_ram_bytes", lambda: 14 * rs.GB)
    monkeypatch.setattr(rs, "free_disk_bytes", lambda p: 100 * rs.GB)
    with pytest.raises(rs.ResourceFloorError, match="RAM"):
        rs.check_resources(tmp_path)
    monkeypatch.setattr(rs, "available_ram_bytes", lambda: 50 * rs.GB)
    monkeypatch.setattr(rs, "free_disk_bytes", lambda p: 9 * rs.GB)
    with pytest.raises(rs.ResourceFloorError, match="디스크"):
        rs.check_resources(tmp_path)
    monkeypatch.setattr(rs, "free_disk_bytes", lambda p: 10 * rs.GB)
    assert rs.check_resources(tmp_path)["free_disk_gb"] == 10


def test_gpu_fraction_limit():
    with pytest.raises(ValueError):
        rs.limit_gpu_memory(0.5)
    pytest.importorskip("torch")
    assert rs.limit_gpu_memory(0.4) in (True, False)


def test_unset_mlflow_path_env(monkeypatch):
    for v in ph.PATH_ENV_NAMES:
        monkeypatch.setenv(v, "file:///x")
    removed = ph.unset_path_env()
    assert set(ph.PATH_ENV_NAMES) <= set(removed)
    assert not any(v in os.environ for v in ph.PATH_ENV_NAMES)
    assert set(ph.PATH_ENV_NAMES) == {"MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_URI", "MLFLOW_REGISTRY_URI"}


def test_dump_scores_json_nan_to_null(tmp_path):
    import numpy as np
    from lst import metrics as mt
    rng = np.random.default_rng(0)
    n = 600
    h = rng.integers(1, 61, n)
    y = (rng.random(n) < 0.2).astype(int)
    y[h == 7] = 0  # h 7 PR-AUC NaN
    rep = mt.score_report(y, rng.random(n), h, rng.random(n))
    with pytest.raises(ValueError):
        cf.dump_json(tmp_path / "raw.json", rep)  # 그대로는 NaN 이라 멈춤
    assert not list(tmp_path.glob("*.tmp"))  # 실패하면 임시 파일 안 남김
    p = cf.dump_scores_json(tmp_path / "s.json", {**rep, "arr": np.array([1.0, np.nan]), "x": np.float64("nan")})
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["pr_auc_by_h"][6] is None and back["arr"] == [1.0, None] and back["x"] is None
    assert back["pr_auc_band_weighted"] == pytest.approx(rep["pr_auc_band_weighted"])
    with pytest.raises(ValueError):
        cf.dump_scores_json(tmp_path / "inf.json", {"a": float("inf")})  # inf 는 그대로 멈춤
    with pytest.raises(cf.ConfigError):
        cf.load_json_strict(_write(tmp_path, '{"a": NaN}'))  # 엄격 읽기 규칙 그대로
