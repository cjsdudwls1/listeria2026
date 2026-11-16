"""scripts/s0_9_timing.py 시험 — 가짜 시간으로 측정 틀이 끝까지 도는지·이어하기·횟수 × 시간 셈(부록 A-9)."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from lst import blocks as bl
from tests._s09_fake import FAKES, REFUSE
from tests.test_history import synth

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "tests"))


def load():
    spec = importlib.util.spec_from_file_location("s0_9_timing", CODE_ROOT / "scripts" / "s0_9_timing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()
STUB = lambda path: {"available_ram_gb": None, "free_disk_gb": None, "note": "test stub"}  # noqa: E731 — 로컬 RAM 바닥값


@pytest.fixture()
def env(tmp_path):
    root, inputs, out = tmp_path / "data", tmp_path / "results", tmp_path / "results" / "s0" / "0-9"
    (inputs / "s0" / "0-1" / "apply").mkdir(parents=True)
    (root / "x").mkdir(parents=True)
    synth(4000, seed=3, pos=0.05).to_parquet(inputs / "s0" / "0-1" / "apply" / "samples.parquet", index=False)
    (inputs / "s0" / "0-1" / "apply" / "14_theil_target_columns.json").write_text('{"columns": ["prduct_nm"]}',
                                                                                 encoding="utf-8")
    (root / "x" / "sample.parquet").write_bytes(b"dummy")
    cfg = json.loads((CODE_ROOT / "configs" / "s0_9_timing.json").read_text(encoding="utf-8"))
    cfg.update(sample_file="x/sample.parquet", use_mlflow=False,
               context={"results": {"samples": "s0/0-1/apply/samples.parquet"}, "data": {}, "y_cut": "2025-06-30"})
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    argv = ["--root", str(root), "--inputs", str(inputs), "--out", str(out), "--config", str(p)]
    return argv, out, inputs


def calls(out):
    f = out / "calls.txt"
    return f.read_text(encoding="utf-8").split() if f.exists() else []


def test_config_fixed_ak_mapping():
    from lst import tuning as tu

    cfg = M.load_cfg(CODE_ROOT / "configs" / "s0_9_timing.json")
    assert cfg["autokeras"]["mapping"] == json.loads(json.dumps(tu.AK_PROPOSAL))   # 담당자 결정 11-07
    assert "attention" in cfg["autokeras"]["approx_note"] and "dilation" in cfg["autokeras"]["approx_note"]


def test_full_run_fake_times(env):
    argv, out, inputs = env
    sd = M.main(argv, overrides=FAKES, isolate=False, check_resources=STUB)
    c = calls(out)
    assert c[:12] == [f"1-1-{a}-W{W:03d}-{m}" for a in M.TREE1 for W in (10, 100) for m in ("sum", "raw")]
    assert c[12:14] == ["1-2-LSTM-W010", "1-2-LSTM-W100"]
    assert c[14] == "chain-CAT-it02000"                       # 가장 느린 트리(가짜 시간 CAT)가 먼저
    assert "chain-XGB-it04000" in c and "chain-XGB-it08000" not in c   # 마지막 시점이면 2배 한 번
    assert c.index("chain-LGB-it02000") < c.index("imp-tree-CAT") < c.index("optuna-CAT")
    assert c[-1] == "4a-CAT"
    mx = pd.read_csv(sd / "max_iter_decision.csv")
    assert set(mx.loc[mx["algo"] == "XGB", "max_iter"]) == {2000, 4000}
    assert (mx.loc[mx["algo"] == "XGB", "decided_max_iter"] == 4000).all()
    ak = pd.read_csv(sd / "ak_check.csv").set_index("label")
    assert ak.loc["TRF", "status"].startswith("대응 없음") and ak.loc["CMBV", "status"].startswith("대응 없음")
    assert ak.loc["x7-D", "status"].startswith("대응 없음")
    assert ak.loc["x6-GRUA", "status"] == "대응 확인 실패" and ak.loc["LSTM", "status"] == "대응 확인"
    est = json.loads((sd / "estimate.json").read_text(encoding="utf-8"))
    assert est["count_total"] == M.A9_TOTAL and est["count_matches_A9"]
    assert est["total_seconds"] > 0 and est["expected_end"] and not est["nan_rows"]
    assert (sd / "run_meta.json").exists() and (sd / "timing_table.csv").exists()
    log = [json.loads(x) for x in (inputs / "logs" / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()]
    fixed = [e for e in log if e["event"] == "ak_mapping_fixed"]
    assert len(fixed) == 1 and fixed[0]["same_as_AK_PROPOSAL"] and fixed[0]["approx_note"]


def test_resume_and_stop(env):
    argv, out, inputs = env
    (out).mkdir(parents=True, exist_ok=True)
    (out / "FAIL_dl-TRF").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="가짜 실패"):
        M.main(argv, overrides=FAKES, isolate=False, check_resources=STUB)
    first = calls(out)
    assert first[-1] == "dl-TRF" and not (out / "items" / "dl-TRF" / "done.json").exists()
    (out / "FAIL_dl-TRF").unlink()
    M.main(argv, overrides=FAKES, isolate=False, check_resources=STUB)
    second = calls(out)[len(first):]
    assert second[0] == "dl-TRF" and set(second).isdisjoint(set(first[:-1]))   # 끝난 항목은 다시 안 돎
    assert list((out / "_incomplete").iterdir())[0].name.startswith("dl-TRF__")   # 멈춘 항목 폴더는 옮김
    sd = M.main(argv + ["--summary-only"], overrides=REFUSE, isolate=False, check_resources=STUB)
    assert (sd / "estimate_table.csv").exists()
    ev = [json.loads(x)["event"] for x in (inputs / "logs" / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "rerun" in ev and "stopped" in ev and ev.count("ak_mapping_fixed") == 1


def test_child_process_measures_rss(env, tmp_path):
    argv, out, inputs = env
    cfg = M.load_cfg(Path(argv[-1]))
    T = M.Timing(cfg, Path(argv[1]), Path(argv[3]), out, overrides=FAKES, isolate=True, check_resources=STUB)
    r = T.item("dl-LSTM", "dl_fold", "cpu", {"arch": "LSTM", "version": M.v_proxy_dl(), "save_model": False,
                                             "components": []})
    assert r["seconds"] == 7.0 and r["wall_seconds"] >= 0 and r["pid"] != __import__("os").getpid()
    assert r["peak_rss_gb"] is None or r["peak_rss_gb"] > 0


class _AllOne(dict):
    """모든 측정 항목 1회 시간 = 1초(가짜)."""

    def get(self, k, d=None):
        return {"status": "ok", "seconds": 1.0, "train_rows": 1000, "seconds_per_target": 1.0, "seconds_per_pair": 1.0,
                "largest_split": "refit", "reduced_rows": {s: 1000 for s in M.rn.SPLIT_ORDER},
                "replicated_rows": {s: 1000 for s in M.rn.SPLIT_ORDER}, "setting": {"m": None}}


def uniform_info():
    """모든 비(행 수·칼럼 수·쌍 수)가 1 이 되게 만든 자료 정보."""
    ci = M.column_info()
    one = 100
    cols = {"order1": {k: one for k in ci["order1"]}, "proxy": one, "2-1": one, "2-1b": one, "2-R": [one] * 10,
            "seq_tree": one, "lims": one, "meta": one, "dl_order1_size": {k: one for k in ci["dl_order1_size"]},
            "dl_C_order1": one, "dl_C_2-1b": one, "targets": ci["targets"]}
    tune = ((bl.DATA_START, M.fd.day("2021-06-30")),)
    blocks = [{"name": m.name, "rows": 1000, "full": m.is_split_full, "tune": m.train_intervals == tune, "role_pairs": 1}
              for m in bl.block_plan(reuse=True)]
    s = M.rn.SPLIT_ORDER
    return {"rows": {x: 1000 for x in s}, "pos": {x: 200 for x in s}, "pairs": {x: 100 for x in s}, "n_known": 1500,
            "blocks": blocks, "cols": cols}


def test_count_times_seconds_matches_A9():
    chain = {a: {"item": f"chain-{a}-it02000"} for a in M.TREE1}
    est = M.estimate(_AllOne(), chain, uniform_info(), proxy_arch="LSTM", ak_arch="LSTM")
    plan = dict(M.A9_COUNTS)
    assert [r["단계"] for r in est["rows"]] == [k for k, _ in M.A9_COUNTS]
    for r in est["rows"]:
        assert r["셈한_횟수"] == plan[r["단계"]] == r["A9_횟수"]
        assert r["예상_초"] == pytest.approx(plan[r["단계"]])          # 1초 × 횟수
    assert est["count_total"] == 15415 == sum(plan.values()) and est["count_matches_A9"]
    assert sum(plan[k] for k in plan if k != "3-0b 변수 빼기") == 1531 and plan["3-0b 변수 빼기"] == 13 * 1068
    # 병렬(1-1 CPU·1-2 GPU)은 긴 쪽만
    assert est["train_seconds"] == pytest.approx(15415 - 40)
    nt = {r["단계"]: r["예상_초"] for r in est["non_training"]}
    assert nt["3-0b 중요도(학습 아님)"] == pytest.approx(13 * (3 * 44 + 3 * 74) * 3)   # fold 1~3 쌍 비 = 3
    assert nt["2-R1 중요도(학습 아님)"] == pytest.approx(15 * 3)
    assert nt["딥러닝 추론(학습 아님)"] == pytest.approx(3 * 15 + 4 * 19 + 3 * 19 + 3 * 18)


def test_unmeasured_W_column_ratio():
    info = uniform_info()
    info["cols"]["order1"]["W060|raw"] = 300            # 가까운 측정 W = 100(같은 거리면 큰 W 아님 — 60 은 100 이 가까움)
    U = M.Units(_AllOne(), {}, info, "LSTM", "LSTM")
    assert U.t11("XGB", 60, "raw") == pytest.approx(3.0)
    assert U._near(50) == 10 and U._near(60) == 100 and U._near(55) == 100


def test_mlflow_runs(env, tmp_path):
    argv, out, inputs = env
    cfgp = Path(argv[-1])
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg.update(use_mlflow=True, mlflow_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    cfgp.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    M.main(argv, overrides=FAKES, isolate=False, check_resources=STUB)
    from lst.stage_cli import make_tracker

    tr = make_tracker(M.load_cfg(cfgp))
    sr = tr.find_run("listeria-준비", "0-9", None)
    r = tr.find_run("listeria-준비", "0-9-1-1-XGB-W010-sum", sr.info.run_id)
    assert r.info.status == "FINISHED" and r.data.metrics["seconds"] == 3.0
    assert tr.find_run("listeria-준비", "0-9-ak-check-x6-GRUA", sr.info.run_id).info.status == "FAILED"
    M.main(argv, overrides=REFUSE, isolate=False, check_resources=STUB)       # 다시 돌려도 run 이 겹치지 않음


def test_top3_dl_by_speed_and_block_ratio():
    """구조마다 1 fold 시간이 다르면 상위 3 = 블록 학습 예상이 느린 순, 블록 예상 = 블록 측정 × 구조 1 fold 비."""
    times = {"LSTM": 2.0, "GRU": 9.0, "TRF": 7.0, "LSTMA": 1.0, "TCN": 8.0, "CMB": 3.0, "CMBV": 4.0}

    class R(_AllOne):
        def get(self, k, d=None):
            r = dict(super().get(k, d))
            if k.startswith("dl-"):
                r["seconds"] = times[k[3:]]
            return r

    U = M.Units(R(), {}, uniform_info(), "LSTM", "LSTM")
    assert U.top3_dl() == ["GRU", "TCN", "TRF"]
    assert U.blk("GRU", 1000) == pytest.approx(1.0 * 9.0 / 2.0)
    assert U.blk("LSTMA", 500) == pytest.approx(1.0 * 0.5 * 1.0 / 2.0)
    assert U.blk("LSTM", 1000) == pytest.approx(1.0)
