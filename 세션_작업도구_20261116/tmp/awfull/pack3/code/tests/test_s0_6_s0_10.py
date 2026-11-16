"""scripts/s0_6_history.py·s0_10_te.py 연기 시험 — 합성 시료 표로 끝까지 돌려 산출물·test 막기 확인."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst import folds as fd
from lst import history as hi
from tests.test_history import synth

CODE_ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 로컬 PC 는 가용 RAM 이 바닥값(15GB)보다 작을 수 있음 — 합성 자료 시험에선 감시 함수만 바꿈(서버 실행은 그대로)
    mod.check_resources = lambda path: {"available_ram_gb": None, "free_disk_gb": None, "note": "test stub"}
    return mod


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("s0")
    root = base / "data"
    root.mkdir()
    inputs = base / "results"
    p = inputs / "s0" / "0-1" / "apply"
    p.mkdir(parents=True)
    S = synth(30000, seed=11, pos=0.05)
    # test 막기 확인: test 기간(d > 2025-06-30) 정답을 0/1 아닌 값으로 — 읽으면 멈춰야 함
    test = fd.to_days(S["d"]) > fd.SPLITS["refit"].train_end
    S.loc[test, "y"] = 9
    S.to_parquet(p / "samples.parquet", index=False)
    return root, inputs, base, S


def test_s0_6_runs_and_records(roots):
    root, inputs, base, S = roots
    out = base / "out6"
    load("s0_6_history").main(["--root", str(root), "--inputs", str(inputs), "--out", str(out)])
    chk = json.loads((out / "history_check.json").read_text(encoding="utf-8"))
    for name in ("fold1", "fold2", "fold3", "refit"):
        r = chk["splits"][name]
        assert r["train"]["d_gt_C"] and r["train"]["C_le_train_end"]
        assert r["eval"]["d_gt_C"] and r["eval"]["all_pairs_C_eq_train_end"]
        assert r["aggregate_d_le_C"] and r["aggregate_d_le_train_end"]
        assert r["train"]["spot_check"]["max_rel_diff"] < 1e-10
        assert all(r["b6_domestic_sgg_u_counts_without_overseas_same"].values())
        assert r["population_d_max"] <= str(fd.as_date([fd.SPLITS[name].train_end])[0])
    st = pd.read_csv(out / "history_stats.csv", encoding="utf-8-sig")
    assert set(st["part"]) == {"train", "eval_pairs"}
    assert len(pd.read_csv(out / "history_columns.csv", encoding="utf-8-sig")) == 55
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["code_md5"] and list(meta["inputs_md5"]) == ["s0/0-1/apply/samples.parquet"]
    assert not list(out.glob("*.parquet"))  # 큰 행렬 파일 없음
    with pytest.raises(FileExistsError):
        load("s0_6_history").main(["--root", str(root), "--inputs", str(inputs), "--out", str(out)])


def test_s0_10_runs_cv_only(roots):
    root, inputs, base, S = roots
    out = base / "out10"
    load("s0_10_te").main(["--root", str(root), "--inputs", str(inputs), "--out", str(out)])
    grid = pd.read_csv(out / "te7a_grid_cv.csv", encoding="utf-8-sig")
    assert len(grid) == 75 and grid["name"].is_unique
    assert grid["rank"].tolist() == list(range(1, 76))
    r4 = grid["cv_mean_round4"].to_numpy()
    assert (np.diff(r4) <= 0).all()
    sel = json.loads((out / "te7a_selected.json").read_text(encoding="utf-8"))
    assert sel["selected"] == grid["name"].iloc[0] and sel["test_scored"] is False
    assert set(sel["fold_reports"]) == {"fold1", "fold2", "fold3"}
    tp = pd.read_parquet(out / f"te7a_test_pred_{sel['selected']}.parquet")
    assert "y" not in tp.columns and (fd.to_days(tp["d"]) > fd.SPLITS["refit"].train_end).all()
    assert np.isfinite(tp["p_hat"]).all()
    cvp = pd.read_parquet(out / f"te7a_cv_pred_{sel['selected']}.parquet")
    assert "y" not in cvp.columns
    # 고른 칸 fold 1 점수를 다시 계산해 대조(지표 = lst.metrics, 값 = lst.history)
    from lst import metrics as mt
    from lst.hassign import u_tie_many
    sp = fd.SPLITS["fold1"]
    known = S[fd.to_days(S["d"]) <= fd.SPLITS["refit"].train_end].reset_index(drop=True)
    pairs = fd.scoring_pairs(fd.to_days(known["d"]), sp)
    rows, pos = np.unique(pairs["row"].to_numpy(), return_inverse=True)
    ev = known.iloc[rows].copy()
    ev["C"] = fd.as_date(np.full(len(ev), sp.train_end))
    pop = hi.Population.from_samples(known[fd.to_days(known["d"]) <= sp.train_end])
    v = hi.te7a_grid(pop, ev, sel["length"], [sel["k_u"]], [sel["k_p3"]])[(sel["k_u"], sel["k_p3"])]
    y = known["y"].to_numpy()[pairs["row"].to_numpy()]
    u = u_tie_many(known["lims_data_key"].to_numpy()[pairs["row"].to_numpy()], pairs["h"].to_numpy())
    exp = mt.pr_auc_band_weighted(y, v[pos], pairs["h"].to_numpy(), u)
    assert grid.loc[grid["name"] == sel["selected"], "fold1_band_weighted"].iloc[0] == pytest.approx(exp)
    # validation 쌍은 모든 h 같은 값(시료당 한 값)
    assert cvp.groupby(["fold", "lims_data_key"]).size().max() == 1


def test_read_samples_blind_test(roots):
    """test 막기: test 기간(d > 2025-06-30) y 는 읽지 않음(<NA>), 모집단에 넣으면 멈춤."""
    root, inputs, base, S = roots
    df = hi.read_samples_blind_test(inputs / "s0" / "0-1" / "apply" / "samples.parquet",
                                    ("lims_data_key", "y", "src_route", "sgg", "sido", "p2", "p3", "d", "h"))
    test = fd.to_days(df["d"]) > fd.SPLITS["refit"].train_end
    assert test.any() and df.loc[test, "y"].isna().all()
    assert (df.loc[~test, "y"] != 9).all() and df.loc[~test, "y"].notna().all()
    with pytest.raises(ValueError):
        hi.Population.from_samples(df)
    assert len(hi.Population.from_samples(df[~test]).d) == int((~test).sum())
