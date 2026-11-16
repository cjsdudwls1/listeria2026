"""단계 4 다운사이징(§6 단계 4, 부록 A-6·A-9·B-1·B-2·B-8) 시험 — 합성 자료만."""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lst import blocks as bl
from lst import downsize as ds
from lst import folds as fd
from lst import guard as gd
from lst.hassign import assign_h_many
from lst.models_tree import TreeModel, make_model
from lst.theil import theils_u


def D(s):
    return int(fd.to_days(s)[0])


def make_known(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.integers(D("2015-04-07"), D("2025-06-30") + 1, n)
    inst = rng.integers(0, 20, n)
    y = (rng.random(n) < np.where(inst < 3, 0.15, 0.005)).astype(np.int8)
    keys = np.array([f"K{i:06d}" for i in range(n)], dtype=object)
    known = pd.DataFrame({"lims_data_key": keys, "d": fd.as_date(d), "y": y})
    cols = pd.DataFrame({
        "lims_data_key": keys,
        "hazard_label": "리스테리아",                                          # 한 값뿐 → 제외
        "prduct_nm": [f"제품{i}" for i in range(n)],                          # 고유값 비율 1 → 제외
        "inspct_instt_code": [f"I{v:02d}" for v in inst],
        "addr": [f"주소{v:02d}" for v in inst],                                # 같은 묶음(이름 순 동점)
        "data_source": np.where(inst < 10, "A", "B"),
        "lat": (35.0 + inst / 100).astype(np.float32),
    })
    cols.loc[rng.random(n) < 0.05, "data_source"] = None
    return known, cols


def test_theil_targets_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"columns": ["prduct_nm", "addr"], "note": "x"}), encoding="utf-8")
    assert ds.load_theil_targets(p) == ["prduct_nm", "addr"]
    p.write_text(json.dumps({"columns": ["prduct_nm", "mnfctur_cmpny_nm"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        ds.load_theil_targets(p)
    p.write_text(json.dumps({"columns": ["judge_result"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        ds.load_theil_targets(p)


def test_read_sample_columns(tmp_path):
    t = pa.table({"lims_data_key": ["a", "b", "c"],
                  "judge_result": pa.array(["검출", "불검출", "불검출"]).dictionary_encode(),
                  "addr": pa.array(["x", None, "y"]).dictionary_encode(),
                  "lat": pa.array([1.5, 2.5, None], pa.float32())})
    f = tmp_path / "s.parquet"
    pq.write_table(t, f)
    out = ds.read_sample_columns(f, ["c", "a"], ["addr", "lat"])
    assert list(out["lims_data_key"]) == ["c", "a"] and list(out.columns) == ["lims_data_key", "addr", "lat"]
    assert out["addr"].dtype == object and out["addr"].tolist() == ["y", "x"]
    assert out["lat"].dtype == np.float32
    with pytest.raises(ValueError):
        ds.read_sample_columns(f, ["a", "zz"], ["addr"])
    with pytest.raises(ValueError):
        ds.read_sample_columns(f, ["a"], ["judge_result"])


def test_theil_table_rules():
    known, cols = make_known()
    data = ds.ReductionData.build(known, cols)
    t = ds.theil_table(data.values, data.y, data.columns).set_index("column")
    assert t.loc["hazard_label", "excluded"] == "한 값뿐"
    assert t.loc["prduct_nm", "excluded"] == "고유값 비율 0.5 초과"
    # NaN = 한 범주(고유값 수·U)
    assert t.loc["data_source", "n_unique"] == 3
    ds_filled = cols["data_source"].fillna("<결측>")
    assert t.loc["data_source", "theil_u"] == pytest.approx(theils_u(data.y, ds_filled))
    # 같은 묶음 3칼럼(inspct·addr·lat) 동점 → 고유값 수 같음 → 이름 순: addr
    assert t.loc["addr", "theil_u_r4"] == t.loc["inspct_instt_code", "theil_u_r4"] == t.loc["lat", "theil_u_r4"]
    assert ds.pick_column(t.reset_index()) == "addr"
    # 고유값 수 적은 쪽 우선: 검출 0 값 하나를 둘로 나눈 칼럼(U 같음, 고유값 +1)
    v = data.values.copy()
    zero_val = next(x for x in v["addr"].unique() if data.y[(v["addr"] == x).to_numpy()].sum() == 0)
    v["legal_dong_code"] = v["addr"]
    sel = (v["addr"] == zero_val).to_numpy()
    v.loc[np.flatnonzero(sel)[::2], "legal_dong_code"] = "나눈값"
    t2 = ds.theil_table(v, data.y, ["legal_dong_code", "inspct_instt_code"]).set_index("column")
    assert t2.loc["legal_dong_code", "theil_u_r4"] == t2.loc["inspct_instt_code", "theil_u_r4"]
    assert t2.loc["legal_dong_code", "n_unique"] == t2.loc["inspct_instt_code", "n_unique"] + 1
    assert t2.loc["inspct_instt_code", "rank"] == 1


def test_zero_detection_keep_nan_group():
    v = np.array(["a", "a", None, None, "b", np.nan], dtype=object)
    y = np.array([0, 1, 0, 1, 0, 0])
    keep, vt = ds.zero_detection_keep(v, y)
    assert keep.tolist() == [True, True, True, True, False, True]   # None·NaN = 같은 한 범주(검출 있음)
    assert vt["pos"].sum() == 2


def test_reduce_split_uses_train_only():
    known, cols = make_known()
    # fold1 train 끝 뒤에만 검출이 있는 값 → fold1 에서 빠지고 refit 에선 남음
    d = fd.to_days(known["d"])
    late = np.flatnonzero(d > D("2022-06-30"))[:1]
    k = known.copy()
    c = cols.copy()
    c.loc[late, "addr"] = "늦은값"
    c.loc[late, "inspct_instt_code"] = "늦은값"
    c.loc[late, "lat"] = np.float32(99.0)
    k.loc[late, "y"] = 1
    data = ds.ReductionData.build(k, c)
    reds = ds.plan_reductions(data)
    for s, r in reds.items():
        tr = d <= fd.SPLITS[s].train_end
        assert r.stats["n_before"] == tr.sum() and r.stats["pos_after"] == r.stats["pos_before"]
        assert set(r.kept_keys) <= set(data.keys[tr])
        assert r.table["n_rows"].iloc[0] == tr.sum()
    assert data.keys[late[0]] not in set(reds["fold1"].kept_keys)          # fold1 train 밖 행
    assert data.keys[late[0]] in set(reds["refit"].kept_keys)
    assert ds.column_by_split(reds) == {s: "addr" for s in bl.SPLIT_ORDER}
    # 남긴 행: 칼럼 값이 그 판 train 에서 검출 있는 값만
    r = reds["fold2"]
    tr = np.flatnonzero(d <= fd.SPLITS["fold2"].train_end)
    pos_vals = set(c["addr"].iloc[tr][data.y[tr] == 1])
    kept = r.keep_mask(data.keys)
    assert set(c["addr"][kept]) == pos_vals
    # test 기간 시료가 들어오면 멈춤
    bad = k.copy()
    bad.loc[0, "d"] = np.datetime64("2025-07-02")
    with pytest.raises(ValueError):
        ds.ReductionData.build(bad, c)


def test_block_model_mask_redecides_within_pool():
    known, cols = make_known()
    data = ds.ReductionData.build(known, cols)
    reds = ds.plan_reductions(data)
    plan = bl.block_plan(reuse=False)
    for m in plan:
        mask = ds.block_model_mask(m, data, reds)
        pool = m.train_mask(data.d)
        assert not (mask & ~pool).any()
        col = reds[m.owner.split].column
        pv = set(data.values[col][pool & (data.y == 1)])
        assert set(data.values[col][mask]) == pv          # 그 모델 학습 블록 안 검출 값만
    with pytest.raises(ValueError):
        ds.block_model_mask(bl.block_plan(reuse=True)[3], data, reds)  # fold1-b4 = 재사용 모델


def test_replicate_modes_and_m_formula():
    known, _ = make_known(300)
    rows = fd.make_train_rows(known, fd.SPLITS["refit"])
    a = ds.replicate_h(rows, "4-a")
    assert len(a) == 60 * len(rows)
    g = a.groupby("lims_data_key")["h"]
    assert (g.nunique() == 60).all() and (g.min() == 1).all() and (g.max() == 60).all()
    assert (fd.to_days(a["d"]) - fd.to_days(a["T"]) == a["h"].to_numpy()).all()
    a0 = ds.replicate_h(rows, "4-a0")
    assert (a0["h"].to_numpy() == rows["h"].to_numpy()).all() and len(a0) == len(rows)
    x1 = ds.replicate_h(rows, "x1")
    assert (x1["h"] == 1).all()
    am = ds.replicate_h(rows, "4-a", m=30)
    assert len(am) == 30 * len(rows)
    own = dict(zip(rows["lims_data_key"], rows["h"]))
    for key, sub in list(am.groupby("lims_data_key"))[:20]:
        hs = sub["h"].to_numpy()
        assert own[key] in hs and len(set(hs)) == 30 and (np.diff(hs) > 0).all()
        h = own[key]
        rng = np.random.default_rng(int.from_bytes(hashlib.md5(f'{key}|m42'.encode('utf-8')).digest()[:8], 'big'))
        exp = rng.choice([x for x in range(1, 61) if x != h], 30 - 1, replace=False)
        assert sorted(hs.tolist()) == sorted([h] + exp.tolist())
    with pytest.raises(ValueError):
        ds.replicate_h(rows.assign(h=(rows["h"] % 60) + 1), "4-a")
    with pytest.raises(ValueError):
        ds.replicate_h(rows.assign(E=rows["T"]), "4-a")
    with pytest.raises(ValueError):
        ds.replicate_h(rows, "4-a0", m=30)


def test_counts_largest_and_reduced_rows():
    known, cols = make_known()
    data = ds.ReductionData.build(known, cols)
    reds = ds.plan_reductions(data)
    cnt = ds.replicated_counts(reds)
    assert cnt["refit"] == 60 * reds["refit"].stats["n_after"]
    assert ds.replicated_counts(reds, m=20)["fold1"] == 20 * reds["fold1"].stats["n_after"]
    assert ds.largest_split(reds) == "refit"
    rr = ds.reduced_train_rows(known, "fold3", reds["fold3"])
    assert len(rr) == reds["fold3"].stats["n_after"] and (rr["h"].to_numpy() == assign_h_many(rr["lims_data_key"])).all()


def test_save_and_load_reduction(tmp_path):
    known, cols = make_known(800)
    data = ds.ReductionData.build(known, cols)
    r = ds.reduce_split(data, "fold1")
    ds.save_reduction(r, tmp_path)
    assert list(ds.load_kept_keys(tmp_path, "fold1")) == list(r.kept_keys)
    with pytest.raises(FileExistsError):
        ds.save_reduction(r, tmp_path)


# ---------------------------------------------------------------- 메모리 판정
def test_memory_ladder_order():
    lv = lambda a, f=True: [(s.level, s.float32, s.disk, s.m) for s in ds.memory_ladder(a, f)]
    assert lv("XGB") == [("float32", True, False, None), ("disk", True, True, None), ("m30", True, True, 30),
                         ("m20", True, True, 20), ("m10", True, True, 10)]
    assert lv("CAT") == [("float32", True, False, None), ("m30", True, False, 30), ("m20", True, False, 20),
                         ("m10", True, False, 10)]
    assert lv("LGBM", False)[0] == ("base", False, False, None)


def test_decide_memory_same_process():
    seen = []

    def att(s):
        seen.append(s.level)
        if s.m is None or s.m > 20:
            raise MemoryError("Unable to allocate")
        return "ok"

    dec = ds.decide_memory(att, algo="XGB")
    assert dec.setting.m == 20 and dec.setting.disk and seen == ["float32", "disk", "m30", "m20"]
    assert [a["status"] for a in dec.attempts] == ["memory"] * 3 + ["ok"]
    with pytest.raises(ValueError):
        ds.decide_memory(lambda s: (_ for _ in ()).throw(ValueError("x")), algo="CAT")
    with pytest.raises(ds.MemoryLadderExhausted):
        ds.decide_memory(lambda s: (_ for _ in ()).throw(MemoryError("x")), algo="CAT")


def test_decide_memory_in_child(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.rs, "check_resources", lambda p: {})
    log = tmp_path / "mem.jsonl"
    dec = ds.decide_memory_in_child("_downsize_targets:memory_until_m20", {"tag": "t"}, algo="CAT",
                                    log_path=log, run_id="r1")
    assert dec.setting.level == "m20" and dec.result == {"tag": "t", "level": "m20", "m": 20, "disk": False}
    assert dec.seconds is not None
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["failed", "failed", "ok"] and lines[0]["cause"] == "memory"
    with pytest.raises(gd.TrainingHalted):
        ds.decide_memory_in_child("_downsize_targets:value_error", {"tag": "t"}, algo="XGB", log_path=log, run_id="r2")


def test_combine_and_name():
    mk = lambda algo, s: ds.MemoryDecision(algo, s)
    L = {s.level: s for s in ds.memory_ladder("XGB")}
    C = {s.level: s for s in ds.memory_ladder("CAT")}
    out = ds.combine_settings({"XGB": mk("XGB", L["disk"]), "CAT": mk("CAT", C["m20"]), "LGBM": mk("LGBM", C["float32"])})
    assert out["m"] == 20 and out["components"]["XGB"]["disk"] and not out["components"]["CAT"]["disk"]
    assert ds.combine_settings({"XGB": mk("XGB", L["float32"])})["m"] is None
    assert ds.name_with_m("4-a-CAT", 30) == "4-a-CAT-m30"
    assert ds.name_with_m("5-4-a-CAT-m30", 30) == "5-4-a-CAT-m30"
    assert ds.name_with_m("4-a-CAT", None) == "4-a-CAT"


def test_to_float32():
    X = pd.DataFrame({"a": np.arange(3, dtype=np.float64), "b": np.ones(3, np.float32), "c": ["x", "y", "z"]})
    Y = ds.to_float32(X)
    assert Y["a"].dtype == np.float32 and Y["b"].dtype == np.float32 and Y["c"].dtype == object


# ---------------------------------------------------------------- XGBoost 디스크 기반
def _xy(n=6000, p=12, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"f{i}" for i in range(p)])
    X[X > 2.0] = np.nan
    y = (rng.random(n) < 0.03 + 0.2 * (X["f0"].fillna(0) > 1)).astype(np.int64)
    return X, y


def test_xgb_disk_matches_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.rs, "check_resources", lambda p: {})
    X, y = _xy()
    mem = make_model("XGB", cat_cols=[], max_iter=60, params={"max_depth": 4, "scale_pos_weight": 2.0})
    mem.fit(X, y, seed=42, n_jobs=2)
    tag = ds.disk_cache_tag({"code_md5": "c", "input_md5s": {"a": "b"}, "spec_md5": "s", "split": "refit",
                             "setting": ds.memory_ladder("XGB")[1].to_dict()})
    chunks = ((X.iloc[a:b], y[a:b]) for a, b in [(0, 2000), (2000, 4500), (4500, 6000)])
    disk = ds.XGBDiskModel(max_iter=60, params={"max_depth": 4, "scale_pos_weight": 2.0})
    disk.fit_chunks(chunks, cache_root=tmp_path, tag=tag, seed=42, n_jobs=2)
    assert not (tmp_path / f"xgbext_{tag}").exists()            # 묶음 파일 지움
    assert disk.fit_info_["n_rows"] == 6000 and disk.fit_info_["disk"]["n_chunks"] == 3
    np.testing.assert_allclose(disk.predict_proba(X), mem.predict_proba(X), rtol=0, atol=1e-6)
    np.testing.assert_allclose(disk.predict_proba(X, n_iter=20), mem.predict_proba(X, n_iter=20), rtol=0, atol=1e-6)
    p = disk.save(tmp_path / "m.pkl")
    back = TreeModel.load(p)
    np.testing.assert_array_equal(back.predict_proba(X), disk.predict_proba(X))
    assert back.config_dump()["disk_based"] is True
    with pytest.raises(TypeError):
        ds.XGBDiskModel().fit(X, y, seed=1, n_jobs=1)
    (tmp_path / f"xgbext_{tag}").mkdir()
    with pytest.raises(FileExistsError):
        ds.XGBDiskModel(max_iter=10).fit_chunks([(X, y)], cache_root=tmp_path, tag=tag, seed=1, n_jobs=1)
    with pytest.raises(ValueError):
        ds.disk_cache_tag({"code_md5": "c"})
