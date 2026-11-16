"""식품 고정표 v2(aw 출처 거르기 칸·보충 규칙·출처 세기·결측 목록 대조) · 5-LSTM 다시 학습(rerun — 같은 설정·새 이름·새
단계 run) · 최종 판 바꿔 끼우기(final_override) · 추가 판 KO1(Koutsoumanis 생육확률) 시험 — 합성 자료(2026-11-12)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from lst import config as cf  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import inputs as ip  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage4 as s4  # noqa: E402
from lst import static_feats as sf  # noqa: E402
from lst import windows as wn  # noqa: E402
from lst import wxfeat as wf  # noqa: E402

CODE = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", CODE / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"_t_{name}"] = m
    spec.loader.exec_module(m)
    return m


# ================================================================ 1. 식품 고정표
def _cl():
    # A01(2단계) ← A0101·A0102·A0103(3단계) ← A010101(4단계, A0101 아래)·A010201(A0102 아래)
    rows = [("A0100000000000", None), ("A0101000000000", "A0100000000000"), ("A0102000000000", "A0100000000000"),
            ("A0103000000000", "A0100000000000"), ("A0101010000000", "A0101000000000"),
            ("A0102010000000", "A0102000000000"), ("A0104000000000", "A0100000000000")]
    return pd.DataFrame(rows, columns=["prdlst_cd", "hrnk_prdlst_cd"])


def _aw():
    return pd.DataFrame({
        "prdlst_cd": ["A0100000000000", "A0101000000000", "A0101010000000", "A0102010000000", "A0103000000000"],
        "aw": ["0.97", "0.95~0.99", "0.90", "0.85", "-"],
        "ph": ["6.0", None, "5.5", None, "< 4.6"],
        "na": [None, "1.2", None, "2.0", "-0.85"],
        "aw_source": ["original", "xlsx_20261111", "original", "xlsx_20261111", "xlsx_20261111"]})


CODES3 = ["A0101000000000", "A0102000000000", "A0103000000000", "A0104000000000"]
P2OF = {k: "A0100000000000" for k in CODES3}


def test_food_filter_default_original_and_all():
    old = sf.build_food_table(_cl(), _aw(), CODES3, P2OF).table.set_index("p3")
    same = sf.build_food_table(_cl(), _aw(), CODES3, P2OF, aw_source_filter="original").table.set_index("p3")
    pd.testing.assert_frame_equal(old, same)
    new = sf.build_food_table(_cl(), _aw(), CODES3, P2OF, aw_source_filter="all").table.set_index("p3")
    # original 만: A0101 자기 aw(xlsx)는 안 씀 → 하위 A010101(0.90, original) 중앙값
    assert old.loc["A0101000000000", "aw_rule"] == "하위중앙" and old.loc["A0101000000000", "food_aw"] == 0.90
    # 전부: A0101 자기 aw 범위 → 가운데 값 0.97(표기 중앙)
    assert new.loc["A0101000000000", "aw_rule"] == "자기값" and np.isclose(new.loc["A0101000000000", "food_aw"], 0.97)
    assert new.loc["A0101000000000", "aw_label"] == "중앙"
    # A0102: 자기 aw 없음 → 하위 A010201(0.85 xlsx): 전부면 하위중앙, original 만이면 2단계값 0.97
    assert new.loc["A0102000000000", "aw_rule"] == "하위중앙" and new.loc["A0102000000000", "food_aw"] == 0.85
    assert old.loc["A0102000000000", "aw_rule"] == "2단계값" and old.loc["A0102000000000", "food_aw"] == 0.97
    # A0103: 자기 aw '-' = 해석 못 함 → 하위 없음 → 2단계값(규칙 그대로)
    assert new.loc["A0103000000000", "aw_rule"] == "2단계값"
    # A0104: 표에 코드 없음 → 2단계값(aw)·결측(na — 2단계 na 없음)
    assert new.loc["A0104000000000", "aw_rule"] == "2단계값" and new.loc["A0104000000000", "na_rule"] == "결측"
    # ph·na 는 거르기와 무관(두 판 같음), 부등호·음수 원문 규칙 그대로
    for c in ("ph", "na"):
        pd.testing.assert_series_equal(old[f"food_{c}"], new[f"food_{c}"])
    assert new.loc["A0103000000000", "food_ph"] == 4.6 and new.loc["A0103000000000", "ph_label"] == "미만"
    assert new.loc["A0103000000000", "na_rule"] == "2단계값" or new.loc["A0103000000000", "na_rule"] == "결측"
    with pytest.raises(ValueError):
        sf.build_food_table(_cl(), _aw(), CODES3, P2OF, aw_source_filter="xlsx")


def test_static_config_field_and_counts(tmp_path):
    m = _load("s0_6_static")
    old = m.load_cfg(CODE / "configs" / "s0_6_static.json")
    assert "aw_source_filter" not in old and "missing_check" not in old
    new = m.load_cfg(CODE / "configs" / "s0_6_static_food2.json")
    assert new["aw_source_filter"] == "all" and new["inputs"]["aw_cl"].endswith("추출_20261112/t_aw_cl.parquet")
    raw = json.loads((CODE / "configs" / "s0_6_static_food2.json").read_text(encoding="utf-8"))
    raw["aw_source_filter"] = "some"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(cf.ConfigError):
        m.load_cfg(p)
    t = sf.build_food_table(_cl(), _aw(), CODES3, P2OF, aw_source_filter="all").table
    sc = m.source_counts(t)
    assert len(sc) == 12 and (sc.groupby("항목")["3단계 코드 수"].sum() == 4).all()
    sv = m.self_value_sources(t, _aw())
    assert sv["aw"] == {"xlsx_20261111": 1}
    cmp_ = m.compare_static(t, sf.build_food_table(_cl(), _aw(), CODES3, P2OF).table)
    assert cmp_["same_codes"] and cmp_["aw"]["값_다른_코드_수"] == 2 and cmp_["ph"]["값_다른_코드_수"] == 0


def test_missing_list_check():
    m = _load("s0_6_static")
    t = sf.build_food_table(_cl(), _aw(), CODES3, P2OF, aw_source_filter="all").table
    samples = pd.DataFrame({"p3": ["A0101000000000", "코드없음", "A0104000000000"],
                            "d": pd.to_datetime(["2020-01-01", "2020-02-01", "2026-01-01"]).date})
    full = set(m.MISSING_FOOD_VARS) | set(m.GR2_VARS)
    r = m.missing_list_check(t, samples, {"order2": {"variables": sorted(full)}}, "2025-06-30")
    assert r["same"] and r["variables"]["food_aw"]["nan_rows_d_le_cut"] == 1     # '코드없음' 행
    # 결측이 없어지면 목록과 달라짐 → changed
    s2 = samples[samples["p3"] != "코드없음"]
    r2 = m.missing_list_check(t, s2, {"order2": {"variables": sorted(full)}}, "2025-06-30")
    assert not r2["same"] and "food_aw" in r2["changed"] and "gr2_mu_mean" in r2["changed"]


# ================================================================ 2. KO1
def _ko_ref(T, aw, ph, coef):
    import math

    a0, a1, a2, a3, a4, a5, a6, a7, a8 = coef
    bw = math.sqrt(1 - aw)
    z = a0 + a1 * T + a2 * ph + a3 * bw + a4 * T * ph + a5 * T * bw + a6 * ph * bw + a7 * T * T + a8 * ph * ph
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def test_ko_coefficients_and_values():
    assert wf.KOUTSOUMANIS["broth"] == (-605.935, 1.272, 180.260, -809.232, 0.507, -4.411, -183.674, -0.065, -11.460)
    assert wf.KOUTSOUMANIS["agar"] == (-516.908, 3.218, 130.741, -957.558, 0.0, -7.574, -186.298, -0.031, -6.687)
    rng = np.random.default_rng(0)
    T, aw, ph = rng.uniform(-5, 40, 300), rng.uniform(0.85, 1.0, 300), rng.uniform(3.5, 8.0, 300)
    aw[:5] = [1.0, 0.999, 0.995, 0.993, 0.9]
    for med in wf.KO_MEDIA:
        got = wf.ko_P(T, aw, ph, med)
        ref = np.array([_ko_ref(a, b, c, wf.KOUTSOUMANIS[med]) for a, b, c in zip(T, aw, ph)])
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-300)
    with np.errstate(over="raise", invalid="raise"):   # 넘침 경고 없음(아주 작은·큰 z)
        p = wf.ko_P(np.array([-50.0, 30.0]), np.array([0.5, 1.0]), np.array([2.0, 7.0]), "broth")
    assert 0 <= p[0] < 1e-100 and np.isfinite(p).all()
    x = wf.ko_P(np.array([np.nan, 20.0, 20.0]), np.array([0.99, np.nan, 0.99]), np.array([6.0, 6.0, np.nan]), "agar")
    assert np.isnan(x).all()
    assert np.isnan(wf.ko_P(np.array([20.0]), np.array([1.01]), np.array([6.0]), "broth")).all()   # aw > 1 그대로 NaN


def test_ko_features_bruteforce():
    from tests.test_wxfeat import D, _bounds, _make_long, _nm, _nmax, _rows, _val

    df = _make_long()
    cube = wn.DailyCube.from_long(df, "unit", "date", ("avg_temp",))
    look = {(r.unit, r.day): r for r in df.itertuples(index=False)}
    rows = _rows()
    codes, d, h = [r[0] for r in rows], np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    rng = np.random.default_rng(3)
    aw, ph = rng.uniform(0.95, 1.0, len(rows)), rng.uniform(5, 7.5, len(rows))
    aw[::5] = np.nan
    W = 30
    got = wf.ko_features(cube, codes, d, h, W, aw, ph, chunk_rows=7)
    assert list(got.columns) == wf.KO_NAMES == list(rg.KO)
    for i, (u, dd, hh) in enumerate(rows):
        T, S, E, R = _bounds(dd, hh, W)
        days = list(range(S, E + 1))
        exp = []
        for med in wf.KO_MEDIA:
            v = [float(wf.ko_P(_val(look, u, t, "avg_temp"), aw[i], ph[i], med)) for t in days]
            exp += [_nm(v), _nm([x for x, t in zip(v, days) if t >= E - 9]), _nmax(v)]
        a, b = got.iloc[i].to_numpy(), np.array(exp)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12, atol=1e-300)
    assert D("2025-01-01") > 0


def test_registry_extra_group_does_not_touch_existing():
    assert len(rg.UNITS) == 110 and len(rg.GROUPS) == 44 and "KO1" not in rg.IDS and "KO1" not in rg.ALL_IDS
    g = rg.group(rg.KO_GROUP)
    assert g.no == 45 and g.units == rg.KO and [rg.unit(c).no for c in rg.KO] == list(range(111, 117))
    base = dict(include_ids=["WEATHER", "HIS1", "HIS2", "HIS3", "HIS4", "WX1", "WX2", "WX3", "WX4", "WX5", "WX6", "WX7",
                             "FD1", "SO1", "X1", "X2", "L1"], k=100, n_ma=7, n_ma_dl=7)
    a = rg.load_version({"name": "a", "order": 2, "W": 60, "mode": "sum", **base})
    b = rg.load_version({"name": "b", "order": 2, "W": 60, "mode": "sum", **{**base, "include_ids": base["include_ids"]
                                                                            + ["KO1"]}})
    for f in (rg.tree_columns, rg.dl_static_columns, rg.dl_seq_channels):
        ca, cb = [c.name for c in f(a)], [c.name for c in f(b)]
        assert cb[:len(ca)] == ca
        assert cb[len(ca):] == ([] if f is rg.dl_seq_channels else list(rg.KO))
    # 결측 표시: KO = gr2_mu_<요약> 의 목록 여부를 따름
    ml = rg.MissingLists({1: frozenset(), 2: frozenset(["gr2_mu_mean", "gr2_mu_max"])})
    col = {c.name: c for c in rg.dl_static_columns(b)}
    assert ml.needs_indicator(col["gr4_ko_broth_mean"], 2) and ml.needs_indicator(col["gr4_ko_agar_max"], 2)
    assert not ml.needs_indicator(col["gr4_ko_agar_mean10"], 2)


def test_ko_table_features_and_tree_X(ctx):   # noqa: F811
    from tests.test_registry_inputs import _meta

    meta = _meta(ctx, n=120)
    inc = ["WEATHER", "FD1", "WX3"]
    a = rg.VersionSpec("a", 2, 20, "sum", include_ids=tuple(inc))
    b = rg.VersionSpec("b", 2, 20, "sum", include_ids=tuple(inc + ["KO1"]))
    Xa, Xb = ip.build_tree_X(meta, ctx, a), ip.build_tree_X(meta, ctx, b)
    assert list(Xb.columns)[:Xa.shape[1]] == list(Xa.columns) and list(Xb.columns)[Xa.shape[1]:] == list(rg.KO)
    pd.testing.assert_frame_equal(Xa, Xb[list(Xa.columns)])
    food = sf.food_features(ctx.food, meta["p3"].to_numpy(dtype=object))
    ref = wf.ko_features(ctx.wx, meta["sgg"].to_numpy(dtype=object), fd.day_ints(meta["d"]), meta["h"].to_numpy(), 20,
                         food["food_aw"].to_numpy(), food["food_ph"].to_numpy())
    np.testing.assert_allclose(Xb[list(rg.KO)].to_numpy(np.float64), ref.to_numpy().astype(np.float32), equal_nan=True)
    # gr2 µmax 와 결측 행이 같음(같은 입력·같은 규칙)
    for s in ("mean", "mean10", "max"):
        assert np.array_equal(Xb[f"gr4_ko_broth_{s}"].isna(), Xb[f"gr2_mu_{s}"].isna())
    # 딥러닝: x_static 끝 6칸 + 결측 표시는 gr2_mu 를 따름
    raw = ip.build_dl_raw(meta, ctx, b)
    ml = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(list(rg.WEATHER19) + ["gr2_mu_mean"])})
    prep = ip.DLPreprocessor(b, ml, ctx.vocab).fit(raw)
    sn = prep.static_names
    assert "gr4_ko_broth_mean__na" in sn and "gr4_ko_broth_max__na" not in sn
    assert [n for n in sn if n.startswith("gr4_ko_") and not n.endswith("__na")] == list(rg.KO)


from tests.test_registry_inputs import ctx  # noqa: E402,F401  (모듈 범위 합성 자료 묶음)


# ================================================================ 3. 5-LSTM 다시 학습(rerun)
from tests.test_runner import env  # noqa: E402,F401
from tests.test_s4_s5_6 import _a, _best, _fake_run, s4env  # noqa: E402,F401


def _rerun_cfg(tmp, cfg_name, **rr):
    c = json.loads((CODE / "configs" / cfg_name).read_text(encoding="utf-8"))
    c["use_mlflow"] = False
    c["rerun"].update(rr)
    p = tmp / cfg_name
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return p, c


def _prep_56(s4env):
    """단계 4 판정 뒤 사슬 상태(채택 없음) → 5·6 뽑기 → 5-6-T 판(가짜 모델) → 판정 폴더."""
    m56 = _load("s5_6")
    inputs = s4env["inputs"]
    st = s4.load_state(inputs / "logs" / "chain_state_3-0b.json")
    st["stage4"] = {"start_best": st["current_best"], "start_tree_target": st["tree_target"], "adopted": False}
    s4.save_state(inputs / "logs" / "chain_state_4.json", st)
    m56.main(_a(s4env, inputs / "s5_6" / "sel") + ["--config", str(s4env["cfgs"]["s5_6_prep"])])
    b = m56.build(json.loads(s4env["cfgs"]["s5_6_T"].read_text(encoding="utf-8")), inputs)
    _fake_run(s4env, s4env["log"], inputs / "s5_6" / "5-6-T", b["run"])
    (inputs / "s5_6" / "5-6-judge").mkdir(parents=True)
    (inputs / "s5_6" / "5-6-judge" / "5-6_decision.json").write_text("{}", encoding="utf-8")
    return m56, b


def test_rerun_same_spec_new_name_and_dirs(s4env, tmp_path):
    m56, b0 = _prep_56(s4env)
    inputs = s4env["inputs"]
    old = {sp.name: sp for sp in b0["run"]}["5-XGB"]
    p, cfg = _rerun_cfg(tmp_path, "s5_6_T_food2.json", of="5-XGB", of_dir="s5_6/5-6-T/5-XGB", name="5-XGB-food2",
                        compare=[{"name": "5-XGB", "dir": "s5_6/5-6-T/5-XGB", "desc": "원 판"}])
    assert m56.main(_a(s4env, "x") + ["--config", str(p), "--list"]) == ["5-XGB-food2"]
    b = m56.build(cfg, inputs)
    (sp,) = b["run"]
    assert sp.name == "5-XGB-food2" and sp.stage == "5-6-food2" and sp.input_set == "5-XGB-food2-입력"
    d_old, d_new = old.to_dict(), sp.to_dict()
    for k in ("name", "stage", "input_set", "extra"):
        d_old.pop(k), d_new.pop(k)
    assert d_old == d_new                                                   # 설정 같음(이름·단계·입력 id 만 다름)
    assert {k: v for k, v in sp.extra.items() if k != "rerun"} == dict(old.extra)
    assert s4.s4_of(sp)["row_plan"]["sel_dir"] == "s5_6/sel-hon5" and s4.s4_of(sp)["model_seed"] == 0   # 같은 뽑은 키
    assert b["bests"]["5-XGB-food2"]["dir"] == "s5_6/5-6-T-food2/5-XGB-food2"
    # 새 폴더에 돌림 — 옛 판 폴더 그대로
    before = {f.name: f.stat().st_mtime_ns for f in (inputs / "s5_6" / "5-6-T" / "5-XGB").iterdir()}
    _fake_run(s4env, s4env["log"], inputs / "s5_6" / "5-6-T-food2", b["run"])
    assert (inputs / "s5_6" / "5-6-T-food2" / "5-XGB-food2" / "done.json").exists()
    assert before == {f.name: f.stat().st_mtime_ns for f in (inputs / "s5_6" / "5-6-T" / "5-XGB").iterdir()}
    # 보고 비교 줄: 원 판 포함
    rows = m56.report_rows_for(inputs)(cfg, inputs, s4env["st"], s4env["log"])
    assert [r.name for r in rows][-1] == "5-XGB" and rows[-1].desc == "원 판"
    # --out 이 새 폴더가 아니면 멈춤
    with pytest.raises(cf.ConfigError):
        m56.main(_a(s4env, inputs / "s5_6" / "5-6-T") + ["--config", str(p)])
    # KO 추가 판: version include_ids 뒤에 KO1
    pk, ck = _rerun_cfg(tmp_path, "s5_6_T_food2_ko.json", of="5-XGB", of_dir="s5_6/5-6-T/5-XGB", name="5-XGB-KO",
                        compare=[])
    (k,) = m56.build(ck, inputs)["run"]
    assert k.extra["version"]["include_ids"] == list(old.extra["version"]["include_ids"]) + ["KO1"]
    assert k.stage == "5-6-food2-KO" and k.name == "5-XGB-KO"


def test_rerun_guards(s4env, tmp_path):
    m56, _ = _prep_56(s4env)
    inputs = s4env["inputs"]
    # 원 판 spec 과 다르면 멈춤
    d = inputs / "s5_6" / "5-6-T" / "5-XGB"
    sj = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    alt = inputs / "s5_6" / "alt" / "5-XGB"
    alt.mkdir(parents=True)
    (alt / "spec.json").write_text(json.dumps({**sj, "W": 30}), encoding="utf-8")
    _, cfg = _rerun_cfg(tmp_path, "s5_6_T_food2.json", of="5-XGB", of_dir="s5_6/alt/5-XGB", name="5-XGB-food2",
                        compare=[])
    with pytest.raises(s4.StageError, match="W"):
        m56.build(cfg, inputs)
    for bad in ({"name": "5-XGB"}, {"stage": "5-6"}, {"add_ids": ["WX3"]}):
        c = json.loads(json.dumps(cfg))
        c["rerun"].update(of_dir="s5_6/5-6-T/5-XGB", **bad)
        if "stage" in bad:
            c["stage"] = bad["stage"]
        with pytest.raises(cf.ConfigError):
            m56.check_cfg(c)
    c = json.loads(json.dumps(cfg))
    c["rerun"]["of"] = "5-없음"
    with pytest.raises(s4.StageError):
        m56.build(c, inputs)


def test_mlflow_new_stage_run_no_collision(tmp_path):
    import lst.tracking as tk

    tr = tk.Tracker(f"sqlite:///{(tmp_path / 'm.db').as_posix()}", artifact_root=(tmp_path / "a").as_uri())
    sr = tr.stage_run("listeria-2차", "5-6")
    rid = tr.start_variant("listeria-2차", sr, "5-LSTM", {"단계": "5-6"})
    tr.finish(rid)
    tr.finish(sr)
    with pytest.raises(tk.TrackingError, match="이미 끝난"):
        tr.start_variant("listeria-2차", sr, "5-LSTM", {"단계": "5-6"})
    sr2 = tr.stage_run("listeria-2차", "5-6-food2")
    assert sr2 != sr
    r2 = tr.start_variant("listeria-2차", sr2, "5-LSTM-food2", {"단계": "5-6-food2"})
    sr3 = tr.stage_run("listeria-2차", "5-6-food2-KO")
    r3 = tr.start_variant("listeria-2차", sr3, "5-LSTM-KO", {"단계": "5-6-food2-KO"})
    assert len({rid, r2, r3}) == 3
    assert tr.client.get_run(rid).info.status == "FINISHED"


def test_food2_configs_consistent():
    t = json.loads((CODE / "configs" / "s5_6_T_food2.json").read_text(encoding="utf-8"))
    k = json.loads((CODE / "configs" / "s5_6_T_food2_ko.json").read_text(encoding="utf-8"))
    f = json.loads((CODE / "configs" / "s_final_food2.json").read_text(encoding="utf-8"))
    o = json.loads((CODE / "configs" / "s5_6_T.json").read_text(encoding="utf-8"))
    st = json.loads((CODE / "configs" / "s0_6_static_food2.json").read_text(encoding="utf-8"))
    food2 = "s0/0-6_static_v2/식품_고정표.parquet"
    for c in (t, k, f):
        assert c["context"]["results"]["food_table"] == food2
        assert c["context"]["results"]["missing_lists"] == "s0/0-6_missing/missing_lists.json"   # 목록 다시 씀
        assert c["context"]["results"]["livestock"] == o["context"]["results"]["livestock"]
    for c in (t, k):
        assert {kk: v for kk, v in c["dirs"].items() if kk != "5-6-T"} == {kk: v for kk, v in o["dirs"].items() if kk != "5-6-T"}
        assert c["dirs"]["5-6-T"] not in (o["dirs"]["5-6-T"],) and c["stage"] == c["rerun"]["stage"] != "5-6"
        assert c["speed"]["dl_keyed_inputs"] and c["gpu_memory_fraction"] == 0.25 and c["n_jobs"] == 8
        for kk in ("chain_state_in", "selection_2", "decision_1_3", "theil_targets", "samples"):
            assert c[kk] == o[kk]
    assert t["dirs"]["5-6-T"] != k["dirs"]["5-6-T"] and t["stage"] != k["stage"]
    assert f["final_override"]["dir"] == f"{t['dirs']['5-6-T']}/{t['rerun']['name']}"
    assert f["final_override"]["replaces"] == t["rerun"]["of"] and len(f["importance_parts"]) == 6
    assert f["speed"]["gpu_importance"] and f["gpu_memory_fraction"] == 0.25
    assert st["results_inputs"]["prev_food_static"] == "s0/0-6_static/식품_고정표.parquet"
    _load("s_final").load_cfg(CODE / "configs" / "s_final_food2.json")


# ================================================================ 4. 최종 판 바꿔 끼우기
from tests.test_s_final import _argv, _cfg, _run, _spec, _state, sf as S_FINAL, world  # noqa: E402,F401


def _override(w, tag, state_rel, name, src, replaces):
    """src 판과 같은 설정으로 새 이름 판을 돌리고(가짜 모델), food 파일 md5 를 run_meta 에 넣음."""
    sp = w["V"][src][1]
    new = rn.VariantSpec(**{**sp.to_dict(), "name": name, "stage": sp.stage + "-food2"})
    rel = f"s_ov_{tag}"
    _run(w, w["inputs"] / rel, new)
    food = w["inputs"] / "s0" / f"food_{tag}.parquet"
    pd.DataFrame({"p3": ["x"]}).to_parquet(food)
    frel = f"s0/food_{tag}.parquet"
    rm = w["inputs"] / rel / name / "run_meta.json"
    meta = json.loads(rm.read_text(encoding="utf-8"))
    meta["inputs_md5"][f"inputs:{frel}"] = cf.file_md5(food)
    rm.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    p = _cfg(w, tag, state_rel)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["context"] = {"results": {"food_table": frel}}
    raw["final_override"] = {"name": name, "dir": f"{rel}/{name}", "replaces": replaces, "label": "최종(식품 갱신)",
                             "reason": "시험"}
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return p


def test_final_override_select_and_change_table(world):
    w = world
    st = _state(w, "s4/state_ov.json", "4-a-CAT", "s4")
    p = _override(w, "ov", st, "4-a-CAT-food2", "4-a-CAT", "4-a-CAT")
    d = S_FINAL.main(_argv(w, "ov", p))
    assert d["final"] == "4-a-CAT-food2" and d["chain_last"] == "4-a-CAT"
    assert d["final_def"]["dir"] == "s_ov_ov/4-a-CAT-food2" and d["final_def"]["members"][0]["name"] == "4-a-CAT-food2"
    assert d["final_override"]["label"] == "최종(식품 갱신)"
    rdir = S_FINAL.main(_argv(w, "ov", p, "--report", "--reason", "시험"), tracker_fn=lambda c: None)
    ch = pd.read_csv(rdir / "change_table.csv", encoding="utf-8-sig")
    assert list(ch["구분"])[-2:] == ["최종 모델(식품 갱신 전)", "최종(식품 갱신)"]
    assert list(ch["판"])[-2:] == ["4-a-CAT", "4-a-CAT-food2"]
    t = json.loads((rdir / "final_test_scores.json").read_text(encoding="utf-8"))
    assert t["final"] == "4-a-CAT-food2"
    notes = json.loads((rdir / "research_notes.json").read_text(encoding="utf-8"))
    assert "식품 고정표 갱신 전" in notes["header"]


def test_final_override_guards(world):
    w = world
    st = _state(w, "s4/state_ov2.json", "4-a-CAT", "s4")
    p = _override(w, "ov2", st, "4-a-CAT-food2x", "4-a-CAT", "8-T-CAT")          # replaces 다름
    with pytest.raises(S_FINAL.FinalError, match="replaces"):
        S_FINAL.main(_argv(w, "ov2", p))
    st3 = _state(w, "s8/state_ov3.json", "8-T-CAT", "s8")
    p3 = _override(w, "ov3", st3, "8-T-CAT-food2", "8-T-CAT", "8-T-CAT")          # 베이스라인이 이김 → 멈춤
    with pytest.raises(S_FINAL.FinalError, match="지휘자 확인"):
        S_FINAL.main(_argv(w, "ov3", p3))
    assert not (w["inputs"] / "final" / "ov3" / "final_decision.json").exists()
    st4 = _state(w, "s4/state_ov4.json", "4-a-CAT", "s4")
    p4 = _override(w, "ov4", st4, "4-a-CAT-food2y", "4-a-CAT", "4-a-CAT")
    raw = json.loads(p4.read_text(encoding="utf-8"))
    raw["context"]["results"]["food_table"] = "s0/food_ov.parquet"                 # 다른 고정표 → 멈춤
    p4.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(S_FINAL.FinalError, match="food_table"):
        S_FINAL.main(_argv(w, "ov4", p4))


class _FakeTracker:
    def __init__(self):
        self.art, self.tags = [], {}

    def chain_run(self):
        return "chain"

    def log_artifact(self, run, path, artifact_path=None):
        self.art.append((run, Path(path).name, artifact_path))

    def set_tags(self, rid, tags):
        self.tags.setdefault(rid, {}).update(tags)


def test_final_override_report_mlflow_keeps_old_final(world):
    """바꿔 끼운 최종 보고: 원 최종의 MLflow final/ 기록을 덮지 않고(결과 폴더 이름 경로), '최종' 태그는 새 판 '예(식품 갱신)'·원 판
    '이전(식품 갱신 전)'(검수 지적 2026-11-12)."""
    w = world
    st = _state(w, "s4/state_ov5.json", "4-a-CAT", "s4")
    p = _override(w, "ov5", st, "4-a-CAT-food2z", "4-a-CAT", "4-a-CAT")
    S_FINAL.main(_argv(w, "ov5", p))
    sj = json.loads((w["inputs"] / st).read_text(encoding="utf-8"))
    dirs = {"run_old": w["inputs"] / sj["current_best"]["dir"], "run_new": w["inputs"] / "s_ov_ov5" / "4-a-CAT-food2z"}
    saved = {k: (d / "done.json").read_text(encoding="utf-8") for k, d in dirs.items()}
    try:
        for rid, d in dirs.items():
            dj = json.loads(saved[rid])
            dj["mlflow_run_id"] = rid
            (d / "done.json").write_text(json.dumps(dj, ensure_ascii=False), encoding="utf-8")
        fake = _FakeTracker()
        rdir = S_FINAL.main(_argv(w, "ov5", p, "--report", "--reason", "시험"), tracker_fn=lambda c: fake)
    finally:
        for rid, d in dirs.items():
            (d / "done.json").write_text(saved[rid], encoding="utf-8")
    assert fake.art == [("chain", "change_table.csv", "ov5")] and rdir.parent.name == "ov5"
    assert fake.tags == {"run_new": {"최종": "예(식품 갱신)"}, "run_old": {"최종": "이전(식품 갱신 전)"}}


# ================================================================ 5. KO 판도 최종 후보(사용자 결정 2026-11-12 'KO 점수 더 좋으면 그게 최종')
def test_targets_final_extra_group_only_when_present():
    """최종 중요도: KO 그룹은 KO 단위가 입력에 있을 때만, 없으면 기존 44 그룹 대상·씨앗 그대로."""
    from lst import importance as im
    base = [u.name for u in rg.UNITS]
    a = im.targets_final(base)
    assert all(t.no <= len(rg.GROUPS) for t in a if t.kind == "group")
    b = im.targets_final(base + [u.name for u in rg.EXTRA_UNITS])
    extra = [t for t in b if t.kind == "group" and t.no > len(rg.GROUPS)]
    assert [t.name for t in extra] == [g.name for g in rg.EXTRA_GROUPS] and len(extra) == 1
    assert extra[0].seed_key == (im.SEED_FINAL_GROUP, rg.EXTRA_GROUPS[0].no)
    assert [t for t in b if t not in extra] == a


def test_final_ko_config_consistent(tmp_path):
    f = json.loads((CODE / "configs" / "s_final_food2.json").read_text(encoding="utf-8"))
    k = json.loads((CODE / "configs" / "s_final_food2_ko.json").read_text(encoding="utf-8"))
    t = json.loads((CODE / "configs" / "s5_6_T_food2_ko.json").read_text(encoding="utf-8"))
    assert [kk for kk in f if f[kk] != k[kk]] == ["plan", "final_override"] and set(f) == set(k)
    assert k["final_override"]["name"] == t["rerun"]["name"] == "5-LSTM-KO"
    assert k["final_override"]["dir"] == f"{t['dirs']['5-6-T']}/{t['rerun']['name']}"
    assert k["final_override"]["replaces"] == f["final_override"]["replaces"] == "5-LSTM"
    assert k["final_override"]["via"] == [{"name": f["final_override"]["name"], "dir": f["final_override"]["dir"],
                                           "label": "식품 갱신(생장식 KO 없음)"}]
    assert "via" not in f["final_override"] and "tag" not in f["final_override"]
    S = _load("s_final")
    S.load_cfg(CODE / "configs" / "s_final_food2_ko.json")
    bad = json.loads(json.dumps(k))
    bad["final_override"]["via"] = [{"name": "5-LSTM", "dir": "x", "label": "y"}]   # 원 판 이름 → 멈춤
    bp = tmp_path / "bad.json"
    bp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(cf.ConfigError, match="via"):
        S.load_cfg(bp)


def _fake_pick_run(root, rel, name, cv, many_ties=False):
    d = root / rel / name
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"name": name, "train_count": 4, "algorithm": "LSTM", "kind": "dl"}),
                                 encoding="utf-8")
    (d / "done.json").write_text("{}", encoding="utf-8")
    (d / "cv_scores.json").write_text(json.dumps({"cv_band_weighted": cv, "ties": {"many_ties": many_ties}}),
                                      encoding="utf-8")


@pytest.mark.parametrize("cv_ko,expect", [(0.03505, "5-LSTM-KO"), (0.03504, "5-LSTM-food2"), (0.0349, "5-LSTM-food2")])
def test_food2_pick(tmp_path, cv_ko, expect):
    """val(CV) 4자리 반올림 뒤 KO 가 오를 때만 KO 설정(0.03496 → 0.0350 기준: 0.03505 → 0.0351 채택, 0.03504 → 같음)."""
    m = _load("s_food2_pick")
    inputs = tmp_path / "results"
    (inputs / "logs").mkdir(parents=True)
    _fake_pick_run(inputs, "s5_6/5-6-T-food2", "5-LSTM-food2", 0.03496)
    _fake_pick_run(inputs, "s5_6/5-6-T-food2-KO", "5-LSTM-KO", cv_ko)
    cf_f, cf_k = CODE / "configs" / "s_final_food2.json", CODE / "configs" / "s_final_food2_ko.json"
    argv = ["--inputs", str(inputs), "--config-food2", str(cf_f), "--config-ko", str(cf_k)]
    got = m.main(argv)
    assert got == str(cf_k if expect == "5-LSTM-KO" else cf_f)
    rec = json.loads((inputs / m.PICK_FILE).read_text(encoding="utf-8"))
    assert rec["chosen"] == expect and rec["judge"]["adopt"] == (expect == "5-LSTM-KO")
    assert "test" not in json.dumps(rec["current"]) + json.dumps(rec["candidate"])
    log = [json.loads(x) for x in (inputs / "logs" / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["event"] == "decision" and log[-1]["name"] == expect
    with pytest.raises(m.PickError, match="이미 있음"):
        m.main(argv)


def test_food2_pick_guards(tmp_path):
    m = _load("s_food2_pick")
    inputs = tmp_path / "results"
    (inputs / "logs").mkdir(parents=True)
    _fake_pick_run(inputs, "s5_6/5-6-T-food2", "5-LSTM-food2", 0.035)
    cf_f = CODE / "configs" / "s_final_food2.json"
    with pytest.raises(FileNotFoundError):                                   # KO 판 없음 → 멈춤
        m.main(["--inputs", str(inputs), "--config-food2", str(cf_f), "--config-ko",
                str(CODE / "configs" / "s_final_food2_ko.json")])
    with pytest.raises(m.PickError, match="name"):                            # 같은 설정 둘 → 멈춤
        m.main(["--inputs", str(inputs), "--config-food2", str(cf_f), "--config-ko", str(cf_f)])
    assert not (inputs / m.PICK_FILE).exists()


def test_final_override_via_rows_and_tag(world):
    """KO 판이 최종일 때(검수 2026-11-12): 변화 표 = 원 판 → via(5-LSTM-food2 자리) → 최종, MLflow 태그 = 설정 tag,
    리서치 표 머리 주석 = header_note, 입력 변수 표 정의 칸(추가 ID) — 계산은 그대로."""
    w = world
    st = _state(w, "s4/state_ov6.json", "4-a-CAT", "s4")
    _override(w, "ov6a", st, "4-a-CAT-f2v", "4-a-CAT", "4-a-CAT")                 # via 판(식품 갱신만)
    p = _override(w, "ov6", st, "4-a-CAT-kov", "4-a-CAT", "4-a-CAT")               # 최종 판(KO 자리)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["final_override"].update(via=[{"name": "4-a-CAT-f2v", "dir": "s_ov_ov6a/4-a-CAT-f2v", "label": "식품 갱신(생장식 KO 없음)"}],
                                 tag="예(식품 갱신 + 생장식 KO)", header_note="식품 고정표 갱신 + 생장식 KO 6칼럼 판",
                                 label="최종(식품 갱신 + 생장식 KO)")
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    S_FINAL.main(_argv(w, "ov6", p))
    new_dir = w["inputs"] / "s_ov_ov6" / "4-a-CAT-kov"
    saved = (new_dir / "done.json").read_text(encoding="utf-8")
    try:
        dj = json.loads(saved)
        dj["mlflow_run_id"] = "run_kov"
        (new_dir / "done.json").write_text(json.dumps(dj, ensure_ascii=False), encoding="utf-8")
        fake = _FakeTracker()
        rdir = S_FINAL.main(_argv(w, "ov6", p, "--report", "--reason", "시험"), tracker_fn=lambda c: fake)
    finally:
        (new_dir / "done.json").write_text(saved, encoding="utf-8")
    ch = pd.read_csv(rdir / "change_table.csv", encoding="utf-8-sig")
    assert list(ch["구분"])[-3:] == ["최종 모델(식품 갱신 전)", "식품 갱신(생장식 KO 없음)", "최종(식품 갱신 + 생장식 KO)"]
    assert list(ch["판"])[-3:] == ["4-a-CAT", "4-a-CAT-f2v", "4-a-CAT-kov"]
    assert fake.tags["run_kov"] == {"최종": "예(식품 갱신 + 생장식 KO)"}
    notes = json.loads((rdir / "research_notes.json").read_text(encoding="utf-8"))
    assert "4-a-CAT-kov(식품 고정표 갱신 + 생장식 KO 6칼럼 판)" in notes["header"]
