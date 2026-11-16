"""계산 중복 없애기(inputs 구현 세부 (g)·runner 같은 행 한 번 예측) = 예전 계산과 바이트 단위로 같은지(합성 자료).

- 입력: 키 고유값으로 계산 + h 묶음 사이 메모리 캐시(pair_cache) ↔ 모든 행으로 계산(inputs.DEDUP = False).
- 예측: predict_pairs_multi(같은 행 묶어 한 번 예측 — XGB·LGB·CAT) ↔ 모든 행 예측, 로지스틱·RF·ET·딥러닝(LSTM,
  CPU)은 입력 캐시만 쓰고 예측은 모든 행 그대로.
- 행 묶기(unique_rows): NaN 비트·±0·문자열 범주·해시 충돌에도 정확.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import folds as fd
from lst import inputs as ip
from lst import registry as rg
from lst import runner as rn
from lst import wiring as wr
from lst import wxfeat as wf
from lst.hassign import assign_h_many
from tests.test_registry_inputs import _food, _indicator, _soil  # noqa: F401

UNITS = ("11110", "11140", "26110", "OVCHN")
START, END = "2015-01-01", "2026-06-30"


def _wx(rng):
    dates = pd.date_range(START, END, freq="D")
    dom, ov = [], []
    for u in UNITS[:3]:
        df = pd.DataFrame({"sgg_code": u, "date": dates})
        for v in wf.CUBE_VARS:
            x = np.round(rng.normal(10, 5, len(dates)), 1)
            if v == "rain":
                x = np.abs(x)
            x[rng.random(len(dates)) < 0.05] = np.nan
            df[v] = x
        dom.append(df)
    df = pd.DataFrame({"alpha_3": "CHN", "sgg": "OVCHN", "date": dates})
    for v in rg.WEATHER19:
        df[v] = np.nan if v in rg.STRUCT_VARS else rng.normal(12, 4, len(dates))
    df["avg_rhm"] = rng.normal(60, 10, len(dates))
    df["avg_td"] = np.nan
    ov.append(df)
    return pd.concat(dom, ignore_index=True), pd.concat(ov, ignore_index=True)


def _spi(rng):
    dates = pd.date_range(START, END, freq="D")
    obs = pd.concat([pd.DataFrame({"sgg_code": u, "date": dates, "spi3": rng.normal(0, 1, len(dates)),
                                   "spi6": rng.normal(0, 1, len(dates))}) for u in UNITS[:3]], ignore_index=True)
    fc = obs.copy()
    fc["spi3"] += 0.5
    ov = pd.DataFrame({"sgg": "OVCHN", "alpha_3": "CHN", "date": dates, "spi3": rng.normal(0, 1, len(dates)),
                       "spi6": rng.normal(0, 1, len(dates))})
    return obs, fc, ov


P3 = ("A0101000000000", "A0102000000000", "B0204000000000", "코드없음")
P2 = {"A0101000000000": "A0100000000000", "A0102000000000": "A0100000000000", "B0204000000000": "B0200000000000",
      "코드없음": "A0100000000000"}


def _samples(rng, n=2600):
    # 같은 (sgg, d)·(범주 5, d) 시료가 많게: 날짜를 적은 수에서 뽑음
    lo, hi = fd.day("2016-01-01"), fd.day("2026-06-20")
    days = np.sort(rng.choice(np.arange(lo, hi), 700, replace=False))
    d = days[rng.integers(0, days.size, n)]
    sgg = np.array(UNITS)[rng.integers(0, len(UNITS), n)]
    keys = [f"Q{i:05d}" for i in range(n)]
    p3 = np.array(P3)[rng.integers(0, 2, n) + (rng.random(n) < 0.3) * 2]
    y = (rng.random(n) < 0.08).astype("int8")
    df = pd.DataFrame({"lims_data_key": keys, "y": pd.array(y, dtype="Int8"),
                       "src_route": np.where(pd.Series(sgg).str.startswith("OV"), "overseas", "domestic"),
                       "sgg": sgg, "sido": [("OV" if u.startswith("OV") else u[:2]) for u in sgg],
                       "p2": [P2[x] for x in p3], "p3": p3, "d": fd.as_date(d)})
    df.loc[fd.to_days(df["d"]) > fd.SPLITS["refit"].train_end, "y"] = pd.NA
    df["h"] = assign_h_many(keys).astype(np.int16)
    return df


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(11)
    dom, ov = _wx(rng)
    obs, fc, sov = _spi(rng)
    s = _samples(rng)
    c = ip.InputContext(samples=s.drop(columns=[]), y_cut=fd.day("2025-06-30"))
    c.wx = wf.combined_cube(dom, ov)
    c.spi_obs, c.spi_fc = ip.spi_cubes(obs, fc, sov)
    c.food = _food()
    c.livestock = pd.DataFrame({"unit": ["11110", "OVCHN"], "L1": [3.5, 0.2]})
    c.soil = _soil(rng)
    c.eco = _indicator("eco_ratio")
    c.gw = _indicator("gw_pass_rate")
    c.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(
        list(rg.WEATHER19) + ["spi3", "food_aw", "food_ph", "hist_rate5y_p3", "td1", "L1"] + list(rg.RG_SOILT))})
    return c, ev.SampleTable(s)


ALL2 = {"include_ids": list(rg.ALL_IDS), "k": 50, "n_ma": 15, "n_ma_dl": 7}


def vspec(alg, W, mode, order="1차", version=None, kind="tree"):
    return rn.VariantSpec(name=f"t-{alg}-{W}-{mode}", stage="t", order=order, kind=kind, algorithm=alg, W=W,
                          mode=mode, input_set="t", role="후보", train_count=4,
                          extra={"version": version or {"include_ids": ["WEATHER"]}})


def _bytes(X):
    if isinstance(X, pd.DataFrame):
        num = X.select_dtypes(exclude=[object])
        return (list(X.columns), [str(t) for t in X.dtypes], num.to_numpy().tobytes(),
                X.select_dtypes(include=[object]).to_numpy().tolist())
    return {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in X.items()}


def _pair_inputs(asm, st, split, W, scope=True):
    """runner 처럼 h 마다 평가 쌍 입력(pair_cache 안/밖)."""
    meta = st.meta()
    out = []
    ctxm = asm.pair_cache() if scope else __import__("contextlib").nullcontext()
    with ctxm:
        for h, idx in fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS[split]):
            if idx.size == 0:
                continue
            m = meta.iloc[idx].copy()
            m["h"] = np.int16(h)
            out.append(asm.transform(rn.add_bounds(m, W)))
    return out


CASES = [("XGB", 20, "sum", "1차", None), ("CAT", 30, "raw", "1차", None),
         ("XGB", 20, "sum", "2차", ALL2), ("LGB", 20, "raw", "2차", ALL2), ("LR", 20, "sum", "2차", ALL2)]


@pytest.mark.parametrize("alg,W,mode,order,ver", CASES)
def test_tree_inputs_same_bytes(world, monkeypatch, alg, W, mode, order, ver):
    c, st = world
    sp = vspec(alg, W, mode, order, ver)
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), W)
    a_new = wr.Wiring(c).make_inputs(sp, "fold1").fit(rows)
    x_tr_new = a_new.transform(rows)
    new = _pair_inputs(a_new, st, "fold1", W)
    monkeypatch.setattr(ip, "DEDUP", False)
    a_old = wr.Wiring(c).make_inputs(sp, "fold1").fit(rows)
    x_tr_old = a_old.transform(rows)
    old = _pair_inputs(a_old, st, "fold1", W, scope=False)
    assert _bytes(x_tr_new) == _bytes(x_tr_old)
    assert len(new) == len(old) and sum(len(x) for x in new) > 2000
    for a, b in zip(new, old):
        assert _bytes(a) == _bytes(b)


def test_dl_inputs_same_bytes(world, monkeypatch):
    c, st = world
    for ver, W in (({"include_ids": ["WEATHER"]}, 20), ({**ALL2}, 20)):
        sp = vspec("LSTM", W, None, "1차" if len(ver) == 1 else "2차", ver, kind="dl")
        rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold2"]), W)
        a_new = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        a_new.chunk_rows = 97                                  # 묶음 경계를 넘는 캐시도 시험
        new = _pair_inputs(a_new, st, "fold2", W)
        tr_new = a_new.transform(rows)
        monkeypatch.setattr(ip, "DEDUP", False)
        a_old = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        old = _pair_inputs(a_old, st, "fold2", W, scope=False)
        tr_old = a_old.transform(rows)
        monkeypatch.setattr(ip, "DEDUP", True)
        assert a_new.prep.state() == a_old.prep.state()       # 표준화 통계 같음(train 행 수·가중 그대로)
        assert _bytes(tr_new) == _bytes(tr_old)
        for a, b in zip(new, old):
            assert _bytes(a) == _bytes(b)


def test_memo_reuses_keys(world):
    """h 묶음 사이 캐시: (sgg, E) 키 수 = 평가 쌍 전체의 고유 (sgg, E) 수(쌍 수보다 훨씬 적음)."""
    c, st = world
    sp = vspec("XGB", 20, "sum")
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 20)
    asm = wr.Wiring(c).make_inputs(sp, "fold1").fit(rows)
    meta = st.meta()
    keys, n = set(), 0
    with asm.pair_cache() as memo:
        for h, idx in fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS["fold1"]):
            m = rn.add_bounds(meta.iloc[idx].assign(h=np.int16(h)), 20)
            asm.transform(m)
            keys |= set(zip(m["sgg"], fd.day_ints(m["E"])))
            n += len(m)
        got = memo.n_keys()
    assert list(got.values()) == [len(keys)] and len(keys) < n / 5
    assert asm._memo is None                                   # 끝나면 비움


# ---------------------------------------------------------------- 예측
def _preds(models, asm, st, split, W, monkeypatch, dedup: bool, specs=()):
    """dedup = 새 경로(입력 캐시 + run_group 이 정하는 같은 행 한 번 예측), 아니면 예전 경로(모든 행으로 계산·예측)."""
    if dedup:
        calls = []
        real = rn.unique_rows

        def spy(X):
            r = real(X)
            calls.append((len(r[0]), len(r[1])))
            return r
        monkeypatch.setattr(rn, "unique_rows", spy)
        out = rn.predict_pairs_multi(models, asm, st, split, W, dedupe_rows=rn.row_dedupe_ok(specs))[0]
        monkeypatch.undo()
        return out[next(iter(models))], calls
    monkeypatch.setattr(ip, "DEDUP", False)

    class NoCache:   # pair_cache 없는 조립 객체(예전 runner 경로)
        def __init__(self, a):
            self.a = a

        def transform(self, m):
            return self.a.transform(m)
    out = rn.predict_pairs_multi(models, NoCache(asm), st, split, W)[0]
    monkeypatch.undo()
    return out[next(iter(models))], None


@pytest.mark.parametrize("alg,W,mode,order,ver", CASES + [("RF", 20, "sum", "1차", None),
                                                         ("ET", 20, "raw", "2차", ALL2)])
def test_tree_predictions_same_bytes(world, monkeypatch, alg, W, mode, order, ver):
    c, st = world
    sp = vspec(alg, W, mode, order, ver)
    w = wr.Wiring(c)
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), W)
    asm = w.make_inputs(sp, "fold1").fit(rows)
    X = asm.transform(rows)
    sp2 = sp if alg not in ("XGB", "LGB", "CAT") else rn.VariantSpec(**{**sp.to_dict(), "params": {"max_iter": 30}})
    model = w.make_model(sp2, asm).fit(X, rows["y"].to_numpy(np.int64), seed=42, n_jobs=4)
    a, calls = _preds({"m": model}, asm, st, "fold1", W, monkeypatch, True, [sp])
    b, _ = _preds({"m": model}, asm, st, "fold1", W, monkeypatch, False)
    assert len(a) > 2000
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a["pred"].to_numpy().tobytes() == b["pred"].to_numpy().tobytes()
    if alg not in rn.ROW_DEDUPE_ALGS:   # 로지스틱·RF·ET 는 같은 행 한 번 예측을 쓰지 않음(입력 계산 중복 없애기만)
        assert calls == []
    else:               # 같은 행이 실제로 묶임
        assert sum(u for u, _ in calls) < sum(n for _, n in calls)


def test_dl_predictions_same_bytes(world, monkeypatch):
    """딥러닝: 입력 계산 중복 없애기만(예측은 모든 행 그대로 — 배치 크기에 따라 마지막 자리가 바뀌므로)."""
    pytest.importorskip("torch")
    c, st = world
    sp = vspec("LSTM", 20, None, kind="dl")
    w = wr.Wiring(c, dl_kwargs={"max_epochs": 2}, device="cpu")
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 20)
    asm = w.make_inputs(sp, "fold1").fit(rows)
    X = asm.transform(rows)
    model = w.make_model(sp, asm).fit(X, rows["y"].to_numpy(np.int64), seed=42, n_jobs=4,
                                      **w.fit_kwargs(sp, "fold1", rows, X))
    a, calls = _preds({"m": model}, asm, st, "fold1", 20, monkeypatch, True, [sp])
    b, _ = _preds({"m": model}, asm, st, "fold1", 20, monkeypatch, False)
    assert calls == [] and not rn.row_dedupe_ok([sp])
    assert a["pred"].to_numpy().tobytes() == b["pred"].to_numpy().tobytes()


# ---------------------------------------------------------------- 행 묶기
def _check_groups(X, rep, inv):
    Xr = rn.take_rows(X, rep)
    Xe = rn.take_rows(Xr, inv)
    assert _bytes(Xe) == _bytes(rn.take_rows(X, np.arange(len(inv))))
    return len(rep)


def test_unique_rows_exact(monkeypatch):
    nan2 = np.frombuffer(np.uint32(0x7FC00001).tobytes(), np.float32)[0]   # 다른 NaN 비트
    X = pd.DataFrame({"s": ["a", "a", "b", "a", "a", "a"],
                      "x": np.array([1.0, 1.0, 1.0, np.nan, nan2, -0.0], np.float32),
                      "y": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)})
    X = pd.concat([X, X.iloc[[3, 4, 5]]], ignore_index=True)
    rep, inv = rn.unique_rows(X)
    assert rep.tolist() == [0, 2, 3, 4, 5] and inv.tolist() == [0, 0, 1, 2, 3, 4, 2, 3, 4]
    _check_groups(X, rep, inv)
    # 해시가 모두 같아도(충돌) 틀리게 묶지 않음
    monkeypatch.setattr(rn, "_part_hash", lambda p, rng: np.zeros(p.shape[0], np.uint64))
    rep2, inv2 = rn.unique_rows(X)
    _check_groups(X, rep2, inv2)
    assert len(rep2) >= len(rep)


def test_unique_rows_dict_and_fallback():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(5, 4, 3)).astype(np.float32)
    D = {"x_seq": seq[[0, 1, 0, 2, 1, 3, 4, 0]], "x_static": np.zeros((8, 2), np.float32),
         "x_cat": np.array([[1], [2], [1], [3], [2], [4], [5], [9]], np.int64)}
    rep, inv = rn.unique_rows(D)
    assert rep.tolist() == [0, 1, 3, 5, 6, 7] and inv.tolist() == [0, 1, 0, 2, 1, 3, 4, 5]
    _check_groups(D, rep, inv)
    # 결측 든 범주 칼럼 → 묶지 않음(항등)
    X = pd.DataFrame({"s": ["a", None, "a"], "x": np.ones(3, np.float32)})
    rep, inv = rn.unique_rows(X)
    assert rep.tolist() == [0, 1, 2] and inv.tolist() == [0, 1, 2]
