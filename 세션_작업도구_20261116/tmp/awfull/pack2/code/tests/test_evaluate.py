"""lst/evaluate.py 시험 — 채점이 metrics 와 같은지, threshold(합친 validation), test 막기, 덜 찬 달 분리, 동점 다시 채점."""
import math

import numpy as np
import pandas as pd
import pytest

from lst import evaluate as ev
from lst import folds as fd
from lst import metrics as mt
from lst.hassign import assign_h_many, u_tie_many


def make_samples(n=2500, seed=0, pos=0.05) -> pd.DataFrame:
    """합성 시료 표(단계 0-1 samples.parquet 꼴, y 는 전 기간)."""
    rng = np.random.default_rng(seed)
    lo, hi = fd.day("2015-06-01"), fd.day("2026-06-30")
    d = rng.integers(lo, hi + 1, n)
    keys = [f"K{i:06d}" for i in range(n)]
    ov = rng.random(n) < 0.12
    sgg_dom = np.array(["11110", "11140", "26110", "41110", "48120"])
    sgg = np.where(ov, np.where(rng.random(n) < 0.5, "OVCHN", "OVUSA"), sgg_dom[rng.integers(0, 5, n)])
    p3s = np.array(["A01", "A02", "B01", "B02", "C01"])
    p3 = p3s[rng.integers(0, 5, n)]
    df = pd.DataFrame({
        "lims_data_key": keys,
        "y": (rng.random(n) < pos).astype(np.int8),
        "src_route": np.where(ov, "overseas", "domestic"),
        "sgg": sgg,
        "sido": np.where(ov, "OV", pd.Series(sgg).str[:2].to_numpy()),
        "p2": pd.Series(p3).str[:1].to_numpy(),
        "p3": p3,
        "d": fd.as_date(d),
        "h": assign_h_many(keys).astype(np.int16),
    })
    # 덜 찬 달 시험: 해외 2026-06 시료가 꼭 있게
    m = df.index[df["src_route"] == "overseas"][:15]
    df.loc[m, "d"] = np.datetime64("2026-06-10")
    df["d"] = pd.to_datetime(df["d"]).dt.date
    return df


@pytest.fixture(scope="module")
def samp(tmp_path_factory):
    p = tmp_path_factory.mktemp("ev") / "samples.parquet"
    df = make_samples()
    df.to_parquet(p, index=False)
    return p, df, ev.SampleTable.read(p)


def fake_preds(st: ev.SampleTable, split: str, seed=1, const=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for h, idx in fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS[split]):
        if idx.size:
            p = np.full(idx.size, const) if const is not None else rng.random(idx.size)
            parts.append(pd.DataFrame({"lims_data_key": st.keys[idx], "h": np.int16(h), "fold": split, "pred": p}))
    return pd.concat(parts, ignore_index=True)


def test_sample_table_blind(samp):
    p, df, st = samp
    cut = fd.SPLITS["refit"].train_end
    assert np.isnan(st._y[st.d_days > cut]).all()
    with pytest.raises(ev.TestPeekError):
        ev.SampleTable(df)  # test 기간 y 가 든 표는 거부


def test_score_cv_matches_metrics(samp):
    p, df, st = samp
    preds = {f: fake_preds(st, f, seed=i) for i, f in enumerate(ev.CV_FOLDS)}
    cv = ev.score_cv(preds, st)
    ys, ss = [], []
    for f in ev.CV_FOLDS:
        P = preds[f]
        pos = st.positions(P["lims_data_key"])
        y = df["y"].to_numpy()[pos].astype(int)
        u = u_tie_many(P["lims_data_key"].tolist(), P["h"].astype(int).tolist())
        assert cv["fold_band_weighted"][f] == mt.pr_auc_band_weighted(y, P["pred"].to_numpy(), P["h"].to_numpy(), u)
        ys.append(y)
        ss.append(P["pred"].to_numpy())
    assert cv["cv_band_weighted"] == pytest.approx(np.mean(list(cv["fold_band_weighted"].values())), abs=0)
    thr = mt.select_threshold(np.concatenate(ys), np.concatenate(ss))
    assert cv["threshold"] == thr
    assert cv["ties"]["many_ties"] is False
    # 빠진 쌍이 있으면 멈춤(§3 쌍 전부)
    bad = dict(preds)
    bad["fold1"] = preds["fold1"].iloc[1:]
    with pytest.raises(ValueError, match="채점 쌍"):
        ev.score_cv(bad, st)


def test_ties_and_rescore(samp):
    _, _, st = samp
    preds = {f: fake_preds(st, f, const=0.3) for f in ev.CV_FOLDS}
    cv = ev.score_cv(preds, st)
    assert cv["ties"]["many_ties"] is True
    cv2 = ev.score_cv(preds, st, rescore=True)
    assert cv2["cv_band_weighted"] != cv["cv_band_weighted"]   # 다 동점이면 u_tie 가 순서를 정함
    row = ev.table_row("x", "y", cv)
    assert ev.MANY_TIES_MARK in row["표시"]


def test_test_gate(samp, tmp_path):
    p, _, st = samp
    with ev.selection_guard():
        with pytest.raises(ev.TestPeekError):
            ev.report_permit("1-1", "보고")
    permit = ev.report_permit("1-1", "보고")
    truth = ev.TestTruth.load(p, permit)
    with ev.selection_guard():
        with pytest.raises(ev.TestPeekError):
            truth.y_for(st.keys[:1])
    with pytest.raises(ev.TestPeekError):
        ev.TestTruth.load(p, object())
    fake = ev.ReportPermit("1-1", "가짜")
    with pytest.raises(ev.TestPeekError):
        ev.check_permit(fake)
    # 고르는 코드에 test 칸이 들어오면 멈춤
    with pytest.raises(ev.TestPeekError):
        ev.rank_candidates([ev.Candidate("a", 0.1, 4, False, {"test_band_weighted": 0.2})])


def test_incomplete_split(samp):
    p, df, st = samp
    permit = ev.report_permit("1-1", "보고")
    truth = ev.TestTruth.load(p, permit)
    tp = fake_preds(st, "refit", seed=5)
    inc = {("overseas", "2026-06")}
    t = ev.score_test(tp, st, truth, 0.5, inc)
    pos = st.positions(tp["lims_data_key"])
    d = pd.to_datetime(df["d"]).to_numpy()[pos]
    is_inc = (df["src_route"].to_numpy()[pos] == "overseas") & (pd.DatetimeIndex(d).strftime("%Y-%m") == "2026-06")
    assert is_inc.sum() > 0
    assert t["incomplete"]["n_pairs"] == int(is_inc.sum())
    assert t["report"]["n_pairs"] == len(tp) - int(is_inc.sum())
    assert t["h_composition_changed"] is True
    assert sum(c["n_pairs"] for c in t["incomplete"]["counts"]) == int(is_inc.sum())
    assert all(c["n_pairs"] == 0 for c in t["incomplete"]["counts"] if c["src_route"] == "domestic")
    # 본 점수 = 덜 찬 달 쌍을 뺀 쌍으로 metrics 그대로
    y = df["y"].to_numpy()[pos].astype(int)
    keep = ~is_inc
    u = u_tie_many(tp["lims_data_key"][keep].tolist(), tp["h"][keep].astype(int).tolist())
    assert t["report"]["pr_auc_band_weighted"] == mt.pr_auc_band_weighted(y[keep], tp["pred"].to_numpy()[keep],
                                                                        tp["h"].to_numpy()[keep], u)
    t0 = ev.score_test(tp, st, truth, 0.5, set())
    assert t0["incomplete"]["n_pairs"] == 0 and t0["report"]["n_pairs"] == len(tp)
    notes = ev.table_notes([t])
    assert any("덜 찬 달" in n for n in notes)


def test_rank_and_tie_check():
    c = [ev.Candidate("b", 0.10004, 4, False), ev.Candidate("a", 0.10001, 4, False),
         ev.Candidate("c", 0.10003, 0, False), ev.Candidate("d", 0.2, 4, True)]
    r = [x.name for x in ev.rank_candidates(c)]
    assert r == ["d", "c", "a", "b"]       # 0.1000 동점 → 학습 횟수 → 이름
    calls = []
    s = ev.select_with_tie_check([ev.Candidate("t", 0.3000, 4, True), ev.Candidate("u", 0.2995, 4, False),
                                  ev.Candidate("v", 0.1, 4, False)],
                                 lambda n: calls.append(n) or {"t": 0.2990, "u": 0.2995}[n])
    assert s["selected"] == "t" and set(calls) == {"t", "u"}
    assert s["ranking_rescore"][0] == "u" and s["rescore_same_top"] is False
    assert s["alerts"]
    j = ev.judge_adoption(ev.Candidate("n", 0.30004, 4, False), ev.Candidate("o", 0.3, 4, False), None)
    assert j["adopt"] is False   # 4자리 반올림 같음 → 상승 아님


def test_expand_sample_preds(samp):
    _, _, st = samp
    split = "fold2"
    m = fd.scoring_sample_mask(st.d_days, fd.SPLITS[split])
    sp = pd.DataFrame({"lims_data_key": st.keys[m], "p_hat": np.linspace(0, 1, int(m.sum()))})
    e = ev.expand_sample_preds(sp, st, split)
    assert len(e) == st.expected_pairs(split)
    assert set(e.groupby("lims_data_key")["pred"].nunique()) == {1}
    with pytest.raises(KeyError):
        ev.expand_sample_preds(sp.iloc[1:], st, split)


def test_table_row_columns(samp):
    p, _, st = samp
    preds = {f: fake_preds(st, f, seed=i) for i, f in enumerate(ev.CV_FOLDS)}
    cv = ev.score_cv(preds, st)
    permit = ev.report_permit("1-1", "보고")
    t = ev.score_test(fake_preds(st, "refit"), st, ev.TestTruth.load(p, permit), cv["threshold"]["threshold"],
                      {("overseas", "2026-06")})
    row = ev.table_row("1-CAT-W060-sum", "d", cv, t, selection_count=3, fold_recall=True)
    assert list(row) == list(ev.TABLE_COLUMNS)
    assert row["threshold"] == cv["threshold"]["threshold"]
    assert not math.isnan(row["test 구간 가중"]) and ev.INCOMPLETE_MARK in row["표시"]
    assert row["합친 threshold 의 fold 별 Recall"].count("/") == 2
