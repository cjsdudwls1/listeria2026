"""블록 OOF·엄격 OOF 계획(§7-3, 부록 A-12·A-9) 시험 — 합성 날짜만."""
import numpy as np
import pandas as pd
import pytest

from lst import blocks as bl
from lst import folds as fd


def D(s):
    return int(fd.to_days(s)[0])


def all_days():
    return np.arange(D("2015-04-07"), D("2025-06-30") + 1, dtype=np.int64)


def test_global_and_fold_blocks():
    assert [(b, str(fd.as_date([lo])[0]), str(fd.as_date([hi])[0])) for b, lo, hi in bl.GLOBAL_BLOCKS] == [
        (1, "2015-04-07", "2017-06-30"), (2, "2017-07-01", "2019-06-30"), (3, "2019-07-01", "2021-06-30"),
        (4, "2021-07-01", "2023-06-30"), (5, "2023-07-01", "2025-06-30")]
    f1, f2, f3, rf = (bl.fold_blocks(s) for s in bl.SPLIT_ORDER)
    assert len(f1) == 4 and f1[-1].as_dates() == ("2021-07-01", "2022-06-30")
    assert len(f2) == 4 and f2[-1].as_dates() == ("2021-07-01", "2023-06-30")
    assert len(f3) == 5 and f3[-1].as_dates() == ("2023-07-01", "2024-06-30")
    assert [(b.no, b.lo, b.hi) for b in rf] == list(bl.GLOBAL_BLOCKS)
    # 블록이 train 전체를 겹치지 않고 덮음
    d = all_days()
    for s in bl.SPLIT_ORDER:
        b = bl.block_of(d, s)
        assert ((b > 0) == (d <= fd.SPLITS[s].train_end)).all()


def test_plan_counts_reuse_19_and_22():
    p = bl.block_plan(reuse=True)
    c = bl.plan_counts(p)
    assert c["total"] == 19 and c["strict"] == 14 and c["global"] == 6 and c["strict_only"] == 13
    assert c["split_full"] == 4 and c["roles"] == 22
    names = {m.name: m for m in p}
    assert names["fold1-b4"].aliases == ["fold2-b4"]
    assert names["fold1-b4"].intervals_as_dates() == [("2015-04-07", "2021-06-30")]
    assert names["fold2-full"].aliases == ["fold3-b5", "refit-b5"]
    assert names["fold2-full"].intervals_as_dates() == [("2015-04-07", "2023-06-30")]
    q = bl.block_plan(reuse=False)
    cq = bl.plan_counts(q)
    assert cq["total"] == 22 and cq["strict"] == 16 and cq["global"] == 6 and all(len(m.roles) == 1 for m in q)
    t = bl.plan_table(p)
    assert len(t) == 22 and set(t["kind"]) == {"oof", "full"}


@pytest.mark.parametrize("reuse", [True, False])
def test_oof_property_and_assemble(reuse):
    p = bl.block_plan(reuse=reuse)
    d = all_days()
    for s in bl.SPLIT_ORDER:
        sp = fd.SPLITS[s]
        tr = d <= sp.train_end
        who = bl.assemble_oof(p, s, d)
        assert ((who >= 0) == tr).all()
        for i in np.unique(who[who >= 0]):
            rows = who == i
            m = p[i]
            assert not m.train_mask(d[rows]).any()                         # 예측 행을 학습 안 함
            # 학습 = 그 판 train 중 그 블록 밖 전부
            assert (m.train_mask(d) == (tr & ~rows)).all()
        full = bl.full_model(p, s)
        assert (full.train_mask(d) == tr).all()
        assert len(bl.models_for_split(p, s)) == len(bl.fold_blocks(s)) + 1


def test_split_subset_and_bad():
    p = bl.block_plan(splits=("refit",))
    assert bl.plan_counts(p)["total"] == 6
    with pytest.raises(ValueError):
        bl.block_plan(splits=("fold4",))
    with pytest.raises(ValueError):
        bl.block_of(np.array([D("2015-04-06")]), "fold1")


def test_model_train_rows_and_sizes():
    rng = np.random.default_rng(3)
    n = 3000
    d = rng.integers(D("2015-04-07"), D("2025-06-30") + 1, n)
    known = pd.DataFrame({"lims_data_key": [f"K{i}" for i in range(n)], "d": fd.as_date(d),
                          "y": (rng.random(n) < 0.05).astype(np.int8)})
    p = bl.block_plan(reuse=True)
    sizes = bl.model_sizes(p, known["d"])
    for m in p:
        r = bl.model_train_rows(m, known)
        assert len(r) == sizes[m.name] and m.train_mask(r["d"]).all()
        assert (fd.to_days(r["d"]) - fd.to_days(r["T"]) == r["h"].to_numpy()).all()
    q = bl.block_plan(reuse=False)
    keep = known["lims_data_key"][q[0].train_mask(known["d"])].iloc[::2]
    assert len(bl.model_train_rows(q[0], known, keep)) == len(keep)
    with pytest.raises(ValueError):
        bl.model_train_rows(p[3], known, keep)
