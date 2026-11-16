"""lst/wiring.py 시험 — 판 설정 변환·모델 연결(자료 없이)."""
import numpy as np
import pandas as pd
import pytest

from lst import registry as rg
from lst import runner as rn
from lst import wiring as wr


def spec(alg="LGB", kind="tree", W=10, mode="sum", **kw):
    return rn.VariantSpec(name=f"1-{alg}-W{W:03d}" + (f"-{mode}" if mode else ""), stage="1-1", order="1차", kind=kind,
                          algorithm=alg, W=W, mode=mode, input_set="1차", role="후보", train_count=4,
                          judgement="해당 없음", extra={"version": {"include_ids": ["WEATHER"]}}, **kw)


def test_version_spec():
    vs = wr.version_spec(spec())
    assert vs.order == 1 and vs.W == 10 and vs.mode == "sum" and vs.include_ids == ("WEATHER",)
    assert wr.tree_cat_cols(vs) == list(rg.CAT5)
    vd = wr.version_spec(spec("LSTM", "dl", 20, None))
    assert vd.mode == wr.DL_MODE_PLACEHOLDER and vd.dl_layout == "standard"
    with pytest.raises(ValueError):
        wr.version_spec(rn.VariantSpec(
            name="x", stage="1-1", order="1차", kind="tree", algorithm="XGB", W=10, mode="sum", input_set="",
            role="후보", train_count=4, extra={"version": {"모름": 1}}))


def test_make_model_and_fit_kwargs():
    w = wr.Wiring(ctx=None)
    m = w.make_model(spec("LGB"), asm=None)
    assert type(m).__name__ == "LGBMModel" and m.max_iter is None and m.cat_cols == list(rg.CAT5)
    with pytest.raises(ValueError):
        w.make_model(spec("XGB", params={"depth": 3}), asm=None)
    meta = pd.DataFrame({"sgg": ["11110", "11140"], "d": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    g = w.fit_kwargs(spec("LSTM", "dl", 10, None), "fold1", meta, None)["groups"]
    assert list(g[0]) == ["11110", "11140"] and g[1].dtype == np.int64
    assert w.fit_kwargs(spec(), "fold1", meta, None) == {}
