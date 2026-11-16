"""국내만 버전 '지금 판 그대로' 시험용 지문(md5) — 기본 경로(설정에 scope 칸 없음)의 칼럼 목록·입력 행렬·예측·점수.

이 파일은 lst.scope 를 쓰지 않아 국내만 버전을 넣기 전 코드에서도 그대로 돔. tests/test_scope_domestic.py 의 GOLDEN 은
국내만 버전을 넣기 전 코드(사본)로 `digest_all` 을 돌려 얻은 값 — 지금 코드의 기본 경로가 같은 값을 내야 함
(칼럼 이름·순서·형, 입력 행렬 바이트, 판 하나 끝까지 돈 fold·refit 예측 바이트·CV 점수).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _h(*parts) -> str:
    m = hashlib.md5()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            m.update(p)
        else:
            m.update(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        m.update(b"\x1e")
    return m.hexdigest()


def frame_md5(X) -> str:
    if isinstance(X, pd.DataFrame):
        parts = [list(map(str, X.columns)), [str(t) for t in X.dtypes]]
        for c in X.columns:
            v = X[c].to_numpy()
            parts.append(v.tobytes() if v.dtype != object else [None if pd.isna(x) else str(x) for x in v])
        return _h(*parts)
    return _h(*[(k, v.dtype.str, list(v.shape), v.tobytes()) for k, v in sorted(X.items())])


def _o(order: str) -> str:
    return {"1차": "o1", "2차": "o2"}[order]


def _cols(cols) -> list:
    return [(c.name, c.kind, c.role, c.source, c.unit, c.group, bool(c.structural)) for c in cols]


def registry_digest() -> dict[str, str]:
    from lst import registry as rg

    specs = []
    for W in (10, 60, 100):
        for mode in ("sum", "raw"):
            specs.append({"name": f"v1-{W}-{mode}", "order": 1, "W": W, "mode": mode, "include_ids": ["WEATHER"]})
            specs.append({"name": f"v2-{W}-{mode}", "order": 2, "W": W, "mode": mode, "include_ids": list(rg.ALL_IDS),
                          "k": 50, "n_ma": 15, "n_ma_dl": 7})
    specs.append({"name": "lims", "order": 2, "W": 30, "mode": "sum", "include_ids": []})
    specs.append({"name": "seqT", "order": 2, "W": 30, "mode": "raw", "include_ids": ["WEATHER", "WX1", "WX2"],
                  "n_ma": 7, "n_ma_dl": 7, "tree_layout": "seq_only"})
    specs.append({"name": "seqD", "order": 2, "W": 30, "mode": "sum", "include_ids": ["WEATHER", "WX1", "WX2"],
                  "n_ma": 7, "n_ma_dl": 7, "dl_layout": "seq_only"})
    out = {}
    for s in specs:
        v = rg.load_version(s)
        out[s["name"]] = _h(_cols(rg.tree_columns(v)), _cols(rg.dl_seq_channels(v)), _cols(rg.dl_fixed_static(v)),
                            _cols(rg.dl_static_columns(v)))
    ml = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(list(rg.WEATHER19) + ["spi3", "td1", "L1"])})
    out["membership"] = _h(rg.membership_table(60, 50, 15, 7, missing=ml).to_csv(index=False))
    return out


def inputs_digest() -> dict[str, str]:
    from lst import folds as fd
    from lst import runner as rn
    from lst import wiring as wr
    from tests import test_dedup as td

    rng = np.random.default_rng(11)
    dom, ov = td._wx(rng)
    obs, fc, sov = td._spi(rng)
    s = td._samples(rng)
    c, st = _world(dom, ov, obs, fc, sov, s, rng)
    out = {}
    cases = [("XGB", 20, "sum", "1차", None), ("XGB", 20, "raw", "2차", td.ALL2), ("LR", 20, "sum", "2차", td.ALL2),
             ("CAT", 10, "sum", "2차", {"include_ids": []})]
    for alg, W, mode, order, ver in cases:
        sp = td.vspec(alg, W, mode, order, ver)
        rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), W)
        asm = wr.Wiring(c).make_inputs(sp, "fold1").fit(rows)
        parts = [list(asm.columns), frame_md5(asm.transform(rows))]
        parts += [frame_md5(x) for x in td._pair_inputs(asm, st, "fold1", W)[::7]]
        out[f"tree-{alg}-{W}-{mode}-{_o(order)}"] = _h(*parts)
    for ver, W, order, lay in (({"include_ids": ["WEATHER"]}, 20, "1차", None), (td.ALL2, 20, "2차", None),
                               ({"include_ids": ["WEATHER", "WX1", "WX2"], "n_ma_dl": 7, "dl_layout": "seq_only"},
                                20, "2차", "seq")):
        sp = td.vspec("LSTM", W, None, order, ver, kind="dl")
        rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold2"]), W)
        asm = wr.Wiring(c).make_inputs(sp, "fold2").fit(rows)
        parts = [asm.prep.static_names, asm.prep.seq_channel_names, frame_md5(asm.transform(rows))]
        parts += [frame_md5(x) for x in td._pair_inputs(asm, st, "fold2", W)[::9]]
        out[f"dl-{_o(order)}-{lay}"] = _h(*parts)
    return out


def _world(dom, ov, obs, fc, sov, s, rng):
    from lst import evaluate as ev
    from lst import folds as fd
    from lst import inputs as ip
    from lst import registry as rg
    from lst import wxfeat as wf
    from tests import test_registry_inputs as tri

    c = ip.InputContext(samples=s, y_cut=fd.day("2025-06-30"))
    c.wx = wf.combined_cube(dom, ov)
    c.spi_obs, c.spi_fc = ip.spi_cubes(obs, fc, sov)
    c.food = tri._food()
    c.livestock = pd.DataFrame({"unit": ["11110", "OVCHN"], "L1": [3.5, 0.2]})
    c.soil = tri._soil(rng)
    c.eco = tri._indicator("eco_ratio")
    c.gw = tri._indicator("gw_pass_rate")
    c.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(
        list(rg.WEATHER19) + ["spi3", "food_aw", "food_ph", "hist_rate5y_p3", "td1", "L1"] + list(rg.RG_SOILT))})
    return c, ev.SampleTable(s)


def stage4_digest(tmp: Path) -> dict[str, str]:
    from lst import folds as fd
    from lst import runner as rn
    from lst import stage4 as s4
    from tests import test_stage4 as t4

    ctx, st = t4._syn_ctx()
    w = s4.Stage4Wiring(ctx, st, tmp, n_jobs=1, device="cpu")
    ver = {"include_ids": ["WEATHER", "HIS1", "WX7", "FD1", "SO1"], "k": 30}
    rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]), 10)
    out = {}
    for alg in ("XGB", "LR"):
        sp = rn.VariantSpec(name=f"4-a0-{alg}", stage="4", order="2차", kind="tree", algorithm=alg, W=10, mode="sum",
                            input_set="x", role="후보", train_count=4,
                            extra={"version": ver, "s4": {"catenc_k": 30, "row_plan": s4.RowPlan().to_dict()}})
        asm = w.make_inputs(sp, "fold1").fit(rows)
        out[f"s4-{alg}"] = _h(list(asm.columns), frame_md5(asm.transform(rows)))
    nn = rn.VariantSpec(name="x2-NN60", stage="4", order="2차", kind="dl", algorithm="NN60", W=10, mode="sum",
                        input_set="x", role="교체 후보", train_count=4, extra={"version": ver, "s4": {}})
    an = w.make_inputs(nn, "fold1").fit(rows)
    out["s4-NN60"] = _h(list(an.columns), frame_md5(an.transform(rows)))
    return out


def run_digest(tmp: Path) -> dict[str, str]:
    """판 하나를 runner 로 끝까지(fold 1~3·refit, 연기 시험 크기) — 실제 입력 조립(Wiring)·실제 모델."""
    from lst import folds as fd  # noqa: F401
    from lst import runner as rn
    from lst import wiring as wr
    from tests import test_dedup as td

    rng = np.random.default_rng(11)
    dom, ov = td._wx(rng)
    obs, fc, sov = td._spi(rng)
    s = td._samples(rng)
    c, st = _world(dom, ov, obs, fc, sov, s, rng)
    w = wr.Wiring(c, dl_kwargs={"max_epochs": 1}, device="cpu")
    out = {}
    cases = [("XGB", 10, "sum", "1차", None, "tree"), ("LR", 10, "sum", "2차", td.ALL2, "tree"),
             ("LSTM", 10, None, "1차", None, "dl")]
    for alg, W, mode, order, ver, kind in cases:
        sp = td.vspec(alg, W, mode, order, ver, kind=kind)
        sp = rn.VariantSpec(**{**sp.to_dict(), "name": f"g-{alg}-{order}", "stage": "1-1"})
        od = tmp / f"run_{alg}_{order}"
        ctx = rn.RunContext(samples=st, out_dir=od, code_root=Path(__file__).resolve().parents[1],
                            input_md5s={"x": "0"}, exp_log=rn.ExperimentLog(tmp / f"log_{alg}_{order}.jsonl"),
                            make_inputs=w.make_inputs, make_model=w.make_model, fit_kwargs=w.fit_kwargs, n_jobs=2,
                            check_resources=lambda p: {"note": "test"}, require_7a_logged=False,
                            smoke={"max_train_rows": 400, "max_pairs_per_h": 25})
        rn.run_variant(sp, ctx)
        vd = rn.variant_dir(od, sp.name)
        parts = [frame_md5(pd.read_parquet(rn.pred_path(vd, s_))) for s_ in rn.SPLIT_ORDER]
        parts.append(json.loads((vd / "cv_scores.json").read_text(encoding="utf-8")))
        out[f"run-{alg}-{_o(order)}"] = _h(*parts)
    return out


def digest_all(tmp: Path, with_dl_run: bool = True) -> dict[str, str]:
    tmp = Path(tmp)
    out = {}
    out.update(registry_digest())
    out.update(inputs_digest())
    out.update(stage4_digest(tmp / "s4"))
    out.update(run_digest(tmp / "run"))
    if not with_dl_run:
        out.pop("run-LSTM-o1", None)
    return out
