"""s0_9_timing 시험용 가짜 측정 함수(가짜 시간). 부르면 결과 뿌리의 calls.txt 에 항목 이름을 남김.
결과 뿌리에 FAIL_<항목> 파일이 있으면 그 항목은 멈춤(이어하기 시험)."""
from __future__ import annotations

import os
from pathlib import Path

from lst import folds as fd
from lst.models_tree import checkpoints


def _mark(env):
    d = Path(env["item_dir"])
    root = Path(env["items_root"]).parent
    with open(root / "calls.txt", "a", encoding="utf-8") as f:
        f.write(d.name + "\n")
    if (root / f"FAIL_{d.name}").exists():
        raise ValueError(f"가짜 실패: {d.name}")
    return d


def tree_fold(env, algo, version, max_iter, params, curve, read_cat_lr, save_model, components):
    _mark(env)
    sec = {"XGB": 3.0, "LGB": 2.0, "CAT": 5.0, "RF": 4.0, "ET": 3.5, "LR": 1.0}[algo] * (version["W"] / 10)
    out = {"seconds": sec, "seconds_fit": sec / 2, "train_rows": 1000, "n_cols": 10, "n_pairs": 500,
           "learning_rate": 0.05 if read_cat_lr else None}
    if curve:
        pts = checkpoints(max_iter)
        at_last = algo == "XGB" and max_iter < 4000           # XGB 만 한 번 2배
        out.update(points=pts, n_points=len(pts), own_best_n=pts[-1] if at_last else pts[5],
                   own_best_at_last=at_last, curve=[0.1] * len(pts))
    return out


def dl_fold(env, arch, version, save_model, components):
    _mark(env)
    return {"seconds": 7.0 if arch != "CMBV" else 9.0, "train_rows": 1000, "n_pairs": 500}


def importance(env, model_kind, algo, version, model_item, n_iter, M, components):
    _mark(env)
    if model_kind == "tree" and algo in ("XGB", "LGB", "CAT"):
        assert n_iter is not None
    return {"seconds": 44.0, "n_targets": 44, "seconds_per_target": 1.0, "n_pairs": 500}


def optuna(env, algo, version, max_iter, cache_val, components):
    _mark(env)
    return {"seconds": 6.0, "max_iter": max_iter}


def block(env, arch, version, block_model, components):
    _mark(env)
    return {"seconds": 5.0, "train_rows": 800, "block_model": block_model}


def ak(env, arch, version, mapping, rows_limit, epochs, components):
    _mark(env)
    if arch == "GRUA":
        raise ValueError("가짜: 그래프가 입력을 못 받음")
    return {"seconds": 11.0, "keys": ["x_seq", "x_static"] + (["x_cat"] if version.get("dl_layout") != "seq_only" else [])}


def infer60(env, arch, version, model_item, components):
    _mark(env)
    return {"seconds": 6.0, "n_pairs": 600, "seconds_per_pair": 0.01}


def fa(env, algo, version, max_iter, params, k, code_md5, theil_targets, sample_file):
    _mark(env)
    rows = {s: 900 for s in ("fold1", "fold2", "fold3", "refit")}
    return {"status": "ok", "seconds": 30.0, "largest_split": "refit", "reduced_rows": rows,
            "replicated_rows": {s: 900 * 60 for s in rows}, "setting": {"level": "float32", "m": None}, "pid": os.getpid()}


FAKES = {k: f"_s09_fake:{k}" for k in ("tree_fold", "dl_fold", "importance", "optuna", "block", "ak", "infer60", "fa")}
REFUSE = {k: "_s09_fake:refuse" for k in FAKES}


def refuse(**_kw):
    raise AssertionError("끝난 항목을 다시 부름")


_ = fd
