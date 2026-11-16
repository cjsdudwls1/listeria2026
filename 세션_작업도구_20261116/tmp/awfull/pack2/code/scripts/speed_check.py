# -*- coding: utf-8 -*-
"""speed_check.py — 속도 개선 스위치(lst/speed) 켬·끔 대조(실제 판 하나, 결과 파일 안 만듦 — 보고 json 하나만).

끝난 판 폴더(model_<fold>·inputs_<fold>·columns_<fold>.json·spec.json — 트리·딥러닝 판, 결합·VAE 판은 아님)에서 fold 하나로:
  1. 평가 쌍 입력: h 마다 끔(예전 길)·켬(dl_keyed_inputs)으로 만들어 키마다 바이트 md5 대조 + 초(runner 처럼 pair_cache 안)
  2. 예측: h 하나의 입력으로 predict_proba 끔·켬(fast_predict) 바이트 대조 + 초
  3. 순열 중요도: h 하나(그 판 입력에 있는 부록 A-5 그룹 전부, M = 10)를 끔·켬(gpu_importance + fast_predict)으로
     계산해 섞기 전·섞은 뒤 h PR-AUC 바이트 대조(다르면 최대 차이·4자리 반올림 같음 여부) + 초
결과 = <out>/speed_check_<판>_<fold>.json(대조 결과·초). 아무 결과 폴더도 고치지 않음.

실행(서버 — 지휘자):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/speed_check.py $A --out ~/cyj_listeria_final/results/_speed_check --config configs/s5_6_T.json \\
     --variant-dir s5/<단계 폴더>/<판 이름> --fold fold1 --pred-h 30 --imp-h 30
  (--h 1,2,3 = 평가 입력을 그 h 만, 없으면 그 fold 평가 쌍 h 전부. --skip inputs,predict,importance 로 건너뛰기)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import inputs as ip  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import speed  # noqa: E402
from lst.paths import resolve  # noqa: E402

COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
OFF = None
ON_INPUT = {"dl_keyed_inputs": True}
ON_PRED = {"fast_predict": True}
ON_IMP = {"gpu_importance": True, "fast_predict": True}


def _md5s(X) -> dict:
    out = {}
    if isinstance(X, dict):
        for k, v in X.items():
            out[k] = [str(v.dtype), list(v.shape), hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest()]
        return out
    num = X.select_dtypes(exclude=[object])
    out["columns"] = hashlib.md5(json.dumps([list(map(str, X.columns)), [str(t) for t in X.dtypes]]).encode()).hexdigest()
    out["num"] = hashlib.md5(np.ascontiguousarray(num.to_numpy()).tobytes()).hexdigest()
    out["obj"] = hashlib.md5(json.dumps(X.select_dtypes(include=[object]).to_numpy().tolist(),
                                        ensure_ascii=False).encode()).hexdigest()
    return out


def _pairs(samples, split, hs):
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split):
        if idx.size and (hs is None or h in hs):
            yield h, idx


def _meta(samples, idx, h, W):
    m = samples.meta().iloc[idx].copy()
    m["h"] = np.int16(h)
    return rn.add_bounds(m, W)


def check_inputs(asm, samples, split, W, hs) -> dict:
    res = {}
    for name, sw in (("off", OFF), ("on", ON_INPUT)):
        speed.configure(sw)
        t0, md5 = time.time(), {}
        with asm.pair_cache():
            for h, idx in _pairs(samples, split, hs):
                md5[h] = _md5s(asm.transform(_meta(samples, idx, h, W)))
        res[name] = {"seconds": round(time.time() - t0, 1), "md5": md5}
    speed.configure(None)
    same = res["off"]["md5"] == res["on"]["md5"]
    return {"same_bytes": same, "n_h": len(res["off"]["md5"]), "seconds_off": res["off"]["seconds"],
            "seconds_on": res["on"]["seconds"],
            "different_h": [h for h in res["off"]["md5"] if res["off"]["md5"][h] != res["on"]["md5"].get(h)]}


def check_predict(leaf, samples, split, W, h) -> dict:
    idx = dict(_pairs(samples, split, {h}))[h]
    X = leaf.assembler.transform(_meta(samples, idx, h, W))
    out = {}
    for name, sw in (("off", OFF), ("on", ON_PRED)):
        speed.configure(sw)
        t0 = time.time()
        p = leaf.predict(X)
        out[name] = (p, round(time.time() - t0, 2))
    speed.configure(None)
    a, b = out["off"][0], out["on"][0]
    return {"h": h, "n": int(a.size), "same_bytes": a.tobytes() == b.tobytes(),
            "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
            "seconds_off": out["off"][1], "seconds_on": out["on"][1]}


def check_importance(leaf, samples, split, fold, h, M) -> dict:
    from lst import importance as im

    targets = im.targets_groups(leaf.present_units())
    out = {}
    for name, sw in (("off", OFF), ("on", ON_IMP)):
        speed.configure(sw)
        t0 = time.time()
        o = im._fold_pass(leaf, samples, split, im.fold_number(fold), targets, M, M, h_from=h, h_to=h)
        out[name] = (o["bh"][h - 1], o["ph"][:, :, h - 1].copy(), round(time.time() - t0, 1))
    speed.configure(None)
    (b0, p0, s0), (b1, p1, s1) = out["off"], out["on"]
    return {"h": h, "M": M, "n_targets": len(targets),
            "base_same": np.float64(b0).tobytes() == np.float64(b1).tobytes(), "base_off": float(b0),
            "base_on": float(b1),
            "perm_same_bytes": p0.tobytes() == p1.tobytes(),
            "perm_max_abs_diff": float(np.nanmax(np.abs(p0 - p1))) if p0.size else 0.0,
            "perm_same_round4": bool(np.array_equal(np.round(p0, 4), np.round(p1, 4))),
            "seconds_off": s0, "seconds_on": s1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, help="그 판 단계 설정(context·samples·gpu_memory_fraction 을 읽음)")
    ap.add_argument("--variant-dir", required=True, help="--inputs 아래 끝난 판 폴더")
    ap.add_argument("--fold", default="fold1", choices=list(ev.CV_FOLDS))
    ap.add_argument("--h", default=None, help="평가 입력 대조 h(쉼표) — 없으면 전부")
    ap.add_argument("--pred-h", type=int, default=30)
    ap.add_argument("--imp-h", type=int, default=30)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--gpu-fraction", type=float, default=0.25)
    ap.add_argument("--skip", default="", help="inputs,predict,importance 중 건너뛸 것")
    a = ap.parse_args(argv)
    root, inputs_root = Path(a.root).expanduser().resolve(), Path(a.inputs).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    cfg = cf.load_json_strict(a.config)
    vdir = resolve(inputs_root, a.variant_dir)
    spec = rn.VariantSpec(**rn.read_json(vdir / "spec.json"))
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    from lst import models_dl as md
    from lst import stage3 as s3
    from lst.resources import limit_gpu_memory

    if spec.kind == "dl":
        limit_gpu_memory(a.gpu_fraction)
        md.set_gpu_fraction(a.gpu_fraction)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(COMPONENTS))
    leaf = s3.importance_nodes(ctx, samples, vdir, spec)[a.fold]
    split = fd.SPLITS[a.fold]
    hs = None if a.h is None else {int(x) for x in a.h.split(",")}
    rep = {"variant": spec.name, "fold": a.fold, "kind": spec.kind, "code_md5": cf.code_md5(CODE_ROOT)}
    if "inputs" not in skip:
        rep["inputs"] = check_inputs(leaf.assembler, samples, split, spec.W, hs)
        print("평가 입력:", json.dumps(rep["inputs"], ensure_ascii=False), flush=True)
    if "predict" not in skip:
        rep["predict"] = check_predict(leaf, samples, split, spec.W, a.pred_h)
        print("예측:", json.dumps(rep["predict"], ensure_ascii=False), flush=True)
    if "importance" not in skip:
        rep["importance"] = check_importance(leaf, samples, split, a.fold, a.imp_h, a.M)
        print("중요도:", json.dumps(rep["importance"], ensure_ascii=False), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"speed_check_{spec.name}_{a.fold}.json"
    cf.dump_scores_json(p, rep, overwrite=True)
    print(f"보고: {p}")
    return rep


if __name__ == "__main__":
    main()
