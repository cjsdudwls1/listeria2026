"""실행 틀 ↔ 입력 조립·모델 연결 — runner 주입 함수(make_inputs·make_model·fit_kwargs)와 판 설정 변환.

- 판 설정: runner.VariantSpec → registry.VersionSpec(`version_spec`). VersionSpec 의 단위 칸(include_ids 등)은
  VariantSpec.extra['version'] 에서 옴(1차 = {'include_ids': ['WEATHER']} — 범주 5·h·g 는 tree_columns 가 늘 넣음).
- 입력: 판·split 마다 새 조립 객체(inputs.TreeInputs — 로지스틱만 scale=True / inputs.DLInputs) — 표준화 통계는
  모델마다 그 train 으로(§4-2).
- 트리 모델: models_tree.make_model(알고리즘, cat_cols = tree_columns 의 범주 칼럼, max_iter = 1차 None(라이브러리 기본)).
  2차 사슬 부스팅 판(params['iter_select'])은 runner 가 실행 때 정한 max_iter·CatBoost learning_rate 를 params 에 넣어
  넘김(runner.model_spec) — iter_select 칸 자체는 모델에 안 넘김.
- 딥러닝: models_dl.DLClassifier(구조 약어, cat_sizes = 조립 객체의 범주 번호표 크기), fit 에 groups = (train 행의
  sgg, d)(부록 A-8 조기 종료 (sgg, d) 묶음 10%).
- 입력 md5(B-2): 자료 묶음 설정(context)에 적힌 파일 전부.

구현 세부(계획서에 없음 — 담당자 확인):
  (1) 판 이름 약어 LGB(부록 A-14) ↔ models_tree 'LGBM'.
  (2) 딥러닝 판의 VersionSpec.mode 는 'sum' 으로 둠(VersionSpec 이 mode 를 요구 — 딥러닝 입력에는 트리 입력 방식이 안 쓰임,
      2차 x_static 의 window 요약 칼럼은 부록 A-8 대로 요약값).
  (3) 로지스틱 indicator_cols = 조립 객체 칼럼 중 결측 표시('__na')·'ov_struct_na'(3-0 부터 — 1차엔 없음).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from . import config as cf
from . import folds as fd
from . import registry as rg
from .paths import resolve
from .runner import VariantSpec

TREE_ALGO = {"XGB": "XGB", "LGB": "LGBM", "CAT": "CAT", "RF": "RF", "ET": "ET", "LR": "LR"}
ORDER_NO = {"1차": 1, "2차": 2}
VERSION_KEYS = ("include_ids", "include_units", "exclude_units", "exclude_groups", "k", "n_ma", "n_ma_dl",
                "tree_layout", "dl_layout")
DL_MODE_PLACEHOLDER = "sum"
TREE_PARAM_KEYS = ("max_iter", "params", "iter_select")   # iter_select = 2차 사슬 반복 수 고르기(runner)


def version_spec(spec: VariantSpec) -> rg.VersionSpec:
    if spec.order not in ORDER_NO:
        raise ValueError(f"입력 조립 판은 1차·2차만: {spec.order}")
    if spec.W is None:
        raise ValueError("W 없는 판은 아직 연결 안 함")
    ver = dict(spec.extra.get("version", {}))
    bad = sorted(set(ver) - set(VERSION_KEYS))
    if bad:
        raise ValueError(f"extra.version 의 모르는 칸: {bad}")
    mode = spec.mode if spec.kind == "tree" else DL_MODE_PLACEHOLDER
    return rg.load_version({"name": spec.name, "order": ORDER_NO[spec.order], "W": spec.W, "mode": mode, **ver})


def tree_cat_cols(vs: rg.VersionSpec) -> list[str]:
    return [c.name for c in rg.tree_columns(vs) if c.kind == "category"]


class Wiring:
    """runner 주입 함수 묶음. ctx = inputs.InputContext(판마다 같은 자료 — 한 번 읽음)."""

    def __init__(self, ctx, *, dl_kwargs: Mapping[str, Any] | None = None, device: str | None = None):
        self.ctx = ctx
        self.dl_kwargs = dict(dl_kwargs or {})   # 시험에서만(max_epochs 등) — 실제 실행은 비움(부록 A-8 그대로)
        self.device = device

    def make_inputs(self, spec: VariantSpec, split: str):
        from . import inputs as ip

        vs = version_spec(spec)
        if spec.kind == "tree":
            return ip.TreeInputs(self.ctx, vs, split, scale=(spec.algorithm == "LR"))
        return ip.DLInputs(self.ctx, vs, split)

    def make_model(self, spec: VariantSpec, asm):
        if spec.kind == "tree":
            from . import models_tree as mt

            bad = sorted(set(spec.params) - set(TREE_PARAM_KEYS))
            if bad:
                raise ValueError(f"트리 판 params 의 모르는 칸: {bad}")
            vs = version_spec(spec)
            ind = ([c for c in asm.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN]
                   if spec.algorithm == "LR" else [])
            return mt.make_model(TREE_ALGO[spec.algorithm], cat_cols=tree_cat_cols(vs), indicator_cols=ind,
                                 max_iter=spec.params.get("max_iter"), params=spec.params.get("params"))
        from . import models_dl as md

        return md.DLClassifier(spec.algorithm, asm.cat_sizes, device=self.device, **dl_model_kwargs(spec, self.dl_kwargs))

    def fit_kwargs(self, spec: VariantSpec, split: str, train_meta: pd.DataFrame, X) -> dict:
        if spec.kind == "dl":
            return {"groups": (train_meta["sgg"].astype(str).to_numpy(), fd.day_ints(train_meta["d"]))}
        return {}


DL_PARAM_KEY = "dl"     # 딥러닝 판 params = {} (부록 A-8 기본) 또는 {"dl": 단계 8-D 튜닝 설정}


def dl_model_kwargs(spec: VariantSpec, dl_kwargs: Mapping[str, Any] | None = None) -> dict:
    """딥러닝 판 params → DLClassifier 인자. {} = 기본 그대로, {"dl": {...}} = 단계 8-D 설정(batch_size 는 생성자 인자,
    나머지 = models_dl.HP_KEYS). 그 밖 칸이면 멈춤."""
    p = dict(spec.params)
    hp = p.pop(DL_PARAM_KEY, None)
    if p:
        raise ValueError(f"딥러닝 판 params 는 비움 또는 '{DL_PARAM_KEY}'(단계 8-D) — 모르는 칸 {sorted(p)}")
    kw = dict(dl_kwargs or {})
    if hp:
        hp = dict(hp)
        bs = hp.pop("batch_size", None)
        if bs is not None:
            if "batch_size" in kw:
                raise ValueError("batch_size 가 dl_kwargs 와 판 설정에 둘 다 있음")
            kw["batch_size"] = int(bs)
        if hp:
            kw["hparams"] = hp
    return kw


def context_paths(context_cfg: Mapping) -> tuple[list[str], list[str]]:
    """자료 묶음 설정 → (--inputs 아래 경로들, --root 아래 경로들)."""
    return list(context_cfg.get("results", {}).values()), list(context_cfg.get("data", {}).values())


def input_md5s(context_cfg: Mapping, data_root, inputs_root) -> dict[str, str]:
    r, d = context_paths(context_cfg)
    out = {f"inputs:{k}": v for k, v in cf.inputs_md5(inputs_root, r).items()}
    out.update({f"root:{k}": v for k, v in cf.inputs_md5(data_root, d).items()})
    return out


def load(context_cfg: Mapping, data_root, inputs_root, components: Iterable[str], **wiring_kw) -> Wiring:
    from . import inputs as ip

    return Wiring(ip.load_context(context_cfg, data_root, inputs_root, components=list(components)), **wiring_kw)
