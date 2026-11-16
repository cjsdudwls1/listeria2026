# -*- coding: utf-8 -*-
"""s1_1_trees.py — 단계 1-1 트리 window 탐색(계획서 §6 1차 1-1, §0, §3, 부록 A-9·A-12·A-14·A-15).

60 설정 = XGBoost·LightGBM·CatBoost × W 10~100(10 간격) × 입력 방식(요약 sum·원값 raw), 라이브러리 기본 설정 그대로
(반복 수·학습률 포함 — CatBoost 학습률은 학습마다 자동값, 검증으로 고르는 값 없음), 모든 설정 3 fold + refit·test.
입력 = 범주 sido·sgg·p2·p3·src_route + h·g + 1차 기상 19변수(국내 §4-3 표, 해외 DB 값만 §4-4) — 조립은 `lst.inputs`
(다른 에이전트 작성 중, 아래 `factories` 에서 붙임). 판 이름 = 1-<알고리즘>-W<3자리>-<sum|raw>.

실행(서버, 검수 뒤 지휘자 — 1-2 와 나란히, 0-1·0-4·0-5·0-7·0-9 뒤):
  ~/cyj_listeria_final/venv/bin/python scripts/s1_1_trees.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s1/1-1 --config configs/s1_1_trees.json
  판 목록: … --list / 판 하나: … --variant 1-CAT-W060-sum
  보고(0-10 기준 줄 뒤, 단계가 끝난 뒤): … --report --reason "1-1 보고"
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

from lst import config as cf  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.windows import W_GRID  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALGORITHMS = ("XGB", "LGB", "CAT")
ORDER1_IDS = ("WEATHER",)          # registry.ORDER1_IDS — 범주 5·h·g 는 tree_columns 가 늘 넣음
COMPONENTS = ("weather",)          # inputs.load_context 에서 읽을 자료
MODES = ("sum", "raw")
SCHEMA_EXTRA = {
    "algorithms": cf.Field(list, item_types=str),
    "W_grid": cf.Field(list, item_types=int),
    "modes": cf.Field(list, item_types=str),
    "input_set": cf.Field(str),
    "model_params": cf.Field(dict),
}


def specs(cfg: dict) -> list[rn.VariantSpec]:
    """1-1 판 60개(§6 1-1)."""
    if tuple(cfg["algorithms"]) != ALGORITHMS:
        raise cf.ConfigError(f"algorithms 는 {list(ALGORITHMS)}(§6 1-1)")
    if tuple(cfg["W_grid"]) != W_GRID:
        raise cf.ConfigError(f"W_grid 는 {list(W_GRID)}(§0)")
    if tuple(cfg["modes"]) != MODES:
        raise cf.ConfigError(f"modes 는 {list(MODES)}(§4-2)")
    if cfg["model_params"]:
        raise cf.ConfigError("1차 = 라이브러리 기본 설정 그대로 — model_params 는 {}(§0)")
    if cfg["stage"] != "1-1" or cfg["order"] != "1차" or cfg["experiment"] != "listeria-1차":
        raise cf.ConfigError("stage 1-1 · order 1차 · experiment listeria-1차")
    out = [rn.VariantSpec(name=rn.name_stage1(a, W, m), stage="1-1", order="1차", kind="tree", algorithm=a, W=W, mode=m,
                          input_set=cfg["input_set"], role="후보", train_count=4, params={}, judgement="해당 없음",
                          extra={"version": {"include_ids": list(ORDER1_IDS)}})
           for a in ALGORITHMS for W in W_GRID for m in MODES]
    assert len(out) == 60
    return out


def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """연결(lst.wiring): 입력 = inputs.TreeInputs(판·split 마다 새로), 모델 = models_tree.make_model(라이브러리 기본),
    자료 = 국내·해외 기상 일자료(1차 = WEATHER 만), 입력 md5 = context 에 적힌 파일 전부."""
    from lst import wiring as wr

    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


def main(argv=None):
    return sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0],
                         schema_extra=SCHEMA_EXTRA, specs_fn=specs, factories_fn=factories)


if __name__ == "__main__":
    main()
