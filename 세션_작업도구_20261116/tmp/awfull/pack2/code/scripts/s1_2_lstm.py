# -*- coding: utf-8 -*-
"""s1_2_lstm.py — 단계 1-2 LSTM window 탐색(계획서 §6 1차 1-2, §7, 부록 A-8·A-9·A-14·A-15).

10 설정 = LSTM × W_DL 10~100(10 간격), 구조·학습 = 부록 A-8 LSTM·학습 줄 그대로(튜닝 없음), 입력 = 부록 A-8 1차 입력
(x_seq = 1차 기상 19 + 결측 표시, x_static = g ÷ 50·src_route·해외 구조적 결측, x_cat = sido·sgg·p2·p3), 모든 설정
3 fold + refit·test. 판 이름 = 1-LSTM-W<3자리>. 판별 학습 횟수(부록 A-12) = 그 모델 학습 수 = 4.
입력 조립(`lst.inputs`)·모델(`lst.models_dl`)은 다른 에이전트 작성 중 — `factories` 에서 붙임.

실행(서버 GPU, 검수 뒤 지휘자 — 1-1 과 나란히):
  ~/cyj_listeria_final/venv/bin/python scripts/s1_2_lstm.py --root ~/cyj_listeria_final/data \
      --inputs ~/cyj_listeria_final/results --out ~/cyj_listeria_final/results/s1/1-2 --config configs/s1_2_lstm.json
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
from lst.resources import GPU_MEMORY_FRACTION, limit_gpu_memory  # noqa: E402
from lst.windows import W_GRID  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COMPONENTS = ("weather", "missing")   # inputs.load_context 에서 읽을 자료

SCHEMA_EXTRA = {
    "algorithms": cf.Field(list, item_types=str),
    "W_grid": cf.Field(list, item_types=int),
    "input_set": cf.Field(str),
    "model_params": cf.Field(dict),
    "gpu_memory_fraction": cf.Field(float),
}


def specs(cfg: dict) -> list[rn.VariantSpec]:
    """1-2 판 10개(§6 1-2)."""
    if cfg["algorithms"] != ["LSTM"]:
        raise cf.ConfigError("algorithms 는 ['LSTM'](§6 1-2)")
    if tuple(cfg["W_grid"]) != W_GRID:
        raise cf.ConfigError(f"W_grid 는 {list(W_GRID)}(§0)")
    if cfg["model_params"]:
        raise cf.ConfigError("1-2 = 부록 A-8 구조·학습 그대로(튜닝 없음) — model_params 는 {}")
    if cfg["stage"] != "1-2" or cfg["order"] != "1차" or cfg["experiment"] != "listeria-1차":
        raise cf.ConfigError("stage 1-2 · order 1차 · experiment listeria-1차")
    out = [rn.VariantSpec(name=rn.name_stage1("LSTM", W), stage="1-2", order="1차", kind="dl", algorithm="LSTM", W=W,
                          mode=None, input_set=cfg["input_set"], role="후보", train_count=4, params={},
                          judgement="해당 없음", extra={"version": {"include_ids": ["WEATHER"]}})
           for W in W_GRID]
    assert len(out) == 10
    return out


def before_variant(cfg: dict) -> None:
    """GPU 메모리 상한(공통 규칙: 0.4 이하) — 판 하위 프로세스마다."""
    f = cfg["gpu_memory_fraction"]
    if not (0 < f <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    limit_gpu_memory(f)


def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """연결(lst.wiring): 입력 = inputs.DLInputs(판·split 마다 새로 — 표준화는 그 모델 train 으로), 모델 =
    models_dl.DLClassifier('LSTM', cat_sizes), fit 에 groups = (train 행 sgg, d)(부록 A-8 조기 종료 묶음).
    자료 = 국내·해외 기상 일자료 + 단계 0-6 결측 표시 목록. smoke.max_epochs 는 연기 시험에서만."""
    from lst import wiring as wr

    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS, dl_kwargs=dl_kw)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


def main(argv=None):
    return sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0],
                         schema_extra=SCHEMA_EXTRA, specs_fn=specs, factories_fn=factories,
                         before_variant=before_variant)


if __name__ == "__main__":
    main()
