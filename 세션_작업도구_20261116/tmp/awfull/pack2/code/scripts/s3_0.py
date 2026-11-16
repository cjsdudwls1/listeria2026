# -*- coding: utf-8 -*-
"""s3_0.py — 단계 3-0 알고리즘 넓히기(13 구조 × 변수 전체) — 계획서 요약 12, §0, §3(반복 수·학습률), §6 2차 공통·단계 3-0,
§7, 부록 A-8·A-9·A-12·A-14·A-15.

판(13, 이름 = 3-0-<구조> — 부록 A-14):
  설정 configs/s3_0_T.json(sub 'T', CPU): 3-0-XGB·3-0-LGB·3-0-CAT(§3 반복 수 고르기 — 교차 N·공통 N, CatBoost 학습률 =
      fold 별 자동 학습률 평균, 최대 반복 = 알고리즘마다 첫 2차 사슬 부스팅 판(= 이 판)에서 fold 3 먼저 정함)·
      3-0-RF·3-0-ET(기본 설정) — 같은 입력이라 한 묶음(입력 한 번), 3-0-LR(표준화 입력 — 혼자, max_iter 2,000·수렴 기록)
  설정 configs/s3_0_D.json(sub 'D', GPU): 3-0-LSTM·GRU·TRF·LSTMA·TCN·CMB·CMBV — 같은 입력(부록 A-8 2차 딥러닝 입력)이라
      한 묶음(전처리기·입력 한 번, 모델은 차례로)
모두: 1차 W(= W_DL)·트리 입력 방식(1-3 결정), 변수 전체(§5 후보 전부 + 1차 기상), k·N_ma = 단계 2 준비 고르기
(2_prep_selection.json), 3 fold + refit·test 예측, 판정 tag '판정 없음'(3-0 판), 역할 '후보'.

실행(서버 — 검수 뒤 지휘자, 단계 2 준비 고르기 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_0.py $A --out ~/cyj_listeria_final/results/s3/3-0-T --config configs/s3_0_T.json      # CPU
  $P scripts/s3_0.py $A --out ~/cyj_listeria_final/results/s3/3-0-D --config configs/s3_0_D.json      # GPU(나란히 가능 —
      MLflow 단계 run '3-0' 을 두 프로세스가 같이 쓰므로 T 를 먼저 띄우고 몇 초 뒤 D)
  보고(3-0 이 다 끝난 뒤, 3-0b 고르기는 CV 만이라 순서 무관): 같은 명령 + --report --reason "단계 3-0 보고"(설정마다)
  판 목록 --list / 판 하나 --variant 3-0-LR(stage_cli 공통 인자)

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3 머리 (a)·(b), runner 의 2차 사슬 반복 수 고르기 구현 세부(곡선 뒤 fold
예측을 다시 만듦), 딥러닝 7종 한 묶음(runner.input_key — 결과는 따로 돌릴 때와 같음, 묶음 하나가 멈추면 7판 모두 다시).
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
from lst import scope as lst_scope  # noqa: E402
from lst import stage3 as s3  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.paths import resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, limit_gpu_memory  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "3-0"
COMPONENTS = ("weather", "spi", "food", "livestock", "soil", "eco", "gw", "missing")
N_JOBS = {"T": 8, "D": None}
SCHEMA = {
    "sub": cf.Field(str, choices=("T", "D")),
    "decision_1_3": cf.Field(str),        # --inputs 아래 1-3 결정 json
    "selection_2": cf.Field(str),         # --inputs 아래 단계 2 준비 고르기 json
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "dirs": {"3-0-T": cf.Field(str), "3-0-D": cf.Field(str)},
    "gpu_memory_fraction": cf.Field(float, required=False),
}


def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    sub = cfg["sub"]
    if N_JOBS[sub] is not None and cfg["n_jobs"] != N_JOBS[sub]:
        raise cf.ConfigError(f"sub {sub} n_jobs 는 {N_JOBS[sub]}")
    if sub == "D":
        f = cfg.get("gpu_memory_fraction")
        if f is None or not (0 < f <= GPU_MEMORY_FRACTION):
            raise cf.ConfigError(f"D gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    elif "gpu_memory_fraction" in cfg:
        raise cf.ConfigError("gpu_memory_fraction 은 D 만")


def make_specs_fn(inputs_root: Path):
    holder = {}

    def specs_fn(cfg):
        check_cfg(cfg)
        if "specs" not in holder:
            dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
            sel = s3.load_selection_2(inputs_root, cfg["selection_2"], dec)
            holder["specs"] = s3.specs_3_0(cfg["sub"], dec, sel)
        return holder["specs"]
    return specs_fn


def factories(cfg, rp, inputs_root: Path) -> sc.Factories:
    from lst import wiring as wr

    dl_kw = {}
    if "smoke" in cfg and "max_epochs" in cfg["smoke"]:
        dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]}
    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS, dl_kwargs=dl_kw)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


def before_variant(cfg) -> None:
    if cfg["sub"] == "D":
        limit_gpu_memory(cfg["gpu_memory_fraction"])


def report_rows(cfg, inputs_root, samples, exp_log):
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    return [s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])]


def _pre(argv):
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    ns = ap.parse_known_args(list(argv))[0]
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None,
         before_variant_fn=before_variant):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre(argv)
    if not ns.inputs or not ns.config:
        raise SystemExit("--inputs·--config 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    raw = cf.load_json_strict(ns.config)
    if not (ns.list or ns.verify_run):        # 3-0b 가 dirs 자리를 읽음
        want = resolve(inputs_root, raw["dirs"][f"3-0-{raw['sub']}"]).resolve()
        if Path(ns.out).expanduser().resolve() != want:
            raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[3-0-{raw['sub']}] = {want}")
    return sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=SCHEMA,
                         specs_fn=make_specs_fn(inputs_root), factories_fn=factories_fn or factories,
                         before_variant=before_variant_fn, tracker_fn=tracker_fn, check_resources=check_resources_fn,
                         report_rows_fn=report_rows,
                         report_notes=("부스팅 판: CV = 교차 N 예측, threshold = 공통 N 예측, test = refit 공통 N(§3·§1-2)",))


if __name__ == "__main__":
    main()
