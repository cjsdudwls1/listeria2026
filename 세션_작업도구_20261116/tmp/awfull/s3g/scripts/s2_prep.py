# -*- coding: utf-8 -*-
"""s2_prep.py — 단계 2 준비(2-0·2-1 k·2-1b N_ma·2-R 리서치팀 변수·고르기) — 계획서 요약 11, §0, §6 단계 2 준비·2차 공통,
§5 표·§5-1, §1-2(동점)·§1-3(test 막기), 부록 A-5·A-9·A-10·A-12·A-13·A-14·A-15.

하위 단계(설정의 "sub" 로 고름 — 설정 파일 하나 = 하위 단계 하나, 결과 폴더 = 설정 dirs[sub]):
  2-0     configs/s2_0_base.json      학습 없음: 1차 A 의 (W, 입력 방식) 판을 그대로 2-0-<A> 로(점수·예측 위치 복사,
                                      MLflow listeria-2차 단계 run '2' 아래 판 run, tag 바탕 판 = 그 1차 판 이름)
  2-1     configs/s2_1_k.json         A + HIS1~HIS4, k ∈ {10, 30, 50, 100, 200} — 5판(2-1-k010 …)
  2-1b-T  configs/s2_1b_tree.json     A + 차감 칼럼만(WX2), N_ma ∈ {5, 7, 15, 30} — 4판(2-1b-T-N05 …)
  2-1b-L  configs/s2_1b_lstm.json     1차 W_DL LSTM + 차감 채널만(2차 결측 표시 목록), N_ma 4값 — 4판(2-1b-L-N05 …)
  2-R     configs/s2_R_research.json  A + 리서치 칼럼(2-R1 = 9개 전부, 2-R2 ~ 2-R10 = 하나씩) — 10판,
                                      판이 다 끝나면 2-R1 중요도(칼럼 9·리서치 전체 묶음·기상 그룹, 부록 A-5 계산법, 학습 없음)
  select  configs/s2_select.json      고르기(CV 만): k = 2-1 CV 1위, 트리 N_ma = 2-1b-T 1위, 딥러닝 N_ma = 2-1b-L 1위
                                      (§1-2 동점: 소수 4자리 → 학습 횟수 → 판 이름) → 2_prep_selection.json(다음 단계가 읽음)
                                      + MLflow '선택' tag. --report = 단계 보고(test 채점, 하위 단계마다 표 1개)
모두 라이브러리 기본 설정(반복 수 고르기 없음 — §0), 3 fold + refit·test(runner), 1차 window 고정(1-3 결정 json).

실행 순서(서버, 검수 뒤 지휘자 — 1-3 이 끝난 뒤, §8 '2-0 → 2-1·2-1b → 미리보기 4-a(단계 4 담당) → 2-R'):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-0    --config configs/s2_0_base.json
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-1    --config configs/s2_1_k.json
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-1b-T --config configs/s2_1b_tree.json   # CPU
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-1b-L --config configs/s2_1b_lstm.json   # GPU(나란히 가능)
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-R    --config configs/s2_R_research.json
  $P scripts/s2_prep.py $A --out ~/cyj_listeria_final/results/s2/2-select --config configs/s2_select.json
  (하위 단계 끝난 뒤 보고)  … --config configs/s2_select.json --report --reason "단계 2 준비 보고"
  판 목록: … --list / 판 하나(하위 프로세스용): … --variant 2-1-k030 (stage_cli 공통 인자 그대로)

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 판 역할 tag = '비교만'(§6 최종 모델 고르기 "비교만 = …·2-0·2-1 k 비교·2-1b·…·2-R"), 판정 = '해당 없음'(부록 A-15).
     '바탕 판' tag 는 2-0 만(= 1차 A 판 이름) — 부록 A-15 바탕 판 목록에 2-1·2-1b·2-R 이 없어 비움.
 (2) 2-1b 를 설정 2개(트리 2-1b-T·LSTM 2-1b-L, 결과 폴더 따로)로 나눔 — 1-1·1-2 처럼 CPU·GPU 나란히, n_jobs 8·4.
     VariantSpec.stage 는 둘 다 '2-1b'(부록 A-13 자동 학습률 표시 단계 이름).
 (3) 차수 = 2차(order 2) 판 설정: 2-1 = WEATHER + HIS1~HIS4(k), 2-1b-T = WEATHER + WX2(n_ma), 2-1b-L = WEATHER + WX2
     (n_ma_dl, 2차 결측 표시 목록), 2-R = WEATHER + include_units(리서치 단위). 트리(로지스틱 아님)는 차수가 입력 값에
     영향 없음(결측 표시·표준화를 안 씀) — 2-0(1차 판)과 같은 기상 칼럼.
 (4) 2-0 결과 폴더 = spec.json·source.json(1차 판 위치·파일 md5)·cv_scores.json·h_metrics.csv 사본·done.json
     (예측 parquet 은 복사하지 않고 위치만 — 보고 때 source.json 의 md5 로 1차 파일이 그대로인지 확인).
     2-0 의 '같은 validation 으로 고른 횟수' = 1차 판 값 그대로(새 CV 채점이 아니라 실험 로그에 안 셈).
     MLflow 2-0 판 run = 1차 판 run 의 param 복사 + source_variant·source_run_id·pred_location param, CV metric 복사.
 (5) 단계 표(tables)는 MLflow 단계 run '2' 의 'tables/<하위 단계>' 아래(하위 단계 표 이름이 같아 덮이지 않게 — 이 스크립트의
     tracker 감싸기). 보고 표마다 기준 줄 = 7-a·1차 베이스라인·비교 기준 판(2-1·2-1b-T·2-R = 2-0, 2-1b-L = 1차 W_DL LSTM).
     2-R 보고에 2-0 대비 CV·test 구간 가중 변화 표(2-R_vs_2-0.csv — 부록 A-10 재료) 하나 더.
 (6) 보고(test 채점)는 고르기(2_prep_selection.json)와 2-1·2-1b-T·2-1b-L·2-R 판이 모두 끝난 뒤 한 번(§1-3 test 막기).
 (7) 2-R1 중요도 = 2-R 기본 실행(모든 판 끝)의 마지막에 한 번: 2-R1 fold 모델 3개(저장 모델 다시 읽음, 트리 전부로 예측 —
     n_iter 없음), 대상 = importance.targets_2r1(), M = 10, 섞기 전 점수가 2-R1 fold 점수와 같은지 확인. 결과 =
     <2-R 폴더>/2-R1_importance/ + MLflow 2-R1 판 run artifact 'importance/'.
 (8) 같은 입력 판끼리 묶음 실행(runner.group_specs): 단계 2 준비의 판은 알고리즘이 하나(A 또는 LSTM)이고 판마다 k·N_ma·
     리서치 칼럼이 달라 입력이 같은 판이 없음 → 모두 혼자 실행(묶음 1판). 입력 계산 일부 공유(기상 window 등)는 runner 가
     지원하지 않음(보고만 — runner 를 고치지 않음).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.history import K_VALUES  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources, limit_gpu_memory  # noqa: E402
from lst.windows import W_GRID  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPERIMENT = "listeria-2차"
STAGE = "2"                                   # 부록 A-15: 단계 2 준비 판 = listeria-2차 단계 run '2'
ORDER = "2차"
ROLE = "비교만"                               # 구현 세부 (1)
JUDGEMENT = "해당 없음"
TRAIN_SUBS = ("2-1", "2-1b-T", "2-1b-L", "2-R")
SUBS = ("2-0",) + TRAIN_SUBS + ("select",)
SUB_STAGE = {"2-1": "2-1", "2-1b-T": "2-1b", "2-1b-L": "2-1b", "2-R": "2-R"}   # VariantSpec.stage
ORDER1_TREE_ALGS = ("XGB", "LGB", "CAT")      # A 후보(§6 1-3: 그 W·입력 방식의 트리 설정 중 CV 1위 알고리즘)
HIS_IDS = ("WEATHER", "HIS1", "HIS2", "HIS3", "HIS4")
ANOM_IDS = ("WEATHER", "WX2")
SELECTION_FILE = "2_prep_selection.json"
IMPORTANCE_DIR = "2-R1_importance"
COMPONENTS = {"2-1": ("weather",), "2-1b-T": ("weather",), "2-1b-L": ("weather", "missing"),
              "2-R": ("weather", "livestock")}
N_JOBS = {"2-1": 8, "2-1b-T": 8, "2-1b-L": None, "2-R": 8}   # 2-1b-L 은 설정 값(≤ 8) — GPU 판

# ------------------------------------------------------------------ 설정 schema
DIRS_SCHEMA = {k: cf.Field(str) for k in SUBS}           # --inputs 아래 하위 단계 결과 폴더
LINK_SCHEMA = {
    "sub": cf.Field(str, choices=SUBS),
    "decision_1_3": cf.Field(str),       # --inputs 아래 1-3 결정 json
    "dir_1_1": cf.Field(str),            # --inputs 아래 1-1·1-2 결과 폴더
    "dir_1_2": cf.Field(str),
    "dirs": DIRS_SCHEMA,
}
TRAIN_SCHEMA = {
    **LINK_SCHEMA,
    "model_params": cf.Field(dict),
    "k_values": cf.Field(list, item_types=int, required=False),
    "n_ma_values": cf.Field(list, item_types=int, required=False),
    "research_units": cf.Field(list, item_types=str, required=False),
    "gpu_memory_fraction": cf.Field(float, required=False),
}
BASE_SCHEMA = {
    "plan": cf.Field(str), **LINK_SCHEMA,
    "stage": cf.Field(str), "order": cf.Field(str), "experiment": cf.Field(str),
    "samples": cf.Field(str), "experiment_log": cf.Field(str),
    "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str), "code_reviewed": cf.Field(bool),
}
BASE_SCHEMA = lst_scope.with_field(BASE_SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)
SELECT_SCHEMA = {
    **BASE_SCHEMA,                                      # scope 칸 포함(국내만 버전)
    "incomplete_months": cf.Field(str), "te7a_dir": cf.Field(str),
    "configs": {k: cf.Field(str) for k in TRAIN_SUBS},    # 코드 뿌리 아래 하위 단계 설정(판 목록 다시 만들기)
}


class PrepError(RuntimeError):
    pass


# ================================================================== 1-3 결정
def load_decision(inputs_root: Path, cfg: Mapping) -> dict:
    """1-3 결정 json(W·트리 입력 방식·A·A 판·1차 베이스라인) 읽고 점검."""
    p = resolve(inputs_root, cfg["decision_1_3"])
    if not p.exists():
        raise PrepError(f"1-3 결정이 없음(1-3 을 먼저): {p}")
    d = cf.load_json_strict(p)
    for k in ("baseline", "W", "W_DL", "tree_mode", "A", "A_variant"):
        if k not in d:
            raise PrepError(f"1-3 결정 json 에 {k} 없음")
    if d["W"] not in W_GRID or d["W_DL"] != d["W"]:
        raise PrepError(f"1-3 W·W_DL 이상: {d['W']}·{d['W_DL']}(§6 1-3: W_DL = W)")
    if d["tree_mode"] not in rn.MODES or d["A"] not in ORDER1_TREE_ALGS:
        raise PrepError(f"1-3 트리 입력 방식·A 이상: {d['tree_mode']}·{d['A']}")
    if d["A_variant"] != rn.name_stage1(d["A"], d["W"], d["tree_mode"]):
        raise PrepError(f"A 판 이름이 (A, W, 입력 방식)과 다름: {d['A_variant']}")
    d["_md5"] = cf.file_md5(p)
    return d


# ================================================================== 판 이름·판 설정(§6 단계 2 준비, 부록 A-14)
def name_base(A: str) -> str:
    return f"2-0-{A}"


def name_k(k: int) -> str:
    return f"2-1-k{k:03d}"


def name_nma(part: str, n: int) -> str:
    return f"2-1b-{'T' if part == '2-1b-T' else 'L'}-N{n:02d}"


def name_research(i: int) -> str:
    return f"2-R{i}"


def research_sets() -> list[tuple[str, list[str]]]:
    """2-R1 = 리서치 칼럼 9개 전부, 2-R2 ~ 2-R10 = 앞 8개 각각 + 축사(§6 2-R — 부록 A-10 순서 = registry.RESEARCH_UNITS)."""
    units = list(rg.RESEARCH_UNITS)
    return [(name_research(1), units)] + [(name_research(i + 2), [u]) for i, u in enumerate(units)]


def _check_part_cfg(part: str, cfg: Mapping) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != ORDER or cfg["experiment"] != EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '{ORDER}' · experiment '{EXPERIMENT}'(부록 A-15)")
    if cfg["model_params"]:
        raise cf.ConfigError("단계 2 준비 = 라이브러리 기본 설정 그대로 — model_params 는 {}(§0)")
    need = {"2-1": "k_values", "2-1b-T": "n_ma_values", "2-1b-L": "n_ma_values", "2-R": "research_units"}[part]
    extra = {"k_values", "n_ma_values", "research_units"} - {need}
    if need not in cfg or any(k in cfg for k in extra):
        raise cf.ConfigError(f"{part} 설정 칸: {need} 만(그 밖 {sorted(extra)} 없음)")
    if part == "2-1" and tuple(cfg["k_values"]) != K_VALUES:
        raise cf.ConfigError(f"k_values 는 {list(K_VALUES)}(§6 2-1)")
    if part in ("2-1b-T", "2-1b-L") and tuple(cfg["n_ma_values"]) != rg.N_MA_VALUES:
        raise cf.ConfigError(f"n_ma_values 는 {list(rg.N_MA_VALUES)}(§6 2-1b)")
    if part == "2-R" and tuple(cfg["research_units"]) != rg.RESEARCH_UNITS:
        raise cf.ConfigError(f"research_units 는 {list(rg.RESEARCH_UNITS)}(§6 2-R·부록 A-10)")
    if part == "2-1b-L":
        f = cfg.get("gpu_memory_fraction")
        if f is None or not (0 < f <= GPU_MEMORY_FRACTION):
            raise cf.ConfigError(f"2-1b-L gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    elif "gpu_memory_fraction" in cfg:
        raise cf.ConfigError("gpu_memory_fraction 은 2-1b-L 만")
    if N_JOBS[part] is not None and cfg["n_jobs"] != N_JOBS[part]:
        raise cf.ConfigError(f"{part} n_jobs 는 {N_JOBS[part]}")


def part_specs(part: str, cfg: Mapping, dec: Mapping) -> list[rn.VariantSpec]:
    """하위 단계 판 목록(§6 단계 2 준비). dec = 1-3 결정."""
    _check_part_cfg(part, cfg)
    W, mode, A = int(dec["W"]), dec["tree_mode"], dec["A"]
    com = dict(order=ORDER, role=ROLE, train_count=4, params={}, judgement=JUDGEMENT)
    if part == "2-1":
        out = [rn.VariantSpec(name=name_k(k), stage="2-1", kind="tree", algorithm=A, W=W, mode=mode,
                              input_set=f"2-1-his-k{k:03d}", **com,
                              extra={"version": {"include_ids": list(HIS_IDS), "k": k}})
               for k in cfg["k_values"]]
    elif part == "2-1b-T":
        out = [rn.VariantSpec(name=name_nma(part, n), stage="2-1b", kind="tree", algorithm=A, W=W, mode=mode,
                              input_set=f"2-1b-anom-tree-N{n:02d}", **com,
                              extra={"version": {"include_ids": list(ANOM_IDS), "n_ma": n}})
               for n in cfg["n_ma_values"]]
    elif part == "2-1b-L":
        W_DL = int(dec["W_DL"])
        out = [rn.VariantSpec(name=name_nma(part, n), stage="2-1b", kind="dl", algorithm="LSTM", W=W_DL, mode=None,
                              input_set=f"2-1b-anom-dl-N{n:02d}", **com,
                              extra={"version": {"include_ids": list(ANOM_IDS), "n_ma_dl": n}})
               for n in cfg["n_ma_values"]]
    elif part == "2-R":
        out = [rn.VariantSpec(name=nm, stage="2-R", kind="tree", algorithm=A, W=W, mode=mode,
                              input_set="2-R-research-" + ("all" if len(us) > 1 else us[0]), **com,
                              extra={"version": {"include_ids": ["WEATHER"], "include_units": list(us)}})
               for nm, us in research_sets()]
    else:
        raise ValueError(f"학습 하위 단계 아님: {part}")
    n_exp = {"2-1": 5, "2-1b-T": 4, "2-1b-L": 4, "2-R": 10}[part]
    if len(out) != n_exp:
        raise AssertionError(f"{part} 판 수 {len(out)} ≠ {n_exp}(부록 A-9)")
    from lst import wiring as wr

    for s in out:                     # 판 설정이 등록부 규칙에 맞는지 미리(모르는 단위·k·N_ma 면 멈춤)
        wr.version_spec(s)
    return out


def chosen_value(part: str, spec: rn.VariantSpec) -> int:
    v = spec.extra["version"]
    return int({"2-1": v.get("k"), "2-1b-T": v.get("n_ma"), "2-1b-L": v.get("n_ma_dl")}[part])


# ================================================================== 공통 도움
def _pre_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    for a in ("--root", "--out", "--inputs", "--config", "--variant", "--only", "--reason"):
        ap.add_argument(a)
    for a in ("--list", "--report", "--verify-run"):
        ap.add_argument(a, action="store_true")
    ns, _ = ap.parse_known_args(list(argv))
    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)
    return ns


def _check_out(rp_out: Path, inputs_root: Path, cfg: Mapping, part: str) -> None:
    want = resolve(inputs_root, cfg["dirs"][part]).resolve()
    if Path(rp_out).resolve() != want:
        raise cf.ConfigError(f"--out 은 --inputs 아래 dirs[{part}] = {want} 이어야 함(다음 하위 단계가 그 자리를 읽음)")


class PartTracker:
    """Tracker 감싸기: 단계 run '2' 의 표 artifact('tables')를 'tables/<하위 단계>' 로(구현 세부 5). 나머지는 그대로."""

    def __init__(self, tracker, part: str):
        self._tr, self._part = tracker, part

    def __getattr__(self, k):
        return getattr(self._tr, k)

    def log_artifact(self, run_id: str, path, artifact_path: str | None = None) -> None:
        if artifact_path == "tables":
            artifact_path = f"tables/{self._part}"
        return self._tr.log_artifact(run_id, path, artifact_path)


def _tracker(cfg: Mapping, part: str, tracker_fn: Callable[[dict], Any]):
    t = tracker_fn(cfg)
    return None if t is None else PartTracker(t, part)


def _src_dir(inputs_root: Path, cfg: Mapping, name: str) -> Path:
    """1차 판 폴더(LSTM 판 = 1-2, 트리 = 1-1)."""
    d = cfg["dir_1_2"] if name.startswith("1-LSTM-") else cfg["dir_1_1"]
    return resolve(inputs_root, d)


SOURCE_FILES = tuple(f"pred_{s}.parquet" for s in rn.SPLIT_ORDER) + ("cv_scores.json",)


def _files_md5(vdir: Path) -> dict[str, str]:
    miss = [f for f in SOURCE_FILES if not (vdir / f).exists()]
    if miss:
        raise PrepError(f"1차 판 파일 없음: {vdir.name} {miss}")
    return {f: cf.file_md5(vdir / f) for f in SOURCE_FILES}


# ================================================================== 2-0(학습 없음)
def base_spec(dec: Mapping, src_spec: Mapping, src_rel: str) -> rn.VariantSpec:
    return rn.VariantSpec(name=name_base(dec["A"]), stage="2-0", order=ORDER, kind="tree", algorithm=dec["A"],
                          W=int(dec["W"]), mode=dec["tree_mode"], input_set=src_spec["input_set"], role=ROLE,
                          train_count=int(src_spec["train_count"]), params=dict(src_spec["params"]),
                          judgement=JUDGEMENT, base_variant=dec["A_variant"],
                          extra={**dict(src_spec["extra"]), "source": {"variant": dec["A_variant"], "dir": src_rel}})


def run_base(argv: Sequence[str], *, tracker_fn: Callable[[dict], Any] = sc.make_tracker) -> dict:
    """2-0: 1차 A 판 그대로(다시 학습 안 함) — 점수 사본·예측 위치·MLflow 판 run(구현 세부 4)."""
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, BASE_SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    if cfg["sub"] != "2-0" or cfg["stage"] != STAGE or cfg["experiment"] != EXPERIMENT or cfg["order"] != ORDER:
        raise cf.ConfigError("2-0 설정: sub 2-0 · stage 2 · order 2차 · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    _check_out(rp.out, inputs_root, cfg, "2-0")
    dec = load_decision(inputs_root, cfg)
    d11 = resolve(inputs_root, cfg["dir_1_1"])
    src_spec, src_done, src_cv = rn.load_done(d11, dec["A_variant"])
    if (src_spec["algorithm"], src_spec["W"], src_spec["mode"]) != (dec["A"], dec["W"], dec["tree_mode"]):
        raise PrepError("1차 A 판 설정이 1-3 결정과 다름")
    src_rel = f"{cfg['dir_1_1'].rstrip('/')}/{dec['A_variant']}"
    src = d11 / dec["A_variant"]
    spec = base_spec(dec, src_spec, src_rel)
    code_md5 = rn.snapshot_code(CODE_ROOT, rp.out)
    vdir = rn.variant_dir(rp.out, spec.name)
    md5s = _files_md5(src)
    if (vdir / "done.json").exists():
        d = rn.read_json(vdir / "done.json")
        if d["spec_md5"] != spec.md5() or rn.read_json(vdir / "source.json")["files_md5"] != md5s:
            raise rn.ResumeError(f"{spec.name}: 끝난 2-0 의 설정·1차 파일이 지금과 다름 — 지휘자 결정")
        print(f"{spec.name}: 이미 끝남(건너뜀)")
        return d
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    if vdir.exists():                       # 멈춘 2-0 — 옮기고 다시(B-1)
        import datetime as _dt

        dst = rp.out / "_incomplete" / f"{vdir.name}__{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(vdir, dst)
        log.append("rerun", stage="2-0", name=spec.name, reason="멈춘 2-0 다시", moved_to=dst.name)
    vdir.mkdir(parents=True)
    cf.dump_json(vdir / "spec.json", spec.to_dict())
    source = {"variant": dec["A_variant"], "dir": src_rel, "files_md5": md5s,
              "mlflow_run_id": src_done.get("mlflow_run_id"), "selection_count": src_done["selection_count"],
              "decision_1_3_md5": dec["_md5"],
              "pred_location": {s: f"{src_rel}/pred_{s}.parquet" for s in rn.SPLIT_ORDER}}
    cf.dump_json(vdir / "source.json", source)
    shutil.copyfile(src / "cv_scores.json", vdir / "cv_scores.json")
    if (src / rn.H_METRICS_FILE).exists():
        shutil.copyfile(src / rn.H_METRICS_FILE, vdir / rn.H_METRICS_FILE)
    tr = _tracker(cfg, "2-0", tracker_fn)
    run_id = None
    if tr is not None:
        sr = tr.stage_run(EXPERIMENT, STAGE)
        params = {}
        if src_done.get("mlflow_run_id"):
            params.update(tr.client.get_run(src_done["mlflow_run_id"]).data.params)
        params.update({"source_variant": dec["A_variant"], "source_run_id": src_done.get("mlflow_run_id"),
                       "pred_location": f"--inputs/{src_rel}/pred_<split>.parquet"})
        tags = {"단계": "2-0", "역할": ROLE, "바탕 판": dec["A_variant"],
                "코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오", "설정 md5": spec.md5(), "code_md5": code_md5}
        run_id = tr.start_variant(EXPERIMENT, sr, spec.name, tags, params)
        tr.set_judgement(run_id, JUDGEMENT)
        tr.log_cv(run_id, src_cv, selection_count=src_done["selection_count"])
        for f in ("spec.json", "source.json", "cv_scores.json", rn.H_METRICS_FILE):
            if (vdir / f).exists():
                tr.log_artifact(run_id, vdir / f)
        tr.finish(run_id)
    done = {"name": spec.name, "stage": "2-0", "spec_md5": spec.md5(), "code_md5": code_md5,
            "cv_band_weighted": src_cv["cv_band_weighted"], "many_ties": src_cv["ties"]["many_ties"],
            "selection_count": src_done["selection_count"], "mlflow_run_id": run_id, "copied_from": src_rel}
    cf.dump_scores_json(vdir / "done.json", done)
    log.append("copied", stage="2-0", name=spec.name, source=src_rel, cv=src_cv["cv_band_weighted"])
    print(f"{spec.name} = {dec['A_variant']} 그대로(학습 없음) CV {src_cv['cv_band_weighted']}")
    return done


# ================================================================== 학습 하위 단계(2-1·2-1b-T·2-1b-L·2-R)
def default_factories(part: str):
    def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
        from lst import wiring as wr

        dl_kw = {}
        if part == "2-1b-L" and "smoke" in cfg and "max_epochs" in cfg["smoke"]:
            dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]}
        w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS[part], dl_kwargs=dl_kw)
        return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                            input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)
    return factories


def _before_variant(cfg: dict) -> None:
    if cfg["sub"] == "2-1b-L":
        limit_gpu_memory(cfg["gpu_memory_fraction"])


def run_train(argv: Sequence[str], part: str, *, factories_fn=None, tracker_fn=sc.make_tracker,
              check_resources_fn=None, before_variant_fn=_before_variant, nodes_fn=None,
              importance_M: int | None = None):
    """하위 단계 판 실행(stage_cli 그대로 — 판마다 하위 프로세스·이어하기·CV 표). 2-R 기본 실행 끝에 2-R1 중요도.
    factories_fn·tracker_fn·check_resources_fn·before_variant_fn·nodes_fn·importance_M = 시험에서만 바꿈."""
    ns = _pre_args(argv)
    if ns.report:
        raise SystemExit("보고는 configs/s2_select.json 으로 한 번에(구현 세부 6)")
    if not ns.inputs or not ns.config:
        raise SystemExit("--inputs·--config 필요")
    inputs_root = Path(ns.inputs).expanduser().resolve()
    holder: dict = {}

    def specs_fn(cfg):
        if "dec" not in holder:
            holder["dec"] = load_decision(inputs_root, cfg)
        return part_specs(part, cfg, holder["dec"])

    schema = dict(TRAIN_SCHEMA)
    if not (ns.list or ns.verify_run):       # 코드 바뀜 확인 다시 돌리기는 임시 폴더에
        _check_out(Path(ns.out).expanduser(), inputs_root, cf.load_json_strict(ns.config), part)
    res = sc.stage_main(argv, script=Path(__file__), description=__doc__.splitlines()[0], schema_extra=schema,
                        specs_fn=specs_fn, factories_fn=factories_fn or default_factories(part),
                        before_variant=before_variant_fn,
                        tracker_fn=lambda cfg: _tracker(cfg, part, tracker_fn), check_resources=check_resources_fn)
    default_mode = not (ns.list or ns.variant or ns.verify_run or ns.only)
    if part == "2-R" and default_mode:
        cfg = sc.load_cfg(ns.config, schema)
        kw = {} if importance_M is None else {"M": importance_M}
        run_2r1_importance(cfg, Path(ns.root).expanduser().resolve(), inputs_root, Path(ns.out).expanduser().resolve(),
                           specs_fn(cfg), nodes_fn=nodes_fn, tracker=_tracker(cfg, part, tracker_fn), **kw)
    return res


# ------------------------------------------------------------------ 2-R1 중요도(부록 A-5 계산법, 학습 없음)
def default_nodes(cfg: Mapping, root: Path, inputs_root: Path, vdir: Path, spec: rn.VariantSpec) -> dict:
    """2-R1 fold 모델 3개(저장 모델) + 입력 조립 객체(트리, 로지스틱 아님 — fit 이 값을 안 만듦) → importance.Leaf."""
    from lst import importance as im
    from lst import inputs as ip
    from lst import models_tree as mt
    from lst import wiring as wr

    if spec.algorithm not in ORDER1_TREE_ALGS:
        raise PrepError(f"2-R 알고리즘 {spec.algorithm} — A(XGB·LGB·CAT)만")
    ctx = ip.load_context(cfg["context"], root, inputs_root, components=list(COMPONENTS["2-R"]))
    vs = wr.version_spec(spec)
    nodes = {}
    for f in ev.CV_FOLDS:
        asm = ip.TreeInputs(ctx, vs, f)
        saved = rn.read_json(vdir / f"columns_{f}.json")["columns"]
        if list(asm.columns) != list(saved):
            raise PrepError(f"{spec.name} {f}: 저장 칼럼과 지금 조립 칼럼이 다름")
        nodes[f] = im.Leaf(f"{spec.name}-{f}", asm, mt.TreeModel.load(vdir / f"model_{f}"), n_iter=None)
    return nodes


def run_2r1_importance(cfg: Mapping, root: Path, inputs_root: Path, out: Path, specs: Sequence[rn.VariantSpec], *,
                       nodes_fn=None, tracker=None, M: int | None = None,
                       check: Callable[[], Any] | None = None) -> Path:
    """§6 2-R: 2-R1 에서 칼럼별(9)·리서치 변수 전체 묶음·각 변수가 속한 기상 그룹 섞기 중요도(학습한 트리 전부로 예측 —
    교차 N 없음). 결과 = <out>/2-R1_importance/(이미 있으면 건너뜀)."""
    from lst import importance as im

    sp = next(s for s in specs if s.name == name_research(1))
    vdir = rn.variant_dir(out, sp.name)
    dst = out / IMPORTANCE_DIR
    if (dst / "importance_meta.json").exists():
        print(f"2-R1 중요도 이미 있음: {dst}")
        return dst
    if dst.exists():                       # 멈춘 중요도 계산 — 옮기고 다시(B-1)
        import datetime as _dt

        inc = out / "_incomplete"
        inc.mkdir(parents=True, exist_ok=True)
        os.replace(dst, inc / f"{IMPORTANCE_DIR}__{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    _, done, cv = rn.load_done(out, sp.name)
    nodes = (nodes_fn or default_nodes)(cfg, root, inputs_root, vdir, sp)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    M = im.M_DEFAULT if M is None else int(M)
    res = im.permutation_importance(nodes, samples, im.targets_2r1(), M=M,
                                    check=check or (lambda: check_resources(out)),
                                    expect_base={f: float(cv["fold_band_weighted"][f]) for f in ev.CV_FOLDS})
    res.meta.update({"variant": sp.name, "n_iter": None, "note": "2-R = 학습한 트리 전부로 예측(교차 N 없음, §6 2-R)"})
    paths = res.save(dst)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    log.append("importance", stage="2-R", name=sp.name, M=M, n_targets=len(res.summary),
               seconds=round(sum(res.seconds.values()), 2))
    if tracker is not None and done.get("mlflow_run_id"):
        for p in paths:
            tracker.log_artifact(done["mlflow_run_id"], p, "importance")
    print(f"2-R1 중요도: {dst}")
    return dst


# ================================================================== 고르기(CV 만)
def _sub_cfg(sel_cfg: Mapping, part: str) -> dict:
    p = CODE_ROOT / sel_cfg["configs"][part]
    cfg = sc.load_cfg(p, TRAIN_SCHEMA)
    if cfg["sub"] != part:
        raise cf.ConfigError(f"{p.name} 의 sub {cfg['sub']} ≠ {part}")
    for k in ("decision_1_3", "dir_1_1", "dir_1_2", "dirs", "samples", "experiment_log"):
        if cfg[k] != sel_cfg[k]:
            raise cf.ConfigError(f"{p.name} 의 {k} 가 고르기 설정과 다름")
    return cfg


def choose(cands: Sequence[ev.Candidate], rescore) -> dict:
    """§1-2 고르기: CV(소수 4자리) 큰 순 → 판의 학습 횟수 적은 순 → 판 이름 순, 동점 많은 판 다시 채점 기록(부록 A-12).
    test 칸이 들어오면 멈춤(§1-3)."""
    with ev.selection_guard():
        return ev.select_with_tie_check(list(cands), rescore)


def run_select(argv: Sequence[str], *, tracker_fn: Callable[[dict], Any] = sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reason")
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, SELECT_SCHEMA))
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    if cfg["sub"] != "select" or cfg["stage"] != STAGE or cfg["experiment"] != EXPERIMENT or cfg["order"] != ORDER:
        raise cf.ConfigError("고르기 설정: sub select · stage 2 · order 2차 · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    _check_out(rp.out, inputs_root, cfg, "select")
    if args.report:
        if not args.reason:
            raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
        return report_prep(cfg, rp, inputs_root, args.reason, tracker_fn=tracker_fn)
    sel_path = rp.out / SELECTION_FILE
    if sel_path.exists():
        raise FileExistsError(f"이미 고름: {sel_path}")
    dec = load_decision(inputs_root, cfg)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    result: dict = {"W": dec["W"], "W_DL": dec["W_DL"], "tree_mode": dec["tree_mode"], "A": dec["A"],
                    "A_variant": dec["A_variant"], "baseline_1": dec["baseline"], "decision_1_3_md5": dec["_md5"],
                    "base_2_0": name_base(dec["A"])}
    keys = {"2-1": "k", "2-1b-T": "n_ma", "2-1b-L": "n_ma_dl"}
    labels = {"2-1": "k", "2-1b-T": "N_ma(트리)", "2-1b-L": "N_ma(딥러닝)"}
    picks: dict[str, tuple[str, str]] = {}
    alerts: list[str] = []
    for part, key in keys.items():
        scfg = _sub_cfg(cfg, part)
        specs = part_specs(part, scfg, dec)
        d = resolve(inputs_root, cfg["dirs"][part])
        cands = rn.candidates(d, [s.name for s in specs])      # 끝나지 않은 판이 있으면 멈춤
        s = choose(cands, rn.rescore_fn(d, samples))
        sp = next(x for x in specs if x.name == s["selected"])
        result[key] = chosen_value(part, sp)
        result[f"{key}_variant"] = sp.name
        result[f"select_{part}"] = s
        picks[part] = (sp.name, f"{labels[part]} = {result[key]}")
        alerts += s["alerts"]
    result["alerts"] = alerts
    cf.dump_scores_json(sel_path, result)
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    log.append("decision", stage="2", k=result["k"], n_ma=result["n_ma"], n_ma_dl=result["n_ma_dl"], alerts=alerts)
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(EXPERIMENT, STAGE)
        for part, (name, val) in picks.items():
            vr = tr.find_run(EXPERIMENT, name, sr)
            if vr is None:
                raise PrepError(f"MLflow 판 run 없음: {name}")
            tr.set_tags(vr.info.run_id, {"선택": val})
            for n, v in result[f"select_{part}"]["rescored"].items():
                r = tr.find_run(EXPERIMENT, n, sr)
                if r is not None:
                    tr.log_rescore(r.info.run_id, v)
        tr.log_artifact(sr, sel_path, "selection")
    print(f"k {result['k']} · 트리 N_ma {result['n_ma']} · 딥러닝 N_ma {result['n_ma_dl']} · 알림 {alerts}")
    return result


# ================================================================== 보고(test 채점 — 단계가 끝난 뒤 한 번)
def _base_row(inputs_root: Path, cfg: Mapping, name: str, desc: str, run_id: str | None = None,
              source_dir: Path | None = None, row_name: str | None = None) -> rn.BaseRow:
    vdir = (source_dir or _src_dir(inputs_root, cfg, name)) / name
    done = rn.read_json(vdir / "done.json")
    return rn.BaseRow(row_name or name, desc, rn.load_preds(vdir, ev.CV_FOLDS),
                      pd.read_parquet(rn.pred_path(vdir, "refit")), int(done["selection_count"]), run_id)


def report_prep(cfg: Mapping, rp, inputs_root: Path, reason: str, *,
                tracker_fn: Callable[[dict], Any] = sc.make_tracker) -> dict:
    """단계 2 준비 보고(§1-3): 고르기 뒤 한 번, 하위 단계마다 표(부록 A-12) — 기준 줄 7-a·1차 베이스라인·비교 기준 판."""
    sel_path = rp.out / SELECTION_FILE
    if not sel_path.exists():
        raise SystemExit("고르기를 먼저(test 는 고른 뒤에만 봄 — §1-3)")
    sel = rn.read_json(sel_path)
    done_tables = [resolve(inputs_root, cfg["dirs"][p]) / "stage_table_report.csv" for p in TRAIN_SUBS]
    if any(t.exists() for t in done_tables):
        raise FileExistsError(f"이미 보고함(보고는 한 번): {[str(t) for t in done_tables if t.exists()]}")
    dec = load_decision(inputs_root, cfg)
    if dec["_md5"] != sel["decision_1_3_md5"]:
        raise PrepError("1-3 결정 파일이 고르기 때와 다름")
    names = {}
    for part in TRAIN_SUBS:
        specs = part_specs(part, _sub_cfg(cfg, part), dec)
        names[part] = [s.name for s in specs]
        d = resolve(inputs_root, cfg["dirs"][part])
        for n in names[part]:
            rn.load_done(d, n)                 # 하위 단계가 다 끝났는지(구현 세부 6)
    d20 = resolve(inputs_root, cfg["dirs"]["2-0"])
    b20 = name_base(dec["A"])
    src = rn.read_json(d20 / b20 / "source.json")
    if _files_md5(resolve(inputs_root, src["dir"])) != src["files_md5"]:
        raise PrepError("2-0 이 가리키는 1차 판 파일이 2-0 때와 다름")
    d20_done = rn.read_json(d20 / b20 / "done.json")
    permit = ev.report_permit(STAGE, reason)
    samples_path = resolve(inputs_root, cfg["samples"])
    samples = ev.SampleTable.read(samples_path)
    inc = ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"]))
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    te7a = rn.te7a_baseline(resolve(inputs_root, cfg["te7a_dir"]), samples, log)   # test 지표는 0-10·1-1 보고 run 에 이미
    base_name = dec["baseline"]
    lstm_ref = rn.name_stage1("LSTM", int(dec["W_DL"]))
    out = {}
    logged_20 = False
    for part in TRAIN_SUBS:
        tr = _tracker(cfg, part, tracker_fn)
        sr = tr.stage_run(EXPERIMENT, STAGE) if tr is not None else None
        rows = [te7a, _base_row(inputs_root, cfg, base_name, "1차 베이스라인(1차 70 설정 중 CV 1위) — 비교 줄")]
        notes = []
        if part == "2-1b-L":
            if lstm_ref != base_name:
                rows.append(_base_row(inputs_root, cfg, lstm_ref, "1차 W_DL LSTM(2-1b LSTM 의 바탕, 차감 채널 없음)"))
        else:
            rid = d20_done.get("mlflow_run_id") if not logged_20 else None   # 2-0 test 지표는 한 번만
            logged_20 = True
            rows.append(_base_row(inputs_root, cfg, dec["A_variant"], "2-0 기준 판 = 1차 A 판 그대로(다시 학습 안 함)",
                                  run_id=rid, row_name=b20))
        if any("CAT" in r.name.split("-") for r in rows[1:]):     # table_row 의 표시 칸은 판 폴더 줄에만 붙음
            notes.append(f"1차 베이스라인·비교 기준 줄 중 CatBoost 판: {ev.CAT_AUTO_LR_MARK}")
        p = rn.report_stage(out_dir=resolve(inputs_root, cfg["dirs"][part]), stage=SUB_STAGE[part], names=names[part],
                            samples=samples, samples_path=samples_path, permit=permit, incomplete=inc, exp_log=log,
                            tracker=tr, stage_run_id=sr, base_rows=rows, extra_notes=notes)
        out[part] = str(p)
        if part == "2-R":
            out["2-R_vs_2-0"] = str(research_delta(p, b20, tr, sr))
    print(f"단계 2 준비 보고 표: {out}")
    return out


def research_delta(table: Path, base: str, tracker=None, stage_run_id=None) -> Path:
    """2-R 판마다 2-0 대비 CV·test 구간 가중 변화(부록 A-10 '기준 대비 CV·test 구간 가중 PR-AUC 변화')."""
    t = pd.read_csv(table, encoding="utf-8-sig")
    b = t[t["단계·판"] == base]
    if len(b) != 1:
        raise PrepError(f"2-R 표에 기준 판 {base} 줄이 없음")
    b = b.iloc[0]
    r = t[t["단계·판"].astype(str).str.match(r"^2-R\d+$")].copy()
    units = dict(research_sets())
    out = pd.DataFrame({
        "판": r["단계·판"], "더한 리서치 칼럼": r["단계·판"].map(lambda n: "·".join(units[n])),
        "CV 구간 가중": r["CV 구간 가중(평균)"], "CV 변화(2-0 대비)": r["CV 구간 가중(평균)"] - b["CV 구간 가중(평균)"],
        "test 구간 가중": r["test 구간 가중"], "test 변화(2-0 대비)": r["test 구간 가중"] - b["test 구간 가중"],
    })
    p = table.with_name("2-R_vs_2-0.csv")
    with open(p, "x", newline="", encoding="utf-8-sig") as fh:
        out.to_csv(fh, index=False)
    if tracker is not None and stage_run_id is not None:
        tracker.log_artifact(stage_run_id, p, "tables")
    return p


# ================================================================== 진입점
def main(argv=None, **inject):
    argv = list(sys.argv[1:] if argv is None else argv)
    ns = _pre_args(argv)
    if not ns.config:
        raise SystemExit("--config 필요")
    sub = cf.load_json_strict(ns.config).get("sub")
    if sub not in SUBS:
        raise cf.ConfigError(f"설정 sub 는 {SUBS} 중: {sub!r}")
    if sub == "2-0":
        return run_base(argv, **{k: v for k, v in inject.items() if k == "tracker_fn"})
    if sub == "select":
        return run_select(argv, **{k: v for k, v in inject.items() if k == "tracker_fn"})
    return run_train(argv, sub, **inject)


if __name__ == "__main__":
    main()
