"""단계 스크립트 공통 실행 틀(1-1·1-2 와 이후 단계) — 계획서 §8, 부록 A-15, B-1, B-2.

모드(한 스크립트):
  (기본)                 판 전부를 하위 프로세스로 차례로(끝난 판 건너뜀, 멈추면 전체 멈춤) → CV 단계 표
  --variant <판 이름>    판 하나(하위 프로세스 안에서 도는 쪽)
  --list                 판 이름 목록만 출력
  --report --reason "…"  단계 보고: test 정답 채점(보고 허가) → 단계 표(부록 A-12) + MLflow test 지표
공통 인자: --root(자료 뿌리)·--out(단계 결과 폴더)·--inputs(결과 뿌리)·--config(JSON).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import config as cf
from . import evaluate as ev
from . import runner as rn
from . import scope as scope_
from .paths import build_parser, parse_paths, resolve

COMMON_SCHEMA = {
    "plan": cf.Field(str),
    "stage": cf.Field(str),
    "order": cf.Field(str, choices=("준비", "1차", "2차")),
    "experiment": cf.Field(str, choices=("listeria-준비", "listeria-1차", "listeria-2차")),
    "samples": cf.Field(str),                 # --inputs 아래(단계 0-1 결과)
    "incomplete_months": cf.Field(str),       # --inputs 아래(단계 0-1 덜 찬 달)
    "experiment_log": cf.Field(str),          # --inputs 아래 jsonl
    "te7a_dir": cf.Field(str, required=False),  # 보고 때 기준 줄(7-a) — --inputs 아래 0-10 결과 폴더
    "n_jobs": cf.Field(int),
    "seed": cf.Field(int),
    "use_mlflow": cf.Field(bool),
    "mlflow_uri": cf.Field(str),
    "code_reviewed": cf.Field(bool),
    "log_pred_artifacts": cf.Field(bool),
    "context": cf.Field(dict),                # inputs.load_context 설정(CONTEXT_SCHEMA — 그쪽이 검사)
    "scope": scope_.FIELD,                    # 국내만 버전(configs_dom — lst/scope.py), 없으면 지금 판
    "smoke": {                                # 연기 시험만(실제 실행 설정엔 없음)
        "max_train_rows": cf.Field(int),
        "max_pairs_per_h": cf.Field(int),
        "max_epochs": cf.Field(int, required=False),
        "skip_7a_check": cf.Field(bool, required=False),   # 0-10 기록 확인 끄기(연기 시험만)
    },
}
COMMON_SCHEMA["accept_previous_code"] = {   # 단계 도중 코드가 바뀔 때: 같은 결과 확인 뒤 이전 코드로 끝난 판 인정
    "code_md5": cf.Field(str),               # 단계 코드 사본(_code)의 이전 code_md5
    "reason": cf.Field(str),
    "check_variant": cf.Field(str),          # 새 코드로 다시 돌려 바이트 비교할 판(그 판이 든 묶음 전체를 비교)
}
OPTIONAL_SUB = ("smoke", "accept_previous_code")
SMOKE_ONLY_URI = ("sqlite:",)   # 연기 시험 설정은 서버 MLflow 에 못 씀


@dataclass
class Factories:
    make_inputs: rn.MakeInputs
    make_model: rn.MakeModel
    input_md5s: Mapping[str, str]                 # 입력 조립이 읽는 파일 md5(B-2)
    fit_kwargs: rn.FitKwargs | None = None


def parser(description: str):
    ap = build_parser(description)
    ap.add_argument("--inputs", required=True, help="결과 뿌리(단계 0 결과·실험 로그) — 서버 ~/cyj_listeria_final/results")
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--variant", help="판 하나 또는 같은 입력 묶음(쉼표 구분) — 하위 프로세스용")
    g.add_argument("--list", action="store_true")
    g.add_argument("--report", action="store_true", help="단계 보고(test 정답 채점) — 단계가 끝난 뒤만")
    ap.add_argument("--reason", help="--report 이유(실험 로그에 남김)")
    ap.add_argument("--verify-run", action="store_true",
                    help="(내부) 코드 바뀜 확인용 다시 돌리기 — MLflow·공용 실험 로그에 안 씀")
    ap.add_argument("--only", help="판 이름 일부만(쉼표 구분) — 기본·--report 모드(단계 표도 그 판만)")
    return ap


def load_cfg(path, schema_extra: Mapping[str, Any]) -> dict:
    schema = {**COMMON_SCHEMA, **schema_extra}
    raw = cf.load_json_strict(path)
    schema = {k: v for k, v in schema.items() if k not in OPTIONAL_SUB or k in raw}
    cfg = dict(cf.validate(raw, schema))
    scope_.activate(cfg.get("scope"))         # 범위·결과 뿌리 표지 대조(실행 인자는 parse_paths·_pre 가 받아 둠)
    if "smoke" in cfg and cfg["use_mlflow"] and not cfg["mlflow_uri"].startswith(SMOKE_ONLY_URI):
        raise cf.ConfigError("연기 시험(smoke) 설정은 로컬 sqlite MLflow 에만")
    from .resources import check_n_jobs

    check_n_jobs(cfg["n_jobs"])
    if cfg["seed"] != 42:
        raise cf.ConfigError("씨앗은 42(§8·부록 B-5)")
    return cfg


def make_tracker(cfg):
    if not cfg["use_mlflow"]:
        return None
    from .tracking import Tracker

    uri = cfg["mlflow_uri"]
    if uri.startswith("sqlite:///"):   # 연기 시험: artifact 를 db 옆 artifacts/ 에(작업 폴더 mlruns 에 안 쌓이게)
        db = Path(uri[len("sqlite:///"):])
        return Tracker(uri, artifact_root=(db.parent / "artifacts").resolve().as_uri())
    return Tracker(uri)


def stage_main(argv: Sequence[str] | None, *, script: Path, description: str, schema_extra: Mapping[str, Any],
               specs_fn: Callable[[dict], list[rn.VariantSpec]],
               factories_fn: Callable[[dict, Any, Path], Factories],
               before_variant: Callable[[dict], Any] | None = None,
               fold_recall: bool = False, tracker_fn: Callable[[dict], Any] = make_tracker,
               check_resources: Callable | None = None,
               report_rows_fn: Callable[[dict, Path, ev.SampleTable, rn.ExperimentLog], list] | None = None,
               report_notes: Sequence[str] = ()) -> Any:
    """report_rows_fn = 보고 표에 더할 비교 줄(rn.BaseRow — 1차 베이스라인·현재 최고 판 등, 7-a 기준 줄 다음),
    report_notes = 보고 표 주석에 더할 문장."""
    ap = parser(description)
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config, schema_extra)
    specs = specs_fn(cfg)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("판 이름 중복")
    if args.only:
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in want if n not in names]
        if unknown:
            raise KeyError(f"이 단계에 없는 판: {unknown}")
        specs = [s for s in specs if s.name in want]
        names = [s.name for s in specs]
    if args.list:
        print("\n".join(names))
        return names
    inputs_root = Path(args.inputs).expanduser().resolve()
    samples_path = resolve(inputs_root, cfg["samples"])
    if args.verify_run:   # 임시 폴더에서만: 공용 실험 로그·MLflow 에 안 남김
        exp_log = rn.ExperimentLog(rp.out / "_verify_experiment_log.jsonl")
        tracker = None
    else:
        exp_log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
        tracker = tracker_fn(cfg)
    stage_run_id = tracker.stage_run(cfg["experiment"], cfg["stage"]) if tracker is not None else None

    if args.variant:
        by = {s.name: s for s in specs}
        want = [n.strip() for n in args.variant.split(",") if n.strip()]
        unknown = [n for n in want if n not in by]
        if unknown:
            raise KeyError(f"이 단계에 없는 판: {unknown}")
        group = [by[n] for n in want]
        if before_variant is not None:
            before_variant(cfg)
        fac = factories_fn(cfg, rp, inputs_root)
        md5s = {**cf.inputs_md5(inputs_root, [cfg["samples"], cfg["incomplete_months"]]), **dict(fac.input_md5s)}
        kw = {} if check_resources is None else {"check_resources": check_resources}
        ctx = rn.RunContext(samples=ev.SampleTable.read(samples_path), out_dir=rp.out, code_root=script.resolve().parents[1],
                            input_md5s=md5s, exp_log=exp_log, make_inputs=fac.make_inputs, make_model=fac.make_model,
                            n_jobs=cfg["n_jobs"], seed=cfg["seed"], tracker=tracker, experiment=cfg["experiment"],
                            stage_run_id=stage_run_id, code_reviewed=cfg["code_reviewed"], fit_kwargs=fac.fit_kwargs,
                            log_pred_artifacts=cfg["log_pred_artifacts"],
                            smoke=({k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None),
                            require_7a_logged=(not args.verify_run
                                               and not cfg.get("smoke", {}).get("skip_7a_check", False)), **kw)
        res = rn.run_group(group, ctx)
        for sp, r in zip(group, res):
            print(f"{sp.name}: {r['status']} CV {r.get('cv_band_weighted')}")
        return res if len(res) > 1 else res[0]

    if args.report:
        if not args.reason:
            raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
        permit = ev.report_permit(cfg["stage"], args.reason)
        samples = ev.SampleTable.read(samples_path)
        base = []
        if cfg.get("te7a_dir"):
            b = rn.te7a_baseline(resolve(inputs_root, cfg["te7a_dir"]), samples, exp_log)
            if tracker is not None:
                sr = tracker.find_run("listeria-준비", "0-10", None)
                vr = tracker.find_run("listeria-준비", b.name, sr.info.run_id) if sr is not None else None
                b.mlflow_run_id = vr.info.run_id if vr is not None else None
            base.append(b)
        if report_rows_fn is not None:
            base += list(report_rows_fn(cfg, inputs_root, samples, exp_log))
        p = rn.report_stage(out_dir=rp.out, stage=cfg["stage"], names=names, samples=samples, samples_path=samples_path,
                            permit=permit, incomplete=ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"])),
                            exp_log=exp_log, tracker=tracker, stage_run_id=stage_run_id, base_rows=base,
                            fold_recall=fold_recall, extra_notes=tuple(report_notes))
        print(f"보고 표: {p}")
        return p

    # 기본: 판 전부(하위 프로세스)
    if not cfg.get("smoke", {}).get("skip_7a_check", False):
        rn.check_prior_7a(exp_log)
    code_root = script.resolve().parents[1]
    groups = rn.group_specs(specs)

    def argv_for(out):
        return ["--root", str(rp.root), "--out", str(out), "--inputs", str(inputs_root),
                "--config", str(Path(args.config).resolve())]

    if "accept_previous_code" in cfg:
        acc = cfg["accept_previous_code"]
        g = next((g for g in rn.group_specs(specs_fn(cfg)) if acc["check_variant"] in [s.name for s in g]), None)
        if g is None:
            raise KeyError(f"확인 판이 이 단계에 없음: {acc['check_variant']}")

        def verify(td: Path, gnames):
            rc = subprocess.call([sys.executable, str(script.resolve()), *argv_for(td), "--variant", ",".join(gnames),
                                  "--verify-run"])
            if rc != 0:
                raise rn.RunStopped(f"코드 바뀜 확인 다시 돌리기 멈춤({rn._rc_reason(rc)})")

        st = rn.accept_previous_code(rp.out, code_root, acc, g, verify, exp_log, cfg["stage"])
        print(f"이전 코드 인정 확인: {st}")
    cur_md5 = rn.snapshot_code(code_root, rp.out)
    pend = set(rn.pending_names(rp.out, specs, cur_md5))   # 설정 md5 다르면 멈춤·인정 범위 밖 판은 다시
    todo = [[s.name for s in g if s.name in pend] for g in groups]
    rn.run_variants_isolated(script.resolve(), [t for t in todo if t], argv_for(rp.out), exp_log, cfg["stage"])
    table = rp.out / "stage_table_cv.csv"
    if not table.exists():
        ev.write_table(rn.stage_cv_rows(rp.out, names), ev.table_notes((), ("보고 전 — test 칸 비움(§1-3)",)), table)
        if tracker is not None:
            tracker.log_artifact(stage_run_id, table, "tables")
            tracker.finish(stage_run_id)
    print(f"단계 {cfg['stage']}: 판 {len(names)}개 끝, CV 표 {table}")
    return names

