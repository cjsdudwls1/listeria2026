from pathlib import Path

p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\stage_cli.py")
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''SMOKE_ONLY_URI = ("sqlite:",)''', '''COMMON_SCHEMA["accept_previous_code"] = {   # 단계 도중 코드가 바뀔 때: 같은 결과 확인 뒤 이전 코드로 끝난 판 인정
    "code_md5": cf.Field(str),               # 단계 코드 사본(_code)의 이전 code_md5
    "reason": cf.Field(str),
    "check_variant": cf.Field(str),          # 새 코드로 다시 돌려 바이트 비교할 판(그 판이 든 묶음 전체를 비교)
}
OPTIONAL_SUB = ("smoke", "accept_previous_code")
SMOKE_ONLY_URI = ("sqlite:",)''')
rep('''    if "smoke" not in raw:
        schema = {k: v for k, v in schema.items() if k != "smoke"}''',
    '''    schema = {k: v for k, v in schema.items() if k not in OPTIONAL_SUB or k in raw}''')
rep('''    g.add_argument("--variant", help="판 하나만(하위 프로세스용)")''',
    '''    g.add_argument("--variant", help="판 하나 또는 같은 입력 묶음(쉼표 구분) — 하위 프로세스용")''')
rep('''    ap.add_argument("--only",''', '''    ap.add_argument("--verify-run", action="store_true",
                    help="(내부) 코드 바뀜 확인용 다시 돌리기 — MLflow·공용 실험 로그에 안 씀")
    ap.add_argument("--only",''')
rep('''    exp_log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    samples_path = resolve(inputs_root, cfg["samples"])
    tracker = tracker_fn(cfg)
    stage_run_id = tracker.stage_run(cfg["experiment"], cfg["stage"]) if tracker is not None else None
''', '''    samples_path = resolve(inputs_root, cfg["samples"])
    if args.verify_run:   # 임시 폴더에서만: 공용 실험 로그·MLflow 에 안 남김
        exp_log = rn.ExperimentLog(rp.out / "_verify_experiment_log.jsonl")
        tracker = None
    else:
        exp_log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
        tracker = tracker_fn(cfg)
    stage_run_id = tracker.stage_run(cfg["experiment"], cfg["stage"]) if tracker is not None else None
''')
rep('''    if args.variant:
        spec = {s.name: s for s in specs}.get(args.variant)
        if spec is None:
            raise KeyError(f"이 단계에 없는 판: {args.variant}")''', '''    if args.variant:
        by = {s.name: s for s in specs}
        want = [n.strip() for n in args.variant.split(",") if n.strip()]
        unknown = [n for n in want if n not in by]
        if unknown:
            raise KeyError(f"이 단계에 없는 판: {unknown}")
        group = [by[n] for n in want]''')
rep('''                            require_7a_logged=not cfg.get("smoke", {}).get("skip_7a_check", False), **kw)
        res = rn.run_variant(spec, ctx)
        print(f"{spec.name}: {res['status']} CV {res.get('cv_band_weighted')}")
        return res''', '''                            require_7a_logged=(not args.verify_run
                                               and not cfg.get("smoke", {}).get("skip_7a_check", False)), **kw)
        res = rn.run_group(group, ctx)
        for sp, r in zip(group, res):
            print(f"{sp.name}: {r['status']} CV {r.get('cv_band_weighted')}")
        return res if len(res) > 1 else res[0]''')
rep('''    rn.snapshot_code(script.resolve().parents[1], rp.out)
    base_argv = ["--root", str(rp.root), "--out", str(rp.out), "--inputs", str(inputs_root),
                 "--config", str(Path(args.config).resolve())]
    rn.run_variants_isolated(script.resolve(), names, base_argv, exp_log, cfg["stage"])''', '''    code_root = script.resolve().parents[1]
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

        st = rn.accept_previous_code(rp.out, code_root, acc, [s.name for s in g], verify, exp_log, cfg["stage"])
        print(f"이전 코드 인정 확인: {st}")
    rn.snapshot_code(code_root, rp.out)
    todo = [[s.name for s in g if not (rp.out / s.name / "done.json").exists()] for g in groups]
    rn.run_variants_isolated(script.resolve(), [t for t in todo if t], argv_for(rp.out), exp_log, cfg["stage"])''')
rep('''from dataclasses import dataclass''', '''import subprocess
import sys
from dataclasses import dataclass''')
p.write_text(s, encoding="utf-8")
print("ok")
