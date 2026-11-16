from pathlib import Path

F = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")


def patch(path, pairs):
    p = F / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


patch("lst/runner.py", [
    # ---- resume_state: 판별 인정
    ('''def resume_state(vdir: Path, spec_md5: str, code_md5: str, accepted: Iterable[str] = ()) -> str:
    """'new' | 'done' | 'partial'. done 인데 설정·코드가 다르면 멈춤.
    accepted = 같은 결과로 확인돼 인정된 이전 code_md5(`accept_previous_code`) — 그 코드로 끝난 판도 끝난 판."""''',
     '''def resume_state(vdir: Path, spec_md5: str, code_md5: str, accepted: Mapping[str, Iterable[str]] | None = None) -> str:
    """'new' | 'done' | 'partial' | 'stale'. done 인데 설정이 다르면 멈춤.
    accepted = {인정된 이전 code_md5: 인정한 판 이름들}(`accept_previous_code`) — 그 코드로 끝난 판 중 인정 목록의 판만
    끝난 판, 목록 밖이면 'stale'(새 코드로 다시), 인정 기록이 없는 다른 코드면 멈춤."""
    accepted = accepted or {}'''),
    ('''        if d["code_md5"] != code_md5 and d["code_md5"] not in set(accepted):
            raise ResumeError''', '''        if d["code_md5"] != code_md5 and d["code_md5"] in accepted:
            return "done" if vdir.name in set(accepted[d["code_md5"]]) else "stale"
        if d["code_md5"] != code_md5:
            raise ResumeError'''),
    ('''def _move_partial(vdir: Path, out_dir: Path) -> tuple[Path, str]:
    reason = "비정상 종료(이유 기록 없음 — 메모리 부족 강제 종료 등)"''',
     '''def _move_partial(vdir: Path, out_dir: Path, sub: str = "_incomplete", reason: str | None = None) -> tuple[Path, str]:
    if reason is not None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = out_dir / sub / f"{vdir.name}__{ts}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(vdir, dst)
        return dst, reason
    reason = "비정상 종료(이유 기록 없음 — 메모리 부족 강제 종료 등)"'''),
    # ---- accepted map
    ('''def accepted_codes(out_dir) -> dict:
    """단계 코드 사본에 적힌 '같은 결과로 인정된 이전 code_md5' 들."""
    p = Path(out_dir) / "_code" / ACCEPTED_FILE
    return read_json(p) if p.exists() else {}
''', '''def accepted_codes(out_dir) -> dict:
    """단계 코드 사본에 적힌 '같은 결과로 인정된 이전 code_md5' 기록들(code_md5 → 기록)."""
    p = Path(out_dir) / "_code" / ACCEPTED_FILE
    return read_json(p) if p.exists() else {}


def accepted_map(out_dir) -> dict[str, set[str]]:
    """{인정된 이전 code_md5: 인정한 판 이름 집합}."""
    return {k: set(v["accepted_variants"]) for k, v in accepted_codes(out_dir).items()}


def pending_names(out_dir, specs: Sequence[VariantSpec], code_md5: str) -> list[str]:
    """돌릴 판(끝나지 않았거나 인정 범위 밖). 같은 이름인데 설정 md5 가 다르면 멈춤(이어받지 않음) — 부모 프로세스용."""
    acc = accepted_map(out_dir)
    return [sp.name for sp in specs
            if resume_state(variant_dir(out_dir, sp.name), sp.md5(), code_md5, acc) != "done"]
'''),
    ('''    accepted = accepted_codes(out_dir)
    results: dict[str, dict] = {}''', '''    accepted = accepted_map(out_dir)
    results: dict[str, dict] = {}'''),
    ('''        if state == "partial":
            moved, reason = _move_partial(vdir, out_dir)
            ctx.exp_log.append("rerun", stage=sp.stage, name=sp.name, reason=reason, moved_to=str(moved.name))
        todo.append(sp)''', '''        if state == "partial":
            moved, reason = _move_partial(vdir, out_dir)
            ctx.exp_log.append("rerun", stage=sp.stage, name=sp.name, reason=reason, moved_to=str(moved.name))
        if state == "stale":   # 이전 코드로 끝났으나 인정 범위 밖 → 옮기고 새 코드로 다시
            d = read_json(vdir / "done.json")
            reason = f"이전 코드({d['code_md5']}) 결과 인정 범위 밖 — 새 코드로 다시"
            moved, _ = _move_partial(vdir, out_dir, "_superseded", reason)
            if ctx.tracker is not None and d.get("mlflow_run_id"):
                ctx.tracker.supersede(d["mlflow_run_id"], reason, sp.name)
            ctx.exp_log.append("rerun", stage=sp.stage, name=sp.name, reason=reason, moved_to=str(moved.name))
        todo.append(sp)'''),
    # ---- accept_previous_code: 인정 범위 좁히기
    ('''def accept_previous_code(out_dir, code_root, acc: Mapping[str, str], group_names: Sequence[str],
                         verify_fn: Callable[[Path, Sequence[str]], Any], exp_log: ExperimentLog, stage: str) -> str:''',
     '''def acceptable(spec: VariantSpec, group: Sequence[VariantSpec]) -> bool:
    """인정 범위: 확인한 경로와 같은 종류의 판 = 묶음 실행 트리 판(로지스틱·딥러닝·묶이지 않는 판 제외), 알고리즘이
    확인 묶음 안의 것, 입력 묶음 id 가 같은 판."""
    return (input_key(spec) is not None and spec.algorithm in {g.algorithm for g in group}
            and spec.input_set in {g.input_set for g in group})


def accept_previous_code(out_dir, code_root, acc: Mapping[str, str], group: Sequence[VariantSpec],
                         verify_fn: Callable[[Path, Sequence[str]], Any], exp_log: ExperimentLog, stage: str) -> str:'''),
    ('''    다르면 실험 로그에 남기고 멈춤. 돌려줌 = 'no_snapshot'·'same_code'·'accepted'·'already_accepted'."""
    import tempfile
''', '''    다르면 실험 로그에 남기고 멈춤. 돌려줌 = 'no_snapshot'·'same_code'·'accepted'·'already_accepted'.
    인정하는 판 = 이전 코드로 끝난 판 중 `acceptable`(확인 묶음과 같은 실행 경로)인 판만 — 그 밖의 이전 코드 판은
    run_group 에서 '_superseded' 로 옮기고 새 코드로 다시."""
    import tempfile

    group = list(group)
    group_names = [g.name for g in group]
    if len({input_key(g) for g in group}) != 1 or input_key(group[0]) is None:
        raise ValueError("확인 묶음은 묶음 실행 트리 판들(같은 입력)이어야 함")
'''),
    ('''    prev = accepted_codes(out_dir)
    os.replace(snap, out_dir / f"_code_prev_{old}")
    snapshot_code(code_root, out_dir)
    prev[old] = {"reason": acc["reason"], "check_variant": check, "compared": olddone,
                 "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_json(snap / ACCEPTED_FILE, prev)
    exp_log.append("code_change_same_results", stage=stage, **rec)''', '''    accepted, not_accepted = [], []
    for p in sorted(out_dir.iterdir()):
        if not (p / "done.json").exists() or p.name.startswith(("_", ".")):
            continue
        if read_json(p / "done.json")["code_md5"] != old:
            continue
        (accepted if acceptable(VariantSpec(**read_json(p / "spec.json")), group) else not_accepted).append(p.name)
    rec.update({"accepted_variants": accepted, "not_accepted_variants": not_accepted,
                "accepted_rule": {"algorithms": sorted({g.algorithm for g in group}),
                                  "input_sets": sorted({g.input_set for g in group}), "grouped_tree_only": True}})
    prev = accepted_codes(out_dir)
    os.replace(snap, out_dir / f"_code_prev_{old}")
    snapshot_code(code_root, out_dir)
    prev[old] = {"reason": acc["reason"], "check_variant": check, "compared": olddone,
                 "accepted_variants": accepted, "not_accepted_variants": not_accepted,
                 "accepted_rule": rec["accepted_rule"],
                 "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_json(snap / ACCEPTED_FILE, prev)
    exp_log.append("code_change_same_results", stage=stage, **rec)'''),
])

patch("lst/stage_cli.py", [
    ('''        st = rn.accept_previous_code(rp.out, code_root, acc, [s.name for s in g], verify, exp_log, cfg["stage"])''',
     '''        st = rn.accept_previous_code(rp.out, code_root, acc, g, verify, exp_log, cfg["stage"])'''),
    ('''    rn.snapshot_code(code_root, rp.out)
    todo = [[s.name for s in g if not (rp.out / s.name / "done.json").exists()] for g in groups]''',
     '''    cur_md5 = rn.snapshot_code(code_root, rp.out)
    pend = set(rn.pending_names(rp.out, specs, cur_md5))   # 설정 md5 다르면 멈춤·인정 범위 밖 판은 다시
    todo = [[s.name for s in g if s.name in pend] for g in groups]'''),
])

patch("scripts/export_h_metrics.py", [
    ('''    ap.add_argument("--incomplete", default=DEFAULT_INCOMPLETE)''', '''    ap.add_argument("--incomplete", default=DEFAULT_INCOMPLETE)
    ap.add_argument("--experiment-log", default="logs/experiment_log.jsonl", help="--inputs 아래 실험 로그")'''),
    ('''        inc = ev.load_incomplete(resolve(inputs_root, args.incomplete))''', '''        inc = ev.load_incomplete(resolve(inputs_root, args.incomplete))
        rn.ExperimentLog(resolve(inputs_root, args.experiment_log)).append(
            "test_viewed", stage=args.stage, names=list(names), reason=args.reason, tool="export_h_metrics")'''),
])

t = F / "tests" / "test_group_run.py"
s = t.read_text(encoding="utf-8")
s = s.replace('rn.accept_previous_code(out, code_b, acc, [s.name for s in sp], verify, log, "1-1")',
              'rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1")')
s += '''

def test_accept_scope_and_spec_check(env, tmp_path):
    """인정 범위 = 확인 묶음과 같은 실행 경로(묶음 트리·같은 알고리즘·같은 입력 묶음) — 로지스틱 판은 새 코드로 다시.
    부모의 pending_names 는 같은 이름·다른 설정 md5 면 멈춤."""
    sp, out, code_a, code_b, log, acc, verify = _accept(env, tmp_path, AlgModel)
    lr = rn.VariantSpec(name="1-LR-W020-sum", stage="1-1", order="1차", kind="tree", algorithm="LR", W=20, mode="sum",
                        input_set="1차-트리", role="후보", train_count=4, extra={"version": {"include_ids": ["WEATHER"]}})
    other = specs3(20, "raw")
    la = rn.ExperimentLog(tmp_path / "la.jsonl")
    rn.run_variant(lr, ctx(env, out, la, code_a))
    rn.run_group(other, ctx(env, out, la, code_a))
    assert rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1") == "accepted"
    rec = rn.accepted_codes(out)[acc["code_md5"]]
    assert set(rec["accepted_variants"]) == {s.name for s in sp + other} and rec["not_accepted_variants"] == [lr.name]
    ev_ = [e for e in log.entries() if e["event"] == "code_change_same_results"][0]
    assert ev_["accepted_variants"] == rec["accepted_variants"]
    cur = cf.code_md5(code_b)
    assert rn.pending_names(out, sp + other + [lr], cur) == [lr.name]
    r = rn.run_variant(lr, ctx(env, out, log, code_b))           # 인정 범위 밖 → 옮기고 새 코드로 다시
    assert r["status"] == "done" and r["code_md5"] == cur
    assert any(p.name.startswith(lr.name + "__") for p in (out / "_superseded").iterdir())
    bad = rn.VariantSpec(**{**sp[0].to_dict(), "params": {"max_iter": 10}})
    with pytest.raises(rn.ResumeError, match="설정이 다름"):
        rn.pending_names(out, [bad], cur)
'''
t.write_text(s, encoding="utf-8")

h = F / "tests" / "test_h_metrics.py"
s = h.read_text(encoding="utf-8")
s = s.replace('''    long2 = pd.read_excel(p2, sheet_name="long")
    assert (long2["split"] == "test").sum() == 60''', '''    long2 = pd.read_excel(p2, sheet_name="long")
    assert (long2["split"] == "test").sum() == 60
    seen = [e for e in env["log"].entries() if e["event"] == "test_viewed"]
    assert len(seen) == 1 and seen[0]["names"] == [s.name] and seen[0]["reason"] == "시험"''')
h.write_text(s, encoding="utf-8")
print("ok")
