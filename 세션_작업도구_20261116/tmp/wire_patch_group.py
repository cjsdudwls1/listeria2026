from pathlib import Path

p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\runner.py")
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


# ---------------------------------------------------------------- resume_state: 인정된 이전 코드
rep('''def resume_state(vdir: Path, spec_md5: str, code_md5: str) -> str:
    """'new' | 'done' | 'partial'. done 인데 설정·코드가 다르면 멈춤."""''',
    '''def resume_state(vdir: Path, spec_md5: str, code_md5: str, accepted: Iterable[str] = ()) -> str:
    """'new' | 'done' | 'partial'. done 인데 설정·코드가 다르면 멈춤.
    accepted = 같은 결과로 확인돼 인정된 이전 code_md5(`accept_previous_code`) — 그 코드로 끝난 판도 끝난 판."""''')
rep('''        if d["code_md5"] != code_md5:
            raise ResumeError''', '''        if d["code_md5"] != code_md5 and d["code_md5"] not in set(accepted):
            raise ResumeError''')

# ---------------------------------------------------------------- predict: 여러 모델이 같은 입력으로
rep('''def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
                  check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """평가 쌍을 h 마다 메모리에서 만들어 바로 예측(B-1). 행 표에 y 없음.
    max_pairs_per_h = 연기 시험용(쌍 일부, 씨앗 고정) — 실제 실행은 None(쌍 전부, §3)."""
    sp = fd.SPLITS[split]
    meta = samples.meta()
    pos_l, h_l, p_l = [], [], []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, sp):''', '''def _copy_X(X):
    """모델마다 입력 사본(한 모델이 입력을 고쳐도 다음 모델에 안 번지게 — 값은 그대로)."""
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=True)
    if isinstance(X, dict):
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in X.items()}
    return X


def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
                  check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """평가 쌍을 h 마다 메모리에서 만들어 바로 예측(B-1). 행 표에 y 없음.
    max_pairs_per_h = 연기 시험용(쌍 일부, 씨앗 고정) — 실제 실행은 None(쌍 전부, §3)."""
    return predict_pairs_multi({"_": model}, asm, samples, split, W, check, max_pairs_per_h, seed)[0]["_"]


def predict_pairs_multi(models: Mapping[str, Any], asm, samples: ev.SampleTable, split: str, W: int | None,
                        check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                        seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict[str, float], float]:
    """h 마다 평가 쌍 입력을 한 번 만들어 여러 모델이 예측(같은 입력 설정의 판 묶음). 메모리에서만(B-1).
    돌려줌 = (판별 예측 표, 판별 예측 초, 입력 만들기 초)."""
    sp = fd.SPLITS[split]
    meta = samples.meta()
    pos_l, h_l = [], []
    p_l = {k: [] for k in models}
    secs = {k: 0.0 for k in models}
    t_in = 0.0
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, sp):''')
rep('''        m = meta.iloc[idx].copy()
        m["h"] = np.int16(h)
        m = add_bounds(m, W)
        p = _check_proba(model.predict_proba(asm.transform(m)), idx.size)
        pos_l.append(idx)
        h_l.append(np.full(idx.size, h, np.int16))
        p_l.append(p)
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    return pd.DataFrame({"lims_data_key": samples.keys[pos], "h": np.concatenate(h_l) if h_l else np.empty(0, np.int16),
                         "fold": split, "pred": np.concatenate(p_l) if p_l else np.empty(0)})''',
    '''        t0 = time.time()
        m = meta.iloc[idx].copy()
        m["h"] = np.int16(h)
        m = add_bounds(m, W)
        X = asm.transform(m)
        t_in += time.time() - t0
        for k, model in models.items():
            t0 = time.time()
            p_l[k].append(_check_proba(model.predict_proba(_copy_X(X) if len(models) > 1 else X), idx.size))
            secs[k] += time.time() - t0
        del X
        pos_l.append(idx)
        h_l.append(np.full(idx.size, h, np.int16))
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    keys = samples.keys[pos]
    hh = np.concatenate(h_l) if h_l else np.empty(0, np.int16)
    out = {k: pd.DataFrame({"lims_data_key": keys, "h": hh, "fold": split,
                            "pred": np.concatenate(v) if v else np.empty(0)}) for k, v in p_l.items()}
    return out, {k: round(v, 2) for k, v in secs.items()}, round(t_in, 2)''')

# ---------------------------------------------------------------- run_variant → run_group
i0 = s.index("def run_variant(spec: VariantSpec, ctx: RunContext) -> dict:")
i1 = s.index("# ================================================================ 여러 판(하위 프로세스)")
s = s[:i0] + '''def input_key(spec: VariantSpec) -> str | None:
    """같은 입력을 쓰는 판 묶음 열쇠. 트리(로지스틱 제외 — 입력 표준화가 모델마다 다름)만 묶음:
    입력 묶음 id·W·입력 방식·extra(판 설정의 단위 칸 등)가 같으면 같은 입력 X(알고리즘만 다름).
    딥러닝·로지스틱 = None(혼자)."""
    if spec.kind != "tree" or spec.algorithm == "LR":
        return None
    return json.dumps([spec.kind, spec.input_set, spec.W, spec.mode, dict(spec.extra)], sort_keys=True,
                      ensure_ascii=False)


def group_specs(specs: Sequence[VariantSpec]) -> list[list[VariantSpec]]:
    """판 목록 → 묶음 목록(묶음 순서 = 그 묶음 첫 판의 순서, 묶음 안 = 원래 순서)."""
    out: list[list[VariantSpec]] = []
    by: dict[str, list[VariantSpec]] = {}
    for sp in specs:
        k = input_key(sp)
        if k is None:
            out.append([sp])
        elif k in by:
            by[k].append(sp)
        else:
            by[k] = [sp]
            out.append(by[k])
    return out


ACCEPTED_FILE = "accepted_prev.json"


def accepted_codes(out_dir) -> dict:
    """단계 코드 사본에 적힌 '같은 결과로 인정된 이전 code_md5' 들."""
    p = Path(out_dir) / "_code" / ACCEPTED_FILE
    return read_json(p) if p.exists() else {}


def run_variant(spec: VariantSpec, ctx: RunContext) -> dict:
    """판 하나(§3: 3 fold + refit·test 예측) → CV 채점·기록. 끝난 판이면 건너뜀(B-1)."""
    return run_group([spec], ctx)[0]


def run_group(specs: Sequence[VariantSpec], ctx: RunContext) -> list[dict]:
    """입력 설정이 같은 판 묶음을 함께(split 마다 train 입력 1번 → 모델 차례로 fit → h 마다 평가 쌍 입력 1번 → 모든
    모델 예측). 판마다 결과 폴더·예측·점수·MLflow run·실험 로그는 판 하나씩 돌릴 때와 같음. 끝난 판은 빼고 돌림."""
    specs = list(specs)
    if not specs:
        return []
    keys = {input_key(sp) for sp in specs}
    if len(specs) > 1 and (None in keys or len(keys) != 1):
        raise ValueError("입력 설정이 다른 판(또는 묶지 않는 판)은 같이 돌리지 않음")
    if len({sp.name for sp in specs}) != len(specs):
        raise ValueError("묶음 안 판 이름 중복")
    if tuple(ctx.splits) != SPLIT_ORDER:
        raise ValueError("판 하나 = fold1~3 + refit")
    smoke = dict(ctx.smoke) if ctx.smoke else None
    if smoke is not None and set(smoke) != set(SMOKE_KEYS):
        raise ValueError(f"smoke 칸은 {SMOKE_KEYS}")
    out_dir = Path(ctx.out_dir)
    if ctx.require_7a_logged:
        check_prior_7a(ctx.exp_log)
    code_md5 = snapshot_code(ctx.code_root, out_dir)   # 단계 시작 때 사본과 다른 코드면 멈춤(한 단계 표에 코드 섞임 방지)
    accepted = accepted_codes(out_dir)
    results: dict[str, dict] = {}
    todo: list[VariantSpec] = []
    for sp in specs:
        vdir = variant_dir(out_dir, sp.name)
        state = resume_state(vdir, sp.md5(), code_md5, accepted)
        if state == "done":
            results[sp.name] = {"status": "skipped", **read_json(vdir / "done.json")}
            continue
        if state == "partial":
            moved, reason = _move_partial(vdir, out_dir)
            ctx.exp_log.append("rerun", stage=sp.stage, name=sp.name, reason=reason, moved_to=str(moved.name))
        todo.append(sp)
    if todo:
        for sp, r in zip(todo, _run_todo(todo, ctx, out_dir, code_md5, smoke)):
            results[sp.name] = r
    return [results[sp.name] for sp in specs]


def _run_todo(todo: list[VariantSpec], ctx: RunContext, out_dir: Path, code_md5: str, smoke) -> list[dict]:
    names = [sp.name for sp in todo]
    group = names if len(names) > 1 else None
    vdirs = {sp.name: variant_dir(out_dir, sp.name) for sp in todo}
    smd5 = {sp.name: sp.md5() for sp in todo}
    tr = ctx.tracker
    run_ids: dict[str, str | None] = {n: None for n in names}
    finished: set[str] = set()
    for sp in todo:
        vdirs[sp.name].mkdir(parents=True)
        cf.dump_json(vdirs[sp.name] / "spec.json", sp.to_dict())
        cf.dump_json(vdirs[sp.name] / "started.json", {"code_md5": code_md5, "spec_md5": smd5[sp.name],
                                                        "pid": os.getpid(), "group": group,
                                                        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
        ctx.exp_log.append("start", stage=sp.stage, name=sp.name, spec_md5=smd5[sp.name], code_md5=code_md5, group=group)
    try:
        if tr is not None:
            for sp in todo:
                tags = {"단계": sp.stage, "역할": sp.role, "코드 검수 완료": "예" if ctx.code_reviewed else "아니오",
                        "설정 md5": smd5[sp.name], "code_md5": code_md5}
                if sp.base_variant:
                    tags["바탕 판"] = sp.base_variant
                params = {"algorithm": sp.algorithm, "W": sp.W, "mode": sp.mode,
                          "W_DL": sp.W if sp.kind == "dl" else None, "input_set_id": sp.input_set,
                          "seed": ctx.seed, "n_jobs": ctx.n_jobs, "model_params": dict(sp.params),
                          "train_count": sp.train_count, "lib_versions": _lib_versions(), **dict(sp.extra)}
                run_ids[sp.name] = tr.start_variant(ctx.experiment, ctx.stage_run_id, sp.name, tags, params)
                if sp.judgement:
                    tr.set_judgement(run_ids[sp.name], sp.judgement)
        timings: dict[str, dict] = {n: {} for n in names}
        known = ctx.samples.known_frame()
        res0 = ctx.check_resources(out_dir)
        lead = todo[0]
        for split in SPLIT_ORDER:
            ctx.check_resources(out_dir)
            t0 = time.time()
            rows = fd.make_train_rows(known, fd.SPLITS[split])
            if smoke is not None:
                rows = _smoke_train(rows, smoke["max_train_rows"], ctx.seed)
            rows = add_bounds(rows, lead.W)
            y = rows["y"].to_numpy(dtype=np.int64)
            asm = ctx.make_inputs(lead, split).fit(rows)
            X = asm.transform(rows)
            t_train_in = round(time.time() - t0, 2)
            models = {}
            fit_s = {}
            for sp in todo:
                vdir = vdirs[sp.name]
                if sp is lead:
                    asm_i = asm
                else:   # 저장용(판마다 자기 설정) — 묶는 판은 fit 이 값을 안 만드는 조립 객체(로지스틱 제외)
                    asm_i = ctx.make_inputs(sp, split).fit(rows)
                    if hasattr(asm, "columns") and list(asm_i.columns) != list(asm.columns):
                        raise AssertionError(f"묶음 안 판의 입력 칼럼이 다름: {sp.name}")
                model = ctx.make_model(sp, asm_i)
                kw = dict(ctx.fit_kwargs(sp, split, rows, X)) if ctx.fit_kwargs else {}
                t1 = time.time()
                model.fit(_copy_X(X) if len(todo) > 1 else X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
                fit_s[sp.name] = round(time.time() - t1, 2)
                model.save(vdir / f"model_{split}")
                if hasattr(asm_i, "save"):
                    asm_i.save(vdir / f"inputs_{split}")
                if hasattr(asm_i, "columns"):
                    cf.dump_json(vdir / f"columns_{split}.json", {"columns": list(asm_i.columns)})
                dump = _model_dump(model)   # B-4 실제 적용값
                if dump is not None:
                    cf.dump_json(vdir / f"applied_params_{split}.json", cf.nan_to_null(dump))
                models[sp.name] = model
            del X
            preds, pred_s, t_eval_in = predict_pairs_multi(
                models, asm, ctx.samples, split, lead.W, check=lambda: ctx.check_resources(out_dir),
                max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None, seed=ctx.seed)
            for sp in todo:
                preds[sp.name].to_parquet(pred_path(vdirs[sp.name], split), index=False)
                timings[sp.name][split] = {
                    "train_rows": int(len(rows)), "train_pos": int(y.sum()), "n_pairs": int(len(preds[sp.name])),
                    "fit_seconds": fit_s[sp.name], "predict_seconds": pred_s[sp.name],
                    "shared_train_input_seconds": t_train_in, "shared_eval_input_seconds": t_eval_in}
            del models, asm, preds, rows
        out = []
        for sp in todo:
            vdir, run_id = vdirs[sp.name], run_ids[sp.name]
            cv = ev.score_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples, require_complete=smoke is None)
            count = ctx.exp_log.cv_scored(sp.stage, sp.name)
            cf.dump_scores_json(vdir / "cv_scores.json", cv)
            tm = timings[sp.name]
            own_s = round(sum(t["fit_seconds"] + t["predict_seconds"] for t in tm.values()), 2)
            shared_s = round(sum(t["shared_train_input_seconds"] + t["shared_eval_input_seconds"] for t in tm.values()), 2)
            cf.save_run_meta(vdir / "run_meta.json", sp.to_dict(), ctx.input_md5s, ctx.code_root, n_jobs=ctx.n_jobs,
                             extra={"timings": tm, "seconds_own": own_s, "seconds_shared_inputs": shared_s,
                                    "group": group, "resources_at_start": res0, "spec_md5": smd5[sp.name],
                                    "smoke": smoke})
            if tr is not None:
                tr.log_cv(run_id, cv, selection_count=count)
                tr.log_metrics(run_id, {"train_seconds": own_s, "shared_input_seconds": shared_s})
                for f in ("spec.json", "run_meta.json", "cv_scores.json"):
                    tr.log_artifact(run_id, vdir / f)
                for f in sorted(vdir.glob("columns_*.json")) + sorted(vdir.glob("applied_params_*.json")):
                    tr.log_artifact(run_id, f, "config")
                if ctx.log_pred_artifacts:
                    for s_ in SPLIT_ORDER:
                        tr.log_artifact(run_id, pred_path(vdir, s_), "predictions")
                tr.finish(run_id)
            done = {"name": sp.name, "stage": sp.stage, "spec_md5": smd5[sp.name], "code_md5": code_md5,
                    "cv_band_weighted": cv["cv_band_weighted"], "many_ties": cv["ties"]["many_ties"],
                    "selection_count": count, "mlflow_run_id": run_id, "seconds_total": round(own_s + shared_s, 2),
                    "group": group, "smoke": smoke}
            cf.dump_scores_json(vdir / "done.json", done)
            finished.add(sp.name)
            ctx.exp_log.append("done", stage=sp.stage, name=sp.name, cv=cv["cv_band_weighted"], count=count)
            out.append({"status": "done", **done})
        return out
    except BaseException as e:
        for sp in todo:
            if sp.name in finished:
                continue
            try:
                cf.dump_json(vdirs[sp.name] / "failed.json", {"type": type(e).__name__, "message": str(e)[:2000],
                                                              "traceback": traceback.format_exc()[-8000:],
                                                              "group": group}, overwrite=True)
                ctx.exp_log.append("failed", stage=sp.stage, name=sp.name, type=type(e).__name__,
                                   message=str(e)[:500], group=group)
                if tr is not None and run_ids.get(sp.name) is not None:
                    tr.finish(run_ids[sp.name], "FAILED")
            except Exception:  # 기록 실패가 원래 예외를 가리지 않게
                pass
        raise


# ================================================================ 코드 바뀜 인정(같은 결과 확인)
COMPARE_FILES = tuple(f"pred_{s}.parquet" for s in SPLIT_ORDER) + ("cv_scores.json",)


def compare_variant_files(a: Path, b: Path) -> list[str]:
    """두 판 폴더의 예측 parquet·CV 점수 파일을 바이트 단위로 비교 — 다른 파일 이름 목록."""
    diff = []
    for f in COMPARE_FILES:
        pa, pb = a / f, b / f
        if not (pa.exists() and pb.exists()) or pa.read_bytes() != pb.read_bytes():
            diff.append(f)
    return diff


def accept_previous_code(out_dir, code_root, acc: Mapping[str, str], group_names: Sequence[str],
                         verify_fn: Callable[[Path, Sequence[str]], Any], exp_log: ExperimentLog, stage: str) -> str:
    """단계 코드 사본(_code)의 code_md5 가 acc['code_md5'](이전 코드)이고 지금 코드가 다를 때: 확인 판이 든 묶음
    (group_names)을 새 코드로 임시 폴더에 다시 돌려(verify_fn(임시 폴더, 판 이름들)) 이전 코드로 끝난 그 판들의
    pred_*.parquet·cv_scores.json 과 바이트 단위로 같으면 → 실험 로그 'code_change_same_results', 옛 사본을
    `_code_prev_<md5>` 로 남기고 새 코드 사본으로 바꾸고 이전 code_md5 를 인정 목록에 적음(그 코드로 끝난 판은 건너뜀).
    다르면 실험 로그에 남기고 멈춤. 돌려줌 = 'no_snapshot'·'same_code'·'accepted'·'already_accepted'."""
    import tempfile

    out_dir, code_root = Path(out_dir), Path(code_root)
    snap = out_dir / "_code"
    if not snap.exists():
        return "no_snapshot"
    cur = cf.code_md5(code_root)
    old = (snap / "code_md5.txt").read_text(encoding="utf-8").strip()
    if old == cur:
        return "already_accepted" if acc["code_md5"] in accepted_codes(out_dir) else "same_code"
    if old != acc["code_md5"]:
        raise ResumeError(f"단계 코드 사본 code_md5 {old} ≠ 설정 accept_previous_code {acc['code_md5']}")
    check = acc["check_variant"]
    if check not in group_names:
        raise ValueError("확인 판이 묶음에 없음")
    olddone = [n for n in group_names if (out_dir / n / "done.json").exists()
               and read_json(out_dir / n / "done.json")["code_md5"] == old]
    if check not in olddone:
        raise ResumeError(f"확인 판 {check} 이 이전 코드({old})로 끝나 있지 않음")
    with tempfile.TemporaryDirectory(prefix="lst_verify_") as td:
        verify_fn(Path(td), list(group_names))
        diffs = {n: compare_variant_files(out_dir / n, Path(td) / n) for n in olddone}
    rec = {"old_code_md5": old, "new_code_md5": cur, "reason": acc["reason"], "check_variant": check,
           "compared": olddone, "files": list(COMPARE_FILES)}
    if any(diffs.values()):
        exp_log.append("code_change_different_results", stage=stage, diffs=diffs, **rec)
        raise ResumeError(f"새 코드 결과가 이전 코드와 다름 — 인정 안 함: {diffs}")
    prev = accepted_codes(out_dir)
    os.replace(snap, out_dir / f"_code_prev_{old}")
    snapshot_code(code_root, out_dir)
    prev[old] = {"reason": acc["reason"], "check_variant": check, "compared": olddone,
                 "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_json(snap / ACCEPTED_FILE, prev)
    exp_log.append("code_change_same_results", stage=stage, **rec)
    return "accepted"


''' + s[i1:]

# ---------------------------------------------------------------- 하위 프로세스: 묶음 단위
rep('''def run_variants_isolated(script, names: Sequence[str], base_argv: Sequence[str], exp_log: ExperimentLog,
                          stage: str, python: str | None = None) -> list[str]:
    """판마다 `python <script> <base_argv> --variant <판>` 하위 프로세스(B-1). 하나라도 멈추면 전체 멈춤."""
    py = python or sys.executable
    finished = []
    for name in names:
        rc = subprocess.call([py, str(script), *base_argv, "--variant", name])
        if rc != 0:
            reason = _rc_reason(rc)
            exp_log.append("stopped", stage=stage, name=name, returncode=rc, reason=reason)
            raise RunStopped(f"판 {name} 멈춤 — {reason}. 원인을 없앤 뒤 같은 명령을 다시 실행하면 이 판부터(B-1)")
        finished.append(name)
    return finished''', '''def run_variants_isolated(script, groups: Sequence[Sequence[str] | str], base_argv: Sequence[str],
                          exp_log: ExperimentLog, stage: str, python: str | None = None) -> list[str]:
    """묶음(판 이름 목록 — 문자열 하나면 판 하나)마다 `python <script> <base_argv> --variant <판,판,…>` 하위
    프로세스(B-1). 하나라도 멈추면 전체 멈춤."""
    py = python or sys.executable
    finished = []
    for g in groups:
        names = [g] if isinstance(g, str) else list(g)
        if not names:
            continue
        arg = ",".join(names)
        rc = subprocess.call([py, str(script), *base_argv, "--variant", arg])
        if rc != 0:
            reason = _rc_reason(rc)
            exp_log.append("stopped", stage=stage, name=arg, returncode=rc, reason=reason)
            raise RunStopped(f"판 {arg} 멈춤 — {reason}. 원인을 없앤 뒤 같은 명령을 다시 실행하면 이 판부터(B-1)")
        finished.extend(names)
    return finished''')
p.write_text(s, encoding="utf-8")
print("ok")
