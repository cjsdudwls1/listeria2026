def _run_todo(todo: list[VariantSpec], ctx: RunContext, out_dir: Path, code_md5: str, smoke) -> list[dict]:
    names = [sp.name for sp in todo]
    group = names if len(names) > 1 else None
    vdirs = {sp.name: variant_dir(out_dir, sp.name) for sp in todo}
    smd5 = {sp.name: sp.md5() for sp in todo}
    tr = ctx.tracker
    run_ids: dict[str, str | None] = {n: None for n in names}
    finished: set[str] = set()
    chain_mode = any(is_chain_boost(sp) for sp in todo)
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
        res0 = ctx.check_resources(out_dir)
        if chain_mode:
            timings, chain = _train_chain(todo, ctx, out_dir, vdirs, smoke, code_md5)
        else:
            timings, chain = _train_plain(todo, ctx, out_dir, vdirs, smoke), {}
        out = []
        for sp in todo:
            vdir, run_id = vdirs[sp.name], run_ids[sp.name]
            ch = chain.get(sp.name)
            thr = load_preds(vdir, [f"{THR_PREFIX}{f}" for f in ev.CV_FOLDS]) if ch is not None else None
            thr = {f: thr[f"{THR_PREFIX}{f}"] for f in ev.CV_FOLDS} if thr is not None else None
            cv = ev.score_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples, thr_preds=thr,
                             require_complete=smoke is None)
            if ch is not None:
                _check_chain_cv(sp.name, cv, ch)
                write_n_iter(vdir, ch)
            count = ctx.exp_log.cv_scored(sp.stage, sp.name)
            cf.dump_scores_json(vdir / "cv_scores.json", cv)
            write_h_metrics(vdir, ev.h_metrics_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples,
                                                  float(cv["threshold"]["threshold"]),
                                                  require_complete=smoke is None))
            tm = timings[sp.name]
            own_s = round(sum(t["fit_seconds"] + t["predict_seconds"] for t in tm.values()), 2)
            shared_s = round(sum(t["shared_train_input_seconds"] + t["shared_eval_input_seconds"] for t in tm.values()), 2)
            extra = {"timings": tm, "seconds_own": own_s, "seconds_shared_inputs": shared_s,
                     "group": group, "resources_at_start": res0, "spec_md5": smd5[sp.name], "smoke": smoke}
            if ch is not None:
                extra["n_iter"] = n_iter_summary(ch)
            cf.save_run_meta(vdir / "run_meta.json", sp.to_dict(), ctx.input_md5s, ctx.code_root, n_jobs=ctx.n_jobs,
                             extra=extra)
            if tr is not None:
                tr.log_cv(run_id, cv, selection_count=count)
                tr.log_metrics(run_id, {"train_seconds": own_s, "shared_input_seconds": shared_s})
                files = ["spec.json", "run_meta.json", "cv_scores.json", H_METRICS_FILE]
                if ch is not None:
                    s = n_iter_summary(ch)
                    tr.log_params(run_id, {"max_iter": s["max_iter"], "common_n": s["common_n"],
                                           **{f"cross_n_{f}": s["cross_n"][f] for f in ev.CV_FOLDS},
                                           **({"learning_rate": s["learning_rate"]} if s["learning_rate"] else {})})
                    tr.set_tags(run_id, {CAP_MARK: "예" if s["at_cap"] else "아니오"})
                    files += [N_ITER_FILE, N_ITER_CURVES_FILE]
                for f in files:
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
            if ch is not None:
                done["n_iter"] = n_iter_summary(ch)
                if done["n_iter"]["at_cap"]:          # §3: "상한" 표시하고 알림
                    msg = (f"{sp.name}: 교차 N·공통 N 이 마지막 시점(최대 반복 {done['n_iter']['max_iter']}) — "
                           f"표에 '{CAP_MARK}' 표시(§3)")
                    ctx.exp_log.append("alert", stage=sp.stage, name=sp.name, message=msg)
                    print(f"[알림] {msg}")
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


def _train_rows(ctx: RunContext, split: str, W: int | None, smoke, known: pd.DataFrame) -> pd.DataFrame:
    rows = fd.make_train_rows(known, fd.SPLITS[split])
    if smoke is not None:
        rows = _smoke_train(rows, smoke["max_train_rows"], ctx.seed)
    return add_bounds(rows, W)


def _save_split_model(vdir: Path, split: str, model, asm_i) -> None:
    model.save(vdir / f"model_{split}")
    if hasattr(asm_i, "save"):
        asm_i.save(vdir / f"inputs_{split}")
    if hasattr(asm_i, "columns"):
        cf.dump_json(vdir / f"columns_{split}.json", {"columns": list(asm_i.columns)})
    dump = _model_dump(model)   # B-4 실제 적용값
    if dump is not None:
        cf.dump_json(vdir / f"applied_params_{split}.json", cf.nan_to_null(dump))


def _asm_for(sp: VariantSpec, lead: VariantSpec, asm, ctx: RunContext, split: str, rows: pd.DataFrame):
    """판마다 저장용 조립 객체: 묶음 첫 판 = 그 객체, 딥러닝 = 같은 객체(같은 판 설정 → 같은 전처리기 — 다시 fit 하면
    train 입력을 또 만듦), 트리 = 자기 설정으로 다시 fit(값을 안 만드는 조립 객체 — 로지스틱 제외)."""
    if sp is lead or sp.kind == "dl":
        return asm
    asm_i = ctx.make_inputs(sp, split).fit(rows)
    if hasattr(asm, "columns") and list(asm_i.columns) != list(asm.columns):
        raise AssertionError(f"묶음 안 판의 입력 칼럼이 다름: {sp.name}")
    return asm_i


def _train_plain(todo: list[VariantSpec], ctx: RunContext, out_dir: Path, vdirs: Mapping[str, Path], smoke) -> dict:
    """1차·단계 2 준비·2차 기본 설정 판(반복 수 고르기 없음): split 마다 train 입력 1번 → 모델 차례로 fit → 평가 쌍 예측."""
    timings: dict[str, dict] = {sp.name: {} for sp in todo}
    known = ctx.samples.known_frame()
    lead = todo[0]
    copy = _copy_needed(todo)
    for split in SPLIT_ORDER:
        ctx.check_resources(out_dir)
        t0 = time.time()
        rows = _train_rows(ctx, split, lead.W, smoke, known)
        y = rows["y"].to_numpy(dtype=np.int64)
        asm = ctx.make_inputs(lead, split).fit(rows)
        X = asm.transform(rows)
        t_train_in = round(time.time() - t0, 2)
        models = {}
        fit_s = {}
        for sp in todo:
            asm_i = _asm_for(sp, lead, asm, ctx, split, rows)
            model = ctx.make_model(sp, asm_i)
            kw = dict(ctx.fit_kwargs(sp, split, rows, X)) if ctx.fit_kwargs else {}
            t1 = time.time()
            model.fit(_copy_X(X) if copy else X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
            fit_s[sp.name] = round(time.time() - t1, 2)
            _save_split_model(vdirs[sp.name], split, model, asm_i)
            models[sp.name] = model
        del X
        preds, pred_s, t_eval_in = predict_pairs_multi(
            models, asm, ctx.samples, split, lead.W, check=lambda: ctx.check_resources(out_dir),
            max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None, seed=ctx.seed,
            dedupe_rows=row_dedupe_ok(todo), copy_inputs=copy)
        for sp in todo:
            preds[sp.name].to_parquet(pred_path(vdirs[sp.name], split), index=False)
            timings[sp.name][split] = {
                "train_rows": int(len(rows)), "train_pos": int(y.sum()), "n_pairs": int(len(preds[sp.name])),
                "fit_seconds": fit_s[sp.name], "predict_seconds": pred_s[sp.name],
                "shared_train_input_seconds": t_train_in, "shared_eval_input_seconds": t_eval_in}
        del models, asm, preds, rows
    return timings


# ================================================================ 2차 사슬 반복 수 고르기(§3)
ITER_SELECT = "iter_select"      # spec.params 칸: 2차 사슬 부스팅 판(반복 수 고르기·CatBoost 학습률 평균 켬)
BOOST_ALGS = ("XGB", "LGB", "CAT")
THR_PREFIX = "thr_"              # threshold 용 공통 N 예측 = pred_thr_fold<f>.parquet
N_ITER_FILE = "n_iter.json"
N_ITER_CURVES_FILE = "n_iter_curves.csv"
N_ITER_BY_H_FILE = "n_iter_curves_by_h.csv"
CAP_MARK = "상한"


def is_chain_boost(spec: VariantSpec) -> bool:
    """2차 사슬 부스팅 판(§3 반복 수 고르기) 여부 = params['iter_select'] 이 참. 부스팅 트리 판만 켤 수 있고, 최대 반복·
    학습률은 실행 때 정함(실험 로그·fold 자동 학습률) — 판 설정에 max_iter 를 같이 적으면 멈춤."""
    if not spec.params.get(ITER_SELECT):
        return False
    if spec.kind != "tree" or spec.algorithm not in BOOST_ALGS:
        raise ValueError(f"{spec.name}: 반복 수 고르기는 부스팅(XGB·LGB·CAT) 트리 판만")
    if "max_iter" in spec.params:
        raise ValueError(f"{spec.name}: 2차 사슬 판의 max_iter 는 실행 때 정함(§3) — 설정에 적지 않음")
    return True


def model_spec(sp: VariantSpec, max_iter: int | None, learning_rate: float | None) -> VariantSpec:
    """모델 만들기용 판 설정(실행 때 정한 최대 반복·CatBoost 학습률을 params 에 — 판 설정 md5 와 무관)."""
    p = dict(sp.params)
    if max_iter is not None:
        p["max_iter"] = int(max_iter)
    if learning_rate is not None:
        lib = dict(p.get("params") or {})
        lib["learning_rate"] = float(learning_rate)
        p["params"] = lib
    return dataclasses.replace(sp, params=p)


def argmax_round4(values: Sequence[float], points: Sequence[int]) -> int:
    """§3·§1-2: 4자리 반올림 뒤 최댓값 시점, 같으면 작은 N(points 오름차순)."""
    from .metrics import round4

    r = np.array([round4(float(v)) for v in values])
    if np.isnan(r).any():
        raise ValueError("곡선에 NaN — 반복 수를 고를 수 없음")
    return int(points[int(np.flatnonzero(r == r.max())[0])])


def curve_from_by_h(by_h: np.ndarray) -> np.ndarray:
    """(시점 수, 60) h 별 PR-AUC → 시점별 구간 가중 PR-AUC(metrics.band_weighted — score_cv 와 같은 셈)."""
    from .metrics import band_weighted

    return np.array([band_weighted(r) for r in np.asarray(by_h, dtype=np.float64)], dtype=np.float64)


def max_iter_step(curve: Sequence[float], points: Sequence[int], max_iter: int) -> dict:
    """§3 최대 반복 판정 한 걸음(fold 3 자기 곡선): 최적 N 이 마지막 시점이면 2배(상한 16,000 — 넘으면 '상한')."""
    from . import models_tree as mtr

    own = argmax_round4(curve, points)
    last = int(points[-1])
    rec = {"max_iter": int(max_iter), "own_best_n": own, "last_point": last, "n_points": len(points),
           "own_best_at_last": own == last}
    if own != last:
        rec.update(decided=True, next=None, at_cap=False)
    else:
        nxt = mtr.next_max_iter(max_iter)
        rec.update(decided=nxt is None, next=nxt, at_cap=nxt is None)
    return rec


def _check_multi(P, n: int, k: int) -> np.ndarray:
    a = np.asarray(P, dtype=np.float64)
    if a.shape != (n, k):
        raise ValueError(f"predict_proba_multi 꼴 {a.shape} ≠ ({n}, {k})")
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("predict_proba_multi 값이 [0, 1] 밖이거나 NaN")
    return a


def eval_pairs_chain(jobs: Mapping[str, tuple], asm, samples: ev.SampleTable, split: str, W: int | None,
                     check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None, seed: int = 42,
                     dedupe_rows: bool = False, copy_inputs: bool = False) -> dict:
    """평가 쌍을 h 마다 메모리에서 만들어(B-1) 판마다 한 가지로 씀 — jobs = {판: (모델, 종류, 인자)}:
      'plain' = predict_proba 예측 표(인자 없음),
      'curve' = 시점(인자 = 오름차순 트리 수 목록)마다 h 별 PR-AUC(fold validation 정답 — CV fold 만) → (시점 수, 60),
                예측은 남기지 않음(메모리 — 4.4M 쌍 × 56 시점을 들고 있지 않음),
      'iters' = 트리 수(인자 = 목록)마다 예측 표.
    부스팅 시점 예측 = predict_proba_multi(같은 행 한 번 — 열 k = predict_proba(n_iter=k) 와 같음, models_tree 구현 세부 9).
    돌려줌 = {'plain': {판: 표}, 'iters': {판: {N: 표}}, 'curve': {판: 배열}, 'seconds': {판: 초}, 'input_seconds': 초,
             'n_pairs': 쌍 수}."""
    import contextlib

    from .hassign import u_tie_many
    from .metrics import pr_auc

    spl = fd.SPLITS[split]
    if any(j[1] == "curve" for j in jobs.values()) and split not in ev.CV_FOLDS:
        raise ValueError("곡선은 CV fold validation 쌍에서만(test 정답 안 읽음 — §1-3)")
    meta = samples.meta()
    pos_l, h_l = [], []
    plain = {k: [] for k, j in jobs.items() if j[1] == "plain"}
    iters = {k: {int(n): [] for n in j[2]} for k, j in jobs.items() if j[1] == "iters"}
    curves = {k: np.full((len(j[2]), fd_h_max()), np.nan) for k, j in jobs.items() if j[1] == "curve"}
    secs = {k: 0.0 for k in jobs}
    t_in = 0.0
    scope = asm.pair_cache() if hasattr(asm, "pair_cache") else contextlib.nullcontext()
    with scope:
        for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, spl):
            if max_pairs_per_h is not None and idx.size > max_pairs_per_h:
                idx = np.sort(np.random.default_rng([seed, h]).choice(idx, max_pairs_per_h, replace=False))
            if idx.size == 0:
                continue
            if check is not None:
                check()
            t0 = time.time()
            m = meta.iloc[idx].copy()
            m["h"] = np.int16(h)
            m = add_bounds(m, W)
            X = asm.transform(m)
            rep, inv = unique_rows(X) if dedupe_rows else (np.arange(idx.size), None)
            if rep.size < idx.size:
                X = take_rows(X, rep)
            else:
                inv = None
            if curves:
                y = samples.cv_y(idx)
                u = u_tie_many(samples.keys[idx], np.full(idx.size, h, dtype=np.int64))
            t_in += time.time() - t0
            for k, (model, kind, arg) in jobs.items():
                t0 = time.time()
                Xk = _copy_X(X) if copy_inputs else X
                if kind == "plain":
                    p = _check_proba(model.predict_proba(Xk), rep.size)
                    plain[k].append(p[inv] if inv is not None else p)
                else:
                    ks = [int(n) for n in arg]
                    P = _check_multi(model.predict_proba_multi(Xk, ks), rep.size, len(ks))
                    if inv is not None:
                        P = P[inv]
                    if kind == "curve":
                        for j in range(len(ks)):
                            curves[k][j, h - 1] = pr_auc(y, P[:, j], u)
                    else:
                        for j, n in enumerate(ks):
                            iters[k][n].append(np.ascontiguousarray(P[:, j]))
                secs[k] += time.time() - t0
            del X
            pos_l.append(idx)
            h_l.append(np.full(idx.size, h, np.int16))
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    keys = samples.keys[pos]
    hh = np.concatenate(h_l) if h_l else np.empty(0, np.int16)

    def frame(v):
        return pd.DataFrame({"lims_data_key": keys, "h": hh, "fold": split,
                             "pred": np.concatenate(v) if v else np.empty(0)})
    return {"plain": {k: frame(v) for k, v in plain.items()},
            "iters": {k: {n: frame(v) for n, v in d.items()} for k, d in iters.items()},
            "curve": curves, "seconds": {k: round(v, 2) for k, v in secs.items()},
            "input_seconds": round(t_in, 2), "n_pairs": int(pos.size)}


def fd_h_max() -> int:
    from .hassign import H_MAX

    return H_MAX


def _train_chain(todo: list[VariantSpec], ctx: RunContext, out_dir: Path, vdirs: Mapping[str, Path], smoke,
                 code_md5: str) -> tuple[dict, dict]:
    """2차 사슬 판 묶음(§3·§1-2): 부스팅 = fold 마다 최대 반복까지 학습 → fold validation 시점별 구간 가중 PR-AUC 곡선 →
    교차 N(나머지 두 fold 평균 곡선)·공통 N(세 fold 평균 곡선) → fold 예측(CV) = 교차 N, threshold 용 fold 예측 = 공통 N
    (같은 fold 모델·트리 수만 다르게), refit = 최대 반복까지 학습 뒤 공통 N 예측. CatBoost 학습률 = fold 별 자동
    학습률(부록 A-12 읽기) 평균으로 모든 fold·refit. 최대 반복 = 실험 로그 기록(알고리즘마다 첫 2차 사슬 부스팅 판에서
    fold 3 을 먼저 학습해 정함 — 자기 곡선 최적 N 이 마지막 시점이면 2배씩, 상한 16,000). 같은 묶음의 기본 설정 판
    (RF·ExtraTrees)은 같은 입력으로 기본대로.
    구현 세부(계획서에 없음): 곡선을 낼 때는 예측을 들고 있지 않고(쌍 × 시점 메모리), N 을 정한 뒤 fold 모델로 평가 쌍
    입력을 다시 만들어 교차 N·공통 N 두 열만 예측(fold 마다 평가 쌍 입력 한 번 더)."""
    from . import models_tree as mtr

    lead = todo[0]
    boost = [sp for sp in todo if is_chain_boost(sp)]
    bnames = {sp.name for sp in boost}
    if len({sp.algorithm for sp in boost}) != len(boost):
        raise ValueError("한 묶음에 같은 알고리즘의 2차 사슬 부스팅 판이 둘 — 따로 돌릴 것")
    known = ctx.samples.known_frame()
    copy = _copy_needed(todo)
    dedupe = row_dedupe_ok(todo)
    timings: dict[str, dict] = {sp.name: {} for sp in todo}
    info: dict[str, dict] = {sp.name: {"train_rows": {}, "train_pos": {}} for sp in boost}
    check = lambda: ctx.check_resources(out_dir)  # noqa: E731
    mp = smoke["max_pairs_per_h"] if smoke else None

    def train_inputs(split):
        ctx.check_resources(out_dir)
        t0 = time.time()
        rows = _train_rows(ctx, split, lead.W, smoke, known)
        y = rows["y"].to_numpy(dtype=np.int64)
        asm = ctx.make_inputs(lead, split).fit(rows)
        X = asm.transform(rows)
        return rows, y, asm, X, round(time.time() - t0, 2)

    # (1) CatBoost 학습률 = fold 별 자동 학습률 평균(§3·부록 A-12) — 판 설정에 학습률이 있으면(단계 8 시도) 그대로
    lr: dict[str, float] = {}
    cats = [sp for sp in boost if sp.algorithm == "CAT" and "learning_rate" not in (sp.params.get("params") or {})]
    if cats:
        rates = {sp.name: {} for sp in cats}
        t_lr = {sp.name: 0.0 for sp in cats}
        for split in ev.CV_FOLDS:
            rows, y, asm, X, _ = train_inputs(split)
            for sp in cats:
                t1 = time.time()
                m = ctx.make_model(model_spec(sp, None, None), asm)
                rates[sp.name][split] = float(m.read_auto_learning_rate(_copy_X(X), y, seed=ctx.seed,
                                                                        n_jobs=ctx.n_jobs))
                t_lr[sp.name] += time.time() - t1
            del rows, asm, X
        for sp in cats:
            lr[sp.name] = mtr.mean_learning_rate([rates[sp.name][f] for f in ev.CV_FOLDS])
            info[sp.name]["learning_rate"] = {"fold_auto": rates[sp.name], "mean": lr[sp.name],
                                              "read_seconds": round(t_lr[sp.name], 2)}
            ctx.exp_log.append("catboost_learning_rate", stage=sp.stage, name=sp.name, fold_auto=rates[sp.name],
                               mean=lr[sp.name])

    # (2) 최대 반복: 실험 로그 기록 또는 이 판에서 정함(fold 3 먼저)
    mi: dict[str, int] = {}
    determine: list[VariantSpec] = []
    for sp in boost:
        rec = ctx.exp_log.max_iter_record(sp.algorithm)
        if rec is None:
            determine.append(sp)
            mi[sp.name] = mtr.MAX_ITER_DEFAULT_CHAIN
            info[sp.name]["max_iter_source"] = {"decided_here": True}
        else:
            mi[sp.name] = int(rec["max_iter"])
            info[sp.name]["max_iter_source"] = {"decided_here": False, "record": rec}
    order = ("fold3", "fold1", "fold2") if determine else ev.CV_FOLDS
    by_h: dict[str, dict] = {sp.name: {} for sp in boost}
    keep: dict[tuple[str, str], Any] = {}
    for split in order:
        rows, y, asm, X, t_tr = train_inputs(split)
        active = list(todo)
        hist: dict[str, list] = {sp.name: [] for sp in determine}
        rnd = 0
        while active:
            rnd += 1
            fitted, asms, fit_s = {}, {}, {}
            for sp in active:
                asm_i = _asm_for(sp, lead, asm, ctx, split, rows)
                model = ctx.make_model(model_spec(sp, mi.get(sp.name), lr.get(sp.name)), asm_i)
                kw = dict(ctx.fit_kwargs(sp, split, rows, X)) if ctx.fit_kwargs else {}
                t1 = time.time()
                model.fit(_copy_X(X) if (copy or len(active) > 1) else X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
                fit_s[sp.name] = round(time.time() - t1, 2)
                fitted[sp.name], asms[sp.name] = model, asm_i
            jobs = {sp.name: (fitted[sp.name], "curve", mtr.checkpoints(mi[sp.name])) if sp.name in bnames
                    else (fitted[sp.name], "plain", None) for sp in active}
            res = eval_pairs_chain(jobs, asm, ctx.samples, split, lead.W, check=check, max_pairs_per_h=mp,
                                   seed=ctx.seed, dedupe_rows=dedupe, copy_inputs=copy or len(jobs) > 1)
            redo = []
            if split == "fold3":
                for sp in [s for s in active if s in determine]:
                    pts = mtr.checkpoints(mi[sp.name])
                    st = max_iter_step(curve_from_by_h(res["curve"][sp.name]), pts, mi[sp.name])
                    hist[sp.name].append(st)
                    if not st["decided"]:
                        mi[sp.name] = int(st["next"])
                        redo.append(sp)
                        continue
                    rec = ctx.exp_log.record_max_iter(sp.algorithm, max_iter=mi[sp.name], at_cap=st["at_cap"],
                                                      history=hist[sp.name], variant=sp.name, stage=sp.stage,
                                                      code_md5=code_md5)
                    info[sp.name]["max_iter_source"]["record"] = rec
                    if st["at_cap"]:
                        msg = (f"{sp.algorithm} 최대 반복 상한 {mi[sp.name]} 에서도 fold 3 최적 N 이 마지막 시점 — "
                               f"'{CAP_MARK}' 표시로 기록하고 진행(§3)")
                        ctx.exp_log.append("alert", stage=sp.stage, name=sp.name, message=msg)
                        print(f"[알림] {msg}")
            for sp in active:
                t = timings[sp.name].setdefault(split, {"fit_seconds": 0.0, "predict_seconds": 0.0, "rounds": 0,
                                                        "shared_train_input_seconds": t_tr,
                                                        "shared_eval_input_seconds": 0.0})
                t["fit_seconds"] = round(t["fit_seconds"] + fit_s[sp.name], 2)
                t["predict_seconds"] = round(t["predict_seconds"] + res["seconds"][sp.name], 2)
                t["shared_eval_input_seconds"] = round(t["shared_eval_input_seconds"] + res["input_seconds"], 2)
                t["rounds"] += 1
                t.update({"train_rows": int(len(rows)), "train_pos": int(y.sum()), "n_pairs": res["n_pairs"]})
                if sp in redo:            # 최대 반복을 늘려 다시 — 이 모델은 버림(저장 안 함)
                    continue
                _save_split_model(vdirs[sp.name], split, fitted[sp.name], asms[sp.name])
                if sp.name in bnames:
                    by_h[sp.name][split] = res["curve"][sp.name]
                    keep[(sp.name, split)] = fitted[sp.name]
                    info[sp.name]["train_rows"][split] = int(len(rows))
                    info[sp.name]["train_pos"][split] = int(y.sum())
                else:
                    res["plain"][sp.name].to_parquet(pred_path(vdirs[sp.name], split), index=False)
            for sp in determine:
                if sp.name in hist and hist[sp.name]:
                    info[sp.name]["max_iter_source"]["history"] = hist[sp.name]
            del res, fitted
            active = redo
        del X, asm, rows

    # (3) 교차 N·공통 N
    for sp in boost:
        pts = mtr.checkpoints(mi[sp.name])
        cur = {f: curve_from_by_h(by_h[sp.name][f]) for f in ev.CV_FOLDS}
        sel = mtr.select_n_iter(cur, pts)
        info[sp.name].update(max_iter=mi[sp.name], points=pts, curves={f: cur[f].tolist() for f in ev.CV_FOLDS},
                             by_h={f: by_h[sp.name][f].tolist() for f in ev.CV_FOLDS}, selection=sel)

    # (4) fold 예측: 교차 N(CV 점수)·공통 N(threshold 용) — 같은 fold 모델, 평가 쌍 입력 다시
    for split in ev.CV_FOLDS:
        ctx.check_resources(out_dir)
        rows = _train_rows(ctx, split, lead.W, smoke, known)
        asm = ctx.make_inputs(lead, split).fit(rows)
        jobs = {}
        for sp in boost:
            s = info[sp.name]["selection"]
            jobs[sp.name] = (keep[(sp.name, split)], "iters", sorted({s["cross_n"][split], s["common_n"]}))
        res = eval_pairs_chain(jobs, asm, ctx.samples, split, lead.W, check=check, max_pairs_per_h=mp,
                               seed=ctx.seed, dedupe_rows=dedupe, copy_inputs=copy or len(jobs) > 1)
        for sp in boost:
            s = info[sp.name]["selection"]
            res["iters"][sp.name][s["cross_n"][split]].to_parquet(pred_path(vdirs[sp.name], split), index=False)
            res["iters"][sp.name][s["common_n"]].to_parquet(pred_path(vdirs[sp.name], THR_PREFIX + split),
                                                           index=False)
            t = timings[sp.name][split]
            t["predict_seconds"] = round(t["predict_seconds"] + res["seconds"][sp.name], 2)
            t["shared_eval_input_seconds"] = round(t["shared_eval_input_seconds"] + res["input_seconds"], 2)
            keep.pop((sp.name, split))
        del res, asm, rows

    # (5) refit: 최대 반복까지 학습 → 공통 N 예측(부스팅), 기본 설정 판은 그대로
    rows, y, asm, X, t_tr = train_inputs("refit")
    fitted, fit_s = {}, {}
    for sp in todo:
        asm_i = _asm_for(sp, lead, asm, ctx, "refit", rows)
        model = ctx.make_model(model_spec(sp, mi.get(sp.name), lr.get(sp.name)), asm_i)
        kw = dict(ctx.fit_kwargs(sp, "refit", rows, X)) if ctx.fit_kwargs else {}
        t1 = time.time()
        model.fit(_copy_X(X) if copy else X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
        fit_s[sp.name] = round(time.time() - t1, 2)
        _save_split_model(vdirs[sp.name], "refit", model, asm_i)
        fitted[sp.name] = model
        if sp.name in bnames:
            info[sp.name]["train_rows"]["refit"] = int(len(rows))
            info[sp.name]["train_pos"]["refit"] = int(y.sum())
    del X
    jobs = {sp.name: (fitted[sp.name], "iters", [info[sp.name]["selection"]["common_n"]]) if sp.name in bnames
            else (fitted[sp.name], "plain", None) for sp in todo}
    res = eval_pairs_chain(jobs, asm, ctx.samples, "refit", lead.W, check=check, max_pairs_per_h=mp, seed=ctx.seed,
                           dedupe_rows=dedupe, copy_inputs=copy)
    for sp in todo:
        if sp.name in bnames:
            res["iters"][sp.name][info[sp.name]["selection"]["common_n"]].to_parquet(
                pred_path(vdirs[sp.name], "refit"), index=False)
        else:
            res["plain"][sp.name].to_parquet(pred_path(vdirs[sp.name], "refit"), index=False)
        timings[sp.name]["refit"] = {"train_rows": int(len(rows)), "train_pos": int(y.sum()), "n_pairs": res["n_pairs"],
                                     "fit_seconds": fit_s[sp.name], "predict_seconds": res["seconds"][sp.name],
                                     "shared_train_input_seconds": t_tr,
                                     "shared_eval_input_seconds": res["input_seconds"]}
    for sp in todo:      # fold 순서대로(표시용)
        timings[sp.name] = {s: timings[sp.name][s] for s in SPLIT_ORDER}
    return timings, info


def n_iter_summary(ch: Mapping) -> dict:
    s = ch["selection"]
    return {"max_iter": int(ch["max_iter"]), "n_points": len(ch["points"]), "cross_n": dict(s["cross_n"]),
            "common_n": int(s["common_n"]), "own_best_n": dict(s["own_best_n"]),
            "at_cap": bool(s["any_at_cap"]), "train_rows": dict(ch["train_rows"]),
            "learning_rate": (ch.get("learning_rate") or {}).get("mean"),
            "max_iter_decided_here": bool(ch["max_iter_source"]["decided_here"])}


def write_n_iter(vdir: Path, ch: Mapping) -> None:
    """설정 덤프(§1-2 'fold 별 교차 N·최적 N·train 행 수와 공통 반복 수 N 을 설정 덤프에 기록') + 곡선 표."""
    body = {k: v for k, v in ch.items() if k not in ("curves", "by_h")}
    body["summary"] = n_iter_summary(ch)
    cf.dump_scores_json(vdir / N_ITER_FILE, body)
    pts = ch["points"]
    df = pd.DataFrame({"n_iter": pts, **{f: ch["curves"][f] for f in ev.CV_FOLDS}})
    df["mean_3fold"] = df[list(ev.CV_FOLDS)].mean(axis=1)
    for f in ev.CV_FOLDS:
        df[f"mean_without_{f}"] = df[[g for g in ev.CV_FOLDS if g != f]].mean(axis=1)
    with open(vdir / N_ITER_CURVES_FILE, "x", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False, float_format="%.17g")
    rows = []
    for f in ev.CV_FOLDS:
        for n, r in zip(pts, ch["by_h"][f]):
            rows.append({"fold": f, "n_iter": n, **{f"h{h:02d}": v for h, v in enumerate(r, 1)}})
    with open(vdir / N_ITER_BY_H_FILE, "x", newline="", encoding="utf-8-sig") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False, float_format="%.17g")


def _check_chain_cv(name: str, cv: Mapping, ch: Mapping, tol: float = 1e-12) -> None:
    """저장한 교차 N 예측으로 낸 fold 점수 = 곡선의 교차 N 값(같은 예측 — 다르면 멈춤)."""
    for f in ev.CV_FOLDS:
        a, b = float(cv["fold_band_weighted"][f]), float(ch["selection"]["fold_score"][f])
        if not abs(a - b) <= tol:
            raise AssertionError(f"{name} {f}: 저장 예측 fold 점수 {a} ≠ 곡선 교차 N 점수 {b}")


def load_n_iter(vdir: Path) -> dict | None:
    p = Path(vdir) / N_ITER_FILE
    return read_json(p) if p.exists() else None


