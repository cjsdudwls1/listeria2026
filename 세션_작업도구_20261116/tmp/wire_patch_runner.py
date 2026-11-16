from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\runner.py"); s = p.read_text(encoding="utf-8")


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''MakeModel = Callable[[VariantSpec], Model]''', '''MakeModel = Callable[[VariantSpec, InputAssembler], Model]   # (판 설정, fit 한 조립 객체 — 딥러닝 cat_sizes 등)''')
rep('''    def cv_scored(self, stage: str, name: str, n_settings: int = 1, kind: str = "판") -> int:
        """같은 3 fold 로 CV 점수를 낸 판·설정 수 누적(부록 A-12) — 이 기록 포함 값을 돌려줌.
        같은 (단계, 이름, 종류) 기록이 이미 있으면 그 값(두 번 세지 않음)."""''',
    '''    def cv_scored(self, stage: str, name: str, n_settings: int = 1, kind: str = "판") -> int:
        """'같은 validation 으로 고른 횟수'(부록 A-12: "그 판 앞까지 같은 3 fold 로 CV 점수를 낸 모든 판·설정 수의 누적")
        — 이 기록을 남기고, 그 앞 기록들의 합(이 판 자신은 뺌)을 돌려줌. 기록의 count = 이 기록까지 누적.
        같은 (단계, 이름, 종류) 기록이 이미 있으면 그 값(두 번 세지 않음)."""''')
rep('''                if (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                    return int(e["count"])
                total = int(e["count"])
            rec = self._append({"event": "cv_scored", "stage": stage, "name": name, "kind": kind,
                                "n_settings": n_settings, "count": total + n_settings})
            return rec["count"]''', '''                if (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                    return int(e["count_before"])
                total = int(e["count"])
            rec = self._append({"event": "cv_scored", "stage": stage, "name": name, "kind": kind,
                                "n_settings": n_settings, "count_before": total, "count": total + n_settings})
            return rec["count_before"]''')
rep('''            if e.get("event") == "cv_scored" and (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                return int(e["count"])''', '''            if e.get("event") == "cv_scored" and (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                return int(e["count_before"])''')
rep('''def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
                  check: Callable[[], Any] | None = None) -> pd.DataFrame:
    """평가 쌍을 h 마다 메모리에서 만들어 바로 예측(B-1). 행 표에 y 없음."""
    sp = fd.SPLITS[split]
    meta = samples.meta()
    pos_l, h_l, p_l = [], [], []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, sp):
        if idx.size == 0:
            continue''', '''def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
                  check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """평가 쌍을 h 마다 메모리에서 만들어 바로 예측(B-1). 행 표에 y 없음.
    max_pairs_per_h = 연기 시험용(쌍 일부, 씨앗 고정) — 실제 실행은 None(쌍 전부, §3)."""
    sp = fd.SPLITS[split]
    meta = samples.meta()
    pos_l, h_l, p_l = [], [], []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, sp):
        if max_pairs_per_h is not None and idx.size > max_pairs_per_h:
            idx = np.sort(np.random.default_rng([seed, h]).choice(idx, max_pairs_per_h, replace=False))
        if idx.size == 0:
            continue''')
rep('''    check_resources: Callable[[Any], Any] = _check_resources
    splits: Sequence[str] = SPLIT_ORDER
    log_pred_artifacts: bool = True
''', '''    check_resources: Callable[[Any], Any] = _check_resources
    splits: Sequence[str] = SPLIT_ORDER
    log_pred_artifacts: bool = True
    smoke: Mapping[str, int] | None = None   # 연기 시험만: {max_train_rows, max_pairs_per_h} — 실제 실행은 None


SMOKE_KEYS = ("max_train_rows", "max_pairs_per_h")


def _smoke_train(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """연기 시험용 train 행 줄이기(씨앗 고정, 검출 행은 다 남김 — 수천 행에서 검출 0 이 되지 않게)."""
    if len(rows) <= n:
        return rows
    y = rows["y"].to_numpy()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y != 1)
    k = max(0, n - pos.size)
    keep = np.sort(np.concatenate([pos, np.random.default_rng(seed).choice(neg, min(k, neg.size), replace=False)]))
    return rows.iloc[keep]
''')
rep('''    if tuple(ctx.splits) != SPLIT_ORDER:
        raise ValueError("판 하나 = fold1~3 + refit")''', '''    if tuple(ctx.splits) != SPLIT_ORDER:
        raise ValueError("판 하나 = fold1~3 + refit")
    smoke = dict(ctx.smoke) if ctx.smoke else None
    if smoke is not None and set(smoke) != set(SMOKE_KEYS):
        raise ValueError(f"smoke 칸은 {SMOKE_KEYS}")''')
rep('''            rows = fd.make_train_rows(known, fd.SPLITS[split])
            rows = add_bounds(rows, spec.W)''', '''            rows = fd.make_train_rows(known, fd.SPLITS[split])
            if smoke is not None:
                rows = _smoke_train(rows, smoke["max_train_rows"], ctx.seed)
            rows = add_bounds(rows, spec.W)''')
rep('''            model = ctx.make_model(spec)''', '''            model = ctx.make_model(spec, asm)''')
rep('''            if hasattr(model, "applied_params"):
                cf.dump_json(vdir / f"applied_params_{split}.json", cf.nan_to_null(model.applied_params()))
            preds = predict_pairs(model, asm, ctx.samples, split, spec.W, check=lambda: ctx.check_resources(out_dir))''',
    '''            dump = _model_dump(model)   # B-4 실제 적용값
            if dump is not None:
                cf.dump_json(vdir / f"applied_params_{split}.json", cf.nan_to_null(dump))
            preds = predict_pairs(model, asm, ctx.samples, split, spec.W, check=lambda: ctx.check_resources(out_dir),
                                  max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None, seed=ctx.seed)''')
rep('''        cv = ev.score_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples)''',
    '''        cv = ev.score_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples, require_complete=smoke is None)''')
rep('''                         extra={"timings": timings, "seconds_total": total_s, "resources_at_start": res0,
                                "spec_md5": smd5})''', '''                         extra={"timings": timings, "seconds_total": total_s, "resources_at_start": res0,
                                "spec_md5": smd5, "smoke": smoke})''')
rep('''                "selection_count": count, "mlflow_run_id": run_id, "seconds_total": total_s}''',
    '''                "selection_count": count, "mlflow_run_id": run_id, "seconds_total": total_s, "smoke": smoke}''')
rep('''def _lib_versions() -> dict:''', '''def _model_dump(model) -> Any:
    """B-4 실제 적용값: models_tree.config_dump·models_dl.get_config(있는 것)."""
    for m in ("config_dump", "get_config", "applied_params"):
        if hasattr(model, m):
            return getattr(model, m)()
    return None


def _lib_versions() -> dict:''')
rep('''            t = ev.score_test(pd.read_parquet(pred_path(vdir, "refit")), samples, truth,
                              float(cv["threshold"]["threshold"]), inc)''', '''            t = ev.score_test(pd.read_parquet(pred_path(vdir, "refit")), samples, truth,
                              float(cv["threshold"]["threshold"]), inc, require_complete=not done.get("smoke"))''')
p.write_text(s, encoding="utf-8")
print("ok")
