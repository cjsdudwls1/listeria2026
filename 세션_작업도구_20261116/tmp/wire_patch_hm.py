from pathlib import Path

p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\runner.py")
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''def pred_path(vdir: Path, split: str) -> Path:''', '''H_METRICS_FILE = "h_metrics.csv"


def write_h_metrics(vdir: Path, df: pd.DataFrame) -> Path:
    """판 폴더 h 별 표(utf-8-sig). 보고 때 test 행을 더하면 CV 행이 그대로인지 확인한 뒤 바꿔 씀(원자적 교체)."""
    p = Path(vdir) / H_METRICS_FILE
    if p.exists():
        old = pd.read_csv(p, encoding="utf-8-sig")
        cv_old = old[old["split"] != "test"].reset_index(drop=True)
        cv_new = df[df["split"] != "test"].reset_index(drop=True)
        pd.testing.assert_frame_equal(cv_old, cv_new.astype(cv_old.dtypes.to_dict()), check_exact=False, rtol=0,
                                      atol=1e-12)
        if (old["split"] == "test").any() and not (df["split"] == "test").any():
            raise AssertionError("test 행이 있는 h 별 표를 test 없는 표로 바꾸지 않음")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False)
    os.replace(tmp, p)
    return p


def pred_path(vdir: Path, split: str) -> Path:''')
rep('''            cf.dump_scores_json(vdir / "cv_scores.json", cv)
            tm = timings[sp.name]''', '''            cf.dump_scores_json(vdir / "cv_scores.json", cv)
            write_h_metrics(vdir, ev.h_metrics_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples,
                                                  float(cv["threshold"]["threshold"]),
                                                  require_complete=smoke is None))
            tm = timings[sp.name]''')
rep('''                for f in ("spec.json", "run_meta.json", "cv_scores.json"):
                    tr.log_artifact(run_id, vdir / f)''', '''                for f in ("spec.json", "run_meta.json", "cv_scores.json", H_METRICS_FILE):
                    tr.log_artifact(run_id, vdir / f)''')
rep('''            cf.dump_scores_json(tpath, t)
            if tracker is not None and done.get("mlflow_run_id"):
                tracker.log_test(done["mlflow_run_id"], t, permit)
                tracker.log_artifact(done["mlflow_run_id"], tpath)
            scored.append(n)''', '''            cf.dump_scores_json(tpath, t)
            if tracker is not None and done.get("mlflow_run_id"):
                tracker.log_test(done["mlflow_run_id"], t, permit)
                tracker.log_artifact(done["mlflow_run_id"], tpath)
            scored.append(n)
        thr = float(cv["threshold"]["threshold"])
        full = not done.get("smoke")
        hm = pd.concat([ev.h_metrics_cv(load_preds(vdir, ev.CV_FOLDS), samples, thr, require_complete=full),
                        ev.h_metrics_test(pd.read_parquet(pred_path(vdir, "refit")), samples, truth, thr, inc,
                                          require_complete=full)], ignore_index=True)
        hp = write_h_metrics(vdir, hm)   # 이전 코드로 끝난 판(표 없음)도 저장 예측으로 만듦
        if tracker is not None and done.get("mlflow_run_id"):
            tracker.log_artifact(done["mlflow_run_id"], hp)''')
p.write_text(s, encoding="utf-8")
print("ok")
