# -*- coding: utf-8 -*-
"""s_x3.py — 추가 판 ③ 시료 무작위 나누기(1차 뒤) — 계획서 §6 추가 판 표 ③ 줄, §1-2(지표·threshold), §1-3(test 막기),
부록 A-9(③ = 3)·A-14(판 이름 x3-<1차 이름>)·A-15(listeria-1차 · 단계 run '③').

하는 일(계획서 ③ 줄 그대로):
  1차 베이스라인 설정(1-3 결정 baseline — 트리면 1-1, LSTM 이면 1-2 의 그 판 spec.json) 하나를 시료 무작위 3겹 CV 로 다시
  (1차 규칙 — 라이브러리 기본 설정, 반복 수 고르기 없음, CatBoost 학습률 = 학습마다 자동값):
  대상 = d ≤ 2025-06-30 시료(행 처리 뒤 — 시료 표 known_frame), 시료 단위 무작위 3겹(씨앗 42),
  학습 = 겹 train 시료당 자기 h 1개, 채점 = 겹 validation 시료의 (시료, h 1~60) 쌍 전부(h 마다 메모리에서 입력 → 바로
  예측, 디스크 캐시 없음 — B-1), 겹 3개 구간 가중 PR-AUC 평균 = 무작위 CV.
  보조 점수 = 겹 validation 쌍 중 T 의 월일이 07-01 ~ 05-01 인 쌍만(시간순 CV 와 같은 채점 계절).
  test = 1-1·1-2 의 같은 설정 refit 예측(pred_refit.parquet) 그대로(추가 학습 없음), test threshold = 무작위 겹
  validation 예측을 합쳐 §1-2 규칙(metrics.select_threshold), 시간순 threshold(그 판 cv_scores.json)도 같이 적음.
  test 채점은 --report 때만(보고 허가 — §1-3).

결과(--out = 단계 폴더, 예: results/s1/x3):
  _code/                      실행 코드 사본 + code_md5.txt(runner.snapshot_code — 다르면 멈춤)
  x3-<1차 이름>/fold_assign.parquet   lims_data_key·fold(x3f1~x3f3)
               x3f<k>/model·inputs·pred.parquet(lims_data_key·h·fold·pred)·done.json   겹마다(끝난 겹은 건너뜀)
               scores.json·table_cv.csv·run_meta.json·done.json         CV(겹 3개 끝난 뒤)
               test_scores.json·table_report.csv                        --report 때만

실행(서버 — 검수 뒤 지휘자, 1-3 뒤):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s_x3.py $A --out ~/cyj_listeria_final/results/s1/x3 --config configs/s_x3.json
  보고: 같은 명령 + --report --reason "추가 판 ③ 보고"
  (베이스라인이 LSTM 이면 GPU — 설정 gpu_memory_fraction ≤ 0.4)

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 겹 나누기: known_frame 순서(시료 표 순서)의 시료 n 개에 `perm = np.random.default_rng(42).permutation(n)`,
     겹 번호 fold[perm[i]] = i % 3 + 1(겹 크기 차이 ≤ 1). fold_assign.parquet 에 저장, 다시 돌리면 같은지 확인.
 (2) 입력 조립 판(split) = 'refit'(학습 끝 2025-06-30) — 1차 입력엔 이력 칼럼이 없어 값과 무관, 누수 점검(C ≤ 학습 끝)만.
 (3) 실험 로그 '같은 validation 으로 고른 횟수' — 시간순 3 fold 와 다른 validation 이라 kind '무작위 겹' 으로 따로 셈
     (단계 '③', 1 설정). 시간순 셈(kind '판')에는 안 더함.
 (4) 보조 점수의 '월일 07-01 ~ 05-01' = T 의 월일이 05-02 ~ 06-30 이 아닌 쌍(연도 무관, 07-01·05-01 포함).
 (5) 겹 하나 = 이 프로세스 안에서 학습(하위 프로세스 아님) — 멈추면 그 겹 폴더를 _incomplete 로 옮기고 그 겹부터 다시(B-1).
 (6) MLflow 판 run 은 겹 3개가 다 끝난 뒤 한 번에(점수·artifact), 판정 tag '해당 없음'·역할 '비교만'·바탕 판 = 1차 이름.
     test 지표: 무작위 threshold = test_*(tracking.log_test), 시간순 threshold = test_chrono_* metric.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import sys
import time
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import scope as lst_scope  # noqa: E402
from lst import evaluate as ev  # noqa: E402
from lst import folds as fd  # noqa: E402
from lst import metrics as mt  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import stage_cli as sc  # noqa: E402
from lst.hassign import H_MAX, u_tie_many  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import GPU_MEMORY_FRACTION, check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STAGE = "③"
EXPERIMENT = "listeria-1차"
KIND_LOG = "무작위 겹"                # 구현 세부 (3)
N_FOLDS = 3
SEED_FOLDS = 42                       # 계획서 ③: 씨앗 42
FOLDS = tuple(f"x3f{k}" for k in range(1, N_FOLDS + 1))
SPLIT_FOR_INPUTS = "refit"            # 구현 세부 (2)
COMPONENTS_TREE = ("weather",)
COMPONENTS_DL = ("weather", "missing")
SCHEMA = {
    "plan": cf.Field(str),
    "stage": cf.Field(str, choices=(STAGE,)),
    "order": cf.Field(str, choices=("1차",)),
    "experiment": cf.Field(str, choices=(EXPERIMENT,)),
    "samples": cf.Field(str),
    "incomplete_months": cf.Field(str),
    "experiment_log": cf.Field(str),
    "n_jobs": cf.Field(int),
    "seed": cf.Field(int),
    "use_mlflow": cf.Field(bool),
    "mlflow_uri": cf.Field(str),
    "code_reviewed": cf.Field(bool),
    "decision_1_3": cf.Field(str),
    "dir_1_1": cf.Field(str),
    "dir_1_2": cf.Field(str),
    "context_tree": cf.Field(dict),
    "context_lstm": cf.Field(dict),
    "gpu_memory_fraction": cf.Field(float),
    "smoke": {"max_train_rows": cf.Field(int, required=False), "max_pairs_per_h": cf.Field(int, required=False),
              "max_epochs": cf.Field(int, required=False)},
}
SCHEMA = lst_scope.with_field(SCHEMA)   # 국내만 버전 scope 칸(없으면 지금 판)


class X3Error(RuntimeError):
    pass


# ================================================================ 설정·바탕 판
def load_cfg(path) -> dict:
    raw = cf.load_json_strict(path)
    schema = {k: v for k, v in SCHEMA.items() if k != "smoke" or "smoke" in raw}
    cfg = dict(cf.validate(raw, schema))
    from lst.resources import check_n_jobs

    check_n_jobs(cfg["n_jobs"])
    if cfg["seed"] != 42:
        raise cf.ConfigError("씨앗은 42(§8·부록 B-5)")
    if not (0 < cfg["gpu_memory_fraction"] <= GPU_MEMORY_FRACTION):
        raise cf.ConfigError(f"gpu_memory_fraction 은 0 초과 {GPU_MEMORY_FRACTION} 이하")
    if "smoke" in cfg and cfg["use_mlflow"] and not cfg["mlflow_uri"].startswith(sc.SMOKE_ONLY_URI):
        raise cf.ConfigError("연기 시험(smoke) 설정은 로컬 sqlite MLflow 에만")
    return cfg


def baseline_source(cfg, inputs_root: Path) -> tuple[str, Path, rn.VariantSpec]:
    """1-3 결정 baseline → (이름, 판 폴더, 판 설정). 끝난 판이어야 함."""
    dec = cf.load_json_strict(resolve(inputs_root, cfg["decision_1_3"]))
    if "baseline" not in dec:
        raise X3Error("1-3 결정 json 에 baseline 없음")
    name = dec["baseline"]
    rel = cfg["dir_1_2"] if name.startswith("1-LSTM-") else cfg["dir_1_1"]
    vdir = resolve(inputs_root, rel) / name
    for f in ("spec.json", "done.json", "cv_scores.json", "pred_refit.parquet"):
        if not (vdir / f).exists():
            raise X3Error(f"1차 베이스라인 판 파일 없음: {vdir / f}")
    spec = rn.VariantSpec(**rn.read_json(vdir / "spec.json"))
    if spec.order != "1차" or spec.name != name:
        raise X3Error(f"1차 판이 아님: {spec.name} {spec.order}")
    return name, vdir, spec


def variant_name(base: str) -> str:
    return f"x3-{base}"


# ================================================================ 겹 나누기·행
def assign_folds(keys: np.ndarray) -> np.ndarray:
    """구현 세부 (1): 시료 표 순서 키 → 겹 번호 1~3."""
    n = len(keys)
    perm = np.random.default_rng(SEED_FOLDS).permutation(n)
    out = np.empty(n, dtype=np.int64)
    out[perm] = np.arange(n) % N_FOLDS + 1
    return out


def fold_table(known: pd.DataFrame) -> pd.DataFrame:
    keys = known["lims_data_key"].astype(str).to_numpy(dtype=object)
    return pd.DataFrame({"lims_data_key": keys, "fold": [FOLDS[k - 1] for k in assign_folds(keys)]})


def train_rows(known: pd.DataFrame, assign: pd.DataFrame, fold: str, W: int | None) -> pd.DataFrame:
    """겹 train 시료당 자기 h 1개(folds.make_train_rows 와 같은 h·T) + E·S·g·R."""
    m = (assign["fold"] != fold).to_numpy()
    rows = fd.make_train_rows(known.loc[m], fd.SPLITS["refit"])
    return rn.add_bounds(rows, W)


def _aux_mask(T_days: np.ndarray) -> np.ndarray:
    """구현 세부 (4): T 의 월일이 05-02 ~ 06-30 이 아닌 쌍."""
    t = np.asarray(T_days, dtype=np.int64).astype("datetime64[D]")
    mon = t.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day = (t - t.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1
    off = ((mon == 5) & (day >= 2)) | (mon == 6)
    return ~off


def predict_val(model, asm, samples: ev.SampleTable, keys: np.ndarray, fold: str, W: int | None,
                check=None, max_pairs_per_h: int | None = None) -> pd.DataFrame:
    """겹 validation 시료 × h 1~60 — h 마다 메모리에서 입력 → 바로 예측."""
    import contextlib

    pos = samples.positions(keys)
    meta = samples.meta().iloc[pos].reset_index(drop=True)
    kk, hh, pp = [], [], []
    scope = asm.pair_cache() if hasattr(asm, "pair_cache") else contextlib.nullcontext()
    with scope:
        for h in range(1, H_MAX + 1):
            if check is not None:
                check()
            idx = np.arange(len(meta))
            if max_pairs_per_h is not None and idx.size > max_pairs_per_h:
                idx = np.sort(np.random.default_rng([42, h]).choice(idx, max_pairs_per_h, replace=False))
            m = meta.iloc[idx].copy()
            m["h"] = np.int16(h)
            m = rn.add_bounds(m, W)
            p = rn._check_proba(model.predict_proba(asm.transform(m)), len(m))
            kk.append(m["lims_data_key"].astype(str).to_numpy(dtype=object))
            hh.append(np.full(len(m), h, np.int16))
            pp.append(p)
    return pd.DataFrame({"lims_data_key": np.concatenate(kk), "h": np.concatenate(hh), "fold": fold,
                         "pred": np.concatenate(pp)})


# ================================================================ 채점(정답 = d ≤ 2025-06-30 만)
def score_fold(pred: pd.DataFrame, samples: ev.SampleTable) -> dict:
    keys = pred["lims_data_key"].astype(str).to_numpy(dtype=object)
    if pd.DataFrame({"k": keys, "h": pred["h"]}).duplicated().any():
        raise X3Error("(시료, h) 중복 예측")
    pos = samples.positions(keys)
    y = samples.cv_y(pos)
    h = pred["h"].to_numpy(dtype=np.int64)
    s = pred["pred"].to_numpy(dtype=np.float64)
    u = u_tie_many(keys, h)
    rep = mt.score_report(y, s, h, u)
    aux = _aux_mask(samples.d_days[pos] - h)
    rep["aux_0701_0501_band_weighted"] = mt.pr_auc_band_weighted(y[aux], s[aux], h[aux], u[aux])
    rep["aux_0701_0501_n_pairs"] = int(aux.sum())
    return rep


def score_cv(preds: dict[str, pd.DataFrame], samples: ev.SampleTable) -> dict:
    folds = {f: score_fold(preds[f], samples) for f in FOLDS}
    allp = pd.concat([preds[f] for f in FOLDS], ignore_index=True)
    y = samples.cv_y(samples.positions(allp["lims_data_key"].astype(str).to_numpy(dtype=object)))
    thr = mt.select_threshold(y, allp["pred"].to_numpy(dtype=np.float64))
    return {"kind": "x3_cv", "cv_band_weighted": mt.cv_mean({f: folds[f]["pr_auc_band_weighted"] for f in FOLDS}),
            "fold_band_weighted": {f: folds[f]["pr_auc_band_weighted"] for f in FOLDS},
            "cv_pooled": mt.cv_mean([folds[f]["pr_auc_pooled"] for f in FOLDS]),
            "cv_aux_0701_0501_band_weighted": mt.cv_mean([folds[f]["aux_0701_0501_band_weighted"] for f in FOLDS]),
            "threshold": thr, "n_pairs": int(len(allp)), "n_pos": int(y.sum()), "folds": folds}


# ================================================================ 겹 하나(이어하기)
def run_fold(fold, *, spec, fac, samples, known, assign, vroot: Path, code_md5: str, cfg, exp_log, check) -> dict:
    fdir = vroot / fold
    done = fdir / "done.json"
    smd5 = spec.md5()
    if done.exists():
        d = rn.read_json(done)
        if d["spec_md5"] != smd5 or d["code_md5"] != code_md5:
            raise rn.ResumeError(f"{fold}: 끝난 겹의 설정·code_md5 가 지금과 다름 — 지휘자 결정")
        return d
    if fdir.exists():
        dst = vroot / "_incomplete" / f"{fold}__{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fdir), str(dst))
        exp_log.append("rerun", stage=STAGE, name=f"{variant_name(spec.name)}/{fold}", reason="멈춘 겹 다시",
                       moved_to=dst.name)
    fdir.mkdir(parents=True)
    smoke = cfg.get("smoke") or {}
    try:
        check()
        t0 = time.time()
        rows = train_rows(known, assign, fold, spec.W)
        if smoke.get("max_train_rows"):
            rows = rn._smoke_train(rows, smoke["max_train_rows"], 42)
        y = rows["y"].to_numpy(dtype=np.int64)
        asm = fac.make_inputs(spec, SPLIT_FOR_INPUTS).fit(rows)
        X = asm.transform(rows)
        t_in = time.time() - t0
        model = fac.make_model(spec, asm)
        kw = dict(fac.fit_kwargs(spec, SPLIT_FOR_INPUTS, rows, X)) if fac.fit_kwargs else {}
        t1 = time.time()
        model.fit(X, y, seed=cfg["seed"], n_jobs=cfg["n_jobs"], **kw)
        t_fit = time.time() - t1
        del X
        model.save(fdir / "model")
        if hasattr(asm, "save"):
            asm.save(fdir / "inputs")
        val_keys = assign.loc[assign["fold"] == fold, "lims_data_key"].to_numpy(dtype=object)
        t2 = time.time()
        pred = predict_val(model, asm, samples, val_keys, fold, spec.W, check=check,
                           max_pairs_per_h=smoke.get("max_pairs_per_h"))
        pred.to_parquet(fdir / "pred.parquet", index=False)
        rec = {"fold": fold, "spec_md5": smd5, "code_md5": code_md5, "train_rows": int(len(rows)),
               "train_pos": int(y.sum()), "n_val_samples": int(len(val_keys)), "n_pairs": int(len(pred)),
               "seconds": {"train_inputs": round(t_in, 2), "fit": round(t_fit, 2), "predict": round(time.time() - t2, 2)},
               "smoke": smoke or None}
        cf.dump_json(done, rec)
        exp_log.append("done", stage=STAGE, name=f"{variant_name(spec.name)}/{fold}", n_pairs=rec["n_pairs"])
        return rec
    except BaseException as e:
        cf.dump_json(fdir / "failed.json", {"type": type(e).__name__, "message": str(e)[:2000],
                                            "traceback": traceback.format_exc()[-8000:]}, overwrite=True)
        exp_log.append("failed", stage=STAGE, name=f"{variant_name(spec.name)}/{fold}", type=type(e).__name__,
                       message=str(e)[:500])
        raise


def factories(cfg, rp, inputs_root: Path, spec: rn.VariantSpec) -> sc.Factories:
    from lst import wiring as wr

    is_dl = spec.kind == "dl"
    ctxc = cfg["context_lstm"] if is_dl else cfg["context_tree"]
    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    w = wr.load(ctxc, rp.root, inputs_root, COMPONENTS_DL if is_dl else COMPONENTS_TREE, dl_kwargs=dl_kw)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(ctxc, rp.root, inputs_root), fit_kwargs=w.fit_kwargs)


# ================================================================ 진입점
def _table_cv(name, base, base_cv, sc_) -> pd.DataFrame:
    return pd.DataFrame([
        {"판": base, "나누기": "시간순 3 fold(1-1·1-2)", "CV 구간 가중(평균)": base_cv["cv_band_weighted"],
         **{f"fold {i}": base_cv["fold_band_weighted"][f] for i, f in enumerate(ev.CV_FOLDS, 1)},
         "보조(T 월일 07-01~05-01)": None, "threshold": base_cv["threshold"]["threshold"]},
        {"판": name, "나누기": "시료 무작위 3겹(씨앗 42)", "CV 구간 가중(평균)": sc_["cv_band_weighted"],
         **{f"fold {i}": sc_["fold_band_weighted"][f] for i, f in enumerate(FOLDS, 1)},
         "보조(T 월일 07-01~05-01)": sc_["cv_aux_0701_0501_band_weighted"],
         "threshold": sc_["threshold"]["threshold"]}])


def main(argv=None, *, factories_fn=None, tracker_fn=sc.make_tracker, check_resources_fn=None,
         before_fit_fn=None):
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reason")
    rp, args = parse_paths(argv, ap)
    cfg = load_cfg(args.config)
    lst_scope.activate(cfg.get("scope"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)
    inputs_root = Path(args.inputs).expanduser().resolve()
    samples_path = resolve(inputs_root, cfg["samples"])
    base, bdir, spec = baseline_source(cfg, inputs_root)
    name = variant_name(base)
    out = rp.out
    vroot = out / name
    exp_log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    cr = check_resources_fn or check_resources
    check = lambda: cr(out)  # noqa: E731
    if args.report:
        return report(cfg, args, inputs_root, samples_path, base, bdir, name, vroot, exp_log, tracker_fn)

    code_md5 = rn.snapshot_code(CODE_ROOT, out)
    samples = ev.SampleTable.read(samples_path)
    known = samples.known_frame().reset_index(drop=True)
    assign = fold_table(known)
    ap_ = vroot / "fold_assign.parquet"
    if ap_.exists():
        old = pd.read_parquet(ap_)
        if not old.equals(assign):
            raise rn.ResumeError("저장된 겹 나누기와 지금 나누기가 다름")
    else:
        vroot.mkdir(parents=True, exist_ok=True)
        assign.to_parquet(ap_, index=False)
    if (vroot / "done.json").exists():
        print(f"{name}: 이미 끝남 — CV {rn.read_json(vroot / 'done.json')['cv_band_weighted']}")
        return rn.read_json(vroot / "done.json")
    if spec.kind == "dl" and before_fit_fn is None:
        from lst.resources import limit_gpu_memory

        limit_gpu_memory(cfg["gpu_memory_fraction"])
    elif before_fit_fn is not None:
        before_fit_fn(cfg)
    fac = (factories_fn or factories)(cfg, rp, inputs_root, spec)
    recs = {f: run_fold(f, spec=spec, fac=fac, samples=samples, known=known, assign=assign, vroot=vroot,
                        code_md5=code_md5, cfg=cfg, exp_log=exp_log, check=check) for f in FOLDS}
    preds = {f: pd.read_parquet(vroot / f / "pred.parquet") for f in FOLDS}
    full = "smoke" not in cfg or not cfg["smoke"].get("max_pairs_per_h")
    if full:
        for f in FOLDS:
            want = int((assign["fold"] == f).sum()) * H_MAX
            if len(preds[f]) != want:
                raise X3Error(f"{f}: validation 쌍 {len(preds[f])} ≠ 시료 × 60 = {want}")
    scores = score_cv(preds, samples)
    base_cv = rn._load_scores(bdir / "cv_scores.json")
    scores["chronological"] = {"name": base, "cv_band_weighted": base_cv["cv_band_weighted"],
                               "fold_band_weighted": base_cv["fold_band_weighted"],
                               "threshold": base_cv["threshold"]["threshold"]}
    count = exp_log.cv_scored(STAGE, name, 1, kind=KIND_LOG)
    cf.dump_scores_json(vroot / "scores.json", scores)
    with open(vroot / "table_cv.csv", "w", newline="", encoding="utf-8-sig") as fh:
        _table_cv(name, base, base_cv, scores).to_csv(fh, index=False, float_format="%.17g")
    md5s = {**cf.inputs_md5(inputs_root, [cfg["samples"], cfg["incomplete_months"]]), **dict(fac.input_md5s),
            "base_cv_scores": cf.file_md5(bdir / "cv_scores.json"), "base_pred_refit": cf.file_md5(bdir / "pred_refit.parquet")}
    cf.save_run_meta(vroot / "run_meta.json", {"cfg": cfg, "base_spec": spec.to_dict()}, md5s, CODE_ROOT,
                     n_jobs=cfg["n_jobs"], extra={"folds": recs, "fold_seed": SEED_FOLDS, "train_count": N_FOLDS})
    run_id = None
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(EXPERIMENT, STAGE)
        run_id = tr.start_variant(EXPERIMENT, sr, name,
                                  {"단계": STAGE, "역할": "비교만", "바탕 판": base,
                                   "코드 검수 완료": "예" if cfg["code_reviewed"] else "아니오",
                                   "설정 md5": spec.md5(), "code_md5": code_md5},
                                  {"algorithm": spec.algorithm, "W": spec.W, "mode": spec.mode,
                                   "split": "시료 무작위 3겹(씨앗 42)", "seed": cfg["seed"], "train_count": N_FOLDS})
        tr.set_judgement(run_id, "해당 없음")
        tr.log_metrics(run_id, {"cv_band_weighted": scores["cv_band_weighted"],
                                **{f"{f}_band_weighted": v for f, v in scores["fold_band_weighted"].items()},
                                "cv_pooled": scores["cv_pooled"],
                                "cv_aux_0701_0501_band_weighted": scores["cv_aux_0701_0501_band_weighted"],
                                "threshold": scores["threshold"]["threshold"], "selection_count_random": count,
                                "chrono_cv_band_weighted": base_cv["cv_band_weighted"]})
        for f in ("scores.json", "table_cv.csv", "run_meta.json"):
            tr.log_artifact(run_id, vroot / f)
        tr.finish(run_id)
    done = {"name": name, "base": base, "spec_md5": spec.md5(), "code_md5": code_md5,
            "cv_band_weighted": scores["cv_band_weighted"], "selection_count_random": count, "mlflow_run_id": run_id,
            "smoke": cfg.get("smoke")}
    cf.dump_scores_json(vroot / "done.json", done)
    exp_log.append("done", stage=STAGE, name=name, cv=scores["cv_band_weighted"], kind=KIND_LOG)
    print(f"{name}: 무작위 CV {scores['cv_band_weighted']:.4f}(시간순 {base_cv['cv_band_weighted']:.4f}), "
          f"보조 {scores['cv_aux_0701_0501_band_weighted']:.4f}")
    return done


def report(cfg, args, inputs_root, samples_path, base, bdir, name, vroot: Path, exp_log, tracker_fn) -> Path:
    """test 채점(보고 허가 — §1-3): 1차 같은 설정 refit 예측 그대로, threshold 둘(무작위·시간순)."""
    if not args.reason:
        raise SystemExit("--report 에는 --reason 이 필요(실험 로그)")
    if not (vroot / "done.json").exists():
        raise SystemExit(f"{name}: 겹 3개가 끝나야 보고")
    done = rn.read_json(vroot / "done.json")
    permit = ev.report_permit(STAGE, args.reason)
    samples = ev.SampleTable.read(samples_path)
    truth = ev.TestTruth.load(samples_path, permit)
    inc = ev.load_incomplete(resolve(inputs_root, cfg["incomplete_months"]))
    scores = rn._load_scores(vroot / "scores.json")
    tp = pd.read_parquet(rn.pred_path(bdir, "refit"))
    thr_r = float(scores["threshold"]["threshold"])
    thr_c = float(scores["chronological"]["threshold"])
    full = not done.get("smoke")
    t_r = ev.score_test(tp, samples, truth, thr_r, inc, require_complete=full)
    t_c = ev.score_test(tp, samples, truth, thr_c, inc, require_complete=full)
    out = {"base": base, "threshold_random": thr_r, "threshold_chronological": thr_c,
           "test_random_threshold": t_r, "test_chronological_threshold": t_c}
    p = vroot / "test_scores.json"
    if not p.exists():
        cf.dump_scores_json(p, out)
    rows = []
    for lab, t, thr in (("무작위 겹 threshold", t_r, thr_r), ("시간순 threshold", t_c, thr_c)):
        r, tm = t["report"], t["report"]["threshold_metrics"]
        rows.append({"판": name, "threshold 출처": lab, "threshold": thr, "test 구간 가중": r["pr_auc_band_weighted"],
                     "test 60일 통": r["pr_auc_pooled"], "Recall(test 쌍 전체)": tm["recall"],
                     "Recall(구간 가중)": tm["recall_band_weighted"], "Precision": tm["precision"], "F1": tm["f1"],
                     "F2": tm["f2"], "TP": tm["tp"], "FN": tm["fn"], "FP": tm["fp"], "TN": tm["tn"],
                     "무작위 CV": scores["cv_band_weighted"], "시간순 CV": scores["chronological"]["cv_band_weighted"],
                     "보조(T 월일 07-01~05-01)": scores["cv_aux_0701_0501_band_weighted"]})
    tab = vroot / "table_report.csv"
    with open(tab, "w", newline="", encoding="utf-8-sig") as fh:
        pd.DataFrame(rows).to_csv(fh, index=False, float_format="%.17g")
    tr = tracker_fn(cfg)
    if tr is not None and done.get("mlflow_run_id"):
        tr.log_test(done["mlflow_run_id"], t_r, permit)
        tmc = t_c["report"]["threshold_metrics"]
        tr.log_metrics(done["mlflow_run_id"], {f"test_chrono_{k}": tmc[k] for k in
                                                ("threshold", "recall", "recall_band_weighted", "precision", "f1",
                                                 "f2", "tp", "fn", "fp", "tn")})
        tr.log_artifact(done["mlflow_run_id"], tab)
    exp_log.append("test_reported", stage=STAGE, names=[name], permit_reason=permit.reason)
    print(f"보고 표: {tab}")
    return tab


if __name__ == "__main__":
    main()
