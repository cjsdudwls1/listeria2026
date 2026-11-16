"""단계 8 튜닝·불균형 — 계획서 §6 단계 8, 부록 A-7(탐색 범위·AutoKeras), A-12 '단계 8 점수', §3(반복 수 규칙),
부록 B-1(멈추면 그 시도부터 다시)·B-4(실제 적용값 기록)·B-5(씨앗).

트리(8-T): Optuna TPE(씨앗 42) 99 시도 + 첫 시도(대상 판 설정 — 점수만, TPE 이력에 안 넣음), 목표 = CV 평균 구간 가중
PR-AUC(§3 교차 N 규칙 — ④ 는 라이브러리 기본 반복 수·learning_rate 탐색 안 함), 검출 행 가중치 탐색 포함.
Focal: CatBoost 손실만 Focal(γ 2·α 0.75). 딥러닝(8-TD): AutoKeras(Keras 3) max_trials 19·씨앗 42·greedy, objective =
조기 종료용 (sgg, d) 묶음 10% 의 val_loss, 시도 점수 = trial 체크포인트로 1년 검증 쌍(T ∈ 2021-07-01 ~ 2022-05-01)
구간 가중 PR-AUC, 기본 설정 = 20번째 후보.

공개 API(0-9 스크립트·단계 8 이 씀):
  범위:     search_space(algo, *, rep=1, tune_lr=True) · suggest_params(trial, algo, *, rep, tune_lr) ·
            model_params(algo, raw) — raw = Optuna 값, 반환 = models_tree.make_model(params=…)
  트리 점수: TreeFold(name, samples, assembler, *, split=None, train_rows=None, cache_val=True, check=None)
            score_tree_config(algo, params, folds, *, max_iter, seed=42, n_jobs=8, check=None) → dict
  Optuna:   run_optuna(algo, score_fn, *, log_path, n_trials=99, seed=42, rep=1, tune_lr=True, stop_after=None) → 기록 list
            select_tuned(baseline_cv, records, top=5) → dict
  Focal:    focal_params(params) · focal_supported(n_jobs=1)
  AutoKeras: AK_PROPOSAL · ak_mapping(arch, mapping=None) · tune_train_rows(samples) · TuneValidation(samples, assembler)
            autokeras_search(arch, inputs, y, groups, *, directory, project_name, cat_sizes=None, train_d=None,
                             max_trials=19, seed=42, mapping=None, overwrite=False, …) → AKSearch
            score_ak_trials(search, val) → list · score_predict_fn(predict_fn, val) → dict · select_ak(trial_scores, default_score)
            rebuild_selected(search, trial_id) → 컴파일된 Keras 모델 · fit_keras_model(model, inputs, y, groups, …)

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 범위 꼴: '(로그)' = Optuna log=True, 그 밖 균등. 정수 = depth·max_depth·num_leaves·min_child_samples·min_samples_leaf,
     실수 = 나머지(XGBoost min_child_weight 포함). RF·ET max_depth = 'max_depth_unlimited'(True/False) 를 먼저 뽑고
     False 면 max_depth 4~30. 양성 가중 = class_weight {0: 1, 1: w}. 고정 = CatBoost bootstrap_type 'Bernoulli',
     LightGBM bagging_freq 1. 4-a 자료면 행 수 기준 범위(min_child_samples·min_child_weight·min_samples_leaf)의
     양 끝 × rep(정수는 그대로 정수).
 (2) TPE = optuna.samplers.TPESampler(seed=42) 기본값, maximize, 목표값 = 반올림 안 한 CV 평균. 이어하기 =
     시도마다 jsonl 한 줄(raw·점수) — 다시 시작하면 같은 sampler 로 ask 를 되풀이해 기록과 같은 값인지 확인하고 기록 점수를
     tell(되풀이 불일치면 멈춤) → 멈춘 시도부터 새로 돎(B-1 — 건너뛰지 않음). 시도 점수가 NaN 이면 멈춤.
 (3) 고르기: 시도 점수 소수 4자리 반올림 큰 순 → 시도 번호 작은 순, 첫 시도(대상 판 설정) 점수보다 반올림 뒤 높을 때만
     채택(같으면 기본 유지 — §1-2).
 (4) TreeFold 의 평가 쌍 입력은 cache_val=True 면 시도 사이에 메모리에 둠(디스크 없음 — B-1), False 면 시도마다 h 별로
     다시 만듦. 범주 칼럼 = X 의 object 칼럼, 로지스틱 결측 표시 = '__na'·'ov_struct_na' 칼럼(wiring 과 같음).
 (5) 3 fold 가 다 없을 때(0-9 시간 측정)는 반복 수를 자기 곡선 최고 시점으로 매기고 partial=True 로 표시(고르기에 안 씀).
 (6) Focal 문자열 = 'Focal:focal_alpha=0.75;focal_gamma=2'.
 (7) AutoKeras 그래프(AK_PROPOSAL — 단계 0-9 확인 때 담당자가 구조마다 정해 실험 로그에 고정): x_seq → 시퀀스 블록
     (RNNBlock(layer_type 고정)·ConvBlock), x_static(+ x_cat Embedding) → Merge → DenseBlock(정적·임베딩 갈래), 두 갈래
     Merge(concatenate) → DenseBlock('concat → Dense 64' 자리) → ClassificationHead. x_cat 4칼럼 = 칼럼마다 번호표 크기만큼
     밀어 한 Embedding(max_features = 크기 합)에 넣음(칼럼마다 다른 번호 구간 — OOV 행도 칼럼마다 따로, 우리 모델과 달리
     0 고정 아님). optimizer 는 'adam' 고정(계획서 '구조·학습률 탐색만'), 학습률은 AutoKeras 선택지. AutoKeras 의 '메모리
     부족이면 배치 절반' 은 끔(배치 1024 그대로 — B-1, 메모리 오류는 그대로 멈춤). 조기 종료 = Keras EarlyStopping(val_loss,
     patience 3, restore_best_weights), 묶음 10% = models_dl.split_es_groups(같은 씨앗 42). AutoKeras 최종 fit 은 우리
     EarlyStopping·validation_data 를 넘기면 돌지 않음(AutoKeras 3.0.0 원본 확인). overwrite=False 로 같은 폴더면 이어함.
     keras.utils.set_random_seed(42) 를 탐색 전에.
 (8) AutoKeras 시도 순서 = keras-tuner trial_id 순 → t01, t02 …. 끝나지 않은 시도가 있으면 멈춤.
 (9) AutoKeras 후보 고르기: 기본 설정(t00) + 시도 점수 4자리 반올림 큰 순 → 같으면 기본 설정 → 시도 번호 작은 순.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import folds as fd
from . import metrics as mt
from . import models_tree as mtree
from . import registry as rg
from .hassign import H_MAX, u_tie_many

TREE_ALGOS = mtree.ALGOS                      # XGB·LGBM·CAT·RF·ET·LR
N_TPE_TRIALS = 99                             # §6 단계 8: 첫 시도 + TPE 99
TPE_SEED = 42
POS_WEIGHT = (1.0, 300.0)
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2
FOCAL_LOSS = f"Focal:focal_alpha={FOCAL_ALPHA};focal_gamma={FOCAL_GAMMA}"   # 구현 세부 (6)
AK_MAX_TRIALS = 19
AK_SEED = 42
AK_EPOCHS = 20
AK_BATCH = 1024
AK_PATIENCE = 3
TUNE_SPLIT = fd.Split("tune", fd.day("2021-06-30"), fd.day("2021-07-01"), fd.day("2022-05-01"))  # §6 단계 8


class TuningError(ValueError):
    pass


# ================================================================ 탐색 범위(부록 A-7)
@dataclass(frozen=True)
class P:
    name: str
    kind: str                 # float·int·bool
    low: float = 0.0
    high: float = 0.0
    log: bool = False
    rows: bool = False        # 행 수 기준 범위(4-a 면 × rep)
    lr: bool = False          # learning_rate(④ 는 탐색 안 함)


_SPACE: dict[str, tuple[P, ...]] = {
    "CAT": (P("learning_rate", "float", 0.005, 0.3, True, lr=True), P("depth", "int", 2, 10),
            P("l2_leaf_reg", "float", 0.1, 100, True), P("random_strength", "float", 0.01, 20, True),
            P("subsample", "float", 0.5, 1), P("rsm", "float", 0.1, 1), P("scale_pos_weight", "float", *POS_WEIGHT, True)),
    "XGB": (P("learning_rate", "float", 0.01, 0.3, True, lr=True), P("max_depth", "int", 2, 10),
            P("reg_lambda", "float", 0.1, 100, True), P("min_child_weight", "float", 1, 100, True, rows=True),
            P("subsample", "float", 0.5, 1), P("colsample_bytree", "float", 0.1, 1),
            P("scale_pos_weight", "float", *POS_WEIGHT, True)),
    "LGBM": (P("learning_rate", "float", 0.01, 0.3, True, lr=True), P("num_leaves", "int", 4, 1024, True),
             P("lambda_l2", "float", 0.1, 100, True), P("min_child_samples", "int", 5, 500, True, rows=True),
             P("bagging_fraction", "float", 0.5, 1), P("feature_fraction", "float", 0.1, 1),
             P("scale_pos_weight", "float", *POS_WEIGHT, True)),
    "RF": (P("max_depth_unlimited", "bool"), P("max_depth", "int", 4, 30), P("min_samples_leaf", "int", 1, 200, True,
                                                                              rows=True),
           P("max_features", "float", 0.1, 1), P("pos_weight", "float", *POS_WEIGHT, True)),
    "LR": (P("C", "float", 1e-4, 1e2, True), P("pos_weight", "float", *POS_WEIGHT, True)),
}
_SPACE["ET"] = _SPACE["RF"]
FIXED_PARAMS = {"CAT": {"bootstrap_type": "Bernoulli"}, "LGBM": {"bagging_freq": 1}}


def _check_algo(algo: str) -> None:
    if algo not in TREE_ALGOS:
        raise TuningError(f"모르는 알고리즘: {algo!r}({TREE_ALGOS})")


def _check_rep(rep) -> int:
    if isinstance(rep, bool) or not isinstance(rep, (int, np.integer)) or int(rep) < 1:
        raise TuningError(f"rep 는 1 이상 정수(4-a 복제 수 60 또는 m): {rep!r}")
    return int(rep)


def search_space(algo: str, *, rep: int = 1, tune_lr: bool = True) -> dict[str, dict]:
    """기록용 범위표(이름 → kind·low·high·log). rep = 4-a 복제 수(행 수 기준 범위 × rep)."""
    _check_algo(algo)
    rep = _check_rep(rep)
    out = {}
    for p in _SPACE[algo]:
        if p.lr and not tune_lr:
            continue
        d = {"kind": p.kind}
        if p.kind != "bool":
            k = rep if p.rows else 1
            d.update(low=p.low * k, high=p.high * k, log=p.log)
            if p.kind == "int":
                d["low"], d["high"] = int(d["low"]), int(d["high"])
        if algo in ("RF", "ET") and p.name == "max_depth":
            d["when"] = "max_depth_unlimited == False"
        out[p.name] = d
    for k, v in FIXED_PARAMS.get(algo, {}).items():
        out[k] = {"kind": "fixed", "value": v}
    return out


def suggest_params(trial, algo: str, *, rep: int = 1, tune_lr: bool = True) -> dict:
    """Optuna trial 에서 부록 A-7 범위 값을 뽑음(raw — 기록·되풀이 확인용)."""
    _check_algo(algo)
    sp = search_space(algo, rep=rep, tune_lr=tune_lr)
    raw = {}
    for name, d in sp.items():
        if d["kind"] == "fixed":
            continue
        if d.get("when") and raw.get("max_depth_unlimited", True):
            continue
        if d["kind"] == "bool":
            raw[name] = bool(trial.suggest_categorical(name, [True, False]))
        elif d["kind"] == "int":
            raw[name] = int(trial.suggest_int(name, d["low"], d["high"], log=d["log"]))
        else:
            raw[name] = float(trial.suggest_float(name, d["low"], d["high"], log=d["log"]))
    return raw


def model_params(algo: str, raw: Mapping[str, Any]) -> dict:
    """raw(Optuna 값) → models_tree.make_model(params=…) 인자(부록 A-7 이름, 고정값 포함)."""
    _check_algo(algo)
    p = dict(raw)
    out = {}
    if algo in ("RF", "ET"):
        unl = p.pop("max_depth_unlimited")
        out["max_depth"] = None if unl else int(p.pop("max_depth"))
        if unl and "max_depth" in p:
            raise TuningError("max_depth_unlimited 인데 max_depth 가 있음")
        out["min_samples_leaf"] = int(p.pop("min_samples_leaf"))
        out["max_features"] = float(p.pop("max_features"))
        out["class_weight"] = {0: 1.0, 1: float(p.pop("pos_weight"))}
    elif algo == "LR":
        out["C"] = float(p.pop("C"))
        out["class_weight"] = {0: 1.0, 1: float(p.pop("pos_weight"))}
    else:
        out.update(p)
        p = {}
        out.update(FIXED_PARAMS.get(algo, {}))
    if p:
        raise TuningError(f"쓰지 않은 값: {sorted(p)}")
    bad = sorted(set(out) - set(mtree.ALLOWED_PARAMS[algo]))
    if bad:
        raise TuningError(f"models_tree 가 받지 않는 이름: {bad}")
    return out


# ================================================================ 트리 시도 점수
def _pair_meta(samples: ev.SampleTable, idx: np.ndarray, h: int) -> pd.DataFrame:
    from .runner import add_bounds

    m = samples.meta().iloc[idx].copy()
    m["h"] = np.int16(h)
    return add_bounds(m.reset_index(drop=True), None)


class TreeFold:
    """fold 하나의 시도용 자료: train 입력(assembler.fit(train 행) → transform) + 평가 쌍(h 마다, 정답·u_tie).

    train_rows = 이미 만든 train 행(4-a 복제·줄인 행 등, y 칼럼 포함). 없으면 folds.make_train_rows(시료당 h 1개).
    딥러닝 칼럼은 assembler 가 만들지 않는 입력으로(§6 단계 8: 트리 튜닝은 딥러닝 칼럼을 빼고) 넘김."""

    def __init__(self, name: str, samples: ev.SampleTable, assembler, *, split: fd.Split | None = None,
                 train_rows: pd.DataFrame | None = None, cache_val: bool = True,
                 check: Callable[[], Any] | None = None):
        from .runner import add_bounds

        self.name = name
        self.split = split or fd.SPLITS[name]
        self.samples = samples
        rows = train_rows if train_rows is not None else fd.make_train_rows(samples.known_frame(), self.split)
        if (fd.day_ints(rows["d"]) > self.split.train_end).any():
            raise TuningError(f"{name}: train 행 d 가 train 끝을 넘음")
        rows = add_bounds(rows.reset_index(drop=True), None)
        self.y = rows["y"].to_numpy(dtype=np.int64)
        self.assembler = assembler.fit(rows)
        self.X = assembler.transform(rows)
        if not isinstance(self.X, pd.DataFrame):
            raise TuningError("트리 시도 입력은 DataFrame")
        self.cat_cols = [c for c in self.X.columns if self.X[c].dtype == object]
        self.indicator_cols = [c for c in self.X.columns if c.endswith(rg.NA_SUFFIX) or c == rg.STRUCT_COLUMN]
        self.cache_val = cache_val
        self.check = check
        self._val: list | None = None

    def iter_val(self) -> Iterator[tuple[int, Any, np.ndarray, np.ndarray]]:
        if self._val is not None:
            yield from self._val
            return
        keep = [] if self.cache_val else None
        for h, idx in fd.iter_scoring_pairs_by_h(self.samples.d_days, self.split):
            if idx.size == 0:
                continue
            if self.check is not None:
                self.check()
            Xh = self.assembler.transform(_pair_meta(self.samples, idx, h))
            yh = self.samples.cv_y(idx)
            uh = u_tie_many(self.samples.keys[idx], np.full(idx.size, h, dtype=np.int64))
            item = (h, Xh, yh, uh)
            if keep is not None:
                keep.append(item)
            yield item
        if keep is not None:
            self._val = keep


def _own_best(curve: np.ndarray, points: Sequence[int]) -> tuple[int, float]:
    r = np.array([mt.round4(float(v)) for v in curve])
    i = int(np.flatnonzero(r == r.max())[0])
    return int(points[i]), float(curve[i])


def score_tree_config(algo: str, params: Mapping[str, Any], folds: Sequence[TreeFold], *, max_iter: int | None,
                      seed: int = 42, n_jobs: int = 8, check: Callable[[], Any] | None = None) -> dict:
    """설정 하나를 fold 마다 학습해 CV 점수(§3). 부스팅 + max_iter = 56 시점 곡선 → 교차 N(3 fold 일 때) —
    max_iter None = 라이브러리 기본 반복 수(④·RF·ET·로지스틱), 학습한 트리 전부로 예측."""
    _check_algo(algo)
    if not folds:
        raise TuningError("fold 없음")
    names = [f.name for f in folds]
    if len(set(names)) != len(names) or set(names) - set(fd.CV_FOLDS):
        raise TuningError(f"fold 이름: {names}")
    boosting = algo in mtree.BOOSTING and max_iter is not None
    if max_iter is not None and algo not in mtree.BOOSTING:
        raise TuningError(f"{algo} 는 max_iter 없음")
    points = mtree.checkpoints(max_iter) if boosting else None
    t0 = time.time()
    curves, fold_plain, applied, fit_info = {}, {}, {}, {}
    for fo in folds:
        if check is not None:
            check()
        m = mtree.make_model(algo, cat_cols=fo.cat_cols, indicator_cols=fo.indicator_cols if algo == "LR" else (),
                             max_iter=max_iter if boosting else None, params=dict(params))
        m.fit(fo.X, fo.y, seed=seed, n_jobs=n_jobs)
        if boosting:
            byh = np.full((H_MAX, len(points)), np.nan)
            for h, Xh, yh, uh in fo.iter_val():
                Pm = m.predict_proba_multi(Xh, points)
                for k in range(len(points)):
                    byh[h - 1, k] = mt.pr_auc(yh, Pm[:, k], uh)
            curves[fo.name] = np.array([mt.band_weighted(byh[:, k]) for k in range(len(points))])
        else:
            byh = np.full(H_MAX, np.nan)
            for h, Xh, yh, uh in fo.iter_val():
                byh[h - 1] = mt.pr_auc(yh, m.predict_proba(Xh), uh)
            fold_plain[fo.name] = mt.band_weighted(byh)
        dump = m.config_dump()
        applied[fo.name] = {"applied_params": dump["applied_params"], "non_default_params": dump["non_default_params"]}
        fit_info[fo.name] = dump["fit_info"]
        del m
    out = {"algo": algo, "params": dict(params), "max_iter": max_iter if boosting else None, "boosting_curve": boosting,
           "folds": names, "partial": set(names) != set(fd.CV_FOLDS)}
    if boosting:
        out["points"] = list(points)
        out["curves"] = {f: c.tolist() for f, c in curves.items()}
        if not out["partial"]:
            sel = mtree.select_n_iter(curves, points)
            out.update(cv=sel["cv_mean"], fold_scores=sel["fold_score"], cross_n=sel["cross_n"], common_n=sel["common_n"],
                       any_at_cap=sel["any_at_cap"], own_best_n=sel["own_best_n"])
        else:
            ob = {f: _own_best(c, points) for f, c in curves.items()}
            out.update(cv=float(np.mean([v for _, v in ob.values()])), fold_scores={f: v for f, (_, v) in ob.items()},
                       own_best_n={f: n for f, (n, _) in ob.items()})
    else:
        out.update(cv=mt.cv_mean([fold_plain[f] for f in names]), fold_scores=fold_plain)
    out["applied"] = applied
    out["fit_info"] = fit_info
    out["seconds"] = time.time() - t0
    return out


# ================================================================ Optuna(8-T)
def _json_line(path: Path, rec: Mapping) -> None:
    from .config import nan_to_null
    from .models_tree import _jsonable

    line = json.dumps(nan_to_null(_jsonable(rec)), ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def run_optuna(algo: str, score_fn: Callable[[dict], Mapping], *, log_path, n_trials: int = N_TPE_TRIALS,
               seed: int = TPE_SEED, rep: int = 1, tune_lr: bool = True, stop_after: int | None = None,
               label: str = "") -> list[dict]:
    """TPE 시도(첫 시도 = 대상 판 설정은 여기 밖 — 점수만 select_tuned 에). score_fn(model_params) → {'cv': …, …}
    (보통 lambda p: score_tree_config(algo, p, folds, max_iter=…)). 기록 = log_path jsonl(구현 세부 2).
    stop_after = 이번 실행에서 새로 돌릴 시도 수 상한(0-9 '시도 1회' 측정용 — 실제 단계 8 은 None)."""
    import optuna

    _check_algo(algo)
    rep = _check_rep(rep)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    head = {"event": "study", "algo": algo, "n_trials": int(n_trials), "seed": int(seed), "rep": rep,
            "tune_lr": bool(tune_lr), "label": label, "sampler": "TPESampler", "optuna": optuna.__version__,
            "space": search_space(algo, rep=rep, tune_lr=tune_lr)}
    old = _read_log(log)
    if old:
        h0 = {k: v for k, v in old[0].items() if k != "utc"}
        want = json.loads(json.dumps(_jsonable_head(head)))
        if h0 != want:
            raise TuningError(f"기록 파일의 탐색 설정이 다름: {log}")
    else:
        _json_line(log, head)
    done = [r for r in old if r.get("event") == "trial"]
    if any(r["number"] != i + 1 for i, r in enumerate(done)):
        raise TuningError("기록의 시도 번호가 1 부터 이어지지 않음")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    records, ran = list(done), 0
    for i in range(int(n_trials)):
        t = study.ask()
        raw = suggest_params(t, algo, rep=rep, tune_lr=tune_lr)
        if i < len(done):
            if json.loads(json.dumps(raw)) != done[i]["raw"]:
                raise TuningError(f"시도 {i + 1}: 되풀이한 값이 기록과 다름 — 같은 sampler·범위로 다시 시작해야 함")
            study.tell(t, float(done[i]["value"]))
            continue
        if stop_after is not None and ran >= stop_after:
            break
        t0 = time.time()
        res = dict(score_fn(model_params(algo, raw)))
        v = float(res["cv"])
        if not math.isfinite(v):
            raise TuningError(f"시도 {i + 1}: 점수 NaN(B-1 — 실패로 넘기지 않음)")
        rec = {"event": "trial", "number": i + 1, "raw": raw, "params": model_params(algo, raw), "value": v,
               "result": {k: res[k] for k in ("fold_scores", "cross_n", "common_n", "any_at_cap", "own_best_n",
                                              "partial", "applied", "fit_info") if k in res},
               "seconds": time.time() - t0}
        _json_line(log, rec)
        records.append(json.loads(json.dumps(_jsonable_head(rec))))
        study.tell(t, v)
        ran += 1
    return records


def _jsonable_head(obj):
    from .config import nan_to_null
    from .models_tree import _jsonable

    return nan_to_null(_jsonable(obj))


def trial_label(number: int, width: int = 3) -> str:
    """부록 A-15: 8-T = t001 ~ t099(첫 시도 t000), 8-TD = t01 ~ t19(기본 설정 t00)."""
    return f"t{int(number):0{width}d}"


def select_tuned(baseline_cv: float, records: Sequence[Mapping], top: int = 5) -> dict:
    """구현 세부 (3): 튜닝 설정 = 시도 중 CV 1위, 첫 시도(대상 판 설정)보다 오를 때만 채택(같으면 기본 유지)."""
    trials = [r for r in records if r.get("event", "trial") == "trial"]
    if not trials:
        raise TuningError("시도 기록 없음")
    if math.isnan(float(baseline_cv)):
        raise TuningError("첫 시도 점수 NaN")
    ranked = sorted(trials, key=lambda r: (-mt.round4(float(r["value"])), int(r["number"])))
    best = ranked[0]
    adopt = mt.is_improvement(float(best["value"]), float(baseline_cv))
    table = [{"trial": "t000", "cv": float(baseline_cv), "params": "대상 판 설정"}]
    table += [{"trial": trial_label(r["number"]), "cv": float(r["value"]), "params": r["params"]} for r in ranked[:top]]
    return {"adopt": adopt, "judgement": "채택 후보" if adopt else "기본 유지",
            "selected": best if adopt else None, "best_trial": trial_label(best["number"]),
            "best_cv": float(best["value"]), "baseline_cv": float(baseline_cv), "n_trials": len(trials),
            "top": table}


# ================================================================ Focal
def focal_params(params: Mapping[str, Any] | None) -> dict:
    """그 앞까지 채택한 판의 CatBoost 설정에서 손실만 Focal(γ 2·α 0.75)로."""
    p = dict(params or {})
    lf = p.get("loss_function")
    if lf not in (None, "Logloss"):
        raise TuningError(f"이미 다른 손실: {lf}")
    p["loss_function"] = FOCAL_LOSS
    return p


def focal_supported(n_jobs: int = 1) -> dict:
    """CatBoost Focal 이 이 환경에서 학습·확률 예측되는지(합성 자료 200행, 20 반복)."""
    try:
        from catboost import CatBoostClassifier, __version__

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0.8).astype(int)
        m = CatBoostClassifier(iterations=20, loss_function=FOCAL_LOSS, verbose=False, allow_writing_files=False,
                               thread_count=n_jobs, random_seed=42)
        m.fit(X, y)
        p = m.predict_proba(X)[:, 1]
        ok = bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
        return {"supported": ok, "catboost": __version__, "loss_function": m.get_all_params().get("loss_function"),
                "message": "" if ok else "확률이 0~1 밖"}
    except Exception as e:  # noqa: BLE001 — 지원 여부 기록
        return {"supported": False, "message": f"{type(e).__name__}: {e}"}


# ================================================================ AutoKeras(8-TD)
AK_PROPOSAL: dict[str, dict] = {
    # 구현 세부 (7) — 제안(단계 0-9 확인 때 담당자가 정해 실험 로그에 고정, mapping 인자로 바꿈)
    "LSTM": {"supported": True, "seq": "rnn", "layer_type": "lstm",
             "note": "RNNBlock(layer_type='lstm' 고정 — 층 수·양방향은 AutoKeras 탐색)"},
    "GRU": {"supported": True, "seq": "rnn", "layer_type": "gru", "note": "RNNBlock(layer_type='gru' 고정)"},
    "LSTMA": {"supported": True, "seq": "rnn", "layer_type": "lstm",
              "note": "AutoKeras 에 attention 블록 없음 — 시퀀스 갈래를 RNNBlock(lstm) 으로만(담당자 확인)"},
    "GRUA": {"supported": True, "seq": "rnn", "layer_type": "gru",
             "note": "AutoKeras 에 attention 블록 없음 — 시퀀스 갈래를 RNNBlock(gru) 으로만(담당자 확인)"},
    "TCN": {"supported": True, "seq": "conv",
            "note": "ConvBlock(입력 3차원 → Conv1D) — 인과·dilation 합성곱 아님(담당자 확인)"},
    "CMB": {"supported": True, "seq": "rnn", "layer_type": "lstm",
            "note": "시계열 갈래 RNNBlock(lstm)(트랜스포머 1블록 대응 없음), 정형 갈래 DenseBlock(담당자 확인)"},
    "TRF": {"supported": False,
            "reason": "AutoKeras 3.0.0 공개 블록에 실수 시퀀스용 트랜스포머 인코더 없음(TextBlock 은 토큰 입력)"},
    "CMBV": {"supported": False, "reason": "VAE 손실(재구성 MSE + KL)을 AutoKeras 그래프로 못 만듦(부록 A-7 예시)"},
}


class AKUnsupported(TuningError):
    pass


def ak_mapping(arch: str, mapping: Mapping[str, Mapping] | None = None) -> dict:
    m = dict((mapping or AK_PROPOSAL).get(arch) or {})
    if not m:
        raise AKUnsupported(f"{arch}: 대응표에 없음")
    if not m.get("supported"):
        raise AKUnsupported(f"{arch}: 튜닝 안 함 — {m.get('reason', '대응 블록 없음')}")
    if m.get("seq") not in ("rnn", "conv"):
        raise TuningError(f"{arch}: seq 블록 {m.get('seq')!r}")
    if m["seq"] == "rnn" and m.get("layer_type") not in ("lstm", "gru"):
        raise TuningError(f"{arch}: layer_type {m.get('layer_type')!r}")
    return m


def ak_prepare_inputs(inputs: Mapping[str, np.ndarray], cat_sizes: Sequence[int] | None = None) -> dict:
    """구현 세부 (7): x_cat 칼럼 j 번호에 앞 칼럼 번호표 크기 합을 더해 한 Embedding 의 다른 구간으로."""
    out = dict(inputs)
    if "x_cat" in inputs:
        if cat_sizes is None:
            raise TuningError("x_cat 이 있으면 cat_sizes 필요")
        cs = np.asarray([int(c) for c in cat_sizes], dtype=np.int64)
        xc = np.asarray(inputs["x_cat"])
        if xc.dtype != np.int64 or xc.ndim != 2 or xc.shape[1] != cs.size:
            raise TuningError("x_cat 은 int64 (N, 4)")
        if (xc < 0).any() or (xc >= cs[None, :]).any():
            raise TuningError("x_cat 번호가 번호표 밖")
        off = np.concatenate([[0], np.cumsum(cs)[:-1]]).astype(np.int64)
        out["x_cat"] = np.ascontiguousarray(xc + off[None, :])
    return out


def build_ak_graph(arch: str, *, has_cat: bool, cat_sizes: Sequence[int] | None = None,
                   mapping: Mapping[str, Mapping] | None = None, ak=None) -> tuple[list, Any, tuple]:
    """(입력 노드 list, 출력 노드, 입력 키 순서). 구현 세부 (7)."""
    m = ak_mapping(arch, mapping)
    if ak is None:
        import autokeras as ak
    in_seq = ak.Input(name="x_seq")
    seq = (ak.RNNBlock(layer_type=m["layer_type"]) if m["seq"] == "rnn" else ak.ConvBlock())(in_seq)
    in_st = ak.Input(name="x_static")
    nodes, keys = [in_seq, in_st], ["x_seq", "x_static"]
    if has_cat:
        if not cat_sizes:
            raise TuningError("x_cat 이 있으면 cat_sizes 필요")
        in_cat = ak.Input(name="x_cat")
        emb = ak.Embedding(max_features=int(sum(int(c) for c in cat_sizes)))(in_cat)
        tab_in = ak.Merge(merge_type="concatenate")([in_st, emb])
        nodes.append(in_cat)
        keys.append("x_cat")
    else:
        tab_in = in_st
    tab = ak.DenseBlock()(tab_in)
    merged = ak.Merge(merge_type="concatenate")([seq, tab])
    out = ak.ClassificationHead()(ak.DenseBlock()(merged))
    return nodes, out, tuple(keys)


def disable_adaptive_batch_size(ak_utils=None) -> bool:
    """구현 세부 (7): AutoKeras 의 메모리 부족 시 배치 절반을 끔(배치 1024 그대로, 메모리 오류는 그대로 올라감 — B-1)."""
    if ak_utils is None:
        from autokeras.utils import utils as ak_utils
    if getattr(ak_utils.run_with_adaptive_batch_size, "_lst_fixed_batch", False):
        return False

    def run_fixed(batch_size, func, **fit_kwargs):
        validation_data = fit_kwargs.pop("validation_data", None)
        return func(validation_data=validation_data, batch_size=batch_size, **fit_kwargs)

    run_fixed._lst_fixed_batch = True
    ak_utils.run_with_adaptive_batch_size = run_fixed
    return True


@dataclass
class AKSearch:
    arch: str
    keys: tuple
    cat_sizes: list | None
    auto_model: Any
    tuner: Any
    mapping: dict
    info: dict = field(default_factory=dict)


def tune_train_rows(samples: ev.SampleTable) -> pd.DataFrame:
    """튜닝 학습 행 = d ≤ 2021-06-30 시료당 h 1개(§6 단계 8 — 표준화 통계도 이 행으로)."""
    from .runner import add_bounds

    return add_bounds(fd.make_train_rows(samples.known_frame(), TUNE_SPLIT).reset_index(drop=True), None)


def _es_split(groups, seed: int) -> np.ndarray:
    from .models_dl import split_es_groups

    return split_es_groups(groups, seed)


def autokeras_search(arch: str, inputs: Mapping[str, np.ndarray], y, groups, *, directory, project_name: str,
                     cat_sizes: Sequence[int] | None = None, train_d=None, max_trials: int = AK_MAX_TRIALS,
                     seed: int = AK_SEED, mapping: Mapping[str, Mapping] | None = None, overwrite: bool = False,
                     epochs: int = AK_EPOCHS, batch_size: int = AK_BATCH, patience: int = AK_PATIENCE,
                     verbose: int = 0, ak=None, keras=None, keras_tuner=None) -> AKSearch:
    """AutoKeras 탐색(부록 A-7). inputs = 튜닝 학습 행(d ≤ 2021-06-30)의 x_seq·x_static(·x_cat) — DLInputs 를 그 행으로
    fit 한 것, groups = (sgg, d)(조기 종료 묶음 10%), train_d = 학습 행 d(1년 검증 기간과 안 겹침 assert).
    대응 블록이 없는 구조는 AKUnsupported(기록하고 튜닝 안 함)."""
    m = ak_mapping(arch, mapping)
    if train_d is not None and (fd.day_ints(train_d) > TUNE_SPLIT.train_end).any():
        raise TuningError("AutoKeras 학습 행에 d > 2021-06-30 — 1년 검증 기간과 겹침")
    info = {}
    if ak is None:
        from . import ak_compat
        from .resources import limit_gpu_memory

        info = ak_compat.prepare_autokeras()   # KERAS_BACKEND=torch·Dense units 패치
        import autokeras as ak

        info["fixed_batch_patch"] = disable_adaptive_batch_size()
        info["gpu_limit"] = limit_gpu_memory()
    if keras is None:
        import keras
    if keras_tuner is None:
        import keras_tuner
    keras.utils.set_random_seed(int(seed))
    has_cat = "x_cat" in inputs
    x = ak_prepare_inputs(inputs, cat_sizes)
    nodes, out, keys = build_ak_graph(arch, has_cat=has_cat, cat_sizes=cat_sizes, mapping=mapping, ak=ak)
    unknown = set(x) - set(keys)
    if unknown:
        raise TuningError(f"그래프에 없는 입력: {sorted(unknown)}")
    yv = np.asarray(y)
    if not np.isin(yv, (0, 1)).all():
        raise TuningError("y 는 0/1")
    yv = yv.astype(np.int64)
    es = _es_split(groups, seed)
    tr, va = np.flatnonzero(~es), np.flatnonzero(es)
    x_tr = [np.ascontiguousarray(x[k][tr]) for k in keys]
    x_va = [np.ascontiguousarray(x[k][va]) for k in keys]
    hp = keras_tuner.HyperParameters()
    hp.Fixed("optimizer", "adam")
    am = ak.AutoModel(inputs=nodes, outputs=out, project_name=project_name, max_trials=int(max_trials),
                      directory=os.fspath(directory), objective="val_loss", tuner="greedy", overwrite=overwrite,
                      seed=int(seed), hyperparameters=hp, tune_new_entries=True)
    t0 = time.time()
    am.fit(x=x_tr, y=yv[tr], validation_data=(x_va, yv[va]), epochs=int(epochs), batch_size=int(batch_size),
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience),
                                                    restore_best_weights=True)], verbose=verbose)
    _ensure_io_shapes(am, x_tr, yv[tr], x_va, yv[va])
    info.update({"seconds": time.time() - t0, "n_train": int(tr.size), "n_es": int(va.size), "max_trials": int(max_trials),
                 "seed": int(seed), "epochs": int(epochs), "batch_size": int(batch_size), "patience": int(patience),
                 "objective": "val_loss", "tuner": "greedy", "optimizer": "adam(고정)", "mapping": m})
    return AKSearch(arch, keys, None if cat_sizes is None else [int(c) for c in cat_sizes], am, am.tuner, m, info)


def _ensure_io_shapes(am, x_tr, y_tr, x_va, y_va) -> None:
    """이어한 탐색(이미 끝난 tuner)에서도 hypermodel.build 가 되게 입출력 모양을 다시 넣음(AutoKeras 내부 함수)."""
    try:
        dataset, _ = am._check_and_adapt(x=x_tr, y=y_tr, validation_data=(x_va, y_va))
        am.tuner._prepare_model_build(am.tuner.oracle.get_space(), x=dataset[0], y=dataset[1])
    except Exception as e:  # noqa: BLE001 — 같은 프로세스에서 막 끝난 탐색이면 이미 들어 있음
        am._lst_io_shape_note = f"{type(e).__name__}: {e}"


def completed_trials(search: AKSearch) -> list:
    trials = search.tuner.oracle.trials
    out = [trials[k] for k in sorted(trials)]
    bad = [t.trial_id for t in out if str(getattr(t.status, "value", t.status)) != "COMPLETED"]
    if bad:
        raise TuningError(f"끝나지 않은 AutoKeras 시도: {bad}(B-1 — 그 시도부터 다시)")
    return out


class TuneValidation:
    """1년 검증 쌍(T ∈ 2021-07-01 ~ 2022-05-01, 모든 fold validation 보다 앞): h 마다 (h, 입력, y, u_tie).
    assembler = 튜닝 학습 행으로 fit 한 조립 객체(DLInputs 등). cache=True 면 메모리에 둠(디스크 없음)."""

    def __init__(self, samples: ev.SampleTable, assembler, *, split: fd.Split = TUNE_SPLIT, cache: bool = True,
                 check: Callable[[], Any] | None = None):
        self.samples, self.assembler, self.split, self.cache, self.check = samples, assembler, split, cache, check
        self._items: list | None = None

    def __iter__(self):
        if self._items is not None:
            yield from self._items
            return
        keep = [] if self.cache else None
        for h, idx in fd.iter_scoring_pairs_by_h(self.samples.d_days, self.split):
            if idx.size == 0:
                continue
            if self.check is not None:
                self.check()
            if (self.samples.d_days[idx] <= self.split.train_end).any():
                raise TuningError("1년 검증 쌍이 튜닝 학습 기간과 겹침")
            item = (h, self.assembler.transform(_pair_meta(self.samples, idx, h)), self.samples.cv_y(idx),
                    u_tie_many(self.samples.keys[idx], np.full(idx.size, h, dtype=np.int64)))
            if keep is not None:
                keep.append(item)
            yield item
        if keep is not None:
            self._items = keep


def score_predict_fn(predict_fn: Callable[[Any], np.ndarray], val: Iterable) -> dict:
    """예측 함수 하나의 1년 검증 쌍 구간 가중 PR-AUC(기본 설정 후보 = 이미 학습한 d ≤ 2021-06-30 모델 등)."""
    byh = np.full(H_MAX, np.nan)
    n = 0
    for h, X, y, u in val:
        p = np.asarray(predict_fn(X), dtype=np.float64).reshape(-1)
        byh[h - 1] = mt.pr_auc(y, p, u)
        n += len(y)
    return {"score": mt.band_weighted(byh), "by_h": byh.tolist(), "n_pairs": n}


def score_ak_trials(search: AKSearch, val: Iterable) -> list[dict]:
    """시도마다 keras-tuner 체크포인트(최저 val_loss 에폭)에서 불러 1년 검증 쌍으로 점수(구현 세부 8)."""
    from .ak_compat import keras_predict_proba

    out = []
    for i, t in enumerate(completed_trials(search), start=1):
        model = search.tuner.load_model(t)
        r = score_predict_fn(lambda X: keras_predict_proba(model, ak_prepare_inputs(X, search.cat_sizes), search.keys),
                             val)
        hpv = dict(t.hyperparameters.values)
        out.append({"trial": trial_label(i, 2), "trial_id": t.trial_id, "score": r["score"], "by_h": r["by_h"],
                    "n_pairs": r["n_pairs"], "hyperparameters": hpv,
                    "val_loss": getattr(t, "score", None)})
        del model
    return out


def select_ak(trial_scores: Sequence[Mapping], default_score: float) -> dict:
    """구현 세부 (9): 후보 20(기본 설정 t00 + 시도) 중 1위, 기본 설정이 1위(같음 포함)면 기본 유지."""
    if math.isnan(float(default_score)):
        raise TuningError("기본 설정 점수 NaN")
    cands = [{"trial": "t00", "score": float(default_score), "order": 0}]
    cands += [{"trial": s["trial"], "trial_id": s.get("trial_id"), "score": float(s["score"]), "order": i + 1}
              for i, s in enumerate(trial_scores)]
    if any(math.isnan(c["score"]) for c in cands):
        raise TuningError("시도 점수 NaN")
    ranked = sorted(cands, key=lambda c: (-mt.round4(c["score"]), c["order"]))
    best = ranked[0]
    keep = best["trial"] == "t00"
    return {"adopt": not keep, "judgement": "기본 유지" if keep else "채택 후보", "selected": None if keep else best,
            "ranking": [(c["trial"], c["score"]) for c in ranked], "default_score": float(default_score)}


def rebuild_selected(search: AKSearch, trial_id: str):
    """고른 시도의 설정으로 새 모델 = hypermodel.build(hp)(build 가 컴파일한 옵티마이저·학습률 그대로 — 부록 A-7)."""
    t = search.tuner.oracle.trials[trial_id]
    return search.tuner.hypermodel.build(t.hyperparameters)


def fit_keras_model(model, inputs: Mapping[str, np.ndarray], y, groups, *, keys: Sequence[str],
                    cat_sizes: Sequence[int] | None = None, seed: int = AK_SEED, epochs: int = AK_EPOCHS,
                    batch_size: int = AK_BATCH, patience: int = AK_PATIENCE, verbose: int = 0, keras=None):
    """고른 설정 다시 학습(부록 A-8 배치·에폭·조기 종료 — (sgg, d) 묶음 10%, 최저 val_loss 에폭으로 되돌림)."""
    if keras is None:
        import keras
    keras.utils.set_random_seed(int(seed))
    x = ak_prepare_inputs(inputs, cat_sizes)
    yv = np.asarray(y).astype(np.int64)
    es = _es_split(groups, seed)
    tr, va = np.flatnonzero(~es), np.flatnonzero(es)
    hist = model.fit([x[k][tr] for k in keys], yv[tr], validation_data=([x[k][va] for k in keys], yv[va]),
                     epochs=int(epochs), batch_size=int(batch_size), verbose=verbose,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience),
                                                              restore_best_weights=True)])
    return hist
