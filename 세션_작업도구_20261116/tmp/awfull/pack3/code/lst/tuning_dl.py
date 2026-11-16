"""단계 8-D 딥러닝 튜닝 — 단계 8(트리 8-T) 판정 뒤 현재 최고 판의 딥러닝 부분 설정 탐색(담당자 결정 11-11, 계획서 단계 8-D 절
초안). 틀은 8-T 와 같음: 첫 시도 t000(대상 판 설정 그대로 — 대상 판 CV, 학습 없음) + TPE(씨앗 42) 19회, 목표 = CV 평균 구간
가중 PR-AUC(채점 = evaluate.score_cv 와 같은 fold 별 계산), 튜닝 설정 = 시도 중 CV 1위(4자리 반올림으로 t000 보다 오를
때만 — tuning.select_tuned 그대로). 자료·입력·h 규칙·에폭(최대 20·인내 3)·조기 종료 규칙·평가 쌍·채점·threshold 는 대상
판 그대로, 바꾸는 것은 아래 범위의 설정만(씨앗 하나 — 대상 판 학습 씨앗 그대로, 음성 4배 고정).

공개 API(scripts/s8_d.py 가 씀):
  범위:   search_space(arch) · suggest(trial, arch) · dl_params(arch, raw) → 판 params {"dl": {...}}(lst.wiring.dl_model_kwargs)
  점수:   fold_score(preds, samples, split) · DLTrialScorer(…)(items, batch) → {번호: {"cv", "fold_scores"}}
  Optuna: run_batched(arch, score_batch, *, log_path, n_trials=19, seed=42, batches=(10, 3, 3, 3), label) → 시도 기록 list

구현 세부(계획서에 없음 — 담당자 확인):
 (d1) 탐색 도구 = Optuna(TPESampler(seed=42) 기본값, maximize) — 모델이 PyTorch(lst.models_dl)라 keras_tuner 로는 모델을
      다시 써야 하고, 8-T 와 같은 jsonl 이어하기·고르기(tuning.select_tuned)를 그대로 씀.
 (d2) 묶어 묻기(batches): 시도 점수 계산의 대부분(5-LSTM 실측 fold 마다 평가 쌍 입력 만들기 ≈ 550~630 초, 학습 ≈ 2~3 초,
      예측 ≈ 50 초)은 설정과 무관한 평가 쌍 입력 만들기라, 한 묶음의 설정을 한꺼번에 묻고(ask) fold 마다 평가 쌍 입력을 한
      번 만들어 묶음 모델 모두가 예측(runner.predict_pairs_multi — 메모리에서만, 디스크 캐시 없음 B-1)한 뒤 함께 알려 줌
      (tell). 기본 (10, 3, 3, 3): TPE 기본 n_startup_trials 10 이라 앞 10 회는 무작위 탐색이고 한 번에 물어도 차례로 물을
      때와 같은 값, 뒤 9 회만 3 개씩 TPE(같은 묶음 안 시도끼리는 서로의 점수를 모름 — constant_liar 안 씀).
      batches = (1,) * 19 면 8-T 와 같은 차례 탐색(시간 ≈ 묶음 수 × 평가 쌍 입력 3 fold).
 (d3) 기록 = <log>.jsonl: 'study'(탐색 설정 — 다르면 멈춤) → 묶음마다 'ask'(번호·raw — 이어할 때 같은 sampler 로 되풀이해
      대조, 다르면 멈춤) → 'trial'(점수). fold 단위 기록 = DLTrialScorer 의 fold 기록 jsonl('fold' — 번호·fold·점수·학습
      정보, 'pass' — 묶음·fold·평가 쌍 입력 초): 멈추면 같은 명령으로 끝난 (시도, fold) 는 건너뛰고 남은 것부터(B-1).
 (d4) 범위(구조마다): lr 1e-4~3e-3(로그 — 기본 1e-3 이 안쪽), 배치 {256, 512, 1024, 2048}(기본 1024 포함, 2 의 거듭제곱),
      은닉(시퀀스 갈래 폭) {32, 64, 128, 256}(기본 64 포함, 헤드 4 로 나눠짐), 층 수 RNN·결합 1~2·TRF 1~3(기본 2 가
      안쪽)·TCN 탐색 안 함(블록 5개 = 받는 범위 125일 고정), dropout 0~0.5(균등), 양성 가중 1~4(로그 — 음성 4배 고정이라
      4 = 손실상 양성·음성 균형, 1 = 지금), 손실 ∈ {bce, focal} → focal 이면 γ ∈ {1, 2}·α ∈ {0.25, 0.75}(조건부).
      쓰는 자리는 lst.models_dl 구현 세부 16. 정수·범주는 Optuna categorical·int, 실수는 float.
 (d5) 시도 학습 = 대상 판과 같은 train 행(RowPlan — 5·6 판이면 뽑은 키)·같은 입력 조립(fold train 으로 fit)·같은 씨앗
      (5·6 판 = 0 — SeedOverride)·같은 조기 종료 묶음. 첫 묶음에 t000(대상 판 설정)도 다시 학습해 점수를 대상 판 CV 와
      대조만 함(번호 0 — 고르기엔 대상 판 CV 를 씀, 차이는 기록·알림). 평가 쌍 입력 사본 없음(딥러닝은 입력을 고치지 않음).
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import metrics as mt
from . import models_dl as md
from . import tuning as tu

N_TPE = 19
TPE_SEED = tu.TPE_SEED
DEFAULT_BATCHES = (10, 3, 3, 3)
LR_RANGE = (1e-4, 3e-3)
BATCH_CHOICES = (256, 512, 1024, 2048)
HIDDEN_CHOICES = (32, 64, 128, 256)
DROPOUT_RANGE = (0.0, 0.5)
POS_WEIGHT_RANGE = (1.0, 4.0)
LOSS_CHOICES = ("bce", "focal")
FOCAL_GAMMAS = (1, 2)
FOCAL_ALPHAS = (0.25, 0.75)
LAYER_RANGE = {"LSTM": (1, 2), "GRU": (1, 2), "LSTMA": (1, 2), "GRUA": (1, 2), "CMB": (1, 2), "CMBV": (1, 2),
               "TRF": (1, 3), "TCN": None}
T000_ALERT_TOL = 1e-4          # t000 다시 학습 점수와 대상 판 CV 차이가 이보다 크면 알림(규칙은 그대로)


class DLTuningError(tu.TuningError):
    pass


# ================================================================ 범위(구현 세부 d4)
def _check_arch(arch: str) -> None:
    if arch not in md.CLASSIFIER_ARCHS:
        raise DLTuningError(f"모르는 딥러닝 구조: {arch!r}({md.CLASSIFIER_ARCHS})")


def search_space(arch: str) -> dict[str, dict]:
    """기록용 범위표(묻는 차례 그대로)."""
    _check_arch(arch)
    sp = {"lr": {"kind": "float", "low": LR_RANGE[0], "high": LR_RANGE[1], "log": True},
          "batch_size": {"kind": "categorical", "choices": list(BATCH_CHOICES)},
          "hidden": {"kind": "categorical", "choices": list(HIDDEN_CHOICES)}}
    lay = LAYER_RANGE[arch]
    if lay is not None:
        sp["layers"] = {"kind": "int", "low": lay[0], "high": lay[1], "log": False}
    sp["dropout"] = {"kind": "float", "low": DROPOUT_RANGE[0], "high": DROPOUT_RANGE[1], "log": False}
    sp["pos_weight"] = {"kind": "float", "low": POS_WEIGHT_RANGE[0], "high": POS_WEIGHT_RANGE[1], "log": True}
    sp["loss"] = {"kind": "categorical", "choices": list(LOSS_CHOICES)}
    sp["focal_gamma"] = {"kind": "categorical", "choices": list(FOCAL_GAMMAS), "when": "loss == focal"}
    sp["focal_alpha"] = {"kind": "categorical", "choices": list(FOCAL_ALPHAS), "when": "loss == focal"}
    return sp


def suggest(trial, arch: str) -> dict:
    """Optuna trial → raw(기록·되풀이 대조용)."""
    raw: dict[str, Any] = {}
    for name, d in search_space(arch).items():
        if d.get("when") and raw.get("loss") != "focal":
            continue
        if d["kind"] == "categorical":
            raw[name] = trial.suggest_categorical(name, list(d["choices"]))
        elif d["kind"] == "int":
            raw[name] = int(trial.suggest_int(name, d["low"], d["high"], log=d["log"]))
        else:
            raw[name] = float(trial.suggest_float(name, d["low"], d["high"], log=d["log"]))
    return raw


def dl_params(arch: str, raw: Mapping[str, Any]) -> dict:
    """raw → 판 params {"dl": {...}}(lst.wiring.dl_model_kwargs 가 읽음). models_dl.normalize_hparams 로 점검."""
    _check_arch(arch)
    r = dict(raw)
    if "batch_size" not in r or int(r["batch_size"]) not in BATCH_CHOICES:
        raise DLTuningError(f"batch_size: {r.get('batch_size')}")
    out = {"batch_size": int(r.pop("batch_size"))}
    hp = {k: r.pop(k) for k in list(r) if k in md.HP_KEYS}
    if r:
        raise DLTuningError(f"쓰지 않은 값: {sorted(r)}")
    md.normalize_hparams(arch, hp)
    out.update(hp)
    return {"dl": out}


# ================================================================ 점수(구현 세부 d5)
def fold_score(preds: pd.DataFrame, samples: ev.SampleTable, split: str, *, require_complete: bool = True) -> float:
    """fold 하나의 구간 가중 PR-AUC — evaluate.score_cv 의 fold 계산 그대로(prepare_pairs → cv_y → u_tie → score_report)."""
    P = ev.prepare_pairs(preds, samples, split, require_complete=require_complete)
    y = samples.cv_y(P.pos)
    return float(mt.score_report(y, P.pred, P.h, P.u(False))["pr_auc_band_weighted"])


def _read(path: Path) -> list[dict]:
    return tu._read_log(Path(path))


def _num(v):
    return math.nan if v is None else float(v)


class DLTrialScorer:
    """설정 묶음 → fold 마다 (train 입력 1번 — 프로세스 안 메모리) → 설정마다 학습 → 평가 쌍 입력 1번으로 묶음 모두 예측 →
    fold 점수 기록(fold 기록 jsonl — 끝난 (번호, fold) 는 건너뜀).

    spec = 대상 판 구성의 판 설정(이름·extra 는 대상 판 것 — 시도마다 params 만 바꿈), train_rows_fn(split) = E·S·g·R 붙은
    train 행 표(대상 판 RowPlan), make_inputs·make_model·fit_kwargs = runner 주입 함수(Stage4Wiring)."""

    def __init__(self, *, spec, samples: ev.SampleTable, make_inputs: Callable, make_model: Callable,
                 fit_kwargs: Callable | None, train_rows_fn: Callable[[str], pd.DataFrame], fold_log, n_jobs: int,
                 seed: int = 42, check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                 folds: Sequence[str] = ev.CV_FOLDS):
        self.spec, self.samples = spec, samples
        self.make_inputs, self.make_model, self.fit_kwargs = make_inputs, make_model, fit_kwargs
        self.train_rows_fn, self.fold_log = train_rows_fn, Path(fold_log)
        self.n_jobs, self.seed, self.check = int(n_jobs), int(seed), check
        self.max_pairs_per_h, self.folds = max_pairs_per_h, tuple(folds)
        self._train: dict[str, tuple] = {}
        self.n_fits = 0
        self.n_passes = 0

    def _fold_data(self, f: str):
        if f not in self._train:
            rows = self.train_rows_fn(f)
            y = rows["y"].to_numpy(dtype=np.int64)
            asm = self.make_inputs(self.spec, f).fit(rows)
            self._train[f] = (rows, asm, asm.transform(rows), y)
        return self._train[f]

    def done(self) -> dict[tuple[int, str], dict]:
        return {(int(r["number"]), r["fold"]): r for r in _read(self.fold_log) if r.get("event") == "fold"}

    def __call__(self, items: Sequence[tuple[int, Mapping]], batch: int) -> dict[int, dict]:
        from . import runner as rn

        nums = [int(n) for n, _ in items]
        if len(set(nums)) != len(nums):
            raise DLTuningError(f"묶음 번호 중복: {nums}")
        done = self.done()
        for f in self.folds:
            need = [(int(n), dict(p)) for n, p in items if (int(n), f) not in done]
            if not need:
                continue
            if self.check is not None:
                self.check()
            rows, asm, X, y = self._fold_data(f)
            models, fit = {}, {}
            for n, p in need:
                sp = dataclasses.replace(self.spec, params=p)
                m = self.make_model(sp, asm)
                kw = dict(self.fit_kwargs(sp, f, rows, X)) if self.fit_kwargs else {}
                t0 = time.time()
                m.fit(X, y, seed=self.seed, n_jobs=self.n_jobs, **kw)
                fi = dict(getattr(m, "fit_info", {}) or {})
                fit[n] = {"seconds": round(time.time() - t0, 2), "best_epoch": fi.get("best_epoch"),
                          "epochs_run": fi.get("epochs_run"), "best_es_loss": fi.get("best_es_loss"),
                          "oom_events": len(fi.get("oom_events") or [])}
                models[str(n)] = m
                self.n_fits += 1
            preds, pred_s, t_in = rn.predict_pairs_multi(
                models, asm, self.samples, f, self.spec.W, check=self.check, max_pairs_per_h=self.max_pairs_per_h,
                seed=self.seed, dedupe_rows=False, copy_inputs=False)
            self.n_passes += 1
            for n, _ in need:
                sc = fold_score(preds[str(n)], self.samples, f, require_complete=self.max_pairs_per_h is None)
                rec = {"event": "fold", "number": n, "fold": f, "batch": int(batch), "score": sc,
                       "n_pairs": int(len(preds[str(n)])), "fit": fit[n], "predict_seconds": pred_s[str(n)]}
                tu._json_line(self.fold_log, rec)
                done[(n, f)] = rec
            tu._json_line(self.fold_log, {"event": "pass", "batch": int(batch), "fold": f, "numbers": [n for n, _ in need],
                                          "train_rows": int(len(rows)), "eval_input_seconds": t_in})
            del models, preds
        out = {}
        for n in nums:
            fs = {f: _num(done[(n, f)]["score"]) for f in self.folds}
            out[n] = {"cv": mt.cv_mean(fs), "fold_scores": fs}
        return out


# ================================================================ Optuna 묶어 묻기(구현 세부 d2·d3)
def check_batches(batches: Sequence[int], n_trials: int) -> tuple[int, ...]:
    b = tuple(int(x) for x in batches)
    if not b or any(x < 1 for x in b) or sum(b) != int(n_trials):
        raise DLTuningError(f"batches 합 {sum(b)} ≠ 시도 수 {n_trials}(모두 1 이상)")
    return b


def run_batched(arch: str, score_batch: Callable[[list, int], Mapping[int, Mapping]], *, log_path,
                n_trials: int = N_TPE, seed: int = TPE_SEED, batches: Sequence[int] = DEFAULT_BATCHES,
                label: str = "") -> list[dict]:
    """TPE 시도(t000 은 밖 — select_tuned 에 점수만). score_batch([(번호, params)…], 묶음 번호) → {번호: {'cv', …}}."""
    import optuna

    _check_arch(arch)
    batches = check_batches(batches, n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    head = {"event": "study", "kind": "dl", "arch": arch, "n_trials": int(n_trials), "seed": int(seed),
            "batches": list(batches), "label": label, "sampler": "TPESampler", "optuna": optuna.__version__,
            "space": search_space(arch)}
    old = _read(log)
    if old:
        h0 = {k: v for k, v in old[0].items() if k != "utc"}
        if h0 != json.loads(json.dumps(tu._jsonable_head(head))):
            raise DLTuningError(f"기록 파일의 탐색 설정이 다름: {log}")
    else:
        tu._json_line(log, head)
    asks = {int(r["number"]): r for r in old if r.get("event") == "ask"}
    trials = {int(r["number"]): r for r in old if r.get("event") == "trial"}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    num, out = 0, []
    for bi, k in enumerate(batches, start=1):
        items = []
        for _ in range(k):
            num += 1
            t = study.ask()
            raw = suggest(t, arch)
            rawj = json.loads(json.dumps(raw))
            if num in asks:
                if asks[num]["raw"] != rawj:
                    raise DLTuningError(f"시도 {num}: 되풀이한 값이 기록과 다름 — 같은 sampler·범위로 다시 시작해야 함")
            else:
                tu._json_line(log, {"event": "ask", "number": num, "batch": bi, "raw": raw,
                                    "params": dl_params(arch, raw)})
            items.append((num, t, raw))
        todo = [(n, dl_params(arch, raw)) for n, _, raw in items if n not in trials]
        if todo:
            res = score_batch(todo, bi)
            for n, _, raw in items:
                if n in trials:
                    continue
                r = dict(res[n])
                v = float(r["cv"])
                if not math.isfinite(v):
                    raise DLTuningError(f"시도 {n}: 점수 NaN(B-1 — 실패로 넘기지 않음)")
                rec = {"event": "trial", "number": n, "batch": bi, "raw": raw, "params": dl_params(arch, raw),
                       "value": v, "result": {"fold_scores": r.get("fold_scores")}}
                tu._json_line(log, rec)
                trials[n] = json.loads(json.dumps(tu._jsonable_head(rec)))
        for n, t, _ in items:
            study.tell(t, float(trials[n]["value"]))
            out.append(trials[n])
    return out


def trial_table(records: Sequence[Mapping]) -> pd.DataFrame:
    """시도 표(보고용): 번호·묶음·CV·fold 점수·설정."""
    rows = []
    for r in records:
        fs = (r.get("result") or {}).get("fold_scores") or {}
        rows.append({"trial": tu.trial_label(r["number"]), "batch": r.get("batch"), "cv": r["value"],
                     **{f"{f}": _num(fs.get(f)) for f in ev.CV_FOLDS},
                     "params": json.dumps(r["params"], ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)
