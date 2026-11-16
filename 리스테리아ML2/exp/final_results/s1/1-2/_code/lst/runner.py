"""실행 틀 — 판 하나 실행·결과 폴더·이어하기·실험 로그·판 이름. 계획서 §1-3(판 이름·test 막기), §3(train 행·
채점 쌍·refit), §8(저장·실험 로그), 부록 A-12('같은 validation 으로 고른 횟수'), A-14(이름 조각), A-15, B-1, B-2.

판 하나 = fold 1~3·refit 마다
  train 행(`folds.make_train_rows`, 시료당 h 1개) → 입력 조립(주입: `make_inputs(spec, split)` → fit(train 행 표)·
  transform(행 표)) → 모델(주입: `make_model(spec)` → fit(X, y, *, seed, n_jobs[, …]) / predict_proba(X) / save)
  → 평가 쌍을 h 마다 메모리에서 만들어 바로 예측(`folds.iter_scoring_pairs_by_h`, 디스크 캐시 없음 — B-1)
  → 예측 parquet(lims_data_key·h·fold·pred) 저장
→ CV 채점(`evaluate.score_cv` — test 정답은 읽지 않음) → 실험 로그('같은 validation 으로 고른 횟수') → MLflow 기록.
test 채점은 `report_stage`(보고 허가 필요)만.

결과 폴더(--out = 단계 폴더, 예: results/s1/1-1):
  _code/                      실행 코드 사본(lst·scripts·configs) + code_md5.txt(B-1)
  <판 이름>/spec.json          판 설정(설정 md5 의 원본)
           started.json        시작 시각·code_md5·설정 md5
           model_<split>/      모델 save(path)
           inputs_<split>/     입력 조립 객체 save(path)(있을 때 — 딥러닝 전처리기 등)
           columns_<split>.json 입력 칼럼 목록(조립 객체에 columns 가 있을 때)
           pred_<split>.parquet  fold1·fold2·fold3 = validation 쌍, refit = test 쌍(정답 없음)
           cv_scores.json      CV 채점(evaluate.score_cv)
           run_meta.json       code_md5·입력 md5·스레드·설정 덤프·시간(B-2)
           done.json           끝 표시(설정 md5·code_md5·CV·고른 횟수·MLflow run id) — 있으면 건너뜀
           failed.json         멈춘 이유(예외) — 다시 돌릴 때 실험 로그에
           test_scores.json    보고 때만(evaluate.score_test)
  _incomplete/<판>__<시각>/   멈춘 판의 이전 폴더(지우지 않고 옮김)

구현 세부(계획서에 없음 — 담당자 확인):
  - 이어하기: done.json 의 설정 md5·code_md5 가 같으면 건너뜀, 설정 같고 code_md5 다르면 멈춤(코드가 바뀐 채 이어
    돌리지 않음 — 지휘자 결정), done.json 없는 폴더(멈춘 판)는 _incomplete 로 옮기고 그 판부터 다시(실험 로그에 이유).
  - 하위 프로세스 격리(B-1): 판마다 같은 스크립트를 `--variant <판>` 로 따로 띄움. 0 아닌 종료(예외·강제 종료)면
    거기서 전체를 멈춤(건너뛰지 않음) — 원인을 없앤 뒤 다시 실행하면 끝난 판은 건너뛰고 그 판부터.
  - 실험 로그 = --inputs(결과 뿌리) 아래 jsonl 하나(설정의 experiment_log), 여러 프로세스가 같이 쓰므로 파일 잠금.
  - '같은 validation 으로 고른 횟수' = 실험 로그의 CV 채점 기록 수 누적(그 판 포함), 기록 순서 = CV 채점이 끝난 순서.
  - 행 표의 E·S·g·R 은 `windows.window_bounds` 로(W 없는 판은 S 없음).
  - 딥러닝 조기 종료 묶음((sgg, d) 10%)처럼 fit 에 더 넘길 것은 주입 `fit_kwargs(spec, split, train_meta, X)` 로.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from . import config as cf
from . import evaluate as ev
from . import folds as fd
from . import windows as wn
from .resources import check_resources as _check_resources

SPLIT_ORDER = ("fold1", "fold2", "fold3", "refit")

# ================================================================ 판 이름(§1-3·부록 A-14)
ALG_ABBR = ("XGB", "LGB", "CAT", "RF", "ET", "LR", "LSTM", "GRU", "TRF", "LSTMA", "TCN", "CMB", "CMBV", "GRUA")
TREE_ALGS = ("XGB", "LGB", "CAT", "RF", "ET", "LR")
DL_ALGS = ("LSTM", "GRU", "TRF", "LSTMA", "TCN", "CMB", "CMBV", "GRUA")
MODES = wn.MODES  # ('sum', 'raw')


def name_stage1(alg: str, W: int, mode: str | None = None) -> str:
    """1차 판 이름: 트리 = 1-CAT-W060-sum, LSTM = 1-LSTM-W060."""
    if W not in wn.W_GRID:
        raise ValueError(f"W 는 {wn.W_GRID}: {W}")
    if alg in ("XGB", "LGB", "CAT"):
        if mode not in MODES:
            raise ValueError(f"입력 방식: {mode}")
        return f"1-{alg}-W{W:03d}-{mode}"
    if alg == "LSTM":
        if mode is not None:
            raise ValueError("LSTM 판 이름에는 입력 방식 없음")
        return f"1-LSTM-W{W:03d}"
    raise ValueError(f"1차 알고리즘 아님: {alg}")


def dedupe_name(name: str, existing: Iterable[str]) -> str:
    """같은 이름이면 끝에 -2·-3(§1-3)."""
    ex = set(existing)
    if name not in ex:
        return name
    k = 2
    while f"{name}-{k}" in ex:
        k += 1
    return f"{name}-{k}"


# ================================================================ 판 설정
@dataclass(frozen=True)
class VariantSpec:
    name: str
    stage: str               # 단계 번호(예: 1-1)
    order: str               # 차수(준비·1차·2차)
    kind: str                # tree | dl
    algorithm: str           # 구조 약어(A-14)
    W: int | None
    mode: str | None         # sum | raw(트리), 딥러닝 None
    input_set: str           # 입력 묶음 id(칼럼 목록은 artifact)
    role: str                # 후보·교체 후보·비교만·기준 줄
    train_count: int         # 판별 학습 횟수(부록 A-12 — 트리 판 4)
    params: Mapping[str, Any] = field(default_factory=dict)   # 모델 설정(1차 = {} = 라이브러리 기본)
    judgement: str | None = None   # 처음부터 정해진 판정 tag(1차·단계 2 준비 = '해당 없음')
    base_variant: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("tree", "dl"):
            raise ValueError(f"kind: {self.kind}")
        if self.algorithm not in ALG_ABBR:
            raise ValueError(f"구조 약어: {self.algorithm}")
        if self.mode is not None and self.mode not in MODES:
            raise ValueError(f"mode: {self.mode}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = dict(self.params)
        d["extra"] = dict(self.extra)
        return d

    def md5(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def describe(self) -> str:
        parts = [self.input_set, self.algorithm]
        if self.W is not None:
            parts.append(f"W {self.W}")
        if self.mode:
            parts.append({"sum": "요약", "raw": "원값"}[self.mode])
        return " · ".join(parts)


# ================================================================ 주입 꼴(인터페이스 약속)
class InputAssembler(Protocol):
    def fit(self, train_meta: pd.DataFrame) -> "InputAssembler": ...
    def transform(self, meta: pd.DataFrame) -> Any: ...


class Model(Protocol):
    def fit(self, X, y, *, seed: int, n_jobs: int, eval_set=None) -> "Model": ...
    def predict_proba(self, X, n_iter: int | None = None) -> np.ndarray: ...
    def save(self, path) -> None: ...


MakeInputs = Callable[[VariantSpec, str], InputAssembler]
MakeModel = Callable[[VariantSpec, InputAssembler], Model]   # (판 설정, fit 한 조립 객체 — 딥러닝 cat_sizes 등)
FitKwargs = Callable[[VariantSpec, str, pd.DataFrame, Any], Mapping[str, Any]]


# ================================================================ 실험 로그
class ExperimentLog:
    """jsonl 실험 로그(§8): CV 채점 기록('같은 validation 으로 고른 횟수')·다시 돌린 판·멈춤·보고."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_name(self.path.name + ".lock")

    def _locked(self):
        return _FileLock(self.lock)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, rec: dict) -> dict:
        rec = {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "pid": os.getpid(), **rec}
        line = json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return rec

    def append(self, event: str, **kw) -> dict:
        with self._locked():
            return self._append({"event": event, **kw})

    def cv_scored(self, stage: str, name: str, n_settings: int = 1, kind: str = "판") -> int:
        """'같은 validation 으로 고른 횟수'(부록 A-12: "그 판 앞까지 같은 3 fold 로 CV 점수를 낸 모든 판·설정 수의 누적")
        — 이 기록을 남기고, 그 앞 기록들의 합(이 판 자신은 뺌)을 돌려줌. 기록의 count = 이 기록까지 누적.
        같은 (단계, 이름, 종류) 기록이 이미 있으면 그 값(두 번 세지 않음)."""
        if n_settings < 1:
            raise ValueError("n_settings ≥ 1")
        with self._locked():
            total = 0
            for e in self.entries():
                if e.get("event") != "cv_scored":
                    continue
                if (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                    return int(e["count_before"])
                total = int(e["count"])
            rec = self._append({"event": "cv_scored", "stage": stage, "name": name, "kind": kind,
                                "n_settings": n_settings, "count_before": total, "count": total + n_settings})
            return rec["count_before"]

    def has_cv(self, stage: str, kind: str) -> bool:
        return any(e.get("event") == "cv_scored" and e["stage"] == stage and e["kind"] == kind for e in self.entries())

    def count_for(self, stage: str, name: str, kind: str = "판") -> int | None:
        for e in self.entries():
            if e.get("event") == "cv_scored" and (e["stage"], e["name"], e["kind"]) == (stage, name, kind):
                return int(e["count_before"])
        return None


class _FileLock:
    """프로세스 사이 잠금(리눅스 fcntl·Windows msvcrt)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, "a+b")
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            import msvcrt

            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        self.f.close()
        return False


# ================================================================ 코드 사본·이어하기
class ResumeError(RuntimeError):
    pass


class RunStopped(RuntimeError):
    """판 하나가 멈춤 — 전체를 멈춤(B-1: 건너뛰지 않음)."""


def snapshot_code(code_root, out_dir) -> str:
    """결과 폴더에 실행 코드 사본(B-1). 이미 있으면 code_md5 가 같은지 확인(다르면 멈춤)."""
    code_root, out_dir = Path(code_root), Path(out_dir)
    md5 = cf.code_md5(code_root)
    dst = out_dir / "_code"
    if dst.exists():
        old = (dst / "code_md5.txt").read_text(encoding="utf-8").strip()
        if old != md5:
            raise ResumeError(f"결과 폴더의 코드 사본(code_md5 {old})과 지금 코드({md5})가 다름 — 지휘자 결정")
        return md5
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("lst", "scripts", "configs"):
        if (code_root / d).is_dir():
            shutil.copytree(code_root / d, dst / d, ignore=ign)
    (dst / "code_md5.txt").write_text(md5, encoding="utf-8")
    return md5


def variant_dir(out_dir, name: str) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith((".", "_")):
        raise ValueError(f"판 이름이 폴더 이름으로 안 됨: {name!r}")
    return Path(out_dir) / name


def read_json(p) -> Any:
    return cf.load_json_strict(p)


def resume_state(vdir: Path, spec_md5: str, code_md5: str) -> str:
    """'new' | 'done' | 'partial'. done 인데 설정·코드가 다르면 멈춤."""
    if not vdir.exists():
        return "new"
    done = vdir / "done.json"
    if done.exists():
        d = read_json(done)
        if d["spec_md5"] != spec_md5:
            raise ResumeError(f"{vdir.name}: 같은 이름 판의 설정이 다름(설정 md5 {d['spec_md5']} ≠ {spec_md5})")
        if d["code_md5"] != code_md5:
            raise ResumeError(f"{vdir.name}: 끝난 판의 code_md5({d['code_md5']}) ≠ 지금({code_md5}) — 지휘자 결정")
        return "done"
    return "partial"


def _move_partial(vdir: Path, out_dir: Path) -> tuple[Path, str]:
    reason = "비정상 종료(이유 기록 없음 — 메모리 부족 강제 종료 등)"
    f = vdir / "failed.json"
    if f.exists():
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            reason = f"{j.get('type')}: {j.get('message')}"
        except Exception:
            pass
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = out_dir / "_incomplete" / f"{vdir.name}__{ts}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(vdir, dst)
    return dst, reason


# ================================================================ 행 표·예측
def add_bounds(meta: pd.DataFrame, W: int | None) -> pd.DataFrame:
    """행 표에 T·E·S·g·R(§1-1). h 칼럼은 그 행(쌍)의 h. W 없는 판은 S 없음."""
    b = wn.window_bounds(fd.to_days(meta["d"]), meta["h"].to_numpy(), W if W is not None else wn.W_GRID[0])
    out = meta.copy()
    out["h"] = b.h.astype(np.int16)
    out["T"] = fd.as_date(b.T)
    out["E"] = fd.as_date(b.E)
    if W is not None:
        out["S"] = fd.as_date(b.S)
    out["g"] = b.g.astype(np.int16)
    out["R"] = fd.as_date(b.R)
    return out


def _check_proba(p, n: int) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"predict_proba 꼴 {a.shape} ≠ ({n},)")
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("predict_proba 값이 [0, 1] 밖이거나 NaN")
    return a


def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
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
            continue
        if check is not None:
            check()
        m = meta.iloc[idx].copy()
        m["h"] = np.int16(h)
        m = add_bounds(m, W)
        p = _check_proba(model.predict_proba(asm.transform(m)), idx.size)
        pos_l.append(idx)
        h_l.append(np.full(idx.size, h, np.int16))
        p_l.append(p)
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    return pd.DataFrame({"lims_data_key": samples.keys[pos], "h": np.concatenate(h_l) if h_l else np.empty(0, np.int16),
                         "fold": split, "pred": np.concatenate(p_l) if p_l else np.empty(0)})


def pred_path(vdir: Path, split: str) -> Path:
    return vdir / f"pred_{split}.parquet"


def load_preds(vdir: Path, splits: Sequence[str]) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(pred_path(vdir, s)) for s in splits}


def _model_dump(model) -> Any:
    """B-4 실제 적용값: models_tree.config_dump·models_dl.get_config(있는 것)."""
    for m in ("config_dump", "get_config", "applied_params"):
        if hasattr(model, m):
            return getattr(model, m)()
    return None


def _lib_versions() -> dict:
    from importlib import metadata

    out = {}
    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch", "keras", "autokeras", "mlflow"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


# ================================================================ 판 하나 실행
@dataclass
class RunContext:
    samples: ev.SampleTable
    out_dir: Path
    code_root: Path
    input_md5s: Mapping[str, str]
    exp_log: ExperimentLog
    make_inputs: MakeInputs
    make_model: MakeModel
    n_jobs: int
    seed: int = 42
    tracker: Any = None            # tracking.Tracker | None
    experiment: str | None = None  # listeria-1차 등
    stage_run_id: str | None = None
    code_reviewed: bool = False
    fit_kwargs: FitKwargs | None = None
    check_resources: Callable[[Any], Any] = _check_resources
    splits: Sequence[str] = SPLIT_ORDER
    log_pred_artifacts: bool = True
    smoke: Mapping[str, int] | None = None   # 연기 시험만: {max_train_rows, max_pairs_per_h} — 실제 실행은 None
    require_7a_logged: bool = True           # CV 채점 단계 시작 전 0-10(7-a 격자 75) 기록 확인 — 연기 시험에서만 끔


TE7A_STAGE, TE7A_KIND = "0-10", "7-a 격자"   # s0_10_log_mlflow 가 남기는 기록


def check_prior_7a(exp_log: ExperimentLog) -> None:
    """'같은 validation 으로 고른 횟수'(부록 A-12)가 7-a 격자 75 부터 세어지도록, CV 채점 단계 전에 0-10 기록 확인."""
    if not exp_log.has_cv(TE7A_STAGE, TE7A_KIND):
        raise ResumeError("실험 로그에 0-10(7-a 격자) 기록이 없음 — s0_10_log_mlflow.py 를 먼저(부록 A-12 셈)")


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


def run_variant(spec: VariantSpec, ctx: RunContext) -> dict:
    """판 하나(§3: 3 fold + refit·test 예측) → CV 채점·기록. 끝난 판이면 건너뜀(B-1)."""
    if tuple(ctx.splits) != SPLIT_ORDER:
        raise ValueError("판 하나 = fold1~3 + refit")
    smoke = dict(ctx.smoke) if ctx.smoke else None
    if smoke is not None and set(smoke) != set(SMOKE_KEYS):
        raise ValueError(f"smoke 칸은 {SMOKE_KEYS}")
    out_dir = Path(ctx.out_dir)
    vdir = variant_dir(out_dir, spec.name)
    if ctx.require_7a_logged:
        check_prior_7a(ctx.exp_log)
    code_md5 = snapshot_code(ctx.code_root, out_dir)   # 단계 시작 때 사본과 다른 코드면 멈춤(한 단계 표에 코드 섞임 방지)
    smd5 = spec.md5()
    state = resume_state(vdir, smd5, code_md5)
    if state == "done":
        return {"status": "skipped", **read_json(vdir / "done.json")}
    if state == "partial":
        moved, reason = _move_partial(vdir, out_dir)
        ctx.exp_log.append("rerun", stage=spec.stage, name=spec.name, reason=reason, moved_to=str(moved.name))
    vdir.mkdir(parents=True)
    cf.dump_json(vdir / "spec.json", spec.to_dict())
    cf.dump_json(vdir / "started.json", {"code_md5": code_md5, "spec_md5": smd5, "pid": os.getpid(),
                                          "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    ctx.exp_log.append("start", stage=spec.stage, name=spec.name, spec_md5=smd5, code_md5=code_md5)
    run_id = None
    tr = ctx.tracker
    try:
        if tr is not None:
            tags = {"단계": spec.stage, "역할": spec.role, "코드 검수 완료": "예" if ctx.code_reviewed else "아니오",
                    "설정 md5": smd5, "code_md5": code_md5}
            if spec.base_variant:
                tags["바탕 판"] = spec.base_variant
            params = {"algorithm": spec.algorithm, "W": spec.W, "mode": spec.mode,
                      "W_DL": spec.W if spec.kind == "dl" else None, "input_set_id": spec.input_set,
                      "seed": ctx.seed, "n_jobs": ctx.n_jobs, "model_params": dict(spec.params),
                      "train_count": spec.train_count, "lib_versions": _lib_versions(), **dict(spec.extra)}
            run_id = tr.start_variant(ctx.experiment, ctx.stage_run_id, spec.name, tags, params)
            if spec.judgement:
                tr.set_judgement(run_id, spec.judgement)
        timings: dict[str, dict] = {}
        known = ctx.samples.known_frame()
        res0 = ctx.check_resources(out_dir)
        for split in SPLIT_ORDER:
            ctx.check_resources(out_dir)
            t0 = time.time()
            rows = fd.make_train_rows(known, fd.SPLITS[split])
            if smoke is not None:
                rows = _smoke_train(rows, smoke["max_train_rows"], ctx.seed)
            rows = add_bounds(rows, spec.W)
            y = rows["y"].to_numpy(dtype=np.int64)
            asm = ctx.make_inputs(spec, split).fit(rows)
            X = asm.transform(rows)
            model = ctx.make_model(spec, asm)
            kw = dict(ctx.fit_kwargs(spec, split, rows, X)) if ctx.fit_kwargs else {}
            model.fit(X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
            t1 = time.time()
            del X
            model.save(vdir / f"model_{split}")
            if hasattr(asm, "save"):
                asm.save(vdir / f"inputs_{split}")
            if hasattr(asm, "columns"):
                cf.dump_json(vdir / f"columns_{split}.json", {"columns": list(asm.columns)})
            dump = _model_dump(model)   # B-4 실제 적용값
            if dump is not None:
                cf.dump_json(vdir / f"applied_params_{split}.json", cf.nan_to_null(dump))
            preds = predict_pairs(model, asm, ctx.samples, split, spec.W, check=lambda: ctx.check_resources(out_dir),
                                  max_pairs_per_h=smoke["max_pairs_per_h"] if smoke else None, seed=ctx.seed)
            p = pred_path(vdir, split)
            preds.to_parquet(p, index=False)
            timings[split] = {"train_rows": int(len(rows)), "train_pos": int(y.sum()), "fit_seconds": round(t1 - t0, 2),
                              "predict_seconds": round(time.time() - t1, 2), "n_pairs": int(len(preds))}
            del model, asm, preds, rows
        cv = ev.score_cv(load_preds(vdir, ev.CV_FOLDS), ctx.samples, require_complete=smoke is None)
        count = ctx.exp_log.cv_scored(spec.stage, spec.name)
        cf.dump_scores_json(vdir / "cv_scores.json", cv)
        total_s = round(sum(t["fit_seconds"] + t["predict_seconds"] for t in timings.values()), 2)
        cf.save_run_meta(vdir / "run_meta.json", spec.to_dict(), ctx.input_md5s, ctx.code_root, n_jobs=ctx.n_jobs,
                         extra={"timings": timings, "seconds_total": total_s, "resources_at_start": res0,
                                "spec_md5": smd5, "smoke": smoke})
        if tr is not None:
            tr.log_cv(run_id, cv, selection_count=count)
            tr.log_metrics(run_id, {"train_seconds": total_s})
            for f in ("spec.json", "run_meta.json", "cv_scores.json"):
                tr.log_artifact(run_id, vdir / f)
            for f in sorted(vdir.glob("columns_*.json")) + sorted(vdir.glob("applied_params_*.json")):
                tr.log_artifact(run_id, f, "config")
            if ctx.log_pred_artifacts:
                for s in SPLIT_ORDER:
                    tr.log_artifact(run_id, pred_path(vdir, s), "predictions")
            tr.finish(run_id)
        done = {"name": spec.name, "stage": spec.stage, "spec_md5": smd5, "code_md5": code_md5,
                "cv_band_weighted": cv["cv_band_weighted"], "many_ties": cv["ties"]["many_ties"],
                "selection_count": count, "mlflow_run_id": run_id, "seconds_total": total_s, "smoke": smoke}
        cf.dump_scores_json(vdir / "done.json", done)
        ctx.exp_log.append("done", stage=spec.stage, name=spec.name, cv=cv["cv_band_weighted"], count=count)
        return {"status": "done", **done}
    except BaseException as e:
        try:
            cf.dump_json(vdir / "failed.json", {"type": type(e).__name__, "message": str(e)[:2000],
                                                "traceback": traceback.format_exc()[-8000:]}, overwrite=True)
            ctx.exp_log.append("failed", stage=spec.stage, name=spec.name, type=type(e).__name__, message=str(e)[:500])
            if tr is not None and run_id is not None:
                tr.finish(run_id, "FAILED")
        except Exception:  # 기록 실패가 원래 예외를 가리지 않게
            pass
        raise


# ================================================================ 여러 판(하위 프로세스)
def _rc_reason(rc: int) -> str:
    if rc < 0:
        return f"신호 {-rc}로 강제 종료(9 = SIGKILL — 메모리 부족 강제 종료 가능)"
    if rc == 137:
        return "137(SIGKILL — 메모리 부족 강제 종료 가능)"
    return f"종료 코드 {rc}(failed.json 참고)"


def run_variants_isolated(script, names: Sequence[str], base_argv: Sequence[str], exp_log: ExperimentLog,
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
    return finished


# ================================================================ 단계 표·보고
def load_done(out_dir, name: str) -> tuple[dict, dict, dict]:
    vdir = variant_dir(out_dir, name)
    if not (vdir / "done.json").exists():
        raise FileNotFoundError(f"끝나지 않은 판: {name}")
    return read_json(vdir / "spec.json"), read_json(vdir / "done.json"), _load_scores(vdir / "cv_scores.json")


def _load_scores(p) -> dict:
    """NaN 을 null 로 저장한 점수 json 을 다시 읽음(null → NaN)."""
    def fix(o):
        if o is None:
            return float("nan")
        if isinstance(o, dict):
            return {(int(k) if isinstance(k, str) and k.isdigit() else k): fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        return o
    return fix(read_json(p))


AUTO_LR_STAGES = ("2-0", "2-1", "2-1b", "2-R")   # 단계 2 준비(부록 A-13: 1차·단계 2 준비 CatBoost 판 표시)


def _marks(spec: Mapping) -> list[str]:
    """표시 칸: 1차·단계 2 준비 CatBoost 판 = '학습률 = 학습마다 자동값'(부록 A-13). ③·④ 는 그 판 스크립트가 extra 로."""
    auto = spec["order"] == "1차" or spec["stage"] in AUTO_LR_STAGES or bool(spec.get("extra", {}).get("cat_auto_lr"))
    return [ev.CAT_AUTO_LR_MARK] if spec["algorithm"] == "CAT" and auto else []


def stage_cv_rows(out_dir, names: Sequence[str]) -> list[dict]:
    """보고 전(test 없음) 단계 표 줄."""
    rows = []
    for n in names:
        spec, done, cv = load_done(out_dir, n)
        rows.append(ev.table_row(n, VariantSpec(**spec).describe(), cv, None,
                                 selection_count=done["selection_count"], marks=_marks(spec)))
    return rows


def candidates(out_dir, names: Sequence[str]) -> list[ev.Candidate]:
    """고르기 후보(CV 값만 — test 칸 없음)."""
    out = []
    for n in names:
        spec, done, cv = load_done(out_dir, n)
        out.append(ev.Candidate(n, float(cv["cv_band_weighted"]), int(spec["train_count"]), bool(cv["ties"]["many_ties"]),
                                {"algorithm": spec["algorithm"], "W": spec["W"], "mode": spec["mode"], "kind": spec["kind"]}))
    return out


def rescore_fn(out_dir, samples: ev.SampleTable) -> Callable[[str], float]:
    """다시 채점(부록 A-12): 저장된 fold 예측을 u_tie = md5(key|h|1) 로 다시 채점한 CV 평균."""
    def f(name: str) -> float:
        vdir = variant_dir(out_dir, name)
        return float(ev.score_cv(load_preds(vdir, ev.CV_FOLDS), samples, rescore=True)["cv_band_weighted"])
    return f


@dataclass
class BaseRow:
    """표의 기준 줄 등 판 폴더 밖 줄(7-a). cv_preds = {fold: 쌍 예측}, test_preds = refit 쌍 예측."""
    name: str
    desc: str
    cv_preds: Mapping[str, pd.DataFrame]
    test_preds: pd.DataFrame | None
    selection_count: int | None
    mlflow_run_id: str | None = None


def te7a_baseline(dir_0_10, samples: ev.SampleTable, exp_log: ExperimentLog | None = None) -> BaseRow:
    """단계 0-10 결과(고른 7-a 칸)를 기준 줄로: 시료 단위 예측을 쌍으로 펼침."""
    d = Path(dir_0_10)
    sel = read_json(d / "te7a_selected.json")
    name = sel["selected"]
    cvp = pd.read_parquet(d / f"te7a_cv_pred_{name}.parquet")
    cv_preds = {f: ev.expand_sample_preds(cvp[cvp["fold"] == f], samples, f) for f in ev.CV_FOLDS}
    tp = d / f"te7a_test_pred_{name}.parquet"
    test_preds = ev.expand_sample_preds(pd.read_parquet(tp), samples, "refit") if tp.exists() else None
    cnt = exp_log.count_for("0-10", name, "7-a 격자") if exp_log is not None else None
    return BaseRow(name, "타겟 인코딩 단독(7-a, 학습 없음) — 기준 줄", cv_preds, test_preds, cnt)


def report_stage(*, out_dir, stage: str, names: Sequence[str], samples: ev.SampleTable, samples_path,
                 permit: ev.ReportPermit, incomplete: Iterable[tuple[str, str]], exp_log: ExperimentLog,
                 tracker=None, stage_run_id: str | None = None, base_rows: Sequence[BaseRow] = (),
                 fold_recall: bool = False, table_name: str = "stage_table_report.csv",
                 extra_notes: Sequence[str] = ()) -> Path:
    """단계 보고(§1-3): 보고 허가로 test 정답을 읽어 모든 판 test 채점 → 단계 표(부록 A-12) + MLflow test 지표."""
    ev.check_permit(permit)
    out_dir = Path(out_dir)
    truth = ev.TestTruth.load(samples_path, permit)
    inc = set(incomplete)
    rows, tests, scored = [], [], []
    for b in base_rows:
        cv = ev.score_cv(b.cv_preds, samples)
        t = ev.score_test(b.test_preds, samples, truth, cv["threshold"]["threshold"], inc) if b.test_preds is not None else None
        if t is not None:
            tests.append(t)
            if tracker is not None and b.mlflow_run_id:
                tracker.log_test(b.mlflow_run_id, t, permit)
        rows.append(ev.table_row(b.name, b.desc, cv, t, selection_count=b.selection_count))
    for n in names:
        spec, done, cv = load_done(out_dir, n)
        vdir = variant_dir(out_dir, n)
        tpath = vdir / "test_scores.json"
        if tpath.exists():
            t = _load_scores(tpath)
        else:
            t = ev.score_test(pd.read_parquet(pred_path(vdir, "refit")), samples, truth,
                              float(cv["threshold"]["threshold"]), inc, require_complete=not done.get("smoke"))
            cf.dump_scores_json(tpath, t)
            if tracker is not None and done.get("mlflow_run_id"):
                tracker.log_test(done["mlflow_run_id"], t, permit)
                tracker.log_artifact(done["mlflow_run_id"], tpath)
            scored.append(n)
        tests.append(t)
        rows.append(ev.table_row(n, VariantSpec(**spec).describe(), cv, t, selection_count=done["selection_count"],
                                 marks=_marks(spec), fold_recall=fold_recall))
    notes = ev.table_notes(tests, extra_notes)
    p = ev.write_table(rows, notes, out_dir / table_name)
    exp_log.append("test_reported", stage=stage, names=list(names) + [b.name for b in base_rows],
                   newly_scored=scored, permit_reason=permit.reason)
    if tracker is not None and stage_run_id is not None:
        tracker.log_artifact(stage_run_id, p, "tables")
        tracker.log_artifact(stage_run_id, p.with_suffix(".notes.json"), "tables")
    return p
