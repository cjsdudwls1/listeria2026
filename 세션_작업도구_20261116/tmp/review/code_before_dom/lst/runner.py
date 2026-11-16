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
  - (11-08) 같은 판 설정의 딥러닝 판도 한 묶음(입력·전처리기 한 번, 모델 차례로 — `input_key`), 묶음 하나가 멈추면 그 묶음
    판 모두 다시.

2차 사슬 부스팅 판(판 설정 params['iter_select'] — §3·§1-2, `_train_chain`):
  CatBoost 학습률 읽기(fold 1~3, 부록 A-12) → 평균 → 최대 반복 = 실험 로그 기록(없으면 이 판에서 fold 3 먼저 학습해 자기
  곡선 최적 N 이 마지막 시점이면 2배씩, 상한 16,000 — 기록 'max_iter_decided', 그 알고리즘의 이후 판이 읽음) → fold 마다
  최대 반복까지 학습 → 시점(10~200 은 10, 그 뒤 50 트리마다)별 h 별 PR-AUC → 구간 가중 곡선 → 교차 N·공통 N
  (models_tree.select_n_iter) → fold 모델로 평가 쌍을 다시 만들어 교차 N 예측(pred_<fold>.parquet — CV 점수)·공통 N 예측
  (pred_thr_<fold>.parquet — threshold) → refit 최대 반복까지 학습 → test 쌍 공통 N 예측. 같은 묶음의 기본 설정 판
  (RF·ExtraTrees)은 같은 입력으로 기본대로. 기록 = n_iter.json(fold 별 교차 N·최적 N·train 행 수·공통 N·최대 반복·학습률)·
  n_iter_curves.csv·n_iter_curves_by_h.csv, done.json 'n_iter', 표 '상한'(교차 N·공통 N 이 마지막 시점) + 실험 로그 알림.
  구현 세부(계획서에 없음): 곡선을 내는 동안은 쌍 × 시점 예측을 들고 있지 않음(메모리) — N 을 정한 뒤 평가 쌍 입력을 fold
  마다 한 번 더 만듦. 최대 반복을 늘려 fold 3 을 다시 학습할 때 버린 모델은 저장하지 않음.
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
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from . import config as cf
from . import evaluate as ev
from . import folds as fd
from . import windows as wn
from .hassign import H_MAX
from .resources import check_resources as _check_resources

SPLIT_ORDER = ("fold1", "fold2", "fold3", "refit")
MAX_ITER_EVENT = "max_iter_decided"

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

    # ---- 2차 사슬 부스팅 최대 반복(§3: 알고리즘마다 첫 2차 사슬 부스팅 판에서 한 번 정함)
    def max_iter_record(self, algorithm: str) -> dict | None:
        recs = [e for e in self.entries() if e.get("event") == MAX_ITER_EVENT and e["algorithm"] == algorithm]
        if len(recs) > 1:
            raise ResumeError(f"실험 로그에 {algorithm} 최대 반복 기록이 {len(recs)}개 — 지휘자 확인")
        return recs[0] if recs else None

    def record_max_iter(self, algorithm: str, **kw) -> dict:
        """최대 반복 판정 기록(한 번만 — 이미 있으면 멈춤)."""
        with self._locked():
            if any(e.get("event") == MAX_ITER_EVENT and e["algorithm"] == algorithm for e in self.entries()):
                raise ResumeError(f"{algorithm} 최대 반복이 이미 정해짐 — 두 번 정하지 않음")
            return self._append({"event": MAX_ITER_EVENT, "algorithm": algorithm, **kw})


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


def resume_state(vdir: Path, spec_md5: str, code_md5: str, accepted: Mapping[str, Iterable[str]] | None = None) -> str:
    """'new' | 'done' | 'partial' | 'stale'. done 인데 설정이 다르면 멈춤.
    accepted = {인정된 이전 code_md5: 인정한 판 이름들}(`accept_previous_code`) — 그 코드로 끝난 판 중 인정 목록의 판만
    끝난 판, 목록 밖이면 'stale'(새 코드로 다시), 인정 기록이 없는 다른 코드면 멈춤."""
    accepted = accepted or {}
    if not vdir.exists():
        return "new"
    done = vdir / "done.json"
    if done.exists():
        d = read_json(done)
        if d["spec_md5"] != spec_md5:
            raise ResumeError(f"{vdir.name}: 같은 이름 판의 설정이 다름(설정 md5 {d['spec_md5']} ≠ {spec_md5})")
        if d["code_md5"] != code_md5 and d["code_md5"] in accepted:
            return "done" if vdir.name in set(accepted[d["code_md5"]]) else "stale"
        if d["code_md5"] != code_md5:
            raise ResumeError(f"{vdir.name}: 끝난 판의 code_md5({d['code_md5']}) ≠ 지금({code_md5}) — 지휘자 결정")
        return "done"
    return "partial"


def _move_partial(vdir: Path, out_dir: Path, sub: str = "_incomplete", reason: str | None = None) -> tuple[Path, str]:
    if reason is not None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = out_dir / sub / f"{vdir.name}__{ts}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(vdir, dst)
        return dst, reason
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


def _copy_X(X):
    """모델마다 입력 사본(한 모델이 입력을 고쳐도 다음 모델에 안 번지게 — 값은 그대로)."""
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=True)
    if isinstance(X, dict):
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in X.items()}
    return X


# ---------------------------------------------------------------- 같은 입력 행은 한 번만 예측(결과 같음)
_HASH_SEED = 20261108


def _byte_parts(X) -> list[np.ndarray] | None:
    """입력 → 행 비교용 조각들: 1-D (n,) uint64(문자열 범주 = 그 입력 안 번호) 또는 2-D (n, b) uint8(숫자 = 값 비트 그대로 —
    NaN·부호 포함, DataFrame 은 같은 dtype 으로 이어진 칼럼 묶음마다, 딥러닝 배열은 행 바이트). 다룰 수 없는 꼴(결측 든
    범주·확장 dtype 등)이면 None(중복 제거 안 함)."""
    if isinstance(X, pd.DataFrame):
        n = len(X)
        parts = []
        cols = list(X.columns)
        dts = [X[c].dtype for c in cols]
        j = 0
        while j < len(cols):
            dt = dts[j]
            if isinstance(dt, np.dtype) and dt.kind in "fiub":
                k = j
                while k < len(cols) and dts[k] == dt:
                    k += 1
                v = np.asarray(X.iloc[:, j:k].to_numpy(dtype=dt))      # 같은 dtype 이어진 칼럼 — 대개 복사 없는 보기
                if not v.flags["C_CONTIGUOUS"] and v.strides[-1] != v.itemsize:
                    v = np.ascontiguousarray(v)
                parts.append(v.view(np.uint8))                         # (n, 바이트) — 값 비트 그대로
                j = k
                continue
            if isinstance(dt, np.dtype) and dt.kind == "O":
                v = X.iloc[:, j].to_numpy()
                if n and pd.api.types.infer_dtype(v, skipna=False) != "string":
                    return None
                parts.append(pd.factorize(v)[0].astype(np.int64).view(np.uint64))
                j += 1
                continue
            return None
        return parts
    if isinstance(X, Mapping):
        out = []
        for k in sorted(X):
            a = np.asarray(X[k])
            if a.dtype.kind not in "fiub":
                return None
            a = np.ascontiguousarray(a)
            out.append(a.reshape(a.shape[0], -1).view(np.uint8).reshape(a.shape[0], -1))
        if len({p.shape[0] for p in out}) > 1:
            return None
        return out
    return None


def _words(p: np.ndarray) -> np.ndarray:
    """(n, b) uint8 → 같은 바이트를 가장 큰 부호 없는 정수 낱말로 본 (n, b / w)."""
    b = p.shape[1]
    w = 8 if b % 8 == 0 else 4 if b % 4 == 0 else 2 if b % 2 == 0 else 1
    return p.view(np.dtype(f"<u{w}")) if w > 1 else p


def _part_hash(p: np.ndarray, rng) -> np.ndarray:
    """조각 → 행 해시 uint64(무작위 홀수 가중 합, 2^64 나머지 — 한 칸만 다른 행은 늘 다른 값)."""
    if p.ndim == 1:
        w = rng.integers(0, 2 ** 63, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        return p.astype(np.uint64) * w
    n = p.shape[0]
    words = _words(p)
    k = words.shape[1]
    wt = rng.integers(0, 2 ** 63, size=k, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    h = np.zeros(n, np.uint64)
    step = 256
    for j in range(0, k, step):
        blk = words[:, j:j + step].astype(np.uint64)
        blk *= wt[j:j + step]
        h += blk.sum(axis=1, dtype=np.uint64)
    return h


def unique_rows(X) -> tuple[np.ndarray, np.ndarray]:
    """바이트가 완전히 같은 입력 행 묶기 → (대표 행 위치(첫 등장 순), 행 → 대표 번호). 해시로 묶은 뒤 대표 행과
    바이트를 다시 맞춰 봄(다르면 그 행은 따로 — 결과는 늘 정확). 다룰 수 없는 입력이면 묶지 않음(항등)."""
    parts = _byte_parts(X)
    n = len(X) if isinstance(X, pd.DataFrame) else (len(next(iter(X.values()))) if isinstance(X, Mapping) and X else 0)
    if parts is None or n == 0 or not parts:
        return np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(_HASH_SEED)
    h = np.zeros(n, np.uint64)
    for p in parts:
        if p.ndim == 1:
            h += _part_hash(p, rng)
        else:
            h = h * np.uint64(0x9E3779B97F4A7C15) + _part_hash(p, rng)
    code, _ = pd.factorize(h)
    code = code.astype(np.int64)
    cm = np.maximum.accumulate(code)                   # 번호 = 첫 등장 순 → 첫 행 = 누적 최댓값이 늘어나는 자리
    first = np.flatnonzero(np.concatenate([[True], cm[1:] > cm[:-1]])).astype(np.int64)
    rep = first[code]
    chk = np.flatnonzero(rep != np.arange(n))
    bad = np.zeros(chk.size, bool)
    rc = rep[chk]
    for p in parts:
        if p.ndim == 1:
            bad |= p[chk] != p[rc]
        else:
            q = _words(p)
            for lo in range(0, chk.size, 8192):
                r = chk[lo:lo + 8192]
                bad[lo:lo + 8192] |= (q[r] != q[rep[r]]).any(axis=1)
    if bad.any():   # 해시 충돌 — 그 행은 각자 대표
        br = chk[bad]
        code[br] = first.size + np.arange(br.size)
        first = np.concatenate([first, br])
    return first.astype(np.int64), code


def take_rows(X, pos: np.ndarray):
    """입력의 행 일부(새 객체, 칼럼·dtype 그대로)."""
    if isinstance(X, pd.DataFrame):
        out = X.iloc[pos]
        out.index = pd.RangeIndex(len(pos))      # reset_index 는 값까지 한 번 더 복사 — 색인만 바꿈
        return out
    return {k: np.ascontiguousarray(np.asarray(v)[pos]) for k, v in X.items()}


# 같은 행 한 번 예측을 쓰는 알고리즘: 행마다 예측이 다른 행과 무관하다고 확인한 것만(합성·실제 자료로 바이트 같음 —
# tests/test_dedup, 1-1 연기 시험). 로지스틱(BLAS 행렬곱)·딥러닝(배치 크기에 따라 부동소수 합 순서가 바뀜)은 예측 행 수가
# 바뀌면 마지막 자리가 달라지고(합성 자료 실측: 로지스틱 쌍 0.6 % 다름, LSTM CPU 다름) — 이들은 모든 행 예측 그대로,
# 입력 계산 중복 없애기만 씀. RF·ExtraTrees 는 11-08 부터 나무 순서 더하기 예측(models_tree 구현 세부 11 — 실행마다 같은 값,
# 행마다 다른 행과 무관, tests/test_chain)이라 넣음.
ROW_DEDUPE_ALGS = ("XGB", "LGB", "CAT", "RF", "ET")


def row_dedupe_ok(specs: Sequence["VariantSpec"]) -> bool:
    return bool(specs) and all(sp.kind == "tree" and sp.algorithm in ROW_DEDUPE_ALGS for sp in specs)


def predict_pairs(model, asm, samples: ev.SampleTable, split: str, W: int | None,
                  check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                  seed: int = 42, dedupe_rows: bool = False) -> pd.DataFrame:
    """평가 쌍을 h 마다 메모리에서 만들어 바로 예측(B-1). 행 표에 y 없음.
    max_pairs_per_h = 연기 시험용(쌍 일부, 씨앗 고정) — 실제 실행은 None(쌍 전부, §3)."""
    return predict_pairs_multi({"_": model}, asm, samples, split, W, check, max_pairs_per_h, seed,
                               dedupe_rows=dedupe_rows)[0]["_"]


def predict_pairs_multi(models: Mapping[str, Any], asm, samples: ev.SampleTable, split: str, W: int | None,
                        check: Callable[[], Any] | None = None, max_pairs_per_h: int | None = None,
                        seed: int = 42, dedupe_rows: bool = False, copy_inputs: bool | None = None
                        ) -> tuple[dict[str, pd.DataFrame], dict[str, float], float]:
    """h 마다 평가 쌍 입력을 한 번 만들어 여러 모델이 예측(같은 입력 설정의 판 묶음). 메모리에서만(B-1).
    계산 중복 없애기(구현 세부 — 결과 같음, 담당자 승인 11-08): 조립 객체에 `pair_cache` 가 있으면 h 묶음 사이에
    블록 계산을 키 고유값으로 이어 쓰고(inputs 구현 세부 (g)), dedupe_rows 면(ROW_DEDUPE_ALGS 판만 — run_group 이 정함)
    각 h 의 입력에서 바이트가 완전히 같은 행(NaN 포함)은 한 번만 예측해 행으로 펼침(`unique_rows`).
    돌려줌 = (판별 예측 표, 판별 예측 초, 입력 만들기 초 — 같은 행 묶기 포함)."""
    import contextlib

    sp = fd.SPLITS[split]
    meta = samples.meta()
    copy = len(models) > 1 if copy_inputs is None else bool(copy_inputs)
    pos_l, h_l = [], []
    p_l = {k: [] for k in models}
    secs = {k: 0.0 for k in models}
    t_in = 0.0
    scope = asm.pair_cache() if hasattr(asm, "pair_cache") else contextlib.nullcontext()
    with scope:
        for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, sp):
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
            t_in += time.time() - t0
            for k, model in models.items():
                t0 = time.time()
                p = _check_proba(model.predict_proba(_copy_X(X) if copy else X), rep.size)
                p_l[k].append(p[inv] if rep.size < idx.size else p)
                secs[k] += time.time() - t0
            del X
            pos_l.append(idx)
            h_l.append(np.full(idx.size, h, np.int16))
    pos = np.concatenate(pos_l) if pos_l else np.empty(0, np.int64)
    keys = samples.keys[pos]
    hh = np.concatenate(h_l) if h_l else np.empty(0, np.int16)
    out = {k: pd.DataFrame({"lims_data_key": keys, "h": hh, "fold": split,
                            "pred": np.concatenate(v) if v else np.empty(0)}) for k, v in p_l.items()}
    return out, {k: round(v, 2) for k, v in secs.items()}, round(t_in, 2)


H_METRICS_FILE = "h_metrics.csv"


def write_h_metrics(vdir: Path, df: pd.DataFrame) -> Path:
    """판 폴더 h 별 표(utf-8-sig). 보고 때 test 행을 더하면 CV 행이 그대로인지 확인한 뒤 바꿔 씀(원자적 교체)."""
    p = Path(vdir) / H_METRICS_FILE
    if p.exists():
        old = read_h_metrics(p)
        cv_old = old[old["split"] != "test"].reset_index(drop=True)
        cv_new = df[df["split"] != "test"].reset_index(drop=True)
        pd.testing.assert_frame_equal(cv_old, cv_new.astype(cv_old.dtypes.to_dict()), check_exact=False, rtol=0,
                                      atol=1e-12)
        if (old["split"] == "test").any() and not (df["split"] == "test").any():
            raise AssertionError("test 행이 있는 h 별 표를 test 없는 표로 바꾸지 않음")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        df.to_csv(fh, index=False, float_format="%.17g")   # 다시 읽어도 같은 값(왕복 정확)
    os.replace(tmp, p)
    return p


def read_h_metrics(p) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")


def pred_path(vdir: Path, split: str) -> Path:
    return vdir / f"pred_{split}.parquet"


def load_preds(vdir: Path, splits: Sequence[str]) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(pred_path(vdir, s)) for s in splits}


def load_thr_preds(vdir: Path) -> dict[str, pd.DataFrame] | None:
    """2차 사슬 부스팅 판의 threshold 용 공통 N fold 예측(pred_thr_fold<f>.parquet) — 없으면 None(CV 예측으로)."""
    ps = {f: pred_path(vdir, THR_PREFIX + f) for f in ev.CV_FOLDS}
    if not any(p.exists() for p in ps.values()):
        return None
    return {f: pd.read_parquet(p) for f, p in ps.items()}


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


def input_key(spec: VariantSpec) -> str | None:
    """같은 입력을 쓰는 판 묶음 열쇠: 입력 묶음 id·W·입력 방식·extra(판 설정의 단위 칸 등)가 같으면 같은 입력
    (알고리즘만 다름). 트리(로지스틱 제외 — 입력 표준화가 모델마다 다름)와 딥러닝(11-08 — 같은 판 설정이면 전처리기
    fit 이 같은 train 으로 같은 값이라 입력 dict 가 같음, 3-0 딥러닝·결합 7종·⑥⑦-D)을 묶음. 로지스틱 = None(혼자)."""
    if spec.kind == "tree" and spec.algorithm == "LR":
        return None
    return json.dumps([spec.kind, spec.input_set, spec.W, spec.mode, dict(spec.extra)], sort_keys=True,
                      ensure_ascii=False)


def _copy_needed(specs: Sequence[VariantSpec]) -> bool:
    """묶음에서 모델마다 입력 사본을 줄지: 트리 묶음만(딥러닝은 받은 배열을 고치지 않음 — models_dl 은 a[idx] 로 새 배열만
    만듦 — 사본을 주면 x_seq 수 GB 를 모델마다 복사함)."""
    return len(specs) > 1 and any(sp.kind == "tree" for sp in specs)


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
    accepted = accepted_map(out_dir)
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
        if state == "stale":   # 이전 코드로 끝났으나 인정 범위 밖 → 옮기고 새 코드로 다시
            d = read_json(vdir / "done.json")
            reason = f"이전 코드({d['code_md5']}) 결과 인정 범위 밖 — 새 코드로 다시"
            moved, _ = _move_partial(vdir, out_dir, "_superseded", reason)
            if ctx.tracker is not None and d.get("mlflow_run_id"):
                ctx.tracker.supersede(d["mlflow_run_id"], reason, sp.name)
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
            thr = load_thr_preds(vdir) if ch is not None else None
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
    curves = {k: np.full((len(j[2]), H_MAX), np.nan) for k, j in jobs.items() if j[1] == "curve"}
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
    fit_copy = copy or bool(determine)   # 최대 반복을 늘려 다시 학습할 때 같은 X 를 또 씀
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
                model.fit(_copy_X(X) if fit_copy else X, y, seed=ctx.seed, n_jobs=ctx.n_jobs, **kw)
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


def acceptable(spec: VariantSpec, group: Sequence[VariantSpec]) -> bool:
    """인정 범위: 확인한 경로와 같은 종류의 판 = 묶음 실행 트리 판(로지스틱·딥러닝·묶이지 않는 판 제외), 알고리즘이
    확인 묶음 안의 것, 입력 묶음 id 가 같은 판."""
    return (spec.kind == "tree" and input_key(spec) is not None and spec.algorithm in {g.algorithm for g in group}
            and spec.input_set in {g.input_set for g in group})


def accept_previous_code(out_dir, code_root, acc: Mapping[str, str], group: Sequence[VariantSpec],
                         verify_fn: Callable[[Path, Sequence[str]], Any], exp_log: ExperimentLog, stage: str) -> str:
    """단계 코드 사본(_code)의 code_md5 가 acc['code_md5'](이전 코드)이고 지금 코드가 다를 때: 확인 판이 든 묶음
    (group_names)을 새 코드로 임시 폴더에 다시 돌려(verify_fn(임시 폴더, 판 이름들)) 이전 코드로 끝난 그 판들의
    pred_*.parquet·cv_scores.json 과 바이트 단위로 같으면 → 실험 로그 'code_change_same_results', 옛 사본을
    `_code_prev_<md5>` 로 남기고 새 코드 사본으로 바꾸고 이전 code_md5 를 인정 목록에 적음(그 코드로 끝난 판은 건너뜀).
    다르면 실험 로그에 남기고 멈춤. 돌려줌 = 'no_snapshot'·'same_code'·'accepted'·'already_accepted'.
    인정하는 판 = 이전 코드로 끝난 판 중 `acceptable`(확인 묶음과 같은 실행 경로)인 판만 — 그 밖의 이전 코드 판은
    run_group 에서 '_superseded' 로 옮기고 새 코드로 다시.
    이어 인정(코드 A → B 인정 뒤 B → C): 확인 묶음의 비교 대상 = B 로 끝난 판 + A 로 끝나 인정 목록에 든 판(확인 판도
    둘 중 하나면 됨). A 로 끝나 인정된 판은 이번 확인 묶음 기준 `acceptable` 이면 계속 인정, 아니면 A 기록의 인정 목록에서
    빠짐(다시 돌림)."""
    import tempfile

    group = list(group)
    group_names = [g.name for g in group]
    if (len({input_key(g) for g in group}) != 1 or input_key(group[0]) is None
            or any(g.kind != "tree" for g in group)):
        raise ValueError("확인 묶음은 묶음 실행 트리 판들(같은 입력)이어야 함")

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
    prev_map = accepted_map(out_dir)

    def valid_old(name: str) -> bool:
        """지금 단계에서 끝난 판으로 치는 결과: 이전 코드(old)로 끝났거나, 그 전에 인정된 코드로 끝나 인정 목록에 든 판."""
        p = out_dir / name / "done.json"
        if not p.exists():
            return False
        c = read_json(p)["code_md5"]
        return c == old or name in prev_map.get(c, ())

    olddone = [n for n in group_names if valid_old(n)]
    if check not in olddone:
        raise ResumeError(f"확인 판 {check} 이 이전 코드({old}) 또는 인정된 그 전 코드로 끝나 있지 않음")
    with tempfile.TemporaryDirectory(prefix="lst_verify_") as td:
        verify_fn(Path(td), list(group_names))
        diffs = {n: compare_variant_files(out_dir / n, Path(td) / n) for n in olddone}
    rec = {"old_code_md5": old, "new_code_md5": cur, "reason": acc["reason"], "check_variant": check,
           "compared": olddone, "compared_code_md5": {n: read_json(out_dir / n / "done.json")["code_md5"] for n in olddone},
           "files": list(COMPARE_FILES)}
    if any(diffs.values()):
        exp_log.append("code_change_different_results", stage=stage, diffs=diffs, **rec)
        raise ResumeError(f"새 코드 결과가 이전 코드와 다름 — 인정 안 함: {diffs}")
    accepted, not_accepted = [], []
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
    # 그 전에 인정된 코드로 끝난 판도 이번 인정 범위(같은 규칙 `acceptable`, 이번 확인 묶음)에 들어야 계속 인정 —
    # 밖이면 그 기록의 인정 목록에서 빼(run_group 에서 '_superseded' 로 옮기고 새 코드로 다시).
    dropped = {}
    for code, ent in prev.items():
        keep, drop = [], []
        for n in ent["accepted_variants"]:
            sp_ = out_dir / n / "spec.json"
            ok = sp_.exists() and acceptable(VariantSpec(**read_json(sp_)), group)
            (keep if ok else drop).append(n)
        if drop:
            ent["accepted_variants"] = keep
            ent["not_accepted_variants"] = list(ent.get("not_accepted_variants", [])) + drop
            ent.setdefault("dropped_by", []).append({"new_code_md5": cur, "variants": drop})
            dropped[code] = drop
    rec["prev_dropped"] = dropped
    os.replace(snap, out_dir / f"_code_prev_{old}")
    snapshot_code(code_root, out_dir)
    prev[old] = {"reason": acc["reason"], "check_variant": check, "compared": olddone,
                 "accepted_variants": accepted, "not_accepted_variants": not_accepted,
                 "accepted_rule": rec["accepted_rule"],
                 "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    cf.dump_json(snap / ACCEPTED_FILE, prev)
    exp_log.append("code_change_same_results", stage=stage, **rec)
    return "accepted"


# ================================================================ 여러 판(하위 프로세스)
def _rc_reason(rc: int) -> str:
    if rc < 0:
        return f"신호 {-rc}로 강제 종료(9 = SIGKILL — 메모리 부족 강제 종료 가능)"
    if rc == 137:
        return "137(SIGKILL — 메모리 부족 강제 종료 가능)"
    return f"종료 코드 {rc}(failed.json 참고)"


def run_variants_isolated(script, groups: Sequence[Sequence[str] | str], base_argv: Sequence[str],
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


def _marks(spec: Mapping, done: Mapping | None = None) -> list[str]:
    """표시 칸: 1차·단계 2 준비 CatBoost 판 = '학습률 = 학습마다 자동값'(부록 A-13). ③·④ 는 그 판 스크립트가 extra 로.
    2차 사슬 부스팅 판에서 교차 N·공통 N 이 마지막 시점(최대 반복)이면 '상한'(§3)."""
    auto = spec["order"] == "1차" or spec["stage"] in AUTO_LR_STAGES or bool(spec.get("extra", {}).get("cat_auto_lr"))
    out = [ev.CAT_AUTO_LR_MARK] if spec["algorithm"] == "CAT" and auto else []
    if done is not None and (done.get("n_iter") or {}).get("at_cap"):
        out.append(CAP_MARK)
    return out


def stage_cv_rows(out_dir, names: Sequence[str]) -> list[dict]:
    """보고 전(test 없음) 단계 표 줄."""
    rows = []
    for n in names:
        spec, done, cv = load_done(out_dir, n)
        rows.append(ev.table_row(n, VariantSpec(**spec).describe(), cv, None,
                                 selection_count=done["selection_count"], marks=_marks(spec, done)))
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
    thr_preds: Mapping[str, pd.DataFrame] | None = None   # 2차 사슬 부스팅 판: threshold 용 공통 N fold 예측(§1-2)


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
        cv = ev.score_cv(b.cv_preds, samples, thr_preds=b.thr_preds)
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
        thr = float(cv["threshold"]["threshold"])
        full = not done.get("smoke")
        hm = pd.concat([ev.h_metrics_cv(load_preds(vdir, ev.CV_FOLDS), samples, thr, require_complete=full),
                        ev.h_metrics_test(pd.read_parquet(pred_path(vdir, "refit")), samples, truth, thr, inc,
                                          require_complete=full)], ignore_index=True)
        hp = write_h_metrics(vdir, hm)   # 이전 코드로 끝난 판(표 없음)도 저장 예측으로 만듦
        if tracker is not None and done.get("mlflow_run_id"):
            tracker.log_artifact(done["mlflow_run_id"], hp)
        tests.append(t)
        rows.append(ev.table_row(n, VariantSpec(**spec).describe(), cv, t, selection_count=done["selection_count"],
                                 marks=_marks(spec, done), fold_recall=fold_recall))
    notes = ev.table_notes(tests, extra_notes)
    p = ev.write_table(rows, notes, out_dir / table_name)
    exp_log.append("test_reported", stage=stage, names=list(names) + [b.name for b in base_rows],
                   newly_scored=scored, permit_reason=permit.reason)
    if tracker is not None and stage_run_id is not None:
        tracker.log_artifact(stage_run_id, p, "tables")
        tracker.log_artifact(stage_run_id, p.with_suffix(".notes.json"), "tables")
    return p
