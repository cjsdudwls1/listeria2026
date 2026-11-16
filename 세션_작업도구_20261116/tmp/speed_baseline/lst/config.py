"""엄격한 JSON 설정·md5·설정 덤프 — 계획서 부록 B-2, §8.

- 모르는 칸이면 오류로 멈춤, 같은 칸 이름 두 번이면 멈춤(옵션 이름 겸용 금지).
- 계획서에 없는 칼럼이 설정에 있으면 멈춤(`check_columns`).
- 결과마다 code_md5·입력 md5·스레드 수·설정 덤프 저장(`save_run_meta`).
- 기본값을 채워 넣지 않음(설정에 없는 값을 코드가 만들지 않게) — 선택 칸은
  없으면 그냥 없음.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import is_absolute_like, resolve

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- JSON 읽기
def _no_duplicate_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ConfigError(f"같은 칸 이름이 두 번: {k!r}")
        d[k] = v
    return d


def _reject_constant(name):
    raise ConfigError(f"JSON 에 {name} 금지")


def load_json_strict(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_no_duplicate_pairs, parse_constant=_reject_constant)


# ---------------------------------------------------------------- 스키마
@dataclass(frozen=True)
class Field:
    """설정 칸 하나. types = 허용 타입(bool 은 int 로 안 봄)."""
    types: type | tuple[type, ...]
    required: bool = True
    choices: tuple | None = None
    item_types: type | tuple[type, ...] | None = None  # list 칸의 원소 타입


def _type_ok(v, types) -> bool:
    types = types if isinstance(types, tuple) else (types,)
    if isinstance(v, bool) and bool not in types:
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return isinstance(v, types)


def validate(cfg: Mapping, schema: Mapping[str, Any], where: str = "") -> Mapping:
    """schema: 칸 이름 → Field 또는 하위 schema(dict)."""
    if not isinstance(cfg, Mapping):
        raise ConfigError(f"{where or '설정'} 이 객체가 아님")
    unknown = [k for k in cfg if k not in schema]
    if unknown:
        raise ConfigError(f"모르는 칸(부록 B-2): {[where + k for k in unknown]}")
    for name, spec in schema.items():
        key = where + name
        if name not in cfg:
            req = spec.required if isinstance(spec, Field) else True
            if req:
                raise ConfigError(f"필수 칸 없음: {key}")
            continue
        v = cfg[name]
        if isinstance(spec, Mapping):
            validate(v, spec, where=key + ".")
            continue
        if not _type_ok(v, spec.types):
            raise ConfigError(f"타입 오류: {key} = {v!r}")
        if spec.choices is not None and v not in spec.choices:
            raise ConfigError(f"허용 값 아님: {key} = {v!r} (허용 {spec.choices})")
        if spec.item_types is not None:
            for i, item in enumerate(v):
                if not _type_ok(item, spec.item_types):
                    raise ConfigError(f"원소 타입 오류: {key}[{i}] = {item!r}")
    assert_no_absolute_paths(cfg)
    return cfg


def assert_no_absolute_paths(obj: Any, where: str = "") -> None:
    """설정 안 문자열에 절대 경로가 있으면 멈춤(B-2)."""
    if isinstance(obj, str):
        if is_absolute_like(obj):
            raise ConfigError(f"절대 경로 금지(부록 B-2): {where} = {obj!r}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            assert_no_absolute_paths(v, f"{where}.{k}" if where else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_absolute_paths(v, f"{where}[{i}]")


def load_config(path: str | os.PathLike, schema: Mapping[str, Any]) -> Mapping:
    return validate(load_json_strict(path), schema)


def check_columns(columns: Iterable[str], allowed: Iterable[str], what: str = "칼럼") -> None:
    """계획서에 없는 칼럼이면 멈춤(B-2). allowed = 계획서에서 옮긴 칼럼 목록."""
    allowed = set(allowed)
    extra = [c for c in columns if c not in allowed]
    if extra:
        raise ConfigError(f"계획에 없는 {what}(부록 B-2): {extra}")


# ---------------------------------------------------------------- md5
def file_md5(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


CODE_DIRS = ("lst", "scripts")


def code_md5(code_root: str | os.PathLike, dirs: Iterable[str] = CODE_DIRS) -> str:
    """code_root 아래 dirs 의 .py 전부(상대 경로 정렬 순)로 md5 하나.

    구현 세부(계획서에 없음): 줄바꿈을 LF 로 맞춘 뒤 해시(로컬 Windows CRLF 사본과
    서버 LF 사본이 같은 값이 되게), 파일마다 '상대 경로 NUL 내용 md5 줄바꿈' 을 이어 해시.
    """
    root = Path(code_root)
    files = []
    for d in dirs:
        base = root / d
        if base.is_dir():
            files += [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.md5()
    for p in files:
        content = p.read_bytes().replace(b"\r\n", b"\n")
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.md5(content).hexdigest().encode("ascii") + b"\n")
    return h.hexdigest()


def inputs_md5(root: str | os.PathLike, rel_paths: Iterable[str]) -> dict[str, str]:
    """입력 파일(뿌리 아래 상대 경로) → md5."""
    return {str(r): file_md5(resolve(root, r)) for r in rel_paths}


def thread_info() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in THREAD_ENV_VARS}


# ---------------------------------------------------------------- 저장
def dump_json(path: str | os.PathLike, obj: Any, overwrite: bool = False) -> Path:
    """JSON 저장(칸 순서 유지). 기존 파일은 덮지 않음(공통 규칙)."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"덮어쓰기 금지: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, p)
    return p


def nan_to_null(obj: Any) -> Any:
    """점수 저장용: float NaN(numpy 포함) → None(JSON null). inf 는 그대로 두어 dump 에서 멈춤.
    numpy 배열·정수·실수는 파이썬 값으로. 읽기 규칙(load_json_strict)은 그대로."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [nan_to_null(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, Mapping):
        return {k: nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_null(v) for v in obj]
    return obj


def dump_scores_json(path: str | os.PathLike, obj: Any, overwrite: bool = False) -> Path:
    """점수(h 별 NaN 포함) 저장: NaN → null 로 바꾼 뒤 dump_json."""
    return dump_json(path, nan_to_null(obj), overwrite=overwrite)


def run_meta(config: Any, input_md5s: Mapping[str, str], code_root: str | os.PathLike,
             n_jobs: int | None = None, extra: Mapping | None = None) -> dict:
    """B-2 결과 메타: code_md5·입력 md5·스레드 수·설정 덤프."""
    meta = {
        "code_md5": code_md5(code_root),
        "inputs_md5": dict(input_md5s),
        "threads": {**thread_info(), "n_jobs": n_jobs},
        "config": config,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        overlap = set(extra) & set(meta)
        if overlap:
            raise ConfigError(f"메타 칸 겹침: {sorted(overlap)}")
        meta.update(extra)
    return meta


def save_run_meta(path: str | os.PathLike, config: Any, input_md5s: Mapping[str, str],
                  code_root: str | os.PathLike, n_jobs: int | None = None,
                  extra: Mapping | None = None) -> Path:
    return dump_json(path, run_meta(config, input_md5s, code_root, n_jobs, extra))
