"""경로 규칙 — 공통 규칙 '코드 위치·경로', 계획서 부록 B-2.

- 자료 뿌리 `--root`(필수)·결과 뿌리 `--out`(필수) 실행 인자.
- 코드·설정 안의 경로는 뿌리 아래 상대 경로만(Windows 절대 경로 금지).
  계획서의 경로 표기(`리스테리아ML\\data\\...`)는 역슬래시라 '/' 로 바꿔 해석.
- 스크립트 맨 앞에서 경로를 덮는 환경변수(PYTHONPATH 제외) unset.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence

# 구현 세부(계획서에 없음): '경로를 덮는 환경변수' 의 범위를 계획서가 정하지 않아
# 이 패키지 접두어 LST_ 로 시작하는 변수 + MLflow 경로 변수 + 호출자가 넘긴 이름으로 정함.
PATH_ENV_PREFIX = "LST_"
PATH_ENV_NAMES = ("MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACT_URI", "MLFLOW_REGISTRY_URI")
KEEP_ENV = ("PYTHONPATH",)


def unset_path_env(extra_names: Iterable[str] = ()) -> list[str]:
    """경로를 덮는 환경변수를 지우고 지운 이름 목록을 돌려줌(B-2)."""
    extra = set(PATH_ENV_NAMES) | set(extra_names)
    removed = []
    for name in sorted(os.environ):
        if name in KEEP_ENV:
            continue
        if name.startswith(PATH_ENV_PREFIX) or name in extra:
            removed.append(name)
            del os.environ[name]
    return removed


def is_absolute_like(p: str | os.PathLike) -> bool:
    """절대 경로·드라이브 경로·홈(~) 경로이면 True(B-2: 금지 대상)."""
    s = str(p)
    w = PureWindowsPath(s)
    return bool(w.drive) or w.is_absolute() or s.startswith(("/", "\\", "~"))


def to_rel(p: str | os.PathLike) -> PurePosixPath:
    """뿌리 아래 상대 경로로 바꿈. 절대 경로·'..' 는 오류."""
    if is_absolute_like(p):
        raise ValueError(f"절대 경로 금지(부록 B-2): {p!r}")
    rel = PurePosixPath(str(p).replace("\\", "/"))
    if ".." in rel.parts:
        raise ValueError(f"뿌리 밖으로 나가는 경로 금지: {p!r}")
    if str(rel) in ("", "."):
        raise ValueError("빈 경로")
    return rel


def resolve(root: str | os.PathLike, rel: str | os.PathLike) -> Path:
    """뿌리 + 상대 경로."""
    return Path(root) / Path(*to_rel(rel).parts)


@dataclass(frozen=True)
class RunPaths:
    root: Path  # 자료 뿌리(--root)
    out: Path   # 결과 뿌리(--out)

    def data(self, rel: str | os.PathLike) -> Path:
        return resolve(self.root, rel)

    def result(self, rel: str | os.PathLike) -> Path:
        return resolve(self.out, rel)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", required=True, help="자료 뿌리(로컬 '천영진' 폴더 또는 서버 data)")
    ap.add_argument("--out", required=True, help="결과 뿌리")
    return ap


def parse_paths(argv: Sequence[str] | None = None,
                parser: argparse.ArgumentParser | None = None,
                extra_env: Iterable[str] = ()) -> tuple[RunPaths, argparse.Namespace]:
    """env unset → 인자 읽기 → 뿌리 확인. 결과 뿌리는 없으면 만듦."""
    unset_path_env(extra_env)
    ap = parser or build_parser()
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--root 폴더 없음: {root}")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, out=out), args
