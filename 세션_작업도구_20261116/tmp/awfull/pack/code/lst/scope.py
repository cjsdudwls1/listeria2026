"""실행 범위 — '국내만 버전'(사용자 결정 2026-11-09: 같은 실험을 국내 행(src_route == 'domestic')만으로 한 벌 더).

기본(설정에 `scope` 칸이 없음) = 지금 판(국내 + 해외) 그대로. 이 모듈의 함수는 기본일 때 지금 값을 그대로 돌려주고,
다른 모듈은 기본일 때 예전과 같은 길로만 감(바이트 동일 — tests/test_scope_domestic.py).

`scope: "domestic"`(configs_dom/ 설정 — scripts/make_configs_dom.py 가 configs/ 에서 만듦) 일 때:
  - 행: 단계 0 국내 준비(scripts/s0_dom_prep.py)가 시료 표를 국내 행만으로 만들고, 그 뒤 모든 단계(학습 행·평가 쌍·
    이력·타겟 인코딩 모집단·7-a·CV·test 채점·threshold)는 그 표만 씀 — 행을 고르는 코드는 따로 없음.
  - 칼럼: 해외 정보만 담는 칼럼을 뺌 — 트리 범주 `src_route`(범주 5 → 4), 딥러닝 x_static `src_route01`·② 의
    `src_route01`, '해외 구조적 결측' `ov_struct_na`(구조적 결측 대상 칼럼·채널은 국내 행 규칙 그대로 칼럼별 결측 표시).
    그 밖 칼럼은 지금 판과 같음(우연히 한 값뿐인 칼럼도 그대로 둠).
  - MLflow: 같은 서버, experiment 이름 = 논리 이름 + '-국내'(listeria-준비-국내·listeria-1차-국내·listeria-2차-국내).
    코드 안 이름(설정 experiment 칸·ORDERS·chain_run 등)은 논리 이름 그대로 두고 Tracker 가 실제 이름으로 바꿈 —
    그래서 국내판 run 은 지금 판 experiment 에 절대 안 들어감(찾기·만들기 모두 실제 이름의 experiment 안에서만).
  - 결과 뿌리 표지: --inputs 뿌리의 `SCOPE.json`({"scope": "domestic"} — s0_dom_prep 가 만듦)과 설정 scope 가
    같아야 돎(국내판 설정 → 지금 판 뿌리, 지금 판 설정 → 국내판 뿌리 모두 멈춤). 국내판의 --out 은 --inputs 뿌리
    안(심볼릭 링크를 따라간 실제 경로 기준 — 공유하는 지금 판 단계 0 폴더에는 못 씀). 실행 인자는 paths.parse_paths·
    스크립트 앞 인자 읽기(_pre)가 note_roots 로 받아 두고, 설정을 읽어 범위를 정하는 순간(activate) 대조함.

한 프로세스 안에서 범위가 바뀌면 멈춤(국내판 설정이 지금 판 설정을 가리키는 등).
구현 세부(계획서에 없음 — 담당자 확인): 위 전부(국내만 버전은 계획서 v78 밖의 추가 실험).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import config as cf

DOMESTIC = "domestic"
SCOPES = (DOMESTIC,)
MARKER = "SCOPE.json"
EXPERIMENT_SUFFIX = {DOMESTIC: "-국내"}
ROUTE_COLUMN = "src_route"              # 트리 범주 5 중 해외 정보 칼럼(§2-3 — 국내만이면 늘 'domestic')
FIELD = cf.Field(str, choices=SCOPES, required=False)


class ScopeError(RuntimeError):
    pass


_STATE: dict[str, Any] = {"set": False, "scope": None, "roots": []}


def with_field(schema: Mapping[str, Any]) -> dict:
    """설정 스키마에 선택 칸 `scope` 를 더함(없으면 지금 판)."""
    if "scope" in schema:
        raise ScopeError("스키마에 scope 칸이 이미 있음")
    return {**schema, "scope": FIELD}


def activate(scope: str | None) -> str | None:
    """이 프로세스의 범위를 정함(설정 scope 칸 값, 없으면 None = 지금 판). 이미 다른 범위로 정했으면 멈춤.
    실행 인자에서 받아 둔 결과 뿌리(note_roots)가 있으면 바로 표지·--out 대조(쓰기 전에 멈추게)."""
    if scope is not None and scope not in SCOPES:
        raise ScopeError(f"모르는 scope: {scope!r}")
    if _STATE["set"] and _STATE["scope"] != scope:
        raise ScopeError(f"한 프로세스에서 범위가 바뀜: {_STATE['scope']!r} → {scope!r}")
    _STATE["set"] = True
    _STATE["scope"] = scope
    for inputs_root, out in _STATE["roots"]:
        check_roots(inputs_root, out)
    return scope


def note_roots(inputs_root, out=None) -> None:
    """실행 인자의 --inputs(결과 뿌리)·--out 을 받아 둠(paths.parse_paths·스크립트 앞 인자 읽기에서). 범위를 이미
    정했으면 바로 대조, 아니면 activate 때 대조 — 어느 쪽이 먼저든 설정을 읽은 직후 쓰기 전에 멈춤."""
    if inputs_root is None:
        return
    r = (Path(inputs_root).expanduser().resolve(), None if out is None else Path(out).expanduser().resolve())
    _STATE["roots"].append(r)
    if _STATE["set"]:
        check_roots(*r)


def current() -> str | None:
    return _STATE["scope"]


def domestic_only() -> bool:
    return _STATE["scope"] == DOMESTIC


def experiment_name(logical: str) -> str:
    """MLflow experiment 실제 이름(기본 = 논리 이름 그대로)."""
    s = _STATE["scope"]
    return logical if s is None else logical + EXPERIMENT_SUFFIX[s]


def read_marker(inputs_root) -> str | None:
    p = Path(inputs_root) / MARKER
    if not p.exists():
        return None
    obj = cf.load_json_strict(p)
    if not isinstance(obj, Mapping) or obj.get("scope") not in SCOPES:
        raise ScopeError(f"{p} 의 scope 값이 이상함")
    return obj["scope"]


def _under(child, parent) -> bool:
    c, p = Path(child).resolve(), Path(parent).resolve()
    return c == p or p in c.parents


def check_roots(inputs_root, out=None) -> None:
    """결과 뿌리 표지 ↔ 이 프로세스 범위 대조, 국내판이면 --out 이 --inputs 뿌리 안인지."""
    mk = read_marker(inputs_root)
    if mk != _STATE["scope"]:
        raise ScopeError(f"결과 뿌리 표지({MARKER} = {mk!r})와 설정 scope({_STATE['scope']!r})가 다름: {inputs_root}")
    if _STATE["scope"] is not None and out is not None and not _under(out, inputs_root):
        raise ScopeError(f"국내판 --out 은 --inputs 뿌리 안이어야 함(공유 단계 0 폴더 포함 바깥 금지): {out}")


def apply(cfg: Mapping, inputs_root=None, out=None) -> str | None:
    """설정을 읽은 직후: (--inputs 뿌리·--out 을 받아 두고) 범위를 정함(scope 칸) — 표지·--out 대조 포함."""
    if inputs_root is not None:
        note_roots(inputs_root, out)
    return activate(cfg.get("scope"))


def write_marker(inputs_root, scope: str) -> Path:
    """s0_dom_prep 만 씀. 있으면 값이 같은지만 확인."""
    if scope not in SCOPES:
        raise ScopeError(f"모르는 scope: {scope!r}")
    p = Path(inputs_root) / MARKER
    if p.exists():
        if read_marker(inputs_root) != scope:
            raise ScopeError(f"{p} 가 다른 범위")
        return p
    os.makedirs(p.parent, exist_ok=True)
    cf.dump_json(p, {"scope": scope, "note": "국내만 버전 결과 뿌리 — 지금 판(국내 + 해외) 설정으로 돌리면 멈춤"})
    return p


def _reset_for_tests() -> None:
    _STATE["set"] = False
    _STATE["scope"] = None
    _STATE["roots"] = []
