"""그룹·단위·칼럼 섞기 중요도(permutation importance) — 계획서 §6 단계 3-0b·2-R1, 부록 A-5(빼기 단위·그룹·중요도
계산), 부록 A-10(최종 모델 안 순열 중요도), 부록 B-5(섞기 씨앗·`h_block`), 부록 B-1(평가 쌍은 h 마다 메모리에서).

계산(부록 A-5):
  fold 모델(재학습 없음)과 그 fold validation 쌍으로, 대상(그룹·단위·칼럼·묶음)마다 r = 1~M(M = 10) 번
  대상 칼럼·채널 전부를 같은 행 순서로 섞음(`h_block` = (h, src_route) 같은 행끼리만) →
  I = 섞기 전 − 섞은 뒤 구간 가중 PR-AUC(fold validation 쌍, metrics 의 trapz PR-AUC 하나),
  ψ_f = fold f 의 M 회 중 I > 0 비율, π = 3M 칸 중 I > 0 비율, 판정은 반올림 없이.
  3-0b 건너뛰기 = 3 fold 모두 ψ_f ≥ 0.8 이고 3M 칸 평균 I > 0. 나머지 빼기 순서 = π 낮은 순 → 평균 I 낮은 순 → 번호.
  딥러닝·결합 모델 = 그 대상의 x_seq 채널(시퀀스 통째)·x_static 칼럼·결측 표시 칼럼·채널을 같은 순열로 행끼리 바꿈.
  여러 구성 모델(스태킹의 트리·딥러닝 확률·이상 점수 모델, 앙상블·씨앗 평균 구성)에 같은 칼럼·채널이 있으면 모두 같은
  순열로 섞고 구성 모델을 섞인 입력으로 다시 추론(학습 없음) — 이상 점수의 robust z 기준값은 모델에 저장된 값 그대로,
  딥러닝 확률·이상 점수 칼럼 자체는 섞는 대상 아님(등록부에 단위가 없음).

공개 API(0-9 스크립트·3-0b·2-R1·최종 보고가 씀):
  대상:   group_target(g) · unit_target(u) · targets_groups(units) · targets_units(units) · targets_multi_unit(units)
          targets_2r1() · targets_final(units) · Target.rng(fold_no, r)
  모델 그래프(fold 마다 하나):
          Leaf(name, assembler, model, *, n_iter=None, output='proba'|'score', layout=None, predict_fn=None)
          Stack(name, base: Leaf, feeders: {칼럼 이름: Node}, combine=None)       # 트리 + 딥러닝 확률·이상 점수 칼럼
          Mean(name, members)                                                     # 앙상블·씨앗 평균
  계산:   permutation_importance(models: {'fold1'|'fold2'|'fold3': Node}, samples: evaluate.SampleTable,
                                 targets, *, M=10, splits=None, r_batch=None, check=None, expect_base=None)
          → ImportanceResult(records, summary, base, seconds, …) — .removal_order() · .save(out_dir)
  나눠 돌리기(구현 세부 h): importance_part(models, samples, targets, *, fold, h_from, h_to, M, state_dir, …)
          · pending_h(samples, fold, h_from, h_to, state_dir) · state_targets(state_dir) · target_from_dict(d)
          · merge_parts(samples, targets, *, state_dir, M, expect_base, …) → ImportanceResult(위와 같은 꼴)
  Leaf 의 layout 은 assembler 가 inputs.TreeInputs·inputs.DLInputs 면 자동(등록부 칼럼 소속), 그 밖(표 AE·VAE x_tab 등)은
  layout_from_names(...) 로 넘김.

구현 세부(계획서에 없음 — 담당자 확인):
 (a) 섞는 법: 칸((h, src_route), h 오름차순 → src_route 문자열 순) 마다 칸 안 행을 lims_data_key 문자열 순으로 세우고
     p = rng.permutation(칸 크기) → 섞은 뒤 i 번째 행 = 섞기 전 p[i] 번째 행의 값(대상 칼럼·채널만). rng 하나 =
     (fold, 대상, r) 하나 — h 1 → 60 차례로 이어 씀(빈 칸은 건너뜀).
 (b) 구간 가중 PR-AUC 는 h 별 PR-AUC(그 h 쌍만, u_tie 로 동점 풀기)를 metrics.band_weighted 로 — metrics.pr_auc_band_weighted
     와 같은 값(시험으로 확인). 평가 쌍 = folds.iter_scoring_pairs_by_h, 행 표 = 시료 표 + h + runner.add_bounds(W 없이 —
     S 는 각 조립 객체가 자기 W 로 계산).
 (c) 섞은 입력은 M 회를 이어 붙여 한 번에 추론(r_batch 로 나눌 수 있음 — 결과 같음). 대상 칼럼이 없는 구성 모델은
     다시 추론하지 않고 섞기 전 예측을 씀(같은 값).
 (d) '해외 구조적 결측'(ov_struct_na)·범주 5·h·g·g ÷ 50·src_route 0/1·딥러닝 확률·이상 점수 칼럼은 어느 대상에도 안 묶임.
     결측 표시 '<칼럼>__na' 는 원 칼럼의 단위. 칼럼 대상(리서치 칼럼)은 같은 이름 칼럼 + 그 결측 표시만.
 (e) 대상이 모델 입력에 하나도 없으면 멈춤(3-0b 에서 빠진 그룹을 실수로 넣는 것 방지). 최종 모델 대상은 targets_final 이
     입력에 있는 것만 만듦.
 (f) Stack 의 기본 결합 = 바탕 조립 객체 출력 뒤에 feeders 칼럼을 그 순서대로 붙임(float64). 로지스틱 바탕처럼 붙인 칼럼도
     표준화해야 하면 combine(X, {칼럼: 값}) 을 넘김.
 (g) expect_base 를 주면 섞기 전 점수가 그 값(판의 fold 점수)과 base_tol 안인지 확인(다르면 멈춤).
 (h) 나눠 돌리기·이어 하기(importance_part·merge_parts — 3-0b 설정 importance_parts): 일 단위 = (fold, h 구간). 부분은
     fold 시작 때와 같은 rng(대상, r)를 만들고, 구간 앞 h 는 예측 없이 칸 크기만으로 rng.permutation 을 같은 차례로
     불러 난수 상태를 맞춘 뒤(재생) 구간 h 를 계산 — 한 번에 돌린 것과 같은 순열·같은 점수. h 하나가 끝날 때마다
     <state_dir>/<fold>/hNN.npz(섞기 전 bh·섞은 뒤 ph(대상 × r)·쌍 수·초·그 h 뒤 rng 상태 md5·설정 서명)를 원자적으로
     씀. 다시 돌리면 있는 h 는 계산 없이 재생만(rng 상태 md5 대조) → 다음 h 부터. 합치기 = 파일의 bh·ph 로
     permutation_importance 와 같은 records·summary·meta(seconds 만 = h 계산 초의 합). 설정 서명 = perm_seed·fold·
     fold 범위·M·대상·시료(키·d·src_route)·tag(부른 쪽 — 판·코드 md5 등)의 md5, 다르면 멈춤.
 (i) 속도 개선(결과 불변 — 사용자 승인 11-12, 설정 speed.gpu_importance — lst/speed, 꺼져 있으면 예전 길 그대로):
     딥러닝 잎(DLClassifier 확률·AnomalyModel 이상 점수, predict_fn 없음)은 h 마다 그 h 의 입력(그 모델이 쓰는 키)을 모델
     장치(GPU)에 한 번 올리고, 섞은 입력을 R 회 이어 붙여 CPU 에 만드는 대신 예측 묶음(8192 행, 같은 경계)마다 GPU 색인으로
     만듦: 묶음 행 = 원 입력[base_take 구간], 대상 칼럼·채널만 원 입력[sel_take 구간] 값으로(행 사본 — 값 같음). 섞기
     순열(rng·칸·차례)·base_take·sel_take·묶음 순서·크기는 예전과 같음. 입력 점검(NaN·dtype·모양·범주 번호 범위)은 원 입력으로
     한 번(섞은 행은 원 행의 사본). GPU 에 못 올리면(메모리 오류) 그 잎은 예전 길(CPU 이어 붙이기). 트리·그 밖 잎은 CPU 그대로.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import folds as fd
from . import metrics as mt
from . import registry as rg
from .hassign import H_MAX, u_tie_many

PERM_SEED = 2026            # §8 perm_seed
M_DEFAULT = 10              # 부록 A-5 섞기 횟수
PSI_MIN = 0.8               # 부록 A-5 ψ_f 기준
SEED_GROUP = 1000           # 부록 B-5: 3-0b 그룹·2-R1 기상 그룹 = 1000 + 그룹 번호
SEED_UNIT = 5000            # 단위 = 5000 + 단위 번호
SEED_2R1 = 9000             # 2-R1 칼럼 = 9000 + 칼럼 번호, 리서치 묶음 = 9000
SEED_FINAL_COL = 20000      # 최종 모델: 칼럼 [.., 20000, 칼럼 번호, r]·묶음 [.., 20000, 0, r]
SEED_FINAL_GROUP = 21000    # 최종 모델: 그룹 [.., 21000, 그룹 번호, r]
KINDS = ("group", "unit", "column", "bundle")
RESEARCH_COLUMNS = tuple(rg.RESEARCH_UNITS)   # §6 2-R 나열 순 = 칼럼 번호 1~9(축사 = 9)
RESEARCH_BUNDLE_NAME = "리서치 변수 전체"


class ImportanceError(ValueError):
    pass


def fold_number(name: str) -> int:
    if name not in fd.CV_FOLDS:
        raise ImportanceError(f"fold 이름은 {fd.CV_FOLDS}: {name!r}")
    return int(name[-1])


# ================================================================ 대상
@dataclass(frozen=True)
class Target:
    """섞는 대상 하나. kind = group·unit(단위 소속으로 고름)·bundle(단위 묶음)·column(같은 이름 칼럼만).
    seed_key = 부록 B-5 씨앗에서 perm_seed·f 뒤, r 앞에 오는 원소들."""
    kind: str
    no: int
    name: str
    units: frozenset = frozenset()
    columns: frozenset = frozenset()
    seed_key: tuple = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ImportanceError(f"대상 종류: {self.kind}")
        if self.kind == "column" and not self.columns:
            raise ImportanceError("칼럼 대상에 칼럼 이름 없음")
        if self.kind != "column" and not self.units:
            raise ImportanceError(f"{self.kind} 대상에 단위 없음")
        if not self.seed_key:
            raise ImportanceError("seed_key 없음")

    @property
    def key(self) -> tuple:
        return (self.kind, self.no, self.seed_key)

    def rng(self, fold_no: int, r: int) -> np.random.Generator:
        """부록 B-5: default_rng([perm_seed, f, *seed_key, r]), r = 1~M."""
        return np.random.default_rng([PERM_SEED, int(fold_no), *[int(x) for x in self.seed_key], int(r)])

    def matches(self, base: str, unit: str | None) -> bool:
        if self.kind == "column":
            return base in self.columns
        return unit is not None and unit in self.units


def group_target(g, seed_base: int = SEED_GROUP) -> Target:
    gr = rg.group(g)
    return Target("group", gr.no, gr.name, units=frozenset(gr.units), seed_key=(seed_base + gr.no,))


def unit_target(u) -> Target:
    un = rg.unit(u)
    return Target("unit", un.no, un.name, units=frozenset({un.name}), seed_key=(SEED_UNIT + un.no,))


def _unit_names(units: Iterable) -> list[str]:
    out = []
    for u in units:
        out.append(u.name if isinstance(u, rg.Unit) else rg.unit(u).name)
    return out


def targets_groups(units: Iterable) -> list[Target]:
    """3-0b 그룹 대상: 단위 목록(판의 남은 단위 — VersionSpec.units() 또는 이름)에 단위가 하나라도 있는 그룹(번호 순).
    그룹 칼럼 = 그 그룹 단위 중 모델 입력에 있는 것(빠진 단위는 입력에 없으니 자동으로 안 섞임)."""
    have = set(_unit_names(units))
    return [group_target(g.no) for g in rg.GROUPS if have & set(g.units)]


def targets_units(units: Iterable) -> list[Target]:
    return [unit_target(n) for n in sorted(set(_unit_names(units)), key=lambda n: rg.unit(n).no)]


def targets_multi_unit(units: Iterable) -> list[Target]:
    """3-0b 단위 빼기 대상: 남은 단위가 2개 이상인 그룹의 단위 전부(단위 번호 순). '그룹 마지막 단위는 안 뺌' 은
    빼기 차례를 도는 쪽이 판정."""
    have = set(_unit_names(units))
    out = []
    for g in rg.GROUPS:
        left = [u for u in g.units if u in have]
        if len(left) >= 2:
            out += [unit_target(u) for u in left]
    return sorted(out, key=lambda t: t.no)


def research_column_target(name: str, seed_base: int = SEED_2R1, final: bool = False) -> Target:
    if name not in RESEARCH_COLUMNS:
        raise ImportanceError(f"리서치 칼럼 아님: {name}")
    k = RESEARCH_COLUMNS.index(name) + 1
    key = (SEED_FINAL_COL, k) if final else (seed_base + k,)
    return Target("column", k, name, columns=frozenset({name}), seed_key=key)


def research_bundle_target(final: bool = False) -> Target:
    key = (SEED_FINAL_COL, 0) if final else (SEED_2R1,)
    return Target("bundle", 0, RESEARCH_BUNDLE_NAME, units=frozenset(RESEARCH_COLUMNS), seed_key=key)


def targets_2r1() -> list[Target]:
    """§6 2-R1: 리서치 칼럼 9개(칼럼마다 9000 + 번호)·리서치 변수 전체(9000)·각 변수가 속한 기상 그룹(1000 + 그룹 번호 —
    3-0b 그룹과 같은 순열). 기상 그룹 칼럼은 2-R1 입력에 있는 칼럼만 섞임(모델 입력에서 고르므로 자동)."""
    out = [research_column_target(c) for c in RESEARCH_COLUMNS] + [research_bundle_target()]
    gnos = sorted({rg.unit(c).group_no for c in RESEARCH_COLUMNS if rg.group(rg.unit(c).group_no).weather})
    return out + [group_target(g) for g in gnos]


def targets_final(units_present: Iterable[str], columns_present: Iterable[str] | None = None) -> list[Target]:
    """부록 A-5 끝 줄·A-10: 리서치 칼럼 중 최종 모델 입력에 남은 것(칼럼마다 [.., 20000, 번호, r])·리서치 변수 전체 묶음
    ([.., 20000, 0, r] — 남은 리서치 칼럼이 하나라도 있을 때)·부록 A-5 그룹 중 입력에 칼럼·채널이 있는 그룹 전부
    ([.., 21000, 그룹 번호, r]). units_present = 모델 그래프의 present_units(), columns_present = 입력 칼럼 이름
    (없으면 단위로 판정 — 리서치 9칼럼은 단위 하나 = 칼럼 하나)."""
    have = set(units_present)
    cols = set(columns_present) if columns_present is not None else have
    out = [research_column_target(c, final=True) for c in RESEARCH_COLUMNS if c in cols]
    if out:
        out.append(research_bundle_target(final=True))
    out += [Target("group", g.no, g.name, units=frozenset(g.units), seed_key=(SEED_FINAL_GROUP, g.no))
            for g in rg.GROUPS if have & set(g.units)]                     # 원소 5개 씨앗(부록 B-5)
    return out


# ================================================================ 칼럼 소속(layout)
NA = rg.NA_SUFFIX


@dataclass(frozen=True)
class Slot:
    name: str              # 입력 안 이름(칼럼·채널)
    base: str              # 원 칼럼 이름(결측 표시면 원 칼럼)
    unit: str | None       # 단위 이름(후보 아님 = None)


@dataclass(frozen=True)
class Layout:
    """모델 입력의 칼럼·채널 소속. kind 'frame' = DataFrame(slots[None] = 칼럼 순), 'dict' = 배열 dict
    (slots[key] = 마지막 축 순서 — x_seq 채널·x_static·x_tab 칼럼). slots 에 없는 키(x_cat)는 안 섞음."""
    kind: str
    slots: Mapping[Any, tuple]

    def select(self, t: Target) -> dict:
        out = {}
        for k, ss in self.slots.items():
            idx = np.array([i for i, s in enumerate(ss) if t.matches(s.base, s.unit)], dtype=np.int64)
            if idx.size:
                out[k] = idx
        return out

    def units(self) -> set[str]:
        return {s.unit for ss in self.slots.values() for s in ss if s.unit is not None}

    def names(self) -> set[str]:
        return {s.base for ss in self.slots.values() for s in ss}


def _slot(name: str, lookup: Mapping[str, str | None]) -> Slot:
    if name == rg.STRUCT_COLUMN:
        return Slot(name, name, None)
    if name.endswith(NA) and name[: -len(NA)] in lookup:
        base = name[: -len(NA)]
    else:
        base = name
    return Slot(name, base, lookup.get(base))


def column_lookup(cols: Iterable[rg.Column]) -> dict[str, str | None]:
    return {c.name: c.unit for c in cols}


def layout_from_names(names, lookup: Mapping[str, str | None]) -> Layout:
    """names = 칼럼 목록(DataFrame 입력) 또는 {키: 마지막 축 이름 목록}(배열 dict 입력), lookup = 원 칼럼 → 단위.
    예: 표 AE·VAE 의 x_tab = layout_from_names({'x_tab': 칼럼 목록}, column_lookup(registry.tree_columns(spec)))."""
    if isinstance(names, Mapping):
        return Layout("dict", {k: tuple(_slot(n, lookup) for n in v) for k, v in names.items()})
    return Layout("frame", {None: tuple(_slot(n, lookup) for n in names)})


def layout_auto(assembler) -> Layout:
    from . import inputs as ip

    if isinstance(assembler, ip.TreeInputs):
        return layout_from_names(list(assembler.columns), column_lookup(rg.tree_columns(assembler.spec)))
    if isinstance(assembler, ip.DLInputs):
        prep = assembler.prep
        if prep is None:
            raise ImportanceError("DLInputs 가 fit 전")
        lk = column_lookup(list(prep.seq_cols) + list(prep.st_cols))
        return layout_from_names({"x_seq": prep.seq_channel_names, "x_static": prep.static_names}, lk)
    raise ImportanceError(f"layout 을 자동으로 못 정함({type(assembler).__name__}) — layout_from_names 로 넘길 것")


# ================================================================ 모델 그래프
class Node:
    name: str

    def leaves(self) -> list["Leaf"]:
        raise NotImplementedError

    def present_units(self) -> set[str]:
        return set().union(*[lf.get_layout().units() for lf in self.leaves()])

    def present_names(self) -> set[str]:
        return set().union(*[lf.get_layout().names() for lf in self.leaves()])

    def has(self, t: Target) -> bool:
        return any(lf.get_layout().select(t) for lf in self.leaves())


class Leaf(Node):
    """조립 객체(fit 끝남) + 모델(학습 끝남). output 'proba' = predict_proba(X[, n_iter]), 'score' = model.score(X)
    (AnomalyModel 이상 점수 robust z — 기준값은 모델에 저장된 값). predict_fn(X) 를 주면 그것으로(AutoKeras 모델 등)."""

    def __init__(self, name: str, assembler, model, *, n_iter: int | None = None, output: str = "proba",
                 layout: Layout | None = None, predict_fn: Callable[[Any], np.ndarray] | None = None):
        if output not in ("proba", "score"):
            raise ImportanceError(f"output: {output}")
        self.name, self.assembler, self.model = name, assembler, model
        self.n_iter, self.output, self._layout, self.predict_fn = n_iter, output, layout, predict_fn

    def get_layout(self) -> Layout:
        if self._layout is None:
            self._layout = layout_auto(self.assembler)
        return self._layout

    def leaves(self):
        return [self]

    def build(self, meta: pd.DataFrame):
        return self.assembler.transform(meta)

    def predict(self, X) -> np.ndarray:
        if self.predict_fn is not None:
            p = self.predict_fn(X)
        elif self.output == "score":
            p = self.model.score(X)
        elif self.n_iter is not None:
            p = self.model.predict_proba(X, n_iter=self.n_iter)
        else:
            p = self.model.predict_proba(X)
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if not np.isfinite(p).all():
            raise ImportanceError(f"{self.name}: 예측에 NaN·inf")
        return p


def append_columns(X: pd.DataFrame, extra: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """구현 세부 (f): 바탕 입력 뒤에 칼럼을 붙임(이름 겹치면 멈춤)."""
    if not isinstance(X, pd.DataFrame):
        raise ImportanceError("기본 결합은 DataFrame 바탕만 — combine 을 넘길 것")
    dup = [c for c in extra if c in X.columns]
    if dup:
        raise ImportanceError(f"붙일 칼럼이 바탕 입력에 이미 있음: {dup}")
    add = pd.DataFrame({c: np.asarray(v, dtype=np.float64) for c, v in extra.items()}, index=X.index)
    return pd.concat([X, add], axis=1)


class Stack(Node):
    """스태킹 판(3-2 (a)~(e)·3-1 이상 점수 판): 바탕 트리(Leaf — 그 조립 객체는 붙일 칼럼 없는 입력)를
    combine(바탕 입력, {칼럼: feeder 예측}) 으로 예측. feeders = 딥러닝 확률·이상 점수를 만드는 모델(Leaf·Mean 등)."""

    def __init__(self, name: str, base: Leaf, feeders: Mapping[str, Node],
                 combine: Callable[[Any, Mapping[str, np.ndarray]], Any] | None = None):
        if not isinstance(base, Leaf):
            raise ImportanceError("Stack 바탕은 Leaf")
        self.name, self.base, self.feeders = name, base, dict(feeders)
        self.combine = combine or append_columns

    def leaves(self):
        out = [self.base]
        for f in self.feeders.values():
            out += f.leaves()
        return out


class Mean(Node):
    """앙상블(3-3)·씨앗 평균(5·6): 구성 예측 단순 평균."""

    def __init__(self, name: str, members: Sequence[Node]):
        if not members:
            raise ImportanceError("Mean 구성 없음")
        self.name, self.members = name, list(members)

    def leaves(self):
        out = []
        for m in self.members:
            out += m.leaves()
        return out


# ================================================================ 섞기
class _FramePack:
    """DataFrame 입력을 가장 많은 dtype 덩어리(보통 float32) + 나머지 칼럼으로 나눠 두고 행 뽑기를 빠르게."""

    def __init__(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        dts = list(X.dtypes)
        keys = [str(d) for d in dts]
        main_key = max(set(keys), key=lambda k: (keys.count(k), k == "float32"))
        self.main_pos = [i for i, k in enumerate(keys) if k == main_key]
        self.local = {p: j for j, p in enumerate(self.main_pos)}
        self.main = X.iloc[:, self.main_pos].to_numpy(copy=True)
        self.other = [(p, X.iloc[:, p].to_numpy(), dts[p]) for p in range(len(keys)) if keys[p] != main_key]
        self.main_names = [self.columns[p] for p in self.main_pos]

    def take(self, base_take: np.ndarray, sel_pos: np.ndarray | None, sel_take: np.ndarray | None) -> pd.DataFrame:
        big = self.main[base_take]
        sel = set() if sel_pos is None else set(int(p) for p in sel_pos)
        ms = [self.local[p] for p in sorted(sel) if p in self.local]
        if ms:
            big[:, ms] = self.main[np.ix_(sel_take, ms)]
        df = pd.DataFrame(big, columns=self.main_names, copy=False)
        for p, arr, dt in self.other:
            v = arr[sel_take] if p in sel else arr[base_take]
            df.insert(p, self.columns[p], pd.Series(v, dtype=dt, copy=False))
        return df


def _take_dict(X: Mapping[str, np.ndarray], base_take, sel: Mapping, sel_take) -> dict:
    out = {}
    for k, a in X.items():
        b = a[base_take]
        if k in sel:
            idx = sel[k]
            if a.ndim == 3:
                b[:, :, idx] = a[:, :, idx][sel_take]
            elif a.ndim == 2:
                b[:, idx] = a[:, idx][sel_take]
            else:
                raise ImportanceError(f"{k}: 섞을 수 없는 차원 {a.shape}")
        out[k] = np.ascontiguousarray(b)
    return out


def _speed_on(flag: str) -> bool:
    from . import speed

    return speed.on(flag)


def _gpu_capable(leaf: "Leaf") -> bool:
    """구현 세부 (i): GPU 색인 길을 쓸 수 있는 잎(배열 dict 입력의 딥러닝 확률·이상 점수, predict_fn 없음)."""
    if leaf.predict_fn is not None or leaf.get_layout().kind != "dict":
        return False
    m = leaf.model
    if leaf.output == "score":
        return hasattr(m, "score_source") and hasattr(m, "source_keys")
    return leaf.n_iter is None and hasattr(m, "predict_proba_source") and hasattr(m, "source_keys")


class _GPUInput:
    """구현 세부 (i): h 하나·잎 하나의 입력(모델이 쓰는 키)을 모델 장치에 한 번 올려 둠."""

    def __init__(self, leaf: "Leaf", X: Mapping[str, np.ndarray]):
        import torch

        m = leaf.model
        self.n = int(m.check_source_inputs(X))
        self.dev = m.device
        self.torch = torch
        self.t = {}
        for k in m.source_keys():
            self.t[k] = torch.from_numpy(np.ascontiguousarray(X[k])).to(self.dev)

    def fetch_fn(self, sel: Mapping[str, np.ndarray], base_take: np.ndarray, sel_take: np.ndarray):
        torch = self.torch
        bt = torch.from_numpy(np.ascontiguousarray(base_take, dtype=np.int64)).to(self.dev)
        st = torch.from_numpy(np.ascontiguousarray(sel_take, dtype=np.int64)).to(self.dev)
        si = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.int64)).to(self.dev)
              for k, v in sel.items() if k in self.t}
        for k in sel:
            if k in self.t and self.t[k].dim() not in (2, 3):
                raise ImportanceError(f"{k}: 섞을 수 없는 차원 {tuple(self.t[k].shape)}")

        def fetch(s: int, e: int) -> dict:
            out = {}
            for k, a in self.t.items():
                b = a.index_select(0, bt[s:e])                  # 새 텐서(예전 .to(dev) 처럼 새 할당)
                if k in si:
                    d = a.dim() - 1
                    b.index_copy_(d, si[k], a.index_select(0, st[s:e]).index_select(d, si[k]))
                out[k] = b
            return out
        return fetch

    def predict(self, leaf: "Leaf", sel, base_take, sel_take) -> np.ndarray:
        fetch = self.fetch_fn(sel, base_take, sel_take)
        n = int(base_take.shape[0])
        if leaf.output == "score":
            p = leaf.model.score_source(n, fetch)
        else:
            p = leaf.model.predict_proba_source(n, fetch)
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if not np.isfinite(p).all():
            raise ImportanceError(f"{leaf.name}: 예측에 NaN·inf")
        return p


class _HEval:
    """h 하나의 평가 쌍 위에서 그래프 예측(섞기 전·섞은 뒤). 조립 객체 출력·섞기 전 예측은 h 마다 한 번."""

    def __init__(self, meta: pd.DataFrame):
        self.meta = meta
        self.n = len(meta)
        self._X: dict[int, Any] = {}
        self._pack: dict[int, _FramePack] = {}
        self._base: dict[int, np.ndarray] = {}
        self._gpu: dict[int, _GPUInput | None] = {}     # 구현 세부 (i) — None = 못 올림(예전 길)
        self.gpu_fallbacks: list[str] = []

    def gpu_input(self, leaf: "Leaf") -> _GPUInput | None:
        k = id(leaf)
        if k not in self._gpu:
            try:
                self._gpu[k] = _GPUInput(leaf, self.X(leaf))
            except Exception as e:  # torch.cuda.OutOfMemoryError 등 — 예전 길로(값 같음)
                if type(e).__name__ != "OutOfMemoryError":
                    raise
                self._gpu[k] = None
                self.gpu_fallbacks.append(leaf.name)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover
                    pass
        return self._gpu[k]

    def release(self) -> None:
        self._gpu.clear()

    def X(self, leaf: Leaf):
        k = id(leaf)
        if k not in self._X:
            X = leaf.build(self.meta)
            lay = leaf.get_layout()
            if lay.kind == "frame":
                if not isinstance(X, pd.DataFrame):
                    raise ImportanceError(f"{leaf.name}: frame layout 인데 입력이 DataFrame 아님")
                if [s.name for s in lay.slots[None]] != list(X.columns):
                    raise ImportanceError(f"{leaf.name}: 입력 칼럼이 layout 과 다름")
            else:
                for key, ss in lay.slots.items():
                    if key not in X or X[key].shape[-1] != len(ss):
                        raise ImportanceError(f"{leaf.name}: 입력 {key} 모양이 layout 과 다름")
            self._X[k] = X
        return self._X[k]

    def pack(self, leaf: Leaf) -> _FramePack:
        k = id(leaf)
        if k not in self._pack:
            self._pack[k] = _FramePack(self.X(leaf))
        return self._pack[k]

    def _take(self, leaf: Leaf, sel: Mapping, base_take, sel_take):
        lay = leaf.get_layout()
        if lay.kind == "frame":
            return self.pack(leaf).take(base_take, sel.get(None), sel_take)
        return _take_dict(self.X(leaf), base_take, sel, sel_take)

    # ---- 섞기 전
    def base(self, node: Node) -> np.ndarray:
        k = id(node)
        if k in self._base:
            return self._base[k]
        if isinstance(node, Leaf):
            p = node.predict(self.X(node))
        elif isinstance(node, Stack):
            extra = {c: self.base(f) for c, f in node.feeders.items()}
            p = node.base.predict(node.combine(self.X(node.base), extra))
        elif isinstance(node, Mean):
            p = np.mean(np.vstack([self.base(m) for m in node.members]), axis=0)
        else:
            raise ImportanceError(f"모르는 노드: {type(node).__name__}")
        if p.shape != (self.n,):
            raise ImportanceError(f"{node.name}: 예측 길이 {p.shape} ≠ {self.n}")
        self._base[k] = p
        return p

    # ---- 섞은 뒤(R 회를 이어 붙임)
    def perm(self, node: Node, t: Target, base_take, sel_take, R: int, memo: dict) -> tuple[np.ndarray, bool]:
        k = id(node)
        if k in memo:
            return memo[k]
        if isinstance(node, Leaf):
            sel = node.get_layout().select(t)
            if not sel:
                out = (np.tile(self.base(node), R), False)
            else:
                g = self.gpu_input(node) if _speed_on("gpu_importance") and _gpu_capable(node) else None
                if g is not None:
                    out = (g.predict(node, sel, base_take, sel_take), True)
                else:
                    out = (node.predict(self._take(node, sel, base_take, sel_take)), True)
        elif isinstance(node, Stack):
            sel = node.base.get_layout().select(t)
            fe = {c: self.perm(f, t, base_take, sel_take, R, memo) for c, f in node.feeders.items()}
            if not sel and not any(ch for _, ch in fe.values()):
                out = (np.tile(self.base(node), R), False)
            else:
                Xb = self._take(node.base, sel, base_take, sel_take)
                out = (node.base.predict(node.combine(Xb, {c: v for c, (v, _) in fe.items()})), True)
        elif isinstance(node, Mean):
            parts = [self.perm(m, t, base_take, sel_take, R, memo) for m in node.members]
            if not any(ch for _, ch in parts):
                out = (np.tile(self.base(node), R), False)
            else:
                out = (np.mean(np.vstack([v for v, _ in parts]), axis=0), True)
        else:
            raise ImportanceError(f"모르는 노드: {type(node).__name__}")
        if out[0].shape != (R * self.n,):
            raise ImportanceError(f"{node.name}: 섞은 예측 길이 {out[0].shape} ≠ {R * self.n}")
        memo[k] = out
        return out


def h_block_cells(src_route: np.ndarray, keys: np.ndarray) -> list[np.ndarray]:
    """h 하나의 쌍(위치 0..n−1) → 칸(src_route 문자열 순) 목록, 칸 안은 lims_data_key 문자열 순(부록 B-5)."""
    route = np.asarray(src_route, dtype=object)
    keys = np.asarray(keys, dtype=object)
    cells = []
    for r in sorted(set(route.tolist())):
        pos = np.flatnonzero(route == r)
        order = sorted(range(pos.size), key=lambda i: keys[pos[i]])
        cells.append(pos[np.asarray(order, dtype=np.int64)])
    return cells


def permute_rows(n: int, cells: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """구현 세부 (a): 행 뽑기 배열 P(섞은 뒤 행 i = 섞기 전 행 P[i]), 칸마다 rng.permutation."""
    P = np.arange(n, dtype=np.int64)
    for c in cells:
        if c.size:
            P[c] = c[rng.permutation(c.size)]
    return P


# ================================================================ 계산
@dataclass
class ImportanceResult:
    records: pd.DataFrame            # fold·대상·r 마다 I
    summary: pd.DataFrame            # 대상마다 ψ_f·π·평균 I·건너뛰기
    base: dict                       # fold → 섞기 전 구간 가중 PR-AUC
    base_by_h: dict                  # fold → h 별 PR-AUC(60)
    M: int
    folds: tuple
    seconds: dict
    n_pairs: dict
    meta: dict = field(default_factory=dict)

    def removal_order(self) -> list[tuple[str, int]]:
        """3-0b: 건너뛰지 않는 대상을 π 낮은 순 → 평균 I 낮은 순 → 번호(반올림 없이)."""
        s = self.summary[~self.summary["skip"].fillna(False).astype(bool)]
        s = s.sort_values(["pi", "mean_I", "target_no"], kind="mergesort")
        return list(zip(s["target_kind"], s["target_no"].astype(int)))

    def save(self, out_dir, prefix: str = "importance") -> list[Path]:
        from .config import dump_scores_json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for nm, df in (("records", self.records), ("summary", self.summary)):
            p = out / f"{prefix}_{nm}.csv"
            with open(p, "x", newline="", encoding="utf-8-sig") as fh:
                df.to_csv(fh, index=False)
            paths.append(p)
        paths.append(dump_scores_json(out / f"{prefix}_meta.json",
                                      {"M": self.M, "folds": list(self.folds), "base": self.base,
                                       "base_by_h": {f: list(v) for f, v in self.base_by_h.items()},
                                       "seconds": self.seconds, "n_pairs": self.n_pairs, **self.meta}))
        return paths


def summarize(records: pd.DataFrame, folds: Sequence[str], targets: Sequence[Target], M: int) -> pd.DataFrame:
    """ψ_f·π·평균 I(반올림 없이), 건너뛰기 = 3 fold 모두 ψ_f ≥ 0.8 이고 3M 칸 평균 I > 0(3 fold 가 다 있을 때만)."""
    rows = []
    full = set(folds) == set(fd.CV_FOLDS)
    for t in targets:
        r = records[(records["target_kind"] == t.kind) & (records["target_no"] == t.no)]
        I = r["I"].to_numpy(dtype=np.float64)
        if I.size != M * len(folds) or np.isnan(I).any():
            raise ImportanceError(f"{t.kind} {t.name}: I 칸 수 {I.size} ≠ {M * len(folds)} 또는 NaN")
        row = {"target_kind": t.kind, "target_no": t.no, "target": t.name, "seed_key": "|".join(map(str, t.seed_key))}
        psis = {}
        for f in folds:
            v = r.loc[r["fold"] == f, "I"].to_numpy(dtype=np.float64)
            psis[f] = float((v > 0).mean())
            row[f"psi_{f}"] = psis[f]
            row[f"mean_I_{f}"] = float(v.mean())
        row["pi"] = float((I > 0).mean())
        row["mean_I"] = float(I.mean())
        row["skip"] = bool(full and all(psis[f] >= PSI_MIN for f in fd.CV_FOLDS) and row["mean_I"] > 0) if full else None
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_meta(samples: ev.SampleTable, idx: np.ndarray, h: int) -> pd.DataFrame:
    from .runner import add_bounds

    m = samples.meta().iloc[idx].copy()
    m["h"] = np.int16(h)
    return add_bounds(m.reset_index(drop=True), None)


def _prepare(models: Mapping[str, Node], targets: Sequence[Target], M: int, r_batch: int | None,
             only: Sequence[str] | None = None) -> tuple[list[Target], tuple, int]:
    """입력 점검(permutation_importance 와 같은 차례·문구). only = 대상 유무를 볼 fold(부분 계산 — 그 fold 만)."""
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ImportanceError(f"M ≥ 1 정수: {M!r}")
    if not models:
        raise ImportanceError("fold 모델 없음")
    targets = list(targets)
    if not targets:
        raise ImportanceError("대상 없음")
    keys = [t.key for t in targets]
    if len(set(keys)) != len(keys):
        raise ImportanceError("대상 중복")
    folds = tuple(f for f in fd.CV_FOLDS if f in models)
    if set(models) - set(folds):
        raise ImportanceError(f"fold 이름은 {fd.CV_FOLDS}: {sorted(models)}")
    for f in (folds if only is None else tuple(only)):
        if f not in models:
            raise ImportanceError(f"{f}: fold 모델 없음")
        absent = [t.name for t in targets if not models[f].has(t)]
        if absent:
            raise ImportanceError(f"{f}: 모델 입력에 없는 대상(구현 세부 e): {absent[:5]}")
    rb = M if r_batch is None else int(r_batch)
    if rb < 1:
        raise ImportanceError("r_batch ≥ 1")
    return targets, folds, rb


def _cell_sizes(src_route: np.ndarray) -> list[int]:
    """h_block_cells 와 같은 칸 차례(src_route 문자열 순)의 칸 크기 — 재생용(구현 세부 h)."""
    route = np.asarray(src_route, dtype=object)
    return [int((route == r).sum()) for r in sorted(set(route.tolist()))]


def _replay(rngs: Mapping, sizes: Sequence[int]) -> None:
    """구현 세부 (h): 예측 없이 permute_rows 와 같은 rng.permutation 호출(칸 차례·크기)만 — 난수 상태를 맞춤."""
    for g in rngs.values():
        for s in sizes:
            if s:
                g.permutation(s)


def _rng_md5(rngs: Mapping) -> str:
    h = hashlib.md5()
    for g in rngs.values():
        h.update(json.dumps(g.bit_generator.state, sort_keys=True).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _target_dict(t: Target) -> dict:
    return {"kind": t.kind, "no": t.no, "name": t.name, "seed_key": list(t.seed_key)}


def _samples_md5(samples: ev.SampleTable) -> str:
    h = hashlib.md5()
    h.update("\n".join(map(str, samples.keys)).encode("utf-8"))
    h.update(b"\x1e")
    h.update(np.ascontiguousarray(np.asarray(samples.d_days, dtype=np.int64)).tobytes())
    h.update(b"\x1e")
    h.update("\n".join(map(str, samples.src_route)).encode("utf-8"))
    return h.hexdigest()


def state_sig(fold: str, split: fd.Split, M: int, targets: Sequence[Target], samples: ev.SampleTable,
              tag: str = "") -> str:
    """구현 세부 (h) 설정 서명 — 이어 하기·합치기 때 같아야 함."""
    d = {"perm_seed": PERM_SEED, "fold": fold, "fold_no": fold_number(fold),
         "split": [int(split.train_end), int(split.score_start), int(split.score_end)], "M": int(M),
         "targets": [_target_dict(t) for t in targets], "samples": _samples_md5(samples), "tag": str(tag)}
    return hashlib.md5(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class _HStore:
    """구현 세부 (h): <state_dir>/<fold>/hNN.npz — h 하나의 bh·ph·쌍 수·초·rng 상태 md5·서명·대상."""

    def __init__(self, state_dir, fold: str, sig: str | None, targets: Sequence[Target] | None = None):
        self.dir = Path(state_dir) / fold
        self.fold, self.sig = fold, sig
        self.targets_json = json.dumps([_target_dict(t) for t in targets], ensure_ascii=False) if targets else ""

    def path(self, h: int) -> Path:
        return self.dir / f"h{int(h):02d}.npz"

    def has(self, h: int) -> bool:
        return self.path(h).exists()

    def load(self, h: int) -> dict | None:
        p = self.path(h)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if int(d["h"]) != int(h):
            raise ImportanceError(f"{p}: 안의 h {int(d['h'])} ≠ {h}")
        if self.sig is not None and str(d["sig"]) != self.sig:
            raise ImportanceError(f"{p}: 이어 하기 상태의 설정 서명이 지금과 다름(판·코드·대상·M·시료 중 무엇이 바뀜) — "
                                  "이 상태 폴더를 옮기고 처음부터 다시(지휘자 결정)")
        return d

    def save(self, h: int, *, bh: float, ph: np.ndarray, n: int, sec: float, rng_md5: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(h)
        tmp = self.dir / f".{p.name}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, sig=np.array(self.sig), targets=np.array(self.targets_json), h=np.int64(h),
                     bh=np.float64(bh), ph=np.asarray(ph, dtype=np.float64), n=np.int64(n), sec=np.float64(sec),
                     rng_md5=np.array(rng_md5))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return p


def _fold_pass(node: Node, samples: ev.SampleTable, split: fd.Split, fno: int, targets: Sequence[Target], M: int,
               rb: int, *, check: Callable[[], Any] | None = None, h_from: int = 1, h_to: int = H_MAX,
               store: _HStore | None = None, progress: Callable[[str], Any] | None = None) -> dict:
    """fold 하나의 h 차례 계산. h < h_from = 재생, [h_from, h_to] = 계산(store 에 있으면 재생 + 저장 값),
    h > h_to = 멈춤. 기본(h 1~60, store 없음) = 예전 permutation_importance 의 fold 안 계산 그대로."""
    rngs = {(ti, r): t.rng(fno, r) for ti, t in enumerate(targets) for r in range(1, M + 1)}
    bh = np.full(H_MAX, np.nan)
    ph = np.full((len(targets), M, H_MAX), np.nan)
    npairs, secs, computed, resumed = 0, 0.0, [], []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split):
        if h > h_to:
            break
        if idx.size == 0:
            continue
        if h < h_from:
            _replay(rngs, _cell_sizes(samples.src_route[idx]))
            continue
        if store is not None:
            d = store.load(h)
            if d is not None:
                _replay(rngs, _cell_sizes(samples.src_route[idx]))
                if str(d["rng_md5"]) != _rng_md5(rngs):
                    raise ImportanceError(f"{store.fold} h {h}: 재생한 난수 상태가 저장 값과 다름")
                if int(d["n"]) != idx.size or d["ph"].shape != (len(targets), M):
                    raise ImportanceError(f"{store.fold} h {h}: 저장된 쌍 수·모양이 지금과 다름")
                bh[h - 1] = float(d["bh"])
                ph[:, :, h - 1] = d["ph"]
                npairs += int(d["n"])
                secs += float(d["sec"])
                resumed.append(h)
                continue
        t_h = time.time()
        if check is not None:
            check()
        n = idx.size
        npairs += n
        meta = _pair_meta(samples, idx, h)
        y = samples.cv_y(idx)
        kk = samples.keys[idx]
        u = u_tie_many(kk, np.full(n, h, dtype=np.int64))
        E = _HEval(meta)
        pb = E.base(node)
        bh[h - 1] = mt.pr_auc(y, pb, u)
        cells = h_block_cells(samples.src_route[idx], kk)
        for ti, t in enumerate(targets):
            for lo in range(1, M + 1, rb):
                rs = list(range(lo, min(M, lo + rb - 1) + 1))
                P = [permute_rows(n, cells, rngs[(ti, r)]) for r in rs]
                base_take = np.tile(np.arange(n, dtype=np.int64), len(rs))
                sel_take = np.concatenate(P)
                pred, changed = E.perm(node, t, base_take, sel_take, len(rs), {})
                if not changed:  # pragma: no cover — 대상 없음은 위에서 멈춤
                    raise ImportanceError(f"{t.name}: 섞인 칼럼이 없음")
                for j, r in enumerate(rs):
                    ph[ti, r - 1, h - 1] = mt.pr_auc(y, pred[j * n:(j + 1) * n], u)
        if E.gpu_fallbacks and progress is not None:
            progress(f"h {h}: GPU 에 못 올린 잎(예전 길로) {sorted(set(E.gpu_fallbacks))}")
        E.release()
        del E
        computed.append(h)
        if store is not None:
            sec = time.time() - t_h
            secs += sec
            store.save(h, bh=bh[h - 1], ph=ph[:, :, h - 1], n=n, sec=sec, rng_md5=_rng_md5(rngs))
            if progress is not None:
                progress(f"{store.fold} h {h} 끝(쌍 {n}, {sec:.0f}초)")
    return {"bh": bh, "ph": ph, "npairs": npairs, "seconds": secs, "computed": computed, "resumed": resumed}


def _fold_base(f: str, bh: np.ndarray, expect_base, base_tol: float) -> float:
    b = mt.band_weighted(bh)
    if math.isnan(b):
        raise ImportanceError(f"{f}: 섞기 전 점수가 NaN")
    if expect_base is not None and f in expect_base and abs(b - float(expect_base[f])) > base_tol:
        raise ImportanceError(f"{f}: 섞기 전 점수 {b} ≠ 판의 fold 점수 {expect_base[f]}(구현 세부 g)")
    return b


def _fold_records(f: str, b: float, ph: np.ndarray, targets: Sequence[Target], M: int) -> list[dict]:
    rows = []
    for ti, t in enumerate(targets):
        for r in range(1, M + 1):
            s = mt.band_weighted(ph[ti, r - 1])
            rows.append({"fold": f, "target_kind": t.kind, "target_no": t.no, "target": t.name, "r": r,
                         "score_base": b, "score_perm": s, "I": b - s})
    return rows


def _result(rec_rows, targets, folds, M, rb, base, base_by_h, seconds, n_pairs) -> ImportanceResult:
    records = pd.DataFrame(rec_rows)
    summary = summarize(records, folds, targets, M)
    return ImportanceResult(records, summary, base, base_by_h, M, folds, seconds, n_pairs,
                            meta={"perm_seed": PERM_SEED, "psi_min": PSI_MIN, "r_batch": rb,
                                  "targets": [_target_dict(t) for t in targets]})


def permutation_importance(models: Mapping[str, Node], samples: ev.SampleTable, targets: Sequence[Target], *,
                           M: int = M_DEFAULT, splits: Mapping[str, fd.Split] | None = None,
                           r_batch: int | None = None, check: Callable[[], Any] | None = None,
                           expect_base: Mapping[str, float] | None = None, base_tol: float = 1e-9) -> ImportanceResult:
    """fold 모델 그래프({fold 이름: Node}, 재학습 없음)로 대상마다 섞기 중요도(부록 A-5).

    splits = fold 이름 → folds.Split(기본 folds.SPLITS — 시험에서만 바꿈). check = 자원 감시(h 마다 부름).
    반환 records 칼럼: fold·target_kind·target_no·target·r·score_base·score_perm·I.
    """
    targets, folds, rb = _prepare(models, targets, M, r_batch)
    rec_rows, base, base_by_h, seconds, n_pairs = [], {}, {}, {}, {}
    for f in folds:
        t0 = time.time()
        fno = fold_number(f)
        split = (splits or {}).get(f, fd.SPLITS[f])
        o = _fold_pass(models[f], samples, split, fno, targets, M, rb, check=check)
        bh = o["bh"]
        b = _fold_base(f, bh, expect_base, base_tol)
        base[f], base_by_h[f], n_pairs[f] = b, bh, o["npairs"]
        rec_rows += _fold_records(f, b, o["ph"], targets, M)
        seconds[f] = time.time() - t0
    return _result(rec_rows, targets, folds, M, rb, base, base_by_h, seconds, n_pairs)


# ================================================================ 나눠 돌리기·이어 하기(구현 세부 h)
def _check_range(h_from: int, h_to: int) -> None:
    for v in (h_from, h_to):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ImportanceError(f"h 구간은 정수: {h_from!r}~{h_to!r}")
    if not (1 <= h_from <= h_to <= H_MAX):
        raise ImportanceError(f"h 구간 1 ≤ h_from ≤ h_to ≤ {H_MAX}: {h_from}~{h_to}")


def importance_part(models: Mapping[str, Node], samples: ev.SampleTable, targets: Sequence[Target], *, fold: str,
                    h_from: int = 1, h_to: int = H_MAX, M: int = M_DEFAULT, state_dir,
                    splits: Mapping[str, fd.Split] | None = None, r_batch: int | None = None,
                    check: Callable[[], Any] | None = None, tag: str = "",
                    progress: Callable[[str], Any] | None = None) -> dict:
    """(fold, h 구간) 하나 계산 — h 마다 상태 파일, 있는 h 는 건너뜀(재생만). models 에 다른 fold 가 있어도 그 fold 만 씀."""
    fold_number(fold)
    _check_range(h_from, h_to)
    targets, _, rb = _prepare(models, targets, M, r_batch, only=(fold,))
    split = (splits or {}).get(fold, fd.SPLITS[fold])
    store = _HStore(state_dir, fold, state_sig(fold, split, M, targets, samples, tag), targets)
    t0 = time.time()
    o = _fold_pass(models[fold], samples, split, fold_number(fold), targets, M, rb, check=check, h_from=h_from,
                   h_to=h_to, store=store, progress=progress)
    return {"fold": fold, "h_from": int(h_from), "h_to": int(h_to), "computed": o["computed"], "resumed": o["resumed"],
            "seconds": time.time() - t0}


def pending_h(samples: ev.SampleTable, fold: str, h_from: int, h_to: int, state_dir,
              splits: Mapping[str, fd.Split] | None = None) -> list[int]:
    """구간 안 평가 쌍이 있는 h 중 상태 파일이 없는 h(서명은 안 봄 — 부분·합치기가 봄)."""
    fold_number(fold)
    _check_range(h_from, h_to)
    split = (splits or {}).get(fold, fd.SPLITS[fold])
    store = _HStore(state_dir, fold, None)
    out = []
    for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split):
        if h > h_to:
            break
        if h >= h_from and idx.size and not store.has(h):
            out.append(h)
    return out


def state_targets(state_dir) -> list[dict]:
    """상태 파일에 적힌 대상 목록(모든 파일이 같아야 함)."""
    files = sorted(Path(state_dir).glob("fold*/h*.npz"))
    if not files:
        raise ImportanceError(f"상태 파일 없음: {state_dir}")
    seen = set()
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            seen.add(str(z["targets"]))
    if len(seen) != 1:
        raise ImportanceError(f"상태 파일마다 대상 목록이 다름: {state_dir}")
    return json.loads(seen.pop())


def target_from_dict(d: Mapping) -> Target:
    """상태·meta 의 대상 dict → Target(이 모듈의 대상 생성 함수 중 dict 가 똑같이 나오는 것)."""
    kind, no, name = d["kind"], int(d["no"]), d["name"]
    cands: list[Target] = []
    if kind == "group":
        gr = rg.group(no)
        cands = [group_target(no), Target("group", gr.no, gr.name, units=frozenset(gr.units),
                                          seed_key=(SEED_FINAL_GROUP, gr.no))]
    elif kind == "unit":
        cands = [unit_target(name)]
    elif kind == "column" and name in RESEARCH_COLUMNS:
        cands = [research_column_target(name), research_column_target(name, final=True)]
    elif kind == "bundle":
        cands = [research_bundle_target(), research_bundle_target(final=True)]
    want = {"kind": kind, "no": no, "name": name, "seed_key": [int(x) for x in d["seed_key"]]}
    for t in cands:
        if _target_dict(t) == want:
            return t
    raise ImportanceError(f"대상 dict 를 Target 으로 못 바꿈: {dict(d)}")


def merge_parts(samples: ev.SampleTable, targets: Sequence[Target], *, state_dir, folds: Sequence[str] = fd.CV_FOLDS,
                M: int = M_DEFAULT, splits: Mapping[str, fd.Split] | None = None, r_batch: int | None = None,
                expect_base: Mapping[str, float] | None = None, base_tol: float = 1e-9,
                tag: str = "") -> ImportanceResult:
    """상태 파일(모든 fold·평가 쌍 있는 h)을 합쳐 permutation_importance 와 같은 결과. 빠진 h 가 있으면 멈춤."""
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ImportanceError(f"M ≥ 1 정수: {M!r}")
    targets = list(targets)
    if not targets:
        raise ImportanceError("대상 없음")
    keys = [t.key for t in targets]
    if len(set(keys)) != len(keys):
        raise ImportanceError("대상 중복")
    folds = tuple(f for f in fd.CV_FOLDS if f in set(folds))
    rb = M if r_batch is None else int(r_batch)
    if rb < 1:
        raise ImportanceError("r_batch ≥ 1")
    missing = {}
    loaded = {}
    for f in folds:
        split = (splits or {}).get(f, fd.SPLITS[f])
        store = _HStore(state_dir, f, state_sig(f, split, M, targets, samples, tag))
        bh = np.full(H_MAX, np.nan)
        ph = np.full((len(targets), M, H_MAX), np.nan)
        npairs, secs, miss = 0, 0.0, []
        for h, idx in fd.iter_scoring_pairs_by_h(samples.d_days, split):
            if idx.size == 0:
                continue
            d = store.load(h)
            if d is None:
                miss.append(h)
                continue
            if int(d["n"]) != idx.size or d["ph"].shape != (len(targets), M):
                raise ImportanceError(f"{f} h {h}: 저장된 쌍 수·모양이 지금과 다름")
            bh[h - 1] = float(d["bh"])
            ph[:, :, h - 1] = d["ph"]
            npairs += int(d["n"])
            secs += float(d["sec"])
        if miss:
            missing[f] = miss
        loaded[f] = (bh, ph, npairs, secs)
    if missing:
        raise ImportanceError(f"안 끝난 h 가 있음(부분을 다시 돌릴 것): {missing}")
    rec_rows, base, base_by_h, seconds, n_pairs = [], {}, {}, {}, {}
    for f in folds:
        bh, ph, npairs, secs = loaded[f]
        b = _fold_base(f, bh, expect_base, base_tol)
        base[f], base_by_h[f], n_pairs[f] = b, bh, npairs
        rec_rows += _fold_records(f, b, ph, targets, M)
        seconds[f] = secs
    return _result(rec_rows, targets, folds, M, rb, base, base_by_h, seconds, n_pairs)
