"""블록 OOF·엄격 OOF 학습 계획 — 계획서 §7-3, 부록 A-12('fold 안 블록'·'같은 행 모델 재사용'·'3-0b 딥러닝 예측
범위'), 부록 A-9(횟수), §6 2차 공통('단계 4~6 을 현재 최고 판에 하는 법' — 줄인·뽑은 자료로 다시 만들 때 재사용 없음).

규칙(계획서 그대로)
- 전역 블록(d 기준): 1 = 2015-04-07 ~ 2017-06-30, 2 = 2017-07-01 ~ 2019-06-30, 3 = 2019-07-01 ~ 2021-06-30,
  4 = 2021-07-01 ~ 2023-06-30, 5 = 2023-07-01 ~ 2025-06-30. 블록마다 나머지 4블록으로 학습한 모델이 그 블록 시료 예측.
- fold 안 블록 = 전역 블록 경계를 fold train 끝에서 자름: fold 1 = 블록 1·2·3 + 2021-07-01 ~ 2022-06-30(4블록),
  fold 2 = 블록 1~4, fold 3 = 블록 1~4 + 2023-07-01 ~ 2024-06-30(5블록). refit 의 블록 = 전역 블록(train 끝 = 블록 5 끝).
- 엄격 OOF(fold f): train 행 값 = fold train 안 블록 OOF, validation 값 = fold train 전체로 학습한 모델.
  refit·test 값 = 전역 블록 OOF(train 행) + test 전 전체 학습 모델(test 행, 2015-04-07 ~ 2025-06-30).
- 같은 행 모델 재사용: 학습 행이 같은 모델은 한 번만 학습해 같이 씀 — d ≤ 2021-06-30 모델 = fold 1 넷째 블록 OOF =
  fold 2 블록 4 OOF, d ≤ 2023-06-30 모델 = fold 2 train 전체 = fold 3 다섯째 블록 OOF = 전역 블록 5 OOF.
  → 엄격 14(= 16 − 2)·전역 6·합 19(= 전역 6 + 엄격 13).
- 줄인·뽑은 자료로 학습하는 모델(단계 4~6·4-c 의 다시 학습한 값 모델·④ 의 블록 트리·그 위 단계 8)은 재사용 안 함
  (`reuse=False`) → 22(= 전역 6 + 엄격 16), 모델마다 그 모델의 학습 블록 시료로 줄이기·뽑기를 다시 정함
  (`downsize.block_model_mask`).

공개 API
    GLOBAL_BLOCKS                                   # ((번호, 시작 일수, 끝 일수), …) 5개
    fold_blocks(split) -> list[Block]               # 그 판(fold1~3·refit)의 블록(train 끝에서 자름)
    block_of(d, split) -> int64 (N,)                # 시료마다 그 판의 블록 번호(1~), train 밖 = 0
    block_plan(*, reuse=True, splits=SPLIT_ORDER) -> list[BlockModel]
    plan_counts(plan) -> dict                       # 모델 수(전체·엄격·전역·판 전체 모델 수)
    plan_table(plan) -> DataFrame                   # 기록용(모델·역할 한 줄씩)
    BlockModel.train_mask(d) -> bool (N,)           # 그 모델 학습 시료(줄이기 전 — 시료당 자기 h 1개 행)
    oof_target_mask(d, split, block) -> bool (N,)   # 그 판 train 행 중 그 블록(d) — OOF 예측 대상 행
    assemble_oof(plan, split, d) -> int64 (N,)      # train 행마다 값을 내는 모델 번호(plan 안 위치), train 밖 −1
    full_model(plan, split) -> BlockModel           # 그 판 train 전체 모델(fold = validation 쌍, refit = test 쌍)
    models_for_split(plan, split) -> list[int]      # 그 판 OOF 를 만드는 모델 위치(블록 순 + full)
    model_train_rows(model, known_frame, keep_keys=None) -> DataFrame   # 학습 행(시료당 자기 h 1개, h·T)
    model_sizes(plan, d) -> dict[이름, 시료 수]

예측 대상(역할 Role.kind)
- 'oof'  : 그 판(split) train 행 중 d ∈ 그 블록인 행 — 행의 h 는 부르는 쪽이 정함(부록 A-12 '3-0b 딥러닝 예측 범위':
           트리 train 행 = 그 행 h, 단계 4·① 은 그 판 train 행의 h 전부 — 4-a = h 1~60(또는 m 개), 4-a0 = 자기 h,
           ① = h 1).
- 'full' : fold = 그 fold validation 쌍(모든 h), refit = test 쌍(모든 h). CV·refit 의 그 판 모델과 학습 행이 같으면
           (재사용 켬 + 줄이지 않은 자료) 이미 학습한 모델을 씀(부록 A-9 '− CV·refit 에서 이미 학습한 4').

구현 세부(계획서에 없음 — 담당자 확인):
 (1) 판 이름: fold 판은 'fold1'~'fold3', 전역 블록 = 'refit' 판의 블록(같은 경계). 모델 이름 = '<판>-b<번호>'(블록 OOF)·
     '<판>-full'(판 train 전체), 재사용하면 처음 나온 이름 + aliases(나머지 이름).
 (2) 모델 순서 = 판 순서(fold1 → fold2 → fold3 → refit), 판 안은 블록 번호 → full. 재사용 모델은 처음 나온 자리.
 (3) '같은 행' 판정 = 학습 d 구간(합친 구간 목록)이 같음(학습 행 = 그 구간 d 의 시료, 시료당 h 1개 — 같은 시료 표면
     같은 행). 시료 표가 판마다 같다는 전제(단계 0-1 시료 표 하나).
 (4) 시료 d 가 2015-04-07 전이면 멈춤(블록 1 시작 전 — 단계 0-1 시료에는 없음).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .folds import SPLITS, Split, as_date, day, day_ints, get_split

SPLIT_ORDER = ("fold1", "fold2", "fold3", "refit")
GLOBAL_SPLIT = "refit"   # 전역 블록 = refit 판의 블록

GLOBAL_BLOCKS: tuple[tuple[int, int, int], ...] = (
    (1, day("2015-04-07"), day("2017-06-30")),
    (2, day("2017-07-01"), day("2019-06-30")),
    (3, day("2019-07-01"), day("2021-06-30")),
    (4, day("2021-07-01"), day("2023-06-30")),
    (5, day("2023-07-01"), day("2025-06-30")),
)
DATA_START = GLOBAL_BLOCKS[0][1]

ROLE_KINDS = ("oof", "full")


@dataclass(frozen=True)
class Block:
    no: int
    lo: int   # d 시작(일수, 포함)
    hi: int   # d 끝(일수, 포함)

    def as_dates(self) -> tuple[str, str]:
        return str(as_date([self.lo])[0]), str(as_date([self.hi])[0])


def _split_obj(split) -> Split:
    return split if isinstance(split, Split) else get_split(split)


def fold_blocks(split) -> list[Block]:
    """그 판의 블록 = 전역 블록 경계를 train 끝에서 자름(빈 블록은 뺌)."""
    sp = _split_obj(split)
    out = []
    for no, lo, hi in GLOBAL_BLOCKS:
        if lo > sp.train_end:
            break
        out.append(Block(no, lo, min(hi, sp.train_end)))
    if out[-1].hi != sp.train_end:
        raise AssertionError(f"{sp.name}: train 끝이 전역 블록 밖")
    return out


def _check_d(d) -> np.ndarray:
    dd = day_ints(d)
    if dd.size and dd.min() < DATA_START:
        raise ValueError("2015-04-07 전 d 시료(블록 1 시작 전)")
    return dd


def block_of(d, split) -> np.ndarray:
    """시료마다 그 판의 블록 번호(1~), d > train 끝 = 0."""
    dd = _check_d(d)
    out = np.zeros(dd.shape, dtype=np.int64)
    for b in fold_blocks(split):
        out[(dd >= b.lo) & (dd <= b.hi)] = b.no
    return out


def _merge(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    iv = sorted((int(a), int(b)) for a, b in intervals if a <= b)
    out: list[list[int]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return tuple((a, b) for a, b in out)


@dataclass(frozen=True)
class Role:
    split: str          # fold1~3·refit
    kind: str           # 'oof' | 'full'
    block: int | None   # oof 의 블록 번호(full = None)

    @property
    def name(self) -> str:
        return f"{self.split}-b{self.block}" if self.kind == "oof" else f"{self.split}-full"

    @property
    def scope(self) -> str:
        """'strict'(fold1~3 — 엄격 OOF) | 'global'(refit — 전역 블록 OOF·test 전 모델)."""
        return "global" if self.split == GLOBAL_SPLIT else "strict"


@dataclass
class BlockModel:
    """한 번 학습하는 모델 하나 — 학습 d 구간과 그 모델이 값을 내는 역할 목록."""
    name: str
    train_intervals: tuple[tuple[int, int], ...]
    roles: list[Role] = field(default_factory=list)

    @property
    def aliases(self) -> list[str]:
        return [r.name for r in self.roles[1:]]

    @property
    def is_split_full(self) -> bool:
        """판 train 전체 모델 역할이 있음 — 재사용 켬·줄이지 않은 자료면 CV·refit 의 그 판 모델과 같은 행."""
        return any(r.kind == "full" for r in self.roles)

    @property
    def owner(self) -> Role:
        return self.roles[0]

    def train_mask(self, d) -> np.ndarray:
        """학습 시료(줄이기·뽑기 전) = d ∈ 학습 구간."""
        dd = _check_d(d)
        m = np.zeros(dd.shape, dtype=bool)
        for a, b in self.train_intervals:
            m |= (dd >= a) & (dd <= b)
        return m

    def intervals_as_dates(self) -> list[tuple[str, str]]:
        return [(str(as_date([a])[0]), str(as_date([b])[0])) for a, b in self.train_intervals]


def _role_intervals(split: str, kind: str, block: Block | None) -> tuple[tuple[int, int], ...]:
    sp = SPLITS[split]
    if kind == "full":
        return _merge([(DATA_START, sp.train_end)])
    assert block is not None
    parts = []
    for b in fold_blocks(sp):
        if b.no != block.no:
            parts.append((b.lo, b.hi))
    return _merge(parts)


def block_plan(*, reuse: bool = True, splits: Sequence[str] = SPLIT_ORDER) -> list[BlockModel]:
    """블록 OOF·엄격 OOF 학습 계획(어느 모델이 어느 행을 학습·예측하는지).

    reuse=True  : 같은 학습 행 모델 한 번(줄이지 않은 자료 — 3-0b 딥러닝 상위 3·이상 점수 4종·⑥ 3종·④ 블록 트리)
    reuse=False : 재사용 없음(줄인·뽑은 자료 — 단계 4~6·4-c 값 모델·④ 블록 트리·그 위 단계 8 값 모델)
    """
    bad = [s for s in splits if s not in SPLIT_ORDER]
    if bad or len(set(splits)) != len(splits):
        raise ValueError(f"splits 는 {SPLIT_ORDER} 중 겹치지 않게: {splits!r}")
    splits = [s for s in SPLIT_ORDER if s in splits]
    plan: list[BlockModel] = []
    by_key: dict[tuple, BlockModel] = {}
    for s in splits:
        items = [(Role(s, "oof", b.no), b) for b in fold_blocks(s)] + [(Role(s, "full", None), None)]
        for role, b in items:
            iv = _role_intervals(s, role.kind, b)
            if reuse and iv in by_key:
                by_key[iv].roles.append(role)
                continue
            m = BlockModel(name=role.name, train_intervals=iv, roles=[role])
            plan.append(m)
            if reuse:
                by_key[iv] = m
    return plan


def plan_counts(plan: Sequence[BlockModel]) -> dict:
    """모델 수: total(학습 횟수)·strict(엄격 OOF 에 쓰는 모델)·global(전역 OOF·test 전 모델)·
    strict_only(전역에 안 쓰이는 엄격 모델)·split_full(판 train 전체 모델 — CV·refit 판 모델과 같은 행일 수 있음)."""
    strict = [m for m in plan if any(r.scope == "strict" for r in m.roles)]
    glob = [m for m in plan if any(r.scope == "global" for r in m.roles)]
    return {"total": len(plan), "strict": len(strict), "global": len(glob),
            "strict_only": sum(1 for m in strict if m not in glob),
            "split_full": sum(1 for m in plan if m.is_split_full),
            "roles": sum(len(m.roles) for m in plan)}


def plan_table(plan: Sequence[BlockModel]) -> pd.DataFrame:
    """기록용 표: 모델 × 역할 한 줄씩(모델 번호·이름·학습 d 구간·역할 판·종류·블록·예측 대상 d 구간)."""
    rows = []
    for i, m in enumerate(plan):
        iv = "; ".join(f"{a} ~ {b}" for a, b in m.intervals_as_dates())
        for r in m.roles:
            if r.kind == "oof":
                b = next(x for x in fold_blocks(r.split) if x.no == r.block)
                tgt = "train 행 d {} ~ {}".format(*b.as_dates())
            else:
                tgt = "validation 쌍" if r.split != GLOBAL_SPLIT else "test 쌍"
            rows.append({"model_no": i, "model": m.name, "train_d": iv, "split": r.split, "scope": r.scope,
                         "kind": r.kind, "block": r.block, "target": tgt})
    return pd.DataFrame(rows)


def oof_target_mask(d, split, block: int) -> np.ndarray:
    """그 판 train 행(시료) 중 d ∈ 그 블록 — OOF 예측 대상. 행의 h 는 부르는 쪽(부록 A-12 예측 범위)."""
    return block_of(d, split) == int(block)


def _find(plan: Sequence[BlockModel], split: str, kind: str, block: int | None) -> int:
    hits = [i for i, m in enumerate(plan) for r in m.roles if (r.split, r.kind, r.block) == (split, kind, block)]
    if len(hits) != 1:
        raise KeyError(f"계획에 ({split}, {kind}, {block}) 역할이 {len(hits)}개")
    return hits[0]


def assemble_oof(plan: Sequence[BlockModel], split: str, d) -> np.ndarray:
    """train 행(시료 d)마다 그 행의 OOF 값을 내는 모델의 plan 위치. 그 판 train 밖 = −1.
    점검: 값을 내는 모델이 그 행 시료를 학습하지 않았는지(OOF) assert."""
    dd = _check_d(d)
    blk = block_of(dd, split)
    out = np.full(dd.shape, -1, dtype=np.int64)
    for b in fold_blocks(split):
        i = _find(plan, split, "oof", b.no)
        rows = blk == b.no
        if plan[i].train_mask(dd[rows]).any():
            raise AssertionError(f"{plan[i].name}: 예측할 블록 시료를 학습함(OOF 아님)")
        out[rows] = i
    return out


def full_model(plan: Sequence[BlockModel], split: str) -> BlockModel:
    """그 판 train 전체 모델(fold = validation 값, refit = test 값)."""
    return plan[_find(plan, split, "full", None)]


def models_for_split(plan: Sequence[BlockModel], split: str) -> list[int]:
    """그 판의 엄격(또는 전역) OOF 를 만드는 데 필요한 모델 위치(블록 순 + full)."""
    return [_find(plan, split, "oof", b.no) for b in fold_blocks(split)] + [_find(plan, split, "full", None)]


def model_train_rows(model: BlockModel, known_frame: pd.DataFrame, keep_keys=None) -> pd.DataFrame:
    """그 모델 학습 행 = d ∈ 학습 구간인 시료마다 자기 h 1개(folds.make_train_rows 꼴 — h·T 칼럼).
    keep_keys = 줄인·뽑은 자료일 때 남길 시료 키(downsize.block_model_mask 결과의 키), None 이면 줄이지 않음."""
    from .folds import make_train_rows

    rows = make_train_rows(known_frame, SPLITS[GLOBAL_SPLIT])
    m = model.train_mask(rows["d"])
    if keep_keys is not None:
        if len(model.roles) != 1:
            raise ValueError(f"{model.name}: 줄인·뽑은 자료는 block_plan(reuse=False) 로(부록 A-12)")
        keep = pd.Index(np.asarray(keep_keys, dtype=object)).get_indexer(
            pd.Index(rows["lims_data_key"].astype(str).to_numpy(dtype=object))) >= 0
        if (keep & ~m).any():
            raise AssertionError(f"{model.name}: 남길 키가 학습 구간 밖")
        m &= keep
    return rows.loc[m]


def model_sizes(plan: Sequence[BlockModel], d) -> dict[str, int]:
    """모델마다 학습 시료 수(줄이기 전) — 0-9 측정 대상 고르기용."""
    return {m.name: int(m.train_mask(d).sum()) for m in plan}
