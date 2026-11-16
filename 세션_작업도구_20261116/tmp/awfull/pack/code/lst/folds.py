"""평가 틀(단계 0-7) — 계획서 §3, §1-1, §1-2, 부록 A-12·A-13.

| 판 | train(d) | 채점 T |
|---|---|---|
| fold 1 | ≤ 2022-06-30 | 2022-07-01 ~ 2023-05-01 |
| fold 2 | ≤ 2023-06-30 | 2023-07-01 ~ 2024-05-01 |
| fold 3 | ≤ 2024-06-30 | 2024-07-01 ~ 2025-05-01 |
| refit·test | ≤ 2025-06-30 | 2025-07-01 ~ 2026-05-01(test 기간 d 2025-07-01 ~ 2026-06-30) |

- train 행: d ≤ train 끝 시료마다 h 1개(부록 A-1 해시), T = d − h.
- 채점: T ∈ 채점 범위인 (시료, h) 쌍 전부(h 1~60, gap 0일, 모든 h 같은 T 범위).
- 덜 찬 달(§3): test 기간 끝 3개월(d 2026-04 ~ 06)만, 국내·해외 따로, 시료 수가 이전
  3년 같은 달 평균의 70% 미만이면 그 달 d 시료의 쌍을 test 채점에서 뺌 — 판정은 단계 0-1
  (여기는 계산 함수·적용 함수 틀).
날짜는 모두 numpy datetime64[D] 로 다룸(내부 계산은 1970-01-01 기준 일수 정수).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from .hassign import H_MAX, assign_h_many


def to_days(x) -> np.ndarray:
    """날짜(문자열·datetime·datetime64 배열/Series/스칼라) → 1970-01-01 기준 일수(int64)."""
    if not isinstance(x, (pd.Series, pd.Index)):
        x = np.atleast_1d(x)
        if x.dtype.kind in "iub":
            raise TypeError("정수는 날짜로 안 받음(헷갈림 방지)")
    s = pd.Series(pd.to_datetime(x))
    if s.isna().any():
        raise ValueError("날짜에 결측")
    if s.dt.tz is not None:
        raise ValueError("시간대 있는 날짜 금지")
    return s.to_numpy().astype("datetime64[D]").astype(np.int64)


def day_ints(x) -> np.ndarray:
    """날짜 입력 공통 변환: 정수 배열은 일수로 그대로, 그 밖(datetime64·문자열·Series)은 to_days.
    (datetime64[ns] 를 int64 로 바로 바꾸면 나노초 정수가 돼 조용히 틀리므로 반드시 이 함수로.)"""
    if isinstance(x, (pd.Series, pd.Index)):
        a = x.to_numpy()
    else:
        a = np.asarray(x)
    if a.dtype.kind in "iu":
        return np.atleast_1d(a.astype(np.int64))
    if a.dtype.kind == "b":
        raise TypeError("bool 은 날짜로 안 받음")
    return to_days(x)


def day(s: str) -> int:
    return int(to_days(s)[0])


def as_date(days) -> np.ndarray:
    return np.asarray(days, dtype=np.int64).astype("datetime64[D]")


@dataclass(frozen=True)
class Split:
    name: str
    train_end: int        # d ≤ train_end (일수)
    score_start: int      # T 채점 범위 시작
    score_end: int        # T 채점 범위 끝(포함)
    test_start: int | None = None  # test 기간(d) — refit 만
    test_end: int | None = None

    @property
    def score_days(self) -> int:
        return self.score_end - self.score_start + 1


SPLITS: dict[str, Split] = {
    "fold1": Split("fold1", day("2022-06-30"), day("2022-07-01"), day("2023-05-01")),
    "fold2": Split("fold2", day("2023-06-30"), day("2023-07-01"), day("2024-05-01")),
    "fold3": Split("fold3", day("2024-06-30"), day("2024-07-01"), day("2025-05-01")),
    "refit": Split("refit", day("2025-06-30"), day("2025-07-01"), day("2026-05-01"),
                   test_start=day("2025-07-01"), test_end=day("2026-06-30")),
}
CV_FOLDS = ("fold1", "fold2", "fold3")

# §3 표 기록값(RR2 361행만 뺀 수, 그 밖 규칙 적용 전) — 단계 0-1 대조용(다르면 멈춤)
RECORDED_COUNTS = {
    "fold1": {"score_days": 305, "train_n": 341_476, "train_pos": 1_045, "score_n": 74_869, "score_pos": 232, "score_pos_overseas": 4},
    "fold2": {"score_days": 306, "train_n": 416_625, "train_pos": 1_279, "score_n": 80_935, "score_pos": 212, "score_pos_overseas": 16},
    "fold3": {"score_days": 305, "train_n": 497_602, "train_pos": 1_491, "score_n": 87_253, "score_pos": 191, "score_pos_overseas": 27},
    "refit": {"score_days": 305, "train_n": 585_111, "train_pos": 1_682, "score_n": 90_740, "score_pos": 287, "score_pos_overseas": 21,
              "score_n_domestic": 86_540, "score_pos_domestic": 266, "score_n_overseas": 4_200},
}


def get_split(f) -> Split:
    """1·2·3·'fold1'…·'refit' 를 받음."""
    key = f"fold{f}" if isinstance(f, (int, np.integer)) and not isinstance(f, bool) else str(f)
    if key not in SPLITS:
        raise KeyError(f"모르는 판: {f!r}")
    return SPLITS[key]


# ---------------------------------------------------------------- train 행
def train_mask(d_days: np.ndarray, split: Split) -> np.ndarray:
    return day_ints(d_days) <= split.train_end


def make_train_rows(samples: pd.DataFrame, split: Split, key_col: str = "lims_data_key",
                    d_col: str = "d") -> pd.DataFrame:
    """시료 단위 h 1개 학습 행(§3): d ≤ train 끝, h = 부록 A-1 해시, T = d − h.

    돌려주는 칼럼 = 입력 칼럼 + h(int64)·T(datetime64[D]). 입력 순서 유지.
    입력에 h 칼럼이 이미 있으면 해시와 같은지 확인(다르면 멈춤).
    """
    if samples[key_col].duplicated().any():
        raise ValueError("lims_data_key 중복")
    d = to_days(samples[d_col])
    m = train_mask(d, split)
    out = samples.loc[m].copy()
    h = assign_h_many(out[key_col].tolist())
    if "h" in out.columns:
        if not np.array_equal(out["h"].to_numpy(dtype=np.int64), h):
            raise ValueError("입력 h 가 부록 A-1 해시와 다름")
    out["h"] = h
    out["T"] = as_date(d[m] - h)
    return out


# ---------------------------------------------------------------- 채점 쌍
def iter_scoring_pairs_by_h(d_days: np.ndarray, split: Split) -> Iterator[tuple[int, np.ndarray]]:
    """h = 1..60 마다 (h, 채점 시료 위치) — T = d − h ∈ [채점 시작, 끝](B-1: h 별로 메모리에서)."""
    d = day_ints(d_days)
    for h in range(1, H_MAX + 1):
        t = d - h
        idx = np.flatnonzero((t >= split.score_start) & (t <= split.score_end))
        if (d[idx] <= split.train_end).any():
            raise AssertionError("채점 쌍이 train 기간과 겹침")
        yield h, idx


def scoring_pairs(d_days: np.ndarray, split: Split) -> pd.DataFrame:
    """(시료 위치 row, h, T, d) 전부. 정렬 = h → 시료 위치."""
    d_days = day_ints(d_days)
    rows, hs = [], []
    for h, idx in iter_scoring_pairs_by_h(d_days, split):
        rows.append(idx)
        hs.append(np.full(idx.shape, h, dtype=np.int64))
    row = np.concatenate(rows) if rows else np.empty(0, np.int64)
    h = np.concatenate(hs) if hs else np.empty(0, np.int64)
    d = day_ints(d_days)[row]
    return pd.DataFrame({"row": row, "h": h, "T": as_date(d - h), "d": as_date(d)})


def scoring_sample_mask(d_days: np.ndarray, split: Split) -> np.ndarray:
    """채점 시료(쌍 하나라도 채점 범위 T 안) = d ∈ [채점 시작 + 1, 채점 끝 + 60]."""
    d = day_ints(d_days)
    return (d >= split.score_start + 1) & (d <= split.score_end + H_MAX)


def common_d_range(split: Split) -> tuple[int, int]:
    """모든 h 공통 d 범위(부록 A-12 보조 점수·A-13 그림) = [채점 시작 + 60, 채점 끝 + 1]
    (예: fold 1 = 2022-08-30 ~ 2023-05-02)."""
    return split.score_start + H_MAX, split.score_end + 1


def common_d_mask(pair_d_days: np.ndarray, split: Split) -> np.ndarray:
    lo, hi = common_d_range(split)
    d = day_ints(pair_d_days)
    return (d >= lo) & (d <= hi)


def count_split(d_days, y, src_route, split: Split) -> dict:
    """§3 표의 셈(시료 단위): train 시료·검출, 채점 시료·검출, 해외 채점 검출 등."""
    d = day_ints(d_days)
    y = np.asarray(y, dtype=np.int64)
    ov = np.asarray(src_route) == "overseas"
    tr = train_mask(d, split)
    sc = scoring_sample_mask(d, split)
    out = {
        "score_days": split.score_days,
        "train_n": int(tr.sum()), "train_pos": int(y[tr].sum()),
        "score_n": int(sc.sum()), "score_pos": int(y[sc].sum()),
        "score_pos_overseas": int(y[sc & ov].sum()),
    }
    if split.name == "refit":
        out.update({"score_n_domestic": int((sc & ~ov).sum()), "score_pos_domestic": int(y[sc & ~ov].sum()),
                    "score_n_overseas": int((sc & ov).sum())})
    return out


def check_split_counts(observed: Mapping[str, int], split: Split) -> None:
    """§3: 같은 기준으로 센 수가 기록값과 다르면 멈춤."""
    rec = RECORDED_COUNTS[split.name]
    diff = {k: (observed.get(k), v) for k, v in rec.items() if observed.get(k) != v}
    if diff:
        raise AssertionError(f"{split.name} 행 수가 §3 기록값과 다름 (관측, 기록): {diff}")


# ---------------------------------------------------------------- 덜 찬 달(§3)
TEST_TAIL_MONTHS = ("2026-04", "2026-05", "2026-06")
PRIOR_YEARS = 3
INCOMPLETE_RATIO = (7, 10)  # 70% 미만


def incomplete_month_table(month_counts: pd.DataFrame) -> pd.DataFrame:
    """month_counts: 칼럼 src_route·ym('YYYY-MM', d 의 달)·n(시료 수).

    test 기간 끝 3개월마다·국내/해외마다 이전 3년 같은 달 평균과 비교 →
    incomplete = n < 0.7 × 평균(정수 비교: 30·n < 7·합). 판정은 단계 0-1 이 이 표로 함.
    구현 세부(계획서에 없음): month_counts 에 없는 (src_route, 달) 은 0건으로 봄.
    """
    need = {"src_route", "ym", "n"}
    if not need <= set(month_counts.columns):
        raise ValueError(f"칼럼 필요: {sorted(need)}")
    if month_counts.duplicated(["src_route", "ym"]).any():
        raise ValueError("(src_route, ym) 중복")
    cnt = {(r, ym): int(n) for r, ym, n in month_counts[["src_route", "ym", "n"]].itertuples(index=False)}
    rows = []
    for route in ("domestic", "overseas"):
        for ym in TEST_TAIL_MONTHS:
            y, mth = int(ym[:4]), ym[5:]
            prior = [cnt.get((route, f"{y - k}-{mth}"), 0) for k in range(1, PRIOR_YEARS + 1)]
            n = cnt.get((route, ym), 0)
            num, den = INCOMPLETE_RATIO
            s = sum(prior)
            rows.append({"src_route": route, "ym": ym, "n": n, "prior_counts": prior,
                         "prior_mean": s / PRIOR_YEARS,
                         "incomplete": bool(PRIOR_YEARS * den * n < num * s)})
    return pd.DataFrame(rows)


def incomplete_pair_mask(pair_d_days, pair_src_route, incomplete: set[tuple[str, str]]) -> np.ndarray:
    """test 쌍 중 덜 찬 달(단계 0-1 이 정한 (src_route, 'YYYY-MM') 집합)의 d 시료 쌍 = True
    → test 채점에서 빼고 '미완 구간' 으로 따로 보고."""
    ym = pd.Series(as_date(day_ints(pair_d_days))).dt.strftime("%Y-%m").to_numpy()
    route = np.asarray(pair_src_route)
    bad = np.zeros(ym.shape[0], dtype=bool)
    for r, m in incomplete:
        if m not in TEST_TAIL_MONTHS:
            raise ValueError(f"덜 찬 달 대상은 test 끝 3개월만: {m}")
        bad |= (route == r) & (ym == m)
    return bad
