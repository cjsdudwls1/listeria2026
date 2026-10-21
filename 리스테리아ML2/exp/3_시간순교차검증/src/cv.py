# -*- coding: utf-8 -*-
"""3차 실험(시간순 교차검증) 틀: 날짜 경계(설정값), 행 고르기, 입력 묶음, 입력 캐시, 채점.

사양 = exp/3_시간순교차검증/사양_20261020.md. 설정 = config3.json(cfg3.py 머리말).
- 2차(exp/2_고도화/src)·1차(exp/1_베이스라인_lims기상/src) 코드는 고치지 않고 import. 맞추는 감싸기는 cfg3.py.
- 행 = 단위(시군구 x 품목 2단계 x 3단계) x 기준일 d (1차 data.build_unit_rows, 2차 Ctx2). 3차는 Ctx2 의 분할 이름을 쓰지 않음:
  cfg3 가 모든 행을 Ctx2 'train' 한 분할(SP)에 넣고, 여기서 행의 날짜로 직접 고른다.
- 예측하는 날 T = d − h (h = 1..60). 입력 = T 에 알 수 있는 것만(사양 2절): 기상 창 T−L .. T−1.
- 과거 검사 결과로 만드는 칸(PAST = 과거 이력 hist_core·logn1·sd1, WOE, 제품명 과거 유형 N1, 그 안의 전국 p0) - 사양 2절 규칙(PAST_RULE_DESC):
  학습 행 = 그 행 예측일 T 전날까지 굴러가는 창(기준일 ∈ [T−P, T−1], P = 1년·5년) - 2차 e2hist·e2nb 계산 그대로(h 그대로 넘김).
  맞혀 보는 행·시험 행 = 그 겹(최종) 학습 끝 날 TE 까지 시료로 만든 값(기준일 ∈ [TE−P+1, TE])을 그 해 모든 행·모든 h 에 고정(학습 끝 값 고정)
  - 2차 코드는 창을 'T−P .. T−1' 로 세므로 h 대신 d − (TE + 1일) 을 넘겨 창 끝을 TE 로 맞춘다(RowSet.past_h). 2차 코드는 고치지 않음.
  그날 제품명(n2·n0)·기상·외부 자료는 이 규칙과 무관(그대로).
- 겹 k: 학습 행 = d <= train_end (= 맞혀 보는 해 시작 전날, 행마다 h 하나), 맞혀 보는 행 = (행, h) 중 T ∈ [t_start, t_end] (모든 h 같은 날 수).
- 시험 = T ∈ [시험 시작, 데이터 끝 − 60일]. 시험 행은 final 단계에서만 만든다: test_rowset 은 ALLOW_TEST(final 이 켬) 없이 부르면 오류로 멈춤,
  만든 기록은 TEST_ACCESS 에 남김.
- 학습 행 h: cfg3.BOUNDS['h_rule'].
"""
import hashlib
import json
import os
import time
from pathlib import Path

import cfg3 as G     # 반드시 먼저(설정·감싸기)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import pyarrow as pa        # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

E, C1 = G.E, G.C1
import features as F1        # noqa: E402
import metrics as M1         # noqa: E402
from e2data import Ctx2      # noqa: E402
from e2hist import hist_names, woe_names  # noqa: E402
from e2models import FeatureBuilder  # noqa: E402

SRC3, EXP3_DIR, EXP2_SRC = G.SRC3, G.EXP3_DIR, G.EXP2_SRC
L = G.L
H_MAX = G.H_MAX
SP = "train"                                  # cfg3 가 모든 행을 넣은 Ctx2 분할
BASE_CATS = list(C1.CAT_COLS)                 # sido, sgg, p2, p3
TOPK = (1, 5, 10, 20)
TEST_ACCESS = []                              # 시험 행을 만든 기록(고르는 단계에서는 비어 있어야 함)

# ---------------------------------------------------------------- 입력 묶음 (2차 묶음 정의 그대로, 계산은 2차 FeatureBuilder)
# 범주 4칸 + h 는 항상 넣음(빼기 대상 아님). premise=True 는 운영 전제(그날 제품명)가 있는 묶음 - '변수 전부'에 넣지 않음.
BUNDLES = {
    "wx":        {"name": "기상 요약", "add": {"weather": True}},
    "spi":       {"name": "표준강수지수(SPI3·SPI6)", "add": {"ext": ["spi"]}},
    "soilmed":   {"name": "토양 중앙값", "add": {"ext": ["soilmed"]}},
    "food":      {"name": "식품 속성", "add": {"ext": ["food"]}},
    "extidx":    {"name": "외부 지표(친환경·지하수·HACCP)", "add": {"ext": ["eco", "gw", "haccp"]}},
    "rg":        {"name": "생장 범위", "add": {"ext": ["rgwx", "rgsoil", "rgfood"]}},
    "g123":      {"name": "생장식 지수", "add": {"ext": ["g1", "g2", "g3"]}},
    "wxanom":    {"name": "이동평균 차감 기상", "add": {"ext": ["wxanom"]}},
    "hist_core": {"name": "과거 검출 이력(5년·1년 스무딩 검출률, 5년 로그 검사 수)", "add": {"hist": ["rate5", "logn5", "rate1"]}, "k": True},
    "logn1":     {"name": "1년 로그 검사 수", "add": {"hist": ["logn1"]}},
    "sd1":       {"name": "1년 검출률 불확실성", "add": {"hist": ["sd1"]}, "k": True},
    "woe":       {"name": "WOE", "add": {"woe": True}},
    "n1":        {"name": "제품명 과거 유형", "add": {"nb": ["n1"]}, "k": True},
    "ae4":       {"name": "이상치 점수(AE·VAE 4개, 겹마다 재학습)", "add": {}, "ae": True},
    "n2":        {"name": "그날 제품명 유형(운영 전제)", "add": {"nb": ["n2"]}, "premise": True},
    "n0":        {"name": "그날 제품명 TF-IDF(운영 전제)", "add": {"nb": ["n0"]}, "premise": True},
}
BUNDLE_ORDER = list(BUNDLES)
HIST_ORDER = ["rate5", "logn5", "rate1", "logn1", "sd1"]
NB_ORDER = ["n1", "n2", "n0"]
PAST = {"hist_core", "logn1", "sd1", "woe", "n1"}   # 과거 검사 결과로 만드는 칸(사양 2절)
PAST_RULE = "v2_trainRollT-1_evalFixTE"              # 캐시 기록·이름에 넣는 규칙 이름(옛 캐시와 섞이지 않게)
PAST_RULE_DESC = ("과거 이력(품목군 3단계·시군구·단위 x 5년·1년 스무딩 검출률, log(1+검사 수), 1년 불확실성)·WOE 4개·제품명 과거 유형 N1·전국 p0: "
                  "학습 행 = 그 행 예측일 T 전날까지 굴러가는 창(기준일 ∈ [T−P, T−1], P = 365·1,826일), "
                  "맞혀 보는 행·시험 행 = 그 겹(최종) 학습 끝 날 TE 까지 시료로 만든 값(기준일 ∈ [TE−P+1, TE])을 그 해 모든 행·모든 h 에 고정(학습 끝 값 고정). "
                  "계산 = 2차 e2hist·e2nb 그대로, 맞혀 보는 행·시험 행은 h 대신 d − (TE + 1일)을 넘김")
ALLOW_TEST = False                           # final 단계만 True(그 밖에서 test_rowset 을 부르면 오류)
FEATSETS = {"base": ["wx"], "all": list(G.CFG["featset_all"]), "tune": list(G.CFG.get("featset_tune") or G.CFG["featset_all"]), "none": []}
AE_KINDS = list(E.AE_KINDS)                   # AEt, VAEt, AEs, VAEs
AE_NAMES = [f"ae_{k}" for k in AE_KINDS]


def resolve_featset(fs):
    """'base'·'all' 이름, 또는 묶음 코드 목록(쉼표 문자열 가능) -> 정렬된 묶음 목록(BUNDLE_ORDER 순)."""
    if isinstance(fs, str):
        fs = FEATSETS[fs] if fs in FEATSETS else [x for x in fs.split(",") if x]
    bad = [b for b in fs if b not in BUNDLES]
    if bad:
        raise ValueError(f"모르는 묶음: {bad}")
    return [b for b in BUNDLE_ORDER if b in set(fs)]


def needs_k(bundles):
    return any(BUNDLES.get(b, {}).get("k") for b in bundles)


def feats_of(bundles, k):
    """FeatureBuilder 입력 dict(AE 제외). 순서는 2차 정의 순서(EXT_BLOCKS·HIST·NB_ORDER)."""
    f = {"weather": False, "cats": True, "hist": [], "woe": False, "k": int(k), "ext": [], "nb": []}
    ext, hist, nb = set(), set(), set()
    for b in bundles:
        a = BUNDLES[b]["add"]
        if a.get("weather"):
            f["weather"] = True
        if a.get("woe"):
            f["woe"] = True
        ext |= set(a.get("ext", []))
        hist |= set(a.get("hist", []))
        nb |= set(a.get("nb", []))
    import e2fg
    f["ext"] = [b for b in e2fg.EXT_BLOCKS if b in ext]
    f["hist"] = [h for h in HIST_ORDER if h in hist]
    f["nb"] = [b for b in NB_ORDER if b in nb]
    return f


def column_bundles(names, cat_names, bundles):
    """칸 이름 -> 묶음 코드('base' = h·범주 4칸). 숫자 칸 목록·범주 칸 목록 각각."""
    import e2fg
    import e2nb
    mp = {"h": "base"}
    for b in bundles:
        a = BUNDLES[b]["add"]
        nm = []
        if a.get("weather"):
            nm += F1.feature_names("summary", L)
        for hs in a.get("hist", []):
            nm += hist_names([hs])
        if a.get("woe"):
            nm += woe_names()
        for eb in a.get("ext", []):
            nm += e2fg.ext_names([eb])
        for nb in a.get("nb", []):
            nm += e2nb.nb_names([nb])
        if BUNDLES[b].get("ae"):
            nm += AE_NAMES
        for n in nm:
            if n in mp:
                raise ValueError(f"칸 {n} 이 두 묶음에 있음")
            mp[n] = b
    num_b = [mp[n] for n in names]
    cat_b = ["base" if c in BASE_CATS else ("food" if c == "aw_label" else "?") for c in cat_names]
    if "?" in cat_b:
        raise ValueError(f"묶음을 모르는 범주 칸: {cat_names}")
    return num_b, cat_b


# ---------------------------------------------------------------- 자료 문맥
def load_ctx(log):
    """2차 Ctx2 (cfg3 감싸기 아래: 모든 행이 'train' 한 분할, 2차 기상 캐시 안 씀)."""
    ctx = Ctx2(False, log, use_cache=False, cache_dir=None)
    ctx.log = log
    return ctx


def fold_def(k):
    return G.FOLD_BY_K[k]


def limitations(bundles, k_info):
    """결과 기록용 한계 문구(누수 점검 결과)."""
    out = []
    lim = G.CFG.get("limits", {})
    if needs_k(bundles) and not k_info.get("selected_by_cv"):
        out.append(f"스무딩 계수 k = {k_info.get('k')} 고정: " + lim.get("k_fixed", "교차검증으로 고르지 않은 값(설정 smoothing_k.fixed)"))
    if "n1" in bundles or "n2" in bundles:
        out.append(f"제품명 뭉치 정의는 학습 자료 전체(mnfctur_de ≤ {G.BOUNDS['learn_end']})의 검출 수로 다시 고른 것 - 교차검증 겹 1·2·3 모두 "
                   f"맞혀 보는 해 검출이 정의에 섞임(겹 점수에서 제품명 칸이 조금 좋게 나올 수 있음, 시험은 해당 없음). 쓴 정의 파일의 '선정 기간' 칸: {nb_def_period()}")
    if any(b in bundles for b in PAST):
        out.append("과거 검사 결과 칸(과거 이력·WOE·제품명 과거 유형): 학습 행은 예측일 전날까지 자료, 맞혀 보는 행·시험 행은 그 겹(최종) 학습 끝 값으로 고정 "
                   "-> 그 해 뒤쪽 예측일일수록 오래된 값(운영과 같은 조건, 1년마다 다시 학습 전제). 결과 시차는 학습 범위에 반영하지 않음")
    if "food" in bundles or "rg" in bundles or "g123" in bundles:
        out.append("식품 속성 3단계 대표값 = 학습 기간 시료 최빈값(정답 안 씀, 기간 고정 표 - 그대로 둠)")
    if "ae4" in bundles:
        out.append("이상치 점수: 겹마다 그 겹 학습 정상 행으로 재학습(5겹 교차 적합 v2). 묶음 빼기 동안에는 처음 계산한 점수를 고정 묶음으로 씀(다시 계산 안 함)")
    if "n2" in bundles or "n0" in bundles:
        out.append("운영 때 그날 검사 제품명 입력 전제(그날 제품명 유형 n2·낱말 빈도 TF-IDF n0)")
    if "n0" in bundles:
        out.append("TF-IDF: 겹 학습 끝 날짜별 파일(제품명/단위날짜_tfidf_train<날짜>.parquet)이 없으면 하나짜리 판(전체 기간 사전·IDF, 정답 안 씀) 사용 - 실행 기록 tfidf 참고")
    return out


def nb_def_period():
    """제품명/뭉치_정의.csv 의 '선정 기간' 칸 값(서로 다른 값, 기록용). 파일·칸이 없으면 그 사실."""
    f = C1.DATA_DIR / "제품명" / "뭉치_정의.csv"
    try:
        d = pd.read_csv(f, encoding="utf-8-sig")
        return " / ".join(sorted(d["선정 기간"].astype(str).unique().tolist())) if "선정 기간" in d.columns else "('선정 기간' 칸 없음)"
    except Exception as e:  # 기록용
        return f"(읽지 못함: {e})"


class RowSet:
    """SP 분할 안 위치 pos 와 h 의 묶음. kind = 'train'(행마다 h 하나) | 'eval'(h 우선, h 안에서 단위 키+d 순) | 'test'.
    anchor = 과거 검사 결과 칸을 고정할 학습 끝 날(맞혀 보는 행 = 그 겹 학습 끝, 시험 행 = 최종 학습 끝), 학습 행은 None(굴러가는 창)."""

    def __init__(self, ctx, pos, h, kind, anchor=None):
        self.ctx, self.kind = ctx, kind
        self.anchor = None if anchor is None else np.datetime64(anchor, "D")
        if (kind == "train") != (self.anchor is None):
            raise ValueError(f"RowSet: 학습 행만 anchor 없음(kind={kind}, anchor={anchor})")
        self.pos = np.asarray(pos, dtype=np.int64)
        self.h = np.asarray(h, dtype=np.int16)
        self.n = len(self.pos)
        v = ctx.sp[SP]
        self.y = v["y"][self.pos].astype(np.int8)
        w = ctx.weather
        self.d = (w.date0 + v["d"][self.pos].astype("timedelta64[D]")).astype("datetime64[D]")
        self.T = self.d - self.h.astype("timedelta64[D]")
        self.s = v["s"][self.pos].astype(np.int64)                    # 기상 시군구 번호
        self.t = (v["d"][self.pos] - self.h.astype(np.int64)).astype(np.int64)   # T 의 기상 날짜 번호

    def blocks(self):
        """같은 h 가 이어진 구간 (a, b, h) - 맞혀 보는 행·시험 행은 h 마다 한 구간."""
        if self.kind == "train":
            return [(0, self.n, None)]
        out = []
        for hh in range(1, H_MAX + 1):
            idx = np.flatnonzero(self.h == hh)
            if len(idx):
                assert idx[-1] - idx[0] + 1 == len(idx)
                out.append((int(idx[0]), int(idx[-1]) + 1, hh))
        return out

    def h_offsets(self):
        b = self.blocks()
        assert len(b) == H_MAX, f"h 구간 {len(b)}개(60 이어야 함)"
        return np.array([(a, c) for a, c, _ in b], dtype=np.int64)

    def md5(self):
        return hashlib.md5(self.pos.tobytes() + self.h.tobytes() + str(self.anchor).encode()).hexdigest()

    def subset(self, idx):
        return RowSet(self.ctx, self.pos[idx], self.h[idx], self.kind, self.anchor)

    def past_h(self, a, b, h):
        """과거 검사 결과 칸(PAST)에 넘길 h (행 a:b): 학습 행 = h 그대로(예측일 T 전날까지 굴러가는 창),
        맞혀 보는 행·시험 행 = d − (anchor + 1일) -> 2차 계산의 창 'T−P .. T−1' 이 [anchor−P+1, anchor] 가 됨(학습 끝 값 고정, h·T 와 무관)."""
        if self.anchor is None:
            return h
        he = (self.d[a:b] - (self.anchor + np.timedelta64(1, "D"))).astype(np.int64)
        if len(he) and he.min() < 1:
            raise RuntimeError(f"맞혀 보는 행 기준일이 학습 끝 {self.anchor} + 1일 이하 - 경계 이상")
        return he

    def past_tag(self):
        """캐시 이름 꼬리표: 학습 행 '_pastR'(굴러가는 창), 맞혀 보는 행·시험 행 '_pastF<학습 끝 날짜>'."""
        return "_pastR" if self.anchor is None else "_pastF" + str(self.anchor).replace("-", "")

    def keys_frame(self):
        ctx = self.ctx
        codes = ctx.sp[SP]["cats"][self.pos]
        cols = {}
        for name, j, lev in (("sgg_code", 1, "sgg"), ("train_prdlst_code", 2, "p2"), ("train_prdlst_code_2", 3, "p3")):
            cols[name] = pa.DictionaryArray.from_arrays(pa.array(codes[:, j].astype(np.int32)), pa.array(ctx.levels[lev], type=pa.string()))
        cols["d"] = pa.array(self.d, type=pa.date32())
        cols["h"] = pa.array(self.h.astype(np.int8))
        cols["T"] = pa.array(self.T, type=pa.date32())
        cols["y"] = pa.array(self.y)
        return cols


class CV:
    def __init__(self, ctx, log, cache_dir, seed=None, smoke=None):
        self.ctx, self.log = ctx, log
        self.cache_dir = Path(cache_dir)
        self.seed = G.SEED if seed is None else int(seed)
        self.smoke = dict(G.SMOKE) if smoke is None else dict(smoke)
        w = ctx.weather
        v = ctx.sp[SP]
        self.date = (w.date0 + v["d"].astype("timedelta64[D]")).astype("datetime64[D]")
        self.N = len(self.date)
        self.D_end = np.datetime64(G.BOUNDS["data_end"], "D")
        self.learn_end = np.datetime64(G.BOUNDS["learn_end"], "D")
        self.keep = np.ones(self.N, dtype=bool)
        self.keep_desc = "전체"
        if self.smoke.get("enabled"):
            self._apply_smoke()
        self.hfit = self._h_assign(self.seed)
        self._cat_levels = None
        self.mem_cache = None                     # 묶음별 입력 칸 메모리 재사용(enable_mem_cache - 기상 거르기·k 고르기처럼 같은 칸을 여러 번 만들 때만)
        self.mem_cache_bytes = 0
        self.mem_cache_budget = 0

    def enable_mem_cache(self, gb):
        """같은 프로세스 안에서 묶음별 입력 칸(행 묶음·묶음·k·TF-IDF·과거 칸 규칙이 같으면 같은 값)을 메모리에 두고 다시 씀 - 디스크 캐시 대신. 값은 같음."""
        self.mem_cache, self.mem_cache_bytes, self.mem_cache_budget = {}, 0, float(gb) * 1e9

    # ---------------- 시험 실행용 부분 자료
    def _apply_smoke(self):
        s = self.smoke
        rng = np.random.default_rng(int(s.get("seed", 0)))
        sgg = np.array(sorted(self.ctx.levels["sgg"]))
        pick = set(rng.choice(sgg, size=min(int(s["n_sgg"]), len(sgg)), replace=False).tolist())
        cs = self.ctx.sp[SP]["cat_str"][:, 1].astype(str)
        self.keep = np.isin(cs, list(pick))
        y = self.ctx.sp[SP]["y"]
        lm = self.keep & (self.date <= self.learn_end)
        neg = np.flatnonzero(lm & (y == 0))
        nmax = int(s.get("train_neg_max", 10**12))
        if len(neg) > nmax:
            drop = rng.choice(neg, size=len(neg) - nmax, replace=False)
            self.keep_learn = self.keep.copy()
            self.keep_learn[drop] = False
        else:
            self.keep_learn = self.keep.copy()
        self.keep_desc = (f"시험 실행: 시군구 {len(pick)}개({sorted(pick)}), 학습 자료 불검출 최대 {nmax}행(검출 모두), "
                          f"맞혀 보는 행·시험 행 h 마다 불검출 최대 {s.get('eval_per_h_max')}행(검출 모두), seed {s.get('seed', 0)}")

    # ---------------- 학습 행 h (날짜 규칙, 분할 이름 안 씀)
    def _h_assign(self, seed):
        last_te = np.datetime64(G.FOLDS[-1]["train_end"], "D")
        h = np.zeros(self.N, dtype=np.int16)
        ma = self.date <= last_te
        mb = (self.date > last_te) & (self.date <= self.learn_end)
        h[ma] = np.random.default_rng(seed).integers(1, H_MAX + 1, size=int(ma.sum()))
        h[mb] = np.random.default_rng(seed + 1).integers(1, H_MAX + 1, size=int(mb.sum()))
        return h

    # ---------------- 행 고르기
    def train_rows(self, train_end) -> RowSet:
        end = np.datetime64(train_end, "D")
        assert end <= self.learn_end, "학습 행 끝이 학습 자료 끝을 넘음"
        keep = self.keep_learn if self.smoke.get("enabled") else self.keep
        pos = np.flatnonzero(keep & (self.date <= end))
        return RowSet(self.ctx, pos, self.hfit[pos], "train")

    def _t_rows(self, t_start, t_end, kind, anchor):
        ts, te = np.datetime64(t_start, "D"), np.datetime64(t_end, "D")
        pos_l, h_l = [], []
        rng = np.random.default_rng([int(self.smoke.get("seed", 0)), 7, 0 if kind == "eval" else 1])
        y = self.ctx.sp[SP]["y"]
        for hh in range(1, H_MAX + 1):
            T = self.date - np.timedelta64(hh, "D")
            m = self.keep & (T >= ts) & (T <= te) & (self.date <= self.D_end)
            pos = np.flatnonzero(m)
            if self.smoke.get("enabled"):
                neg = pos[y[pos] == 0]
                mx = int(self.smoke.get("eval_per_h_max", 10**12))
                if len(neg) > mx:
                    keepn = np.sort(rng.choice(neg, size=mx, replace=False))
                    pos = np.sort(np.concatenate([pos[y[pos] == 1], keepn]))
            pos_l.append(pos)
            h_l.append(np.full(len(pos), hh, dtype=np.int16))
        return RowSet(self.ctx, np.concatenate(pos_l), np.concatenate(h_l), kind, anchor)

    def eval_rows(self, k) -> RowSet:
        f = fold_def(k)
        rs = self._t_rows(f["t_start"], f["t_end"], "eval", f["train_end"])
        assert rs.d.max() <= np.datetime64(G.BOUNDS["cv_end"], "D"), "맞혀 보는 행이 시험 시작 전날을 넘음"
        return rs

    def fold_train(self, k) -> RowSet:
        return self.train_rows(fold_def(k)["train_end"])

    def learn_rows(self) -> RowSet:
        return self.train_rows(str(self.learn_end))

    def test_rowset(self, why, allow=False) -> RowSet:
        """시험 행(final 단계 전용 - ALLOW_TEST 또는 allow=True(점검의 위치 세기) 없이 부르면 오류로 멈춤). 만든 기록을 TEST_ACCESS 에 남김."""
        if not (ALLOW_TEST or allow):
            raise RuntimeError(f"고르는 단계에서 시험 행을 만들려고 함({why}) - 시험은 final 단계에서만")
        TEST_ACCESS.append({"why": why, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        t = G.BOUNDS["test"]
        return self._t_rows(t["t_start"], t["t_end"], "test", G.BOUNDS["learn_end"])

    # ---------------- 고위험 품목(겹마다 그 겹 학습 자료로)
    def top_items(self, train_end, n_top=None):
        n_top = int(n_top or G.CFG["group_items"]["top_n"])
        S = self.ctx.samples
        d = S["d"].to_numpy().astype("datetime64[D]")
        m = d <= np.datetime64(train_end, "D")
        g = S[m].groupby("train_prdlst_code_2", observed=True)["pos"].agg(["size", "sum"]).reset_index()
        g = g.sort_values(["sum", "train_prdlst_code_2"], ascending=[False, True]).head(n_top)
        tot = int(S.loc[m, "pos"].sum())
        return {"train_end": str(train_end), "codes": g["train_prdlst_code_2"].astype(str).tolist(),
                "pos": g["sum"].astype(int).tolist(), "n": g["size"].astype(int).tolist(),
                "share_of_pos": float(g["sum"].sum() / tot) if tot else float("nan"),
                "rule": f"기준일 <= {train_end} 시료의 품목군 3단계별 검출 수 내림차순(같으면 코드 순) 상위 {n_top}"}

    def item_mask(self, rs, codes):
        p3 = self.ctx.sp[SP]["p3"][rs.pos]
        return np.isin(p3.astype(str), list(codes))

    # ---------------- 범주
    def cat_levels(self, cat_names, fb):
        """범주 칸 목록(전체 행 기준 고정 목록). 기본 4칸 = ctx.levels, aw_label 등 추가 칸 = 전체 행 값 정렬."""
        lv = {}
        extra = [c for c in cat_names if c not in BASE_CATS]
        allv = None
        if extra:
            allv = np.asarray(fb.cats(SP, np.arange(self.N)))
        for j, c in enumerate(cat_names):
            if c in BASE_CATS:
                lv[c] = list(self.ctx.levels[{"sido": "sido", "sgg": "sgg", "p2": "p2", "p3": "p3"}[c]])
            else:
                lv[c] = sorted(np.unique(allv[:, j].astype(str)).tolist())
        return lv

    def cats(self, rs, fb, levels):
        """(cat_str (n, c) object, codes (n, c) int32)."""
        cs = np.asarray(fb.cats(SP, rs.pos))
        if cs.ndim == 1:
            cs = cs[:, None]
        codes = np.empty(cs.shape, dtype=np.int32)
        for j, c in enumerate(fb.cat_names):
            if c in BASE_CATS:
                codes[:, j] = self.ctx.sp[SP]["cats"][rs.pos, BASE_CATS.index(c)]
            else:
                codes[:, j] = pd.Categorical(cs[:, j].astype(str), categories=levels[c]).codes
        if (codes < 0).any():
            raise ValueError("범주 코드 -1")
        return np.ascontiguousarray(cs.astype(object)), codes

    # ---------------- 입력 행렬
    def builder(self, bundles, k, fold_file_tag=None):
        """FeatureBuilder(AE 제외). n0 이 있으면 그 겹의 TF-IDF 파일로 e2nb 경로를 바꾼 뒤 만든다."""
        if "n0" in bundles:
            self.set_tfidf(fold_file_tag)
        return FeatureBuilder(self.ctx, feats_of([b for b in bundles if not BUNDLES[b].get("ae")], k), seed=self.seed)

    def tfidf_file(self, train_end_tag):
        """겹 학습 끝 날짜(또는 'final') -> 단위날짜_tfidf 파일. 설정 data.nb_tfidf_by_train_end 또는 이름 규칙
        제품명/단위날짜_tfidf_train<날짜>.parquet. 없으면 하나짜리 판(제품명/단위날짜_tfidf.parquet)."""
        import e2nb
        nb_dir = C1.DATA_DIR / "제품명"
        mp = G.CFG["data"].get("nb_tfidf_by_train_end") or {}
        cand = []
        if train_end_tag in mp:
            cand.append(Path(mp[train_end_tag]))
        cand.append(nb_dir / f"단위날짜_tfidf_train{train_end_tag}.parquet")
        for c in cand:
            if c.exists():
                voc = c.with_name(c.name.replace("단위날짜_tfidf_", "단위날짜_tfidf_어휘_").replace(".parquet", ".csv"))
                if not voc.exists():
                    raise FileNotFoundError(f"TF-IDF 어휘 파일 없음: {voc.name}")
                return c, voc, True
        if G.CFG["data"].get("require_fold_tfidf", True):
            raise FileNotFoundError(f"겹별 TF-IDF 파일 없음(학습 끝 {train_end_tag}): {[str(c.name) for c in cand]} - 하나짜리 판으로 대신하지 않음(설정 data.require_fold_tfidf)")
        return nb_dir / "단위날짜_tfidf.parquet", nb_dir / "단위날짜_tfidf_어휘.csv", False

    def set_tfidf(self, tag):
        import e2nb
        f, voc, fold_specific = self.tfidf_file(tag)
        if e2nb.TFIDF_PATH != f:
            e2nb.TFIDF_PATH, e2nb.TFIDF_VOCAB_PATH = f, voc
            if getattr(self.ctx, "_nb", None) is not None:
                self.ctx._nb = None                     # N0 행렬을 새 파일로 다시 읽게
        self.tfidf_used = {"tag": tag, "file": f.name, "fold_specific": fold_specific}
        return self.tfidf_used

    @staticmethod
    def h_for(bundle, rs, a, b, h):
        """묶음별로 FeatureBuilder 에 넘길 h (행 a:b): 과거 검사 결과 칸(PAST)은 rs.past_h(학습 행 = h, 맞혀 보는 행·시험 행 = 학습 끝 고정용 h),
        그 밖은 h 그대로."""
        return rs.past_h(a, b, h) if bundle in PAST else h

    def bundle_num(self, fb_b, bundle, rs, a, b, h):
        """묶음 하나의 칸(h 칸 뺌) float32 (행 a:b). 과거 검사 결과 칸 규칙(PAST_RULE) 반영."""
        return fb_b.num(SP, rs.pos[a:b], self.h_for(bundle, rs, a, b, h))[:, 1:]

    def compute_X(self, rs, bundles, k, tfidf_tag=None):
        """캐시 없이 직접 계산(점검·작은 묶음용) - assemble_X 와 같은 칸 순서·같은 과거 칸 규칙. 돌려줌 (X, fb_full)."""
        bl = [b for b in resolve_featset(bundles) if not BUNDLES[b].get("ae")]
        fb_full = self.builder(bl, k, tfidf_tag)
        idx = {n: j for j, n in enumerate(fb_full.names)}
        X = np.empty((rs.n, len(fb_full.names)), dtype=np.float32)
        X[:, 0] = rs.h.astype(np.float32)
        for b in bl:
            fb_b = self.builder([b], k, tfidf_tag)
            cols = [idx[n] for n in fb_b.names[1:]]
            key = None
            if self.mem_cache is not None:
                key = (rs.md5(), b, int(k) if BUNDLES[b].get("k") else None, tfidf_tag if b == "n0" else None,
                       rs.past_tag() if b in PAST else None, L, tuple(fb_b.names))
                if key in self.mem_cache:
                    X[:, cols] = self.mem_cache[key]
                    continue
            for a, c, h in rs.blocks():
                X[a:c, cols] = self.bundle_num(fb_b, b, rs, a, c, h if h is not None else rs.h[a:c])
            if key is not None and self.mem_cache_bytes + rs.n * len(cols) * 4 <= self.mem_cache_budget:
                self.mem_cache[key] = np.ascontiguousarray(X[:, cols])
                self.mem_cache_bytes += rs.n * len(cols) * 4
        return X, fb_full

    def cached_X(self, rs, fb, tag, in_memory=True, drop_h=False, bundle=None):
        """입력 행렬 float32 파일(np.save 형식). 이름 = <tag>_<칸 지문>_L<L>.npy. meta(json) 가 다르면 다시 만듦.
        drop_h=True 면 첫 칸(h)을 빼고 저장(묶음별 캐시). bundle 이 과거 검사 결과 칸(PAST)이면 rs.past_h 규칙으로 계산."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        c0 = 1 if drop_h else 0
        sig = hashlib.md5(json.dumps({"names": fb.names, "feats": fb.feats, "drop_h": drop_h}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]
        f = self.cache_dir / f"{tag}_{sig}_L{L:03d}.npy"
        mf = f.with_suffix(".json")
        w = self.ctx.weather
        past = {"rule": PAST_RULE, "anchor": None if rs.anchor is None else str(rs.anchor)} if bundle in PAST else None
        meta = {"n": int(rs.n), "ncol": len(fb.names) - c0, "drop_h": drop_h, "rows_md5": rs.md5(), "L": L, "feats": fb.feats,
                "past_rule": past,
                "names_md5": hashlib.md5("|".join(fb.names).encode()).hexdigest(), "weather_shape": list(w.W.shape),
                "weather_date0": str(w.date0), "code_md5": feature_code_md5(), "data": data_stamp(),
                "tfidf": getattr(self, "tfidf_used", None) if fb.feats.get("nb") and "n0" in fb.feats["nb"] else None}
        ok = False
        if f.exists() and mf.exists():
            try:
                ok = json.loads(mf.read_text(encoding="utf-8")) == json.loads(json.dumps(meta, ensure_ascii=False))
            except Exception:
                ok = False
        if not ok:
            t0 = time.time()
            if mf.exists():
                mf.unlink()
            tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")          # 원자적 쓰기: 다른 프로세스가 반쯤 쓴 파일을 읽지 않게
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(rs.n, len(fb.names) - c0))
            for a, b, h in rs.blocks():
                for a2 in range(a, b, 200000):
                    b2 = min(b, a2 + 200000)
                    hh = h if h is not None else rs.h[a2:b2]
                    mm[a2:b2] = fb.num(SP, rs.pos[a2:b2], self.h_for(bundle, rs, a2, b2, hh))[:, c0:]
            mm.flush()
            del mm
            os.replace(tmp, f)
            mtmp = mf.with_name(f"{mf.stem}.tmp{os.getpid()}.json")
            mtmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            os.replace(mtmp, mf)
            self.log(f"  입력 캐시 생성 {f.name}: {rs.n:,}행 x {len(fb.names) - c0}칸 ({rs.n * (len(fb.names) - c0) * 4 / 1e9:.2f} GB, {time.time() - t0:.1f}s)")
        X = np.load(f, mmap_mode="r")
        assert X.shape == (rs.n, len(fb.names) - c0)
        return (np.array(X) if in_memory else X), f

    def assemble_X(self, rs, bundles, k, tag, tfidf_tag=None):
        """입력 행렬을 묶음별 캐시에서 모아 만든다(묶음마다 한 번 계산 -> k 고르기·튜닝·최종이 기상 요약 캐시를 같이 씀).
        칸 순서 = 그 묶음 전체로 만든 FeatureBuilder 의 칸 순서(h, 기상, 이력, WOE, 외부·생장, 제품명 - 2차 정의 순서).
        묶음별 계산 = 그 묶음만 넣은 FeatureBuilder(h + 그 묶음 칸)의 h 뒤 칸 - 행마다 독립 계산이라 한꺼번에 만든 값과 같음(selftest3 cache 확인).
        돌려줌 (X float32, fb_full).
        설정 disk_cache = false(기본)면 디스크에 쓰지 않고 메모리에서 바로 계산(compute_X, 같은 값 - selftest3 cache) - 팀 공용 서버 디스크 보호
        (10-20 22:46 디스크가 가득 차 실행이 멈춘 뒤 바꿈)."""
        if not G.CFG.get("disk_cache", False):
            return self.compute_X(rs, bundles, k, tfidf_tag)
        bl = [b for b in resolve_featset(bundles) if not BUNDLES[b].get("ae")]
        fb_full = self.builder(bl, k, tfidf_tag)
        idx = {n: j for j, n in enumerate(fb_full.names)}
        X = np.empty((rs.n, len(fb_full.names)), dtype=np.float32)
        X[:, 0] = rs.h.astype(np.float32)
        assert fb_full.names[0] == "h"
        filled = 1
        for b in bl:
            fb_b = self.builder([b], k, tfidf_tag)
            kt = f"_k{int(k)}" if BUNDLES[b].get("k") else ""
            tt = f"_tf{tfidf_tag}" if b == "n0" else ""
            lt = rs.past_tag() if b in PAST else ""
            Xb, _ = self.cached_X(rs, fb_b, f"{tag}_b-{b}{kt}{lt}{tt}", in_memory=False, drop_h=True, bundle=b)
            cols = [idx[n] for n in fb_b.names[1:]]
            for a in range(0, rs.n, 500000):
                X[a:a + 500000, cols] = Xb[a:a + 500000]
            filled += len(cols)
            del Xb
        assert filled == len(fb_full.names), f"칸 수 {filled} != {len(fb_full.names)}"
        return X, fb_full

    def check_windows(self, rsets):
        w = self.ctx.weather
        lo = min(int(r.t.min()) for r in rsets if r.n) - L
        hi = max(int(r.t.max()) for r in rsets if r.n) - 1
        assert lo >= 0 and hi <= w.D - 1, f"창이 기상표 밖: idx {lo}..{hi}"
        return str(w.date0 + np.timedelta64(lo, "D")), str(w.date0 + np.timedelta64(hi, "D"))


def feature_code_md5():
    fs = [G.E.EXP1_SRC / "features.py", G.E.EXP1_SRC / "data.py"] + [EXP2_SRC / f for f in
                                                                     ("e2data.py", "e2models.py", "e2hist.py", "e2fg.py", "e2nb.py")]
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in fs}


def data_stamp():
    out = {}
    for p in (C1.SAMPLE_PATH, C1.WEATHER_PATH):
        st = Path(p).stat()
        out[Path(p).name] = [int(st.st_size), int(st.st_mtime)]
    return out


MY_FILES = ("cfg3.py", "cv.py", "models3.py", "ae3.py", "engine3.py", "raw3.py", "dl3.py", "run3.py", "selftest3.py")


def code_md5():
    files = [SRC3 / f for f in MY_FILES] + [EXP2_SRC / f for f in ("e2cfg.py", "e2data.py", "e2models.py", "e2hist.py", "e2fg.py",
                                                               "e2nb.py", "e2ae.py")] \
        + [G.E.EXP1_SRC / f for f in ("config.py", "data.py", "features.py", "metrics.py", "models.py")]
    out = {}
    for f in files:
        if f.exists():
            tag = f.name if f.parent == SRC3 else (f"2차/{f.name}" if f.parent == EXP2_SRC else f"1차/{f.name}")
            out[tag] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------- 채점
def ap_fast(y, p):
    """sklearn average_precision_score 와 같은 식(값 차이는 더하는 순서에서 오는 1e-15 수준). 조기 종료 지표용."""
    o = np.argsort(-p)                      # 동점 순서는 AP 에 영향 없음(동점 묶음 끝에서만 셈) -> 빠른 정렬
    ps = p[o]
    tp = np.cumsum(y[o].astype(np.int64))
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tps = tp[ends].astype(np.float64)
    fps = (ends + 1).astype(np.float64) - tps
    if tps[-1] == 0:
        return 0.0
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ap60_fast(y, p, offs):
    """h 별 AP 의 h 1..60 단순 평균(빠른 계산) - 조기 종료 지표(지금 코드 그대로)."""
    return float(np.mean([ap_fast(y[a:b], p[a:b]) for a, b in offs]))


def ap60(y, p, offs):
    return float(np.mean([M1.pr_auc(y[a:b], p[a:b]) for a, b in offs]))


H_BANDS = [(int(a), int(b), float(w)) for a, b, w in G.E.H_BANDS]      # (1,20,0.5), (21,40,0.3), (41,60,0.2) - 2차와 같은 가중치
assert H_BANDS == [(1, 20, 0.5), (21, 40, 0.3), (41, 60, 0.2)], H_BANDS
BAND_DESC = "0.5 × 평균(H1~20) + 0.3 × 평균(H21~40) + 0.2 × 평균(H41~60)"


def band_weighted(v):
    """h 1..60 값(길이 60, h 순서) -> 구간 가중값 = 0.5 × 평균(H1~20) + 0.3 × 평균(H21~40) + 0.2 × 평균(H41~60).
    구간 안에서 값이 없는 h(검출 또는 불검출이 없는 h)는 그 구간 평균에서 뺌."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) != H_MAX:
        raise ValueError(f"구간 가중: 값 {len(v)}개(60 이어야 함)")
    return float(sum(w * np.nanmean(v[a - 1:b]) for a, b, w in H_BANDS))


def h_pr_auc(y, p, offs):
    """h 별 PR-AUC(sklearn average precision) 60개 - 검출·불검출이 다 있는 h 만 값, 아니면 NaN."""
    out = []
    for a, b in offs:
        yy = y[a:b]
        out.append(M1.pr_auc(yy, p[a:b]) if 0 < int(yy.sum()) < len(yy) else np.nan)
    return np.array(out)


def wpr(y, p, offs):
    """고르기 점수(겹 하나): 구간 가중 PR-AUC = band_weighted(h 별 PR-AUC)."""
    return band_weighted(h_pr_auc(np.asarray(y), np.asarray(p), offs))


def topk_recall(yy, pp, k):
    """2차 analysis/T기준_통합/rescore_Tint.py 와 같음: 확률 내림차순(같으면 행 순서) 상위 ceil(n x k/100) 행의 검출 ÷ 그 h 검출."""
    pos = int(yy.sum())
    if pos == 0:
        return np.nan
    m = -(-len(yy) * k // 100)
    o = np.argsort(-pp, kind="stable")[:m]
    return float(yy[o].sum() / pos)


def inspect_reduction(tp, fp, fn, n):
    """검사축소율 = 1 − (TP + FP + FN) / N (사용자 정의)."""
    return 1.0 - (tp + fp + fn) / n if n else np.nan


THR_CFG = G.CFG.get("threshold", {"rule": "max_precision_at_recall", "recall_min": 0.80})
RECALL_MIN = float(THR_CFG["recall_min"])
THRESHOLD_RULE = f"max_precision_at_recall>={RECALL_MIN:.2f}"
THRESHOLD_DESC = (f"판정 기준값 = 60개 h 합친 예측에서 Recall ≥ {RECALL_MIN:.2f} 를 지키는 기준값 가운데 Precision 이 가장 큰 값"
                  "(같은 Precision 이면 Recall 이 큰 쪽), 판정 = 예측 확률 ≥ 기준값")


SCORE_VERSION = "v4: 고르기 = 구간 가중 PR-AUC(0.5·0.3·0.2), 기준값 = 60개 h 합친 예측에서 Recall>=0.80 중 Precision 최대"
METRIC_KEYS = ("wpr_auc", "pr_auc", "p_at_r", "r_at_thr", "wrecall", "precision", "recall", "f1", "accuracy", "roc_auc", "inspect_reduction",
               "top1", "top5", "top10", "top20")
OBJECTIVES = list(G.CFG.get("optuna_objectives") or ["wpr_auc"])
SELECT_KEY = "wpr_auc"                                   # 고르기 점수(사양 3절) - 바닐라 순위·k·튜닝 목표·변수 빼기·묶음 섞기 중요도
if OBJECTIVES != [SELECT_KEY]:
    raise ValueError(f"optuna_objectives = {OBJECTIVES}: 사양 3절 고르기 점수는 구간 가중 PR-AUC 하나(['wpr_auc'])")
METRIC_NAMES = {"wpr_auc": "구간 가중 PR-AUC", "pr_auc": "60일 평균 PR-AUC", "p_at_r": f"Recall≥{RECALL_MIN:.2f} 에서의 Precision(60개 h 합친 행)",
                "r_at_thr": "기준값 Recall(60개 h 합친 행)", "wrecall": "구간 가중 Recall", "precision": "Precision", "recall": "Recall",
                "f1": "F1", "accuracy": "Accuracy", "roc_auc": "ROC-AUC", "inspect_reduction": "검사축소율",
                "top1": "상위 1%", "top5": "상위 5%", "top10": "상위 10%", "top20": "상위 20%"}


def threshold_at_recall(y, p, r_min=None):
    """Recall ≥ r_min 인 기준값(sklearn precision_recall_curve 의 서로 다른 확률값) 중 Precision 최대, 같으면 Recall 큰 쪽.
    가장 낮은 기준값(모든 행 검출 판정)은 Recall = 1 이라 후보가 늘 있음."""
    from sklearn.metrics import precision_recall_curve
    r_min = RECALL_MIN if r_min is None else float(r_min)
    y = np.asarray(y)
    if not (0 < int(y.sum()) < len(y)):
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "rule": THRESHOLD_RULE, "n_candidates": 0}
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]                      # 마지막 점(Precision 1, Recall 0)은 기준값 없음
    ok = np.flatnonzero(rec >= r_min - 1e-12)
    if not len(ok):
        raise RuntimeError(f"Recall >= {r_min} 를 지키는 기준값이 없음(있을 수 없음)")
    i = ok[np.lexsort((-rec[ok], -prec[ok]))[0]]
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i]), "rule": THRESHOLD_RULE,
            "n_candidates": int(len(ok))}


def per_h(y, p, offs, thr):
    recs = []
    for hh, (a, b) in enumerate(offs, start=1):
        yy, pp = y[a:b], p[a:b]
        ok = 0 < int(yy.sum()) < len(yy)
        cm = M1.confusion(yy, pp, thr)
        recs.append({"h": hh, "n": int(b - a), "pos": int(yy.sum()),
                     "pr_auc": M1.pr_auc(yy, pp) if ok else np.nan, "roc_auc": M1.roc_auc(yy, pp) if ok else np.nan,
                     "threshold": thr, **{c: cm[c] for c in ("tp", "fp", "fn", "tn", "n", "precision", "recall", "f1", "accuracy")},
                     "inspect_reduction": inspect_reduction(cm["tp"], cm["fp"], cm["fn"], cm["n"]),
                     **{f"top{k}": topk_recall(yy, pp, k) for k in TOPK}})
    return pd.DataFrame(recs)


def _summarize(df):
    """h 별 표 -> h 평균 지표 + 구간 가중 PR-AUC·Recall."""
    return {"wpr_auc": band_weighted(df["pr_auc"].to_numpy()), "pr_auc": float(np.nanmean(df["pr_auc"])),
            "wrecall": band_weighted(df["recall"].to_numpy()),
            **{c: float(np.nanmean(df[c])) for c in ("precision", "recall", "f1", "accuracy", "roc_auc", "inspect_reduction",
                                                     "tp", "fp", "fn", "tn", "n")},
            **{f"top{k}": float(np.nanmean(df[f"top{k}"])) for k in TOPK}}


def score_fold(y, p, offs):
    """겹 점수(기준값 규칙 THRESHOLD_RULE).
    - wpr_auc(고르기 점수) = h 별 PR-AUC 의 구간 가중값(0.5·0.3·0.2), pr_auc = h 1..60 단순 평균, ROC-AUC = h 1..60 평균
    - 기준값 = 60개 h 합친 예측에서 Recall ≥ 목표를 지키는 기준값 가운데 Precision 최대(같으면 Recall 큰 쪽)
    - p_at_r = 그 기준값에서 60개 h 합친 Precision, r_at_thr = 그때 합친 Recall(≥ 목표)
    - Accuracy·Precision·Recall·F1·검사축소율·상위 k% = 그 기준값에서 h 마다 -> h 평균(Recall 은 구간 가중값 wrecall 도), TP·FP·FN·TN·N = h 당 평균"""
    y = np.asarray(y).astype(np.int8)
    thr = threshold_at_recall(y, p)
    t = thr["threshold"]
    df = per_h(y, p, offs, t)
    pooled = M1.confusion(y, p, t)
    s = {**_summarize(df), "p_at_r": float(pooled["precision"]), "r_at_thr": float(pooled["recall"]),
         "threshold": float(t), "threshold_rule": THRESHOLD_RULE, "score_version": SCORE_VERSION,
         "n_rows": int(len(y)), "n_pos": int(y.sum()), "h_without_both_classes": [int(r.h) for r in df.itertuples() if not (0 < r.pos < r.n)]}
    if abs(s["p_at_r"] - thr["precision"]) > 1e-12 or s["r_at_thr"] < RECALL_MIN - 1e-12:
        raise RuntimeError(f"기준값 점검 실패: {s['p_at_r']} vs {thr}")
    return s, df


def score_test(y, p, offs, thr):
    """시험 표: 고정 기준값(세 겹 맞혀 보는 행 예측 합친 것에서 정한 값)에서 score_fold 와 같은 지표. 60개 h 합친 Precision·Recall 도 같이(p_pooled·r_pooled)."""
    y = np.asarray(y).astype(np.int8)
    df = per_h(y, p, offs, thr)
    s = _summarize(df)
    s["h_without_both_classes"] = [int(r.h) for r in df.itertuples() if not (0 < r.pos < r.n)]   # 그 h 는 PR-AUC·ROC-AUC·상위 k% 평균에서 빠짐
    cm = M1.confusion(y, p, thr)
    s.update({"threshold": float(thr), "threshold_rule": THRESHOLD_RULE, "p_pooled": float(cm["precision"]), "r_pooled": float(cm["recall"])})
    return s, df


def offsets_from_h(h):
    """저장 예측의 h 칸(h 우선 연속) -> (60, 2) 구간."""
    h = np.asarray(h)
    offs = []
    for hh in range(1, H_MAX + 1):
        idx = np.flatnonzero(h == hh)
        assert len(idx) and idx[-1] - idx[0] + 1 == len(idx)
        offs.append((idx[0], idx[-1] + 1))
    return np.array(offs, dtype=np.int64)


SCORE_RULE = {
    "selection": f"고르기 점수(바닐라 순위·k·튜닝 목표·변수 빼기·묶음 섞기 중요도) = 교차검증 세 겹 평균 구간 가중 PR-AUC = {BAND_DESC} (h 마다 PR-AUC)",
    "threshold_rule": THRESHOLD_RULE,
    "threshold": THRESHOLD_DESC + " - 겹마다 그 겹 맞혀 보는 행으로, 시험용은 세 겹 맞혀 보는 행 예측을 합친 것에서 같은 규칙",
    "wpr_auc": f"겹마다 h 별 average precision(sklearn) -> {BAND_DESC} (검출·불검출이 다 있지 않은 h 는 그 구간 평균에서 뺌) -> 겹 평균",
    "pr_auc": "겹마다 h 별 average precision(sklearn) -> h 1..60 단순 평균 -> 겹 평균 (60일 평균 PR-AUC)",
    "p_at_r": f"겹마다 그 기준값에서 60개 h 합친 Precision(= Recall ≥ {RECALL_MIN:.2f} 에서의 최대 Precision) -> 겹 평균",
    "wrecall": f"그 기준값에서 h 마다 Recall -> {BAND_DESC} -> 겹 평균",
    "metrics": "Accuracy·Precision·Recall·F1·검사축소율(= 1 − (TP+FP+FN)/N)·상위 k%·ROC-AUC = 그 기준값에서 h 마다 -> h 평균 -> 겹 평균, TP·FP·FN·TN·N = h 당 평균 -> 겹 평균. 검출(또는 불검출)이 없는 h 는 PR-AUC·ROC-AUC·상위 k% 평균에서 빠짐(h_without_both_classes 에 기록)",
    "topk": "h 마다 확률 내림차순(같으면 행 순서) 상위 ceil(행 수 x k/100) 행의 검출 ÷ 그 h 검출 (k = 1·5·10·20)",
    "early_stopping": "부스팅 3종·BiLSTM: 그 겹 맞혀 보는 행 h 별 average precision 의 h 1..60 평균(빠른 계산 ap60_fast)을 매 반복(에폭), 인내 100(BiLSTM 5) - 지금 코드 그대로",
    "test": "시험: 기준값 = 세 겹 맞혀 보는 행 예측을 합친 것에서 같은 규칙. 점수 = h 1~60 평균·구간 가중, 건수 = h 당 평균",
    "past_rule": PAST_RULE_DESC,
}


def write_pred(path, rs, p, fold):
    cols = rs.keys_frame()
    cols["p"] = pa.array(np.asarray(p, dtype=np.float32))
    cols["fold"] = pa.array(np.full(rs.n, fold, dtype=np.int8))
    pq.write_table(pa.table(cols), path, compression="zstd")


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")
    os.replace(tmp, path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
