# -*- coding: utf-8 -*-
"""compare_samples.py — 옛 시료 파일과 새 시료 파일을 나란히 비교한다. 두 파일은 읽기만 한다(값을 바꾸거나 다시 저장하지 않음).

실행 (pyarrow 와 표준 라이브러리만 씀 — pandas 안 씀. 아래 둘 다 됨)
  C:\\Users\\NT751\\Desktop\\천영진\\리스테리아ML\\.venv-exp2\\Scripts\\python.exe -I compare_samples.py <옛 파일> <새 파일> [--out 결과폴더]
  py -I compare_samples.py <옛 파일> <새 파일> [--out 결과폴더]
  --out     결과 폴더. 없으면 exp\\3_시간순교차검증\\비교결과_<날짜_시각>\\ 에 쓴다.
  --krc     법정동 표(기본 리스테리아ML2\\data\\raw\\raw_korea_region_code.parquet). 새 법정동 코드가 표에 있는지 본다. 없으면 건너뜀.
  --weather 시군구 기상표(기본 리스테리아ML2\\data\\데이터셋\\식정원_db_기상기후\\시군구_일별_기상.parquet). 끝 날짜만 읽는다. 없으면 건너뜀.

결과 (csv 는 utf-8-sig — 엑셀에서 바로 열림, md 는 utf-8)
  비교_요약.md            아래 표 전부 + 자동 알림(사람이 판단할 곳)
  비교_칼럼.csv           칼럼 이름·형식 나란히
  비교_기간별.csv         기준일(mnfctur_de) 구간별 행·검출(전체·국내·해외) + 한쪽 파일에만 있는 시료 수
  비교_연도별.csv         기준일 연도별 행·검출
  비교_범주값.csv         작은 범주 칼럼(judge_result·src_route 등)의 값별 행 수
  비교_품목코드.csv       train_prdlst_code·train_prdlst_code_2 코드별 행·검출, 한쪽에만 있는 코드 표시
  비교_법정동코드.csv     국내 행의 legal_dong_code 가운데 새 파일에만 있는 코드(행·검출·주소 예시·법정동 표 상태)
  비교_겹친키_값차이.csv  두 파일에 모두 있는 lims_data_key 에서 칼럼별로 값이 다른 행 수
  비교_값차이_예시.csv    값이 다른 행 예시(칼럼마다 최대 20개)

규칙
  - 값은 그대로 둔다. 비교할 때만 범주형(dictionary)을 글자로, 기준일을 날짜로 읽는다(글자 날짜면 몇 가지 꼴로 읽어 보고 못 읽은 수를 적음).
  - 검출 = judge_result 가 '검출'인 행. 국내·해외 = src_route 가 'domestic'·'overseas' 인 행.
  - 겹친 키 비교는 lims_data_key 가 각 파일에서 한 번만 나오는 시료끼리만 한다(두 번 이상 나오는 키는 수만 적음).
"""
import argparse
import csv
import datetime as dt
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T0 = time.time()
EXP3_DIR = Path(__file__).resolve().parents[1]
ML2 = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
DEF_KRC = ML2 / "data" / "raw" / "raw_korea_region_code.parquet"
DEF_WEATHER = ML2 / "data" / "데이터셋" / "식정원_db_기상기후" / "시군구_일별_기상.parquet"

KEY = "lims_data_key"
# 통계용으로 처음에 읽는 칼럼(있는 것만). 겹친 키 값 비교는 두 파일에 다 있는 칼럼 전부를 하나씩 따로 읽는다.
CORE = [KEY, "judge_result", "src_route", "mnfctur_de", "train_prdlst_code", "train_prdlst_nm",
        "train_prdlst_code_2", "train_prdlst_nm_2", "addr", "legal_dong_code", "lat", "lon",
        "date_basis", "addr_source", "data_source", "hazard_label", "expr_iem_nm"]
CAT_SMALL = ["judge_result", "src_route", "date_basis", "addr_source", "data_source", "hazard_label", "expr_iem_nm"]
CODE_LEVELS = [("train_prdlst_code", "train_prdlst_nm", "2단계"), ("train_prdlst_code_2", "train_prdlst_nm_2", "3단계")]
# 기준일 구간: 시간순 교차검증 겹과 지금 시험 해(2025-07-01~2026-06-30), 그 밖
PERIODS = [
    ("~2022-06-30", None, "2022-06-30"),
    ("2022-07-01~2023-06-30", "2022-07-01", "2023-06-30"),
    ("2023-07-01~2024-06-30", "2023-07-01", "2024-06-30"),
    ("2024-07-01~2025-06-30", "2024-07-01", "2025-06-30"),
    ("2025-07-01~2026-06-30 (지금 시험 해)", "2025-07-01", "2026-06-30"),
    ("2026-07-01~ (지금 코드의 기간 밖)", "2026-07-01", None),
]
TEST_END = dt.date(2026, 6, 30)       # 1차 config.py TEST_END (넘는 행이 있으면 1차 data.assign_split 이 멈춤)
N_EXAMPLE = 20
MISSING = "칼럼 없음"
EMPTY = "(빈 값)"


def log(*a):
    print(f"[{time.time() - T0:5.1f}초]", *a, flush=True)


# ---------------------------------------------------------------- 형 맞추기 (비교용 사본만)
def decode(col):
    """범주형(dictionary)이면 값 형식으로 푼다."""
    if pa.types.is_dictionary(col.type):
        return col.cast(col.type.value_type)
    return col


def as_text(col):
    col = decode(col)
    if pa.types.is_string(col.type):
        return col
    return col.cast(pa.string())


def as_ts(col):
    """기준일 → timestamp[ms]. (배열, 못 읽은 수). 글자면 몇 가지 꼴로 읽어 본다."""
    col = decode(col)
    t = col.type
    if pa.types.is_timestamp(t):
        if t.tz is not None:
            col = pc.local_timestamp(col)
        return col.cast(pa.timestamp("ms")), 0
    if pa.types.is_date(t):
        return col.cast(pa.timestamp("ms")), 0
    s = col.cast(pa.string())
    tries = [pc.strptime(s, format=f, unit="ms", error_is_null=True)
             for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y/%m/%d")]
    out = pc.coalesce(*tries)
    bad = int(pc.sum(pc.and_(pc.is_valid(s), pc.is_null(out))).as_py() or 0)
    return out, bad


def norm_cmp(col):
    """겹친 키 값 비교용: 실수는 float64, 정수는 int64, 날짜·시각은 timestamp[ms], 나머지는 글자."""
    col = decode(col)
    t = col.type
    if pa.types.is_floating(t):
        return col.cast(pa.float64())
    if pa.types.is_integer(t):
        return col.cast(pa.int64())
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        return as_ts(col)[0]
    if pa.types.is_boolean(t):
        return col
    return col.cast(pa.string())


def to_np_bool(arr):
    return np.asarray(arr.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


# ---------------------------------------------------------------- 파일 하나 읽고 세기
class Sample:
    def __init__(self, path, label):
        self.path = Path(path)
        self.label = label
        self.pf = pq.ParquetFile(self.path)
        self.schema = self.pf.schema_arrow
        self.names = self.schema.names
        self.n = self.pf.metadata.num_rows
        st = self.path.stat()
        self.size_mb = st.st_size / 1e6
        self.mtime = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        cols = [c for c in CORE if c in self.names]
        t = self.pf.read(columns=cols)
        self.c = {c: t[c] for c in cols}
        del t
        self.s = {}            # 한눈에 표에 들어갈 값
        self._count()

    def has(self, c):
        return c in self.c

    def text(self, c):
        return as_text(self.c[c]) if self.has(c) else None

    def _count(self):
        n, s = self.n, self.s
        s["행 수"] = n
        s["칼럼 수"] = len(self.names)
        # 키
        if self.has(KEY):
            k = self.text(KEY)
            self.key = k
            nn = k.null_count
            dist = pc.count_distinct(k, mode="only_valid").as_py()
            s["lims_data_key 빈 값"] = nn
            s["lims_data_key 종류"] = dist
            s["lims_data_key 두 번 이상 나온 행"] = (n - nn) - dist
        else:
            self.key = None
            for x in ("lims_data_key 빈 값", "lims_data_key 종류", "lims_data_key 두 번 이상 나온 행"):
                s[x] = MISSING
        # 판정·경로
        j = self.text("judge_result")
        r = self.text("src_route")
        self.det = to_np_bool(pc.equal(j, "검출")) if j is not None else np.zeros(n, bool)
        nondet = to_np_bool(pc.equal(j, "불검출")) if j is not None else np.zeros(n, bool)
        self.dom = to_np_bool(pc.equal(r, "domestic")) if r is not None else np.zeros(n, bool)
        self.ovs = to_np_bool(pc.equal(r, "overseas")) if r is not None else np.zeros(n, bool)
        s["검출"] = int(self.det.sum()) if j is not None else MISSING
        s["불검출"] = int(nondet.sum()) if j is not None else MISSING
        s["judge_result 그 밖 값(빈 값 포함)"] = int(n - self.det.sum() - nondet.sum()) if j is not None else MISSING
        s["국내(domestic) 행"] = int(self.dom.sum()) if r is not None else MISSING
        s["국내 검출"] = int((self.dom & self.det).sum()) if r is not None else MISSING
        s["해외(overseas) 행"] = int(self.ovs.sum()) if r is not None else MISSING
        s["해외 검출"] = int((self.ovs & self.det).sum()) if r is not None else MISSING
        s["src_route 그 밖 값(빈 값 포함)"] = int(n - self.dom.sum() - self.ovs.sum()) if r is not None else MISSING
        # 기준일
        if self.has("mnfctur_de"):
            ts, bad = as_ts(self.c["mnfctur_de"])
            ms = ts.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.float64)   # 빈 값 = NaN
            ok = ~np.isnan(ms)
            self.days = np.where(ok, np.floor(ms / 86_400_000), np.nan)               # 1970-01-01 부터 며칠째
            s["기준일 형식"] = str(self.schema.field("mnfctur_de").type)
            s["기준일 빈 값(못 읽은 것 포함)"] = int((~ok).sum())
            s["기준일 글자라 못 읽은 행"] = bad
            s["기준일 가장 이른 날"] = str(self.day(np.nanmin(self.days))) if ok.any() else ""
            s["기준일 가장 늦은 날"] = str(self.day(np.nanmax(self.days))) if ok.any() else ""
            s["기준일에 시각(0시 아님)이 있는 행"] = int((ok & (np.mod(np.where(ok, ms, 0), 86_400_000) != 0)).sum())
            s["기준일 > 2026-06-30 (1차 config 시험 끝 뒤) 행"] = int((self.days > self.dnum(TEST_END)).sum())
            s["국내 기준일 가장 이른 날"] = str(self.day(np.nanmin(self.days[self.dom]))) if (ok & self.dom).any() else ""
            s["국내 기준일 가장 늦은 날"] = str(self.day(np.nanmax(self.days[self.dom]))) if (ok & self.dom).any() else ""
        else:
            self.days = np.full(n, np.nan)
            for x in ("기준일 형식", "기준일 빈 값(못 읽은 것 포함)", "기준일 글자라 못 읽은 행", "기준일 가장 이른 날",
                      "기준일 가장 늦은 날", "기준일에 시각(0시 아님)이 있는 행", "기준일 > 2026-06-30 (1차 config 시험 끝 뒤) 행",
                      "국내 기준일 가장 이른 날", "국내 기준일 가장 늦은 날"):
                s[x] = MISSING
        # 품목 코드
        self.codes = {}
        for code_c, nm_c, lv in CODE_LEVELS:
            if not self.has(code_c):
                s[f"{lv} 코드({code_c}) 종류"] = MISSING
                s[f"{lv} 코드 빈 값"] = MISSING
                s[f"{lv} 코드 하나에 이름 여러 개"] = MISSING
                continue
            codes = self.text(code_c).to_pylist()
            names = self.text(nm_c).to_pylist() if self.has(nm_c) else [None] * n
            rows, det, drows, ddet = Counter(), Counter(), Counter(), Counter()
            nm = defaultdict(set)
            for i, (cd, na) in enumerate(zip(codes, names)):
                key = cd if cd not in (None, "") else EMPTY
                rows[key] += 1
                if self.det[i]:
                    det[key] += 1
                if self.dom[i]:
                    drows[key] += 1
                    if self.det[i]:
                        ddet[key] += 1
                nm[key].add(na)
            self.codes[code_c] = dict(rows=rows, det=det, drows=drows, ddet=ddet,
                                      name={k: " / ".join(sorted(str(x) for x in v if x is not None)) for k, v in nm.items()})
            s[f"{lv} 코드({code_c}) 종류"] = len([k for k in rows if k != EMPTY])
            s[f"{lv} 코드 빈 값"] = rows.get(EMPTY, 0)
            s[f"{lv} 코드 하나에 이름 여러 개"] = sum(1 for k, v in nm.items() if k != EMPTY and len({x for x in v if x is not None}) > 1)
        # 주소 (국내 행 — 시군구 배정은 국내 행만 함)
        if self.has("addr"):
            a = self.text("addr")
            empty = to_np_bool(pc.or_kleene(pc.is_null(a), pc.equal(pc.utf8_trim_whitespace(a), "")))
            s["국내 addr 빈 값"] = int((empty & self.dom).sum())
        else:
            s["국내 addr 빈 값"] = MISSING
        self.ldc = None
        if self.has("legal_dong_code"):
            lc = self.text("legal_dong_code").to_pylist()
            addr = self.text("addr").to_pylist() if self.has("addr") else [None] * n
            rows, det, ex = Counter(), Counter(), {}
            bad = 0
            for i in np.flatnonzero(self.dom):
                cd = lc[i]
                if cd is None or len(cd) != 10 or not cd.isdigit():
                    bad += 1
                key = cd if cd not in (None, "") else EMPTY
                rows[key] += 1
                if self.det[i]:
                    det[key] += 1
                if key not in ex:
                    ex[key] = addr[i]
            self.ldc = dict(rows=rows, det=det, ex=ex)
            s["국내 legal_dong_code 빈 값·10자리 숫자 아님"] = bad
            s["국내 legal_dong_code 종류"] = len([k for k in rows if k != EMPTY])
        else:
            s["국내 legal_dong_code 빈 값·10자리 숫자 아님"] = MISSING
            s["국내 legal_dong_code 종류"] = MISSING
        for c in ("lat", "lon"):
            if self.has(c):
                v = decode(self.c[c])
                nulls = to_np_bool(pc.is_null(v, nan_is_null=True))
                s[f"국내 {c} 빈 값"] = int((nulls & self.dom).sum())
            else:
                s[f"국내 {c} 빈 값"] = MISSING
        # 작은 범주 칼럼 값
        self.cats = {}
        for c in CAT_SMALL:
            if self.has(c):
                self.cats[c] = Counter(x if x is not None else EMPTY for x in self.text(c).to_pylist())

    @staticmethod
    def day(x):
        return (dt.date(1970, 1, 1) + dt.timedelta(days=int(x))) if x is not None and not np.isnan(x) else None

    @staticmethod
    def dnum(d):
        return (d - dt.date(1970, 1, 1)).days

    def period_mask(self, lo, hi):
        m = ~np.isnan(self.days)
        if lo:
            m &= self.days >= self.dnum(dt.date.fromisoformat(lo))
        if hi:
            m &= self.days <= self.dnum(dt.date.fromisoformat(hi))
        return m

    def years(self):
        ok = ~np.isnan(self.days)
        y = np.full(self.n, -1, dtype=np.int64)
        y[ok] = self.days[ok].astype("int64").astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
        return y


# ---------------------------------------------------------------- 겹친 키
def key_table(smp):
    """한 번만 나오는 키 → (키, 행 번호) 표, 두 번 이상 나온 키 종류 수"""
    k = smp.key.combine_chunks() if isinstance(smp.key, pa.ChunkedArray) else smp.key
    vc = pc.value_counts(k.drop_null())
    dup = pc.filter(vc.field("values"), pc.greater(vc.field("counts"), 1))
    t = pa.table({"k": k, "i": pa.array(np.arange(len(k), dtype=np.int64))})
    m = pc.and_(pc.is_valid(t["k"]), pc.invert(pc.is_in(t["k"], value_set=dup)))
    return t.filter(m), len(dup)


def compare_keys(old, new):
    ko, dup_o = key_table(old)
    kn, dup_n = key_table(new)
    both = ko.join(kn, keys="k", join_type="inner", left_suffix="_o", right_suffix="_n").sort_by("i_o")
    only_o = ko.join(kn, keys="k", join_type="left anti")
    only_n = kn.join(ko, keys="k", join_type="left anti")
    io = both["i_o"].to_numpy()
    inn = both["i_n"].to_numpy()
    res = dict(dup_o=dup_o, dup_n=dup_n, both=len(io),
               only_o=only_o["i"].to_numpy(), only_n=only_n["i"].to_numpy(), io=io, inn=inn)
    # 칼럼별 값 비교 (두 파일에 다 있는 칼럼, 하나씩 읽음)
    common = [c for c in old.names if c in new.names and c != KEY]
    diff_rows, examples = [], []
    keys = both["k"].to_pylist()
    for c in common:
        a = norm_cmp(old.pf.read(columns=[c])[c])
        b = norm_cmp(new.pf.read(columns=[c])[c])
        cast_note = ""
        if a.type != b.type:
            a, b = a.cast(pa.string()), b.cast(pa.string())
            cast_note = "형식이 달라 글자로 비교"
        a = a.take(pa.array(io))
        b = b.take(pa.array(inn))
        same = pc.or_kleene(pc.equal(a, b), pc.and_(pc.is_null(a), pc.is_null(b)))
        if pa.types.is_floating(a.type):
            same = pc.or_kleene(same, pc.and_(pc.is_nan(a), pc.is_nan(b)))
        same = to_np_bool(same)
        nd = int((~same).sum())
        dd = int((~same & old.det[io]).sum()) if len(io) else 0
        diff_rows.append({"칼럼": c, "겹친 키 행": len(io), "값이 다른 행": nd, "그 가운데 옛 파일 검출 행": dd, "비고": cast_note})
        if nd:
            idx = np.flatnonzero(~same)[:N_EXAMPLE]
            av, bv = a.take(pa.array(idx)).to_pylist(), b.take(pa.array(idx)).to_pylist()
            for j, x, y in zip(idx, av, bv):
                examples.append({"칼럼": c, KEY: keys[j], "옛 값": "" if x is None else str(x), "새 값": "" if y is None else str(y)})
        del a, b
    res["diff_rows"] = diff_rows
    res["examples"] = examples
    return res


# ---------------------------------------------------------------- 쓰기
def write_csv(path, rows, head=None):
    head = head or (list(rows[0].keys()) if rows else ["(없음)"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_table(rows, head=None, limit=None):
    if not rows:
        return "(없음)\n"
    head = head or list(rows[0].keys())
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows[:limit] if limit else rows:
        out.append("| " + " | ".join(fmt(r.get(h)).replace("|", "/") for h in head) + " |")
    if limit and len(rows) > limit:
        out.append(f"\n(앞 {limit}줄만 — 전체는 csv)")
    return "\n".join(out) + "\n"


def diff_val(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return b - a
    return "같음" if a == b else "다름"


def main():
    ap = argparse.ArgumentParser(description="옛·새 시료 파일 비교(읽기만)")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", default=None)
    ap.add_argument("--krc", default=str(DEF_KRC))
    ap.add_argument("--weather", default=str(DEF_WEATHER))
    args = ap.parse_args()
    out = Path(args.out) if args.out else EXP3_DIR / f"비교결과_{dt.datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)

    log("옛 파일 읽기:", args.old)
    old = Sample(args.old, "옛")
    log("새 파일 읽기:", args.new)
    new = Sample(args.new, "새")

    # 1. 한눈에
    summary = []
    for k in old.s:
        a, b = old.s[k], new.s.get(k, MISSING)
        summary.append({"항목": k, "옛 파일": a, "새 파일": b, "차이(새−옛)": diff_val(a, b)})

    # 2. 칼럼
    col_rows = []
    allc = list(old.names) + [c for c in new.names if c not in old.names]
    for c in allc:
        to = str(old.schema.field(c).type) if c in old.names else ""
        tn = str(new.schema.field(c).type) if c in new.names else ""
        state = "새 파일에 없음" if not tn else "새 파일에만 있음" if not to else "같음" if to == tn else "형식 바뀜"
        col_rows.append({"칼럼": c, "옛 순서": old.names.index(c) + 1 if c in old.names else "",
                         "새 순서": new.names.index(c) + 1 if c in new.names else "", "옛 형식": to, "새 형식": tn, "상태": state})
    write_csv(out / "비교_칼럼.csv", col_rows)

    # 3. 겹친 키
    keyres = None
    if old.key is not None and new.key is not None:
        log("겹친 키 값 비교")
        keyres = compare_keys(old, new)
        write_csv(out / "비교_겹친키_값차이.csv", keyres["diff_rows"])
        write_csv(out / "비교_값차이_예시.csv", keyres["examples"], head=["칼럼", KEY, "옛 값", "새 값"])

    # 4. 기간별
    per_rows = []
    only_o = np.zeros(old.n, bool)
    only_n = np.zeros(new.n, bool)
    if keyres is not None:
        only_o[keyres["only_o"]] = True
        only_n[keyres["only_n"]] = True
    for lab, lo, hi in PERIODS + [("기준일 빈 값", "NA", "NA")]:
        r = {"기준일 구간": lab}
        for smp, tag, only in ((old, "옛", only_o), (new, "새", only_n)):
            m = np.isnan(smp.days) if lo == "NA" else smp.period_mask(lo, hi)
            r[f"{tag} 행"] = int(m.sum())
            r[f"{tag} 검출"] = int((m & smp.det).sum())
            r[f"{tag} 국내 행"] = int((m & smp.dom).sum())
            r[f"{tag} 국내 검출"] = int((m & smp.dom & smp.det).sum())
            r[f"{tag} 해외 행"] = int((m & smp.ovs).sum())
            r[f"{tag} 해외 검출"] = int((m & smp.ovs & smp.det).sum())
            r[f"{tag} 파일에만 있는 시료"] = int((m & only).sum())
            r[f"{tag} 파일에만 있는 시료 검출"] = int((m & only & smp.det).sum())
        r["행 차이(새−옛)"] = r["새 행"] - r["옛 행"]
        r["검출 차이(새−옛)"] = r["새 검출"] - r["옛 검출"]
        per_rows.append(r)
    write_csv(out / "비교_기간별.csv", per_rows)

    yo, yn = old.years(), new.years()
    year_rows = []
    for y in sorted(set(yo.tolist()) | set(yn.tolist())):
        r = {"연도": str(y) if y >= 0 else "기준일 빈 값"}
        for smp, tag, yy in ((old, "옛", yo), (new, "새", yn)):
            m = yy == y
            r[f"{tag} 행"] = int(m.sum())
            r[f"{tag} 검출"] = int((m & smp.det).sum())
            r[f"{tag} 국내 행"] = int((m & smp.dom).sum())
            r[f"{tag} 국내 검출"] = int((m & smp.dom & smp.det).sum())
        r["행 차이(새−옛)"] = r["새 행"] - r["옛 행"]
        r["검출 차이(새−옛)"] = r["새 검출"] - r["옛 검출"]
        year_rows.append(r)
    write_csv(out / "비교_연도별.csv", year_rows)

    # 5. 범주 값
    cat_rows = []
    for c in CAT_SMALL:
        co, cn = old.cats.get(c), new.cats.get(c)
        vals = sorted(set(co or {}) | set(cn or {}), key=lambda v: (-(co or {}).get(v, 0), str(v)))
        for v in vals:
            cat_rows.append({"칼럼": c, "값": v, "옛 행": (co or {}).get(v, 0) if co is not None else MISSING,
                             "새 행": (cn or {}).get(v, 0) if cn is not None else MISSING})
    write_csv(out / "비교_범주값.csv", cat_rows)

    # 6. 품목 코드
    code_rows, code_sum = [], []
    for code_c, nm_c, lv in CODE_LEVELS:
        a, b = old.codes.get(code_c), new.codes.get(code_c)
        if a is None or b is None:
            code_sum.append({"단계": lv, "칼럼": code_c, "새 파일에만 있는 코드": MISSING, "그 코드 행": "", "그 코드 검출": "",
                             "옛 파일에만 있는 코드": MISSING, "그 코드 행(옛)": "", "그 코드 검출(옛)": ""})
            continue
        keys = sorted(set(a["rows"]) | set(b["rows"]))
        n_new = r_new = d_new = n_old = r_old = d_old = 0
        for k in keys:
            state = "둘 다" if k in a["rows"] and k in b["rows"] else "새 파일에만" if k in b["rows"] else "옛 파일에만"
            if state == "새 파일에만" and k != EMPTY:
                n_new += 1
                r_new += b["rows"][k]
                d_new += b["det"].get(k, 0)
            if state == "옛 파일에만" and k != EMPTY:
                n_old += 1
                r_old += a["rows"][k]
                d_old += a["det"].get(k, 0)
            code_rows.append({"단계": lv, "코드": k, "이름(새, 없으면 옛)": b["name"].get(k) or a["name"].get(k, ""),
                              "구분": state, "옛 행": a["rows"].get(k, 0), "옛 검출": a["det"].get(k, 0),
                              "새 행": b["rows"].get(k, 0), "새 검출": b["det"].get(k, 0),
                              "옛 국내 행": a["drows"].get(k, 0), "옛 국내 검출": a["ddet"].get(k, 0),
                              "새 국내 행": b["drows"].get(k, 0), "새 국내 검출": b["ddet"].get(k, 0)})
        code_sum.append({"단계": lv, "칼럼": code_c, "새 파일에만 있는 코드": n_new, "그 코드 행": r_new, "그 코드 검출": d_new,
                         "옛 파일에만 있는 코드": n_old, "그 코드 행(옛)": r_old, "그 코드 검출(옛)": d_old})
    write_csv(out / "비교_품목코드.csv", code_rows,
              head=["단계", "코드", "이름(새, 없으면 옛)", "구분", "옛 행", "옛 검출", "새 행", "새 검출",
                    "옛 국내 행", "옛 국내 검출", "새 국내 행", "새 국내 검출"])

    # 7. 법정동 코드 (국내, 새 파일에만)
    krc = {}
    krc_note = ""
    if args.krc and Path(args.krc).exists():
        kt = pq.read_table(args.krc, columns=["legal_dong_code", "molit_use_yn"])
        krc = dict(zip(as_text(kt["legal_dong_code"]).to_pylist(), as_text(kt["molit_use_yn"]).to_pylist()))
        krc_note = f"법정동 표: {args.krc} ({len(krc):,}행)"
    else:
        krc_note = "법정동 표 없음 — 상태 칸 비움"
    ldc_rows = []
    if old.ldc is not None and new.ldc is not None:
        for k in sorted(set(new.ldc["rows"]) - set(old.ldc["rows"])):
            ldc_rows.append({"legal_dong_code": k, "새 국내 행": new.ldc["rows"][k], "새 국내 검출": new.ldc["det"].get(k, 0),
                             "법정동 표 상태(molit_use_yn)": (krc.get(k, "표에 없음") if krc else ""),
                             "주소 예시": new.ldc["ex"].get(k) or ""})
    write_csv(out / "비교_법정동코드.csv", ldc_rows,
              head=["legal_dong_code", "새 국내 행", "새 국내 검출", "법정동 표 상태(molit_use_yn)", "주소 예시"])

    # 8. 기상표 끝 날짜
    wx_end = None
    if args.weather and Path(args.weather).exists():
        wx_end = pc.max(pq.read_table(args.weather, columns=["date"])["date"]).as_py()

    # 9. 자동 알림 (판단할 곳 — 정하지는 않음)
    flags = []
    gone = [r["칼럼"] for r in col_rows if r["상태"] == "새 파일에 없음"]
    added = [r["칼럼"] for r in col_rows if r["상태"] == "새 파일에만 있음"]
    retyped = [f"{r['칼럼']}({r['옛 형식']}→{r['새 형식']})" for r in col_rows if r["상태"] == "형식 바뀜"]
    if gone:
        flags.append(f"새 파일에 없는 칼럼: {', '.join(gone)} → 이 칼럼을 읽는 스크립트가 멈춤(순서표 ① 칼럼 표 참고).")
    if added:
        flags.append(f"새 파일에만 있는 칼럼: {', '.join(added)} → 02_시군구_배정.py 는 원래 칼럼을 모두 넘기므로 시료_시군구.parquet 에도 생김. 쓸지 말지 판단.")
    if retyped:
        flags.append(f"형식이 바뀐 칼럼: {', '.join(retyped)} → 날짜·코드 칼럼이면 스크립트 동작 확인.")
    if old.s.get("기준일 가장 이른 날") and new.s.get("기준일 가장 이른 날") and new.s["기준일 가장 이른 날"] < old.s["기준일 가장 이른 날"]:
        flags.append(f"기준일 시작이 당겨짐({old.s['기준일 가장 이른 날']} → {new.s['기준일 가장 이른 날']}) → 03 기상표 시작일(가장 이른 기준일 − 365일)이 바뀜 → 03·04·05·생장 범위 일별 표 다시.")
    n_after = new.s.get("기준일 > 2026-06-30 (1차 config 시험 끝 뒤) 행")
    if isinstance(n_after, int) and n_after > 0:
        flags.append(f"기준일이 2026-06-30 보다 늦은 행 {n_after:,}개 → 1차 config.py TEST_END 를 넘음(1차 data.assign_split 이 '분할 기간 밖 행' 으로 멈춤). 시험 기간 고정 규칙과 함께 판단 필요.")
    if wx_end and new.s.get("기준일 가장 늦은 날"):
        last = dt.date.fromisoformat(new.s["기준일 가장 늦은 날"])
        # 기상 창 = T−L … T−1, T = d − h(h ≥ 1) → 필요한 마지막 날 = 가장 늦은 기준일 − 2일
        if last - dt.timedelta(days=2) > wx_end:
            flags.append(f"가장 늦은 기준일 {last} 에 필요한 기상 날({last - dt.timedelta(days=2)})이 지금 기상표 끝({wx_end}) 뒤 → 01(DB 원자료 다시 받기)·03 이후 다시.")
    for cs in code_sum:
        if isinstance(cs["새 파일에만 있는 코드"], int) and cs["새 파일에만 있는 코드"] > 0:
            flags.append(f"새 파일에만 있는 {cs['단계']} 코드 {cs['새 파일에만 있는 코드']}개(행 {cs['그 코드 행']:,}·검출 {cs['그 코드 검출']:,}) → "
                         + ("06 식품 속성·생장 범위 품목표에 새 줄, HACCP 업종-품목 목록(C 로 시작하는 2단계)도 확인." if cs["단계"] == "2단계"
                            else "06 식품 속성 표에 새 줄(다시 만들면 생김) → 생장 범위 품목표도 다시 요청."))
    if ldc_rows:
        n_nokrc = sum(1 for r in ldc_rows if r["법정동 표 상태(molit_use_yn)"] == "표에 없음")
        flags.append(f"국내 행에서 새 파일에만 있는 legal_dong_code {len(ldc_rows)}개(그 가운데 법정동 표에 없는 코드 {n_nokrc}개) → "
                     "02_시군구_배정.py 가 시군구를 못 정하면 삭제 사유(A~D)에 안 맞아 멈출 수 있음(비교_법정동코드.csv).")
    if keyres is not None:
        changed = [r for r in keyres["diff_rows"] if r["값이 다른 행"] > 0]
        if changed:
            flags.append("두 파일에 다 있는 시료에서 값이 바뀐 칼럼: " + ", ".join(f"{r['칼럼']} {r['값이 다른 행']:,}행" for r in changed)
                         + " → 옛 판과 겹치는 기간의 결과도 달라짐.")
        if len(keyres["only_o"]):
            flags.append(f"옛 파일에만 있는 시료 {len(keyres['only_o']):,}행(검출 {int(old.det[keyres['only_o']].sum()):,}) — 새 파일에서 빠짐(기간별 표 참고).")
        if keyres["dup_o"] or keyres["dup_n"]:
            flags.append(f"lims_data_key 가 두 번 이상 나온 키: 옛 {keyres['dup_o']:,}종 · 새 {keyres['dup_n']:,}종 → 비교에서 뺐음. 제품명 시료별 파일은 키 순서로 맞추므로 확인 필요.")
    if not flags:
        flags.append("자동 알림 없음(칼럼·기간·코드·키 값이 옛 파일과 같은 범위).")

    # 10. md
    L = []
    L.append("# 시료 파일 비교 (옛 vs 새)\n")
    L.append(f"- 만든 때: {dt.datetime.now():%Y-%m-%d %H:%M} · 스크립트: `{Path(__file__).resolve()}`")
    L.append(f"- 옛 파일: `{old.path}` ({old.size_mb:.1f}MB, 수정 {old.mtime})")
    L.append(f"- 새 파일: `{new.path}` ({new.size_mb:.1f}MB, 수정 {new.mtime})")
    L.append(f"- {krc_note}" + (f" · 지금 시군구 기상표 끝 날짜: {wx_end}" if wx_end else ""))
    L.append("- 값은 바꾸지 않음. 검출 = judge_result '검출', 국내·해외 = src_route 'domestic'·'overseas'.\n")
    L.append("## 자동 알림 (판단할 곳)\n")
    L.extend(f"{i}. {f}" for i, f in enumerate(flags, 1))
    L.append("\n## 1. 한눈에\n")
    L.append(md_table(summary, ["항목", "옛 파일", "새 파일", "차이(새−옛)"]))
    L.append("## 2. 칼럼 (같지 않은 것만 — 전체는 비교_칼럼.csv)\n")
    L.append(md_table([r for r in col_rows if r["상태"] != "같음"]))
    L.append(f"같은 칼럼 {sum(r['상태'] == '같음' for r in col_rows)}개 / 전체 {len(col_rows)}개\n")
    L.append("## 3. 기준일(mnfctur_de) 구간별 행·검출\n")
    L.append(md_table(per_rows, ["기준일 구간", "옛 행", "새 행", "행 차이(새−옛)", "옛 검출", "새 검출", "검출 차이(새−옛)",
                                 "옛 국내 행", "새 국내 행", "옛 국내 검출", "새 국내 검출", "옛 해외 행", "새 해외 행",
                                 "옛 파일에만 있는 시료", "새 파일에만 있는 시료", "새 파일에만 있는 시료 검출"]))
    L.append("## 4. 연도별 (기준일 연도)\n")
    L.append(md_table(year_rows))
    L.append("## 5. 범주 칼럼 값\n")
    L.append(md_table(cat_rows, limit=60))
    L.append("## 6. 품목 코드\n")
    L.append(md_table(code_sum))
    newonly = [r for r in code_rows if r["구분"] == "새 파일에만"]
    oldonly = [r for r in code_rows if r["구분"] == "옛 파일에만"]
    L.append("**새 파일에만 있는 코드**\n")
    L.append(md_table(sorted(newonly, key=lambda r: -r["새 행"]), ["단계", "코드", "이름(새, 없으면 옛)", "새 행", "새 검출", "새 국내 행"], limit=30))
    L.append("**옛 파일에만 있는 코드**\n")
    L.append(md_table(sorted(oldonly, key=lambda r: -r["옛 행"]), ["단계", "코드", "이름(새, 없으면 옛)", "옛 행", "옛 검출", "옛 국내 행"], limit=30))
    L.append("## 7. 국내 행 법정동 코드 — 새 파일에만 있는 것\n")
    L.append(md_table(sorted(ldc_rows, key=lambda r: -r["새 국내 행"]), limit=30))
    L.append("## 8. 두 파일에 다 있는 시료(lims_data_key)\n")
    if keyres is not None:
        L.append(md_table([
            {"항목": "두 파일에 다 있는 키(한 번만 나오는 키끼리)", "값": keyres["both"]},
            {"항목": "옛 파일에만 있는 키", "값": len(keyres["only_o"])},
            {"항목": "옛 파일에만 있는 키 — 검출", "값": int(old.det[keyres["only_o"]].sum())},
            {"항목": "새 파일에만 있는 키", "값": len(keyres["only_n"])},
            {"항목": "새 파일에만 있는 키 — 검출", "값": int(new.det[keyres["only_n"]].sum())},
            {"항목": "두 번 이상 나온 키 종류(옛 / 새)", "값": f"{keyres['dup_o']:,} / {keyres['dup_n']:,}"},
        ]))
        L.append("**칼럼별 값이 다른 행 (두 파일에 다 있는 키)**\n")
        L.append(md_table(keyres["diff_rows"]))
        if keyres["examples"]:
            L.append("예시는 비교_값차이_예시.csv\n")
    else:
        L.append("lims_data_key 칼럼이 한쪽에 없어 비교하지 않음.\n")
    L.append(f"\n(걸린 시간 {time.time() - T0:.1f}초)\n")
    (out / "비교_요약.md").write_text("\n".join(L), encoding="utf-8")
    log("결과 폴더:", out)
    for f in sorted(out.iterdir()):
        log("  ", f.name, f"{f.stat().st_size:,} bytes")
    log("자동 알림:")
    for f in flags:
        print("  -", f)


if __name__ == "__main__":
    main()
