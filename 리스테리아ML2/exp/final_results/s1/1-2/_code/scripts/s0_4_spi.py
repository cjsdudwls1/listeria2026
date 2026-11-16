# -*- coding: utf-8 -*-
"""s0_4_spi.py — 단계 0-4 SPI3·SPI6(계획서 §6 단계 0-4, §4-5, §5 WX1, 부록 A-2).

하는 일(계획서 그대로):
  1) 국내 관측 칸용 = 기상청 SPI 일자료(66지점 = ASOS 지점)의 시군구 대표 좌표 가까운 3지점 평균(기상청 값은 원값 그대로).
  2) 국내 예보 칸용 = 우리 식: 가까운 ASOS 3곳 일강수(`sum_rn`, 빈 값 = 무강수 0mm — 부록 A-2) 평균 → 시군구마다 감마 적합.
  3) 해외 = 우리 식: 나라 강수(DB 값, 해외 나라 일별 기상표 rain, 2015-01-01 부터)로 나라마다(관측·예보 같은 값).
  4) 대조(§6 0-4 "국내 관측 날은 우리 식 값과 기상청 값을 같은 날·지점으로 대조해 기록"): 기상청 66지점마다 그 지점
     ASOS `sum_rn` 으로 우리 식 SPI 를 계산해 같은 날 기상청 값과 비교(지점별·전체 상관·MAE), 그리고 시군구 단위로
     1) 표와 2) 표를 같은 날 비교(T 에서 이어 붙일 때 생기는 차이).
  5) 감마 모수 표(단위·달력일마다), 점검.json, run_meta.json.
  우리 식 = lst/spi.py(직전 n 개월 누적 → 달력일 ±15일 감마(Thom, 0 강수 확률 따로) → 표준정규, 적합 기간 고정).
  가까운 3곳 = lst/kweather.py(§4-3 규칙: 날마다 그날 값 있는 관측소 중 가까운 3곳 단순 평균, 관측소 위치 이력 반영).

출력(<out>/, 이미 있으면 멈춤) — 일별 표는 lst/windows.DailyCube.from_long 로 바로 읽힘:
  SPI_국내_관측_0-4.parquet   sgg_code·date·spi3·spi6 (기상청 값 3지점 평균) — 관측 칸 표(cube)
  SPI_국내_예보_0-4.parquet   sgg_code·date·spi3·spi6 (우리 식) — window 함수의 forecast_cube 로(이음 위치 T)
  SPI_해외_0-4.parquet        sgg('OV'+alpha_3)·alpha_3·date·spi3·spi6 (우리 식, 관측·예보 같은 표)
  SPI_감마모수_0-4.parquet    kind·unit·months·cal_day·n_pool·n_zero·n_pos·q·A·alpha·beta·status
  SPI_대조_0-4.csv            level(지점·시군구)·unit·months·기간·n·r·mae·mean_diff
  점검.json · run_meta.json(code_md5·입력 md5·스레드·설정)
  예: cube = DailyCube.from_long(pd.read_parquet(관측), "sgg_code", "date", ["spi3", "spi6"]),
      fc = DailyCube.from_long(pd.read_parquet(예보), "sgg_code", "date", ["spi3", "spi6"]),
      window_features(cube, sgg, d, h, W, mode, forecast_cube=fc).

구현 세부(계획서에 없음 — 담당자 확인):
  - 우리 식 세부는 lst/spi.py 머리 설명(누적 시작일 규칙·결측·달력일·못 맞춘 묶음·H 0/1).
  - 기상청 3지점 평균: 계획서 문구는 "시군구 대표 좌표 가까운 3지점 평균"뿐 — §4-3 규칙(날마다 그날 그 칼럼 값이 있는 지점
    중 가까운 3곳, 거리 상한 없음, 3곳이 안 되면 있는 곳만, 없으면 NaN, 지점 위치 = ASOS 지점정보의 그날 위치)을
    SPI3·SPI6 칼럼마다 따로 적용. 빠진 날 채우지 않음. 평균은 float32 로 저장되는 kweather 결과 그대로.
  - 국내 예보 칸 강수 = kweather 의 ASOS sum_rn 규칙(그날 행이 있는 ASOS 관측소 전부가 후보, 빈 sum_rn = 0mm) 그대로.
    날짜 축 = rain_start(2015-01-01) ~ ASOS 원자료 마지막 날.
  - 국내 관측 표 날짜 축 = 기상청 파일 날짜 범위 그대로(2014년 값도 원값으로 둠 — 국내 행은 RR8 로 S ≥ 2015-01-01).
  - 대조 '같은 지점' = 기상청 지점 번호 = ASOS stn_id 로 그 지점 하나의 sum_rn(3곳 평균 아님)으로 우리 식 계산, 같은 적합
    기간·규칙. 기간 = 전체·적합 기간 안·적합 기간 뒤로 나눠 기록. 시군구 단위 대조는 1)·2) 표 그대로.
  - 해외 단위 = sgg('OV'+alpha_3) — 단계 0-1 시료 sgg 와 같은 코드.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import kweather as kw  # noqa: E402
from lst import spi as sp  # noqa: E402
from lst.folds import day  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402
from lst.windows import DailyCube  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SPI_VARS = ("spi3", "spi6")
SCHEMA = {
    "plan": cf.Field(str),
    "inputs": {k: cf.Field(str) for k in ("kma_spi", "asos_daily", "asos_station")},
    "stage0_inputs": {k: cf.Field(str) for k in ("sgg_coord", "overseas_weather")},
    "kma_columns": {k: cf.Field(str) for k in SPI_VARS},
    "months": cf.Field(list, item_types=int),
    "pool_half_days": cf.Field(int),
    "fit_start": cf.Field(str),
    "fit_end": cf.Field(str),
    "rain_start": cf.Field(str),
    "first_valid": {k: cf.Field(str) for k in SPI_VARS},
}
OUT_NAMES = {
    "kma": "SPI_국내_관측_0-4.parquet",
    "dom": "SPI_국내_예보_0-4.parquet",
    "ov": "SPI_해외_0-4.parquet",
    "gamma": "SPI_감마모수_0-4.parquet",
    "cmp_csv": "SPI_대조_0-4.csv",
    "check": "점검.json",
    "meta": "run_meta.json",
}
KMA_HEADER = ["지점", "지점명", "일시", "SPI1", "SPI3", "SPI6", "SPI12"]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- 읽기
def read_kma_spi(path, kma_columns: dict) -> pa.Table:
    """기상청 SPI 일자료(cp949) → stn_id(str)·meas_date(date32)·spi3·spi6(float64, 빈 칸 = null). 값은 그대로."""
    df = pd.read_csv(path, encoding="cp949", dtype=str, keep_default_na=False)
    if list(df.columns) != KMA_HEADER:
        raise ValueError(f"기상청 SPI 머리줄이 다름: {list(df.columns)}")
    if not df["일시"].map(lambda s: bool(DATE_RE.match(s))).all():
        raise ValueError("기상청 SPI 일시 형식이 YYYY-MM-DD 가 아닌 행이 있음")
    if (df["지점"].str.strip() == "").any() or not df["지점"].str.fullmatch(r"\d+").all():
        raise ValueError("기상청 SPI 지점 번호가 숫자가 아닌 행이 있음")
    cols = {"stn_id": pa.array(df["지점"].tolist(), pa.string()),
            "meas_date": pa.array(pd.to_datetime(df["일시"], format="%Y-%m-%d").dt.date, pa.date32())}
    for v, src in kma_columns.items():
        s = df[src].str.strip()
        vals = pd.to_numeric(s.where(s != "", None), errors="raise").astype(np.float64)
        cols[v] = pa.array(vals.to_numpy(), pa.float64(), from_pandas=True)
    return pa.table(cols)


def table_buffer(t: pa.Table) -> pa.BufferReader:
    """kweather.load_asos 가 파일처럼 읽게 메모리 parquet(디스크에 안 씀)."""
    sink = pa.BufferOutputStream()
    pq.write_table(t, sink)
    return pa.BufferReader(sink.getvalue())


def grid_of(dmin, dmax) -> kw.DateGrid:
    return kw.DateGrid(np.datetime64(dmin, "D"), np.datetime64(dmax, "D"))


# ---------------------------------------------------------------- 쓰기
def write_parquet(path: Path, df: pd.DataFrame, meta: dict) -> int:
    if path.exists():
        raise FileExistsError(f"덮어쓰기 금지: {path}")
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.replace_schema_metadata({k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).encode("utf-8")
                                   for k, v in meta.items()})
    tmp = path.with_name(path.name + ".part")
    pq.write_table(t, tmp)
    tmp.replace(path)
    back = pq.read_metadata(path)
    assert back.num_rows == t.num_rows
    return t.num_rows


def long_frame(unit_col: str, units, d0: int, mats: dict, extra: dict | None = None) -> pd.DataFrame:
    """(단위, 날짜) 행렬 → 긴 표(단위·날짜 정렬, NaN = null)."""
    n_u = len(units)
    n_d = next(iter(mats.values())).shape[1]
    out = {unit_col: np.repeat(np.array(units, dtype=object), n_d)}
    if extra:
        for k, m in extra.items():
            out[k] = np.repeat(np.array([m[u] for u in units], dtype=object), n_d)
    out["date"] = np.tile((d0 + np.arange(n_d)).astype("datetime64[D]"), n_u)
    for k, M in mats.items():
        assert M.shape == (n_u, n_d), (k, M.shape)
        out[k] = M.astype(np.float64).ravel()
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def gamma_frame(kind: str, units, months: int, fit: sp.GammaFit) -> pd.DataFrame:
    U = len(units)
    c = np.arange(1, sp.N_CAL + 1)
    return pd.DataFrame({
        "kind": kind, "unit": np.repeat(np.array(units, dtype=object), sp.N_CAL), "months": np.int8(months),
        "cal_day": np.tile(c, U).astype(np.int16),
        "n_pool": fit.n_pool.ravel(), "n_zero": fit.n_zero.ravel(), "n_pos": fit.n_pos.ravel(),
        "q": fit.q.ravel(), "A": fit.A.ravel(), "alpha": fit.alpha.ravel(), "beta": fit.beta.ravel(),
        "status": np.array(sp.STATUS, dtype=object)[fit.status.ravel()],
    })


# ---------------------------------------------------------------- 점검
def knn_summary(res: dict, grid: kw.DateGrid) -> dict:
    D = res["dist"][~np.isnan(res["dist"])].astype(np.float64)
    has = ~np.isnan(res["value"])
    return {"칸": int(has.size), "결측 칸": int((~has).sum()), "모든 시군구 결측인 날 수": int((~has.any(axis=0)).sum()),
            "값 있는 칸 중 3곳 미만 칸": int(((res["count"] < kw.K_NEAREST) & has).sum()),
            "고른 지점까지 평균 거리 중앙값(km)": round(float(np.median(D)), 3) if D.size else None,
            "고른 지점까지 평균 거리 90%(km)": round(float(np.percentile(D, 90)), 3) if D.size else None,
            "고른 지점까지 평균 거리 최대(km)": round(float(D.max()), 3) if D.size else None,
            "날짜": f"{grid.date_min} ~ {grid.date_max}"}


def spi_summary(r: sp.SpiResult, d0: int, fit_end: int, expect_first: int) -> dict:
    fv = sp.first_valid_day(r.spi, d0)
    if fv is not None and fv < expect_first:
        raise RuntimeError(f"SPI{r.months}: 계획서 첫 값 날짜({np.datetime64(expect_first, 'D')}) 보다 이른 값 {np.datetime64(fv, 'D')}")
    days = d0 + np.arange(r.spi.shape[1])
    after = days >= expect_first
    inside = after & (days <= fit_end)
    st = {s: int((r.fit.status == i).sum()) for i, s in enumerate(sp.STATUS)}
    pool = r.fit.n_pool[r.fit.status == 0]
    return {"값 있는 첫 날": str(np.datetime64(fv, "D")) if fv is not None else None,
            "계획서 첫 값 날짜": str(np.datetime64(expect_first, "D")),
            "첫 값 날짜 이후 결측 칸": int(np.isnan(r.spi[:, after]).sum()),
            "첫 값 날짜 ~ 적합 끝 결측 칸": int(np.isnan(r.spi[:, inside]).sum()),
            "누적 강수 결측 칸(첫 값 날짜 이후)": int(np.isnan(r.X[:, after]).sum()),
            "H 0·1 이라 NaN 칸": r.n_edge, "감마 묶음 상태(단위 × 달력일)": st,
            "묶음 크기 최소·중앙·최대(ok)": [int(pool.min()), float(np.median(pool)), int(pool.max())] if pool.size else None,
            "q 최대": float(np.nanmax(r.fit.q)) if np.isfinite(r.fit.q).any() else None,
            "값 범위": [float(np.nanmin(r.spi)), float(np.nanmax(r.spi))] if np.isfinite(r.spi).any() else None}


def period_masks(days, fit_start, fit_end) -> dict:
    return {"전체": np.ones(days.shape, bool), "적합 기간 안": (days >= fit_start) & (days <= fit_end),
            "적합 기간 뒤": days > fit_end}


def overlap(a0: int, na: int, b0: int, nb: int):
    lo, hi = max(a0, b0), min(a0 + na, b0 + nb)
    if hi <= lo:
        raise ValueError("대조할 겹치는 날짜 없음")
    return slice(lo - a0, hi - a0), slice(lo - b0, hi - b0), np.arange(lo, hi, dtype=np.int64)


# ---------------------------------------------------------------- 본체
def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True, help="단계 0 결과 뿌리(서버 ~/cyj_listeria_final/results)")
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_4_spi.json"))
    rp, args = parse_paths(argv, ap)
    cfg = cf.load_config(args.config, SCHEMA)
    if sorted(cfg["months"]) != [3, 6] or len(cfg["months"]) != 2:
        raise cf.ConfigError(f"months 는 계획서 §4-5 대로 [3, 6]: {cfg['months']}")
    in_root = Path(args.inputs).expanduser().resolve()
    if not in_root.is_dir():
        raise FileNotFoundError(f"--inputs 폴더 없음: {in_root}")
    out = {k: rp.result(v) for k, v in OUT_NAMES.items()}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    paths = {k: rp.data(v) for k, v in cfg["inputs"].items()}
    s0 = {k: resolve(in_root, v) for k, v in cfg["stage0_inputs"].items()}
    check_resources(rp.out)
    fit_start, fit_end, rain_start = day(cfg["fit_start"]), day(cfg["fit_end"]), day(cfg["rain_start"])
    half = cfg["pool_half_days"]
    first_valid = {v: day(cfg["first_valid"][v]) for v in SPI_VARS}
    var_of = {3: "spi3", 6: "spi6"}

    # 1) 시군구 대표 좌표(단계 0-4)
    coord = pq.read_table(s0["sgg_coord"])
    sgg = sorted(coord["sgg_code"].to_pylist())
    xy = kw.coords_in_order(coord, sgg)
    print(f"시군구 대표 좌표 {len(sgg)}곳")

    # 2) 기상청 SPI → 시군구 3지점 평균
    kt = read_kma_spi(paths["kma_spi"], cfg["kma_columns"])
    kmin, kmax = pc.min_max(kt["meas_date"]).values()
    g_kma = grid_of(kmin.as_py(), kmax.as_py())
    net_kma = kw.load_asos(table_buffer(kt), paths["asos_station"], list(SPI_VARS), g_kma)
    kma_res, kma_info = {}, {"행": kt.num_rows, "지점": len(net_kma.sids),
                             "빈 칸": {v: int(kt[v].null_count) for v in SPI_VARS},
                             "지점 위치 바뀜 수": net_kma.info.get("position_changes")}
    for v in SPI_VARS:
        res = kw.compute_var({"ASOS": net_kma}, dict(var=v, prim=("ASOS", v), fb=None), xy, g_kma.n)
        kma_res[v] = res
        kma_info[v] = knn_summary(res, g_kma)
    print(f"기상청 SPI: {kt.num_rows}행·{len(net_kma.sids)}지점 {g_kma.date_min} ~ {g_kma.date_max} ({time.time() - t0:.0f}초)")
    check_resources(rp.out)

    # 3) 국내 우리 식: ASOS sum_rn 3곳 평균 → 시군구마다 감마
    amax = pc.max(pq.read_table(paths["asos_daily"], columns=["meas_date"])["meas_date"]).as_py()
    g_a = grid_of(np.datetime64(rain_start, "D"), amax)
    rain_spec = dict(kw.FEB29_RAIN_SPEC, var="rain_spi", ko="강수량(SPI 예보 칸용, ASOS sum_rn 빈 값 0mm)")
    nets = kw.load_networks({k: paths[k] for k in ("asos_daily", "asos_station")}, [rain_spec], g_a)
    rr = kw.compute_var(nets, rain_spec, xy, g_a.n)
    rain = rr["value"].astype(np.float64)
    dom = {n: sp.spi_from_rain(rain, g_a.d0, n, fit_start, fit_end, half) for n in cfg["months"]}
    dom_info = {"강수 3곳 평균": knn_summary(rr, g_a), "ASOS": {"관측소 수": len(nets["ASOS"].sids), **nets["ASOS"].info},
                **{var_of[n]: spi_summary(r, g_a.d0, fit_end, first_valid[var_of[n]]) for n, r in dom.items()}}
    print(f"국내 우리 식: {len(sgg)} 시군구 × {g_a.n}일 ({time.time() - t0:.0f}초)")
    check_resources(rp.out)

    # 4) 대조 — 기상청 지점마다 그 지점 sum_rn 으로 우리 식
    net = nets["ASOS"]
    st_ids = [s for s in net_kma.sids if s in net.sids]
    not_in_asos = [s for s in net_kma.sids if s not in net.sids]
    Pst = np.stack([net.values["sum_rn"][:, net.sids.index(s)] for s in st_ids]).astype(np.float64)
    stn = {n: sp.spi_from_rain(Pst, g_a.d0, n, fit_start, fit_end, half) for n in cfg["months"]}
    cmp_rows, cmp_all = [], {}
    sa, sk, cdays = overlap(g_a.d0, g_a.n, g_kma.d0, g_kma.n)
    pm = period_masks(cdays, fit_start, fit_end)
    for n in cfg["months"]:
        v = var_of[n]
        ours = stn[n].spi[:, sa]
        kma = np.stack([net_kma.values[v][:, net_kma.sids.index(s)] for s in st_ids]).astype(np.float64)[:, sk]
        cmp_all[f"지점 {v}"] = {p: sp.agreement(ours[:, m], kma[:, m]) for p, m in pm.items()}
        for i, s in enumerate(st_ids):
            for p, m in pm.items():
                cmp_rows.append({"level": "지점", "unit": s, "months": n, "기간": p, **sp.agreement(ours[i, m], kma[i, m])})
        cmp_all[f"지점 {v} 지점별 r 최소·중앙·최대(전체 기간)"] = _spread([r["r"] for r in cmp_rows
                                                              if r["level"] == "지점" and r["months"] == n and r["기간"] == "전체"])
    # 시군구 단위: 1) 기상청 3지점 평균 vs 2) 우리 식
    for n in cfg["months"]:
        v = var_of[n]
        a = dom[n].spi[:, sa]
        b = kma_res[v]["value"].astype(np.float64)[:, sk]
        cmp_all[f"시군구 {v}"] = {p: sp.agreement(a[:, m], b[:, m]) for p, m in pm.items()}
        for i, s in enumerate(sgg):
            cmp_rows.append({"level": "시군구", "unit": s, "months": n, "기간": "전체", **sp.agreement(a[i], b[i])})
    print(f"대조: {json.dumps(cmp_all, ensure_ascii=False)[:600]}")

    # 5) 해외 — 나라 강수로 나라마다
    ow = pq.read_table(s0["overseas_weather"], columns=["alpha_3", "sgg", "date", "rain"]).to_pandas()
    if (ow["sgg"] != "OV" + ow["alpha_3"]).any():
        raise ValueError("해외 표 sgg 가 'OV'+alpha_3 가 아닌 행")
    cube = DailyCube.from_long(ow, "sgg", "date", ["rain"])
    if cube.start != rain_start:
        raise ValueError(f"해외 강수 첫 날 {np.datetime64(cube.start, 'D')} ≠ rain_start {cfg['rain_start']}")
    a3_of = dict(zip(ow["sgg"], ow["alpha_3"]))
    P_ov = cube.values[:, :, 0]
    ov = {n: sp.spi_from_rain(P_ov, cube.start, n, fit_start, fit_end, half) for n in cfg["months"]}
    ov_info = {"나라": len(cube.units), "날짜": f"{np.datetime64(cube.start, 'D')} ~ {np.datetime64(cube.end, 'D')}",
               "강수 결측 칸(표에 행 없음·null)": int(np.isnan(P_ov).sum()),
               **{var_of[n]: spi_summary(r, cube.start, fit_end, first_valid[var_of[n]]) for n, r in ov.items()}}
    print(f"해외: {len(cube.units)}나라 ({time.time() - t0:.0f}초)")
    check_resources(rp.out)

    # 6) 저장
    fit_txt = f"감마 적합 기간 {cfg['fit_start']} ~ {cfg['fit_end']}"
    ours_txt = (f"우리 식(§4-5): 직전 n 개월(달력, 그날 포함) 누적 강수 → 달력일 ±{half}일 묶음 감마 적합(Thom 근사, "
                f"0 강수 확률 따로) → 표준정규, {fit_txt}")
    w = {}
    w["kma"] = write_parquet(out["kma"], long_frame("sgg_code", sgg, g_kma.d0, {v: kma_res[v]["value"] for v in SPI_VARS}), {
        "description": "국내 관측 칸용 SPI(§4-5) — 기상청 SPI 일자료(SPI3·SPI6 원값) 시군구 대표 좌표 가까운 3지점 평균"
                       "(날마다 그날 값 있는 지점 중, §4-3 규칙), 빠진 날 채우지 않음",
        "columns": {"spi3": "기상청 SPI3 3지점 평균", "spi6": "기상청 SPI6 3지점 평균"}})
    w["dom"] = write_parquet(out["dom"], long_frame("sgg_code", sgg, g_a.d0, {var_of[n]: dom[n].spi for n in cfg["months"]}), {
        "description": "국내 예보 칸용 SPI(§4-5) — 가까운 ASOS 3곳 sum_rn(빈 값 0mm, 부록 A-2) 평균 → 시군구마다 " + ours_txt
                       + f"; SPI3 {cfg['first_valid']['spi3']}·SPI6 {cfg['first_valid']['spi6']} 부터(그 전 NaN)",
        "columns": {v: f"{v.upper()} 우리 식, {fit_txt}" for v in SPI_VARS}})
    w["ov"] = write_parquet(out["ov"], long_frame("sgg", list(cube.units), cube.start, {var_of[n]: ov[n].spi for n in cfg["months"]},
                                                  extra={"alpha_3": a3_of}), {
        "description": "해외 SPI(§4-5, 관측·예보 같은 값) — 나라 강수(DB global_weather_info, 해외 나라 일별 기상표 rain)로 나라마다 "
                       + ours_txt + f"; SPI3 {cfg['first_valid']['spi3']}·SPI6 {cfg['first_valid']['spi6']} 부터(그 전 NaN)",
        "columns": {v: f"{v.upper()} 우리 식, {fit_txt}" for v in SPI_VARS}})
    gf = pd.concat([gamma_frame("국내_시군구", sgg, n, dom[n].fit) for n in cfg["months"]]
                   + [gamma_frame("해외_나라", list(cube.units), n, ov[n].fit) for n in cfg["months"]]
                   + [gamma_frame("대조_지점", st_ids, n, stn[n].fit) for n in cfg["months"]], ignore_index=True)
    w["gamma"] = write_parquet(out["gamma"], gf, {
        "description": f"SPI 감마 모수(Thom 근사) — 단위·누적 개월·달력일(1..365, 2/29 = 59)마다, {fit_txt}, 묶음 ±{half}일; "
                       "status: ok·no_data(묶음 값 없음)·no_positive(양수 없음)·degenerate(양수 2개 미만 또는 A ≤ 0) — ok 아니면 그 달력일 SPI NaN"})
    with open(out["cmp_csv"], "x", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["level", "unit", "months", "기간", "n", "r", "mae", "mean_diff"])
        wr.writeheader()
        wr.writerows(cmp_rows)
    check = {"기상청 SPI": kma_info, "국내 우리 식": dom_info, "해외 우리 식": ov_info,
             "대조": {"설명": "지점 = 기상청 지점마다 그 지점 ASOS sum_rn 으로 우리 식(같은 적합 기간) vs 같은 날 기상청 값; "
                              "시군구 = 국내 예보 칸 표 vs 국내 관측 칸 표(같은 시군구·날)", "기상청 지점 중 ASOS 원자료에 없는 지점": not_in_asos,
                      **cmp_all},
             "저장 행 수": w, "걸린 초": round(time.time() - t0, 1)}
    cf.dump_json(out["check"], cf.nan_to_null(check))
    md5_s0 = {v: cf.file_md5(s0[k]) for k, v in cfg["stage0_inputs"].items()}
    cf.save_run_meta(out["meta"], cfg, cf.inputs_md5(rp.root, list(cfg["inputs"].values())), CODE_ROOT, n_jobs=None,
                     extra={"stage0_inputs_md5": md5_s0, "outputs": {k: v.name for k, v in out.items()},
                            "config_path": Path(args.config).name})
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


def _spread(rs):
    r = np.array([x for x in rs if x is not None], dtype=np.float64)
    return [float(r.min()), float(np.median(r)), float(r.max())] if r.size else None


if __name__ == "__main__":
    main()
