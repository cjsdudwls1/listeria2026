# -*- coding: utf-8 -*-
"""s0_4_weather_overseas.py — 단계 0-4 해외 나라 일자료표(1차 = DB 값만, 2차 = Open-Meteo 로 채운 표).

계획서: §4-4, §4-5(해외 SPI 용 강수 2013-05-01 ~ — SPI 계산은 다음 작업), §6 단계 0-4(해외 나라 표·DB 대조·보정·
해외 칸 원천 표시), 부록 A-2(칼럼 대응·단위·보정·'근사' 표기).

입력: global_weather_info 추출본 + s0_4_openmeteo_fetch.py 가 받은 조각(--fetch-dir, fetch 단계까지 끝난 폴더).
받을 조각이 하나라도 없거나 요청 값(좌표·모델·기간)이 지금 계획과 다르면 멈춤.

--db-only(담당자 결정 11-07 — 해외 Open-Meteo 수집 안 함): 받은 조각 없이 DB 값만으로 표 하나를 만들어 1차·2차
파일에 같은 내용(메타 same_as_1st), 날짜 2015-01-01 ~ 2026-06-30, DB 에 없는 변수 NaN, 보정 없음. --nations 없으면 DB 나라 전부.
  실행: <python> scripts/s0_4_weather_overseas.py --root <자료 뿌리> --out <결과> --db-only [--nations <파일>]

만드는 것(--out, 기존 파일 안 덮음):
- 해외_나라_일별_기상_2차.parquet: 대상 나라 × 날짜(2013-05-01 ~ 2026-06-30)
    alpha_3·sgg('OV' + alpha-3, §2-3)·date·21변수(국내 19변수 + avg_rhm 습도·avg_td 이슬점)·src_<변수>
    값: DB 나라 = DB 값(2015-01-01 ~) / DB 에 없는 기간 = Open-Meteo + 2015 겹치는 기간 나라·변수별 보정(om_cal) /
        DB 에 없는 변수(지중온도·풍향·이슬점) = Open-Meteo 그대로(om) / DB 에 없는 나라 = Open-Meteo 그대로(om) /
        soil_t_nas·no2·co·so2 = NaN / 2014-04-07 전은 강수만(SPI 누적용), 나머지 변수 NaN.
    src_<변수> = 'db'·'om_cal'·'om'·'om_calc'(이슬점을 기온·습도로 계산한 날) — 값이 NaN 인 칸은 null.
- 해외_나라_일별_기상_1차.parquet: 같은 모양, src = 'db' 칸만 값(그 밖 = NaN·null) — §4-4 '1차 = DB 값만'.
- 해외_기상_보정표.csv: 나라 × 변수 보정값(평균 차이 / 강수 합 비율)·쓴 날 수.
- 해외_기상_보고.json: 단위 확인 비·DB 대조(2015, 보정 전·후)·자른 칸 수·원천별 칸 수·결측·응답 격자·시간대.
- 해외_기상_칼럼설명.json: 칼럼마다 단위·원천·'근사' 표기(부록 A-2). 같은 내용을 parquet 메타에도.
- run_meta_*.json: code_md5·입력 md5(받은 조각 md5 묶음 포함)·스레드·설정(B-2).

구현 세부(계획서에 없음 — 담당자 확인 필요):
- 단위 확인: 2015 겹치는 기간 모든 DB 나라 칸에서 Σ|DB| / Σ|Open-Meteo| 가 [0.8, 1.25] 안이면 같은 단위 → DB 값에도
  Open-Meteo 와 같은 변환(풍속 ÷ 3.6·일조 ÷ 3600·토양수분 × 100). 밖이면 멈춤.
- 보정 평균·합 = DB·Open-Meteo 둘 다 값 있는 날만. 자르기(0 미만·0~100)는 보정한 칸(om_cal)에만 —
  DB 에 없는 나라(보정 없음)·DB 값은 안 자름. 바람 등 다른 변수는 보정 뒤 음수여도 안 자름(칸 수만 보고).
- DB 값이 NaN 인 칸(기록: wind_speed_max 1칸)은 Open-Meteo 로 안 채움(DB 에 없는 '기간·변수' 가 아님) — 수만 보고.
- 이슬점 = era5 dew_point_2m_mean, 그 값이 NaN 이고 기온·습도가 있는 날만 Magnus 식(avg_temp·avg_rhm, 2차 값).
- 표 날짜 끝 = 2026-06-30(d 최대) — DB 의 그 뒤 날(~ 2026-09-30)은 표에 안 넣음.
- 값 자료형 float64(국내 표는 float32 — 바꾸지 않고 둠, window 생성기가 float64 로 읽음).

실행(서버, fetch 끝난 뒤 검수 거쳐 지휘자):
  ~/cyj_listeria_final/venv/bin/python scripts/s0_4_weather_overseas.py --root ~/cyj_listeria_final/data \
      --out ~/cyj_listeria_final/results/s0/0-4_weather_overseas \
      --fetch-dir ~/cyj_listeria_final/results/s0/0-4_openmeteo --nations <fetch 때와 같은 나라 목록 파일>
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import resources as rs  # noqa: E402

rs.set_thread_env(rs.THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import ovweather as ow  # noqa: E402
from lst import paths as ph  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = "configs/s0_4_weather_overseas.json"
OUT_2 = "해외_나라_일별_기상_2차.parquet"
OUT_1 = "해외_나라_일별_기상_1차.parquet"
OUT_CAL = "해외_기상_보정표.csv"
OUT_REP = "해외_기상_보고.json"
OUT_DESC = "해외_기상_칼럼설명.json"


class BuildStop(RuntimeError):
    pass


def build_parser():
    ap = ph.build_parser(__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--nations", default=None, help="나라 목록(자료 뿌리 아래). --db-only 에선 없으면 DB 나라 전부")
    ap.add_argument("--nations-col", default="alpha_3")
    ap.add_argument("--fetch-dir", default=None, help="s0_4_openmeteo_fetch.py 의 --out 폴더(--db-only 아니면 필수)")
    ap.add_argument("--db-only", action="store_true",
                    help="Open-Meteo 없이 DB global_weather_info 값만(담당자 결정 11-07) — 1차 = 2차 같은 표")
    ap.add_argument("--coord", choices=(ow.COORD_DB, ow.COORD_COUNTRY), default=None,
                    help="fetch 때 --coord 를 썼으면 같은 값")
    return ap


# ---------------------------------------------------------------- 조각 읽기
def read_chunks(fetch_dir: Path, items, models, timezone: str):
    """(조각, 위도, 경도) 목록 → 묶음별 DataFrame 이어 붙임 + 응답 머리 + md5 목록. 빠진 조각·요청 다름이면 멈춤."""
    frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    heads, md5s, missing = {}, [], []
    for c, la, lo in items:
        p, q = ow.chunk_files(fetch_dir, c)
        if not (p.exists() and q.exists()):
            missing.append(c.relpath)
            continue
        req = json.loads(q.read_text(encoding="utf-8"))
        want = ow.request_params(c, la, lo, models, timezone)
        if req["params"] != want:
            raise BuildStop(f"조각 요청 값이 지금 계획과 다름: {c.relpath}\n  조각 {req['params']}\n  계획 {want}")
        body = p.read_bytes()
        m = hashlib.md5(body).hexdigest()
        if m != req["body_md5"]:
            raise BuildStop(f"조각 md5 다름: {c.relpath}")
        df, head = ow.parse_response(body, c, models, la, lo)
        frames.setdefault((c.a3, c.group), []).append(df)
        heads.setdefault(c.a3, {})[c.group] = head
        md5s.append((c.relpath, m))
    if missing:
        raise BuildStop(f"받지 않은 조각 {len(missing)}개(예: {missing[:5]}) — fetch 를 먼저 끝까지")
    out = {}
    for k, parts in frames.items():
        df = pd.concat(parts).sort_index()
        if df.index.duplicated().any():
            raise BuildStop(f"조각 날짜 겹침: {k}")
        out[k] = df
    return out, heads, md5s


def om_dbvars(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return df.rename(columns={f"{k}@{model}": v for k, v in ow.OM_DB_VARS.items()})[list(ow.CAL_VARS)]


# ---------------------------------------------------------------- 나라 하나
def extra_values(df: pd.DataFrame, models) -> pd.DataFrame:
    """extra 조각 → soil_t5~t30·wind_dir_sin·cos·avg_td(Open-Meteo 그대로, 단위 변환 없음)."""
    st = [df[f"{v}@{models['soil_temp']}"].to_numpy() for v in ow.OM_SOIL_T]
    soil = ow.interp_soil(*st)
    s, c = ow.wind_dir_sincos(df[f"{ow.OM_WIND_DIR}@{models['wind_dir']}"].to_numpy())
    out = pd.DataFrame(soil, index=df.index)
    out["wind_dir_sin"], out["wind_dir_cos"] = s, c
    out[ow.DEW] = df[f"{ow.OM_DEW}@{models['dew_point']}"].to_numpy()
    return out


def nation_table(a3: str, dates: pd.DatetimeIndex, p: ow.Periods, chunks: dict, models,
                 db: pd.DataFrame | None, cal: dict | None, clip_counts: dict):
    """나라 하나의 (값, 원천) 표. db = 변환된 DB 값(날짜 색인, CAL_VARS) 또는 None(DB 에 없는 나라)."""
    vals = pd.DataFrame(np.nan, index=dates, columns=list(ow.TABLE_VARS))
    src = pd.DataFrame(None, index=dates, columns=list(ow.TABLE_VARS), dtype=object)

    def put(var, series: pd.Series, tag: str):
        s = series.reindex(dates)
        m = s.notna() & s.index.isin(series.index)
        vals.loc[m, var] = s[m]
        src.loc[m, var] = tag

    tag_om = ow.SRC_OM_CAL if db is not None else ow.SRC_OM
    # 강수(2013-05-01 ~ 표 시작 전날)
    rain_pre = chunks.get((a3, "rain"))
    if rain_pre is not None:
        r = rain_pre.rename(columns={f"{ow.OM_RAIN}@{models['rain']}": "rain"})[["rain"]]
        r = ow.convert_units(r)
        if cal is not None:
            r, cl = ow.apply_calibration(r, cal)
            clip_counts[(a3, "rain")] = clip_counts.get((a3, "rain"), 0) + cl.get("rain", 0)
        put("rain", r["rain"], tag_om)
    # DB 에도 있는 10변수
    om = ow.convert_units(om_dbvars(chunks[(a3, "dbvars")], models["dbvars"]))
    if db is not None:
        gap = om.loc[~om.index.isin(db.index)]           # DB 에 없는 날만(2015 겹치는 날은 보정에만 씀)
        gap, cl = ow.apply_calibration(gap, cal)
        for v, n in cl.items():
            clip_counts[(a3, v)] = clip_counts.get((a3, v), 0) + n
        for v in ow.CAL_VARS:
            put(v, gap[v], ow.SRC_OM_CAL)
            dv = db[v]
            dv = dv[(dv.index >= dates[0]) & (dv.index <= dates[-1])]
            put(v, dv, ow.SRC_DB)
    else:
        for v in ow.CAL_VARS:
            put(v, om[v], ow.SRC_OM)
    # DB 에 없는 변수
    ex = extra_values(chunks[(a3, "extra")], models)
    for v in ex.columns:
        put(v, ex[v], ow.SRC_OM)
    # 이슬점을 못 받은 날만 계산(부록 A-2)
    need = vals[ow.DEW].isna() & vals["avg_temp"].notna() & vals[ow.HUM].notna() & (dates >= pd.Timestamp(p.table_start))
    n_calc = int(need.sum())
    if n_calc:
        vals.loc[need, ow.DEW] = ow.dewpoint_magnus(vals.loc[need, "avg_temp"], vals.loc[need, ow.HUM])
        src.loc[need & vals[ow.DEW].notna(), ow.DEW] = ow.SRC_OM_CALC
    for v in ow.ALWAYS_NAN:
        assert vals[v].isna().all()
    return vals, src, n_calc


def to_long(a3_tables: dict, only_db: bool) -> pd.DataFrame:
    parts = []
    for a3, (vals, src) in sorted(a3_tables.items()):
        v, s = vals.copy(), src.copy()
        if only_db:
            keep = s.eq(ow.SRC_DB)
            v = v.where(keep)
            s = s.where(keep)
        df = pd.DataFrame({"alpha_3": a3, "sgg": "OV" + a3, "date": vals.index.date})
        for c in ow.TABLE_VARS:
            df[c] = v[c].to_numpy(dtype=np.float64)
        for c in ow.TABLE_VARS:
            df["src_" + c] = s[c].to_numpy(dtype=object)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def write_table(df: pd.DataFrame, path: Path, meta: dict):
    if path.exists():
        raise BuildStop(f"덮어쓰기 금지: {path}")
    fields = [pa.field("alpha_3", pa.string()), pa.field("sgg", pa.string()), pa.field("date", pa.date32())]
    fields += [pa.field(c, pa.float64()) for c in ow.TABLE_VARS]
    fields += [pa.field("src_" + c, pa.string()) for c in ow.TABLE_VARS]
    schema = pa.schema(fields, metadata={k: json.dumps(v, ensure_ascii=False) for k, v in meta.items()})
    t = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(t, tmp)
    tmp.replace(path)


def db_compare(db: pd.DataFrame, om: pd.DataFrame) -> dict:
    """변수마다 2015 겹치는 기간 DB − Open-Meteo: 평균 차이·평균 절대 차이·상관(둘 다 값 있는 날)."""
    out = {}
    for v in ow.CAL_VARS:
        a, b = db[v], om[v].reindex(db.index)
        m = a.notna() & b.notna()
        if not m.any():
            out[v] = {"n": 0}
            continue
        x, y = a[m].to_numpy(), b[m].to_numpy()
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        out[v] = {"n": int(m.sum()), "mean_diff": float((x - y).mean()), "mae": float(np.abs(x - y).mean()),
                  "corr": corr}
    return out


# ---------------------------------------------------------------- main
def main_db_only(rp, args) -> int:
    """--db-only: DB 값만으로 나라 일자료표 하나 → 1차·2차 파일로 같은 내용(메타 same_as_1st = true).

    담당자 결정(11-07): 해외 Open-Meteo 수집 안 함, 해외 기상 = DB global_weather_info 값만, 1차·2차 같음.
    부록 A-2 DB 칼럼 대응·단위 변환(풍속 ÷ 3.6·일조 ÷ 3600·토양수분 × 100 — DB 단위 = Open-Meteo 단위는 11-07
    시험 호출로 확인), 보정 없음, 날짜 = DB 첫날(2015-01-01) ~ use_end(2026-06-30), 원천 표시 = db,
    DB 에 없는 변수(지중온도 4·soil_t_nas·풍향 sin·cos·no2·co·so2·이슬점) = NaN.
    """
    for name in (OUT_1, OUT_2, OUT_REP, OUT_DESC):
        if rp.result(name).exists():
            raise BuildStop(f"이미 있음 — 덮지 않음: {rp.result(name)}")
    cfg = ow.load_cfg(ph.resolve(CODE_ROOT, args.config))
    p = ow.periods_of(cfg)
    rs.check_resources(rp.out)
    gwi_rel = cfg["inputs"]["global_weather_info"]
    inputs = [gwi_rel]
    if args.nations is not None:
        targets = ow.read_targets(rp.data(args.nations), args.nations_col)
        inputs.append(args.nations)
        gwi = ow.read_gwi(rp.data(gwi_rel), targets)
    else:
        gwi = ow.read_gwi(rp.data(gwi_rel))
        targets = sorted(gwi["alpha_3"].unique().tolist())
    info = ow.db_info(gwi)
    missing = [a for a in targets if a not in info.index]
    if info.empty:
        raise BuildStop("대상 나라가 DB 에 하나도 없음")
    start = info["db_start"].min()
    print(f"DB 값만: 나라 {len(targets)}(DB 에 없음 {missing}), 날짜 {start.date()} ~ {p.use_end}")
    t = ow.db_only_table(gwi, targets, start, p.use_end)
    desc = ow.column_descriptions_db_only()
    src_counts = {c: {str(k): int(n) for k, n in t["src_" + c].value_counts(dropna=False).items()}
                  for c in ow.TABLE_VARS}
    db_nan = {}
    for a, g in t[t.alpha_3.isin(info.index)].groupby("alpha_3"):
        nn = {c: int(g[c].isna().sum()) for c in ow.CAL_VARS if g[c].isna().any()}
        if nn:
            db_nan[a] = nn
    for v in ow.DB_ONLY_NAN:
        assert t[v].isna().all()
    report = {"mode": "db_only", "decision": "담당자 결정 11-07: Open-Meteo 수집 안 함, 1차 = 2차 = DB 값만",
              "date_range": [str(start.date()), p.use_end.isoformat()], "targets": targets, "db_missing": missing,
              "conversion": dict(ow.CONVERT), "nan_vars": list(ow.DB_ONLY_NAN), "db_nan_cells_not_filled": db_nan,
              "src_counts": src_counts, "rows": len(t),
              "db_range_by_nation": {a: [str(r.db_start.date()), str(r.db_end.date())] for a, r in info.iterrows()}}
    meta = {"description": "해외 나라 일자료표(단계 0-4, DB 값만 — 계획서 §4-4·부록 A-2, 담당자 결정 11-07)",
            "columns": desc, "mode": "db_only", "same_as_1st": True, "date_range": report["date_range"]}
    write_table(t, rp.result(OUT_1), {**meta, "table": "1차 — DB 값만(2차와 같은 내용)"})
    write_table(t, rp.result(OUT_2), {**meta, "table": "2차 — DB 값만(1차와 같은 내용, Open-Meteo 수집 안 함)"})
    cf.dump_json(rp.result(OUT_DESC), desc)
    cf.dump_json(rp.result(OUT_REP), report)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cf.save_run_meta(rp.result(f"run_meta_build_dbonly_{stamp}.json"), cfg, cf.inputs_md5(rp.root, inputs), CODE_ROOT,
                     extra={"run": {"mode": "db_only", "outputs": {OUT_1: cf.file_md5(rp.result(OUT_1)),
                                                                   OUT_2: cf.file_md5(rp.result(OUT_2))}}})
    print(f"{len(t):,}행(1차 = 2차), 원천(avg_temp) {src_counts['avg_temp']}")
    return 0


def main(argv=None):
    ap = build_parser()
    rp, args = ph.parse_paths(argv, ap)
    if args.db_only:
        if args.fetch_dir is not None or args.coord is not None:
            raise BuildStop("--db-only 에는 --fetch-dir·--coord 를 쓰지 않음")
        return main_db_only(rp, args)
    if args.fetch_dir is None or args.nations is None:
        raise BuildStop("--fetch-dir·--nations 필수(--db-only 가 아니면)")
    fetch_dir = Path(args.fetch_dir).expanduser().resolve()
    if not fetch_dir.is_dir():
        raise BuildStop(f"--fetch-dir 없음: {fetch_dir}")
    for name in (OUT_1, OUT_2, OUT_CAL, OUT_REP, OUT_DESC):
        if rp.result(name).exists():
            raise BuildStop(f"이미 있음 — 덮지 않음: {rp.result(name)}")
    cfg = ow.load_cfg(ph.resolve(CODE_ROOT, args.config))
    p = ow.periods_of(cfg)
    models, tz = cfg["openmeteo"]["models"], cfg["openmeteo"]["timezone"]
    rs.check_resources(rp.out)

    li = ow.load_inputs(rp, cfg, args.nations, args.nations_col)
    targets, gwi, info, in_db, missing, poly = (li[k] for k in ("targets", "gwi", "info", "in_db", "missing", "poly"))
    db_coord, how = ow.db_coord_choice(fetch_dir, args.coord) if in_db else (ow.COORD_DB, "DB 나라 없음")
    print(f"대상 {len(targets)}(DB {len(in_db)}, DB 에 없음 {missing}), DB 나라 좌표 = {db_coord}({how})")
    items = ow.all_chunks(p, info, in_db, missing, poly, db_coord)
    chunks, heads, md5s = read_chunks(fetch_dir, items, models, tz)
    raw_md5 = hashlib.md5("\n".join(f"{a}\t{b}" for a, b in sorted(md5s)).encode("utf-8")).hexdigest()

    # DB 값(원 단위) → 우리 변수 이름
    dbw = {a: gwi[gwi["alpha_3"] == a].set_index("date").rename(columns=ow.DB_TO_VAR)[list(ow.CAL_VARS)].sort_index()
           for a in in_db}
    ov = (pd.Timestamp(p.overlap_start), pd.Timestamp(p.overlap_end))

    def in_ov(df):
        return df.loc[(df.index >= ov[0]) & (df.index <= ov[1])]

    # 1) 단위 확인(겹치는 기간, 원 단위)
    unit = {}
    if in_db:
        for v in ow.CAL_VARS:
            xa, xb = [], []
            for a in in_db:
                dba = in_ov(dbw[a])[v]
                oma = in_ov(om_dbvars(chunks[(a, "dbvars")], models["dbvars"]))[v].reindex(dba.index)
                xa.append(dba.to_numpy(float))
                xb.append(oma.to_numpy(float))
            unit[v] = ow.unit_ratio(np.concatenate(xa), np.concatenate(xb))
        bad = {v: r for v, r in unit.items() if not (ow.UNIT_RATIO_OK[0] <= r <= ow.UNIT_RATIO_OK[1])}
        if bad:
            raise BuildStop(f"DB·Open-Meteo 단위가 달라 보임(Σ|DB|/Σ|OM| {bad}) — 담당자 확인 전 멈춤")
    print("단위 확인 비:", {k: round(v, 4) for k, v in unit.items()})

    # 2) 변환·보정·나라 표
    dates = ow.date_range(p.rain_start, p.use_end)
    tables, cal_rows, clip_counts, compare, n_calc, db_nan = {}, [], {}, {}, {}, {}
    for a in targets:
        if a in in_db:
            dbc = ow.convert_units(dbw[a])
            omc = ow.convert_units(om_dbvars(chunks[(a, "dbvars")], models["dbvars"]))
            cal = ow.calibration(in_ov(dbc), in_ov(omc))
            for v, c in cal.items():
                cal_rows.append({"alpha_3": a, "var": v, **c})
            before = db_compare(in_ov(dbc), in_ov(omc))
            after_om, _ = ow.apply_calibration(in_ov(omc), cal)
            compare[a] = {"before": before, "after": db_compare(in_ov(dbc), after_om)}
            db_nan[a] = {v: int(n) for v, n in dbc.loc[dbc.index <= dates[-1]].isna().sum().items() if n}
            vals, src, nc = nation_table(a, dates, p, chunks, models, dbc, cal, clip_counts)
        else:
            vals, src, nc = nation_table(a, dates, p, chunks, models, None, None, clip_counts)
        tables[a] = (vals, src)
        n_calc[a] = nc

    t2 = to_long(tables, only_db=False)
    t1 = to_long(tables, only_db=True)

    # 3) 보고
    desc = ow.column_descriptions(models)
    src_counts = {c: {str(k): int(n) for k, n in t2["src_" + c].value_counts(dropna=False).items()}
                  for c in ow.TABLE_VARS}
    tstart = pd.Timestamp(p.table_start).date()
    after_start = t2["date"] >= tstart
    nan_by_nation = {}
    for a, g in t2[after_start].groupby("alpha_3"):
        nn = {c: int(g[c].isna().sum()) for c in ow.TABLE_VARS if c not in ow.ALWAYS_NAN and g[c].isna().any()}
        if nn:
            nan_by_nation[a] = nn
    neg_after_cal = {}
    for c in ("avg_wind", "max_wind", "solar_rad"):
        m = (t2["src_" + c] == ow.SRC_OM_CAL) & (t2[c] < 0)
        if m.any():
            neg_after_cal[c] = int(m.sum())
    report = {
        "periods": cfg["periods"], "models": models, "timezone": tz, "db_coord": db_coord, "db_coord_how": how,
        "targets": targets, "db_missing": missing, "polygon_coords": poly,
        "unit_ratio_db_over_om": unit, "unit_ratio_ok": list(ow.UNIT_RATIO_OK),
        "conversion": {k: v for k, v in ow.CONVERT.items()},
        "db_compare_2015": compare,
        "clipped_cells": {f"{a}|{v}": n for (a, v), n in sorted(clip_counts.items()) if n},
        "negative_after_calibration_not_clipped": neg_after_cal,
        "dewpoint_computed_days": {a: n for a, n in n_calc.items() if n},
        "db_nan_cells_not_filled": {a: x for a, x in db_nan.items() if x},
        "src_counts_2nd": src_counts,
        "nan_cells_from_table_start_2nd": nan_by_nation,
        "rows": {"2nd": len(t2), "1st": len(t1), "nations": len(targets), "dates": len(dates)},
        "response_heads": heads,
        "openmeteo_raw_md5": raw_md5, "openmeteo_chunks": len(md5s),
    }
    meta = {"description": "해외 나라 일자료표(단계 0-4, 계획서 §4-4·부록 A-2)", "columns": desc,
            "periods": cfg["periods"], "models": models, "db_coord": db_coord}
    write_table(t2, rp.result(OUT_2), {**meta, "table": "2차 — DB 값 + Open-Meteo 로 채움(칸마다 src_*)"})
    write_table(t1, rp.result(OUT_1), {**meta, "table": "1차 — DB 값만(그 밖 NaN)"})
    pd.DataFrame(cal_rows).to_csv(rp.result(OUT_CAL), index=False, encoding="utf-8")
    cf.dump_json(rp.result(OUT_DESC), desc)
    cf.dump_json(rp.result(OUT_REP), report)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md5_in = cf.inputs_md5(rp.root, li["inputs"])
    md5_in["openmeteo_raw(조각 md5 묶음)"] = raw_md5
    cf.save_run_meta(rp.result(f"run_meta_build_{stamp}.json"), cfg, md5_in, CODE_ROOT,
                     extra={"run": {"fetch_dir_name": fetch_dir.name, "db_coord": db_coord, "db_coord_how": how,
                                    "outputs": {OUT_1: cf.file_md5(rp.result(OUT_1)),
                                                OUT_2: cf.file_md5(rp.result(OUT_2))}}})
    print(f"2차 {len(t2):,}행, 1차 {len(t1):,}행, 나라 {len(targets)}, 원천 칸 수 예(rain) {src_counts['rain']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildStop as e:
        print(f"멈춤: {e}")
        sys.exit(3)
