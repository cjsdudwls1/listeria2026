# -*- coding: utf-8 -*-
"""s0_3_forecast_prep.py — 단계 0-3 예보 준비(계획서 §6 단계 0-3, §4-1, 부록 A-2, 부록 A-13).

하는 일:
  1) 국내 11일 예보 변수 목록: 예보 표 2개(`predict_data.p0830_fcst_long`·`api_data_info.kma_weather_predict_info`,
     추출본 리스테리아ML2/data/추출_20261107/) 칼럼 목록·값 범위, 11일(발표 당일 포함) 확인, 우리 기상 19변수·습도·이슬점
     21개마다 대응 예보 칼럼(설정 forecast_map — 근거 칸, 담당자 확인 대상)과 '예보에 없음' 표시.
  2) 예보 좌표(`api_data_info.kma_sigungu_point`)에 시군구 코드로 연결 안 되는 시군구 목록 + 가장 가까운 예보 좌표
     (부록 A-2 — 우리 시군구 대표 좌표에서 구면 거리). 256 시군구 전체의 예보 좌표 표도 남김.
  3) 예보 좌표 차이 측정(부록 A-13 정의 그대로): 차이 = 예보 좌표에서 가장 가까운 관측소 1곳 값 − 3곳 평균,
     기간 = d ≤ 2025-06-30, 변수 = 11일 예보 변수 목록에 있는 기상 변수, 시군구·변수별 평균. 측정·보고만.
  4) 원천별 공개 지연(일) 확인 표 — 담당자 보고용, 반영 안 함(기본값 = 반영 안 함, 부록 A-13).

구현 세부(계획서에 없음 — 담당자 확인):
  - 예보 칼럼 ↔ 우리 변수 대응은 DB 칼럼 설명이 추출본에 없어 설정 forecast_map 의 근거(같은 이름의 NAS 칼럼,
    아플라톡신 팀 코드 주석)로만 적음. '대응 확인용 비교' 표(예보 값 vs 같은 시군구·같은 날 우리 3곳 평균 관측값)를 같이 냄.
    대응이 바뀌면 설정만 고쳐 다시 돌림.
  - '예보에 있음' = 두 예보 표 중 하나라도 대응 칼럼이 있음.
  - 1곳·3곳 규칙 = 03_시군구_기상.py 규칙(lst/kweather.py)을 k=1·k=3 으로(그날 값 있는 관측소 중, 기본 칼럼이 빈 칸만 대체).
    '3곳 평균' = 우리 시군구 대표 좌표(옹진군 = 단계 0-4 새 좌표, 같은 함수로 계산)의 3곳 평균. 19변수는 다시 계산한 값이
    기존 표와 같은지(28720 밖) 확인하고 다르면 멈춤. 습도(rh)는 같은 규칙으로 새로 계산.
  - 차이 측정 기간 시작 = 2015-01-01(§4-3 국내 값 시작), 두 값이 다 있는 날만 평균.
  - 연결 = sigungu_code 와 sgg_code 문자열이 같음. 연결 안 되면 가장 가까운 예보 좌표(동률이면 표 순서 앞).
  - 공개 지연: 계획서에 재는 방법이 없음. 자료 안에 있는 것만 씀 — 원천 파일의 마지막 날짜, 받은 시각(추출 json 의
    started_at 이 있는 원천만), 지점별 마지막 날짜 분포, 칼럼별 마지막 값 있는 날. 받은 시각 기록이 없는 원천은 '기록 없음'.
    (지연을 정하는 것은 담당자 — 이 표는 판단 자료.)

출력(<out>/, 이미 있으면 멈춤):
  예보_칼럼목록.csv · 예보_11일_확인.json · 예보_변수대응.csv · 예보_대응확인_비교.csv
  예보좌표_시군구별.csv · 예보좌표_연결안됨.csv · 예보좌표_차이.csv · 공개지연_원천별.csv · 공개지연_칼럼별.csv
  점검.json · run_meta.json

실행(서버, 검수 뒤):
  ~/cyj_listeria_final/venv/bin/python -I scripts/s0_3_forecast_prep.py \
      --root ~/cyj_listeria_final/data --out ~/cyj_listeria_final/results/s0/0-3
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from lst.resources import set_thread_env  # noqa: E402

set_thread_env()

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import kweather as kw  # noqa: E402
from lst.paths import build_parser, parse_paths, resolve  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_KEYS = ("fcst_long", "fcst_info", "sigungu_point", "weather_table", "sgg_coord", "sample_sgg", "region_code",
              "asos_daily", "asos_station", "nas_daily", "nas_station", "air_daily", "air_station")
STR_OR_NONE = (str, type(None))
SCHEMA = {
    "plan": cf.Field(str),
    "inputs": {k: cf.Field(str) for k in INPUT_KEYS},
    "ongjin_sgg_code": cf.Field(str),
    "forecast_len_days": cf.Field(int),
    "diff_from": cf.Field(str),
    "diff_to": cf.Field(str),
    "forecast_map": cf.Field(list, item_types=dict),
    "delay_sources": cf.Field(list, item_types=dict),
}
MAP_SCHEMA = {"var": cf.Field(str), "fcst_long": cf.Field(STR_OR_NONE), "fcst_info": cf.Field(STR_OR_NONE),
              "basis": cf.Field(str)}
DELAY_SCHEMA = {"name": cf.Field(str), "file": cf.Field(str), "format": cf.Field(str, choices=("parquet", "csv_cp949")),
                "date_col": cf.Field(str), "station_col": cf.Field(str), "extract_json": cf.Field(STR_OR_NONE),
                "columns": cf.Field(list, item_types=str)}
FCST_TABLES = {"fcst_long": "predict_data.p0830_fcst_long", "fcst_info": "api_data_info.kma_weather_predict_info"}
OUT_NAMES = {
    "cols": "예보_칼럼목록.csv", "days": "예보_11일_확인.json", "map": "예보_변수대응.csv", "mapchk": "예보_대응확인_비교.csv",
    "pt_all": "예보좌표_시군구별.csv", "pt_unl": "예보좌표_연결안됨.csv", "diff": "예보좌표_차이.csv",
    "delay": "공개지연_원천별.csv", "delay_col": "공개지연_칼럼별.csv", "check": "점검.json", "meta": "run_meta.json",
}
POINT_TOL_DEG = 1e-6  # fcst_info 좌표 ↔ kma_sigungu_point 좌표 같음(범위 검색) — 구현 세부


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        rows = [{"(없음)": ""}]
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _py(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 6)
    return v


# ---------------------------------------------------------------- 1) 예보 변수 목록
def column_list(name, t: pa.Table) -> list[dict]:
    rows = []
    for f in t.schema:
        c = t[f.name]
        r = {"표": FCST_TABLES[name], "칼럼": f.name, "자료형": str(f.type), "행 수": t.num_rows, "빈 값": c.null_count,
             "최소": None, "최대": None, "고유 값 수": pc.count_distinct(c).as_py(), "값 예(문자 칼럼, 최대 10개)": None}
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
            vals = sorted(v for v in pc.unique(c).to_pylist() if v is not None)
            r["값 예(문자 칼럼, 최대 10개)"] = " | ".join(vals[:10])
        elif c.length() - c.null_count > 0:
            mm = pc.min_max(c)
            r["최소"], r["최대"] = _py(mm["min"].as_py()), _py(mm["max"].as_py())
        rows.append(r)
    return rows


def eleven_day_check(fl: pa.Table, fi: pa.Table, n_days: int) -> dict:
    out = {}
    sg = np.array(fl["sigungu_code"].to_pylist(), dtype=object)
    off = fl["day_offset"].to_numpy()
    fd = kw.day_num(fl["forecast_date"])
    issue = fd - off
    per = {}
    for s, o in zip(sg, off):
        per.setdefault(s, set()).add(int(o))
    full = set(range(n_days))
    out["fcst_long"] = {
        "시군구 수": len(per), "day_offset 최소·최대": [int(off.min()), int(off.max())],
        "발표일(forecast_date − day_offset)": sorted({str(np.datetime64(int(x), "D")) for x in issue}),
        f"day_offset 0~{n_days - 1} 이 다 있는 시군구 수": sum(v == full for v in per.values()),
        "그 밖 시군구(offset 집합)": {s: sorted(v) for s, v in per.items() if v != full},
        "(시군구, forecast_date) 중복 행": int(len(sg) - len({(a, b) for a, b in zip(sg, fd)})),
    }
    md = kw.day_num(fi["meas_date"])
    fdi = kw.day_num(fi["forecast_date"])
    lead = fdi - md
    key = list(zip(md.tolist(), fi["lat"].to_pylist(), fi["lon"].to_pylist()))
    grp: dict = {}
    for k, l in zip(key, lead.tolist()):
        grp.setdefault(k, []).append(l)
    cnt = {}
    for v in grp.values():
        cnt[len(v)] = cnt.get(len(v), 0) + 1
    out["fcst_info"] = {
        "meas_date 값": sorted({str(np.datetime64(int(x), "D")) for x in md}),
        "forecast_date − meas_date 최소·최대(일)": [int(lead.min()), int(lead.max())],
        "(meas_date, lat, lon) 묶음 수": len(grp), "묶음당 행 수 분포": {str(k): v for k, v in sorted(cnt.items())},
        "좌표(lat, lon) 수": len({(a, b) for _, a, b in key}),
        "meas_date 별 좌표 수": {str(np.datetime64(int(m), "D")): len({(a, b) for mm, a, b in key if mm == m}) for m in sorted(set(md.tolist()))},
    }
    return out


def info_point_codes(fi: pa.Table, sp: pa.Table) -> tuple[list, dict]:
    """fcst_info 행 좌표 → kma_sigungu_point 코드(범위 1e-6 안 한 점일 때만, 아니면 None)."""
    sla, slo = sp["lat"].to_numpy(), sp["lon"].to_numpy()
    scode = sp["sigungu_code"].to_pylist()
    memo, out = {}, []
    for a, b in zip(fi["lat"].to_pylist(), fi["lon"].to_pylist()):
        if (a, b) not in memo:
            hit = np.flatnonzero((np.abs(sla - a) < POINT_TOL_DEG) & (np.abs(slo - b) < POINT_TOL_DEG))
            memo[(a, b)] = scode[hit[0]] if len(hit) == 1 else None
        out.append(memo[(a, b)])
    info = {"fcst_info 좌표 수": len(memo), "kma_sigungu_point 한 점과 같은 좌표 수": sum(v is not None for v in memo.values())}
    return out, info


# ---------------------------------------------------------------- 4) 공개 지연
def read_delay_source(root, s):
    p = resolve(root, s["file"])
    cols = [s["station_col"], s["date_col"]] + s["columns"]
    if s["format"] == "parquet":
        t = pq.read_table(p, columns=cols)
        sid = np.array(pc.cast(t[s["station_col"]], pa.string()).to_pylist(), dtype=object)
        days = kw.day_num(t[s["date_col"]])
        vals = {c: np.asarray(t[c].is_valid()) for c in s["columns"]}
    else:
        with open(p, "r", encoding="cp949", newline="") as f:
            rd = csv.DictReader(f)
            missing = [c for c in cols if c not in rd.fieldnames]
            if missing:
                raise KeyError(f"{s['name']}: 칼럼 없음 {missing}")
            sid_l, day_l, val_l = [], [], {c: [] for c in s["columns"]}
            for r in rd:
                sid_l.append(r[s["station_col"]])
                day_l.append(int(np.datetime64(r[s["date_col"]], "D").astype(np.int64)))
                for c in s["columns"]:
                    val_l[c].append(r[c].strip() != "")
        sid, days = np.array(sid_l, dtype=object), np.array(day_l, dtype=np.int64)
        vals = {c: np.array(v, dtype=bool) for c, v in val_l.items()}
    return sid, days, vals


def delay_tables(root, sources) -> tuple[list, list]:
    rows, crow = [], []
    for s in sources:
        sid, days, vals = read_delay_source(root, s)
        dmax, dmin = int(days.max()), int(days.min())
        got, got_src = None, "기록 없음(원천 파일 안 메타·옆 json 에 받은 시각 없음)"
        if s["extract_json"]:
            j = json.loads(resolve(root, s["extract_json"]).read_text(encoding="utf-8"))
            if j.get("started_at"):
                got = dt.datetime.fromisoformat(j["started_at"])
                got_src = f"{s['extract_json']} started_at"
        _, inv = np.unique(sid.astype(str), return_inverse=True)
        last = np.full(inv.max() + 1, np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(last, inv, days)
        gaps = dmax - last
        recent = gaps[gaps <= 365]
        r = {"원천": s["name"], "파일": s["file"], "첫 날": str(np.datetime64(dmin, "D")), "마지막 날": str(np.datetime64(dmax, "D")),
             "받은 시각": got.isoformat() if got else None, "받은 시각 근거": got_src,
             "받은 날 − 마지막 날(일)": (got.date() - np.datetime64(dmax, "D").astype(object)).days if got else None,
             "지점 수": len(last), "마지막 날에 행 있는 지점 수": int((gaps == 0).sum()),
             "최근 365일 안 지점: 마지막 날보다 1일 앞에서 끝": int((recent == 1).sum()),
             "2일": int((recent == 2).sum()), "3~7일": int(((recent >= 3) & (recent <= 7)).sum()),
             "8~30일": int(((recent >= 8) & (recent <= 30)).sum()), "31~365일": int(((recent >= 31) & (recent <= 365)).sum()),
             "365일 넘게 앞에서 끝난 지점": int((gaps > 365).sum())}
        rows.append(r)
        for c, ok in vals.items():
            lv = int(days[ok].max()) if ok.any() else None
            crow.append({"원천": s["name"], "칼럼": c, "값 있는 마지막 날": str(np.datetime64(lv, "D")) if lv is not None else None,
                         "원천 마지막 날과 차이(일)": (dmax - lv) if lv is not None else None,
                         "원천 마지막 날에 값 있는 지점 수": int(len(set(sid[ok & (days == dmax)])))})
    return rows, crow


# ---------------------------------------------------------------- main
def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_3_forecast_prep.json"))
    rp, args = parse_paths(argv, ap)
    cfg = cf.load_config(args.config, SCHEMA)
    for i, m in enumerate(cfg["forecast_map"]):
        cf.validate(m, MAP_SCHEMA, where=f"forecast_map[{i}].")
    for i, s in enumerate(cfg["delay_sources"]):
        cf.validate(s, DELAY_SCHEMA, where=f"delay_sources[{i}].")
    out = {k: rp.result(v) for k, v in OUT_NAMES.items()}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    paths = {k: rp.data(v) for k, v in cfg["inputs"].items()}
    check_resources(rp.out)
    check = {}

    # 1) 예보 변수 목록
    fl = pq.read_table(paths["fcst_long"])
    fi = pq.read_table(paths["fcst_info"])
    sp = pq.read_table(paths["sigungu_point"])
    col_rows = column_list("fcst_long", fl) + column_list("fcst_info", fi)
    days_chk = eleven_day_check(fl, fi, cfg["forecast_len_days"])
    all_vars = [s["var"] for s in kw.VARIABLES + kw.EXTRA_VARIABLES]
    fmap = {m["var"]: m for m in cfg["forecast_map"]}
    if list(fmap) != all_vars or len(cfg["forecast_map"]) != len(all_vars):
        raise cf.ConfigError(f"forecast_map 변수 목록이 기상 19변수 + rh·td 순서와 다름: {list(fmap)}")
    tabs = {"fcst_long": fl, "fcst_info": fi}
    used = {"fcst_long": set(), "fcst_info": set()}
    map_rows = []
    for v in all_vars:
        m = fmap[v]
        for tk in ("fcst_long", "fcst_info"):
            if m[tk] is not None:
                if m[tk] not in tabs[tk].column_names:
                    raise cf.ConfigError(f"forecast_map {v}.{tk} = {m[tk]!r} 가 {FCST_TABLES[tk]} 에 없음")
                used[tk].add(m[tk])
        spec = kw.SPECS[v]
        map_rows.append({"변수": v, "한글": spec["ko"], "우리 단위": spec["unit"],
                         "p0830_fcst_long 칼럼": m["fcst_long"], "kma_weather_predict_info 칼럼": m["fcst_info"],
                         "예보에 있음": "있음" if (m["fcst_long"] or m["fcst_info"]) else "예보에 없음",
                         "대응 근거(확인 필요)": m["basis"]})
    key_cols = {"fcst_long": {"sigungu_code", "day_offset", "forecast_date"},
                "fcst_info": {"predict_seq", "meas_date", "forecast_date", "lat", "lon", "created_at"}}
    for tk in ("fcst_long", "fcst_info"):
        for c in tabs[tk].column_names:
            if c not in used[tk] and c not in key_cols[tk]:
                map_rows.append({"변수": "(우리 변수에 대응 없음)", "한글": None, "우리 단위": None,
                                 "p0830_fcst_long 칼럼": c if tk == "fcst_long" else None,
                                 "kma_weather_predict_info 칼럼": c if tk == "fcst_info" else None,
                                 "예보에 있음": None, "대응 근거(확인 필요)": "예보 표에만 있는 칼럼"})
    fc_vars = [v for v in all_vars if fmap[v]["fcst_long"] or fmap[v]["fcst_info"]]
    print(f"예보에 있는 변수 {len(fc_vars)}개: {fc_vars}")

    # 2) 예보 좌표 연결
    old_t, grid, sgg = kw.read_weather_table(paths["weather_table"], ["sgg_code", "date", *kw.VARS])
    n_sgg = len(sgg)
    old = {v: kw.table_matrix(old_t, v, n_sgg, grid.n) for v in kw.VARS if v in fc_vars}
    del old_t
    ong = cfg["ongjin_sgg_code"]
    i_ong = sgg.index(ong)
    others = np.array([i for i in range(n_sgg) if i != i_ong])
    sample = pq.read_table(paths["sample_sgg"], columns=["sgg_code", "lat", "lon"])
    region = pq.read_table(paths["region_code"], columns=["legal_dong_code", "level", "lat", "lon"])
    olat, olon, ong_info = kw.ongjin_coord(sample, region, ong)
    del sample
    coord_old = pq.read_table(paths["sgg_coord"])
    coord = kw.new_sgg_coords(coord_old, olat, olon, ong, "단계 0-4 와 같은 함수")
    xy = kw.coords_in_order(coord, sgg)
    nm = dict(zip(coord["sgg_code"].to_pylist(), coord["sgg_nm"].to_pylist()))
    scode = sp["sigungu_code"].to_pylist()
    snm = sp["sigungu_nm"].to_pylist()
    sxy = np.column_stack([sp["lat"].to_numpy(), sp["lon"].to_numpy()]).astype(np.float64)
    if len(set(scode)) != len(scode):
        raise ValueError("kma_sigungu_point sigungu_code 중복")
    fxy = np.zeros_like(xy)
    pt_rows, unl_rows, fpt_code = [], [], []
    for i, c in enumerate(sgg):
        d = kw.haversine_km(xy[i, 0], xy[i, 1], sxy[:, 0], sxy[:, 1])
        if c in scode:
            j, how = scode.index(c), "코드 같음"
        else:
            j, how = int(np.argmin(d)), "연결 안 됨 → 가장 가까운 예보 좌표"
        fxy[i] = sxy[j]
        fpt_code.append(scode[j])
        r = {"sgg_code": c, "시군구 이름": nm[c], "대표 좌표 lat": xy[i, 0], "대표 좌표 lon": xy[i, 1], "연결": how,
             "예보 좌표 sigungu_code": scode[j], "예보 좌표 이름": snm[j], "예보 lat": sxy[j, 0], "예보 lon": sxy[j, 1],
             "대표 좌표 ~ 예보 좌표(km)": round(float(d[j]), 3)}
        pt_rows.append(r)
        if how != "코드 같음":
            unl_rows.append(r)
    not_ours = [{"kma_sigungu_point 에만 있는 코드": c, "이름": n} for c, n in zip(scode, snm) if c not in sgg]
    check["예보 좌표"] = {"우리 시군구": n_sgg, "kma_sigungu_point": len(scode), "연결 안 된 시군구": len(unl_rows),
                       "kma_sigungu_point 에만 있는 코드": not_ours, "옹진군 좌표": ong_info}
    print(f"예보 좌표: 연결 안 된 시군구 {len(unl_rows)}곳")

    # 3) 좌표 차이 측정
    check_resources(rp.out)
    specs = [kw.SPECS[v] for v in fc_vars]
    nets = kw.load_networks(paths, specs, grid)
    i0, i1 = grid.index(cfg["diff_from"]), grid.index(cfg["diff_to"])
    diff_rows, port, m3_all = [], {}, {}
    for v in fc_vars:
        spec = kw.SPECS[v]
        r3 = kw.compute_var(nets, spec, xy, grid.n, k=3)
        r1 = kw.compute_var(nets, spec, fxy, grid.n, k=1)
        if v in old:
            cmp_o = kw.compare_matrices(r3["value"], old[v], others)
            port[v] = cmp_o
            if cmp_o["nan_mismatch"] or cmp_o["value_mismatch"]:
                raise RuntimeError(f"{v}: 3곳 평균 다시 계산 값이 기존 표와 다름(28720 밖) — 규칙 옮기기 확인: {cmp_o}")
        m3_all[v] = r3["value"]
        a1 = r1["value"][:, i0:i1 + 1].astype(np.float64)
        a3 = r3["value"][:, i0:i1 + 1].astype(np.float64)
        both = ~np.isnan(a1) & ~np.isnan(a3)
        dd = np.where(both, a1 - a3, 0.0)
        n = both.sum(axis=1)
        d1 = r1["dist"][:, i0:i1 + 1].astype(np.float64)
        for i, c in enumerate(sgg):
            diff_rows.append({"sgg_code": c, "변수": v, "예보 좌표 sigungu_code": fpt_code[i],
                              "평균 차이(1곳 − 3곳)": float(dd[i].sum() / n[i]) if n[i] else None,
                              "두 값 다 있는 날 수": int(n[i]),
                              "1곳 관측소까지 거리 중앙값(km)": round(float(np.nanmedian(d1[i])), 3) if (~np.isnan(d1[i])).any() else None})
        print(f"  {v:10s} 차이 측정 끝 · 기존 표 대조 {port.get(v, '해당 없음(새 변수)')} ({time.time() - t0:.0f}초)")
    check["규칙 대조(3곳 다시 계산 vs 기존 표, 28720 밖)"] = port
    check["차이 기간"] = [str(grid.dates[i0]), str(grid.dates[i1])]

    # 대응 확인용 비교(구현 세부): 예보 값 vs 같은 시군구·같은 날 3곳 평균 관측값
    sidx = {c: i for i, c in enumerate(sgg)}
    info_codes, pinfo = info_point_codes(fi, sp)
    check["fcst_info 좌표 연결"] = pinfo
    chk_rows = []
    fl_sg = fl["sigungu_code"].to_pylist()
    fl_d = kw.day_num(fl["forecast_date"]) - grid.d0
    fl_off = fl["day_offset"].to_numpy()
    fi_d = kw.day_num(fi["forecast_date"]) - grid.d0
    fi_off = kw.day_num(fi["forecast_date"]) - kw.day_num(fi["meas_date"])
    for v in fc_vars:
        for tk, codes, di, off in (("fcst_long", fl_sg, fl_d, fl_off), ("fcst_info", info_codes, fi_d, fi_off)):
            col = fmap[v][tk]
            if col is None:
                continue
            fv = tabs[tk][col].to_numpy(zero_copy_only=False).astype(np.float64)
            ok_i = np.array([c in sidx for c in codes]) & (di >= 0) & (di < grid.n)
            ii = np.array([sidx.get(c, 0) for c in codes])
            ov = np.full(len(codes), np.nan)
            ov[ok_i] = m3_all[v][ii[ok_i], di[ok_i]]
            for lab, sel in (("발표 당일(offset 0)", off == 0), ("전체 offset", np.ones(len(codes), bool))):
                s = sel & ok_i & ~np.isnan(fv) & ~np.isnan(ov)
                r = {"변수": v, "예보 표": FCST_TABLES[tk], "예보 칼럼": col, "행": lab, "비교 쌍 수": int(s.sum()),
                     "예보 평균": None, "관측(3곳) 평균": None, "평균 차이(예보 − 관측)": None, "상관계수": None}
                if s.sum() >= 2:
                    r.update({"예보 평균": round(float(fv[s].mean()), 4), "관측(3곳) 평균": round(float(ov[s].mean()), 4),
                              "평균 차이(예보 − 관측)": round(float((fv[s] - ov[s]).mean()), 4),
                              "상관계수": round(float(np.corrcoef(fv[s], ov[s])[0, 1]), 4) if fv[s].std() > 0 and ov[s].std() > 0 else None})
                chk_rows.append(r)

    # 4) 공개 지연
    delay_rows, delay_col_rows = delay_tables(rp.root, cfg["delay_sources"])
    check["공개 지연"] = "보고용 — 반영 안 함(기본값). 재는 법은 계획서에 없음(구현 세부)"

    # 저장
    write_csv(out["cols"], col_rows)
    cf.dump_json(out["days"], days_chk)
    write_csv(out["map"], map_rows)
    write_csv(out["mapchk"], chk_rows)
    write_csv(out["pt_all"], pt_rows)
    write_csv(out["pt_unl"], unl_rows)
    write_csv(out["diff"], diff_rows)
    write_csv(out["delay"], delay_rows)
    write_csv(out["delay_col"], delay_col_rows)
    check["예보에 있는 변수"] = fc_vars
    check["걸린 초"] = round(time.time() - t0, 1)
    cf.dump_json(out["check"], check)
    rel_inputs = list(cfg["inputs"].values()) + [s["file"] for s in cfg["delay_sources"]] + \
        [s["extract_json"] for s in cfg["delay_sources"] if s["extract_json"]]
    rel_inputs = list(dict.fromkeys(rel_inputs))
    cf.save_run_meta(out["meta"], cfg, cf.inputs_md5(rp.root, rel_inputs), CODE_ROOT, n_jobs=None,
                     extra={"outputs": {k: v.name for k, v in out.items()}, "config_path": Path(args.config).name})
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
