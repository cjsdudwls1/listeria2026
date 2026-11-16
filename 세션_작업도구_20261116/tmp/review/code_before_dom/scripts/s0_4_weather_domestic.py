# -*- coding: utf-8 -*-
"""s0_4_weather_domestic.py — 단계 0-4 국내 기상 일자료표(계획서 §6 단계 0-4, §4-3, 부록 A-2).

하는 일(계획서 그대로):
  1) 옹진군(28720) 새 대표 좌표 = 28720 에 배정된 시료 전체(모든 d) 좌표 중앙값(시도청 점·빈 좌표 제외) → 새 파일.
  2) 옹진군 기상 19변수를 새 좌표에서 가까운 3곳 평균으로 다시 계산 → 새 파일.
  3) 습도·이슬점 = ASOS avg_rhm·avg_td(습도 빈 칸 NAS hum), 같은 3곳 평균 규칙 → 새 파일(256 시군구, 옹진군은 새 좌표).
  4) NAS 2/29(2016·2020·2024) 채움: rain = 같은 날 ASOS sum_rn 가까운 3곳 평균(빈 sum_rn = 무강수 0mm),
     soil_t_nas = 2/28 값 + ASOS soil_t10 의 2/28→2/29 변화, soil_moist = 2/28 값 → 새 파일.
  기존 `시군구_일별_기상.parquet`·`시군구_대표좌표.parquet` 은 읽기만(덮지 않음). 3곳 평균 규칙 = lst/kweather.py
  (= 리스테리아ML2/notebook/03_시군구_기상.py 규칙 그대로 옮김). 규칙이 같은지: 256 시군구 × 19변수를 다시 계산해
  28720 밖 모든 칸이 기존 표와 같은지 확인(다르면 멈춤).

출력(<out>/, 이미 있으면 멈춤):
  시군구_대표좌표_0-4.parquet        256행(기존과 같은 칼럼, 28720 만 새 좌표)
  옹진군_일별_기상_0-4.parquet       28720 × 날짜, 19변수(새 좌표, 2/29 채움 전 원 계산값)
  시군구_일별_습도이슬점_0-4.parquet  256 × 날짜, rh·td
  시군구_2월29일_채움_0-4.parquet     256 × 3일, rain·soil_t_nas·soil_moist(채운 값)
  시군구_일별_기상_0-4_합본.parquet   256 × 날짜, 19변수(기존 표 + 옹진군 새 값 + 2/29 채움) + rh·td
  점검_변수별.csv · 점검.json · run_meta.json(code_md5·입력 md5·스레드·설정)

구현 세부(계획서에 없음 — 담당자 확인):
  - 합본 파일(위 5번째)은 단계 0-5 가 한 표로 읽게 만든 편의 파일. 값은 앞 파일들을 합친 것뿐.
  - 2/29 soil_t_nas·soil_moist 의 '2/28 값'·'soil_t10 변화' = 시군구 표 값(3곳 평균 뒤) 기준. 2/28 값·soil_t10 둘 중
    하나라도 비면 그 칸은 결측. 채움은 2/29 값이 빈 칸에만(기존 표는 세 날 모두 256 시군구가 빈 칸).
  - 날짜 축 = 기존 표 날짜 범위. 점검 기간 = check_from(2015-01-01, §4-3 값 시작) ~ 끝, 그리고 ~ check_cv_end(2025-06-30).
  - 칼럼 이름 rh·td, 그 밖 kweather 모듈 docstring.
담당자 결정(설정 raw_overrides): 지정한 원자료 칸을 결측으로 둠 — 원자료를 읽은 직후·3곳 평균 전에 적용, 적용 칸 수는
  점검.json 에 기록, 대상 칸이 없으면 멈춤(2026-11-07: ASOS 295 2025-06-14 avg_td 79.7).

실행(서버, 검수 뒤):
  ~/cyj_listeria_final/venv/bin/python -I scripts/s0_4_weather_domestic.py \
      --root ~/cyj_listeria_final/data --out ~/cyj_listeria_final/results/s0/0-4_v2
"""
from __future__ import annotations

import csv
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
import pyarrow.parquet as pq  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import kweather as kw  # noqa: E402
from lst.paths import build_parser, parse_paths  # noqa: E402
from lst.resources import check_resources  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_KEYS = ("weather_table", "sgg_coord", "sample_sgg", "region_code", "asos_daily", "asos_station",
              "nas_daily", "nas_station", "air_daily", "air_station")
SCHEMA = {
    "plan": cf.Field(str),
    "inputs": {k: cf.Field(str) for k in INPUT_KEYS},
    "ongjin_sgg_code": cf.Field(str),
    "feb29_dates": cf.Field(list, item_types=str),
    "check_from": cf.Field(str),
    "check_cv_end": cf.Field(str),
    "raw_overrides": cf.Field(list, item_types=dict),
}
OVERRIDE_SCHEMA = {"input": cf.Field(str, choices=("asos_daily", "nas_daily", "air_daily")), "station": cf.Field(str),
                   "date": cf.Field(str), "column": cf.Field(str), "value": cf.Field(type(None)), "reason": cf.Field(str)}
OUT_NAMES = {
    "coord": "시군구_대표좌표_0-4.parquet",
    "ongjin": "옹진군_일별_기상_0-4.parquet",
    "rhtd": "시군구_일별_습도이슬점_0-4.parquet",
    "feb29": "시군구_2월29일_채움_0-4.parquet",
    "merged": "시군구_일별_기상_0-4_합본.parquet",
    "check_csv": "점검_변수별.csv",
    "check_json": "점검.json",
    "meta": "run_meta.json",
}
FEB29_VARS = ("rain", "soil_t_nas", "soil_moist")


def write_parquet(path: Path, cols: dict, types: dict, meta: dict):
    if path.exists():
        raise FileExistsError(f"덮어쓰기 금지: {path}")
    schema = pa.schema([pa.field(c, types[c]) for c in cols],
                       metadata={k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).encode("utf-8")
                                 for k, v in meta.items()})
    arrays = []
    for c in cols:
        v = cols[c]
        if types[c] == pa.float32():
            arrays.append(pa.array(np.asarray(v, dtype=np.float32), type=pa.float32(), from_pandas=True))  # NaN → null
        else:
            arrays.append(pa.array(v, type=types[c]))
    t = pa.Table.from_arrays(arrays, schema=schema)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(t, tmp)
    tmp.replace(path)
    back = pq.read_table(path)
    assert back.num_rows == t.num_rows and back.schema.names == list(cols)
    return t.num_rows


def long_cols(codes, grid: kw.DateGrid, mats: dict) -> tuple[dict, dict]:
    """(시군구, 날짜) 행렬들 → 긴 표 칼럼(sgg_code, date 정렬)."""
    n = len(codes)
    cols = {"sgg_code": np.repeat(np.array(codes, dtype=object), grid.n).tolist(),
            "date": np.tile(grid.dates, n)}
    types = {"sgg_code": pa.string(), "date": pa.date32()}
    for k, M in mats.items():
        assert M.shape == (n, grid.n), (k, M.shape)
        cols[k] = M.ravel()
        types[k] = pa.float32()
    return cols, types


def missing_record(var, M, grid, i_from, i_cv, extra=None):
    has = ~np.isnan(M)
    day_any = has.any(axis=0)
    first = grid.dates[day_any.argmax()] if day_any.any() else None
    last = grid.dates[len(day_any) - 1 - day_any[::-1].argmax()] if day_any.any() else None
    r = {
        "칼럼": var, "전체 칸": int(M.size), "결측 칸": int((~has).sum()),
        f"{grid.dates[i_from]} 이후 칸": int(has[:, i_from:].size), f"{grid.dates[i_from]} 이후 결측 칸": int((~has[:, i_from:]).sum()),
        f"{grid.dates[i_from]} ~ {grid.dates[i_cv]} 결측 칸": int((~has[:, i_from:i_cv + 1]).sum()),
        f"{grid.dates[i_from]} 이후 모든 시군구 결측인 날 수": int((~day_any[i_from:]).sum()),
        "값 있는 첫 날": str(first) if first is not None else None, "값 있는 마지막 날": str(last) if last is not None else None,
    }
    if extra:
        r.update(extra)
    return r


def dist_record(res, i_from):
    D = res["dist"][:, i_from:]
    C = res["count"][:, i_from:]
    d = D[~np.isnan(D)].astype(np.float64)
    has = ~np.isnan(res["value"][:, i_from:])
    return {
        "대체 칼럼 값이 들어간 칸": int(res["fb_mask"].sum()),
        "고른 관측소까지 평균 거리 중앙값(km)": round(float(np.median(d)), 3) if d.size else None,
        "고른 관측소까지 평균 거리 90%(km)": round(float(np.percentile(d, 90)), 3) if d.size else None,
        "고른 관측소까지 평균 거리 최대(km)": round(float(d.max()), 3) if d.size else None,
        "값 있는 칸 중 관측소 3곳 미만 칸": int(((C < kw.K_NEAREST) & has).sum()),
    }


def main(argv=None):
    t0 = time.time()
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CODE_ROOT / "configs" / "s0_4_weather_domestic.json"))
    rp, args = parse_paths(argv, ap)
    cfg = cf.load_config(args.config, SCHEMA)
    for i, o in enumerate(cfg["raw_overrides"]):
        cf.validate(o, OVERRIDE_SCHEMA, where=f"raw_overrides[{i}].")
    out = {k: rp.result(v) for k, v in OUT_NAMES.items()}
    exist = [str(p) for p in out.values() if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있는 결과 — 덮지 않음: {exist}")
    paths = {k: rp.data(v) for k, v in cfg["inputs"].items()}
    check_resources(rp.out)
    ong = cfg["ongjin_sgg_code"]

    # 1) 기존 표·날짜 축
    old_t, grid, sgg = kw.read_weather_table(paths["weather_table"], ["sgg_code", "date", *kw.VARS])
    n_sgg = len(sgg)
    old = {v: kw.table_matrix(old_t, v, n_sgg, grid.n) for v in kw.VARS}
    del old_t
    i_ong = sgg.index(ong)
    others = np.array([i for i in range(n_sgg) if i != i_ong])
    i_from, i_cv = grid.index(cfg["check_from"]), grid.index(cfg["check_cv_end"])
    print(f"기존 표: 시군구 {n_sgg} × {grid.n}일 ({grid.date_min} ~ {grid.date_max})")

    # 2) 옹진군 새 대표 좌표
    sample = pq.read_table(paths["sample_sgg"], columns=["sgg_code", "lat", "lon"])
    region = pq.read_table(paths["region_code"], columns=["legal_dong_code", "level", "lat", "lon"])
    lat, lon, ong_info = kw.ongjin_coord(sample, region, ong)
    del sample
    old_coord = pq.read_table(paths["sgg_coord"])
    oc = old_coord.to_pydict()
    j = oc["sgg_code"].index(ong)
    ong_info["old_lat"], ong_info["old_lon"] = oc["lat"][j], oc["lon"][j]
    ong_info["moved_km"] = float(kw.haversine_km(oc["lat"][j], oc["lon"][j], lat, lon))
    new_coord = kw.new_sgg_coords(old_coord, lat, lon, ong, "단계 0-4: 28720 배정 시료 전체 좌표 중앙값(시도청 점·빈 좌표 제외)")
    xy = kw.coords_in_order(new_coord, sgg)
    print(f"옹진군 새 좌표 {lat:.6f}, {lon:.6f} (기존 {ong_info['old_lat']:.6f}, {ong_info['old_lon']:.6f}, 옮김 {ong_info['moved_km']:.3f}km) · {ong_info}")

    # 3) 관측망
    specs = kw.VARIABLES + kw.EXTRA_VARIABLES + [kw.FEB29_RAIN_SPEC]
    nets = kw.load_networks(paths, specs, grid, cfg["raw_overrides"])   # raw_overrides = 읽은 직후·3곳 평균 전
    applied = [r for v in nets.values() for r in v.info.get("raw_overrides", [])]
    if len(applied) != len(cfg["raw_overrides"]):
        raise RuntimeError(f"raw_overrides 적용 {len(applied)}칸 ≠ 설정 {len(cfg['raw_overrides'])}칸")
    print(f"원자료 바꾼 칸 {len(applied)}: {applied}")
    net_info = {n: {"관측소 수": len(v.sids), **{k: x for k, x in v.info.items()}} for n, v in nets.items()}
    print(f"관측망 읽음: { {n: len(v.sids) for n, v in nets.items()} } ({time.time() - t0:.0f}초)")
    check_resources(rp.out)

    # 4) 19변수 다시 계산 → 기존 표 대조, 옹진군 새 값
    rows, port, new_ong = [], {}, {}
    for spec in kw.VARIABLES:
        v = spec["var"]
        res = kw.compute_var(nets, spec, xy, grid.n)
        cmp_o = kw.compare_matrices(res["value"], old[v], others)
        port[v] = {"28720 밖": cmp_o, "28720": kw.compare_matrices(res["value"], old[v], [i_ong])}
        if cmp_o["nan_mismatch"] or cmp_o["value_mismatch"]:
            raise RuntimeError(f"{v}: 다시 계산한 값이 기존 표와 다름(28720 밖) — 규칙 옮기기 확인 필요: {cmp_o}")
        if ong_info["moved_km"] == 0.0 and (port[v]["28720"]["nan_mismatch"] or port[v]["28720"]["value_mismatch"]):
            raise RuntimeError(f"{v}: 옹진군 좌표가 같은데 값이 다름: {port[v]['28720']}")
        new_ong[v] = res["value"][i_ong].copy()
        rows.append(("19", v, spec, res))
        print(f"  {v:13s} 기존 표 대조(28720 밖) 같음 · 28720 다른 칸 {port[v]['28720']['value_mismatch'] + port[v]['28720']['nan_mismatch']} "
              f"({time.time() - t0:.0f}초)")

    # 5) 습도·이슬점
    rhtd = {}
    for spec in kw.EXTRA_VARIABLES:
        res = kw.compute_var(nets, spec, xy, grid.n)
        rhtd[spec["var"]] = res["value"]
        rows.append(("새", spec["var"], spec, res))
    rain29 = kw.compute_var(nets, kw.FEB29_RAIN_SPEC, xy, grid.n)

    # 6) 합본 바탕(옹진군 새 값) + 2/29 채움
    merged = {v: old[v].copy() for v in kw.VARS}
    for v in kw.VARS:
        merged[v][i_ong] = new_ong[v]
    feb = [grid.index(d) for d in cfg["feb29_dates"]]
    all29 = [i for i, d in enumerate(grid.dates) if str(d)[5:] == "02-29"]
    if sorted(feb) != all29:
        raise ValueError(f"설정 2/29 날짜 {cfg['feb29_dates']} 와 날짜 축의 2/29 {[str(grid.dates[i]) for i in all29]} 가 다름")
    fill = {v: np.full((n_sgg, len(feb)), np.nan, dtype=np.float32) for v in FEB29_VARS}
    feb_info = {}
    for k, i in enumerate(feb):
        cand = {
            "rain": rain29["value"][:, i],
            "soil_t_nas": (merged["soil_t_nas"][:, i - 1].astype(np.float64)
                           + (merged["soil_t10"][:, i].astype(np.float64) - merged["soil_t10"][:, i - 1].astype(np.float64))).astype(np.float32),
            "soil_moist": merged["soil_moist"][:, i - 1],
        }
        day = str(grid.dates[i])
        feb_info[day] = {}
        for v in FEB29_VARS:
            empty = np.isnan(merged[v][:, i])
            put = empty & ~np.isnan(cand[v])
            fill[v][:, k] = np.where(empty, cand[v], np.nan)
            merged[v][put, i] = cand[v][put]
            feb_info[day][v] = {"빈 칸(채우기 전)": int(empty.sum()), "채운 칸": int(put.sum()),
                                "채울 값이 없어 빈 칸": int((empty & np.isnan(cand[v])).sum()),
                                "값 있던 칸(안 바꿈)": int((~empty).sum())}
        feb_info[day]["rain 고른 ASOS 관측소 3곳 미만 시군구"] = int((rain29["count"][:, i] < kw.K_NEAREST).sum())
    print(f"2/29 채움: {feb_info}")

    # 7) 저장
    check_resources(rp.out)
    w = {}
    cc = new_coord.to_pydict()
    w["coord"] = write_parquet(out["coord"], {k: cc[k] for k in new_coord.column_names},
                               {f.name: f.type for f in new_coord.schema},
                               {"description": "시군구 대표 좌표(단계 0-4) — 28720 만 새 좌표, 나머지 기존 시군구_대표좌표.parquet 그대로",
                                "ongjin": ong_info})
    c, t = long_cols([ong], grid, {v: new_ong[v][None, :] for v in kw.VARS})
    w["ongjin"] = write_parquet(out["ongjin"], c, t, {"description": "옹진군(28720) 19변수 — 새 대표 좌표에서 가까운 3곳 평균(03 규칙), 2/29 채움 전",
                                                      "lat": lat, "lon": lon})
    c, t = long_cols(sgg, grid, rhtd)
    w["rhtd"] = write_parquet(out["rhtd"], c, t, {"description": "rh = ASOS avg_rhm(빈 칸 NAS hum), td = ASOS avg_td, 가까운 3곳 평균(03 규칙), 옹진군 = 새 좌표"})
    fc = {"sgg_code": np.repeat(np.array(sgg, dtype=object), len(feb)).tolist(),
          "date": np.tile(grid.dates[feb], n_sgg)}
    ft = {"sgg_code": pa.string(), "date": pa.date32()}
    for v in FEB29_VARS:
        fc[v] = fill[v].ravel()
        ft[v] = pa.float32()
    w["feb29"] = write_parquet(out["feb29"], fc, ft, {"description": "NAS 2/29 채움 값(부록 A-2) — rain = ASOS sum_rn 3곳 평균(빈 값 0mm), "
                                                        "soil_t_nas = 2/28 값 + soil_t10 2/28→2/29 변화, soil_moist = 2/28 값, 채우기 전 값이 있던 칸은 NaN"})
    c, t = long_cols(sgg, grid, {**merged, **rhtd})
    w["merged"] = write_parquet(out["merged"], c, t, {"description": "기존 시군구_일별_기상 + 옹진군 새 값 + 2/29 채움 + rh·td (단계 0-4 합본)",
                                                      "variables": [s["var"] for s in kw.VARIABLES + kw.EXTRA_VARIABLES]})

    # 8) 점검 기록
    recs = []
    for kind, v, spec, res in rows:
        M = merged[v] if kind == "19" else rhtd[v]
        sc = kw.station_counts(nets, spec, i_from)
        extra = {"구분": "기존 19변수(옹진군 새 값·2/29 채움 뒤)" if kind == "19" else "새 파일",
                 "기본": sc["기본"]["원천"], "기본 값 있는 관측소 수": sc["기본"]["값 있는 관측소 수"],
                 "기본 날마다 관측소 수 최소": sc["기본"]["날마다 값 있는 관측소 수 최소"],
                 "기본 날마다 관측소 수 중앙값": sc["기본"]["날마다 값 있는 관측소 수 중앙값"],
                 "대체": sc["대체"]["원천"] if "대체" in sc else "없음",
                 "대체 값 있는 관측소 수": sc["대체"]["값 있는 관측소 수"] if "대체" in sc else None,
                 **dist_record(res, i_from)}
        recs.append(missing_record(v, M, grid, i_from, i_cv, extra))
    with open(out["check_csv"], "x", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        wr.writeheader()
        wr.writerows(recs)
    check = {"기간": {"date_min": str(grid.date_min), "date_max": str(grid.date_max), "days": grid.n, "sgg": n_sgg,
                      "check_from": cfg["check_from"], "check_cv_end": cfg["check_cv_end"]},
             "옹진군 좌표": ong_info, "규칙 대조(다시 계산 vs 기존 표)": port, "2/29 채움": feb_info,
             "원자료 바꾼 칸(raw_overrides)": {"적용 칸 수": len(applied), "칸": applied},
             "관측망": net_info, "rain_feb29 ASOS sum_rn": kw.station_counts(nets, kw.FEB29_RAIN_SPEC, i_from),
             "저장 행 수": w, "걸린 초": round(time.time() - t0, 1)}
    cf.dump_json(out["check_json"], check)
    rel_inputs = list(cfg["inputs"].values())
    cf.save_run_meta(out["meta"], cfg, cf.inputs_md5(rp.root, rel_inputs), CODE_ROOT, n_jobs=None,
                     extra={"outputs": {k: v.name for k, v in out.items()}, "config_path": Path(args.config).name})
    print(f"끝: {rp.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
