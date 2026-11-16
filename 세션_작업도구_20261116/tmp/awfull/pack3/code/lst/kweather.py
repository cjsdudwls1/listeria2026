"""국내 시군구 기상 규칙(가까운 관측소 k곳 평균) — 계획서 §4-3, 부록 A-2, 부록 A-13.

규칙 원본 = `리스테리아ML2/notebook/03_시군구_기상.py`(기존 `시군구_일별_기상.parquet` 을 만든 스크립트).
이 모듈은 그 규칙을 그대로 옮긴 것(실행 경로·출력만 다름). 바꾼 곳 없음:
  - 날짜마다, 그날 그 칼럼 값이 있는 관측소 중 구면 거리(지구 반지름 6371.0088km)로 가까운 k곳 단순 평균.
    k곳이 안 되면 있는 곳만 평균, 한 곳도 없으면 결측. 거리 상한 없음. 같은 거리면 관측소 번호(이름) 순서가 앞인 곳.
  - 기본 칼럼 결과가 결측인 칸에만 대체 칼럼 결과를 넣음.
  - 값은 원자료 그대로(채움·변환 없음). ASOS sum_sshr 는 DB 에 텍스트라 숫자로만 바꿈.
  - ASOS 좌표: kma_ground_obs_station_info lat·lon, 날짜마다 'oper_start_dt 가 그날 이전인 행 중 가장 최근 행'
    (없으면 가장 이른 행). NAS 좌표: instl_la·instl_lo, 561034A001(동방로거테스트) 안 씀.
    에어코리아 좌표: station_nm → dm_x(위도)·dm_y(경도), 같은 이름 여러 행은 좌표가 같을 때만(다르면 멈춤),
    측정소 정보에 없는 측정소는 뺌. 같은 관측소·같은 날 행이 둘 이상이면 멈춤.
  - 풍향: ASOS max_ws1(도) → 관측소·날짜마다 sin·cos 뒤 각각 k곳 평균.

계획서로 더한 것(부록 A-2):
  - 습도 `rh` = ASOS avg_rhm(습도 빈 칸 NAS hum), 이슬점 `td` = ASOS avg_td, 같은 3곳 평균 규칙.
  - NAS 2/29 채움용 ASOS `sum_rn` — 빈 값 = 무강수(0mm, 기상청 자료 규칙): 그날 행이 있는 관측소의
    빈 sum_rn 을 0 으로 읽음(행이 없는 관측소는 후보 아님 — 값 있는 관측소 규칙 그대로).

구현 세부(계획서에 없음):
  - `rh`·`td` 칼럼 이름.
  - 날짜 범위는 기존 `시군구_일별_기상.parquet` 의 날짜 범위(2014-04-07 ~ 2026-10-13)를 그대로 받음
    (`grid_from_table`) — 새 파일이 기존 표와 같은 날짜 축이 되게.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

K_NEAREST = 3
EARTH_R_KM = 6371.0088
EXCLUDE_NAS = ("561034A001",)
BLOCK = 32  # 한 번에 계산하는 날짜 수(메모리) — 값에는 영향 없음

# 03_시군구_기상.py VARIABLES 그대로(칼럼 이름·기본·대체)
VARIABLES = [
    dict(var="avg_temp", ko="평균기온", unit="°C", prim=("ASOS", "avg_ta"), fb=("NAS", "temp")),
    dict(var="max_temp", ko="최고기온", unit="°C", prim=("ASOS", "max_ta"), fb=("NAS", "hghst_artmp")),
    dict(var="min_temp", ko="최저기온", unit="°C", prim=("ASOS", "min_ta"), fb=("NAS", "lowst_artmp")),
    dict(var="rain", ko="강수량", unit="mm", prim=("NAS", "rn"), fb=None),
    dict(var="avg_wind", ko="평균풍속", unit="m/s", prim=("ASOS", "avg_ws"), fb=("NAS", "wind")),
    dict(var="max_wind", ko="최대풍속", unit="m/s", prim=("ASOS", "max_ws"), fb=("NAS", "max_wind")),
    dict(var="sunshine", ko="일조시간", unit="시간", prim=("ASOS", "sum_sshr"), fb=None),
    dict(var="solar_rad", ko="일사량", unit="MJ/m²", prim=("ASOS", "sum_gsr"), fb=("NAS", "srqty")),
    dict(var="soil_t5", ko="지중온도 5cm", unit="°C", prim=("ASOS", "avg_cm5_te"), fb=None),
    dict(var="soil_t10", ko="지중온도 10cm", unit="°C", prim=("ASOS", "avg_cm10_te"), fb=None),
    dict(var="soil_t20", ko="지중온도 20cm", unit="°C", prim=("ASOS", "avg_cm20_tes"), fb=None),
    dict(var="soil_t30", ko="지중온도 30cm", unit="°C", prim=("ASOS", "avg_cm30_te"), fb=None),
    dict(var="soil_t_nas", ko="지중온도(NAS, 깊이 설명 없음)", unit="°C", prim=("NAS", "soil_temp"), fb=None),
    dict(var="soil_moist", ko="토양수분", unit="%", prim=("NAS", "soil_wt"), fb=None),
    dict(var="no2", ko="이산화질소", unit="ppm", prim=("AIR", "no2_value"), fb=None),
    dict(var="co", ko="일산화탄소", unit="ppm", prim=("AIR", "co_value"), fb=None),
    dict(var="so2", ko="아황산가스", unit="ppm", prim=("AIR", "so2_value"), fb=None),
    dict(var="wind_dir_sin", ko="최다풍향 sin", unit="(−1~1)", prim=("ASOS", "max_ws1_sin"), fb=None),
    dict(var="wind_dir_cos", ko="최다풍향 cos", unit="(−1~1)", prim=("ASOS", "max_ws1_cos"), fb=None),
]
VARS = tuple(v["var"] for v in VARIABLES)
assert len(set(VARS)) == len(VARS) == 19

# 부록 A-2 습도·이슬점(새 파일)
EXTRA_VARIABLES = [
    dict(var="rh", ko="평균 상대습도", unit="%", prim=("ASOS", "avg_rhm"), fb=("NAS", "hum")),
    dict(var="td", ko="평균 이슬점온도", unit="°C", prim=("ASOS", "avg_td"), fb=None),
]
# 부록 A-2 NAS 2/29 rain 채움 — ASOS sum_rn(빈 값 = 0mm)
FEB29_RAIN_SPEC = dict(var="rain_feb29", ko="강수량(2/29 채움용 ASOS)", unit="mm", prim=("ASOS", "sum_rn"), fb=None)
NULL_AS_ZERO = {("ASOS", "sum_rn")}

SPECS = {s["var"]: s for s in VARIABLES + EXTRA_VARIABLES + [FEB29_RAIN_SPEC]}

NET_TABLE = {"ASOS": "api_data_info.kma_asos_daily_info", "NAS": "api_data_info.nas_agri_weather_daily_info",
             "AIR": "api_data_info.keco_airkorea_fine_dust_info"}
NET_KEYS = {"ASOS": ("stn_id", "meas_date"), "NAS": ("obsr_spot_code", "meas_date"), "AIR": ("station_nm", "msur_dt")}
# 입력 파일 칸 이름(설정 JSON 의 inputs 칸과 같음)
NET_FILE_KEYS = {"ASOS": ("asos_daily", "asos_station"), "NAS": ("nas_daily", "nas_station"),
                 "AIR": ("air_daily", "air_station")}


def haversine_km(lat1, lon1, lat2, lon2):
    """두 지점 사이의 구면 거리(km), 도 단위(03 과 같은 식)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return EARTH_R_KM * 2 * np.arcsin(np.sqrt(a))


def day_num(arr) -> np.ndarray:
    """date32 배열 → 1970-01-01 부터 일수(int64)."""
    return pc.cast(arr, pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64)


def to_float(col, name):
    """원자료 칼럼 → float64(null = NaN). 텍스트 칼럼은 숫자로만 바꿈(안 바뀌는 값이 있으면 멈춤)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        uniq = pc.unique(col.drop_null()).to_pylist()
        conv = {}
        for s in uniq:
            try:
                conv[s] = float(s)
            except ValueError:
                raise ValueError(f"{name}: 숫자로 안 바뀌는 값 {s!r}") from None
        vals = col.to_pylist()
        return np.array([np.nan if s is None else conv[s] for s in vals], dtype=np.float64), len(uniq)
    return col.to_numpy(zero_copy_only=False).astype(np.float64), None


# ---------------------------------------------------------------- 날짜 축
@dataclass(frozen=True)
class DateGrid:
    date_min: np.datetime64
    date_max: np.datetime64

    @property
    def dates(self) -> np.ndarray:
        return np.arange(self.date_min, self.date_max + np.timedelta64(1, "D"), dtype="datetime64[D]")

    @property
    def n(self) -> int:
        return int((self.date_max - self.date_min).astype(int)) + 1

    @property
    def d0(self) -> int:
        return int(self.date_min.astype(np.int64))

    def index(self, day) -> int:
        i = int((np.datetime64(day, "D") - self.date_min).astype(int))
        if not 0 <= i < self.n:
            raise IndexError(f"날짜 범위 밖: {day}")
        return i


def read_weather_table(path, columns=None) -> tuple[pa.Table, DateGrid, list[str]]:
    """기존 시군구 일별 기상표 → (표, 날짜 축, 시군구 순서). 시군구 × 날짜 빠짐없는 격자·정렬인지 확인."""
    t = pq.read_table(path, columns=columns)
    codes = t["sgg_code"].to_pylist()
    days = day_num(t["date"])
    sgg = sorted(set(codes))
    grid = DateGrid(np.datetime64(int(days.min()), "D"), np.datetime64(int(days.max()), "D"))
    if t.num_rows != len(sgg) * grid.n:
        raise ValueError(f"기상표가 시군구 × 날짜 격자가 아님: {t.num_rows} != {len(sgg)} × {grid.n}")
    exp_codes = np.repeat(np.array(sgg, dtype=object), grid.n)
    exp_days = np.tile(np.arange(grid.d0, grid.d0 + grid.n), len(sgg))
    if not (np.array(codes, dtype=object) == exp_codes).all() or not (days == exp_days).all():
        raise ValueError("기상표 행 순서가 (sgg_code, date) 정렬 격자가 아님")
    return t, grid, sgg


def table_matrix(t: pa.Table, var: str, n_sgg: int, n_days: int) -> np.ndarray:
    """정렬 격자 표의 칼럼 → (시군구, 날짜) float32(null = NaN)."""
    v = t[var].combine_chunks()
    a = v.to_numpy(zero_copy_only=False).astype(np.float32)
    if v.null_count:
        a[np.asarray(v.is_null())] = np.nan
    return a.reshape(n_sgg, n_days)


# ---------------------------------------------------------------- 관측망 읽기
def daily_table(tbl, sid_col, date_col, value_cols, sids, label, grid: DateGrid, null_zero=()):
    """(날짜 × 관측소) float32 표. 같은 관측소·같은 날 행이 둘 이상이면 멈춤(원자료 전체에서 확인).

    null_zero: 그 칼럼은 그날 행이 있는 관측소의 빈 값을 0 으로(부록 A-2 ASOS sum_rn).
    """
    sid_all = tbl[sid_col]
    days = day_num(tbl[date_col])
    codes = pc.index_in(sid_all, value_set=pc.unique(sid_all)).to_numpy(zero_copy_only=False).astype(np.int64)
    key = codes * 1_000_000 + (days - days.min())
    n_dup = len(key) - len(np.unique(key))
    if n_dup:
        raise RuntimeError(f"{label}: 같은 관측소·같은 날 행이 {n_dup}건 있음 — 어느 값을 쓸지 정해야 함")
    pos = pc.index_in(sid_all, value_set=pa.array(sids, pa.string())).to_numpy(zero_copy_only=False)
    keep = (~np.isnan(pos.astype(np.float64))) & (days >= grid.d0) & (days <= grid.d0 + grid.n - 1)
    ri = (days[keep] - grid.d0).astype(np.int64)
    ci = pos[keep].astype(np.int64)
    out, info = {}, {}
    for c in value_cols:
        v64, n_text = to_float(tbl[c], f"{label}.{c}")
        v = v64[keep]
        if c in null_zero:
            info[f"{c}_null_as_zero_rows"] = int(np.isnan(v).sum())
            v = np.where(np.isnan(v), 0.0, v)
        V = np.full((grid.n, len(sids)), np.nan, dtype=np.float32)
        V[ri, ci] = v.astype(np.float32)
        out[c] = V
        if n_text is not None:
            info[c] = n_text
    return out, info


def apply_overrides(vals: dict, sids, grid: DateGrid, overrides, label: str) -> list[dict]:
    """원자료 (날짜 × 관측소) 표의 지정 칸을 바꿈(담당자 결정, 설정 raw_overrides). 바꿀 값은 null(결측)만.

    대상 칸이 없으면 멈춤: 관측소·칼럼이 없음, 날짜가 날짜 축 밖, 그 칸이 이미 결측(원자료에 행·값 없음),
    같은 칸이 두 번. 반환 = 칸마다 적용 기록(원래 값 포함).
    """
    pos = {s: i for i, s in enumerate(sids)}
    done, seen = [], set()
    for o in overrides:
        key = (o["station"], o["date"], o["column"])
        if key in seen:
            raise ValueError(f"{label} raw_overrides 같은 칸 두 번: {key}")
        seen.add(key)
        if o["value"] is not None:
            raise ValueError(f"{label} raw_overrides 바꿀 값은 null 만: {o}")
        if o["column"] not in vals:
            raise ValueError(f"{label} raw_overrides 대상 칼럼을 읽지 않음: {o['column']!r}")
        if o["station"] not in pos:
            raise ValueError(f"{label} raw_overrides 대상 관측소 없음: {o['station']!r}")
        di = grid.index(o["date"])   # 날짜 축 밖이면 IndexError
        V = vals[o["column"]]
        old = V[di, pos[o["station"]]]
        if np.isnan(old):
            raise ValueError(f"{label} raw_overrides 대상 칸이 이미 결측(원자료 행·값 없음): {key}")
        V[di, pos[o["station"]]] = np.nan
        done.append({**o, "원래 값": float(old), "적용 칸 수": 1})
    return done


@dataclass
class Network:
    name: str
    sids: list
    periods: list            # [(시작 날짜 위치, 끝 날짜 위치(미포함), 좌표 (n, 2))]
    values: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def needed_columns(specs) -> dict[str, list[str]]:
    """변수 정의 → 관측망별 원자료 칼럼(풍향 sin·cos 은 max_ws1)."""
    need: dict[str, set] = {}
    for s in specs:
        for src in (s["prim"], s["fb"]):
            if not src:
                continue
            net, col = src
            need.setdefault(net, set()).add("max_ws1" if col.startswith("max_ws1_") else col)
    return {k: sorted(v) for k, v in need.items()}


def load_asos(daily_path, station_path, cols, grid: DateGrid, overrides=()) -> Network:
    sid_col, dcol = NET_KEYS["ASOS"]
    asos = pq.read_table(daily_path, columns=[sid_col, dcol] + cols)
    st = pq.read_table(station_path, columns=["stn_id", "lat", "lon", "oper_start_dt"]).to_pylist()
    sids = sorted(pc.unique(asos[sid_col]).to_pylist())
    sset = set(sids)
    rows: dict = {}
    for r in st:
        if r["stn_id"] in sset:
            rows.setdefault(r["stn_id"], []).append(r)
    if set(rows) != sset:
        raise ValueError(f"ASOS 좌표가 없는 관측소: {sorted(sset - set(rows))[:10]}")
    for s in sids:
        starts = [r["oper_start_dt"] for r in rows[s]]
        if None in starts or len(set(starts)) != len(starts):
            raise ValueError(f"ASOS {s}: 운영 시작일이 비었거나 겹침")
        rows[s].sort(key=lambda r: r["oper_start_dt"])
    d_min, d_max = grid.date_min.astype(object), grid.date_max.astype(object)
    changes = sorted({r["oper_start_dt"] for s in sids for r in rows[s] if d_min < r["oper_start_dt"] <= d_max})
    bounds = [d_min] + changes + [(grid.date_max + np.timedelta64(1, "D")).astype(object)]
    periods = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        xy = []
        for s in sids:
            ok = [r for r in rows[s] if r["oper_start_dt"] <= a]
            r = ok[-1] if ok else rows[s][0]
            xy.append((float(r["lat"]), float(r["lon"])))
        periods.append((int((np.datetime64(a, "D") - grid.date_min).astype(int)),
                        min(int((np.datetime64(b, "D") - grid.date_min).astype(int)), grid.n), np.array(xy)))
    nz = {c for (n, c) in NULL_AS_ZERO if n == "ASOS" and c in cols}
    vals, info = daily_table(asos, sid_col, dcol, cols, sids, "ASOS", grid, null_zero=nz)
    info["raw_overrides"] = apply_overrides(vals, sids, grid, overrides, "ASOS")   # 풍향 sin·cos 만들기 전
    if "max_ws1" in vals:
        rad = np.deg2rad(vals["max_ws1"].astype(np.float64))
        vals["max_ws1_sin"] = np.sin(rad).astype(np.float32)
        vals["max_ws1_cos"] = np.cos(rad).astype(np.float32)
    info["position_changes"] = len(changes)
    return Network("ASOS", sids, periods, vals, info)


def load_nas(daily_path, station_path, cols, grid: DateGrid, overrides=()) -> Network:
    sid_col, dcol = NET_KEYS["NAS"]
    nas = pq.read_table(daily_path, columns=[sid_col, dcol] + cols)
    st = pq.read_table(station_path, columns=["obsr_spot_code", "instl_la", "instl_lo"]).to_pylist()
    xy_of = {}
    for r in st:
        if r["obsr_spot_code"] in xy_of:
            raise ValueError("NAS 관측지점 정보에 같은 지점이 두 행")
        xy_of[r["obsr_spot_code"]] = (r["instl_la"], r["instl_lo"])
    sids = [s for s in sorted(pc.unique(nas[sid_col]).to_pylist()) if s not in EXCLUDE_NAS]
    if not set(sids) <= set(xy_of):
        raise ValueError("NAS 좌표가 없는 관측소가 있음")
    xy = np.array([xy_of[s] for s in sids], dtype=np.float64)
    if np.isnan(xy).any():
        raise ValueError("NAS 좌표에 빈 값")
    vals, info = daily_table(nas, sid_col, dcol, cols, sids, "NAS", grid)
    info["raw_overrides"] = apply_overrides(vals, sids, grid, overrides, "NAS")
    return Network("NAS", sids, [(0, grid.n, xy)], vals, info)


def load_air(daily_path, station_path, cols, grid: DateGrid, overrides=()) -> Network:
    sid_col, dcol = NET_KEYS["AIR"]
    air = pq.read_table(daily_path, columns=[sid_col, dcol] + cols)
    st = pq.read_table(station_path, columns=["station_nm", "dm_x", "dm_y"]).to_pylist()
    xy_of = {}
    for r in st:
        if r["dm_x"] is None or r["dm_y"] is None or not (33 <= r["dm_x"] <= 39 and 124 <= r["dm_y"] <= 132):
            raise ValueError("에어코리아 좌표 범위가 예상(dm_x 위도·dm_y 경도)과 다름")
        xy_r = (r["dm_x"], r["dm_y"])
        if r["station_nm"] in xy_of and xy_of[r["station_nm"]] != xy_r:
            raise ValueError(f"같은 이름인데 좌표가 다른 측정소: {r['station_nm']}")
        xy_of[r["station_nm"]] = xy_r
    all_air = sorted(pc.unique(air[sid_col]).to_pylist())
    sids = [s for s in all_air if s in xy_of]
    vals, info = daily_table(air, sid_col, dcol, cols, sids, "AIR", grid)
    info["raw_overrides"] = apply_overrides(vals, sids, grid, overrides, "AIR")
    info["no_coord_stations"] = [s for s in all_air if s not in xy_of]
    return Network("AIR", sids, [(0, grid.n, np.array([xy_of[s] for s in sids], dtype=np.float64))], vals, info)


LOADERS = {"ASOS": load_asos, "NAS": load_nas, "AIR": load_air}


def load_networks(paths: dict, specs, grid: DateGrid, overrides=()) -> dict[str, Network]:
    """paths: 입력 칸 이름(asos_daily 등) → 실제 경로. specs 에 필요한 관측망·칼럼만 읽음.

    overrides: 원자료 칸 바꾸기 목록(담당자 결정 — 설정 raw_overrides), 칸마다 dict(input·station·date·column·value·reason).
    읽은 직후 3곳 평균 전에 적용. 읽지 않는 관측망을 가리키는 항목이 있으면 멈춤.
    """
    by_net: dict = {}
    key_net = {dk: net for net, (dk, _) in NET_FILE_KEYS.items()}
    for o in overrides:
        if o["input"] not in key_net:
            raise ValueError(f"raw_overrides input 은 {sorted(key_net)} 중 하나: {o['input']!r}")
        by_net.setdefault(key_net[o["input"]], []).append(o)
    need = needed_columns(specs)
    unused = sorted(set(by_net) - set(need))
    if unused:
        raise ValueError(f"raw_overrides 대상 관측망을 이번 계산에서 읽지 않음: {unused}")
    nets = {}
    for net, cols in need.items():
        dk, sk = NET_FILE_KEYS[net]
        nets[net] = LOADERS[net](paths[dk], paths[sk], cols, grid, by_net.get(net, ()))
    return nets


# ---------------------------------------------------------------- k곳 평균
def knn_mean(V, periods, loc_xy, n_dates, k=K_NEAREST, block=BLOCK):
    """V: (날짜 × 관측소). 반환 (좌표 × 날짜) float32: 평균값, 고른 관측소까지 평균 거리(km), 고른 관측소 수.

    날짜마다 그날 값 있는 관측소 중 거리순 k 곳 평균(k 곳이 안 되면 있는 곳만, 없으면 NaN).
    같은 거리면 관측소 순서(sids)가 앞인 곳(stable 정렬) — 03 knn_mean 그대로, 고른 수만 더 돌려줌.
    """
    have = ~np.isnan(V).all(axis=0)
    Vh = V[:, have]
    n_loc = len(loc_xy)
    M = np.full((n_loc, n_dates), np.nan, dtype=np.float32)
    Dm = np.full((n_loc, n_dates), np.nan, dtype=np.float32)
    C = np.zeros((n_loc, n_dates), dtype=np.int8)
    if Vh.shape[1] == 0:
        return M, Dm, C
    for a, b, st_xy in periods:
        xy = st_xy[have]
        dist = haversine_km(loc_xy[:, None, 0], loc_xy[:, None, 1], xy[None, :, 0], xy[None, :, 1])
        order = np.argsort(dist, axis=1, kind="stable")
        dsort = np.take_along_axis(dist, order, axis=1)
        for s0 in range(a, b, block):
            s1 = min(s0 + block, b)
            vals = Vh[s0:s1][:, order]
            ok = ~np.isnan(vals)
            sel = ok & (np.cumsum(ok, axis=2) <= k)
            cnt = sel.sum(axis=2)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(sel, vals, 0).sum(axis=2, dtype=np.float64) / cnt
                dmean = (sel * dsort[None]).sum(axis=2) / cnt
            M[:, s0:s1] = mean.T
            Dm[:, s0:s1] = dmean.T
            C[:, s0:s1] = cnt.T
    return M, Dm, C


def _src_values(nets, src):
    net, col = src
    return nets[net].values[col], nets[net].periods


def compute_var(nets, spec, loc_xy, n_dates, k=K_NEAREST):
    """변수 하나: 기본 칼럼 k곳 평균 → 결측 칸에만 대체 칼럼 k곳 평균(03 규칙).

    반환 dict: value·dist·count (좌표 × 날짜), fb_mask(대체가 들어간 칸).
    """
    V, per = _src_values(nets, spec["prim"])
    M, D, C = knn_mean(V, per, loc_xy, n_dates, k)
    use_fb = np.zeros_like(M, dtype=bool)
    if spec["fb"]:
        Vf, perf = _src_values(nets, spec["fb"])
        Mf, Df, Cf = knn_mean(Vf, perf, loc_xy, n_dates, k)
        use_fb = np.isnan(M) & ~np.isnan(Mf)
        M[use_fb] = Mf[use_fb]
        D[use_fb] = Df[use_fb]
        C[use_fb] = Cf[use_fb]
    return {"value": M, "dist": D, "count": C, "fb_mask": use_fb}


VALUE_RTOL = 1e-5  # 03 원자료 대조와 같은 허용(|차| ≤ 1e-5·max(1, |값|))


def compare_matrices(new: np.ndarray, old: np.ndarray, rows=None) -> dict:
    """같은 규칙으로 다시 계산한 값(new)과 기존 표 값(old) 대조(rows = 비교할 시군구 위치, None 이면 전부).

    반환: 결측 위치가 다른 칸 수, 값이 허용 밖으로 다른 칸 수, 최대 절대 차.
    """
    if rows is not None:
        new, old = new[rows], old[rows]
    nn, on = np.isnan(new), np.isnan(old)
    both = ~nn & ~on
    diff = np.abs(new[both].astype(np.float64) - old[both].astype(np.float64))
    tol = VALUE_RTOL * np.maximum(1.0, np.abs(old[both].astype(np.float64)))
    return {"cells": int(new.size), "nan_mismatch": int((nn != on).sum()), "value_mismatch": int((diff > tol).sum()),
            "max_abs_diff": float(diff.max()) if diff.size else 0.0}


def station_counts(nets, spec, start_idx=0, end_idx=None) -> dict:
    """점검용: 기본·대체 칼럼마다 (기간 안) 값이 한 번이라도 있는 관측소 수와 날마다 값 있는 관측소 수 최소·중앙·최대."""
    out = {}
    for role, src in (("기본", spec["prim"]), ("대체", spec["fb"])):
        if not src:
            continue
        V, _ = _src_values(nets, src)
        Vs = V[start_idx:end_idx]
        per_day = (~np.isnan(Vs)).sum(axis=1)
        out[role] = {"원천": f"{NET_TABLE[src[0]]}.{src[1]}", "값 있는 관측소 수": int((~np.isnan(Vs)).any(axis=0).sum()),
                     "날마다 값 있는 관측소 수 최소": int(per_day.min()) if per_day.size else None,
                     "날마다 값 있는 관측소 수 중앙값": float(np.median(per_day)) if per_day.size else None,
                     "날마다 값 있는 관측소 수 최대": int(per_day.max()) if per_day.size else None}
    return out


# ---------------------------------------------------------------- 옹진군 대표 좌표(부록 A-2)
COORD_TOL_DEG = 1e-4  # 시도청 점과 '같은 좌표' 범위 검색(02_시군구_배정.py 와 같은 1e-4) 뒤 float32 등호 재확인(부록 B-6)


def ongjin_coord(sample_sgg: pa.Table, region: pa.Table, sgg_code: str = "28720") -> tuple[float, float, dict]:
    """28720 에 배정된 시료 전체(모든 d) 좌표 중앙값(시도청 점·빈 좌표 제외), 위도·경도 각각 중앙값.

    구현 세부(계획서에 없음): '시도청 점' = korea_region_code level='sido' 행 전부의 lat·lon
      (02_시군구_배정.py 는 인천광역시청 한 점만 봄 — 28720 시료에서는 결과 같음을 반환 정보로 확인),
      같은 좌표 = 위도·경도 차 < 1e-4(double 범위 검색) 이고 float32 로도 같음(재확인). 범위 안인데 float32 가
      다르면 판단이 갈리므로 멈춤. 위도·경도 따로 중앙값(02 와 같음). 시료 = 시료_시군구.parquet(RR1 로 옮겨지는
      해외 행은 이 파일에 없음).
    """
    codes = np.array(sample_sgg["sgg_code"].to_pylist(), dtype=object)
    lat = sample_sgg["lat"].combine_chunks()
    lon = sample_sgg["lon"].combine_chunks()
    m = codes == sgg_code
    la = np.array(lat.to_pylist(), dtype=object)[m]
    lo = np.array(lon.to_pylist(), dtype=object)[m]
    null_xy = np.array([a is None or b is None for a, b in zip(la, lo)], dtype=bool)
    la64 = np.array([np.nan if v is None else v for v in la], dtype=np.float64)
    lo64 = np.array([np.nan if v is None else v for v in lo], dtype=np.float64)
    null_xy |= np.isnan(la64) | np.isnan(lo64)
    lv = np.array(region["level"].to_pylist(), dtype=object)
    sido = [(r["legal_dong_code"], r["lat"], r["lon"]) for r, l in zip(region.to_pylist(), lv) if l == "sido"]
    sido = [s for s in sido if s[1] is not None and s[2] is not None]
    is_sido = np.zeros(m.sum(), dtype=bool)
    hit = {}
    for code, sla, slo in sido:
        rng = (np.abs(la64 - sla) < COORD_TOL_DEG) & (np.abs(lo64 - slo) < COORD_TOL_DEG) & ~null_xy
        eq32 = (la64.astype(np.float32) == np.float32(sla)) & (lo64.astype(np.float32) == np.float32(slo))
        if (rng & ~eq32).any():
            raise ValueError(f"시도청 점 {code} 와 범위 안인데 float32 로는 다른 시료 좌표 {int((rng & ~eq32).sum())}행 — 판단 필요")
        if rng.any():
            hit[code] = int(rng.sum())
        is_sido |= rng
    use = ~null_xy & ~is_sido
    if not use.any():
        raise ValueError(f"{sgg_code}: 쓸 수 있는 시료 좌표 없음")
    out_lat, out_lon = float(np.median(la64[use])), float(np.median(lo64[use]))
    info = {"sgg_code": sgg_code, "rows_all": int(m.sum()), "excluded_null_xy": int(null_xy.sum()),
            "excluded_sido_point": int(is_sido.sum()), "excluded_by_sido_code": hit, "used": int(use.sum()),
            "distinct_coords_used": int(len(set(zip(la64[use].tolist(), lo64[use].tolist())))),
            "lat": out_lat, "lon": out_lon}
    return out_lat, out_lon, info


def new_sgg_coords(old: pa.Table, ong_lat: float, ong_lon: float, sgg_code: str = "28720",
                   source_label: str = "") -> pa.Table:
    """기존 대표 좌표 표(256행)에서 28720 만 새 좌표로 바꾼 새 표(같은 칼럼)."""
    d = old.to_pydict()
    if d["sgg_code"].count(sgg_code) != 1:
        raise ValueError(f"대표 좌표 표에 {sgg_code} 가 한 행이 아님")
    i = d["sgg_code"].index(sgg_code)
    d["lat"][i], d["lon"][i] = ong_lat, ong_lon
    if "source" in d:
        d["source"][i] = source_label
    return pa.table(d, schema=old.schema.remove_metadata())


def coords_in_order(coord_tbl: pa.Table, sgg_order) -> np.ndarray:
    """대표 좌표 표 → sgg_order 순서 (n, 2) float64. 빠지거나 겹치면 멈춤."""
    codes = coord_tbl["sgg_code"].to_pylist()
    if len(set(codes)) != len(codes):
        raise ValueError("대표 좌표 sgg_code 중복")
    pos = {c: i for i, c in enumerate(codes)}
    missing = [c for c in sgg_order if c not in pos]
    if missing or len(codes) != len(sgg_order):
        raise ValueError(f"대표 좌표 표와 시군구 목록이 다름: 없음 {missing[:5]}, 수 {len(codes)} vs {len(sgg_order)}")
    la, lo = coord_tbl["lat"].to_numpy(), coord_tbl["lon"].to_numpy()
    xy = np.array([(la[pos[c]], lo[pos[c]]) for c in sgg_order], dtype=np.float64)
    if np.isnan(xy).any():
        raise ValueError("대표 좌표에 빈 값")
    return xy
