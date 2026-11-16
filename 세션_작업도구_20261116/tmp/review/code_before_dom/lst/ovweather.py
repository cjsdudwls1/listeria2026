"""해외 기상(나라 단위) 공통 함수 — 계획서 §4-4, §4-5(강수만 준비), §6 단계 0-4, 부록 A-2.

s0_4_openmeteo_fetch.py(받기)·s0_4_weather_overseas.py(표 만들기)가 같이 씀. 네트워크 없음
(순수 함수 + 설정·입력 파일 읽기 도우미) — 시험은 합성 자료로.

계획서에서 옮긴 것(부록 A-2):
- 기간: DB = global_weather_info, DB 에 없는 기간(2014-04-07 ~ 12-31, DB 끝 뒤)·변수 = Open-Meteo Historical
  Weather API(ERA5·ERA5-Land), 강수만 2013-05-01 부터(SPI 누적용), 하루 경계 = 나라 현지 시간(`timezone`).
- DB 칼럼 대응: temperature_mean·max·min → avg·max·min_temp, precipitation_sum → rain, wind_speed_mean·max →
  avg·max_wind, sunshine_duration → sunshine, shortwave_radiation_sum → solar_rad, relative_humidity_mean → 습도,
  soil_moisture_mean → soil_moist(단위는 겹치는 날 Open-Meteo 값으로 확인해 맞춤).
- Open-Meteo: 기온 temperature_2m_mean/max/min, rain precipitation_sum, avg·max_wind wind_speed_10m_mean·_max ÷ 3.6,
  sunshine sunshine_duration ÷ 3600, solar_rad shortwave_radiation_sum, 습도 relative_humidity_2m_mean,
  soil_moist soil_moisture_0_to_7cm_mean × 100, soil_t5~t30 = ERA5-Land 0-7·7-28·28-100cm 층 일평균을
  3.5·17.5·64cm 로 선형 보간, 풍향 wind_direction_10m_dominant sin·cos, 이슬점 = 일 평균 이슬점(ERA5, 못 받으면
  기온·습도로 계산).
- 보정: 대상 = DB 에도 있는 변수(기온 3·강수·바람 2·일조·일사·습도·토양수분), 겹치는 기간 2015-01-01 ~ 2015-12-31,
  변수·나라별 평균 차이(강수는 합 비율, Open-Meteo 합이 0 이면 1)를 Open-Meteo 값에 적용, 보정 뒤 강수·일조·
  토양수분 0 미만은 0, 습도는 0~100. DB 에 없는 변수(지중온도·풍향·이슬점)는 Open-Meteo 그대로.
- 칼럼 설명 '근사': max_wind·sunshine·soil_t5~t30·soil_moist·풍향 sin·cos (rg·soil_ph 는 다른 표).
- §4-4: soil_t_nas·no2·co·so2 = NaN. 1차 = DB 값만, 2차 = Open-Meteo 로 채운 표, 칸마다 원천 표시.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 칼럼 이름
# 국내 표(시군구_일별_기상.parquet)와 같은 19변수 순서(§4-3)
VARS19 = ("avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind", "sunshine", "solar_rad",
          "soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas", "soil_moist", "no2", "co", "so2",
          "wind_dir_sin", "wind_dir_cos")
# 구현 세부(계획서에 없음): 습도·이슬점 칼럼 이름 = 계획서 부록 A-2 국내 원천 칼럼 이름(avg_rhm·avg_td) —
# 국내 습도·이슬점 새 파일 칼럼 이름과 맞춰야 함(담당자·국내 담당 확인).
HUM = "avg_rhm"
DEW = "avg_td"
TABLE_VARS = VARS19 + (HUM, DEW)
ALWAYS_NAN = ("soil_t_nas", "no2", "co", "so2")          # §4-4

# DB 칼럼 → 우리 변수(부록 A-2 DB 칼럼 대응)
DB_TO_VAR = {
    "temperature_mean": "avg_temp", "temperature_max": "max_temp", "temperature_min": "min_temp",
    "precipitation_sum": "rain", "wind_speed_mean": "avg_wind", "wind_speed_max": "max_wind",
    "sunshine_duration": "sunshine", "shortwave_radiation_sum": "solar_rad",
    "relative_humidity_mean": HUM, "soil_moisture_mean": "soil_moist",
}
CAL_VARS = tuple(DB_TO_VAR.values())                    # 보정 대상 10변수(DB 에도 있는 변수)
# Open-Meteo 일 변수 → 우리 변수(부록 A-2 Open-Meteo 줄)
OM_DB_VARS = {
    "temperature_2m_mean": "avg_temp", "temperature_2m_max": "max_temp", "temperature_2m_min": "min_temp",
    "precipitation_sum": "rain", "wind_speed_10m_mean": "avg_wind", "wind_speed_10m_max": "max_wind",
    "sunshine_duration": "sunshine", "shortwave_radiation_sum": "solar_rad",
    "relative_humidity_2m_mean": HUM, "soil_moisture_0_to_7cm_mean": "soil_moist",
}
OM_RAIN = "precipitation_sum"
OM_SOIL_T = ("soil_temperature_0_to_7cm_mean", "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean")
OM_WIND_DIR = "wind_direction_10m_dominant"
OM_DEW = "dew_point_2m_mean"
# 단위 변환(Open-Meteo 단위 → 국내 표 단위, 부록 A-2). DB 값도 단위 확인(겹치는 날) 뒤 같은 변환.
CONVERT = {"avg_wind": 1 / 3.6, "max_wind": 1 / 3.6, "sunshine": 1 / 3600, "soil_moist": 100.0}
UNITS = {"avg_temp": "°C", "max_temp": "°C", "min_temp": "°C", "rain": "mm", "avg_wind": "m/s", "max_wind": "m/s",
         "sunshine": "시간", "solar_rad": "MJ/m²", "soil_t5": "°C", "soil_t10": "°C", "soil_t20": "°C", "soil_t30": "°C",
         "soil_t_nas": "°C", "soil_moist": "%", "no2": "ppm", "co": "ppm", "so2": "ppm",
         "wind_dir_sin": "(−1~1)", "wind_dir_cos": "(−1~1)", HUM: "%", DEW: "°C"}
APPROX = ("max_wind", "sunshine", "soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_moist",
          "wind_dir_sin", "wind_dir_cos")               # 부록 A-2 '근사' 표기(이 표에 있는 칼럼만)
CLIP_ZERO = ("rain", "sunshine", "soil_moist")          # 보정 뒤 0 미만 → 0
CLIP_0_100 = (HUM,)                                     # 보정 뒤 0~100
SOIL_LAYER_MID_CM = (3.5, 17.5, 64.0)                  # 0-7·7-28·28-100cm 층 가운데
SOIL_TARGET_CM = {"soil_t5": 5.0, "soil_t10": 10.0, "soil_t20": 20.0, "soil_t30": 30.0}

# 원천 표시(칸마다)
SRC_DB, SRC_OM_CAL, SRC_OM, SRC_OM_CALC = "db", "om_cal", "om", "om_calc"

# 좌표 종류
COORD_DB = "db_latlon"            # global_weather_info lat·lon
COORD_COUNTRY = "country_latlon"  # global_weather_info country_lat·country_lon
COORD_POLY = "polygon"            # DB 에 없는 나라(부록 A-2 다각형 안 시료 좌표 중앙값 / 대표점)
DB_COORD_COLS = {COORD_DB: ("lat", "lon"), COORD_COUNTRY: ("country_lat", "country_lon")}

GROUPS = ("rain", "dbvars", "extra")

A3_RE = re.compile(r"^[A-Z]{3}$")


# ---------------------------------------------------------------- 날짜
def d(s) -> _dt.date:
    if isinstance(s, _dt.datetime):
        return s.date()
    if isinstance(s, _dt.date):
        return s
    return _dt.date.fromisoformat(str(s))


def date_range(start, end) -> pd.DatetimeIndex:
    return pd.date_range(d(start), d(end), freq="D")


def split_years(start, end) -> list[tuple[_dt.date, _dt.date]]:
    """[start, end] 를 달력 해 단위로 자름(빈 구간이면 [])."""
    s, e = d(start), d(end)
    out = []
    while s <= e:
        ye = min(e, _dt.date(s.year, 12, 31))
        out.append((s, ye))
        s = ye + _dt.timedelta(days=1)
    return out


# ---------------------------------------------------------------- 받기 계획
@dataclass(frozen=True)
class Chunk:
    a3: str
    coord: str            # 좌표 종류
    group: str            # rain·dbvars·extra
    start: _dt.date
    end: _dt.date

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def relpath(self) -> str:
        return f"raw/{self.a3}/{self.coord}/{self.group}_{self.start:%Y%m%d}_{self.end:%Y%m%d}.json"


def group_request(group: str, models: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """묶음 → (일 변수 목록, 모델 목록).

    구현 세부(계획서에 없음 — 설정 'models'): 계획서는 'ERA5·ERA5-Land' 만 적음.
    rain·dbvars = 설정 모델 하나(기본 era5_seamless = ERA5 + ERA5-Land — 11-07 시험 호출에서 DB 값과 같았음),
    extra(지중온도 3층·풍향·이슬점) = 설정 모델들을 한 호출에 (soil_temp·dew_point·wind_dir 모델의 합집합).
    """
    if group == "rain":
        return [OM_RAIN], [models["rain"]]
    if group == "dbvars":
        return list(OM_DB_VARS), [models["dbvars"]]
    if group == "extra":
        ms = sorted({models["soil_temp"], models["dew_point"], models["wind_dir"]})
        return list(OM_SOIL_T) + [OM_WIND_DIR, OM_DEW], ms
    raise ValueError(group)


def extra_model_of(om_var: str, models: Mapping[str, str]) -> str:
    if om_var in OM_SOIL_T:
        return models["soil_temp"]
    if om_var == OM_DEW:
        return models["dew_point"]
    if om_var == OM_WIND_DIR:
        return models["wind_dir"]
    raise ValueError(om_var)


def est_cost(n_vars: int, n_models: int, n_days: int, n_locations: int = 1) -> float:
    """Open-Meteo 호출 무게 추정: max(1, 변수×모델/10) × max(1, 날/14) × 지점 수.

    근거: 가격 안내 '10 변수 넘거나 2주 넘으면 여러 호출, 분수로 셈(2주·15 변수 = 1.5)' +
    11-07 시험 호출 응답 머리 X-RateLimit-Cost(15 변수 × 2 모델 × 3일 = 3.0, 10 변수 × 1 모델 × 2지점 × 5일 = 2.0).
    실제 무게는 응답 머리 값으로 기록(이 값은 미리 재는 데만 씀).
    """
    return max(1.0, n_vars * n_models / 10.0) * max(1.0, n_days / 14.0) * n_locations


def chunk_cost(c: Chunk, models: Mapping[str, str]) -> float:
    v, m = group_request(c.group, models)
    return est_cost(len(v), len(m), c.n_days)


@dataclass(frozen=True)
class Periods:
    rain_start: _dt.date     # 2013-05-01
    table_start: _dt.date    # 2014-04-07
    overlap_start: _dt.date  # 2015-01-01
    overlap_end: _dt.date    # 2015-12-31
    use_end: _dt.date        # 2026-06-30 (구현 세부: 우리가 쓰는 마지막 날)

    def check(self):
        if not (self.rain_start < self.table_start <= self.overlap_start <= self.overlap_end <= self.use_end):
            raise ValueError("기간 순서 이상")


def db_gap_ranges(p: Periods, db_start, db_end) -> list[tuple[_dt.date, _dt.date]]:
    """DB 에 있는 나라의 'DB 에 없는 기간'(표 시작 ~ DB 시작 전날, DB 끝 다음 날 ~ use_end)."""
    s, e = d(db_start), d(db_end)
    out = []
    if p.table_start < s:
        out.append((p.table_start, s - _dt.timedelta(days=1)))
    if e < p.use_end:
        out.append((e + _dt.timedelta(days=1), p.use_end))
    return out


def plan_chunks(a3: str, coord: str, p: Periods, db_range: tuple | None) -> list[Chunk]:
    """나라 하나의 받을 조각(달력 해 단위).

    db_range = (DB 첫날, DB 끝날) — None 이면 DB 에 없는 나라.
    - rain: rain_start ~ table_start − 1 (강수만, SPI 누적용)
    - dbvars: DB 나라 = DB 에 없는 기간 + 겹치는 기간(보정) / DB 에 없는 나라 = table_start ~ use_end
    - extra(DB 에 없는 변수): table_start ~ use_end (모든 날)
    """
    p.check()
    out: list[Chunk] = []
    for s, e in split_years(p.rain_start, p.table_start - _dt.timedelta(days=1)):
        out.append(Chunk(a3, coord, "rain", s, e))
    if db_range is None:
        ranges = [(p.table_start, p.use_end)]
    else:
        ds, de = d(db_range[0]), d(db_range[1])
        if not (ds <= p.overlap_start and p.overlap_end <= de):
            raise ValueError(f"{a3}: 겹치는 기간 {p.overlap_start}~{p.overlap_end} 이 DB 기간 {ds}~{de} 밖")
        ranges = db_gap_ranges(p, ds, de) + [(p.overlap_start, p.overlap_end)]
    for s0, e0 in ranges:
        for s, e in split_years(s0, e0):
            out.append(Chunk(a3, coord, "dbvars", s, e))
    for s, e in split_years(p.table_start, p.use_end):
        out.append(Chunk(a3, coord, "extra", s, e))
    return out


def overlap_chunks(a3: str, coord: str, p: Periods) -> list[Chunk]:
    return [Chunk(a3, coord, "dbvars", s, e) for s, e in split_years(p.overlap_start, p.overlap_end)]


# ---------------------------------------------------------------- 요청·응답
def fmt_coord(x: float) -> str:
    """좌표 문자열(구현 세부: repr — DB 값 자릿수 그대로)."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("좌표 결측")
    return repr(x)


def request_params(c: Chunk, lat: float, lon: float, models: Mapping[str, str], timezone: str) -> dict:
    v, m = group_request(c.group, models)
    return {"latitude": fmt_coord(lat), "longitude": fmt_coord(lon),
            "start_date": c.start.isoformat(), "end_date": c.end.isoformat(),
            "daily": ",".join(v), "models": ",".join(m), "timezone": timezone}


def response_key(var: str, model: str, models: Sequence[str]) -> str:
    """응답 칼럼 이름: 모델 하나면 변수 이름 그대로, 여럿이면 '변수_모델'(11-07 시험 호출로 확인)."""
    return var if len(models) == 1 else f"{var}_{model}"


class ResponseError(ValueError):
    pass


MAX_GRID_SHIFT_DEG = 0.5   # 응답 격자 좌표가 요청 좌표에서 이보다 멀면 이상(검수 지적, 11-07)


def lon_diff(a: float, b: float) -> float:
    """경도 차(±180 넘나듦 고려, 도)."""
    x = abs(float(a) - float(b)) % 360.0
    return min(x, 360.0 - x)


def parse_response(body: bytes | str, c: Chunk, models: Mapping[str, str],
                   req_lat: float | None = None, req_lon: float | None = None) -> tuple[pd.DataFrame, dict]:
    """응답 JSON → (날짜 색인 DataFrame, 칼럼 '변수@모델', 값 float(null → NaN)), 응답 머리 정보.

    ResponseError: 날짜 목록이 [start, end] 하루 간격과 정확히 같지 않음 / 요청한 칼럼 없음 /
    요청한 칼럼이 모두(모든 칼럼·모든 날) null / (req_lat·req_lon 을 주면) 응답 격자 위도·경도가 요청 좌표에서
    MAX_GRID_SHIFT_DEG(0.5°) 넘게 옮겨짐(또는 응답에 좌표 없음).
    구현 세부(계획서에 없음): 0.5° 기준·'모두 null' 기준은 검수 지적(11-07)으로 정함 — 칼럼 일부만 null 인 것
    (예: era5_land 풍향은 늘 null)은 정상.
    """
    try:
        j = json.loads(body)
    except (ValueError, TypeError) as e:
        raise ResponseError(f"JSON 아님: {e}") from None
    if not isinstance(j, dict) or j.get("error"):
        raise ResponseError(f"오류 응답: {str(j)[:300]}")
    daily = j.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise ResponseError("daily.time 없음")
    want = [x.strftime("%Y-%m-%d") for x in date_range(c.start, c.end)]
    if list(daily["time"]) != want:
        raise ResponseError(f"날짜 목록 다름: {daily['time'][:2]}… {len(daily['time'])}일 (기대 {len(want)}일)")
    vars_, ms = group_request(c.group, models)
    cols = {}
    for v in vars_:
        for m in ms:
            k = response_key(v, m, ms)
            if k not in daily:
                raise ResponseError(f"칼럼 없음: {k}")
            arr = daily[k]
            if len(arr) != len(want):
                raise ResponseError(f"칼럼 길이 다름: {k}")
            cols[f"{v}@{m}"] = np.array([np.nan if x is None else float(x) for x in arr], dtype=np.float64)
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(want), name="date"))
    if df.isna().all().all():
        raise ResponseError(f"요청한 칼럼 {df.shape[1]}개가 모든 날 null")
    head = {k: j.get(k) for k in ("latitude", "longitude", "elevation", "timezone", "timezone_abbreviation",
                                  "utc_offset_seconds")}
    if not head.get("timezone"):
        raise ResponseError("timezone 없음")
    if req_lat is not None or req_lon is not None:
        la, lo = head.get("latitude"), head.get("longitude")
        if not isinstance(la, (int, float)) or not isinstance(lo, (int, float)):
            raise ResponseError("응답에 격자 좌표 없음")
        dla, dlo = abs(float(la) - float(req_lat)), lon_diff(lo, req_lon)
        if dla > MAX_GRID_SHIFT_DEG or dlo > MAX_GRID_SHIFT_DEG:
            raise ResponseError(f"응답 격자 ({la}, {lo}) 가 요청 ({req_lat}, {req_lon}) 에서 "
                                f"{max(dla, dlo):.3f}° 옮겨짐(> {MAX_GRID_SHIFT_DEG}°)")
    return df, head


# ---------------------------------------------------------------- 호출 무게 제한
class RateBudget:
    """흘러가는 창(1분·1시간·1일)마다 무게 합 ≤ 상한(넘으면 기다림), 월(31일 창) 상한은 넘으면 멈춤.
    기록 = [(시각 초, 무게)].

    구현 세부(계획서에 없음): 무료판 상한(600/분·5,000/시간·10,000/일·300,000/월)보다 낮게 설정값으로 지킴, 창은
    고정 시각이 아니라 흘러가는 창(더 보수적 — 월은 31일), 다시 실행해도 지난 기록(calls.jsonl)을 읽어 이어서 셈.
    """
    WINDOWS = (("minute", 60.0), ("hour", 3600.0), ("day", 86400.0))
    MONTH_SPAN = 31 * 86400.0

    def __init__(self, limits: Mapping[str, float], history: Iterable[tuple[float, float]] = ()):
        self.limits = {k: float(limits[k]) for k, _ in self.WINDOWS}
        self.limits["month"] = float(limits["month"])
        self.hist: list[tuple[float, float]] = sorted((float(t), float(w)) for t, w in history)

    def month_used(self, now: float) -> float:
        return sum(w for t, w in self.hist if t > now - self.MONTH_SPAN)

    def over_month(self, now: float, cost: float) -> bool:
        """이 호출로 월 상한(31일 창)을 넘으면 True — 기다리지 않고 멈출 일."""
        return self.month_used(now) + cost > self.limits["month"]

    def wait_needed(self, now: float, cost: float) -> float:
        """지금 cost 를 쓰려면 기다릴 초(0 이면 바로)."""
        wait = 0.0
        for name, span in self.WINDOWS:
            lim = self.limits[name]
            if cost > lim:
                raise ValueError(f"한 호출 무게 {cost:.1f} 가 {name} 상한 {lim} 보다 큼")
            inwin = [(t, w) for t, w in self.hist if t > now - span]
            used = sum(w for _, w in inwin)
            if used + cost <= lim:
                continue
            # 가장 오래된 것부터 창 밖으로 나가야 할 만큼
            need = used + cost - lim
            freed = 0.0
            for t, w in inwin:
                freed += w
                if freed >= need - 1e-9:
                    wait = max(wait, t + span - now + 0.001)
                    break
        return wait

    def add(self, t: float, cost: float):
        self.hist.append((float(t), float(cost)))
        cut = t - self.MONTH_SPAN
        if len(self.hist) > 200000:
            self.hist = [(a, b) for a, b in self.hist if a > cut]


def classify_429(body: bytes | str) -> str:
    """429 응답 이유 → 'month'·'minute'·'hour'·'day'·'unknown'(월을 먼저 봄)."""
    s = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    s = s.lower()
    for k in ("month", "minute", "hour", "daily", "day"):
        if k in s:
            return "day" if k == "daily" else k
    return "unknown"


# ---------------------------------------------------------------- 좌표 대조(DB 좌표 두 벌)
def coord_errors(db: pd.DataFrame, om: pd.DataFrame) -> pd.DataFrame:
    """나라·변수별 평균 절대 차이(겹치는 날, 둘 다 값 있는 날, 원 단위 그대로).

    db: 칼럼 = 우리 변수(CAL_VARS), 색인 = 날짜 / om: 같은 모양(Open-Meteo, 원 단위).
    돌려줌: 변수 → (mae, n).
    """
    rows = []
    for v in CAL_VARS:
        a = db[v].astype(float)
        b = om[v].reindex(a.index).astype(float)
        m = a.notna() & b.notna()
        rows.append({"var": v, "mae": float((a[m] - b[m]).abs().mean()) if m.any() else np.nan, "n": int(m.sum())})
    return pd.DataFrame(rows).set_index("var")


def decide_coord(per_country: Mapping[str, Mapping[str, pd.DataFrame]]) -> dict:
    """좌표 두 벌 중 DB 값과 더 잘 맞는 쪽.

    구현 세부(계획서에 없음 — 담당자 확인): (나라, 변수) 마다 평균 절대 차이가 더 작은 좌표가 1승(같으면 무승부),
    승수가 많은 좌표를 모든 DB 나라에 씀. 승수가 같으면 결정 못 함(None) → 멈추고 담당자에게.
    per_country: a3 → {좌표 종류: coord_errors 결과}.
    """
    wins = {COORD_DB: 0, COORD_COUNTRY: 0}
    ties = 0
    by_var = {v: {COORD_DB: 0, COORD_COUNTRY: 0, "tie": 0} for v in CAL_VARS}
    detail = {}
    for a3, e in sorted(per_country.items()):
        da, ca = e[COORD_DB], e[COORD_COUNTRY]
        detail[a3] = {}
        for v in CAL_VARS:
            x, y = da.loc[v, "mae"], ca.loc[v, "mae"]
            detail[a3][v] = {COORD_DB: None if pd.isna(x) else round(float(x), 6),
                             COORD_COUNTRY: None if pd.isna(y) else round(float(y), 6)}
            if pd.isna(x) or pd.isna(y) or abs(x - y) <= 1e-12:
                ties += 1
                by_var[v]["tie"] += 1
            elif x < y:
                wins[COORD_DB] += 1
                by_var[v][COORD_DB] += 1
            else:
                wins[COORD_COUNTRY] += 1
                by_var[v][COORD_COUNTRY] += 1
    if wins[COORD_DB] == wins[COORD_COUNTRY]:
        winner = None
    else:
        winner = COORD_DB if wins[COORD_DB] > wins[COORD_COUNTRY] else COORD_COUNTRY
    return {"winner": winner, "wins": wins, "ties": ties, "by_var": by_var, "detail": detail,
            "rule": "(나라, 변수)마다 2015 겹치는 기간 평균 절대 차이(원 단위)가 작은 좌표 1승, 승수 많은 좌표"}


# ---------------------------------------------------------------- 변환·보정
def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo(또는 같은 단위로 확인된 DB) 값 → 국내 표 단위(부록 A-2)."""
    out = df.copy()
    for v, f in CONVERT.items():
        if v in out:
            out[v] = out[v] * f
    return out


def unit_ratio(db_vals: np.ndarray, om_vals: np.ndarray) -> float:
    """단위 확인용 비: Σ|DB| / Σ|OM| (둘 다 값 있는 칸). 같은 단위면 ≈ 1.

    구현 세부(계획서에 없음): '단위는 겹치는 날 Open-Meteo 값으로 확인해 맞춤' 의 확인 방법 =
    겹치는 기간 모든 DB 나라 칸을 모아 이 비가 [0.8, 1.25] 안이면 같은 단위로 보고 Open-Meteo 와 같은 변환,
    밖이면 멈춤(단위를 추측해 바꾸지 않음).
    """
    a, b = np.asarray(db_vals, float), np.asarray(om_vals, float)
    m = np.isfinite(a) & np.isfinite(b)
    sb = np.abs(b[m]).sum()
    return float(np.abs(a[m]).sum() / sb) if sb > 0 else float("nan")


UNIT_RATIO_OK = (0.8, 1.25)


def calibration(db: pd.DataFrame, om: pd.DataFrame) -> dict[str, dict]:
    """나라 하나의 보정값(겹치는 기간, 같은 단위 — 변환 뒤). 변수 → {kind, value, n, db_stat, om_stat}.

    평균 차이 = mean(DB) − mean(OM), 강수 = Σ DB / Σ OM(Σ OM = 0 이면 1).
    구현 세부(계획서에 없음): 평균·합은 DB·OM 둘 다 값 있는 날로만. 그런 날이 0 이면 보정값 없음(NaN) →
    그 변수의 보정 대상 칸은 NaN(값을 만들지 않음) 이고 보고에 셈.
    """
    out = {}
    for v in CAL_VARS:
        a = db[v].astype(float)
        b = om[v].reindex(a.index).astype(float)
        m = a.notna() & b.notna()
        n = int(m.sum())
        if v == "rain":
            sa, sb = float(a[m].sum()), float(b[m].sum())
            val = (sa / sb if sb != 0 else 1.0) if n else np.nan
            out[v] = {"kind": "ratio", "value": val, "n": n, "db_stat": sa if n else np.nan, "om_stat": sb if n else np.nan}
        else:
            ma = float(a[m].mean()) if n else np.nan
            mb = float(b[m].mean()) if n else np.nan
            out[v] = {"kind": "diff", "value": ma - mb if n else np.nan, "n": n, "db_stat": ma, "om_stat": mb}
    return out


def apply_calibration(om: pd.DataFrame, cal: Mapping[str, dict]) -> tuple[pd.DataFrame, dict]:
    """보정 적용 + 자르기. 돌려줌: (보정 값, 자른 칸 수 {변수: n})."""
    out = om.copy()
    clipped = {}
    for v in CAL_VARS:
        if v not in out:
            continue
        c = cal[v]
        x = out[v].astype(float)
        x = x * c["value"] if c["kind"] == "ratio" else x + c["value"]
        n = 0
        if v in CLIP_ZERO:
            lo = x < 0
            n += int(lo.sum())
            x = x.mask(lo, 0.0)
        if v in CLIP_0_100:
            lo, hi = x < 0, x > 100
            n += int(lo.sum() + hi.sum())
            x = x.mask(lo, 0.0).mask(hi, 100.0)
        out[v] = x
        clipped[v] = n
    return out, clipped


def interp_soil(l0, l1, l2) -> dict[str, np.ndarray]:
    """ERA5-Land 0-7·7-28·28-100cm 층 일평균 → soil_t5·t10·t20·t30(층 가운데 3.5·17.5·64cm 사이 선형 보간).

    두 이웃 층 중 하나라도 NaN 이면 NaN(채우지 않음). 5·10cm = 3.5~17.5 사이, 20·30cm = 17.5~64 사이.
    """
    l0, l1, l2 = (np.asarray(x, dtype=np.float64) for x in (l0, l1, l2))
    z0, z1, z2 = SOIL_LAYER_MID_CM
    out = {}
    for name, z in SOIL_TARGET_CM.items():
        if z0 <= z <= z1:
            out[name] = l0 + (z - z0) / (z1 - z0) * (l1 - l0)
        elif z1 < z <= z2:
            out[name] = l1 + (z - z1) / (z2 - z1) * (l2 - l1)
        else:  # pragma: no cover — 목표 깊이는 모두 범위 안
            raise ValueError(z)
    return out


def wind_dir_sincos(deg) -> tuple[np.ndarray, np.ndarray]:
    """풍향(도) → sin·cos(국내 03_시군구_기상.py 와 같은 np.sin(np.deg2rad(·)) 규칙, NaN 그대로)."""
    r = np.deg2rad(np.asarray(deg, dtype=np.float64))
    return np.sin(r), np.cos(r)


def dewpoint_magnus(t_c, rh_pct) -> np.ndarray:
    """이슬점 계산(이슬점을 못 받은 날만 — 부록 A-2 '못 받으면 기온·습도로 계산').

    구현 세부(계획서에 없음): Magnus 식, 상수 a = 17.625, b = 243.04 °C(Alduchov·Eskridge 1996),
    일 평균 기온·일 평균 습도로 계산. 습도 ≤ 0 이면 NaN.
    """
    t = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)
    a, b = 17.625, 243.04
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(np.where(rh > 0, rh, np.nan) / 100.0) + a * t / (b + t)
        return b * g / (a - g)


# ---------------------------------------------------------------- DB 에 없는 나라 좌표
def ne_code(props: Mapping) -> str | None:
    """Natural Earth 나라 코드 = ISO_A3_EH(없거나 '-99' 면 ADM0_A3) — 부록 A-2."""
    c = props.get("ISO_A3_EH")
    if c in (None, "", "-99"):
        c = props.get("ADM0_A3")
    return None if c in (None, "", "-99") else str(c)


def polygon_coords(geojson: Mapping, a3s: Iterable[str], lat, lon) -> dict[str, dict]:
    """DB 에 없는 나라 좌표(부록 A-2): 그 나라 다각형(같은 코드 여러 개는 합침) 안에 찍힌 시료 좌표 중앙값,
    없으면 다각형 대표점.

    lat·lon = 시료 파일 lat·lon 전체(행마다, float32 → float64, NaN 행 빼고).
    구현 세부(계획서에 없음): 중앙값 = 위도·경도 따로 중앙값(np.median), '안' = shapely contains(경계선 위 점 제외),
    대표점 = shapely representative_point(). 경도 ±180 을 넘나드는 나라도 따로 처리 안 함(그대로 중앙값).
    shapely 필요(서버 venv 에 없으면 설치 — DB 에 없는 나라가 있을 때만 씀).
    """
    import shapely
    from shapely.geometry import shape

    feats: dict[str, list] = {}
    for f in geojson["features"]:
        c = ne_code(f.get("properties") or {})
        if c is not None:
            feats.setdefault(c, []).append(shape(f["geometry"]))
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    out = {}
    for a3 in sorted(set(a3s)):
        if a3 not in feats:
            raise KeyError(f"나라 경계 파일에 없는 나라 코드: {a3}")
        g = shapely.union_all(feats[a3])
        x0, y0, x1, y1 = g.bounds
        bb = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        inside = np.zeros(lat.shape[0], dtype=bool)
        if bb.any():
            inside[bb] = shapely.contains_xy(g, lon[bb], lat[bb])
        n = int(inside.sum())
        if n:
            out[a3] = {"lat": float(np.median(lat[inside])), "lon": float(np.median(lon[inside])),
                       "how": "시료 좌표 중앙값", "n_samples": n, "n_polygons": len(feats[a3])}
        else:
            rp = g.representative_point()
            out[a3] = {"lat": float(rp.y), "lon": float(rp.x), "how": "다각형 대표점", "n_samples": 0,
                       "n_polygons": len(feats[a3])}
    return out


# ---------------------------------------------------------------- 칼럼 설명
def column_descriptions(models: Mapping[str, str]) -> dict[str, dict]:
    """칼럼마다 설명·단위·원천·'근사' 여부(부록 A-2 표기 목록)."""
    om_name = {v: k for k, v in OM_DB_VARS.items()}
    db_name = {v: k for k, v in DB_TO_VAR.items()}
    desc = {}
    for v in TABLE_VARS:
        if v in ALWAYS_NAN:
            src = "해외 값 없음 — 늘 NaN(§4-4)"
        elif v in CAL_VARS:
            src = (f"DB global_weather_info.{db_name[v]}(×변환) / DB 에 없는 기간 = Open-Meteo {om_name[v]}"
                   f"({models['dbvars']}, 2015 겹치는 기간 나라별 보정), DB 에 없는 나라 = Open-Meteo 그대로")
            if v == "rain":
                src += f" / 2013-05-01 ~ 표 시작 전날도 강수만(SPI 누적용)"
        elif v in SOIL_TARGET_CM:
            src = (f"Open-Meteo {models['soil_temp']} 0-7·7-28·28-100cm 층 일평균을 3.5·17.5·64cm 로 선형 보간한 "
                   f"{int(SOIL_TARGET_CM[v])}cm 값(보정 없음)")
        elif v in ("wind_dir_sin", "wind_dir_cos"):
            src = f"Open-Meteo {OM_WIND_DIR}({models['wind_dir']}) 의 {'sin' if v.endswith('sin') else 'cos'}(보정 없음)"
        elif v == DEW:
            src = f"Open-Meteo {OM_DEW}({models['dew_point']}, 보정 없음), 못 받은 날은 Magnus 식(기온·습도)"
        else:  # pragma: no cover
            src = ""
        desc[v] = {"unit": UNITS[v], "approx": v in APPROX,
                   "description": ("근사 — " if v in APPROX else "") + src}
    return desc


# ---------------------------------------------------------------- DB 값만(담당자 결정 11-07: Open-Meteo 수집 안 함)
DB_ONLY_NAN = tuple(v for v in TABLE_VARS if v not in CAL_VARS)   # 지중온도 4·soil_t_nas·풍향 sin·cos·no2·co·so2·이슬점


def column_descriptions_db_only() -> dict[str, dict]:
    """--db-only 표의 칼럼 설명('근사' 표기는 부록 A-2 목록 그대로)."""
    db_name = {v: k for k, v in DB_TO_VAR.items()}
    desc = {}
    for v in TABLE_VARS:
        if v in CAL_VARS:
            conv = f" × {CONVERT[v]:.6g}" if v in CONVERT else ""
            src = f"DB global_weather_info.{db_name[v]}{conv}(보정 없음)"
        else:
            src = "DB 에 없는 변수 — 늘 NaN(Open-Meteo 수집 안 함, 담당자 결정 11-07)"
        approx = v in APPROX and v in CAL_VARS   # 늘 NaN 인 칼럼엔 '근사' 안 붙임
        desc[v] = {"unit": UNITS[v], "approx": approx,
                   "description": ("근사 — " if approx else "") + src}
    return desc


def db_only_table(gwi: pd.DataFrame, targets: Sequence[str], start, end) -> pd.DataFrame:
    """DB 값만으로 나라 일자료표(긴 표): alpha_3·sgg·date·TABLE_VARS·src_<변수>.

    DB 칼럼 대응·단위 변환(부록 A-2 — DB 단위 = Open-Meteo 단위는 확인됨), 보정 없음, 날짜 = [start, end],
    DB 값이 있는 칸만 src = 'db', DB 에 없는 변수·DB 값 NaN 칸·DB 에 없는 나라는 NaN(src null).
    """
    dates = date_range(start, end)
    parts = []
    for a in targets:
        g = gwi[gwi["alpha_3"] == a].set_index("date").rename(columns=DB_TO_VAR)[list(CAL_VARS)].sort_index()
        g = convert_units(g).reindex(dates)
        df = pd.DataFrame({"alpha_3": a, "sgg": "OV" + a, "date": dates.date})
        for v in TABLE_VARS:
            df[v] = g[v].to_numpy(dtype=np.float64) if v in CAL_VARS else np.nan
        for v in TABLE_VARS:
            src = np.full(len(dates), None, dtype=object)
            if v in CAL_VARS:
                src[~np.isnan(df[v].to_numpy())] = SRC_DB
            df["src_" + v] = src
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- 설정·읽기 도우미(두 스크립트 공통)
def config_schema():
    from .config import Field

    return {
        "inputs": {"global_weather_info": Field(str), "samples": Field(str), "countries_geojson": Field(str)},
        "periods": {k: Field(str) for k in ("rain_start", "table_start", "overlap_start", "overlap_end", "use_end")},
        "openmeteo": {
            "endpoint": Field(str), "timezone": Field(str),
            "models": {k: Field(str) for k in ("rain", "dbvars", "soil_temp", "dew_point", "wind_dir")},
            "limits": {k: Field((int, float)) for k in ("minute", "hour", "day", "month")},
            "min_interval_s": Field((int, float)), "timeout_s": Field((int, float)), "max_tries": Field(int),
            "backoff_s": Field(list, item_types=(int, float)),
            "wait_429_s": {k: Field((int, float)) for k in ("minute", "hour", "day", "unknown")},
            "user_agent": Field(str),
        },
    }


FREE_LIMITS = {"minute": 600, "hour": 5000, "day": 10000, "month": 300000}   # Open-Meteo 무료판(비상업) 상한


def load_cfg(path) -> dict:
    from .config import ConfigError, load_config

    cfg = dict(load_config(path, config_schema()))
    lim = cfg["openmeteo"]["limits"]
    for k, v in lim.items():
        if not (0 < v <= FREE_LIMITS[k]):
            raise ConfigError(f"openmeteo.limits.{k} = {v} — 무료판 상한 {FREE_LIMITS[k]} 이하여야 함")
    if not cfg["openmeteo"]["endpoint"].startswith("https://archive-api.open-meteo.com/"):
        raise ConfigError("endpoint 는 Open-Meteo Historical Weather API(archive-api)만")
    periods_of(cfg).check()
    return cfg


def periods_of(cfg) -> Periods:
    p = cfg["periods"]
    return Periods(*(d(p[k]) for k in ("rain_start", "table_start", "overlap_start", "overlap_end", "use_end")))


def read_targets(path, col: str = "alpha_3") -> list[str]:
    """나라 목록 파일(csv·parquet, 칼럼 col) → 정렬한 고유 alpha-3. 형식이 다르면 멈춤."""
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        s = pq.read_table(p, columns=[col]).column(col).to_pandas()
    elif p.suffix.lower() == ".csv":
        s = pd.read_csv(p, dtype=str, encoding="utf-8", usecols=[col], keep_default_na=False)[col]
    else:
        raise ValueError(f"나라 목록은 .csv·.parquet: {p.name}")
    s = s.astype("string")
    if s.isna().any() or (s == "").any():
        raise ValueError(f"나라 목록 {col} 에 빈 값 {int((s.isna() | (s == '')).sum())}행")
    vals = sorted(set(s.tolist()))
    bad = [v for v in vals if not A3_RE.match(v)]
    if bad:
        raise ValueError(f"alpha-3 형식 아님: {bad[:10]}")
    if not vals:
        raise ValueError("나라 목록이 비어 있음")
    return vals


GWI_COLS = ["alpha_3", "date", *DB_TO_VAR, "lat", "lon", "country_lat", "country_lon"]


def read_gwi(path, a3s: Iterable[str] | None = None) -> pd.DataFrame:
    """global_weather_info 추출본(값 그대로). 칼럼이 없으면 멈춤."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    miss = [c for c in GWI_COLS if c not in names]
    if miss:
        raise KeyError(f"global_weather_info 칼럼 없음: {miss}")
    filters = [("alpha_3", "in", sorted(set(a3s)))] if a3s is not None else None
    t = pq.read_table(path, columns=GWI_COLS, filters=filters).to_pandas()
    t["date"] = pd.to_datetime(t["date"])
    return t


def db_info(gwi: pd.DataFrame) -> pd.DataFrame:
    """나라마다 DB 첫날·끝날·날 수·좌표 두 벌. 나라 안 좌표가 여러 개·날짜 중복·빠진 날이면 멈춤."""
    if gwi.duplicated(["alpha_3", "date"]).any():
        raise ValueError("global_weather_info (alpha_3, date) 중복")
    if gwi.empty:
        return pd.DataFrame(columns=["db_start", "db_end", "n", "lat", "lon", "country_lat", "country_lon"],
                            index=pd.Index([], name="alpha_3"))
    g = gwi.groupby("alpha_3", sort=True)
    info = g.agg(db_start=("date", "min"), db_end=("date", "max"), n=("date", "size"),
                 lat=("lat", "first"), lon=("lon", "first"),
                 country_lat=("country_lat", "first"), country_lon=("country_lon", "first"))
    nun = g[["lat", "lon", "country_lat", "country_lon"]].nunique(dropna=False)
    if (nun > 1).any().any():
        raise ValueError(f"나라 안 좌표가 여러 개: {nun[(nun > 1).any(axis=1)].index.tolist()[:10]}")
    span = (info["db_end"] - info["db_start"]).dt.days + 1
    if (span != info["n"]).any():
        raise ValueError(f"DB 날짜에 빠진 날: {info.index[span != info['n']].tolist()[:10]}")
    for c in ("lat", "lon", "country_lat", "country_lon"):
        if info[c].isna().any():
            raise ValueError(f"DB 좌표 결측: {c}")
    return info


def load_inputs(rp, cfg: dict, nations_rel: str, col: str = "alpha_3") -> dict:
    """대상 나라·DB 값·DB 나라 정보·DB 에 없는 나라 좌표(부록 A-2). rp = lst.paths.RunPaths."""
    import pyarrow.parquet as pq

    targets = read_targets(rp.data(nations_rel), col)
    gwi = read_gwi(rp.data(cfg["inputs"]["global_weather_info"]), targets)
    info = db_info(gwi)
    in_db = [a for a in targets if a in info.index]
    missing = [a for a in targets if a not in info.index]
    inputs = [nations_rel, cfg["inputs"]["global_weather_info"]]
    poly = {}
    if missing:
        s = pq.read_table(rp.data(cfg["inputs"]["samples"]), columns=["lat", "lon"]).to_pandas()
        geo = json.loads(rp.data(cfg["inputs"]["countries_geojson"]).read_text(encoding="utf-8"))
        poly = polygon_coords(geo, missing, s["lat"].to_numpy(np.float64), s["lon"].to_numpy(np.float64))
        inputs += [cfg["inputs"]["samples"], cfg["inputs"]["countries_geojson"]]
    return {"targets": targets, "gwi": gwi, "info": info, "in_db": in_db, "missing": missing, "poly": poly,
            "inputs": inputs}


def all_chunks(p: Periods, info: pd.DataFrame, in_db, missing, poly, db_coord: str):
    """(조각, 위도, 경도) 전부 — DB 나라 = db_coord 좌표, DB 에 없는 나라 = 다각형 좌표."""
    out = []
    la, lo = DB_COORD_COLS[db_coord]
    for a in in_db:
        r = info.loc[a]
        for c in plan_chunks(a, db_coord, p, (r["db_start"], r["db_end"])):
            out.append((c, float(r[la]), float(r[lo])))
    for a in missing:
        for c in plan_chunks(a, COORD_POLY, p, None):
            out.append((c, poly[a]["lat"], poly[a]["lon"]))
    return out


def coordcheck_chunks(p: Periods, info: pd.DataFrame, in_db):
    out = []
    for a in in_db:
        r = info.loc[a]
        for kind, (la, lo) in DB_COORD_COLS.items():
            for c in overlap_chunks(a, kind, p):
                out.append((c, float(r[la]), float(r[lo])))
    return out


def chunk_files(out_dir, c: Chunk):
    """조각 응답 파일·요청 기록 파일 경로."""
    from .paths import resolve

    pth = resolve(out_dir, c.relpath)
    return pth, pth.with_name(pth.name[:-5] + ".req.json")


def db_coord_choice(fetch_dir, override: str | None) -> tuple[str, str]:
    """DB 나라 좌표 종류: --coord(담당자 지정) 또는 coordcheck.json 이긴 좌표."""
    from pathlib import Path

    if override is not None:
        if override not in DB_COORD_COLS:
            raise ValueError(override)
        return override, "--coord(담당자 지정)"
    pth = Path(fetch_dir) / "coordcheck.json"
    if not pth.exists():
        raise FileNotFoundError("coordcheck.json 없음 — 먼저 --stage coordcheck")
    dec = json.loads(pth.read_text(encoding="utf-8"))
    if dec.get("winner") not in DB_COORD_COLS:
        raise ValueError("coordcheck 에서 좌표를 못 정함 — --coord 로 지정")
    return dec["winner"], "coordcheck.json"
