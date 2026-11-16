"""좌표 → 나라(나라 경계 파일), 거리 — 계획서 부록 A-2, §2-2 RR1·RR4·RR5, 부록 B-6.

- 나라 경계 파일 = Natural Earth v5.1.2 1:10m admin-0 geojson(부록 A-2).
  나라 코드 = `ISO_A3_EH`(없거나 '-99' 면 `ADM0_A3`), 같은 코드 다각형 여러 개는 합쳐 한 나라.
- 좌표는 시료 파일 `lat`·`lon`(float32)을 float64 로 바꿔 씀(부록 B-6). 등호 비교 없음.

구현 세부(계획서에 없음):
- 다각형 안 판정 = even-odd 광선법(나라의 모든 고리 — 바깥·구멍 — 를 함께 셈). 경계선 위 점은
  어느 쪽으로든 갈 수 있음. 바깥 의존성(shapely 등) 없이 numpy 만 씀(서버 venv 목록에 shapely 없음).
- 어느 다각형에도 안 드는 점(해안·섬 — 10m 해상도 한계) = 가장 가까운 나라 다각형의 나라
  (`method='nearest'`, 거리 km 를 함께 돌려줌). 거리 = 점 위도에서의 등장방형 근사
  (경도 × cos(위도)) 평면 거리 × 111.195 km/도. 두 나라 이상 안에 드는 점 = `method='ambiguous'`, 나라 없음.
- 대권 거리(haversine) = 지구 반지름 6371.0088 km.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG = 111.195
_CHUNK = 4_000_000  # (점 × 변) 한 번에 계산할 최대 칸 수(메모리)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class Country:
    code: str
    rings: list = field(default_factory=list)   # 고리마다 (k, 2) float64 [lon, lat], 닫힘(첫 점 = 끝 점)
    bbox: tuple = (np.inf, np.inf, -np.inf, -np.inf)  # lon_min, lat_min, lon_max, lat_max

    def add_ring(self, coords) -> None:
        r = np.asarray(coords, dtype=np.float64)[:, :2]
        if r.shape[0] < 3:
            return
        if not np.array_equal(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        self.rings.append(r)
        b = self.bbox
        self.bbox = (min(b[0], r[:, 0].min()), min(b[1], r[:, 1].min()),
                     max(b[2], r[:, 0].max()), max(b[3], r[:, 1].max()))

    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x1 = np.concatenate([r[:-1, 0] for r in self.rings])
        y1 = np.concatenate([r[:-1, 1] for r in self.rings])
        x2 = np.concatenate([r[1:, 0] for r in self.rings])
        y2 = np.concatenate([r[1:, 1] for r in self.rings])
        return x1, y1, x2, y2


def country_code_of(props: dict) -> str:
    """부록 A-2: `ISO_A3_EH`, 없거나 '-99' 면 `ADM0_A3`."""
    c = props.get("ISO_A3_EH")
    if c is None or str(c).strip() in ("", "-99"):
        c = props.get("ADM0_A3")
    if c is None or str(c).strip() in ("", "-99"):
        raise ValueError(f"나라 코드 없음: {props.get('ADMIN')!r}")
    return str(c)


class CountryPolygons:
    def __init__(self, countries: dict[str, Country]):
        if not countries:
            raise ValueError("나라 다각형 없음")
        self.countries = countries
        self._edges = {c: v.edges() for c, v in countries.items()}

    @classmethod
    def from_geojson(cls, path: str | os.PathLike) -> "CountryPolygons":
        with open(path, "r", encoding="utf-8") as f:
            g = json.load(f)
        return cls.from_features(g["features"])

    @classmethod
    def from_features(cls, features) -> "CountryPolygons":
        out: dict[str, Country] = {}
        for ft in features:
            code = country_code_of(ft["properties"])
            geom = ft["geometry"]
            if geom is None:
                continue
            c = out.setdefault(code, Country(code))
            if geom["type"] == "Polygon":
                polys = [geom["coordinates"]]
            elif geom["type"] == "MultiPolygon":
                polys = geom["coordinates"]
            else:
                raise ValueError(f"모르는 도형: {geom['type']}")
            for poly in polys:
                for ring in poly:
                    c.add_ring(ring)
        return cls({k: v for k, v in out.items() if v.rings})

    # ------------------------------------------------------------ 안 판정
    def _inside_one(self, code: str, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self._edges[code]
        res = np.zeros(px.shape[0], dtype=bool)
        step = max(1, _CHUNK // max(1, x1.shape[0]))
        for s in range(0, px.shape[0], step):
            X = px[s:s + step, None]
            Y = py[s:s + step, None]
            cond = (y1[None, :] > Y) != (y2[None, :] > Y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1[None, :] + (Y - y1[None, :]) * (x2 - x1)[None, :] / (y2 - y1)[None, :]
            cross = cond & (X < xint)
            res[s:s + step] = (cross.sum(axis=1) % 2) == 1
        return res

    def inside_codes(self, lat, lon) -> list[list[str]]:
        """점마다 그 점을 안에 품는 나라 코드 목록."""
        py = np.asarray(lat, dtype=np.float64)
        px = np.asarray(lon, dtype=np.float64)
        hits: list[list[str]] = [[] for _ in range(px.shape[0])]
        ok = np.isfinite(px) & np.isfinite(py)
        for code, c in self.countries.items():
            b = c.bbox
            cand = np.flatnonzero(ok & (px >= b[0]) & (px <= b[2]) & (py >= b[1]) & (py <= b[3]))
            if cand.size == 0:
                continue
            ins = self._inside_one(code, px[cand], py[cand])
            for i in cand[ins]:
                hits[i].append(code)
        return hits

    # ------------------------------------------------------------ 가장 가까운 나라
    def nearest(self, lat: float, lon: float) -> tuple[str, float]:
        """(나라 코드, 가장 가까운 변까지 근사 거리 km)."""
        k = np.cos(np.radians(lat))
        best_code, best = None, np.inf
        for code in sorted(self.countries):
            x1, y1, x2, y2 = self._edges[code]
            ax, ay = (x1 - lon) * k, y1 - lat
            bx, by = (x2 - lon) * k, y2 - lat
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(L > 0, -(ax * dx + ay * dy) / L, 0.0)
            t = np.clip(t, 0.0, 1.0)
            d2 = (ax + t * dx) ** 2 + (ay + t * dy) ** 2
            m = float(np.sqrt(d2.min())) * KM_PER_DEG
            if m < best:  # 같으면 코드 순 앞쪽(정렬 순으로 돌기 때문)
                best, best_code = m, code
        return best_code, best

    def assign(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """점마다 (나라 코드 또는 None, method, 거리 km).

        method: 'inside'(다각형 안, 거리 0) / 'nearest'(어느 다각형에도 안 듦 → 가장 가까운 나라)
        / 'ambiguous'(둘 이상 안에 듦 — 나라 없음) / 'no_coord'(좌표 결측).
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        n = lat.shape[0]
        code = np.full(n, None, dtype=object)
        method = np.full(n, "no_coord", dtype=object)
        dist = np.full(n, np.nan)
        hits = self.inside_codes(lat, lon)
        for i in range(n):
            if not (np.isfinite(lat[i]) and np.isfinite(lon[i])):
                continue
            h = hits[i]
            if len(h) == 1:
                code[i], method[i], dist[i] = h[0], "inside", 0.0
            elif len(h) > 1:
                method[i] = "ambiguous"
            else:
                c, dkm = self.nearest(float(lat[i]), float(lon[i]))
                code[i], method[i], dist[i] = c, "nearest", dkm
        return code, method, dist
