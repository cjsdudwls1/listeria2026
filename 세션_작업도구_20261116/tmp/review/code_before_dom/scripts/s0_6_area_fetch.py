# -*- coding: utf-8 -*-
"""s0_6_area_fetch.py — 단계 0-6 축사(L1) 면적 통계 받기(부록 A-4 '시군구 면적(공개 통계)'·'나라 면적(공개 통계 한 출처)').

표준 라이브러리만 씀 — 로컬 `py -I`(인터넷 됨) 또는 서버 venv 로 돌림. 받은 원본은 손대지 않고 그대로 저장하고,
같은 폴더 `출처.json` 에 URL·받은 시각(UTC)·HTTP Last-Modified·md5·크기·기준 연도·설명을 적음. 이미 있는 파일은 덮지 않고 멈춤.

받는 것(configs/s0_6_static.json 의 "area_fetch"):
- 국내: 국토교통부 지적통계 '행정구역별·지목별 국토이용현황_시군구'(국토교통 통계누리 formId 2300) 기준 연도 1해 —
  화면이 부르는 JSON 두 개(칼럼 정의 columns.do · 값 data.do)를 그대로 저장. 값 단위 ㎡(칼럼 '계'·'면적').
- 해외: FAOSTAT Land Use 일괄 파일(Inputs_LandUse_E_All_Data_(Normalized).zip) — 'Land area'(item 6601, 1000 ha).

구현 세부(계획서에 없음 — 담당자 확인 필요): 출처 선택(국내 = 국토교통부 지적통계, 해외 = FAOSTAT — 계획서 예시 World Bank
국토 면적(AG.LND.TOTL.K2)은 대만(TWN)·쿡 제도(COK)가 없어 '한 출처로 94개 나라'를 못 채움. World Bank 값도 FAO 가 원천).

실행: py -I 리스테리아ML2/exp/final/scripts/s0_6_area_fetch.py --root <자료 뿌리> [--config <설정>]
  → <root>/<area_fetch.dir>/ 에 저장(설정 기본 = 리스테리아ML2/data/외부_수집/면적_20261107)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import paths as ph  # noqa: E402  (표준 라이브러리만 씀)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = CODE_ROOT / "configs" / "s0_6_static.json"
FETCH_KEYS = {"dir", "timeout_s", "user_agent", "items"}
ITEM_KEYS = {"name", "url", "file", "base_year", "description", "headers"}
PROVENANCE = "출처.json"


def load_fetch_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    fc = cfg.get("area_fetch")
    if not isinstance(fc, dict) or set(fc) != FETCH_KEYS:
        raise ValueError(f"area_fetch 칸이 다름: {sorted(fc) if isinstance(fc, dict) else fc!r} (필요 {sorted(FETCH_KEYS)})")
    for it in fc["items"]:
        if set(it) != ITEM_KEYS:
            raise ValueError(f"area_fetch.items 칸이 다름: {sorted(it)}")
        ph.to_rel(it["file"])
        if "/" in it["file"] or "\\" in it["file"]:
            raise ValueError(f"file 은 이름만: {it['file']}")
    ph.to_rel(fc["dir"])
    return fc


def fetch(url: str, headers: dict, ua: str, timeout: float) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": ua, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        info = {"status": r.status, "last_modified": r.headers.get("Last-Modified"),
                "content_type": r.headers.get("Content-Type"), "final_url": r.geturl()}
    return body, info


def main(argv=None) -> None:
    ph.unset_path_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    fc = load_fetch_cfg(Path(args.config))
    out = ph.resolve(root, fc["dir"])
    out.mkdir(parents=True, exist_ok=True)
    prov_p = out / PROVENANCE
    targets = [out / it["file"] for it in fc["items"]] + [prov_p]
    exist = [p.name for p in targets if p.exists()]
    if exist:
        raise FileExistsError(f"이미 있음 — 덮어쓰지 않음: {exist}")
    rec = {"fetched_by": "scripts/s0_6_area_fetch.py", "python": sys.version.split()[0], "items": []}
    for it in fc["items"]:
        url = it["url"]
        urllib.parse.urlsplit(url)  # 형식 확인
        body, info = fetch(url, it["headers"], fc["user_agent"], fc["timeout_s"])
        if info["status"] != 200 or not body:
            raise RuntimeError(f"받기 실패: {it['name']} {info}")
        p = out / it["file"]
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(p)
        rec["items"].append({
            "name": it["name"], "file": it["file"], "url": url, "base_year": it["base_year"],
            "description": it["description"], "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(body), "md5": hashlib.md5(body).hexdigest(), **info,
        })
        print(f"받음 {it['name']}: {len(body):,} bytes md5 {rec['items'][-1]['md5']}")
    with open(prov_p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"출처 기록: {prov_p}")


if __name__ == "__main__":
    main()
