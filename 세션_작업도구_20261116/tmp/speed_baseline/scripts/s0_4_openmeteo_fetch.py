# -*- coding: utf-8 -*-
"""s0_4_openmeteo_fetch.py — 단계 0-4 해외 기상: Open-Meteo Historical Weather API(ERA5·ERA5-Land) 받기.

계획서: §4-4(DB 나라 값 먼저, DB 에 없는 기간·변수는 같은 좌표에서 Open-Meteo), §4-5(해외 SPI 용 강수 —
2013-05-01 부터, SPI 계산은 다음 작업), §6 단계 0-4, 부록 A-2(기간·DB 에 없는 나라·나라 경계·보정용 겹치는 기간·
Open-Meteo 변수).

대상 나라 = --nations 파일(칼럼 alpha_3)의 나라(우리 해외 시료의 나라 — 시료 표 담당 에이전트가 만듦).
그중 global_weather_info 에 있는 나라 = DB 에 없는 기간·변수만, 없는 나라 = 모든 기간·변수(부록 A-2 좌표).

받는 것(나라마다, 달력 해 단위 조각):
- rain  : precipitation_sum, 2013-05-01 ~ 2014-04-06 (SPI 누적용 — 강수만)
- dbvars: DB 에도 있는 10변수, DB 나라 = DB 에 없는 기간(2014-04-07 ~ DB 첫날 전날, DB 끝 다음 날 ~ 2026-06-30)
          + 보정용 겹치는 기간 2015-01-01 ~ 2015-12-31 / DB 에 없는 나라 = 2014-04-07 ~ 2026-06-30
- extra : DB 에 없는 변수(지중온도 3층·풍향·이슬점), 2014-04-07 ~ 2026-06-30(모든 날)
하루 경계 = 나라 현지 시간(timezone=auto — 좌표의 시간대).

단계(--stage):
- plan       : 받을 조각·예상 무게만 셈(네트워크 없음) → plan_<시각>.json
- coordcheck : DB 나라마다 2015 겹치는 기간 dbvars 를 좌표 두 벌(lat·lon / country_lat·country_lon)로 받아
               DB 값과 비교 → coordcheck.json(이긴 좌표). 이긴 좌표의 2015 조각은 fetch 에서 그대로 다시 씀.
- fetch      : coordcheck.json 의 좌표(DB 나라)·부록 A-2 좌표(DB 에 없는 나라)로 남은 조각 전부.

이어받기: 조각 파일(raw/<A3>/<좌표 종류>/<묶음>_<시작>_<끝>.json, 응답 원본 바이트 그대로)과 요청 기록
(<같은 이름>.req.json)이 있고 요청 값이 같고 응답이 온전하면 건너뜀. 요청 값이 다르면(좌표가 바뀌는 등) 멈춤(덮지 않음).
호출 기록: calls.jsonl(시도마다 한 줄 — 시각·무게·상태). 무게 제한은 이 기록을 이어서 셈(다시 실행해도).

구현 세부(계획서에 없음 — 담당자 확인 필요, 반환 보고에 적음):
- 모델: rain·dbvars = era5_seamless(ERA5 + ERA5-Land — 11-07 시험 호출에서 DB 값과 같음: 기온·습도·토양수분 = ERA5-Land,
  바람·일조·일사 = ERA5), 지중온도 = era5_land(계획서 'ERA5-Land 층'), 이슬점 = era5(계획서 'ERA5'), 풍향 = era5
  (era5_land 는 풍향 null). extra 는 era5·era5_land 를 한 호출로(무게 같음).
- 마지막 날 = 2026-06-30(d 최대, E ≤ d — 그 뒤 날은 안 씀). DB 끝(2026-09-30)이 그보다 늦어 'DB 끝 뒤' 조각은 없음.
- 좌표 두 벌 대조 = 2015 겹치는 기간 전체(보정에 어차피 받는 1년이라 이긴 쪽은 다시 안 받음), 규칙은 lst.ovweather.decide_coord.
- 속도: 설정 limits(분 400·시간 4,000·하루 8,000 — 무료판 600·5,000·10,000 의 2/3~4/5), 흘러가는 창, 호출 사이 1초 이상,
  429 는 이유(분·시간·하루)별로 기다린 뒤 다시(시도 수에 안 셈), 5xx·접속 오류는 max_tries 까지 늘어나는 간격으로 다시.
  월 상한: 설정 250,000(무료판 300,000)을 31일 흘러가는 창으로 셈 — 넘을 호출이면 기다리지 않고 멈춤, 429 이유가 월 상한이어도 멈춤.
- 응답 검사(검수 지적 11-07): 응답 격자 좌표가 요청 좌표에서 0.5° 넘게 옮겨졌거나 요청 칼럼이 모두 null 이면 저장하지 않고 멈춤.
- 쓰기 순서: 요청 기록 .part → 조각 .part → 조각 → 요청 기록. 끊겨 남은 .part·요청 기록 없는 조각은 '불완전/' 로 옮기고(지우지 않음) 다시 받음.

실행(서버, 검수 뒤 지휘자):
  cd ~/cyj_listeria_final/code
  ~/cyj_listeria_final/venv/bin/python scripts/s0_4_openmeteo_fetch.py --root ~/cyj_listeria_final/data \
      --out ~/cyj_listeria_final/results/s0/0-4_openmeteo --nations <자료 뿌리 아래 나라 목록 파일> --stage plan
  … --stage coordcheck   (결정 못 하면 종료 코드 2 — 담당자 판단 뒤 --coord 로 지정)
  … --stage fetch        (여러 날 걸림 — nohup, 끊기면 같은 명령으로 이어받기)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lst import resources as rs  # noqa: E402  (numpy 전에 스레드 환경변수)

rs.set_thread_env(rs.THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import config as cf  # noqa: E402
from lst import ovweather as ow  # noqa: E402
from lst import paths as ph  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CONFIG = "configs/s0_4_weather_overseas.json"


# ---------------------------------------------------------------- 인자
def build_parser():
    ap = ph.build_parser(__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="코드 폴더 아래 설정(상대 경로)")
    ap.add_argument("--nations", required=True, help="자료 뿌리 아래 나라 목록(csv·parquet) 상대 경로")
    ap.add_argument("--nations-col", default="alpha_3")
    ap.add_argument("--stage", required=True, choices=("plan", "coordcheck", "fetch"))
    ap.add_argument("--coord", choices=(ow.COORD_DB, ow.COORD_COUNTRY), default=None,
                    help="fetch: coordcheck 결과 대신 담당자가 정한 좌표(기록됨)")
    ap.add_argument("--max-cost", type=float, default=None, help="이번 실행에서 쓸 무게 상한(시험용, 없으면 끝까지)")
    return ap


# ---------------------------------------------------------------- HTTP
def http_get(url: str, timeout: float, user_agent: str):
    """(상태, 머리 dict, 본문 bytes). 접속 오류는 예외 그대로."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


class FetchStop(RuntimeError):
    pass


# 시험에서 바꿔 끼우는 자리(네트워크·시계·잠자기)
HOOKS = {"http": http_get, "clock": time.time, "sleep": time.sleep}


class Fetcher:
    def __init__(self, out: Path, cfg: dict, code_md5: str, http=http_get, clock=time.time, sleep=time.sleep,
                 max_cost: float | None = None, log=print):
        self.out, self.cfg, self.om = out, cfg, cfg["openmeteo"]
        self.code_md5 = code_md5
        self.http, self.clock, self.sleep, self.log = http, clock, sleep, log
        self.calls_path = out / "calls.jsonl"
        hist = []
        if self.calls_path.exists():
            with open(self.calls_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        hist.append((r["t"], r["cost"]))
        self.budget = ow.RateBudget(self.om["limits"], hist)
        self.last_t = max((t for t, _ in hist), default=0.0)
        self.spent = 0.0
        self.max_cost = max_cost
        self.n_fetched = self.n_skipped = 0

    # 기록
    def _log_call(self, rec: dict):
        with open(self.calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def paths(self, c: ow.Chunk) -> tuple[Path, Path]:
        return ow.chunk_files(self.out, c)

    def _quarantine(self, path: Path) -> Path:
        """끊겨 남은 불완전 파일을 지우지 않고 <out>/불완전/<같은 상대 경로>.<UTC 시각> 으로 옮김."""
        rel = path.relative_to(self.out)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dst = self.out / "불완전" / rel.parent / f"{rel.name}.{stamp}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dst)
        self.log(f"  불완전 파일 옮김: {rel.as_posix()} → {dst.relative_to(self.out).as_posix()}")
        return dst

    @staticmethod
    def _parts(c_paths):
        return [x.with_name(x.name + ".part") for x in c_paths]

    def have(self, c: ow.Chunk, params: dict) -> bool:
        """이미 받은 조각이면 True. 요청 값이 다르거나 응답이 깨졌으면 멈춤(덮지 않음).

        쓰기 순서(get): 요청 기록 .part → 조각 .part → 조각 이름 바꿈 → 요청 기록 이름 바꿈. 그 사이에 끊겨
        남은 것(.part, 요청 기록 없는 조각 파일)은 '불완전/' 로 옮기고 다시 받음(지우지 않음).
        요청 기록만 있고 조각 파일이 없으면(이 순서로는 안 생김) 멈춤.
        """
        p, q = self.paths(c)
        for x in self._parts((p, q)):
            if x.exists():
                self._quarantine(x)
        if p.exists() and not q.exists():
            self._quarantine(p)
        if not p.exists() and not q.exists():
            return False
        if not p.exists():
            raise FetchStop(f"요청 기록만 있고 조각 파일 없음 — 손으로 확인: {c.relpath}")
        req = json.loads(q.read_text(encoding="utf-8"))
        if req["params"] != params:
            raise FetchStop(f"이미 받은 조각의 요청 값이 지금과 다름(좌표·모델 등) — 덮지 않음: {c.relpath}\n"
                            f"  있던 값 {req['params']}\n  지금 값 {params}")
        body = p.read_bytes()
        if hashlib.md5(body).hexdigest() != req["body_md5"]:
            raise FetchStop(f"조각 파일 md5 가 요청 기록과 다름: {c.relpath}")
        try:
            ow.parse_response(body, c, self.om["models"], float(params["latitude"]), float(params["longitude"]))
        except ow.ResponseError as e:
            raise FetchStop(f"이미 받은 조각 응답 이상: {c.relpath}: {e}") from None
        return True

    def _wait_budget(self, cost: float):
        while True:
            now = self.clock()
            w = max(self.budget.wait_needed(now, cost), self.last_t + float(self.om["min_interval_s"]) - now)
            if w <= 0:
                return
            if w > 30:
                self.log(f"  무게 제한 — {w / 60:.1f}분 기다림")
            self.sleep(w)

    def get(self, c: ow.Chunk, lat: float, lon: float) -> str:
        """조각 하나 받기(있으면 건너뜀). 'skip'·'fetched'."""
        params = ow.request_params(c, lat, lon, self.om["models"], self.om["timezone"])
        if self.have(c, params):
            self.n_skipped += 1
            return "skip"
        est = ow.chunk_cost(c, self.om["models"])
        if self.max_cost is not None and self.spent + est > self.max_cost:
            raise FetchStop(f"--max-cost {self.max_cost} 도달(이번 실행 {self.spent:.1f}) — 이어받기로 다시")
        url = self.om["endpoint"] + "?" + urllib.parse.urlencode(params)
        backoff = list(self.om["backoff_s"])
        tries = 0
        while True:
            if self.budget.over_month(self.clock(), est):
                raise FetchStop(f"월 상한(31일 창) {self.budget.limits['month']:.0f} 도달 "
                                f"(쓴 무게 {self.budget.month_used(self.clock()):.1f}) — 멈춤, 담당자 확인")
            self._wait_budget(est)
            t0 = self.clock()
            status, headers, body, err = None, {}, b"", None
            try:
                status, headers, body = self.http(url, float(self.om["timeout_s"]), self.om["user_agent"])
            except Exception as e:  # 접속 오류·시간 초과
                err = f"{type(e).__name__}: {e}"
            self.last_t = self.clock()
            cost_h = headers.get("X-RateLimit-Cost") or headers.get("x-ratelimit-cost")
            cost = float(cost_h) if cost_h not in (None, "") else est
            self.budget.add(t0, cost)
            self.spent += cost
            rec = {"t": t0, "utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(timespec="seconds"),
                   "chunk": c.relpath, "status": status, "cost": cost, "cost_est": est, "error": err}
            if status == 200:
                try:
                    ow.parse_response(body, c, self.om["models"], lat, lon)
                except ow.ResponseError as e:   # 좌표 0.5° 넘게 옮겨짐·모두 null 등 — 저장 안 함
                    rec["error"] = f"응답 이상(저장 안 함): {e}"
                    rec["body"] = body[:300].decode("utf-8", "replace")
                    self._log_call(rec)
                    raise FetchStop(f"응답 이상 — 저장하지 않고 멈춤: {c.relpath}: {e}") from None
                self._log_call(rec)
                p, q = self.paths(c)
                p.parent.mkdir(parents=True, exist_ok=True)
                for x in (p, q):
                    if x.exists():
                        raise FetchStop(f"덮어쓰기 금지: {x}")
                req = {"params": params, "url": url, "fetched_utc": rec["utc"], "status": status, "cost": cost,
                       "cost_est": est, "tries": tries + 1, "body_md5": hashlib.md5(body).hexdigest(),
                       "headers": {k: v for k, v in headers.items() if k.lower().startswith(("x-ratelimit", "date"))},
                       "code_md5": self.code_md5}
                p_part, q_part = self._parts((p, q))
                with open(q_part, "w", encoding="utf-8") as f:      # 1) 요청 기록 .part
                    json.dump(req, f, ensure_ascii=False, indent=1, allow_nan=False)
                p_part.write_bytes(body)                            # 2) 조각 .part
                os.replace(p_part, p)                               # 3) 조각
                os.replace(q_part, q)                               # 4) 요청 기록
                self.n_fetched += 1
                return "fetched"
            snippet = body[:300].decode("utf-8", "replace") if body else ""
            rec["body"] = snippet
            self._log_call(rec)
            if status == 429:
                kind = ow.classify_429(body)
                if kind == "month":
                    raise FetchStop(f"429 월 상한 — 재시도 안 하고 멈춤: {snippet[:200]}")
                w = float(self.om["wait_429_s"].get(kind, self.om["wait_429_s"]["unknown"]))
                self.log(f"  429({kind}) — {w:.0f}초 기다린 뒤 다시: {snippet[:120]}")
                self.sleep(w)
                continue
            if status is not None and 400 <= status < 500:
                raise FetchStop(f"요청 오류 {status} — 멈춤(설정·변수 확인): {c.relpath}: {snippet}")
            tries += 1
            if tries >= int(self.om["max_tries"]):
                raise FetchStop(f"{tries}번 실패 — 멈춤(이어받기로 다시): {c.relpath}: {status} {err} {snippet[:120]}")
            w = float(backoff[min(tries - 1, len(backoff) - 1)])
            self.log(f"  실패({status} {err}) — {w:.0f}초 뒤 다시({tries}/{self.om['max_tries']})")
            self.sleep(w)


# ---------------------------------------------------------------- 요약
def summarize(chunks, models) -> dict:
    tot = {}
    for c, _, _ in chunks:
        k = c.group
        t = tot.setdefault(k, {"chunks": 0, "cost_est": 0.0})
        t["chunks"] += 1
        t["cost_est"] += ow.chunk_cost(c, models)
    all_cost = sum(v["cost_est"] for v in tot.values())
    return {"by_group": tot, "chunks": len(chunks), "cost_est": round(all_cost, 1)}


def eta_days(cost: float, limits: dict) -> float:
    return cost / float(limits["day"])


# ---------------------------------------------------------------- 좌표 대조
def om_dbvars_frame(fetcher: Fetcher, chunks) -> pd.DataFrame:
    """dbvars 조각들 → 날짜 × 우리 변수(원 단위)."""
    model = fetcher.om["models"]["dbvars"]
    parts = []
    for c in chunks:
        p, _ = fetcher.paths(c)
        df, _h = ow.parse_response(p.read_bytes(), c, fetcher.om["models"])
        parts.append(df.rename(columns={f"{k}@{model}": v for k, v in ow.OM_DB_VARS.items()}))
    return pd.concat(parts).sort_index()


def run_coordcheck(fetcher: Fetcher, p: ow.Periods, gwi: pd.DataFrame, info: pd.DataFrame, in_db) -> dict:
    per = {}
    grid = {}
    for a in in_db:
        dba = gwi[gwi["alpha_3"] == a].set_index("date").rename(columns=ow.DB_TO_VAR)
        dba = dba.loc[(dba.index >= pd.Timestamp(p.overlap_start)) & (dba.index <= pd.Timestamp(p.overlap_end))]
        per[a] = {}
        grid[a] = {}
        for kind in ow.DB_COORD_COLS:
            ch = ow.overlap_chunks(a, kind, p)
            om = om_dbvars_frame(fetcher, ch)
            per[a][kind] = ow.coord_errors(dba[list(ow.CAL_VARS)], om)
            pth, _ = fetcher.paths(ch[0])
            j = json.loads(pth.read_bytes())
            grid[a][kind] = [j.get("latitude"), j.get("longitude")]
    dec = ow.decide_coord(per)
    dec["same_grid_countries"] = sorted(a for a, g in grid.items() if g[ow.COORD_DB] == g[ow.COORD_COUNTRY])
    dec["grid"] = grid
    dec["countries"] = list(in_db)
    dec["period"] = [p.overlap_start.isoformat(), p.overlap_end.isoformat()]
    dec["model"] = fetcher.om["models"]["dbvars"]
    return dec


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = build_parser()
    rp, args = ph.parse_paths(argv, ap)
    cfg_path = ph.resolve(CODE_ROOT, args.config)
    cfg = ow.load_cfg(cfg_path)
    p = ow.periods_of(cfg)
    om = cfg["openmeteo"]
    code = cf.code_md5(CODE_ROOT)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    li = ow.load_inputs(rp, cfg, args.nations, args.nations_col)
    targets, gwi, info, in_db, missing, poly, inputs = (li[k] for k in (
        "targets", "gwi", "info", "in_db", "missing", "poly", "inputs"))
    print(f"대상 나라 {len(targets)} (DB 에 있음 {len(in_db)}, DB 에 없음 {len(missing)}: {missing})")
    late = [a for a in in_db if info.loc[a, "db_end"].date() < p.use_end]
    if late:
        print(f"  DB 끝이 {p.use_end} 보다 이른 나라 {late} — 그 뒤 날도 받음")

    fetcher = Fetcher(rp.out, cfg, code, http=HOOKS["http"], clock=HOOKS["clock"], sleep=HOOKS["sleep"],
                      max_cost=args.max_cost)
    extra_meta = {"stage": args.stage, "targets": targets, "db_missing": missing, "polygon_coords": poly}

    if args.stage == "plan":
        cc = ow.coordcheck_chunks(p, info, in_db)
        rest = ow.all_chunks(p, info, in_db, missing, poly, ow.COORD_DB)   # 좌표 종류는 무게와 무관
        # coordcheck 이긴 쪽 2015 조각은 fetch 에서 다시 안 받음
        overlap_keys = {(c.a3, c.group, c.start) for c, _, _ in cc}
        rest_new = [x for x in rest if (x[0].a3, x[0].group, x[0].start) not in overlap_keys]
        s_cc, s_rest = summarize(cc, om["models"]), summarize(rest_new, om["models"])
        total = s_cc["cost_est"] + s_rest["cost_est"]
        plan = {"coordcheck": s_cc, "fetch_after_coordcheck": s_rest, "cost_est_total": round(total, 1),
                "days_at_day_limit": round(eta_days(total, om["limits"]), 2), "limits": om["limits"],
                "periods": cfg["periods"], "models": om["models"]}
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        cf.dump_json(rp.result(f"plan_{stamp}.json"), {**plan, **extra_meta})
        cf.save_run_meta(rp.result(f"run_meta_plan_{stamp}.json"), cfg, cf.inputs_md5(rp.root, inputs), CODE_ROOT,
                         extra={"run": extra_meta})
        return 0

    rs.check_resources(rp.out)
    if args.stage == "coordcheck":
        out_js = rp.result("coordcheck.json")
        if out_js.exists():
            raise FetchStop(f"이미 있음 — 덮지 않음: {out_js}")
        cc = ow.coordcheck_chunks(p, info, in_db)
        print(f"좌표 대조 조각 {len(cc)}, 예상 무게 {summarize(cc, om['models'])['cost_est']}")
        for i, (c, la, lo) in enumerate(cc, 1):
            st = fetcher.get(c, la, lo)
            if st == "fetched" and i % 20 == 0:
                print(f"  {i}/{len(cc)} 받음(이번 무게 {fetcher.spent:.1f})")
        dec = run_coordcheck(fetcher, p, gwi, info, in_db)
        cf.dump_json(out_js, {**dec, "code_md5": code})
        print(f"좌표 대조: 승 {dec['wins']}, 무 {dec['ties']}, 이긴 좌표 = {dec['winner']}, "
              f"같은 격자 나라 {len(dec['same_grid_countries'])}")
        cf.save_run_meta(rp.result(f"run_meta_coordcheck_{stamp}.json"), cfg, cf.inputs_md5(rp.root, inputs),
                         CODE_ROOT, extra={"run": {**extra_meta, "fetched": fetcher.n_fetched,
                                                   "skipped": fetcher.n_skipped, "cost": fetcher.spent}})
        if dec["winner"] is None:
            print("좌표를 정하지 못함(승수 같음) — 담당자 판단 뒤 --stage fetch --coord <종류>")
            return 2
        return 0

    # fetch
    if in_db:
        try:
            db_coord, how = ow.db_coord_choice(rp.out, args.coord)
        except (FileNotFoundError, ValueError) as e:
            raise FetchStop(str(e)) from None
        print(f"DB 나라 좌표 = {db_coord} ({how})")
        extra_meta["db_coord"], extra_meta["db_coord_how"] = db_coord, how
    else:
        db_coord = ow.COORD_DB
    chunks = ow.all_chunks(p, info, in_db, missing, poly, db_coord)
    s = summarize(chunks, om["models"])
    print(f"조각 {s['chunks']}, 예상 무게(이미 받은 것 포함) {s['cost_est']}, 하루 상한 기준 {eta_days(s['cost_est'], om['limits']):.1f}일")
    status = 0
    try:
        for i, (c, la, lo) in enumerate(chunks, 1):
            if i % 50 == 0:
                rs.check_resources(rp.out)
            st = fetcher.get(c, la, lo)
            if st == "fetched" and fetcher.n_fetched % 25 == 0:
                print(f"  {i}/{len(chunks)} (받음 {fetcher.n_fetched}, 건너뜀 {fetcher.n_skipped}, 이번 무게 {fetcher.spent:.1f})")
    except FetchStop as e:
        print(f"멈춤: {e}")
        status = 3
    print(f"받음 {fetcher.n_fetched}, 건너뜀 {fetcher.n_skipped}, 이번 무게 {fetcher.spent:.1f}")
    cf.save_run_meta(rp.result(f"run_meta_fetch_{stamp}.json"), cfg, cf.inputs_md5(rp.root, inputs), CODE_ROOT,
                     extra={"run": {**extra_meta, "fetched": fetcher.n_fetched, "skipped": fetcher.n_skipped,
                                    "cost": fetcher.spent, "status": status}})
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchStop as e:
        print(f"멈춤: {e}")
        sys.exit(3)
