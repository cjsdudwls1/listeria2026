from pathlib import Path

BASE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
p = BASE / "scripts" / "s0_4_weather_overseas.py"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep('''    ap.add_argument("--nations", required=True)
    ap.add_argument("--nations-col", default="alpha_3")
    ap.add_argument("--fetch-dir", required=True, help="s0_4_openmeteo_fetch.py 의 --out 폴더")''',
    '''    ap.add_argument("--nations", default=None, help="나라 목록(자료 뿌리 아래). --db-only 에선 없으면 DB 나라 전부")
    ap.add_argument("--nations-col", default="alpha_3")
    ap.add_argument("--fetch-dir", default=None, help="s0_4_openmeteo_fetch.py 의 --out 폴더(--db-only 아니면 필수)")
    ap.add_argument("--db-only", action="store_true",
                    help="Open-Meteo 없이 DB global_weather_info 값만(담당자 결정 11-07) — 1차 = 2차 같은 표")''')

rep('''def main(argv=None):
    ap = build_parser()
    rp, args = ph.parse_paths(argv, ap)
    fetch_dir''', '''def main_db_only(rp, args) -> int:
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
    fetch_dir''')

rep('''만드는 것(--out, 기존 파일 안 덮음):''', '''--db-only(담당자 결정 11-07 — 해외 Open-Meteo 수집 안 함): 받은 조각 없이 DB 값만으로 표 하나를 만들어 1차·2차
파일에 같은 내용(메타 same_as_1st), 날짜 2015-01-01 ~ 2026-06-30, DB 에 없는 변수 NaN, 보정 없음. --nations 없으면 DB 나라 전부.
  실행: <python> scripts/s0_4_weather_overseas.py --root <자료 뿌리> --out <결과> --db-only [--nations <파일>]

만드는 것(--out, 기존 파일 안 덮음):''')
p.write_text(s, encoding="utf-8")

t = BASE / "tests" / "test_ovweather.py"
x = t.read_text(encoding="utf-8")
x += '''

def test_db_only(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    bu = _load("s0_4_weather_overseas")
    monkeypatch.setattr(bu.rs, "check_resources", lambda path: {})
    root = make_root(tmp_path)
    out = tmp_path / "dbonly"
    with pytest.raises(bu.BuildStop):
        bu.main(["--root", str(root), "--out", str(out), "--db-only", "--fetch-dir", str(tmp_path)])
    assert bu.main(["--root", str(root), "--out", str(out), "--db-only"]) == 0       # 나라 = DB 전부(AAA·BBB)
    t1 = pd.read_parquet(out / bu.OUT_1)
    t2 = pd.read_parquet(out / bu.OUT_2)
    pd.testing.assert_frame_equal(t1, t2)
    days = ow.date_range("2015-01-01", P.use_end)
    assert sorted(t2.alpha_3.unique()) == ["AAA", "BBB"] and len(t2) == 2 * len(days)
    t2["date"] = pd.to_datetime(t2["date"])
    a = t2[t2.alpha_3 == "AAA"].set_index("date")
    x = pd.Timestamp("2016-03-01")
    assert a.loc[x, "avg_temp"] == pytest.approx(om_value("temperature_2m_mean", x, 10.0) - 0.5)
    assert a.loc[x, "avg_wind"] == pytest.approx((om_value("wind_speed_10m_mean", x, 10.0) - 0.5) / 3.6)
    assert a.loc[x, "sunshine"] == pytest.approx(om_value("sunshine_duration", x, 10.0) * 0.98 / 3600)
    assert a.loc[x, "soil_moist"] == pytest.approx(om_value("soil_moisture_0_to_7cm_mean", x, 10.0) * 0.98 * 100)
    assert a.loc[x, ow.HUM] > 0 and a.loc[x, "src_avg_rhm"] == "db"
    assert a.index.max() == pd.Timestamp(P.use_end) and a.index.min() == pd.Timestamp("2015-01-01")
    nan_day = pd.Timestamp("2015-01-06")
    assert np.isnan(a.loc[nan_day, "max_wind"]) and a.loc[nan_day, "src_max_wind"] is None
    for v in ow.DB_ONLY_NAN:
        assert t2[v].isna().all() and t2["src_" + v].isna().all()
    assert set(t2["src_avg_temp"].dropna()) == {"db"}
    assert json.loads(pq.read_schema(out / bu.OUT_2).metadata[b"same_as_1st"]) is True
    rep = json.loads((out / bu.OUT_REP).read_text(encoding="utf-8"))
    assert rep["db_nan_cells_not_filled"]["AAA"] == {"max_wind": 1}
    # 나라 목록을 주면 그 나라만(DB 에 없는 CCC 는 NaN 행)
    out2 = tmp_path / "dbonly2"
    assert bu.main(["--root", str(root), "--out", str(out2), "--db-only", "--nations", "nations.csv"]) == 0
    t3 = pd.read_parquet(out2 / bu.OUT_2)
    assert sorted(t3.alpha_3.unique()) == ["AAA", "BBB", "CCC"]
    assert t3.loc[t3.alpha_3 == "CCC", list(ow.TABLE_VARS)].isna().all().all()
    with pytest.raises(bu.BuildStop):      # 덮지 않음
        bu.main(["--root", str(root), "--out", str(out), "--db-only"])
'''
t.write_text(x, encoding="utf-8")
print("ok")
