import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
i = s.index('from __future__ import annotations')
s = '''# -*- coding: utf-8 -*-
"""s3_extra.py — 추가 판 ⑥·⑦(3-0 뒤·3-1 전)·⑧(3-0b 뒤) — 계획서(v78) 요약 13·19, §6 2차 공통(교체 후보·채택 판정·사슬
시작)·단계 3-0·추가 판 표, §7, 부록 A-8·A-9·A-12·A-14·A-15.

판(§6 추가 판 표, 이름 = 부록 A-14):
  설정 configs/s3_extra_G.json(sub 'G', GPU — 3-0 딥러닝이 끝나면 바로 가능):
    x6-LSTMA  ⑥ 시계열만 딥러닝 = LSTM + attention, 입력 = 기상 시퀀스(기상·차감·SPI 채널 전부, 딥러닝 N_ma) + g ÷ 50·해외
              구조적 결측 1칼럼(지역·범주·계절 없음), W_DL = 1차 W_DL — 교체 후보
    x7-D      ⑦-D = 트랜스포머 인코더, 입력 = ⑥ 과 같음 — 교체 후보(⑥ 과 같은 입력이라 한 묶음)
  설정 configs/s3_extra_C.json(sub 'C', CPU — 3-0 트리가 끝난 뒤(XGBoost 최대 반복 기록 필요)):
    x7-T      ⑦-T = XGBoost 기본 설정, 입력 = ⑦-D 의 x_seq 채널을 원값 lag(lag001~lagW, 표준화 없음·NaN 그대로·결측 표시
              없음) + g(지역·범주 없음), 반복 수 §3(교차 N·공통 N — XGBoost 최대 반복은 3-0 에서 정한 값) — 교체 후보
  설정 configs/s3_extra_judge.json(sub 'judge', 학습 없음 — 3-0·G·C 가 끝난 뒤, 3-1 전):
    3-0 순위(13 판 중 CV 1위 = 첫 현재 최고 판 '출발', 트리 6종 CV 순 = 트리 대상 판·트리 상위 3개 — §6 단계 3-0 산출) →
    ⑥·⑦ 교체 후보(x6-LSTMA·x7-T·x7-D) 중 val 1위가 그 현재 최고 판보다 오르면 채택(현재 최고 판만 바뀜 — 그 위에 3-1~3-3 을
    쌓지 않음, 트리 대상 판 그대로), 아니면 버림 → x67_decision.json(3-1~3-3 시작 상태) + 사슬 기록 두 줄('출발'·⑥⑦).
    판정 tag: 1위 = 채택/버림(판정 상대 = 현재 최고 판), 나머지 = 버림(단계 1위 아님). 사슬 시작 알림: 3-0 1위 val 이 1차
    베이스라인보다 낮으면 기록·알림.
  설정 configs/s3_extra_E.json(sub 'E', CPU — 3-0b 판정 뒤(§8 순서), 입력은 3-0 판만 씀):
    x8-<알고리즘> ⑧ 정형만 = 3-0 트리 6종 중 CV 1위 판(변수 전체)에서 기상 8그룹 칼럼을 모두 뺀 판(같은 설정, 부스팅은 반복
              수만 다시) — 비교만, MLflow 단계 run '⑧'
보고(--report, 판정 뒤): G·C·E 설정마다 stage_cli 보고(7-a·1차 베이스라인·3-0 1위 줄, E 는 ⑧ 바탕 판 줄 — 원래 판과 비교).

실행(서버 — 검수 뒤 지휘자):
  P=~/cyj_listeria_final/venv/bin/python; A="--root ~/cyj_listeria_final/data --inputs ~/cyj_listeria_final/results"
  $P scripts/s3_extra.py $A --out ~/cyj_listeria_final/results/s3/x678-G --config configs/s3_extra_G.json   # GPU(끝남)
  $P scripts/s3_extra.py $A --out ~/cyj_listeria_final/results/s3/x678-C --config configs/s3_extra_C.json   # CPU(끝남)
  $P scripts/s3_extra.py $A --out ~/cyj_listeria_final/results/s3/x67-judge --config configs/s3_extra_judge.json  # 3-1 전
  $P scripts/s3_extra.py $A --out ~/cyj_listeria_final/results/s3/x8 --config configs/s3_extra_E.json         # 3-0b 뒤
  보고: G·C·E 명령 + --report --reason "추가 판 ⑥⑦⑧ 보고"

구현 세부(계획서에 없음 — 담당자 확인): lst/stage3 머리 (h)·(i). 3-0 순위(첫 현재 최고 판·트리 대상 판·상위 3개)는 따로
스크립트가 없어 ⑥⑦ 판정이 같이 정해 x67_decision.json 에 적음. '출발' 은 사슬 기록·판정 파일에 남기고, MLflow 판정 tag 는
3-0 1위 run 에 아직 판정 tag 가 없을 때만 '출발'(3-0 판은 만들 때 '판정 없음' 을 받아 보통 그대로 — tag 는 한 번만 씀).
"""
''' + s[i:]


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b)


rep('''STAGE = s3.EXTRA_STAGE
''', '''STAGE = s3.EXTRA_STAGE                              # ⑥⑦(G·C·judge) — 구현 세부 (i)
SUB_STAGE = {"G": s3.EXTRA_STAGE, "C": s3.EXTRA_STAGE, "E": s3.X8_STAGE, "judge": s3.EXTRA_STAGE}
''')
rep('''DIRS_SCHEMA = {"3-0b": cf.Field(str), "x678-G": cf.Field(str), "x678-C": cf.Field(str), "x8": cf.Field(str),
               "x67-judge": cf.Field(str)}''', '''DIRS_SCHEMA = {"3-0-T": cf.Field(str), "3-0-D": cf.Field(str), "x678-G": cf.Field(str), "x678-C": cf.Field(str),
               "x8": cf.Field(str), "x67-judge": cf.Field(str)}''')
rep('''    "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str), "code_reviewed": cf.Field(bool),
    "dirs": DIRS_SCHEMA, "chain_log": cf.Field(str),
}''', '''    "use_mlflow": cf.Field(bool), "mlflow_uri": cf.Field(str), "code_reviewed": cf.Field(bool),
    "decision_1_3": cf.Field(str), "selection_2": cf.Field(str), "dir_1_1": cf.Field(str), "dir_1_2": cf.Field(str),
    "dirs": DIRS_SCHEMA, "chain_log": cf.Field(str),
}''')
rep('''def check_cfg(cfg) -> None:
    if cfg["stage"] != STAGE or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{STAGE}' · order '2차' · experiment 'listeria-2차'(부록 A-15)")
    sub = cfg["sub"]''', '''def check_cfg(cfg) -> None:
    sub = cfg["sub"]
    if cfg["stage"] != SUB_STAGE[sub] or cfg["order"] != s3.ORDER or cfg["experiment"] != s3.EXPERIMENT:
        raise cf.ConfigError(f"stage '{SUB_STAGE[sub]}' · order '2차' · experiment 'listeria-2차'(부록 A-15)")''')
i = s.index('def decision_3_0b(')
j = s.index('def make_specs_fn(')
s = s[:i] + '''def ranking_3_0(inputs_root: Path, cfg: Mapping, samples=None) -> dict:
    """3-0 순위(lst/stage3.rank_3_0) — 판 목록을 결정·고르기 파일로 다시 만들어 끝난 판과 대조."""
    t3 = s3.three_0(inputs_root, cfg["decision_1_3"], cfg["selection_2"], cfg["dirs"])
    return s3.rank_3_0(t3, samples)


def x8_base_spec(inputs_root: Path, rank: Mapping) -> rn.VariantSpec:
    """⑧ 바탕 = 3-0 트리 6종 중 CV 1위 판(변수 전체)의 설정."""
    b = rank["boards"][rank["tree_target"]]
    return rn.VariantSpec(**rn.read_json(resolve(inputs_root, b["dir"]) / "spec.json"))


''' + s[j:]
rep('''            if cfg["sub"] == "E":
                base = x8_base_spec(inputs_root, decision_3_0b(inputs_root, cfg))''', '''            if cfg["sub"] == "E":
                base = x8_base_spec(inputs_root, ranking_3_0(inputs_root, cfg))''')
i = s.index('def report_rows(')
j = s.index('# ================================================================ ⑥·⑦ 판정(학습 없음)')
s = s[:i] + '''def report_rows(cfg, inputs_root, samples, exp_log):
    """보고 비교 줄: 1차 베이스라인 + G·C 는 3-0 1위(⑥⑦ 판정 상대), E 는 ⑧ 바탕 판(3-0 트리 6종 1위 — 원래 판)."""
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    rows = [s3.baseline_1_row(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])]
    if cfg["sub"] == "E":
        rank = ranking_3_0(inputs_root, cfg)
        n = rank["tree_target"]
        rows.append(s3.base_row_from(resolve(inputs_root, rank["boards"][n]["dir"]), n,
                                     "⑧ 바탕 판(3-0 트리 6종 1위 — 원래 판)"))
        return rows
    jd = resolve(inputs_root, cfg["dirs"]["x67-judge"]) / s3.X67_DECISION_FILE
    if not jd.exists():
        raise SystemExit("⑥·⑦ 판정(judge)을 먼저 — test 는 판정 뒤에만(§1-3)")
    d = rn.read_json(jd)
    n = d["current_best_before"]
    rows.append(s3.base_row_from(resolve(inputs_root, d["boards_3_0"][n]["dir"]), n,
                                 "3-0 13 판 중 CV 1위(첫 현재 최고 판 — ⑥⑦ 판정 상대)"))
    return rows


''' + s[j:]
i = s.index('def judge(argv')
j = s.index('# ================================================================ 진입점')
s = s[:i] + '''def judge(argv, *, tracker_fn=sc.make_tracker) -> dict:
    ap = build_parser(__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", required=True)
    rp, args = parse_paths(argv, ap)
    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))
    if cfg["stage"] != STAGE or cfg["experiment"] != s3.EXPERIMENT or cfg["order"] != s3.ORDER:
        raise cf.ConfigError(f"judge 설정: stage {STAGE} · order 2차 · experiment listeria-2차")
    inputs_root = Path(args.inputs).expanduser().resolve()
    out = rp.out
    if out.resolve() != resolve(inputs_root, cfg["dirs"]["x67-judge"]).resolve():
        raise cf.ConfigError("--out 은 --inputs 아래 dirs[x67-judge]")
    p = out / s3.X67_DECISION_FILE
    if p.exists():
        print(f"⑥·⑦ 판정 이미 있음: {p}")
        return rn.read_json(p)
    samples = ev.SampleTable.read(resolve(inputs_root, cfg["samples"]))
    log = rn.ExperimentLog(resolve(inputs_root, cfg["experiment_log"]))
    rank = ranking_3_0(inputs_root, cfg, samples)          # §6 단계 3-0 산출(첫 현재 최고 판·트리 대상 판·상위 3개)
    boards = rank["boards"]
    cur = rank["first"]
    alerts = list(rank["selection"]["alerts"])
    dec = s3.load_decision_1_3(inputs_root, cfg["decision_1_3"])
    base_cv = s3.baseline_1_cv(inputs_root, dec, cfg["dir_1_1"], cfg["dir_1_2"])
    if mt.round4(boards[cur]["cv"]) < mt.round4(base_cv):
        alerts.append(f"사슬 시작: 3-0 1위 {cur} val {boards[cur]['cv']:.4f} < 1차 베이스라인 {dec['baseline']} "
                      f"{base_cv:.4f} — 사슬은 그대로 3-0 1위에서(§6 2차 공통)")
    cpath = resolve(inputs_root, cfg["chain_log"])
    if cpath.exists():
        old = [json.loads(x) for x in cpath.read_text(encoding="utf-8").splitlines() if x.strip()]
        if any(ln.get("judgement") == "출발" for ln in old):
            alerts.append("사슬 기록에 이미 '출발' 줄이 있음(옛 순서 3-0b 가 쓴 줄일 수 있음) — 새 순서에선 안 씀, "
                          "최종 보고(s_final)는 step '3-0' 의 출발 줄만 읽음")
    where = {"x6-LSTMA": "x678-G", "x7-D": "x678-G", "x7-T": "x678-C"}
    dir_of = {n: resolve(inputs_root, cfg["dirs"][where[n]]) / n for n in REPL_NAMES}
    dir_of[cur] = resolve(inputs_root, boards[cur]["dir"])

    def rescore(name):
        return float(ev.score_cv(rn.load_preds(dir_of[name], ev.CV_FOLDS), samples, rescore=True)["cv_band_weighted"])

    cands = [s3.candidate_of(dir_of[n].parent, n) for n in REPL_NAMES]
    cur_c = ev.Candidate(cur, float(boards[cur]["cv"]), int(boards[cur]["train_count"]), bool(boards[cur]["many_ties"]))
    with ev.selection_guard():
        ranked = ev.rank_candidates(cands)
        top = ranked[0]
        j = ev.judge_adoption(top, cur_c, rescore)
    alerts += list(j["alerts"])
    after = top.name if j["adopt"] else cur
    utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    top3 = rank["tree_top3"]
    decision = {"stage": "⑥⑦", "first_3_0": cur, "first_3_0_selection": rank["selection"],
                "boards_3_0": boards, "tree_rank_3_0": rank["tree_rank"], "tree_top3": top3,
                "tree_target": rank["tree_target"],
                "current_inputs": {boards[n]["structure"]: boards[n]["version"] for n in top3},
                "baseline_1": {"name": dec["baseline"], "cv": base_cv},
                "candidates": {c.name: c.cv for c in ranked}, "ranking": [c.name for c in ranked],
                "top": top.name, "judge": j, "adopted": bool(j["adopt"]),
                "current_best_before": cur, "current_best_after": after,
                "current_best_after_dir": (f"{cfg['dirs'][where[after]].rstrip('/')}/{after}" if j["adopt"]
                                           else boards[cur]["dir"]),
                "replacement_adopted": bool(j["adopt"]), "alerts": alerts, "utc": utc}
    tr = tracker_fn(cfg)
    if tr is not None:
        sr = tr.stage_run(s3.EXPERIMENT, STAGE)
        for c in ranked:
            r = tr.find_run(s3.EXPERIMENT, c.name, sr)
            if r is None:
                raise s3.StageError(f"MLflow 판 run 없음: {c.name}")
            if "판정" in r.data.tags:
                continue
            if c is top:
                tr.set_judgement(r.info.run_id, "채택" if j["adopt"] else "버림", against=cur)
                if j["rescore"]:
                    tr.log_rescore(r.info.run_id, j["rescore"]["candidate"])
            else:
                tr.set_judgement(r.info.run_id, "버림(단계 1위 아님)", against=cur)
        r0 = tr.find_run(s3.EXPERIMENT, cur, tr.stage_run(s3.EXPERIMENT, "3-0"))
        if r0 is not None and "판정" not in r0.data.tags:
            tr.set_judgement(r0.info.run_id, "출발")
    lines = [{"step": "3-0", "judgement": "출발", "before": None, "after": cur, "after_dir": boards[cur]["dir"],
              "tree_target": rank["tree_target"], "current_inputs": decision["current_inputs"], "utc": utc},
             {"step": "⑥⑦", "judgement": "채택" if j["adopt"] else "버림", "candidate": top.name, "before": cur,
              "after": after, "after_dir": decision["current_best_after_dir"], "tree_target": rank["tree_target"],
              "current_inputs": decision["current_inputs"], "utc": utc}]
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\\n")
    for a in alerts:
        log.append("alert", stage="⑥⑦", message=a)
        print(f"[알림] {a}")
    log.append("decision", stage="3-0", first=cur, tree_target=rank["tree_target"], tree_top3=top3)
    log.append("decision", stage="⑥⑦", top=top.name, adopted=bool(j["adopt"]), current_best=after)
    out.mkdir(parents=True, exist_ok=True)
    cf.dump_scores_json(p, decision)
    if tr is not None:
        tr.log_artifact(tr.chain_run(), cpath)
        tr.log_artifact(tr.stage_run(s3.EXPERIMENT, STAGE), p, "decision")
    print(f"3-0 1위 {cur}(트리 대상 판 {rank['tree_target']}, 상위 3 {top3}) — ⑥·⑦: val 1위 {top.name} "
          f"{'채택' if j['adopt'] else '버림'} — 현재 최고 판 = {after}")
    return decision


''' + s[j:]
rep('''        msg = f"⑧: 바탕 판 {b.name} 에 기상 8그룹 칼럼이 이미 없음 — 학습 없이 기록(§6 추가 판 ⑧)"''',
    '''        msg = f"⑧: 바탕 판 {b.name} 에 기상 8그룹 칼럼이 이미 없음 — 학습 없이 기록"''')
rep('''from lst import evaluate as ev  # noqa: E402
''', '''from lst import evaluate as ev  # noqa: E402
from lst import metrics as mt  # noqa: E402
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
