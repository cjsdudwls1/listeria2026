import json
from pathlib import Path

F = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")


def patch(path, pairs):
    p = F / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


patch("scripts/s1_1_trees.py", [
    ('''                          input_set=cfg["input_set"], role="후보", train_count=4, params={}, judgement="해당 없음")
           for a in ALGORITHMS''', '''                          input_set=cfg["input_set"], role="후보", train_count=4, params={}, judgement="해당 없음",
                          extra={"version": {"include_ids": list(ORDER1_IDS)}})
           for a in ALGORITHMS'''),
    ('''def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """다른 에이전트 모듈(lst.inputs·lst.registry)에 붙이는 곳 — 인터페이스 약속(브리프 11-07 저녁):
    make_inputs(spec, split) → fit(train 행 표)·transform(행 표) → X(트리 DataFrame: 숫자 float32·범주 원 문자열),
    make_model(spec) → fit(X, y, *, seed, n_jobs, eval_set=None)·predict_proba(X, n_iter=None)·save(path).
    연결 이름은 그 모듈이 끝나면 지휘자 확인 뒤 맞춤(지금은 가정한 이름)."""
    from lst import inputs as li  # noqa: F401  (다른 에이전트 작성 중)
    from lst import registry as rg  # noqa: F401

    return sc.Factories(make_inputs=li.make_assembler(cfg, rp, inputs_root),
                        make_model=rg.make_model,
                        input_md5s=li.input_md5s(cfg, rp, inputs_root))''',
     '''def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """연결(lst.wiring): 입력 = inputs.TreeInputs(판·split 마다 새로), 모델 = models_tree.make_model(라이브러리 기본),
    자료 = 국내·해외 기상 일자료(1차 = WEATHER 만), 입력 md5 = context 에 적힌 파일 전부."""
    from lst import wiring as wr

    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)'''),
    ('''ALGORITHMS = ("XGB", "LGB", "CAT")''', '''ALGORITHMS = ("XGB", "LGB", "CAT")
ORDER1_IDS = ("WEATHER",)          # registry.ORDER1_IDS — 범주 5·h·g 는 tree_columns 가 늘 넣음
COMPONENTS = ("weather",)          # inputs.load_context 에서 읽을 자료'''),
])

patch("scripts/s1_2_lstm.py", [
    ('''                          mode=None, input_set=cfg["input_set"], role="후보", train_count=4, params={},
                          judgement="해당 없음")''', '''                          mode=None, input_set=cfg["input_set"], role="후보", train_count=4, params={},
                          judgement="해당 없음", extra={"version": {"include_ids": ["WEATHER"]}})'''),
    ('''def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """다른 에이전트 모듈에 붙이는 곳 — 딥러닝 입력 dict(x_seq float32 (N, W, C)·x_static float32 (N, S)·
    x_cat int64 (N, 4)), 표준화·결측 0·결측 표시는 조립 쪽 전처리기(그 모델 train 으로 fit). 조기 종료 묶음
    ((sgg, d) 10%, 부록 A-8)을 모델에 넘기는 방법은 fit_kwargs 로(연결 때 정함)."""
    from lst import inputs as li  # noqa: F401  (다른 에이전트 작성 중)
    from lst import registry as rg  # noqa: F401

    return sc.Factories(make_inputs=li.make_assembler(cfg, rp, inputs_root),
                        make_model=rg.make_model,
                        input_md5s=li.input_md5s(cfg, rp, inputs_root),
                        fit_kwargs=getattr(li, "dl_fit_kwargs", None))''',
     '''def factories(cfg: dict, rp, inputs_root: Path) -> sc.Factories:
    """연결(lst.wiring): 입력 = inputs.DLInputs(판·split 마다 새로 — 표준화는 그 모델 train 으로), 모델 =
    models_dl.DLClassifier('LSTM', cat_sizes), fit 에 groups = (train 행 sgg, d)(부록 A-8 조기 종료 묶음).
    자료 = 국내·해외 기상 일자료 + 단계 0-6 결측 표시 목록. smoke.max_epochs 는 연기 시험에서만."""
    from lst import wiring as wr

    dl_kw = {"max_epochs": cfg["smoke"]["max_epochs"]} if "smoke" in cfg and "max_epochs" in cfg["smoke"] else {}
    w = wr.load(cfg["context"], rp.root, inputs_root, COMPONENTS, dl_kwargs=dl_kw)
    return sc.Factories(make_inputs=w.make_inputs, make_model=w.make_model,
                        input_md5s=wr.input_md5s(cfg["context"], rp.root, inputs_root), fit_kwargs=w.fit_kwargs)'''),
    ('''SCHEMA_EXTRA = {''', '''COMPONENTS = ("weather", "missing")   # inputs.load_context 에서 읽을 자료

SCHEMA_EXTRA = {'''),
])

ctx1 = {"results": {"samples": "s0/0-1/apply/samples.parquet",
                    "domestic_weather": "s0/0-4_v2/시군구_일별_기상_0-4_합본.parquet",
                    "overseas_weather": "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet"},
        "data": {}, "y_cut": "2025-06-30"}
for name, extra in (("s1_1_trees.json", {}), ("s1_2_lstm.json", {"missing_lists": "s0/0-6_missing/missing_lists.json"})):
    p = F / "configs" / name
    c = json.loads(p.read_text(encoding="utf-8"))
    ctx = json.loads(json.dumps(ctx1))
    ctx["results"].update(extra)
    c["context"] = ctx
    p.write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("ok")
