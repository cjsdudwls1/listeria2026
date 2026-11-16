from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\stage_cli.py"); s = p.read_text(encoding="utf-8")


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''    "log_pred_artifacts": cf.Field(bool),
}
''', '''    "log_pred_artifacts": cf.Field(bool),
    "context": cf.Field(dict),                # inputs.load_context 설정(CONTEXT_SCHEMA — 그쪽이 검사)
    "smoke": {                                # 연기 시험만(실제 실행 설정엔 없음)
        "max_train_rows": cf.Field(int),
        "max_pairs_per_h": cf.Field(int),
        "max_epochs": cf.Field(int, required=False),
    },
}
SMOKE_ONLY_URI = ("sqlite:",)   # 연기 시험 설정은 서버 MLflow 에 못 씀
''')
rep('''    cfg = dict(cf.load_config(path, schema))
''', '''    for k in ("smoke",):   # 선택 칸(하위 schema)
        pass
    raw = cf.load_json_strict(path)
    if "smoke" not in raw:
        schema = {k: v for k, v in schema.items() if k != "smoke"}
    cfg = dict(cf.validate(raw, schema))
    if "smoke" in cfg and cfg["use_mlflow"] and not cfg["mlflow_uri"].startswith(SMOKE_ONLY_URI):
        raise cf.ConfigError("연기 시험(smoke) 설정은 로컬 sqlite MLflow 에만")
''')
rep('''    from .tracking import Tracker

    return Tracker(cfg["mlflow_uri"])''', '''    from .tracking import Tracker

    uri = cfg["mlflow_uri"]
    if uri.startswith("sqlite:///"):   # 연기 시험: artifact 를 db 옆 artifacts/ 에(작업 폴더 mlruns 에 안 쌓이게)
        db = Path(uri[len("sqlite:///"):])
        return Tracker(uri, artifact_root=(db.parent / "artifacts").resolve().as_uri())
    return Tracker(uri)''')
rep('''                            log_pred_artifacts=cfg["log_pred_artifacts"], **kw)''',
    '''                            log_pred_artifacts=cfg["log_pred_artifacts"],
                            smoke=({k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None), **kw)''')
p.write_text(s.replace('''    for k in ("smoke",):   # 선택 칸(하위 schema)
        pass
''', ""), encoding="utf-8")
print("ok")
