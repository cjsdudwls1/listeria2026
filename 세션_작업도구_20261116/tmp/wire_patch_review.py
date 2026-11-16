from pathlib import Path

F = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")


def patch(path, pairs):
    p = F / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


patch("lst/runner.py", [
    ('''    def count_for(self, stage: str, name: str, kind: str = "판") -> int | None:''',
     '''    def has_cv(self, stage: str, kind: str) -> bool:
        return any(e.get("event") == "cv_scored" and e["stage"] == stage and e["kind"] == kind for e in self.entries())

    def count_for(self, stage: str, name: str, kind: str = "판") -> int | None:'''),
    ('''    smoke: Mapping[str, int] | None = None   # 연기 시험만: {max_train_rows, max_pairs_per_h} — 실제 실행은 None
''', '''    smoke: Mapping[str, int] | None = None   # 연기 시험만: {max_train_rows, max_pairs_per_h} — 실제 실행은 None
    require_7a_logged: bool = True           # CV 채점 단계 시작 전 0-10(7-a 격자 75) 기록 확인 — 연기 시험에서만 끔


TE7A_STAGE, TE7A_KIND = "0-10", "7-a 격자"   # s0_10_log_mlflow 가 남기는 기록


def check_prior_7a(exp_log: ExperimentLog) -> None:
    """'같은 validation 으로 고른 횟수'(부록 A-12)가 7-a 격자 75 부터 세어지도록, CV 채점 단계 전에 0-10 기록 확인."""
    if not exp_log.has_cv(TE7A_STAGE, TE7A_KIND):
        raise ResumeError("실험 로그에 0-10(7-a 격자) 기록이 없음 — s0_10_log_mlflow.py 를 먼저(부록 A-12 셈)")
'''),
    ('''    code_md5 = cf.code_md5(ctx.code_root)
''', '''    if ctx.require_7a_logged:
        check_prior_7a(ctx.exp_log)
    code_md5 = snapshot_code(ctx.code_root, out_dir)   # 단계 시작 때 사본과 다른 코드면 멈춤(한 단계 표에 코드 섞임 방지)
'''),
])

patch("lst/stage_cli.py", [
    ('''        "max_epochs": cf.Field(int, required=False),
''', '''        "max_epochs": cf.Field(int, required=False),
        "skip_7a_check": cf.Field(bool, required=False),   # 0-10 기록 확인 끄기(연기 시험만)
'''),
    ('''                            smoke=({k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None), **kw)''',
     '''                            smoke=({k: cfg["smoke"][k] for k in rn.SMOKE_KEYS} if "smoke" in cfg else None),
                            require_7a_logged=not cfg.get("smoke", {}).get("skip_7a_check", False), **kw)'''),
    ('''    # 기본: 판 전부(하위 프로세스)
    rn.snapshot_code(script.resolve().parents[1], rp.out)''', '''    # 기본: 판 전부(하위 프로세스)
    if not cfg.get("smoke", {}).get("skip_7a_check", False):
        rn.check_prior_7a(exp_log)
    rn.snapshot_code(script.resolve().parents[1], rp.out)'''),
])

patch("tests/test_runner.py", [
    ('''def ctx_of(env, out, asm=FakeAssembler, model=FakeModel):''',
     '''def ctx_of(env, out, asm=FakeAssembler, model=FakeModel, require_7a=False):'''),
    ('''                         check_resources=STUB)
''', '''                         check_resources=STUB, require_7a_logged=require_7a)


def test_code_changed_mid_stage_stops(env):
    """단계 시작 때 남긴 코드 사본(_code/code_md5.txt)과 지금 코드가 다르면 판을 시작하지 않음."""
    out = env["tmp"] / "s1" / "1-1"
    rn.snapshot_code(CODE_ROOT, out)
    (out / "_code" / "code_md5.txt").write_text("0" * 32, encoding="utf-8")
    with pytest.raises(rn.ResumeError, match="code_md5"):
        rn.run_variant(spec_of(), ctx_of(env, out))
    assert not (out / spec_of().name).exists()


def test_requires_prior_7a_log(env):
    """CV 채점 단계는 실험 로그에 0-10(7-a 격자) 기록이 있어야 시작(부록 A-12 셈)."""
    out = env["tmp"] / "s1" / "1-1"
    with pytest.raises(rn.ResumeError, match="0-10"):
        rn.run_variant(spec_of(), ctx_of(env, out, require_7a=True))
    env["log"].cv_scored(rn.TE7A_STAGE, "7-a-5y-ku050-kp050", 75, rn.TE7A_KIND)
    r = rn.run_variant(spec_of(), ctx_of(env, out, require_7a=True))
    assert r["status"] == "done" and r["selection_count"] == 75
'''),
])

p = Path(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\smoke_s1\run_smoke.py")
s = p.read_text(encoding="utf-8")
old = '''        c["smoke"] = dict(SMOKE) if name == "s1_2_lstm" else {k: SMOKE[k] for k in ("max_train_rows", "max_pairs_per_h")}'''
new = '''        c["smoke"] = dict(SMOKE) if name == "s1_2_lstm" else {k: SMOKE[k] for k in ("max_train_rows", "max_pairs_per_h")}
        c["smoke"]["skip_7a_check"] = True   # 0-10 은 서버에만'''
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding="utf-8")
print("ok")
