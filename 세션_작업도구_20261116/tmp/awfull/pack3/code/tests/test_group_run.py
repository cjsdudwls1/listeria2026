"""묶음 실행(같은 입력 설정의 트리 판들) = 판마다 따로 실행과 결과가 바이트 단위로 같은지, 끝난 판 빼기,
코드 바뀜 인정(accept_previous_code — 같으면 인정·다르면 멈춤)."""
import json
from pathlib import Path

import numpy as np
import pytest

from lst import config as cf
from lst import runner as rn
from tests.test_runner import STUB, FakeAssembler, env  # noqa: F401  (env = fixture)

C_OF = {"XGB": 1.0, "LGB": 0.05, "CAT": 20.0, "LR": 3.0}


class AlgModel:
    """알고리즘마다 다른 예측 + 받은 입력을 일부러 고침(묶음에서 입력이 번지면 결과가 달라지게)."""

    def __init__(self, spec, asm=None, shift=0.0):
        self.spec, self.shift = spec, shift

    def fit(self, X, y, *, seed, n_jobs, eval_set=None):
        from sklearn.linear_model import LogisticRegression

        self.m = LogisticRegression(C=C_OF[self.spec.algorithm], max_iter=300).fit(X.fillna(0).to_numpy(), y)
        X.loc[:, "h"] = -999.0          # 입력 고침(사본을 받았는지 시험)
        return self

    def predict_proba(self, X, n_iter=None):
        p = self.m.predict_proba(X.fillna(0).to_numpy())[:, 1].astype(np.float64)
        X.loc[:, "g"] = -999.0
        return np.clip(p + self.shift, 0, 1)

    def save(self, path):
        Path(path).write_text(json.dumps(self.m.coef_.tolist()), encoding="utf-8")


class ShiftedModel(AlgModel):
    def __init__(self, spec, asm=None):
        super().__init__(spec, asm, shift=1e-6)


def specs3(W=20, mode="sum"):
    return [rn.VariantSpec(name=f"1-{a}-W{W:03d}-{mode}", stage="1-1", order="1차", kind="tree", algorithm=a, W=W,
                           mode=mode, input_set="1차-트리", role="후보", train_count=4, judgement="해당 없음",
                           extra={"version": {"include_ids": ["WEATHER"]}}) for a in ("XGB", "LGB", "CAT")]


def ctx(env, out, log, code_root, model=AlgModel):
    return rn.RunContext(samples=env["st"], out_dir=out, code_root=code_root, input_md5s={"x": "0"}, exp_log=log,
                         make_inputs=FakeAssembler, make_model=model, n_jobs=8, check_resources=STUB,
                         require_7a_logged=False)


def fake_code(root: Path, text: str) -> Path:
    (root / "lst").mkdir(parents=True)
    (root / "lst" / "x.py").write_text(text, encoding="utf-8")
    return root


def test_group_specs():
    t = specs3() + specs3(20, "raw") + [rn.VariantSpec(name="1-LSTM-W020", stage="1-2", order="1차", kind="dl",
                                                       algorithm="LSTM", W=20, mode=None, input_set="dl", role="후보",
                                                       train_count=4)]
    lr = rn.VariantSpec(name="3-0-LR", stage="3-0", order="2차", kind="tree", algorithm="LR", W=20, mode="sum",
                        input_set="1차-트리", role="후보", train_count=4, extra={"version": {"include_ids": ["WEATHER"]}})
    g = rn.group_specs([t[0], t[3], lr, t[1], t[6], t[2], t[4], t[5]])
    assert [[s.name for s in x] for x in g] == [
        ["1-XGB-W020-sum", "1-LGB-W020-sum", "1-CAT-W020-sum"], ["1-XGB-W020-raw", "1-LGB-W020-raw", "1-CAT-W020-raw"],
        ["3-0-LR"], ["1-LSTM-W020"]]
    with pytest.raises(ValueError):
        rn.run_group([t[0], t[3]], None)


def test_group_bytes_identical_to_separate(env, tmp_path):
    code = fake_code(tmp_path / "code", "a = 1\n")
    sp = specs3()
    a, b = tmp_path / "A", tmp_path / "B"
    ra = rn.run_group(sp, ctx(env, a, rn.ExperimentLog(tmp_path / "la.jsonl"), code))
    lb = rn.ExperimentLog(tmp_path / "lb.jsonl")
    rb = [rn.run_variant(s, ctx(env, b, lb, code)) for s in sp]
    for s, x, y in zip(sp, ra, rb):
        assert rn.compare_variant_files(a / s.name, b / s.name) == [], s.name
        assert (a / s.name / "model_fold1").read_bytes() == (b / s.name / "model_fold1").read_bytes()
        assert x["selection_count"] == y["selection_count"] and x["group"] == [q.name for q in sp]
    # 알고리즘마다 예측이 실제로 다름(같은 모델을 세 번 쓴 게 아님)
    p = [(a / s.name / "pred_fold1.parquet").read_bytes() for s in sp]
    assert len(set(p)) == 3


def test_group_skips_done(env, tmp_path):
    code = fake_code(tmp_path / "code", "a = 1\n")
    sp = specs3()
    log = rn.ExperimentLog(tmp_path / "l.jsonl")
    rn.run_variant(sp[1], ctx(env, tmp_path / "A", log, code))
    r = rn.run_group(sp, ctx(env, tmp_path / "A", log, code))
    assert [x["status"] for x in r] == ["done", "skipped", "done"]
    assert r[0]["group"] == [sp[0].name, sp[2].name]


def _accept(env, tmp_path, verify_model):
    sp = specs3()
    out = tmp_path / "out"
    code_a = fake_code(tmp_path / "codeA", "a = 1\n")
    code_b = fake_code(tmp_path / "codeB", "a = 2\n")
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    rn.run_group(sp, ctx(env, out, log, code_a))                 # 이전 코드로 끝난 판들
    with pytest.raises(rn.ResumeError):                          # 새 코드로 그냥 이어 돌리면 멈춤
        rn.run_group(sp, ctx(env, out, log, code_b))
    acc = {"code_md5": cf.code_md5(code_a), "reason": "묶음 실행(계산 공유)", "check_variant": sp[0].name}

    def verify(td, names):
        rn.run_group([s for s in sp if s.name in names], ctx(env, td, rn.ExperimentLog(td / "_v.jsonl"), code_b,
                                                              verify_model))
    return sp, out, code_a, code_b, log, acc, verify


def test_accept_previous_code_same(env, tmp_path):
    sp, out, code_a, code_b, log, acc, verify = _accept(env, tmp_path, AlgModel)
    assert rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1") == "accepted"
    old = acc["code_md5"]
    assert (out / f"_code_prev_{old}").is_dir()
    assert (out / "_code" / "code_md5.txt").read_text(encoding="utf-8") == cf.code_md5(code_b)
    assert old in rn.accepted_codes(out)
    ev_ = [e for e in log.entries() if e["event"] == "code_change_same_results"]
    assert ev_ and ev_[0]["compared"] == [s.name for s in sp]
    r = rn.run_group(sp, ctx(env, out, log, code_b))             # 이전 코드로 끝난 판 = 끝난 판
    assert [x["status"] for x in r] == ["skipped"] * 3
    assert rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1") == "already_accepted"


def test_accept_previous_code_different_stops(env, tmp_path):
    sp, out, code_a, code_b, log, acc, verify = _accept(env, tmp_path, ShiftedModel)
    with pytest.raises(rn.ResumeError, match="다름"):
        rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1")
    assert any(e["event"] == "code_change_different_results" for e in log.entries())
    assert (out / "_code" / "code_md5.txt").read_text(encoding="utf-8") == acc["code_md5"]   # 사본 그대로


def test_accept_scope_and_spec_check(env, tmp_path):
    """인정 범위 = 확인 묶음과 같은 실행 경로(묶음 트리·같은 알고리즘·같은 입력 묶음) — 로지스틱 판은 새 코드로 다시.
    부모의 pending_names 는 같은 이름·다른 설정 md5 면 멈춤."""
    sp, out, code_a, code_b, log, acc, verify = _accept(env, tmp_path, AlgModel)
    lr = rn.VariantSpec(name="1-LR-W020-sum", stage="1-1", order="1차", kind="tree", algorithm="LR", W=20, mode="sum",
                        input_set="1차-트리", role="후보", train_count=4, extra={"version": {"include_ids": ["WEATHER"]}})
    other = specs3(20, "raw")
    la = rn.ExperimentLog(tmp_path / "la.jsonl")
    rn.run_variant(lr, ctx(env, out, la, code_a))
    rn.run_group(other, ctx(env, out, la, code_a))
    assert rn.accept_previous_code(out, code_b, acc, sp, verify, log, "1-1") == "accepted"
    rec = rn.accepted_codes(out)[acc["code_md5"]]
    assert set(rec["accepted_variants"]) == {s.name for s in sp + other} and rec["not_accepted_variants"] == [lr.name]
    ev_ = [e for e in log.entries() if e["event"] == "code_change_same_results"][0]
    assert ev_["accepted_variants"] == rec["accepted_variants"]
    cur = cf.code_md5(code_b)
    assert rn.pending_names(out, sp + other + [lr], cur) == [lr.name]
    r = rn.run_variant(lr, ctx(env, out, log, code_b))           # 인정 범위 밖 → 옮기고 새 코드로 다시
    assert r["status"] == "done" and r["code_md5"] == cur
    assert any(p.name.startswith(lr.name + "__") for p in (out / "_superseded").iterdir())
    bad = rn.VariantSpec(**{**sp[0].to_dict(), "params": {"max_iter": 10}})
    with pytest.raises(rn.ResumeError, match="설정이 다름"):
        rn.pending_names(out, [bad], cur)


def test_accept_chain_two_previous_codes(env, tmp_path):
    """이어 인정: 코드 A → B 인정 뒤(A 로 끝난 판 일부 + B 로 끝난 판) B → C 도 인정 — 두 이전 코드 계열 모두 이어받음.
    확인 판이 A 로 끝나 인정된 판이어도 되고, 비교는 묶음 전부(A·B 결과 모두)."""
    sp = specs3()
    out = tmp_path / "out"
    code_a = fake_code(tmp_path / "codeA", "a = 1\n")
    code_b = fake_code(tmp_path / "codeB", "a = 2\n")
    code_c = fake_code(tmp_path / "codeC", "a = 3\n")
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    rn.run_variant(sp[0], ctx(env, out, log, code_a))                      # A 로 XGB 만

    def verify_with(code):
        def f(td, names):
            rn.run_group([s for s in sp if s.name in names], ctx(env, td, rn.ExperimentLog(td / "_v.jsonl"), code))
        return f
    acc_ab = {"code_md5": cf.code_md5(code_a), "reason": "묶음 실행", "check_variant": sp[0].name}
    assert rn.accept_previous_code(out, code_b, acc_ab, sp, verify_with(code_b), log, "1-1") == "accepted"
    r = rn.run_group(sp, ctx(env, out, log, code_b))                       # B 로 LGB·CAT
    assert [x["status"] for x in r] == ["skipped", "done", "done"]
    acc_bc = {"code_md5": cf.code_md5(code_b), "reason": "계산 중복 없애기", "check_variant": sp[0].name}
    assert rn.accept_previous_code(out, code_c, acc_bc, sp, verify_with(code_c), log, "1-1") == "accepted"
    rec = [e for e in log.entries() if e["event"] == "code_change_same_results"][-1]
    assert rec["compared"] == [s.name for s in sp]
    assert rec["compared_code_md5"] == {sp[0].name: cf.code_md5(code_a), sp[1].name: cf.code_md5(code_b),
                                        sp[2].name: cf.code_md5(code_b)}
    acc = rn.accepted_codes(out)
    assert set(acc) == {cf.code_md5(code_a), cf.code_md5(code_b)}
    assert acc[cf.code_md5(code_a)]["accepted_variants"] == [sp[0].name]
    assert set(acc[cf.code_md5(code_b)]["accepted_variants"]) == {sp[1].name, sp[2].name}
    assert (out / f"_code_prev_{cf.code_md5(code_a)}").is_dir() and (out / f"_code_prev_{cf.code_md5(code_b)}").is_dir()
    r = rn.run_group(sp, ctx(env, out, log, code_c))                       # 셋 다 끝난 판
    assert [x["status"] for x in r] == ["skipped"] * 3
    assert rn.pending_names(out, sp, cf.code_md5(code_c)) == []
    assert rn.accept_previous_code(out, code_c, acc_bc, sp, verify_with(code_c), log, "1-1") == "already_accepted"


def test_accept_chain_drops_out_of_scope(env, tmp_path):
    """A 로 끝나 인정된 판도 이번 확인 묶음 기준 인정 범위(같은 규칙)를 벗어나면 인정 목록에서 빠져 다시 돎."""
    sp = specs3()
    out = tmp_path / "out"
    code_a = fake_code(tmp_path / "codeA", "a = 1\n")
    code_b = fake_code(tmp_path / "codeB", "a = 2\n")
    code_c = fake_code(tmp_path / "codeC", "a = 3\n")
    log = rn.ExperimentLog(tmp_path / "log.jsonl")
    rn.run_group(sp, ctx(env, out, log, code_a))

    def verify_with(code):
        def f(td, names):
            rn.run_group([s for s in sp if s.name in names], ctx(env, td, rn.ExperimentLog(td / "_v.jsonl"), code))
        return f
    acc_ab = {"code_md5": cf.code_md5(code_a), "reason": "묶음 실행", "check_variant": sp[0].name}
    rn.accept_previous_code(out, code_b, acc_ab, sp, verify_with(code_b), log, "1-1")
    acc_bc = {"code_md5": cf.code_md5(code_b), "reason": "x", "check_variant": sp[0].name}
    # 확인 묶음을 XGB 하나로 좁힘 → A 로 끝난 LGB·CAT 은 이번 인정 범위 밖
    assert rn.accept_previous_code(out, code_c, acc_bc, sp[:1], verify_with(code_c), log, "1-1") == "accepted"
    a = rn.accepted_codes(out)[cf.code_md5(code_a)]
    assert a["accepted_variants"] == [sp[0].name] and set(a["not_accepted_variants"]) >= {sp[1].name, sp[2].name}
    assert rn.pending_names(out, sp, cf.code_md5(code_c)) == [sp[1].name, sp[2].name]
