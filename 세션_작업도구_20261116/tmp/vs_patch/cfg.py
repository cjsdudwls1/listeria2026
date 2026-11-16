import json
import sys
from pathlib import Path

C = Path(sys.argv[1])


def load(n):
    return json.loads((C / f"{n}.json").read_text(encoding="utf-8"))


def save(n, d):
    (C / f"{n}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


EXTRA_DIRS = {"3-0-T": "s3/3-0-T", "3-0-D": "s3/3-0-D", "x678-G": "s3/x678-G", "x678-C": "s3/x678-C",
              "x8": "s3/x8", "x67-judge": "s3/x67-judge"}
for n in ("s3_extra_G", "s3_extra_C", "s3_extra_E", "s3_extra_judge"):
    d = load(n)
    d["dirs"] = dict(EXTRA_DIRS)
    if n == "s3_extra_E":
        d["stage"] = "⑧"
        d["plan"] = "추가 판 ⑧ x8(정형만 — 3-0 트리 6종 중 CV 1위 판에서 기상 8그룹 뺌, 비교만, 3-0b 뒤) — §6 추가 판, §3"
    if n == "s3_extra_judge":
        d["plan"] = "3-0 순위(첫 현재 최고 판·트리 대상 판·상위 3개) + 추가 판 ⑥·⑦ 채택 판정(교체 후보 val 1위 대 3-0 1위) — §6 2차 공통·3-0, §1-2"
        out = {}
        for k, v in d.items():
            out[k] = v
            if k == "code_reviewed":
                out["decision_1_3"] = "s1/1-3/1-3_decision.json"
                out["selection_2"] = "s2/2-select/2_prep_selection.json"
                out["dir_1_1"] = "s1/1-1"
                out["dir_1_2"] = "s1/1-2"
        d = out
    save(n, d)

for n in ("s3_1", "s3_2", "s3_3"):
    d = load(n)
    d["dirs"] = {k: v for k, v in d["dirs"].items() if k != "3-0b"}
    save(n, d)

for n in ("s4_T", "s4_X1", "s4_NN", "s4_C", "s4_judge"):
    d = load(n)
    assert d["chain_state_in"] == "logs/chain_state_3-3.json", n
    d["chain_state_in"] = "logs/chain_state_3-0b.json"
    save(n, d)

d = load("s3_0b")
new = {}
for k, v in d.items():
    if k == "plan":
        v = "단계 3-0b 변수 빼기(1등 = 3-3 판정 뒤 현재 최고 판) — §6 3-0b, 부록 A-5·A-9·A-12·A-14·A-15"
    if k == "dirs":
        v = {"3-0-T": "s3/3-0-T", "3-0-D": "s3/3-0-D", "3-0b": "s3/3-0b-best"}
        new[k] = v
        new["reuse_3_0b_old"] = "s3/3-0b"
        continue
    if k == "chain_log":
        new[k] = v
        new["chain_state_in"] = "logs/chain_state_3-3.json"
        new["chain_state_out"] = "logs/chain_state_3-0b.json"
        continue
    new[k] = v
save("s3_0b", new)

d = load("s_final")
if "decision_3_0b" in d:
    d["decision_3_0b"] = "s3/3-0b-best/3-0b_decision.json"
save("s_final", d)
print("ok")
