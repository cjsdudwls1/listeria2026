import sys, json, importlib.util
from pathlib import Path
C = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, str(C))
spec = importlib.util.spec_from_file_location("s56", C / "scripts" / "s5_6.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
inp = Path(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\food2\inp")
for n in ("s5_6_T_food2.json", "s5_6_T_food2_ko.json"):
    cfg = json.loads((C / "configs" / n).read_text(encoding="utf-8"))
    m.check_cfg(cfg)
    b = m.build(cfg, inp)
    sp = b["run"][0]
    print(n, sp.name, sp.stage, sp.input_set, sp.extra["version"]["include_ids"][-2:], sp.extra["s4"], b["bests"][sp.name]["dir"])
