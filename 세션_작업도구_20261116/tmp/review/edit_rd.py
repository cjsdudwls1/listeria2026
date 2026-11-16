p = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\run_dom\run_dom_cpu.sh"
s = open(p, encoding="utf-8", newline="").read()
a = "need s8/8-judge/8_decision.json logs/chain_state_8.json\necho DOM_S8_DONE\n"
b = a + """
# ---- 단계 8-D(딥러닝 튜닝 — 대상에 딥러닝 부분이 없으면 판 없이 상태만 넘김, GPU 상한 0.25)
$P -I scripts/s8_d.py $A --out $RD/s8/8-D-tune --config $CFG/s8_D_tune.json
$P -I scripts/s8_d.py $A --out $RD/s8/8-D --config $CFG/s8_D.json
$P -I scripts/s8_d.py $A --out $RD/s8/8-D-judge --config $CFG/s8_D_judge.json
need s8/8-D-judge/8D_decision.json logs/chain_state_8D.json
echo DOM_S8D_DONE
"""
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
