p = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\tests\test_scope_domestic.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''"s3_1", "s3_2", "s3_3", "s3_extra", "s4", "s5_6", "s8", "s_final", "s0_dom_prep",
                                     "s_dom_rescore")}''', '''"s3_1", "s3_2", "s3_3", "s3_extra", "s4", "s5_6", "s8", "s8_d", "s_final",
                                     "s0_dom_prep", "s_dom_rescore")}''')
rep('''        if stem.startswith("s8_"):
            mod = m["s8"]''', '''        if stem.startswith("s8_D"):
            mod = m["s8_d"]
            return cf.load_config(p, mod.JUDGE_SCHEMA) if sub == "judge" else mod.load_cfg(p)
        if stem.startswith("s8_"):
            mod = m["s8"]''')
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
print("ok")
