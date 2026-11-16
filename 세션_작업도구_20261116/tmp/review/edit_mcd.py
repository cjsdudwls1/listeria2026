p = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts\make_configs_dom.py"
s = open(p, encoding="utf-8", newline="").read()
crlf = "\r\n" in s
s = s.replace("\r\n", "\n")
BS = chr(92)
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''  - code_reviewed → false(지휘자가 검수 뒤 바꿈 — 바꾼 파일은 --check 에서 그 칸만 차이로 허용)
''', '''  - code_reviewed → false(지휘자가 검수 뒤 바꿈 — 바꾼 파일은 --check 에서 그 칸만 차이로 허용). 다시 만들 때 그 칸 말고는
    내용이 같은 기존 파일은 기존 code_reviewed 값을 그대로 둠(지휘자가 바꾼 값을 되돌리지 않게 — 새·바뀐 파일만 false)
''')
rep('''"s3/x8", "s4/4-pre", "s4/4-T", "s4/4-X1", "s4/4-NN", "s4/4-C", "s5_6/5-6-T", "s8/8-T", "s8/8-F"],''',
    '''"s3/x8", "s4/4-pre", "s4/4-T", "s4/4-X1", "s4/4-NN", "s4/4-C", "s5_6/5-6-T", "s8/8-T", "s8/8-F",
                   "s8/8-D"],''')
old = '''    out = []
    for n, cfg in want.items():
        p = dst / n
        p.write_text(dumps(cfg), encoding="utf-8", newline="NL")
'''.replace("NL", BS + "n")
new = '''    out = []
    for n, cfg in want.items():
        p = dst / n
        if p.exists() and "code_reviewed" in cfg:
            with open(p, "r", encoding="utf-8") as f:
                old = json.load(f)
            a, b = dict(old), dict(cfg)
            a.pop("code_reviewed", None)
            b.pop("code_reviewed", None)
            if a == b and "code_reviewed" in old:
                cfg = dict(cfg)
                cfg["code_reviewed"] = old["code_reviewed"]
        p.write_text(dumps(cfg), encoding="utf-8", newline="NL")
'''.replace("NL", BS + "n")
rep(old, new)
if crlf:
    s = s.replace("\n", "\r\n")
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok", crlf)
