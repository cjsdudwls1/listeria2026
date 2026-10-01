"""노트북 셀 소스(출력 제외)에서 정규식에 맞는 셀을 통째로 출력한다. 사용: py nb_grep_cells.py <노트북> <정규식>"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

path, pat = sys.argv[1], re.compile(sys.argv[2])
nb = json.load(open(path, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if pat.search(src):
        print(f"===== [{i}] {c['cell_type']} id={c.get('id')} =====")
        print(src)
