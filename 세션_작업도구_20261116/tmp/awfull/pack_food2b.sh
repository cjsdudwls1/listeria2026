#!/bin/bash
# 로컬 final/ → 줄바꿈 LF 로 맞춘 사본 → code_food2b.tgz (x/·캐시 제외)
set -euo pipefail
SRC="/c/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final"
W="/c/Users/NT751/.claude/jobs/a19471fc/tmp/awfull/pack2"
rm -rf "$W" && mkdir -p "$W/code"
cd "$SRC" && tar --force-local --exclude=./x --exclude=__pycache__ --exclude=.pytest_cache -cf - . | tar --force-local -xf - -C "$W/code"
py -3 - "$W/code" <<'PY'
import pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file() and p.suffix in (".py", ".sh", ".json", ".ini", ".md", ".txt"):
        b = p.read_bytes()
        if b"\r\n" in b:
            p.write_bytes(b.replace(b"\r\n", b"\n")); n += 1
print("LF 로 바꾼 파일", n)
PY
cd "$W/code" && find . -type f | sort | xargs md5sum > "$W/code_md5.txt" && tar --force-local -czf "$W/code_food2b.tgz" . && md5sum "$W/code_food2b.tgz" && wc -l < "$W/code_md5.txt"
