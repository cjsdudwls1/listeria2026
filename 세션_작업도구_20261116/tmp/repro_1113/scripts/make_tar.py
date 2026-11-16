# -*- coding: utf-8 -*-
"""패키지 폴더를 무압축 tar(PAX, UTF-8 이름)로 묶고 tar 파일 md5 를 출력한다.
사용: py -3 -I make_tar.py <패키지 폴더> <출력 tar>
"""
import hashlib
import sys
import tarfile
from pathlib import Path

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
n = 0
size = 0
with tarfile.open(out, "w", format=tarfile.PAX_FORMAT, encoding="utf-8") as tf:
    for p in sorted(src.rglob("*"), key=lambda x: x.relative_to(src).as_posix()):
        if p.is_file():
            arc = src.name + "/" + p.relative_to(src).as_posix()
            tf.add(p, arcname=arc, recursive=False)
            n += 1
            size += p.stat().st_size
h = hashlib.md5()
with open(out, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
print(f"files {n} bytes {size} tar_md5 {h.hexdigest()} tar_bytes {out.stat().st_size}")
