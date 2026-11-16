# -*- coding: utf-8 -*-
"""쪽 PNG 를 n장씩 한 장으로(검수용). 사용: py sheet.py <png_dir> <out_prefix> [per=4]"""
import glob
import os
import sys

from PIL import Image

d, prefix = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 4
files = sorted(glob.glob(os.path.join(d, "p*.png")))
for k in range(0, len(files), per):
    ims = [Image.open(f) for f in files[k:k + per]]
    w = sum(i.width for i in ims) + 10 * (len(ims) - 1)
    h = max(i.height for i in ims)
    sheet = Image.new("RGB", (w, h), "white")
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0))
        x += i.width + 10
    sheet.save(f"{prefix}_{k // per + 1:02d}.png")
print("ok", len(files))
