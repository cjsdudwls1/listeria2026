# -*- coding: utf-8 -*-
"""PPT 에서 꺼낸 그림의 머리 부분을 잘라 2배로 키워 저장(읽기만, 결과는 work3 아래)."""
import sys
from pathlib import Path
from PIL import Image
src = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_수집\식품안전지수\_문서추출")
out = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\경복대양식\work3\ff_scripts\_img")
jobs = [("slide51_1.png", (0, 0, 630, 170)), ("slide52_2.png", (0, 0, 630, 180)), ("slide16_1.png", (0, 0, 500, 140))]
for name, box in jobs:
    im = Image.open(src / name)
    print(name, im.size)
    c = im.crop(box)
    c = c.resize((c.width * 2, c.height * 2), Image.LANCZOS)
    c.save(out / f"zoom_{name}")
