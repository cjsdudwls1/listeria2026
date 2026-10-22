# -*- coding: utf-8 -*-
"""out3/png/pNN.png 를 4쪽씩(2x2) 묶은 검수용 그림 -> work3/review/<태그>_sheet_XX.png"""
import glob, os, sys
from PIL import Image, ImageDraw
png_dir, out_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 4
cols = 2 if per == 4 else per
files = sorted(glob.glob(os.path.join(png_dir, "p*.png")))
os.makedirs(out_dir, exist_ok=True)
for f in glob.glob(os.path.join(out_dir, tag + "_sheet_*.png")):
    os.remove(f)
for s in range(0, len(files), per):
    ims = [Image.open(f).convert("RGB") for f in files[s:s + per]]
    w, h = ims[0].size
    sc = 0.6
    tw, th = int(w * sc), int(h * sc)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 10, rows * th + (rows + 1) * 10), (120, 120, 120))
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        x, y = 10 + c * (tw + 10), 10 + r * (th + 10)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + 60, y + 22], fill=(255, 255, 0))
        d.text((x + 4, y + 4), os.path.basename(files[s + k]), fill=(0, 0, 0))
    sheet.save(os.path.join(out_dir, "%s_sheet_%02d.png" % (tag, s // per + 1)))
print(len(files), "pages")
