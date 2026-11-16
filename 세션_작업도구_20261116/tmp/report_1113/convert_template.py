# -*- coding: utf-8 -*-
"""양식 hwp 사본 -> hwpx, pdf, txt (원본은 건드리지 않음)."""
import os, shutil, sys
from pyhwpx import Hwp

W = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\NT751\Downloads\리스테리아_예측모델_개발_연구보고서_20261027.hwp"
TPL = os.path.join(W, "tpl")
os.makedirs(TPL, exist_ok=True)
copy = os.path.join(TPL, "template.hwp")
shutil.copyfile(SRC, copy)

hwp = Hwp(new=True, visible=False)
com = hwp.hwp
com.Open(copy, "HWP", "forceopen:true")
com.SaveAs(os.path.join(TPL, "template.hwpx"), "HWPX", "")
com.SaveAs(os.path.join(TPL, "template.pdf"), "PDF", "")
txt = com.GetTextFile("TEXT", "")
with open(os.path.join(TPL, "template.txt"), "w", encoding="utf-8") as f:
    f.write(txt)
com.Clear(1)
com.Quit()
print("done")
