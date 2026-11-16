# -*- coding: utf-8 -*-
"""Downloads 의 결과 hwp 를 읽기 전용으로 열어 쪽 수·글자 수만 확인(저장 안 함)."""
import sys
from pyhwpx import Hwp

path = sys.argv[1]
hwp = Hwp(new=True, visible=False)
com = hwp.hwp
try:
    ok = com.Open(path, "HWP", "forceopen:true;versionwarning:false")
    txt = com.GetTextFile("TEXT", "")
    print("open", ok, "pages", com.PageCount, "chars", len(txt))
finally:
    com.Clear(1)
    com.Quit()
