# -*- coding: utf-8 -*-
"""docx 안에서 글자를 담은 칸의 XML(칸 속성·글자 속성)을 출력(점검용, 읽기만)."""
import re
import sys
import zipfile

path, needle = sys.argv[1], sys.argv[2]
x = zipfile.ZipFile(path).read('word/document.xml').decode('utf-8')
i = x.find(needle)
a = x.rfind('<w:tc>', 0, i)
b = x.find('</w:tc>', i)
print(x[a:b + 7])
# 그 표의 tblPr·tblGrid
t = x.rfind('<w:tbl>', 0, i)
g = x.find('</w:tblGrid>', t)
print(x[t:g + 12])
