# -*- coding: utf-8 -*-
"""양식 document.xml 의 표 속성·글꼴·크기 등 확인 (읽기 전용)."""
import re
import sys

x = open(sys.argv[1], encoding='utf-8').read()
for m in re.finditer(r'<w:tblPr>.*?</w:tblGrid>', x):
    print(m.group(0))
    print('---')
print('tcBorders', x.count('tcBorders'), 'tblBorders', x.count('tblBorders'), 'shd', x.count('<w:shd'))
print('sz', sorted(set(re.findall(r'<w:sz w:val="(\d+)"/>', x))))
print('fonts', set(re.findall(r'<w:rFonts [^>]*', x)))
print('jc', set(re.findall(r'<w:jc w:val="(\w+)"/>', x)))
print('shd', set(re.findall(r'<w:shd [^>]*>', x)))
print('color', set(re.findall(r'<w:color [^>]*>', x)))
print('b', x.count('<w:b/>'))
print('sect', re.findall(r'<w:sectPr.*?</w:sectPr>', x))
i = x.find('[전체 모델 요약 성능]')
print(x[i - 1600:i + 60])
print('=====')
i = x.find('모델명')
print(x[i - 1600:i + 400])
print('=====')
# 두 번째 표 첫 데이터 행 셀 하나
i = x.find('XGBoost')
print(x[i - 1200:i + 30])
print('=====')
# 세 번째 표 데이터 행 하나
i = x.find('>1일<')
print(x[i - 1500:i + 1600])
