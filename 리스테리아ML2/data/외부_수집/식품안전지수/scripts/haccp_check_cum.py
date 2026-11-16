# 15058303: 식품제조가공업소 LCNS_NO 의 '가장 이른 HACCP_APPN_DT' 로 연말 누적 업소 수를 세어,
# 엑셀 식품안전지수_protocol_20190103.xlsx 의 '식품제조가공업' HACCP 업체 수(출처 글 없음)와 나란히 놓음(참고용).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from haccp_load import load
sys.stdout.reconfigure(encoding='utf-8')
df, _, _ = load('mfds')
ref = {2008: 436, 2009: 692, 2010: 1127, 2011: 1814, 2012: 2966, 2013: 3990, 2014: 2891, 2015: 3517, 2016: 4089}
m = df[df.INDUTY_CD_NM == '식품제조가공업소']
first = m.groupby('LCNS_NO').HACCP_APPN_DT.min().str[:4].astype(int)
allf = df.groupby('LCNS_NO').HACCP_APPN_DT.min().str[:4].astype(int)
print('연도\t엑셀값\tAPI_식품제조가공업소_누적업소\tAPI_전체업종_누적업소\tAPI_그해_처음_지정된_식품제조가공업소')
for y in range(2006, 2021):
    print(y, ref.get(y, ''), int((first <= y).sum()), int((allf <= y).sum()), int((first == y).sum()), sep='\t')
