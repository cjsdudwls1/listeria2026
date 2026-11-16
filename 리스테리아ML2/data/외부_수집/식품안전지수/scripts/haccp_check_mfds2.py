# 15058303 추가 점검: 품목명 '-' 행과 품목 있는 행의 관계, 같은 지정번호의 날짜 여러 개, 같은 업소의 주소 여러 개.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from haccp_load import load
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
df, _, _ = load('mfds')
df['no'] = df.HACCP_APPN_NO.str.strip()
df['dash'] = df.PRDLST_NM.fillna('').str.strip().isin(['', '-'])
k = df[~df.dash].groupby(['LCNS_NO', 'HACCP_APPN_DT']).size()
d = df[df.dash]
hit = d.set_index(['LCNS_NO', 'HACCP_APPN_DT']).index.isin(k.index)
print(f"품목명 빈칸/'-' 행 {len(d):,} 중 같은 LCNS_NO·같은 HACCP_APPN_DT 에 품목 있는 행이 있는 것 {int(hit.sum()):,} ({hit.mean()*100:.1f}%)")
lc_has = d.LCNS_NO.isin(df[~df.dash].LCNS_NO)
print(f"품목명 빈칸/'-' 행 중 그 LCNS_NO 에 품목 있는 행이 하나라도 있는 것 {int(lc_has.sum()):,} ({lc_has.mean()*100:.1f}%)")
only = df.groupby('LCNS_NO').dash.all()
print(f"모든 행이 품목명 빈칸/'-' 인 LCNS_NO {int(only.sum()):,} / {len(only):,}")
n = df.groupby('no').HACCP_APPN_DT.nunique()
print(f"같은 HACCP_APPN_NO 에 지정일 2개 이상: {int((n>1).sum()):,} 개 번호")
ex = n[n > 1].index[:5]
print(df[df.no.isin(ex)][['LCNS_NO', 'HACCP_APPN_NO', 'HACCP_APPN_DT', 'PRDLST_NM', 'SITE_ADDR']].sort_values(['HACCP_APPN_NO', 'HACCP_APPN_DT']).to_string())
a = df.groupby('LCNS_NO').SITE_ADDR.nunique()
print(f"같은 LCNS_NO 에 SITE_ADDR 2개 이상: {int((a>1).sum()):,}")
ex = a[a > 1].index[:8]
print(df[df.LCNS_NO.isin(ex)].groupby(['LCNS_NO', 'SITE_ADDR']).size().to_string())
# 시도 기준으로 LCNS_NO 앞자리(연도 4자리?) 꼴
print('LCNS_NO 길이:', df.LCNS_NO.str.len().value_counts().to_dict())
print('LCNS_NO 앞 4자리(연도로 보임) 범위:', df.LCNS_NO.str[:4].min(), df.LCNS_NO.str[:4].max())
