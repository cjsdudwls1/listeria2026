# I0580 사건표 + 업종 대응표 + 첫 사건표에 업종묶음 붙인 판.
#   1) 데이터셋\외부_지표\haccp_인증_I0580.parquet : I0580 한 행 = 한 행(지우거나 합치지 않음). 날짜 5개는 원값을 date32 로만 바꿈.
#   2) 중간\외부지표\haccp_업종_대응.csv : 자료·업종별 행 수·업소 수·업종묶음·우리 2단계 코드 목록·근거
#   3) 데이터셋\외부_지표\haccp_인증_업종묶음.parquet : 첫 사건표(haccp_인증.parquet, 덮지 않음)에 업종묶음 칸만 붙인 사본
#   4) 중간\외부지표\haccp_I0580_시군구_대응표.csv : SITE_ADDR 원문별 시군구 결과
# 시군구 규칙은 haccp_sgg_rules.py(= 첫 사건표 haccp_build.py 의 규칙 그대로). 누적 수·비율은 계산하지 않음.
# 실행: <리스테리아ML\.venv-exp2>\python.exe -I i0580_build.py
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from i0580_load import load, COLS
from haccp_sgg_rules import map_addr, CODE2NAME
import haccp_induty_groups as IG

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data'
OUT = BASE + r'\데이터셋\외부_지표\haccp_인증_I0580.parquet'
FIRST = BASE + r'\데이터셋\외부_지표\haccp_인증.parquet'
FIRST_OUT = BASE + r'\데이터셋\외부_지표\haccp_인증_업종묶음.parquet'
MID = BASE + r'\중간\외부지표'
for p in (OUT, FIRST_OUT):
    if os.path.exists(p) and not (len(sys.argv) > 1 and sys.argv[1] == 'overwrite'):
        raise SystemExit(f'이미 있음(덮지 않음): {p}')

L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)


def to_date(s):
    if s is None or not str(s).strip():
        return None
    s = str(s).strip()
    try:
        if re.fullmatch(r'\d{8}', s):
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None
    return None


# ---------------- I0580 ----------------
df, files, info = load()
dup = df.duplicated(COLS, keep='first')
out = pd.DataFrame({
    'src': 'I0580',
    'lcns_no': df.LCNS_NO, 'haccp_no': df.HACCP_APPN_NO, 'bssh_nm': df.BSSH_NM,
    'induty_nm': df.INDUTY_CD_NM, 'prdlst_nm': df.PRDLST_NM,
    'appn_de_raw': df.HACCP_APPN_DT, 'crtfc_end_de_raw': df.CRTFC_ENDDT, 'asgn_cancl_de_raw': df.ASGN_CANCL_DT,
    'crtfc_retn_de_raw': df.CRTFC_RETN_DT, 'clsbiz_de_raw': df.CLSBIZ_DT,
    'clsbiz_dvs_nm': df.CLSBIZ_DVS_CD_NM, 'addr_raw': df.SITE_ADDR,
})
DATE_MAP = {'appn_de': 'appn_de_raw', 'crtfc_end_de': 'crtfc_end_de_raw', 'asgn_cancl_de': 'asgn_cancl_de_raw',
            'crtfc_retn_de': 'crtfc_retn_de_raw', 'clsbiz_de': 'clsbiz_de_raw'}
bad = {}
for c, r in DATE_MAP.items():
    out[c] = out[r].map(to_date)
    bad[c] = int((out[c].isna() & out[r].fillna('').str.strip().ne('')).sum())
amap = {a: map_addr(a) for a in out.addr_raw.fillna('').drop_duplicates()}
k = out.addr_raw.fillna('')
out['sgg_code'] = k.map(lambda a: amap[a][0])
out['sgg_nm'] = out.sgg_code.map(CODE2NAME)
out['sgg_시단위'] = k.map(lambda a: amap[a][1])
out['sgg_대응'] = k.map(lambda a: amap[a][2])
gm = {v: IG.group_of(v) for v in out.induty_nm.drop_duplicates()}
out['업종묶음'] = out.induty_nm.map(lambda v: gm[v][0])
out['dup_in_src'] = dup.values
out['raw_start'] = df._start.astype('int32').values
out['raw_pos'] = df._pos.astype('int16').values

ORDER = ['src', 'lcns_no', 'haccp_no', 'bssh_nm', 'induty_nm', '업종묶음', 'prdlst_nm',
         'appn_de', 'crtfc_end_de', 'asgn_cancl_de', 'crtfc_retn_de', 'clsbiz_de',
         'appn_de_raw', 'crtfc_end_de_raw', 'asgn_cancl_de_raw', 'crtfc_retn_de_raw', 'clsbiz_de_raw',
         'clsbiz_dvs_nm', 'addr_raw', 'sgg_code', 'sgg_nm', 'sgg_시단위', 'sgg_대응', 'dup_in_src', 'raw_start', 'raw_pos']


def to_table(d, order, dates, bools=(), ints=None):
    ints = ints or {}
    arrs = []
    for c in order:
        if c in dates:
            arrs.append(pa.array([None if (x is None or (isinstance(x, float) and pd.isna(x))) else x for x in d[c]], type=pa.date32()))
        elif c in bools:
            arrs.append(pa.array(d[c].astype(bool).tolist(), type=pa.bool_()))
        elif c in ints:
            arrs.append(pa.array(d[c].tolist(), type=ints[c]))
        else:
            v = d[c].astype(object).where(d[c].notna(), None).tolist()
            arrs.append(pa.array([None if x is None else str(x) for x in v], type=pa.string()))
    return pa.Table.from_arrays(arrs, names=order)


tbl = to_table(out, ORDER, set(DATE_MAP), bools={'dup_in_src'}, ints={'raw_start': pa.int32(), 'raw_pos': pa.int16()})
pq.write_table(tbl, OUT)
P('저장:', OUT, f'{tbl.num_rows:,}행 × {tbl.num_columns}칼럼')
P('날짜로 못 읽은 값(채워진 값 중):', bad)
P('완전 중복 표시(dup_in_src):', int(out.dup_in_src.sum()))

# 시군구 대응표
a = out.groupby(k, dropna=False).size().rename('행 수').reset_index().rename(columns={'addr_raw': 'SITE_ADDR 원문'})
a.columns = ['SITE_ADDR 원문', '행 수']
a['sgg_code'] = a['SITE_ADDR 원문'].map(lambda x: amap[x][0]); a['sgg_nm'] = a.sgg_code.map(CODE2NAME)
a['sgg_시단위'] = a['SITE_ADDR 원문'].map(lambda x: amap[x][1]); a['대응 방법'] = a['SITE_ADDR 원문'].map(lambda x: amap[x][2])
a.sort_values('행 수', ascending=False).to_csv(MID + r'\haccp_I0580_시군구_대응표.csv', index=False, encoding='utf-8-sig')
P('\n== I0580 시군구')
P('sgg_code 있음', int(out.sgg_code.notna().sum()), f'({out.sgg_code.notna().mean()*100:.2f}%)', '/ 시 단위만', int(out.sgg_시단위.notna().sum()),
  '/ 그 밖 비움', int((out.sgg_code.isna() & out.sgg_시단위.isna()).sum()))
meth = out.sgg_대응.map(lambda s: re.sub(r'\(뒤 낱말 ".*" 은 구 이름이 아님\)', '(뒤 낱말 생략)', s))
meth = meth.map(lambda s: re.sub(r'에 ".*" 없음', '에 (이름) 없음', s))
P('대응 방법별 행:'); P(meth.value_counts().to_string())
P('시 단위만인 시:', out.sgg_시단위.value_counts().to_dict())
P('비운 행 원문 예:', out[out.sgg_code.isna() & out.sgg_시단위.isna()].addr_raw.value_counts().head(20).to_dict())
P('시군구 256개 중 행 있는 곳:', out.sgg_code.nunique(), '/ 시군구별 행 수 중앙값:', int(out.groupby('sgg_code').size().median()))

# ---------------- 첫 사건표 + 업종묶음 ----------------
first = pq.read_table(FIRST)
fd = first.to_pandas()
gm1 = {v: IG.group_of(v) for v in fd.induty_nm.drop_duplicates()}
grp = [gm1[v][0] for v in fd.induty_nm]
names = first.column_names
pos = names.index('induty_nm') + 1
first2 = first.add_column(pos, '업종묶음', pa.array(grp, type=pa.string()))
pq.write_table(first2, FIRST_OUT)
P('\n저장:', FIRST_OUT, f'{first2.num_rows:,}행 × {first2.num_columns}칼럼 (첫 사건표 칼럼 그대로 + 업종묶음)')

# ---------------- 업종 대응표 ----------------
pairs = IG.our_pairs()
gc = IG.group_codes(pairs)
fd['_업소'] = fd.lcns_no.where(fd.src == '15058303', fd.bssh_nm.fillna('') + '|' + fd.area1.fillna('') + '|' + fd.area2.fillna(''))
rows = []
for src, d, ukey, uname in [('I0580', out, out.lcns_no, 'LCNS_NO'),
                            ('15058303', fd[fd.src == '15058303'], fd[fd.src == '15058303'].lcns_no, 'LCNS_NO'),
                            ('15033311', fd[fd.src == '15033311'], fd[fd.src == '15033311']._업소, '업체명+area1+area2')]:
    dd = d.assign(_u=ukey.values, _i=d.induty_nm.fillna('(빈 값)').values)
    for ind, g in dd.groupby('_i', dropna=False):
        grpn, why = IG.group_of(None if ind == '(빈 값)' else ind)
        codes = gc[grpn]
        rows.append({'자료': src, '업종(원값)': ind, '행 수': len(g), '업소 수': g._u.nunique(), '업소 세는 기준': uname,
                     '업종묶음': grpn, '묶음 우리 2단계 코드 수': len(codes),
                     '묶음 우리 2단계 코드 목록': ';'.join(c for c, _ in codes),
                     '묶음 우리 2단계 이름 목록': ';'.join(n for _, n in codes), '근거': why})
mp = pd.DataFrame(rows).sort_values(['자료', '행 수'], ascending=[True, False])
mp.to_csv(MID + r'\haccp_업종_대응.csv', index=False, encoding='utf-8-sig')
P('\n== 업종 대응 (자료별)')
for src, d in mp.groupby('자료', sort=False):
    tot_r = d['행 수'].sum()
    P(f'-- {src}: 업종 {len(d)}개, 행 {tot_r:,}')
    s = d.groupby('업종묶음')['행 수'].sum().sort_values(ascending=False)
    P('   묶음별 행:', {k2: f'{v:,} ({v / tot_r * 100:.2f}%)' for k2, v in s.items()})
for src, d, ukey in [('I0580', out, out.lcns_no), ('15058303', fd[fd.src == '15058303'], fd[fd.src == '15058303'].lcns_no),
                     ('15033311', fd[fd.src == '15033311'], fd[fd.src == '15033311']._업소)]:
    gg = d.induty_nm.map(lambda v: IG.group_of(v)[0])
    has = gg != IG.NONE
    withcode = gg.isin([g for g, c in gc.items() if c])
    u = pd.DataFrame({'u': ukey.values, 'has': has.values, 'wc': withcode.values}).groupby('u').agg(has=('has', 'any'), wc=('wc', 'any'))
    P(f'{src}: 묶음(대응 없음 아님) 행 {int(has.sum()):,}/{len(d):,} ({has.mean()*100:.2f}%), 코드 있는 묶음 행 {int(withcode.sum()):,} ({withcode.mean()*100:.2f}%)'
      f' / 업소 {len(u):,} 중 묶음 있음 {int(u.has.sum()):,} ({u.has.mean()*100:.2f}%), 코드 있는 묶음 {int(u.wc.sum()):,} ({u.wc.mean()*100:.2f}%)')
P('\n묶음별 우리 2단계 코드:')
for g, c in gc.items():
    P(f'  {g}: {len(c)}개', [f'{a}({b})' for a, b in c])
covered = {c for v in gc.values() for c, _ in v}
P('어느 묶음에도 들지 않은 우리 2단계 코드:', [f'{a}({b})' for a, b in pairs.itertuples(index=False, name=None) if a not in covered])
P('대응 없음 업종 목록:')
P(mp[mp.업종묶음 == IG.NONE][['자료', '업종(원값)', '행 수', '업소 수', '근거']].to_string(index=False))

open(MID + r'\_haccp_I0580_만들기_요약.txt', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
