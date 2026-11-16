# HACCP 유효 구간표: I0580 사건표 → 데이터셋\외부_지표\haccp_유효구간.parquet (확정 규칙 2026-10-19, 보고서 '확정 규칙(10-19)' 절)
#  1) 번호 시작 S = 같은 HACCP_APPN_NO 의 15058303 HACCP_APPN_DT · 15033311 issuedate · I0580 HACCP_APPN_DT 중 가장 이른 날
#     (셋 다 없으면 번호 앞 4자리 연도의 12월 31일. 가장 이른 날의 연도가 번호 연도보다 늦어도 가장 이른 날 그대로)
#  2) 행 끝 = min(CRTFC_ENDDT, CRTFC_RETN_DT, ASGN_CANCL_DT(CRTFC_ENDDT 와 다를 때만)); 번호 끝 E = 같은 번호 행 끝 중 가장 늦은 날. CLSBIZ_DT 는 끝에 안 넣음
#  3) 사용시작일 = 2017-08-03 (칸으로 둠)
#  6) 세는 단위 = 지정번호. 행마다 [행 시작, 행 끝] + 그 행 sgg_code. 행 시작 = 그 행 HACCP_APPN_DT(번호에서 가장 이른 I0580 지정일 행은 S).
#     번호는 [S, E] 동안 끊기지 않고 유효. 날 t 의 주소 = t 를 덮는 행 중 가장 늦게 시작한 행(덮는 행이 없으면 t 전에 시작한 행 중 가장 늦게 시작한 행).
#     같은 날 시작한 행끼리(이전한 업소: 옛 주소 행은 폐업, 새 주소 행은 정상)는 CLSBIZ_DT 를 '주소가 바뀐 날'로만 써서 고름(끝에는 안 씀):
#       t 에 아직 문을 닫지 않은(CLSBIZ_DT ≥ t) 행 중 가장 먼저 닫는 행 → 없으면 CLSBIZ_DT 없는 행('정상' 먼저, 그다음 원자료 순서) → 다 닫았으면 가장 늦게 닫은 행
#     결과는 겹치지 않는 구간(한 번호의 구간끼리 겹치지 않음).
# 실행: <리스테리아ML\.venv-exp2>\python.exe -I haccp_interval_build.py
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
import haccp_load
import haccp_induty_groups as IG

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data'
SRC = BASE + r'\데이터셋\외부_지표\haccp_인증_I0580.parquet'
OUT = BASE + r'\데이터셋\외부_지표\haccp_유효구간.parquet'
MID = BASE + r'\중간\외부지표'
if os.path.exists(OUT) and not (len(sys.argv) > 1 and sys.argv[1] == 'overwrite'):
    raise SystemExit(f'이미 있음(덮지 않음): {OUT}')
TODAY = pd.Timestamp('2026-10-19')
USE_START = datetime.date(2017, 8, 3)
ONE = pd.Timedelta(days=1)
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)

d = pq.read_table(SRC).to_pandas()
for c in ['appn_de', 'crtfc_end_de', 'asgn_cancl_de', 'crtfc_retn_de', 'clsbiz_de']:
    d[c] = pd.to_datetime(d[c])
d['no'] = d.haccp_no.str.strip()
d['row_id'] = np.arange(len(d))

# ---------- 1) 번호 시작 ----------
m3, _, _ = haccp_load.load('mfds'); m3['no'] = m3.HACCP_APPN_NO.fillna('').str.strip()
m3['d'] = pd.to_datetime(m3.HACCP_APPN_DT, format='%Y%m%d', errors='coerce')
c3, _, _ = haccp_load.load('cert'); c3['no'] = c3.appointno.fillna('').str.strip()
c3['d'] = pd.to_datetime(c3.issuedate, format='%Y-%m-%d', errors='coerce')
d58 = m3[m3.no != ''].groupby('no').d.min()
d33 = c3[c3.no != ''].groupby('no').d.min()
num = d.groupby('no').agg(lcns=('lcns_no', 'first'), dI=('appn_de', 'min'), n_rows=('row_id', 'size'))
num['d58'] = num.index.map(d58); num['d33'] = num.index.map(d33)
num['noyr'] = pd.to_numeric(num.index.str[:4], errors='coerce')
num['S'] = num[['d58', 'd33', 'dI']].min(axis=1)
SRCNAME = {'d58': '15058303 HACCP_APPN_DT', 'd33': '15033311 issuedate', 'dI': 'I0580 HACCP_APPN_DT'}
def s_reason(r):
    if pd.isna(r.S):
        return None
    return ' = '.join(SRCNAME[k] for k in ['d58', 'd33', 'dI'] if pd.notna(r[k]) and r[k] == r.S)
num['S_why'] = num.apply(s_reason, axis=1)
fb = num.S.isna()
num.loc[fb, 'S'] = pd.to_datetime(num.loc[fb, 'noyr'].astype('Int64').astype(str) + '-12-31', errors='coerce')
num.loc[fb, 'S_why'] = '번호 앞 4자리 연도의 12월 31일(세 자료에 날짜 없음)'
P('# HACCP 유효 구간표 만들기', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
P('\n## 규칙 1 시작')
P('번호', len(num), '/ 시작을 정한 자료(같은 날이면 함께 적음):'); P(num.S_why.value_counts().to_string())
P('셋 다 없어 번호 연도 12-31 을 쓴 번호:', int(fb.sum()))
P('시작일 연도 > 번호 앞 4자리 연도 (가장 이른 날 그대로 씀):', int((num.S.dt.year > num.noyr).sum()), '/ 시작일 연도 < 번호 연도:', int((num.S.dt.year < num.noyr).sum()))
# 15058303 날짜가 다른 업소 번호에서 왔는지(번호만으로 맞댐)
m3l = m3[m3.no != ''].groupby('no').LCNS_NO.apply(set)
use58 = num[num.S_why.str.contains('15058303', na=False)]
diffl = sum(1 for no, r in use58.iterrows() if r.lcns not in m3l.get(no, set()))
P('시작을 15058303 이 정한 번호', len(use58), '중 15058303 쪽 LCNS_NO 에 I0580 LCNS_NO 가 없는 번호:', diffl)

# ---------- 2) 행 끝 ----------
cancl_real = d.asgn_cancl_de.where(d.asgn_cancl_de != d.crtfc_end_de)
ends = pd.concat([d.crtfc_end_de.rename('CRTFC_ENDDT'), d.crtfc_retn_de.rename('CRTFC_RETN_DT'), cancl_real.rename('ASGN_CANCL_DT(CRTFC_ENDDT 와 다름)')], axis=1)
d['re'] = ends.min(axis=1)
d['re_why'] = [' = '.join(c for c in ends.columns if pd.notna(v[c]) and v[c] == e) for (_, v), e in zip(ends.iterrows(), d.re)]
# 행 시작
d['dI'] = d.no.map(num.dI); d['S'] = d.no.map(num.S)
d['rs'] = d.appn_de.where(d.appn_de != d.dI, d.S)
d['rs_why'] = np.where(d.appn_de == d.dI, d.no.map(num.S_why), '이 행의 I0580 HACCP_APPN_DT')
d['valid'] = d.re >= d.rs
num['E'] = d.groupby('no').re.max()
P('\n## 규칙 2 끝')
P('행 끝을 정한 칼럼:'); P(d.re_why.value_counts().to_string())
P('행 끝 < 행 시작인 행(그 행만으로는 구간 없음):', int((~d.valid).sum()))
P('번호 끝 E < 번호 시작 S 인 번호(구간 없음):', int((num.E < num.S).sum()))

# ---------- 6) 구간 ----------
pairs = IG.our_pairs(); gc = IG.group_codes(pairs)
codes_str = {g: (','.join(c for c, _ in v) if v else None) for g, v in gc.items()}
segs = []
stat = {'행 하나': 0, '여러 행': 0, '여러 행·시군구 하나': 0}
howcnt = {}

def pick(cands, t):
    """같은 날 시작한 후보 중 하나: CLSBIZ_DT 를 주소 바뀐 날로만 씀."""
    if len(cands) == 1:
        return cands[0], None
    still = [r for r in cands if pd.notna(r.clsbiz_de) and r.clsbiz_de >= t]
    if still:
        return min(still, key=lambda r: (r.clsbiz_de, r.row_id)), '같은 날 시작: 아직 문 닫기 전 행 중 가장 먼저 닫는 행(CLSBIZ_DT)'
    openr = [r for r in cands if pd.isna(r.clsbiz_de)]
    if openr:
        return min(openr, key=lambda r: (r.clsbiz_dvs_nm != '정상', r.row_id)), '같은 날 시작: CLSBIZ_DT 없는 행'
    return max(cands, key=lambda r: (r.clsbiz_de, -r.row_id)), '같은 날 시작: 모두 닫음 — 가장 늦게 닫은 행'

for no, g in d.groupby('no', sort=True):
    S, E = num.at[no, 'S'], num.at[no, 'E']
    if pd.isna(E) or E < S:
        continue
    rows = list(g.itertuples(index=False))
    E_rows = [r for r in rows if r.re == E]
    E_why = ' / '.join(sorted({r.re_why for r in E_rows}))
    if E >= TODAY:
        E_why += ' (받은 날 2026-10-19 뒤 — 받은 날 현재 유효)'
    if len(rows) == 1:
        r = rows[0]; stat['행 하나'] += 1
        segs.append((no, r, S, E, num.at[no, 'S_why'], E_why, '행 하나'))
        continue
    stat['여러 행'] += 1
    if g.sgg_code.nunique(dropna=False) == 1:
        stat['여러 행·시군구 하나'] += 1
    bset = {S, E + ONE}
    for r in rows:
        for x in (r.rs, r.re + ONE):
            if S <= x <= E + ONE:
                bset.add(x)
        if pd.notna(r.clsbiz_de) and S <= r.clsbiz_de + ONE <= E + ONE:
            bset.add(r.clsbiz_de + ONE)
    bs = sorted(bset)
    chosen = []
    for a, b in zip(bs[:-1], bs[1:]):
        cover = [r for r in rows if r.valid and r.rs <= a <= r.re]
        how = '덮는 행'
        if not cover:
            cover = [r for r in rows if r.valid and r.rs <= a]; how = '빈 기간: 그 전에 시작한 행'
        if not cover:
            cover = [r for r in rows if r.rs <= a]; how = '빈 기간: 유효한 행 없음 — 그 전에 시작한 행'
        top = max(r.rs for r in cover)
        c2 = [r for r in cover if r.rs == top]
        r, tie = pick(c2, a)
        if len(cover) > 1 and len({x.rs for x in cover}) > 1:
            how += ' 중 가장 늦게 시작한 행'
        if tie:
            how += ' / ' + tie
        chosen.append([a, b - ONE, r, how])
    # 같은 행이 이어지면 합침
    merged = []
    for a, e, r, how in chosen:
        if merged and merged[-1][2].row_id == r.row_id and merged[-1][1] + ONE == a:
            merged[-1][1] = e
            if how not in merged[-1][3]:
                merged[-1][3] += ' ; ' + how
        else:
            merged.append([a, e, r, how])
    for i, (a, e, r, how) in enumerate(merged):
        if i == 0:
            sw = num.at[no, 'S_why']
        else:
            pr = merged[i - 1][2]
            same = (pr.sgg_code == r.sgg_code)
            head = '다음 행으로 넘어감(시군구 같음)' if same else '주소 바뀜(시군구 다름)'
            if a == r.rs:
                sw = head + ': 이 행의 시작(HACCP_APPN_DT, 더 늦게 시작한 행이 이어받음)'
            elif pd.notna(pr.clsbiz_de) and a == pr.clsbiz_de + ONE:
                sw = head + ': 앞 구간 행의 CLSBIZ_DT 다음 날'
            elif a == pr.re + ONE:
                sw = head + ': 앞 구간 행의 끝 다음 날'
            else:
                sw = head + ': 그 밖'
        if i == len(merged) - 1:
            ew = E_why
        else:
            ew = '이 행 끝(' + r.re_why + ')' if e == r.re else '다음 구간으로 넘어감(같은 번호 다른 행)'
        segs.append((no, r, a, e, sw, ew, how))
        howcnt[how] = howcnt.get(how, 0) + 1

P('\n## 규칙 6 구간')
P('번호:', stat)
seg = pd.DataFrame([{
    'HACCP_APPN_NO': r.haccp_no, 'LCNS_NO': r.lcns_no, 'sgg_code': r.sgg_code, 'sgg_nm': r.sgg_nm,
    '업종': r.induty_nm, '업종묶음': r.업종묶음, '우리_2단계_코드목록': codes_str[r.업종묶음],
    '시작일': a.date(), '시작일_근거': sw, '끝일': e.date(), '끝일_근거': ew, '사용시작일': USE_START,
    '번호_시작일': num.at[no, 'S'].date(), '번호_끝일': num.at[no, 'E'].date(), '주소_고른_방법': how,
    'raw_start': int(r.raw_start), 'raw_pos': int(r.raw_pos), '_no': no, '_row': r.row_id}
    for (no, r, a, e, sw, ew, how) in segs])
seg = seg.sort_values(['_no', '시작일']).reset_index(drop=True)
seg['구간_순번'] = seg.groupby('_no').cumcount() + 1
seg['구간_수'] = seg.groupby('_no')._no.transform('size')
P('구간 행', len(seg), '/ 번호', seg._no.nunique(), '/ 구간 2개 이상인 번호', int((seg.groupby('_no').size() > 1).sum()))
P('여러 행 번호의 주소 고른 방법(구간 수):'); P(pd.Series(howcnt).sort_values(ascending=False).to_string())
# 겹침 검사: 같은 번호 구간끼리 겹치지 않는지, 번호 구간 합이 [S,E] 와 같은지
bad = 0
for no, g in seg.groupby('_no'):
    s = g.sort_values('시작일')
    if (pd.to_datetime(s.시작일.iloc[1:].values) != pd.to_datetime(s.끝일.iloc[:-1].values) + ONE).any():
        bad += 1
P('같은 번호 구간이 끊기거나 겹치는 번호(0 이어야 함):', bad)
# 원 행끼리의 겹침 수(보고용): 같은 번호에서 시군구가 다른 두 행의 [행 시작, 행 끝] 이 겹치는 번호
ov = 0; ov_same = 0
for no, g in d[d.no.duplicated(keep=False)].groupby('no'):
    rr = list(g.itertuples(index=False))
    hit = hit_same = False
    for i in range(len(rr)):
        for j in range(i + 1, len(rr)):
            a, b = rr[i], rr[j]
            if a.valid and b.valid and a.rs <= b.re and b.rs <= a.re and a.sgg_code != b.sgg_code:
                hit = True
                if a.rs == b.rs:
                    hit_same = True
    ov += hit; ov_same += hit_same
P('원 행 구간이 겹치고 시군구가 다른 번호:', ov, '/ 그중 같은 날 시작한 행끼리 겹치는 번호:', ov_same)
unused = d[~d.row_id.isin(set(seg._row))]
P('구간에 한 번도 안 쓰인 I0580 행:', len(unused), '/ 그 까닭 — 번호 구간 없음(E<S):', int(unused.no.isin(num[num.E < num.S].index).sum()))
unused_out = unused[['haccp_no', 'lcns_no', 'prdlst_nm', 'addr_raw', 'sgg_code', 'appn_de', 'crtfc_end_de', 'crtfc_retn_de', 'asgn_cancl_de', 'clsbiz_de', 'clsbiz_dvs_nm', 'raw_start', 'raw_pos']].copy()
unused_out['안_쓰인_까닭'] = np.where(unused.no.isin(num[num.E < num.S].index), '번호 끝 < 번호 시작(구간 없음)', '같은 번호의 다른 행이 그 기간의 주소로 골림(같은 시군구거나 더 늦게 시작·더 오래 엶)')
unused_out.to_csv(MID + r'\_haccp_유효구간_안쓴행.csv', index=False, encoding='utf-8-sig')

# ---------- 저장 ----------
ORDER = ['HACCP_APPN_NO', 'LCNS_NO', 'sgg_code', 'sgg_nm', '업종', '업종묶음', '우리_2단계_코드목록', '시작일', '시작일_근거', '끝일', '끝일_근거',
         '사용시작일', '번호_시작일', '번호_끝일', '구간_순번', '구간_수', '주소_고른_방법', 'raw_start', 'raw_pos']
arrs = []
for c in ORDER:
    if c in ('시작일', '끝일', '사용시작일', '번호_시작일', '번호_끝일'):
        arrs.append(pa.array(seg[c].tolist(), type=pa.date32()))
    elif c in ('구간_순번', '구간_수', 'raw_pos'):
        arrs.append(pa.array(seg[c].astype(int).tolist(), type=pa.int16()))
    elif c == 'raw_start':
        arrs.append(pa.array(seg[c].astype(int).tolist(), type=pa.int32()))
    else:
        arrs.append(pa.array([None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in seg[c]], type=pa.string()))
tbl = pa.Table.from_arrays(arrs, names=ORDER)
pq.write_table(tbl, OUT)
P('\n저장:', OUT, f'{tbl.num_rows:,}행 × {tbl.num_columns}칼럼')
P('업종묶음별 구간 행:', seg.업종묶음.value_counts().to_dict())
P('sgg_code 빈 구간 행:', int(seg.sgg_code.isna().sum()))

# ---------- 연도 말 유효 수 ----------
s0 = pd.to_datetime(seg.시작일); e0 = pd.to_datetime(seg.끝일)
rows = []
dates = [pd.Timestamp('2017-08-03')] + [pd.Timestamp(f'{y}-12-31') for y in range(2009, 2026)] + [TODAY]
for T in sorted(set(dates)):
    m = (s0 <= T) & (e0 >= T)
    v = seg[m]
    food = v[v.업종묶음 != IG.NONE]
    coded = v[v.업종묶음.isin([g for g, c in gc.items() if c])]
    rows.append({'기준일': T.date(), '번호(모든 업종)': v._no.nunique(), '업소(모든 업종)': v.LCNS_NO.nunique(),
                 '업소(대응 없음 뺌)': food.LCNS_NO.nunique(), '업소(코드 있는 묶음)': coded.LCNS_NO.nunique(),
                 '번호(코드 있는 묶음)': coded._no.nunique(), '사용시작일 이후': T.date() >= USE_START})
ref = pd.DataFrame(rows)
ref.to_csv(MID + r'\_haccp_유효구간_연말유효수.csv', index=False, encoding='utf-8-sig')
P('\n## 연도 말 유효 수(구간표로 셈)'); P(ref.to_string(index=False))
open(MID + r'\_haccp_유효구간_만들기_요약.txt', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
