# HACCP 사건표 만들기: 원자료(쪽별 JSON) 두 자료 → 데이터셋/외부_지표/haccp_인증.parquet
#   한 행 = 원자료 한 행(한 업소·품목의 인증/지정 기록). 행을 지우거나 합치지 않음(완전 중복은 dup_in_src 로 표시만).
#   시군구: 시군구_대표좌표.parquet 의 256개 이름과 같은 것만 잇고, 옛 이름은 아래 대응표(OLD)로, 나뉜 곳·시 단위만 있는 곳은 비워 둠.
#   품목: 시료_시군구.parquet 의 train_prdlst_nm / train_prdlst_nm_2 와 글자가 정확히 같은 것만 잇고, 나머지는 후보표로.
#   누적 수·비율 계산은 하지 않음.
# 실행: <리스테리아ML\.venv-exp2>\python.exe -I haccp_build.py
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from haccp_load import load

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data'
SGG = BASE + r'\데이터셋\시군구_대표좌표.parquet'
SAMPLE = BASE + r'\데이터셋\시료_시군구.parquet'
OUT = BASE + r'\데이터셋\외부_지표\haccp_인증.parquet'
MID = BASE + r'\중간\외부지표'
if len(sys.argv) > 1 and sys.argv[1] == 'test':  # 시험 실행: 결과를 시험 폴더에만
    MID = BASE + r'\중간\외부지표\_haccp_시험'; OUT = MID + r'\haccp_인증_시험.parquet'; os.makedirs(MID, exist_ok=True)

# ---------------- 시군구 ----------------
sgg = pd.read_parquet(SGG, columns=['sgg_code', 'sgg_nm'])
NAME2CODE = dict(zip(sgg.sgg_nm, sgg.sgg_code))
CODE2NAME = dict(zip(sgg.sgg_code, sgg.sgg_nm))
# 일반구가 있는 시(256개 이름 중 '시도 ○○시 ○○구' 꼴의 가운데 낱말)
GU_CITY = sorted({' '.join(n.split()[:2]) for n in sgg.sgg_nm if len(n.split()) == 3})
SIDO_NOW = sorted({n.split()[0] for n in sgg.sgg_nm})
SIDO_ALIAS = {  # 원문 첫 낱말 → 지금 시도 이름(256개 이름에 쓰인 것)
    '서울': '서울특별시', '서울시': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시',
    '대전': '대전광역시', '울산': '울산광역시', '세종': '세종특별자치시', '세종시': '세종특별자치시', '경기': '경기도',
    '강원': '강원특별자치도', '강원도': '강원특별자치도', '충북': '충청북도', '충남': '충청남도',
    '전북': '전북특별자치도', '전라북도': '전북특별자치도', '경북': '경상북도', '경남': '경상남도',
    '제주': '제주특별자치도', '제주도': '제주특별자치도',
    '광주': '전남광주통합특별시', '광주광역시': '전남광주통합특별시', '전남': '전남광주통합특별시', '전라남도': '전남광주통합특별시',
}
# 옛 이름 → 지금 코드 (하나로 이어지는 경우만)
OLD = {
    '인천광역시 남구': ('28177', '옛 이름: 인천광역시 남구 → 미추홀구(2018-07-01 이름 바뀜)'),
    '인천광역시 동구': ('28125', '옛 이름: 인천광역시 동구 → 제물포구(인천 행정체제 개편, 동구 전체가 제물포구로 들어감)'),
    '경상북도 군위군': ('27720', '옛 이름: 경상북도 군위군 → 대구광역시 군위군(2023-07-01 편입)'),
    '충청남도 연기군': ('36110', '옛 이름: 충청남도 연기군 → 세종특별자치시(2012-07-01)'),
    '경기도 여주군': ('41670', '옛 이름: 여주군 → 여주시(2013 시 승격)'),
    '충청남도 당진군': ('44270', '옛 이름: 당진군 → 당진시(2012 시 승격)'),
    '경상남도 진해시': ('48129', '옛 이름: 진해시 → 창원시 진해구(2010 통합)'),
}
# 옛 이름인데 지금 여러 시군구로 나뉜 곳 → 정하지 않음
SPLIT = {
    '인천광역시 중구': '나뉨: 인천광역시 중구 → 제물포구(내륙)·영종구(영종도) — 정하지 않음',
    '인천광역시 서구': '나뉨: 인천광역시 서구 → 서해구·검단구 — 정하지 않음',
    '충청북도 청원군': '나뉨: 청원군 → 청주시 4개 구(2014 통합) — 정하지 않음',
    '경상남도 마산시': '나뉨: 마산시 → 창원시 마산합포구·마산회원구(2010 통합) — 정하지 않음',
    '경상남도 창원시 마산구': '나뉨 — 정하지 않음',
    '경기도 고양시 일산구': '나뉨: 일산구 → 일산동구·일산서구(2005) — 정하지 않음',
}

def map_addr(s):
    """원문 주소 → (sgg_code, sgg_시단위, 대응방법). 대응 못 하면 code=None."""
    if s is None or not str(s).strip():
        return None, None, '주소 없음'
    toks = str(s).split()
    first = toks[0]
    notes = []
    if first in SIDO_NOW:
        sido = first
    elif first in SIDO_ALIAS:
        sido = SIDO_ALIAS[first]
        notes.append(f'시도 이름 맞춤({first}→{sido})')
    else:
        return None, None, f'대응 못 함: 첫 낱말 "{first}" 이 시도 이름이 아님'
    rest = toks[1:]
    # 1) 지금 이름과 같은지: 앞 3낱말 → 2낱말 → 1낱말(세종)
    for k in (2, 1, 0):
        if len(rest) < k:
            continue
        cand = ' '.join([sido] + rest[:k])
        if cand in NAME2CODE:
            how = '이름 같음' if len(toks) == k + 1 else f'주소 앞 {k+1}낱말이 이름과 같음(뒤 낱말 버림)'
            return NAME2CODE[cand], None, '; '.join(notes + [how])
    # 2) 옛 이름(하나로 이어짐) / 나뉜 곳
    for k in (2, 1):
        if len(rest) < k:
            continue
        cand = ' '.join([sido] + rest[:k])
        if cand in OLD:
            code, why = OLD[cand]
            return code, None, '; '.join(notes + [why])
        if cand in SPLIT:
            return None, None, '; '.join(notes + [SPLIT[cand]])
    # 3) 일반구가 있는 시인데 구가 없음
    if rest and f'{sido} {rest[0]}' in GU_CITY:
        city = f'{sido} {rest[0]}'
        extra = '' if len(rest) == 1 else f'(뒤 낱말 "{" ".join(rest[1:])}" 은 구 이름이 아님)'
        return None, city, '; '.join(notes + [f'시 단위만 있음 — 일반구 정하지 않음{extra}'])
    if not rest:
        return None, None, '; '.join(notes + ['대응 못 함: 시도만 있음'])
    return None, None, '; '.join(notes + [f'대응 못 함: {sido} 에 "{rest[0]}" 없음'])

# ---------------- 품목 ----------------
pairs = (pd.read_parquet(SAMPLE, columns=['train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2'])
         .astype('string').drop_duplicates())
pairs = pairs[pairs.train_prdlst_code.notna()]
N3 = {}
for r in pairs.itertuples(index=False):
    if pd.notna(r.train_prdlst_code_2):
        N3.setdefault(r.train_prdlst_nm_2, set()).add((r.train_prdlst_code, r.train_prdlst_nm, r.train_prdlst_code_2, r.train_prdlst_nm_2))
N2 = {}
for r in pairs[['train_prdlst_code', 'train_prdlst_nm']].drop_duplicates().itertuples(index=False):
    N2.setdefault(r.train_prdlst_nm, set()).add((r.train_prdlst_code, r.train_prdlst_nm))
GENERIC = {'기타'}  # 정액,수정란,종란 > '기타' 같은 이름은 뜻이 달라 자동으로 잇지 않음

def map_item(nm):
    """정확히 같은 이름만. → (code, nm, code2, nm2, 방법)"""
    if nm is None or not str(nm).strip() or str(nm).strip() == '-':
        return None, None, None, None, '품목명 없음(빈 칸 또는 "-")'
    if nm in GENERIC:
        return None, None, None, None, '자동 대응 안 함: 뜻이 넓은 이름'
    if nm in N3:
        c = N3[nm]
        if len(c) == 1:
            a, b, x, y = next(iter(c))
            return a, b, x, y, '3단계 이름과 정확히 같음'
        return None, None, None, None, '3단계 이름이 여러 코드에 있음 — 정하지 않음'
    if nm in N2:
        c = N2[nm]
        if len(c) == 1:
            a, b = next(iter(c))
            return a, b, None, None, '2단계 이름과 정확히 같음(3단계는 비움)'
        return None, None, None, None, '2단계 이름이 여러 코드에 있음 — 정하지 않음'
    return None, None, None, None, '정확히 같은 이름 없음 — 후보표 참고'

def norm(s):
    return re.sub(r'[\s\.\·ㆍ‧･・,()\[\]\-/]', '', str(s))
STOP = {'기타', '식품', '가공품', '제품', '가공', '가공식품', '식품류'}
OURS = ([('3', a, b, x, y) for v in N3.values() for (a, b, x, y) in v] +
        [('2', a, b, None, None) for v in N2.values() for (a, b) in v])

def candidates(nm):
    h = norm(nm); hc = h.rstrip('류')
    out = []
    for lv, a, b, x, y in OURS:
        our = y if lv == '3' else b
        o = norm(our); oc = o.rstrip('류')
        if h == o:
            why, rank = '띄어쓰기·문장부호만 다름', 0
        elif hc == oc and len(oc) >= 2:
            why, rank = "끝의 '류' 자만 다름", 1
        elif len(oc) >= 2 and oc not in STOP and oc in h:
            why, rank = f"HACCP 품목명 안에 우리 이름 '{our}' 의 글자가 들어 있음", 2
        elif len(hc) >= 2 and hc not in STOP and hc in o:
            why, rank = f"우리 이름 '{our}' 안에 HACCP 품목명 글자가 들어 있음", 3
        else:
            continue
        out.append((rank, -len(oc), lv, a, b, x, y, why))
    out.sort(key=lambda t: (t[0], t[1], t[2] != '3'))
    seen = set(); res = []
    for t in out:
        k = (t[3], t[5])
        if k in seen:
            continue
        seen.add(k); res.append(t)
    return res[:5]

# ---------------- 날짜 ----------------
def to_date(s):
    if s is None or not str(s).strip():
        return None
    s = str(s).strip()
    try:
        if re.fullmatch(r'\d{8}', s):
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
            return datetime.date.fromisoformat(s)
    except ValueError:
        return None
    return None

# ---------------- 두 자료를 같은 꼴로 ----------------
def build_src(svc):
    df, files, totals = load(svc)
    raw_cols = [c for c in df.columns if not c.startswith('_')]
    dup = df.duplicated(raw_cols, keep='first')
    n = len(df)
    if svc == 'mfds':
        out = pd.DataFrame({
            'src': '15058303',
            'lcns_no': df.LCNS_NO, 'haccp_no': df.HACCP_APPN_NO, 'bssh_nm': df.BSSH_NM,
            'induty_nm': df.INDUTY_CD_NM, 'cert_companykind_nm': None, 'cert_product_gb': None,
            'prdlst_nm': df.PRDLST_NM, 'cert_businessitem_cd': None, 'cert_businesstype_cd': None, 'cert_companykind_cd': None,
            'appn_de_raw': df.HACCP_APPN_DT, 'end_de_raw': None,
            'ssize_yn': df.SSIZESCALE_BSSH_YN, 'addr_raw': df.SITE_ADDR, 'area1': None, 'area2': None,
        })
        addr_key = df.SITE_ADDR.fillna('')
    else:
        out = pd.DataFrame({
            'src': '15033311',
            'lcns_no': None, 'haccp_no': df.appointno, 'bssh_nm': df.company,
            'induty_nm': df.businesstypeNm, 'cert_companykind_nm': df.companykindNm, 'cert_product_gb': df.productGb,
            'prdlst_nm': df.businessitemNm, 'cert_businessitem_cd': df.businessitem, 'cert_businesstype_cd': df.businesstype, 'cert_companykind_cd': df.companykind,
            'appn_de_raw': df.issuedate, 'end_de_raw': df.issueenddate,
            'ssize_yn': None, 'addr_raw': None, 'area1': df.area1, 'area2': df.area2,
        })
        addr_key = (df.area1.fillna('') + ' ' + df.area2.fillna('')).str.strip()
    out['addr_key'] = addr_key
    out['dup_in_src'] = dup.values
    out['raw_page'] = df._page.astype('int16').values
    out['raw_pos'] = df._pos.astype('int16').values
    return out

parts = [build_src('mfds'), build_src('cert')]
df = pd.concat(parts, ignore_index=True)

# 시군구
amap = {a: map_addr(a) for a in df.addr_key.drop_duplicates()}
df['sgg_code'] = df.addr_key.map(lambda a: amap[a][0])
df['sgg_nm'] = df.sgg_code.map(CODE2NAME)
df['sgg_시단위'] = df.addr_key.map(lambda a: amap[a][1])
df['sgg_대응'] = df.addr_key.map(lambda a: amap[a][2])

# 품목
df['_pkey'] = df.prdlst_nm.fillna('')  # 빈 값(None)은 '' 로 찾음(값 자체는 prdlst_nm 에 그대로)
imap = {p: map_item(p) for p in df._pkey.drop_duplicates()}
for i, c in enumerate(['train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2', 'prdlst_대응']):
    df[c] = df._pkey.map(lambda p: imap[p][i])

# 날짜
df['appn_de'] = df.appn_de_raw.map(to_date)
df['end_de'] = df.end_de_raw.map(to_date)
bad_appn = int((df.appn_de.isna() & df.appn_de_raw.notna()).sum())
bad_end = int((df.end_de.isna() & df.end_de_raw.notna()).sum())

cols_str = ['src', 'lcns_no', 'haccp_no', 'bssh_nm', 'induty_nm', 'cert_companykind_nm', 'cert_product_gb', 'prdlst_nm',
            'cert_businessitem_cd', 'cert_businesstype_cd', 'cert_companykind_cd', 'ssize_yn', 'addr_raw', 'area1', 'area2',
            'sgg_code', 'sgg_nm', 'sgg_시단위', 'sgg_대응', 'train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2', 'prdlst_대응']
order = ['src', 'lcns_no', 'haccp_no', 'bssh_nm', 'induty_nm', 'cert_companykind_nm', 'cert_product_gb', 'prdlst_nm',
         'cert_businessitem_cd', 'cert_businesstype_cd', 'cert_companykind_cd',
         'appn_de', 'end_de', 'appn_de_raw', 'end_de_raw', 'ssize_yn', 'addr_raw', 'area1', 'area2',
         'sgg_code', 'sgg_nm', 'sgg_시단위', 'sgg_대응',
         'train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2', 'prdlst_대응',
         'dup_in_src', 'raw_page', 'raw_pos']
arrays = []
fields = []
for c in order:
    if c in ('appn_de', 'end_de'):
        arrays.append(pa.array(list(df[c]), type=pa.date32()))
    elif c == 'dup_in_src':
        arrays.append(pa.array(df[c].astype(bool).tolist(), type=pa.bool_()))
    elif c in ('raw_page', 'raw_pos'):
        arrays.append(pa.array(df[c].astype('int16').tolist(), type=pa.int16()))
    else:
        v = df[c].astype(object).where(df[c].notna(), None).tolist() if c in df else [None] * len(df)
        arrays.append(pa.array([None if x is None else str(x) for x in v], type=pa.string()))
    fields.append(c)
tbl = pa.Table.from_arrays(arrays, names=fields)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(tbl, OUT)
print('저장:', OUT, tbl.num_rows, '행', tbl.num_columns, '칼럼')
print('날짜로 못 읽은 값: appn_de', bad_appn, '/ end_de', bad_end)

# ---------------- 대응표·후보표 ----------------
a = df.groupby(['src', 'addr_key'], dropna=False).size().rename('행 수').reset_index()
a['sgg_code'] = a.addr_key.map(lambda k: amap[k][0]); a['sgg_nm'] = a.sgg_code.map(CODE2NAME)
a['sgg_시단위'] = a.addr_key.map(lambda k: amap[k][1]); a['대응 방법'] = a.addr_key.map(lambda k: amap[k][2])
a = a.rename(columns={'src': '자료', 'addr_key': '원문 주소(15058303=SITE_ADDR, 15033311=area1+area2)'})
a.sort_values(['자료', '행 수'], ascending=[True, False]).to_csv(MID + r'\haccp_시군구_대응표.csv', index=False, encoding='utf-8-sig')

ukey = df.lcns_no.where(df.src == '15058303', df.bssh_nm.fillna('') + '|' + df.area1.fillna('') + '|' + df.area2.fillna(''))
df['_업소'] = ukey
rows = []
g = df.groupby(['src', '_pkey'], dropna=False)
for (src, nm), d in g:
    m = imap[nm]
    if m[0] is not None:
        continue
    base = {'자료': src, 'HACCP 품목명': nm, '건수(행)': len(d), '업소 수': d._업소.nunique(), '자동 대응 안 된 까닭': m[4]}
    cs = [] if (nm is None or m[4].startswith('품목명 없음')) else candidates(nm)
    if not cs:
        rows.append({**base, '후보 순위': None, '후보 코드(2단계)': None, '후보 이름(2단계)': None, '후보 코드(3단계)': None, '후보 이름(3단계)': None, '근거': '후보 못 찾음' if not m[4].startswith('품목명 없음') else '품목명 없음'})
    for i, t in enumerate(cs, 1):
        rows.append({**base, '후보 순위': i, '후보 코드(2단계)': t[3], '후보 이름(2단계)': t[4], '후보 코드(3단계)': t[5], '후보 이름(3단계)': t[6], '근거': t[7]})
cand = pd.DataFrame(rows).sort_values(['자료', '건수(행)', 'HACCP 품목명', '후보 순위'], ascending=[True, False, True, True])
cand.to_csv(MID + r'\haccp_품목_대응후보.csv', index=False, encoding='utf-8-sig')

auto = (df[df.train_prdlst_code.notna()].groupby(['src', 'prdlst_nm', 'train_prdlst_code', 'train_prdlst_nm', 'train_prdlst_code_2', 'train_prdlst_nm_2', 'prdlst_대응'], dropna=False)
        .size().rename('건수(행)').reset_index().sort_values(['src', '건수(행)'], ascending=[True, False]))
auto.to_csv(MID + r'\haccp_품목_자동대응.csv', index=False, encoding='utf-8-sig')

# ---------------- 요약 수치 ----------------
L = []
def P(*x):
    s = ' '.join(str(v) for v in x); print(s); L.append(s)
for src, d in df.groupby('src'):
    P(f'== {src}: 행 {len(d):,}, 완전 중복 표시 {int(d.dup_in_src.sum()):,}')
    P('  시군구 코드 있음', int(d.sgg_code.notna().sum()), f'({d.sgg_code.notna().mean()*100:.2f}%)',
      '/ 시 단위만', int(d.sgg_시단위.notna().sum()), '/ 그 밖 비움', int((d.sgg_code.isna() & d.sgg_시단위.isna()).sum()))
    P('  대응 방법별 행:'); P(d.sgg_대응.value_counts().to_string())
    P('  시 단위만인 시:', d.sgg_시단위.value_counts().to_dict())
    nodash = ~d.prdlst_대응.str.startswith('품목명 없음')
    P('  품목: 자동 대응 행', int(d.train_prdlst_code.notna().sum()), f'({d.train_prdlst_code.notna().mean()*100:.2f}%)',
      '/ 3단계까지', int(d.train_prdlst_code_2.notna().sum()), f'({d.train_prdlst_code_2.notna().mean()*100:.2f}%)',
      f'/ 품목명 있는 행 {int(nodash.sum()):,} 중 {d[nodash].train_prdlst_code.notna().mean()*100:.2f}%')
    P('  품목 대응 방법별 행:'); P(d.prdlst_대응.value_counts().to_string())
    u = d.groupby('_업소').train_prdlst_code.agg(lambda s: s.notna().any())
    ua = d.groupby('_업소').train_prdlst_code.agg(lambda s: s.notna().all())
    P(f'  업소 {len(u):,} 중 한 행이라도 품목 대응 {int(u.sum()):,} ({u.mean()*100:.2f}%), 모든 행 대응 {int(ua.sum()):,} ({ua.mean()*100:.2f}%)')
    P(f'  품목명 고유 {d.prdlst_nm.nunique():,} 중 자동 대응된 이름 {d[d.train_prdlst_code.notna()].prdlst_nm.nunique():,}')
    y = pd.Series([x.year if x else None for x in d.appn_de]).value_counts().sort_index()
    P('  연도별 인증(지정)일 행 수:', {int(k): int(v) for k, v in y.items()})
    top = d[d.sgg_code.notna()].groupby(['sgg_code', 'sgg_nm']).size().sort_values(ascending=False).head(15)
    P('  시군구별 행 수 상위 15:'); P(top.to_string())
open(MID + r'\_haccp_만들기_요약.txt', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
