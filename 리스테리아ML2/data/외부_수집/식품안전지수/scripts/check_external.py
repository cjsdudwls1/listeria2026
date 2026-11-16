# 받은 원자료 점검(읽기만): 행 수·기간·지역 단위·우리 시군구 256개에 닿는 정도(대략).
# 사용: python -I check_external.py <식품안전지수 폴더> <시군구_대표좌표.parquet>
# 주의: 원자료 파일은 고치지 않음. 이름 맞추기는 '대략 몇 개 닿는지' 보려는 임시 비교일 뿐 조인 규칙이 아님.
import sys, os, glob, re
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

base, sggp = sys.argv[1], sys.argv[2]
sgg = pd.read_parquet(sggp)

ALIAS = {'서울': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시', '광주': '광주광역시',
         '대전': '대전광역시', '울산': '울산광역시', '세종': '세종특별자치시', '경기': '경기도', '강원': '강원도',
         '충북': '충청북도', '충남': '충청남도', '전북': '전라북도', '전남': '전라남도', '경북': '경상북도',
         '경남': '경상남도', '제주': '제주특별자치도'}
NEW = {'강원도': '강원특별자치도', '전라북도': '전북특별자치도', '광주광역시': '전남광주통합특별시',
       '전라남도': '전남광주통합특별시', '제주도': '제주특별자치도'}


def norm_sido(s):
    s = str(s).strip()
    s = ALIAS.get(s, s)
    return NEW.get(s, s)


# 우리 목록: (시도, 시군구 끝말) → 키 두 가지(구까지 / 시까지)
ours = []
for nm in sgg.sgg_nm:
    t = nm.split()
    sido = norm_sido(t[0])
    full = ' '.join([sido] + t[1:])
    city = ' '.join([sido] + t[1:2])  # 일반시의 구는 시까지
    ours.append((full, city))
OUR_FULL = {f for f, _ in ours}


def key_from_addr(a):
    """주소 문자열 → '시도 시군구(구)' 와 '시도 시군구' 두 가지."""
    t = str(a).replace(',', ' ').split()
    if not t:
        return None, None
    sido = norm_sido(t[0])
    if sido == '세종특별자치시':
        return sido, sido
    rest = [x for x in t[1:3]]
    if not rest:
        return sido, sido
    city = f'{sido} {rest[0]}'
    if len(rest) > 1 and rest[0].endswith('시') and rest[1].endswith('구'):
        return f'{sido} {rest[0]} {rest[1]}', city
    return city, city


def coverage(keys_full, keys_city):
    kf, kc = set(keys_full), set(keys_city)
    exact = sum(1 for f, c in ours if f in kf)
    city_only = sum(1 for f, c in ours if f not in kf and (c in kc or c in kf))
    return exact, city_only, 256 - exact - city_only


def read_csv_any(p, **kw):
    for enc in ('utf-8-sig', 'cp949'):
        try:
            sep = '\t' if open(p, encoding=enc).readline().count('\t') > 1 else ','
            return pd.read_csv(p, encoding=enc, sep=sep, dtype=str, **kw), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(p)


def nm(p):
    return os.path.basename(p)


print('######## GAP 인증정보 (인증 단위, 유효기간 있음)')
for p in sorted(glob.glob(os.path.join(base, 'GAP_인증', '원자료', 'GAP_인증정보_*.csv'))):
    d, enc = read_csv_any(p)
    cols = list(d.columns)
    st = [c for c in cols if '시작일' in c][0]; en = [c for c in cols if '종료일' in c][0]
    ad = [c for c in cols if c == '주소'][0]
    keys = d[ad].map(key_from_addr)
    kf = [k[0] for k in keys if k[0]]; kc = [k[1] for k in keys if k[1]]
    ex, co, no = coverage(kf, kc)
    mush = d['품목'].astype(str).str.contains('버섯').sum()
    addr_tokens = d[ad].astype(str).str.split().str.len()
    print(f'{nm(p)} | {enc} | 행 {len(d):,} | 열 {len(cols)} | 유효시작 {d[st].min()}~{d[st].max()} | 유효종료 {d[en].min()}~{d[en].max()} '
          f'| 지정일자 {d["지정일자"].min()}~{d["지정일자"].max()} | 주소 낱말 수 중앙값 {addr_tokens.median():.0f} '
          f'| 시군구 종류 {len(set(kf))} | 우리 256 중 구까지 일치 {ex}·시까지만 {co}·없음 {no} | 버섯 품목 행 {mush}')

print('\n######## GAP 인증농가현황 (필지 단위로 보임, 날짜 없음)')
for p in sorted(glob.glob(os.path.join(base, 'GAP_인증', '원자료', 'GAP_인증농가현황_*.csv'))):
    d, enc = read_csv_any(p)
    loc = d['소재지']
    keys = loc.map(key_from_addr)
    kf = [k[0] for k in keys if k[0]]; kc = [k[1] for k in keys if k[1]]
    ex, co, no = coverage(kf, kc)
    mush = d['품목'].astype(str).str.contains('버섯').sum()
    print(f'{nm(p)} | {enc} | 행 {len(d):,} | 열 {list(d.columns)} | 인증번호 종류 {d["인증번호"].nunique():,} '
          f'| 소재지 종류 {loc.nunique()} | 우리 256 중 구까지 {ex}·시까지만 {co}·없음 {no} | 버섯 품목 행 {mush:,}')

print('\n######## GAP 인증 시군별 통계 (앞 8줄 원문)')
for p in sorted(glob.glob(os.path.join(base, 'GAP_인증', '원자료', 'GAP_인증_시군별통계_*.csv'))):
    raw = open(p, 'rb').read().decode('cp949', errors='replace').splitlines()
    print(nm(p), '| 줄수', len(raw))
    for ln in raw[:8]:
        print('   ', ln[:250])
    print('   ...', raw[-2][:200])

print('\n######## 친환경 인증품목 (인증 단위, 지역 없음)')
for p in sorted(glob.glob(os.path.join(base, '친환경_인증', '원자료', '친환경_인증품목_*.csv'))):
    d, enc = read_csv_any(p)
    cols = list(d.columns)
    st = [c for c in cols if '시작일' in c][0]
    kinds = d['인증종류명'].value_counts().head(6).to_dict()
    print(f'{nm(p)} | {enc} | 행 {len(d):,} | 열 {cols} | 인증시작 {d[st].min()}~{d[st].max()} | 종류 {kinds}')

print('\n######## 친환경 시도별 통계(농산물) — 연도·시도·시군구')
for p in sorted(glob.glob(os.path.join(base, '친환경_인증', '원자료', '친환경_시도별통계_농산물_*.csv'))):
    d, enc = read_csv_any(p)
    c = list(d.columns)
    y, sd, sg, gb = c[0], c[1], c[2], c[3]
    per = d.groupby(y).agg(행=(sg, 'size'), 시도=(sd, 'nunique'), 시군구=(sg, 'nunique'))
    print(f'{nm(p)} | {enc} | 행 {len(d):,} | 열 {c} | 구분 값 {sorted(d[gb].unique())}')
    print('   연도별:', per.to_dict('index'))
    last = d[d[y] == d[y].max()]
    kf = []; kc = []
    for a, b in zip(last[sd], last[sg]):
        k = key_from_addr(f'{a} {b}')
        kf.append(k[0]); kc.append(k[1])
    ex, co, no = coverage(kf, kc)
    print(f'   마지막 해 {d[y].max()} 우리 256 중 구까지 {ex}·시까지만 {co}·없음 {no}')
    odd = sorted(set(last[sg]) - {x.split()[-1] for x in OUR_FULL})[:15]
    print('   우리 목록 끝말에 없는 시군구 이름(최대 15):', odd)

print('\n######## 경지면적 (KOSIS 시군별 논밭)')
for p in sorted(glob.glob(os.path.join(base, '경지면적', '원자료', '*.csv'))):
    d, enc = read_csv_any(p)
    code = d['법정동코드'].astype(str)
    lv = code.map(lambda c: 'sido' if c[2:] == '0' * 8 else ('sgg' if c[5:] == '00000' else 'other'))
    s5 = set(code.str[:5]); s4 = set(code.str[:4])
    ex = sum(1 for c in sgg.sgg_code if c in s5)
    city = sum(1 for c in sgg.sgg_code if c not in s5 and (c[:4] + '0') in s5)
    print(f'{nm(p)} | {enc} | 행 {len(d):,} | 열 수 {d.shape[1]} | 코드 수준 {lv.value_counts().to_dict()} '
          f'| 우리 256 중 코드 앞5자리 일치 {ex}·시(앞4+0)까지만 {city}·없음 {256 - ex - city}')
    nulls = d.isna().sum(); nulls = nulls[nulls > 0]
    print('   빈 칸 있는 열:', nulls.to_dict())
    miss = [f'{c} {n}' for c, n in zip(sgg.sgg_code, sgg.sgg_nm) if c not in s5 and (c[:4] + '0') not in s5]
    print('   못 닿는 우리 시군구 예(최대 30):', miss[:30])

print('\n######## 지하수 수질측정망 수질검사결과(공공데이터포털 파일)')
for p in sorted(glob.glob(os.path.join(base, '음용지하수_수질', '원자료', '*.csv'))):
    d, enc = read_csv_any(p)
    print(nm(p), enc, d.shape, list(d.columns), '| 연도', sorted(d[d.columns[0]].unique()), '| 반기', sorted(d['반기'].unique()))
