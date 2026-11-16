# 시군구 이름 대응표(친환경·지하수·경지면적) 하나로 묶기(읽기만, 결과 csv 1개).
# 사용: python -I build_mapping_table.py <식품안전지수 폴더> <시군구_대표좌표.parquet> <raw_korea_region_code.parquet> <중간/외부지표 폴더>
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import ext_sgg_map as M

sys.stdout.reconfigure(encoding='utf-8')
base, sgg_path, rc_path, mid = sys.argv[1:5]
sgg = M.load_sgg(sgg_path)
nmof = dict(zip(sgg['sgg_code'], sgg['sgg_nm']))
eco = pd.read_csv(os.path.join(mid, '_대응_친환경.csv'), dtype=str, keep_default_na=False)
gw = pd.read_csv(os.path.join(mid, '_대응_지하수.csv'), dtype=str, keep_default_na=False)

# 경지면적 코드
k = pd.read_csv(os.path.join(base, '경지면적', '원자료', 'KOSIS_시군구_논밭면적_2008_2025_20251231.csv'), encoding='cp949', dtype={'법정동코드': str})
rc = pd.read_parquet(rc_path)[['legal_dong_code', 'legal_dong_nm', 'molit_use_yn']]
rcn = dict(zip(rc['legal_dong_code'], rc['legal_dong_nm'])); rcu = dict(zip(rc['legal_dong_code'], rc['molit_use_yn']))
info = {}
for yr in range(2016, 2026):
    units, ratio_unit = M.kosis_units(sgg, yr)
    users = {}
    for c, u in ratio_unit.items():
        users.setdefault(u, []).append(c)
    for code, kind, name, mem in units:
        d = info.setdefault(code, {'kind': kind, 'name': name, 'mem': {}, 'users': {}})
        d['mem'].setdefault(tuple(mem), []).append(yr)
        d['users'].setdefault(tuple(sorted(users.get(code, []))), []).append(yr)
KR = []
for code in k['법정동코드']:
    nm = rcn.get(code, '(지역코드표에 이름 없음)')
    if code in info:
        d = info[code]
        mems = list(d['mem'].items())
        mem_txt = ' / '.join(f"{min(y)}~{max(y)}: " + ';'.join(m) for m, y in mems) if len(mems) > 1 else ';'.join(mems[0][0])
        users = list(d['users'].items())
        use_txt = ' / '.join((f"{min(y)}~{max(y)}: " if len(users) > 1 else '') + (';'.join(nmof[c] for c in u) if u else '없음') for u, y in users)
        typ = f"분모 단위({d['kind']})"
        desc = f"2016~2025 분모로 씀. 이 단위 비율을 받는 우리 시군구: {use_txt}"
        if code == '2700000000':
            desc += '. 대구 합계는 2023년까지 군위군을 빼고(달성군+2771900000), 2024년부터 군위군을 넣은 값과 맞음 → 분자 구성원도 해마다 맞춤'
        codes_txt = mem_txt
        names_txt = ''
        if len(mems) == 1:
            names_txt = ';'.join(nmof[c] for c in mems[0][0])
        flags = dict(이름바뀜=False, 나뉨=False, 합쳐짐=False, 시단위값=d['kind'] == '시')
        if code[:2] in ('42', '45', '46', '29'):
            flags['이름바뀜'] = True
            desc += f'. 옛 코드({code[:5]}) → 오늘 코드, 이름으로 맞춤'
    else:
        typ = '쓰지 않음'
        desc = M.KOSIS_UNUSED.get(code, '2016~2025 분모로 쓰지 않음')
        codes_txt = ''; names_txt = ''
        flags = dict(이름바뀜=False, 나뉨=False, 합쳐짐=False, 시단위값=False)
    KR.append({'자료': '경지면적(KOSIS 논·밭 면적)', '원문_시도': '', '원문_이름': nm, '법정동코드': code,
               '나오는_해(자료연도)': '2008~2025', '대응_sgg_code': codes_txt, '대응_sgg_nm': names_txt,
               '대응_유형': typ, **flags, '설명': desc + (f" (지역코드표 상태: {rcu.get(code)})" if code in rcu else '')})
kr = pd.DataFrame(KR)
allm = pd.concat([eco, gw, kr], ignore_index=True)
for c in ['이름바뀜', '나뉨', '합쳐짐', '시단위값']:
    allm[c] = allm[c].astype(str).map({'True': True, 'False': False, True: True, False: False})
out = os.path.join(mid, '시군구_이름_대응.csv')
allm.to_csv(out, index=False, encoding='utf-8-sig')
print('대응표 행', len(allm), allm['자료'].value_counts().to_dict())

# 우리 256 이 각 자료에서 대응되는지
for src, d in [('친환경', eco), ('지하수', gw)]:
    got = set()
    for s in d['대응_sgg_code']:
        got |= set(x for x in s.split(';') if x)
    print(src, '대응된 우리 시군구', len(got), '빠진 것', sorted(set(sgg['sgg_code']) - got))
