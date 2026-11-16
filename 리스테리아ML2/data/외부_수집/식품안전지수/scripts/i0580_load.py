# I0580 원자료(구간별 JSON)를 읽어 표로 합침. 값은 모두 문자 그대로(가공 없음). venv 파이썬(pandas)용, 네트워크 안 씀.
import os, glob, json
import pandas as pd

RAW = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_수집\식품안전지수\HACCP\원자료\I0580"
COLS = ['PRSDNT_NM', 'SITE_ADDR', 'LCNS_NO', 'HACCP_APPN_DT', 'PRDLST_NM', 'BSSH_NM', 'CRTFC_RETN_DT', 'CLSBIZ_DT',
        'INDUTY_CD_NM', 'HACCP_APPN_NO', 'CRTFC_ENDDT', 'CLSBIZ_DVS_CD_NM', 'ASGN_CANCL_DT']


def load(raw=RAW):
    """→ (표, 파일 목록, 파일별 정보 표). 표에는 _start(구간 시작 번호)·_pos(구간 안 순서)·_rownum(전체 순번) 추가."""
    files = sorted(glob.glob(os.path.join(raw, 'r*_*.json')))
    rows = []; info = []
    for f in files:
        b = os.path.basename(f)
        s, e = int(b[1:7]), int(b[8:14])
        o = json.load(open(f, encoding='utf-8'))['I0580']
        items = o.get('row') or []
        info.append({'file': b, 'start': s, 'end': e, 'total_count': o.get('total_count'), 'n': len(items),
                     'code': (o.get('RESULT') or {}).get('CODE')})
        for i, x in enumerate(items):
            r = {k: (None if v is None else str(v)) for k, v in x.items()}
            r['_start'] = s; r['_pos'] = i; r['_rownum'] = s + i
            rows.append(r)
    df = pd.DataFrame(rows)
    return df, files, pd.DataFrame(info)
