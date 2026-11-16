# HACCP 원자료(쪽별 JSON)를 읽어 표로 합침. 값은 모두 문자 그대로(가공 없음). venv 파이썬(pandas)용, 네트워크 안 씀.
import os, glob, json
import pandas as pd

RAW = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_수집\식품안전지수\HACCP\원자료"
DIRS = {
    'mfds': os.path.join(RAW, '15058303_식약처_HACCP지정현황_20261019'),
    'cert': os.path.join(RAW, '15033311_인증원_HACCP인증업체_20261019'),
}

def _items(obj):
    b = obj['body']
    it = b['items']
    if isinstance(it, dict):
        it = it.get('item', [])
        it = it if isinstance(it, list) else [it]
    out = [x.get('item', x) if isinstance(x, dict) and 'item' in x else x for x in it]
    return b.get('totalCount'), out

def load(svc):
    files = sorted(glob.glob(os.path.join(DIRS[svc], 'p*_n*.json')))
    rows = []; totals = set()
    for f in files:
        page = int(os.path.basename(f)[1:5])
        tot, items = _items(json.load(open(f, encoding='utf-8')))
        totals.add(tot)
        for i, x in enumerate(items):
            r = {k: (None if v is None else str(v)) for k, v in x.items()}
            r['_page'] = page; r['_pos'] = i
            rows.append(r)
    df = pd.DataFrame(rows)
    return df, files, totals
