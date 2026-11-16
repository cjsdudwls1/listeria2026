# HACCP 주소 → 우리 시군구 256개 코드 규칙. haccp_build.py(첫 사건표)의 map_addr·OLD·SPLIT·SIDO_ALIAS 를 글자 그대로 옮긴 것.
# (haccp_build.py 는 불러오면 사건표를 다시 만들어 버리므로, 규칙만 이 파일로 따로 둠. 규칙을 바꾸지 않음.)
import pandas as pd

SGG = r'C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\데이터셋\시군구_대표좌표.parquet'
sgg = pd.read_parquet(SGG, columns=['sgg_code', 'sgg_nm'])
NAME2CODE = dict(zip(sgg.sgg_nm, sgg.sgg_code))
CODE2NAME = dict(zip(sgg.sgg_code, sgg.sgg_nm))
GU_CITY = sorted({' '.join(n.split()[:2]) for n in sgg.sgg_nm if len(n.split()) == 3})
SIDO_NOW = sorted({n.split()[0] for n in sgg.sgg_nm})
SIDO_ALIAS = {
    '서울': '서울특별시', '서울시': '서울특별시', '부산': '부산광역시', '대구': '대구광역시', '인천': '인천광역시',
    '대전': '대전광역시', '울산': '울산광역시', '세종': '세종특별자치시', '세종시': '세종특별자치시', '경기': '경기도',
    '강원': '강원특별자치도', '강원도': '강원특별자치도', '충북': '충청북도', '충남': '충청남도',
    '전북': '전북특별자치도', '전라북도': '전북특별자치도', '경북': '경상북도', '경남': '경상남도',
    '제주': '제주특별자치도', '제주도': '제주특별자치도',
    '광주': '전남광주통합특별시', '광주광역시': '전남광주통합특별시', '전남': '전남광주통합특별시', '전라남도': '전남광주통합특별시',
}
OLD = {
    '인천광역시 남구': ('28177', '옛 이름: 인천광역시 남구 → 미추홀구(2018-07-01 이름 바뀜)'),
    '인천광역시 동구': ('28125', '옛 이름: 인천광역시 동구 → 제물포구(인천 행정체제 개편, 동구 전체가 제물포구로 들어감)'),
    '경상북도 군위군': ('27720', '옛 이름: 경상북도 군위군 → 대구광역시 군위군(2023-07-01 편입)'),
    '충청남도 연기군': ('36110', '옛 이름: 충청남도 연기군 → 세종특별자치시(2012-07-01)'),
    '경기도 여주군': ('41670', '옛 이름: 여주군 → 여주시(2013 시 승격)'),
    '충청남도 당진군': ('44270', '옛 이름: 당진군 → 당진시(2012 시 승격)'),
    '경상남도 진해시': ('48129', '옛 이름: 진해시 → 창원시 진해구(2010 통합)'),
}
SPLIT = {
    '인천광역시 중구': '나뉨: 인천광역시 중구 → 제물포구(내륙)·영종구(영종도) — 정하지 않음',
    '인천광역시 서구': '나뉨: 인천광역시 서구 → 서해구·검단구 — 정하지 않음',
    '충청북도 청원군': '나뉨: 청원군 → 청주시 4개 구(2014 통합) — 정하지 않음',
    '경상남도 마산시': '나뉨: 마산시 → 창원시 마산합포구·마산회원구(2010 통합) — 정하지 않음',
    '경상남도 창원시 마산구': '나뉨 — 정하지 않음',
    '경기도 고양시 일산구': '나뉨: 일산구 → 일산동구·일산서구(2005) — 정하지 않음',
}


def map_addr(s):
    """원문 주소 → (sgg_code, sgg_시단위, 대응방법). 대응 못 하면 code=None. (haccp_build.py 와 같음)"""
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
    for k in (2, 1, 0):
        if len(rest) < k:
            continue
        cand = ' '.join([sido] + rest[:k])
        if cand in NAME2CODE:
            how = '이름 같음' if len(toks) == k + 1 else f'주소 앞 {k+1}낱말이 이름과 같음(뒤 낱말 버림)'
            return NAME2CODE[cand], None, '; '.join(notes + [how])
    for k in (2, 1):
        if len(rest) < k:
            continue
        cand = ' '.join([sido] + rest[:k])
        if cand in OLD:
            code, why = OLD[cand]
            return code, None, '; '.join(notes + [why])
        if cand in SPLIT:
            return None, None, '; '.join(notes + [SPLIT[cand]])
    if rest and f'{sido} {rest[0]}' in GU_CITY:
        city = f'{sido} {rest[0]}'
        extra = '' if len(rest) == 1 else f'(뒤 낱말 "{" ".join(rest[1:])}" 은 구 이름이 아님)'
        return None, city, '; '.join(notes + [f'시 단위만 있음 — 일반구 정하지 않음{extra}'])
    if not rest:
        return None, None, '; '.join(notes + ['대응 못 함: 시도만 있음'])
    return None, None, '; '.join(notes + [f'대응 못 함: {sido} 에 "{rest[0]}" 없음'])
