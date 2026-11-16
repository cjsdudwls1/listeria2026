# -*- coding: utf-8 -*-
"""
07_축사_시군구.py — 축사 목록 엑셀만으로 시군구별 농장 수를 묶는다(우리 시료·시군구 표와 합치지 않음).

입력 : 리스테리아ML2/data/외부_수집/축사/축사 현황 데이터(8.24)_최종v2.xlsx (시트 '농장목록', '데이터_모집_지역')
출력 : 리스테리아ML2/data/외부_수집/축사/축사_시군구별_농장수.xlsx (엑셀에 나온 시군구만, 한 행 = 한 시군구)
  시도, 시군구           : 주소 앞 단어. 시도 이름 변형만 하나로 맞춤
                           (강원도→강원특별자치도, 전라북도→전북특별자치도, 전라남도→전남광주통합특별시,
                            충남→충청남도, 경상냠도→경상남도). 화성시·전주시는 주소의 구까지, 세종은 시 하나.
  모집지역               : 엑셀 '데이터_모집_지역' 시트에 있으면 1
  농장수                 : 행 수
  낙농장(젖소)           : 가축종(원본)에 '젖소' 글자가 있는 농장 수
  소, 돼지, 가금류       : 엑셀 0/1 칸 = 1 인 농장 수(둘 이상이면 각각 셈)
  가축종빈칸             : 가축종(원본)이 빈 농장 수
  데이터기준일_최소·최대 : 빈 값 빼고
"""
import sys
from collections import Counter, defaultdict

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

DIR = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\외부_수집\축사"
XLSX = DIR + r"\축사 현황 데이터(8.24)_최종v2.xlsx"
OUT = DIR + r"\축사_시군구별_농장수.xlsx"

SIDO_ALIAS = {"강원도": "강원특별자치도", "전라북도": "전북특별자치도", "전라남도": "전남광주통합특별시",
              "충남": "충청남도", "경상냠도": "경상남도"}
SPLIT_CITY = {"화성시", "전주시"}


def key_of(addr):
    w = str(addr).split()
    sido = SIDO_ALIAS.get(w[0], w[0])
    if sido == "세종특별자치시":
        return sido, "세종특별자치시"
    if w[1] in SPLIT_CITY:
        return sido, f"{w[1]} {w[2]}"
    return sido, w[1]


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    rows = list(wb["농장목록"].iter_rows(values_only=True))
    hdr, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(hdr)}
    rec = [r[0] for r in wb["데이터_모집_지역"].iter_rows(values_only=True)][2:]
    rec = [" ".join([SIDO_ALIAS.get(s.split()[0], s.split()[0])] + s.split()[1:]) for s in rec if s]

    agg = defaultdict(Counter)
    dates = defaultdict(list)
    for r in rows:
        k = key_of(r[ix["소재지 주소"]])
        a = agg[k]
        a["농장수"] += 1
        sp = r[ix["가축종(원본)"]]
        a["가축종빈칸"] += int(sp is None)
        a["낙농장(젖소)"] += int(sp is not None and "젖소" in sp)
        a["소"] += int(r[ix["소"]] == 1)
        a["돼지"] += int(r[ix["돼지"]] == 1)
        a["가금류"] += int(r[ix["가금류"]] == 1)
        if r[ix["데이터기준일"]] is not None:
            dates[k].append(r[ix["데이터기준일"]].date())

    def recruit(sido, sgg):
        full = sido if sido == sgg else f"{sido} {sgg}"
        return int(any(full == x or full.startswith(x) or x.startswith(full) for x in rec))

    cols = ["농장수", "낙농장(젖소)", "소", "돼지", "가금류", "가축종빈칸"]
    out = openpyxl.Workbook()
    ws = out.active
    ws.title = "시군구별_농장수"
    ws.append(["시도", "시군구", "모집지역"] + cols + ["데이터기준일_최소", "데이터기준일_최대"])
    for (sido, sgg) in sorted(agg):
        ds = dates.get((sido, sgg))
        ws.append([sido, sgg, recruit(sido, sgg)] + [agg[(sido, sgg)][c] for c in cols]
                  + [min(ds) if ds else None, max(ds) if ds else None])
    out.save(OUT)

    tot = Counter()
    for v in agg.values():
        tot.update(v)
    print(f"저장: {OUT} — 시군구 {len(agg)}개, 합계 {dict(tot)} (엑셀 행 {len(rows)})")
    for row in ws.iter_rows(min_row=2, values_only=True):
        print(row)


if __name__ == "__main__":
    main()
