# -*- coding: utf-8 -*-
"""
02_시군구_배정.py — 국내 시료 행마다 '오늘 기준 시군구 코드'(sgg_code, 5자리 문자열)를 붙인다.

입력 (읽기만 함)
  - 시료 파일   : C:\\Users\\NT751\\Desktop\\천영진\\리스테리아ML\\data\\listeria_data.parquet
  - 법정동 표   : 리스테리아ML2/data/raw/raw_korea_region_code.parquet  (ba_data_info.korea_region_code)
  - 대응표      : 리스테리아ML2/data/시군구_코드_대응표.xlsx (시트 '미상_대응표', '폐지코드_대응표')

출력
  - 리스테리아ML2/data/데이터셋/시료_시군구.parquet  : 원본 국내 행(src_route='domestic') 가운데 sgg_code 를 정한 행의
                                                    원래 칼럼 전부(값·자료형·행 순서 그대로) + sgg_code
  - 리스테리아ML2/data/데이터셋/시군구_대표좌표.parquet : sgg_code, sgg_nm, lat, lon, source
  - 리스테리아ML2/data/중간/시군구_배정_요약.csv      : 원래 시군구 코드(5자리) x sgg_code 별 행 수·검출 수 (점검용, 못 정한 행 포함)
  - 리스테리아ML2/data/중간/시군구_미배정_삭제행.csv  : 못 정해 시료_시군구.parquet 에서 뺀 행 목록과 사유(사유코드 A~D)
  - 리스테리아ML2/data/중간/시군구_미배정_삭제행_요약.csv : 사유 x 원래 시군구 코드별 행 수·검출 수

오늘 기준 = korea_region_code 에서 molit_use_yn='존재' 인 level='sigungu' 행.
일반구를 가진 시의 상위 코드(같은 앞 4자리에 '존재' 일반구가 있는 시)는 배정 대상에서 뺀다.

사용자가 승인한 규칙 (2026-10-16)
  0) 대응표 먼저 적용: 미상_대응표 (주소, 현재 10자리 코드) -> 바꿀 시군구 코드,
     폐지코드_대응표 (시료 10자리 코드, 주소) -> 바꿀 시군구 코드. 대응표 결과는 확정(아래 1 만 더 적용).
  1) 1:1 이름 변경: 광주(29xxx)·전남(46xxx) -> 전남광주통합특별시(12xxx) 같은 시군구 이름, 군위 47720 -> 27720.
  2) 10자리 코드가 읍면동·리 단위면 읍면동 이름(리 단위면 그 읍면 이름)으로, 승계 후보 안에서만 '존재' 시군구를 찾는다.
     (a) 같은 시군구가 '존재'이고 그 아래 같은 이름이 '존재'면 그대로
     (b) 일반구가 있는 시는 같은 시의 일반구 안에서
     (c) 인천 28110 -> {28125, 28155}, 28140 -> {28125}, 28260 -> {28275, 28290}
     후보 밖의 같은 이름은 쓰지 않는다. 같은 시군구 안에서 이름만 바뀐 경우는 시군구 그대로.
     폐지 동인데 같은 어간의 '존재' 리가 후보 안 한 구에만 있으면 그 구(화성 남양 쪽 북양동·무송동·장덕동·온석동·신남동 -> 41591).
  3) 주소의 구 이름 우선(일반구 있는 시의 모든 행 — 상위 시 코드 행과 일반구 코드 행 모두, 그리고 인천 나뉜 코드 행):
     주소에 후보 구('존재' 일반구) 이름이 단어로 있으면 그 구.
     단 주소의 읍·면·동 이름이 DB 에서 다른 한 구에만 '존재'하면 읍·면·동 쪽을 따른다.
     (예: 전주 완산구 코드인데 주소가 '…(중동)' -> 중동은 덕진구에만 '존재' -> 52113)
  4) 10자리 코드가 시군구 단위(뒤 5자리 00000)이고 그 코드가 나뉜 곳이면: 주소의 구 이름 -> 주소의 읍·면·동 이름
     -> 같은 도로명+건물번호를 가진 다른 국내 시료(이미 코드가 정해진 행)의 코드가 한 곳으로 모이면 그 코드 -> 못 정함(None).
     (도로명과 건물번호 사이의 띄어쓰기·쉼표는 무시: '영종해안남로 321번길, 190' = '영종해안남로321번길 190')
  5) 못 정한 행은 sgg_code=None (학습에서 제외).
     시료_시군구.parquet 에는 이 행을 넣지 않고(사용자 결정 2026-10-16), 행 목록·사유는 시군구_미배정_삭제행.csv 에 남긴다.
  6) 창원 의창구(48121)로 남은 행 중 의창구 대원동(4812112300)·두대동(4812111100) 코드 행,
     주소가 '의창구 중앙대로 151 창원시청' 인 행 -> 48123.
  대표좌표: korea_region_code 시군구 행 lat·lon. 예외 28720 옹진군 = 학습 기간(mnfctur_de <= 2024-06-30) 옹진군 행의
  시료 lat·lon 중앙값(위도·경도 각각, 시료 좌표가 시도청 점과 같은 행은 뺌).

다시 돌리면 같은 결과가 나온다(무작위 없음). 근거 칼럼은 결과 파일에 넣지 않는다(사용자 지시).

다른 스크립트에서 같은 규칙 쓰기 (파일 이름이 숫자로 시작해 import 문 대신 importlib 로 불러온다; main() 은 돌지 않는다)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sgg02", r"C:\\Users\\NT751\\Desktop\\천영진\\리스테리아ML2\\notebook\\02_시군구_배정.py")
    sgg02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sgg02)
    region = sgg02.Region(sgg02.KRC)                       # 법정동 표 (raw_korea_region_code.parquet)
    codes = sgg02.assign_sgg(c10_list, addr_list_or_None, region=region)   # 5자리 시군구 코드 또는 None
"""
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

import openpyxl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2"
SAMPLE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_data.parquet"
KRC = BASE + r"\data\raw\raw_korea_region_code.parquet"
XLSX = BASE + r"\data\시군구_코드_대응표.xlsx"
OUT_SAMPLE = BASE + r"\data\데이터셋\시료_시군구.parquet"
OUT_COORD = BASE + r"\data\데이터셋\시군구_대표좌표.parquet"
OUT_SUMMARY = BASE + r"\data\중간\시군구_배정_요약.csv"
OUT_DROP = BASE + r"\data\중간\시군구_미배정_삭제행.csv"
OUT_DROP_SUMMARY = BASE + r"\data\중간\시군구_미배정_삭제행_요약.csv"

TRAIN_END = datetime(2024, 6, 30, 23, 59, 59)
INCHEON = {"28110": {"28125", "28155"}, "28140": {"28125"}, "28260": {"28275", "28290"}}
CHANGWON_TO_SEONGSAN_C10 = {"4812112300", "4812111100"}
CHANGWON_CITYHALL = "의창구 중앙대로 151 창원시청"


def words(s):
    return (s or "").split()


def level_of(c10):
    if c10[2:] == "00000000":
        return "sido"
    if c10[5:] == "00000":
        return "sigungu"
    if c10[8:] == "00":
        return "eupmyeondong"
    return "ri"


STEM_RE = re.compile(r"(읍|면|동|리|가)$")


def stem(n):
    return STEM_RE.sub("", n) if n else n


EMD_TOKEN_RE = re.compile(r"(?<![가-힣0-9])([가-힣0-9]+(?:읍|면|동|가))(?![가-힣])")
# 도로명 + 건물번호. 도로명과 건물번호 사이에 띄어쓰기·쉼표가 끼어도 된다('영종해안남로321번길, 190' -> ('영종해안남로321번길', '190'))
ROAD_RE = re.compile(r"([가-힣A-Za-z0-9]+(?:로|길))[\s,]*(\d+(?:-\d+)?)")


def road_keys(addr):
    """주소의 (도로명, 건물번호) 집합. '…로 321번길' 처럼 띄어 쓴 번길은 먼저 붙인다('…로321번길')."""
    s = re.sub(r"(로|길)\s+(\d+번?길)", r"\1\2", addr or "")
    return {(m.group(1), m.group(2)) for m in ROAD_RE.finditer(s)}


# ---------------------------------------------------------------- 법정동 표
class Region:
    def __init__(self, path):
        rows = pq.read_table(path).to_pylist()
        self.K = {r["legal_dong_code"]: r for r in rows}
        sg = {c[:5]: r for c, r in self.K.items() if r["level"] == "sigungu"}
        self.sg_nm = {c5: " ".join(words(r["legal_dong_nm"])) for c5, r in sg.items()}
        self.sg_status = {c5: r["molit_use_yn"] for c5, r in sg.items()}
        self.exist = {c5 for c5, s in self.sg_status.items() if s == "존재"}
        # 일반구를 가진 시: 같은 앞 4자리 + 이름이 '시 이름 + 공백'으로 시작하는 '존재' 구가 있는 '존재' 시 코드
        self.city_gus = {}
        for p in sorted(self.exist):
            if p[4] != "0":
                continue
            kids = {x for x in self.exist if x != p and x[:4] == p[:4] and self.sg_nm[x].startswith(self.sg_nm[p] + " ")}
            if kids:
                self.city_gus[p] = kids
        self.parents = set(self.city_gus)
        self.city_of = {}
        for p, kids in self.city_gus.items():
            self.city_of[p] = p
            for k in kids:
                self.city_of[k] = p
        self.targets = sorted(self.exist - self.parents)
        # '존재' 읍면동 이름, 리 이름 (시군구별)
        self.emd_names = defaultdict(set)
        self.ri_stems = defaultdict(set)
        for c, r in self.K.items():
            if r["molit_use_yn"] != "존재":
                continue
            if r["level"] == "eupmyeondong":
                self.emd_names[c[:5]].add(words(r["legal_dong_nm"])[-1])
            elif r["level"] == "ri":
                self.ri_stems[c[:5]].add(stem(words(r["legal_dong_nm"])[-1]))
        self.emd_stems = {c5: {stem(n) for n in ns} for c5, ns in self.emd_names.items()}
        # 1) 1:1 이름 변경 (29xxx·46xxx -> 12xxx, 47720 -> 27720)
        self.rename = {}
        self.rename_problem = {}
        for c5, s in self.sg_status.items():
            if s != "폐지" or c5[2:] == "000":
                continue
            if c5[:2] in ("29", "46"):
                tgt_sido = "12"
            elif c5 == "47720":
                tgt_sido = "27"
            else:
                continue
            body = " ".join(words(self.sg_nm[c5])[1:])
            hit = sorted(x for x in self.exist if x[:2] == tgt_sido and " ".join(words(self.sg_nm[x])[1:]) == body)
            if len(hit) == 1:
                self.rename[c5] = hit[0]
            else:
                self.rename_problem[c5] = hit
        for old, news in INCHEON.items():
            assert self.sg_status.get(old) == "폐지", old
            assert all(n in self.exist for n in news), news

    def emd_name_of(self, c10):
        """읍면동 단위 코드면 그 이름, 리 단위면 그 읍면 이름."""
        r = self.K.get(c10)
        if r is None:
            return None
        lv = level_of(c10)
        if lv == "eupmyeondong":
            return words(r["legal_dong_nm"])[-1]
        if lv == "ri":
            p = self.K.get(c10[:8] + "00")
            if p is not None:
                return words(p["legal_dong_nm"])[-1]
            w = words(r["legal_dong_nm"])
            return w[-2] if len(w) >= 2 else None
        return None

    def candidates(self, s5):
        """승계 후보. 일반구 있는 시 -> 그 시의 일반구들, 인천 나뉜 코드 -> 지정 구들, 그 밖 '존재' -> 자기 자신."""
        if s5 in INCHEON:
            return set(INCHEON[s5])
        if s5 in self.city_of:
            return set(self.city_gus[self.city_of[s5]])
        if s5 in self.exist:
            return {s5}
        return set()

    def gu_word(self, c5):
        return words(self.sg_nm[c5])[-1]

    def gu_in_addr(self, addr, cand, s5):
        """주소에 후보 구 이름이 단어로 들어 있는 구들."""
        city_short = ""
        if s5 in self.city_of:
            city_short = words(self.sg_nm[self.city_of[s5]])[-1]  # 예: '고양시'
        hits = set()
        for x in cand:
            gw = self.gu_word(x)
            pre = r"(?:^|(?<=[^가-힣0-9])" + (r"|(?<=" + re.escape(city_short) + r")" if city_short else "") + r")"
            if re.search(pre + re.escape(gw) + r"(?![가-힣])", addr or ""):
                hits.add(x)
        return hits

    def emd_in_addr(self, addr, cand):
        """주소에 든 읍·면·동 이름(단어)이 '존재' 읍면동으로 속한 후보 구들."""
        toks = set(EMD_TOKEN_RE.findall(addr or ""))
        return {x for x in cand if toks & self.emd_names[x]}


# ---------------------------------------------------------------- 대응표
def load_mapping(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    unk = {}
    for i, r in enumerate(wb["미상_대응표"].iter_rows(values_only=True)):
        if i == 0 or r[0] is None:
            continue
        unk[(r[0], str(r[1]))] = str(r[4])
    abol = {}
    for i, r in enumerate(wb["폐지코드_대응표"].iter_rows(values_only=True)):
        if i == 0 or r[0] is None:
            continue
        abol[(str(r[2]), r[4])] = str(r[9])
    wb.close()
    return unk, abol


# ---------------------------------------------------------------- 배정
def assign(region, C10, ADDR, mapping=None):
    """행별 (sgg_code 또는 None, 근거 문자열) 리스트와 요약 dict 를 돌려준다. 근거는 점검용(결과 파일에는 넣지 않음).

    C10  : 10자리 법정동 코드 문자열 리스트 (None 이면 못 정함)
    ADDR : 같은 길이의 주소 리스트 (주소가 없으면 None — 주소를 쓰는 규칙 0·3·4·6 이 걸리지 않는다)
    mapping : load_mapping() 결과 (unk, abol). None 이면 XLSX 에서 읽는다.
    규칙 4c(같은 도로명+건물번호)의 참고 행은 이번에 넘긴 행들 가운데 코드가 정해진 행이다.
    """
    unk, abol = mapping if mapping is not None else load_mapping(XLSX)
    combos = Counter(zip(C10, ADDR))
    dec = {}       # (c10, addr) -> (code, how)
    pending4c = {}  # (c10, addr) -> 후보 집합
    used_unk = Counter()
    used_abol = Counter()

    def step1(c5):
        return region.rename.get(c5, c5)

    for (c10, a), n in combos.items():
        key = (c10, a)
        if c10 is None or len(c10) != 10 or not c10.isdigit():
            dec[key] = (None, "X 10자리 법정동 코드 아님")
            continue
        # 0) 대응표
        if (c10, a) in abol:
            used_abol[(c10, a)] += n
            v = abol[(c10, a)]
            dec[key] = (step1(v), "0 폐지코드_대응표" + (" +1 이름 변경" if v in region.rename else ""))
            continue
        if (a, c10) in unk:
            used_unk[(a, c10)] += n
            v = unk[(a, c10)]
            dec[key] = (step1(v), "0 미상_대응표" + (" +1 이름 변경" if v in region.rename else ""))
            continue
        s5 = c10[:5]
        # 1) 1:1 이름 변경
        if s5 in region.rename:
            dec[key] = (region.rename[s5], "1 이름 변경 1:1")
            continue
        cand = region.candidates(s5)
        if not cand:
            dec[key] = (None, "X 후보 없음(폐지 코드, 규칙 밖)")
            continue
        split = (s5 in region.parents) or (s5 in INCHEON)  # 나뉜 코드(상위 시·인천 폐지 구)
        if len(cand) == 1 and not split:
            dec[key] = (s5, "A 그대로")
            continue
        if len(cand) == 1 and split:  # 인천 28140 -> 28125
            dec[key] = (next(iter(cand)), "2c 인천 승계 1곳")
            continue
        lv = level_of(c10)
        # 3) 주소의 구 이름 우선 (일반구 있는 시의 모든 행 + 인천 나뉜 코드 행)
        if split or s5 in region.city_of:
            G = region.gu_in_addr(a, cand, s5)
            if len(G) == 1:
                g = next(iter(G))
                E = region.emd_in_addr(a, cand)
                if len(E) == 1 and g not in E:
                    dec[key] = (next(iter(E)), "3 주소 구 이름이나 주소 읍면동이 다른 구")
                else:
                    dec[key] = (g, "3 주소 구 이름")
                continue
        if lv in ("eupmyeondong", "ri"):
            nm = region.emd_name_of(c10)
            hits = {x for x in cand if nm in region.emd_names[x]}
            if s5 in hits:
                dec[key] = (s5, "2a 같은 시군구 아래 같은 이름 존재")
                continue
            if len(hits) == 1:
                dec[key] = (next(iter(hits)), "2b/2c 후보 안 같은 이름 1곳")
                continue
            if len(hits) > 1:
                dec[key] = (None, "X 2 같은 이름이 후보 안 여러 구:" + ",".join(sorted(hits)))
                continue
            if s5 in region.exist and s5 not in region.parents:
                dec[key] = (s5, "2 이름만 바뀜(시군구 그대로)")
                continue
            sh = {x for x in cand if stem(nm) in region.emd_stems.get(x, set())}
            if len(sh) == 1:
                dec[key] = (next(iter(sh)), "2 어간 같은 읍면동 1곳")
                continue
            rh = {x for x in cand if stem(nm) in region.ri_stems.get(x, set())}
            if len(rh) == 1:
                dec[key] = (next(iter(rh)), "2 폐지 동 -> 어간 같은 리 1곳:" + nm)
                continue
            dec[key] = (None, "X 2 후보 안 같은 이름 없음:" + str(nm))
            continue
        # 나뉘지 않은 '존재' 구(일반구) 단위 코드 -> 그대로
        if lv == "sigungu" and not split:
            dec[key] = (s5, "A 그대로(구 단위 코드)")
            continue
        # 4) 시군구 단위 코드인데 나뉜 곳
        if lv == "sigungu" and split:
            E = region.emd_in_addr(a, cand)
            if len(E) == 1:
                dec[key] = (next(iter(E)), "4 주소 읍면동 이름")
                continue
            pending4c[key] = cand
            continue
        dec[key] = (s5 if s5 in region.exist and s5 not in region.parents else None, "X 규칙 밖")

    # 4c) 같은 도로명+건물번호 다른 시료
    road_idx = defaultdict(Counter)
    for (c10, a), (code, how) in dec.items():
        if code is None:
            continue
        n = combos[(c10, a)]
        for rk in road_keys(a):
            road_idx[rk][code] += n
    for key, cand in pending4c.items():
        m = Counter()
        for rk in road_keys(key[1]):
            for code, v in road_idx.get(rk, {}).items():
                if code in cand:
                    m[code] += v
        if len(m) == 1:
            dec[key] = (next(iter(m)), "4c 같은 도로명+건물번호 시료 한 곳")
        elif len(m) > 1:
            dec[key] = (None, "X 4c 같은 도로명+건물번호 시료가 여러 구:" + ",".join(sorted(m)))
        else:
            dec[key] = (None, "X 4c 같은 도로명+건물번호 시료 없음")

    # 6) 창원 의창구 -> 성산구
    for key, (code, how) in list(dec.items()):
        if code == "48121" and (key[0] in CHANGWON_TO_SEONGSAN_C10 or CHANGWON_CITYHALL in (key[1] or "")):
            dec[key] = ("48123", how + " +6 창원 성산구")

    info = dict(unk_keys=len(unk), unk_used_keys=len(used_unk), unk_rows=sum(used_unk.values()),
                abol_keys=len(abol), abol_used_keys=len(used_abol), abol_rows=sum(used_abol.values()),
                road_idx_keys=len(road_idx), pending4c_keys=len(pending4c))
    out = [dec[(c, a)] for c, a in zip(C10, ADDR)]
    return out, info


def assign_sgg(c10s, addrs=None, region=None, mapping=None, with_reason=False):
    """10자리 법정동 코드(+주소) -> 오늘 기준 시군구 코드(5자리 문자열, 못 정하면 None). 다른 스크립트(토양 등)에서 쓰는 입구.

    c10s   : 10자리 법정동 코드 리스트 (문자열·정수 모두 받음, None 허용)
    addrs  : 주소 리스트 또는 None (None 이면 모든 행 주소 없음)
    region : Region(KRC) — 여러 번 부를 때 한 번 만들어 넘기면 빠르다
    mapping: load_mapping(XLSX) 결과 — 대응표(주소 기준)는 시료 주소와 같은 (코드, 주소) 에만 걸린다
    with_reason=True 면 (코드 리스트, 근거 리스트, 요약 dict) 를 돌려준다.
    규칙은 이 파일 맨 위 설명(사용자 승인 2026-10-16)과 같다. 결과는 assign() 과 같다.
    """
    region = region if region is not None else Region(KRC)
    C10 = [None if c is None else str(c) for c in c10s]
    ADDR = [None] * len(C10) if addrs is None else list(addrs)
    assert len(ADDR) == len(C10), "c10s 와 addrs 길이가 다름"
    res, info = assign(region, C10, ADDR, mapping=mapping)
    codes = [c for c, _ in res]
    if with_reason:
        return codes, [h for _, h in res], info
    return codes


# ---------------------------------------------------------------- 못 정한 행의 삭제 사유 (사용자 결정 2026-10-16)
DROP_REASON = {
    "A": "법정동 코드가 구 단위로만 적혀 있고 그 구가 지금 둘로 나뉨(인천 서구→서해구·검단구, 인천 중구→영종구·제물포구). "
         "주소에 지금의 구·동 이름과 정확히 같은 이름이 없고, 같은 도로명·건물번호를 가진 시료 가운데 후보 구로 정해진 시료도 없어 지금의 구를 정할 수 없음",
    "B": "법정동 코드가 시 단위로만 적혀 있고 그 시가 일반구로 나뉨(화성·부천·청주·천안·수원). "
         "주소에 지금의 구·동 이름과 정확히 같은 이름이 없고(청주·천안은 주소가 시 이름뿐), 같은 도로명·건물번호를 가진 시료 가운데 후보 구로 정해진 시료도 없어 지금의 구를 정할 수 없음",
    "C": "같은 이름의 동(능동)이 지금 두 구(화성시 병점구·동탄구)에 모두 있어 정할 수 없음",
    "D": "법정동 코드(4159012900 화성시 오산동)가 현행 법정동 목록에서 폐지되었고 화성시 안에 같은 이름의 현행 동이 없어 지금의 구를 정할 수 없음",
}
DROP_A_CODES = {"28110", "28260"}                               # 사유 A 문장에 적은 인천 중구·서구
DROP_B_CODES = {"41590", "41190", "43110", "44130", "41110"}    # 사유 B 문장에 적은 화성·부천·청주·천안·수원


def drop_reason(region, c10, addr, how):
    """sgg_code 를 못 정한 행의 사유코드(A~D). assign() 의 근거(how)와 원래 코드로 정한다.
    문장(DROP_REASON)과 맞지 않는 행이 나오면 멈춘다(새 사유를 정해야 함)."""
    s5 = (c10 or "")[:5]
    if how == "X 4c 같은 도로명+건물번호 시료 없음":
        cand = region.candidates(s5)
        no_name = not region.gu_in_addr(addr, cand, s5) and not region.emd_in_addr(addr, cand)
        if no_name and s5 in DROP_A_CODES:
            return "A"
        if no_name and s5 in DROP_B_CODES and s5 in region.parents:
            return "B"
    if (how == "X 2 같은 이름이 후보 안 여러 구:41595,41597" and region.emd_name_of(c10) == "능동"
            and region.gu_word("41595") == "병점구" and region.gu_word("41597") == "동탄구"):
        return "C"
    if c10 == "4159012900" and how == "X 2 후보 안 같은 이름 없음:오산동" and region.K[c10]["molit_use_yn"] == "폐지":
        return "D"
    raise ValueError(f"삭제 사유 A~D 에 맞지 않는 못 정한 행: {c10} | {addr} | {how} — 사유를 새로 정해야 함")


def load_domestic():
    t = pq.read_table(SAMPLE)
    mask = pc.equal(pc.cast(t["src_route"], pa.string()), "domestic")
    return t.filter(mask)


def representative_coords(region, t, codes):
    """대상 시군구 대표좌표. 옹진군(28720)만 학습 기간 시료 좌표 중앙값."""
    LA = t["lat"].to_pylist()
    LO = t["lon"].to_pylist()
    D = t["mnfctur_de"].to_pylist()
    sido = region.K["2800000000"]
    s_la, s_lo = sido["lat"], sido["lon"]
    la_l, lo_l = [], []
    n_all = n_train = n_sido = n_nullxy = 0
    for code, la, lo, d in zip(codes, LA, LO, D):
        if code != "28720":
            continue
        n_all += 1
        if d is None or d > TRAIN_END:
            continue
        n_train += 1
        if la is None or lo is None:
            n_nullxy += 1
            continue
        if abs(la - s_la) < 1e-4 and abs(lo - s_lo) < 1e-4:
            n_sido += 1
            continue
        la_l.append(la)
        lo_l.append(lo)
    ong = (float(statistics.median(la_l)), float(statistics.median(lo_l)))
    rec = []
    for c5 in region.targets:
        r = region.K[c5 + "00000"]
        if c5 == "28720":
            rec.append((c5, region.sg_nm[c5], ong[0], ong[1], "학습기간 시료 좌표 중앙값(옹진군 예외)"))
        else:
            rec.append((c5, region.sg_nm[c5], r["lat"], r["lon"], "korea_region_code"))
    ong_info = dict(rows_all=n_all, rows_train=n_train, excluded_sido_point=n_sido, excluded_null_xy=n_nullxy,
                    used=len(la_l), lat=ong[0], lon=ong[1])
    return rec, ong_info


def main():
    region = Region(KRC)
    t = load_domestic()
    C10 = pc.cast(t["legal_dong_code"], pa.string()).to_pylist()
    ADDR = t["addr"].to_pylist()
    res, info = assign(region, C10, ADDR)
    codes = [c for c, _ in res]
    bad = {c for c in codes if c is not None and c not in region.targets}
    assert not bad, f"대상 밖 코드: {bad}"
    # 못 정한 행의 삭제 사유 (문장과 맞지 않는 행이 있으면 파일을 쓰기 전에 멈춘다)
    drop_idx = [i for i, c in enumerate(codes) if c is None]
    drop_code = {i: drop_reason(region, C10[i], ADDR[i], res[i][1]) for i in drop_idx}

    # 시료_시군구.parquet (원래 칼럼 + sgg_code). sgg_code 를 못 정한 행(None)은 넣지 않는다(사용자 결정 2026-10-16).
    keep = pa.array([c is not None for c in codes], type=pa.bool_())
    out = t.filter(keep).append_column(pa.field("sgg_code", pa.string()),
                                       pa.array([c for c in codes if c is not None], type=pa.string()))
    meta = dict(t.schema.metadata or {})
    if b"pandas" in meta:
        pm = json.loads(meta[b"pandas"])
        pm["columns"].append({"name": "sgg_code", "field_name": "sgg_code", "pandas_type": "unicode",
                              "numpy_type": "object", "metadata": None})
        meta[b"pandas"] = json.dumps(pm, ensure_ascii=False).encode("utf-8")
    out = out.replace_schema_metadata(meta)
    pq.write_table(out, OUT_SAMPLE)

    # 시군구_대표좌표.parquet
    rec, ong = representative_coords(region, t, codes)
    ct = pa.table({
        "sgg_code": pa.array([r[0] for r in rec], pa.string()),
        "sgg_nm": pa.array([r[1] for r in rec], pa.string()),
        "lat": pa.array([r[2] for r in rec], pa.float64()),
        "lon": pa.array([r[3] for r in rec], pa.float64()),
        "source": pa.array([r[4] for r in rec], pa.string()),
    })
    pq.write_table(ct, OUT_COORD)

    # 시군구_배정_요약.csv
    J = pc.cast(t["judge_result"], pa.string()).to_pylist()
    cnt = Counter()
    det = Counter()
    for c10, code, j in zip(C10, codes, J):
        k = (c10[:5], code)
        cnt[k] += 1
        det[k] += (j == "검출")
    with open(OUT_SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["원래_시군구코드", "원래_이름", "sgg_code", "sgg_nm", "바뀜", "행수", "검출수"])
        for (o, n), v in sorted(cnt.items(), key=lambda x: (x[0][0], x[0][1] or "")):
            o_nm = " ".join(words(region.K[o + "00000"]["legal_dong_nm"])) if (o + "00000") in region.K else ""
            o_nm += "" if region.K.get(o + "00000", {}).get("molit_use_yn") != "폐지" else " (폐지)"
            w.writerow([o, o_nm, n or "", region.sg_nm.get(n, "") if n else "(못 정함)",
                        "" if o == n else "Y", v, det[(o, n)]])

    # 시군구_미배정_삭제행.csv (원래 행 순서) · 시군구_미배정_삭제행_요약.csv
    KEY = t["lims_data_key"].to_pylist()
    DE = t["mnfctur_de"].to_pylist()
    with open(OUT_DROP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lims_data_key", "legal_dong_code", "addr", "mnfctur_de", "judge_result", "사유코드", "사유"])
        for i in drop_idx:
            w.writerow([KEY[i], C10[i], ADDR[i], "" if DE[i] is None else DE[i].isoformat(sep=" "), J[i],
                        drop_code[i], DROP_REASON[drop_code[i]]])

    def orig_nm(o):
        r = region.K.get(o + "00000")
        if r is None:
            return ""
        return " ".join(words(r["legal_dong_nm"])) + (" (폐지)" if r["molit_use_yn"] == "폐지" else "")

    dcnt = Counter()
    ddet = Counter()
    for i in drop_idx:
        k = (drop_code[i], C10[i][:5])
        dcnt[k] += 1
        ddet[k] += (J[i] == "검출")
    with open(OUT_DROP_SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["사유코드", "사유", "원래_시군구코드", "원래_이름", "행수", "검출수"])
        for (rc, o), v in sorted(dcnt.items()):
            w.writerow([rc, DROP_REASON[rc], o, orig_nm(o), v, ddet[(rc, o)]])

    print("domestic rows", t.num_rows, "det", sum(j == "검출" for j in J))
    print("written rows (sgg_code 있음)", out.num_rows, "det", sum(j == "검출" for c, j in zip(codes, J) if c is not None))
    print("dropped rows", len(drop_idx), "by reason", dict(sorted(Counter(drop_code.values()).items())))
    print("parents", sorted(region.parents), len(region.parents))
    print("targets", len(region.targets))
    print("rename problems", region.rename_problem)
    print("mapping", info)
    print("None rows", sum(c is None for c in codes), "det", sum(c is None and j == "검출" for c, j in zip(codes, J)))
    print("changed rows", sum(c != o[:5] for c, o in zip(codes, C10)),
          "det", sum(c != o[:5] and j == "검출" for c, o, j in zip(codes, C10, J)))
    print("ongjin", ong)


if __name__ == "__main__":
    main()
