import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

# 국민건강영양조사 2024(hn24_all·hn24_24rc) 전체 컬럼 목록 + 리스테리아용 선정 컬럼 → xlsx, 선정 근거 수치 출력.
# 라벨은 SAS 파일에 들어 있는 변수 라벨(euc-kr)이다. 코드값의 뜻(예 region 1~17)은 SAS 파일에 없어 이용지침서 확인이 필요하다.
# 입력: knhanes_to_parquet.py 산출 parquet, 원본 SAS(라벨), data/listeria_df_joined_full.parquet(이름 대조)
SRC = Path(r"C:\Users\NT751\Downloads\ㅇ")
PQ = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\knhanes_2024")
LIS = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_df_joined_full.parquet")
OUT = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\02_데이터수집\국민건강영양조사2024_컬럼목록.xlsx")
FILES = {"hn24_24rc": SRC / "HN24_24RC(SAS)" / "hn24_24rc.sas7bdat",
         "hn24_all": SRC / "HN24_ALL(SAS)" / "hn24_all.sas7bdat"}


def dec(x):
    # pandas 가 라벨을 latin-1 로 푼 str 로 주므로 bytes 로 되돌려 cp949 로 다시 푼다
    if isinstance(x, str):
        x = x.encode("latin-1")
    return (x or b"").decode("cp949", errors="replace").strip()


def fmtv(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v)
    return s if len(s) <= 30 else s[:30] + "…"


def catalog(key):
    rdr = pd.read_sas(FILES[key], format="sas7bdat", iterator=True)
    meta = pd.DataFrame([dict(컬럼=dec(c.name), SAS라벨=dec(c.label),
                              SAS형식="문자" if c.ctype == b"s" else "숫자", SAS길이=c.length)
                         for c in rdr.columns])
    rdr.close()
    df = pd.read_parquet(PQ / f"{key}.parquet")
    assert list(df.columns) == list(meta["컬럼"])
    rows = []
    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        r = dict(비결측=nn, 비결측pct=round(nn / len(df) * 100, 1), 고유값=int(s.nunique()),
                 상위값="; ".join(f"{fmtv(k)}({v:,})" for k, v in s.value_counts().head(6).items()),
                 최소="", 최대="")
        if pd.api.types.is_numeric_dtype(s) and nn:
            r["최소"], r["최대"] = fmtv(s.min()), fmtv(s.max())
        rows.append(r)
    cat = pd.concat([meta, pd.DataFrame(rows)], axis=1)
    cat.insert(0, "순서", range(1, len(cat) + 1))
    return df, cat


rc, cat_rc = catalog("hn24_24rc")
al, cat_al = catalog("hn24_all")
lis = pd.read_parquet(LIS, columns=["prduct_nm", "mnfctur_cmpny_nm", "judge_result", "train_prdlst_code"])
lis["jr"] = lis["judge_result"].astype(str)

# ---------------------------------------------------------------- 선정 근거 수치
print("hn24_24rc", rc.shape, "대상자", rc["ID"].nunique(), "/ hn24_all", al.shape)
print("조사 날짜·월 컬럼: 라벨에 '요일'만 있음 →", [c for c, l in zip(cat_rc["컬럼"], cat_rc["SAS라벨"]) if "요일" in l or "일자" in l or "날짜" in l])
print("리스테리아 train_prdlst_code 길이:", lis["train_prdlst_code"].astype(str).str.len().value_counts().head(3).to_dict(),
      "첫 글자:", lis["train_prdlst_code"].astype(str).str[0].value_counts().to_dict())
print("N_FCODE·N_DCODE 길이:", rc["N_FCODE"].str.len().value_counts().to_dict(), rc["N_DCODE"].str.len().value_counts().to_dict())
print("N_KINDG1 != N_FCODE 앞 2자리 행:", int((rc["N_KINDG1"] != rc["N_FCODE"].str[:2]).sum()))
print("식품코드당 3차 코드가 2개 이상인 식품코드:", int((rc.groupby("N_FCODE")["N_FCODE3"].nunique() > 1).sum()), "/", rc["N_FCODE"].nunique())
for k in ["소고기", "돼지고기", "닭고기", "우유", "달걀", "고등어", "쌀", "마늘", "양파", "배추"]:
    print("  N_AP", k, rc.loc[rc["N_FNAME"].str.startswith(k), "N_AP"].value_counts().to_dict())
x = rc[rc["NF_INTK"] > 0]
print("NF_INTK == 0 행:", int((rc["NF_INTK"] == 0).sum()))
for c in ["NF_WATER", "NF_NA", "NF_PROT", "NF_FAT", "NF_CHO", "NF_SUGAR"]:
    rng = (x[c] / x["NF_INTK"]).groupby(x["N_FCODE"]).agg(lambda s: s.max() - s.min())
    print(f"  {c}/NF_INTK 가 식품코드 안에서 일정(차이<1e-6)한 코드 {int((rng < 1e-6).sum())}/{len(rng)}")
last = rc["N_FNAME"].str.split(",").str[-1].str.strip()
print("N_FNAME 쉼표 뒤 낱말(행, 코드 수):",
      {k: (int(v), int(rc.loc[last == k, "N_FCODE"].nunique())) for k, v in last.value_counts().items()
       if k in ["생것", "말린것", "볶은것", "삶은것", "데친것", "구운것", "훈제"]})
print("'마늘, 생것' 이 든 음식:", rc.loc[rc["N_FNAME"] == "마늘, 생것", "N_DNAME"].value_counts().head(6).to_dict())
fd = rc["N_FD_NS"].dropna()
print("N_FD_NS 비결측", len(fd), {k: int(fd.str.contains(k).sum()) for k in ["냉동", "냉장", "훈제", "살균", "통조림", "캔"]})
for k in ["팽이버섯", "연어", "치즈", "가공육", "새싹채소", "어패류젓", "어패류액젓", "두부", "어묵", "우유", "아이스크림", "요구르트"]:
    m = rc["N_FNAME3"] == k
    print(f"  3차 {k}: {int(m.sum())}행 {rc.loc[m, 'ID'].nunique()}명 식품코드 {rc.loc[m, 'N_FCODE'].nunique()}개")
print("  '연어' 세부:", rc.loc[rc["N_FNAME3"] == "연어", "N_FNAME"].value_counts().to_dict())
for k in ["김밥", "샐러드", "샌드위치", "초밥", "육회"]:
    m = rc["N_DNAME"].str.contains(k)
    print(f"  음식명 '{k}': {int(m.sum())}행 {rc.loc[m, 'ID'].nunique()}명")
for a, b in [("N_PRODUCT", "prduct_nm"), ("N_CNAME", "mnfctur_cmpny_nm")]:
    kn = rc[a].dropna()
    ku = set(kn[kn != "888"].unique())
    lu = set(lis[b].dropna().unique())
    ex = ku & lu
    ex2 = {v.replace(" ", "") for v in ku} & {v.replace(" ", "") for v in lu}
    m = lis[b].isin(ex)
    print(f"{a}: 값 있는 행 {len(kn)}({len(kn) / len(rc) * 100:.1f}%), '888' {int((kn == '888').sum())}행, 고유 {len(ku)}개 · "
          f"리스테리아 {b} 와 원문 일치 {len(ex)}개(공백 제거 {len(ex2)}개) → 시료 {int(m.sum())}행, judge_result",
          lis.loc[m, "jr"].value_counts().to_dict(), "검출 시료 이름", lis.loc[m & (lis["jr"] == "검출"), b].value_counts().to_dict())
for k in ["N_KINDG1"]:
    for g, grp in rc.groupby(k):
        print(f"  {k} {g}: {len(grp)}행", grp["N_FNAME3"].value_counts().head(3).index.tolist())
for c in ["region", "N_MEAL", "N_MEAL_P", "N_MTYPE", "N_AP"]:
    print(" ", c, rc[c].value_counts().sort_index().to_dict())
for c in ["HE_prg", "HE_dprg", "DC01_dg", "DE1_dg", "DN1_dg", "HE_hepaB", "HE_hepaC"]:
    print(" ", c, al[c].value_counts(dropna=False).sort_index().to_dict())
print("  age 최댓값", al["age"].max(), "· age==80", int((al["age"] == 80).sum()), "명 · age 79", int((al["age"] == 79).sum()), "명")

# ---------------------------------------------------------------- 선정 컬럼
A, B, C, D, E = ("A 식품 연결·특성", "B 섭취량·성분", "C 집계·연결", "D 섭취 맥락(참고)", "E 고위험군(모델 변수 아님)")
RC, AL, BOTH = "hn24_24rc", "hn24_all", "두 파일"
SEL = [
    (A, RC, "N_FCODE", "리스테리아 품목과 잇는 기준 후보 — 대응표를 새로 만들어야 함",
     "식품코드 2,301개, 모두 숫자 5자리. 리스테리아 파일 품목 코드(train_prdlst_code, 영문 1자+숫자 13자리)와 체계가 다름",
     "식품코드 ↔ 식품공전 품목코드 대응표가 식약처·식정원에 있는지"),
    (A, RC, "N_FNAME", "세부 식품과 재료 상태 구분(생것·훈제·말린것 등)",
     "'연어, 생것' 42행 · '연어, 훈제' 38행, 3차 코드 '치즈' 아래 식품코드 20개. 쉼표 뒤 상태 낱말: 생것 221,226행(식품코드 464개) · 말린것 17,114 · 볶은것 17,003 · 삶은것 5,960 · 데친것 4,545 · 구운것 3,015",
     "'생것'은 조리 전 재료 상태로 보임('마늘, 생것'이 소고기미역국·된장찌개에도 들어감) — 가열 없이 먹었는지는 음식명으로 따로 판단해야 함"),
    (A, RC, "N_FCODE3", "품목 대응표를 만들 단위 후보(식품코드보다 굵은 묶음)",
     "487개. 식품코드 하나는 3차 코드 하나에만 속함(2,301개 모두)", ""),
    (A, RC, "N_FNAME3", "N_FCODE3 의 이름",
     "예: 팽이버섯 1,385행/1,177명 · 가공육(햄·소시지·베이컨, 식품코드 18개) · 치즈 1,990행 · 새싹채소 382행 · 어패류젓/액젓 · 두부 · 어묵 · 우유 · 아이스크림 · 요구르트. '햄'이라는 3차 이름은 없고 '가공육'에 들어 있음", ""),
    (A, RC, "N_KINDG1", "식품군 단위 집계",
     "20개 코드. 식품코드 앞 2자리와 다름(262,251행 불일치). 값 예: 07=표고·새송이·팽이버섯, 15=돼지고기·소고기·가공육, 17=멸치·어패류액젓·새우, 18=우유·치즈·요구르트",
     "코드별 식품군 이름 — 이용지침서"),
    (A, RC, "N_AP", "동물성·식물성 구분",
     "1: 470,896행, 2: 72,886행. 소고기·돼지고기·닭고기·우유·달걀·고등어 행은 2(소고기 6행만 1), 쌀·마늘·양파·배추 행은 1",
     "1·2 뜻 — 이용지침서"),
    (A, RC, "N_DNAME", "재료가 어떤 음식으로 먹혔는지 → 가열 없이 먹는 음식(RTE)인지 판단 근거",
     "음식명 12,871개. 낱말 포함(행/대상자): 김밥 11,732/422 · 샐러드 8,675/728 · 샌드위치 3,992/270 · 초밥 4,042/115 · 육회 840/48. 행은 재료 행이라 먹은 횟수와 다름", ""),
    (A, RC, "N_DCODE", "음식 단위 묶음", "음식코드 4,464개, 모두 숫자 5자리", "음식코드 체계(앞자리 뜻) — 이용지침서·음식코드표"),
    (A, RC, "N_FD_NS", "재료 상태(냉동·훈제·통조림) 보조",
     "19.5% 행만 값. 응답 그대로의 자유 기술(예 '냉동떡갈비', '참치통조림 기름제외', '삶은달걀'). 낱말 포함 행: 냉동 333 · 훈제 55 · 통조림 87 · 캔 240, 냉장·살균 0", ""),
    (B, RC, "NF_INTK", "식품(코드)별 섭취량·섭취 인구", "모든 행 값 있음, 0 없음(라벨 단위 g)", ""),
    (B, RC, "NF_WATER", "식품코드별 수분 함량(g/100g) → 수분활성도 대용 후보(추정 — 확인 필요)",
     "NF_WATER÷NF_INTK 가 같은 식품코드 안에서 모두 같음(2,301개 코드 전부) → 식품코드마다 고정된 성분값에서 나온 수치",
     "수분 함량을 수분활성도 대신 써도 되는지(전문가)"),
    (B, RC, "NF_NA", "식품코드별 나트륨 함량 → 염도 대용 후보(추정 — 확인 필요)",
     "NF_NA÷NF_INTK 도 식품코드마다 일정(2,301개 전부)", "나트륨 함량을 염도 대신 써도 되는지(전문가)"),
    (B, RC, "NF_PROT", "식품코드별 성분(선택)", "÷NF_INTK 가 식품코드마다 일정(2,301개 전부)", ""),
    (B, RC, "NF_FAT", "식품코드별 성분(선택)", "÷NF_INTK 가 식품코드마다 일정(2,301개 전부)", ""),
    (B, RC, "NF_CHO", "식품코드별 성분(선택)", "÷NF_INTK 가 식품코드마다 일정(2,301개 전부)", ""),
    (B, RC, "NF_SUGAR", "식품코드별 성분(선택)", "÷NF_INTK 가 식품코드마다 일정(2,301개 전부)", ""),
    (C, BOTH, "ID", "대상자 수 세기, 두 파일 연결", "hn24_24rc 6,802명(식품 재료 1개가 1행), hn24_all 6,997명(1명 1행)", ""),
    (C, BOTH, "wt_ntr", "국민 전체 섭취량 추정(가중 집계)", "라벨 '영양조사 가중치'. hn24_all 97.3% 행에 값", "가중 집계 방법 — 이용지침서"),
    (C, BOTH, "kstrata", "가중 집계의 표준오차 계산", "라벨 '분산추정 층', 27개", ""),
    (C, BOTH, "psu", "가중 집계의 표준오차 계산", "라벨 '조사구번호', 192개", ""),
    (C, BOTH, "region", "시도별 집계(리스테리아 시료 주소의 시도와 잇는다면)", "코드 1~17, 시도 이름은 파일에 없음", "코드 ↔ 시도 이름 — 이용지침서"),
    (C, BOTH, "sex", "성별 섭취량", "1·2", "1·2 뜻 — 이용지침서"),
    (C, BOTH, "age", "나이대(고령 등)별 섭취량", "최댓값 80, 80 이 426명으로 가장 많음(hn24_all)", "80 이 '80세 이상'을 묶은 값인지 — 이용지침서"),
    (D, RC, "N_MEAL", "끼니별 섭취 맥락(배경 설명용)", "값 1~4", "코드 뜻 — 이용지침서"),
    (D, RC, "N_MEAL_P", "집·밖 등 먹은 장소(배경 설명용)", "값 1·2·5·7·11·98", "코드 뜻 — 이용지침서"),
    (D, RC, "N_MTYPE", "사 먹은 음식인지(배경 설명용)", "라벨 '매식 여부', 값 1·5·12·14·21", "코드 뜻 — 이용지침서"),
    (D, RC, "N_PRODUCT", "가공식품 제품 단위 대조(효과 작음)",
     "값 있는 행 3.2%, '888' 4,756행. 리스테리아 prduct_nm 과 원문이 같은 이름 347개(공백 빼면 520개) → 시료 6,030행, 그중 검출 2행(둘 다 '도시락')",
     "'888' 뜻 — 이용지침서"),
    (D, RC, "N_CNAME", "제조회사 단위 대조(효과 작음)",
     "값 있는 행 3.3%, '888' 4,716행. 리스테리아 mnfctur_cmpny_nm 과 원문이 같은 이름 207개(공백 빼면 228개) → 시료 24,897행, 그중 검출 7행",
     "'888' 뜻 — 이용지침서"),
]
HR = ("리스테리아 고위험군(임신부·고령·면역저하자로 흔히 꼽힘) 규모 — 시료와 이을 키가 없어 위해평가·보고서 배경용",
      "고위험군 기준을 이 변수들로 잡아도 되는지(전문가), 코드 뜻 — 이용지침서")
for c, fact in [("HE_prg", "0: 3,911 · 8: 3,070 · 1: 16"), ("HE_dprg", "값 있는 행 0.2%(16명)"),
                ("DC01_dg", "0: 5,112 · 1: 380 · 8: 957 · 9: 539"), ("DE1_dg", "0: 5,209 · 1: 821 · 8: 957 · 9: 1"),
                ("DN1_dg", "0: 5,354 · 1: 139 · 8: 957 · 9: 538"), ("HE_hepaB", "0: 6,158 · 1: 137"), ("HE_hepaC", "0: 6,247 · 1: 48")]:
    SEL.append((E, AL, c, HR[0], fact, HR[1]))

sel = pd.DataFrame(SEL, columns=["구분", "파일", "컬럼", "쓸 곳(제안)", "확인한 사실(값)", "확인 필요"])
lab = pd.concat([cat_rc.assign(파일=RC), cat_al.assign(파일=AL)])
look = cat_rc.set_index("컬럼")
look_al = cat_al.set_index("컬럼")
sel["SAS 라벨"] = [(look if f != AL else look_al).loc[c, "SAS라벨"] for f, c in zip(sel["파일"], sel["컬럼"])]
src = [(look if f != AL else look_al).loc[c] for f, c in zip(sel["파일"], sel["컬럼"])]
sel["비결측%"] = [s["비결측pct"] for s in src]
sel["고유값"] = [s["고유값"] for s in src]
sel["상위값"] = [s["상위값"] for s in src]
sel = sel[["구분", "파일", "컬럼", "SAS 라벨", "쓸 곳(제안)", "확인한 사실(값)", "확인 필요", "비결측%", "고유값", "상위값"]]


def reason(key, c):
    if key == RC:
        if c == "N_DAY":
            return "제외 — 조사 날짜가 아니라 요일만 있음(두 파일 모두 조사 날짜·월 컬럼 없음)"
        if c == "N_ENAME":
            return "제외 — N_FNAME 의 영문판"
        if c in ("NF_INTK3", "N_CD_VOL", "N_CD_WT", "N_TD_VOL", "N_TD_WT", "N_FM_WT"):
            return "제외 — 음식 단위 조리량·섭취량이나 3차 코드용 섭취량. 식품 단위 섭취량은 NF_INTK 로 충분"
        if c.startswith("NF_"):
            return "제외 — 세부 영양소(지방산·비타민·무기질 등). 리스테리아와 연결할 근거 없음"
        return "제외 — 사람·가구 정보(소득·교육·식사요법·수유 등)나 조사 관리용"
    return "제외 — 사람 단위 건강·생활 설문·검진 값. 식품 시료나 식품코드와 이을 키가 없음"


for key, cat in [(RC, cat_rc), (AL, cat_al)]:
    chosen = dict(zip(sel.loc[sel["파일"].isin([key, BOTH]), "컬럼"], sel.loc[sel["파일"].isin([key, BOTH]), "구분"]))
    cat["구분"] = [chosen.get(c, "제외") for c in cat["컬럼"]]
    cat["비고"] = ["선정 컬럼 시트 참고" if c in chosen else reason(key, c) for c in cat["컬럼"]]

readme = pd.DataFrame({"항목": [
    "만든 날", "원자료", "변환 파일", "행 단위", "컬럼 설명(SAS 라벨)", "코드값의 뜻", "조사 시기", "리스테리아 시료와 연결",
    "구분 A~E", "만든 코드"], "내용": [
    "2026-09-28",
    r"사용자 제공 국민건강영양조사 2024 SAS 파일: C:\Users\NT751\Downloads\ㅇ\HN24_ALL(SAS)\hn24_all.sas7bdat, HN24_24RC(SAS)\hn24_24rc.sas7bdat",
    "리스테리아ML/data/collected/knhanes_2024/hn24_all.parquet(6,997행×813열), hn24_24rc.parquet(543,782행×110열) — scripts/knhanes_to_parquet.py, 값 변환 없음",
    "hn24_all = 1명 1행. hn24_24rc = 1명이 하루 동안 먹은 음식의 재료 1개가 1행(대상자 6,802명)",
    "SAS 파일 안의 변수 라벨을 그대로 옮김(파일 헤더 인코딩 euc-kr). 라벨이 빈 컬럼 없음",
    "SAS 파일에 없음(형식 카탈로그 미제공). 코드별 뜻은 국민건강영양조사 원시자료 이용지침서로 확인해야 함",
    "두 파일 모두 2024년 한 해. 조사 날짜·월 컬럼은 없고 요일(N_DAY)만 있음 → 계절·기상과 연결 불가",
    "공통 키 없음. 식품 단위로 잇는다면 식품코드(N_FCODE/N_FCODE3) ↔ 식품공전 품목코드 대응표가 필요. 제품명·회사명 원문 일치는 시료 검출 2·7행만 걸림",
    "A 식품 연결·특성 / B 섭취량·성분 / C 집계·연결 / D 섭취 맥락(참고) / E 고위험군(모델 변수 아님). '쓸 곳'은 제안이고 사용 여부는 사용자 결정",
    "리스테리아ML/scripts/knhanes_columns.py (이 파일과 선정 근거 수치 출력)"]})

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="읽는 법", index=False)
    sel.to_excel(xw, sheet_name="선정 컬럼", index=False)
    for key, cat in [(RC, cat_rc), (AL, cat_al)]:
        cat.rename(columns={"SAS라벨": "SAS 라벨", "비결측pct": "비결측%"}).to_excel(xw, sheet_name=f"{key} 전체", index=False)
    widths = {"항목": 18, "내용": 110, "구분": 22, "파일": 11, "컬럼": 13, "SAS 라벨": 28, "쓸 곳(제안)": 40,
              "확인한 사실(값)": 60, "확인 필요": 36, "상위값": 50, "비고": 50, "순서": 6}
    wrap = {"내용", "쓸 곳(제안)", "확인한 사실(값)", "확인 필요", "상위값", "비고", "SAS 라벨"}
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        for col in ws.columns:
            h = str(col[0].value)
            ws.column_dimensions[col[0].column_letter].width = widths.get(h, 10)
            col[0].font = Font(bold=True)
            col[0].fill = PatternFill("solid", fgColor="DDEBF7")
            for cell in col[1:]:
                cell.alignment = Alignment(wrap_text=h in wrap, vertical="top")
print("\n저장:", OUT, "선정", len(sel), "행 (구분별", sel["구분"].value_counts().sort_index().to_dict(), ")")
