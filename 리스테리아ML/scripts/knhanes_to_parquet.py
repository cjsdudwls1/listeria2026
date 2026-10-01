import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd

# 국민건강영양조사 2024 SAS 원자료(사용자 제공, 다운로드 폴더) → parquet. 값은 바꾸지 않는다.
# 문자열은 bytes 로 읽어 cp949 로 푼다(SAS 파일 헤더 인코딩 euc-kr).
# cp949 로 풀리지 않는 칸은 U+FFFD(�)로 바꾸고 칸 수를 출력한다
# (2026-09-28: hn24_24rc 39칸 — N_DNAME 32·N_PRODUCT 6·N_CNAME 1, 모두 문자열 끝 글자가 잘린 것).
# 사용: py knhanes_to_parquet.py [출력 폴더]   (기본 data/collected/knhanes_2024)
SRC = Path(r"C:\Users\NT751\Downloads\ㅇ")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\knhanes_2024")
FILES = [SRC / "HN24_ALL(SAS)" / "hn24_all.sas7bdat", SRC / "HN24_24RC(SAS)" / "hn24_24rc.sas7bdat"]

OUT.mkdir(parents=True, exist_ok=True)
for f in FILES:
    bad = {}

    def dec(b, col):
        try:
            return b.decode("cp949")
        except UnicodeDecodeError:
            bad[col] = bad.get(col, 0) + 1
            return b.decode("cp949", errors="replace")

    parts = []
    for ch in pd.read_sas(f, chunksize=200_000):
        for c in ch.columns:
            if ch[c].dtype == object:
                ch[c] = ch[c].map(lambda b, c=c: dec(b, c) if isinstance(b, bytes) else b)
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    dst = OUT / f"{f.stem}.parquet"
    df.to_parquet(dst, index=False)
    print(f"{f.name} -> {dst}: 행 {len(df):,} / 열 {df.shape[1]} / cp949 로 안 풀린 칸 {bad or 0}")
