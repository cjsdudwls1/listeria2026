# -*- coding: utf-8 -*-
"""3차 보고서 수정본(sections4) Ⅰ부 새 표 '시료 단위와 단위 × 날짜 단위 견주기' 수치 (2026-10-22).

읽기만 한다: 리스테리아ML2/data/데이터셋/시료_시군구.parquet (실험이 읽는 국내·시군구 배정 시료 628,517건).
쓰는 곳: work3/out_facts/unit_date_summary_r4.csv (새 파일) + 화면 출력.

정의(실험 1차 data.build_unit_rows 와 같은 규칙)
  예측 단위 = (sgg_code, train_prdlst_code, train_prdlst_code_2), 코드가 빈 값이면 '코드없음'
  단위 × 날짜 행 = 단위 × 기준일 d(mnfctur_de), Y = 그 행 시료 중 검출(judge_result '검출')이 1건 이상
  기간: 전체 = 기준일 전부, 학습 자료 = d <= 2025-06-30, 시험 기간 = 2025-07-01 <= d <= 2026-06-30
대조(사실 모음 facts_data.md): 전체 291,812행·검출 1,141 / 학습 257,042·973 / 시험 기간 34,770·168 / 단위 6,631·검출 단위 359
실행: <리스테리아ML/.venv-exp2/Scripts/python.exe> -B 07_unit_date_summary.py
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ML2 = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2")
SAMPLE = ML2 / "data" / "데이터셋" / "시료_시군구.parquet"
OUT = ML2 / "notebook" / "보고서" / "경복대양식" / "work3" / "out_facts" / "unit_date_summary_r4.csv"
NULL_CODE = "코드없음"
LEARN_END = pd.Timestamp("2025-06-30")
TEST_START, DATA_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30")
SCORE_START = pd.Timestamp("2025-07-02")      # 시험 채점 (행, h) 에 드는 첫 기준일(T >= 2025-07-01, h >= 1)

if OUT.exists():
    raise SystemExit("이미 있는 파일 — 덮어쓰지 않음: %s" % OUT)

cols = ["judge_result", "mnfctur_de", "sgg_code", "train_prdlst_code", "train_prdlst_code_2"]
t = pq.read_table(SAMPLE, columns=cols).to_pandas()
for c in ["judge_result", "sgg_code", "train_prdlst_code", "train_prdlst_code_2"]:
    t[c] = t[c].astype(object)
t["pos"] = (t["judge_result"] == "검출").astype(int)
t["d"] = pd.to_datetime(t["mnfctur_de"]).dt.normalize()
t["p2"] = t["train_prdlst_code"].where(t["train_prdlst_code"].notna(), NULL_CODE)
t["p3"] = t["train_prdlst_code_2"].where(t["train_prdlst_code_2"].notna(), NULL_CODE)

rows = t.groupby(["sgg_code", "p2", "p3", "d"], sort=True).agg(n=("pos", "size"), npos=("pos", "sum")).reset_index()
rows["y"] = (rows["npos"] > 0).astype(int)
rows["unit"] = rows["sgg_code"].astype(str) + "|" + rows["p2"].astype(str) + "|" + rows["p3"].astype(str)

PERIODS = [("전체 기간", t["d"].min(), DATA_END), ("학습 자료", t["d"].min(), LEARN_END), ("시험 기간", TEST_START, DATA_END)]
out = []
for name, a, b in PERIODS:
    ms = (t["d"] >= a) & (t["d"] <= b)
    mr = (rows["d"] >= a) & (rows["d"] <= b)
    r = rows[mr]
    u_all = r["unit"].nunique()
    u_pos = r.loc[r["y"] == 1, "unit"].nunique()
    out.append({
        "기간": name, "기준일 처음": str(t.loc[ms, "d"].min().date()), "기준일 끝": str(t.loc[ms, "d"].max().date()),
        "시료 행 수": int(ms.sum()), "시료 검출 행": int(t.loc[ms, "pos"].sum()),
        "시료 검출률(%)": round(100.0 * t.loc[ms, "pos"].sum() / ms.sum(), 3),
        "단위x날짜 행 수": int(mr.sum()), "단위x날짜 검출 행": int(r["y"].sum()),
        "단위x날짜 검출률(%)": round(100.0 * r["y"].sum() / mr.sum(), 3),
        "행당 시료 수 평균": round(float(r["n"].mean()), 2),
        "예측 단위 수(그 기간에 행이 있는)": int(u_all), "검출이 한 번이라도 있었던 단위": int(u_pos),
        "검출 단위 비율(%)": round(100.0 * u_pos / u_all, 1),
    })
res = pd.DataFrame(out)

# 시험 채점 (행, h) 에 드는 행(기준일 2025-07-02 ~ 2026-06-30) — 참고
msc = (rows["d"] >= SCORE_START) & (rows["d"] <= DATA_END)
score_rows, score_pos = int(msc.sum()), int(rows.loc[msc, "y"].sum())

# 대조(사실 모음 값과 다르면 멈춤)
chk = {("전체 기간", "단위x날짜 행 수"): 291812, ("전체 기간", "단위x날짜 검출 행"): 1141,
       ("학습 자료", "단위x날짜 행 수"): 257042, ("학습 자료", "단위x날짜 검출 행"): 973,
       ("시험 기간", "단위x날짜 행 수"): 34770, ("시험 기간", "단위x날짜 검출 행"): 168,
       ("전체 기간", "시료 행 수"): 628517, ("전체 기간", "시료 검출 행"): 1844,
       ("학습 자료", "시료 행 수"): 541677, ("학습 자료", "시료 검출 행"): 1578,
       ("전체 기간", "예측 단위 수(그 기간에 행이 있는)"): 6631, ("전체 기간", "검출이 한 번이라도 있었던 단위"): 359}
bad = []
for (p, c), v in chk.items():
    got = int(res.loc[res["기간"] == p, c].iloc[0])
    if got != v:
        bad.append("%s %s: 계산 %d, 사실 모음 %d" % (p, c, got, v))
if (score_rows, score_pos) != (34632, 168):
    bad.append("시험 채점 행: 계산 %d·%d, 사실 모음 34,632·168" % (score_rows, score_pos))
if bad:
    raise SystemExit("대조 불일치:\n  " + "\n  ".join(bad))

res.to_csv(OUT, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(res.T.to_string())
print("시험 채점 (행, h) 에 드는 행(기준일 2025-07-02 ~ 2026-06-30): %d행·검출 %d" % (score_rows, score_pos))
print("대조: 사실 모음 값과 모두 같음")
print("저장:", OUT)
