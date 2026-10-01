"""train_prdlst_code 결측 행이 train_prdlst_nm '기타' 인지 확인 (읽기 전용, data/listeria_body.parquet)."""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet",
                     columns=["train_prdlst_code", "train_prdlst_nm", "prdlst_code", "judge_result", "mnfctur_de"])
code_na = df["train_prdlst_code"].isna()
nm = df["train_prdlst_nm"].astype("string")
print("전체 행:", len(df))
print("train_prdlst_code 결측 행:", int(code_na.sum()))
print("결측 행의 train_prdlst_nm 값:", nm[code_na].value_counts(dropna=False).to_dict())
print("train_prdlst_nm == '기타' 행:", int((nm == "기타").sum()))
print("'기타' 행의 train_prdlst_code 값:",
      df.loc[nm == "기타", "train_prdlst_code"].astype("string").value_counts(dropna=False).to_dict())
print("train_prdlst_nm 결측 행:", int(nm.isna().sum()))
print("train_prdlst_code 범주에 '기타' 가 이미 있는가:", "기타" in set(df["train_prdlst_code"].dropna().astype(str).unique()))

# 코드 ↔ 이름 1:1 여부 (결측 코드는 '(결측)' 으로 묶어 봄)
code = df["train_prdlst_code"].astype("string").fillna("(결측)")
pair = pd.DataFrame({"code": code, "nm": nm}).drop_duplicates()
print("(코드, 이름) 쌍 수:", len(pair), "| 코드 종류:", pair["code"].nunique(), "| 이름 종류:", pair["nm"].nunique())
print("이름이 2개 이상인 코드:", pair.groupby("code")["nm"].nunique().loc[lambda s: s > 1].to_dict())
print("코드가 2개 이상인 이름:", pair.groupby("nm")["code"].nunique().loc[lambda s: s > 1].to_dict())

print("결측 행의 prdlst_code 값:", df.loc[code_na, "prdlst_code"].astype("string").value_counts(dropna=False).to_dict())
print("결측 행의 judge_result:", df.loc[code_na, "judge_result"].astype(str).value_counts().to_dict())
