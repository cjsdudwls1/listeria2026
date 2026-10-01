# -*- coding: utf-8 -*-
# 노트북 6 용 t_prdlst_cl 계층 조인 셀 모음 (2026-09-21).
# 노트북에 삽입된 셀이 사라졌을 때: 빈 코드 셀에  %load nb6_t_prdlst_cl_계층조인_셀.py  를 입력해 실행하면 이 파일이 셀로 들어온다.
# 전제: 위쪽 셀에서 pandas 를 pd 로 불러왔고 df_filtered 가 만들어져 있어야 한다. DB 는 읽기 전용(SELECT 만).
# 셀 경계는 '# %%' 줄. 한 셀에 통째로 넣고 실행해도 된다.

# %%
# [1] DB 읽기 전용 접속 + 식품코드 계층표(ba_data.t_prdlst_cl) 내려받기 — SELECT 만
import os
import psycopg

PGPASS = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")   # 비밀번호는 libpq 가 이 파일에서 읽음
DB_HOST = "10.31.72.23"


def db_connect_readonly():
    """pgpass.conf 의 host 항목에서 port·dbname·user 만 읽어 읽기 전용 세션으로 접속한다."""
    with open(PGPASS, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace("\\:", "\x00").split(":")
            if len(parts) < 5:
                continue
            host, port, dbname, user = (x.replace("\x00", ":") for x in parts[:4])
            if host in (DB_HOST, "*"):
                conn = psycopg.connect(
                    host=DB_HOST, port=5432 if port == "*" else int(port), dbname=dbname, user=user,
                    options="-c default_transaction_read_only=on -c statement_timeout=120000",
                    application_name="listeria_nb6", connect_timeout=15)
                conn.autocommit = True
                return conn
    raise RuntimeError(f"pgpass.conf 에 {DB_HOST} 항목이 없습니다: {PGPASS}")


def db_query(sql, params=None):
    """SELECT 결과를 DataFrame 으로 (읽기 전용 세션, 실행마다 접속·종료)."""
    with db_connect_readonly() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


print("읽기 전용 세션:", db_query("show default_transaction_read_only").iloc[0, 0])

prdlst_cl = db_query("""
    select prdlst_cd, kor_nm, lv::int as lv, hrnk_prdlst_cd, htrk_prdlst_cd,
           prdlst_yn, use_yn, vald_begn_dt, vald_end_dt
    from ba_data.t_prdlst_cl
    order by prdlst_cd
""")
print(f"t_prdlst_cl: {len(prdlst_cl):,}행 / prdlst_cd 고유 {prdlst_cl['prdlst_cd'].nunique():,}")
print("레벨(lv)별 코드 수:", prdlst_cl["lv"].value_counts().sort_index().to_dict())
display(prdlst_cl.head())

# %%
# [2] 계층 전개: 코드마다 상위 코드(hrnk_prdlst_cd)를 따라 1단계(최상위)까지 올라가 단계별 코드·한글명 컬럼을 만든다
parent = dict(zip(prdlst_cl["prdlst_cd"], prdlst_cl["hrnk_prdlst_cd"]))
kor    = dict(zip(prdlst_cl["prdlst_cd"], prdlst_cl["kor_nm"]))
level  = dict(zip(prdlst_cl["prdlst_cd"], prdlst_cl["lv"]))
MAX_LV = int(prdlst_cl["lv"].max())


def code_chain(code):
    """code 에서 상위 코드를 따라 최상위까지: {1: 최상위코드, 2: ..., lv: code}. 상위 코드가 표에 없으면 거기서 멈춘다."""
    chain, seen = {}, set()
    while code in level and code not in seen:
        seen.add(code)
        chain[level[code]] = code
        p = parent.get(code)
        if not p or p == code:
            break
        code = p
    return chain


rows = []
for code in prdlst_cl["prdlst_cd"]:
    ch = code_chain(code)
    rec = {"prdlst_cd": code}
    for lv in range(1, MAX_LV + 1):
        rec[f"lv{lv}_cd"] = ch.get(lv)
        rec[f"lv{lv}_nm"] = kor.get(ch[lv]) if lv in ch else None
    rec["chain_top_lv"] = min(ch) if ch else None   # 체인이 실제로 닿은 가장 위 레벨 (1 이면 최상위까지 완결)
    rows.append(rec)
prdlst_hier = prdlst_cl.merge(pd.DataFrame(rows), on="prdlst_cd")

incomplete = prdlst_hier[prdlst_hier["chain_top_lv"] != 1]
print(f"최상위(1단계)까지 닿지 않는 코드: {len(incomplete):,}개 (상위 코드가 표에 없는 경우 — 아래 25개는 우리 품목과 무관)")
display(incomplete[["prdlst_cd", "kor_nm", "lv", "hrnk_prdlst_cd", "chain_top_lv"]])

print("1단계(최상위) 코드:")
display(prdlst_hier.loc[prdlst_hier["lv"] == 1, ["prdlst_cd", "kor_nm"]])

# %%
# [3] df_filtered 에 조인 — 키는 train_prdlst_code (train_prdlst_nm 은 표에서 이름이 중복되므로 키로 쓰지 않음)
hier_cols = ["prdlst_cd", "kor_nm", "lv"] + [f"lv{l}_{k}" for l in range(1, MAX_LV + 1) for k in ("cd", "nm")]
join_tbl = (prdlst_hier[hier_cols]
            .rename(columns={"prdlst_cd": "train_prdlst_code", "kor_nm": "cl_kor_nm", "lv": "cl_lv"}))

df_joined = df_filtered.copy()
df_joined["train_prdlst_code"] = df_joined["train_prdlst_code"].astype(str)
df_joined = df_joined.merge(join_tbl, on="train_prdlst_code", how="left", validate="many_to_one")

matched   = df_joined["cl_lv"].notna()
name_same = df_joined["train_prdlst_nm"].astype(str) == df_joined["cl_kor_nm"].astype(str)
print(f"조인 행 수: {len(df_joined):,} (df_filtered 와 동일: {len(df_joined) == len(df_filtered)})")
print(f"코드 매칭: {matched.mean() * 100:.2f}%  /  train_prdlst_nm 과 표 한글명(kor_nm) 동일: {name_same.mean() * 100:.2f}%")

# 이름을 키로 쓰면 안 되는 근거: 같은 한글명이 표에 여러 코드로 있다
names15 = df_filtered["train_prdlst_nm"].astype(str).unique()
dup_names = (prdlst_cl[prdlst_cl["kor_nm"].isin(names15)]
             .groupby("kor_nm")["prdlst_cd"].agg(코드수="size", 코드=lambda s: ", ".join(s)))
print("\n15개 품목명 중 표에 코드가 2개 이상인 이름:")
display(dup_names[dup_names["코드수"] > 1])

# %%
# [4] 현재 품목명 15개가 어느 단계인지 + 최상위까지의 한글명 경로 (df_filtered 기준 행 수·검출)
lvl_map = (df_joined.groupby(["train_prdlst_code", "train_prdlst_nm"], observed=True)
           .agg(행수=("judge_result", "size"), 검출=("judge_result", lambda s: int((s == "검출").sum())))
           .reset_index()
           .merge(join_tbl, on="train_prdlst_code", how="left"))
lvl_map["검출률(%)"] = (lvl_map["검출"] / lvl_map["행수"] * 100).round(3)
lvl_map["계층 경로(한글명)"] = lvl_map.apply(
    lambda r: " > ".join(str(r[f"lv{l}_nm"]) for l in range(1, MAX_LV + 1) if pd.notna(r[f"lv{l}_nm"])), axis=1)
lvl_map = lvl_map.sort_values(["lv1_cd", "cl_lv", "train_prdlst_code"]).reset_index(drop=True)

show_cols = ["train_prdlst_code", "train_prdlst_nm", "cl_lv", "lv1_cd", "lv1_nm", "lv2_nm", "lv3_nm",
             "계층 경로(한글명)", "행수", "검출", "검출률(%)"]
pd.set_option("display.max_colwidth", 60)
display(lvl_map[show_cols])
print("품목이 속한 단계(cl_lv)별 품목 수:", lvl_map["cl_lv"].value_counts().sort_index().to_dict())
# 파일 저장이 필요하면 주석 해제
# lvl_map[show_cols].to_csv(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\docs\00_보고서\표_품목15개_코드계층.csv", index=False, encoding="utf-8-sig")

# %%
# [5] 라벨링한 코드 계층별 유니크 값 — 표 전체 vs df_filtered 에 실제로 나오는 값
uniq = []
for l in range(1, MAX_LV + 1):
    in_tbl = prdlst_hier[f"lv{l}_cd"].dropna().nunique()
    in_df  = df_joined[f"lv{l}_cd"].dropna().nunique()
    vals   = (df_joined[[f"lv{l}_cd", f"lv{l}_nm"]].dropna().drop_duplicates()
              .sort_values(f"lv{l}_cd").apply(lambda r: f"{r.iloc[0]} {r.iloc[1]}", axis=1).tolist())
    uniq.append({"단계(lv)": l, "표 전체 유니크 코드 수": in_tbl, "df_filtered 유니크 코드 수": in_df,
                 "df_filtered 에 나오는 값": ", ".join(vals) if vals else "(없음 — 품목이 2·3단계까지만 있음)"})
pd.set_option("display.max_colwidth", 400)
display(pd.DataFrame(uniq))

# 최상위(1단계) 분류별 행 수·검출·검출률 — 가공식품(C) 만이 아니라 Z(기준규격외)·B(동물성)·A(식물성) 으로 나눠 본다
top = (df_joined.groupby(["lv1_cd", "lv1_nm"], dropna=False)
       .agg(행수=("judge_result", "size"), 검출=("judge_result", lambda s: int((s == "검출").sum())))
       .reset_index())
top["검출률(%)"]   = (top["검출"] / top["행수"] * 100).round(3)
top["검출비중(%)"] = (top["검출"] / top["검출"].sum() * 100).round(1)
display(top.sort_values("검출", ascending=False).reset_index(drop=True))
