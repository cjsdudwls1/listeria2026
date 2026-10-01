import shutil
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# 공유용 폴더(공유용_데이터3종_20261001)의 파일을 다시 만든다. 값은 바꾸지 않는다(칼럼을 옆으로 붙이기만 함).
#   2_LIMS+기상/listeria_body_wx.parquet
#       = data/listeria_body.parquet(LIMS 23칼럼) + src_route_fixed + 기상·대기 17변수(각 100칼럼) + food.parquet
#   3_LIMS+품목조인/listeria_df_joined_full.parquet
#       = data/listeria_df_joined_full.parquet 사본 (LIMS 23 + 품목 계층 14 + 의뢰 시스템 코드 3)
#   4_LIMS+품목조인+기상/listeria_body_hier_wx.parquet
#       = data/listeria_df_joined_full.parquet + src_route_fixed + 기상·대기 17변수 + food.parquet
# src_route_fixed = data/X.parquet 의 src_route (EXP/1/1_데이터정제.ipynb 에서 로마자로 적힌 한국 주소 78행을 domestic 으로 옮긴 값).
#   기상·대기 칼럼은 이 값을 기준으로 붙었다(domestic = 국내 관측소, overseas = api_data_info.global_factory_weather).
# 모든 파일의 lims_data_key 가 listeria_body.parquet 와 같은 순서인지 확인하고, 다르면 멈춘다.
# 사용: py make_share_wx_merged.py
ROOT = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML")
BODY = ROOT / "data" / "listeria_body.parquet"
JOINED = ROOT / "data" / "listeria_df_joined_full.parquet"
X_PATH = ROOT / "data" / "X.parquet"
WX_DIR = ROOT / "data" / "collected" / "exp1_wx"
SHARE = ROOT / "공유용_데이터3종_20261001"
KEY = "lims_data_key"
VARS = ["avg_temp", "max_temp", "min_temp", "rain", "avg_wind", "max_wind", "sunshine", "solar_rad",
        "soil_t5", "soil_t10", "soil_t20", "soil_t30", "soil_t_nas", "soil_moist", "no2", "co", "so2"]


def keys_of(t):
    # 파일마다 문자열 저장 형식(string/large_string)이 달라 형식을 맞춘 뒤 값·순서를 비교한다
    return t.column(KEY).combine_chunks().cast(pa.large_string())


body = pq.read_table(BODY)
keys = keys_of(body)
x = pq.read_table(X_PATH, columns=[KEY, "src_route"])
assert keys_of(x).equals(keys), "X.parquet: lims_data_key 순서가 listeria_body.parquet 와 다름"
src_fixed = x.column("src_route").cast(pa.string())
extra = []   # (이름, 칼럼) — 기상·대기·식품
for f in [WX_DIR / f"wx_{v}.parquet" for v in VARS] + [WX_DIR / "food.parquet"]:
    t = pq.read_table(f)
    assert keys_of(t).equals(keys), f"{f.name}: lims_data_key 순서가 listeria_body.parquet 와 다름"
    extra += [(name, t.column(name)) for name in t.column_names if name != KEY]
    print(f"{f.name}: {t.num_columns - 1}칼럼")


def build(base, out):
    assert keys_of(base).equals(keys), f"{out.name}: 기준 파일의 lims_data_key 순서가 다름"
    tbl = base.add_column(base.column_names.index("src_route") + 1, "src_route_fixed", src_fixed)
    for name, col in extra:
        assert name not in tbl.column_names, f"칼럼 이름 겹침: {name}"
        tbl = tbl.append_column(name, col)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out, compression="zstd", row_group_size=50_000)
    m = pq.read_metadata(out)
    print(f"저장: {out} | 행 {m.num_rows:,} × 열 {m.num_columns:,} | {out.stat().st_size / 1e6:,.0f} MB")


build(body, SHARE / "2_LIMS+기상" / "listeria_body_wx.parquet")
joined = pq.read_table(JOINED)
build(joined, SHARE / "4_LIMS+품목조인+기상" / "listeria_body_hier_wx.parquet")
dst = SHARE / "3_LIMS+품목조인" / "listeria_df_joined_full.parquet"
shutil.copy2(JOINED, dst)
m = pq.read_metadata(dst)
print(f"복사: {dst} | 행 {m.num_rows:,} × 열 {m.num_columns:,}")
