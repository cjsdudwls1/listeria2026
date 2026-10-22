# -*- coding: utf-8 -*-
"""3차 보고서 I 부 보충(2026-10-22): 기상 원자료 관측망별 관측소 수·기간(03_시군구_기상.py 와 같은 관측소 고르기 규칙), 비율 계산.
읽기만 함. 출력 work3/out_facts/stations.json
실행: <.venv-exp2 python> -B 04_station_counts.py
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\data\raw")
OUT = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\notebook\보고서\경복대양식\work3\out_facts")
R = {}
a = pq.read_table(RAW / "raw_kma_asos_daily_info.parquet", columns=["stn_id", "meas_date"])
R["ASOS"] = {"table": "api_data_info.kma_asos_daily_info", "rows": a.num_rows, "stations": int(pc.count_distinct(a["stn_id"]).as_py()),
             "date_min": str(pc.min(a["meas_date"]).as_py()), "date_max": str(pc.max(a["meas_date"]).as_py())}
n = pq.read_table(RAW / "raw_nas_agri_weather_daily_info.parquet", columns=["obsr_spot_code", "meas_date"])
nas_ids = set(pc.unique(n["obsr_spot_code"]).to_pylist())
R["NAS"] = {"table": "api_data_info.nas_agri_weather_daily_info", "rows": n.num_rows, "stations_all": len(nas_ids),
            "stations_used": len(nas_ids - {"561034A001"}), "excluded": "561034A001(동방로거테스트, 사용자 결정 2026-10-06)",
            "date_min": str(pc.min(n["meas_date"]).as_py()), "date_max": str(pc.max(n["meas_date"]).as_py())}
k = pq.read_table(RAW / "raw_keco_airkorea_fine_dust_info.parquet", columns=["station_nm", "msur_dt"])
ks = pq.read_table(RAW / "raw_keco_airkorea_obs_station.parquet", columns=["station_nm", "dm_x", "dm_y"])
k_ids = set(pc.unique(k["station_nm"]).to_pylist())
k_xy = set(ks["station_nm"].to_pylist())
R["AIR"] = {"table": "api_data_info.keco_airkorea_fine_dust_info", "rows": k.num_rows, "stations_in_daily": len(k_ids),
            "stations_with_coord": len(k_ids & k_xy), "stations_without_coord": sorted(k_ids - k_xy),
            "date_min": str(pc.min(k["msur_dt"]).as_py()), "date_max": str(pc.max(k["msur_dt"]).as_py())}
# 비율(보고용 반올림 전 값)
R["rates"] = {
    "sample_det_rate_pct": 1844 / 628517 * 100,
    "unitday_pos_rate_pct": 1141 / 291812 * 100,
    "learn_pos_rate_pct": 973 / 257042 * 100,
    "testperiod_pos_rate_pct": 168 / 34770 * 100,
    "fold1_pair_pos_rate_pct": 6708 / 1601116 * 100,
    "fold2_pair_pos_rate_pct": 6474 / 1654771 * 100,
    "fold3_pair_pos_rate_pct": 6028 / 1703850 * 100,
    "test_pair_pos_rate_pct": 8080 / 1737545 * 100,
    "overseas_share_pct": 47642 / 676520 * 100,
    "domestic_det_share_pct": 1844 / 1969 * 100,
    "date_basis_request_pct": 332772 / 628517 * 100,
    "date_basis_mnf_pct": 295736 / 628517 * 100,
    "learn_rows_share_of_all_pct": 257042 / 291812 * 100,
}
(OUT / "stations.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
