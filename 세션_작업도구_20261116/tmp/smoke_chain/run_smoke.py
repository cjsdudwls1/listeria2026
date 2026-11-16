"""연기 시험(임시 폴더만): 실제 자료 조각으로 2차 사슬 트리 묶음(XGB·LGB·CAT 반복 수 고르기 + RF·ET) 한 번.
SPI·결측 표시 목록이 로컬 사본에 없어 WX1·로지스틱은 뺌. 결과는 이 폴더 아래 out/ 에만."""
import sys
import time
from pathlib import Path

CODE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, str(CODE))
from lst.resources import set_thread_env  # noqa: E402

set_thread_env()
from lst import evaluate as ev  # noqa: E402
from lst import registry as rg  # noqa: E402
from lst import runner as rn  # noqa: E402
from lst import wiring as wr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\NT751\Desktop\천영진")
INPUTS = Path(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror")
OUT = HERE / "out" / time.strftime("%H%M%S")
ctx_cfg = {
    "results": {"samples": "s0/0-1/apply/samples.parquet",
                "domestic_weather": "s0/0-4_v2/시군구_일별_기상_0-4_합본.parquet",
                "overseas_weather": "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet",
                "food_table": "s0/0-6_static/식품_고정표.parquet", "livestock": "s0/0-6_static/축사_L1.parquet"},
    "data": {"soil": "리스테리아ML2/data/데이터셋/식정원_db_토양/시군구_일별_토양.parquet",
             "global_soil_ph": "리스테리아ML2/data/추출_20261107/global_soil_ph.parquet",
             "eco": "리스테리아ML2/data/데이터셋/외부_지표/친환경_인증비율.parquet",
             "gw": "리스테리아ML2/data/데이터셋/외부_지표/지하수_정기수질검사.parquet"},
    "y_cut": "2025-06-30"}
ids = [i for i in rg.IDS if i != "WX1"]
ver = {"include_ids": ids, "k": 50, "n_ma": 15}
specs = [rn.VariantSpec(name=f"3-0-{a}", stage="3-0", order="2차", kind="tree", algorithm=a, W=60, mode="sum",
                        input_set="smoke", role="후보", train_count=4, judgement="판정 없음",
                        params={rn.ITER_SELECT: True} if a in rn.BOOST_ALGS else {}, extra={"version": ver})
         for a in ("XGB", "LGB", "CAT", "RF", "ET")]
t0 = time.time()
w = wr.load(ctx_cfg, ROOT, INPUTS, ("weather", "food", "livestock", "soil", "eco", "gw"))
print("context", round(time.time() - t0, 1), flush=True)
st = ev.SampleTable.read(INPUTS / "s0/0-1/apply/samples.parquet")
log = rn.ExperimentLog(OUT / "log.jsonl")
ctx = rn.RunContext(samples=st, out_dir=OUT / "s3", code_root=CODE, input_md5s={}, exp_log=log,
                    make_inputs=w.make_inputs, make_model=w.make_model, n_jobs=8, fit_kwargs=w.fit_kwargs,
                    smoke={"max_train_rows": 15000, "max_pairs_per_h": 250}, require_7a_logged=False,
                    check_resources=lambda p: {"note": "local smoke stub"})
res = rn.run_group(specs, ctx)
for r in res:
    print(r["name"], r["cv_band_weighted"], r.get("n_iter"), r["seconds_total"], flush=True)
for e in log.entries():
    if e["event"] in ("max_iter_decided", "catboost_learning_rate", "alert"):
        print(e)
print("total", round(time.time() - t0, 1))
