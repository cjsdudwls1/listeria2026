"""로컬 실측(결과 파일 안 만듦): 실제 시료·기상으로 딥러닝 평가 입력 끔·켬 시간·바이트 대조(1차 WEATHER, W 60)."""
import sys
import time
from pathlib import Path

CODE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, str(CODE))
from lst.resources import set_thread_env  # noqa: E402

set_thread_env()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lst import evaluate as ev, folds as fd, inputs as ip, registry as rg, runner as rn, speed, wxfeat as wf  # noqa: E402

MIR = Path(r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\results_mirror")
st = ev.SampleTable.read(MIR / "s0/0-1/apply/samples.parquet")
dom = pd.read_parquet(MIR / "s0/0-4_v2/시군구_일별_기상_0-4_합본.parquet", columns=["sgg_code", "date"] + list(wf.CUBE_VARS))
ov = pd.read_parquet(MIR / "s0/0-4_weather_overseas/해외_나라_일별_기상_1차.parquet",
                     columns=["alpha_3", "sgg", "date"] + list(rg.WEATHER19) + list(wf.OVERSEAS_RENAME))
ctx = ip.InputContext(samples=pd.read_parquet(MIR / "s0/0-1/apply/samples.parquet", columns=list(ip.SAMPLE_COLS)),
                      y_cut=fd.day("2025-06-30"))
ctx.wx = wf.combined_cube(dom, ov)
ctx.missing = rg.MissingLists({1: frozenset(rg.WEATHER19), 2: frozenset(rg.WEATHER19)})
W = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NH = int(sys.argv[2]) if len(sys.argv) > 2 else 4
spec = rg.VersionSpec("bench", 1, W, "sum", include_ids=("WEATHER",))
rows = rn.add_bounds(fd.make_train_rows(st.known_frame(), fd.SPLITS["fold1"]).iloc[:50000], W)
asm = ip.DLInputs(ctx, spec, "fold1").fit(rows)
meta = st.meta()
hs = [(h, idx) for h, idx in fd.iter_scoring_pairs_by_h(st.d_days, fd.SPLITS["fold1"]) if idx.size][:NH]
res = {}
for name, sw in (("off", None), ("on", {"dl_keyed_inputs": True})):
    speed.configure(sw)
    t0 = time.time()
    outs = []
    with asm.pair_cache():
        for h, idx in hs:
            m = meta.iloc[idx].copy()
            m["h"] = np.int16(h)
            X = asm.transform(rn.add_bounds(m, W))
            outs.append({k: __import__("hashlib").md5(v.tobytes()).hexdigest() for k, v in X.items()})
            del X
    res[name] = (time.time() - t0, outs)
    print(name, round(res[name][0], 1), "s", flush=True)
print("pairs", sum(i.size for _, i in hs), "same_bytes", res["off"][1] == res["on"][1])
