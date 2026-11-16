"""로컬 실측: 순열 중요도 대상 하나(R = 10) 예측 — 예전(CPU 이어 붙이기 + a[arange]) vs GPU 색인. 합성 입력·무작위 LSTM."""
import sys
import time
from pathlib import Path

CODE = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lst import importance as im, models_dl as md, speed  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
arch = sys.argv[2] if len(sys.argv) > 2 else "LSTM"
W, C, S, R = 60, 76, 3, 10
rng = np.random.default_rng(0)
X = {"x_seq": rng.normal(size=(n, W, C)).astype(np.float32), "x_static": rng.normal(size=(n, S)).astype(np.float32),
     "x_cat": rng.integers(1, 5, size=(n, 4)).astype(np.int64)}
md.set_determinism(42)
m = md.DLClassifier(arch, [10, 10, 10, 10], device="cuda")
m.dims = {"W": W, "C": C, "S": S}
torch.manual_seed(42)
m.net = m._build().to(md.resolve_device("cuda"))
m.net.eval()
leaf = im.Leaf("L", None, m, layout=im.Layout("dict", {}))
sel = {"x_seq": np.array([0, 1, 2, 3, 38, 39, 40, 41]), "x_static": np.array([2])}
bt = np.tile(np.arange(n), R)
st = np.concatenate([rng.permutation(n) for _ in range(R)])
t0 = time.time()
old = leaf.predict(im._take_dict(X, bt, sel, st))
t_old = time.time() - t0
speed.configure({"gpu_importance": True, "fast_predict": True})
t0 = time.time()
g = im._GPUInput(leaf, X)
t_up = time.time() - t0
t0 = time.time()
new = g.predict(leaf, sel, bt, st)
torch.cuda.synchronize()
t_new = time.time() - t0
print(f"n={n} R={R} arch={arch} old {t_old:.1f}s new {t_new:.1f}s (+올리기 {t_up:.1f}s, h 마다 한 번) "
      f"same_bytes={old.tobytes() == new.tobytes()} maxdiff={np.max(np.abs(old - new))}")
