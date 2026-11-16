import sys
import numpy as np
sys.path.insert(0, r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final")
sys.path.insert(0, r"C:\Users\NT751\.claude\jobs\a19471fc\tmp\review\pkgcheck")
from lstorig import models_dl as old
from lst import models_dl as new
rng = np.random.default_rng(0)
n, W, C, S = 300, 15, 4, 3
x = {"x_seq": rng.normal(size=(n, W, C)).astype(np.float32), "x_static": rng.normal(size=(n, S)).astype(np.float32),
     "x_cat": np.stack([rng.integers(0, m, n) for m in (3, 5, 4, 6)], 1).astype(np.int64)}
y = (x["x_seq"][:, -1, 0] + rng.normal(scale=0.5, size=n) > 0.8).astype(np.int64)
g = (np.array([f"s{i % 7}" for i in range(n)]), np.arange(n) // 3)
for arch in old.CLASSIFIER_ARCHS:
    for cs in ((3, 5, 4, 6), None):
        if cs is None and arch in ("CMB", "CMBV"):
            continue
        xx = x if cs else {k: v for k, v in x.items() if k != "x_cat"}
        a = old.DLClassifier(arch, cs, device="cpu", max_epochs=3).fit(xx, y, seed=0, n_jobs=1, groups=g)
        b = new.DLClassifier(arch, cs, device="cpu", max_epochs=3).fit(xx, y, seed=0, n_jobs=1, groups=g)
        sa, sb = a.net.state_dict(), b.net.state_dict()
        same_keys = list(sa) == list(sb) and all(sa[k].shape == sb[k].shape and bool((sa[k] == sb[k]).all()) for k in sa)
        pa, pb = a.predict_proba(xx), b.predict_proba(xx)
        print(arch, "cat" if cs else "nocat", "state_dict 같음", same_keys, "예측 같음", np.array_equal(pa, pb),
              "fit_info 같음", a.fit_info["es_loss"] == b.fit_info["es_loss"], a.get_config() == b.get_config())
# old 로 저장한 모델을 new 로 불러오기
import tempfile, os
d = tempfile.mkdtemp()
a = old.DLClassifier("LSTM", (3, 5, 4, 6), device="cpu", max_epochs=2).fit(x, y, seed=0, n_jobs=1, groups=g)
p = a.save(os.path.join(d, "m.pt"))
m = new.DLClassifier.load(p, device="cpu")
print("옛 모델 불러오기 예측 같음", np.array_equal(m.predict_proba(x), a.predict_proba(x)), m.hparams)
