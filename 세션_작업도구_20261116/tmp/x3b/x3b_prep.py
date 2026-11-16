# x3b 준비(학습 없음, 2026-11-13): 식품 v2 로 다시 학습한 '결합 스태킹'(s3/3-2-food2/3-2-CAT)을 1등으로 둔 사슬 상태 사본
# logs/chain_state_x3b_in.json(= chain_state_3-3.json 에서 current_best 만 바꿈) — s3_0b 가 읽음. 기존 파일은 읽기만.
import copy, json, sys
from pathlib import Path
CODE, R = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(CODE))
from lst import stage4 as s4
out = R / "logs" / "chain_state_x3b_in.json"
assert not out.exists(), f"이미 있음(덮지 않음): {out}"
cat_dir, vae_dir = "s3/3-2-food2/3-2-CAT", "s3/3-1-food2"
cat = json.loads((R / cat_dir / "spec.json").read_text(encoding="utf-8"))
assert (R / cat_dir / "done.json").exists(), "3-2-CAT(식품 v2) 안 끝남"
assert cat["extra"]["anomaly"] == {"column": "anom_vae", "source": "3-1-VAEt", "dir": vae_dir}, cat["extra"]["anomaly"]
vae = json.loads((R / vae_dir / "3-1-VAEt" / "spec.json").read_text(encoding="utf-8"))
st = json.loads((R / "logs" / "chain_state_3-3.json").read_text(encoding="utf-8"))
best = {"name": "3-2-CAT", "kind": "tree", "dir": cat_dir, "replacement": False,
        "members": [{"name": "3-2-CAT", "dir": cat_dir,
                     "anomaly": {"column": "anom_vae", "version": vae["extra"]["version"], "W": int(vae["W"]),
                                 "mode": vae["mode"], "source": "3-1-VAEt"},
                     "stack4": None, "row_plan": None, "catenc_k": None}]}
s4.check_best(best)
st2 = copy.deepcopy(st)
st2["current_best"] = best
out.write_text(json.dumps(st2, ensure_ascii=False, indent=1), encoding="utf-8")
cv = json.loads((R / cat_dir / "cv_scores.json").read_text(encoding="utf-8"))["cv_band_weighted"]
print("1등 = 3-2-CAT(식품 v2)", cat_dir, "CV", cv, "→", out)
