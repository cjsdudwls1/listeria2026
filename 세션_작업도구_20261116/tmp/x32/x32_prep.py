# x32 준비(학습 없음, 2026-11-13 사용자 요청: '결합 스태킹'(3-2-CAT = CatBoost + VAE 이상 점수) + 손 교수 방식 + 식품 갱신 + KO):
#  1) 사슬 상태 사본 logs/chain_state_x32.json = chain_state_4.json 에서 stage4.start_best 만 3-2-CAT BestDef 로(따로 판 바탕)
#  2) 그 상태로 s5_6.build 가 만드는 '5-CAT'(손 교수 방식 5, 바탕 3-2-CAT) 판 설정을 s5_6/x32_spec/5-CAT/spec.json 에 —
#     rerun(of = 5-CAT)이 이 설정과 칸 하나하나 같은지 확인한 뒤 KO1 만 더하고 이름을 바꿈. 기존 결과는 읽기만.
import copy, importlib.util, json, sys
from pathlib import Path
CODE, R, CFG = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location("s56", CODE / "scripts" / "s5_6.py")
m = importlib.util.module_from_spec(spec); sys.modules["s56"] = m; spec.loader.exec_module(m)
from lst import stage4 as s4
cs = json.loads(CFG.read_text(encoding="utf-8"))
out_state = R / cs["chain_state_in"]
spec_dir = R / cs["rerun"]["of_dir"]
for p in (out_state, spec_dir):
    assert not p.exists(), f"이미 있음(덮지 않음): {p}"
src_state = json.loads((R / "logs" / "chain_state_4.json").read_text(encoding="utf-8"))
vae = json.loads((R / "s3/3-1/3-1-VAEt/spec.json").read_text(encoding="utf-8"))
cat = json.loads((R / "s3/3-2/3-2-CAT/spec.json").read_text(encoding="utf-8"))
assert cat["extra"]["anomaly"] == {"column": "anom_vae", "source": "3-1-VAEt", "dir": "s3/3-1"}, cat["extra"]["anomaly"]
base = {"name": "3-2-CAT", "kind": "tree", "dir": "s3/3-2/3-2-CAT", "replacement": False,
        "members": [{"name": "3-2-CAT", "dir": "s3/3-2/3-2-CAT",
                     "anomaly": {"column": "anom_vae", "version": vae["extra"]["version"], "W": int(vae["W"]),
                                 "mode": vae["mode"], "source": "3-1-VAEt"},
                     "stack4": None, "row_plan": None, "catenc_k": None}]}
st = copy.deepcopy(src_state)
st["stage4"]["start_best"] = base
st["stage4"]["adopted"] = False
s4.check_best(base)
out_state.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
cfg = dict(cs); cfg.pop("rerun")
b = m.build(cfg, R)
names = [s.name for s in b["run"]]
print("만든 판:", names)
sp = {s.name: s for s in b["run"]}[cs["rerun"]["of"]]
spec_dir.mkdir(parents=True)
(spec_dir / "spec.json").write_text(json.dumps(sp.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
d = sp.to_dict()
print("5-CAT:", {k: d[k] for k in ("name", "algorithm", "kind", "W", "mode", "input_set")},
      "row_plan", d["extra"]["s4"]["row_plan"], "provider", (d["extra"]["s4"].get("provider") or {}).get("kind"),
      "include_ids", d["extra"]["version"]["include_ids"][-3:])
