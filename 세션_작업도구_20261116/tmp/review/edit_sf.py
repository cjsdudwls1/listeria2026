p = r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\scripts\s_final.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")
a = '''def _limit_gpu(cfg: Mapping) -> None:
    if cfg.get("gpu_memory_fraction"):
        from lst.resources import limit_gpu_memory

        limit_gpu_memory(cfg["gpu_memory_fraction"])
'''
b = '''def _limit_gpu(cfg: Mapping) -> None:
    if cfg.get("gpu_memory_fraction"):
        from lst import models_dl as md
        from lst.resources import limit_gpu_memory

        limit_gpu_memory(cfg["gpu_memory_fraction"])
        md.set_gpu_fraction(cfg["gpu_memory_fraction"])   # 딥러닝 모델 불러올 때 0.4 로 다시 올라가지 않게(models_dl 17)
'''
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
print("ok", repr(nl))
