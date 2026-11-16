from pathlib import Path

p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\tests\test_runner.py")
s = p.read_text(encoding="utf-8")
pairs = [
    ('''class FakeModel:
    def __init__(self, spec):
        self.spec = spec''', '''class FakeModel:
    def __init__(self, spec, asm=None):
        assert asm is None or asm.fitted          # make_model(spec, fit 한 조립 객체)
        self.spec = spec'''),
    ('''    assert r1["status"] == "done" and r1["selection_count"] == 1 and r2["selection_count"] == 2''',
     '''    assert r1["status"] == "done" and r1["selection_count"] == 0 and r2["selection_count"] == 1   # 그 판 앞까지'''),
    ('''    assert run.data.metrics["selection_count"] == 1 and "cv_band_weighted" in run.data.metrics''',
     '''    assert run.data.metrics["selection_count"] == 0 and "cv_band_weighted" in run.data.metrics'''),
    ('''    assert t["test 구간 가중"].notna().all() and t["같은 validation 으로 고른 횟수"].tolist() == [1, 2]''',
     '''    assert t["test 구간 가중"].notna().all() and t["같은 validation 으로 고른 횟수"].tolist() == [0, 1]'''),
    ('''    assert log.cv_scored("0-10", "7-a-5y-ku050-kp050", 75, "7-a 격자") == 75
    assert log.cv_scored("1-1", "x") == 76 and log.cv_scored("1-1", "x") == 76''',
     '''    assert log.cv_scored("0-10", "7-a-5y-ku050-kp050", 75, "7-a 격자") == 0
    assert log.cv_scored("1-1", "x") == 75 and log.cv_scored("1-1", "x") == 75   # 1-1 첫 판 = 75
    assert log.cv_scored("1-1", "y") == 76'''),
    ('''    assert res["selection_count"] == 75''', '''    assert res["selection_count"] == 0'''),
    ('''    assert t.loc[0, "단계·판"] == res["name"] and t.loc[0, "같은 validation 으로 고른 횟수"] == 75
    assert t["같은 validation 으로 고른 횟수"].tolist()[1:] == [76, 77]''',
     '''    assert t.loc[0, "단계·판"] == res["name"] and t.loc[0, "같은 validation 으로 고른 횟수"] == 0
    assert t["같은 validation 으로 고른 횟수"].tolist()[1:] == [75, 76]'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("ok")
