import sys
from pathlib import Path

R = Path(sys.argv[1])


def sub(p, pairs):
    p = R / p
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (str(p), a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


sub("tests/test_s4_s5_6.py", [
    ('''    state = {"step": "3-3", "current_best": best, "tree_target": best, "tree_top3": ["3-0b-XGB-end"],
             "tree_rank_3_0b": {"XGB": 1}, "history": []}
    s4.save_state(inputs / "logs" / "chain_state_3-3.json", state)''',
     '''    state = {"step": "3-0b", "current_best": best, "tree_target": best, "tree_top3": ["3-0b-XGB-end"],
             "tree_rank_3_0b": {"XGB": 1}, "history": []}
    s4.save_state(inputs / "logs" / "chain_state_3-0b.json", state)      # v78: 단계 4 는 3-0b 판정 뒤 사슬 상태를 읽음'''),
    ('''    st = s4.load_state(inputs / "logs" / "chain_state_3-3.json")''',
     '''    st = s4.load_state(inputs / "logs" / "chain_state_3-0b.json")'''),
    ('''    (inputs / "logs" / "chain_state_3-3.json").unlink()
    s4.save_state(inputs / "logs" / "chain_state_3-3.json", st)''',
     '''    (inputs / "logs" / "chain_state_3-0b.json").unlink()
    s4.save_state(inputs / "logs" / "chain_state_3-0b.json", st)'''),
])
sub("tests/test_s_final.py", [
    ('''    lines = [{"step": "3-0b", "judgement": "출발", "before": None, "after": "3-0b-CAT-end"},''',
     '''    lines = [{"step": "3-0b", "judgement": "출발", "before": None, "after": "옛-순서-줄"},     # 옛 순서 줄 — 안 읽음
             {"step": "3-0", "judgement": "출발", "before": None, "after": "3-0b-CAT-end"},'''),
    ('''["1차 베이스라인", "3-0b 1위(첫 현재 최고 판)", "채택한 기법(4)", "튜닝(8-T)", "최종 모델"]''',
     '''["1차 베이스라인", "3-0 1위(첫 현재 최고 판)", "채택한 기법(4)", "튜닝(8-T)", "최종 모델"]'''),
    ('''    assert "해당 없음" in r3["최종 모델 구조의 변수 빼기 결과"]          # 최종 = 1차 베이스라인''',
     '''    assert "안 거친" in r3["최종 모델 구조의 변수 빼기 결과"]            # 최종 = 1차 베이스라인(v78 A-10)'''),
    ('''    assert rt.loc["t3", "최종 모델 구조의 변수 빼기 결과"] == "CAT: 남음"
    assert rt.loc["rh1", "최종 모델 구조의 변수 빼기 결과"] == "CAT: 그룹째 빠짐"''',
     '''    assert rt.loc["t3", "최종 모델 구조의 변수 빼기 결과"] == "남음"
    assert rt.loc["rh1", "최종 모델 구조의 변수 빼기 결과"] == "해당 없음(최종 모델 입력에 없음)"   # 습도 그룹째 빠져 입력에 없음'''),
])
print("ok")
