import json
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


sub("scripts/s_final.py", [
    ("change_table.csv(1차 베이스라인 → 3-0b 1위 → 채택한 기법마다", "change_table.csv(1차 베이스라인 → 3-0 1위 → 채택한 기법마다"),
    ("(2) 변화 표 줄 = 사슬 기록(jsonl) 차례: '출발' 줄(3-0b 1위)",
     "(2) 변화 표 줄 = 사슬 기록(jsonl) 차례: '출발' 줄(step '3-0' = 3-0 1위 — v78, 옛 순서 3-0b 의 '출발' 줄은 안 씀)"),
    ('''    "decision_3_0b": cf.Field(str),          # --inputs 아래 3-0b_decision.json''',
     '''    "decision_3_0b": cf.Field(str),          # --inputs 아래 3-0b_decision.json(v78 — 1등 변수 빼기, s3/3-0b-best)'''),
    ('''        if j == "출발":
            out.append({"label": "3-0b 1위(첫 현재 최고 판)", "step": step, "name": ln["after"], "adopted": False,
                        "line": ln})
            continue''', '''        if j == "출발":
            if step != "3-0":                       # 옛 순서 3-0b 가 쓴 출발 줄 — 새 순서에선 안 씀(v78)
                continue
            out.append({"label": "3-0 1위(첫 현재 최고 판)", "step": step, "name": ln["after"], "adopted": False,
                        "line": ln})
            continue'''),
    ('''    if not out or out[0]["label"].startswith("3-0b") is False:
        raise FinalError("사슬 기록 첫 줄이 3-0b '출발' 이 아님")''', '''    if not out or out[0]["label"].startswith("3-0 1위") is False:
        raise FinalError("사슬 기록 첫 줄이 3-0 '출발' 이 아님")'''),
    ('''    def drop_result(col: str) -> str:
        if decision["final_source"] == "1차 베이스라인" or fkind in ("stack4",) or decision["final_def"].get("replacement"):
            return "해당 없음(최종 모델이 1차 베이스라인·④·⑥·⑦·②)"
        out_ = []
        g = rg.group(rg.unit(col).group_no).name
        for a in structs:
            if a != d3["target_structure"]:
                out_.append(f"{a}: 빼기 안 한 구조")
            elif not d3.get("adopted"):
                out_.append(f"{a}: 빼기 후보 버림 — 남음")
            else:
                out_.append(f"{a}: {'그룹째 빠짐' if g in d3['drop_groups'] else '남음'}")
        return "; ".join(out_)''', '''    def drop_result(col: str) -> str:
        """부록 A-10(v78): 최종 모델이 3-0b 끝 판에서 이어진 판이면 그 결과(1등이 앙상블이면 구성마다 같은 결과), 1차
        베이스라인이면 '3-0b 를 안 거친 판', 그 판 입력에 없는 칼럼이면 해당 없음. 구현 세부: '이어진 판' = 사슬 기록에서
        3-0b 판정 뒤에 최종 모델이 정해짐(최종 = 2차 사슬 끝 판) — ② 가 최종이면 ② 입력을 준 트리 대상 판 기준은 판정 파일
        에서 따로 못 가려 '3-0b 를 거친 사슬(② 입력 판 확인 필요)' 로 적음."""
        if decision["final_source"] == "1차 베이스라인":
            return "3-0b 를 안 거친 판(1차 베이스라인)"
        if col not in final_cols:
            return "해당 없음(최종 모델 입력에 없음)"
        if fkind == "nn60":
            return "3-0b 를 거친 사슬(② 입력 판 확인 필요)"
        g = rg.group(rg.unit(col).group_no).name
        if not d3.get("adopted"):
            return "3-0b 후보 버림·학습 없음 — 남음"
        groups = set(d3.get("drop_groups") or [])
        return "그룹째 빠짐" if g in groups else "남음"'''),
])

c = R / "configs" / "s_final.json"
d = json.loads(c.read_text(encoding="utf-8"))
if "s3/3-0b-best" not in d["search_dirs"]:
    d["search_dirs"] = ["s3/3-0b-best"] + list(d["search_dirs"])
c.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

sub("scripts/s4.py", [
    ('''  T      단계 시작 때 현재 최고 판(사슬 상태 chain_state_in)에''', '''  T      단계 시작 때 현재 최고 판(사슬 상태 chain_state_in = 3-0b 판정 뒤 logs/chain_state_3-0b.json — v78)에'''),
    ('''  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-T --config configs/s4_T.json         # 3-3 판정 뒤''',
     '''  $P scripts/s4.py $A --out ~/cyj_listeria_final/results/s4/4-T --config configs/s4_T.json         # 3-0b 판정 뒤'''),
    ('''      트리 중 3-0b CV 1위' = 사슬 상태 tree_rank_3_0b.''', '''      트리 중 3-0 CV 1위'(v78) = 사슬 상태 tree_rank_3_0b(칸 이름 그대로, 값 = 3-0 트리 CV 순위).'''),
    ('''    "chain_state_in": cf.Field(str, required=False),     # 3-3 판정 뒤 사슬 상태(pre 는 없음)''',
     '''    "chain_state_in": cf.Field(str, required=False),     # 3-0b 판정 뒤 사슬 상태(pre 는 없음)'''),
    ('''    """트리 대상 판의 트리 하나(앙상블이면 구성 트리 중 3-0b CV 1위 — tree_rank_3_0b)."""''',
     '''    """트리 대상 판의 트리 하나(앙상블이면 구성 트리 중 3-0 CV 1위 — tree_rank_3_0b 칸, v78)."""'''),
])
sub("lst/stage4.py", [
    ('''사슬 상태(state json — 3-3 판정 뒤 지휘자·3-3 스크립트가 이 꼴로 씀, 단계 4·5·6·8·최종이 이어 씀)''',
     '''사슬 상태(state json — 3-3 스크립트가 이 꼴로 쓰고(chain_state_3-3), 3-0b(scripts/s3_0b)가 끝 판으로 바꿔 씀
(chain_state_3-0b — 단계 4 가 읽음), 단계 4·5·6·8·최종이 이어 씀)'''),
    ('''   "tree_top3": [3-0b 트리 상위 3 이름…], "tree_rank_3_0b": {알고리즘: 3-0b CV 순위(1~)},   # ②·4-c-L 의 '3-0b CV 1위 트리\'''',
     '''   "tree_top3": [3-0 트리 상위 3 이름…], "tree_rank_3_0b": {알고리즘: 3-0 CV 순위(1~)},   # ②·4-c-L 의 '3-0 CV 1위 트리\''''),
])
print("ok")
