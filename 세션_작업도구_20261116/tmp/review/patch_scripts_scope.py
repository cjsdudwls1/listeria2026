"""국내만 버전: 스크립트 설정 스키마에 scope 칸·범위 정하기·앞 인자 읽기의 결과 뿌리 받아 두기를 넣는 한 번짜리 패치."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
IMPORT_ANCHOR = "from lst import config as cf  # noqa: E402\n"
IMPORT_LINE = "from lst import scope as lst_scope  # noqa: E402\n"

# (파일, 스키마 상수 이름들, 설정을 읽는 줄(그 뒤에 범위 정하기))
PLAN = {
    "s0_6_missing": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s0_6_history": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s0_8_groups": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s0_10_te": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s0_10_log_mlflow": (["SCHEMA"], ["    cfg = dict(cf.load_config(args.config, SCHEMA))\n"]),
    "s1_3_decide": (["SCHEMA"], ["    cfg = dict(cf.load_config(args.config, SCHEMA))\n"]),
    "s_x3": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s_final": (["SCHEMA"], ["    cfg = load_cfg(args.config)\n"]),
    "s2_prep": (["BASE_SCHEMA", "SELECT_SCHEMA"], ["    cfg = dict(cf.load_config(args.config, BASE_SCHEMA))\n",
                                                   "    cfg = dict(cf.load_config(args.config, SELECT_SCHEMA))\n"]),
    "s3_extra": (["JUDGE_SCHEMA"], ["    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))\n"]),
    "s4": (["PREP_SCHEMA", "JUDGE_SCHEMA"], ["    cfg = dict(cf.load_config(args.config, PREP_SCHEMA))\n",
                                             "    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))\n"]),
    "s5_6": (["PREP_SCHEMA", "JUDGE_SCHEMA"], ["    cfg = dict(cf.load_config(args.config, PREP_SCHEMA))\n",
                                               "    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))\n"]),
    "s8": (["JUDGE_SCHEMA"], ["    cfg = dict(cf.load_config(args.config, JUDGE_SCHEMA))\n"]),
    "s3_0": ([], []), "s3_0b": ([], []), "s3_1": ([], []), "s3_2": ([], []), "s3_3": ([], []),
}
ACT = "    lst_scope.activate(cfg.get(\"scope\"))      # 국내만 버전: 범위·결과 뿌리 표지 대조(lst/scope.py)\n"
PRE_RET = "    return ap.parse_known_args(list(argv))[0]\n"
PRE_NEW = ("    ns = ap.parse_known_args(list(argv))[0]\n"
           "    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)\n"
           "    return ns\n")
PRE_RET2 = "    ns, _ = ap.parse_known_args(list(argv))\n    return ns\n"
PRE_NEW2 = ("    ns, _ = ap.parse_known_args(list(argv))\n"
            "    lst_scope.note_roots(ns.inputs, ns.out)   # 국내만 버전: 설정을 읽어 범위를 정할 때 대조(lst/scope.py)\n"
            "    return ns\n")

for name, (schemas, loads) in PLAN.items():
    p = ROOT / "scripts" / f"{name}.py"
    s = p.read_text(encoding="utf-8")
    assert s.count(IMPORT_ANCHOR) == 1, name
    s = s.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE)
    # 스키마 상수 뒤에 scope 칸 더하기(AST 로 끝 줄 찾기)
    for sch in schemas:
        tree = ast.parse(s)
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == sch]
        assert len(nodes) == 1, (name, sch)
        lines = s.splitlines(keepends=True)
        end = nodes[0].end_lineno
        lines.insert(end, f"{sch} = lst_scope.with_field({sch})   # 국내만 버전 scope 칸(없으면 지금 판)\n")
        s = "".join(lines)
    for ln in loads:
        assert s.count(ln) == 1, (name, ln)
        s = s.replace(ln, ln + ACT)
    n1, n2 = s.count(PRE_RET), s.count(PRE_RET2)
    if n1:
        assert n1 == 1, name
        s = s.replace(PRE_RET, PRE_NEW)
    if n2:
        assert n2 == 1, name
        s = s.replace(PRE_RET2, PRE_NEW2)
    p.write_text(s, encoding="utf-8", newline="\n")
    print(name, "schemas", schemas, "loads", len(loads), "pre", n1 + n2)
