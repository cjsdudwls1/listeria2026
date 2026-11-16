from pathlib import Path
p = Path(r"C:\Users\NT751\Desktop\천영진\리스테리아ML2\exp\final\lst\stage_cli.py"); s = p.read_text(encoding="utf-8")
old1 = '    ap.add_argument("--reason", help="--report 이유(실험 로그에 남김)")\n'
new1 = old1 + '    ap.add_argument("--only", help="판 이름 일부만(쉼표 구분) — 기본·--report 모드(단계 표도 그 판만)")\n'
old2 = '        raise ValueError("판 이름 중복")\n'
new2 = old2 + '''    if args.only:
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in want if n not in names]
        if unknown:
            raise KeyError(f"이 단계에 없는 판: {unknown}")
        specs = [s for s in specs if s.name in want]
        names = [s.name for s in specs]
'''
for o, n in ((old1, new1), (old2, new2)):
    assert s.count(o) == 1
    s = s.replace(o, n)
p.write_text(s, encoding="utf-8"); print("ok")
