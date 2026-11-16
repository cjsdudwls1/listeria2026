

def test_dom_final_cfg(tmp_path):
    """최종 단계 설정 채우기: dir_2_0 = 국내판 1-3 의 A, x1_dir = 국내판 4-X1 의 끝난 판 하나."""
    fc = load_script("dom_final_cfg")
    rd = tmp_path / "rd"
    rd.mkdir()
    scope.write_marker(rd, "domestic")
    (rd / "s1" / "1-3").mkdir(parents=True)
    (rd / "s1" / "1-3" / "1-3_decision.json").write_text(json.dumps({"A": "XGB"}), encoding="utf-8")
    (rd / "s2" / "2-0" / "2-0-XGB").mkdir(parents=True)
    (rd / "s2" / "2-0" / "2-0-XGB" / "done.json").write_text("{}", encoding="utf-8")
    argv = ["--inputs", str(rd), "--config", str(CODE / "configs_dom" / "s_final.json"), "--write", "logs/f.json"]
    p = fc.main(argv)
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["dir_2_0"] == "s2/2-0/2-0-XGB" and "x1_dir" not in got and got["scope"] == "domestic"
    for n in ("x1-LGB", "x1-T3"):
        (rd / "s4" / "4-X1" / n).mkdir(parents=True)
    (rd / "s4" / "4-X1" / "x1-LGB" / "done.json").write_text("{}", encoding="utf-8")
    scope._reset_for_tests()
    with pytest.raises(FileExistsError):                   # 이미 있고 내용이 다름(x1_dir 생김)
        fc.main(argv)
    scope._reset_for_tests()
    got2 = json.loads(fc.main(argv[:-1] + ["logs/g.json"]).read_text(encoding="utf-8"))
    assert got2["x1_dir"] == "s4/4-X1/x1-LGB"
    (rd / "s4" / "4-X1" / "x1-T3" / "done.json").write_text("{}", encoding="utf-8")
    scope._reset_for_tests()
    with pytest.raises(RuntimeError, match="여럿"):
        fc.main(argv[:-1] + ["logs/h.json"])
    scope._reset_for_tests()
    with pytest.raises(scope.ScopeError):                  # 지금 판 뿌리(표지 없음)에선 멈춤
        fc.main(["--inputs", str(tmp_path), "--config", str(CODE / "configs_dom" / "s_final.json"), "--write", "x.json"])
