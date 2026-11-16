import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
old = "단계 2 준비 판(2-0·2-1·2-1b·2-R) = `listeria-2차` 의 단계 run '2',"
assert s.count(old) == 1
s = s.replace(old, "단계 2 준비 판(2-0·2-1·2-1b·2-R) = `listeria-2차` 의 단계 run '2'(2-0 은 학습 없이 판 run 을 만들고 1차 A 판 run 의 점수·예측을 복사, tag 바탕 판 = 그 1차 판 이름),")
p.write_text(s, encoding="utf-8"); print("ok")
