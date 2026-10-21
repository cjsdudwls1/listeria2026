#!/bin/bash
# 시험 실행(post·dl11 경로): 가짜 데이터 끝 2024-12-31(가짜 시험 = 2024년, 진짜 시험 해는 건드리지 않음), smoke 표본, 창 20일
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3_smoke4
rm -rf $R; mkdir -p $R/params
S2=~/cyj_listeria/root3_smoke2
cp $S2/params/xgb.json $R/params/; cp $S2/selection_drop_xgb.json $R/; cp $S2/selection_wx_summary.json $R/selection_wx.json; cp $S2/selection_k_summary.json $R/selection_k.json
../venv/bin/python - "$R" "$S2" <<'PY2'
import json, sys
r, s2 = sys.argv[1:]
b = json.load(open(f"{s2}/selection_baseline_real.json", encoding="utf-8"))
b.update({"model": "xgb", "input": "summary", "L": 20, "run": "xgb__base__L020__s42"})
json.dump(b, open(f"{r}/selection_baseline.json", "w", encoding="utf-8"), ensure_ascii=False)
PY2
export EXP3_OVERRIDE='{"chain": {"Ls": [20]}, "smoke": {"n_sgg": 60}, "data": {"require_fold_tfidf": false}}'
PY="nice -n 15 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py"
set -e
$PY post --root $R --smoke --L 20 --data-end 2024-12-31
$PY dl11 --root $R --smoke --L 20 --data-end 2024-12-31
echo SMOKE_POST_DONE
