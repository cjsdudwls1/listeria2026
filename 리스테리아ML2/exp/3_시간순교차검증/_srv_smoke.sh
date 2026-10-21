#!/bin/bash
# 3차 run3.py 시험 실행(smoke: 시군구 12개·표본) - 바닐라 L=20 7개 설정 -> baseline -> (입력 3종 각각) wxsel 1회차·kselect 2후보 -> tune 2회 -> drop 2단계
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=6
PY="nice -n 10 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py"
R=~/cyj_listeria/root3_smoke2
rm -rf $R; mkdir -p $R
export EXP3_OVERRIDE='{"chain": {"Ls": [20]}, "bundle_drop": {"max_passes": 1}, "smoothing_k": {"candidates": [10, 50]}, "smoke": {"n_sgg": 60}}'
set -e
$PY vanilla --root $R --smoke --L 20
$PY baseline --root $R --smoke
cp $R/selection_baseline.json $R/selection_baseline_real.json
for inp in summary:xgb raw:cat seq:bilstm; do
  i=${inp%%:*}; m=${inp##*:}
  ../venv/bin/python - "$R" "$m" "$i" <<'EOF'
import json, sys
r, m, i = sys.argv[1:]
b = json.load(open(f"{r}/selection_baseline_real.json", encoding="utf-8"))
b.update({"model": m, "input": i, "L": 20, "run": f"{m}__{ {'summary': 'base', 'raw': 'raw', 'seq': 'seq'}[i] }__L020__s42"})
json.dump(b, open(f"{r}/selection_baseline.json", "w", encoding="utf-8"), ensure_ascii=False)
EOF
  rm -f $R/selection_wx.json $R/selection_k.json
  $PY wxsel --root $R --smoke --L 20
  cp $R/selection_wx.json $R/selection_wx_$i.json
  $PY kselect --root $R --smoke --L 20
  cp $R/selection_k.json $R/selection_k_$i.json
  cp $R/tables/기상거르기.csv $R/tables/기상거르기_$i.csv
  cp $R/tables/k선택.csv $R/tables/k선택_$i.csv
  if [ "$i" != "raw" ]; then
    $PY tune --root $R --smoke --L 20 --trials 2
    $PY choose --root $R --smoke --L 20 --trial 0
    $PY drop --root $R --smoke --L 20 --max-steps 2
  fi
done
echo SMOKE_ALL_DONE
