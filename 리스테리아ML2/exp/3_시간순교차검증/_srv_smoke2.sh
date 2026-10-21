#!/bin/bash
# 시험 실행 2부(입력 원값·BiLSTM 경로): 이미 만든 root3_smoke2 바닐라 결과 위에서 wxsel(시도 상한)·kselect·tune·choose·drop
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
PY="nice -n 15 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py"
R=~/cyj_listeria/root3_smoke2
export EXP3_OVERRIDE='{"chain": {"Ls": [20]}, "bundle_drop": {"max_passes": 1}, "smoothing_k": {"candidates": [10, 50]}, "smoke": {"n_sgg": 60}}'
set -e
for inp in raw:cat seq:bilstm; do
  i=${inp%%:*}; m=${inp##*:}
  ../venv/bin/python - "$R" "$m" "$i" <<'PY2'
import json, sys
r, m, i = sys.argv[1:]
b = json.load(open(f"{r}/selection_baseline_real.json", encoding="utf-8"))
b.update({"model": m, "input": i, "L": 20, "run": f"{m}__{ {'summary': 'base', 'raw': 'raw', 'seq': 'seq'}[i] }__L020__s42"})
json.dump(b, open(f"{r}/selection_baseline.json", "w", encoding="utf-8"), ensure_ascii=False)
PY2
  rm -f $R/selection_wx.json $R/selection_k.json
  $PY wxsel --root $R --smoke --L 20 --max-steps 2
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
echo SMOKE2_ALL_DONE
