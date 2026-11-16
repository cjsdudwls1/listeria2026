#!/bin/bash
# 국내만 버전 띄우기(검수 통과 뒤 지휘자가 서버에서 — 이 파일 자체도 그때 실행).
# 앞서 할 일(로컬 → 서버): 코드 폴더(lst·scripts·configs_dom·run_dom)를 $B/code_frozen/dom_20261110 로 올리고
#   그 안에서 `$B/venv/bin/python -I scripts/make_configs_dom.py --check` 가 '맞음' 인지, configs_dom 의 code_reviewed 를
#   검수 뒤 true 로 바꿨는지 확인(바꾼 칸은 --check 가 허용).
# 지금 판(results·logs·code_frozen/s3b_20261109)은 건드리지 않음 — 이 스크립트는 results_dom·logs_dom 만 만듦.
set -e
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/dom_20261110
L=$B/logs_dom
[ -d "$C/run_dom" ] || { echo "얼린 코드 없음: $C"; exit 1; }
[ -e "$B/results_dom/SCOPE.json" ] && echo "results_dom 이미 있음 — 끝난 판은 건너뜀(이어 돌리기)"
mkdir -p $L $B/results_dom $B/tmp
for q in cpu gpu; do
  [ -e "$L/run_dom_$q.log" ] && mv "$L/run_dom_$q.log" "$L/run_dom_${q}_$(date +%Y%m%dT%H%M%S).log"
done
cd $L
setsid nohup nice -n 10 bash $C/run_dom/run_dom_cpu.sh > $L/run_dom_cpu.log 2>&1 < /dev/null &
echo "CPU 줄 PID $!"
setsid nohup nice -n 10 bash $C/run_dom/run_dom_gpu.sh > $L/run_dom_gpu.log 2>&1 < /dev/null &
echo "GPU 줄 PID $!"
