#!/bin/bash
# 8·9·10번(post): 7번 변수 빼기(drop)가 끝난 뒤 이어서 - selection_drop_cat.json 이 새로 생겼는지 확인하고 시작
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3
while pgrep -f "run3.py drop --root" > /dev/null; do sleep 60; done
if ! grep -q "DROP_EXIT 0" $R/logs/drop_console.log; then echo "drop 이 정상 종료되지 않음 - post 시작 안 함 $(date)"; exit 3; fi
echo "== post 시작 $(date)"; uptime; free -g | head -2; df -h ~ | tail -1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nice -n 10 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py post --root $R --model cat --L 80
echo "POST_EXIT $? $(date)"
