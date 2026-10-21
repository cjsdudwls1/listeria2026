#!/bin/bash
# 7번 변수 빼기(사양 4절 7번): CatBoost 시도 18 설정값, 묶음 섞기 중요도(단위 = 묶음, 외부 지표·토양·식품은 변수 하나씩), 범주 4칸·h 제외
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3
echo "== 시작 $(date)"; uptime; free -g | head -2; df -h ~ | tail -1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nice -n 10 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py drop --root $R --model cat --L 80
echo "DROP_EXIT $? $(date)"
