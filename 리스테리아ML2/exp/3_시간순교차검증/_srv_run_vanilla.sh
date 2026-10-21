#!/bin/bash
# 3차 바닐라(사양 4절 2번): 창 10~100 x (트리 3종 x 요약·원값 + BiLSTM) -> 1차 베이스라인. 코드 = 고정 사본 exp3_v(실행 중 개발 코드와 분리)
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1
export OMP_WAIT_POLICY=PASSIVE
export EXP3_OVERRIDE='{"chain": {"wxsel": false, "kselect": false, "tune": false}}'
R=~/cyj_listeria/root3
echo "== 시작 $(date)"; uptime; free -g | head -2; df -h ~ | tail -1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,user,ni,etimes,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-160
nice -n 10 ../venv/bin/python ../exp3_v/3_시간순교차검증/src/run3.py chain --root $R
echo "CHAIN_EXIT $? $(date)"
uptime; free -g | head -2; df -h ~ | tail -1
