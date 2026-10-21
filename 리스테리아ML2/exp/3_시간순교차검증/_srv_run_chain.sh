#!/bin/bash
# 3차 이어 돌리기(재시작 10-20 23시, 병렬 낮춤: 트리 일꾼 2 x OMP 4, 디스크 캐시 없음, 메모리·디스크 10GB 멈춤):
# 남은 바닐라 -> 1차 베이스라인 -> 기상 거르기 -> k 고르기 -> 튜닝(제품명 점검 통과 시) -> 6번 보고에서 멈춤
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1
export OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3
echo "== 시작 $(date)"; uptime; free -g | head -2; df -h ~ | tail -1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,user,ni,etimes,pcpu,rss,args --sort=-pcpu | head -8 | cut -c1-160
nice -n 10 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py chain --root $R
echo "CHAIN_EXIT $? $(date)"
uptime; free -g | head -2; df -h ~ | tail -1
