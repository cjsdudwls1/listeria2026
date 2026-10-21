#!/bin/bash
# 11번 딥러닝 단계(GPU): BiLSTM·트랜스포머 x 방식 A·B x 학습 행 순서, 창 80일, 기상 변수 18개, 교차검증만
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3
echo "== 시작 $(date)"; uptime; free -g | head -2; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nice -n 10 ../venv/bin/python ../exp3/3_시간순교차검증/src/run3.py dl11 --root $R --L 80
echo "DL11_EXIT $? $(date)"
