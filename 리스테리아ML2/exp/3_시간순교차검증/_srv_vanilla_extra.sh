#!/bin/bash
# 바닐라 세 번째 트리 일꾼(시간 효율): chain 일꾼이 맨 뒤에 할 짧은 창(40·20·30·10)을 먼저 - 끝난 설정은 chain 이 건너뜀. 코드 = 고정 사본 exp3_v
cd ~/cyj_listeria/cwd
export PYTHONUTF8=1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
R=~/cyj_listeria/root3
for L in 40 20 30 10; do
  EXP2_L=$L nice -n 10 ../venv/bin/python ../exp3_v/3_시간순교차검증/src/run3.py vanilla --pairs xgb:summary,lgbm:summary,cat:summary,xgb:raw,lgbm:raw,cat:raw --root $R --device cpu --seed 42 --L $L
  echo "EXTRA_L${L}_EXIT $? $(date)"
done
