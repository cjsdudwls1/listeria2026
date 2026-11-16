#!/bin/bash
# 대기열 A(2026-11-12 18시, 사용자 "KO 점수 더 좋으면 그게 최종" 으로 run_food2.sh 를 둘로 나눔): 원 대기열의 2단계(5-LSTM-food2 학습)
# 프로세스가 끝나기를 기다린 뒤 원 대기열 3~5단계를 같은 코드 사본(food2_20261112)·같은 명령으로. 최종(6~8단계)은 대기열 B.
set -euo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2_20261112
R=$B/results
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "2 기다림: 5-LSTM-food2 학습(pid 4041813) 끝"
while kill -0 4041813 2>/dev/null; do sleep 30; done
test -s $R/s5_6/5-6-T-food2/5-LSTM-food2/done.json || { echo "5-LSTM-food2 done.json 없음 — 2단계 실패"; exit 1; }
echo TRAIN_FOOD2_DONE

step "3 5-LSTM-KO 학습(비교 판)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2-KO --config configs/s5_6_T_food2_ko.json
echo TRAIN_KO_DONE

step "4 test 보고(두 판)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2 --config configs/s5_6_T_food2.json \
  --report --reason "5-LSTM 식품 고정표 v2 다시 학습 보고(사용자 결정 2026-11-12)"
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T-food2-KO --config configs/s5_6_T_food2_ko.json \
  --report --reason "5-LSTM-KO(Koutsoumanis 6칼럼) 보고 — 최종 후보(사용자 결정 2026-11-12: 점수 더 좋으면 최종)"
echo REPORTS_DONE

step "5 KO 6칼럼 분포"
$P -I scripts/s_ko_dist.py $A --out $R/s5_6/ko_dist --config configs/s_ko_dist.json
echo KO_DIST_DONE
echo A_DONE
