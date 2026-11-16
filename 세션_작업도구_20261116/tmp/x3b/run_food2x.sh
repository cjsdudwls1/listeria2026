#!/bin/bash
# x3b(2026-11-13 사용자 요청): 식품 고정표 v2 로 '결합 스태킹'(3-2-CAT = CatBoost + VAE 이상 점수) 다시 학습 → 변수 빼기(3-0b).
# 새 폴더(s3/3-1-food2·3-2-food2·3-0b-food2)·새 사슬 기록(logs/chain_log_food2x.jsonl)·새 사슬 상태(x3b_in·out)·MLflow 끔.
set -euo pipefail
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/food2b_20261112
R=$B/results
L=$B/logs
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
cd $C
step() { echo "=== $(date '+%F %T') $1"; }

step "0 점검"
for d in s3/3-1-food2 s3/3-2-food2 s3/3-0b-food2 logs/chain_log_food2x.jsonl logs/chain_state_x3b_in.json logs/chain_state_x3b_out.json; do
  test ! -e "$R/$d" || { echo "이미 있음(덮지 않음): $R/$d"; exit 1; }
done
for f in s0/0-6_static_v2/식품_고정표.parquet s3/x67-judge/x67_decision.json logs/chain_state_3-3.json; do
  test -s "$R/$f" || { echo "없음: $R/$f"; exit 1; }
done

step "1 단계 3-1 다시(VAE 이상 점수 + 트리 대상 판 +VAE, 판정 — 새 사슬 기록)"
$P -I scripts/s3_1.py $A --out $R/s3/3-1-food2 --config $L/s3_1_food2x.json
echo S31_DONE

step "2 '결합 스태킹' 3-2-CAT 다시(식품 v2)"
$P -I scripts/s3_2.py $A --out $R/s3/3-2-food2 --config $L/s3_2_food2x.json --variant 3-2-CAT
echo S32_DONE

step "3 사슬 상태 사본(1등 = 3-2-CAT 식품 v2)"
$P -I $L/x3b_prep.py $C $R
echo PREP_DONE

step "4 변수 빼기(중요도 6부분 → 평균 I ≤ 0 그룹 빼고 다시 학습 → 판정)"
$P -I scripts/s3_0b.py $A --out $R/s3/3-0b-food2 --config $L/s3_0b_food2x.json
echo X3B_DONE
