#!/bin/bash
# 지금 판 단계 8-D 딥러닝 튜닝(씨앗 1개·음성 4배 고정 — 사용자 결정 2026-11-11) → 최종 고르기 다시 → 최종 중요도(fold 3개 동시).
set -e
B=/home/[계정 가림]/cyj_listeria_final
cd $B/code_frozen/s3e_20261111
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
R=$B/results
$P -I scripts/s8_d.py $A --out $R/s8/8-D-tune --config configs/s8_D_tune.json
$P -I scripts/s8_d.py $A --out $R/s8/8-D --config configs/s8_D.json
$P -I scripts/s8_d.py $A --out $R/s8/8-D-judge --config configs/s8_D_judge.json
echo S8D_DONE
$P -I scripts/s8_d.py $A --out $R/s8/8-D --config configs/s8_D.json --report --reason "단계 8-D 보고(사용자 요청 2026-11-11 단계별 test 상승·하락)"
echo S8D_REPORT_DONE
mkdir -p $R/_superseded && mv $R/final $R/_superseded/final_before_8D_$(date +%Y%m%d_%H%M)
$P -I scripts/s_final.py $A --out $R/final --config configs/s_final.json
echo FINAL_DECIDED
$P -I scripts/s_final.py $A --out $R/final --config configs/s_final.json --importance
echo ALL_DONE
