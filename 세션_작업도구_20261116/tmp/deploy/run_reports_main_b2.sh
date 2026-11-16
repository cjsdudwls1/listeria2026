#!/bin/bash
# 2묶음: 단계 2 준비(2-0·2-1 k·2-1b 트리/딥러닝 N_ma·2-R 리서치 변수) test 보고 — s2 사본.
B=/home/[계정 가림]/cyj_listeria_final
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
R=$B/results
WHY="단계별 test 상승·하락 보고(사용자 요청 2026-11-11) — 고르기 모두 끝난 뒤"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
rep() {
  echo "== $3"
  ( cd $B/code_frozen/$1 && $P -I scripts/$2 $A --out $R/$3 --config configs/$4 --report --reason "$WHY" ) && echo "OK $3" || echo "FAIL $3"
}
rep s2_20261108 s2_prep.py s2/2-0 s2_0_base.json
rep s2_20261108 s2_prep.py s2/2-1 s2_1_k.json
rep s2_20261108 s2_prep.py s2/2-1b-T s2_1b_tree.json
rep s2_20261108 s2_prep.py s2/2-1b-L s2_1b_lstm.json
rep s2_20261108 s2_prep.py s2/2-R s2_R_research.json
echo REPORTS_B2_DONE
