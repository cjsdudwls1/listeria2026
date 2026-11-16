#!/bin/bash
# 지금 판 단계별 test 보고(1묶음: ⑥⑦·4-pre·3-1·3-2·3-3·⑧·3-0b·단계 4·5·6) — 사용자 요청(2026-11-11) 단계별 test 상승·하락.
# 각 단계는 그 판을 만든 코드 사본으로 보고(채점 함수는 같음). 하나가 실패해도 다음으로 감.
B=/home/[계정 가림]/cyj_listeria_final
P=$B/venv/bin/python
A="--root $B/data --inputs $B/results"
R=$B/results
WHY="단계별 test 상승·하락 보고(사용자 요청 2026-11-11) — 고르기 모두 끝난 뒤(8-D 는 CV 로만 고름)"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TMPDIR=$B/tmp KERAS_BACKEND=torch
rep() {   # rep <사본> <스크립트> <out> <설정>
  echo "== $3"
  ( cd $B/code_frozen/$1 && $P -I scripts/$2 $A --out $R/$3 --config configs/$4 --report --reason "$WHY" ) && echo "OK $3" || echo "FAIL $3"
}
[ -f $R/s5_6/5-6-T/stage_table_report.csv ] && mv $R/s5_6/5-6-T/stage_table_report.csv $R/s5_6/5-6-T/stage_table_report_5LSTM_only.csv
[ -f $R/s5_6/5-6-T/stage_table_report.notes.json ] && mv $R/s5_6/5-6-T/stage_table_report.notes.json $R/s5_6/5-6-T/stage_table_report_5LSTM_only.notes.json
rep s3_20261109 s3_extra.py s3/x678-G s3_extra_G.json
rep s3_20261109 s4.py s4/4-pre s4_pre.json
rep s3c_20261110 s3_1.py s3/3-1 s3_1.json
rep s3c_20261110 s3_2.py s3/3-2 s3_2.json
rep s3c_20261110 s3_3.py s3/3-3 s3_3.json
rep s3c_20261110 s3_extra.py s3/x8 s3_extra_E.json
rep s3d_20261110 s3_0b.py s3/3-0b-best s3_0b.json
rep s3d_20261110 s4.py s4/4-T s4_T.json
rep s3d_20261110 s4.py s4/4-X1 s4_X1.json
rep s3d_20261110 s4.py s4/4-NN s4_NN.json
rep s3d_20261110 s4.py s4/4-C s4_C.json
rep s3d_20261110 s5_6.py s5_6/5-6-T s5_6_T.json
echo REPORTS_B1_DONE
