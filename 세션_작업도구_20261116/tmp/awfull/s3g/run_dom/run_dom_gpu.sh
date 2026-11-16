#!/bin/bash
# 국내만 버전 GPU 줄(스레드 4, 설정 gpu_memory_fraction 0.4): 1-2 LSTM → 2-1b-L → 3-0-D(딥러닝·결합 7) → ⑥·⑦-D →
# ② 60칸 신경망. 앞 단계는 CPU 줄 로그 표시·결정 파일로 기다림. 실행은 run_dom/start_dom.sh 참고(실행하지 말 것).
source "$(dirname "$0")/common.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
K=$L/run_dom_cpu.log

wait_mark DOM_S0_DONE $K
$P -I scripts/s1_2_lstm.py $A --out $RD/s1/1-2 --config $CFG/s1_2_lstm.json
echo DOM_G_1_2_DONE

wait_file s1/1-3/1-3_decision.json $K
$P -I scripts/s2_prep.py $A --out $RD/s2/2-1b-L --config $CFG/s2_1b_lstm.json
echo DOM_G_2_1B_L_DONE

wait_file s2/2-select/2_prep_selection.json $K
$P -I scripts/s3_0.py $A --out $RD/s3/3-0-D --config $CFG/s3_0_D.json
echo DOM_G_3_0_D_DONE
$P -I scripts/s3_extra.py $A --out $RD/s3/x678-G --config $CFG/s3_extra_G.json
echo DOM_G_X678_G_DONE

wait_file logs/chain_state_3-0b.json $K
$P -I scripts/s4.py $A --out $RD/s4/4-NN --config $CFG/s4_NN.json
echo DOM_G_4_NN_DONE
echo ALL_DONE
