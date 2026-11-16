#!/bin/bash
# 지금 판(국내+해외) 단계 4 → 5·6 → 8 → 최종 — CPU 줄(스레드 8). 3-0b 판정(logs/chain_state_3-0b.json) 뒤 시작.
# 코드 = code_frozen/s3d_20261110(3-0b 와 같은 사본 — 로컬 코드 그대로). 순서는 국내판 run_dom_cpu.sh 와 같음(설정 = configs/).
# 보고(--report, test 채점)는 넣지 않음 — 단계마다 지휘자가.
set -e
B=/home/[계정 가림]/cyj_listeria_final
C=$B/code_frozen/s3d_20261110
R=$B/results
L=$B/logs
P=$B/venv/bin/python
A="--root $B/data --inputs $R"
G=$L/run_s4_main_gpu.log
export TMPDIR=$B/tmp KERAS_BACKEND=torch OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd $C

wait_file() {   # 결정 파일이 생길 때까지 — $2 로그에 멈춤 문구가 보이면 끝냄
  until [ -s "$R/$1" ]; do
    if [ -n "$2" ] && grep -qE "Traceback|RunStopped" "$2" 2>/dev/null; then echo "앞 단계 멈춤 — $1 기다림 끝냄"; exit 1; fi
    sleep 120
  done
}
wait_mark() {   # 다른 줄 로그의 표시 줄
  until grep -q "^$1\$" "$2" 2>/dev/null; do
    if grep -qE "Traceback|RunStopped" "$2" 2>/dev/null; then echo "$2 멈춤 — $1 기다림 끝냄"; exit 1; fi
    sleep 120
  done
}
need() {
  for f in "$@"; do
    if [ ! -s "$R/$f" ]; then echo "결정 파일 없음 — 멈춤: $f"; exit 1; fi
  done
}

wait_file logs/chain_state_3-0b.json $L/run_s3_0b_main.log
echo MAIN_3_0B_READY

# ---- 단계 4(4-a·4-a0·①·4-c·4-c-L — ② 는 GPU 줄) → 판정
$P -I scripts/s4.py $A --out $R/s4/4-T --config configs/s4_T.json
$P -I scripts/s4.py $A --out $R/s4/4-X1 --config configs/s4_X1.json
$P -I scripts/s4.py $A --out $R/s4/4-C --config configs/s4_C.json
wait_mark MAIN_G_4_NN_DONE $G
$P -I scripts/s4.py $A --out $R/s4/4-judge --config configs/s4_judge.json
need s4/4-judge/4_decision.json logs/chain_state_4.json
echo MAIN_S4_DONE

# ---- 단계 5·6
$P -I scripts/s5_6.py $A --out $R/s5_6/sel --config configs/s5_6_prep.json
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-T --config configs/s5_6_T.json
$P -I scripts/s5_6.py $A --out $R/s5_6/5-6-judge --config configs/s5_6_judge.json
need s5_6/5-6-judge/5-6_decision.json logs/chain_state_5-6.json
echo MAIN_S56_DONE

# ---- 단계 8(Optuna → Focal) — 현재 최고 판에 트리 부분이 없으면 판 없이 사슬 상태 그대로
$P -I scripts/s8.py $A --out $R/s8/8-tune --config configs/s8_tune.json
$P -I scripts/s8.py $A --out $R/s8/8-T --config configs/s8_T.json
$P -I scripts/s8.py $A --out $R/s8/8-judge --config configs/s8_judgeT.json
need s8/8-judge/8T_decision.json logs/chain_state_8T.json
$P -I scripts/s8.py $A --out $R/s8/8-F --config configs/s8_F.json
$P -I scripts/s8.py $A --out $R/s8/8-judge --config configs/s8_judgeF.json
need s8/8-judge/8_decision.json logs/chain_state_8.json
echo MAIN_S8_DONE

# ---- 최종 모델 고르기·중요도(보고는 지휘자가 --report)
$P -I scripts/s_final.py $A --out $R/final --config configs/s_final.json
need final/final_decision.json
$P -I scripts/s_final.py $A --out $R/final --config configs/s_final.json --importance
echo ALL_DONE
