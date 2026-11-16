#!/bin/bash
# 국내만 버전 CPU 줄(스레드 8): 단계 0 국내 준비 → 0-10 → 1-1 → (1-2 GPU 줄) → 1-3 → ③ → 단계 2 준비 → 단계 4 준비·
# 미리보기 → 3-0-T → (3-0-D·⑥⑦-D GPU 줄) → ⑥⑦ 판정 → 3-1 → 3-2 → 3-3 → 3-0b → ⑧ → 단계 4(T·X1·C, NN GPU 줄)
# → 판정 → 5·6 → 8 → 최종. 보고(--report, test 채점)는 넣지 않음 — 단계마다 지휘자가.
# 실행은 run_dom/start_dom.sh 참고(실행하지 말 것 — 검수 뒤 지휘자).
source "$(dirname "$0")/common.sh"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
G=$L/run_dom_gpu.log

# ---- 단계 0(국내): 표지·공유 폴더 연결·국내 시료 표 → 결측 표시 목록 → 이력 점검 → 그룹 표 → 7-a → MLflow
run_unless s0/0-1/run_meta.json $P -I scripts/s0_dom_prep.py $A --source $B/results --out $RD/s0/0-1 --config $CFG/s0_dom_prep.json
need s0/0-1/apply/samples.parquet s0/0-1/apply/12_incomplete_months.json s0/0-1/apply/14_theil_target_columns.json SCOPE.json
run_unless s0/0-6_missing/run_meta.json $P -I scripts/s0_6_missing.py $A --out $RD/s0/0-6_missing --config $CFG/s0_6_missing.json
need s0/0-6_missing/missing_lists.json
run_unless s0/0-6_history/run_meta.json $P -I scripts/s0_6_history.py $A --out $RD/s0/0-6_history --config $CFG/s0_6_history.json
run_unless s0/0-8_groups/run_meta.json $P -I scripts/s0_8_groups.py $A --out $RD/s0/0-8_groups --config $CFG/s0_8_groups.json
run_unless s0/0-10/run_meta.json $P -I scripts/s0_10_te.py $A --out $RD/s0/0-10 --config $CFG/s0_10_te.json
need s0/0-10/te7a_selected.json
run_unless s0/0-10_mlflow/mlflow_logged.json $P -I scripts/s0_10_log_mlflow.py $A --out $RD/s0/0-10_mlflow --config $CFG/s0_10_log_mlflow.json
need s0/0-10_mlflow/mlflow_logged.json
echo DOM_S0_DONE

# ---- 1차
run_unless s1/1-1/stage_table_cv.csv $P -I scripts/s1_1_trees.py $A --out $RD/s1/1-1 --config $CFG/s1_1_trees.json
echo DOM_1_1_DONE
wait_mark DOM_G_1_2_DONE $G
run_unless s1/1-3/1-3_decision.json $P -I scripts/s1_3_decide.py $A --out $RD/s1/1-3 --config $CFG/s1_3_decide.json
need s1/1-3/1-3_decision.json
echo DOM_1_3_DONE
$P -I scripts/s_x3.py $A --out $RD/s1/x3 --config $CFG/s_x3.json
echo DOM_X3_DONE

# ---- 단계 2 준비
$P -I scripts/s2_prep.py $A --out $RD/s2/2-0 --config $CFG/s2_0_base.json
$P -I scripts/s2_prep.py $A --out $RD/s2/2-1 --config $CFG/s2_1_k.json
$P -I scripts/s2_prep.py $A --out $RD/s2/2-1b-T --config $CFG/s2_1b_tree.json
$P -I scripts/s2_prep.py $A --out $RD/s2/2-R --config $CFG/s2_R_research.json
wait_mark DOM_G_2_1B_L_DONE $G
$P -I scripts/s2_prep.py $A --out $RD/s2/2-select --config $CFG/s2_select.json
need s2/2-select/2_prep_selection.json
echo DOM_S2_DONE

# ---- 단계 4 준비(줄이기 칼럼·상위 5 품목)·미리보기 4-a
$P -I scripts/s4.py $A --out $RD/s4/sel --config $CFG/s4_prep.json
$P -I scripts/s4.py $A --out $RD/s4/4-pre --config $CFG/s4_pre.json
echo DOM_PRE_DONE

# ---- 3-0 트리 → ⑥⑦ 판정(3-0 순위 포함) — ⑦-T(x7-T, 시계열만 XGBoost)는 제외(담당자 결정 2026-11-10)
$P -I scripts/s3_0.py $A --out $RD/s3/3-0-T --config $CFG/s3_0_T.json
echo DOM_3_0_T_DONE
wait_mark DOM_G_X678_G_DONE $G
$P -I scripts/s3_extra.py $A --out $RD/s3/x67-judge --config $CFG/s3_extra_judge.json
need s3/x67-judge/x67_decision.json
echo DOM_X67_DONE

# ---- 3-1 → 3-2 → 3-3 → 3-0b(1등) → ⑧
$P -I scripts/s3_1.py $A --out $RD/s3/3-1 --config $CFG/s3_1.json
need s3/3-1/3-1_decision.json
$P -I scripts/s3_2.py $A --out $RD/s3/3-2 --config $CFG/s3_2.json
need s3/3-2/3-2_decision.json
$P -I scripts/s3_3.py $A --out $RD/s3/3-3 --config $CFG/s3_3.json
need s3/3-3/3-3_decision.json logs/chain_state_3-3.json
$P -I scripts/s3_0b.py $A --out $RD/s3/3-0b-best --config $CFG/s3_0b.json
need s3/3-0b-best/3-0b_decision.json logs/chain_state_3-0b.json
echo DOM_3_0B_DONE
$P -I scripts/s3_extra.py $A --out $RD/s3/x8 --config $CFG/s3_extra_E.json
echo DOM_X8_DONE

# ---- 단계 4(4-a·4-a0·①·4-c·4-c-L — ② 는 GPU 줄) → 판정
$P -I scripts/s4.py $A --out $RD/s4/4-T --config $CFG/s4_T.json
$P -I scripts/s4.py $A --out $RD/s4/4-X1 --config $CFG/s4_X1.json
$P -I scripts/s4.py $A --out $RD/s4/4-C --config $CFG/s4_C.json
wait_mark DOM_G_4_NN_DONE $G
$P -I scripts/s4.py $A --out $RD/s4/4-judge --config $CFG/s4_judge.json
need s4/4-judge/4_decision.json logs/chain_state_4.json
echo DOM_S4_DONE

# ---- 단계 5·6
$P -I scripts/s5_6.py $A --out $RD/s5_6/sel --config $CFG/s5_6_prep.json
$P -I scripts/s5_6.py $A --out $RD/s5_6/5-6-T --config $CFG/s5_6_T.json
$P -I scripts/s5_6.py $A --out $RD/s5_6/5-6-judge --config $CFG/s5_6_judge.json
need s5_6/5-6-judge/5-6_decision.json logs/chain_state_5-6.json
echo DOM_S56_DONE

# ---- 단계 8(Optuna → Focal)
$P -I scripts/s8.py $A --out $RD/s8/8-tune --config $CFG/s8_tune.json
$P -I scripts/s8.py $A --out $RD/s8/8-T --config $CFG/s8_T.json
$P -I scripts/s8.py $A --out $RD/s8/8-judge --config $CFG/s8_judgeT.json
need s8/8-judge/8T_decision.json logs/chain_state_8T.json
$P -I scripts/s8.py $A --out $RD/s8/8-F --config $CFG/s8_F.json
$P -I scripts/s8.py $A --out $RD/s8/8-judge --config $CFG/s8_judgeF.json
need s8/8-judge/8_decision.json logs/chain_state_8.json
echo DOM_S8_DONE

# ---- 최종 모델 고르기·중요도(보고는 지휘자가 --report)
# s_final 의 dir_2_0·x1_dir 는 앞 단계 결과에 따른 판 폴더라 국내판 결과로 채운 설정을 국내판 뿌리에 만들어 씀
$P -I scripts/dom_final_cfg.py --inputs $RD --config $CFG/s_final.json --write logs/s_final_dom.json
need logs/s_final_dom.json
$P -I scripts/s_final.py $A --out $RD/final --config $RD/logs/s_final_dom.json
need final/final_decision.json
$P -I scripts/s_final.py $A --out $RD/final --config $RD/logs/s_final_dom.json --importance
echo ALL_DONE
