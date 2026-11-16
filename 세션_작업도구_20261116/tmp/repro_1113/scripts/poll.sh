#!/usr/bin/env bash
# 서버 진행 상황 감시(로컬에서): 새 줄만 출력
CFG=C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg
prev=""
while true; do
  cur=$(timeout 50 ssh -F $CFG gpu 'R=~/cyj_listeria_repro_1113; cat $R/logs/steps.log 2>/dev/null; for f in $R/재학습_3-0-LSTM_*/3-0-LSTM/{model_*,pred_*,cv_scores.json,done.json,failed.json}; do [ -e "$f" ] && echo "재학습 파일: $(basename $f)"; done; grep -E "종료 코드|끝 rc|Traceback|Error|오류|멈춤|Killed" $R/logs/20_retrain_s3.log 2>/dev/null | tail -5; grep -E "Traceback|Error" $R/logs/run_checks_main.log 2>/dev/null | tail -3; awk "{split(\$3,a,\" \"); if (a[1]+0 > 16384) print \"GPU 메모리 40% 넘음: \" \$0}" $R/logs/gpu_monitor.log 2>/dev/null | tail -1' 2>/dev/null | sed "s#/home/[^/]*/#~/#g")
  if [ -n "$cur" ]; then
    comm -13 <(printf "%s\n" "$prev" | sort) <(printf "%s\n" "$cur" | sort)
    prev=$cur
  fi
  sleep 60
done
