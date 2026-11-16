#!/bin/bash
# 사용: sync_now.sh <results|results_dom> <단계 폴더 ...>   예: sync_now.sh results s3/3-2 s3/x8
# 단계 폴더 안 끝난 판(done.json 있고 .synced_local 없고 pred_*.parquet 있음)을 로컬로 받고 md5 대조 뒤 서버 pred 삭제.
ROOT="$1"; shift
CFG="C:/Users/NT751/.claude/jobs/a19471fc/tmp/ssh_gpu.cfg"
case "$ROOT" in
  results) LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results"; LREL="final_results" ;;
  results_dom) LBASE="C:/Users/NT751/Desktop/천영진/리스테리아ML2/exp/final_results_dom"; LREL="final_results_dom" ;;
  *) echo "뿌리는 results 또는 results_dom"; exit 1 ;;
esac
for ST in "$@"; do
  LIST=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$ST && for d in */; do d=\${d%/}; [ -f \$d/done.json ] && [ ! -f \$d/.synced_local ] && ls \$d/pred_*.parquet >/dev/null 2>&1 && echo \$d; done" < /dev/null)
  for P in $LIST; do
    case " $EXCLUDE " in *" $P "*) echo "남김: $ST/$P"; continue ;; esac
    SP="$ST/$P"
    SM=$(ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$SP && md5sum pred_*.parquet | sort -k2" < /dev/null)
    mkdir -p "$LBASE/$ST"
    scp -q -F "$CFG" -r "gpu:cyj_listeria_final/$ROOT/$SP" "$LBASE/$ST/" < /dev/null || { echo "$SP 받기 실패"; continue; }
    LM=$(cd "$LBASE/$SP" && md5sum pred_*.parquet | sed 's/ \*/  /' | sort -k2)
    if [ -n "$SM" ] && [ "$(echo "$SM" | awk '{print $1,$2}')" = "$(echo "$LM" | awk '{print $1,$2}')" ]; then
      ssh -F "$CFG" gpu "cd ~/cyj_listeria_final/$ROOT/$SP && rm -f pred_*.parquet && date -u +%FT%TZ > .synced_local && mkdir -p ~/cyj_listeria_final/$ROOT/logs && echo '{\"stage\": \"$ST\", \"variant\": \"$P\", \"local\": \"리스테리아ML2/exp/$LREL/$SP\", \"utc\": \"'\$(date -u +%FT%TZ)'\", \"md5_checked\": true}' >> ~/cyj_listeria_final/$ROOT/logs/moved_to_local.jsonl" < /dev/null
      echo "$(date +%T) $SP 옮김"
    else
      echo "$(date +%T) $SP md5 다름 — 안 지움"
    fi
  done
done
ssh -F "$CFG" gpu "df -h ~ | tail -1" < /dev/null
