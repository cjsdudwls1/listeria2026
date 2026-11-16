# -*- coding: utf-8 -*-
"""멈춘(사용자 결정으로 뺀) 판 run 에 '중단됨' tag 를 달고 끝냄 — lst.tracking.supersede 와 같은 꼴(이름 뒤 '-중단1').
사용: python mark_stopped.py <experiment> <판 이름> <이유>   — 끝나지 않은(RUNNING·SCHEDULED) 같은 이름 run 만 대상, 정확히 1개여야 함.
"""
import sys

from mlflow.tracking import MlflowClient

exp_name, name, reason = sys.argv[1], sys.argv[2], sys.argv[3]
c = MlflowClient("http://[IP 가림]:5055")
exp = c.get_experiment_by_name(exp_name)
if exp is None:
    sys.exit(f"실험 없음: {exp_name}")
rs = [r for r in c.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = '{name}'", max_results=100)
      if not r.data.tags.get("중단됨") and r.info.status in ("RUNNING", "SCHEDULED")]
if len(rs) != 1:
    sys.exit(f"끝나지 않은 '{name}' run 이 {len(rs)}개 — 확인 필요")
rid = rs[0].info.run_id
c.set_tag(rid, "중단됨", reason)
c.set_tag(rid, "mlflow.runName", f"{name}-중단1")
c.set_terminated(rid, status="KILLED")
print(f"중단 표시: {exp_name}/{name} ({rid})")
