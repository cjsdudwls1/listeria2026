# -*- coding: utf-8 -*-
"""옛 순서 3-0b 의 MLflow 단계 run('3-0b') 이름을 '3-0b-옛순서' 로 바꿈.

새 순서 3-0b(1등 대상)는 같은 단계 이름 '3-0b' 로 단계 run 을 찾는다(lst/tracking.stage_run → find_run, run 이름 기준).
옛 run 이 그대로면 새 후보·끝 판 run 이 옛 자식 run 과 이름이 겹쳐 멈추거나(start_variant) 기록이 빠진다.
옛 프로세스가 끝난 뒤에만 실행(대기열이 pid 종료를 기다린 다음 부름). 코드(lst) 는 바꾸지 않음 — 옛 3-0b 이어받기 코드 대조 유지.
"""
import sys

from mlflow.tracking import MlflowClient

URI = "http://[IP 가림]:5055"
EXPERIMENT = "listeria-2차"
OLD, NEW = "3-0b", "3-0b-옛순서"

c = MlflowClient(URI)
exp = c.get_experiment_by_name(EXPERIMENT)
if exp is None:
    sys.exit(f"실험 없음: {EXPERIMENT}")


def top_runs(name):
    rs = c.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = '{name}'", max_results=100)
    return [r for r in rs if not r.data.tags.get("mlflow.parentRunId") and not r.data.tags.get("중단됨")]


if top_runs(NEW):
    sys.exit(f"'{NEW}' 단계 run 이 이미 있음 — 확인 필요")
old = top_runs(OLD)
if not old:
    print(f"'{OLD}' 단계 run 없음 — 바꿀 것 없음")
    sys.exit(0)
if len(old) > 1:
    sys.exit(f"'{OLD}' 단계 run 이 {len(old)}개 — 확인 필요")
rid = old[0].info.run_id
c.set_tag(rid, "mlflow.runName", NEW)
c.set_tag(rid, "옛 순서", "3-0 CV 1위 구조 대상 3-0b(2026-11-09 순서 변경 전 실행, 비교·이어받기용)")
if top_runs(OLD) or len(top_runs(NEW)) != 1:
    sys.exit("이름 바꾸기 확인 실패")
print(f"이름 바꿈: {rid} '{OLD}' → '{NEW}'")
