r"""실험 2 본 실행 — 노트북을 순서대로 실행해 출력을 노트북 파일에 저장한다.

    cd 리스테리아ML
    .venv-exp2\Scripts\python.exe EXP\2\run_all.py            # 2_autogluon → 3_딥러닝
    .venv-exp2\Scripts\python.exe EXP\2\run_all.py 3_딥러닝    # 하나만

- 각 노트북의 설정 셀에서 SMOKE_TEST = False, LOG_TO_MLFLOW = True 인지 확인한 뒤 실행한다.
- 진행 로그: output/exp2/run_all_<시각>.log (셀 단위 시작·끝 시각). 오류가 나면 그 노트북까지의 출력을 저장하고 다음 노트북으로 넘어간다.
- 실행 중 노트북 파일이 편집기에서 저장되면 출력을 덮어쓰지 않고 output/exp2/ 에 사본으로 저장한다.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import nbformat
from nbclient import NotebookClient

EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parent.parent
LOG_DIR = PROJECT_DIR / "output" / "exp2"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ORDER = ["2_autogluon", "3_딥러닝"]
CHECK_ONLY = "--check" in sys.argv          # 설정 스위치·커널만 확인하고 실행하지 않음
names = [a for a in sys.argv[1:] if not a.startswith("--")] or ORDER
log_path = LOG_DIR / f"run_all_{datetime.now():%Y%m%d_%H%M}.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def flags(nb):
    """설정 셀의 맨 왼쪽(들여쓰기 없는) 대입만 본다 — SMOKE_TEST 블록 안의 LOG_TO_MLFLOW = False 는 무시"""
    lines = [l for c in nb.cells if c.cell_type == "code" for l in c.source.splitlines()]
    smoke = any(l.startswith("SMOKE_TEST = True") for l in lines)
    nolog = any(l.startswith("LOG_TO_MLFLOW = False") for l in lines)
    return smoke, nolog


class Progress(NotebookClient):
    def on_cell_start(self, cell, cell_index):
        first = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
        log(f"  셀 {cell_index:2d} 시작: {first}")


if CHECK_ONLY:
    from jupyter_client.kernelspec import KernelSpecManager
    specs = KernelSpecManager().find_kernel_specs()
    print("커널 listeria-exp2:", specs.get("listeria-exp2", "없음"))
    for name in names:
        nb = nbformat.read(EXP_DIR / f"{name}.ipynb", as_version=4)
        smoke, nolog = flags(nb)
        print(f"{name}: 코드 셀 {sum(c.cell_type == 'code' for c in nb.cells)}개 · SMOKE_TEST={smoke} · LOG_TO_MLFLOW={not nolog} → "
              + ("본 실행 가능" if not (smoke or nolog) else "건너뜀 (설정 셀에서 SMOKE_TEST=False, LOG_TO_MLFLOW=True 로)"))
    sys.exit(0)

for name in names:
    path = EXP_DIR / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    smoke, nolog = flags(nb)
    if smoke or nolog:
        log(f"{name}: SMOKE_TEST={smoke} / LOG_TO_MLFLOW={not nolog} — 본 실행 설정이 아니라 건너뜀")
        continue
    mtime0 = path.stat().st_mtime
    log(f"{name} 시작")
    t0 = time.time()
    client = Progress(nb, kernel_name="listeria-exp2", timeout=None, resources={"metadata": {"path": str(EXP_DIR)}})
    try:
        client.execute()
        status = "완료"
    except Exception as e:  # noqa: BLE001 — 어떤 오류든 기록하고 다음 노트북으로
        status = f"오류 {type(e).__name__}: {str(e)[-1500:]}"
    if path.stat().st_mtime == mtime0:
        nbformat.write(nb, path)
        where = str(path)
    else:
        where = str(LOG_DIR / f"{name}_executed_{datetime.now():%Y%m%d_%H%M}.ipynb")
        nbformat.write(nb, where)
    log(f"{name} {status} · {(time.time() - t0) / 3600:.2f}시간 · 출력 저장 {where}")
log("전체 종료")
