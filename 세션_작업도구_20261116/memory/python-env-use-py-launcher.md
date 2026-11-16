---
name: python-env-use-py-launcher
description: "On this machine anaconda3 python fails from Bash/PowerShell (DLL not found); use `py` (Python 3.14, has openpyxl 3.1.5 + pandas 3.0.3) to run scripts"
metadata: 
  node_type: memory
  type: project
  originSessionId: c146b845-bf17-419a-b8d1-12c574830ada
  modified: 2026-10-14T05:38:02.446Z
---

`C:\Users\NT751\anaconda3\python.exe` crashes when launched from the Bash/PowerShell tools
(exit -1073741515, STATUS_DLL_NOT_FOUND; `python` on PATH resolves to it). `py` (Python 3.14 at
C:\Users\NT751\AppData\Local\Programs\Python\Python314) works and has openpyxl 3.1.5 and pandas 3.0.3.
Anaconda also has openpyxl 3.1.5, so scripts written for `py` run unchanged in the user's Jupyter kernel.

2026-10-14: `import pandas` under `py` now fails (`No module named 'dateutil'`). pyarrow still works —
read parquet with `pyarrow.parquet` / `pyarrow.compute` instead (did not install anything). Console prints
Korean as mojibake (cp949), so write results to a UTF-8 text file and `cat` it.

**Why:** Saves the trial-and-error of hitting the silent anaconda crash each session (observed 2026-09-10).

**How to apply:** Run Python via `py script.py` from PowerShell. Write scripts to a .py file with UTF-8 and run it,
rather than `-c` one-liners with Korean text. Related: [[hazard-data-source-red-items]].
