"""실험 관리(MLflow) — 계획서 부록 A-15, §8, 공통 규칙 '환경 확인(11-07)'.

- 기록은 서버의 `mlflow server`(sqlite 백엔드 + `--artifacts-destination`) `http://[IP 가림]:5055` 를 거침.
- run 계층 = experiment(listeria-준비·listeria-1차·listeria-2차) → 단계 run(이름 = 단계 번호) → 판 run(이름 = 판 이름).
  3-0b 바퀴 run·Optuna/AutoKeras 시도 run 은 판 run·단계 run 의 자식(같은 `child_run`).
- tag 는 계획서 이름 그대로(한글 키), 판정 tag 는 한 번만 씀. test 지표는 보고 허가가 있을 때만(§1-3).
- 로컬 복사 = mlflow.db 를 sqlite3 backup 으로 뜬 사본 + artifacts 폴더, 로컬 확인 = 같은 두 경로로 `mlflow ui`.
- 서버 띄우기·끄기 = 이 모듈이 띄운 PID 만(pid 파일의 PID·시작 시각이 맞을 때만 끔, 프로세스 트리째).

구현 세부(계획서에 없음):
  - metric·param 키는 ASCII(예: cv_band_weighted, test_recall), tag 키는 계획서 한글 이름. 'run 종류' tag 를 더 둠
    (단계·판·바퀴·시도·사슬 — 찾기용).
  - 멈춘 판을 다시 돌릴 때 이전 run 은 지우지 않고 tag '중단됨'(이유)·이름 끝 '-중단<n>'·상태 FAILED 로 남김.
  - h 별 PR-AUC 는 metric(step = h)으로도 남김(fold1_pr_auc_h …, test_pr_auc_h).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACKING_URI = "http://[IP 가림]:5055"
HOST = "[IP 가림]"
PORT = 5055
UI_PORT = 5000
EXPERIMENTS = ("listeria-준비", "listeria-1차", "listeria-2차")

# 부록 A-15 tag(계획서 이름) — 이 밖의 키는 멈춤
TAG_KEYS = ("차수", "단계", "판 이름", "역할", "판정", "판정 상대", "바탕 판", "선택", "코드 검수 완료", "최종",
            "원 판", "값 모델", "동점 많음", "상한", "run 종류", "중단됨", "설정 md5", "code_md5")
ROLES = ("후보", "교체 후보", "비교만", "기준 줄")
JUDGEMENTS = ("출발", "판정 없음", "채택", "버림", "버림(단계 1위 아님)", "기본 유지", "해당 없음")
RUN_KINDS = ("단계", "판", "바퀴", "시도", "사슬")
ORDERS = {"listeria-준비": "준비", "listeria-1차": "1차", "listeria-2차": "2차"}


class TrackingError(RuntimeError):
    pass


def _quiet_env() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


class Tracker:
    """MlflowClient 감싸기. uri 기본 = 서버([IP 가림]:5055). 시험은 sqlite:/// + artifact_root."""

    def __init__(self, uri: str = TRACKING_URI, *, artifact_root: str | None = None,
                 allowed_experiments: Iterable[str] = EXPERIMENTS):
        _quiet_env()
        from mlflow.tracking import MlflowClient

        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)
        self.artifact_root = artifact_root   # 시험(로컬 sqlite)에서만 — 서버는 mlflow-artifacts:/
        self.allowed = tuple(allowed_experiments)

    # ------------------------------------------------------------ experiment·run 찾기
    def experiment_id(self, name: str) -> str:
        if name not in self.allowed:
            raise TrackingError(f"모르는 experiment: {name} (허용 {self.allowed})")
        e = self.client.get_experiment_by_name(name)
        if e is not None:
            return e.experiment_id
        try:
            import hashlib

            loc = None if self.artifact_root is None else \
                f"{self.artifact_root.rstrip('/')}/{hashlib.md5(name.encode('utf-8')).hexdigest()[:8]}"
            return self.client.create_experiment(name, artifact_location=loc)
        except Exception:  # 다른 프로세스가 먼저 만듦(1-1·1-2 나란히)
            e = self.client.get_experiment_by_name(name)
            if e is None:
                raise
            return e.experiment_id

    def _children(self, exp_id: str, parent_id: str | None, run_name: str):
        flt = f"attributes.run_name = '{_q(run_name)}'"
        if parent_id is not None:
            flt += f" and tags.mlflow.parentRunId = '{_q(parent_id)}'"
        runs = self.client.search_runs([exp_id], filter_string=flt, max_results=1000)
        out = []
        for r in runs:
            if r.data.tags.get("중단됨"):
                continue
            if parent_id is None and r.data.tags.get("mlflow.parentRunId"):
                continue
            out.append(r)
        return out

    def find_run(self, experiment: str, run_name: str, parent_id: str | None):
        rs = self._children(self.experiment_id(experiment), parent_id, run_name)
        if len(rs) > 1:
            raise TrackingError(f"같은 이름 run 이 여럿: {experiment}/{run_name}")
        return rs[0] if rs else None

    def _create(self, experiment: str, run_name: str, parent_id: str | None, tags: Mapping[str, str]) -> str:
        t = dict(tags)
        if parent_id is not None:
            t["mlflow.parentRunId"] = parent_id
        r = self.client.create_run(self.experiment_id(experiment), run_name=run_name, tags=t)
        return r.info.run_id

    def stage_run(self, experiment: str, stage: str, parent_id: str | None = None) -> str:
        """단계 run(이름 = 단계 번호, 예: 0-10·1-1·1-2·2·3-0b-CAT) — 있으면 그대로."""
        r = self.find_run(experiment, stage, parent_id)
        if r is not None:
            return r.info.run_id
        return self._create(experiment, stage, parent_id,
                            {"run 종류": "단계", "단계": stage, "차수": ORDERS[experiment]})

    def chain_run(self) -> str:
        """'사슬' run = listeria-2차 최상위 run 하나."""
        r = self.find_run("listeria-2차", "사슬", None)
        if r is not None:
            return r.info.run_id
        return self._create("listeria-2차", "사슬", None, {"run 종류": "사슬", "차수": "2차"})

    def start_variant(self, experiment: str, parent_id: str, name: str, tags: Mapping[str, str],
                      params: Mapping[str, Any] | None = None, kind: str = "판") -> str:
        """판 run 만들기. 같은 이름의 끝나지 않은 run(멈춘 판)은 '중단됨' 으로 돌리고 새로 만듦.
        끝난(FINISHED) run 이 있으면 멈춤 — 끝난 판은 다시 안 돌림(B-1)."""
        if kind not in RUN_KINDS:
            raise TrackingError(f"run 종류: {kind}")
        exp_id = self.experiment_id(experiment)
        for i, r in enumerate(self._children(exp_id, parent_id, name), 1):
            if r.info.status == "FINISHED":
                raise TrackingError(f"이미 끝난 판 run 이 있음: {name} ({r.info.run_id})")
            self.supersede(r.info.run_id, f"다시 돌림(이전 상태 {r.info.status})", name, i)
        t = {"run 종류": kind, "판 이름": name, "차수": ORDERS[experiment], **dict(tags)}
        self._check_tags(t)
        if "역할" in t and t["역할"] not in ROLES:
            raise TrackingError(f"역할 값: {t['역할']}")
        rid = self._create(experiment, name, parent_id, t)
        if params:
            self.log_params(rid, params)
        return rid

    def child_run(self, experiment: str, parent_id: str, name: str, kind: str, tags: Mapping[str, str] = ()) -> str:
        """3-0b 바퀴 run·시도 run 등 자식 run(이름 규칙은 부록 A-15)."""
        return self.start_variant(experiment, parent_id, name, dict(tags), None, kind=kind)

    def supersede(self, run_id: str, reason: str, name: str, n: int = 1) -> None:
        self.client.set_tag(run_id, "중단됨", reason)
        self.client.set_tag(run_id, "mlflow.runName", f"{name}-중단{n}")
        info = self.client.get_run(run_id).info
        if info.status in ("RUNNING", "SCHEDULED"):
            self.client.set_terminated(run_id, status="FAILED")

    # ------------------------------------------------------------ 쓰기
    @staticmethod
    def _check_tags(tags: Mapping[str, str]) -> None:
        bad = [k for k in tags if k not in TAG_KEYS and not k.startswith("mlflow.")]
        if bad:
            raise TrackingError(f"부록 A-15 에 없는 tag: {bad}")

    def set_tags(self, run_id: str, tags: Mapping[str, Any]) -> None:
        self._check_tags(tags)
        if "판정" in tags:
            raise TrackingError("판정 tag 는 set_judgement 로만(한 번만 씀)")
        for k, v in tags.items():
            self.client.set_tag(run_id, k, str(v))

    def set_judgement(self, run_id: str, judgement: str, against: str | None = None) -> None:
        """판정 tag — 판정 때 한 번 쓰고 안 고침(A-15)."""
        if judgement not in JUDGEMENTS:
            raise TrackingError(f"판정 값: {judgement}")
        cur = self.client.get_run(run_id).data.tags
        if "판정" in cur:
            raise TrackingError(f"판정 tag 가 이미 있음({cur['판정']}) — 고치지 않음")
        self.client.set_tag(run_id, "판정", judgement)
        if against:
            self.client.set_tag(run_id, "판정 상대", against)

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        from mlflow.entities import Param

        items = [Param(str(k), _param_str(v)) for k, v in params.items() if v is not None]
        for i in range(0, len(items), 100):
            self.client.log_batch(run_id, params=items[i:i + 100])

    def log_metrics(self, run_id: str, metrics: Mapping[str, Any], step: int = 0) -> None:
        from mlflow.entities import Metric

        ts = int(time.time() * 1000)
        items = []
        for k, v in metrics.items():
            if v is None:
                continue
            fv = float(v)
            if math.isinf(fv):
                raise TrackingError(f"metric inf: {k}")
            items.append(Metric(str(k), fv, ts, step))
        for i in range(0, len(items), 500):
            self.client.log_batch(run_id, metrics=items[i:i + 500])

    def log_series(self, run_id: str, key: str, values_by_h) -> None:
        """h 별 값(길이 60) → metric(step = h). NaN 도 남김."""
        from mlflow.entities import Metric

        ts = int(time.time() * 1000)
        vals = list(values_by_h)
        items = [Metric(key, float("nan") if v is None else float(v), ts, h) for h, v in enumerate(vals, 1)]
        self.client.log_batch(run_id, metrics=items)

    def log_artifact(self, run_id: str, path, artifact_path: str | None = None) -> None:
        self.client.log_artifact(run_id, str(path), artifact_path)

    def finish(self, run_id: str, status: str = "FINISHED") -> None:
        self.client.set_terminated(run_id, status=status)

    # ------------------------------------------------------------ 점수 기록
    def log_cv(self, run_id: str, cv: Mapping, selection_count: int | None = None) -> None:
        m = {
            "cv_band_weighted": cv["cv_band_weighted"],
            **{f"{f}_band_weighted": v for f, v in cv["fold_band_weighted"].items()},
            **{f"cv_{b}": v for b, v in cv["cv_h_bands"].items()},
            "cv_pooled": cv["cv_pooled"],
            "cv_sklearn_default_band_weighted": cv["cv_sklearn_default_band_weighted"],
            "cv_common_d_band_weighted": cv["cv_common_d_band_weighted"],
            "cv_lift_band_weighted": cv["cv_lift_band_weighted"],
            "threshold": cv["threshold"]["threshold"],
            "cv_threshold_recall": cv["threshold"]["recall"],
            "cv_threshold_precision": cv["threshold"]["precision"],
            "cv_max_tie_ratio": cv["ties"]["max_tie_ratio"],
            **{f"{f}_recall_at_threshold": v for f, v in cv["fold_recall_at_threshold"].items()},
        }
        if selection_count is not None:
            m["selection_count"] = selection_count
        self.log_metrics(run_id, _nan_ok(m))
        for f, rep in cv["folds"].items():
            self.log_series(run_id, f"{f}_pr_auc_h", rep["pr_auc_by_h"])
        self.client.set_tag(run_id, "동점 많음", "예" if cv["ties"]["many_ties"] else "아니오")

    def log_rescore(self, run_id: str, cv_rescore: float) -> None:
        self.log_metrics(run_id, {"cv_band_weighted_rescore": cv_rescore})

    def log_test(self, run_id: str, test: Mapping, permit) -> None:
        """test 지표 — 그 단계 보고 때 같은 run 에(§1-3·A-15). 허가 없으면 멈춤."""
        from .evaluate import check_permit

        check_permit(permit)
        r = test["report"]
        tm = r["threshold_metrics"]
        m = {
            "test_band_weighted": r["pr_auc_band_weighted"], "test_h01_20": r["pr_auc_h01_20"],
            "test_h21_40": r["pr_auc_h21_40"], "test_h41_60": r["pr_auc_h41_60"], "test_pooled": r["pr_auc_pooled"],
            "test_sklearn_default_band_weighted": r["sklearn_default_band_weighted"],
            "test_common_d_band_weighted": r["common_d_band_weighted"],
            "test_lift_band_weighted": r["lift_band_weighted"], "test_lift_pooled": r["lift_pooled"],
            "test_threshold": tm["threshold"], "test_recall": tm["recall"],
            "test_recall_band_weighted": tm["recall_band_weighted"], "test_precision": tm["precision"],
            "test_f1": tm["f1"], "test_f2": tm["f2"], "test_tp": tm["tp"], "test_fn": tm["fn"],
            "test_fp": tm["fp"], "test_tn": tm["tn"],
            **{f"test_top{q}": v for q, v in r["top_q_capture"].items()},
            "test_incomplete_n_pairs": test["incomplete"]["n_pairs"],
            "test_incomplete_n_pos": test["incomplete"]["n_pos"],
            "test_incomplete_band_weighted": test["incomplete"]["pr_auc_band_weighted"],
        }
        self.log_metrics(run_id, _nan_ok(m))
        self.log_series(run_id, "test_pr_auc_h", r["pr_auc_by_h"])


def _nan_ok(m: Mapping[str, Any]) -> dict:
    return {k: (float("nan") if v is None else v) for k, v in m.items()}


def _param_str(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
    if len(s) > 6000:
        raise TrackingError("param 값이 6000자 초과 — artifact 로")
    return s


# ================================================================ 로컬 복사
def sqlite_backup(src_db, dst_db) -> Path:
    """mlflow.db 를 sqlite3 backup API 로 뜸(서버가 쓰는 중에도 일관된 사본). 사본은 새 파일만."""
    src, dst = Path(src_db), Path(dst_db)
    if dst.exists():
        raise FileExistsError(f"덮어쓰기 금지: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    s = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
        ok = d.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise TrackingError(f"사본 무결성 검사 실패: {ok}")
    finally:
        d.close()
        s.close()
    os.replace(tmp, dst)
    return dst


def copy_artifacts(src_dir, dst_dir) -> int:
    """artifacts 폴더를 통째로(새·바뀐 파일만 덮음, 지우지 않음). 복사한 파일 수."""
    src, dst = Path(src_dir), Path(dst_dir)
    n = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        q = dst / p.relative_to(src)
        if q.exists() and q.stat().st_size == p.stat().st_size and q.stat().st_mtime >= p.stat().st_mtime:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def snapshot_store(mlflow_dir, snapshot_dir) -> dict:
    """(서버 쪽) 저장소 사본 하나 = snapshot_dir/mlflow.db(backup) + snapshot_dir/artifacts. 단계 끝마다."""
    m, s = Path(mlflow_dir), Path(snapshot_dir)
    if s.exists():
        raise FileExistsError(f"덮어쓰기 금지: {s}")
    sqlite_backup(m / "mlflow.db", s / "mlflow.db")
    n = copy_artifacts(m / "artifacts", s / "artifacts") if (m / "artifacts").is_dir() else 0
    info = {"created_utc": _now(), "n_artifact_files": n}
    (s / "snapshot.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def install_local_copy(snapshot_dir, local_mlflow_dir) -> dict:
    """(로컬 쪽) 받은 사본을 `리스테리아ML2/exp/mlflow/` 에: mlflow.db 는 이전 것을 mlflow.db.prev-<시각> 로 남기고
    교체, artifacts 는 합침. `mlflow ui` 를 끈 뒤에 부를 것(Windows 는 db 잠김)."""
    s, l = Path(snapshot_dir), Path(local_mlflow_dir)
    l.mkdir(parents=True, exist_ok=True)
    db = l / "mlflow.db"
    new = l / "mlflow.db.new"
    new.unlink(missing_ok=True)
    sqlite_backup(s / "mlflow.db", new)
    if db.exists():
        os.replace(db, l / f"mlflow.db.prev-{_dt.datetime.now().strftime('%Y%m%d%H%M%S')}")
    os.replace(new, db)
    n = copy_artifacts(s / "artifacts", l / "artifacts") if (s / "artifacts").is_dir() else 0
    return {"db": str(db), "n_artifact_files": n}


# ================================================================ 서버 띄우기·끄기
def _mlflow_exe(mlflow_exe=None) -> str:
    if mlflow_exe:
        return str(mlflow_exe)
    name = "mlflow.exe" if sys.platform == "win32" else "mlflow"
    p = Path(sys.executable).with_name(name)
    if not p.exists():
        raise FileNotFoundError(f"mlflow 실행 파일 없음: {p}")
    return str(p)


def server_command(mlflow_dir, host: str = HOST, port: int = PORT, mlflow_exe=None, ui: bool = False) -> list[str]:
    """`mlflow server|ui --backend-store-uri sqlite:///<dir>/mlflow.db --artifacts-destination <dir>/artifacts`."""
    m = Path(mlflow_dir).resolve()
    return [_mlflow_exe(mlflow_exe), "ui" if ui else "server",
            "--backend-store-uri", "sqlite:///" + (m / "mlflow.db").as_posix(),
            "--artifacts-destination", (m / "artifacts").as_uri(),
            "--host", host, "--port", str(port), "--workers", "1"]


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _proc_start_time(pid: int) -> str | None:
    """프로세스 시작 시각 표지(PID 재사용 구별용). 없으면 None."""
    try:
        import psutil

        return repr(psutil.Process(pid).create_time())
    except ImportError:
        pass
    except Exception:
        return None
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[19]  # 22번째 칸 starttime
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class ServerHandle:
    pid: int
    start_time: str | None
    cmd: tuple
    pidfile: Path


def start_server(mlflow_dir, host: str = HOST, port: int = PORT, mlflow_exe=None, ui: bool = False,
                 wait_s: int = 90) -> ServerHandle:
    """mlflow server(또는 ui)를 띄우고 pid 파일(`<mlflow_dir>/server_<port>.pid.json`)에 PID·시작 시각을 남김.
    포트가 이미 쓰이면 멈춤(남이 띄운 것을 건드리지 않음)."""
    m = Path(mlflow_dir).resolve()
    (m / "artifacts").mkdir(parents=True, exist_ok=True)
    if _port_open(host, port):
        raise TrackingError(f"{host}:{port} 가 이미 쓰임 — 내가 띄운 것이 아니면 건드리지 않음")
    pidfile = m / f"server_{port}.pid.json"
    if pidfile.exists():
        raise TrackingError(f"pid 파일이 남아 있음: {pidfile} — stop_server 로 먼저 정리")
    cmd = server_command(m, host, port, mlflow_exe, ui)
    log = open(m / f"server_{port}.log", "a", encoding="utf-8")
    kw: dict = {"stdout": log, "stderr": subprocess.STDOUT, "env": {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, **kw)
    h = ServerHandle(p.pid, _proc_start_time(p.pid), tuple(cmd), pidfile)
    pidfile.write_text(json.dumps({"pid": h.pid, "start_time": h.start_time, "cmd": list(cmd), "started_utc": _now()},
                                  ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if p.poll() is not None:
            raise TrackingError(f"mlflow 가 바로 끝남(rc {p.returncode}) — {m / f'server_{port}.log'}")
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
            return h
        except Exception:
            time.sleep(1)
    raise TrackingError(f"{wait_s}초 안에 /health 응답 없음(PID {p.pid}) — stop_server 로 끌 것")


def stop_server(pidfile, timeout_s: int = 30) -> dict:
    """pid 파일의 PID 만 끔 — 시작 시각이 같을 때만(PID 재사용 방지). 프로세스 트리째
    (Windows `taskkill /T /F /PID`, 리눅스 그 세션 프로세스 그룹)."""
    pf = Path(pidfile)
    info = json.loads(pf.read_text(encoding="utf-8"))
    pid = int(info["pid"])
    now = _proc_start_time(pid)
    alive = now is not None
    if alive and info.get("start_time") is not None and now != info["start_time"]:
        raise TrackingError(f"PID {pid} 의 시작 시각이 기록과 다름 — 다른 프로세스라 끄지 않음")
    if alive:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)   # start_new_session=True → pgid = pid
                t0 = time.time()
                while time.time() - t0 < timeout_s and _proc_start_time(pid) is not None:
                    time.sleep(0.5)
                if _proc_start_time(pid) is not None:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    os.replace(pf, pf.with_name(pf.name + f".stopped-{int(time.time())}"))
    return {"pid": pid, "was_alive": alive}


def ui_command(local_mlflow_dir, port: int = UI_PORT, mlflow_exe=None) -> list[str]:
    """로컬 확인용 `mlflow ui`(같은 두 경로) → 브라우저 [IP 가림]:5000(A-15)."""
    return server_command(local_mlflow_dir, HOST, port, mlflow_exe, ui=True)
