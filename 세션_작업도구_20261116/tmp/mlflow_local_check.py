import os, subprocess, sys, time, pathlib, urllib.request
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
D = pathlib.Path(sys.argv[1]).resolve()
exe = pathlib.Path(sys.executable).with_name("mlflow.exe")
p = subprocess.Popen([str(exe), "server", "--backend-store-uri", "sqlite:///" + (D / "mlflow.db").as_posix(),
                      "--artifacts-destination", (D / "artifacts").as_uri(), "--host", "[IP 가림]", "--port", "5056", "--workers", "1"],
                     stdout=open(D / "local_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try: urllib.request.urlopen("http://[IP 가림]:5056/health", timeout=2); break
        except Exception: time.sleep(1)
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri("http://[IP 가림]:5056"); c = MlflowClient()
    e = c.get_experiment_by_name("zz-연결시험"); runs = c.search_runs([e.experiment_id])
    r = runs[0]; print("run", r.info.run_name, r.data.metrics, r.data.tags.get("판 이름"), r.info.artifact_uri)
    out = c.download_artifacts(r.info.run_id, "a.csv", str(D / "dl")); print("artifact", open(out).read().strip().replace("\n", "|"))
finally:
    p.terminate(); p.wait(timeout=30); print("local server stopped", p.returncode)
