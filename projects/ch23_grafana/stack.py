"""Start and stop the local monitoring stack: instrumented API → Prometheus → Grafana.

The video runs all three on an EC2 instance with Docker. The pieces and the ports are the same here,
they just run as processes on this machine:

    instrumented API  :5002   exposes /metrics
    Prometheus        :9090   scrapes /metrics every 5s, stores the samples, evaluates alert rules
    Grafana           :3000   queries Prometheus and draws the dashboards

Prometheus and Grafana are downloaded once to ~/.cache/mlops-course-tools (about 1.5 GB in total).
Delete that folder when you're finished with the chapter.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = Path(os.environ.get("MLOPS_TOOLS", Path.home() / ".cache" / "mlops-course-tools"))
RUNTIME = HERE / ".runtime"                      # Prometheus TSDB + Grafana database (throwaway)
PROM_VERSION, GRAFANA_VERSION = "3.14.0", "13.2.2"
API_PORT, PROM_PORT, GRAFANA_PORT = 5002, 9090, 3000
_procs: dict[str, subprocess.Popen] = {}


# --------------------------------------------------------------------------- download once
def _download(url, dest_dir, rename_to):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / rename_to
    if target.exists():
        return target
    print(f"downloading {url.split('/')[-1]} … (once; cached in {dest_dir})")
    tgz = dest_dir / "tmp.tgz"
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as t:
        root = t.getnames()[0].split("/")[0]
        t.extractall(dest_dir, filter="data")
    (dest_dir / root).rename(target)
    tgz.unlink()
    subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(target)], capture_output=True)
    return target


def ensure_tools():
    arch = "arm64" if os.uname().machine == "arm64" else "amd64"
    system = "darwin" if sys.platform == "darwin" else "linux"
    prom = _download(f"https://github.com/prometheus/prometheus/releases/download/v{PROM_VERSION}/"
                     f"prometheus-{PROM_VERSION}.{system}-{arch}.tar.gz", TOOLS, "prometheus")
    graf = _download(f"https://dl.grafana.com/oss/release/grafana-{GRAFANA_VERSION}.{system}-{arch}.tar.gz",
                     TOOLS, "grafana")
    return prom, graf


# --------------------------------------------------------------------------- helpers
def wait_for(url, timeout=90, expect=200):
    for _ in range(timeout * 2):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == expect:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _spawn(name, cmd, cwd=None, env=None):
    log = open(RUNTIME / f"{name}.log", "w")
    _procs[name] = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **(env or {})}, start_new_session=True)
    return _procs[name]


# --------------------------------------------------------------------------- the three processes
def start_api():
    RUNTIME.mkdir(exist_ok=True)
    _spawn("api", [sys.executable, str(HERE / "instrumented_api.py")], cwd=HERE, env={"PORT": str(API_PORT)})
    ok = wait_for(f"http://127.0.0.1:{API_PORT}/")
    print(f"API        http://127.0.0.1:{API_PORT}        {'ready' if ok else 'FAILED (see .runtime/api.log)'}")
    return ok


def start_prometheus(prom_dir):
    RUNTIME.mkdir(exist_ok=True)
    data = RUNTIME / "prom-data"
    shutil.rmtree(data, ignore_errors=True)
    _spawn("prometheus", [str(prom_dir / "prometheus"),
                          f"--config.file={HERE / 'prometheus.yml'}",
                          f"--storage.tsdb.path={data}",
                          f"--web.listen-address=127.0.0.1:{PROM_PORT}",
                          "--web.enable-lifecycle"], cwd=HERE)
    ok = wait_for(f"http://127.0.0.1:{PROM_PORT}/-/ready")
    print(f"Prometheus http://127.0.0.1:{PROM_PORT}        {'ready' if ok else 'FAILED (see .runtime/prometheus.log)'}")
    return ok


def start_grafana(graf_dir):
    """Provisioning files replace the video's click-through: data source + dashboard are configured on boot."""
    RUNTIME.mkdir(exist_ok=True)
    prov = RUNTIME / "provisioning"
    shutil.rmtree(prov, ignore_errors=True)
    shutil.copytree(HERE / "grafana" / "provisioning", prov)
    cfg = prov / "dashboards" / "dashboards.yml"
    cfg.write_text(cfg.read_text().replace("DASHBOARD_DIR", str(HERE / "grafana" / "dashboards")))
    _spawn("grafana", [str(graf_dir / "bin" / "grafana"), "server",
                       "--homepath", str(graf_dir), "--config", "/dev/null"],
           cwd=graf_dir,
           env={"GF_PATHS_DATA": str(RUNTIME / "grafana-data"),
                "GF_PATHS_LOGS": str(RUNTIME / "grafana-logs"),
                "GF_PATHS_PLUGINS": str(RUNTIME / "grafana-plugins"),
                "GF_PATHS_PROVISIONING": str(prov),
                "GF_SERVER_HTTP_PORT": str(GRAFANA_PORT),
                "GF_SERVER_HTTP_ADDR": "127.0.0.1",
                # the video logs in with admin/admin; anonymous view access keeps this notebook scriptable
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
                "GF_AUTH_ANONYMOUS_ORG_ROLE": "Admin",
                "GF_SECURITY_ADMIN_PASSWORD": "mlops-demo",
                "GF_ANALYTICS_REPORTING_ENABLED": "false",
                "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
                "GF_NEWS_NEWS_FEED_ENABLED": "false"})
    ok = wait_for(f"http://127.0.0.1:{GRAFANA_PORT}/api/health")
    print(f"Grafana    http://127.0.0.1:{GRAFANA_PORT}        {'ready' if ok else 'FAILED (see .runtime/grafana.log)'}")
    return ok


def stop_all():
    for name, p in list(_procs.items()):
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        print(f"stopped {name}")
        _procs.pop(name, None)
    for pattern in ("instrumented_api.py", "prometheus --config.file", "grafana server"):
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
