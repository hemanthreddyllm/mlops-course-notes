"""Generate the Chapter 23 notebook: a real Prometheus + Grafana stack watching the Chapter 17 model API."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ch23_grafana", "grafana_walkthrough.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def py(s): return nbf.v4.new_code_cell(s.strip("\n"))


cells = [
md(r"""
# Chapter 23 · Grafana + Prometheus: walkthrough

Companion to `notes/23_grafana.html` (video 11:08:48 – end).

The video launches an EC2 instance, installs Docker, runs **Grafana** and **Prometheus** as containers, opens ports
3000 and 9090, connects Grafana to Prometheus and builds a dashboard from Prometheus's own metrics.

This notebook runs the same three processes **locally**, with one important addition: the thing being monitored is
our own **Chapter 17 sentiment API**, so the dashboards show real application and model metrics instead of
Prometheus watching itself.

| Part | What happens |
|---|---|
| 1 | instrument the API: `/metrics`, and the four metric types |
| 2 | Prometheus: `prometheus.yml`, scrape targets, the pull model |
| 3 | generate traffic: baseline → spike → errors → **drift** → recovery |
| 4 | PromQL: ask Prometheus the questions a dashboard asks |
| 5 | alert rules, and what fired |
| 6 | Grafana: provisioned data source + dashboard, and a screenshot of it with live data |
| 7 | what to monitor for an ML system (the part the video only hints at) |
| 8 | shut everything down |

**Downloads:** Prometheus (~100 MB) and Grafana (~450 MB) are fetched once into `~/.cache/mlops-course-tools`.
Delete that folder when you're done with the chapter.
"""),
py(r"""
import json, subprocess, shutil, sys, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from IPython.display import Image, display

HERE = Path.cwd()
assert (HERE / "stack.py").exists(), "run this notebook from projects/ch23_grafana"
sys.path.insert(0, str(HERE))
import importlib, stack, traffic
importlib.reload(stack); importlib.reload(traffic)
REPORTS = HERE / "reports"; REPORTS.mkdir(exist_ok=True)
pd.set_option("display.max_colwidth", 100); pd.set_option("display.width", 200)

print("docker:", shutil.which("docker") or "not installed — running the binaries directly instead")
PROM_DIR, GRAFANA_DIR = stack.ensure_tools()
print("prometheus:", PROM_DIR, "\ngrafana   :", GRAFANA_DIR)
"""),
md(r"""
## 1 · Instrument the application
This is the step the video skips, and it's the one that matters: **Prometheus can only scrape metrics that your
application exposes.** The video's dashboard shows Prometheus's own `/metrics`, which says nothing about a model.

`instrumented_api.py` wraps the Chapter 17 Flask API and adds a `/metrics` endpoint with:

| Metric type | Example here | Use it for |
|---|---|---|
| **Counter** (only goes up) | `http_requests_total`, `model_predictions_total{label}` | totals; take `rate()` for per-second values |
| **Gauge** (up and down) | `http_requests_in_flight`, `model_info` | current values: queue depth, temperature, version |
| **Histogram** (buckets) | `http_request_duration_seconds`, `model_batch_size` | latency and size distributions → `histogram_quantile()` for p95 |
| **Summary** (client-side quantiles) | not used | rarely: histograms aggregate across instances, summaries don't |
"""),
py(r"""
stack.start_api()
time.sleep(1)
body = json.dumps({"comments": ["great explanation, thanks", "worst tutorial ever", "where is the dataset"]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{stack.API_PORT}/predict", data=body,
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=1)[:260], "…")
"""),
py(r"""
metrics_text = urllib.request.urlopen(f"http://127.0.0.1:{stack.API_PORT}/metrics").read().decode()
keep = ("http_requests_total", "model_predictions_total", "http_request_duration_seconds_bucket",
        "http_request_duration_seconds_count", "http_requests_in_flight", "model_info", "model_batch_size_sum")
print("\n".join(l for l in metrics_text.splitlines()
                if l.startswith(keep) or (l.startswith("# HELP") and any(k in l for k in keep)))[:1600])
"""),
md(r"""
That plain-text page is the whole contract: no agent, no push, no SDK — just numbers with labels over HTTP.
Labels (`endpoint`, `status`, `label`) are what let you slice the same counter many ways later.

## 2 · Prometheus: scrape, store, evaluate
"""),
py(r"""
print((HERE / "prometheus.yml").read_text())
"""),
py(r"""
stack.start_prometheus(PROM_DIR)
time.sleep(8)          # let it scrape a couple of times
targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{stack.PROM_PORT}/api/v1/targets"))["data"]["activeTargets"]
pd.DataFrame([{"job": t["labels"]["job"], "target": t["scrapeUrl"], "health": t["health"],
               "last scrape": t["lastScrape"][11:19]} for t in targets])
"""),
md(r"""
Prometheus **pulls**: it asks each target for `/metrics` every `scrape_interval`. That's why a target must be
reachable from Prometheus (a security-group rule in the video's EC2 setup), and why short-lived jobs need a
separate Pushgateway.

## 3 · Generate traffic
Five phases so the graphs have something to say. Watch for the **drift** phase: the incoming comments turn
negative, the code and the model stay exactly the same, and only the *prediction mix* moves.
"""),
py(r"""
t0 = time.time()
marks = traffic.run(scale=1.0)
t1 = time.time()
print(f"\ntotal {t1 - t0:.0f}s of traffic")
time.sleep(8)          # one last scrape so the final phase lands in the database
"""),
md(r"""
## 4 · PromQL: the questions a dashboard asks
A dashboard panel is just a query. These are the same expressions used in the Grafana dashboard below.
"""),
py(r"""
PROM = f"http://127.0.0.1:{stack.PROM_PORT}"

def query_range(expr, start, end, step=5):
    url = (f"{PROM}/api/v1/query_range?query={urllib.parse.quote(expr)}"
           f"&start={start}&end={end}&step={step}")
    data = json.load(urllib.request.urlopen(url))["data"]["result"]
    out = {}
    for series in data:
        name = ", ".join(f"{v}" for k, v in series["metric"].items() if k != "__name__") or "value"
        s = pd.Series({pd.Timestamp(float(t), unit="s"): float(v) for t, v in series["values"]})
        out[name] = s
    return pd.DataFrame(out).sort_index()

def query(expr):
    url = f"{PROM}/api/v1/query?query={urllib.parse.quote(expr)}"
    return json.load(urllib.request.urlopen(url))["data"]["result"]

print("requests served :", round(sum(float(r['value'][1]) for r in query('sum(http_requests_total)')), 0))
print("predictions made:", round(sum(float(r['value'][1]) for r in query('sum(model_predictions_total)')), 0))
for r in query('sum by (label) (model_predictions_total)'):
    print(f"   {r['metric']['label']:<9}{float(r['value'][1]):>8.0f}")
"""),
py(r"""
rate_by_endpoint = query_range("sum by (endpoint) (rate(http_requests_total[30s]))", t0, t1)
latency = query_range("histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[1m])))", t0, t1)
by_status = query_range("sum by (status) (rate(http_requests_total[30s]))", t0, t1)
mix = query_range("sum by (label) (rate(model_predictions_total[1m])) "
                  "/ ignoring(label) group_left sum(rate(model_predictions_total[1m]))", t0, t1)

fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
rate_by_endpoint.plot(ax=axes[0], linewidth=2); axes[0].set_title("Request rate by endpoint (req/s)")
latency.plot(ax=axes[1], linewidth=2, color="#8b5cf6", legend=False); axes[1].set_title("p95 latency (seconds)")
by_status.plot.area(ax=axes[2], linewidth=0, alpha=.7); axes[2].set_title("Responses by status code (req/s)")
colors = {"positive": "#1baf7a", "neutral": "#9a9890", "negative": "#e34948"}
mix[[c for c in ["positive", "neutral", "negative"] if c in mix]].plot.area(
    ax=axes[3], color=[colors[c] for c in mix.columns if c in colors], linewidth=0, alpha=.85)
axes[3].set_title("Prediction mix (share of predictions)"); axes[3].set_ylim(0, 1)
for name, when in marks[:-1]:
    for ax in axes:
        ax.axvline(pd.Timestamp(when, unit="s"), color="#0b0b0b", alpha=.35, linestyle=":")
    axes[0].annotate(name, (pd.Timestamp(when, unit="s"), axes[0].get_ylim()[1]),
                     fontsize=8, rotation=90, va="top", ha="right")
for ax in axes:
    ax.grid(alpha=.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.tight_layout(); plt.savefig(REPORTS / "ch23_promql.png", dpi=110); plt.show()
"""),
md(r"""
Read the bottom panel: during **drift** the negative share climbs even though nothing was deployed. That is the
signal a model owner cares about, and it is invisible in CPU or latency graphs.

The top three panels are the classic service signals (traffic, latency, errors) — the "RED method": **R**ate,
**E**rrors, **D**uration.

## 5 · Alerts live with the data, not the dashboard
`alerts.yml` is evaluated by Prometheus every few seconds. Grafana can alert too; pick one place and stick to it.
"""),
py(r"""
print((HERE / "alerts.yml").read_text()[:900])
rules = json.load(urllib.request.urlopen(f"{PROM}/api/v1/rules"))["data"]["groups"]
rows = [{"alert": r["name"], "state": r["state"], "for": r["duration"],
         "firing now": len(r.get("alerts", []))}
        for g in rules for r in g["rules"] if r["type"] == "alerting"]
pd.DataFrame(rows)
"""),
py(r"""
# ALERTS is a real series, so you can graph when an alert was firing
fired = query_range('ALERTS{alertstate="firing"}', t0, t1)
if len(fired.columns):
    print("alerts that fired during the run:")
    for c in fired.columns: print("   ", c)
else:
    print("no alert fired in this window (the run may have been too short for the `for:` clause)")
"""),
md(r"""
## 6 · Grafana
The video does this by hand: log in with admin/admin, add the Prometheus data source, paste the URL, Save & test,
then build panels. Here the same configuration is **provisioned from files**, so a fresh Grafana comes up ready —
which is how you'd keep dashboards in git.
"""),
py(r"""
print((HERE / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text())
dash = json.loads((HERE / "grafana" / "dashboards" / "mlops.json").read_text())
pd.DataFrame([{"panel": p["title"], "type": p["type"],
               "query": p["targets"][0]["expr"][:78] + ("…" if len(p["targets"][0]["expr"]) > 78 else "")}
              for p in dash["panels"]])
"""),
py(r"""
stack.start_grafana(GRAFANA_DIR)
time.sleep(6)
health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{stack.GRAFANA_PORT}/api/health"))
ds = json.load(urllib.request.urlopen(f"http://127.0.0.1:{stack.GRAFANA_PORT}/api/datasources"))
found = json.load(urllib.request.urlopen(f"http://127.0.0.1:{stack.GRAFANA_PORT}/api/search?type=dash-db"))
print("grafana        :", health["version"])
print("data source    :", ds[0]["name"], "→", ds[0]["url"], "(provisioned, no clicking)")
print("dashboard      :", found[0]["title"], "→", found[0]["url"])
"""),
py(r"""
# screenshot the live dashboard with headless Chrome (kiosk mode hides the navigation)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
shot = REPORTS / "ch23_grafana_dashboard.png"
url = (f"http://127.0.0.1:{stack.GRAFANA_PORT}{found[0]['url']}"
       f"?kiosk&from=now-15m&to=now&theme=light")
if Path(CHROME).exists():
    shot.unlink(missing_ok=True)
    # headless Chrome writes the PNG but can linger, so don't wait on the process: poll for the file
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={HERE / '.runtime' / 'chrome'}",
                               "--hide-scrollbars", "--window-size=1400,900", "--virtual-time-budget=25000",
                               f"--screenshot={shot}", url],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        time.sleep(1)
        if shot.exists() and shot.stat().st_size > 20_000:
            time.sleep(2)
            break
    chrome.terminate()
    print("saved", shot.name, f"({shot.stat().st_size // 1024} KB)" if shot.exists() else "— screenshot failed")
    if shot.exists():
        display(Image(str(shot), width=940))
else:
    print("Chrome not found — open this URL yourself:", url)
"""),
md(r"""
The panels are the PromQL queries from part 4, drawn by Grafana against the same Prometheus. Nothing here knows
anything about the model except what the API chose to expose.

## 7 · What to monitor for an ML system
The video shows CPU-style metrics and then makes the right point in passing: *save your predictions and watch them,
so you can tell whether the model is going wrong.* Four layers are worth separating:
"""),
py(r"""
layers = pd.DataFrame([
    ("Infrastructure", "CPU, memory, disk, GPU, restarts", "node_exporter / cAdvisor / cloud metrics",
     "the video's Prometheus self-metrics live here"),
    ("Service (RED)", "request rate, error rate, latency p50/p95/p99", "prometheus_client in the app",
     "our http_* metrics"),
    ("Data", "schema changes, missing values, out-of-range inputs, input drift", "validation in the pipeline; Evidently / Great Expectations",
     "compute in a job, export as metrics"),
    ("Model", "prediction distribution, confidence, ground-truth accuracy once labels arrive", "counters per label + a delayed evaluation job",
     "our model_predictions_total{label}"),
], columns=["layer", "what you watch", "where it comes from", "in this project"])
layers
"""),
md(r"""
A practical rule: **anything you would page someone about belongs in an alert; everything else is a dashboard.**
Ground-truth accuracy usually arrives late (hours or days), so the fast signals are the prediction mix, the input
statistics and the service metrics — which is exactly what the drift phase demonstrated.

## 8 · Shut everything down
The video's equivalent is terminating the EC2 instance so it stops charging.
"""),
py(r"""
stack.stop_all()
print("\nleft on disk:")
print(f"  {stack.RUNTIME}  (Prometheus TSDB + Grafana database for this run — safe to delete)")
print(f"  {stack.TOOLS}  (the downloaded binaries, ~1.5 GB — delete when you're done with the chapter)")
print("\n  rm -rf", stack.RUNTIME)
print("  rm -rf", stack.TOOLS)
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)
