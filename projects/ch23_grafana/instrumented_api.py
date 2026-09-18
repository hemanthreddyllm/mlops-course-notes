"""The Chapter 17 sentiment API + Prometheus instrumentation.

The video only ever looks at Prometheus's own metrics. The step it skips is this one: an application has to
*expose* metrics before anything can scrape them. Here we wrap the existing Flask app, so the model API from
Chapter 17 gains a `/metrics` endpoint without its own code changing.

Run:  PORT=5002 python instrumented_api.py       → http://localhost:5002/metrics
"""
import os
import sys
import time
from pathlib import Path

from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

CH17 = Path(__file__).resolve().parents[1] / "ch17_youtube_sentiment"
sys.path.insert(0, str(CH17))
os.chdir(CH17)                       # the app loads lgbm_model.pkl relative to its own folder
os.environ.pop("MLFLOW_TRACKING_URI", None)      # use the local pickle, no server needed

from flask_api.app import LABELS, app, predict_sentiments  # noqa: E402

# ----------------------------------------------------------------- the four metric types
# Counter   – only goes up (totals). Rates come from PromQL: rate(http_requests_total[1m])
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "endpoint", "status"])
PREDICTIONS = Counter("model_predictions_total", "Predictions returned, by label", ["label"])
ERRORS = Counter("model_errors_total", "Failed predictions", ["kind"])
# Histogram – buckets of observations; gives you p50/p95/p99 through histogram_quantile()
LATENCY = Histogram("http_request_duration_seconds", "Request duration", ["endpoint"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
BATCH = Histogram("model_batch_size", "Comments per request", buckets=(1, 2, 5, 10, 25, 50, 100, 250))
# Gauge     – goes up and down (a current value)
IN_FLIGHT = Gauge("http_requests_in_flight", "Requests being handled right now")
MODEL_INFO = Gauge("model_info", "Always 1; the labels carry the model version", ["version", "source"])
MODEL_INFO.labels(version=os.getenv("MODEL_VERSION", "v1"), source="local-pickle").set(1)


@app.before_request
def _start_timer():
    from flask import g
    g._t0 = time.perf_counter()
    IN_FLIGHT.inc()


@app.after_request
def _record(response):
    from flask import g, request
    IN_FLIGHT.dec()
    if request.path != "/metrics":
        endpoint = request.path
        REQUESTS.labels(request.method, endpoint, response.status_code).inc()
        LATENCY.labels(endpoint).observe(time.perf_counter() - getattr(g, "_t0", time.perf_counter()))
    return response


# count what the model actually predicts: the distribution is the ML-specific signal
_predict_sentiments = predict_sentiments


def counted_predict(texts):
    BATCH.observe(len(texts))
    try:
        preds = _predict_sentiments(texts)
    except Exception:
        ERRORS.labels(kind="prediction").inc()
        raise
    for p in preds:
        PREDICTIONS.labels(label=LABELS[p]).inc()
    return preds


import flask_api.app as _app_module  # noqa: E402
_app_module.predict_sentiments = counted_predict


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5002)), threaded=True)
