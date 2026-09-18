"""API tests: run with `pytest -q tests` from the project root (needs lgbm_model.pkl + tfidf_vectorizer.pkl)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("MLFLOW_TRACKING_URI", None)   # tests use the local pickle, no server needed

from flask_api.app import app  # noqa: E402

PNG = b"\x89PNG"


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "model_source" in r.get_json()


def test_predict_labels(client):
    comments = ["this video is amazing, thank you", "worst tutorial ever, terrible audio", "what time is it"]
    r = client.post("/predict", json={"comments": comments})
    assert r.status_code == 200
    body = r.get_json()
    assert [b["comment"] for b in body] == comments
    assert all(b["sentiment"] in (-1, 0, 1) for b in body)


def test_predict_rejects_bad_payload(client):
    assert client.post("/predict", json={}).status_code == 400
    assert client.post("/predict", json={"comments": "not a list"}).status_code == 400


def test_predict_with_timestamps_keeps_timestamp(client):
    r = client.post("/predict_with_timestamps",
                    json={"comments": [{"text": "great work", "timestamp": "2026-01-05T10:00:00Z"}]})
    assert r.status_code == 200
    assert r.get_json()[0]["timestamp"] == "2026-01-05T10:00:00Z"


def test_images_are_png(client):
    r1 = client.post("/generate_chart", json={"sentiment_counts": {"1": 5, "0": 3, "-1": 2}})
    r2 = client.post("/generate_wordcloud", json={"comments": ["great course on mlops", "docker and mlflow"]})
    r3 = client.post("/generate_trend_graph", json={"sentiment_data": [
        {"timestamp": "2026-01-05T10:00:00Z", "sentiment": 1},
        {"timestamp": "2026-02-05T10:00:00Z", "sentiment": -1},
        {"timestamp": "2026-02-07T10:00:00Z", "sentiment": 0}]})
    for r in (r1, r2, r3):
        assert r.status_code == 200 and r.mimetype == "image/png" and r.data.startswith(PNG)
