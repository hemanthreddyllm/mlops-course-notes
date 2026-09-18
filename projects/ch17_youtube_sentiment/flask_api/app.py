"""Backend for the YouTube Sentiment Insights Chrome extension.

Model source (checked in order):
  1. MLflow registry  models:/yt_chrome_plugin_model@staging   (when MLFLOW_TRACKING_URI is set and reachable)
  2. local pickle     lgbm_model.pkl                           (fallback, also used inside Docker)
The TF-IDF vectorizer is always loaded from tfidf_vectorizer.pkl.

Run:  python flask_api/app.py          (http://localhost:5001; macOS AirPlay already uses port 5000)
"""
import io
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, request, send_file  # noqa: E402
from flask_cors import CORS  # noqa: E402
from wordcloud import WordCloud  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import preprocess_comment  # noqa: E402

MODEL_NAME, MODEL_ALIAS = "yt_chrome_plugin_model", "staging"
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
COLORS = {"positive": "#1baf7a", "neutral": "#9a9890", "negative": "#e34948"}

app = Flask(__name__)
CORS(app)   # the extension calls this API from a chrome-extension:// origin


def load_model():
    """Registry first, local pickle as a fallback. Returns (model, source description)."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow.lightgbm
            model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
            return model, f"mlflow registry: {MODEL_NAME}@{MODEL_ALIAS}"
        except Exception as exc:  # server down, alias missing, …
            app.logger.warning("registry load failed (%s); using local pickle", exc)
    with open(ROOT / "lgbm_model.pkl", "rb") as f:
        return pickle.load(f), "local file: lgbm_model.pkl"


model, MODEL_SOURCE = load_model()
with open(ROOT / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict_sentiments(texts: list[str]) -> list[int]:
    cleaned = [preprocess_comment(t) for t in texts]
    features = vectorizer.transform(cleaned).toarray()
    frame = pd.DataFrame(features, columns=vectorizer.get_feature_names_out())
    return [int(p) for p in model.predict(frame)]


def png_response(fig) -> "flask.Response":
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/")
def home():
    return jsonify(message="YouTube Sentiment Insights API", model_source=MODEL_SOURCE)


@app.post("/predict")
def predict():
    comments = (request.get_json(silent=True) or {}).get("comments")
    if not comments or not isinstance(comments, list):
        return jsonify(error="send JSON like {\"comments\": [\"text\", ...]}"), 400
    try:
        preds = predict_sentiments([str(c) for c in comments])
    except Exception as exc:
        return jsonify(error=f"prediction failed: {exc}"), 500
    return jsonify([{"comment": c, "sentiment": p, "label": LABELS[p]} for c, p in zip(comments, preds)])


@app.post("/predict_with_timestamps")
def predict_with_timestamps():
    items = (request.get_json(silent=True) or {}).get("comments")
    if not items or not isinstance(items, list):
        return jsonify(error="send JSON like {\"comments\": [{\"text\": ..., \"timestamp\": ...}]}"), 400
    try:
        preds = predict_sentiments([str(i.get("text", "")) for i in items])
    except Exception as exc:
        return jsonify(error=f"prediction failed: {exc}"), 500
    return jsonify([{"comment": i.get("text"), "sentiment": p, "label": LABELS[p], "timestamp": i.get("timestamp")}
                    for i, p in zip(items, preds)])


@app.post("/generate_chart")
def generate_chart():
    counts = (request.get_json(silent=True) or {}).get("sentiment_counts") or {}
    values = {LABELS[int(k)]: int(v) for k, v in counts.items() if int(v) > 0}
    if not values:
        return jsonify(error="sentiment_counts is empty"), 400
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    wedges, *_ = ax.pie(values.values(), labels=[f"{k}\n{v}" for k, v in values.items()],
                        colors=[COLORS[k] for k in values], autopct="%1.0f%%", startangle=90,
                        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
                        textprops={"fontsize": 10})
    ax.set_title("Sentiment split", fontsize=12)
    return png_response(fig)


@app.post("/generate_wordcloud")
def generate_wordcloud():
    comments = (request.get_json(silent=True) or {}).get("comments") or []
    text = " ".join(preprocess_comment(c) for c in comments)
    if not text.strip():
        return jsonify(error="no usable words in comments"), 400
    cloud = WordCloud(width=800, height=400, background_color="white", colormap="Blues",
                      collocations=False, max_words=120).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return png_response(fig)


@app.post("/generate_trend_graph")
def generate_trend_graph():
    data = (request.get_json(silent=True) or {}).get("sentiment_data") or []
    if not data:
        return jsonify(error="sentiment_data is empty"), 400
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["label"] = df["sentiment"].astype(int).map(LABELS)
    monthly = (df.set_index("timestamp").groupby([pd.Grouper(freq="MS"), "label"]).size()
                 .unstack(fill_value=0))
    share = monthly.div(monthly.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for label in ("positive", "neutral", "negative"):
        if label in share:
            ax.plot(share.index, share[label], marker="o", linewidth=2, color=COLORS[label], label=label)
    ax.set_ylabel("% of comments"); ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    return png_response(fig)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)))
