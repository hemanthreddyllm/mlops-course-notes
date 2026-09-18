# YouTube Sentiment Insights (Chapter 17)

A Chrome extension that shows how positive, neutral and negative a YouTube video's comments are.
Its back end is a Flask API serving a LightGBM + TF-IDF model, trained by a DVC pipeline and registered in MLflow.
Notes: `notes/17_end_to_end_youtube_sentiment.html`.

## Run order

```bash
source ../.venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 1_preprocessing_eda.ipynb   # EDA, writes reddit_preprocessing.csv
jupyter nbconvert --to notebook --execute --inplace 2_experiments.ipynb         # 6 experiment rounds (~7 min); starts MLflow on :5050 (fresh)
cd ..
jupyter nbconvert --to notebook --execute --inplace pipeline_walkthrough.ipynb  # git/dvc init → dvc repro → registry → API → pytest
```

By hand:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5050
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
dvc repro                                  # data_ingestion → … → model_registration (@staging)
PORT=5001 python flask_api/app.py          # port 5000 is taken by macOS AirPlay
pytest -q tests
```

## Try the extension

* **Without an API key:** start the API, then open `yt_chrome_plugin_frontend/dev_preview.html` in Chrome. It uses fake comments.
* **For real:**
  1. Put a YouTube Data API v3 key in `popup.js`.
  2. Open `chrome://extensions`, turn on Developer mode, click **Load unpacked** and pick `yt_chrome_plugin_frontend/`.
  3. Open a video and click the extension.

## Docker (needs Docker Desktop)

```bash
docker build -t yt-sentiment-api .
docker run -d -p 8080:8080 yt-sentiment-api
```

`.github/workflows/cicd.yaml` (test → ECR → EC2) is used in Chapter 18.
