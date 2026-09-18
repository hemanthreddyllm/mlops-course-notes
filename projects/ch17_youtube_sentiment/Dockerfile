# Backend image for the YouTube Sentiment Insights extension.
# The course uses python:3.8-slim-buster; both 3.8 and Debian buster are end-of-life, so this uses 3.11-slim.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet omw-1.4

# only what the API needs at runtime
COPY src/__init__.py src/common.py src/
COPY flask_api/ flask_api/
COPY lgbm_model.pkl tfidf_vectorizer.pkl ./

EXPOSE 8080
# gunicorn instead of Flask's development server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "flask_api.app:app"]
