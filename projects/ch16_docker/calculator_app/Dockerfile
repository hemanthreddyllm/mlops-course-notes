# 1. base image: Debian-based slim Python (the course uses python:3.11-slim-buster, which is end-of-life)
FROM python:3.11-slim

# 2. working directory inside the image
WORKDIR /app

# 3. dependencies first, so this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. then the application code
COPY . .

# 5. document the port the app listens on
EXPOSE 8080

# 6. start command (the course runs `python3 app.py`; gunicorn is the production-grade server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
