# ---------- Stage 1: build wheels in a full toolchain image ----------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
# Prefix install keeps the final copy to a single directory.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: slim runtime image ----------
FROM python:3.12-slim

# libgomp1: OpenMP runtime required by xgboost / shap wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY app/ app/

# Trained models + test split are NOT baked into the image (gitignored,
# large). Mount them at runtime:
#   docker run -p 8501:8501 -v ./Data:/app/Data xai-ids
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
