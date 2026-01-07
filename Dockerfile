#FROM continuumio/anaconda3:2020.11
#FROM ubuntu:latest
#FROM python:3.8
#FROM python:3.9-bullseye
#FROM python:3.8.3-slim
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV PORT 8080
ENV TRANSFORMERS_CACHE /mnt/bertcache
ENV HF_HOME /mnt/bertcache
ENV TORCH_HOME /mnt/bertcache

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install dependencies first (for better layer caching)
RUN pip install -U flask-cors
RUN pip install --no-cache-dir -r requirements.txt

# Download the trained model checkpoint from Azure Blob Storage during build
# Note: We remove any local copy first (which may be a Git LFS pointer from GitHub Actions)
# and download a fresh copy from Azure Blob Storage to ensure integrity
RUN mkdir -p app && \
    rm -f app/checkpoint_17_Ahmed_768_new.pth.tar && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    echo "========================================" && \
    echo "Downloading model checkpoint from Azure..." && \
    echo "========================================" && \
    MODEL_URL="https://vqastorage6305.blob.core.windows.net/models/checkpoint_17_Ahmed_768_new.pth.tar" && \
    curl -f -L --max-time 600 --retry 3 --retry-delay 5 \
    -o app/checkpoint_17_Ahmed_768_new.pth.tar \
    "$MODEL_URL" && \
    if [ ! -f app/checkpoint_17_Ahmed_768_new.pth.tar ]; then \
        echo "ERROR: Model download failed - file not found"; \
        exit 1; \
    fi && \
    MODEL_SIZE=$(stat -c%s app/checkpoint_17_Ahmed_768_new.pth.tar) && \
    echo "Downloaded model size: $((MODEL_SIZE / 1048576)) MB" && \
    if [ $MODEL_SIZE -lt 500000000 ]; then \
        echo "ERROR: Model file too small ($MODEL_SIZE bytes), download likely incomplete"; \
        exit 1; \
    fi && \
    echo "✓ Model checkpoint downloaded successfully!" && \
    echo "========================================" && \
    apt-get remove -y curl && apt-get autoremove -y

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
