#FROM continuumio/anaconda3:2020.11
#FROM ubuntu:latest
#FROM python:3.8
#FROM python:3.9-bullseye
#FROM python:3.8.3-slim
FROM python:3.10-slim

ARG AZURE_SAS_TOKEN

ENV PYTHONUNBUFFERED True
ENV PORT 8080
ENV TRANSFORMERS_CACHE /mnt/bertcache
ENV HF_HOME /mnt/bertcache
ENV TORCH_HOME /mnt/bertcache

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

RUN echo "=== Starting Docker build ===" && \
    echo "AZURE_SAS_TOKEN length: ${#AZURE_SAS_TOKEN}" && \
    echo "APP_HOME: $APP_HOME" && \
    echo "PWD: $(pwd)" && \
    echo "Listing build context files:" && \
    ls -la /

COPY . ./

RUN echo "=== After COPY ===" && \
    ls -la . && \
    echo "Files copied successfully"

# Install dependencies first (for better layer caching)
RUN echo "=== Installing flask-cors ===" && \
    pip install -U flask-cors

RUN echo "=== Installing requirements ===" && \
    pip install --no-cache-dir -r requirements.txt

# Install curl for model download
RUN echo "=== Installing curl ===" && \
    apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Download model checkpoint from Azure using the script
RUN echo "=== Preparing download script ===" && \
    echo "Script exists: $([ -f download_model.sh ] && echo 'YES' || echo 'NO')" && \
    echo "Script content:" && \
    head -5 download_model.sh && \
    echo "Making script executable..." && \
    chmod +x download_model.sh && \
    echo "=== Running download script ===" && \
    env | grep AZURE_SAS_TOKEN && \
    ./download_model.sh && \
    echo "=== Download script completed ==="

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
