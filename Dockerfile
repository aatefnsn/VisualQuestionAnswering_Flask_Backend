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
COPY . ./

# Install dependencies first (for better layer caching)
RUN pip install -U flask-cors
RUN pip install --no-cache-dir -r requirements.txt

# Download the trained model checkpoint from Azure Blob Storage during build
RUN mkdir -p app && rm -f app/checkpoint_17_Ahmed_768_new.pth.tar

RUN apt-get update && apt-get install -y --no-install-recommends curl

# Create a script to handle model download
RUN cat > /tmp/download_model.sh << 'EOF'
#!/bin/bash
set -e

if [ -z "$AZURE_SAS_TOKEN" ]; then
    echo "AZURE_SAS_TOKEN not provided - model will load on first use"
    exit 0
fi

echo "Downloading model checkpoint from Azure..."
MODEL_URL="https://vqastorage6305.blob.core.windows.net/models/checkpoint_17_Ahmed_768_new.pth.tar"
MODEL_FILE="/app/checkpoint_17_Ahmed_768_new.pth.tar"

curl -L --max-time 600 --retry 5 --retry-delay 10 \
  -o "$MODEL_FILE" \
  "${MODEL_URL}?${AZURE_SAS_TOKEN}"

MODEL_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
echo "Downloaded file size: $MODEL_SIZE bytes"

if [ "$MODEL_SIZE" -lt 500000000 ]; then
    echo "ERROR: File is too small ($MODEL_SIZE bytes). Expected >= 500000000"
    exit 1
fi

echo "✓ Model downloaded successfully!"
EOF
chmod +x /tmp/download_model.sh

RUN /tmp/download_model.sh

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
