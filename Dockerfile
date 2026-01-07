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

# Install curl for model download
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Download model checkpoint from Azure using the script
RUN chmod +x download_model.sh && AZURE_SAS_TOKEN="$AZURE_SAS_TOKEN" ./download_model.sh

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
