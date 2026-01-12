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

# Install dependencies
RUN pip install -U flask-cors
RUN pip install --no-cache-dir -r requirements.txt

# Copy model checkpoint (downloaded in GitHub Actions)
COPY checkpoint_17_Ahmed_768_new.pth.tar /app/

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
