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


#ADD . /app
#WORKDIR /app

#RUN set -xe \
#    && apt-get -y update \
#    && apt-get -y install python3-pip
#RUN pip install --upgrade pip

#RUN pip install -r requirements.txt
RUN pip install -U flask-cors
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model during build to avoid runtime delays (commented out due to AMD64 build compatibility issues)
# RUN python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
#ENTRYPOINT ["python", "main.py"]
