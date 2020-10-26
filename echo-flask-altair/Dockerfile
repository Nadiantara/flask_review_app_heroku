FROM python:3.7-slim

# ADD local.conf /etc/nginx/conf.d/

# RUN echo $(ls /etc/nginx/conf.d/)

COPY requirements.txt flask_test config.py ./

RUN pip install -r requirements.txt && python -m nltk.downloader stopwords