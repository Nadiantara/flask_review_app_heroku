FROM python:3.7

# ADD local.conf /etc/nginx/conf.d/

# RUN echo $(ls /etc/nginx/conf.d/)

COPY requirements.txt flask_test config.py ./

RUN apt-get install -y libpq-dev 
RUN pip install -r requirements.txt && python -m nltk.downloader stopwords