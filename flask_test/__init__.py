from flask import Flask
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import nltk

#configuration
app = Flask(__name__)
app.config.from_object('config.Config')
db = SQLAlchemy(app)
cache = Cache(app)
nltk.download('stopwords')

scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


from flask_test import routes