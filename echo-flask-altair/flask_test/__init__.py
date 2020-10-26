from flask import Flask
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

#configuration
app = Flask(__name__)
app.config.from_object('config.Config')
db = SQLAlchemy(app)
cache = Cache(app)

scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


from flask_test import routes