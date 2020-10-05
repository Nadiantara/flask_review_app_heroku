from flask_test import app, forms, db, scheduler, cache
from flask_test.webapp_functions.preprocessing_tools import apple_scrapper, google_scrapper
from flask_cors import cross_origin
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from altair import Chart, X, Y, Axis, Data, DataFormat
from google_play_scraper import app as app_info
import pandas as pd
import pickle as pickle
from scipy.stats import zscore
import scipy.stats as stats
import csv
import sqlite3
import sqlalchemy
import requests
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# DEBUG
import time
from datetime import datetime, timedelta
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
# DEBUG END

#connect to SQLite database
conn = db.engine

# load dummy datasets as a pandas DataFrame
from vega_datasets import data
cars = data.cars()
electricity = data.iowa_electricity()
barley_yield = data.barley()

##########################
# Flask routes
##########################
# render index.html home page

# DEBUG
# scheduler = BackgroundScheduler()
def print_date_time(mytext):
    print(mytext, time.strftime("%A, %d. %B %Y %I:%M:%S %p"))
def schedule_dummy_time_job():
    # Define Scheduler
    run_date = datetime.now() + timedelta(seconds=5)
    scheduler.add_job(func=print_date_time, trigger="date", run_date=run_date, args=["this text"])
    # Shut down the scheduler when exiting the app
# DEBUG END

#Error handler
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.route("/")
@app.route("/home")
@cross_origin()
# @cache.cached()
def index():
    QUERIED_TABLE = None
    playstore_id = request.cookies.get("playstore_id")
    country_id = request.cookies.get("country_code")
    appinfo = app_info(playstore_id, lang='en', country=country_id)
    # check if the queried cookies exist
    if(playstore_id is not None and country_id is not None):
        temp_name = f"{playstore_id.lower()}_{country_id.lower()}"    
        # check if table exists 
        if(conn.dialect.has_table(conn.connect(), temp_name)):
            QUERIED_TABLE = temp_name
            from flask_test.webapp_functions.visualization_plot_plotly import make_plots
            print(QUERIED_TABLE)
            plots = make_plots(QUERIED_TABLE)
            return render_template('index.html', title='Home', plots=plots, appinfo=appinfo["title"])
        else:
            QUERIED_TABLE = None
            return render_template('user_landing.html', title='Welcome')
    return render_template('user_landing.html', title='Welcome')   
            
            



@app.route("/scrape", methods=["POST"])
def scrape():
    playstore_id = request.cookies.get("playstore_id")
    country_id = request.cookies.get("country_code")
    temp_name = f"{playstore_id.lower()}_{country_id.lower()}"
    if(conn.dialect.has_table(conn.connect(), temp_name)):
        QUERIED_TABLE = temp_name
        data = {'message': 'DB Table Exist', 'code': 'SUCCESS'}
        
    else:
        condition = google_scrapper(request.form["play_store_id"], request.form["google_country"], conn)
        if condition == False:
            abort(404)
        data = {'message': 'DB Table Created', 'code': 'SUCCESS'}
    return make_response(jsonify(data), 200)
    
    

@app.route("/submit", methods=["POST"])
@cross_origin()
def submit_form():
    #retrieving user input
    #start_date 
    date_dep = request.form["Dep_Time"]
    start_date = str(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M"))
    #end_date
    date_arr = request.form["Arrival_Time"]
    end_date = str(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M"))
    print(end_date)
    APPID = request.form["app_id"]
    PLAYSTORE_ID = request.form["play_store_id"]
    APPLE_COUNTRY = request.form["country_code"].upper()
    GOOGLE_COUNTRY = request.form["country_code"].lower()
    
    #check if user input is valid or not
    STOREURL = f'https://play.google.com/store/apps/details?id={PLAYSTORE_ID}&hl={GOOGLE_COUNTRY}'
    url_res = requests.get(STOREURL)
    
    #if valid do this
    if url_res.status_code == 200:
        res = make_response(render_template("loading.html", play_store_id=PLAYSTORE_ID, google_country=GOOGLE_COUNTRY, loading_visibility="hidden", home_visibility=""))
        res.set_cookie("playstore_id", request.form["play_store_id"])
        res.set_cookie("country_code", request.form["country_code"])
        res.set_cookie("start_date", start_date)
        res.set_cookie("end_date", end_date)

        print(request.form)
        return res
    #else
    abort(404)


@app.route("/about")
def about():
    return render_template('about.html', title='About')




