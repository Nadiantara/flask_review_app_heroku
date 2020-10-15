from flask_test import app, forms, db, scheduler, cache
from flask_test.webapp_functions.preprocessing_tools import apple_scrapper, google_scrapper
from flask_test.webapp_functions.processing_tools import get_importancescore, get_urgencyscore, get_priority_score_scaled
from flask_test.webapp_functions.visualization_plot_altair import make_basic_plots_and_stats, make_sentiment_plots
from flask_test.forms import AppForm, _guess_store, validate_appid
from flask_cors import cross_origin
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from altair import Chart, X, Y, Axis, Data, DataFormat
from google_play_scraper import app as app_info
import pandas as pd
import numpy as np
import pickle
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
# render content.html home page

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
    form = AppForm()
    QUERIED_TABLE = None
    app_id = request.cookies.get("app_id")
    country_code = request.cookies.get("country_code")
 
    if(app_id is not None and country_code is not None):
        store_type = _guess_store(app_id)
        
        temp_name = f"{str(app_id)}_{str(country_code)}"
        
        # check if table exists 
        if(conn.dialect.has_table(conn.connect(), temp_name)):
            QUERIED_TABLE = temp_name
            print(QUERIED_TABLE)
            fetched_df = pd.read_sql_table(temp_name, conn)
            feature_list = ["Topic", "importance_score_scaled", "score"]
            priority_score_scaled = pd.DataFrame("initializing . . .", index=range(6), columns=feature_list)
            #priority_score_scaled.to_csv(
                #"flask_test/webapp_functions/models/priority_score_scaled.csv", index=False)
            return render_template('content.html', title='Home', form=form, content_title=app_id, priority_score_scaled=priority_score_scaled)
        else:
            QUERIED_TABLE = None
            return render_template('user_landing.html', title='Welcome', form=form)
    
    return render_template('user_landing.html', title='Welcome', form=form)
                
        
@app.route("/scrape", methods=["POST"])
def scrape():
    """
    Scraping data perform checking with _guess_store and scrape it accordingly
    """
    start_date = request.cookies.get("start_date")
    end_date = request.cookies.get("end_date")
    app_id = request.cookies.get("app_id")
    country_code = request.cookies.get("country_code")
    store_type = _guess_store(app_id)
    if store_type == "PlayStore":
        
        temp_name = f"{app_id}_{country_code}"

        if(conn.dialect.has_table(conn.connect(), temp_name)):
            print("TEMP NAME1", temp_name)
            QUERIED_TABLE = temp_name
            data = {'message': 'DB Table Exist', 'code': 'SUCCESS'}
            
        else:
            print("TEMP NAME2", temp_name)
            condition = google_scrapper(
                app_id, country_code, conn)
            if condition == False:
                abort(404)
            data = {'message': 'DB Table Created', 'code': 'SUCCESS'}
            
    elif store_type == "AppStore":
        temp_name = f"{app_id}_{country_code}"
        print("REQUEST", request)

        if(conn.dialect.has_table(conn.connect(), temp_name)):
            print("TEMP NAME1", temp_name)
            QUERIED_TABLE = temp_name
            data = {'message': 'DB Table Exist', 'code': 'SUCCESS'}
            
        else:
            print("TEMP NAME2 will be downloaded", temp_name)
            condition = apple_scrapper(
                app_id, country_code, conn)
            if condition == False:
                abort(404)
            data = {'message': 'DB Table Created', 'code': 'SUCCESS'}
        
    return make_response(jsonify(data), 200)
    
    
@app.route("/submit", methods=["POST"])
@cross_origin()
def submit_form():
    form = AppForm()
    store_list = ["AppStore", "PlayStore"]
    feature_list = ["Topic", "importance_score_scaled", "score"]
    priority_score_scaled = pd.DataFrame(
        "initializing . . .", index=range(6), columns=feature_list)
    #new way retrieving user input
    if form.validate_on_submit():
        
        date_start = request.form["start_date"]
        start_date = str(pd.to_datetime(date_start, format="%Y-%m-%dT%H:%M"))
        
        date_end = request.form["end_date"]
        end_date = str(pd.to_datetime(date_end, format="%Y-%m-%dT%H:%M"))

        app_id = form.app_id.data
        country_code = request.form["country_code"]
        
        feature_list = ["Topic", "importance_score_scaled", "score"]
        priority_score_scaled = pd.DataFrame(
            "initializing . . .", index=range(6), columns=feature_list)
        store_type = _guess_store(app_id)
        if store_type == "PlayStore":
            STOREURL = f'https://play.google.com/store/apps/details?id={app_id}&hl={country_code}'
            url_res = requests.get(STOREURL)
        elif store_type == "AppStore":
            STOREURL = f"http://apps.apple.com/{country_code}/app/id{app_id}"
            url_res = requests.get(STOREURL)
        else:
            abort(404)
    
        if store_type in store_list and url_res.status_code == 200:
            res = make_response(render_template(
                                    "loading.html", 
                                    start_date=start_date, 
                                    end_date=end_date, 
                                    app_id=app_id,
                                    country_code=country_code, 
                                    loading_visibility="hidden", 
                                    home_visibility="", 
                                    form=form,
                                    priority_score_scaled=priority_score_scaled,
                                    content_title=""
                                    ))

            res.set_cookie("app_id", form.app_id.data)
            res.set_cookie("country_code", request.form["country_code"])
            res.set_cookie("start_date", start_date)
            res.set_cookie("end_date", end_date)
            return res
        else:
            abort(404)

    return render_template('user_landing.html', title='Welcome', form=form, priority_score_scaled=priority_score_scaled)


@app.route("/basic-plots")
def fetch_basic_plots():
    start_date = request.cookies.get("start_date")
    end_date = request.cookies.get("end_date")
    app_id = request.cookies.get("app_id")
    store_type = _guess_store(app_id)
    if store_type == "PlayStore":
        country_code = request.cookies.get("country_code")
        temp_name = f"{app_id}_{country_code}"
    else:
        country_code = request.cookies.get("country_code")
        temp_name = f"{app_id}_{country_code}"
    

    return (make_basic_plots_and_stats(temp_name, start_date, end_date))

@app.route("/sentiment-plots")
def fetch_sentiment_plots():
    start_date = request.cookies.get("start_date")
    end_date = request.cookies.get("end_date")
    app_id = request.cookies.get("app_id")
    store_type = _guess_store(app_id)
    if store_type == "PlayStore":
        country_code = request.cookies.get("country_code")
        temp_name = f"{app_id}_{country_code}"
    else:
        country_code = request.cookies.get("country_code")
        temp_name = f"{app_id}_{country_code}"

    return (make_sentiment_plots(temp_name, start_date, end_date))

@app.route("/about")
def about():
    return render_template('about.html', title='About')




