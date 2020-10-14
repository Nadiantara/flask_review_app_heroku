import csv
import datetime
import emoji
import gensim
import json
import math
import matplotlib.pyplot as plt
import matplotlib.dates
import nltk
import numpy as np
import operator
import os  # accessing directory structure
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import requests
import re
import regex
import seaborn as sns
import semver
import spacy
import statistics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm
import string
import time
import wordcloud
from bs4 import BeautifulSoup
from datetime import datetime
from distutils.version import LooseVersion
from distutils.version import StrictVersion
from gensim import corpora
from matplotlib.dates import date2num
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords
import altair as alt
from textblob import TextBlob
from tqdm import tqdm
from wordcloud import WordCloud
from scipy.ndimage import gaussian_filter1d
from flask_test.webapp_functions.preprocessing_tools import get_by_date

from flask_test.webapp_functions.preprocessing_tools import change_google_dtype
from flask_test.webapp_functions.processing_tools import get_negreview_topic, get_importancescore, get_urgencyscore
from flask_test import db, cache

def OLD_plot_alt_totalreview_1(dataframe, start_date, end_date):
  '''
  This function is used to plot total review

  Parameters:
  dataframe-- data from get_crawl_google (but we can put data from apple scrapper if the columns names are identical)
    
  returns:
  plot
  '''
  # Bar Chart " User Total Review across Months"

  # Preprocessing
  start_date = start_date
  end_date = end_date
  df = get_by_date(dataframe, start_date, end_date) 
  df['at'] = pd.to_datetime(df['at'])
  df.index = df['at']
  df = pd.DataFrame(df['review'].resample('MS').count()).join(pd.DataFrame(df['rating'].resample('MS').mean()))

  # Defining selection
  highlight = alt.selection(type='single', on='mouseover',
                        fields=['symbol'], nearest=True)

  # Plot
  base = alt.Chart(df.reset_index()).encode(
      alt.X('at:T', axis=alt.Axis(
                                  # title=" User Total Review across Months", 
                                  grid=False,
                                  format=("%m/%Y"), 
                                  tickCount=7, 
                                  titleFontSize=20, 
                                  # labelColor='black'
                                  )),
      tooltip=['at','review']
  ).properties(
      width=alt.Step(40),
      height=500
  )

  bar = base.mark_bar(opacity=1, color='#15D7D6', size=30).encode(
      alt.Y('review:Q',
            axis=alt.Axis(labels=False, title=None, grid=False, tickCount=0, titleFontSize=12, labelColor='#999999')),
            tooltip=['at','review'],       
  )
  points = base.mark_circle().encode(
      opacity=alt.value(0) 
  #Functions selections
  ).add_selection(
      highlight
  )

  plot = alt.layer(bar, points).resolve_scale(
      y = 'independent'
  ).configure_title().configure_axis(
      domain=False,
      grid=False,
      # labelFontSize = 12,
      # titleFontSize = 25,
  ).configure_view(
      strokeOpacity=0,
      strokeWidth=0,
      # height=700,
      # width=1700
  )
  return plot

def plot_alt_totalreview_1(dataframe, start_date, end_date):
  '''
  This function is used to plot total review

  Parameters:
  dataframe-- data from get_crawl_google (but we can put data from apple scrapper if the columns names are identical)
    
  returns:
  plot
  '''
  # Bar Chart "5Miles User Total Review across Months"

  # Preprocessing
  start_date = start_date
  end_date = end_date
  df = get_by_date(dataframe, start_date, end_date) 
  df['at'] = pd.to_datetime(df['at'])
  df.index = df['at']
  df = pd.DataFrame(df['review'].resample('MS').count()).join(pd.DataFrame(df['rating'].resample('MS').mean()))
  
  # Defining selection
  highlight = alt.selection(type='single', on='mouseover',
                          fields=['symbol'], nearest=True)
  
  # Plot
  base = alt.Chart(df[-12:].reset_index()).encode(
      alt.X('at:T', timeUnit= 'yearmonth', axis=alt.Axis(labels=True, title=None, grid=False, tickCount=12, labelColor='black')),
      tooltip=[alt.Tooltip('review:Q'),
               alt.Tooltip('at:T', format='%b, %Y')]
  ).properties(
        width=alt.Step(60),
        height=300
  )
  
  bar = base.mark_bar(opacity=1, color='#15D7D6', size=40).encode(
      alt.Y('review:Q',
            axis=alt.Axis(labels=False, title=None, grid=False, tickCount=0, titleFontSize=12, labelColor='#999999')),
            tooltip=[alt.Tooltip('review:Q'),
                     alt.Tooltip('at:T', format='%b, %Y')]     
  )
  points = base.mark_circle().encode(
      opacity=alt.value(0) 

  #Functions selections
  ).add_selection(
      highlight
  )

  plot = alt.layer(bar, points).resolve_scale(
      y = 'independent'

  ).configure_title().configure_axis(
      domain=False,
      grid=False,
      labelFontSize = 12,
      titleFontSize = 25  
  ).configure_view(
      strokeOpacity=0,
      strokeWidth=0,
      width=700,
      height=500
  )
  return plot

def OLD_plot_alt_totalreview_2(dataframe):
    '''
    This function is used to plot total review by version

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    
    returns:
    plot
    '''
    # Bar Chart " User Total Review across Months"

    # Preprocessing

    df = dataframe
    conditions = [
        (df['rating'] > 3),
        (df['rating'] <= 3)
        ]
    values = ['positive review', 'negative review']
    df['label'] = np.select(conditions, values)
    grouped_multiple = df.groupby(['version', 'label']).agg({'review': ['count']})
    grouped_multiple.columns = ['review']
    grouped_multiple = grouped_multiple.reset_index()

    # Defining selection
    highlight = alt.selection(type='single', on='mouseover',
                          fields=['symbol'], nearest=True)
    color_scale = alt.Scale(
        domain=[
            "positive review",
            "negative review"
                ],
        range=["#19F509", "#FC1D12",]
    )

    # Plot
    base = alt.Chart(grouped_multiple).encode(
        alt.X('version:O', axis=alt.Axis(
                                          # title=" User Total Review across Version", 
                                          grid=False, 
                                          # titleFontSize=30, 
                                          # labelColor='black', 
                                          labels=False
                                          )),
        tooltip=['version', 'label:N', 'review']
    ).properties(
        width=alt.Step(50),
        height=500
    )

    bar = base.mark_bar(opacity=1, color='#15D7D6', size=40).encode(
        alt.Y('review:Q',
              axis=alt.Axis(labels=False, title=None, grid=False, tickCount=0, titleFontSize=12, labelColor='#999999')),
              tooltip=['version', 'label','review'],
              color=alt.Color('label:N', scale=color_scale)     
    ) 
    points = base.mark_circle().encode(
        opacity=alt.value(1),
    #Functions selections
    ).add_selection(
        highlight
    )

    plot = alt.layer(bar).resolve_scale(
        y = 'independent'
    ).configure_title().configure_axis(
        domain=False,
        grid=False,
        labelFontSize = 12,
        titleFontSize = 25,
    ).configure_view(
        strokeOpacity=0,
        strokeWidth=0,
        # height=500,
        # width=900
    )
    return plot

def plot_alt_totalreview_2(dataframe, start_date, end_date):
    '''
    This function is used to plot total review by version

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    
    returns:
    plot
    '''
    # Bar Chart "5Miles User Total Review across Months"

    # Preprocessing
    # start_date = start_date
    # end_date = end_date
    # df = get_by_date(dataframe, start_date, end_date) 
    # df['at'] = pd.to_datetime(df['at'])
    # df.index = df['at']
    df = dataframe.copy()
    conditions = [
        (df['rating'] > 3),
        (df['rating'] <= 3)
        ]
    values = ['positive review', 'negative review']
    df['label'] = np.select(conditions, values)
    grouped_multiple = df.groupby(['version', 'label']).agg({'review': ['count']}) 
    grouped_multiple.columns = ['review']
    grouped_multiple.reset_index(inplace=True)
    grouped_multiple = grouped_multiple.loc[grouped_multiple['version'].isin(grouped_multiple['version'].unique()[-13:])]


    # Defining selection
    highlight = alt.selection(type='single', on='mouseover',
                          fields=['symbol'], nearest=True)
    color_scale = alt.Scale(
        domain=[
            "positive review",
            "negative review"
                ],
        range=["#19F509", "#FC1D12",]
    )

    # Plot
    base = alt.Chart(grouped_multiple).encode(
        alt.X('version:O', axis=alt.Axis(title=None, ticks=False, grid=False, titleFontSize=20, labelColor='black', labels=True)),
        tooltip=['version', 'label:N', 'review']
    ).properties(
        width=alt.Step(50),
        height=300
    )
  
    bar = base.mark_bar(opacity=1, size=40).encode(
        alt.Y('review:Q',
              axis=alt.Axis(labels=False, title=None, grid=False, tickCount=0, titleFontSize=12, labelColor='#999999')),
              tooltip=['version', 'label','review'],
              color=alt.Color('label:N', legend=None, scale=color_scale)
    )
    points = base.mark_circle().encode(
        opacity=alt.value(1),
 
    #Functions selections
    ).add_selection(
        highlight
    )

    plot = alt.layer(bar).resolve_scale(
        y = 'independent'
    ).configure_title().configure_axis(
        domain=False,
        grid=False,
        labelFontSize = 12,
        titleFontSize = 25,
    ).configure_view(
        strokeOpacity=0,
        strokeWidth=0,
    )
    return plot

def plot_alt_rating_3(dataframe):
    '''
    This function is used to plot total review by version

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    
    returns:
    plot
    '''
    # Bar Chart "5Miles User Rating"
    # Preprocessing
    df = dataframe
    grouped_multiple = df.groupby(['rating']).agg({'review': ['count']})
    grouped_multiple.columns = ['review']
    grouped_multiple = grouped_multiple.reset_index()
    grouped_multiple.sort_values(by=['rating', 'review'], inplace=True, ascending=False)
    grouped_multiple['value'] = (grouped_multiple['review'] / grouped_multiple['review'].sum()) * 100
    grouped_multiple['value'] = grouped_multiple['value'].round(2).astype(str) + '%'
    grouped_multiple['star'] = grouped_multiple['rating']
    grouped_multiple['star'] = grouped_multiple['star'].astype(str) + '★'
    grouped_multiple
    
    #Defining selection
    highlight = alt.selection(type='single', on='mouseover',
                              fields=['symbol'], nearest=True)
    
    # Plot
    # Double percentage calculation, needs to be improved
    base = alt.Chart(grouped_multiple).transform_joinaggregate(
        TotalReview='sum(review)',
    ).transform_calculate(
        PercentofReview='datum.review / datum.TotalReview' 
    ).mark_bar().encode(
        alt.X('PercentofReview:Q', axis=alt.Axis(title=None, grid=False, titleFontSize=20, labelColor='black', tickCount=0, format=('.0%'), labels=False)),
        alt.Y('star:N', sort=alt.EncodingSortField(field="rating", op="count", order='ascending'),
            # axis=alt.Axis(labels=True, title=None, grid=False, tickCount=0, ticks=False, labelColor='black', labelFontSize=15)),
            axis=alt.Axis(grid=False)),
            tooltip=['value'],
    ).properties(
        width=350,
        height=250
    )
    # bar = base.mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, cornerRadiusBottomLeft=10, cornerRadiusBottomRight=10, opacity=1, color='#8BFC12', size=20).encode(
    bar = base.mark_bar().encode(
            color=alt.Color('rating', legend=None, 
                              scale=alt.Scale(
                                  domain=['1', '2', '3', '4', '5'],
                                  range=['#FF0000', '#ff6600', '#FFFF00', '#99ff00', '#33ff00'])
          ) #Color condition is too complicated, should be improved
    )

    text = base.mark_text(
          align='left',
          baseline='middle',
          dx=3,
        ).encode(
            text=alt.Text('PercentofReview:Q', format=",.3f")
        )

    plot = alt.layer(bar, text).resolve_scale(
        # y = 'independent'
    ).configure_title().configure_axis(
        domain=False,
        grid=False,
        labelFontSize = 100,
        titleFontSize = 100,
    ).configure_view(
        strokeOpacity=1,
        strokeWidth=0,
        height=100,
        width=700    
    ).configure_axis(
        grid=True, 
        domain=False
    ).properties(height=400)

    return plot 

def plot_alt_total_negative_three(DATAFRAME, NEG_DATAFRAME, TOPIC, start_date, end_date):
  '''
  Description: Function to show bar chart for complaint by review (v.3).

  Parameter: Total dataframe, negative dataframe, list of topics.

  Return: Altair Chart.

  '''
  # Summarize the data in each month
  df_init = DATAFRAME
  df_neg = NEG_DATAFRAME
  topic = TOPIC
  
  start_date = start_date
  end_date = end_date
  df = get_by_date(DATAFRAME, start_date, end_date)

  
  df.index = pd.to_datetime(df['at'])
  df_neg = get_by_date(df_neg, start_date, end_date)
  df_neg.index = pd.to_datetime(df_neg['at'])
  df_neg = df_neg[topic].resample('MS').sum().join(pd.DataFrame(df['review'].resample('MS').count()))

  for kolom in topic:
    df_neg[kolom] = df_neg[kolom].astype(float)

  df_neg.reset_index(inplace=True)

  # Calculate the data and turn it into percent
  for i in range(len(df_neg)):
    for kolom in topic:
      if df_neg[kolom][i] > 0:
        df_neg.at[i,kolom] = round((df_neg[kolom][i] / df_neg['review'][i] * 100),2)

  df_neg.index = df_neg['at']

  # Drop the review column
  no_review = df_neg.drop(['review'], axis=1)

  # Melting the dataframe into something that can easily process in Altair
  new_data = no_review.melt(id_vars=["at"], var_name="Topic", value_name="Total")

  # Dropdown input to filter the bar but, as I expected but still a little bit messy on the look 
  input_dropdown = alt.binding_select(options=topic+[None])
  selection = alt.selection_single(fields=['Topic'], bind=input_dropdown, name='Topic')
  color = alt.condition(selection,
                      alt.Color('Topic:N', legend=None),
                      alt.value('lightgray'))

  chart = alt.Chart(new_data, 
                    # title="User Complaints"
  ).mark_bar().encode(
    column=alt.Column('at'),
    x=alt.X('Topic', axis=None),
    y=alt.Y('Total', axis=None),
    color=color,
    tooltip=['Topic', 'Total', 'at']
  ).add_selection(selection).configure_title(fontSize=18).configure_axis(grid=False)

  return chart

def plot_alt_total_negative_trendline(DATAFRAME, NEG_DATAFRAME, TOPIC, start_date, end_date):
  '''
  Description: Function to show trendline for complaint by review.

  Parameter: Total dataframe, negative dataframe, list of topics.

  Return: Altair Chart.

  '''
  # Summarize the data in each month
  df_init = DATAFRAME
  df_neg = NEG_DATAFRAME
  topic = TOPIC

  start_date = start_date
  end_date = end_date
  df = get_by_date(DATAFRAME, start_date, end_date)

  df.index = pd.to_datetime(df['at'])
  df_neg = get_by_date(df_neg, start_date, end_date)
  df_neg.index = pd.to_datetime(df_neg['at'])
  df_neg = df_neg[topic].resample('MS').sum().join(
      pd.DataFrame(df['review'].resample('MS').count()))

  for kolom in topic:
    df_neg[kolom] = df_neg[kolom].astype(float)

  df_neg.reset_index(inplace=True)

  # Calculate the data and turn it into percent
  for i in range(len(df_neg)):
    for kolom in topic:
      if df_neg[kolom][i] > 0:
        df_neg.at[i,kolom] = round((df_neg[kolom][i] / df_neg['review'][i] * 100),2)

  df_neg.index = df_neg['at']

  # Drop the review column
  no_review = df_neg.drop(['review'], axis=1)

  # Melting the dataframe into something that can easily process in Altair
  new_data = no_review.melt(id_vars=["at"], var_name="Topic", value_name="Total")

  # Calculating the Gaussian to smoothen the trendline
  a = gaussian_filter1d(new_data['Total'], 2, mode='nearest')

  # Adding the result as a new column to the dataframe
  new_data['gauss'] = a

  # Make a trendline with drowdown input
  input_dropdown = alt.binding_select(options=topic+[None])
  selection = alt.selection_single(fields=['Topic'], bind=input_dropdown, name='Selected')
  color = alt.condition(selection,
                      alt.Color('Topic:N', legend=None),
                      alt.value('lightgray'))

  chart = alt.Chart(new_data, 
                    # title="User Complaints Trendline"
  ).mark_line(
  ).encode(
      x=alt.X('at', title='Time'),
      y=alt.Y('gauss', title='Complaint by Reviews (%)', axis=None),
      color=color,
      tooltip=['Topic', 'gauss']
  ).properties(width=700, height=500).configure_title(fontSize=20
  ).add_selection(selection).transform_filter(selection).configure_title(fontSize=18).configure_axis(grid=False).configure_view(
        strokeWidth=0
  ).configure_view(
        width=700,    
        height=500,
  )
  return chart

def plot_alt_prioritymatrix(dataframe, neg_dataframe, topic, x=6): 
    '''
    This function is used to plot problem solving priority matrix

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    neg_dataframe-- neg_df from get_negreview_topic
    topic-- topic from get_negreview_topic
    x-- number of months to inspect

    returns:
    priority matrix
    '''
    # priority dataframe from importance score and urgency score
    priority_score = pd.merge(get_importancescore(dataframe, neg_dataframe, topic, x),get_urgencyscore(dataframe, neg_dataframe, topic, x), on='Topic')
    
    # min max normalization
    importance_score_scaled=[]
    urgency_score_scaled=[]
    score=[]
    for i in range(len(priority_score)):
      z = (priority_score['importance_score'][i] - priority_score['importance_score'].min(axis=0)) / (priority_score['importance_score'].max(axis=0) - priority_score['importance_score'].min(axis=0))
      importance_score_scaled.insert(i,z)
      c = (priority_score['urgency_score'][i] - priority_score['urgency_score'].min(axis=0)) / (priority_score['urgency_score'].max(axis=0) - priority_score['urgency_score'].min(axis=0))
      urgency_score_scaled.insert(i,c)
      a = c+z
      score.insert(i,a)
    priority_score['importance_score_scaled'] = importance_score_scaled
    priority_score['urgency_score_scaled'] = urgency_score_scaled
    priority_score['score'] = score
    priority_score.sort_values(by='score',inplace=True,ascending=False)
    priority_score.reset_index(inplace=True,drop=True)
    priority = list(range(1,len(priority_score)+1))
    priority_score['priority'] = priority


    # plot
    chart = alt.Chart(priority_score ).mark_circle(size=250).encode(
            alt.X('importance_score_scaled', axis=alt.Axis(title='Importance', values=[0,0.5,1])),
            alt.Y('urgency_score_scaled', axis=alt.Axis(title='Urgency', values=[0,0.5,1])),
            tooltip=['Topic','priority'],
            color=alt.Color('Topic', legend=None)
            ).properties(width=700, height=500).configure_title(fontSize=20)
  
    return chart

def apps_rating (dataframe):
  '''
    This function is used to get the overall apps rating

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    
    returns:
    apps rating value
  '''
  overall_apps_rating = round(statistics.mean(dataframe['rating']),1)
  return overall_apps_rating

def apps_review (dataframe):
  '''
    This function is used to get the total number of reviews

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    
    returns:
    number of reviews
  '''
  total_review = len(dataframe)
  return total_review

def make_basic_plots_and_stats(table_name, start_date, end_date):
  #connect to SQLite database
  start_date = start_date
  end_date = end_date
  conn = db.engine
  # fetch dataset
  fetched_df = pd.read_sql_table(table_name, conn)

  # preprocess
  #   Disable for a while to test newly integrated plots
  #   processed_df = change_google_dtype(fetched_df)
  '''
  Order:
    - review accross time
    - review accross version
    - total rating
  '''
  return json.dumps([
    plot_alt_totalreview_1(fetched_df, start_date, end_date).to_json(),
    # plot_alt_totalreview_2(fetched_df, start_date, end_date).to_json(),
    plot_alt_totalreview_2(fetched_df, start_date, end_date).to_json(),
    plot_alt_rating_3(fetched_df).to_json(),
    {
      "stats": {
        "average_rating": apps_rating(fetched_df),
        "n_reviews": apps_review(fetched_df),
      }
    }
  ])


def make_sentiment_plots(table_name, start_date, end_date):
  #connect to SQLite database
  start_date = start_date
  end_date = end_date
  conn = db.engine
  # fetch dataset
  fetched_df = pd.read_sql_table(table_name, conn)
  neg_df, topic = get_negreview_topic(fetched_df)

  # preprocess
  #   Disable for a while to test newly integrated plots
  #   processed_df = change_google_dtype(fetched_df)
  '''
  Order:
    - Negative Review count
    - Trendline
    - Priority matrix
  '''
  return json.dumps([
      plot_alt_total_negative_three(
          fetched_df, neg_df, topic, start_date, end_date).to_json(),
      plot_alt_total_negative_trendline(
          fetched_df, neg_df, topic, start_date, end_date).to_json(),
    plot_alt_prioritymatrix(fetched_df, neg_df, topic).to_json()
  ])

