import numpy as np
from datetime import datetime,timedelta, date
# check your pickle compability, perhaps its pickle not pickle
import pickle as pickle
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report

#APP ID default
#APPID = 1031922175
#PLAYSTORE_ID = 'echo.co.uk'
#sidebar header
#date

# Display version plot for each OS
# def version_plot(selected_store, APPID, PLAYSTORE_ID):
#   if selected_store == 'Apple':
#     from flask_test.webapp_functions.processing_tools import apple_scrapper, apple_plot
#     apple_file, APPID = apple_scrapper(APPID)
#     df_apple = pd.read_csv("dataset/"+ apple_file)
#     apple_plot(df_apple)
 
#   else:
#     from flask_test.webapp_functions.processing_tools import google_scrapper, google_plot
#     google_file, PLAYSTORE_ID = google_scrapper(PLAYSTORE_ID)
#     df_google = pd.read_csv("dataset/"+ google_file)
#     google_plot(df_google)
    



#default echo function
def analyze_review(APPID, PLAYSTORE_ID, start_date='2017-01-01', end_date='2020-08-15'):
  #load default dataset
  from flask_test.webapp_functions.processing_tools import final_by_date, load_default_dataset
  df, combined_df, combined_df_preprocessed, df_google_date, google_neg_rev, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = load_default_dataset(APPID, PLAYSTORE_ID)


  
  google_neg_rev_recent = final_by_date(google_neg_rev, start_date, end_date)
  similar_index=combined_df[combined_df['review'].isin(df['review'])].index
  #version_plot(selected_store, 1031922175, 'echo.co.uk')
  # our default solver is 'liblinear'
  
  #final_df = pd.read_pickle(f'dataset/final_df_{APPID}_{PLAYSTORE_ID}.pkl')
  from flask_test.webapp_functions.processing_tools import get_summed_scores

  #assign each sum result with their label into a dictionary
  sum_dict,normalized_sum_dict = get_summed_scores(google_neg_rev_recent,df.columns[1:7].values.astype(str))
  #Sort the value 
  sum_dict={k: v for k, v in sorted(sum_dict.items(), key=lambda item: item[1])}
  normalized_sum_dict={k: v for k, v in sorted(normalized_sum_dict.items(), key=lambda item: item[1])}
  sum_df = pd.DataFrame.from_dict([sum_dict])
  normalized_sum_df = pd.DataFrame.from_dict([normalized_sum_dict])
  # plot the labeled topic severance with barplot
  plot_dict(normalized_sum_dict)
  
    
#this function still error for input solver
def get_severance(log_r,x_train_count, combined_df_preprocessed, combined_df, df, google_neg_rev_recent, similar_index, APPID, PLAYSTORE_ID):
  vocab_size=x_train_count.shape[1]
  features=pickle.load(open(f"dataset/feature_names_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))
  coeffs=log_r.coef_
  result_dict=dict(tuple([(features[i],coeffs[0][i]) for i in range(vocab_size)]))
  result_df=pd.DataFrame(sorted(result_dict.items(),key=lambda x:x[1]),columns=['word','coeff'])
  #find outlier
  from flask_test.webapp_functions.processing_tools import find_outliers
  lengths=combined_df_preprocessed['review'].apply(lambda x: len(x.split(" "))).values
  lower_outlier,upper_outlier=find_outliers(lengths)
  combined_df_preprocessed['z_score']=zscore(lengths)
  combined_df_preprocessed['lengths']=lengths

  # constructing another one column to previous combined dataset which contain its "zscore" probability
  zscores=combined_df_preprocessed['z_score'].values
  combined_df_preprocessed['probabilities']=[0]*len(combined_df_preprocessed)
  combined_df_preprocessed['probabilities']=combined_df_preprocessed['z_score'].apply(lambda x:1-stats.norm.cdf(x))

  #take a look for the score and assign it to a new dataframe
  from flask_test.webapp_functions.processing_tools import lookup
  score=lookup(combined_df_preprocessed,result_dict)
  log_r_scored_df=combined_df_preprocessed.copy()
  log_r_scored_df['score']=score

  #scaling: divide outlier with its zscore
  from flask_test.webapp_functions.processing_tools import outlier_scaling
  log_r_scored_df_scaled=outlier_scaling(log_r_scored_df,upper_outlier,lower_outlier)


  #get the final dataframe
  combined_df['score']=log_r_scored_df_scaled['score'] 
  final_df=pd.merge(combined_df.loc[similar_index][['review','score']],df,how='inner',on='review')
  # fill null with zero
  final_df.fillna(0,inplace=True)

  from flask_test.webapp_functions.processing_tools import get_summed_scores
  #assign each sum result with their label into a dictionary
  sum_dict, normalized_sum_dict = get_summed_scores(google_neg_rev_recent,df.columns[1:7].values.astype(str))
  #Sort the value 
  sum_dict={k: v for k, v in sorted(sum_dict.items(), key=lambda item: item[1])}
  normalized_sum_dict={k: v for k, v in sorted(normalized_sum_dict.items(), key=lambda item: item[1])}
  sum_df = pd.DataFrame.from_dict([sum_dict])
  normalized_sum_df = pd.DataFrame.from_dict([sum_df])
  return sum_df, sum_dict, normalized_sum_dict, normalized_sum_df


# function for plotting topic severance
def plot_dict(dict_value):
  sns.set(style="whitegrid", context="poster",font_scale=0.6)
  fig,ax=plt.subplots(figsize=(15,8))
  bar=sns.barplot(x=list(dict_value.keys()),y=list(dict_value.values()),ax=ax,palette='rocket')
  bar.set(xlabel='Categories',ylabel='Severance Score')
  plt.show()
  return bar

def load_dataset(APPID, PLAYSTORE_ID):
  df = pd.read_csv(f"dataset/labeled_negative_reviews_with_versions_ratings_type_{APPID}_{PLAYSTORE_ID}.csv")
  combined_df = pd.read_pickle(f'dataset/combined_df_{APPID}_{PLAYSTORE_ID}.pkl')
  combined_df_preprocessed = pd.read_pickle(f'dataset/data_clean_and_mapped_{APPID}_{PLAYSTORE_ID}.pkl')
  df_google_date = pd.read_pickle(f"dataset/df_google_date_{PLAYSTORE_ID}.pkl")
  google_neg_rev = pd.read_pickle(f"dataset/google_neg_rev_{PLAYSTORE_ID}.pkl")
  #pickle non dataframe type
  x_train_count = pickle.load(open(f"dataset/x_train_count_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))
  x_test_count = pickle.load(open(f"dataset/x_test_count_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))

  x_train_tfidf = pickle.load(open(f"dataset/x_train_tfidf_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))
  x_test_tfidf = pickle.load(open(f"dataset/x_test_tfidf_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))

  y_train = pickle.load(open(f"dataset/y_train_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))
  y_test = pickle.load(open(f"dataset/y_test_{APPID}_{PLAYSTORE_ID}.pkl", "rb"))
  
  return df, combined_df, combined_df_preprocessed, df_google_date, google_neg_rev, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test

#######################################################################################
#conditional for selected APP
# def main():
#   #input_solver = selected_solver
#   start_date = start_date_app
#   end_date = end_date_app
#   if selected_APP == 'Echo-NHS':
#     APPID = 1031922175
#     PLAYSTORE_ID = 'echo.co.uk'
#     main_echo(APPID, PLAYSTORE_ID, start_date, end_date, input_solver)
#   else:
#     APPID = 883158179
#     PLAYSTORE_ID = 'com.kwiboo.p2urepeatsapp.android'
#     main_pharmacy2u(APPID, PLAYSTORE_ID, start_date, end_date, input_solver)


