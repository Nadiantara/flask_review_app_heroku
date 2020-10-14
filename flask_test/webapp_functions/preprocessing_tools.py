#basic
import pandas as pd
import numpy as np
import seaborn as sns
import re
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import zscore
import scipy.stats as stats
import sqlalchemy
#scraping
import requests
import json
from datetime import datetime,timedelta, date
from distutils.version import LooseVersion
import math
from tqdm import tqdm
from google_play_scraper import app, reviews, reviews_all, Sort
# for NLP preprocessing
import nltk
import spacy
import pickle
import gensim 
# for main model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report
from sqlalchemy.sql import text
from apscheduler.schedulers.background import BackgroundScheduler
from flask_test import db, scheduler
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from random import randint
#

def delete_table_job(tablename):
  conn = db.engine
  delete_query = f"DROP TABLE '{tablename}'"
  conn.execute(delete_query)


def apple_scrapper(APPID, COUNTRY, db_connection):
    apple_sqlite_table = f'{APPID}_{COUNTRY}'
    if(db_connection.dialect.has_table(db_connection.connect(), apple_sqlite_table)):
        return True
    print("INITIALIZING APPLE SCRAPER")
    APPID = APPID
    COUNTRY = COUNTRY
    pages = 20
    STOREURL = f"https://sensortower.com/api/ios/review/get_reviews?app_id={APPID}&page=1&limit=100&sort_by=date&sort_order=desc"
    res = requests.get(STOREURL)
    if res.status_code == 200:

        def extract_response(APPID, pages=20):
            for i in range(pages):
                URL = f"https://sensortower.com/api/ios/review/get_reviews?app_id={APPID}&page={i+1}&limit=100&sort_by=date&sort_order=desc"
                res = requests.get(URL)
                #dummy =[token_hex for i in range(pages*25)]
                if res.status_code == 200:
                    data = res.json()
                    feeds = data["feedback"]
                    if i == 0:
                        df = pd.DataFrame(columns=list(feeds[1].keys())[
                            0:8], index=range(len(feeds)))
                    try:
                        for j, feed in enumerate(feeds):
                            for column in df.columns:
                                df.loc[j, column] = feed[column]

                        df.set_index(['username', 'date'],
                                     inplace=True, drop=False)
                    except:
                        df.drop(['username', 'date'], axis=1, inplace=True)
                        apple_df = df
                        return apple_df

            df.drop(['username', 'date'], axis=1, inplace=True)

            apple_df = df
            return apple_df
        df = extract_response(APPID)
        #fake index
        #dummy = [token_hex(7) for i in range(len(df))]
        df.reset_index(inplace=True)
        #df.reindex(dummy)
        data = [df['username'], df['content'],
                df['version'], df['rating'], df['date'], df['country']]
        headers = ['reviewId', 'review', 'version', 'rating', 'at', 'country']
        df = pd.concat(data, axis=1, keys=headers)
        df.dropna(subset=['rating'], inplace=True)
        df.dropna(subset=['review'], inplace=True)
        df['version'].fillna("null", inplace=True)
        df["version"] = df["version"].astype(str)

        # fill the null value on the version
        for idx in range(len(df)-1):
            if df['version'][idx] == 'null':
                df.loc[idx, 'version'] = df['version'][idx+1]

        # drop version which lead to error (ex: '334280')
        for i in range(len(df)):
            try:
                if "." in df['version'][i][1]:
                    pass
                elif "." in df['version'][i][2]:
                    pass
                else:
                    df.drop(index=i, inplace=True)
            except:
                pass
        
        df.reset_index(drop=True, inplace=True)
        # set the 'at' column as datetime
        df['at'] = pd.to_datetime(df['at'])


        df.to_sql(apple_sqlite_table,
                    db_connection, if_exists='append')

        apple_remove_duplicate_query = text(f"""DELETE FROM '{apple_sqlite_table}'
          WHERE ROWID NOT IN (SELECT MIN(rowid)
          FROM '{apple_sqlite_table}' GROUP BY reviewId, review, version, rating,
          at, country
          )""")
        db_connection.execute(apple_remove_duplicate_query)

        job_id = "delete_apple_table_job"
        delta_timeunit = 1000
        run_date = datetime.now() + timedelta(hours=delta_timeunit)
        scheduler.add_job(func=delete_table_job, trigger="date",
                          run_date=run_date, args=[apple_sqlite_table])

        return True
    return False

def google_scrapper(PLAYSTORE_ID, COUNTRY, db_connection):
  
  google_sqlite_table = f'{str(PLAYSTORE_ID)}_{str(COUNTRY)}'
  
  # End the function early if the queried table exists 
  if(db_connection.dialect.has_table(db_connection.connect(), google_sqlite_table)):
    return True
  
  STOREURL = f'https://play.google.com/store/apps/details?id={PLAYSTORE_ID}&hl={COUNTRY}'
  res = requests.get(STOREURL)
  
  if res.status_code == 200:

    BATCH_SIZE = 50
    MAX_REVIEWS = 10000 
    appinfo = app(
        PLAYSTORE_ID,
        lang='en',
        country=COUNTRY)

    AVAIL_REVIEWS = appinfo.get('reviews')
    TOFETCH_REVIEWS = min(AVAIL_REVIEWS, MAX_REVIEWS)
    if TOFETCH_REVIEWS == 0:
      return False

    ints = list(range(max(1, TOFETCH_REVIEWS//BATCH_SIZE)))
    
    t = tqdm(total=TOFETCH_REVIEWS)
    for i in ints:
        if i == 0:
          result, continuation_token = reviews(PLAYSTORE_ID,
                                            count=BATCH_SIZE,
                                            country=COUNTRY
                                            )
        res, continuation_token = reviews(PLAYSTORE_ID, count=BATCH_SIZE, continuation_token=continuation_token)
        result.extend(res)
        t.update(BATCH_SIZE)
    t.close()

    dfp = pd.DataFrame(result)
    dfp.drop_duplicates('reviewId', inplace=True) #droppping the duplicates

    # creating the dataframe
    data = [dfp['reviewId'],dfp['content'],dfp['reviewCreatedVersion'],dfp['score'],dfp['at']]
    headers = ['reviewId','review','version','rating','at']
    df_google = pd.concat(data, axis=1, keys=headers)
    df_google['version'].fillna("null",inplace=True)
    
    # fill the null value on the version
    for idx in range(len(df_google)-1):
      if df_google['version'][idx] == 'null' :
        df_google.loc[idx,'version']= df_google['version'][idx+1]
    
    # drop version which lead to error (ex: '334280')
    for i in range(len(df_google)):
      if "." in df_google['version'][i][1]:
        pass
      elif "." in df_google['version'][i][2]:
        pass
      else:  
        df_google.drop(index=i,inplace=True)
    df_google.reset_index(drop=True, inplace=True)
    df_google['at'] = pd.to_datetime(df_google['at']) #set the 'at' column as datetime


    df_google.to_sql(google_sqlite_table,
                        db_connection, if_exists='append')

    google_remove_duplicate_query = text(f"""DELETE FROM '{google_sqlite_table}'
          WHERE ROWID NOT IN (SELECT MIN(rowid)
          FROM '{google_sqlite_table}' GROUP BY reviewId,
          review, rating,  version, at
          )""")
    db_connection.execute(google_remove_duplicate_query)

    job_id = "delete_google_table_job"
    delta_timeunit = 1000
    run_date = datetime.now() + timedelta(hours=delta_timeunit)
    scheduler.add_job(func=delete_table_job, trigger="date", run_date=run_date, args=[google_sqlite_table]) 

    # I dont use this because Pandas dataframe still returning sqlite instance as an object
    # Its better to change the data type after we load it into dataframe

    # df_google_ps.to_sql(google_sqlite_table, db_connection, if_exists='append',
    #                     dtype={'reviewId': sqlalchemy.VARCHAR(),
    #                            'userName':  sqlalchemy.types.VARCHAR(),
    #                            'content': sqlalchemy.types.VARCHAR(),
    #                            'score': sqlalchemy.types.Float(precision=1, asdecimal=True),
    #                            'reviewCreatedVersion': sqlalchemy.types.VARCHAR})

    return True

  return False
  # return filename, google_sqlite_table, PLAYSTORE_ID


def get_by_date(df, start_date, end_date):
  mask = (df['at'] > start_date) & (
      df['at'] <= end_date)
  df_date = df.loc[mask]
  df_date.fillna(0, inplace=True)
  return df_date


#get the plot
def google_plot(filename):
  pd.crosstab(filename['score'], filename['reviewCreatedVersion'], margins=True)
  ct = pd.crosstab(filename['reviewCreatedVersion'], filename['score'])
  plot = ct.tail(15).plot(kind="bar",figsize=(15, 3))
  plot.set_xlabel("version")
  plot.set_ylabel("rating")
  return plot

def apple_plot(filename):
  pd.crosstab(filename['im:rating'], filename['im:version'], margins=True)
  ct = pd.crosstab(filename['im:version'], filename['im:rating'])
  plot = ct.tail(15).plot(kind="bar",figsize=(15, 3))
  plot.set_xlabel("version")
  plot.set_ylabel("rating")
  return plot

#change apple datatype
def change_apple_dtype(df_apple):
  df_apple["author"] = df_apple["author"].astype(str)
  df_apple["im:version"] = df_apple["im:version"].astype(str)
  df_apple["im:rating"] = df_apple["im:rating"].astype(str).astype(int)
  df_apple["title"] = df_apple["title"].astype(str)
  df_apple["content"] = df_apple["content"].astype(str)
  return df_apple

#change google datatype
def change_google_dtype(df_google):
  df_google.dropna(subset=['version'], inplace=True)
  #df_google["userName"] = df_google.userName.astype(str)
  df_google["review"] = df_google.review.astype('string')
  df_google["rating"] = df_google["rating"].astype(str).astype(int)
  df_google["version"] = df_google["version"].astype(str)
  df_google.reset_index(inplace=True, drop=True)

  #check version format
  for i in range(len(df_google)):
    if "." in df_google['version'][i][1]:
      pass
    elif "." in df_google['version'][i][2]:
      pass
    else:
      df_google.drop(index=i, inplace=True)
  return df_google





# apple_file, APPID = apple_scrapper(1031922175)
# google_file, PLAYSTORE_ID = google_scrapper('echo.co.uk')

# #loading data, optional if scrapper isnt called by scraper
# df = pd.read_csv(f"dataset/labeled_negative_reviews_with_versions_ratings_type_{APPID}_{PLAYSTORE_ID}.csv")
# df_apple = pd.read_csv("dataset/"+ apple_file)
# df_google = pd.read_csv("dataset/"+ google_file)
# df_apple["content"] = df_apple["title"].astype(str)+" "+df_apple["content"].astype(str)



#for this better make a pd df of combined reviews first
def combined_review(df_apple, df_google):
  review_rating_dict={
    'user_name':list(df_apple['author'].values) + list(df_google['userName'].values),
    'review': list(df_apple['content'].values) + list(df_google['content'].values),
    'rating': list(df_apple['im:rating'].values) + list(df_google['score'].values)
  }
  combined_df= pd.DataFrame(review_rating_dict)
  combined_df.drop_duplicates(['review'],inplace=True)
  combined_df.reset_index(inplace=True,drop=True)
  combined_df["user_name"]= combined_df.user_name.astype(str)
  combined_df["review"] = combined_df.review.astype(str)
  combined_df["rating"] = combined_df["rating"].astype(str).astype(int)
  return combined_df

#combined_df = combined_review(df_apple, df_google)
#create dataframe that contain google username and date (join this table later with final df)
def google_review_date(df_google):
  df_google_date = df_google[["userName", "at"]]
  df_google_date["at"]=df_google_date["at"].astype('datetime64[ns]')
  df_google_date["userName"]=df_google_date.userName.astype('string') 
  df_google_date.rename(columns={'userName': 'user_name'}, inplace=True)
  return df_google_date

#df_google_date = google_review_date(df_google)

#pickling for later use


def pickle_main(combined_df, df_google_date, APPID, PLAYSTORE_ID):
  combined_df.to_pickle(f"dataset/combined_df_{APPID}_{PLAYSTORE_ID}.pkl")
  df_google_date.to_pickle(f"dataset/df_google_date_{PLAYSTORE_ID}.pkl")
#get similiar index
#similar_index=combined_df[combined_df['review'].isin(df['review'])].index

# get words length of each review (nadi)
#raw_lengths=combined_df['review'].apply(lambda x: len(x.split(" "))).values

#removing emoji
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text=regrex_pattern.sub(r'',text)
    text=text.replace('\n',' ')
    text=re.sub(' +', ' ', text)
    
    return text

# contractions dictionary


# expand contraction with this function
def expand_contractions(entry):
  contractions = { 
  "ain't": "am not / are not / is not / has not / have not",
  "aren't": "are not / am not",
  "can't": "cannot",
  "can't've": "cannot have",
  "cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he had / he would",
  "he'd've": "he would have",
  "he'll": "he shall / he will",
  "he'll've": "he shall have / he will have",
  "he's": "he has / he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how has / how is / how does",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i shall have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had / it would",
  "it'd've": "it would have",
  "it'll": "it shall / it will",
  "it'll've": "it shall have / it will have",
  "it's": "it has / it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she had / she would",
  "she'd've": "she would have",
  "she'll": "she shall / she will",
  "she'll've": "she shall have / she will have",
  "she's": "she has / she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "that'd": "that would / that had",
  "that'd've": "that would have",
  "that's": "that has / that is",
  "there'd": "there had / there would",
  "there'd've": "there would have",
  "there's": "there has / there is",
  "they'd": "they had / they would",
  "they'd've": "they would have",
  "they'll": "they shall / they will",
  "they'll've": "they shall have / they will have",
  "they're": "they are",
  "they've": "they have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what shall / what will",
  "what'll've": "what shall have / what will have",
  "what're": "what are",
  "what's": "what has / what is",
  "what've": "what have",
  "when's": "when has / when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where has / where is",
  "where've": "where have",
  "who'll": "who shall / who will",
  "who'll've": "who shall have / who will have"
  }
  entry = entry.lower()
  entry=re.sub(r"’","'",entry)
  entry=entry.split(" ")

  for idx,word in enumerate(entry):
      if word in contractions:
          
          entry[idx]=contractions[word]
  return " ".join(entry)

# remove punctuation with this function
def remove_punctuation(entry):
  entry=re.sub(r"[^\w\s]"," ",entry)
  return entry

#get stopwords
# nltk.download('stopwords')



from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
 
stop_words = stopwords.words('english')

# remove stopwords with this function
def remove_stop_words(sentence):
  dummy=sentence.split(" ")
  out=[i for i in dummy if i not in stop_words]
  return " ".join(out)
 
    
# remove short words with this function
def remove_short_words(sentence):
  dummy=sentence.split(" ")
  out=[i for i in dummy if len(i)>3 ]
  return " ".join(out)

# spacy method for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# lemmatizer function
def lemmatize(reviews):
#download spacy en module first (python download -m spacy en)
  doc = nlp(reviews)
  return " ".join([token.lemma_ for token in doc])


# combine all functions above to one "preprocess_data" function below
def preprocess_data(df):
  df=df.copy()
  print("function0 ok")
  df['review'] = df['review'].apply(lambda x: x.lower())  # lowering case
  print("function1 ok")
  df['review']=df['review'].apply(deEmojify)   
  print("function2 ok")        #removing emoji
  df['review']=df['review'].apply(expand_contractions) 
  print("function3 ok")  
  df['review']=df['review'].apply(remove_stop_words) 
  print("function4 ok")  
  df['review']=df['review'].apply(remove_punctuation) 
  print("function5 ok") 
  df['review']=df['review'].apply(lambda s: re.sub(r"[^a-zA-Z]"," ",s)) #keep only alphabetical words
  print("function6 ok") 
  df['review']=df['review'].apply(lambda s:re.sub(' +', ' ', s))   #remove + sign
  print("function7 ok") 
  df['review']=df['review'].apply(lemmatize) #apply lemmatizer
  print("function8 ok") 
  df['review']=df['review'].apply(remove_short_words)
  print("function9 ok") 
  df.drop(df[df['review'].apply(lambda x: len(x)==0)].index,inplace=True)
  print("function10 ok") 
  #df.reset_index(inplace=True,drop=True) 
  return df

# combined_df_preprocessed=preprocess_data(combined_df)

#NGram analysis

#keep the index this way, will use the indexes later
#spliting data to positive and negative review and get the lemma
def pos_neg_lemma(combined_df_preprocessed):
  pos=combined_df_preprocessed[combined_df_preprocessed['rating'].isin([4,5])] #positive reviews
  neg=combined_df_preprocessed[combined_df_preprocessed['rating'].isin([1,2,3])] #negative reviews
  #tokenizing the separated reviews
  pos_lemma=[val.split(" ") for val in pos['review'].values]
  neg_lemma=[val.split(" ") for val in neg['review'].values]
  
  return pos, neg, pos_lemma, neg_lemma

#pos, neg, pos_lemma, neg_lemma = pos_neg_lemma(combined_df_preprocessed)

#construct negative bigram with minimal frequency = 3, and minimum score(threshold)= 6
#construct positive bigram with minimal frequency = 3, and minimum score(threshold)= 10
def bigram(neg_lemma, pos_lemma):
  neg_bigrams=gensim.models.Phrases(neg_lemma,min_count=3,threshold=6)
  neg_bigram_reviews=neg_bigrams[neg_lemma] 
  pos_bigrams=gensim.models.Phrases(pos_lemma,min_count=3,threshold=10)
  pos_bigram_reviews=pos_bigrams[pos_lemma] 
  return pos_bigrams, pos_bigram_reviews, neg_bigrams, neg_bigram_reviews

#pos_bigrams, pos_bigram_reviews, neg_bigrams, neg_bigram_reviews = bigram(neg_lemma, pos_lemma)



#construct positive trigram with minimal frequency = 2, and minimum score(threshold)= 15
def trigram(pos_bigram_reviews, neg_bigram_reviews):
  pos_trigrams=gensim.models.Phrases(pos_bigram_reviews,min_count=2,threshold=15)
  pos_trigram_reviews=pos_trigrams[pos_bigram_reviews]
  neg_trigrams=gensim.models.Phrases(neg_bigram_reviews,min_count=2,threshold=8)
  neg_trigram_reviews=neg_trigrams[neg_bigram_reviews]
  return pos_trigrams, pos_trigram_reviews, neg_trigrams, neg_trigram_reviews


#pos_trigrams, pos_trigram_reviews, neg_trigrams, neg_trigram_reviews = trigram(pos_bigram_reviews, neg_bigram_reviews)

# replace initial positive review with "grammed" positive review
# replace initial negative review with "grammed" negative review
def replace_gram(combined_df_preprocessed, neg_trigram_reviews, pos_trigram_reviews, neg_index, pos_index):
  dummy=[" ".join(tri) for tri in neg_trigram_reviews]
  pointer=0
  for idx in neg_index:
    combined_df_preprocessed['review'][idx]=dummy[pointer]
    pointer+=1

  dummy2=[" ".join(tri) for tri in pos_trigram_reviews]
  pointer=0
  for idx2 in pos_index:
    combined_df_preprocessed['review'][idx2]=dummy2[pointer]
    pointer+=1
  return  combined_df_preprocessed

# combined_df_preprocessed = replace_gram(combined_df_preprocessed, neg_trigram_reviews, pos_trigram_reviews, neg.index, pos.index)

# map rating into positive (1) and negative (0) values for later logistic regression analysis


def mapping(combined_df_preprocessed, APPID, PLAYSTORE_ID):
  mapping={4:1,
          5:1,
          3:0,
          2:0,
          1:0
          }      
  combined_df_preprocessed['rating']=combined_df_preprocessed['rating'].map(lambda x: mapping[x])

  #put to pickle for later use
  combined_df_preprocessed.to_pickle(f"dataset/data_clean_and_mapped_{APPID}_{PLAYSTORE_ID}.pkl")
  return combined_df_preprocessed

#combined_df_preprocessed = mapping(combined_df_preprocessed)

# split dataset into train and test set
def train_test(combined_df_preprocessed):

  df_train, df_test=train_test_split(combined_df_preprocessed,test_size=0.05,
                                    stratify=combined_df_preprocessed.rating.values,random_state=2020)

  # assigning each train and test axis
  x_train=df_train['review'].values
  x_test=df_test['review'].values
  y_train=df_train['rating'].values
  y_test=df_test['rating'].values

  # transforming every train and test reviews based on its frequency
  return x_train, x_test, y_train, y_test

#x_train, x_test, y_train, y_test = train_test(combined_df_preprocessed)

def vectorizer_tfidf(x_train, x_test ):
  cv=CountVectorizer()
  x_train_count=cv.fit_transform(x_train)
  x_test_count=cv.transform(x_test)
  feature_names = cv.get_feature_names()
  # Vectorize words in each review
  tfidf=TfidfTransformer()
  x_train_tfidf=tfidf.fit_transform(x_train_count)
  x_test_tfidf=tfidf.transform(x_test_count)
  
  return cv, x_train_count, x_test_count, feature_names, tfidf, x_train_tfidf, x_test_tfidf

#cv, x_train_count, x_test_count, feature_names, tfidf, x_train_tfidf, x_test_tfidf = vectorizer_tfidf(x_train, x_test)

# put it to pickle for later use in webapp
def pickle_save(APPID, PLAYSTORE_ID, y_train, y_test, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, feature_names):
  pickle.dump(y_train, open(f"dataset/y_train_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))
  pickle.dump(y_test, open(f"dataset/y_test_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))

  pickle.dump(x_train_count, open(f"dataset/x_train_count_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))
  pickle.dump(x_test_count, open(f"dataset/x_test_count_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))

  pickle.dump(x_train_tfidf, open(f"dataset/x_train_tfidf_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))
  pickle.dump(x_test_tfidf, open(f"dataset/x_test_tfidf_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))
  pickle.dump(feature_names, open(f"dataset/feature_names_{APPID}_{PLAYSTORE_ID}.pkl", "wb"))

#pickle_save(APPID, PLAYSTORE_ID, y_train, y_test, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, feature_names)

# Logistic regression
#This applies a heuristic class balancing, because we have imbalance towards the number of 1s
def logistic_regresion_binary(x_train_tfidf, y_train, x_test_tfidf, input_solver='liblinear' ):
  log_r=LogisticRegression(random_state=2020,class_weight='balanced',solver=input_solver) 
  log_r.fit(x_train_tfidf,y_train)
  log_r_pred=log_r.predict(x_test_tfidf)
  return log_r, log_r_pred


#Function for finding outliers for length
def find_outliers(array):
  lower_outlier=int(np.quantile(array,0.25)-stats.iqr(array)*1.5)
  upper_outlier=int(stats.iqr(array)*1.5+np.quantile(array,0.75))
  return lower_outlier, upper_outlier


#ref should be the reference/dictionary used so pass in result_dict
def lookup(df,ref):

  val=df['review'].values
  score_list=[]
  for i in val:
    score=0
    for j in i.split(" "):
      if j in ref:
        score+=ref[j]
      else:
        score+=0
  
    score_list.append(score)
  return score_list 



#Here should include checking if lower outlier <0, if yes then ignore this and do not do scaling
def outlier_scaling(df, upper, lower):
  #Function assumes that lengths have been added to sentences
  longg=df[df['lengths']>upper].index
  df['score'].loc[longg]=df['score'].loc[longg]/abs(df['z_score'].loc[longg])
  if lower>0:
    #so if outlier is lets say 1.6, rounded to 1, we find and scale for those sentences with lengths <2
    small=df[df['lengths']<lower+1].index
    df['score'].loc[small]=df['score'].loc[small]*abs(df['z_score'].loc[small])
  return df


# merge the review and score with labeled negative topic
def get_final_df(combined_df, df, similar_index, APPID, PLAYSTORE_ID):
  final_df=pd.merge(combined_df.loc[similar_index][['user_name','review','score']],df,how='inner',on='review')
  final_df.fillna(0,inplace=True)
  final_df.to_pickle(f"dataset/final_df_{APPID}_{PLAYSTORE_ID}.pkl")
  return final_df

#final_df = get_final_df(combined_df, df, similar_index)


def google_final(final_df, df_google_date, PLAYSTORE_ID):
  google_neg_rev = pd.merge(final_df,df_google_date, on='user_name')
  google_neg_rev.drop_duplicates(['review'],inplace=True)
  google_neg_rev.to_pickle(f"dataset/google_neg_rev_{PLAYSTORE_ID}.pkl")
  return google_neg_rev

#google_neg_rev = google_final(final_df, df_google_date)



def final_by_date(google_neg_rev, start_date, end_date ):
  mask = (google_neg_rev['at'] > start_date) & (google_neg_rev['at'] <= end_date )
  google_neg_rev_recent = google_neg_rev.loc[mask]
  google_neg_rev_recent.fillna(0,inplace=True)
  return google_neg_rev_recent

#get summed scores
def get_summed_scores(df,col):
  """ Inputs: 
  df--dataframe to get sum of scores from. (should be scaled)
  col--list of column names

  Outputs:
  Dictionary containing the mappings of the column names, and the severance score values """
  sum_array=[]
  sums_only=[]
  for c in col:
    sum=abs(np.sum(df[df[c]==1]['score'].values))
    sum_array.append((c,sum))
    sums_only.append(sum)
  raw_sum_dict=dict(tuple(sum_array))
  normalizer=np.sum(np.array(sums_only))
  normalized_sum_dict={i[0]:i[1]/normalizer for i in raw_sum_dict.items()}


  return raw_sum_dict,normalized_sum_dict

 

