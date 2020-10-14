import numpy as np
from datetime import datetime,timedelta, date
# check your pickle compability, perhaps its pickle not pickle5
import pickle
import pandas as pd
import re
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
from flask_test.webapp_functions.constants import nlp, contractions, stop_words
import gensim
import statistics
import scipy.stats as st




## TEXT PROCESSING BASIC UTILS
    
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

# expand contraction with this function
def expand_contractions(entry):
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

#might need to extend stop words to include the name of the app --> so 5miles (miles), carousell, etc
# remove stopwords with this function
def remove_stop_words(sentence):
    dummy=sentence.split(" ")
    out=[i for i in dummy if (i not in stop_words) and (len(i)>2)]
    return " ".join(out)

# lemmatizer function
#trying topic modelling with 30 topics taking all tags
def lemmatize(reviews):
    tags=['NOUN','ADJ']
    doc = nlp(reviews)
    output=[token.lemma_ for token in doc if token.pos_ in tags]
    return " ".join(output)

# remove short words
def remove_short_words(sentence):
  dummy=sentence.split(" ")
  out=[i for i in dummy if len(i)>3 ]
  return " ".join(out)

# combine most of the functions above to one "preprocess_data" function below
def get_preprocess_data(df):
    df=df.copy()
    df['review']=df['review'].astype(str)
    df['review'] = df['review'].apply(lambda x: x.lower())  # lowering case
    df['review']=df['review'].apply(deEmojify)           #removing emoji
    df['review']=df['review'].apply(expand_contractions) 
    df['review']=df['review'].apply(remove_punctuation)   
    df['review']=df['review'].apply(lambda s: re.sub(r"[^a-zA-Z]"," ",s)) #keep only alphabetical words
    df['review']=df['review'].apply(lemmatize)
    df['review']=df['review'].apply(remove_stop_words) #also removes short words
    df['review']=df['review'].apply(lambda s:re.sub(' +', ' ', s))   #remove + sign
    df['review']=df['review'].apply(remove_short_words)
    df.drop(df[df['review'].apply(lambda x: len(x)==0)].index,inplace=True)
    return df

def get_ngrams_init(df):
  pos=df[df['rating'].isin([4,5])] #keep the index this way, will use the indexes later
  neg=df[df['rating'].isin([1,2,3])] 
  pos_lemma=[val.split(" ") for val in pos['review'].values]
  neg_lemma=[val.split(" ") for val in neg['review'].values]
  return pos.index,neg.index,pos_lemma,neg_lemma

def get_bigrams(pos_lemma,neg_lemma):
  #Can make this customizable in the package..
  pos_bigrams=gensim.models.Phrases(pos_lemma,min_count=3,threshold=10)
  pos_bigram_reviews=pos_bigrams[pos_lemma] 
  neg_bigrams=gensim.models.Phrases(neg_lemma,min_count=3,threshold=6)
  neg_bigram_reviews=neg_bigrams[neg_lemma] 
  return pos_bigram_reviews,neg_bigram_reviews

def get_trigrams(pos_bigram_reviews,neg_bigram_reviews):
  #Can make this customizable later on in the package..
  pos_trigrams=gensim.models.Phrases(pos_bigram_reviews,min_count=2,threshold=15)
  pos_trigram_reviews=pos_trigrams[pos_bigram_reviews]
  neg_trigrams=gensim.models.Phrases(neg_bigram_reviews,min_count=2,threshold=8)
  neg_trigram_reviews=neg_trigrams[neg_bigram_reviews]
  return pos_trigram_reviews,neg_trigram_reviews

def get_put_n_grams_in_df(df,index,reviews):
  dummy=[" ".join(tri) for tri in reviews]
  pointer=0
  for idx in index:
    df['review'][idx]=dummy[pointer]
    pointer+=1
  return df

def get_n_grams(df):
  #Inputs:
  #df--combined_df_preprocessed 
  #Outputs:
  #A dataframe with the sentences transformed to contain bigrams, trigrams
  pos_index,neg_index,pos_lemma,neg_lemma= get_ngrams_init(df)
  pos_bigram_reviews,neg_bigram_reviews=get_bigrams(pos_lemma,neg_lemma)
  pos_trigram_reviews,neg_trigram_reviews=get_trigrams(pos_bigram_reviews,neg_bigram_reviews)
  df= get_put_n_grams_in_df(df,pos_index,pos_trigram_reviews)
  df= get_put_n_grams_in_df(df,neg_index,neg_trigram_reviews)
  return df

## NEGATIVE DATAFRAME AND TOPIC

def get_negreview_topic(dataframe):
    '''
    This function is used create a negative reviews data and list of negative reviews topic

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)

    returns:
    neg_df-- negative review dataframe (columns= 'review',[topic],'rating','version','at')
    topic-- list of negative review's topic
    '''
    
    # topic modelling
    MODEL_BASE_DIR = "flask_test/webapp_functions/models/"
    vec=pickle.load(open(f'{MODEL_BASE_DIR}dummy_cv.pickel',mode='rb')) #vec is the count vectorizer associated with the LDA model
    LDA=pickle.load(open(f'{MODEL_BASE_DIR}dummy_lda.pickel',mode='rb')) #LDA is the pretrained lda model
    
    neg_df_1=dataframe[dataframe['rating']<4] #make sure to take the negative set here
    negative_preprocessed = get_preprocess_data(neg_df_1)
    negative_preprocessed_value=negative_preprocessed['review'].values
    topic_names=['network_issues','topic_2','interface/listing','scammers/transaction_experience','topic_4','updates/message/notif_issues'] #transaction related--> everything between buyer seller
    dtm=vec.transform(negative_preprocessed_value).toarray()
    topic_distributions=LDA.transform(dtm)
    #we get topic distributions by transforming the document term matrix, so the distribution is according to negative_preprocessed (the data we made the dtm on)
    #it has a shape of (n_documnet, num_topics)
    dummy_1={}
    dummy_2={} #dummy_2 to contain the all 0s at first
    for idx,top in enumerate(topic_names):
      dummy_1[top]=topic_distributions[:,idx] #extract topic column
      dummy_2[top]=[0]*len(topic_distributions[:,idx])
    dummy_2=pd.DataFrame(dummy_2)
    raw_distributions=pd.DataFrame(dummy_1)
    for idx,row in raw_distributions.iterrows():
      top_two=np.argsort(row.values)[-2:]
      dummy_2.iloc[idx,top_two]=1
    vectorized_df=dummy_2

    #Here create dataframe with column labels associated with the topic distribution
    topic_distributed_df_raw= pd.concat([negative_preprocessed[['reviewId']].reset_index(drop=True),raw_distributions],axis=1)

    topic_distributed_df= pd.concat([negative_preprocessed[['reviewId']].reset_index(drop=True),vectorized_df],axis=1)
    #Here I concatenate with the negative_preprocessed's reviewId, because the negative_preprocessed variable (which was used to make dtm), was extracted from here
    neg_df_raw = pd.merge(dataframe,topic_distributed_df_raw,on='reviewId')
    neg_df = pd.merge(dataframe,topic_distributed_df,on='reviewId')

    #merge into original dataframe
    neg_df=neg_df[['review']+topic_names+['rating','version','at']]

    neg_df_raw=neg_df_raw[['review']+topic_names+['rating','version','at']]
  
    #create a list of topics from neg_df 
    topic=[]
    for i in range(len(neg_df.columns)):
      if i > neg_df.columns.get_loc("review") and i < neg_df.columns.get_loc("rating"):
        topic.append(neg_df.columns[i])

    return neg_df,topic

## SEVERITY, URGENCY, AND IMPORTANCE SCORE

def get_mapper(df):
  mapping={4:1,
         5:1,
         3:0,
         2:0,
         1:0,
         0:0}
  df['rating']=df['rating'].map(lambda x: mapping[x])
  return df

def get_init_params(df):
  #Inputs:
  #df--combined_df_preprocessed

  df_train,df_test=train_test_split(df,test_size=0.05,stratify=df.rating.values,random_state=2020)
  x_train=df_train['review'].values
  x_test=df_test['review'].values
  #train_index=df_train['review'].index
  #test_index=df_train['review'].index
  y_train=df_train['rating'].values
  y_test=df_test['rating'].values
  return x_train,y_train,x_test,y_test

def get_vectorizer(x_train,x_test):
  cv=CountVectorizer()
  x_train_count=cv.fit_transform(x_train)
  x_test_count=cv.transform(x_test)
  vocab_size=x_train_count.shape[1]
  tfidf=TfidfTransformer()
  x_train=tfidf.fit_transform(x_train_count)
  x_test=tfidf.transform(x_test_count)
  features=cv.get_feature_names()
  return x_train,x_test,vocab_size,features

def get_train_model(x_train,y_train):
  #In this case, it will be a logistic regression model, might need to test out if it shud be trained over entire set? Or hold some of the set back?
  #In the exploratory notebook we held back a bit of the train set, do we need to do that here? --> Should be tested out first 
  log_r=LogisticRegression(random_state=2020,class_weight='balanced',solver='liblinear') #This applies a heuristic class balancing, because we have imbalance towards the number of 1s
  log_r.fit(x_train,y_train)
  coeffs=log_r.coef_
  return coeffs

def get_coeff_df(vocab_size,features,coeffs):
  result_dict=dict(tuple([(features[i],coeffs[0][i]) for i in range(vocab_size)]))
  result_df=pd.DataFrame(sorted(result_dict.items(),key=lambda x:x[1]),columns=['word','coeff'])
  return result_dict,result_df


def get_run_model(df):
  #Inputs:
  #df--combined_df_preprocessed
  df= get_mapper(df)
  x_train,y_train,x_test,y_test= get_init_params(df)
  x_train,x_test,vocab_size,features= get_vectorizer(x_train,x_test)
  
  #running the model
  coeffs= get_train_model(x_train,y_train)
  #pred=log_r.predict(x_test)

  coeff_dict,coeff_df=get_coeff_df(vocab_size,features,coeffs)
  return coeff_dict,coeff_df

def get_lookup(df,ref):
  #df should be the original dataframe
  #ref should be the reference/dictionary used so pass in result_dict
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

def get_scoring(df,result_dict):
  #Inputs:
  #df-- combined_df_preprocessed
  score= get_lookup(df,result_dict)
  log_r_scored_df=df.copy()
  log_r_scored_df['score']=score
  return log_r_scored_df

def get_add_attributes(df):
  #This function is to add lengths and add the zscores
  #Input:
  #df--scored_unscaled_df

  lengths=df['review'].apply(lambda x: len(x.split(" "))).values
  #z-statistic will be calculated from lengths

  df['z_score']=zscore(lengths)
  df['lengths']=lengths
  return df

def get_find_outliers(array):
  lower_outlier=int(np.quantile(array,0.25)-st.iqr(array)*1.5)
  upper_outlier=int(st.iqr(array)*1.5+np.quantile(array,0.75))
  return lower_outlier, upper_outlier

def get_outlier_scaling(df, upper, lower):
  #Function assumes that lengths have been added to sentences
  longg=df[df['lengths']>upper].index
  df['score'].loc[longg]=df['score'].loc[longg]/abs(df['z_score'].loc[longg])
  if lower>0:
    small=df[df['lengths']<lower+1].index #so if outlier is lets say 1.6, rounded to 1, we find and scale for those sentences with lengths <2
    df['score'].loc[small]=df['score'].loc[small]*abs(df['z_score'].loc[small])
  return df

def get_match_merge_df(combined_df,scored_scaled_df,similar_index,neg_dataframe):
  #This function is to match. Combined_df contains the reviews which are same with the negative labeled set.
  #So we assign the scaled scores with their respective index, to the combined_df.
  #Then we merge the combined_df (now containing scores) with the labeled set, according "on" the review

  combined_df['score']=scored_scaled_df['score'] #match 
  final_df=pd.merge(combined_df.loc[similar_index][['review','score']],neg_dataframe,how='inner',on=['review']) #merge
  final_df.fillna(0,inplace=True) #deal with nulls
  return final_df

# severity score function as "get_severity_scores(dataframe,neg_dataframe,topic)"
def get_severity_scores(dataframe,neg_dataframe,topic):
  review_rating_dict={
  'review': list(dataframe['review'].values),
  'rating': list(dataframe['rating'].values)
  }
  combined_df= pd.DataFrame(review_rating_dict)
  combined_df.drop_duplicates(['review'],inplace=True)
  combined_df.reset_index(inplace=True,drop=True)
  similar_index=combined_df[combined_df['review'].isin(neg_dataframe['review'])].index

  combined_df_preprocessed = get_preprocess_data(combined_df)
  combined_df_preprocessed=get_n_grams(combined_df_preprocessed)
  coeff_dict,coeff_df= get_run_model(combined_df_preprocessed)
  scored_unscaled_df= get_scoring(combined_df_preprocessed,coeff_dict)
  scored_unscaled_df= get_add_attributes(scored_unscaled_df)
  lower_outlier,upper_outlier= get_find_outliers(scored_unscaled_df.lengths.values)
  scored_scaled_df= get_outlier_scaling(scored_unscaled_df,upper_outlier,lower_outlier)
  final_df= get_match_merge_df(combined_df,scored_scaled_df,similar_index,neg_dataframe)

  sum_array=[]
  sums_only=[]
  for c in topic:
    sum=abs(np.sum(final_df[final_df[c]==1]['score'].values))
    sum_array.append((c,sum))
    sums_only.append(sum)

  raw_sum_dict=dict(tuple(sum_array))
  normalizer=np.sum(np.array(sums_only))
  normalized_sum_dict={i[0]:i[1]/normalizer for i in raw_sum_dict.items()}

  raw_sum_dict={k: v for k, v in sorted(raw_sum_dict.items(), key=lambda item: item[1])} #sort this 
  normalized_sum_dict={k: v for k, v in sorted(normalized_sum_dict.items(), key=lambda item: item[1])}
  severity_score = pd.DataFrame([normalized_sum_dict])
  return severity_score

def get_importancescore (dataframe, neg_dataframe, topic, x=6):
    '''
    This function is used to get importance score

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    neg_dataframe-- neg_df from get_negreview_topic
    topic-- topic from get_negreview_topic
    x-- number of months to inspect

    returns:
    importance_score-- importance score
    '''
    severity_score = get_severity_scores(dataframe,neg_dataframe,topic)
    df = neg_dataframe
    topic = topic    
    df_cubi = dataframe
    df_cubi.index = df_cubi['at']
    df.index = df['at']
    df = df[topic].resample('MS').sum().join(pd.DataFrame(df_cubi['review'].resample('MS').count()))
    for kolom in topic:
      df[kolom] = df[kolom].astype(float)
    df.reset_index(inplace=True)
    for i in range(len(df)):
        for kolom in topic:
          if df[kolom][i] > 0:
            df.at[i,kolom] = round((df[kolom][i] / df['review'][i] * 100),2)

    importance_score = pd.DataFrame()
    importance_score['Topic']= topic
    score=[]
    for i in range(len(topic)):
      x1 = statistics.mean(df[topic[i]][-x:]) / 100 * sum(df['review'][-x:]) * severity_score[topic[i]][0]
      score.insert(i,x1)
    importance_score['importance_score']=score
    return importance_score

def get_urgencyscore(dataframe, neg_dataframe, topic, x=6):
    '''
    This function is used to get urgency score

    Parameters:
    dataframe-- data from get_crawl_data (but we can put data from apple scrapper if the columns names are identical)
    neg_dataframe-- neg_df from get_negreview_topic
    topic-- topic from get_negreview_topic
    x-- number of months to inspect

    returns:
    urgency_score-- urgency score
    '''
  
    df = neg_dataframe
    topic = topic    
    df_cubi = dataframe
    df_cubi.index = df_cubi['at']
    df.index = df['at']
    df = df[topic].resample('MS').sum().join(pd.DataFrame(df_cubi['review'].resample('MS').count()))
    for kolom in topic:
      df[kolom] = df[kolom].astype(float)
    df.reset_index(inplace=True)
    for i in range(len(df)):
        for kolom in topic:
          if df[kolom][i] > 0:
            df.at[i,kolom] = round((df[kolom][i] / df['review'][i] * 100),2)
  
    urgency_score = pd.DataFrame()
    urgency_score['Topic']= topic
    score = []
    for i in range(len(topic)):
      y = -(statistics.mean(df[topic[i]][-x:]) - df[topic[i]][-1:]).sum() / np.std(df[topic[i]])
      score.insert(i, y)  
    urgency_score['urgency_score']=score
    return urgency_score

def get_priority_score_scaled(priority_score):
  importance_score_scaled = []
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
  return priority_score
### VERY OLD FUNCTIONS

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


