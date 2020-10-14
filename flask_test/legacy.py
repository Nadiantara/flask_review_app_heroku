# import requests
# import json
# import pandas as pd
# from secrets import token_hex

# Echo apple scrapper only being run in app.py
# def apple_scrapper(APPID, COUNTRY, db_connection):


#   STOREURL = f'http://apps.apple.com/{COUNTRY}/app/id{APPID}'
#   res = requests.get(STOREURL)
#   if res.status_code == 200:
#       try:
#         appname = re.search('(?<="name":").*?(?=")', res.text).group(0)
#         print(appname)
#       except:
#         appname = None

#   #extracting from appstore
#   def extract_itunes(app_id, country="US", pages=1, save=True):
#       for i in range(pages):
#           URL = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/page={i+1}/sortBy=mostRecent/json"
#           res = requests.get(URL)
#           if res.status_code == 200:
#               entries = res.json()['feed']['entry']
#               if i == 0:
#                   df = pd.DataFrame(columns=list(entries[1].keys())[0:6], index=range(len(entries)))

#               for j, entry in enumerate(entries):
#                   for column in df.columns:
#                       try:
#                           df.loc[j, column] = entry[column]['name']['label']
#                       except:
#                           df.loc[j, column] = entry[column]['label']

#               df.set_index('id', inplace=True, drop=False)

#       df.drop('id', axis=1, inplace=True)

#       if save:
#           datenow = datetime.today().strftime('%d/%m/%Y').replace('/', '')
#           filename = f'{app_id}_{country}_{datenow}_AAS.csv'
#           df.to_csv(f"dataset/{filename}")

#       df.index = df.index.astype('int64')
#       apple_df = df
#       return apple_df

#   #Have to make the date suitable first. Make sure to adjust according to what you have,
#   #E.g if today's date is 23/8/2020, and your old date is 18/8/2020, then make sure timedelta =5

#   apple_df = extract_itunes(APPID, COUNTRY, pages=10)
#   old_date=datetime.today()
#   old_date=old_date.strftime('%d/%m/%Y').replace('/', '')
#   apple_df.to_csv(f"dataset/{APPID}_{COUNTRY}_overall_{old_date}_AAS.csv" )
#   filename=f'{APPID}_{COUNTRY}_overall_{old_date}_AAS.csv'
#   apple_sqlite_table = f'{APPID}_{COUNTRY}'
#   apple_df.to_sql(apple_sqlite_table, db_connection, if_exists='append')


#   # I dont use this because Pandas dataframe still returning some of sqlite instances as an object
#   # Its better to change the data type after we load it into dataframe
#   #If we want to change datatype
#   # apple_df.to_sql(apple_sqlite_table, db_connection, if_exists='append',
#   #                 dtype={'id': sqlalchemy.INTEGER(),
#   #                        'author':  sqlalchemy.types.VARCHAR(length=100),
#   #                       'im:version': sqlalchemy.types.VARCHAR(),
#   #                       'im:rating': sqlalchemy.types.Float(precision=1, asdecimal=True),
#   #                        'title': sqlalchemy.VARCHAR(),
#   #                        'content': sqlalchemy.VARCHAR()})

#   # apple_df.to_sql(apple_sqlite_table, db_connection, if_exists='append',
#   #                 dtype={'datefld': sqlalchemy.DateTime(),
#   #                        'intfld':  sqlalchemy.types.INTEGER(),
#   #                        'strfld': sqlalchemy.types.VARCHAR(length=1000),
#   #                        'floatfld': sqlalchemy.types.Float(precision=3, asdecimal=True),
#   #                        'booleanfld': sqlalchemy.types.Boolean})

#   return filename, apple_sqlite_table, APPID







#print(json.dumps(data, indent=2))


# def apple_scrapper2(APPID, COUNTRY, db_connection):
#   driver = webdriver.Chrome('C:\Program Files (x86)\chromedriver')
#   apple_sqlite_table = f'{APPID}_{COUNTRY}'
#   # End the function early if the queried table exists
#   if(db_connection.dialect.has_table(db_connection.connect(), apple_sqlite_table)):
#       return True
#   APPID = APPID
#   COUNTRY = COUNTRY
#   MAX_ROWS = 10000

#   STOREURL = f'http://apps.apple.com/{COUNTRY}/app/id{APPID}'
#   res = requests.get(STOREURL)
#   if res.status_code == 200:

#       driver.get(
#           f'https://sensortower.com/ios/{COUNTRY}/publisher/app/appName/{APPID}/review-history?selected_tab=reviews')

#       def get_page():
#           doc = BeautifulSoup(driver.page_source)
#           rows = doc.select("tbody tr")

#           datapoints = []
#           for row in rows:
#               cells = row.select("td")
#               data = {
#                   'reviewId': str(row) + str(randint(100, 10000)),
#                   'country': cells[0].text.strip(),
#                   'at': cells[1].text.strip(),
#                   'rating': cells[2].select_one('.gold')['style'],
#                   'review': cells[3].select_one('.break-wrap-review').text.strip(),
#                   'version': cells[4].text.strip()
#               }
#               datapoints.append(data)
#           return datapoints

#       all_data = []
#       wait = WebDriverWait(driver, 5, poll_frequency=0.05)
#       while len(all_data) < MAX_ROWS:
#           #wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, '.ajax-loading-cover')))

#           results = get_page()
#           all_data.extend(results)

#           next_button = driver.find_elements_by_css_selector(
#               ".universal-flat-button-group .universal-flat-button+.universal-flat-button ")[1]
#           if next_button.get_attribute('disabled'):
#               break
#           next_button.click()
#           time.sleep(0.5)
#           # Doesn't trigger fast enough!
#           # wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.ajax-loading-cover')))

#       df = pd.DataFrame(all_data)
#       df.rating = df.rating.replace({
#           'width: 99%;': 5,
#           'width: 79%;': 4,
#           'width: 59%;': 3,
#           'width: 39%;': 2,
#           'width: 19%;': 1
#       })
#       df['version'].fillna("null", inplace=True)
#       df["version"] = df["version"].astype(str)

#       # fill the null value on the version
#       for idx in range(len(df)-1):
#         if df['version'][idx] == 'null':
#             df.loc[idx, 'version'] = df['version'][idx+1]

#       # drop version which lead to error (ex: '334280')
#       for i in range(len(df)):
#         try:
#           if "." in df['version'][i][1]:
#             pass
#           elif "." in df['version'][i][2]:
#             pass
#           else:
#             df.drop(index=i, inplace=True)
#         except:
#           pass
#       df.reset_index(drop=True, inplace=True)
#       # set the 'at' column as datetime
#       df['at'] = pd.to_datetime(df['at'])

#       df.to_sql(apple_sqlite_table,
#                 db_connection, if_exists='append')

#       apple_remove_duplicate_query = text(f"""DELETE FROM '{apple_sqlite_table}'
#           WHERE ROWID NOT IN (SELECT MIN(rowid)
#           FROM '{apple_sqlite_table}' GROUP BY reviewId, country,
#           at, rating, review, version
#           )""")
#       db_connection.execute(apple_remove_duplicate_query)

#       job_id = "delete_apple_table_job"
#       delta_timeunit = 1000
#       run_date = datetime.now() + timedelta(hours=delta_timeunit)
#       scheduler.add_job(func=delete_table_job, trigger="date",
#                         run_date=run_date, args=[apple_sqlite_table])

#       return True
#   return False
