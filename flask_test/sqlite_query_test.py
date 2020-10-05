

######################################################################################
#    Use this program to perform manual checking with SQLite querry to 'review.db'   #
#                   dont forget to change directory to 'flask_test'                  #
######################################################################################
import pandas as pd
import sqlite3

#you can use either sqlite3 connector or sqlalchemy to our sqlite database
# I prefer sqlite for this test and sqlalchemy when in the model
conn = sqlite3.connect('review.db')

c = conn.cursor()

# c.execute(""" CREATE TABLE customers (
#     first_name text,
#     last_name text,
#     email text

# # Create your connection.
# cnx = sqlite3.connect('file.db')

# df = pd.read_sql_query("SELECT * FROM table_name", cnx)


# )""")

def sql_fetch(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    print(cursorObj.fetchall())


c.execute("PRAGMA table_info('echo.co.uk_gb')")
print(c.fetchall())
sql_fetch(conn)
#c.execute("SELECT COUNT(*) from '1031922175_GB'")
# cur_result = c.fetchone()
# print(cur_result)


# #removing duplicate from apple playstore review
# c.execute("""DELETE FROM '1031922175_GB'
# WHERE ROWID NOT IN (SELECT MIN(rowid)
# FROM '1031922175_GB' GROUP BY id, author, 'im:version', 'im:rating', title, content)""")
# # conn.commit()

# #removing duplicate from google playstore review
# c.execute("""DELETE FROM 'echo.co.uk_gb'
# WHERE ROWID NOT IN (SELECT MIN(rowid)
# FROM 'echo.co.uk_gb' GROUP BY reviewId,
# userName, userImage, content,
# 'score', thumbsUpCount, reviewCreatedVersion, at,
# replyContent, repliedAt )""")

conn.commit()

# c.execute(
#     """CREATE TABLE dummy (
# 	contact_id INTEGER PRIMARY KEY,
# 	first_name TEXT NOT NULL,
# 	last_name TEXT NOT NULL,
# 	email TEXT NOT NULL UNIQUE,
# 	phone TEXT NOT NULL UNIQUE
# );"""
# )
c.execute("DROP TABLE 'com.ibuild.idothabit_us'")
cur_result = c.fetchone()
print(cur_result)
#sql_fetch(conn)

conn.commit()





#############
conn.close()
