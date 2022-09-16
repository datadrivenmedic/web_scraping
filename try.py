
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/HOURS/Documents/redditpathproject.csv')
#print(df['created'])
x = df['created'].str.split('.', expand = True)
df = pd.concat([df, x[0]], axis = 1)
#print(df[0])
""" for i in df[0]:
    datetime.utcfromtimestamp(int(float(i))) """
#print(df[df[0].notnull()])
""" df = df.rename(columns = {0 : 'new_date'})
df['new_date'] = df['new_date'].mask(pd.to_numeric(df['new_date']).notna())
df['new_date'] = df['new_date'].astype(int)
 """

df = df.drop(['created'], axis = 1 )
df = df.rename(columns = {0 : 'new_date'})
#pd.to_datetime(df['new_date'])
df['new_date'] = pd.to_numeric(df['new_date'], errors = 'coerce')

""" for i in df['new_date']:
    datetime.utcfromtimestamp(i) """

df = df[df['new_date'].notnull()]
df['new_date'] = df['new_date'].astype(int)

def getUTCstr(timestamp):
    return datetime.utcfromtimestamp(timestamp)

df['utc_new_date'] =df.apply(lambda row:getUTCstr(row["new_date"]),  axis=1)
print(df)

#print(df[df['new_date'].isna()])
#print(df[df['new_date'].notnull()])
#df['new_date'].astype('float').astype('int64')
#df['new_date'] = int(df['new_date'])
#print(df['new_date'].astype(int))
#print(df['new_date'].dtypes)
#df['new_date'].astype(str).astype(int)
#print(df[0].dtypes)

#df[0].str.replace('  ', '').astype(int)
#print(datetime.utcfromtimestamp(x[0]))
