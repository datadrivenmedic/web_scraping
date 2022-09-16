
import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


""" device_list = []
price_list = []
country_withcondition = []
root = 'https://bimedis.com/search/search-items/laboratory-equipment?mtid=0&buy=0&ps=10&ucur=1&page='
for i in range(0, 101, 1):
    data = rq.get(f'{root}{i}')
    soup = BeautifulSoup(data.content, 'html.parser')
    for devices in soup.find_all('h3', attrs = {'class' : 'b-third-level-title'}):
        device_list.append(devices.text)
    for prices in soup.find_all('span', attrs = {'class' : 'e-product-cost'}):
        price_list.append(prices.text)
    for country_condition in soup.find_all('span', attrs = {'class' : 'e-short-description-product-item-value'}):
        country_withcondition.append(country_condition.text)
        country_list = country_withcondition[::2]
        condition_list = country_withcondition[1::2]
bimedis_imgPrices = pd.DataFrame.from_dict({'imaging equipment':device_list, 'cost':price_list, 'condition':condition_list, 'country':country_list},orient= 'index')
#print(bimedis_imgPrices)
bimedis_imgPricesT = bimedis_imgPrices.transpose()
bimedis_imgPricesT.to_csv('C:/Users/HOURS/Documents/bimedis_data_project.csv', index = False, sep = ',', header = False)  """
              
df = pd.read_csv('C:/Users/HOURS/Documents/bimedis_data_project.csv', sep = ',', header = None, index_col =False)
df = df.rename(columns = {0 : 'equipment', 1 : 'cost', 2 : 'condition', 3: 'country'})

#print(df['country'].unique()) 
#print(df['condition'].unique())
conditions = (' Used', ' Refurbished', ' New',' Used Very Good', ' Used Good', 'For Parts', 'Used Like New', 'Certified Refurbished')



for c in conditions:
    idx = (df['country'] == c)
#df.loc[idx,['country']] = df.loc[idx,['condition']]
    df.loc[idx,['country','condition']] = df.loc[idx,['condition','country']].values

#print(df['country'].unique())

#print(df.loc[x, :]) 
#print(np.where(df['condition'] == '  USA'))
x = (df['condition'] == '  USA')
countries = ('  USA', '  Netherlands', '  Germany', '  Spain', '  France')
for f in countries:
    ind = (df['condition'] == f)
    df.loc[ind,['condition','country']] = df.loc[ind,['country','condition']].values
 

#print(df.isna())
tag = df['equipment'].isna()
#df.drop(df.loc[tag, :], index = 1)
#print(df.loc[tag, :])
#df.drop(np.where(df.loc[tag, :]))
#df.drop(df[df.loc[tag, :]])
df = df[df['equipment'].notnull()]
df = df[df['cost'].notnull()]

""" df['condition'].replace({r'\d': np.nan}, regex = True)
df['country'].replace({r'\d': np.nan}, regex = True) """
#convert both columns to string, regex replace and convert back
#df['country'] = df['country'].where(df['country'].str.isalpha())
df['country'] = df['country'].mask(pd.to_numeric(df['country'], errors='coerce').notna())
df['condition'] = df['condition'].mask(pd.to_numeric(df['condition'], errors='coerce').notna())
#df = df['country'].replace(r'[0-9]', np.nan, regex  = True)
#df = df['country'].replace(r'\d', np.nan, regex = True)
#x =df.groupby(['condition', 'country'])
#print(x.first())
#print(df.groupby(['country']).boxplot(df['cost']))
#print(df['cost'].dtypes)
#df['cost'] = df['cost'].astype('int64')

""" df['equipment'] = df['equipment'].replace(r'^[\t]', '', regex  = True)
df['cost'] = df['cost'].replace(r'^[\t]', '', regex  = True)
df['country'] = df['country'].replace(r'^[\t]', '', regex  = True)
df['condition'] = df['country'].replace(r'^[\t]', '', regex = True)
 """


df['country'] = df['country'].replace(r'^[ \t]+', '', regex  = True)
df['condition'] = df['condition'].replace(r'^[ \t]+', '', regex  = True)
df['equipment'] = df['equipment'].replace(r'^[ \t]+', '', regex  = True)
df['equipment'] = df['equipment'].str.strip()
#df['cost'] = df['cost'].replace(r'^[ \t\$]+', '', regex  = True)
df['cost'] = df['cost'].str.replace(r'$', '', regex  = True)
df['cost'] = df['cost'].str.replace(r' ', '', regex  = True)
df['condition'] = df['condition'].str.lstrip()
df['cost'] = df['cost'].str.lstrip()


#print(df['country'].where(df['country'] == 'USA'))
""" df.apply(pd.to_numeric, errors='ignore')
pd.to_numeric(df['cost'], errors='coerce')
df["cost"] = pd.to_numeric(df["cost"])
print(df['cost'].dtypes) """
#df['cost'] = int(float(df['cost']))
#df['cost'] = df['cost'].astype(float)
df['cost'] = df['cost'].astype(int)

df_numerized = df.copy()
for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype=='object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

#print(df.head())
#print(df.tail())

#print(df.dtypes)

#print(df.head(40))
print(df[df['equipment'].str.contains('Microscope', regex = True)])
print(df[df['equipment'].str.contains('microscope', regex = True)]) 

df_query1 = df[df['equipment'].str.contains('Microscope', regex = True)]
df_query2 = df[df['equipment'].str.contains('microscope', regex = True)]
#df_query = df_query1.append(df_query2, ignore_index= True)
df_query = pd.concat([df_query1, df_query2], ignore_index=True, join='inner')
#print(df_query.nlargest(10, 'cost'))
""" for e in (df_query['equipment']).apply(str):
    e.str.split('Microscope')
print(e(0)[-1]) 

df_types = df_query['equipment'].str.split('Microscope', n =1)
types = df_types[-1]
microscope_types = ['Light', 'Stereo', 'Electron', 'Flourescence']  """
print(df_query.describe())


#print(df_types[:10])
#rint(df_query.head(10))

#microscope_types = [Light, Stereo, Electron, Flourescence]

#print(df['equipment'].str.contains('microscope', regex = True))
#ids = np.where(df['equipment'].str.contains('microscope', regex = True))
#print(df.loc[ids, ['cost']])

#can i use advanced indexing to get the names of different microscopes

#df.groupby(['country']).plot()
#df.groupby(['condition']).sum().plot.bar()
#df.groupby(['condition']).size().plot().bar(x = 'condition', height ='cost')
#df.groupby(['country']).size().plot(kind = 'bar')
#df.boxplot(column=['cost'], by = 'condition', grid=False)
#df['country'].value_counts().plot(kind ='bar')


#bins = [100, 500, 1000, 1500, 2000, 2500, 3000]
#plt.hist(df.cost, bins, rwidth = 0.8)
""" sns.catplot(x = 'country', y = 'cost', data = df )
sns.catplot(x = 'country', y = 'cost', kind = 'box', data = df )"""

sns.catplot(x = 'condition', y = 'cost', kind = 'box', data = df )

""" crosstab = pd.crosstab(df.country, df.condition, margins = True, margins_name='total')
chi2, p, dof, exp = stats.chi2_contingency(crosstab)
print(f'chi2 value = {chi2} \np-value = {p}\ndegrees of freedom = {dof}\nexpected = {exp}') 
 """
plt.show()
#print(pd.crosstab(df.country, df.condition, margins = True, margins_name='total'))

#print(pd.crosstab(df_query.country, df_query.condition, margins = True, margins_name='total'))
#print(pd.crosstab(df_query.condition, df_query.cost, margins = True, margins_name='total'))
#sns.catplot(x = 'condition', y = 'cost', kind = 'box', data = df_query)
""" sns.catplot(x = 'condition', y = 'cost', data = df_query)
plt.show()  """

#df.plot.bar()
#df[' condition'].plot.bar()
""" df.plot.bar(x= 'condition', y ='cost')
plt.show()
 """

#print(df['cost'].describe())



#print(df.nlargest(10, 'cost'))
#print(df.nsmallest(10, 'cost'))

""" x =df['equipment'].astype(str).split('Microscope')
print(x[0]) """

#df.to_csv('C:/Users/HOURS/Documents/clean_bimedis_data_project.csv', index = False, sep = ',', header = True)



""" for x in conditions:
    if df['country'].values == x:
        df['country'].values == df['condition'].values
    else:
        df['country'].values == df['country'].values """ 
 
#for condition in conditions: 
  #df[['country','condition']] = df[['condition', 'country']].where(df['country'] == condition, df[['country','condition']].values)

#print(df['equipment'])
#print(df.columns)
