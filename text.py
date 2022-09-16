import praw
import pandas  as pd
import numpy as np
import os
from time import time
from time import sleep

reddit = praw.Reddit(client_id = '796c7Tr9w1KrmKIsV5XI1A', client_secret ='12jRI4YgliEXxSpsHKAyYKA3t3Qlrw', user_agent ='thinker')
#To check if the code works
""" top_posts = reddit.subreddit('pathology').top(limit =30)
for post in top_posts:
    print(post.title)
 """

""" def reddit_data_grabber()
posts =[]
pathology_subreddit = reddit.subreddit('pathology')
instances = ['controversial', 'gilded', 'hot','new','rising','top']
for i in instances:
    for post in pathology_subreddit.$i(limit = 50):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.self.text, post.created])
df =pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'body', 'created'])
print(df) """

def reddit_data_grabber():
    posts =[]

    pathology_subreddit = reddit.subreddit('pathology')
    for post in pathology_subreddit.controversial(limit =150):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    """df1 = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments','subreddit', 'body', 'created'])
    for post in pathology_subreddit.gilded(limit =50): 
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    df2 = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'subreddit', 'body', 'created'])"""
    for post in pathology_subreddit.hot(limit =150):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    #df3 = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'subreddit', 'body', 'created'])
    for post in pathology_subreddit.new(limit =150):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    #df4 = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'subreddit', 'body', 'created'])
    for post in pathology_subreddit.rising(limit =150):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    #df5 = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'subreddit', 'body', 'created'])
    for post in pathology_subreddit.top(limit =150):
        posts.append([post.title, post.score, post.id, post.url, post.num_comments, post.subreddit, post.selftext, post.created])
    df = pd.DataFrame(posts, columns = ['title', 'score','id', 'url', 'num_comments', 'subreddit', 'body', 'created'])
    #print(df.head(30))
    #print(df.tail(15))
    if not os.path.isfile('C:/Users/HOURS/Documents/redditpathproject.csv'):
        df.to_csv('C:/Users/HOURS/Documents/redditpathproject.csv', index = False)
    else:
        df.to_csv('C:/Users/HOURS/Documents/redditpathproject.csv', mode ='a', index = False)


    """ if not os.path.isfile('C:/Users/HOURS/Documents/redditpath.xlsx'):
           df.to_excel('C:/Users/HOURS/Documents/redditpathfile.xlsx', header = 'column_names', index = False)
    else:
        df.to_excel('C:/Users/HOURS/Documents/redditpath.xlsx', mode ='a', header = False) """ 
   
""" dfs = [df1, df2, df3, df4, df5, df6]
df = ft.reduce(lambda left, right: pd.merge(left, right, on='subreddit'), dfs) 
print(df6) """

for i in range(33):
    reddit_data_grabber()
    sleep(100)
exit()  