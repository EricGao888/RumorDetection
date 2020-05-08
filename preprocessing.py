#code to get user properties for a tweet id
import os
import sys
import tweepy

# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = 'EZguHG1uhJCKqbwnU2YwkFscS'
CONSUMER_SECRET = 'tkwOry3TYP9MV42ufJAeYJDqgdsSoIgujMsDfs8d3pji2KSkFq'
OAUTH_TOKEN = '1188553124530798593-Gi3zGQwLDP1dgtP8uN3i2eAQQsD7ke'
OAUTH_TOKEN_SECRET = 'gyYWVTfi5qeKiCie7AR61IaBtW1I7RwoAnixmQbbXmDvd'

# connect to twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True)
# batch size depends on Twitter limit, 100 at this time
fpath="Rumor_RvNN/nfold/RNNtrainSet_Twitter150_tree.txt"
import pandas as pd
import numpy as np
data = pd.read_csv(fpath, sep=" ", header=None)
data.columns = ["tweet_id"]
def get_user_prop(df):
    try:
        user_det = api.get_status(df).user
        follow_count=user_det.followers_count
        friend_count=user_det.friends_count
        follower_friend_ratio=user_det.followers_count/user_det.friends_count
        verify_status=user_det.verified
        reg_year=str(user_det.created_at).split('-')[0]
        history_tweet_count=user_det.listed_count
    except:
        #print("user doesn't have account with Twitter anymore")
        follow_count = "UNK"
        friend_count = "UNK"
        follower_friend_ratio = "UNK"
        verify_status = "UNK"
        reg_year = "UNK"
        history_tweet_count = "UNK"
    return follow_count,friend_count,follower_friend_ratio,verify_status,reg_year,history_tweet_count
df_new = data
df_new['new_user_properties']=data['tweet_id'].apply(get_user_prop)
df_new['follow_count']=np.nan
df_new['friend_count']=np.nan
df_new['follower_friend_ratio']=np.nan
df_new['verify_status']=np.nan
df_new['reg_year']=np.nan
df_new['history_tweet_count']=np.nan
def separate_cols(df):
    df['follow_count']=df['new_user_properties'][0]
    df['friend_count']=df['new_user_properties'][1]
    df['follower_friend_ratio']=df['new_user_properties'][2]
    df['verify_status']=df['new_user_properties'][3]
    df['reg_year']=df['new_user_properties'][4]
    df['history_tweet_count']=df['new_user_properties'][5]
    return df
df_new=df_new.apply(separate_cols,axis=1)
df_new= df_new.drop(['new_user_properties'], axis=1)
topath="Rumor_RvNN/nfold_new/RNNtrainSet_Twitter150_tree.txt"
df_new.to_csv(topath,sep=' ', index=False, header=False) 
