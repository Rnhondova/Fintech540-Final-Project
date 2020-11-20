import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
# Import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt




# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# TextBlob
from textblob import TextBlob

#Twitter data imports
import tweepy as tw
from tweepy.streaming import StreamListener
import json
import re #regular expression
import string
import preprocessor as p
import tools.StockNewsAnalysis as sna
import tools.stock_prediction as stp

import nltk


from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# @st.cache
# nltk.download('vader_lexicon')
# @st.cache
# nltk.download('punkt')

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#display image
image = Image.open('./img/word_cloud.png')
st.image(image, width = 700)
st.title ('Stock Predictions for top 20 Stocks of S&P500 by Market Cap')


#About
expander_bar = st.beta_expander('About')
expander_bar.markdown("""
**Data Sources**: placeholder \n 
**Python Libraries**: placeholder \n
**References**: placeholder""")


#Twitter authorization
api_ = sna.get_twitter_authorization()
import tweepy as tw
import pandas as pd

consumer_key= 'OthHUPglwDY8It289lYXmfvNU'
consumer_secret= 'tbJNuuLAi5mVq1xzmVyapYde08CVsAMGYGDafueXV2nEGJSbrq'
access_token= '264003814-kJU4WfOnldjOomFz9u5jNp9iiJ7gbvaL1xgQUKXw'
access_token_secret= 'zbbTD8bi6JCn2M8oG9BkiKFOHG8orFxoT24pyxArWLf1U'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api_ = tw.API(auth, wait_on_rate_limit=True)
stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'V', 'WMT',
       'JNJ', 'TSLA', 'PG', 'MA', 'JPM', 'NVDA', 'UNH', 'HD', 'VZ', 'DIS',
       'ADBE', 'CRM']
selected_stock = st.selectbox('Stock', sorted(stocks))
# Define the search term and the date_since date as variables
search_words = "$" +selected_stock
date_since = "2020-11-05"
tweets = tw.Cursor(api_.search,
                       q=search_words,
                       lang="en",
                       since=date_since).items(500)


test = sna.get_tweets(api_,search_words,date_since,number_of_tweets = 10,include_retweets = False)

wordcloud, vader, blob = sna.create_word_cloud(test.clean_tweet)
pos = vader['pos']
neg = vader['neg']
subjectivity = blob.subjectivity

fig = plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig)
st.markdown(f'Positive sentiment: {pos} \n Negative Sentiment:{neg} \n Subjectivity: {subjectivity}')

stock_prediction = stp.latest_predictions(symbol=selected_stock, root_dir='Stock_Prediction_models')[1]
st.dataframe(stock_prediction)
