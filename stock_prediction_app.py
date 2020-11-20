import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
# Import libraries
from urllib.request import urlopen, Request
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


#Custom libraries
import tools.StockNewsAnalysis as sna
import tools.stock_prediction as stp


# @st.cache
# nltk.download('vader_lexicon')
# @st.cache
# nltk.download('punkt')

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#display image
image = Image.open('./img/stocks.jpg')
st.image(image, width = 700)
st.title ('Stock Predictions for top 20 Stocks of S&P500 by Market Cap')


#About
expander_bar = st.beta_expander('About this App')
expander_bar.markdown("""
**Description**: This apps allows you to check sentiment for the top S&P500 stocks for the past week and check the estimated closing price of the selected stock for the next business day.\n
**Audience**: Short-term traders /n 
**Data Sources**: yahoofinancials & alpha_vantage (python packages)\n 
**Methods**: Sentiment analysis was conducted on Twitter Data for the past week. [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) was used to predict clsoing price of the stock.\n 
**Python Libraries**: Streamlit, nltk, textblob, wordcloud and others \n
**Authors**: Ronald Nhondova and Alena Kalodzitsa\n """)


#Twitter authorization
api_ = sna.get_twitter_authorization()

import pandas as pd


stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'V', 'WMT',
       'JNJ', 'TSLA', 'PG', 'MA', 'JPM', 'NVDA', 'UNH', 'HD', 'VZ', 'DIS',
       'ADBE', 'CRM']
selected_stock = st.selectbox('Stock', sorted(stocks))
# Define the search term and the date_since date as variables
search_words = "$" +selected_stock
date_since = "2020-11-05"

test = sna.get_tweets(api_,search_words,date_since,number_of_tweets = 50,include_retweets = False)

wordcloud, vader, blob = sna.create_word_cloud(test.clean_tweet)
pos = np.round(vader['pos'], 2)
neg = np.round(vader['neg'], 2)
subjectivity = np.round(blob.subjectivity, 2)

fig = plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title(f'Wordcloud for tweets relating to {selected_stock}')
st.pyplot(fig)

st.markdown(f'*Size of a word indicates it\'s frequency and importance*:\n', )

st.empty()
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.markdown(f'**Sentiment Analysis**:\n \n')
st.markdown('\n')
st.markdown(f'Positive sentiment: {pos}')
st.markdown(f'Negative sentiment: {neg}')
st.markdown(f'Subjectivity: {subjectivity}')

st.markdown('🤔 [Wondering how to interpret these sentiment scores?](https://en.wikipedia.org/wiki/Sentiment_analysis)')

stock_prediction = stp.latest_predictions(symbol=selected_stock, root_dir='Stock_Prediction_models')[1]
#st.dataframe(stock_prediction)
fig2 = px.line(stock_prediction, x=stock_prediction.index, y=stock_prediction.columns,

              title=f'Actual vs Predicted Price for {selected_stock}, including next business day')
fig2.update_xaxes(title_text = 'Time')
fig2.update_yaxes(title_text = 'Stock Price ($)')
fig2.update_layout(legend_title_text='Price')
st.write(fig2)
st.markdown(f'Prediction of Closing Price for {selected_stock} for the Next Business Day:')
#price_next_day = stock_prediction[['Predicted Price']].iloc[-1,:]
price_next_day = stock_prediction.iloc[-1,1]
st.write(price_next_day)
st.markdown('\n')
st.markdown('\n')
st.markdown('\n')
st.markdown('**Disclaimer**: Current content is for informational purpose only and it is not intended to be financial advice.')
