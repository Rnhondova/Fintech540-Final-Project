################
###Accounts to follow:
### @CNBC,@Benziga, @Stocktwits, @ @BreakoutStocks, @bespokeinvest, @WSJmarkets, @Stephanie_Link, @nytimesbusiness
### @IBDinvestors, @WSJDealJournal, @jimcramer, @MarketWatch
################

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

import nltk
nltk.data.path.append(os.path.dirname(__file__))
print('downloading nltk')
#try:
#   stopwords.words('english')
#except LookupError:
#   nltk.download('stopwords')
#   stopwords.words('english')
print('done')

print('downloading punkt')
#try:
#   nltk.data.find('tokenizers/punkt')
#except LookupError:
#   nltk.download('punkt')
print('done')


from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


consumer_key= 'OthHUPglwDY8It289lYXmfvNU'
consumer_secret= 'tbJNuuLAi5mVq1xzmVyapYde08CVsAMGYGDafueXV2nEGJSbrq'
access_token= '264003814-kJU4WfOnldjOomFz9u5jNp9iiJ7gbvaL1xgQUKXw'
access_token_secret= 'zbbTD8bi6JCn2M8oG9BkiKFOHG8orFxoT24pyxArWLf1U'

stop_words = set(stopwords.words('english'))

#HappyEmoticons
emoticons_happy = set([
                        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
                        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
                        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
                        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)','<3'
                      ])

# Sad Emoticons
emoticons_sad = set([
                        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
                        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
                        ':c', ':{', '>:\\', ';('
                    ])

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

def parse_news_data(news_tables):
    
    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splite text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                time = date_scrape[0]

            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]

            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, text])

    return parsed_news


def get_raw_news_data_for_ticker(symbol):
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    url = finwiz_url + symbol.lower()
    print(url)
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response,features="html.parser")
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    
    return news_table
    
def get_news_table_for_tickers(tickers, return_sentiment = True):
    news_tables = {}
    for ticker in tickers:

        # Add the table to our dictionary
        news_tables[ticker] = get_raw_news_data_for_ticker(ticker)
    
    news_output = parse_news_data(news_tables)
    
    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    news_output = pd.DataFrame(news_output, columns=columns)
    
    if return_sentiment:
        news_output = get_news_sentiment_analysis(news_output)
        
    return news_output

def get_textblob_sentiment(textstr):
    testimonial = TextBlob(textstr)
    return testimonial.sentiment

def get_vader_sentiment(textstr):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    return vader.polarity_scores(textstr)

def get_combined_vader_textblob_sentiment(df,column_name_with_text='headline'):
    
    #Textblob
    scores_txtblb = pd.DataFrame(df[column_name_with_text].apply(get_textblob_sentiment).tolist())
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = df[column_name_with_text].apply(get_vader_sentiment).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    df = df.join(scores_df, rsuffix='_right')
    df = df.join(scores_txtblb, rsuffix='_right')
    
    return df

def get_news_sentiment_analysis(parsed_news):
    
    #Sentiment analysis
    parsed_news = get_combined_vader_textblob_sentiment(parsed_news,column_name_with_text='headline')

    # Convert the date column from string to datetime
    parsed_news['date'] = pd.to_datetime(parsed_news.date).dt.date
    
    return parsed_news

def get_twitter_authorization():
    
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    return api

def post_to_twitter(api,text_twit):
    #Post to twitter
    return ''

    
def clean_tweets(twitter_text):
    clean_text = p.clean(twitter_text)
    
    #after tweepy preprocessing the colon symbol left remain after removing mentions
    clean_text = re.sub(r':', '', clean_text)
    clean_text = re.sub(r'‚Ä¶', '', clean_text)
    
    #replace consecutive non-ASCII characters with a space
    clean_text = re.sub(r'[^\x00-\x7F]+',' ', clean_text)
    
    #remove emojis from tweet
    clean_text = emoji_pattern.sub(r'', clean_text)
    
    word_tokens = nltk.word_tokenize(clean_text)
    
    #filter using NLTK library append it to a string
    #filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    
    #looping through conditions
    for w in word_tokens:
    
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)return tweet


def get_tweets(api,search_words,date_since,number_of_tweets = 1000,include_retweets = False):
    if include_retweets:
        search_words = search_words + " -filter:retweets"
    # Collect tweets
    tweets = tw.Cursor(api.search,
                  q=search_words,
                  lang="en",
                  since=date_since,
                  exclude_replies=True).items(number_of_tweets)
    # Collect a list of tweets
    tweets_text = [[tweet.id,tweet.created_at,tweet.lang,tweet.favorite_count, tweet.retweet_count, tweet.place, tweet.coordinates,tweet.text] for tweet in tweets]
    
    cols = [
            'id', 'created_at', 'lang',
            'favorite_count', 'retweet_count', 
            'place','coordinates', 'original_tweet'
           ]
    
    
    tweet_df = pd.DataFrame(data=tweets_text, columns=cols)
    tweet_df = tweet_df[~tweet_df.original_tweet.str.contains("RT @",case=True,regex=True)]
    
    #Clean tweet
    tweet_df['clean_tweet'] = tweet_df['original_tweet'].apply(clean_tweets)
    
    #Sentiment
    tweet_df = get_combined_vader_textblob_sentiment(tweet_df,column_name_with_text='clean_tweet')
    
    return tweet_df

def create_word_cloud(text_vec):
    
    text = " ".join(review for review in text_vec)
    #print ("There are {} words in the combination of all review.".format(len(text)))
    
    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["my", "trade"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    # plt.figure(figsize = (8, 8), facecolor = None)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.tight_layout(pad = 0)
    # plt.show()
    return wordcloud, get_vader_sentiment(text), get_textblob_sentiment(text)
    # Get sentiment score
    # print(get_vader_sentiment(text))
    # print(get_textblob_sentiment(text))
    
    

    
    

    
