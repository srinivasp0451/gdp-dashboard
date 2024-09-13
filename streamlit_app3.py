import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
from datetime import date, timedelta

# Function to fetch news for Nifty 50 or Bank Nifty
def fetch_news(query):
    try:
        url = f"https://economictimes.indiatimes.com/quicksearch.cms?query={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = soup.find_all('a', class_='eachStory')
        news_content = [headline.get_text() for headline in headlines[:5]]  # Limit to top 5 headlines
        return news_content
    except Exception as e:
        return [f"Error fetching news: {e}"]

# Function to perform sentiment analysis on the news
def analyze_sentiment(news_list):
    sentiment_scores = []
    for news in news_list:
        blob = TextBlob(news)
        sentiment_scores.append(blob.sentiment.polarity)
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Function to get backtested sentiment results for the past 5 days
def get_backtested_results(query):
    results = []
    for i in range(5):
        day = date.today() - timedelta(days=i)
        news = fetch_news(query)
        sentiment = analyze_sentiment(news)
        direction = "Bullish" if sentiment > 0 else "Bearish" if sentiment < 0 else "Neutral"
        results.append((day, direction))
    return results

# Streamlit app layout
st.title("Nifty 50 and Bank Nifty Market Sentiment")

# Show the news and sentiment for Nifty 50
st.subheader("Nifty 50 News and Sentiment")
news_nifty = fetch_news("Nifty 50")
for news in news_nifty:
    st.write(news)

# Show sentiment analysis result for Nifty 50
sentiment_nifty = analyze_sentiment(news_nifty)
if sentiment_nifty > 0:
    st.success("Nifty 50 sentiment is Bullish (Green)")
elif sentiment_nifty < 0:
    st.warning("Nifty 50 sentiment is Bearish (Red)")
else:
    st.info("Nifty 50 sentiment is Neutral")

# Show the news and sentiment for Bank Nifty
st.subheader("Bank Nifty News and Sentiment")
news_banknifty = fetch_news("Bank Nifty")
for news in news_banknifty:
    st.write(news)

# Show sentiment analysis result for Bank Nifty
sentiment_banknifty = analyze_sentiment(news_banknifty)
if sentiment_banknifty > 0:
    st.success("Bank Nifty sentiment is Bullish (Green)")
elif sentiment_banknifty < 0:
    st.warning("Bank Nifty sentiment is Bearish (Red)")
else:
    st.info("Bank Nifty sentiment is Neutral")

# Backtested results for Nifty 50
st.subheader("Backtested Results: Nifty 50 (Last 5 Days)")
backtested_nifty = get_backtested_results("Nifty 50")
df_nifty = pd.DataFrame(backtested_nifty, columns=["Date", "Prediction"])
st.dataframe(df_nifty)

# Backtested results for Bank Nifty
st.subheader("Backtested Results: Bank Nifty (Last 5 Days)")
backtested_banknifty = get_backtested_results("Bank Nifty")
df_banknifty = pd.DataFrame(backtested_banknifty, columns=["Date", "Prediction"])
st.dataframe(df_banknifty)
