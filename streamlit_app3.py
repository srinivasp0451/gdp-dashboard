import streamlit as st
import requests
from newspaper import Article
from textblob import TextBlob
import yfinance as yf

# Function to fetch news article content
def fetch_news(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article.text

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to get market sentiment
def get_market_sentiment(news_urls):
    sentiment_scores = []
    for url in news_urls:
        try:
            news_content = fetch_news(url)
            sentiment = analyze_sentiment(news_content)
            sentiment_scores.append(sentiment)
        except Exception as e:
            st.error(f"Error fetching or analyzing news: {e}")
    
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Function to get index price and identify support level
def check_support_level(index_symbol):
    index_data = yf.Ticker(index_symbol)
    hist = index_data.history(period="5d")
    last_close = hist['Close'][-1]
    min_close = hist['Close'].min()  # Assuming a simple support level
    return last_close, min_close

# Streamlit app layout
st.title("Market Sentiment for Nifty 50 and Bank Nifty")
st.write("This app fetches news articles and performs sentiment analysis to determine market sentiment. It also checks if the index is at a support level.")

# Input for news URLs
news_urls = st.text_area("Enter news article URLs (one per line):").splitlines()

# Index symbols
indices = {'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK'}

# Refresh button
if st.button("Refresh Sentiment"):
    if news_urls:
        market_sentiment = get_market_sentiment(news_urls)
        
        # Show sentiment analysis result
        if market_sentiment > 0:
            st.success("Market sentiment is Bullish (Green)")
        elif market_sentiment < 0:
            st.warning("Market sentiment is Bearish (Red)")
        else:
            st.info("Market sentiment is Neutral")

        # Check for support levels and show buy signal
        for index_name, index_symbol in indices.items():
            st.write(f"Checking support level for {index_name}...")
            last_price, support_price = check_support_level(index_symbol)
            st.write(f"{index_name} last close: {last_price}, support level: {support_price}")
            
            if last_price <= support_price:
                st.success(f"Buy signal for {index_name}: Index is at support level.")
            else:
                st.info(f"No buy signal for {index_name}.")
    else:
        st.error("Please enter at least one news URL.")
