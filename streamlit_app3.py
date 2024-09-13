import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import yfinance as yf

# Function to scrape news from Economic Times or other sources
def fetch_news_from_website(query):
    try:
        url = f"https://economictimes.indiatimes.com/quicksearch.cms?query={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('a', class_='eachStory')
        news_content = " ".join([headline.get_text() for headline in headlines[:5]])  # Limit to top 5 headlines
        st.write('news content', news_content)
        return news_content
    except Exception as e:
        return f"Error fetching news: {e}"

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to get market sentiment from scraped news
def get_market_sentiment(query):
    news_content = fetch_news_from_website(query)
    sentiment = analyze_sentiment(news_content)
    return sentiment, news_content

# Function to get index price and identify support level
def check_support_level(index_symbol):
    index_data = yf.Ticker(index_symbol)
    hist = index_data.history(period="5d")
    last_close = hist['Close'][-1]
    min_close = hist['Close'].min()  # Assuming a simple support level
    return last_close, min_close

# Streamlit app layout
st.title("Market Sentiment for Nifty 50 and Bank Nifty")
#st.write("This app fetches news articles and performs sentiment analysis to determine market sentiment. It also checks if the index is at a support level.")

# Index symbols
indices = {'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK'}

# Refresh button
if st.button("Refresh Sentiment"):
    # Fetch sentiment for Nifty 50
    sentiment_nifty,nifty_news = get_market_sentiment("Nifty 50")
    sentiment_banknifty,bn_news = get_market_sentiment("Bank Nifty")
    st.write('Nifty50 news - ',nifty_news)
    st.write('Bank nifty news - ',bn_news)
    # Show sentiment analysis result for Nifty 50
    if sentiment_nifty > 0:
        st.success("Nifty 50 sentiment is Bullish (Green)")
    elif sentiment_nifty < 0:
        st.warning("Nifty 50 sentiment is Bearish (Red)")
    else:
        st.info("Nifty 50 sentiment is Neutral")

    # Show sentiment analysis result for Bank Nifty
    if sentiment_banknifty > 0:
        st.success("Bank Nifty sentiment is Bullish (Green)")
    elif sentiment_banknifty < 0:
        st.warning("Bank Nifty sentiment is Bearish (Red)")
    else:
        st.info("Bank Nifty sentiment is Neutral")

    # Check for support levels and show buy signal
    for index_name, index_symbol in indices.items():
        #st.write(f"Checking support level for {index_name}...")
        last_price, support_price = check_support_level(index_symbol)
        #st.write(f"{index_name} last close: {last_price}, support level: {support_price}")
        
        if last_price <= support_price:
            st.success(f"Buy signal for {index_name}")
        else:
            st.info(f"No buy signal for {index_name}.")
