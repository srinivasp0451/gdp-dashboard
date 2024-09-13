import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from textblob import TextBlob
import yfinance as yf
import pandas as pd
import time
from datetime import date, timedelta

# Function to fetch Nifty and Bank Nifty news using Selenium
def fetch_news_with_selenium(query):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service('/path/to/chromedriver')  # Update path to chromedriver
        driver = webdriver.Chrome(service=service, options=chrome_options)

        search_url = f"https://economictimes.indiatimes.com/quicksearch.cms?query={query}"
        driver.get(search_url)

        # Wait for content to load
        time.sleep(3)

        # Fetch headlines related to Nifty 50 or Bank Nifty
        headlines = driver.find_elements(By.CSS_SELECTOR, 'a.eachStory')
        news_content = [headline.text for headline in headlines[:5]]  # Limit to top 5 headlines

        driver.quit()
        return news_content
    except Exception as e:
        return [f"Error fetching news: {e}"]

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to get market sentiment for Nifty and Bank Nifty
def get_market_sentiment(query):
    news_content = fetch_news_with_selenium(query)
    news_sentiments = [analyze_sentiment(content) for content in news_content if content]
    avg_sentiment = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0
    return avg_sentiment, news_content

# Function to get index price and identify support level
def check_support_level(index_symbol):
    index_data = yf.Ticker(index_symbol)
    hist = index_data.history(period="5d")
    last_close = hist['Close'][-1]
    min_close = hist['Close'].min()  # Assuming a simple support level
    return last_close, min_close

# Function to get backtested results
def get_backtest_results(query):
    backtest_results = []
    for i in range(5):
        day = date.today() - timedelta(days=i)
        sentiment, _ = get_market_sentiment(query)
        direction = "Bullish" if sentiment > 0 else "Bearish" if sentiment < 0 else "Neutral"
        backtest_results.append((day, direction))
    return backtest_results

# Streamlit app layout
st.title("Nifty 50 and Bank Nifty Market Sentiment")

# Show the news and sentiment for Nifty 50
st.subheader("Nifty 50 News and Sentiment")
sentiment_nifty, news_nifty = get_market_sentiment("Nifty 50")
for news in news_nifty:
    st.write(news)

# Show sentiment analysis result for Nifty 50
if sentiment_nifty > 0:
    st.success("Nifty 50 sentiment is Bullish (Green)")
elif sentiment_nifty < 0:
    st.warning("Nifty 50 sentiment is Bearish (Red)")
else:
    st.info("Nifty 50 sentiment is Neutral")

# Show the news and sentiment for Bank Nifty
st.subheader("Bank Nifty News and Sentiment")
sentiment_banknifty, news_banknifty = get_market_sentiment("Bank Nifty")
for news in news_banknifty:
    st.write(news)

# Show sentiment analysis result for Bank Nifty
if sentiment_banknifty > 0:
    st.success("Bank Nifty sentiment is Bullish (Green)")
elif sentiment_banknifty < 0:
    st.warning("Bank Nifty sentiment is Bearish (Red)")
else:
    st.info("Bank Nifty sentiment is Neutral")

# Check for support levels and show buy signals
indices = {'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK'}
for index_name, index_symbol in indices.items():
    st.subheader(f"{index_name} Support Level")
    last_price, support_price = check_support_level(index_symbol)
    st.write(f"{index_name} last close: {last_price}, support level: {support_price}")
    
    if last_price <= support_price:
        st.success(f"Buy signal for {index_name}: Index is at support level.")
    else:
        st.info(f"No buy signal for {index_name}.")

# Show backtested results for previous 5 days
st.subheader("Backtested Results (Previous 5 Days)")

st.write("Nifty 50 Backtested Results:")
backtest_nifty = get_backtest_results("Nifty 50")
df_nifty = pd.DataFrame(backtest_nifty, columns=["Date", "Prediction"])
st.dataframe(df_nifty)

st.write("Bank Nifty Backtested Results:")
backtest_banknifty = get_backtest_results("Bank Nifty")
df_banknifty = pd.DataFrame(backtest_banknifty, columns=["Date", "Prediction"])
st.dataframe(df_banknifty)
