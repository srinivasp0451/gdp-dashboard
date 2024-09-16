import streamlit as st
from bs4 import BeautifulSoup as BS
import requests as req
from textblob import TextBlob

# Function to categorize overall sentiment based on average polarity score
def categorize_overall_sentiment(avg_polarity):
    if avg_polarity > 0.7:
        return "Very Strong Bullish"
    elif avg_polarity > 0.4:
        return "Strong Bullish"
    elif avg_polarity > 0.1:
        return "Medium Bullish"
    elif avg_polarity > 0.0:
        return "Weak Bullish"
    elif avg_polarity == 0.0:
        return "Neutral"
    elif avg_polarity > -0.1:
        return "Weak Bearish"
    elif avg_polarity > -0.4:
        return "Medium Bearish"
    elif avg_polarity > -0.7:
        return "Strong Bearish"
    else:
        return "Very Strong Bearish"

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to scrape market news
def get_market_news(url):
    try:
        webpage = req.get(url)
        trav = BS(webpage.content, "html.parser")

        news_list = []
        for link in trav.find_all('a'):
            if(str(type(link.string)) == "<class 'bs4.element.NavigableString'>"
               and len(link.string) > 35):
                news_list.append(link.string)
        return news_list
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Streamlit app
st.title("Market News Sentiment Analysis")

# URL of the market news website
url = "https://m.economictimes.com/markets"

st.write("Fetching latest market news...")
news_headlines = get_market_news(url)

if news_headlines:
    st.write("### Latest Market News Headlines:")

    total_polarity = 0
    num_headlines = 0

    for idx, headline in enumerate(news_headlines):
        polarity = analyze_sentiment(headline)
        sentiment = categorize_overall_sentiment(polarity)
        st.write(f"{idx + 1}. {headline} - Sentiment: {sentiment}")

        # Accumulate total polarity and count the number of headlines
        total_polarity += polarity
        num_headlines += 1

    # Calculate the average polarity and overall sentiment
    if num_headlines > 0:
        avg_polarity = total_polarity / num_headlines
        overall_sentiment = categorize_overall_sentiment(avg_polarity)
        st.write(f"### Overall Market Sentiment: {overall_sentiment} (Average Polarity: {avg_polarity})")
    else:
        st.write("No valid news headlines found.")
else:
    st.write("No news found or failed to fetch news.")
