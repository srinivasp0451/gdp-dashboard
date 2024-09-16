import streamlit as st
from bs4 import BeautifulSoup as BS
import requests as req
from textblob import TextBlob
import matplotlib.pyplot as plt

# Function to categorize sentiment based on polarity score
def categorize_sentiment(polarity):
    if polarity > 0.7:
        return "Very Strong Bullish"
    elif polarity > 0.4:
        return "Strong Bullish"
    elif polarity > 0.1:
        return "Medium Bullish"
    elif polarity > 0.0:
        return "Weak Bullish"
    elif polarity == 0.0:
        return "Neutral"
    elif polarity > -0.1:
        return "Weak Bearish"
    elif polarity > -0.4:
        return "Medium Bearish"
    elif polarity > -0.7:
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

    # Initialize sentiment counts
    sentiment_counts = {
        "Very Strong Bullish": 0,
        "Strong Bullish": 0,
        "Medium Bullish": 0,
        "Weak Bullish": 0,
        "Neutral": 0,
        "Weak Bearish": 0,
        "Medium Bearish": 0,
        "Strong Bearish": 0,
        "Very Strong Bearish": 0
    }

    for idx, headline in enumerate(news_headlines):
        polarity = analyze_sentiment(headline)
        sentiment = categorize_sentiment(polarity)
        st.write(f"{idx + 1}. {headline} - Sentiment: {sentiment}")

        # Update sentiment counts
        sentiment_counts[sentiment] += 1

    # Display sentiment counts
    st.write("### Sentiment Counts:")

    # Create a bar plot
    fig, ax = plt.subplots()
    sentiments = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    
    ax.barh(sentiments, counts, color='skyblue')
    ax.set_xlabel('Count')
    ax.set_title('Sentiment Counts of Market News')
    plt.tight_layout()
    
    # Show the plot in Streamlit
    st.pyplot(fig)

else:
    st.write("No news found or failed to fetch news.")
