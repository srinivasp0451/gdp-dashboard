import streamlit as st
from bs4 import BeautifulSoup as BS
import requests as req
from textblob import TextBlob
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import pytz

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

# Function to calculate the next Thursday (expiry day) dynamically
def get_next_expiry(current_date):
    days_until_thursday = (3 - current_date.weekday()) % 7  # Thursday is weekday 3
    if days_until_thursday == 0:  # If today is Thursday, consider next Thursday
        days_until_thursday = 7
    return current_date + timedelta(days=days_until_thursday)

# Function to predict index price range with confidence intervals for the next business day
def predict_next_day_range(symbol, historical_days=100, confidence_levels=[68, 95, 99]):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=historical_days)
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        return f"No data available for the symbol {symbol}."

    data['Return'] = data['Adj Close'].pct_change()
    daily_avg_return = data['Return'].mean()
    daily_sd = data['Return'].std()
    current_price = data['Adj Close'].iloc[-1]

    confidence_intervals = {}
    for confidence_level in confidence_levels:
        z_score = norm.ppf(1 - (1 - confidence_level / 100) / 2)
        upper_range = current_price * (1 + (daily_avg_return + z_score * daily_sd))
        lower_range = current_price * (1 + (daily_avg_return - z_score * daily_sd))

        confidence_intervals[f"{confidence_level}% Confidence"] = {
            "Upper Range": upper_range,
            "Lower Range": lower_range
        }

    next_expiry = get_next_expiry(datetime.now())
    days_to_expiry = (next_expiry - datetime.now()).days

    result = {
        "Current Price": current_price,
        "Daily Avg Return": daily_avg_return * 100,
        "Daily Std Dev": daily_sd * 100,
        "Next Expiry Date": next_expiry.strftime('%Y-%m-%d'),
        "Days to Expiry": days_to_expiry,
        "Confidence Intervals": confidence_intervals
    }
    
    return result

# Streamlit app layout
st.title("Market Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Module", ("Index Price Prediction", "Market News Sentiment Analysis"))

# Module for Index Price Prediction
if options == "Index Price Prediction":
    st.header("Index Price Prediction with Confidence Intervals")

    indices = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCAP NIFTY": "NIFTY_MID_SELECT.NS"
    }

    selected_index = st.selectbox("Select an Index", list(indices.keys()))
    historical_days = st.slider("Select Number of Historical Days for Analysis", min_value=1, max_value=365, value=100)
    confidence_levels = st.multiselect("Select Confidence Levels", options=[68, 95, 99], default=[68, 95])

    if st.button("Predict"):
        symbol = indices[selected_index]
        result = predict_next_day_range(symbol, historical_days, confidence_levels)
        
        if isinstance(result, dict):
            st.write(f"**Current Price:** {result['Current Price']:.2f}")
            st.write(f"**Daily Avg Return:** {result['Daily Avg Return']:.4f}%")
            st.write(f"**Daily Std Dev:** {result['Daily Std Dev']:.4f}%")
            st.write(f"**Next Expiry Date:** {result['Next Expiry Date']}")
            st.write(f"**Days to Expiry:** {result['Days to Expiry']}")
            
            for confidence, levels in result['Confidence Intervals'].items():
                st.write(f"### {confidence}:")
                st.write(f"  **Upper Range:** {levels['Upper Range']:.2f}")
                st.write(f"  **Lower Range:** {levels['Lower Range']:.2f}")
        else:
            st.error(result)

# Module for Market News Sentiment Analysis
elif options == "Market News Sentiment Analysis":
    st.header("Market News Sentiment Analysis")

    # Display current date and time
    st.write(datetime.now(pytz.timezone('Asia/Kolkata')))

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

        # Sort sentiments by count in descending order
        sorted_sentiments = sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=False)
        sentiments, counts = zip(*sorted_sentiments)

        # Create a bar plot
        fig, ax = plt.subplots()
        bars = ax.barh(sentiments, counts, color='skyblue')

        # Add count labels on top of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width}', va='center')

        ax.set_xlabel('Count')
        ax.set_title('Sentiment Counts of Market News')
        plt.tight_layout()
        
        # Show the plot in Streamlit
        st.pyplot(fig)

    else:
        st.write("No news found or failed to fetch news.")
