import streamlit as st
from bs4 import BeautifulSoup as BS
import requests as req
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import json

# ==================== News Sentiment Analysis Functions =====================

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

# ==================== Option Chain Functions =====================

# URL for Nifty 50 Option Chain
nse_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

# Headers to avoid 403 error
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Function to fetch data from NSE
def fetch_option_chain():
    session = req.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(nse_url, headers=headers)
    
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        st.error("Failed to fetch data from NSE")
        return None

# Parse and convert option chain data to DataFrame
def parse_option_chain(data):
    ce_data = []
    pe_data = []
    
    for row in data['records']['data']:
        if 'CE' in row:
            ce_data.append(row['CE'])
        if 'PE' in row:
            pe_data.append(row['PE'])
    
    ce_df = pd.DataFrame(ce_data)
    pe_df = pd.DataFrame(pe_data)
    
    return ce_df, pe_df

# Get spot price
def get_spot_price(data):
    return data['records']['underlyingValue']

# Filter option chain to get nearby strike prices around spot price
def filter_nearby_strikes(ce_df, pe_df, spot_price, num_strikes=5):
    nearest_strike = ce_df.iloc[(ce_df['strikePrice'] - spot_price).abs().argsort()[:1]]['strikePrice'].values[0]
    strike_range = ce_df[(ce_df['strikePrice'] >= nearest_strike - num_strikes * 50) & 
                         (ce_df['strikePrice'] <= nearest_strike + num_strikes * 50)]
    
    ce_filtered = ce_df[ce_df['strikePrice'].isin(strike_range['strikePrice'])]
    pe_filtered = pe_df[pe_df['strikePrice'].isin(strike_range['strikePrice'])]
    
    return ce_filtered, pe_filtered, nearest_strike

# Plot the premiums of CE and PE using line charts
def plot_premiums(ce_filtered, pe_filtered, nearest_strike, spot_price):
    plt.figure(figsize=(12, 8))
    
    # Call Option Premiums
    plt.subplot(2, 1, 1)
    plt.plot(ce_filtered['strikePrice'], ce_filtered['lastPrice'], marker='o', color='blue', label='CE Premium')
    plt.axvline(x=spot_price, color='red', linestyle='--', label='Spot Price')
    plt.axvline(x=nearest_strike, color='green', linestyle='--', label='Nearest Strike Price')
    plt.title('Call Option (CE) Premiums')
    plt.ylabel('Premium')
    plt.legend()
    
    # Put Option Premiums
    plt.subplot(2, 1, 2)
    plt.plot(pe_filtered['strikePrice'], pe_filtered['lastPrice'], marker='o', color='red', label='PE Premium')
    plt.axvline(x=spot_price, color='red', linestyle='--', label='Spot Price')
    plt.axvline(x=nearest_strike, color='green', linestyle='--', label='Nearest Strike Price')
    plt.title('Put Option (PE) Premiums')
    plt.xlabel('Strike Price')
    plt.ylabel('Premium')
    plt.legend()
    
    st.pyplot(plt)

# ==================== Streamlit App =====================

# Sidebar navigation for different functionalities
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Option Chain Premium Analysis", "Market Sentiment Analysis"])

if page == "Option Chain Premium Analysis":
    st.title("Nifty50 Option Chain Premium Analysis")
    
    # Fetch option chain data
    data = fetch_option_chain()
    if data:
        # Get spot price
        spot_price = get_spot_price(data)
        st.write(f"**Spot Price:** {spot_price}")
        
        # Parse the data
        ce_df, pe_df = parse_option_chain(data)

        # Dropdown to select strike price levels above and below spot price
        num_strikes = st.selectbox("Select number of strike price levels:", options=[3, 5, 7, 10], index=1)

        # Filter the data to get nearby strike prices
        ce_filtered, pe_filtered, nearest_strike = filter_nearby_strikes(ce_df, pe_df, spot_price, num_strikes)

        # Plot premiums of CE and PE options using line charts
        plot_premiums(ce_filtered, pe_filtered, nearest_strike, spot_price)
    else:
        st.write("No data fetched.")

elif page == "Market Sentiment Analysis":
    st.title("Market News Sentiment Analysis")
    
    # Fetch current date and time in the required timezone
    import datetime
    import pytz
    st.write(datetime.datetime.now(pytz.timezone('Asia/Kolkata')))
    
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
