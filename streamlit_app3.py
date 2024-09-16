import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import streamlit as st

# Function to scrape general market news from Economic Times
def get_market_news():
    url = 'https://m.economictimes.com/markets'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Scrape news headlines
    news_list = []
    for item in soup.find_all('h3', class_='headline'):
        headline = item.get_text(strip=True)
        news_list.append(headline)
    
    return news_list

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Determine sentiment polarity
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Streamlit App
def main():
    st.title("Market News Sentiment Analysis")

    # Add a button to scrape the latest news
    if st.button("Get Latest Market News & Sentiment"):
        st.write("Fetching latest market news...")
        news = get_market_news()
        
        if news:
            st.write("## Latest Market News Headlines & Sentiment:")
            for idx, headline in enumerate(news):
                sentiment = analyze_sentiment(headline)
                
                # Display news headline with corresponding sentiment
                st.write(f"**{idx + 1}. {headline}**")
                st.write(f"Sentiment: {sentiment}\n")
        else:
            st.write("No news found.")
    
    st.write("Click the button above to get the latest market news and sentiment.")

if __name__ == "__main__":
    main()
