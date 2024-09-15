import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import streamlit as st

# Function to scrape news from MoneyControl
def scrape_moneycontrol_news():
    url = 'https://www.moneycontrol.com/news/'
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the news headlines (adjust the class based on website structure)
        headlines = soup.find_all('h2', class_='widget-title')  # Example class
        
        # Store headlines in a list
        news_list = []
        for headline in headlines[:5]:  # Limit to the top 5 headlines
            news_list.append(headline.get_text())
        
        return news_list
    else:
        return []

# Function to analyze the sentiment of headlines
def analyze_sentiment(news_list):
    sentiments = []
    for headline in news_list:
        analysis = TextBlob(headline)
        sentiments.append(analysis.sentiment.polarity)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    sentiment_prediction = "Bullish" if avg_sentiment > 0 else "Bearish"
    
    return sentiment_prediction, avg_sentiment

# Streamlit UI to display the news and sentiment prediction
def main():
    st.title("Live Market Sentiment and News Scraping")

    # Scrape news from MoneyControl
    st.subheader("Latest News from MoneyControl")
    news_list = scrape_moneycontrol_news()
    
    if news_list:
        for news in news_list:
            st.write(f"- {news}")
        
        # Analyze sentiment of the news
        sentiment_prediction, avg_sentiment = analyze_sentiment(news_list)
        
        st.subheader("Market Sentiment Prediction")
        if sentiment_prediction == "Bullish":
            st.markdown("<h3 style='color: green;'>Bullish</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>Bearish</h3>", unsafe_allow_html=True)
        
        st.write(f"Sentiment Score: {avg_sentiment:.2f}")
    else:
        st.write("Failed to fetch news.")

if __name__ == '__main__':
    main()
