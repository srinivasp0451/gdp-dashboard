import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# Function to scrape Nifty 50 news from Economic Times or MoneyControl
def get_nifty50_news():
    url = 'https://economictimes.indiatimes.com/markets/indices/nifty-50'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Scrape news headlines
    news_list = []
    for item in soup.find_all('h3'):
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

# Main function to get news and analyze sentiment
def main():
    news = get_nifty50_news()
    if news:
        print("Latest Nifty 50 News Headlines & Sentiment Analysis:")
        for idx, headline in enumerate(news):
            sentiment = analyze_sentiment(headline)
            print(f"{idx + 1}. {headline} - Sentiment: {sentiment}")
    else:
        print("No news found.")

if __name__ == "__main__":
    main()
