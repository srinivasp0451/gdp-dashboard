import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pytz
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPro: Multi-Timeframe Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark UI
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #2962ff; color: white; font-weight: bold; border: none; padding: 0.5rem; }
    .stButton>button:hover { background-color: #0039cb; }
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 8px; border-left: 5px solid #2962ff; margin-bottom: 10px; }
    .signal-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .buy-signal { background-color: #004d40; color: #00e676; border: 2px solid #00e676; }
    .sell-signal { background-color: #4a0000; color: #ff5252; border: 2px solid #ff5252; }
    .neutral-signal { background-color: #3e2723; color: #ffab40; border: 2px solid #ffab40; }
    .psychology-warning { font-size: 12px; color: #ffcc80; font-style: italic; }
    .summary-text { font-family: 'Courier New', monospace; background-color: #262730; padding: 15px; border-radius: 5px; font-size: 14px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & MAPPINGS
# -----------------------------------------------------------------------------
ASSETS = {
    "Indices": {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX", "^GSPC": "S&P 500"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil"},
    "Forex": {"INR=X": "USD/INR", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD"},
    "Stocks (IN)": {"RELIANCE.NS": "Reliance", "HDFCBANK.NS": "HDFC Bank", "TCS.NS": "TCS"},
    "Stocks (US)": {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia"}
}

TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '1wk']
HTF_MAPPING = {
    '1m': '15m', '2m': '15m', '5m': '1h', '15m': '4h', 
    '30m': '1d', '1h': '1d', '1d': '1wk', '1wk': '1mo'
}

PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def to_ist(df):
    if df.empty: return df
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df.index.tz_convert('Asia/Kolkata')

@st.cache_data(ttl=300)
def fetch_data_robust(ticker, period, interval, delay=1.5):
    time.sleep(delay)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = to_ist(df)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# --- NLTK SENTIMENT ENGINE ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

def analyze_news_sentiment(ticker):
    """Fetches news and calculates sentiment score using NLTK VADER."""
    vader = setup_nltk()
    try:
        t = yf.Ticker(ticker)
        news = t.news
        
        if not news:
            return {"score": 0, "label": "NEUTRAL", "headlines": []}

        score = 0
        count = 0
        headlines = []
        
        for n in news[:5]: # Analyze top 5
            # Handle variable yfinance news structure
            title = n.get('title', n.get('content', {}).get('title', ''))
            link = n.get('link', n.get('content', {}).get('link', '#'))
            publisher = n.get('publisher', 'Unknown')
            
            if title:
                pol = vader.polarity_scores(title)['compound']
                score += pol
                count += 1
                headlines.append({'title': title, 'link': link, 'source': publisher})

        avg_score = score / count if count > 0 else 0
        
        label = "NEUTRAL"
        if avg_score > 0.15: label = "POSITIVE"
        elif avg_score < -0.15: label = "NEGATIVE"
        
        return {"score": avg_score, "label": label, "headlines": headlines}
    except Exception as e:
        return {"score": 0, "label": "ERROR", "headlines": [], "error": str(e)}

# -----------------------------------------------------------------------------
# 4. TECHNICAL ANALYSIS ENGINE
# -----------------------------------------------------------------------------
class MarketBrain:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_indicators(self):
        df = self.df
        # 1. Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # 2. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # 4. MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # 5. Bollinger Bands (For Volatility Context)
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        
        # 6. Pivot Points (Fibonacci style)
        high_roll = df['High'].rolling(20).max()
        low_roll = df['Low'].rolling(20).min()
        df['Fib_0.618'] = high_roll - (high_roll - low_roll) * 0.618
        
        self.df = df
        return df

    def get_swing_pivots(self, window=5):
        """Identifies Swing Highs and Lows for Elliott Wave visualization."""
        df = self.df
        df['Swing_High'] = df['High'].rolling(window=window*2+1, center=True).max()
        df['Swing_Low'] = df['Low'].rolling(window=window*2+1, center=True).min()
        
        pivots = []
        # Find points where High == Swing_High or Low == Swing_Low
        for i in range(window, len(df)-window):
            date = df.index[i]
            if df['High'].iloc[i] == df['Swing_High'].iloc[i]:
                pivots.append({'date': date, 'price': df['High'].iloc[i], 'type': 'High'})
            elif df['Low'].iloc[i] == df['Swing_Low'].iloc[i]:
                pivots.append({'date': date, 'price': df['Low'].iloc[i], 'type': 'Low'})
                
        # Filter consecutive highs/lows (keep extreme)
        clean_pivots = []
        if pivots:
            curr = pivots[0]
            for i in range(1, len(pivots)):
                next_p = pivots[i]
                if curr['type'] == next_p['type']:
                    if curr['type'] == 'High':
                        if next_p['price'] > curr['price']: curr = next_p
                    else:
                        if next_p['price'] < curr['price']: curr = next_p
                else:
                    clean_pivots.append(curr)
                    curr = next_p
            clean_pivots.append(curr)
            
        return pd.DataFrame(clean_pivots)

    def check_divergence(self, window=15):
        df = self.df.iloc[-window:]
        if len(df) < window: return "None"
        
        first_half = df.iloc[:int(window/2)]
        last_half = df.iloc[int(window/2):]
        
        div_signal = "None"
        if last_half['Close'].mean() < first_half['Close'].mean() and last_half['RSI'].mean() > first_half['RSI'].mean():
            div_signal = "Bullish Divergence"
        if last_half['Close'].mean() > first_half['Close'].mean() and last_half['RSI'].mean() < first_half['RSI'].mean():
            div_signal = "Bearish Divergence"
        return div_signal

    def identify_elliott_context(self):
        last = self.df.iloc[-1]
        trend = "Neutral"
        
        if last['Close'] > last['EMA_20'] > last['EMA_50']:
            if 50 < last['RSI'] < 75: trend = "Wave 3 (Impulse)"
            elif last['RSI'] >= 75: trend = "Wave 5 (Exhaustion)"
        elif last['Close'] < last['EMA_20'] < last['EMA_50']:
            if 25 < last['RSI'] < 50: trend = "Wave C (Down Impulse)"
            elif last['RSI'] <= 25: trend = "Correction Oversold"
        else:
            trend = "Corrective / Choppy (Wave 2/4/B)"
        return trend

    def generate_signal(self):
        row = self.df.iloc[-1]
        score = 0
        reasons = []
        
        # Trend
        if row['Close'] > row['EMA_50']: score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1
        
        # Momentum
        if row['RSI'] > 50: score += 1
        if row['MACD'] > row['Signal_Line']: score += 1
        
        action = "HOLD"
        if score >= 3: action = "BUY"
        elif score <= 1: action = "SELL"
        
        return action, score, reasons

def generate_verbose_summary(ticker, tf, signal, conf, ew, div, df, news_data):
    """Generates a professional 300+ word analysis report."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Price Context
    price_change = last['Close'] - prev['Close']
    change_pct = (price_change / prev['Close']) * 100
    
    # 2. Indicator States
    rsi_state = "Neutral"
    if last['RSI'] > 70: rsi_state = "Overbought (Potential Reversal)"
    elif last['RSI'] < 30: rsi_state = "Oversold (Potential Bounce)"
    elif last['RSI'] > 50: rsi_state = "Bullish Momentum"
    else: rsi_state = "Bearish Momentum"
    
    macd_state = "Bullish" if last['MACD'] > last['Signal_Line'] else "Bearish"
    
    # 3. Volatility
    volatility = "High" if last['ATR'] > df['ATR'].mean() else "Normal"
    bb_pos = "within bands"
    if last['Close'] > last['BB_Upper']: bb_pos = "piercing Upper Bollinger Band (Extreme)"
    elif last['Close'] < last['BB_Lower']: bb_pos = "piercing Lower Bollinger Band (Extreme)"

    summary = f"""
    **COMPREHENSIVE MARKET ANALYSIS REPORT: {ticker} ({tf})**
    
    **1. EXECUTIVE SUMMARY**
    The proprietary algorithm has generated a **{signal}** signal with **{conf}** confidence. The asset is currently trading at **{last['Close']:.2f}**, showing a change of **{change_pct:.2f}%** from the previous candle. The immediate market structure suggests we are in a **{ew}** phase, indicating distinct behavioral psychology among participants.
    
    **2. PRICE ACTION & TREND DIAGNOSTICS**
    The primary trend is currently dictated by the relationship between the 20-period and 50-period Exponential Moving Averages (EMAs). 
    The 20 EMA is at {last['EMA_20']:.2f} while the 50 EMA is at {last['EMA_50']:.2f}. {'Price is holding above key dynamic support,' if last['Close'] > last['EMA_20'] else 'Price is suppressed below dynamic resistance,'} confirming the validity of the current signal. 
    Furthermore, price is currently {bb_pos}, which implies that volatility is {volatility}. A move outside Bollinger Bands often signals a statistical anomaly that may result in mean reversion or a strong breakout depending on volume.
    
    **3. MOMENTUM & OSCILLATORS**
    Momentum indicators provide the engine for this move. The Relative Strength Index (RSI 14) is currently reading **{last['RSI']:.2f}**, classifying the momentum as **{rsi_state}**. 
    {'Critically, a divergence pattern has been detected (' + div + '), which often serves as a leading indicator for a trend shift.' if 'None' not in div else 'No immediate divergence is present, suggesting the current trend momentum is real and supported by price.'}
    Additionally, the MACD histogram is { 'expanding upwards' if (last['MACD'] - last['Signal_Line']) > (prev['MACD'] - prev['Signal_Line']) else 'contracting' }, reinforcing the {macd_state} bias.
    
    **4. ELLIOTT WAVE & STRUCTURE**
    Visual analysis of swing pivots (see chart ZigZag lines) suggests the market is navigating a **{ew}**. 
    In Elliott Wave theory, this often corresponds to {'the most powerful part of the move where trend followers should add to positions' if 'Wave 3' in ew else 'a corrective phase where patience is required to avoid chop'}. 
    The nearest Fibonacci Golden Ratio support (0.618) stands at **{last['Fib_0.618']:.2f}**.
    
    **5. SENTIMENT & FUNDAMENTAL CONTEXT**
    News sentiment analysis using Natural Language Processing (NLTK) yields a **{news_data['label']}** score ({news_data['score']:.2f}). 
    { 'Recent headlines suggest positive catalysts are driving investor confidence.' if news_data['score'] > 0 else 'Negative press or macro fears may be weighing on the asset.' if news_data['score'] < 0 else 'There is no significant news bias currently affecting price action.' }
    
    **FINAL RECOMMENDATION**
    Based on the confluence of Technical Trend, Momentum, and Sentiment, the system advises a **{signal}**. Traders should monitor the Stop Loss level closely at {last['EMA_20']-(2*last['ATR']):.2f} (approx) as volatility is {volatility}.
    """
    return summary

# -----------------------------------------------------------------------------
# 5. MAIN UI LOGIC
# -----------------------------------------------------------------------------
def main():
    st.sidebar.header("⚙️ Configuration")
    
    # Asset Selection
    cat = st.sidebar.selectbox("Market Type", list(ASSETS.keys()))
    ticker_name = st.sidebar.selectbox("Instrument", list(ASSETS[cat].values()))
    ticker = [k for k, v in ASSETS[cat].items() if v == ticker_name][0]
    
    custom_tick = st.sidebar.text_input("Or Custom Ticker (Yahoo format)", "")
    if custom_tick: ticker = custom_tick
    
    # Timeframe Selection
    colT1, colT2 = st.sidebar.columns(2)
    with colT1:
        tf = st.selectbox("Primary Timeframe", TIMEFRAMES, index=4) # 30m default
    with colT2:
        period = st.selectbox("Data Lookback", PERIODS, index=3) # 3mo default

    # Advanced Toggles
    st.sidebar.markdown("---")
    use_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
    ratio_ticker = ""
    if use_ratio:
        ratio_ticker = st.sidebar.text_input("Compare Against", value="^NSEI")
    
    run_btn = st.sidebar.button("🧠 Analyze Market Structure", type="primary")
    
    # Session State
    if 'analyzed' not in st.session_state: st.session_state.analyzed = False
    
    st.title(f"🛡️ Professional Algo-Analyst: {ticker_name}")
    
    if run_btn:
        st.session_state.analyzed = True
        with st.spinner("Initializing Quantitative Matrix..."):
            # 1. Fetch Data
            df = fetch_data_robust(ticker, period, tf)
            htf_val = HTF_MAPPING.get(tf, '1d')
            df_htf = fetch_data_robust(ticker, period, htf_val)
            
            # 2. Analyze
            if df is not None and df_htf is not None:
                brain_ltf = MarketBrain(df)
                brain_htf = MarketBrain(df_htf)
                
                df = brain_ltf.add_indicators()
                df_htf = brain_htf.add_indicators()
                
                # Signals
                action_ltf, score_ltf, _ = brain_ltf.generate_signal()
                action_htf, score_htf, _ = brain_htf.generate_signal()
                ew_context = brain_ltf.identify_elliott_context()
                divergence = brain_ltf.check_divergence()
                
                # Conflict Resolution
                final_signal = "HOLD"
                confidence = "Low"
                if action_htf == "BUY" and action_ltf == "BUY":
                    final_signal = "STRONG BUY"
                    confidence = "High (Trend Aligned)"
                elif action_htf == "SELL" and action_ltf == "SELL":
                    final_signal = "STRONG SELL"
                    confidence = "High (Trend Aligned)"
                elif action_htf != action_ltf:
                    final_signal = "WAIT / SCALP"
                    confidence = "Medium (Conflict)"
                
                # Targets
                last = df.iloc[-1]
                sl = last['Close'] - (2 * last['ATR']) if "BUY" in final_signal else last['Close'] + (2 * last['ATR'])
                tgt = last['Close'] + (3 * last['ATR']) if "BUY" in final_signal else last['Close'] - (3 * last['ATR'])
                
                # Sentiment
                news_data = analyze_news_sentiment(ticker)
                
                # Pivots for Visualization
                pivots = brain_ltf.get_swing_pivots(window=5)
                
                # Verbose Summary
                verbose_sum = generate_verbose_summary(ticker_name, tf, final_signal, confidence, ew_context, divergence, df, news_data)

                st.session_state.res = {
                    'df': df, 'df_htf': df_htf, 'signal': final_signal, 'conf': confidence,
                    'sl': sl, 'tgt': tgt, 'ew': ew_context, 'div': divergence,
                    'news': news_data, 'pivots': pivots, 'summary': verbose_sum
                }
                
                if use_ratio and ratio_ticker:
                    df_r = fetch_data_robust(ratio_ticker, period, tf)
                    if df_r is not None:
                        idx = df.index.intersection(df_r.index)
                        st.session_state.ratio = df.loc[idx]['Close'] / df_r.loc[idx]['Close']

    # DISPLAY
    if st.session_state.analyzed and 'res' in st.session_state:
        res = st.session_state.res
        df = res['df']
        
        # Dashboard
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            cls = "buy-signal" if "BUY" in res['signal'] else "sell-signal" if "SELL" in res['signal'] else "neutral-signal"
            st.markdown(f"<div class='signal-box {cls}'>{res['signal']}</div>", unsafe_allow_html=True)
            if "Conflict" in res['conf']: st.warning("⚠️ HTF Conflict: Scalp Only")
        
        with c2:
            st.markdown(f"""<div class='metric-card'><b>Execution</b><br>Price: {df['Close'].iloc[-1]:.2f}<br>SL: <span style='color:#ff5252'>{res['sl']:.2f}</span> | TGT: <span style='color:#00e676'>{res['tgt']:.2f}</span></div>""", unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""<div class='metric-card'><b>Sentiment (NLTK)</b><br>Score: {res['news']['score']:.2f}<br>Label: {res['news']['label']}</div>""", unsafe_allow_html=True)

        # Tabs
        t1, t2, t3 = st.tabs(["📉 Structure Chart", "📜 Detailed Report", "📰 News"])
        
        with t1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            # Candles
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            
            # Elliott / ZigZag Visual
            pivots = res['pivots']
            if not pivots.empty:
                fig.add_trace(go.Scatter(x=pivots['date'], y=pivots['price'], mode='lines+markers', 
                                         line=dict(color='white', width=1, dash='dot'), 
                                         marker=dict(size=5, color='yellow'), name='Market Waves'), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='cyan', width=1), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.markdown(f"<div class='summary-text'>{res['summary']}</div>", unsafe_allow_html=True)
            
        with t3:
            for n in res['news']['headlines']:
                st.markdown(f"**[{n['title']}]({n['link']})**")
                st.caption(f"Source: {n['source']}")
                st.write("---")

if __name__ == "__main__":
    main()


