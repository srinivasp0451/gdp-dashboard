import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from scipy.signal import argrelextrema
import pytz
from datetime import datetime
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="ProAlgo Trader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "GOLD (Comex)": "GC=F",
    "SILVER": "SI=F",
    "USD/INR": "INR=X",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS"
}

IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def format_time_ago(dt_val):
    if pd.isna(dt_val): return "N/A"
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=IST)
    now = datetime.now(IST)
    diff = now - dt_val
    seconds = diff.total_seconds()
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 30: return dt_val.strftime("%Y-%m-%d %H:%M")
    if days > 0: return f"{days} days ago"
    if hours > 0: return f"{hours} hours ago"
    return f"{minutes} min ago"

@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    time.sleep(1.0) # Respectful delay
    try:
        # Fetch data
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Check if data is empty
        if df.empty:
            return None
        
        # Handle MultiIndex columns (Fix for yfinance updates)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try to drop the Ticker level if it exists
                df.columns = df.columns.droplevel(1)
            except:
                # Fallback: just get the first level
                df.columns = df.columns.get_level_values(0)
            
        # Ensure index is datetime and localized
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        df = df.dropna(subset=['Close'])
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# ==========================================
# TECHNICAL ANALYSIS ENGINE
# ==========================================

class MarketAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.calc_indicators()

    def calc_indicators(self):
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # EMAs
        self.df['EMA_9'] = self.df['Close'].ewm(span=9, adjust=False).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=20, adjust=False).mean()
        self.df['EMA_50'] = self.df['Close'].ewm(span=50, adjust=False).mean()
        self.df['EMA_200'] = self.df['Close'].ewm(span=200, adjust=False).mean()
        
        # Z-Score & Volatility
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Volatility'] = self.df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
        self.df['Z_Score'] = zscore(self.df['Close'].fillna(0))

    def get_support_resistance(self, n_bins=20):
        data = self.df['Close']
        counts, bins = np.histogram(data, bins=n_bins)
        sorted_indices = np.argsort(counts)[::-1]
        top_levels = []
        for i in sorted_indices[:3]:
            level = (bins[i] + bins[i+1]) / 2
            top_levels.append({'Level': level, 'Strength': counts[i]})
        return top_levels

    def get_fibonacci_levels(self):
        recent_period = min(100, len(self.df))
        sub = self.df.iloc[-recent_period:]
        max_p = sub['High'].max()
        min_p = sub['Low'].min()
        diff = max_p - min_p
        return {
            '0.0': min_p, '0.236': min_p + 0.236 * diff,
            '0.382': min_p + 0.382 * diff, '0.5': min_p + 0.5 * diff,
            '0.618': min_p + 0.618 * diff, '1.0': max_p
        }, max_p, min_p

    def elliott_wave_approx(self):
        closes = self.df['Close'].values
        try:
            pivots = argrelextrema(closes, np.greater, order=5)[0]
            if len(pivots) < 3: return "Indeterminate"
            last_3 = closes[pivots[-3:]]
            if last_3[0] < last_3[1] < last_3[2]: return "Impulse Phase (Potential Wave 3/5)"
            elif last_3[1] > last_3[2]: return "Correction Phase"
        except:
            return "Indeterminate"
        return "Consolidation"

# ==========================================
# BACKTESTING ENGINE
# ==========================================
def backtest_strategy(df):
    # Strategy: Buy Pullback (Price > 200EMA, RSI < 40)
    # Ensure indicators exist
    if 'EMA_200' not in df.columns: return {}
    
    df = df.copy()
    position = 0
    entry_price = 0
    pnl = []
    log = []
    
    start_idx = max(200, 1)
    
    for i in range(start_idx, len(df)):
        curr = df.iloc[i]
        date = df.index[i]
        
        # BUY
        if position == 0:
            if curr['Close'] > curr['EMA_200'] and curr['RSI'] < 40:
                position = 1
                entry_price = curr['Close']
                log.append({'Date': date, 'Type': 'BUY', 'Price': entry_price, 'Reason': 'Pullback Entry'})
        
        # SELL
        elif position == 1:
            if curr['Close'] >= entry_price * 1.05: # 5% TP
                pnl.append(curr['Close'] - entry_price)
                log.append({'Date': date, 'Type': 'SELL (TP)', 'Price': curr['Close'], 'Reason': 'Target Hit'})
                position = 0
            elif curr['Close'] <= entry_price * 0.98: # 2% SL
                pnl.append(curr['Close'] - entry_price)
                log.append({'Date': date, 'Type': 'SELL (SL)', 'Price': curr['Close'], 'Reason': 'Stop Hit'})
                position = 0
                
    accuracy = (len([x for x in pnl if x > 0]) / len(pnl)) * 100 if len(pnl) > 0 else 0
    return {'total_trades': len(pnl), 'accuracy': accuracy, 'total_points': sum(pnl), 'log': pd.DataFrame(log)}

# ==========================================
# MAIN UI
# ==========================================
st.sidebar.title("🛠️ Algo Setup")

ticker1_sym = st.sidebar.selectbox("Asset 1", list(TICKER_MAP.keys()), index=0)
enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
ticker2_sym = st.sidebar.selectbox("Asset 2 (Ratio)", list(TICKER_MAP.keys()), index=1) if enable_ratio else None

col1, col2 = st.sidebar.columns(2)
timeframe = col1.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)
period = col2.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)

if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("Fetching and Analyzing Data..."):
        # 1. Fetch Data
        t1_code = TICKER_MAP[ticker1_sym]
        df_raw = fetch_data(t1_code, period, timeframe)
        
        if df_raw is None or len(df_raw) < 20:
            st.error("Insufficient data. Try a longer period.")
            st.stop()
            
        # 2. Analyze (This creates the 'EMA_20' columns inside ana1.df)
        ana1 = MarketAnalyzer(df_raw)
        
        # IMPORTANT: Use this processed dataframe for charts and stats
        df_main = ana1.df 
        
        # Ratio Analysis
        ratio_df = None
        if enable_ratio and ticker2_sym:
            t2_code = TICKER_MAP[ticker2_sym]
            df2 = fetch_data(t2_code, period, timeframe)
            if df2 is not None:
                common_idx = df_raw.index.intersection(df2.index)
                if not common_idx.empty:
                    r_series = df_raw.loc[common_idx]['Close'] / df2.loc[common_idx]['Close']
                    ratio_df = pd.DataFrame(r_series, columns=['Close'])
                    ratio_ana = MarketAnalyzer(ratio_df) # Just to calculate indicators on ratio

        # ==================== TABS ====================
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Stats", "Backtest", "Charts"])
        
        # --- TAB 1: SUMMARY ---
        with tab1:
            curr_price = df_main['Close'].iloc[-1]
            # Use df_main (the processed df) to access EMA_20
            ema20 = df_main['EMA_20'].iloc[-1]
            ema200 = df_main['EMA_200'].iloc[-1]
            rsi = df_main['RSI'].iloc[-1]
            
            trend = "BULLISH" if curr_price > ema200 else "BEARISH"
            sr_levels = ana1.get_support_resistance()
            nearest_sr = min(sr_levels, key=lambda x: abs(x['Level'] - curr_price))
            
            fibs, _, _ = ana1.get_fibonacci_levels()
            
            st.markdown(f"""
            ### {ticker1_sym} Analysis
            * **Trend**: :{ 'green' if trend == 'BULLISH' else 'red' }[{trend}] (Price vs 200 EMA)
            * **RSI**: {rsi:.2f} ({ 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral' })
            * **Support/Resistance**: Nearest level at **{nearest_sr['Level']:.2f}**
            * **Key Fibonacci**: 0.618 level at **{fibs['0.618']:.2f}**
            * **Elliott Wave**: {ana1.elliott_wave_approx()}
            """)
            
            c1, c2 = st.columns(2)
            c1.metric("Price", f"{curr_price:.2f}", f"{curr_price - df_main['Close'].iloc[-2]:.2f}")
            c2.metric("Volatility", f"{df_main['Volatility'].iloc[-1]:.2f}")

        # --- TAB 2: STATS ---
        with tab2:
            st.write("#### Support & Resistance Levels")
            st.dataframe(pd.DataFrame(sr_levels), use_container_width=True)
            if ratio_df is not None:
                st.write(f"#### Ratio: {ticker1_sym}/{ticker2_sym}")
                st.line_chart(ratio_df['Close'])

        # --- TAB 3: BACKTEST ---
        with tab3:
            # Pass df_main (processed) to backtest
            res = backtest_strategy(df_main)
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Trades", res['total_trades'])
                c2.metric("Accuracy", f"{res['accuracy']:.1f}%")
                c3.metric("Total Points", f"{res['total_points']:.2f}")
                if not res['log'].empty:
                    st.dataframe(res['log'], use_container_width=True)
            else:
                st.warning("Not enough data for backtest.")

        # --- TAB 4: CHARTS ---
        with tab4:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            # Use df_main here
            fig.add_trace(go.Candlestick(x=df_main.index, open=df_main['Open'], high=df_main['High'],
                                         low=df_main['Low'], close=df_main['Close'], name='Price'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_main.index, y=df_main['EMA_20'], line=dict(color='orange'), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_main.index, y=df_main['EMA_200'], line=dict(color='blue'), name='EMA 200'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_main.index, y=df_main['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
