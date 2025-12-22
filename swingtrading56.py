import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from scipy.signal import argrelextrema
import pytz
from datetime import datetime, timedelta
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

# Custom CSS for Professional Look
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #262730; padding: 15px; border-radius: 5px; margin-bottom: 10px;}
    .bullish {color: #00FF00; font-weight: bold;}
    .bearish {color: #FF0000; font-weight: bold;}
    .neutral {color: #FFA500; font-weight: bold;}
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
    """Converts datetime to human readable 'X hours ago'"""
    if pd.isna(dt_val): return "N/A"
    
    # Ensure dt_val is timezone-aware before comparison
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=IST)
    
    now = datetime.now(IST)
    diff = now - dt_val
    
    seconds = diff.total_seconds()
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days > 30:
        return dt_val.strftime("%Y-%m-%d %H:%M")
    if days > 0:
        return f"{days} days ago"
    if hours > 0:
        return f"{hours} hours ago"
    return f"{minutes} min ago"

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_data(ticker, period, interval):
    """Fetches data with rate limiting and error handling"""
    time.sleep(1.5) # Rate limiting
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        
        # Handling MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Timezone conversion
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        df = df.dropna(subset=['Close'])
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
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
        
        # Volatility (ATR-like) & Z-Score
        self.df['Log_Ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Volatility'] = self.df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) # Annualized
        self.df['Z_Score'] = zscore(self.df['Close'].fillna(0))
        
        # Bollinger Bands for Volatility Squeeze
        self.df['BB_Mid'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_Std'] = self.df['Close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Mid'] + (2 * self.df['BB_Std'])
        self.df['BB_Lower'] = self.df['BB_Mid'] - (2 * self.df['BB_Std'])

    def detect_rsi_divergence(self, window=5):
        """Detects Regular Bullish/Bearish Divergence"""
        # Simplistic local extrema detection
        self.df['min_local'] = self.df['Close'][argrelextrema(self.df['Close'].values, np.less_equal, order=window)[0]]
        self.df['max_local'] = self.df['Close'][argrelextrema(self.df['Close'].values, np.greater_equal, order=window)[0]]
        
        divergences = []
        # Logic: Compare last two local mins/maxs
        # (This is a simplified vector implementation for the demo)
        if len(self.df) < 50: return []
        
        current_rsi = self.df['RSI'].iloc[-1]
        current_price = self.df['Close'].iloc[-1]
        
        # Check last 30 bars for divergence setup
        subset = self.df.iloc[-30:]
        low_min = subset['Close'].min()
        rsi_at_low = subset.loc[subset['Close'] == low_min, 'RSI'].values[0] if not subset.loc[subset['Close'] == low_min].empty else 50
        
        if current_price < low_min and current_rsi > rsi_at_low:
             divergences.append("Possible Bullish Divergence (Lower Price, Higher RSI)")
             
        return divergences

    def get_support_resistance(self, n_bins=20):
        """Calculates Support/Resistance based on volume/price density"""
        data = self.df['Close']
        counts, bins = np.histogram(data, bins=n_bins)
        
        sorted_indices = np.argsort(counts)[::-1] # indices of highest density
        top_levels = []
        for i in sorted_indices[:3]: # Top 3 zones
            level = (bins[i] + bins[i+1]) / 2
            strength = counts[i]
            top_levels.append({'Level': level, 'Strength': strength})
        return top_levels

    def get_fibonacci_levels(self):
        """Calculates Fib levels based on recent high/low"""
        recent_period = 100 # lookback
        if len(self.df) < recent_period: recent_period = len(self.df)
        
        sub = self.df.iloc[-recent_period:]
        max_p = sub['High'].max()
        min_p = sub['Low'].min()
        diff = max_p - min_p
        
        levels = {
            '0.0': min_p,
            '0.236': min_p + 0.236 * diff,
            '0.382': min_p + 0.382 * diff,
            '0.5': min_p + 0.5 * diff,
            '0.618': min_p + 0.618 * diff,
            '1.0': max_p
        }
        return levels, max_p, min_p

    def elliott_wave_approx(self):
        """Rough approximation of wave count based on pivot points"""
        # Real EW is very subjective. This counts sequences of Higher Highs.
        closes = self.df['Close'].values
        pivots = argrelextrema(closes, np.greater, order=5)[0]
        
        if len(pivots) < 3: return "Indeterminate"
        
        last_3 = closes[pivots[-3:]]
        if last_3[0] < last_3[1] < last_3[2]:
            return "Potential Impulse Wave 3 or 5"
        elif last_3[1] > last_3[2]:
            return "Potential Correction Wave"
        return "Consolidation"

# ==========================================
# BACKTESTING ENGINE
# ==========================================
def backtest_strategy(df):
    """
    Strategy: Trend Following + Mean Reversion
    Buy: Price > 200 EMA AND RSI < 40 (Pullback in uptrend)
    Sell: Price < 200 EMA AND RSI > 60 (Rally in downtrend) or Trailing SL
    """
    df = df.copy()
    initial_capital = 100000
    position = 0
    entry_price = 0
    pnl = []
    log = []
    
    for i in range(200, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]
        
        # BUY SIGNAL
        if position == 0:
            if curr['Close'] > curr['EMA_200'] and curr['RSI'] < 40 and curr['Close'] > curr['EMA_50']:
                position = 1
                entry_price = curr['Close']
                log.append({'Date': date, 'Type': 'BUY', 'Price': entry_price, 'Reason': 'Pullback to Support'})
        
        # SELL / EXIT SIGNAL
        elif position == 1:
            # Take Profit (5%) or Stop Loss (2%)
            if curr['Close'] >= entry_price * 1.05:
                pnl.append((curr['Close'] - entry_price))
                log.append({'Date': date, 'Type': 'SELL (TP)', 'Price': curr['Close'], 'Reason': 'Target Hit'})
                position = 0
            elif curr['Close'] <= entry_price * 0.98:
                pnl.append((curr['Close'] - entry_price))
                log.append({'Date': date, 'Type': 'SELL (SL)', 'Price': curr['Close'], 'Reason': 'Stop Hit'})
                position = 0
                
    total_pnl = sum(pnl)
    accuracy = (len([x for x in pnl if x > 0]) / len(pnl)) * 100 if len(pnl) > 0 else 0
    
    return {
        'total_trades': len(pnl),
        'accuracy': accuracy,
        'total_points': total_pnl,
        'log': pd.DataFrame(log)
    }

# ==========================================
# MAIN APP UI
# ==========================================

st.sidebar.title("🛠️ Algo Setup")

# Input Section
ticker1_sym = st.sidebar.selectbox("Asset 1", list(TICKER_MAP.keys()), index=0)
enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Asset 2)")
ticker2_sym = None
if enable_ratio:
    ticker2_sym = st.sidebar.selectbox("Asset 2 (Ratio)", list(TICKER_MAP.keys()), index=1)

# Time Settings
col1, col2 = st.sidebar.columns(2)
with col1:
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)
with col2:
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=5)

# Action Button
if st.sidebar.button("🚀 Run Comprehensive Analysis"):
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching Data...")
    progress_bar.progress(10)
    
    t1_code = TICKER_MAP[ticker1_sym]
    if t1_code == "Custom": t1_code = st.sidebar.text_input("Enter Ticker")
    
    df1 = fetch_data(t1_code, period, timeframe)
    
    if df1 is None or len(df1) < 20:
        st.error("Insufficient data fetched. Try a longer period or different timeframe.")
        st.stop()
        
    status_text.text("Processing Ticker 1 Indicators...")
    progress_bar.progress(30)
    ana1 = MarketAnalyzer(df1)
    
    df2 = None
    ana2 = None
    ratio_df = None
    
    if enable_ratio and ticker2_sym:
        status_text.text("Processing Ticker 2 & Ratios...")
        t2_code = TICKER_MAP[ticker2_sym]
        df2 = fetch_data(t2_code, period, timeframe)
        if df2 is not None:
            ana2 = MarketAnalyzer(df2)
            # Align Indexes for Ratio
            common_idx = df1.index.intersection(df2.index)
            ratio_series = df1.loc[common_idx]['Close'] / df2.loc[common_idx]['Close']
            ratio_df = pd.DataFrame(ratio_series, columns=['Close'])
            ratio_ana = MarketAnalyzer(ratio_df)
    
    progress_bar.progress(60)
    
    # ==========================================
    # DISPLAY: TAB STRUCTURE
    # ==========================================
    
    tab_summary, tab_detailed, tab_backtest, tab_charts = st.tabs([
        "🧠 AI Signal & Summary", 
        "📊 Detailed Statistics", 
        "🧪 Backtest Engine",
        "📈 Interactive Charts"
    ])
    
    # ==========================================
    # TAB 1: AI SUMMARY & SIGNAL
    # ==========================================
    with tab_summary:
        st.subheader(f"Analysis Summary: {ticker1_sym} ({timeframe})")
        
        curr_price = df1['Close'].iloc[-1]
        prev_price = df1['Close'].iloc[-2]
        change = curr_price - prev_price
        pct_change = (change / prev_price) * 100
        
        # Determine Trend
        ema20 = df1['EMA_20'].iloc[-1]
        ema200 = df1['EMA_200'].iloc[-1]
        rsi = df1['RSI'].iloc[-1]
        trend = "BULLISH" if curr_price > ema200 else "BEARISH"
        micro_trend = "UP" if curr_price > ema20 else "DOWN"
        
        # Support/Resistance
        sr_levels = ana1.get_support_resistance()
        nearest_sr = min(sr_levels, key=lambda x: abs(x['Level'] - curr_price))
        dist_sr = curr_price - nearest_sr['Level']
        sr_type = "Support" if dist_sr > 0 else "Resistance"
        
        # Fibonacci
        fibs, fib_h, fib_l = ana1.get_fibonacci_levels()
        
        # Constructing the "100 words" professional summary
        summary_color = "green" if trend == "BULLISH" else "red"
        
        col_sum1, col_sum2 = st.columns([2, 1])
        
        with col_sum1:
            st.markdown(f"""
            ### 📢 Market Forecast
            The market for **{ticker1_sym}** is currently in a **:{summary_color}[{trend}]** phase on the {timeframe} timeframe. 
            Price is trading at **{curr_price:.2f}**, which is {abs(dist_sr):.2f} points away from the nearest strong {sr_type} zone ({nearest_sr['Level']:.2f}).
            
            **Key Insights:**
            * **Trend:** Price is {"above" if curr_price > ema200 else "below"} the 200 EMA and {"above" if curr_price > ema20 else "below"} the 20 EMA, indicating a {micro_trend} momentum.
            * **RSI Analysis:** RSI is at **{rsi:.1f}**. { "Oversold condition detected, expect bounce." if rsi < 30 else "Overbought, caution advised." if rsi > 70 else "Neutral territory."}
            * **Elliott Wave:** Structure suggests a **{ana1.elliott_wave_approx()}**.
            * **Fibonacci:** Key reaction level is at **{fibs['0.618']:.2f}** (0.618 Retracement).
            
            **Signal Verdict:**
            Based on historical Z-Score of **{df1['Z_Score'].iloc[-1]:.2f}** and volatility of **{df1['Volatility'].iloc[-1]:.2f}**, the probability favors a **{"LONG" if trend=="BULLISH" and rsi < 70 else "SHORT" if trend=="BEARISH" and rsi > 30 else "HOLD"}** setup. 
            Estimated move: **{df1['Volatility'].iloc[-1] * curr_price / 10:.2f} points** in the next period.
            """)
        
        with col_sum2:
            st.metric("Current Price", f"{curr_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
            st.metric("RSI (14)", f"{rsi:.1f}", delta=None)
            st.metric("Volatility (Ann)", f"{df1['Volatility'].iloc[-1]:.2f}", delta=None)
            
            # Target / SL logic
            target = curr_price * (1.02 if trend == "BULLISH" else 0.98)
            sl = curr_price * (0.99 if trend == "BULLISH" else 1.01)
            
            st.info(f"🎯 Target: {target:.2f}")
            st.error(f"🛑 Stop Loss: {sl:.2f}")

    # ==========================================
    # TAB 2: DETAILED STATS
    # ==========================================
    with tab_detailed:
        st.markdown("### 🧬 Deep Dive Analysis")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### Support & Resistance Density")
            sr_df = pd.DataFrame(sr_levels)
            st.dataframe(sr_df, hide_index=True, use_container_width=True)
            
        with c2:
            st.markdown("#### Fibonacci Levels")
            st.dataframe(pd.DataFrame.from_dict(fibs, orient='index', columns=['Price']), use_container_width=True)
            
        with c3:
            st.markdown("#### Z-Score Extremes (Last 5)")
            # Show times where Z-score was > 2 or < -2
            extremes = df1[(df1['Z_Score'] > 2) | (df1['Z_Score'] < -2)].tail(5)
            display_ext = extremes[['Close', 'Z_Score', 'RSI']].copy()
            display_ext['Time Ago'] = display_ext.index.map(format_time_ago)
            st.dataframe(display_ext[['Close', 'Z_Score', 'Time Ago']], use_container_width=True)

        if enable_ratio and ratio_df is not None:
            st.markdown("---")
            st.markdown(f"#### ⚖️ Ratio Analysis: {ticker1_sym} / {ticker2_sym}")
            st.line_chart(ratio_df['Close'])
            st.write(f"Current Ratio: {ratio_df['Close'].iloc[-1]:.4f}")

    # ==========================================
    # TAB 3: BACKTESTING
    # ==========================================
    with tab_backtest:
        st.subheader("Strategy Performance (Trend + Mean Reversion)")
        st.write("Backtesting on loaded data period...")
        
        bt_res = backtest_strategy(ana1.df)
        
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Total Trades", bt_res['total_trades'])
        bc2.metric("Accuracy", f"{bt_res['accuracy']:.1f}%")
        bc3.metric("Total Points", f"{bt_res['total_points']:.2f}")
        bc4.metric("Exp. Profit/Year", "N/A (Demo)")
        
        st.markdown("#### 📜 Trade Log")
        if not bt_res['log'].empty:
            log_df = bt_res['log']
            log_df['Date'] = log_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(log_df, use_container_width=True)
        else:
            st.warning("No trades generated with current strategy parameters.")

    # ==========================================
    # TAB 4: CHARTS
    # ==========================================
    with tab_charts:
        st.subheader("Interactive Technical Chart")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.6, 0.2, 0.2])

        # Candlestick
        fig.add_trace(go.Candlestick(x=df1.index,
                        open=df1['Open'], high=df1['High'],
                        low=df1['Low'], close=df1['Close'], name='Price'), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(x=df1.index, y=df1['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1.index, y=df1['EMA_200'], line=dict(color='blue', width=2), name='EMA 200'), row=1, col=1)
        
        # S/R Lines (Top 2)
        for level in sr_levels[:2]:
            fig.add_hline(y=level['Level'], line_dash="dash", line_color="white", row=1, col=1, annotation_text="Key Level")

        # RSI
        fig.add_trace(go.Scatter(x=df1.index, y=df1['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        # Volatility/Z-Score
        fig.add_trace(go.Bar(x=df1.index, y=df1['Z_Score'], name='Z-Score'), row=3, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    progress_bar.progress(100)
    status_text.text("Analysis Complete.")
