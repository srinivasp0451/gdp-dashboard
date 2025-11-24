import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {width: 100%; background-color: #1f77b4; color: white; border-radius: 5px; height: 50px; font-weight: bold;}
    .metric-card {background-color: #1e2130; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .positive {color: #00ff00; font-weight: bold;}
    .negative {color: #ff0000; font-weight: bold;}
    .neutral {color: #ffaa00; font-weight: bold;}
    h1, h2, h3 {color: #ffffff;}
    .dataframe {font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def get_ticker_symbol(asset_name):
    """Convert asset names to yfinance ticker symbols"""
    ticker_map = {
        'NIFTY 50': '^NSEI',
        'BANK NIFTY': '^NSEBANK',
        'SENSEX': '^BSESN',
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'USD/INR': 'INR=X',
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'JPY=X'
    }
    return ticker_map.get(asset_name, asset_name)

def fetch_data_with_delay(ticker, interval, period, delay=2):
    """Fetch data with rate limiting"""
    try:
        time.sleep(delay)
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
            
        # Reset index and flatten multi-index columns
        data = data.reset_index()
        
        # Flatten column names if multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
        
        # Rename columns to standard format
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'time' in col_lower:
                column_mapping[col] = 'Datetime'
            elif 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        data = data.rename(columns=column_mapping)
        
        # Keep only required columns
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in data.columns:
            required_cols.append('Volume')
        
        data = data[required_cols]
        
        # Convert to IST timezone
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI manually"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate SMA manually"""
    return data.rolling(window=period).mean()

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        'Level 0%': high,
        'Level 23.6%': high - (diff * 0.236),
        'Level 38.2%': high - (diff * 0.382),
        'Level 50%': high - (diff * 0.5),
        'Level 61.8%': high - (diff * 0.618),
        'Level 78.6%': high - (diff * 0.786),
        'Level 100%': low
    }

def find_support_resistance(data, window=20):
    """Find support and resistance levels"""
    highs = data['High'].rolling(window=window).max()
    lows = data['Low'].rolling(window=window).min()
    
    # Get unique levels
    resistance_levels = highs.dropna().unique()[-3:]
    support_levels = lows.dropna().unique()[:3]
    
    return sorted(support_levels), sorted(resistance_levels, reverse=True)

def calculate_volatility(data, window=20):
    """Calculate historical volatility"""
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility

def detect_divergence(price, rsi, window=14):
    """Detect RSI divergences"""
    divergences = []
    
    for i in range(window, len(price)-1):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price.iloc[i] < price.iloc[i-window] and 
            rsi.iloc[i] > rsi.iloc[i-window] and 
            rsi.iloc[i] < 40):
            divergences.append(('Bullish', i))
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (price.iloc[i] > price.iloc[i-window] and 
            rsi.iloc[i] < rsi.iloc[i-window] and 
            rsi.iloc[i] > 60):
            divergences.append(('Bearish', i))
    
    return divergences

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

# ==================== MAIN APPLICATION ====================

st.title("🚀 Professional Algorithmic Trading Dashboard")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset Selection
    asset_options = ['NIFTY 50', 'BANK NIFTY', 'SENSEX', 'BTC', 'ETH', 'GOLD', 
                     'SILVER', 'USD/INR', 'EUR/USD', 'GBP/USD', 'USD/JPY', 'Custom']
    
    ticker1_option = st.selectbox("Select Ticker 1", asset_options, key='ticker1_select')
    
    if ticker1_option == 'Custom':
        ticker1 = st.text_input("Enter Custom Ticker 1", "AAPL", key='ticker1_custom')
    else:
        ticker1 = get_ticker_symbol(ticker1_option)
    
    # Ratio Analysis Option
    enable_ratio = st.checkbox("Enable Ratio Analysis (Compare with Ticker 2)", key='enable_ratio')
    
    ticker2 = None
    if enable_ratio:
        ticker2_option = st.selectbox("Select Ticker 2", asset_options, key='ticker2_select')
        if ticker2_option == 'Custom':
            ticker2 = st.text_input("Enter Custom Ticker 2", "MSFT", key='ticker2_custom')
        else:
            ticker2 = get_ticker_symbol(ticker2_option)
    
    # Timeframe and Period
    interval = st.selectbox("Interval", 
                           ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d'],
                           index=4, key='interval')
    
    period = st.selectbox("Period",
                         ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', 
                          '6y', '10y', '15y', '20y', '25y', '30y'],
                         index=3, key='period')
    
    # API Rate Limiting
    api_delay = st.slider("API Delay (seconds)", 1.0, 5.0, 2.0, 0.5, key='api_delay')
    
    # Pattern Detection Threshold
    pattern_threshold = st.number_input("Pattern Detection Threshold (points)", 
                                       min_value=10, max_value=100, value=30, key='pattern_threshold')
    
    # Fetch Button
    fetch_button = st.button("🔄 Fetch Data & Analyze", type="primary", key='fetch_button')

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

# Main Analysis
if fetch_button:
    with st.spinner("Fetching and analyzing data..."):
        
        # Fetch primary ticker data
        data1 = fetch_data_with_delay(ticker1, interval, period, api_delay)
        
        if data1 is not None and not data1.empty:
            st.session_state.data1 = data1
            st.session_state.ticker1 = ticker1
            st.session_state.ticker1_option = ticker1_option
            
            # Calculate indicators for ticker 1
            data1['RSI'] = calculate_rsi(data1['Close'])
            data1['EMA_9'] = calculate_ema(data1['Close'], 9)
            data1['EMA_20'] = calculate_ema(data1['Close'], 20)
            data1['EMA_21'] = calculate_ema(data1['Close'], 21)
            data1['EMA_33'] = calculate_ema(data1['Close'], 33)
            data1['EMA_50'] = calculate_ema(data1['Close'], 50)
            data1['EMA_100'] = calculate_ema(data1['Close'], 100)
            data1['EMA_150'] = calculate_ema(data1['Close'], 150)
            data1['EMA_200'] = calculate_ema(data1['Close'], 200)
            data1['SMA_20'] = calculate_sma(data1['Close'], 20)
            data1['SMA_50'] = calculate_sma(data1['Close'], 50)
            data1['SMA_100'] = calculate_sma(data1['Close'], 100)
            data1['SMA_150'] = calculate_sma(data1['Close'], 150)
            data1['SMA_200'] = calculate_sma(data1['Close'], 200)
            data1['Volatility'] = calculate_volatility(data1)
            data1['ATR'] = calculate_atr(data1)
            data1['Returns'] = data1['Close'].pct_change() * 100
            
            # Fetch ticker 2 if ratio analysis enabled
            if enable_ratio and ticker2:
                data2 = fetch_data_with_delay(ticker2, interval, period, api_delay)
                if data2 is not None and not data2.empty:
                    st.session_state.data2 = data2
                    st.session_state.ticker2 = ticker2
                    st.session_state.ticker2_option = ticker2_option
                    
                    # Calculate indicators for ticker 2
                    data2['RSI'] = calculate_rsi(data2['Close'])
                    data2['EMA_9'] = calculate_ema(data2['Close'], 9)
                    data2['EMA_20'] = calculate_ema(data2['Close'], 20)
                    data2['EMA_21'] = calculate_ema(data2['Close'], 21)
                    data2['EMA_33'] = calculate_ema(data2['Close'], 33)
                    data2['EMA_50'] = calculate_ema(data2['Close'], 50)
                    data2['EMA_100'] = calculate_ema(data2['Close'], 100)
                    data2['EMA_150'] = calculate_ema(data2['Close'], 150)
                    data2['EMA_200'] = calculate_ema(data2['Close'], 200)
                    data2['SMA_20'] = calculate_sma(data2['Close'], 20)
                    data2['SMA_50'] = calculate_sma(data2['Close'], 50)
                    data2['SMA_100'] = calculate_sma(data2['Close'], 100)
                    data2['SMA_150'] = calculate_sma(data2['Close'], 150)
                    data2['SMA_200'] = calculate_sma(data2['Close'], 200)
                    data2['Volatility'] = calculate_volatility(data2)
                    data2['ATR'] = calculate_atr(data2)
                    data2['Returns'] = data2['Close'].pct_change() * 100
            
            st.session_state.data_fetched = True
            st.session_state.enable_ratio = enable_ratio
            st.success("✅ Data fetched successfully!")
        else:
            st.error("Failed to fetch data. Please check ticker symbols and try again.")

# Display analysis if data is fetched
if st.session_state.data_fetched:
    data1 = st.session_state.data1
    ticker1 = st.session_state.ticker1
    ticker1_option = st.session_state.ticker1_option
    
    # ==================== SECTION 1: MARKET OVERVIEW ====================
    st.header("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price1 = data1['Close'].iloc[-1]
    prev_price1 = data1['Close'].iloc[0]
    change1 = current_price1 - prev_price1
    pct_change1 = (change1 / prev_price1) * 100
    
    with col1:
        change_color1 = "positive" if change1 > 0 else "negative"
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{ticker1_option}</h3>
            <h2>₹{current_price1:.2f}</h2>
            <p class='{change_color1}'>
                {'+' if change1 > 0 else ''}{change1:.2f} ({pct_change1:+.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.enable_ratio and 'data2' in st.session_state:
        data2 = st.session_state.data2
        ticker2 = st.session_state.ticker2
        ticker2_option = st.session_state.ticker2_option
        
        current_price2 = data2['Close'].iloc[-1]
        prev_price2 = data2['Close'].iloc[0]
        change2 = current_price2 - prev_price2
        pct_change2 = (change2 / prev_price2) * 100
        
        with col2:
            change_color2 = "positive" if change2 > 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{ticker2_option}</h3>
                <h2>₹{current_price2:.2f}</h2>
                <p class='{change_color2}'>
                    {'+' if change2 > 0 else ''}{change2:.2f} ({pct_change2:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate ratio
        ratio = current_price1 / current_price2
        prev_ratio = prev_price1 / prev_price2
        ratio_change = ratio - prev_ratio
        ratio_pct_change = (ratio_change / prev_ratio) * 100
        
        with col3:
            ratio_color = "positive" if ratio_change > 0 else "negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Ratio (T1/T2)</h3>
                <h2>{ratio:.4f}</h2>
                <p class='{ratio_color}'>
                    {'+' if ratio_change > 0 else ''}{ratio_change:.4f} ({ratio_pct_change:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        rsi_current = data1['RSI'].iloc[-1]
        rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
        rsi_color = "negative" if rsi_current > 70 else "positive" if rsi_current < 30 else "neutral"
        
        st.markdown(f"""
        <div class='metric-card'>
            <h3>RSI (14)</h3>
            <h2>{rsi_current:.2f}</h2>
            <p class='{rsi_color}'>{rsi_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Table
    st.subheader("📈 Complete Data Table")
    display_data = data1[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'RSI']].copy()
    display_data['Datetime'] = display_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    st.dataframe(display_data.tail(50), use_container_width=True)
    
    # Export functionality
    csv = display_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv,
        file_name=f"{ticker1_option}_data.csv",
        mime="text/csv"
    )
    
    # ==================== SECTION 2: RATIO ANALYSIS ====================
    if st.session_state.enable_ratio and 'data2' in st.session_state:
        st.header("⚖️ Ratio Analysis")
        
        # Align data by datetime
        merged_data = pd.merge(
            data1[['Datetime', 'Close', 'RSI']],
            data2[['Datetime', 'Close', 'RSI']],
            on='Datetime',
            suffixes=('_T1', '_T2')
        )
        
        merged_data['Ratio'] = merged_data['Close_T1'] / merged_data['Close_T2']
        merged_data['Ratio_RSI'] = calculate_rsi(merged_data['Ratio'])
        
        # Display ratio table
        st.subheader("Ratio Comparison Table")
        ratio_display = merged_data.copy()
        ratio_display['Datetime'] = ratio_display['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
        ratio_display = ratio_display.rename(columns={
            'Close_T1': f'{ticker1_option} Price',
            'Close_T2': f'{ticker2_option} Price',
            'RSI_T1': f'{ticker1_option} RSI',
            'RSI_T2': f'{ticker2_option} RSI'
        })
        
        st.dataframe(ratio_display.tail(50), use_container_width=True)
        
        # Export ratio data
        ratio_csv = ratio_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Ratio Data (CSV)",
            data=ratio_csv,
            file_name=f"ratio_{ticker1_option}_{ticker2_option}.csv",
            mime="text/csv"
        )
        
        # Ratio Binning Analysis
        st.subheader("📊 Ratio Binning Analysis")
        
        ratio_values = merged_data['Ratio'].dropna()
        bins = pd.qcut(ratio_values, q=5, duplicates='drop')
        merged_data['Ratio_Bin'] = pd.cut(merged_data['Ratio'], bins=bins.cat.categories)
        
        bin_analysis = []
        for bin_label in merged_data['Ratio_Bin'].dropna().unique():
            bin_data = merged_data[merged_data['Ratio_Bin'] == bin_label]
            
            t1_returns_points = bin_data['Close_T1'].mean() - bin_data['Close_T1'].iloc[0]
            t1_returns_pct = (t1_returns_points / bin_data['Close_T1'].iloc[0]) * 100
            
            t2_returns_points = bin_data['Close_T2'].mean() - bin_data['Close_T2'].iloc[0]
            t2_returns_pct = (t2_returns_points / bin_data['Close_T2'].iloc[0]) * 100
            
            bin_analysis.append({
                'Ratio Bin': f"{bin_label.left:.4f} - {bin_label.right:.4f}",
                f'{ticker1_option} Returns (Points)': f"{t1_returns_points:+.2f}",
                f'{ticker1_option} Returns (%)': f"{t1_returns_pct:+.2f}%",
                f'{ticker2_option} Returns (Points)': f"{t2_returns_points:+.2f}",
                f'{ticker2_option} Returns (%)': f"{t2_returns_pct:+.2f}%",
                'Occurrences': len(bin_data)
            })
        
        bin_df = pd.DataFrame(bin_analysis)
        st.dataframe(bin_df, use_container_width=True)
        
        # Bin Summary
        current_ratio_bin = merged_data['Ratio_Bin'].iloc[-1]
        st.info(f"""
        **📌 Ratio Binning Insights:**
        
        - Current ratio: **{ratio:.4f}**
        - Current bin: **{current_ratio_bin.left:.4f} - {current_ratio_bin.right:.4f}**
        - Historically, in this bin:
          - {ticker1_option} has shown {'positive' if t1_returns_pct > 0 else 'negative'} returns
          - {ticker2_option} has shown {'positive' if t2_returns_pct > 0 else 'negative'} returns
        - **Forecast**: Based on historical behavior in this ratio range, expect {'continuation' if t1_returns_pct * pct_change1 > 0 else 'reversal'} in current trend
        """)
    
    # ==================== SECTION 3: MULTI-TIMEFRAME ANALYSIS ====================
    st.header("🔍 Multi-Timeframe Analysis")
    
    timeframe_configs = [
        ('1m', '1d'), ('5m', '5d'), ('15m', '5d'), ('30m', '1mo'),
        ('1h', '1mo'), ('2h', '3mo'), ('4h', '6mo'), ('1d', '1y'),
        ('1wk', '5y'), ('1mo', '10y')
    ]
    
    def analyze_timeframe(ticker, tf_interval, tf_period):
        """Analyze single timeframe"""
        tf_data = fetch_data_with_delay(ticker, tf_interval, tf_period, api_delay)
        if tf_data is None or tf_data.empty:
            return None
        
        # Calculate indicators
        tf_data['RSI'] = calculate_rsi(tf_data['Close'])
        tf_data['EMA_9'] = calculate_ema(tf_data['Close'], 9)
        tf_data['EMA_20'] = calculate_ema(tf_data['Close'], 20)
        tf_data['EMA_21'] = calculate_ema(tf_data['Close'], 21)
        tf_data['EMA_33'] = calculate_ema(tf_data['Close'], 33)
        tf_data['EMA_50'] = calculate_ema(tf_data['Close'], 50)
        tf_data['EMA_100'] = calculate_ema(tf_data['Close'], 100)
        tf_data['EMA_150'] = calculate_ema(tf_data['Close'], 150)
        tf_data['EMA_200'] = calculate_ema(tf_data['Close'], 200)
        tf_data['SMA_20'] = calculate_sma(tf_data['Close'], 20)
        tf_data['SMA_50'] = calculate_sma(tf_data['Close'], 50)
        tf_data['SMA_100'] = calculate_sma(tf_data['Close'], 100)
        tf_data['SMA_150'] = calculate_sma(tf_data['Close'], 150)
        tf_data['SMA_200'] = calculate_sma(tf_data['Close'], 200)
        tf_data['Volatility'] = calculate_volatility(tf_data)
        
        current_price = tf_data['Close'].iloc[-1]
        max_close = tf_data['Close'].max()
        min_close = tf_data['Close'].min()
        
        # Fibonacci levels
        fib_levels = calculate_fibonacci_levels(max_close, min_close)
        fib_str = ', '.join([f"{k}: {v:.2f}" for k, v in fib_levels.items()])
        
        # Support and Resistance
        support, resistance = find_support_resistance(tf_data)
        support_str = ', '.join([f"{s:.2f}" for s in support])
        resistance_str = ', '.join([f"{r:.2f}" for r in resistance])
        
        # Trend determination
        trend = "Up" if current_price > tf_data['Close'].iloc[0] else "Down"
        
        # RSI status
        rsi_val = tf_data['RSI'].iloc[-1]
        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        
        # Price change
        price_change = current_price - tf_data['Close'].iloc[0]
        pct_change = (price_change / tf_data['Close'].iloc[0]) * 100
        
        # EMA positions
        ema_positions = {
            'EMA_9': 'Above' if current_price > tf_data['EMA_9'].iloc[-1] else 'Below',
            'EMA_20': 'Above' if current_price > tf_data['EMA_20'].iloc[-1] else 'Below',
            'EMA_21': 'Above' if current_price > tf_data['EMA_21'].iloc[-1] else 'Below',
            'EMA_33': 'Above' if current_price > tf_data['EMA_33'].iloc[-1] else 'Below',
            'EMA_50': 'Above' if current_price > tf_data['EMA_50'].iloc[-1] else 'Below',
            'EMA_100': 'Above' if current_price > tf_data['EMA_100'].iloc[-1] else 'Below',
            'EMA_150': 'Above' if current_price > tf_data['EMA_150'].iloc[-1] else 'Below',
            'EMA_200': 'Above' if current_price > tf_data['EMA_200'].iloc[-1] else 'Below',
        }
        
        # SMA positions
        sma_positions = {
            'SMA_20': 'Above' if current_price > tf_data['SMA_20'].iloc[-1] else 'Below',
            'SMA_50': 'Above' if current_price > tf_data['SMA_50'].iloc[-1] else 'Below',
            'SMA_100': 'Above' if current_price > tf_data['SMA_100'].iloc[-1] else 'Below',
            'SMA_150': 'Above' if current_price > tf_data['SMA_150'].iloc[-1] else 'Below',
            'SMA_200': 'Above' if current_price > tf_data['SMA_200'].iloc[-1] else 'Below',
        }
        
        return {
            'Timeframe': f"{tf_interval}/{tf_period}",
            'Trend': trend,
            'Max Close': f"{max_close:.2f}",
            'Min Close': f"{min_close:.2f}",
            'Fibonacci': fib_str,
            'Volatility': f"{tf_data['Volatility'].iloc[-1]:.2f}%",
            'Change %': f"{pct_change:+.2f}%",
            'Change Points': f"{price_change:+.2f}",
            'Support': support_str,
            'Resistance': resistance_str,
            'RSI': f"{rsi_val:.2f}",
            'RSI Status': rsi_status,
            **{f'EMA_{k.split("_")[1]}': f"{tf_data[k].iloc[-1]:.2f}" for k in ema_positions.keys()},
            **{f'{k} Pos': ema_positions[k] for k in ema_positions.keys()},
            **{f'SMA_{k.split("_")[1]}': f"{tf_data[k].iloc[-1]:.2f}" for k in sma_positions.keys()},
            **{f'{k} Pos': sma_positions[k] for k in sma_positions.keys()},
        }
    
    # Analyze Ticker 1
    st.subheader(f"📊 {ticker1_option} Multi-Timeframe Analysis")
    
    mtf_results_t1 = []
    progress_bar = st.progress(0)
    
    for idx, (tf_int, tf_per) in enumerate(timeframe_configs):
        result = analyze_timeframe(ticker1, tf_int, tf_per)
        if result:
            mtf_results_t1.append(result)
        progress_bar.progress((idx + 1) / len(timeframe_configs))
    
    progress_bar.empty()
    
    if mtf_results_t1:
        mtf_df_t1 = pd.DataFrame(mtf_results_t1)
        st.dataframe(mtf_df_t1, use_container_width=True)
        
        # Multi-timeframe summary for T1
        up_trends = sum(1 for r in mtf_results_t1 if r['Trend'] == 'Up')
        down_trends = len(mtf_results_t1) - up_trends
        overall_trend = "Bullish" if up_trends > down_trends else "Bearish"
        
        avg_change = np.mean([float(r['Change %'].strip('%+')) for r in mtf_results_t1])
        
        st.success(f"""
        **📊 {ticker1_option} Multi-Timeframe Summary:**
        
        - **Overall Trend**: {overall_trend} ({up_trends} timeframes up, {down_trends} down)
        - **Average Change**: {avg_change:+.2f}%
        - **Price Action**: Current price is showing {'strength' if avg_change > 0 else 'weakness'} across multiple timeframes
        - **Key Insight**: The majority of timeframes suggest a {overall_trend.lower()} bias
        - **Recommendation**: {'Consider long positions' if overall_trend == 'Bullish' else 'Consider short positions or wait for reversal signals'}
        """)
    
    # Analyze Ticker 2 if ratio enabled
    if st.session_state.enable_ratio and 'data2' in st.session_state:
        st.subheader(f"📊 {ticker2_option} Multi-Timeframe Analysis")
        
        mtf_results_t2 = []
        progress_bar2 = st.progress(0)
        
        for idx, (tf_int, tf_per) in enumerate(timeframe_configs):
            result = analyze_timeframe(ticker2, tf_int, tf_per)
            if result:
                mtf_results_t2.append(result)
            progress_bar2.progress((idx + 1) / len(timeframe_configs))
        
        progress_bar2.empty()
        
        if mtf_results_t2:
            mtf_df_t2 = pd.DataFrame(mtf_results_t2)
            st.dataframe(mtf_df_t2, use_container_width=True)
            
            # Multi-timeframe summary for T2
            up_trends_t2 = sum(1 for r in mtf_results_t2 if r['Trend'] == 'Up')
            down_trends_t2 = len(mtf_results_t2) - up_trends_t2
            overall_trend_t2 = "Bullish" if up_trends_t2 > down_trends_t2 else "Bearish"
            
            avg_change_t2 = np.mean([float(r['Change %'].strip('%+')) for r in mtf_results_t2])
            
            st.success(f"""
            **📊 {ticker2_option} Multi-Timeframe Summary:**
            
            - **Overall Trend**: {overall_trend_t2} ({up_trends_t2} timeframes up, {down_trends_t2} down)
            - **Average Change**: {avg_change_t2:+.2f}%
            - **Price Action**: Current price is showing {'strength' if avg_change_t2 > 0 else 'weakness'} across multiple timeframes
            - **Key Insight**: The majority of timeframes suggest a {overall_trend_t2.lower()} bias
            - **Recommendation**: {'Consider long positions' if overall_trend_t2 == 'Bullish' else 'Consider short positions or wait for reversal signals'}
            """)
    
    # ==================== SECTION 4: VOLATILITY BINS ANALYSIS ====================
    st.header("📈 Volatility Bins Analysis")
    
    vol_data = data1[['Datetime', 'Close', 'Volatility', 'Returns']].dropna()
    
    # Create volatility bins
    vol_bins = pd.qcut(vol_data['Volatility'], q=5, duplicates='drop')
    vol_data['Vol_Bin'] = pd.cut(vol_data['Volatility'], bins=vol_bins.cat.categories)
    
    # Create volatility table
    vol_table = vol_data.copy()
    vol_table['Datetime'] = vol_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    vol_table['Vol_Bin_Label'] = vol_table['Vol_Bin'].apply(
        lambda x: f"{x.left:.2f} - {x.right:.2f}" if pd.notna(x) else "N/A"
    )
    vol_table['Returns_Points'] = vol_data['Close'].diff()
    
    display_vol = vol_table[['Datetime', 'Vol_Bin_Label', 'Volatility', 'Close', 'Returns_Points', 'Returns']].copy()
    display_vol.columns = ['Datetime', 'Volatility Bin', 'Volatility %', 'Price', 'Returns (Points)', 'Returns (%)']
    
    st.dataframe(display_vol.tail(50), use_container_width=True)
    
    # Volatility statistics
    max_vol = vol_data['Volatility'].max()
    min_vol = vol_data['Volatility'].min()
    mean_vol = vol_data['Volatility'].mean()
    current_vol = vol_data['Volatility'].iloc[-1]
    current_vol_bin = vol_data['Vol_Bin'].iloc[-1]
    
    max_return = vol_data['Returns'].max()
    min_return = vol_data['Returns'].min()
    
    # Returns by volatility bin
    bin_returns = vol_data.groupby('Vol_Bin')['Returns'].agg(['mean', 'std', 'count'])
    
    st.info(f"""
    **📊 Volatility Analysis Summary:**
    
    - **Current Volatility**: {current_vol:.2f}% (Bin: {current_vol_bin.left:.2f} - {current_vol_bin.right:.2f})
    - **Volatility Range**: {min_vol:.2f}% to {max_vol:.2f}% (Mean: {mean_vol:.2f}%)
    - **Returns Range**: {min_return:.2f}% to {max_return:.2f}%
    - **Historical Behavior in Current Bin**: Average return of {bin_returns.loc[current_vol_bin, 'mean']:.2f}% based on {int(bin_returns.loc[current_vol_bin, 'count'])} observations
    - **Forecast**: {'Higher volatility typically precedes larger moves' if current_vol > mean_vol else 'Lower volatility suggests consolidation phase'}
    - **What This Means**: Volatility measures how much the price moves. High volatility means bigger price swings (more risk and opportunity), while low volatility means smaller, steadier moves. Currently, the market is in a {'high' if current_vol > mean_vol else 'low'} volatility phase.
    """)
    
    # ==================== SECTION 5: PATTERN RECOGNITION ====================
    st.header("🔍 Advanced Pattern Recognition")
    
    # Detect significant moves
    data1['Price_Change'] = data1['Close'].diff().abs()
    significant_moves = data1[data1['Price_Change'] > pattern_threshold].copy()
    
    if len(significant_moves) > 0:
        pattern_results = []
        
        for idx in significant_moves.index:
            if idx < 10:
                continue
            
            move_size = data1.loc[idx, 'Close'] - data1.loc[idx-1, 'Close']
            move_pct = (move_size / data1.loc[idx-1, 'Close']) * 100
            direction = "Up" if move_size > 0 else "Down"
            
            # Analyze preceding 10 candles
            preceding = data1.loc[idx-10:idx-1]
            
            # Check patterns
            vol_spike = preceding['Volatility'].max() > preceding['Volatility'].mean() * 1.5
            rsi_before = preceding['RSI'].iloc[-1]
            rsi_at_move = data1.loc[idx, 'RSI']
            rsi_divergence = "Yes" if abs(rsi_at_move - rsi_before) > 10 else "No"
            
            # EMA crossovers
            ema_20_50_cross = "Yes" if (
                (preceding['EMA_20'].iloc[-2] < preceding['EMA_50'].iloc[-2] and 
                 preceding['EMA_20'].iloc[-1] > preceding['EMA_50'].iloc[-1]) or
                (preceding['EMA_20'].iloc[-2] > preceding['EMA_50'].iloc[-2] and 
                 preceding['EMA_20'].iloc[-1] < preceding['EMA_50'].iloc[-1])
            ) else "No"
            
            # Consecutive moves
            consecutive_up = sum(1 for i in range(len(preceding)-1) if preceding['Close'].iloc[i+1] > preceding['Close'].iloc[i])
            consecutive_pattern = "Yes" if consecutive_up >= 7 or consecutive_up <= 3 else "No"
            
            # Large body candles
            body_sizes = abs(preceding['Close'] - preceding['Open'])
            avg_body = body_sizes.mean()
            large_body = "Yes" if body_sizes.iloc[-1] > avg_body * 1.5 else "No"
            
            # Support/Resistance breakout
            support_levels, resistance_levels = find_support_resistance(preceding)
            breakout = "Yes"
            if direction == "Up" and len(resistance_levels) > 0:
                breakout = "Yes" if data1.loc[idx, 'Close'] > resistance_levels[0] else "No"
            elif direction == "Down" and len(support_levels) > 0:
                breakout = "Yes" if data1.loc[idx, 'Close'] < support_levels[-1] else "No"
            
            # Correlation with prior moves
            if len(significant_moves) > 1:
                prior_moves = significant_moves.loc[:idx-1]
                if len(prior_moves) > 0:
                    correlation = np.corrcoef(
                        prior_moves['Price_Change'].values,
                        [move_size] * len(prior_moves)
                    )[0, 1] if len(prior_moves) > 1 else 0
                else:
                    correlation = 0
            else:
                correlation = 0
            
            pattern_results.append({
                'Datetime': data1.loc[idx, 'Datetime'].strftime('%Y-%m-%d %H:%M:%S IST'),
                'Move (Points)': f"{move_size:.2f}",
                'Move (%)': f"{move_pct:.2f}%",
                'Direction': direction,
                'Volatility Burst': "Yes" if vol_spike else "No",
                'RSI Divergence': rsi_divergence,
                'RSI Before': f"{rsi_before:.2f}",
                'RSI At Move': f"{rsi_at_move:.2f}",
                'EMA 20/50 Cross': ema_20_50_cross,
                'Consecutive Pattern': consecutive_pattern,
                'Large Body Candle': large_body,
                'Support/Resistance Breakout': breakout,
                'Correlation': f"{correlation:.2f}",
            })
        
        if pattern_results:
            pattern_df = pd.DataFrame(pattern_results)
            st.dataframe(pattern_df, use_container_width=True)
            
            # Pattern summary
            total_patterns = len(pattern_results)
            vol_burst_count = sum(1 for p in pattern_results if p['Volatility Burst'] == 'Yes')
            rsi_div_count = sum(1 for p in pattern_results if p['RSI Divergence'] == 'Yes')
            breakout_count = sum(1 for p in pattern_results if p['Support/Resistance Breakout'] == 'Yes')
            
            # Current market similarity
            current_rsi = data1['RSI'].iloc[-1]
            recent_vol = data1['Volatility'].iloc[-5:].mean()
            
            warning_signals = []
            if vol_burst_count / total_patterns > 0.5:
                warning_signals.append("⚠️ High frequency of volatility bursts detected")
            if rsi_div_count / total_patterns > 0.3:
                warning_signals.append("⚠️ Significant RSI divergences present")
            if current_rsi > 70:
                warning_signals.append("⚠️ Current RSI is overbought")
            elif current_rsi < 30:
                warning_signals.append("⚠️ Current RSI is oversold")
            
            warning_text = "\n".join(warning_signals) if warning_signals else "✅ No major warning signals detected"
            
            st.warning(f"""
            **🔍 Pattern Recognition Summary:**
            
            - **Total Significant Moves Detected**: {total_patterns}
            - **Volatility Bursts**: {vol_burst_count} ({vol_burst_count/total_patterns*100:.1f}%)
            - **RSI Divergences**: {rsi_div_count} ({rsi_div_count/total_patterns*100:.1f}%)
            - **Breakouts**: {breakout_count} ({breakout_count/total_patterns*100:.1f}%)
            - **Current Market Similarity**: {'High' if recent_vol > data1['Volatility'].mean() else 'Low'} volatility environment
            
            {warning_text}
            
            **What This Means for You**: 
            We analyzed all major price movements (over {pattern_threshold} points) and looked at what happened right before each move. This helps us understand if similar conditions exist now. Think of it like studying past weather patterns to predict tomorrow's weather. The patterns show us that {'volatility spikes often precede big moves' if vol_burst_count > total_patterns/2 else 'moves happen across different conditions'}.
            
            **Forecast Based on Patterns**: {'Expect increased volatility and potential reversal' if len(warning_signals) > 1 else 'Market showing stable patterns, follow the trend'}
            """)
    else:
        st.info("No significant price movements detected in the selected period.")
    
    # ==================== SECTION 6: INTERACTIVE CHARTS ====================
    st.header("📊 Interactive Charts")
    
    # Chart 1: Ticker 1 Price + RSI
    st.subheader(f"{ticker1_option} - Price & RSI")
    
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker1_option} Price with EMAs', 'RSI Indicator')
    )
    
    # Candlestick
    fig1.add_trace(
        go.Candlestick(
            x=data1['Datetime'],
            open=data1['Open'],
            high=data1['High'],
            low=data1['Low'],
            close=data1['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig1.add_trace(go.Scatter(x=data1['Datetime'], y=data1['EMA_20'], name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=data1['Datetime'], y=data1['EMA_50'], name='EMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=data1['Datetime'], y=data1['EMA_200'], name='EMA 200', line=dict(color='purple', width=2)), row=1, col=1)
    
    # RSI
    fig1.add_trace(
        go.Scatter(x=data1['Datetime'], y=data1['RSI'], name='RSI', line=dict(color='cyan', width=2)),
        row=2, col=1
    )
    fig1.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig1.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig1.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2 & 3: Ratio analysis if enabled
    if st.session_state.enable_ratio and 'data2' in st.session_state:
        st.subheader(f"{ticker2_option} - Price & RSI")
        
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker2_option} Price with EMAs', 'RSI Indicator')
        )
        
        fig2.add_trace(
            go.Candlestick(
                x=data2['Datetime'],
                open=data2['Open'],
                high=data2['High'],
                low=data2['Low'],
                close=data2['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        fig2.add_trace(go.Scatter(x=data2['Datetime'], y=data2['EMA_20'], name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=data2['Datetime'], y=data2['EMA_50'], name='EMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=data2['Datetime'], y=data2['EMA_200'], name='EMA 200', line=dict(color='purple', width=2)), row=1, col=1)
        
        fig2.add_trace(
            go.Scatter(x=data2['Datetime'], y=data2['RSI'], name='RSI', line=dict(color='cyan', width=2)),
            row=2, col=1
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig2.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Ratio chart
        st.subheader("Ratio Analysis - Ratio & RSI")
        
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Ratio (T1/T2)', 'Ratio RSI')
        )
        
        fig3.add_trace(
            go.Scatter(x=merged_data['Datetime'], y=merged_data['Ratio'], name='Ratio', 
                      line=dict(color='yellow', width=2), fill='tozeroy'),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Scatter(x=merged_data['Datetime'], y=merged_data['Ratio_RSI'], name='Ratio RSI',
                      line=dict(color='magenta', width=2)),
            row=2, col=1
        )
        fig3.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig3.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig3.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    # ==================== SECTION 7: STATISTICAL DISTRIBUTION ====================
    st.header("📉 Statistical Distribution Analysis")
    
    returns_data = data1['Returns'].dropna()
    
    # Histogram of returns
    st.subheader("Returns Distribution")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=returns_data, nbinsx=50, name='Returns', marker_color='lightblue'))
    fig_hist.update_layout(title='Returns Distribution', xaxis_title='Returns (%)', yaxis_title='Frequency', height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Returns with normal curve
    st.subheader("Returns Distribution with Normal Curve")
    
    mean_return = returns_data.mean()
    std_return = returns_data.std()
    
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Histogram(x=returns_data, nbinsx=50, name='Actual Returns', 
                                    histnorm='probability density', marker_color='lightblue'))
    
    # Normal curve
    x_range = np.linspace(returns_data.min(), returns_data.max(), 100)
    normal_curve = stats.norm.pdf(x_range, mean_return, std_return)
    fig_norm.add_trace(go.Scatter(x=x_range, y=normal_curve, name='Normal Distribution',
                                  line=dict(color='red', width=2)))
    
    fig_norm.update_layout(title='Returns vs Normal Distribution', 
                          xaxis_title='Returns (%)', yaxis_title='Density', height=400)
    st.plotly_chart(fig_norm, use_container_width=True)
    
    # Bell curve with zones
    st.subheader("Bell Curve Visualization")
    
    fig_bell = go.Figure()
    
    x_bell = np.linspace(mean_return - 4*std_return, mean_return + 4*std_return, 1000)
    y_bell = stats.norm.pdf(x_bell, mean_return, std_return)
    
    # Green zone (±1 std)
    mask_green = (x_bell >= mean_return - std_return) & (x_bell <= mean_return + std_return)
    fig_bell.add_trace(go.Scatter(x=x_bell[mask_green], y=y_bell[mask_green], 
                                  fill='tozeroy', fillcolor='rgba(0,255,0,0.3)', 
                                  line=dict(color='green'), name='±1σ (68%)'))
    
    # Yellow zone (±2 std)
    mask_yellow_left = (x_bell >= mean_return - 2*std_return) & (x_bell < mean_return - std_return)
    mask_yellow_right = (x_bell > mean_return + std_return) & (x_bell <= mean_return + 2*std_return)
    fig_bell.add_trace(go.Scatter(x=x_bell[mask_yellow_left], y=y_bell[mask_yellow_left],
                                  fill='tozeroy', fillcolor='rgba(255,255,0,0.3)',
                                  line=dict(color='yellow'), name='±2σ (95%)'))
    fig_bell.add_trace(go.Scatter(x=x_bell[mask_yellow_right], y=y_bell[mask_yellow_right],
                                  fill='tozeroy', fillcolor='rgba(255,255,0,0.3)',
                                  line=dict(color='yellow'), showlegend=False))
    
    # Red zone (beyond ±2 std)
    mask_red_left = x_bell < mean_return - 2*std_return
    mask_red_right = x_bell > mean_return + 2*std_return
    fig_bell.add_trace(go.Scatter(x=x_bell[mask_red_left], y=y_bell[mask_red_left],
                                  fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                                  line=dict(color='red'), name='Beyond ±2σ (Extreme)'))
    fig_bell.add_trace(go.Scatter(x=x_bell[mask_red_right], y=y_bell[mask_red_right],
                                  fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                                  line=dict(color='red'), showlegend=False))
    
    # Current position
    current_return = returns_data.iloc[-1]
    fig_bell.add_vline(x=current_return, line_dash="dash", line_color="white", line_width=3,
                      annotation_text=f"Current: {current_return:.2f}%")
    
    fig_bell.update_layout(title='Bell Curve with Standard Deviation Zones',
                          xaxis_title='Returns (%)', yaxis_title='Probability Density', height=500)
    st.plotly_chart(fig_bell, use_container_width=True)
    
    # Z-Score Analysis
    st.subheader("Z-Score Analysis")
    
    data1['Z_Score'] = (data1['Returns'] - mean_return) / std_return
    
    z_score_table = data1[['Datetime', 'Returns', 'Z_Score']].dropna().tail(50).copy()
    z_score_table['Datetime'] = z_score_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    z_score_table['Returns_Points'] = data1['Close'].diff()
    z_score_table['Returns_%'] = data1['Returns']
    
    # Color code extreme z-scores
    def color_zscore(val):
        try:
            v = float(val)
            if abs(v) > 2:
                return 'background-color: #ff4444'
            elif abs(v) > 1:
                return 'background-color: #ffaa44'
            else:
                return 'background-color: #44ff44'
        except:
            return ''
    
    styled_z = z_score_table.style.applymap(color_zscore, subset=['Z_Score'])
    st.dataframe(styled_z, use_container_width=True)
    
    # Statistical summary
    current_zscore = data1['Z_Score'].iloc[-1]
    skewness = returns_data.skew()
    kurtosis_val = returns_data.kurtosis()
    percentile = stats.percentileofscore(returns_data, current_return)
    
    st.info(f"""
    **📊 Statistical Analysis Summary:**
    
    - **Mean Return**: {mean_return:.4f}%
    - **Standard Deviation**: {std_return:.4f}%
    - **Skewness**: {skewness:.4f} ({'Right-skewed (more positive outliers)' if skewness > 0 else 'Left-skewed (more negative outliers)'})
    - **Kurtosis**: {kurtosis_val:.4f} ({'Fat tails (more extreme events)' if kurtosis_val > 0 else 'Thin tails (fewer extreme events)'})
    - **Current Return**: {current_return:.4f}%
    - **Current Z-Score**: {current_zscore:.4f}
    - **Percentile Rank**: {percentile:.1f}th percentile
    
    **Probability Ranges:**
    - 68% of returns fall between {mean_return - std_return:.2f}% and {mean_return + std_return:.2f}%
    - 95% of returns fall between {mean_return - 2*std_return:.2f}% and {mean_return + 2*std_return:.2f}%
    - 99.7% of returns fall between {mean_return - 3*std_return:.2f}% and {mean_return + 3*std_return:.2f}%
    
    **What This Means Simply:**
    Think of the bell curve like a hill - most returns cluster in the middle (normal days), with fewer extreme movements on either side. The Z-score tells us how unusual today's movement is:
    - Z-score between -1 and +1: Normal day (68% of all days fall here)
    - Z-score between -2 and +2: Somewhat unusual (95% of days fall here)
    - Z-score beyond ±2: Very unusual day (only 5% of days are this extreme)
    
    **Current Status**: Your current Z-score of {current_zscore:.2f} means today's movement is {'very unusual and extreme' if abs(current_zscore) > 2 else 'somewhat unusual but not extreme' if abs(current_zscore) > 1 else 'normal and typical'}.
    
    **Trading Implications:**
    {
    'Extreme movements often reverse - consider taking profits or entering counter-trend positions' if abs(current_zscore) > 2
    else 'Moderate deviation - monitor for continuation or reversal signals' if abs(current_zscore) > 1
    else 'Normal range - trend likely to continue without major disruptions'
    }
    
    **Forecast with Confidence:**
    - If positive Z-score: {'Very likely to see mean reversion (pullback)' if current_zscore > 2 else 'May see slight pullback or consolidation' if current_zscore > 1 else 'Upward momentum can continue'}
    - If negative Z-score: {'Very likely to see mean reversion (bounce)' if current_zscore < -2 else 'May see slight bounce or consolidation' if current_zscore < -1 else 'Downward momentum can continue'}
    """)
    
    # ==================== SECTION 8: FINAL TRADING RECOMMENDATION ====================
    st.header("🎯 Final Trading Recommendation")
    
    # Multi-factor signal generation
    signals = {}
    weights = {
        'multi_timeframe': 0.30,
        'rsi': 0.20,
        'zscore': 0.20,
        'ema_alignment': 0.30
    }
    
    # 1. Multi-timeframe signal
    if 'mtf_results_t1' in locals() and mtf_results_t1:
        up_count = sum(1 for r in mtf_results_t1 if r['Trend'] == 'Up')
        mtf_score = (up_count / len(mtf_results_t1)) * 2 - 1  # Scale to [-1, 1]
        signals['multi_timeframe'] = mtf_score
    else:
        signals['multi_timeframe'] = 0
    
    # 2. RSI signal
    current_rsi = data1['RSI'].iloc[-1]
    if current_rsi > 70:
        rsi_signal = -0.8  # Overbought - bearish
    elif current_rsi < 30:
        rsi_signal = 0.8  # Oversold - bullish
    elif current_rsi > 60:
        rsi_signal = -0.3
    elif current_rsi < 40:
        rsi_signal = 0.3
    else:
        rsi_signal = 0  # Neutral
    signals['rsi'] = rsi_signal
    
    # 3. Z-Score signal
    current_zscore = data1['Z_Score'].iloc[-1]
    if current_zscore > 2:
        zscore_signal = -0.8  # Extreme positive - expect reversal
    elif current_zscore < -2:
        zscore_signal = 0.8  # Extreme negative - expect bounce
    elif current_zscore > 1:
        zscore_signal = -0.3
    elif current_zscore < -1:
        zscore_signal = 0.3
    else:
        zscore_signal = 0
    signals['zscore'] = zscore_signal
    
    # 4. EMA alignment signal
    current_price = data1['Close'].iloc[-1]
    ema_20 = data1['EMA_20'].iloc[-1]
    ema_50 = data1['EMA_50'].iloc[-1]
    ema_200 = data1['EMA_200'].iloc[-1]
    
    ema_bullish_count = 0
    ema_bearish_count = 0
    
    if current_price > ema_20:
        ema_bullish_count += 1
    else:
        ema_bearish_count += 1
        
    if current_price > ema_50:
        ema_bullish_count += 1
    else:
        ema_bearish_count += 1
        
    if current_price > ema_200:
        ema_bullish_count += 1
    else:
        ema_bearish_count += 1
    
    if ema_20 > ema_50 > ema_200:
        ema_bullish_count += 2  # Perfect alignment
    elif ema_20 < ema_50 < ema_200:
        ema_bearish_count += 2  # Perfect bearish alignment
    
    ema_signal = (ema_bullish_count - ema_bearish_count) / 5  # Scale to [-1, 1]
    signals['ema_alignment'] = ema_signal
    
    # Calculate weighted combined signal
    combined_signal = sum(signals[k] * weights[k] for k in signals.keys())
    
    # Determine action
    if combined_signal > 0.3:
        action = "🟢 BUY"
        action_color = "positive"
        confidence = "High" if combined_signal > 0.6 else "Moderate"
    elif combined_signal < -0.3:
        action = "🔴 SELL / SHORT"
        action_color = "negative"
        confidence = "High" if combined_signal < -0.6 else "Moderate"
    else:
        action = "🟡 HOLD / NEUTRAL"
        action_color = "neutral"
        confidence = "Low"
    
    # Calculate entry, target, and stop loss
    atr_current = data1['ATR'].iloc[-1]
    entry_price = current_price
    
    if combined_signal > 0.3:  # Buy signal
        target_price = entry_price + (2 * atr_current)
        stop_loss = entry_price - (1 * atr_current)
        position_type = "LONG"
    elif combined_signal < -0.3:  # Sell signal
        target_price = entry_price - (2 * atr_current)
        stop_loss = entry_price + (1 * atr_current)
        position_type = "SHORT"
    else:  # Hold
        target_price = entry_price
        stop_loss = entry_price
        position_type = "NONE"
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    pct_gain_target = ((target_price - entry_price) / entry_price) * 100
    pct_loss_stop = ((stop_loss - entry_price) / entry_price) * 100
    
    # Position sizing (1-2% risk)
    account_risk_pct = 1.5  # 1.5% of account
    position_size_pct = (account_risk_pct / abs(pct_loss_stop)) * 100 if abs(pct_loss_stop) > 0 else 0
    position_size_pct = min(position_size_pct, 20)  # Cap at 20% of account
    
    # Display recommendation
    st.markdown(f"""
    <div class='metric-card'>
        <h2 class='{action_color}'>{action}</h2>
        <h3>Confidence Level: {confidence}</h3>
        <h3>Combined Signal Strength: {combined_signal:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Trade setup
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entry Price", f"₹{entry_price:.2f}")
    with col2:
        st.metric("Target Price", f"₹{target_price:.2f}", f"{pct_gain_target:+.2f}%")
    with col3:
        st.metric("Stop Loss", f"₹{stop_loss:.2f}", f"{pct_loss_stop:+.2f}%")
    with col4:
        st.metric("Risk/Reward Ratio", f"1:{risk_reward:.2f}")
    
    # Signal breakdown
    st.subheader("📊 Signal Component Analysis")
    
    signal_breakdown = pd.DataFrame({
        'Component': ['Multi-Timeframe Trend', 'RSI Indicator', 'Z-Score Analysis', 'EMA Alignment'],
        'Raw Score': [signals['multi_timeframe'], signals['rsi'], signals['zscore'], signals['ema_alignment']],
        'Weight': [weights['multi_timeframe'], weights['rsi'], weights['zscore'], weights['ema_alignment']],
        'Weighted Score': [signals['multi_timeframe'] * weights['multi_timeframe'],
                          signals['rsi'] * weights['rsi'],
                          signals['zscore'] * weights['zscore'],
                          signals['ema_alignment'] * weights['ema_alignment']],
        'Interpretation': [
            'Bullish' if signals['multi_timeframe'] > 0.2 else 'Bearish' if signals['multi_timeframe'] < -0.2 else 'Neutral',
            'Bullish' if signals['rsi'] > 0.2 else 'Bearish' if signals['rsi'] < -0.2 else 'Neutral',
            'Bullish' if signals['zscore'] > 0.2 else 'Bearish' if signals['zscore'] < -0.2 else 'Neutral',
            'Bullish' if signals['ema_alignment'] > 0.2 else 'Bearish' if signals['ema_alignment'] < -0.2 else 'Neutral'
        ]
    })
    
    st.dataframe(signal_breakdown, use_container_width=True)
    
    # Risk management
    st.subheader("⚠️ Risk Management Guidelines")
    
    st.info(f"""
    **Position Sizing:**
    - Recommended position size: **{position_size_pct:.1f}%** of your trading capital
    - This ensures you risk only 1.5% of your account on this trade
    - Example: If you have ₹100,000, invest ₹{position_size_pct * 1000:.0f}
    
    **Exit Strategy:**
    - **Take Profit**: Exit at ₹{target_price:.2f} (Expected gain: {pct_gain_target:+.2f}%)
    - **Stop Loss**: Exit immediately if price hits ₹{stop_loss:.2f} (Max loss: {pct_loss_stop:+.2f}%)
    - **Trailing Stop**: Once in profit by {abs(pct_gain_target)/2:.1f}%, move stop loss to breakeven
    
    **Trade Management:**
    - Monitor the position every {interval} (your selected timeframe)
    - If RSI reaches extreme levels (>80 or <20), consider partial profit booking
    - Re-evaluate if multi-timeframe trend changes
    - Set alerts at key support/resistance levels
    
    **Trailing Stop Recommendations:**
    - After +{abs(pct_gain_target)/4:.1f}% profit: Trail stop to entry price (breakeven)
    - After +{abs(pct_gain_target)/2:.1f}% profit: Trail stop to +{abs(pct_gain_target)/4:.1f}%
    - After reaching target: Book partial profits (50-70%) and let rest run with trailing stop
    """)
    
    # Detailed rationale
    st.subheader("🧠 Why This Signal Was Generated")
    
    # Build rationale based on components
    rationale_parts = []
    
    # Multi-timeframe analysis
    if signals['multi_timeframe'] > 0.3:
        rationale_parts.append(f"✅ **Strong multi-timeframe bullish trend**: {up_count} out of {len(mtf_results_t1)} timeframes are showing upward momentum, indicating a robust uptrend across different time horizons.")
    elif signals['multi_timeframe'] < -0.3:
        rationale_parts.append(f"❌ **Strong multi-timeframe bearish trend**: Most timeframes are showing downward momentum, indicating a robust downtrend.")
    else:
        rationale_parts.append(f"⚠️ **Mixed multi-timeframe signals**: The market is showing conflicting signals across different timeframes, suggesting consolidation or transition phase.")
    
    # RSI analysis
    if current_rsi > 70:
        rationale_parts.append(f"❌ **RSI Overbought** ({current_rsi:.1f}): The market is potentially overextended to the upside, increasing the probability of a pullback.")
    elif current_rsi < 30:
        rationale_parts.append(f"✅ **RSI Oversold** ({current_rsi:.1f}): The market is potentially oversold, creating a buying opportunity as prices may bounce.")
    else:
        rationale_parts.append(f"➡️ **RSI Neutral** ({current_rsi:.1f}): RSI is in the middle range, not providing strong directional bias.")
    
    # Z-score analysis
    if current_zscore > 2:
        rationale_parts.append(f"❌ **Extreme Positive Z-Score** ({current_zscore:.2f}): Current returns are {abs(current_zscore):.1f} standard deviations above average - statistically unusual and likely to revert to the mean.")
    elif current_zscore < -2:
        rationale_parts.append(f"✅ **Extreme Negative Z-Score** ({current_zscore:.2f}): Current returns are {abs(current_zscore):.1f} standard deviations below average - statistically unusual and likely to bounce back.")
    elif abs(current_zscore) > 1:
        rationale_parts.append(f"⚠️ **Moderate Z-Score** ({current_zscore:.2f}): Returns are somewhat unusual but not extreme, suggesting cautious approach.")
    else:
        rationale_parts.append(f"➡️ **Normal Z-Score** ({current_zscore:.2f}): Returns are within normal range, supporting trend continuation.")
    
    # EMA alignment
    if signals['ema_alignment'] > 0.3:
        rationale_parts.append(f"✅ **Bullish EMA Alignment**: Price is above key moving averages (20/50/200 EMA), and shorter EMAs are above longer ones - classic bullish structure.")
    elif signals['ema_alignment'] < -0.3:
        rationale_parts.append(f"❌ **Bearish EMA Alignment**: Price is below key moving averages, and shorter EMAs are below longer ones - classic bearish structure.")
    else:
        rationale_parts.append(f"⚠️ **Mixed EMA Alignment**: EMAs are not clearly aligned, suggesting the market lacks clear directional conviction.")
    
    st.markdown("\n\n".join(rationale_parts))
    
    # Historical context
    st.subheader("📚 Historical Context")
    
    recent_performance = (current_price - data1['Close'].iloc[-20]) / data1['Close'].iloc[-20] * 100
    monthly_performance = (current_price - data1['Close'].iloc[0]) / data1['Close'].iloc[0] * 100
    
    st.write(f"""
    **Recent Performance:**
    - Last 20 periods: {recent_performance:+.2f}%
    - Full period ({period}): {monthly_performance:+.2f}%
    - Current volatility: {data1['Volatility'].iloc[-1]:.2f}% ({'High' if data1['Volatility'].iloc[-1] > data1['Volatility'].mean() else 'Low'} compared to average)
    - Average True Range (ATR): {atr_current:.2f} points
    
    **What History Tells Us:**
    {
    f"The {ticker1_option} has been in a strong uptrend, gaining {recent_performance:.1f}% recently. However, the current signal suggests caution as some indicators show overextension." 
    if recent_performance > 5 and combined_signal < 0
    else f"The {ticker1_option} has been declining, losing {abs(recent_performance):.1f}% recently. The current signal suggests this may be a reversal opportunity." 
    if recent_performance < -5 and combined_signal > 0
    else f"The {ticker1_option} has been moving sideways with a {abs(recent_performance):.1f}% change. The signal suggests building {'bullish' if combined_signal > 0 else 'bearish'} momentum."
    }
    """)
    
    # Expected scenario
    st.subheader("🔮 Expected Scenario")
    
    if combined_signal > 0.3:
        scenario = f"""
        **Bullish Scenario (Primary):**
        - Expected move: Price should rally towards ₹{target_price:.2f} ({pct_gain_target:+.2f}%)
        - Key support: ₹{stop_loss:.2f} must hold
        - Confirmation: Look for higher highs and higher lows on shorter timeframes
        - Catalyst: {'Multi-timeframe alignment' if signals['multi_timeframe'] > 0.5 else 'RSI oversold bounce' if signals['rsi'] > 0.5 else 'Mean reversion from extreme levels'}
        
        **Risk Scenario:**
        - If stop loss is hit at ₹{stop_loss:.2f}, the bullish thesis is invalidated
        - In that case, wait for price to stabilize before re-entering
        - Alternative: Consider waiting for confirmation break above ₹{entry_price + (entry_price * 0.01):.2f}
        """
    elif combined_signal < -0.3:
        scenario = f"""
        **Bearish Scenario (Primary):**
        - Expected move: Price should decline towards ₹{target_price:.2f} ({pct_gain_target:+.2f}%)
        - Key resistance: ₹{stop_loss:.2f} must not break
        - Confirmation: Look for lower highs and lower lows on shorter timeframes
        - Catalyst: {'Multi-timeframe breakdown' if signals['multi_timeframe'] < -0.5 else 'RSI overbought correction' if signals['rsi'] < -0.5 else 'Mean reversion from extreme levels'}
        
        **Risk Scenario:**
        - If stop loss is hit at ₹{stop_loss:.2f}, the bearish thesis is invalidated
        - In that case, the trend may be reversing to bullish
        - Alternative: Consider waiting for confirmation break below ₹{entry_price - (entry_price * 0.01):.2f}
        """
    else:
        scenario = f"""
        **Neutral Scenario (Hold):**
        - Current signals are conflicting - no clear directional edge
        - Price likely to consolidate between ₹{entry_price * 0.97:.2f} and ₹{entry_price * 1.03:.2f}
        - Wait for clearer signals before entering positions
        - Watch for: RSI moving to extremes, multi-timeframe alignment, or breakout from consolidation
        
        **What to Watch For:**
        - **Bullish Breakout**: Entry above ₹{entry_price * 1.02:.2f} with volume
        - **Bearish Breakdown**: Entry below ₹{entry_price * 0.98:.2f} with volume
        - Patience is key - don't force trades in unclear market conditions
        """
    
    st.info(scenario)
    
    # Market structure analysis
    st.subheader("🏗️ Market Structure Analysis")
    
    recent_highs = data1['High'].tail(20)
    recent_lows = data1['Low'].tail(20)
    
    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
    lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])
    
    if higher_highs > 15:
        structure = "Strong Uptrend (Consistent Higher Highs)"
    elif lower_lows > 15:
        structure = "Strong Downtrend (Consistent Lower Lows)"
    elif higher_highs > 10 and lower_lows < 5:
        structure = "Uptrend (More Higher Highs than Lower Lows)"
    elif lower_lows > 10 and higher_highs < 5:
        structure = "Downtrend (More Lower Lows than Higher Highs)"
    else:
        structure = "Consolidation/Sideways (Mixed Price Action)"
    
    st.write(f"""
    **Current Market Structure:** {structure}
    
    **What This Means:**
    {
    "The market is making higher highs and higher lows - a textbook uptrend. This is bullish and supports long positions." if "Uptrend" in structure and "Strong" in structure
    else "The market is making lower highs and lower lows - a textbook downtrend. This is bearish and supports short positions." if "Downtrend" in structure and "Strong" in structure
    else "The market is moving sideways without clear direction. Wait for a breakout before taking positions." if "Consolidation" in structure
    else "The market shows some trending behavior but lacks strong momentum. Be cautious with position sizing."
    }
    
    **Professional Tip:** Market structure is one of the most reliable indicators. Always trade in the direction of the structure for higher probability setups.
    """)
    
    # Final summary for beginners
    st.subheader("👶 Simple Explanation (For Beginners)")
    
    st.success(f"""
    **What Should You Do?**
    
    Our analysis says: **{action}** with **{confidence}** confidence.
    
    **In Simple Words:**
    {
    f"We analyzed {ticker1_option} from multiple angles (trends, momentum, statistics, moving averages) and most indicators suggest the price will GO UP. "
    f"We recommend BUYING at current price ₹{entry_price:.2f}, with a target of ₹{target_price:.2f} (expected profit: {abs(pct_gain_target):.1f}%). "
    f"If the price falls to ₹{stop_loss:.2f}, exit immediately to limit losses to {abs(pct_loss_stop):.1f}%."
    if combined_signal > 0.3
    
    else f"We analyzed {ticker1_option} from multiple angles and most indicators suggest the price will GO DOWN. "
    f"We recommend SELLING/SHORTING at current price ₹{entry_price:.2f}, with a target of ₹{target_price:.2f} (expected profit: {abs(pct_gain_target):.1f}%). "
    f"If the price rises to ₹{stop_loss:.2f}, exit immediately to limit losses to {abs(pct_loss_stop):.1f}%."
    if combined_signal < -0.3
    
    else f"We analyzed {ticker1_option} and the signals are MIXED. Some indicators say up, others say down. "
    f"In such situations, it's best to WAIT and not trade. Wait for clearer signals to avoid losing money on unclear setups."
    }
    
    **Remember:**
    1. **Never risk more than 1-2% of your capital** on a single trade
    2. **Always use stop losses** - they protect you from big losses
    3. **Don't get greedy** - book profits at the target price
    4. **Market can be unpredictable** - this is analysis, not a guarantee
    5. **Practice first** - consider paper trading before real money
    
    **Risk vs Reward:**
    For every ₹{abs(pct_loss_stop):.1f} you risk, you can potentially gain ₹{abs(pct_gain_target):.1f}. That's a {risk_reward:.1f}:1 risk-reward ratio, which is {'excellent' if risk_reward > 2 else 'good' if risk_reward > 1.5 else 'acceptable' if risk_reward > 1 else 'not ideal'}.
    """)
    
    # Disclaimer
    st.warning("""
    ⚠️ **DISCLAIMER**: This analysis is for educational purposes only and should not be considered as financial advice. 
    Always do your own research, understand the risks involved, and consider consulting with a qualified financial advisor 
    before making any investment decisions. Past performance does not guarantee future results. Trading involves substantial 
    risk of loss and is not suitable for all investors.
    """)

else:
    st.info("👆 Please configure your parameters in the sidebar and click 'Fetch Data & Analyze' to begin analysis.")
    
    # Show feature highlights
    st.markdown("""
    ## 🌟 Dashboard Features
    
    ### 📊 Core Analytics
    - **Real-time Market Data**: Fetch live data from Yahoo Finance for stocks, indices, crypto, forex, and commodities
    - **Multi-Asset Support**: Analyze NIFTY 50, Bank NIFTY, SENSEX, BTC, ETH, Gold, Silver, USD/INR, and more
    - **Ratio Analysis**: Compare two assets and analyze their price ratio dynamics
    
    ### 📈 Technical Analysis
    - **Multi-Timeframe Analysis**: Analyze trends across 10 different timeframes simultaneously
    - **Advanced Indicators**: RSI, EMA (9,20,21,33,50,100,150,200), SMA (20,50,100,150,200), Fibonacci levels
    - **Support & Resistance**: Automatic detection of key price levels
    - **Volatility Analysis**: Historical volatility bins with performance statistics
    
    ### 🔍 Pattern Recognition
    - **Significant Move Detection**: Identifies large price movements and analyzes preceding conditions
    - **Divergence Detection**: Spots RSI divergences that often signal reversals
    - **EMA Crossovers**: Detects important moving average crossovers
    - **Volume Analysis**: Identifies volume spikes and unusual trading activity
    
    ### 📊 Statistical Analysis
    - **Distribution Analysis**: Understand return distributions with histograms and normal curves
    - **Z-Score Analysis**: Identify statistically unusual price movements
    - **Bell Curve Visualization**: See where current price stands in statistical context
    - **Probability Ranges**: 68%, 95%, and 99.7% confidence intervals
    
    ### 🎯 Trading Signals
    - **Multi-Factor Analysis**: Combines 4 key factors (multi-timeframe, RSI, Z-score, EMA alignment)
    - **Weighted Scoring**: Intelligent weighting system for more accurate signals
    - **Clear Buy/Sell/Hold**: Simple actionable recommendations
    - **Risk Management**: ATR-based stop loss and target calculations
    
    ### 📉 Risk Management
    - **Position Sizing**: Automatic calculation based on 1-2% account risk
    - **Risk/Reward Ratios**: Clear risk vs reward assessment for each trade
    - **Stop Loss & Targets**: Precise entry, exit, and stop loss levels
    - **Trailing Stop Recommendations**: Dynamic exit strategies for maximum profit
    
    ### 📊 Interactive Visualizations
    - **Candlestick Charts**: Professional trading charts with multiple EMAs
    - **RSI Indicators**: Visual representation with overbought/oversold zones
    - **Ratio Charts**: Compare two assets visually with RSI
    - **Statistical Distributions**: Interactive histograms and bell curves
    
    ### 💾 Data Export
    - **CSV Export**: Download complete OHLCV data with indicators
    - **Ratio Data Export**: Export comparison data for further analysis
    - **Timezone Support**: All data properly converted to IST
    
    ---
    
    **🚀 Ready to start? Configure your parameters in the sidebar and click "Fetch Data & Analyze"!**
    """)
