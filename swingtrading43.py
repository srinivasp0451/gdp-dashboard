import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Professional Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stButton>button {width: 100%; margin-top: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .green-text {color: #00cc00; font-weight: bold;}
    .red-text {color: #ff0000; font-weight: bold;}
    .yellow-text {color: #ffaa00; font-weight: bold;}
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Ticker mapping for Indian indices
TICKER_MAP = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'JPY=X'
}

# Utility Functions
def get_ist_time():
    """Get current time in IST"""
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def convert_to_ist(df):
    """Convert DataFrame datetime to IST"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_volatility(data, window=20):
    """Calculate historical volatility"""
    log_returns = np.log(data / data.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252) * 100

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100.0%': low
    }

def find_support_resistance(data, num_levels=3):
    """Find support and resistance levels using peak detection"""
    try:
        peaks, _ = find_peaks(data.values, distance=5)
        troughs, _ = find_peaks(-data.values, distance=5)
        
        resistance_levels = sorted(data.iloc[peaks].values, reverse=True)[:num_levels]
        support_levels = sorted(data.iloc[troughs].values)[:num_levels]
        
        return support_levels, resistance_levels
    except:
        return [], []

def detect_divergence(price, rsi, window=10):
    """Detect RSI divergence patterns"""
    if len(price) < window:
        return "Insufficient Data"
    
    price_trend = (price.iloc[-1] - price.iloc[-window]) / price.iloc[-window] * 100
    rsi_trend = rsi.iloc[-1] - rsi.iloc[-window]
    
    if price_trend > 2 and rsi_trend < -5:
        return "Bearish Divergence"
    elif price_trend < -2 and rsi_trend > 5:
        return "Bullish Divergence"
    else:
        return "No Divergence"

def fetch_data_with_retry(ticker, period, interval, max_retries=3, delay=2):
    """Fetch data with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            time.sleep(delay)  # Rate limiting
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if not data.empty:
                return convert_to_ist(data)
            else:
                st.warning(f"No data returned for {ticker} (Attempt {attempt + 1})")
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(delay * (attempt + 1))
            else:
                st.error(f"Failed to fetch data for {ticker}: {str(e)}")
                return pd.DataFrame()
    return pd.DataFrame()

# Sidebar Configuration
st.sidebar.title("🎯 Trading Dashboard Config")
st.sidebar.markdown("---")

# Ticker Selection
st.sidebar.subheader("📊 Asset Selection")
ticker1_type = st.sidebar.selectbox("Ticker 1 Type", ["Predefined", "Custom"])
if ticker1_type == "Predefined":
    ticker1 = st.sidebar.selectbox("Select Ticker 1", list(TICKER_MAP.keys()))
    ticker1_symbol = TICKER_MAP[ticker1]
else:
    ticker1_symbol = st.sidebar.text_input("Enter Custom Ticker 1", "AAPL").upper()
    ticker1 = ticker1_symbol

# Enable/Disable Ratio Analysis
enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)", value=False)

ticker2_symbol = None
ticker2 = None
if enable_ratio:
    ticker2_type = st.sidebar.selectbox("Ticker 2 Type", ["Predefined", "Custom"])
    if ticker2_type == "Predefined":
        ticker2 = st.sidebar.selectbox("Select Ticker 2", list(TICKER_MAP.keys()), index=1)
        ticker2_symbol = TICKER_MAP[ticker2]
    else:
        ticker2_symbol = st.sidebar.text_input("Enter Custom Ticker 2", "MSFT").upper()
        ticker2 = ticker2_symbol

# Timeframe and Period Selection
st.sidebar.markdown("---")
st.sidebar.subheader("⏰ Timeframe Settings")
interval = st.sidebar.selectbox(
    "Interval",
    ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"],
    index=7
)
period = st.sidebar.selectbox(
    "Period",
    ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"],
    index=6
)

# API Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ API Settings")
api_delay = st.sidebar.slider("API Delay (seconds)", 1.0, 5.0, 2.0, 0.5)

# Analysis Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Analysis Settings")
pattern_threshold = st.sidebar.number_input("Pattern Detection Threshold (points)", value=30, min_value=1)
num_bins = st.sidebar.slider("Number of Bins for Analysis", 3, 15, 10)

# Fetch Data Button
st.sidebar.markdown("---")
fetch_button = st.sidebar.button("🔄 Fetch Data", type="primary")

# Main Dashboard
st.title("🚀 Professional Algorithmic Trading Dashboard")
st.markdown(f"*Last Updated: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')}*")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None

# Fetch Data Logic
if fetch_button:
    with st.spinner(f"Fetching data for {ticker1}..."):
        st.session_state.df1 = fetch_data_with_retry(ticker1_symbol, period, interval, delay=api_delay)
        
        if enable_ratio and ticker2_symbol:
            with st.spinner(f"Fetching data for {ticker2}..."):
                st.session_state.df2 = fetch_data_with_retry(ticker2_symbol, period, interval, delay=api_delay)
        else:
            st.session_state.df2 = None
        
        if not st.session_state.df1.empty:
            st.session_state.data_fetched = True
            st.success("✅ Data fetched successfully!")
        else:
            st.error("❌ Failed to fetch data. Please check your ticker symbol and try again.")
            st.session_state.data_fetched = False

# Main Analysis - Only show if data is fetched
if st.session_state.data_fetched and st.session_state.df1 is not None:
    df1 = st.session_state.df1.copy()
    df2 = st.session_state.df2.copy() if st.session_state.df2 is not None else None
    
    # Calculate indicators for Ticker 1
    df1['RSI'] = calculate_rsi(df1['Close'])
    df1['EMA9'] = calculate_ema(df1['Close'], 9)
    df1['EMA20'] = calculate_ema(df1['Close'], 20)
    df1['EMA21'] = calculate_ema(df1['Close'], 21)
    df1['EMA33'] = calculate_ema(df1['Close'], 33)
    df1['EMA50'] = calculate_ema(df1['Close'], 50)
    df1['EMA100'] = calculate_ema(df1['Close'], 100)
    df1['EMA150'] = calculate_ema(df1['Close'], 150)
    df1['EMA200'] = calculate_ema(df1['Close'], 200)
    df1['SMA20'] = calculate_sma(df1['Close'], 20)
    df1['SMA50'] = calculate_sma(df1['Close'], 50)
    df1['SMA100'] = calculate_sma(df1['Close'], 100)
    df1['SMA150'] = calculate_sma(df1['Close'], 150)
    df1['SMA200'] = calculate_sma(df1['Close'], 200)
    df1['Volatility'] = calculate_volatility(df1['Close'])
    df1['Returns'] = df1['Close'].pct_change() * 100
    df1['Returns_Points'] = df1['Close'].diff()
    
    # Calculate indicators for Ticker 2 if enabled
    if df2 is not None and not df2.empty:
        df2['RSI'] = calculate_rsi(df2['Close'])
        df2['EMA9'] = calculate_ema(df2['Close'], 9)
        df2['EMA20'] = calculate_ema(df2['Close'], 20)
        df2['EMA21'] = calculate_ema(df2['Close'], 21)
        df2['EMA33'] = calculate_ema(df2['Close'], 33)
        df2['EMA50'] = calculate_ema(df2['Close'], 50)
        df2['EMA100'] = calculate_ema(df2['Close'], 100)
        df2['EMA150'] = calculate_ema(df2['Close'], 150)
        df2['EMA200'] = calculate_ema(df2['Close'], 200)
        df2['SMA20'] = calculate_sma(df2['Close'], 20)
        df2['SMA50'] = calculate_sma(df2['Close'], 50)
        df2['SMA100'] = calculate_sma(df2['Close'], 100)
        df2['SMA150'] = calculate_sma(df2['Close'], 150)
        df2['SMA200'] = calculate_sma(df2['Close'], 200)
        df2['Volatility'] = calculate_volatility(df2['Close'])
        df2['Returns'] = df2['Close'].pct_change() * 100
        df2['Returns_Points'] = df2['Close'].diff()
    
    # ===========================================
    # SECTION 2: BASIC STATISTICS DISPLAY
    # ===========================================
    st.header("📊 Current Market Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price1 = df1['Close'].iloc[-1]
    prev_price1 = df1['Close'].iloc[0]
    change1 = current_price1 - prev_price1
    change_pct1 = (change1 / prev_price1) * 100
    
    with col1:
        color1 = "green" if change1 >= 0 else "red"
        st.metric(
            f"{ticker1} Price",
            f"₹{current_price1:.2f}",
            f"{change1:+.2f} ({change_pct1:+.2f}%)",
            delta_color="normal"
        )
    
    if df2 is not None and not df2.empty:
        current_price2 = df2['Close'].iloc[-1]
        prev_price2 = df2['Close'].iloc[0]
        change2 = current_price2 - prev_price2
        change_pct2 = (change2 / prev_price2) * 100
        
        with col2:
            color2 = "green" if change2 >= 0 else "red"
            st.metric(
                f"{ticker2} Price",
                f"₹{current_price2:.2f}",
                f"{change2:+.2f} ({change_pct2:+.2f}%)",
                delta_color="normal"
            )
        
        with col3:
            ratio = current_price1 / current_price2
            st.metric(
                "Current Ratio",
                f"{ratio:.4f}",
                "T1/T2"
            )
        
        with col4:
            st.metric(
                "Data Points",
                len(df1),
                f"{interval} interval"
            )
    else:
        with col2:
            st.metric(
                "Data Points",
                len(df1),
                f"{interval} interval"
            )
        with col3:
            rsi_current = df1['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
            st.metric("RSI Status", rsi_status, f"{rsi_current:.2f}")
    
    # Data Table
    st.subheader("📋 Recent Price Data")
    display_df1 = df1[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Returns_Points', 'RSI']].tail(20).copy()
    display_df1.index = display_df1.index.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_df1.style.format({
        'Open': '{:.2f}',
        'High': '{:.2f}',
        'Low': '{:.2f}',
        'Close': '{:.2f}',
        'Volume': '{:,.0f}',
        'Returns': '{:+.2f}%',
        'Returns_Points': '{:+.2f}',
        'RSI': '{:.2f}'
    }), use_container_width=True)
    
    # ===========================================
    # SECTION 3: RATIO ANALYSIS
    # ===========================================
    if enable_ratio and df2 is not None and not df2.empty:
        st.header("⚖️ Ratio Analysis")
        
        # Align dataframes
        df_ratio = pd.DataFrame()
        df_ratio['Ticker1_Price'] = df1['Close']
        df_ratio['Ticker2_Price'] = df2['Close']
        df_ratio['Ratio'] = df_ratio['Ticker1_Price'] / df_ratio['Ticker2_Price']
        df_ratio['RSI_T1'] = df1['RSI']
        df_ratio['RSI_T2'] = df2['RSI']
        df_ratio['RSI_Ratio'] = calculate_rsi(df_ratio['Ratio'])
        df_ratio = df_ratio.dropna()
        
        # Display Ratio Table
        st.subheader("📊 Ratio Comparison Table")
        display_ratio = df_ratio.tail(20).copy()
        display_ratio.index = display_ratio.index.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_ratio.style.format({
            'Ticker1_Price': '{:.2f}',
            'Ticker2_Price': '{:.2f}',
            'Ratio': '{:.4f}',
            'RSI_T1': '{:.2f}',
            'RSI_T2': '{:.2f}',
            'RSI_Ratio': '{:.2f}'
        }), use_container_width=True)
        
        # Export functionality
        csv = df_ratio.to_csv()
        st.download_button(
            label="📥 Download Ratio Data (CSV)",
            data=csv,
            file_name=f"ratio_analysis_{ticker1}_{ticker2}_{get_ist_time().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Ratio Binning Analysis
        st.subheader("📊 Ratio Binning Analysis")
        
        df_ratio['Ratio_Bin'] = pd.qcut(df_ratio['Ratio'], q=num_bins, duplicates='drop')
        df_ratio['T1_Returns'] = df1['Returns']
        df_ratio['T2_Returns'] = df2['Returns']
        df_ratio['T1_Returns_Points'] = df1['Returns_Points']
        df_ratio['T2_Returns_Points'] = df2['Returns_Points']
        
        # Calculate next candle returns
        for i in [1, 2, 3, 4, 5]:
            df_ratio[f'T1_Next_{i}_Returns'] = df_ratio['T1_Returns'].shift(-i)
            df_ratio[f'T2_Next_{i}_Returns'] = df_ratio['T2_Returns'].shift(-i)
            df_ratio[f'T1_Next_{i}_Points'] = df_ratio['T1_Returns_Points'].shift(-i)
            df_ratio[f'T2_Next_{i}_Points'] = df_ratio['T2_Returns_Points'].shift(-i)
        
        bin_analysis = df_ratio.groupby('Ratio_Bin').agg({
            'T1_Returns': ['mean', 'std', 'count'],
            'T2_Returns': ['mean', 'std', 'count'],
            'T1_Returns_Points': ['mean', 'std'],
            'T2_Returns_Points': ['mean', 'std'],
            'T1_Next_1_Returns': 'mean',
            'T1_Next_2_Returns': 'mean',
            'T1_Next_3_Returns': 'mean',
            'T1_Next_4_Returns': 'mean',
            'T1_Next_5_Returns': 'mean',
            'T2_Next_1_Returns': 'mean',
            'T2_Next_2_Returns': 'mean',
            'T2_Next_3_Returns': 'mean',
            'T2_Next_4_Returns': 'mean',
            'T2_Next_5_Returns': 'mean'
        }).round(2)
        
        st.dataframe(bin_analysis, use_container_width=True)
        
        # Current bin analysis
        current_ratio = df_ratio['Ratio'].iloc[-1]
        current_bin = df_ratio['Ratio_Bin'].iloc[-1]
        
        st.markdown(f"""
        <div class="info-box">
        <h4>📍 Current Ratio Bin: {current_bin}</h4>
        <p><strong>Current Ratio:</strong> {current_ratio:.4f}</p>
        <p><strong>Historical Performance in This Bin:</strong></p>
        <ul>
            <li>{ticker1} Average Return: {bin_analysis.loc[current_bin, ('T1_Returns', 'mean')]:.2f}% ({bin_analysis.loc[current_bin, ('T1_Returns_Points', 'mean')]:+.2f} points)</li>
            <li>{ticker2} Average Return: {bin_analysis.loc[current_bin, ('T2_Returns', 'mean')]:.2f}% ({bin_analysis.loc[current_bin, ('T2_Returns_Points', 'mean')]:+.2f} points)</li>
            <li>Occurrences in this bin: {int(bin_analysis.loc[current_bin, ('T1_Returns', 'count')])}</li>
        </ul>
        <p><strong>Forecast:</strong> Based on historical patterns, when the ratio is in this range, {ticker1} tends to move {bin_analysis.loc[current_bin, ('T1_Returns', 'mean')]:+.2f}% while {ticker2} tends to move {bin_analysis.loc[current_bin, ('T2_Returns', 'mean')]:+.2f}%.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===========================================
    # SECTION 4: MULTI-TIMEFRAME ANALYSIS
    # ===========================================
    st.header("🕐 Multi-Timeframe Analysis")
    
    timeframes = [
        ("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("30m", "1mo"),
        ("1h", "1mo"), ("2h", "3mo"), ("4h", "6mo"), ("1d", "1y"),
        ("1wk", "5y"), ("1mo", "10y")
    ]
    
    def analyze_timeframe(ticker_symbol, interval_tf, period_tf):
        """Analyze a single timeframe"""
        df_tf = fetch_data_with_retry(ticker_symbol, period_tf, interval_tf, delay=api_delay)
        if df_tf.empty:
            return None
        
        df_tf['RSI'] = calculate_rsi(df_tf['Close'])
        df_tf['EMA9'] = calculate_ema(df_tf['Close'], 9)
        df_tf['EMA20'] = calculate_ema(df_tf['Close'], 20)
        df_tf['EMA21'] = calculate_ema(df_tf['Close'], 21)
        df_tf['EMA33'] = calculate_ema(df_tf['Close'], 33)
        df_tf['EMA50'] = calculate_ema(df_tf['Close'], 50)
        df_tf['EMA100'] = calculate_ema(df_tf['Close'], 100)
        df_tf['EMA150'] = calculate_ema(df_tf['Close'], 150)
        df_tf['EMA200'] = calculate_ema(df_tf['Close'], 200)
        df_tf['SMA20'] = calculate_sma(df_tf['Close'], 20)
        df_tf['SMA50'] = calculate_sma(df_tf['Close'], 50)
        df_tf['SMA100'] = calculate_sma(df_tf['Close'], 100)
        df_tf['SMA150'] = calculate_sma(df_tf['Close'], 150)
        df_tf['SMA200'] = calculate_sma(df_tf['Close'], 200)
        df_tf['Volatility'] = calculate_volatility(df_tf['Close'])
        
        current_price = df_tf['Close'].iloc[-1]
        max_price = df_tf['High'].max()
        min_price = df_tf['Low'].min()
        
        fib_levels = calculate_fibonacci_levels(max_price, min_price)
        support, resistance = find_support_resistance(df_tf['Close'])
        
        # Trend determination
        if current_price > df_tf['EMA20'].iloc[-1] and df_tf['EMA20'].iloc[-1] > df_tf['EMA50'].iloc[-1]:
            trend = "Up"
        elif current_price < df_tf['EMA20'].iloc[-1] and df_tf['EMA20'].iloc[-1] < df_tf['EMA50'].iloc[-1]:
            trend = "Down"
        else:
            trend = "Sideways"
        
        # RSI status
        rsi_val = df_tf['RSI'].iloc[-1]
        if rsi_val > 70:
            rsi_status = "Overbought"
        elif rsi_val < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"
        
        change = current_price - df_tf['Close'].iloc[0]
        change_pct = (change / df_tf['Close'].iloc[0]) * 100
        
        return {
            'Timeframe': f"{interval_tf}/{period_tf}",
            'Trend': trend,
            'Max_Close': max_price,
            'Min_Close': min_price,
            'Fib_0': fib_levels['0.0%'],
            'Fib_236': fib_levels['23.6%'],
            'Fib_382': fib_levels['38.2%'],
            'Fib_50': fib_levels['50.0%'],
            'Fib_618': fib_levels['61.8%'],
            'Fib_786': fib_levels['78.6%'],
            'Fib_100': fib_levels['100.0%'],
            'Volatility': df_tf['Volatility'].iloc[-1],
            'Change_%': change_pct,
            'Change_Points': change,
            'Support_1': support[0] if len(support) > 0 else np.nan,
            'Support_2': support[1] if len(support) > 1 else np.nan,
            'Support_3': support[2] if len(support) > 2 else np.nan,
            'Resistance_1': resistance[0] if len(resistance) > 0 else np.nan,
            'Resistance_2': resistance[1] if len(resistance) > 1 else np.nan,
            'Resistance_3': resistance[2] if len(resistance) > 2 else np.nan,
            'RSI': rsi_val,
            'RSI_Status': rsi_status,
            'EMA9': df_tf['EMA9'].iloc[-1],
            'EMA20': df_tf['EMA20'].iloc[-1],
            'EMA21': df_tf['EMA21'].iloc[-1],
            'EMA33': df_tf['EMA33'].iloc[-1],
            'EMA50': df_tf['EMA50'].iloc[-1],
            'EMA100': df_tf['EMA100'].iloc[-1],
            'EMA150': df_tf['EMA150'].iloc[-1],
            'EMA200': df_tf['EMA200'].iloc[-1],
            'Price_vs_EMA9': 'Above' if current_price > df_tf['EMA9'].iloc[-1] else 'Below',
            'Price_vs_EMA20': 'Above' if current_price > df_tf['EMA20'].iloc[-1] else 'Below',
            'Price_vs_EMA50': 'Above' if current_price > df_tf['EMA50'].iloc[-1] else 'Below',
            'Price_vs_EMA100': 'Above' if current_price > df_tf['EMA100'].iloc[-1] else 'Below',
            'Price_vs_EMA200': 'Above' if current_price > df_tf['EMA200'].iloc[-1] else 'Below',
            'SMA20': df_tf['SMA20'].iloc[-1],
            'SMA50': df_tf['SMA50'].iloc[-1],
            'SMA100': df_tf['SMA100'].iloc[-1],
            'SMA150': df_tf['SMA150'].iloc[-1],
            'SMA200': df_tf['SMA200'].iloc[-1],
            'Price_vs_SMA20': 'Above' if current_price > df_tf['SMA20'].iloc[-1] else 'Below',
            'Price_vs_SMA50': 'Above' if current_price > df_tf['SMA50'].iloc[-1] else 'Below',
            'Price_vs_SMA100': 'Above' if current_price > df_tf['SMA100'].iloc[-1] else 'Below',
            'Price_vs_SMA200': 'Above' if current_price > df_tf['SMA200'].iloc[-1] else 'Below'
        }
    
    # Analyze Ticker 1
    st.subheader(f"📈 {ticker1} Multi-Timeframe Analysis")
    with st.spinner(f"Analyzing {ticker1} across multiple timeframes..."):
        mtf_results1 = []
        for tf in timeframes:
            result = analyze_timeframe(ticker1_symbol, tf[0], tf[1])
            if result:
                mtf_results1.append(result)
        
        if mtf_results1:
            df_mtf1 = pd.DataFrame(mtf_results1)
            st.dataframe(df_mtf1.style.format({
                'Max_Close': '{:.2f}',
                'Min_Close': '{:.2f}',
                'Fib_0': '{:.2f}',
                'Fib_236': '{:.2f}',
                'Fib_382': '{:.2f}',
                'Fib_50': '{:.2f}',
                'Fib_618': '{:.2f}',
                'Fib_786': '{:.2f}',
                'Fib_100': '{:.2f}',
                'Volatility': '{:.2f}%',
                'Change_%': '{:+.2f}%',
                'Change_Points': '{:+.2f}',
                'RSI': '{:.2f}',
                'EMA9': '{:.2f}',
                'EMA20': '{:.2f}',
                'EMA50': '{:.2f}',
                'EMA200': '{:.2f}',
                'SMA20': '{:.2f}',
                'SMA50': '{:.2f}',
                'SMA200': '{:.2f}'
            }), use_container_width=True)
            
            # Summary for Ticker 1
            st.markdown(f"""
            <div class="info-box">
            <h4>📊 {ticker1} Multi-Timeframe Summary</h4>
            <p><strong>Overall Trend Assessment:</strong></p>
            <ul>
                <li>Bullish Timeframes: {sum(1 for r in mtf_results1 if r['Trend'] == 'Up')}/{len(mtf_results1)}</li>
                <li>Bearish Timeframes: {sum(1 for r in mtf_results1 if r['Trend'] == 'Down')}/{len(mtf_results1)}</li>
                <li>Average RSI across timeframes: {np.mean([r['RSI'] for r in mtf_results1]):.2f}</li>
                <li>Overbought timeframes: {sum(1 for r in mtf_results1 if r['RSI_Status'] == 'Overbought')}/{len(mtf_results1)}</li>
                <li>Oversold timeframes: {sum(1 for r in mtf_results1 if r['RSI_Status'] == 'Oversold')}/{len(mtf_results1)}</li>
            </ul>
            <p><strong>Key Insights:</strong> {'Strong bullish momentum across multiple timeframes. Price consistently above key EMAs.' if sum(1 for r in mtf_results1 if r['Trend'] == 'Up') > len(mtf_results1)/2 else 'Bearish pressure evident. Price struggling below moving averages.' if sum(1 for r in mtf_results1 if r['Trend'] == 'Down') > len(mtf_results1)/2 else 'Mixed signals. Market in consolidation phase.'}</p>
            <p><strong>Forecast:</strong> {'Upward momentum likely to continue if price holds above EMA20 support.' if sum(1 for r in mtf_results1 if r['Trend'] == 'Up') > len(mtf_results1)/2 else 'Downward pressure may persist. Watch for support at key levels.' if sum(1 for r in mtf_results1 if r['Trend'] == 'Down') > len(mtf_results1)/2 else 'Sideways movement expected. Breakout direction will determine trend.'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Analyze Ticker 2 if enabled
    if enable_ratio and df2 is not None:
        st.subheader(f"📈 {ticker2} Multi-Timeframe Analysis")
        with st.spinner(f"Analyzing {ticker2} across multiple timeframes..."):
            mtf_results2 = []
            for tf in timeframes:
                result = analyze_timeframe(ticker2_symbol, tf[0], tf[1])
                if result:
                    mtf_results2.append(result)
            
            if mtf_results2:
                df_mtf2 = pd.DataFrame(mtf_results2)
                st.dataframe(df_mtf2.style.format({
                    'Max_Close': '{:.2f}',
                    'Min_Close': '{:.2f}',
                    'Volatility': '{:.2f}%',
                    'Change_%': '{:+.2f}%',
                    'Change_Points': '{:+.2f}',
                    'RSI': '{:.2f}'
                }), use_container_width=True)
                
                # Summary for Ticker 2
                st.markdown(f"""
                <div class="info-box">
                <h4>📊 {ticker2} Multi-Timeframe Summary</h4>
                <p><strong>Overall Trend Assessment:</strong></p>
                <ul>
                    <li>Bullish Timeframes: {sum(1 for r in mtf_results2 if r['Trend'] == 'Up')}/{len(mtf_results2)}</li>
                    <li>Bearish Timeframes: {sum(1 for r in mtf_results2 if r['Trend'] == 'Down')}/{len(mtf_results2)}</li>
                    <li>Average RSI across timeframes: {np.mean([r['RSI'] for r in mtf_results2]):.2f}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # ===========================================
    # SECTION 5: VOLATILITY BINS ANALYSIS
    # ===========================================
    st.header("📊 Volatility Bins Analysis")
    
    df_vol = df1[['Close', 'Volatility', 'Returns', 'Returns_Points']].copy().dropna()
    df_vol['Volatility_Bin'] = pd.qcut(df_vol['Volatility'], q=num_bins, duplicates='drop')
    
    vol_bin_analysis = df_vol.groupby('Volatility_Bin').agg({
        'Returns': ['mean', 'std', 'min', 'max', 'count'],
        'Returns_Points': ['mean', 'std', 'min', 'max'],
        'Volatility': ['mean', 'min', 'max']
    }).round(2)
    
    st.dataframe(vol_bin_analysis, use_container_width=True)
    
    # Current volatility analysis
    current_vol = df_vol['Volatility'].iloc[-1]
    current_vol_bin = df_vol['Volatility_Bin'].iloc[-1]
    
    st.markdown(f"""
    <div class="info-box">
    <h4>📍 Current Volatility Analysis</h4>
    <p><strong>Current Volatility:</strong> {current_vol:.2f}%</p>
    <p><strong>Current Bin:</strong> {current_vol_bin}</p>
    <p><strong>Statistical Summary:</strong></p>
    <ul>
        <li>Highest Volatility: {df_vol['Volatility'].max():.2f}%</li>
        <li>Lowest Volatility: {df_vol['Volatility'].min():.2f}%</li>
        <li>Mean Volatility: {df_vol['Volatility'].mean():.2f}%</li>
        <li>Max Return in Dataset: {df_vol['Returns_Points'].max():.2f} points ({df_vol['Returns'].max():.2f}%)</li>
        <li>Min Return in Dataset: {df_vol['Returns_Points'].min():.2f} points ({df_vol['Returns'].min():.2f}%)</li>
    </ul>
    <p><strong>Historical Performance in Current Volatility Bin:</strong></p>
    <ul>
        <li>Average Return: {vol_bin_analysis.loc[current_vol_bin, ('Returns_Points', 'mean')]:.2f} points ({vol_bin_analysis.loc[current_vol_bin, ('Returns', 'mean')]:.2f}%)</li>
        <li>Volatility Range: {vol_bin_analysis.loc[current_vol_bin, ('Volatility', 'min')]:.2f}% - {vol_bin_analysis.loc[current_vol_bin, ('Volatility', 'max')]:.2f}%</li>
        <li>Occurrences: {int(vol_bin_analysis.loc[current_vol_bin, ('Returns', 'count')])}</li>
    </ul>
    <p><strong>Forecast:</strong> {'High volatility suggests larger price movements ahead. Expect increased risk and opportunity.' if current_vol > df_vol['Volatility'].quantile(0.75) else 'Low volatility indicates stable price action. Smaller moves expected.' if current_vol < df_vol['Volatility'].quantile(0.25) else 'Moderate volatility. Normal price behavior expected.'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===========================================
    # SECTION 6: ADVANCED PATTERN RECOGNITION
    # ===========================================
    st.header("🔍 Advanced Pattern Recognition")
    
    # Detect significant moves
    df_patterns = df1.copy()
    df_patterns['Large_Move'] = abs(df_patterns['Returns_Points']) > pattern_threshold
    
    pattern_results = []
    
    for idx in df_patterns[df_patterns['Large_Move']].index:
        try:
            idx_pos = df_patterns.index.get_loc(idx)
            if idx_pos < 10:
                continue
            
            # Get 10 candles before the move
            lookback = df_patterns.iloc[idx_pos-10:idx_pos]
            move_candle = df_patterns.loc[idx]
            
            # Pattern detection
            volatility_burst = lookback['Volatility'].iloc[-1] > lookback['Volatility'].mean() * 1.5
            volume_spike = False
            if 'Volume' in lookback.columns and lookback['Volume'].iloc[-1] > 0:
                volume_spike = lookback['Volume'].iloc[-1] > lookback['Volume'].mean() * 2
            
            # RSI divergence
            rsi_before = lookback['RSI'].iloc[0]
            rsi_at_move = move_candle['RSI']
            price_change = (lookback['Close'].iloc[-1] - lookback['Close'].iloc[0]) / lookback['Close'].iloc[0] * 100
            rsi_change = rsi_at_move - rsi_before
            
            divergence = "No"
            if price_change > 2 and rsi_change < -5:
                divergence = "Bearish"
            elif price_change < -2 and rsi_change > 5:
                divergence = "Bullish"
            
            # EMA crossovers
            ema_20_50_cross = False
            if len(lookback) >= 2:
                if (lookback['EMA20'].iloc[-2] < lookback['EMA50'].iloc[-2] and 
                    lookback['EMA20'].iloc[-1] > lookback['EMA50'].iloc[-1]):
                    ema_20_50_cross = True
                elif (lookback['EMA20'].iloc[-2] > lookback['EMA50'].iloc[-2] and 
                      lookback['EMA20'].iloc[-1] < lookback['EMA50'].iloc[-1]):
                    ema_20_50_cross = True
            
            # Support/Resistance breakout
            support_levels, resistance_levels = find_support_resistance(lookback['Close'])
            sr_breakout = False
            if resistance_levels and move_candle['Close'] > max(resistance_levels):
                sr_breakout = True
            elif support_levels and move_candle['Close'] < min(support_levels):
                sr_breakout = True
            
            # Large body candles
            body_size = abs(move_candle['Close'] - move_candle['Open'])
            avg_body = abs(lookback['Close'] - lookback['Open']).mean()
            large_body = body_size > avg_body * 2
            
            # Consecutive moves
            consecutive_up = (lookback['Returns_Points'] > 0).rolling(3).sum().max() >= 3
            consecutive_down = (lookback['Returns_Points'] < 0).rolling(3).sum().max() >= 3
            
            # Correlation with prior moves
            correlation = 0
            if len(lookback) > 5:
                correlation = lookback['Returns'].corr(lookback['Returns'].shift(1))
            
            pattern_results.append({
                'DateTime': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'Move_Points': move_candle['Returns_Points'],
                'Move_%': move_candle['Returns'],
                'Direction': 'Up' if move_candle['Returns_Points'] > 0 else 'Down',
                'Volatility_Burst': 'Yes' if volatility_burst else 'No',
                'Volume_Spike': 'Yes' if volume_spike else 'No',
                'RSI_Divergence': divergence,
                'RSI_Before': rsi_before,
                'RSI_At_Move': rsi_at_move,
                'EMA_20/50_Cross': 'Yes' if ema_20_50_cross else 'No',
                'SR_Breakout': 'Yes' if sr_breakout else 'No',
                'Large_Body': 'Yes' if large_body else 'No',
                'Consecutive_Up': 'Yes' if consecutive_up else 'No',
                'Consecutive_Down': 'Yes' if consecutive_down else 'No',
                'Correlation': correlation
            })
        except Exception as e:
            continue
    
    if pattern_results:
        df_pattern_results = pd.DataFrame(pattern_results)
        st.dataframe(df_pattern_results.style.format({
            'Move_Points': '{:+.2f}',
            'Move_%': '{:+.2f}%',
            'RSI_Before': '{:.2f}',
            'RSI_At_Move': '{:.2f}',
            'Correlation': '{:.2f}'
        }), use_container_width=True)
        
        # Pattern Summary
        total_patterns = len(pattern_results)
        vol_burst_count = sum(1 for p in pattern_results if p['Volatility_Burst'] == 'Yes')
        volume_spike_count = sum(1 for p in pattern_results if p['Volume_Spike'] == 'Yes')
        divergence_count = sum(1 for p in pattern_results if p['RSI_Divergence'] != 'No')
        
        st.markdown(f"""
        <div class="info-box">
        <h4>🎯 Pattern Detection Summary</h4>
        <p><strong>Total Significant Moves Detected:</strong> {total_patterns}</p>
        <p><strong>Pattern Frequency:</strong></p>
        <ul>
            <li>Volatility Bursts: {vol_burst_count}/{total_patterns} ({vol_burst_count/total_patterns*100:.1f}%)</li>
            <li>Volume Spikes: {volume_spike_count}/{total_patterns} ({volume_spike_count/total_patterns*100:.1f}%)</li>
            <li>RSI Divergences: {divergence_count}/{total_patterns} ({divergence_count/total_patterns*100:.1f}%)</li>
        </ul>
        <p><strong>Current Market Analysis:</strong></p>
        <ul>
            <li>Recent Volatility: {df1['Volatility'].iloc[-5:].mean():.2f}% (5-period avg)</li>
            <li>Current RSI: {df1['RSI'].iloc[-1]:.2f}</li>
            <li>Price vs EMA20: {df1['Close'].iloc[-1] - df1['EMA20'].iloc[-1]:.2f} points</li>
        </ul>
        <p><strong>Warning Signals:</strong> {'⚠️ High volatility and RSI divergence detected - Exercise caution!' if df1['Volatility'].iloc[-1] > df1['Volatility'].quantile(0.8) and abs(df1['RSI'].iloc[-1] - 50) > 20 else '✅ Normal market conditions'}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No significant patterns detected with current threshold. Try lowering the threshold.")
    
    # ===========================================
    # SECTION 7: INTERACTIVE CHARTS
    # ===========================================
    st.header("📈 Interactive Charts")
    
    # Chart 1: Ticker 1 Price + RSI
    st.subheader(f"📊 {ticker1} - Price Action & RSI")
    
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker1} Price with EMAs', 'RSI Indicator')
    )
    
    # Candlestick
    fig1.add_trace(go.Candlestick(
        x=df1.index,
        open=df1['Open'],
        high=df1['High'],
        low=df1['Low'],
        close=df1['Close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['EMA20'], name='EMA20', line=dict(color='orange', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['EMA50'], name='EMA50', line=dict(color='blue', width=1)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['EMA200'], name='EMA200', line=dict(color='red', width=1)), row=1, col=1)
    
    # RSI
    fig1.add_trace(go.Scatter(x=df1.index, y=df1['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig1.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig1.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig1.update_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False)
    fig1.update_yaxes(title_text="Price", row=1, col=1)
    fig1.update_yaxes(title_text="RSI", row=2, col=1)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2 & 3: Ticker 2 and Ratio (if enabled)
    if enable_ratio and df2 is not None:
        st.subheader(f"📊 {ticker2} - Price Action & RSI")
        
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker2} Price with EMAs', 'RSI Indicator')
        )
        
        fig2.add_trace(go.Candlestick(
            x=df2.index,
            open=df2['Open'],
            high=df2['High'],
            low=df2['Low'],
            close=df2['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig2.add_trace(go.Scatter(x=df2.index, y=df2['EMA20'], name='EMA20', line=dict(color='orange', width=1)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df2.index, y=df2['EMA50'], name='EMA50', line=dict(color='blue', width=1)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df2.index, y=df2['EMA200'], name='EMA200', line=dict(color='red', width=1)), row=1, col=1)
        
        fig2.add_trace(go.Scatter(x=df2.index, y=df2['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig2.update_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False)
        fig2.update_yaxes(title_text="Price", row=1, col=1)
        fig2.update_yaxes(title_text="RSI", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Ratio Chart
        st.subheader(f"📊 Ratio ({ticker1}/{ticker2}) & RSI")
        
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Ratio Value', 'Ratio RSI')
        )
        
        fig3.add_trace(go.Scatter(x=df_ratio.index, y=df_ratio['Ratio'], name='Ratio', line=dict(color='teal', width=2)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df_ratio.index, y=df_ratio['RSI_Ratio'], name='Ratio RSI', line=dict(color='purple')), row=2, col=1)
        fig3.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig3.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig3.update_layout(height=700, showlegend=True)
        fig3.update_yaxes(title_text="Ratio", row=1, col=1)
        fig3.update_yaxes(title_text="RSI", row=2, col=1)
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # ===========================================
    # SECTION 8: STATISTICAL DISTRIBUTION ANALYSIS
    # ===========================================
    st.header("📊 Statistical Distribution Analysis")
    
    returns_data = df1['Returns'].dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns Histogram
        fig_hist1 = go.Figure()
        fig_hist1.add_trace(go.Histogram(
            x=returns_data,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='lightblue'
        ))
        fig_hist1.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns (%)',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig_hist1, use_container_width=True)
    
    with col2:
        # Returns with Normal Curve
        fig_hist2 = go.Figure()
        fig_hist2.add_trace(go.Histogram(
            x=returns_data,
            nbinsx=50,
            name='Returns',
            marker_color='lightblue',
            histnorm='probability density'
        ))
        
        # Normal curve overlay
        mu, std = returns_data.mean(), returns_data.std()
        x_range = np.linspace(returns_data.min(), returns_data.max(), 100)
        normal_curve = stats.norm.pdf(x_range, mu, std)
        fig_hist2.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig_hist2.update_layout(
            title='Returns with Normal Curve Overlay',
            xaxis_title='Returns (%)',
            yaxis_title='Density',
            height=400
        )
        st.plotly_chart(fig_hist2, use_container_width=True)
    
    # Bell Curve Visualization with Zones
    st.subheader("🔔 Normal Distribution Analysis with Color Zones")
    
    fig_bell = go.Figure()
    
    x_range = np.linspace(mu - 4*std, mu + 4*std, 1000)
    y_range = stats.norm.pdf(x_range, mu, std)
    
    # Color zones
    # Green zone: ±1 std dev (68%)
    green_mask = (x_range >= mu - std) & (x_range <= mu + std)
    # Yellow zone: ±2 std dev (95%)
    yellow_mask = ((x_range >= mu - 2*std) & (x_range < mu - std)) | ((x_range > mu + std) & (x_range <= mu + 2*std))
    # Red zone: beyond ±2 std dev
    red_mask = (x_range < mu - 2*std) | (x_range > mu + 2*std)
    
    fig_bell.add_trace(go.Scatter(x=x_range[green_mask], y=y_range[green_mask], 
                                   fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)',
                                   line=dict(color='green'), name='±1σ (68%)'))
    fig_bell.add_trace(go.Scatter(x=x_range[yellow_mask], y=y_range[yellow_mask],
                                   fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.3)',
                                   line=dict(color='orange'), name='±2σ (95%)'))
    fig_bell.add_trace(go.Scatter(x=x_range[red_mask], y=y_range[red_mask],
                                   fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                                   line=dict(color='red'), name='Beyond ±2σ'))
    
    # Current position marker
    current_return = df1['Returns'].iloc[-1]
    current_y = stats.norm.pdf(current_return, mu, std)
    fig_bell.add_trace(go.Scatter(x=[current_return], y=[current_y],
                                   mode='markers', marker=dict(size=15, color='black'),
                                   name='Current Position'))
    
    fig_bell.update_layout(
        title='Normal Distribution with Color-Coded Zones',
        xaxis_title='Returns (%)',
        yaxis_title='Probability Density',
        height=500
    )
    st.plotly_chart(fig_bell, use_container_width=True)
    
    # Z-Score Analysis
    st.subheader("📊 Z-Score Analysis")
    
    df1['Z_Score'] = (df1['Returns'] - mu) / std
    
    z_score_df = df1[['Returns', 'Returns_Points', 'Z_Score']].tail(30).copy()
    z_score_df.index = z_score_df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    def highlight_z_score(val):
        if abs(val) > 2:
            return 'background-color: #ffcccc'
        elif abs(val) > 1:
            return 'background-color: #ffffcc'
        else:
            return 'background-color: #ccffcc'
    
    st.dataframe(z_score_df.style.format({
        'Returns': '{:+.2f}%',
        'Returns_Points': '{:+.2f}',
        'Z_Score': '{:.2f}'
    }).applymap(highlight_z_score, subset=['Z_Score']), use_container_width=True)
    
    # Statistical Summary
    current_z = df1['Z_Score'].iloc[-1]
    percentile = stats.norm.cdf(current_z) * 100
    skewness = returns_data.skew()
    kurtosis = returns_data.kurtosis()
    
    st.markdown(f"""
    <div class="info-box">
    <h4>📈 Statistical Summary</h4>
    <p><strong>Distribution Characteristics:</strong></p>
    <ul>
        <li>Mean Return: {mu:.2f}%</li>
        <li>Standard Deviation: {std:.2f}%</li>
        <li>Skewness: {skewness:.2f} {'(Right-tailed)' if skewness > 0 else '(Left-tailed)' if skewness < 0 else '(Symmetric)'}</li>
        <li>Kurtosis: {kurtosis:.2f} {'(Fat-tailed)' if kurtosis > 0 else '(Thin-tailed)'}</li>
    </ul>
    <p><strong>Current Position:</strong></p>
    <ul>
        <li>Current Return: {current_return:.2f}%</li>
        <li>Z-Score: {current_z:.2f}</li>
        <li>Percentile Rank: {percentile:.1f}%</li>
        <li>Position: {'<span class="red-text">Extreme (>2σ)</span>' if abs(current_z) > 2 else '<span class="yellow-text">Moderate (1-2σ)</span>' if abs(current_z) > 1 else '<span class="green-text">Normal (±1σ)</span>'}</li>
