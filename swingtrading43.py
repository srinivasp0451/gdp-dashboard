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
        padding: 1.5rem;
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
                # Flatten multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Ensure standard column names
                data.columns = [col.strip().title() for col in data.columns]
                
                # Convert to IST
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
        df2['EMA50'] = calculate_ema(df2['Close'], 50)
        df2['EMA200'] = calculate_ema(df2['Close'], 200)
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
        st.metric(
            f"{ticker1} Price",
            f"₹{current_price1:.2f}",
            f"{change1:+.2f} ({change_pct1:+.2f}%)"
        )
    
    if df2 is not None and not df2.empty:
        current_price2 = df2['Close'].iloc[-1]
        prev_price2 = df2['Close'].iloc[0]
        change2 = current_price2 - prev_price2
        change_pct2 = (change2 / prev_price2) * 100
        
        with col2:
            st.metric(
                f"{ticker2} Price",
                f"₹{current_price2:.2f}",
                f"{change2:+.2f} ({change_pct2:+.2f}%)"
            )
        
        with col3:
            ratio = current_price1 / current_price2
            st.metric("Current Ratio", f"{ratio:.4f}", "T1/T2")
        
        with col4:
            st.metric("Data Points", len(df1), f"{interval} interval")
    else:
        with col2:
            st.metric("Data Points", len(df1), f"{interval} interval")
        with col3:
            rsi_current = df1['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
            st.metric("RSI Status", rsi_status, f"{rsi_current:.2f}")
    
    # Data Table
    st.subheader("📋 Recent Price Data")
    display_df1 = df1[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Returns_Points', 'RSI']].tail(20).copy()
    display_df1.index = display_df1.index.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_df1, use_container_width=True)
    
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
            
            divergence = "No"
            price_change = (lookback['Close'].iloc[-1] - lookback['Close'].iloc[0]) / lookback['Close'].iloc[0] * 100
            rsi_change = rsi_at_move - rsi_before
            
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
            
            # Correlation
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
        st.dataframe(df_pattern_results, use_container_width=True)
        
        # Pattern Summary
        total_patterns = len(pattern_results)
        vol_burst_count = sum(1 for p in pattern_results if p['Volatility_Burst'] == 'Yes')
        volume_spike_count = sum(1 for p in pattern_results if p['Volume_Spike'] == 'Yes')
        divergence_count = sum(1 for p in pattern_results if p['RSI_Divergence'] != 'No')
        
        st.write(f"**🎯 Pattern Detection Summary**")
        st.write(f"- Total Significant Moves Detected: **{total_patterns}**")
        st.write(f"- Volatility Bursts: **{vol_burst_count}/{total_patterns}** ({vol_burst_count/total_patterns*100:.1f}%)")
        st.write(f"- Volume Spikes: **{volume_spike_count}/{total_patterns}** ({volume_spike_count/total_patterns*100:.1f}%)")
        st.write(f"- RSI Divergences: **{divergence_count}/{total_patterns}** ({divergence_count/total_patterns*100:.1f}%)")
    else:
        st.info("No significant patterns detected with current threshold. Try lowering the threshold.")
    
    # ===========================================
    # SECTION 9: FINAL TRADING RECOMMENDATION
    # ===========================================
    st.header("🎯 AI-Powered Trading Recommendation")
    
    # Gather all analysis data
    current_price = df1['Close'].iloc[-1]
    current_rsi = df1['RSI'].iloc[-1]
    current_vol = df1['Volatility'].iloc[-1]
    ema20 = df1['EMA20'].iloc[-1]
    ema50 = df1['EMA50'].iloc[-1]
    ema200 = df1['EMA200'].iloc[-1]
    
    # Support and Resistance
    support_levels, resistance_levels = find_support_resistance(df1['Close'])
    
    # Fibonacci levels
    fib_levels = calculate_fibonacci_levels(df1['High'].max(), df1['Low'].min())
    
    # Price action analysis
    last_5_candles = df1.tail(5)
    bullish_candles = int((last_5_candles['Close'] > last_5_candles['Open']).sum())
    bearish_candles = 5 - bullish_candles
    
    # Volume analysis
    volume_increasing = False
    if 'Volume' in df1.columns and not df1['Volume'].empty:
        vol_mean = df1['Volume'].tail(20).mean()
        vol_current = df1['Volume'].iloc[-1]
        if vol_mean > 0 and vol_current > 0:
            volume_increasing = vol_current > vol_mean * 1.5
    
    # EMA alignment
    emas_aligned_bullish = ema20 > ema50 > ema200
    emas_aligned_bearish = ema20 < ema50 < ema200
    
    # Price position
    above_ema20 = current_price > ema20
    above_ema50 = current_price > ema50
    
    # RSI conditions
    rsi_oversold = current_rsi < 30
    rsi_overbought = current_rsi > 70
    
    # Divergence check
    divergence_signal = detect_divergence(df1['Close'].tail(10), df1['RSI'].tail(10))
    
    # Historical pattern accuracy
    if pattern_results:
        successful_patterns = sum(1 for p in pattern_results if 
                                 (p['Direction'] == 'Up' and p['RSI_At_Move'] < 50) or
                                 (p['Direction'] == 'Down' and p['RSI_At_Move'] > 50))
        pattern_accuracy = (successful_patterns / len(pattern_results)) * 100
    else:
        successful_patterns = 0
        pattern_accuracy = 0
    
    # Calculate signal strength
    signal_strength = 0
    signals_list = []
    
    # Bullish signals
    if above_ema20:
        signal_strength += 1
        signals_list.append("✅ Price above EMA20")
    if above_ema50:
        signal_strength += 1
        signals_list.append("✅ Price above EMA50")
    if emas_aligned_bullish:
        signal_strength += 2
        signals_list.append("✅ EMAs aligned bullish")
    if rsi_oversold:
        signal_strength += 2
        signals_list.append("✅ RSI oversold - bounce opportunity")
    if divergence_signal == "Bullish Divergence":
        signal_strength += 2
        signals_list.append("✅ Bullish RSI divergence detected")
    if volume_increasing and bullish_candles > 3:
        signal_strength += 2
        signals_list.append("✅ Volume breakout with bullish momentum")
    
    # Bearish signals
    if not above_ema20:
        signal_strength -= 1
        signals_list.append("❌ Price below EMA20")
    if not above_ema50:
        signal_strength -= 1
        signals_list.append("❌ Price below EMA50")
    if emas_aligned_bearish:
        signal_strength -= 2
        signals_list.append("❌ EMAs aligned bearish")
    if rsi_overbought:
        signal_strength -= 2
        signals_list.append("❌ RSI overbought - correction likely")
    if divergence_signal == "Bearish Divergence":
        signal_strength -= 2
        signals_list.append("❌ Bearish RSI divergence detected")
    
    # Determine recommendation
    if signal_strength >= 4:
        recommendation = "🟢 STRONG BUY"
        confidence = "High"
    elif signal_strength >= 2:
        recommendation = "🟢 BUY"
        confidence = "Moderate"
    elif signal_strength <= -4:
        recommendation = "🔴 STRONG SELL"
        confidence = "High"
    elif signal_strength <= -2:
        recommendation = "🔴 SELL"
        confidence = "Moderate"
    else:
        recommendation = "🟡 HOLD / WAIT"
        confidence = "Low"
    
    # Calculate entry, stop loss, and targets
    if signal_strength > 0:  # Bullish scenario
        entry_price = current_price
        pullback_entry = ema20 if current_price > ema20 else current_price * 0.98
        
        if support_levels:
            stop_loss = min(support_levels)
        else:
            stop_loss = entry_price * 0.98
        
        if resistance_levels:
            target1 = resistance_levels[0] if len(resistance_levels) > 0 else entry_price * 1.02
            target2 = resistance_levels[1] if len(resistance_levels) > 1 else entry_price * 1.04
            target3 = resistance_levels[2] if len(resistance_levels) > 2 else entry_price * 1.06
        else:
            target1 = entry_price * 1.02
            target2 = entry_price * 1.04
            target3 = entry_price * 1.06
        
        trailing_stop = "Move to breakeven after Target 1. Trail below EMA20."
        
    else:  # Bearish scenario
        entry_price = current_price
        pullback_entry = ema20 if current_price < ema20 else current_price * 1.02
        
        if resistance_levels:
            stop_loss = max(resistance_levels)
        else:
            stop_loss = entry_price * 1.02
        
        if support_levels:
            target1 = support_levels[0] if len(support_levels) > 0 else entry_price * 0.98
            target2 = support_levels[1] if len(support_levels) > 1 else entry_price * 0.96
            target3 = support_levels[2] if len(support_levels) > 2 else entry_price * 0.94
        else:
            target1 = entry_price * 0.98
            target2 = entry_price * 0.96
            target3 = entry_price * 0.94
        
        trailing_stop = "Move to breakeven after Target 1. Trail above EMA20."
    
    # Risk-reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(target1 - entry_price)
    risk_reward = reward / risk if risk > 0 else 0
    
    # Display recommendation
    st.subheader(f"{recommendation}")
    st.write(f"**Confidence Level:** {confidence}")
    st.write(f"**Signal Strength:** {signal_strength}/10")
    
    # Trading Plan
    st.subheader("📋 Detailed Trading Plan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🎯 Entry Strategy**")
        st.write(f"- Primary Entry: ₹{entry_price:.2f}")
        st.write(f"- Optimal Entry: ₹{pullback_entry:.2f}")
        st.write(f"- Risk per trade: 1-2% of capital")
    
    with col2:
        st.write("**🛑 Risk Management**")
        st.write(f"- Stop Loss: ₹{stop_loss:.2f}")
        st.write(f"- Risk: ₹{risk:.2f} ({(risk/entry_price)*100:.2f}%)")
        st.write(f"- Risk:Reward: 1:{risk_reward:.2f}")
    
    with col3:
        st.write("**🎯 Profit Targets**")
        st.write(f"- Target 1: ₹{target1:.2f} (Book 30%)")
        st.write(f"- Target 2: ₹{target2:.2f} (Book 40%)")
        st.write(f"- Target 3: ₹{target3:.2f} (Book 30%)")
        st.write(f"- Trail: {trailing_stop}")
    
    # Key Levels
    st.subheader("📊 Key Technical Levels")
    st.write(f"**Support Levels:** {', '.join([f'₹{s:.2f}' for s in support_levels]) if support_levels else 'Not detected'}")
    st.write(f"**Resistance Levels:** {', '.join([f'₹{r:.2f}' for r in resistance_levels]) if resistance_levels else 'Not detected'}")
    st.write(f"**Fibonacci 50%:** ₹{fib_levels['50.0%']:.2f}")
    st.write(f"**Fibonacci 61.8%:** ₹{fib_levels['61.8%']:.2f}")
    st.write(f"**EMA20:** ₹{ema20:.2f}")
    st.write(f"**EMA50:** ₹{ema50:.2f}")
    st.write(f"**EMA200:** ₹{ema200:.2f}")
    
    # Signal Summary
    st.subheader("📊 Signal Summary")
    for signal in signals_list:
        st.write(signal)
    
    # Detailed Reasoning
    st.subheader("🧠 Detailed Analysis & Reasoning")
    
    st.write("**Technical Setup:**")
    if above_ema20 and above_ema50:
        st.write(f"Strong bullish structure with price at ₹{current_price:.2f} trading above key EMAs (EMA20: ₹{ema20:.2f}, EMA50: ₹{ema50:.2f})")
    elif not above_ema20 and not above_ema50:
        st.write(f"Bearish structure with price at ₹{current_price:.2f} below key EMAs (EMA20: ₹{ema20:.2f}, EMA50: ₹{ema50:.2f})")
    else:
        st.write(f"Mixed structure. Price at ₹{current_price:.2f} testing critical levels")
    
    st.write("")
    st.write("**Momentum Analysis:**")
    st.write(f"RSI at {current_rsi:.2f} - ", end="")
    if rsi_oversold:
        st.write("Oversold conditions indicate high probability bounce")
    elif rsi_overbought:
        st.write("Overbought conditions suggest correction expected")
    else:
        st.write("Neutral zone indicates trend continuation likely")
    
    if divergence_signal != "No Divergence":
        st.write(f"⚠️ **{divergence_signal}** detected - strong reversal signal")
    
    st.write("")
    st.write("**Price Action:**")
    st.write(f"Last 5 candles show {bullish_candles} bullish vs {bearish_candles} bearish")
    if bullish_candles >= 4:
        st.write("Strong bullish momentum building")
    elif bearish_candles >= 4:
        st.write("Selling pressure evident")
    
    st.write("")
    st.write("**Volatility Context:**")
    st.write(f"Current volatility at {current_vol:.2f}% - ", end="")
    if current_vol > df1['Volatility'].quantile(0.75):
        st.write("Elevated (expect larger swings, adjust position size)")
    else:
        st.write("Normal (standard position sizing applicable)")
    
    # ===========================================
    # COMPREHENSIVE BACKTESTING RESULTS
    # ===========================================
    st.header("📈 Comprehensive Backtesting Results")
    
    if pattern_results and len(pattern_results) > 0:
        
        # Calculate detailed statistics
        up_moves = [p for p in pattern_results if p['Direction'] == 'Up']
        down_moves = [p for p in pattern_results if p['Direction'] == 'Down']
        
        # Volatility burst correlation
        vol_burst_moves = [p for p in pattern_results if p['Volatility_Burst'] == 'Yes']
        vol_burst_success = sum(1 for p in vol_burst_moves if 
                               (p['Direction'] == 'Up' and p['RSI_At_Move'] < 50) or
                               (p['Direction'] == 'Down' and p['RSI_At_Move'] > 50))
        vol_burst_accuracy = (vol_burst_success / len(vol_burst_moves) * 100) if vol_burst_moves else 0
        
        # RSI divergence correlation
        divergence_moves = [p for p in pattern_results if p['RSI_Divergence'] != 'No']
        div_success = sum(1 for p in divergence_moves if 
                         (p['Direction'] == 'Up' and p['RSI_Divergence'] == 'Bullish') or
                         (p['Direction'] == 'Down' and p['RSI_Divergence'] == 'Bearish'))
        div_accuracy = (div_success / len(divergence_moves) * 100) if divergence_moves else 0
        
        # EMA crossover correlation
        ema_cross_moves = [p for p in pattern_results if p['EMA_20/50_Cross'] == 'Yes']
        ema_success = sum(1 for p in ema_cross_moves if abs(p['Move_Points']) > pattern_threshold * 0.8)
        ema_accuracy = (ema_success / len(ema_cross_moves) * 100) if ema_cross_moves else 0
        
        # Current market similarity score
        current_pattern_score = 0
        current_volatility = df1['Volatility'].iloc[-1]
        
        if current_volatility > df1['Volatility'].quantile(0.75):
            current_pattern_score += 2
        
        if divergence_signal != "No Divergence":
            current_pattern_score += 3
        
        if volume_increasing:
            current_pattern_score += 2
        
        # Find most similar historical patterns
        similar_patterns = []
        for p in pattern_results:
            similarity = 0
            if abs(p['RSI_At_Move'] - current_rsi) < 10:
                similarity += 3
            if p['Volatility_Burst'] == 'Yes' and current_volatility > df1['Volatility'].quantile(0.75):
                similarity += 2
            if p['RSI_Divergence'] == divergence_signal:
                similarity += 3
            
            if similarity > 4:
                similar_patterns.append((p, similarity))
        
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Display Overall Statistics
        st.subheader("📊 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patterns", len(pattern_results))
        with col2:
            st.metric("Successful", successful_patterns)
        with col3:
            st.metric("Accuracy", f"{pattern_accuracy:.1f}%")
        with col4:
            avg_move = np.mean([p['Move_Points'] for p in pattern_results])
            st.metric("Avg Move", f"{avg_move:+.2f} pts")
        
        st.write("")
        st.write(f"**Direction Distribution:**")
        st.write(f"- Upward Moves: {len(up_moves)} ({len(up_moves)/len(pattern_results)*100:.1f}%)")
        st.write(f"- Downward Moves: {len(down_moves)} ({len(down_moves)/len(pattern_results)*100:.1f}%)")
        
        # Pattern-Specific Accuracy
        st.subheader("🎯 Pattern-Specific Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Volatility Burst",
                f"{vol_burst_accuracy:.1f}%",
                f"{len(vol_burst_moves)} patterns"
            )
        
        with col2:
            st.metric(
                "RSI Divergence",
                f"{div_accuracy:.1f}%",
                f"{len(divergence_moves)} patterns"
            )
        
        with col3:
            st.metric(
                "EMA Crossover",
                f"{ema_accuracy:.1f}%",
                f"{len(ema_cross_moves)} patterns"
            )
        
        # Current Market Similarity
        st.subheader("🔍 Current Market Similarity Analysis")
        
        st.write(f"**Pattern Match Score:** {current_pattern_score}/10")
        st.write(f"**Similar Historical Setups Found:** {len(similar_patterns)}")
        
        if similar_patterns:
            top_5 = similar_patterns[:5]
            avg_similar_move = np.mean([p[0]['Move_Points'] for p in top_5])
            st.write(f"**Average Move in Similar Setups:** {avg_similar_move:+.2f} points")
            
            direction_consistency = sum(1 for p in top_5 if np.sign(p[0]['Move_Points']) == np.sign(avg_similar_move))
            st.write(f"**Direction Consistency:** {direction_consistency}/{len(top_5)}")
        
        # Optimization Insights
        st.subheader("💡 Optimization Insights")
        
        if vol_burst_accuracy > pattern_accuracy:
            improvement = vol_burst_accuracy - pattern_accuracy
            current_vol_status = "HIGH" if current_volatility > df1['Volatility'].quantile(0.75) else "NORMAL"
            st.write(f"✅ Volatility burst patterns show **{improvement:.1f}%** higher accuracy")
            st.write(f"   Current volatility is **{current_vol_status}**")
        
        if div_accuracy > pattern_accuracy:
            improvement = div_accuracy - pattern_accuracy
            st.write(f"✅ RSI divergence patterns show **{improvement:.1f}%** higher accuracy")
            st.write(f"   Current divergence: **{divergence_signal}**")
        
        if ema_accuracy > pattern_accuracy:
            improvement = ema_accuracy - pattern_accuracy
            st.write(f"✅ EMA crossover patterns show **{improvement:.1f}%** higher accuracy")
        
        # Recommendation Confidence
        st.subheader("🎯 Recommendation Confidence")
        
        confidence_score = pattern_accuracy
        
        if current_pattern_score >= 6:
            confidence_score += 15
            st.success("🟢 **HIGH CONFIDENCE:** Current market setup closely matches historically successful patterns")
        elif current_pattern_score >= 4:
            confidence_score += 10
            st.warning("🟡 **MODERATE CONFIDENCE:** Current setup shows some similarity to historical patterns")
        else:
            confidence_score += 5
            st.info("🟠 **LOWER CONFIDENCE:** Current setup is unique. Exercise caution")
        
        st.write(f"**Adjusted Confidence Score:** {min(confidence_score, 95):.1f}%")
        
        # Most Similar Historical Setups
        if similar_patterns:
            st.subheader("📜 Most Similar Historical Setups")
            
            for i, (pattern, sim_score) in enumerate(similar_patterns[:5], 1):
                direction_emoji = "📈" if pattern['Direction'] == 'Up' else "📉"
                st.write(f"{i}. {direction_emoji} **{pattern['DateTime']}**: Moved **{pattern['Move_Points']:+.2f}** points ({pattern['Move_%']:+.2f}%) - Similarity: **{sim_score}/10**")
        
        # Expected Outcome
        st.subheader("🔮 Expected Outcome Based on History")
        
        if similar_patterns and len(similar_patterns) >= 3:
            expected_direction = "UP" if avg_similar_move > 0 else "DOWN"
            expected_magnitude = abs(avg_similar_move)
            confidence_pct = (sum(1 for p in similar_patterns[:5] if np.sign(p[0]['Move_Points']) == np.sign(avg_similar_move)) / min(len(similar_patterns), 5)) * 100
            
            st.success(f"Based on **{len(similar_patterns)}** similar historical patterns, expect a **{expected_direction}** move of approximately **{expected_magnitude:.2f} points** with **{confidence_pct:.0f}%** directional confidence")
        else:
            st.info("Insufficient similar patterns for precise prediction. Rely on technical analysis and risk management")
        
        # Detailed Pattern Performance Table
        st.subheader("📋 Detailed Pattern Performance Table")
        
        backtest_df = pd.DataFrame(pattern_results)
        backtest_df['Success'] = backtest_df.apply(
            lambda x: 'Yes' if (x['Direction'] == 'Up' and x['RSI_At_Move'] < 50) or 
                              (x['Direction'] == 'Down' and x['RSI_At_Move'] > 50) else 'No',
            axis=1
        )
        
        st.dataframe(backtest_df, use_container_width=True)
        
    else:
        st.warning("⚠️ No significant patterns detected for backtesting. Lower the pattern threshold in settings to detect more patterns")
    
    # Final Notes
    st.markdown("---")
    st.subheader("⚠️ Important Notes")
    st.write("- This is an algorithmic recommendation based on technical analysis - not financial advice")
    st.write("- Always manage risk and never risk more than 1-2% of capital per trade")
    st.write("- Market conditions can change rapidly - stay nimble and follow your trading plan")
    st.write("- Use stop losses religiously - they are your insurance against catastrophic loss")
    st.write("- Consider broader market context and news events before executing")
    st.write("- Paper trade this strategy first to validate its effectiveness for your style")

else:
    # Show welcome message when no data
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Fetch Data' to begin")
    
    st.markdown("""
    ## 🎯 Welcome to the Professional Algo Trading Dashboard
    
    This comprehensive trading platform provides:
    
    - ✅ **Multi-Asset Support**: Analyze stocks, indices, crypto, commodities, and forex
    - ✅ **Multi-Timeframe Analysis**: From 1-minute to monthly charts
    - ✅ **Advanced Indicators**: RSI, EMAs, SMAs, Fibonacci, Support/Resistance
    - ✅ **Pattern Recognition**: Detect 10+ technical patterns automatically
    - ✅ **Statistical Analysis**: Z-scores, distributions, volatility bins
    - ✅ **AI-Powered Recommendations**: Get actionable BUY/SELL/HOLD signals
    - ✅ **Comprehensive Backtesting**: Historical pattern performance analysis
    - ✅ **Risk Management**: Calculated entry, stop-loss, and target levels
    
    ### 🚀 Getting Started
    
    1. Select your ticker(s) from the sidebar
    2. Choose timeframe and period
    3. Optionally enable ratio analysis for pair trading
    4. Click "Fetch Data" to load and analyze
    5. Review the comprehensive analysis and trading recommendation
    
    **Ready to start? Configure your settings and fetch data!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Professional Algo Trading Dashboard v1.0 | Built with Streamlit & Python</p>
    <p>⚠️ For Educational Purposes Only - Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)
