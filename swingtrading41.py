import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Pro Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def safe_format_number(value, decimals=2):
    """Safely format numbers handling NaN and infinity"""
    try:
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except:
        return "N/A"

def safe_percentage(current, previous):
    """Safely calculate percentage change"""
    try:
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    except:
        return 0.0

def convert_to_ist(df):
    """Convert timezone-aware datetime to IST"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(ist)
        # Remove timezone info for clean display
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.warning(f"Timezone conversion warning: {str(e)}")
        return df

def calculate_rsi(data, period=14):
    """Calculate RSI manually with error handling"""
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Prevent division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    except Exception as e:
        return pd.Series([50] * len(data), index=data.index)

def calculate_ema(data, period):
    """Calculate EMA manually"""
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return data

def calculate_sma(data, period):
    """Calculate SMA manually"""
    try:
        return data.rolling(window=period).mean()
    except:
        return data

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels"""
    try:
        if len(data) < window:
            return data['Low'].min(), data['High'].max()
        highs = data['High'].rolling(window=window).max()
        lows = data['Low'].rolling(window=window).min()
        return lows.iloc[-1], highs.iloc[-1]
    except:
        return data['Low'].min(), data['High'].max()

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    try:
        diff = high - low
        return {
            '0.0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50.0%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '100.0%': low
        }
    except:
        return {'50.0%': (high + low) / 2}

def fetch_data_with_delay(ticker, interval, period, delay=1.5):
    """Fetch data with delay to respect API limits"""
    time.sleep(delay)
    try:
        # Use Ticker object for more reliable data fetching
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(interval=interval, period=period)
        
        if not data.empty:
            data = convert_to_ist(data)
        return data
    except Exception as e:
        # Fallback to download method
        try:
            data = yf.download(ticker, interval=interval, period=period, progress=False)
            if not data.empty:
                data = convert_to_ist(data)
            return data
        except Exception as e2:
            st.error(f"Error fetching {ticker} for {interval}/{period}: {str(e2)}")
            return pd.DataFrame()

def analyze_timeframe(data, timeframe_name):
    """Analyze a single timeframe with comprehensive metrics"""
    try:
        if data.empty or len(data) < 20:
            return None
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Ensure we have enough data
        min_length = min(200, len(data))
        
        # Trend
        if len(close) > 0:
            trend = "Up" if close.iloc[-1] > close.iloc[0] else "Down"
        else:
            trend = "Neutral"
        
        # Price metrics
        max_close = close.max()
        min_close = close.min()
        current_close = close.iloc[-1]
        
        # Fibonacci
        fib_levels = calculate_fibonacci_levels(max_close, min_close)
        fib_50 = fib_levels.get('50.0%', (max_close + min_close) / 2)
        
        # Volatility (standard deviation)
        volatility = close.std() if len(close) > 1 else 0
        
        # Returns
        if len(close) > 0 and close.iloc[0] != 0:
            pct_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
            points_change = close.iloc[-1] - close.iloc[0]
        else:
            pct_change = 0
            points_change = 0
        
        # Support and Resistance
        support, resistance = calculate_support_resistance(data)
        
        # RSI
        rsi = calculate_rsi(close)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        if current_rsi < 30:
            rsi_status = "Oversold"
            rsi_color = "🟢"
        elif current_rsi > 70:
            rsi_status = "Overbought"
            rsi_color = "🔴"
        else:
            rsi_status = "Neutral"
            rsi_color = "🟡"
        
        # EMAs - with safe calculation
        def safe_ema(data, period):
            if len(data) >= period:
                return calculate_ema(data, period).iloc[-1]
            return data.iloc[-1] if len(data) > 0 else np.nan
        
        ema_9 = safe_ema(close, 9)
        ema_20 = safe_ema(close, 20)
        ema_21 = safe_ema(close, 21)
        ema_33 = safe_ema(close, 33)
        ema_50 = safe_ema(close, 50)
        ema_100 = safe_ema(close, 100)
        ema_150 = safe_ema(close, 150)
        ema_200 = safe_ema(close, 200)
        
        # SMAs - with safe calculation
        def safe_sma(data, period):
            if len(data) >= period:
                return calculate_sma(data, period).iloc[-1]
            return data.iloc[-1] if len(data) > 0 else np.nan
        
        sma_20 = safe_sma(close, 20)
        sma_50 = safe_sma(close, 50)
        sma_100 = safe_sma(close, 100)
        sma_150 = safe_sma(close, 150)
        sma_200 = safe_sma(close, 200)
        
        return {
            'Timeframe': timeframe_name,
            'Trend': trend,
            'Max': safe_format_number(max_close),
            'Min': safe_format_number(min_close),
            'Fib 50%': safe_format_number(fib_50),
            'Volatility': safe_format_number(volatility),
            '% Change': pct_change,
            'Points': points_change,
            'Support': safe_format_number(support),
            'Resistance': safe_format_number(resistance),
            'RSI': safe_format_number(current_rsi),
            'RSI Status': f"{rsi_color} {rsi_status}",
            '9 EMA': safe_format_number(ema_9),
            '20 EMA': safe_format_number(ema_20),
            '21 EMA': safe_format_number(ema_21),
            '33 EMA': safe_format_number(ema_33),
            '50 EMA': safe_format_number(ema_50),
            '100 EMA': safe_format_number(ema_100),
            '150 EMA': safe_format_number(ema_150),
            '200 EMA': safe_format_number(ema_200),
            'vs 20 EMA': '🟢 Above' if current_close > ema_20 else '🔴 Below',
            'vs 50 EMA': '🟢 Above' if current_close > ema_50 else '🔴 Below',
            'vs 100 EMA': '🟢 Above' if current_close > ema_100 else '🔴 Below',
            'vs 150 EMA': '🟢 Above' if current_close > ema_150 else '🔴 Below',
            'vs 200 EMA': '🟢 Above' if current_close > ema_200 else '🔴 Below',
            '20 SMA': safe_format_number(sma_20),
            '50 SMA': safe_format_number(sma_50),
            '100 SMA': safe_format_number(sma_100),
            '150 SMA': safe_format_number(sma_150),
            '200 SMA': safe_format_number(sma_200),
            'vs 20 SMA': '🟢 Above' if current_close > sma_20 else '🔴 Below',
            'vs 50 SMA': '🟢 Above' if current_close > sma_50 else '🔴 Below',
            'vs 100 SMA': '🟢 Above' if current_close > sma_100 else '🔴 Below',
            'vs 150 SMA': '🟢 Above' if current_close > sma_150 else '🔴 Below',
            'vs 200 SMA': '🟢 Above' if current_close > sma_200 else '🔴 Below',
        }
    except Exception as e:
        st.warning(f"Error analyzing timeframe {timeframe_name}: {str(e)}")
        return None

def detect_patterns(data, threshold=30):
    """Detect significant price movements and preceding patterns"""
    try:
        patterns = []
        close = data['Close'].values
        
        if len(close) < 15:
            return patterns
        
        # Find significant moves
        for i in range(10, len(close)):
            move = close[i] - close[i-10]
            
            if abs(move) >= threshold:
                # Analyze preceding 10 candles
                preceding = data.iloc[i-10:i]
                
                volatility_burst = preceding['Close'].std() > data['Close'].std()
                rsi_vals = calculate_rsi(data['Close'].iloc[:i])
                rsi_divergence = False
                
                if len(rsi_vals) >= 10:
                    rsi_divergence = (rsi_vals.iloc[-1] > rsi_vals.iloc[-10] and close[i] < close[i-10]) or \
                                    (rsi_vals.iloc[-1] < rsi_vals.iloc[-10] and close[i] > close[i-10])
                
                patterns.append({
                    'Index': i,
                    'DateTime': data.index[i],
                    'Move': move,
                    'Direction': 'Up' if move > 0 else 'Down',
                    'Volatility_Burst': volatility_burst,
                    'RSI_Divergence': rsi_divergence
                })
        
        return patterns
    except Exception as e:
        return []

# Title
st.markdown('<h1 class="main-header">🚀 PRO ALGO TRADING DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Panel")

# Ticker Selection
ticker_options = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom": "Custom"
}

st.sidebar.subheader("📊 Ticker Selection")
ticker1_name = st.sidebar.selectbox("Select Ticker 1", list(ticker_options.keys()), key="ticker1_select")
if ticker1_name == "Custom":
    ticker1 = st.sidebar.text_input("Enter Ticker 1 Symbol", "AAPL", key="ticker1_input")
else:
    ticker1 = ticker_options[ticker1_name]

# Ratio Analysis Option
include_ratio = st.sidebar.checkbox("📈 Include Ratio Analysis (Ticker 2)", value=False, key="include_ratio")

ticker2 = None
ticker2_name = None
if include_ratio:
    ticker2_name = st.sidebar.selectbox("Select Ticker 2", list(ticker_options.keys()), index=1, key="ticker2_select")
    if ticker2_name == "Custom":
        ticker2 = st.sidebar.text_input("Enter Ticker 2 Symbol", "MSFT", key="ticker2_input")
    else:
        ticker2 = ticker_options[ticker2_name]

st.sidebar.markdown("---")

# Timeframe and Period
st.sidebar.subheader("⏰ Time Settings")
interval = st.sidebar.selectbox(
    "Select Interval",
    ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
    index=4,
    key="interval_select"
)

period = st.sidebar.selectbox(
    "Select Period",
    ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
    index=6,
    key="period_select"
)

st.sidebar.markdown("---")

# Advanced Options
st.sidebar.subheader("🎛️ Advanced Settings")
pattern_threshold = st.sidebar.slider("Pattern Detection Threshold (Points)", 10, 100, 30, key="pattern_threshold")
api_delay = st.sidebar.slider("API Delay (seconds)", 1.0, 3.0, 1.5, 0.5, key="api_delay")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ticker1_data' not in st.session_state:
    st.session_state.ticker1_data = None
if 'ticker2_data' not in st.session_state:
    st.session_state.ticker2_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Fetch Data Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 FETCH DATA & ANALYZE", type="primary", use_container_width=True):
    with st.spinner("⏳ Fetching data... Please wait."):
        try:
            st.session_state.ticker1_data = fetch_data_with_delay(ticker1, interval, period, delay=api_delay)
            
            if include_ratio and ticker2:
                st.session_state.ticker2_data = fetch_data_with_delay(ticker2, interval, period, delay=api_delay)
            else:
                st.session_state.ticker2_data = None
            
            st.session_state.data_fetched = True
            st.session_state.analysis_complete = True
            st.sidebar.success("✅ Data fetched successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.session_state.data_fetched = False

# Main Analysis Section
if st.session_state.data_fetched and st.session_state.ticker1_data is not None:
    data1 = st.session_state.ticker1_data
    data2 = st.session_state.ticker2_data
    
    if not data1.empty:
        # Extract safe values
        try:
            current_price_1 = float(data1['Close'].iloc[-1])
            first_price_1 = float(data1['Close'].iloc[0])
            pct_change_1 = safe_percentage(current_price_1, first_price_1)
        except:
            current_price_1 = 0
            pct_change_1 = 0
        
        # Section 1: Key Metrics Dashboard
        st.header("📊 Market Overview")
        
        if include_ratio and data2 is not None and not data2.empty:
            try:
                current_price_2 = float(data2['Close'].iloc[-1])
                first_price_2 = float(data2['Close'].iloc[0])
                pct_change_2 = safe_percentage(current_price_2, first_price_2)
                
                ratio_value = current_price_1 / current_price_2 if current_price_2 != 0 else 0
                first_ratio = float(data1['Close'].iloc[0]) / float(data2['Close'].iloc[0]) if float(data2['Close'].iloc[0]) != 0 else 0
                ratio_pct_change = safe_percentage(ratio_value, first_ratio)
            except:
                current_price_2 = 0
                pct_change_2 = 0
                ratio_value = 0
                ratio_pct_change = 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"{ticker1} Current Price",
                    safe_format_number(current_price_1),
                    f"{safe_format_number(pct_change_1)}%"
                )
            with col2:
                st.metric(
                    f"{ticker2} Current Price",
                    safe_format_number(current_price_2),
                    f"{safe_format_number(pct_change_2)}%"
                )
            with col3:
                st.metric(
                    "Ratio (T1/T2)",
                    safe_format_number(ratio_value, 4),
                    f"{safe_format_number(ratio_pct_change)}%"
                )
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"{ticker1} Current Price",
                    safe_format_number(current_price_1),
                    f"{safe_format_number(pct_change_1)}%"
                )
            with col2:
                rsi_1 = calculate_rsi(data1['Close'])
                current_rsi_1 = float(rsi_1.iloc[-1]) if len(rsi_1) > 0 else 50
                st.metric("RSI", safe_format_number(current_rsi_1), 
                         "Oversold" if current_rsi_1 < 30 else "Overbought" if current_rsi_1 > 70 else "Neutral")
            with col3:
                volatility_1 = data1['Close'].std()
                st.metric("Volatility", safe_format_number(volatility_1))
        
        st.markdown("---")
        
        # Statistical Hypothesis Testing Section
        st.header("📊 Statistical Hypothesis Testing - Market Direction Forecast")
        
        st.markdown("""
        ### Question: Will the market move UP, DOWN, or remain NEUTRAL?
        
        We'll use statistical hypothesis testing to answer this question with **95% confidence (α = 0.05)**.
        """)
        
        try:
            from scipy import stats
            
            returns = data1['Close'].pct_change().dropna()
            returns_points = data1['Close'].diff().dropna()
            
            if len(returns) > 30:
                # Test 1: Will market move UP?
                st.subheader("📈 Test 1: Upward Movement Hypothesis")
                
                st.markdown("""
                **Null Hypothesis (H₀):** The market will NOT move up (mean return ≤ 0)  
                **Alternative Hypothesis (H₁):** The market WILL move up (mean return > 0)  
                **Significance Level:** α = 0.05 (95% confidence)
                """)
                
                # One-sample t-test (one-tailed, right)
                t_stat_up, p_value_up_two = stats.ttest_1samp(returns, 0)
                p_value_up = p_value_up_two / 2 if t_stat_up > 0 else 1 - (p_value_up_two / 2)
                
                mean_return = returns.mean() * 100
                std_return = returns.std() * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat_up:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_up:.4f}")
                
                if p_value_up < 0.05:
                    st.success(f"""
                    ✅ **REJECT NULL HYPOTHESIS** (p = {p_value_up:.4f} < 0.05)
                    
                    **Statistical Conclusion:** With 95% confidence, we can conclude that the market is **LIKELY TO MOVE UP**.
                    
                    **Evidence:** 
                    - Mean return of {mean_return:.3f}% is statistically significant
                    - T-statistic of {t_stat_up:.3f} indicates upward bias
                    - Only {p_value_up*100:.2f}% probability this is due to chance
                    - Based on {len(returns)} observations over {interval}/{period}
                    
                    **Forecast:** **BULLISH** - Statistical evidence supports upward price movement
                    """)
                else:
                    st.warning(f"""
                    ❌ **FAIL TO REJECT NULL HYPOTHESIS** (p = {p_value_up:.4f} ≥ 0.05)
                    
                    **Statistical Conclusion:** Insufficient evidence to conclude market will move up.
                    
                    **Evidence:**
                    - Mean return of {mean_return:.3f}% is NOT statistically significant
                    - {p_value_up*100:.2f}% probability the observed pattern is due to chance
                    - Cannot confidently predict upward movement
                    """)
                
                st.markdown("---")
                
                # Test 2: Will market move DOWN?
                st.subheader("📉 Test 2: Downward Movement Hypothesis")
                
                st.markdown("""
                **Null Hypothesis (H₀):** The market will NOT move down (mean return ≥ 0)  
                **Alternative Hypothesis (H₁):** The market WILL move down (mean return < 0)  
                **Significance Level:** α = 0.05 (95% confidence)
                """)
                
                # One-sample t-test (one-tailed, left)
                p_value_down = p_value_up_two / 2 if t_stat_up < 0 else 1 - (p_value_up_two / 2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat_up:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_down:.4f}")
                
                if p_value_down < 0.05:
                    st.error(f"""
                    ✅ **REJECT NULL HYPOTHESIS** (p = {p_value_down:.4f} < 0.05)
                    
                    **Statistical Conclusion:** With 95% confidence, we can conclude that the market is **LIKELY TO MOVE DOWN**.
                    
                    **Evidence:**
                    - Mean return of {mean_return:.3f}% is statistically significant (negative)
                    - T-statistic of {t_stat_up:.3f} indicates downward bias
                    - Only {p_value_down*100:.2f}% probability this is due to chance
                    - Based on {len(returns)} observations over {interval}/{period}
                    
                    **Forecast:** **BEARISH** - Statistical evidence supports downward price movement
                    """)
                else:
                    st.warning(f"""
                    ❌ **FAIL TO REJECT NULL HYPOTHESIS** (p = {p_value_down:.4f} ≥ 0.05)
                    
                    **Statistical Conclusion:** Insufficient evidence to conclude market will move down.
                    
                    **Evidence:**
                    - Mean return of {mean_return:.3f}% is NOT statistically significant
                    - {p_value_down*100:.2f}% probability the observed pattern is due to chance
                    - Cannot confidently predict downward movement
                    """)
                
                st.markdown("---")
                
                # Test 3: Will market remain NEUTRAL?
                st.subheader("⚖️ Test 3: Neutral/Sideways Movement Hypothesis")
                
                st.markdown("""
                **Null Hypothesis (H₀):** The market IS in neutral range (mean return = 0)  
                **Alternative Hypothesis (H₁):** The market is NOT neutral (mean return ≠ 0)  
                **Significance Level:** α = 0.05 (95% confidence)
                """)
                
                # Two-tailed t-test
                p_value_neutral = p_value_up_two
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat_up:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_neutral:.4f}")
                
                if p_value_neutral >= 0.05:
                    st.info(f"""
                    ✅ **FAIL TO REJECT NULL HYPOTHESIS** (p = {p_value_neutral:.4f} ≥ 0.05)
                    
                    **Statistical Conclusion:** With 95% confidence, we can conclude that the market is **LIKELY TO REMAIN NEUTRAL**.
                    
                    **Evidence:**
                    - Mean return of {mean_return:.3f}% is NOT significantly different from zero
                    - {p_value_neutral*100:.2f}% probability supports neutral/sideways movement
                    - No strong directional bias detected
                    - Based on {len(returns)} observations over {interval}/{period}
                    
                    **Forecast:** **NEUTRAL/SIDEWAYS** - Market likely to consolidate in current range
                    """)
                else:
                    st.warning(f"""
                    ❌ **REJECT NULL HYPOTHESIS** (p = {p_value_neutral:.4f} < 0.05)
                    
                    **Statistical Conclusion:** The market is NOT neutral - it has a directional bias.
                    
                    **Evidence:**
                    - Mean return of {mean_return:.3f}% is statistically different from zero
                    - Only {p_value_neutral*100:.2f}% probability this is neutral movement
                    - Directional trend detected (see Tests 1 & 2 for direction)
                    """)
                
                st.markdown("---")
                
                # Final Statistical Summary
                st.subheader("🎯 Final Statistical Verdict")
                
                # Determine the strongest signal
                if p_value_up < 0.05 and p_value_up < p_value_down:
                    verdict = "📈 **STATISTICAL FORECAST: UPWARD MOVEMENT**"
                    verdict_color = "success"
                    confidence_pct = (1 - p_value_up) * 100
                    direction = "UP"
                    expected_move = mean_return
                elif p_value_down < 0.05 and p_value_down < p_value_up:
                    verdict = "📉 **STATISTICAL FORECAST: DOWNWARD MOVEMENT**"
                    verdict_color = "error"
                    confidence_pct = (1 - p_value_down) * 100
                    direction = "DOWN"
                    expected_move = mean_return
                elif p_value_neutral >= 0.05:
                    verdict = "⚖️ **STATISTICAL FORECAST: NEUTRAL/SIDEWAYS**"
                    verdict_color = "info"
                    confidence_pct = p_value_neutral * 100
                    direction = "SIDEWAYS"
                    expected_move = 0
                else:
                    verdict = "🟡 **STATISTICAL FORECAST: INCONCLUSIVE**"
                    verdict_color = "warning"
                    confidence_pct = 50
                    direction = "UNCERTAIN"
                    expected_move = mean_return
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
                    <h2 style="color: white; margin: 0;">{verdict}</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">Statistical Confidence: <strong>{confidence_pct:.1f}%</strong></p>
                    <p style="margin: 0;">Based on {len(returns)} observations from {interval}/{period} timeframe</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **Comprehensive Statistical Analysis:**
                
                **Methodology:**
                - Sample Size: {len(returns)} returns
                - Timeframe: {interval} interval over {period} period
                - Statistical Test: One-sample t-test
                - Confidence Level: 95% (α = 0.05)
                
                **Test Results:**
                1. **Upward Movement:** p-value = {p_value_up:.4f} {'✓ Significant' if p_value_up < 0.05 else '✗ Not Significant'}
                2. **Downward Movement:** p-value = {p_value_down:.4f} {'✓ Significant' if p_value_down < 0.05 else '✗ Not Significant'}
                3. **Neutral Movement:** p-value = {p_value_neutral:.4f} {'✓ Significant' if p_value_neutral >= 0.05 else '✗ Not Significant'}
                
                **Statistical Interpretation:**
                - Mean Return: {mean_return:.3f}% per period
                - Standard Deviation: {std_return:.3f}%
                - T-Statistic: {t_stat_up:.3f}
                - Direction Indicated: **{direction}**
                
                **Probability Interpretation:**
                - P < 0.05: Strong evidence (< 5% chance result is random)
                - P < 0.01: Very strong evidence (< 1% chance result is random)
                - P ≥ 0.05: Weak evidence (≥ 5% chance result is random)
                
                **Trading Implication:**
                {
                    f"The statistical evidence strongly supports **{direction}** movement. Expected return per period: {expected_move:.2f}%. This aligns with a confidence level of {confidence_pct:.1f}%."
                    if confidence_pct > 90
                    else f"The statistical evidence moderately supports **{direction}** movement. Expected return per period: {expected_move:.2f}%. Exercise caution as confidence is {confidence_pct:.1f}%."
                    if confidence_pct > 70
                    else f"The statistical evidence is weak. Market direction is **{direction}** but with low confidence ({confidence_pct:.1f}%). Consider waiting for clearer signals."
                }
                
                **Risk Disclaimer:** Past statistical patterns do not guarantee future results. Always use proper risk management and position sizing.
                """)
                
            else:
                st.warning("Insufficient data points for reliable statistical testing. Need at least 30 observations.")
                
        except ImportError:
            st.error("scipy library required for statistical testing. Install with: pip install scipy")
        except Exception as e:
            st.error(f"Error in statistical analysis: {str(e)}")
        
        st.markdown("---")
        
        # Multi-Timeframe Analysis for Ticker 2 (if ratio enabled)
        if include_ratio and data2 is not None and not data2.empty:
            st.header(f"📈 Multi-Timeframe Analysis - {ticker2}")
            
            analysis_results_t2 = []
            
            with st.spinner(f"Performing multi-timeframe analysis for {ticker2}..."):
                progress_bar = st.progress(0)
                for idx, (tf_interval, tf_period) in enumerate(timeframes):
                    try:
                        tf_data = fetch_data_with_delay(ticker2, tf_interval, tf_period, delay=api_delay)
                        if not tf_data.empty:
                            result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                            if result:
                                analysis_results_t2.append(result)
                    except Exception as e:
                        st.warning(f"Skipped {tf_interval}/{tf_period}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(timeframes))
            
            if analysis_results_t2:
                mtf_df_t2 = pd.DataFrame(analysis_results_t2)
                st.dataframe(mtf_df_t2, use_container_width=True, height=500)
                
                # Key Insights for Ticker 2
                st.subheader(f"🔍 {ticker2} Multi-Timeframe Summary")
                
                up_trends_t2 = sum(1 for r in analysis_results_t2 if r['Trend'] == 'Up')
                down_trends_t2 = len(analysis_results_t2) - up_trends_t2
                
                try:
                    avg_rsi_t2 = np.mean([float(r['RSI']) for r in analysis_results_t2 if r['RSI'] != 'N/A'])
                except:
                    avg_rsi_t2 = 50
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Bullish Timeframes", f"{up_trends_t2}/{len(analysis_results_t2)}")
                with col2:
                    st.metric("Bearish Timeframes", f"{down_trends_t2}/{len(analysis_results_t2)}")
                with col3:
                    st.metric("Average RSI", safe_format_number(avg_rsi_t2))
                with col4:
                    overall_bias_t2 = "🟢 BULLISH" if up_trends_t2 > down_trends_t2 else "🔴 BEARISH" if down_trends_t2 > up_trends_t2 else "🟡 NEUTRAL"
                    st.metric("Overall Bias", overall_bias_t2)
                
                current_price_2 = float(data2['Close'].iloc[-1])
                first_price_2 = float(data2['Close'].iloc[0])
                pct_change_2 = safe_percentage(current_price_2, first_price_2)
                
                st.info(f"""
                **{ticker2} Analysis Summary:**
                
                Based on {len(analysis_results_t2)} timeframes, {ticker2} shows {'**strong bullish momentum**' if up_trends_t2 > down_trends_t2 * 1.5 else '**moderate bullish bias**' if up_trends_t2 > down_trends_t2 else '**strong bearish pressure**' if down_trends_t2 > up_trends_t2 * 1.5 else '**moderate bearish bias**' if down_trends_t2 > up_trends_t2 else '**consolidation pattern**'}.
                
                - Current Price: {safe_format_number(current_price_2)}
                - Price Change: {safe_format_number(pct_change_2)}%
                - RSI Indicator: {'Oversold - potential bounce' if avg_rsi_t2 < 30 else 'Overbought - potential pullback' if avg_rsi_t2 > 70 else 'Neutral - no extreme conditions'}
                """)
        
        st.markdown("---")
        
        # Ratio Bins Analysis (if ratio enabled)
        if include_ratio and data2 is not None and not data2.empty:
            st.header("📊 Ratio Bins Analysis")
            
            try:
                min_len = min(len(data1), len(data2))
                data1_aligned = data1.iloc[:min_len].copy()
                data2_aligned = data2.iloc[:min_len].copy()
                ratio_data = data1_aligned['Close'] / data2_aligned['Close']
                
                # Create ratio bins
                ratio_clean = ratio_data.dropna()
                if len(ratio_clean) > 5:
                    ratio_bins_cat = pd.qcut(ratio_clean, q=5, labels=False, duplicates='drop')
                    bin_edges = pd.qcut(ratio_clean, q=5, retbins=True, duplicates='drop')[1]
                    
                    # Create labels with ranges
                    bin_labels = []
                    for i in range(len(bin_edges)-1):
                        label = f"{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}"
                        bin_labels.append(label)
                    
                    # Map to descriptive names
                    ratio_descriptions = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                    ratio_bin_names = [f"{ratio_descriptions[i]} ({bin_labels[i]})" for i in range(len(bin_labels))]
                    ratio_bins = pd.Series([ratio_bin_names[int(x)] if not pd.isna(x) else 'Unknown' for x in ratio_bins_cat], 
                                        index=ratio_clean.index)
                    
                    # Calculate returns for both tickers in aligned data
                    t1_returns_points = data1_aligned['Close'].diff()
                    t1_returns_pct = data1_aligned['Close'].pct_change() * 100
                    t2_returns_points = data2_aligned['Close'].diff()
                    t2_returns_pct = data2_aligned['Close'].pct_change() * 100
                    
                    # Align all data
                    start_idx = len(ratio_clean) - len(ratio_bins)
                    
                    ratio_analysis = pd.DataFrame({
                        'DateTime (IST)': ratio_clean.index[start_idx:].strftime('%Y-%m-%d %H:%M:%S'),
                        'Ratio Bin': ratio_bins.values[start_idx:],
                        'Ratio Value': ratio_clean.values[start_idx:],
                        'T1 Returns (Points)': t1_returns_points.iloc[start_idx:].values,
                        'T1 Returns (%)': t1_returns_pct.iloc[start_idx:].values,
                        'T2 Returns (Points)': t2_returns_points.iloc[start_idx:].values,
                        'T2 Returns (%)': t2_returns_pct.iloc[start_idx:].values
                    })
                    
                    st.dataframe(ratio_analysis.tail(50), use_container_width=True, height=400)
                    
                    # Ratio statistics
                    st.subheader("📈 Ratio Statistics & Insights")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Highest Ratio", f"{ratio_clean.max():.4f}")
                    with col2:
                        st.metric("Lowest Ratio", f"{ratio_clean.min():.4f}")
                    with col3:
                        st.metric("Mean Ratio", f"{ratio_clean.mean():.4f}")
                    with col4:
                        st.metric("Current Ratio", f"{ratio_clean.iloc[-1]:.4f}")
                    
                    current_ratio_bin = ratio_bins.iloc[-1] if len(ratio_bins) > 0 else 'Unknown'
                    st.success(f"📊 **Current Ratio Regime:** {current_ratio_bin}")
                    
                    # Bin-specific analysis
                    bin_stats = ratio_analysis.groupby('Ratio Bin').agg({
                        'T1 Returns (Points)': ['mean', 'std', 'count'],
                        'T1 Returns (%)': ['mean', 'std'],
                        'T2 Returns (Points)': ['mean', 'std'],
                        'T2 Returns (%)': ['mean', 'std']
                    }).round(3)
                    
                    st.subheader("📊 Performance by Ratio Regime")
                    st.dataframe(bin_stats, use_container_width=True)
                    
                    st.info(f"""
                    **Comprehensive Ratio Analysis:**
                    
                    **Historical Context:**
                    - Ratio has ranged from {ratio_clean.min():.4f} to {ratio_clean.max():.4f}
                    - Average ratio: {ratio_clean.mean():.4f}
                    - Current ratio: {ratio_clean.iloc[-1]:.4f} ({'above' if ratio_clean.iloc[-1] > ratio_clean.mean() else 'below'} average)
                    
                    **Current Ratio Regime:** {current_ratio_bin}
                    - This regime has occurred {len(ratio_analysis[ratio_analysis['Ratio Bin'] == current_ratio_bin])} times historically
                    
                    **Performance in Current Regime:**
                    - Avg {ticker1} return: {ratio_analysis[ratio_analysis['Ratio Bin'] == current_ratio_bin]['T1 Returns (%)'].mean():.2f}%
                    - Avg {ticker2} return: {ratio_analysis[ratio_analysis['Ratio Bin'] == current_ratio_bin]['T2 Returns (%)'].mean():.2f}%
                    
                    **Forecast:** When ratio is {'high' if 'High' in current_ratio_bin else 'low' if 'Low' in current_ratio_bin else 'medium'}, 
                    {ticker1} {'typically underperforms - consider shorting T1 or buying T2' if 'High' in current_ratio_bin 
                    else 'typically outperforms - consider buying T1 or shorting T2' if 'Low' in current_ratio_bin 
                    else 'shows balanced performance relative to ' + ticker2}
                    """)
                    
            except Exception as e:
                st.warning(f"Ratio bins analysis: {str(e)}")
        
        st.markdown("---")
        
        # Ratio Analysis Section (Only if enabled)
        if include_ratio and data2 is not None and not data2.empty:
            st.header("📊 Ratio Analysis")
            
            try:
                # Align data to same length
                min_len = min(len(data1), len(data2))
                data1_aligned = data1.iloc[:min_len].copy()
                data2_aligned = data2.iloc[:min_len].copy()
                
                # Calculate Ratio
                ratio_data = data1_aligned['Close'] / data2_aligned['Close']
                
                # Create comparison table
                ratio_df = pd.DataFrame({
                    'DateTime (IST)': data1_aligned.index.strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker1 Price': data1_aligned['Close'].values,
                    'Ticker2 Price': data2_aligned['Close'].values,
                    'Ratio': ratio_data.values,
                    'RSI Ticker1': calculate_rsi(data1_aligned['Close']).values,
                    'RSI Ticker2': calculate_rsi(data2_aligned['Close']).values,
                    'RSI Ratio': calculate_rsi(ratio_data).values
                })
                
                st.dataframe(ratio_df.tail(50), use_container_width=True, height=400)
                
                # Export functionality
                col1, col2 = st.columns([3, 1])
                with col2:
                    def convert_df_to_excel(df):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Ratio Analysis')
                        return output.getvalue()
                    
                    excel_data = convert_df_to_excel(ratio_df)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"{ticker1}_{ticker2}_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error in ratio analysis: {str(e)}")
            
            st.markdown("---")
        
        # Multi-Timeframe Analysis
        st.header("📈 Multi-Timeframe Technical Analysis")
        
        timeframes = [
            ("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("30m", "1mo"),
            ("1h", "1mo"), ("2h", "3mo"), ("4h", "6mo"), ("1d", "1y"),
            ("1wk", "5y")
        ]
        
        analysis_results = []
        
        with st.spinner("🔍 Performing multi-timeframe analysis..."):
            progress_bar = st.progress(0)
            for idx, (tf_interval, tf_period) in enumerate(timeframes):
                try:
                    tf_data = fetch_data_with_delay(ticker1, tf_interval, tf_period, delay=api_delay)
                    if not tf_data.empty:
                        result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                        if result:
                            analysis_results.append(result)
                except Exception as e:
                    st.warning(f"Skipped {tf_interval}/{tf_period}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(timeframes))
        
        if analysis_results:
            mtf_df = pd.DataFrame(analysis_results)
            
            # Display with better formatting
            st.dataframe(mtf_df, use_container_width=True, height=500)
            
            # Key Insights
            st.subheader("🔍 Multi-Timeframe Key Insights")
            
            up_trends = sum(1 for r in analysis_results if r['Trend'] == 'Up')
            down_trends = len(analysis_results) - up_trends
            
            try:
                avg_rsi = np.mean([float(r['RSI']) for r in analysis_results if r['RSI'] != 'N/A'])
            except:
                avg_rsi = 50
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bullish Timeframes", f"{up_trends}/{len(analysis_results)}")
            with col2:
                st.metric("Bearish Timeframes", f"{down_trends}/{len(analysis_results)}")
            with col3:
                st.metric("Average RSI", safe_format_number(avg_rsi))
            with col4:
                overall_bias = "🟢 BULLISH" if up_trends > down_trends else "🔴 BEARISH" if down_trends > up_trends else "🟡 NEUTRAL"
                st.metric("Overall Bias", overall_bias)
            
            # Detailed Summary
            st.info(f"""
            **Analysis Summary:**
            
            Based on {len(analysis_results)} timeframes, the market shows {'**strong bullish momentum**' if up_trends > down_trends * 1.5 else '**moderate bullish bias**' if up_trends > down_trends else '**strong bearish pressure**' if down_trends > up_trends * 1.5 else '**moderate bearish bias**' if down_trends > up_trends else '**consolidation pattern**'}.
            
            - Current Price: {safe_format_number(current_price_1)}
            - Price Change: {safe_format_number(pct_change_1)}%
            - RSI Indicator: {'Oversold - potential bounce' if avg_rsi < 30 else 'Overbought - potential pullback' if avg_rsi > 70 else 'Neutral - no extreme conditions'}
            
            **Strategic Recommendation:** {'Consider long positions with tight stops' if up_trends > down_trends else 'Consider short positions or wait for better entry' if down_trends > up_trends else 'Wait for clear directional bias'}
            """)
        
        st.markdown("---")
        
        # Volatility Analysis
        st.header("📊 Volatility & Returns Analysis")
        
        try:
            # Calculate volatility
            returns = data1['Close'].pct_change()
            volatility = returns.rolling(window=min(20, len(returns))).std() * np.sqrt(252) * 100
            
            # Create bins with explicit boundaries
            vol_clean = volatility.dropna()
            if len(vol_clean) > 5:
                vol_bins_cat = pd.qcut(vol_clean, q=5, labels=False, duplicates='drop')
                bin_edges = pd.qcut(vol_clean, q=5, retbins=True, duplicates='drop')[1]
                
                # Create labels with ranges
                bin_labels = []
                for i in range(len(bin_edges)-1):
                    label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                    bin_labels.append(label)
                
                # Map to descriptive names with ranges
                vol_descriptions = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                vol_bin_names = [f"{vol_descriptions[i]} ({bin_labels[i]})" for i in range(len(bin_labels))]
                vol_bins = pd.Series([vol_bin_names[int(x)] if not pd.isna(x) else 'Unknown' for x in vol_bins_cat], 
                                    index=vol_clean.index)
                
                # Calculate price ranges for each volatility bin
                start_idx = len(data1) - len(vol_bins)
                price_data = data1['Close'].iloc[start_idx:]
                
                vol_analysis = pd.DataFrame({
                    'DateTime (IST)': data1.index[start_idx:].strftime('%Y-%m-%d %H:%M:%S'),
                    'Volatility Bin': vol_bins.values,
                    'Volatility %': vol_clean.values,
                    'Price': price_data.values,
                    'Returns (Points)': data1['Close'].diff().iloc[start_idx:].values,
                    'Returns (%)': (returns.iloc[start_idx:].values * 100)
                })
                
                st.dataframe(vol_analysis.tail(50), use_container_width=True, height=400)
                
                # Statistical Summary
                st.subheader("📈 Volatility Statistics & Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Highest Volatility", f"{vol_clean.max():.2f}%")
                with col2:
                    st.metric("Lowest Volatility", f"{vol_clean.min():.2f}%")
                with col3:
                    st.metric("Mean Volatility", f"{vol_clean.mean():.2f}%")
                with col4:
                    current_vol = vol_clean.iloc[-1]
                    st.metric("Current Volatility", f"{current_vol:.2f}%")
                
                # Returns statistics
                returns_points = data1['Close'].diff().dropna()
                returns_pct = returns.dropna() * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Return (Points)", f"{returns_points.max():.2f}")
                with col2:
                    st.metric("Min Return (Points)", f"{returns_points.min():.2f}")
                with col3:
                    st.metric("Mean Return (%)", f"{returns_pct.mean():.3f}%")
                with col4:
                    st.metric("Current Return (%)", f"{returns_pct.iloc[-1]:.2f}%")
                
                # Determine current bin
                current_vol_bin = vol_bins.iloc[-1] if len(vol_bins) > 0 else 'Unknown'
                
                # Bin-specific analysis
                vol_analysis_copy = vol_analysis.copy()
                bin_stats = vol_analysis_copy.groupby('Volatility Bin').agg({
                    'Returns (Points)': ['mean', 'std', 'min', 'max', 'count'],
                    'Returns (%)': ['mean', 'std']
                }).round(3)
                
                st.success(f"📊 **Current Volatility Regime:** {current_vol_bin}")
                
                st.info(f"""
                **Comprehensive Volatility Analysis:**
                
                **Historical Context:**
                - Volatility has ranged from {vol_clean.min():.2f}% to {vol_clean.max():.2f}%
                - Average volatility: {vol_clean.mean():.2f}%
                - Current volatility: {current_vol:.2f}% ({'above' if current_vol > vol_clean.mean() else 'below'} average)
                
                **Returns Analysis:**
                - Largest gain: {returns_points.max():.2f} points ({returns_pct.max():.2f}%)
                - Largest loss: {returns_points.min():.2f} points ({returns_pct.min():.2f}%)
                - Average return: {returns_pct.mean():.3f}% per period
                
                **Current Market Status:**
                - Current bin: **{current_vol_bin}**
                - This volatility regime has occurred {len(vol_analysis[vol_analysis['Volatility Bin'] == current_vol_bin])} times in history
                
                **Forecast Based on Historical Bin Performance:**
                When volatility is in the **{current_vol_bin.split('(')[0].strip()}** regime:
                - Average return: {vol_analysis_copy[vol_analysis_copy['Volatility Bin'] == current_vol_bin]['Returns (%)'].mean():.2f}%
                - Typical range: {vol_analysis_copy[vol_analysis_copy['Volatility Bin'] == current_vol_bin]['Returns (Points)'].min():.2f} to {vol_analysis_copy[vol_analysis_copy['Volatility Bin'] == current_vol_bin]['Returns (Points)'].max():.2f} points
                
                **Prediction:** {'Higher volatility suggests larger price swings expected - trade with wider stops' if current_vol > vol_clean.mean() else 'Lower volatility suggests smaller moves - tighter stops appropriate'}
                """)
                
                # Bin performance table
                st.subheader("📊 Performance by Volatility Regime")
                st.dataframe(bin_stats, use_container_width=True)
                
            else:
                st.warning("Insufficient data for volatility binning analysis")
                
        except Exception as e:
            st.warning(f"Volatility analysis requires more data points: {str(e)}")
        
        st.markdown("---")
        
        # Pattern Recognition
        st.header("🔍 Advanced Pattern Recognition & Historical Similarities")
        
        try:
            patterns = detect_patterns(data1, threshold=pattern_threshold)
            
            if patterns:
                pattern_df = pd.DataFrame(patterns)
                pattern_df['DateTime'] = pattern_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(pattern_df, use_container_width=True, height=400)
                
                # Pattern statistics
                total_patterns = len(patterns)
                up_moves = sum(1 for p in patterns if 'Up' in p['Direction'])
                down_moves = total_patterns - up_moves
                vol_bursts = sum(1 for p in patterns if '✓' in p['Volatility_Burst'])
                volume_spikes = sum(1 for p in patterns if '✓' in p['Volume_Spike'])
                rsi_divs = sum(1 for p in patterns if '✓' in p['RSI_Divergence'])
                ema_crosses = sum(1 for p in patterns if '✓' in p['EMA_Crossover'])
                breakouts = sum(1 for p in patterns if '✓' in p['Support/Resistance_Break'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Patterns", total_patterns)
                    st.metric("Upward Moves", f"{up_moves} ({up_moves/total_patterns*100:.1f}%)")
                with col2:
                    st.metric("Volatility Bursts", f"{vol_bursts} ({vol_bursts/total_patterns*100:.1f}%)")
                    st.metric("Volume Spikes", f"{volume_spikes} ({volume_spikes/total_patterns*100:.1f}%)")
                with col3:
                    st.metric("RSI Divergences", f"{rsi_divs} ({rsi_divs/total_patterns*100:.1f}%)")
                    st.metric("EMA Crossovers", f"{ema_crosses} ({ema_crosses/total_patterns*100:.1f}%)")
                with col4:
                    st.metric("Breakouts", f"{breakouts} ({breakouts/total_patterns*100:.1f}%)")
                
                # Calculate average characteristics
                avg_move = np.mean([abs(p['Move (Points)']) for p in patterns])
                avg_move_pct = np.mean([abs(p['Move (%)']) for p in patterns])
                
                st.subheader("📊 Pattern Analysis Insights")
                st.info(f"""
                **Historical Pattern Summary:**
                
                **Move Characteristics:**
                - Detected {total_patterns} significant moves (>{pattern_threshold} points threshold)
                - Average move size: {avg_move:.2f} points ({avg_move_pct:.2f}%)
                - Direction bias: {up_moves} up vs {down_moves} down moves
                
                **Pre-Move Indicators (Before Significant Moves):**
                - Volatility bursts preceded {vol_bursts}/{total_patterns} moves ({vol_bursts/total_patterns*100:.1f}%)
                - Volume spikes preceded {volume_spikes}/{total_patterns} moves ({volume_spikes/total_patterns*100:.1f}%)
                - RSI divergences detected in {rsi_divs}/{total_patterns} cases ({rsi_divs/total_patterns*100:.1f}%)
                - EMA crossovers preceded {ema_crosses}/{total_patterns} moves ({ema_crosses/total_patterns*100:.1f}%)
                - Support/Resistance breakouts in {breakouts}/{total_patterns} cases ({breakouts/total_patterns*100:.1f}%)
                
                **Key Findings:**
                - {'Volatility bursts are strong predictor of upcoming moves' if vol_bursts/total_patterns > 0.6 else 'Volatility bursts show moderate correlation with moves' if vol_bursts/total_patterns > 0.3 else 'Volatility bursts are weak predictor'}
                - {'RSI divergence is reliable signal' if rsi_divs/total_patterns > 0.5 else 'RSI divergence shows moderate reliability' if rsi_divs/total_patterns > 0.25 else 'RSI divergence is less reliable in this timeframe'}
                - {'Volume confirmation is important' if volume_spikes/total_patterns > 0.4 else 'Volume shows mixed signals'}
                """)
                
                # Current market similarity check
                st.subheader("🎯 Current Market Pattern Similarity")
                
                current_close = data1['Close'].values
                if len(current_close) >= 10:
                    # Analyze last 10 candles
                    recent_vol = data1['Close'].iloc[-10:].std()
                    overall_vol = data1['Close'].std()
                    recent_vol_burst = recent_vol > overall_vol * 1.5
                    
                    recent_rsi = calculate_rsi(data1['Close'])
                    current_rsi = float(recent_rsi.iloc[-1]) if len(recent_rsi) > 0 else 50
                    
                    recent_moves = [current_close[i] - current_close[i-1] for i in range(-10, 0)]
                    consecutive_up = sum(1 for m in recent_moves if m > 0)
                    consecutive_down = sum(1 for m in recent_moves if m < 0)
                    
                    warnings = []
                    if recent_vol_burst:
                        warnings.append("⚠️ **Volatility Burst Detected** - Historical data shows this precedes major moves")
                    if current_rsi > 70:
                        warnings.append("⚠️ **Overbought RSI** - Potential divergence setup for downward move")
                    elif current_rsi < 30:
                        warnings.append("⚠️ **Oversold RSI** - Potential divergence setup for upward move")
                    if consecutive_up >= 5:
                        warnings.append("⚠️ **Extended Rally** - 5+ consecutive up candles, exhaustion possible")
                    elif consecutive_down >= 5:
                        warnings.append("⚠️ **Extended Decline** - 5+ consecutive down candles, reversal possible")
                    
                    if warnings:
                        for warning in warnings:
                            st.warning(warning)
                    else:
                        st.success("✓ No immediate warning patterns detected in current market structure")
                    
                    st.info(f"""
                    **Current Market Analysis:**
                    - Recent volatility: {recent_vol:.2f} ({'elevated' if recent_vol_burst else 'normal'})
                    - Current RSI: {current_rsi:.1f}
                    - Last 10 candles: {consecutive_up} up, {consecutive_down} down
                    - Pattern similarity: {'High' if recent_vol_burst or abs(current_rsi - 50) > 20 else 'Moderate' if abs(current_rsi - 50) > 10 else 'Low'} alignment with pre-move patterns
                    
                    **Forecast:** Based on historical patterns, {'significant move likely within next few periods' if recent_vol_burst and (current_rsi > 65 or current_rsi < 35) else 'moderate probability of directional move' if recent_vol_burst or abs(current_rsi - 50) > 15 else 'consolidation likely to continue'}
                    """)
                
            else:
                st.info(f"No significant patterns detected with current threshold ({pattern_threshold} points). Consider lowering threshold or using longer period for more pattern data.")
        except Exception as e:
            st.warning(f"Pattern recognition analysis: {str(e)}")
        
        st.markdown("---")
        
        # Visualization Section
        st.header("📈 Interactive Technical Charts")
        
        try:
            if include_ratio and data2 is not None and not data2.empty:
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        f'{ticker1} Price Action',
                        f'{ticker2} Price Action',
                        'Ratio Chart',
                        f'RSI - {ticker1}',
                        f'RSI - {ticker2}',
                        'RSI - Ratio'
                    ),
                    vertical_spacing=0.08,
                    horizontal_spacing=0.1
                )
                
                # Price charts with EMAs
                fig.add_trace(go.Scatter(x=data1.index, y=data1['Close'], name=ticker1, 
                                        line=dict(color='blue', width=2)), row=1, col=1)
                
                # Add EMAs to ticker1
                if len(data1) >= 20:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 20), 
                                            name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                if len(data1) >= 50:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 50), 
                                            name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=data2.index, y=data2['Close'], name=ticker2, 
                                        line=dict(color='green', width=2)), row=1, col=2)
                
                # Ratio
                ratio_data = data1['Close'] / data2['Close']
                fig.add_trace(go.Scatter(x=data1.index, y=ratio_data, name='Ratio', 
                                        line=dict(color='purple', width=2)), row=2, col=1)
                
                # RSI charts
                rsi1 = calculate_rsi(data1['Close'])
                rsi2 = calculate_rsi(data2['Close'])
                rsi_ratio = calculate_rsi(ratio_data)
                
                fig.add_trace(go.Scatter(x=data1.index, y=rsi1, name=f'RSI {ticker1}', 
                                        line=dict(color='blue', width=2)), row=2, col=2)
                fig.add_trace(go.Scatter(x=data2.index, y=rsi2, name=f'RSI {ticker2}', 
                                        line=dict(color='green', width=2)), row=3, col=1)
                fig.add_trace(go.Scatter(x=data1.index, y=rsi_ratio, name='RSI Ratio', 
                                        line=dict(color='purple', width=2)), row=3, col=2)
                
                # Add RSI levels
                for row_idx in [2, 3]:
                    for col_idx in [1, 2]:
                        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, 
                                     row=row_idx, col=col_idx)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, 
                                     row=row_idx, col=col_idx)
                        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, 
                                     row=row_idx, col=col_idx)
                
                fig.update_layout(height=1200, showlegend=True, title_text="Comprehensive Technical Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Single ticker analysis
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f'{ticker1} Price with EMAs', 'RSI Indicator'),
                    vertical_spacing=0.15,
                    row_heights=[0.7, 0.3]
                )
                
                # Price chart
                fig.add_trace(go.Candlestick(
                    x=data1.index,
                    open=data1['Open'],
                    high=data1['High'],
                    low=data1['Low'],
                    close=data1['Close'],
                    name='OHLC'
                ), row=1, col=1)
                
                # Add EMAs
                if len(data1) >= 20:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 20), 
                                            name='EMA 20', line=dict(color='orange', width=2)), row=1, col=1)
                if len(data1) >= 50:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 50), 
                                            name='EMA 50', line=dict(color='red', width=2)), row=1, col=1)
                if len(data1) >= 200:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 200), 
                                            name='EMA 200', line=dict(color='blue', width=2)), row=1, col=1)
                
                # RSI
                rsi = calculate_rsi(data1['Close'])
                fig.add_trace(go.Scatter(x=data1.index, y=rsi, name='RSI', 
                                        line=dict(color='purple', width=2)), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                
                fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.success("📊 **Chart Analysis:** Look for divergences between price and RSI. Bullish divergence (price lower low + RSI higher low) suggests reversal to upside.")
            
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
        
        st.markdown("---")
        
        # Returns Distribution with Bell Curve
        st.header("📊 Statistical Distribution Analysis with Normal Distribution")
        
        try:
            returns_points = data1['Close'].diff().dropna()
            returns_pct = data1['Close'].pct_change().dropna() * 100
            
            # Create histograms with normal distribution overlay
            fig_dist = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Returns Distribution (Points) with Normal Curve', 
                               'Returns Distribution (%) with Normal Curve'),
                vertical_spacing=0.15
            )
            
            # Points distribution
            fig_dist.add_trace(go.Histogram(x=returns_points, nbinsx=50, name='Actual Returns', 
                                           marker_color='blue', opacity=0.7, histnorm='probability density'), 
                              row=1, col=1)
            
            # Fit normal distribution
            mu_points = returns_points.mean()
            sigma_points = returns_points.std()
            x_points = np.linspace(returns_points.min(), returns_points.max(), 100)
            normal_points = (1/(sigma_points * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x_points - mu_points)/sigma_points)**2)
            
            fig_dist.add_trace(go.Scatter(x=x_points, y=normal_points, mode='lines',
                                         name='Normal Distribution', line=dict(color='red', width=3)),
                              row=1, col=1)
            
            # Percentage distribution
            fig_dist.add_trace(go.Histogram(x=returns_pct, nbinsx=50, name='Actual Returns %', 
                                           marker_color='green', opacity=0.7, histnorm='probability density'), 
                              row=2, col=1)
            
            mu_pct = returns_pct.mean()
            sigma_pct = returns_pct.std()
            x_pct = np.linspace(returns_pct.min(), returns_pct.max(), 100)
            normal_pct = (1/(sigma_pct * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x_pct - mu_pct)/sigma_pct)**2)
            
            fig_dist.add_trace(go.Scatter(x=x_pct, y=normal_pct, mode='lines',
                                         name='Normal Distribution', line=dict(color='red', width=3)),
                              row=2, col=1)
            
            fig_dist.update_layout(height=800, showlegend=True)
            fig_dist.update_xaxes(title_text="Returns (Points)", row=1, col=1)
            fig_dist.update_xaxes(title_text="Returns (%)", row=2, col=1)
            fig_dist.update_yaxes(title_text="Probability Density", row=1, col=1)
            fig_dist.update_yaxes(title_text="Probability Density", row=2, col=1)
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Statistical insights
            st.subheader("📈 Distribution Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean (Points)", safe_format_number(mu_points, 3))
                st.metric("Std Dev (Points)", safe_format_number(sigma_points, 3))
            with col2:
                st.metric("Mean (%)", safe_format_number(mu_pct, 3))
                st.metric("Std Dev (%)", safe_format_number(sigma_pct, 3))
            with col3:
                skew_points = returns_points.skew()
                st.metric("Skewness", safe_format_number(skew_points, 3))
                st.metric("Interpretation", "Right Tail" if skew_points > 0 else "Left Tail")
            with col4:
                kurt_points = returns_points.kurtosis()
                st.metric("Kurtosis", safe_format_number(kurt_points, 3))
                st.metric("Interpretation", "Fat Tails" if kurt_points > 0 else "Thin Tails")
            
            # Z-Score Analysis
            st.subheader("📊 Z-Score Statistical Analysis")
            
            z_returns_points = (returns_points - mu_points) / sigma_points
            z_returns_pct = (returns_pct - mu_pct) / sigma_pct
            
            z_df = pd.DataFrame({
                'DateTime (IST)': data1.index[1:len(z_returns_points)+1].strftime('%Y-%m-%d %H:%M:%S'),
                'Returns (Points)': returns_points.values,
                'Z-Score (Points)': z_returns_points.values,
                'Returns (%)': returns_pct.values,
                'Z-Score (%)': z_returns_pct.values
            })
            
            st.dataframe(z_df.tail(30), use_container_width=True, height=300)
            
            current_z_points = z_returns_points.iloc[-1] if len(z_returns_points) > 0 else 0
            current_z_pct = z_returns_pct.iloc[-1] if len(z_returns_pct) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Z-Score (Points)", safe_format_number(current_z_points, 3))
            with col2:
                st.metric("Current Z-Score (%)", safe_format_number(current_z_pct, 3))
            with col3:
                percentile = (1 + np.tanh(current_z_points/2)) * 50
                st.metric("Percentile Rank", f"{percentile:.1f}%")
            with col4:
                interpretation = "🔴 Extreme High" if abs(current_z_points) > 2 else "🟡 Moderate" if abs(current_z_points) > 1 else "🟢 Normal"
                st.metric("Signal", interpretation)
            
            st.info(f"""
            **Comprehensive Z-Score Analysis:**
            
            **Understanding Normal Distribution (Bell Curve):**
            - **68% of data** falls within ±1 standard deviation (Z-score: -1 to +1)
            - **95% of data** falls within ±2 standard deviations (Z-score: -2 to +2)
            - **99.7% of data** falls within ±3 standard deviations (Z-score: -3 to +3)
            
            **Current Position:**
            - Current Z-Score: **{safe_format_number(current_z_points, 2)}** (Points) | **{safe_format_number(current_z_pct, 2)}** (%)
            - This places current return at **{percentile:.1f}th percentile**
            - Interpretation: {
                'Price movement is EXTREMELY unusual (>95th percentile) - strong mean reversion expected' if abs(current_z_points) > 2 
                else 'Price movement is somewhat unusual (68-95th percentile) - moderate reversion possible' if abs(current_z_points) > 1 
                else 'Price movement is NORMAL (within 68% range) - no extreme signals'
            }
            
            **Statistical Insights:**
            - Mean return: {safe_format_number(mu_points, 2)} points ({safe_format_number(mu_pct, 3)}%)
            - Standard deviation: {safe_format_number(sigma_points, 2)} points ({safe_format_number(sigma_pct, 3)}%)
            - Skewness: {safe_format_number(skew_points, 2)} ({'positive (upside bias)' if skew_points > 0 else 'negative (downside bias)'})
            - Kurtosis: {safe_format_number(kurt_points, 2)} ({'fat tails - more extreme moves than normal' if kurt_points > 0 else 'thin tails - fewer extreme moves'})
            
            **Probabilistic Forecast:**
            Based on normal distribution:
            - **68% probability** next move will be between {safe_format_number(mu_points - sigma_points, 1)} and {safe_format_number(mu_points + sigma_points, 1)} points
            - **95% probability** next move will be between {safe_format_number(mu_points - 2*sigma_points, 1)} and {safe_format_number(mu_points + 2*sigma_points, 1)} points
            
            **Trading Implication:**
            {
                f'Current extreme reading (Z={safe_format_number(current_z_points, 2)}) suggests HIGH PROBABILITY of reversal. ' + 
                ('Consider SELLING/SHORTING as price likely to revert downward' if current_z_points > 2 else 'Consider BUYING as price likely to revert upward' if current_z_points < -2 else '')
                if abs(current_z_points) > 2
                else f'Current moderate reading suggests price within normal range. Continue monitoring for extreme signals.'
            }
            """)
            
            # Bell curve visualization with zones
            fig_bell = go.Figure()
            
            x_range = np.linspace(-4, 4, 1000)
            y_range = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x_range**2)
            
            fig_bell.add_trace(go.Scatter(x=x_range, y=y_range, fill='tozeroy',
                                         name='Normal Distribution', line=dict(color='blue')))
            
            # Color zones
            fig_bell.add_vrect(x0=-1, x1=1, fillcolor="green", opacity=0.2, 
                              annotation_text="68% (Normal)", annotation_position="top left")
            fig_bell.add_vrect(x0=-2, x1=-1, fillcolor="yellow", opacity=0.2,
                              annotation_text="95% Range", annotation_position="top left")
            fig_bell.add_vrect(x0=1, x1=2, fillcolor="yellow", opacity=0.2)
            fig_bell.add_vrect(x0=-4, x1=-2, fillcolor="red", opacity=0.2,
                              annotation_text="Extreme", annotation_position="top left")
            fig_bell.add_vrect(x0=2, x1=4, fillcolor="red", opacity=0.2)
            
            # Mark current position
            current_y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*current_z_points**2)
            fig_bell.add_trace(go.Scatter(x=[current_z_points], y=[current_y], mode='markers',
                                         marker=dict(size=15, color='red'),
                                         name=f'Current Position (Z={current_z_points:.2f})'))
            
            fig_bell.update_layout(
                title="Bell Curve with Current Position",
                xaxis_title="Z-Score (Standard Deviations from Mean)",
                yaxis_title="Probability Density",
                height=500
            )
            
            st.plotly_chart(fig_bell, use_container_width=True)
            
            # Ratio Z-Score (if applicable)
            if include_ratio and data2 is not None and not data2.empty:
                min_len = min(len(data1), len(data2))
                ratio_data = (data1['Close'].iloc[:min_len] / data2['Close'].iloc[:min_len])
                ratio_returns = ratio_data.diff().dropna()
                
                if len(ratio_returns) > 0:
                    z_ratio = (ratio_returns - ratio_returns.mean()) / ratio_returns.std()
                    
                    st.subheader("📊 Ratio Z-Score Analysis")
                    current_z_ratio = z_ratio.iloc[-1] if len(z_ratio) > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Ratio Z-Score", safe_format_number(current_z_ratio, 3))
                    with col2:
                        ratio_signal = "🔴 T1 Overvalued" if current_z_ratio > 2 else "🟢 T1 Undervalued" if current_z_ratio < -2 else "🟡 Normal"
                        st.metric("Ratio Signal", ratio_signal)
                    
                    st.write(f"""
                    **Ratio Interpretation:** Z-score of {safe_format_number(current_z_ratio, 2)} indicates that {ticker1} is 
                    {'significantly overpriced relative to ' + ticker2 + ' - consider selling T1 or buying T2' if current_z_ratio > 2 
                    else 'significantly underpriced relative to ' + ticker2 + ' - consider buying T1 or selling T2' if current_z_ratio < -2 
                    else 'fairly priced relative to ' + ticker2}
                    """)
        
        except Exception as e:
            st.warning(f"Statistical analysis: {str(e)}")
        
        st.markdown("---")
        
        # Final Trading Recommendation
        st.header("🎯 FINAL TRADING RECOMMENDATION")
        
        try:
            # Initialize variables with defaults
            current_z = 0
            current_z_points = 0
            
            # Calculate Z-score if data available
            try:
                returns_points = data1['Close'].diff().dropna()
                if len(returns_points) > 0:
                    mu_points = returns_points.mean()
                    sigma_points = returns_points.std()
                    z_returns_points = (returns_points - mu_points) / sigma_points
                    current_z = z_returns_points.iloc[-1] if len(z_returns_points) > 0 else 0
                    current_z_points = current_z
            except:
                pass
            
            # Gather all signals
            signals = []
            signal_weights = []
            
            # 1. Multi-timeframe trend signal
            if 'up_trends' in locals() and 'down_trends' in locals() and 'analysis_results' in locals() and len(analysis_results) > 0:
                trend_signal = 1 if up_trends > down_trends else -1 if down_trends > up_trends else 0
                signals.append(trend_signal)
                signal_weights.append(0.3)
            else:
                trend_signal = 0
            
            # 2. RSI signal
            if 'avg_rsi' in locals():
                rsi_signal = -1 if avg_rsi > 70 else 1 if avg_rsi < 30 else 0
                signals.append(rsi_signal)
                signal_weights.append(0.2)
            else:
                rsi_signal = 0
                avg_rsi = 50
            
            # 3. Z-Score signal
            z_signal = -1 if current_z > 2 else 1 if current_z < -2 else 0
            signals.append(z_signal)
            signal_weights.append(0.2)
            
            # 4. EMA alignment
            current_price = float(data1['Close'].iloc[-1])
            ema_20_val = calculate_ema(data1['Close'], 20).iloc[-1] if len(data1) >= 20 else current_price
            ema_50_val = calculate_ema(data1['Close'], 50).iloc[-1] if len(data1) >= 50 else current_price
            
            ema_signal = 1 if current_price > ema_20_val > ema_50_val else -1 if current_price < ema_20_val < ema_50_val else 0
            signals.append(ema_signal)
            signal_weights.append(0.3)
            
            # Calculate weighted signal
            if signals and signal_weights:
                total_signal = sum(s * w for s, w in zip(signals, signal_weights))
            else:
                total_signal = 0
            
            # Calculate ATR for risk management
            if 'High' in data1.columns and 'Low' in data1.columns:
                atr = (data1['High'] - data1['Low']).rolling(14).mean().iloc[-1]
            else:
                atr = data1['Close'].std()
            
            # Generate recommendation
            if total_signal >= 0.3:
                action = "🟢 STRONG BUY"
                confidence = "High"
                entry = current_price
                target = current_price + (2.5 * atr)
                sl = current_price - (atr)
                risk_reward = 2.5
            elif total_signal >= 0.15:
                action = "🟢 BUY"
                confidence = "Moderate"
                entry = current_price
                target = current_price + (2 * atr)
                sl = current_price - (0.8 * atr)
                risk_reward = 2.5
            elif total_signal <= -0.3:
                action = "🔴 STRONG SELL"
                confidence = "High"
                entry = current_price
                target = current_price - (2.5 * atr)
                sl = current_price + (atr)
                risk_reward = 2.5
            elif total_signal <= -0.15:
                action = "🔴 SELL"
                confidence = "Moderate"
                entry = current_price
                target = current_price - (2 * atr)
                sl = current_price + (0.8 * atr)
                risk_reward = 2.5
            else:
                action = "🟡 HOLD / WAIT"
                confidence = "Low"
                entry = current_price
                target = current_price
                sl = current_price
                risk_reward = 0
            
            # Display recommendation
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">🎯 {action}</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence Level: <strong>{confidence}</strong></p>
                <p style="margin: 0;">Signal Strength: <strong>{abs(total_signal):.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed trade setup
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Trade Setup")
                st.markdown(f"""
                **Entry Price:** `{safe_format_number(entry)}`
                
                **Target Price:** `{safe_format_number(target)}` 
                - Profit: {safe_format_number(abs(target - entry))} points
                - Gain: {safe_format_number(abs((target - entry) / entry * 100))}%
                
                **Stop Loss:** `{safe_format_number(sl)}`
                - Risk: {safe_format_number(abs(sl - entry))} points
                - Loss: {safe_format_number(abs((sl - entry) / entry * 100))}%
                
                **Risk/Reward Ratio:** `1:{safe_format_number(risk_reward)}`
                """)
            
            with col2:
                st.subheader("📊 Signal Breakdown")
                trend_count = f"{up_trends}/{len(analysis_results)}" if 'up_trends' in locals() and 'analysis_results' in locals() else "N/A"
                st.markdown(f"""
                **Component Signals:**
                1. Multi-Timeframe Trend: {'🟢 Bullish' if trend_signal > 0 else '🔴 Bearish' if trend_signal < 0 else '🟡 Neutral'} ({trend_count})
                2. RSI Indicator: {'🟢 Oversold' if avg_rsi < 30 else '🔴 Overbought' if avg_rsi > 70 else '🟡 Neutral'} ({safe_format_number(avg_rsi)})
                3. Z-Score: {'🔴 Extreme High' if current_z > 2 else '🟢 Extreme Low' if current_z < -2 else '🟡 Normal'} ({safe_format_number(current_z, 2)})
                4. EMA Alignment: {'🟢 Bullish' if ema_signal > 0 else '🔴 Bearish' if ema_signal < 0 else '🟡 Mixed'}
                
                **Combined Signal:** {safe_format_number(total_signal, 2)}
                """)
            
            st.subheader("⚠️ Risk Management Guidelines")
            st.warning(f"""
            **Position Sizing:**
            - Risk maximum 1-2% of portfolio per trade
            - Position size = (Account Risk / Trade Risk)
            - If account is $10,000 and risk 1% = $100
            - Trade risk = {safe_format_number(abs(sl - entry))} points
            
            **Exit Strategy:**
            - Use trailing stop loss after 50% profit achieved
            - Consider partial profit booking at intermediate levels
            - Monitor volume and momentum for early exit signals
            - Re-evaluate if market structure changes significantly
            
            **Trade Management:**
            - Set alerts at entry, target, and stop loss levels
            - Monitor for 15-30 minutes after entry for confirmation
            - Be prepared to exit if price action contradicts thesis
            - Review position at end of trading session
            """)
            
            st.subheader("📝 Trading Rationale")
            pct_change_1 = safe_percentage(current_price, float(data1['Close'].iloc[0]))
            st.info(f"""
            **Why This Signal:**
            
            The algorithmic analysis across multiple timeframes and indicators suggests {action.lower()} based on:
            
            1. **Trend Alignment:** {trend_count} timeframes analyzed
            2. **Mean Reversion:** Z-score of {safe_format_number(current_z, 2)} indicates {'extreme conditions likely to reverse' if abs(current_z) > 2 else 'normal market conditions'}
            3. **Momentum:** RSI at {safe_format_number(avg_rsi)} shows {'oversold conditions' if avg_rsi < 30 else 'overbought conditions' if avg_rsi > 70 else 'neutral momentum'}
            4. **Technical Structure:** Price {'above' if current_price > ema_20_val else 'below'} key EMAs suggests {'bullish' if current_price > ema_20_val else 'bearish'} bias
            
            **Market Context:**
            Current price of {safe_format_number(current_price)} has moved {safe_format_number(pct_change_1)}% in this period. 
            Historical volatility (ATR) of {safe_format_number(atr)} suggests reasonable target and stop levels.
            
            **Expected Scenario:**
            {'Price likely to move higher toward resistance' if total_signal > 0 else 'Price likely to move lower toward support' if total_signal < 0 else 'Price likely to consolidate - wait for clearer signals'}
            """)
            
        except Exception as e:
            st.error(f"Error generating recommendation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("""
        <div style="background-color: #fee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f00;">
        <strong>⚠️ DISCLAIMER:</strong> This analysis is for educational purposes only and should not be considered financial advice. 
        Trading involves substantial risk of loss. Always conduct your own research, use proper risk management, 
        and consult with a licensed financial advisor before making investment decisions. Past performance does not guarantee future results.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Unable to fetch data for Ticker 1. Please check the ticker symbol and try again.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Pro Algo Trading Dashboard</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Professional-grade algorithmic trading analysis at your fingertips
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Multi-Asset Support
        - Global Indices (NIFTY, SENSEX, Bank NIFTY)
        - Cryptocurrencies (BTC, ETH)
        - Commodities (Gold, Silver)
        - Forex Pairs (USD/INR, EUR/USD)
        - Custom Tickers
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Advanced Analytics
        - Multi-timeframe analysis
        - RSI, EMA, SMA indicators
        - Fibonacci levels
        - Support/Resistance
        - Pattern recognition
        - Z-Score statistics
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Trading Signals
        - AI-powered recommendations
        - Risk/Reward calculations
        - Entry, Target, Stop Loss
        - Position sizing guidance
        - Real-time analysis
        - Excel export
        """)
    
    st.markdown("---")
    
    st.info("""
    ### 🚀 Quick Start Guide:
    
    1. **Select Your Asset** - Choose from popular indices, crypto, forex or enter custom ticker
    2. **Optional: Enable Ratio Analysis** - Compare two tickers for spread/pairs trading
    3. **Configure Timeframe** - Select interval (1m to 1d) and period (1d to 10y)
    4. **Fetch & Analyze** - Click the button to fetch data and generate comprehensive analysis
    5. **Review Signals** - Get actionable buy/sell/hold recommendations with complete trade setup
    
    💡 **Pro Tip:** Start with daily interval and 1-year period for reliable signals, then drill down to lower timeframes for entry timing.
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
    <strong>Built with ❤️ for Algo Traders</strong><br>
    Powered by Streamlit • yFinance • Plotly
    </div>
    """, unsafe_allow_html=True)
