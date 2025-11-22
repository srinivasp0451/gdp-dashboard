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
</style>
""", unsafe_allow_html=True)

# Define timeframes globally
TIMEFRAMES = [
    ("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("30m", "1mo"),
    ("1h", "1mo"), ("2h", "3mo"), ("4h", "6mo"), ("1d", "1y"),
    ("1wk", "5y")
]

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
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        return df

def calculate_rsi(data, period=14):
    """Calculate RSI manually with error handling"""
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
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
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(interval=interval, period=period)
        
        if not data.empty:
            data = convert_to_ist(data)
        return data
    except Exception as e:
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
        
        trend = "Up" if close.iloc[-1] > close.iloc[0] else "Down"
        max_close = close.max()
        min_close = close.min()
        current_close = close.iloc[-1]
        
        fib_levels = calculate_fibonacci_levels(max_close, min_close)
        fib_50 = fib_levels.get('50.0%', (max_close + min_close) / 2)
        
        volatility = close.std() if len(close) > 1 else 0
        
        if len(close) > 0 and close.iloc[0] != 0:
            pct_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
            points_change = close.iloc[-1] - close.iloc[0]
        else:
            pct_change = 0
            points_change = 0
        
        support, resistance = calculate_support_resistance(data)
        
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
            'vs 20 EMA': '🟢 Above' if current_close > ema_20 else '🔴 Below',
            '50 EMA': safe_format_number(ema_50),
            'vs 50 EMA': '🟢 Above' if current_close > ema_50 else '🔴 Below',
            '200 EMA': safe_format_number(ema_200),
            'vs 200 EMA': '🟢 Above' if current_close > ema_200 else '🔴 Below',
        }
    except Exception as e:
        return None

def detect_patterns(data, threshold=30):
    """Detect significant price movements and preceding patterns"""
    try:
        patterns = []
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values if 'Volume' in data.columns else np.zeros(len(close))
        
        if len(close) < 15:
            return patterns
        
        rsi_full = calculate_rsi(data['Close'])
        ema_20 = calculate_ema(data['Close'], 20)
        ema_50 = calculate_ema(data['Close'], 50)
        
        for i in range(10, len(close)):
            move = close[i] - close[i-10]
            
            if abs(move) >= threshold:
                preceding = data.iloc[i-10:i]
                
                volatility_current = preceding['Close'].std()
                volatility_overall = data['Close'][:i].std()
                volatility_burst = volatility_current > volatility_overall * 1.5
                
                vol_avg = volume[:i].mean() if volume.sum() > 0 else 0
                vol_spike = volume[i-1] > vol_avg * 2 if vol_avg > 0 else False
                
                rsi_before = float(rsi_full.iloc[i-10]) if i-10 < len(rsi_full) else 50
                rsi_at_move = float(rsi_full.iloc[i-1]) if i-1 < len(rsi_full) else 50
                rsi_divergence = (rsi_at_move > rsi_before and move < 0) or (rsi_at_move < rsi_before and move > 0)
                
                if i >= 20:
                    prior_moves = [close[j] - close[j-1] for j in range(i-10, i)]
                    current_move_direction = 1 if move > 0 else -1
                    prior_moves_direction = [1 if m > 0 else -1 for m in prior_moves]
                    correlation = np.corrcoef([current_move_direction] + prior_moves_direction)[0, 1]
                else:
                    correlation = 0
                
                ema_cross = False
                if len(ema_20) > i and len(ema_50) > i and i >= 11:
                    ema_20_before = float(ema_20.iloc[i-10])
                    ema_50_before = float(ema_50.iloc[i-10])
                    ema_20_now = float(ema_20.iloc[i-1])
                    ema_50_now = float(ema_50.iloc[i-1])
                    ema_cross = (ema_20_before < ema_50_before and ema_20_now > ema_50_now) or \
                               (ema_20_before > ema_50_before and ema_20_now < ema_50_now)
                
                support_level = low[max(0, i-20):i].min()
                resistance_level = high[max(0, i-20):i].max()
                breakout = close[i] > resistance_level or close[i] < support_level
                
                body_size = abs(close[i-1] - data['Open'].iloc[i-1])
                total_range = high[i-1] - low[i-1]
                large_body = body_size > total_range * 0.7 if total_range > 0 else False
                
                consecutive_up = sum(1 for j in range(i-5, i) if close[j] > close[j-1])
                consecutive_down = sum(1 for j in range(i-5, i) if close[j] < close[j-1])
                
                patterns.append({
                    'Index': i,
                    'DateTime': data.index[i],
                    'Move (Points)': move,
                    'Move (%)': (move / close[i-10] * 100) if close[i-10] != 0 else 0,
                    'Direction': '🟢 Up' if move > 0 else '🔴 Down',
                    'Volatility_Burst': '✓ Yes' if volatility_burst else '✗ No',
                    'Volume_Spike': '✓ Yes' if vol_spike else '✗ No',
                    'RSI_Before': f"{rsi_before:.1f}",
                    'RSI_At_Move': f"{rsi_at_move:.1f}",
                    'RSI_Divergence': '✓ Yes' if rsi_divergence else '✗ No',
                    'Price_Correlation': f"{correlation:.2f}",
                    'EMA_Crossover': '✓ Yes' if ema_cross else '✗ No',
                    'Support/Resistance_Break': '✓ Yes' if breakout else '✗ No',
                    'Large_Body_Candle': '✓ Yes' if large_body else '✗ No',
                    'Consecutive_Up': consecutive_up,
                    'Consecutive_Down': consecutive_down
                })
        
        return patterns
    except Exception as e:
        st.warning(f"Pattern detection error: {str(e)}")
        return []

# Title
st.markdown('<h1 class="main-header">🚀 PRO ALGO TRADING DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Panel")

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

st.sidebar.subheader("🎛️ Advanced Settings")
pattern_threshold = st.sidebar.slider("Pattern Detection Threshold (Points)", 10, 100, 30, key="pattern_threshold")
api_delay = st.sidebar.slider("API Delay (seconds)", 1.0, 3.0, 1.5, 0.5, key="api_delay")

if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ticker1_data' not in st.session_state:
    st.session_state.ticker1_data = None
if 'ticker2_data' not in st.session_state:
    st.session_state.ticker2_data = None

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
            st.sidebar.success("✅ Data fetched successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.session_state.data_fetched = False

# Main Analysis Section
if st.session_state.data_fetched and st.session_state.ticker1_data is not None:
    data1 = st.session_state.ticker1_data
    data2 = st.session_state.ticker2_data
    
    if not data1.empty:
        try:
            current_price_1 = float(data1['Close'].iloc[-1])
            first_price_1 = float(data1['Close'].iloc[0])
            pct_change_1 = safe_percentage(current_price_1, first_price_1)
        except:
            current_price_1 = 0
            pct_change_1 = 0
        
        # Key Metrics
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
                st.metric(f"{ticker1} Current Price", safe_format_number(current_price_1), f"{safe_format_number(pct_change_1)}%")
            with col2:
                st.metric(f"{ticker2} Current Price", safe_format_number(current_price_2), f"{safe_format_number(pct_change_2)}%")
            with col3:
                st.metric("Ratio (T1/T2)", safe_format_number(ratio_value, 4), f"{safe_format_number(ratio_pct_change)}%")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{ticker1} Current Price", safe_format_number(current_price_1), f"{safe_format_number(pct_change_1)}%")
            with col2:
                rsi_1 = calculate_rsi(data1['Close'])
                current_rsi_1 = float(rsi_1.iloc[-1]) if len(rsi_1) > 0 else 50
                st.metric("RSI", safe_format_number(current_rsi_1), 
                         "Oversold" if current_rsi_1 < 30 else "Overbought" if current_rsi_1 > 70 else "Neutral")
            with col3:
                volatility_1 = data1['Close'].std()
                st.metric("Volatility", safe_format_number(volatility_1))
        
        st.markdown("---")
        
        # Ratio Analysis
        if include_ratio and data2 is not None and not data2.empty:
            st.header("📊 Ratio Analysis")
            
            try:
                min_len = min(len(data1), len(data2))
                data1_aligned = data1.iloc[:min_len].copy()
                data2_aligned = data2.iloc[:min_len].copy()
                
                ratio_data = data1_aligned['Close'] / data2_aligned['Close']
                
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
        
        # Multi-Timeframe Analysis for Ticker 1
        st.header(f"📈 Multi-Timeframe Analysis - {ticker1}")
        
        analysis_results = []
        
        with st.spinner("🔍 Performing multi-timeframe analysis..."):
            progress_bar = st.progress(0)
            for idx, (tf_interval, tf_period) in enumerate(TIMEFRAMES):
                try:
                    tf_data = fetch_data_with_delay(ticker1, tf_interval, tf_period, delay=api_delay)
                    if not tf_data.empty:
                        result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                        if result:
                            analysis_results.append(result)
                except Exception as e:
                    pass
                
                progress_bar.progress((idx + 1) / len(TIMEFRAMES))
        
        if analysis_results:
            mtf_df = pd.DataFrame(analysis_results)
            st.dataframe(mtf_df, use_container_width=True, height=500)
            
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
        
        st.markdown("---")
        
        # Multi-Timeframe Analysis for Ticker 2
        if include_ratio and data2 is not None and not data2.empty:
            st.header(f"📈 Multi-Timeframe Analysis - {ticker2}")
            
            analysis_results_t2 = []
            
            with st.spinner(f"Performing multi-timeframe analysis for {ticker2}..."):
                progress_bar = st.progress(0)
                for idx, (tf_interval, tf_period) in enumerate(TIMEFRAMES):
                    try:
                        tf_data = fetch_data_with_delay(ticker2, tf_interval, tf_period, delay=api_delay)
                        if not tf_data.empty:
                            result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                            if result:
                                analysis_results_t2.append(result)
                    except Exception as e:
                        pass
                    
                    progress_bar.progress((idx + 1) / len(TIMEFRAMES))
            
            if analysis_results_t2:
                mtf_df_t2 = pd.DataFrame(analysis_results_t2)
                st.dataframe(mtf_df_t2, use_container_width=True, height=500)
                
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
        
        st.markdown("---")
        
        # Volatility Analysis
        st.header("📊 Volatility & Returns Analysis")
        
        try:
            returns = data1['Close'].pct_change()
            volatility = returns.rolling(window=min(20, len(returns))).std() * np.sqrt(252) * 100
            
            vol_clean = volatility.dropna()
            if len(vol_clean) > 5:
                vol_bins_cat = pd.qcut(vol_clean, q=5, labels=False, duplicates='drop')
                bin_edges = pd.qcut(vol_clean, q=5, retbins=True, duplicates='drop')[1]
                
                bin_labels = []
                for i in range(len(bin_edges)-1):
                    label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                    bin_labels.append(label)
                
                vol_descriptions = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                vol_bin_names = [f"{vol_descriptions[i]} ({bin_labels[i]})" for i in range(len(bin_labels))]
                vol_bins = pd.Series([vol_bin_names[int(x)] if not pd.isna(x) else 'Unknown' for x in vol_bins_cat], 
                                    index=vol_clean.index)
                
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
                
                st.subheader("📈 Volatility Statistics")
                
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
                
                current_vol_bin = vol_bins.iloc[-1] if len(vol_bins) > 0 else 'Unknown'
                st.success(f"📊 **Current Volatility Regime:** {current_vol_bin}")
        
        except Exception as e:
            st.warning(f"Volatility analysis: {str(e)}")
        
        st.markdown("---")
        
        # Pattern Recognition
        st.header("🔍 Advanced Pattern Recognition")
        
        try:
            patterns = detect_patterns(data1, threshold=pattern_threshold)
            
            if patterns:
                pattern_df = pd.DataFrame(patterns)
                pattern_df['DateTime'] = pattern_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(pattern_df, use_container_width=True, height=400)
                
                total_patterns = len(patterns)
                up_moves = sum(1 for p in patterns if 'Up' in str(p['Direction']))
                vol_bursts = sum(1 for p in patterns if '✓' in str(p['Volatility_Burst']))
                rsi_divs = sum(1 for p in patterns if '✓' in str(p['RSI_Divergence']))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patterns", total_patterns)
                with col2:
                    st.metric("Volatility Bursts", f"{vol_bursts}/{total_patterns}")
                with col3:
                    st.metric("RSI Divergences", f"{rsi_divs}/{total_patterns}")
        
        except Exception as e:
            st.warning(f"Pattern analysis: {str(e)}")
        
        st.markdown("---")
        
        # Charts
        st.header("📈 Interactive Technical Charts")
        
        try:
            if include_ratio and data2 is not None and not data2.empty:
                min_len = min(len(data1), len(data2))
                data1_aligned = data1.iloc[:min_len].copy()
                data2_aligned = data2.iloc[:min_len].copy()
                ratio_data = data1_aligned['Close'] / data2_aligned['Close']
                
                st.subheader(f"📊 {ticker1} Price Chart")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=data1_aligned.index, y=data1_aligned['Close'], 
                                        name=ticker1, line=dict(color='blue', width=2)))
                if len(data1_aligned) >= 20:
                    fig1.add_trace(go.Scatter(x=data1_aligned.index, y=calculate_ema(data1_aligned['Close'], 20), 
                                            name='EMA 20', line=dict(color='orange', width=1.5)))
                fig1.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig1, use_container_width=True)
                
                rsi1 = calculate_rsi(data1_aligned['Close'])
                fig1_rsi = go.Figure()
                fig1_rsi.add_trace(go.Scatter(x=data1_aligned.index, y=rsi1, 
                                             name=f'RSI {ticker1}', line=dict(color='blue', width=2)))
                fig1_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig1_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig1_rsi.update_layout(height=250)
                st.plotly_chart(fig1_rsi, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader(f"📊 {ticker2} Price Chart")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data2_aligned.index, y=data2_aligned['Close'], 
                                        name=ticker2, line=dict(color='green', width=2)))
                if len(data2_aligned) >= 20:
                    fig2.add_trace(go.Scatter(x=data2_aligned.index, y=calculate_ema(data2_aligned['Close'], 20), 
                                            name='EMA 20', line=dict(color='orange', width=1.5)))
                fig2.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig2, use_container_width=True)
                
                rsi2 = calculate_rsi(data2_aligned['Close'])
                fig2_rsi = go.Figure()
                fig2_rsi.add_trace(go.Scatter(x=data2_aligned.index, y=rsi2, 
                                             name=f'RSI {ticker2}', line=dict(color='green', width=2)))
                fig2_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig2_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig2_rsi.update_layout(height=250)
                st.plotly_chart(fig2_rsi, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("📊 Ratio Chart")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data1_aligned.index, y=ratio_data, 
                                        name='Ratio', line=dict(color='purple', width=2)))
                fig3.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig3, use_container_width=True)
                
                rsi_ratio = calculate_rsi(ratio_data)
                fig3_rsi = go.Figure()
                fig3_rsi.add_trace(go.Scatter(x=data1_aligned.index, y=rsi_ratio, 
                                             name='RSI Ratio', line=dict(color='purple', width=2)))
                fig3_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig3_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig3_rsi.update_layout(height=250)
                st.plotly_chart(fig3_rsi, use_container_width=True)
                
            else:
                fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{ticker1} Price', 'RSI'), 
                                   vertical_spacing=0.15, row_heights=[0.7, 0.3])
                
                fig.add_trace(go.Candlestick(x=data1.index, open=data1['Open'], high=data1['High'],
                                            low=data1['Low'], close=data1['Close'], name='OHLC'), row=1, col=1)
                
                if len(data1) >= 20:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 20), 
                                            name='EMA 20', line=dict(color='orange', width=2)), row=1, col=1)
                
                rsi = calculate_rsi(data1['Close'])
                fig.add_trace(go.Scatter(x=data1.index, y=rsi, name='RSI', 
                                        line=dict(color='purple', width=2)), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
        
        st.markdown("---")
        
        # Returns Distribution
        st.header("📊 Statistical Distribution Analysis")
        
        try:
            returns_points = data1['Close'].diff().dropna()
            returns_pct = data1['Close'].pct_change().dropna() * 100
            
            if len(returns_points) > 0:
                mu_points = returns_points.mean()
                sigma_points = returns_points.std()
                
                fig_dist = make_subplots(rows=2, cols=1, 
                                        subplot_titles=('Returns (Points)', 'Returns (%)'),
                                        vertical_spacing=0.15)
                
                fig_dist.add_trace(go.Histogram(x=returns_points, nbinsx=50, name='Returns Points', 
                                               marker_color='blue', opacity=0.7), row=1, col=1)
                
                fig_dist.add_trace(go.Histogram(x=returns_pct, nbinsx=50, name='Returns %', 
                                               marker_color='green', opacity=0.7), row=2, col=1)
                
                fig_dist.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Z-Score Analysis
                st.subheader("📊 Z-Score Analysis")
                
                z_returns_points = (returns_points - mu_points) / sigma_points
                current_z_points = z_returns_points.iloc[-1] if len(z_returns_points) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Z-Score", safe_format_number(current_z_points, 3))
                with col2:
                    interpretation = "🔴 Extreme" if abs(current_z_points) > 2 else "🟡 Moderate" if abs(current_z_points) > 1 else "🟢 Normal"
                    st.metric("Status", interpretation)
                with col3:
                    forecast = "Mean Reversion Expected" if abs(current_z_points) > 2 else "Normal Range"
                    st.metric("Forecast", forecast)
        
        except Exception as e:
            st.warning(f"Distribution analysis: {str(e)}")
        
        st.markdown("---")
        
        # Statistical Hypothesis Testing
        st.header("📊 Statistical Hypothesis Testing")
        
        st.markdown("""
        ### Question: Will the market move UP, DOWN, or remain NEUTRAL?
        Using statistical hypothesis testing with **95% confidence (α = 0.05)**
        """)
        
        try:
            from scipy import stats
            
            returns = data1['Close'].pct_change().dropna()
            
            if len(returns) > 30:
                # Test 1: Upward Movement
                st.subheader("📈 Test 1: Upward Movement")
                
                st.markdown("""
                **H₀:** Market will NOT move up (μ ≤ 0)  
                **H₁:** Market WILL move up (μ > 0)  
                **α = 0.05**
                """)
                
                t_stat, p_value_two = stats.ttest_1samp(returns, 0)
                p_value_up = p_value_two / 2 if t_stat > 0 else 1 - (p_value_two / 2)
                
                mean_return = returns.mean() * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_up:.4f}")
                
                if p_value_up < 0.05 and mean_return > 0:
                    st.success(f"""
                    ✅ **REJECT H₀** (p = {p_value_up:.4f} < 0.05)
                    
                    **Conclusion:** Strong statistical evidence for **UPWARD** movement.
                    Mean return of {mean_return:.3f}% is significant.
                    Confidence: {(1-p_value_up)*100:.1f}%
                    """)
                else:
                    st.info(f"""
                    ❌ **FAIL TO REJECT H₀** (p = {p_value_up:.4f})
                    
                    Insufficient evidence for upward movement.
                    """)
                
                st.markdown("---")
                
                # Test 2: Downward Movement
                st.subheader("📉 Test 2: Downward Movement")
                
                st.markdown("""
                **H₀:** Market will NOT move down (μ ≥ 0)  
                **H₁:** Market WILL move down (μ < 0)  
                **α = 0.05**
                """)
                
                p_value_down = p_value_two / 2 if t_stat < 0 else 1 - (p_value_two / 2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_down:.4f}")
                
                if p_value_down < 0.05 and mean_return < 0:
                    st.error(f"""
                    ✅ **REJECT H₀** (p = {p_value_down:.4f} < 0.05)
                    
                    **Conclusion:** Strong statistical evidence for **DOWNWARD** movement.
                    Mean return of {mean_return:.3f}% is significant.
                    Confidence: {(1-p_value_down)*100:.1f}%
                    """)
                else:
                    st.info(f"""
                    ❌ **FAIL TO REJECT H₀** (p = {p_value_down:.4f})
                    
                    Insufficient evidence for downward movement.
                    """)
                
                st.markdown("---")
                
                # Test 3: Neutral Movement
                st.subheader("⚖️ Test 3: Neutral Movement")
                
                st.markdown("""
                **H₀:** Market IS neutral (μ = 0)  
                **H₁:** Market is NOT neutral (μ ≠ 0)  
                **α = 0.05**
                """)
                
                p_value_neutral = p_value_two
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_neutral:.4f}")
                
                if p_value_neutral >= 0.05:
                    st.info(f"""
                    ✅ **FAIL TO REJECT H₀** (p = {p_value_neutral:.4f} ≥ 0.05)
                    
                    **Conclusion:** Market is statistically **NEUTRAL/SIDEWAYS**.
                    Mean return {mean_return:.3f}% not significantly different from zero.
                    Confidence: {p_value_neutral*100:.1f}%
                    """)
                else:
                    st.warning(f"""
                    ❌ **REJECT H₀** (p = {p_value_neutral:.4f} < 0.05)
                    
                    Market has directional bias (not neutral).
                    """)
                
                st.markdown("---")
                
                # Final Verdict
                st.subheader("🎯 Statistical Verdict")
                
                if p_value_up < 0.05 and mean_return > 0 and p_value_up < p_value_down:
                    verdict = "📈 **UPWARD MOVEMENT**"
                    color = "success"
                    confidence = (1 - p_value_up) * 100
                elif p_value_down < 0.05 and mean_return < 0 and p_value_down < p_value_up:
                    verdict = "📉 **DOWNWARD MOVEMENT**"
                    color = "error"
                    confidence = (1 - p_value_down) * 100
                elif p_value_neutral >= 0.05:
                    verdict = "⚖️ **NEUTRAL/SIDEWAYS**"
                    color = "info"
                    confidence = p_value_neutral * 100
                else:
                    verdict = "🟡 **INCONCLUSIVE**"
                    color = "warning"
                    confidence = 50
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 1rem; color: white;">
                    <h2 style="color: white;">{verdict}</h2>
                    <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                    <p>Based on {len(returns)} observations | Timeframe: {interval}/{period}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"""
                **Summary:**
                - Sample Size: {len(returns)} returns
                - Mean Return: {mean_return:.3f}%
                - T-Statistic: {t_stat:.3f}
                - Upward p-value: {p_value_up:.4f}
                - Downward p-value: {p_value_down:.4f}
                - Neutral p-value: {p_value_neutral:.4f}
                
                **Interpretation:** Statistical tests {'support' if confidence > 90 else 'moderately support' if confidence > 70 else 'weakly support'} the {verdict} forecast.
                """)
                
        except ImportError:
            st.error("scipy required. Install: pip install scipy")
        except Exception as e:
            st.error(f"Statistical testing error: {str(e)}")
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("""
        <div style="background-color: #fee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f00;">
        <strong>⚠️ DISCLAIMER:</strong> This analysis is for educational purposes only. 
        Trading involves substantial risk. Always use proper risk management and consult a financial advisor.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Unable to fetch data. Please check ticker symbol.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Pro Algo Trading Dashboard</h2>
        <p style="font-size: 1.2rem;">Configure settings and click "Fetch Data & Analyze"</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ### 🚀 Features:
    - Multi-timeframe analysis across 9 timeframes
    - Advanced pattern recognition
    - Statistical hypothesis testing (p-values)
    - Volatility regime analysis
    - Ratio analysis for pairs trading
    - Z-score normalization
    - Professional charts with EMAs
    
    ### 📝 Quick Start:
    1. Select ticker(s)
    2. Choose interval & period
    3. Click "Fetch Data & Analyze"
    4. Review statistical forecasts
    """)
