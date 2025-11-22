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
        
        # Ratio Analysis Section (Only if enabled)
        if include_ratio and data2 is not None and not data2.empty:
            st.header("📊 Ratio Analysis")
            
            try:
                # Calculate Ratio
                ratio_data = data1['Close'] / data2['Close']
                
                # Create comparison table
                ratio_df = pd.DataFrame({
                    'DateTime (IST)': data1.index.strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker1 Price': data1['Close'].values,
                    'Ticker2 Price': data2['Close'].values,
                    'Ratio': ratio_data.values,
                    'RSI Ticker1': calculate_rsi(data1['Close']).values,
                    'RSI Ticker2': calculate_rsi(data2['Close']).values,
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
            
            # Create bins
            vol_bins = pd.qcut(volatility.dropna(), q=min(5, len(volatility.dropna())), labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            
            vol_analysis = pd.DataFrame({
                'DateTime (IST)': data1.index[len(data1) - len(vol_bins):].strftime('%Y-%m-%d %H:%M:%S'),
                'Volatility Bin': vol_bins.values,
                'Returns (Points)': data1['Close'].diff().iloc[len(data1) - len(vol_bins):].values,
                'Returns (%)': (returns.iloc[len(data1) - len(vol_bins):].values * 100)
            })
            
            st.dataframe(vol_analysis.tail(30), use_container_width=True, height=300)
            
            current_vol_bin = vol_bins.iloc[-1] if len(vol_bins) > 0 else 'Unknown'
            st.success(f"📊 **Current Volatility Regime:** {current_vol_bin}")
            
            # Volatility insights
            avg_return_by_vol = vol_analysis.groupby('Volatility Bin')['Returns (%)'].mean()
            st.write("**Average Returns by Volatility Regime:**")
            st.bar_chart(avg_return_by_vol)
            
        except Exception as e:
            st.warning(f"Volatility analysis requires more data points: {str(e)}")
        
        st.markdown("---")
        
        # Pattern Recognition
        st.header("🔍 Pattern Recognition & Historical Similarities")
        
        try:
            patterns = detect_patterns(data1, threshold=pattern_threshold)
            
            if patterns:
                pattern_df = pd.DataFrame(patterns)
                pattern_df['DateTime'] = pattern_df['DateTime'].astype(str)
                
                st.dataframe(pattern_df, use_container_width=True, height=300)
                
                st.info(f"""
                **Pattern Analysis:**
                - Detected {len(patterns)} significant movements (>{pattern_threshold} points)
                - Volatility bursts preceded {sum(1 for p in patterns if p['Volatility_Burst'])} moves
                - RSI divergences detected in {sum(1 for p in patterns if p['RSI_Divergence'])} cases
                
                **Current Market Context:** Monitoring for similar pre-move patterns...
                """)
            else:
                st.info("No significant patterns detected in current timeframe. Consider lowering threshold or using longer period.")
        except Exception as e:
            st.warning(f"Pattern recognition requires more data: {str(e)}")
        
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
        
        # Returns Distribution
        st.header("📊 Statistical Distribution Analysis")
        
        try:
            returns_points = data1['Close'].diff().dropna()
            returns_pct = data1['Close'].pct_change().dropna() * 100
            
            fig_dist = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Returns Distribution (Points)', 'Returns Distribution (%)'),
            )
            
            fig_dist.add_trace(go.Histogram(x=returns_points, nbinsx=50, name='Points', 
                                           marker_color='blue'), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=returns_pct, nbinsx=50, name='Percentage', 
                                           marker_color='green'), row=1, col=2)
            
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Z-Score Analysis
            st.subheader("📈 Z-Score Statistical Analysis")
            
            z_returns_points = (returns_points - returns_points.mean()) / returns_points.std()
            z_returns_pct = (returns_pct - returns_pct.mean()) / returns_pct.std()
            
            z_df = pd.DataFrame({
                'DateTime (IST)': data1.index[1:].strftime('%Y-%m-%d %H:%M:%S')[:len(z_returns_points)],
                'Returns (Points)': returns_points.values,
                'Z-Score (Points)': z_returns_points.values,
                'Returns (%)': returns_pct.values,
                'Z-Score (%)': z_returns_pct.values
            })
            
            st.dataframe(z_df.tail(30), use_container_width=True, height=300)
            
            current_z = z_returns_points.iloc[-1] if len(z_returns_points) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Z-Score", safe_format_number(current_z, 3))
            with col2:
                interpretation = "🔴 Extreme High" if current_z > 2 else "🟢 Extreme Low" if current_z < -2 else "🟡 Normal"
                st.metric("Interpretation", interpretation)
            with col3:
                forecast = "Reversal Down" if current_z > 2 else "Reversal Up" if current_z < -2 else "Continue Trend"
                st.metric("Forecast", forecast)
            
            st.info(f"""
            **Z-Score Interpretation Guide:**
            - **Z > 2.0:** Price movement is extremely high (99th percentile) - expect mean reversion downward
            - **Z < -2.0:** Price movement is extremely low (1st percentile) - expect mean reversion upward
            - **-1 < Z < 1:** Normal price behavior (68% of data falls here)
            
            **Current Analysis:** Z-Score is {safe_format_number(current_z, 2)}, suggesting the recent price movement is {'extremely high and likely to reverse' if current_z > 2 else 'extremely low and likely to bounce' if current_z < -2 else 'within normal ranges'}
            """)
            
            # Ratio Z-Score (if applicable)
            if include_ratio and data2 is not None and not data2.empty:
                ratio_data = data1['Close'] / data2['Close']
                ratio_returns = ratio_data.diff().dropna()
                z_ratio = (ratio_returns - ratio_returns.mean()) / ratio_returns.std()
                
                st.subheader("📊 Ratio Z-Score Analysis")
                current_z_ratio = z_ratio.iloc[-1] if len(z_ratio) > 0 else 0
                st.metric("Ratio Z-Score", safe_format_number(current_z_ratio, 3))
                
                st.write(f"**Ratio Signal:** {'Ratio extremely high - T1 overvalued vs T2' if current_z_ratio > 2 else 'Ratio extremely low - T1 undervalued vs T2' if current_z_ratio < -2 else 'Ratio in normal range'}")
        
        except Exception as e:
            st.warning(f"Statistical analysis requires more data points: {str(e)}")
        
        st.markdown("---")
        
        # Final Trading Recommendation
        st.header("🎯 FINAL TRADING RECOMMENDATION")
        
        try:
            # Gather all signals
            signals = []
            signal_weights = []
            
            # 1. Multi-timeframe trend signal
            if 'up_trends' in locals() and 'down_trends' in locals():
                trend_signal = 1 if up_trends > down_trends else -1 if down_trends > up_trends else 0
                signals.append(trend_signal)
                signal_weights.append(0.3)
            
            # 2. RSI signal
            if 'avg_rsi' in locals():
                rsi_signal = -1 if avg_rsi > 70 else 1 if avg_rsi < 30 else 0
                signals.append(rsi_signal)
                signal_weights.append(0.2)
            
            # 3. Z-Score signal
            if 'current_z' in locals():
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
            
            # Display recommendation in a prominent box
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
                st.markdown(f"""
                **Component Signals:**
                1. Multi-Timeframe Trend: {'🟢 Bullish' if up_trends > down_trends else '🔴 Bearish' if down_trends > up_trends else '🟡 Neutral'} ({up_trends}/{len(analysis_results)})
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
            st.info(f"""
            **Why This Signal:**
            
            The algorithmic analysis across multiple timeframes and indicators suggests {action.lower()} based on:
            
            1. **Trend Alignment:** {up_trends} out of {len(analysis_results)} timeframes show upward momentum
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
