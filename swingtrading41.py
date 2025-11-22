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
        return {'50.0%': high - 0.5 * diff}
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
        except:
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
            rsi_status = "🟢 Oversold"
        elif current_rsi > 70:
            rsi_status = "🔴 Overbought"
        else:
            rsi_status = "🟡 Neutral"
        
        def safe_ema(data, period):
            if len(data) >= period:
                return calculate_ema(data, period).iloc[-1]
            return data.iloc[-1] if len(data) > 0 else np.nan
        
        ema_20 = safe_ema(close, 20)
        ema_50 = safe_ema(close, 50)
        ema_200 = safe_ema(close, 200)
        
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
            'RSI Status': rsi_status,
            '20 EMA': safe_format_number(ema_20),
            '50 EMA': safe_format_number(ema_50),
            '200 EMA': safe_format_number(ema_200),
            'vs 20 EMA': '🟢 Above' if current_close > ema_20 else '🔴 Below',
            'vs 50 EMA': '🟢 Above' if current_close > ema_50 else '🔴 Below',
            'vs 200 EMA': '🟢 Above' if current_close > ema_200 else '🔴 Below',
        }
    except Exception as e:
        return None

def detect_patterns(data, threshold=30):
    """Detect significant price movements with detailed analysis"""
    try:
        patterns = []
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        if len(close) < 15:
            return patterns
        
        rsi_full = calculate_rsi(data['Close'])
        
        for i in range(10, len(close)):
            move = close[i] - close[i-10]
            
            if abs(move) >= threshold:
                preceding = data.iloc[i-10:i]
                volatility_current = preceding['Close'].std()
                volatility_overall = data['Close'][:i].std()
                
                # Use explicit boolean conversion
                vol_burst = "✓ Yes" if float(volatility_current) > float(volatility_overall) * 1.5 else "✗ No"
                
                rsi_before = float(rsi_full.iloc[i-10]) if i-10 < len(rsi_full) else 50.0
                rsi_at_move = float(rsi_full.iloc[i-1]) if i-1 < len(rsi_full) else 50.0
                
                # Explicit boolean check
                has_divergence = (rsi_at_move > rsi_before and move < 0) or (rsi_at_move < rsi_before and move > 0)
                rsi_div = "✓ Yes" if has_divergence else "✗ No"
                
                patterns.append({
                    'DateTime': data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'Move (Points)': float(move),
                    'Move (%)': float((move / close[i-10] * 100) if close[i-10] != 0 else 0),
                    'Direction': '🟢 Up' if move > 0 else '🔴 Down',
                    'Volatility_Burst': vol_burst,
                    'RSI_Before': f"{rsi_before:.1f}",
                    'RSI_At_Move': f"{rsi_at_move:.1f}",
                    'RSI_Divergence': rsi_div
                })
        
        return patterns
    except Exception as e:
        st.warning(f"Pattern detection: {str(e)}")
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
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=4, key="interval_select")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=6, key="period_select")

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
    with st.spinner("⏳ Fetching data..."):
        try:
            st.session_state.ticker1_data = fetch_data_with_delay(ticker1, interval, period, delay=api_delay)
            if include_ratio and ticker2:
                st.session_state.ticker2_data = fetch_data_with_delay(ticker2, interval, period, delay=api_delay)
            else:
                st.session_state.ticker2_data = None
            st.session_state.data_fetched = True
            st.sidebar.success("✅ Data fetched!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.session_state.data_fetched = False

# Main Analysis
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
        
        st.header("📊 Market Overview")
        
        if include_ratio and data2 is not None and not data2.empty:
            try:
                current_price_2 = float(data2['Close'].iloc[-1])
                first_price_2 = float(data2['Close'].iloc[0])
                pct_change_2 = safe_percentage(current_price_2, first_price_2)
                ratio_value = current_price_1 / current_price_2 if current_price_2 != 0 else 0
                first_ratio = first_price_1 / first_price_2 if first_price_2 != 0 else 0
                ratio_pct_change = safe_percentage(ratio_value, first_ratio)
            except:
                current_price_2 = 0
                pct_change_2 = 0
                ratio_value = 0
                ratio_pct_change = 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{ticker1} Price", safe_format_number(current_price_1), f"{safe_format_number(pct_change_1)}%")
            with col2:
                st.metric(f"{ticker2} Price", safe_format_number(current_price_2), f"{safe_format_number(pct_change_2)}%")
            with col3:
                st.metric("Ratio", safe_format_number(ratio_value, 4), f"{safe_format_number(ratio_pct_change)}%")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{ticker1} Price", safe_format_number(current_price_1), f"{safe_format_number(pct_change_1)}%")
            with col2:
                rsi_1 = calculate_rsi(data1['Close'])
                current_rsi_1 = float(rsi_1.iloc[-1]) if len(rsi_1) > 0 else 50
                st.metric("RSI", safe_format_number(current_rsi_1))
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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Ratio analysis error: {str(e)}")
            
            st.markdown("---")
        
        # Multi-Timeframe Analysis
        st.header("📈 Multi-Timeframe Analysis")
        
        # Define timeframes
        timeframes = [
            ("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("30m", "1mo"),
            ("1h", "1mo"), ("2h", "3mo"), ("4h", "6mo"), ("1d", "1y"),
            ("1wk", "5y")
        ]
        
        analysis_results = []
        
        with st.spinner("🔍 Analyzing timeframes..."):
            progress_bar = st.progress(0)
            for idx, (tf_interval, tf_period) in enumerate(timeframes):
                try:
                    tf_data = fetch_data_with_delay(ticker1, tf_interval, tf_period, delay=api_delay)
                    if not tf_data.empty:
                        result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                        if result:
                            analysis_results.append(result)
                except:
                    pass
                progress_bar.progress((idx + 1) / len(timeframes))
        
        if analysis_results:
            mtf_df = pd.DataFrame(analysis_results)
            st.dataframe(mtf_df, use_container_width=True, height=500)
            
            up_trends = sum(1 for r in analysis_results if r['Trend'] == 'Up')
            down_trends = len(analysis_results) - up_trends
            
            try:
                avg_rsi = np.mean([float(r['RSI']) for r in analysis_results if r['RSI'] != 'N/A'])
            except:
                avg_rsi = 50
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bullish TFs", f"{up_trends}/{len(analysis_results)}")
            with col2:
                st.metric("Bearish TFs", f"{down_trends}/{len(analysis_results)}")
            with col3:
                st.metric("Avg RSI", safe_format_number(avg_rsi))
            with col4:
                bias = "🟢 BULLISH" if up_trends > down_trends else "🔴 BEARISH" if down_trends > up_trends else "🟡 NEUTRAL"
                st.metric("Bias", bias)
        
        st.markdown("---")
        
        # Pattern Recognition
        st.header("🔍 Pattern Recognition")
        
        try:
            patterns = detect_patterns(data1, threshold=pattern_threshold)
            
            if patterns:
                pattern_df = pd.DataFrame(patterns)
                st.dataframe(pattern_df, use_container_width=True, height=400)
                
                total_patterns = len(patterns)
                vol_bursts = sum(1 for p in patterns if "✓" in str(p['Volatility_Burst']))
                rsi_divs = sum(1 for p in patterns if "✓" in str(p['RSI_Divergence']))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patterns", total_patterns)
                with col2:
                    st.metric("Volatility Bursts", f"{vol_bursts}")
                with col3:
                    st.metric("RSI Divergences", f"{rsi_divs}")
        except Exception as e:
            st.warning(f"Pattern analysis: {str(e)}")
        
        st.markdown("---")
        
        # Charts
        st.header("📈 Technical Charts")
        
        try:
            if include_ratio and data2 is not None and not data2.empty:
                min_len = min(len(data1), len(data2))
                data1_aligned = data1.iloc[:min_len].copy()
                data2_aligned = data2.iloc[:min_len].copy()
                ratio_data = data1_aligned['Close'] / data2_aligned['Close']
                
                st.subheader(f"📊 {ticker1}")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=data1_aligned.index, y=data1_aligned['Close'], name=ticker1, line=dict(color='blue', width=2)))
                if len(data1_aligned) >= 20:
                    fig1.add_trace(go.Scatter(x=data1_aligned.index, y=calculate_ema(data1_aligned['Close'], 20), name='EMA 20', line=dict(color='orange')))
                fig1.update_layout(height=400, xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig1, use_container_width=True)
                
                rsi1 = calculate_rsi(data1_aligned['Close'])
                fig1_rsi = go.Figure()
                fig1_rsi.add_trace(go.Scatter(x=data1_aligned.index, y=rsi1, name='RSI', line=dict(color='blue')))
                fig1_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig1_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig1_rsi.update_layout(height=250, xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig1_rsi, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader(f"📊 {ticker2}")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data2_aligned.index, y=data2_aligned['Close'], name=ticker2, line=dict(color='green', width=2)))
                if len(data2_aligned) >= 20:
                    fig2.add_trace(go.Scatter(x=data2_aligned.index, y=calculate_ema(data2_aligned['Close'], 20), name='EMA 20', line=dict(color='orange')))
                fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig2, use_container_width=True)
                
                rsi2 = calculate_rsi(data2_aligned['Close'])
                fig2_rsi = go.Figure()
                fig2_rsi.add_trace(go.Scatter(x=data2_aligned.index, y=rsi2, name='RSI', line=dict(color='green')))
                fig2_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig2_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig2_rsi.update_layout(height=250, xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig2_rsi, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("📊 Ratio")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data1_aligned.index, y=ratio_data, name='Ratio', line=dict(color='purple', width=2)))
                fig3.update_layout(height=400, xaxis_title="Date", yaxis_title="Ratio")
                st.plotly_chart(fig3, use_container_width=True)
                
                rsi_ratio = calculate_rsi(ratio_data)
                fig3_rsi = go.Figure()
                fig3_rsi.add_trace(go.Scatter(x=data1_aligned.index, y=rsi_ratio, name='RSI Ratio', line=dict(color='purple')))
                fig3_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig3_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig3_rsi.update_layout(height=250, xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig3_rsi, use_container_width=True)
                
            else:
                fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{ticker1} Price', 'RSI'), vertical_spacing=0.15, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=data1.index, open=data1['Open'], high=data1['High'], low=data1['Low'], close=data1['Close'], name='OHLC'), row=1, col=1)
                if len(data1) >= 20:
                    fig.add_trace(go.Scatter(x=data1.index, y=calculate_ema(data1['Close'], 20), name='EMA 20', line=dict(color='orange')), row=1, col=1)
                rsi = calculate_rsi(data1['Close'])
                fig.add_trace(go.Scatter(x=data1.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
        
        st.markdown("---")
        
        # Statistical Hypothesis Testing
        st.header("📊 Statistical Hypothesis Testing")
        
        try:
            from scipy import stats
            
            returns = data1['Close'].pct_change().dropna()
            
            if len(returns) > 30:
                mean_return = returns.mean()
                std_return = returns.std()
                
                # One-sample t-test
                t_stat, p_value_two = stats.ttest_1samp(returns, 0)
                
                # Calculate one-tailed p-values
                p_value_up = p_value_two / 2 if t_stat > 0 else 1 - (p_value_two / 2)
                p_value_down = p_value_two / 2 if t_stat < 0 else 1 - (p_value_two / 2)
                p_value_neutral = p_value_two
                
                st.subheader("📈 Test 1: Upward Movement")
                st.markdown("**H₀:** Market will NOT move up (mean ≤ 0) | **H₁:** Market WILL move up (mean > 0)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return*100:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_up:.4f}")
                
                if p_value_up < 0.05 and mean_return > 0:
                    st.success(f"✅ REJECT H₀ (p={p_value_up:.4f} < 0.05) → **UPWARD MOVEMENT LIKELY** | Confidence: {(1-p_value_up)*100:.1f}%")
                else:
                    st.warning(f"❌ FAIL TO REJECT H₀ (p={p_value_up:.4f} ≥ 0.05) → Insufficient evidence for upward movement")
                
                st.markdown("---")
                
                st.subheader("📉 Test 2: Downward Movement")
                st.markdown("**H₀:** Market will NOT move down (mean ≥ 0) | **H₁:** Market WILL move down (mean < 0)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return*100:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_down:.4f}")
                
                if p_value_down < 0.05 and mean_return < 0:
                    st.error(f"✅ REJECT H₀ (p={p_value_down:.4f} < 0.05) → **DOWNWARD MOVEMENT LIKELY** | Confidence: {(1-p_value_down)*100:.1f}%")
                else:
                    st.warning(f"❌ FAIL TO REJECT H₀ (p={p_value_down:.4f} ≥ 0.05) → Insufficient evidence for downward movement")
                
                st.markdown("---")
                
                st.subheader("⚖️ Test 3: Neutral Movement")
                st.markdown("**H₀:** Market IS neutral (mean = 0) | **H₁:** Market is NOT neutral (mean ≠ 0)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{mean_return*100:.3f}%")
                with col2:
                    st.metric("T-Statistic", f"{t_stat:.3f}")
                with col3:
                    st.metric("P-Value", f"{p_value_neutral:.4f}")
                
                if p_value_neutral >= 0.05:
                    st.info(f"✅ FAIL TO REJECT H₀ (p={p_value_neutral:.4f} ≥ 0.05) → **NEUTRAL/SIDEWAYS LIKELY** | Confidence: {p_value_neutral*100:.1f}%")
                else:
                    st.warning(f"❌ REJECT H₀ (p={p_value_neutral:.4f} < 0.05) → Market NOT neutral, has directional bias")
                
                st.markdown("---")
                
                # Final Verdict
                st.subheader("🎯 Statistical Verdict")
                
                # Determine verdict based on p-values and mean
                if p_value_up < 0.05 and mean_return > 0:
                    verdict = "📈 **UPWARD MOVEMENT**"
                    verdict_color = "green"
                    direction = "UP"
                    confidence = (1 - p_value_up) * 100
                elif p_value_down < 0.05 and mean_return < 0:
                    verdict = "📉 **DOWNWARD MOVEMENT**"
                    verdict_color = "red"
                    direction = "DOWN"
                    confidence = (1 - p_value_down) * 100
                elif p_value_neutral >= 0.05:
                    verdict = "⚖️ **NEUTRAL/SIDEWAYS**"
                    verdict_color = "blue"
                    direction = "NEUTRAL"
                    confidence = p_value_neutral * 100
                else:
                    verdict = "🟡 **INCONCLUSIVE**"
                    verdict_color = "orange"
                    direction = "UNCERTAIN"
                    confidence = 50
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 1rem; color: white;">
                    <h2 style="color: white;">{verdict}</h2>
                    <p style="font-size: 1.2rem;">Statistical Confidence: <strong>{confidence:.1f}%</strong></p>
                    <p>Timeframe: {interval}/{period} | Observations: {len(returns)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"""
                **Statistical Summary:**
                - Mean Return: {mean_return*100:.3f}% per period
                - Std Dev: {std_return*100:.3f}%
                - T-Statistic: {t_stat:.3f}
                - Direction: **{direction}**
                
                **Interpretation:**
                - P < 0.05: Strong evidence (reject null hypothesis)
                - P ≥ 0.05: Weak evidence (fail to reject null hypothesis)
                
                **Trading Implication:** {
                    f"Statistical evidence supports {direction} movement with {confidence:.1f}% confidence. Expected return: {mean_return*100:.2f}% per period."
                    if confidence > 90
                    else f"Moderate evidence for {direction} movement ({confidence:.1f}% confidence). Use caution."
                    if confidence > 70
                    else f"Weak statistical signal. Market direction unclear. Consider waiting for better setup."
                }
                """)
                
            else:
                st.warning("Need at least 30 data points for statistical testing.")
                
        except ImportError:
            st.error("Install scipy: pip install scipy")
        except Exception as e:
            st.error(f"Statistical analysis error: {str(e)}")
        
        st.markdown("---")
        
        # Final Recommendation
        st.header("🎯 FINAL TRADING RECOMMENDATION")
        
        try:
            # Initialize variables
            current_z = 0
            
            # Calculate Z-score
            try:
                returns_points = data1['Close'].diff().dropna()
                if len(returns_points) > 0:
                    mu = returns_points.mean()
                    sigma = returns_points.std()
                    if sigma > 0:
                        z_returns = (returns_points - mu) / sigma
                        current_z = float(z_returns.iloc[-1]) if len(z_returns) > 0 else 0
            except:
                pass
            
            # Gather signals
            signals = []
            weights = []
            
            # Trend signal
            if 'up_trends' in locals() and 'down_trends' in locals():
                trend_signal = 1 if up_trends > down_trends else -1 if down_trends > up_trends else 0
                signals.append(trend_signal)
                weights.append(0.3)
            else:
                trend_signal = 0
            
            # RSI signal
            if 'avg_rsi' in locals():
                rsi_signal = -1 if avg_rsi > 70 else 1 if avg_rsi < 30 else 0
                signals.append(rsi_signal)
                weights.append(0.2)
            else:
                rsi_signal = 0
                avg_rsi = 50
            
            # Z-score signal
            z_signal = -1 if current_z > 2 else 1 if current_z < -2 else 0
            signals.append(z_signal)
            weights.append(0.2)
            
            # EMA signal
            ema_20_val = calculate_ema(data1['Close'], 20).iloc[-1] if len(data1) >= 20 else current_price_1
            ema_50_val = calculate_ema(data1['Close'], 50).iloc[-1] if len(data1) >= 50 else current_price_1
            ema_signal = 1 if current_price_1 > ema_20_val > ema_50_val else -1 if current_price_1 < ema_20_val < ema_50_val else 0
            signals.append(ema_signal)
            weights.append(0.3)
            
            # Calculate total signal
            total_signal = sum(s * w for s, w in zip(signals, weights)) if signals else 0
            
            # ATR for stops
            atr = (data1['High'] - data1['Low']).rolling(14).mean().iloc[-1] if 'High' in data1.columns else data1['Close'].std()
            
            # Generate recommendation
            if total_signal >= 0.3:
                action = "🟢 STRONG BUY"
                entry = current_price_1
                target = current_price_1 + (2.5 * atr)
                sl = current_price_1 - atr
            elif total_signal >= 0.15:
                action = "🟢 BUY"
                entry = current_price_1
                target = current_price_1 + (2 * atr)
                sl = current_price_1 - (0.8 * atr)
            elif total_signal <= -0.3:
                action = "🔴 STRONG SELL"
                entry = current_price_1
                target = current_price_1 - (2.5 * atr)
                sl = current_price_1 + atr
            elif total_signal <= -0.15:
                action = "🔴 SELL"
                entry = current_price_1
                target = current_price_1 - (2 * atr)
                sl = current_price_1 + (0.8 * atr)
            else:
                action = "🟡 HOLD"
                entry = current_price_1
                target = current_price_1
                sl = current_price_1
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 1rem; color: white;">
                <h2 style="color: white;">{action}</h2>
                <p style="font-size: 1.2rem;">Signal Strength: <strong>{abs(total_signal):.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Trade Setup")
                st.markdown(f"""
                **Entry:** {safe_format_number(entry)}
                **Target:** {safe_format_number(target)} (+{safe_format_number(abs((target-entry)/entry*100))}%)
                **Stop Loss:** {safe_format_number(sl)} (-{safe_format_number(abs((sl-entry)/entry*100))}%)
                **Risk/Reward:** 1:2.5
                """)
            
            with col2:
                st.subheader("📊 Signals")
                trend_count = f"{up_trends}/{len(analysis_results)}" if 'up_trends' in locals() else "N/A"
                st.markdown(f"""
                1. Trend: {'🟢 Bull' if trend_signal > 0 else '🔴 Bear' if trend_signal < 0 else '🟡 Neutral'} ({trend_count})
                2. RSI: {'🟢 Oversold' if avg_rsi < 30 else '🔴 Overbought' if avg_rsi > 70 else '🟡 Neutral'} ({safe_format_number(avg_rsi)})
                3. Z-Score: {'🔴 High' if current_z > 2 else '🟢 Low' if current_z < -2 else '🟡 Normal'} ({safe_format_number(current_z, 2)})
                4. EMA: {'🟢 Bull' if ema_signal > 0 else '🔴 Bear' if ema_signal < 0 else '🟡 Mixed'}
                """)
            
            st.info(f"""
            **Analysis:** Based on multi-factor analysis, the recommendation is **{action}**. 
            Current price: {safe_format_number(current_price_1)} with {safe_format_number(pct_change_1)}% change.
            """)
            
        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #fee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f00;">
        <strong>⚠️ DISCLAIMER:</strong> This is for educational purposes only. Not financial advice. 
        Trading involves risk. Always do your own research and consult a financial advisor.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Unable to fetch data. Please check ticker and try again.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Pro Algo Trading Dashboard</h2>
        <p style="font-size: 1.2rem;">Professional algorithmic trading analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Multi-Asset
        - Indices (NIFTY, SENSEX)
        - Crypto (BTC, ETH)
        - Commodities (Gold, Silver)
        - Forex (USD/INR)
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Analytics
        - Multi-timeframe analysis
        - Technical indicators
        - Pattern recognition
        - Statistical testing
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Signals
        - Buy/Sell recommendations
        - Risk management
        - Entry/Target/Stop Loss
        - Hypothesis testing
        """)
    
    st.info("""
    ### 🚀 Quick Start:
    1. Select ticker(s) from sidebar
    2. Choose interval and period
    3. Click "FETCH DATA & ANALYZE"
    4. Review comprehensive analysis and signals
    """)
    
    st.markdown("""
    <div style="text-align: center; color: #888; margin-top: 2rem;">
    <strong>Built for Algo Traders</strong><br>
    Powered by Streamlit • yFinance • Plotly • SciPy
    </div>
    """, unsafe_allow_html=True)
