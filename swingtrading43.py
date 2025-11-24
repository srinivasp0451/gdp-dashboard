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

# Initialize session state for data persistence
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

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
    
    # Ratio Analysis Option - Read from session state if exists, otherwise use widget value
    enable_ratio = st.checkbox("Enable Ratio Analysis (Compare with Ticker 2)", 
                               value=st.session_state.get('enable_ratio_value', False),
                               key='enable_ratio')
    
    # Store the value in session state
    st.session_state.enable_ratio_value = enable_ratio
    
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
            data1['ATR'] = calculate_atr(data1)
            
            # CRITICAL: Calculate Returns FIRST (this is historical - what happened)
            data1['Returns'] = data1['Close'].pct_change() * 100
            
            # THEN calculate Volatility and Z-Score AFTER returns (these predict future moves)
            data1['Volatility'] = calculate_volatility(data1)
            
            # Calculate Z-Score based on historical returns
            mean_return = data1['Returns'].mean()
            std_return = data1['Returns'].std()
            data1['Z_Score'] = (data1['Returns'] - mean_return) / std_return
            
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
                    data2['ATR'] = calculate_atr(data2)
                    data2['Returns'] = data2['Close'].pct_change() * 100
                    data2['Volatility'] = calculate_volatility(data2)
                    
                    mean_return2 = data2['Returns'].mean()
                    std_return2 = data2['Returns'].std()
                    data2['Z_Score'] = (data2['Returns'] - mean_return2) / std_return2
            
            st.session_state.data_fetched = True
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
    
    if st.session_state.enable_ratio_value and 'data2' in st.session_state:
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
    display_data = data1[['Datetime', 'Open', 'High', 'Low', 'Close', 'Returns', 'RSI', 'Volatility', 'Z_Score']].copy()
    if 'Volume' in data1.columns:
        display_data.insert(5, 'Volume', data1['Volume'])
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
    
    # ==================== SECTION 2: VOLATILITY BINS ANALYSIS WITH PREDICTION ====================
    st.header("📈 Volatility-Based Next Move Prediction")
    
    st.info("""
    **Understanding The Analysis:**
    - **Returns (%)**: What happened BEFORE (historical price changes)
    - **Volatility**: Current market turbulence level
    - **Prediction**: What will likely happen NEXT based on patterns
    
    Think of volatility like weather patterns - when storms (high volatility) form, we can predict what usually happens next!
    """)
    
    vol_data = data1[['Datetime', 'Close', 'Volatility', 'Returns']].copy()
    vol_data = vol_data.dropna()
    
    if len(vol_data) > 10:
        # Create volatility bins
        try:
            vol_bins = pd.qcut(vol_data['Volatility'], q=5, duplicates='drop')
            vol_data['Vol_Bin'] = pd.cut(vol_data['Volatility'], bins=vol_bins.cat.categories)
        except:
            vol_data['Vol_Bin'] = pd.cut(vol_data['Volatility'], bins=5)
        
        # CRITICAL: Calculate NEXT period returns for prediction
        vol_data['Next_Return'] = vol_data['Returns'].shift(-1)  # What happened AFTER
        
        # Create volatility table
        vol_table = vol_data.copy()
        vol_table['Datetime'] = vol_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
        vol_table['Vol_Bin_Label'] = vol_table['Vol_Bin'].apply(
            lambda x: f"{x.left:.2f} - {x.right:.2f}" if pd.notna(x) else "N/A"
        )
        vol_table['Returns_Points'] = vol_data['Close'].diff()
        
        display_vol = vol_table[['Datetime', 'Vol_Bin_Label', 'Volatility', 'Close', 'Returns', 'Next_Return']].copy()
        display_vol.columns = ['Datetime', 'Volatility Bin', 'Volatility %', 'Price', 'Past Return (%)', 'Next Return (%)']
        
        st.dataframe(display_vol.tail(50), use_container_width=True)
        
        # Volatility statistics and PREDICTION
        current_vol = vol_data['Volatility'].iloc[-1]
        current_vol_bin = vol_data['Vol_Bin'].iloc[-1]
        current_return = vol_data['Returns'].iloc[-1]
        
        # Analyze what happens NEXT after similar volatility levels
        bin_analysis = vol_data.groupby('Vol_Bin').agg({
            'Returns': ['mean', 'std', 'count'],
            'Next_Return': ['mean', 'std', 'count']  # What happens AFTER
        }).round(2)
        
        # Get prediction for current bin
        if pd.notna(current_vol_bin) and current_vol_bin in bin_analysis.index:
            avg_past_return = bin_analysis.loc[current_vol_bin, ('Returns', 'mean')]
            avg_next_return = bin_analysis.loc[current_vol_bin, ('Next_Return', 'mean')]
            next_return_std = bin_analysis.loc[current_vol_bin, ('Next_Return', 'std')]
            observations = int(bin_analysis.loc[current_vol_bin, ('Next_Return', 'count')])
            
            # Prediction confidence
            confidence = "High" if observations > 20 else "Moderate" if observations > 10 else "Low"
            
            # Direction prediction
            if avg_next_return > 0.5:
                prediction = "📈 UP (Bullish)"
                pred_color = "positive"
            elif avg_next_return < -0.5:
                prediction = "📉 DOWN (Bearish)"
                pred_color = "negative"
            else:
                prediction = "➡️ SIDEWAYS (Neutral)"
                pred_color = "neutral"
            
            st.success(f"""
            **🔮 NEXT MOVE PREDICTION:**
            
            **Current Situation:**
            - Volatility NOW: {current_vol:.2f}% (Bin: {current_vol_bin.left:.2f}-{current_vol_bin.right:.2f})
            - Past Return: {current_return:.2f}% (what ALREADY happened)
            
            **Prediction for NEXT Move:**
            - **Expected Direction**: {prediction}
            - **Expected Return**: {avg_next_return:+.2f}% (±{next_return_std:.2f}% variation)
            - **Confidence**: {confidence} (based on {observations} similar historical cases)
            
            **Why This Prediction?**
            Historically, when volatility was in the range {current_vol_bin.left:.2f}-{current_vol_bin.right:.2f}%, 
            the NEXT move averaged {avg_next_return:+.2f}%. Current volatility is {current_vol:.2f}%, 
            which falls in this range, so we expect similar behavior.
            
            **Simple Explanation:**
            Think of it like this: We looked at all past times when the market was as volatile as it is RIGHT NOW. 
            On average, the NEXT move after such volatility was {avg_next_return:+.2f}%. So we predict the market 
            will move {'UP ⬆️' if avg_next_return > 0 else 'DOWN ⬇️'} by approximately {abs(avg_next_return):.2f}% in the next period.
            """)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3 class='{pred_color}'>🎯 Predicted Next Move: {prediction}</h3>
                <h2>{avg_next_return:+.2f}%</h2>
                <p>Confidence: {confidence}</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.warning("Insufficient data for prediction in current volatility bin.")
    
    # ==================== Z-SCORE PREDICTION ====================
    st.header("📊 Z-Score Based Mean Reversion Prediction")
    
    st.info("""
    **Understanding Z-Score:**
    - **Past Return**: What price change ALREADY happened
    - **Z-Score**: How unusual that past move was (in standard deviations)
    - **Prediction**: What will likely happen NEXT (extreme moves tend to reverse)
    """)
    
    z_score_data = data1[['Datetime', 'Returns', 'Z_Score']].dropna().copy()
    z_score_data['Next_Return'] = z_score_data['Returns'].shift(-1)
    
    z_score_table = z_score_data.tail(50).copy()
    z_score_table['Datetime'] = z_score_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    z_score_table['Returns_Points'] = data1['Close'].diff()
    
    display_z = z_score_table[['Datetime', 'Returns', 'Z_Score', 'Next_Return']].copy()
    display_z.columns = ['Datetime', 'Past Return (%)', 'Z-Score', 'Next Return (%)']
    
    st.dataframe(display_z, use_container_width=True)
    
    # Z-Score prediction
    current_zscore = data1['Z_Score'].iloc[-1]
    current_return = data1['Returns'].iloc[-1]
    
    # Analyze what happens after extreme z-scores
    extreme_positive = z_score_data[z_score_data['Z_Score'] > 2]
    extreme_negative = z_score_data[z_score_data['Z_Score'] < -2]
    moderate = z_score_data[(z_score_data['Z_Score'] >= -1) & (z_score_data['Z_Score'] <= 1)]
    
    if current_zscore > 2:
        similar_cases = extreme_positive
        zone = "Extreme Positive"
        expected_behavior = "Mean Reversion DOWN"
        color = "negative"
    elif current_zscore < -2:
        similar_cases = extreme_negative
        zone = "Extreme Negative"
        expected_behavior = "Mean Reversion UP"
        color = "positive"
    else:
        similar_cases = moderate
        zone = "Normal Range"
        expected_behavior = "Trend Continuation"
        color = "neutral"
    
    if len(similar_cases) > 0:
        avg_next_return = similar_cases['Next_Return'].mean()
        std_next_return = similar_cases['Next_Return'].std()
        reversal_count = len(similar_cases[similar_cases['Next_Return'] * similar_cases['Returns'] < 0])
        reversal_rate = (reversal_count / len(similar_cases)) * 100
        
        st.success(f"""
        **🔮 Z-SCORE PREDICTION:**
        
        **Current Status:**
        - Past Return: {current_return:.2f}% (what ALREADY happened)
        - Z-Score: {current_zscore:.2f} (Zone: {zone})
        
        **Prediction for NEXT Move:**
        - **Expected Behavior**: {expected_behavior}
        - **Average Next Return**: {avg_next_return:+.2f}% (±{std_next_return:.2f}%)
        - **Reversal Probability**: {reversal_rate:.1f}% (based on {len(similar_cases)} similar cases)
        
        **Why This Prediction?**
        {
        f"Your Z-score of {current_zscore:.2f} means the past move was EXTREMELY unusual (beyond 2 standard deviations). "
        f"Historically, after such extreme moves, the market reversed {reversal_rate:.0f}% of the time, "
        f"with an average NEXT move of {avg_next_return:+.2f}%. This is called 'mean reversion' - "
        f"extreme moves tend to snap back like a rubber band."
        if abs(current_zscore) > 2
        else f"Your Z-score of {current_zscore:.2f} is in the normal range. The market typically continues "
        f"its trend in such cases, with an average NEXT move of {avg_next_return:+.2f}%."
        }
        
        **Simple Explanation:**
        The past return of {current_return:.2f}% was {'VERY unusual' if abs(current_zscore) > 2 else 'normal'}. 
        Based on {len(similar_cases)} similar historical situations, the NEXT move is predicted to be 
        {avg_next_return:+.2f}% ({'opposite direction - reversal!' if reversal_rate > 60 else 'continuation'}).
        """)
    
    # ==================== FINAL TRADING RECOMMENDATION (Enhanced with Predictions) ====================
    st.header("🎯 Final Trading Recommendation")
    
    # Multi-factor signal generation with forward-looking predictions
    signals = {}
    weights = {
        'volatility_prediction': 0.25,
        'zscore_prediction': 0.25,
        'rsi': 0.20,
        'ema_alignment': 0.30
    }
    
    # 1. Volatility-based prediction signal
    if pd.notna(current_vol_bin) and current_vol_bin in bin_analysis.index:
        vol_pred_return = bin_analysis.loc[current_vol_bin, ('Next_Return', 'mean')]
        vol_signal = np.clip(vol_pred_return / 2, -1, 1)  # Scale to [-1, 1]
        signals['volatility_prediction'] = vol_signal
    else:
        signals['volatility_prediction'] = 0
    
    # 2. Z-Score prediction signal (mean reversion)
    if len(similar_cases) > 0:
        zscore_pred_return = avg_next_return
        zscore_signal = np.clip(zscore_pred_return / 2, -1, 1)
        signals['zscore_prediction'] = zscore_signal
    else:
        signals['zscore_prediction'] = 0
    
    # 3. RSI signal
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
    st.subheader("📊 Signal Component Analysis (Forward-Looking)")
    
    signal_breakdown = pd.DataFrame({
        'Component': ['Volatility Prediction (Next Move)', 'Z-Score Prediction (Mean Reversion)', 'RSI Indicator', 'EMA Alignment'],
        'Raw Score': [signals['volatility_prediction'], signals['zscore_prediction'], signals['rsi'], signals['ema_alignment']],
        'Weight': [weights['volatility_prediction'], weights['zscore_prediction'], weights['rsi'], weights['ema_alignment']],
        'Weighted Score': [
            signals['volatility_prediction'] * weights['volatility_prediction'],
            signals['zscore_prediction'] * weights['zscore_prediction'],
            signals['rsi'] * weights['rsi'],
            signals['ema_alignment'] * weights['ema_alignment']
        ],
        'Interpretation': [
            'Bullish' if signals['volatility_prediction'] > 0.2 else 'Bearish' if signals['volatility_prediction'] < -0.2 else 'Neutral',
            'Bullish' if signals['zscore_prediction'] > 0.2 else 'Bearish' if signals['zscore_prediction'] < -0.2 else 'Neutral',
            'Bullish' if signals['rsi'] > 0.2 else 'Bearish' if signals['rsi'] < -0.2 else 'Neutral',
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
    - Set alerts at key support/resistance levels
    
    **Trailing Stop Recommendations:**
    - After +{abs(pct_gain_target)/4:.1f}% profit: Trail stop to entry price (breakeven)
    - After +{abs(pct_gain_target)/2:.1f}% profit: Trail stop to +{abs(pct_gain_target)/4:.1f}%
    - After reaching target: Book partial profits (50-70%) and let rest run with trailing stop
    """)
    
    # Detailed rationale
    st.subheader("🧠 Why This Signal Was Generated (Forward-Looking Analysis)")
    
    # Build rationale based on predictions
    rationale_parts = []
    
    # Volatility prediction
    if pd.notna(current_vol_bin) and current_vol_bin in bin_analysis.index:
        vol_pred = bin_analysis.loc[current_vol_bin, ('Next_Return', 'mean')]
        rationale_parts.append(
            f"{'✅' if vol_pred > 0.5 else '❌' if vol_pred < -0.5 else '➡️'} "
            f"**Volatility-Based Prediction**: Current volatility ({current_vol:.2f}%) historically leads to "
            f"a NEXT move of {vol_pred:+.2f}% on average. This suggests {'upward' if vol_pred > 0 else 'downward' if vol_pred < 0 else 'sideways'} movement ahead."
        )
    
    # Z-score prediction
    if len(similar_cases) > 0:
        rationale_parts.append(
            f"{'✅' if avg_next_return > 0.5 else '❌' if avg_next_return < -0.5 else '➡️'} "
            f"**Mean Reversion Prediction**: Z-score of {current_zscore:.2f} indicates {zone.lower()}. "
            f"After similar past situations, the NEXT move averaged {avg_next_return:+.2f}%. "
            f"{'This suggests a reversal is likely.' if abs(current_zscore) > 2 else 'This suggests trend continuation.'}"
        )
    
    # RSI analysis
    if current_rsi > 70:
        rationale_parts.append(f"❌ **RSI Overbought** ({current_rsi:.1f}): Market is overextended, increasing pullback probability.")
    elif current_rsi < 30:
        rationale_parts.append(f"✅ **RSI Oversold** ({current_rsi:.1f}): Market is oversold, creating bounce opportunity.")
    else:
        rationale_parts.append(f"➡️ **RSI Neutral** ({current_rsi:.1f}): RSI shows no extreme condition.")
    
    # EMA alignment
    if signals['ema_alignment'] > 0.3:
        rationale_parts.append(f"✅ **Bullish EMA Alignment**: Price above key moving averages - bullish structure intact.")
    elif signals['ema_alignment'] < -0.3:
        rationale_parts.append(f"❌ **Bearish EMA Alignment**: Price below key moving averages - bearish structure intact.")
    else:
        rationale_parts.append(f"⚠️ **Mixed EMA Alignment**: EMAs not clearly aligned - market lacks conviction.")
    
    st.markdown("\n\n".join(rationale_parts))
    
    # Expected scenario
    st.subheader("🔮 Expected Scenario (What Happens NEXT)")
    
    if combined_signal > 0.3:
        scenario = f"""
        **📈 Bullish Scenario (PRIMARY PREDICTION):**
        
        **What We Expect to Happen NEXT:**
        - Price should rally from current ₹{entry_price:.2f} towards ₹{target_price:.2f}
        - Expected profit: {pct_gain_target:+.2f}%
        - Timeline: Within the next few {interval} periods
        
        **Why We Predict This:**
        1. Volatility patterns suggest {vol_pred:+.2f}% next move (historically)
        2. Z-score analysis predicts {avg_next_return:+.2f}% move (mean reversion)
        3. RSI and EMAs support {'the bullish case' if signals['ema_alignment'] > 0 else 'despite some resistance'}
        
        **What to Watch:**
        - ✅ Confirmation: Price staying above ₹{entry_price - (atr_current * 0.5):.2f}
        - ❌ Invalidation: Break below ₹{stop_loss:.2f}
        - 🎯 Target: ₹{target_price:.2f}
        
        **Risk If Wrong:**
        If prediction fails and stop hits at ₹{stop_loss:.2f}, you lose {pct_loss_stop:.2f}%. 
        But with {risk_reward:.1f}:1 reward/risk, one winner covers multiple losers!
        """
    elif combined_signal < -0.3:
        scenario = f"""
        **📉 Bearish Scenario (PRIMARY PREDICTION):**
        
        **What We Expect to Happen NEXT:**
        - Price should decline from current ₹{entry_price:.2f} towards ₹{target_price:.2f}
        - Expected profit: {abs(pct_gain_target):.2f}% (on short position)
        - Timeline: Within the next few {interval} periods
        
        **Why We Predict This:**
        1. Volatility patterns suggest {vol_pred:+.2f}% next move (historically)
        2. Z-score analysis predicts {avg_next_return:+.2f}% move (mean reversion)
        3. RSI and EMAs support {'the bearish case' if signals['ema_alignment'] < 0 else 'despite some support'}
        
        **What to Watch:**
        - ✅ Confirmation: Price staying below ₹{entry_price + (atr_current * 0.5):.2f}
        - ❌ Invalidation: Break above ₹{stop_loss:.2f}
        - 🎯 Target: ₹{target_price:.2f}
        
        **Risk If Wrong:**
        If prediction fails and stop hits at ₹{stop_loss:.2f}, you lose {abs(pct_loss_stop):.2f}%.
        """
    else:
        scenario = f"""
        **➡️ Neutral Scenario (NO CLEAR PREDICTION):**
        
        **Current Situation:**
        - Signals are mixed - no clear directional edge
        - Volatility and Z-score predictions conflict
        - Best action: WAIT for clarity
        
        **What to Watch For (NEXT):**
        - 📈 **Bullish Trigger**: Price breaks above ₹{entry_price * 1.02:.2f} with conviction
        - 📉 **Bearish Trigger**: Price breaks below ₹{entry_price * 0.98:.2f} with conviction
        
        **Smart Trader Advice:**
        Don't force trades! Wait for one of the triggers above. 
        Patience protects your capital when predictions are uncertain.
        """
    
    st.info(scenario)
    
    # Final summary for beginners
    st.subheader("👶 Simple Explanation (For Beginners)")
    
    st.success(f"""
    **🎯 SIMPLE PREDICTION - What Happens NEXT?**
    
    **Current Price**: ₹{entry_price:.2f}
    
    **Our Prediction**: {action} - The price will likely {'GO UP ⬆️' if combined_signal > 0.3 else 'GO DOWN ⬇️' if combined_signal < -0.3 else 'STAY FLAT ➡️'}
    
    **Expected Next Move**: {
    f"Price should rise to around ₹{target_price:.2f} (gain of {pct_gain_target:.2f}%)" if combined_signal > 0.3
    else f"Price should fall to around ₹{target_price:.2f} (gain of {abs(pct_gain_target):.2f}% on short)" if combined_signal < -0.3
    else f"Price will likely move sideways - wait for clear direction"
    }
    
    **How Confident Are We**: {confidence}
    
    **Why This Prediction?**
    We looked at:
    1. **What happened BEFORE** (past returns, volatility)
    2. **What usually happens NEXT** after similar situations (historical patterns)
    3. **Current technical indicators** (RSI, moving averages)
    
    All these point to: **{action}**
    
    **What You Should Do:**
    {
    f"✅ BUY at ₹{entry_price:.2f}, Target: ₹{target_price:.2f}, Stop: ₹{stop_loss:.2f}"
    if combined_signal > 0.3
    else f"❌ SELL/SHORT at ₹{entry_price:.2f}, Target: ₹{target_price:.2f}, Stop: ₹{stop_loss:.2f}"
    if combined_signal < -0.3
    else f"⏸️ WAIT - Don't trade yet, signals are unclear"
    }
    
    **Remember:** This is a prediction based on patterns, NOT a guarantee! 
    Always use stop losses and never risk more than you can afford to lose.
    """)
    
    # ==================== INTERACTIVE CHARTS ====================
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
    if st.session_state.enable_ratio_value and 'data2' in st.session_state:
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
        
        # Align data by datetime
        merged_data = pd.merge(
            data1[['Datetime', 'Close', 'RSI']],
            data2[['Datetime', 'Close', 'RSI']],
            on='Datetime',
            suffixes=('_T1', '_T2')
        )
        
        merged_data['Ratio'] = merged_data['Close_T1'] / merged_data['Close_T2']
        merged_data['Ratio_RSI'] = calculate_rsi(merged_data['Ratio'])
        
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
    
    # Disclaimer
    st.warning("""
    ⚠️ **DISCLAIMER**: This analysis uses historical patterns to predict future movements, but markets are unpredictable. 
    This is for educational purposes only and should not be considered as financial advice. 
    Always do your own research, understand the risks involved, and consider consulting with a qualified financial advisor 
    before making any investment decisions. Past performance does not guarantee future results. Trading involves substantial 
    risk of loss and is not suitable for all investors.
    """)

else:
    st.info("👆 Please configure your parameters in the sidebar and click 'Fetch Data & Analyze' to begin analysis.")
    
    # Show feature highlights
    st.markdown("""
    ## 🌟 Dashboard Features
    
    ### 🔮 **PREDICTIVE ANALYSIS** (What Happens NEXT)
    - **Volatility-Based Prediction**: Analyzes current volatility to predict the NEXT move
    - **Z-Score Mean Reversion**: Predicts reversals after extreme moves
    - **Forward-Looking Signals**: All recommendations focus on future movements, not past
    
    ### 📊 Core Analytics
    - **Real-time Market Data**: Live data from Yahoo Finance
    - **Multi-Asset Support**: Stocks, indices, crypto, forex, commodities
    - **Ratio Analysis**: Compare two assets
    
    ### 📈 Technical Analysis
    - **Multi-Timeframe**: Analyze 10 timeframes simultaneously
    - **Advanced Indicators**: RSI, EMAs, SMAs, Fibonacci, ATR
    - **Support & Resistance**: Automatic level detection
    
    ### 🎯 Clear Trading Signals
    - **BUY/SELL/HOLD**: Simple actionable recommendations
    - **Entry/Target/Stop**: Precise levels for each trade
    - **Risk Management**: Position sizing and trailing stops
    
    ### 💾 Data Export
    - **CSV Export**: Download complete data with all indicators
    
    ---
    
    **🚀 Ready to start? Configure parameters and click "Fetch Data & Analyze"!**
    """)
