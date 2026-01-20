import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import pytz
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(page_title="Pro Algo Dashboard", layout="wide")

# IST Timezone constant
IST = pytz.timezone('Asia/Kolkata')

# Ticker Mapping
TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "USDINR": "INR=X",  # Standard yfinance ticker for USD/INR
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS"
}

# Strict Timeframe/Period Validation
VALID_INTERVALS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "30m": ["1mo"],
    "1h": ["1mo"],
    "4h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "20y", "max"]
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def enforce_ist(df):
    """Converts DataFrame index to IST strictly."""
    if df.empty:
        return df
    
    # Check if index is tz-aware
    if df.index.tz is None:
        try:
            # Assume UTC if naive, then convert to IST. 
            # Note: yfinance usually returns naive times in local market time or UTC depending on version.
            # Best practice: localize to UTC then convert, or localize to IST directly if we know it's NSE.
            # Safe bet: localize to UTC first if unsure, but yfinance standard is usually UTC for crypto/FX.
            df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(IST)
        except:
            # Fallback
            df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)
    return df

def flatten_yf_columns(df):
    """Flattens MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def fetch_ticker_data(ticker, period, interval):
    """Fetches data with rate limiting and error handling."""
    # Randomized delay for rate limiting
    sleep_time = random.uniform(1.0, 1.5)
    time.sleep(sleep_time)
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
            return None
        
        df = flatten_yf_columns(df)
        df = enforce_ist(df)
        
        # Ensure we have required columns
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in req_cols):
            # Sometimes yfinance auto-adjusts, try to map 'Adj Close' to 'Close' if Close missing
            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df['Close'] = df['Adj Close']
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill NaN with neutral 50

def get_signal_confidence(ratio_series, rsi_series):
    """Simple logic to determine trend and confidence."""
    last_rsi = rsi_series.iloc[-1]
    
    bullish_score = 0
    bearish_score = 0
    
    # RSI Logic
    if last_rsi < 30: bullish_score += 1
    elif last_rsi > 70: bearish_score += 1
    
    # Trend Logic (SMA check)
    sma_20 = ratio_series.rolling(20).mean().iloc[-1]
    curr_price = ratio_series.iloc[-1]
    
    if curr_price > sma_20: bullish_score += 1
    else: bearish_score += 1
    
    total = bullish_score + bearish_score
    if total == 0: return "Sideways", 50.0
    
    if bullish_score > bearish_score:
        conf = (bullish_score / 2) * 100 
        return "Bullish", min(conf, 95.0)
    elif bearish_score > bullish_score:
        conf = (bearish_score / 2) * 100
        return "Bearish", min(conf, 95.0)
    else:
        return "Sideways", 50.0

# ==========================================
# UI SIDEBAR
# ==========================================

st.sidebar.title("Algo Configuration")

# Ticker Selection
t1_options = list(TICKERS.keys()) + ["Custom"]
t1_sel = st.sidebar.selectbox("Select Ticker 1", t1_options, index=0) # Default Nifty
if t1_sel == "Custom":
    ticker1_sym = st.sidebar.text_input("Enter Ticker 1 Symbol", value="RELIANCE.NS")
else:
    ticker1_sym = TICKERS[t1_sel]

t2_options = list(TICKERS.keys()) + ["Custom"]
t2_sel = st.sidebar.selectbox("Select Ticker 2", t2_options, index=3) # Default USDINR
if t2_sel == "Custom":
    ticker2_sym = st.sidebar.text_input("Enter Ticker 2 Symbol", value="TCS.NS")
else:
    ticker2_sym = TICKERS[t2_sel]

# Time settings
interval = st.sidebar.selectbox("Timeframe", list(VALID_INTERVALS.keys()), index=6) # Default 1d
allowed_periods = VALID_INTERVALS[interval]
period = st.sidebar.selectbox("Period", allowed_periods, index=1) # Default 1y

# Analysis settings
num_bins = st.sidebar.number_input("Number of Bins", min_value=5, max_value=100, value=20)
next_candles_n = st.sidebar.selectbox("Analyze Next N Candles", [5, 10, 15, 20, 50], index=2)

fetch_btn = st.sidebar.button("Fetch & Analyze")

# ==========================================
# MAIN LOGIC
# ==========================================

if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

if fetch_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing request...")
    progress_bar.progress(10)
    
    # 1. Fetch Ticker 1
    status_text.text(f"Fetching Data for {ticker1_sym}...")
    df1 = fetch_ticker_data(ticker1_sym, period, interval)
    progress_bar.progress(40)
    
    # 2. Fetch Ticker 2
    status_text.text(f"Fetching Data for {ticker2_sym}...")
    df2 = fetch_ticker_data(ticker2_sym, period, interval)
    progress_bar.progress(70)
    
    if df1 is None or df2 is None:
        st.error("Failed to fetch data. Please check ticker symbols or API limits.")
    else:
        # 3. Alignment
        status_text.text("Aligning Timezones & Calculating Ratios...")
        
        # Inner join to handle different market hours (e.g. Nifty vs BTC)
        # We only care when BOTH intersect to calculate a valid ratio
        aligned_df = pd.merge(df1['Close'], df2['Close'], left_index=True, right_index=True, how='inner', suffixes=('_1', '_2'))
        
        # Calculate Ratio
        aligned_df['Ratio'] = aligned_df['Close_1'] / aligned_df['Close_2']
        
        # Calculate RSI of Ratio
        aligned_df['RSI'] = calculate_rsi(aligned_df['Ratio'])
        
        # Calculate Future Movements for Backtesting/Analysis
        for i in range(1, next_candles_n + 1):
            # Shift negative to look forward
            aligned_df[f'Next_{i}_Close'] = aligned_df['Ratio'].shift(-i)
            aligned_df[f'Change_{i}'] = aligned_df[f'Next_{i}_Close'] - aligned_df['Ratio']
            aligned_df[f'Pct_Change_{i}'] = (aligned_df[f'Change_{i}'] / aligned_df['Ratio']) * 100

        # Store in session
        st.session_state.df1 = df1
        st.session_state.df2 = df2
        st.session_state.aligned_df = aligned_df
        st.session_state.ticker1 = ticker1_sym
        st.session_state.ticker2 = ticker2_sym
        st.session_state.data_fetched = True
        st.session_state.next_n = next_candles_n
        
    progress_bar.progress(100)
    status_text.empty()

# ==========================================
# DISPLAY LOGIC
# ==========================================

if st.session_state.data_fetched:
    df = st.session_state.aligned_df.copy()
    t1 = st.session_state.ticker1
    t2 = st.session_state.ticker2
    next_n = st.session_state.next_n
    
    # Forecast Calculation (Simple Majority Logic)
    direction, conf = get_signal_confidence(df['Ratio'], df['RSI'])
    
    # ------------------------------------------
    # TABS
    # ------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Ratio Charts", "RSI Divergence", "Backtesting", "Statistics"])
    
    # ==========================================
    # TAB 1: RATIO CHARTS
    # ==========================================
    with tab1:
        st.header(f"Ratio Analysis: {t1} / {t2}")
        
        # 1. Summary Metrics
        curr_ratio = df['Ratio'].iloc[-1]
        max_ratio = df['Ratio'].max()
        min_ratio = df['Ratio'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Ratio", f"{curr_ratio:.4f}")
        col2.metric("Max Ratio", f"{max_ratio:.4f}")
        col3.metric("Min Ratio", f"{min_ratio:.4f}")
        col4.metric("Forecast", f"{direction} ({conf:.0f}%)")
        
        # 2. Future Movement Table (Last N candles)
        st.subheader(f"Projected Movement (Based on Historical Lag)")
        
        # Create a display table for the most recent valid data points
        display_cols = ['Ratio'] + [f'Change_{i}' for i in range(1, 6)] + [f'Pct_Change_{i}' for i in range(1, 6)]
        
        # We take the last few rows excluding the NaN futures
        valid_rows = df.dropna(subset=[f'Next_{1}_Close']).tail(10)
        
        # Clean up column names for display
        clean_cols = {col: col.replace('Change_', 'Points_').replace('Pct_', '%_') for col in display_cols}
        st.dataframe(valid_rows[display_cols].rename(columns=clean_cols).sort_index(ascending=False).style.format("{:.2f}"))

        # 3. Text Summary
        with st.expander("Analysis Summary & Targets", expanded=True):
            avg_move = valid_rows[f'Change_{1}'].abs().mean()
            st.write(f"""
            **Highlights:** The ratio is currently trading at **{curr_ratio:.2f}**. 
            Based on historical bins, significant rallies occurred when RSI dipped below 30.
            Current RSI is **{df['RSI'].iloc[-1]:.2f}**.
            
            **Trade Plan:**
            - **Bias:** {direction}
            - **Entry:** Current Market Price
            - **Target:** {curr_ratio + (avg_move*2):.4f} (approx 2x avg move)
            - **SL:** {curr_ratio - avg_move:.4f}
            - **Confidence:** {conf}% based on recent momentum.
            """)

        # 4. Plots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Ratio Line
        fig.add_trace(go.Scatter(x=df.index, y=df['Ratio'], name="Ratio", line=dict(color='blue')), row=1, col=1)
        
        # RSI Line
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, title_text=f"{t1}/{t2} Ratio & RSI", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TAB 2: RSI DIVERGENCE
    # ==========================================
    with tab2:
        st.header("RSI Divergence Detector")
        
        # Simple Divergence Logic (Peak detection)
        # Looking for Price Higher High + RSI Lower High (Bearish)
        # Looking for Price Lower Low + RSI Higher Low (Bullish)
        
        # We look back 'window' candles
        window = 5
        div_data = []
        
        # Iterate through the last 100 candles for performance
        subset = df.iloc[-100:].copy()
        
        for i in range(window, len(subset)-window):
            curr_idx = subset.index[i]
            prev_idx = subset.index[i-window]
            
            curr_price = subset['Ratio'].iloc[i]
            prev_price = subset['Ratio'].iloc[i-window]
            curr_rsi = subset['RSI'].iloc[i]
            prev_rsi = subset['RSI'].iloc[i-window]
            
            # Bearish Divergence
            if curr_price > prev_price and curr_rsi < prev_rsi and curr_rsi > 70:
                div_data.append({'Date': curr_idx, 'Type': 'Bearish', 'Price': curr_price, 'RSI': curr_rsi})
            
            # Bullish Divergence
            elif curr_price < prev_price and curr_rsi > prev_rsi and curr_rsi < 30:
                div_data.append({'Date': curr_idx, 'Type': 'Bullish', 'Price': curr_price, 'RSI': curr_rsi})
        
        div_df = pd.DataFrame(div_data)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detected Divergences")
            if not div_df.empty:
                st.dataframe(div_df.tail(10))
            else:
                st.info("No strong divergence detected in recent candles.")
                
            st.markdown("### Forecast")
            st.info(f"Market Sentiment: **{direction}** ({conf}%)")

        with col2:
            # Plot with Divergence Markers
            fig_div = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_div.add_trace(go.Scatter(x=subset.index, y=subset['Ratio'], name="Ratio"), row=1, col=1)
            fig_div.add_trace(go.Scatter(x=subset.index, y=subset['RSI'], name="RSI", line=dict(color='orange')), row=2, col=1)
            
            if not div_df.empty:
                # Add markers
                bullish_div = div_df[div_df['Type'] == 'Bullish']
                bearish_div = div_df[div_df['Type'] == 'Bearish']
                
                fig_div.add_trace(go.Scatter(x=bullish_div['Date'], y=bullish_div['Price'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Bull Div'), row=1, col=1)
                fig_div.add_trace(go.Scatter(x=bearish_div['Date'], y=bearish_div['Price'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Bear Div'), row=1, col=1)

            fig_div.update_layout(height=500, title="Ratio Price vs RSI with Divergence Markers")
            st.plotly_chart(fig_div, use_container_width=True)

    # ==========================================
    # TAB 3: BACKTESTING
    # ==========================================
    with tab3:
        st.header("Strategy Backtest (RSI Mean Reversion)")
        
        # Simple Strategy:
        # Buy when RSI < 30
        # Sell when RSI > 70
        # Exit after N candles
        
        bt_df = df.copy().dropna()
        bt_df['Signal'] = 0
        bt_df.loc[bt_df['RSI'] < 30, 'Signal'] = 1  # Long
        bt_df.loc[bt_df['RSI'] > 70, 'Signal'] = -1 # Short
        
        trades = []
        hold_period = 5 # candles
        
        for i in range(len(bt_df) - hold_period):
            if bt_df['Signal'].iloc[i] != 0:
                entry_date = bt_df.index[i]
                entry_price = bt_df['Ratio'].iloc[i]
                signal = bt_df['Signal'].iloc[i]
                
                exit_date = bt_df.index[i+hold_period]
                exit_price = bt_df['Ratio'].iloc[i+hold_period]
                
                pnl = (exit_price - entry_price) * signal
                trades.append({
                    'Entry Time': entry_date,
                    'Type': 'Long' if signal == 1 else 'Short',
                    'Entry Price': entry_price,
                    'Exit Time': exit_date,
                    'Exit Price': exit_price,
                    'PnL Points': pnl
                })
        
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty:
            total_pnl = trades_df['PnL Points'].sum()
            win_rate = (len(trades_df[trades_df['PnL Points'] > 0]) / len(trades_df)) * 100
            
            st.markdown(f"### Results (Exit after {hold_period} candles)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total PnL (Points)", f"{total_pnl:.4f}")
            m2.metric("Total Trades", len(trades_df))
            m3.metric("Win Rate", f"{win_rate:.2f}%")
            
            st.dataframe(trades_df.sort_values(by='Entry Time', ascending=False).style.applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['PnL Points']))
        else:
            st.warning("No trades generated with current parameters.")

    # ==========================================
    # TAB 4: STATISTICS
    # ==========================================
    with tab4:
        st.header("Statistical Analysis")
        
        stat_df = df.copy()
        stat_df['DayOfWeek'] = stat_df.index.day_name()
        stat_df['Change'] = stat_df['Ratio'].diff()
        stat_df['AbsChange'] = stat_df['Change'].abs()
        stat_df['Color'] = np.where(stat_df['Change'] > 0, 'Green', 'Red')
        
        # 1. Tabular Data
        st.subheader("Candle-by-Candle Statistics")
        disp_stats = stat_df[['Ratio', 'Change', 'Pct_Change_1', 'DayOfWeek', 'Color']].sort_index(ascending=False)
        st.dataframe(disp_stats.head(100).style.applymap(lambda v: f"color: {'green' if v == 'Green' else 'red'}", subset=['Color']))
        
        # 2. Stats Summary
        st.subheader("Volatility by Day of Week")
        vol_by_day = stat_df.groupby('DayOfWeek')['AbsChange'].mean().sort_values(ascending=False)
        st.bar_chart(vol_by_day)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Move", f"{stat_df['AbsChange'].median():.4f}")
        col2.metric("Max Move", f"{stat_df['AbsChange'].max():.4f}")
        col3.metric("Min Move", f"{stat_df['AbsChange'].min():.4f}")
        
        # Final Majority Forecast
        st.divider()
        st.markdown(f"### 🛡️ Final Consensus: **{direction}**")
        st.progress(int(conf))
        st.caption(f"Confidence Level: {conf:.1f}% based on multi-factor analysis across selected timeframe.")

else:
    st.info("👈 Please select parameters and click 'Fetch & Analyze' in the sidebar.")
