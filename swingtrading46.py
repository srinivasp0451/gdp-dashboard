import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import pytz

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Z-Score Mean Reversion Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    .summary-box {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
    }
</style>
""", unsafe_allow_html=True)

COMMON_TICKERS = {
    "Indices": {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "BANK NIFTY",
        "^BSESN": "SENSEX",
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ"
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana"
    },
    "Commodities": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "CL=F": "Crude Oil"
    },
    "Forex": {
        "INR=X": "USD/INR",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD"
    },
    "Indian Stocks": {
        "RELIANCE.NS": "Reliance",
        "TCS.NS": "TCS",
        "HDFCBANK.NS": "HDFC Bank",
        "INFY.NS": "Infosys"
    }
}

TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']

# -----------------------------------------------------------------------------
# Helper Functions & Logic
# -----------------------------------------------------------------------------

def convert_to_ist(df):
    """Converts DataFrame index to IST."""
    if df.empty:
        return df
    
    # Check if index is tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    ist = pytz.timezone('Asia/Kolkata')
    df.index = df.index.tz_convert(ist)
    return df

def fetch_data(ticker, period, interval, delay=1.5):
    """Robust data fetching with rate limiting and error handling."""
    try:
        time.sleep(delay) # Rate limiting
        data = yf.download(
            tickers=ticker, 
            period=period, 
            interval=interval, 
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            return None, "No data returned from API."
            
        # Clean multi-level columns if present (common in new yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = convert_to_ist(data)
        return data, None
    except Exception as e:
        return None, str(e)

def calculate_technical_indicators(df, z_window=20):
    """Calculates Z-Score, RSI, EMAs, ADX, and Fibonacci Levels."""
    if df is None or len(df) < z_window:
        return df

    # 1. Z-Score (Mean Reversion)
    df['Rolling_Mean'] = df['Close'].rolling(window=z_window).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=z_window).std()
    df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']

    # 2. Trend Filters (EMAs)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # 3. RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. ADX (Trend Strength - Simplified)
    # True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Directional Movement
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=14).mean()

    # 5. Fibonacci Retracement (Based on last 100 bars)
    recent_high = df['High'].tail(100).max()
    recent_low = df['Low'].tail(100).min()
    diff = recent_high - recent_low
    df['Fib_0.236'] = recent_high - 0.236 * diff
    df['Fib_0.382'] = recent_high - 0.382 * diff
    df['Fib_0.5'] = recent_high - 0.5 * diff
    df['Fib_0.618'] = recent_high - 0.618 * diff

    return df

def analyze_market_structure(df):
    """Determines if market is Trending (Bull/Bear) or Rangebound."""
    if df is None or len(df) < 50:
        return "Insufficient Data", 0

    last_row = df.iloc[-1]
    adx = last_row['ADX']
    price = last_row['Close']
    ema_50 = last_row['EMA_50']
    ema_200 = last_row['EMA_200']

    structure = "Rangebound"
    strength = "Weak"

    if adx > 25:
        strength = "Strong"
        if price > ema_50 > ema_200:
            structure = "Trending Uptrend"
        elif price < ema_50 < ema_200:
            structure = "Trending Downtrend"
        else:
            structure = "Choppy Trend"
    else:
        strength = "Weak"
        structure = "Rangebound / Sideways"

    return structure, strength

def generate_signals(df, z_threshold=2.0):
    """Generates signals based on confluence."""
    last_row = df.iloc[-1]
    z_score = last_row['Z_Score']
    rsi = last_row['RSI']
    close = last_row['Close']
    
    signal = "NEUTRAL"
    reason = []

    # Z-Score Logic
    if z_score < -z_threshold:
        # Potential Buy
        if rsi < 30:
            signal = "STRONG BUY"
            reason.append(f"Z-Score Oversold ({z_score:.2f})")
            reason.append(f"RSI Oversold ({rsi:.2f})")
        else:
            signal = "BUY"
            reason.append(f"Z-Score Oversold ({z_score:.2f})")
    
    elif z_score > z_threshold:
        # Potential Sell
        if rsi > 70:
            signal = "STRONG SELL"
            reason.append(f"Z-Score Overbought ({z_score:.2f})")
            reason.append(f"RSI Overbought ({rsi:.2f})")
        else:
            signal = "SELL"
            reason.append(f"Z-Score Overbought ({z_score:.2f})")

    # Fibonacci Confluence Check
    fib_levels = [last_row['Fib_0.5'], last_row['Fib_0.618']]
    for level in fib_levels:
        if abs(close - level) / close < 0.005: # Within 0.5%
            reason.append("Price at Key Golden Ratio/Fib Level")
            if signal == "NEUTRAL":
                # If neutral but at Fib support?
                pass 
            elif "BUY" in signal:
                signal = "GOLDEN " + signal # Upgrade signal

    return signal, reason

def get_detailed_recommendation(df, ticker, interval):
    """Generates the 200-word verbose summary."""
    structure, strength = analyze_market_structure(df)
    signal, reasons = generate_signals(df)
    
    last = df.iloc[-1]
    curr_price = last['Close']
    mean_price = last['Rolling_Mean']
    std_dev = last['Rolling_Std']
    
    # Calculate Targets & SL
    atr = last['ATR']
    
    if "BUY" in signal:
        sl = curr_price - (2 * atr)
        target1 = mean_price
        target2 = mean_price + (2 * std_dev)
    elif "SELL" in signal:
        sl = curr_price + (2 * atr)
        target1 = mean_price
        target2 = mean_price - (2 * std_dev)
    else:
        sl = 0
        target1 = 0
        target2 = 0

    summary = f"""
    ### **Strategy Analysis Report: {ticker} ({interval})**
    
    **Market Structure Analysis:**
    The current market regime for {ticker} is identified as **{structure}** with **{strength}** momentum (ADX: {last['ADX']:.2f}). 
    Prices are currently trading at {curr_price:.2f}.
    
    **Technical Pattern Recognition:**
    Our Z-Score Mean Reversion algorithm determines a standardized score of **{last['Z_Score']:.2f}**. 
    {'This indicates a statistical anomaly, suggesting prices are overextended.' if abs(last['Z_Score']) > 2 else 'Prices are moving within standard statistical deviations.'}
    
    Confluence checks reveal: {', '.join(reasons) if reasons else 'No significant pattern confluence (Fibonacci/RSI) detected at this moment.'}
    
    **Trade Recommendation:**
    Based on the synthesis of Volatility, Momentum (RSI: {last['RSI']:.2f}), and Trend status, the system generates a **{signal}** signal.
    
    """
    
    if signal != "NEUTRAL":
        summary += f"""
    **Action Plan:**
    * **Entry Zone:** {curr_price:.2f} (Current Market Price)
    * **Stop Loss (Risk Management):** {sl:.2f} (Calculated using 2x ATR Volatility)
    * **Target 1 (Mean Reversion):** {target1:.2f} (Reverting to Rolling Mean)
    * **Target 2 (Extension):** {target2:.2f} (Statistical Extreme)
    
    **Rationale:**
    The strategy relies on the probability of price returning to its statistical average ({mean_price:.2f}). 
    {'The strong trend strength suggests caution against counter-trend trades.' if strength == 'Strong' and 'Trending' in structure else 'The rangebound nature supports mean reversion setups.'}
        """
    else:
        summary += f"""
    **Action Plan:**
    * **Wait & Watch.** The market is currently in equilibrium or conflicting states.
    * We are waiting for price to hit ±2.0 Standard Deviations or align with a key Fibonacci Level before engaging.
    * Current Support: {last['Fib_0.618']:.2f} | Current Resistance: {last['High'] + atr:.2f}
        """
        
    return summary, signal, sl, target1

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.title("🛠️ Algo Config")
    
    # Asset Selection
    asset_category = st.sidebar.selectbox("Asset Class", list(COMMON_TICKERS.keys()))
    ticker_display = st.sidebar.selectbox("Select Asset", list(COMMON_TICKERS[asset_category].values()))
    
    # Reverse lookup for ticker symbol
    ticker_symbol = [k for k, v in COMMON_TICKERS[asset_category].items() if v == ticker_display][0]
    
    # Custom Ticker Option
    use_custom = st.sidebar.checkbox("Use Custom Ticker")
    if use_custom:
        ticker_symbol = st.sidebar.text_input("Enter Ticker (e.g., AAPL, TATAMOTORS.NS)", value="AAPL")

    # Time Parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_interval = st.selectbox("Timeframe", TIMEFRAMES, index=4) # Default 30m
    with col2:
        selected_period = st.selectbox("Lookback", PERIODS, index=3) # Default 3mo

    # Strategy Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Params")
    z_window = st.sidebar.slider("Z-Score Window", 10, 100, 20)
    z_thresh = st.sidebar.slider("Entry Threshold (Std Dev)", 1.0, 4.0, 2.0, 0.1)
    
    # Fetch Button
    fetch_btn = st.sidebar.button("🚀 Analyze Market", type="primary")

    # Session State Management for Data persistence
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""

    # Header
    st.title(f"📊 Mean Reversion Strategy: {ticker_display}")
    st.markdown(f"Analysis for **{ticker_symbol}** | Timeframe: **{selected_interval}**")

    # Fetch Logic
    if fetch_btn:
        with st.spinner(f"Fetching {ticker_symbol} data & Calculating Matrix..."):
            df, error = fetch_data(ticker_symbol, selected_period, selected_interval)
            
            if error:
                st.error(f"Error fetching data: {error}")
            else:
                # Process Data
                df = calculate_technical_indicators(df, z_window)
                st.session_state.data = df
                st.session_state.ticker = ticker_symbol
                st.rerun() # Refresh to show data

    # Display Logic (runs if data exists in session state)
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # 1. Summary Section
        summary, signal, sl, target = get_detailed_recommendation(df, st.session_state.ticker, selected_interval)
        
        # Color coding for signal
        color_map = {"STRONG BUY": "green", "BUY": "lightgreen", "NEUTRAL": "gray", "SELL": "orange", "STRONG SELL": "red"}
        base_signal = next((k for k in color_map if k in signal), "NEUTRAL")
        signal_color = color_map.get(base_signal, "blue")
        
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.markdown(f"<h2 style='color:{signal_color}; text-align:center; border: 2px solid {signal_color}; border-radius:10px; padding:10px;'>{signal}</h2>", unsafe_allow_html=True)
            if "BUY" in signal or "SELL" in signal:
                st.metric("Entry Price", f"{df.iloc[-1]['Close']:.2f}")
                st.metric("Target", f"{target:.2f}")
                st.metric("Stop Loss", f"{sl:.2f}")
        
        with col_res2:
            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

        # 2. Advanced Charts
        st.markdown("### 📉 Technical Chart")
        
        # Main Candlestick Chart
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"Price Action & Bollinger Bands", "Z-Score", "RSI & Trend"))

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # Bollinger Bands (derived from Z-Score logic)
        fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'], line=dict(color='orange', width=1), name='Mean'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'] + (z_thresh * df['Rolling_Std']), line=dict(color='gray', width=1, dash='dot'), name=f'+{z_thresh} Std'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'] - (z_thresh * df['Rolling_Std']), line=dict(color='gray', width=1, dash='dot'), name=f'-{z_thresh} Std'), row=1, col=1)

        # Fibonacci Lines (Static based on current view)
        last_row = df.iloc[-1]
        fig.add_hline(y=last_row['Fib_0.618'], line_dash="dash", line_color="gold", annotation_text="Fib 0.618", row=1, col=1)

        # Z-Score Subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], line=dict(color='blue', width=2), name='Z-Score'), row=2, col=1)
        fig.add_hline(y=z_thresh, line_color="red", line_dash="dash", row=2, col=1)
        fig.add_hline(y=-z_thresh, line_color="green", line_dash="dash", row=2, col=1)
        fig.add_hline(y=0, line_color="black", width=1, row=2, col=1)

        # RSI & ADX Subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_color="red", line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dot", row=3, col=1)

        # Layout styling
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # 3. Data Details
        with st.expander("See Raw Data & Metrics"):
            st.dataframe(df.tail(10).style.format("{:.2f}"))

if __name__ == "__main__":
    main()
