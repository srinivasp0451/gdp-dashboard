import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Algo Trading", layout="wide")

st.title("📈 Algorithmic Trading Analysis")

# Test if basic Streamlit works
st.write("✅ App is running!")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Sidebar
st.sidebar.header("Configuration")

# Ticker selection
ticker = st.sidebar.selectbox(
    "Select Asset",
    ["^NSEI (NIFTY 50)", "^NSEBANK (Bank NIFTY)", "^BSESN (SENSEX)", "BTC-USD", "ETH-USD"],
    index=0
)

ticker_symbol = ticker.split(" ")[0]

# Timeframe selection
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h", "1d"])
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

st.sidebar.write("")
st.sidebar.write(f"**Selected:** {ticker_symbol}")
st.sidebar.write(f"**Timeframe/Period:** {timeframe}/{period}")

# Manual calculation functions
def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(data, period):
    """Calculate EMA"""
    return data.ewm(span=period, adjust=False, min_periods=1).mean()

def calculate_macd(data):
    """Calculate MACD"""
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    return macd, signal

def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr.fillna(0)

def calculate_volatility(data, period=20):
    """Calculate volatility"""
    returns = data.pct_change()
    vol = returns.rolling(window=period, min_periods=1).std() * np.sqrt(252) * 100
    return vol.fillna(0)

# Fetch button
if st.sidebar.button("🔄 Fetch & Analyze", use_container_width=True):
    try:
        with st.spinner(f"Fetching {ticker_symbol} data for {timeframe}/{period}..."):
            
            # Fetch data
            data = yf.download(ticker_symbol, period=period, interval=timeframe, progress=False)
            
            if data.empty:
                st.error(f"❌ No data received for {ticker_symbol}")
                st.stop()
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            st.success(f"✅ Successfully fetched {len(data)} data points!")
            
            # Calculate indicators
            st.info("Calculating indicators...")
            
            data['RSI'] = calculate_rsi(data['Close'])
            data['EMA20'] = calculate_ema(data['Close'], 20)
            data['EMA50'] = calculate_ema(data['Close'], 50)
            data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
            data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
            data['Volatility'] = calculate_volatility(data['Close'])
            
            # Calculate Z-Score
            mean = data['Close'].rolling(window=20, min_periods=1).mean()
            std = data['Close'].rolling(window=20, min_periods=1).std()
            data['ZScore'] = (data['Close'] - mean) / std.replace(0, np.nan)
            data['ZScore'] = data['ZScore'].fillna(0)
            
            st.session_state.data = data
            st.session_state.analysis_done = True
            
            st.success("✅ Analysis complete!")
            
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        st.exception(e)

# Display results
if st.session_state.analysis_done and st.session_state.data is not None:
    
    data = st.session_state.data
    
    st.header(f"Analysis Results - {timeframe}/{period}")
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric("Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
    
    with col2:
        rsi = data['RSI'].iloc[-1]
        st.metric("RSI", f"{rsi:.2f}")
    
    with col3:
        vol = data['Volatility'].iloc[-1]
        st.metric("Volatility", f"{vol:.2f}%")
    
    with col4:
        zscore = data['ZScore'].iloc[-1]
        st.metric("Z-Score", f"{zscore:.2f}")
    
    # Generate signal
    st.divider()
    st.subheader(f"🎯 Trading Signal - {timeframe}/{period}")
    
    signals = []
    reasons = []
    
    # RSI signal
    if rsi < 30:
        signals.append(1)
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        signals.append(-1)
        reasons.append(f"RSI overbought ({rsi:.1f})")
    
    # Z-Score signal
    if zscore < -2:
        signals.append(1)
        reasons.append(f"Z-Score oversold ({zscore:.2f})")
    elif zscore > 2:
        signals.append(-1)
        reasons.append(f"Z-Score overbought ({zscore:.2f})")
    
    # Trend signal
    if current_price > data['EMA20'].iloc[-1] > data['EMA50'].iloc[-1]:
        signals.append(1)
        reasons.append("Strong uptrend")
    elif current_price < data['EMA20'].iloc[-1] < data['EMA50'].iloc[-1]:
        signals.append(-1)
        reasons.append("Strong downtrend")
    
    # MACD signal
    if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
        signals.append(1)
        reasons.append("MACD bullish")
    else:
        signals.append(-1)
        reasons.append("MACD bearish")
    
    if signals:
        avg_signal = np.mean(signals)
        confidence = min(99, int(60 + len(reasons) * 7))
        
        if avg_signal > 0.3:
            st.success(f"🟢 BUY Signal - {confidence}% Confidence")
            signal_type = "BUY"
        elif avg_signal < -0.3:
            st.error(f"🔴 SELL Signal - {confidence}% Confidence")
            signal_type = "SELL"
        else:
            st.warning(f"🟡 HOLD Signal - {confidence}% Confidence")
            signal_type = "HOLD"
        
        st.write("**Reasons:**")
        for reason in reasons:
            st.write(f"• {reason}")
        
        # Trade levels
        if signal_type == "BUY":
            atr = data['ATR'].iloc[-1]
            stop_loss = current_price - (2 * atr)
            target1 = current_price + (2 * atr)
            target2 = current_price + (3 * atr)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry", f"${current_price:.2f}")
            with col2:
                st.metric("Stop Loss", f"${stop_loss:.2f}")
            with col3:
                st.metric("Target 1", f"${target1:.2f}")
    
    # Charts
    st.divider()
    st.subheader(f"📊 Charts - {timeframe}/{period}")
    
    # Price chart
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Price - {timeframe}/{period}', 'RSI'),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(x=data.index, y=data['EMA20'], name='EMA20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['EMA50'], name='EMA50', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.divider()
    st.subheader(f"📋 Recent Data - {timeframe}/{period}")
    
    display_df = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20', 'EMA50', 'Volatility', 'ZScore']].tail(20).copy()
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download
    csv = data.to_csv()
    st.download_button(
        label=f"📥 Download Data ({timeframe}/{period})",
        data=csv,
        file_name=f"{ticker_symbol}_{timeframe}_{period}.csv",
        mime="text/csv"
    )

else:
    # Welcome screen
    st.info("👈 Select asset and timeframe in the sidebar, then click 'Fetch & Analyze' to begin")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Features")
        st.write("• RSI Analysis")
        st.write("• Moving Averages")
        st.write("• MACD Indicator")
        st.write("• Volatility Analysis")
        st.write("• Z-Score Analysis")
    
    with col2:
        st.markdown("### 💡 Signals")
        st.write("• BUY/SELL/HOLD")
        st.write("• Confidence Scores")
        st.write("• Entry/Exit Levels")
        st.write("• Risk Management")
    
    with col3:
        st.markdown("### 📈 Assets")
        st.write("• NIFTY 50")
        st.write("• Bank NIFTY")
        st.write("• SENSEX")
        st.write("• Cryptocurrencies")
        st.write("• And more...")

st.sidebar.markdown("---")
st.sidebar.caption("v1.0 - Simplified Working Version")
