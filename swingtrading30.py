import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Advanced Trading Platform", layout="wide")

# Title
st.title("🚀 Advanced Elliott Wave & Fibonacci Trading Platform")

# Sidebar inputs
st.sidebar.header("Configuration")

# Asset selection
asset_type = st.sidebar.selectbox(
    "Asset Type",
    ["Indian Stocks (NSE)", "US Stocks", "Crypto", "Forex", "Custom Ticker"]
)

# Ticker input based on asset type
if asset_type == "Indian Stocks (NSE)":
    ticker_options = ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
elif asset_type == "US Stocks":
    ticker_options = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
elif asset_type == "Crypto":
    ticker_options = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
elif asset_type == "Forex":
    ticker_options = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
else:
    ticker = st.sidebar.text_input("Enter Custom Ticker", "AAPL")

# Timeframe selection
timeframe_map = {
    "1 Minute": "1m", "3 Minutes": "3m", "5 Minutes": "5m", 
    "10 Minutes": "10m", "15 Minutes": "15m", "30 Minutes": "30m",
    "1 Hour": "1h", "2 Hours": "2h", "4 Hours": "4h",
    "1 Day": "1d", "1 Week": "1wk"
}
timeframe_display = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=8)
timeframe = timeframe_map[timeframe_display]

# Period selection
period_options = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"]
period = st.sidebar.selectbox("Period", period_options, index=10)

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30)
min_divergence_bars = st.sidebar.slider("Min Divergence Bars", 3, 20, 5)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.5, 0.5)

# Fibonacci levels
fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618, 2.618]

# Initialize session state for caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = {}

# Function to fetch data with caching
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        data = data.sort_index(ascending=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to calculate RSI
def calculate_rsi(data, period):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to find swing points
def find_swing_points(data, order=5):
    highs = argrelextrema(data['High'].values, np.greater, order=order)[0]
    lows = argrelextrema(data['Low'].values, np.less, order=order)[0]
    return highs, lows

# Function to calculate Fibonacci levels
def calculate_fibonacci_levels(swing_high, swing_low, direction='up'):
    diff = swing_high - swing_low
    if direction == 'up':
        levels = {f'Fib_{level}': swing_low + diff * level for level in fib_levels}
    else:
        levels = {f'Fib_{level}': swing_high - diff * level for level in fib_levels}
    return levels

# Function to detect RSI divergence
def detect_rsi_divergence(data, rsi, lookback=20):
    divergences = []
    
    for i in range(lookback, len(data)):
        price_slice = data['Close'].iloc[i-lookback:i+1]
        rsi_slice = rsi.iloc[i-lookback:i+1]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_slice) > min_divergence_bars:
            price_min_idx = price_slice.idxmin()
            rsi_min_idx = rsi_slice.idxmin()
            
            if price_slice.iloc[-1] < price_slice.iloc[0] and rsi_slice.iloc[-1] > rsi_slice.iloc[0]:
                if rsi_slice.iloc[-1] < rsi_oversold + 10:
                    divergences.append({
                        'index': i,
                        'type': 'bullish',
                        'strength': (rsi_slice.iloc[-1] - rsi_slice.iloc[0]) / rsi_slice.iloc[0]
                    })
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_slice) > min_divergence_bars:
            if price_slice.iloc[-1] > price_slice.iloc[0] and rsi_slice.iloc[-1] < rsi_slice.iloc[0]:
                if rsi_slice.iloc[-1] > rsi_overbought - 10:
                    divergences.append({
                        'index': i,
                        'type': 'bearish',
                        'strength': abs((rsi_slice.iloc[-1] - rsi_slice.iloc[0]) / rsi_slice.iloc[0])
                    })
    
    return divergences

# Function to calculate Elliott Wave patterns
def calculate_elliott_waves(data, swing_highs, swing_lows):
    waves = []
    
    # Combine and sort swing points
    swings = []
    for idx in swing_highs:
        if idx < len(data):
            swings.append({'index': idx, 'price': data['High'].iloc[idx], 'type': 'high'})
    for idx in swing_lows:
        if idx < len(data):
            swings.append({'index': idx, 'price': data['Low'].iloc[idx], 'type': 'low'})
    
    swings = sorted(swings, key=lambda x: x['index'])
    
    # Look for 5-wave patterns
    if len(swings) >= 5:
        for i in range(len(swings) - 4):
            wave_sequence = swings[i:i+5]
            
            # Check for impulse wave pattern (up: low-high-low-high-low)
            if (wave_sequence[0]['type'] == 'low' and 
                wave_sequence[1]['type'] == 'high' and
                wave_sequence[2]['type'] == 'low' and
                wave_sequence[3]['type'] == 'high' and
                wave_sequence[4]['type'] == 'low'):
                
                waves.append({
                    'start_idx': wave_sequence[0]['index'],
                    'end_idx': wave_sequence[4]['index'],
                    'direction': 'up',
                    'wave_type': 'impulse',
                    'waves': wave_sequence
                })
    
    return waves

# Function to generate trading signals
def generate_signals(data, rsi, divergences, waves, swing_highs, swing_lows):
    signals = []
    
    for div in divergences:
        idx = div['index']
        if idx >= len(data):
            continue
            
        entry_price = data['Close'].iloc[idx]
        entry_date = data.index[idx]
        
        # Find nearest swing points for stop loss and target
        if div['type'] == 'bullish':
            # Find recent swing low for stop loss
            recent_lows = [i for i in swing_lows if i < idx and i > max(0, idx-50)]
            if recent_lows:
                recent_low_idx = max(recent_lows)
                stop_loss = data['Low'].iloc[recent_low_idx] * 0.995
            else:
                stop_loss = entry_price * 0.97
            
            # Calculate Fibonacci targets
            if len(swing_lows) > 0 and len(swing_highs) > 0:
                recent_swing_low = data['Low'].iloc[max([i for i in swing_lows if i < idx], default=idx-10)]
                recent_swing_high = data['High'].iloc[max([i for i in swing_highs if i < idx], default=idx-10)]
                fib_levels_dict = calculate_fibonacci_levels(recent_swing_high, recent_swing_low, 'up')
                target = entry_price + (entry_price - stop_loss) * risk_reward_ratio
            else:
                target = entry_price * 1.05
            
            # Calculate probability based on divergence strength and RSI
            rsi_val = rsi.iloc[idx]
            prob = min(95, 60 + div['strength'] * 100 + (rsi_oversold - rsi_val) * 0.5)
            
            # Find exit point
            exit_idx = None
            exit_price = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 100, len(data))):
                if data['High'].iloc[j] >= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "Target Hit"
                    break
                elif data['Low'].iloc[j] <= stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    break
            
            if exit_idx is None and idx + 100 < len(data):
                exit_idx = idx + 100
                exit_price = data['Close'].iloc[exit_idx]
                exit_reason = "Time Exit"
            elif exit_idx is None:
                exit_idx = len(data) - 1
                exit_price = data['Close'].iloc[exit_idx]
                exit_reason = "Open Position"
            
            pnl = ((exit_price - entry_price) / entry_price) * 100
            
            signals.append({
                'Entry Date': entry_date,
                'Entry Price': round(entry_price, 2),
                'Target': round(target, 2),
                'Stop Loss': round(stop_loss, 2),
                'Exit Date': data.index[exit_idx],
                'Exit Price': round(exit_price, 2),
                'PnL %': round(pnl, 2),
                'Logic': f"Bullish RSI Divergence (Strength: {div['strength']:.2f})",
                'Probability': round(prob, 1),
                'Type': 'BUY'
            })
        
        elif div['type'] == 'bearish':
            # Find recent swing high for stop loss
            recent_highs = [i for i in swing_highs if i < idx and i > max(0, idx-50)]
            if recent_highs:
                recent_high_idx = max(recent_highs)
                stop_loss = data['High'].iloc[recent_high_idx] * 1.005
            else:
                stop_loss = entry_price * 1.03
            
            target = entry_price - (stop_loss - entry_price) * risk_reward_ratio
            
            # Calculate probability
            rsi_val = rsi.iloc[idx]
            prob = min(95, 60 + div['strength'] * 100 + (rsi_val - rsi_overbought) * 0.5)
            
            # Find exit point
            exit_idx = None
            exit_price = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 100, len(data))):
                if data['Low'].iloc[j] <= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "Target Hit"
                    break
                elif data['High'].iloc[j] >= stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    exit_reason = "Stop Loss Hit"
                    break
            
            if exit_idx is None and idx + 100 < len(data):
                exit_idx = idx + 100
                exit_price = data['Close'].iloc[exit_idx]
                exit_reason = "Time Exit"
            elif exit_idx is None:
                exit_idx = len(data) - 1
                exit_price = data['Close'].iloc[exit_idx]
                exit_reason = "Open Position"
            
            pnl = ((entry_price - exit_price) / entry_price) * 100
            
            signals.append({
                'Entry Date': entry_date,
                'Entry Price': round(entry_price, 2),
                'Target': round(target, 2),
                'Stop Loss': round(stop_loss, 2),
                'Exit Date': data.index[exit_idx],
                'Exit Price': round(exit_price, 2),
                'PnL %': round(pnl, 2),
                'Logic': f"Bearish RSI Divergence (Strength: {div['strength']:.2f})",
                'Probability': round(prob, 1),
                'Type': 'SELL'
            })
    
    return signals

# Main execution
if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner("Fetching data and analyzing..."):
        # Fetch data
        data = fetch_data(ticker, period, timeframe)
        
        if data is not None and not data.empty:
            st.success(f"✅ Data fetched successfully for {ticker}")
            
            # Display raw data
            st.subheader("📊 Fetched Data")
            st.dataframe(data, use_container_width=True)
            
            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Date", data.index.min().strftime('%Y-%m-%d'))
            with col2:
                st.metric("Max Date", data.index.max().strftime('%Y-%m-%d'))
            with col3:
                #st.metric("Min Close", f"{data['Close'].min():.2f}")
                st.metric("min close",f"{data['Close']})
            with col4:
                st.metric("Max Close", f"{data['Close']}")
            
            # Calculate indicators
            data['RSI'] = calculate_rsi(data, rsi_period)
            
            # Find swing points
            swing_highs, swing_lows = find_swing_points(data)
            
            # Calculate Elliott Waves
            waves = calculate_elliott_waves(data, swing_highs, swing_lows)
            
            # Detect divergences
            divergences = detect_rsi_divergence(data, data['RSI'])
            
            # Generate signals
            signals = generate_signals(data, data['RSI'], divergences, waves, swing_highs, swing_lows)
            
            if signals:
                signals_df = pd.DataFrame(signals)
                
                # Backtesting results
                st.subheader("📈 Backtesting Results")
                
                total_trades = len(signals_df)
                winning_trades = len(signals_df[signals_df['PnL %'] > 0])
                accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl = signals_df['PnL %'].sum()
                avg_pnl = signals_df['PnL %'].mean()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Winning Trades", winning_trades)
                with col3:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                with col4:
                    st.metric("Total PnL", f"{total_pnl:.2f}%")
                with col5:
                    st.metric("Avg PnL/Trade", f"{avg_pnl:.2f}%")
                
                # Strategy validation
                st.subheader("✅ Strategy Validation")
                
                buy_hold_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                with col2:
                    outperformance = total_pnl - buy_hold_return
                    st.metric("Strategy Outperformance", f"{outperformance:.2f}%", 
                             delta=f"{outperformance:.2f}%")
                
                if accuracy >= 80 and total_pnl > buy_hold_return:
                    st.success("✅ Strategy VALIDATED: Accuracy >80% and beats Buy & Hold!")
                elif accuracy >= 80:
                    st.warning("⚠️ Strategy meets accuracy target but doesn't beat Buy & Hold significantly")
                else:
                    st.error("❌ Strategy needs optimization: Accuracy below 80% target")
                
                # Display all signals
                st.subheader("📋 All Trading Signals")
                st.dataframe(signals_df, use_container_width=True)
                
                # Live recommendation (last signal)
                st.subheader("🎯 Live Recommendation")
                if len(signals_df) > 0:
                    last_signal = signals_df.iloc[-1]
                    
                    if last_signal['Exit Date'] == data.index[-1] or \
                       (data.index[-1] - last_signal['Entry Date']).days < 5:
                        st.info("📢 Active Signal")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Type", last_signal['Type'])
                            st.metric("Entry Price", f"${last_signal['Entry Price']}")
                        with col2:
                            st.metric("Target", f"${last_signal['Target']}")
                            st.metric("Stop Loss", f"${last_signal['Stop Loss']}")
                        with col3:
                            st.metric("Probability", f"{last_signal['Probability']}%")
                            st.metric("Risk/Reward", f"1:{risk_reward_ratio}")
                        
                        st.info(f"**Logic:** {last_signal['Logic']}")
                    else:
                        st.warning("No active signal on the last candle")
                
                # Returns heatmap
                st.subheader("🔥 Returns Heatmap")
                
                signals_df['Year'] = pd.to_datetime(signals_df['Entry Date']).dt.year
                signals_df['Month'] = pd.to_datetime(signals_df['Entry Date']).dt.month
                
                heatmap_data = signals_df.pivot_table(
                    values='PnL %', 
                    index='Month', 
                    columns='Year', 
                    aggfunc='sum'
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Year", y="Month", color="PnL %"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto"
                )
                fig_heatmap.update_layout(
                    title="Monthly Returns Heatmap",
                    font=dict(size=14)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Price chart with signals
                st.subheader("📊 Price Chart with Signals")
                
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                # Buy signals
                buy_signals = signals_df[signals_df['Type'] == 'BUY']
                fig.add_trace(go.Scatter(
                    x=buy_signals['Entry Date'],
                    y=buy_signals['Entry Price'],
                    mode='markers',
                    marker=dict(color='green', size=12, symbol='triangle-up'),
                    name='Buy Signal'
                ))
                
                # Sell signals
                sell_signals = signals_df[signals_df['Type'] == 'SELL']
                fig.add_trace(go.Scatter(
                    x=sell_signals['Entry Date'],
                    y=sell_signals['Entry Price'],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='triangle-down'),
                    name='Sell Signal'
                ))
                
                # Buy & Hold points
                fig.add_trace(go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[data['Close'].iloc[0], data['Close'].iloc[-1]],
                    mode='markers+lines',
                    marker=dict(color='blue', size=10),
                    name='Buy & Hold',
                    line=dict(dash='dash', color='blue')
                ))
                
                fig.update_layout(
                    title=f'{ticker} - Price Chart with Trading Signals',
                    yaxis_title='Price',
                    xaxis_title='Date',
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Chart
                st.subheader("📊 RSI Indicator")
                
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ))
                
                fig_rsi.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig_rsi.update_layout(
                    title='RSI Indicator',
                    yaxis_title='RSI',
                    xaxis_title='Date',
                    height=300
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            else:
                st.warning("No trading signals generated. Try adjusting parameters.")
        else:
            st.error("❌ Failed to fetch data. Please check ticker symbol and try again.")

# Footer
st.markdown("---")
st.markdown("**Note:** This is a professional trading platform for educational purposes. Past performance doesn't guarantee future results.")
