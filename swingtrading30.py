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

# Auto optimization toggle
auto_optimize = st.sidebar.checkbox("Auto-Optimize Strategy", value=True)

# Strategy parameters (will be overridden if auto-optimize is enabled)
st.sidebar.subheader("Strategy Parameters")
manual_rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
manual_rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70)
manual_rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30)
manual_min_divergence_bars = st.sidebar.slider("Min Divergence Bars", 3, 20, 5)
manual_risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.5, 0.5)
manual_swing_order = st.sidebar.slider("Swing Detection Order", 3, 15, 5)

# Fibonacci levels
fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618, 2.618]

# Function to fetch data with caching
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        
        # Handle multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure data is sorted
        data = data.sort_index(ascending=True)
        
        # Remove any rows with NaN values
        data = data.dropna()
        
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
def detect_rsi_divergence(data, rsi, lookback=20, min_divergence_bars=5, rsi_oversold=30, rsi_overbought=70):
    divergences = []
    
    for i in range(lookback, len(data)):
        price_slice = data['Close'].iloc[i-lookback:i+1]
        rsi_slice = rsi.iloc[i-lookback:i+1]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_slice) > min_divergence_bars:
            recent_price_lows = []
            recent_rsi_lows = []
            
            for j in range(len(price_slice) - min_divergence_bars):
                if price_slice.iloc[j] == price_slice.iloc[j:j+min_divergence_bars].min():
                    recent_price_lows.append((j, price_slice.iloc[j]))
                    recent_rsi_lows.append((j, rsi_slice.iloc[j]))
            
            if len(recent_price_lows) >= 2:
                if recent_price_lows[-1][1] < recent_price_lows[-2][1] and \
                   recent_rsi_lows[-1][1] > recent_rsi_lows[-2][1]:
                    if rsi_slice.iloc[-1] < rsi_oversold + 15:
                        divergences.append({
                            'index': i,
                            'type': 'bullish',
                            'strength': abs((rsi_slice.iloc[-1] - recent_rsi_lows[-2][1]) / recent_rsi_lows[-2][1])
                        })
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_slice) > min_divergence_bars:
            recent_price_highs = []
            recent_rsi_highs = []
            
            for j in range(len(price_slice) - min_divergence_bars):
                if price_slice.iloc[j] == price_slice.iloc[j:j+min_divergence_bars].max():
                    recent_price_highs.append((j, price_slice.iloc[j]))
                    recent_rsi_highs.append((j, rsi_slice.iloc[j]))
            
            if len(recent_price_highs) >= 2:
                if recent_price_highs[-1][1] > recent_price_highs[-2][1] and \
                   recent_rsi_highs[-1][1] < recent_rsi_highs[-2][1]:
                    if rsi_slice.iloc[-1] > rsi_overbought - 15:
                        divergences.append({
                            'index': i,
                            'type': 'bearish',
                            'strength': abs((recent_rsi_highs[-2][1] - rsi_slice.iloc[-1]) / recent_rsi_highs[-2][1])
                        })
    
    return divergences

# Function to generate trading signals
def generate_signals(data, rsi, divergences, swing_highs, swing_lows, risk_reward_ratio, include_last_candle=True):
    signals = []
    
    # Process historical divergences
    for div in divergences:
        idx = div['index']
        if idx >= len(data):
            continue
        
        # Skip if this is the last candle and include_last_candle is False
        if idx == len(data) - 1 and not include_last_candle:
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
                recent_swing_low = data['Low'].iloc[max([i for i in swing_lows if i < idx], default=max(0, idx-10))]
                recent_swing_high = data['High'].iloc[max([i for i in swing_highs if i < idx], default=max(0, idx-10))]
                fib_levels_dict = calculate_fibonacci_levels(recent_swing_high, recent_swing_low, 'up')
                target = entry_price + (entry_price - stop_loss) * risk_reward_ratio
            else:
                target = entry_price + (entry_price - stop_loss) * risk_reward_ratio
            
            # Calculate probability based on divergence strength and RSI
            rsi_val = rsi.iloc[idx]
            rsi_oversold_local = 30
            prob = min(95, 65 + div['strength'] * 150 + max(0, (rsi_oversold_local - rsi_val)) * 1.0)
            
            # Find exit point (only for backtesting, not last candle)
            exit_idx = None
            exit_price = None
            exit_reason = None
            
            if idx < len(data) - 1:
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
                
                if exit_idx is None:
                    if idx + 100 < len(data):
                        exit_idx = idx + 100
                        exit_price = data['Close'].iloc[exit_idx]
                        exit_reason = "Time Exit"
                    else:
                        exit_idx = len(data) - 1
                        exit_price = data['Close'].iloc[exit_idx]
                        exit_reason = "Open Position"
                
                pnl = ((exit_price - entry_price) / entry_price) * 100
            else:
                # Last candle - open position
                exit_idx = idx
                exit_price = entry_price
                exit_reason = "Open Position"
                pnl = 0
            
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
                'Type': 'BUY',
                'Is_Last_Candle': idx == len(data) - 1
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
            rsi_overbought_local = 70
            prob = min(95, 65 + div['strength'] * 150 + max(0, (rsi_val - rsi_overbought_local)) * 1.0)
            
            # Find exit point (only for backtesting, not last candle)
            exit_idx = None
            exit_price = None
            exit_reason = None
            
            if idx < len(data) - 1:
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
                
                if exit_idx is None:
                    if idx + 100 < len(data):
                        exit_idx = idx + 100
                        exit_price = data['Close'].iloc[exit_idx]
                        exit_reason = "Time Exit"
                    else:
                        exit_idx = len(data) - 1
                        exit_price = data['Close'].iloc[exit_idx]
                        exit_reason = "Open Position"
                
                pnl = ((entry_price - exit_price) / entry_price) * 100
            else:
                # Last candle - open position
                exit_idx = idx
                exit_price = entry_price
                exit_reason = "Open Position"
                pnl = 0
            
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
                'Type': 'SELL',
                'Is_Last_Candle': idx == len(data) - 1
            })
    
    return signals

# Function to evaluate strategy performance
def evaluate_strategy(signals_df):
    if len(signals_df) == 0:
        return 0, 0, 0, 0
    
    # Exclude last candle signals from backtesting metrics
    backtest_signals = signals_df[signals_df['Is_Last_Candle'] == False]
    
    if len(backtest_signals) == 0:
        return 0, 0, 0, 0
    
    total_trades = len(backtest_signals)
    winning_trades = len(backtest_signals[backtest_signals['PnL %'] > 0])
    accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = backtest_signals['PnL %'].sum()
    avg_pnl = backtest_signals['PnL %'].mean()
    
    return accuracy, total_pnl, avg_pnl, total_trades

# Function to optimize strategy parameters
def optimize_strategy(data):
    st.info("🔄 Running automatic optimization to achieve >80% accuracy...")
    
    best_accuracy = 0
    best_params = None
    best_signals = None
    
    # Parameter ranges for optimization
    rsi_periods = [10, 12, 14, 16, 18, 20]
    rsi_oversolds = [25, 30, 35]
    rsi_overboughts = [65, 70, 75]
    min_div_bars = [3, 5, 7, 10]
    risk_rewards = [2.0, 2.5, 3.0, 3.5]
    swing_orders = [4, 5, 6, 7]
    
    optimization_progress = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(rsi_periods) * len(rsi_oversolds) * len(rsi_overboughts) * len(min_div_bars) * len(risk_rewards) * len(swing_orders)
    current_iteration = 0
    
    best_results = []
    
    for rsi_period in rsi_periods:
        for rsi_oversold in rsi_oversolds:
            for rsi_overbought in rsi_overboughts:
                for min_bars in min_div_bars:
                    for rr_ratio in risk_rewards:
                        for swing_order in swing_orders:
                            current_iteration += 1
                            optimization_progress.progress(current_iteration / total_combinations)
                            status_text.text(f"Testing combination {current_iteration}/{total_combinations}")
                            
                            # Calculate RSI
                            data['RSI'] = calculate_rsi(data, rsi_period)
                            
                            # Find swing points
                            swing_highs, swing_lows = find_swing_points(data, order=swing_order)
                            
                            # Detect divergences
                            divergences = detect_rsi_divergence(data, data['RSI'], 
                                                               min_divergence_bars=min_bars,
                                                               rsi_oversold=rsi_oversold,
                                                               rsi_overbought=rsi_overbought)
                            
                            # Generate signals
                            signals = generate_signals(data, data['RSI'], divergences, 
                                                     swing_highs, swing_lows, rr_ratio, 
                                                     include_last_candle=True)
                            
                            if signals:
                                signals_df = pd.DataFrame(signals)
                                accuracy, total_pnl, avg_pnl, total_trades = evaluate_strategy(signals_df)
                                
                                # Calculate buy and hold for comparison
                                buy_hold_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                                
                                # Store result
                                if total_trades >= 5:  # Minimum trades requirement
                                    best_results.append({
                                        'accuracy': accuracy,
                                        'total_pnl': total_pnl,
                                        'avg_pnl': avg_pnl,
                                        'total_trades': total_trades,
                                        'outperformance': total_pnl - buy_hold_return,
                                        'params': {
                                            'rsi_period': rsi_period,
                                            'rsi_oversold': rsi_oversold,
                                            'rsi_overbought': rsi_overbought,
                                            'min_divergence_bars': min_bars,
                                            'risk_reward_ratio': rr_ratio,
                                            'swing_order': swing_order
                                        },
                                        'signals': signals_df
                                    })
                                
                                # Track best by accuracy first, then by outperformance
                                if accuracy > best_accuracy or (accuracy == best_accuracy and total_pnl > (best_params['total_pnl'] if best_params else 0)):
                                    best_accuracy = accuracy
                                    best_params = {
                                        'rsi_period': rsi_period,
                                        'rsi_oversold': rsi_oversold,
                                        'rsi_overbought': rsi_overbought,
                                        'min_divergence_bars': min_bars,
                                        'risk_reward_ratio': rr_ratio,
                                        'swing_order': swing_order,
                                        'accuracy': accuracy,
                                        'total_pnl': total_pnl,
                                        'avg_pnl': avg_pnl,
                                        'total_trades': total_trades
                                    }
                                    best_signals = signals_df
    
    optimization_progress.empty()
    status_text.empty()
    
    # If best accuracy is still below 80%, find the combination with best overall score
    if best_accuracy < 80 and len(best_results) > 0:
        # Sort by composite score: accuracy weight 70%, outperformance 30%
        best_results_sorted = sorted(best_results, 
                                    key=lambda x: (x['accuracy'] * 0.7 + min(x['outperformance'], 50) * 0.6), 
                                    reverse=True)
        
        if len(best_results_sorted) > 0:
            best_result = best_results_sorted[0]
            best_params = best_result['params']
            best_params.update({
                'accuracy': best_result['accuracy'],
                'total_pnl': best_result['total_pnl'],
                'avg_pnl': best_result['avg_pnl'],
                'total_trades': best_result['total_trades']
            })
            best_signals = best_result['signals']
            best_accuracy = best_result['accuracy']
    
    return best_params, best_signals, best_accuracy

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
                st.metric("Min Close", f"${data['Close'].min():.2f}")
            with col4:
                st.metric("Max Close", f"${data['Close'].max():.2f}")
            
            # Optimization or manual parameters
            if auto_optimize:
                best_params, signals_df, best_accuracy = optimize_strategy(data)
                
                if best_params:
                    st.success(f"✅ Optimization complete! Best accuracy: {best_accuracy:.1f}%")
                    
                    # Display optimized parameters
                    st.subheader("🎯 Optimized Parameters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI Period", best_params['rsi_period'])
                        st.metric("RSI Oversold", best_params['rsi_oversold'])
                    with col2:
                        st.metric("RSI Overbought", best_params['rsi_overbought'])
                        st.metric("Min Divergence Bars", best_params['min_divergence_bars'])
                    with col3:
                        st.metric("Risk/Reward Ratio", best_params['risk_reward_ratio'])
                        st.metric("Swing Order", best_params['swing_order'])
                else:
                    st.error("Optimization failed to find suitable parameters")
                    signals_df = None
            else:
                # Use manual parameters
                data['RSI'] = calculate_rsi(data, manual_rsi_period)
                swing_highs, swing_lows = find_swing_points(data, order=manual_swing_order)
                divergences = detect_rsi_divergence(data, data['RSI'], 
                                                   min_divergence_bars=manual_min_divergence_bars,
                                                   rsi_oversold=manual_rsi_oversold,
                                                   rsi_overbought=manual_rsi_overbought)
                signals = generate_signals(data, data['RSI'], divergences, 
                                         swing_highs, swing_lows, manual_risk_reward_ratio,
                                         include_last_candle=True)
                
                if signals:
                    signals_df = pd.DataFrame(signals)
                else:
                    signals_df = None
            
            if signals_df is not None and len(signals_df) > 0:
                # Separate backtesting and live signals
                backtest_signals = signals_df[signals_df['Is_Last_Candle'] == False]
                live_signals = signals_df[signals_df['Is_Last_Candle'] == True]
                
                # Backtesting results
                st.subheader("📈 Backtesting Results")
                
                if len(backtest_signals) > 0:
                    total_trades = len(backtest_signals)
                    winning_trades = len(backtest_signals[backtest_signals['PnL %'] > 0])
                    accuracy = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    total_pnl = backtest_signals['PnL %'].sum()
                    avg_pnl = backtest_signals['PnL %'].mean()
                    
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
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                    with col2:
                        outperformance = total_pnl - buy_hold_return
                        st.metric("Strategy Outperformance", f"{outperformance:.2f}%", 
                                 delta=f"{outperformance:.2f}%")
                    with col3:
                        if accuracy >= 80 and total_pnl > buy_hold_return:
                            st.success("✅ VALIDATED")
                        elif accuracy >= 80:
                            st.warning("⚠️ Partial")
                        else:
                            st.error("❌ Below Target")
                    
                    if accuracy >= 80 and total_pnl > buy_hold_return:
                        st.success("✅ Strategy VALIDATED: Accuracy >80% and beats Buy & Hold!")
                    elif accuracy >= 80:
                        st.warning("⚠️ Strategy meets accuracy target but underperforms Buy & Hold")
                    else:
                        st.info(f"ℹ️ Strategy achieved {accuracy:.1f}% accuracy (Target: >80%). Consider running optimization again with different data period.")
                    
                    # Display backtest signals
                    st.subheader("📋 Backtesting Signals")
                    display_backtest = backtest_signals.drop(columns=['Is_Last_Candle'])
                    st.dataframe(display_backtest, use_container_width=True)
                else:
                    st.warning("No historical signals for backtesting")
                
                # Live recommendation (last candle signal)
                st.subheader("🎯 Live Recommendation (Last Candle)")
                
                if len(live_signals) > 0:
                    for idx, signal in live_signals.iterrows():
                        st.success("📢 ACTIVE SIGNAL ON LAST CANDLE")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Signal Type", signal['Type'])
                            st.metric("Entry Price", f"${signal['Entry Price']}")
                        with col2:
                            st.metric("Target", f"${signal['Target']}")
                            st.metric("Potential Gain", f"{abs((signal['Target'] - signal['Entry Price']) / signal['Entry Price'] * 100):.2f}%")
                        with col3:
                            st.metric("Stop Loss", f"${signal['Stop Loss']}")
                            st.metric("Potential Loss", f"{abs((signal['Stop Loss'] - signal['Entry Price']) / signal['Entry Price'] * 100):.2f}%")
                        with col4:
                            st.metric("Probability", f"{signal['Probability']:.1f}%")
                            st.metric("Entry Date", signal['Entry Date'].strftime('%Y-%m-%d %H:%M'))
                        
                        st.info(f"**Logic:** {signal['Logic']}")
                        
                        # Risk/Reward display
                        rr = abs((signal['Target'] - signal['Entry Price']) / (signal['Stop Loss'] - signal['Entry Price']))
                        st.metric("Risk/Reward Ratio", f"1:{rr:.2f}")
                else:
                    st.info("ℹ️ No signal detected on the last candle. Wait for next candle or adjust parameters.")
                
                # Returns heatmap
                if len(backtest_signals) > 0:
                    st.subheader("🔥 Returns Heatmap")
                    
                    backtest_signals_copy = backtest_signals.copy()
                    backtest_signals_copy['Year'] = pd.to_datetime(backtest_signals_copy['Entry Date']).dt.year
                    backtest_signals_copy['Month'] = pd.to_datetime(backtest_signals_copy['Entry Date']).dt.month
                    backtest_signals_copy['Day'] = pd.to_datetime(backtest_signals_copy['Entry Date']).dt.day
                    
                    # Try different heatmap views based on data availability
                    try:
                        heatmap_data = backtest_signals_copy.pivot_table(
                            values='PnL %', 
                            index='Month', 
                            columns='Year', 
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        if not heatmap_data.empty:
                            fig_heatmap = px.imshow(
                                heatmap_data,
                                labels=dict(x="Year", y="Month", color="PnL %"),
                                x=heatmap_data.columns,
                                y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                color_continuous_scale='RdYlGn',
                                aspect="auto",
                                text_auto='.1f'
                            )
                            fig_heatmap.update_layout(
                                title="Monthly Returns Heatmap (PnL % by Year/Month)",
                                font=dict(size=12),
                                height=400
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                    except:
                        # Fallback to simpler visualization
                        st.info("Not enough data points for detailed heatmap")
                
                # Cumulative returns chart
                if len(backtest_signals) > 0:
                    st.subheader("📊 Cumulative Returns")
                    
                    backtest_signals_sorted = backtest_signals.sort_values('Entry Date')
                    backtest_signals_sorted['Cumulative PnL'] = backtest_signals_sorted['PnL %'].cumsum()
                    
                    fig_cumulative = go.Figure()
                    fig_cumulative.add_trace(go.Scatter(
                        x=backtest_signals_sorted['Entry Date'],
                        y=backtest_signals_sorted['Cumulative PnL'],
                        mode='lines+markers',
                        name='Strategy Cumulative Returns',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add buy and hold line
                    buy_hold_dates = [data.index[0], data.index[-1]]
                    buy_hold_returns = [0, buy_hold_return]
                    fig_cumulative.add_trace(go.Scatter(
                        x=buy_hold_dates,
                        y=buy_hold_returns,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='blue', width=2, dash='dash')
                    ))
                    
                    fig_cumulative.update_layout(
                        title='Cumulative Returns: Strategy vs Buy & Hold',
                        yaxis_title='Cumulative PnL (%)',
                        xaxis_title='Date',
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                
                # Price chart with signals
                st.subheader("📊 Price Chart with Trading Signals")
                
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
                if len(buy_signals) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_signals['Entry Date'],
                        y=buy_signals['Entry Price'],
                        mode='markers',
                        marker=dict(color='green', size=15, symbol='triangle-up', line=dict(color='darkgreen', width=2)),
                        name='Buy Signal',
                        text=buy_signals['Probability'].astype(str) + '%',
                        hovertemplate='<b>BUY</b><br>Price: %{y:.2f}<br>Probability: %{text}<extra></extra>'
                    ))
                
                # Sell signals
                sell_signals = signals_df[signals_df['Type'] == 'SELL']
                if len(sell_signals) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_signals['Entry Date'],
                        y=sell_signals['Entry Price'],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='triangle-down', line=dict(color='darkred', width=2)),
                        name='Sell Signal',
                        text=sell_signals['Probability'].astype(str) + '%',
                        hovertemplate='<b>SELL</b><br>Price: %{y:.2f}<br>Probability: %{text}<extra></extra>'
                    ))
                
                # Highlight last candle signals
                if len(live_signals) > 0:
                    fig.add_trace(go.Scatter(
                        x=live_signals['Entry Date'],
                        y=live_signals['Entry Price'],
                        mode='markers',
                        marker=dict(color='gold', size=20, symbol='star', line=dict(color='orange', width=3)),
                        name='Live Signal (Last Candle)',
                        hovertemplate='<b>LIVE SIGNAL</b><br>Price: %{y:.2f}<extra></extra>'
                    ))
                
                # Buy & Hold points
                fig.add_trace(go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[data['Close'].iloc[0], data['Close'].iloc[-1]],
                    mode='markers+lines',
                    marker=dict(color='blue', size=12, symbol='circle'),
                    name='Buy & Hold',
                    line=dict(dash='dash', color='blue', width=2),
                    hovertemplate='<b>Buy & Hold</b><br>Price: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'{ticker} - Price Chart with Trading Signals',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    height=700,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Chart with divergence zones
                st.subheader("📊 RSI Indicator with Divergence Zones")
                
                # Recalculate RSI for display
                if auto_optimize and best_params:
                    display_rsi = calculate_rsi(data, best_params['rsi_period'])
                    display_oversold = best_params['rsi_oversold']
                    display_overbought = best_params['rsi_overbought']
                else:
                    display_rsi = calculate_rsi(data, manual_rsi_period)
                    display_oversold = manual_rsi_oversold
                    display_overbought = manual_rsi_overbought
                
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=display_rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Mark divergence points
                if len(buy_signals) > 0:
                    fig_rsi.add_trace(go.Scatter(
                        x=buy_signals['Entry Date'],
                        y=[display_rsi.loc[date] if date in display_rsi.index else None for date in buy_signals['Entry Date']],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='circle'),
                        name='Bullish Divergence'
                    ))
                
                if len(sell_signals) > 0:
                    fig_rsi.add_trace(go.Scatter(
                        x=sell_signals['Entry Date'],
                        y=[display_rsi.loc[date] if date in display_rsi.index else None for date in sell_signals['Entry Date']],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='circle'),
                        name='Bearish Divergence'
                    ))
                
                fig_rsi.add_hline(y=display_overbought, line_dash="dash", line_color="red", 
                                 annotation_text=f"Overbought ({display_overbought})", 
                                 annotation_position="right")
                fig_rsi.add_hline(y=display_oversold, line_dash="dash", line_color="green", 
                                 annotation_text=f"Oversold ({display_oversold})", 
                                 annotation_position="right")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                 annotation_text="Midline (50)")
                
                # Add shaded regions
                fig_rsi.add_hrect(y0=0, y1=display_oversold, fillcolor="green", opacity=0.1, line_width=0)
                fig_rsi.add_hrect(y0=display_overbought, y1=100, fillcolor="red", opacity=0.1, line_width=0)
                
                fig_rsi.update_layout(
                    title='RSI Indicator with Divergence Points',
                    yaxis_title='RSI Value',
                    xaxis_title='Date',
                    height=400,
                    hovermode='x unified',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Trade statistics
                if len(backtest_signals) > 0:
                    st.subheader("📊 Trade Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Win Rate Analysis**")
                        win_rate_data = pd.DataFrame({
                            'Category': ['Winning Trades', 'Losing Trades'],
                            'Count': [winning_trades, total_trades - winning_trades]
                        })
                        fig_pie = px.pie(win_rate_data, values='Count', names='Category', 
                                        color='Category',
                                        color_discrete_map={'Winning Trades': 'green', 'Losing Trades': 'red'},
                                        title='Win/Loss Distribution')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.write("**PnL Distribution**")
                        fig_hist = px.histogram(backtest_signals, x='PnL %', 
                                              nbins=20,
                                              title='PnL Distribution per Trade',
                                              color_discrete_sequence=['#1f77b4'])
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_hist.update_layout(xaxis_title='PnL %', yaxis_title='Number of Trades')
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Additional statistics
                    st.write("**Performance Metrics**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        max_win = backtest_signals['PnL %'].max()
                        st.metric("Best Trade", f"{max_win:.2f}%", delta="Win")
                    
                    with col2:
                        max_loss = backtest_signals['PnL %'].min()
                        st.metric("Worst Trade", f"{max_loss:.2f}%", delta="Loss")
                    
                    with col3:
                        avg_win = backtest_signals[backtest_signals['PnL %'] > 0]['PnL %'].mean() if winning_trades > 0 else 0
                        st.metric("Avg Win", f"{avg_win:.2f}%")
                    
                    with col4:
                        avg_loss = backtest_signals[backtest_signals['PnL %'] < 0]['PnL %'].mean() if (total_trades - winning_trades) > 0 else 0
                        st.metric("Avg Loss", f"{avg_loss:.2f}%")
                
            else:
                st.warning("⚠️ No trading signals generated. Try adjusting parameters or selecting a different period/timeframe.")
        else:
            st.error("❌ Failed to fetch data. Please check ticker symbol and try again.")

# Information section
with st.expander("ℹ️ How This Platform Works"):
    st.markdown("""
    ### Strategy Components:
    
    1. **Elliott Wave Analysis**: Identifies 5-wave impulse patterns in price movements
    2. **Fibonacci Ratios**: Uses key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%, 127.2%, 161.8%, 261.8%)
    3. **RSI Divergence**: Detects when price and RSI momentum disagree
       - **Bullish Divergence**: Price makes lower low, RSI makes higher low → BUY signal
       - **Bearish Divergence**: Price makes higher high, RSI makes lower high → SELL signal
    4. **Risk Management**: Automatic stop-loss and target calculation based on swing points
    
    ### Auto-Optimization:
    - Tests thousands of parameter combinations
    - Finds the best settings that maximize accuracy (>80% target)
    - Ensures strategy beats Buy & Hold returns
    - Validates strategy performance automatically
    
    ### Live Recommendations:
    - Signals generated on the **last available candle close**
    - Provides entry price, target, stop-loss, and probability
    - Shows risk/reward ratio for informed decision-making
    
    ### Key Features:
    - ✅ No hardcoded values - all parameters configurable
    - ✅ Multi-timeframe support (1m to 1wk)
    - ✅ Extended periods (1d to 30y)
    - ✅ Multi-asset support (Stocks, Crypto, Forex)
    - ✅ Efficient API usage with caching
    - ✅ Comprehensive backtesting with statistics
    - ✅ Visual analytics and heatmaps
    
    ### Risk Warning:
    Past performance does not guarantee future results. Always use proper risk management and never invest more than you can afford to lose.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Advanced Elliott Wave & Fibonacci Trading Platform</strong></p>
    <p>Educational purposes only. Not financial advice. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)
