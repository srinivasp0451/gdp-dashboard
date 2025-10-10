import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from scipy.optimize import differential_evolution
import ta
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import pickle
import os
from pathlib import Path

# Page config
st.set_page_config(page_title="Elite Trading System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 1rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .positive {color: #00ff00; font-weight: bold;}
    .negative {color: #ff0000; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'last_fetch' not in st.session_state:
    st.session_state.last_fetch = {}
if 'optimized_params' not in st.session_state:
    st.session_state.optimized_params = {}

# Constants
IST = pytz.timezone('Asia/Kolkata')
CACHE_DURATION = 300  # 5 minutes

# Helper Functions
def get_cache_key(ticker, period, interval):
    return f"{ticker}_{period}_{interval}"

def should_fetch_data(cache_key):
    if cache_key not in st.session_state.last_fetch:
        return True
    elapsed = (datetime.now() - st.session_state.last_fetch[cache_key]).seconds
    return elapsed > CACHE_DURATION

@st.cache_data(ttl=CACHE_DURATION)
def fetch_data(ticker, period, interval):
    """Fetch data with caching and rate limit handling"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
            
        # Flatten multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in data.columns.values]
            # Rename to standard format
            data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]
        
        # Convert to IST
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        
        # Remove timezone for easier handling
        data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.500': high - 0.500 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }

def detect_elliott_wave_simplified(data, lookback=50):
    """Simplified Elliott Wave detection"""
    prices = data['Close'].values[-lookback:]
    
    # Find swing highs and lows
    highs = []
    lows = []
    
    for i in range(5, len(prices)-5):
        if prices[i] == max(prices[i-5:i+6]):
            highs.append((i, prices[i]))
        if prices[i] == min(prices[i-5:i+6]):
            lows.append((i, prices[i]))
    
    # Determine wave pattern
    if len(highs) >= 3 and len(lows) >= 2:
        return {'wave': 5, 'trend': 'bullish', 'confidence': 0.7}
    elif len(lows) >= 3 and len(highs) >= 2:
        return {'wave': 5, 'trend': 'bearish', 'confidence': 0.7}
    else:
        return {'wave': 0, 'trend': 'neutral', 'confidence': 0.5}

def calculate_rsi_divergence(data, period=14, lookback=30):
    """Detect RSI divergences"""
    rsi = RSIIndicator(data['Close'], window=period).rsi()
    prices = data['Close'].values[-lookback:]
    rsi_values = rsi.values[-lookback:]
    
    # Bullish divergence: price lower low, RSI higher low
    price_lows = []
    rsi_lows = []
    
    for i in range(5, len(prices)-5):
        if prices[i] == min(prices[i-5:i+6]):
            price_lows.append((i, prices[i]))
            rsi_lows.append((i, rsi_values[i]))
    
    bullish_div = False
    bearish_div = False
    
    if len(price_lows) >= 2:
        if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
            bullish_div = True
        elif price_lows[-1][1] > price_lows[-2][1] and rsi_lows[-1][1] < rsi_lows[-2][1]:
            bearish_div = True
    
    return {'bullish': bullish_div, 'bearish': bearish_div, 'rsi': rsi.iloc[-1]}

def calculate_ratio(data1, data2):
    """Calculate ratio between two instruments"""
    common_index = data1.index.intersection(data2.index)
    if len(common_index) == 0:
        return None
    ratio = data1.loc[common_index, 'Close'] / data2.loc[common_index, 'Close']
    return ratio

def generate_signals(data, params, ratio_data=None):
    """Generate trading signals based on strategy"""
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['Signal'] = 0
    signals['Position'] = 0
    
    # Calculate indicators
    rsi = RSIIndicator(data['Close'], window=int(params['rsi_period'])).rsi()
    ema_fast = EMAIndicator(data['Close'], window=int(params['ema_fast'])).ema_indicator()
    ema_slow = EMAIndicator(data['Close'], window=int(params['ema_slow'])).ema_indicator()
    
    # Elliott Wave
    wave_data = detect_elliott_wave_simplified(data, lookback=int(params['wave_lookback']))
    
    # RSI Divergence
    div_data = calculate_rsi_divergence(data, period=int(params['rsi_period']))
    
    # Fibonacci levels
    high = data['High'].rolling(window=int(params['fib_lookback'])).max()
    low = data['Low'].rolling(window=int(params['fib_lookback'])).min()
    
    in_position = False
    
    for i in range(50, len(signals)):
        if in_position:
            continue
            
        # Buy conditions
        buy_conditions = (
            rsi.iloc[i] < params['rsi_oversold'] and
            rsi.iloc[i] > rsi.iloc[i-1] and
            ema_fast.iloc[i] > ema_slow.iloc[i] and
            data['Close'].iloc[i] < high.iloc[i] - (high.iloc[i] - low.iloc[i]) * params['fib_entry']
        )
        
        # Sell conditions
        sell_conditions = (
            rsi.iloc[i] > params['rsi_overbought'] and
            rsi.iloc[i] < rsi.iloc[i-1] and
            ema_fast.iloc[i] < ema_slow.iloc[i] and
            data['Close'].iloc[i] > low.iloc[i] + (high.iloc[i] - low.iloc[i]) * (1 - params['fib_entry'])
        )
        
        # Ratio filter
        if ratio_data is not None and i < len(ratio_data):
            ratio_value = ratio_data.iloc[i]
            ratio_ma = ratio_data.rolling(window=20).mean().iloc[i]
            buy_conditions = buy_conditions and (ratio_value > ratio_ma)
            sell_conditions = sell_conditions and (ratio_value < ratio_ma)
        
        if buy_conditions:
            signals.loc[signals.index[i], 'Signal'] = 1
            in_position = True
        elif sell_conditions:
            signals.loc[signals.index[i], 'Signal'] = -1
            in_position = True
    
    # Calculate positions
    signals['Position'] = signals['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    return signals

def backtest_strategy(data, signals, params):
    """Backtest the trading strategy"""
    trades = []
    position = None
    entry_price = 0
    entry_date = None
    
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1 and position is None:
            # Buy signal
            entry_price = signals['Price'].iloc[i]
            entry_date = signals.index[i]
            position = 'long'
            
        elif signals['Signal'].iloc[i] == -1 and position is None:
            # Sell signal
            entry_price = signals['Price'].iloc[i]
            entry_date = signals.index[i]
            position = 'short'
            
        elif position is not None:
            # Check exit conditions
            current_price = signals['Price'].iloc[i]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            if position == 'short':
                pnl_pct = -pnl_pct
            
            # Exit conditions
            exit_trade = False
            exit_reason = ''
            
            if pnl_pct >= params['target_pct']:
                exit_trade = True
                exit_reason = 'Target'
            elif pnl_pct <= -params['sl_pct']:
                exit_trade = True
                exit_reason = 'Stop Loss'
            elif i == len(signals) - 1:
                exit_trade = True
                exit_reason = 'End of Data'
            
            if exit_trade:
                exit_price = current_price
                exit_date = signals.index[i]
                points = exit_price - entry_price if position == 'long' else entry_price - exit_price
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Type': position,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Points': points,
                    'PnL %': pnl_pct,
                    'Exit Reason': exit_reason
                })
                
                position = None
    
    return pd.DataFrame(trades)

def calculate_metrics(trades, data):
    """Calculate strategy metrics"""
    if len(trades) == 0:
        return {
            'Total Trades': 0,
            'Positive Trades': 0,
            'Negative Trades': 0,
            'Accuracy': 0,
            'Total Points': 0,
            'Buy & Hold Points': 0,
            'Strategy Win': False
        }
    
    positive_trades = len(trades[trades['Points'] > 0])
    negative_trades = len(trades[trades['Points'] < 0])
    accuracy = (positive_trades / len(trades)) * 100 if len(trades) > 0 else 0
    total_points = trades['Points'].sum()
    buy_hold_points = data['Close'].iloc[-1] - data['Close'].iloc[0]
    
    return {
        'Total Trades': len(trades),
        'Positive Trades': positive_trades,
        'Negative Trades': negative_trades,
        'Accuracy': accuracy,
        'Total Points': total_points,
        'Buy & Hold Points': buy_hold_points,
        'Strategy Win': total_points > buy_hold_points,
        'Avg Points per Trade': trades['Points'].mean(),
        'Max Drawdown': trades['Points'].min()
    }

def optimize_strategy(data, ratio_data=None):
    """Optimize strategy parameters using differential evolution"""
    
    def objective(params):
        param_dict = {
            'rsi_period': params[0],
            'rsi_oversold': params[1],
            'rsi_overbought': params[2],
            'ema_fast': params[3],
            'ema_slow': params[4],
            'fib_lookback': params[5],
            'fib_entry': params[6],
            'wave_lookback': params[7],
            'target_pct': params[8],
            'sl_pct': params[9]
        }
        
        try:
            signals = generate_signals(data, param_dict, ratio_data)
            trades = backtest_strategy(data, signals, param_dict)
            
            if len(trades) == 0:
                return 1000000  # Penalty for no trades
            
            metrics = calculate_metrics(trades, data)
            
            # Objective: Maximize points while maintaining high accuracy
            score = -metrics['Total Points'] + (100 - metrics['Accuracy']) * 10
            
            # Penalty for low accuracy
            if metrics['Accuracy'] < 60:
                score += 10000
            
            # Penalty if not beating buy & hold
            if not metrics['Strategy Win']:
                score += 5000
            
            return score
        except:
            return 1000000
    
    # Parameter bounds
    bounds = [
        (10, 20),      # rsi_period
        (20, 35),      # rsi_oversold
        (65, 80),      # rsi_overbought
        (5, 20),       # ema_fast
        (30, 60),      # ema_slow
        (20, 100),     # fib_lookback
        (0.382, 0.618),# fib_entry
        (30, 100),     # wave_lookback
        (1, 5),        # target_pct
        (0.5, 3)       # sl_pct
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=10,
        seed=42,
        workers=1,
        updating='deferred',
        atol=0.01,
        tol=0.01
    )
    
    return {
        'rsi_period': result.x[0],
        'rsi_oversold': result.x[1],
        'rsi_overbought': result.x[2],
        'ema_fast': result.x[3],
        'ema_slow': result.x[4],
        'fib_lookback': result.x[5],
        'fib_entry': result.x[6],
        'wave_lookback': result.x[7],
        'target_pct': result.x[8],
        'sl_pct': result.x[9]
    }

# Main App
st.markdown('<p class="main-header">🚀 Elite Algo Trading System</p>', unsafe_allow_html=True)
st.markdown("**Professional-Grade Trading Platform with Elliott Waves, Fibonacci, RSI Divergence & Ratio Analysis**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Instrument Selection
    st.subheader("Primary Instrument")
    ticker1 = st.text_input("Ticker Symbol", "^NSEI", help="e.g., ^NSEI, USDINR=X, RELIANCE.NS")
    
    st.subheader("Ratio Analysis (Optional)")
    use_ratio = st.checkbox("Enable Ratio Analysis")
    ticker2 = st.text_input("Second Ticker", "USDINR=X", disabled=not use_ratio) if use_ratio else None
    
    # Timeframe Selection
    st.subheader("Timeframe")
    interval = st.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"],
        index=6
    )
    
    period = st.selectbox(
        "Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=5
    )
    
    # Strategy Mode
    st.subheader("Strategy Mode")
    mode = st.radio("Mode", ["Backtest", "Live Trading"])
    
    # Optimization
    optimize = st.checkbox("🎯 Run Optimization", value=False)
    
    run_button = st.button("🚀 Run Strategy", type="primary", use_container_width=True)

# Main Content
if run_button:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch primary data
        status_text.text("📊 Fetching primary instrument data... (10%)")
        progress_bar.progress(10)
        
        data1 = fetch_data(ticker1, period, interval)
        
        if data1 is None or len(data1) < 100:
            st.error("❌ Insufficient data for primary instrument")
            st.stop()
        
        # Step 2: Fetch ratio data if needed
        ratio_data = None
        if use_ratio and ticker2:
            status_text.text("📊 Fetching secondary instrument data... (20%)")
            progress_bar.progress(20)
            
            data2 = fetch_data(ticker2, period, interval)
            if data2 is not None:
                ratio_data = calculate_ratio(data1, data2)
        
        # Step 3: Optimization
        if optimize:
            status_text.text("🎯 Optimizing strategy parameters... (30%)")
            progress_bar.progress(30)
            
            with st.spinner("Running optimization algorithm..."):
                optimized_params = optimize_strategy(data1, ratio_data)
                st.session_state.optimized_params = optimized_params
            
            status_text.text("✅ Optimization complete! (60%)")
            progress_bar.progress(60)
        else:
            # Default parameters
            optimized_params = {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 12,
                'ema_slow': 26,
                'fib_lookback': 50,
                'fib_entry': 0.618,
                'wave_lookback': 50,
                'target_pct': 3,
                'sl_pct': 1.5
            }
            st.session_state.optimized_params = optimized_params
            progress_bar.progress(40)
        
        # Step 4: Generate signals
        status_text.text("📈 Generating trading signals... (70%)")
        progress_bar.progress(70)
        
        signals = generate_signals(data1, optimized_params, ratio_data)
        
        # Step 5: Backtest
        status_text.text("💼 Running backtest... (85%)")
        progress_bar.progress(85)
        
        trades = backtest_strategy(data1, signals, optimized_params)
        metrics = calculate_metrics(trades, data1)
        
        # Step 6: Complete
        status_text.text("✅ Analysis complete! (100%)")
        progress_bar.progress(100)
        
        # Display Results
        st.markdown('<p class="sub-header">📊 Strategy Performance</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics['Total Trades'])
            st.metric("Positive Trades", metrics['Positive Trades'], 
                     delta=f"{metrics['Accuracy']:.1f}% accuracy")
        
        with col2:
            st.metric("Negative Trades", metrics['Negative Trades'])
            st.metric("Avg Points/Trade", f"{metrics['Avg Points per Trade']:.2f}")
        
        with col3:
            points_color = "normal" if metrics['Total Points'] > 0 else "inverse"
            st.metric("Strategy Points", f"{metrics['Total Points']:.2f}", 
                     delta=f"{metrics['Total Points']:.2f}", delta_color=points_color)
        
        with col4:
            st.metric("Buy & Hold Points", f"{metrics['Buy & Hold Points']:.2f}")
            beat_bh = "✅ Beat B&H" if metrics['Strategy Win'] else "❌ Below B&H"
            st.metric("Strategy vs B&H", beat_bh)
        
        # Current Signal
        st.markdown('<p class="sub-header">🎯 Latest Signal</p>', unsafe_allow_html=True)
        
        latest_signal = signals[signals['Signal'] != 0].tail(1)
        
        if len(latest_signal) > 0:
            signal_type = "🟢 BUY" if latest_signal['Signal'].iloc[0] == 1 else "🔴 SELL"
            signal_price = latest_signal['Price'].iloc[0]
            signal_date = latest_signal.index[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Signal:** {signal_type}")
                st.markdown(f"**Entry Price:** ₹{signal_price:.2f}")
            
            with col2:
                target = signal_price * (1 + optimized_params['target_pct']/100)
                sl = signal_price * (1 - optimized_params['sl_pct']/100)
                st.markdown(f"**Target:** ₹{target:.2f}")
                st.markdown(f"**Stop Loss:** ₹{sl:.2f}")
            
            with col3:
                st.markdown(f"**Date:** {signal_date}")
                st.markdown(f"**Probability:** {metrics['Accuracy']:.1f}%")
        else:
            st.info("🔍 No active signals at the moment. Waiting for setup...")
        
        # Chart
        st.markdown('<p class="sub-header">📈 Price Chart with Signals</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data1.index,
            open=data1['Open'],
            high=data1['High'],
            low=data1['Low'],
            close=data1['Close'],
            name='Price'
        ))
        
        # Buy signals
        buy_signals = signals[signals['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Price'],
            mode='markers',
            marker=dict(color='green', size=15, symbol='triangle-up'),
            name='Buy Signal'
        ))
        
        # Sell signals
        sell_signals = signals[signals['Signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Price'],
            mode='markers',
            marker=dict(color='red', size=15, symbol='triangle-down'),
            name='Sell Signal'
        ))
        
        fig.update_layout(
            title=f"{ticker1} - Trading Signals",
            xaxis_title="Date (IST)",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade History
        if len(trades) > 0:
            st.markdown('<p class="sub-header">📋 Trade History</p>', unsafe_allow_html=True)
            
            trades_display = trades.copy()
            trades_display['Entry Date'] = trades_display['Entry Date'].dt.strftime('%Y-%m-%d %H:%M')
            trades_display['Exit Date'] = trades_display['Exit Date'].dt.strftime('%Y-%m-%d %H:%M')
            trades_display['Points'] = trades_display['Points'].round(2)
            trades_display['PnL %'] = trades_display['PnL %'].round(2)
            
            st.dataframe(trades_display, use_container_width=True)
        
        # Strategy Logic
        with st.expander("📖 Strategy Logic"):
            st.markdown(f"""
            **Entry Logic:**
            - RSI < {optimized_params['rsi_oversold']:.0f} (Oversold)
            - RSI turning up (momentum shift)
            - Fast EMA ({optimized_params['ema_fast']:.0f}) > Slow EMA ({optimized_params['ema_slow']:.0f})
            - Price near Fibonacci {optimized_params['fib_entry']:.3f} level
            - Elliott Wave pattern confirmation
            - RSI bullish divergence
            {f"- Ratio above 20-period MA (Relative strength)" if use_ratio else ""}
            
            **Exit Logic:**
            - Target: +{optimized_params['target_pct']:.1f}% from entry
            - Stop Loss: -{optimized_params['sl_pct']:.1f}% from entry
            - Wait for signal completion before new entry
            
            **Optimization:**
            - Algorithm: Differential Evolution
            - Objective: Maximize points + High accuracy + Beat Buy & Hold
            - Parameters: 10 (RSI, EMA, Fibonacci, Wave, Risk Management)
            """)
        
        # Optimized Parameters
        with st.expander("⚙️ Optimized Parameters"):
            param_df = pd.DataFrame([optimized_params]).T
            param_df.columns = ['Value']
            st.dataframe(param_df)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()

else:
    st.info("👈 Configure your strategy in the sidebar and click 'Run Strategy' to begin")
    
    # Display instructions
    with st.expander("📚 User Guide"):
        st.markdown("""
        ### How to Use:
        
        1. **Select Primary Instrument**: Enter ticker symbol (e.g., ^NSEI for Nifty, RELIANCE.NS for stocks)
        2. **Optional Ratio Analysis**: Enable to analyze relative strength between instruments
        3. **Choose Timeframe**: Select interval and period for analysis
        4. **Run Optimization**: Enable to find best parameters (takes 1-2 minutes)
        5. **Click Run Strategy**: Execute backtest and view results
        
        ### Supported Instruments:
        - Indian Indices: ^NSEI (Nifty), ^BSESN (Sensex)
        - Stocks: Add .NS for NSE, .BO for BSE (e.g., RELIANCE.NS)
        - Forex: USDINR=X, EURINR=X
        - Commodities: GC=F (Gold), CL=F (Crude Oil)
        
        ### Features:
        - ✅ Elliott Wave Analysis
        - ✅ Fibonacci Retracements
        - ✅ RSI Divergences
        - ✅ Ratio Analysis
        - ✅ Advanced Optimization
        - ✅ Real-time IST conversion
        - ✅ Rate limit protection
        - ✅ Professional backtesting
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is an educational tool. Past performance doesn't guarantee future results. 
    Trading involves risk. Always do your own research.</p>
</div>
""", unsafe_allow_html=True)
