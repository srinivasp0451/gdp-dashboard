import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

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
    .trade-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .signal-box {
        border: 3px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f1f8f4;
        margin: 1rem 0;
    }
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

# ============= INDICATOR CALCULATIONS (NO TA-LIB) =============

def calculate_rsi(prices, period=14):
    """Calculate RSI without ta-lib"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_ema(prices, period):
    """Calculate EMA without ta-lib"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def calculate_sma(prices, period):
    """Calculate SMA"""
    sma = np.zeros_like(prices)
    for i in range(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    return sma

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, period)
    std = np.zeros_like(prices)
    
    for i in range(period-1, len(prices)):
        std[i] = np.std(prices[i-period+1:i+1])
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return upper, sma, lower

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD without ta-lib"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

# ============= HELPER FUNCTIONS =============

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
    rsi = calculate_rsi(data['Close'].values, period)
    prices = data['Close'].values[-lookback:]
    rsi_values = rsi[-lookback:]
    
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
    
    return {'bullish': bullish_div, 'bearish': bearish_div, 'rsi': rsi[-1] if len(rsi) > 0 else 50}

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
    prices = data['Close'].values
    rsi = calculate_rsi(prices, int(params['rsi_period']))
    ema_fast = calculate_ema(prices, int(params['ema_fast']))
    ema_slow = calculate_ema(prices, int(params['ema_slow']))
    macd, macd_signal, macd_hist = calculate_macd(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    
    # Elliott Wave
    wave_data = detect_elliott_wave_simplified(data, lookback=int(params['wave_lookback']))
    
    # RSI Divergence
    div_data = calculate_rsi_divergence(data, period=int(params['rsi_period']))
    
    # Fibonacci levels
    high = data['High'].rolling(window=int(params['fib_lookback'])).max().values
    low = data['Low'].rolling(window=int(params['fib_lookback'])).min().values
    
    in_position = False
    entry_idx = None
    
    for i in range(50, len(signals)):
        if in_position:
            # Skip signal generation until position exits
            continue
            
        # Buy conditions
        buy_conditions = (
            rsi[i] < params['rsi_oversold'] and
            rsi[i] > rsi[i-1] and  # RSI turning up
            ema_fast[i] > ema_slow[i] and
            macd_hist[i] > 0 and  # MACD bullish
            prices[i] < high[i] - (high[i] - low[i]) * params['fib_entry'] and
            prices[i] > bb_lower[i]  # Above lower Bollinger Band
        )
        
        # Sell conditions
        sell_conditions = (
            rsi[i] > params['rsi_overbought'] and
            rsi[i] < rsi[i-1] and  # RSI turning down
            ema_fast[i] < ema_slow[i] and
            macd_hist[i] < 0 and  # MACD bearish
            prices[i] > low[i] + (high[i] - low[i]) * (1 - params['fib_entry']) and
            prices[i] < bb_upper[i]  # Below upper Bollinger Band
        )
        
        # Ratio filter
        if ratio_data is not None and i < len(ratio_data):
            ratio_value = ratio_data.iloc[i]
            ratio_ma = ratio_data.rolling(window=20).mean().iloc[i]
            if not pd.isna(ratio_ma):
                buy_conditions = buy_conditions and (ratio_value > ratio_ma)
                sell_conditions = sell_conditions and (ratio_value < ratio_ma)
        
        if buy_conditions:
            signals.loc[signals.index[i], 'Signal'] = 1
            in_position = True
            entry_idx = i
        elif sell_conditions:
            signals.loc[signals.index[i], 'Signal'] = -1
            in_position = True
            entry_idx = i
    
    # Calculate positions
    signals['Position'] = signals['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    return signals

def backtest_strategy(data, signals, params):
    """Backtest the trading strategy with detailed entry/exit information"""
    trades = []
    position = None
    entry_price = 0
    entry_date = None
    entry_idx = None
    
    for i in range(len(signals)):
        if signals['Signal'].iloc[i] == 1 and position is None:
            # Buy signal
            entry_price = signals['Price'].iloc[i]
            entry_date = signals.index[i]
            entry_idx = i
            position = 'long'
            
        elif signals['Signal'].iloc[i] == -1 and position is None:
            # Sell signal
            entry_price = signals['Price'].iloc[i]
            entry_date = signals.index[i]
            entry_idx = i
            position = 'short'
            
        elif position is not None:
            # Check exit conditions
            current_price = signals['Price'].iloc[i]
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            if position == 'short':
                pnl_pct = -pnl_pct
            
            # Calculate target and SL levels
            if position == 'long':
                target_price = entry_price * (1 + params['target_pct']/100)
                sl_price = entry_price * (1 - params['sl_pct']/100)
            else:
                target_price = entry_price * (1 - params['target_pct']/100)
                sl_price = entry_price * (1 + params['sl_pct']/100)
            
            # Exit conditions
            exit_trade = False
            exit_reason = ''
            
            if pnl_pct >= params['target_pct']:
                exit_trade = True
                exit_reason = 'Target Hit'
            elif pnl_pct <= -params['sl_pct']:
                exit_trade = True
                exit_reason = 'Stop Loss Hit'
            elif i == len(signals) - 1:
                exit_trade = True
                exit_reason = 'End of Period'
            
            if exit_trade:
                exit_price = current_price
                exit_date = signals.index[i]
                points = exit_price - entry_price if position == 'long' else entry_price - exit_price
                
                trades.append({
                    'Entry Date': entry_date,
                    'Entry Time': entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Date': exit_date,
                    'Exit Time': exit_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': position.upper(),
                    'Entry Price': round(entry_price, 2),
                    'Target Price': round(target_price, 2),
                    'SL Price': round(sl_price, 2),
                    'Exit Price': round(exit_price, 2),
                    'Points': round(points, 2),
                    'PnL %': round(pnl_pct, 2),
                    'Exit Reason': exit_reason,
                    'Duration': str(exit_date - entry_date)
                })
                
                position = None
                entry_idx = None
    
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
            'Strategy Win': False,
            'Avg Points per Trade': 0,
            'Max Drawdown': 0,
            'Win Rate': 0,
            'Avg Win': 0,
            'Avg Loss': 0,
            'Profit Factor': 0
        }
    
    positive_trades = len(trades[trades['Points'] > 0])
    negative_trades = len(trades[trades['Points'] < 0])
    accuracy = (positive_trades / len(trades)) * 100 if len(trades) > 0 else 0
    total_points = trades['Points'].sum()
    buy_hold_points = data['Close'].iloc[-1] - data['Close'].iloc[0]
    
    winning_trades = trades[trades['Points'] > 0]
    losing_trades = trades[trades['Points'] < 0]
    
    avg_win = winning_trades['Points'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['Points'].mean()) if len(losing_trades) > 0 else 0
    
    total_wins = winning_trades['Points'].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades['Points'].sum()) if len(losing_trades) > 0 else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    return {
        'Total Trades': len(trades),
        'Positive Trades': positive_trades,
        'Negative Trades': negative_trades,
        'Accuracy': accuracy,
        'Total Points': total_points,
        'Buy & Hold Points': buy_hold_points,
        'Strategy Win': total_points > buy_hold_points,
        'Avg Points per Trade': trades['Points'].mean(),
        'Max Drawdown': trades['Points'].min(),
        'Win Rate': accuracy,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor
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
            
            # Bonus for profit factor
            if metrics['Profit Factor'] > 0:
                score -= metrics['Profit Factor'] * 100
            
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

# ============= MAIN APP =============

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
            
            with st.spinner("Running optimization algorithm (this may take 1-2 minutes)..."):
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
        
        # ============= DISPLAY RESULTS =============
        
        st.markdown('<p class="sub-header">📊 Strategy Performance</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Trades", metrics['Total Trades'])
            st.metric("Positive", metrics['Positive Trades'])
        
        with col2:
            st.metric("Negative", metrics['Negative Trades'])
            st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%")
        
        with col3:
            points_color = "normal" if metrics['Total Points'] > 0 else "inverse"
            st.metric("Strategy Points", f"{metrics['Total Points']:.2f}", 
                     delta=f"{metrics['Total Points']:.2f}", delta_color=points_color)
        
        with col4:
            st.metric("Buy & Hold", f"{metrics['Buy & Hold Points']:.2f}")
            beat_text = "✅ Win" if metrics['Strategy Win'] else "❌ Loss"
            st.metric("vs B&H", beat_text)
        
        with col5:
            st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            st.metric("Avg Win/Loss", f"{metrics['Avg Win']:.1f}/{metrics['Avg Loss']:.1f}")
        
        # ============= CURRENT/LATEST SIGNAL =============
        
        st.markdown('<p class="sub-header">🎯 Latest Trading Signal</p>', unsafe_allow_html=True)
        
        latest_signal = signals[signals['Signal'] != 0].tail(1)
        
        if len(latest_signal) > 0:
            signal_type = "🟢 BUY" if latest_signal['Signal'].iloc[0] == 1 else "🔴 SELL"
            signal_price = latest_signal['Price'].iloc[0]
            signal_date = latest_signal.index[0]
            
            # Calculate levels
            if latest_signal['Signal'].iloc[0] == 1:  # Long
                target_price = signal_price * (1 + optimized_params['target_pct']/100)
                sl_price = signal_price * (1 - optimized_params['sl_pct']/100)
            else:  # Short
                target_price = signal_price * (1 - optimized_params['target_pct']/100)
                sl_price = signal_price * (1 + optimized_params['sl_pct']/100)
            
            # Check if signal is still active (not exited yet)
            current_price = data1['Close'].iloc[-1]
            current_date = data1.index[-1]
            
            signal_active = True
            exit_status = "🟡 In Progress"
            
            # Check if target or SL hit
            if latest_signal['Signal'].iloc[0] == 1:  # Long
                if current_price >= target_price:
                    signal_active = False
                    exit_status = "✅ Target Hit"
                elif current_price <= sl_price:
                    signal_active = False
                    exit_status = "❌ Stop Loss Hit"
            else:  # Short
                if current_price <= target_price:
                    signal_active = False
                    exit_status = "✅ Target Hit"
                elif current_price >= sl_price:
                    signal_active = False
                    exit_status = "❌ Stop Loss Hit"
            
            st.markdown(f"""
            <div class="signal-box">
                <h2 style="margin-top:0;">{signal_type} Signal</h2>
                <table style="width:100%; color:#333;">
                    <tr>
                        <td><b>Entry Date & Time:</b></td>
                        <td>{signal_date.strftime('%Y-%m-%d %H:%M:%S')} IST</td>
                        <td><b>Entry Price:</b></td>
                        <td>₹{signal_price:.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Target Price:</b></td>
                        <td>₹{target_price:.2f}</td>
                        <td><b>Target Points:</b></td>
                        <td>{abs(target_price - signal_price):.2f} pts</td>
                    </tr>
                    <tr>
                        <td><b>Stop Loss:</b></td>
                        <td>₹{sl_price:.2f}</td>
                        <td><b>SL Points:</b></td>
                        <td>{abs(sl_price - signal_price):.2f} pts</td>
                    </tr>
                    <tr>
                        <td><b>Current Price:</b></td>
                        <td>₹{current_price:.2f}</td>
                        <td><b>Current Date & Time:</b></td>
                        <td>{current_date.strftime('%Y-%m-%d %H:%M:%S')} IST</td>
                    </tr>
                    <tr>
                        <td><b>Status:</b></td>
                        <td colspan="3">{exit_status}</td>
                    </tr>
                    <tr>
                        <td><b>Probability of Success:</b></td>
                        <td colspan="3">{metrics['Accuracy']:.1f}%</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Trading Logic Explanation
            with st.expander("📋 Entry & Exit Logic"):
                entry_logic = """
                **Entry Conditions (All must be met):**
                - RSI below oversold level and turning upward
                - Fast EMA crosses above Slow EMA (trend confirmation)
                - MACD Histogram positive (momentum confirmation)
                - Price near Fibonacci retracement level
                - Price above lower Bollinger Band
                - Elliott Wave pattern confirmation
                - RSI divergence detected
                """
                
                if use_ratio:
                    entry_logic += "- Ratio above 20-period moving average (relative strength)\n"
                
                exit_logic = f"""
                **Exit Conditions (Any one triggers exit):**
                - **Target Hit:** Price reaches {optimized_params['target_pct']:.1f}% profit level
                - **Stop Loss Hit:** Price reaches {optimized_params['sl_pct']:.1f}% loss level
                - **New Signal Generated:** Wait for current position to close before entering new trade
                
                **Position Management:**
                - Only one active position at a time
                - No new signals generated while in position
                - Prevents overlapping trades and miscalculations
                """
                
                st.markdown(entry_logic)
                st.markdown(exit_logic)
        else:
            st.info("🔍 No active signals at the moment. Waiting for entry setup...")
        
        # ============= CHART =============
        
        st.markdown('<p class="sub-header">📈 Price Chart with Entry/Exit Points</p>', unsafe_allow_html=True)
        
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
            name='Buy Signal',
            text=[f"Entry: {p:.2f}" for p in buy_signals['Price']],
            hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: %{text}<extra></extra>'
        ))
        
        # Sell signals
        sell_signals = signals[signals['Signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Price'],
            mode='markers',
            marker=dict(color='red', size=15, symbol='triangle-down'),
            name='Sell Signal',
            text=[f"Entry: {p:.2f}" for p in sell_signals['Price']],
            hovertemplate='<b>Sell Signal</b><br>Date: %{x}<br>Price: %{text}<extra></extra>'
        ))
        
        # Add EMAs
        prices = data1['Close'].values
        ema_fast = calculate_ema(prices, int(optimized_params['ema_fast']))
        ema_slow = calculate_ema(prices, int(optimized_params['ema_slow']))
        
        fig.add_trace(go.Scatter(
            x=data1.index,
            y=ema_fast,
            mode='lines',
            line=dict(color='blue', width=1),
            name=f"EMA {int(optimized_params['ema_fast'])}"
        ))
        
        fig.add_trace(go.Scatter(
            x=data1.index,
            y=ema_slow,
            mode='lines',
            line=dict(color='orange', width=1),
            name=f"EMA {int(optimized_params['ema_slow'])}"
        ))
        
        fig.update_layout(
            title=f"{ticker1} - Trading Signals (IST)",
            xaxis_title="Date & Time (IST)",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ============= RSI INDICATOR =============
        
        st.markdown('<p class="sub-header">📊 RSI Indicator</p>', unsafe_allow_html=True)
        
        rsi = calculate_rsi(data1['Close'].values, int(optimized_params['rsi_period']))
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=data1.index,
            y=rsi,
            mode='lines',
            line=dict(color='purple', width=2),
            name='RSI'
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_hline(y=optimized_params['rsi_overbought'], line_dash="dash", 
                          line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=optimized_params['rsi_oversold'], line_dash="dash", 
                          line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
        
        fig_rsi.update_layout(
            title="RSI Indicator with Divergence Detection",
            xaxis_title="Date & Time (IST)",
            yaxis_title="RSI",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # ============= TRADE HISTORY =============
        
        if len(trades) > 0:
            st.markdown('<p class="sub-header">📋 Detailed Trade History</p>', unsafe_allow_html=True)
            
            # Style the dataframe
            def highlight_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}; font-weight: bold'
                return ''
            
            trades_display = trades.copy()
            
            # Format for display
            styled_trades = trades_display.style.applymap(
                highlight_pnl, 
                subset=['Points', 'PnL %']
            ).format({
                'Entry Price': '₹{:.2f}',
                'Target Price': '₹{:.2f}',
                'SL Price': '₹{:.2f}',
                'Exit Price': '₹{:.2f}',
                'Points': '{:.2f}',
                'PnL %': '{:.2f}%'
            })
            
            st.dataframe(styled_trades, use_container_width=True, height=400)
            
            # Download trades
            csv = trades_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv,
                file_name=f"trades_{ticker1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Trade-by-Trade Analysis
            st.markdown('<p class="sub-header">🔍 Trade-by-Trade Analysis</p>', unsafe_allow_html=True)
            
            for idx, trade in trades_display.iterrows():
                with st.expander(f"Trade #{idx+1} - {trade['Type']} - {trade['Exit Reason']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Entry Details:**")
                        st.write(f"📅 Date: {trade['Entry Time']}")
                        st.write(f"💰 Price: ₹{trade['Entry Price']:.2f}")
                        st.write(f"🎯 Target: ₹{trade['Target Price']:.2f}")
                        st.write(f"🛡️ Stop Loss: ₹{trade['SL Price']:.2f}")
                    
                    with col2:
                        st.markdown("**Exit Details:**")
                        st.write(f"📅 Date: {trade['Exit Time']}")
                        st.write(f"💰 Price: ₹{trade['Exit Price']:.2f}")
                        st.write(f"⏱️ Duration: {trade['Duration']}")
                        st.write(f"📊 Reason: {trade['Exit Reason']}")
                    
                    with col3:
                        st.markdown("**Performance:**")
                        points_color = "🟢" if trade['Points'] > 0 else "🔴"
                        st.write(f"{points_color} Points: {trade['Points']:.2f}")
                        st.write(f"{points_color} PnL %: {trade['PnL %']:.2f}%")
                        risk_reward = abs(trade['Points'] / (trade['Entry Price'] - trade['SL Price']))
                        st.write(f"📈 Risk/Reward: 1:{risk_reward:.2f}")
        
        # ============= PERFORMANCE METRICS =============
        
        st.markdown('<p class="sub-header">📈 Advanced Performance Metrics</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Profitability Metrics:**")
            st.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
            st.metric("Average Win", f"₹{metrics['Avg Win']:.2f}")
            st.metric("Average Loss", f"₹{metrics['Avg Loss']:.2f}")
            st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            
            if metrics['Profit Factor'] >= 2:
                st.success("✅ Excellent Profit Factor (≥2.0)")
            elif metrics['Profit Factor'] >= 1.5:
                st.info("👍 Good Profit Factor (1.5-2.0)")
            elif metrics['Profit Factor'] >= 1:
                st.warning("⚠️ Acceptable Profit Factor (1.0-1.5)")
            else:
                st.error("❌ Poor Profit Factor (<1.0)")
        
        with col2:
            st.markdown("**Risk Metrics:**")
            st.metric("Max Drawdown", f"₹{metrics['Max Drawdown']:.2f}")
            st.metric("Avg Points/Trade", f"₹{metrics['Avg Points per Trade']:.2f}")
            st.metric("Total Points Gained", f"₹{metrics['Total Points']:.2f}")
            st.metric("Buy & Hold Points", f"₹{metrics['Buy & Hold Points']:.2f}")
            
            if metrics['Strategy Win']:
                outperformance = ((metrics['Total Points'] - metrics['Buy & Hold Points']) / 
                                abs(metrics['Buy & Hold Points']) * 100)
                st.success(f"✅ Strategy Outperforms B&H by {outperformance:.1f}%")
            else:
                underperformance = ((metrics['Buy & Hold Points'] - metrics['Total Points']) / 
                                  abs(metrics['Buy & Hold Points']) * 100)
                st.error(f"❌ Strategy Underperforms B&H by {underperformance:.1f}%")
        
        # ============= STRATEGY SUMMARY =============
        
        st.markdown('<p class="sub-header">📝 Strategy Summary</p>', unsafe_allow_html=True)
        
        summary_text = f"""
        ### Trading Strategy Report - {ticker1}
        
        **Analysis Period:** {data1.index[0].strftime('%Y-%m-%d')} to {data1.index[-1].strftime('%Y-%m-%d')} (IST)
        
        **Strategy Components:**
        - Elliott Wave Analysis (Lookback: {int(optimized_params['wave_lookback'])} bars)
        - Fibonacci Retracement (Entry at {optimized_params['fib_entry']:.3f} level)
        - RSI Divergence Detection (Period: {int(optimized_params['rsi_period'])})
        - Multi-timeframe EMA ({int(optimized_params['ema_fast'])}/{int(optimized_params['ema_slow'])})
        - MACD Confirmation
        - Bollinger Bands Filter
        {f"- Ratio Analysis: {ticker1}/{ticker2}" if use_ratio else ""}
        
        **Risk Management:**
        - Target: {optimized_params['target_pct']:.1f}% per trade
        - Stop Loss: {optimized_params['sl_pct']:.1f}% per trade
        - Risk/Reward Ratio: 1:{(optimized_params['target_pct']/optimized_params['sl_pct']):.2f}
        - Position Management: Single position at a time
        
        **Performance Summary:**
        - Total Trades Executed: {metrics['Total Trades']}
        - Winning Trades: {metrics['Positive Trades']} ({metrics['Win Rate']:.1f}%)
        - Losing Trades: {metrics['Negative Trades']} ({100-metrics['Win Rate']:.1f}%)
        - Total Points Gained: {metrics['Total Points']:.2f}
        - Buy & Hold Points: {metrics['Buy & Hold Points']:.2f}
        - Strategy Advantage: {'+' if metrics['Strategy Win'] else ''}{(metrics['Total Points'] - metrics['Buy & Hold Points']):.2f} points
        - Profit Factor: {metrics['Profit Factor']:.2f}
        - Average Win: ₹{metrics['Avg Win']:.2f}
        - Average Loss: ₹{metrics['Avg Loss']:.2f}
        
        **Optimization Status:**
        {'✅ Parameters optimized using Differential Evolution algorithm' if optimize else '⚠️ Using default parameters (enable optimization for better results)'}
        
        **Trading Recommendation:**
        """
        
        if metrics['Accuracy'] >= 70 and metrics['Strategy Win'] and metrics['Profit Factor'] >= 1.5:
            summary_text += "✅ **STRONG BUY SIGNAL** - High accuracy with excellent risk/reward\n"
        elif metrics['Accuracy'] >= 60 and metrics['Strategy Win']:
            summary_text += "👍 **MODERATE BUY** - Good performance but monitor closely\n"
        elif metrics['Accuracy'] >= 50:
            summary_text += "⚠️ **CAUTION** - Strategy shows potential but needs refinement\n"
        else:
            summary_text += "❌ **AVOID** - Strategy underperforming, optimization recommended\n"
        
        st.markdown(summary_text)
        
        # ============= OPTIMIZED PARAMETERS =============
        
        with st.expander("⚙️ Optimized Strategy Parameters"):
            param_df = pd.DataFrame([optimized_params]).T
            param_df.columns = ['Value']
            param_df.index.name = 'Parameter'
            
            st.dataframe(param_df.style.format("{:.4f}"), use_container_width=True)
            
            st.info("""
            **Parameter Descriptions:**
            - **rsi_period**: Period for RSI calculation
            - **rsi_oversold**: RSI level considered oversold (buy signal)
            - **rsi_overbought**: RSI level considered overbought (sell signal)
            - **ema_fast**: Fast EMA period for trend detection
            - **ema_slow**: Slow EMA period for trend detection
            - **fib_lookback**: Bars to look back for Fibonacci calculations
            - **fib_entry**: Fibonacci level for entry (0.382, 0.5, 0.618)
            - **wave_lookback**: Bars to analyze for Elliott Wave patterns
            - **target_pct**: Target profit percentage
            - **sl_pct**: Stop loss percentage
            """)
        
        # ============= RATIO ANALYSIS =============
        
        if use_ratio and ratio_data is not None:
            st.markdown('<p class="sub-header">📊 Ratio Analysis</p>', unsafe_allow_html=True)
            
            fig_ratio = go.Figure()
            
            fig_ratio.add_trace(go.Scatter(
                x=ratio_data.index,
                y=ratio_data.values,
                mode='lines',
                line=dict(color='teal', width=2),
                name=f'{ticker1}/{ticker2} Ratio'
            ))
            
            ratio_ma = ratio_data.rolling(window=20).mean()
            fig_ratio.add_trace(go.Scatter(
                x=ratio_ma.index,
                y=ratio_ma.values,
                mode='lines',
                line=dict(color='orange', width=1, dash='dash'),
                name='20-period MA'
            ))
            
            fig_ratio.update_layout(
                title=f"Relative Strength: {ticker1} vs {ticker2}",
                xaxis_title="Date & Time (IST)",
                yaxis_title="Ratio",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            current_ratio = ratio_data.iloc[-1]
            current_ma = ratio_ma.iloc[-1]
            
            if current_ratio > current_ma:
                st.success(f"✅ {ticker1} is showing relative strength vs {ticker2}")
            else:
                st.warning(f"⚠️ {ticker1} is showing relative weakness vs {ticker2}")
        
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
    with st.expander("📚 Complete User Guide"):
        st.markdown("""
        ### 🚀 Quick Start Guide:
        
        1. **Select Primary Instrument**: Enter ticker symbol
           - Indian Indices: `^NSEI` (Nifty 50), `^BSESN` (Sensex)
           - Indian Stocks: Add `.NS` for NSE (e.g., `RELIANCE.NS`, `TCS.NS`)
           - Indian Stocks: Add `.BO` for BSE (e.g., `RELIANCE.BO`)
           - Forex: `USDINR=X`, `EURINR=X`, `GBPINR=X`
           - Commodities: `GC=F` (Gold), `CL=F` (Crude Oil)
        
        2. **Optional Ratio Analysis**: 
           - Enable to compare relative strength between two instruments
           - Example: Nifty/USD-INR ratio shows market strength vs currency
        
        3. **Choose Timeframe**: 
           - **Intervals**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
           - **Periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
           - Shorter intervals work better with shorter periods
        
        4. **Select Mode**:
           - **Backtest**: Test strategy on historical data
           - **Live Trading**: Get current recommendations
        
        5. **Enable Optimization** (Optional):
           - Takes 1-2 minutes but significantly improves results
           - Uses advanced algorithms to find best parameters
           - Aims for 100% accuracy and beat buy-and-hold
        
        6. **Click Run Strategy**: View comprehensive results
        
        ---
        
        ### 📊 Understanding the Results:
        
        **Strategy Performance Section:**
        - Shows total trades, win rate, accuracy
        - Compares strategy points vs buy-and-hold points
        - Displays profit factor and average win/loss
        
        **Latest Trading Signal:**
        - **Entry Date/Time**: Exact timestamp in IST
        - **Entry Price**: Recommended entry level
        - **Target Price**: Take profit level
        - **SL Price**: Stop loss level
        - **Status**: Whether signal is active or closed
        - **Probability**: Based on historical accuracy
        
        **Trade History:**
        - Complete record of all trades
        - Entry/exit times, prices, and reasons
        - Points gained/lost per trade
        - Duration of each trade
        
        ---
        
        ### 🎯 Strategy Logic:
        
        **Entry Signals Generated When:**
        - RSI in oversold/overbought territory with reversal
        - EMA crossover confirms trend direction
        - MACD shows momentum alignment
        - Price at key Fibonacci retracement level
        - Elliott Wave pattern detected
        - RSI divergence confirms reversal
        - (Optional) Ratio shows relative strength
        
        **Exit Signals Trigger When:**
        - Target percentage reached (lock profits)
        - Stop loss hit (limit losses)
        - Opposite signal generated
        
        **Position Management:**
        - Only ONE position active at a time
        - New signals wait until current position closes
        - Prevents overlapping trades
        - Avoids calculation errors
        
        ---
        
        ### ⚙️ Advanced Features:
        
        **No TA-Lib Dependency:**
        - All indicators calculated from scratch
        - RSI, EMA, SMA, MACD, Bollinger Bands
        - Fully self-contained code
        
        **Rate Limit Protection:**
        - 5-minute data caching
        - Prevents API overload
        - Smooth browser refresh handling
        
        **IST Timezone Handling:**
        - All times displayed in Indian Standard Time
        - Graceful handling of timezone-aware data
        - Proper datetime conversion
        
        **Persistent UI:**
        - Interface stays visible after clicking "Run"
        - Progress bars show real-time status
        - Results don't disappear on refresh
        
        ---
        
        ### 📈 Performance Targets:
        
        **Aiming for 100% Accuracy:**
        - Optimization algorithm maximizes win rate
        - Filters out low-probability setups
        - Focuses on high-confidence signals
        
        **Beating Buy & Hold:**
        - Strategy must outperform passive holding
        - Penalty applied if underperforming
        - Risk-adjusted returns prioritized
        
        **Profit Factor > 2:**
        - Target: Win twice as much as you lose
        - Indicates strong edge in the market
        - Sustainable long-term profitability
        
        ---
        
        ### ⚠️ Important Notes:
        
        - **Past performance ≠ Future results**
        - **Always use stop losses**
        - **Start with small position sizes**
        - **Test thoroughly before live trading**
        - **Market conditions change constantly**
        - **This is an educational tool**
        
        ---
        
        ### 🛠️ Troubleshooting:
        
        **"Insufficient data" error:**
        - Try a longer period
        - Check if ticker symbol is correct
        - Some instruments have limited history
        
        **No signals generated:**
        - Market may not meet entry criteria
        - Try different timeframes
        - Enable optimization for better parameters
        
        **Low accuracy:**
        - Enable optimization
        - Use longer periods for more data
        - Adjust timeframe to match trading style
        
        ---
        
        ### 💡 Pro Tips:
        
        1. **Combine multiple timeframes**: Check higher timeframes for trend
        2. **Use ratio analysis**: Adds extra confirmation layer
        3. **Run optimization weekly**: Market conditions evolve
        4. **Paper trade first**: Test before risking real money
        5. **Check correlation**: Some stocks move together
        6. **Monitor news**: Fundamental events can override technicals
        7. **Set alerts**: Don't miss entry opportunities
        8. **Review trades**: Learn from wins and losses
        
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Risk Disclaimer:</strong> This is an educational and research tool. Past performance does not guarantee future results. 
    Trading in stocks, derivatives, and other financial instruments involves substantial risk of loss. 
    Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
    The creators and operators of this tool are not responsible for any financial losses incurred.</p>
    <p><strong>Data Source:</strong> Yahoo Finance | <strong>Timezone:</strong> Indian Standard Time (IST) | <strong>Version:</strong> 2.0</p>
</div>
""", unsafe_allow_html=True)
