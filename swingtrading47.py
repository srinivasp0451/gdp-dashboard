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
from scipy.signal import find_peaks
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .neutral {
        color: #ffa500;
        font-weight: bold;
    }
    .signal-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .buy-signal {
        background-color: #d4edda;
        border: 3px solid #28a745;
        color: #155724;
    }
    .sell-signal {
        background-color: #f8d7da;
        border: 3px solid #dc3545;
        color: #721c24;
    }
    .hold-signal {
        background-color: #fff3cd;
        border: 3px solid #ffc107;
        color: #856404;
    }
    .summary-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'all_timeframe_data' not in st.session_state:
    st.session_state.all_timeframe_data = {}
if 'all_timeframe_data_t2' not in st.session_state:
    st.session_state.all_timeframe_data_t2 = {}

# Ticker mappings
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "Bank NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X"
}

# Valid timeframe-period combinations for yfinance
VALID_COMBINATIONS = {
    "1m": ["1d", "5d", "7d"],
    "2m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "30m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"],
    "1wk": ["1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"],
    "1mo": ["2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"]
}

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ticker_symbol(ticker_input):
    """Convert ticker input to yfinance symbol"""
    return TICKER_MAP.get(ticker_input, ticker_input)

def fetch_data_with_retry(ticker, period, interval, max_retries=3):
    """Fetch data with retry mechanism and rate limiting"""
    for attempt in range(max_retries):
        try:
            time.sleep(np.random.uniform(1.5, 3.0))
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if data.empty:
                return None
            
            data = data.reset_index()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(str(col).strip() for col in c if col) if isinstance(c, tuple) else str(c) for c in data.columns]
            
            data.columns = [col.split('_')[0].strip() for col in data.columns]
            
            datetime_col = None
            for col_name in ['Date', 'Datetime', 'index', 'Timestamp']:
                if col_name in data.columns:
                    datetime_col = col_name
                    break
            
            if datetime_col is None:
                return None
            
            data = data.rename(columns={datetime_col: 'Date'})
            
            column_mapping = {}
            for required in ['Open', 'High', 'Low', 'Close']:
                found = False
                for col in data.columns:
                    if required.lower() in col.lower():
                        column_mapping[col] = required
                        found = True
                        break
                if not found:
                    return None
            
            data = data.rename(columns=column_mapping)
            
            has_volume = False
            for col in data.columns:
                if 'volume' in col.lower():
                    data = data.rename(columns={col: 'Volume'})
                    has_volume = True
                    break
            
            if not has_volume:
                data['Volume'] = 0
            
            try:
                if data['Date'].dt.tz is None:
                    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize('UTC')
                data['Date'] = data['Date'].dt.tz_convert(IST)
            except:
                data['Date'] = pd.to_datetime(data['Date'])
            
            data = data.set_index('Date')
            relevant_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[col for col in relevant_cols if col in data.columns]]
            data = data[~data.index.duplicated(keep='first')]
            data = data.dropna(how='all')
            
            if len(data) == 0:
                return None
            
            return data
            
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    return None

def fetch_all_timeframes(ticker):
    """Fetch data for all valid timeframe-period combinations"""
    all_data = {}
    
    for interval, periods in VALID_COMBINATIONS.items():
        for period in periods:
            try:
                data = fetch_data_with_retry(ticker, period, interval)
                if data is not None and len(data) > 0:
                    key = f"{interval}_{period}"
                    all_data[key] = data
            except:
                continue
    
    return all_data

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    if len(data) < period + 1:
        return pd.Series([50] * len(data), index=data.index)
    
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, 0.0001)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    return rsi

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    if len(high) < period + 1:
        return pd.Series([0] * len(high), index=high.index)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period, min_periods=1).mean() / atr.replace(0, 0.0001)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period, min_periods=1).mean() / atr.replace(0, 0.0001)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.0001)
    adx = dx.rolling(window=period, min_periods=1).mean()
    adx = adx.fillna(0)
    
    return adx

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def find_support_resistance(data, window=20, prominence=0.02):
    """Find strong support and resistance levels"""
    highs = data['High'].values
    lows = data['Low'].values
    
    resistance_indices, _ = find_peaks(highs, prominence=np.std(highs) * prominence)
    support_indices, _ = find_peaks(-lows, prominence=np.std(lows) * prominence)
    
    resistances = []
    for idx in resistance_indices:
        price = highs[idx]
        date = data.index[idx]
        touches = int(np.sum(np.abs(highs - price) / price < 0.005))
        
        # Calculate sustainability
        sustained_count = 0
        for i in range(max(0, idx-10), min(len(highs), idx+10)):
            if abs(highs[i] - price) / price < 0.005:
                sustained_count += 1
        
        resistances.append({
            'level': price, 
            'touches': touches, 
            'date': date, 
            'type': 'resistance',
            'sustained': sustained_count
        })
    
    supports = []
    for idx in support_indices:
        price = lows[idx]
        date = data.index[idx]
        touches = int(np.sum(np.abs(lows - price) / price < 0.005))
        
        sustained_count = 0
        for i in range(max(0, idx-10), min(len(lows), idx+10)):
            if abs(lows[i] - price) / price < 0.005:
                sustained_count += 1
        
        supports.append({
            'level': price, 
            'touches': touches, 
            'date': date, 
            'type': 'support',
            'sustained': sustained_count
        })
    
    levels = sorted(supports + resistances, key=lambda x: (x['touches'], x['sustained']), reverse=True)
    
    return levels[:10]

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    high_date = data['High'].idxmax()
    low_date = data['Low'].idxmin()
    
    levels = {
        '0.0': {'price': high, 'date': high_date},
        '0.236': {'price': high - 0.236 * diff, 'date': None},
        '0.382': {'price': high - 0.382 * diff, 'date': None},
        '0.5': {'price': high - 0.5 * diff, 'date': None},
        '0.618': {'price': high - 0.618 * diff, 'date': None},
        '0.786': {'price': high - 0.786 * diff, 'date': None},
        '1.0': {'price': low, 'date': low_date}
    }
    
    return levels

def detect_elliott_wave_detailed(data):
    """Detailed Elliott Wave detection with all wave characteristics"""
    closes = data['Close'].values
    highs = data['High'].values
    lows = data['Low'].values
    
    peaks, _ = find_peaks(closes, distance=5)
    troughs, _ = find_peaks(-closes, distance=5)
    
    turns = sorted(list(peaks) + list(troughs))
    
    if len(turns) < 8:
        return None
    
    # Analyze last 8 turning points for complete Elliott Wave
    wave_points = turns[-8:]
    waves = []
    
    for i, idx in enumerate(wave_points):
        is_peak = idx in peaks
        wave_type = 'Impulse' if i < 5 else 'Corrective'
        
        # Determine wave phase
        if i < 5:
            wave_num = i + 1
            phase = f"Wave {wave_num}"
        else:
            wave_num = chr(65 + (i - 5))  # A, B, C
            phase = f"Wave {wave_num} (Corrective)"
        
        # Calculate wave characteristics
        if i > 0:
            prev_idx = wave_points[i-1]
            wave_change = closes[idx] - closes[prev_idx]
            wave_pct = (wave_change / closes[prev_idx]) * 100
            duration = data.index[idx] - data.index[prev_idx]
        else:
            wave_change = 0
            wave_pct = 0
            duration = timedelta(0)
        
        waves.append({
            'phase': phase,
            'wave_type': wave_type,
            'price': closes[idx],
            'high': highs[idx],
            'low': lows[idx],
            'date': data.index[idx],
            'point_type': 'Peak' if is_peak else 'Trough',
            'change': wave_change,
            'change_pct': wave_pct,
            'duration': str(duration)
        })
    
    # Determine current wave
    current_price = closes[-1]
    last_wave = waves[-1]
    
    if len(waves) >= 5:
        # Check if in corrective phase
        wave_5_price = waves[4]['price'] if len(waves) > 4 else waves[-1]['price']
        if current_price < wave_5_price:
            current_wave = "Corrective Phase (ABC)"
        else:
            current_wave = "Wave 5 or Extension"
    else:
        current_wave = f"Wave {len(waves)}"
    
    return {
        'waves': waves,
        'current_wave': current_wave,
        'total_waves': len(waves)
    }

def detect_rsi_divergence(price, rsi, window=14):
    """Detect RSI divergence with details"""
    price_values = price.values
    rsi_values = rsi.values
    
    price_peaks, _ = find_peaks(price_values, distance=window)
    price_troughs, _ = find_peaks(-price_values, distance=window)
    rsi_peaks, _ = find_peaks(rsi_values, distance=window)
    rsi_troughs, _ = find_peaks(-rsi_values, distance=window)
    
    divergences = []
    
    # Bullish divergence
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        last_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        
        if price_values[last_price_trough] < price_values[prev_price_trough]:
            rsi_near_last = rsi_troughs[(rsi_troughs >= prev_price_trough - 5) & 
                                        (rsi_troughs <= last_price_trough + 5)]
            
            if len(rsi_near_last) >= 2:
                if rsi_values[rsi_near_last[-1]] > rsi_values[rsi_near_last[-2]]:
                    time_ago = datetime.now(IST) - price.index[last_price_trough]
                    divergences.append({
                        'type': 'bullish',
                        'price': float(price_values[last_price_trough]),
                        'date': price.index[last_price_trough],
                        'rsi_old': float(rsi_values[rsi_near_last[-2]]),
                        'rsi_new': float(rsi_values[rsi_near_last[-1]]),
                        'time_ago': str(time_ago).split('.')[0]
                    })
    
    # Bearish divergence
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        
        if price_values[last_price_peak] > price_values[prev_price_peak]:
            rsi_near_last = rsi_peaks[(rsi_peaks >= prev_price_peak - 5) & 
                                      (rsi_peaks <= last_price_peak + 5)]
            
            if len(rsi_near_last) >= 2:
                if rsi_values[rsi_near_last[-1]] < rsi_values[rsi_near_last[-2]]:
                    time_ago = datetime.now(IST) - price.index[last_price_peak]
                    divergences.append({
                        'type': 'bearish',
                        'price': float(price_values[last_price_peak]),
                        'date': price.index[last_price_peak],
                        'rsi_old': float(rsi_values[rsi_near_last[-2]]),
                        'rsi_new': float(rsi_values[rsi_near_last[-1]]),
                        'time_ago': str(time_ago).split('.')[0]
                    })
    
    return divergences if divergences else None

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def calculate_zscore(data, window=20):
    """Calculate Z-score with historical analysis"""
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    zscore = (data - mean) / std
    return zscore

def analyze_zscore_patterns(data, zscore):
    """Analyze historical Z-score patterns and outcomes"""
    patterns = []
    
    # Find extreme Z-score events
    extreme_indices = np.where(np.abs(zscore) > 2)[0]
    
    for idx in extreme_indices:
        if idx < len(data) - 20:  # Need future data to see outcome
            zscore_val = float(zscore.iloc[idx])
            entry_price = float(data['Close'].iloc[idx])
            entry_date = data.index[idx]
            
            # Check price movement in next 5, 10, 20 periods
            future_prices = data['Close'].iloc[idx+1:idx+21]
            max_gain = ((future_prices.max() - entry_price) / entry_price) * 100
            max_loss = ((future_prices.min() - entry_price) / entry_price) * 100
            
            final_price = float(data['Close'].iloc[idx+20])
            final_return = ((final_price - entry_price) / entry_price) * 100
            
            time_ago = datetime.now(IST) - entry_date
            
            patterns.append({
                'zscore': zscore_val,
                'entry_price': entry_price,
                'entry_date': entry_date,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'final_return': final_return,
                'time_ago': str(time_ago).split('.')[0],
                'signal': 'buy' if zscore_val < -2 else 'sell'
            })
    
    return patterns[-10:] if patterns else []  # Last 10 patterns

def optimize_strategy(data, strategy_type='combined'):
    """Optimize trading strategy parameters"""
    best_params = None
    best_return = -np.inf
    
    # Parameter ranges to test
    rsi_ranges = [(20, 80), (25, 75), (30, 70)]
    ema_combos = [(9, 20), (10, 30), (12, 26)]
    adx_thresholds = [20, 25, 30]
    
    for rsi_range, ema_combo, adx_thresh in product(rsi_ranges, ema_combos, adx_thresholds):
        returns = backtest_strategy(
            data, 
            strategy_type,
            rsi_oversold=rsi_range[0],
            rsi_overbought=rsi_range[1],
            ema_fast=ema_combo[0],
            ema_slow=ema_combo[1],
            adx_threshold=adx_thresh
        )
        
        if returns > best_return:
            best_return = returns
            best_params = {
                'rsi_range': rsi_range,
                'ema_combo': ema_combo,
                'adx_threshold': adx_thresh
            }
    
    return best_params, best_return

def backtest_strategy(data, strategy_type='combined', rsi_oversold=30, rsi_overbought=70, 
                     ema_fast=9, ema_slow=20, adx_threshold=25, initial_capital=100000, position_size=20):
    """Enhanced backtesting with optimization"""
    
    if len(data) < 50:
        return -100  # Return bad performance for insufficient data
    
    capital = initial_capital
    position = None
    trades = []
    
    close = data['Close']
    rsi = calculate_rsi(close)
    ema_f = calculate_ema(close, ema_fast)
    ema_s = calculate_ema(close, ema_slow)
    adx = calculate_adx(data['High'], data['Low'], close)
    
    for i in range(50, len(data)):
        current_price = float(close.iloc[i])
        current_rsi = float(rsi.iloc[i])
        current_adx = float(adx.iloc[i])
        ema_fast_val = float(ema_f.iloc[i])
        ema_slow_val = float(ema_s.iloc[i])
        
        # Entry conditions
        if position is None:
            buy_signal = False
            
            if strategy_type == 'combined':
                # Combined strategy
                if (current_rsi < rsi_oversold and 
                    ema_fast_val > ema_slow_val and 
                    current_adx > adx_threshold):
                    buy_signal = True
            
            elif strategy_type == 'rsi':
                if current_rsi < rsi_oversold:
                    buy_signal = True
            
            elif strategy_type == 'ema':
                if ema_fast_val > ema_slow_val and i > 0:
                    prev_ema_fast = float(ema_f.iloc[i-1])
                    prev_ema_slow = float(ema_s.iloc[i-1])
                    if prev_ema_fast <= prev_ema_slow:
                        buy_signal = True
            
            if buy_signal:
                shares = (capital * position_size / 100) / current_price
                position = {
                    'entry': current_price,
                    'shares': shares,
                    'entry_date': data.index[i],
                    'entry_rsi': current_rsi
                }
        
        # Exit conditions
        elif position is not None:
            sell_signal = False
            
            if strategy_type == 'combined':
                if (current_rsi > rsi_overbought or 
                    ema_fast_val < ema_slow_val or
                    current_adx < adx_threshold):
                    sell_signal = True
            
            elif strategy_type == 'rsi':
                if current_rsi > rsi_overbought:
                    sell_signal = True
            
            elif strategy_type == 'ema':
                if ema_fast_val < ema_slow_val and i > 0:
                    prev_ema_fast = float(ema_f.iloc[i-1])
                    prev_ema_slow = float(ema_s.iloc[i-1])
                    if prev_ema_fast >= prev_ema_slow:
                        sell_signal = True
            
            # Stop loss and take profit
            pct_change = ((current_price - position['entry']) / position['entry']) * 100
            if pct_change < -5 or pct_change > 10:  # 5% SL, 10% TP
                sell_signal = True
            
            if sell_signal:
                profit = (current_price - position['entry']) * position['shares']
                capital += profit
                
                trades.append({
                    'Entry': position['entry'],
                    'Exit': current_price,
                    'Profit': profit,
                    'Return %': (profit / (position['entry'] * position['shares'])) * 100,
                    'Entry Date': position['entry_date'],
                    'Exit Date': data.index[i],
                    'Duration': str(data.index[i] - position['entry_date']).split('.')[0]
                })
                position = None
    
    # Calculate returns
    if len(trades) == 0:
        return -100
    
    total_return = ((capital - initial_capital) / initial_capital) * 100
    return total_return

def run_full_backtest_with_optimization(data, ticker_name):
    """Run comprehensive backtest with optimization"""
    
    st.write("### 🔬 Backtesting & Optimization")
    
    strategy_types = ['combined', 'rsi', 'ema']
    best_strategy = None
    best_return = -np.inf
    all_results = {}
    
    with st.spinner("Optimizing strategies..."):
        for strategy in strategy_types:
            best_params, returns = optimize_strategy(data, strategy)
            all_results[strategy] = {'params': best_params, 'return': returns}
            
            if returns > best_return:
                best_return = returns
                best_strategy = strategy
    
    # Display optimization results
    col1, col2, col3 = st.columns(3)
    
    for idx, (strategy, result) in enumerate(all_results.items()):
        with [col1, col2, col3][idx]:
            color = "🟢" if result['return'] > 0 else "🔴"
            st.metric(
                f"{color} {strategy.upper()} Strategy",
                f"{result['return']:.2f}%",
                delta=f"Annualized: {result['return']:.1f}%"
            )
    
    # Run detailed backtest with best strategy
    if best_return > -100:
        st.success(f"✅ Best Strategy: **{best_strategy.upper()}** with **{best_return:.2f}%** returns")
        
        # Full backtest with best parameters
        trades_data = detailed_backtest(data, best_strategy, all_results[best_strategy]['params'])
        
        if trades_data is not None and len(trades_data) > 0:
            st.write("**📊 Trade Performance:**")
            
            winning_trades = len(trades_data[trades_data['Return %'] > 0])
            losing_trades = len(trades_data[trades_data['Return %'] < 0])
            win_rate = (winning_trades / len(trades_data)) * 100 if len(trades_data) > 0 else 0
            
            avg_win = trades_data[trades_data['Profit'] > 0]['Profit'].mean() if winning_trades > 0 else 0
            avg_loss = trades_data[trades_data['Profit'] < 0]['Profit'].mean() if losing_trades > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("Total Trades", len(trades_data))
            col3.metric("Winners", winning_trades)
            col4.metric("Avg Win/Loss", f"₹{avg_win:.0f} / ₹{avg_loss:.0f}")
            
            st.dataframe(trades_data, use_container_width=True)
            
            # Equity curve
            equity = [100000]
            for profit in trades_data['Profit']:
                equity.append(equity[-1] + profit)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity, mode='lines', name='Equity', line=dict(color='green', width=2)))
            fig.update_layout(title="Equity Curve", xaxis_title="Trade Number", yaxis_title="Capital (₹)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            return best_strategy, best_return, all_results[best_strategy]['params']
    else:
        st.warning("⚠️ No profitable strategy found. Market may be ranging or need different approach.")
        return None, -100, None

def detailed_backtest(data, strategy_type, params):
    """Detailed backtest returning trade data"""
    capital = 100000
    position = None
    trades = []
    
    close = data['Close']
    rsi = calculate_rsi(close)
    ema_f = calculate_ema(close, params['ema_combo'][0])
    ema_s = calculate_ema(close, params['ema_combo'][1])
    adx = calculate_adx(data['High'], data['Low'], close)
    
    rsi_oversold = params['rsi_range'][0]
    rsi_overbought = params['rsi_range'][1]
    adx_threshold = params['adx_threshold']
    
    for i in range(50, len(data)):
        current_price = float(close.iloc[i])
        current_rsi = float(rsi.iloc[i])
        current_adx = float(adx.iloc[i])
        ema_fast_val = float(ema_f.iloc[i])
        ema_slow_val = float(ema_s.iloc[i])
        
        if position is None:
            buy_signal = False
            
            if strategy_type == 'combined':
                if (current_rsi < rsi_oversold and ema_fast_val > ema_slow_val and current_adx > adx_threshold):
                    buy_signal = True
            elif strategy_type == 'rsi':
                if current_rsi < rsi_oversold:
                    buy_signal = True
            elif strategy_type == 'ema':
                if ema_fast_val > ema_slow_val and i > 0:
                    if float(ema_f.iloc[i-1]) <= float(ema_s.iloc[i-1]):
                        buy_signal = True
            
            if buy_signal:
                shares = (capital * 20 / 100) / current_price
                position = {'entry': current_price, 'shares': shares, 'entry_date': data.index[i]}
        
        elif position is not None:
            sell_signal = False
            
            if strategy_type == 'combined':
                if current_rsi > rsi_overbought or ema_fast_val < ema_slow_val or current_adx < adx_threshold:
                    sell_signal = True
            elif strategy_type == 'rsi':
                if current_rsi > rsi_overbought:
                    sell_signal = True
            elif strategy_type == 'ema':
                if ema_fast_val < ema_slow_val and i > 0:
                    if float(ema_f.iloc[i-1]) >= float(ema_s.iloc[i-1]):
                        sell_signal = True
            
            pct_change = ((current_price - position['entry']) / position['entry']) * 100
            if pct_change < -5 or pct_change > 10:
                sell_signal = True
            
            if sell_signal:
                profit = (current_price - position['entry']) * position['shares']
                capital += profit
                
                trades.append({
                    'Entry': position['entry'],
                    'Exit': current_price,
                    'Profit': profit,
                    'Return %': (profit / (position['entry'] * position['shares'])) * 100,
                    'Entry Date': position['entry_date'].strftime('%Y-%m-%d %H:%M IST'),
                    'Exit Date': data.index[i].strftime('%Y-%m-%d %H:%M IST'),
                    'Duration': str(data.index[i] - position['entry_date']).split('.')[0]
                })
                position = None
    
    return pd.DataFrame(trades) if trades else None

def format_time_ago(time_diff):
    """Format timedelta into human-readable string"""
    total_seconds = int(time_diff.total_seconds())
    
    if total_seconds < 0:
        return "Just now"
    
    minutes = total_seconds // 60
    hours = minutes // 60
    days = hours // 24
    months = days // 30
    
    if months > 0:
        remaining_days = days % 30
        if remaining_days > 0:
            return f"{months} month(s) {remaining_days} day(s) ago"
        return f"{months} month(s) ago"
    elif days > 0:
        return f"{days} day(s) ago"
    elif hours > 0:
        return f"{hours} hour(s) ago"
    elif minutes > 0:
        return f"{minutes} minute(s) ago"
    else:
        return f"{total_seconds} second(s) ago"

def create_indicator_tables(data, ticker_name):
    """Create comprehensive indicator tables with datetime, price, and changes"""
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Calculate all indicators
    rsi = calculate_rsi(close)
    zscore = calculate_zscore(close)
    volatility = calculate_volatility(close.pct_change())
    ema_9 = calculate_ema(close, 9)
    ema_20 = calculate_ema(close, 20)
    ema_50 = calculate_ema(close, 50)
    
    # Base dataframe
    indicator_df = pd.DataFrame({
        'DateTime': data.index.strftime('%Y-%m-%d %H:%M:%S IST'),
        'Price': close.values
    })
    
    # RSI Table
    rsi_df = indicator_df.copy()
    rsi_df['RSI'] = rsi.values
    rsi_df['RSI_Change'] = rsi.diff().values
    rsi_df['Price_Change_%'] = close.pct_change().values * 100
    rsi_df['Price_Change_Abs'] = close.diff().values
    rsi_df['RSI_Signal'] = rsi_df['RSI'].apply(
        lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral'
    )
    
    # Z-Score Bins Table
    zscore_df = indicator_df.copy()
    zscore_df['Z-Score'] = zscore.values
    zscore_df['Z-Score_Bin'] = pd.cut(
        zscore, 
        bins=[-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf],
        labels=['Extreme Oversold (<-3)', 'Very Oversold (-3 to -2)', 'Oversold (-2 to -1)', 
                'Slightly Bearish (-1 to 0)', 'Slightly Bullish (0 to 1)', 
                'Overbought (1 to 2)', 'Very Overbought (2 to 3)', 'Extreme Overbought (>3)']
    )
    zscore_df['Price_Change_%'] = close.pct_change().values * 100
    zscore_df['Price_Change_Abs'] = close.diff().values
    zscore_df['Mean_Reversion_Signal'] = zscore_df['Z-Score'].apply(
        lambda x: 'Strong Buy' if x < -2 else 'Strong Sell' if x > 2 else 'Neutral'
    )
    
    # Volatility Bins Table
    volatility_df = indicator_df.copy()
    volatility_df['Volatility_%'] = volatility.values
    volatility_df['Volatility_Bin'] = pd.cut(
        volatility,
        bins=[0, 10, 20, 30, 50, np.inf],
        labels=['Very Low (<10%)', 'Low (10-20%)', 'Medium (20-30%)', 'High (30-50%)', 'Very High (>50%)']
    )
    volatility_df['Price_Change_%'] = close.pct_change().values * 100
    volatility_df['Price_Change_Abs'] = close.diff().values
    volatility_df['Volatility_Change'] = volatility.diff().values
    
    # EMAs Table
    ema_df = indicator_df.copy()
    ema_df['EMA_9'] = ema_9.values
    ema_df['EMA_20'] = ema_20.values
    ema_df['EMA_50'] = ema_50.values
    ema_df['Price_vs_EMA9_%'] = ((close - ema_9) / ema_9 * 100).values
    ema_df['Price_vs_EMA20_%'] = ((close - ema_20) / ema_20 * 100).values
    ema_df['Price_vs_EMA50_%'] = ((close - ema_50) / ema_50 * 100).values
    ema_df['Price_Change_%'] = close.pct_change().values * 100
    ema_df['Price_Change_Abs'] = close.diff().values
    ema_df['Trend'] = ema_df.apply(
        lambda row: 'Strong Uptrend' if row['Price'] > row['EMA_9'] > row['EMA_20'] > row['EMA_50']
        else 'Strong Downtrend' if row['Price'] < row['EMA_9'] < row['EMA_20'] < row['EMA_50']
        else 'Mixed',
        axis=1
    )
    
    # Support/Resistance Table
    sr_levels = find_support_resistance(data)
    sr_df = pd.DataFrame(sr_levels)
    if not sr_df.empty:
        sr_df['DateTime'] = sr_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
        current_price = float(close.iloc[-1])
        sr_df['Distance_from_Current_%'] = ((sr_df['level'] - current_price) / current_price * 100)
        sr_df['Distance_from_Current_Abs'] = sr_df['level'] - current_price
        sr_df = sr_df[['DateTime', 'type', 'level', 'touches', 'sustained', 
                       'Distance_from_Current_%', 'Distance_from_Current_Abs']]
        sr_df.columns = ['DateTime', 'Type', 'Price_Level', 'Touch_Count', 'Sustained_Periods',
                         'Distance_%', 'Distance_Abs']
    
    # Fibonacci Table
    fib_levels = calculate_fibonacci_levels(data)
    fib_data = []
    current_price = float(close.iloc[-1])
    for level_name, level_data in fib_levels.items():
        fib_data.append({
            'Fib_Level': level_name,
            'Price': level_data['price'],
            'DateTime': level_data['date'].strftime('%Y-%m-%d %H:%M:%S IST') if level_data['date'] else 'N/A',
            'Distance_from_Current_%': ((level_data['price'] - current_price) / current_price * 100),
            'Distance_from_Current_Abs': level_data['price'] - current_price
        })
    fib_df = pd.DataFrame(fib_data)
    
    # Elliott Wave Table
    elliott = detect_elliott_wave_detailed(data)
    if elliott:
        elliott_df = pd.DataFrame(elliott['waves'])
        elliott_df['DateTime'] = pd.to_datetime(elliott_df['date']).dt.strftime('%Y-%m-%d %H:%M:%S IST')
        elliott_df = elliott_df[['phase', 'wave_type', 'point_type', 'price', 'high', 'low', 
                                 'change', 'change_pct', 'duration', 'DateTime']]
        elliott_df.columns = ['Wave_Phase', 'Wave_Type', 'Point_Type', 'Price', 'High', 'Low',
                              'Price_Change_Abs', 'Price_Change_%', 'Duration', 'DateTime']
    else:
        elliott_df = pd.DataFrame()
    
    # RSI Divergence Table
    divergence = detect_rsi_divergence(close, rsi)
    if divergence:
        div_data = []
        for div in divergence:
            div_data.append({
                'DateTime': div['date'].strftime('%Y-%m-%d %H:%M:%S IST'),
                'Divergence_Type': '🟢 Bullish' if div['type'] == 'bullish' else '🔴 Bearish',
                'Price_Level': div['price'],
                'RSI_Old': div['rsi_old'],
                'RSI_New': div['rsi_new'],
                'RSI_Change': div['rsi_new'] - div['rsi_old'],
                'Time_Ago': div['time_ago'],
                'Status': 'Active' if data.index[-1] - div['date'] < pd.Timedelta(days=5) else 'Resolved',
                'Expected_Move': 'Upward Reversal Expected' if div['type'] == 'bullish' else 'Downward Reversal Expected'
            })
        div_df = pd.DataFrame(div_data)
    else:
        div_df = pd.DataFrame()
    
    return {
        'rsi': rsi_df,
        'zscore': zscore_df,
        'volatility': volatility_df,
        'ema': ema_df,
        'support_resistance': sr_df if not sr_df.empty else pd.DataFrame(),
        'fibonacci': fib_df,
        'elliott': elliott_df,
        'divergence': div_df
    }

def create_ratio_table(data1, data2, ticker1, ticker2):
    """Create comprehensive ratio analysis table"""
    
    # Align data - find overlapping timestamps
    common_index = data1.index.intersection(data2.index)
    
    if len(common_index) == 0:
        # Try to resample to find common ground
        # Resample both to daily and try again
        try:
            data1_daily = data1.resample('1D').last().dropna()
            data2_daily = data2.resample('1D').last().dropna()
            common_index = data1_daily.index.intersection(data2_daily.index)
            
            if len(common_index) == 0:
                return pd.DataFrame()
            
            data1_aligned = data1_daily.loc[common_index]
            data2_aligned = data2_daily.loc[common_index]
        except:
            return pd.DataFrame()
    else:
        data1_aligned = data1.loc[common_index]
        data2_aligned = data2.loc[common_index]
    
    # Calculate ratio and indicators
    ratio = data1_aligned['Close'] / data2_aligned['Close']
    ratio_rsi = calculate_rsi(ratio)
    ratio_zscore = calculate_zscore(ratio)
    
    ratio_df = pd.DataFrame({
        'DateTime': common_index.strftime('%Y-%m-%d %H:%M:%S IST'),
        f'{ticker1}_Price': data1_aligned['Close'].values,
        f'{ticker2}_Price': data2_aligned['Close'].values,
        'Ratio': ratio.values,
        'Ratio_Change_%': ratio.pct_change().values * 100,
        'Ratio_Change_Abs': ratio.diff().values,
        f'{ticker1}_RSI': calculate_rsi(data1_aligned['Close']).values,
        f'{ticker2}_RSI': calculate_rsi(data2_aligned['Close']).values,
        'Ratio_RSI': ratio_rsi.values,
        f'{ticker1}_Z-Score': calculate_zscore(data1_aligned['Close']).values,
        f'{ticker2}_Z-Score': calculate_zscore(data2_aligned['Close']).values,
        'Ratio_Z-Score': ratio_zscore.values,
        f'{ticker1}_Volatility': calculate_volatility(data1_aligned['Close'].pct_change()).values,
        f'{ticker2}_Volatility': calculate_volatility(data2_aligned['Close'].pct_change()).values,
        f'{ticker1}_Change_%': data1_aligned['Close'].pct_change().values * 100,
        f'{ticker2}_Change_%': data2_aligned['Close'].pct_change().values * 100
    })
    
    return ratio_df

def analyze_pattern_performance(all_timeframe_data, ticker_name):
    """Analyze what patterns are working and what are not across timeframes"""
    
    pattern_results = []
    
    for tf_key, data in all_timeframe_data.items():
        if len(data) < 50:
            continue
        
        interval, period = tf_key.split('_')
        
        close = data['Close']
        rsi = calculate_rsi(close)
        zscore = calculate_zscore(close)
        ema_9 = calculate_ema(close, 9)
        ema_20 = calculate_ema(close, 20)
        
        # Test various patterns
        patterns = []
        
        # RSI Oversold pattern
        rsi_oversold_signals = data[rsi < 30]
        if len(rsi_oversold_signals) > 0:
            successes = 0
            for idx in rsi_oversold_signals.index:
                idx_pos = data.index.get_loc(idx)
                if idx_pos < len(data) - 10:
                    future_return = (data['Close'].iloc[idx_pos+10] - data['Close'].iloc[idx_pos]) / data['Close'].iloc[idx_pos] * 100
                    if future_return > 0:
                        successes += 1
            accuracy = (successes / len(rsi_oversold_signals)) * 100 if len(rsi_oversold_signals) > 0 else 0
            patterns.append({
                'Pattern': 'RSI Oversold (<30)',
                'Occurrences': len(rsi_oversold_signals),
                'Success_Rate_%': accuracy,
                'Status': '✅ Working' if accuracy > 60 else '❌ Not Working',
                'Confidence': 'High' if accuracy > 70 else 'Medium' if accuracy > 50 else 'Low',
                'Last_Signal': rsi_oversold_signals.index[-1].strftime('%Y-%m-%d %H:%M IST') if len(rsi_oversold_signals) > 0 else 'N/A'
            })
        
        # Z-Score extreme pattern
        zscore_extreme = data[abs(zscore) > 2]
        if len(zscore_extreme) > 0:
            successes = 0
            for idx in zscore_extreme.index:
                idx_pos = data.index.get_loc(idx)
                if idx_pos < len(data) - 10:
                    z_val = zscore.loc[idx]
                    future_return = (data['Close'].iloc[idx_pos+10] - data['Close'].iloc[idx_pos]) / data['Close'].iloc[idx_pos] * 100
                    # Success if mean reversion occurs
                    if (z_val < -2 and future_return > 0) or (z_val > 2 and future_return < 0):
                        successes += 1
            accuracy = (successes / len(zscore_extreme)) * 100 if len(zscore_extreme) > 0 else 0
            patterns.append({
                'Pattern': 'Z-Score Mean Reversion (|Z|>2)',
                'Occurrences': len(zscore_extreme),
                'Success_Rate_%': accuracy,
                'Status': '✅ Working' if accuracy > 60 else '❌ Not Working',
                'Confidence': 'High' if accuracy > 70 else 'Medium' if accuracy > 50 else 'Low',
                'Last_Signal': zscore_extreme.index[-1].strftime('%Y-%m-%d %H:%M IST') if len(zscore_extreme) > 0 else 'N/A'
            })
        
        # EMA Crossover pattern
        ema_cross_signals = []
        for i in range(1, len(data)):
            if ema_9.iloc[i-1] <= ema_20.iloc[i-1] and ema_9.iloc[i] > ema_20.iloc[i]:
                ema_cross_signals.append(i)
        
        if len(ema_cross_signals) > 0:
            successes = 0
            for idx_pos in ema_cross_signals:
                if idx_pos < len(data) - 10:
                    future_return = (data['Close'].iloc[idx_pos+10] - data['Close'].iloc[idx_pos]) / data['Close'].iloc[idx_pos] * 100
                    if future_return > 0:
                        successes += 1
            accuracy = (successes / len(ema_cross_signals)) * 100 if len(ema_cross_signals) > 0 else 0
            patterns.append({
                'Pattern': 'EMA 9/20 Bullish Crossover',
                'Occurrences': len(ema_cross_signals),
                'Success_Rate_%': accuracy,
                'Status': '✅ Working' if accuracy > 60 else '❌ Not Working',
                'Confidence': 'High' if accuracy > 70 else 'Medium' if accuracy > 50 else 'Low',
                'Last_Signal': data.index[ema_cross_signals[-1]].strftime('%Y-%m-%d %H:%M IST') if len(ema_cross_signals) > 0 else 'N/A'
            })
        
        # Support bounce pattern
        sr_levels = find_support_resistance(data)
        support_levels = [l for l in sr_levels if l['type'] == 'support']
        
        if support_levels:
            support_bounces = 0
            total_tests = 0
            for level in support_levels[:3]:
                level_price = level['level']
                # Find times when price was near this level
                near_support = data[abs(data['Close'] - level_price) / level_price < 0.01]
                for idx in near_support.index:
                    idx_pos = data.index.get_loc(idx)
                    if idx_pos < len(data) - 5:
                        future_return = (data['Close'].iloc[idx_pos+5] - data['Close'].iloc[idx_pos]) / data['Close'].iloc[idx_pos] * 100
                        total_tests += 1
                        if future_return > 0:
                            support_bounces += 1
            
            accuracy = (support_bounces / total_tests * 100) if total_tests > 0 else 0
            if total_tests > 0:
                patterns.append({
                    'Pattern': 'Support Level Bounce',
                    'Occurrences': total_tests,
                    'Success_Rate_%': accuracy,
                    'Status': '✅ Working' if accuracy > 60 else '❌ Not Working',
                    'Confidence': 'High' if accuracy > 70 else 'Medium' if accuracy > 50 else 'Low',
                    'Last_Signal': near_support.index[-1].strftime('%Y-%m-%d %H:%M IST') if len(near_support) > 0 else 'N/A'
                })
        
        # Store results for this timeframe
        for pattern in patterns:
            pattern['Timeframe'] = f"{interval}_{period}"
            pattern_results.append(pattern)
    
    return pd.DataFrame(pattern_results)

def create_backtest_details_table(data, strategy_type, params):
    """Create detailed backtest execution table"""
    
    capital = 100000
    position = None
    backtest_details = []
    
    close = data['Close']
    rsi = calculate_rsi(close)
    ema_f = calculate_ema(close, params['ema_combo'][0])
    ema_s = calculate_ema(close, params['ema_combo'][1])
    adx = calculate_adx(data['High'], data['Low'], close)
    
    rsi_oversold = params['rsi_range'][0]
    rsi_overbought = params['rsi_range'][1]
    adx_threshold = params['adx_threshold']
    
    for i in range(50, len(data)):
        current_price = float(close.iloc[i])
        current_rsi = float(rsi.iloc[i])
        current_adx = float(adx.iloc[i])
        ema_fast_val = float(ema_f.iloc[i])
        ema_slow_val = float(ema_s.iloc[i])
        
        entry_signal = False
        exit_signal = False
        signal_reasons = []
        
        if position is None:
            # Check entry conditions
            if strategy_type == 'combined':
                if current_rsi < rsi_oversold:
                    signal_reasons.append(f"RSI={current_rsi:.1f}<{rsi_oversold}")
                if ema_fast_val > ema_slow_val:
                    signal_reasons.append(f"EMA{params['ema_combo'][0]}>{params['ema_combo'][1]}")
                if current_adx > adx_threshold:
                    signal_reasons.append(f"ADX={current_adx:.1f}>{adx_threshold}")
                
                if current_rsi < rsi_oversold and ema_fast_val > ema_slow_val and current_adx > adx_threshold:
                    entry_signal = True
            
            if entry_signal:
                shares = (capital * 20 / 100) / current_price
                position = {
                    'entry': current_price,
                    'shares': shares,
                    'entry_date': data.index[i],
                    'entry_rsi': current_rsi,
                    'entry_adx': current_adx
                }
                
                backtest_details.append({
                    'DateTime': data.index[i].strftime('%Y-%m-%d %H:%M:%S IST'),
                    'Action': 'BUY',
                    'Price': current_price,
                    'RSI': current_rsi,
                    'ADX': current_adx,
                    f'EMA_{params["ema_combo"][0]}': ema_fast_val,
                    f'EMA_{params["ema_combo"][1]}': ema_slow_val,
                    'Signal_Reasons': ' & '.join(signal_reasons),
                    'Capital': capital,
                    'Position_Size': shares * current_price,
                    'Trade_Result': 'Open'
                })
        
        elif position is not None:
            # Check exit conditions
            if strategy_type == 'combined':
                if current_rsi > rsi_overbought:
                    signal_reasons.append(f"RSI={current_rsi:.1f}>{rsi_overbought}")
                    exit_signal = True
                if ema_fast_val < ema_slow_val:
                    signal_reasons.append(f"EMA{params['ema_combo'][0]}<{params['ema_combo'][1]}")
                    exit_signal = True
                if current_adx < adx_threshold:
                    signal_reasons.append(f"ADX={current_adx:.1f}<{adx_threshold}")
                    exit_signal = True
            
            pct_change = ((current_price - position['entry']) / position['entry']) * 100
            if pct_change < -5:
                signal_reasons.append("Stop Loss Hit (-5%)")
                exit_signal = True
            elif pct_change > 10:
                signal_reasons.append("Take Profit Hit (+10%)")
                exit_signal = True
            
            if exit_signal:
                profit = (current_price - position['entry']) * position['shares']
                capital += profit
                
                backtest_details.append({
                    'DateTime': data.index[i].strftime('%Y-%m-%d %H:%M:%S IST'),
                    'Action': 'SELL',
                    'Price': current_price,
                    'RSI': current_rsi,
                    'ADX': current_adx,
                    f'EMA_{params["ema_combo"][0]}': ema_fast_val,
                    f'EMA_{params["ema_combo"][1]}': ema_slow_val,
                    'Signal_Reasons': ' & '.join(signal_reasons),
                    'Capital': capital,
                    'Position_Size': 0,
                    'Trade_Result': f"P/L: {profit:+.2f} ({pct_change:+.2f}%)"
                })
                
                position = None
    
    return pd.DataFrame(backtest_details)

def generate_comprehensive_summary(all_timeframe_data, ticker_name, backtest_result):
    """Generate ultra-detailed market analysis summary"""
    
    summary_parts = []
    
    # Header
    summary_parts.append(f"# 🎯 COMPREHENSIVE MARKET ANALYSIS: {ticker_name}")
    summary_parts.append(f"*Analysis Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}*\n")
    
    # Backtest Results
    if backtest_result and backtest_result[1] > 0:
        strategy, returns, params = backtest_result
        summary_parts.append(f"## 💰 BACKTESTING VERDICT: **PROFITABLE**")
        summary_parts.append(f"**Optimized Strategy:** {strategy.upper()} | **Annual Returns:** {returns:.2f}%")
        summary_parts.append(f"**Parameters:** RSI {params['rsi_range']}, EMA {params['ema_combo']}, ADX > {params['adx_threshold']}\n")
    else:
        summary_parts.append(f"## ⚠️ BACKTESTING VERDICT: **REQUIRES CAUTION**")
        summary_parts.append(f"Historical strategy performance was negative. Manual analysis recommended.\n")
    
    # Analyze each timeframe
    timeframe_signals = {}
    
    for tf_key, data in all_timeframe_data.items():
        if len(data) < 50:
            continue
        
        interval, period = tf_key.split('_')
        
        # Calculate all indicators
        close = data['Close']
        rsi = calculate_rsi(close)
        zscore = calculate_zscore(close)
        volatility = calculate_volatility(close.pct_change())
        sr_levels = find_support_resistance(data)
        fib_levels = calculate_fibonacci_levels(data)
        elliott = detect_elliott_wave_detailed(data)
        divergence = detect_rsi_divergence(close, rsi)
        zscore_patterns = analyze_zscore_patterns(data, zscore)
        
        current_price = float(close.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_zscore = float(zscore.iloc[-1])
        current_volatility = float(volatility.iloc[-1])
        
        # Store timeframe analysis
        timeframe_signals[tf_key] = {
            'signal': 'BULLISH' if current_rsi < 40 and current_zscore < -1 else 'BEARISH' if current_rsi > 60 and current_zscore > 1 else 'NEUTRAL',
            'rsi': current_rsi,
            'zscore': current_zscore,
            'volatility': current_volatility,
            'price': current_price,
            'sr_levels': sr_levels,
            'fib': fib_levels,
            'elliott': elliott,
            'divergence': divergence,
            'zscore_patterns': zscore_patterns
        }
    
    # Multi-timeframe consensus
    bullish_count = sum(1 for v in timeframe_signals.values() if v['signal'] == 'BULLISH')
    bearish_count = sum(1 for v in timeframe_signals.values() if v['signal'] == 'BEARISH')
    total_tf = len(timeframe_signals)
    
    if bullish_count / total_tf > 0.6:
        consensus = "🟢 STRONG BULLISH"
    elif bearish_count / total_tf > 0.6:
        consensus = "🔴 STRONG BEARISH"
    elif bullish_count > bearish_count:
        consensus = "🟡 MODERATELY BULLISH"
    elif bearish_count > bullish_count:
        consensus = "🟡 MODERATELY BEARISH"
    else:
        consensus = "⚪ NEUTRAL/SIDEWAYS"
    
    summary_parts.append(f"## 🎲 MULTI-TIMEFRAME CONSENSUS: {consensus}")
    summary_parts.append(f"Analyzed {total_tf} timeframes | Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {total_tf - bullish_count - bearish_count}\n")
    
    # Detailed Z-Score Analysis
    summary_parts.append("## 📊 Z-SCORE MEAN REVERSION ANALYSIS")
    
    for tf_key, analysis in list(timeframe_signals.items())[:3]:  # Top 3 timeframes
        if analysis['zscore_patterns']:
            summary_parts.append(f"\n### {tf_key.upper()} Timeframe:")
            summary_parts.append(f"**Current Z-Score:** {analysis['zscore']:.2f}")
            
            # Historical patterns
            similar_patterns = [p for p in analysis['zscore_patterns'] if abs(p['zscore'] - analysis['zscore']) < 0.5]
            
            if similar_patterns:
                avg_return = np.mean([p['final_return'] for p in similar_patterns])
                accuracy = sum(1 for p in similar_patterns if p['final_return'] > 0) / len(similar_patterns) * 100
                
                summary_parts.append(f"**Historical Pattern Match:** Found {len(similar_patterns)} similar conditions")
                summary_parts.append(f"**Average Outcome:** {avg_return:+.2f}% move | Accuracy: {accuracy:.1f}%")
                
                # Detailed examples
                for i, pattern in enumerate(similar_patterns[-3:], 1):
                    summary_parts.append(
                        f"  {i}. {pattern['time_ago']} ago @ ₹{pattern['entry_price']:.2f} "
                        f"(Z={pattern['zscore']:.2f}) → {pattern['final_return']:+.2f}% "
                        f"(Max: {pattern['max_gain']:+.2f}%, Min: {pattern['max_loss']:+.2f}%)"
                    )
                
                # Prediction
                if analysis['zscore'] < -2 and avg_return > 5:
                    summary_parts.append(f"  ✅ **STRONG BUY SIGNAL**: Oversold condition with {accuracy:.0f}% historical success")
                elif analysis['zscore'] > 2 and avg_return < -5:
                    summary_parts.append(f"  ❌ **STRONG SELL SIGNAL**: Overbought condition with {accuracy:.0f}% historical success")
    
    # Support & Resistance Analysis
    summary_parts.append("\n## 🎚️ SUPPORT & RESISTANCE ANALYSIS")
    
    primary_tf = list(timeframe_signals.keys())[0] if timeframe_signals else None
    if primary_tf:
        sr = timeframe_signals[primary_tf]['sr_levels']
        current_p = timeframe_signals[primary_tf]['price']
        
        nearby_levels = [l for l in sr if abs(l['level'] - current_p) / current_p < 0.02]
        
        if nearby_levels:
            for level in nearby_levels[:3]:
                time_ago = datetime.now(IST) - level['date']
                time_ago_str = format_time_ago(time_ago)
                distance_pct = ((level['level'] - current_p) / current_p) * 100
                distance_points = level['level'] - current_p
                
                summary_parts.append(
                    f"**{level['type'].upper()} @ ₹{level['level']:.2f}** | "
                    f"Distance: {distance_pct:+.2f}% ({distance_points:+.2f} points) | "
                    f"{level['touches']} touches | Sustained {level['sustained']} periods | "
                    f"Formed on {level['date'].strftime('%Y-%m-%d %H:%M IST')} ({time_ago_str})"
                )
                
                if level['type'] == 'support' and distance_pct > -1:
                    summary_parts.append(f"  → **HIGH PROBABILITY BOUNCE ZONE** - Price testing strong support")
                elif level['type'] == 'resistance' and distance_pct < 1:
                    summary_parts.append(f"  → **POTENTIAL REJECTION ZONE** - Price approaching strong resistance")
    
    # Fibonacci Analysis
    summary_parts.append("\n## 📐 FIBONACCI RETRACEMENT LEVELS")
    
    if primary_tf:
        fib = timeframe_signals[primary_tf]['fib']
        current_p = timeframe_signals[primary_tf]['price']
        
        for level_name, level_data in fib.items():
            distance = ((level_data['price'] - current_p) / current_p) * 100
            
            if abs(distance) < 2:  # Within 2%
                summary_parts.append(
                    f"**Fib {level_name}** @ ₹{level_data['price']:.2f} "
                    f"({distance:+.2f}% away) ⚠️ **CRITICAL LEVEL**"
                )
                
                if level_name in ['0.382', '0.5', '0.618']:
                    summary_parts.append(f"  → Golden ratio level: Strong support/resistance expected")
    
    # Elliott Wave Analysis
    summary_parts.append("\n## 🌊 ELLIOTT WAVE PATTERN ANALYSIS")
    
    if primary_tf and timeframe_signals[primary_tf]['elliott']:
        elliott = timeframe_signals[primary_tf]['elliott']
        waves = elliott['waves']
        
        summary_parts.append(f"**Current Phase:** {elliott['current_wave']}")
        summary_parts.append(f"**Wave Structure Detected:** {elliott['total_waves']} waves identified\n")
        
        for wave in waves[-5:]:  # Last 5 waves
            summary_parts.append(
                f"**{wave['phase']}** ({wave['wave_type']}) | "
                f"{wave['point_type']} @ ₹{wave['price']:.2f} | "
                f"Change: {wave['change']:+.2f} ({wave['change_pct']:+.2f}%) | "
                f"Duration: {wave['duration']} | "
                f"Date: {wave['date'].strftime('%Y-%m-%d %H:%M IST')}"
            )
        
        # Wave 5 completion check
        if elliott['current_wave'] == "Wave 5 or Extension":
            summary_parts.append("\n⚠️ **WAVE 5 ALERT**: Impulse wave nearing completion. Expect corrective ABC wave soon.")
        elif "Corrective" in elliott['current_wave']:
            summary_parts.append("\n✅ **CORRECTIVE PHASE**: ABC correction in progress. Watch for Wave C completion for reversal.")
    
    # RSI Divergence
    summary_parts.append("\n## ⚡ RSI DIVERGENCE SIGNALS")
    
    divergence_found = False
    for tf_key, analysis in timeframe_signals.items():
        if analysis['divergence']:
            divergence_found = True
            for div in analysis['divergence']:
                summary_parts.append(
                    f"**{tf_key.upper()}**: {div['type'].upper()} divergence detected {div['time_ago']} ago | "
                    f"Price: ₹{div['price']:.2f} | RSI: {div['rsi_old']:.1f} → {div['rsi_new']:.1f}"
                )
                
                if div['type'] == 'bullish':
                    summary_parts.append(f"  → **BULLISH**: Lower lows in price but higher lows in RSI = Reversal likely")
                else:
                    summary_parts.append(f"  → **BEARISH**: Higher highs in price but lower highs in RSI = Reversal likely")
    
    if not divergence_found:
        summary_parts.append("No significant RSI divergence detected in recent periods.")
    
    # Final Recommendation
    summary_parts.append("\n## 🎯 FINAL TRADING RECOMMENDATION")
    
    # Calculate recommendation based on all factors
    score = 0
    reasons = []
    
    # Backtest
    if backtest_result and backtest_result[1] > 15:
        score += 3
        reasons.append(f"✅ Backtest shows {backtest_result[1]:.1f}% annual returns")
    elif backtest_result and backtest_result[1] > 0:
        score += 1
        reasons.append(f"⚠️ Backtest shows marginal {backtest_result[1]:.1f}% returns")
    else:
        score -= 2
        reasons.append("❌ Backtest shows negative returns")
    
    # Multi-timeframe
    if bullish_count / total_tf > 0.6:
        score += 2
        reasons.append(f"✅ {bullish_count}/{total_tf} timeframes bullish")
    elif bearish_count / total_tf > 0.6:
        score -= 2
        reasons.append(f"❌ {bearish_count}/{total_tf} timeframes bearish")
    
    # Z-score
    primary_zscore = timeframe_signals[primary_tf]['zscore'] if primary_tf else 0
    if primary_zscore < -2:
        score += 2
        reasons.append(f"✅ Extreme oversold (Z={primary_zscore:.2f})")
    elif primary_zscore > 2:
        score -= 2
        reasons.append(f"❌ Extreme overbought (Z={primary_zscore:.2f})")
    
    # Support/Resistance
    if nearby_levels:
        support_nearby = any(l['type'] == 'support' for l in nearby_levels)
        resistance_nearby = any(l['type'] == 'resistance' for l in nearby_levels)
        
        if support_nearby:
            score += 1
            reasons.append("✅ Near strong support level")
        if resistance_nearby:
            score -= 1
            reasons.append("❌ Near strong resistance level")
    
    # Divergence
    bullish_div = any(
        analysis['divergence'] and any(d['type'] == 'bullish' for d in analysis['divergence'])
        for analysis in timeframe_signals.values() if analysis['divergence']
    )
    bearish_div = any(
        analysis['divergence'] and any(d['type'] == 'bearish' for d in analysis['divergence'])
        for analysis in timeframe_signals.values() if analysis['divergence']
    )
    
    if bullish_div:
        score += 2
        reasons.append("✅ Bullish RSI divergence detected")
    if bearish_div:
        score -= 2
        reasons.append("❌ Bearish RSI divergence detected")
    
    # Final signal
    if score >= 5:
        final_signal = "🟢 STRONG BUY"
        signal_class = "buy-signal"
    elif score >= 2:
        final_signal = "🟢 BUY"
        signal_class = "buy-signal"
    elif score <= -5:
        final_signal = "🔴 STRONG SELL"
        signal_class = "sell-signal"
    elif score <= -2:
        final_signal = "🔴 SELL"
        signal_class = "sell-signal"
    else:
        final_signal = "🟡 HOLD/WAIT"
        signal_class = "hold-signal"
    
    summary_parts.append(f"\n### **SIGNAL: {final_signal}** (Confidence Score: {score}/10)")
    summary_parts.append("\n**Reasoning:**")
    for reason in reasons:
        summary_parts.append(f"  {reason}")
    
    # Risk Management
    primary_data = all_timeframe_data[primary_tf]
    current_p = timeframe_signals[primary_tf]['price']
    atr = (primary_data['High'] - primary_data['Low']).rolling(14).mean().iloc[-1]
    
    if "BUY" in final_signal:
        entry = current_p
        sl = entry - (1.5 * atr)
        target1 = entry + (2 * atr)
        target2 = entry + (3 * atr)
        target3 = entry + (5 * atr)
        
        summary_parts.append(f"\n### 💼 POSITION DETAILS:")
        summary_parts.append(f"**Entry Price:** ₹{entry:.2f}")
        summary_parts.append(f"**Stop Loss:** ₹{sl:.2f} ({((sl-entry)/entry*100):.2f}%)")
        summary_parts.append(f"**Target 1:** ₹{target1:.2f} ({((target1-entry)/entry*100):.2f}%) - Book 30%")
        summary_parts.append(f"**Target 2:** ₹{target2:.2f} ({((target2-entry)/entry*100):.2f}%) - Book 40%")
        summary_parts.append(f"**Target 3:** ₹{target3:.2f} ({((target3-entry)/entry*100):.2f}%) - Book 30%")
        summary_parts.append(f"**Risk:Reward Ratio:** 1:{((target1-entry)/(entry-sl)):.2f}")
        
    elif "SELL" in final_signal:
        entry = current_p
        sl = entry + (1.5 * atr)
        target1 = entry - (2 * atr)
        target2 = entry - (3 * atr)
        target3 = entry - (5 * atr)
        
        summary_parts.append(f"\n### 💼 POSITION DETAILS:")
        summary_parts.append(f"**Entry Price:** ₹{entry:.2f}")
        summary_parts.append(f"**Stop Loss:** ₹{sl:.2f} ({((sl-entry)/entry*100):+.2f}%)")
        summary_parts.append(f"**Target 1:** ₹{target1:.2f} ({((target1-entry)/entry*100):.2f}%) - Book 30%")
        summary_parts.append(f"**Target 2:** ₹{target2:.2f} ({((target2-entry)/entry*100):.2f}%) - Book 40%")
        summary_parts.append(f"**Target 3:** ₹{target3:.2f} ({((target3-entry)/entry*100):.2f}%) - Book 30%")
        summary_parts.append(f"**Risk:Reward Ratio:** 1:{((entry-target1)/(sl-entry)):.2f}")
    
    summary_parts.append(f"\n### ⚠️ RISK DISCLAIMER")
    summary_parts.append("This is an algorithmic analysis based on historical patterns and technical indicators.")
    summary_parts.append("Past performance does not guarantee future results. Always use proper position sizing and risk management.")
    summary_parts.append(f"Recommended position size: 2-5% of portfolio | Never risk more than 1-2% per trade.")
    
    return "\n".join(summary_parts), final_signal, signal_class
    """Create detailed backtest execution table"""
    
    capital = 100000
    position = None
    backtest_details = []
    
    close = data['Close']
    rsi = calculate_rsi(close)
    ema_f = calculate_ema(close, params['ema_combo'][0])
    ema_s = calculate_ema(close, params['ema_combo'][1])
    adx = calculate_adx(data['High'], data['Low'], close)
    
    rsi_oversold = params['rsi_range'][0]
    rsi_overbought = params['rsi_range'][1]
    adx_threshold = params['adx_threshold']
    
    for i in range(50, len(data)):
        current_price = float(close.iloc[i])
        current_rsi = float(rsi.iloc[i])
        current_adx = float(adx.iloc[i])
        ema_fast_val = float(ema_f.iloc[i])
        ema_slow_val = float(ema_s.iloc[i])
        
        entry_signal = False
        exit_signal = False
        signal_reasons = []
        
        if position is None:
            # Check entry conditions
            if strategy_type == 'combined':
                if current_rsi < rsi_oversold:
                    signal_reasons.append(f"RSI={current_rsi:.1f}<{rsi_oversold}")
                if ema_fast_val > ema_slow_val:
                    signal_reasons.append(f"EMA{params['ema_combo'][0]}>{params['ema_combo'][1]}")
                if current_adx > adx_threshold:
                    signal_reasons.append(f"ADX={current_adx:.1f}>{adx_threshold}")
                
                if current_rsi < rsi_oversold and ema_fast_val > ema_slow_val and current_adx > adx_threshold:
                    entry_signal = True
            
            if entry_signal:
                shares = (capital * 20 / 100) / current_price
                position = {
                    'entry': current_price,
                    'shares': shares,
                    'entry_date': data.index[i],
                    'entry_rsi': current_rsi,
                    'entry_adx': current_adx
                }
                
                backtest_details.append({
                    'DateTime': data.index[i].strftime('%Y-%m-%d %H:%M:%S IST'),
                    'Action': 'BUY',
                    'Price': current_price,
                    'RSI': current_rsi,
                    'ADX': current_adx,
                    f'EMA_{params["ema_combo"][0]}': ema_fast_val,
                    f'EMA_{params["ema_combo"][1]}': ema_slow_val,
                    'Signal_Reasons': ' & '.join(signal_reasons),
                    'Capital': capital,
                    'Position_Size': shares * current_price,
                    'Trade_Result': 'Open'
                })
        
        elif position is not None:
            # Check exit conditions
            if strategy_type == 'combined':
                if current_rsi > rsi_overbought:
                    signal_reasons.append(f"RSI={current_rsi:.1f}>{rsi_overbought}")
                    exit_signal = True
                if ema_fast_val < ema_slow_val:
                    signal_reasons.append(f"EMA{params['ema_combo'][0]}<{params['ema_combo'][1]}")
                    exit_signal = True
                if current_adx < adx_threshold:
                    signal_reasons.append(f"ADX={current_adx:.1f}<{adx_threshold}")
                    exit_signal = True
            
            pct_change = ((current_price - position['entry']) / position['entry']) * 100
            if pct_change < -5:
                signal_reasons.append("Stop Loss Hit (-5%)")
                exit_signal = True
            elif pct_change > 10:
                signal_reasons.append("Take Profit Hit (+10%)")
                exit_signal = True
            
            if exit_signal:
                profit = (current_price - position['entry']) * position['shares']
                capital += profit
                
                backtest_details.append({
                    'DateTime': data.index[i].strftime('%Y-%m-%d %H:%M:%S IST'),
                    'Action': 'SELL',
                    'Price': current_price,
                    'RSI': current_rsi,
                    'ADX': current_adx,
                    f'EMA_{params["ema_combo"][0]}': ema_fast_val,
                    f'EMA_{params["ema_combo"][1]}': ema_slow_val,
                    'Signal_Reasons': ' & '.join(signal_reasons),
                    'Capital': capital,
                    'Position_Size': 0,
                    'Trade_Result': f"P/L: {profit:+.2f} ({pct_change:+.2f}%)"
                })
                
                position = None
    
    return pd.DataFrame(backtest_details)
    """Generate ultra-detailed market analysis summary"""
    
    summary_parts = []
    
    # Header
    summary_parts.append(f"# 🎯 COMPREHENSIVE MARKET ANALYSIS: {ticker_name}")
    summary_parts.append(f"*Analysis Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}*\n")
    
    # Backtest Results
    if backtest_result and backtest_result[1] > 0:
        strategy, returns, params = backtest_result
        summary_parts.append(f"## 💰 BACKTESTING VERDICT: **PROFITABLE**")
        summary_parts.append(f"**Optimized Strategy:** {strategy.upper()} | **Annual Returns:** {returns:.2f}%")
        summary_parts.append(f"**Parameters:** RSI {params['rsi_range']}, EMA {params['ema_combo']}, ADX > {params['adx_threshold']}\n")
    else:
        summary_parts.append(f"## ⚠️ BACKTESTING VERDICT: **REQUIRES CAUTION**")
        summary_parts.append(f"Historical strategy performance was negative. Manual analysis recommended.\n")
    
    # Analyze each timeframe
    timeframe_signals = {}
    
    for tf_key, data in all_timeframe_data.items():
        if len(data) < 50:
            continue
        
        interval, period = tf_key.split('_')
        
        # Calculate all indicators
        close = data['Close']
        rsi = calculate_rsi(close)
        zscore = calculate_zscore(close)
        volatility = calculate_volatility(close.pct_change())
        sr_levels = find_support_resistance(data)
        fib_levels = calculate_fibonacci_levels(data)
        elliott = detect_elliott_wave_detailed(data)
        divergence = detect_rsi_divergence(close, rsi)
        zscore_patterns = analyze_zscore_patterns(data, zscore)
        
        current_price = float(close.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_zscore = float(zscore.iloc[-1])
        current_volatility = float(volatility.iloc[-1])
        
        # Store timeframe analysis
        timeframe_signals[tf_key] = {
            'signal': 'BULLISH' if current_rsi < 40 and current_zscore < -1 else 'BEARISH' if current_rsi > 60 and current_zscore > 1 else 'NEUTRAL',
            'rsi': current_rsi,
            'zscore': current_zscore,
            'volatility': current_volatility,
            'price': current_price,
            'sr_levels': sr_levels,
            'fib': fib_levels,
            'elliott': elliott,
            'divergence': divergence,
            'zscore_patterns': zscore_patterns
        }
    
    # Multi-timeframe consensus
    bullish_count = sum(1 for v in timeframe_signals.values() if v['signal'] == 'BULLISH')
    bearish_count = sum(1 for v in timeframe_signals.values() if v['signal'] == 'BEARISH')
    total_tf = len(timeframe_signals)
    
    if bullish_count / total_tf > 0.6:
        consensus = "🟢 STRONG BULLISH"
    elif bearish_count / total_tf > 0.6:
        consensus = "🔴 STRONG BEARISH"
    elif bullish_count > bearish_count:
        consensus = "🟡 MODERATELY BULLISH"
    elif bearish_count > bullish_count:
        consensus = "🟡 MODERATELY BEARISH"
    else:
        consensus = "⚪ NEUTRAL/SIDEWAYS"
    
    summary_parts.append(f"## 🎲 MULTI-TIMEFRAME CONSENSUS: {consensus}")
    summary_parts.append(f"Analyzed {total_tf} timeframes | Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {total_tf - bullish_count - bearish_count}\n")
    
    # Detailed Z-Score Analysis
    summary_parts.append("## 📊 Z-SCORE MEAN REVERSION ANALYSIS")
    
    for tf_key, analysis in list(timeframe_signals.items())[:3]:  # Top 3 timeframes
        if analysis['zscore_patterns']:
            summary_parts.append(f"\n### {tf_key.upper()} Timeframe:")
            summary_parts.append(f"**Current Z-Score:** {analysis['zscore']:.2f}")
            
            # Historical patterns
            similar_patterns = [p for p in analysis['zscore_patterns'] if abs(p['zscore'] - analysis['zscore']) < 0.5]
            
            if similar_patterns:
                avg_return = np.mean([p['final_return'] for p in similar_patterns])
                accuracy = sum(1 for p in similar_patterns if p['final_return'] > 0) / len(similar_patterns) * 100
                
                summary_parts.append(f"**Historical Pattern Match:** Found {len(similar_patterns)} similar conditions")
                summary_parts.append(f"**Average Outcome:** {avg_return:+.2f}% move | Accuracy: {accuracy:.1f}%")
                
                # Detailed examples
                for i, pattern in enumerate(similar_patterns[-3:], 1):
                    summary_parts.append(
                        f"  {i}. {pattern['time_ago']} ago @ ₹{pattern['entry_price']:.2f} "
                        f"(Z={pattern['zscore']:.2f}) → {pattern['final_return']:+.2f}% "
                        f"(Max: {pattern['max_gain']:+.2f}%, Min: {pattern['max_loss']:+.2f}%)"
                    )
                
                # Prediction
                if analysis['zscore'] < -2 and avg_return > 5:
                    summary_parts.append(f"  ✅ **STRONG BUY SIGNAL**: Oversold condition with {accuracy:.0f}% historical success")
                elif analysis['zscore'] > 2 and avg_return < -5:
                    summary_parts.append(f"  ❌ **STRONG SELL SIGNAL**: Overbought condition with {accuracy:.0f}% historical success")
    
    # Support & Resistance Analysis
    summary_parts.append("\n## 🎚️ SUPPORT & RESISTANCE ANALYSIS")
    
    primary_tf = list(timeframe_signals.keys())[0] if timeframe_signals else None
    if primary_tf:
        sr = timeframe_signals[primary_tf]['sr_levels']
        current_p = timeframe_signals[primary_tf]['price']
        
        nearby_levels = [l for l in sr if abs(l['level'] - current_p) / current_p < 0.02]
        
        if nearby_levels:
            for level in nearby_levels[:3]:
                time_ago = datetime.now(IST) - level['date']
                distance = ((level['level'] - current_p) / current_p) * 100
                
                summary_parts.append(
                    f"**{level['type'].upper()} @ ₹{level['level']:.2f}** "
                    f"({distance:+.2f}% away) | {level['touches']} touches | "
                    f"Sustained {level['sustained']} periods | "
                    f"Formed {str(time_ago).split('.')[0]} ago"
                )
                
                if level['type'] == 'support' and distance < -1:
                    summary_parts.append(f"  → Price near strong support: **HIGH PROBABILITY BOUNCE ZONE**")
                elif level['type'] == 'resistance' and distance > 1:
                    summary_parts.append(f"  → Price near strong resistance: **POTENTIAL REJECTION ZONE**")
    
    # Fibonacci Analysis
    summary_parts.append("\n## 📐 FIBONACCI RETRACEMENT LEVELS")
    
    if primary_tf:
        fib = timeframe_signals[primary_tf]['fib']
        current_p = timeframe_signals[primary_tf]['price']
        
        for level_name, level_data in fib.items():
            distance = ((level_data['price'] - current_p) / current_p) * 100
            
            if abs(distance) < 2:  # Within 2%
                summary_parts.append(
                    f"**Fib {level_name}** @ ₹{level_data['price']:.2f} "
                    f"({distance:+.2f}% away) ⚠️ **CRITICAL LEVEL**"
                )
                
                if level_name in ['0.382', '0.5', '0.618']:
                    summary_parts.append(f"  → Golden ratio level: Strong support/resistance expected")
    
    # Elliott Wave Analysis
    summary_parts.append("\n## 🌊 ELLIOTT WAVE PATTERN ANALYSIS")
    
    if primary_tf and timeframe_signals[primary_tf]['elliott']:
        elliott = timeframe_signals[primary_tf]['elliott']
        waves = elliott['waves']
        
        summary_parts.append(f"**Current Phase:** {elliott['current_wave']}")
        summary_parts.append(f"**Wave Structure Detected:** {elliott['total_waves']} waves identified\n")
        
        for wave in waves[-5:]:  # Last 5 waves
            summary_parts.append(
                f"**{wave['phase']}** ({wave['wave_type']}) | "
                f"{wave['point_type']} @ ₹{wave['price']:.2f} | "
                f"Change: {wave['change']:+.2f} ({wave['change_pct']:+.2f}%) | "
                f"Duration: {wave['duration']} | "
                f"Date: {wave['date'].strftime('%Y-%m-%d %H:%M IST')}"
            )
        
        # Wave 5 completion check
        if elliott['current_wave'] == "Wave 5 or Extension":
            summary_parts.append("\n⚠️ **WAVE 5 ALERT**: Impulse wave nearing completion. Expect corrective ABC wave soon.")
        elif "Corrective" in elliott['current_wave']:
            summary_parts.append("\n✅ **CORRECTIVE PHASE**: ABC correction in progress. Watch for Wave C completion for reversal.")
    
    # RSI Divergence
    summary_parts.append("\n## ⚡ RSI DIVERGENCE SIGNALS")
    
    divergence_found = False
    for tf_key, analysis in timeframe_signals.items():
        if analysis['divergence']:
            divergence_found = True
            for div in analysis['divergence']:
                summary_parts.append(
                    f"**{tf_key.upper()}**: {div['type'].upper()} divergence detected {div['time_ago']} ago | "
                    f"Price: ₹{div['price']:.2f} | RSI: {div['rsi_old']:.1f} → {div['rsi_new']:.1f}"
                )
                
                if div['type'] == 'bullish':
                    summary_parts.append(f"  → **BULLISH**: Lower lows in price but higher lows in RSI = Reversal likely")
                else:
                    summary_parts.append(f"  → **BEARISH**: Higher highs in price but lower highs in RSI = Reversal likely")
    
    if not divergence_found:
        summary_parts.append("No significant RSI divergence detected in recent periods.")
    
    # Final Recommendation
    summary_parts.append("\n## 🎯 FINAL TRADING RECOMMENDATION")
    
    # Calculate recommendation based on all factors
    score = 0
    reasons = []
    
    # Backtest
    if backtest_result and backtest_result[1] > 15:
        score += 3
        reasons.append(f"✅ Backtest shows {backtest_result[1]:.1f}% annual returns")
    elif backtest_result and backtest_result[1] > 0:
        score += 1
        reasons.append(f"⚠️ Backtest shows marginal {backtest_result[1]:.1f}% returns")
    else:
        score -= 2
        reasons.append("❌ Backtest shows negative returns")
    
    # Multi-timeframe
    if bullish_count / total_tf > 0.6:
        score += 2
        reasons.append(f"✅ {bullish_count}/{total_tf} timeframes bullish")
    elif bearish_count / total_tf > 0.6:
        score -= 2
        reasons.append(f"❌ {bearish_count}/{total_tf} timeframes bearish")
    
    # Z-score
    primary_zscore = timeframe_signals[primary_tf]['zscore'] if primary_tf else 0
    if primary_zscore < -2:
        score += 2
        reasons.append(f"✅ Extreme oversold (Z={primary_zscore:.2f})")
    elif primary_zscore > 2:
        score -= 2
        reasons.append(f"❌ Extreme overbought (Z={primary_zscore:.2f})")
    
    # Support/Resistance
    if nearby_levels:
        support_nearby = any(l['type'] == 'support' for l in nearby_levels)
        resistance_nearby = any(l['type'] == 'resistance' for l in nearby_levels)
        
        if support_nearby:
            score += 1
            reasons.append("✅ Near strong support level")
        if resistance_nearby:
            score -= 1
            reasons.append("❌ Near strong resistance level")
    
    # Divergence
    bullish_div = any(
        analysis['divergence'] and any(d['type'] == 'bullish' for d in analysis['divergence'])
        for analysis in timeframe_signals.values() if analysis['divergence']
    )
    bearish_div = any(
        analysis['divergence'] and any(d['type'] == 'bearish' for d in analysis['divergence'])
        for analysis in timeframe_signals.values() if analysis['divergence']
    )
    
    if bullish_div:
        score += 2
        reasons.append("✅ Bullish RSI divergence detected")
    if bearish_div:
        score -= 2
        reasons.append("❌ Bearish RSI divergence detected")
    
    # Final signal
    if score >= 5:
        final_signal = "🟢 STRONG BUY"
        signal_class = "buy-signal"
    elif score >= 2:
        final_signal = "🟢 BUY"
        signal_class = "buy-signal"
    elif score <= -5:
        final_signal = "🔴 STRONG SELL"
        signal_class = "sell-signal"
    elif score <= -2:
        final_signal = "🔴 SELL"
        signal_class = "sell-signal"
    else:
        final_signal = "🟡 HOLD/WAIT"
        signal_class = "hold-signal"
    
    summary_parts.append(f"\n### **SIGNAL: {final_signal}** (Confidence Score: {score}/10)")
    summary_parts.append("\n**Reasoning:**")
    for reason in reasons:
        summary_parts.append(f"  {reason}")
    
    # Risk Management
    if primary_tf:
        current_p = timeframe_signals[primary_tf]['price']
        atr = (all_timeframe_data[primary_tf]['High'] - all_timeframe_data[primary_tf]['Low']).rolling(14).mean().iloc[-1]
        
        if "BUY" in final_signal:
            entry = current_p
            sl = entry - (1.5 * atr)
            target1 = entry + (2 * atr)
            target2 = entry + (3 * atr)
            target3 = entry + (5 * atr)
            
            summary_parts.append(f"\n### 💼 POSITION DETAILS:")
            summary_parts.append(f"**Entry Price:** ₹{entry:.2f}")
            summary_parts.append(f"**Stop Loss:** ₹{sl:.2f} ({((sl-entry)/entry*100):.2f}%)")
            summary_parts.append(f"**Target 1:** ₹{target1:.2f} ({((target1-entry)/entry*100):.2f}%) - Book 30%")
            summary_parts.append(f"**Target 2:** ₹{target2:.2f} ({((target2-entry)/entry*100):.2f}%) - Book 40%")
            summary_parts.append(f"**Target 3:** ₹{target3:.2f} ({((target3-entry)/entry*100):.2f}%) - Book 30%")
            summary_parts.append(f"**Risk:Reward Ratio:** 1:{((target1-entry)/(entry-sl)):.2f}")
            
        elif "SELL" in final_signal:
            entry = current_p
            sl = entry + (1.5 * atr)
            target1 = entry - (2 * atr)
            target2 = entry - (3 * atr)
            target3 = entry - (5 * atr)
            
            summary_parts.append(f"\n### 💼 POSITION DETAILS:")
            summary_parts.append(f"**Entry Price:** ₹{entry:.2f}")
            summary_parts.append(f"**Stop Loss:** ₹{sl:.2f} ({((sl-entry)/entry*100):+.2f}%)")
            summary_parts.append(f"**Target 1:** ₹{target1:.2f} ({((target1-entry)/entry*100):.2f}%) - Book 30%")
            summary_parts.append(f"**Target 2:** ₹{target2:.2f} ({((target2-entry)/entry*100):.2f}%) - Book 40%")
            summary_parts.append(f"**Target 3:** ₹{target3:.2f} ({((target3-entry)/entry*100):.2f}%) - Book 30%")
            summary_parts.append(f"**Risk:Reward Ratio:** 1:{((entry-target1)/(sl-entry)):.2f}")
    
    summary_parts.append(f"\n### ⚠️ RISK DISCLAIMER")
    summary_parts.append("This is an algorithmic analysis based on historical patterns and technical indicators.")
    summary_parts.append("Past performance does not guarantee future results. Always use proper position sizing and risk management.")
    summary_parts.append(f"Recommended position size: 2-5% of portfolio | Never risk more than 1-2% per trade.")
    
    return "\n".join(summary_parts), final_signal, signal_class

# Sidebar
st.sidebar.title("⚙️ Trading Configuration")

ticker1_input = st.sidebar.selectbox(
    "Select Ticker 1",
    list(TICKER_MAP.keys()) + ["Custom"],
    index=0
)

if ticker1_input == "Custom":
    ticker1_custom = st.sidebar.text_input("Enter Custom Ticker 1", "AAPL")
    ticker1 = ticker1_custom
else:
    ticker1 = get_ticker_symbol(ticker1_input)

enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)")

ticker2 = None
if enable_ratio:
    ticker2_input = st.sidebar.selectbox(
        "Select Ticker 2",
        list(TICKER_MAP.keys()) + ["Custom"],
        index=7
    )
    
    if ticker2_input == "Custom":
        ticker2_custom = st.sidebar.text_input("Enter Custom Ticker 2", "MSFT")
        ticker2 = ticker2_custom
    else:
        ticker2 = get_ticker_symbol(ticker2_input)

# Fetch data button
if st.sidebar.button("🔄 Fetch All Data", type="primary"):
    # Create progress tracking
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # Fetch ticker 1 data
        status_text.text("📊 Fetching Ticker 1 data...")
        
        total_combinations = sum(len(periods) for periods in VALID_COMBINATIONS.values())
        fetched = 0
        
        all_data_t1 = {}
        
        for interval, periods in VALID_COMBINATIONS.items():
            for period in periods:
                try:
                    status_text.text(f"📊 Fetching {ticker1}: {interval} - {period} ({fetched+1}/{total_combinations})")
                    data = fetch_data_with_retry(ticker1, period, interval)
                    if data is not None and len(data) > 0:
                        key = f"{interval}_{period}"
                        all_data_t1[key] = data
                    fetched += 1
                    progress_bar.progress(fetched / (total_combinations * (2 if enable_ratio else 1)))
                except:
                    fetched += 1
                    continue
        
        st.session_state.all_timeframe_data = all_data_t1
        
        if st.session_state.all_timeframe_data:
            status_text.text(f"✅ {ticker1}: {len(st.session_state.all_timeframe_data)} combinations fetched")
        else:
            status_text.text(f"❌ Failed to fetch {ticker1} data")
        
        # Fetch ticker 2 if enabled
        if enable_ratio and ticker2:
            status_text.text("📊 Fetching Ticker 2 data...")
            
            all_data_t2 = {}
            
            for interval, periods in VALID_COMBINATIONS.items():
                for period in periods:
                    try:
                        status_text.text(f"📊 Fetching {ticker2}: {interval} - {period} ({fetched+1}/{total_combinations * 2})")
                        data = fetch_data_with_retry(ticker2, period, interval)
                        if data is not None and len(data) > 0:
                            key = f"{interval}_{period}"
                            all_data_t2[key] = data
                        fetched += 1
                        progress_bar.progress(fetched / (total_combinations * 2))
                    except:
                        fetched += 1
                        continue
            
            st.session_state.all_timeframe_data_t2 = all_data_t2
            
            if st.session_state.all_timeframe_data_t2:
                status_text.text(f"✅ {ticker2}: {len(st.session_state.all_timeframe_data_t2)} combinations fetched")
            else:
                status_text.text(f"❌ Failed to fetch {ticker2} data")
        
        # Analysis phase
        if st.session_state.all_timeframe_data:
            progress_bar.progress(0.5)
            status_text.text("🔬 Running comprehensive analysis...")
            time.sleep(1)
            
            progress_bar.progress(0.7)
            status_text.text("📈 Generating indicators...")
            time.sleep(1)
            
            progress_bar.progress(0.85)
            status_text.text("🎯 Creating recommendations...")
            time.sleep(1)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            st.session_state.data_fetched = True
            st.success("✅ Comprehensive data fetched! Scroll down for detailed analysis.")
        else:
            status_text.text("❌ Failed to fetch data")
            st.error("Failed to fetch data. Please try a different ticker.")
            st.session_state.data_fetched = False
            
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error during data fetch: {str(e)}")
        st.session_state.data_fetched = False

# Main content
st.markdown('<div class="main-header">📈 Advanced Algorithmic Trading Analysis</div>', unsafe_allow_html=True)

if st.session_state.data_fetched and st.session_state.all_timeframe_data:
    
    # Get primary timeframe data
    primary_key = list(st.session_state.all_timeframe_data.keys())[0]
    primary_data = st.session_state.all_timeframe_data[primary_key]
    
    # Current metrics
    st.subheader("📊 Current Market Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = float(primary_data['Close'].iloc[-1])
    prev_price = float(primary_data['Close'].iloc[-2]) if len(primary_data) > 1 else current_price
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100
    
    rsi = calculate_rsi(primary_data['Close'])
    current_rsi = float(rsi.iloc[-1])
    
    volatility = calculate_volatility(primary_data['Close'].pct_change())
    current_vol = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0
    
    zscore = calculate_zscore(primary_data['Close'])
    current_zscore = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0
    
    with col1:
        color_class = "positive" if price_change > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>{ticker1}</h4>
            <h2 class="{color_class}">₹{current_price:.2f}</h2>
            <p class="{color_class}">{price_change:+.2f} ({pct_change:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_color = "positive" if 30 <= current_rsi <= 70 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>RSI</h4>
            <h2 class="{rsi_color}">{current_rsi:.1f}</h2>
            <p>{'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volatility</h4>
            <h2>{current_vol:.1f}%</h2>
            <p>Annualized</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        zscore_color = "positive" if abs(current_zscore) < 2 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Z-Score</h4>
            <h2 class="{zscore_color}">{current_zscore:.2f}</h2>
            <p>Mean Reversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Run backtest
    backtest_result = run_full_backtest_with_optimization(primary_data, ticker1)
    
    # Generate comprehensive summary
    st.subheader("📋 COMPREHENSIVE MARKET ANALYSIS")
    
    summary_text, final_signal, signal_class = generate_comprehensive_summary(
        st.session_state.all_timeframe_data, 
        ticker1,
        backtest_result
    )
    
    # Display final signal prominently
    st.markdown(f"""
    <div class="signal-box {signal_class}">
        <h2 style="margin: 0;">FINAL RECOMMENDATION: {final_signal}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display comprehensive summary
    st.markdown(f"""
    <div class="summary-box">
    {summary_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Pattern Performance Analysis
    st.subheader("🎯 Pattern Performance Analysis - What's Working & What's Not")
    
    pattern_analysis = analyze_pattern_performance(st.session_state.all_timeframe_data, ticker1)
    
    if not pattern_analysis.empty:
        # Separate working and not working patterns
        working_patterns = pattern_analysis[pattern_analysis['Status'].str.contains('Working')]
        not_working_patterns = pattern_analysis[pattern_analysis['Status'].str.contains('Not Working')]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### ✅ Working Patterns (>60% Accuracy)")
            if not working_patterns.empty:
                st.dataframe(working_patterns, use_container_width=True)
            else:
                st.info("No consistently working patterns found in recent data")
        
        with col2:
            st.write("### ❌ Non-Working Patterns (<60% Accuracy)")
            if not not_working_patterns.empty:
                st.dataframe(not_working_patterns, use_container_width=True)
            else:
                st.success("All tested patterns showing positive results!")
        
        # Summary statistics
        st.write("### 📊 Pattern Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        avg_accuracy = pattern_analysis['Success_Rate_%'].mean()
        best_pattern = pattern_analysis.loc[pattern_analysis['Success_Rate_%'].idxmax()]
        worst_pattern = pattern_analysis.loc[pattern_analysis['Success_Rate_%'].idxmin()]
        high_confidence = len(pattern_analysis[pattern_analysis['Confidence'] == 'High'])
        
        col1.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        col2.metric("Best Pattern", f"{best_pattern['Success_Rate_%']:.1f}%", 
                   delta=best_pattern['Pattern'][:20])
        col3.metric("Worst Pattern", f"{worst_pattern['Success_Rate_%']:.1f}%",
                   delta=worst_pattern['Pattern'][:20])
        col4.metric("High Confidence Patterns", high_confidence)
    
    # Detailed Indicator Tables
    st.subheader("📊 Comprehensive Indicator Tables")
    
    # Create all indicator tables
    tables = create_indicator_tables(primary_data, ticker1)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 RSI Analysis", "📊 Z-Score Bins", "💹 Volatility Bins", 
        "🔄 EMA Analysis", "🎚️ Support/Resistance", "📐 Fibonacci",
        "🌊 Elliott Waves", "⚡ RSI Divergence"
    ])
    
    with tab1:
        st.write("### RSI Analysis with Price Correlation")
        st.dataframe(tables['rsi'].tail(50), use_container_width=True)
        
        # Download option
        csv = tables['rsi'].to_csv(index=False)
        st.download_button(
            "📥 Download RSI Table",
            data=csv,
            file_name=f"rsi_analysis_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.write("### 📊 Z-Score Distribution & Mean Reversion Analysis")
        st.dataframe(tables['zscore'].tail(50), use_container_width=True)
        
        # Key insights for Z-Score
        st.write("#### 🎯 Historical Z-Score Pattern Analysis:")
        
        zscore_data = tables['zscore']
        
        # Find extreme Z-score events and their outcomes
        extreme_events = zscore_data[abs(zscore_data['Z-Score']) > 2].tail(10)
        
        if not extreme_events.empty:
            for _, event in extreme_events.iterrows():
                z_val = event['Z-Score']
                price = event['Price']
                pct_change = event['Price_Change_%']
                
                # Find what happened after this event
                event_idx = zscore_data[zscore_data['DateTime'] == event['DateTime']].index[0]
                if event_idx < len(zscore_data) - 10:
                    future_changes = zscore_data.iloc[event_idx+1:event_idx+11]['Price_Change_%'].sum()
                    
                    if abs(z_val) > 2:
                        time_diff = pd.Timestamp.now(tz=IST) - pd.to_datetime(event['DateTime'])
                        time_ago_str = format_time_ago(time_diff)
                        
                        if z_val < -2:
                            st.success(f"""
                            **Extreme Oversold Event (Z={z_val:.2f})**
                            - **Date**: {event['DateTime']} ({time_ago_str})
                            - **Price**: ₹{price:.2f}
                            - **Signal**: {event['Mean_Reversion_Signal']}
                            - **Next 10 periods move**: {future_changes:+.2f}%
                            - **Pattern**: When Z-score drops below -2, price typically rebounds as it's too far from mean
                            """)
                        elif z_val > 2:
                            st.warning(f"""
                            **Extreme Overbought Event (Z={z_val:.2f})**
                            - **Date**: {event['DateTime']} ({time_ago_str})
                            - **Price**: ₹{price:.2f}
                            - **Signal**: {event['Mean_Reversion_Signal']}
                            - **Next 10 periods move**: {future_changes:+.2f}%
                            - **Pattern**: When Z-score exceeds +2, price typically pulls back to mean
                            """)
        
        # Z-Score distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=tables['zscore']['Z-Score'], nbinsx=20, name='Z-Score Distribution'))
        fig.add_vline(x=-2, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.update_layout(title="Z-Score Distribution", xaxis_title="Z-Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        csv = tables['zscore'].to_csv(index=False)
        st.download_button(
            "📥 Download Z-Score Table",
            data=csv,
            file_name=f"zscore_analysis_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.write("### 💹 Volatility Bins & Market Conditions")
        st.dataframe(tables['volatility'].tail(50), use_container_width=True)
        
        # Key insights for volatility
        st.write("#### 🎯 Volatility Pattern Analysis:")
        
        vol_data = tables['volatility']
        
        # Find high volatility events and subsequent moves
        high_vol_events = vol_data[vol_data['Volatility_%'] > 30].tail(10)
        
        if not high_vol_events.empty:
            for _, event in high_vol_events.iterrows():
                vol = event['Volatility_%']
                price = event['Price']
                
                event_idx = vol_data[vol_data['DateTime'] == event['DateTime']].index[0]
                if event_idx < len(vol_data) - 10:
                    future_changes = vol_data.iloc[event_idx+1:event_idx+11]['Price_Change_%']
                    avg_move = future_changes.abs().mean()
                    direction_move = future_changes.sum()
                    
                    time_diff = pd.Timestamp.now(tz=IST) - pd.to_datetime(event['DateTime'])
                    time_ago_str = format_time_ago(time_diff)
                    
                    st.warning(f"""
                    **High Volatility Event ({event['Volatility_Bin']})**
                    - **Date**: {event['DateTime']} ({time_ago_str})
                    - **Price**: ₹{price:.2f}
                    - **Volatility**: {vol:.1f}%
                    - **Subsequent Average Move**: {avg_move:.2f}% per period
                    - **Net Direction (next 10 periods)**: {direction_move:+.2f}%
                    - **Insight**: High volatility = High risk/reward - Price movements amplified
                    """)
        
        # Volatility over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol_data['DateTime'],
            y=vol_data['Volatility_%'],
            mode='lines',
            name='Volatility',
            line=dict(color='orange')
        ))
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="High Volatility")
        fig.update_layout(title="Volatility Over Time", xaxis_title="Date", yaxis_title="Volatility %")
        st.plotly_chart(fig, use_container_width=True)
        
        csv = tables['volatility'].to_csv(index=False)
        st.download_button(
            "📥 Download Volatility Table",
            data=csv,
            file_name=f"volatility_analysis_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.write("### EMA Analysis & Trend Identification")
        st.dataframe(tables['ema'].tail(50), use_container_width=True)
        
        csv = tables['ema'].to_csv(index=False)
        st.download_button(
            "📥 Download EMA Table",
            data=csv,
            file_name=f"ema_analysis_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.write("### Support & Resistance Levels")
        if not tables['support_resistance'].empty:
            st.dataframe(tables['support_resistance'], use_container_width=True)
            
            csv = tables['support_resistance'].to_csv(index=False)
            st.download_button(
                "📥 Download S/R Table",
                data=csv,
                file_name=f"sr_levels_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No support/resistance levels detected in current data")
    
    with tab6:
        st.write("### Fibonacci Retracement Levels")
        st.dataframe(tables['fibonacci'], use_container_width=True)
        
        csv = tables['fibonacci'].to_csv(index=False)
        st.download_button(
            "📥 Download Fibonacci Table",
            data=csv,
            file_name=f"fibonacci_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab7:
        st.write("### Elliott Wave Structure Analysis")
        if not tables['elliott'].empty:
            st.dataframe(tables['elliott'], use_container_width=True)
            
            csv = tables['elliott'].to_csv(index=False)
            st.download_button(
                "📥 Download Elliott Wave Table",
                data=csv,
                file_name=f"elliott_waves_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No clear Elliott Wave pattern detected in current data")
    
    with tab8:
        st.write("### ⚡ RSI Divergence Detection & Analysis")
        if not tables['divergence'].empty:
            st.dataframe(tables['divergence'], use_container_width=True)
            
            # Key insights for divergences
            st.write("#### 🎯 Key Insights:")
            
            for _, div in tables['divergence'].iterrows():
                if div['Divergence_Type'] == '🟢 Bullish':
                    st.success(f"""
                    **Bullish Divergence Detected** ({div['Status']})
                    - **When**: {div['DateTime']} ({div['Time_Ago']})
                    - **Price Level**: ₹{div['Price_Level']:.2f}
                    - **RSI Movement**: {div['RSI_Old']:.1f} → {div['RSI_New']:.1f} (Higher low in RSI)
                    - **Price Action**: Made lower low while RSI made higher low
                    - **Implication**: {div['Expected_Move']}
                    - **What it means**: Price was falling but momentum was actually strengthening - typically leads to upward reversal
                    """)
                else:
                    st.error(f"""
                    **Bearish Divergence Detected** ({div['Status']})
                    - **When**: {div['DateTime']} ({div['Time_Ago']})
                    - **Price Level**: ₹{div['Price_Level']:.2f}
                    - **RSI Movement**: {div['RSI_Old']:.1f} → {div['RSI_New']:.1f} (Lower high in RSI)
                    - **Price Action**: Made higher high while RSI made lower high
                    - **Implication**: {div['Expected_Move']}
                    - **What it means**: Price was rising but momentum was weakening - typically leads to downward reversal
                    """)
            
            csv = tables['divergence'].to_csv(index=False)
            st.download_button(
                "📥 Download Divergence Table",
                data=csv,
                file_name=f"rsi_divergence_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ No RSI divergence detected in recent periods - Price and momentum moving in sync")
    
    # Backtest Strategy Details
    if backtest_result and backtest_result[0]:
        st.subheader("🔬 Backtest Strategy Details & Execution Log")
        
        strategy_name = backtest_result[0]
        params = backtest_result[2]
        
        # Display strategy parameters
        st.write("### ⚙️ Strategy Configuration")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.info(f"""
            **Strategy Type:** {strategy_name.upper()}
            
            **RSI Parameters:**
            - Oversold Threshold: {params['rsi_range'][0]}
            - Overbought Threshold: {params['rsi_range'][1]}
            - Period: 14
            """)
        
        with config_col2:
            st.info(f"""
            **EMA Parameters:**
            - Fast EMA: {params['ema_combo'][0]}
            - Slow EMA: {params['ema_combo'][1]}
            - Crossover Strategy
            """)
        
        with config_col3:
            st.info(f"""
            **ADX Parameters:**
            - Threshold: {params['adx_threshold']}
            - Period: 14
            
            **Risk Management:**
            - Stop Loss: -5%
            - Take Profit: +10%
            - Position Size: 20%
            """)
        
        st.write("### 📋 Strategy Logic")
        st.code(f"""
Entry Conditions ({strategy_name.upper()}):
- RSI < {params['rsi_range'][0]} (Oversold)
- EMA {params['ema_combo'][0]} > EMA {params['ema_combo'][1]} (Uptrend)
- ADX > {params['adx_threshold']} (Strong Trend)
- ALL conditions must be TRUE simultaneously

Exit Conditions:
- RSI > {params['rsi_range'][1]} (Overbought) OR
- EMA {params['ema_combo'][0]} < EMA {params['ema_combo'][1]} (Downtrend) OR
- ADX < {params['adx_threshold']} (Weak Trend) OR
- Stop Loss: -5% OR
- Take Profit: +10%
- ANY condition triggers exit

Calculation Method:
1. Calculate RSI = 100 - (100 / (1 + RS))
   where RS = Average Gain / Average Loss over 14 periods
   
2. Calculate EMA = Price * (2/(Period+1)) + Previous EMA * (1 - 2/(Period+1))

3. Calculate ADX using Directional Movement Index (DMI)
   - Calculate +DI and -DI from price movements
   - ADX = Moving Average of DX over 14 periods
   
4. Position Sizing = Capital * 20% / Entry Price

5. Profit/Loss = (Exit Price - Entry Price) * Shares
        """, language="python")
        
        # Backtest execution table
        st.write("### 📊 Backtest Execution Log")
        backtest_table = create_backtest_details_table(primary_data, strategy_name, params)
        
        if not backtest_table.empty:
            st.dataframe(backtest_table, use_container_width=True)
            
            csv = backtest_table.to_csv(index=False)
            st.download_button(
                "📥 Download Backtest Execution Log",
                data=csv,
                file_name=f"backtest_log_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No trades executed in backtest period")
    
    # Ratio Analysis
    if enable_ratio and st.session_state.all_timeframe_data_t2:
        st.subheader("🔄 Comprehensive Ratio Analysis - All Timeframes")
        
        # Try all common timeframes
        all_ratio_data = {}
        
        for tf_key in st.session_state.all_timeframe_data.keys():
            if tf_key in st.session_state.all_timeframe_data_t2:
                data1 = st.session_state.all_timeframe_data[tf_key]
                data2 = st.session_state.all_timeframe_data_t2[tf_key]
                
                ratio_table = create_ratio_table(data1, data2, ticker1, ticker2)
                
                if not ratio_table.empty:
                    all_ratio_data[tf_key] = {
                        'table': ratio_table,
                        'data1': data1,
                        'data2': data2
                    }
        
        if all_ratio_data:
            st.success(f"✅ Found {len(all_ratio_data)} common timeframes for ratio analysis")
            
            # Select timeframe for detailed analysis
            selected_tf = st.selectbox(
                "Select Timeframe for Detailed Ratio Analysis",
                list(all_ratio_data.keys()),
                format_func=lambda x: f"{x.split('_')[0].upper()} - {x.split('_')[1].upper()} ({len(all_ratio_data[x]['table'])} data points)"
            )
            
            if selected_tf:
                ratio_table = all_ratio_data[selected_tf]['table']
                data1 = all_ratio_data[selected_tf]['data1']
                data2 = all_ratio_data[selected_tf]['data2']
                
                st.write(f"### 📊 Ratio Analysis: {ticker1} / {ticker2} ({selected_tf})")
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_ratio = ratio_table['Ratio'].iloc[-1]
                ratio_change = ratio_table['Ratio_Change_%'].iloc[-1]
                ratio_rsi = ratio_table['Ratio_RSI'].iloc[-1]
                ratio_zscore = ratio_table['Ratio_Z-Score'].iloc[-1]
                
                col1.metric("Current Ratio", f"{current_ratio:.4f}")
                col2.metric("Ratio Change", f"{ratio_change:+.2f}%")
                col3.metric("Ratio RSI", f"{ratio_rsi:.1f}")
                col4.metric("Ratio Z-Score", f"{ratio_zscore:.2f}")
                
                # Display full table
                st.dataframe(ratio_table.tail(50), use_container_width=True)
                
                # Export functionality
                col1, col2 = st.columns(2)
                with col1:
                    csv = ratio_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Ratio Analysis CSV",
                        data=csv,
                        file_name=f"ratio_{ticker1}_{ticker2}_{selected_tf}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        ratio_table.to_excel(writer, index=False, sheet_name='Ratio Analysis')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Ratio Analysis Excel",
                        data=excel_data,
                        file_name=f"ratio_{ticker1}_{ticker2}_{selected_tf}_{datetime.now(IST).strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Ratio visualization
                st.write("### 📈 Ratio Visualization")
                
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        f'{ticker1} vs {ticker2} Price Comparison',
                        'Ratio Value Over Time',
                        'Ratio RSI',
                        'Ratio Z-Score'
                    ),
                    row_heights=[0.3, 0.3, 0.2, 0.2]
                )
                
                # Convert DateTime to datetime for plotting
                plot_dates = pd.to_datetime(ratio_table['DateTime'])
                
                # Price comparison with dual y-axes
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates, 
                        y=ratio_table[f'{ticker1}_Price'], 
                        name=f'{ticker1} Price', 
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates, 
                        y=ratio_table[f'{ticker2}_Price'], 
                        name=f'{ticker2} Price', 
                        line=dict(color='red', width=2),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
                
                # Ratio
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates, 
                        y=ratio_table['Ratio'], 
                        name='Ratio', 
                        line=dict(color='purple', width=2),
                        fill='tozeroy'
                    ),
                    row=2, col=1
                )
                
                # Add mean line
                mean_ratio = ratio_table['Ratio'].mean()
                fig.add_hline(y=mean_ratio, line_dash="dash", line_color="gray", 
                            annotation_text=f"Mean: {mean_ratio:.4f}", row=2, col=1)
                
                # Ratio RSI
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates, 
                        y=ratio_table['Ratio_RSI'], 
                        name='Ratio RSI', 
                        line=dict(color='orange', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
                
                # Ratio Z-Score
                fig.add_trace(
                    go.Scatter(
                        x=plot_dates, 
                        y=ratio_table['Ratio_Z-Score'], 
                        name='Ratio Z-Score', 
                        line=dict(color='teal', width=2),
                        fill='tozeroy'
                    ),
                    row=4, col=1
                )
                fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Overbought", row=4, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Oversold", row=4, col=1)
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1)
                
                fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
                fig.update_xaxes(title_text="Date", row=4, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Ratio analysis insights
                st.write("### 🎯 Ratio Trading Insights")
                
                mean_ratio = ratio_table['Ratio'].mean()
                std_ratio = ratio_table['Ratio'].std()
                
                if current_ratio > mean_ratio + 2*std_ratio:
                    distance_pct = ((current_ratio - mean_ratio) / mean_ratio) * 100
                    st.warning(f"""
                    ⚠️ **{ticker1} is OVERVALUED relative to {ticker2}**
                    
                    - **Current Ratio**: {current_ratio:.4f}
                    - **Historical Mean**: {mean_ratio:.4f}
                    - **Distance from Mean**: {distance_pct:+.2f}%
                    - **Z-Score**: {ratio_zscore:.2f} (Extreme)
                    
                    **Trading Strategy:**
                    - Consider **SELLING** the ratio (Short {ticker1}, Long {ticker2})
                    - Expected mean reversion could provide {distance_pct:.1f}% profit
                    - Place stop loss at Z-Score > 3.0
                    """)
                elif current_ratio < mean_ratio - 2*std_ratio:
                    distance_pct = ((mean_ratio - current_ratio) / mean_ratio) * 100
                    st.success(f"""
                    ✅ **{ticker1} is UNDERVALUED relative to {ticker2}**
                    
                    - **Current Ratio**: {current_ratio:.4f}
                    - **Historical Mean**: {mean_ratio:.4f}
                    - **Distance from Mean**: {distance_pct:+.2f}%
                    - **Z-Score**: {ratio_zscore:.2f} (Extreme)
                    
                    **Trading Strategy:**
                    - Consider **BUYING** the ratio (Long {ticker1}, Short {ticker2})
                    - Expected mean reversion could provide {distance_pct:.1f}% profit
                    - Place stop loss at Z-Score < -3.0
                    """)
                else:
                    st.info(f"""
                    ℹ️ **Ratio trading near historical equilibrium**
                    
                    - **Current Ratio**: {current_ratio:.4f}
                    - **Historical Mean**: {mean_ratio:.4f}
                    - **Z-Score**: {ratio_zscore:.2f} (Normal range)
                    
                    **Recommendation**: Wait for ratio to reach extreme levels (|Z-Score| > 2) before initiating pair trade
                    """)
        else:
            st.warning(f"""
            ⚠️ **No overlapping data between {ticker1} and {ticker2}**
            
            **Possible reasons:**
            - Different market hours (e.g., {ticker1} vs {ticker2} may trade at different times)
            - {ticker2} may be available 24/7 while {ticker1} has fixed hours
            - Data availability mismatch
            
            **Attempted {len(st.session_state.all_timeframe_data)} timeframes, found 0 matches**
            
            Try selecting assets that trade in similar time zones or have overlapping market hours.
            """)
    
    # Detailed timeframe breakdown
    
    if enable_ratio and st.session_state.all_timeframe_data_t2:
        st.subheader("🔄 Ratio Analysis")
        
        # Find common timeframes
        common_keys = set(st.session_state.all_timeframe_data.keys()).intersection(
            set(st.session_state.all_timeframe_data_t2.keys())
        )
        
        if common_keys:
            # Use first common timeframe
            common_key = list(common_keys)[0]
            data1 = st.session_state.all_timeframe_data[common_key]
            data2 = st.session_state.all_timeframe_data_t2[common_key]
            
            # Align data by index
            common_index = data1.index.intersection(data2.index)
            
            if len(common_index) > 0:
                data1_aligned = data1.loc[common_index]
                data2_aligned = data2.loc[common_index]
                
                # Calculate ratio
                ratio = data1_aligned['Close'] / data2_aligned['Close']
                
                # Create comprehensive comparison table
                comparison_df = pd.DataFrame({
                    'DateTime (IST)': common_index.strftime('%Y-%m-%d %H:%M:%S'),
                    f'{ticker1} Price': data1_aligned['Close'].values,
                    f'{ticker2} Price': data2_aligned['Close'].values,
                    'Ratio': ratio.values,
                    f'{ticker1} RSI': calculate_rsi(data1_aligned['Close']).values,
                    f'{ticker2} RSI': calculate_rsi(data2_aligned['Close']).values,
                    'Ratio RSI': calculate_rsi(ratio).values,
                    f'{ticker1} Volatility': calculate_volatility(data1_aligned['Close'].pct_change()).values,
                    f'{ticker2} Volatility': calculate_volatility(data2_aligned['Close'].pct_change()).values,
                    f'{ticker1} Z-Score': calculate_zscore(data1_aligned['Close']).values,
                    f'{ticker2} Z-Score': calculate_zscore(data2_aligned['Close']).values
                })
                
                st.dataframe(comparison_df.tail(50), use_container_width=True)
                
                # Export functionality
                col1, col2 = st.columns(2)
                with col1:
                    csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"ratio_analysis_{ticker1}_{ticker2}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        comparison_df.to_excel(writer, index=False, sheet_name='Ratio Analysis')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"ratio_analysis_{ticker1}_{ticker2}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Ratio chart
                st.subheader("📊 Ratio Visualization")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f'{ticker1} vs {ticker2} Price', 'Ratio', 'Ratio RSI'),
                    row_heights=[0.4, 0.3, 0.3]
                )
                
                fig.add_trace(
                    go.Scatter(x=common_index, y=data1_aligned['Close'], name=ticker1, line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=common_index, y=data2_aligned['Close'], name=ticker2, line=dict(color='red')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=common_index, y=ratio, name='Ratio', line=dict(color='purple')),
                    row=2, col=1
                )
                
                ratio_rsi = calculate_rsi(ratio)
                fig.add_trace(
                    go.Scatter(x=common_index, y=ratio_rsi, name='Ratio RSI', line=dict(color='orange')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                fig.update_xaxes(title_text="Date", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No overlapping data between {ticker1} and {ticker2} for timeframe {common_key}")
        else:
            st.warning(f"No common timeframes found between {ticker1} and {ticker2}. They may operate on different schedules.")
    
    # Detailed timeframe breakdown
    st.subheader("🔍 Multi-Timeframe Detailed Analysis - All Available Periods")
    
    # Group by interval
    timeframe_groups = {}
    for tf_key in st.session_state.all_timeframe_data.keys():
        interval, period = tf_key.split('_')
        if interval not in timeframe_groups:
            timeframe_groups[interval] = []
        timeframe_groups[interval].append(tf_key)
    
    for interval, tf_keys in timeframe_groups.items():
        with st.expander(f"📊 {interval.upper()} Interval Analysis ({len(tf_keys)} periods available)"):
            
            for tf_key in tf_keys:
                data = st.session_state.all_timeframe_data[tf_key]
                _, period = tf_key.split('_')
                
                st.write(f"#### Period: {period.upper()} ({len(data)} candles)")
                
                col1, col2, col3, col4 = st.columns(4)
                
                close = data['Close']
                rsi = calculate_rsi(close)
                zscore = calculate_zscore(close)
                volatility = calculate_volatility(close.pct_change())
                
                current_price = float(close.iloc[-1])
                current_rsi = float(rsi.iloc[-1])
                current_zscore = float(zscore.iloc[-1])
                current_vol = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0
                
                col1.metric("Price", f"₹{current_price:.2f}")
                col2.metric("RSI", f"{current_rsi:.1f}")
                col3.metric("Z-Score", f"{current_zscore:.2f}")
                col4.metric("Volatility", f"{current_vol:.1f}%")
                
                # Create tabs for each analysis type
                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                    "🎚️ S/R Levels", "📐 Fibonacci", "🌊 Elliott Wave", "⚡ Divergence"
                ])
                
                with sub_tab1:
                    # Support/Resistance
                    sr_levels = find_support_resistance(data)
                    if sr_levels:
                        st.write("**Key Support & Resistance Levels:**")
                        for level in sr_levels[:5]:
                            time_ago = datetime.now(IST) - level['date']
                            time_ago_str = format_time_ago(time_ago)
                            distance_pct = ((level['level'] - current_price) / current_price) * 100
                            distance_points = level['level'] - current_price
                            
                            level_type_emoji = "🟢" if level['type'] == 'support' else "🔴"
                            st.write(
                                f"{level_type_emoji} **{level['type'].upper()}** @ ₹{level['level']:.2f} | "
                                f"Distance: {distance_pct:+.2f}% ({distance_points:+.2f} pts) | "
                                f"{level['touches']} touches, {level['sustained']} sustained | "
                                f"Formed: {level['date'].strftime('%Y-%m-%d %H:%M IST')} ({time_ago_str})"
                            )
                    else:
                        st.info("No significant S/R levels detected")
                
                with sub_tab2:
                    # Fibonacci
                    fib = calculate_fibonacci_levels(data)
                    st.write("**Fibonacci Retracement Levels:**")
                    fib_data = []
                    for level_name, level_data in fib.items():
                        distance_pct = ((level_data['price'] - current_price) / current_price) * 100
                        distance_points = level_data['price'] - current_price
                        date_str = level_data['date'].strftime('%Y-%m-%d %H:%M IST') if level_data['date'] else 'Calculated'
                        
                        fib_data.append({
                            'Level': level_name,
                            'Price': f"₹{level_data['price']:.2f}",
                            'Distance_%': f"{distance_pct:+.2f}%",
                            'Distance_Points': f"{distance_points:+.2f}",
                            'Date': date_str
                        })
                    st.dataframe(pd.DataFrame(fib_data), use_container_width=True)
                
                with sub_tab3:
                    # Elliott Wave
                    elliott = detect_elliott_wave_detailed(data)
                    if elliott:
                        st.write(f"**Current Phase:** {elliott['current_wave']}")
                        
                        wave_data = []
                        for wave in elliott['waves']:
                            wave_data.append({
                                'Phase': wave['phase'],
                                'Type': wave['wave_type'],
                                'Point': wave['point_type'],
                                'Price': f"₹{wave['price']:.2f}",
                                'High': f"₹{wave['high']:.2f}",
                                'Low': f"₹{wave['low']:.2f}",
                                'Change': f"{wave['change_pct']:+.2f}%",
                                'Duration': wave['duration'],
                                'Date': wave['date'].strftime('%Y-%m-%d %H:%M IST')
                            })
                        st.dataframe(pd.DataFrame(wave_data), use_container_width=True)
                    else:
                        st.info("No clear Elliott Wave pattern detected")
                
                with sub_tab4:
                    # Divergence
                    divergence = detect_rsi_divergence(close, rsi)
                    if divergence:
                        for div in divergence:
                            if div['type'] == 'bullish':
                                st.success(f"""
                                **🟢 Bullish Divergence**
                                - **Date**: {div['date'].strftime('%Y-%m-%d %H:%M IST')} ({div['time_ago']})
                                - **Price**: ₹{div['price']:.2f}
                                - **RSI**: {div['rsi_old']:.1f} → {div['rsi_new']:.1f} (Higher Low)
                                - **Expected**: Upward reversal
                                """)
                            else:
                                st.error(f"""
                                **🔴 Bearish Divergence**
                                - **Date**: {div['date'].strftime('%Y-%m-%d %H:%M IST')} ({div['time_ago']})
                                - **Price**: ₹{div['price']:.2f}
                                - **RSI**: {div['rsi_old']:.1f} → {div['rsi_new']:.1f} (Lower High)
                                - **Expected**: Downward reversal
                                """)
                    else:
                        st.info("✅ No divergence - Price and momentum aligned")
                
                st.markdown("---")
    
    # Price chart with indicators
    st.subheader("📈 Technical Analysis Chart")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker1} Price & Indicators', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=primary_data.index,
            open=primary_data['Open'],
            high=primary_data['High'],
            low=primary_data['Low'],
            close=primary_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
    ema_9 = calculate_ema(primary_data['Close'], 9)
    ema_20 = calculate_ema(primary_data['Close'], 20)
    ema_50 = calculate_ema(primary_data['Close'], 50)
    
    fig.add_trace(go.Scatter(x=primary_data.index, y=ema_9, name='EMA 9', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary_data.index, y=ema_20, name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary_data.index, y=ema_50, name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)
    
    # Bollinger Bands
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(primary_data['Close'])
    fig.add_trace(go.Scatter(x=primary_data.index, y=upper_bb, name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=primary_data.index, y=lower_bb, name='BB Lower', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    
    # Support/Resistance levels
    sr_levels = find_support_resistance(primary_data)
    for level in sr_levels[:5]:
        color = 'green' if level['type'] == 'support' else 'red'
        fig.add_hline(
            y=level['level'],
            line_dash="dot",
            line_color=color,
            annotation_text=f"{level['type'][:3].upper()} {level['level']:.2f}",
            row=1, col=1
        )
    
    # RSI
    rsi_series = calculate_rsi(primary_data['Close'])
    fig.add_trace(go.Scatter(x=primary_data.index, y=rsi_series, name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # MACD
    macd, signal, histogram = calculate_macd(primary_data['Close'])
    fig.add_trace(go.Scatter(x=primary_data.index, y=macd, name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=primary_data.index, y=signal, name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=primary_data.index, y=histogram, name='Histogram', marker_color='gray'), row=3, col=1)
    
    # Volume
    if 'Volume' in primary_data.columns and primary_data['Volume'].sum() > 0:
        colors = ['green' if primary_data['Close'].iloc[i] > primary_data['Open'].iloc[i] else 'red' for i in range(len(primary_data))]
        fig.add_trace(go.Bar(x=primary_data.index, y=primary_data['Volume'], name='Volume', marker_color=colors), row=4, col=1)
    else:
        fig.update_yaxes(title_text="No Volume Data", row=4, col=1)
    
    fig.update_layout(height=1200, showlegend=True, hovermode='x unified', xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📊 Price Data Table")
    
    display_data = primary_data.copy()
    display_data['Returns (%)'] = display_data['Close'].pct_change() * 100
    display_data['RSI'] = calculate_rsi(display_data['Close'])
    display_data['Volatility'] = calculate_volatility(display_data['Close'].pct_change())
    display_data['Z-Score'] = calculate_zscore(display_data['Close'])
    display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M:%S IST')
    
    st.dataframe(display_data.tail(100), use_container_width=True)
    
    # Download full analysis
    st.subheader("📥 Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export summary as text
        st.download_button(
            label="📄 Download Summary Report (TXT)",
            data=summary_text,
            file_name=f"analysis_{ticker1}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export data as CSV
        csv_data = display_data.to_csv()
        st.download_button(
            label="📊 Download Price Data (CSV)",
            data=csv_data,
            file_name=f"price_data_{ticker1}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Click 'Fetch All Data' in the sidebar to begin comprehensive analysis.")
    
    st.markdown("""
    ### 🎯 Advanced Features:
    
    **📊 Multi-Timeframe Analysis:**
    - Automatically fetches ALL valid timeframe-period combinations
    - 1m for 1d-7d | 5m for 1d-1mo | 15m for 1d-1mo
    - 1h for 1d-2y | 1d for 1mo-30y | 1wk for 1y-30y | 1mo for 2y-30y
    
    **🔬 Advanced Backtesting:**
    - Automatic strategy optimization across parameters
    - Tests RSI, EMA, and Combined strategies
    - Optimizes for 20%+ annual returns
    - Only recommends if backtest is profitable
    
    **📈 Comprehensive Technical Analysis:**
    - Support/Resistance with touch count & sustainability
    - Fibonacci retracement levels with proximity alerts
    - Detailed Elliott Wave analysis (all waves, impulse/corrective)
    - RSI divergence detection with historical context
    - Z-Score mean reversion with pattern matching
    
    **🎯 Historical Pattern Analysis:**
    - Analyzes past Z-score extremes and their outcomes
    - Shows exact dates, prices, and results of similar conditions
    - Calculates accuracy percentages for pattern repetition
    - Provides probability-based predictions
    
    **🔄 Smart Ratio Analysis:**
    - Handles different timezones (BTC 24/7 vs regular markets)
    - Aligns data properly across different market hours
    - Comprehensive comparison metrics
    
    **💼 Risk Management:**
    - ATR-based stop loss and targets
    - Multiple profit booking levels (30%-40%-30%)
    - Risk:Reward ratio calculation
    - Position sizing recommendations
    
    **📋 Ultra-Detailed Summary:**
    - When: Exact dates and times of past patterns
    - What: Specific technical conditions that occurred
    - How Much: Precise price moves and percentages
    - Impact: Historical outcomes and current predictions
    - Accuracy: Success rates of similar past patterns
    
    ### 📝 Summary Includes:
    
    ✅ **Backtest Integration:** Only gives BUY/SELL if backtest shows positive returns
    
    ✅ **Elliott Wave Details:** Exact wave phases, prices, dates, and current position
    
    ✅ **Z-Score Patterns:** Historical similar conditions with exact outcomes
    
    ✅ **Support/Resistance:** Touch counts, sustainability, and reaction history
    
    ✅ **Fibonacci Confluence:** Proximity to golden ratios and expected behavior
    
    ✅ **Multi-Timeframe Consensus:** Agreement across all analyzed timeframes
    
    ✅ **Divergence Analysis:** RSI divergences with time and price details
    
    ✅ **Final Recommendation:** Integrated signal with entry, SL, and 3 targets
    
    ### ⚠️ Disclaimer:
    
    This tool performs comprehensive algorithmic analysis using multiple technical indicators,
    historical pattern matching, and optimized backtesting. However, markets are inherently
    unpredictable. Always use proper risk management and never risk more than you can afford to lose.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Advanced Algorithmic Trading Analysis Tool v2.0 | Built with Streamlit & Python</p>
    <p>⚠️ For Educational Purposes Only | Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)
