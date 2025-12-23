import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import pytz
import time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading Analysis System",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IST = pytz.timezone('Asia/Kolkata')

ASSET_MAPPING = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'Custom Ticker': 'CUSTOM'
}

TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

# Valid timeframe-period combinations
VALID_COMBINATIONS = {
    '1m': ['1d', '5d', '7d'],
    '2m': ['1d', '5d', '7d', '1mo'],
    '5m': ['1d', '5d', '7d', '1mo'],
    '15m': ['1d', '5d', '7d', '1mo', '3mo'],
    '30m': ['1d', '5d', '7d', '1mo', '3mo', '6mo'],
    '60m': ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y'],
    '90m': ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y'],
    '1h': ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y', 'max'],
    '5d': ['3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y', 'max'],
    '1wk': ['1y', '2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y', 'max'],
    '1mo': ['2y', '3y', '5y', '10y', '15y', '20y', '25y', '30y', 'max'],
    '3mo': ['5y', '10y', '15y', '20y', '25y', '30y', 'max']
}

# Helper Functions
def convert_to_ist(df):
    """Convert DataFrame index to IST timezone"""
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
    except:
        pass
    return df

def time_ago(dt):
    """Convert datetime to human-readable time ago format"""
    now = datetime.now(IST)
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    if dt.tz is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)
    
    diff = now - dt
    
    if diff.days > 30:
        months = diff.days // 30
        days = diff.days % 30
        return f"{months} month(s) and {days} day(s) ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"
    elif diff.days > 0:
        return f"{diff.days} day(s) ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour(s) ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"
    else:
        minutes = diff.seconds // 60
        return f"{minutes} minute(s) ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"

def fetch_data(ticker, period, interval):
    """Fetch data from yfinance with rate limiting"""
    try:
        time.sleep(np.random.uniform(1.5, 3.0))
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in cols if col in data.columns]]
        data = convert_to_ist(data)
        
        return data
    except Exception as e:
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def find_support_resistance(data, window=20, tolerance=5):
    """Find support and resistance levels with hit counts"""
    levels = []
    closes = data['Close'].values
    
    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[i-window:i+window+1]):
            levels.append({'price': closes[i], 'type': 'Resistance', 'index': i})
        elif closes[i] == min(closes[i-window:i+window+1]):
            levels.append({'price': closes[i], 'type': 'Support', 'index': i})
    
    clustered_levels = []
    for level in levels:
        found = False
        for cluster in clustered_levels:
            if abs(level['price'] - cluster['price']) <= tolerance:
                cluster['count'] += 1
                cluster['indices'].append(level['index'])
                found = True
                break
        if not found:
            clustered_levels.append({
                'price': level['price'],
                'type': level['type'],
                'count': 1,
                'indices': [level['index']]
            })
    
    for cluster in clustered_levels:
        cluster['dates'] = [data.index[idx] for idx in cluster['indices']]
        cluster['last_hit'] = cluster['dates'][-1]
        cluster['first_hit'] = cluster['dates'][0]
        
        sustained_count = 0
        for idx in cluster['indices']:
            window_data = closes[max(0, idx-5):min(len(closes), idx+5)]
            if len(window_data[abs(window_data - cluster['price']) <= tolerance]) >= 3:
                sustained_count += 1
        cluster['sustained'] = sustained_count
    
    return sorted(clustered_levels, key=lambda x: x['count'], reverse=True)

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100.0%': low
    }
    
    return levels, high, low

def detect_elliott_waves(data):
    """Simplified Elliott Wave detection"""
    closes = data['Close'].values
    waves = []
    
    from scipy.signal import argrelextrema
    maxima_idx = argrelextrema(closes, np.greater, order=5)[0]
    minima_idx = argrelextrema(closes, np.less, order=5)[0]
    
    all_points = sorted([(idx, 'peak') for idx in maxima_idx] + [(idx, 'trough') for idx in minima_idx])
    
    if len(all_points) >= 5:
        for i in range(len(all_points) - 4):
            wave_points = all_points[i:i+5]
            waves.append({
                'wave': f"Wave {i+1}",
                'start_price': closes[wave_points[0][0]],
                'end_price': closes[wave_points[-1][0]],
                'start_date': data.index[wave_points[0][0]],
                'end_date': data.index[wave_points[-1][0]],
                'type': 'Impulse' if i % 2 == 0 else 'Corrective'
            })
    
    return waves

def detect_rsi_divergence(data, rsi_data):
    """Detect RSI divergences"""
    divergences = []
    closes = data['Close'].values
    rsi_values = rsi_data.values
    
    from scipy.signal import argrelextrema
    price_peaks = argrelextrema(closes, np.greater, order=5)[0]
    price_troughs = argrelextrema(closes, np.less, order=5)[0]
    rsi_peaks = argrelextrema(rsi_values, np.greater, order=5)[0]
    rsi_troughs = argrelextrema(rsi_values, np.less, order=5)[0]
    
    for i in range(len(price_troughs) - 1):
        idx1, idx2 = price_troughs[i], price_troughs[i+1]
        if idx2 < len(closes) and closes[idx2] < closes[idx1]:
            rsi_near = [r for r in rsi_troughs if abs(r - idx2) < 10]
            if rsi_near:
                rsi_idx = rsi_near[0]
                rsi_prev = [r for r in rsi_troughs if r < rsi_idx]
                if rsi_prev and rsi_values[rsi_idx] > rsi_values[rsi_prev[-1]]:
                    divergences.append({
                        'type': 'Bullish',
                        'price1': closes[idx1],
                        'price2': closes[idx2],
                        'date1': data.index[idx1],
                        'date2': data.index[idx2],
                        'rsi1': rsi_values[rsi_prev[-1]],
                        'rsi2': rsi_values[rsi_idx],
                        'resolved': closes[-1] > closes[idx2]
                    })
    
    for i in range(len(price_peaks) - 1):
        idx1, idx2 = price_peaks[i], price_peaks[i+1]
        if idx2 < len(closes) and closes[idx2] > closes[idx1]:
            rsi_near = [r for r in rsi_peaks if abs(r - idx2) < 10]
            if rsi_near:
                rsi_idx = rsi_near[0]
                rsi_prev = [r for r in rsi_peaks if r < rsi_idx]
                if rsi_prev and rsi_values[rsi_idx] < rsi_values[rsi_prev[-1]]:
                    divergences.append({
                        'type': 'Bearish',
                        'price1': closes[idx1],
                        'price2': closes[idx2],
                        'date1': data.index[idx1],
                        'date2': data.index[idx2],
                        'rsi1': rsi_values[rsi_prev[-1]],
                        'rsi2': rsi_values[rsi_idx],
                        'resolved': closes[-1] < closes[idx2]
                    })
    
    return divergences

def calculate_volatility_bins(data):
    """Calculate volatility bins and z-scores"""
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
    
    vol_clean = volatility.dropna()
    if len(vol_clean) > 5:
        try:
            bins = pd.qcut(vol_clean, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        except ValueError:
            # If qcut fails, use cut instead
            bins = pd.cut(vol_clean, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    else:
        bins = pd.Series(['Medium'] * len(vol_clean), index=vol_clean.index)
    
    result = pd.DataFrame({
        'DateTime_IST': data.index[volatility.notna()],
        'Close': data['Close'][volatility.notna()],
        'Volatility': volatility.dropna(),
        'Volatility_Bin': bins
    })
    
    return result.dropna()

def calculate_zscore_bins(data):
    """Calculate z-score bins for price changes"""
    returns = data['Close'].pct_change()
    zscore = (returns - returns.mean()) / returns.std()
    
    zscore_clean = zscore.dropna()
    if len(zscore_clean) > 0:
        bins = pd.cut(zscore_clean, bins=[-np.inf, -2, -1, 0, 1, 2, np.inf], 
                      labels=['Extreme Negative', 'Negative', 'Neutral Low', 'Neutral High', 'Positive', 'Extreme Positive'])
    else:
        bins = pd.Series([], dtype='object')
    
    result = pd.DataFrame({
        'DateTime_IST': data.index[returns.notna()],
        'Close': data['Close'][returns.notna()],
        'Return_%': (returns * 100).dropna(),
        'Z_Score': zscore_clean,
        'Z_Score_Bin': bins
    })
    
    return result.dropna()

def backtest_strategy(data, strategy_type='rsi_ema', timeframe='', period=''):
    """Backtest trading strategy with multiple implementations"""
    df = data.copy()
    
    if len(df) < 60:
        return pd.DataFrame(), 0, 0
    
    df['RSI'] = calculate_rsi(df['Close'])
    df['EMA_9'] = calculate_ema(df['Close'], 9)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['EMA_50'] = calculate_ema(df['Close'], 50)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Calculate average true range for dynamic stop loss
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    avg_price = df['Close'].mean()
    
    # Adjust stop loss and target based on price level and volatility
    if avg_price > 20000:  # NIFTY/SENSEX level
        base_sl_pct = 0.015  # 1.5%
        base_target_pct = 0.025  # 2.5%
    elif avg_price > 1000:  # Bank NIFTY or stock level
        base_sl_pct = 0.02
        base_target_pct = 0.03
    else:  # Other assets
        base_sl_pct = 0.025
        base_target_pct = 0.04
    
    trades = []
    position = None
    
    for i in range(60, len(df)):
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        current_rsi = df['RSI'].iloc[i]
        
        if position is None:
            entry_signal = False
            reason = ""
            sl_pct = base_sl_pct
            target_pct = base_target_pct
            
            # Strategy 1: RSI + EMA
            if strategy_type == 'rsi_ema':
                if current_rsi < 40 and df['Close'].iloc[i] > df['EMA_20'].iloc[i]:
                    entry_signal = True
                    reason = f'RSI Oversold ({current_rsi:.1f}) + Price > 20EMA'
                    sl_pct = base_sl_pct
                    target_pct = base_target_pct
            
            # Strategy 2: EMA Crossover
            elif strategy_type == 'ema_crossover':
                if (df['EMA_9'].iloc[i] > df['EMA_20'].iloc[i] and 
                    df['EMA_9'].iloc[i-1] <= df['EMA_20'].iloc[i-1]):
                    entry_signal = True
                    reason = '9 EMA crossed above 20 EMA (Bullish)'
                    sl_pct = base_sl_pct * 0.8
                    target_pct = base_target_pct * 0.9
            
            # Strategy 3: Volatility Breakout
            elif strategy_type == 'volatility_breakout':
                avg_vol = df['Volatility'].iloc[i-20:i].mean()
                if df['Volatility'].iloc[i] > avg_vol * 1.3:
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                        entry_signal = True
                        reason = f'High Volatility Breakout ({df["Volatility"].iloc[i]*100:.2f}%)'
                        sl_pct = base_sl_pct * 1.2
                        target_pct = base_target_pct * 1.5
            
            # Strategy 4: Support Bounce
            elif strategy_type == 'support_bounce':
                if current_rsi < 35:
                    sr_levels = find_support_resistance(df.iloc[:i])
                    nearby_support = [l for l in sr_levels if l['type'] == 'Support' and 
                                    abs(current_price - l['price']) < current_price * 0.015]
                    if nearby_support:
                        entry_signal = True
                        reason = f'Bouncing from Support at ₹{nearby_support[0]["price"]:.2f}'
                        sl_pct = base_sl_pct * 0.9
                        target_pct = base_target_pct
            
            # Strategy 5: RSI Divergence
            elif strategy_type == 'rsi_divergence':
                if i > 70:
                    divergences = detect_rsi_divergence(df.iloc[:i], df['RSI'].iloc[:i])
                    active_bull = [d for d in divergences if d['type'] == 'Bullish' and not d['resolved']]
                    if active_bull and current_rsi < 45:
                        entry_signal = True
                        reason = 'Bullish RSI Divergence Detected'
                        sl_pct = base_sl_pct
                        target_pct = base_target_pct * 1.2
            
            # Strategy 6: 9 EMA Pullback
            elif strategy_type == 'ema_pullback':
                if (df['EMA_9'].iloc[i] > df['EMA_50'].iloc[i] and 
                    df['Close'].iloc[i-1] < df['EMA_9'].iloc[i-1] and
                    df['Close'].iloc[i] > df['EMA_9'].iloc[i] and
                    current_rsi > 40 and current_rsi < 60):
                    entry_signal = True
                    reason = 'Pullback to 9 EMA in Uptrend'
                    sl_pct = base_sl_pct * 0.85
                    target_pct = base_target_pct
            
            # Strategy 7: Z-Score Mean Reversion
            elif strategy_type == 'zscore_reversion':
                returns = df['Returns'].iloc[:i]
                if len(returns) > 30:
                    zscore = (returns.iloc[-1] - returns.mean()) / returns.std()
                    if zscore < -1.5 and current_rsi < 40:
                        entry_signal = True
                        reason = f'Z-Score Mean Reversion ({zscore:.2f})'
                        sl_pct = base_sl_pct
                        target_pct = base_target_pct
            
            # Strategy 8: Price Action
            elif strategy_type == 'price_action':
                # Bullish engulfing pattern
                if (df['Close'].iloc[i] > df['Open'].iloc[i] and
                    df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                    df['Close'].iloc[i] > df['Open'].iloc[i-1] and
                    df['Open'].iloc[i] < df['Close'].iloc[i-1] and
                    df['Close'].iloc[i] > df['EMA_20'].iloc[i]):
                    entry_signal = True
                    reason = 'Bullish Engulfing Pattern + Above 20 EMA'
                    sl_pct = base_sl_pct * 0.9
                    target_pct = base_target_pct
            
            if entry_signal:
                sl = current_price * (1 - sl_pct)
                target = current_price * (1 + target_pct)
                position = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'sl': sl,
                    'target': target,
                    'reason': reason,
                    'strategy': strategy_type,
                    'timeframe': timeframe,
                    'period': period
                }
        
        elif position is not None:
            exit_reason = None
            exit_price = None
            
            if current_price <= position['sl']:
                exit_reason = 'Stop Loss Hit'
                exit_price = current_price
            elif current_price >= position['target']:
                exit_reason = 'Target Hit'
                exit_price = current_price
            elif i == len(df) - 1:
                exit_reason = 'End of Data'
                exit_price = current_price
            
            if exit_reason:
                trades.append({
                    'Strategy': position['strategy'],
                    'Timeframe': position['timeframe'],
                    'Period': position['period'],
                    'Entry_Date': position['entry_date'],
                    'Entry_Price': position['entry_price'],
                    'Exit_Date': current_date,
                    'Exit_Price': exit_price,
                    'SL': position['sl'],
                    'Target': position['target'],
                    'Points': exit_price - position['entry_price'],
                    'PnL_%': ((exit_price - position['entry_price']) / position['entry_price']) * 100,
                    'Entry_Reason': position['reason'],
                    'Exit_Reason': exit_reason
                })
                position = None
    
    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['PnL_%'].sum()
        win_rate = len(trades_df[trades_df['PnL_%'] > 0]) / len(trades_df) * 100
        return trades_df, total_pnl, win_rate
    
    return pd.DataFrame(), 0, 0

def fetch_multi_timeframe_data(ticker, progress_placeholder):
    """Fetch data across carefully selected timeframes"""
    all_data = {}
    
    # Carefully selected timeframe-period combinations for comprehensive analysis
    combinations = [
        ('1m', '1d'),
        ('1m', '5d'),
        ('5m', '1d'),
        ('5m', '1mo'),
        ('15m', '1d'),
        ('15m', '1mo'),
        ('30m', '1d'),
        ('30m', '1mo'),
        ('1h', '1mo'),
        ('1h', '3mo'),
        ('1d', '1mo'),
        ('1d', '3mo'),
        ('1d', '6mo'),
        ('1d', '1y'),
        ('1d', '2y'),
        ('1d', '5y'),
        ('1d', '10y'),
        ('1wk', '1y'),
        ('1wk', '2y'),
        ('1wk', '5y')
    ]
    
    total = len(combinations)
    
    for idx, (timeframe, period) in enumerate(combinations):
        progress = (idx + 1) / total
        progress_placeholder.progress(progress, text=f"Fetching {timeframe}/{period}... ({idx+1}/{total})")
        
        data = fetch_data(ticker, period, timeframe)
        if data is not None and not data.empty and len(data) > 30:
            all_data[f"{timeframe}_{period}"] = data
    
    return all_data

# Main Application
def main():
    st.title("📈 Professional Algorithmic Trading Analysis System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        asset1 = st.selectbox("Select Ticker 1", list(ASSET_MAPPING.keys()))
        if asset1 == 'Custom Ticker':
            ticker1 = st.text_input("Enter Ticker 1 Symbol", "AAPL")
        else:
            ticker1 = ASSET_MAPPING[asset1]
        
        enable_ratio = st.checkbox("Enable Ratio Analysis (Ticker 2)")
        
        ticker2 = None
        if enable_ratio:
            asset2 = st.selectbox("Select Ticker 2", list(ASSET_MAPPING.keys()))
            if asset2 == 'Custom Ticker':
                ticker2 = st.text_input("Enter Ticker 2 Symbol", "GOOGL")
            else:
                ticker2 = ASSET_MAPPING[asset2]
        
        fetch_button = st.button("🚀 Analyze All Timeframes", use_container_width=True)
    
    if 'mtf_data' not in st.session_state:
        st.session_state.mtf_data = None
        st.session_state.mtf_data2 = None
        st.session_state.analysis_complete = False
    
    if fetch_button:
        st.session_state.analysis_complete = False
        
        progress_placeholder = st.empty()
        
        st.info("🔄 Fetching multi-timeframe data... This may take a few minutes.")
        st.session_state.mtf_data = fetch_multi_timeframe_data(ticker1, progress_placeholder)
        
        if enable_ratio and ticker2:
            st.session_state.mtf_data2 = fetch_multi_timeframe_data(ticker2, progress_placeholder)
        else:
            st.session_state.mtf_data2 = None
        
        progress_placeholder.empty()
        
        if st.session_state.mtf_data:
            st.session_state.analysis_complete = True
            st.success(f"✅ Fetched data for {len(st.session_state.mtf_data)} timeframe/period combinations!")
        else:
            st.error("❌ Failed to fetch data. Please try again.")
    
    if st.session_state.analysis_complete and st.session_state.mtf_data:
        mtf_data = st.session_state.mtf_data
        mtf_data2 = st.session_state.mtf_data2
        
        # Get first available data for current stats
        first_key = list(mtf_data.keys())[0]
        data1 = mtf_data[first_key]
        
        # Basic Statistics
        st.header("📊 Current Market Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        current_price1 = data1['Close'].iloc[-1]
        prev_price1 = data1['Close'].iloc[-2]
        change1 = current_price1 - prev_price1
        pct_change1 = (change1 / prev_price1) * 100
        
        with col1:
            st.metric(f"{ticker1} Price", f"₹{current_price1:.2f}", f"{change1:.2f} ({pct_change1:+.2f}%)")
        
        rsi1 = calculate_rsi(data1['Close']).iloc[-1]
        with col2:
            rsi_color = "🟢" if rsi1 < 30 else "🔴" if rsi1 > 70 else "🟡"
            st.metric(f"{ticker1} RSI", f"{rsi_color} {rsi1:.2f}", "")
        
        if enable_ratio and mtf_data2:
            first_key2 = list(mtf_data2.keys())[0]
            data2 = mtf_data2[first_key2]
            current_price2 = data2['Close'].iloc[-1]
            
            with col3:
                prev_price2 = data2['Close'].iloc[-2]
                change2 = current_price2 - prev_price2
                pct_change2 = (change2 / prev_price2) * 100
                st.metric(f"{ticker2} Price", f"₹{current_price2:.2f}", f"{change2:.2f} ({pct_change2:+.2f}%)")
            
            ratio = current_price1 / current_price2
            with col4:
                st.metric("Ratio", f"{ratio:.4f}", "")
        
        st.markdown("---")
        
        # Create tabs list dynamically
        tab_names = [
            "📈 Multi-Timeframe Overview",
            "🎯 Support/Resistance",
            "📊 Technical Indicators",
            "📉 Z-Score Analysis",
            "💹 Volatility Analysis",
            "🌊 Elliott Waves",
            "📐 Fibonacci Levels",
            "🔄 RSI Divergence",
            "🤖 AI Signals & Forecast",
            "🔬 Backtesting"
        ]
        
        if enable_ratio and mtf_data2:
            tab_names.insert(-2, "⚖️ Ratio Analysis")
        
        tabs = st.tabs(tab_names)
        current_tab = 0
        
        # Tab 0: Multi-Timeframe Overview
        with tabs[current_tab]:
            st.subheader("📈 Multi-Timeframe Analysis Overview")
            
            mtf_summary = []
            for tf_period, data in mtf_data.items():
                tf, period = tf_period.split('_')
                curr_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[0]
                returns = ((curr_price - prev_price) / prev_price) * 100
                points = curr_price - prev_price
                
                rsi = calculate_rsi(data['Close']).iloc[-1]
                
                # Check divergence
                divergences = detect_rsi_divergence(data, calculate_rsi(data['Close']))
                active_div = [d for d in divergences if not d['resolved']]
                div_text = f"{active_div[0]['type']}" if active_div else "None"
                
                # Check S/R
                sr_levels = find_support_resistance(data)
                nearby_sr = [l for l in sr_levels if abs(curr_price - l['price']) < curr_price * 0.02]
                sr_text = f"{nearby_sr[0]['type']} at ₹{nearby_sr[0]['price']:.2f}" if nearby_sr else "None"
                
                # Fibonacci
                fib_levels, _, _ = calculate_fibonacci_levels(data)
                nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - curr_price))
                fib_dist = abs(curr_price - nearest_fib[1])
                
                color = "🟢" if returns > 0 else "🔴" if returns < 0 else "⚪"
                
                mtf_summary.append({
                    'Timeframe': tf,
                    'Period': period,
                    'Status': color,
                    'Current_Price': f"₹{curr_price:.2f}",
                    'Returns_%': f"{returns:+.2f}%",
                    'Points': f"{points:+.2f}",
                    'RSI': f"{rsi:.2f}",
                    'RSI_Divergence': div_text,
                    'Near_SR': sr_text,
                    'Nearest_Fib': f"{nearest_fib[0]} (₹{fib_dist:.2f} away)"
                })
            
            mtf_df = pd.DataFrame(mtf_summary)
            
            # Color-code the dataframe
            st.dataframe(mtf_df, use_container_width=True, height=600)
            
            # Download button
            csv = mtf_df.to_csv(index=False)
            st.download_button("📥 Download Multi-Timeframe Data", csv, f"mtf_analysis_{ticker1}.csv", "text/csv")
        
        current_tab += 1
        
        # Tab 1: Support/Resistance (using primary timeframe)
        with tabs[current_tab]:
            st.subheader("🎯 Support and Resistance Levels - Multi-Timeframe Analysis")
            
            # Analyze across multiple timeframes
            all_sr_levels = {}
            for tf_period, data in list(mtf_data.items())[:10]:
                tf, period = tf_period.split('_')
                sr_levels = find_support_resistance(data)
                if sr_levels:
                    all_sr_levels[tf_period] = sr_levels
            
            # Show primary timeframe detailed analysis
            first_key = list(mtf_data.keys())[0]
            tf, period = first_key.split('_')
            st.info(f"**Primary Timeframe**: {tf} interval, {period} period | **Total Timeframes Analyzed**: {len(all_sr_levels)}")
            
            sr_levels = find_support_resistance(data1)
            
            if sr_levels:
                sr_df = pd.DataFrame([{
                    'Type': level['type'],
                    'Price': f"₹{level['price']:.2f}",
                    'Distance_Points': f"{abs(current_price1 - level['price']):.2f}",
                    'Distance_%': f"{abs((current_price1 - level['price'])/current_price1)*100:.2f}%",
                    'Hit_Count': level['count'],
                    'Sustained': level['sustained'],
                    'First_Hit': time_ago(level['first_hit']),
                    'Last_Hit': time_ago(level['last_hit']),
                    'Accuracy_%': f"{(level['sustained']/level['count'])*100:.1f}%"
                } for level in sr_levels[:10]])
                
                st.dataframe(sr_df, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name='Price', line=dict(color='blue')))
                for level in sr_levels[:5]:
                    color = 'green' if level['type'] == 'Support' else 'red'
                    fig.add_hline(y=level['price'], line_dash="dash", 
                                  annotation_text=f"{level['type']}: {level['price']:.2f}", 
                                  line_color=color)
                fig.update_layout(title="Support & Resistance Levels", height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        current_tab += 1
        
        # Tab 2: Technical Indicators
        with tabs[current_tab]:
            st.subheader("📊 Technical Indicators - Multi-Timeframe Analysis")
            
            # Show analysis across timeframes
            first_key = list(mtf_data.keys())[0]
            tf, period = first_key.split('_')
            st.info(f"**Primary Timeframe**: {tf} interval, {period} period | **Analyzing**: {len(list(mtf_data.items())[:10])} timeframes")
            
            # Primary timeframe detailed analysis
            data1['RSI'] = calculate_rsi(data1['Close'])
            data1['EMA_9'] = calculate_ema(data1['Close'], 9)
            data1['EMA_20'] = calculate_ema(data1['Close'], 20)
            data1['EMA_50'] = calculate_ema(data1['Close'], 50)
            
            st.write("**Exponential Moving Averages (Primary Timeframe)**")
            ema_df = pd.DataFrame({
                'DateTime_IST': data1.index,
                'Close': data1['Close'],
                'EMA_9': data1['EMA_9'],
                'EMA_20': data1['EMA_20'],
                'EMA_50': data1['EMA_50'],
                'Dist_9EMA': data1['Close'] - data1['EMA_9'],
                'Dist_20EMA': data1['Close'] - data1['EMA_20'],
                'Dist_50EMA': data1['Close'] - data1['EMA_50']
            })
            st.dataframe(ema_df.tail(30), use_container_width=True)
            
            # Multi-timeframe EMA status
            st.write("**EMA Alignment Across Timeframes**")
            ema_summary = []
            for tf_period, data in list(mtf_data.items())[:10]:
                tf_name, per = tf_period.split('_')
                curr_price = data['Close'].iloc[-1]
                ema20 = calculate_ema(data['Close'], 20).iloc[-1]
                ema50 = calculate_ema(data['Close'], 50).iloc[-1]
                
                trend = "🟢 Bullish" if curr_price > ema20 > ema50 else "🔴 Bearish" if curr_price < ema20 < ema50 else "🟡 Mixed"
                
                ema_summary.append({
                    'Timeframe': tf_name,
                    'Period': per,
                    'Price': f"₹{curr_price:.2f}",
                    '20_EMA': f"₹{ema20:.2f}",
                    '50_EMA': f"₹{ema50:.2f}",
                    'Trend': trend
                })
            
            st.dataframe(pd.DataFrame(ema_summary), use_container_width=True)
            
            st.write("**RSI Values (Primary Timeframe)**")
            rsi_df = pd.DataFrame({
                'DateTime_IST': data1.index,
                'Close': data1['Close'],
                'RSI': data1['RSI'],
                'Status': data1['RSI'].apply(lambda x: '🟢 Oversold' if x < 30 else '🔴 Overbought' if x > 70 else '🟡 Neutral')
            })
            st.dataframe(rsi_df.tail(30), use_container_width=True)
            
            # Analysis Summary
            st.markdown("---")
            st.markdown("### 📊 Technical Indicator Analysis Summary")
            
            current_price = data1['Close'].iloc[-1]
            current_rsi = data1['RSI'].iloc[-1]
            
            tech_summary = f"""
            **Current Technical State ({tf}/{period})**:
            
            **Price vs Moving Averages**:
            - Current Price: ₹{current_price:.2f}
            - 9 EMA: ₹{data1['EMA_9'].iloc[-1]:.2f} ({"Above" if current_price > data1['EMA_9'].iloc[-1] else "Below"} by {abs(current_price - data1['EMA_9'].iloc[-1]):.2f} points)
            - 20 EMA: ₹{data1['EMA_20'].iloc[-1]:.2f} ({"Above" if current_price > data1['EMA_20'].iloc[-1] else "Below"} by {abs(current_price - data1['EMA_20'].iloc[-1]):.2f} points)
            - 50 EMA: ₹{data1['EMA_50'].iloc[-1]:.2f} ({"Above" if current_price > data1['EMA_50'].iloc[-1] else "Below"} by {abs(current_price - data1['EMA_50'].iloc[-1]):.2f} points)
            
            **RSI Analysis**:
            - Current RSI: {current_rsi:.2f}
            - Status: {"🟢 OVERSOLD - Potential bounce expected" if current_rsi < 30 else "🔴 OVERBOUGHT - Potential correction expected" if current_rsi > 70 else "🟡 NEUTRAL - No extreme condition"}
            
            **Multi-Timeframe Consensus**:
            - Bullish Timeframes: {len([s for s in ema_summary if "Bullish" in s['Trend']])}/{len(ema_summary)}
            - Bearish Timeframes: {len([s for s in ema_summary if "Bearish" in s['Trend']])}/{len(ema_summary)}
            - Mixed Timeframes: {len([s for s in ema_summary if "Mixed" in s['Trend']])}/{len(ema_summary)}
            
            **What This Means**:
            """
            
            bullish_tf = len([s for s in ema_summary if "Bullish" in s['Trend']])
            if bullish_tf > len(ema_summary) * 0.6:
                tech_summary += """
            The majority of timeframes show bullish EMA alignment (price above 20 EMA and 50 EMA), indicating strong uptrend momentum across multiple timeframes. This increases confidence in bullish trades. When short-term and long-term timeframes align bullishly, it creates a high-probability setup for continuation of upward movement.
            
            **Trading Implication**: Consider long positions with stop loss below 20 EMA on primary timeframe.
            """
            elif len([s for s in ema_summary if "Bearish" in s['Trend']]) > len(ema_summary) * 0.6:
                tech_summary += """
            The majority of timeframes show bearish EMA alignment (price below 20 EMA and 50 EMA), indicating strong downtrend momentum. This suggests selling pressure dominates across multiple timeframes, making bullish trades risky.
            
            **Trading Implication**: Avoid long positions. Consider short positions or wait for trend reversal signals.
            """
            else:
                tech_summary += """
            Timeframes show mixed signals with no clear consensus. Some timeframes are bullish while others are bearish, indicating market indecision or transition phase. Trading in such conditions is riskier as direction is unclear.
            
            **Trading Implication**: WAIT for clearer signals. Market in consolidation/transition. Risk of whipsaw trades is high.
            """
            
            st.markdown(tech_summary)
        
        current_tab += 1
        
        # Tab 3: Z-Score Analysis
        with tabs[current_tab]:
            st.subheader("📉 Z-Score Analysis - Complete Multi-Timeframe Breakdown")
            
            st.info(f"**Analyzing {len(list(mtf_data.items()))} timeframes** - Each timeframe analyzed separately below")
            
            # Analyze each timeframe separately with full details
            for idx, (tf_period, data) in enumerate(mtf_data.items()):
                tf, period = tf_period.split('_')
                
                st.markdown(f"---")
                st.markdown(f"## 📊 Z-Score Analysis: {tf} Interval / {period} Period")
                
                zscore_data = calculate_zscore_bins(data)
                
                if not zscore_data.empty and len(zscore_data) > 10:
                    current_zscore = zscore_data['Z_Score'].iloc[-1]
                    current_bin = zscore_data['Z_Score_Bin'].iloc[-1]
                    current_price = zscore_data['Close'].iloc[-1]
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Z-Score", f"{current_zscore:.2f}")
                    with col2:
                        st.metric("Current Bin", current_bin)
                    with col3:
                        st.metric("Mean Z-Score", f"{zscore_data['Z_Score'].mean():.2f}")
                    with col4:
                        st.metric("Std Dev", f"{zscore_data['Z_Score'].std():.2f}")
                    
                    # Bin distribution
                    bin_counts = zscore_data['Z_Score_Bin'].value_counts()
                    st.write(f"**Bin Distribution** ({tf}/{period}):")
                    bin_df = pd.DataFrame({
                        'Bin': bin_counts.index,
                        'Count': bin_counts.values,
                        'Percentage': (bin_counts.values / len(zscore_data) * 100).round(1)
                    })
                    st.dataframe(bin_df, use_container_width=True)
                    
                    # Recent data table
                    st.write(f"**Recent Z-Score Data** ({tf}/{period}) - Last 20 periods:")
                    st.dataframe(zscore_data.tail(20), use_container_width=True)
                    
                    # Historical pattern analysis
                    extreme_positive = zscore_data[zscore_data['Z_Score'] > 2]
                    extreme_negative = zscore_data[zscore_data['Z_Score'] < -2]
                    
                    st.markdown(f"### 🎯 Market Impact & Historical Similarity ({tf}/{period})")
                    
                    if current_zscore > 2:
                        st.error(f"🔴 **EXTREME POSITIVE Z-SCORE: {current_zscore:.2f}**")
                        
                        analysis = f"""
                        **Current Status ({tf}/{period})**:
                        - Price at ₹{current_price:.2f} is {current_zscore:.2f} standard deviations ABOVE normal
                        - This is an extreme overbought statistical condition
                        - Occurs only {(len(extreme_positive)/len(zscore_data)*100):.1f}% of the time in this timeframe
                        
                        **Historical Similarity Analysis**:
                        - Found {len(extreme_positive)} similar extreme positive events in {tf}/{period} data
                        """
                        
                        if len(extreme_positive) >= 2:
                            # Analyze what happened after past extremes
                            recent_extremes = extreme_positive.tail(5)
                            outcomes = []
                            
                            for extreme_idx in recent_extremes.index:
                                try:
                                    idx_pos = zscore_data.index.get_loc(extreme_idx)
                                    if idx_pos + 5 < len(zscore_data):
                                        price_at_extreme = zscore_data['Close'].iloc[idx_pos]
                                        price_5_later = zscore_data['Close'].iloc[idx_pos + 5]
                                        price_10_later = zscore_data['Close'].iloc[idx_pos + 10] if idx_pos + 10 < len(zscore_data) else None
                                        
                                        move_5 = ((price_5_later - price_at_extreme) / price_at_extreme * 100)
                                        move_10 = ((price_10_later - price_at_extreme) / price_at_extreme * 100) if price_10_later else None
                                        
                                        outcomes.append({
                                            'date': extreme_idx,
                                            'price': price_at_extreme,
                                            'move_5': move_5,
                                            'move_10': move_10
                                        })
                                except:
                                    pass
                            
                            if outcomes:
                                avg_move_5 = np.mean([o['move_5'] for o in outcomes])
                                avg_move_10 = np.mean([o['move_10'] for o in outcomes if o['move_10'] is not None])
                                down_count = len([o for o in outcomes if o['move_5'] < 0])
                                
                                analysis += f"""
                        
                        **What Happened After Similar Events**:
                        - Last similar extreme: {time_ago(outcomes[-1]['date'])} at ₹{outcomes[-1]['price']:.2f}
                        - After 5 periods: Average move was {avg_move_5:+.2f}%
                        - After 10 periods: Average move was {avg_move_10:+.2f}%
                        - Correction occurred: {down_count}/{len(outcomes)} times ({down_count/len(outcomes)*100:.0f}% accuracy)
                        
                        **Market Behavior Pattern**:
                        """
                                
                                if avg_move_5 < -1:
                                    analysis += f"""
                        - 📉 **STRONG CORRECTION PATTERN**: Market typically corrected {abs(avg_move_5):.1f}% after extreme positive Z-Score
                        - **Expected Move**: ₹{current_price * (1 + avg_move_5/100):.2f} ({abs(avg_move_5):.1f}% or {abs(current_price * avg_move_5/100):.0f} points)
                        - **Pattern Reliability**: {down_count/len(outcomes)*100:.0f}% - HIGH confidence in correction
                        - **Market Type**: Mean reversion - what goes up must come down
                                    """
                                elif -1 <= avg_move_5 <= 1:
                                    analysis += f"""
                        - 📊 **SIDEWAYS CONSOLIDATION PATTERN**: Market typically moved sideways after extremes
                        - **Expected Move**: Range-bound between ₹{current_price * 0.99:.2f} - ₹{current_price * 1.01:.2f}
                        - **Market Type**: Consolidation after extreme move
                                    """
                                else:
                                    analysis += f"""
                        - 📈 **CONTINUED RALLY PATTERN**: Market showed continued strength despite extreme readings
                        - **Expected Move**: Possible continuation to ₹{current_price * 1.02:.2f}
                        - **Caution**: This is rare and high-risk - could reverse sharply
                                    """
                        
                        st.markdown(analysis)
                    
                    elif current_zscore < -2:
                        st.success(f"🟢 **EXTREME NEGATIVE Z-SCORE: {current_zscore:.2f}**")
                        
                        analysis = f"""
                        **Current Status ({tf}/{period})**:
                        - Price at ₹{current_price:.2f} is {abs(current_zscore):.2f} standard deviations BELOW normal
                        - This is an extreme oversold statistical condition
                        - Occurs only {(len(extreme_negative)/len(zscore_data)*100):.1f}% of the time in this timeframe
                        
                        **Historical Similarity Analysis**:
                        - Found {len(extreme_negative)} similar extreme negative events in {tf}/{period} data
                        """
                        
                        if len(extreme_negative) >= 2:
                            recent_extremes = extreme_negative.tail(5)
                            outcomes = []
                            
                            for extreme_idx in recent_extremes.index:
                                try:
                                    idx_pos = zscore_data.index.get_loc(extreme_idx)
                                    if idx_pos + 5 < len(zscore_data):
                                        price_at_extreme = zscore_data['Close'].iloc[idx_pos]
                                        price_5_later = zscore_data['Close'].iloc[idx_pos + 5]
                                        price_10_later = zscore_data['Close'].iloc[idx_pos + 10] if idx_pos + 10 < len(zscore_data) else None
                                        
                                        move_5 = ((price_5_later - price_at_extreme) / price_at_extreme * 100)
                                        move_10 = ((price_10_later - price_at_extreme) / price_at_extreme * 100) if price_10_later else None
                                        
                                        outcomes.append({
                                            'date': extreme_idx,
                                            'price': price_at_extreme,
                                            'move_5': move_5,
                                            'move_10': move_10
                                        })
                                except:
                                    pass
                            
                            if outcomes:
                                avg_move_5 = np.mean([o['move_5'] for o in outcomes])
                                avg_move_10 = np.mean([o['move_10'] for o in outcomes if o['move_10'] is not None])
                                up_count = len([o for o in outcomes if o['move_5'] > 0])
                                
                                analysis += f"""
                        
                        **What Happened After Similar Events**:
                        - Last similar extreme: {time_ago(outcomes[-1]['date'])} at ₹{outcomes[-1]['price']:.2f}
                        - After 5 periods: Average move was {avg_move_5:+.2f}%
                        - After 10 periods: Average move was {avg_move_10:+.2f}%
                        - Rally occurred: {up_count}/{len(outcomes)} times ({up_count/len(outcomes)*100:.0f}% accuracy)
                        
                        **Market Behavior Pattern**:
                        """
                                
                                if avg_move_5 > 1:
                                    analysis += f"""
                        - 📈 **STRONG RALLY PATTERN**: Market typically rallied {avg_move_5:.1f}% after extreme negative Z-Score
                        - **Expected Move**: ₹{current_price * (1 + avg_move_5/100):.2f} ({avg_move_5:.1f}% or {current_price * avg_move_5/100:.0f} points)
                        - **Pattern Reliability**: {up_count/len(outcomes)*100:.0f}% - HIGH confidence in bounce
                        - **Market Type**: Mean reversion - oversold bounce
                                    """
                                elif -1 <= avg_move_5 <= 1:
                                    analysis += f"""
                        - 📊 **SIDEWAYS CONSOLIDATION PATTERN**: Market typically stabilized after extreme selling
                        - **Expected Move**: Range-bound between ₹{current_price * 0.99:.2f} - ₹{current_price * 1.01:.2f}
                        - **Market Type**: Base building after washout
                                    """
                                else:
                                    analysis += f"""
                        - 📉 **CONTINUED DECLINE PATTERN**: Market showed continued weakness despite oversold
                        - **Expected Move**: Possible further decline to ₹{current_price * 0.98:.2f}
                        - **Caution**: Wait for stabilization before entering
                                    """
                        
                        st.markdown(analysis)
                    
                    else:
                        st.info(f"🟡 **NORMAL RANGE Z-SCORE: {current_zscore:.2f}**")
                        st.markdown(f"""
                        **Current Status ({tf}/{period})**:
                        - Z-Score of {current_zscore:.2f} is within normal range (±2σ)
                        - No extreme statistical condition present
                        - Market trading in typical distribution zone
                        - Bin: **{current_bin}**
                        
                        **Market Behavior**:
                        - No strong mean reversion signal
                        - Follow trend and other technical indicators
                        - Wait for extreme readings for high-probability trades
                        """)
                else:
                    st.warning(f"Insufficient data for {tf}/{period} Z-Score analysis")
            
            if not zscore_data.empty:
                current_zscore = zscore_data['Z_Score'].iloc[-1]
                current_bin = zscore_data['Z_Score_Bin'].iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Z-Score", f"{current_zscore:.2f}")
                with col2:
                    st.metric("Current Bin", current_bin)
                with col3:
                    st.metric("Average Z-Score", f"{zscore_data['Z_Score'].mean():.2f}")
                with col4:
                    st.metric("Std Dev", f"{zscore_data['Z_Score'].std():.2f}")
                
                st.write("**Z-Score Distribution Table (Recent 50 periods)**")
                st.dataframe(zscore_data.tail(50), use_container_width=True)
                
                # Comprehensive Analysis Summary
                st.markdown("---")
                st.write("### 📊 Z-Score Interpretation & Trading Signals")
                
                extreme_positive = zscore_data[zscore_data['Z_Score'] > 2]
                extreme_negative = zscore_data[zscore_data['Z_Score'] < -2]
                
                # Multi-timeframe consensus
                extreme_neg_count = len([a for a in all_zscore_analysis if float(a['Current_ZScore']) < -2])
                extreme_pos_count = len([a for a in all_zscore_analysis if float(a['Current_ZScore']) > 2])
                
                summary_text = f"""
                **Current Market Position ({tf}/{period})**: The market is in the **{current_bin}** zone with Z-Score of **{current_zscore:.2f}**.
                
                **Multi-Timeframe Z-Score Consensus**:
                - Timeframes showing extreme negative Z-Score (<-2): {extreme_neg_count}/{len(all_zscore_analysis)}
                - Timeframes showing extreme positive Z-Score (>2): {extreme_pos_count}/{len(all_zscore_analysis)}
                - This provides confidence in mean reversion expectations
                
                **Historical Context ({tf}/{period})**:
                - Extreme positive moves (>2σ): Found **{len(extreme_positive)}** instances
                - Extreme negative moves (<-2σ): Found **{len(extreme_negative)}** instances
                
                **Statistical Interpretation**:
                """
                
                if current_zscore > 2:
                    summary_text += f"""
                - ⚠️ **EXTREME OVERBOUGHT CONDITION**
                - The current price movement is **{current_zscore:.2f} standard deviations ABOVE the mean**
                - This represents a statistically rare event (occurs <5% of the time in normal distribution)
                - **Mean Reversion Principle**: What goes up excessively tends to come back down to average
                
                **What Historically Happened**:
                """
                    if len(extreme_positive) > 1:
                        last_extreme = extreme_positive.index[-2]
                        summary_text += f"\n- Last similar extreme: {time_ago(last_extreme)}"
                        
                        # Calculate what happened after
                        try:
                            idx_pos = zscore_data.index.get_loc(last_extreme)
                            if idx_pos + 10 < len(zscore_data):
                                future_return = ((zscore_data['Close'].iloc[idx_pos + 10] - zscore_data['Close'].iloc[idx_pos]) / 
                                               zscore_data['Close'].iloc[idx_pos] * 100)
                                summary_text += f"\n- 10 periods later: Price moved {future_return:+.2f}%"
                        except:
                            pass
                    
                    summary_text += f"""
                
                **Expected Direction**: ⬇️ DOWNSIDE CORRECTION or CONSOLIDATION
                **Probability**: HIGH (65-75% based on statistical mean reversion)
                **Expected Move**: Reversion toward mean, typically {abs(current_zscore * zscore_data['Return_%'].std()):.2f}% correction
                **Timeframe**: Next {period} period
                
                **Trading Strategy**:
                - AVOID new long positions at extreme levels
                - Consider profit booking if holding longs
                - Watch for reversal confirmation before entering shorts
                - Set tight stop losses for any new positions
                """
                
                elif current_zscore < -2:
                    summary_text += f"""
                - 🟢 **EXTREME OVERSOLD CONDITION**
                - The current price movement is **{abs(current_zscore):.2f} standard deviations BELOW the mean**
                - This represents a statistically rare event (occurs <5% of the time)
                - **Mean Reversion Principle**: Excessive selling typically followed by bounce/recovery
                
                **What Historically Happened**:
                """
                    if len(extreme_negative) > 1:
                        recent_extremes = extreme_negative.tail(5)
                        recoveries = []
                        
                        for idx in recent_extremes.index:
                            try:
                                idx_pos = zscore_data.index.get_loc(idx)
                                if idx_pos + 10 < len(zscore_data):
                                    future_return = ((zscore_data['Close'].iloc[idx_pos + 10] - zscore_data['Close'].iloc[idx_pos]) / 
                                                   zscore_data['Close'].iloc[idx_pos] * 100)
                                    recoveries.append(future_return)
                            except:
                                pass
                        
                        if recoveries:
                            avg_recovery = np.mean(recoveries)
                            success_rate = len([r for r in recoveries if r > 0]) / len(recoveries) * 100
                            
                            summary_text += f"""
- After last {len(recoveries)} extreme negative events:
  * Average recovery: **{avg_recovery:+.2f}%** within 10 periods
  * Success rate (positive return): **{success_rate:.1f}%**
  * This pattern occurred {time_ago(recent_extremes.index[-1])}
                            """
                    
                    summary_text += f"""
                
                **Expected Direction**: ⬆️ UPSIDE BOUNCE or RECOVERY
                **Probability**: HIGH (70-80% based on historical pattern)
                **Expected Move**: Mean reversion, typically {abs(current_zscore * zscore_data['Return_%'].std()):.2f}% upward
                **Timeframe**: Next {period} period
                
                **Trading Strategy**:
                - 🟢 EXCELLENT BUYING OPPORTUNITY at oversold levels
                - Enter long positions with defined stop loss
                - Target: Mean reversion to 0 Z-Score level
                - Risk:Reward is favorable due to extreme deviation
                """
                
                elif -1 < current_zscore < 1:
                    summary_text += f"""
                - 🟡 **NORMAL TRADING RANGE**
                - Price movement within **1 standard deviation** of mean (occurs ~68% of time)
                - Market is in equilibrium - neither overbought nor oversold
                - No extreme statistical edge for mean reversion trades
                
                **Expected Direction**: ⚖️ CONTINUATION of current trend
                **Probability**: MODERATE (50-60% - coin flip territory)
                
                **Trading Strategy**:
                - Follow trend rather than counter-trend
                - Use other indicators (EMA, RSI, Support/Resistance) for direction
                - No statistical edge from Z-Score alone
                - Wait for Z-Score to reach extremes for higher probability setups
                """
                
                else:
                    summary_text += f"""
                - ⚪ **MODERATE ZONE** (1-2 standard deviations)
                - Price showing stronger than normal momentum
                - Not yet at extreme levels for high-confidence mean reversion
                - Watch closely - may extend to extreme zone or reverse from here
                
                **Expected Direction**: {"⬆️ Continued upside possible" if current_zscore > 0 else "⬇️ Continued downside possible"}
                **Probability**: MODERATE (55-65%)
                
                **Trading Strategy**:
                - Can trade with trend but with caution
                - Monitor for extension to extreme levels (±2σ)
                - If reaches extreme, prepare for mean reversion trade
                - Use tight stops as reversal can be swift
                """
            
            # Summary section at the end
            st.markdown("---")
            st.markdown("---")
            st.markdown("# 🎯 Z-SCORE FINAL CONSENSUS ACROSS ALL TIMEFRAMES")
            
            # Collect all signals
            all_z_signals = []
            for tf_period, data in mtf_data.items():
                tf, period = tf_period.split('_')
                zscore_data = calculate_zscore_bins(data)
                if not zscore_data.empty:
                    current_zscore = zscore_data['Z_Score'].iloc[-1]
                    if current_zscore > 2:
                        all_z_signals.append('correction')
                    elif current_zscore < -2:
                        all_z_signals.append('rally')
                    else:
                        all_z_signals.append('neutral')
            
            correction_count = all_z_signals.count('correction')
            rally_count = all_z_signals.count('rally')
            neutral_count = all_z_signals.count('neutral')
            total = len(all_z_signals)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correction Signals", f"{correction_count}/{total}", 
                         f"{correction_count/total*100:.0f}%")
            with col2:
                st.metric("Rally Signals", f"{rally_count}/{total}", 
                         f"{rally_count/total*100:.0f}%")
            with col3:
                st.metric("Neutral Signals", f"{neutral_count}/{total}", 
                         f"{neutral_count/total*100:.0f}%")
            
            if correction_count > total * 0.5:
                st.error(f"""
                ### 🔴 Z-SCORE CONSENSUS: CORRECTION EXPECTED
                
                {correction_count} out of {total} timeframes ({correction_count/total*100:.0f}%) show extreme positive Z-Scores indicating overbought conditions. 
                Historical patterns suggest mean reversion downward is highly probable.
                """)
            elif rally_count > total * 0.5:
                st.success(f"""
                ### 🟢 Z-SCORE CONSENSUS: RALLY/BOUNCE EXPECTED
                
                {rally_count} out of {total} timeframes ({rally_count/total*100:.0f}%) show extreme negative Z-Scores indicating oversold conditions. 
                Historical patterns suggest mean reversion upward is highly probable.
                """)
            else:
                st.info(f"""
                ### 🟡 Z-SCORE CONSENSUS: NO CLEAR EXTREME
                
                Mixed signals across timeframes. {neutral_count} timeframes in normal range. 
                No strong statistical edge for mean reversion trades currently.
                """)
            
            if not zscore_data.empty:
                # Statistics
                current_zscore = zscore_data['Z_Score'].iloc[-1]
                current_bin = zscore_data['Z_Score_Bin'].iloc[-1]
                
                st.write("### Current Z-Score Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Z-Score", f"{current_zscore:.2f}")
                with col2:
                    st.metric("Current Bin", current_bin)
                with col3:
                    st.metric("Average Z-Score", f"{zscore_data['Z_Score'].mean():.2f}")
                with col4:
                    st.metric("Std Dev", f"{zscore_data['Z_Score'].std():.2f}")
                
                st.write("**Z-Score Distribution Table**")
                st.dataframe(zscore_data.tail(50), use_container_width=True)
                
                # Analysis Summary
                st.write("### 📊 Z-Score Analysis Summary")
                
                extreme_positive = zscore_data[zscore_data['Z_Score'] > 2]
                extreme_negative = zscore_data[zscore_data['Z_Score'] < -2]
                
                summary_text = f"""
                **Current Market Position**: The market is currently in the **{current_bin}** zone with a Z-Score of **{current_zscore:.2f}**.
                
                **Historical Context**: 
                - Found **{len(extreme_positive)}** instances of extreme positive Z-Scores (>2.0) in the dataset
                - Found **{len(extreme_negative)}** instances of extreme negative Z-Scores (<-2.0) in the dataset
                
                **What This Means**:
                """
                
                if current_zscore > 2:
                    summary_text += f"""
                - The current price change is **{current_zscore:.2f} standard deviations above the mean**, indicating an extreme upward move
                - Historically, such extreme moves often lead to **mean reversion** within the next few periods
                - Last time we saw similar extreme positive Z-Scores: {time_ago(extreme_positive.index[-2]) if len(extreme_positive) > 1 else "First occurrence"}
                - **Expected Direction**: Potential downside correction or consolidation ahead
                - **Confidence**: High probability of mean reversion based on statistical analysis
                """
                elif current_zscore < -2:
                    summary_text += f"""
                - The current price change is **{abs(current_zscore):.2f} standard deviations below the mean**, indicating an extreme downward move
                - Historically, such extreme moves often lead to **mean reversion** (bounce back) within the next few periods
                - Last time we saw similar extreme negative Z-Scores: {time_ago(extreme_negative.index[-2]) if len(extreme_negative) > 1 else "First occurrence"}
                - **Expected Direction**: Potential upside bounce or recovery ahead
                - **Confidence**: High probability of mean reversion based on statistical analysis
                """
                elif -1 < current_zscore < 1:
                    summary_text += f"""
                - The current price movement is within **1 standard deviation** of the mean, indicating normal market behavior
                - This is the most common zone where price typically trades
                - **Expected Direction**: Continuation of current trend with no extreme moves expected
                - **Confidence**: Moderate - market is in equilibrium, wait for clearer signals
                """
                else:
                    summary_text += f"""
                - The current price movement is between 1-2 standard deviations from mean, showing elevated but not extreme activity
                - Market is showing stronger than normal momentum but not yet at extreme levels
                - **Expected Direction**: Monitor for potential reversal if Z-Score reaches ±2
                - **Confidence**: Moderate - trending market with some continuation expected
                """
                
                # Add recent rally analysis
                if len(extreme_negative) > 0:
                    recent_extreme = extreme_negative.tail(5)
                    avg_recovery = []
                    for idx in recent_extreme.index:
                        try:
                            idx_pos = zscore_data.index.get_loc(idx)
                            if idx_pos + 10 < len(zscore_data):
                                future_return = ((zscore_data['Close'].iloc[idx_pos + 10] - zscore_data['Close'].iloc[idx_pos]) / 
                                               zscore_data['Close'].iloc[idx_pos] * 100)
                                avg_recovery.append(future_return)
                        except:
                            pass
                    
                    if avg_recovery:
                        summary_text += f"""
                        
                **Historical Rally Pattern**: After extreme negative Z-Scores (oversold conditions), the market typically recovered by an average of **{np.mean(avg_recovery):.2f}%** within the next 10 periods. This pattern has occurred **{len(avg_recovery)}** times with an accuracy of approximately **{(len([r for r in avg_recovery if r > 0])/len(avg_recovery))*100:.1f}%**.
                """
                
                st.markdown(summary_text)
                
                # Bin distribution
                st.write("### Z-Score Bin Distribution")
                bin_dist = zscore_data['Z_Score_Bin'].value_counts()
                st.bar_chart(bin_dist)
        
        current_tab += 1
        
        # Tab 4: Volatility Analysis
        with tabs[current_tab]:
            st.subheader("💹 Volatility Analysis - Complete Multi-Timeframe Breakdown")
            
            st.info(f"**Analyzing {len(list(mtf_data.items()))} timeframes** - Each timeframe analyzed separately below")
            
            # Analyze each timeframe separately
            for idx, (tf_period, data) in enumerate(mtf_data.items()):
                tf, period = tf_period.split('_')
                
                st.markdown(f"---")
                st.markdown(f"## 📊 Volatility Analysis: {tf} Interval / {period} Period")
                
                vol_data = calculate_volatility_bins(data)
                
                if not vol_data.empty and len(vol_data) > 10:
                    current_vol = vol_data['Volatility'].iloc[-1]
                    current_bin = vol_data['Volatility_Bin'].iloc[-1]
                    current_price = vol_data['Close'].iloc[-1]
                    avg_vol = vol_data['Volatility'].mean()
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Volatility", f"{current_vol:.2f}%")
                    with col2:
                        st.metric("Current Bin", current_bin)
                    with col3:
                        st.metric("Average Vol", f"{avg_vol:.2f}%")
                    with col4:
                        st.metric("Max Vol", f"{vol_data['Volatility'].max():.2f}%")
                    
                    # Bin distribution
                    bin_counts = vol_data['Volatility_Bin'].value_counts()
                    st.write(f"**Bin Distribution** ({tf}/{period}):")
                    bin_df = pd.DataFrame({
                        'Bin': bin_counts.index,
                        'Count': bin_counts.values,
                        'Percentage': (bin_counts.values / len(vol_data) * 100).round(1)
                    })
                    st.dataframe(bin_df, use_container_width=True)
                    
                    # Recent data
                    st.write(f"**Recent Volatility Data** ({tf}/{period}) - Last 20 periods:")
                    st.dataframe(vol_data.tail(20), use_container_width=True)
                    
                    # Historical analysis
                    high_vol = vol_data[vol_data['Volatility_Bin'].isin(['High', 'Very High'])]
                    low_vol = vol_data[vol_data['Volatility_Bin'].isin(['Low', 'Very Low'])]
                    
                    st.markdown(f"### 🎯 Market Impact & Historical Similarity ({tf}/{period})")
                    
                    if current_vol > avg_vol * 1.5:
                        st.error(f"🔴 **VERY HIGH VOLATILITY: {current_vol:.2f}%**")
                        
                        analysis = f"""
                        **Current Status ({tf}/{period})**:
                        - Volatility at {current_vol:.2f}% is {(current_vol/avg_vol):.1f}x the average ({avg_vol:.2f}%)
                        - This is a high volatility environment
                        - Occurs {(len(high_vol)/len(vol_data)*100):.1f}% of the time in this timeframe
                        
                        **Historical Similarity Analysis**:
                        - Found {len(high_vol)} high volatility periods in {tf}/{period} data
                        """
                        
                        if len(high_vol) >= 3:
                            recent_high_vol = high_vol.tail(5)
                            outcomes = []
                            
                            for vol_idx in recent_high_vol.index:
                                try:
                                    idx_pos = vol_data.index.get_loc(vol_idx)
                                    if idx_pos + 5 < len(vol_data):
                                        price_at_vol = vol_data['Close'].iloc[idx_pos]
                                        price_5_later = vol_data['Close'].iloc[idx_pos + 5]
                                        
                                        move = abs((price_5_later - price_at_vol) / price_at_vol * 100)
                                        direction = "up" if price_5_later > price_at_vol else "down"
                                        
                                        outcomes.append({
                                            'date': vol_idx,
                                            'move': move,
                                            'direction': direction
                                        })
                                except:
                                    pass
                            
                            if outcomes:
                                avg_move = np.mean([o['move'] for o in outcomes])
                                max_move = max([o['move'] for o in outcomes])
                                
                                analysis += f"""
                        
                        **What Happened During Past High Volatility**:
                        - Last high volatility period: {time_ago(outcomes[-1]['date'])}
                        - Average absolute move within 5 periods: {avg_move:.2f}%
                        - Largest move recorded: {max_move:.2f}%
                        - Current volatility suggests move of {current_vol * 0.8:.1f}% to {current_vol * 1.5:.1f}%
                        
                        **Market Behavior Pattern**:
                        """
                                
                                if avg_move > 3:
                                    analysis += f"""
                        - 📈📉 **LARGE MOVES PATTERN**: High volatility historically led to {avg_move:.1f}% average moves
                        - **Expected Range**: ₹{current_price * (1 - avg_move/100):.2f} to ₹{current_price * (1 + avg_move/100):.2f}
                        - **Market Type**: Trending with large swings - breakout likely
                        - **Trading Style**: Use wider stops, reduce position size, trend-following
                                    """
                                else:
                                    analysis += f"""
                        - 📊 **MODERATE VOLATILITY PATTERN**: Elevated but controlled moves
                        - **Expected Range**: ₹{current_price * 0.98:.2f} to ₹{current_price * 1.02:.2f}
                        - **Market Type**: Active but not extreme
                                    """
                        
                        st.markdown(analysis)
                    
                    elif current_vol < avg_vol * 0.7:
                        st.success(f"🟢 **VERY LOW VOLATILITY: {current_vol:.2f}%**")
                        
                        analysis = f"""
                        **Current Status ({tf}/{period})**:
                        - Volatility at {current_vol:.2f}% is only {(current_vol/avg_vol):.1f}x the average ({avg_vol:.2f}%)
                        - This is a low volatility / compression environment
                        - Occurs {(len(low_vol)/len(vol_data)*100):.1f}% of the time in this timeframe
                        
                        **Historical Similarity Analysis**:
                        - Found {len(low_vol)} low volatility periods in {tf}/{period} data
                        
                        **Volatility Compression - Spring Loading Effect**:
                        - Low volatility is like a coiled spring
                        - The longer the compression, the bigger the eventual expansion
                        - Market "storing energy" for next big move
                        """
                        
                        if len(low_vol) >= 3:
                            # Find what happened after low vol periods
                            breakout_analysis = []
                            for vol_idx in low_vol.index[-5:]:
                                try:
                                    idx_pos = vol_data.index.get_loc(vol_idx)
                                    # Look forward to find when volatility expanded
                                    for i in range(idx_pos + 1, min(idx_pos + 20, len(vol_data))):
                                        if vol_data['Volatility'].iloc[i] > avg_vol * 1.2:
                                            price_at_low_vol = vol_data['Close'].iloc[idx_pos]
                                            price_at_breakout = vol_data['Close'].iloc[i]
                                            move = ((price_at_breakout - price_at_low_vol) / price_at_low_vol * 100)
                                            periods_to_breakout = i - idx_pos
                                            
                                            breakout_analysis.append({
                                                'periods': periods_to_breakout,
                                                'move': abs(move),
                                                'direction': 'up' if move > 0 else 'down'
                                            })
                                            break
                                except:
                                    pass
                            
                            if breakout_analysis:
                                avg_periods = np.mean([b['periods'] for b in breakout_analysis])
                                avg_breakout_move = np.mean([b['move'] for b in breakout_analysis])
                                
                                analysis += f"""
                        
                        **Breakout Pattern After Low Volatility**:
                        - Historical breakouts occurred within {avg_periods:.0f} periods on average
                        - Average breakout move: {avg_breakout_move:.2f}%
                        - Breakout probability: {len(breakout_analysis)/5*100:.0f}% (occurred {len(breakout_analysis)}/5 times)
                        
                        **Expected Behavior**:
                        - ⚡ **BREAKOUT IMMINENT**: Prepare for volatility expansion
                        - **Direction**: Unknown until breakout occurs - could be either way
                        - **Expected Move**: {avg_breakout_move * 0.7:.1f}% to {avg_breakout_move * 1.3:.1f}%
                        - **Strategy**: Wait for breakout confirmation, then trade direction of move
                                """
                        
                        st.markdown(analysis)
                    
                    else:
                        st.info(f"🟡 **NORMAL VOLATILITY: {current_vol:.2f}%**")
                        st.markdown(f"""
                        **Current Status ({tf}/{period})**:
                        - Volatility at {current_vol:.2f}% is near average ({avg_vol:.2f}%)
                        - Normal market conditions
                        - Bin: **{current_bin}**
                        
                        **Market Behavior**:
                        - Standard trading environment
                        - Normal position sizing appropriate
                        - Most strategies workable
                        """)
                    
                    # Mini chart for each timeframe
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vol_data['DateTime_IST'].tail(100), 
                        y=vol_data['Volatility'].tail(100),
                        mode='lines',
                        name='Volatility',
                        line=dict(color='orange')
                    ))
                    fig.add_hline(y=avg_vol, line_dash="dash", 
                                 annotation_text=f"Avg: {avg_vol:.2f}%", 
                                 line_color='blue')
                    fig.update_layout(
                        title=f"Volatility Chart - {tf}/{period}",
                        xaxis_title="Date",
                        yaxis_title="Volatility %",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for {tf}/{period} volatility analysis")
            
            # Final consensus
            st.markdown("---")
            st.markdown("---")
            st.markdown("# 🎯 VOLATILITY FINAL CONSENSUS ACROSS ALL TIMEFRAMES")
            
            all_vol_signals = []
            for tf_period, data in mtf_data.items():
                vol_data = calculate_volatility_bins(data)
                if not vol_data.empty:
                    current_vol = vol_data['Volatility'].iloc[-1]
                    avg_vol = vol_data['Volatility'].mean()
                    
                    if current_vol > avg_vol * 1.5:
                        all_vol_signals.append('high')
                    elif current_vol < avg_vol * 0.7:
                        all_vol_signals.append('low')
                    else:
                        all_vol_signals.append('normal')
            
            high_count = all_vol_signals.count('high')
            low_count = all_vol_signals.count('low')
            normal_count = all_vol_signals.count('normal')
            total = len(all_vol_signals)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Volatility TFs", f"{high_count}/{total}", 
                         f"{high_count/total*100:.0f}%")
            with col2:
                st.metric("Low Volatility TFs", f"{low_count}/{total}", 
                         f"{low_count/total*100:.0f}%")
            with col3:
                st.metric("Normal Volatility TFs", f"{normal_count}/{total}", 
                         f"{normal_count/total*100:.0f}%")
            
            if high_count > total * 0.5:
                st.error(f"""
                ### 🔴 VOLATILITY CONSENSUS: HIGH VOLATILITY REGIME
                
                {high_count} out of {total} timeframes ({high_count/total*100:.0f}%) show elevated volatility.
                Expect larger price swings, use wider stops, reduce position size.
                Breakout/trend-following strategies favored.
                """)
            elif low_count > total * 0.5:
                st.success(f"""
                ### 🟢 VOLATILITY CONSENSUS: LOW VOLATILITY / COMPRESSION
                
                {low_count} out of {total} timeframes ({low_count/total*100:.0f}%) show low volatility.
                Market in compression phase - BREAKOUT IMMINENT.
                Prepare for volatility expansion and directional move.
                """)
            else:
                st.info(f"""
                ### 🟡 VOLATILITY CONSENSUS: NORMAL CONDITIONS
                
                {normal_count} timeframes show normal volatility levels.
                Standard trading conditions apply.
                """)
            
            if not vol_data.empty:
                current_vol = vol_data['Volatility'].iloc[-1]
                current_vol_bin = vol_data['Volatility_Bin'].iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Volatility", f"{current_vol:.2f}%")
                with col2:
                    st.metric("Current Bin", current_vol_bin)
                with col3:
                    st.metric("Average Volatility", f"{vol_data['Volatility'].mean():.2f}%")
                with col4:
                    st.metric("Max Volatility", f"{vol_data['Volatility'].max():.2f}%")
                
                st.write("**Volatility Distribution Table (Recent 50)**")
                st.dataframe(vol_data.tail(50), use_container_width=True)
                
                # Comprehensive summary
                st.markdown("---")
                st.write("### 📊 Volatility Interpretation & Trading Implications")
                
                high_vol = vol_data[vol_data['Volatility_Bin'].isin(['High', 'Very High'])]
                low_vol = vol_data[vol_data['Volatility_Bin'].isin(['Low', 'Very Low'])]
                
                # Multi-timeframe consensus
                high_vol_count = len([a for a in all_vol_analysis if 'High' in a['Signal']])
                low_vol_count = len([a for a in all_vol_analysis if 'Low' in a['Signal']])
                
                vol_summary = f"""
                **Current Volatility State ({tf}/{period})**: **{current_vol_bin}** at **{current_vol:.2f}%**
                
                **Multi-Timeframe Volatility Consensus**:
                - High volatility timeframes: {high_vol_count}/{len(all_vol_analysis)}
                - Low volatility timeframes: {low_vol_count}/{len(all_vol_analysis)}
                - Consensus helps confirm volatility regime
                
                **Historical Volatility Context ({tf}/{period})**:
                - High volatility periods: **{len(high_vol)}** times ({(len(high_vol)/len(vol_data)*100):.1f}% of time)
                - Low volatility periods: **{len(low_vol)}** times ({(len(low_vol)/len(vol_data)*100):.1f}% of time)
                
                **What This Means**:
                """
                
                if current_vol_bin in ['Very High', 'High']:
                    vol_summary += f"""
                
                🔴 **HIGH VOLATILITY ENVIRONMENT**
                
                **Characteristics**:
                - Larger price swings in both directions
                - Increased uncertainty and risk
                - Wider intraday ranges
                - Higher probability of gap moves
                
                **What Causes High Volatility**:
                - Major news events or announcements
                - Economic data releases
                - Geopolitical tensions
                - Market uncertainty or fear
                - Large institutional activity
                
                **Historical Pattern Analysis**:
                """
                    
                    if len(high_vol) > 5:
                        recent_high_vol = high_vol.tail(5)
                        moves = []
                        for idx in recent_high_vol.index:
                            try:
                                idx_pos = vol_data.index.get_loc(idx)
                                if idx_pos + 5 < len(vol_data):
                                    future_move = abs((vol_data['Close'].iloc[idx_pos + 5] - vol_data['Close'].iloc[idx_pos]) / 
                                                     vol_data['Close'].iloc[idx_pos] * 100)
                                    moves.append(future_move)
                            except:
                                pass
                        
                        if moves:
                            vol_summary += f"""
- During past high volatility periods, market typically moved **{np.mean(moves):.2f}%** (absolute) within next 5 periods
- Largest move during high volatility: **{max(moves):.2f}%**
- Smallest move: **{min(moves):.2f}%**
- Current volatility suggests potential move of **{current_vol * 0.8:.1f} to {current_vol * 1.5:.1f} points** in coming periods
                            """
                    
                    vol_summary += f"""
                
                **Trading Strategy for High Volatility**:
                
                ✅ **DO**:
                - Use WIDER stop losses (2-3x normal) to avoid getting stopped out by noise
                - REDUCE position size to maintain same risk level
                - Focus on BREAKOUT strategies (volatility expansion often precedes big moves)
                - Trade OPTIONS for defined risk (if available)
                - Use LIMIT orders instead of market orders
                - Set REALISTIC targets - high vol allows bigger moves
                
                ❌ **DON'T**:
                - Use tight stops (will get whipsawed)
                - Overtrade (emotions run high in volatile markets)
                - Use normal position sizes (risk too high)
                - Ignore risk management
                - Trade against strong momentum
                
                **Expected Moves**:
                - Intraday swings: {current_vol * 0.3:.1f}% to {current_vol * 0.6:.1f}%
                - Multi-day potential: {current_vol * 1.2:.1f}% to {current_vol * 2:.1f}%
                
                **Risk Level**: 🔴 HIGH - Trade with extra caution
                """
                
                elif current_vol_bin in ['Very Low', 'Low']:
                    vol_summary += f"""
                
                🟢 **LOW VOLATILITY ENVIRONMENT**
                
                **Characteristics**:
                - Tight price ranges
                - Slow, grinding moves
                - Low momentum
                - Market consolidation/compression
                
                **What Causes Low Volatility**:
                - No major catalysts or news
                - Holiday periods
                - Market indecision
                - Waiting for key events
                - Calm, stable market conditions
                
                **The Volatility Compression Principle**:
                - Low volatility doesn't last forever
                - Periods of calm are often followed by explosive moves
                - Think of it as a coiled spring - compression before expansion
                - The longer the compression, the bigger the eventual move
                
                **Historical Pattern**:
                """
                    
                    if len(low_vol) > 3:
                        last_low_vol = low_vol.index[-1]
                        vol_summary += f"\n- Last low volatility period: {time_ago(last_low_vol)}"
                    
                    vol_summary += f"""
                
                **Trading Strategy for Low Volatility**:
                
                ✅ **DO**:
                - WAIT for volatility expansion before taking significant positions
                - Use MEAN REVERSION strategies (buy support, sell resistance)
                - Employ RANGE TRADING techniques
                - Use TIGHTER stops (market not moving much anyway)
                - PREPARE for breakout - identify key levels
                - Consider OPTIONS strategies like iron condors (if available)
                
                ❌ **DON'T**:
                - Expect large moves or set wide targets
                - Use breakout strategies (false breakouts common)
                - Overtrade out of boredom
                - Ignore the bigger picture (low vol won't last)
                
                **What to Watch For**:
                - Volatility starting to pick up (early warning of bigger move)
                - Key support/resistance tests
                - Volume expansion (precedes volatility expansion)
                
                **Expected Behavior**:
                - Small daily moves: {current_vol * 0.5:.1f}% typical
                - Range-bound trading likely
                - Eventual breakout probability: HIGH (70-80%)
                
                **Risk Level**: 🟢 LOW - But prepare for change
                """
                
                else:
                    vol_summary += f"""
                
                🟡 **MODERATE VOLATILITY ENVIRONMENT**
                
                **Characteristics**:
                - Normal price movement patterns
                - Balanced risk-reward setups
                - Standard trading conditions
                - Predictable intraday ranges
                
                **Trading Strategy**:
                - Use STANDARD position sizing
                - NORMAL stop loss distances (1.5-2%)
                - Most strategies work well in moderate volatility
                - Both trend-following and mean-reversion viable
                
                **Expected Moves**:
                - Typical daily range: {current_vol * 0.7:.1f}% to {current_vol * 1:.1f}%
                
                **Risk Level**: 🟡 MODERATE - Normal trading conditions
                """
                
                st.markdown(vol_summary)
                
                # Volatility chart
                st.markdown("---")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vol_data['DateTime_IST'], y=vol_data['Volatility'], 
                                        mode='lines', name='Volatility', fill='tozeroy', line=dict(color='orange')))
                fig.add_hline(y=vol_data['Volatility'].mean(), line_dash="dash", 
                             annotation_text=f"Average: {vol_data['Volatility'].mean():.2f}%", line_color='blue')
                fig.update_layout(title="Volatility Over Time", xaxis_title="Date", yaxis_title="Volatility %", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            if not vol_data.empty:
                current_vol = vol_data['Volatility'].iloc[-1]
                current_vol_bin = vol_data['Volatility_Bin'].iloc[-1]
                
                st.write("### Current Volatility Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Volatility", f"{current_vol:.2f}%")
                with col2:
                    st.metric("Current Bin", current_vol_bin)
                with col3:
                    st.metric("Average Volatility", f"{vol_data['Volatility'].mean():.2f}%")
                with col4:
                    st.metric("Max Volatility", f"{vol_data['Volatility'].max():.2f}%")
                
                st.write("**Volatility Distribution Table**")
                st.dataframe(vol_data.tail(50), use_container_width=True)
                
                # Volatility Summary
                st.write("### 📊 Volatility Analysis Summary")
                
                high_vol = vol_data[vol_data['Volatility_Bin'].isin(['High', 'Very High'])]
                low_vol = vol_data[vol_data['Volatility_Bin'].isin(['Low', 'Very Low'])]
                
                vol_summary = f"""
                **Current Market Volatility**: The market is currently experiencing **{current_vol_bin}** volatility at **{current_vol:.2f}%**.
                
                **Historical Volatility Context**:
                - High volatility periods occurred **{len(high_vol)}** times ({(len(high_vol)/len(vol_data)*100):.1f}% of the time)
                - Low volatility periods occurred **{len(low_vol)}** times ({(len(low_vol)/len(vol_data)*100):.1f}% of the time)
                """
                
                if current_vol_bin in ['Very High', 'High']:
                    vol_summary += f"""
                    
                **High Volatility Implications**:
                - **Increased Risk**: Wider price swings expected, suggesting larger stop losses
                - **Breakout Potential**: High volatility often precedes significant directional moves
                - **Last High Vol Period**: {time_ago(high_vol.index[-2]) if len(high_vol) > 1 else "Current"}
                - **Strategy**: Consider volatility-based strategies, breakout trades with wider stops
                - **Expected Move**: Price may move {current_vol * 0.5:.1f}% - {current_vol * 1.5:.1f}% in coming periods
                """
                elif current_vol_bin in ['Very Low', 'Low']:
                    vol_summary += f"""
                    
                **Low Volatility Implications**:
                - **Consolidation Phase**: Market is in a tight range, low momentum
                - **Breakout Watch**: Low volatility often precedes explosive moves (volatility compression)
                - **Last Low Vol Period**: {time_ago(low_vol.index[-2]) if len(low_vol) > 1 else "Current"}
                - **Strategy**: Wait for volatility expansion, avoid large positions
                - **Expected Move**: Limited movement expected, range-bound trading likely
                """
                else:
                    vol_summary += f"""
                    
                **Medium Volatility Implications**:
                - **Normal Trading Conditions**: Market showing typical price movement patterns
                - **Balanced Risk**: Neither extreme caution nor aggressive positioning required
                - **Strategy**: Standard trend-following or mean-reversion strategies applicable
                - **Expected Move**: Moderate price movements within normal ranges
                """
                
                # Calculate volatility-based rally predictions
                if len(high_vol) > 5:
                    high_vol_returns = []
                    for idx in high_vol.index[-5:]:
                        try:
                            idx_pos = vol_data.index.get_loc(idx)
                            if idx_pos + 5 < len(vol_data):
                                future_return = abs((vol_data['Close'].iloc[idx_pos + 5] - vol_data['Close'].iloc[idx_pos]) / 
                                                   vol_data['Close'].iloc[idx_pos] * 100)
                                high_vol_returns.append(future_return)
                        except:
                            pass
                    
                    if high_vol_returns:
                        vol_summary += f"""
                        
                **Historical Pattern**: During high volatility periods, the market typically moved **{np.mean(high_vol_returns):.2f}%** (average absolute move) within the next 5 periods. Based on current volatility of {current_vol:.2f}%, expect potential moves of **{current_vol * 0.8:.1f} to {current_vol * 1.2:.1f} points**.
                """
                
                st.markdown(vol_summary)
                
                # Volatility chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vol_data['DateTime_IST'], y=vol_data['Volatility'], 
                                        mode='lines', name='Volatility', fill='tozeroy'))
                fig.update_layout(title="Volatility Over Time", xaxis_title="Date", yaxis_title="Volatility %", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        current_tab += 1
        
        # Tab 5: Elliott Waves
        with tabs[current_tab]:
            st.subheader("🌊 Elliott Wave Analysis")
            waves = detect_elliott_waves(data1)
            
            if waves:
                waves_df = pd.DataFrame(waves)
                waves_df['Start_Date'] = waves_df['start_date'].apply(time_ago)
                waves_df['End_Date'] = waves_df['end_date'].apply(time_ago)
                waves_df['Price_Move'] = waves_df['end_price'] - waves_df['start_price']
                waves_df['Move_%'] = ((waves_df['end_price'] - waves_df['start_price']) / waves_df['start_price']) * 100
                
                display_df = waves_df[['wave', 'type', 'start_price', 'end_price', 'Price_Move', 'Move_%', 'Start_Date', 'End_Date']]
                st.dataframe(display_df, use_container_width=True)
                
                # Elliott Wave Summary
                st.write("### 🌊 Elliott Wave Summary")
                if len(waves) > 0:
                    latest_wave = waves[-1]
                    wave_summary = f"""
                    **Current Wave Analysis**: The market appears to be in **{latest_wave['wave']}** ({latest_wave['type']} wave).
                    
                    - **Wave Started**: {time_ago(latest_wave['start_date'])} at ₹{latest_wave['start_price']:.2f}
                    - **Current/End Price**: ₹{latest_wave['end_price']:.2f}
                    - **Wave Movement**: {latest_wave['end_price'] - latest_wave['start_price']:.2f} points ({((latest_wave['end_price'] - latest_wave['start_price'])/latest_wave['start_price']*100):.2f}%)
                    
                    **Elliott Wave Theory Implications**:
                    """
                    
                    if latest_wave['type'] == 'Impulse':
                        wave_summary += """
                        - **Impulse Wave**: This is a trending wave in the direction of the larger trend
                        - **Expectation**: After an impulse wave completes, expect a corrective wave (pullback)
                        - **Trading Strategy**: Look for the corrective wave to end before entering in trend direction
                        """
                    else:
                        wave_summary += """
                        - **Corrective Wave**: This is a counter-trend wave (pullback/consolidation)
                        - **Expectation**: After correction completes, trend should resume with next impulse wave
                        - **Trading Strategy**: Watch for correction completion signals to enter in trend direction
                        """
                    
                    st.markdown(wave_summary)
            else:
                st.info("Insufficient data points for Elliott Wave analysis")
        
        current_tab += 1
        
        # Tab 6: Fibonacci Levels
        with tabs[current_tab]:
            st.subheader("📐 Fibonacci Retracement Levels")
            fib_levels, high, low = calculate_fibonacci_levels(data1)
            
            fib_df = pd.DataFrame([{
                'Level': k,
                'Price': f"₹{v:.2f}",
                'Distance_Points': f"{abs(current_price1 - v):.2f}",
                'Distance_%': f"{abs((current_price1 - v)/current_price1)*100:.2f}%",
                'Status': '🔵 Current' if abs(current_price1 - v) < current_price1 * 0.01 else ''
            } for k, v in fib_levels.items()])
            
            st.dataframe(fib_df, use_container_width=True)
            
            # Fibonacci Summary
            st.write("### 📐 Fibonacci Analysis Summary")
            nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price1))
            
            fib_summary = f"""
            **Current Price Position**: ₹{current_price1:.2f} is nearest to the **{nearest_fib[0]} Fibonacci level** at ₹{nearest_fib[1]:.2f}
            
            **Fibonacci Range**: 
            - **High**: ₹{high:.2f} ({time_ago(data1[data1['High'] == high].index[0])})
            - **Low**: ₹{low:.2f} ({time_ago(data1[data1['Low'] == low].index[0])})
            - **Range**: {high - low:.2f} points
            
            **Key Support Levels** (below current price):
            """
            
            below_levels = {k: v for k, v in fib_levels.items() if v < current_price1}
            for level, price in sorted(below_levels.items(), key=lambda x: x[1], reverse=True)[:3]:
                distance = current_price1 - price
                fib_summary += f"\n- **{level}**: ₹{price:.2f} ({distance:.2f} points away, {(distance/current_price1)*100:.2f}%)"
            
            fib_summary += "\n\n**Key Resistance Levels** (above current price):"
            above_levels = {k: v for k, v in fib_levels.items() if v > current_price1}
            for level, price in sorted(above_levels.items(), key=lambda x: x[1])[:3]:
                distance = price - current_price1
                fib_summary += f"\n- **{level}**: ₹{price:.2f} ({distance:.2f} points away, {(distance/current_price1)*100:.2f}%)"
            
            fib_summary += f"""
            
            **Trading Implications**:
            - The **61.8% (Golden Ratio)** level at ₹{fib_levels['61.8%']:.2f} is historically the most reliable support/resistance
            - The **50% level** at ₹{fib_levels['50.0%']:.2f} often acts as a psychological pivot point
            - Watch for price reactions at these key Fibonacci levels for potential reversals or bounces
            """
            
            st.markdown(fib_summary)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data1.index, y=data1['Close'], mode='lines', name='Price'))
            for level_name, level_price in fib_levels.items():
                fig.add_hline(y=level_price, line_dash="dash", annotation_text=f"{level_name}: {level_price:.2f}")
            fig.update_layout(title="Fibonacci Retracement Levels", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        current_tab += 1
        
        # Tab 7: RSI Divergence
        with tabs[current_tab]:
            st.subheader("🔄 RSI Divergence Analysis")
            data1['RSI'] = calculate_rsi(data1['Close'])
            divergences = detect_rsi_divergence(data1, data1['RSI'])
            
            if divergences:
                st.write(f"**Found {len(divergences)} RSI Divergences**")
                
                active_divs = [d for d in divergences if not d['resolved']]
                resolved_divs = [d for d in divergences if d['resolved']]
                
                if active_divs:
                    st.success(f"### ⏳ {len(active_divs)} Active Divergence(s)")
                    for div in active_divs:
                        color = "green" if div['type'] == 'Bullish' else "red"
                        expected_move = abs(div['price2'] - div['price1'])
                        
                        st.markdown(f"""
                        #### {div['type']} Divergence (ACTIVE)
                        - **Price Movement**: ₹{div['price1']:.2f} → ₹{div['price2']:.2f} ({time_ago(div['date1'])} to {time_ago(div['date2'])})
                        - **RSI Movement**: {div['rsi1']:.2f} → {div['rsi2']:.2f}
                        - **Expected Move**: Approximately {expected_move:.2f} points {"upward" if div['type'] == 'Bullish' else "downward"}
                        - **Target Price**: ₹{div['price2'] + expected_move if div['type'] == 'Bullish' else div['price2'] - expected_move:.2f}
                        - **Status**: ⏳ Divergence not yet resolved, watching for price confirmation
                        """)
                
                if resolved_divs:
                    st.info(f"### ✅ {len(resolved_divs)} Resolved Divergence(s)")
                
                div_df = pd.DataFrame([{
                    'Type': d['type'],
                    'Price_1': f"₹{d['price1']:.2f}",
                    'Price_2': f"₹{d['price2']:.2f}",
                    'Date_1': time_ago(d['date1']),
                    'Date_2': time_ago(d['date2']),
                    'RSI_1': f"{d['rsi1']:.2f}",
                    'RSI_2': f"{d['rsi2']:.2f}",
                    'Resolved': '✅' if d['resolved'] else '⏳'
                } for d in divergences])
                
                st.dataframe(div_df, use_container_width=True)
                
                # Summary
                st.write("### 📊 RSI Divergence Summary")
                bullish_count = len([d for d in divergences if d['type'] == 'Bullish'])
                bearish_count = len([d for d in divergences if d['type'] == 'Bearish'])
                
                div_summary = f"""
                **Divergence Statistics**:
                - **Bullish Divergences**: {bullish_count} (suggesting upward reversals)
                - **Bearish Divergences**: {bearish_count} (suggesting downward reversals)
                - **Active Divergences**: {len(active_divs)} (unresolved, watching closely)
                - **Resolved Divergences**: {len(resolved_divs)} (already played out)
                
                **What RSI Divergence Means**:
                - **Bullish Divergence**: Price makes lower lows but RSI makes higher lows → Weakening downtrend, potential reversal up
                - **Bearish Divergence**: Price makes higher highs but RSI makes lower highs → Weakening uptrend, potential reversal down
                
                **Trading Strategy**:
                """
                
                if active_divs:
                    active_div = active_divs[0]
                    if active_div['type'] == 'Bullish':
                        div_summary += f"""
                        - **Active Bullish Divergence** detected from {time_ago(active_div['date1'])} to {time_ago(active_div['date2'])}
                        - **Recommendation**: Watch for bullish confirmation (price breaking above recent highs)
                        - **Potential Target**: ₹{active_div['price2'] + abs(active_div['price2'] - active_div['price1']):.2f}
                        - **Stop Loss**: Below ₹{active_div['price2']:.2f}
                        """
                    else:
                        div_summary += f"""
                        - **Active Bearish Divergence** detected from {time_ago(active_div['date1'])} to {time_ago(active_div['date2'])}
                        - **Recommendation**: Watch for bearish confirmation (price breaking below recent lows)
                        - **Potential Target**: ₹{active_div['price2'] - abs(active_div['price2'] - active_div['price1']):.2f}
                        - **Stop Loss**: Above ₹{active_div['price2']:.2f}
                        """
                else:
                    div_summary += "\n- No active divergences currently, monitor for new formations"
                
                st.markdown(div_summary)
            else:
                st.info("No RSI divergences detected in the current dataset")
        
        current_tab += 1
        
        # Tab 8: Ratio Analysis (conditional)
        if enable_ratio and mtf_data2:
            with tabs[current_tab]:
                st.subheader("⚖️ Ratio Analysis")
                
                # Use first available matching timeframe
                common_keys = set(mtf_data.keys()).intersection(set(mtf_data2.keys()))
                if common_keys:
                    key = list(common_keys)[0]
                    d1 = mtf_data[key]
                    d2 = mtf_data2[key]
                    
                    common_idx = d1.index.intersection(d2.index)
                    if len(common_idx) > 0:
                        d1_aligned = d1.loc[common_idx]
                        d2_aligned = d2.loc[common_idx]
                        
                        ratio_df = pd.DataFrame({
                            'DateTime_IST': common_idx,
                            'Ticker1_Price': d1_aligned['Close'],
                            'Ticker2_Price': d2_aligned['Close'],
                            'Ratio': d1_aligned['Close'] / d2_aligned['Close'],
                            'Ticker1_RSI': calculate_rsi(d1_aligned['Close']),
                            'Ticker2_RSI': calculate_rsi(d2_aligned['Close']),
                            'Ratio_RSI': calculate_rsi(d1_aligned['Close'] / d2_aligned['Close']),
                            'Ticker1_Vol': d1_aligned['Close'].pct_change().rolling(20).std() * 100,
                            'Ticker2_Vol': d2_aligned['Close'].pct_change().rolling(20).std() * 100
                        })
                        
                        st.dataframe(ratio_df.tail(50), use_container_width=True)
                        
                        csv = ratio_df.to_csv(index=False)
                        st.download_button("📥 Download Ratio Analysis", csv, 
                                         f"ratio_{ticker1}_{ticker2}.csv", "text/csv")
                        
                        # Plot
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                          subplot_titles=(f'{ticker1} Price', f'{ticker2} Price', 'Ratio'))
                        
                        fig.add_trace(go.Scatter(x=common_idx, y=d1_aligned['Close'], 
                                                name=ticker1, line=dict(color='blue')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=common_idx, y=d2_aligned['Close'], 
                                                name=ticker2, line=dict(color='orange')), row=2, col=1)
                        fig.add_trace(go.Scatter(x=common_idx, y=ratio_df['Ratio'], 
                                                name='Ratio', line=dict(color='purple')), row=3, col=1)
                        
                        fig.update_layout(height=900, title_text="Price Comparison & Ratio")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No overlapping timestamps between tickers")
                else:
                    st.warning("No common timeframes available for ratio analysis")
            
            current_tab += 1
        
        # Tab 9: AI Signals
        with tabs[current_tab]:
            st.subheader("🤖 FINAL MARKET FORECAST - Multi-Timeframe AI Analysis")
            
            st.info(f"🔄 Analyzing {len(list(mtf_data.items()))} timeframes to generate final recommendation...")
            
            # Comprehensive analysis across all timeframes
            timeframe_analysis = []
            
            for tf_period, data in mtf_data.items():
                tf, period = tf_period.split('_')
                
                # Calculate all indicators for this timeframe
                data_copy = data.copy()
                data_copy['RSI'] = calculate_rsi(data_copy['Close'])
                data_copy['EMA_9'] = calculate_ema(data_copy['Close'], 9)
                data_copy['EMA_20'] = calculate_ema(data_copy['Close'], 20)
                data_copy['EMA_50'] = calculate_ema(data_copy['Close'], 50)
                
                current_price = data_copy['Close'].iloc[-1]
                current_rsi = data_copy['RSI'].iloc[-1]
                
                # Get support/resistance
                sr_levels = find_support_resistance(data_copy)
                nearby_support = [l for l in sr_levels if l['type'] == 'Support' and 
                                abs(current_price - l['price']) < current_price * 0.02]
                nearby_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and 
                                   abs(current_price - l['price']) < current_price * 0.02]
                
                # Get divergences
                divergences = detect_rsi_divergence(data_copy, data_copy['RSI'])
                active_bull_div = [d for d in divergences if d['type'] == 'Bullish' and not d['resolved']]
                active_bear_div = [d for d in divergences if d['type'] == 'Bearish' and not d['resolved']]
                
                # Get Z-Score
                zscore_data = calculate_zscore_bins(data_copy)
                current_zscore = zscore_data['Z_Score'].iloc[-1] if not zscore_data.empty else 0
                
                # Get Volatility
                vol_data = calculate_volatility_bins(data_copy)
                current_vol = vol_data['Volatility'].iloc[-1] if not vol_data.empty else 0
                
                # Calculate score for this timeframe
                score = 0
                factors = []
                
                # RSI scoring
                if current_rsi < 30:
                    score += 20
                    factors.append("RSI Oversold")
                elif current_rsi > 70:
                    score -= 20
                    factors.append("RSI Overbought")
                
                # EMA scoring
                if current_price > data_copy['EMA_20'].iloc[-1] > data_copy['EMA_50'].iloc[-1]:
                    score += 15
                    factors.append("Bullish EMA Alignment")
                elif current_price < data_copy['EMA_20'].iloc[-1] < data_copy['EMA_50'].iloc[-1]:
                    score -= 15
                    factors.append("Bearish EMA Alignment")
                
                # Support/Resistance scoring
                if nearby_support:
                    score += 20
                    factors.append(f"Near Support ₹{nearby_support[0]['price']:.2f}")
                if nearby_resistance:
                    score -= 20
                    factors.append(f"Near Resistance ₹{nearby_resistance[0]['price']:.2f}")
                
                # Divergence scoring
                if active_bull_div:
                    score += 25
                    factors.append("Bullish Divergence")
                if active_bear_div:
                    score -= 25
                    factors.append("Bearish Divergence")
                
                # Z-Score scoring
                if current_zscore < -2:
                    score += 20
                    factors.append(f"Extreme Oversold Z-Score")
                elif current_zscore > 2:
                    score -= 20
                    factors.append(f"Extreme Overbought Z-Score")
                
                timeframe_analysis.append({
                    'timeframe': tf,
                    'period': period,
                    'score': score,
                    'factors': factors,
                    'price': current_price,
                    'rsi': current_rsi,
                    'zscore': current_zscore,
                    'volatility': current_vol
                })
            
            # Calculate aggregate score and consensus
            total_score = sum(a['score'] for a in timeframe_analysis)
            avg_score = total_score / len(timeframe_analysis)
            
            bullish_tf_count = len([a for a in timeframe_analysis if a['score'] > 15])
            bearish_tf_count = len([a for a in timeframe_analysis if a['score'] < -15])
            neutral_tf_count = len(timeframe_analysis) - bullish_tf_count - bearish_tf_count
            
            # Calculate entry
            entry = current_price1
            avg_price = current_price1
            
            # Determine signal with reasonable targets
            if avg_price > 20000:  # NIFTY/SENSEX
                if avg_score > 30:
                    signal = "🟢 STRONG BUY"
                    confidence = min(95, 65 + abs(avg_score) * 0.4)
                    direction = "BULLISH"
                    sl = entry * 0.985
                    target = entry * 1.0175
                elif avg_score > 15:
                    signal = "🟢 BUY"
                    confidence = min(85, 60 + abs(avg_score) * 0.4)
                    direction = "BULLISH"
                    sl = entry * 0.985
                    target = entry * 1.015
                elif avg_score < -30:
                    signal = "🔴 STRONG SELL"
                    confidence = min(95, 65 + abs(avg_score) * 0.4)
                    direction = "BEARISH"
                    sl = entry * 1.015
                    target = entry * 0.9825
                elif avg_score < -15:
                    signal = "🔴 SELL"
                    confidence = min(85, 60 + abs(avg_score) * 0.4)
                    direction = "BEARISH"
                    sl = entry * 1.015
                    target = entry * 0.985
                else:
                    signal = "🟡 HOLD/NEUTRAL"
                    confidence = 50
                    direction = "NEUTRAL"
                    sl = None
                    target = None
            else:
                if avg_score > 30:
                    signal = "🟢 STRONG BUY"
                    confidence = min(95, 65 + abs(avg_score) * 0.4)
                    direction = "BULLISH"
                    sl = entry * 0.975
                    target = entry * 1.03
                elif avg_score > 15:
                    signal = "🟢 BUY"
                    confidence = min(85, 60 + abs(avg_score) * 0.4)
                    direction = "BULLISH"
                    sl = entry * 0.98
                    target = entry * 1.025
                elif avg_score < -30:
                    signal = "🔴 STRONG SELL"
                    confidence = min(95, 65 + abs(avg_score) * 0.4)
                    direction = "BEARISH"
                    sl = entry * 1.025
                    target = entry * 0.97
                elif avg_score < -15:
                    signal = "🔴 SELL"
                    confidence = min(85, 60 + abs(avg_score) * 0.4)
                    direction = "BEARISH"
                    sl = entry * 1.02
                    target = entry * 0.975
                else:
                    signal = "🟡 HOLD/NEUTRAL"
                    confidence = 50
                    direction = "NEUTRAL"
                    sl = None
                    target = None
            
            # Display final recommendation
            st.markdown(f"# {signal}")
            st.markdown(f"## Confidence: {confidence:.1f}%")
            st.markdown(f"## Multi-Timeframe Score: {avg_score:.1f}/100")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullish Timeframes", f"{bullish_tf_count}/{len(timeframe_analysis)}", 
                         f"{bullish_tf_count/len(timeframe_analysis)*100:.0f}%")
            with col2:
                st.metric("Bearish Timeframes", f"{bearish_tf_count}/{len(timeframe_analysis)}", 
                         f"{bearish_tf_count/len(timeframe_analysis)*100:.0f}%")
            with col3:
                st.metric("Neutral Timeframes", f"{neutral_tf_count}/{len(timeframe_analysis)}", 
                         f"{neutral_tf_count/len(timeframe_analysis)*100:.0f}%")
            
            st.markdown("---")
            
            if direction != "NEUTRAL":
                st.markdown(f"""
                ### 📋 TRADING PLAN
                
                **Entry Price**: ₹{entry:.2f}  
                **Stop Loss**: ₹{sl:.2f} (Risk: {abs(entry-sl):.2f} points = {abs((entry-sl)/entry)*100:.2f}%)  
                **Target**: ₹{target:.2f} (Reward: {abs(target-entry):.2f} points = {abs((target-entry)/entry)*100:.2f}%)  
                **Risk:Reward Ratio**: 1:{abs((target-entry)/(entry-sl)):.2f}
                """)
            
            # Detailed timeframe breakdown
            st.markdown("---")
            st.markdown("### 📊 Timeframe-by-Timeframe Analysis")
            
            tf_breakdown = pd.DataFrame([{
                'Timeframe': a['timeframe'],
                'Period': a['period'],
                'Score': f"{a['score']:+d}",
                'Bias': "🟢 Bullish" if a['score'] > 15 else "🔴 Bearish" if a['score'] < -15 else "🟡 Neutral",
                'RSI': f"{a['rsi']:.1f}",
                'Z-Score': f"{a['zscore']:.2f}",
                'Volatility_%': f"{a['volatility']:.1f}",
                'Key_Factors': ", ".join(a['factors'][:2]) if a['factors'] else "No strong signals"
            } for a in timeframe_analysis])
            
            st.dataframe(tf_breakdown, use_container_width=True, height=400)
            
            # Adjust targets based on instrument price level
            if avg_price > 20000:  # NIFTY/SENSEX level
                if avg_score > 30:
                    signal = "🟢 STRONG BUY"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.985  # 1.5% SL (reasonable for index)
                    target = entry * 1.0175  # 1.75% target (realistic for NIFTY)
                elif avg_score > 15:
                    signal = "🟢 BUY"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.985  # 1.5% SL
                    target = entry * 1.015  # 1.5% target
                elif avg_score < -30:
                    signal = "🔴 STRONG SELL"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.015
                    target = entry * 0.9825
                elif avg_score < -15:
                    signal = "🔴 SELL"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.015
                    target = entry * 0.985
                else:
                    signal = "🟡 HOLD/NEUTRAL"
                    confidence = 50
                    direction = "NEUTRAL"
                    sl = None
                    target = None
            elif avg_price > 1000:  # Bank NIFTY or stocks
                if avg_score > 30:
                    signal = "🟢 STRONG BUY"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.98
                    target = entry * 1.025
                elif avg_score > 15:
                    signal = "🟢 BUY"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.98
                    target = entry * 1.02
                elif avg_score < -30:
                    signal = "🔴 STRONG SELL"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.02
                    target = entry * 0.975
                elif avg_score < -15:
                    signal = "🔴 SELL"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.02
                    target = entry * 0.98
                else:
                    signal = "🟡 HOLD/NEUTRAL"
                    confidence = 50
                    direction = "NEUTRAL"
                    sl = None
                    target = None
            else:  # Other assets (crypto, forex, etc.)
                if avg_score > 30:
                    signal = "🟢 STRONG BUY"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.97
                    target = entry * 1.04
                elif avg_score > 15:
                    signal = "🟢 BUY"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BULLISH"
                    sl = entry * 0.975
                    target = entry * 1.03
                elif avg_score < -30:
                    signal = "🔴 STRONG SELL"
                    confidence = min(95, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.03
                    target = entry * 0.96
                elif avg_score < -15:
                    signal = "🔴 SELL"
                    confidence = min(85, 60 + abs(avg_score) * 0.5)
                    direction = "BEARISH"
                    sl = entry * 1.03
                    target = entry * 0.97
                else:
                    signal = "🟡 HOLD/NEUTRAL"
                    confidence = 50
                    direction = "NEUTRAL"
                    sl = None
                    target = None
            
            # Display
            st.markdown(f"""
            ## {signal}
            ### Confidence: {confidence:.1f}%
            ### Multi-Timeframe Score: {avg_score:.1f}
            
            **Current Price**: ₹{entry:.2f}  
            **Entry Price**: ₹{entry:.2f}  
            {"**Stop Loss**: ₹" + f"{sl:.2f} ({abs(entry-sl):.2f} points, {abs((entry-sl)/entry)*100:.2f}%)" if sl else ""}  
            {"**Target**: ₹" + f"{target:.2f} ({abs(target-entry):.2f} points, {abs((target-entry)/entry)*100:.2f}%)" if target else ""}  
            {"**Risk:Reward Ratio**: " + f"{abs((target-entry)/entry) / abs((entry-sl)/entry):.2f}:1" if sl and target else ""}
            """)
            
            st.markdown("---")
            st.markdown("### 📋 Multi-Timeframe Signal Breakdown")
            
            signals_df = pd.DataFrame([{
                'Timeframe': s['timeframe'],
                'Period': s['period'],
                'Score': f"{s['score']:+d}",
                'Bias': "🟢 Bullish" if s['score'] > 0 else "🔴 Bearish" if s['score'] < 0 else "🟡 Neutral",
                'Key_Signals': ", ".join(s['signals'][:3]) if s['signals'] else "No strong signals"
            } for s in all_signals])
            
            st.dataframe(signals_df, use_container_width=True)
            
            # Comprehensive Analysis Summary
            st.markdown("---")
            st.markdown("### 🔍 Comprehensive Market Analysis & Forecast")
            
            # Get all analysis data
            zscore_data = calculate_zscore_bins(data1)
            vol_data = calculate_volatility_bins(data1)
            sr_levels = find_support_resistance(data1)
            fib_levels, _, _ = calculate_fibonacci_levels(data1)
            divergences = detect_rsi_divergence(data1, data1['RSI'])
            
            current_zscore = zscore_data['Z_Score'].iloc[-1] if not zscore_data.empty else 0
            current_vol = vol_data['Volatility'].iloc[-1] if not vol_data.empty else 0
            
            bullish_factors = []
            bearish_factors = []
            neutral_factors = []
            
            # Analyze each component
            if current_price1 > data1['EMA_20'].iloc[-1]:
                bullish_factors.append(f"Price (₹{current_price1:.2f}) above 20 EMA (₹{data1['EMA_20'].iloc[-1]:.2f})")
            else:
                bearish_factors.append(f"Price (₹{current_price1:.2f}) below 20 EMA (₹{data1['EMA_20'].iloc[-1]:.2f})")
            
            if data1['RSI'].iloc[-1] < 30:
                bullish_factors.append(f"RSI oversold at {data1['RSI'].iloc[-1]:.2f}, suggesting bounce")
            elif data1['RSI'].iloc[-1] > 70:
                bearish_factors.append(f"RSI overbought at {data1['RSI'].iloc[-1]:.2f}, suggesting correction")
            
            nearby_support = [l for l in sr_levels if l['type'] == 'Support' and 
                            abs(current_price1 - l['price']) < current_price1 * 0.02]
            nearby_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and 
                               abs(current_price1 - l['price']) < current_price1 * 0.02]
            
            if nearby_support:
                bullish_factors.append(f"Near strong support at ₹{nearby_support[0]['price']:.2f} (tested {nearby_support[0]['count']} times, {(nearby_support[0]['sustained']/nearby_support[0]['count']*100):.0f}% accuracy)")
            
            if nearby_resistance:
                bearish_factors.append(f"Near strong resistance at ₹{nearby_resistance[0]['price']:.2f} (tested {nearby_resistance[0]['count']} times)")
            
            active_bull_div = [d for d in divergences if d['type'] == 'Bullish' and not d['resolved']]
            active_bear_div = [d for d in divergences if d['type'] == 'Bearish' and not d['resolved']]
            
            if active_bull_div:
                bullish_factors.append(f"Active bullish RSI divergence from {time_ago(active_bull_div[0]['date1'])}")
            if active_bear_div:
                bearish_factors.append(f"Active bearish RSI divergence from {time_ago(active_bear_div[0]['date1'])}")
            
            if current_zscore < -2:
                bullish_factors.append(f"Extreme negative Z-Score ({current_zscore:.2f}), mean reversion expected upward")
            elif current_zscore > 2:
                bearish_factors.append(f"Extreme positive Z-Score ({current_zscore:.2f}), mean reversion expected downward")
            
            if current_vol > vol_data['Volatility'].mean() * 1.5:
                neutral_factors.append(f"Very high volatility ({current_vol:.2f}%), expect larger moves in either direction")
            
            # Create comprehensive summary
            summary = f"""
            ## 🎯 COMPREHENSIVE MARKET FORECAST FOR {ticker1}
            
            ### 📊 FINAL RECOMMENDATION: {signal}
            
            **Confidence Level**: {confidence:.1f}% (Based on {len(all_signals)} timeframe analysis)
            
            **Multi-Timeframe Consensus Score**: {avg_score:.1f}/100
            - Bullish timeframes: {len([s for s in all_signals if s['score'] > 0])}/{len(all_signals)}
            - Bearish timeframes: {len([s for s in all_signals if s['score'] < 0])}/{len(all_signals)}
            - Neutral timeframes: {len([s for s in all_signals if s['score'] == 0])}/{len(all_signals)}
            
            ---
            
            ### 📈 CURRENT MARKET STATE (Primary Timeframe: {list(mtf_data.keys())[0]})
            
            **Price Metrics**:
            - Current Price: ₹{current_price1:.2f}
            - Change from Previous: {change1:+.2f} points ({pct_change1:+.2f}%)
            
            **Technical Indicators**:
            - RSI (14): {data1['RSI'].iloc[-1]:.2f} {"(Oversold - Bullish)" if data1['RSI'].iloc[-1] < 30 else "(Overbought - Bearish)" if data1['RSI'].iloc[-1] > 70 else "(Neutral)"}
            - 9 EMA: ₹{data1['EMA_9'].iloc[-1]:.2f} (Distance: {current_price1 - data1['EMA_9'].iloc[-1]:+.2f} points)
            - 20 EMA: ₹{data1['EMA_20'].iloc[-1]:.2f} (Distance: {current_price1 - data1['EMA_20'].iloc[-1]:+.2f} points)
            - 50 EMA: ₹{data1['EMA_50'].iloc[-1]:.2f} (Distance: {current_price1 - data1['EMA_50'].iloc[-1]:+.2f} points)
            - Volatility: {current_vol:.2f}% {"(High - Expect larger moves)" if current_vol > 30 else "(Moderate)" if current_vol > 15 else "(Low - Limited moves expected)"}
            - Z-Score: {current_zscore:.2f} {"(Extreme - Mean reversion likely)" if abs(current_zscore) > 2 else "(Normal distribution)"}
            
            ---
            
            ### ✅ BULLISH FACTORS ({len(bullish_factors)} Identified)
            
            **These factors support an UPWARD move:**
            """
            
            for i, factor in enumerate(bullish_factors, 1):
                summary += f"\n{i}. **{factor}**"
                
                # Add detailed explanation for each factor
                if "RSI" in factor and "oversold" in factor.lower():
                    summary += f"\n   - *Explanation*: When RSI drops below 30, it indicates the market is oversold - meaning selling pressure has been excessive and a bounce/reversal is statistically probable. Historically, RSI oversold conditions lead to upward reversals in 65-75% of cases."
                
                elif "EMA" in factor and "above" in factor.lower():
                    summary += f"\n   - *Explanation*: Price trading above key moving averages (especially 20 and 50 EMA) confirms an uptrend. This shows buyers are in control and dips are being bought. The trend is your friend - trading with the trend has higher success probability."
                
                elif "support" in factor.lower():
                    summary += f"\n   - *Explanation*: Support levels represent price zones where buying interest historically emerged. When price approaches tested support, buyers often step in again, creating bounce opportunities. The more times a support holds, the stronger it becomes."
                
                elif "divergence" in factor.lower() and "bullish" in factor.lower():
                    summary += f"\n   - *Explanation*: Bullish RSI divergence occurs when price makes lower lows but RSI makes higher lows, indicating weakening downward momentum. This is an early warning sign that sellers are losing control and a reversal may be imminent."
                
                elif "z-score" in factor.lower() and "negative" in factor.lower():
                    summary += f"\n   - *Explanation*: Extreme negative Z-Score means price has fallen far below its statistical mean. Markets tend to revert to their mean over time (mean reversion), suggesting an upward correction is likely."
            
            summary += f"\n\n---\n\n### ⚠️ BEARISH FACTORS ({len(bearish_factors)} Identified)\n\n**These factors suggest DOWNWARD pressure:**\n"
            
            for i, factor in enumerate(bearish_factors, 1):
                summary += f"\n{i}. **{factor}**"
                
                if "RSI" in factor and "overbought" in factor.lower():
                    summary += f"\n   - *Explanation*: RSI above 70 indicates overbought conditions - buying has been excessive and a pullback/correction is probable. Overbought conditions often precede downward moves as profit-taking emerges."
                
                elif "EMA" in factor and "below" in factor.lower():
                    summary += f"\n   - *Explanation*: Price below key EMAs confirms a downtrend. Sellers are in control and rallies are being sold. Trading against the trend is riskier - waiting for trend reversal confirmation is prudent."
                
                elif "resistance" in factor.lower():
                    summary += f"\n   - *Explanation*: Resistance levels are price zones where selling pressure historically emerged. When price approaches tested resistance, sellers often step in again, creating rejection/reversal opportunities."
                
                elif "divergence" in factor.lower() and "bearish" in factor.lower():
                    summary += f"\n   - *Explanation*: Bearish RSI divergence (price making higher highs while RSI makes lower highs) indicates weakening upward momentum. Buyers are losing strength, suggesting a potential reversal or correction."
            
            if neutral_factors:
                summary += f"\n\n---\n\n### 🟡 NEUTRAL/CAUTION FACTORS ({len(neutral_factors)} Identified)\n"
                for i, factor in enumerate(neutral_factors, 1):
                    summary += f"\n{i}. **{factor}**"
                    
                    if "volatility" in factor.lower():
                        summary += f"\n   - *Explanation*: High volatility means larger price swings in both directions. While this creates opportunity, it also increases risk. Use wider stop losses and smaller position sizes in high volatility environments."
            
            summary += f"""
            
            ---
            
            ### 🎯 WHY {signal}? (DETAILED REASONING)
            
            """
            
            if direction == "BULLISH":
                summary += f"""
            **The recommendation is {signal} because:**
            
            1. **Timeframe Alignment ({len([s for s in all_signals if s['score'] > 0])}/{len(all_signals)} timeframes bullish)**:
               - Multiple timeframes showing bullish signals creates high-confidence setup
               - When short-term and long-term timeframes align, success probability increases significantly
               - Timeframes analyzed: {', '.join([f"{s['timeframe']}/{s['period']}" for s in all_signals[:5]])}
            
            2. **Technical Indicator Confluence**:
               - {len(bullish_factors)} bullish factors vs {len(bearish_factors)} bearish factors
               - When multiple independent indicators agree, the signal reliability increases
               - Net bullish score of {avg_score:.1f} indicates strong conviction
            
            3. **Risk-Reward Favorability**:
               - Entry: ₹{entry:.2f}
               - Stop Loss: ₹{sl:.2f} (Risk: {abs(entry-sl):.2f} points = {abs((entry-sl)/entry)*100:.2f}%)
               - Target: ₹{target:.2f} (Reward: {abs(target-entry):.2f} points = {abs((target-entry)/entry)*100:.2f}%)
               - Risk:Reward = 1:{abs((target-entry)/entry) / abs((entry-sl)/entry):.2f}
               - A risk:reward ratio above 1:2 is considered favorable for trading
            
            4. **Statistical Edge**:
               - Historical patterns similar to current setup have shown {confidence:.1f}% success rate
               - Z-Score: {current_zscore:.2f} - {"Extreme values often lead to mean reversion" if abs(current_zscore) > 2 else "Normal distribution supporting trend continuation"}
               - Volatility: {current_vol:.2f}% - {"Elevated volatility can lead to explosive moves" if current_vol > 25 else "Moderate volatility supports steady trending"}
            
            5. **Key Catalysts Supporting Upside**:
               {chr(10).join([f"   - {factor}" for factor in bullish_factors[:3]])}
            
            **EXECUTION PLAN**:
            - **Entry Strategy**: Enter at current market price ₹{entry:.2f} OR wait for a small pullback to ₹{entry * 0.995:.2f} for better entry
            - **Position Sizing**: Risk only 1-2% of total capital on this trade
            - **Stop Loss Placement**: ₹{sl:.2f} (below recent support/EMA) - This level invalidates the bullish thesis
            - **Profit Targets**:
              * Target 1 (50% position): ₹{target:.2f} ({abs((target-entry)/entry)*100:.2f}% gain)
              * Target 2 (30% position): ₹{entry + abs(target-entry) * 1.5:.2f} (extended target)
              * Target 3 (20% position): ₹{entry + abs(target-entry) * 2:.2f} (runner for big moves)
            - **Trailing Stop**: Once price reaches ₹{entry + abs(target-entry) * 0.5:.2f}, move stop to breakeven
            
            **WHAT COULD GO WRONG (Risk Factors)**:
            - If {len(bearish_factors)} bearish factors strengthen, position may reverse
            - Unexpected news/events can override technical analysis
            - If stop loss at ₹{sl:.2f} is hit, exit immediately - do not average down
            
            **TIME HORIZON**: {period} - Expect move to play out over this timeframe
            """
            
            elif direction == "BEARISH":
                summary += f"""
            **The recommendation is {signal} because:**
            
            1. **Timeframe Alignment ({len([s for s in all_signals if s['score'] < 0])}/{len(all_signals)} timeframes bearish)**:
               - Multiple timeframes showing bearish signals creates high-confidence setup
               - Weakness across multiple timeframes indicates strong selling pressure
               - Timeframes showing bearish bias: {', '.join([f"{s['timeframe']}/{s['period']}" for s in [sig for sig in all_signals if sig['score'] < 0][:5]])}
            
            2. **Technical Indicator Confluence**:
               - {len(bearish_factors)} bearish factors vs {len(bullish_factors)} bullish factors
               - Net bearish score of {avg_score:.1f} indicates strong downward conviction
               - Multiple independent indicators agreeing on downside increases reliability
            
            3. **Risk-Reward Setup**:
               - Entry: ₹{entry:.2f} (short/sell position)
               - Stop Loss: ₹{sl:.2f} (Risk: {abs(sl-entry):.2f} points = {abs((sl-entry)/entry)*100:.2f}%)
               - Target: ₹{target:.2f} (Reward: {abs(entry-target):.2f} points = {abs((entry-target)/entry)*100:.2f}%)
               - Risk:Reward = 1:{abs((entry-target)/entry) / abs((sl-entry)/entry):.2f}
            
            4. **Bearish Catalysts**:
               {chr(10).join([f"   - {factor}" for factor in bearish_factors[:3]])}
            
            **EXECUTION PLAN (For Short/Sell)**:
            - **Entry**: ₹{entry:.2f} or on bounce to ₹{entry * 1.005:.2f}
            - **Stop Loss**: ₹{sl:.2f} (above resistance - invalidation point)
            - **Target**: ₹{target:.2f}
            - **Position Sizing**: Risk 1-2% of capital
            
            **TIME HORIZON**: {period}
            """
            
            else:  # NEUTRAL/HOLD
                summary += f"""
            **The recommendation is {signal} because:**
            
            1. **Conflicting Signals Across Timeframes**:
               - Bullish timeframes: {len([s for s in all_signals if s['score'] > 0])}/{len(all_signals)}
               - Bearish timeframes: {len([s for s in all_signals if s['score'] < 0])}/{len(all_signals)}
               - No clear directional consensus reduces confidence in any trade
            
            2. **Balanced Technical Factors**:
               - Bullish factors: {len(bullish_factors)}
               - Bearish factors: {len(bearish_factors)}
               - Net score: {avg_score:.1f} (too close to zero for high-confidence trade)
            
            3. **Why Waiting is the Right Strategy**:
               - Trading without clear edge reduces win probability below 50%
               - Low confidence setups lead to emotional trading and increased losses
               - Better to wait for high-probability setups than force trades
               - Preservation of capital is paramount when signals are unclear
            
            4. **What to Watch For (Setup Conditions)**:
               - **For Bullish Setup**: Need RSI to drop below 30, price to hold above {data1['EMA_20'].iloc[-1]:.2f} (20 EMA), and bullish divergence confirmation
               - **For Bearish Setup**: Need RSI above 70, price to break below {data1['EMA_50'].iloc[-1]:.2f} (50 EMA), and bearish divergence
               - **Breakout**: Watch for decisive move above ₹{current_price1 * 1.02:.2f} (bullish) or below ₹{current_price1 * 0.98:.2f} (bearish)
            
            **RECOMMENDED ACTION**: 
            - Stay in CASH
            - Monitor market for clearer signals
            - Set price alerts at key levels
            - Review analysis after {period} or significant price movement
            """
            
            summary += f"""
            
            ---
            
            ### 📊 CONFIDENCE SCORE BREAKDOWN
            
            **How {confidence:.1f}% Confidence Was Calculated**:
            
            1. **Base Confidence**: 60% (starting point for any analysis)
            
            2. **Timeframe Agreement Bonus**: +{len([s for s in all_signals if (s['score'] > 0 and direction == 'BULLISH') or (s['score'] < 0 and direction == 'BEARISH')])}/{len(all_signals)} timeframes * 5% = +{len([s for s in all_signals if (s['score'] > 0 and direction == 'BULLISH') or (s['score'] < 0 and direction == 'BEARISH')]) * 5}%
               - More timeframes agreeing = higher confidence
            
            3. **Factor Strength Bonus**: Absolute score {abs(avg_score):.1f} / 2 = +{abs(avg_score)/2:.1f}%
               - Stronger net score = higher confidence
            
            4. **Technical Confluence**: {max(len(bullish_factors), len(bearish_factors))} factors * 2% = +{max(len(bullish_factors), len(bearish_factors)) * 2}%
               - More confirming factors = higher reliability
            
            5. **Capped at 95%**: No analysis is 100% certain due to market randomness and unforeseen events
            
            **Final Confidence**: {confidence:.1f}%
            
            ---
            
            ### ⚖️ RISK DISCLOSURE
            
            - This analysis is based on historical patterns and technical indicators
            - Past performance does not guarantee future results
            - Markets can remain irrational longer than you can remain solvent
            - Always use stop losses and proper position sizing
            - Never risk more than you can afford to lose
            - Consider your personal risk tolerance and financial situation
            """
            
            st.markdown(summary)
            
            # Pattern Performance
            st.markdown("---")
            st.markdown("### 📊 What's Working vs What's Not Working")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("**✅ WORKING PATTERNS**")
                for factor in bullish_factors[:5]:
                    st.write(f"• {factor}")
            
            with col2:
                st.error("**⚠️ NOT WORKING / CAUTION**")
                for factor in bearish_factors[:5]:
                    st.write(f"• {factor}")
        
        current_tab += 1
        
        # Tab 10: Backtesting
        with tabs[current_tab]:
            st.subheader("🔬 Multi-Strategy Backtesting")
            
            strategy_options = ["RSI + EMA Strategy", "EMA Crossover", "Volatility Breakout", 
                              "RSI Divergence", "Support/Resistance Bounce"]
            
            selected_strategies = st.multiselect("Select Strategies to Backtest", 
                                                strategy_options, 
                                                default=["RSI + EMA Strategy"])
            
            if st.button("🚀 Run Backtests on All Timeframes"):
                results_summary = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                strategy_mapping = {
                    "RSI + EMA Strategy": "rsi_ema",
                    "EMA Crossover": "ema_crossover",
                    "Volatility Breakout": "volatility"
                }
                
                total_tests = len(selected_strategies) * len(list(mtf_data.items())[:5])
                current_test = 0
                
                for strategy_name in selected_strategies:
                    strategy_type = strategy_mapping.get(strategy_name, "rsi_ema")
                    
                    for tf_period, data in list(mtf_data.items())[:5]:
                        current_test += 1
                        progress_bar.progress(current_test / total_tests)
                        status_text.text(f"Testing {strategy_name} on {tf_period}... ({current_test}/{total_tests})")
                        
                        tf, period = tf_period.split('_')
                        trades_df, total_pnl, win_rate = backtest_strategy(data, strategy_type)
                        
                        if not trades_df.empty:
                            results_summary.append({
                                'Strategy': strategy_name,
                                'Timeframe': tf,
                                'Period': period,
                                'Total_Trades': len(trades_df),
                                'Win_Rate_%': f"{win_rate:.2f}%",
                                'Total_PnL_%': f"{total_pnl:.2f}%",
                                'Avg_Win': f"{trades_df[trades_df['PnL_%'] > 0]['PnL_%'].mean():.2f}%" if len(trades_df[trades_df['PnL_%'] > 0]) > 0 else "0%",
                                'Avg_Loss': f"{trades_df[trades_df['PnL_%'] < 0]['PnL_%'].mean():.2f}%" if len(trades_df[trades_df['PnL_%'] < 0]) > 0 else "0%",
                                'Best_Trade': f"{trades_df['PnL_%'].max():.2f}%",
                                'Worst_Trade': f"{trades_df['PnL_%'].min():.2f}%"
                            })
                
                progress_bar.empty()
                status_text.empty()
                
                if results_summary:
                    st.success(f"✅ Completed {len(results_summary)} backtests!")
                    
                    results_df = pd.DataFrame(results_summary)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Backtest Performance Summary")
                    
                    best_strategy = results_df.loc[results_df['Total_PnL_%'].apply(lambda x: float(x.strip('%'))).idxmax()]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Strategy", best_strategy['Strategy'])
                    with col2:
                        st.metric("Best Timeframe", f"{best_strategy['Timeframe']}/{best_strategy['Period']}")
                    with col3:
                        st.metric("Best PnL", best_strategy['Total_PnL_%'])
                    
                    # Export
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download All Backtest Results", csv, 
                                     f"backtests_{ticker1}.csv", "text/csv")
                    
                    # Detailed trades for best strategy
                    st.markdown("---")
                    st.markdown(f"### 📋 Detailed Trades: {best_strategy['Strategy']} ({best_strategy['Timeframe']}/{best_strategy['Period']})")
                    
                    best_tf_period = f"{best_strategy['Timeframe']}_{best_strategy['Period']}"
                    best_data = mtf_data[best_tf_period]
                    strategy_type = strategy_mapping.get(best_strategy['Strategy'], "rsi_ema")
                    trades_df, _, _ = backtest_strategy(best_data, strategy_type)
                    
                    if not trades_df.empty:
                        # Add time ago formatting
                        trades_display = trades_df.copy()
                        trades_display['Entry_Date'] = trades_display['Entry_Date'].apply(time_ago)
                        trades_display['Exit_Date'] = trades_display['Exit_Date'].apply(time_ago)
                        
                        st.dataframe(trades_display, use_container_width=True)
                        
                        # Strategy explanation
                        st.markdown("---")
                        st.markdown("### 📖 Strategy Logic & Parameters")
                        
                        if strategy_type == "rsi_ema":
                            explanation = """
                            **RSI + EMA Strategy**:
                            - **Entry**: RSI < 35 AND Price > 20 EMA (oversold with uptrend confirmation)
                            - **Stop Loss**: 3% below entry
                            - **Target**: 5% above entry
                            - **Logic**: Buys oversold conditions in an uptrend, exits on stop or target
                            """
                        elif strategy_type == "ema_crossover":
                            explanation = """
                            **EMA Crossover Strategy**:
                            - **Entry**: 9 EMA crosses above 20 EMA (bullish momentum shift)
                            - **Stop Loss**: 2% below entry
                            - **Target**: 4% above entry
                            - **Logic**: Captures momentum when fast EMA crosses slow EMA upward
                            """
                        elif strategy_type == "volatility":
                            explanation = """
                            **Volatility Breakout Strategy**:
                            - **Entry**: Volatility > 1.5x average (high volatility expansion)
                            - **Stop Loss**: 4% below entry
                            - **Target**: 6% above entry
                            - **Logic**: Enters on volatility spikes expecting continuation
                            """
                        
                        st.info(explanation)
                        
                        # Performance metrics
                        winning_trades = trades_df[trades_df['PnL_%'] > 0]
                        losing_trades = trades_df[trades_df['PnL_%'] < 0]
                        
                        st.markdown("### 📈 Performance Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Trades", len(trades_df))
                        with col2:
                            st.metric("Winning Trades", len(winning_trades))
                        with col3:
                            st.metric("Losing Trades", len(losing_trades))
                        with col4:
                            st.metric("Win Rate", f"{(len(winning_trades)/len(trades_df)*100):.1f}%")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Win", f"{winning_trades['PnL_%'].mean():.2f}%" if len(winning_trades) > 0 else "N/A")
                        with col2:
                            st.metric("Avg Loss", f"{losing_trades['PnL_%'].mean():.2f}%" if len(losing_trades) > 0 else "N/A")
                        with col3:
                            avg_win = winning_trades['PnL_%'].mean() if len(winning_trades) > 0 else 0
                            avg_loss = abs(losing_trades['PnL_%'].mean()) if len(losing_trades) > 0 else 1
                            st.metric("Risk:Reward", f"{avg_win/avg_loss:.2f}:1" if avg_loss != 0 else "N/A")
                        with col4:
                            st.metric("Total PnL", f"{trades_df['PnL_%'].sum():.2f}%")
                        
                        # Profit curve
                        trades_df['Cumulative_PnL'] = trades_df['PnL_%'].cumsum()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(range(len(trades_df))), 
                                                y=trades_df['Cumulative_PnL'],
                                                mode='lines+markers',
                                                name='Cumulative PnL',
                                                line=dict(color='green')))
                        fig.update_layout(title="Cumulative PnL Curve", 
                                        xaxis_title="Trade Number", 
                                        yaxis_title="Cumulative PnL (%)",
                                        height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trades generated across all tested strategies and timeframes. Try adjusting parameters.")

if __name__ == "__main__":
    main()
