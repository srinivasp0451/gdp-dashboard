import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Advanced Trading Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 20px;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .bullish {color: #00ff00; font-weight: bold;}
    .bearish {color: #ff0000; font-weight: bold;}
    .neutral {color: #ffaa00; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px; background-color: #f0f2f6; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
INDIAN_TZ = pytz.timezone('Asia/Kolkata')

TIMEFRAME_PERIODS = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '1mo'],
    '15m': ['1mo'],
    '30m': ['1mo'],
    '1h': ['1mo'],
    '4h': ['1mo'],
    '1d': ['1mo', '1y', '2y', '5y'],
    '1wk': ['1mo', '1y', '5y', '10y', '15y', '20y'],
    '1mo': ['1y', '2y', '5y', '10y', '15y', '20y', '25y', '30y']
}

ASSET_GROUPS = {
    'Indian Indices': {
        'NIFTY 50': '^NSEI',
        'BANK NIFTY': '^NSEBANK',
        'SENSEX': '^BSESN',
        'NIFTY IT': '^CNXIT',
        'NIFTY PHARMA': '^CNXPHARMA'
    },
    'Crypto': {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Binance Coin': 'BNB-USD'
    },
    'Forex': {
        'USD/INR': 'USDINR=X',
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X'
    },
    'Commodities': {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil': 'CL=F'
    }
}

# NIFTY 50 constituents with weights (approximate)
NIFTY50_TICKERS = {
    'RELIANCE.NS': 10.50, 'TCS.NS': 7.20, 'HDFCBANK.NS': 9.80, 'INFY.NS': 6.10,
    'ICICIBANK.NS': 7.50, 'HINDUNILVR.NS': 4.80, 'ITC.NS': 4.20, 'SBIN.NS': 3.90,
    'BHARTIARTL.NS': 3.80, 'KOTAKBANK.NS': 4.10, 'LT.NS': 3.50, 'AXISBANK.NS': 3.20,
    'BAJFINANCE.NS': 3.60, 'ASIANPAINT.NS': 2.80, 'MARUTI.NS': 2.70, 'HCLTECH.NS': 2.60,
    'WIPRO.NS': 2.10, 'ULTRACEMCO.NS': 2.40, 'TITAN.NS': 2.30, 'SUNPHARMA.NS': 2.20,
    'NESTLEIND.NS': 2.00, 'ONGC.NS': 1.90, 'NTPC.NS': 1.80, 'TATAMOTORS.NS': 1.70,
    'POWERGRID.NS': 1.60, 'M&M.NS': 1.90, 'BAJAJFINSV.NS': 2.10, 'TECHM.NS': 1.80,
    'ADANIPORTS.NS': 1.50, 'COALINDIA.NS': 1.40, 'TATASTEEL.NS': 1.60, 'HINDALCO.NS': 1.30,
    'JSWSTEEL.NS': 1.40, 'DIVISLAB.NS': 1.50, 'DRREDDY.NS': 1.30, 'CIPLA.NS': 1.20,
    'BRITANNIA.NS': 1.50, 'EICHERMOT.NS': 1.30, 'HEROMOTOCO.NS': 1.20, 'GRASIM.NS': 1.10,
    'BPCL.NS': 1.20, 'UPL.NS': 0.90, 'INDUSINDBK.NS': 1.50, 'APOLLOHOSP.NS': 1.30,
    'ADANIENT.NS': 1.40, 'TATACONSUM.NS': 1.00, 'SBILIFE.NS': 1.10, 'HDFCLIFE.NS': 1.20,
    'BAJAJ-AUTO.NS': 1.30, 'SHREECEM.NS': 0.90
}

BANKNIFTY_TICKERS = {
    'HDFCBANK.NS': 24.50, 'ICICIBANK.NS': 22.80, 'SBIN.NS': 11.20, 'KOTAKBANK.NS': 12.30,
    'AXISBANK.NS': 10.50, 'INDUSINDBK.NS': 6.80, 'AUBANK.NS': 3.90, 'BANDHANBNK.NS': 2.40,
    'FEDERALBNK.NS': 2.10, 'IDFCFIRSTB.NS': 1.80, 'PNB.NS': 1.70
}

# ==================== UTILITY FUNCTIONS ====================
def safe_yfinance_call(ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """Fetch data with rate limiting and error handling"""
    try:
        delay = random.uniform(1.0, 1.5)
        time.sleep(delay)
        
        data = yf.download(ticker, progress=False, **kwargs)
        
        if data.empty:
            return None
            
        # Flatten multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Handle timezone
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(INDIAN_TZ)
        else:
            data.index = data.index.tz_convert(INDIAN_TZ)
        
        # Select OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in data.columns]
        
        return data[available_cols]
    
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)}")
        return None

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return upper, middle, lower

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr = calculate_atr(high, low, close, 1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_ema_crossover_angle(ema_fast: pd.Series, ema_slow: pd.Series) -> float:
    """Calculate angle between EMAs in degrees"""
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return 0.0
    
    fast_slope = ema_fast.iloc[-1] - ema_fast.iloc[-2]
    slow_slope = ema_slow.iloc[-1] - ema_slow.iloc[-2]
    
    angle = np.arctan(fast_slope - slow_slope) * (180 / np.pi)
    return abs(angle)

def detect_gap(data: pd.DataFrame) -> Dict:
    """Detect price gaps"""
    if len(data) < 2:
        return {'has_gap': False, 'gap_type': None, 'gap_size': 0}
    
    prev_close = data['Close'].iloc[-2]
    curr_open = data['Open'].iloc[-1]
    gap_size = curr_open - prev_close
    gap_pct = (gap_size / prev_close) * 100
    
    if abs(gap_pct) > 0.5:  # Gap threshold 0.5%
        gap_type = 'up' if gap_size > 0 else 'down'
        return {
            'has_gap': True,
            'gap_type': gap_type,
            'gap_size': gap_size,
            'gap_pct': gap_pct,
            'prev_close': prev_close,
            'curr_open': curr_open
        }
    
    return {'has_gap': False, 'gap_type': None, 'gap_size': 0}

def identify_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict:
    """Identify support and resistance levels"""
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    
    resistance_levels = highs[highs == data['High']].dropna().unique()[-3:]
    support_levels = lows[lows == data['Low']].dropna().unique()[-3:]
    
    return {
        'resistance': sorted(resistance_levels, reverse=True),
        'support': sorted(support_levels, reverse=True)
    }

def calculate_pivot_points(data: pd.DataFrame) -> Dict:
    """Calculate pivot points"""
    if len(data) < 1:
        return {}
    
    high = data['High'].iloc[-1]
    low = data['Low'].iloc[-1]
    close = data['Close'].iloc[-1]
    
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }

def analyze_price_action(data: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 15) -> Dict:
    """Analyze price action for trading signals"""
    if len(data) < max(ema_fast, ema_slow) + 10:
        return {'signal': 'NEUTRAL', 'reason': 'Insufficient data'}
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Calculate indicators
    ema_f = calculate_ema(close, ema_fast)
    ema_s = calculate_ema(close, ema_slow)
    rsi = calculate_rsi(close)
    atr = calculate_atr(high, low, close)
    macd_line, signal_line, histogram = calculate_macd(close)
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close)
    adx = calculate_adx(high, low, close)
    
    # Current values
    curr_close = close.iloc[-1]
    curr_ema_f = ema_f.iloc[-1]
    curr_ema_s = ema_s.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    curr_atr = atr.iloc[-1]
    curr_macd = histogram.iloc[-1]
    curr_adx = adx.iloc[-1]
    
    # EMA crossover analysis
    crossover_angle = calculate_ema_crossover_angle(ema_f, ema_s)
    
    # Signal generation
    signals = []
    signal_strength = 0
    
    # Bullish conditions
    if curr_ema_f > curr_ema_s and crossover_angle > 5:
        signals.append(f"Bullish EMA crossover (angle: {crossover_angle:.1f}°)")
        signal_strength += 2
    
    if curr_close > curr_ema_f and curr_close > curr_ema_s:
        signals.append("Price above both EMAs")
        signal_strength += 1
    
    if 30 < curr_rsi < 70:
        signals.append(f"RSI in normal range ({curr_rsi:.1f})")
        signal_strength += 1
    elif curr_rsi < 30:
        signals.append(f"RSI oversold ({curr_rsi:.1f})")
        signal_strength += 2
    
    if curr_macd > 0:
        signals.append("MACD positive")
        signal_strength += 1
    
    if curr_adx > 25:
        signals.append(f"Strong trend (ADX: {curr_adx:.1f})")
        signal_strength += 1
    
    # Bearish conditions
    bearish_strength = 0
    if curr_ema_f < curr_ema_s and crossover_angle > 5:
        signals.append(f"Bearish EMA crossover (angle: {crossover_angle:.1f}°)")
        bearish_strength += 2
    
    if curr_close < curr_ema_f and curr_close < curr_ema_s:
        signals.append("Price below both EMAs")
        bearish_strength += 1
    
    if curr_rsi > 70:
        signals.append(f"RSI overbought ({curr_rsi:.1f})")
        bearish_strength += 2
    
    # Determine signal
    if signal_strength - bearish_strength >= 3:
        signal = 'BULLISH'
    elif bearish_strength - signal_strength >= 3:
        signal = 'BEARISH'
    else:
        signal = 'NEUTRAL'
    
    return {
        'signal': signal,
        'strength': abs(signal_strength - bearish_strength),
        'indicators': {
            'ema_fast': curr_ema_f,
            'ema_slow': curr_ema_s,
            'rsi': curr_rsi,
            'atr': curr_atr,
            'macd': curr_macd,
            'adx': curr_adx,
            'crossover_angle': crossover_angle
        },
        'reasons': signals
    }

def gap_fill_strategy(data: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """Gap fill trading strategy"""
    gap_info = detect_gap(data)
    
    if not gap_info['has_gap']:
        return None
    
    curr_price = data['Close'].iloc[-1]
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    
    if gap_info['gap_type'] == 'up':
        # Gap up - look for gap fill short
        entry = curr_price
        target = gap_info['prev_close']
        sl = curr_price + (1.5 * atr)
        direction = 'SHORT'
        reason = f"Gap up of {gap_info['gap_pct']:.2f}% detected. Price likely to fill gap down to {target:.2f}"
    else:
        # Gap down - look for gap fill long
        entry = curr_price
        target = gap_info['prev_close']
        sl = curr_price - (1.5 * atr)
        direction = 'LONG'
        reason = f"Gap down of {abs(gap_info['gap_pct']):.2f}% detected. Price likely to fill gap up to {target:.2f}"
    
    return {
        'strategy': 'Gap Fill',
        'ticker': ticker,
        'direction': direction,
        'entry': entry,
        'target': target,
        'sl': sl,
        'risk_reward': abs((target - entry) / (sl - entry)) if sl != entry else 0,
        'reason': reason
    }

def support_resistance_strategy(data: pd.DataFrame, ticker: str, ema_fast: int = 9, ema_slow: int = 15, min_angle: float = 10, min_candle_size: float = 5) -> Optional[Dict]:
    """Support/Resistance + EMA crossover strategy"""
    if len(data) < 50:
        return None
    
    curr_price = data['Close'].iloc[-1]
    curr_high = data['High'].iloc[-1]
    curr_low = data['Low'].iloc[-1]
    
    # Get support/resistance levels
    sr_levels = identify_support_resistance(data)
    
    # Calculate EMAs
    ema_f = calculate_ema(data['Close'], ema_fast)
    ema_s = calculate_ema(data['Close'], ema_slow)
    
    # Crossover angle
    crossover_angle = calculate_ema_crossover_angle(ema_f, ema_s)
    
    # Candle size
    candle_size = curr_high - curr_low
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    
    # Check conditions
    if crossover_angle < min_angle:
        return None
    
    if candle_size < min(min_candle_size, atr):
        return None
    
    # Bullish setup
    if ema_f.iloc[-1] > ema_s.iloc[-1] and curr_price > ema_f.iloc[-1]:
        # Check if near support
        near_support = any(abs(curr_price - sup) / curr_price < 0.01 for sup in sr_levels['support'])
        
        if near_support:
            entry = curr_price
            target = entry + (2 * atr)
            sl = entry - (1 * atr)
            
            return {
                'strategy': 'Support/Resistance + EMA',
                'ticker': ticker,
                'direction': 'LONG',
                'entry': entry,
                'target': target,
                'sl': sl,
                'risk_reward': abs((target - entry) / (sl - entry)),
                'reason': f"Bullish EMA crossover ({crossover_angle:.1f}°) near support. Strong candle ({candle_size:.2f} points)"
            }
    
    # Bearish setup
    elif ema_f.iloc[-1] < ema_s.iloc[-1] and curr_price < ema_f.iloc[-1]:
        # Check if near resistance
        near_resistance = any(abs(curr_price - res) / curr_price < 0.01 for res in sr_levels['resistance'])
        
        if near_resistance:
            entry = curr_price
            target = entry - (2 * atr)
            sl = entry + (1 * atr)
            
            return {
                'strategy': 'Support/Resistance + EMA',
                'ticker': ticker,
                'direction': 'SHORT',
                'entry': entry,
                'target': target,
                'sl': sl,
                'risk_reward': abs((target - entry) / (sl - entry)),
                'reason': f"Bearish EMA crossover ({crossover_angle:.1f}°) near resistance. Strong candle ({candle_size:.2f} points)"
            }
    
    return None

def breakout_strategy(data: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """Breakout trading strategy"""
    if len(data) < 20:
        return None
    
    # Get recent high/low
    recent_high = data['High'].iloc[-20:].max()
    recent_low = data['Low'].iloc[-20:].min()
    
    curr_price = data['Close'].iloc[-1]
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].iloc[-20:].mean()
    
    # Bullish breakout
    if curr_price > recent_high and volume > avg_volume * 1.5:
        entry = curr_price
        target = entry + (2.5 * atr)
        sl = entry - (1 * atr)
        
        return {
            'strategy': 'Breakout',
            'ticker': ticker,
            'direction': 'LONG',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Breakout above recent high ({recent_high:.2f}) with strong volume ({volume/avg_volume:.1f}x avg)"
        }
    
    # Bearish breakdown
    elif curr_price < recent_low and volume > avg_volume * 1.5:
        entry = curr_price
        target = entry - (2.5 * atr)
        sl = entry + (1 * atr)
        
        return {
            'strategy': 'Breakout',
            'ticker': ticker,
            'direction': 'SHORT',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Breakdown below recent low ({recent_low:.2f}) with strong volume ({volume/avg_volume:.1f}x avg)"
        }
    
    return None

def mean_reversion_strategy(data: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """Mean reversion strategy using Bollinger Bands"""
    if len(data) < 30:
        return None
    
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data['Close'])
    curr_price = data['Close'].iloc[-1]
    rsi = calculate_rsi(data['Close']).iloc[-1]
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    
    # Oversold - potential long
    if curr_price < lower_bb.iloc[-1] and rsi < 30:
        entry = curr_price
        target = middle_bb.iloc[-1]
        sl = entry - (1.5 * atr)
        
        return {
            'strategy': 'Mean Reversion',
            'ticker': ticker,
            'direction': 'LONG',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Price below lower BB and RSI oversold ({rsi:.1f}). Mean reversion expected to {target:.2f}"
        }
    
    # Overbought - potential short
    elif curr_price > upper_bb.iloc[-1] and rsi > 70:
        entry = curr_price
        target = middle_bb.iloc[-1]
        sl = entry + (1.5 * atr)
        
        return {
            'strategy': 'Mean Reversion',
            'ticker': ticker,
            'direction': 'SHORT',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Price above upper BB and RSI overbought ({rsi:.1f}). Mean reversion expected to {target:.2f}"
        }
    
    return None

def momentum_strategy(data: pd.DataFrame, ticker: str) -> Optional[Dict]:
    """Momentum strategy using MACD and RSI"""
    if len(data) < 30:
        return None
    
    macd_line, signal_line, histogram = calculate_macd(data['Close'])
    rsi = calculate_rsi(data['Close'])
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    
    curr_price = data['Close'].iloc[-1]
    curr_macd = histogram.iloc[-1]
    prev_macd = histogram.iloc[-2]
    curr_rsi = rsi.iloc[-1]
    
    # Bullish momentum
    if curr_macd > 0 and curr_macd > prev_macd and 40 < curr_rsi < 70:
        entry = curr_price
        target = entry + (3 * atr)
        sl = entry - (1 * atr)
        
        return {
            'strategy': 'Momentum',
            'ticker': ticker,
            'direction': 'LONG',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Strong bullish momentum - MACD positive and rising, RSI: {curr_rsi:.1f}"
        }
    
    # Bearish momentum
    elif curr_macd < 0 and curr_macd < prev_macd and 30 < curr_rsi < 60:
        entry = curr_price
        target = entry - (3 * atr)
        sl = entry + (1 * atr)
        
        return {
            'strategy': 'Momentum',
            'ticker': ticker,
            'direction': 'SHORT',
            'entry': entry,
            'target': target,
            'sl': sl,
            'risk_reward': abs((target - entry) / (sl - entry)),
            'reason': f"Strong bearish momentum - MACD negative and falling, RSI: {curr_rsi:.1f}"
        }
    
    return None

# ==================== SCREENER FUNCTIONS ====================
def fetch_index_constituents(index_name: str, tickers_dict: Dict, interval: str, period: str) -> List[Dict]:
    """Fetch data for all constituents of an index"""
    results = []
    total = len(tickers_dict)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (ticker, weight) in enumerate(tickers_dict.items()):
        status_text.text(f"Fetching {ticker} ({idx+1}/{total})...")
        progress_bar.progress((idx + 1) / total)
        
        data = safe_yfinance_call(ticker, interval=interval, period=period)
        
        if data is not None and len(data) > 0:
            try:
                # Basic info
                curr_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else curr_price
                change = curr_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                
                # 52-week high/low (approximate from available data)
                high_52w = data['High'].max()
                low_52w = data['Low'].min()
                dist_from_high = ((curr_price - high_52w) / high_52w) * 100
                dist_from_low = ((curr_price - low_52w) / low_52w) * 100
                
                # Volume analysis
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].mean()
                volume_ratio = volume / avg_volume if avg_volume != 0 else 0
                
                # Technical indicators
                rsi = calculate_rsi(data['Close']).iloc[-1]
                atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
                adx = calculate_adx(data['High'], data['Low'], data['Close']).iloc[-1]
                macd_line, signal_line, histogram = calculate_macd(data['Close'])
                
                # Price action analysis
                price_action = analyze_price_action(data)
                
                results.append({
                    'ticker': ticker,
                    'weight': weight,
                    'data': data,
                    'price': curr_price,
                    'change': change,
                    'change_pct': change_pct,
                    'high_52w': high_52w,
                    'low_52w': low_52w,
                    'dist_from_high': dist_from_high,
                    'dist_from_low': dist_from_low,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio,
                    'rsi': rsi,
                    'atr': atr,
                    'adx': adx,
                    'macd': histogram.iloc[-1],
                    'price_action': price_action
                })
            except Exception as e:
                st.warning(f"Error processing {ticker}: {str(e)}")
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def calculate_index_impact(results: List[Dict]) -> Dict:
    """Calculate index impact based on constituent performance"""
    total_impact = 0
    positive_contributors = []
    negative_contributors = []
    
    for result in results:
        impact = result['change_pct'] * (result['weight'] / 100)
        total_impact += impact
        
        if result['change_pct'] > 0:
            positive_contributors.append({
                'ticker': result['ticker'],
                'change_pct': result['change_pct'],
                'impact': impact
            })
        else:
            negative_contributors.append({
                'ticker': result['ticker'],
                'change_pct': result['change_pct'],
                'impact': impact
            })
    
    # Sort by impact
    positive_contributors.sort(key=lambda x: x['impact'], reverse=True)
    negative_contributors.sort(key=lambda x: x['impact'])
    
    return {
        'total_impact': total_impact,
        'positive_contributors': positive_contributors[:5],
        'negative_contributors': negative_contributors[:5]
    }

def generate_trade_recommendations(results: List[Dict], strategies: List[str], ema_fast: int, ema_slow: int, min_angle: float, min_candle_size: float) -> List[Dict]:
    """Generate trade recommendations based on selected strategies"""
    recommendations = []
    
    for result in results:
        ticker = result['ticker']
        data = result['data']
        
        # Apply each selected strategy
        for strategy_name in strategies:
            trade = None
            
            if strategy_name == 'Gap Fill':
                trade = gap_fill_strategy(data, ticker)
            elif strategy_name == 'Support/Resistance + EMA':
                trade = support_resistance_strategy(data, ticker, ema_fast, ema_slow, min_angle, min_candle_size)
            elif strategy_name == 'Breakout':
                trade = breakout_strategy(data, ticker)
            elif strategy_name == 'Mean Reversion':
                trade = mean_reversion_strategy(data, ticker)
            elif strategy_name == 'Momentum':
                trade = momentum_strategy(data, ticker)
            
            if trade:
                # Add current price and indicators
                trade['current_price'] = result['price']
                trade['rsi'] = result['rsi']
                trade['adx'] = result['adx']
                trade['volume_ratio'] = result['volume_ratio']
                recommendations.append(trade)
    
    # Sort by risk-reward ratio
    recommendations.sort(key=lambda x: x['risk_reward'], reverse=True)
    
    return recommendations

# ==================== STREAMLIT UI ====================
def main():
    st.markdown('<div class="main-header">📈 Advanced Quantitative Trading Screener</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Index/Asset selection
        st.subheader("Select Index/Asset")
        index_type = st.selectbox(
            "Index Type",
            ['NIFTY 50', 'BANK NIFTY', 'Custom Ticker']
        )
        
        if index_type == 'Custom Ticker':
            custom_ticker = st.text_input("Enter Ticker Symbol", "AAPL")
        
        # Timeframe selection
        st.subheader("Timeframe")
        interval = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()), index=1)
        period = st.selectbox("Period", TIMEFRAME_PERIODS[interval])
        
        # Strategy selection
        st.subheader("Trading Strategies")
        strategies = st.multiselect(
            "Select Strategies",
            ['Gap Fill', 'Support/Resistance + EMA', 'Breakout', 'Mean Reversion', 'Momentum'],
            default=['Gap Fill', 'Support/Resistance + EMA', 'Breakout']
        )
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        ema_fast = st.slider("Fast EMA", 5, 20, 9)
        ema_slow = st.slider("Slow EMA", 10, 50, 15)
        min_angle = st.slider("Min Crossover Angle (degrees)", 0, 45, 10)
        min_candle_size = st.slider("Min Candle Size (points)", 1, 20, 5)
        
        # Filters
        st.subheader("Filters")
        min_volume_ratio = st.slider("Min Volume Ratio", 1.0, 5.0, 1.5)
        min_rr = st.slider("Min Risk:Reward", 1.0, 5.0, 1.5)
        
        # Run screener button
        run_screener = st.button("🚀 Run Screener", type="primary", use_container_width=True)
    
    # Main content
    if run_screener:
        with st.spinner("Fetching data and analyzing..."):
            # Determine which tickers to fetch
            if index_type == 'NIFTY 50':
                tickers_dict = NIFTY50_TICKERS
            elif index_type == 'BANK NIFTY':
                tickers_dict = BANKNIFTY_TICKERS
            else:
                tickers_dict = {custom_ticker: 100.0}
            
            # Fetch data
            results = fetch_index_constituents(index_type, tickers_dict, interval, period)
            
            if not results:
                st.error("❌ No data fetched. Please check your inputs and try again.")
                return
            
            st.success(f"✅ Successfully fetched data for {len(results)} stocks")
            
            # Calculate index impact
            if index_type in ['NIFTY 50', 'BANK NIFTY']:
                impact = calculate_index_impact(results)
                
                st.markdown('<div class="sub-header">📊 Index Impact Analysis</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    impact_color = "green" if impact['total_impact'] > 0 else "red"
                    st.metric(
                        "Estimated Index Movement",
                        f"{impact['total_impact']:.2f}%",
                        delta=f"{impact['total_impact']:.2f}%"
                    )
                
                with col2:
                    st.markdown("**Top Positive Contributors:**")
                    for contrib in impact['positive_contributors'][:3]:
                        st.write(f"• {contrib['ticker']}: +{contrib['change_pct']:.2f}%")
                
                with col3:
                    st.markdown("**Top Negative Contributors:**")
                    for contrib in impact['negative_contributors'][:3]:
                        st.write(f"• {contrib['ticker']}: {contrib['change_pct']:.2f}%")
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🎯 Trade Recommendations",
                "📈 All Stocks Overview",
                "🔥 Top Gainers/Losers",
                "📊 Active Stocks",
                "🎚️ 52-Week High/Low",
                "📉 Technical Signals"
            ])
            
            # Tab 1: Trade Recommendations
            with tab1:
                st.markdown('<div class="sub-header">🎯 Intraday Trade Recommendations</div>', unsafe_allow_html=True)
                
                recommendations = generate_trade_recommendations(
                    results, strategies, ema_fast, ema_slow, min_angle, min_candle_size
                )
                
                # Filter by criteria
                filtered_recs = [
                    rec for rec in recommendations
                    if rec['risk_reward'] >= min_rr and rec.get('volume_ratio', 0) >= min_volume_ratio
                ]
                
                if filtered_recs:
                    st.write(f"**Found {len(filtered_recs)} trading opportunities**")
                    
                    for idx, rec in enumerate(filtered_recs[:10], 1):
                        with st.expander(f"#{idx} {rec['ticker']} - {rec['strategy']} ({rec['direction']}) | R:R = {rec['risk_reward']:.2f}"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Entry", f"₹{rec['entry']:.2f}")
                                st.metric("Current Price", f"₹{rec['current_price']:.2f}")
                            
                            with col2:
                                st.metric("Target", f"₹{rec['target']:.2f}")
                                profit = abs(rec['target'] - rec['entry'])
                                st.metric("Potential Profit", f"₹{profit:.2f}")
                            
                            with col3:
                                st.metric("Stop Loss", f"₹{rec['sl']:.2f}")
                                risk = abs(rec['sl'] - rec['entry'])
                                st.metric("Risk", f"₹{risk:.2f}")
                            
                            with col4:
                                st.metric("Risk:Reward", f"{rec['risk_reward']:.2f}")
                                st.metric("RSI", f"{rec['rsi']:.1f}")
                            
                            st.info(f"**Reason:** {rec['reason']}")
                            
                            # Additional indicators
                            st.write(f"**ADX:** {rec['adx']:.1f} | **Volume Ratio:** {rec['volume_ratio']:.2f}x")
                else:
                    st.warning("No trading opportunities found matching your criteria. Try adjusting filters.")
            
            # Tab 2: All Stocks Overview
            with tab2:
                st.markdown('<div class="sub-header">📈 All Stocks Overview</div>', unsafe_allow_html=True)
                
                overview_data = []
                for result in results:
                    color_class = "bullish" if result['change_pct'] > 0 else "bearish"
                    overview_data.append({
                        'Ticker': result['ticker'],
                        'Price': f"₹{result['price']:.2f}",
                        'Change': f"{result['change']:.2f}",
                        'Change %': f"{result['change_pct']:.2f}%",
                        'Volume Ratio': f"{result['volume_ratio']:.2f}x",
                        'RSI': f"{result['rsi']:.1f}",
                        'Signal': result['price_action']['signal']
                    })
                
                df_overview = pd.DataFrame(overview_data)
                
                # Style the dataframe
                def color_change(val):
                    if isinstance(val, str) and '%' in val:
                        num = float(val.replace('%', ''))
                        color = 'green' if num > 0 else 'red' if num < 0 else 'gray'
                        return f'color: {color}; font-weight: bold'
                    return ''
                
                styled_df = df_overview.style.applymap(color_change, subset=['Change %'])
                st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Tab 3: Top Gainers/Losers
            with tab3:
                st.markdown('<div class="sub-header">🔥 Top Gainers & Losers</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚀 Top Gainers")
                    gainers = sorted(results, key=lambda x: x['change_pct'], reverse=True)[:10]
                    
                    for idx, gainer in enumerate(gainers, 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>#{idx} {gainer['ticker']}</b><br>
                            Price: ₹{gainer['price']:.2f}<br>
                            <span class="bullish">+{gainer['change_pct']:.2f}%</span> (₹{gainer['change']:.2f})
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("📉 Top Losers")
                    losers = sorted(results, key=lambda x: x['change_pct'])[:10]
                    
                    for idx, loser in enumerate(losers, 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>#{idx} {loser['ticker']}</b><br>
                            Price: ₹{loser['price']:.2f}<br>
                            <span class="bearish">{loser['change_pct']:.2f}%</span> (₹{loser['change']:.2f})
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tab 4: Active Stocks
            with tab4:
                st.markdown('<div class="sub-header">📊 Active Stocks Analysis</div>', unsafe_allow_html=True)
                
                # Filter stocks with >1% change and high volume
                active_stocks = [
                    r for r in results
                    if abs(r['change_pct']) > 1.0 and r['volume_ratio'] > 1.2
                ]
                
                st.write(f"**{len(active_stocks)} stocks with >1% price change and high volume**")
                
                for stock in active_stocks[:15]:
                    with st.expander(f"{stock['ticker']} - {stock['change_pct']:+.2f}%"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Price", f"₹{stock['price']:.2f}", f"{stock['change_pct']:.2f}%")
                            st.write(f"**Volume Ratio:** {stock['volume_ratio']:.2f}x")
                        
                        with col2:
                            st.write(f"**RSI:** {stock['rsi']:.1f}")
                            st.write(f"**ADX:** {stock['adx']:.1f}")
                            st.write(f"**MACD:** {stock['macd']:.4f}")
                        
                        with col3:
                            signal = stock['price_action']['signal']
                            signal_class = "bullish" if signal == "BULLISH" else "bearish" if signal == "BEARISH" else "neutral"
                            st.markdown(f"**Signal:** <span class='{signal_class}'>{signal}</span>", unsafe_allow_html=True)
                            st.write(f"**Strength:** {stock['price_action']['strength']}/5")
                        
                        # Technical details
                        st.write("**Key Indicators:**")
                        for reason in stock['price_action']['reasons'][:3]:
                            st.write(f"• {reason}")
            
            # Tab 5: 52-Week High/Low
            with tab5:
                st.markdown('<div class="sub-header">🎚️ 52-Week High/Low Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Near 52-Week High")
                    near_high = sorted(
                        [r for r in results if r['dist_from_high'] > -5],
                        key=lambda x: x['dist_from_high'],
                        reverse=True
                    )[:10]
                    
                    for stock in near_high:
                        color = "green" if stock['dist_from_high'] >= 0 else "orange"
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>{stock['ticker']}</b><br>
                            Price: ₹{stock['price']:.2f} | 52W High: ₹{stock['high_52w']:.2f}<br>
                            <span style="color: {color}; font-weight: bold">{stock['dist_from_high']:.2f}% from high</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Near 52-Week Low")
                    near_low = sorted(
                        [r for r in results if r['dist_from_low'] < 5],
                        key=lambda x: x['dist_from_low']
                    )[:10]
                    
                    for stock in near_low:
                        color = "red" if stock['dist_from_low'] <= 0 else "orange"
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>{stock['ticker']}</b><br>
                            Price: ₹{stock['price']:.2f} | 52W Low: ₹{stock['low_52w']:.2f}<br>
                            <span style="color: {color}; font-weight: bold">{stock['dist_from_low']:.2f}% from low</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tab 6: Technical Signals
            with tab6:
                st.markdown('<div class="sub-header">📉 Technical Signals Summary</div>', unsafe_allow_html=True)
                
                # Categorize by signals
                bullish_stocks = [r for r in results if r['price_action']['signal'] == 'BULLISH']
                bearish_stocks = [r for r in results if r['price_action']['signal'] == 'BEARISH']
                neutral_stocks = [r for r in results if r['price_action']['signal'] == 'NEUTRAL']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bullish Signals", len(bullish_stocks))
                with col2:
                    st.metric("Bearish Signals", len(bearish_stocks))
                with col3:
                    st.metric("Neutral Signals", len(neutral_stocks))
                
                # Show detailed technical analysis
                st.subheader("Detailed Technical Analysis")
                
                selected_stock = st.selectbox(
                    "Select Stock for Detailed View",
                    [r['ticker'] for r in results]
                )
                
                stock_data = next((r for r in results if r['ticker'] == selected_stock), None)
                
                if stock_data:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Price", f"₹{stock_data['price']:.2f}", f"{stock_data['change_pct']:.2f}%")
                    with col2:
                        st.metric("RSI", f"{stock_data['rsi']:.1f}")
                    with col3:
                        st.metric("ADX", f"{stock_data['adx']:.1f}")
                    with col4:
                        signal = stock_data['price_action']['signal']
                        st.metric("Signal", signal)
                    
                    st.write("**Analysis Reasons:**")
                    for reason in stock_data['price_action']['reasons']:
                        st.write(f"• {reason}")
                    
                    # Plot candlestick chart
                    st.subheader("Price Chart")
                    data = stock_data['data'].tail(100)
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3]
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
                    ema_f = calculate_ema(data['Close'], ema_fast)
                    ema_s = calculate_ema(data['Close'], ema_slow)
                    
                    fig.add_trace(
                        go.Scatter(x=data.index, y=ema_f, name=f'EMA {ema_fast}', line=dict(color='blue', width=1)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=ema_s, name=f'EMA {ema_slow}', line=dict(color='red', width=1)),
                        row=1, col=1
                    )
                    
                    # Volume
                    colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red' for i in range(len(data))]
                    fig.add_trace(
                        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f'{selected_stock} Price Chart',
                        xaxis_rangeslider_visible=False,
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Configure your parameters in the sidebar and click 'Run Screener' to start analysis")
        
        # Show example information
        st.markdown("""
        ### 📚 Features
        
        **Supported Indices:**
        - NIFTY 50 (50 constituents)
        - BANK NIFTY (11 constituents)
        - Custom tickers via yfinance
        
        **Trading Strategies:**
        1. **Gap Fill Strategy** - Identifies gap ups/downs with fill potential
        2. **Support/Resistance + EMA** - Price action near key levels with EMA crossovers
        3. **Breakout Strategy** - Volume-backed breakouts above/below recent ranges
        4. **Mean Reversion** - Bollinger Band extremes with RSI confirmation
        5. **Momentum Strategy** - MACD and RSI alignment for trend continuation
        
        **Analysis Tabs:**
        - 🎯 **Trade Recommendations** - Entry, SL, Target with R:R ratios
        - 📈 **All Stocks** - Complete overview with price changes
        - 🔥 **Top Gainers/Losers** - Best and worst performers
        - 📊 **Active Stocks** - High volume with >1% moves
        - 🎚️ **52-Week Levels** - Stocks near highs/lows
        - 📉 **Technical Signals** - Detailed indicator analysis with charts
        
        **Index Impact Analysis:**
        - Calculates weighted contribution to index movement
        - Identifies top positive and negative contributors
        - Shows estimated index movement based on constituents
        """)

if __name__ == "__main__":
    main()
