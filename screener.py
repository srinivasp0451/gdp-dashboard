import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
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
    .impact-explanation {background-color: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0;}
    .gap-card {background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0;}
    .fundamental-strong {background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 3px solid #28a745;}
    .fundamental-weak {background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 3px solid #dc3545;}
    .fundamental-neutral {background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 3px solid #ffc107;}
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

def get_fundamental_data(ticker: str) -> Dict:
    """Fetch fundamental data for a stock"""
    try:
        delay = random.uniform(1.0, 1.5)
        time.sleep(delay)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract fundamental metrics
        fundamentals = {
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'roe': info.get('returnOnEquity', None),
            'eps': info.get('trailingEps', None),
            'div_yield': info.get('dividendYield', None),
            'book_value': info.get('bookValue', None),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', None),
            'profit_margins': info.get('profitMargins', None),
            'current_ratio': info.get('currentRatio', None),
            'quick_ratio': info.get('quickRatio', None)
        }
        
        return fundamentals
    
    except Exception as e:
        return {}

def analyze_fundamentals(fundamentals: Dict, ticker: str) -> Dict:
    """Analyze fundamental metrics and provide rating"""
    
    score = 0
    max_score = 0
    reasons = []
    
    # PE Ratio Analysis
    if fundamentals.get('pe_ratio'):
        max_score += 1
        pe = fundamentals['pe_ratio']
        if 10 < pe < 25:
            score += 1
            reasons.append(f"✓ Healthy P/E ratio ({pe:.2f}) - fairly valued")
        elif pe < 10:
            score += 0.5
            reasons.append(f"⚠ Low P/E ratio ({pe:.2f}) - potentially undervalued or concerns")
        else:
            reasons.append(f"✗ High P/E ratio ({pe:.2f}) - potentially overvalued")
    
    # PB Ratio Analysis
    if fundamentals.get('pb_ratio'):
        max_score += 1
        pb = fundamentals['pb_ratio']
        if 1 < pb < 3:
            score += 1
            reasons.append(f"✓ Good P/B ratio ({pb:.2f}) - reasonable book value")
        elif pb < 1:
            score += 0.5
            reasons.append(f"⚠ P/B below 1 ({pb:.2f}) - trading below book value")
        else:
            reasons.append(f"✗ High P/B ratio ({pb:.2f}) - premium valuation")
    
    # Debt to Equity
    if fundamentals.get('debt_to_equity'):
        max_score += 1
        de = fundamentals['debt_to_equity']
        if de < 1:
            score += 1
            reasons.append(f"✓ Low debt-to-equity ({de:.2f}) - strong balance sheet")
        elif de < 2:
            score += 0.5
            reasons.append(f"⚠ Moderate debt-to-equity ({de:.2f}) - manageable debt")
        else:
            reasons.append(f"✗ High debt-to-equity ({de:.2f}) - high leverage risk")
    
    # ROE Analysis
    if fundamentals.get('roe'):
        max_score += 1
        roe = fundamentals['roe'] * 100
        if roe > 15:
            score += 1
            reasons.append(f"✓ Strong ROE ({roe:.2f}%) - efficient capital use")
        elif roe > 10:
            score += 0.5
            reasons.append(f"⚠ Moderate ROE ({roe:.2f}%) - acceptable returns")
        else:
            reasons.append(f"✗ Low ROE ({roe:.2f}%) - poor capital efficiency")
    
    # Profit Margins
    if fundamentals.get('profit_margins'):
        max_score += 1
        margin = fundamentals['profit_margins'] * 100
        if margin > 10:
            score += 1
            reasons.append(f"✓ Healthy profit margins ({margin:.2f}%) - good profitability")
        elif margin > 5:
            score += 0.5
            reasons.append(f"⚠ Moderate profit margins ({margin:.2f}%)")
        else:
            reasons.append(f"✗ Low profit margins ({margin:.2f}%) - profitability concerns")
    
    # Dividend Yield
    if fundamentals.get('div_yield'):
        div_yield = fundamentals['div_yield'] * 100
        if div_yield > 2:
            reasons.append(f"✓ Attractive dividend yield ({div_yield:.2f}%)")
        elif div_yield > 0:
            reasons.append(f"⚠ Moderate dividend yield ({div_yield:.2f}%)")
    
    # Calculate rating
    if max_score > 0:
        rating_pct = (score / max_score) * 100
        
        if rating_pct >= 75:
            rating = "Strong"
            rating_class = "fundamental-strong"
        elif rating_pct >= 50:
            rating = "Moderate"
            rating_class = "fundamental-neutral"
        else:
            rating = "Weak"
            rating_class = "fundamental-weak"
    else:
        rating = "Insufficient Data"
        rating_class = "fundamental-neutral"
        rating_pct = 0
    
    return {
        'rating': rating,
        'rating_class': rating_class,
        'rating_pct': rating_pct,
        'score': score,
        'max_score': max_score,
        'reasons': reasons
    }

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
    curr_close = data['Close'].iloc[-1]
    gap_size = curr_open - prev_close
    gap_pct = (gap_size / prev_close) * 100
    
    if abs(gap_pct) > 0.5:  # Gap threshold 0.5%
        gap_type = 'up' if gap_size > 0 else 'down'
        
        # Calculate gap fill probability
        if gap_type == 'up':
            fill_progress = ((prev_close - curr_close) / (prev_close - curr_open)) * 100 if curr_open != prev_close else 0
            is_filling = curr_close < curr_open
        else:
            fill_progress = ((curr_close - prev_close) / (curr_open - prev_close)) * 100 if curr_open != prev_close else 0
            is_filling = curr_close > curr_open
        
        fill_progress = max(0, min(100, fill_progress))
        
        return {
            'has_gap': True,
            'gap_type': gap_type,
            'gap_size': gap_size,
            'gap_pct': gap_pct,
            'prev_close': prev_close,
            'curr_open': curr_open,
            'curr_close': curr_close,
            'fill_progress': fill_progress,
            'is_filling': is_filling
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
    
    # Calculate probability based on gap size and current price action
    gap_fill_prob = 70  # Base probability
    
    if gap_info['is_filling']:
        gap_fill_prob += 15
    
    if abs(gap_info['gap_pct']) < 2:
        gap_fill_prob += 10
    elif abs(gap_info['gap_pct']) > 5:
        gap_fill_prob -= 20
    
    if gap_info['gap_type'] == 'up':
        entry = curr_price
        target = gap_info['prev_close']
        sl = curr_price + (1.5 * atr)
        direction = 'SHORT'
        reason = f"Gap up of {gap_info['gap_pct']:.2f}% detected. Price likely to fill gap down to {target:.2f}. Fill progress: {gap_info['fill_progress']:.1f}%"
    else:
        entry = curr_price
        target = gap_info['prev_close']
        sl = curr_price - (1.5 * atr)
        direction = 'LONG'
        reason = f"Gap down of {abs(gap_info['gap_pct']):.2f}% detected. Price likely to fill gap up to {target:.2f}. Fill progress: {gap_info['fill_progress']:.1f}%"
    
    return {
        'strategy': 'Gap Fill',
        'ticker': ticker,
        'direction': direction,
        'entry': entry,
        'target': target,
        'sl': sl,
        'risk_reward': abs((target - entry) / (sl - entry)) if sl != entry else 0,
        'reason': reason,
        'gap_info': gap_info,
        'probability': min(95, max(40, gap_fill_prob))
    }

def support_resistance_strategy(data: pd.DataFrame, ticker: str, ema_fast: int = 9, ema_slow: int = 15, min_angle: float = 10, min_candle_size: float = 5) -> Optional[Dict]:
    """Support/Resistance + EMA crossover strategy"""
    if len(data) < 50:
        return None
    
    curr_price = data['Close'].iloc[-1]
    curr_high = data['High'].iloc[-1]
    curr_low = data['Low'].iloc[-1]
    
    sr_levels = identify_support_resistance(data)
    
    ema_f = calculate_ema(data['Close'], ema_fast)
    ema_s = calculate_ema(data['Close'], ema_slow)
    
    crossover_angle = calculate_ema_crossover_angle(ema_f, ema_s)
    
    candle_size = curr_high - curr_low
    atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
    
    if crossover_angle < min_angle:
        return None
    
    if candle_size < min(min_candle_size, atr):
        return None
    
    # Bullish setup
    if ema_f.iloc[-1] > ema_s.iloc[-1] and curr_price > ema_f.iloc[-1]:
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
def fetch_index_constituents(index_name: str, tickers_dict: Dict, interval: str, period: str, fetch_fundamentals: bool = False) -> List[Dict]:
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
                
                # 52-week high/low
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
                
                # Gap detection
                gap_info = detect_gap(data)
                
                result = {
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
                    'price_action': price_action,
                    'gap_info': gap_info
                }
                
                # Fetch fundamentals if requested
                if fetch_fundamentals:
                    fundamentals = get_fundamental_data(ticker)
                    if fundamentals:
                        result['fundamentals'] = fundamentals
                        result['fundamental_analysis'] = analyze_fundamentals(fundamentals, ticker)
                
                results.append(result)
            except Exception as e:
                st.warning(f"Error processing {ticker}: {str(e)}")
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def calculate_index_impact(results: List[Dict]) -> Dict:
    """Calculate index impact with detailed explanation"""
    total_impact = 0
    positive_contributors = []
    negative_contributors = []
    contribution_details = []
    
    for result in results:
        # Calculate weighted contribution
        price_change_pct = result['change_pct']
        weight = result['weight']
        
        # Impact = (Price Change %) × (Weight % / 100)
        impact = price_change_pct * (weight / 100)
        total_impact += impact
        
        contribution_details.append({
            'ticker': result['ticker'],
            'price_change_pct': price_change_pct,
            'weight': weight,
            'impact': impact,
            'calculation': f"({price_change_pct:.2f}% × {weight:.2f}% / 100) = {impact:.4f}%"
        })
        
        if result['change_pct'] > 0:
            positive_contributors.append({
                'ticker': result['ticker'],
                'change_pct': result['change_pct'],
                'weight': weight,
                'impact': impact
            })
        else:
            negative_contributors.append({
                'ticker': result['ticker'],
                'change_pct': result['change_pct'],
                'weight': weight,
                'impact': impact
            })
    
    # Sort by impact
    positive_contributors.sort(key=lambda x: x['impact'], reverse=True)
    negative_contributors.sort(key=lambda x: x['impact'])
    
    # Build explanation
    explanation = f"""
    **Index Impact Calculation Method:**
    
    The index movement is calculated as a weighted sum of individual stock movements:
    
    **Formula:** Index Change (%) = Σ [(Stock Change % × Stock Weight %) / 100]
    
    **Example Calculation:**
    - If Stock A changes by +2% and has 10% weight: Impact = (2 × 10) / 100 = +0.20%
    - If Stock B changes by -1% and has 5% weight: Impact = (-1 × 5) / 100 = -0.05%
    - Total Index Change = +0.20% - 0.05% = +0.15%
    
    **Current Analysis:**
    - Total positive impact: {sum(c['impact'] for c in positive_contributors):.4f}%
    - Total negative impact: {sum(c['impact'] for c in negative_contributors):.4f}%
    - **Net index movement: {total_impact:.4f}%**
    """
    
    return {
        'total_impact': total_impact,
        'positive_contributors': positive_contributors[:5],
        'negative_contributors': negative_contributors[:5],
        'contribution_details': contribution_details,
        'explanation': explanation
    }

def generate_trade_recommendations(results: List[Dict], strategies: List[str], ema_fast: int, ema_slow: int, min_angle: float, min_candle_size: float) -> List[Dict]:
    """Generate trade recommendations based on selected strategies"""
    recommendations = []
    
    for result in results:
        ticker = result['ticker']
        data = result['data']
        
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
                trade['current_price'] = result['price']
                trade['rsi'] = result['rsi']
                trade['adx'] = result['adx']
                trade['volume_ratio'] = result['volume_ratio']
                recommendations.append(trade)
    
    recommendations.sort(key=lambda x: x['risk_reward'], reverse=True)
    
    return recommendations

def get_gap_fill_stocks(results: List[Dict]) -> List[Dict]:
    """Filter stocks with gaps that are likely to fill"""
    gap_stocks = []
    
    for result in results:
        gap_info = result['gap_info']
        
        if gap_info['has_gap']:
            gap_stocks.append({
                'ticker': result['ticker'],
                'price': result['price'],
                'gap_type': gap_info['gap_type'],
                'gap_size': gap_info['gap_size'],
                'gap_pct': gap_info['gap_pct'],
                'prev_close': gap_info['prev_close'],
                'curr_open': gap_info['curr_open'],
                'curr_close': gap_info['curr_close'],
                'fill_progress': gap_info['fill_progress'],
                'is_filling': gap_info['is_filling'],
                'distance_to_fill': abs(gap_info['curr_close'] - gap_info['prev_close']),
                'rsi': result['rsi'],
                'volume_ratio': result['volume_ratio']
            })
    
    # Sort by fill progress (descending)
    gap_stocks.sort(key=lambda x: x['fill_progress'], reverse=True)
    
    return gap_stocks

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
        interval = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()), index=4)
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
        
        # Fetch fundamentals
        fetch_fundamentals = st.checkbox("Fetch Fundamental Data", value=True)
        
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
            results = fetch_index_constituents(index_type, tickers_dict, interval, period, fetch_fundamentals)
            
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
                    st.metric(
                        "Estimated Index Movement",
                        f"{impact['total_impact']:.4f}%",
                        delta=f"{impact['total_impact']:.4f}%"
                    )
                
                with col2:
                    st.markdown("**Top Positive Contributors:**")
                    for contrib in impact['positive_contributors'][:3]:
                        st.write(f"• **{contrib['ticker']}**: +{contrib['change_pct']:.2f}% (weight: {contrib['weight']:.2f}%) → +{contrib['impact']:.4f}%")
                
                with col3:
                    st.markdown("**Top Negative Contributors:**")
                    for contrib in impact['negative_contributors'][:3]:
                        st.write(f"• **{contrib['ticker']}**: {contrib['change_pct']:.2f}% (weight: {contrib['weight']:.2f}%) → {contrib['impact']:.4f}%")
                
                # Show detailed explanation
                with st.expander("📖 How is the index movement calculated? (Click to expand)"):
                    st.markdown(f"""
                    <div class="impact-explanation">
                    {impact['explanation']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Detailed Contribution Breakdown:**")
                    contrib_df = pd.DataFrame(impact['contribution_details'])
                    contrib_df = contrib_df.sort_values('impact', ascending=False)
                    st.dataframe(contrib_df[['ticker', 'price_change_pct', 'weight', 'impact', 'calculation']], use_container_width=True)
            
            # Get gap fill stocks
            gap_fill_stocks = get_gap_fill_stocks(results)
            
            # Calculate sentiment
            bullish_count = sum(1 for r in results if r['price_action']['signal'] == 'BULLISH')
            bearish_count = sum(1 for r in results if r['price_action']['signal'] == 'BEARISH')
            neutral_count = sum(1 for r in results if r['price_action']['signal'] == 'NEUTRAL')
            total_count = len(results)
            
            # Create tabs
            tabs = st.tabs([
                "🎯 Trade Recommendations",
                "🔄 Gap Fill Opportunities",
                "💰 Fundamental Analysis",
                "🗺️ Market Heatmap",
                "📈 All Stocks Overview",
                "🔥 Top Gainers/Losers",
                "📊 Active Stocks",
                "🎚️ 52-Week High/Low",
                "📉 Technical Signals"
            ])
            
            # Tab 1: Trade Recommendations
            with tabs[0]:
                st.markdown('<div class="sub-header">🎯 Intraday Trade Recommendations</div>', unsafe_allow_html=True)
                
                recommendations = generate_trade_recommendations(
                    results, strategies, ema_fast, ema_slow, min_angle, min_candle_size
                )
                
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
                                if 'probability' in rec:
                                    st.metric("Success Probability", f"{rec['probability']:.0f}%")
                            
                            st.info(f"**Reason:** {rec['reason']}")
                            st.write(f"**ADX:** {rec['adx']:.1f} | **Volume Ratio:** {rec['volume_ratio']:.2f}x")
                else:
                    st.warning("No trading opportunities found matching your criteria. Try adjusting filters.")
            
            # Tab 2: Gap Fill Opportunities
            with tabs[1]:
                st.markdown('<div class="sub-header">🔄 Gap Fill Opportunities</div>', unsafe_allow_html=True)
                
                if gap_fill_stocks:
                    st.write(f"**Found {len(gap_fill_stocks)} stocks with gaps**")
                    
                    for idx, gap_stock in enumerate(gap_fill_stocks, 1):
                        gap_type_color = "#28a745" if gap_stock['gap_type'] == 'down' else "#dc3545"
                        fill_status = "✅ Filling" if gap_stock['is_filling'] else "⏳ Not filling yet"
                        
                        with st.expander(f"#{idx} {gap_stock['ticker']} - Gap {gap_stock['gap_type'].upper()} ({gap_stock['gap_pct']:.2f}%) - {fill_status}"):
                            st.markdown(f"""
                            <div class="gap-card">
                                <h4 style="color: {gap_type_color};">Gap {gap_stock['gap_type'].upper()} detected</h4>
                                <p><strong>Gap Size:</strong> ₹{abs(gap_stock['gap_size']):.2f} ({abs(gap_stock['gap_pct']):.2f}%)</p>
                                <p><strong>Previous Close:</strong> ₹{gap_stock['prev_close']:.2f}</p>
                                <p><strong>Current Open:</strong> ₹{gap_stock['curr_open']:.2f}</p>
                                <p><strong>Current Price:</strong> ₹{gap_stock['curr_close']:.2f}</p>
                                <p><strong>Distance to Fill:</strong> ₹{gap_stock['distance_to_fill']:.2f}</p>
                                <p><strong>Fill Progress:</strong> {gap_stock['fill_progress']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if gap_stock['gap_type'] == 'up':
                                    st.write(f"**📉 Expected Direction:** Downward (to fill gap)")
                                    st.write(f"**🎯 Target:** ₹{gap_stock['prev_close']:.2f}")
                                else:
                                    st.write(f"**📈 Expected Direction:** Upward (to fill gap)")
                                    st.write(f"**🎯 Target:** ₹{gap_stock['prev_close']:.2f}")
                            
                            with col2:
                                st.metric("RSI", f"{gap_stock['rsi']:.1f}")
                                st.metric("Volume Ratio", f"{gap_stock['volume_ratio']:.2f}x")
                            
                            with col3:
                                # Calculate probability
                                prob = 70
                                if gap_stock['is_filling']:
                                    prob += 15
                                if abs(gap_stock['gap_pct']) < 2:
                                    prob += 10
                                elif abs(gap_stock['gap_pct']) > 5:
                                    prob -= 20
                                prob = min(95, max(40, prob))
                                
                                st.metric("Fill Probability", f"{prob}%")
                                
                                if gap_stock['is_filling']:
                                    st.success("Price is currently filling the gap!")
                                else:
                                    st.info("Waiting for gap fill to begin")
                            
                            st.markdown(f"""
                            **Trading Strategy:**
                            - **Entry:** Current price (₹{gap_stock['curr_close']:.2f})
                            - **Target:** Previous close (₹{gap_stock['prev_close']:.2f})
                            - **Reason:** Gaps have a high probability of being filled. Current progress: {gap_stock['fill_progress']:.1f}%
                            """)
                else:
                    st.info("No significant gaps detected in the current timeframe.")
            
            # Tab 3: Fundamental Analysis
            with tabs[2]:
                st.markdown('<div class="sub-header">💰 Fundamental Analysis</div>', unsafe_allow_html=True)
                
                if not fetch_fundamentals:
                    st.warning("Fundamental data fetching is disabled. Enable it in the sidebar to see this analysis.")
                else:
                    stocks_with_fundamentals = [r for r in results if 'fundamentals' in r and r['fundamentals']]
                    
                    if stocks_with_fundamentals:
                        st.write(f"**Fundamental analysis for {len(stocks_with_fundamentals)} stocks**")
                        
                        # Metric descriptions
                        with st.expander("📚 Understanding Fundamental Metrics"):
                            st.markdown("""
                            **Key Fundamental Metrics Explained:**
                            
                            1. **Market Cap**: Total market value of company's shares. Indicates company size.
                            2. **P/E Ratio (Price-to-Earnings)**: Price relative to earnings. Lower may indicate undervaluation.
                               - Good: 10-25 | High: >25 | Low: <10
                            3. **P/B Ratio (Price-to-Book)**: Price relative to book value. Measures valuation.
                               - Good: 1-3 | High: >3 | Low: <1
                            4. **Debt-to-Equity**: Total debt relative to equity. Measures financial leverage.
                               - Good: <1 | Moderate: 1-2 | High: >2
                            5. **ROE (Return on Equity)**: Profitability relative to equity. Higher is better.
                               - Strong: >15% | Moderate: 10-15% | Weak: <10%
                            6. **EPS (Earnings Per Share)**: Company's profit per share. Higher indicates better profitability.
                            7. **Dividend Yield**: Annual dividend as % of price. Income for investors.
                               - Attractive: >2% | Moderate: 1-2% | Low: <1%
                            8. **Profit Margins**: Net income as % of revenue. Shows operational efficiency.
                               - Healthy: >10% | Moderate: 5-10% | Low: <5%
                            9. **Current Ratio**: Current assets / current liabilities. Measures liquidity.
                               - Good: >1.5 | Acceptable: 1-1.5 | Concern: <1
                            10. **Revenue Growth**: YoY revenue increase. Indicates business expansion.
                            """)
                        
                        for stock_data in stocks_with_fundamentals:
                            ticker = stock_data['ticker']
                            fundamentals = stock_data['fundamentals']
                            analysis = stock_data['fundamental_analysis']
                            
                            with st.expander(f"{ticker} - {analysis['rating']} ({analysis['rating_pct']:.0f}/100)"):
                                # Display rating
                                st.markdown(f"""
                                <div class="{analysis['rating_class']}">
                                    <h3>Overall Rating: {analysis['rating']}</h3>
                                    <p>Fundamental Score: {analysis['score']:.1f} / {analysis['max_score']:.0f} ({analysis['rating_pct']:.0f}%)</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    if fundamentals.get('market_cap'):
                                        market_cap_cr = fundamentals['market_cap'] / 10000000
                                        st.metric("Market Cap", f"₹{market_cap_cr:.0f} Cr")
                                    if fundamentals.get('pe_ratio'):
                                        st.metric("P/E Ratio", f"{fundamentals['pe_ratio']:.2f}")
                                    if fundamentals.get('pb_ratio'):
                                        st.metric("P/B Ratio", f"{fundamentals['pb_ratio']:.2f}")
                                
                                with col2:
                                    if fundamentals.get('debt_to_equity'):
                                        st.metric("Debt/Equity", f"{fundamentals['debt_to_equity']:.2f}")
                                    if fundamentals.get('roe'):
                                        st.metric("ROE", f"{fundamentals['roe']*100:.2f}%")
                                    if fundamentals.get('eps'):
                                        st.metric("EPS", f"₹{fundamentals['eps']:.2f}")
                                
                                with col3:
                                    if fundamentals.get('div_yield'):
                                        st.metric("Div Yield", f"{fundamentals['div_yield']*100:.2f}%")
                                    if fundamentals.get('book_value'):
                                        st.metric("Book Value", f"₹{fundamentals['book_value']:.2f}")
                                    if fundamentals.get('profit_margins'):
                                        st.metric("Profit Margin", f"{fundamentals['profit_margins']*100:.2f}%")
                                
                                with col4:
                                    if fundamentals.get('current_ratio'):
                                        st.metric("Current Ratio", f"{fundamentals['current_ratio']:.2f}")
                                    if fundamentals.get('revenue_growth'):
                                        st.metric("Revenue Growth", f"{fundamentals['revenue_growth']*100:.2f}%")
                                    st.write(f"**Sector:** {fundamentals.get('sector', 'N/A')}")
                                
                                # Analysis summary
                                st.markdown("**Analysis Summary:**")
                                for reason in analysis['reasons']:
                                    st.write(f"• {reason}")
                                
                                # Investment summary
                                if analysis['rating'] == 'Strong':
                                    st.success(f"**Investment Summary:** {ticker} shows strong fundamentals with healthy financial metrics. Good for long-term investment consideration.")
                                elif analysis['rating'] == 'Moderate':
                                    st.info(f"**Investment Summary:** {ticker} has moderate fundamentals. Some strengths and some areas of concern. Requires careful analysis before investment.")
                                else:
                                    st.warning(f"**Investment Summary:** {ticker} shows weak fundamentals. Higher risk investment with concerns in key financial metrics.")
                    else:
                        st.info("No fundamental data available. This may be due to API limitations or data unavailability.")
            
            # Tab 4: Market Heatmap
            with tabs[3]:
                st.markdown('<div class="sub-header">🗺️ Market Sentiment Heatmap</div>', unsafe_allow_html=True)
                
                # Show sentiment metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Stocks", total_count)
                with col2:
                    st.metric("Bullish", bullish_count, f"{(bullish_count/total_count)*100:.1f}%")
                with col3:
                    st.metric("Bearish", bearish_count, f"{(bearish_count/total_count)*100:.1f}%")
                with col4:
                    st.metric("Neutral", neutral_count, f"{(neutral_count/total_count)*100:.1f}%")
                
                # Overall sentiment
                if bullish_count > bearish_count + neutral_count:
                    st.success("🟢 **Overall Market Sentiment: BULLISH** - Majority of stocks showing positive signals")
                elif bearish_count > bullish_count + neutral_count:
                    st.error("🔴 **Overall Market Sentiment: BEARISH** - Majority of stocks showing negative signals")
                else:
                    st.info("🟡 **Overall Market Sentiment: MIXED** - Market is showing mixed signals")
                
                # Create heatmap data
                heatmap_data = []
                for result in results:
                    signal = result['price_action']['signal']
                    change_pct = result['change_pct']
                    
                    if signal == 'BULLISH':
                        color_val = min(100, 50 + change_pct * 10)
                    elif signal == 'BEARISH':
                        color_val = max(-100, -50 + change_pct * 10)
                    else:
                        color_val = change_pct * 10
                    
                    heatmap_data.append({
                        'Ticker': result['ticker'],
                        'Signal': signal,
                        'Change %': change_pct,
                        'Value': color_val,
                        'RSI': result['rsi'],
                        'Price': result['price']
                    })
                
                df_heatmap = pd.DataFrame(heatmap_data)
                
                # Create treemap
                fig = px.treemap(
                    df_heatmap,
                    path=['Signal', 'Ticker'],
                    values=abs(df_heatmap['Change %']) + 0.1,
                    color='Value',
                    color_continuous_scale=['#ff0000', '#ffaa00', '#00ff00'],
                    color_continuous_midpoint=0,
                    hover_data=['Change %', 'RSI', 'Price'],
                    title=f'{index_type} Stock Heatmap - Color: Green (Bullish) | Yellow (Neutral) | Red (Bearish)'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal breakdown
                st.markdown("**Signal Distribution by Stocks:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🟢 Bullish Stocks:**")
                    bullish_stocks = [r for r in results if r['price_action']['signal'] == 'BULLISH']
                    for stock in bullish_stocks[:10]:
                        st.write(f"• {stock['ticker']}: {stock['change_pct']:+.2f}%")
                
                with col2:
                    st.markdown("**🔴 Bearish Stocks:**")
                    bearish_stocks = [r for r in results if r['price_action']['signal'] == 'BEARISH']
                    for stock in bearish_stocks[:10]:
                        st.write(f"• {stock['ticker']}: {stock['change_pct']:+.2f}%")
                
                with col3:
                    st.markdown("**🟡 Neutral Stocks:**")
                    neutral_stocks = [r for r in results if r['price_action']['signal'] == 'NEUTRAL']
                    for stock in neutral_stocks[:10]:
                        st.write(f"• {stock['ticker']}: {stock['change_pct']:+.2f}%")
            
            # Tab 5: All Stocks Overview
            with tabs[4]:
                st.markdown('<div class="sub-header">📈 All Stocks Overview</div>', unsafe_allow_html=True)
                
                overview_data = []
                for result in results:
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
                
                def color_change(val):
                    if isinstance(val, str) and '%' in val:
                        num = float(val.replace('%', ''))
                        color = 'green' if num > 0 else 'red' if num < 0 else 'gray'
                        return f'color: {color}; font-weight: bold'
                    return ''
                
                styled_df = df_overview.style.applymap(color_change, subset=['Change %'])
                st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Tab 6: Top Gainers/Losers
            with tabs[5]:
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
            
            # Tab 7: Active Stocks
            with tabs[6]:
                st.markdown('<div class="sub-header">📊 Active Stocks Analysis</div>', unsafe_allow_html=True)
                
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
                        
                        st.write("**Key Indicators:**")
                        for reason in stock['price_action']['reasons'][:3]:
                            st.write(f"• {reason}")
            
            # Tab 8: 52-Week High/Low
            with tabs[7]:
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
            
            # Tab 9: Technical Signals
            with tabs[8]:
                st.markdown('<div class="sub-header">📉 Technical Signals Summary</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bullish Signals", bullish_count)
                with col2:
                    st.metric("Bearish Signals", bearish_count)
                with col3:
                    st.metric("Neutral Signals", neutral_count)
                
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
        
        st.markdown("""
        ### 📚 Features
        
        **Supported Indices:**
        - NIFTY 50 (50 constituents)
        - BANK NIFTY (11 constituents)
        - Custom tickers via yfinance
        
        **Trading Strategies:**
        1. **Gap Fill Strategy** - Identifies gap ups/downs with fill potential and probability
        2. **Support/Resistance + EMA** - Price action near key levels with EMA crossovers
        3. **Breakout Strategy** - Volume-backed breakouts above/below recent ranges
        4. **Mean Reversion** - Bollinger Band extremes with RSI confirmation
        5. **Momentum Strategy** - MACD and RSI alignment for trend continuation
        
        **Analysis Tabs:**
        - 🎯 **Trade Recommendations** - Entry, SL, Target with R:R ratios
        - 🔄 **Gap Fill Opportunities** - Dedicated tab for gap analysis with probabilities
        - 💰 **Fundamental Analysis** - PE, PB, ROE, Debt/Equity, margins with ratings
        - 🗺️ **Market Heatmap** - Visual sentiment map with bullish/bearish breakdown
        - 📈 **All Stocks** - Complete overview with price changes
        - 🔥 **Top Gainers/Losers** - Best and worst performers
        - 📊 **Active Stocks** - High volume with >1% moves
        - 🎚️ **52-Week Levels** - Stocks near highs/lows
        - 📉 **Technical Signals** - Detailed indicator analysis with charts
        
        **Index Impact Analysis:**
        - Detailed calculation methodology explained
        - Formula breakdown: (Stock Change % × Weight %) / 100
        - Top positive and negative contributors
        - Complete contribution breakdown table
        """)

if __name__ == "__main__":
    main()
