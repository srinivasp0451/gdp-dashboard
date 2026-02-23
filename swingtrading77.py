"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         COMPLETE ALGORITHMIC TRADING SYSTEM v2.0                               ║
║  Strategies: 17 | SL Types: 18 | Target Types: 13 | Broker: Dhan              ║
║  Features: Backtest (Method1/2) | Live Trading | Multi-Account | Multi-Strike  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time as dtime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dhanhq import dhanhq
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.utc

ASSET_MAP = {
    "NIFTY 50":        "^NSEI",
    "NIFTY BANK":      "^NSEBANK",
    "SENSEX":          "^BSESN",
    "S&P 500":         "^GSPC",
    "NASDAQ":          "^IXIC",
    "DOW JONES":       "^DJI",
    "Reliance":        "RELIANCE.NS",
    "TCS":             "TCS.NS",
    "Infosys":         "INFY.NS",
    "HDFC Bank":       "HDFCBANK.NS",
    "ICICI Bank":      "ICICIBANK.NS",
    "Custom Ticker":   "CUSTOM",
}

INTERVAL_MAP = {
    "1 minute":  "1m",
    "5 minutes": "5m",
    "15 minutes":"15m",
    "30 minutes":"30m",
    "1 hour":    "1h",
    "1 day":     "1d",
    "1 week":    "1wk",
}

PERIOD_MAP = {
    "1 day":    "1d",
    "5 days":   "5d",
    "1 month":  "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year":   "1y",
    "2 years":  "2y",
    "5 years":  "5y",
}

STRATEGIES = [
    "EMA Crossover",
    "Simple Buy",
    "Simple Sell",
    "Price Crosses Threshold",
    "RSI-ADX-EMA Combined",
    "Percentage Change",
    "AI Price Action",
    "Custom Strategy (Multi-Indicator)",
    "SuperTrend AI",
    "VWAP + Volume Spike",
    "Bollinger Squeeze Breakout",
    "Elliott Waves + Ratio Charts",
    # 5 New High-Probability Strategies
    "Morning Breakout (ORB)",
    "MACD Divergence",
    "Triple EMA Momentum",
    "Mean Reversion (Bollinger + RSI)",
    "Ichimoku Cloud Breakout",
]

SL_TYPES = [
    "Custom Points",
    "P&L Based (Rupees)",
    "ATR-based",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "Signal-based (Reverse Crossover)",
    "Trailing SL (Points)",
    "Trailing Profit (Rupees)",
    "Trailing Loss (Rupees)",
    "Trailing SL + Current Candle",
    "Trailing SL + Previous Candle",
    "Trailing SL + Current Swing",
    "Trailing SL + Previous Swing",
    "Volatility-Adjusted Trailing SL",
    "Break-even After 50% Target",
    "Cost-to-Cost + N Points Trailing SL",
]

TARGET_TYPES = [
    "Custom Points",
    "P&L Based (Rupees)",
    "Trailing Target (Points)",
    "Trailing Target + Signal Based",
    "Dynamic Trailing SL+Target (Lock Profits)",
    "50% Exit at Target (Partial)",
    "Current Candle Low/High",
    "Previous Candle Low/High",
    "Current Swing Low/High",
    "Previous Swing Low/High",
    "ATR-based",
    "Risk-Reward Based",
    "Signal-based (Reverse Crossover)",
]

# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and convert timestamps to IST.
    Returns a DataFrame with columns: Datetime, Open, High, Low, Close, Volume
    """
    if not YFINANCE_AVAILABLE:
        st.error("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        # Rename datetime column
        if 'Datetime' in raw.columns:
            raw = raw.rename(columns={'Datetime': 'Datetime'})
        elif 'Date' in raw.columns:
            raw = raw.rename(columns={'Date': 'Datetime'})

        # Convert to IST
        if raw['Datetime'].dtype == 'object':
            raw['Datetime'] = pd.to_datetime(raw['Datetime'])

        if raw['Datetime'].dt.tz is None:
            raw['Datetime'] = raw['Datetime'].dt.tz_localize('UTC')
        raw['Datetime'] = raw['Datetime'].dt.tz_convert(IST)

        raw = raw[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])
        raw = raw.reset_index(drop=True)
        return raw
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ──────────────────────────────────────────────────────────────────────────────
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_supertrend(df: pd.DataFrame, period: int = 10,
                         multiplier: float = 3.0):
    """SuperTrend indicator. Returns (trend_series, direction_series)."""
    atr = calculate_atr(df, period)
    hl_avg = (df['High'] + df['Low']) / 2
    upper = hl_avg + multiplier * atr
    lower = hl_avg - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            supertrend.iloc[i] = np.nan
            direction.iloc[i]  = 1
            continue

        prev_upper = upper.iloc[i - 1] if not pd.isna(upper.iloc[i - 1]) else upper.iloc[i]
        prev_lower = lower.iloc[i - 1] if not pd.isna(lower.iloc[i - 1]) else lower.iloc[i]
        prev_dir   = direction.iloc[i - 1] if i > 0 else 1

        # Adjust bands
        curr_upper = upper.iloc[i] if (upper.iloc[i] < prev_upper or df['Close'].iloc[i - 1] > prev_upper) else prev_upper
        curr_lower = lower.iloc[i] if (lower.iloc[i] > prev_lower or df['Close'].iloc[i - 1] < prev_lower) else prev_lower

        if prev_dir == 1:
            curr_dir = -1 if df['Close'].iloc[i] < curr_lower else 1
        else:
            curr_dir = 1 if df['Close'].iloc[i] > curr_upper else -1

        supertrend.iloc[i] = curr_lower if curr_dir == 1 else curr_upper
        direction.iloc[i]  = curr_dir

    return supertrend, direction


def calculate_ichimoku(df: pd.DataFrame,
                       tenkan_period: int = 9,
                       kijun_period:  int = 26,
                       senkou_b_period: int = 52):
    """Ichimoku Cloud components."""
    def donchian_mid(series_high, series_low, n):
        return (series_high.rolling(n).max() + series_low.rolling(n).min()) / 2

    tenkan_sen  = donchian_mid(df['High'], df['Low'], tenkan_period)
    kijun_sen   = donchian_mid(df['High'], df['Low'], kijun_period)
    senkou_a    = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    senkou_b    = donchian_mid(df['High'], df['Low'], senkou_b_period).shift(kijun_period)
    chikou_span = df['Close'].shift(-kijun_period)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span


def calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate ALL indicators needed by all strategies.
    This ensures backtest and live trading use identical calculations.
    """
    df = df.copy()

    fast = config.get('ema_fast', 9)
    slow = config.get('ema_slow', 21)

    # EMA
    df['EMA_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['EMA_9']    = df['Close'].ewm(span=9,    adjust=False).mean()
    df['EMA_20']   = df['Close'].ewm(span=20,   adjust=False).mean()
    df['EMA_50']   = df['Close'].ewm(span=50,   adjust=False).mean()
    df['EMA_200']  = df['Close'].ewm(span=200,  adjust=False).mean()

    # SMA
    df['SMA_20']   = df['Close'].rolling(20).mean()
    df['SMA_50']   = df['Close'].rolling(50).mean()
    df['SMA_200']  = df['Close'].rolling(200).mean()

    # Bollinger Bands
    bb_period  = config.get('bb_period', 20)
    bb_std_dev = config.get('bb_std_dev', 2.0)
    bb_mid     = df['Close'].rolling(bb_period).mean()
    bb_std     = df['Close'].rolling(bb_period).std()
    df['BB_Upper']  = bb_mid + bb_std_dev * bb_std
    df['BB_Lower']  = bb_mid - bb_std_dev * bb_std
    df['BB_Mid']    = bb_mid
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, np.nan)
    df['BB_%B']     = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)

    # RSI
    rsi_period = config.get('rsi_period', 14)
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    macd_fast   = config.get('macd_fast', 12)
    macd_slow   = config.get('macd_slow', 26)
    macd_signal = config.get('macd_signal', 9)
    ema_f = df['Close'].ewm(span=macd_fast,   adjust=False).mean()
    ema_s = df['Close'].ewm(span=macd_slow,   adjust=False).mean()
    df['MACD']           = ema_f - ema_s
    df['MACD_Signal']    = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # ADX
    adx_period = config.get('adx_period', 14)
    plus_dm  = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm  < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr       = pd.concat([df['High'] - df['Low'],
                          (df['High'] - df['Close'].shift(1)).abs(),
                          (df['Low']  - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr_adx  = tr.rolling(adx_period).mean()
    plus_di  = 100 * (plus_dm.rolling(adx_period).mean() / atr_adx.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr_adx.replace(0, np.nan))
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX']       = dx.rolling(adx_period).mean()
    df['ADX_Plus']  = plus_di
    df['ADX_Minus'] = minus_di

    # ATR
    df['ATR'] = calculate_atr(df, 14)
    df['ATR_Fast'] = calculate_atr(df, config.get('st_atr_period', 10))

    # Volume
    df['Volume_MA']   = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA'].replace(0, np.nan)

    # VWAP (cumulative within dataset - for intraday use)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, np.nan)

    # Historical Volatility
    df['HV'] = df['Close'].pct_change().rolling(20).std() * (252 ** 0.5) * 100

    # Standard Deviation
    df['StdDev'] = df['Close'].rolling(20).std()

    # EMA Angle
    df['EMA_Fast_Angle'] = np.degrees(np.arctan(df['EMA_Fast'].diff()))
    df['EMA_Slow_Angle'] = np.degrees(np.arctan(df['EMA_Slow'].diff()))

    # Swing High/Low (5-candle lookback)
    window = 5
    df['Swing_High'] = df['High'].rolling(window=window, center=True).max()
    df['Swing_Low']  = df['Low'].rolling(window=window,  center=True).min()

    # SuperTrend
    st_supertrend, st_direction = calculate_supertrend(
        df, config.get('st_atr_period', 10), config.get('st_multiplier', 3.0))
    df['SuperTrend']  = st_supertrend
    df['ST_Direction'] = st_direction

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(df)
    df['Tenkan_Sen']  = tenkan
    df['Kijun_Sen']   = kijun
    df['Senkou_A']    = senkou_a
    df['Senkou_B']    = senkou_b
    df['Chikou_Span'] = chikou

    # Opening Range (first 15-min high/low of each day)
    if 'Datetime' in df.columns:
        df['_date'] = df['Datetime'].apply(
            lambda x: x.date() if hasattr(x, 'date') else x)
        orb_highs = {}
        orb_lows  = {}
        for d, grp in df.groupby('_date'):
            cutoff = grp[grp['Datetime'].apply(
                lambda x: x.time() <= dtime(9, 30))].copy() if len(grp) > 0 else grp
            if not cutoff.empty:
                orb_highs[d] = cutoff['High'].max()
                orb_lows[d]  = cutoff['Low'].min()
            else:
                orb_highs[d] = grp.iloc[0]['High'] if len(grp) > 0 else np.nan
                orb_lows[d]  = grp.iloc[0]['Low']  if len(grp) > 0 else np.nan
        df['ORB_High'] = df['_date'].map(orb_highs)
        df['ORB_Low']  = df['_date'].map(orb_lows)
        df.drop(columns=['_date'], inplace=True, errors='ignore')
    else:
        df['ORB_High'] = np.nan
        df['ORB_Low']  = np.nan

    return df

# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def _safe_val(series, idx, default=np.nan):
    """Safely get value from series at index."""
    try:
        v = series.iloc[idx]
        return default if pd.isna(v) else v
    except Exception:
        return default


def check_ema_crossover_strategy(df, idx, config, position=None):
    """Strategy 1: EMA Crossover with optional filters."""
    if idx < 2:
        return None, {}

    fast_now  = _safe_val(df['EMA_Fast'], idx)
    fast_prev = _safe_val(df['EMA_Fast'], idx - 1)
    slow_now  = _safe_val(df['EMA_Slow'], idx)
    slow_prev = _safe_val(df['EMA_Slow'], idx - 1)

    if any(pd.isna(v) for v in [fast_now, fast_prev, slow_now, slow_prev]):
        return None, {}

    angle = abs(_safe_val(df['EMA_Fast_Angle'], idx, 0))
    min_angle = config.get('ema_min_angle', 0.0)
    if angle < min_angle:
        return None, {}

    # ADX filter
    if config.get('use_adx_filter', False):
        adx = _safe_val(df['ADX'], idx, 0)
        if adx < config.get('adx_threshold', 20):
            return None, {}

    # RSI filter
    if config.get('use_rsi_filter', False):
        rsi = _safe_val(df['RSI'], idx, 50)
        rsi_min = config.get('rsi_filter_min', 30)
        rsi_max = config.get('rsi_filter_max', 70)
        if not (rsi_min <= rsi <= rsi_max):
            return None, {}

    bullish = (fast_prev <= slow_prev) and (fast_now > slow_now)
    bearish = (fast_prev >= slow_prev) and (fast_now < slow_now)

    # Entry filters
    entry_filter = config.get('ema_entry_filter', 'Simple Crossover')
    if entry_filter == 'Volume Confirmation':
        vol_ratio = _safe_val(df['Volume_Ratio'], idx, 1)
        if vol_ratio < 1.3:
            return None, {}

    info = {'angle': angle, 'ema_fast': fast_now, 'ema_slow': slow_now}
    if bullish:
        return 'LONG', info
    if bearish:
        return 'SHORT', info
    return None, {}


def check_simple_buy_strategy(df, idx, config, position=None):
    """Strategy 2: Always BUY."""
    return 'LONG', {}


def check_simple_sell_strategy(df, idx, config, position=None):
    """Strategy 3: Always SELL."""
    return 'SHORT', {}


def check_price_crosses_threshold(df, idx, config, position=None):
    """Strategy 4: Price crosses threshold (4 combinations)."""
    if idx < 1:
        return None, {}
    threshold  = config.get('threshold_price', 0)
    cross_dir  = config.get('threshold_cross_dir', 'Above')
    pos_type   = config.get('threshold_position', 'LONG')

    price_now  = _safe_val(df['Close'], idx)
    price_prev = _safe_val(df['Close'], idx - 1)

    if any(pd.isna(v) for v in [price_now, price_prev]):
        return None, {}

    if cross_dir == 'Above':
        crossed = (price_prev <= threshold) and (price_now > threshold)
    else:
        crossed = (price_prev >= threshold) and (price_now < threshold)

    return (pos_type, {}) if crossed else (None, {})


def check_rsi_adx_ema_combined(df, idx, config, position=None):
    """Strategy 5: RSI + ADX + EMA Combined."""
    if idx < 50:
        return None, {}
    rsi   = _safe_val(df['RSI'],     idx)
    adx   = _safe_val(df['ADX'],     idx)
    price = _safe_val(df['Close'],   idx)
    ema50 = _safe_val(df['EMA_50'],  idx)
    adx_thresh = config.get('adx_threshold', 25)

    if any(pd.isna(v) for v in [rsi, adx, price, ema50]):
        return None, {}

    if rsi < 30 and adx > adx_thresh and price > ema50:
        return 'LONG', {'rsi': rsi, 'adx': adx}
    if rsi > 70 and adx > adx_thresh and price < ema50:
        return 'SHORT', {'rsi': rsi, 'adx': adx}
    return None, {}


def check_percentage_change(df, idx, config, position=None):
    """Strategy 6: Percentage Change (4 combinations)."""
    if idx < 1:
        return None, {}
    threshold  = config.get('pct_threshold', 2.0)
    change_dir = config.get('pct_change_dir', 'Positive')
    pos_type   = config.get('pct_position', 'LONG')

    prev_close = _safe_val(df['Close'], idx - 1)
    curr_close = _safe_val(df['Close'], idx)
    if pd.isna(prev_close) or prev_close == 0 or pd.isna(curr_close):
        return None, {}

    pct_change = ((curr_close - prev_close) / prev_close) * 100

    if change_dir == 'Positive' and pct_change >= threshold:
        return pos_type, {'pct_change': pct_change}
    if change_dir == 'Negative' and pct_change <= -threshold:
        return pos_type, {'pct_change': pct_change}
    return None, {}


def check_ai_price_action(df, idx, config, position=None):
    """Strategy 7: AI Price Action - 3 consecutive higher/lower closes."""
    if idx < 3:
        return None, {}
    closes = [_safe_val(df['Close'], idx - i) for i in range(4)]
    if any(pd.isna(c) for c in closes):
        return None, {}
    if closes[2] > closes[3] and closes[1] > closes[2] and closes[0] > closes[1]:
        return 'LONG', {'pattern': '3 higher closes'}
    if closes[2] < closes[3] and closes[1] < closes[2] and closes[0] < closes[1]:
        return 'SHORT', {'pattern': '3 lower closes'}
    return None, {}


def check_custom_strategy(df, idx, config, position=None):
    """Strategy 8: Custom Multi-Indicator with AND/OR logic."""
    conditions = config.get('custom_conditions', [])
    if not conditions:
        return None, {}

    logic = config.get('custom_logic', 'AND')
    results = []

    for cond in conditions:
        ctype    = cond.get('type', '')
        pos_type = cond.get('position', 'LONG')
        met      = False

        if ctype == 'Price Crosses Indicator':
            indicator  = cond.get('indicator', 'EMA')
            period     = cond.get('period', 20)
            cross_type = cond.get('cross_type', 'Above')
            price_now  = _safe_val(df['Close'], idx)
            price_prev = _safe_val(df['Close'], idx - 1) if idx > 0 else np.nan
            ind_key    = f'EMA_{period}' if 'EMA' in indicator else f'SMA_{period}'
            if 'BB Upper' in indicator:
                ind_now  = _safe_val(df['BB_Upper'],  idx)
                ind_prev = _safe_val(df['BB_Upper'],  idx - 1)
            elif 'BB Lower' in indicator:
                ind_now  = _safe_val(df['BB_Lower'],  idx)
                ind_prev = _safe_val(df['BB_Lower'],  idx - 1)
            elif 'BB Mid' in indicator:
                ind_now  = _safe_val(df['BB_Mid'],    idx)
                ind_prev = _safe_val(df['BB_Mid'],    idx - 1)
            elif 'EMA' in indicator:
                col      = f'EMA_Fast' if period == config.get('ema_fast', 9) else f'EMA_Slow'
                try:
                    col = f'EMA_{period}'
                    ind_now  = _safe_val(df[col], idx)
                    ind_prev = _safe_val(df[col], idx - 1)
                except:
                    ind_now  = _safe_val(df['EMA_Fast'], idx)
                    ind_prev = _safe_val(df['EMA_Fast'], idx - 1)
            else:
                try:
                    col      = f'SMA_{period}'
                    ind_now  = _safe_val(df[col], idx)
                    ind_prev = _safe_val(df[col], idx - 1)
                except:
                    continue

            if not any(pd.isna(v) for v in [price_now, price_prev, ind_now, ind_prev]):
                if cross_type == 'Above':
                    met = (price_prev <= ind_prev) and (price_now > ind_now)
                else:
                    met = (price_prev >= ind_prev) and (price_now < ind_now)

        elif ctype == 'Indicator Crosses Level':
            indicator = cond.get('indicator', 'RSI')
            level     = cond.get('level', 50)
            cross_dir = cond.get('cross_dir', 'Above')
            ind_map   = {
                'RSI': 'RSI', 'MACD': 'MACD', 'MACD Histogram': 'MACD_Histogram',
                'ADX': 'ADX', 'Volume': 'Volume', 'BB %B': 'BB_%B',
                'ATR': 'ATR', 'Historical Volatility': 'HV', 'Std Dev': 'StdDev'
            }
            col = ind_map.get(indicator, 'RSI')
            ind_now  = _safe_val(df[col], idx) if col in df.columns else np.nan
            ind_prev = _safe_val(df[col], idx - 1) if col in df.columns else np.nan
            if not any(pd.isna(v) for v in [ind_now, ind_prev]):
                if cross_dir == 'Above':
                    met = (ind_prev <= level) and (ind_now > level)
                else:
                    met = (ind_prev >= level) and (ind_now < level)

        elif ctype == 'Indicator Crossover':
            cross_kind = cond.get('cross_kind', 'Fast EMA x Slow EMA')
            direction  = cond.get('direction', 'Bullish')
            if 'EMA' in cross_kind:
                a_now  = _safe_val(df['EMA_Fast'], idx)
                a_prev = _safe_val(df['EMA_Fast'], idx - 1)
                b_now  = _safe_val(df['EMA_Slow'], idx)
                b_prev = _safe_val(df['EMA_Slow'], idx - 1)
            elif 'RSI' in cross_kind:
                a_now  = _safe_val(df['RSI'], idx)
                a_prev = _safe_val(df['RSI'], idx - 1)
                b_now  = 50.0; b_prev = 50.0
            else:
                continue
            if not any(pd.isna(v) for v in [a_now, a_prev, b_now, b_prev]):
                if direction == 'Bullish':
                    met = (a_prev <= b_prev) and (a_now > b_now)
                else:
                    met = (a_prev >= b_prev) and (a_now < b_now)

        results.append((met, pos_type))

    if not results:
        return None, {}

    if logic == 'AND':
        if all(r[0] for r in results):
            return results[0][1], {'logic': 'AND', 'conditions': len(results)}
    else:  # OR
        for met, pt in results:
            if met:
                return pt, {'logic': 'OR', 'conditions': len(results)}
    return None, {}


def check_supertrend_ai(df, idx, config, position=None):
    """Strategy 9: SuperTrend AI with ADX and Volume confirmation."""
    if idx < 2:
        return None, {}
    adx_thresh = config.get('adx_threshold', 25)
    vol_mult   = config.get('st_vol_multiplier', 1.5)

    st_dir_now  = _safe_val(df['ST_Direction'], idx, 0)
    st_dir_prev = _safe_val(df['ST_Direction'], idx - 1, 0)
    adx         = _safe_val(df['ADX'], idx, 0)
    vol_ratio   = _safe_val(df['Volume_Ratio'], idx, 1)

    if any(pd.isna(v) for v in [st_dir_now, st_dir_prev]):
        return None, {}

    flipped_bull = (st_dir_prev == -1) and (st_dir_now == 1)
    flipped_bear = (st_dir_prev == 1)  and (st_dir_now == -1)

    if flipped_bull and adx > adx_thresh and vol_ratio >= vol_mult:
        return 'LONG', {'adx': adx, 'vol_ratio': vol_ratio}
    if flipped_bear and adx > adx_thresh and vol_ratio >= vol_mult:
        return 'SHORT', {'adx': adx, 'vol_ratio': vol_ratio}
    return None, {}


def check_vwap_volume_spike(df, idx, config, position=None):
    """Strategy 10: VWAP + Volume Spike."""
    if idx < 1:
        return None, {}
    vol_mult    = config.get('vwap_vol_multiplier', 2.0)
    max_dist    = config.get('vwap_max_dist', 0.3)
    rsi_ob      = config.get('vwap_rsi_ob', 70)
    rsi_os      = config.get('vwap_rsi_os', 30)

    price_now  = _safe_val(df['Close'], idx)
    price_prev = _safe_val(df['Close'], idx - 1)
    vwap_now   = _safe_val(df['VWAP'],  idx)
    vwap_prev  = _safe_val(df['VWAP'],  idx - 1)
    vol_ratio  = _safe_val(df['Volume_Ratio'], idx, 1)
    rsi        = _safe_val(df['RSI'], idx, 50)

    if any(pd.isna(v) for v in [price_now, price_prev, vwap_now, vwap_prev]):
        return None, {}

    dist_pct = abs(price_now - vwap_now) / vwap_now * 100 if vwap_now != 0 else 999

    bull = (price_prev <= vwap_prev) and (price_now > vwap_now)
    bear = (price_prev >= vwap_prev) and (price_now < vwap_now)

    if bull and vol_ratio >= vol_mult and rsi < 55 and dist_pct <= max_dist:
        return 'LONG', {'vwap': vwap_now, 'vol_ratio': vol_ratio}
    if bear and vol_ratio >= vol_mult and rsi > 45 and dist_pct <= max_dist:
        return 'SHORT', {'vwap': vwap_now, 'vol_ratio': vol_ratio}
    return None, {}


def check_bollinger_squeeze_breakout(df, idx, config, position=None):
    """Strategy 11: Bollinger Squeeze Breakout."""
    if idx < 1:
        return None, {}
    squeeze_thresh = config.get('bb_squeeze_thresh', 0.02)
    vol_mult       = config.get('bb_vol_multiplier', 1.8)

    bb_width  = _safe_val(df['BB_Width'], idx, 1)
    bb_upper  = _safe_val(df['BB_Upper'], idx)
    bb_lower  = _safe_val(df['BB_Lower'], idx)
    price_now = _safe_val(df['Close'],    idx)
    price_prev= _safe_val(df['Close'],    idx - 1)
    vol_ratio = _safe_val(df['Volume_Ratio'], idx, 1)

    if any(pd.isna(v) for v in [bb_upper, bb_lower, price_now]):
        return None, {}

    squeeze = bb_width < squeeze_thresh

    if squeeze and price_now > bb_upper and vol_ratio >= vol_mult:
        return 'LONG', {'bb_width': bb_width, 'vol_ratio': vol_ratio}
    if squeeze and price_now < bb_lower and vol_ratio >= vol_mult:
        return 'SHORT', {'bb_width': bb_width, 'vol_ratio': vol_ratio}
    return None, {}


def check_elliott_waves_ratio_charts(df, idx, config, position=None):
    """Strategy 12: Elliott Waves + Ratio Charts."""
    lookback      = config.get('wave_lookback', 13)
    ratio_period  = config.get('ratio_ema_period', 21)
    ratio_thresh  = config.get('ratio_threshold', 1.0)

    if idx < max(lookback + 1, ratio_period + 1):
        return None, {}

    highs  = df['High'].iloc[idx - lookback: idx + 1]
    lows   = df['Low'].iloc[idx - lookback: idx + 1]
    closes = df['Close'].iloc[idx - lookback: idx + 1]

    bull_waves = 0
    bear_waves = 0
    for i in range(1, len(highs)):
        if highs.iloc[i] > highs.iloc[i - 1]:
            bull_waves += 1
        if lows.iloc[i] < lows.iloc[i - 1]:
            bear_waves += 1

    price = _safe_val(df['Close'], idx)
    ema50 = _safe_val(df['EMA_50'], idx)
    if pd.isna(ema50) or ema50 == 0:
        return None, {}

    ratio = price / ema50
    ratio_series = (df['Close'] / df['EMA_50'].replace(0, np.nan)).iloc[:idx + 1]
    ratio_ma = ratio_series.rolling(ratio_period).mean().iloc[-1]
    if pd.isna(ratio_ma):
        return None, {}

    if bull_waves >= 5 and ratio > ratio_thresh and ratio > ratio_ma:
        return 'LONG', {'bull_waves': bull_waves, 'ratio': ratio}
    if bear_waves >= 5 and ratio < (2.0 - ratio_thresh) and ratio < ratio_ma:
        return 'SHORT', {'bear_waves': bear_waves, 'ratio': ratio}
    return None, {}


# ──────────── 5 NEW HIGH-PROBABILITY STRATEGIES ────────────────────────────

def check_morning_breakout_orb(df, idx, config, position=None):
    """
    Strategy 13: Morning Breakout / Opening Range Breakout (ORB).
    High probability intraday strategy.
    Enter when price breaks ORB High (LONG) or ORB Low (SHORT) with volume.
    """
    if idx < 1:
        return None, {}
    orb_high  = _safe_val(df['ORB_High'], idx)
    orb_low   = _safe_val(df['ORB_Low'],  idx)
    price_now = _safe_val(df['Close'],    idx)
    price_prev= _safe_val(df['Close'],    idx - 1)
    vol_ratio = _safe_val(df['Volume_Ratio'], idx, 1)
    orb_vol_mult = config.get('orb_vol_multiplier', 1.5)

    if any(pd.isna(v) for v in [orb_high, orb_low, price_now, price_prev]):
        return None, {}
    if orb_high == orb_low:
        return None, {}

    # Only trade during market hours
    if 'Datetime' in df.columns:
        t = df['Datetime'].iloc[idx]
        if hasattr(t, 'time'):
            trade_time = t.time()
            if not (dtime(9, 30) <= trade_time <= dtime(14, 30)):
                return None, {}

    bull = (price_prev <= orb_high) and (price_now > orb_high) and (vol_ratio >= orb_vol_mult)
    bear = (price_prev >= orb_low)  and (price_now < orb_low)  and (vol_ratio >= orb_vol_mult)

    if bull:
        return 'LONG', {'orb_high': orb_high, 'vol_ratio': vol_ratio}
    if bear:
        return 'SHORT', {'orb_low': orb_low, 'vol_ratio': vol_ratio}
    return None, {}


def check_macd_divergence(df, idx, config, position=None):
    """
    Strategy 14: MACD Divergence.
    Detects regular divergence between price and MACD for high-probability reversals.
    Bullish divergence: Price makes lower low, MACD makes higher low.
    Bearish divergence: Price makes higher high, MACD makes lower high.
    """
    lookback = config.get('macd_div_lookback', 20)
    if idx < lookback + 2:
        return None, {}

    # Look at last `lookback` candles
    prices = df['Close'].iloc[idx - lookback: idx + 1]
    macds  = df['MACD'].iloc[idx - lookback: idx + 1]

    if prices.isna().any() or macds.isna().any():
        return None, {}

    # Find recent pivot low (bullish divergence check)
    pivot_range = 5
    if idx < pivot_range * 2:
        return None, {}

    # Recent low vs previous low in price
    p_recent_lo_idx = prices.iloc[-pivot_range:].idxmin()
    p_prev_lo_idx   = prices.iloc[:-pivot_range].idxmin()
    m_recent_lo     = macds.loc[p_recent_lo_idx]
    m_prev_lo       = macds.loc[p_prev_lo_idx]
    p_recent_lo     = prices.loc[p_recent_lo_idx]
    p_prev_lo       = prices.loc[p_prev_lo_idx]

    # Recent high vs previous high in price
    p_recent_hi_idx = prices.iloc[-pivot_range:].idxmax()
    p_prev_hi_idx   = prices.iloc[:-pivot_range].idxmax()
    m_recent_hi     = macds.loc[p_recent_hi_idx]
    m_prev_hi       = macds.loc[p_prev_hi_idx]
    p_recent_hi     = prices.loc[p_recent_hi_idx]
    p_prev_hi       = prices.loc[p_prev_hi_idx]

    # MACD Histogram for confirmation
    hist = _safe_val(df['MACD_Histogram'], idx, 0)

    # Bullish divergence: price lower low, MACD higher low → expect UP
    bull_div = (p_recent_lo < p_prev_lo) and (m_recent_lo > m_prev_lo) and (hist > 0)
    # Bearish divergence: price higher high, MACD lower high → expect DOWN
    bear_div = (p_recent_hi > p_prev_hi) and (m_recent_hi < m_prev_hi) and (hist < 0)

    rsi = _safe_val(df['RSI'], idx, 50)

    if bull_div and rsi < 50:
        return 'LONG', {'divergence': 'bullish', 'macd_hist': hist}
    if bear_div and rsi > 50:
        return 'SHORT', {'divergence': 'bearish', 'macd_hist': hist}
    return None, {}


def check_triple_ema_momentum(df, idx, config, position=None):
    """
    Strategy 15: Triple EMA Momentum.
    All 3 EMAs (9, 20, 50) aligned in same direction = strong trend.
    Entry on pullback to fast EMA with RSI confirmation.
    Very high probability in trending markets.
    """
    if idx < 51:
        return None, {}

    ema9  = _safe_val(df['EMA_9'],  idx)
    ema20 = _safe_val(df['EMA_20'], idx)
    ema50 = _safe_val(df['EMA_50'], idx)
    ema9_prev  = _safe_val(df['EMA_9'],  idx - 1)
    ema20_prev = _safe_val(df['EMA_20'], idx - 1)
    price = _safe_val(df['Close'],  idx)
    rsi   = _safe_val(df['RSI'],    idx, 50)
    adx   = _safe_val(df['ADX'],    idx, 0)

    if any(pd.isna(v) for v in [ema9, ema20, ema50, price]):
        return None, {}

    # Bull alignment: EMA9 > EMA20 > EMA50, ADX strong
    bull_aligned = (ema9 > ema20 > ema50) and adx > 20
    # Bear alignment: EMA9 < EMA20 < EMA50, ADX strong
    bear_aligned = (ema9 < ema20 < ema50) and adx > 20

    # Pullback: price touched or crossed EMA9 from above (bull) or below (bear)
    bull_pullback = bull_aligned and (price > ema9) and (ema9_prev >= ema20_prev) and rsi > 45 and rsi < 65
    bear_pullback = bear_aligned and (price < ema9) and (ema9_prev <= ema20_prev) and rsi < 55 and rsi > 35

    if bull_pullback:
        return 'LONG', {'ema9': ema9, 'ema20': ema20, 'ema50': ema50, 'adx': adx}
    if bear_pullback:
        return 'SHORT', {'ema9': ema9, 'ema20': ema20, 'ema50': ema50, 'adx': adx}
    return None, {}


def check_mean_reversion_bb_rsi(df, idx, config, position=None):
    """
    Strategy 16: Mean Reversion with Bollinger Bands + RSI.
    High-probability reversal strategy.
    LONG: Price touches/breaks BB Lower + RSI oversold + RSI starts turning up.
    SHORT: Price touches/breaks BB Upper + RSI overbought + RSI starts turning down.
    """
    if idx < 21:
        return None, {}

    price    = _safe_val(df['Close'],    idx)
    bb_upper = _safe_val(df['BB_Upper'], idx)
    bb_lower = _safe_val(df['BB_Lower'], idx)
    bb_mid   = _safe_val(df['BB_Mid'],   idx)
    rsi      = _safe_val(df['RSI'],      idx, 50)
    rsi_prev = _safe_val(df['RSI'],      idx - 1, 50)
    vol_ratio= _safe_val(df['Volume_Ratio'], idx, 1)

    if any(pd.isna(v) for v in [price, bb_upper, bb_lower, bb_mid]):
        return None, {}

    mr_rsi_os = config.get('mr_rsi_os', 30)
    mr_rsi_ob = config.get('mr_rsi_ob', 70)

    # Oversold at lower band + RSI turning up
    bull = (price <= bb_lower) and (rsi <= mr_rsi_os) and (rsi > rsi_prev)
    # Overbought at upper band + RSI turning down
    bear = (price >= bb_upper) and (rsi >= mr_rsi_ob) and (rsi < rsi_prev)

    if bull:
        return 'LONG', {'bb_lower': bb_lower, 'rsi': rsi, 'mean_target': bb_mid}
    if bear:
        return 'SHORT', {'bb_upper': bb_upper, 'rsi': rsi, 'mean_target': bb_mid}
    return None, {}


def check_ichimoku_cloud_breakout(df, idx, config, position=None):
    """
    Strategy 17: Ichimoku Cloud Breakout.
    One of the most reliable trend-following strategies.
    LONG: Price breaks above cloud + Tenkan > Kijun + Chikou above cloud.
    SHORT: Price breaks below cloud + Tenkan < Kijun + Chikou below cloud.
    """
    if idx < 53:
        return None, {}

    price    = _safe_val(df['Close'],      idx)
    price_prev = _safe_val(df['Close'],    idx - 1)
    tenkan   = _safe_val(df['Tenkan_Sen'], idx)
    kijun    = _safe_val(df['Kijun_Sen'],  idx)
    senkou_a = _safe_val(df['Senkou_A'],   idx)
    senkou_b = _safe_val(df['Senkou_B'],   idx)
    adx      = _safe_val(df['ADX'],        idx, 0)

    if any(pd.isna(v) for v in [price, tenkan, kijun, senkou_a, senkou_b]):
        return None, {}

    cloud_top    = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    # Bullish: price breaks above cloud + Tenkan > Kijun
    bull = (price_prev <= cloud_top) and (price > cloud_top) and (tenkan > kijun) and adx > 20
    # Bearish: price breaks below cloud + Tenkan < Kijun
    bear = (price_prev >= cloud_bottom) and (price < cloud_bottom) and (tenkan < kijun) and adx > 20

    if bull:
        return 'LONG', {'cloud_top': cloud_top, 'tenkan': tenkan, 'kijun': kijun}
    if bear:
        return 'SHORT', {'cloud_bottom': cloud_bottom, 'tenkan': tenkan, 'kijun': kijun}
    return None, {}


# ──── Strategy dispatcher ────
STRATEGY_FUNC_MAP = {
    "EMA Crossover":                    check_ema_crossover_strategy,
    "Simple Buy":                       check_simple_buy_strategy,
    "Simple Sell":                      check_simple_sell_strategy,
    "Price Crosses Threshold":          check_price_crosses_threshold,
    "RSI-ADX-EMA Combined":             check_rsi_adx_ema_combined,
    "Percentage Change":                check_percentage_change,
    "AI Price Action":                  check_ai_price_action,
    "Custom Strategy (Multi-Indicator)":check_custom_strategy,
    "SuperTrend AI":                    check_supertrend_ai,
    "VWAP + Volume Spike":              check_vwap_volume_spike,
    "Bollinger Squeeze Breakout":       check_bollinger_squeeze_breakout,
    "Elliott Waves + Ratio Charts":     check_elliott_waves_ratio_charts,
    "Morning Breakout (ORB)":           check_morning_breakout_orb,
    "MACD Divergence":                  check_macd_divergence,
    "Triple EMA Momentum":              check_triple_ema_momentum,
    "Mean Reversion (Bollinger + RSI)": check_mean_reversion_bb_rsi,
    "Ichimoku Cloud Breakout":          check_ichimoku_cloud_breakout,
}

def get_strategy_signal(strategy_name, df, idx, config, position=None):
    """Unified entry point for strategy signals."""
    func = STRATEGY_FUNC_MAP.get(strategy_name)
    if func is None:
        return None, {}
    try:
        return func(df, idx, config, position)
    except Exception as e:
        return None, {}

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def is_within_trade_window(dt, config: dict) -> bool:
    """Check if datetime is within the configured trade window."""
    if not config.get('use_trade_window', False):
        return True
    try:
        if hasattr(dt, 'time'):
            t = dt.time()
        elif hasattr(dt, 'hour'):
            t = dt
        else:
            return True
        start = config.get('window_start', dtime(9, 30))
        end   = config.get('window_end',   dtime(15, 0))
        return start <= t <= end
    except Exception:
        return True


def should_allow_trade_direction(signal: str, config: dict) -> bool:
    """Filter by trade direction (Both / LONG Only / SHORT Only)."""
    direction_filter = config.get('direction_filter', 'Both')
    if direction_filter == 'Both':
        return True
    if direction_filter == 'LONG Only' and signal in ('LONG', 'BUY'):
        return True
    if direction_filter == 'SHORT Only' and signal in ('SHORT', 'SELL'):
        return True
    return False


def calculate_brokerage(entry_price: float, exit_price: float,
                        quantity: int, config: dict) -> float:
    """Calculate brokerage charges for a trade."""
    if not config.get('include_brokerage', False):
        return 0.0
    calc_type = config.get('brokerage_calc_type', 'Fixed')
    if calc_type == 'Fixed':
        return config.get('brokerage_fixed', 20.0) * 2  # entry + exit
    else:
        pct = config.get('brokerage_pct', 0.03) / 100
        return (entry_price + exit_price) * quantity * pct


def is_same_day_trade(entry_time, exit_time) -> bool:
    """Check if trade was entered and exited on same trading day."""
    try:
        e = entry_time if hasattr(entry_time, 'date') else entry_time
        x = exit_time  if hasattr(exit_time, 'date')  else exit_time
        if hasattr(e, 'date') and hasattr(x, 'date'):
            if e.date() != x.date():
                return False
            start_t = dtime(9, 15)
            end_t   = dtime(15, 0)
            et = e.time() if hasattr(e, 'time') else e
            xt = x.time() if hasattr(x, 'time') else x
            return (start_t <= et <= end_t) and (start_t <= xt <= end_t)
    except Exception:
        pass
    return True


def find_swing_high(df, idx, lookback=20):
    """Find most recent swing high."""
    start = max(0, idx - lookback)
    swing_h = _safe_val(df['Swing_High'], idx)
    if pd.isna(swing_h):
        swing_h = df['High'].iloc[start:idx + 1].max()
    return swing_h


def find_swing_low(df, idx, lookback=20):
    """Find most recent swing low."""
    start = max(0, idx - lookback)
    swing_l = _safe_val(df['Swing_Low'], idx)
    if pd.isna(swing_l):
        swing_l = df['Low'].iloc[start:idx + 1].min()
    return swing_l


def find_prev_swing_high(df, idx, lookback=20):
    """Find previous swing high (2nd most recent)."""
    start = max(0, idx - lookback)
    sub   = df['High'].iloc[start:idx]
    if len(sub) < 2:
        return _safe_val(df['High'], idx)
    return sub.nlargest(2).iloc[-1]


def find_prev_swing_low(df, idx, lookback=20):
    """Find previous swing low (2nd most recent)."""
    start = max(0, idx - lookback)
    sub   = df['Low'].iloc[start:idx]
    if len(sub) < 2:
        return _safe_val(df['Low'], idx)
    return sub.nsmallest(2).iloc[-1]


# ──────────────────────────────────────────────────────────────────────────────
# SL / TARGET CALCULATION
# ──────────────────────────────────────────────────────────────────────────────
def calculate_initial_sl(entry_price: float, position_type: str,
                         df: pd.DataFrame, idx: int, config: dict) -> float:
    """
    Calculate initial stop loss price.
    Returns the SL price (or None for signal-based).
    """
    sl_type  = config.get('sl_type', 'Custom Points')
    sl_pts   = config.get('sl_points', 10.0)
    quantity = config.get('quantity', 1)
    direction = 1 if position_type == 'LONG' else -1

    if sl_type == 'Custom Points':
        return entry_price - direction * sl_pts

    elif sl_type == 'P&L Based (Rupees)':
        sl_rupees = config.get('sl_rupees', 500.0)
        pts = sl_rupees / max(quantity, 1)
        return entry_price - direction * pts

    elif sl_type == 'ATR-based':
        atr = _safe_val(df['ATR'], idx, sl_pts)
        mult = config.get('sl_atr_mult', 1.5)
        return entry_price - direction * atr * mult

    elif sl_type == 'Current Candle Low/High':
        if position_type == 'LONG':
            return _safe_val(df['Low'],  idx, entry_price - sl_pts)
        else:
            return _safe_val(df['High'], idx, entry_price + sl_pts)

    elif sl_type == 'Previous Candle Low/High':
        prev = max(0, idx - 1)
        if position_type == 'LONG':
            return _safe_val(df['Low'],  prev, entry_price - sl_pts)
        else:
            return _safe_val(df['High'], prev, entry_price + sl_pts)

    elif sl_type == 'Current Swing Low/High':
        if position_type == 'LONG':
            return find_swing_low(df, idx)
        else:
            return find_swing_high(df, idx)

    elif sl_type == 'Previous Swing Low/High':
        if position_type == 'LONG':
            return find_prev_swing_low(df, idx)
        else:
            return find_prev_swing_high(df, idx)

    elif sl_type == 'Signal-based (Reverse Crossover)':
        return None  # Will use signal exit

    elif sl_type in ['Trailing SL (Points)', 'Trailing SL + Current Candle',
                     'Trailing SL + Previous Candle', 'Trailing SL + Current Swing',
                     'Trailing SL + Previous Swing', 'Volatility-Adjusted Trailing SL',
                     'Cost-to-Cost + N Points Trailing SL']:
        # Initial SL same as custom points; will trail
        if sl_type == 'Volatility-Adjusted Trailing SL':
            atr = _safe_val(df['ATR'], idx, sl_pts)
            return entry_price - direction * atr * 1.5
        return entry_price - direction * sl_pts

    elif sl_type == 'Trailing Profit (Rupees)':
        return entry_price  # SL at entry initially (break-even style)

    elif sl_type == 'Trailing Loss (Rupees)':
        sl_rupees = config.get('sl_rupees', 500.0)
        pts = sl_rupees / max(quantity, 1)
        return entry_price - direction * pts

    elif sl_type == 'Break-even After 50% Target':
        return entry_price - direction * sl_pts

    return entry_price - direction * sl_pts


def calculate_initial_target(entry_price: float, position_type: str,
                             df: pd.DataFrame, idx: int, config: dict,
                             initial_sl: float = None) -> float:
    """
    Calculate initial target price.
    Returns the target price (or None for signal-based).
    """
    target_type = config.get('target_type', 'Custom Points')
    target_pts  = config.get('target_points', 20.0)
    quantity    = config.get('quantity', 1)
    direction   = 1 if position_type == 'LONG' else -1

    if target_type == 'Signal-based (Reverse Crossover)':
        return None  # No price target; exits on reverse signal

    elif target_type == 'Custom Points':
        return entry_price + direction * target_pts

    elif target_type == 'P&L Based (Rupees)':
        target_rupees = config.get('target_rupees', 1000.0)
        pts = target_rupees / max(quantity, 1)
        return entry_price + direction * pts

    elif target_type in ['Trailing Target (Points)', 'Trailing Target + Signal Based',
                         'Dynamic Trailing SL+Target (Lock Profits)', '50% Exit at Target (Partial)']:
        return entry_price + direction * target_pts

    elif target_type == 'Current Candle Low/High':
        if position_type == 'LONG':
            return _safe_val(df['High'], idx, entry_price + target_pts)
        else:
            return _safe_val(df['Low'],  idx, entry_price - target_pts)

    elif target_type == 'Previous Candle Low/High':
        prev = max(0, idx - 1)
        if position_type == 'LONG':
            return _safe_val(df['High'], prev, entry_price + target_pts)
        else:
            return _safe_val(df['Low'],  prev, entry_price - target_pts)

    elif target_type == 'Current Swing Low/High':
        if position_type == 'LONG':
            return find_swing_high(df, idx)
        else:
            return find_swing_low(df, idx)

    elif target_type == 'Previous Swing Low/High':
        if position_type == 'LONG':
            return find_prev_swing_high(df, idx)
        else:
            return find_prev_swing_low(df, idx)

    elif target_type == 'ATR-based':
        atr = _safe_val(df['ATR'], idx, target_pts)
        mult = config.get('target_atr_mult', 2.0)
        return entry_price + direction * atr * mult

    elif target_type == 'Risk-Reward Based':
        if initial_sl is not None:
            sl_dist = abs(entry_price - initial_sl)
            rr = config.get('target_rr_ratio', 2.0)
            return entry_price + direction * sl_dist * rr
        return entry_price + direction * target_pts

    return entry_price + direction * target_pts


def update_trailing_sl(position: dict, current_price: float,
                       df: pd.DataFrame, idx: int, config: dict) -> float:
    """
    Update trailing stop loss. Returns updated SL price.
    Ensures SL only moves in favorable direction.
    """
    sl_type   = config.get('sl_type', 'Custom Points')
    sl_pts    = config.get('sl_points', 10.0)
    direction = 1 if position['type'] == 'LONG' else -1
    current_sl = position.get('sl_price', position['entry_price'])

    if sl_type == 'Trailing SL (Points)':
        new_sl = current_price - direction * sl_pts
        if direction == 1:
            return max(current_sl, new_sl)
        else:
            return min(current_sl, new_sl) if current_sl is not None else new_sl

    elif sl_type == 'Trailing Profit (Rupees)':
        qty = config.get('quantity', 1)
        trail_rupees = config.get('sl_rupees', 500.0)
        pts = trail_rupees / max(qty, 1)
        new_sl = current_price - direction * pts
        if direction == 1:
            return max(current_sl or position['entry_price'], new_sl)
        else:
            return min(current_sl or position['entry_price'], new_sl)

    elif sl_type == 'Trailing Loss (Rupees)':
        return current_sl  # Fixed, doesn't trail

    elif sl_type == 'Trailing SL + Current Candle':
        if position['type'] == 'LONG':
            candle_sl = _safe_val(df['Low'],  idx, current_price - sl_pts)
        else:
            candle_sl = _safe_val(df['High'], idx, current_price + sl_pts)
        new_sl = candle_sl
        if direction == 1:
            return max(current_sl, new_sl)
        else:
            return min(current_sl, new_sl)

    elif sl_type == 'Trailing SL + Previous Candle':
        prev = max(0, idx - 1)
        if position['type'] == 'LONG':
            candle_sl = _safe_val(df['Low'],  prev, current_price - sl_pts)
        else:
            candle_sl = _safe_val(df['High'], prev, current_price + sl_pts)
        if direction == 1:
            return max(current_sl, candle_sl)
        else:
            return min(current_sl, candle_sl)

    elif sl_type == 'Trailing SL + Current Swing':
        if position['type'] == 'LONG':
            swing_sl = find_swing_low(df, idx)
        else:
            swing_sl = find_swing_high(df, idx)
        if direction == 1:
            return max(current_sl, swing_sl)
        else:
            return min(current_sl, swing_sl)

    elif sl_type == 'Trailing SL + Previous Swing':
        if position['type'] == 'LONG':
            swing_sl = find_prev_swing_low(df, idx)
        else:
            swing_sl = find_prev_swing_high(df, idx)
        if direction == 1:
            return max(current_sl, swing_sl)
        else:
            return min(current_sl, swing_sl)

    elif sl_type == 'Volatility-Adjusted Trailing SL':
        atr = _safe_val(df['ATR'], idx, sl_pts)
        new_sl = current_price - direction * atr * 1.5
        if direction == 1:
            return max(current_sl, new_sl)
        else:
            return min(current_sl, new_sl)

    elif sl_type == 'Break-even After 50% Target':
        entry = position['entry_price']
        target = position.get('target_price', entry + direction * 20)
        half_target_reached = False
        if target is not None:
            half_dist = abs(target - entry) * 0.5
            if direction == 1 and current_price >= entry + half_dist:
                half_target_reached = True
            elif direction == -1 and current_price <= entry - half_dist:
                half_target_reached = True
        if half_target_reached:
            # Move SL to break-even
            breakeven = entry
            if direction == 1:
                return max(current_sl, breakeven)
            else:
                return min(current_sl, breakeven)
        return current_sl

    elif sl_type == 'Cost-to-Cost + N Points Trailing SL':
        entry      = position['entry_price']
        trigger_k  = config.get('c2c_trigger_k', 3.0)
        offset_n   = config.get('c2c_offset_n', 2.0)
        profit_pts = (current_price - entry) * direction

        if profit_pts < trigger_k:
            # Phase 1: Normal trailing
            new_sl = current_price - direction * sl_pts
        elif profit_pts < sl_pts:
            # Phase 2: Lock at entry + N
            new_sl = entry + direction * offset_n
        else:
            # Phase 3: Resume trailing with reduced distance
            reduced_dist = sl_pts - offset_n
            new_sl = current_price - direction * reduced_dist

        if direction == 1:
            return max(current_sl, new_sl)
        else:
            return min(current_sl, new_sl)

    return current_sl


def update_trailing_target(position: dict, current_price: float,
                           df: pd.DataFrame, idx: int, config: dict):
    """
    Update trailing target. Returns (new_target, partial_exit_triggered).
    """
    target_type   = config.get('target_type', 'Custom Points')
    target_pts    = config.get('target_points', 20.0)
    direction     = 1 if position['type'] == 'LONG' else -1
    current_target = position.get('target_price')
    partial_exit   = False

    if target_type == 'Trailing Target (Points)':
        new_target = current_price + direction * target_pts
        if current_target is None:
            return new_target, False
        if direction == 1:
            return max(current_target, new_target), False
        else:
            return min(current_target, new_target), False

    elif target_type == '50% Exit at Target (Partial)':
        # Flag partial exit when target hit
        if not position.get('partial_exit_done', False):
            partial_exit = True
        return current_target, partial_exit

    elif target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
        new_target = current_price + direction * target_pts
        if current_target is None:
            return new_target, False
        if direction == 1:
            return max(current_target, new_target), False
        else:
            return min(current_target, new_target), False

    return current_target, partial_exit

# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, config: dict) -> dict:
    """
    Core backtesting engine. Implements Method 1 (current close) & Method 2 (next open).
    Uses IDENTICAL logic to live trading for consistent results.
    """
    if df.empty:
        return {'trades': [], 'skipped_trades': [], 'metrics': {}, 'debug': {}}

    strategy_name   = config.get('strategy', 'EMA Crossover')
    use_method2     = config.get('use_method2', False)
    prevent_overlap = config.get('prevent_overlap', True)
    sl_type         = config.get('sl_type', 'Custom Points')
    target_type     = config.get('target_type', 'Custom Points')

    trades          = []
    skipped_trades  = []
    position        = None
    signals_skipped = 0
    signals_total   = 0

    # Min starting index to ensure indicators are computed
    start_idx = max(60, len(df) // 10)

    for idx in range(start_idx, len(df)):
        row        = df.iloc[idx]
        price      = row['Close']
        datetime_  = row.get('Datetime', idx)

        # ── Trade Window Exit ──
        if position is not None:
            if config.get('use_trade_window', False):
                if not is_within_trade_window(datetime_, config):
                    # Force-exit position
                    exit_price  = price
                    exit_reason = 'Trade Window Closed'
                    pnl, brokerage, net_pnl = _close_position(
                        position, exit_price, datetime_, exit_reason, df, idx, config)
                    trades.append(position.copy())
                    position = None
                    continue

        # ── Manage Active Position ──
        if position is not None:
            pos_type  = position['type']
            direction = 1 if pos_type == 'LONG' else -1

            # Track high/low during trade
            position['highest_price'] = max(position.get('highest_price', price), row['High'])
            position['lowest_price']  = min(position.get('lowest_price', price), row['Low'])

            # Update trailing SL
            if sl_type in ['Trailing SL (Points)', 'Trailing Profit (Rupees)',
                           'Trailing SL + Current Candle', 'Trailing SL + Previous Candle',
                           'Trailing SL + Current Swing', 'Trailing SL + Previous Swing',
                           'Volatility-Adjusted Trailing SL', 'Break-even After 50% Target',
                           'Cost-to-Cost + N Points Trailing SL']:
                if position.get('sl_price') is not None:
                    new_sl = update_trailing_sl(position, price, df, idx, config)
                    position['sl_price'] = new_sl

            # Update Dynamic Trailing Target
            if target_type in ['Trailing Target (Points)', 'Dynamic Trailing SL+Target (Lock Profits)']:
                if target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
                    # Also update trailing SL alongside target
                    sl_pts = config.get('sl_points', 10.0)
                    new_sl = price - direction * sl_pts
                    if direction == 1:
                        position['sl_price'] = max(position.get('sl_price') or 0, new_sl)
                    else:
                        position['sl_price'] = min(position.get('sl_price') or 999999, new_sl)
                new_target, partial = update_trailing_target(position, price, df, idx, config)
                position['target_price'] = new_target

            # ── Check SL Hit ──
            exit_reason = None
            if position.get('sl_price') is not None:
                if pos_type == 'LONG'  and row['Low']  <= position['sl_price']:
                    exit_reason = 'SL Hit'
                    price = position['sl_price']  # Simulate exact SL fill
                elif pos_type == 'SHORT' and row['High'] >= position['sl_price']:
                    exit_reason = 'SL Hit'
                    price = position['sl_price']

            # ── Check Target Hit ──
            if exit_reason is None and position.get('target_price') is not None:
                if pos_type == 'LONG'  and row['High'] >= position['target_price']:
                    exit_reason = 'Target Hit'
                    price = position['target_price']
                elif pos_type == 'SHORT' and row['Low']  <= position['target_price']:
                    exit_reason = 'Target Hit'
                    price = position['target_price']

            # ── Check Signal-Based Exit ──
            if exit_reason is None:
                use_signal_exit = (
                    sl_type     == 'Signal-based (Reverse Crossover)' or
                    target_type == 'Signal-based (Reverse Crossover)'
                )
                if use_signal_exit:
                    sig, _ = get_strategy_signal(strategy_name, df, idx, config, position)
                    if sig:
                        if (pos_type == 'LONG'  and sig in ('SELL', 'SHORT')) or \
                           (pos_type == 'SHORT' and sig in ('BUY',  'LONG')):
                            exit_reason = 'Signal Exit'
                            price = row['Close']

            # ── Close Position ──
            if exit_reason:
                _close_position(position, price, datetime_, exit_reason, df, idx, config)
                trades.append(position.copy())
                position = None
                continue

        # ── Look for New Signal ──
        if position is not None and prevent_overlap:
            # Track skipped signal
            sig, sig_info = get_strategy_signal(strategy_name, df, idx, config, position)
            if sig and should_allow_trade_direction(sig, config):
                signals_total += 1
                signals_skipped += 1
                # Simulate skipped trade
                skip = _simulate_skipped_trade(df, idx, sig, config)
                if skip:
                    skipped_trades.append(skip)
            continue

        if position is None:
            sig, sig_info = get_strategy_signal(strategy_name, df, idx, config, None)
            if sig is None:
                continue
            if not should_allow_trade_direction(sig, config):
                continue

            dt = datetime_
            if not is_within_trade_window(dt, config):
                continue

            signals_total += 1

            # ── Method 1 vs Method 2 Entry ──
            if use_method2:
                if idx + 1 >= len(df):
                    continue
                entry_row   = df.iloc[idx + 1]
                entry_price = entry_row['Open']
                entry_time  = entry_row.get('Datetime', idx + 1)
                entry_idx   = idx + 1
            else:
                entry_price = row['Close']
                entry_time  = datetime_
                entry_idx   = idx

            initial_sl = calculate_initial_sl(entry_price, sig, df, entry_idx, config)
            initial_tgt = calculate_initial_target(entry_price, sig, df, entry_idx, config, initial_sl)

            position = {
                'type':          sig,
                'entry_price':   entry_price,
                'entry_time':    entry_time,
                'sl_price':      initial_sl,
                'target_price':  initial_tgt,
                'exit_price':    None,
                'exit_time':     None,
                'exit_reason':   None,
                'pnl':           0.0,
                'brokerage':     0.0,
                'net_pnl':       0.0,
                'highest_price': entry_price,
                'lowest_price':  entry_price,
                'signal_info':   sig_info,
                'partial_exit_done': False,
                'quantity':      config.get('quantity', 1),
            }

    # Close any open position at end of data
    if position is not None:
        last_idx = len(df) - 1
        exit_price = df.iloc[last_idx]['Close']
        _close_position(position, exit_price, df.iloc[last_idx].get('Datetime', last_idx),
                        'End of Data', df, last_idx, config)
        trades.append(position.copy())

    metrics = _calculate_metrics(trades, config)
    debug   = {
        'total_signals': signals_total,
        'signals_skipped': signals_skipped,
        'overlapping_trades_tracked': len(skipped_trades),
        'candles_analyzed': len(df) - start_idx,
    }

    return {
        'trades':        trades,
        'skipped_trades': skipped_trades,
        'metrics':       metrics,
        'debug':         debug,
        'df':            df,
    }


def _close_position(position, exit_price, exit_time, exit_reason,
                    df, idx, config):
    """Compute P&L and populate exit fields on position dict."""
    pos_type  = position['type']
    direction = 1 if pos_type == 'LONG' else -1
    qty       = position.get('quantity', config.get('quantity', 1))

    raw_pnl   = direction * (exit_price - position['entry_price']) * qty
    brokerage = calculate_brokerage(position['entry_price'], exit_price, qty, config)

    position['exit_price']  = exit_price
    position['exit_time']   = exit_time
    position['exit_reason'] = exit_reason
    position['pnl']         = round(raw_pnl, 2)
    position['brokerage']   = round(brokerage, 2)
    position['net_pnl']     = round(raw_pnl - brokerage, 2)

    if 'entry_time' in position and exit_time is not None:
        try:
            et = position['entry_time']
            xt = exit_time
            if hasattr(et, 'timestamp') and hasattr(xt, 'timestamp'):
                duration = (xt.timestamp() - et.timestamp()) / 60
            else:
                duration = 0
            position['duration_min'] = round(duration, 1)
        except:
            position['duration_min'] = 0
    return raw_pnl, brokerage, raw_pnl - brokerage


def _simulate_skipped_trade(df, signal_idx, signal, config):
    """Simulate outcome of a skipped (overlapping) trade."""
    try:
        direction = 1 if signal in ('LONG', 'BUY') else -1
        entry_price = df.iloc[signal_idx]['Close']
        sl_pts  = config.get('sl_points', 10.0)
        tgt_pts = config.get('target_points', 20.0)
        sl_price  = entry_price - direction * sl_pts
        tgt_price = entry_price + direction * tgt_pts
        qty = config.get('quantity', 1)

        max_sim = min(signal_idx + 100, len(df))
        exit_price  = df.iloc[max_sim - 1]['Close']
        exit_reason = 'Simulated End'

        for i in range(signal_idx + 1, max_sim):
            h = df.iloc[i]['High']
            l = df.iloc[i]['Low']
            if direction == 1:
                if l <= sl_price:
                    exit_price = sl_price; exit_reason = 'SL Hit'; break
                if h >= tgt_price:
                    exit_price = tgt_price; exit_reason = 'Target Hit'; break
            else:
                if h >= sl_price:
                    exit_price = sl_price; exit_reason = 'SL Hit'; break
                if l <= tgt_price:
                    exit_price = tgt_price; exit_reason = 'Target Hit'; break

        pnl = direction * (exit_price - entry_price) * qty
        return {
            'type':       signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl':        round(pnl, 2),
            'exit_reason': exit_reason,
            'note':       'Overlapped with active trade',
        }
    except Exception:
        return None


def _calculate_metrics(trades, config):
    """Calculate comprehensive backtest metrics."""
    if not trades:
        return {}

    use_net = config.get('include_brokerage', False)
    pnl_key = 'net_pnl' if use_net else 'pnl'

    total   = len(trades)
    winners = [t for t in trades if t.get(pnl_key, 0) > 0]
    losers  = [t for t in trades if t.get(pnl_key, 0) <= 0]

    total_pnl     = sum(t.get('pnl',     0) for t in trades)
    total_net_pnl = sum(t.get('net_pnl', 0) for t in trades)
    total_brokerage = sum(t.get('brokerage', 0) for t in trades)

    win_rate = len(winners) / total * 100 if total > 0 else 0
    avg_pnl  = total_pnl / total if total > 0 else 0
    avg_net  = total_net_pnl / total if total > 0 else 0

    # Max drawdown (using cumulative P&L)
    pnl_series = [t.get(pnl_key, 0) for t in trades]
    cum_pnl    = 0
    peak       = 0
    max_dd     = 0
    for p in pnl_series:
        cum_pnl += p
        peak     = max(peak, cum_pnl)
        dd       = peak - cum_pnl
        max_dd   = max(max_dd, dd)

    avg_win  = sum(t.get(pnl_key, 0) for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.get(pnl_key, 0) for t in losers)  / len(losers)  if losers  else 0
    profit_factor = abs(sum(t.get(pnl_key, 0) for t in winners)) / abs(sum(t.get(pnl_key, 0) for t in losers)) if losers and sum(t.get(pnl_key, 0) for t in losers) != 0 else float('inf')

    return {
        'total_trades':    total,
        'win_rate':        round(win_rate, 2),
        'winning_trades':  len(winners),
        'losing_trades':   len(losers),
        'total_pnl':       round(total_pnl, 2),
        'total_net_pnl':   round(total_net_pnl, 2),
        'total_brokerage': round(total_brokerage, 2),
        'avg_pnl':         round(avg_pnl, 2),
        'avg_net_pnl':     round(avg_net, 2),
        'avg_win':         round(avg_win, 2),
        'avg_loss':        round(avg_loss, 2),
        'max_drawdown':    round(max_dd, 2),
        'profit_factor':   round(profit_factor, 2),
    }

# ──────────────────────────────────────────────────────────────────────────────
# DHAN BROKER INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────
class DhanBrokerIntegration:
    """Dhan broker integration with full order management."""

    EXCHANGE_MAP = {
        'NSE': 'NSE_EQ',
        'BSE': 'BSE_EQ',
        'NSE_FO': 'NSE_FNO',
    }
    PRODUCT_INTRADAY  = 'INTRADAY'
    PRODUCT_CNC       = 'CNC'
    PRODUCT_BO        = 'BO'

    def __init__(self, client_id: str, access_token: str):
        self.client_id    = client_id
        self.access_token = access_token
        self.dhan         = None
        self.connected    = False

        if DHAN_AVAILABLE and client_id and access_token:
            try:
                self.dhan      = dhanhq(client_id, access_token)
                self.connected = True
            except Exception as e:
                self.connected = False

    def _get_exchange_segment(self, config: dict) -> str:
        asset_type = config.get('asset_type', 'Stocks')
        exchange   = config.get('exchange', 'NSE')
        if asset_type in ('Options', 'Stock Options'):
            return 'NSE_FNO'
        return self.EXCHANGE_MAP.get(exchange, 'NSE_EQ')

    def place_order(self, transaction_type: str, security_id: str,
                    quantity: int, config: dict,
                    price: float = 0, log_func=None) -> dict:
        """Place a single order on Dhan."""
        if not self.connected or self.dhan is None:
            msg = f"[BROKER SIMULATION] {transaction_type} {quantity} @ {price}"
            if log_func:
                log_func(msg)
            return {'status': 'simulated', 'orderId': 'SIM001'}

        exchange  = self._get_exchange_segment(config)
        prod_type = self.PRODUCT_INTRADAY if config.get('trading_type') == 'Intraday' else self.PRODUCT_CNC
        order_type_str = config.get('order_type', 'Market Order')
        use_bracket    = config.get('use_bracket', False)

        try:
            if use_bracket:
                sl_pts  = config.get('broker_sl_points', 50)
                tgt_pts = config.get('broker_target_points', 100)
                trail   = config.get('broker_trail_sl', 0)
                resp = self.dhan.place_order(
                    transaction_type = transaction_type,
                    exchange_segment = exchange,
                    product_type     = self.PRODUCT_BO,
                    order_type       = 'LIMIT',
                    security_id      = str(security_id),
                    quantity         = quantity,
                    price            = price,
                    bo_profit_value  = tgt_pts,
                    bo_stop_loss_value = sl_pts,
                    trail_stop_loss  = trail,
                )
            else:
                otype  = 'MARKET' if order_type_str == 'Market Order' else 'LIMIT'
                fill_p = 0 if otype == 'MARKET' else price
                resp   = self.dhan.place_order(
                    transaction_type = transaction_type,
                    exchange_segment = exchange,
                    product_type     = prod_type,
                    order_type       = otype,
                    security_id      = str(security_id),
                    quantity         = quantity,
                    price            = fill_p,
                )

            if log_func:
                log_func(f"✅ Order placed: {transaction_type} {quantity}x {security_id} - {resp}")
            return resp
        except Exception as e:
            if log_func:
                log_func(f"❌ Order error: {e}")
            return {'status': 'error', 'message': str(e)}

    def enter_broker_position(self, signal: str, entry_price: float,
                               config: dict, log_func=None) -> list:
        """Enter position(s) - handles multi-account and multi-strike."""
        results   = []
        tx_type   = 'BUY' if signal in ('LONG', 'BUY') else 'SELL'
        quantity  = config.get('quantity', 1)
        sec_id    = config.get('security_id', '')

        accounts = config.get('multi_accounts', []) or []
        if not accounts:
            accounts = [{'client_id': self.client_id, 'access_token': self.access_token}]

        # Determine strike list
        asset_type = config.get('asset_type', 'Stocks')
        use_multi_strike = config.get('use_multi_strike', False) and asset_type == 'Options'
        option_side = config.get('option_side', 'CE')

        if use_multi_strike:
            strike_key = f'multi_strikes_{option_side.lower()}'
            strikes = config.get(strike_key, [sec_id]) or [sec_id]
        else:
            strikes = [sec_id]

        for acc in accounts:
            broker = DhanBrokerIntegration(acc['client_id'], acc['access_token'])
            for sid in strikes:
                r = broker.place_order(tx_type, sid, quantity, config, entry_price, log_func)
                results.append(r)

        return results

    def exit_broker_position(self, signal: str, exit_price: float,
                              config: dict, log_func=None) -> list:
        """Exit position(s)."""
        tx_type  = 'SELL' if signal in ('LONG', 'BUY') else 'BUY'
        quantity = config.get('quantity', 1)
        sec_id   = config.get('security_id', '')

        accounts = config.get('multi_accounts', []) or []
        if not accounts:
            accounts = [{'client_id': self.client_id, 'access_token': self.access_token}]

        asset_type = config.get('asset_type', 'Stocks')
        use_multi_strike = config.get('use_multi_strike', False) and asset_type == 'Options'
        option_side = config.get('option_side', 'CE')
        if use_multi_strike:
            strike_key = f'multi_strikes_{option_side.lower()}'
            strikes = config.get(strike_key, [sec_id]) or [sec_id]
        else:
            strikes = [sec_id]

        results = []
        for acc in accounts:
            broker = DhanBrokerIntegration(acc['client_id'], acc['access_token'])
            for sid in strikes:
                r = broker.place_order(tx_type, sid, quantity, config, exit_price, log_func)
                results.append(r)
        return results

    def clear_all_positions(self, config: dict, log_func=None):
        """Cancel pending orders and close all positions."""
        if not self.connected or self.dhan is None:
            if log_func:
                log_func("🧹 [Simulation] Clearing all positions...")
            return
        try:
            if log_func:
                log_func("🧹 Clearing all existing positions...")
            order_list = self.dhan.get_order_list()
            cancelled  = 0
            closed     = 0
            if order_list and 'data' in order_list:
                for order in order_list['data']:
                    status = order.get('orderStatus', '')
                    if status == 'PENDING':
                        try:
                            self.dhan.cancel_order(order['orderId'])
                            cancelled += 1
                            if log_func:
                                log_func(f"✅ Cancelled pending order: {order['orderId']}")
                        except Exception as e:
                            if log_func:
                                log_func(f"⚠️ Cancel failed: {e}")
                    elif status == 'TRANSIT':
                        try:
                            reverse_tx = 'SELL' if order['transactionType'] == 'BUY' else 'BUY'
                            self.dhan.place_order(
                                transaction_type=reverse_tx,
                                exchange_segment=order.get('exchangeSegment', 'NSE_EQ'),
                                product_type='INTRADAY',
                                order_type='MARKET',
                                security_id=order['securityId'],
                                quantity=order['quantity'],
                                price=0,
                            )
                            closed += 1
                            if log_func:
                                log_func(f"✅ Closed transit position: {order['orderId']}")
                        except Exception as e:
                            if log_func:
                                log_func(f"⚠️ Close failed: {e}")
            if log_func:
                log_func(f"✅ Cleared: {cancelled} orders cancelled, {closed} positions closed")
        except Exception as e:
            if log_func:
                log_func(f"❌ Clear positions error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# LIVE TRADING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def add_log(message: str, log_key: str = 'live_logs'):
    """Add timestamped log entry."""
    now = datetime.now(IST).strftime('%H:%M:%S')
    entry = f"[{now}] {message}"
    if log_key not in st.session_state:
        st.session_state[log_key] = []
    st.session_state[log_key].insert(0, entry)
    if len(st.session_state[log_key]) > 200:
        st.session_state[log_key] = st.session_state[log_key][:200]


def live_trading_iteration(config: dict):
    """
    Single iteration of live trading loop.
    Uses IDENTICAL logic to backtesting for result consistency.
    """
    ticker   = config.get('ticker', '^NSEI')
    interval = config.get('interval', '5m')
    period   = '1d'  # Minimal data for live

    # Fetch fresh data
    df_raw = fetch_data(ticker, interval, period)
    if df_raw.empty:
        add_log("⚠️ No data fetched – skipping iteration")
        return

    df = calculate_all_indicators(df_raw, config)
    if df.empty or len(df) < 30:
        add_log("⚠️ Insufficient data for indicators")
        return

    st.session_state['current_data'] = df

    idx         = len(df) - 1  # Latest candle
    row         = df.iloc[idx]
    price       = row['Close']
    datetime_   = row.get('Datetime', datetime.now(IST))
    position    = st.session_state.get('position')
    broker      = st.session_state.get('dhan_broker')
    sl_type     = config.get('sl_type', 'Custom Points')
    target_type = config.get('target_type', 'Custom Points')

    # ── Trade Window Exit ──
    if position is not None and config.get('use_trade_window', False):
        if not is_within_trade_window(datetime_, config):
            add_log("⏰ Trade window closed – force exiting position")
            _live_exit(position, price, 'Trade Window Closed', broker, config, df, idx)
            return

    # ── Manage Active Position ──
    if position is not None:
        pos_type  = position['type']
        direction = 1 if pos_type == 'LONG' else -1

        # Update trailing SL
        if sl_type in ['Trailing SL (Points)', 'Trailing Profit (Rupees)',
                       'Trailing SL + Current Candle', 'Trailing SL + Previous Candle',
                       'Trailing SL + Current Swing', 'Trailing SL + Previous Swing',
                       'Volatility-Adjusted Trailing SL', 'Break-even After 50% Target',
                       'Cost-to-Cost + N Points Trailing SL']:
            if position.get('sl_price') is not None:
                new_sl = update_trailing_sl(position, price, df, idx, config)
                if new_sl != position.get('sl_price'):
                    position['sl_price'] = new_sl
                    add_log(f"📊 Trailing SL updated → ₹{new_sl:.2f}")

        # Update Dynamic Trailing Target
        if target_type in ['Trailing Target (Points)', 'Dynamic Trailing SL+Target (Lock Profits)']:
            if target_type == 'Dynamic Trailing SL+Target (Lock Profits)':
                sl_pts = config.get('sl_points', 10.0)
                new_sl = price - direction * sl_pts
                if direction == 1:
                    position['sl_price'] = max(position.get('sl_price') or 0, new_sl)
                else:
                    position['sl_price'] = min(position.get('sl_price') or 999999, new_sl)
            new_target, _ = update_trailing_target(position, price, df, idx, config)
            position['target_price'] = new_target

        # Check SL
        exit_reason = None
        exit_price  = price
        if position.get('sl_price') is not None:
            if pos_type == 'LONG'  and price <= position['sl_price']:
                exit_reason = 'SL Hit'
                exit_price  = position['sl_price']
            elif pos_type == 'SHORT' and price >= position['sl_price']:
                exit_reason = 'SL Hit'
                exit_price  = position['sl_price']

        # Check Target
        if exit_reason is None and position.get('target_price') is not None:
            if pos_type == 'LONG'  and price >= position['target_price']:
                exit_reason = 'Target Hit'
                exit_price  = position['target_price']
            elif pos_type == 'SHORT' and price <= position['target_price']:
                exit_reason = 'Target Hit'
                exit_price  = position['target_price']

        # Signal Exit
        if exit_reason is None:
            use_sig = (sl_type == 'Signal-based (Reverse Crossover)' or
                       target_type == 'Signal-based (Reverse Crossover)')
            if use_sig:
                sig, _ = get_strategy_signal(config.get('strategy'), df, idx, config, position)
                if sig:
                    if (pos_type == 'LONG'  and sig in ('SELL', 'SHORT')) or \
                       (pos_type == 'SHORT' and sig in ('BUY',  'LONG')):
                        exit_reason = 'Signal Exit'
                        exit_price  = price

        if exit_reason:
            _live_exit(position, exit_price, exit_reason, broker, config, df, idx)
            return

        pnl = (price - position['entry_price']) * (1 if pos_type == 'LONG' else -1) * config.get('quantity', 1)
        add_log(f"📈 {'🟢' if pos_type == 'LONG' else '🔴'} {pos_type} Active | Price: ₹{price:.2f} | P&L: ₹{pnl:.2f} | SL: ₹{position.get('sl_price', 0):.2f}")
        st.session_state['position'] = position
        return

    # ── Look for New Entry ──
    if not is_within_trade_window(datetime_, config):
        return

    sig, sig_info = get_strategy_signal(config.get('strategy'), df, idx, config, None)
    if sig is None:
        return
    if not should_allow_trade_direction(sig, config):
        return

    # Check no existing broker position (prevent duplicates)
    if st.session_state.get('broker_position'):
        add_log("⚠️ Broker position already active – skipping new signal")
        return

    entry_price = price
    add_log(f"🚨 Signal detected: {sig} @ ₹{entry_price:.2f}")

    # Clear positions if enabled
    if config.get('clear_positions_before_entry', False) and broker:
        broker.clear_all_positions(config, lambda m: add_log(m))

    # Calculate SL & Target
    initial_sl  = calculate_initial_sl(entry_price, sig, df, idx, config)
    initial_tgt = calculate_initial_target(entry_price, sig, df, idx, config, initial_sl)

    # Enter broker position
    if broker and config.get('enable_broker', False):
        broker.enter_broker_position(sig, entry_price, config, lambda m: add_log(m))
        st.session_state['broker_position'] = {'signal': sig, 'price': entry_price}

    position = {
        'type':        sig,
        'entry_price': entry_price,
        'entry_time':  datetime_,
        'sl_price':    initial_sl,
        'target_price': initial_tgt,
        'exit_price':  None,
        'exit_time':   None,
        'exit_reason': None,
        'pnl':         0.0,
        'brokerage':   0.0,
        'net_pnl':     0.0,
        'highest_price': entry_price,
        'lowest_price':  entry_price,
        'signal_info': sig_info,
        'quantity':    config.get('quantity', 1),
    }
    st.session_state['position'] = position
    add_log(f"📈 ENTERED {'🟢 LONG' if sig == 'LONG' else '🔴 SHORT'} @ ₹{entry_price:.2f} | SL: ₹{initial_sl:.2f} | Target: ₹{initial_tgt:.2f}" if initial_tgt else f"📈 ENTERED {sig} @ ₹{entry_price:.2f} | SL: ₹{initial_sl:.2f} | Target: Signal-based")


def _live_exit(position, exit_price, reason, broker, config, df, idx):
    """Execute live trade exit."""
    sig       = position['type']
    qty       = position.get('quantity', config.get('quantity', 1))
    direction = 1 if sig in ('LONG', 'BUY') else -1
    pnl       = direction * (exit_price - position['entry_price']) * qty
    brokerage = calculate_brokerage(position['entry_price'], exit_price, qty, config)
    net_pnl   = pnl - brokerage

    position['exit_price']  = exit_price
    position['exit_time']   = datetime.now(IST)
    position['exit_reason'] = reason
    position['pnl']         = round(pnl, 2)
    position['brokerage']   = round(brokerage, 2)
    position['net_pnl']     = round(net_pnl, 2)

    emoji = '✅' if pnl > 0 else '❌'
    add_log(f"{emoji} EXIT {sig} @ ₹{exit_price:.2f} | Reason: {reason} | P&L: ₹{pnl:.2f} | Net: ₹{net_pnl:.2f}")

    # Exit broker
    if broker and config.get('enable_broker', False):
        broker.exit_broker_position(sig, exit_price, config, lambda m: add_log(m))
    st.session_state['broker_position'] = None

    # Store in trade history
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []
    st.session_state['trade_history'].append(position.copy())
    st.session_state['position'] = None

# ──────────────────────────────────────────────────────────────────────────────
# CHARTING & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def build_candlestick_chart(df: pd.DataFrame, trades: list, config: dict,
                            title: str = "Backtest Chart",
                            max_candles: int = 300,
                            position: dict = None) -> "go.Figure":
    """Build plotly candlestick chart with overlays and trade markers."""
    if not PLOTLY_AVAILABLE:
        return None

    plot_df = df.tail(max_candles).copy()

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=plot_df['Datetime'],
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ))

    # EMA overlays
    if 'EMA_Fast' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df['Datetime'], y=plot_df['EMA_Fast'],
            name=f"EMA {config.get('ema_fast', 9)}",
            line=dict(color='#FF9800', width=1.5),
        ))
    if 'EMA_Slow' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df['Datetime'], y=plot_df['EMA_Slow'],
            name=f"EMA {config.get('ema_slow', 21)}",
            line=dict(color='#2196F3', width=1.5),
        ))

    # Bollinger Bands
    strat = config.get('strategy', '')
    if 'BB' in strat or 'Bollinger' in strat or 'Mean Reversion' in strat:
        if 'BB_Upper' in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df['Datetime'], y=plot_df['BB_Upper'],
                name='BB Upper', line=dict(color='#9C27B0', width=1, dash='dot'),
            ))
            fig.add_trace(go.Scatter(
                x=plot_df['Datetime'], y=plot_df['BB_Lower'],
                name='BB Lower', line=dict(color='#9C27B0', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(156,39,176,0.05)',
            ))

    # VWAP
    if 'VWAP' in strat or config.get('strategy', '') == 'VWAP + Volume Spike':
        if 'VWAP' in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df['Datetime'], y=plot_df['VWAP'],
                name='VWAP', line=dict(color='#FFEB3B', width=1.5, dash='dash'),
            ))

    # Ichimoku Cloud
    if 'Ichimoku' in strat:
        for col, color, name in [('Senkou_A', 'rgba(38,166,154,0.2)', 'Senkou A'),
                                  ('Senkou_B', 'rgba(239,83,80,0.2)', 'Senkou B')]:
            if col in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['Datetime'], y=plot_df[col],
                    name=name, line=dict(color=color, width=1),
                    fill='tonexty' if col == 'Senkou_B' else None,
                    fillcolor='rgba(100,200,180,0.08)',
                ))

    # Trade markers on chart
    for trade in trades:
        entry_t = trade.get('entry_time')
        exit_t  = trade.get('exit_time')
        ep      = trade.get('entry_price', 0)
        xp      = trade.get('exit_price',  0)
        ttype   = trade.get('type', 'LONG')
        pnl     = trade.get('pnl', 0)

        if entry_t is not None:
            fig.add_trace(go.Scatter(
                x=[entry_t], y=[ep],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if ttype == 'LONG' else 'triangle-down',
                    size=10,
                    color='#00FF00' if ttype == 'LONG' else '#FF4444',
                ),
                name='Entry',
                text=f"Entry {ttype} @ ₹{ep:.2f}",
                hoverinfo='text',
                showlegend=False,
            ))
        if exit_t is not None:
            fig.add_trace(go.Scatter(
                x=[exit_t], y=[xp],
                mode='markers',
                marker=dict(symbol='x', size=10,
                            color='#FFA500' if pnl > 0 else '#FF0000'),
                name='Exit',
                text=f"Exit @ ₹{xp:.2f} | P&L: ₹{pnl:.2f}",
                hoverinfo='text',
                showlegend=False,
            ))

    # Live position markers
    if position is not None:
        ep = position.get('entry_price', 0)
        sl = position.get('sl_price')
        tp = position.get('target_price')
        if ep:
            fig.add_hline(y=ep, line_dash="dash", line_color="green",
                          annotation_text=f"Entry ₹{ep:.2f}")
        if sl:
            fig.add_hline(y=sl, line_dash="dot", line_color="red",
                          annotation_text=f"SL ₹{sl:.2f}")
        if tp:
            fig.add_hline(y=tp, line_dash="dot", line_color="cyan",
                          annotation_text=f"Target ₹{tp:.2f}")

    height = 500 if max_candles == 300 else 420
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_pnl_chart(trades: list, config: dict) -> "go.Figure":
    """Build cumulative P&L chart."""
    if not PLOTLY_AVAILABLE or not trades:
        return None

    use_net = config.get('include_brokerage', False)
    pnl_key = 'net_pnl' if use_net else 'pnl'

    cum_pnl = []
    running  = 0
    labels   = []
    for i, t in enumerate(trades):
        running += t.get(pnl_key, 0)
        cum_pnl.append(running)
        labels.append(f"Trade {i + 1}")

    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in cum_pnl]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum_pnl) + 1)),
        y=cum_pnl,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#2196F3', width=2),
        marker=dict(color=colors, size=6),
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)

    fig.update_layout(
        title=f"Cumulative {'Net ' if use_net else ''}P&L",
        xaxis_title='Trade Number',
        yaxis_title='P&L (₹)',
        template='plotly_dark',
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
def render_config_ui() -> dict:
    """Render sidebar configuration. Returns config dict."""
    config = {}

    st.sidebar.title("⚙️ Configuration")

    # ── Asset & Data ──
    with st.sidebar.expander("📊 Asset & Timeframe", expanded=True):
        asset_name = st.selectbox("Asset", list(ASSET_MAP.keys()), key='sb_asset')
        if asset_name == "Custom Ticker":
            ticker = st.text_input("Custom Yahoo Ticker", value="AAPL", key='sb_custom_ticker')
        else:
            ticker = ASSET_MAP[asset_name]
        config['asset_name'] = asset_name
        config['ticker']     = ticker

        interval_name = st.selectbox("Interval", list(INTERVAL_MAP.keys()),
                                     index=1, key='sb_interval')
        config['interval'] = INTERVAL_MAP[interval_name]
        config['interval_name'] = interval_name

        period_name = st.selectbox("Historical Period", list(PERIOD_MAP.keys()),
                                   index=4, key='sb_period')
        config['period'] = PERIOD_MAP[period_name]

    # ── Strategy ──
    with st.sidebar.expander("🎯 Strategy", expanded=True):
        strategy = st.selectbox("Strategy", STRATEGIES, key='sb_strategy')
        config['strategy'] = strategy

        if strategy == 'EMA Crossover':
            config['ema_fast'] = st.number_input("Fast EMA", value=9,  min_value=2, key='sb_ema_fast')
            config['ema_slow'] = st.number_input("Slow EMA", value=21, min_value=3, key='sb_ema_slow')
            config['ema_min_angle'] = st.number_input("Min Angle (°)", value=0.0, step=0.1, key='sb_angle')
            filters = st.multiselect("Entry Filters",
                ['Simple Crossover', 'Volume Confirmation', 'RSI Filter'],
                default=['Simple Crossover'], key='sb_ema_filters')
            config['ema_entry_filter'] = filters[0] if filters else 'Simple Crossover'
            if 'RSI Filter' in filters:
                config['use_rsi_filter'] = True
                c1, c2 = st.columns(2)
                config['rsi_filter_min'] = c1.number_input("RSI Min", value=30, key='sb_rsi_min')
                config['rsi_filter_max'] = c2.number_input("RSI Max", value=70, key='sb_rsi_max')
            if 'Volume Confirmation' in filters:
                config['use_rsi_filter'] = False
            config['use_adx_filter'] = st.checkbox("ADX Filter", value=False, key='sb_adx_filt')
            if config.get('use_adx_filter'):
                config['adx_threshold'] = st.number_input("ADX Threshold", value=20, key='sb_adx_thresh')
                config['adx_period']    = st.number_input("ADX Period", value=14, key='sb_adx_per')

        elif strategy == 'Price Crosses Threshold':
            config['threshold_price']    = st.number_input("Price Threshold", value=25000.0, step=10.0, key='sb_thresh')
            config['threshold_cross_dir'] = st.radio("Cross Direction", ['Above', 'Below'], key='sb_cross_dir')
            config['threshold_position']  = st.radio("Position Type", ['LONG', 'SHORT'], key='sb_thresh_pos')

        elif strategy == 'RSI-ADX-EMA Combined':
            config['adx_threshold'] = st.number_input("ADX Threshold", value=25, key='sb_radx')
            config['rsi_period']    = st.number_input("RSI Period", value=14, key='sb_rsi_per')
            config['ema_slow']      = 50

        elif strategy == 'Percentage Change':
            config['pct_threshold'] = st.number_input("% Threshold", value=2.0, step=0.001, format="%.3f", key='sb_pct')
            config['pct_change_dir'] = st.radio("Change Direction", ['Positive', 'Negative'], key='sb_pct_dir')
            config['pct_position']   = st.radio("Position Type", ['LONG', 'SHORT'], key='sb_pct_pos')

        elif strategy == 'Custom Strategy (Multi-Indicator)':
            _render_custom_strategy_ui(config)

        elif strategy == 'SuperTrend AI':
            config['st_atr_period']   = st.number_input("ATR Period", value=10, key='sb_st_atr')
            config['st_multiplier']   = st.number_input("Multiplier", value=3.0, step=0.1, key='sb_st_mult')
            config['adx_threshold']   = st.number_input("ADX Threshold", value=25, key='sb_st_adx')
            config['st_vol_multiplier'] = st.number_input("Volume Multiplier", value=1.5, step=0.1, key='sb_st_vol')

        elif strategy == 'VWAP + Volume Spike':
            config['vwap_vol_multiplier'] = st.number_input("Vol Spike Multiplier", value=2.0, step=0.1, key='sb_vwap_vol')
            config['vwap_max_dist']       = st.number_input("Max VWAP Distance (%)", value=0.3, step=0.1, key='sb_vwap_dist')
            config['vwap_rsi_ob']         = st.number_input("RSI Overbought", value=70, key='sb_vwap_ob')
            config['vwap_rsi_os']         = st.number_input("RSI Oversold", value=30, key='sb_vwap_os')

        elif strategy == 'Bollinger Squeeze Breakout':
            config['bb_period']          = st.number_input("BB Period", value=20, key='sb_bbs_per')
            config['bb_std_dev']         = st.number_input("BB Std Dev", value=2.0, step=0.1, key='sb_bbs_std')
            config['bb_squeeze_thresh']  = st.number_input("Squeeze Threshold", value=0.02, step=0.001, format="%.3f", key='sb_bbs_sq')
            config['bb_vol_multiplier']  = st.number_input("Breakout Vol Multiplier", value=1.8, step=0.1, key='sb_bbs_vol')

        elif strategy == 'Elliott Waves + Ratio Charts':
            config['wave_lookback']     = st.number_input("Wave Lookback", value=13, key='sb_ew_lb')
            config['ratio_ema_period']  = st.number_input("Ratio EMA Period", value=21, key='sb_ew_rep')
            config['ratio_threshold']   = st.number_input("Ratio Threshold", value=1.0, step=0.01, key='sb_ew_rt')

        elif strategy == 'Morning Breakout (ORB)':
            config['orb_vol_multiplier'] = st.number_input("Volume Multiplier", value=1.5, step=0.1, key='sb_orb_vol')
            st.info("ℹ️ Breaks Opening Range (first 15 min) with volume surge")

        elif strategy == 'MACD Divergence':
            config['macd_div_lookback'] = st.number_input("Lookback Period", value=20, key='sb_md_lb')
            config['macd_fast']  = st.number_input("MACD Fast", value=12, key='sb_mf')
            config['macd_slow']  = st.number_input("MACD Slow", value=26, key='sb_ms')
            config['macd_signal']= st.number_input("MACD Signal", value=9, key='sb_msig')

        elif strategy == 'Triple EMA Momentum':
            st.info("ℹ️ Uses EMA 9, 20, 50 alignment with pullback entry")
            config['adx_threshold'] = st.number_input("Min ADX for Signal", value=20, key='sb_tem_adx')

        elif strategy == 'Mean Reversion (Bollinger + RSI)':
            config['bb_period']   = st.number_input("BB Period", value=20, key='sb_mr_bbp')
            config['bb_std_dev']  = st.number_input("BB Std Dev", value=2.0, step=0.1, key='sb_mr_bbs')
            config['mr_rsi_os']   = st.number_input("RSI Oversold Level", value=30, key='sb_mr_os')
            config['mr_rsi_ob']   = st.number_input("RSI Overbought Level", value=70, key='sb_mr_ob')

        elif strategy == 'Ichimoku Cloud Breakout':
            st.info("ℹ️ Tenkan(9), Kijun(26), Senkou B(52) - cloud breakout entry")
            config['adx_threshold'] = st.number_input("Min ADX", value=20, key='sb_ich_adx')

        # Defaults for missing params
        config.setdefault('ema_fast', 9)
        config.setdefault('ema_slow', 21)
        config.setdefault('bb_period', 20)
        config.setdefault('bb_std_dev', 2.0)
        config.setdefault('adx_threshold', 25)
        config.setdefault('adx_period', 14)
        config.setdefault('rsi_period', 14)
        config.setdefault('macd_fast', 12)
        config.setdefault('macd_slow', 26)
        config.setdefault('macd_signal', 9)
        config.setdefault('st_atr_period', 10)
        config.setdefault('st_multiplier', 3.0)

    # ── SL Configuration ──
    with st.sidebar.expander("🛑 Stop Loss", expanded=True):
        sl_type = st.selectbox("SL Type", SL_TYPES, key='sb_sl_type')
        config['sl_type'] = sl_type

        if sl_type in ['Custom Points', 'Trailing SL (Points)', 'Trailing SL + Current Candle',
                       'Trailing SL + Previous Candle', 'Trailing SL + Current Swing',
                       'Trailing SL + Previous Swing', 'Break-even After 50% Target',
                       'Cost-to-Cost + N Points Trailing SL']:
            config['sl_points'] = st.number_input("SL Points", value=10.0, step=0.5, min_value=0.1, key='sb_sl_pts')

        if sl_type in ['P&L Based (Rupees)', 'Trailing Profit (Rupees)', 'Trailing Loss (Rupees)']:
            config['sl_rupees'] = st.number_input("SL Rupees", value=500.0, step=50.0, key='sb_sl_rs')

        if sl_type == 'ATR-based':
            config['sl_atr_mult'] = st.number_input("ATR Multiplier", value=1.5, step=0.1, key='sb_sl_atr')

        if sl_type == 'Cost-to-Cost + N Points Trailing SL':
            c1, c2 = st.columns(2)
            config['c2c_trigger_k'] = c1.number_input("Trigger K pts", value=3.0, step=0.5, key='sb_c2c_k')
            config['c2c_offset_n']  = c2.number_input("Offset N pts", value=2.0, step=0.5, key='sb_c2c_n')

    # ── Target Configuration ──
    with st.sidebar.expander("🎯 Target", expanded=True):
        target_type = st.selectbox("Target Type", TARGET_TYPES, key='sb_tgt_type')
        config['target_type'] = target_type

        if target_type in ['Custom Points', 'Trailing Target (Points)',
                           'Dynamic Trailing SL+Target (Lock Profits)', '50% Exit at Target (Partial)']:
            config['target_points'] = st.number_input("Target Points", value=20.0, step=0.5, min_value=0.1, key='sb_tgt_pts')

        if target_type == 'P&L Based (Rupees)':
            config['target_rupees'] = st.number_input("Target Rupees", value=1000.0, step=100.0, key='sb_tgt_rs')

        if target_type == 'ATR-based':
            config['target_atr_mult'] = st.number_input("ATR Multiplier", value=2.0, step=0.1, key='sb_tgt_atr')

        if target_type == 'Risk-Reward Based':
            config['target_rr_ratio'] = st.number_input("R:R Ratio", value=2.0, step=0.1, min_value=0.1, key='sb_rr')

        if target_type == 'Signal-based (Reverse Crossover)':
            st.info("ℹ️ Exits only on reverse signal. No price target is set.")

    # ── Trade Configuration ──
    with st.sidebar.expander("📋 Trade Settings"):
        config['quantity'] = st.number_input("Quantity (Lots/Shares)", value=1, min_value=1, key='sb_qty')
        config['direction_filter'] = st.radio("Trade Direction",
            ['Both', 'LONG Only', 'SHORT Only'], key='sb_dir_filter')

        config['use_trade_window'] = st.checkbox("⏰ Enable Trade Window", key='sb_tw')
        if config['use_trade_window']:
            c1, c2 = st.columns(2)
            ws = c1.time_input("Start Time (IST)", value=dtime(9, 30), key='sb_tw_start')
            we = c2.time_input("End Time (IST)",   value=dtime(15, 0), key='sb_tw_end')
            config['window_start'] = ws
            config['window_end']   = we

        config['prevent_overlap'] = st.checkbox("🚫 Prevent Overlapping Trades",
                                                 value=True, key='sb_overlap')

    # ── Backtesting Options ──
    with st.sidebar.expander("🔬 Backtesting Options"):
        config['use_method2'] = st.checkbox(
            "🔬 Method 2 (Realistic Entry – Next Open)", key='sb_m2',
            help="Entry at next candle open. More realistic, avoids look-ahead bias.")
        config['filter_same_day'] = st.checkbox(
            "📅 Filter Same-Day Trades Only", key='sb_same_day',
            help="Show only intraday trades (9:15 AM – 3:00 PM)")

    # ── Brokerage ──
    with st.sidebar.expander("💰 Brokerage & Charges"):
        config['include_brokerage'] = st.checkbox("Include Brokerage", key='sb_brok')
        if config['include_brokerage']:
            config['brokerage_calc_type'] = st.radio("Calculation", ['Fixed', 'Percentage'], key='sb_brok_type')
            if config['brokerage_calc_type'] == 'Fixed':
                config['brokerage_fixed'] = st.number_input("Brokerage per Trade (₹)", value=20.0, key='sb_brok_fixed')
            else:
                config['brokerage_pct'] = st.number_input("Turnover % (e.g. 0.03)", value=0.03, step=0.001, format="%.3f", key='sb_brok_pct')

    # ── Broker (Dhan) ──
    with st.sidebar.expander("🏦 Dhan Broker"):
        config['enable_broker'] = st.checkbox("Enable Dhan Broker", key='sb_dhan')
        if config['enable_broker']:
            config['dhan_client_id']    = st.text_input("Client ID", key='sb_dhan_cid')
            config['dhan_access_token'] = st.text_input("Access Token", type='password', key='sb_dhan_tok')
            config['asset_type']  = st.selectbox("Asset Type", ['Stocks', 'Options', 'Stock Options'], key='sb_atype')
            config['security_id'] = st.text_input("Security ID", value='', key='sb_secid')
            config['exchange']    = st.selectbox("Exchange", ['NSE', 'BSE'], key='sb_exch')
            config['trading_type']= st.selectbox("Trading Type", ['Intraday', 'Delivery (CNC)'], key='sb_ttype')
            config['order_type']  = st.selectbox("Order Type", ['Market Order', 'Limit Order'], key='sb_otype')

            config['use_bracket'] = st.checkbox("🎯 Bracket Order (SL+Target)", key='sb_bo')
            if config['use_bracket']:
                c1, c2, c3 = st.columns(3)
                config['broker_sl_points']     = c1.number_input("SL Pts", value=50, key='sb_bo_sl')
                config['broker_target_points'] = c2.number_input("Tgt Pts", value=100, key='sb_bo_tgt')
                config['broker_trail_sl']      = c3.number_input("Trail SL", value=0, key='sb_bo_trail')

            config['clear_positions_before_entry'] = st.checkbox("🧹 Clear Positions Before Entry", key='sb_clear')

            # Multi-account
            st.markdown("**🔀 Multi-Account Trading**")
            if 'multi_accounts' not in st.session_state:
                st.session_state['multi_accounts'] = []
            accts = st.session_state['multi_accounts']
            if accts:
                st.write(f"Configured Accounts: {len(accts)}")
                for i, a in enumerate(accts):
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"{i+1}. Client: {str(a['client_id'])[:8]}...")
                    if col2.button("❌", key=f'del_acc_{i}'):
                        accts.pop(i)
                        st.rerun()
            with st.expander("➕ Add Account"):
                new_cid = st.text_input("Client ID", key='sb_new_cid')
                new_tok = st.text_input("Access Token", type='password', key='sb_new_tok')
                if st.button("Add Account", key='sb_add_acc'):
                    if new_cid and new_tok:
                        accts.append({'client_id': new_cid, 'access_token': new_tok})
                        st.success("Account added!")
            config['multi_accounts'] = accts

            # Multi-strike
            if config.get('asset_type') == 'Options':
                config['use_multi_strike'] = st.checkbox("Enable Multi-Strike Orders", key='sb_ms')
                if config.get('use_multi_strike'):
                    config['option_side'] = st.radio("Option Side", ['CE', 'PE'], key='sb_os')
                    for side in ['CE', 'PE']:
                        key = f'multi_strikes_{side.lower()}'
                        if key not in st.session_state:
                            st.session_state[key] = []
                        strikes = st.session_state[key]
                        if strikes:
                            st.write(f"{side} Strikes: {len(strikes)}")
                            for i, s in enumerate(strikes):
                                c1, c2 = st.columns([4, 1])
                                c1.write(f"{i+1}. {s}")
                                if c2.button("❌", key=f'del_{side}_{i}'):
                                    strikes.pop(i)
                                    st.rerun()
                        with st.expander(f"➕ Add {side} Strike"):
                            new_sid = st.text_input(f"{side} Security ID", key=f'new_{side}_sid')
                            if st.button(f"Add {side}", key=f'add_{side}'):
                                if new_sid:
                                    strikes.append(new_sid)
                                    st.success(f"{side} strike added!")
                        config[key] = strikes

    return config


def _render_custom_strategy_ui(config: dict):
    """Render dynamic condition builder for Custom Strategy."""
    if 'custom_indicator_conditions' not in st.session_state:
        st.session_state['custom_indicator_conditions'] = []
    conds = st.session_state['custom_indicator_conditions']

    config['custom_logic'] = st.radio("Logic", ['AND', 'OR'], horizontal=True, key='sb_logic')

    col1, col2 = st.columns(2)
    if col1.button("➕ Add Condition", key='sb_add_cond'):
        conds.append({
            'type': 'Indicator Crosses Level',
            'indicator': 'RSI', 'level': 50, 'cross_dir': 'Above', 'position': 'LONG'
        })
        st.rerun()
    if col2.button("🗑️ Clear All", key='sb_clear_cond'):
        st.session_state['custom_indicator_conditions'] = []
        st.rerun()

    for i, cond in enumerate(conds):
        with st.expander(f"Condition #{i+1} – {cond.get('type', '')}", expanded=True):
            ctype = st.selectbox("Type", ['Price Crosses Indicator', 'Indicator Crosses Level',
                                          'Indicator Crossover'],
                                 key=f'ctype_{i}',
                                 index=['Price Crosses Indicator', 'Indicator Crosses Level',
                                        'Indicator Crossover'].index(cond.get('type', 'Indicator Crosses Level')))
            cond['type'] = ctype

            if ctype == 'Price Crosses Indicator':
                cond['indicator']  = st.selectbox("Indicator", ['EMA', 'SMA', 'BB Upper', 'BB Lower', 'BB Mid'], key=f'ind_{i}')
                cond['period']     = st.number_input("Period", value=20, key=f'per_{i}')
                cond['cross_type'] = st.radio("Cross", ['Above', 'Below'], key=f'ct_{i}')

            elif ctype == 'Indicator Crosses Level':
                cond['indicator'] = st.selectbox("Indicator",
                    ['RSI', 'MACD', 'MACD Histogram', 'ADX', 'Volume', 'BB %B', 'ATR', 'Historical Volatility', 'Std Dev'],
                    key=f'ind_{i}')
                cond['level']     = st.number_input("Level", value=50.0, step=0.5, key=f'lev_{i}')
                cond['cross_dir'] = st.radio("Cross Direction", ['Above', 'Below'], key=f'cd_{i}')

            elif ctype == 'Indicator Crossover':
                cond['cross_kind'] = st.selectbox("Crossover Type",
                    ['Fast EMA x Slow EMA', 'Fast SMA x Slow SMA', 'Price x EMA', 'RSI Crossover'],
                    key=f'ck_{i}')
                cond['direction'] = st.radio("Direction", ['Bullish', 'Bearish'], key=f'dir_{i}')

            cond['position'] = st.radio("Position", ['LONG', 'SHORT'], key=f'pos_{i}')

            if st.button(f"🗑️ Delete #{i+1}", key=f'del_cond_{i}'):
                conds.pop(i)
                st.rerun()

    config['custom_conditions'] = conds
    st.session_state['custom_indicator_conditions'] = conds

# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST UI TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_backtest_ui(config: dict):
    """Render the Backtest tab."""
    st.subheader("📈 Backtesting")
    method_label = "Method 2 – Realistic (Next Open)" if config.get('use_method2') else "Method 1 – Signal Close"
    st.caption(f"Entry: **{method_label}** | Strategy: **{config.get('strategy')}** | Asset: **{config.get('asset_name')}**")

    if st.button("▶️ Run Backtest", type='primary', use_container_width=True, key='run_bt'):
        with st.spinner("Fetching data & running backtest..."):
            df_raw = fetch_data(config['ticker'], config['interval'], config['period'])
            if df_raw.empty:
                st.error("❌ No data fetched. Check ticker/interval/period.")
                return

            df = calculate_all_indicators(df_raw, config)
            results = run_backtest(df, config)

        st.session_state['backtest_results'] = results
        st.success(f"✅ Backtest complete! {results['metrics'].get('total_trades', 0)} trades found.")

    if 'backtest_results' not in st.session_state:
        st.info("👆 Configure your strategy in the sidebar and click **Run Backtest**")
        return

    results = st.session_state['backtest_results']
    trades  = results.get('trades', [])
    metrics = results.get('metrics', {})
    df      = results.get('df', pd.DataFrame())

    # ── Filter Same-Day Trades ──
    display_trades = trades
    if config.get('filter_same_day') and trades:
        display_trades = [t for t in trades if is_same_day_trade(t.get('entry_time'), t.get('exit_time'))]
        if display_trades:
            metrics = _calculate_metrics(display_trades, config)
        st.info(f"📅 Filtered to {len(display_trades)} same-day trades (from {len(trades)} total)")

    if not display_trades:
        st.warning("⚠️ No trades found. Try adjusting strategy parameters or period.")
        _render_debug_info(results.get('debug', {}))
        return

    # ── Metrics Row 1 ──
    st.markdown("### 📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades",  metrics.get('total_trades', 0))
    c2.metric("Win Rate",      f"{metrics.get('win_rate', 0):.1f}%")
    c3.metric("Total P&L",     f"₹{metrics.get('total_pnl', 0):,.2f}",
              delta=f"{'▲' if metrics.get('total_pnl', 0) >= 0 else '▼'}")
    c4.metric("Avg Trade P&L", f"₹{metrics.get('avg_pnl', 0):,.2f}")

    # ── Metrics Row 2 (Brokerage) ──
    if config.get('include_brokerage'):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Brokerage",  f"₹{metrics.get('total_brokerage', 0):,.2f}")
        c2.metric("Net P&L",           f"₹{metrics.get('total_net_pnl', 0):,.2f}",
                  delta=f"{'▲' if metrics.get('total_net_pnl', 0) >= 0 else '▼'}")
        c3.metric("Avg Net P&L",       f"₹{metrics.get('avg_net_pnl', 0):,.2f}")
        c4.metric("Profit Factor",     f"{metrics.get('profit_factor', 0):.2f}")

    # ── Metrics Row 3 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winning Trades",  metrics.get('winning_trades', 0))
    c2.metric("Losing Trades",   metrics.get('losing_trades', 0))
    c3.metric("Max Drawdown",    f"₹{metrics.get('max_drawdown', 0):,.2f}")
    c4.metric("Avg Win",         f"₹{metrics.get('avg_win', 0):,.2f}")

    # ── Charts ──
    st.markdown("### 📉 Price Chart with Trades")
    if PLOTLY_AVAILABLE and not df.empty:
        chart = build_candlestick_chart(df, display_trades, config,
                                        title=f"Backtest: {config.get('asset_name')} | {config.get('interval_name')}")
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        pnl_chart = build_pnl_chart(display_trades, config)
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)

    # ── Trade Table ──
    st.markdown("### 📋 Trade Records")
    use_net = config.get('include_brokerage', False)
    trade_rows = []
    for t in display_trades:
        et = t.get('entry_time')
        xt = t.get('exit_time')
        row = {
            'Entry Time':    str(et)[:19] if et else '',
            'Exit Time':     str(xt)[:19] if xt else '',
            'Type':          t.get('type', ''),
            'Duration (min)':t.get('duration_min', 0),
            'Entry ₹':       f"{t.get('entry_price', 0):.2f}",
            'Exit ₹':        f"{t.get('exit_price', 0):.2f}",
            'High ₹':        f"{t.get('highest_price', 0):.2f}",
            'Low ₹':         f"{t.get('lowest_price', 0):.2f}",
            'SL ₹':          f"{t.get('sl_price', 0):.2f}" if t.get('sl_price') else 'Signal',
            'Target ₹':      f"{t.get('target_price', 0):.2f}" if t.get('target_price') else 'Signal',
            'P&L ₹':         f"{t.get('pnl', 0):.2f}",
            'Reason':        t.get('exit_reason', ''),
        }
        if use_net:
            row['Brokerage ₹'] = f"{t.get('brokerage', 0):.2f}"
            row['Net P&L ₹']   = f"{t.get('net_pnl', 0):.2f}"
        trade_rows.append(row)

    if trade_rows:
        tdf = pd.DataFrame(trade_rows)
        # Color positive/negative P&L
        def color_pnl(val):
            try:
                v = float(str(val).replace(',', ''))
                return 'color: #26a69a' if v > 0 else ('color: #ef5350' if v < 0 else '')
            except:
                return ''
        pnl_col = 'Net P&L ₹' if use_net else 'P&L ₹'
        styled = tdf.style.applymap(color_pnl, subset=[pnl_col])
        st.dataframe(styled, use_container_width=True, height=400)

        # Download
        csv = tdf.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "backtest_trades.csv", "text/csv")

    # ── Skipped Trades ──
    skipped = results.get('skipped_trades', [])
    if skipped:
        with st.expander(f"⚠️ Skipped/Overlapping Trades ({len(skipped)}) – Not in P&L"):
            sk_rows = []
            for s in skipped:
                sk_rows.append({
                    'Type':    s.get('type', ''),
                    'Entry ₹': f"{s.get('entry_price', 0):.2f}",
                    'Exit ₹':  f"{s.get('exit_price', 0):.2f}",
                    'P&L ₹':   f"{s.get('pnl', 0):.2f}",
                    'Reason':  s.get('exit_reason', ''),
                    'Note':    s.get('note', ''),
                })
            st.dataframe(pd.DataFrame(sk_rows), use_container_width=True)
            skip_win = sum(1 for s in skipped if s.get('pnl', 0) > 0)
            total_skip_pnl = sum(s.get('pnl', 0) for s in skipped)
            c1, c2, c3 = st.columns(3)
            c1.metric("Skipped Winning",  skip_win)
            c2.metric("Skipped Losing",   len(skipped) - skip_win)
            c3.metric("Total Skipped P&L", f"₹{total_skip_pnl:.2f}")

    # ── Debug Info ──
    _render_debug_info(results.get('debug', {}))


def _render_debug_info(debug: dict):
    """Render debug information expander."""
    if not debug:
        return
    with st.expander("🔍 Debug Information"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candles Analyzed",      debug.get('candles_analyzed', 0))
        c2.metric("Total Signals",         debug.get('total_signals', 0))
        c3.metric("Signals Skipped",       debug.get('signals_skipped', 0))
        c4.metric("Overlapping Tracked",   debug.get('overlapping_trades_tracked', 0))


# ──────────────────────────────────────────────────────────────────────────────
# LIVE TRADING UI TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_live_trading_ui(config: dict):
    """Render the Live Trading tab."""
    st.subheader("🔴 Live Trading")
    st.caption(f"Strategy: **{config.get('strategy')}** | Asset: **{config.get('asset_name')}** | Interval: **{config.get('interval_name')}**")

    is_trading = st.session_state.get('trading_active', False)

    # ── Control Buttons ──
    c1, c2, c3, c4 = st.columns(4)

    if not is_trading:
        if c1.button("▶️ Start Trading", type='primary', use_container_width=True, key='btn_start'):
            # Init broker
            if config.get('enable_broker') and config.get('dhan_client_id'):
                broker = DhanBrokerIntegration(config['dhan_client_id'], config['dhan_access_token'])
                st.session_state['dhan_broker'] = broker
                add_log(f"{'✅ Broker connected' if broker.connected else '⚠️ Broker not connected (simulation mode)'}")
            else:
                st.session_state['dhan_broker'] = None

            # Reset session
            st.session_state['trading_active'] = True
            st.session_state['position']       = None
            st.session_state['broker_position'] = None
            st.session_state['live_logs']      = []
            st.session_state['trade_history']  = []
            st.session_state['current_data']   = None
            st.session_state['config']         = config
            add_log(f"🚀 Live trading started | {config.get('strategy')} | {config.get('asset_name')}")
            st.rerun()
    else:
        if c1.button("⏹️ Stop Trading", use_container_width=True, key='btn_stop'):
            st.session_state['trading_active'] = False
            add_log("⏹️ Trading stopped by user")
            st.rerun()

    position = st.session_state.get('position')
    if c2.button("🔄 Close Position", use_container_width=True,
                 disabled=(not is_trading or position is None), key='btn_close'):
        if position:
            df_live = st.session_state.get('current_data')
            if df_live is not None and not df_live.empty:
                manual_price = df_live.iloc[-1]['Close']
                broker = st.session_state.get('dhan_broker')
                _live_exit(position, manual_price, 'Manual Close', broker, config, df_live, len(df_live) - 1)
            st.rerun()

    if c3.button("🔍 Price Check", use_container_width=True, key='btn_price'):
        df_chk = fetch_data(config['ticker'], config['interval'], '1d')
        if not df_chk.empty:
            p = df_chk.iloc[-1]['Close']
            add_log(f"💲 Current price: ₹{p:.2f}")
        st.rerun()

    if c4.button("🔄 Refresh", use_container_width=True, key='btn_refresh'):
        st.rerun()

    # ── Position Display ──
    if position is not None:
        pos_type = position['type']
        ep       = position['entry_price']
        sl       = position.get('sl_price')
        tp       = position.get('target_price')
        et       = position.get('entry_time')
        df_live  = st.session_state.get('current_data')
        curr_p   = df_live.iloc[-1]['Close'] if df_live is not None and not df_live.empty else ep
        pnl_live = (curr_p - ep) * (1 if pos_type == 'LONG' else -1) * config.get('quantity', 1)

        color = "🟢" if pos_type == 'LONG' else "🔴"
        bg    = "#1a3a1a" if pos_type == 'LONG' else "#3a1a1a"

        st.markdown(f"""
        <div style="background:{bg}; padding:16px; border-radius:12px; border-left:4px solid {'#26a69a' if pos_type == 'LONG' else '#ef5350'}; margin:10px 0">
            <h4>{color} {pos_type} Position Active</h4>
            <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:10px; margin-top:8px">
                <div><b>Entry</b><br>₹{ep:,.2f}</div>
                <div><b>Current</b><br>₹{curr_p:,.2f} (<span style="color:{'#26a69a' if pnl_live >= 0 else '#ef5350'}">₹{pnl_live:+.2f}</span>)</div>
                <div><b>SL</b><br>₹{sl:,.2f if sl else 'Signal'}</div>
                <div><b>Target</b><br>{'₹' + f'{tp:,.2f}' if tp else 'Signal'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💤 No active position" if is_trading else "⏸️ Trading not started")

    # ── Live Chart ──
    if is_trading:
        df_live = st.session_state.get('current_data')
        if df_live is None or df_live.empty:
            # Fetch data now for chart
            df_live_raw = fetch_data(config['ticker'], config['interval'], '1d')
            if not df_live_raw.empty:
                df_live = calculate_all_indicators(df_live_raw, config)
                st.session_state['current_data'] = df_live

        if df_live is not None and not df_live.empty:
            chart = build_candlestick_chart(
                df_live, [], config,
                title=f"Live: {config.get('asset_name')} | {config.get('interval_name')}",
                max_candles=150,
                position=position,
            )
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    # ── Auto-Refresh & Iteration ──
    if is_trading:
        live_config = st.session_state.get('config', config)
        with st.spinner("🔄 Running iteration..."):
            live_trading_iteration(live_config)

        # Auto-refresh using st.rerun after delay
        time.sleep(1.5)
        st.rerun()

    # ── Logs ──
    logs = st.session_state.get('live_logs', [])
    st.markdown("### 📝 Trading Logs")
    log_text = "\n".join(logs[:50]) if logs else "No logs yet..."
    st.text_area("", value=log_text, height=200, key='log_display', disabled=True)


# ──────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY UI TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_trade_history_ui(config: dict):
    """Render the Trade History tab."""
    st.subheader("📊 Trade History (Live Session)")

    history = st.session_state.get('trade_history', [])
    if not history:
        st.info("No completed trades in this session yet. Start live trading to see trade history.")
        return

    # Metrics
    use_net = config.get('include_brokerage', False)
    pnl_key = 'net_pnl' if use_net else 'pnl'

    total_trades = len(history)
    total_pnl    = sum(t.get(pnl_key, 0) for t in history)
    winners      = [t for t in history if t.get(pnl_key, 0) > 0]
    win_rate     = len(winners) / total_trades * 100 if total_trades > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades",  total_trades)
    c2.metric("Win Rate",      f"{win_rate:.1f}%")
    c3.metric("Total P&L",     f"₹{total_pnl:,.2f}",
              delta=f"{'▲' if total_pnl >= 0 else '▼'}")

    # Table
    rows = []
    for t in history:
        rows.append({
            'Entry Time':   str(t.get('entry_time', ''))[:19],
            'Exit Time':    str(t.get('exit_time', ''))[:19],
            'Type':         t.get('type', ''),
            'Entry ₹':      f"{t.get('entry_price', 0):.2f}",
            'Exit ₹':       f"{t.get('exit_price', 0):.2f}",
            'P&L ₹':        f"{t.get('pnl', 0):.2f}",
            'Net P&L ₹':    f"{t.get('net_pnl', 0):.2f}",
            'Reason':       t.get('exit_reason', ''),
        })
    hdf = pd.DataFrame(rows)
    st.dataframe(hdf, use_container_width=True, height=400)

    csv = hdf.to_csv(index=False)
    st.download_button("⬇️ Download Trade History CSV", csv, "live_trade_history.csv", "text/csv")

    # P&L chart
    if PLOTLY_AVAILABLE and history:
        pnl_chart = build_pnl_chart(history, config)
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="AlgoTrader Pro v2",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    /* Dark theme enhancements */
    .stMetric { background: #1e1e2e; padding: 12px; border-radius: 8px; border-left: 3px solid #2196F3; }
    .stButton > button { border-radius: 8px; }
    .block-container { padding-top: 1rem; }
    div[data-testid="stExpander"] { border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
    /* Metric colors */
    [data-testid="metric-container"] { background: #0e1117; border-radius: 10px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
                padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
                border: 1px solid #2196F3;">
        <h1 style="margin:0; color:#fff">📈 AlgoTrader Pro v2.0</h1>
        <p style="margin:4px 0 0 0; color:#90caf9">
            17 Strategies | 18 SL Types | 13 Target Types | Dhan Integration | Backtest ↔ Live Consistency
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check dependencies
    if not YFINANCE_AVAILABLE:
        st.error("❌ yfinance not installed. Run: `pip install yfinance`")
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ plotly not installed. Charts disabled. Run: `pip install plotly`")
    if not DHAN_AVAILABLE:
        st.info("ℹ️ dhanhq not installed. Broker features disabled. Run: `pip install dhanhq`")

    # Render sidebar config
    config = render_config_ui()

    # Store config in session
    if not st.session_state.get('trading_active', False):
        st.session_state['config'] = config

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Backtest", "🔴 Live Trading", "📊 Trade History"])

    with tab1:
        render_backtest_ui(config)

    with tab2:
        render_live_trading_ui(config)

    with tab3:
        render_trade_history_ui(config)

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>AlgoTrader Pro v2.0 | 17 Strategies | Backtesting Method 1 & 2 | "
        "Dhan Broker Integration | Built with Streamlit</small></center>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
