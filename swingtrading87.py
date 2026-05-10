import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import random
import itertools
import io

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from dhanhq import dhanhq as DhanHQ
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False

st.set_page_config(page_title="Swing Trading Recommender", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IST = 'Asia/Kolkata'

INSTRUMENTS = {
    # Indian Indices
    'Nifty 50':          '^NSEI',
    'Bank Nifty':        '^NSEBANK',
    'Sensex':            '^BSESN',
    'Nifty IT':          '^CNXIT',
    'Nifty Auto':        '^CNXAUTO',
    'Nifty Pharma':      '^CNXPHARMA',
    'Nifty FMCG':        '^CNXFMCG',
    'Nifty Metal':       '^CNXMETAL',
    'Nifty Energy':      '^CNXENERGY',
    'Nifty Midcap 50':   '^NSEMDCP50',
    # Crypto
    'BTC/USD':           'BTC-USD',
    'ETH/USD':           'ETH-USD',
    'XRP/USD':           'XRP-USD',
    'BNB/USD':           'BNB-USD',
    'SOL/USD':           'SOL-USD',
    # Forex
    'USD/INR':           'USDINR=X',
    'EUR/INR':           'EURINR=X',
    'EUR/USD':           'EURUSD=X',
    'GBP/USD':           'GBPUSD=X',
    'USD/JPY':           'JPY=X',
    'AUD/USD':           'AUDUSD=X',
    'USD/CHF':           'CHFUSD=X',
    # Commodities
    'Gold (COMEX)':      'GC=F',
    'Silver (COMEX)':    'SI=F',
    'Crude Oil WTI':     'CL=F',
    'Brent Crude':       'BZ=F',
    'Natural Gas':       'NG=F',
    'Copper':            'HG=F',
    # Custom
    '── Custom Ticker ──': '__custom__',
}

TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk']
PERIODS    = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y']

# max days yfinance supports per intraday interval
_INTERVAL_MAX_DAYS = {
    '1m': 7, '5m': 60, '15m': 60, '30m': 60, '1h': 730, '4h': 730
}
_PERIOD_DAYS = {
    '1d': 1, '5d': 5, '7d': 7, '1mo': 30, '3mo': 90,
    '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, '20y': 7300
}

# ─────────────────────────────────────────────────────────────────────────────
# yFinance Fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance_data(symbol, interval, period, rate_limit_sleep=1.5):
    """
    Fetch OHLCV data from yfinance.
    - Date column is always IST (Asia/Kolkata) timezone-aware.
    - 4h interval is synthesised from 1h data via resample.
    - period/interval incompatibility is handled gracefully (period is capped).
    - rate_limit_sleep (default 1.5 s) is respected before every request.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    days = _PERIOD_DAYS.get(period, 365)

    # Enforce interval/period compatibility — cap days silently
    if interval in _INTERVAL_MAX_DAYS:
        max_d = _INTERVAL_MAX_DAYS[interval]
        if days > max_d:
            days = max_d

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    # Rate-limit guard — always wait before calling yfinance
    time.sleep(rate_limit_sleep)

    fetch_interval = '1h' if interval == '4h' else interval

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_dt.strftime('%Y-%m-%d'),
            end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval=fetch_interval,
            auto_adjust=True,
            back_adjust=False,
        )
    except Exception as exc:
        raise ValueError(f"yfinance error for {symbol!r}: {exc}")

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{symbol}' (interval={interval}, period={period}). "
            "Verify the symbol or try a shorter period."
        )

    df = df.reset_index()

    # Rename the date/datetime index column → 'Date'
    for candidate in ('Datetime', 'Date', 'index'):
        if candidate in df.columns:
            df = df.rename(columns={candidate: 'Date'})
            break

    # Ensure IST timezone
    if df['Date'].dt.tz is None:
        df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert(IST)
    else:
        df['Date'] = df['Date'].dt.tz_convert(IST)

    # Resample 1h → 4h in IST
    if interval == '4h':
        df = df.set_index('Date')
        df = (
            df.resample('4h', label='left', closed='left')
            .agg(Open=('Open', 'first'), High=('High', 'max'),
                 Low=('Low', 'min'),   Close=('Close', 'last'),
                 Volume=('Volume', 'sum'))
            .dropna(subset=['Open', 'Close'])
            .reset_index()
        )

    # Standardise column names (handle mixed case from yfinance)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if   cl == 'open':   col_map[c] = 'Open'
        elif cl == 'high':   col_map[c] = 'High'
        elif cl == 'low':    col_map[c] = 'Low'
        elif cl == 'close':  col_map[c] = 'Close'
        elif cl == 'volume': col_map[c] = 'Volume'
    df = df.rename(columns=col_map)

    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def infer_columns(df):
    """Robust inference of common OHLCV and date column names using tokenized matching.
    Returns mapping for keys: date, open, high, low, close, volume (original column names or None).
    """
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}

    for col in df.columns:
        low = str(col).lower()
        tokens = re.findall(r"[a-z]+", low)
        token_set = set(tokens)

        if mapping['date'] is None and any(t in token_set for t in ("date", "time", "timestamp", "datetime", "trade", "trade_date")):
            mapping['date'] = col
            continue
        if mapping['open'] is None and ("open" in token_set or low.strip() in ("op", "openprice")):
            mapping['open'] = col
            continue
        if mapping['high'] is None and ("high" in token_set or low.strip() in ("h", "hi", "highprice")):
            mapping['high'] = col
            continue
        if mapping['low'] is None and ("low" in token_set or low.strip() in ("l", "lo", "lowprice")):
            mapping['low'] = col
            continue
        if mapping['close'] is None and ("close" in token_set or ("adj" in token_set and "close" in token_set)
                                          or low.strip() in ("c", "cl", "last", "price", "closeprice")):
            mapping['close'] = col
            continue
        if mapping['volume'] is None and any(t in token_set for t in ("volume", "vol", "qty", "quantity", "tradevolume")):
            mapping['volume'] = col
            continue

    # fallback: detect date-like columns by parsing
    if mapping['date'] is None:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.6:
                    mapping['date'] = col
                    break
            except Exception:
                continue

    return mapping


def standardize_df(df):
    """Map columns, parse dates, sort ascending.
    Returns (dataframe, mapping) with Date always tz-aware in IST.
    """
    mapping = infer_columns(df)
    if mapping['date'] is None:
        raise ValueError("Could not infer a date column. Please include a date/time column.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df[mapping['date']], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)

    try:
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize(IST)
        else:
            df['Date'] = df['Date'].dt.tz_convert(IST)
    except Exception:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(IST)

    renames = {}
    for std in ['open', 'high', 'low', 'close', 'volume']:
        if mapping.get(std) is not None:
            renames[mapping[std]] = std.capitalize()
    df = df.rename(columns=renames)

    if 'Close' not in df.columns:
        possible_price_cols = [c for c in df.columns if re.search(r"price", str(c).lower())]
        if possible_price_cols:
            df = df.rename(columns={possible_price_cols[0]: 'Close'})

    if 'Close' not in df.columns:
        candidates = [c for c in df.columns if c != 'Date']
        if candidates:
            df['Close'] = pd.to_numeric(df[candidates[0]], errors='coerce')
        else:
            raise ValueError('No price/close column could be found or inferred.')

    if 'Open'   not in df.columns: df['Open']   = df['Close']
    if 'High'   not in df.columns: df['High']   = df[['Open', 'Close']].max(axis=1)
    if 'Low'    not in df.columns: df['Low']    = df[['Open', 'Close']].min(axis=1)
    if 'Volume' not in df.columns: df['Volume'] = np.nan

    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values('Date').reset_index(drop=True)
    return df, mapping


# ─────────────────────────────────────────────────────────────────────────────
# Price-action helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_pivots(price_series, left=3, right=3):
    ph, pl = [], []
    s = price_series.values
    L = len(s)
    for i in range(left, L - right):
        left_slice  = s[i - left:i]
        right_slice = s[i + 1:i + 1 + right]
        if s[i] > left_slice.max() and s[i] > right_slice.max():
            ph.append((i, s[i]))
        if s[i] < left_slice.min() and s[i] < right_slice.min():
            pl.append((i, s[i]))
    return ph, pl


def cluster_levels(levels, tol=0.01):
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    current = [levels[0]]
    for p in levels[1:]:
        if abs(p - np.mean(current)) <= tol * np.mean(current):
            current.append(p)
        else:
            clusters.append(np.mean(current))
            current = [p]
    clusters.append(np.mean(current))
    return clusters


def detect_double_top_bottom(ph, pl, price, tol=0.01, min_bars_between=3):
    patterns = []
    for i in range(len(ph) - 1):
        idx1, p1 = ph[i]; idx2, p2 = ph[i + 1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2) / 2):
            patterns.append(('double_top', idx1, idx2, p1, price[idx1:idx2 + 1].min()))
    for i in range(len(pl) - 1):
        idx1, p1 = pl[i]; idx2, p2 = pl[i + 1]
        if idx2 - idx1 >= min_bars_between and abs(p1 - p2) <= tol * ((p1 + p2) / 2):
            patterns.append(('double_bottom', idx1, idx2, p1, price[idx1:idx2 + 1].max()))
    return patterns


def detect_triple_top_bottom(ph, pl, price, tol=0.01, min_bars_between=3):
    patterns = []
    for i in range(len(ph) - 2):
        idx1, p1 = ph[i]; idx2, p2 = ph[i + 1]; idx3, p3 = ph[i + 2]
        if (idx2 - idx1 >= min_bars_between and idx3 - idx2 >= min_bars_between
                and abs(p1 - p2) <= tol * ((p1 + p2) / 2)
                and abs(p2 - p3) <= tol * ((p2 + p3) / 2)):
            patterns.append(('triple_top', idx1, idx2, idx3, p1, p2, p3, price[idx1:idx3 + 1].min()))
    for i in range(len(pl) - 2):
        idx1, p1 = pl[i]; idx2, p2 = pl[i + 1]; idx3, p3 = pl[i + 2]
        if (idx2 - idx1 >= min_bars_between and idx3 - idx2 >= min_bars_between
                and abs(p1 - p2) <= tol * ((p1 + p2) / 2)
                and abs(p2 - p3) <= tol * ((p2 + p3) / 2)):
            patterns.append(('triple_bottom', idx1, idx2, idx3, p1, p2, p3, price[idx1:idx3 + 1].max()))
    return patterns


def detect_head_shoulders(ph, pl, tol=0.03, min_spacing=3, max_spacing=60):
    patterns = []
    for i in range(len(ph) - 2):
        l_idx, l_price = ph[i]; m_idx, m_price = ph[i + 1]; r_idx, r_price = ph[i + 2]
        if (m_idx - l_idx >= min_spacing and r_idx - m_idx >= min_spacing
                and m_idx - l_idx <= max_spacing and r_idx - m_idx <= max_spacing):
            sa = (l_price + r_price) / 2
            if m_price > sa and abs(l_price - r_price) <= tol * sa:
                patterns.append(('head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    for i in range(len(pl) - 2):
        l_idx, l_price = pl[i]; m_idx, m_price = pl[i + 1]; r_idx, r_price = pl[i + 2]
        if (m_idx - l_idx >= min_spacing and r_idx - m_idx >= min_spacing
                and m_idx - l_idx <= max_spacing and r_idx - m_idx <= max_spacing):
            sa = (l_price + r_price) / 2
            if m_price < sa and abs(l_price - r_price) <= tol * sa:
                patterns.append(('inverse_head_and_shoulders', l_idx, m_idx, r_idx, l_price, m_price, r_price))
    return patterns


def detect_engulfing(df):
    patterns = []
    for i in range(1, len(df)):
        prev_o, prev_c = df.loc[i - 1, 'Open'], df.loc[i - 1, 'Close']
        o, c           = df.loc[i, 'Open'],      df.loc[i, 'Close']
        if prev_c < prev_o and c > o and (c - o) > (prev_o - prev_c):
            patterns.append(('bullish_engulfing', i - 1, i))
        if prev_c > prev_o and c < o and (o - c) > (prev_c - prev_o):
            patterns.append(('bearish_engulfing', i - 1, i))
    return patterns


def detect_triangles(ph, pl, min_points=3, tol=0.01):
    patterns = []
    pivots = sorted([(i, p, 'H') for i, p in ph] + [(i, p, 'L') for i, p in pl], key=lambda x: x[0])
    for w in range(0, max(0, len(pivots) - (2 * min_points))):
        window = pivots[w:w + 2 * min_points]
        highs  = [p for _, p, t in window if t == 'H']
        lows   = [p for _, p, t in window if t == 'L']
        if len(highs) >= min_points and len(lows) >= min_points:
            highs_desc = all(highs[i] >= highs[i + 1] * (1 - tol) for i in range(len(highs) - 1))
            lows_asc   = all(lows[i]  <= lows[i + 1]  * (1 + tol) for i in range(len(lows) - 1))
            if highs_desc and lows_asc:
                patterns.append(('sym_triangle', window[0][0], window[-1][0]))
    return patterns


def detect_flags(df, trend_lookback=20, flag_lookback=8, range_ratio_threshold=0.5):
    patterns = []
    if len(df) < trend_lookback + flag_lookback + 1:
        return patterns
    for i in range(trend_lookback + flag_lookback, len(df)):
        prev       = df['Close'].iloc[i - trend_lookback - flag_lookback:i - flag_lookback]
        flag       = df['Close'].iloc[i - flag_lookback:i]
        prev_range = prev.max() - prev.min()
        flag_range = flag.max() - flag.min()
        if prev_range > 0 and flag_range < prev_range * range_ratio_threshold:
            if df['Close'].iloc[i] > flag.max():
                patterns.append(('flag_breakout',  i - flag_lookback, i))
            if df['Close'].iloc[i] < flag.min():
                patterns.append(('flag_breakdown', i - flag_lookback, i))
    return patterns


def build_zones(levels, width_pct=0.005):
    return [(lv * (1 - width_pct), lv * (1 + width_pct)) for lv in levels]


def in_zone(price, zone):
    return zone[0] <= price <= zone[1]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy & Backtest
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df, params):
    price = df['Close']
    ph, pl = get_pivots(price, left=params['pivot_window'], right=params['pivot_window'])

    resistances = cluster_levels([p for _, p in ph], tol=params['cluster_tol'])
    supports    = cluster_levels([p for _, p in pl], tol=params['cluster_tol'])
    sup_zones   = build_zones(supports,    width_pct=params['zone_width'])
    res_zones   = build_zones(resistances, width_pct=params['zone_width'])

    patterns_db     = detect_double_top_bottom(ph, pl, price.values, tol=params['pattern_tol'], min_bars_between=params['min_bars_between'])
    patterns_triple = detect_triple_top_bottom(ph, pl, price.values, tol=params['pattern_tol'], min_bars_between=params['min_bars_between'])
    hs_patterns     = detect_head_shoulders(ph, pl, tol=params.get('hs_tol', 0.03), min_spacing=params.get('min_bars_between', 3))
    engulf          = detect_engulfing(df)
    triangles       = detect_triangles(ph, pl, min_points=3, tol=params.get('triangle_tol', 0.01))
    flags           = detect_flags(df, trend_lookback=params.get('flag_trend_lookback', 20),
                                   flag_lookback=params.get('flag_lookback', 8), range_ratio_threshold=0.5)

    weights = {
        'head_and_shoulders': -3.0, 'inverse_head_and_shoulders': 3.0,
        'double_top': -2.0,          'double_bottom': 2.0,
        'triple_top': -2.5,          'triple_bottom': 2.5,
        'bullish_engulfing': 1.5,    'bearish_engulfing': -1.5,
        'sym_triangle': 1.8,         'flag_breakout': 1.3,
        'flag_breakdown': -1.3,      'breakout': 1.5,
        'support_zone': 1.0,         'resistance_zone': -1.0,
        'upper_wick_liquidity_trap': -1.2, 'lower_wick_liquidity_trap': 1.2,
    }

    # precompute pattern-by-end-index lookup
    pattern_by_index = {}
    def add_pattern(idx, name):
        pattern_by_index.setdefault(idx, []).append(name)

    for p in patterns_db:     add_pattern(p[2], p[0])
    for p in patterns_triple: add_pattern(p[3], p[0])
    for p in hs_patterns:     add_pattern(p[2], p[0])
    for p in engulf:          add_pattern(p[2], p[0])
    for p in triangles:       add_pattern(p[2], p[0])
    for p in flags:           add_pattern(p[2], p[0])

    vol_median = df['Volume'].median()
    if np.isnan(vol_median):
        vol_median = 0

    signals, reasons = [], []
    L = len(df)

    for i in range(L):
        score = 0.0
        reason_list = []
        close = df.loc[i, 'Close']

        recent_patterns = []
        for j in range(max(0, i - params['pattern_lookahead']), i + 1):
            recent_patterns += pattern_by_index.get(j, [])
        for rp in recent_patterns:
            w = weights.get(rp, 0)
            score += w
            reason_list.append(f"{rp}({w:+.1f})")

        for z in sup_zones:
            if in_zone(close, z):
                score += weights['support_zone']
                reason_list.append(f"near_support_zone {round((z[0]+z[1])/2,2)}")
        for z in res_zones:
            if in_zone(close, z):
                score += weights['resistance_zone']
                reason_list.append(f"near_resistance_zone {round((z[0]+z[1])/2,2)}")

        look = params['breakout_lookback']
        if i > look:
            recent_high = df.loc[i - look:i - 1, 'High'].max()
            recent_low  = df.loc[i - look:i - 1, 'Low'].min()
            if close > recent_high:
                score += weights['breakout']
                reason_list.append(f"breakout_above_{recent_high:.2f}")
            if close < recent_low:
                score -= weights['breakout']
                reason_list.append(f"breakdown_below_{recent_low:.2f}")

        high  = df.loc[i, 'High'];  low   = df.loc[i, 'Low']
        openp = df.loc[i, 'Open'];  body  = abs(close - openp) + 1e-9
        upper_wick = high - max(close, openp)
        lower_wick = min(close, openp) - low
        if upper_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > vol_median * params['volume_factor']:
            score += weights['upper_wick_liquidity_trap']
            reason_list.append('upper_wick_liquidity_trap')
        if lower_wick > params['wick_factor'] * body and df.loc[i, 'Volume'] > vol_median * params['volume_factor']:
            score += weights['lower_wick_liquidity_trap']
            reason_list.append('lower_wick_liquidity_trap')

        thresh = params.get('signal_threshold', 1.0)
        sig = (1 if score >= thresh and 'long' in params['allowed_dirs']
               else (-1 if score <= -thresh and 'short' in params['allowed_dirs'] else 0))

        signals.append(sig)
        reasons.append(';'.join(reason_list) if reason_list else '')

    df_signals = df.copy()
    df_signals['signal'] = signals
    df_signals['reason'] = reasons
    meta = {
        'supports': supports, 'resistances': resistances,
        'patterns': {'double': patterns_db, 'triple': patterns_triple,
                     'hs': hs_patterns, 'engulf': engulf,
                     'triangles': triangles, 'flags': flags}
    }
    return df_signals, meta


def backtest_signals(df_signals, params):
    """
    Backtest with correct SL/TP priority and gap-open handling.

    Rules:
      • Signal on candle n  →  entry at candle n+1 OPEN price.
      • Gap-open handling: if candle j opens beyond SL (adverse) or TP (favourable),
        fill is at that candle's OPEN (not at SL/TP level).
      • Intrabar order (conservative): SL checked first using Low (long) / High (short),
        then TP using High (long) / Low (short).
    """
    trades = []
    L = len(df_signals)
    i = 0

    while i < L - 1:
        row = df_signals.loc[i]
        sig = row['signal']
        if sig == 0:
            i += 1
            continue

        entry_idx   = i + 1 if i + 1 < L else i
        entry_open  = df_signals.loc[entry_idx, 'Open']
        entry_price = entry_open if not pd.isna(entry_open) else df_signals.loc[entry_idx, 'Close']

        if params.get('use_target_points') and params.get('target_points') is not None:
            tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
        else:
            tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])

        sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])

        exit_price  = None
        exit_idx    = None
        exit_reason = None
        max_hold    = params['max_hold']

        for j in range(entry_idx, min(L, entry_idx + max_hold + 1)):
            day_open = df_signals.loc[j, 'Open']
            day_high = df_signals.loc[j, 'High']
            day_low  = df_signals.loc[j, 'Low']

            # ── Gap-open handling (only for candles AFTER entry candle) ───────
            # On the entry candle entry_price == day_open, so SL/TP are always
            # strictly beyond open — no gap check needed there.
            if j > entry_idx:
                if sig == 1:
                    if day_open <= sl:   # gap-down blew through SL
                        exit_price, exit_idx, exit_reason = day_open, j, 'sl_gap_down'
                        break
                    if day_open >= tp:   # gap-up jumped past TP (lucky)
                        exit_price, exit_idx, exit_reason = day_open, j, 'tp_gap_up'
                        break
                else:                   # short
                    if day_open >= sl:   # gap-up blew through SL
                        exit_price, exit_idx, exit_reason = day_open, j, 'sl_gap_up'
                        break
                    if day_open <= tp:   # gap-down jumped past TP (lucky)
                        exit_price, exit_idx, exit_reason = day_open, j, 'tp_gap_down'
                        break

            # ── Intrabar: SL first (conservative), then TP ───────────────────
            if sig == 1:
                if day_low <= sl:    # check SL with Low
                    exit_price, exit_idx, exit_reason = sl, j, 'sl'
                    break
                if day_high >= tp:   # check TP with High
                    exit_price, exit_idx, exit_reason = tp, j, 'tp'
                    break
            else:
                if day_high >= sl:   # check SL with High
                    exit_price, exit_idx, exit_reason = sl, j, 'sl'
                    break
                if day_low <= tp:    # check TP with Low
                    exit_price, exit_idx, exit_reason = tp, j, 'tp'
                    break

        if exit_price is None:
            exit_idx    = min(L - 1, entry_idx + max_hold)
            exit_price  = df_signals.loc[exit_idx, 'Close']
            exit_reason = 'time_exit'

        pnl        = ((exit_price - entry_price) / entry_price  if sig == 1
                      else (entry_price - exit_price) / entry_price)
        pnl_points = (exit_price - entry_price if sig == 1 else entry_price - exit_price)

        trades.append({
            'entry_idx':   entry_idx,
            'exit_idx':    exit_idx,
            'entry_time':  df_signals.loc[entry_idx, 'Date'],
            'exit_time':   df_signals.loc[exit_idx,  'Date'],
            'direction':   'long' if sig == 1 else 'short',
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'sl':          sl,
            'tp':          tp,
            'pnl':         pnl,
            'pnl_points':  pnl_points,
            'hold_days':   (df_signals.loc[exit_idx, 'Date'] - df_signals.loc[entry_idx, 'Date']).days,
            'exit_reason': exit_reason,
            'signal_reason': df_signals.loc[i, 'reason'] or exit_reason,
        })
        i = (exit_idx + 1) if exit_idx is not None else (i + 1)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {
            'total_trades': 0, 'positive_trades': 0, 'negative_trades': 0,
            'accuracy': 0, 'total_pnl_pct': 0, 'avg_pnl_pct': 0,
            'avg_hold_days': 0, 'total_points': 0,
        }
    else:
        stats = {
            'total_trades':    len(trades_df),
            'positive_trades': int((trades_df['pnl'] > 0).sum()),
            'negative_trades': int((trades_df['pnl'] <= 0).sum()),
            'accuracy':        float((trades_df['pnl'] > 0).mean()),
            'total_pnl_pct':   float(trades_df['pnl'].sum() * 100),
            'avg_pnl_pct':     float(trades_df['pnl'].mean() * 100),
            'avg_hold_days':   float(trades_df['hold_days'].mean()),
            'total_points':    float(trades_df['pnl_points'].sum()),
        }
    return trades_df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_params(n_samples=50, param_space=None):
    if param_space is None:
        param_space = {
            'pivot_window':    [2, 3, 4, 5],
            'cluster_tol':     [0.003, 0.005, 0.01],
            'zone_width':      [0.003, 0.005, 0.01],
            'sl_pct':          [0.005, 0.01, 0.02],
            'tp_pct':          [0.01, 0.02, 0.05],
            'max_hold':        [3, 5, 10],
            'breakout_lookback': [3, 5, 10],
            'pattern_tol':     [0.01, 0.02],
            'min_bars_between': [2, 3, 5],
        }
    keys = list(param_space.keys())
    return [{k: random.choice(param_space[k]) for k in keys} for _ in range(n_samples)]


def grid_params(param_grid):
    keys   = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def find_best_strategy(df_train, search_type='random', random_iters=50, grid=None,
                       allowed_dirs=None, desired_accuracy=0.7, min_trades=5,
                       progress_callback=None, use_points=False, target_points=None):
    if allowed_dirs is None:
        allowed_dirs = ['long', 'short']
    best = None
    best_metric = -np.inf
    tried = 0

    samples = (sample_random_params(random_iters, param_space=grid) if search_type == 'random'
               else list(grid_params(grid)))
    total = len(samples)

    for s in samples:
        tried += 1
        params = {
            'pivot_window':      s.get('pivot_window', 3),
            'cluster_tol':       s.get('cluster_tol', 0.005),
            'zone_width':        s.get('zone_width', 0.005),
            'sl_pct':            s.get('sl_pct', 0.01),
            'tp_pct':            s.get('tp_pct', 0.02),
            'max_hold':          s.get('max_hold', 5),
            'breakout_lookback': s.get('breakout_lookback', 5),
            'pattern_tol':       s.get('pattern_tol', 0.02),
            'min_bars_between':  s.get('min_bars_between', 3),
            'wick_factor':       1.5,
            'volume_factor':     1.5,
            'pattern_lookahead': 5,
            'hs_tol':            0.03,
            'signal_threshold':  s.get('signal_threshold', 1.0),
            'triangle_tol':      s.get('triangle_tol', 0.01),
            'allowed_dirs':      allowed_dirs,
            'use_target_points': use_points,
            'target_points':     target_points,
        }
        df_signals, _ = generate_signals(df_train, params)
        trades_df, stats = backtest_signals(df_signals, params)

        if stats['total_trades'] < min_trades:
            metric = -np.inf
        else:
            metric = stats['total_pnl_pct']
            if stats['accuracy'] >= desired_accuracy:
                metric += 1000

        if metric > best_metric:
            best_metric = metric
            best = {'params': params, 'trades': trades_df, 'stats': stats}

        if progress_callback:
            try:
                progress_callback(tried, total)
            except Exception:
                pass

    return best, tried


# ─────────────────────────────────────────────────────────────────────────────
# Helper outputs (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(df):
    try:
        returns   = df['Close'].pct_change().dropna()
        mean_ret  = returns.mean() * 252
        vol       = returns.std() * np.sqrt(252)
        last_close = df['Close'].iloc[-1]
        trend = ('uptrend' if len(df) >= 30 and last_close > df['Close'].iloc[-30:].mean()
                 else 'sideways/downtrend')
        opp = []
        if mean_ret > 0.05: opp.append('long-biased')
        if vol > 0.2:       opp.append('high volatility')
        return (f"Data from {df['Date'].min().date()} to {df['Date'].max().date()}. "
                f"Latest close {last_close:.2f}. Recent trend: {trend}. "
                f"Annualised mean return ≈ {mean_ret:.2%} with volatility {vol:.2%}. "
                f"Potential opportunities: {', '.join(opp) or 'range trading and breakout plays'}. "
                "Look for support/resistance, double bottom/top patterns and liquidity wick traps near key levels. "
                "Use disciplined SL and position sizing.")
    except Exception:
        return 'Could not generate automated summary.'


def generate_live_recommendation(df_signals, params, target_points=50,
                                 backtest_stats=None, capital=0, risk_pct=1.0):
    if df_signals.empty:
        return None
    last = df_signals.iloc[-1]
    sig  = last['signal']
    if sig == 0:
        return None

    entry_price = last['Close']  # estimated — actual entry = next candle open

    if params.get('use_target_points') and params.get('target_points') is not None:
        tp = entry_price + params['target_points'] if sig == 1 else entry_price - params['target_points']
    else:
        tp = entry_price * (1 + params['tp_pct']) if sig == 1 else entry_price * (1 - params['tp_pct'])

    sl = entry_price * (1 - params['sl_pct']) if sig == 1 else entry_price * (1 + params['sl_pct'])

    unit_risk       = abs(entry_price - sl)
    suggested_units = None
    if capital and capital > 0 and unit_risk > 0:
        risk_amount     = capital * (risk_pct / 100.0)
        suggested_units = int(risk_amount // unit_risk)

    return {
        'direction':                 'long' if sig == 1 else 'short',
        'estimated_entry_price':     round(float(entry_price), 4),
        'note':                      'Actual entry = OPEN of next candle',
        'stop_loss':                 round(float(sl), 4),
        'target_price':              round(float(tp), 4),
        'unit_risk':                 round(float(unit_risk), 4),
        'suggested_units_by_capital': int(suggested_units) if suggested_units is not None else None,
        'reason':                    last.get('reason', '') or 'signal',
        'probability_of_profit':     (float(backtest_stats['accuracy'])
                                      if backtest_stats and backtest_stats.get('total_trades', 0) > 0
                                      else None),
    }


def backtest_human_readable(stats, params, buy_hold_points=None):
    if stats['total_trades'] == 0:
        return 'No trades executed with these parameters.'
    s = (f"Backtest: sl {params['sl_pct']*100:.2f}%, tp {params['tp_pct']*100:.2f}%. "
         f"Total trades: {stats['total_trades']}. "
         f"Winners: {stats['positive_trades']} ({stats['accuracy']*100:.2f}% win rate). "
         f"Total return: {stats['total_pnl_pct']:.2f}%.")
    if buy_hold_points is not None:
        s += f" Buy-and-hold points: {buy_hold_points:.2f}. Strategy points: {stats.get('total_points',0):.2f}."
    return s


def explain_signal_reason(reason_str, direction, entry, sl, tp, accuracy=None):
    """Convert raw semi-colon-delimited reason string into a human-readable trade rationale."""
    PATTERN_DESC = {
        'bullish_engulfing':          "A **Bullish Engulfing** candle — a large green candle fully engulfs the prior red candle, signalling strong bullish reversal momentum.",
        'bearish_engulfing':          "A **Bearish Engulfing** candle — a large red candle fully engulfs the prior green candle, signalling bearish reversal momentum.",
        'double_bottom':              "A **Double Bottom (W-shape)** — price tested the same support twice and bounced, confirming buying interest.",
        'double_top':                 "A **Double Top (M-shape)** — price was rejected at the same resistance twice, confirming selling pressure.",
        'triple_bottom':              "A **Triple Bottom** — three support tests without breakdown: a high-conviction bullish reversal.",
        'triple_top':                 "A **Triple Top** — three failed breakout attempts at resistance: a high-conviction bearish reversal.",
        'head_and_shoulders':         "A **Head & Shoulders** — classic bearish reversal; central peak (head) flanked by two lower peaks (shoulders).",
        'inverse_head_and_shoulders': "An **Inverse Head & Shoulders** — classic bullish reversal; central trough flanked by two shallower troughs.",
        'sym_triangle':               "A **Symmetrical Triangle** — price coiling between converging highs and lows, building energy for a breakout.",
        'flag_breakout':              "A **Bull Flag Breakout** — tight consolidation after a rally, then a topside breakout.",
        'flag_breakdown':             "A **Bear Flag Breakdown** — tight consolidation after a decline, then a resumption lower.",
        'upper_wick_liquidity_trap':  "An **Upper Wick Liquidity Trap** (large wick + high volume) — price spiked up to trigger stops, then reversed sharply lower.",
        'lower_wick_liquidity_trap':  "A **Lower Wick Liquidity Trap** (large wick + high volume) — price was pushed below support to trigger stops, then reversed sharply higher.",
    }
    parts   = [p.strip() for p in (reason_str or '').split(';') if p.strip()]
    bullets = []
    for part in parts:
        matched = False
        for key, desc in PATTERN_DESC.items():
            if key in part:
                bullets.append(f"• {desc}"); matched = True; break
        if not matched:
            if 'breakout_above_' in part:
                lvl = part.split('breakout_above_')[-1]
                bullets.append(f"• **Price Breakout** above the recent `{lvl}` high — bullish momentum building.")
            elif 'breakdown_below_' in part:
                lvl = part.split('breakdown_below_')[-1]
                bullets.append(f"• **Price Breakdown** below the recent `{lvl}` low — bearish momentum accelerating.")
            elif 'near_support_zone' in part:
                lvl = part.split('near_support_zone')[-1].strip()
                bullets.append(f"• **Near Support Zone** at `{lvl}` — buyers have historically stepped in strongly here.")
            elif 'near_resistance_zone' in part:
                lvl = part.split('near_resistance_zone')[-1].strip()
                bullets.append(f"• **Near Resistance Zone** at `{lvl}` — sellers have historically dominated here.")

    dir_label = "🟢 LONG (Buy)" if direction == 'long' else "🔴 SHORT (Sell)"
    rr        = abs(tp - entry) / max(abs(entry - sl), 1e-9)
    rr_label  = f"1 : {rr:.2f}  {'✅ Favourable' if rr >= 1.5 else '⚠️ Below ideal 1:1.5'}"
    lines = [
        f"**Direction:** {dir_label}",
        f"**Est. Entry:** `{entry:.4f}`  |  **SL:** `{sl:.4f}`  |  **Target:** `{tp:.4f}`",
        f"**Risk : Reward:** {rr_label}",
    ]
    if accuracy is not None:
        lines.append(f"**Backtest Win Rate:** {accuracy * 100:.1f}%")
    lines += ["", "**Why this trade was recommended:**",
              *(bullets if bullets else ["• Multiple scoring factors aligned — no single dominant pattern."])]
    return "\n\n".join(lines[:4]) + "\n\n" + "\n".join(lines[4:])


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick chart
# ─────────────────────────────────────────────────────────────────────────────

def draw_candlestick_chart(df, title='', sl=None, tp=None, entry=None, n_candles=80):
    """Dark-theme OHLC candlestick chart with optional SL/TP/entry overlay lines."""
    df  = df.tail(n_candles).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    for i, row in df.iterrows():
        is_up  = row['Close'] >= row['Open']
        color  = '#26a69a' if is_up else '#ef5350'
        bottom = min(row['Open'], row['Close'])
        height = max(abs(row['Close'] - row['Open']), (row['High'] - row['Low']) * 0.001)
        ax.bar(i, height, bottom=bottom, color=color, width=0.6, alpha=0.92, zorder=2)
        ax.plot([i, i], [row['Low'],          bottom],         color=color, linewidth=0.9, zorder=1)
        ax.plot([i, i], [bottom + height,     row['High']],    color=color, linewidth=0.9, zorder=1)
    if entry is not None:
        ax.axhline(entry, color='#2196F3', linestyle='--', linewidth=1.3, label=f'Entry {entry:.2f}', zorder=3)
    if sl is not None:
        ax.axhline(sl,    color='#f44336', linestyle='--', linewidth=1.3, label=f'SL {sl:.2f}',    zorder=3)
    if tp is not None:
        ax.axhline(tp,    color='#4CAF50', linestyle='--', linewidth=1.3, label=f'TP {tp:.2f}',    zorder=3)
    if any(x is not None for x in [entry, sl, tp]):
        ax.legend(fontsize=8, facecolor='#1e2130', labelcolor='white', edgecolor='#444')
    n    = len(df); step = max(1, n // 8); ticks = list(range(0, n, step))
    lbls = []
    for j in ticks:
        try:    lbls.append(df.loc[j, 'Date'].strftime('%m-%d %H:%M'))
        except: lbls.append(str(j))
    ax.set_xticks(ticks); ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=7, color='#aaaaaa')
    ax.tick_params(axis='y', colors='#aaaaaa', labelsize=7)
    ax.set_title(title, fontsize=9, color='#dddddd')
    ax.grid(axis='y', alpha=0.15, color='#555')
    for sp in ['bottom', 'left']:  ax.spines[sp].set_color('#444')
    for sp in ['top',    'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dhan Broker Integration
# ─────────────────────────────────────────────────────────────────────────────

def execute_dhan_order(dhan_cfg, direction, ltp, action='entry'):
    """
    Place an order via dhanhq.
      action    : 'entry' or 'exit'
      direction : 'long'  or 'short'
      ltp       : last traded price (used as price for LIMIT orders)
    Returns response dict or {'error': ...}.

    Equity logic  — entry: BUY signal→BUY, SELL signal→SELL
                  — exit : reverse of entry direction
    Options logic — entry: BUY signal→CE BUY, SELL signal→PE BUY  (always buyer)
                  — exit : SELL the option that was bought
    """
    if not DHAN_AVAILABLE:
        return {'error': 'dhanhq not installed. Run: pip install dhanhq'}
    try:
        dhan = DhanHQ(dhan_cfg['client_id'], dhan_cfg['access_token'])
    except Exception as e:
        return {'error': f'Dhan init failed: {e}'}

    try:
        if dhan_cfg.get('options_mode'):
            if action == 'entry':
                security_id = (dhan_cfg['ce_security_id'] if direction == 'long'
                               else dhan_cfg['pe_security_id'])
                tx_type    = 'BUY'
                order_type = dhan_cfg['opt_entry_order_type']
            else:
                security_id = dhan_cfg.get('open_opt_security_id', '')
                tx_type    = 'SELL'
                order_type = dhan_cfg['opt_exit_order_type']
            price = round(float(ltp), 2) if order_type == 'LIMIT' else 0
            resp  = dhan.place_order(
                transactionType=tx_type, exchangeSegment=dhan_cfg['opt_exchange'],
                productType='INTRADAY', orderType=order_type, validity='DAY',
                securityId=security_id, quantity=int(dhan_cfg['opt_qty']),
                price=price, triggerPrice=0)
        else:
            if action == 'entry':
                tx_type    = 'BUY' if direction == 'long' else 'SELL'
                order_type = dhan_cfg['eq_entry_order_type']
            else:
                tx_type    = 'SELL' if direction == 'long' else 'BUY'
                order_type = dhan_cfg['eq_exit_order_type']
            price = round(float(ltp), 2) if order_type == 'LIMIT' else 0
            resp  = dhan.place_order(
                transactionType=tx_type, exchangeSegment=dhan_cfg['eq_exchange'],
                productType=dhan_cfg['eq_product'], orderType=order_type,
                validity='DAY', securityId=str(dhan_cfg['eq_security_id']),
                quantity=int(dhan_cfg['eq_qty']), price=price, triggerPrice=0)
        return resp if isinstance(resp, dict) else {'status': 'sent', 'raw': str(resp)}
    except Exception as e:
        return {'error': f'Order placement failed: {e}'}


def render_dhan_config(key_prefix='live'):
    """
    Renders Dhan broker checkbox + settings UI.
    Returns (dhan_enabled: bool, dhan_cfg: dict | None).
    """
    dhan_enabled = st.checkbox(
        "🏦 Enable Dhan Broker",
        value=False, key=f'{key_prefix}_dhan_enabled',
        help="Disabled by default. Enable to place real orders via Dhan API.")
    if not dhan_enabled:
        return False, None
    if not DHAN_AVAILABLE:
        st.error("dhanhq not installed. Run: `pip install dhanhq` then restart.")
        return False, None

    with st.expander("🔐 Dhan Credentials & Order Settings", expanded=True):
        cr1, cr2 = st.columns(2)
        with cr1: client_id    = st.text_input("Client ID",     value="",  key=f'{key_prefix}_dhan_cid')
        with cr2: access_token = st.text_input("Access Token",  value="",  key=f'{key_prefix}_dhan_tok', type='password')

        st.markdown("---")
        options_mode = st.checkbox(
            "📈 Options Trading Mode", value=False, key=f'{key_prefix}_dhan_opt',
            help="OFF → Equity/Futures orders.  ON → F&O options orders (always buyer).")

        if not options_mode:
            st.markdown("**Equity / Futures Settings**")
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                eq_product       = st.selectbox("Product Type", ['INTRADAY', 'CNC'],
                                                 key=f'{key_prefix}_eq_prod',
                                                 help="INTRADAY = MIS, CNC = Delivery")
                eq_exch_lbl      = st.selectbox("Exchange", ['NSE', 'BSE'], key=f'{key_prefix}_eq_exch')
                eq_exchange      = 'NSE_EQ' if eq_exch_lbl == 'NSE' else 'BSE_EQ'
            with rc2:
                eq_security_id   = st.text_input("Security ID", value="1594",
                                                  key=f'{key_prefix}_eq_sec',
                                                  help="Dhan Security ID (default 1594 = Reliance)")
                eq_qty           = st.number_input("Quantity", min_value=1, value=1, key=f'{key_prefix}_eq_qty')
            with rc3:
                eq_entry_ot      = st.selectbox("Entry Order Type", ['LIMIT', 'MARKET'],
                                                 key=f'{key_prefix}_eq_entry_ot',
                                                 help="LIMIT uses current LTP as the price.")
                eq_exit_ot       = st.selectbox("Exit Order Type",  ['MARKET', 'LIMIT'],
                                                 key=f'{key_prefix}_eq_exit_ot')
            st.caption(f"Exchange: {eq_exchange} | Product: {eq_product} | Qty: {eq_qty} | "
                       f"Entry: {eq_entry_ot} | Exit: {eq_exit_ot}")
            dhan_cfg = {
                'client_id': client_id, 'access_token': access_token,
                'options_mode': False, 'eq_exchange': eq_exchange,
                'eq_product': eq_product, 'eq_security_id': eq_security_id,
                'eq_qty': eq_qty, 'eq_entry_order_type': eq_entry_ot, 'eq_exit_order_type': eq_exit_ot,
            }
        else:
            st.markdown("**Options Settings** *(always buyer — CE on BUY signal, PE on SELL signal)*")
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                opt_exch    = st.selectbox("Exchange", ['NSE_FNO', 'BSE_FNO'], key=f'{key_prefix}_opt_exch')
                opt_qty     = st.number_input("Lot Size / Quantity", min_value=1, value=65, key=f'{key_prefix}_opt_qty')
            with oc2:
                ce_sec_id   = st.text_input("CE Security ID", value="", key=f'{key_prefix}_ce_sec',
                                             help="Security ID of the CALL option contract.")
                pe_sec_id   = st.text_input("PE Security ID", value="", key=f'{key_prefix}_pe_sec',
                                             help="Security ID of the PUT option contract.")
            with oc3:
                opt_entry_ot = st.selectbox("Entry Order Type", ['MARKET', 'LIMIT'], key=f'{key_prefix}_opt_entry_ot')
                opt_exit_ot  = st.selectbox("Exit Order Type",  ['MARKET', 'LIMIT'], key=f'{key_prefix}_opt_exit_ot')
            st.caption(f"CE ID: {ce_sec_id or 'not set'} | PE ID: {pe_sec_id or 'not set'} | "
                       f"Qty: {opt_qty} | Entry: {opt_entry_ot} | Exit: {opt_exit_ot}")
            dhan_cfg = {
                'client_id': client_id, 'access_token': access_token,
                'options_mode': True, 'opt_exchange': opt_exch, 'opt_qty': opt_qty,
                'ce_security_id': ce_sec_id, 'pe_security_id': pe_sec_id,
                'opt_entry_order_type': opt_entry_ot, 'opt_exit_order_type': opt_exit_ot,
            }
    return True, dhan_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Default params (used when no optimisation has been run yet)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    'pivot_window': 3, 'cluster_tol': 0.005, 'zone_width': 0.005,
    'sl_pct': 0.01, 'tp_pct': 0.02, 'max_hold': 5,
    'breakout_lookback': 5, 'pattern_tol': 0.02, 'min_bars_between': 3,
    'wick_factor': 1.5, 'volume_factor': 1.5, 'pattern_lookahead': 5,
    'hs_tol': 0.03, 'signal_threshold': 1.0, 'triangle_tol': 0.01,
    'allowed_dirs': ['long', 'short'],
    'use_target_points': False, 'target_points': None,
    'flag_trend_lookback': 20, 'flag_lookback': 8,
}


# ─────────────────────────────────────────────────────────────────────────────
# Live Trading Tab  (no-flicker cached approach)
# ─────────────────────────────────────────────────────────────────────────────

def _open_position(direction, ltp, params, capital, risk_pct, symbol, timeframe, dhan_enabled, dhan_cfg):
    """Create a position dict and optionally fire a Dhan entry order."""
    sl_v = ltp * (1 - params['sl_pct']) if direction == 'long' else ltp * (1 + params['sl_pct'])
    tp_v = ltp * (1 + params['tp_pct']) if direction == 'long' else ltp * (1 - params['tp_pct'])
    qty  = 1
    if capital > 0 and abs(ltp - sl_v) > 0:
        qty = max(1, int((capital * risk_pct / 100) // abs(ltp - sl_v)))
    pos = {'direction': direction, 'entry_price': ltp, 'sl': sl_v, 'tp': tp_v, 'qty': qty,
           'entry_time': pd.Timestamp.now(tz=IST).strftime('%Y-%m-%d %H:%M:%S %Z'),
           'symbol': symbol, 'timeframe': timeframe, 'dhan_entry_response': None}
    if dhan_enabled and dhan_cfg:
        resp = execute_dhan_order(dhan_cfg, direction, ltp, 'entry')
        pos['dhan_entry_response'] = resp
        if dhan_cfg.get('options_mode'):
            dhan_cfg['open_opt_security_id'] = (dhan_cfg['ce_security_id'] if direction == 'long'
                                                else dhan_cfg['pe_security_id'])
            st.session_state['dhan_cfg'] = dhan_cfg
        st.toast(f"Dhan entry order sent ✅", icon="📤")
    return pos


def _close_position(pos, exit_price, exit_reason, dhan_enabled, dhan_cfg):
    """Log trade and clear the open position."""
    pnl_pts = (exit_price - pos['entry_price']) if pos['direction'] == 'long' else (pos['entry_price'] - exit_price)
    trade = {**pos, 'exit_price': exit_price,
             'exit_time': pd.Timestamp.now(tz=IST).strftime('%Y-%m-%d %H:%M:%S %Z'),
             'pnl_points': round(pnl_pts, 4),
             'pnl_pct': round(pnl_pts / pos['entry_price'] * 100, 4),
             'exit_reason': exit_reason, 'dhan_exit_response': None}
    if dhan_enabled and dhan_cfg:
        resp = execute_dhan_order(dhan_cfg, pos['direction'], exit_price, 'exit')
        trade['dhan_exit_response'] = resp
        st.toast(f"Dhan exit order sent ✅", icon="📤")
    st.session_state['trade_history'].append(trade)
    st.session_state['live_position'] = None


def live_trading_section():
    """
    Live monitoring tab.
    - Fetches data only when needed (cache-guarded → no full-page flicker).
    - Displays last completed candle row prominently.
    - Candlestick chart with SL/TP/Entry overlay.
    - Start / Stop / Take Position / Square Off buttons.
    - Dhan broker integration (equity & options).
    """
    st.header("🔴 Live Trading Monitor")
    st.caption("Signal detected on the last **completed** candle. Entry = OPEN of the next candle.")

    # ── Session-state initialisation ──────────────────────────────────────────
    for k, v in [('live_monitoring', False), ('live_position', None),
                 ('trade_history', []),       ('live_cache', None),
                 ('live_last_fetch_ts', 0.0), ('live_manual_refresh', False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Instrument / TF / Period ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        instr_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key='live_instr')
        if INSTRUMENTS[instr_name] == '__custom__':
            symbol       = st.text_input("Custom Symbol (e.g. TATAMOTORS.NS, AAPL)",
                                          value='TATAMOTORS.NS', key='live_custom_sym').strip().upper()
            display_name = symbol
        else:
            symbol       = INSTRUMENTS[instr_name]
            display_name = instr_name
    with col2:
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=4, key='live_tf')
    with col3:
        max_days_tf    = _INTERVAL_MAX_DAYS.get(timeframe, 99999)
        compat_periods = [p for p in PERIODS if _PERIOD_DAYS[p] <= max_days_tf] or ['1d']
        period = st.selectbox("Period (strategy context)", compat_periods,
                               index=min(2, len(compat_periods) - 1), key='live_period')

    # ── Strategy params ───────────────────────────────────────────────────────
    best_params = st.session_state.get('best_params')
    if best_params:
        st.success("✅ Using optimised params from Backtest tab.")
        params = best_params.copy()
    else:
        st.info("ℹ️ Using default params. Run optimisation in the Backtest tab for best results.")
        params = DEFAULT_PARAMS.copy()

    d1, d2, d3 = st.columns(3)
    with d1:
        side_live = st.selectbox("Direction filter", ['both', 'long only', 'short only'], key='live_side')
        params['allowed_dirs'] = (['long', 'short'] if side_live == 'both'
                                   else ['long'] if 'long' in side_live else ['short'])
    with d2:
        live_capital  = st.number_input("Capital for sizing (0 = off)", min_value=0, value=0, key='live_cap')
    with d3:
        live_risk_pct = st.number_input("Risk %", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key='live_risk')

    # ── Dhan broker config ────────────────────────────────────────────────────
    dhan_enabled, dhan_cfg = render_dhan_config(key_prefix='live')
    if dhan_cfg:
        st.session_state['dhan_cfg'] = dhan_cfg

    st.markdown("---")

    # ── Refresh controls ──────────────────────────────────────────────────────
    ref1, ref2, ref3 = st.columns([1, 1, 2])
    with ref1:
        refresh_interval = st.number_input(
            "Refresh (sec)", min_value=1.5, max_value=3600.0,
            value=10.0, step=0.5, key='live_refresh_interval',
            help="Minimum 1.5 s (yfinance rate limit).")
    with ref2:
        st.write("")  # vertical spacer
        manual_refresh_btn = st.button("🔄 Refresh Now", use_container_width=True)

    # ── Monitoring / position control buttons ─────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    cache_snap = st.session_state.get('live_cache')
    sig_snap   = cache_snap['sig'] if cache_snap else 0
    position   = st.session_state.get('live_position')

    with mc1:
        if not st.session_state['live_monitoring']:
            if st.button("▶ Start Monitoring", use_container_width=True):
                st.session_state['live_monitoring']     = True
                st.session_state['live_manual_refresh'] = True
                st.rerun()
        else:
            if st.button("⏹ Stop Monitoring", use_container_width=True):
                st.session_state['live_monitoring'] = False
                st.rerun()

    with mc2:
        if sig_snap != 0 and position is None:
            lbl = "🟢 Take LONG" if sig_snap == 1 else "🔴 Take SHORT"
            if st.button(lbl, use_container_width=True):
                ltp_now = cache_snap.get('ltp', cache_snap.get('entry_est', 0))
                pos = _open_position('long' if sig_snap == 1 else 'short', ltp_now,
                                     params, live_capital, live_risk_pct, symbol, timeframe,
                                     dhan_enabled, dhan_cfg)
                st.session_state['live_position'] = pos
                st.rerun()
        else:
            st.button("Take Position", disabled=True, use_container_width=True)

    with mc3:
        if position is not None:
            if st.button("🟥 Square Off", use_container_width=True):
                sq_price = (cache_snap['ltp'] if cache_snap else position['entry_price'])
                _close_position(position, sq_price, 'manual_squareoff',
                                dhan_enabled, st.session_state.get('dhan_cfg'))
                st.rerun()
        else:
            st.button("Square Off", disabled=True, use_container_width=True)

    with mc4:
        mon_label = "🟢 ACTIVE" if st.session_state['live_monitoring'] else "⚫ STOPPED"
        pos_label = (f"📌 {position['direction'].upper()} @ {position['entry_price']:.4f}"
                     if position else "💤 No position")
        st.markdown(f"**Monitor:** {mon_label}  \n**Position:** {pos_label}")

    st.markdown("---")

    # ── Decide if fetch is needed ─────────────────────────────────────────────
    now_ts     = time.time()
    last_fetch = float(st.session_state.get('live_last_fetch_ts', 0.0))
    need_fetch = (
        manual_refresh_btn
        or st.session_state.get('live_manual_refresh', False)
        or st.session_state.get('live_cache') is None
        or (st.session_state['live_monitoring']
            and (now_ts - last_fetch) >= float(refresh_interval))
    )
    if manual_refresh_btn or st.session_state.get('live_manual_refresh'):
        st.session_state['live_manual_refresh'] = False

    if need_fetch:
        try:
            df_live = fetch_yfinance_data(symbol, timeframe, period, rate_limit_sleep=1.5)
        except Exception as exc:
            st.error(f"Fetch error: {exc}")
            df_live = None

        if df_live is not None and len(df_live) >= 10:
            df_completed   = df_live.iloc[:-1].reset_index(drop=True)
            current_candle = df_live.iloc[-1]
            last_candle    = df_completed.iloc[-1]
            df_sig, _      = generate_signals(df_completed, params)
            last_sig_row   = df_sig.iloc[-1]
            sig            = int(last_sig_row['signal'])
            ltp            = float(current_candle['Close'])
            entry_est      = float(last_candle['Close'])
            sl_v = entry_est * (1 - params['sl_pct']) if sig == 1 else entry_est * (1 + params['sl_pct'])
            tp_v = entry_est * (1 + params['tp_pct']) if sig == 1 else entry_est * (1 - params['tp_pct'])

            st.session_state['live_cache'] = {
                'df_live': df_live, 'df_completed': df_completed,
                'df_sig': df_sig, 'last_candle': last_candle,
                'current_candle': current_candle,
                'sig': sig, 'reason': last_sig_row.get('reason', ''),
                'ltp': ltp, 'entry_est': entry_est, 'sl': sl_v, 'tp': tp_v,
                'fetch_time': pd.Timestamp.now(tz=IST).strftime('%Y-%m-%d %H:%M:%S %Z'),
                'symbol': symbol, 'timeframe': timeframe, 'display_name': display_name,
            }
            st.session_state['live_last_fetch_ts'] = time.time()

            # Auto SL/TP check on open position
            pos = st.session_state.get('live_position')
            if pos:
                d, ep = pos['direction'], pos['entry_price']
                hit = None
                if d == 'long':
                    if ltp <= pos['sl']:   hit = 'sl_hit'
                    elif ltp >= pos['tp']: hit = 'tp_hit'
                else:
                    if ltp >= pos['sl']:   hit = 'sl_hit'
                    elif ltp <= pos['tp']: hit = 'tp_hit'
                if hit:
                    _close_position(pos, ltp, hit, dhan_enabled, st.session_state.get('dhan_cfg'))
                    st.toast(f"Position auto-closed: {hit} @ {ltp:.4f}", icon="🔔")

            # Auto-execute new signal when monitoring is on and no position
            if (st.session_state['live_monitoring']
                    and sig != 0
                    and st.session_state.get('live_position') is None):
                direction = 'long' if sig == 1 else 'short'
                pos = _open_position(direction, ltp, params, live_capital, live_risk_pct,
                                     symbol, timeframe, dhan_enabled, dhan_cfg)
                st.session_state['live_position'] = pos
                st.toast(f"Auto-entry: {direction.upper()} @ {ltp:.4f}", icon="🚀")

    # ── Render from cache ─────────────────────────────────────────────────────
    cache    = st.session_state.get('live_cache')
    position = st.session_state.get('live_position')   # may have been updated above

    if cache is None:
        st.info("Press **▶ Start Monitoring** or **🔄 Refresh Now** to load data.")
    else:
        ltp      = cache['ltp']
        entry_e  = cache['entry_est']
        sl_disp  = position['sl']          if position else cache['sl']
        tp_disp  = position['tp']          if position else cache['tp']
        ep_disp  = position['entry_price'] if position else entry_e
        sig      = cache['sig']
        reason   = cache.get('reason', '')
        last_c   = cache['last_candle']
        curr_c   = cache['current_candle']

        st.markdown(
            f"**Fetched (IST):** `{cache['fetch_time']}`  |  "
            f"**Symbol:** `{cache['display_name']}` [{cache['timeframe']}]")

        # ── Last completed candle row ─────────────────────────────────────────
        candle_date_str = last_c['Date'].strftime('%Y-%m-%d %H:%M:%S %Z') if hasattr(last_c['Date'], 'strftime') else str(last_c['Date'])
        with st.container():
            st.markdown("##### 🕯️ Last Completed Candle")
            lc1, lc2, lc3, lc4, lc5, lc6, lc7 = st.columns(7)
            lc1.metric("Date / Time", candle_date_str[:16])
            lc2.metric("Open",   f"{last_c['Open']:.4f}")
            lc3.metric("High",   f"{last_c['High']:.4f}")
            lc4.metric("Low",    f"{last_c['Low']:.4f}")
            lc5.metric("Close",  f"{last_c['Close']:.4f}",
                       delta=f"{last_c['Close'] - last_c['Open']:+.4f}")
            vol_val = last_c.get('Volume', float('nan'))
            lc6.metric("Volume", f"{vol_val:,.0f}" if not pd.isna(vol_val) else "—")
            sig_labels = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⏸ NONE"}
            lc7.metric("Signal", sig_labels.get(sig, "—"))

        st.markdown("---")

        # ── Live metrics bar ──────────────────────────────────────────────────
        st.markdown("##### 📊 Live Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("LTP (forming candle)", f"{ltp:.4f}")
        m2.metric("Est. Entry",           f"{ep_disp:.4f}", help="Next candle open estimate")
        m3.metric("Stop Loss",  f"{sl_disp:.4f}",
                  delta=f"{((sl_disp / ep_disp) - 1) * 100:.2f}%" if ep_disp else None,
                  delta_color='inverse')
        m4.metric("Target",     f"{tp_disp:.4f}",
                  delta=f"{((tp_disp / ep_disp) - 1) * 100:.2f}%" if ep_disp else None)

        if position:
            d   = position['direction']
            ep  = position['entry_price']
            qty = position.get('qty', 1)
            pnl_pts = (ltp - ep) if d == 'long' else (ep - ltp)
            pnl_pct = pnl_pts / ep * 100
            pnl_amt = pnl_pts * qty
            m5.metric("Live P&L", f"₹ {pnl_amt:+.2f}",
                      delta=f"{pnl_pts:+.4f} pts ({pnl_pct:+.2f}%)",
                      delta_color='normal' if pnl_pts >= 0 else 'inverse')
        else:
            m5.metric("Live P&L", "—")

        # Dhan response
        if position and position.get('dhan_entry_response'):
            with st.expander("📤 Dhan Entry Response", expanded=False):
                st.json(position['dhan_entry_response'])

        # ── Signal box ───────────────────────────────────────────────────────
        st.markdown("---")
        if sig == 0:
            st.warning("⏸ No signal on last completed candle. Watching for next opportunity…")
        else:
            dir_label = "🟢 LONG (Buy)" if sig == 1 else "🔴 SHORT (Sell)"
            st.subheader(f"Signal: {dir_label}")
            rr = abs(cache['tp'] - entry_e) / max(abs(entry_e - cache['sl']), 1e-9)
            qty_hint = ""
            if live_capital > 0 and abs(entry_e - cache['sl']) > 0:
                qty_hint = (f"  |  Suggested Qty: **"
                            f"{max(1, int((live_capital * live_risk_pct / 100) // abs(entry_e - cache['sl'])))}**")
            st.caption(f"R:R ≈ 1:{rr:.2f}{qty_hint}  |  Reason: `{reason or 'N/A'}`")
            with st.expander("📖 Why this signal?", expanded=False):
                st.markdown(explain_signal_reason(reason, 'long' if sig == 1 else 'short',
                                                   entry_e, cache['sl'], cache['tp']))
            st.info("⚡ **Entry rule:** Wait for the next candle to OPEN, then place your order.")

        # ── Candlestick chart ─────────────────────────────────────────────────
        st.markdown("---")
        chart_entry = position['entry_price'] if position else None
        chart_sl    = position['sl']          if position else None
        chart_tp    = position['tp']          if position else None
        fig = draw_candlestick_chart(
            cache['df_live'],
            title=f"{display_name} — {timeframe}  |  LTP: {ltp:.4f}",
            sl=chart_sl, tp=chart_tp, entry=chart_entry, n_candles=80)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Recent candles table ──────────────────────────────────────────────
        with st.expander("📋 Recent completed candles (last 20)", expanded=False):
            disp     = cache['df_completed'].tail(20).copy()
            disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d %H:%M %Z')
            sig_tail = cache['df_sig'].tail(20)[['signal', 'reason']].reset_index(drop=True)
            disp     = disp.reset_index(drop=True)
            disp['signal'] = sig_tail['signal']
            disp['reason'] = sig_tail['reason']
            # Highlight the last row
            def highlight_last(row):
                return ['background-color: #1a3a1a' if row.name == len(disp) - 1 else '' for _ in row]
            st.dataframe(disp.style.apply(highlight_last, axis=1), use_container_width=True)

    # ── Auto-refresh scheduling ───────────────────────────────────────────────
    if st.session_state['live_monitoring']:
        elapsed   = time.time() - float(st.session_state.get('live_last_fetch_ts', 0.0))
        remaining = max(1.5, float(refresh_interval) - elapsed)
        st.caption(f"⏳ Next refresh in ~{remaining:.0f} s")
        time.sleep(remaining)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Trade History Tab
# ─────────────────────────────────────────────────────────────────────────────

def render_trade_history_tab():
    st.header("📒 Trade History")
    history = st.session_state.get('trade_history', [])

    if not history:
        st.info("No trades logged yet. Trades appear here after Square Off or auto SL/TP closure.")
        return

    df_hist = pd.DataFrame(history)
    total    = len(df_hist)
    wins     = int((df_hist['pnl_points'] > 0).sum())
    losses   = total - wins
    win_rt   = wins / total * 100 if total else 0
    tot_pts  = float(df_hist['pnl_points'].sum())
    avg_pts  = float(df_hist['pnl_points'].mean())

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Total Trades",   total)
    s2.metric("Winners",        wins)
    s3.metric("Losers",         losses)
    s4.metric("Win Rate",       f"{win_rt:.1f}%")
    s5.metric("Total P&L Pts",  f"{tot_pts:+.2f}")
    s6.metric("Avg P&L Pts",    f"{avg_pts:+.2f}")

    st.markdown("---")
    show_cols = [c for c in ['entry_time', 'exit_time', 'symbol', 'timeframe', 'direction',
                              'entry_price', 'exit_price', 'sl', 'tp',
                              'pnl_points', 'pnl_pct', 'exit_reason', 'qty'] if c in df_hist.columns]
    df_show = df_hist[show_cols].copy()
    if 'pnl_points' in df_show.columns:
        df_show['pnl_points'] = df_show['pnl_points'].map(lambda x: f"{x:+.4f}")
    if 'pnl_pct' in df_show.columns:
        df_show['pnl_pct'] = df_show['pnl_pct'].map(lambda x: f"{x:+.2f}%")
    st.dataframe(df_show, use_container_width=True)

    buf = io.StringIO()
    df_hist.to_csv(buf, index=False)
    st.download_button("⬇ Download Trade Log (CSV)", data=buf.getvalue(),
                        file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv')
    if st.button("🗑 Clear Trade History"):
        st.session_state['trade_history'] = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def app():
    st.title("Swing Trading Recommender — Price Action + Auto Optimisation")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        use_yfinance = st.checkbox("Use yfinance live data", value=True,
                                    help="Uncheck to upload a CSV/Excel file instead.")
        if use_yfinance:
            if not YFINANCE_AVAILABLE:
                st.error("yfinance not installed. Run: pip install yfinance")
                use_yfinance = False
            else:
                st.subheader("yfinance Settings")
                yfin_instr  = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key='yfin_instr')
                if INSTRUMENTS[yfin_instr] == '__custom__':
                    yfin_symbol  = st.text_input("Custom Symbol", value='TATAMOTORS.NS',
                                                  key='yfin_custom').strip().upper()
                    yfin_display = yfin_symbol
                else:
                    yfin_symbol  = INSTRUMENTS[yfin_instr]
                    yfin_display = yfin_instr
                yfin_tf        = st.selectbox("Timeframe", TIMEFRAMES, index=6, key='yfin_tf')
                max_days_bt    = _INTERVAL_MAX_DAYS.get(yfin_tf, 99999)
                compat_bt      = [p for p in PERIODS if _PERIOD_DAYS[p] <= max_days_bt] or ['1d']
                yfin_period    = st.selectbox("Period", compat_bt,
                                               index=min(4, len(compat_bt) - 1), key='yfin_period')
                st.caption(f"Symbol: `{yfin_symbol}`")

        st.markdown("---")
        upload = None if use_yfinance else st.file_uploader("Upload CSV / Excel",
                                                             type=['csv', 'xlsx', 'xls'])

        st.subheader("Optimisation")
        side_opt         = st.selectbox("Side", ['both', 'long', 'short'])
        search_type      = st.selectbox("Search method", ['random', 'grid'])
        random_iters     = st.number_input("Random iterations", min_value=10, max_value=2000, value=60)
        desired_accuracy = st.slider("Desired win rate", 0.0, 1.0, 0.7)
        min_trades       = st.number_input("Min trades required", min_value=1, max_value=500, value=5)
        target_points    = st.number_input("Target points per trade", min_value=1, value=50)
        use_pts_bt       = st.checkbox("Use target points (not % TP)", value=False)
        capital          = st.number_input("Capital for sizing (0 = off)", min_value=0, value=0)
        risk_pct         = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0)

        st.markdown("---")
        if search_type == 'grid':
            st.markdown("Grid params:")
            gw_pivot = st.multiselect("pivot_window", [2, 3, 4, 5], default=[3, 4])
            gw_sl    = st.multiselect("sl_pct", [0.005, 0.01, 0.02], default=[0.01])
            gw_tp    = st.multiselect("tp_pct", [0.01, 0.02, 0.05],  default=[0.02])
            grid = {'pivot_window': gw_pivot or [3], 'sl_pct': gw_sl or [0.01],
                    'tp_pct': gw_tp or [0.02], 'cluster_tol': [0.005],
                    'zone_width': [0.005], 'max_hold': [5]}
        else:
            grid = None

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_bt, tab_live, tab_hist = st.tabs([
        "📊 Backtest & Optimisation", "🔴 Live Trading", "📒 Trade History"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Backtest
    # ══════════════════════════════════════════════════════════════════════════
    with tab_bt:
        df = None
        if use_yfinance:
            with st.spinner(f"Fetching {yfin_display} [{yfin_tf}]…"):
                try:
                    df = fetch_yfinance_data(yfin_symbol, yfin_tf, yfin_period)
                    st.success(f"Loaded {len(df)} candles — {yfin_display} [{yfin_tf}] period={yfin_period}")
                except Exception as exc:
                    st.error(f"Fetch failed: {exc}"); st.stop()
        else:
            if upload is None:
                st.info("Upload a file or enable yfinance in the sidebar."); st.stop()
            try:
                raw = (pd.read_csv(upload) if upload.name.endswith('.csv')
                       else pd.read_excel(upload))
                df, mapping = standardize_df(raw)
                st.subheader("Detected columns"); st.json(mapping)
            except Exception as exc:
                st.error(f"File error: {exc}"); st.stop()

        if df is None or df.empty:
            st.warning("No data loaded."); st.stop()

        # Data preview
        st.subheader("Data preview")
        df_disp = df.copy()
        try:
            df_disp['Date'] = df_disp['Date'].dt.tz_convert(IST).dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        pc1, pc2 = st.columns(2)
        with pc1: st.caption("Top 5");    st.dataframe(df_disp.head())
        with pc2: st.caption("Bottom 5"); st.dataframe(df_disp.tail())
        st.write(f"Date range: `{df['Date'].min()}` → `{df['Date'].max()}`  |  "
                 f"Close: `{df['Close'].min():.4f}` → `{df['Close'].max():.4f}`")

        # Price chart
        st.subheader("Price chart")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df['Date'], df['Close'])
        ax.set_title("Close Price (IST)")
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig); plt.close(fig)

        # Backtest date range
        max_date = df['Date'].max(); min_date = df['Date'].min()
        user_end  = st.date_input("End date for backtest", value=max_date.date(),
                                   min_value=min_date.date(), max_value=max_date.date())
        end_dt    = pd.Timestamp(user_end, tz=IST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_train  = df[df['Date'] <= end_dt].reset_index(drop=True)
        st.write(f"Backtest rows: {len(df_train)}")

        # EDA heatmap
        st.subheader("Year × Month Returns Heatmap")
        df_train = df_train.copy()
        df_train['returns'] = df_train['Close'].pct_change()
        df_train['year']    = df_train['Date'].dt.year
        df_train['month']   = df_train['Date'].dt.month
        try:
            heat = df_train.pivot_table(values='returns', index='year', columns='month',
                                         aggfunc=lambda x: (x + 1.0).prod() - 1) * 100
            fig2, ax2 = plt.subplots(figsize=(10, max(3, 0.5 * len(heat.index))))
            im = ax2.imshow(heat.fillna(0).values, aspect='auto')
            ax2.set_xticks(range(len(heat.columns))); ax2.set_xticklabels(heat.columns)
            ax2.set_yticks(range(len(heat.index)));   ax2.set_yticklabels(heat.index)
            for (j, k), val in np.ndenumerate(heat.fillna(0).values):
                ax2.text(k, j, f"{val:.1f}%", ha='center', va='center', fontsize=7)
            fig2.colorbar(im, ax=ax2, fraction=0.046)
            st.pyplot(fig2); plt.close(fig2)
        except Exception as exc:
            st.warning(f"Heatmap error: {exc}")

        st.subheader("Automated summary"); st.write(generate_summary(df_train))

        # Optimisation
        st.subheader("Run Optimisation")
        if st.button("▶ Start Optimisation"):
            pb   = st.progress(0); stxt = st.empty()
            def cb(done, total):
                pb.progress(int(done / total * 100) if total else 0)
                stxt.text(f"{done}/{total}")
            allowed_dirs = []
            if side_opt in ('both', 'long'):  allowed_dirs.append('long')
            if side_opt in ('both', 'short'): allowed_dirs.append('short')
            with st.spinner("Searching for best strategy…"):
                best, tried = find_best_strategy(
                    df_train, search_type=search_type, random_iters=random_iters, grid=grid,
                    allowed_dirs=allowed_dirs, desired_accuracy=desired_accuracy,
                    min_trades=min_trades, progress_callback=cb,
                    use_points=use_pts_bt,
                    target_points=target_points if use_pts_bt else None)
            pb.progress(100); stxt.success("Done")

            if best is None:
                st.warning("No strategy met the filters. Relax constraints or add iterations.")
                st.stop()

            st.session_state['best_params'] = best['params']
            st.success(f"Best strategy found  (tried {tried} combos)")
            col_p, col_s = st.columns(2)
            with col_p: st.subheader("Params");  st.json(best['params'])
            with col_s: st.subheader("Stats");   st.json(best['stats'])

            trades_df = best['trades'].copy()
            if not trades_df.empty:
                for col in ('entry_time', 'exit_time'):
                    trades_df[col] = (trades_df[col]
                                      .dt.tz_convert(IST)
                                      .dt.strftime('%Y-%m-%d %H:%M:%S %Z'))
            st.write("Sample trades (first 100)")
            st.dataframe(trades_df.head(100))

            bh_points = (df_train['Close'].iloc[-1] - df_train['Close'].iloc[0]
                         if len(df_train) >= 2 else 0)
            bh_pct    = ((df_train['Close'].iloc[-1] / df_train['Close'].iloc[0] - 1) * 100
                         if len(df_train) >= 2 else 0)
            strat_pts = best['stats'].get('total_points', 0)

            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("Buy-and-hold pts", f"{bh_points:.2f}")
            bm2.metric("Buy-and-hold %",   f"{bh_pct:.2f}%")
            bm3.metric("Strategy pts",     f"{strat_pts:.2f}")

            df_full_sig, meta = generate_signals(df_train, best['params'])
            rec = generate_live_recommendation(
                df_full_sig, best['params'],
                target_points=target_points, backtest_stats=best['stats'],
                capital=capital, risk_pct=risk_pct)

            st.subheader("Live Recommendation (last completed candle)")
            if rec is None:
                st.info("No signal at last candle.")
            else:
                st.json(rec)
                # Human-readable rationale
                st.subheader("📖 Signal Rationale")
                st.markdown(explain_signal_reason(
                    rec.get('reason', ''),
                    rec['direction'],
                    rec['estimated_entry_price'],
                    rec['stop_loss'],
                    rec['target_price'],
                    accuracy=rec.get('probability_of_profit')))

            st.subheader("Pattern summary")
            st.json({k: len(v) for k, v in meta['patterns'].items()})
            st.subheader("Backtest summary (human readable)")
            st.write(backtest_human_readable(best['stats'], best['params'], bh_points))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Live Trading
    # ══════════════════════════════════════════════════════════════════════════
    with tab_live:
        live_trading_section()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Trade History
    # ══════════════════════════════════════════════════════════════════════════
    with tab_hist:
        render_trade_history_tab()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app()
