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

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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
# Live Trading Section
# ─────────────────────────────────────────────────────────────────────────────

def live_trading_section(sidebar_use_yfinance):
    """Live monitoring tab — fetches latest data, runs strategy, shows next-candle recommendation."""

    st.header("🔴 Live Trading Monitor")
    st.caption("Signal detection runs on last completed candle. Entry executes at the OPEN of the following candle.")

    # ── Instrument / Timeframe selectors ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        instrument_name = st.selectbox("Instrument", list(INSTRUMENTS.keys()),
                                       key='live_instrument')
        symbol = INSTRUMENTS[instrument_name]
    with col2:
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=6, key='live_tf')
    with col3:
        live_period_options = [p for p in PERIODS
                               if _PERIOD_DAYS[p] <= _INTERVAL_MAX_DAYS.get(timeframe, 99999)]
        if not live_period_options:
            live_period_options = ['1d']
        period = st.selectbox("Period (for strategy context)", live_period_options,
                              index=min(2, len(live_period_options) - 1), key='live_period')

    # ── Strategy params: use optimised or defaults ───────────────────────────
    best_params = st.session_state.get('best_params')
    if best_params:
        st.success("✅ Using optimised params from Backtest tab.")
        params = best_params
    else:
        st.info("ℹ️ No optimised params found. Using defaults. Run optimisation in the Backtest tab first for best results.")
        params = DEFAULT_PARAMS.copy()

    # Allow side override
    side_live = st.selectbox("Direction filter", ['both', 'long', 'short'], key='live_side')
    allowed_dirs = (['long', 'short'] if side_live == 'both'
                    else [side_live])
    params = {**params, 'allowed_dirs': allowed_dirs}

    # ── Capital / Risk ────────────────────────────────────────────────────────
    col4, col5 = st.columns(2)
    with col4:
        live_capital  = st.number_input("Capital for sizing (0 = off)", min_value=0, value=0, key='live_cap')
    with col5:
        live_risk_pct = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0, key='live_risk')

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    col6, col7 = st.columns(2)
    with col6:
        auto_refresh = st.toggle("Auto-refresh", value=False, key='live_auto')
    with col7:
        refresh_interval = st.number_input("Refresh interval (seconds, min 10)",
                                           min_value=10, max_value=3600, value=60, key='live_interval')

    refresh_btn = st.button("🔄 Refresh Now", key='live_refresh_btn')

    # ── Fetch & analyse ───────────────────────────────────────────────────────
    should_run = refresh_btn or auto_refresh or ('live_first_run' not in st.session_state)
    st.session_state['live_first_run'] = True

    status_placeholder = st.empty()
    result_placeholder = st.empty()

    if should_run:
        with st.spinner(f"Fetching {instrument_name} [{timeframe}] — respecting 1.5s rate limit…"):
            try:
                df_live = fetch_yfinance_data(symbol, timeframe, period, rate_limit_sleep=1.5)
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")
                return

        if df_live.empty or len(df_live) < 10:
            st.warning("Not enough data to generate a signal. Try a longer period.")
            return

        # Use all candles except the last (which may still be forming)
        df_completed = df_live.iloc[:-1].reset_index(drop=True)
        last_complete_candle = df_completed.iloc[-1]
        current_candle       = df_live.iloc[-1]

        # Run signal generation
        df_sig, meta = generate_signals(df_completed, params)
        last_sig_row = df_sig.iloc[-1]
        sig          = last_sig_row['signal']

        now_ist = datetime.now(tz=pd.Timestamp.now(tz=IST).tzinfo)

        with result_placeholder.container():
            st.markdown(f"**Last updated (IST):** `{pd.Timestamp.now(tz=IST).strftime('%Y-%m-%d %H:%M:%S %Z')}`")
            st.markdown(f"**Last completed candle:** `{last_complete_candle['Date'].strftime('%Y-%m-%d %H:%M:%S %Z')}`")
            st.markdown(f"**Current forming candle open:** `{current_candle['Open']:.4f}`")

            st.markdown("---")

            if sig == 0:
                st.warning("⏸ No signal on last completed candle. No trade recommendation.")
            else:
                direction_emoji = "🟢 LONG" if sig == 1 else "🔴 SHORT"
                st.subheader(f"Signal Detected: {direction_emoji}")

                entry_est = last_complete_candle['Close']
                if params.get('use_target_points') and params.get('target_points') is not None:
                    tp_live = entry_est + params['target_points'] if sig == 1 else entry_est - params['target_points']
                else:
                    tp_live = entry_est * (1 + params['tp_pct']) if sig == 1 else entry_est * (1 - params['tp_pct'])
                sl_live = entry_est * (1 - params['sl_pct']) if sig == 1 else entry_est * (1 + params['sl_pct'])

                unit_risk = abs(entry_est - sl_live)
                qty = None
                if live_capital > 0 and unit_risk > 0:
                    qty = int((live_capital * live_risk_pct / 100) // unit_risk)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Est. Entry (next candle open)", f"{entry_est:.4f}", help="Approximate — actual entry = next candle open price")
                m2.metric("Stop Loss", f"{sl_live:.4f}", delta=f"{((sl_live/entry_est)-1)*100:.2f}%")
                m3.metric("Target", f"{tp_live:.4f}", delta=f"{((tp_live/entry_est)-1)*100:.2f}%")
                if qty is not None:
                    m4.metric("Suggested Qty", str(qty))

                rr = abs(tp_live - entry_est) / max(unit_risk, 1e-9)
                st.caption(f"Risk:Reward ≈ 1 : {rr:.2f}  |  Unit risk: {unit_risk:.4f}  |  Signal reason: {last_sig_row['reason'] or 'N/A'}")

                st.info(
                    "⚡ **Entry rule:** Place your order at the OPEN of the next candle. "
                    "Do NOT enter at current close — wait for the next bar to open."
                )

            # ── Recent candles table ───────────────────────────────────────
            with st.expander("Recent completed candles (last 20)"):
                disp = df_completed.tail(20).copy()
                disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                sig_disp = df_sig.tail(20)[['signal', 'reason']].reset_index(drop=True)
                disp = disp.reset_index(drop=True)
                disp['signal'] = sig_disp['signal']
                disp['reason'] = sig_disp['reason']
                st.dataframe(disp, use_container_width=True)

            # ── Mini price chart ───────────────────────────────────────────
            with st.expander("Price chart (last 100 candles)"):
                chart_df = df_live.tail(100)
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(chart_df['Date'], chart_df['Close'], linewidth=1)
                ax.set_title(f"{instrument_name} — {timeframe}")
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
                fig.autofmt_xdate()
                st.pyplot(fig)
                plt.close(fig)

        # ── Auto-refresh wait ──────────────────────────────────────────────
        if auto_refresh:
            # fetch_yfinance_data already slept 1.5 s; wait the remainder
            wait_s = max(1.5, refresh_interval - 1.5)
            status_placeholder.info(f"⏳ Next refresh in {int(wait_s)} seconds…")
            time.sleep(wait_s)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def app():
    st.title("Swing Trading Recommender — Price Action + Auto Optimisation")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        # ── yfinance data source ─────────────────────────────────────────────
        use_yfinance = st.checkbox(
            "Use yfinance live data",
            value=True,
            help="Fetch OHLCV data directly from Yahoo Finance. Uncheck to upload your own CSV/Excel."
        )

        if use_yfinance:
            if not YFINANCE_AVAILABLE:
                st.error("yfinance not installed. Run: pip install yfinance")
                use_yfinance = False
            else:
                st.subheader("yfinance Settings")
                yfin_instrument = st.selectbox("Instrument", list(INSTRUMENTS.keys()), key='yfin_instr')
                yfin_symbol     = INSTRUMENTS[yfin_instrument]
                yfin_tf         = st.selectbox("Timeframe", TIMEFRAMES, index=6, key='yfin_tf')

                # Filter periods compatible with chosen timeframe
                max_days_tf     = _INTERVAL_MAX_DAYS.get(yfin_tf, 99999)
                compat_periods  = [p for p in PERIODS if _PERIOD_DAYS[p] <= max_days_tf]
                if not compat_periods:
                    compat_periods = ['1d']
                yfin_period = st.selectbox("Period", compat_periods,
                                           index=min(4, len(compat_periods) - 1), key='yfin_period')

                st.caption(
                    f"Symbol: `{yfin_symbol}` | Max period for {yfin_tf}: "
                    f"{max_days_tf if yfin_tf in _INTERVAL_MAX_DAYS else '∞'} days"
                )

        st.markdown("---")

        if not use_yfinance:
            upload = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx', 'xls'])
        else:
            upload = None

        st.subheader("Strategy / Optimisation")
        side_opt       = st.selectbox("Side", ['both', 'long', 'short'])
        search_type    = st.selectbox("Search method", ['random', 'grid'])
        random_iters   = st.number_input("Random search iterations", min_value=10, max_value=2000, value=60)
        desired_accuracy = st.slider("Desired accuracy (win rate)", 0.0, 1.0, 0.7)
        min_trades     = st.number_input("Min trades required", min_value=1, max_value=500, value=5)
        target_points  = st.number_input("Target points per trade", min_value=1, value=50)
        use_pts_bt     = st.checkbox("Use target points (not % TP) in backtest", value=False)
        capital        = st.number_input("Capital for position sizing (0 = off)", min_value=0, value=0)
        risk_pct       = st.number_input("Risk % per trade", min_value=0.1, max_value=10.0, value=1.0)

        st.markdown("---")
        if search_type == 'grid':
            st.markdown("Grid parameters:")
            gw_pivot = st.multiselect("pivot_window", [2, 3, 4, 5], default=[3, 4])
            gw_sl    = st.multiselect("sl_pct",  [0.005, 0.01, 0.02], default=[0.01])
            gw_tp    = st.multiselect("tp_pct",  [0.01, 0.02, 0.05],  default=[0.02])
            grid = {
                'pivot_window': gw_pivot or [3], 'sl_pct': gw_sl or [0.01],
                'tp_pct': gw_tp or [0.02], 'cluster_tol': [0.005],
                'zone_width': [0.005], 'max_hold': [5],
            }
        else:
            grid = None

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_bt, tab_live = st.tabs(["📊 Backtest & Optimisation", "🔴 Live Trading"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Backtest & Optimisation
    # ════════════════════════════════════════════════════════════════════════
    with tab_bt:
        # ── Load data ────────────────────────────────────────────────────────
        df = None

        if use_yfinance:
            with st.spinner(f"Fetching {yfin_instrument} [{yfin_tf}] from yfinance…"):
                try:
                    df = fetch_yfinance_data(yfin_symbol, yfin_tf, yfin_period)
                    st.success(f"Loaded {len(df)} candles for {yfin_instrument} ({yfin_symbol}) "
                               f"[{yfin_tf}] — period: {yfin_period}")
                except Exception as exc:
                    st.error(f"yfinance fetch failed: {exc}")
                    return
        else:
            if upload is None:
                st.info("Upload a CSV/Excel file or enable yfinance data source in the sidebar.")
                return
            try:
                raw = pd.read_csv(upload) if upload.name.endswith('.csv') else pd.read_excel(upload)
            except Exception as exc:
                st.error(f"Failed to read file: {exc}")
                return
            try:
                df, mapping = standardize_df(raw)
                st.subheader("Mapped columns (detected)")
                st.json(mapping)
            except Exception as exc:
                st.error(f"Column mapping error: {exc}")
                return

        if df is None or df.empty:
            st.warning("No data loaded.")
            return

        # ── Data preview ─────────────────────────────────────────────────────
        st.subheader("Data preview")
        df_display = df.copy()
        try:
            df_display['Date'] = df_display['Date'].dt.tz_convert(IST).dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        c1, c2 = st.columns(2)
        with c1:
            st.write("Top 5 rows")
            st.dataframe(df_display.head())
        with c2:
            st.write("Bottom 5 rows")
            st.dataframe(df_display.tail())

        st.write("Date range:", df['Date'].min(), "→", df['Date'].max())
        st.write("Close range:", f"{df['Close'].min():.4f}", "→", f"{df['Close'].max():.4f}")

        # ── Price chart ───────────────────────────────────────────────────────
        st.subheader("Price chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df['Close'])
        ax.set_title("Close Price (IST)")
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        # ── Backtest date range ───────────────────────────────────────────────
        max_date = df['Date'].max()
        min_date = df['Date'].min()
        user_end = st.date_input(
            "End date for backtest (rows after this are excluded)",
            value=max_date.date(), min_value=min_date.date(), max_value=max_date.date()
        )

        end_dt = pd.Timestamp(user_end, tz=IST) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_train = df[df['Date'] <= end_dt].reset_index(drop=True)
        st.write(f"Rows for backtest: {len(df_train)} (up to {end_dt.date()})")

        # ── EDA ───────────────────────────────────────────────────────────────
        st.subheader("Exploratory Data Analysis")
        df_train['returns'] = df_train['Close'].pct_change()
        df_train['year']    = df_train['Date'].dt.year
        df_train['month']   = df_train['Date'].dt.month

        st.write("Year × Month returns heatmap (values in %)")
        try:
            heat = df_train.pivot_table(
                values='returns', index='year', columns='month',
                aggfunc=lambda x: (x + 1.0).prod() - 1
            ) * 100
            fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.5 * len(heat.index))))
            im = ax2.imshow(heat.fillna(0).values, aspect='auto', interpolation='nearest')
            ax2.set_xticks(range(len(heat.columns))); ax2.set_xticklabels(heat.columns)
            ax2.set_yticks(range(len(heat.index)));   ax2.set_yticklabels(heat.index)
            ax2.set_title("Year-Month Returns (%)")
            for (j, k), val in np.ndenumerate(heat.fillna(0).values):
                ax2.text(k, j, f"{val:.2f}%", ha='center', va='center', fontsize=8)
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as exc:
            st.warning(f"Could not generate heatmap: {exc}")
            tbl = df_train.groupby(['year', 'month'])['returns'].apply(
                lambda x: (x + 1.0).prod() - 1).unstack(fill_value=0) * 100
            st.dataframe(tbl)

        st.subheader("Automated summary")
        st.write(generate_summary(df_train))

        # ── Optimisation ─────────────────────────────────────────────────────
        st.subheader("Run Optimisation")
        if st.button("Start Optimisation"):
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def progress_cb(done, total):
                pct = int(done / total * 100) if total else 0
                progress_bar.progress(pct)
                status_text.text(f"Progress: {pct}% ({done}/{total})")

            allowed_dirs = []
            if side_opt in ('both', 'long'):  allowed_dirs.append('long')
            if side_opt in ('both', 'short'): allowed_dirs.append('short')

            with st.spinner("Searching for best strategy…"):
                best, tried = find_best_strategy(
                    df_train, search_type=search_type, random_iters=random_iters,
                    grid=grid, allowed_dirs=allowed_dirs,
                    desired_accuracy=desired_accuracy, min_trades=min_trades,
                    progress_callback=progress_cb,
                    use_points=use_pts_bt,
                    target_points=target_points if use_pts_bt else None,
                )

            progress_bar.progress(100)
            status_text.success("Optimisation completed")

            if best is None:
                st.warning("No strategy met the filters. Relax filters or increase iterations.")
                return

            # ── Save to session state for Live tab ────────────────────────
            st.session_state['best_params'] = best['params']

            st.success("Best strategy found")
            st.write(f"Combinations tried: {tried}")

            st.write("Strategy params:")
            st.json(best['params'])
            st.write("Backtest stats:")
            st.json(best['stats'])

            trades_df = best['trades'].copy()
            if not trades_df.empty:
                for col in ('entry_time', 'exit_time'):
                    trades_df[col] = (trades_df[col]
                                      .dt.tz_convert(IST)
                                      .dt.strftime('%Y-%m-%d %H:%M:%S %Z'))

            st.write("Sample trades (first 100)")
            st.dataframe(trades_df.head(100))

            if len(df_train) >= 2:
                bh_points = df_train['Close'].iloc[-1] - df_train['Close'].iloc[0]
                bh_pct    = (df_train['Close'].iloc[-1] / df_train['Close'].iloc[0] - 1) * 100
            else:
                bh_points = bh_pct = 0

            strat_pts = best['stats'].get('total_points', 0)
            pct_more  = (strat_pts - bh_points) / (abs(bh_points) if bh_points != 0 else 1) * 100

            st.metric("Buy-and-hold points", f"{bh_points:.2f}")
            st.metric("Buy-and-hold return %", f"{bh_pct:.2f}%")
            st.metric("Strategy total points", f"{strat_pts:.2f}")
            st.write(f"Strategy gave {pct_more:.2f}% more points vs buy-and-hold.")

            df_full_sig, meta = generate_signals(df_train, best['params'])
            rec = generate_live_recommendation(
                df_full_sig, best['params'],
                target_points=target_points,
                backtest_stats=best['stats'],
                capital=capital, risk_pct=risk_pct
            )

            st.subheader("Live Recommendation (last closed candle)")
            if rec is None:
                st.write("No signal at last candle.")
            else:
                st.json(rec)

            st.subheader("Pattern summary")
            st.json({k: len(v) for k, v in meta['patterns'].items()})

            st.subheader("Backtest summary (human readable)")
            st.write(backtest_human_readable(best['stats'], best['params'], bh_points))

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Live Trading
    # ════════════════════════════════════════════════════════════════════════
    with tab_live:
        live_trading_section(sidebar_use_yfinance=use_yfinance)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app()
