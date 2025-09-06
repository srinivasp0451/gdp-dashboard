# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from math import isfinite

st.set_page_config(layout="wide")

# ------------------- Helpers / Robust Column Mapping & Normalization -------------------
def normalize_df(df):
    """
    Robust normalization:
      - maps any column name containing 'open', 'high', 'low', 'close', 'vol', 'share', 'traded', 'date', 'time' to canonical names.
      - strips commas, spaces, currency symbols from numeric fields then converts to numeric.
      - sets index to Date (if available) and sorts.
    """
    df = df.copy()
    # build mapping tolerant to many forms
    mapping = {}
    for orig in df.columns:
        key = str(orig).strip().lower()
        if 'open' in key and 'open' not in mapping.values():
            mapping[orig] = "Open"
        elif 'high' in key and 'high' not in mapping.values():
            mapping[orig] = "High"
        elif 'low' in key and 'low' not in mapping.values():
            mapping[orig] = "Low"
        elif 'close' in key and 'close' not in mapping.values():
            mapping[orig] = "Close"
        elif any(tok in key for tok in ['vol','volume','shares','traded','turnover']) and 'Volume' not in mapping.values():
            mapping[orig] = "Volume"
        elif any(tok in key for tok in ['date','datetime','time','timestamp']) and 'Date' not in mapping.values():
            mapping[orig] = "Date"
        # else leave unmapped (user may have extra cols we keep)

    df = df.rename(columns=mapping)

    # ensure required cols
    for required in ["Open","High","Low","Close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    # parse date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        # fallback more lenient parsing
        if df["Date"].isna().any():
            df["Date"] = pd.to_datetime(df["Date"].astype(str), dayfirst=False, errors='coerce')
        df = df.dropna(subset=["Date"]).copy()
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
    else:
        # try to coerce index to datetime
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(axis=0, subset=[df.index.name])
        df = df.sort_index()

    # Clean numeric columns: remove commas, spaces, currency symbols then convert
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
            df[col] = df[col].str.replace(r'[₹$€£,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows with missing price data
    df = df.dropna(subset=["Open","High","Low","Close"])

    return df

# ------------------- Technical Indicators (causal) -------------------
def compute_indicators(df, params):
    """
    Causal indicator computation (only uses past/current bars).
    Does not overwrite price columns; indicator columns forward/back filled where safe.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0)

    # SMA and EMA
    df[f"sma_{params['sma_fast']}"] = close.rolling(params['sma_fast'], min_periods=1).mean()
    df[f"sma_{params['sma_slow']}"] = close.rolling(params['sma_slow'], min_periods=1).mean()
    df[f"ema_{params['ema_fast']}"] = close.ewm(span=params['ema_fast'], adjust=False).mean()
    df[f"ema_{params['ema_slow']}"] = close.ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD
    ema_f = df[f"ema_{params['ema_fast']}"]
    ema_s = df[f"ema_{params['ema_slow']}"]
    macd = ema_f - ema_s
    macd_signal = macd.ewm(span=params.get('macd_signal', 9), adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal

    # RSI (Wilder)
    period = params['rsi_period']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Bollinger
    ma = df[f"sma_{params['sma_fast']}"]
    std = close.rolling(params['sma_fast'], min_periods=1).std()
    df['bb_upper'] = ma + params.get('bb_mult', 2) * std
    df['bb_lower'] = ma - params.get('bb_mult', 2) * std

    # Momentum
    df[f"mom_{params['mom_period']}"] = close - close.shift(params['mom_period'])

    # Stochastic
    stoch_k = ((close - low.rolling(params['stoch_period'], min_periods=1).min()) /
               (high.rolling(params['stoch_period'], min_periods=1).max() - low.rolling(params['stoch_period'], min_periods=1).min()).replace(0, np.nan)) * 100
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_k.rolling(3, min_periods=1).mean()

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).fillna(0).cumsum()
    df['obv'] = obv

    # VWMA & vol sma
    wsum = (close * vol).rolling(params.get('vwma_period', 14)).sum()
    vsum = vol.rolling(params.get('vwma_period', 14)).sum().replace(0, np.nan)
    df['vwma'] = (wsum / vsum).fillna(df['Close'])
    df['vol_sma'] = vol.rolling(params.get('vol_sma_period', 20), min_periods=1).mean()

    # CCI
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(params['cci_period'], min_periods=1).mean()
    md = tp.rolling(params['cci_period'], min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
    df['cci'] = (tp - tp_ma) / (0.015 * md)

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{params['atr_period']}"] = tr.rolling(params['atr_period'], min_periods=1).mean()

    # ADX / DI approximations
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_for_adx = tr.ewm(alpha=1/params['adx_period'], adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/params['adx_period'], adjust=False).mean() / atr_for_adx)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/params['adx_period'], adjust=False).mean() / atr_for_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx'] = dx.ewm(alpha=1/params['adx_period'], adjust=False).mean()
    df['pdi'] = plus_di
    df['mdi'] = minus_di

    # fill indicator columns only
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_cols = [c for c in df.columns if c not in price_cols]
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill')

    return df

# ------------------- Candlestick & Basic/Multi-bar Patterns -------------------
def detect_candlestick_patterns(df, i):
    """
    Detects single and a few multi-bar candlestick patterns at index position i.
    Only uses current and past rows (no future).
    Returns list of pattern names.
    """
    patterns = []
    n = len(df)
    if i < 0 or i >= n:
        return patterns
    row = df.iloc[i]
    prev = df.iloc[i-1] if i-1 >= 0 else None
    prev2 = df.iloc[i-2] if i-2 >= 0 else None

    O = row['Open']; H = row['High']; L = row['Low']; C = row['Close']
    body = abs(C - O)
    rng = max(H - L, 1e-9)
    lower_shadow = min(O, C) - L
    upper_shadow = H - max(O, C)

    # Doji
    if body <= 0.1 * rng:
        patterns.append('DOJI')

    # Engulfing
    if prev is not None:
        if (prev['Close'] < prev['Open']) and (C > O) and (C >= prev['Open']) and (O <= prev['Close']):
            patterns.append('BULL_ENGULF')
        if (prev['Close'] > prev['Open']) and (C < O) and (C <= prev['Open']) and (O >= prev['Close']):
            patterns.append('BEAR_ENGULF')

    # Hammer / Hanging man / Shooting star / Inverted hammer
    if body > 0:
        if lower_shadow >= 2 * body and upper_shadow <= 0.5 * body:
            if C > O:
                patterns.append('HAMMER')
            else:
                patterns.append('HANGING_MAN')
        if upper_shadow >= 2 * body and lower_shadow <= 0.5 * body:
            if C < O:
                patterns.append('SHOOTING_STAR')
            else:
                patterns.append('INVERTED_HAMMER')

    # Inside bar
    if prev is not None:
        if (H < prev['High']) and (L > prev['Low']):
            patterns.append('INSIDE_BAR')

    # 3-bar Morning/Evening star simplified (conservative)
    if prev is not None and prev2 is not None:
        O1, C1 = prev2['Open'], prev2['Close']
        O2, C2 = prev['Open'], prev['Close']
        O3, C3 = row['Open'], row['Close']
        b1 = abs(C1 - O1); b2 = abs(C2 - O2); b3 = abs(C3 - O3)
        if (C1 < O1) and (b2 <= 0.6 * b1) and (C3 > O3) and (C3 > (O1 + C1) / 2):
            patterns.append('MORNING_STAR')
        if (C1 > O1) and (b2 <= 0.6 * b1) and (C3 < O3) and (C3 < (O1 + C1) / 2):
            patterns.append('EVENING_STAR')

    # 3 soldiers / crows
    if i >= 2:
        c0, c1, c2 = df['Close'].iloc[i-2:i+1]
        o0, o1, o2 = df['Open'].iloc[i-2:i+1]
        if (c0 < c1 < c2) and (o1 > c0) and (o2 > c1):
            patterns.append('THREE_WHITE')
        if (c0 > c1 > c2) and (o1 < c0) and (o2 < c1):
            patterns.append('THREE_BLACK')

    return patterns

# ------------------- Market Structure Utilities -------------------
def local_peaks(price_series, order=3):
    """
    Return boolean Series marking local peaks (uses only neighbor comparisons).
    order parameter is the number of bars on each side to compare.
    """
    s = price_series
    peaks = np.zeros(len(s), dtype=bool)
    for i in range(order, len(s) - order):
        left = s.iloc[i-order:i]
        right = s.iloc[i+1:i+1+order]
        if s.iloc[i] > left.max() and s.iloc[i] > right.max():
            peaks[i] = True
    return pd.Series(peaks, index=s.index)

def local_troughs(price_series, order=3):
    s = price_series
    troughs = np.zeros(len(s), dtype=bool)
    for i in range(order, len(s) - order):
        left = s.iloc[i-order:i]
        right = s.iloc[i+1:i+1+order]
        if s.iloc[i] < left.min() and s.iloc[i] < right.min():
            troughs[i] = True
    return pd.Series(troughs, index=s.index)

# ------------------- Simple peak/trough locator (positions) -------------------
def find_last_peaks_positions(prices, upto_i, n=3):
    """
    Return indices of last n peaks (strict local peaks) prior to upto_i (exclusive of upto_i).
    """
    res = []
    for j in range(upto_i-1, 0, -1):
        if j-1 >= 0 and j+1 < len(prices):
            if prices.iloc[j] > prices.iloc[j-1] and prices.iloc[j] > prices.iloc[j+1]:
                res.append(j)
                if len(res) >= n:
                    break
    return res[::-1]

def find_last_troughs_positions(prices, upto_i, n=3):
    res = []
    for j in range(upto_i-1, 0, -1):
        if j-1 >= 0 and j+1 < len(prices):
            if prices.iloc[j] < prices.iloc[j-1] and prices.iloc[j] < prices.iloc[j+1]:
                res.append(j)
                if len(res) >= n:
                    break
    return res[::-1]

# ------------------- Advanced Chart Patterns Heuristics -------------------
def detect_head_and_shoulders(df, i, lookback=60):
    """
    Heuristic H&S detection in the last lookback bars ending at i.
    Looks for peak-left shoulder-peak-head-peak-right pattern. Conservative checks.
    """
    patterns = []
    if i < 6:
        return patterns
    start = max(0, i - lookback + 1)
    window = df['Close'].iloc[start:i+1]
    prices = window.reset_index(drop=True)
    # find peaks
    peaks_idx = find_last_peaks_positions(prices, len(prices), n=5)
    # peaks relative positions needed: left shoulder < head > right shoulder, and head higher
    if len(peaks_idx) >= 3:
        # use most recent 3 peaks
        p = peaks_idx[-3:]
        p0, p1, p2 = p
        # ensure p0 < p1 < p2 (positions in window)
        if p0 < p1 < p2:
            val0, val1, val2 = prices.iloc[p0], prices.iloc[p1], prices.iloc[p2]
            # head must be highest and shoulders somewhat lower and similar
            if val1 > val0 and val1 > val2 and 0.8 * val1 < (val0 + val2)/2 + 1e-9:
                # confirm neckline roughly horizontal: troughs between peaks
                trough1_idx = p0 + 1 + np.argmin(prices.iloc[p0+1:p1]) if (p1 - p0 - 1) > 0 else None
                trough2_idx = p1 + 1 + np.argmin(prices.iloc[p1+1:p2]) if (p2 - p1 - 1) > 0 else None
                if trough1_idx is not None and trough2_idx is not None:
                    t1, t2 = prices.iloc[trough1_idx], prices.iloc[trough2_idx]
                    if abs(t1 - t2) / max(1e-9, (t1+t2)/2) < 0.12:
                        patterns.append('HEAD_SHOULDERS')
    return patterns

def detect_cup_and_handle(df, i, lookback=120):
    """
    Very rough cup & handle detection:
      - Find rounded bottom within lookback with left and right peaks roughly similar
      - Then a small handle pullback within recent bars
    This is an approximation and conservative (returns [] if unsure).
    """
    patterns = []
    if i < 40:
        return patterns
    start = max(0, i - lookback + 1)
    window = df['Close'].iloc[start:i+1].reset_index(drop=True)
    if len(window) < 40:
        return patterns
    # find global min and two sides
    mid_idx = window.idxmin()
    left_peak = window.iloc[:mid_idx].max() if mid_idx > 5 else None
    right_peak = window.iloc[mid_idx+1:].max() if mid_idx < (len(window)-6) else None
    if left_peak is None or right_peak is None:
        return patterns
    # peaks should be roughly similar and mid low is significantly lower
    if left_peak > 0 and right_peak > 0 and (abs(left_peak - right_peak) / max(left_peak, right_peak) < 0.18):
        trough = window.iloc[mid_idx]
        if trough < 0.85 * min(left_peak, right_peak):
            # now check for a small handle in last 15 bars: small pullback after right peak
            right_peak_idx = mid_idx + 1 + window.iloc[mid_idx+1:].idxmax() if mid_idx+1 < len(window) else None
            if right_peak_idx is not None and len(window) - 1 - right_peak_idx <= 20:
                # handle is last part where price pulls back < 15% of cup depth
                cup_depth = min(left_peak, right_peak) - trough
                if cup_depth > 0:
                    handle_depth = (window.iloc[-1] - min(window.iloc[right_peak_idx:])) if right_peak_idx < len(window) else 0
                    if handle_depth >= 0 and handle_depth <= 0.35 * cup_depth:
                        patterns.append('CUP_HANDLE')
    return patterns

def detect_flag_or_pennant(df, i, pole_lookback=10, consolidation_lookback=8, vol_multiplier=1.2):
    """
    Heuristic flag/pennant:
      - look for a strong pole (big move) immediately before a consolidation range with decreasing volume.
    """
    patterns = []
    if i < pole_lookback + consolidation_lookback:
        return patterns
    end = i
    pole_start = max(0, end - pole_lookback - consolidation_lookback)
    pole = df['Close'].iloc[pole_start: end - consolidation_lookback]
    if len(pole) < pole_lookback:
        return patterns
    pole_move = pole.iloc[-1] - pole.iloc[0]
    avg_tr = (df['High'] - df['Low']).rolling(pole_lookback).mean().iloc[end - consolidation_lookback - 1]
    if not np.isnan(avg_tr) and abs(pole_move) > 2.5 * avg_tr:
        # consolidation range small relative to pole
        consol = df['Close'].iloc[end - consolidation_lookback: end]
        if consol.max() - consol.min() < 0.5 * abs(pole_move):
            # check volume decreasing
            vols = df['Volume'].iloc[end - consolidation_lookback: end]
            if vols.iloc[-1] < vols.mean() * vol_multiplier:
                # flag: direction same as pole_move sign
                patterns.append('FLAG' if pole_move > 0 else 'FLAG_BEAR')
    return patterns

def detect_triangles_wedges(df, i, lookback=30):
    """
    Heuristic triangle/wedge detection using linear fits on highs and lows in a lookback window.
    Checks whether lines are converging.
    """
    patterns = []
    if i < lookback:
        return patterns
    start = i - lookback + 1
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    x = np.arange(len(highs))
    # linear fits
    try:
        ph = np.polyfit(x, highs, 1)
        pl = np.polyfit(x, lows, 1)
        slope_h, slope_l = ph[0], pl[0]
        # check convergence: difference between start and end of high-low channel reduced
        start_dist = highs[0] - lows[0]
        end_dist = highs[-1] - lows[-1]
        if end_dist < 0.6 * start_dist and abs(slope_h) < 2 * abs(slope_l) and abs(slope_l) < 2 * abs(slope_h):
            # if slopes opposite sign -> symmetric
            if slope_h < 0 and slope_l > 0:
                patterns.append('SYMMETRIC_TRIANGLE')
            elif slope_h < 0 and slope_l < 0:
                patterns.append('FALLING_WEDGE' if abs(slope_h) > abs(slope_l) else 'DESCENDING_TRIANGLE')
            elif slope_h > 0 and slope_l > 0:
                patterns.append('RISING_WEDGE' if abs(slope_h) > abs(slope_l) else 'ASCENDING_TRIANGLE')
    except Exception:
        pass
    return patterns

def detect_channel(df, i, lookback=50, tol_rel=0.06):
    """
    Detect simple channels: highs and lows trendlines roughly parallel over lookback.
    """
    patterns = []
    if i < lookback:
        return patterns
    start = i - lookback + 1
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    x = np.arange(len(highs))
    try:
        ph = np.polyfit(x, highs, 1)
        pl = np.polyfit(x, lows, 1)
        # slopes roughly equal
        if ph[0] != 0 and pl[0] != 0:
            rel_diff = abs(ph[0] - pl[0]) / (abs(ph[0]) + 1e-9)
            if rel_diff < tol_rel:
                patterns.append('CHANNEL')
    except Exception:
        pass
    return patterns

def detect_supply_demand_zones(df, i, lookback=100):
    """
    Simple clustered support/resistance detection:
      - find local peaks/troughs in lookback and cluster close price areas as zones.
    """
    patterns = []
    if i < 10:
        return patterns
    start = max(0, i - lookback + 1)
    sub = df['Close'].iloc[start:i+1]
    # find significant peaks/troughs
    peaks = find_last_peaks_positions(sub, len(sub), n=6)
    troughs = find_last_troughs_positions(sub, len(sub), n=6)
    # if many peaks cluster around similar level -> supply zone
    if peaks:
        vals = [sub.iloc[p] for p in peaks]
        if (np.std(vals) / (np.mean(vals) + 1e-9)) < 0.03 and len(vals) >= 2:
            patterns.append('SUPPLY_ZONE')
    if troughs:
        vals2 = [sub.iloc[t] for t in troughs]
        if (np.std(vals2) / (np.mean(vals2) + 1e-9)) < 0.03 and len(vals2) >= 2:
            patterns.append('DEMAND_ZONE')
    return patterns

def detect_order_blocks(df, i, lookback=40):
    """
    Heuristic order block: a strong bearish/bullish candle with significant volume followed by reversal.
    """
    patterns = []
    if i < 3:
        return patterns
    # look for recent large-bodied opposite-direction candle
    for j in range(max(1, i - 6), i):
        body = abs(df['Close'].iloc[j] - df['Open'].iloc[j])
        rng = df['High'].iloc[j] - df['Low'].iloc[j]
        vol = df['Volume'].iloc[j]
        avg_vol = df['Volume'].rolling(20, min_periods=1).mean().iloc[j]
        if avg_vol is None or avg_vol == 0:
            continue
        if body > 0.8 * rng and vol > 1.5 * avg_vol:
            # if price subsequently reversed (current close above that candle's high) -> bullish order block
            if df['Close'].iloc[i] > df['High'].iloc[j]:
                patterns.append('ORDER_BLOCK_BULL')
            if df['Close'].iloc[i] < df['Low'].iloc[j]:
                patterns.append('ORDER_BLOCK_BEAR')
    return patterns

def detect_fair_value_gap(df, i):
    """
    Simple FVG detection: identifies a 3-bar gap-like area where middle bar body does not overlap with adjacent bodies.
    Conservative heuristic.
    """
    patterns = []
    if i < 2:
        return patterns
    b0o, b0c = df['Open'].iloc[i-2], df['Close'].iloc[i-2]
    b1o, b1c = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
    b2o, b2c = df['Open'].iloc[i], df['Close'].iloc[i]
    # bullish FVG: body of middle bar above high of previous? We'll approximate by body non-overlap
    if max(b0o, b0c) < min(b1o, b1c) and max(b1o, b1c) < min(b2o, b2c):
        patterns.append('FVG_BULL')
    if min(b0o, b0c) > max(b1o, b1c) and min(b1o, b1c) > max(b2o, b2c):
        patterns.append('FVG_BEAR')
    return patterns

def detect_liquidity_sweep(df, i):
    """
    Detect wick sweep above/below recent highs/lows with rejection (close back inside)
    """
    patterns = []
    if i < 1:
        return patterns
    prev_high = df['High'].iloc[i-1]
    prev_low = df['Low'].iloc[i-1]
    H, L, C = df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
    # sweep above: high > prev high but close significantly below high (rejection)
    if H > prev_high and C < (H - 0.3 * (H - L)):
        patterns.append('LIQUIDITY_SWEEP_HIGH')
    if L < prev_low and C > (L + 0.3 * (H - L)):
        patterns.append('LIQUIDITY_SWEEP_LOW')
    return patterns

def detect_fake_breakout(df, i, lookback=10):
    """
    Fake breakout: breakout beyond recent highs/lows followed quickly by reversal within short window.
    Here we only flag a potential fake breakout if price breaks recently and then closes back inside same day or quickly.
    Conservative flag.
    """
    patterns = []
    if i < 2:
        return patterns
    prev_highs = df['High'].iloc[max(0, i - lookback):i]
    prev_lows = df['Low'].iloc[max(0, i - lookback):i]
    if len(prev_highs) == 0:
        return patterns
    if df['High'].iloc[i] > prev_highs.max():
        # if close within body and next bar down (we cannot look ahead); but we can mark as breakout which may be fake if large wick
        if df['Close'].iloc[i] < prev_highs.max() + 0.001*(df['Close'].iloc[i]):
            patterns.append('POT_FAKE_BREAKOUT_HIGH')
    if df['Low'].iloc[i] < prev_lows.min():
        if df['Close'].iloc[i] > prev_lows.min() - 0.001*(df['Close'].iloc[i]):
            patterns.append('POT_FAKE_BREAKOUT_LOW')
    return patterns

def detect_bos_choch(df, i, lookback=30):
    """
    Break of structure (BOS) and Change of Character (CHoCH) heuristics:
      - BOS up: price makes higher high above last significant swing high
      - CHoCH: break opposite structure after prior bias
    """
    patterns = []
    if i < 5:
        return patterns
    # last two swing highs/lows
    highs = find_last_peaks_positions(df['Close'].iloc[:i+1], i+1, n=2)
    lows = find_last_troughs_positions(df['Close'].iloc[:i+1], i+1, n=2)
    cur = df['Close'].iloc[i]
    if highs and highs[-1] < i and df['Close'].iloc[i] > df['Close'].iloc[highs[-1]]:
        patterns.append('BOS_UP')
    if lows and lows[-1] < i and df['Close'].iloc[i] < df['Close'].iloc[lows[-1]]:
        patterns.append('BOS_DOWN')
    # CHoCH: quick reversal condition: price crosses last structure then closes opposite side
    # (these rules are intentionally simple)
    return patterns

def detect_chart_patterns_all(df, i, params):
    """
    Run many pattern detectors and return aggregated pattern list for bar i.
    Uses only df up to i (no future info).
    """
    patterns = []
    # basic candlesticks
    patterns += detect_candlestick_patterns(df, i)
    # head & shoulders
    patterns += detect_head_and_shoulders(df, i, lookback=params.get('hs_lookback', 80))
    # cup & handle
    patterns += detect_cup_and_handle(df, i, lookback=params.get('cup_lookback', 120))
    # flag/pennant
    patterns += detect_flag_or_pennant(df, i, pole_lookback=params.get('pole_lookback', 12),
                                      consolidation_lookback=params.get('consol_lookback', 8),
                                      vol_multiplier=params.get('vol_multiplier', 1.2))
    # triangles/wedges
    patterns += detect_triangles_wedges(df, i, lookback=params.get('triangle_lookback', 30))
    # channel
    patterns += detect_channel(df, i, lookback=params.get('channel_lookback', 50))
    # supply/demand zones
    patterns += detect_supply_demand_zones(df, i, lookback=params.get('sd_lookback', 120))
    # order blocks
    patterns += detect_order_blocks(df, i, lookback=params.get('order_block_lookback', 40))
    # fair value gap
    patterns += detect_fair_value_gap(df, i)
    # liquidity sweeps
    patterns += detect_liquidity_sweep(df, i)
    # fake breakout potential
    patterns += detect_fake_breakout(df, i, lookback=params.get('fake_lookback', 10))
    # BOS/CHoCH
    patterns += detect_bos_choch(df, i, lookback=params.get('bos_lookback', 30))
    # keep unique
    patterns = list(dict.fromkeys(patterns))
    return patterns

# ------------------- Divergence utility -------------------
def find_last_n_peaks(prices, upto_i, n=2):
    out = []
    for j in range(upto_i-1, 0, -1):
        if j - 1 >= 0 and j + 1 < len(prices):
            if prices.iloc[j] > prices.iloc[j-1] and prices.iloc[j] > prices.iloc[j+1]:
                out.append(j)
                if len(out) >= n:
                    break
    return out[::-1]

def find_last_n_troughs(prices, upto_i, n=2):
    out = []
    for j in range(upto_i-1, 0, -1):
        if j - 1 >= 0 and j + 1 < len(prices):
            if prices.iloc[j] < prices.iloc[j-1] and prices.iloc[j] < prices.iloc[j+1]:
                out.append(j)
                if len(out) >= n:
                    break
    return out[::-1]

def detect_divergences(df, i, rsi_col):
    res = []
    prices = df['Close']
    rsi = df[rsi_col]
    if i < 5:
        return res
    peaks = find_last_n_peaks(prices, i, n=2)
    if len(peaks) == 2:
        p0, p1 = peaks
        if prices.iloc[p1] > prices.iloc[p0] and rsi.iloc[p1] < rsi.iloc[p0]:
            res.append('DIV_BEAR')
    troughs = find_last_n_troughs(prices, i, n=2)
    if len(troughs) == 2:
        t0, t1 = troughs
        if prices.iloc[t1] < prices.iloc[t0] and rsi.iloc[t1] > rsi.iloc[t0]:
            res.append('DIV_BULL')
    return res

# ------------------- Confluence Signal Generator (integrates everything) -------------------
def generate_confluence_signals(df_local, params, side="Both", signal_mode="Both", advanced_leading=True):
    """
    Produces signals as before, but integrates:
      - lagging indicators (EMA/SMA/MACD/RSI/ADX/CCI etc.)
      - leading signals: candlesticks, volume spikes, RVOL, imbalances, breakouts, many chart patterns
      - optional advanced leading (divergence, FVG, order blocks, supply/demand, etc.)
      - signal_mode filter: 'Lagging', 'Price Action', 'Both'
    """
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    # what we consider "leading" pattern prefixes (used for filtering by signal_mode)
    leading_prefixes = ('BULL_ENGULF','BEAR_ENGULF','HAMMER','HANGING_MAN','SHOOTING_STAR','INVERTED_HAMMER',
                        'INSIDE_BAR','VOL_SPIKE','BREAK_HI','BREAK_LO','DOJI','MORNING_STAR','EVENING_STAR',
                        'THREE_WHITE','THREE_BLACK','DIV_BULL','DIV_BEAR','RVOL','IMBAL_BULL','IMBAL_BEAR',
                        'FLAG','FLAG_BEAR','SYMMETRIC_TRIANGLE','ASCENDING_TRIANGLE','DESCENDING_TRIANGLE',
                        'FALLING_WEDGE','RISING_WEDGE','CHANNEL','SUPPLY_ZONE','DEMAND_ZONE','ORDER_BLOCK_BULL',
                        'ORDER_BLOCK_BEAR','FVG_BULL','FVG_BEAR','LIQUIDITY_SWEEP_HIGH','LIQUIDITY_SWEEP_LOW',
                        'POT_FAKE_BREAKOUT_HIGH','POT_FAKE_BREAKOUT_LOW','BOS_UP','BOS_DOWN','HEAD_SHOULDERS',
                        'CUP_HANDLE')

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []
        i = df_calc.index.get_loc(idx)

        # --- LAGGING indicators (same as before) ---
        # SMA crossover
        sma_fast = row.get(f"sma_{params['sma_fast']}", np.nan)
        sma_slow = row.get(f"sma_{params['sma_slow']}", np.nan)
        if not np.isnan(sma_fast) and not np.isnan(sma_slow):
            if sma_fast > sma_slow:
                indicators_that_long.append(f"SMA{params['sma_fast']}>{params['sma_slow']}")
            elif sma_fast < sma_slow:
                indicators_that_short.append(f"SMA{params['sma_fast']}<{params['sma_slow']}")

        # EMA crossover
        ema_f = row.get(f"ema_{params['ema_fast']}", np.nan)
        ema_s = row.get(f"ema_{params['ema_slow']}", np.nan)
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if ema_f > ema_s:
                indicators_that_long.append(f"EMA{params['ema_fast']}>{params['ema_slow']}")
            elif ema_f < ema_s:
                indicators_that_short.append(f"EMA{params['ema_fast']}<{params['ema_slow']}")

        # MACD
        if not np.isnan(row.get("macd_hist", np.nan)):
            if row["macd_hist"] > 0:
                indicators_that_long.append("MACD+")
            elif row["macd_hist"] < 0:
                indicators_that_short.append("MACD-")

        # RSI
        rsi_val = row.get(f"rsi_{params['rsi_period']}", np.nan)
        if not np.isnan(rsi_val):
            if rsi_val < params.get('rsi_oversold', 35):
                indicators_that_long.append(f"RSI<{params.get('rsi_oversold', 35)}")
            elif rsi_val > params.get('rsi_overbought', 65):
                indicators_that_short.append(f"RSI>{params.get('rsi_overbought', 65)}")

        # Bollinger bands
        price = row['Close']
        if not np.isnan(row.get('bb_upper', np.nan)) and not np.isnan(row.get('bb_lower', np.nan)):
            if price < row['bb_lower']:
                indicators_that_long.append("BB_Lower")
            elif price > row['bb_upper']:
                indicators_that_short.append("BB_Upper")

        # Momentum
        mom = row.get(f"mom_{params['mom_period']}", np.nan)
        if not np.isnan(mom):
            if mom > 0:
                indicators_that_long.append(f"MOM+({params['mom_period']})")
            elif mom < 0:
                indicators_that_short.append(f"MOM-({params['mom_period']})")

        # Stochastic
        if not np.isnan(row.get('stoch_k', np.nan)) and not np.isnan(row.get('stoch_d', np.nan)):
            if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < params.get('stoch_oversold', 30):
                indicators_that_long.append("STOCH")
            elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > params.get('stoch_overbought', 70):
                indicators_that_short.append("STOCH")

        # ADX
        if not np.isnan(row.get('adx', np.nan)):
            if row['adx'] > params.get('adx_threshold', 20) and row['pdi'] > row['mdi']:
                indicators_that_long.append("ADX+")
            elif row['adx'] > params.get('adx_threshold', 20) and row['mdi'] > row['pdi']:
                indicators_that_short.append("ADX-")

        # OBV rising (lagging-ish)
        if i >= 3:
            recent = df_calc['obv'].iloc[max(0,i-2):i+1].mean()
            prev = df_calc['obv'].iloc[max(0,i-5):max(0,i-2)].mean()
            if recent > prev:
                indicators_that_long.append('OBV')
            elif recent < prev:
                indicators_that_short.append('OBV')

        # VWMA
        if not np.isnan(row.get('vwma', np.nan)):
            if row['Close'] > row['vwma']:
                indicators_that_long.append('VWMA')
            elif row['Close'] < row['vwma']:
                indicators_that_short.append('VWMA')

        # CCI
        if not np.isnan(row.get('cci', np.nan)):
            if row['cci'] < -100:
                indicators_that_long.append('CCI')
            elif row['cci'] > 100:
                indicators_that_short.append('CCI')

        # Volume spike (basic leading)
        if not np.isnan(row.get('vol_sma', np.nan)) and row['Volume'] > 0:
            if row['Volume'] > row['vol_sma'] * params.get('vol_multiplier', 1.5):
                prev_close = df_calc['Close'].shift(1).iloc[i] if i > 0 else np.nan
                if not np.isnan(prev_close) and row['Close'] > prev_close:
                    indicators_that_long.append('VOL_SPIKE')
                elif not np.isnan(prev_close) and row['Close'] < prev_close:
                    indicators_that_short.append('VOL_SPIKE')

        # ----------------- Chart patterns & Advanced Leading signals -----------------
        patterns = detect_chart_patterns_all(df_calc, i, params)

        for p in patterns:
            # decide long/short mapping heuristically:
            # many patterns either imply reversal or continuation. We'll map conservatively:
            if p in ('BULL_ENGULF','HAMMER','MORNING_STAR','THREE_WHITE','CUP_HANDLE','SUPPLY_ZONE') or 'DEMAND_ZONE' in p:
                # supply_zone/ demand_zone mapping: if demand zone exists -> long; supply -> short
                if p == 'SUPPLY_ZONE':
                    indicators_that_short.append(p)
                elif p == 'DEMAND_ZONE':
                    indicators_that_long.append(p)
                else:
                    indicators_that_long.append(p)
            elif p in ('BEAR_ENGULF','HANGING_MAN','EVENING_STAR','THREE_BLACK'):
                indicators_that_short.append(p)
            elif p in ('FLAG','SYMMETRIC_TRIANGLE','ASCENDING_TRIANGLE','DESCENDING_TRIANGLE','FALLING_WEDGE','RISING_WEDGE','CHANNEL'):
                # pattern direction ambiguous -> check last close vs prior trend
                # simple rule: if recent slope positive -> treat as continuation long, else short
                recent_slope = df_calc['Close'].iloc[max(0,i-10):i+1].diff().mean()
                if recent_slope >= 0:
                    indicators_that_long.append(p)
                else:
                    indicators_that_short.append(p)
            elif p in ('ORDER_BLOCK_BULL','IMBAL_BULL','FVG_BULL','DIV_BULL','BOS_UP','LIQUIDITY_SWEEP_LOW'):
                indicators_that_long.append(p)
            elif p in ('ORDER_BLOCK_BEAR','IMBAL_BEAR','FVG_BEAR','DIV_BEAR','BOS_DOWN','LIQUIDITY_SWEEP_HIGH'):
                indicators_that_short.append(p)
            elif p.startswith('POT_FAKE_BREAKOUT'):
                # fake breakout could be either side; mark as caution (both)
                indicators_that_long.append(p)
                indicators_that_short.append(p)
            else:
                # fallback: if pattern name contains BULL or WHITE or similar -> long; BEAR/BLACK -> short
                if 'BULL' in p or 'WHITE' in p:
                    indicators_that_long.append(p)
                if 'BEAR' in p or 'BLACK' in p:
                    indicators_that_short.append(p)

        # Advanced leading: RVOL, imbalances, divergence
        if advanced_leading:
            # RVOL
            if not np.isnan(row.get('vol_sma', np.nan)) and row['vol_sma'] > 0:
                rv = row['Volume'] / (row['vol_sma'] + 1e-9)
                if rv >= params.get('rvol_threshold', 2.0):
                    if row['Close'] > row['Open']:
                        indicators_that_long.append('RVOL')
                    elif row['Close'] < row['Open']:
                        indicators_that_short.append('RVOL')

            # Imbalance (close near high/low + volume)
            rng = row['High'] - row['Low'] if (row['High'] - row['Low']) != 0 else 1e-9
            close_near_high = (row['High'] - row['Close']) / rng < 0.2
            close_near_low = (row['Close'] - row['Low']) / rng < 0.2
            if row['Volume'] > row.get('vol_sma', 0) * 1.2:
                if close_near_high:
                    indicators_that_long.append('IMBAL_BULL')
                if close_near_low:
                    indicators_that_short.append('IMBAL_BEAR')

            # Divergences (RSI)
            rsi_col = f"rsi_{params['rsi_period']}"
            if rsi_col in df_calc.columns:
                divs = detect_divergences(df_calc, i, rsi_col)
                for d in divs:
                    if d == 'DIV_BULL':
                        indicators_that_long.append(d)
                    if d == 'DIV_BEAR':
                        indicators_that_short.append(d)

        # ---- filter indicator lists based on user-selected signal_mode ----
        def is_leading(ind):
            return any(ind.startswith(p) for p in leading_prefixes)

        if signal_mode == 'Lagging':
            indicators_that_long = [x for x in indicators_that_long if not is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if not is_leading(x)]
        elif signal_mode == 'Price Action':
            indicators_that_long = [x for x in indicators_that_long if is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if is_leading(x)]
        # else 'Both' -> keep all

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # side filter
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        final_sig = vote_direction if total_votes >= params['min_confluence'] else 0

        votes.append({
            'index': idx,
            'long_votes': long_votes,
            'short_votes': short_votes,
            'total_votes': total_votes,
            'direction': vote_direction,
            'signal': final_sig,
            'indicators_long': indicators_that_long,
            'indicators_short': indicators_that_short,
            'patterns': patterns
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index('index')
    result = df_calc.join(votes_df)
    result['Signal'] = sig_series
    return result

# ------------------- Backtester (entries on same-bar close; exits on subsequent bars) -------------------
def choose_primary_indicator(indicators_list):
    # priority for display: leading patterns first, then others
    priority = ['CUP_HANDLE','HEAD_SHOULDERS','MORNING_STAR','EVENING_STAR','BULL_ENGULF','BEAR_ENGULF',
                'RVOL','IMBAL_BULL','IMBAL_BEAR','DIV_BULL','DIV_BEAR','FVG_BULL','FVG_BEAR',
                'FLAG','SYMMETRIC_TRIANGLE','FALLING_WEDGE','RISING_WEDGE','CHANNEL',
                'EMA','SMA','MACD','RSI','BB','VWMA','OBV','MOM','STOCH','ADX','CCI']
    for p in priority:
        for it in indicators_list:
            if p in it:
                return it
    return indicators_list[0] if indicators_list else ''

def backtest_point_strategy(df_signals, params):
    """
    Same logic as before:
      - Entry executed at same-bar Close when signal appears (no next-bar entry).
      - Exits evaluated on subsequent bars' OHLC (so exits use later bars but entry never uses future).
    """
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    # buy & hold baseline (points)
    first_price = df_signals['Close'].iloc[0]
    last_price = df_signals['Close'].iloc[-1]
    buy_hold_points = last_price - first_price

    n = len(df_signals)
    for i in range(n):
        row = df_signals.iloc[i]
        sig = row['Signal']

        # 1) If in position -> check exits using THIS row (future relative to entry)
        if in_pos:
            h = row['High']; l = row['Low']; closep = row['Close']
            exit_price = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target; reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl; reason = "Stopped"
                elif sig == -1 and row['total_votes'] >= params['min_confluence']:
                    exit_price = closep; reason = "Opposite signal"
            else:  # short
                if not pd.isna(l) and l <= target:
                    exit_price = target; reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl; reason = "Stopped"
                elif sig == 1 and row['total_votes'] >= params['min_confluence']:
                    exit_price = closep; reason = "Opposite signal"

            # final day fallback: exit at close if last bar
            if i == (n-1) and in_pos and exit_price is None:
                exit_price = closep; reason = "End of data"

            if exit_price is not None:
                exit_date = row.name
                points = (exit_price - entry_price) if pos_side == 1 else (entry_price - exit_price)
                trades.append({
                    **entry_details,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Reason Exit": reason,
                    "Points": points,
                    "Hold Days": (pd.to_datetime(exit_date).date() - pd.to_datetime(entry_details['Entry Date']).date()).days
                })
                # reset
                in_pos = False; pos_side = 0
                entry_price = None; entry_details = None; target = None; sl = None

        # 2) If not in position -> enter using THIS row's Close
        if (not in_pos) and sig != 0:
            entry_date = row.name
            entry_price = row['Close'] if not pd.isna(row['Close']) else row['Open']
            pos_side = sig
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params['target_atr_mult'] * atr_val
                sl = entry_price - params['sl_atr_mult'] * atr_val
            else:
                target = entry_price - params['target_atr_mult'] * atr_val
                sl = entry_price + params['sl_atr_mult'] * atr_val

            indicators = (row['indicators_long'] if pos_side==1 else row['indicators_short'])
            primary = choose_primary_indicator(indicators)

            entry_details = {
                "Entry Date": entry_date, "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                "Indicators": indicators,
                "Primary Indicator": primary,
                "Confluences": row['total_votes']
            }
            in_pos = True
            # do not check exits on same bar as entry
            continue

    trades_df = pd.DataFrame(trades)
    total_points = trades_df['Points'].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df['Points'] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades>0 else 0.0

    percent_vs_buyhold = (total_points / (abs(buy_hold_points)+1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        'total_points': total_points,
        'num_trades': num_trades,
        'wins': wins,
        'prob_of_profit': prob_of_profit,
        'buy_hold_points': buy_hold_points,
        'pct_vs_buyhold': percent_vs_buyhold
    }
    return summary, trades_df

# ------------------- Parameter optimization (random search) -------------------
def sample_random_params(base):
    p = base.copy()
    p['sma_fast'] = random.choice([5,8,10,12,15,20])
    p['sma_slow'] = random.choice([50,100,150,200])
    if p['sma_fast'] >= p['sma_slow']:
        p['sma_fast'] = max(5, p['sma_slow']//10)
    p['ema_fast'] = random.choice([5,9,12,15])
    p['ema_slow'] = random.choice([21,26,34,50])
    if p['ema_fast'] >= p['ema_slow']:
        p['ema_fast'] = max(5, p['ema_slow']//3)
    p['rsi_period'] = random.choice([7,9,14,21])
    p['mom_period'] = random.choice([5,10,20])
    p['atr_period'] = random.choice([7,14,21])
    p['target_atr_mult'] = round(random.uniform(0.6,3.0),2)
    p['sl_atr_mult'] = round(random.uniform(0.6,3.0),2)
    p['min_confluence'] = random.randint(1,6)
    p['vol_multiplier'] = round(random.uniform(1.0,3.0),2)
    p['breakout_lookback'] = random.choice([5,10,20])
    p['rvol_threshold'] = round(random.uniform(1.5,3.0),2)
    p['pole_lookback'] = random.choice([8,12,16])
    p['consol_lookback'] = random.choice([6,8,10])
    p['triangle_lookback'] = random.choice([20,30,40])
    p['hs_lookback'] = random.choice([60,80,100])
    p['cup_lookback'] = random.choice([80,120,160])
    return p

def optimize_parameters(df, base_params, n_iter, target_acc, target_points, side, signal_mode='Both', advanced_leading=True, progress_bar=None, status_text=None):
    best = None
    best_score = None
    target_frac = target_acc
    for i in range(n_iter):
        p = sample_random_params(base_params)
        try:
            df_sig = generate_confluence_signals(df, p, side, signal_mode, advanced_leading)
            summary, trades = backtest_point_strategy(df_sig, p)
        except Exception:
            if progress_bar:
                progress_bar.progress(int((i+1)/n_iter*100))
            if status_text:
                status_text.text(f"Iteration {i+1}/{n_iter} (error)")
            continue
        prob = summary['prob_of_profit']
        score = abs(prob - target_frac) - 0.0001 * summary['total_points']
        if best is None or score < best_score:
            best = (p, summary, trades)
            best_score = score
        if progress_bar:
            progress_bar.progress(int((i+1)/n_iter*100))
        if status_text:
            status_text.text(f"Iteration {i+1}/{n_iter}")
        if prob >= target_frac and summary['total_points'] >= target_points:
            return p, summary, trades, True
    return best[0], best[1], best[2], False

# ------------------- Streamlit App UI (Run button placed at bottom as requested) -------------------
st.title("Backtester with Rich Chart-Pattern Library + Leading Signals")
st.write("Upload OHLCV CSV/XLSX (Date,Open,High,Low,Close,Volume). Column names are handled case-insensitively (e.g. 'SHARES TRADED', 'vol', 'Volume' accepted).")
st.write("Entries execute at candle Close (no future lookahead). Use 'Select last date' to restrict dataset (start date = min available).")

uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv','xlsx'])
if uploaded_file is None:
    st.info("Upload a CSV/XLSX to start.")
# UI controls (these appear before Run but per request the actual 'Run' button placed at bottom)
side = st.selectbox("Trade Side", options=["Both","Long","Short"], index=0)
signal_mode = st.selectbox("Signal source/mode", options=["Lagging","Price Action","Both"], index=2)
advanced_leading = st.checkbox("Use advanced leading signals (divergence/RVOL/imbalance and advanced chart patterns)", value=True)
random_iters = st.number_input("Random iterations (1-2000)", min_value=1, max_value=2000, value=200, step=1)
expected_returns = st.number_input("Expected strategy returns (total points)", value=2000.0, step=1.0, format="%.2f")
expected_accuracy_pct = st.number_input("Expected accuracy % (probability of profit, e.g. 70)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)

if uploaded_file is not None:
    try:
        if str(uploaded_file).lower().endswith('.xlsx') or (hasattr(uploaded_file, 'name') and 'xls' in uploaded_file.name.lower()):
            raw = pd.read_excel(uploaded_file)
        else:
            raw = pd.read_csv(uploaded_file)
        df_full = normalize_df(raw)
    except Exception as e:
        st.error(f"Failed to read/normalize file: {e}")
        st.stop()

    # show uploaded top/bottom 5 rows (raw)
    st.subheader("Uploaded file sample (raw)")
    try:
        st.write(f"Rows: {raw.shape[0]} Columns: {raw.shape[1]}")
        st.dataframe(raw.head(5))
        st.subheader("Uploaded file - bottom 5 rows (raw)")
        st.dataframe(raw.tail(5))
    except Exception:
        pass

    # show normalized dataframe tail so user can verify last candle values used
    st.subheader("Normalized (used) data - last 5 rows")
    try:
        st.dataframe(df_full.tail(5))
        st.write("Last normalized close:", df_full['Close'].iloc[-1])
        st.write("Last normalized index:", df_full.index[-1])
    except Exception:
        pass

    # Dropdown: select last date (inclusive)
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    st.markdown(f"**Data range available:** {min_date} to {max_date}")
    unique_dates = sorted({d.date() for d in df_full.index})
    selected_last_date = st.selectbox("Select last date (restrict data up to this date)", options=unique_dates, index=len(unique_dates)-1, format_func=lambda x: x.strftime('%Y-%m-%d'))
    # slice dataframe inclusive
    df = df_full[df_full.index.date <= selected_last_date]

    # put Run button and related controls at *end* (as requested)
    run_btn = st.button("Run Backtest & Optimize (this runs optimization on the restricted dataset)")

    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Running optimization on restricted dataset..."):
            base_params = {
                'sma_fast': 10, 'sma_slow': 50,
                'ema_fast': 9, 'ema_slow': 21,
                'macd_signal': 9, 'rsi_period': 14, 'mom_period': 10,
                'stoch_period': 14, 'cci_period': 20, 'adx_period': 14,
                'atr_period': 14, 'target_atr_mult': 1.5, 'sl_atr_mult': 1.0,
                'min_confluence': 3, 'vol_multiplier': 1.5,
                'vwma_period': 14, 'vol_sma_period': 20,
                'breakout_lookback': 10, 'rvol_threshold': 2.0,
                'pole_lookback': 12, 'consol_lookback': 8, 'triangle_lookback': 30,
                'hs_lookback': 80, 'cup_lookback': 120
            }

            target_acc = expected_accuracy_pct/100.0
            target_points = expected_returns

            best_params, best_summary, best_trades, perfect = optimize_parameters(
                df, base_params, int(random_iters), target_acc, target_points, side,
                signal_mode=signal_mode, advanced_leading=advanced_leading,
                progress_bar=progress_bar, status_text=status_text
            )

            progress_bar.progress(100)
            status_text.text("Optimization completed")

            st.subheader("Optimization Result")
            st.write("Signal source:", signal_mode)
            st.write("Advanced leading enabled:", advanced_leading)
            st.write("Target accuracy:", expected_accuracy_pct, "% ; Target points:", target_points)
            st.write("Perfect match found:" , perfect)
            st.json(best_params)

            st.subheader("Summary (best candidate)")
            st.write(best_summary)

            if best_trades is None or best_trades.empty:
                st.info("No trades found with best parameters.")
            else:
                best_trades_display = best_trades.copy()
                st.subheader("Top 5 trades (by Points)")
                st.dataframe(best_trades_display.nlargest(5, 'Points'))
                st.subheader("Bottom 5 trades (by Points)")
                st.dataframe(best_trades_display.nsmallest(5, 'Points'))

                # Heatmap: monthly percent returns
                best_trades_display['Exit Date'] = pd.to_datetime(best_trades_display['Exit Date'])
                best_trades_display['Year'] = best_trades_display['Exit Date'].dt.year
                best_trades_display['Month'] = best_trades_display['Exit Date'].dt.month
                monthly_points = best_trades_display.groupby(['Year','Month'])['Points'].sum().reset_index()

                month_start = df['Close'].resample('MS').first().reset_index()
                month_start['Year'] = month_start['Date'].dt.year
                month_start['Month'] = month_start['Date'].dt.month
                month_start = month_start.rename(columns={'Close':'Month_Start_Close'})

                monthly = monthly_points.merge(month_start[['Year','Month','Month_Start_Close']], on=['Year','Month'], how='left')
                avg_close = df['Close'].mean()
                monthly['Month_Start_Close'] = monthly['Month_Start_Close'].fillna(avg_close)
                monthly['Pct_Return'] = (monthly['Points'] / monthly['Month_Start_Close']) * 100.0

                pivot_pct = monthly.pivot(index='Year', columns='Month', values='Pct_Return').fillna(0)
                for m in range(1,13):
                    if m not in pivot_pct.columns:
                        pivot_pct[m] = 0
                pivot_pct = pivot_pct.reindex(sorted(pivot_pct.columns), axis=1)

                st.subheader("Monthly % returns heatmap (Year vs Month)")
                fig, ax = plt.subplots(figsize=(10, max(2, 0.6*len(pivot_pct.index)+1)))
                sns.heatmap(pivot_pct, annot=True, fmt='.2f', linewidths=0.5, ax=ax)
                ax.set_ylabel('Year')
                ax.set_xlabel('Month')
                st.pyplot(fig)

                st.subheader("All trades (best candidate)")
                st.dataframe(best_trades_display)

            # live recommendation using best params on restricted df
            latest_sig_df = generate_confluence_signals(df, best_params, side, signal_mode, advanced_leading)
            latest_row = latest_sig_df.iloc[-1]

            st.subheader("Signal DataFrame - last 10 rows (debug)")
            st.dataframe(latest_sig_df.tail(10))

            sig_val = int(latest_row['Signal'])
            sig_text = "Buy" if sig_val == 1 else ("Sell" if sig_val == -1 else "No Signal")

            atr_val = latest_row.get(f"atr_{best_params['atr_period']}", np.nan)
            entry_price_est = float(latest_row['Close'])
            actual_last_close = float(df['Close'].iloc[-1])
            actual_last_idx = df.index[-1]

            if sig_val == 1:
                target_price = entry_price_est + best_params['target_atr_mult'] * atr_val
                sl_price = entry_price_est - best_params['sl_atr_mult'] * atr_val
                indicators_list = latest_row['indicators_long']
            elif sig_val == -1:
                target_price = entry_price_est - best_params['target_atr_mult'] * atr_val
                sl_price = entry_price_est + best_params['sl_atr_mult'] * atr_val
                indicators_list = latest_row['indicators_short']
            else:
                target_price = np.nan; sl_price = np.nan; indicators_list = []

            primary = choose_primary_indicator(indicators_list)
            confluences = int(latest_row.get('total_votes', 0))
            prob_of_profit = (best_summary.get('prob_of_profit', np.nan) * 100.0) if isinstance(best_summary, dict) else np.nan

            def explain_indicator(it):
                if it.startswith('EMA'):
                    return f"{it}: short-term EMA vs long-term EMA momentum"
                if it.startswith('SMA'):
                    return f"{it}: SMA crossover trend bias"
                if 'MACD' in it:
                    return "MACD histogram sign indicates momentum"
                if it.startswith('RSI'):
                    return "RSI extreme indicates overbought/oversold"
                if 'BB' in it:
                    return "MBB bands extreme suggests extension/reversion"
                if it == 'VOL_SPIKE':
                    return "Large volume spike confirming direction"
                if it == 'RVOL':
                    return "Relative Volume spike — unusual activity"
                if it == 'IMBAL_BULL':
                    return "Close near high + volume: bullish imbalance"
                if it == 'IMBAL_BEAR':
                    return "Close near low + volume: bearish imbalance"
                if it == 'DIV_BULL':
                    return "Bullish divergence (price lower low, RSI higher low)"
                if it == 'DIV_BEAR':
                    return "Bearish divergence (price higher high, RSI lower high)"
                if it == 'CUP_HANDLE':
                    return "Cup & handle pattern (possible continuation/breakout)"
                if it == 'HEAD_SHOULDERS':
                    return "Head & Shoulders (reversal structure)"
                if it == 'FLAG':
                    return "Flag pattern after strong pole (continuation likely)"
                return it

            reasons = [explain_indicator(ii) for ii in indicators_list]
            reason_text = (f"Primary: {primary}. ") + ("; ".join(reasons) if reasons else "No strong indicator explanation.")

            indicator_values = {
                'sma_fast': latest_row.get(f"sma_{best_params['sma_fast']}", np.nan),
                'sma_slow': latest_row.get(f"sma_{best_params['sma_slow']}", np.nan),
                'ema_fast': latest_row.get(f"ema_{best_params['ema_fast']}", np.nan),
                'ema_slow': latest_row.get(f"ema_{best_params['ema_slow']}", np.nan),
                'macd_hist': latest_row.get('macd_hist', np.nan),
                f"rsi_{best_params['rsi_period']}": latest_row.get(f"rsi_{best_params['rsi_period']}", np.nan),
                'bb_upper': latest_row.get('bb_upper', np.nan),
                'bb_lower': latest_row.get('bb_lower', np.nan),
                'obv': latest_row.get('obv', np.nan),
                'vwma': latest_row.get('vwma', np.nan),
                'cci': latest_row.get('cci', np.nan),
                'vol': latest_row.get('Volume', np.nan),
                'vol_sma': latest_row.get('vol_sma', np.nan),
                f"atr_{best_params['atr_period']}": latest_row.get(f"atr_{best_params['atr_period']}", np.nan)
            }

            st.subheader("Latest live recommendation (based on best params)")
            st.markdown(f"**Date (last available used):** {latest_sig_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Signal:** {sig_text}")
            st.markdown(f"**Entry (executed at candle close):** {entry_price_est:.2f}")
            st.markdown(f"**Target:** {target_price:.2f}  |  **Stop-loss:** {sl_price:.2f}")
            st.markdown(f"**Confluences (votes):** {confluences}  |  **Primary indicator:** {primary}")
            st.markdown(f"**Probability of profit (backtested):** {prob_of_profit:.2f}%")
            st.markdown("**Indicators that voted:**")
            st.write(indicators_list)
            st.markdown("**Reason / Logic (brief):**")
            st.write(reason_text)

            st.subheader("Latest indicator values (key ones)")
            ind_df = pd.DataFrame([indicator_values]).T.reset_index()
            ind_df.columns = ['Indicator', 'Value']
            st.dataframe(ind_df)

            # debug mismatch
            if not np.isclose(entry_price_est, actual_last_close, atol=1e-8):
                st.warning("Detected discrepancy between entry price (from signal DataFrame) and actual last Close in the sliced data.")
                st.write(f"entry_price_est (signal df close): {entry_price_est}")
                st.write(f"actual_last_close (sliced df close): {actual_last_close}  at index {actual_last_idx}")
                st.info("Shown below: raw uploaded last rows and normalized data used for calculations.")
                st.subheader("Raw uploaded - last 5 rows")
                try:
                    st.dataframe(raw.tail(5))
                except Exception:
                    pass
                st.subheader("Normalized (used) - last 10 rows")
                st.dataframe(df_full.tail(10))
                st.subheader("Signal DF - last 10 rows")
                st.dataframe(latest_sig_df.tail(10))

                st.error("Possible causes: (1) 'Close' column had non-numeric formatting and got coerced; (2) duplicate/misaligned timestamps; (3) the dataset slice used by optimizer/backtest is different from uploaded file. The app attempts cleaning; inspect the three tables above. If needed I can add extra parsing rules.")

            st.success("Done")
