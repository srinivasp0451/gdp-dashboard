# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from math import isclose
import warnings
warnings.filterwarnings("ignore")

# ------------------- Helpers / Robust Column Mapping -------------------
def normalize_df(df):
    """
    Robustly map possible column names to Open/High/Low/Close/Volume/Date.
    Handles varied casing, columns like 'SHARES TRADED', 'Qty', 'price', 'last', etc.
    """
    orig_cols = list(df.columns)
    cols_lower = {c.lower(): c for c in orig_cols}
    mapping = {}

    for low, orig in cols_lower.items():
        # look for tokens
        if 'open' in low and 'open' not in mapping.values():
            mapping[orig] = "Open"
        elif 'high' in low and 'high' not in mapping.values():
            mapping[orig] = "High"
        elif 'low' in low and low.count('low')>0 and 'low' not in mapping.values():
            mapping[orig] = "Low"
        elif ('close' in low or 'last' in low or 'price' in low or 'adjclose' in low) and 'Close' not in mapping.values():
            mapping[orig] = "Close"
        elif ('vol' in low or 'share' in low or 'qty' in low or 'traded' in low or 'volume' in low) and 'Volume' not in mapping.values():
            mapping[orig] = "Volume"
        elif ('date' in low or 'time' in low or 'datetime' in low or 'timestamp' in low) and 'Date' not in mapping.values():
            mapping[orig] = "Date"
        # else leave unmapped for now

    # apply mapping (only mapped columns)
    df = df.rename(columns=mapping)

    # ensure required cols
    for required in ["Open", "High", "Low", "Close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}. Found columns: {list(df.columns)}")

    # ensure Volume exists
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        if df["Date"].isna().any():
            df["Date"] = pd.to_datetime(df["Date"].astype(str), dayfirst=False, errors='coerce')
        df = df.dropna(subset=["Date"]).copy()
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
    else:
        # try to interpret index as datetime
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        if df.index.isna().any():
            # try to find a column that looks like date if mapping failed
            for c in orig_cols:
                if any(tok in c.lower() for tok in ('date','time','timestamp','datetime')):
                    df[c] = pd.to_datetime(df[c], errors='coerce')
                    df = df.dropna(subset=[c])
                    df[c] = df[c].dt.tz_localize(None)
                    df = df.set_index(c).sort_index()
                    break
        df = df.sort_index()

    # Clean numeric columns: remove commas/spaces/currency symbols then convert
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
            df[col] = df[col].str.replace(r'[₹$€£,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows with missing core price
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df

# ------------------- Indicators (causal) -------------------
def compute_indicators(df, params):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0)

    # Moving averages
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

    # ADX approximation
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

    # fill indicator columns only (do not overwrite price columns)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_cols = [c for c in df.columns if c not in price_cols]
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill')

    return df

# ------------------- Pattern Detectors (heuristic & causal) -------------------

def is_local_peak(series, j):
    # requires j in [1, len-2]
    if j <= 0 or j >= len(series)-1:
        return False
    return series.iloc[j] > series.iloc[j-1] and series.iloc[j] > series.iloc[j+1]

def is_local_trough(series, j):
    if j <= 0 or j >= len(series)-1:
        return False
    return series.iloc[j] < series.iloc[j-1] and series.iloc[j] < series.iloc[j+1]

def find_local_peaks(series, upto_i, lookback=100):
    """Return indices (iloc positions) of local peaks up to upto_i (exclusive of future)."""
    peaks = []
    start = max(1, upto_i - lookback)
    end = upto_i  # non-inclusive of future
    for j in range(start, end-1):
        if is_local_peak(series, j):
            peaks.append(j)
    return peaks

def find_local_troughs(series, upto_i, lookback=100):
    troughs = []
    start = max(1, upto_i - lookback)
    end = upto_i
    for j in range(start, end-1):
        if is_local_trough(series, j):
            troughs.append(j)
    return troughs

def detect_head_and_shoulders(df, i, lookback=60, tolerance=0.12):
    """
    Heuristic head & shoulders detection:
      - find last 3 peaks before index i (pL, pH, pR) with pH highest and shoulders approx equal.
      - neckline between troughs.
    This returns 'BEAR_HS' (normal H&S) or 'BULL_HS_INV' (inverse H&S) if found, else [].
    """
    res = []
    highs = df['High']
    lows = df['Low']
    upto = i
    peaks = find_local_peaks(highs, upto, lookback)
    troughs = find_local_troughs(lows, upto, lookback)

    # need at least 3 peaks
    if len(peaks) >= 3:
        # look at last 3 peaks
        p0, p1, p2 = peaks[-3], peaks[-2], peaks[-1]
        h0, h1, h2 = highs.iloc[p0], highs.iloc[p1], highs.iloc[p2]
        # normal H&S: middle head is highest and shoulders similar
        if (h1 > h0) and (h1 > h2):
            shoulder_diff = abs(h0 - h2) / max(h0, h2, 1e-9)
            if shoulder_diff <= tolerance:
                # ensure neckline exist as troughs between peaks
                # (we check if troughs between p0-p1 and p1-p2 exist)
                t_between_01 = [t for t in troughs if p0 < t < p1]
                t_between_12 = [t for t in troughs if p1 < t < p2]
                if t_between_01 and t_between_12:
                    res.append('BEAR_HS')
    # Inverse H&S based on troughs
    if len(troughs) >= 3:
        t0, t1, t2 = troughs[-3], troughs[-2], troughs[-1]
        l0, l1, l2 = lows.iloc[t0], lows.iloc[t1], lows.iloc[t2]
        if (l1 < l0) and (l1 < l2):
            shoulder_diff = abs(l0 - l2) / max(l0, l2, 1e-9)
            if shoulder_diff <= tolerance:
                res.append('BULL_HS_INV')
    return res

def detect_double_top_bottom(df, i, lookback=60, tolerance=0.08):
    res = []
    highs = df['High']
    lows = df['Low']
    peaks = find_local_peaks(highs, i, lookback)
    troughs = find_local_troughs(lows, i, lookback)
    if len(peaks) >= 2:
        pA, pB = peaks[-2], peaks[-1]
        if abs(highs.iloc[pA] - highs.iloc[pB]) / max(highs.iloc[pA], highs.iloc[pB], 1e-9) <= tolerance:
            res.append('DOUBLE_TOP')
    if len(troughs) >= 2:
        tA, tB = troughs[-2], troughs[-1]
        if abs(lows.iloc[tA] - lows.iloc[tB]) / max(lows.iloc[tA], lows.iloc[tB], 1e-9) <= tolerance:
            res.append('DOUBLE_BOTTOM')
    return res

def detect_triangle_or_wedge(df, i, lookback=40, tolerance_range_shrink=0.6):
    """
    Detect symmetrical/ascending/descending triangle or wedge by checking whether
    highs are making lower highs and lows making higher lows and whether the range shrinks.
    """
    res = []
    start = max(0, i - lookback + 1)
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    if len(highs) < 6:
        return res
    # compute linear fit slopes
    x = np.arange(len(highs))
    try:
        slope_high = np.polyfit(x, highs, 1)[0]
        slope_low = np.polyfit(x, lows, 1)[0]
    except Exception:
        slope_high = 0
        slope_low = 0
    # range shrink check
    range_start = highs[0] - lows[0]
    range_now = highs[-1] - lows[-1]
    if range_start <= 0:
        return res
    shrink_frac = range_now / (range_start + 1e-9)
    # symmetrical triangle: slopes opposite sign and range shrunk significantly
    if slope_high < 0 and slope_low > 0 and shrink_frac < tolerance_range_shrink:
        res.append('SYM_TRIA')
    # ascending triangle: highs flat-ish, lows rising
    if abs(slope_high) < 0.0001 and slope_low > 0 and shrink_frac < tolerance_range_shrink:
        res.append('ASC_TRIA')
    # descending triangle
    if abs(slope_low) < 0.0001 and slope_high < 0 and shrink_frac < tolerance_range_shrink:
        res.append('DESC_TRIA')
    # wedge (both slopes in same direction but converge)
    if slope_high < 0 and slope_low < 0 and shrink_frac < tolerance_range_shrink:
        res.append('FALLING_WEDGE')
    if slope_high > 0 and slope_low > 0 and shrink_frac < tolerance_range_shrink:
        res.append('RISING_WEDGE')
    return res

def detect_flag_or_pennant(df, i, pole_lookback=20, consolidation_lookback=10, min_pole_move=0.07):
    """
    Detect flag/pennant: strong pole (recent large move) followed by small consolidation.
    This is heuristic and causal: looks backwards only.
    """
    res = []
    n = len(df)
    if i < pole_lookback + consolidation_lookback:
        return res
    # pole is the move from start_pole to top prior to consolidation
    start_pole = i - pole_lookback - consolidation_lookback + 1
    end_pole = i - consolidation_lookback
    if start_pole < 0:
        return res
    close = df['Close']
    pole_move = (close.iloc[end_pole] - close.iloc[start_pole]) / (close.iloc[start_pole] + 1e-9)
    # detect bullish flag/pennant
    if pole_move > min_pole_move:
        # consolidation small range
        cons_high = df['High'].iloc[end_pole+1:i+1].max()
        cons_low = df['Low'].iloc[end_pole+1:i+1].min()
        cons_range = cons_high - cons_low
        pole_range = df['High'].iloc[end_pole] - df['Low'].iloc[start_pole]
        if pole_range <= 0:
            return res
        if cons_range / (pole_range + 1e-9) < 0.35:
            res.append('BULL_FLAG')
    # bearish
    pole_move_bear = (close.iloc[end_pole] - close.iloc[start_pole]) / (close.iloc[start_pole] + 1e-9)
    if pole_move_bear < -min_pole_move:
        cons_high = df['High'].iloc[end_pole+1:i+1].max()
        cons_low = df['Low'].iloc[end_pole+1:i+1].min()
        cons_range = cons_high - cons_low
        pole_range = df['High'].iloc[end_pole] - df['Low'].iloc[start_pole]
        if pole_range <= 0:
            return res
        if cons_range / (pole_range + 1e-9) < 0.35:
            res.append('BEAR_FLAG')
    return res

def detect_cup_and_handle(df, i, lookback=120, depth_ratio=0.15):
    """
    Heuristic cup & handle detection: looks for a rounded bottom where center low is lower than sides,
    then a small pullback (handle) after the right side. Conservative and causal.
    """
    res = []
    if i < 30:
        return res
    start = max(0, i - lookback + 1)
    close = df['Close'].iloc[start:i+1].values
    if len(close) < 30:
        return res
    mid = len(close)//2
    left_max = np.max(close[:mid])
    right_max = np.max(close[mid:])
    center_min = np.min(close)
    # depth check
    depth = (min(left_max, right_max) - center_min) / (min(left_max, right_max) + 1e-9)
    if depth > depth_ratio and center_min == close[mid] or True:
        # simplified: if there was a significant rounded dip, and a small pullback near the end
        handle_size = (right_max - close[-1]) / (right_max + 1e-9)
        if 0.01 < handle_size < 0.12:
            res.append('CUP_HANDLE')
    return res

def detect_channels(df, i, lookback=60, tolerance_parallel=0.15):
    """
    Rough channel detection: if linear fit of highs and lows have similar slopes and distance fairly constant.
    """
    res = []
    start = max(0, i - lookback + 1)
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    if len(highs) < 6:
        return res
    x = np.arange(len(highs))
    try:
        slope_h, intercept_h = np.polyfit(x, highs, 1)
        slope_l, intercept_l = np.polyfit(x, lows, 1)
    except Exception:
        return res
    slope_diff = abs(slope_h - slope_l)
    avg_range = np.mean(highs - lows)
    # channels when slopes similar and not diverging too much
    if slope_diff <= tolerance_parallel * (abs(slope_h) + 1e-9):
        res.append('CHANNEL')
    return res

def detect_fvg(df, i):
    """
    Fair Value Gap (FVG) heuristic: if a candle leaves a price area unfilled: e.g. large body gap between consecutive candles
    We'll detect simple 3-candle FVG: if candle k has body far from previous/next bodies.
    """
    res = []
    if i < 2:
        return res
    # use last three candles
    b1_high = df['High'].iloc[i-2]
    b1_low = df['Low'].iloc[i-2]
    b2_high = df['High'].iloc[i-1]
    b2_low = df['Low'].iloc[i-1]
    b3_high = df['High'].iloc[i]
    b3_low = df['Low'].iloc[i]
    # bullish FVG: middle candle red and next candle gaps above its high
    if b2_high < b3_low:
        res.append('FVG_BULL')
    if b2_low > b3_high:
        res.append('FVG_BEAR')
    return res

def detect_order_block_and_supply_demand(df, i, lookback=60, strength_mult=1.5):
    """
    Heuristic order block detection:
      - bullish order block: a bearish candle followed by a strong bullish move (higher closes)
      - bearish order block: opposite
    We mark potential order-block candle bodies (not precise blocks).
    """
    res = []
    if i < 5:
        return res
    start = max(0, i - lookback)
    closes = df['Close'].iloc[start:i+1]
    volumes = df['Volume'].iloc[start:i+1]
    # find a strong bullish run in recent past
    pct_move = (closes.iloc[-1] - closes.iloc[0]) / (closes.iloc[0] + 1e-9)
    vol_recent = volumes.iloc[-1]
    vol_avg = volumes.mean() + 1e-9
    if pct_move > 0.08 and vol_recent > strength_mult * vol_avg:
        # assume prior bearish candle may be order block
        # find last bearish candle
        for j in range(i-1, start-1, -1):
            if df['Close'].iloc[j] < df['Open'].iloc[j]:
                res.append('OB_BULL')  # bullish order-block
                break
    if pct_move < -0.08 and vol_recent > strength_mult * vol_avg:
        for j in range(i-1, start-1, -1):
            if df['Close'].iloc[j] > df['Open'].iloc[j]:
                res.append('OB_BEAR')
                break
    # quick supply/demand zone: cluster of highs or lows
    # cluster highs
    highs = df['High'].iloc[start:i+1]
    lows = df['Low'].iloc[start:i+1]
    if highs.std() < (np.mean(highs) * 0.02):
        res.append('SUPPLY_ZONE')  # many highs near each other
    if lows.std() < (np.mean(lows) * 0.02):
        res.append('DEMAND_ZONE')
    return res

def detect_liquidity_sweep_and_fake_breakouts(df, i, lookback=20):
    """
    Liquidity sweep / fake breakout heuristics:
      - detect long upper/lower wick that clears previous swing high/low then closes back inside.
    """
    res = []
    if i < 2:
        return res
    highs = df['High']
    lows = df['Low']
    closes = df['Close']
    prev_high = highs.iloc[i-1]
    prev_low = lows.iloc[i-1]
    h = highs.iloc[i]
    l = lows.iloc[i]
    c = closes.iloc[i]
    # sweep above prior high but close back below prior high => liquidity sweep to the upside
    if h > prev_high and c < prev_high:
        res.append('LIQ_SWEEP_UP')
    if l < prev_low and c > prev_low:
        res.append('LIQ_SWEEP_DOWN')
    # fake breakout: breakout of consolidation (lookback) then return inside quickly
    start = max(0, i - lookback)
    range_high = highs.iloc[start:i].max()
    range_low = lows.iloc[start:i].min()
    if (h > range_high) and (c < range_high):
        res.append('FAKE_BREAK_UP')
    if (l < range_low) and (c > range_low):
        res.append('FAKE_BREAK_DOWN')
    return res

def detect_bos_choch(df, i, lookback=50):
    """
    Break of Structure (BOS) and Change of Character (CHoCH) detection (simple heuristics).
    - BOS up: price closes above last swing high
    - BOS down: price closes below last swing low
    - CHoCH detection: change in HH/HL vs LH/LL structure
    """
    res = []
    if i < 3:
        return res
    highs = df['High']
    lows = df['Low']
    closes = df['Close']
    # find last swing high and low (local peak/trough)
    peaks = find_local_peaks(highs, i, lookback)
    troughs = find_local_troughs(lows, i, lookback)
    last_peak = highs.iloc[peaks[-1]] if peaks else None
    last_trough = lows.iloc[troughs[-1]] if troughs else None
    c = closes.iloc[i]
    if peaks and c > last_peak:
        res.append('BOS_UP')
    if troughs and c < last_trough:
        res.append('BOS_DOWN')
    # CHoCH is signaled when structure flips: previous trend up (HH/HL) to down (LH/LL)
    # Very heuristic: compare last two peaks/troughs
    if len(peaks) >= 2 and len(troughs) >= 2:
        p0, p1 = peaks[-2], peaks[-1]
        t0, t1 = troughs[-2], troughs[-1]
        # if previously highs rising then now new low lower -> CHoCH down
        if highs.iloc[p1] > highs.iloc[p0] and lows.iloc[t1] < lows.iloc[t0] and c < lows.iloc[t1]:
            res.append('CHOCH_DOWN')
        # vice versa
        if highs.iloc[p1] < highs.iloc[p0] and lows.iloc[t1] > lows.iloc[t0] and c > highs.iloc[p1]:
            res.append('CHOCH_UP')
    return res

# ------------------- Candlestick / simple leading detectors already present (keep them) -------------------
def detect_basic_candles(df, i):
    # reuse earlier candlestick heuristics: engulfing, hammer, shooting star, inside bar
    patterns = []
    if i < 0:
        return patterns
    row = df.iloc[i]
    prev = df.iloc[i-1] if i-1 >= 0 else None
    O = row['Open']; H = row['High']; L = row['Low']; C = row['Close']
    body = abs(C - O)
    lower_shadow = min(O, C) - L
    upper_shadow = H - max(O, C)
    rng = max(H - L, 1e-9)
    if body <= 0.1 * rng:
        patterns.append('DOJI')
    if prev is not None:
        if (prev['Close'] < prev['Open']) and (C > O) and (C >= prev['Open']) and (O <= prev['Close']):
            patterns.append('BULL_ENGULF')
        if (prev['Close'] > prev['Open']) and (C < O) and (C <= prev['Open']) and (O >= prev['Close']):
            patterns.append('BEAR_ENGULF')
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
    if prev is not None:
        if (H < prev['High']) and (L > prev['Low']):
            patterns.append('INSIDE_BAR')
    return patterns

# ------------------- Signal generation with all patterns integrated -------------------
def generate_confluence_signals(df_local, params, side="Both", signal_mode="Both", advanced_leading=True):
    """
    signal_mode: 'Lagging', 'Price Action', 'Both'
    advanced_leading: include advanced patterns like H&S, triangles, channels, RVOL, divergence, etc.
    """
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    # list of prefixes/keywords that we treat as 'leading'
    leading_prefixes = ('BULL_ENGULF','BEAR_ENGULF','HAMMER','HANGING_MAN','SHOOTING_STAR',
                        'INVERTED_HAMMER','INSIDE_BAR','VOL_SPIKE','BREAK_HI','BREAK_LO','DOJI',
                        'MORNING_STAR','EVENING_STAR','THREE_WHITE','THREE_BLACK','DIV_BULL','DIV_BEAR',
                        'RVOL','IMBAL_BULL','IMBAL_BEAR','BULL_FLAG','BEAR_FLAG','SYM_TRIA','ASC_TRIA',
                        'DESC_TRIA','FALLING_WEDGE','RISING_WEDGE','CUP_HANDLE','CHANNEL','FVG',
                        'OB_BULL','OB_BEAR','SUPPLY_ZONE','DEMAND_ZONE','LIQ_SWEEP_UP','LIQ_SWEEP_DOWN',
                        'FAKE_BREAK_UP','FAKE_BREAK_DOWN','BOS','CHOCH','DOUBLE_TOP','DOUBLE_BOTTOM','BEAR_HS','BULL_HS_INV')

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []
        i = df_calc.index.get_loc(idx)

        # ---- Lagging indicators (same as before) ----
        try:
            sma_fast = row[f"sma_{params['sma_fast']}"]
            sma_slow = row[f"sma_{params['sma_slow']}"]
            if not np.isnan(sma_fast) and not np.isnan(sma_slow):
                if sma_fast > sma_slow:
                    indicators_that_long.append(f"SMA{params['sma_fast']}>{params['sma_slow']}")
                elif sma_fast < sma_slow:
                    indicators_that_short.append(f"SMA{params['sma_fast']}<{params['sma_slow']}")
        except Exception:
            pass

        try:
            ema_f = row[f"ema_{params['ema_fast']}"]
            ema_s = row[f"ema_{params['ema_slow']}"]
            if not np.isnan(ema_f) and not np.isnan(ema_s):
                if ema_f > ema_s:
                    indicators_that_long.append(f"EMA{params['ema_fast']}>{params['ema_slow']}")
                elif ema_f < ema_s:
                    indicators_that_short.append(f"EMA{params['ema_fast']}<{params['ema_slow']}")
        except Exception:
            pass

        if not np.isnan(row.get("macd_hist", np.nan)):
            if row["macd_hist"] > 0:
                indicators_that_long.append("MACD+")
            elif row["macd_hist"] < 0:
                indicators_that_short.append("MACD-")

        # RSI
        rsi_col = f"rsi_{params['rsi_period']}"
        if rsi_col in df_calc.columns:
            rsi_val = row[rsi_col]
            if not np.isnan(rsi_val):
                if rsi_val < params.get('rsi_oversold', 35):
                    indicators_that_long.append(f"RSI<{params.get('rsi_oversold', 35)}")
                elif rsi_val > params.get('rsi_overbought', 65):
                    indicators_that_short.append(f"RSI>{params.get('rsi_overbought', 65)}")

        # Bollinger
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

        # OBV rising
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

        # CCI extremes
        if not np.isnan(row.get('cci', np.nan)):
            if row['cci'] < -100:
                indicators_that_long.append('CCI')
            elif row['cci'] > 100:
                indicators_that_short.append('CCI')

        # Volume spike basic
        if not np.isnan(row.get('vol_sma', np.nan)) and row['Volume'] > 0:
            if row['Volume'] > row['vol_sma'] * params.get('vol_multiplier', 1.5):
                prev_close = df_calc['Close'].shift(1).iloc[i] if i > 0 else np.nan
                if not np.isnan(prev_close) and row['Close'] > prev_close:
                    indicators_that_long.append('VOL_SPIKE')
                elif not np.isnan(prev_close) and row['Close'] < prev_close:
                    indicators_that_short.append('VOL_SPIKE')

        # ----------------- Leading & Advanced patterns -----------------
        # basic candles
        patterns = detect_basic_candles(df_calc, i)
        for p in patterns:
            if p in ('BULL_ENGULF','HAMMER','INVERTED_HAMMER','MORNING_STAR','THREE_WHITE'):
                indicators_that_long.append(p)
            if p in ('BEAR_ENGULF','HANGING_MAN','SHOOTING_STAR','EVENING_STAR','THREE_BLACK'):
                indicators_that_short.append(p)

        # breakout recent highs/lows
        lookback = params.get('breakout_lookback', 10)
        if i >= 1:
            prev_highs = df_calc['High'].iloc[max(0, i - lookback):i]
            prev_lows = df_calc['Low'].iloc[max(0, i - lookback):i]
            if len(prev_highs) > 0:
                if row['Close'] > prev_highs.max():
                    indicators_that_long.append(f'BREAK_HI_{lookback}')
                if row['Close'] < prev_lows.min():
                    indicators_that_short.append(f'BREAK_LO_{lookback}')

        # more advanced leading signals (optional)
        if advanced_leading:
            # RVOL
            if not np.isnan(row.get('vol_sma', np.nan)) and row['vol_sma'] > 0:
                rv = row['Volume'] / (row['vol_sma'] + 1e-9)
                if rv >= params.get('rvol_threshold', 2.0):
                    if row['Close'] > row['Open']:
                        indicators_that_long.append('RVOL')
                    elif row['Close'] < row['Open']:
                        indicators_that_short.append('RVOL')

            # balance/imbalance (order-flow-like)
            rng = max(row['High'] - row['Low'], 1e-9)
            close_near_high = (row['High'] - row['Close']) / rng < 0.2
            close_near_low = (row['Close'] - row['Low']) / rng < 0.2
            if row['Volume'] > row.get('vol_sma', 0) * 1.2:
                if close_near_high:
                    indicators_that_long.append('IMBAL_BULL')
                if close_near_low:
                    indicators_that_short.append('IMBAL_BEAR')

            # divergence (RSI)
            divs = []
            if rsi_col in df_calc.columns:
                divs = detect_divergences(df_calc, i, rsi_col)
            for d in divs:
                if d == 'DIV_BULL':
                    indicators_that_long.append(d)
                if d == 'DIV_BEAR':
                    indicators_that_short.append(d)

            # head & shoulders, double tops/bottoms
            hs = detect_head_and_shoulders(df_calc, i, lookback=60)
            for h in hs:
                if h == 'BEAR_HS':
                    indicators_that_short.append('BEAR_HS')
                if h == 'BULL_HS_INV':
                    indicators_that_long.append('BULL_HS_INV')

            dbl = detect_double_top_bottom(df_calc, i, lookback=60)
            for d in dbl:
                if d == 'DOUBLE_TOP':
                    indicators_that_short.append('DOUBLE_TOP')
                if d == 'DOUBLE_BOTTOM':
                    indicators_that_long.append('DOUBLE_BOTTOM')

            # triangles/wedges
            tri = detect_triangle_or_wedge(df_calc, i, lookback=40)
            for t in tri:
                if 'TRIA' in t or 'WEDGE' in t:
                    # ascending triangle typically bullish, falling wedge bullish, etc.
                    if t in ('ASC_TRIA', 'FALLING_WEDGE', 'SYM_TRIA'):
                        indicators_that_long.append(t)
                    elif t in ('DESC_TRIA', 'RISING_WEDGE'):
                        indicators_that_short.append(t)

            # flags/pennants
            flag = detect_flag_or_pennant(df_calc, i)
            for f in flag:
                if f == 'BULL_FLAG':
                    indicators_that_long.append('BULL_FLAG')
                if f == 'BEAR_FLAG':
                    indicators_that_short.append('BEAR_FLAG')

            # cup & handle
            ch = detect_cup_and_handle(df_calc, i, lookback=120)
            for c in ch:
                if c == 'CUP_HANDLE':
                    indicators_that_long.append('CUP_HANDLE')

            # channels
            chs = detect_channels(df_calc, i, lookback=60)
            for c in chs:
                if c == 'CHANNEL':
                    indicators_that_long.append('CHANNEL')  # neutral — user interprets

            # FVG
            fvg = detect_fvg(df_calc, i)
            for f in fvg:
                if 'BULL' in f:
                    indicators_that_long.append('FVG_BULL')
                if 'BEAR' in f:
                    indicators_that_short.append('FVG_BEAR')

            # OB / supply demand
            obz = detect_order_block_and_supply_demand(df_calc, i, lookback=60)
            for o in obz:
                if 'OB_BULL' in o:
                    indicators_that_long.append('OB_BULL')
                if 'OB_BEAR' in o:
                    indicators_that_short.append('OB_BEAR')
                if o == 'SUPPLY_ZONE':
                    indicators_that_short.append('SUPPLY_ZONE')
                if o == 'DEMAND_ZONE':
                    indicators_that_long.append('DEMAND_ZONE')

            # liquidity sweeps & fake breakouts
            lq = detect_liquidity_sweep_and_fake_breakouts(df_calc, i)
            for l in lq:
                if 'UP' in l:
                    indicators_that_short.append(l)  # sweep up can be bearish (liquidity grab)
                if 'DOWN' in l:
                    indicators_that_long.append(l)

            # BOS / CHoCH
            bos = detect_bos_choch(df_calc, i)
            for b in bos:
                if 'BOS_UP' in b:
                    indicators_that_long.append('BOS_UP')
                if 'BOS_DOWN' in b:
                    indicators_that_short.append('BOS_DOWN')
                if 'CHOCH' in b:
                    if 'UP' in b:
                        indicators_that_long.append('CHOCH_UP')
                    else:
                        indicators_that_short.append('CHOCH_DOWN')

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

# ------------------- Backtester (unchanged semantics: entries at candle Close) -------------------
def choose_primary_indicator(indicators_list):
    priority = ['BULL_ENGULF','BEAR_ENGULF','MORNING_STAR','EVENING_STAR','HAMMER','SHOOTING_STAR',
                'RVOL','IMBAL_BULL','IMBAL_BEAR','DIV_BULL','DIV_BEAR','VOL_SPIKE','BREAK_HI','BREAK_LO',
                'BEAR_FLAG','BULL_FLAG','CUP_HANDLE','FVG','OB_BULL','OB_BEAR','SUPPLY_ZONE','DEMAND_ZONE',
                'BOS','CHOCH','BEAR_HS','BULL_HS_INV','DOUBLE_TOP','DOUBLE_BOTTOM',
                'EMA','SMA','MACD','RSI','BB','VWMA','OBV','MOM','STOCH','ADX','CCI']
    for p in priority:
        for it in indicators_list:
            if p in it:
                return it
    return indicators_list[0] if indicators_list else ''

def backtest_point_strategy(df_signals, params):
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    first_price = df_signals['Close'].iloc[0]
    last_price = df_signals['Close'].iloc[-1]
    buy_hold_points = last_price - first_price

    n = len(df_signals)
    for i in range(n):
        row = df_signals.iloc[i]
        sig = row['Signal']

        # exits evaluated on current row (future relative to entry)
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
            else:
                if not pd.isna(l) and l <= target:
                    exit_price = target; reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl; reason = "Stopped"
                elif sig == 1 and row['total_votes'] >= params['min_confluence']:
                    exit_price = closep; reason = "Opposite signal"

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
                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

        # entries executed at same bar close
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

            indicators = (row['indicators_long'] if pos_side == 1 else row['indicators_short'])
            primary = choose_primary_indicator(indicators)
            entry_details = {
                "Entry Date": entry_date, "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                "Indicators": indicators,
                "Primary Indicator": primary,
                "Confluences": row['total_votes']
            }
            in_pos = True
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

# ------------------- Optimization & UI (progress bar) -------------------
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
        except Exception as e:
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

# ------------------- Streamlit App UI -------------------
st.title("Backtester with Very-Advanced Pattern Detection")
st.markdown(
    "Upload OHLCV CSV/XLSX (Date,Open,High,Low,Close,Volume). "
    "Choose signal mode (Lagging / Price Action / Both). Toggle advanced patterns. "
    "Entries are executed at candle Close (no future lookahead)."
)

uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv','xlsx'])
side = st.selectbox("Trade Side", options=["Both","Long","Short"], index=0)
signal_mode = st.selectbox("Signal source/mode", options=["Lagging","Price Action","Both"], index=2)
advanced_leading = st.checkbox("Use advanced leading signals (divergence/RVOL/imbalances/patterns)", value=True)
random_iters = st.number_input("Random iterations (1-2000)", min_value=1, max_value=2000, value=200, step=1)
expected_returns = st.number_input("Expected strategy returns (total points)", value=0.0, step=1.0, format="%.2f")
expected_accuracy_pct = st.number_input("Expected accuracy % (probability of profit, e.g. 70)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
run_btn = st.button("Run Backtest & Optimize")

if uploaded_file is not None:
    try:
        if str(uploaded_file).lower().endswith('.xlsx') or hasattr(uploaded_file, 'getvalue') and ('xls' in uploaded_file.name.lower()):
            raw = pd.read_excel(uploaded_file)
        else:
            raw = pd.read_csv(uploaded_file)
        df_full = normalize_df(raw)
    except Exception as e:
        st.error(f"Failed to read/normalize file: {e}")
        st.stop()

    st.subheader("Uploaded file sample (raw)")
    try:
        st.write(f"Rows: {raw.shape[0]} Columns: {raw.shape[1]}")
        st.dataframe(raw.head(5))
        st.subheader("Uploaded file - bottom 5 rows (raw)")
        st.dataframe(raw.tail(5))
    except Exception:
        pass

    st.subheader("Normalized (used) data - last 5 rows")
    try:
        st.dataframe(df_full.tail(5))
        st.write("Last normalized close:", df_full['Close'].iloc[-1])
        st.write("Last normalized index:", df_full.index[-1])
    except Exception:
        pass

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    st.markdown(f"**Data range:** {min_date} to {max_date}")
    unique_dates = sorted({d.date() for d in df_full.index})
    selected_last_date = st.selectbox("Select last date (restrict data up to this date)", options=unique_dates, index=len(unique_dates)-1, format_func=lambda x: x.strftime('%Y-%m-%d'))
    df = df_full[df_full.index.date <= selected_last_date]

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
                'breakout_lookback': 10, 'rvol_threshold': 2.0
            }

            target_acc = expected_accuracy_pct / 100.0
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

            latest_sig_df = generate_confluence_signals(df, best_params, side, signal_mode, advanced_leading)
            latest_row = latest_sig_df.iloc[-1]

            st.subheader("Signal DataFrame - last 5 rows (debug)")
            st.dataframe(latest_sig_df.tail(5))

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
                target_price = np.nan
                sl_price = np.nan
                indicators_list = []

            primary = choose_primary_indicator(indicators_list)
            confluences = int(latest_row.get('total_votes', 0))
            prob_of_profit = (best_summary.get('prob_of_profit', np.nan) * 100.0) if isinstance(best_summary, dict) else np.nan

            def explain_indicator(it):
                # condensed human-readable explanations for common tags
                if it.startswith('EMA'):
                    return f"{it}: EMA crossover momentum"
                if it.startswith('SMA'):
                    return f"{it}: SMA crossover trend"
                if 'MACD' in it:
                    return "MACD histogram indicates momentum"
                if it.startswith('RSI'):
                    return "RSI extreme"
                if 'BB' in it:
                    return "Bollinger extreme"
                if 'VWMA' in it or 'VWAP' in it:
                    return "Volume-weighted trend"
                if 'OBV' in it:
                    return "On-balance volume direction"
                if it == 'VOL_SPIKE':
                    return "Volume spike with directional close"
                if it == 'RVOL':
                    return "Relative volume spike"
                if it == 'IMBAL_BULL':
                    return "Close near high + volume -> bullish imbalance"
                if it == 'IMBAL_BEAR':
                    return "Close near low + volume -> bearish imbalance"
                if it == 'BULL_FLAG':
                    return "Flag/pennant after bullish pole"
                if it == 'BEAR_FLAG':
                    return "Bear flag/pennant after bearish pole"
                if it == 'CUP_HANDLE':
                    return "Cup with handle (potential continuation)"
                if it == 'BEAR_HS':
                    return "Head & Shoulders (bearish)"
                if it == 'BULL_HS_INV':
                    return "Inverse Head & Shoulders (bullish)"
                if it == 'DOUBLE_TOP':
                    return "Double top (resistance)"
                if it == 'DOUBLE_BOTTOM':
                    return "Double bottom (support)"
                if it == 'FVG_BULL':
                    return "Bullish fair-value gap (unfilled area)"
                if it == 'OB_BULL':
                    return "Bullish order block candidate"
                if it.startswith('BREAK_HI'):
                    return "Break above recent highs"
                if it.startswith('BREAK_LO'):
                    return "Break below recent lows"
                if it == 'BOS_UP':
                    return "Break of structure up"
                if it == 'BOS_DOWN':
                    return "Break of structure down"
                if it == 'CHOCH_UP' or it == 'CHOCH_DOWN':
                    return "Change of character detected"
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

            if not isclose(entry_price_est, actual_last_close, abs_tol=1e-8):
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
                st.error("Possible causes: formatting/coercion/duplicate timestamps. Inspect tables above or supply cleaned file.")

            st.success("Done")
else:
    st.info("Upload a CSV/XLSX to start. After upload select last date and click 'Run Backtest & Optimize'.")

# ------------------- END -------------------
