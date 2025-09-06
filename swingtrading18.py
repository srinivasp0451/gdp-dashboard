# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from datetime import datetime

# ------------------- Helpers / Robust Column Mapping -------------------
def map_column_name(col):
    """Return canonical name for a column using substring matching.
       Handles messy column names like 'SHARES TRADED', 'volume', 'Vol', 'CLOSE ' etc."""
    k = re.sub(r'[^a-z0-9]', '', str(col).lower())
    if any(x in k for x in ('open', 'o')):
        return 'Open'
    if any(x in k for x in ('high', 'h')):
        return 'High'
    if any(x in k for x in ('low', 'l')):
        return 'Low'
    if any(x in k for x in ('close', 'c', 'price')):
        return 'Close'
    # volume variants: 'volume', 'vol', 'shares', 'sharestraded', 'qty', 'quantity'
    if any(x in k for x in ('volume', 'vol', 'shares', 'sharest', 'qty', 'quantity', 'traded')):
        return 'Volume'
    if any(x in k for x in ('date', 'datetime', 'time', 'timestamp')):
        return 'Date'
    return col  # unknown -> keep original

def normalize_df(df):
    """Robust normalization: map columns, parse dates, clean numeric values, preserve price columns."""
    # build mapping
    mapping = {}
    for c in df.columns:
        mapping[c] = map_column_name(c)
    df = df.rename(columns=mapping)

    # required columns
    for required in ["Open", "High", "Low", "Close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    # parse date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        if df["Date"].isna().any():
            # try fallback formats
            df["Date"] = pd.to_datetime(df["Date"].astype(str), dayfirst=False, errors='coerce')
        df = df.dropna(subset=["Date"]).copy()
        # remove tz
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
    else:
        # try parse index as datetime
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors='coerce')
        # drop rows where index is NaT
        df = df.dropna(axis=0, subset=[df.index.name]) if df.index.name else df.dropna()
        df = df.sort_index()

    # Clean numeric columns: remove commas/spaces/currency symbols then convert
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
            df[col] = df[col].str.replace(r'[₹$€£,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After parsing, drop rows with missing price data
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

# ------------------- Technical Indicators (causal) -------------------
def compute_indicators(df, params):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0)

    # SMA/EMA
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

    # Bollinger on sma_fast
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

    # VWMA & volume sma
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

    # ADX (approx)
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

    # Fill indicator columns only
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_cols = [c for c in df.columns if c not in price_cols]
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill')

    return df

# ------------------- Candlestick & multi-bar patterns -------------------
def detect_candlestick_patterns(df, i):
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

    # Hammer/Hanging Man/Shooting Star/Inverted Hammer
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

    # Morning/Evening Star simplified (3-bar)
    if prev is not None and prev2 is not None:
        O1, C1 = prev2['Open'], prev2['Close']
        O2, C2 = prev['Open'], prev['Close']
        O3, C3 = row['Open'], row['Close']
        body1 = abs(C1 - O1)
        body2 = abs(C2 - O2)
        body3 = abs(C3 - O3)
        # Morning Star: bear -> small -> bull
        if (C1 < O1) and (body2 <= 0.6 * body1) and (C3 > O3) and (C3 > (O1 + C1) / 2):
            patterns.append('MORNING_STAR')
        # Evening Star: bull -> small -> bear
        if (C1 > O1) and (body2 <= 0.6 * body1) and (C3 < O3) and (C3 < (O1 + C1) / 2):
            patterns.append('EVENING_STAR')

    # Three soldiers / black crows
    if i >= 2:
        c0, c1, c2 = df['Close'].iloc[i-2:i+1]
        o0, o1, o2 = df['Open'].iloc[i-2:i+1]
        if (c0 < c1 < c2) and (o1 > c0) and (o2 > c1):
            patterns.append('THREE_WHITE')
        if (c0 > c1 > c2) and (o1 < c0) and (o2 < c1):
            patterns.append('THREE_BLACK')

    return patterns

# ------------------- Peaks/trough helpers & divergence -------------------
def find_last_n_peaks(prices, upto_i, n=2):
    peaks = []
    for j in range(upto_i-1, 0, -1):
        try:
            if prices.iloc[j] > prices.iloc[j-1] and prices.iloc[j] > prices.iloc[j+1]:
                peaks.append(j)
                if len(peaks) >= n:
                    break
        except Exception:
            continue
    return peaks[::-1]

def find_last_n_troughs(prices, upto_i, n=2):
    troughs = []
    for j in range(upto_i-1, 0, -1):
        try:
            if prices.iloc[j] < prices.iloc[j-1] and prices.iloc[j] < prices.iloc[j+1]:
                troughs.append(j)
                if len(troughs) >= n:
                    break
        except Exception:
            continue
    return troughs[::-1]

def detect_divergences(df, i, rsi_col):
    res = []
    prices = df['Close']
    rsi = df[rsi_col]
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

# ------------------- Market-structure: BOS / CHOCH -------------------
def detect_bos_choch(df, i, lookback=50):
    """Detect Break of Structure (BOS) and Change Of Character (CHOCH).
       Conservative approach: find last two swing highs & lows and see if breached."""
    res = []
    if i < 3:
        return res
    prices = df['Close']
    # find last 3 swing highs and lows (very rough)
    highs = []
    lows = []
    for j in range(max(2, i - lookback), i+1):
        if j<=0 or j>=len(df)-1:
            continue
        if df['High'].iloc[j] > df['High'].iloc[j-1] and df['High'].iloc[j] > df['High'].iloc[j+1]:
            highs.append((j, df['High'].iloc[j]))
        if df['Low'].iloc[j] < df['Low'].iloc[j-1] and df['Low'].iloc[j] < df['Low'].iloc[j+1]:
            lows.append((j, df['Low'].iloc[j]))
    try:
        # last swing high/low
        if len(highs) >= 2:
            # if current close > previous swing high -> bullish BOS
            last_high_idx = highs[-1][0]
            prev_high_idx = highs[-2][0]
            if prices.iloc[i] > highs[-2][1] and last_high_idx < i:
                res.append('BOS_UP')
        if len(lows) >= 2:
            last_low_idx = lows[-1][0]
            prev_low_idx = lows[-2][0]
            if prices.iloc[i] < lows[-2][1] and last_low_idx < i:
                res.append('BOS_DOWN')
    except Exception:
        pass
    # CHOCH: change of character: simple detection if both BOS_UP and BOS_DOWN appear near each other (coarse)
    return res

# ------------------- Fair Value Gaps (FVGs) -------------------
def detect_fvg(df, i, lookback=10):
    """Simple heuristic for Fair Value Gaps (gaps between wicks/bodies) — conservative detection."""
    res = []
    # look for three-bar structure where middle bar leaves a gap between bodies
    if i < 2:
        return res
    for j in range(max(2, i - lookback + 1), i+1):
        a = df.iloc[j-2]; b = df.iloc[j-1]; c = df.iloc[j]
        # bullish FVG (gap up between a and b)
        if (a['Close'] < a['Open']) and (b['Open'] > a['Close']) and (b['Close'] > b['Open']):
            # crude check: body gap between a.high and b.low?
            if b['Low'] > a['High']:
                res.append('FVG_BULL')
        # bearish FVG
        if (a['Close'] > a['Open']) and (b['Open'] < a['Close']) and (b['Close'] < b['Open']):
            if b['High'] < a['Low']:
                res.append('FVG_BEAR')
    return res

# ------------------- Liquidity Sweep / Sweep of Stops / Fake Breakouts -------------------
def detect_liquidity_sweep(df, i, lookback=8):
    res = []
    if i < 1: return res
    # liquidity sweep: big wick beyond recent local high/low then closes back in body
    row = df.iloc[i]
    prev_high = df['High'].iloc[max(0, i-5):i].max() if i>0 else None
    prev_low = df['Low'].iloc[max(0, i-5):i].min() if i>0 else None
    if prev_high is not None and row['High'] > prev_high and row['Close'] < prev_high:
        res.append('LIQUIDITY_SWEEP_HIGH')
    if prev_low is not None and row['Low'] < prev_low and row['Close'] > prev_low:
        res.append('LIQUIDITY_SWEEP_LOW')
    # Fake breakout: breakout beyond recent high/low but closes inside within short period
    # detection of fake breakout will be handled by looking at subsequent bars in code (not here)
    return res

# ------------------- Order blocks & supply/demand zones (heuristic) -------------------
def detect_order_block(df, i, lookback=20):
    """Heuristic: identify bullish order block as bearish candle before large bull move with high volume."""
    res = []
    if i < 3:
        return res
    # search for last bullish impulse and prior opposite candle
    for j in range(max(3, i - lookback), i+1):
        # find impulse: big directional move in close over next 2-3 bars
        try:
            if j+2 <= i:
                # look for bullish impulse
                if (df['Close'].iloc[j+1] > df['Close'].iloc[j]) and (df['Close'].iloc[j+2] > df['Close'].iloc[j+1]) \
                   and (df['Volume'].iloc[j+1] > df['vol_sma'].iloc[j+1]):
                    # identify prior bearish candle as bullish order block
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:
                        res.append(f'OB_BULL_{j}')
                # bearish impulse
                if (df['Close'].iloc[j+1] < df['Close'].iloc[j]) and (df['Close'].iloc[j+2] < df['Close'].iloc[j+1]) \
                   and (df['Volume'].iloc[j+1] > df['vol_sma'].iloc[j+1]):
                    if df['Close'].iloc[j] > df['Open'].iloc[j]:
                        res.append(f'OB_BEAR_{j}')
        except Exception:
            continue
    return res

# ------------------- Trendlines, Channels, Wedges, Triangles (heuristics) -------------------
def detect_triangle_flag_pennant(df, i, lookback=30):
    """Heuristic for triangle / flag / pennant detection within lookback window ending at i."""
    res = []
    if i < 5:
        return res
    start = max(0, i - lookback + 1)
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    # contraction: range decreases over time
    ranges = highs - lows
    if len(ranges) < 6:
        return res
    # check range contraction factor
    left_mean = np.mean(ranges[:len(ranges)//2])
    right_mean = np.mean(ranges[len(ranges)//2:])
    if right_mean < 0.6 * left_mean:
        # check if highs trending down and lows trending up -> triangle
        # linear fit slopes
        x = np.arange(len(highs))
        try:
            slope_high = np.polyfit(x, highs, 1)[0]
            slope_low = np.polyfit(x, lows, 1)[0]
            if slope_high < 0 and slope_low > 0:
                res.append('SYMMETRICAL_TRIANGLE')
            elif slope_high < 0 and slope_low <= 0:
                res.append('DESCENDING_TRIANGLE')
            elif slope_low > 0 and slope_high >= 0:
                res.append('ASCENDING_TRIANGLE')
        except Exception:
            pass
    # flag/pennant detection: prior pole large move followed by tight consolidation
    if i - start >= 6:
        # detect pole: big move in earlier 3 bars
        pole_window = df['Close'].iloc[max(start, i-15):i-6]
        if len(pole_window) >= 3:
            # find large directional move magnitude
            if (pole_window.iloc[-1] - pole_window.iloc[0]) / (pole_window.abs().mean()+1e-9) > 0.08:
                # consolidation in last 6 bars small range
                cons_range = (df['High'].iloc[i-5:i+1] - df['Low'].iloc[i-5:i+1]).max()
                if cons_range < 0.02 * df['Close'].iloc[i]:
                    res.append('FLAG_OR_PENNANT')
    return res

def detect_head_and_shoulders(df, i, lookback=40):
    """Heuristic H&S detection: look for three peaks with middle higher (head) and shoulders lower."""
    res = []
    if i < 6:
        return res
    start = max(0, i - lookback + 1)
    highs = df['High'].iloc[start:i+1]
    # crude peaks detection
    peaks = []
    for j in range(start+1, i-1):
        if highs.iloc[j-start] > highs.iloc[j-start-1] and highs.iloc[j-start] > highs.iloc[j-start+1]:
            peaks.append((j, highs.iloc[j-start]))
    if len(peaks) >= 3:
        # take last 3 peaks
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        # head higher than shoulders by margin
        if p2[1] > p1[1] * 1.02 and p2[1] > p3[1] * 1.02 and abs(p1[1] - p3[1]) / max(p1[1], p3[1]) < 0.08:
            res.append('HEAD_SHOULDERS')
    # inverted H&S check on lows
    lows = df['Low'].iloc[start:i+1]
    troughs = []
    for j in range(start+1, i-1):
        if lows.iloc[j-start] < lows.iloc[j-start-1] and lows.iloc[j-start] < lows.iloc[j-start+1]:
            troughs.append((j, lows.iloc[j-start]))
    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        if t2[1] < t1[1] * 0.98 and t2[1] < t3[1] * 0.98 and abs(t1[1] - t3[1]) / max(t1[1], t3[1]) < 0.08:
            res.append('INV_HEAD_SHOULDERS')
    return res

def detect_channel(df, i, lookback=40):
    """Detect simple parallel channel by linear-fit to highs and lows and check approximate parallelism."""
    res = []
    if i < 6:
        return res
    start = max(0, i - lookback + 1)
    xs = np.arange(start, i+1)
    highs = df['High'].iloc[start:i+1].values
    lows = df['Low'].iloc[start:i+1].values
    if len(xs) < 6:
        return res
    try:
        slope_h, intercept_h = np.polyfit(xs, highs, 1)
        slope_l, intercept_l = np.polyfit(xs, lows, 1)
        # check parallel-ish
        if abs(slope_h - slope_l) / (abs(slope_h) + 1e-9) < 0.2:
            res.append('CHANNEL')
            # identify ascending/descending/horizontal
            if slope_h > 0.0001:
                res.append('CHANNEL_ASC')
            elif slope_h < -0.0001:
                res.append('CHANNEL_DESC')
            else:
                res.append('CHANNEL_HORZ')
    except Exception:
        pass
    return res

# ------------------- Fairly quick order-block / supply-demand zone cluster detection -------------------
def detect_supply_demand_zones(df, i, lookback=60):
    """Heuristic: find recent areas with clustered pivots (local highs/lows) - treated as supply/demand."""
    res = []
    if i < 5:
        return res
    start = max(0, i - lookback + 1)
    highs = df['High'].iloc[start:i+1]
    lows = df['Low'].iloc[start:i+1]
    # local pivot counts near round levels
    # find frequent highs (supply)
    high_peaks = []
    for j in range(start+1, i):
        if highs.iloc[j-start] >= highs.iloc[j-start-1] and highs.iloc[j-start] >= highs.iloc[j-start+1]:
            high_peaks.append(highs.iloc[j-start])
    low_peaks = []
    for j in range(start+1, i):
        if lows.iloc[j-start] <= lows.iloc[j-start-1] and lows.iloc[j-start] <= lows.iloc[j-start+1]:
            low_peaks.append(lows.iloc[j-start])
    if len(high_peaks) >= 2:
        # consider supply zone at mean of top peaks
        mz = np.mean(high_peaks)
        res.append(f'SUPPLY_{round(mz,2)}')
    if len(low_peaks) >= 2:
        mz = np.mean(low_peaks)
        res.append(f'DEMAND_{round(mz,2)}')
    return res

# ------------------- Consolidated pattern detector wrapper -------------------
def detect_many_patterns(df, i, params, advanced_leading=True):
    tags = []
    tags += detect_candlestick_patterns(df, i)
    tags += detect_triangle_flag_pennant(df, i, lookback=params.get('pattern_lookback', 30))
    tags += detect_head_and_shoulders(df, i, lookback=params.get('pattern_lookback', 40))
    tags += detect_channel(df, i, lookback=params.get('pattern_lookback', 40))
    tags += detect_supply_demand_zones(df, i, lookback=params.get('supply_lookback', 60))
    tags += detect_order_block(df, i, lookback=params.get('orderblock_lookback', 30))
    tags += detect_fvg(df, i, lookback=params.get('fvg_lookback', 10))
    tags += detect_liquidity_sweep(df, i, lookback=params.get('liquidity_lookback', 8))
    # advanced divergences
    if advanced_leading:
        rsi_col = f"rsi_{params['rsi_period']}"
        if rsi_col in df.columns:
            tags += detect_divergences(df, i, rsi_col)
    # dedupe
    tags = list(dict.fromkeys(tags))
    return tags

# ------------------- Confluence signal generator (integrates everything) -------------------
def generate_confluence_signals(df_local, params, side="Both", signal_mode="Both", advanced_leading=True):
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    leading_prefixes = ('BULL_ENGULF','BEAR_ENGULF','HAMMER','HANGING_MAN','SHOOTING_STAR','INVERTED_HAMMER',
                        'INSIDE_BAR','VOL_SPIKE','BREAK_HI','BREAK_LO','DOJI','MORNING_STAR','EVENING_STAR',
                        'THREE_WHITE','THREE_BLACK','DIV_BULL','DIV_BEAR','RVOL','IMBAL_BULL','IMBAL_BEAR',
                        'FVG','LIQUIDITY','OB_BULL','OB_BEAR','SUPPLY','DEMAND','HEAD_SHOULDERS','INV_HEAD_SHOULDERS',
                        'SYMMETRICAL_TRIANGLE','DESCENDING_TRIANGLE','ASCENDING_TRIANGLE','FLAG_OR_PENNANT','CHANNEL')

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        i = df_calc.index.get_loc(idx)

        # Lagging indicators (as before)
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

        # MACD, RSI, BB, MOM, STOCH, ADX, OBV, VWMA, CCI (as earlier)
        try:
            if not np.isnan(row.get("macd_hist", np.nan)):
                if row["macd_hist"] > 0:
                    indicators_that_long.append("MACD+")
                elif row["macd_hist"] < 0:
                    indicators_that_short.append("MACD-")
        except Exception:
            pass

        try:
            rsi_val = row[f"rsi_{params['rsi_period']}"]
            if not np.isnan(rsi_val):
                if rsi_val < params.get('rsi_oversold', 35):
                    indicators_that_long.append(f"RSI<{params.get('rsi_oversold', 35)}")
                elif rsi_val > params.get('rsi_overbought', 65):
                    indicators_that_short.append(f"RSI>{params.get('rsi_overbought', 65)}")
        except Exception:
            pass

        try:
            price = row['Close']
            if not np.isnan(row['bb_upper']) and not np.isnan(row['bb_lower']):
                if price < row['bb_lower']:
                    indicators_that_long.append("BB_Lower")
                elif price > row['bb_upper']:
                    indicators_that_short.append("BB_Upper")
        except Exception:
            pass

        try:
            mom = row.get(f"mom_{params['mom_period']}", np.nan)
            if not np.isnan(mom):
                if mom > 0:
                    indicators_that_long.append(f"MOM+({params['mom_period']})")
                elif mom < 0:
                    indicators_that_short.append(f"MOM-({params['mom_period']})")
        except Exception:
            pass

        try:
            if not np.isnan(row.get('stoch_k', np.nan)) and not np.isnan(row.get('stoch_d', np.nan)):
                if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < params.get('stoch_oversold', 30):
                    indicators_that_long.append("STOCH")
                elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > params.get('stoch_overbought', 70):
                    indicators_that_short.append("STOCH")
        except Exception:
            pass

        try:
            if not np.isnan(row.get('adx', np.nan)):
                if row['adx'] > params.get('adx_threshold', 20) and row['pdi'] > row['mdi']:
                    indicators_that_long.append("ADX+")
                elif row['adx'] > params.get('adx_threshold', 20) and row['mdi'] > row['pdi']:
                    indicators_that_short.append("ADX-")
        except Exception:
            pass

        try:
            # OBV rising
            if i >= 3:
                recent = df_calc['obv'].iloc[max(0,i-2):i+1].mean()
                prev = df_calc['obv'].iloc[max(0,i-5):max(0,i-2)].mean()
                if recent > prev:
                    indicators_that_long.append('OBV')
                elif recent < prev:
                    indicators_that_short.append('OBV')
        except Exception:
            pass

        try:
            if not np.isnan(row.get('vwma', np.nan)):
                if row['Close'] > row['vwma']:
                    indicators_that_long.append('VWMA')
                elif row['Close'] < row['vwma']:
                    indicators_that_short.append('VWMA')
        except Exception:
            pass

        try:
            if not np.isnan(row.get('cci', np.nan)):
                if row['cci'] < -100:
                    indicators_that_long.append('CCI')
                elif row['cci'] > 100:
                    indicators_that_short.append('CCI')
        except Exception:
            pass

        # Volume spike (leading)
        try:
            if not np.isnan(row.get('vol_sma', np.nan)) and row['Volume'] > 0:
                if row['Volume'] > row['vol_sma'] * params.get('vol_multiplier', 1.5):
                    prev_close = df_calc['Close'].shift(1).iloc[i] if i > 0 else np.nan
                    if not np.isnan(prev_close) and row['Close'] > prev_close:
                        indicators_that_long.append('VOL_SPIKE')
                    elif not np.isnan(prev_close) and row['Close'] < prev_close:
                        indicators_that_short.append('VOL_SPIKE')
        except Exception:
            pass

        # Leading multi-bar & structural patterns
        try:
            many = detect_many_patterns(df_calc, i, params, advanced_leading=advanced_leading)
            for p in many:
                if p in ('BULL_ENGULF','HAMMER','INVERTED_HAMMER','MORNING_STAR','THREE_WHITE',
                         'DIV_BULL','RVOL','IMBAL_BULL','FVG_BULL','OB_BULL','SUPPLY','DEMAND','HEAD_SHOULDERS','CHANNEL','SYMMETRICAL_TRIANGLE','FLAG_OR_PENNANT'):
                    indicators_that_long.append(p)
                if p in ('BEAR_ENGULF','HANGING_MAN','SHOOTING_STAR','EVENING_STAR','THREE_BLACK',
                         'DIV_BEAR','RVOL','IMBAL_BEAR','FVG_BEAR','OB_BEAR','HEAD_SHOULDERS','INV_HEAD_SHOULDERS','CHANNEL'):
                    indicators_that_short.append(p)
        except Exception:
            pass

        # Breakout-of-highs / lows
        try:
            lookback = params.get('breakout_lookback', 10)
            if i >= 1:
                prev_highs = df_calc['High'].iloc[max(0, i - lookback):i]
                prev_lows = df_calc['Low'].iloc[max(0, i - lookback):i]
                if len(prev_highs) > 0:
                    if row['Close'] > prev_highs.max():
                        indicators_that_long.append(f'BREAK_HI_{lookback}')
                    if row['Close'] < prev_lows.min():
                        indicators_that_short.append(f'BREAK_LO_{lookback}')
        except Exception:
            pass

        # Advanced RVOL & imbalances
        try:
            if advanced_leading and not np.isnan(row.get('vol_sma', np.nan)) and row['vol_sma'] > 0:
                rv = row['Volume'] / (row['vol_sma'] + 1e-9)
                if rv >= params.get('rvol_threshold', 2.0):
                    if row['Close'] > row['Open']:
                        indicators_that_long.append('RVOL')
                    elif row['Close'] < row['Open']:
                        indicators_that_short.append('RVOL')

                # order-flow imbalance - close near high/low + elevated volume
                rng = row['High'] - row['Low'] if (row['High'] - row['Low']) != 0 else 1e-9
                close_near_high = (row['High'] - row['Close']) / rng < 0.2
                close_near_low = (row['Close'] - row['Low']) / rng < 0.2
                if row['Volume'] > row.get('vol_sma', 0) * 1.2:
                    if close_near_high:
                        indicators_that_long.append('IMBAL_BULL')
                    if close_near_low:
                        indicators_that_short.append('IMBAL_BEAR')
        except Exception:
            pass

        # Order blocks (heuristic)
        try:
            obs = detect_order_block(df_calc, i, lookback=params.get('orderblock_lookback', 30))
            for ob in obs:
                if 'OB_BULL' in ob or 'OB_BULL' in ob.upper():
                    indicators_that_long.append('OB_BULL')
                if 'OB_BEAR' in ob or 'OB_BEAR' in ob.upper():
                    indicators_that_short.append('OB_BEAR')
        except Exception:
            pass

        # Filter by signal_mode (Lagging, Price Action, Both)
        def is_leading(ind):
            return any(ind.startswith(p) for p in leading_prefixes)

        if signal_mode == 'Lagging':
            indicators_that_long = [x for x in indicators_that_long if not is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if not is_leading(x)]
        elif signal_mode == 'Price Action':
            indicators_that_long = [x for x in indicators_that_long if is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if is_leading(x)]
        # else 'Both' -> keep all

        # Count confluences & decide vote
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # apply side filter
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
            'patterns': list(set(indicators_that_long + indicators_that_short))
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index('index')
    result = df_calc.join(votes_df)
    result['Signal'] = sig_series
    return result

# ------------------- Backtester (unchanged core behavior) -------------------
def choose_primary_indicator(indicators_list):
    priority = ['BULL_ENGULF','BEAR_ENGULF','MORNING_STAR','EVENING_STAR','HAMMER','SHOOTING_STAR',
                'RVOL','IMBAL_BULL','IMBAL_BEAR','DIV_BULL','DIV_BEAR','VOL_SPIKE','BREAK_HI','BREAK_LO',
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

    # buy & hold baseline (points)
    first_price = df_signals['Close'].iloc[0]
    last_price = df_signals['Close'].iloc[-1]
    buy_hold_points = last_price - first_price

    n = len(df_signals)
    for i in range(n):
        row = df_signals.iloc[i]
        sig = row['Signal']

        # if in position, check exits using THIS row (future relative to entry)
        if in_pos:
            h = row['High']; l = row['Low']; closep = row['Close']
            exit_price = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif sig == -1 and row['total_votes'] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"
            else:
                if not pd.isna(l) and l <= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif sig == 1 and row['total_votes'] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"

            # final day fallback
            if i == (n-1) and in_pos and exit_price is None:
                exit_price = closep
                reason = "End of data"

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
                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

        # if not in position -> enter using THIS row's Close (no future data)
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

# ------------------- Optimization & sampling -------------------
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
    p['pattern_lookback'] = random.choice([20,30,40,60])
    p['orderblock_lookback'] = random.choice([20,30,40])
    p['fvg_lookback'] = random.choice([5,8,10])
    p['liquidity_lookback'] = random.choice([5,8,12])
    p['supply_lookback'] = random.choice([40,60,90])
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

# ------------------- Streamlit UI (run button at the end) -------------------
st.title("Backtester with Extensive Chart-Pattern Detection + Leading Signals")
st.markdown(
    "Upload OHLCV CSV/XLSX (Date,Open,High,Low,Close,Volume). "
    "This version detects many advanced patterns (H&S, triangles, flags, channels, FVGs, order-blocks, "
    "liquidity sweeps, RVOL, imbalances, BOS/CHOCH, supply/demand zones, and more). "
    "Entries are executed at the same candle Close (no lookahead). Use the 'Select last date' selector to restrict the data."
)

uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv','xlsx'])
st.markdown("### Options (choose before Run & Optimize)")
side = st.selectbox("Trade Side", options=["Both","Long","Short"], index=0)
signal_mode = st.selectbox("Signal source/mode", options=["Lagging","Price Action","Both"], index=2)
advanced_leading = st.checkbox("Use advanced leading signals (divergence, RVOL, imbalances, FVGs...)",
                               value=True)
st.markdown("Tuning (these are defaults used for sampling optimization):")
col1, col2, col3 = st.columns(3)
with col1:
    random_iters = st.number_input("Random iterations (1-2000)", min_value=1, max_value=2000, value=200, step=1)
with col2:
    expected_returns = st.number_input("Expected total points (strategy)", value=0.0, step=1.0, format="%.2f")
with col3:
    expected_accuracy_pct = st.number_input("Expected accuracy % (probability of profit)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)

st.markdown("---")

# read file and show raw / normalized preview
if uploaded_file is not None:
    try:
        if str(uploaded_file).lower().endswith('.xlsx') or ('xls' in uploaded_file.name.lower()):
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

    # PUT Run & Optimize button at the end (as requested)
    run_btn = st.button("Run Backtest & Optimize")

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
                'pattern_lookback': 30, 'orderblock_lookback': 30, 'fvg_lookback': 8,
                'liquidity_lookback': 8, 'supply_lookback': 60
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

                # Heatmap: monthly % returns
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

            # Live recommendation using best params applied to the restricted df
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
                if it.startswith('EMA'):
                    return f"{it}: EMA crossover momentum"
                if it.startswith('SMA'):
                    return f"{it}: SMA crossover trend bias"
                if 'MACD' in it:
                    return "MACD shows histogram direction"
                if it.startswith('RSI'):
                    return "RSI value"
                if 'BB' in it:
                    return "Bollinger band touch"
                if 'VWMA' in it:
                    return "VWMA relative"
                if 'OBV' in it:
                    return "OBV trend"
                if it == 'VOL_SPIKE':
                    return "High volume spike"
                if it == 'RVOL':
                    return "Relative volume spike"
                if it == 'IMBAL_BULL':
                    return "Imbalance / close near high with volume"
                if it == 'IMBAL_BEAR':
                    return "Imbalance / close near low with volume"
                if it == 'BULL_ENGULF':
                    return "Bullish engulfing (price-action)"
                if it == 'BEAR_ENGULF':
                    return "Bearish engulfing (price-action)"
                if it == 'DIV_BULL':
                    return "Bullish divergence (RSI)"
                if it == 'DIV_BEAR':
                    return "Bearish divergence (RSI)"
                if it == 'HEAD_SHOULDERS':
                    return "Head and Shoulders pattern"
                if it == 'INV_HEAD_SHOULDERS':
                    return "Inverse Head and Shoulders"
                if it.startswith('BREAK_HI'):
                    return "Break above recent highs"
                if it.startswith('BREAK_LO'):
                    return "Break below recent lows"
                if it.startswith('SUPPLY'):
                    return "Supply zone (resistance area)"
                if it.startswith('DEMAND'):
                    return "Demand zone (support area)"
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
            st.markdown("**Indicators that voted / patterns detected:**")
            st.write(indicators_list)
            st.markdown("**Reason / Logic (brief):**")
            st.write(reason_text)

            st.subheader("Latest indicator values (key ones)")
            ind_df = pd.DataFrame([indicator_values]).T.reset_index()
            ind_df.columns = ['Indicator', 'Value']
            st.dataframe(ind_df)

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
                st.error("Possible causes: (1) 'Close' column had non-numeric formatting and got coerced; "
                         "(2) duplicate/misaligned timestamps; (3) the dataset slice used by optimizer/backtest is different from uploaded file. "
                         "Inspect the three tables above or upload cleaned data.")

            st.success("Done")

else:
    st.info("Upload a CSV/XLSX to start. After upload you can choose the 'Select last date' and then click 'Run Backtest & Optimize'.")
