# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from scipy.stats import linregress

# ------------------- Helpers / Normalization -------------------
def normalize_df(df):
    """
    Robust normalization:
     - maps common column name variations (any case) to Open/High/Low/Close/Volume/Date
     - strips commas, spaces, currency symbols, words like 'Shares' from numeric columns
     - parses Date and sets as datetime index
    """
    # build mapping from lower-case column name -> original name
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for key, orig in cols.items():
        if key in ("open",):
            mapping[orig] = "Open"
        if key in ("high",):
            mapping[orig] = "High"
        if key in ("low",):
            mapping[orig] = "Low"
        if key in ("close",):
            mapping[orig] = "Close"
        if key in ("volume", "vol"):
            mapping[orig] = "Volume"
        if key in ("date", "datetime", "time", "timestamp"):
            mapping[orig] = "Date"
    df = df.rename(columns=mapping)

    # required columns
    for required in ["Open", "High", "Low", "Close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    # parse date if present else ensure index is datetime-like
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].isna().any():
            df["Date"] = pd.to_datetime(df["Date"].astype(str), dayfirst=False, errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(axis=0, subset=[df.index.name])
        df = df.sort_index()

    # Clean numeric columns robustly: remove commas, spaces, currency symbols, 'Shares' etc
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        # ensure string then remove any non-digit except dot and minus
        df[col] = df[col].astype(str).str.strip()
        # remove commas and spaces quickly
        df[col] = df[col].str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        # remove common currency symbols and words
        df[col] = df[col].str.replace(r"[₹$€£,A-Za-z]+", "", regex=True)
        # remove any remaining non-numeric except . and -
        df[col] = df[col].str.replace(r"[^0-9\.\-]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ------------------- Technical Indicators (causal) -------------------
def compute_indicators(df, params):
    """
    Causal indicator computation. Only indicator columns are forward/backfilled.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0)

    # moving averages
    df[f"sma_{params['sma_fast']}"] = close.rolling(params['sma_fast'], min_periods=1).mean()
    df[f"sma_{params['sma_slow']}"] = close.rolling(params['sma_slow'], min_periods=1).mean()
    df[f"ema_{params['ema_fast']}"] = close.ewm(span=params['ema_fast'], adjust=False).mean()
    df[f"ema_{params['ema_slow']}"] = close.ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD
    ema_f = df[f"ema_{params['ema_fast']}"]
    ema_s = df[f"ema_{params['ema_slow']}"]
    macd = ema_f - ema_s
    macd_signal = macd.ewm(span=params.get("macd_signal", 9), adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # RSI (Wilder)
    period = params["rsi_period"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Bollinger
    ma = df[f"sma_{params['sma_fast']}"]
    std = close.rolling(params['sma_fast'], min_periods=1).std()
    df["bb_upper"] = ma + params.get("bb_mult", 2) * std
    df["bb_lower"] = ma - params.get("bb_mult", 2) * std

    # Momentum
    df[f"mom_{params['mom_period']}"] = close - close.shift(params["mom_period"])

    # Stochastic
    stoch_k = (
        (close - low.rolling(params["stoch_period"], min_periods=1).min())
        / (high.rolling(params["stoch_period"], min_periods=1).max() - low.rolling(params["stoch_period"], min_periods=1).min()).replace(
            0, np.nan
        )
    ) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).fillna(0).cumsum()
    df["obv"] = obv

    # VWMA & volume sma
    wsum = (close * vol).rolling(params.get("vwma_period", 14)).sum()
    vsum = vol.rolling(params.get("vwma_period", 14)).sum().replace(0, np.nan)
    df["vwma"] = (wsum / vsum).fillna(df["Close"])
    df["vol_sma"] = vol.rolling(params.get("vol_sma_period", 20), min_periods=1).mean()

    # CCI
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(params["cci_period"], min_periods=1).mean()
    md = tp.rolling(params["cci_period"], min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
    df["cci"] = (tp - tp_ma) / (0.015 * md)

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{params['atr_period']}"] = tr.rolling(params["atr_period"], min_periods=1).mean()

    # ADX, PDI, MDI approx
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_for_adx = tr.ewm(alpha=1 / params["adx_period"], adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / params["adx_period"], adjust=False).mean() / atr_for_adx)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / params["adx_period"], adjust=False).mean() / atr_for_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(alpha=1 / params["adx_period"], adjust=False).mean()
    df["pdi"] = plus_di
    df["mdi"] = minus_di

    # only fill indicator columns
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    indicator_cols = [c for c in df.columns if c not in price_cols]
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].fillna(method="ffill").fillna(method="bfill")

    return df

# ------------------- Pattern helpers (many heuristics) -------------------
def slope_over_range(series, start_i, end_i):
    """Return slope (per bar) of linear regression on series[start_i:end_i+1]"""
    if end_i - start_i < 2:
        return 0.0
    y = series.iloc[start_i:end_i + 1].values
    x = np.arange(len(y))
    res = linregress(x, y)
    return res.slope

def detect_flag(df, i, lookback=20, pole_min_ratio=1.5):
    """
    Flag detection heuristic:
     - find a recent sharp move (flagpole) then small opposing consolidation.
    """
    n = len(df)
    if i < 3 or i - lookback < 0:
        return False
    # search for a strong impulse up within the last lookback bars ending at i
    window = df["Close"].iloc[max(0, i - lookback):i + 1]
    # simple pole: large difference between max and min in the window's first half vs second half
    half = max(1, len(window) // 2)
    left_range = window.iloc[:half].max() - window.iloc[:half].min() if half > 0 else 0
    right_range = window.iloc[half:].max() - window.iloc[half:].min() if half < len(window) else 0
    if left_range <= 0:
        return False
    ratio = right_range / (left_range + 1e-9)
    # we want a sharp move then a narrow consolidation (ratio << 1)
    return ratio < 0.4 and right_range > 0 and left_range > 0

def detect_triangle(df, i, lookback=20):
    """
    Triangle heuristic: contracting highs and lows over lookback.
    """
    if i < 6 or i - lookback < 0:
        return False
    highs = df["High"].iloc[i - lookback + 1:i + 1].values
    lows = df["Low"].iloc[i - lookback + 1:i + 1].values
    # compute high and low linear slopes
    slope_high = slope_over_range(pd.Series(highs), 0, len(highs) - 1)
    slope_low = slope_over_range(pd.Series(lows), 0, len(lows) - 1)
    # triangle roughly has opposite-sign slopes or both small magnitude with contraction
    contraction = (np.std(highs) < np.std(df["High"].iloc[max(0, i - lookback*2):i + 1].values) * 0.7) and \
                  (np.std(lows) < np.std(df["Low"].iloc[max(0, i - lookback*2):i + 1].values) * 0.7)
    return contraction and abs(slope_high) < 0.05 and abs(slope_low) < 0.05

def detect_head_shoulders(df, i, lookback=30):
    """
    Simplified H&S detector: find three peaks with middle higher than sides.
    """
    if i < 6 or i - lookback < 0:
        return False, None
    segment = df.iloc[max(0, i - lookback):i + 1]
    peaks_idx = []
    for j in range(1, len(segment) - 1):
        if segment["High"].iloc[j] > segment["High"].iloc[j - 1] and segment["High"].iloc[j] > segment["High"].iloc[j + 1]:
            peaks_idx.append(segment.index[j])
    if len(peaks_idx) < 3:
        return False, None
    # pick last 3 peaks
    p3 = peaks_idx[-3:]
    hvals = [df.loc[idx]["High"] for idx in p3]
    # head is middle and higher
    if hvals[1] > hvals[0] and hvals[1] > hvals[2]:
        return True, ("HS", p3)
    return False, None

def detect_wedge(df, i, lookback=20):
    """Detect rising/falling wedge by slopes of highs and lows moving together towards each other"""
    if i < 6 or i - lookback < 0:
        return False
    highs = df["High"].iloc[i - lookback + 1:i + 1]
    lows = df["Low"].iloc[i - lookback + 1:i + 1]
    slope_h = slope_over_range(highs, 0, len(highs) - 1)
    slope_l = slope_over_range(lows, 0, len(lows) - 1)
    # wedge: highs and lows converge; slopes roughly same sign but |slope_h - slope_l| small and range contracting
    range_now = (highs.max() - lows.min())
    range_old = (df["High"].iloc[max(0, i - lookback*2):i - lookback + 1].max() - df["Low"].iloc[max(0, i - lookback*2):i - lookback + 1].min()) if (i - lookback*2) >= 0 else range_now * 1.1
    if range_old <= 0:
        return False
    contracting = range_now < range_old * 0.8
    return contracting and abs(slope_h - slope_l) < 0.05

def detect_channel(df, i, lookback=30):
    """Detect approximate parallel channel by regression slopes of highs and lows and stable channel width"""
    if i < 10 or i - lookback < 0:
        return False
    highs = df["High"].iloc[i - lookback + 1:i + 1]
    lows = df["Low"].iloc[i - lookback + 1:i + 1]
    slope_h = slope_over_range(highs, 0, len(highs) - 1)
    slope_l = slope_over_range(lows, 0, len(lows) - 1)
    width = highs.mean() - lows.mean()
    if width <= 0:
        return False
    stable = highs.std() < highs.mean()*0.05 and lows.std() < lows.mean()*0.05
    return abs(slope_h - slope_l) < 0.02 and stable

def detect_liquidity_sweep(df, i, lookback=5):
    """
    Liquidity sweep: a wick that runs beyond recent swing high/low and then price returns.
    """
    if i < 2:
        return False, None
    cur = df.iloc[i]
    prev_high = df["High"].iloc[max(0, i - lookback):i].max()
    prev_low = df["Low"].iloc[max(0, i - lookback):i].min()
    # sweep above
    if cur["High"] > prev_high and cur["Close"] < prev_high:
        return True, "SWEEP_HIGH"
    # sweep below
    if cur["Low"] < prev_low and cur["Close"] > prev_low:
        return True, "SWEEP_LOW"
    return False, None

def detect_fake_breakout(df, i, lookback=5):
    """
    Fake breakout: price breaks above recent high but closes back inside / below the breakout level.
    """
    if i < 1:
        return False, None
    cur = df.iloc[i]
    prev_high = df["High"].iloc[max(0, i - lookback):i].max()
    prev_low = df["Low"].iloc[max(0, i - lookback):i].min()
    if cur["High"] > prev_high and cur["Close"] < prev_high:
        return True, "FAKE_UP"
    if cur["Low"] < prev_low and cur["Close"] > prev_low:
        return True, "FAKE_DOWN"
    return False, None

def detect_order_block(df, i, lookback=5):
    """
    Order block heuristic: a large directional candle followed by consolidation.
    """
    if i < 2:
        return False, None
    cur = df.iloc[i]
    # find previous bullish/bearish big candle
    window = df.iloc[max(0, i - lookback * 2):i]
    if window.empty:
        return False, None
    # large body relative to recent
    bodies = (window["Close"] - window["Open"]).abs()
    if bodies.mean() == 0:
        return False, None
    # if last significant candle was bullish and big
    big_idx = bodies.idxmax()
    body_val = bodies.max()
    avg_body = bodies.mean()
    if body_val > avg_body * 2.0:
        # bullish order block if that big candle was bullish (close>open)
        if df.loc[big_idx]["Close"] > df.loc[big_idx]["Open"]:
            return True, "ORDER_BLOCK_BUY"
        else:
            return True, "ORDER_BLOCK_SELL"
    return False, None

def detect_fvg(df, i):
    """
    Simple Fair Value Gap (FVG) detection (3-candle template):
    bullish FVG if candle i-2 is bearish and i-1 is bullish with gap between bodies.
    We'll return 'FVG_BULL' / 'FVG_BEAR' when a gap-like region exists.
    """
    if i < 2:
        return False, None
    a = df.iloc[i - 2]
    b = df.iloc[i - 1]
    c = df.iloc[i]
    # bullish FVG: a bearish, b bullish and b.low > a.high (rare in OHLC, but check minimal body overlap)
    if (a["Close"] < a["Open"]) and (b["Close"] > b["Open"]) and (b["Low"] > a["High"]):
        return True, "FVG_BULL"
    # bearish FVG
    if (a["Close"] > a["Open"]) and (b["Close"] < b["Open"]) and (b["High"] < a["Low"]):
        return True, "FVG_BEAR"
    return False, None

def detect_bos_choch(df, i, lookback=20):
    """
    Simple BOS/CHoCH detection using pivots:
      - Identify last two swing highs/lows and check if price has broken structure.
    """
    if i < 5:
        return None
    prices = df["Close"].iloc[:i + 1]
    # find last two peaks and troughs
    peaks = []
    troughs = []
    for j in range(1, i):
        if df["High"].iloc[j] > df["High"].iloc[j - 1] and df["High"].iloc[j] > df["High"].iloc[j + 1]:
            peaks.append(j)
        if df["Low"].iloc[j] < df["Low"].iloc[j - 1] and df["Low"].iloc[j] < df["Low"].iloc[j + 1]:
            troughs.append(j)
    # check Break Of Structure: last close breaks last swing high/low
    if peaks:
        last_peak = peaks[-1]
        if df["Close"].iloc[i] > df["High"].iloc[last_peak]:
            return "BOS_UP"
    if troughs:
        last_trough = troughs[-1]
        if df["Close"].iloc[i] < df["Low"].iloc[last_trough]:
            return "BOS_DOWN"
    # CHoCH: change of character - check if previous structure flips (simplified)
    return None

def detect_supply_demand_zone(df, i, lookback=30):
    """
    Very simple SD zone detection: find consolidation area followed by a strong move away with volume.
    Mark the consolidation as SD zone.
    """
    if i < 10:
        return None
    segment = df.iloc[max(0, i - lookback):i + 1]
    # consolidation where highs and lows have low std-dev
    if segment["High"].std() < segment["High"].mean() * 0.03 and segment["Low"].std() < segment["Low"].mean() * 0.03:
        # check next strong move away (we're at i so check current bar direction)
        cur = df.iloc[i]
        if cur["Volume"] > segment["Volume"].mean() * 1.5:
            if cur["Close"] > segment["High"].max():
                return "SUPPLY_BREAK"  # actually demand -> breakout upward
            if cur["Close"] < segment["Low"].min():
                return "DEMAND_BREAK"  # supply -> breakdown downward
    return None

# ------------------- Candlestick & Leading Signals (existing + advanced patterns) -------------------
def detect_candlestick_patterns(df, i):
    """Existing candlestick detection plus multi-bar patterns (morning/evening/three soldiers, etc.)"""
    patterns = []
    n = len(df)
    if i < 0 or i >= n:
        return patterns
    row = df.iloc[i]
    prev = df.iloc[i - 1] if i - 1 >= 0 else None
    prev2 = df.iloc[i - 2] if i - 2 >= 0 else None

    O = row["Open"]; H = row["High"]; L = row["Low"]; C = row["Close"]
    body = abs(C - O)
    lower_shadow = min(O, C) - L
    upper_shadow = H - max(O, C)
    rng = max(H - L, 1e-9)

    # Doji
    if body <= 0.1 * rng:
        patterns.append("DOJI")

    # Engulfing
    if prev is not None:
        if (prev["Close"] < prev["Open"]) and (C > O) and (C >= prev["Open"]) and (O <= prev["Close"]):
            patterns.append("BULL_ENGULF")
        if (prev["Close"] > prev["Open"]) and (C < O) and (C <= prev["Open"]) and (O >= prev["Close"]):
            patterns.append("BEAR_ENGULF")

    # Hammer / Hanging man / Shooting star / Inverted hammer
    if body > 0:
        if lower_shadow >= 2 * body and upper_shadow <= 0.5 * body:
            if C > O:
                patterns.append("HAMMER")
            else:
                patterns.append("HANGING_MAN")
        if upper_shadow >= 2 * body and lower_shadow <= 0.5 * body:
            if C < O:
                patterns.append("SHOOTING_STAR")
            else:
                patterns.append("INVERTED_HAMMER")

    # Inside bar
    if prev is not None:
        if (H < prev["High"]) and (L > prev["Low"]):
            patterns.append("INSIDE_BAR")

    # 3-bar patterns (Morning/Evening Star) simplified
    if prev is not None and prev2 is not None:
        O1, C1 = prev2["Open"], prev2["Close"]
        O2, C2 = prev["Open"], prev["Close"]
        O3, C3 = row["Open"], row["Close"]
        body1 = abs(C1 - O1)
        body2 = abs(C2 - O2)
        body3 = abs(C3 - O3)
        if (C1 < O1) and (body2 <= 0.6 * body1) and (C3 > O3) and (C3 > (O1 + C1) / 2):
            patterns.append("MORNING_STAR")
        if (C1 > O1) and (body2 <= 0.6 * body1) and (C3 < O3) and (C3 < (O1 + C1) / 2):
            patterns.append("EVENING_STAR")

    # Three White Soldiers / Three Black Crows
    if i >= 2:
        c0, c1, c2 = df["Close"].iloc[i - 2:i + 1]
        o0, o1, o2 = df["Open"].iloc[i - 2:i + 1]
        if (c0 < c1 < c2) and (o1 > c0) and (o2 > c1):
            patterns.append("THREE_WHITE")
        if (c0 > c1 > c2) and (o1 < c0) and (o2 < c1):
            patterns.append("THREE_BLACK")

    return patterns

# ------------------- Signal Generation (confluence voting) -------------------
def generate_confluence_signals(df_local, params, side="Both", signal_mode="Both", advanced_leading=True, scan_patterns=True):
    """
    signal_mode: 'Lagging', 'Price Action', 'Both'
    scan_patterns: enable heavy chart-pattern scanning (flags, triangles, H&S, cup-handle, etc.)
    advanced_leading: enable RVOL/divergence/imbalance etc.
    """
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    # leading names for filtering
    leading_prefixes = (
        "BULL_ENGULF","BEAR_ENGULF","HAMMER","HANGING_MAN","SHOOTING_STAR","INVERTED_HAMMER",
        "INSIDE_BAR","VOL_SPIKE","BREAK_HI","BREAK_LO","DOJI","MORNING_STAR","EVENING_STAR",
        "THREE_WHITE","THREE_BLACK","DIV_BULL","DIV_BEAR","RVOL","IMBAL_BULL","IMBAL_BEAR",
        "FLAG","TRIANGLE","CUP_HANDLE","HS","WEDGE","CHANNEL","SWEEP_HIGH","SWEEP_LOW",
        "FAKE_UP","FAKE_DOWN","ORDER_BLOCK","FVG","BOS","CHOCH","SD_ZONE"
    )

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        i = df_calc.index.get_loc(idx)

        # ---------------- lagging indicators (unchanged) ----------------
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

        # MACD hist
        if not np.isnan(row.get("macd_hist", np.nan)):
            if row["macd_hist"] > 0:
                indicators_that_long.append("MACD+")
            elif row["macd_hist"] < 0:
                indicators_that_short.append("MACD-")

        # RSI extremes
        rsi_val = row.get(f"rsi_{params['rsi_period']}", np.nan)
        if not np.isnan(rsi_val):
            if rsi_val < params.get("rsi_oversold", 35):
                indicators_that_long.append(f"RSI<{params.get('rsi_oversold', 35)}")
            elif rsi_val > params.get("rsi_overbought", 65):
                indicators_that_short.append(f"RSI>{params.get('rsi_overbought', 65)}")

        # Bollinger
        price = row["Close"]
        if not np.isnan(row.get("bb_upper", np.nan)) and not np.isnan(row.get("bb_lower", np.nan)):
            if price < row["bb_lower"]:
                indicators_that_long.append("BB_Lower")
            elif price > row["bb_upper"]:
                indicators_that_short.append("BB_Upper")

        # Momentum
        mom = row.get(f"mom_{params['mom_period']}", np.nan)
        if not np.isnan(mom):
            if mom > 0:
                indicators_that_long.append(f"MOM+({params['mom_period']})")
            elif mom < 0:
                indicators_that_short.append(f"MOM-({params['mom_period']})")

        # Stochastic
        if not np.isnan(row.get("stoch_k", np.nan)) and not np.isnan(row.get("stoch_d", np.nan)):
            if row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < params.get("stoch_oversold", 30):
                indicators_that_long.append("STOCH")
            elif row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > params.get("stoch_overbought", 70):
                indicators_that_short.append("STOCH")

        # ADX direction
        if not np.isnan(row.get("adx", np.nan)):
            if row["adx"] > params.get("adx_threshold", 20) and row["pdi"] > row["mdi"]:
                indicators_that_long.append("ADX+")
            elif row["adx"] > params.get("adx_threshold", 20) and row["mdi"] > row["pdi"]:
                indicators_that_short.append("ADX-")

        # OBV
        if i >= 3:
            recent = df_calc["obv"].iloc[max(0, i - 2):i + 1].mean()
            prev = df_calc["obv"].iloc[max(0, i - 5):max(0, i - 2)].mean()
            if recent > prev:
                indicators_that_long.append("OBV")
            elif recent < prev:
                indicators_that_short.append("OBV")

        # VWMA
        if not np.isnan(row.get("vwma", np.nan)):
            if row["Close"] > row["vwma"]:
                indicators_that_long.append("VWMA")
            elif row["Close"] < row["vwma"]:
                indicators_that_short.append("VWMA")

        # CCI extremes
        if not np.isnan(row.get("cci", np.nan)):
            if row["cci"] < -100:
                indicators_that_long.append("CCI")
            elif row["cci"] > 100:
                indicators_that_short.append("CCI")

        # Volume spike
        if not np.isnan(row.get("vol_sma", np.nan)) and row["Volume"] > 0:
            if row["Volume"] > row["vol_sma"] * params.get("vol_multiplier", 1.5):
                prev_close = df_calc["Close"].shift(1).iloc[i] if i > 0 else np.nan
                if not np.isnan(prev_close) and row["Close"] > prev_close:
                    indicators_that_long.append("VOL_SPIKE")
                elif not np.isnan(prev_close) and row["Close"] < prev_close:
                    indicators_that_short.append("VOL_SPIKE")

        # ---------------- leading candlestick patterns (existing) ----------------
        patterns = detect_candlestick_patterns(df_calc, i)
        for p in patterns:
            if p in ("BULL_ENGULF", "HAMMER", "INVERTED_HAMMER", "MORNING_STAR", "THREE_WHITE"):
                indicators_that_long.append(p)
            if p in ("BEAR_ENGULF", "HANGING_MAN", "SHOOTING_STAR", "EVENING_STAR", "THREE_BLACK"):
                indicators_that_short.append(p)

        # breakouts high/low
        lookback = params.get("breakout_lookback", 10)
        if i >= 1:
            prev_highs = df_calc["High"].iloc[max(0, i - lookback):i]
            prev_lows = df_calc["Low"].iloc[max(0, i - lookback):i]
            if len(prev_highs) > 0:
                if row["Close"] > prev_highs.max():
                    indicators_that_long.append(f"BREAK_HI_{lookback}")
                if row["Close"] < prev_lows.min():
                    indicators_that_short.append(f"BREAK_LO_{lookback}")

        # ---------------- advanced leading signals ----------------
        if advanced_leading:
            # RVOL
            if not np.isnan(row.get("vol_sma", np.nan)) and row["vol_sma"] > 0:
                rv = row["Volume"] / (row["vol_sma"] + 1e-9)
                if rv >= params.get("rvol_threshold", 2.0):
                    if row["Close"] > row["Open"]:
                        indicators_that_long.append("RVOL")
                    elif row["Close"] < row["Open"]:
                        indicators_that_short.append("RVOL")

            # imbalance / order-flow style
            rng = row["High"] - row["Low"] if (row["High"] - row["Low"]) != 0 else 1e-9
            close_near_high = (row["High"] - row["Close"]) / rng < 0.2
            close_near_low = (row["Close"] - row["Low"]) / rng < 0.2
            if row["Volume"] > row.get("vol_sma", 0) * 1.2:
                if close_near_high:
                    indicators_that_long.append("IMBAL_BULL")
                if close_near_low:
                    indicators_that_short.append("IMBAL_BEAR")

            # divergence
            rsi_col = f"rsi_{params['rsi_period']}"
            if rsi_col in df_calc.columns:
                divs = detect_divergences(df_calc, i, rsi_col)
                for d in divs:
                    if d == "DIV_BULL":
                        indicators_that_long.append(d)
                    if d == "DIV_BEAR":
                        indicators_that_short.append(d)

        # ---------------- heavy chart-pattern scanning (optional) ----------------
        if scan_patterns:
            # flag detection
            if detect_flag(df_calc, i, lookback=params.get("flag_lookback", 20)):
                indicators_that_long.append("FLAG")
                indicators_that_short.append("FLAG")

            # triangle
            if detect_triangle(df_calc, i, lookback=params.get("triangle_lookback", 20)):
                indicators_that_long.append("TRIANGLE")
                indicators_that_short.append("TRIANGLE")

            # head & shoulders
            hs_detected, hs_meta = detect_head_shoulders(df_calc, i, lookback=params.get("hs_lookback", 30))
            if hs_detected:
                indicators_that_short.append("HS")  # treat as reversal bearish by default

            # wedge / channel
            if detect_wedge(df_calc, i, lookback=params.get("wedge_lookback", 20)):
                indicators_that_long.append("WEDGE")
                indicators_that_short.append("WEDGE")
            if detect_channel(df_calc, i, lookback=params.get("channel_lookback", 30)):
                indicators_that_long.append("CHANNEL")
                indicators_that_short.append("CHANNEL")

            # liquidity sweep
            sweep, typ = detect_liquidity_sweep(df_calc, i, lookback=params.get("sweep_lookback", 5))
            if sweep and typ == "SWEEP_HIGH":
                indicators_that_short.append("SWEEP_HIGH")
            if sweep and typ == "SWEEP_LOW":
                indicators_that_long.append("SWEEP_LOW")

            # fake breakout
            fake, typ_f = detect_fake_breakout(df_calc, i, lookback=params.get("fake_lookback", 5))
            if fake:
                indicators_that_long.append(typ_f) if typ_f == "FAKE_DOWN" else indicators_that_short.append(typ_f)

            # order block
            ob, typ_ob = detect_order_block(df_calc, i, lookback=params.get("orderblock_lookback", 8))
            if ob:
                indicators_that_long.append(typ_ob) if "BUY" in typ_ob else indicators_that_short.append(typ_ob)

            # fair value gap
            fvg, typ_fvg = detect_fvg(df_calc, i)
            if fvg:
                indicators_that_long.append(typ_fvg) if "BULL" in typ_fvg else indicators_that_short.append(typ_fvg)

            # BOS / CHoCH
            bos = detect_bos_choch(df_calc, i, lookback=params.get("bos_lookback", 20))
            if bos == "BOS_UP":
                indicators_that_long.append("BOS_UP")
            elif bos == "BOS_DOWN":
                indicators_that_short.append("BOS_DOWN")

            # supply/demand quick
            sd = detect_supply_demand_zone(df_calc, i, lookback=params.get("sd_lookback", 30))
            if sd:
                indicators_that_long.append(sd) if "DEMAND" in sd else indicators_that_short.append(sd)

        # ---- filter lists based on signal_mode ----
        def is_leading(ind):
            return any(ind.startswith(p) for p in leading_prefixes)

        if signal_mode == "Lagging":
            indicators_that_long = [x for x in indicators_that_long if not is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if not is_leading(x)]
        elif signal_mode == "Price Action":
            indicators_that_long = [x for x in indicators_that_long if is_leading(x)]
            indicators_that_short = [x for x in indicators_that_short if is_leading(x)]
        # else Both -> keep all

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        final_sig = vote_direction if total_votes >= params["min_confluence"] else 0

        votes.append({
            "index": idx,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "total_votes": total_votes,
            "direction": vote_direction,
            "signal": final_sig,
            "indicators_long": indicators_that_long,
            "indicators_short": indicators_that_short,
            "patterns": patterns
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index("index")
    result = df_calc.join(votes_df)
    result["Signal"] = sig_series
    return result

# ------------------- Backtester (entries on same-bar close; exits on subsequent bars) -------------------
def choose_primary_indicator(indicators_list):
    priority = [
        "BULL_ENGULF","BEAR_ENGULF","MORNING_STAR","EVENING_STAR","HAMMER","SHOOTING_STAR",
        "RVOL","IMBAL_BULL","IMBAL_BEAR","DIV_BULL","DIV_BEAR","VOL_SPIKE","BREAK_HI","BREAK_LO",
        "FLAG","TRIANGLE","CUP_HANDLE","HS","WEDGE","CHANNEL","SWEEP_HIGH","SWEEP_LOW",
        "FAKE_UP","FAKE_DOWN","ORDER_BLOCK","FVG","BOS","CHOCH","SD_ZONE",
        "EMA","SMA","MACD","RSI","BB","VWMA","OBV","MOM","STOCH","ADX","CCI"
    ]
    for p in priority:
        for it in indicators_list:
            if p in it:
                return it
    return indicators_list[0] if indicators_list else ""

def backtest_point_strategy(df_signals, params):
    """
    Entries executed at same-bar Close (no future lookahead).
    Exits evaluated on subsequent bars' OHLC (no same-bar exit).
    """
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    first_price = df_signals["Close"].iloc[0]
    last_price = df_signals["Close"].iloc[-1]
    buy_hold_points = last_price - first_price

    n = len(df_signals)
    for i in range(n):
        row = df_signals.iloc[i]
        sig = row["Signal"]

        # exit checks using current row (which is in the future relative to the entry)
        if in_pos:
            h = row["High"]; l = row["Low"]; closep = row["Close"]
            exit_price = None; reason = None
            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target; reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl; reason = "Stopped"
                elif sig == -1 and row["total_votes"] >= params["min_confluence"]:
                    exit_price = closep; reason = "Opposite signal"
            else:
                if not pd.isna(l) and l <= target:
                    exit_price = target; reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl; reason = "Stopped"
                elif sig == 1 and row["total_votes"] >= params["min_confluence"]:
                    exit_price = closep; reason = "Opposite signal"

            if i == (n - 1) and in_pos and exit_price is None:
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
                    "Hold Days": (pd.to_datetime(exit_date).date() - pd.to_datetime(entry_details["Entry Date"]).date()).days
                })
                # reset
                in_pos = False; pos_side = 0; entry_price = None; entry_details = None; target = None; sl = None

        # entry executed at same bar close when signal appears
        if (not in_pos) and sig != 0:
            entry_date = row.name
            entry_price = row["Close"] if not pd.isna(row["Close"]) else row["Open"]
            pos_side = sig
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params["target_atr_mult"] * atr_val
                sl = entry_price - params["sl_atr_mult"] * atr_val
            else:
                target = entry_price - params["target_atr_mult"] * atr_val
                sl = entry_price + params["sl_atr_mult"] * atr_val

            indicators = (row["indicators_long"] if pos_side == 1 else row["indicators_short"])
            primary = choose_primary_indicator(indicators)

            entry_details = {
                "Entry Date": entry_date, "Entry Price": entry_price,
                "Side": "Long" if pos_side == 1 else "Short",
                "Indicators": indicators,
                "Primary Indicator": primary,
                "Confluences": row["total_votes"]
            }
            in_pos = True
            continue

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df["Points"] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades > 0 else 0.0
    percent_vs_buyhold = (total_points / (abs(buy_hold_points) + 1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        "total_points": total_points,
        "num_trades": num_trades,
        "wins": wins,
        "prob_of_profit": prob_of_profit,
        "buy_hold_points": buy_hold_points,
        "pct_vs_buyhold": percent_vs_buyhold
    }
    return summary, trades_df

# ------------------- Parameter optimization -------------------
def sample_random_params(base):
    p = base.copy()
    p["sma_fast"] = random.choice([5, 8, 10, 12, 15, 20])
    p["sma_slow"] = random.choice([50, 100, 150, 200])
    if p["sma_fast"] >= p["sma_slow"]:
        p["sma_fast"] = max(5, p["sma_slow"] // 10)
    p["ema_fast"] = random.choice([5, 9, 12, 15])
    p["ema_slow"] = random.choice([21, 26, 34, 50])
    if p["ema_fast"] >= p["ema_slow"]:
        p["ema_fast"] = max(5, p["ema_slow"] // 3)
    p["rsi_period"] = random.choice([7, 9, 14, 21])
    p["mom_period"] = random.choice([5, 10, 20])
    p["atr_period"] = random.choice([7, 14, 21])
    p["target_atr_mult"] = round(random.uniform(0.6, 3.0), 2)
    p["sl_atr_mult"] = round(random.uniform(0.6, 3.0), 2)
    p["min_confluence"] = random.randint(1, 6)
    p["vol_multiplier"] = round(random.uniform(1.0, 3.0), 2)
    p["breakout_lookback"] = random.choice([5, 10, 20])
    p["rvol_threshold"] = round(random.uniform(1.5, 3.0), 2)
    p["flag_lookback"] = random.choice([10, 20, 30])
    p["triangle_lookback"] = random.choice([10, 20, 30])
    p["hs_lookback"] = random.choice([20, 30, 40])
    p["wedge_lookback"] = random.choice([10, 20, 30])
    p["channel_lookback"] = random.choice([20, 30, 60])
    p["sweep_lookback"] = random.choice([3, 5, 8])
    p["fake_lookback"] = random.choice([3, 5, 8])
    p["orderblock_lookback"] = random.choice([5, 8, 12])
    p["fvg_lookback"] = random.choice([3, 5])
    p["bos_lookback"] = random.choice([10, 20])
    p["sd_lookback"] = random.choice([20, 30])
    return p

def optimize_parameters(df, base_params, n_iter, target_acc, target_points, side, signal_mode="Both", advanced_leading=True, scan_patterns=True, progress_bar=None, status_text=None):
    best = None; best_score = None
    target_frac = target_acc
    for i in range(n_iter):
        p = sample_random_params(base_params)
        try:
            df_sig = generate_confluence_signals(df, p, side, signal_mode, advanced_leading=advanced_leading, scan_patterns=scan_patterns)
            summary, trades = backtest_point_strategy(df_sig, p)
        except Exception:
            if progress_bar:
                progress_bar.progress(int((i + 1) / n_iter * 100))
            if status_text:
                status_text.text(f"Iteration {i+1}/{n_iter} (error)")
            continue
        prob = summary["prob_of_profit"]
        score = abs(prob - target_frac) - 0.0001 * summary["total_points"]
        if best is None or score < best_score:
            best = (p, summary, trades)
            best_score = score
        if progress_bar:
            progress_bar.progress(int((i + 1) / n_iter * 100))
        if status_text:
            status_text.text(f"Iteration {i+1}/{n_iter}")
        if prob >= target_frac and summary["total_points"] >= target_points:
            return p, summary, trades, True
    return best[0], best[1], best[2], False

# ------------------- Streamlit App UI -------------------
st.title("Backtester with Deep Chart-Pattern Scanning + Confluence")
st.markdown(
    "Upload OHLCV CSV/XLSX (Date,Open,High,Low,Close,Volume). "
    "You can choose signal source: Lagging / Price Action / Both. "
    "Enable 'Advanced Leading' for RVOL/divergence/imbalances and 'Scan Patterns' to detect many chart patterns."
)

uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
side = st.selectbox("Trade Side", options=["Both", "Long", "Short"], index=0)
signal_mode = st.selectbox("Signal source/mode", options=["Lagging", "Price Action", "Both"], index=2)
advanced_leading = st.checkbox("Use advanced leading signals (divergence/RVOL/imbalance)", value=True)
scan_patterns = st.checkbox("Scan many chart patterns (flags/triangles/H&S/..)", value=True)
random_iters = st.number_input("Random iterations (1-2000)", min_value=1, max_value=2000, value=300, step=1)
expected_returns = st.number_input("Expected strategy returns (total points)", value=0.0, step=1.0, format="%.2f")
expected_accuracy_pct = st.number_input("Expected accuracy % (probability of profit, e.g. 70)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
run_btn = st.button("Run Backtest & Optimize")

if uploaded_file is not None:
    try:
        if str(uploaded_file).lower().endswith(".xlsx") or (hasattr(uploaded_file, "name") and "xls" in uploaded_file.name.lower()):
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
        st.write("Last normalized close:", df_full["Close"].iloc[-1])
        st.write("Last normalized index:", df_full.index[-1])
    except Exception:
        pass

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    st.markdown(f"**Data range:** {min_date} to {max_date}")
    unique_dates = sorted({d.date() for d in df_full.index})
    selected_last_date = st.selectbox("Select last date (restrict data up to this date)", options=unique_dates, index=len(unique_dates) - 1, format_func=lambda x: x.strftime("%Y-%m-%d"))

    # slice upto selected date inclusive
    df = df_full[df_full.index.date <= selected_last_date]

    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Running optimization on restricted dataset..."):
            base_params = {
                "sma_fast": 10, "sma_slow": 50,
                "ema_fast": 9, "ema_slow": 21,
                "macd_signal": 9, "rsi_period": 14, "mom_period": 10,
                "stoch_period": 14, "cci_period": 20, "adx_period": 14,
                "atr_period": 14, "target_atr_mult": 1.5, "sl_atr_mult": 1.0,
                "min_confluence": 3, "vol_multiplier": 1.5,
                "vwma_period": 14, "vol_sma_period": 20,
                "breakout_lookback": 10, "rvol_threshold": 2.0
            }

            target_acc = expected_accuracy_pct / 100.0
            target_points = expected_returns

            best_params, best_summary, best_trades, perfect = optimize_parameters(
                df, base_params, int(random_iters), target_acc, target_points, side,
                signal_mode=signal_mode, advanced_leading=advanced_leading, scan_patterns=scan_patterns,
                progress_bar=progress_bar, status_text=status_text
            )

            progress_bar.progress(100)
            status_text.text("Optimization completed")

            st.subheader("Optimization Result")
            st.write("Signal source:", signal_mode)
            st.write("Advanced leading enabled:", advanced_leading)
            st.write("Scan patterns enabled:", scan_patterns)
            st.write("Target accuracy:", expected_accuracy_pct, "% ; Target points:", target_points)
            st.write("Perfect match found:", perfect)
            st.json(best_params)

            st.subheader("Summary (best candidate)")
            st.write(best_summary)

            if best_trades is None or best_trades.empty:
                st.info("No trades found with best parameters.")
            else:
                best_trades_display = best_trades.copy()
                st.subheader("Top 5 trades (by Points)")
                st.dataframe(best_trades_display.nlargest(5, "Points"))
                st.subheader("Bottom 5 trades (by Points)")
                st.dataframe(best_trades_display.nsmallest(5, "Points"))

                best_trades_display["Exit Date"] = pd.to_datetime(best_trades_display["Exit Date"])
                best_trades_display["Year"] = best_trades_display["Exit Date"].dt.year
                best_trades_display["Month"] = best_trades_display["Exit Date"].dt.month
                monthly_points = best_trades_display.groupby(["Year", "Month"])["Points"].sum().reset_index()

                month_start = df["Close"].resample("MS").first().reset_index()
                month_start["Year"] = month_start["Date"].dt.year
                month_start["Month"] = month_start["Date"].dt.month
                month_start = month_start.rename(columns={"Close": "Month_Start_Close"})

                monthly = monthly_points.merge(month_start[["Year", "Month", "Month_Start_Close"]], on=["Year", "Month"], how="left")
                avg_close = df["Close"].mean()
                monthly["Month_Start_Close"] = monthly["Month_Start_Close"].fillna(avg_close)
                monthly["Pct_Return"] = (monthly["Points"] / monthly["Month_Start_Close"]) * 100.0

                pivot_pct = monthly.pivot(index="Year", columns="Month", values="Pct_Return").fillna(0)
                for m in range(1, 13):
                    if m not in pivot_pct.columns:
                        pivot_pct[m] = 0
                pivot_pct = pivot_pct.reindex(sorted(pivot_pct.columns), axis=1)

                st.subheader("Monthly % returns heatmap (Year vs Month)")
                fig, ax = plt.subplots(figsize=(10, max(2, 0.6 * len(pivot_pct.index) + 1)))
                sns.heatmap(pivot_pct, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
                ax.set_ylabel("Year")
                ax.set_xlabel("Month")
                st.pyplot(fig)

                st.subheader("All trades (best candidate)")
                st.dataframe(best_trades_display)

            # live recommendation based on best params
            latest_sig_df = generate_confluence_signals(df, best_params, side, signal_mode, advanced_leading=advanced_leading, scan_patterns=scan_patterns)
            latest_row = latest_sig_df.iloc[-1]
            st.subheader("Signal DataFrame - last 5 rows (debug)")
            st.dataframe(latest_sig_df.tail(5))
            sig_val = int(latest_row["Signal"])
            sig_text = "Buy" if sig_val == 1 else ("Sell" if sig_val == -1 else "No Signal")
            atr_val = latest_row.get(f"atr_{best_params['atr_period']}", np.nan)
            entry_price_est = float(latest_row["Close"])
            actual_last_close = float(df["Close"].iloc[-1])
            actual_last_idx = df.index[-1]
            if sig_val == 1:
                target_price = entry_price_est + best_params["target_atr_mult"] * atr_val
                sl_price = entry_price_est - best_params["sl_atr_mult"] * atr_val
                indicators_list = latest_row["indicators_long"]
            elif sig_val == -1:
                target_price = entry_price_est - best_params["target_atr_mult"] * atr_val
                sl_price = entry_price_est + best_params["sl_atr_mult"] * atr_val
                indicators_list = latest_row["indicators_short"]
            else:
                target_price = np.nan; sl_price = np.nan; indicators_list = []

            primary = choose_primary_indicator(indicators_list)
            confluences = int(latest_row.get("total_votes", 0))
            prob_of_profit = (best_summary.get("prob_of_profit", np.nan) * 100.0) if isinstance(best_summary, dict) else np.nan

            def explain_indicator(it):
                # short explanations (keeps long list concise)
                if it.startswith("EMA"): return f"{it}: EMA crossover momentum"
                if it.startswith("SMA"): return f"{it}: SMA crossover trend"
                if "MACD" in it: return "MACD: momentum histogram"
                if it.startswith("RSI"): return "RSI: overbought/oversold"
                if it == "VOL_SPIKE": return "High volume + direction"
                if it == "RVOL": return "Relative volume spike"
                if it in ("IMBAL_BULL", "IMBAL_BEAR"): return "Order-flow imbalance (close near high/low + vol)"
                if it.startswith("BULL_ENGULF"): return "Bullish engulfing pattern"
                if it.startswith("BEAR_ENGULF"): return "Bearish engulfing pattern"
                if it == "MORNING_STAR": return "Morning star (3-bar) bullish"
                if it == "EVENING_STAR": return "Evening star (3-bar) bearish"
                if it in ("FLAG", "TRIANGLE", "HS", "WEDGE", "CHANNEL"): return f"{it}: chart pattern detected"
                if "FVG" in it: return "Fair Value Gap (unfilled area)"
                if "ORDER_BLOCK" in it: return "Order block (institutional-looking candle + consolidation)"
                if "BOS" in it: return "Break of Structure"
                if "SWEEP" in it: return "Liquidity sweep (wick run & return)"
                return it

            reasons = [explain_indicator(ii) for ii in indicators_list]
            reason_text = (f"Primary: {primary}. ") + ("; ".join(reasons) if reasons else "No strong indicator explanation.")
            indicator_values = {
                "sma_fast": latest_row.get(f"sma_{best_params['sma_fast']}", np.nan),
                "sma_slow": latest_row.get(f"sma_{best_params['sma_slow']}", np.nan),
                "ema_fast": latest_row.get(f"ema_{best_params['ema_fast']}", np.nan),
                "ema_slow": latest_row.get(f"ema_{best_params['ema_slow']}", np.nan),
                "macd_hist": latest_row.get("macd_hist", np.nan),
                f"rsi_{best_params['rsi_period']}": latest_row.get(f"rsi_{best_params['rsi_period']}", np.nan),
                "bb_upper": latest_row.get("bb_upper", np.nan),
                "bb_lower": latest_row.get("bb_lower", np.nan),
                "obv": latest_row.get("obv", np.nan),
                "vwma": latest_row.get("vwma", np.nan),
                "cci": latest_row.get("cci", np.nan),
                "vol": latest_row.get("Volume", np.nan),
                "vol_sma": latest_row.get("vol_sma", np.nan),
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
            ind_df.columns = ["Indicator", "Value"]
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
                st.error("Possible causes: (1) 'Close' column had non-numeric formatting and got coerced; (2) duplicate/misaligned timestamps; (3) the dataset slice used by optimizer/backtest is different from uploaded file.")

            st.success("Done")

else:
    st.info("Upload a CSV/XLSX to start. After upload you can choose the 'Select last date' and then click 'Run Backtest & Optimize'.")

# ------------------- End of script -------------------
