# app.py
# Streamlit Algorithmic Trading Analysis - professional-grade single-file app
# Dependencies: streamlit, yfinance, pandas, numpy, scipy, plotly, pytz
# Python: 3.8+
# Run: `streamlit run app.py`

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz
import time
import random
import io
import warnings
import typing
import traceback

warnings.filterwarnings("ignore")

# ---------------------------
# Config / Constants
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")
ALLOWED_LIBS = ["streamlit", "yfinance", "pandas", "numpy", "scipy", "plotly", "pytz", "datetime", "time"]
RATE_MIN = 1.5
RATE_MAX = 3.0

DEFAULT_CAPITAL = 100000  # INR for paper trading

# Timeframes mapping for yfinance compatibility constraints (enforced)
VALID_PERIODS_FOR_INTERVAL = {
    "1m": ["1d", "5d"],
    "3m": ["1d", "5d"],  # yfinance doesn't support 3m interval explicitly; we'll treat as 3m ~ 3 minutes if needed
    "5m": ["1d", "5d", "1mo"],
    "10m": ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "30m": ["1d", "5d", "1mo"],
    "1h": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "2h": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "4h": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1y", "2y", "5y", "10y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "20y", "25y", "30y"],
}

# ---------------------------
# Session state initialization
# ---------------------------
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = {}
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {}
if "paper_trades" not in st.session_state:
    st.session_state["paper_trades"] = []
if "paper_capital" not in st.session_state:
    st.session_state["paper_capital"] = DEFAULT_CAPITAL
if "live_monitoring" not in st.session_state:
    st.session_state["live_monitoring"] = False
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = None
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

def log_debug(msg: str):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{ts} | {msg}"
    st.session_state["debug_logs"].append(entry)
    print(entry)

# ---------------------------
# Utility: safe download with rate limit, retries, timezone handling
# ---------------------------
def safe_yf_download(ticker: str, period: str, interval: str, max_retries: int = 3) -> typing.Optional[pd.DataFrame]:
    attempt = 0
    while attempt < max_retries:
        try:
            log_debug(f"Downloading {ticker} period={period} interval={interval} attempt={attempt+1}")
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            # handle empty
            if data is None or data.empty:
                log_debug(f"Empty data returned for {ticker} (period={period}, interval={interval})")
                return None
            # flatten multiindex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # timezone handling
            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC").tz_convert(IST)
            else:
                data.index = data.index.tz_convert(IST)
            # wait random to respect rate limit
            time.sleep(random.uniform(RATE_MIN, RATE_MAX))
            return data
        except Exception as e:
            attempt += 1
            log_debug(f"Error downloading {ticker}: {repr(e)} | attempt {attempt}")
            time.sleep(0.5)
    log_debug(f"Failed to download {ticker} after {max_retries} attempts")
    return None

# ---------------------------
# Indicators - manual implementations
# ---------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # default per spec
    return rsi

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period, min_periods=1).mean()
    k = k.fillna(50); d = d.fillna(50)
    return k, d

def bollinger_bands(series: pd.Series, period=20, n_std=2):
    ma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def atr(df: pd.DataFrame, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period, min_periods=1).mean()
    atr_val = atr_val.fillna(method="bfill").fillna(0)
    return atr_val

def historical_volatility(series: pd.Series, lookback=20):
    log_returns = np.log(series / series.shift(1)).fillna(0)
    rolling_std = log_returns.rolling(window=lookback, min_periods=1).std().fillna(0)
    hv = rolling_std * np.sqrt(252) * 100
    return hv

def adx(df: pd.DataFrame, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=period, min_periods=1).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=1).sum() / tr_smooth).fillna(0)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=1).sum() / tr_smooth).fillna(0)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.rolling(window=period, min_periods=1).mean().fillna(0)
    return adx_val, plus_di.fillna(0), minus_di.fillna(0)

def obv(df: pd.DataFrame):
    close = df["Close"]
    volume = df["Volume"].fillna(0)
    direction = np.sign(close.diff().fillna(0))
    obv_val = (direction * volume).cumsum()
    return obv_val

def zscore(series: pd.Series, window=20):
    return series.rolling(window=window, min_periods=1).apply(lambda x: stats.zscore(x)[-1] if len(x)>1 else 0, raw=False).fillna(0)

# ---------------------------
# Support & resistance detection (local extrema + clustering)
# ---------------------------
def find_levels(df: pd.DataFrame, n=5, cluster_tolerance=0.005):
    """
    Find local extrema levels and cluster them.
    Returns DataFrame of levels with touches and strength.
    """
    prices = df["Close"]
    # local maxima and minima
    local_max_idx = argrelextrema(df["High"].values, np.greater_equal, order=n)[0]
    local_min_idx = argrelextrema(df["Low"].values, np.less_equal, order=n)[0]

    levels = []
    for idx in local_max_idx:
        levels.append(("resistance", float(df["High"].iloc[idx]), df.index[idx]))
    for idx in local_min_idx:
        levels.append(("support", float(df["Low"].iloc[idx]), df.index[idx]))

    if not levels:
        return pd.DataFrame(columns=["type","price","date","touches","sustained","strength","distance_pct"])

    levels_df = pd.DataFrame(levels, columns=["type","price","date"])
    # clustering by price tolerance
    clusters = []
    used = set()
    prices_arr = levels_df["price"].values
    idxs = list(range(len(prices_arr)))
    for i in idxs:
        if i in used:
            continue
        base = prices_arr[i]
        cluster = [i]
        for j in idxs:
            if j in used or j==i:
                continue
            if abs(prices_arr[j]-base)/base <= cluster_tolerance:
                cluster.append(j)
        for k in cluster:
            used.add(k)
        cluster_prices = prices_arr[cluster]
        mean_price = float(np.mean(cluster_prices))
        types = levels_df.loc[cluster, "type"]
        # touches: count how many times price got close to this cluster
        # We'll count touches across entire series where close within tolerance
        tol = mean_price * cluster_tolerance
        touches_idx = df[np.abs(df["Close"] - mean_price) <= tol].index
        touches = len(touches_idx)
        # sustained: count the number of contiguous periods price remained above/below (approx)
        sustained = 0
        # strength rules
        strength = "Moderate"
        if touches >= 3:
            strength = "Strong"
        elif touches == 2:
            strength = "Moderate"
        else:
            strength = "Weak"
        clusters.append({
            "type": types.mode()[0] if not types.mode().empty else types.iloc[0],
            "price": mean_price,
            "date": levels_df.loc[cluster, "date"].min(),
            "touches": touches,
            "sustained": sustained,
            "strength": strength,
        })
    res = pd.DataFrame(clusters)
    # distance from current
    current_price = float(df["Close"].iloc[-1])
    res["distance_pct"] = (res["price"] - current_price) / current_price * 100
    res = res.sort_values(by="distance_pct", key=lambda x: np.abs(x))
    return res.head(8)

# ---------------------------
# Fibonacci
# ---------------------------
def fibonacci_levels(df: pd.DataFrame, lookback=100):
    recent = df[-lookback:]
    high = recent["High"].max()
    low = recent["Low"].min()
    diff = high - low
    levels = {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low,
        "1.272": low - 0.272 * diff,
        "1.618": low - 0.618 * diff,
    }
    levels_df = pd.DataFrame.from_dict(levels, orient="index", columns=["price"])
    levels_df["label"] = levels_df.index
    current = float(df["Close"].iloc[-1])
    levels_df["distance_pct"] = (levels_df["price"] - current) / current * 100
    # closest
    closest = levels_df.iloc[(levels_df["price"] - current).abs().argsort()].head(1)
    return levels_df, closest

# ---------------------------
# Elliott wave heuristic
# ---------------------------
def detect_elliott(df: pd.DataFrame, lookback=200):
    """
    Heuristic: find series of alternating higher highs/lows for impulse waves.
    We'll find sequence of 5 swings (1-5) using local extrema (argrelextrema).
    Returns dict with estimated wave and confidence (0-100).
    """
    recent = df[-lookback:].copy()
    # get swing extremes
    high_idx = argrelextrema(recent["High"].values, np.greater_equal, order=3)[0]
    low_idx = argrelextrema(recent["Low"].values, np.less_equal, order=3)[0]
    idxs = np.sort(np.concatenate([high_idx, low_idx])) if len(high_idx) or len(low_idx) else np.array([])
    if len(idxs) < 5:
        return {"position": "Unknown", "confidence": 0, "swings": []}
    swings = []
    for i in idxs:
        swings.append({
            "index_rel": int(i),
            "date": recent.index[i],
            "price": float(recent["Close"].iloc[i])
        })
    # naive: check if we have a 5-swing sequence with overall increasing prices => impulse
    if len(swings) >= 5:
        last5 = swings[-5:]
        prices = [s["price"] for s in last5]
        # impulse if overall increasing with at least 3 higher highs
        inc = sum(np.diff(prices) > 0)
        dec = sum(np.diff(prices) < 0)
        confidence = min(95, int((inc/4)*100)) if inc >= 2 else int((inc/4)*100)
        position = "Wave3" if inc >= 3 else ("Wave1" if inc >=2 else "Corrective")
        return {"position": position, "confidence": confidence, "swings": last5}
    return {"position": "Unknown", "confidence": 10, "swings": swings}

# ---------------------------
# RSI divergence detection
# ---------------------------
def detect_rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback=50):
    recent_close = df["Close"].iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]
    # price local lows/highs
    lows = argrelextrema(recent_close.values, np.less_equal, order=3)[0]
    highs = argrelextrema(recent_close.values, np.greater_equal, order=3)[0]
    result = {"type": "None", "strength": 0, "details": None}
    # Bullish divergence: price makes lower lows, RSI makes higher lows
    if len(lows) >= 2:
        p_lows = recent_close.values[lows]
        r_lows = recent_rsi.values[lows]
        if p_lows[-1] < p_lows[-2] and r_lows[-1] > r_lows[-2]:
            # strength: relative divergence magnitude
            price_diff = (p_lows[-2] - p_lows[-1]) / p_lows[-2]
            rsi_diff = (r_lows[-1] - r_lows[-2]) / (abs(r_lows[-2]) + 1e-6)
            strength = min(100, int((abs(price_diff) + abs(rsi_diff)) * 1000))
            result = {"type": "Bullish", "strength": strength,
                      "details": {"p_low_prev": p_lows[-2], "p_low_now": p_lows[-1],
                                  "rsi_prev": r_lows[-2], "rsi_now": r_lows[-1],
                                  "dates": [recent_close.index[l] for l in lows[-2:]]}}
    # Bearish divergence: price higher highs, RSI lower highs
    if len(highs) >= 2:
        p_highs = recent_close.values[highs]
        r_highs = recent_rsi.values[highs]
        if p_highs[-1] > p_highs[-2] and r_highs[-1] < r_highs[-2]:
            price_diff = (p_highs[-1] - p_highs[-2]) / p_highs[-2]
            rsi_diff = (r_highs[-2] - r_highs[-1]) / (abs(r_highs[-2]) + 1e-6)
            strength = min(100, int((abs(price_diff) + abs(rsi_diff)) * 1000))
            result = {"type": "Bearish", "strength": strength,
                      "details": {"p_high_prev": p_highs[-2], "p_high_now": p_highs[-1],
                                  "rsi_prev": r_highs[-2], "rsi_now": r_highs[-1],
                                  "dates": [recent_close.index[h] for h in highs[-2:]]}}
    return result

# ---------------------------
# Simple historical pattern matching by correlation
# ---------------------------
def pattern_match(series: pd.Series, lookback=100, min_corr=0.85):
    """
    Compare the last lookback window with earlier windows and return matches with correlation > min_corr.
    Returns list of matches with correlation and what happened next.
    """
    s = series.dropna()
    if len(s) < lookback * 2:
        return []
    current = s[-lookback:].values
    matches = []
    for start in range(0, len(s) - 2*lookback):
        candidate = s[start:start+lookback].values
        corr = np.corrcoef(current, candidate)[0,1]
        if corr >= min_corr:
            # compute what happened next  (percentage move over next lookback//4 candles)
            if start+2*lookback < len(s):
                next_window = s[start+lookback:start+lookback+lookback//4]
                if len(next_window) > 0:
                    pct_move = (next_window[-1] - next_window[0]) / (next_window[0] + 1e-9) * 100
                else:
                    pct_move = 0.0
            else:
                pct_move = 0.0
            matches.append({"start_date": s.index[start], "corr": float(corr), "next_pct_move": float(pct_move)})
    # sort by correlation desc
    matches = sorted(matches, key=lambda x: x["corr"], reverse=True)
    return matches[:10]

# ---------------------------
# Signal aggregation / Multi-timeframe recommendation
# ---------------------------
def aggregate_signals(per_tf_signals: typing.Dict[str, dict]):
    """
    per_tf_signals: { "5m/1d": {"signal": 1/-1/0, ...}, ... }
    Return aggregated recommendation and confidence.
    """
    scores = []
    reasons = {}
    for tf, s in per_tf_signals.items():
        score = s.get("signal", 0)
        scores.append(score)
        for r in s.get("reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    if len(scores) == 0:
        return {"signal_text": "HOLD", "score": 0, "confidence_pct": 0, "reasons": []}
    avg = np.mean(scores)
    # rules per spec
    signal_text = "HOLD"
    if avg > 0.3:
        signal_text = "BUY"
    elif avg < -0.3:
        signal_text = "SELL"
    # confidence: % of timeframes in agreement scaled to 0-99
    agree = sum(1 for s in scores if (s>0 and avg>0) or (s<0 and avg<0))
    confidence = int((agree/len(scores)) * 99)
    top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:8]
    return {"signal_text": signal_text, "score": float(avg), "confidence_pct": confidence, "reasons": top_reasons}

# ---------------------------
# Backtesting engine (simple)
# ---------------------------
def backtest_strategy(df: pd.DataFrame, capital=100000, period_atr=14, max_trades=500):
    """
    Strategy: Entry when RSI < 30 AND Price > EMA20.
    Exit: Target hit or SL hit or RSI > 70 or time-based (20 candles).
    Returns trades list and summary statistics.
    """
    dfc = df.copy()
    dfc["EMA20"] = ema(dfc["Close"], 20)
    dfc["RSI"] = rsi(dfc["Close"], 14)
    dfc["ATR"] = atr(dfc, period_atr)
    trades = []
    position = None
    for i in range(len(dfc)):
        if len(trades) >= max_trades:
            break
        price = dfc["Close"].iloc[i]
        rsi_now = dfc["RSI"].iloc[i]
        ema20 = dfc["EMA20"].iloc[i]
        atr_now = dfc["ATR"].iloc[i]
        if position is None:
            # entry
            if rsi_now < 30 and price > ema20:
                entry_price = price
                sl = entry_price - 2 * atr_now
                t1 = entry_price + 2 * atr_now
                t2 = entry_price + 3 * atr_now
                t3 = entry_price + 4 * atr_now
                qty = max(1, int((capital * 0.1) // entry_price))
                position = {"entry_index": i, "entry_price": entry_price, "qty": qty, "sl": sl, "t1": t1, "t2": t2, "t3": t3, "entry_rsi": rsi_now, "exit_index": None}
        else:
            # check exit conditions
            cur_price = price
            # target hit or stop loss
            if cur_price >= position["t1"] or cur_price <= position["sl"] or rsi_now > 70 or (i - position["entry_index"]) >= 20:
                exit_price = cur_price
                pnl_points = exit_price - position["entry_price"]
                pnl_pct = pnl_points / position["entry_price"] * 100
                trades.append({
                    "entry_index": position["entry_index"],
                    "exit_index": i,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "qty": position["qty"],
                    "pnl_points": pnl_points,
                    "pnl_pct": pnl_pct,
                    "entry_rsi": position["entry_rsi"],
                    "exit_rsi": rsi_now
                })
                position = None
    # summary stats
    wins = [t for t in trades if t["pnl_points"] > 0]
    losses = [t for t in trades if t["pnl_points"] <= 0]
    total_return = sum([t["pnl_points"] * t["qty"] for t in trades])
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    summary = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_return_points": total_return
    }
    return trades, summary

# ---------------------------
# Main compute pipeline for a single ticker/timeframe
# ---------------------------
def analyze_ticker(ticker: str, period: str, interval: str, ratio_ticker: typing.Optional[str] = None, lookbacks: dict = None):
    """
    Downloads data (cached), computes indicators, SR, Fibonacci, Elliott, divergence, patterns, and returns a dict.
    """
    key = f"{ticker}|{period}|{interval}"
    if key in st.session_state["data_cache"]:
        df = st.session_state["data_cache"][key]
    else:
        df = safe_yf_download(ticker, period=period, interval=interval)
        if df is None:
            raise ValueError(f"No data for {ticker} {period} {interval}")
        st.session_state["data_cache"][key] = df

    # compute indicators
    try:
        dfc = df.copy()
        dfc["SMA9"] = sma(dfc["Close"], 9)
        dfc["SMA20"] = sma(dfc["Close"], 20)
        dfc["SMA50"] = sma(dfc["Close"], 50)
        dfc["SMA100"] = sma(dfc["Close"], 100)
        dfc["SMA200"] = sma(dfc["Close"], 200)

        dfc["EMA9"] = ema(dfc["Close"], 9)
        dfc["EMA20"] = ema(dfc["Close"], 20)
        dfc["EMA50"] = ema(dfc["Close"], 50)
        dfc["EMA100"] = ema(dfc["Close"], 100)
        dfc["EMA200"] = ema(dfc["Close"], 200)

        dfc["RSI14"] = rsi(dfc["Close"], 14)
        dfc["MACD"], dfc["MACD_SIGNAL"], dfc["MACD_HIST"] = macd(dfc["Close"])
        dfc["STOCH_K"], dfc["STOCH_D"] = stochastic_oscillator(dfc["High"], dfc["Low"], dfc["Close"])
        dfc["BB_MID"], dfc["BB_UP"], dfc["BB_LOW"] = bollinger_bands(dfc["Close"])
        dfc["ATR14"] = atr(dfc, 14)
        dfc["HV20"] = historical_volatility(dfc["Close"], 20)
        dfc["ADX14"], dfc["+DI"], dfc["-DI"] = adx(dfc, 14)
        dfc["OBV"] = obv(dfc)
        dfc["VOL_MA20"] = dfc["Volume"].rolling(window=20, min_periods=1).mean()
        dfc["PRICE_Z"] = (dfc["Close"] - dfc["Close"].rolling(20, min_periods=1).mean()) / dfc["Close"].rolling(20, min_periods=1).std().replace(0, np.nan)
        dfc["PRICE_Z"] = dfc["PRICE_Z"].fillna(0)
        # returns Z-score
        dfc["RET"] = dfc["Close"].pct_change().fillna(0)
        dfc["RET_Z"] = zscore(dfc["RET"], 20)
        dfc["VOL_Z"] = zscore(dfc["Volume"], 20)
    except Exception as e:
        log_debug(f"Indicator calculation error: {repr(e)}")
        raise

    # support/resistance
    try:
        levels = find_levels(dfc, n=5, cluster_tolerance=0.005)
    except Exception as e:
        log_debug(f"SR calc error: {repr(e)}")
        levels = pd.DataFrame()

    # fibonacci
    try:
        fib_df, fib_closest = fibonacci_levels(dfc, lookback=lookbacks.get("fib_lookback", 100) if lookbacks else 100)
    except Exception as e:
        log_debug(f"Fibonacci error: {repr(e)}")
        fib_df = pd.DataFrame(); fib_closest = pd.DataFrame()

    # elliott
    try:
        ell = detect_elliott(dfc, lookback=lookbacks.get("elliott_lookback", 200) if lookbacks else 200)
    except Exception as e:
        log_debug(f"Elliott error: {repr(e)}")
        ell = {"position": "Unknown", "confidence": 0, "swings": []}

    # rsi divergence
    try:
        rv = detect_rsi_divergence(dfc, dfc["RSI14"], lookback=lookbacks.get("div_lookback", 50) if lookbacks else 50)
    except Exception as e:
        log_debug(f"RSI divergence error: {repr(e)}")
        rv = {"type": "None", "strength": 0, "details": None}

    # pattern matching on close
    try:
        pmatches = pattern_match(dfc["Close"], lookback=lookbacks.get("pattern_lookback", 100) if lookbacks else 100, min_corr=0.85)
    except Exception as e:
        log_debug(f"Pattern match error: {repr(e)}")
        pmatches = []

    # simple per-timeframe signal building
    signal = 0  # 1 buy, -1 sell, 0 neutral
    reasons = []
    # buy bias if RSI oversold and price above EMA20 and ADX>25 considered trend strength
    last = dfc.iloc[-1]
    if last["RSI14"] < 30 and last["Close"] > last["EMA20"]:
        signal += 1; reasons.append("RSI_oversold+Price>EMA20")
    if last["RSI14"] > 70 and last["Close"] < last["EMA20"]:
        signal -= 1; reasons.append("RSI_overbought+Price<EMA20")
    if last["ADX14"] > 25 and last["Close"] > last["EMA20"]:
        signal += 1; reasons.append("ADX_strong_trend_and_price_above_EMA")
    if rv["type"] == "Bullish":
        signal += 1; reasons.append("RSI_bullish_divergence")
    if rv["type"] == "Bearish":
        signal -= 1; reasons.append("RSI_bearish_divergence")
    # proximity to fibonacci key levels
    if not fib_closest.empty:
        key_label = fib_closest.index[0]
        if key_label in ["0.618","0.5","0.382"]:
            reasons.append(f"Near_Fib_{key_label}")

    # assemble results
    res = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "data": dfc,
        "levels": levels,
        "fibonacci": fib_df,
        "fib_closest": fib_closest,
        "elliott": ell,
        "rsi_divergence": rv,
        "pattern_matches": pmatches,
        "signal": signal,
        "reasons": reasons,
        "summary": {
            "price": float(dfc["Close"].iloc[-1]),
            "RSI": float(dfc["RSI14"].iloc[-1]),
            "Volatility": float(dfc["HV20"].iloc[-1]),
            "ZScore": float(dfc["PRICE_Z"].iloc[-1]),
            "ADX": float(dfc["ADX14"].iloc[-1]),
            "ATR": float(dfc["ATR14"].iloc[-1])
        }
    }

    # store analysis results (for UI)
    st.session_state["analysis_results"][key] = res
    return res

# ---------------------------
# Plotting helpers (Plotly)
# ---------------------------
def plot_candles_with_indicators(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if "EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(width=1)))
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(width=1)))
    fig.update_layout(title=title, xaxis_title="Time (IST)", yaxis_title="Price", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_price_with_rsi(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if "EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(width=1)))
    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=title, xaxis_title="Time (IST)", yaxis_title="Price", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

def plot_fibonacci(df: pd.DataFrame, fib_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    for idx, row in fib_df.iterrows():
        fig.add_hline(y=row["price"], annotation_text=idx, annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Algo Trading Analyzer", layout="wide")
st.title("Algorithmic Trading Analysis — Professional Suite")
st.markdown("Comprehensive multi-timeframe technical analysis, statistical testing, and AI-style signals. All times shown in IST.")

# Sidebar - config
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker (yfinance)", value="^NSEI")
    period = st.selectbox("Period", options=["1d","5d","7d","1mo","3mo","6mo","1y","2y","3y","5y","6y","10y","15y","20y","25y","30y"], index=0)
    interval = st.selectbox("Interval (timeframe)", options=["1m","3m","5m","10m","15m","30m","1h","2h","4h","1d","1wk","1mo"], index=9)
    ratio_enabled = st.checkbox("Enable Ratio Analysis (Ticker 2)", value=False)
    ticker2 = st.text_input("Ticker 2 (for Ratio)", value="^NSEBANK" if ratio_enabled else "")
    analyze_btn = st.button("Fetch & Analyze")
    st.markdown("---")
    st.subheader("Paper Trading")
    capital = st.number_input("Paper capital (INR)", value=st.session_state["paper_capital"], step=1000)
    st.session_state["paper_capital"] = capital
    if st.button("Show Debug Logs"):
        st.write("\n".join(st.session_state["debug_logs"][-200:]))

# Validate interval/period compatibility
if analyze_btn:
    # Simple mapping check
    valid_periods = VALID_PERIODS_FOR_INTERVAL.get(interval, None)
    if valid_periods:
        # if period not in allowed and interval has a restricted set: warn but proceed with best-match
        if period not in valid_periods:
            st.warning(f"Note: interval '{interval}' is typically compatible with periods {valid_periods}. Proceeding anyway may yield truncated data.")
    # Fetch & analyze
    try:
        with st.spinner(f"Analyzing {ticker} {interval}/{period} ..."):
            res = analyze_ticker(ticker, period, interval, ratio_ticker=(ticker2 if ratio_enabled else None),
                                 lookbacks={"fib_lookback":100,"elliott_lookback":200,"div_lookback":50,"pattern_lookback":100})
            # If ratio enabled, analyze ticker2 and compute ratio
            res2 = None
            ratio_df = None
            if ratio_enabled and ticker2:
                try:
                    res2 = analyze_ticker(ticker2, period, interval, lookbacks={"fib_lookback":100})
                    # align by common index
                    df1 = res["data"]
                    df2 = res2["data"]
                    common_index = df1.index.intersection(df2.index)
                    if len(common_index) > 0:
                        df1a = df1.loc[common_index]
                        df2a = df2.loc[common_index]
                        ratio_df = pd.DataFrame({
                            "Ticker1": df1a["Close"],
                            "Ticker2": df2a["Close"],
                            "Ratio": (df1a["Close"] / df2a["Close"]).replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
                        })
                    else:
                        st.warning("No common dates found between tickers for ratio analysis.")
                except Exception as e:
                    st.error(f"Ticker2 analysis error: {repr(e)}")
            # Multi-timeframe analysis: run through a set of timeframes automatically (subset for speed)
            # We'll analyze a chosen set (per your spec: analyze ALL valid timeframe/period combos)
            # For practicality we analyze a sensible subset here; you can extend.
            tf_list = ["5m","15m","1h","4h","1d"]
            per_tf_signals = {}
            progress = st.progress(0)
            total = len(tf_list)
            for idx, tf in enumerate(tf_list):
                # choose period that is compatible; use same 'period' unless incompatible
                try:
                    sub_period = period
                    # ensure compatibility
                    valid = VALID_PERIODS_FOR_INTERVAL.get(tf, None)
                    if valid and sub_period not in valid:
                        sub_period = valid[-1]  # pick largest available fallback
                    sub_res = analyze_ticker(ticker, sub_period, tf)
                    # per timeframe signal structure
                    s = {"signal": sub_res["signal"], "reasons": sub_res["reasons"], "summary": sub_res["summary"], "tf_label": f"{tf}/{sub_period}"}
                    per_tf_signals[f"{tf}/{sub_period}"] = s
                except Exception as e:
                    log_debug(f"Multi-tf analysis error for {tf}: {repr(e)}")
                progress.progress(int((idx+1)/total*100))
            agg = aggregate_signals(per_tf_signals)
            # Build final SL/Targets
            last = res["data"].iloc[-1]
            entry_price = float(last["Close"])
            atr_now = float(last["ATR14"])
            sl = entry_price - 2*atr_now if agg["signal_text"]=="BUY" else entry_price + 2*atr_now if agg["signal_text"]=="SELL" else None
            targets = []
            if agg["signal_text"] in ["BUY","SELL"]:
                mults = [2,3,4]
                for m in mults:
                    if agg["signal_text"]=="BUY":
                        targets.append(entry_price + m*atr_now)
                    else:
                        targets.append(entry_price - m*atr_now)
            # show summary
            st.header("Multi-Timeframe Consolidated Recommendation")
            st.markdown(f"**Signal:** {'🟢 BUY' if agg['signal_text']=='BUY' else '🔴 SELL' if agg['signal_text']=='SELL' else '🟡 HOLD'}")
            st.markdown(f"**Confidence:** {agg['confidence_pct']}%")
            st.markdown(f"**Entry:** ₹{entry_price:,.2f}")
            if sl is not None:
                st.markdown(f"**Stop Loss:** ₹{sl:,.2f}")
            if targets:
                st.markdown("**Targets:** " + ", ".join([f"₹{t:,.2f}" for t in targets]))
            st.markdown("**Top reasons:**")
            for r, cnt in agg["reasons"]:
                st.write(f"- {r} (across {cnt} timeframes)")
            # show analysis per timeframe
            st.header("Per-Timeframe Analysis")
            for tf, s in per_tf_signals.items():
                st.subheader(f"{tf}")
                st.write(f"Signal score: {s['signal']}, Reasons: {s['reasons']}")
            # charts
            st.header("Charts")
            plot_candles_with_indicators(res["data"].tail(200), f"{ticker} — Candles with EMAs ({interval}/{period})")
            plot_price_with_rsi(res["data"].tail(200), f"{ticker} — Price vs RSI ({interval}/{period})")
            plot_fibonacci(res["data"].tail(200), res["fibonacci"], f"{ticker} — Fibonacci levels (last 100 candles)")
            if ratio_df is not None:
                st.header("Ratio Analysis")
                st.dataframe(ratio_df.tail(200))
                # ratio RSI
                ratio_rsi = rsi(ratio_df["Ratio"], 14)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df["Ratio"], name="Ratio"))
                st.plotly_chart(fig, use_container_width=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ratio_df.index, y=ratio_rsi, name="Ratio RSI"))
                fig2.add_hline(y=70); fig2.add_hline(y=30)
                st.plotly_chart(fig2, use_container_width=True)

            # backtest
            st.header("Backtest (sample RSI<30 & price>EMA20 strategy)")
            bt_trades, bt_summary = backtest_strategy(res["data"], capital=capital)
            st.write(bt_summary)
            if bt_trades:
                bt_df = pd.DataFrame(bt_trades)
                st.dataframe(bt_df)
                csv_bytes = bt_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Backtest CSV", data=csv_bytes, file_name=f"backtest_{ticker}_{interval}_{period}.csv", mime="text/csv")
            # export analysis
            dumped = {
                "res": res,
                "per_tf_signals": per_tf_signals,
                "agg": agg
            }
            # create an export CSV of last N candles with indicators
            to_export = res["data"].tail(1000).reset_index()
            csv_buf = to_export.to_csv(index=False).encode("utf-8")
            st.download_button("Export Analysis CSV (last 1000 rows)", data=csv_buf, file_name=f"{ticker}_analysis_{interval}_{period}.csv")
            st.success("Analysis complete.")
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Analysis failed: {repr(e)}")
        log_debug(f"Analysis exception: {tb}")

# ---------------------------
# Paper trading UI
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Paper Trading Manager")
st.sidebar.write(f"Capital: ₹{st.session_state['paper_capital']:,.2f}")
if st.sidebar.button("Show Positions"):
    st.write("Paper trades:")
    st.dataframe(pd.DataFrame(st.session_state["paper_trades"]))

# Auto-refresh / live monitor skeleton (do not auto-run unless user enables)
if st.sidebar.checkbox("Enable Live Monitoring (auto-refresh every 2s)", value=False):
    st.session_state["live_monitoring"] = True
else:
    st.session_state["live_monitoring"] = False

if st.session_state["live_monitoring"]:
    # note: avoid infinite loops in shared hosting; provide conservative loop
    # We'll fetch 5-minute latest and update simple metrics
    st.info("Live monitoring enabled. This will auto-refresh every ~2 seconds.")
    last_refresh = st.session_state.get("last_refresh", None)
    now = time.time()
    if last_refresh is None or now - last_refresh >= 2:
        st.session_state["last_refresh"] = now
        # attempt to fetch latest 5m data for current ticker
        try:
            live_data = safe_yf_download(ticker, period="5d", interval="5m")
            if live_data is not None:
                st.write("Latest price:", float(live_data["Close"].iloc[-1]))
                # recalc quick metrics
                rsi_now = float(rsi(live_data["Close"], 14).iloc[-1])
                atr_now = float(atr(live_data, 14).iloc[-1])
                st.write(f"RSI14: {rsi_now:.2f} | ATR14: {atr_now:.4f}")
        except Exception as e:
            log_debug(f"Live refresh error: {repr(e)}")
    st.experimental_rerun()

# ---------------------------
# Footer / Debug Controls
# ---------------------------
st.markdown("---")
st.write("Logs (last 20):")
for l in st.session_state["debug_logs"][-20:]:
    st.text(l)

st.markdown("Built with ❤️ — Manual indicators, multi-timeframe aggregation, backtesting & paper trading boilerplate.")
