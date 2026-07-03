"""
AlgoTrader Pro — Single-file Streamlit algorithmic trading workbench.

Educational / research tool. Not investment advice. Past backtest performance
never guarantees future returns. Live "trading" in this app is a paper/
simulation layer unless you explicitly enable the Dhan broker checkbox and
wire in verified credentials — do that only after testing in a sandbox.
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AlgoTrader Pro", layout="wide", page_icon="📈")

# ============================================================================
# CONSTANTS
# ============================================================================

TICKER_MAP = {
    "Nifty50": "^NSEI",
    "BankNifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USDINR": "USDINR=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Custom": None,
}

TF_PERIOD_MAP = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "7d", "1mo", "3mo", "6mo", "1y"],
    "1d": ["7d", "1mo", "6mo", "1y", "2y", "3y", "5y", "10y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "30y"],
}

STRATEGIES = [
    "EMA Crossover",
    "Simple Buy/Sell",
    "Threshold Cross",
    "Price Action Support/Resistance",
    "Liquidity Grab Reversal",
    "RSI Cross",
    "Bollinger Bands",
    "Volume Breakout",
    "Elliott Wave (Zigzag)",
    "Pro: VWAP + Supertrend Trend",
    "Pro: Opening Range Breakout + Volume",
    "Pro: BB+RSI Mean Reversion (ATR filtered)",
    "Pro: EMA50 Trend + EMA9/15 Pullback",
]

PRO_STRATEGIES = {
    "Pro: VWAP + Supertrend Trend",
    "Pro: Opening Range Breakout + Volume",
    "Pro: BB+RSI Mean Reversion (ATR filtered)",
    "Pro: EMA50 Trend + EMA9/15 Pullback",
}

# Rough family classification used by the Regime Filter — trend-following
# strategies want ADX confirming a trend, mean-reversion strategies want the
# opposite (a non-trending / ranging tape). "neutral" strategies aren't gated.
STRATEGY_FAMILY = {
    "EMA Crossover": "trend",
    "Simple Buy/Sell": "neutral",
    "Threshold Cross": "neutral",
    "Price Action Support/Resistance": "trend",
    "Liquidity Grab Reversal": "mean_reversion",
    "RSI Cross": "mean_reversion",
    "Bollinger Bands": "mean_reversion",
    "Volume Breakout": "trend",
    "Elliott Wave (Zigzag)": "trend",
    "Pro: VWAP + Supertrend Trend": "trend",
    "Pro: Opening Range Breakout + Volume": "trend",
    "Pro: BB+RSI Mean Reversion (ATR filtered)": "mean_reversion",
    "Pro: EMA50 Trend + EMA9/15 Pullback": "trend",
}

SL_TYPES = [
    "Custom Points", "Trailing SL (Points)", "Trail Candle Low/High (Current)",
    "Trail Candle Low/High (Previous)", "Trail Swing Low/High (Current)",
    "Trail Swing Low/High (Previous)", "Strategy Signal Exit", "EMA Reverse Crossover Exit",
    "ATR Based SL", "Risk:Reward Based (min 1:2)", "Autopilot SL",
]

TARGET_TYPES = [
    "Custom Points", "Trailing Target (Display Only)", "Trail Candle Low/High (Current)",
    "Trail Candle Low/High (Previous)", "Trail Swing Low/High (Current)",
    "Trail Swing Low/High (Previous)", "Strategy Signal Exit", "EMA Reverse Crossover Exit",
    "ATR Based Target", "Risk:Reward Based (min 1:2)", "Autopilot Target",
]

RATE_LIMIT_DELAY = 0.3  # seconds, mandatory pause between yfinance calls

# ============================================================================
# SESSION STATE
# ============================================================================

for key, default in {
    "live_positions": [],
    "live_history": [],
    "sidebar_overrides": {},
    "opt_results": {},
    "last_backtest": None,
    "last_backtest_df": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# INDICATORS
# ============================================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series, period=20, std_mult=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)


def supertrend(df, period=10, mult=3):
    atr_ = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = (hl2 + mult * atr_).copy()
    lowerband = (hl2 - mult * atr_).copy()
    line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            line.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue
        if df["Close"].iloc[i] > upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]
        line.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]
    return line, direction


def swing_points(df, lookback=3):
    highs, lows = df["High"], df["Low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    n = len(df)
    for i in range(lookback, n - lookback):
        wh = highs.iloc[i - lookback: i + lookback + 1]
        wl = lows.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == wh.max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == wl.min():
            swing_low.iloc[i] = True
    return swing_high, swing_low


# ============================================================================
# DATA FETCH (rate-limited)
# ============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data(ticker, interval, period):
    time.sleep(RATE_LIMIT_DELAY)
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df


@st.fragment(run_every=2)
def live_ltp_fragment(ticker, label="LTP"):
    time.sleep(RATE_LIMIT_DELAY)
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data is None or data.empty:
            data = yf.Ticker(ticker).history(period="5d", interval="15m")
        if data is not None and not data.empty:
            ltp = float(data["Close"].iloc[-1])
            prev = float(data["Close"].iloc[-2]) if len(data) > 1 else ltp
            delta = ltp - prev
            st.metric(label, f"{ltp:,.2f}", f"{delta:+.2f}")
            return ltp
        st.info("No live data returned yet.")
    except Exception as exc:
        st.warning(f"Fetch issue (rate limit or symbol): {exc}")
    return None


# ============================================================================
# STRATEGY SIGNAL GENERATION  (no look-ahead: signal at i uses data up to i)
# ============================================================================

def generate_signals(df, strategy, params):
    df = df.copy()
    df["signal"] = 0

    if strategy == "EMA Crossover":
        f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
        ef, es = ema(df["Close"], f), ema(df["Close"], s)
        df.loc[(ef > es) & (ef.shift(1) <= es.shift(1)), "signal"] = 1
        df.loc[(ef < es) & (ef.shift(1) >= es.shift(1)), "signal"] = -1

    elif strategy == "Simple Buy/Sell":
        df.loc[df["Close"] > df["Close"].shift(1), "signal"] = 1
        df.loc[df["Close"] < df["Close"].shift(1), "signal"] = -1

    elif strategy == "Threshold Cross":
        thr = params.get("threshold", float(df["Close"].iloc[0]))
        df.loc[(df["Close"] > thr) & (df["Close"].shift(1) <= thr), "signal"] = 1
        df.loc[(df["Close"] < thr) & (df["Close"].shift(1) >= thr), "signal"] = -1

    elif strategy == "Price Action Support/Resistance":
        w = params.get("sr_window", 20)
        res = df["High"].rolling(w).max().shift(1)
        sup = df["Low"].rolling(w).min().shift(1)
        df.loc[df["Close"] > res, "signal"] = 1
        df.loc[df["Close"] < sup, "signal"] = -1

    elif strategy == "Liquidity Grab Reversal":
        w = params.get("liq_window", 20)
        vol_avg = df["Volume"].rolling(w).mean()
        rec_high = df["High"].rolling(w).max().shift(1)
        rec_low = df["Low"].rolling(w).min().shift(1)
        sweep_high = (df["High"] > rec_high) & (df["Close"] < rec_high) & (df["Volume"] > 1.5 * vol_avg)
        sweep_low = (df["Low"] < rec_low) & (df["Close"] > rec_low) & (df["Volume"] > 1.5 * vol_avg)
        df.loc[sweep_low, "signal"] = 1
        df.loc[sweep_high, "signal"] = -1

    elif strategy == "RSI Cross":
        r = rsi(df["Close"], params.get("rsi_period", 14))
        df["rsi"] = r
        df.loc[(r > 30) & (r.shift(1) <= 30), "signal"] = 1
        df.loc[(r < 70) & (r.shift(1) >= 70), "signal"] = -1

    elif strategy == "Bollinger Bands":
        upper, mid, lower = bollinger(df["Close"], params.get("bb_period", 20), params.get("bb_std", 2))
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = upper, mid, lower
        df.loc[(df["Close"] < lower) & (df["Close"].shift(1) >= lower.shift(1)), "signal"] = 1
        df.loc[(df["Close"] > upper) & (df["Close"].shift(1) <= upper.shift(1)), "signal"] = -1

    elif strategy == "Volume Breakout":
        w = params.get("vol_window", 20)
        factor = params.get("vol_factor", 2)
        vol_avg = df["Volume"].rolling(w).mean()
        rh = df["High"].rolling(w).max().shift(1)
        rl = df["Low"].rolling(w).min().shift(1)
        df.loc[(df["Close"] > rh) & (df["Volume"] > factor * vol_avg), "signal"] = 1
        df.loc[(df["Close"] < rl) & (df["Volume"] > factor * vol_avg), "signal"] = -1

    elif strategy == "Elliott Wave (Zigzag)":
        sh, sl_ = swing_points(df, params.get("zigzag_lookback", 3))
        df["swing_high"], df["swing_low"] = sh, sl_
        df.loc[sl_, "signal"] = 1
        df.loc[sh, "signal"] = -1

    elif strategy == "Pro: VWAP + Supertrend Trend":
        vw = vwap(df)
        st_line, direction = supertrend(df, params.get("st_period", 10), params.get("st_mult", 3))
        df["vwap"], df["supertrend"] = vw, st_line
        buy = (df["Close"] > vw) & (direction == 1) & (direction.shift(1) != 1)
        sell = (df["Close"] < vw) & (direction == -1) & (direction.shift(1) != -1)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Opening Range Breakout + Volume":
        n_candles = params.get("orb_candles", 5)
        dates = pd.Series(df.index.date, index=df.index)
        or_high = df.groupby(dates)["High"].transform(lambda x: x.iloc[:n_candles].max())
        or_low = df.groupby(dates)["Low"].transform(lambda x: x.iloc[:n_candles].min())
        vol_avg = df["Volume"].rolling(20).mean()
        buy = (df["Close"] > or_high) & (df["Volume"] > 1.3 * vol_avg)
        sell = (df["Close"] < or_low) & (df["Volume"] > 1.3 * vol_avg)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: BB+RSI Mean Reversion (ATR filtered)":
        upper, mid, lower = bollinger(df["Close"], 20, 2)
        r = rsi(df["Close"], 14)
        a = atr(df, 14)
        atr_pct = a / df["Close"]
        low_vol = atr_pct < atr_pct.rolling(50).mean() * 1.5
        buy = (df["Close"] < lower) & (r < 35) & low_vol.fillna(False)
        sell = (df["Close"] > upper) & (r > 65) & low_vol.fillna(False)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: EMA50 Trend + EMA9/15 Pullback":
        e50 = ema(df["Close"], 50)
        e9, e15 = ema(df["Close"], 9), ema(df["Close"], 15)
        uptrend, downtrend = df["Close"] > e50, df["Close"] < e50
        buy = uptrend & (e9 > e15) & (e9.shift(1) <= e15.shift(1))
        sell = downtrend & (e9 < e15) & (e9.shift(1) >= e15.shift(1))
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    df["signal"] = df["signal"].fillna(0)
    return df


def apply_filters(df, filters):
    df = df.copy()
    mask_buy = df["signal"] == 1
    mask_sell = df["signal"] == -1

    if filters.get("adx_enabled"):
        a = adx(df, 14)
        df["adx_f"] = a
        ok = (a >= filters["adx_min"]) & (a <= filters["adx_max"])
        mask_buy &= ok.fillna(False)
        mask_sell &= ok.fillna(False)

    if filters.get("rsi_enabled"):
        r = rsi(df["Close"], 14)
        rsi_buy_ok = (r > 30) & (r.shift(1) <= 30)
        rsi_sell_ok = (r < 70) & (r.shift(1) >= 70)
        mask_buy &= rsi_buy_ok.fillna(False)
        mask_sell &= rsi_sell_ok.fillna(False)

    if filters.get("bb_enabled"):
        upper, mid, lower = bollinger(df["Close"], 20, 2)
        mask_buy &= (df["Close"] <= upper).fillna(False)
        mask_sell &= (df["Close"] >= lower).fillna(False)

    if filters.get("ema20_enabled"):
        e20 = ema(df["Close"], 20)
        mask_buy &= (df["Close"] > e20).fillna(False)
        mask_sell &= (df["Close"] < e20).fillna(False)

    if filters.get("sma20_enabled"):
        s20 = sma(df["Close"], 20)
        mask_buy &= (df["Close"] > s20).fillna(False)
        mask_sell &= (df["Close"] < s20).fillna(False)

    if filters.get("smc_enabled"):
        sh, sl_ = swing_points(df, 3)
        last_high = df["High"].where(sh).ffill()
        last_low = df["Low"].where(sl_).ffill()
        bos_up = (df["Close"] > last_high.shift(1)).fillna(False)
        bos_dn = (df["Close"] < last_low.shift(1)).fillna(False)
        mask_buy &= bos_up
        mask_sell &= bos_dn

    if filters.get("atr_enabled"):
        a = atr(df, 14)
        df["atr_f"] = a
        ok = (a >= filters["atr_min"]) & (a <= filters["atr_max"])
        mask_buy &= ok.fillna(False)
        mask_sell &= ok.fillna(False)

    if filters.get("supertrend_enabled"):
        st_line, st_dir = supertrend(df, filters.get("st_filter_period", 10), filters.get("st_filter_mult", 3.0))
        df["supertrend_f"], df["supertrend_dir_f"] = st_line, st_dir
        mask_buy &= (st_dir == 1)
        mask_sell &= (st_dir == -1)

    if filters.get("regime_enabled"):
        a = adx(df, 14)
        df["regime_adx"] = a
        trend_ok = a >= filters.get("regime_trend_min", 25)
        range_ok = a <= filters.get("regime_range_max", 20)
        family = STRATEGY_FAMILY.get(filters.get("current_strategy"), "neutral")
        if family == "trend":
            mask_buy &= trend_ok.fillna(False)
            mask_sell &= trend_ok.fillna(False)
        elif family == "mean_reversion":
            mask_buy &= range_ok.fillna(False)
            mask_sell &= range_ok.fillna(False)
        # "neutral" strategies are left ungated by regime

    new_signal = pd.Series(0, index=df.index)
    new_signal[mask_buy] = 1
    new_signal[mask_sell] = -1
    df["signal"] = new_signal
    return df


# ============================================================================
# SL / TARGET ENGINE
# ============================================================================

def calc_initial_sl_target(direction, entry_price, atr_val, params, sl_type, target_type):
    sl_points = params.get("sl_points", 10.0)
    target_points = params.get("target_points", 20.0)
    rr_ratio = max(params.get("rr_ratio", 2.0), 2.0)

    if sl_type == "ATR Based SL":
        sl_dist = atr_val * params.get("atr_mult_sl", 1.5)
    elif sl_type == "Autopilot SL":
        sl_dist = max(atr_val * 1.2, sl_points)
    else:
        sl_dist = sl_points

    if target_type == "ATR Based Target":
        target_dist = atr_val * params.get("atr_mult_target", 3.0)
    elif target_type == "Risk:Reward Based (min 1:2)":
        target_dist = sl_dist * rr_ratio
    elif target_type == "Autopilot Target":
        target_dist = max(atr_val * 2.5, sl_dist * 2)
    else:
        target_dist = target_points

    if sl_type == "Risk:Reward Based (min 1:2)":
        target_dist = max(target_dist, sl_dist * rr_ratio)

    if direction == 1:
        sl, target = entry_price - sl_dist, entry_price + target_dist
    else:
        sl, target = entry_price + sl_dist, entry_price - target_dist
    return sl, target, sl_dist, target_dist


def update_trade_levels(trade, i, df, params, atr_series):
    direction = trade["direction"]
    candle = df.iloc[i]
    prev_candle = df.iloc[i - 1] if i > 0 else candle
    sl_type, target_type = trade["sl_type"], trade["target_type"]

    trade["highest"] = max(trade.get("highest", candle["High"]), candle["High"])
    trade["lowest"] = min(trade.get("lowest", candle["Low"]), candle["Low"])

    a_val = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else trade["sl_dist"]

    if sl_type == "Trailing SL (Points)":
        d = trade["sl_dist"]
        trade["sl"] = max(trade["sl"], trade["highest"] - d) if direction == 1 else min(trade["sl"], trade["lowest"] + d)
    elif sl_type == "Trail Candle Low/High (Current)":
        trade["sl"] = max(trade["sl"], candle["Low"]) if direction == 1 else min(trade["sl"], candle["High"])
    elif sl_type == "Trail Candle Low/High (Previous)":
        trade["sl"] = max(trade["sl"], prev_candle["Low"]) if direction == 1 else min(trade["sl"], prev_candle["High"])
    elif sl_type in ("Trail Swing Low/High (Current)", "Trail Swing Low/High (Previous)"):
        span = df.iloc[max(0, i - 10):(i if "Previous" in sl_type else i + 1)]
        if direction == 1 and not span.empty:
            trade["sl"] = max(trade["sl"], span["Low"].min())
        elif direction == -1 and not span.empty:
            trade["sl"] = min(trade["sl"], span["High"].max())
    elif sl_type == "ATR Based SL":
        mult = params.get("atr_mult_sl", 1.5)
        trade["sl"] = max(trade["sl"], candle["Close"] - a_val * mult) if direction == 1 else min(trade["sl"], candle["Close"] + a_val * mult)
    elif sl_type == "Autopilot SL":
        profit = (candle["Close"] - trade["entry_price"]) * direction
        tighten = 0.7 if profit > trade["sl_dist"] else 1.4
        trade["sl"] = max(trade["sl"], candle["Close"] - a_val * tighten) if direction == 1 else min(trade["sl"], candle["Close"] + a_val * tighten)

    if target_type == "Trail Candle Low/High (Current)":
        trade["target"] = max(trade["target"], candle["High"]) if direction == 1 else min(trade["target"], candle["Low"])
    elif target_type == "Trail Candle Low/High (Previous)":
        trade["target"] = max(trade["target"], prev_candle["High"]) if direction == 1 else min(trade["target"], prev_candle["Low"])
    elif target_type in ("Trail Swing Low/High (Current)", "Trail Swing Low/High (Previous)"):
        span = df.iloc[max(0, i - 10):(i if "Previous" in target_type else i + 1)]
        if direction == 1 and not span.empty:
            trade["target"] = max(trade["target"], span["High"].max())
        elif direction == -1 and not span.empty:
            trade["target"] = min(trade["target"], span["Low"].min())
    elif target_type == "Autopilot Target":
        trade["target"] = max(trade["target"], candle["Close"] + a_val * 2.5) if direction == 1 else min(trade["target"], candle["Close"] - a_val * 2.5)

    return trade


def detect_signal_exit_condition(trade, i, df, params):
    """
    Detects a strategy-reverse or EMA-reverse exit condition using data known
    at the CLOSE of candle i. This is only ever used to SCHEDULE an exit for
    execution at the OPEN of candle i+1 (see run_backtest / live tab) — never
    executed on candle i itself. Executing it on candle i would mean using
    that candle's own close to justify a fill at that candle's open, which is
    look-ahead bias no live system could actually achieve.
    """
    direction = trade["direction"]
    sl_type, target_type = trade["sl_type"], trade["target_type"]

    if sl_type == "Strategy Signal Exit" or target_type == "Strategy Signal Exit":
        sig = df["signal"].iloc[i]
        if (direction == 1 and sig == -1) or (direction == -1 and sig == 1):
            return True, "Strategy Reverse Signal"

    if sl_type == "EMA Reverse Crossover Exit" or target_type == "EMA Reverse Crossover Exit":
        f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
        close_slice = df["Close"].iloc[: i + 1]
        if len(close_slice) > max(f, s) + 1:
            ef, es = ema(close_slice, f), ema(close_slice, s)
            if direction == 1 and ef.iloc[-1] < es.iloc[-1] and ef.iloc[-2] >= es.iloc[-2]:
                return True, "EMA Reverse Crossover"
            if direction == -1 and ef.iloc[-1] > es.iloc[-1] and ef.iloc[-2] <= es.iloc[-2]:
                return True, "EMA Reverse Crossover"

    return False, None


def check_hard_exit(trade, candle):
    """
    Hard SL/Target check using only the CURRENT candle's own high/low against
    levels set from PAST data (entry price, ATR at signal time, trailing
    updates). No look-ahead here — these levels never depend on this candle's
    own close. Conservative fill order: longs check SL(low) before
    Target(high); shorts check SL(high) before Target(low).
    """
    direction = trade["direction"]
    target_display_only = trade["target_type"] == "Trailing Target (Display Only)"

    if direction == 1:
        if candle["Low"] <= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["High"] >= trade["target"]:
            return True, trade["target"], "Target Hit"
    else:
        if candle["High"] >= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["Low"] <= trade["target"]:
            return True, trade["target"], "Target Hit"

    return False, None, None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(raw_df, strategy, sl_type, target_type, params, filters, qty):
    if raw_df.empty or len(raw_df) < 30:
        return pd.DataFrame(), raw_df

    filters = dict(filters or {})
    filters["current_strategy"] = strategy

    df = generate_signals(raw_df, strategy, params)
    df = apply_filters(df, filters)
    atr_series = atr(df, 14)

    trades = []
    open_trade = None

    def close_trade(exit_price, exit_time, reason):
        points = (exit_price - open_trade["entry_price"]) * open_trade["direction"]
        trades.append({
            "Entry Time": open_trade["entry_time"], "Entry Price": round(open_trade["entry_price"], 2),
            "Direction": "LONG" if open_trade["direction"] == 1 else "SHORT",
            "Exit Time": exit_time, "Exit Price": round(float(exit_price), 2),
            "SL": round(open_trade["initial_sl"], 2), "Target": round(open_trade["initial_target"], 2),
            "Points": round(points, 2), "PnL": round(points * qty, 2),
            "Exit Reason": reason, "Qty": qty,
        })

    for i in range(1, len(df) - 1):
        if open_trade is None:
            sig = df["signal"].iloc[i]
            if sig != 0:
                entry_idx = i + 1
                if entry_idx >= len(df):
                    break
                entry_price = float(df["Open"].iloc[entry_idx])
                a_val = atr_series.iloc[i]
                a_val = a_val if not np.isnan(a_val) else entry_price * 0.005
                sl, target, sl_dist, target_dist = calc_initial_sl_target(sig, entry_price, a_val, params, sl_type, target_type)
                open_trade = {
                    "entry_time": df.index[entry_idx], "entry_price": entry_price,
                    "direction": int(sig), "qty": qty, "sl": sl, "target": target,
                    "initial_sl": sl, "initial_target": target,
                    "sl_dist": sl_dist, "target_dist": target_dist,
                    "sl_type": sl_type, "target_type": target_type,
                    "highest": entry_price, "lowest": entry_price,
                    "signal_candle": df.index[i], "entry_idx": entry_idx,
                    "pending_exit_reason": None,
                }
        else:
            if i < open_trade["entry_idx"]:
                continue
            candle = df.iloc[i]
            open_trade = update_trade_levels(open_trade, i, df, params, atr_series)

            exited, exit_price, reason = False, None, None

            # 1) A signal/EMA-reverse exit detected on the PREVIOUS candle's
            #    close is executed here, at THIS candle's open — mirrors how
            #    entries are delayed one candle, so no look-ahead.
            if open_trade.get("pending_exit_reason"):
                exited, exit_price, reason = True, candle["Open"], open_trade["pending_exit_reason"]

            # 2) Hard SL/Target check on this candle (safe — never depends on
            #    this candle's own close).
            if not exited:
                exited, exit_price, reason = check_hard_exit(open_trade, candle)

            if exited:
                close_trade(exit_price, df.index[i], reason)
                open_trade = None
            else:
                # 3) Detect (but don't act on) a new signal/EMA-reverse exit
                #    using this candle's own close — scheduled for next candle.
                sig_exit, sig_reason = detect_signal_exit_condition(open_trade, i, df, params)
                if sig_exit:
                    open_trade["pending_exit_reason"] = sig_reason

    if open_trade is not None:
        # Ran out of data with a position still open — close it out at the
        # last available close so it's counted, flagged distinctly.
        last_i = len(df) - 1
        close_trade(df["Close"].iloc[last_i], df.index[last_i], "End of Data (Forced Close)")

    return pd.DataFrame(trades), df


def compute_metrics(trades_df):
    if trades_df is None or trades_df.empty:
        return dict(total_trades=0, wins=0, losses=0, accuracy=0.0, total_points=0.0, total_pnl=0.0,
                    avg_win=0.0, avg_loss=0.0, expectancy=0.0, sharpe=0.0, max_drawdown=0.0)
    total = len(trades_df)
    wins = int((trades_df["Points"] > 0).sum())
    losses = total - wins
    accuracy = wins / total * 100
    avg_win = trades_df.loc[trades_df["Points"] > 0, "Points"].mean() if wins else 0.0
    avg_loss = trades_df.loc[trades_df["Points"] <= 0, "Points"].mean() if losses else 0.0
    win_rate = wins / total
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    returns = trades_df["Points"]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0.0
    cum = returns.cumsum()
    drawdown = (cum - cum.cummax()).min()
    return dict(total_trades=total, wins=wins, losses=losses, accuracy=round(accuracy, 2),
                total_points=round(returns.sum(), 2), total_pnl=round(trades_df["PnL"].sum(), 2),
                avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2), expectancy=round(expectancy, 2),
                sharpe=round(sharpe, 2), max_drawdown=round(float(drawdown), 2))


def apply_cost_model(trades_df, cost_cfg, qty):
    """Adds cost-adjusted Points/PnL columns. Costs are subtracted per round-trip
    trade: (slippage + spread) in points, plus brokerage charged on both legs."""
    if trades_df is None or trades_df.empty:
        return trades_df
    trades_df = trades_df.copy()
    per_trade_point_cost = cost_cfg.get("slippage_points", 0.0) + cost_cfg.get("spread_points", 0.0)
    per_trade_cash_cost = cost_cfg.get("brokerage_flat", 0.0) * 2
    trades_df["Points (Net)"] = (trades_df["Points"] - per_trade_point_cost).round(2)
    trades_df["PnL (Net)"] = (trades_df["Points (Net)"] * qty - per_trade_cash_cost).round(2)
    return trades_df


def compute_metrics_from_columns(trades_df, points_col="Points", pnl_col="PnL"):
    """Same as compute_metrics but usable on either raw or cost-adjusted columns."""
    if trades_df is None or trades_df.empty:
        return compute_metrics(trades_df)
    tmp = pd.DataFrame({"Points": trades_df[points_col].values, "PnL": trades_df[pnl_col].values})
    return compute_metrics(tmp)


def walk_forward_folds(trades_df, start_time, end_time, n_folds):
    """Splits the backtest's time range into n sequential folds and computes
    metrics for whichever trades fall (by entry time) into each fold. This is
    an out-of-sample CONSISTENCY check (does the edge hold up across sub
    periods), not a parameter re-optimization walk-forward."""
    if trades_df is None or trades_df.empty:
        return []
    edges = pd.date_range(start=start_time, end=end_time, periods=n_folds + 1)
    fold_rows = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        mask = (trades_df["Entry Time"] >= lo) & (trades_df["Entry Time"] < hi if k < n_folds - 1 else trades_df["Entry Time"] <= hi)
        sub = trades_df[mask]
        m = compute_metrics(sub)
        m["Fold"] = k + 1
        m["From"] = lo
        m["To"] = hi
        fold_rows.append(m)
    return fold_rows


def smart_verdict(metrics, wf_fold_metrics=None, cost_enabled=False, metrics_net=None):
    """
    Rule-based composite score across accuracy, expectancy, Sharpe, drawdown,
    out-of-sample fold consistency (if walk-forward was run), and cost-adjusted
    expectancy (if cost modeling was run). This is a transparent heuristic
    scorecard, NOT a trained machine-learning model — there is no hidden
    "learning" happening beyond aggregating the metrics you can already see
    into one verdict, so you can sanity-check it against the numbers yourself.
    """
    notes = []
    if metrics["total_trades"] < 10:
        return "⚪ NOT ENOUGH DATA", ["Fewer than 10 trades in this sample — verdict would be unreliable. Use a longer period or lower timeframe."]

    score = 0
    if metrics["expectancy"] > 0:
        score += 1; notes.append(f"✅ Positive raw expectancy ({metrics['expectancy']} pts/trade)")
    else:
        score -= 1; notes.append(f"❌ Negative raw expectancy ({metrics['expectancy']} pts/trade)")

    if metrics["accuracy"] >= 50:
        score += 1; notes.append(f"✅ Win rate ≥ 50% ({metrics['accuracy']}%)")
    else:
        notes.append(f"⚠️ Win rate below 50% ({metrics['accuracy']}%) — relies on winners being bigger than losers")

    if metrics["sharpe"] > 0.5:
        score += 1; notes.append(f"✅ Sharpe > 0.5 ({metrics['sharpe']})")
    elif metrics["sharpe"] > 0:
        notes.append(f"⚠️ Sharpe is positive but weak ({metrics['sharpe']})")
    else:
        score -= 1; notes.append(f"❌ Non-positive Sharpe ({metrics['sharpe']})")

    if wf_fold_metrics:
        profitable_folds = sum(1 for f in wf_fold_metrics if f["total_trades"] > 0 and f["expectancy"] > 0)
        valid_folds = sum(1 for f in wf_fold_metrics if f["total_trades"] > 0)
        if valid_folds > 0:
            consistency = profitable_folds / valid_folds
            if consistency >= 0.6:
                score += 2; notes.append(f"✅ Edge held up in {profitable_folds}/{valid_folds} out-of-sample folds ({consistency:.0%}) — reasonably consistent")
            elif consistency >= 0.4:
                notes.append(f"⚠️ Edge held up in only {profitable_folds}/{valid_folds} folds ({consistency:.0%}) — inconsistent across time")
            else:
                score -= 2; notes.append(f"❌ Edge held up in only {profitable_folds}/{valid_folds} folds ({consistency:.0%}) — looks like overfitting to one period, not a real edge")
        else:
            notes.append("⚪ Walk-forward folds had too few trades each to judge.")
    else:
        notes.append("⚪ Walk-forward validation not run — this verdict is based on a single sample only, which is the weakest form of evidence. Turn it on for a more trustworthy read.")

    if cost_enabled and metrics_net is not None:
        if metrics_net["expectancy"] > 0:
            score += 1; notes.append(f"✅ Edge survives realistic costs (net expectancy {metrics_net['expectancy']} pts/trade)")
        else:
            score -= 2; notes.append(f"❌ Edge DISAPPEARS after realistic costs (net expectancy {metrics_net['expectancy']} pts/trade) — this is the single most common reason retail systems look good on paper and lose money live")
    else:
        notes.append("⚪ Cost modeling not run — raw backtest numbers usually overstate real returns once slippage, spread, and brokerage are included.")

    if score >= 4:
        verdict = "🟢 LIKELY TO BE PROFITABLE — reasonable candidate for cautious, small-size live/paper testing"
    elif score >= 1:
        verdict = "🟡 MARGINAL EDGE — proceed only with small size, tight risk control, and continued monitoring"
    else:
        verdict = "🔴 LIKELY TO CAUSE LOSSES — this configuration does not show a reliable edge; not recommended for live deployment as-is"

    return verdict, notes


def recommend_from_metrics(m):
    if m["total_trades"] < 5:
        return "⚪ Not enough trades in this sample to judge the strategy — widen the period."
    if m["accuracy"] >= 55 and m["expectancy"] > 0 and m["sharpe"] > 0.3:
        return f"🟢 Reasonable edge on this sample: {m['accuracy']}% win rate, positive expectancy ({m['expectancy']} pts/trade), Sharpe {m['sharpe']}. Still validate on more regimes and with realistic slippage/costs before sizing up."
    if m["expectancy"] > 0:
        return f"🟡 Marginal edge: expectancy is positive ({m['expectancy']} pts/trade) but win rate ({m['accuracy']}%) or Sharpe ({m['sharpe']}) is weak. Consider tightening filters (ADX/RSI/volume) or a different timeframe."
    return f"🔴 No robust edge detected on this sample (expectancy {m['expectancy']}, accuracy {m['accuracy']}%). Try other strategy/timeframe combos in the Optimization tab before trading this config live."


# ============================================================================
# DHAN BROKER PLACEHOLDER
# ============================================================================

def place_dhan_order(client_id, access_token, security_id, txn_type, product_cfg, qty, price=0.0, order_type="MARKET"):
    """
    Placeholder for Dhan Broker API (v2) integration.
    Real endpoint: POST https://api.dhan.co/orders
    Headers: {"access-token": <token>, "Content-Type": "application/json"}
    Payload keys typically include dhanClientId, transactionType, exchangeSegment,
    productType, orderType, securityId, quantity, price.

    This stub NEVER calls the network. Wire in `requests.post(...)` yourself
    only after validating credentials in Dhan's sandbox environment.
    """
    payload = {
        "dhanClientId": client_id,
        "transactionType": txn_type,
        "exchangeSegment": product_cfg.get("exchange_segment"),
        "productType": product_cfg.get("product"),
        "orderType": order_type,
        "securityId": security_id,
        "quantity": qty,
        "price": price,
        "instrument": product_cfg.get("instrument"),
    }
    return {"status": "SIMULATED_NOT_SENT", "payload": payload,
            "note": "Live Dhan order call is disabled in this build — replace this stub with a requests.post call."}


# ============================================================================
# SIDEBAR
# ============================================================================

ov = st.session_state.sidebar_overrides
st.sidebar.title("⚙️ Algo Configuration")

if ov:
    st.sidebar.success("Optimized config applied ✅")

ticker_names = list(TICKER_MAP.keys())
ticker_choice = st.sidebar.selectbox(
    "Ticker", ticker_names,
    index=ticker_names.index(ov.get("ticker_choice")) if ov.get("ticker_choice") in ticker_names else 0,
)
if ticker_choice == "Custom":
    ticker = st.sidebar.text_input("Custom Ticker (Yahoo Finance symbol)", ov.get("ticker", "RELIANCE.NS"))
else:
    ticker = TICKER_MAP[ticker_choice]

intervals = list(TF_PERIOD_MAP.keys())
interval = st.sidebar.selectbox(
    "Timeframe", intervals,
    index=intervals.index(ov.get("interval")) if ov.get("interval") in intervals else 4,
)
periods_available = TF_PERIOD_MAP[interval]
period = st.sidebar.selectbox(
    "Period", periods_available,
    index=periods_available.index(ov.get("period")) if ov.get("period") in periods_available else 0,
)

qty = st.sidebar.number_input("Quantity", min_value=1, value=int(ov.get("qty", 1)), step=1)

st.sidebar.markdown("### 📐 Strategy")
strategy = st.sidebar.selectbox(
    "Strategy", STRATEGIES,
    index=STRATEGIES.index(ov.get("strategy")) if ov.get("strategy") in STRATEGIES else 0,
)

params = {"ema_fast": ov.get("ema_fast", 9), "ema_slow": ov.get("ema_slow", 15)}

if strategy in ("EMA Crossover", "Pro: EMA50 Trend + EMA9/15 Pullback"):
    params["ema_fast"] = st.sidebar.number_input("EMA Fast", 2, 100, int(ov.get("ema_fast", 9)))
    params["ema_slow"] = st.sidebar.number_input("EMA Slow", 3, 200, int(ov.get("ema_slow", 15)))
if strategy == "Threshold Cross":
    params["threshold"] = st.sidebar.number_input("Threshold Price", value=float(ov.get("threshold", 0.0)))
if strategy == "Price Action Support/Resistance":
    params["sr_window"] = st.sidebar.number_input("S/R Lookback", 5, 200, int(ov.get("sr_window", 20)))
if strategy == "Liquidity Grab Reversal":
    params["liq_window"] = st.sidebar.number_input("Liquidity Lookback", 5, 200, int(ov.get("liq_window", 20)))
if strategy == "RSI Cross":
    params["rsi_period"] = st.sidebar.number_input("RSI Period", 2, 50, int(ov.get("rsi_period", 14)))
if strategy in ("Bollinger Bands", "Pro: BB+RSI Mean Reversion (ATR filtered)"):
    params["bb_period"] = st.sidebar.number_input("BB Period", 5, 100, int(ov.get("bb_period", 20)))
    params["bb_std"] = st.sidebar.number_input("BB Std Dev", 1.0, 4.0, float(ov.get("bb_std", 2.0)))
if strategy == "Volume Breakout":
    params["vol_window"] = st.sidebar.number_input("Volume Lookback", 5, 100, int(ov.get("vol_window", 20)))
    params["vol_factor"] = st.sidebar.number_input("Volume Spike Factor", 1.0, 5.0, float(ov.get("vol_factor", 2.0)))
if strategy == "Elliott Wave (Zigzag)":
    params["zigzag_lookback"] = st.sidebar.number_input("Zigzag Lookback", 2, 20, int(ov.get("zigzag_lookback", 3)))
if strategy == "Pro: VWAP + Supertrend Trend":
    params["st_period"] = st.sidebar.number_input("Supertrend Period", 5, 50, int(ov.get("st_period", 10)))
    params["st_mult"] = st.sidebar.number_input("Supertrend Multiplier", 1.0, 6.0, float(ov.get("st_mult", 3.0)))
if strategy == "Pro: Opening Range Breakout + Volume":
    params["orb_candles"] = st.sidebar.number_input("ORB Candles", 1, 30, int(ov.get("orb_candles", 5)))

if strategy in PRO_STRATEGIES:
    st.sidebar.caption("💡 Professional-grade composite strategy (trend/volatility/liquidity confluence).")

st.sidebar.markdown("### 🛑 Stoploss")
sl_type = st.sidebar.selectbox(
    "Stoploss Type", SL_TYPES,
    index=SL_TYPES.index(ov.get("sl_type")) if ov.get("sl_type") in SL_TYPES else 0,
)
params["sl_points"] = st.sidebar.number_input("SL Points (base)", 0.1, 100000.0, float(ov.get("sl_points", 10.0)))
if sl_type == "ATR Based SL":
    params["atr_mult_sl"] = st.sidebar.number_input("ATR Multiplier (SL)", 0.5, 5.0, float(ov.get("atr_mult_sl", 1.5)))

st.sidebar.markdown("### 🎯 Target")
target_type = st.sidebar.selectbox(
    "Target Type", TARGET_TYPES,
    index=TARGET_TYPES.index(ov.get("target_type")) if ov.get("target_type") in TARGET_TYPES else 0,
)
params["target_points"] = st.sidebar.number_input("Target Points (base)", 0.1, 200000.0, float(ov.get("target_points", 20.0)))
if target_type == "ATR Based Target":
    params["atr_mult_target"] = st.sidebar.number_input("ATR Multiplier (Target)", 1.0, 8.0, float(ov.get("atr_mult_target", 3.0)))
if sl_type == "Risk:Reward Based (min 1:2)" or target_type == "Risk:Reward Based (min 1:2)":
    params["rr_ratio"] = st.sidebar.number_input("Risk:Reward Ratio (min 2)", 2.0, 10.0, float(ov.get("rr_ratio", 2.0)))

st.sidebar.markdown("### 🔍 Additional Entry Filters")
filters = {
    "adx_enabled": st.sidebar.checkbox("ADX Filter", value=False),
}
if filters["adx_enabled"]:
    c1, c2 = st.sidebar.columns(2)
    filters["adx_min"] = c1.number_input("ADX Min", 0, 100, 20)
    filters["adx_max"] = c2.number_input("ADX Max", 0, 100, 100)
filters["rsi_enabled"] = st.sidebar.checkbox("RSI Filter (30 up-cross buy / 70 down-cross sell)", value=False)
filters["bb_enabled"] = st.sidebar.checkbox("Bollinger Band Filter", value=False)
filters["ema20_enabled"] = st.sidebar.checkbox("EMA20 Filter", value=False)
filters["sma20_enabled"] = st.sidebar.checkbox("SMA20 Filter", value=False)
filters["smc_enabled"] = st.sidebar.checkbox("SMC (Structure Break) Filter", value=False)

filters["atr_enabled"] = st.sidebar.checkbox("ATR (Volatility) Filter", value=False)
if filters["atr_enabled"]:
    c1, c2 = st.sidebar.columns(2)
    filters["atr_min"] = c1.number_input("ATR Min (points)", 0.0, 100000.0, 0.0)
    filters["atr_max"] = c2.number_input("ATR Max (points)", 0.0, 100000.0, 100000.0)
    st.sidebar.caption("Only trade when 14-period ATR is inside this band — avoids dead/illiquid tape and blow-off volatility spikes.")

filters["supertrend_enabled"] = st.sidebar.checkbox("Supertrend Filter", value=False)
if filters["supertrend_enabled"]:
    c1, c2 = st.sidebar.columns(2)
    filters["st_filter_period"] = c1.number_input("Supertrend Period (filter)", 5, 50, 10)
    filters["st_filter_mult"] = c2.number_input("Supertrend Mult (filter)", 1.0, 6.0, 3.0)
    st.sidebar.caption("Only takes buys when Supertrend is bullish, sells when Supertrend is bearish — independent of the main strategy.")

filters["regime_enabled"] = st.sidebar.checkbox("Regime Filter (Trend vs Range, adaptive)", value=False)
if filters["regime_enabled"]:
    c1, c2 = st.sidebar.columns(2)
    filters["regime_trend_min"] = c1.number_input("ADX ≥ this = Trending", 10, 60, 25)
    filters["regime_range_max"] = c2.number_input("ADX ≤ this = Ranging", 5, 40, 20)
    st.sidebar.caption(
        "Trend-type strategies (EMA/Supertrend/ORB/S-R/EW) only fire when ADX confirms a trend; "
        "mean-reversion strategies (RSI/Bollinger/Liquidity/BB+RSI) only fire when ADX confirms a range. "
        "This is the 'adapt to changing market regime' control — it doesn't switch strategies for you, "
        "it stops your chosen strategy from firing in the regime it's known to perform badly in."
    )

st.sidebar.markdown("### 🧠 Smart Evaluation (Recommended Before Going Live)")
st.sidebar.caption("Off by default. Turn these on to get a more honest read on whether a config is likely to hold up out-of-sample and after real costs.")

wf_enabled = st.sidebar.checkbox("Enable Walk-Forward Validation", value=False)
wf_folds = 5
if wf_enabled:
    wf_folds = st.sidebar.slider("Number of sequential out-of-sample folds", 3, 10, 5)
    st.sidebar.caption("Splits the backtest period into N sequential chunks and checks whether the edge holds up across most of them, not just in aggregate.")

cost_enabled = st.sidebar.checkbox("Enable Realistic Cost Modeling", value=False)
cost_cfg = {"slippage_points": 0.0, "spread_points": 0.0, "brokerage_flat": 0.0}
if cost_enabled:
    cost_cfg["slippage_points"] = st.sidebar.number_input("Slippage per trade (points)", 0.0, 10000.0, 1.0)
    cost_cfg["spread_points"] = st.sidebar.number_input("Bid-Ask spread cost (points)", 0.0, 10000.0, 0.5)
    cost_cfg["brokerage_flat"] = st.sidebar.number_input("Brokerage per order leg (currency)", 0.0, 10000.0, 20.0)
    st.sidebar.caption("Deducted from every trade: (slippage + spread) in points, plus brokerage charged twice per round trip (entry + exit).")

st.sidebar.markdown("### 🏦 Dhan Broker — Live Order Placement")
dhan_enabled = st.sidebar.checkbox("Enable Dhan Order Placement (LIVE)", value=False)
dhan_client_id, dhan_access_token, product_cfg = "", "", {}
if dhan_enabled:
    st.sidebar.warning("Live orders will be attempted using the credentials below. Test in a sandbox first.")
    dhan_client_id = st.sidebar.text_input("Dhan Client ID")
    dhan_access_token = st.sidebar.text_input("Dhan Access Token", type="password")
    instrument_type = st.sidebar.selectbox(
        "Instrument", [
            "Stock Intraday", "Stock Delivery", "Stock Futures", "Stock Options",
            "Index Futures (Nifty/BankNifty/Sensex)", "Index Options CE", "Index Options PE",
        ],
    )
    product_cfg["instrument"] = instrument_type
    if "Options" in instrument_type:
        c1, c2 = st.sidebar.columns(2)
        product_cfg["strike"] = c1.number_input("Strike Price", 0.0, 200000.0, 0.0)
        product_cfg["expiry"] = c2.text_input("Expiry (YYYY-MM-DD)")
    product_cfg["exchange_segment"] = st.sidebar.selectbox("Exchange Segment", ["NSE_EQ", "NSE_FNO", "BSE_EQ", "MCX_COMM"])
    product_cfg["product"] = st.sidebar.selectbox("Product Type", ["INTRADAY", "CNC", "MARGIN", "MTF"])
    product_cfg["order_mode"] = st.sidebar.selectbox("Order Mode", ["Buy then Buy (allow same-side re-entry)", "Flip only (Buy⇄Sell)"])
else:
    st.sidebar.caption("Disabled by default. Live trading tab runs in paper/simulation mode until enabled.")

config = dict(
    ticker=ticker, ticker_choice=ticker_choice, interval=interval, period=period, qty=qty,
    strategy=strategy, sl_type=sl_type, target_type=target_type, params=params, filters=filters,
    wf_enabled=wf_enabled, wf_folds=wf_folds, cost_enabled=cost_enabled, cost_cfg=cost_cfg,
)

# ============================================================================
# HELPERS SHARED ACROSS TABS
# ============================================================================

def price_chart(df, trades_df=None, title=""):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
    if trades_df is not None and not trades_df.empty:
        longs = trades_df[trades_df["Direction"] == "LONG"]
        shorts = trades_df[trades_df["Direction"] == "SHORT"]
        fig.add_trace(go.Scatter(x=longs["Entry Time"], y=longs["Entry Price"], mode="markers",
                                  marker=dict(symbol="triangle-up", color="lime", size=11), name="Long Entry"))
        fig.add_trace(go.Scatter(x=shorts["Entry Time"], y=shorts["Entry Price"], mode="markers",
                                  marker=dict(symbol="triangle-down", color="red", size=11), name="Short Entry"))
        fig.add_trace(go.Scatter(x=trades_df["Exit Time"], y=trades_df["Exit Price"], mode="markers",
                                  marker=dict(symbol="x", color="orange", size=9), name="Exit"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520, margin=dict(t=40, b=10))
    return fig


def apply_config_to_sidebar(cfg_row):
    """Push a chosen optimization result row into sidebar_overrides and rerun."""
    st.session_state.sidebar_overrides = {
        "ticker_choice": cfg_row.get("ticker_choice", ticker_choice),
        "ticker": cfg_row.get("ticker", ticker),
        "interval": cfg_row["Timeframe"],
        "period": cfg_row["Period"],
        "strategy": cfg_row["Strategy"],
        "sl_type": sl_type,
        "target_type": target_type,
        "qty": qty,
    }
    st.rerun()


# ============================================================================
# TABS
# ============================================================================

tab_bt, tab_live, tab_hist, tab_heat, tab_opt, tab_spread = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "📜 Trade History", "🔥 Heatmaps", "🧪 Optimization", "🔀 Spread Tool"]
)

# ---------------------------------------------------------------- BACKTEST -
with tab_bt:
    st.subheader(f"Backtest — {ticker_choice} ({ticker}) · {interval} · {period} · {strategy}")
    st.caption("Entry rule: signal confirmed on candle *n* → position opened at the **open of candle n+1**. "
               "Longs check SL (candle low) before target (candle high); shorts check SL (candle high) before target (candle low) — conservative fill assumption.")

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Fetching data and running backtest…"):
            raw = fetch_data(ticker, interval, period)
            if raw.empty:
                st.error("No data returned. Check ticker/timeframe/period combination.")
            else:
                trades_df, sig_df = run_backtest(raw, strategy, sl_type, target_type, params, filters, qty)
                st.session_state.last_backtest = trades_df
                st.session_state.last_backtest_df = sig_df

    trades_df = st.session_state.last_backtest
    sig_df = st.session_state.last_backtest_df

    if trades_df is not None and sig_df is not None and not sig_df.empty:
        st.plotly_chart(price_chart(sig_df, trades_df, "Price with Entries/Exits"), use_container_width=True)

        m = compute_metrics(trades_df)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", m["total_trades"])
        c2.metric("Accuracy", f"{m['accuracy']}%")
        c3.metric("Total Points", m["total_points"])
        c4.metric("Total PnL", m["total_pnl"])
        c5.metric("Expectancy/Trade", m["expectancy"])
        c6.metric("Sharpe", m["sharpe"])
        st.info(recommend_from_metrics(m))

        # ---- Realistic cost modeling ----
        m_net, trades_display = None, trades_df
        if cost_enabled:
            trades_costed = apply_cost_model(trades_df, cost_cfg, qty)
            m_net = compute_metrics_from_columns(trades_costed, "Points (Net)", "PnL (Net)")
            trades_display = trades_costed
            st.markdown("#### 💸 Cost-Adjusted Results")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Net Accuracy", f"{m_net['accuracy']}%", delta=f"{m_net['accuracy']-m['accuracy']:.1f}pp")
            cc2.metric("Net Total Points", m_net["total_points"], delta=round(m_net["total_points"] - m["total_points"], 2))
            cc3.metric("Net Total PnL", m_net["total_pnl"])
            cc4.metric("Net Expectancy/Trade", m_net["expectancy"], delta=round(m_net["expectancy"] - m["expectancy"], 2))

        # ---- Walk-forward validation ----
        wf_folds_result = None
        if wf_enabled:
            wf_folds_result = walk_forward_folds(trades_df, sig_df.index[0], sig_df.index[-1], wf_folds)
            st.markdown("#### 🧪 Walk-Forward Consistency (Out-of-Sample Folds)")
            wf_table = pd.DataFrame(wf_folds_result)[["Fold", "From", "To", "total_trades", "accuracy", "expectancy", "sharpe", "total_pnl"]]
            st.dataframe(wf_table, use_container_width=True, hide_index=True)
            profitable = sum(1 for f in wf_folds_result if f["total_trades"] > 0 and f["expectancy"] > 0)
            valid = sum(1 for f in wf_folds_result if f["total_trades"] > 0)
            st.caption(f"Profitable in {profitable}/{valid} folds with trades. This checks whether the edge is consistent across time, not concentrated in one lucky stretch.")

        # ---- Smart verdict ----
        st.markdown("#### 🧠 Smart Evaluation Verdict")
        verdict, notes = smart_verdict(m, wf_fold_metrics=wf_folds_result, cost_enabled=cost_enabled, metrics_net=m_net)
        st.subheader(verdict)
        for n in notes:
            st.write(n)
        st.caption("This is a transparent rule-based scorecard over the metrics above — not a trained ML model. Enable Walk-Forward Validation and Cost Modeling in the sidebar for a materially more trustworthy verdict before risking real capital.")

        st.markdown("#### Trade Log")
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    else:
        st.caption("Run a backtest to see results here. (This never writes into Live Trading or Trade History.)")

# ------------------------------------------------------------- LIVE TRADE -
with tab_live:
    st.subheader(f"Live (Paper) Trading — {ticker_choice} ({ticker})")
    st.caption("This is a simulation layer driven by the latest candle signal, refreshed on a lightweight fragment so the rest of the page doesn't reload.")

    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown("**Live Price**")
        ltp = live_ltp_fragment(ticker, "LTP")

    with colB:
        st.markdown("**Selected Configuration**")
        st.json({
            "Ticker": ticker, "Timeframe": interval, "Period": period, "Quantity": qty,
            "Strategy": strategy, "Stoploss Type": sl_type, "Target Type": target_type,
            "Filters Active": [k for k, v in filters.items() if v is True],
            "Dhan Live Orders": dhan_enabled,
        })

    if st.button("🔄 Evaluate current signal & manage paper position"):
        raw = fetch_data(ticker, interval, period)
        if raw.empty or len(raw) < 30:
            st.error("Not enough data to evaluate a signal.")
        else:
            live_filters = dict(filters)
            live_filters["current_strategy"] = strategy
            sig_df = apply_filters(generate_signals(raw, strategy, params), live_filters)
            a_series = atr(sig_df, 14)
            last_sig = int(sig_df["signal"].iloc[-2])  # last CLOSED candle's signal
            open_pos = st.session_state.live_positions

            if open_pos:
                pos = open_pos[0]
                i = len(sig_df) - 1
                candle = sig_df.iloc[i]
                pos = update_trade_levels(pos, i, sig_df, params, a_series)

                exited, exit_price, reason = False, None, None
                # A signal/EMA-reverse exit flagged on a PRIOR poll executes now
                # (approximates "next candle open" in a polling live loop).
                if pos.get("pending_exit_reason"):
                    exited, exit_price, reason = True, candle["Open"], pos["pending_exit_reason"]
                if not exited:
                    exited, exit_price, reason = check_hard_exit(pos, candle)
                if not exited:
                    sig_exit, sig_reason = detect_signal_exit_condition(pos, i, sig_df, params)
                    if sig_exit:
                        pos["pending_exit_reason"] = sig_reason

                pos["current_price"] = float(sig_df["Close"].iloc[-1])
                if exited:
                    points = (exit_price - pos["entry_price"]) * pos["direction"]
                    st.session_state.live_history.append({
                        "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
                        "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "Exit Time": sig_df.index[-1], "Exit Price": round(float(exit_price), 2),
                        "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
                        "Points": round(points, 2), "PnL": round(points * pos["qty"], 2),
                        "Exit Reason": reason, "Qty": pos["qty"],
                    })
                    st.session_state.live_positions = []
                    st.success(f"Position closed: {reason} @ {exit_price:.2f}")
                    if dhan_enabled:
                        side = "SELL" if pos["direction"] == 1 else "BUY"
                        st.json(place_dhan_order(dhan_client_id, dhan_access_token, ticker, side, product_cfg, pos["qty"], exit_price))
                else:
                    st.session_state.live_positions = [pos]
                    st.info("Position still open — levels updated.")
            elif last_sig != 0:
                entry_price = float(sig_df["Open"].iloc[-1])
                a_val = a_series.iloc[-2] if not np.isnan(a_series.iloc[-2]) else entry_price * 0.005
                sl, target, sl_dist, target_dist = calc_initial_sl_target(last_sig, entry_price, a_val, params, sl_type, target_type)
                new_pos = {
                    "entry_time": sig_df.index[-1], "entry_price": entry_price, "direction": last_sig,
                    "qty": qty, "sl": sl, "target": target, "initial_sl": sl, "initial_target": target,
                    "sl_dist": sl_dist, "target_dist": target_dist, "sl_type": sl_type, "target_type": target_type,
                    "highest": entry_price, "lowest": entry_price, "current_price": entry_price,
                    "pending_exit_reason": None,
                }
                st.session_state.live_positions = [new_pos]
                st.success(f"New {'LONG' if last_sig == 1 else 'SHORT'} position opened @ {entry_price:.2f}")
                if dhan_enabled:
                    side = "BUY" if last_sig == 1 else "SELL"
                    st.json(place_dhan_order(dhan_client_id, dhan_access_token, ticker, side, product_cfg, qty, entry_price))
            else:
                st.caption("No new signal on the latest closed candle.")

    st.markdown("#### Open Position")
    if st.session_state.live_positions:
        pos = st.session_state.live_positions[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Entry Type", "LONG" if pos["direction"] == 1 else "SHORT")
        c2.metric("Entry Price", f"{pos['entry_price']:.2f}")
        c3.metric("SL (live)", f"{pos['sl']:.2f}")
        c4.metric("Target (live)", f"{pos['target']:.2f}")
        c5.metric("Highest", f"{pos['highest']:.2f}")
        c6.metric("Lowest", f"{pos['lowest']:.2f}")
    else:
        st.caption("No open paper position.")

    st.markdown("#### Recent Trades")
    if st.session_state.live_history:
        st.dataframe(pd.DataFrame(st.session_state.live_history[-10:]), use_container_width=True, hide_index=True)
    else:
        st.caption("No live trades yet.")

# ------------------------------------------------------------- TRADE HIST -
with tab_hist:
    st.subheader("Trade History (Live/Paper only — never mixed with backtest)")
    hist_df = pd.DataFrame(st.session_state.live_history)
    if hist_df.empty:
        st.caption("No completed live trades yet.")
    else:
        m = compute_metrics(hist_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("No. of Trades", m["total_trades"])
        c2.metric("Accuracy", f"{m['accuracy']}%")
        c3.metric("Points Gained/Lost", m["total_points"])
        c4.metric("Total PnL", m["total_pnl"])
        c5.metric("Expectancy", m["expectancy"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Trade History"):
            st.session_state.live_history = []
            st.rerun()

# ---------------------------------------------------------------- HEATMAP -
with tab_heat:
    st.subheader("Return Heatmaps")

    st.markdown("##### 1) Monthly % Returns — Past 20 Years")
    if st.button("Generate 20Y Monthly Heatmap"):
        with st.spinner("Fetching 20 years of monthly candles…"):
            monthly = fetch_data(ticker, "1mo", "20y" if "1mo" not in TF_PERIOD_MAP.get("1mo", []) else "20y") \
                if False else None
            # yfinance accepts interval='1mo' with a long period directly:
            time.sleep(RATE_LIMIT_DELAY)
            monthly = yf.download(ticker, interval="1mo", period="20y", progress=False, auto_adjust=True)
            if isinstance(monthly.columns, pd.MultiIndex):
                monthly.columns = monthly.columns.get_level_values(0)
        if monthly is None or monthly.empty:
            st.error("No monthly data available for this ticker.")
        else:
            monthly = monthly.dropna()
            monthly["ret_pct"] = monthly["Close"].pct_change() * 100
            monthly["Year"] = monthly.index.year
            monthly["Month"] = monthly.index.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pivot = monthly.pivot_table(index="Year", columns="Month", values="ret_pct")
            pivot = pivot.reindex(columns=month_order)
            fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn", aspect="auto",
                             labels=dict(color="% return"))
            fig.update_layout(height=550, title=f"{ticker_choice} — Monthly % Returns (20Y)")
            st.plotly_chart(fig, use_container_width=True)

            latest_month = datetime.now().strftime("%b")
            if latest_month in pivot.columns:
                hist_avg = pivot[latest_month].mean()
                if hist_avg > 0.5:
                    st.success(f"📈 Historically, {latest_month} has averaged {hist_avg:.2f}% return over 20 years — seasonally favorable.")
                elif hist_avg < -0.5:
                    st.warning(f"📉 Historically, {latest_month} has averaged {hist_avg:.2f}% return over 20 years — seasonally weak, consider waiting.")
                else:
                    st.info(f"➖ {latest_month} has averaged {hist_avg:.2f}% historically — no strong seasonal bias.")

    st.divider()
    st.markdown(f"##### 2) % Returns Heatmap — Selected Timeframe ({interval}) & Period ({period})")
    if st.button("Generate Selected Timeframe Heatmap"):
        with st.spinner("Fetching and computing…"):
            raw = fetch_data(ticker, interval, period)
        if raw.empty:
            st.error("No data returned for this timeframe/period.")
        else:
            raw = raw.copy()
            raw["ret_pct"] = raw["Close"].pct_change() * 100
            if interval in ("1m", "5m", "15m", "1h"):
                raw["bucket_row"] = raw.index.strftime("%A")
                raw["bucket_col"] = raw.index.hour
                pivot2 = raw.pivot_table(index="bucket_row", columns="bucket_col", values="ret_pct", aggfunc="mean")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                pivot2 = pivot2.reindex([d for d in day_order if d in pivot2.index])
                x_label, y_label = "Hour of Day", "Day of Week"
            else:
                raw["bucket_row"] = raw.index.year
                raw["bucket_col"] = raw.index.strftime("%b")
                pivot2 = raw.pivot_table(index="bucket_row", columns="bucket_col", values="ret_pct", aggfunc="mean")
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot2 = pivot2.reindex(columns=[m for m in month_order if m in pivot2.columns])
                x_label, y_label = "Month", "Year"

            fig2 = px.imshow(pivot2, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto",
                              labels=dict(color="avg % return", x=x_label, y=y_label))
            fig2.update_layout(height=550, title=f"{ticker_choice} — Avg % Return · {interval}/{period}")
            st.plotly_chart(fig2, use_container_width=True)

            recent_momentum = raw["ret_pct"].tail(10).mean()
            overall_bias = pivot2.mean().mean()
            if recent_momentum > 0 and overall_bias > 0:
                st.success("🟢 Recent momentum and historical bias for this slot both lean positive — favorable window to consider buying, subject to your risk rules.")
            elif recent_momentum < 0 and overall_bias < 0:
                st.warning("🔴 Recent momentum and historical bias both lean negative — better to wait for confirmation.")
            else:
                st.info("🟡 Mixed signal between recent momentum and historical bias — no strong directional edge right now, wait for confluence.")

# ------------------------------------------------------------ OPTIMIZATION -
with tab_opt:
    st.subheader("Strategy Optimizer")
    st.caption("Runs backtests across chosen timeframe/period/strategy combinations. Keep the grid small — each combo triggers a rate-limited yfinance call.")

    opt_strategies = st.multiselect("Strategies to test", STRATEGIES, default=[strategy])
    opt_intervals = st.multiselect("Timeframes to test", list(TF_PERIOD_MAP.keys()), default=[interval])
    combo_periods = sorted({p for iv in opt_intervals for p in TF_PERIOD_MAP[iv]}, key=lambda x: x)
    opt_periods = st.multiselect("Periods to test (only valid ones per timeframe are used)", combo_periods, default=[period] if period in combo_periods else combo_periods[:1])

    est_calls = len(opt_strategies) * sum(1 for iv in opt_intervals for p in TF_PERIOD_MAP[iv] if p in opt_periods)
    st.caption(f"Estimated backtest runs: **{est_calls}** (≈{est_calls * RATE_LIMIT_DELAY:.1f}s+ just for data fetch delays).")

    if st.button("🧪 Run Optimization", type="primary"):
        rows = []
        progress = st.progress(0.0, text="Starting…")
        combos = [(s, iv, p) for s in opt_strategies for iv in opt_intervals for p in TF_PERIOD_MAP[iv] if p in opt_periods]
        for idx, (s, iv, p) in enumerate(combos):
            raw = fetch_data(ticker, iv, p)
            if not raw.empty and len(raw) >= 30:
                tdf, _ = run_backtest(raw, s, sl_type, target_type, params, filters, qty)
                m = compute_metrics(tdf)
                rows.append({"Strategy": s, "Timeframe": iv, "Period": p, **m})
            progress.progress((idx + 1) / max(len(combos), 1), text=f"{s} · {iv}/{p}")
        progress.empty()
        st.session_state.opt_results = pd.DataFrame(rows)

    results = st.session_state.opt_results
    if isinstance(results, pd.DataFrame) and not results.empty:
        rank_metric = st.selectbox("Rank by", ["accuracy", "sharpe", "expectancy", "total_pnl", "total_points"], index=1)

        best_overall = results.sort_values(rank_metric, ascending=False).iloc[0]
        st.success(f"🏆 Best overall by {rank_metric}: **{best_overall['Strategy']}** · {best_overall['Timeframe']}/{best_overall['Period']} "
                   f"(accuracy {best_overall['accuracy']}%, sharpe {best_overall['sharpe']}, expectancy {best_overall['expectancy']})")
        if st.button("✅ Apply BEST overall config to sidebar"):
            apply_config_to_sidebar(best_overall)

        st.markdown("---")
        for s in results["Strategy"].unique():
            sub = results[results["Strategy"] == s].sort_values(rank_metric, ascending=False).reset_index(drop=True)
            st.markdown(f"**{s}**")
            st.dataframe(sub, use_container_width=True, hide_index=True)
            rank_choice = st.number_input(f"Apply rank # for '{s}' (1 = best)", min_value=1, max_value=len(sub), value=1, key=f"rank_{s}")
            if st.button(f"Apply rank {rank_choice} config for '{s}'", key=f"apply_{s}"):
                apply_config_to_sidebar(sub.iloc[int(rank_choice) - 1])
    else:
        st.caption("Run the optimizer to see ranked results per strategy.")

# --------------------------------------------------------------- SPREAD --
with tab_spread:
    st.subheader("Cross-Asset Spread / Difference Tool")
    st.caption("Pick two (optionally a third) instruments, compare live prices, and get a simple directional read.")

    all_names = [n for n in TICKER_MAP if n != "Custom"] + ["Custom"]
    c1, c2, c3 = st.columns(3)
    with c1:
        t1_name = st.selectbox("Ticker 1", all_names, index=0, key="sp_t1")
        t1 = st.text_input("Custom symbol 1", "RELIANCE.NS", key="sp_t1_custom") if t1_name == "Custom" else TICKER_MAP[t1_name]
    with c2:
        t2_name = st.selectbox("Ticker 2", all_names, index=2, key="sp_t2")
        t2 = st.text_input("Custom symbol 2", "TCS.NS", key="sp_t2_custom") if t2_name == "Custom" else TICKER_MAP[t2_name]
    with c3:
        use_t3 = st.checkbox("Add Ticker 3 (optional)")
        t3 = None
        if use_t3:
            t3_name = st.selectbox("Ticker 3", all_names, index=3, key="sp_t3")
            t3 = st.text_input("Custom symbol 3", "HDFCBANK.NS", key="sp_t3_custom") if t3_name == "Custom" else TICKER_MAP[t3_name]

    if st.button("🔍 Fetch & Compare"):
        def get_ltp(sym):
            time.sleep(RATE_LIMIT_DELAY)
            try:
                d = yf.Ticker(sym).history(period="1d", interval="1m")
                if d.empty:
                    d = yf.Ticker(sym).history(period="5d", interval="15m")
                return float(d["Close"].iloc[-1]) if not d.empty else None
            except Exception:
                return None

        p1, p2 = get_ltp(t1), get_ltp(t2)
        p3 = get_ltp(t3) if t3 else None

        if p1 is None or p2 is None:
            st.error("Could not fetch one or more prices — check symbols.")
        else:
            diff = p1 - p2
            st.metric(f"{t1_name} price", f"{p1:,.2f}")
            st.metric(f"{t2_name} price", f"{p2:,.2f}")
            st.metric("Difference (T1 − T2)", f"{diff:,.2f}")
            if p3 is not None:
                st.metric(f"{t3_name} price", f"{p3:,.2f}")
                st.metric("Difference (T1 − T3)", f"{p1 - p3:,.2f}")

            base = max(abs(p1), abs(p2), 1e-9)
            rel = diff / base * 100
            if rel > 1:
                bias, sl_hint, tgt_hint = "UP", diff * 0.3, diff * 0.9
                st.success(f"🟢 Spread ({rel:.2f}% of price) leans in favor of {t1_name} strength — heuristic bias: market/pair may drift UP for {t1_name} relative to {t2_name}.")
            elif rel < -1:
                bias, sl_hint, tgt_hint = "DOWN", diff * 0.3, diff * 0.9
                st.warning(f"🔴 Spread ({rel:.2f}% of price) leans in favor of {t2_name} strength — heuristic bias: {t1_name} may drift DOWN relative to {t2_name}.")
            else:
                bias, sl_hint, tgt_hint = "NEUTRAL", 0, 0
                st.info("🟡 Spread is small/neutral — no strong directional edge from this pair right now.")

            if bias != "NEUTRAL":
                entry_ref = p1
                sl_ref = entry_ref - abs(sl_hint) if bias == "UP" else entry_ref + abs(sl_hint)
                tgt_ref = entry_ref + abs(tgt_hint) if bias == "UP" else entry_ref - abs(tgt_hint)
                st.markdown(f"**Heuristic reference levels for {t1_name}:** Entry ≈ `{entry_ref:.2f}` · SL ≈ `{sl_ref:.2f}` · Target ≈ `{tgt_ref:.2f}`")
            st.caption("This is a simple relative-strength heuristic based on the raw price gap — not a validated statistical-arbitrage model. Treat as a directional hint only, and confirm with your chosen strategy/indicators before entering.")

# ============================================================================
# FOOTER / GLOBAL DISCLAIMER
# ============================================================================

st.divider()
st.caption(
    "⚠️ Educational tool. Backtests use simplified conservative fill logic and ignore slippage, "
    "brokerage, taxes, and liquidity constraints — real results will differ. Verify any strategy on "
    "out-of-sample data and paper-trade before committing capital. The Dhan integration is a placeholder "
    "and performs no live network calls until you implement the `requests.post` call yourself."
)
