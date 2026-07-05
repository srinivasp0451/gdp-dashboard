"""
=====================================================================================
 ALGO TRADER PRO  —  Single-file Streamlit Algorithmic Trading Workbench
=====================================================================================
Tabs:
    1. Backtest      -> Click "Run Backtest" to test any of 20 strategies with
                         percentage / points / ATR-multiple / R-multiple / trailing
                         SL & Target, plus brokerage + slippage cost modelling.
    2. Live Trading  -> Manual "Refresh Signal", "Start/Stop Auto-Trading Engine",
                         and "Manual Square Off" controls, wired to guarded Dhan
                         order placement (CE/PE/Both, FUT, EQ).
    3. Optimization  -> 2D parameter grid-search with a minimum-accuracy (win rate)
                         filter, a ranked results table, and an "Apply Best Config
                         to Sidebar" button.
    4. Trade History -> Every order this session actually placed (manual or auto),
                         independent of the Backtest tab's simulated trade log.

Data source for historical bars: yfinance (rate-limited to 1 request / 0.3s and
cached). Dhan is used ONLY for order execution (guarded behind an explicit
checkbox + credentials, and nothing fires until you click a button).

IMPORTANT HONESTY NOTES (read before using real money):
  - No strategy here is guaranteed profitable. A high win rate is NOT the same as
    profitability — a few large losing trades can outweigh many small winners.
    Always look at win rate *together with* Sharpe/CAGR/max drawdown/profit factor.
  - The "Auto-Trading Engine" uses a Streamlit rerun-and-sleep loop, which only
    works while this browser tab stays open and the process stays alive. It is a
    convenience wrapper, not a resilient production scheduler — for real unattended
    automation, run the same signal/order logic from a standalone script under
    cron/systemd/APScheduler.
  - Verify Dhan's scrip-master column names and API parameters against current
    documentation before going live; they can change between API versions.
=====================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

try:
    from dhanhq import dhanhq
    DHAN_AVAILABLE = True
except Exception:
    DHAN_AVAILABLE = False

st.set_page_config(page_title="Algo Trader Pro", layout="wide", page_icon="📈")

# =====================================================================================
# SESSION STATE
# =====================================================================================
def init_state():
    defaults = {
        "backtest_result": None,   # dict: df, trades_df, equity_series, metrics
        "opt_result": None,        # dict: pivot tables, top table, best combo
        "live_running": False,     # auto-trading engine armed?
        "open_position": [],       # list of leg dicts currently open (live)
        "trade_history": [],       # list of dicts: every live order/close this session
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =====================================================================================
# CONSTANTS
# =====================================================================================
INDEX_MAP = {
    "NIFTY 50":   "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX":     "^BSESN",
    "FIN NIFTY":  "NIFTY_FIN_SERVICE.NS",
}

TIMEFRAME_MAP = {
    "1 Minute": "1m", "5 Minute": "5m", "15 Minute": "15m",
    "30 Minute": "30m", "1 Hour": "60m", "1 Day": "1d",
    "1 Week": "1wk", "1 Month": "1mo",
}

MAX_PERIOD_FOR_INTERVAL = {
    "1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d",
    "1d": "max", "1wk": "max", "1mo": "max",
}

PERIOD_OPTIONS = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

STRATEGIES = [
    "1. EMA Crossover (Trend)",
    "2. RSI Mean Reversion",
    "3. MACD Signal Crossover",
    "4. Bollinger Band Mean Reversion",
    "5. Bollinger Band Breakout",
    "6. Supertrend",
    "7. ADX + DI Trend Strength",
    "8. VWAP Reversion (Intraday)",
    "9. Opening Range Breakout (ORB)",
    "10. Donchian Channel Breakout",
    "11. Ichimoku Cloud Trend",
    "12. Stochastic Oscillator",
    "13. Parabolic SAR Trend Flip",
    "14. Triple EMA (TEMA) Crossover",
    "15. Keltner Channel Breakout",
    "16. Rate-of-Change Momentum",
    "17. Rolling Z-Score Mean Reversion",
    "18. RSI + MACD Confluence",
    "19. Heikin Ashi Trend Following",
    "20. CCI + Volume Spike Breakout",
]

STRATEGY_NOTES = {
    "1. EMA Crossover (Trend)": "Robust because it rides sustained trends and uses exponential smoothing to reduce whipsaw vs SMA.",
    "2. RSI Mean Reversion": "Fades extremes (oversold/overbought) — works best in range-bound markets, weak in strong trends.",
    "3. MACD Signal Crossover": "Combines trend + momentum; signal-line smoothing reduces false triggers vs raw MACD line.",
    "4. Bollinger Band Mean Reversion": "Adaptive volatility bands (not fixed %) fade moves back to the mean — self-adjusts to regime.",
    "5. Bollinger Band Breakout": "Same adaptive bands used the opposite way — trades expansion instead of reversion.",
    "6. Supertrend": "ATR-based trailing stop-and-reverse system; very popular for intraday index/stock trend following.",
    "7. ADX + DI Trend Strength": "Only trades DI crossovers when ADX confirms trend strength — filters weak/choppy signals.",
    "8. VWAP Reversion (Intraday)": "Institutional benchmark; price reverting to VWAP is a classic intraday mean-reversion edge.",
    "9. Opening Range Breakout (ORB)": "Classic intraday breakout off the first N minutes' range — robust across many liquid instruments.",
    "10. Donchian Channel Breakout": "Turtle-trader style — buy new highs, sell new lows; a benchmark trend strategy.",
    "11. Ichimoku Cloud Trend": "Multi-timeframe trend/support-resistance system, robust across swing timeframes.",
    "12. Stochastic Oscillator": "Momentum oscillator crossover in overbought/oversold zones, good for range-bound swings.",
    "13. Parabolic SAR Trend Flip": "Accelerating trailing stop that flips with trend reversals — good trend/exit signal.",
    "14. Triple EMA (TEMA) Crossover": "Reduces lag of normal EMA while remaining smooth — reacts faster to genuine trend changes.",
    "15. Keltner Channel Breakout": "ATR-based channel (smoother than Bollinger) — robust breakout confirmation.",
    "16. Rate-of-Change Momentum": "Pure momentum: trades in the direction of recent acceleration in price.",
    "17. Rolling Z-Score Mean Reversion": "Statistically normalized deviation from mean — adapts across instruments/volatility regimes.",
    "18. RSI + MACD Confluence": "Requires two independent signals to agree — reduces false positives vs single-indicator systems.",
    "19. Heikin Ashi Trend Following": "Smoothed candles filter noise, making trend continuation/reversal visually and mechanically clearer.",
    "20. CCI + Volume Spike Breakout": "Confirms price extremes (CCI) with participation (volume spike) — reduces low-conviction breakouts.",
}

SEGMENT_TO_PRODUCT = {
    "Delivery (CNC)": "CNC",
    "Intraday (MIS)": "INTRADAY",
    "Futures (FUT)": "MARGIN",
    "Options (CE/PE)": "MARGIN",
}

EXCHANGE_SEGMENTS = ["NSE_EQ", "NSE_FNO", "BSE_EQ", "BSE_FNO", "MCX_COMM", "IDX_I"]

SL_TYPES = ["Percentage", "Points (Absolute)", "ATR Multiple"]
TARGET_TYPES = ["Percentage", "Points (Absolute)", "ATR Multiple", "R-Multiple (of Stop Loss)"]
TRAIL_TYPES = ["Points (Absolute)", "ATR Multiple"]

# Yahoo Finance is rate-limited; pause briefly before every real network call to it.
# (st.cache_data means this only fires on genuine cache misses, not on every rerun.)
YF_REQUEST_DELAY_SECONDS = 0.3


# =====================================================================================
# INDICATORS
# =====================================================================================
def EMA(s, span):
    return s.ewm(span=span, adjust=False).mean()


def SMA(s, window):
    return s.rolling(window).mean()


def RSI(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def MACD(s, fast=12, slow=26, signal=9):
    ema_fast = EMA(s, fast)
    ema_slow = EMA(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def BBANDS(s, window=20, num_std=2):
    ma = SMA(s, window)
    std = s.rolling(window).std()
    return ma + num_std * std, ma, ma - num_std * std


def ATR(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def SUPERTREND(df, period=10, multiplier=3.0):
    atr = ATR(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    close = df["Close"]
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    n = len(df)
    direction = np.ones(n, dtype=int)
    st_line = np.zeros(n)
    st_line[0] = final_lower.iloc[0]
    for i in range(1, n):
        if close.iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = upperband.iloc[i]
        if close.iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = lowerband.iloc[i]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            direction[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        st_line[i] = final_lower.iloc[i] if direction[i] == 1 else final_upper.iloc[i]
    return pd.Series(st_line, index=df.index), pd.Series(direction, index=df.index)


def ADX(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def VWAP(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    grp = df.index.date
    tp_vol = (typical * df["Volume"])
    return tp_vol.groupby(grp).cumsum() / df["Volume"].groupby(grp).cumsum()


def DONCHIAN(df, window=20):
    upper = df["High"].rolling(window).max()
    lower = df["Low"].rolling(window).min()
    return upper, (upper + lower) / 2, lower


def ICHIMOKU(df):
    high9, low9 = df["High"].rolling(9).max(), df["Low"].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26, low26 = df["High"].rolling(26).max(), df["Low"].rolling(26).min()
    kijun = (high26 + low26) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    high52, low52 = df["High"].rolling(52).max(), df["Low"].rolling(52).min()
    senkou_b = ((high52 + low52) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b


def STOCHASTIC(df, k_period=14, d_period=3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)


def PSAR(df, af_step=0.02, af_max=0.2):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    psar = close.copy().astype(float)
    bull = True
    af = af_step
    hp, lp = high[0], low[0]
    psar[0] = close[0]
    for i in range(1, n):
        prev = psar[i - 1]
        psar[i] = prev + af * ((hp if bull else lp) - prev)
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull, reverse = False, True
                psar[i] = hp
                lp = low[i]
                af = af_step
        else:
            if high[i] > psar[i]:
                bull, reverse = True, True
                psar[i] = lp
                hp = high[i]
                af = af_step
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_step, af_max)
                psar[i] = min(psar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_step, af_max)
                psar[i] = max(psar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
    return pd.Series(psar, index=df.index)


def TEMA(s, span):
    e1 = EMA(s, span)
    e2 = EMA(e1, span)
    e3 = EMA(e2, span)
    return 3 * e1 - 3 * e2 + e3


def KELTNER(df, ema_period=20, atr_period=10, mult=2.0):
    mid = EMA(df["Close"], ema_period)
    atr = ATR(df, atr_period)
    return mid + mult * atr, mid, mid - mult * atr


def ROC(s, period=12):
    return (s - s.shift(period)) / s.shift(period) * 100


def ZSCORE(s, window=20):
    ma = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - ma) / std.replace(0, np.nan)


def HEIKIN_ASHI(df):
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_close.iloc[i - 1]) / 2)
    ha_open = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def CCI(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def build_position(long_cond, short_cond):
    """Convert crossover boolean conditions into a persistent -1/0/1 position series."""
    sig = pd.Series(np.nan, index=long_cond.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig.ffill().fillna(0)


# =====================================================================================
# STRATEGY SIGNAL ENGINE  (20 strategies -> position series of -1/0/1)
# =====================================================================================
def get_position(df, strategy, p):
    close = df["Close"]

    if strategy == STRATEGIES[0]:  # EMA Crossover
        fast, slow = EMA(close, p["fast"]), EMA(close, p["slow"])
        return build_position(fast > slow, fast < slow)

    if strategy == STRATEGIES[1]:  # RSI Mean Reversion
        rsi = RSI(close, p["rsi_period"])
        return build_position(rsi < p["rsi_lower"], rsi > p["rsi_upper"])

    if strategy == STRATEGIES[2]:  # MACD Crossover
        macd_line, signal_line, _ = MACD(close)
        return build_position(macd_line > signal_line, macd_line < signal_line)

    if strategy == STRATEGIES[3]:  # Bollinger Mean Reversion
        upper, mid, lower = BBANDS(close, p["bb_window"], p["bb_std"])
        return build_position(close < lower, close > upper)

    if strategy == STRATEGIES[4]:  # Bollinger Breakout
        upper, mid, lower = BBANDS(close, p["bb_window"], p["bb_std"])
        return build_position(close > upper, close < lower)

    if strategy == STRATEGIES[5]:  # Supertrend
        _, direction = SUPERTREND(df, p["atr_period"], p["st_mult"])
        return direction.astype(float)

    if strategy == STRATEGIES[6]:  # ADX + DI
        adx, plus_di, minus_di = ADX(df, p["adx_period"])
        long_c = (plus_di > minus_di) & (adx > p["adx_threshold"])
        short_c = (minus_di > plus_di) & (adx > p["adx_threshold"])
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[7]:  # VWAP Reversion
        vwap = VWAP(df)
        return build_position(close < vwap * 0.998, close > vwap * 1.002)

    if strategy == STRATEGIES[8]:  # Opening Range Breakout
        df2 = df.copy()
        df2["date"] = df2.index.date
        n_bars = max(int(p["orb_minutes"] / p.get("_bar_minutes", 5)), 1)
        or_high, or_low = {}, {}
        for d, grp in df2.groupby("date"):
            or_high[d] = grp["High"].iloc[:n_bars].max()
            or_low[d] = grp["Low"].iloc[:n_bars].min()
        oh = df2["date"].map(or_high)
        ol = df2["date"].map(or_low)
        long_c = pd.Series(close.values > oh.values, index=df.index)
        short_c = pd.Series(close.values < ol.values, index=df.index)
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[9]:  # Donchian Breakout
        upper, mid, lower = DONCHIAN(df, p["donchian_window"])
        return build_position(close >= upper.shift(1), close <= lower.shift(1))

    if strategy == STRATEGIES[10]:  # Ichimoku
        tenkan, kijun, senkou_a, senkou_b = ICHIMOKU(df)
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        long_c = (tenkan > kijun) & (close > cloud_top)
        short_c = (tenkan < kijun) & (close < cloud_bot)
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[11]:  # Stochastic
        k, d = STOCHASTIC(df, p["stoch_k"], p["stoch_d"])
        long_c = (k > d) & (k < 30)
        short_c = (k < d) & (k > 70)
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[12]:  # Parabolic SAR
        psar = PSAR(df)
        return build_position(close > psar, close < psar)

    if strategy == STRATEGIES[13]:  # TEMA Crossover
        fast, slow = TEMA(close, p["fast"]), TEMA(close, p["slow"])
        return build_position(fast > slow, fast < slow)

    if strategy == STRATEGIES[14]:  # Keltner Breakout
        upper, mid, lower = KELTNER(df, p["slow"], p["atr_period"], p["keltner_mult"])
        return build_position(close > upper, close < lower)

    if strategy == STRATEGIES[15]:  # ROC Momentum
        roc = ROC(close, p["roc_period"])
        return build_position(roc > 0, roc < 0)

    if strategy == STRATEGIES[16]:  # Z-score Mean Reversion
        z = ZSCORE(close, p["zscore_window"])
        return build_position(z < -p["zscore_threshold"], z > p["zscore_threshold"])

    if strategy == STRATEGIES[17]:  # RSI + MACD Confluence
        rsi = RSI(close, p["rsi_period"])
        macd_line, signal_line, _ = MACD(close)
        long_c = (rsi > 50) & (macd_line > signal_line)
        short_c = (rsi < 50) & (macd_line < signal_line)
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[18]:  # Heikin Ashi Trend
        ha_open, ha_high, ha_low, ha_close = HEIKIN_ASHI(df)
        return build_position(ha_close > ha_open, ha_close < ha_open)

    if strategy == STRATEGIES[19]:  # CCI + Volume Spike
        cci = CCI(df, p["cci_period"])
        vol_avg = df["Volume"].rolling(20).mean()
        vol_spike = df["Volume"] > 1.5 * vol_avg
        long_c = (cci > 100) & vol_spike
        short_c = (cci < -100) & vol_spike
        return build_position(long_c, short_c)

    return pd.Series(0, index=df.index)


# =====================================================================================
# DATA FETCH
# =====================================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, interval, period):
    max_p = MAX_PERIOD_FOR_INTERVAL.get(interval, "max")
    order = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
    if max_p != "max" and period in order and order.index(period) > order.index(max_p):
        period = max_p
    try:
        time.sleep(YF_REQUEST_DELAY_SECONDS)  # throttle to avoid Yahoo Finance rate limits
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return df
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


def bar_minutes_from_interval(interval):
    return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 375, "1wk": 375, "1mo": 375}.get(interval, 5)


# =====================================================================================
# SL / TARGET MODELS  (percentage / points / ATR-multiple / R-multiple / trailing)
# =====================================================================================
def compute_initial_sl_target(entry_price, side, atr_val, sl_type, sl_value, target_type, target_value):
    """side: +1 long, -1 short. Returns (sl_price, target_price, sl_distance)."""
    atr_val = atr_val if pd.notna(atr_val) and atr_val > 0 else entry_price * 0.005

    if sl_type == "Percentage":
        sl_dist = entry_price * sl_value / 100
    elif sl_type == "Points (Absolute)":
        sl_dist = sl_value
    else:  # ATR Multiple
        sl_dist = atr_val * sl_value
    sl_dist = max(sl_dist, 1e-6)
    sl_price = entry_price - sl_dist * side

    if target_type == "Percentage":
        tgt_dist = entry_price * target_value / 100
    elif target_type == "Points (Absolute)":
        tgt_dist = target_value
    elif target_type == "ATR Multiple":
        tgt_dist = atr_val * target_value
    else:  # R-Multiple of Stop Loss
        tgt_dist = sl_dist * target_value
    target_price = entry_price + tgt_dist * side

    return sl_price, target_price, sl_dist


def update_trailing_sl(sl_price, side, extreme_price, trail_type, trail_value, atr_val):
    if trail_type == "Points (Absolute)":
        candidate = extreme_price - trail_value * side
    else:  # ATR Multiple
        atr_val = atr_val if pd.notna(atr_val) and atr_val > 0 else 0
        candidate = extreme_price - trail_value * atr_val * side
    return max(sl_price, candidate) if side == 1 else min(sl_price, candidate)


# =====================================================================================
# BACKTEST ENGINE
# =====================================================================================
def run_backtest(df, position, atr_series, qty, sl_type, sl_value, target_type, target_value,
                  trailing_enabled, trail_type, trail_value, brokerage_per_trade, slippage_pct, capital):
    df = df.copy()
    df["position"] = position.reindex(df.index).fillna(0)
    atr_series = atr_series.reindex(df.index)

    trades = []
    equity_curve = []
    cash = capital
    in_trade = False
    side = 0
    entry_price = entry_price_exec = 0.0
    entry_date = None
    sl_price = target_price = 0.0
    extreme_price = 0.0
    pos_prev = 0

    def slip(px, direction):
        # direction=+1 means we are BUYING at px (pay slightly more); -1 means SELLING (receive slightly less)
        return px * (1 + slippage_pct / 100 * direction)

    for ts, row in df.iterrows():
        price, high, low = row["Close"], row["High"], row["Low"]
        pos = row["position"]
        atr_val = atr_series.loc[ts] if ts in atr_series.index else np.nan

        if in_trade:
            if trailing_enabled:
                if side == 1:
                    extreme_price = max(extreme_price, high)
                else:
                    extreme_price = min(extreme_price, low)
                sl_price = update_trailing_sl(sl_price, side, extreme_price, trail_type, trail_value, atr_val)

            hit, exit_price_raw, reason = False, None, None
            if side == 1:
                if low <= sl_price:
                    hit, exit_price_raw, reason = True, sl_price, "SL"
                elif high >= target_price:
                    hit, exit_price_raw, reason = True, target_price, "Target"
            else:
                if high >= sl_price:
                    hit, exit_price_raw, reason = True, sl_price, "SL"
                elif low <= target_price:
                    hit, exit_price_raw, reason = True, target_price, "Target"

            if hit:
                exit_price = slip(exit_price_raw, -side)
                gross = (exit_price - entry_price_exec) * qty * side
                net = gross - brokerage_per_trade
                cash += net
                trades.append(dict(entry_date=entry_date, exit_date=ts, side="LONG" if side == 1 else "SHORT",
                                    entry_price=entry_price_exec, exit_price=exit_price, qty=qty, pnl=net, reason=reason))
                in_trade = False

        if pos != pos_prev:
            if in_trade:
                exit_price = slip(price, -side)
                gross = (exit_price - entry_price_exec) * qty * side
                net = gross - brokerage_per_trade
                cash += net
                trades.append(dict(entry_date=entry_date, exit_date=ts, side="LONG" if side == 1 else "SHORT",
                                    entry_price=entry_price_exec, exit_price=exit_price, qty=qty, pnl=net, reason="Signal Flip"))
                in_trade = False
            if pos != 0:
                in_trade = True
                side = int(pos)
                entry_price = price
                entry_price_exec = slip(price, side)
                entry_date = ts
                sl_price, target_price, _ = compute_initial_sl_target(
                    entry_price_exec, side, atr_val, sl_type, sl_value, target_type, target_value)
                extreme_price = entry_price_exec

        unrealized = (price - entry_price_exec) * qty * side if in_trade else 0
        equity_curve.append(cash + unrealized)
        pos_prev = pos

    if in_trade:
        last_ts = df.index[-1]
        exit_price = slip(df["Close"].iloc[-1], -side)
        gross = (exit_price - entry_price_exec) * qty * side
        net = gross - brokerage_per_trade
        cash += net
        trades.append(dict(entry_date=entry_date, exit_date=last_ts, side="LONG" if side == 1 else "SHORT",
                            entry_price=entry_price_exec, exit_price=exit_price, qty=qty, pnl=net, reason="End of Data"))

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, index=df.index)
    return trades_df, equity_series


def calc_metrics(trades_df, equity_series, capital):
    if trades_df is None or len(trades_df) == 0:
        return dict(total_trades=0, win_rate=0, total_pnl=0, total_return_pct=0,
                    profit_factor=0, sharpe=0, max_dd=0, cagr=0)

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    total_pnl = trades_df["pnl"].sum()
    win_rate = len(wins) / len(trades_df) * 100
    loss_sum = abs(losses["pnl"].sum())
    profit_factor = (wins["pnl"].sum() / loss_sum) if loss_sum > 0 else np.inf

    rets = equity_series.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() not in (0, np.nan) else 0

    running_max = equity_series.cummax()
    dd = (equity_series - running_max) / running_max * 100
    max_dd = dd.min()

    days = max((trades_df["exit_date"].max() - trades_df["entry_date"].min()).days, 1)
    years = max(days / 365, 0.02)
    final_val = equity_series.iloc[-1]
    cagr = ((final_val / capital) ** (1 / years) - 1) * 100 if final_val > 0 else -100

    return dict(total_trades=len(trades_df), win_rate=win_rate, total_pnl=total_pnl,
                total_return_pct=total_pnl / capital * 100, profit_factor=profit_factor,
                sharpe=sharpe, max_dd=max_dd, cagr=cagr)


# =====================================================================================
# HEATMAP HELPERS
# =====================================================================================
def monthly_returns_heatmap_data(ticker, n_years):
    period = f"{min(n_years + 1, 20)}y"
    df = fetch_data(ticker, "1d", period)
    if df.empty:
        return None
    monthly = df["Close"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100
    table = pd.DataFrame({"Year": monthly_ret.index.year, "Month": monthly_ret.index.month, "Return": monthly_ret.values})
    pivot = table.pivot_table(index="Year", columns="Month", values="Return")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[c - 1] for c in pivot.columns]
    return pivot.tail(n_years)


def daily_range_heatmap_data(ticker, n_years):
    period = f"{min(n_years + 1, 20)}y"
    df = fetch_data(ticker, "1d", period)
    if df.empty:
        return None
    rng_pct = (df["High"] - df["Low"]) / df["Close"] * 100
    table = pd.DataFrame({"Year": df.index.year, "Month": df.index.month, "Range": rng_pct.values})
    pivot = table.pivot_table(index="Year", columns="Month", values="Range", aggfunc="mean")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[c - 1] for c in pivot.columns]
    return pivot.tail(n_years)


def selected_timeframe_heatmap(df, interval):
    rets = df["Close"].pct_change().dropna() * 100
    if interval in ("1m", "5m", "15m", "30m", "60m"):
        hour = rets.index.hour
        weekday = rets.index.day_name()
        table = pd.DataFrame({"Weekday": weekday, "Hour": hour, "Return": rets.values})
        pivot = table.pivot_table(index="Weekday", columns="Hour", values="Return", aggfunc="mean")
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        pivot = pivot.reindex([d for d in order if d in pivot.index])
    else:
        year = rets.index.year
        month = rets.index.month
        table = pd.DataFrame({"Year": year, "Month": month, "Return": rets.values})
        pivot = table.pivot_table(index="Year", columns="Month", values="Return", aggfunc="sum")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_names[c - 1] for c in pivot.columns]
    return pivot


def plot_heatmap(pivot, title, colorscale="RdYlGn", zmid=None):
    if pivot is None or pivot.empty:
        st.info("Not enough data to build this heatmap.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
        colorscale=colorscale, zmid=zmid,
        text=np.round(pivot.values, 2), texttemplate="%{text}", hoverongaps=False,
    ))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================================
# DHAN INTEGRATION (guarded)
# =====================================================================================
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_dhan_scrip_master():
    try:
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        return pd.read_csv(url, low_memory=False)
    except Exception:
        return pd.DataFrame()


def place_dhan_order(client_id, token, security_id, exchange_segment, transaction_type,
                      quantity, product_type, order_type="MARKET", price=0):
    if not DHAN_AVAILABLE:
        return {"error": "dhanhq package not installed. Run: pip install dhanhq"}
    if not client_id or not token or not security_id:
        return {"error": "Missing client_id / token / security_id"}
    try:
        dhan = dhanhq(client_id, token)
        resp = dhan.place_order(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=int(quantity),
            order_type=order_type,
            product_type=product_type,
            price=float(price),
        )
        return resp
    except Exception as e:
        return {"error": str(e)}


def log_trade(entry_time, exit_time, side, security_id, qty, entry_price, exit_price, reason, order_resp=None):
    pnl = (exit_price - entry_price) * qty * (1 if side == "LONG" else -1) if exit_price is not None else None
    st.session_state.trade_history.append(dict(
        entry_time=entry_time, exit_time=exit_time, side=side, security_id=security_id,
        qty=qty, entry_price=entry_price, exit_price=exit_price, pnl=pnl, reason=reason,
        order_response=str(order_resp)[:200] if order_resp else "",
    ))


# =====================================================================================
# SIDEBAR — GLOBAL CONFIG
# =====================================================================================
st.sidebar.title("⚙️ Configuration")

asset_class = st.sidebar.selectbox("Asset Class", ["Index", "Stock"])
if asset_class == "Index":
    index_choice = st.sidebar.selectbox("Index", list(INDEX_MAP.keys()))
    yf_ticker = INDEX_MAP[index_choice]
    display_name = index_choice
else:
    custom_ticker = st.sidebar.text_input("NSE Symbol (e.g. RELIANCE, TCS)", "RELIANCE")
    yf_ticker = custom_ticker.strip().upper() + ".NS"
    display_name = custom_ticker.strip().upper()

segment = st.sidebar.selectbox("Segment", ["Delivery (CNC)", "Intraday (MIS)", "Futures (FUT)", "Options (CE/PE)"])
option_side = None
if segment == "Options (CE/PE)":
    option_side = st.sidebar.radio("Option Leg", ["CE", "PE", "Both (Straddle/Strangle)"])

timeframe_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=5)
interval = TIMEFRAME_MAP[timeframe_label]
period = st.sidebar.selectbox("History Period", PERIOD_OPTIONS, index=4)

strategy = st.sidebar.selectbox("Strategy", STRATEGIES)
st.sidebar.caption(STRATEGY_NOTES.get(strategy, ""))

with st.sidebar.expander("Strategy Parameters", expanded=False):
    st.caption("Changed by 'Apply Best Config' on the Optimization tab too.")
    p = {}
    p["fast"] = st.number_input("Fast Period", 2, 100, 9, key="p_fast")
    p["slow"] = st.number_input("Slow Period", 5, 300, 21, key="p_slow")
    p["rsi_period"] = st.number_input("RSI Period", 2, 50, 14, key="p_rsi_period")
    p["rsi_upper"] = st.number_input("RSI Overbought", 50, 95, 70, key="p_rsi_upper")
    p["rsi_lower"] = st.number_input("RSI Oversold", 5, 50, 30, key="p_rsi_lower")
    p["bb_window"] = st.number_input("Bollinger/Keltner Window", 5, 100, 20, key="p_bb_window")
    p["bb_std"] = st.number_input("Bollinger Std Dev", 1.0, 4.0, 2.0, key="p_bb_std")
    p["atr_period"] = st.number_input("ATR Period", 5, 50, 14, key="p_atr_period")
    p["st_mult"] = st.number_input("Supertrend Multiplier", 1.0, 6.0, 3.0, key="p_st_mult")
    p["adx_period"] = st.number_input("ADX Period", 5, 50, 14, key="p_adx_period")
    p["adx_threshold"] = st.number_input("ADX Threshold", 10, 50, 25, key="p_adx_threshold")
    p["orb_minutes"] = st.number_input("ORB Minutes", 5, 60, 15, key="p_orb_minutes")
    p["donchian_window"] = st.number_input("Donchian Window", 5, 100, 20, key="p_donchian_window")
    p["stoch_k"] = st.number_input("Stochastic %K", 5, 30, 14, key="p_stoch_k")
    p["stoch_d"] = st.number_input("Stochastic %D", 2, 10, 3, key="p_stoch_d")
    p["keltner_mult"] = st.number_input("Keltner Multiplier", 1.0, 4.0, 2.0, key="p_keltner_mult")
    p["roc_period"] = st.number_input("ROC Period", 5, 50, 12, key="p_roc_period")
    p["zscore_window"] = st.number_input("Z-Score Window", 5, 100, 20, key="p_zscore_window")
    p["zscore_threshold"] = st.number_input("Z-Score Threshold", 1.0, 4.0, 2.0, key="p_zscore_threshold")
    p["cci_period"] = st.number_input("CCI Period", 5, 50, 20, key="p_cci_period")
    p["_bar_minutes"] = bar_minutes_from_interval(interval)

st.sidebar.subheader("💰 Trade Setup")
if segment in ("Futures (FUT)", "Options (CE/PE)"):
    qty = st.sidebar.number_input("Number of Lots", 1, 100000, 1)
    lot_size = st.sidebar.number_input(
        "Lot Size (shares per lot)", 1, 100000, 1,
        help="⚠️ NSE sets and periodically revises this per instrument. There is no safe generic "
             "default — look up the current lot size for this specific contract on the NSE F&O "
             "website before entering a number here.")
    st.sidebar.caption("⚠️ Verify the real lot size on NSE — don't trust a guessed default.")
else:
    qty = st.sidebar.number_input("Quantity (shares)", 1, 100000, 1)
    lot_size = 1
capital = st.sidebar.number_input("Capital (₹)", 1000, 100_000_000, 100000, step=5000)

st.sidebar.markdown("**Stop Loss**")
with st.sidebar.expander("📏 Check this instrument's actual volatility first", expanded=False):
    st.caption("A fixed-points SL that isn't sized to the instrument's real movement gets stopped "
               "out by noise almost every trade, no matter how good the entry signal is.")
    if st.button("Check current ATR", key="check_atr_btn"):
        _atr_df = fetch_data(yf_ticker, interval, period)
        if _atr_df.empty:
            st.warning("No data returned for this ticker/timeframe.")
        else:
            _atr_val = ATR(_atr_df, p["atr_period"]).iloc[-1]
            _last_px = _atr_df["Close"].iloc[-1]
            st.write(f"**ATR({p['atr_period']}) on {timeframe_label}: {_atr_val:.1f} points** "
                     f"(~{_atr_val / _last_px * 100:.2f}% of last price {_last_px:.1f})")
            st.caption(f"A 10-point SL here is ~{10 / _atr_val:.2f}x ATR — "
                       f"if that's well under 0.5x, expect near-constant stop-outs. "
                       f"Consider SL Type = 'ATR Multiple' instead of fixed points.")

sl_type = st.sidebar.selectbox("SL Type", SL_TYPES, key="sl_type",
                                help="Percentage of entry price, fixed points, or a multiple of ATR "
                                     "(volatility-adjusted — recommended for stocks, since fixed points "
                                     "that make sense for one stock are meaningless for another).")
sl_value = st.sidebar.number_input("SL Value", 0.0, 10000.0, 1.0, key="sl_value",
                                    help="Meaning depends on SL Type above (e.g. 1.0 = 1% or 1.0 = 1x ATR).")

st.sidebar.markdown("**Target**")
target_type = st.sidebar.selectbox("Target Type", TARGET_TYPES, key="target_type",
                                    help="R-Multiple = a multiple of your Stop Loss distance (e.g. 2 = 2:1 reward:risk).")
target_value = st.sidebar.number_input("Target Value", 0.0, 10000.0, 2.0, key="target_value")

trailing_enabled = st.sidebar.checkbox("Enable Trailing Stop Loss", value=False, key="trailing_enabled")
if trailing_enabled:
    trail_type = st.sidebar.selectbox("Trailing Type", TRAIL_TYPES, key="trail_type")
    trail_value = st.sidebar.number_input("Trailing Value", 0.0, 1000.0, 1.0, key="trail_value")
else:
    trail_type, trail_value = "Points (Absolute)", 0.0

st.sidebar.markdown("**Costs** (pro traders always model these — they're often why a 'winning' strategy loses money)")
brokerage = st.sidebar.number_input("Brokerage per Round-Trip (₹)", 0.0, 1000.0, 20.0, key="brokerage",
                                     help="⚠️ This is a placeholder, not Dhan's real published rate. "
                                          "Check your broker's actual tariff sheet and enter that instead.")
slippage_pct = st.sidebar.number_input("Slippage (%)", 0.0, 5.0, 0.05, step=0.01, key="slippage_pct")

st.sidebar.subheader("🔴 Dhan Live Trading")
enable_dhan = st.sidebar.checkbox("Enable Dhan Order Placement", value=False)
dhan_client_id = dhan_token = ""
if enable_dhan:
    dhan_client_id = st.sidebar.text_input("Dhan Client ID")
    dhan_token = st.sidebar.text_input("Dhan Access Token", type="password")
    if not DHAN_AVAILABLE:
        st.sidebar.error("Run: pip install dhanhq")

effective_qty = qty * lot_size if segment in ("Futures (FUT)", "Options (CE/PE)") else qty

# =====================================================================================
# MAIN
# =====================================================================================
st.title("📈 Algo Trader Pro")
st.caption(f"{display_name}  •  {segment}"
           + (f" ({option_side})" if option_side else "")
           + f"  •  {timeframe_label}  •  {strategy}")

with st.expander("📚 All 20 Strategies — Rationale"):
    for s in STRATEGIES:
        st.markdown(f"**{s}** — {STRATEGY_NOTES[s]}")

tab_bt, tab_live, tab_opt, tab_hist, tab_heat = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "🧪 Optimization", "🕘 Trade History", "🌡️ Heatmaps"]
)

# -------------------------------------------------------------------------------------
# TAB 1: BACKTEST
# -------------------------------------------------------------------------------------
with tab_bt:
    st.caption("Nothing runs until you click the button below — this avoids hammering the "
               "Yahoo Finance API on every widget interaction.")
    run_clicked = st.button("▶️ Run Backtest", type="primary")

    if run_clicked:
        with st.spinner("Fetching data and running backtest..."):
            df = fetch_data(yf_ticker, interval, period)
            if df.empty:
                st.session_state.backtest_result = None
                st.warning("No data returned. Try a different ticker/timeframe/period "
                           "(intraday intervals have short max lookback on Yahoo Finance).")
            else:
                position = get_position(df, strategy, p)
                atr_series = ATR(df, p["atr_period"])
                trades_df, equity_series = run_backtest(
                    df, position, atr_series, effective_qty,
                    sl_type, sl_value, target_type, target_value,
                    trailing_enabled, trail_type, trail_value,
                    brokerage, slippage_pct, capital)
                metrics = calc_metrics(trades_df, equity_series, capital)
                st.session_state.backtest_result = dict(
                    df=df, trades_df=trades_df, equity_series=equity_series, metrics=metrics,
                    strategy=strategy, ticker=yf_ticker)

    result = st.session_state.backtest_result
    if result is None:
        st.info("Configure the sidebar, then click **Run Backtest** to see results here.")
    else:
        if result["ticker"] != yf_ticker or result["strategy"] != strategy:
            st.warning("⚠️ Sidebar settings have changed since this backtest ran — click "
                       "**Run Backtest** again to refresh the results below.")

        df, trades_df, equity_series, metrics = result["df"], result["trades_df"], result["equity_series"], result["metrics"]

        atr_now = ATR(df, p["atr_period"]).iloc[-1]
        last_px = df["Close"].iloc[-1]
        if sl_type == "Points (Absolute)" and sl_value > 0 and (sl_value / atr_now) < 0.3:
            st.error(f"⚠️ Your SL of {sl_value} points is only ~{sl_value/atr_now:.2f}x this "
                     f"instrument's ATR({p['atr_period']}) of {atr_now:.1f} points "
                     f"(~{atr_now/last_px*100:.2f}% of price). That's inside normal noise for "
                     f"{display_name} on this timeframe — expect near-constant stop-outs regardless "
                     f"of strategy. Switch SL Type to 'ATR Multiple' (try 1.0–1.5x) or widen the points value.")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Trades", metrics["total_trades"])
        c2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        c3.metric("Net P&L (after costs)", f"₹{metrics['total_pnl']:,.0f}")
        c4.metric("CAGR", f"{metrics['cagr']:.1f}%")
        c5.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        c6.metric("Max Drawdown", f"{metrics['max_dd']:.1f}%")

        if metrics["total_trades"] > 0 and metrics["total_pnl"] <= 0:
            st.info("This configuration is net negative after brokerage/slippage. Try the "
                    "**Optimization** tab to search for better parameters, or a different "
                    "SL/Target model (e.g. ATR-based or R-Multiple) — fixed tiny SL% on a "
                    "noisy timeframe is a common reason strategies bleed on costs.")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                      low=df["Low"], close=df["Close"], name="Price"))
        if not trades_df.empty:
            entries_long = trades_df[trades_df["side"] == "LONG"]
            entries_short = trades_df[trades_df["side"] == "SHORT"]
            fig.add_trace(go.Scatter(x=entries_long["entry_date"], y=entries_long["entry_price"],
                                      mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
                                      name="Long Entry"))
            fig.add_trace(go.Scatter(x=entries_short["entry_date"], y=entries_short["entry_price"],
                                      mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
                                      name="Short Entry"))
            fig.add_trace(go.Scatter(x=trades_df["exit_date"], y=trades_df["exit_price"],
                                      mode="markers", marker=dict(symbol="x", color="black", size=8),
                                      name="Exit"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, name="Equity", line=dict(color="royalblue")))
        fig_eq.add_hline(y=capital, line_dash="dash", line_color="gray")
        fig_eq.update_layout(title="Equity Curve", height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Simulated Trade Log")
        if trades_df.empty:
            st.info("No trades were generated for this configuration.")
        else:
            show = trades_df.copy()
            show["pnl"] = show["pnl"].round(2)
            st.dataframe(show, use_container_width=True, height=300)
            st.download_button("Download Trades CSV", show.to_csv(index=False), "backtest_trades.csv", "text/csv")

# -------------------------------------------------------------------------------------
# TAB 2: LIVE TRADING
# -------------------------------------------------------------------------------------
def get_leg_config():
    """Which security_id(s) apply, based on segment/option_side, read from widget state."""
    if segment == "Options (CE/PE)":
        legs = []
        if option_side in ("CE", "Both (Straddle/Strangle)"):
            legs.append(("CE", st.session_state.get("ce_security_id", "")))
        if option_side in ("PE", "Both (Straddle/Strangle)"):
            legs.append(("PE", st.session_state.get("pe_security_id", "")))
        return legs
    return [("EQ", st.session_state.get("single_security_id", ""))]


def square_off_leg(leg, exit_price, ts, reason, exch_seg, product_type):
    txn = "SELL" if leg["side"] == "LONG" else "BUY"
    resp = place_dhan_order(dhan_client_id, dhan_token, leg["security_id"], exch_seg,
                             txn, leg["qty"], product_type, "MARKET", 0)
    log_trade(leg["entry_time"], ts, leg["side"], leg["security_id"], leg["qty"],
              leg["entry_price"], exit_price, reason, resp)
    st.session_state.open_position = [l for l in st.session_state.open_position if l is not leg]
    return resp


def open_leg(label, side, price, ts, security_id, qty_, exch_seg, product_type):
    txn = "BUY" if side == "LONG" else "SELL"
    resp = place_dhan_order(dhan_client_id, dhan_token, security_id, exch_seg,
                             txn, qty_, product_type, "MARKET", 0)
    st.session_state.open_position.append(dict(
        label=label, side=side, entry_price=price, entry_time=ts,
        security_id=security_id, qty=qty_, sl_price=None, target_price=None))
    return resp


with tab_live:
    st.caption("Nothing is fetched or executed until you click a button below — protects "
               "against accidental API hammering and accidental live orders.")

    colr1, colr2 = st.columns([1, 3])
    refresh_clicked = colr1.button("🔄 Refresh Signal")

    live_df = None
    if refresh_clicked or st.session_state.live_running:
        live_df = fetch_data(yf_ticker, interval, period)

    if live_df is not None and not live_df.empty:
        pos_live = get_position(live_df, strategy, p)
        last_pos = int(pos_live.iloc[-1])
        badge = {1: ("🟢 LONG / BUY", "green"), -1: ("🔴 SHORT / SELL", "red"), 0: ("⚪ FLAT / HOLD", "gray")}
        label, color = badge[last_pos]
        st.markdown(f"### :{color}[{label}]")
        st.write(f"Last Close: **{live_df['Close'].iloc[-1]:.2f}**  |  As of: {live_df.index[-1]}  "
                 f"|  (cached up to 5 min to respect API limits)")
    elif refresh_clicked:
        st.warning("No live data available for this ticker/timeframe.")
    else:
        st.info("Click **Refresh Signal** to check the latest signal, or start the auto engine below.")

    st.divider()
    st.subheader("Dhan Order Ticket & Automation")
    if not enable_dhan:
        st.info("Tick **'Enable Dhan Order Placement'** in the sidebar and enter your Client ID / "
                 "Access Token to arm this panel.")
    else:
        exch_seg = st.selectbox("Exchange Segment", EXCHANGE_SEGMENTS,
                                 index=1 if segment in ("Futures (FUT)", "Options (CE/PE)") else 0)
        product_type = SEGMENT_TO_PRODUCT[segment]

        with st.expander("🔍 Look up Dhan Security ID (scrip master)"):
            scrip_df = load_dhan_scrip_master()
            if scrip_df.empty:
                st.error("Could not load Dhan scrip master (network/columns may differ). Enter Security ID manually below.")
            else:
                search = st.text_input("Search symbol (e.g. NIFTY, BANKNIFTY, RELIANCE)")
                if search:
                    cols = [c for c in scrip_df.columns if scrip_df[c].dtype == object]
                    mask = False
                    for c in cols:
                        mask = mask | scrip_df[c].astype(str).str.contains(search, case=False, na=False)
                    st.dataframe(scrip_df[mask].head(30), use_container_width=True, height=250)

        if segment == "Options (CE/PE)":
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("CE Security ID", key="ce_security_id")
            with col2:
                st.text_input("PE Security ID", key="pe_security_id")
        else:
            st.text_input("Security ID", key="single_security_id")

        st.markdown("**Manual Order Buttons**")
        mb1, mb2, mb3 = st.columns(3)
        legs_cfg = get_leg_config()
        if mb1.button("🟢 Manual BUY"):
            ts_now = datetime.now()
            price_now = live_df["Close"].iloc[-1] if live_df is not None and not live_df.empty else 0.0
            for label, sec_id in legs_cfg:
                if sec_id:
                    resp = open_leg(label, "LONG", price_now, ts_now, sec_id, effective_qty, exch_seg, product_type)
                    st.json({label: resp})
        if mb2.button("🔴 Manual SELL"):
            ts_now = datetime.now()
            price_now = live_df["Close"].iloc[-1] if live_df is not None and not live_df.empty else 0.0
            for label, sec_id in legs_cfg:
                if sec_id:
                    resp = open_leg(label, "SHORT", price_now, ts_now, sec_id, effective_qty, exch_seg, product_type)
                    st.json({label: resp})
        if mb3.button("🟥 Manual Square Off (All Open)"):
            ts_now = datetime.now()
            price_now = live_df["Close"].iloc[-1] if live_df is not None and not live_df.empty else 0.0
            for leg in list(st.session_state.open_position):
                resp = square_off_leg(leg, price_now, ts_now, "Manual Square Off", exch_seg, product_type)
                st.json({leg["label"]: resp})
            st.success("All open legs squared off.")

        st.divider()
        st.markdown("**Automation Engine**")
        st.caption("Auto mode reruns this app every N seconds while the tab stays open, checks the "
                   "signal, and executes entries/exits automatically. This is a convenience loop, "
                   "not a hardened production scheduler — for unattended trading, run the same "
                   "logic from a standalone script (cron/systemd/APScheduler).")
        refresh_secs = st.number_input("Auto-check interval (seconds)", 5, 300, 20)

        eng1, eng2 = st.columns(2)
        if not st.session_state.live_running:
            if eng1.button("▶️ Start Auto-Trading Engine", type="primary"):
                st.session_state.live_running = True
                st.rerun()
        else:
            if eng1.button("⏹ Stop Auto-Trading Engine"):
                st.session_state.live_running = False
                st.rerun()
        eng2.write("🟢 **Engine: ARMED**" if st.session_state.live_running else "⚪ Engine: stopped")

        if st.session_state.open_position:
            st.markdown("**Open Position(s)**")
            open_df = pd.DataFrame(st.session_state.open_position)
            st.dataframe(open_df[["label", "side", "entry_price", "qty", "security_id", "entry_time"]],
                         use_container_width=True)

        if st.session_state.live_running:
            if live_df is not None and not live_df.empty:
                ts_now = live_df.index[-1]
                price_now = live_df["Close"].iloc[-1]
                desired_side = last_pos
                open_legs = st.session_state.open_position
                current_side = 1 if (open_legs and open_legs[0]["side"] == "LONG") else (
                    -1 if (open_legs and open_legs[0]["side"] == "SHORT") else 0)

                if desired_side != current_side:
                    for leg in list(open_legs):
                        square_off_leg(leg, price_now, ts_now, "Signal Flip", exch_seg, product_type)
                    if desired_side != 0:
                        for label, sec_id in legs_cfg:
                            if sec_id:
                                want_long = (desired_side == 1 and label in ("EQ", "CE")) or (desired_side == -1 and label == "PE")
                                if segment != "Options (CE/PE)":
                                    open_leg(label, "LONG" if desired_side == 1 else "SHORT",
                                             price_now, ts_now, sec_id, effective_qty, exch_seg, product_type)
                                elif want_long:
                                    open_leg(label, "LONG", price_now, ts_now, sec_id, effective_qty, exch_seg, product_type)
                st.caption(f"Last auto-check: {ts_now}")
            time.sleep(refresh_secs)
            st.rerun()

# -------------------------------------------------------------------------------------
# TAB 3: OPTIMIZATION
# -------------------------------------------------------------------------------------
with tab_opt:
    st.markdown("""
    **What this does:** for the currently selected strategy, pick two of its parameters,
    give each a min/max/step range, and this will re-run a full backtest for every
    combination (a grid search) over your selected ticker/timeframe/period. Results are
    ranked by whichever performance metric you choose, optionally filtered so only
    combinations that hit your minimum win rate are shown.

    ⚠️ **A high win rate is not the same as profitability.** A strategy can win 90% of
    trades and still lose money overall if the losing 10% are large. Check win rate
    *together with* net P&L, Sharpe, and max drawdown before trusting a result.
    """)

    param_options = ["fast", "slow", "rsi_period", "rsi_upper", "rsi_lower", "bb_window", "bb_std",
                      "atr_period", "st_mult", "adx_period", "adx_threshold", "donchian_window",
                      "stoch_k", "keltner_mult", "roc_period", "zscore_window", "zscore_threshold", "cci_period"]

    colA, colB = st.columns(2)
    with colA:
        param_x = st.selectbox("Parameter X (rows)", param_options, index=0)
        x_min = st.number_input(f"{param_x} min", value=5.0)
        x_max = st.number_input(f"{param_x} max", value=25.0)
        x_step = st.number_input(f"{param_x} step", value=5.0, min_value=0.1)
    with colB:
        param_y = st.selectbox("Parameter Y (cols)", param_options, index=1)
        y_min = st.number_input(f"{param_y} min", value=15.0)
        y_max = st.number_input(f"{param_y} max", value=50.0)
        y_step = st.number_input(f"{param_y} step", value=5.0, min_value=0.1)

    metric_choice = st.selectbox("Rank combinations by", ["sharpe", "cagr", "total_return_pct", "win_rate", "profit_factor"])
    min_accuracy = st.slider("Minimum Accuracy / Win Rate Required (%)", 0, 100, 0,
                              help="Set e.g. 90 to only consider combinations whose win rate is at least 90%.")

    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Fetching data and running grid search..."):
            df_opt = fetch_data(yf_ticker, interval, period)
            if df_opt.empty:
                st.session_state.opt_result = None
                st.warning("No data available for optimization.")
            else:
                atr_series_opt = ATR(df_opt, p["atr_period"])
                x_vals = np.arange(x_min, x_max + x_step / 2, x_step)
                y_vals = np.arange(y_min, y_max + y_step / 2, y_step)
                metric_grid = np.zeros((len(x_vals), len(y_vals)))
                winrate_grid = np.zeros((len(x_vals), len(y_vals)))
                rows = []
                progress = st.progress(0.0)
                total_iters = max(len(x_vals) * len(y_vals), 1)
                done = 0
                for i, xv in enumerate(x_vals):
                    for j, yv in enumerate(y_vals):
                        p_test = dict(p)
                        p_test[param_x] = xv
                        p_test[param_y] = yv
                        pos_test = get_position(df_opt, strategy, p_test)
                        tdf, eq = run_backtest(df_opt, pos_test, atr_series_opt, effective_qty,
                                                sl_type, sl_value, target_type, target_value,
                                                trailing_enabled, trail_type, trail_value,
                                                brokerage, slippage_pct, capital)
                        m = calc_metrics(tdf, eq, capital)
                        val = m[metric_choice]
                        val = 0 if not np.isfinite(val) else val
                        metric_grid[i, j] = val
                        winrate_grid[i, j] = m["win_rate"]
                        rows.append({param_x: round(xv, 2), param_y: round(yv, 2),
                                     "trades": m["total_trades"], "win_rate": round(m["win_rate"], 1),
                                     "sharpe": round(m["sharpe"], 2), "cagr": round(m["cagr"], 1),
                                     "total_return_pct": round(m["total_return_pct"], 1),
                                     "profit_factor": round(m["profit_factor"], 2) if np.isfinite(m["profit_factor"]) else None})
                        done += 1
                        progress.progress(done / total_iters)
                progress.empty()

                results_table = pd.DataFrame(rows)
                metric_pivot = pd.DataFrame(metric_grid, index=[round(v, 2) for v in x_vals], columns=[round(v, 2) for v in y_vals])
                winrate_pivot = pd.DataFrame(winrate_grid, index=[round(v, 2) for v in x_vals], columns=[round(v, 2) for v in y_vals])

                meets_threshold = results_table[results_table["win_rate"] >= min_accuracy]
                threshold_met = len(meets_threshold) > 0
                ranked = (meets_threshold if threshold_met else results_table).sort_values(
                    "win_rate" if not threshold_met else metric_choice, ascending=False)

                st.session_state.opt_result = dict(
                    param_x=param_x, param_y=param_y, metric_choice=metric_choice,
                    min_accuracy=min_accuracy, threshold_met=threshold_met,
                    results_table=results_table, ranked=ranked,
                    metric_pivot=metric_pivot, winrate_pivot=winrate_pivot,
                )

    opt = st.session_state.opt_result
    if opt is None:
        st.info("Set your ranges above and click **Run Optimization**.")
    else:
        if not opt["threshold_met"]:
            best_wr = opt["results_table"]["win_rate"].max()
            st.warning(f"No parameter combination reached {opt['min_accuracy']}% win rate. "
                       f"Showing the top combinations by win rate instead (best achieved: {best_wr:.1f}%).")
        else:
            st.success(f"{len(opt['ranked'])} combination(s) met the {opt['min_accuracy']}% accuracy threshold. "
                       f"Showing the top ones ranked by **{opt['metric_choice']}**.")

        top_n = opt["ranked"].head(10).reset_index(drop=True)
        st.dataframe(top_n, use_container_width=True)

        plot_heatmap(opt["winrate_pivot"], f"Win Rate % ({opt['param_x']} vs {opt['param_y']})", colorscale="Blues")
        plot_heatmap(opt["metric_pivot"], f"{opt['metric_choice']} ({opt['param_x']} vs {opt['param_y']})", colorscale="Viridis")

        if len(top_n) > 0:
            best_x_val = top_n.iloc[0][opt["param_x"]]
            best_y_val = top_n.iloc[0][opt["param_y"]]
            st.write(f"Best available combo: **{opt['param_x']} = {best_x_val}**, **{opt['param_y']} = {best_y_val}**")
            if st.button("✅ Apply Best Config to Sidebar"):
                st.session_state[f"p_{opt['param_x']}"] = best_x_val
                st.session_state[f"p_{opt['param_y']}"] = best_y_val
                st.success("Applied — check the sidebar Strategy Parameters, then go rerun the Backtest tab.")
                st.rerun()

# -------------------------------------------------------------------------------------
# TAB 4: TRADE HISTORY (live orders actually placed this session)
# -------------------------------------------------------------------------------------
with tab_hist:
    st.subheader("Live Trade History (this session)")
    st.caption("Every manual or auto-engine order placed via Dhan in this session — separate "
               "from the Backtest tab's simulated trade log.")

    hist = st.session_state.trade_history
    if not hist:
        st.info("No live trades placed yet in this session.")
    else:
        hist_df = pd.DataFrame(hist)
        closed = hist_df[hist_df["exit_price"].notna()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Round-Trips", len(closed))
        if len(closed) > 0:
            win_rate_live = (closed["pnl"] > 0).mean() * 100
            c2.metric("Win Rate", f"{win_rate_live:.1f}%")
            c3.metric("Net P&L", f"₹{closed['pnl'].sum():,.0f}")
        st.dataframe(hist_df, use_container_width=True, height=350)
        st.download_button("Download Trade History CSV", hist_df.to_csv(index=False), "live_trade_history.csv", "text/csv")
        if st.button("🗑️ Clear Trade History"):
            st.session_state.trade_history = []
            st.rerun()

# -------------------------------------------------------------------------------------
# TAB 5: HEATMAPS
# -------------------------------------------------------------------------------------
with tab_heat:
    st.subheader("Monthly Returns Heatmap (Daily Timeframe)")
    n_years = st.slider("Years of history", 3, 20, 10)
    pivot_ret = monthly_returns_heatmap_data(yf_ticker, n_years)
    plot_heatmap(pivot_ret, f"{display_name} — Monthly Returns (%) by Year", colorscale="RdYlGn", zmid=0)

    st.subheader("Daily OHLC Range Heatmap (Volatility by Month/Year)")
    pivot_rng = daily_range_heatmap_data(yf_ticker, n_years)
    plot_heatmap(pivot_rng, f"{display_name} — Avg Daily Range % (High-Low)/Close", colorscale="Oranges")

    st.subheader(f"Heatmap for Selected Timeframe/Period ({timeframe_label}, {period})")
    df_heat = fetch_data(yf_ticker, interval, period)
    if df_heat.empty:
        st.info("No data for this timeframe/period selection.")
    else:
        pivot_sel = selected_timeframe_heatmap(df_heat, interval)
        note = "Hour vs Weekday (avg % return)" if interval in ("1m","5m","15m","30m","60m") else "Month vs Year (sum % return)"
        plot_heatmap(pivot_sel, f"{display_name} — {note}", colorscale="RdYlGn", zmid=0)

st.divider()
st.caption("⚠️ Educational tool only — not investment advice. No strategy shown here is guaranteed "
           "profitable; a high win rate does not equal profitability. Verify Dhan API field names/limits "
           "against current documentation before live trading. Past backtest performance does not "
           "guarantee future results.")
