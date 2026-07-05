"""
=====================================================================================
 ALGO TRADER PRO  —  Single-file Streamlit Algorithmic Trading Workbench
=====================================================================================
Tabs:
    1. Backtest      -> Run any of 20 strategies on Index/Stock/F&O with SL/Target
    2. Live Trading  -> Signal panel + guarded Dhan order placement (CE/PE/Both, FUT, EQ)
    3. Optimization  -> 2D parameter grid-search with Sharpe/Return heatmap
    4. Heatmaps      -> Yearly monthly-returns heatmap, OHLC daily-range heatmap,
                         and a heatmap for the currently selected timeframe/period

Data source for historical bars: yfinance (free, no auth). Dhan is used ONLY for
order execution (guarded behind an explicit checkbox + credentials).

DISCLAIMER: Educational tool. Not investment advice. Verify Dhan scrip-master
column names against the current API version before going live. Trading involves
risk of loss.
=====================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

try:
    from dhanhq import dhanhq
    DHAN_AVAILABLE = True
except Exception:
    DHAN_AVAILABLE = False

st.set_page_config(page_title="Algo Trader Pro", layout="wide", page_icon="📈")

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

# yfinance intraday history limits (approx, enforced by Yahoo)
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
    ep = low[0]
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
    ha_open = [ (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2 ]
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
        long_c = rsi < p["rsi_lower"]
        short_c = rsi > p["rsi_upper"]
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[2]:  # MACD Crossover
        macd_line, signal_line, _ = MACD(close)
        return build_position(macd_line > signal_line, macd_line < signal_line)

    if strategy == STRATEGIES[3]:  # Bollinger Mean Reversion
        upper, mid, lower = BBANDS(close, p["bb_window"], p["bb_std"])
        long_c = close < lower
        short_c = close > upper
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[4]:  # Bollinger Breakout
        upper, mid, lower = BBANDS(close, p["bb_window"], p["bb_std"])
        long_c = close > upper
        short_c = close < lower
        return build_position(long_c, short_c)

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
        long_c = close < vwap * 0.998
        short_c = close > vwap * 1.002
        return build_position(long_c, short_c)

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
        long_c = close > oh.values
        short_c = close < ol.values
        return build_position(pd.Series(long_c, index=df.index), pd.Series(short_c, index=df.index))

    if strategy == STRATEGIES[9]:  # Donchian Breakout
        upper, mid, lower = DONCHIAN(df, p["donchian_window"])
        long_c = close >= upper.shift(1)
        short_c = close <= lower.shift(1)
        return build_position(long_c, short_c)

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
        long_c = close > upper
        short_c = close < lower
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[15]:  # ROC Momentum
        roc = ROC(close, p["roc_period"])
        return build_position(roc > 0, roc < 0)

    if strategy == STRATEGIES[16]:  # Z-score Mean Reversion
        z = ZSCORE(close, p["zscore_window"])
        long_c = z < -p["zscore_threshold"]
        short_c = z > p["zscore_threshold"]
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[17]:  # RSI + MACD Confluence
        rsi = RSI(close, p["rsi_period"])
        macd_line, signal_line, _ = MACD(close)
        long_c = (rsi > 50) & (macd_line > signal_line)
        short_c = (rsi < 50) & (macd_line < signal_line)
        return build_position(long_c, short_c)

    if strategy == STRATEGIES[18]:  # Heikin Ashi Trend
        ha_open, ha_high, ha_low, ha_close = HEIKIN_ASHI(df)
        long_c = ha_close > ha_open
        short_c = ha_close < ha_open
        return build_position(long_c, short_c)

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
# BACKTEST ENGINE
# =====================================================================================
def run_backtest(df, position, qty=1, sl_pct=0.0, target_pct=0.0, capital=100000.0):
    df = df.copy()
    df["position"] = position.reindex(df.index).fillna(0)

    trades = []
    equity_curve = []
    cash = capital
    in_trade = False
    side = 0
    entry_price = 0.0
    entry_date = None
    pos_prev = 0

    for ts, row in df.iterrows():
        price, high, low = row["Close"], row["High"], row["Low"]
        pos = row["position"]

        if in_trade:
            hit, exit_price, reason = False, None, None
            if side == 1:
                if sl_pct > 0 and low <= entry_price * (1 - sl_pct / 100):
                    hit, exit_price, reason = True, entry_price * (1 - sl_pct / 100), "SL"
                elif target_pct > 0 and high >= entry_price * (1 + target_pct / 100):
                    hit, exit_price, reason = True, entry_price * (1 + target_pct / 100), "Target"
            else:
                if sl_pct > 0 and high >= entry_price * (1 + sl_pct / 100):
                    hit, exit_price, reason = True, entry_price * (1 + sl_pct / 100), "SL"
                elif target_pct > 0 and low <= entry_price * (1 - target_pct / 100):
                    hit, exit_price, reason = True, entry_price * (1 - target_pct / 100), "Target"
            if hit:
                pnl = (exit_price - entry_price) * qty * side
                cash += pnl
                trades.append(dict(entry_date=entry_date, exit_date=ts, side="LONG" if side == 1 else "SHORT",
                                    entry_price=entry_price, exit_price=exit_price, qty=qty, pnl=pnl, reason=reason))
                in_trade = False

        if pos != pos_prev:
            if in_trade:
                exit_price = price
                pnl = (exit_price - entry_price) * qty * side
                cash += pnl
                trades.append(dict(entry_date=entry_date, exit_date=ts, side="LONG" if side == 1 else "SHORT",
                                    entry_price=entry_price, exit_price=exit_price, qty=qty, pnl=pnl, reason="Signal Flip"))
                in_trade = False
            if pos != 0:
                in_trade = True
                side = int(pos)
                entry_price = price
                entry_date = ts

        unrealized = (price - entry_price) * qty * side if in_trade else 0
        equity_curve.append(cash + unrealized)
        pos_prev = pos

    if in_trade:
        last_ts = df.index[-1]
        last_price = df["Close"].iloc[-1]
        pnl = (last_price - entry_price) * qty * side
        cash += pnl
        trades.append(dict(entry_date=entry_date, exit_date=last_ts, side="LONG" if side == 1 else "SHORT",
                            entry_price=entry_price, exit_price=last_price, qty=qty, pnl=pnl, reason="End of Data"))

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
    """Heatmap of returns for the currently loaded timeframe/period."""
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


def plot_heatmap(pivot, title, colorscale="RdYlGn", fmt=".2f"):
    if pivot is None or pivot.empty:
        st.info("Not enough data to build this heatmap.")
        return
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
        colorscale=colorscale, zmid=0 if colorscale == "RdYlGn" else None,
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
    p = {}
    p["fast"] = st.number_input("Fast Period", 2, 100, 9)
    p["slow"] = st.number_input("Slow Period", 5, 300, 21)
    p["rsi_period"] = st.number_input("RSI Period", 2, 50, 14)
    p["rsi_upper"] = st.number_input("RSI Overbought", 50, 95, 70)
    p["rsi_lower"] = st.number_input("RSI Oversold", 5, 50, 30)
    p["bb_window"] = st.number_input("Bollinger/Keltner Window", 5, 100, 20)
    p["bb_std"] = st.number_input("Bollinger Std Dev", 1.0, 4.0, 2.0)
    p["atr_period"] = st.number_input("ATR Period", 5, 50, 14)
    p["st_mult"] = st.number_input("Supertrend Multiplier", 1.0, 6.0, 3.0)
    p["adx_period"] = st.number_input("ADX Period", 5, 50, 14)
    p["adx_threshold"] = st.number_input("ADX Threshold", 10, 50, 25)
    p["orb_minutes"] = st.number_input("ORB Minutes", 5, 60, 15)
    p["donchian_window"] = st.number_input("Donchian Window", 5, 100, 20)
    p["stoch_k"] = st.number_input("Stochastic %K", 5, 30, 14)
    p["stoch_d"] = st.number_input("Stochastic %D", 2, 10, 3)
    p["keltner_mult"] = st.number_input("Keltner Multiplier", 1.0, 4.0, 2.0)
    p["roc_period"] = st.number_input("ROC Period", 5, 50, 12)
    p["zscore_window"] = st.number_input("Z-Score Window", 5, 100, 20)
    p["zscore_threshold"] = st.number_input("Z-Score Threshold", 1.0, 4.0, 2.0)
    p["cci_period"] = st.number_input("CCI Period", 5, 50, 20)
    p["_bar_minutes"] = bar_minutes_from_interval(interval)

st.sidebar.subheader("💰 Trade Setup")
qty = st.sidebar.number_input("Quantity / Lots", 1, 100000, 1)
lot_size = st.sidebar.number_input("Lot Size (F&O)", 1, 10000, 25)
capital = st.sidebar.number_input("Capital (₹)", 1000, 100_000_000, 100000, step=5000)
sl_pct = st.sidebar.number_input("Stop Loss (%)", 0.0, 50.0, 1.0, step=0.1)
target_pct = st.sidebar.number_input("Target (%)", 0.0, 100.0, 2.0, step=0.1)

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

tab_bt, tab_live, tab_opt, tab_heat = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "🧪 Optimization", "🌡️ Heatmaps"]
)

# -------------------------------------------------------------------------------------
# TAB 1: BACKTEST
# -------------------------------------------------------------------------------------
with tab_bt:
    df = fetch_data(yf_ticker, interval, period)
    if df.empty:
        st.warning("No data returned. Try a different ticker/timeframe/period "
                   "(intraday intervals have short max lookback on Yahoo Finance).")
    else:
        position = get_position(df, strategy, p)
        trades_df, equity_series = run_backtest(df, position, effective_qty, sl_pct, target_pct, capital)
        metrics = calc_metrics(trades_df, equity_series, capital)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Trades", metrics["total_trades"])
        c2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        c3.metric("Net P&L", f"₹{metrics['total_pnl']:,.0f}")
        c4.metric("CAGR", f"{metrics['cagr']:.1f}%")
        c5.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        c6.metric("Max Drawdown", f"{metrics['max_dd']:.1f}%")

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

        st.subheader("Trade Log")
        if trades_df.empty:
            st.info("No trades were generated for this configuration.")
        else:
            show = trades_df.copy()
            show["pnl"] = show["pnl"].round(2)
            st.dataframe(show, use_container_width=True, height=300)
            st.download_button("Download Trades CSV", show.to_csv(index=False), "trades.csv", "text/csv")

# -------------------------------------------------------------------------------------
# TAB 2: LIVE TRADING
# -------------------------------------------------------------------------------------
with tab_live:
    st.subheader("Current Signal")
    df_live = fetch_data(yf_ticker, interval, period)
    if df_live.empty:
        st.warning("No live data available.")
    else:
        pos_live = get_position(df_live, strategy, p)
        last_pos = int(pos_live.iloc[-1])
        badge = {1: ("🟢 LONG / BUY", "green"), -1: ("🔴 SHORT / SELL", "red"), 0: ("⚪ FLAT / HOLD", "gray")}
        label, color = badge[last_pos]
        st.markdown(f"### :{color}[{label}]")
        st.write(f"Last Close: **{df_live['Close'].iloc[-1]:.2f}**  |  As of: {df_live.index[-1]}")

        st.divider()
        st.subheader("Dhan Order Ticket")
        if not enable_dhan:
            st.info("Tick **'Enable Dhan Order Placement'** in the sidebar and enter your Client ID / Access Token to arm this panel.")
        else:
            exch_seg = st.selectbox("Exchange Segment", EXCHANGE_SEGMENTS,
                                     index=1 if segment in ("Futures (FUT)", "Options (CE/PE)") else 0)
            product_type = SEGMENT_TO_PRODUCT[segment]
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
            limit_price = st.number_input("Limit Price (if LIMIT)", 0.0, 1_000_000.0, 0.0) if order_type == "LIMIT" else 0

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
                    ce_security_id = st.text_input("CE Security ID")
                with col2:
                    pe_security_id = st.text_input("PE Security ID")

                txn_type = st.radio("Transaction Type", ["BUY", "SELL"], horizontal=True)
                b1, b2, b3 = st.columns(3)
                if option_side in ("CE", "Both (Straddle/Strangle)") and b1.button("Place CE Order"):
                    resp = place_dhan_order(dhan_client_id, dhan_token, ce_security_id, exch_seg,
                                             txn_type, effective_qty, product_type, order_type, limit_price)
                    st.json(resp)
                if option_side in ("PE", "Both (Straddle/Strangle)") and b2.button("Place PE Order"):
                    resp = place_dhan_order(dhan_client_id, dhan_token, pe_security_id, exch_seg,
                                             txn_type, effective_qty, product_type, order_type, limit_price)
                    st.json(resp)
                if option_side == "Both (Straddle/Strangle)" and b3.button("Place BOTH Legs Simultaneously"):
                    resp_ce = place_dhan_order(dhan_client_id, dhan_token, ce_security_id, exch_seg,
                                                txn_type, effective_qty, product_type, order_type, limit_price)
                    resp_pe = place_dhan_order(dhan_client_id, dhan_token, pe_security_id, exch_seg,
                                                txn_type, effective_qty, product_type, order_type, limit_price)
                    st.json({"CE": resp_ce, "PE": resp_pe})
            else:
                security_id = st.text_input("Security ID")
                colb, cols = st.columns(2)
                if colb.button("🟢 Place BUY Order"):
                    resp = place_dhan_order(dhan_client_id, dhan_token, security_id, exch_seg,
                                             "BUY", effective_qty, product_type, order_type, limit_price)
                    st.json(resp)
                if cols.button("🔴 Place SELL Order"):
                    resp = place_dhan_order(dhan_client_id, dhan_token, security_id, exch_seg,
                                             "SELL", effective_qty, product_type, order_type, limit_price)
                    st.json(resp)

# -------------------------------------------------------------------------------------
# TAB 3: OPTIMIZATION
# -------------------------------------------------------------------------------------
with tab_opt:
    st.subheader("2D Parameter Grid Search")
    df_opt = fetch_data(yf_ticker, interval, period)
    if df_opt.empty:
        st.warning("No data available for optimization.")
    else:
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
        metric_choice = st.selectbox("Optimize for", ["sharpe", "cagr", "total_return_pct", "win_rate"])

        if st.button("🚀 Run Optimization"):
            x_vals = np.arange(x_min, x_max + x_step / 2, x_step)
            y_vals = np.arange(y_min, y_max + y_step / 2, y_step)
            results = np.zeros((len(x_vals), len(y_vals)))
            best = (-np.inf, None, None)
            progress = st.progress(0.0)
            total_iters = len(x_vals) * len(y_vals)
            done = 0
            for i, xv in enumerate(x_vals):
                for j, yv in enumerate(y_vals):
                    p_test = dict(p)
                    p_test[param_x] = xv
                    p_test[param_y] = yv
                    pos_test = get_position(df_opt, strategy, p_test)
                    tdf, eq = run_backtest(df_opt, pos_test, effective_qty, sl_pct, target_pct, capital)
                    m = calc_metrics(tdf, eq, capital)
                    val = m[metric_choice]
                    results[i, j] = 0 if not np.isfinite(val) else val
                    if val > best[0]:
                        best = (val, xv, yv)
                    done += 1
                    progress.progress(done / total_iters)
            progress.empty()

            pivot = pd.DataFrame(results, index=[round(v, 2) for v in x_vals], columns=[round(v, 2) for v in y_vals])
            plot_heatmap(pivot, f"{metric_choice} heatmap ({param_x} vs {param_y})", colorscale="Viridis")
            st.success(f"Best {metric_choice}: **{best[0]:.2f}** at {param_x} = {best[1]}, {param_y} = {best[2]}")

# -------------------------------------------------------------------------------------
# TAB 4: HEATMAPS
# -------------------------------------------------------------------------------------
with tab_heat:
    st.subheader("Monthly Returns Heatmap (Daily Timeframe)")
    n_years = st.slider("Years of history", 3, 20, 10)
    pivot_ret = monthly_returns_heatmap_data(yf_ticker, n_years)
    plot_heatmap(pivot_ret, f"{display_name} — Monthly Returns (%) by Year", colorscale="RdYlGn")

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
        plot_heatmap(pivot_sel, f"{display_name} — {note}", colorscale="RdYlGn")

st.divider()
st.caption("⚠️ Educational tool only — not investment advice. Verify Dhan API field names/limits "
           "against current documentation before live trading. Past backtest performance does not "
           "guarantee future results.")
