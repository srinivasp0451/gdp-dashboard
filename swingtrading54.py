# =========================
# PROFESSIONAL ALGO TRADING APP
# STREAMLIT | PYTHON ONLY
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import math
from datetime import datetime
import pytz
from scipy.signal import argrelextrema

# =========================
# CONFIG
# =========================
IST = pytz.timezone("Asia/Kolkata")
RATE_LIMIT = 1.7  # seconds

st.set_page_config(
    page_title="Professional Algo Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# UTILITIES
# =========================
def to_ist(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert(IST)

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def safe_sleep():
    time.sleep(RATE_LIMIT)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(series, period=20):
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std

# =========================
# BASE STRATEGY
# =========================
class BaseStrategy:
    name = "Base Strategy"

    def calculate_indicators(self, df):
        raise NotImplementedError

    def generate_signal(self, df):
        return False, False, {}

    def get_historical_statistics(self, df):
        return {}

# =========================
# EMA CROSSOVER STRATEGY
# =========================
class EMACrossoverStrategy(BaseStrategy):
    name = "EMA / SMA Crossover"

    def __init__(self, fast=9, slow=20, ma_type_fast="EMA", ma_type_slow="EMA",
                 crossover_type="simple", angle_threshold=30, atr_mult=1.5):
        self.fast = fast
        self.slow = slow
        self.ma_type_fast = ma_type_fast
        self.ma_type_slow = ma_type_slow
        self.crossover_type = crossover_type
        self.angle_threshold = angle_threshold
        self.atr_mult = atr_mult

    def _ma(self, series, period, typ):
        return ema(series, period) if typ == "EMA" else sma(series, period)

    def calculate_indicators(self, df):
        df['MA_FAST'] = self._ma(df['Close'], self.fast, self.ma_type_fast)
        df['MA_SLOW'] = self._ma(df['Close'], self.slow, self.ma_type_slow)
        df['ATR'] = atr(df)
        df['BODY'] = abs(df['Close'] - df['Open'])
        return df

    def _angle(self, series):
        slope = series.diff().iloc[-1]
        angle = abs(math.degrees(math.atan(slope)))
        return angle

    def generate_signal(self, df):
        if len(df) < self.slow + 2:
            return False, False, {}

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        bullish = prev.MA_FAST < prev.MA_SLOW and curr.MA_FAST > curr.MA_SLOW
        bearish = prev.MA_FAST > prev.MA_SLOW and curr.MA_FAST < curr.MA_SLOW

        angle = self._angle(df['MA_FAST'])

        strong_candle = True
        if self.crossover_type != "simple":
            avg_body = df['BODY'].rolling(20).mean().iloc[-1]
            if self.crossover_type == "auto_strong_candle":
                strong_candle = curr.BODY > 1.5 * avg_body
            elif self.crossover_type == "atr_strong_candle":
                strong_candle = curr.BODY > self.atr_mult * curr.ATR

        angle_ok = angle >= self.angle_threshold

        bullish &= strong_candle and angle_ok
        bearish &= strong_candle and angle_ok

        return bullish, bearish, {
            "angle": round(angle, 2),
            "ma_fast": curr.MA_FAST,
            "ma_slow": curr.MA_SLOW
        }

# =========================
# Z SCORE MEAN REVERSION
# =========================
class ZScoreStrategy(BaseStrategy):
    name = "Z-Score Mean Reversion"

    def __init__(self, threshold=2.0):
        self.threshold = threshold

    def calculate_indicators(self, df):
        df['Z'] = zscore(df['Close'])
        return df

    def generate_signal(self, df):
        z = df['Z'].iloc[-1]
        return z < -self.threshold, z > self.threshold, {"zscore": round(z, 2)}

# =========================
# RSI DIVERGENCE STRATEGY
# =========================
class RSIDivergenceStrategy(BaseStrategy):
    name = "RSI + Divergence"

    def calculate_indicators(self, df):
        df['RSI'] = rsi(df['Close'])
        return df

    def generate_signal(self, df):
        if len(df) < 20:
            return False, False, {}

        price = df['Close']
        r = df['RSI']

        bullish = price.iloc[-1] < price.iloc[-5] and r.iloc[-1] > r.iloc[-5]
        bearish = price.iloc[-1] > price.iloc[-5] and r.iloc[-1] < r.iloc[-5]

        return bullish, bearish, {"rsi": round(r.iloc[-1], 2)}

# =========================
# FIBONACCI STRATEGY
# =========================
class FibonacciStrategy(BaseStrategy):
    name = "Fibonacci Retracement"

    def __init__(self, tolerance=0.005):
        self.tolerance = tolerance

    def calculate_indicators(self, df):
        high = df['High'].max()
        low = df['Low'].min()
        diff = high - low
        self.levels = {
            "38.2": high - 0.382 * diff,
            "50": high - 0.5 * diff,
            "61.8": high - 0.618 * diff
        }
        return df

    def generate_signal(self, df):
        price = df['Close'].iloc[-1]
        for k, lvl in self.levels.items():
            if abs(price - lvl) / lvl <= self.tolerance:
                return True, False, {"level": k}
        return False, False, {}

# =========================
# ELLIOTT WAVE (SIMPLIFIED BUT REAL)
# =========================
class ElliottWaveStrategy(BaseStrategy):
    name = "Elliott Wave"

    def calculate_indicators(self, df):
        closes = df['Close']
        maxima = argrelextrema(closes.values, np.greater, order=5)[0]
        minima = argrelextrema(closes.values, np.less, order=5)[0]
        df['EXTREMA'] = np.nan
        df.iloc[maxima, df.columns.get_loc('EXTREMA')] = closes.iloc[maxima]
        df.iloc[minima, df.columns.get_loc('EXTREMA')] = closes.iloc[minima]
        return df

    def generate_signal(self, df):
        points = df['EXTREMA'].dropna()
        if len(points) < 5:
            return False, False, {}
        return True, False, {"waves": len(points)}

# =========================
# TRADING SYSTEM
# =========================
class TradingSystem:
    def __init__(self):
        self.position = None

    def enter_trade(self, side, price, qty):
        self.position = {
            "side": side,
            "entry": price,
            "qty": qty,
            "time": datetime.now(IST)
        }

    def exit_trade(self, price, reason):
        trade = self.position.copy()
        trade["exit"] = price
        trade["reason"] = reason
        trade["pnl"] = (price - trade["entry"]) * trade["qty"] * (1 if trade["side"] == "LONG" else -1)
        trade["duration"] = datetime.now(IST) - trade["time"]
        self.position = None
        return trade

# =========================
# SESSION STATE
# =========================
for key, val in {
    "trading_active": False,
    "trade_history": [],
    "trade_log": [],
    "iteration": 0,
    "system": TradingSystem()
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Configuration")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","1d"])
period = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y"])

strategy_name = st.sidebar.selectbox(
    "Strategy",
    [
        "EMA / SMA Crossover",
        "Z-Score Mean Reversion",
        "RSI + Divergence",
        "Fibonacci Retracement",
        "Elliott Wave"
    ]
)

qty = st.sidebar.number_input("Quantity", 1, step=1)

start = st.sidebar.button("▶ Start Trading")
stop = st.sidebar.button("⛔ Stop Trading")

# =========================
# STRATEGY FACTORY
# =========================
if strategy_name == "EMA / SMA Crossover":
    strategy = EMACrossoverStrategy()
elif strategy_name == "Z-Score Mean Reversion":
    strategy = ZScoreStrategy()
elif strategy_name == "RSI + Divergence":
    strategy = RSIDivergenceStrategy()
elif strategy_name == "Fibonacci Retracement":
    strategy = FibonacciStrategy()
else:
    strategy = ElliottWaveStrategy()

# =========================
# MAIN TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📊 Trade History", "🧾 Trade Log"])

# =========================
# LIVE TRADING TAB
# =========================
with tab1:
    st.subheader("Live Trading Engine")

    if start:
        st.session_state.trading_active = True
        st.session_state.trade_log.append("Trading started")

    if stop:
        st.session_state.trading_active = False
        st.session_state.trade_log.append("Trading stopped")

    if st.session_state.trading_active:
        st.info("LIVE — Auto refreshing every 1.7s")

        data = yf.download(ticker, interval=interval, period=period, progress=False)
        safe_sleep()

        if not data.empty:
            data = flatten_columns(data)
            data = to_ist(data)
            data = strategy.calculate_indicators(data)
            bull, bear, info = strategy.generate_signal(data)

            price = data['Close'].iloc[-1]

            if st.session_state.system.position is None:
                if bull:
                    st.session_state.system.enter_trade("LONG", price, qty)
                    st.success("Entered LONG")
                elif bear:
                    st.session_state.system.enter_trade("SHORT", price, qty)
                    st.warning("Entered SHORT")
            else:
                pos = st.session_state.system.position
                pnl = (price - pos["entry"]) * pos["qty"] * (1 if pos["side"]=="LONG" else -1)
                st.metric("P&L", round(pnl,2))

            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.iteration += 1

# =========================
# TRADE HISTORY
# =========================
with tab2:
    st.subheader("Trade History")
    if st.session_state.trade_history:
        df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(df)
    else:
        st.info("No trades yet")

# =========================
# TRADE LOG
# =========================
with tab3:
    st.subheader("System Log")
    st.text_area(
        "Logs",
        "\n".join(st.session_state.trade_log[-100:]),
        height=600
    )
