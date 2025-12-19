import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pytz
from datetime import datetime
from scipy.signal import argrelextrema
import math

# ==========================================
# 1. CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(layout="wide", page_title="Pro Algo-Trader AI", page_icon="📈")

# Sidebar Theme Selection
st.sidebar.header("⚙️ Appearance")
theme_mode = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)

# Dynamic CSS Injection
if theme_mode == "Dark":
    bg_color = "#0e1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    plot_bg = "#0e1117"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    card_bg = "#F0F2F6"
    plot_bg = "#FFFFFF"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .metric-card {{ background-color: {card_bg}; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; color: {text_color}; }}
    h1, h2, h3, h4, h5, h6, p, label {{ color: {text_color} !important; }}
    .stMetricValue {{ color: {text_color} !important; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & DATA
# ==========================================
ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "TSLA", "AAPL"]
}

# TIME CONFIG
TIMEFRAMES = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

# ==========================================
# 3. SESSION STATE
# ==========================================
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.trading_active = False
    st.session_state.position = None
    st.session_state.trade_history = []
    st.session_state.logs = []
    st.session_state.iteration = 0
    st.session_state.last_api_call = datetime.now()

def log_msg(message, type="INFO"):
    tz = pytz.timezone('Asia/Kolkata')
    ts = datetime.now(tz).strftime('%H:%M:%S')
    st.session_state.logs.insert(0, f"[{ts}] [{type}] {message}")
    if len(st.session_state.logs) > 100: st.session_state.logs.pop()

# ==========================================
# 4. DATA ENGINE (Manual Indicators)
# ==========================================
class DataEngine:
    @staticmethod
    def fetch_data(ticker, timeframe, period):
        # Rate Limit
        now = datetime.now()
        delta = (now - st.session_state.last_api_call).total_seconds()
        if delta < 1.5: time.sleep(1.5 - delta)
        st.session_state.last_api_call = datetime.now()
        
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            
            # Handle MultiIndex Columns (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if df.empty: return None
            
            # Timezone
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            return df.dropna()
        except Exception as e:
            log_msg(f"API Error: {str(e)}", "ERROR")
            return None

    @staticmethod
    def calculate_indicators(df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # Helpers
        def sma(data, n): return pd.Series(data).rolling(n).mean().values
        def ema(data, n): return pd.Series(data).ewm(span=n, adjust=False).mean().values
        def rsi(data, n=14):
            delta = np.diff(data, prepend=data[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(alpha=1/n, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1/n, adjust=False).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))
        def atr(high, low, close, n=14):
            tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
            tr[0] = high[0]-low[0]
            return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values

        # Calc
        df['SMA_20'] = sma(close, 20)
        df['EMA_9'] = ema(close, 9)
        df['EMA_20'] = ema(close, 20)
        df['RSI'] = rsi(close)
        df['ATR'] = atr(high, low, close)
        
        return df

# ==========================================
# 5. STRATEGY ENGINE
# ==========================================
class BaseStrategy:
    def __init__(self, params): self.params = params
    def get_signal(self, df): return False, False, {}

class EMACrossoverStrategy(BaseStrategy):
    def get_signal(self, df):
        if len(df) < 50: return False, False, {}
        fast = pd.Series(df['Close']).ewm(span=self.params.get('p1', 9), adjust=False).mean().values
        slow = pd.Series(df['Close']).ewm(span=self.params.get('p2', 20), adjust=False).mean().values
        
        cross_up = (fast[-2] <= slow[-2]) and (fast[-1] > slow[-1])
        cross_down = (fast[-2] >= slow[-2]) and (fast[-1] < slow[-1])
        
        # Confirmation
        confirmed = True
        if self.params.get('type') == 'Strong Candle':
            body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
            avg_body = np.mean(np.abs(df['Close'].values[-10:] - df['Open'].values[-10:]))
            if body < 1.5 * avg_body: confirmed = False
            
        return (cross_up and confirmed), (cross_down and confirmed), {"Fast": fast[-1], "Slow": slow[-1]}

class ElliottWaveStrategy(BaseStrategy):
    def get_signal(self, df):
        if len(df) < 60: return False, False, {}
        # Uses argrelextrema to find pivots
        highs = df['High'].values
        lows = df['Low'].values
        
        # Order 5 means it checks 5 candles on each side
        max_idx = argrelextrema(highs, np.greater, order=5)[0]
        min_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(max_idx) < 3 or len(min_idx) < 3: return False, False, {"Status": "Building Wave Data"}
        
        # Logic: Detect Impulse (Wave 3 strongest) -> Correction -> Wave 5 Entry
        last_high_idx = max_idx[-1]
        
        # Simple Divergence check at potential Wave 5 top
        # Price Higher High but RSI Lower High
        price_hh = highs[max_idx[-1]] > highs[max_idx[-2]]
        rsi_vals = df['RSI'].values
        rsi_lh = rsi_vals[max_idx[-1]] < rsi_vals[max_idx[-2]]
        
        sell_signal = price_hh and rsi_lh and rsi_vals[max_idx[-1]] > 60
        buy_signal = False # Implementing simple W5 top sell for now
        
        return buy_signal, sell_signal, {
            "Wave Status": "Potential W5 Top" if sell_signal else "Monitoring",
            "Divergence": "Bearish" if (price_hh and rsi_lh) else "None"
        }

class FibonacciStrategy(BaseStrategy):
    def get_signal(self, df):
        # Lookback 100 candles for Swing High/Low
        window = df.iloc[-100:]
        high_val = window['High'].max()
        low_val = window['Low'].min()
        diff = high_val - low_val
        
        curr = df['Close'].iloc[-1]
        
        # Levels
        fib_618 = high_val - (0.618 * diff) # Retracement level in uptrend context
        fib_500 = high_val - (0.5 * diff)
        
        # Check if price is near 61.8% (Golden Pocket) with 0.5% tolerance
        tol = curr * 0.005
        on_support = abs(curr - fib_618) < tol
        
        # Signal: Price near 61.8 + Green Candle
        is_green = df['Close'].iloc[-1] > df['Open'].iloc[-1]
        buy = on_support and is_green
        
        return buy, False, {"Fib 61.8%": fib_618, "Dist": abs(curr-fib_618)}

class RSIDivergenceStrategy(BaseStrategy):
    def get_signal(self, df):
        if len(df) < 30: return False, False, {}
        
        # Find pivots
        lows = df['Low'].values
        rsi = df['RSI'].values
        min_idx = argrelextrema(lows, np.less, order=5)[0]
        max_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
        
        if len(min_idx) < 2: return False, False, {}
        
        # Bullish Div: Lower Low in Price, Higher Low in RSI
        last_p = min_idx[-1]
        prev_p = min_idx[-2]
        
        price_ll = lows[last_p] < lows[prev_p]
        rsi_hl = rsi[last_p] > rsi[prev_p]
        
        buy = price_ll and rsi_hl and rsi[last_p] < 40
        
        # Bearish Div
        if len(max_idx) < 2: return buy, False, {}
        last_peak = max_idx[-1]
        prev_peak = max_idx[-2]
        
        price_hh = df['High'].values[last_peak] > df['High'].values[prev_peak]
        rsi_lh = rsi[last_peak] < rsi[prev_peak]
        
        sell = price_hh and rsi_lh and rsi[last_peak] > 60
        
        return buy, sell, {"Div Type": "Bullish" if buy else "Bearish" if sell else "None"}

class SimpleBuySellStrategy(BaseStrategy):
    def get_signal(self, df):
        mode = self.params.get('mode', 'buy')
        return (mode=='buy'), (mode=='sell'), {"Action": "Manual " + mode.upper()}

# ==========================================
# 6. TRADING SYSTEM
# ==========================================
class TradingSystem:
    def __init__(self):
        self.strategies = {
            "EMA Crossover": EMACrossoverStrategy,
            "Elliott Wave": ElliottWaveStrategy,
            "Fibonacci Retracement": FibonacciStrategy,
            "RSI Divergence": RSIDivergenceStrategy,
            "Simple Buy": SimpleBuySellStrategy,
            "Simple Sell": SimpleBuySellStrategy
        }

    def run(self, ticker, tf, per, strat_name, strat_params, risk_params):
        st.session_state.iteration += 1
        
        df = DataEngine.fetch_data(ticker, tf, per)
        if df is None:
            st.warning("Waiting for data...")
            return

        df = DataEngine.calculate_indicators(df)
        curr_price = df['Close'].iloc[-1]
        
        # Strategy Execution
        if strat_name == "Simple Buy": strat_params['mode'] = 'buy'
        elif strat_name == "Simple Sell": strat_params['mode'] = 'sell'
        
        strat_cls = self.strategies.get(strat_name, BaseStrategy)
        strat = strat_cls(strat_params)
        buy_sig, sell_sig, debug_data = strat.get_signal(df)
        
        # Manage Position
        self.manage_position(curr_price, buy_sig, sell_sig, risk_params)
        
        # Render UI
        self.render_ui(df, debug_data, risk_params)

    def manage_position(self, price, buy, sell, risk):
        pos = st.session_state.position
        
        # Check Exits if In Position
        if pos:
            pnl = (price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - price)
            
            reason = None
            if pos['type'] == 'LONG':
                if price <= pos['sl']: reason = "SL Hit"
                elif price >= pos['tgt']: reason = "Target Hit"
            else:
                if price >= pos['sl']: reason = "SL Hit"
                elif price <= pos['tgt']: reason = "Target Hit"
            
            if reason:
                self.close_pos(price, pnl, reason)
                
        # Check Entries if No Position
        elif not pos:
            if buy: self.open_pos("LONG", price, risk)
            elif sell: self.open_pos("SHORT", price, risk)

    def open_pos(self, type, price, risk):
        sl_dist = risk['sl_pts']
        tgt_dist = risk['tgt_pts']
        
        sl = price - sl_dist if type == 'LONG' else price + sl_dist
        tgt = price + tgt_dist if type == 'LONG' else price - tgt_dist
        
        st.session_state.position = {
            "type": type, "entry": price, "sl": sl, "tgt": tgt, "qty": risk['qty']
        }
        log_msg(f"OPEN {type} @ {price:.2f}", "TRADE")

    def close_pos(self, price, pnl, reason):
        log_msg(f"CLOSE @ {price:.2f} | PnL: {pnl:.2f} | {reason}", "TRADE")
        st.session_state.trade_history.append({
            "Time": datetime.now().strftime('%H:%M'),
            "PnL": pnl,
            "Reason": reason
        })
        st.session_state.position = None

    def render_ui(self, df, debug_data, risk):
        curr = df.iloc[-1]
        
        # Dashboard
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{curr['Close']:.2f}", f"{(curr['Close']-df.iloc[-2]['Close']):.2f}")
        c2.metric("RSI", f"{curr['RSI']:.1f}")
        c3.metric("ATR", f"{curr['ATR']:.2f}")
        c4.metric("Status", "LIVE" if st.session_state.trading_active else "STOPPED")

        # FIX: The Error `StreamlitInvalidColumnSpecError` happened here.
        # We must check if debug_data has items before creating columns.
        with st.expander("Strategy Debug", expanded=True):
            if debug_data and len(debug_data) > 0:
                d_cols = st.columns(len(debug_data))
                for i, (k, v) in enumerate(debug_data.items()):
                    d_cols[i].metric(str(k), str(v)[:10]) # Truncate long values
            else:
                st.info("No strategy signals generated yet.")

        # Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange'), name='EMA 20'), row=1, col=1)
        
        # Draw Position Lines
        if st.session_state.position:
            pos = st.session_state.position
            fig.add_hline(y=pos['entry'], line_color="blue", row=1, col=1)
            fig.add_hline(y=pos['sl'], line_color="red", line_dash="dash", row=1, col=1)
            fig.add_hline(y=pos['tgt'], line_color="green", line_dash="dash", row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1, line_dash="dot")
        fig.add_hline(y=30, row=2, col=1, line_dash="dot")
        
        # Apply Theme Colors to Chart
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), 
                         paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, 
                         font=dict(color=text_color))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. MAIN UI
# ==========================================
st.sidebar.header("Asset Setup")
ticker = st.sidebar.selectbox("Asset", ASSETS['Indices'] + ASSETS['Stocks'])
# FIX: Default Timeframe=1m, Period=1d
tf = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=0) # 1m is index 0
per = st.sidebar.selectbox("Period", PERIODS, index=0)      # 1d is index 0

st.sidebar.header("Strategy Setup")
strat_name = st.sidebar.selectbox("Strategy", [
    "EMA Crossover", "Elliott Wave", "Fibonacci Retracement", 
    "RSI Divergence", "Simple Buy", "Simple Sell"
])

# Strategy Specific Params
s_params = {}
if strat_name == "EMA Crossover":
    s_params['p1'] = st.sidebar.number_input("Fast Period", 5, 50, 9)
    s_params['p2'] = st.sidebar.number_input("Slow Period", 10, 200, 20)
    s_params['type'] = st.sidebar.selectbox("Type", ["Simple", "Strong Candle"])

# Risk Setup
st.sidebar.header("Risk Management")
qty = st.sidebar.number_input("Qty", 1, 1000, 1)
sl_pts = st.sidebar.number_input("Stop Loss (Pts)", 1.0, 500.0, 10.0)
tgt_pts = st.sidebar.number_input("Target (Pts)", 1.0, 1000.0, 20.0)

risk_params = {'qty': qty, 'sl_pts': sl_pts, 'tgt_pts': tgt_pts}

# Controls
c1, c2 = st.sidebar.columns(2)
if c1.button("START"): st.session_state.trading_active = True
if c2.button("STOP"): st.session_state.trading_active = False

# Execution
if st.session_state.trading_active:
    sys = TradingSystem()
    sys.run(ticker, tf, per, strat_name, s_params, risk_params)
    time.sleep(1.5)
    st.rerun()
else:
    st.info("System Stopped. Press START to begin.")
