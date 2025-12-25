import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod

# --- CONFIGURATION & SESSION STATE ---
st.set_page_config(layout="wide", page_title="Pro-Algo Trader IST")

if 'trading_active' not in st.set_page_config:
    st.session_state.update({
        'trading_active': False,
        'current_position': None,
        'trade_history': [],
        'trade_log': [],
        'iteration_count': 0,
        'last_api_call': 0
    })

IST = pytz.timezone('Asia/Kolkata')

# --- UTILITY FUNCTIONS ---
def add_log(message):
    now = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{now}] {message}")
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop()

def get_ist_now():
    return datetime.now(IST)

def rate_limit():
    elapsed = time.time() - st.session_state.last_api_call
    if elapsed < 1.6:
        time.sleep(1.6 - elapsed)
    st.session_state.last_api_call = time.time()

# --- INDICATOR CALCULATIONS (MANUAL) ---
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

# --- STRATEGY ENGINE ---
class BaseStrategy(ABC):
    @abstractmethod
    def calculate_indicators(self, data): pass
    
    @abstractmethod
    def generate_signal(self, data): pass

    def get_historical_statistics(self, data):
        stats = data.describe().T
        return stats[['mean', 'std', 'min', 'max']]

class EMACrossover(BaseStrategy):
    def __init__(self, p1=9, p2=20, type1='EMA', type2='EMA', mode='simple', min_angle=0):
        self.p1, self.p2 = p1, p2
        self.type1, self.type2 = type1, type2
        self.mode = mode
        self.min_angle = min_angle

    def calculate_indicators(self, data):
        df = data.copy()
        df['fast'] = calculate_ema(df['Close'], self.p1) if self.type1 == 'EMA' else calculate_sma(df['Close'], self.p1)
        df['slow'] = calculate_ema(df['Close'], self.p2) if self.type2 == 'EMA' else calculate_sma(df['Close'], self.p2)
        
        # Angle Calculation
        df['fast_slope'] = np.degrees(np.arctan(df['fast'].diff()))
        df['slow_slope'] = np.degrees(np.arctan(df['slow'].diff()))
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        bullish = prev['fast'] <= prev['slow'] and curr['fast'] > curr['slow']
        bearish = prev['fast'] >= prev['slow'] and curr['fast'] < curr['slow']
        
        # Confirmation Logic
        if self.mode == 'auto_strong_candle':
            avg_body = abs(df['Close'] - df['Open']).rolling(20).mean().iloc[-1]
            curr_body = abs(curr['Close'] - curr['Open'])
            if curr_body < 1.5 * avg_body:
                bullish = bearish = False

        if abs(curr['fast_slope']) < self.min_angle:
            bullish = bearish = False

        return bullish, bearish, {"fast": curr['fast'], "slow": curr['slow']}

class ElliottWaveStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        df = data.copy()
        # Use scipy to find local peaks/troughs
        n = 5 # neighborhood
        df['min'] = df.iloc[argrelextrema(df.Low.values, np.less_equal, order=n)[0]]['Low']
        df['max'] = df.iloc[argrelextrema(df.High.values, np.greater_equal, order=n)[0]]['High']
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        # Simplified Wave logic: Check for 4 consecutive swing points
        points = df.dropna(subset=['min', 'max'])
        if len(points) >= 4:
            # Logic to detect Wave 5 entry...
            return False, False, {} 
        return False, False, {}

# --- MAIN UI ---
st.title("🚀 Pro-Algo Trading Suite (IST)")

with st.sidebar:
    st.header("⚙️ Global Settings")
    ticker = st.selectbox("Ticker", ["RELIANCE.NS", "NIFTY_F", "BTC-USD", "EURUSD=X", "GC=F"])
    if ticker == "NIFTY_F": ticker = "^NSEI"
    
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
    period = st.selectbox("Period", ["1d", "5d", "1mo", "1y", "max"])
    
    st.divider()
    st.header("🛡️ Risk Management")
    sl_type = st.selectbox("Stop Loss Type", ["Custom Points", "Trail SL", "ATR Based", "Signal Based"])
    sl_val = st.number_input("SL Value / Multiplier", value=10.0)
    
    tp_type = st.selectbox("Target Type", ["Custom Points", "Risk Reward", "Percentage"])
    tp_val = st.number_input("Target Value", value=20.0)
    
    quantity = st.number_input("Quantity", min_value=1, value=1)

tab_live, tab_hist, tab_log = st.tabs(["📺 Live Trading", "📊 History", "📜 System Log"])

# --- LIVE TRADING LOGIC ---
with tab_live:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Control Panel")
        if not st.session_state.trading_active:
            if st.button("START LIVE TRADING", use_container_width=True, type="primary"):
                st.session_state.trading_active = True
                st.rerun()
        else:
            if st.button("STOP TRADING", use_container_width=True, type="secondary"):
                st.session_state.trading_active = False
                st.rerun()
        
        if st.session_state.trading_active:
            st.success("🟢 LIVE - Monitoring Market")
            st.info(f"Iteration: {st.session_state.iteration_count}")
        else:
            st.warning("⚪ System Idle")

    # --- DATA FETCHING & PROCESSING ---
    rate_limit()
    data = yf.download(ticker, period=period, interval=timeframe, progress=False)
    
    if not data.empty:
        # Handle MultiIndex and Timezone
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = data.index.tz_convert(IST)
        
        # Strategy Initialization (Example: EMA Crossover)
        strat = EMACrossover()
        df_processed = strat.calculate_indicators(data)
        bull, bear, sig_data = strat.generate_signal(df_processed)
        
        curr_price = df_processed['Close'].iloc[-1]
        
        with col1:
            # --- CHARTING ---
            fig = go.Figure(data=[go.Candlestick(x=df_processed.index,
                            open=df_processed['Open'], high=df_processed['High'],
                            low=df_processed['Low'], close=df_processed['Close'], name="Market")])
            
            fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['fast'], line=dict(color='blue', width=1), name="Fast MA"))
            fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['slow'], line=dict(color='orange', width=1), name="Slow MA"))
            
            # Position Management Logic
            if st.session_state.trading_active:
                if st.session_state.current_position is None:
                    if bull:
                        st.session_state.current_position = {
                            'type': 'LONG', 'entry': curr_price, 'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                        }
                        add_log(f"LONG Entry at {curr_price}")
                    elif bear:
                        st.session_state.current_position = {
                            'type': 'SHORT', 'entry': curr_price, 'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                        }
                        add_log(f"SHORT Entry at {curr_price}")
                else:
                    # Check Exit Conditions
                    pos = st.session_state.current_position
                    pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)
                    
                    # Trailing SL Logic
                    if sl_type == "Trail SL" and pos['type'] == "LONG":
                        new_sl = curr_price - sl_val
                        if new_sl > pos['sl']:
                            st.session_state.current_position['sl'] = new_sl
                    
                    # Exit Check
                    exit_triggered = False
                    if pos['type'] == 'LONG':
                        if curr_price <= pos['sl']: exit_triggered, reason = True, "Stop Loss Hit"
                        if curr_price >= pos['tp']: exit_triggered, reason = True, "Target Hit"
                    
                    if exit_triggered:
                        st.session_state.trade_history.append({'pnl': pnl, 'reason': reason})
                        st.session_state.current_position = None
                        add_log(f"Trade Closed: {reason} | P&L: {pnl:.2f}")

            st.plotly_chart(fig, use_container_width=True)

        # --- FRIENDLY ADVICE & DASHBOARD ---
        with col2:
            if st.session_state.current_position:
                pos = st.session_state.current_position
                pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)
                st.metric("P&L (Points)", f"{pnl:.2f}", delta=f"{(pnl/pos['entry'])*100:.2f}%")
                
                st.markdown("### 🗣️ Mentor Advice")
                if pnl > 0:
                    st.write("Trade is moving in your favor. Stay calm and let the trend work. If you're nervous, trail your SL to break-even.")
                else:
                    st.write("Price is testing your entry. This is normal. Stick to your stop loss and don't revenge trade.")
            else:
                st.info("Searching for valid setup...")

# --- LOGS TAB ---
with tab_log:
    st.subheader("System Activity Log (IST)")
    st.text_area("Logs", value="\n".join(st.session_state.trade_log), height=400)

# --- AUTO-REFRESH TRIGGER ---
if st.session_state.trading_active:
    st.session_state.iteration_count += 1
    time.sleep(2)
    st.rerun()
