import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
import pytz
from scipy.signal import argrelextrema
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AlgoTrader Pro | AI-Powered Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .trade-log { font-family: 'Courier New', monospace; font-size: 12px; height: 400px; overflow-y: scroll; background-color: #0e1117; border: 1px solid #333; padding: 10px; }
    .stButton>button { width: 100%; font-weight: bold; }
    .profit { color: #00ff00; }
    .loss { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & INDICATORS
# -----------------------------------------------------------------------------

def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

class Indicators:
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def z_score(series, period=20):
        # Enforce Series to avoid MultiIndex DataFrame error
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std

    @staticmethod
    def get_pivots(data, order=5):
        # Find local peaks and troughs
        highs = data['High'].values
        lows = data['Low'].values
        # order defines the number of points on each side to be smaller
        max_idx = argrelextrema(highs, np.greater, order=order)[0]
        min_idx = argrelextrema(lows, np.less, order=order)[0]
        return max_idx, min_idx

# -----------------------------------------------------------------------------
# 3. STRATEGY ENGINE
# -----------------------------------------------------------------------------

class BaseStrategy:
    def __init__(self, name): self.name = name
    def calculate_indicators(self, data): raise NotImplementedError
    def generate_signal(self, data): raise NotImplementedError

# --- ORIGINAL STRATEGIES ---
class CrossoverStrategy(BaseStrategy):
    def __init__(self, type1='EMA', type2='EMA', p1=9, p2=20):
        super().__init__("MA Crossover")
        self.type1, self.type2, self.p1, self.p2 = type1, type2, int(p1), int(p2)

    def calculate_indicators(self, data):
        c = data['Close']
        data['MA1'] = Indicators.ema(c, self.p1) if self.type1 == 'EMA' else Indicators.sma(c, self.p1)
        data['MA2'] = Indicators.ema(c, self.p2) if self.type2 == 'EMA' else Indicators.sma(c, self.p2)
        return data

    def generate_signal(self, data):
        if len(data) < 2: return False, False, {}
        curr, prev = data.iloc[-1], data.iloc[-2]
        return (prev['MA1'] <= prev['MA2'] and curr['MA1'] > curr['MA2']), \
               (prev['MA1'] >= prev['MA2'] and curr['MA1'] < curr['MA2']), \
               {"MA1": curr['MA1'], "MA2": curr['MA2']}

class ZScoreStrategy(BaseStrategy):
    def __init__(self, period=20, threshold=2.0):
        super().__init__("Mean Reversion (Z-Score)")
        self.period, self.threshold = period, threshold

    def calculate_indicators(self, data):
        # FIX: Ensure we pass a Series, not a DataFrame
        data['Z_Score'] = Indicators.z_score(data['Close'], self.period)
        return data

    def generate_signal(self, data):
        curr = data.iloc[-1]
        z = curr['Z_Score']
        # If NaN (start of data), return False
        if pd.isna(z): return False, False, {}
        return (z < -self.threshold), (z > self.threshold), {"Z-Score": z}

# --- NEW STRATEGIES ---

class ElliottWaveStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Elliott Wave (Heuristic)")

    def calculate_indicators(self, data):
        # We need historical pivots to identify waves
        max_idx, min_idx = Indicators.get_pivots(data, order=3)
        data['is_pivot_high'] = False
        data['is_pivot_low'] = False
        data.iloc[max_idx, data.columns.get_loc('is_pivot_high')] = True
        data.iloc[min_idx, data.columns.get_loc('is_pivot_low')] = True
        return data

    def generate_signal(self, data):
        # Simplified Logic: Looking for Wave 3 completion to trade Wave 5
        # Condition: Recent pivot High (3) > Previous Pivot High (1)
        # Entry: If we are correcting from Wave 3, buy for Wave 5
        
        # Get last 3 pivot highs
        pivots = data[data['is_pivot_high']]
        if len(pivots) < 3: return False, False, {"Info": "Not enough data"}
        
        h3 = pivots.iloc[-1]['High']
        h1 = pivots.iloc[-2]['High']
        
        curr = data.iloc[-1]
        
        # Bullish: We broke above H1 (Wave 3 confirmed) and are now pulling back?
        # A simple trigger: If price is above EMA50 but below recent High (Wave 4 dip)
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        
        bullish = (h3 > h1) and (curr['Close'] > ma50) and (curr['Close'] < h3 * 0.98)
        bearish = (h3 < h1) and (curr['Close'] < ma50) # Downtrend wave
        
        return bullish, bearish, {"Wave3_High": h3, "Wave1_High": h1}

class PsychologyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Market Psychology (Panic/Greed)")

    def calculate_indicators(self, data):
        data['RSI'] = Indicators.rsi(data['Close'], 14)
        data['ATR'] = Indicators.atr(data, 14)
        return data

    def generate_signal(self, data):
        curr = data.iloc[-1]
        
        # Logic: 
        # PANIC BUY: High Volatility (ATR spike) + Oversold RSI (<25) = Fear Peak
        # EUPHORIA SELL: High Volatility + Overbought RSI (>75) = Greed Peak
        
        atr_avg = data['ATR'].rolling(50).mean().iloc[-1]
        high_vol = curr['ATR'] > (atr_avg * 1.2)
        
        bullish = high_vol and (curr['RSI'] < 25)
        bearish = high_vol and (curr['RSI'] > 75)
        
        return bullish, bearish, {"RSI": curr['RSI'], "Vol_Factor": round(curr['ATR']/atr_avg, 2) if atr_avg else 0}

class PriceActionZoneStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Price Action Zones (S/R)")

    def calculate_indicators(self, data):
        # Calculate dynamic Support/Resistance using lookback min/max
        lookback = 50
        data['Res_Zone'] = data['High'].rolling(lookback).max()
        data['Sup_Zone'] = data['Low'].rolling(lookback).min()
        return data

    def generate_signal(self, data):
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Breakout or Reject logic? Let's do REJECTION (Bounce off support)
        # Bullish: Price touched Support Zone recently and is now closing GREEN (up)
        
        dist_to_sup = (curr['Close'] - curr['Sup_Zone']) / curr['Close']
        dist_to_res = (curr['Res_Zone'] - curr['Close']) / curr['Close']
        
        is_green = curr['Close'] > curr['Open']
        is_red = curr['Close'] < curr['Open']
        
        # If within 0.5% of support and candle is Green
        bullish = (dist_to_sup < 0.005) and is_green and (curr['Close'] > prev['High'])
        
        # If within 0.5% of resistance and candle is Red
        bearish = (dist_to_res < 0.005) and is_red and (curr['Close'] < prev['Low'])
        
        return bullish, bearish, {"Sup": curr['Sup_Zone'], "Res": curr['Res_Zone']}

class EmaPullbackStrategy(BaseStrategy):
    def __init__(self, ema_period=50):
        super().__init__("EMA Trend Pullback")
        self.ema_period = ema_period

    def calculate_indicators(self, data):
        data['EMA_Trend'] = Indicators.ema(data['Close'], self.ema_period)
        return data

    def generate_signal(self, data):
        if len(data) < self.ema_period: return False, False, {}
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        ema = curr['EMA_Trend']
        
        # 1. Trend Filter: Price is generally above EMA
        # 2. Pullback: Low touched EMA (or came very close)
        # 3. Trigger: Close > Open (Green Candle) indicating bounce
        
        # Distance from EMA
        dist = (curr['Low'] - ema) / ema
        
        # Bullish: Uptrend (Close > EMA) AND Pullback (Low was near EMA) AND Bounce (Green)
        bullish = (curr['Close'] > ema) and (abs(dist) < 0.003) and (curr['Close'] > curr['Open'])
        
        # Bearish: Downtrend (Close < EMA) AND Retest (High near EMA) AND Reject (Red)
        dist_bear = (curr['High'] - ema) / ema
        bearish = (curr['Close'] < ema) and (abs(dist_bear) < 0.003) and (curr['Close'] < curr['Open'])
        
        return bullish, bearish, {"EMA": ema}

# -----------------------------------------------------------------------------
# 4. DATA MANAGER (FIXED)
# -----------------------------------------------------------------------------

def fetch_data(ticker, timeframe, period):
    time.sleep(1.5) # Rate limit
    try:
        # Request data
        df = yf.download(ticker, period=period, interval=timeframe, progress=False)
        
        if df.empty: return None

        # --- CRITICAL FIX: Flatten MultiIndex Columns ---
        # yfinance often returns columns like ('Close', 'RELIANCE.NS'). 
        # We need to flatten this to just 'Close'.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove 'Ticker' suffix if it leaked into column names
        df.columns = [c.split('.')[0] if isinstance(c, str) else c for c in df.columns]

        # Ensure required columns exist
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in req_cols):
            # Try to recover if columns are weirdly named
            return None

        # Timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# -----------------------------------------------------------------------------
# 5. UI & APP LOGIC
# -----------------------------------------------------------------------------

# Initialize Session State
if 'trading_active' not in st.session_state: st.session_state.trading_active = False
if 'current_position' not in st.session_state: st.session_state.current_position = None
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'log_entries' not in st.session_state: st.session_state.log_entries = []
if 'iteration' not in st.session_state: st.session_state.iteration = 0

def log_msg(msg):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
    st.session_state.log_entries.insert(0, f"[{ts}] {msg}")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Config")
ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "1y"], index=1)

# Strategy Selection
st.sidebar.markdown("### Strategy Select")
strat_map = {
    "EMA Crossover": CrossoverStrategy(),
    "Z-Score Reversion": ZScoreStrategy(),
    "Elliott Wave (Heuristic)": ElliottWaveStrategy(),
    "Psychology (Panic/Greed)": PsychologyStrategy(),
    "Price Action Zones (S/R)": PriceActionZoneStrategy(),
    "EMA Trend Pullback": EmaPullbackStrategy()
}
strat_name = st.sidebar.selectbox("Active Strategy", list(strat_map.keys()))
strategy = strat_map[strat_name]

# Risk
qty = st.sidebar.number_input("Qty", 1, 10000, 1)
sl_val = st.sidebar.number_input("SL Points", 1.0, 500.0, 10.0)
tp_val = st.sidebar.number_input("TP Points", 1.0, 1000.0, 20.0)

# Controls
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ START TRADING", type="primary"):
    st.session_state.trading_active = True
    log_msg("System STARTED")
if c2.button("⏹ STOP"):
    st.session_state.trading_active = False
    log_msg("System STOPPED")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Live", "📜 History", "LOGS"])

with tab1:
    placeholder = st.empty()
    
    if st.session_state.trading_active:
        with placeholder.container():
            st.session_state.iteration += 1
            
            # Fetch & Calc
            df = fetch_data(ticker, timeframe, period)
            if df is not None:
                try:
                    df = strategy.calculate_indicators(df)
                    bullish, bearish, tech_data = strategy.generate_signal(df)
                    
                    curr_price = df['Close'].iloc[-1]
                    
                    # Position Logic
                    pos = st.session_state.current_position
                    if pos:
                        # Check Exit
                        pnl = (curr_price - pos['entry']) * qty if pos['type'] == 'LONG' else (pos['entry'] - curr_price) * qty
                        
                        exit_hit = False
                        reason = ""
                        
                        if pos['type'] == 'LONG':
                            if curr_price <= pos['sl']: exit_hit, reason = True, "SL Hit"
                            elif curr_price >= pos['tp']: exit_hit, reason = True, "TP Hit"
                        else:
                            if curr_price >= pos['sl']: exit_hit, reason = True, "SL Hit"
                            elif curr_price <= pos['tp']: exit_hit, reason = True, "TP Hit"
                            
                        if exit_hit:
                            st.session_state.trade_history.append({
                                'time': get_ist_time(), 'symbol': ticker, 'type': pos['type'],
                                'entry': pos['entry'], 'exit': curr_price, 'pnl': pnl, 'reason': reason
                            })
                            st.session_state.current_position = None
                            log_msg(f"Closed {pos['type']}: {reason} (PnL: {pnl:.2f})")
                            
                    elif not pos:
                        # Check Entry
                        if bullish:
                            st.session_state.current_position = {
                                'type': 'LONG', 'entry': curr_price, 'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                            }
                            log_msg(f"Entered LONG at {curr_price}")
                        elif bearish:
                            st.session_state.current_position = {
                                'type': 'SHORT', 'entry': curr_price, 'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                            }
                            log_msg(f"Entered SHORT at {curr_price}")
                    
                    # UI Display
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"{curr_price:.2f}")
                    if st.session_state.current_position:
                        col2.metric("Position", f"{st.session_state.current_position['type']}")
                        col3.metric("Unrealized PnL", f"{pnl:.2f}" if 'pnl' in locals() else "0.00")
                    else:
                        col2.metric("Status", "Scanning...")
                        col3.metric("Signal", "BULL" if bullish else ("BEAR" if bearish else "WAIT"))

                    # Chart
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                    
                    # Plot specific strategy indicators if available
                    if 'Sup_Zone' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Sup_Zone'], line=dict(color='green', width=1), name='Support'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['Res_Zone'], line=dict(color='red', width=1), name='Resistance'))
                    if 'EMA_Trend' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Trend'], line=dict(color='orange'), name='EMA'))
                    
                    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Strategy Data: {tech_data}")

                except Exception as e:
                    st.error(f"Strategy Error: {e}")
            else:
                st.warning("Fetching data...")
            
            time.sleep(2)
            st.rerun()

with tab2:
    if st.session_state.trade_history:
        st.dataframe(pd.DataFrame(st.session_state.trade_history))
    else:
        st.info("No trades yet.")

with tab3:
    st.markdown(f'<div class="trade-log">{"<br>".join(st.session_state.log_entries)}</div>', unsafe_allow_html=True)
