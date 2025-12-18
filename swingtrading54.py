import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import pytz
from scipy.signal import argrelextrema

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & UI SETUP
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ProAlgo Trader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Theme and Professional UI
st.markdown("""
<style>
    /* Global Light Theme tweaks */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* Success/Error text colors */
    .profit-text { color: #008000; font-weight: bold; }
    .loss-text { color: #d32f2f; font-weight: bold; }
    .neutral-text { color: #555555; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Mentor Box */
    .mentor-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #0d47a1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS & SESSION STATE
# -----------------------------------------------------------------------------

IST = pytz.timezone('Asia/Kolkata')

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.trading_active = False
    st.session_state.current_position = None  # {type, entry_price, quantity, sl, target, time, strategy}
    st.session_state.trade_history = []
    st.session_state.trade_log = []
    st.session_state.iteration_count = 0
    st.session_state.last_api_call = 0

def log_event(message):
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{timestamp}] {message}")
    # Keep log manageable
    if len(st.session_state.trade_log) > 100:
        st.session_state.trade_log.pop()

def safe_yf_download(ticker, period, interval):
    """
    Fetches data with rate limiting and handling MultiIndex.
    Enforces 2 second delay between calls globally if needed, 
    though we handle delay in the loop mostly.
    """
    # Rate limit check
    now = time.time()
    diff = now - st.session_state.last_api_call
    if diff < 2.0:
        time.sleep(2.0 - diff)
    
    try:
        # Fetch slightly more data to ensure indicator stability
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        st.session_state.last_api_call = time.time()
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            return None
            
        # Rename for consistency
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # Ensure IST index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        return df
    except Exception as e:
        log_event(f"API Error: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# 3. INDICATOR LIBRARY (Manual Calculation)
# -----------------------------------------------------------------------------

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
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    @staticmethod
    def z_score_price(series, period=20):
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std

    @staticmethod
    def find_pivots(series, order=5):
        # Local Maxima
        high_idx = argrelextrema(series.values, np.greater, order=order)[0]
        # Local Minima
        low_idx = argrelextrema(series.values, np.less, order=order)[0]
        return high_idx, low_idx

# -----------------------------------------------------------------------------
# 4. STRATEGY ENGINE
# -----------------------------------------------------------------------------

class BaseStrategy:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.indicators = {}

    def calculate(self, df):
        """Calculates indicators and adds them to df/self.indicators"""
        pass

    def check_signal(self, df):
        """Returns (bullish: bool, bearish: bool, extra_data: dict)"""
        return False, False, {}

    def get_guidance(self, df, position):
        """Returns mentor text based on current state"""
        return "Analysis in progress..."

# --- Strategy Implementations ---

class EMACrossoverStrategy(BaseStrategy):
    def calculate(self, df):
        p1 = int(self.params.get('p1', 9))
        p2 = int(self.params.get('p2', 20))
        df['ema_fast'] = Indicators.ema(df['close'], p1)
        df['ema_slow'] = Indicators.ema(df['close'], p2)
        
        # Strong Candle Logic
        body = (df['close'] - df['open']).abs()
        df['is_strong'] = body > (df['high'] - df['low']) * 0.6 # Body is 60% of range

    def check_signal(self, df):
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        bullish = (prev['ema_fast'] <= prev['ema_slow']) and (curr['ema_fast'] > curr['ema_slow'])
        bearish = (prev['ema_fast'] >= prev['ema_slow']) and (curr['ema_fast'] < curr['ema_slow'])
        
        if self.params.get('require_strong_candle', False):
            bullish = bullish and (curr['close'] > curr['open']) and curr['is_strong']
            bearish = bearish and (curr['close'] < curr['open']) and curr['is_strong']

        return bullish, bearish, {"ema_fast": curr['ema_fast'], "ema_slow": curr['ema_slow']}

    def get_guidance(self, df, position):
        curr = df.iloc[-1]
        gap = curr['ema_fast'] - curr['ema_slow']
        if position:
            if position['type'] == 'LONG':
                return f"Trend is holding. EMA Gap is {gap:.2f}. If gap narrows significantly, consider tightening SL."
            else:
                return f"Bearish trend active. EMA Gap is {abs(gap):.2f}. Watch for price crossing above fast EMA."
        return f"Waiting for crossover. Current gap: {gap:.2f}. Trend is {'Bullish' if gap > 0 else 'Bearish'}."

class RSIStrategy(BaseStrategy):
    def calculate(self, df):
        p = int(self.params.get('period', 14))
        df['rsi'] = Indicators.rsi(df['close'], p)
        
    def check_signal(self, df):
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Simple Overbought/Oversold Reversion
        bullish = (prev['rsi'] < 30) and (curr['rsi'] >= 30)
        bearish = (prev['rsi'] > 70) and (curr['rsi'] <= 70)
        
        return bullish, bearish, {"rsi": curr['rsi']}
    
    def get_guidance(self, df, position):
        rsi = df.iloc[-1]['rsi']
        if rsi > 70: return "Market is Overbought. Be cautious with Longs, look for bearish divergence."
        if rsi < 30: return "Market is Oversold. Be cautious with Shorts, look for bullish bounce."
        return f"RSI is neutral at {rsi:.1f}. Follow price action."

class PairRatioStrategy(BaseStrategy):
    def __init__(self, name, params, ticker2_data):
        super().__init__(name, params)
        self.ticker2_data = ticker2_data

    def calculate(self, df):
        # Align dataframes
        common_idx = df.index.intersection(self.ticker2_data.index)
        t1 = df.loc[common_idx]['close']
        t2 = self.ticker2_data.loc[common_idx]['close']
        
        ratio = t1 / t2
        df.loc[common_idx, 'ratio'] = ratio
        df.loc[common_idx, 'zscore'] = Indicators.z_score_price(ratio, 20)

    def check_signal(self, df):
        if 'zscore' not in df.columns: return False, False, {}
        curr = df.iloc[-1]
        threshold = float(self.params.get('threshold', 2.0))
        
        # Mean Reversion: If Z > 2, Ratio is too high -> Sell T1 (or Buy T2) -> Bearish for T1
        bearish = curr['zscore'] > threshold
        bullish = curr['zscore'] < -threshold
        
        return bullish, bearish, {"zscore": curr['zscore'], "ratio": curr.get('ratio', 0)}

    def get_guidance(self, df, position):
        z = df.iloc[-1].get('zscore', 0)
        return f"Ratio Z-Score is {z:.2f}. We expect reversion to 0. {'Current deviation is extreme.' if abs(z)>2 else 'Deviation is normal.'}"

class PsychologyStrategy(BaseStrategy):
    def calculate(self, df):
        # Check for 3 consecutive candles of same color (Three White Soldiers / Black Crows)
        # Check for exhaustion (Volume spike on small body)
        df['green'] = df['close'] > df['open']
        df['body'] = (df['close'] - df['open']).abs()
        df['avg_body'] = df['body'].rolling(10).mean()
        
    def check_signal(self, df):
        # Need at least 3 candles
        if len(df) < 4: return False, False, {}
        
        c1, c2, c3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        
        # 3 Consecutive Greens + Expansion
        bullish = (c3['green'] and c2['green'] and c1['green'] and 
                   c1['body'] > c2['body'] and c1['close'] > c1['open'])
                   
        # 3 Consecutive Reds + Expansion
        bearish = (not c3['green'] and not c2['green'] and not c1['green'] and 
                   c1['body'] > c2['body'] and c1['close'] < c1['open'])
                   
        return bullish, bearish, {}

    def get_guidance(self, df, position):
        return "Monitoring market psychology. Looking for consecutive momentum candles indicating herd behavior."

class HybridStrategy(BaseStrategy):
    def calculate(self, df):
        # Hybrid uses EMA, RSI, and Bollinger Bands
        df['ema20'] = Indicators.ema(df['close'], 20)
        df['rsi'] = Indicators.rsi(df['close'], 14)
        u, l = Indicators.bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = u
        df['bb_lower'] = l
        
    def check_signal(self, df):
        curr = df.iloc[-1]
        
        # Bullish: Price > EMA20 AND RSI > 50 AND Price bounced off BB Lower recently
        bullish = (curr['close'] > curr['ema20']) and (curr['rsi'] > 50) and (curr['rsi'] < 70)
        
        # Bearish: Price < EMA20 AND RSI < 50
        bearish = (curr['close'] < curr['ema20']) and (curr['rsi'] < 50) and (curr['rsi'] > 30)
        
        return bullish, bearish, {"hybrid_score": "High" if bullish or bearish else "Neutral"}

    def get_guidance(self, df, position):
        return "Hybrid Mode: Requiring Trend (EMA) + Momentum (RSI) confirmation."

# -----------------------------------------------------------------------------
# 5. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.header("1. Data Feed")
    preset_assets = [
        "^NSEI", "^NSEBANK", "^BSESN", "BTC-USD", "ETH-USD", 
        "GC=F", "SI=F", "INR=X", "EURUSD=X", "GBPUSD=X", "RELIANCE.NS", "TCS.NS"
    ]
    asset_type = st.radio("Asset Source", ["Preset", "Custom"], horizontal=True)
    if asset_type == "Preset":
        ticker = st.selectbox("Select Asset", preset_assets, index=3)
    else:
        ticker = st.text_input("Enter Ticker (e.g., INFY.NS)", value="INFY.NS")
    
    col_tf1, col_tf2 = st.columns(2)
    with col_tf1:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    with col_tf2:
        period = st.selectbox("Lookback", ["1d", "5d", "1mo", "3mo", "1y"], index=0)

    st.header("2. Strategy Logic")
    strategy_type = st.selectbox("Select Strategy", [
        "Hybrid (EMA+RSI+BB)",
        "EMA Crossover", 
        "Pair Ratio Trading", 
        "RSI Mean Reversion", 
        "Psychology (Momentum)",
        
        "Price Action Support/Resistance"
    ])
    
    strat_params = {}
    ticker2_data = None
    
    if strategy_type == "EMA Crossover":
        strat_params['p1'] = st.number_input("Fast Period", 5, 50, 9)
        strat_params['p2'] = st.number_input("Slow Period", 10, 200, 20)
        strat_params['require_strong_candle'] = st.checkbox("Require Strong Candle Close", True)
        
    elif strategy_type == "Pair Ratio Trading":
        ticker2 = st.text_input("Comparison Ticker", value="^NSEBANK")
        strat_params['threshold'] = st.number_input("Z-Score Threshold", 1.0, 4.0, 2.0)
        # We need to fetch ticker2 data later
        
    elif strategy_type == "RSI Mean Reversion":
        strat_params['period'] = st.number_input("RSI Period", 2, 30, 14)

    st.header("3. Risk Management")
    quantity = st.number_input("Quantity/Lots", 1, 10000, 1)
    
    sl_type = st.selectbox("Stop Loss Type", ["Fixed Points", "Trailing Points", "ATR Based", "Signal Based"])
    sl_value = 0.0
    if sl_type != "Signal Based":
        sl_value = st.number_input("SL Value (Pts/Mult)", 0.1, 1000.0, 50.0)
        
    target_type = st.selectbox("Target Type", ["Fixed Points", "Risk:Reward", "Signal Based"])
    target_value = 0.0
    if target_type != "Signal Based":
        target_value = st.number_input("Target Value", 0.1, 1000.0, 100.0)

    st.markdown("---")
    
    # Control Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("▶ START TRADING", use_container_width=True, type="primary")
    with col_btn2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)

    if start_btn:
        st.session_state.trading_active = True
        st.session_state.iteration_count = 0
        log_event(f"System Started. Monitoring {ticker} on {timeframe}.")
        st.rerun()
        
    if stop_btn:
        st.session_state.trading_active = False
        if st.session_state.current_position:
            # Force close
            pos = st.session_state.current_position
            log_event(f"System Stopped. Force closing {pos['type']} at market.")
            st.session_state.current_position = None
        log_event("System Stopped.")
        st.rerun()
        
    if st.session_state.current_position and not st.session_state.trading_active:
        if st.button("Force Close Position"):
             st.session_state.current_position = None
             log_event("Position Manually Closed.")
             st.rerun()

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------

# Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Trading", "📊 Trade History", "📝 System Log"])

def run_trading_cycle():
    """Core loop logic run on every script execution if active"""
    
    # 1. Fetch Data
    df = safe_yf_download(ticker, period, timeframe)
    
    if strategy_type == "Pair Ratio Trading":
        # Fetch Ticker 2
        df2 = safe_yf_download(ticker2, period, timeframe)
        if df is not None and df2 is not None:
             # Instantiate Strategy with data
             strategy = PairRatioStrategy("Ratio", strat_params, df2)
        else:
             st.error("Failed to fetch data for pair.")
             return
    elif strategy_type == "EMA Crossover":
        strategy = EMACrossoverStrategy("EMA", strat_params)
    elif strategy_type == "RSI Mean Reversion":
        strategy = RSIStrategy("RSI", strat_params)
    elif strategy_type == "Psychology (Momentum)":
        strategy = PsychologyStrategy("Psych", strat_params)
    else:
        strategy = HybridStrategy("Hybrid", strat_params)
    
    if df is None or len(df) < 50:
        st.warning("Waiting for sufficient data...")
        return

    # 2. Calculate Indicators
    strategy.calculate(df)
    
    # 3. Get Signals
    bullish, bearish, signal_data = strategy.check_signal(df)
    current_price = df['close'].iloc[-1]
    
    # 4. Position Management
    pos = st.session_state.current_position
    
    # ENTRY LOGIC
    if pos is None:
        if bullish:
            # Calc SL/TP
            sl_price = 0
            tp_price = 0
            
            # Simple Fixed logic for demo (expand for ATR/Trail)
            if sl_type == "Fixed Points": sl_price = current_price - sl_value
            elif sl_type == "ATR Based": 
                atr = Indicators.atr(df).iloc[-1]
                sl_price = current_price - (atr * sl_value)
                
            if target_type == "Fixed Points": tp_price = current_price + target_value
            elif target_type == "Risk:Reward": tp_price = current_price + (current_price - sl_price) * target_value
            
            st.session_state.current_position = {
                'type': 'LONG',
                'entry_price': current_price,
                'quantity': quantity,
                'sl': sl_price,
                'target': tp_price,
                'time': datetime.now(IST),
                'highest_price': current_price # For trailing
            }
            log_event(f"OPEN LONG at {current_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
            st.rerun() # Force refresh to show update
            
        elif bearish:
             # Logic for SHORT (Inverse of above)
            sl_price = 0
            tp_price = 0
            
            if sl_type == "Fixed Points": sl_price = current_price + sl_value
            elif sl_type == "ATR Based": 
                atr = Indicators.atr(df).iloc[-1]
                sl_price = current_price + (atr * sl_value)
                
            if target_type == "Fixed Points": tp_price = current_price - target_value
            elif target_type == "Risk:Reward": tp_price = current_price - (sl_price - current_price) * target_value

            st.session_state.current_position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'quantity': quantity,
                'sl': sl_price,
                'target': tp_price,
                'time': datetime.now(IST),
                'lowest_price': current_price # For trailing
            }
            log_event(f"OPEN SHORT at {current_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
            st.rerun()

    # EXIT / MANAGEMENT LOGIC
    else:
        pnl_points = 0
        exit_reason = None
        
        # Trailing SL Logic
        if sl_type == "Trailing Points":
            if pos['type'] == 'LONG':
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                    pos['sl'] = current_price - sl_value # Move SL up
            else:
                if current_price < pos['lowest_price']:
                    pos['lowest_price'] = current_price
                    pos['sl'] = current_price + sl_value # Move SL down
        
        # Check Exits
        if pos['type'] == 'LONG':
            pnl_points = current_price - pos['entry_price']
            if current_price <= pos['sl'] and pos['sl'] != 0: exit_reason = "Stop Loss Hit"
            elif current_price >= pos['target'] and pos['target'] != 0: exit_reason = "Target Hit"
            elif bearish and sl_type == "Signal Based": exit_reason = "Reversal Signal"
            
        elif pos['type'] == 'SHORT':
            pnl_points = pos['entry_price'] - current_price
            if current_price >= pos['sl'] and pos['sl'] != 0: exit_reason = "Stop Loss Hit"
            elif current_price <= pos['target'] and pos['target'] != 0: exit_reason = "Target Hit"
            elif bullish and sl_type == "Signal Based": exit_reason = "Reversal Signal"
            
        if exit_reason:
            pnl_amt = pnl_points * quantity
            # Log History
            trade_rec = {
                'Strategy': strategy.name,
                'Type': pos['type'],
                'Entry': pos['entry_price'],
                'Exit': current_price,
                'PnL_Pts': pnl_points,
                'PnL_Amt': pnl_amt,
                'Reason': exit_reason,
                'Time_In': pos['time'],
                'Time_Out': datetime.now(IST)
            }
            st.session_state.trade_history.append(trade_rec)
            st.session_state.current_position = None
            log_event(f"CLOSE {pos['type']} at {current_price:.2f}. Reason: {exit_reason}. PnL: {pnl_amt:.2f}")
            st.rerun()
            
    return df, strategy, signal_data

# --- DISPLAY LOGIC ---

with tab1:
    if st.session_state.trading_active:
        # Placeholder for auto-refresh loop
        placeholder = st.empty()
        
        # Execute One Cycle
        result = run_trading_cycle()
        
        if result:
            df, strategy, signal_data = result
            curr_price = df['close'].iloc[-1]
            
            # --- DASHBOARD ---
            st.subheader(f"⚡ Live Monitor: {ticker} ({timeframe})")
            
            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{curr_price:.2f}")
            
            pos = st.session_state.current_position
            if pos:
                pnl = (curr_price - pos['entry_price']) * pos['quantity'] if pos['type'] == 'LONG' else (pos['entry_price'] - curr_price) * pos['quantity']
                color = "profit-text" if pnl >= 0 else "loss-text"
                
                m2.metric("Position", pos['type'], delta=None)
                m3.markdown(f"<div class='{color}'>PnL: {pnl:.2f}</div>", unsafe_allow_html=True)
                m4.metric("Dist to SL", f"{abs(curr_price - pos['sl']):.2f}")
            else:
                m2.metric("Position", "WAITING")
                m3.metric("PnL", "0.00")
                m4.metric("Signal Value", f"{list(signal_data.values())[0]:.2f}" if signal_data else "N/A")

            # Mentor/Guidance Box
            guidance = strategy.get_guidance(df, pos)
            st.markdown(f"""
            <div class="mentor-box">
                <strong>🤖 AI Trading Mentor:</strong><br>
                {guidance}
            </div>
            """, unsafe_allow_html=True)

            # Strategy Parameters Summary
            with st.expander("Strategy Parameters & Live Values", expanded=True):
                sp_cols = st.columns(len(signal_data) + 1 if signal_data else 1)
                for i, (k, v) in enumerate(signal_data.items()):
                    if isinstance(v, (int, float)):
                        sp_cols[i].metric(k.upper(), f"{v:.2f}")
                    else:
                        sp_cols[i].metric(k.upper(), str(v))
            
            # --- CHARTING (PLOTLY) ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'],
                            name="Price"), row=1, col=1)

            # Strategy Overlays
            if strategy_type == "EMA Crossover":
                fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], line=dict(color='orange', width=1), name="Fast EMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], line=dict(color='blue', width=1), name="Slow EMA"), row=1, col=1)
            
            elif strategy_type == "Hybrid (EMA+RSI+BB)":
                fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], line=dict(color='blue', width=1), name="EMA20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='gray', dash='dash'), name="BB Upp"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='gray', dash='dash'), name="BB Low"), row=1, col=1)

            # Position Markers
            if pos:
                fig.add_hline(y=pos['entry_price'], line_dash="solid", line_color="blue", annotation_text="ENTRY", row=1, col=1)
                if pos['sl'] > 0: fig.add_hline(y=pos['sl'], line_dash="dot", line_color="red", annotation_text="SL", row=1, col=1)
                if pos['target'] > 0: fig.add_hline(y=pos['target'], line_dash="dot", line_color="green", annotation_text="TP", row=1, col=1)

            # Secondary Plot (RSI, ZScore etc)
            if 'rsi' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
            elif 'zscore' in df.columns:
                 fig.add_trace(go.Scatter(x=df.index, y=df['zscore'], name="Z-Score"), row=2, col=1)
                 fig.add_hline(y=2, line_color="red", row=2, col=1)
                 fig.add_hline(y=-2, line_color="green", row=2, col=1)
            else:
                # Default Volume
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='rgba(100, 100, 100, 0.5)'), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, 
                              margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="white", plot_bgcolor="white")
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.iteration_count += 1
            time.sleep(1.5) # The critical delay
            st.rerun()
            
    else:
        st.info("System is stopped. Configure settings in sidebar and click START TRADING.")
        # 
        st.markdown("### Market Snapshot (Preview)")
        if st.button("Fetch Preview Data"):
            df = safe_yf_download(ticker, period, timeframe)
            if df is not None:
                st.line_chart(df['close'])

with tab2:
    st.header("Trade History & Analysis")
    if len(st.session_state.trade_history) > 0:
        hist_df = pd.DataFrame(st.session_state.trade_history)
        
        # Summary Metrics
        total_trades = len(hist_df)
        wins = len(hist_df[hist_df['PnL_Amt'] > 0])
        win_rate = (wins / total_trades) * 100
        total_pnl = hist_df['PnL_Amt'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", total_trades)
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Total P&L", f"{total_pnl:.2f}", delta_color="normal")
        c4.metric("Avg Trade", f"{hist_df['PnL_Amt'].mean():.2f}")
        
        st.dataframe(hist_df.style.applymap(lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green' if isinstance(x, (int, float)) and x > 0 else '', subset=['PnL_Amt', 'PnL_Pts']), use_container_width=True)
        
        # AI Feedback on last trade
        last_trade = st.session_state.trade_history[-1]
        feedback = "Analysis: "
        if last_trade['PnL_Amt'] > 0:
            feedback += "Excellent execution. Strategy conditions were met and target/trail secured profit."
        else:
            feedback += f"Loss incurred due to {last_trade['Reason']}. Check if market was chopping or if SL was too tight."
            
        st.info(f"💡 AI Insight on Last Trade: {feedback}")
        
    else:
        st.write("No trades executed yet.")

with tab3:
    st.header("System Logs")
    if st.button("Clear Logs"):
        st.session_state.trade_log = []
        
    log_container = st.container(height=500)
    with log_container:
        for log in st.session_state.trade_log:
            st.text(log)

