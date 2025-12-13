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

# Custom CSS for Professional Look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .status-live {
        color: #00ff00;
        font-weight: bold;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .trade-log {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        height: 600px;
        overflow-y: scroll;
        background-color: #0e1117;
        border: 1px solid #333;
        padding: 10px;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & INDICATORS (Manual Implementation)
# -----------------------------------------------------------------------------

def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def format_ist(dt):
    if dt is None: return "N/A"
    return dt.strftime('%Y-%m-%d %H:%M:%S IST')

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
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma

    @staticmethod
    def donchian_channel(series_high, series_low, period=20):
        upper = series_high.rolling(window=period).max()
        lower = series_low.rolling(window=period).min()
        return upper, lower

    @staticmethod
    def z_score(series, period=20):
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std

# -----------------------------------------------------------------------------
# 3. STRATEGY ENGINE
# -----------------------------------------------------------------------------

class BaseStrategy:
    def __init__(self, name):
        self.name = name

    def calculate_indicators(self, data):
        raise NotImplementedError

    def generate_signal(self, data):
        # Returns: (bullish_bool, bearish_bool, details_dict)
        raise NotImplementedError

class CrossoverStrategy(BaseStrategy):
    def __init__(self, type1='EMA', type2='EMA', p1=9, p2=20):
        super().__init__("MA Crossover")
        self.type1, self.type2 = type1, type2
        self.p1, self.p2 = int(p1), int(p2)

    def calculate_indicators(self, data):
        close = data['Close']
        data['MA1'] = Indicators.ema(close, self.p1) if self.type1 == 'EMA' else Indicators.sma(close, self.p1)
        data['MA2'] = Indicators.ema(close, self.p2) if self.type2 == 'EMA' else Indicators.sma(close, self.p2)
        return data

    def generate_signal(self, data):
        if len(data) < 2: return False, False, {}
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        bullish = (prev['MA1'] <= prev['MA2']) and (curr['MA1'] > curr['MA2'])
        bearish = (prev['MA1'] >= prev['MA2']) and (curr['MA1'] < curr['MA2'])
        
        return bullish, bearish, {f"{self.type1}{self.p1}": curr['MA1'], f"{self.type2}{self.p2}": curr['MA2']}

class RsiDivergenceStrategy(BaseStrategy):
    def __init__(self, period=14):
        super().__init__("RSI Divergence")
        self.period = int(period)

    def calculate_indicators(self, data):
        data['RSI'] = Indicators.rsi(data['Close'], self.period)
        return data

    def generate_signal(self, data):
        if len(data) < 15: return False, False, {}
        curr = data.iloc[-1]
        
        # Simple Logic: Oversold/Overbought + Basic Trend Check
        # Full divergence requires lookback logic which is heavy for live loop, using simplified trigger
        bullish = curr['RSI'] < 30
        bearish = curr['RSI'] > 70
        
        return bullish, bearish, {"RSI": curr['RSI']}

class FibonacciStrategy(BaseStrategy):
    def __init__(self, lookback=50):
        super().__init__("Fibonacci Retracement")
        self.lookback = lookback

    def calculate_indicators(self, data):
        # Calculate High/Low of lookback period
        window = data.tail(self.lookback)
        high = window['High'].max()
        low = window['Low'].min()
        diff = high - low
        
        data['Fib_0'] = low
        data['Fib_236'] = low + 0.236 * diff
        data['Fib_382'] = low + 0.382 * diff
        data['Fib_50'] = low + 0.5 * diff
        data['Fib_618'] = low + 0.618 * diff
        data['Fib_100'] = high
        return data

    def generate_signal(self, data):
        if len(data) < self.lookback: return False, False, {}
        curr = data.iloc[-1]
        c = curr['Close']
        
        # Tolerance 0.5%
        tol = c * 0.005
        
        # Bullish if bouncing off 61.8 or 50 level
        near_618 = abs(c - curr['Fib_618']) < tol
        near_50 = abs(c - curr['Fib_50']) < tol
        
        bullish = (near_618 or near_50) and (c > data.iloc[-2]['Close']) # Small confirmation
        bearish = False # Simplified for example
        
        return bullish, bearish, {"Fib 61.8": curr['Fib_618'], "Fib 50": curr['Fib_50']}

class BreakoutStrategy(BaseStrategy):
    def __init__(self, period=20, vol_mult=1.5):
        super().__init__("Volume Breakout")
        self.period = period
        self.vol_mult = vol_mult

    def calculate_indicators(self, data):
        data['Upper'], data['Lower'] = Indicators.donchian_channel(data['High'], data['Low'], self.period)
        data['Vol_Avg'] = data['Volume'].rolling(window=self.period).mean()
        return data

    def generate_signal(self, data):
        if len(data) < self.period: return False, False, {}
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        vol_spike = curr['Volume'] > (curr['Vol_Avg'] * self.vol_mult)
        
        bullish = (curr['Close'] > prev['Upper']) and vol_spike
        bearish = (curr['Close'] < prev['Lower']) and vol_spike
        
        return bullish, bearish, {"Vol Ratio": round(curr['Volume']/curr['Vol_Avg'], 2) if curr['Vol_Avg'] else 0}

class ZScoreStrategy(BaseStrategy):
    def __init__(self, period=20, threshold=2.0):
        super().__init__("Mean Reversion (Z-Score)")
        self.period = period
        self.threshold = threshold

    def calculate_indicators(self, data):
        data['Z_Score'] = Indicators.z_score(data['Close'], self.period)
        return data

    def generate_signal(self, data):
        curr = data.iloc[-1]
        z = curr['Z_Score']
        
        # Revert to mean: Short if Z > 2, Long if Z < -2
        bullish = z < -self.threshold
        bearish = z > self.threshold
        
        return bullish, bearish, {"Z-Score": z}

class ElliottWaveSimpleStrategy(BaseStrategy):
    def __init__(self, order=5):
        super().__init__("Elliott Wave (Simplified)")
        self.order = order

    def calculate_indicators(self, data):
        # Find local maxima and minima
        data['is_max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=self.order)[0]]['High']
        data['is_min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=self.order)[0]]['Low']
        return data

    def generate_signal(self, data):
        # Very simplified 5th wave detection logic
        # Looking for recent Higher High (Wave 3) followed by Higher Low (Wave 4)
        # Entry for Wave 5
        return False, False, {"Info": "Pattern Analysis"} # Placeholder for complex logic

# -----------------------------------------------------------------------------
# 4. DATA MANAGER
# -----------------------------------------------------------------------------

def fetch_data(ticker, timeframe, period, is_ratio=False, ratio_ticker=None):
    """
    Fetches data handling Rate Limits, Timezones and MultiIndex.
    """
    time.sleep(1.5) # Rate limiting
    
    try:
        if is_ratio and ratio_ticker:
            # Multi-asset logic
            df1 = yf.download(ticker, period=period, interval=timeframe, progress=False)
            time.sleep(1.5)
            df2 = yf.download(ratio_ticker, period=period, interval=timeframe, progress=False)
            
            # Handling yfinance MultiIndex if present
            if isinstance(df1.columns, pd.MultiIndex): df1.columns = df1.columns.droplevel(1)
            if isinstance(df2.columns, pd.MultiIndex): df2.columns = df2.columns.droplevel(1)
            
            # Align Timezones (UTC -> IST)
            if df1.index.tz is None: df1.index = df1.index.tz_localize('UTC')
            else: df1.index = df1.index.tz_convert('UTC')
            
            if df2.index.tz is None: df2.index = df2.index.tz_localize('UTC')
            else: df2.index = df2.index.tz_convert('UTC')
            
            # Intersection
            common_idx = df1.index.intersection(df2.index)
            df1 = df1.loc[common_idx]
            df2 = df2.loc[common_idx]
            
            # Ratio Data
            df = pd.DataFrame(index=common_idx)
            df['Close'] = df1['Close'] / df2['Close']
            df['Open'] = df1['Open'] / df2['Open']
            df['High'] = df1['High'] / df2['High']
            df['Low'] = df1['Low'] / df2['Low']
            df['Volume'] = (df1['Volume'] + df2['Volume']) / 2 # Approx
            
        else:
            # Single asset logic
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        
        if df.empty:
            return None

        # Ensure IST timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# -----------------------------------------------------------------------------
# 5. AI ANALYSIS SIMULATOR
# -----------------------------------------------------------------------------

def generate_ai_analysis(trade_record):
    """
    Simulates an AI analysis based on trade metrics.
    """
    pnl = trade_record['pnl_perc']
    duration = trade_record['duration']
    reason = trade_record['exit_reason']
    
    analysis = ""
    recommendation = ""
    
    if pnl > 0.5:
        analysis = f"Excellent trade capture. The entry aligned well with momentum, yielding a {pnl:.2f}% return."
        recommendation = "Consider scaling up position size on similar setups. Check if trailing SL could have captured more."
    elif pnl > 0:
        analysis = "Profitable trade, but marginal. Price action was choppy after entry."
        recommendation = "Review entry timing. Waiting for a candle close confirmation might improve yield."
    elif pnl > -0.5:
        analysis = f"Minor loss ({pnl:.2f}%). Stop loss performed its job protecting capital."
        recommendation = "Standard operating loss. Ensure market volatility wasn't too high for the tight SL."
    else:
        analysis = f"Significant drawdown. The trade went against the position immediately."
        recommendation = "Check for major support/resistance levels that were ignored. Avoid trading against strong trend."

    return {
        "Performance Assessment": analysis,
        "Exit Quality": f"Exit triggered by {reason}.",
        "Key Insight": recommendation
    }

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION LOGIC
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
    if len(st.session_state.log_entries) > 100:
        st.session_state.log_entries.pop()

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Config")

# Asset Selection
asset_type = st.sidebar.selectbox("Asset Class", ["Indices", "Crypto", "Forex", "Stocks", "Custom"])
ticker_map = {
    "NIFTY 50": "^NSEI", "Bank NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "Gold": "GC=F",
    "EUR/USD": "EURUSD=X", "USD/INR": "INR=X"
}

if asset_type == "Custom":
    ticker = st.sidebar.text_input("YFinance Ticker", "RELIANCE.NS")
else:
    t_choice = st.sidebar.selectbox("Instrument", list(ticker_map.keys()) if asset_type != "Stocks" else ["RELIANCE.NS", "TCS.NS"])
    ticker = ticker_map.get(t_choice, t_choice)

# Timeframe
col1, col2 = st.sidebar.columns(2)
timeframe = col1.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
period = col2.selectbox("Data Period", ["1d", "5d", "1mo", "1y"], index=1)

# Strategy Selection
st.sidebar.subheader("Strategy")
strat_name = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", "SMA Crossover", "RSI Divergence", 
    "Fibonacci Retracement", "Breakout (Volume)", "Z-Score Reversion"
])

strategy = None
if "Crossover" in strat_name:
    s_p1 = st.sidebar.number_input("Fast Period", 5, 200, 9)
    s_p2 = st.sidebar.number_input("Slow Period", 5, 200, 20)
    strategy = CrossoverStrategy('EMA' if "EMA" in strat_name else 'SMA', 'EMA', s_p1, s_p2)
elif "RSI" in strat_name:
    strategy = RsiDivergenceStrategy()
elif "Breakout" in strat_name:
    strategy = BreakoutStrategy()
elif "Z-Score" in strat_name:
    strategy = ZScoreStrategy()
elif "Fibonacci" in strat_name:
    strategy = FibonacciStrategy()
else:
    strategy = CrossoverStrategy() # Default

# Risk Management
st.sidebar.subheader("Risk Management")
qty = st.sidebar.number_input("Quantity", 1, 10000, 1)
sl_type = st.sidebar.selectbox("Stop Loss", ["Fixed Points", "Trailing Points", "Signal Based"])
sl_val = st.sidebar.number_input("SL Value (Pts)", 0.0, 1000.0, 50.0)
tp_type = st.sidebar.selectbox("Target", ["Fixed Points", "Signal Based"])
tp_val = st.sidebar.number_input("Target Value (Pts)", 0.0, 1000.0, 100.0)

# Controls
st.sidebar.markdown("---")
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ START", type="primary"):
    st.session_state.trading_active = True
    log_msg(f"System STARTED for {ticker} ({timeframe})")

if c2.button("⏹ STOP"):
    st.session_state.trading_active = False
    log_msg("System STOPPED by user")
    # Close positions if any
    if st.session_state.current_position:
        log_msg("Forcing position closure...")
        # Logic to close would go here

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📜 History", "SYSTEM LOG"])

# --- TAB 1: LIVE TRADING ---
with tab1:
    placeholder = st.empty()
    
    # MAIN LOOP LOGIC
    if st.session_state.trading_active:
        with placeholder.container():
            st.session_state.iteration += 1
            
            # 1. Fetch Data
            df = fetch_data(ticker, timeframe, period)
            
            if df is not None:
                # 2. Calculate Indicators
                df = strategy.calculate_indicators(df)
                current_price = df['Close'].iloc[-1]
                
                # 3. Check Signals
                bullish, bearish, tech_data = strategy.generate_signal(df)
                
                # 4. Manage Positions
                pos = st.session_state.current_position
                action = "WAIT"
                
                # Exit Logic
                if pos:
                    pnl_pts = (current_price - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - current_price)
                    pnl_perc = (pnl_pts / pos['entry_price']) * 100
                    
                    # Update Trailing SL
                    if sl_type == "Trailing Points":
                        if pos['type'] == 'LONG':
                            new_sl = current_price - sl_val
                            if new_sl > pos['sl']:
                                pos['sl'] = new_sl
                                log_msg(f"Trailing SL updated to {new_sl:.2f}")
                        else:
                            new_sl = current_price + sl_val
                            if new_sl < pos['sl']:
                                pos['sl'] = new_sl
                                log_msg(f"Trailing SL updated to {new_sl:.2f}")

                    # Check Exits
                    exit_reason = None
                    if pos['type'] == 'LONG':
                        if current_price <= pos['sl']: exit_reason = "SL Hit"
                        elif tp_type == "Fixed Points" and current_price >= pos['tp']: exit_reason = "Target Hit"
                        elif bearish and sl_type == "Signal Based": exit_reason = "Signal Reversal"
                    else: # SHORT
                        if current_price >= pos['sl']: exit_reason = "SL Hit"
                        elif tp_type == "Fixed Points" and current_price <= pos['tp']: exit_reason = "Target Hit"
                        elif bullish and sl_type == "Signal Based": exit_reason = "Signal Reversal"
                        
                    if exit_reason:
                        # Close Trade
                        rec = {
                            "symbol": ticker, "type": pos['type'], 
                            "entry_time": pos['entry_time'], "exit_time": get_ist_time(),
                            "entry_price": pos['entry_price'], "exit_price": current_price,
                            "pnl_pts": pnl_pts, "pnl_perc": pnl_perc,
                            "qty": pos['qty'], "exit_reason": exit_reason,
                            "duration": str(get_ist_time() - pos['entry_time'])
                        }
                        rec['ai_analysis'] = generate_ai_analysis(rec)
                        st.session_state.trade_history.append(rec)
                        st.session_state.current_position = None
                        log_msg(f"Position CLOSED ({exit_reason}): P&L {pnl_pts:.2f}")
                        action = "CLOSE"

                # Entry Logic (only if no position)
                elif not pos:
                    if bullish:
                        sl = current_price - sl_val if sl_type != "Signal Based" else current_price * 0.99
                        tp = current_price + tp_val if tp_type != "Signal Based" else current_price * 1.05
                        st.session_state.current_position = {
                            "type": "LONG", "entry_price": current_price, "entry_time": get_ist_time(),
                            "qty": qty, "sl": sl, "tp": tp
                        }
                        log_msg(f"LONG Entry at {current_price:.2f}")
                        action = "ENTRY LONG"
                    elif bearish:
                        sl = current_price + sl_val if sl_type != "Signal Based" else current_price * 1.01
                        tp = current_price - tp_val if tp_type != "Signal Based" else current_price * 0.95
                        st.session_state.current_position = {
                            "type": "SHORT", "entry_price": current_price, "entry_time": get_ist_time(),
                            "qty": qty, "sl": sl, "tp": tp
                        }
                        log_msg(f"SHORT Entry at {current_price:.2f}")
                        action = "ENTRY SHORT"

                # 5. Display UI
                # Header
                st.markdown(f"### 🔴 LIVE MONITORING | Iteration: {st.session_state.iteration}")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"{current_price:.2f}")
                
                pos_status = "FLAT"
                pnl_disp = 0.0
                if st.session_state.current_position:
                    pos = st.session_state.current_position
                    pos_status = f"{pos['type']} @ {pos['entry_price']:.2f}"
                    curr_pnl = (current_price - pos['entry_price']) * qty if pos['type'] == 'LONG' else (pos['entry_price'] - current_price) * qty
                    pnl_disp = curr_pnl
                    m3.metric("Unrealized P&L", f"{curr_pnl:.2f}", delta_color="normal")
                
                m2.metric("Position", pos_status)
                m4.metric("Strategy Signal", "BULL" if bullish else ("BEAR" if bearish else "NEUTRAL"))

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                
                # Add Strategy Indicators to Chart
                if "MA1" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA1'], line=dict(color='orange', width=1), name='Fast MA'))
                if "MA2" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA2'], line=dict(color='blue', width=1), name='Slow MA'))
                if "Upper" in df.columns: 
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', dash='dash'), name='Upper'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', dash='dash'), name='Lower'))

                # Mark Position
                if st.session_state.current_position:
                    pos = st.session_state.current_position
                    fig.add_hline(y=pos['entry_price'], line_color="yellow", annotation_text="ENTRY")
                    fig.add_hline(y=pos['sl'], line_color="red", line_dash="dot", annotation_text="SL")
                    fig.add_hline(y=pos['tp'], line_color="green", line_dash="dot", annotation_text="TP")

                fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Data Box
                st.caption(f"Technical Data: {tech_data}")

            else:
                st.warning("Waiting for data...")
            
            # RERUN
            time.sleep(2)
            st.rerun()

    else:
        st.info("System is STOPPED. Click START in sidebar to begin live monitoring.")
        if fetch_data(ticker, timeframe, period) is not None:
             # Static Preview
             df = fetch_data(ticker, timeframe, period)
             df = strategy.calculate_indicators(df)
             fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
             fig.update_layout(height=500, template="plotly_dark", title=f"Preview: {ticker}")
             st.plotly_chart(fig, use_container_width=True)


# --- TAB 2: HISTORY ---
with tab2:
    if not st.session_state.trade_history:
        st.info("No trades executed yet.")
    else:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        # Summary Metrics
        total_trades = len(history_df)
        wins = len(history_df[history_df['pnl_pts'] > 0])
        win_rate = (wins / total_trades) * 100
        total_pnl = history_df['pnl_pts'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", total_trades)
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Total P&L (Pts)", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}")
        
        st.markdown("### Trade Log")
        st.dataframe(history_df[['entry_time', 'type', 'entry_price', 'exit_price', 'pnl_pts', 'exit_reason']], use_container_width=True)
        
        # AI Analysis View
        st.markdown("### 🤖 AI Trade Analysis")
        selected_trade = st.selectbox("Select Trade for Analysis", options=range(len(history_df)), format_func=lambda x: f"Trade #{x+1} ({history_df.iloc[x]['exit_reason']})")
        
        if selected_trade is not None:
            trade_data = history_df.iloc[selected_trade]
            ai_data = trade_data['ai_analysis']
            
            ac1, ac2 = st.columns(2)
            with ac1:
                st.info(f"**Assessment:** {ai_data['Performance Assessment']}")
            with ac2:
                st.success(f"**Recommendation:** {ai_data['Key Insight']}")

# --- TAB 3: LOGS ---
with tab3:
    st.markdown(f'<div class="trade-log">{"<br>".join(st.session_state.log_entries)}</div>', unsafe_allow_html=True)
    if st.button("Clear Logs"):
        st.session_state.log_entries = []
        st.rerun()

