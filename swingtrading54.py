import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pytz
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import math

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="Pro Algo-Trader AI", page_icon="📈")

# Custom CSS for Professional UI
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; }
    .trade-log { font-family: 'Courier New', monospace; font-size: 12px; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF4B4B; font-weight: bold; }
    .status-live { color: #00FF00; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TSLA", "AAPL"]
}

TIMEFRAMES = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.trading_active = False
    st.session_state.position = None  # {type, entry_price, quantity, sl, target, entry_time, strategy}
    st.session_state.trade_history = []
    st.session_state.logs = []
    st.session_state.iteration = 0
    st.session_state.last_api_call = datetime.now()

def log(message, type="INFO"):
    tz = pytz.timezone('Asia/Kolkata')
    ts = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.logs.insert(0, f"[{ts}] [{type}] {message}")
    if len(st.session_state.logs) > 100:
        st.session_state.logs.pop()

# ==========================================
# DATA & INDICATOR ENGINE (MANUAL CALCS)
# ==========================================
class DataEngine:
    @staticmethod
    def fetch_data(ticker, timeframe, period):
        # Rate limiting
        now = datetime.now()
        delta = (now - st.session_state.last_api_call).total_seconds()
        if delta < 1.5:
            time.sleep(1.5 - delta)
        st.session_state.last_api_call = datetime.now()
        
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False, multi_level_index=False)
            if df.empty:
                return None
            
            # Handle timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            
            # Basic cleanup
            df = df.dropna()
            return df
        except Exception as e:
            log(f"Data Fetch Error: {str(e)}", "ERROR")
            return None

    @staticmethod
    def calculate_indicators(df):
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # SMA / EMA
        def sma(data, period):
            return pd.Series(data).rolling(window=period).mean().values
        
        def ema(data, period):
            return pd.Series(data).ewm(span=period, adjust=False).mean().values

        # RSI
        def rsi(data, period=14):
            delta = np.diff(data, prepend=data[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))

        # ATR
        def atr(high, low, close, period=14):
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = tr1[0]
            return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

        # Bollinger/Donchian
        df['SMA_20'] = sma(close, 20)
        df['EMA_9'] = ema(close, 9)
        df['EMA_20'] = ema(close, 20)
        df['EMA_50'] = ema(close, 50)
        df['EMA_200'] = ema(close, 200)
        df['RSI'] = rsi(close)
        df['ATR'] = atr(high, low, close)
        
        # Donchian Channel (20)
        df['DC_High'] = pd.Series(high).rolling(window=20).max().values
        df['DC_Low'] = pd.Series(low).rolling(window=20).min().values
        
        # Z-Score (20 period)
        std_20 = pd.Series(close).rolling(window=20).std().values
        df['Z_Score'] = (close - df['SMA_20']) / (std_20 + 1e-10)
        
        return df

# ==========================================
# STRATEGY ENGINE
# ==========================================
class BaseStrategy:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def get_signal(self, df):
        return (False, False, {})

    def get_analysis(self, df):
        return "Analyzing market structure..."

class EMACrossoverStrategy(BaseStrategy):
    def get_signal(self, df):
        if len(df) < 50: return (False, False, {})
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        p1 = int(self.params.get('p1', 9))
        p2 = int(self.params.get('p2', 20))
        c_type = self.params.get('cross_type', 'simple')
        
        # Custom calc based on params
        fast = pd.Series(df['Close']).ewm(span=p1, adjust=False).mean().values
        slow = pd.Series(df['Close']).ewm(span=p2, adjust=False).mean().values
        
        cross_up = (fast[-2] <= slow[-2]) and (fast[-1] > slow[-1])
        cross_down = (fast[-2] >= slow[-2]) and (fast[-1] < slow[-1])
        
        # Confirmation Logic
        confirmed = True
        reason = "Simple Crossover"
        
        if c_type == 'auto_strong_candle':
            body_size = abs(curr['Close'] - curr['Open'])
            avg_body = np.mean(np.abs(df['Close'].values[-10:] - df['Open'].values[-10:]))
            if body_size < 1.5 * avg_body:
                confirmed = False
                reason = "Candle not strong enough"
            else:
                reason = "Strong Candle Crossover"
        
        elif c_type == 'atr_strong_candle':
            body_size = abs(curr['Close'] - curr['Open'])
            if body_size < self.params.get('atr_mult', 1.0) * curr['ATR']:
                confirmed = False
            else:
                reason = f"ATR ({self.params.get('atr_mult')}x) Confirmed"

        # Angle Calculation (Approximate)
        angle = 0
        if confirmed:
            try:
                # Normalizing slope
                price_range = df['Close'].max() - df['Close'].min()
                slope = (fast[-1] - fast[-5]) / 5  # slope over 5 bars
                norm_slope = slope / price_range
                angle = math.degrees(math.atan(norm_slope * 100)) # Scaling factor
            except: pass
            
            if abs(angle) < self.params.get('min_angle', 0):
                confirmed = False
                reason = f"Angle {angle:.1f}° too shallow"

        buy = cross_up and confirmed
        sell = cross_down and confirmed
        
        return buy, sell, {
            "Fast EMA": fast[-1], "Slow EMA": slow[-1], 
            "Angle": f"{angle:.1f}°", "Confirmation": reason
        }

class ElliottWaveStrategy(BaseStrategy):
    def get_signal(self, df):
        # Simplified Elliott Wave Logic: 5-wave impulse
        if len(df) < 100: return False, False, {}
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local extrema (Swing Highs/Lows)
        order = 5 # Window for extrema
        max_idx = argrelextrema(highs, np.greater, order=order)[0]
        min_idx = argrelextrema(lows, np.less, order=order)[0]
        
        # Need at least 3 highs and 2 lows for 1-2-3-4-5
        if len(max_idx) < 3 or len(min_idx) < 3:
            return False, False, {"Status": "Insufficient Structure"}
            
        # Extract last few pivots to identify potential Wave 5
        # Logic: Finding a sequence Low(0)-High(1)-Low(2)-High(3)-Low(4)-High(5)
        # Assuming current price is potentially Wave 5 end or start of correction
        
        # Simple detection: Recent trend analysis
        # Check if we are potentially at the end of Wave 5 (Bearish Reversal)
        # or end of Wave C/2 (Bullish)
        
        last_high = highs[max_idx[-1]]
        prev_high = highs[max_idx[-2]]
        last_low = lows[min_idx[-1]]
        
        # Check Divergence (Price Higher High, RSI Lower High) -> Potential Wave 5 Top
        rsi = df['RSI'].values
        price_hh = last_high > prev_high
        rsi_lh = rsi[max_idx[-1]] < rsi[max_idx[-2]]
        
        bearish_wave_5 = price_hh and rsi_lh and rsi[max_idx[-1]] > 60
        
        # Bullish: Price Lower Low, RSI Higher Low
        last_low_val = lows[min_idx[-1]]
        prev_low_val = lows[min_idx[-2]]
        price_ll = last_low_val < prev_low_val
        rsi_hl = rsi[min_idx[-1]] > rsi[min_idx[-2]]
        
        bullish_wave_end = price_ll and rsi_hl and rsi[min_idx[-1]] < 40
        
        return bullish_wave_end, bearish_wave_5, {
            "Wave Stage": "Potential W5 Top" if bearish_wave_5 else "Potential Correction End" if bullish_wave_end else "Forming",
            "Divergence": "Present" if (bearish_wave_5 or bullish_wave_end) else "None"
        }

class PsychologyStrategy(BaseStrategy):
    def get_signal(self, df):
        # Contrarian: Buy when Fear is high (RSI < 25 + 3 red candles), Sell when Greed high
        if len(df) < 5: return False, False, {}
        
        curr = df.iloc[-1]
        closes = df['Close'].values
        opens = df['Open'].values
        
        red_candles = np.sum((closes[-3:] < opens[-3:])) == 3
        green_candles = np.sum((closes[-3:] > opens[-3:])) == 3
        
        fear = curr['RSI'] < 25 and red_candles
        greed = curr['RSI'] > 75 and green_candles
        
        return fear, greed, {"Sentiment": "Extreme Fear" if fear else "Extreme Greed" if greed else "Neutral", "RSI": curr['RSI']}

# ==========================================
# TRADING SYSTEM CORE
# ==========================================
class TradingSystem:
    def __init__(self):
        self.strategies = {
            "EMA Crossover": EMACrossoverStrategy,
            "Elliott Wave (Sim)": ElliottWaveStrategy,
            "Psychology / Contrarian": PsychologyStrategy,
            "Simple Buy": BaseStrategy, # Placeholder
            "Simple Sell": BaseStrategy # Placeholder
        }

    def execute(self, ticker, timeframe, period, strategy_name, strat_params, risk_params):
        st.session_state.iteration += 1
        
        # 1. Fetch Data
        df = DataEngine.fetch_data(ticker, timeframe, period)
        if df is None:
            st.error("Failed to fetch data or rate limit hit.")
            return

        # 2. Calculate Indicators
        df = DataEngine.calculate_indicators(df)
        curr_price = df['Close'].iloc[-1]
        
        # 3. Strategy Signal
        strategy_class = self.strategies.get(strategy_name, BaseStrategy)
        
        # Handle specific strategy instantiation
        if strategy_name == "Simple Buy":
            buy_sig, sell_sig, debug = True, False, {"Mode": "Manual Buy"}
        elif strategy_name == "Simple Sell":
            buy_sig, sell_sig, debug = False, True, {"Mode": "Manual Sell"}
        else:
            strat = strategy_class(strategy_name, strat_params)
            buy_sig, sell_sig, debug = strat.get_signal(df)

        # 4. Position Management
        pos = st.session_state.position
        
        # EXIT LOGIC
        if pos:
            pnl_pts = (curr_price - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - curr_price)
            pnl_pct = (pnl_pts / pos['entry_price']) * 100
            
            exit_reason = None
            
            # SL/Target Checks
            if pos['type'] == 'LONG':
                if curr_price <= pos['sl']: exit_reason = "Stop Loss Hit"
                elif curr_price >= pos['target']: exit_reason = "Target Hit"
                elif sell_sig and risk_params['exit_on_signal']: exit_reason = "Signal Reversal"
            else: # SHORT
                if curr_price >= pos['sl']: exit_reason = "Stop Loss Hit"
                elif curr_price <= pos['target']: exit_reason = "Target Hit"
                elif buy_sig and risk_params['exit_on_signal']: exit_reason = "Signal Reversal"
            
            # Trailing SL Logic
            if risk_params['sl_type'] == 'Trail SL':
                trail_pts = risk_params['sl_value']
                if pos['type'] == 'LONG':
                    new_sl = curr_price - trail_pts
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
                        log(f"Trailing SL moved to {new_sl:.2f}")
                else:
                    new_sl = curr_price + trail_pts
                    if new_sl < pos['sl']:
                        pos['sl'] = new_sl
                        log(f"Trailing SL moved to {new_sl:.2f}")

            if exit_reason:
                self.close_position(curr_price, exit_reason, pnl_pts, pnl_pct)
        
        # ENTRY LOGIC
        elif not pos:
            if buy_sig:
                self.open_position("LONG", curr_price, risk_params, strategy_name)
            elif sell_sig:
                self.open_position("SHORT", curr_price, risk_params, strategy_name)

        # 5. UI Updates
        self.render_live_dashboard(df, debug, risk_params)
    
    def open_position(self, type, price, risk_params, strat_name):
        sl_dist = risk_params['sl_value']
        target_dist = risk_params['target_value']
        
        # Advanced SL/Target Calc (e.g. ATR based)
        # Simplified here for brevity, assumes Points if not specified otherwise
        
        sl_price = price - sl_dist if type == 'LONG' else price + sl_dist
        target_price = price + target_dist if type == 'LONG' else price - target_dist
        
        st.session_state.position = {
            'type': type, 'entry_price': price, 'qty': risk_params['qty'],
            'sl': sl_price, 'target': target_price, 
            'time': datetime.now().strftime('%H:%M:%S'), 'strategy': strat_name
        }
        log(f"OPEN {type} @ {price:.2f} | SL: {sl_price:.2f} | TGT: {target_price:.2f}", "TRADE")

    def close_position(self, price, reason, pnl, pnl_pct):
        pos = st.session_state.position
        
        # AI Feedback Generation
        ai_feedback = "Neutral trade."
        if pnl > 0:
            ai_feedback = "Great execution! Strategy captured the move correctly."
            if reason == "Target Hit": ai_feedback += " Perfect target exit."
        else:
            ai_feedback = "Loss incurred."
            if reason == "Stop Loss Hit": ai_feedback += " Risk management saved capital. Review entry timing."
            
        trade_record = {
            "Time": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "Type": pos['type'], "Entry": pos['entry_price'], "Exit": price,
            "PnL": pnl, "PnL%": pnl_pct, "Reason": reason, "AI Feedback": ai_feedback,
            "Strategy": pos['strategy']
        }
        st.session_state.trade_history.append(trade_record)
        st.session_state.position = None
        log(f"CLOSE {pos['type']} @ {price:.2f} | PnL: {pnl:.2f} ({pnl_pct:.2f}%) | {reason}", "TRADE")

    def render_live_dashboard(self, df, debug_data, risk_params):
        curr = df.iloc[-1]
        
        # Header Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"{curr['Close']:.2f}", f"{(curr['Close'] - df.iloc[-2]['Close']):.2f}")
        c2.metric("RSI (14)", f"{curr['RSI']:.1f}")
        c3.metric("ATR", f"{curr['ATR']:.2f}")
        c4.metric("Iteration", st.session_state.iteration)
        
        # Position Status
        pos = st.session_state.position
        if pos:
            pnl = (curr['Close'] - pos['entry_price']) if pos['type'] == 'LONG' else (pos['entry_price'] - curr['Close'])
            pnl_pct = (pnl / pos['entry_price']) * 100
            color = "green" if pnl >= 0 else "red"
            
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {color}'>
                <h3>LIVE POSITION: {pos['type']}</h3>
                <p>Entry: {pos['entry_price']:.2f} | Qty: {pos['qty']}</p>
                <h2 style='color:{color}'>PnL: {pnl:.2f} ({pnl_pct:.2f}%)</h2>
                <p>SL: {pos['sl']:.2f} | TGT: {pos['target']:.2f}</p>
                <p><i>Guidance: {"Hold position, trend is respecting structure." if pnl > 0 else "Monitor closely, approaching invalidation point."}</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("WAITING FOR SIGNAL... Scanning market structure.")

        # Charting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Price & MA
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='orange', width=1), name='EMA 9'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DC_High'], line=dict(color='gray', dash='dot'), name='DC High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DC_Low'], line=dict(color='gray', dash='dot'), name='DC Low'), row=1, col=1)
        
        # Show Pos Levels
        if pos:
            fig.add_hline(y=pos['entry_price'], line_dash="solid", line_color="blue", annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=pos['sl'], line_dash="dash", line_color="red", annotation_text="SL", row=1, col=1)
            fig.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="TGT", row=1, col=1)

        # Indicator Pane
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy Debug
        with st.expander("Strategy Parameters & AI Analysis", expanded=True):
            cols = st.columns(len(debug_data))
            for i, (k, v) in enumerate(debug_data.items()):
                cols[i].metric(k, str(v))

# ==========================================
# UI SIDEBAR
# ==========================================
st.sidebar.title("🤖 Pro Algo-Trader")

# Asset Config
st.sidebar.header("1. Asset Configuration")
asset_cat = st.sidebar.selectbox("Category", list(ASSETS.keys()))
ticker = st.sidebar.selectbox("Ticker", ASSETS[asset_cat])
custom_ticker = st.sidebar.text_input("Or Custom Ticker (e.g., INFY.NS)")
if custom_ticker: ticker = custom_ticker

timeframe = st.sidebar.select_slider("Timeframe", TIMEFRAMES, value="5m")
period_map = {"1m": "5d", "5m": "5d", "15m": "1mo", "1h": "1y", "1d": "5y"}
period = period_map.get(timeframe, "1y") # Default fallback

# Strategy Config
st.sidebar.header("2. Strategy Configuration")
strategy_name = st.sidebar.selectbox("Select Strategy", [
    "EMA Crossover", 
    "Elliott Wave (Sim)", 
    "Psychology / Contrarian",
    "Pair Ratio Trading (Placeholder)",
    "Simple Buy", 
    "Simple Sell"
])

strat_params = {}
if strategy_name == "EMA Crossover":
    strat_params['p1'] = st.sidebar.number_input("Fast EMA", 5, 50, 9)
    strat_params['p2'] = st.sidebar.number_input("Slow EMA", 10, 200, 20)
    strat_params['cross_type'] = st.sidebar.selectbox("Crossover Type", ["simple", "auto_strong_candle", "atr_strong_candle"])
    if strat_params['cross_type'] == "atr_strong_candle":
        strat_params['atr_mult'] = st.sidebar.number_input("ATR Multiplier", 0.5, 3.0, 1.0)
    strat_params['min_angle'] = st.sidebar.number_input("Min Angle (Deg)", 0, 90, 15)

# Risk Config
st.sidebar.header("3. Risk Management")
qty = st.sidebar.number_input("Quantity", 1, 10000, 1)
sl_type = st.sidebar.selectbox("Stop Loss Type", ["Custom Points", "Trail SL"])
sl_value = st.sidebar.number_input("SL Points / Trail Distance", 0.0, 1000.0, 10.0)
target_value = st.sidebar.number_input("Target Points", 0.0, 1000.0, 20.0)
exit_on_signal = st.sidebar.checkbox("Exit on Opposite Signal", True)

risk_params = {
    'qty': qty, 'sl_type': sl_type, 'sl_value': sl_value, 
    'target_value': target_value, 'exit_on_signal': exit_on_signal
}

# Controls
st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)

if col_start.button("🟢 START", use_container_width=True):
    st.session_state.trading_active = True
    st.session_state.logs.append(f"System Started on {ticker}")
    st.rerun()

if col_stop.button("🔴 STOP", use_container_width=True):
    st.session_state.trading_active = False
    if st.session_state.position:
        # Force Close
        ts = TradingSystem()
        # Fetch current price for close (mocking last close for safety)
        ts.close_position(st.session_state.position['entry_price'], "Force Stop", 0, 0)
    st.session_state.logs.append("System Stopped")
    st.rerun()

# ==========================================
# MAIN LAYOUT
# ==========================================
tab1, tab2, tab3 = st.tabs(["⚡ Live Trading", "📜 Trade History", "📝 System Log"])

with tab1:
    if st.session_state.trading_active:
        st.caption(f"Status: 🟢 LIVE | Ticker: {ticker} | Interval: {timeframe} | Refresh: 1.5s")
        
        system = TradingSystem()
        system.execute(ticker, timeframe, period, strategy_name, strat_params, risk_params)
        
        time.sleep(1.5)
        st.rerun()
    else:
        st.warning("Trading System is OFF. Configure parameters in the sidebar and click START.")
        # Show Static Preview
        df = DataEngine.fetch_data(ticker, timeframe, period)
        if df is not None:
            df = DataEngine.calculate_indicators(df)
            system = TradingSystem()
            system.render_live_dashboard(df, {"Status": "Preview Mode"}, risk_params)

with tab2:
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        # Metrics
        total_trades = len(history_df)
        wins = len(history_df[history_df['PnL'] > 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = history_df['PnL'].sum()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Total PnL", f"{total_pnl:.2f}", delta_color="normal")
        m4.metric("Best Trade", f"{history_df['PnL'].max():.2f}")
        
        st.dataframe(history_df, use_container_width=True)
        
        # Charts
        c1, c2 = st.columns(2)
        c1.plotly_chart(go.Figure(data=[go.Pie(labels=['Win', 'Loss'], values=[wins, total_trades-wins], hole=.3)]), use_container_width=True)
        history_df['Cum PnL'] = history_df['PnL'].cumsum()
        c2.area_chart(history_df['Cum PnL'])
    else:
        st.info("No trades executed yet.")

with tab3:
    st.markdown("### System Logs")
    log_text = "\n".join(st.session_state.logs)
    st.text_area("Log Output", log_text, height=400, disabled=True)
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()
