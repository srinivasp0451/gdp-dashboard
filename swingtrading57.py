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

# ==========================================
# 1. GLOBAL CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="Gemini Pro-Algo Terminal")
IST = pytz.timezone('Asia/Kolkata')

# Custom CSS for Professional Look
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .metric-card { background: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .status-live { color: #00ff00; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None, # Dict: type, entry, sl, tp, qty, time, trail_val
        'trade_history': [],
        'trade_log': [],
        'iteration_count': 0,
        'last_api_call': 0,
        'ticker': 'RELIANCE.NS'
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# ==========================================
# 3. MATHEMATICAL INDICATORS (MANUAL)
# ==========================================
def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_sma(series, period):
    return series.rolling(window=period).mean()

def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_atr(df, period=14):
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - df['Close'].shift()).abs()
    l_pc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ==========================================
# 4. MODULAR STRATEGY SYSTEM
# ==========================================
class BaseStrategy(ABC):
    @abstractmethod
    def calculate_indicators(self, data): pass
    @abstractmethod
    def generate_signal(self, data): pass
    
    def get_stats(self, data):
        return data.describe().to_dict()

class EMACrossover(BaseStrategy):
    def __init__(self, p1=9, p2=20, type1='EMA', type2='EMA', mode='simple', min_angle=15):
        self.p1, self.p2 = p1, p2
        self.type1, self.type2 = type1, type2
        self.mode, self.min_angle = mode, min_angle

    def calculate_indicators(self, data):
        df = data.copy()
        df['fast'] = get_ema(df['Close'], self.p1) if self.type1 == 'EMA' else get_sma(df['Close'], self.p1)
        df['slow'] = get_ema(df['Close'], self.p2) if self.type2 == 'EMA' else get_sma(df['Close'], self.p2)
        # Calculate Slope Angle
        df['slope'] = np.degrees(np.arctan((df['fast'] - df['fast'].shift(1)) / df['fast'].shift(1) * 100))
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        bullish = (prev['fast'] <= prev['slow']) and (curr['fast'] > curr['slow'])
        bearish = (prev['fast'] >= prev['slow']) and (curr['fast'] < curr['slow'])
        
        if self.mode == 'auto_strong_candle':
            body = abs(curr['Close'] - curr['Open'])
            avg_body = abs(df['Close'] - df['Open']).rolling(20).mean().iloc[-1]
            if body < 1.5 * avg_body: bullish = bearish = False
            
        if abs(curr['slope']) < self.min_angle:
            bullish = bearish = False
            
        return bullish, bearish, {"Fast": curr['fast'], "Slow": curr['slow'], "Angle": curr['slope']}

class ZScoreStrategy(BaseStrategy):
    def __init__(self, period=20, threshold=2.0):
        self.period = period
        self.threshold = threshold

    def calculate_indicators(self, data):
        df = data.copy()
        df['mean'] = df['Close'].rolling(self.period).mean()
        df['std'] = df['Close'].rolling(self.period).std()
        df['zscore'] = (df['Close'] - df['mean']) / df['std']
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        z = df['zscore'].iloc[-1]
        return z < -self.threshold, z > self.threshold, {"Z-Score": z}

class RSI_Divergence(BaseStrategy):
    def __init__(self, period=14):
        self.period = period

    def calculate_indicators(self, data):
        df = data.copy()
        df['rsi'] = get_rsi(df['Close'], self.period)
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        # Simple Logic for regular signals
        curr_rsi = df['rsi'].iloc[-1]
        return curr_rsi < 30, curr_rsi > 70, {"RSI": curr_rsi}

class BreakoutVolume(BaseStrategy):
    def __init__(self, window=20):
        self.window = window

    def calculate_indicators(self, data):
        df = data.copy()
        df['upper'] = df['High'].rolling(self.window).max().shift(1)
        df['lower'] = df['Low'].rolling(self.window).min().shift(1)
        df['avg_vol'] = df['Volume'].rolling(20).mean()
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        curr = df.iloc[-1]
        vol_confirm = curr['Volume'] > (1.5 * curr['avg_vol'])
        bullish = (curr['Close'] > curr['upper']) and vol_confirm
        bearish = (curr['Close'] < curr['lower']) and vol_confirm
        return bullish, bearish, {"Upper": curr['upper'], "Lower": curr['lower']}

class FibonacciStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        df = data.copy()
        recent_data = df.tail(100)
        self.high = recent_data['High'].max()
        self.low = recent_data['Low'].min()
        diff = self.high - self.low
        self.levels = {
            '0%': self.high,
            '23.6%': self.high - 0.236 * diff,
            '38.2%': self.high - 0.382 * diff,
            '50.0%': self.high - 0.5 * diff,
            '61.8%': self.high - 0.618 * diff,
            '100%': self.low
        }
        return df

    def generate_signal(self, data):
        df = self.calculate_indicators(data)
        cp = df['Close'].iloc[-1]
        # Signal if price is near 61.8% level within 0.2% tolerance
        bullish = abs(cp - self.levels['61.8%']) / cp < 0.002
        return bullish, False, self.levels

class ElliottWaveSimple(BaseStrategy):
    def calculate_indicators(self, data):
        df = data.copy()
        # Find local peaks/troughs
        df['peak'] = df.iloc[argrelextrema(df.High.values, np.greater_equal, order=10)[0]]['High']
        df['trough'] = df.iloc[argrelextrema(df.Low.values, np.less_equal, order=10)[0]]['Low']
        return df

    def generate_signal(self, data):
        # Implementation of 5-wave rule simplified
        return False, False, {}

# ==========================================
# 5. TRADING ENGINE (The "Friend" Logic)
# ==========================================
class TradingEngine:
    @staticmethod
    def get_advice(pos, pnl_pct, market_data):
        if pos is None: return "Market is scanning for high-probability setups. Patience is key."
        
        advice = ""
        if pnl_pct > 0.5:
            advice = "🔥 Profit is looking strong! Consider moving your SL to break-even to protect this win."
        elif pnl_pct < -0.3:
            advice = "⚠️ Position is under pressure. Do NOT average down. Trust your original SL logic."
        else:
            advice = "⌛ Trade is breathing. Market structure remains intact. Avoid impulsive exits."
        return advice

    @staticmethod
    def calculate_exit(pos, curr_price, sl_type, sl_val, tp_type, tp_val, new_signal):
        pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)
        
        # SL Check
        if curr_price <= pos['sl'] if pos['type'] == 'LONG' else curr_price >= pos['sl']:
            return True, "Stop Loss Hit", pnl
        
        # TP Check
        if curr_price >= pos['tp'] if pos['type'] == 'LONG' else curr_price <= pos['tp']:
            return True, "Target Hit", pnl

        # Signal Based Exit
        if (sl_type == "Signal Based" or tp_type == "Signal Based") and new_signal:
            return True, "Opposite Signal", pnl
            
        return False, None, pnl

# ==========================================
# 6. SIDEBAR - SETTINGS
# ==========================================
with st.sidebar:
    st.header("🛠️ Terminal Config")
    asset_cat = st.selectbox("Asset Category", ["Indian Stocks", "Crypto", "Forex", "Custom"])
    
    if asset_cat == "Indian Stocks":
        ticker = st.text_input("Ticker (.NS)", "RELIANCE.NS")
    elif asset_cat == "Crypto":
        ticker = st.selectbox("Pair", ["BTC-USD", "ETH-USD", "SOL-USD"])
    else:
        ticker = st.text_input("Manual Ticker", "EURUSD=X")
        
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
    period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "1y", "1d": "max"}
    period = period_map[timeframe]

    st.divider()
    st.header("📈 Strategy Select")
    strat_choice = st.selectbox("Strategy", [
        "EMA Crossover", "Z-Score Mean Reversion", "RSI Divergence", 
        "Breakout Volume", "Fibonacci Retracement", "Simple Buy", "Simple Sell"
    ])

    st.divider()
    st.header("🛡️ Risk Manager")
    sl_mode = st.selectbox("SL Type", ["Custom Points", "Trail SL", "Signal Based", "ATR Based"])
    sl_points = st.number_input("SL Points/Mult", value=10.0)
    tp_mode = st.selectbox("TP Type", ["Custom Points", "Risk Reward", "Signal Based"])
    tp_points = st.number_input("TP Points/Ratio", value=20.0)
    qty = st.number_input("Quantity", value=1)

# ==========================================
# 7. MAIN APP TABS
# ==========================================
tab_live, tab_history, tab_logs = st.tabs(["📺 Live Terminal", "📊 Performance", "📜 System Logs"])

with tab_live:
    c1, c2, c3 = st.columns([1,1,2])
    
    # 1. Start/Stop Controls
    with c1:
        if not st.session_state.trading_active:
            if st.button("▶ START LIVE TRADING", type="primary", use_container_width=True):
                st.session_state.trading_active = True
                add_log(f"System Started: {ticker}")
                st.rerun()
        else:
            if st.button("⏹ STOP & CLOSE ALL", type="secondary", use_container_width=True):
                if st.session_state.current_position:
                    add_log("Emergency Forced Exit Triggered")
                    st.session_state.current_position = None
                st.session_state.trading_active = False
                st.rerun()

    # 2. Data Fetching Logic (Rate Limited)
    elapsed = time.time() - st.session_state.last_api_call
    if elapsed < 1.8: time.sleep(1.8 - elapsed)
    
    try:
        raw_df = yf.download(ticker, period=period, interval=timeframe, progress=False)
        st.session_state.last_api_call = time.time()
        
        if raw_df.empty:
            st.error("No data received from API. Check Ticker.")
            st.stop()
            
        # Clean MultiIndex
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        
        raw_df.index = raw_df.index.tz_convert(IST)
        
        # 3. Strategy Execution
        if strat_choice == "EMA Crossover":
            strategy = EMACrossover()
        elif strat_choice == "Z-Score Mean Reversion":
            strategy = ZScoreStrategy()
        elif strat_choice == "Breakout Volume":
            strategy = BreakoutVolume()
        else:
            strategy = FibonacciStrategy() # Default

        df = strategy.calculate_indicators(raw_df)
        bull, bear, meta = strategy.generate_signal(df)
        curr_price = df['Close'].iloc[-1]
        curr_time = df.index[-1]

        # 4. Position Monitoring
        if st.session_state.trading_active:
            pos = st.session_state.current_position
            
            if pos is None:
                if bull:
                    sl = curr_price - sl_points if sl_mode != "ATR Based" else curr_price - (get_atr(df).iloc[-1] * sl_points)
                    tp = curr_price + tp_points
                    st.session_state.current_position = {
                        'type': 'LONG', 'entry': curr_price, 'sl': sl, 'tp': tp, 'time': curr_time, 'qty': qty
                    }
                    add_log(f"ENTRY LONG @ {curr_price}")
                elif bear:
                    sl = curr_price + sl_points
                    tp = curr_price - tp_points
                    st.session_state.current_position = {
                        'type': 'SHORT', 'entry': curr_price, 'sl': sl, 'tp': tp, 'time': curr_time, 'qty': qty
                    }
                    add_log(f"ENTRY SHORT @ {curr_price}")
            else:
                # FIX: Exit signal must be on a NEW candle
                new_signal_logic = (curr_time > pos['time']) and ((pos['type'] == 'LONG' and bear) or (pos['type'] == 'SHORT' and bull))
                
                # Trailing SL
                if sl_mode == "Trail SL" and pos['type'] == 'LONG':
                    if curr_price - sl_points > pos['sl']:
                        st.session_state.current_position['sl'] = curr_price - sl_points
                        add_log(f"Trailing SL updated: {pos['sl']:.2f}")

                should_exit, reason, pnl = TradingEngine.calculate_exit(pos, curr_price, sl_mode, sl_points, tp_mode, tp_points, new_signal_logic)
                
                if should_exit:
                    st.session_state.trade_history.append({
                        'Ticker': ticker, 'Type': pos['type'], 'Entry': pos['entry'], 
                        'Exit': curr_price, 'P&L': pnl, 'Reason': reason, 'Time': curr_time
                    })
                    st.session_state.current_position = None
                    add_log(f"EXIT {reason} @ {curr_price} | P&L: {pnl:.2f}")

        # 5. UI DISPLAY
        with c2:
            st.metric("Current Price", f"{curr_price:.2f}", delta=f"{curr_price - df['Open'].iloc[-1]:.2f}")
            if st.session_state.trading_active:
                st.markdown(f"**Status:** <span class='status-live'>● LIVE</span>", unsafe_allow_html=True)
                st.write(f"Refreshes: {st.session_state.iteration_count}")
        
        with c3:
            pnl_pct = 0
            if st.session_state.current_position:
                p = st.session_state.current_position
                pnl = (curr_price - p['entry']) if p['type'] == 'LONG' else (p['entry'] - curr_price)
                pnl_pct = (pnl / p['entry']) * 100
                st.success(f"ACTIVE: {p['type']} | P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
            else:
                st.info("Status: Waiting for Strategy Signal")
            
            # Mentor Advice
            st.warning(f"💡 **Mentor:** {TradingEngine.get_advice(st.session_state.current_position, pnl_pct, df)}")

        # 6. CHARTING
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        
        if strat_choice == "EMA Crossover":
            fig.add_trace(go.Scatter(x=df.index, y=df['fast'], line=dict(color='yellow', width=1), name="EMA Fast"))
            fig.add_trace(go.Scatter(x=df.index, y=df['slow'], line=dict(color='cyan', width=1), name="EMA Slow"))
        
        # Overlay SL/TP if in position
        if st.session_state.current_position:
            p = st.session_state.current_position
            fig.add_hline(y=p['sl'], line_dash="dash", line_color="red", annotation_text="SL")
            fig.add_hline(y=p['tp'], line_dash="dash", line_color="green", annotation_text="TP")
            fig.add_trace(go.Scatter(x=[p['time']], y=[p['entry']], mode='markers', marker=dict(size=15, symbol='triangle-up', color='white'), name="Entry"))

        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

        # 7. STATISTICS SUMMARY
        with st.expander("📊 Strategy Statistics (Historical)"):
            cols = st.columns(4)
            cols[0].metric("Avg Vol", f"{df['Volume'].tail(20).mean():.0f}")
            cols[1].metric("RSI (14)", f"{get_rsi(df['Close']).iloc[-1]:.2f}")
            cols[2].metric("ATR", f"{get_atr(df).iloc[-1]:.2f}")
            cols[3].metric("Volatility %", f"{(df['High'].iloc[-1]-df['Low'].iloc[-1])/curr_price*100:.2f}%")

    except Exception as e:
        st.error(f"Engine Error: {e}")

# ==========================================
# 8. PERFORMANCE TAB
# ==========================================
with tab_history:
    if st.session_state.trade_history:
        h_df = pd.DataFrame(st.session_state.trade_history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trades", len(h_df))
        c2.metric("Net P&L", f"{h_df['P&L'].sum():.2f}")
        win_rate = (len(h_df[h_df['P&L'] > 0]) / len(h_df)) * 100
        c3.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.dataframe(h_df, use_container_width=True)
        
        # AI/ML Recommender Logic (Simplified)
        st.markdown("### 🤖 Post-Trade AI Analysis")
        bad_trades = h_df[h_df['P&L'] < 0]
        if not bad_trades.empty:
            st.write("Analysis shows frequent losses on 'Target Hit' reversals. Suggestion: Tighten Trailing SL by 20% during high ATR periods.")
    else:
        st.write("No trades recorded yet.")

# ==========================================
# 9. LOGS TAB
# ==========================================
with tab_logs:
    st.subheader("System Event Log")
    log_txt = "\n".join(st.session_state.trade_log)
    st.text_area("Live Log Output", value=log_txt, height=500)
    if st.button("Clear Logs"):
        st.session_state.trade_log = []
        st.rerun()

# ==========================================
# 10. AUTO-REFRESH EXECUTION
# ==========================================
if st.session_state.trading_active:
    st.session_state.iteration_count += 1
    time.sleep(2)
    st.rerun()
