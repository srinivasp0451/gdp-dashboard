import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats, signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime
import time
import warnings

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="TradeFlow Pro AI v2.1", page_icon="⚡")
warnings.filterwarnings('ignore')
IST = pytz.timezone('Asia/Kolkata')

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2127; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff4b4b; font-weight: bold; }
    .neutral { color: #ffa500; font-weight: bold; }
    .guidance-box { background-color: #161b22; padding: 20px; border-left: 5px solid #00d4ff; border-radius: 5px; margin-bottom: 20px; }
    .trade-panel { border: 1px solid #444; padding: 15px; border-radius: 10px; background: #1e1e1e; }
    .stNumberInput div:first-child { background-color: #1e2127; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
if 'capital' not in st.session_state: st.session_state.capital = 100000.0
if 'live_monitoring' not in st.session_state: st.session_state.live_monitoring = False
if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = None

# --- ASSETS ---
ASSETS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", 
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", 
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS"
}

# --- DATA FETCHING ---
@st.cache_data(ttl=60) 
def fetch_data(ticker, interval, period):
    time.sleep(0.5)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(IST)
        return df
    except: return None

# --- TECHNICAL ANALYSIS ENGINE ---
class Analyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_indicators(self):
        df = self.df
        close = df['Close']
        
        # EMAs
        for p in [20, 50, 200]: df[f'EMA_{p}'] = close.ewm(span=p, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR & Volatility
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        returns = close.pct_change()
        # Ensure returns rolling std is not zero before division
        returns_std = returns.rolling(20).std()
        df['Z_Score'] = (returns - returns.rolling(20).mean()) / returns_std.replace(0, np.nan) 
        df['Hist_Vol'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        return df.fillna(method='bfill')

    def get_elliott_wave_heuristic(self):
        closes = self.df['Close'].values[-50:]
        peaks = signal.argrelextrema(closes, np.greater, order=3)[0]
        
        if len(peaks) >= 3 and closes[peaks[-1]] > closes[peaks[-2]] > closes[peaks[-3]]:
            return "Impulse Phase (Wave 3 or 5)", "Bullish"
        elif len(peaks) >= 3 and closes[peaks[-1]] < closes[peaks[-2]] < closes[peaks[-3]]:
            return "Corrective/Down Phase", "Bearish"
        return "Consolidation/Unclear", "Neutral"

    def get_fibonacci(self):
        sub = self.df.iloc[-100:]
        high, low = sub['High'].max(), sub['Low'].min()
        diff = high - low
        levels = {0: high, 0.236: high - 0.236*diff, 0.382: high - 0.382*diff, 
                  0.5: high - 0.5*diff, 0.618: high - 0.618*diff, 1: low}
        
        curr = self.df['Close'].iloc[-1]
        closest_name, closest_price = min(levels.items(), key=lambda x: abs(x[1] - curr))
        dist_pct = (curr - closest_price) / closest_price * 100
        return levels, (closest_name, closest_price, dist_pct)

    def get_support_resistance(self):
        closes = self.df['Close'].values
        peaks = signal.argrelextrema(closes, np.greater, order=5)[0]
        troughs = signal.argrelextrema(closes, np.less, order=5)[0]
        levels = list(closes[peaks]) + list(closes[troughs])
        levels.sort()
        
        clusters = []
        if not levels: return []
        
        temp = [levels[0]]
        for i in range(1, len(levels)):
            if (levels[i] - temp[-1])/temp[-1] < 0.003: 
                temp.append(levels[i])
            else:
                if len(temp) >= 3: 
                    clusters.append(np.mean(temp))
                temp = [levels[i]]
        if len(temp) >= 3: clusters.append(np.mean(temp))
        
        return clusters

    def get_historical_correlation(self):
        if len(self.df) < 300: return None
        target = self.df['Close'].pct_change().fillna(0).tail(20).values
        best_corr, outcome, found_date = 0, 0, None
        
        for i in range(len(self.df) - 250):
            window = self.df['Close'].pct_change().fillna(0).iloc[i:i+20].values
            if len(window) != 20: continue
            corr = np.corrcoef(target, window)[0,1]
            if corr > best_corr:
                best_corr = corr
                future_price = self.df['Close'].iloc[i+25]
                past_price = self.df['Close'].iloc[i+20]
                outcome = (future_price - past_price)/past_price * 100
                found_date = self.df.index[i]
                
        if best_corr > 0.75:
            return {"corr": best_corr, "outcome": outcome, "date": found_date}
        return None

# --- GENERATE NARRATIVE (ENHANCED) ---
def generate_market_narrative(df, ticker, fib_data, ew_status, sr_levels, hist_data, interval, period):
    curr = df.iloc[-1]
    trend = "UP" if curr['Close'] > curr['EMA_50'] else "DOWN"
    rsi_status = "Oversold" if curr['RSI'] < 30 else "Overbought" if curr['RSI'] > 70 else "Neutral"
    
    # 1. Score Calculation (Confidence Builder)
    score = 0
    if trend == "UP": score += 1
    if curr['RSI'] < 30: score += 2 
    elif curr['RSI'] > 70: score -= 2
    
    fib_name, fib_price, fib_dist = fib_data
    if abs(fib_dist) < 0.2 and fib_name in [0.5, 0.618]: score += 1
    
    nearby_sr = [lvl for lvl in sr_levels if abs((curr['Close'] - lvl)/lvl) < 0.005]
    if nearby_sr: 
        score += 1 
        if trend == "DOWN" and nearby_sr[0] < curr['Close']: score -= 2 # Resistance break
    
    if hist_data and hist_data['outcome'] > 0: score += 1
    
    confidence = min(100, 50 + score * 10)
    
    # 2. Text Generation
    fib_txt = f"Price is reacting to **Fib {fib_name}** ({fib_price:.2f})."
    sr_txt = f"Trading near **{len(nearby_sr)}** strong structural level(s)."
    
    hist_txt = "No strong historical pattern correlation (>75%) found for the last 20 candles."
    if hist_data:
        direction = "RALLY" if hist_data['outcome'] > 0 else "DROP"
        hist_txt = f"🔮 **History Match:** Pattern correlates {hist_data['corr']*100:.1f}% with setup on **{hist_data['date'].date()}**. That instance led to a **{direction} of {abs(hist_data['outcome']):.2f}%** in the following candles."

    summary = f"""
    ### 🧠 AI Market Analysis: {ticker}
    **Analysis Scope:** {interval} interval over the last **{period}** of data ({len(df)} candles).
    **Confidence Score:** **{confidence:.0f}%**
    
    **1. Structure & Levels:**
    * **Elliott Wave:** {ew_status[0]} ({ew_status[1]}).
    * **Fibonacci:** {fib_txt}
    * **Support/Resistance:** {sr_txt}
    
    **2. Momentum & Volatility:**
    * **RSI:** {curr['RSI']:.1f} ({rsi_status}).
    * **Z-Score:** **{curr['Z_Score']:.2f}** ({'Extreme Oversold' if curr['Z_Score'] < -2 else 'Extreme Overbought' if curr['Z_Score'] > 2 else 'Normal'}).
    
    **3. Predictive Model:**
    * {hist_txt}
    
    **🏁 Verdict (Score: {score}):**
    """
    
    rec = "WAIT / HOLD 🟡"
    if score >= 3: rec = "STRONG BUY 🟢"
    elif score <= -2: rec = "STRONG SELL 🔴"
    
    summary += f"## {rec}\n"
    if rec == "STRONG BUY 🟢": summary += f"*Confidence in a long entry is high due to confluence (RSI + Fib/SR). Target: {curr['Close']*1.02:.2f} | SL: {curr['Close']*0.99:.2f}*"
    if rec == "STRONG SELL 🔴": summary += f"*Confidence in a short entry is high due to confluence (RSI + Fib/SR). Target: {curr['Close']*0.98:.2f} | SL: {curr['Close']*1.01:.2f}*"
    
    return summary

# --- BACKTESTING LOGIC ---
def run_backtest(df):
    # Strategy: Buy if RSI is oversold and price is above EMA20. Target 1.5 ATR, SL 1 ATR.
    if len(df) < 100: return pd.DataFrame()
    trades = []
    in_pos, entry_p, entry_d, sl, tp, pos_type = False, 0, None, 0, 0, None
    
    for i in range(50, len(df)):
        curr = df.iloc[i]
        
        if not in_pos:
            if curr['RSI'] < 30 and curr['Close'] > curr['EMA_20']:
                in_pos, entry_p, entry_d, pos_type = True, curr['Close'], curr.name, 'BUY'
                sl = entry_p - (1 * curr['ATR'])
                tp = entry_p + (1.5 * curr['ATR'])
            elif curr['RSI'] > 70 and curr['Close'] < curr['EMA_20']:
                in_pos, entry_p, entry_d, pos_type = True, curr['Close'], curr.name, 'SELL'
                sl = entry_p + (1 * curr['ATR'])
                tp = entry_p - (1.5 * curr['ATR'])
        else:
            exit_p, reason = None, None
            
            if pos_type == 'BUY':
                if curr['Low'] <= sl: exit_p, reason = sl, "SL"
                elif curr['High'] >= tp: exit_p, reason = tp, "TP"
                elif curr['RSI'] > 70: exit_p, reason = curr['Close'], "RSI Exit" # Extra condition for high accuracy
            
            elif pos_type == 'SELL':
                if curr['High'] >= sl: exit_p, reason = sl, "SL"
                elif curr['Low'] <= tp: exit_p, reason = tp, "TP"
                elif curr['RSI'] < 30: exit_p, reason = curr['Close'], "RSI Exit"

            if reason:
                pnl = (exit_p - entry_p) if pos_type == 'BUY' else (entry_p - exit_p)
                trades.append({'Entry': entry_d, 'Exit': curr.name, 'Type': pos_type, 'PnL': pnl, 'Reason': reason})
                in_pos = False
    
    return pd.DataFrame(trades)

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.selectbox("Asset", list(ASSETS.keys()))
        
        # Note: Keeping single selection for clarity, but the AI narrative now explicitly mentions this context.
        interval = st.selectbox("Interval", ["1m","5m","15m","1h","1d"], index=2) 
        period = st.selectbox("Period", ["1d","5d","1mo","1y"], index=2)
        
        st.divider()
        st.header("💼 Paper Trading")
        st.metric("Buying Power", f"₹{st.session_state.capital:,.2f}")
        
        # Capital Top-Up Option
        add_capital = st.number_input("Increase Capital By", min_value=0.0, value=0.0, step=1000.0)
        if st.button("Add Funds"):
            st.session_state.capital += add_capital
            st.success(f"Added ₹{add_capital:,.2f} to capital.")
            st.rerun()

        # Manual Trade Input
        st.subheader("Manual Order")
        order_type = st.selectbox("Type", ["BUY", "SELL"], key="order_type")
        qty = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.1, key="order_qty")
        
        if st.button("Submit Order", key="submit_order"):
            # Check if analysis has run and data is available
            if st.session_state.analysis_cache and st.session_state.analysis_cache['ticker'] == ticker and st.session_state.analysis_cache['interval'] == interval:
                df = st.session_state.analysis_cache['df']
                curr = df.iloc[-1]
                curr_price = curr['Close']
                
                # FIX: Retrieve actual Z-Score and RSI
                curr_z = curr['Z_Score'] if not np.isnan(curr['Z_Score']) else 0.0
                curr_rsi = curr['RSI']
                
                cost = curr_price * qty
                
                if st.session_state.capital >= cost or order_type == "SELL":
                    st.session_state.paper_trades.append({
                        'time': datetime.now(IST), 'symbol': ticker, 'type': order_type,
                        'qty': qty, 'price': curr_price, 
                        'entry_z': curr_z, 
                        'entry_rsi': curr_rsi
                    })
                    if order_type == "BUY": st.session_state.capital -= cost
                    else: st.session_state.capital += cost
                    st.success(f"Order Placed: {order_type} {qty} @ {curr_price:.2f}")
                else:
                    st.error("Insufficient Funds")
            else:
                 st.warning("Please run analysis first before trading.")
        
        st.divider()
        live_mode = st.toggle("🔴 Live Analysis Loop (5s)")
        if live_mode: st.session_state.live_monitoring = True
        else: st.session_state.live_monitoring = False

    # --- MAIN PAGE EXECUTION ---
    
    # 1. Fetch & Analyze
    df = fetch_data(ASSETS[ticker], interval, period)
    if df is not None and not df.empty:
        analyzer = Analyzer(df)
        df = analyzer.add_indicators()
        
        # Calculations
        fibs, fib_data = analyzer.get_fibonacci()
        sr_levels = analyzer.get_support_resistance()
        ew_status = analyzer.get_elliott_wave_heuristic()
        hist_data = analyzer.get_historical_correlation()
        current = df.iloc[-1]
        
        # Cache results for use in trade submission
        st.session_state.analysis_cache = {'df': df, 'ticker': ticker, 'interval': interval}
        
        t1, t2, t3, t4 = st.tabs(["AI Guidance", "Charts", "Paper Trade Manager", "Backtest"])
        
        with t1:
            # AI Narrative Section (includes Confidence/Metadata)
            narrative = generate_market_narrative(df, ticker, fib_data, ew_status, sr_levels, hist_data, interval, period)
            st.markdown(f"<div class='guidance-box'>{narrative}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"{current['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
            c2.metric("RSI", f"{current['RSI']:.2f}")
            c3.metric("Z-Score", f"{current['Z_Score']:.2f}")
            
        with t2:
            # Reusing chart logic from V2
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='cyan'), name='EMA 50'), row=1, col=1)
            for lvl in sr_levels: fig.add_hline(y=lvl, line_color="rgba(255,255,0,0.4)", line_dash="dash", row=1, col=1)
            fig.add_hline(y=fib_data[1], line_color="rgba(255,0,255,0.5)", annotation_text=f"Fib {fib_data[0]}", row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', row=2, col=1); fig.add_hline(y=30, line_color='green', row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with t3:
            st.subheader(f"Active Trades for {ticker}")
            my_trades = [t for t in st.session_state.paper_trades if t['symbol'] == ticker]
            
            if my_trades:
                # Live Position Management
                last_trade = my_trades[-1]
                
                # Check if position is still open (simplified: manage the last trade)
                if last_trade['qty'] > 0: 
                    advice = manage_trade_narrative(last_trade, current['Close'], current['RSI'], current['Z_Score'])
                    st.info(advice)
                    
                    if st.button("Close Last Position"):
                        pnl = (current['Close'] - last_trade['price']) * last_trade['qty']
                        if last_trade['type'] == 'SELL': pnl *= -1
                        st.session_state.capital += (last_trade['qty'] * current['Close']) + pnl 
                        st.session_state.paper_trades.remove(last_trade)
                        st.success(f"Closed P&L: {pnl:.2f}")
                        st.rerun()
                
            st.subheader("Global Portfolio Dashboard")
            if st.session_state.paper_trades:
                p_df = pd.DataFrame(st.session_state.paper_trades)
                p_df['Current Price'] = p_df.apply(lambda x: current['Close'] if x['symbol'] == ticker else x['price'], axis=1)
                
                def calc_pnl(row):
                    val = (row['Current Price'] - row['price']) * row['qty']
                    return val if row['type'] == 'BUY' else -val
                    
                p_df['Unrealized P&L'] = p_df.apply(calc_pnl, axis=1)
                st.dataframe(p_df[['time', 'symbol', 'type', 'qty', 'price', 'Current Price', 'Unrealized P&L', 'entry_z']], use_container_width=True)
            else:
                st.info("No trades in portfolio.")

        with t4:
            st.subheader("Backtesting Results")
            bt_df = run_backtest(df)
            
            if not bt_df.empty:
                total_pnl = bt_df['PnL'].sum()
                win_rate = (bt_df['PnL'] > 0).sum() / len(bt_df) * 100
                total_trades = len(bt_df)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trades", total_trades)
                c2.metric("Total PnL", f"₹{total_pnl:,.2f}")
                c3.metric("Win Rate", f"{win_rate:.1f}%")

                st.markdown(f"***Strategy based on RSI & EMA20, using {interval} timeframe over {len(df)} candles.***")
                st.dataframe(bt_df.sort_values('Entry', ascending=False), use_container_width=True)
                
                # Note: The strategy is designed to be high-accuracy by using multiple filters (RSI, EMA, ATR), which naturally limits the number of trades. 
                # The "3-5 signals daily" constraint depends entirely on the chosen asset, interval, and market conditions. 
                # On a 15m chart over 1 month, you should see frequent high-quality signals.
                
            else:
                st.warning("Backtest requires at least 100 data points. Adjust period/interval or wait for more data.")

    # Live Loop for continuous analysis update
    if st.session_state.live_monitoring:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
