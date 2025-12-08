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
st.set_page_config(layout="wide", page_title="TradeFlow Pro AI v2", page_icon="⚡")
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
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'paper_trades' not in st.session_state: st.session_state.paper_trades = []
if 'capital' not in st.session_state: st.session_state.capital = 100000.0
if 'live_monitoring' not in st.session_state: st.session_state.live_monitoring = False
if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = None

# --- ASSETS ---
ASSETS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F",
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
    "EUR/USD": "EURUSD=X"
}

# --- DATA FETCHING ---
@st.cache_data(ttl=60) # Short cache for near-live feel
def fetch_data(ticker, interval, period):
    time.sleep(1.0) # Rate limit
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
        df['Z_Score'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        df['Hist_Vol'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        return df.fillna(method='bfill')

    def get_elliott_wave_heuristic(self):
        # Very simplified 5-wave detection (Higher Highs)
        # 1-2-3-4-5 logic: 3 should be strong, 4 shouldn't overlap 1
        closes = self.df['Close'].values[-50:] # Look at last 50 candles
        peaks = signal.argrelextrema(closes, np.greater, order=3)[0]
        valleys = signal.argrelextrema(closes, np.less, order=3)[0]
        
        if len(peaks) >= 3 and len(valleys) >= 2:
            # Check if making higher highs (Impulse)
            if closes[peaks[-1]] > closes[peaks[-2]] > closes[peaks[-3]]:
                return "Impulse Phase (Wave 3 or 5)", "Bullish"
            # Check lower highs (Correction/Downtrend)
            elif closes[peaks[-1]] < closes[peaks[-2]] < closes[peaks[-3]]:
                return "Corrective/Down Phase", "Bearish"
        return "Consolidation/Unclear", "Neutral"

    def get_fibonacci(self):
        # Swing High/Low of last 100 candles
        sub = self.df.iloc[-100:]
        high, low = sub['High'].max(), sub['Low'].min()
        diff = high - low
        levels = {0: high, 0.236: high - 0.236*diff, 0.382: high - 0.382*diff, 
                  0.5: high - 0.5*diff, 0.618: high - 0.618*diff, 1: low}
        
        curr = self.df['Close'].iloc[-1]
        # Find closest
        closest_name, closest_price = min(levels.items(), key=lambda x: abs(x[1] - curr))
        dist_pct = (curr - closest_price) / closest_price * 100
        return levels, (closest_name, closest_price, dist_pct)

    def get_support_resistance(self):
        # Find levels with at least 3 touches within tolerance
        closes = self.df['Close'].values
        peaks = signal.argrelextrema(closes, np.greater, order=5)[0]
        troughs = signal.argrelextrema(closes, np.less, order=5)[0]
        
        levels = list(closes[peaks]) + list(closes[troughs])
        levels.sort()
        
        # Cluster
        clusters = []
        if not levels: return []
        
        temp = [levels[0]]
        for i in range(1, len(levels)):
            if (levels[i] - temp[-1])/temp[-1] < 0.003: # 0.3% tolerance
                temp.append(levels[i])
            else:
                if len(temp) >= 3: # Only strong levels
                    clusters.append(np.mean(temp))
                temp = [levels[i]]
        if len(temp) >= 3: clusters.append(np.mean(temp))
        
        return clusters

    def get_historical_correlation(self):
        # Check current 20 candles vs history
        if len(self.df) < 300: return None
        target = self.df['Close'].pct_change().fillna(0).tail(20).values
        
        best_corr = 0
        outcome = 0
        found_date = None
        
        for i in range(len(self.df) - 250):
            window = self.df['Close'].pct_change().fillna(0).iloc[i:i+20].values
            if len(window) != 20: continue
            
            # Simple correlation
            corr = np.corrcoef(target, window)[0,1]
            if corr > best_corr:
                best_corr = corr
                # What happened in next 5 candles?
                future_price = self.df['Close'].iloc[i+25]
                past_price = self.df['Close'].iloc[i+20]
                outcome = (future_price - past_price)/past_price * 100
                found_date = self.df.index[i]
                
        if best_corr > 0.75:
            return {"corr": best_corr, "outcome": outcome, "date": found_date}
        return None

# --- GENERATE NARRATIVE ---
def generate_market_narrative(df, ticker, fib_data, ew_status, sr_levels, hist_data):
    curr = df.iloc[-1]
    trend = "UP" if curr['Close'] > curr['EMA_50'] else "DOWN"
    rsi_status = "Oversold" if curr['RSI'] < 30 else "Overbought" if curr['RSI'] > 70 else "Neutral"
    z_stat = "Extreme (Mean Reversion Likely)" if abs(curr['Z_Score']) > 2.5 else "Normal"
    
    # Fib Context
    fib_name, fib_price, fib_dist = fib_data
    fib_txt = f"Price is reacting to **Fib {fib_name}** ({fib_price:.2f})."
    if abs(fib_dist) < 0.2: fib_txt += " ✅ **Perfect test of this level.**"
    
    # S/R Context
    sr_txt = "No major structural levels nearby."
    nearby_sr = [lvl for lvl in sr_levels if abs((curr['Close'] - lvl)/lvl) < 0.01]
    if nearby_sr:
        sr_txt = f"⚠️ **CRITICAL ZONE:** Trading near strong historical level {nearby_sr[0]:.2f} (touched 3+ times)."

    # Historical Context
    hist_txt = "No clear historical correlation found."
    if hist_data:
        direction = "RALLY" if hist_data['outcome'] > 0 else "DROP"
        hist_txt = f"🔮 **History Match:** Pattern matches {hist_data['date'].date()} ({hist_data['corr']*100:.1f}%). " \
                   f"Market **{direction}ED {abs(hist_data['outcome']):.2f}%** after this setup."

    summary = f"""
    ### 🧠 AI Market Analysis: {ticker}
    **Current Status:** The market is in a **{trend}** trend (Price vs EMA50).
    
    **1. Structure & Levels:**
    * **Elliott Wave:** {ew_status[0]} ({ew_status[1]}).
    * **Fibonacci:** {fib_txt}
    * **Support/Resistance:** {sr_txt}
    
    **2. Momentum & Volatility:**
    * **RSI:** {curr['RSI']:.1f} ({rsi_status}).
    * **Z-Score:** {curr['Z_Score']:.2f} -> {z_stat}.
    * **Volatility:** Current {curr['Hist_Vol']:.1f}% vs Avg {df['Hist_Vol'].mean():.1f}%.
    
    **3. Predictive Model:**
    * {hist_txt}
    
    **🏁 Verdict:**
    """
    
    # Logic for Recommendation
    score = 0
    if trend == "UP": score += 1
    if curr['RSI'] < 30: score += 2 # Buy dip
    if curr['RSI'] > 70: score -= 2 # Sell rip
    if abs(fib_dist) < 0.2 and fib_name in [0.5, 0.618]: score += 1 # Fib bounce
    if hist_data and hist_data['outcome'] > 0: score += 1
    
    rec = "WAIT / HOLD 🟡"
    if score >= 3: rec = "STRONG BUY 🟢"
    elif score <= -2: rec = "STRONG SELL 🔴"
    
    summary += f"## {rec}\n"
    if rec == "STRONG BUY 🟢": summary += f"*Target: {curr['Close']*1.02:.2f} | SL: {curr['Close']*0.99:.2f}*"
    if rec == "STRONG SELL 🔴": summary += f"*Target: {curr['Close']*0.98:.2f} | SL: {curr['Close']*1.01:.2f}*"
    
    return summary

def manage_trade_narrative(trade, current_price, current_rsi, current_z):
    pnl_pct = (current_price - trade['price']) / trade['price'] * 100
    if trade['type'] == 'SELL': pnl_pct *= -1
    
    advice = ""
    # Profit taking logic
    if pnl_pct > 2.0: advice = "💰 **Take Profit Alert:** You are up >2%. Consider closing half."
    elif pnl_pct < -1.5: advice = "🛑 **Stop Loss Alert:** Position down >1.5%. Validate thesis."
    
    # Technical invalidation
    if trade['type'] == 'BUY':
        if current_rsi > 70: advice += " ⚠️ **Warning:** RSI is now Overbought. Momentum might fade."
    
    status = f"""
    **Trade Monitor ({trade['symbol']}):**
    * **Entry:** {trade['price']:.2f} | **Current:** {current_price:.2f}
    * **P&L:** {pnl_pct:.2f}%
    * **Thesis Check:** Entry Z-Score was {trade['entry_z']:.2f}, now {current_z:.2f}.
    * **AI Advice:** {advice if advice else "✅ Hold Position. Trends align."}
    """
    return status

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.selectbox("Asset", list(ASSETS.keys()))
        interval = st.selectbox("Interval", ["1m","5m","15m","1h","1d"], index=2)
        period = st.selectbox("Period", ["1d","5d","1mo","1y"], index=2)
        
        st.divider()
        st.header("💼 Paper Trading")
        st.metric("Buying Power", f"₹{st.session_state.capital:,.2f}")
        
        # Manual Trade Input
        st.subheader("Manual Order")
        order_type = st.selectbox("Type", ["BUY", "SELL"])
        qty = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.1)
        
        if st.button("Submit Order"):
            # Get current price from session or fetch
            # Simplified: Fetching brief data to get price
            price_data = fetch_data(ASSETS[ticker], "1m", "1d")
            if price_data is not None:
                curr_price = price_data['Close'].iloc[-1]
                cost = curr_price * qty
                
                if st.session_state.capital >= cost or order_type == "SELL":
                    st.session_state.paper_trades.append({
                        'time': datetime.now(IST), 'symbol': ticker, 'type': order_type,
                        'qty': qty, 'price': curr_price, 
                        'entry_z': 0, # Placeholder, will update in full analysis
                        'entry_rsi': 0
                    })
                    if order_type == "BUY": st.session_state.capital -= cost
                    else: st.session_state.capital += cost
                    st.success("Order Placed!")
                else:
                    st.error("Insufficient Funds")
        
        st.divider()
        live_mode = st.toggle("🔴 Live Analysis Loop")
        if live_mode: st.session_state.live_monitoring = True
        else: st.session_state.live_monitoring = False

    # --- MAIN PAGE ---
    st.title(f"⚡ TradeFlow Pro: {ticker}")
    
    # 1. Fetch & Analyze
    df = fetch_data(ASSETS[ticker], interval, period)
    if df is not None:
        analyzer = Analyzer(df)
        df = analyzer.add_indicators()
        
        # Calculations
        fibs, fib_data = analyzer.get_fibonacci()
        sr_levels = analyzer.get_support_resistance()
        ew_status = analyzer.get_elliott_wave_heuristic()
        hist_data = analyzer.get_historical_correlation()
        
        current = df.iloc[-1]
        
        # 2. UI Layout
        col_mn, col_side = st.columns([3, 1])
        
        with col_mn:
            # AI Narrative Section
            narrative = generate_market_narrative(df, ticker, fib_data, ew_status, sr_levels, hist_data)
            st.markdown(f"<div class='guidance-box'>{narrative}</div>", unsafe_allow_html=True)
            
            # Charts (Plotly)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            # Price
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='cyan'), name='EMA 50'), row=1, col=1)
            
            # Add Levels to Chart
            for lvl in sr_levels:
                fig.add_hline(y=lvl, line_color="rgba(255,255,0,0.4)", line_dash="dash", row=1, col=1)
            fig.add_hline(y=fib_data[1], line_color="rgba(255,0,255,0.5)", annotation_text=f"Fib {fib_data[0]}", row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_color='red', row=2, col=1)
            fig.add_hline(y=30, line_color='green', row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_side:
            st.subheader("Live Metrics")
            c1 = "green" if current['Close'] > df['Open'].iloc[-1] else "red"
            st.metric("Price", f"{current['Close']:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
            
            st.metric("Z-Score", f"{current['Z_Score']:.2f}", help=">2 Overbought, <-2 Oversold")
            st.metric("Volatility", f"{current['Hist_Vol']:.1f}%")
            
            st.write("---")
            st.write("**Active Position Management**")
            
            # Find trades for this symbol
            my_trades = [t for t in st.session_state.paper_trades if t['symbol'] == ticker] # Simplified for now, usually filter open trades
            
            if my_trades:
                last_trade = my_trades[-1] # Manage latest
                advice = manage_trade_narrative(last_trade, current['Close'], current['RSI'], current['Z_Score'])
                st.info(advice)
                
                if st.button("Close Position"):
                    # P&L Logic
                    pnl = (current['Close'] - last_trade['price']) * last_trade['qty']
                    if last_trade['type'] == 'SELL': pnl *= -1
                    st.session_state.capital += (last_trade['qty'] * current['Close']) + pnl # Rough logic return capital
                    st.session_state.paper_trades.remove(last_trade)
                    st.success(f"Closed P&L: {pnl:.2f}")
                    st.rerun()
            else:
                st.write("No active trades for this asset.")

        # 3. Position Table (Global)
        st.subheader("Global Portfolio Dashboard")
        if st.session_state.paper_trades:
            # Reconstruct DataFrame with Current Prices (Mocking live update for all symbols would require multiple API calls)
            # Here we just show the table. In a full app, we loop fetch all symbols.
            
            p_df = pd.DataFrame(st.session_state.paper_trades)
            # Calculate Unrealized P&L roughly based on current asset price if matches
            p_df['Current Price'] = p_df.apply(lambda x: current['Close'] if x['symbol'] == ticker else x['price'], axis=1)
            
            def calc_pnl(row):
                val = (row['Current Price'] - row['price']) * row['qty']
                return val if row['type'] == 'BUY' else -val
                
            p_df['Unrealized P&L'] = p_df.apply(calc_pnl, axis=1)
            
            # Styling
            st.dataframe(p_df, use_container_width=True)

    # Live Loop
    if st.session_state.live_monitoring:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
