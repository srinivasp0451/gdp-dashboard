import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# -----------------------------------------------------------------------------
# 1. SYSTEM CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Quant Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0e11; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
    .stButton>button { width: 100%; background-color: #2962ff; color: white; border-radius: 4px; border: none; font-weight: 600; padding: 0.75rem; transition: all 0.3s; }
    .stButton>button:hover { background-color: #0039cb; box-shadow: 0 4px 12px rgba(41, 98, 255, 0.4); }
    
    /* Signal Banners */
    .signal-banner { padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; animation: fadeIn 1s; }
    .sig-buy { background: linear-gradient(135deg, #004d40 0%, #00695c 100%); border: 1px solid #00e676; box-shadow: 0 0 20px rgba(0, 230, 118, 0.2); }
    .sig-sell { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); border: 1px solid #ff5252; box-shadow: 0 0 20px rgba(255, 82, 82, 0.2); }
    .sig-hold { background: linear-gradient(135deg, #37474f 0%, #455a64 100%); border: 1px solid #90a4ae; }
    
    /* Text Typography */
    .big-signal { font-size: 36px; font-weight: 800; letter-spacing: 2px; margin: 0; }
    .sub-signal { font-size: 14px; opacity: 0.8; font-style: italic; }
    .report-text { font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.7; background: #181b21; padding: 20px; border-radius: 8px; border-left: 4px solid #2962ff; }
    .working-pattern { border-left: 4px solid #00e676; padding: 10px; background: #111; margin: 5px 0; }
    .failing-pattern { border-left: 4px solid #ff5252; padding: 10px; background: #111; margin: 5px 0; }
    
    /* Metrics */
    .kpi-card { background: #1e2228; padding: 15px; border-radius: 8px; border: 1px solid #333; }
    .kpi-val { font-size: 18px; font-weight: bold; color: #fff; }
    .kpi-lbl { font-size: 12px; color: #888; text-transform: uppercase; }
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & DATA LAYER
# -----------------------------------------------------------------------------
ASSETS = {
    "Indices": {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX", "^GSPC": "S&P 500"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
    "Stocks": {"RELIANCE.NS": "Reliance", "HDFCBANK.NS": "HDFC Bank", "TCS.NS": "TCS", "AAPL": "Apple", "NVDA": "Nvidia"},
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver"},
    "Forex": {"INR=X": "USD/INR"}
}

BENCHMARKS = {
    "^NSEI": "INR=X",      # Nifty -> USDINR
    "^NSEBANK": "INR=X",
    "BTC-USD": "GC=F",     # BTC -> Gold
    "ETH-USD": "BTC-USD",  # ETH -> BTC
    "GC=F": "^GSPC"        # Gold -> S&P 500
}

# -----------------------------------------------------------------------------
# 3. ADVANCED ANALYTICS ENGINE
# -----------------------------------------------------------------------------
class DeepMarketAnalyzer:
    def __init__(self, ticker, df, timeframe):
        self.ticker = ticker
        self.df = df
        self.tf = timeframe
        # Note: self.last is NOT initialized here to avoid stale data reference
        
    def calculate_metrics(self):
        """Calculates indicators and updates self.df"""
        df = self.df.copy()
        
        # --- 1. Z-Score (Mean Reversion) ---
        df['Mean'] = df['Close'].rolling(20).mean()
        df['Std'] = df['Close'].rolling(20).std()
        df['Z_Score'] = (df['Close'] - df['Mean']) / df['Std']
        
        # --- 2. Volatility Regimes ---
        df['TR'] = np.maximum((df['High'] - df['Low']), abs(df['Close'].shift(1) - df['High']))
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # --- 3. EMAs ---
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # --- 4. RSI ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # --- 5. MACD (New) ---
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        self.df = df
        return df

    def get_market_memory(self):
        """Greedy Search: Finds historical patterns matching current state."""
        df = self.df
        if len(df) < 50: return {"z_occurrences": 0, "z_success_rate": 0, "examples": []}
        
        # Safe access to last row
        last_row = df.iloc[-1]
        z_current = last_row['Z_Score']
        
        # Search parameters
        tolerance = 0.25
        similar_z = df[(df['Z_Score'] > z_current - tolerance) & (df['Z_Score'] < z_current + tolerance)]
        total_occurrences = len(similar_z)
        
        reverted_count = 0
        examples = []
        
        for date in similar_z.index:
            if date == df.index[-1]: continue # Skip current candle
            
            # Look 5 bars ahead
            try:
                idx = df.index.get_loc(date)
                if idx + 5 < len(df):
                    future_price = df.iloc[idx+5]['Close']
                    price_at_signal = df.iloc[idx]['Close']
                    
                    outcome = "Neutral"
                    if z_current < -1.5: # Oversold condition
                        if future_price > price_at_signal:
                            reverted_count += 1
                            outcome = "Success (Bounce)"
                    elif z_current > 1.5: # Overbought condition
                        if future_price < price_at_signal:
                            reverted_count += 1
                            outcome = "Success (Drop)"
                            
                    if len(examples) < 3: # Keep top 3 for display
                        examples.append({
                            "date": str(date),
                            "price": price_at_signal,
                            "outcome": outcome
                        })
            except:
                pass
                        
        success_rate = (reverted_count / total_occurrences * 100) if total_occurrences > 0 else 0
        
        return {
            "z_val": z_current,
            "z_occurrences": total_occurrences,
            "z_reversions": reverted_count,
            "z_success_rate": success_rate,
            "examples": examples
        }

    def get_support_resistance_advanced(self):
        """Finds S/R levels and counts how many times they were tested."""
        df = self.df
        window = 30 # Lookback
        if len(df) < window: return {}
        
        recent_data = df.iloc[-window:]
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        
        # Find exact time of High/Low
        high_idx = recent_data['High'].idxmax()
        low_idx = recent_data['Low'].idxmin()
        
        now = df.index[-1]
        
        def time_diff(t1, t2):
            diff = t2 - t1
            return str(diff).split('.')[0] # Remove microseconds
            
        # Count Tests (How many candles touched near this level)
        tolerance = 0.002 # 0.2% tolerance
        res_tests = len(recent_data[recent_data['High'] >= recent_high * (1 - tolerance)])
        sup_tests = len(recent_data[recent_data['Low'] <= recent_low * (1 + tolerance)])
            
        return {
            "resistance": recent_high,
            "res_age": time_diff(high_idx, now),
            "res_tests": res_tests,
            "support": recent_low,
            "sup_age": time_diff(low_idx, now),
            "sup_tests": sup_tests,
            "breakout_check": df.iloc[-1]['Close'] > recent_high
        }

    def get_elliott_wave_estimate(self):
        """Heuristic EW: Identifies 1-2 setup and projects Wave 3."""
        df = self.df
        
        # Use simple local min/max for pivots
        n = 5  # number of points to be checked before and after
        df['min'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']
        df['max'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
        
        # Get last 3 pivots (Low -> High -> Higher Low) for 1-2 setup
        pivots = []
        for i in range(len(df)-1, max(0, len(df)-200), -1):
            if not np.isnan(df['max'].iloc[i]):
                pivots.append(('High', df['max'].iloc[i], df.index[i]))
            if not np.isnan(df['min'].iloc[i]):
                pivots.append(('Low', df['min'].iloc[i], df.index[i]))
            if len(pivots) >= 3:
                break
                
        # We need Low (Start) -> High (Wave 1 Top) -> Higher Low (Wave 2 Bottom)
        # pivots list is reversed (newest first) -> [Wave 2 Low, Wave 1 High, Start Low]
        
        if len(pivots) == 3 and pivots[0][0] == 'Low' and pivots[1][0] == 'High' and pivots[2][0] == 'Low':
            w2_low = pivots[0][1]
            w1_high = pivots[1][1]
            start_low = pivots[2][1]
            
            # Check Validity
            if w2_low > start_low and w2_low < w1_high:
                wave_1_height = w1_high - start_low
                wave_3_target = w2_low + (1.618 * wave_1_height)
                
                return {
                    "pattern": "Wave 3 Impulse (In Progress)",
                    "w1_start": start_low,
                    "w1_end": w1_high,
                    "w2_end": w2_low,
                    "w3_target": wave_3_target,
                    "status": "Bullish"
                }
        
        # Bearish Setup
        if len(pivots) == 3 and pivots[0][0] == 'High' and pivots[1][0] == 'Low' and pivots[2][0] == 'High':
            w2_high = pivots[0][1]
            w1_low = pivots[1][1]
            start_high = pivots[2][1]
            
            if w2_high < start_high and w2_high > w1_low:
                wave_1_height = start_high - w1_low
                wave_3_target = w2_high - (1.618 * wave_1_height)
                
                return {
                    "pattern": "Wave 3 Down (In Progress)",
                    "w1_start": start_high,
                    "w1_end": w1_low,
                    "w2_end": w2_high,
                    "w3_target": wave_3_target,
                    "status": "Bearish"
                }

        return {"pattern": "Corrective / Noise", "status": "Neutral", "w3_target": 0}

# -----------------------------------------------------------------------------
# 4. DATA FETCHING (Robust)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Convert to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 5. UI & REPORT GENERATION
# -----------------------------------------------------------------------------
def generate_trade_card(signal, df, ew, sr, score):
    last = df.iloc[-1]
    atr = last['ATR']
    
    # Trade Logic
    action = "WAIT"
    entry = 0.0
    sl = 0.0
    tgt = 0.0
    trailing = 0.0
    
    if "BUY" in signal:
        action = "LONG ENTRY"
        entry = last['Close']
        sl = sr['support'] if sr else last['Close'] - (2 * atr)
        tgt = ew['w3_target'] if ew['status'] == 'Bullish' else last['Close'] + (3 * atr)
        trailing = last['Close'] - atr
    elif "SELL" in signal:
        action = "SHORT ENTRY"
        entry = last['Close']
        sl = sr['resistance'] if sr else last['Close'] + (2 * atr)
        tgt = ew['w3_target'] if ew['status'] == 'Bearish' else last['Close'] - (3 * atr)
        trailing = last['Close'] + atr
        
    return action, entry, sl, tgt, trailing

def main():
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("🎛️ Quant Control")
    
    # 1. Asset Selection
    cat = st.sidebar.selectbox("Market Category", list(ASSETS.keys()))
    ticker_name = st.sidebar.selectbox("Select Asset", list(ASSETS[cat].values()))
    ticker_sym = [k for k,v in ASSETS[cat].items() if v == ticker_name][0]
    
    # 2. Time Control
    c1, c2 = st.sidebar.columns(2)
    with c1:
        sel_interval = st.selectbox("Interval", ['1m', '5m', '15m', '1h', '1d'], index=1)
    with c2:
        sel_period = st.selectbox("Period", ['1d', '5d', '1mo', '3mo', '1y'], index=2)
    
    # 3. Benchmark Logic
    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis", value=False)
    
    # Auto-detect default benchmark
    default_bench = BENCHMARKS.get(ticker_sym, "^NSEI") 
    bench_sym = st.sidebar.selectbox("Benchmark Ticker", list(ASSETS['Indices'].keys()) + list(ASSETS['Commodities'].keys()) + list(ASSETS['Forex'].keys()), index=0 if not enable_ratio else 0)
    
    # Override logic: If default_bench is in the list, try to find its index to set default
    # Note: Streamlit selectbox default index is static on render, handled via smart defaults above if needed.
    # For now, user manually selects, but we guided them with BENCHMARKS dict logic mentally.
    if enable_ratio:
        st.sidebar.caption(f"Recommended Benchmark: {default_bench}")

    run_btn = st.sidebar.button("RUN FULL ANALYSIS")
    
    # --- MAIN EXECUTION ---
    if run_btn:
        with st.spinner(f"🚀 Analyzing {ticker_name} on {sel_interval} timeframe..."):
            
            # Fetch Data
            df = fetch_data(ticker_sym, sel_period, sel_interval)
            
            if df is None:
                st.error("Data fetch failed. Please try a different timeframe or ticker.")
                return
            
            # Analyze
            engine = DeepMarketAnalyzer(ticker_sym, df, sel_interval)
            df = engine.calculate_metrics() # Updates internal df
            
            mem = engine.get_market_memory()
            ew = engine.get_elliott_wave_estimate()
            sr = engine.get_support_resistance_advanced()
            
            # Ratio Analysis
            ratio_data = None
            if enable_ratio:
                df_bench = fetch_data(bench_sym, sel_period, sel_interval)
                if df_bench is not None:
                    # Align indices
                    common_idx = df.index.intersection(df_bench.index)
                    ratio_data = df.loc[common_idx]['Close'] / df_bench.loc[common_idx]['Close']
            
            # --- SIGNAL GENERATION ---
            last_row = df.iloc[-1]
            score = 0
            
            # Trend Check
            trend_status = "SIDEWAYS"
            if last_row['Close'] > last_row['EMA_50']:
                score += 1
                trend_status = "BULLISH"
            elif last_row['Close'] < last_row['EMA_50']:
                score -= 1
                trend_status = "BEARISH"
                
            # Momentum Check
            if last_row['RSI'] < 30: score += 1 # Oversold bounce
            if last_row['RSI'] > 70: score -= 1 # Overbought drop
            
            # Pattern Check
            if ew['status'] == "Bullish": score += 2
            if ew['status'] == "Bearish": score -= 2
            
            # Final Signal
            signal = "NEUTRAL / HOLD"
            if score >= 2: signal = "STRONG BUY"
            elif score <= -2: signal = "STRONG SELL"
            
            # Trade Plan
            action, entry, sl, tgt, trail = generate_trade_card(signal, df, ew, sr, score)
            
            # --- DISPLAY DASHBOARD ---
            
            # 1. Signal Header
            sig_cls = "sig-buy" if "BUY" in signal else "sig-sell" if "SELL" in signal else "sig-hold"
            st.markdown(f"""
            <div class='signal-banner {sig_cls}'>
                <p class='sub-signal'>AI VERDICT ({sel_interval})</p>
                <p class='big-signal'>{signal}</p>
                <p class='sub-signal'>Current Price: {last_row['Close']:.2f} | Trend: {trend_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Main Columns
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                st.markdown("### ✅ Pattern Reconnaissance")
                
                # A. What is Working (Market Memory)
                if mem['z_occurrences'] > 0:
                    status_cls = "working-pattern" if mem['z_success_rate'] > 50 else "failing-pattern"
                    st.markdown(f"""
                    <div class='{status_cls}'>
                        <b>Z-Score Mean Reversion</b><br>
                        Current Z: {mem['z_val']:.2f}<br>
                        History: Detected {mem['z_occurrences']} similar setups.<br>
                        Success Rate: {mem['z_success_rate']:.1f}% (Reverted within 5 bars).
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No statistically significant Z-Score pattern found in recent history.")
                
                # B. Support/Resistance Intelligence
                st.markdown(f"""
                <div class='kpi-card'>
                    <b>Key Levels (Tested)</b><br>
                    🔴 Res: {sr['resistance']:.2f} (Tested {sr['res_tests']} times, {sr['res_age']} ago)<br>
                    🟢 Sup: {sr['support']:.2f} (Tested {sr['sup_tests']} times, {sr['sup_age']} ago)
                </div>
                """, unsafe_allow_html=True)
                
                # C. Smart Trade Plan
                if "HOLD" not in signal:
                    st.markdown(f"""
                    <div class='kpi-card' style='border-left: 4px solid #2962ff;'>
                        <b>🛡️ Smart Trade Plan</b><br>
                        Entry: {entry:.2f}<br>
                        Stop Loss: <span style='color:#ff5252'>{sl:.2f}</span><br>
                        Target: <span style='color:#00e676'>{tgt:.2f}</span><br>
                        <i>Trailing Stop Trigger: {trail:.2f}</i>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_right:
                # D. Advanced Charting
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                                    vertical_spacing=0.05,
                                    subplot_titles=("Price Structure & Waves", "Momentum (RSI)", "MACD"))
                
                # 1. Price
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'), row=1, col=1)
                
                # Wave Projection
                if ew['status'] != 'Neutral':
                    fig.add_hline(y=ew['w3_target'], line_dash="dot", line_color="#00e676", annotation_text="Wave 3 Tgt")
                    fig.add_hline(y=ew['w2_end'], line_dash="dot", line_color="#ff5252", annotation_text="Wave 2 Inv")

                # 2. RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash='dot', line_color='red', row=2, col=1)
                fig.add_hline(y=30, line_dash='dot', line_color='green', row=2, col=1)
                
                # 3. MACD
                fig.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal'], marker_color='gray', name='Hist'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange'), name='Signal'), row=3, col=1)
                
                fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, title=f"{ticker_name} - {sel_interval}")
                st.plotly_chart(fig, use_container_width=True)

            # --- RATIO ANALYSIS SECTION ---
            if enable_ratio and ratio_data is not None:
                st.markdown("### ⚖️ Relative Strength (Ratio Analysis)")
                st.line_chart(ratio_data)
                st.caption(f"Chart shows performance of {ticker_sym} relative to {bench_sym}. Rising line = Outperformance.")

if __name__ == "__main__":
    main()


