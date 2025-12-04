import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import time
from datetime import datetime
import pytz

# ==========================================
# 1. SETUP & STATE
# ==========================================
st.set_page_config(layout="wide", page_title="QuanT Pro: Deep Scan", page_icon="⚡")

# Initialize Session State
if 'balance' not in st.session_state: st.session_state.balance = 100000.0
if 'positions' not in st.session_state: st.session_state.positions = []
if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = None

# Custom CSS
st.markdown("""
<style>
    .metric-box { padding: 10px; background: #1e1e1e; border-radius: 8px; border: 1px solid #333; margin-bottom: 10px; }
    .signal-buy { color: #00ff00; font-weight: 900; font-size: 24px; }
    .signal-sell { color: #ff2a2a; font-weight: 900; font-size: 24px; }
    .signal-hold { color: #aaaaaa; font-weight: 900; font-size: 24px; }
    .report-text { font-family: 'Courier New', monospace; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (MULTI-TIMEFRAME)
# ==========================================

TICKER_MAP = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F", "USD/INR": "INR=X"
}

def get_symbol(t): return TICKER_MAP.get(t, t)

def fetch_data_safe(ticker, period, interval):
    """Fetches data with mandatory delay to respect yfinance limits."""
    time.sleep(2.0) # 2s Delay
    try:
        df = yf.download(get_symbol(ticker), period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tzinfo is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        return df
    except: return None

# ==========================================
# 3. MATH & INDICATORS
# ==========================================

def calculate_indicators(df):
    if df is None or len(df) < 50: return df
    
    # Basic
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility & Z-Score
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(20).std() * np.sqrt(20)
    df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    return df

def get_strong_sr_levels(df):
    """Finds levels hit multiple times."""
    # Local Min/Max
    n = 10 # Window
    df['Min'] = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]
    df['Max'] = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]
    
    levels = []
    # Combine and cluster
    raw_levels = df['Min'].dropna().tolist() + df['Max'].dropna().tolist()
    raw_levels.sort()
    
    if not raw_levels: return []

    current_cluster = [raw_levels[0]]
    for i in range(1, len(raw_levels)):
        if raw_levels[i] <= current_cluster[-1] * 1.005: # 0.5% tolerance
            current_cluster.append(raw_levels[i])
        else:
            if len(current_cluster) >= 3: # Must be hit 3 times to be strong
                levels.append(sum(current_cluster)/len(current_cluster))
            current_cluster = [raw_levels[i]]
            
    return levels

def get_fibonacci(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    return {
        '0.0': high, '0.236': high - 0.236*diff, '0.382': high - 0.382*diff,
        '0.5': high - 0.5*diff, '0.618': high - 0.618*diff, '1.0': low
    }

def detect_elliott_wave(df):
    # Simplified Swing Counter
    # If 3 Higher Highs recently -> Wave 3 or 5 (Impulsive)
    # If Lower Highs -> Correction
    peaks = df['High'].iloc[argrelextrema(df['High'].values, np.greater, order=5)[0]].tail(5)
    if len(peaks) < 3: return "Consolidation/Undefined"
    
    p = peaks.values
    if p[-1] > p[-2] > p[-3]: return "Impulse (Wave 3 or 5)"
    if p[-1] < p[-2] < p[-3]: return "Correction (Wave A or C)"
    return "Complex Consolidation (Wave 4 or B)"

# ==========================================
# 4. BACKTEST & OPTIMIZATION ENGINE
# ==========================================

def optimize_strategy(df):
    """Finds the best EMA Crossover for this specific timeframe."""
    best_roi = -100
    best_pair = (0,0)
    
    pairs = [(9,21), (20,50), (50,200), (10,30)]
    
    for fast, slow in pairs:
        # Vectorized Backtest
        data = df.copy()
        data['Fast'] = data['Close'].ewm(span=fast).mean()
        data['Slow'] = data['Close'].ewm(span=slow).mean()
        data['Signal'] = np.where(data['Fast'] > data['Slow'], 1, -1)
        data['Strat_Ret'] = data['Signal'].shift(1) * data['Log_Ret']
        
        cum_ret = data['Strat_Ret'].sum()
        roi = (np.exp(cum_ret) - 1) * 100
        
        # Annualize approximation based on data length
        days = (data.index[-1] - data.index[0]).days
        if days > 0:
            ann_roi = roi * (365/days)
            if ann_roi > best_roi:
                best_roi = ann_roi
                best_pair = (fast, slow)
                
    return best_pair, best_roi

# ==========================================
# 5. CORE ANALYSIS LOOP
# ==========================================

def run_multi_timeframe_analysis(ticker1, ticker2):
    # Timeframe Config: (Interval, Period)
    config = [
        ("15m", "1mo"), # Short Term
        ("1h", "1y"),   # Medium Term
        ("1d", "5y"),   # Long Term
        ("1wk", "10y")  # Macro
    ]
    
    results = {}
    progress_text = "Initializing Deep Scan..."
    bar = st.progress(0, text=progress_text)
    
    for i, (tf, per) in enumerate(config):
        bar.progress((i+1)/len(config), text=f"Analyzing {tf} Timeframe ({per} data)...")
        
        # 1. Fetch
        df1 = fetch_data_safe(ticker1, per, tf)
        if df1 is None: continue
        
        # 2. Indicators
        df1 = calculate_indicators(df1)
        
        # 3. Ratio
        ratio_stats = {}
        if ticker2:
            df2 = fetch_data_safe(ticker2, per, tf) # 2s delay inside
            if df2 is not None:
                # Align
                c = df1['Close'].to_frame('T1').join(df2['Close'].to_frame('T2'), how='outer').ffill().dropna()
                c['Ratio'] = c['T1']/c['T2']
                c['Ratio_EMA'] = c['Ratio'].ewm(span=20).mean()
                curr_r = c['Ratio'].iloc[-1]
                avg_r = c['Ratio'].mean()
                ratio_stats = {
                    "current": curr_r,
                    "mean": avg_r,
                    "status": "Overvalued" if curr_r > avg_r * 1.05 else "Undervalued" if curr_r < avg_r * 0.95 else "Fair"
                }

        # 4. Structures
        sr_levels = get_strong_sr_levels(df1)
        fibs = get_fibonacci(df1)
        wave = detect_elliott_wave(df1)
        
        # 5. Backtest
        best_ma, roi = optimize_strategy(df1)
        
        # 6. Current Signal (Technical)
        curr = df1.iloc[-1]
        signal = "NEUTRAL"
        score = 0
        
        # Trend
        if curr['Close'] > curr['EMA_20']: score += 1
        if curr['EMA_20'] > curr['EMA_50']: score += 1
        if curr['Close'] > curr['EMA_200']: score += 1 # Long trend
        
        # Momentum
        if 40 < curr['RSI'] < 70: score += 1
        elif curr['RSI'] < 30: score += 2 #(Oversold Bounce)
        
        # Structure
        closest_sr = min(sr_levels, key=lambda x: abs(x - curr['Close'])) if sr_levels else 0
        dist_sr = abs(curr['Close'] - closest_sr) / curr['Close'] if closest_sr else 1
        if dist_sr < 0.01: score += 2 # At Support/Resistance
        
        if score >= 4: signal = "BUY"
        elif score <= 1: signal = "SELL"
        
        # Store Result
        results[tf] = {
            "data": df1,
            "signal": signal,
            "score": score,
            "price": curr['Close'],
            "rsi": curr['RSI'],
            "zscore": curr['Z_Score'],
            "volatility": curr['Volatility'],
            "fibs": fibs,
            "sr": sr_levels,
            "wave": wave,
            "opt_params": best_ma,
            "opt_roi": roi,
            "ratio": ratio_stats
        }
        
    bar.empty()
    return results

# ==========================================
# 6. NARRATIVE GENERATOR
# ==========================================

def generate_final_verdict(results):
    # Weighted Voting
    weights = {"15m": 0.1, "1h": 0.3, "1d": 0.4, "1wk": 0.2}
    total_score = 0
    
    sentiment_counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    
    for tf, res in results.items():
        if tf not in weights: continue
        
        s = res['signal']
        sentiment_counts[s] += 1
        
        val = 1 if s == "BUY" else -1 if s == "SELL" else 0
        total_score += val * weights[tf]
        
    final_signal = "HOLD"
    if total_score > 0.3: final_signal = "BUY"
    if total_score < -0.3: final_signal = "SELL"
    
    confidence = abs(total_score) * 100
    confidence = min(confidence + 30, 98) # Normalize to sensible %
    
    return final_signal, int(confidence), sentiment_counts

# ==========================================
# 7. UI COMPONENTS
# ==========================================

def draw_trading_ui(current_price, ticker):
    st.markdown("### 🛠️ Live Simulation")
    
    # State handling for buttons
    c1, c2, c3 = st.columns(3)
    c1.metric("Balance", f"₹{st.session_state.balance:,.2f}")
    
    qty = c3.number_input("Qty", value=10)
    
    col_b, col_s, col_c = st.columns(3)
    
    if col_b.button("BUY MARKET", key="buy_btn", type="primary"):
        cost = current_price * qty
        if st.session_state.balance >= cost:
            st.session_state.balance -= cost
            st.session_state.positions.append({"type": "BUY", "price": current_price, "qty": qty, "ticker": ticker, "ts": datetime.now()})
            st.success("Order Executed")
            st.rerun()
            
    if col_s.button("SELL MARKET", key="sell_btn", type="primary"):
        st.session_state.positions.append({"type": "SELL", "price": current_price, "qty": qty, "ticker": ticker, "ts": datetime.now()})
        st.success("Short Order Executed")
        st.rerun()

    # Positions Table
    if st.session_state.positions:
        p_df = pd.DataFrame(st.session_state.positions)
        p_df['Current'] = current_price
        p_df['PnL'] = np.where(p_df['type']=='BUY', (p_df['Current'] - p_df['price'])*p_df['qty'], (p_df['price'] - p_df['Current'])*p_df['qty'])
        st.dataframe(p_df)
        
        if col_c.button("CLOSE ALL"):
            total_pnl = p_df['PnL'].sum()
            # Refund principal for buys
            principal = p_df[p_df['type']=='BUY']['price'] * p_df[p_df['type']=='BUY']['qty']
            st.session_state.balance += principal.sum() + total_pnl
            st.session_state.positions = []
            st.rerun()

# ==========================================
# 8. MAIN APP
# ==========================================

def main():
    st.sidebar.header("Algo Config")
    t1 = st.sidebar.selectbox("Asset", list(TICKER_MAP.keys()))
    
    # Auto-select Ratio Ticker
    def_t2 = "USD/INR" if t1 in ["NIFTY 50", "BANK NIFTY", "SENSEX"] else "GOLD" if "BTC" in t1 else "NIFTY 50"
    t2 = st.sidebar.selectbox("Ratio Compare", list(TICKER_MAP.keys()), index=list(TICKER_MAP.keys()).index(def_t2))
    
    if st.sidebar.button("🚀 START DEEP SCAN"):
        st.session_state.analysis_cache = run_multi_timeframe_analysis(t1, t2)
    
    # Check if analysis exists
    if st.session_state.analysis_cache:
        res = st.session_state.analysis_cache
        
        # 1. VERDICT CARD
        final_sig, conf, counts = generate_final_verdict(res)
        
        st.title(f"Analysis Report: {t1}")
        
        vc1, vc2 = st.columns([1,2])
        with vc1:
            color = "#00ff00" if final_sig == "BUY" else "#ff0000" if final_sig == "SELL" else "#888"
            st.markdown(f"""
            <div style="background: {color}; padding: 20px; border-radius: 10px; text-align: center; color: black;">
                <h1 style="margin:0">{final_sig}</h1>
                <h3 style="margin:0">Confidence: {conf}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with vc2:
            st.info(f"""
            **Consensus Logic:**
            * **15m (Short Term):** {res['15m']['signal']} (RSI: {res['15m']['rsi']:.1f})
            * **1h (Medium Term):** {res['1h']['signal']} (Wave: {res['1h']['wave']})
            * **1d (Long Term):** {res['1d']['signal']} (Best Strat ROI: {res['1d']['opt_roi']:.1f}%)
            """)

        # 2. DETAILED EXPLANATION (Natural Language)
        latest_tf = res['1h'] # Use 1h for main commentary
        curr_price = latest_tf['price']
        
        txt_col, chart_col = st.columns([1, 2])
        
        with txt_col:
            st.subheader("📝 Technical Executive Summary")
            
            # Construct Narrative
            narrative = f"""
            **1. Structural Analysis (Elliott & Fibs):**
            The market is currently in a **{latest_tf['wave']}** phase on the hourly chart. 
            Price ({curr_price:.2f}) is reacting to the Fibonacci level of **{min(latest_tf['fibs'].values(), key=lambda x:abs(x-curr_price)):.2f}**.
            Strong Support clusters detected at: {', '.join([f'{x:.0f}' for x in latest_tf['sr'][:2]])}.
            
            **2. Statistical Probability (Z-Score):**
            Current Z-Score is **{latest_tf['zscore']:.2f}**. 
            {'This suggests extreme overvaluation; mean reversion (pullback) is statistically likely.' if latest_tf['zscore'] > 2 else 
             'This suggests undervaluation; a bounce is statistically likely.' if latest_tf['zscore'] < -2 else 
             'Price is within normal statistical deviation.'}
            
            **3. Ratio & Inter-market Analysis:**
            {t1} vs {t2} Ratio is **{latest_tf['ratio'].get('status', 'N/A')}**. 
            When this ratio deviates significantly, {t1} often corrects to align with macro trends.
            
            **4. Backtested Strategy Optimization:**
            We tested Moving Average strategies on this data. The optimal setup found was **EMA {latest_tf['opt_params'][0]} / {latest_tf['opt_params'][1]}**, generating an annualized return of **{latest_tf['opt_roi']:.1f}%**.
            """
            st.markdown(narrative)
            
            draw_trading_ui(curr_price, t1)

        with chart_col:
            # Multi-Tab Charts
            tabs = st.tabs(res.keys())
            for tf, data in res.items():
                with tabs[list(res.keys()).index(tf)]:
                    df = data['data']
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    # Candles
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    
                    # Optimized EMAs
                    f_ma, s_ma = data['opt_params']
                    if f_ma > 0:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=f_ma).mean(), line=dict(color='orange'), name=f"Opt EMA {f_ma}"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=s_ma).mean(), line=dict(color='blue'), name=f"Opt EMA {s_ma}"), row=1, col=1)
                        
                    # S/R Lines
                    for sr in data['sr']:
                        fig.add_hline(y=sr, line_dash="dash", line_color="green", row=1, col=1)
                        
                    # Fibs
                    for k, v in data['fibs'].items():
                        if abs(v - curr_price) / curr_price < 0.05: # Only show nearby fibs
                            fig.add_hline(y=v, line_dash="dot", line_color="gray", annotation_text=f"Fib {k}", row=1, col=1)

                    # Z-Score
                    fig.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], line=dict(color='purple'), name="Z-Score"), row=2, col=1)
                    fig.add_hline(y=2, line_color="red", row=2, col=1)
                    fig.add_hline(y=-2, line_color="green", row=2, col=1)
                    
                    fig.update_layout(height=600, template="plotly_dark", title=f"{t1} - {tf} Timeframe Analysis")
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
