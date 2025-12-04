import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import time
import pytz

# ==========================================
# 1. PAGE CONFIG & STATE MANAGEMENT
# ==========================================
st.set_page_config(layout="wide", page_title="QuanT Pro: AI Trade Logic", page_icon="⚡")

# Initialize Session State for Paper Trading
if 'balance' not in st.session_state:
    st.session_state.balance = 100000.0  # Starting Capital
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# Custom CSS for the "Verdict Card"
st.markdown("""
<style>
    .verdict-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .verdict-buy { background: linear-gradient(135deg, #0f3d0f 0%, #00ff00 100%); border: 2px solid #00ff00; }
    .verdict-sell { background: linear-gradient(135deg, #4a0f0f 0%, #ff0000 100%); border: 2px solid #ff0000; }
    .verdict-hold { background: linear-gradient(135deg, #333333 0%, #888888 100%); border: 2px solid #888888; }
    .confidence-score { font-size: 3em; font-weight: 800; }
    .big-signal { font-size: 1.5em; letter-spacing: 2px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE & HELPERS
# ==========================================

TICKER_PRESETS = [
    "NIFTY 50", "BANK NIFTY", "SENSEX", "BTC-USD", "ETH-USD", 
    "GC=F", "SI=F", "INR=X", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"
]

def get_ticker_symbol(friendly_name):
    mapping = {
        "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
        "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GC=F": "GC=F", 
        "SI=F": "SI=F", "INR=X": "INR=X"
    }
    return mapping.get(friendly_name, friendly_name)

def get_default_ratio_ticker(ticker1):
    if "NIFTY" in ticker1 or "SENSEX" in ticker1 or ".NS" in ticker1:
        return "INR=X" # USD/INR
    if "BTC" in ticker1 or "ETH" in ticker1:
        return "GC=F" # Gold
    return "NIFTY 50"

@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    """Fetches data with strict rate limiting."""
    time.sleep(1.5)  # 1.5s delay to prevent API crash
    try:
        symbol = get_ticker_symbol(ticker)
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df.empty: return None
        
        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Timezone Handling (Convert to IST)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        return df
    except Exception as e:
        return None

def calculate_technical_metrics(df):
    if df is None or len(df) < 50: return df
    
    # 1. Standard Indicators
    df['Returns'] = df['Close'].pct_change()
    df['Range'] = df['High'] - df['Low']
    
    # EMAs
    for span in [9, 20, 50, 200]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility (ATR-like)
    df['Volatility'] = df['Range'].rolling(14).mean()
    
    # Z-Score of Price relative to 20 SMA
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['StdDev_20'] = df['Close'].rolling(20).std()
    df['Z_Score'] = (df['Close'] - df['SMA_20']) / df['StdDev_20']
    
    return df

# ==========================================
# 3. ADVANCED ANALYTICAL ENGINES
# ==========================================

def analyze_historical_similarity(df, current_idx, lookback=60):
    """
    Finds past instances where Price Change %, Volatility, and RSI were 
    similar to the current candle.
    """
    curr = df.iloc[current_idx]
    history = df.iloc[:-1].copy() # Exclude current candle
    
    # Criteria: RSI within +/- 5, Volatility within +/- 10%, Z-Score within +/- 0.5
    matches = history[
        (history['RSI'].between(curr['RSI']-5, curr['RSI']+5)) & 
        (history['Z_Score'].between(curr['Z_Score']-0.5, curr['Z_Score']+0.5))
    ].copy()
    
    if matches.empty:
        return None
    
    # Check what happened in the NEXT 5 candles after the match
    outcomes = []
    for idx in matches.index:
        loc = df.index.get_loc(idx)
        if loc + 5 < len(df):
            future_price = df['Close'].iloc[loc+5]
            past_price = df['Close'].iloc[loc]
            pct_move = ((future_price - past_price) / past_price) * 100
            outcomes.append({
                'Date': idx,
                'Ref_Price': past_price,
                'RSI_Then': df['RSI'].iloc[loc],
                'Next_5_Candles_Return': pct_move
            })
            
    return pd.DataFrame(outcomes)

def check_what_is_working(df):
    """
    Checks which technical levels are currently being respected.
    """
    price = df['Close'].iloc[-1]
    report = []
    
    # Check EMAs
    for ema in [9, 20, 50, 200]:
        val = df[f'EMA_{ema}'].iloc[-1]
        dist = abs(price - val) / price
        if dist < 0.005: # Within 0.5%
            status = "Holding Support" if price > val else "Facing Resistance"
            report.append(f"Price is strictly obeying **{ema} EMA** ({status}).")
            
    # Check RSI
    rsi = df['RSI'].iloc[-1]
    if rsi > 60 and price > df['EMA_20'].iloc[-1]:
        report.append("RSI Momentum is driving the trend (Bullish Zone).")
    elif 40 < rsi < 60:
        report.append("RSI is Neutral/Sideways (Indecision).")
        
    return report

def detect_rsi_divergence(df, window=14):
    """
    Detects simple divergences over the last 'window' periods.
    """
    recent = df.iloc[-window:]
    
    # Price Higher High but RSI Lower High (Bearish)
    price_highs = argrelextrema(recent['High'].values, np.greater)[0]
    rsi_highs = argrelextrema(recent['RSI'].values, np.greater)[0]
    
    divergence = "None"
    
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p_h1, p_h2 = recent['High'].iloc[price_highs[-2]], recent['High'].iloc[price_highs[-1]]
        r_h1, r_h2 = recent['RSI'].iloc[rsi_highs[-2]], recent['RSI'].iloc[rsi_highs[-1]]
        
        if p_h2 > p_h1 and r_h2 < r_h1:
            divergence = "Bearish (Regular)"

    # Price Lower Low but RSI Higher Low (Bullish)
    price_lows = argrelextrema(recent['Low'].values, np.less)[0]
    rsi_lows = argrelextrema(recent['RSI'].values, np.less)[0]
    
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p_l1, p_l2 = recent['Low'].iloc[price_lows[-2]], recent['Low'].iloc[price_lows[-1]]
        r_l1, r_l2 = recent['RSI'].iloc[rsi_lows[-2]], recent['RSI'].iloc[rsi_lows[-1]]
        
        if p_l2 < p_l1 and r_l2 > r_l1:
            divergence = "Bullish (Regular)"
            
    return divergence

def generate_recommendation(df):
    """
    Calculates a weighted Confidence Score (0-100%).
    """
    score = 50 # Start Neutral
    reasons = []
    
    curr = df.iloc[-1]
    
    # 1. Trend (30%)
    if curr['Close'] > curr['EMA_20'] > curr['EMA_50']:
        score += 15
        reasons.append("Strong Uptrend (Price > EMA20 > EMA50).")
    elif curr['Close'] < curr['EMA_20'] < curr['EMA_50']:
        score -= 15
        reasons.append("Strong Downtrend (Price < EMA20 < EMA50).")
        
    # 2. Momentum (20%)
    if 50 < curr['RSI'] < 70:
        score += 10
        reasons.append("RSI shows healthy bullish momentum.")
    elif curr['RSI'] > 75:
        score -= 10
        reasons.append("RSI Overbought (Risk of pullback).")
    elif curr['RSI'] < 25:
        score += 10
        reasons.append("RSI Oversold (Potential bounce).")
        
    # 3. Mean Reversion (Z-Score) (20%)
    if curr['Z_Score'] < -2:
        score += 15
        reasons.append("Z-Score < -2 indicates statistical undervaluation.")
    elif curr['Z_Score'] > 2:
        score -= 15
        reasons.append("Z-Score > 2 indicates statistical overextension.")
        
    # 4. Divergence (30%)
    div = detect_rsi_divergence(df)
    if "Bullish" in div:
        score += 20
        reasons.append(f"Critical {div} Divergence detected.")
    elif "Bearish" in div:
        score -= 20
        reasons.append(f"Critical {div} Divergence detected.")
        
    # Cap Score
    score = max(0, min(100, score))
    
    signal = "HOLD"
    if score >= 65: signal = "BUY"
    elif score <= 35: signal = "SELL"
    
    return signal, score, reasons

# ==========================================
# 4. UI COMPONENTS
# ==========================================

def render_verdict_card(signal, score, reasons):
    color_class = f"verdict-{signal.lower()}"
    
    html = f"""
    <div class="verdict-card {color_class}">
        <div class="big-signal">{signal} RECOMMENDATION</div>
        <div class="confidence-score">{score}%</div>
        <div>CONFIDENCE</div>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <div style="text-align: left; padding: 0 10px;">
            <strong>Analysis Logic:</strong><br>
            {'<br>'.join([f"• {r}" for r in reasons])}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def paper_trade_ui(ticker, current_price, signal):
    st.subheader("🛠️ Live Paper Trading Simulator")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Available Balance", f"₹{st.session_state.balance:,.2f}")
    
    # Calculate P&L of open positions
    open_pnl = 0
    for pos in st.session_state.positions:
        if pos['status'] == 'OPEN':
            curr_val = (current_price - pos['entry_price']) * pos['qty']
            if pos['type'] == 'SELL': curr_val = -curr_val
            open_pnl += curr_val
            
    c2.metric("Unrealized P&L", f"₹{open_pnl:,.2f}", delta_color="normal")
    
    with c3:
        qty = st.number_input("Quantity", min_value=1, value=10)
        
    b1, b2, b3 = st.columns(3)
    
    if b1.button("🟢 MARKET BUY"):
        cost = current_price * qty
        if st.session_state.balance >= cost:
            st.session_state.balance -= cost
            st.session_state.positions.append({
                'ticker': ticker, 'type': 'BUY', 'entry_price': current_price, 
                'qty': qty, 'status': 'OPEN', 'time': datetime.now()
            })
            st.success(f"Bought {qty} {ticker} @ {current_price}")
            st.rerun()
        else:
            st.error("Insufficient Funds")

    if b2.button("🔴 MARKET SELL"):
        # Allow short selling (margin logic simplified)
        st.session_state.positions.append({
            'ticker': ticker, 'type': 'SELL', 'entry_price': current_price, 
            'qty': qty, 'status': 'OPEN', 'time': datetime.now()
        })
        st.success(f"Short Sold {qty} {ticker} @ {current_price}")
        st.rerun()
        
    if b3.button("⚠️ CLOSE ALL POSITIONS"):
        for pos in st.session_state.positions:
            if pos['status'] == 'OPEN':
                pos['status'] = 'CLOSED'
                pos['exit_price'] = current_price
                pnl = (current_price - pos['entry_price']) * pos['qty']
                if pos['type'] == 'SELL': pnl = -pnl
                st.session_state.balance += (pos['entry_price'] * pos['qty']) if pos['type'] == 'BUY' else 0 # Return principal
                st.session_state.balance += pnl
        st.success("All positions closed.")
        st.rerun()

    if st.session_state.positions:
        st.write("### Active Positions")
        active_df = pd.DataFrame([p for p in st.session_state.positions if p['status']=='OPEN'])
        if not active_df.empty:
            active_df['Current PnL'] = active_df.apply(lambda x: (current_price - x['entry_price']) * x['qty'] if x['type'] == 'BUY' else (x['entry_price'] - current_price) * x['qty'], axis=1)
            st.dataframe(active_df)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    st.sidebar.title("🎛️ Market Scanner")
    
    # 1. Inputs
    t1_sel = st.sidebar.selectbox("Ticker 1 (Primary)", TICKER_PRESETS, index=0)
    ticker1 = get_ticker_symbol(t1_sel)
    
    default_t2 = get_default_ratio_ticker(t1_sel)
    t2_sel = st.sidebar.selectbox("Ticker 2 (Ratio/Comparison)", TICKER_PRESETS, index=TICKER_PRESETS.index(default_t2) if default_t2 in TICKER_PRESETS else 0)
    ticker2 = get_ticker_symbol(t2_sel)
    
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=4)
    
    # Smart Period Selection
    period_map = {"1m": "5d", "5m": "1mo", "15m": "1mo", "30m": "3mo", "1h": "1y", "1d": "5y", "1wk": "10y"}
    period = period_map.get(timeframe, "1y")

    if st.sidebar.button("🔍 ANALYZE MARKET", type="primary"):
        with st.spinner("Crunching Numbers & Finding Historical Matches..."):
            
            # Fetch Data
            df1 = fetch_data(ticker1, period, timeframe)
            df2 = fetch_data(ticker2, period, timeframe)
            
            if df1 is None:
                st.error("Data Fetch Failed. Try again in 5 seconds (API Rate Limit).")
                return

            # Process Data
            df1 = calculate_technical_metrics(df1)
            if df2 is not None:
                df2 = calculate_technical_metrics(df2)
                # Align for ratio
                combined = df1['Close'].to_frame(name='T1').join(df2['Close'].to_frame(name='T2'), how='outer').ffill().dropna()
                combined['Ratio'] = combined['T1'] / combined['T2']
                # Calculate Changes
                combined['T1_Chg'] = combined['T1'].pct_change() * 100
                combined['Ratio_Chg'] = combined['Ratio'].pct_change() * 100
            
            # Generate Signals
            signal, conf, reasons = generate_recommendation(df1)
            current_price = df1['Close'].iloc[-1]
            
            # --- LAYOUT ---
            
            # 1. Verdict Section
            col_l, col_r = st.columns([1, 3])
            
            with col_l:
                render_verdict_card(signal, conf, reasons)
                
            with col_r:
                # Top Stats Grid
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{t1_sel} Price", f"{current_price:,.2f}", f"{df1['Returns'].iloc[-1]*100:.2f}%")
                m2.metric("RSI (14)", f"{df1['RSI'].iloc[-1]:.1f}", "Bullish" if df1['RSI'].iloc[-1]>50 else "Bearish")
                m3.metric("Volatility", f"{df1['Volatility'].iloc[-1]:.2f}", "High" if df1['Volatility'].iloc[-1] > df1['Volatility'].mean() else "Low")
                m4.metric("Divergence", detect_rsi_divergence(df1), delta_color="off")
                
                # What is working text
                st.info(f"**Market Reality Check:** {' '.join(check_what_is_working(df1))}")

            # 2. Main Tabs
            tab_trade, tab_deep, tab_ratio, tab_data = st.tabs(["⚡ Paper Trading", "🧠 Deep Analysis", "⚖️ Ratio & Stats", "📅 Historical Data"])
            
            with tab_trade:
                paper_trade_ui(t1_sel, current_price, signal)
            
            with tab_deep:
                st.subheader("🔍 Historical Pattern Matching")
                st.write(f"Finding past instances where **RSI ~{df1['RSI'].iloc[-1]:.0f}** and **Z-Score ~{df1['Z_Score'].iloc[-1]:.1f}**...")
                
                history_matches = analyze_historical_similarity(df1, -1)
                if history_matches is not None and not history_matches.empty:
                    avg_return = history_matches['Next_5_Candles_Return'].mean()
                    win_rate = len(history_matches[history_matches['Next_5_Candles_Return'] > 0]) / len(history_matches) * 100
                    
                    st.write(f"Found **{len(history_matches)}** similar instances in history.")
                    k1, k2 = st.columns(2)
                    k1.metric("Avg Return (Next 5 Candles)", f"{avg_return:.2f}%", delta_color="normal")
                    k2.metric("Historical Win Rate", f"{win_rate:.1f}%")
                    
                    st.dataframe(history_matches.style.format({'Ref_Price': '{:.2f}', 'Next_5_Candles_Return': '{:.2f}%'}))
                else:
                    st.warning("Market conditions are unique. No sufficient historical similarity found.")

                st.subheader("📉 Technical Charts")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df1.index, open=df1['Open'], high=df1['High'], low=df1['Low'], close=df1['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df1.index, y=df1['EMA_20'], line=dict(color='orange'), name="EMA 20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df1.index, y=df1['EMA_50'], line=dict(color='blue'), name="EMA 50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df1.index, y=df1['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)

            with tab_ratio:
                if df2 is not None:
                    st.subheader(f"Ratio Dynamics: {t1_sel} vs {t2_sel}")
                    st.write("This table shows exactly when Ratio/Volatility spikes occurred and their values.")
                    
                    # Custom Data Table for Ratio/Vol Analysis
                    analysis_table = combined.tail(50).copy()
                    analysis_table = analysis_table.sort_index(ascending=False)
                    
                    # Highlight significant moves
                    def highlight_significant(val):
                        if abs(val) > 1.0: return 'background-color: rgba(255, 0, 0, 0.2)'
                        return ''
                    
                    st.dataframe(analysis_table[['T1', 'T2', 'Ratio', 'T1_Chg', 'Ratio_Chg']].style.applymap(highlight_significant, subset=['T1_Chg', 'Ratio_Chg']))
                else:
                    st.warning("Ticker 2 Data not available.")

            with tab_data:
                st.dataframe(df1.sort_index(ascending=False))

if __name__ == "__main__":
    main()
