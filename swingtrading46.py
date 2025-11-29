import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pytz
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPro: Multi-Timeframe Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark UI
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #2962ff; color: white; font-weight: bold; border: none; padding: 0.5rem; }
    .stButton>button:hover { background-color: #0039cb; }
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 8px; border-left: 5px solid #2962ff; margin-bottom: 10px; }
    .signal-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .buy-signal { background-color: #004d40; color: #00e676; border: 2px solid #00e676; }
    .sell-signal { background-color: #4a0000; color: #ff5252; border: 2px solid #ff5252; }
    .neutral-signal { background-color: #3e2723; color: #ffab40; border: 2px solid #ffab40; }
    .psychology-warning { font-size: 12px; color: #ffcc80; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & MAPPINGS
# -----------------------------------------------------------------------------
ASSETS = {
    "Indices": {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX", "^GSPC": "S&P 500"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil"},
    "Forex": {"INR=X": "USD/INR", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD"},
    "Stocks (IN)": {"RELIANCE.NS": "Reliance", "HDFCBANK.NS": "HDFC Bank", "TCS.NS": "TCS"},
    "Stocks (US)": {"AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia"}
}

TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '1wk']
HTF_MAPPING = {
    '1m': '15m', '2m': '15m', '5m': '1h', '15m': '4h', 
    '30m': '1d', '1h': '1d', '1d': '1wk', '1wk': '1mo'
}

PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def to_ist(df):
    """Converts DataFrame index to IST."""
    if df.empty: return df
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df.index.tz_convert('Asia/Kolkata')

@st.cache_data(ttl=300) # Cache for 5 mins to respect API limits logic slightly
def fetch_data_robust(ticker, period, interval, delay=1.5):
    """Fetches data with rate limiting and cleaning."""
    time.sleep(delay) # Mandatory Rate Limit
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        
        # Handle yfinance multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.index = to_ist(df)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def fetch_news(ticker):
    """Fetches news for context."""
    try:
        t = yf.Ticker(ticker)
        return t.news[:3] # Return top 3
    except:
        return []

# -----------------------------------------------------------------------------
# 4. TECHNICAL ANALYSIS ENGINE
# -----------------------------------------------------------------------------
class MarketBrain:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_indicators(self):
        df = self.df
        # 1. Moving Averages (Trend)
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # 2. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR (Volatility for SL)
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # 4. MACD (Momentum)
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # 5. Pivot Points (Fibonacci style for simplicity on last candles)
        high, low = df['High'].rolling(20).max(), df['Low'].rolling(20).min()
        df['Fib_0'] = low
        df['Fib_1'] = high
        df['Fib_0.618'] = high - (high - low) * 0.618
        df['Fib_0.382'] = high - (high - low) * 0.382
        
        self.df = df
        return df

    def check_divergence(self, window=15):
        """Detects Regular Bullish/Bearish Divergence."""
        df = self.df.iloc[-window:]
        price_low = df['Low'].min()
        price_high = df['High'].max()
        rsi_low = df['RSI'].min()
        rsi_high = df['RSI'].max()
        
        # Simple Logic: Price made lower low, RSI made higher low (Bullish)
        # This is a simplified detection for the last window
        first_half = df.iloc[:int(window/2)]
        last_half = df.iloc[int(window/2):]
        
        div_signal = "None"
        
        if last_half['Close'].mean() < first_half['Close'].mean() and last_half['RSI'].mean() > first_half['RSI'].mean():
            div_signal = "Bullish Divergence (Possible)"
            
        if last_half['Close'].mean() > first_half['Close'].mean() and last_half['RSI'].mean() < first_half['RSI'].mean():
            div_signal = "Bearish Divergence (Possible)"
            
        return div_signal

    def identify_elliott_context(self):
        """Heuristic Elliott Wave Context Identifier."""
        last = self.df.iloc[-1]
        trend = "Neutral"
        
        # Impulse vs Correction Logic
        if last['Close'] > last['EMA_20'] > last['EMA_50']:
            if last['RSI'] > 50 and last['RSI'] < 75:
                trend = "Wave 3 (Strong Impulse)"
            elif last['RSI'] >= 75:
                trend = "Wave 5 (Exhaustion Risk)"
        elif last['Close'] < last['EMA_20'] < last['EMA_50']:
            if last['RSI'] < 50 and last['RSI'] > 25:
                trend = "Wave C (Down Impulse)"
            elif last['RSI'] <= 25:
                trend = "Oversold (Bounce Likely)"
        else:
            trend = "Corrective / Choppy (Wave 2/4/B)"
            
        return trend

    def generate_signal(self):
        """Generates logic based on confluence."""
        row = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        score = 0
        reasons = []
        
        # Trend Score
        if row['Close'] > row['EMA_50']: score += 1
        if row['EMA_20'] > row['EMA_50']: score += 1
        
        # Momentum Score
        if row['RSI'] > 50: score += 1
        if row['MACD'] > row['Signal_Line']: score += 1
        
        # Crossovers
        if row['EMA_20'] > row['EMA_50'] and prev['EMA_20'] <= prev['EMA_50']:
            reasons.append("Golden Cross (EMA)")
        
        action = "HOLD"
        if score >= 3: action = "BUY"
        elif score <= 1: action = "SELL"
        
        return action, score, reasons

# -----------------------------------------------------------------------------
# 5. BACKTESTING ENGINE
# -----------------------------------------------------------------------------
def run_backtest(df, signal_logic_fn):
    """Simulates the current strategy on the last N bars."""
    balance = 10000
    position = 0 # 0, 1 (Long), -1 (Short)
    trades = []
    
    # Simple vector loop
    for i in range(50, len(df)):
        slice_df = df.iloc[:i]
        # Recalculate indicators for this slice (simplified for speed, usually pre-calc)
        row = df.iloc[i]
        
        # Simplified Strategy Access for Speed
        buy_cond = row['Close'] > row['EMA_20'] and row['RSI'] > 50 and row['MACD'] > row['Signal_Line']
        sell_cond = row['Close'] < row['EMA_20'] and row['RSI'] < 50 and row['MACD'] < row['Signal_Line']
        
        if position == 0:
            if buy_cond:
                position = 1
                entry_price = row['Close']
                trades.append({'type': 'Buy', 'price': entry_price, 'idx': df.index[i]})
            elif sell_cond:
                position = -1
                entry_price = row['Close']
                trades.append({'type': 'Sell', 'price': entry_price, 'idx': df.index[i]})
        
        elif position == 1: # In Long
            # Exit condition (Cross under EMA 20)
            if row['Close'] < row['EMA_20']:
                position = 0
                pnl = row['Close'] - entry_price
                trades.append({'type': 'Exit Long', 'price': row['Close'], 'pnl': pnl})

        elif position == -1: # In Short
            # Exit condition (Cross over EMA 20)
            if row['Close'] > row['EMA_20']:
                position = 0
                pnl = entry_price - row['Close']
                trades.append({'type': 'Exit Short', 'price': row['Close'], 'pnl': pnl})
                
    wins = len([t for t in trades if t.get('pnl', 0) > 0])
    total = len([t for t in trades if 'pnl' in t])
    win_rate = (wins/total * 100) if total > 0 else 0
    return win_rate, total

# -----------------------------------------------------------------------------
# 6. MAIN UI LOGIC
# -----------------------------------------------------------------------------
def main():
    # Sidebar Setup
    st.sidebar.header("⚙️ Configuration")
    
    # Asset Selection
    cat = st.sidebar.selectbox("Market Type", list(ASSETS.keys()))
    ticker_name = st.sidebar.selectbox("Instrument", list(ASSETS[cat].values()))
    ticker = [k for k, v in ASSETS[cat].items() if v == ticker_name][0]
    
    custom_tick = st.sidebar.text_input("Or Custom Ticker (Yahoo format)", "")
    if custom_tick: ticker = custom_tick
    
    # Timeframe Selection
    colT1, colT2 = st.sidebar.columns(2)
    with colT1:
        tf = st.selectbox("Primary Timeframe", TIMEFRAMES, index=4) # 30m default
    with colT2:
        period = st.selectbox("Data Lookback", PERIODS, index=3) # 3mo default

    # Advanced Toggles
    st.sidebar.markdown("---")
    use_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
    ratio_ticker = ""
    if use_ratio:
        ratio_ticker = st.sidebar.text_input("Compare Against (e.g., ^NSEI)", value="^NSEI")
    
    run_btn = st.sidebar.button("🧠 Analyze Market Structure", type="primary")
    
    # Session State Init
    if 'analyzed' not in st.session_state: st.session_state.analyzed = False
    
    st.title(f"🛡️ Professional Algo-Analyst: {ticker_name}")
    
    if run_btn:
        st.session_state.analyzed = True
        with st.spinner("Initializing Quantitative Matrix..."):
            # 1. Fetch Main Data
            df = fetch_data_robust(ticker, period, tf)
            
            # 2. Fetch Higher Timeframe Data (Context)
            htf_val = HTF_MAPPING.get(tf, '1d')
            df_htf = fetch_data_robust(ticker, period, htf_val)
            
            if df is None or df_htf is None:
                st.error("Data Fetch Failed. Check Ticker or Try Again (Rate Limit).")
                return

            # 3. Analyze Both Timeframes
            brain_ltf = MarketBrain(df)
            brain_htf = MarketBrain(df_htf)
            
            df = brain_ltf.add_indicators()
            df_htf = brain_htf.add_indicators()
            
            # 4. Generate Core Signals
            action_ltf, score_ltf, reasons_ltf = brain_ltf.generate_signal()
            action_htf, score_htf, reasons_htf = brain_htf.generate_signal()
            ew_context = brain_ltf.identify_elliott_context()
            divergence = brain_ltf.check_divergence()
            
            # 5. Conflict Resolution (The "One Signal" Logic)
            final_signal = "HOLD"
            confidence = "Low"
            
            if action_htf == "BUY":
                if action_ltf == "BUY":
                    final_signal = "STRONG BUY"
                    confidence = "High (Trend Aligned)"
                elif action_ltf == "SELL":
                    final_signal = "WAIT (Dip Buying Opp)"
                    confidence = "Medium"
            elif action_htf == "SELL":
                if action_ltf == "SELL":
                    final_signal = "STRONG SELL"
                    confidence = "High (Trend Aligned)"
                elif action_ltf == "BUY":
                    final_signal = "SCALP BUY (Counter-Trend)"
                    confidence = "Low/Risky"
            
            # 6. Calculate Targets (ATR Based)
            last_close = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            stop_loss = last_close - (2 * atr) if "BUY" in final_signal else last_close + (2 * atr)
            target_1 = last_close + (3 * atr) if "BUY" in final_signal else last_close - (3 * atr)
            
            # Store in session
            st.session_state.res = {
                'df': df, 'df_htf': df_htf,
                'signal': final_signal, 'conf': confidence,
                'sl': stop_loss, 'tgt': target_1,
                'ew': ew_context, 'div': divergence,
                'reasons': reasons_ltf
            }
            
            # Optional Ratio
            if use_ratio and ratio_ticker:
                df_r = fetch_data_robust(ratio_ticker, period, tf)
                if df_r is not None:
                    # Align dates
                    common_idx = df.index.intersection(df_r.index)
                    st.session_state.ratio_series = df.loc[common_idx]['Close'] / df_r.loc[common_idx]['Close']

    # -------------------------------------------------------------------------
    # DISPLAY SECTION
    # -------------------------------------------------------------------------
    if st.session_state.analyzed and 'res' in st.session_state:
        res = st.session_state.res
        df = res['df']
        
        # --- TOP DASHBOARD ---
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            css_class = "buy-signal" if "BUY" in res['signal'] else "sell-signal" if "SELL" in res['signal'] else "neutral-signal"
            st.markdown(f"<div class='signal-box {css_class}'>{res['signal']}</div>", unsafe_allow_html=True)
            st.caption(f"Confidence: {res['conf']}")
            
            # Psychology Check
            if "Risky" in res['conf'] or "Counter" in res['signal']:
                st.markdown("<p class='psychology-warning'>⚠️ PSYCHOLOGY ALERT: You are trading against the higher timeframe trend. Reduce position size by 50%.</p>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
            <b>🎯 Execution Plan</b><br>
            Price: {df['Close'].iloc[-1]:.2f}<br>
            Stop Loss: <span style='color:#ff5252'>{res['sl']:.2f}</span><br>
            Target: <span style='color:#00e676'>{res['tgt']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
            <b>🌊 Market Structure</b><br>
            Elliott Context: {res['ew']}<br>
            Divergence: {res['div']}<br>
            RSI (14): {df['RSI'].iloc[-1]:.2f}
            </div>
            """, unsafe_allow_html=True)

        # --- TABS FOR DEPTH ---
        tab1, tab2, tab3 = st.tabs(["📉 Technical Charts", "🧪 Backtest Validator", "📰 News & Sentiment"])
        
        with tab1:
            # PLOTLY CHART
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=("Price Action & EMAs", "Momentum (RSI)", "Trend Strength (MACD)"))

            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='yellow', width=1), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='cyan', width=1), name='EMA 50'), row=1, col=1)
            
            # Fibonacci / Pivot approximations
            fig.add_hline(y=df['Fib_0.618'].iloc[-1], line_dash="dot", line_color="gold", row=1, col=1, annotation_text="Fib 0.618")

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            fig.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal_Line'], name='Hist'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='orange'), name='Signal'), row=3, col=1)

            # Fix for the specific error mentioned by user:
            # Use line_width instead of width for add_hline
            fig.add_hline(y=0, line_width=1, line_color="white", row=3, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Ratio Chart (Optional)
            if 'ratio_series' in st.session_state:
                st.subheader("Relative Strength (Ratio) Analysis")
                st.line_chart(st.session_state.ratio_series)

        with tab2:
            st.subheader("Strategy Validation (Current Trend)")
            st.write("Backtesting the current logic (EMA + RSI + MACD) on the loaded dataset...")
            
            win_rate, trades_count = run_backtest(df, None)
            
            c1, c2 = st.columns(2)
            c1.metric("Historical Win Rate", f"{win_rate:.1f}%")
            c2.metric("Total Signals Generated", trades_count)
            
            if win_rate < 40:
                st.error("⚠️ The current strategy performs poorly in this market structure. Recommendation: WAIT or reduce position.")
            elif win_rate > 60:
                st.success("✅ High probability setup confirmed by historical data.")

        with tab3:
            st.subheader("News & Context")
            news = fetch_news(ticker)
            if news:
                for n in news:
                    st.markdown(f"**[{n['title']}]({n['link']})**")
                    st.caption(f"Source: {n['publisher']} | {datetime.fromtimestamp(n['providerPublishTime'])}")
                    st.write("---")
            else:
                st.write("No specific news found via API.")

            st.subheader("Human Summary")
            explanation = f"""
            **Why {res['signal']}?**
            
            We analyzed **{ticker_name}** on the **{tf}** timeframe. 
            The market is currently in a **{res['ew']}** phase. 
            
            1. **Trend Alignment:** The Higher Timeframe trend is {action_htf}, and the local trend is {action_ltf}. This creates a **{res['conf']}** scenario.
            2. **Momentum:** RSI is at {df['RSI'].iloc[-1]:.0f}, suggesting {'Oversold' if df['RSI'].iloc[-1] < 30 else 'Overbought' if df['RSI'].iloc[-1] > 70 else 'Neutral'} conditions.
            3. **Divergence:** {res['div']} was detected, which often precedes a reversal.
            
            **Psychology Check:**
            {'Do not chase price. Wait for a pullback to the EMA 20.' if 'Exhaustion' in res['ew'] else 'Momentum is strong, but verify stops.'}
            """
            st.info(explanation)

if __name__ == "__main__":
    main()

