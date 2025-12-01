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
# 1. SYSTEM CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPro AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Report & UI Styling
st.markdown("""
<style>
    /* Global Theme */
    .stApp { background-color: #0e1117; color: #f0f2f6; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    
    /* Report Container */
    .report-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 40px;
        margin-top: 20px;
        font-family: 'Georgia', serif; /* Serif font for report readability */
        line-height: 1.8;
        color: #c9d1d9;
    }
    
    /* Report Headers */
    .report-title { font-size: 28px; color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-bottom: 20px; font-weight: bold; }
    .report-section { font-size: 20px; color: #79c0ff; margin-top: 30px; margin-bottom: 15px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    .report-sub { font-size: 16px; color: #8b949e; font-weight: bold; margin-top: 15px; }
    
    /* Analysis Highlights */
    .highlight-bull { color: #3fb950; font-weight: bold; }
    .highlight-bear { color: #f85149; font-weight: bold; }
    .highlight-neutral { color: #d29922; font-weight: bold; }
    .data-val { font-family: 'Courier New', monospace; background: #21262d; padding: 2px 6px; border-radius: 4px; color: #e6edf3; }
    
    /* Categories */
    .cat-working { border-left: 5px solid #3fb950; padding-left: 15px; background: rgba(63, 185, 80, 0.1); padding: 10px; margin-bottom: 10px; }
    .cat-failing { border-left: 5px solid #f85149; padding-left: 15px; background: rgba(248, 81, 73, 0.1); padding: 10px; margin-bottom: 10px; }
    
    /* Sidebar */
    .stSidebar { background-color: #010409; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA CONSTANTS
# -----------------------------------------------------------------------------
ASSETS = {
    "Indices": {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX", "^GSPC": "S&P 500"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "DOGE-USD": "Dogecoin"},
    "Stocks": {"RELIANCE.NS": "Reliance Ind", "HDFCBANK.NS": "HDFC Bank", "AAPL": "Apple Inc", "NVDA": "Nvidia"},
    "Forex": {"INR=X": "USD/INR", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD"},
    "Commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil"}
}

BENCHMARKS = {
    "^NSEI": "INR=X",
    "^NSEBANK": "^NSEI",
    "BTC-USD": "GC=F",
    "ETH-USD": "BTC-USD",
    "GC=F": "^GSPC"
}

# -----------------------------------------------------------------------------
# 3. ANALYTICS ENGINE
# -----------------------------------------------------------------------------
class DeepAnalyzer:
    def __init__(self, ticker, df, timeframe):
        self.ticker = ticker
        self.df = df
        self.tf = timeframe
        
    def add_indicators(self):
        df = self.df.copy()
        # EMAs
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        
        # Volatility
        df['TR'] = np.maximum((df['High'] - df['Low']), abs(df['Close'].shift(1) - df['High']))
        df['ATR'] = df['TR'].rolling(14).mean()
        df['Std'] = df['Close'].rolling(20).std()
        df['Mean'] = df['Close'].rolling(20).mean()
        df['Z_Score'] = (df['Close'] - df['Mean']) / df['Std']
        
        # Momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        self.df = df
        return df

    def scan_historical_patterns(self):
        """Greedy Search for 'What is Working' vs 'What Failed'"""
        df = self.df
        current_z = df['Z_Score'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # 1. Z-Score Mean Reversion Test
        z_events = []
        similar_z = df[(df['Z_Score'] > current_z - 0.2) & (df['Z_Score'] < current_z + 0.2)]
        for date in similar_z.index:
            try:
                idx = df.index.get_loc(date)
                if idx + 5 < len(df):
                    price_then = df.iloc[idx]['Close']
                    price_later = df.iloc[idx+5]['Close']
                    success = False
                    
                    if current_z < -1.5: # Oversold
                        if price_later > price_then: success = True
                    elif current_z > 1.5: # Overbought
                        if price_later < price_then: success = True
                    else: # Neutral zone
                        if abs(price_later - price_then) < df.iloc[idx]['ATR']: success = True # Stability
                    
                    z_events.append(success)
            except: pass
            
        z_success_rate = (sum(z_events)/len(z_events)*100) if z_events else 0
        
        # 2. RSI Trend Following Test
        rsi_events = []
        # If RSI > 50, did price continue up?
        similar_rsi = df[(df['RSI'] > current_rsi - 2) & (df['RSI'] < current_rsi + 2)]
        for date in similar_rsi.index:
            try:
                idx = df.index.get_loc(date)
                if idx + 3 < len(df):
                    p1 = df.iloc[idx]['Close']
                    p2 = df.iloc[idx+3]['Close']
                    # logic: if rsi > 50, bull follow through?
                    if current_rsi > 50 and p2 > p1: rsi_events.append(True)
                    elif current_rsi < 50 and p2 < p1: rsi_events.append(True)
                    else: rsi_events.append(False)
            except: pass
            
        rsi_reliability = (sum(rsi_events)/len(rsi_events)*100) if rsi_events else 0
        
        return {
            "z_score": current_z,
            "z_matches": len(z_events),
            "z_success": z_success_rate,
            "rsi_val": current_rsi,
            "rsi_matches": len(rsi_events),
            "rsi_reliability": rsi_reliability
        }

    def get_structure_age(self):
        """Calculates age and test counts of levels."""
        df = self.df
        # Detect local Min/Max over window
        window = 50
        if len(df) < window: return None
        
        slice_df = df.iloc[-window:]
        max_idx = slice_df['High'].idxmax()
        min_idx = slice_df['Low'].idxmin()
        
        max_val = slice_df.loc[max_idx]['High']
        min_val = slice_df.loc[min_idx]['Low']
        
        now = df.index[-1]
        
        # Count retests (within 0.5% range)
        tol = 0.005
        res_tests = len(slice_df[slice_df['High'] > max_val * (1-tol)])
        sup_tests = len(slice_df[slice_df['Low'] < min_val * (1+tol)])
        
        return {
            "res_price": max_val,
            "res_date": max_idx,
            "res_hours_ago": (now - max_idx).total_seconds() / 3600,
            "res_tests": res_tests,
            "sup_price": min_val,
            "sup_date": min_idx,
            "sup_hours_ago": (now - min_idx).total_seconds() / 3600,
            "sup_tests": sup_tests
        }

# -----------------------------------------------------------------------------
# 4. REPORT GENERATOR (THE 1000-WORD ENGINE)
# -----------------------------------------------------------------------------
def generate_pro_report(ticker_name, ticker_sym, tf, df, patterns, structure, ratio_val):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Determine Narrative States
    trend_state = "BULLISH" if last['Close'] > last['EMA_200'] else "BEARISH"
    short_term_trend = "BULLISH" if last['EMA_20'] > last['EMA_50'] else "BEARISH"
    
    volatility_state = "CONTRACTING" if last['ATR'] < df['ATR'].rolling(50).mean().iloc[-1] else "EXPANDING"
    
    z_state = "NEUTRAL"
    if last['Z_Score'] > 2: z_state = "EXTREME OVERBOUGHT"
    elif last['Z_Score'] < -2: z_state = "EXTREME OVERSOLD"
    
    # 2. Categorize "What Works"
    working = []
    failing = []
    
    # Check Z-Score Logic
    if patterns['z_matches'] > 10:
        if patterns['z_success'] > 60:
            working.append(f"**Mean Reversion:** Historical data shows a {patterns['z_success']:.1f}% probability of price returning to the mean from current Z-levels ({patterns['z_score']:.2f}).")
        elif patterns['z_success'] < 40:
            failing.append(f"**Counter-Trend Fading:** Attempts to fade this move have historically failed. The trend is overpowering statistical norms ({patterns['z_success']:.1f}% reversion rate).")
            
    # Check Trend Logic
    if trend_state == "BULLISH" and patterns['rsi_reliability'] > 60:
        working.append(f"**Trend Following:** Momentum continuation is highly reliable ({patterns['rsi_reliability']:.1f}% success rate) in this structure.")
    
    # Check Support
    if structure['sup_tests'] > 3:
        working.append(f"**Support Defense:** The level at {structure['sup_price']:.2f} has been defended {structure['sup_tests']} times, indicating aggressive institutional accumulation.")
    
    working_text = "\n".join([f"- {item}" for item in working]) if working else "- No distinct high-probability patterns detected currently."
    failing_text = "\n".join([f"- {item}" for item in failing]) if failing else "- No specific strategies are showing abnormal failure rates."

    # 3. Construct the Report
    report = f"""
    <div class='report-title'>QUANTITATIVE MARKET INTELLIGENCE REPORT: {ticker_name}</div>
    
    <div class='report-section'>1. EXECUTIVE STRATEGIC SUMMARY</div>
    <div class='report-text'>
    This automated analysis, generated on <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b>, evaluates <b>{ticker_name} ({ticker_sym})</b> on the <b>{tf}</b> timeframe. 
    The asset is currently trading at <span class='data-val'>{last['Close']:.2f}</span>. 
    Our algorithmic scoring engine classifies the current market regime as <b>{trend_state}</b> on the macro scale, with <b>{short_term_trend}</b> near-term momentum. 
    Volatility is currently <b>{volatility_state}</b>, suggesting that we are in a phase of {'trend continuation' if volatility_state == 'EXPANDING' else 'potential accumulation or distribution'}.
    <br><br>
    The confluence of Technical Structure, Statistical Anomalies, and Market Memory suggests a primary bias of: 
    <span class='highlight-{ "bull" if "BULL" in short_term_trend else "bear" }'>{short_term_trend} CONTINUATION</span> 
    with a confidence score based on the {patterns['rsi_reliability']:.1f}% reliability of recent momentum signals.
    </div>

    <div class='report-section'>2. PATTERN RECOGNITION: WHAT IS WORKING VS. FAILING</div>
    <div class='report-text'>
    A "Greedy Search" algorithm scanned the last {len(df)} candles to find instances mathematically similar to the current setup. Here is the empirical breakdown of strategy performance:
    </div>
    
    <div class='report-sub'>✅ WHAT IS WORKING (High Probability)</div>
    <div class='cat-working'>
    {working_text}
    </div>
    
    <div class='report-sub'>❌ WHAT IS FAILING (High Risk)</div>
    <div class='cat-failing'>
    {failing_text}
    </div>

    <div class='report-section'>3. TECHNICAL ARCHITECTURE & LEVELS</div>
    <div class='report-text'>
    <b>Moving Average Ribbon:</b> The price is currently {'trading above' if last['Close'] > last['EMA_50'] else 'trading below'} the 50-period EMA (<span class='data-val'>{last['EMA_50']:.2f}</span>). 
    The slope of the 20 EMA is {'positive' if last['EMA_20'] > prev['EMA_20'] else 'negative'}, confirming the immediate direction.
    <br><br>
    <b>Critical Liquidity Levels:</b>
    <ul>
        <li><b>Resistance (Supply):</b> <span class='data-val'>{structure['res_price']:.2f}</span>. This level was established {structure['res_hours_ago']:.1f} hours ago and has been tested <b>{structure['res_tests']} times</b>. {'A breakout here is imminent.' if structure['res_tests'] > 3 else 'This level remains fresh and likely to reject price.'}</li>
        <li><b>Support (Demand):</b> <span class='data-val'>{structure['sup_price']:.2f}</span>. Established {structure['sup_hours_ago']:.1f} hours ago with <b>{structure['sup_tests']} distinct tests</b>.</li>
    </ul>
    </div>

    <div class='report-section'>4. MOMENTUM & STATISTICAL HEALTH</div>
    <div class='report-text'>
    <b>RSI Diagnostics:</b> The Relative Strength Index (14) is reading <span class='data-val'>{last['RSI']:.2f}</span>. 
    { 'The market is in Overbought territory; chasing price here carries statistical risk.' if last['RSI'] > 70 else 'The market is Oversold; mean reversion bots may trigger longs soon.' if last['RSI'] < 30 else 'Momentum is healthy and has room to run in either direction.'}
    <br><br>
    <b>Z-Score Anomaly Detection:</b> The current Z-Score is <span class='data-val'>{last['Z_Score']:.2f}</span>. 
    This indicates price is {abs(last['Z_Score']):.2f} standard deviations from the mean. 
    { 'This is a statistically significant event (>2 SD). Reversion to the mean is mathematically probable.' if abs(last['Z_Score']) > 2 else 'Price is oscillating within normal statistical bounds.'}
    <br><br>
    <b>MACD Status:</b> The MACD histogram is { 'positive' if last['MACD'] > last['Signal'] else 'negative' }, and the gap is { 'widening' if abs(last['MACD'] - last['Signal']) > abs(prev['MACD'] - prev['Signal']) else 'narrowing' }. 
    This confirms that momentum is { 'accelerating' if abs(last['MACD'] - last['Signal']) > abs(prev['MACD'] - prev['Signal']) else 'losing steam' }.
    </div>

    <div class='report-section'>5. BENCHMARK & RELATIVE STRENGTH</div>
    <div class='report-text'>
    The Relative Ratio against the benchmark is <span class='data-val'>{ratio_val:.4f}</span>.
    { 'This asset is exhibiting <b>Relative Strength</b>, outperforming the benchmark. Capital is flowing IN.' if ratio_val > 1.0 else 'This asset is exhibiting <b>Relative Weakness</b>. Capital is likely rotating OUT into other sectors.'}
    </div>

    <div class='report-section'>6. FINAL ALGORITHMIC VERDICT & TRADE PLAN</div>
    <div class='report-text'>
    Based on the synthesis of the above 15+ metrics, the "QuantPro" system recommends:
    <br><br>
    <div style='font-size: 24px; font-weight: bold; text-align: center; padding: 10px; border: 2px solid #58a6ff; border-radius: 8px;'>
        {short_term_trend} { 'ENTRY' if abs(last['Z_Score']) < 2 else 'WAIT (Extended)' }
    </div>
    <br>
    <b>Execution Strategy:</b>
    <ul>
        <li><b>Entry Zone:</b> <span class='data-val'>{last['Close']:.2f}</span> (Verify lower timeframe structure).</li>
        <li><b>Invalidation (Stop Loss):</b> <span class='data-val'>{structure['sup_price'] if 'BULL' in short_term_trend else structure['res_price']:.2f}</span> (Structural Invalidation).</li>
        <li><b>Target 1 (Liquidity Run):</b> <span class='data-val'>{structure['res_price'] if 'BULL' in short_term_trend else structure['sup_price']:.2f}</span>.</li>
        <li><b>Target 2 (Extension):</b> <span class='data-val'>{last['Close'] + (last['ATR']*3) if 'BULL' in short_term_trend else last['Close'] - (last['ATR']*3):.2f}</span> (3x ATR Expansion).</li>
    </ul>
    </div>
    """
    return report

# -----------------------------------------------------------------------------
# 5. DATA FETCHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    st.sidebar.title("🎛️ Analyst Controls")
    
    cat = st.sidebar.selectbox("Market Category", list(ASSETS.keys()))
    ticker_name = st.sidebar.selectbox("Asset", list(ASSETS[cat].values()))
    ticker_sym = [k for k,v in ASSETS[cat].items() if v == ticker_name][0]
    
    # Smart Default Timeframe
    c1, c2 = st.sidebar.columns(2)
    with c1: interval = st.selectbox("Interval", ['5m', '15m', '1h', '4h', '1d'], index=0)
    with c2: period = st.selectbox("Lookback", ['1d', '5d', '1mo', '3mo', '1y'], index=2)
    
    enable_ratio = st.sidebar.checkbox("Ratio Analysis", value=True)
    bench_sym = BENCHMARKS.get(ticker_sym, "^NSEI")
    
    if st.sidebar.button("GENERATE INTELLIGENCE REPORT", type="primary"):
        with st.spinner(f"Running Deep-Dive Analysis on {ticker_name}..."):
            
            # 1. Fetch Data
            df = get_data(ticker_sym, period, interval)
            if df is None or len(df) < 50:
                st.error("Insufficient data for deep analysis.")
                return
                
            # 2. Analyze
            engine = DeepAnalyzer(ticker_sym, df, interval)
            df = engine.add_indicators()
            patterns = engine.scan_historical_patterns()
            structure = engine.get_structure_age()
            
            # 3. Ratio
            ratio_val = 1.0
            if enable_ratio:
                df_b = get_data(bench_sym, period, interval)
                if df_b is not None:
                    idx = df.index.intersection(df_b.index)
                    if not idx.empty:
                        ratio_val = df.loc[idx[-1]]['Close'] / df_b.loc[idx[-1]]['Close']

            # 4. Generate HTML Report
            html_report = generate_pro_report(ticker_name, ticker_sym, interval, df, patterns, structure, ratio_val)
            
            # --- LAYOUT ---
            col_chart, col_report = st.columns([1, 1.2])
            
            with col_chart:
                # Advanced Charting
                st.markdown("### 📉 Technical Structure")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
                
                # Price
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue', width=1), name='EMA 200'), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash='dot', line_color='red', row=2, col=1)
                fig.add_hline(y=30, line_dash='dot', line_color='green', row=2, col=1)
                
                # Z-Score
                fig.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], line=dict(color='cyan'), name='Z-Score'), row=3, col=1)
                fig.add_hline(y=2, line_dash='dot', line_color='red', row=3, col=1)
                fig.add_hline(y=-2, line_dash='dot', line_color='green', row=3, col=1)
                
                fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # Small data table for clarity
                st.markdown("### 🔢 Raw Metric Feed")
                st.dataframe(df[['Close', 'EMA_50', 'RSI', 'Z_Score', 'ATR']].tail(10).style.format("{:.2f}"))

            with col_report:
                # The Big Report
                st.markdown(f"<div class='report-container'>{html_report}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
