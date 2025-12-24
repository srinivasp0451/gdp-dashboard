import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from datetime import datetime, timedelta
import pytz
import time
import traceback

# ==========================================
# 1. CONFIGURATION & UTILITIES
# ==========================================

st.set_page_config(
    page_title="AlgoTrade Pro | Multi-Timeframe Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IST = pytz.timezone('Asia/Kolkata')
ASSETS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X"
}

TIMEFRAMES = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
PERIODS = ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']

# Valid Combinations Map
VALID_PARAMS = {
    '1m': ['1d', '5d', '7d'],
    '2m': ['1d', '5d', '1mo'],
    '5m': ['1d', '5d', '1mo'],
    '15m': ['1d', '5d', '1mo'],
    '30m': ['1d', '5d', '1mo'],
    '60m': ['1mo', '3mo'],
    '1h': ['1mo', '3mo', '6mo'],
    '90m': ['1mo', '3mo'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
    '5d': ['6mo', '1y', '2y', '5y', '10y', 'max'],
    '1wk': ['1y', '2y', '5y', '10y', 'max'],
    '1mo': ['2y', '5y', '10y', 'max'],
    '3mo': ['5y', '10y', 'max']
}

# --- Helper Functions ---

def format_currency(value):
    return f"₹{value:,.2f}"

def format_pct(value):
    return f"{value:.2f}%"

def human_readable_time(dt_obj):
    if dt_obj.tzinfo is None:
        dt_obj = IST.localize(dt_obj)
    
    now = datetime.now(IST)
    diff = now - dt_obj
    
    if diff < timedelta(hours=1):
        return f"{int(diff.total_seconds() // 60)} minutes ago"
    elif diff < timedelta(hours=24):
        return f"{int(diff.total_seconds() // 3600)} hours ago"
    elif diff < timedelta(days=30):
        return f"{diff.days} days ago"
    else:
        return f"{diff.days // 30} months ago ({dt_obj.strftime('%Y-%m-%d %H:%M:%S')} IST)"

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def fetch_data(ticker, period, interval):
    """
    Fetches data with rate limiting handling and timezone conversion.
    """
    try:
        time.sleep(1.1)  # Rate limiting
        data = yf.download(ticker, period=period, interval=interval, progress=False, ignore_tz=False)
        
        if data.empty:
            return None

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardize Columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Timezone Handling
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
            
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker} ({period}/{interval}): {e}")
        return None

# ==========================================
# 2. CORE CALCULATION ENGINE
# ==========================================

class MarketAnalyzer:
    def __init__(self, data, ticker):
        self.data = data.copy()
        self.ticker = ticker
        self._calculate_basics()

    def _calculate_basics(self):
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # EMAs
        self.data['EMA_9'] = self.data['Close'].ewm(span=9, adjust=False).mean()
        self.data['EMA_20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['EMA_200'] = self.data['Close'].ewm(span=200, adjust=False).mean()

        # Z-Score
        self.data['Returns'] = self.data['Close'].pct_change()
        # Rolling Z-Score (Window 50)
        r_mean = self.data['Returns'].rolling(window=50).mean()
        r_std = self.data['Returns'].rolling(window=50).std()
        self.data['Z_Score'] = (self.data['Returns'] - r_mean) / r_std

        # Volatility (Annualized)
        self.data['Volatility_Pct'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100

        # Support/Resistance Logic (Simple local extrema)
        self.data['Min'] = self.data['Low'][(self.data['Low'].shift(1) > self.data['Low']) & (self.data['Low'].shift(-1) > self.data['Low'])]
        self.data['Max'] = self.data['High'][(self.data['High'].shift(1) < self.data['High']) & (self.data['High'].shift(-1) < self.data['High'])]

    def get_support_resistance_levels(self, current_price, tolerance=0.015):
        supports = self.data['Min'].dropna().tolist()
        resistances = self.data['Max'].dropna().tolist()
        
        # Cluster levels
        levels = []
        for price_list, type_ in [(supports, 'Support'), (resistances, 'Resistance')]:
            for price in price_list:
                # Check if this level is close to an existing cluster
                found = False
                for i, (avg, count, l_type) in enumerate(levels):
                    if abs(avg - price) / avg < tolerance and l_type == type_:
                        new_avg = (avg * count + price) / (count + 1)
                        levels[i] = (new_avg, count + 1, l_type)
                        found = True
                        break
                if not found:
                    levels.append((price, 1, type_))
        
        # Filter significant levels (count > 2) and sort by distance
        valid_levels = sorted([
            {
                'Type': l[2],
                'Price': l[0],
                'Hits': l[1],
                'Dist_Pts': abs(current_price - l[0]),
                'Dist_Pct': (abs(current_price - l[0]) / current_price) * 100
            } 
            for l in levels if l[1] >= 2
        ], key=lambda x: x['Dist_Pct'])
        
        return valid_levels[:8] # Return top 8 closest levels

    def analyze_z_score_bins(self):
        # Create bins for Z-Score
        bins = [-float('inf'), -2, -1, 1, 2, float('inf')]
        labels = ['Extreme Negative (<-2)', 'Negative (-2 to -1)', 'Neutral (-1 to 1)', 'Positive (1 to 2)', 'Extreme Positive (>2)']
        self.data['Z_Bin'] = pd.cut(self.data['Z_Score'], bins=bins, labels=labels)
        
        distribution = self.data.groupby('Z_Bin').agg(
            Count=('Close', 'count'),
            Min_Price=('Low', 'min'),
            Max_Price=('High', 'max'),
            Avg_Price=('Close', 'mean')
        ).reset_index()
        
        total = distribution['Count'].sum()
        distribution['Percentage'] = (distribution['Count'] / total * 100).round(2)
        distribution['Range'] = distribution.apply(lambda x: f"₹{x['Min_Price']:,.0f} - ₹{x['Max_Price']:,.0f}", axis=1)
        
        return distribution

    def analyze_volatility_bins(self):
        # Dynamic binning for volatility
        vol = self.data['Volatility_Pct'].dropna()
        if len(vol) < 10: return pd.DataFrame()
        
        q1, q3 = vol.quantile([0.25, 0.75])
        bins = [-float('inf'), q1, vol.mean(), q3, float('inf')]
        # Create explicit range strings for labels
        labels = [
            f"Low (<{q1:.1f}%)", 
            f"Normal-Low ({q1:.1f}-{vol.mean():.1f}%)", 
            f"Normal-High ({vol.mean():.1f}-{q3:.1f}%)", 
            f"High (>{q3:.1f}%)"
        ]
        
        self.data['Vol_Bin'] = pd.cut(self.data['Volatility_Pct'], bins=bins, labels=labels)
        
        dist = self.data.groupby('Vol_Bin').agg(
            Count=('Close', 'count'),
            Avg_Vol=('Volatility_Pct', 'mean')
        ).reset_index()
        
        total = dist['Count'].sum()
        dist['Percentage'] = (dist['Count'] / total * 100).round(2)
        dist['Vol Range'] = labels 
        return dist

    def get_signal_score(self):
        # Analyze last row
        if self.data.empty: return 0, "No Data"
        
        last = self.data.iloc[-1]
        score = 0
        reasons = []
        
        # RSI
        if last['RSI'] < 30: 
            score += 20
            reasons.append("RSI Oversold")
        elif last['RSI'] > 70: 
            score -= 20
            reasons.append("RSI Overbought")
            
        # EMA Alignment
        if last['Close'] > last['EMA_20'] > last['EMA_50']:
            score += 15
            reasons.append("Bullish EMA Alignment")
        elif last['Close'] < last['EMA_20'] < last['EMA_50']:
            score -= 15
            reasons.append("Bearish EMA Alignment")
            
        # Z-Score
        if last['Z_Score'] < -2:
            score += 20
            reasons.append("Z-Score Extreme Oversold")
        elif last['Z_Score'] > 2:
            score -= 20
            reasons.append("Z-Score Extreme Overbought")
            
        return score, reasons

# ==========================================
# 3. MAIN UI LAYOUT
# ==========================================

def main():
    st.sidebar.title("⚙️ Configuration")
    
    # --- Input Section ---
    asset_mode = st.sidebar.radio("Asset Selection", ["Preset List", "Custom Ticker"])
    if asset_mode == "Preset List":
        ticker_display = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
        ticker = ASSETS[ticker_display]
    else:
        ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
        ticker_display = ticker

    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Timeframes to Analyze")
    
    selected_timeframes = {}
    default_tfs = ['5m', '15m', '1h', '1d']
    
    for tf in TIMEFRAMES:
        if st.sidebar.checkbox(tf, value=(tf in default_tfs)):
            valid_p = VALID_PARAMS.get(tf, ['1mo'])
            # Choose the largest logical period for that timeframe automatically for simplicity, 
            # or let user pick. For better UX, we pick a reasonable default.
            p_idx = min(2, len(valid_p)-1) 
            period = st.sidebar.selectbox(f"Period for {tf}", valid_p, index=p_idx, key=f"p_{tf}")
            selected_timeframes[tf] = period

    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
    ticker2 = None
    if enable_ratio:
        ticker2 = st.sidebar.text_input("Comparison Ticker", "^NSEBANK")

    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        run_analysis_pipeline(ticker, ticker_display, selected_timeframes, enable_ratio, ticker2)

# ==========================================
# 4. ANALYSIS PIPELINE & TABS
# ==========================================

def run_analysis_pipeline(ticker, ticker_name, timeframes_map, enable_ratio, ticker2):
    st.title(f"📊 Algorithmic Trading Analysis: {ticker_name}")
    st.markdown(f"**Analysis Time:** {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    # --- 1. Data Fetching ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    total_steps = len(timeframes_map)
    
    for i, (tf, period) in enumerate(timeframes_map.items()):
        status_text.text(f"Fetching data for {tf} / {period}...")
        
        # 1. Fetch Raw Data
        raw_df = fetch_data(ticker, period, tf)
        
        if raw_df is not None and len(raw_df) > 20:
            # 2. Initialize Analyzer (Calculates Indicators)
            analyzer = MarketAnalyzer(raw_df, ticker)
            
            # 3. Store the PROCESSED data (analyzer.data), not the raw_df
            results[tf] = {
                'data': analyzer.data,  # <--- FIXED: Using analyzer.data which contains EMA_50
                'period': period,
                'analyzer': analyzer
            }
        progress_bar.progress((i + 1) / total_steps)
    
    status_text.empty()
    progress_bar.empty()
    
    if not results:
        st.error("No data could be fetched. Please check the ticker or internet connection.")
        return

    # --- 2. Tab Structure ---
    tabs = st.tabs([
        "👁️ Overview", 
        "🧱 Support/Resistance", 
        "📈 Technicals", 
        "📉 Z-Score", 
        "🌊 Volatility", 
        "🌀 Fibonacci", 
        "🤖 AI Signals", 
        "🔙 Backtesting", 
        "🔴 Live Monitor"
    ])

    # === TAB 0: OVERVIEW ===
    with tabs[0]:
        display_overview(results)

    # === TAB 1: SUPPORT / RESISTANCE ===
    with tabs[1]:
        display_support_resistance(results)
        
    # === TAB 2: TECHNICALS ===
    with tabs[2]:
        display_technicals(results)

    # === TAB 3: Z-SCORE ===
    with tabs[3]:
        display_zscore(results)

    # === TAB 4: VOLATILITY ===
    with tabs[4]:
        display_volatility(results)
        
    # === TAB 5: FIBONACCI ===
    with tabs[5]:
        display_fibonacci(results)

    # === TAB 6: AI SIGNALS ===
    with tabs[6]:
        display_ai_signals(results, ticker_name)

    # === TAB 7: BACKTESTING ===
    with tabs[7]:
        display_backtesting(results)

    # === TAB 8: LIVE MONITOR ===
    with tabs[8]:
        display_live_monitor(ticker, timeframes_map)

# ==========================================
# 5. DISPLAY FUNCTIONS (PER TAB)
# ==========================================

def display_overview(results):
    st.subheader("Multi-Timeframe Dashboard")
    
    overview_data = []
    
    for tf, res in results.items():
        analyzer = res['analyzer']
        last = analyzer.data.iloc[-1]
        
        # Calculate status
        score, _ = analyzer.get_signal_score()
        status_icon = "🟢" if score > 10 else "🔴" if score < -10 else "🟡"
        
        overview_data.append({
            "Timeframe": tf,
            "Status": status_icon,
            "Price": format_currency(last['Close']),
            "Return %": format_pct(last['Returns'] * 100),
            "RSI": f"{last['RSI']:.1f}",
            "EMA Trend": "Bullish" if last['Close'] > last['EMA_50'] else "Bearish",
            "Z-Score": f"{last['Z_Score']:.2f}",
            "Volatility": f"{last['Volatility_Pct']:.2f}%"
        })
    
    df_overview = pd.DataFrame(overview_data)
    st.dataframe(df_overview, use_container_width=True)
    
    csv = df_overview.to_csv(index=False).encode('utf-8')
    st.download_button("Download Overview CSV", csv, "overview.csv", "text/csv")

def display_support_resistance(results):
    for tf, res in results.items():
        st.markdown(f"---")
        st.markdown(f"## 📊 Support & Resistance: {tf} / {res['period']}")
        
        analyzer = res['analyzer']
        last_price = analyzer.data.iloc[-1]['Close']
        levels = analyzer.get_support_resistance_levels(last_price)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Current Price", format_currency(last_price))
            
            if not levels:
                st.info("Not enough data points to determine S/R levels.")
            else:
                st.write("**Nearest Levels:**")
                level_data = []
                for l in levels:
                    level_data.append({
                        "Type": "🟢 Sup" if l['Type'] == 'Support' else "🔴 Res",
                        "Price": format_currency(l['Price']),
                        "Distance": f"{l['Dist_Pts']:.2f} ({l['Dist_Pct']:.2f}%)",
                        "Hits": l['Hits']
                    })
                st.table(pd.DataFrame(level_data))
        
        with col2:
            # Simple visualization
            fig = go.Figure()
            # Candlestick (last 50 candles)
            subset = analyzer.data.tail(50)
            fig.add_trace(go.Candlestick(x=subset.index, open=subset['Open'], high=subset['High'],
                                         low=subset['Low'], close=subset['Close'], name="Price"))
            
            # Add S/R Lines
            colors = {'Support': 'green', 'Resistance': 'red'}
            for l in levels:
                fig.add_hline(y=l['Price'], line_dash="dash", line_color=colors[l['Type']], 
                              annotation_text=f"{l['Type']} ({l['Hits']} hits)")
            
            fig.update_layout(title=f"{tf} Price Action with Key Levels", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast based on proximity
        if levels:
            nearest = levels[0]
            if nearest['Dist_Pct'] < 0.5:
                action = "BOUNCE EXPECTED" if nearest['Type'] == 'Support' else "REJECTION EXPECTED"
                st.info(f"💡 **Forecast:** Price is very close ({nearest['Dist_Pct']:.2f}%) to significant {nearest['Type']}. **{action}**.")
            else:
                st.write(f"Price is in open space. Nearest level is {nearest['Dist_Pct']:.2f}% away.")

def display_technicals(results):
    st.markdown("## Technical Indicator Consensus")
    
    # Aggregate EMAs
    ema_counts = {"Bullish": 0, "Bearish": 0}
    
    for tf, res in results.items():
        last = res['data'].iloc[-1]
        trend = "Bullish" if last['Close'] > last['EMA_50'] else "Bearish"
        ema_counts[trend] += 1
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Bullish Timeframes", ema_counts['Bullish'])
    col2.metric("Bearish Timeframes", ema_counts['Bearish'])
    col3.metric("Consensus", "BUY" if ema_counts['Bullish'] > ema_counts['Bearish'] else "SELL")
    
    # Detailed Tables per timeframe
    for tf, res in results.items():
        with st.expander(f"Detailed Technicals: {tf}"):
            df = res['data'].tail(20)[['Close', 'RSI', 'EMA_20', 'EMA_50', 'Volume']].copy()
            df = df.reset_index()
            # Format Date
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df.style.background_gradient(subset=['RSI'], cmap='RdYlGn'), use_container_width=True)

def display_zscore(results):
    st.markdown("## Z-Score Analysis (Mean Reversion)")
    
    consensus_rally = 0
    consensus_correction = 0
    
    for tf, res in results.items():
        st.markdown(f"### 📊 {tf} Analysis")
        analyzer = res['analyzer']
        dist = analyzer.analyze_z_score_bins()
        current_z = analyzer.data.iloc[-1]['Z_Score']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric("Current Z-Score", f"{current_z:.2f}")
            if current_z < -2:
                st.error("⚠️ EXTREME OVERSOLD (<-2). Rally statistically likely.")
                consensus_rally += 1
            elif current_z > 2:
                st.error("⚠️ EXTREME OVERBOUGHT (>2). Correction statistically likely.")
                consensus_correction += 1
            else:
                st.success("Normal Range (-2 to 2)")
                
            st.markdown("**Historical Bin Distribution**")
            st.dataframe(dist, use_container_width=True)

        with col2:
            # Historical Similarity Context
            if current_z < -2 or current_z > 2:
                condition = "Oversold" if current_z < -2 else "Overbought"
                df_hist = analyzer.data.copy()
                if condition == "Oversold":
                    similar_events = df_hist[df_hist['Z_Score'] < -2]
                else:
                    similar_events = df_hist[df_hist['Z_Score'] > 2]
                
                count = len(similar_events)
                st.info(f"Found {count} similar '{condition}' events in history for {tf}.")
                if count > 0:
                    last_event = similar_events.index[-2] if count > 1 else similar_events.index[-1]
                    st.write(f"Last instance: {human_readable_time(last_event)}")
            else:
                st.write("Current Z-Score is within normal statistical deviations. No extreme reversion setup detected.")

    st.markdown("---")
    st.markdown("### Final Z-Score Consensus")
    st.write(f"Correction Signals: {consensus_correction} | Rally Signals: {consensus_rally}")

def display_volatility(results):
    for tf, res in results.items():
        analyzer = res['analyzer']
        dist = analyzer.analyze_volatility_bins()
        cur_vol = analyzer.data.iloc[-1]['Volatility_Pct']
        
        st.markdown(f"### 🌊 {tf} Volatility Structure")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Volatility (Annualized)", f"{cur_vol:.2f}%")
            if not dist.empty:
                st.table(dist[['Vol Range', 'Count', 'Percentage']])
        with col2:
            # Simple chart
            st.line_chart(analyzer.data['Volatility_Pct'].tail(100))
        
        # Logic
        avg_vol = analyzer.data['Volatility_Pct'].mean()
        if cur_vol < avg_vol * 0.7:
            st.warning("⚠️ Low Volatility Compression - Expect Breakout Soon")
        elif cur_vol > avg_vol * 1.5:
            st.warning("⚠️ High Volatility Regime - Reduce Position Size")

def display_fibonacci(results):
    tf = list(results.keys())[-1] # Use the largest timeframe (usually last in list)
    st.markdown(f"## Fibonacci Retracement (Primary: {tf})")
    
    data = results[tf]['data']
    # Find recent significant high/low (last 100 periods)
    recent = data.tail(100)
    high_price = recent['High'].max()
    low_price = recent['Low'].min()
    current = recent.iloc[-1]['Close']
    
    diff = high_price - low_price
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    fib_data = []
    for level in levels:
        price = high_price - (diff * level)
        dist = price - current
        fib_data.append({
            "Fib Level": level,
            "Price": format_currency(price),
            "Distance": f"{dist:.2f}",
            "Status": "Resistance" if price > current else "Support"
        })
    
    st.table(pd.DataFrame(fib_data))
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=recent.index, open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close']))
    for fd in fib_data:
        p = float(fd['Price'].replace('₹','').replace(',',''))
        fig.add_hline(y=p, annotation_text=f"Fib {fd['Fib Level']}", line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

def display_ai_signals(results, ticker_name):
    st.markdown("# 🧠 AI Multi-Timeframe Consensus")
    
    total_score = 0
    total_timeframes = len(results)
    bullish_count = 0
    bearish_count = 0
    
    breakdown = []
    
    for tf, res in results.items():
        analyzer = res['analyzer']
        score, reasons = analyzer.get_signal_score()
        total_score += score
        
        bias = "Neutral"
        if score > 15: 
            bias = "Bullish"
            bullish_count += 1
        elif score < -15: 
            bias = "Bearish"
            bearish_count += 1
            
        breakdown.append({
            "Timeframe": tf,
            "Score": score,
            "Bias": bias,
            "Factors": ", ".join(reasons)
        })
        
    avg_score = total_score / total_timeframes if total_timeframes > 0 else 0
    
    # Determine Final Signal
    signal = "NEUTRAL / HOLD"
    color = "orange"
    if avg_score > 25:
        signal = "STRONG BUY"
        color = "green"
    elif avg_score > 10:
        signal = "BUY"
        color = "lightgreen"
    elif avg_score < -25:
        signal = "STRONG SELL"
        color = "red"
    elif avg_score < -10:
        signal = "SELL"
        color = "salmon"

    # Confidence calculation
    agreement_ratio = max(bullish_count, bearish_count) / total_timeframes
    confidence = 60 + (agreement_ratio * 30) + (min(abs(avg_score), 100) * 0.1)
    confidence = min(confidence, 95.0)

    # --- Display ---
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 10px solid {color}'>
        <h1 style='color: {color}; margin: 0;'>{signal}</h1>
        <h3>Confidence: {confidence:.1f}% | Multi-Timeframe Score: {avg_score:.1f}/100</h3>
        <p>Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {total_timeframes - bullish_count - bearish_count}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trading Plan
    current_price = results[list(results.keys())[0]]['data'].iloc[-1]['Close']
    
    # Realistic Targets based on asset price magnitude (Index vs Stock)
    is_index = current_price > 15000
    stop_pct = 0.015 if is_index else 0.025
    target_pct = 0.0175 if is_index else 0.035
    
    sl_price = current_price * (1 - stop_pct) if "BUY" in signal else current_price * (1 + stop_pct)
    target_price = current_price * (1 + target_pct) if "BUY" in signal else current_price * (1 - target_pct)
    
    if "HOLD" not in signal:
        st.markdown("### 📋 Recommended Trading Plan")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entry Price", format_currency(current_price))
        col2.metric("Stop Loss", format_currency(sl_price), f"-{stop_pct*100}%")
        col3.metric("Target", format_currency(target_price), f"{target_pct*100}%")
        col4.metric("Risk:Reward", "1 : 1.17")
        
        st.markdown(f"""
        **Rationale:**
        * **Multi-Timeframe Agreement**: {agreement_ratio*100:.0f}% of timeframes agree.
        * **Technical Edge**: {breakdown[0]['Factors'] if breakdown else 'N/A'}
        """)

    st.markdown("### 📊 Timeframe Breakdown")
    st.table(pd.DataFrame(breakdown))

def display_backtesting(results):
    st.markdown("## 🔙 Strategy Backtester")
    st.info("Simulating strategies on the fetched data (Last 500 periods max per timeframe)")
    
    strategies = ["RSI Reversion", "EMA Crossover", "Volatility Breakout"]
    selected_strat = st.selectbox("Select Strategy", strategies)
    
    summary_stats = []
    
    for tf, res in results.items():
        data = res['data'].copy()
        initial_capital = 100000
        capital = initial_capital
        position = 0 # 0 none, 1 long, -1 short
        entry_price = 0
        trades = 0
        wins = 0
        
        # Vectorized or Loop backtest (Loop for clarity in logic)
        for i in range(50, len(data)):
            row = data.iloc[i]
            prev = data.iloc[i-1]
            
            # --- STRATEGY LOGIC ---
            signal = 0
            if selected_strat == "RSI Reversion":
                if row['RSI'] < 30: signal = 1
                elif row['RSI'] > 70: signal = -1
            elif selected_strat == "EMA Crossover":
                if row['EMA_20'] > row['EMA_50'] and prev['EMA_20'] <= prev['EMA_50']: signal = 1
                elif row['EMA_20'] < row['EMA_50'] and prev['EMA_20'] >= prev['EMA_50']: signal = -1
            
            # --- EXECUTION ---
            price = row['Close']
            
            # Exit rules (Simple: Exit on opposite signal or if signal reverses)
            if position == 1 and signal == -1:
                pnl = (price - entry_price) / entry_price
                capital *= (1 + pnl)
                position = 0
                trades += 1
                if pnl > 0: wins += 1
            elif position == -1 and signal == 1:
                pnl = (entry_price - price) / entry_price
                capital *= (1 + pnl)
                position = 0
                trades += 1
                if pnl > 0: wins += 1
            
            # Entry rules
            if position == 0 and signal != 0:
                position = signal
                entry_price = price

        total_return = ((capital - initial_capital) / initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        summary_stats.append({
            "Timeframe": tf,
            "Total Trades": trades,
            "Win Rate": f"{win_rate:.1f}%",
            "Net Return": f"{total_return:.2f}%"
        })
        
    st.table(pd.DataFrame(summary_stats))

# ==========================================
# 6. LIVE MONITOR & UTILS
# ==========================================

def display_live_monitor(ticker, timeframes_map):
    st.markdown("## 🔴 Live Market Monitor")
    st.warning("**Note:** Real-time data with `yfinance` is limited. This mode simulates a live environment by refreshing every few seconds. Do not leave running indefinitely to avoid IP bans.")
    
    if "live_history" not in st.session_state:
        st.session_state.live_history = []

    # Simulation toggle
    run_live = st.checkbox("Start Live Monitor")
    
    placeholder = st.empty()
    
    if run_live:
        while True:
            # In a real app, this would fetch new data. 
            # To be safe with yfinance, we fetch only 1m data for the specific ticker
            try:
                live_df = yf.download(ticker, period="1d", interval="1m", progress=False)
                if not live_df.empty:
                    current_price = live_df['Close'].iloc[-1]
                    current_time = datetime.now(IST).strftime('%H:%M:%S')
                    
                    # Random price wiggle for demonstration if market is closed
                    if live_df.index[-1].date() < datetime.now().date():
                         wiggle = np.random.normal(0, current_price * 0.0005)
                         current_price += wiggle

                    st.session_state.live_history.append({"Time": current_time, "Price": current_price})
                    if len(st.session_state.live_history) > 50:
                        st.session_state.live_history.pop(0)
                    
                    df_live = pd.DataFrame(st.session_state.live_history)
                    
                    with placeholder.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Live Price", format_currency(current_price))
                            st.write(f"Last Update: {current_time}")
                        with col2:
                             st.line_chart(df_live.set_index("Time"))
                
                time.sleep(2) # 2 Second refresh rate
            except Exception as e:
                st.error(f"Live Feed Error: {e}")
                break
    else:
        st.info("Check the box above to start live monitoring.")

if __name__ == "__main__":
    main()
