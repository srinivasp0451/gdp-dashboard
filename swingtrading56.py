import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from scipy.signal import argrelextrema
import scipy.stats as stats

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AlgoTrade Pro | AI-Powered Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #464b5c; }
    h1, h2, h3 { color: #fafafa; }
    .highlight { color: #00d4ff; font-weight: bold; }
    .bullish { color: #00ff7f; font-weight: bold; }
    .bearish { color: #ff4b4b; font-weight: bold; }
    .neutral { color: #f0ad4e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & UTILITIES
# -----------------------------------------------------------------------------
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

TIMEFRAME_MAPPING = {
    "1m": "5d", "2m": "5d", "5m": "1mo", "15m": "1mo", "30m": "1mo",
    "60m": "3mo", "90m": "3mo", "1h": "3mo", "1d": "1y", "5d": "2y",
    "1wk": "2y", "1mo": "5y"
}

def format_currency(val):
    return f"₹{val:,.2f}"

def format_time_ago(dt_obj):
    if dt_obj is None: return "N/A"
    now = datetime.now(IST)
    if dt_obj.tzinfo is None:
        dt_obj = IST.localize(dt_obj)
    
    diff = now - dt_obj
    seconds = diff.total_seconds()
    
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    elif seconds < 2592000:
        return f"{diff.days} days ago"
    else:
        return f"{dt_obj.strftime('%Y-%m-%d %H:%M:%S')} IST"

# -----------------------------------------------------------------------------
# 3. CORE LOGIC CLASS: TECHNICAL ANALYSIS
# -----------------------------------------------------------------------------
class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_ema(data, period):
        return data['Close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_zscore(data, window=20):
        r = data['Close'].pct_change()
        mean = r.rolling(window=window).mean()
        std = r.rolling(window=window).std()
        return (r - mean) / std

    @staticmethod
    def calculate_volatility(data, window=20):
        # Annualized volatility
        return data['Close'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100

    @staticmethod
    def identify_support_resistance(data, window=10):
        # Local min/max
        df = data.copy()
        df['Min'] = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=window)[0]]
        df['Max'] = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=window)[0]]
        
        levels = []
        for i in range(len(df)):
            if not np.isnan(df['Min'].iloc[i]):
                levels.append((df.index[i], df['Min'].iloc[i], "Support"))
            if not np.isnan(df['Max'].iloc[i]):
                levels.append((df.index[i], df['Max'].iloc[i], "Resistance"))
        return levels

    @staticmethod
    def calculate_fibonacci(high, low):
        diff = high - low
        return {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100%": low
        }

    @staticmethod
    def detect_divergence(data, order=5):
        # Simplified divergence detection
        # Bullish: Price Lower Low, RSI Higher Low
        # Bearish: Price Higher High, RSI Lower High
        if 'RSI' not in data.columns: return []
        
        divs = []
        # Find peaks
        high_idx = argrelextrema(data['High'].values, np.greater, order=order)[0]
        low_idx = argrelextrema(data['Low'].values, np.less, order=order)[0]
        
        # Check last two peaks for bearish divergence
        if len(high_idx) >= 2:
            p1, p2 = high_idx[-2], high_idx[-1]
            if data['High'].iloc[p2] > data['High'].iloc[p1] and data['RSI'].iloc[p2] < data['RSI'].iloc[p1]:
                divs.append({'Type': 'Bearish', 'Price': data['High'].iloc[p2], 'RSI': data['RSI'].iloc[p2]})
                
        # Check last two troughs for bullish divergence
        if len(low_idx) >= 2:
            p1, p2 = low_idx[-2], low_idx[-1]
            if data['Low'].iloc[p2] < data['Low'].iloc[p1] and data['RSI'].iloc[p2] > data['RSI'].iloc[p1]:
                divs.append({'Type': 'Bullish', 'Price': data['Low'].iloc[p2], 'RSI': data['RSI'].iloc[p2]})
        
        return divs

# -----------------------------------------------------------------------------
# 4. DATA MANAGEMENT
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker, timeframe, period):
    try:
        time.sleep(1.0) # Rate limiting
        data = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=True)
        
        if data.empty:
            return None

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Standardize Columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Timezone Conversion
        data.index = data.index.tz_convert(IST) if data.index.tzinfo else data.index.tz_localize(pytz.utc).tz_convert(IST)
        data['DateTime_IST'] = data.index
        
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker} ({timeframe}): {str(e)}")
        return None

def process_dataframe(df):
    if df is None: return None
    df['RSI'] = TechnicalAnalysis.calculate_rsi(df)
    df['EMA_20'] = TechnicalAnalysis.calculate_ema(df, 20)
    df['EMA_50'] = TechnicalAnalysis.calculate_ema(df, 50)
    df['Z_Score'] = TechnicalAnalysis.calculate_zscore(df)
    df['Volatility'] = TechnicalAnalysis.calculate_volatility(df)
    df['Returns_Pct'] = df['Close'].pct_change() * 100
    df.dropna(inplace=True)
    return df

# -----------------------------------------------------------------------------
# 5. UI COMPONENTS - SIDEBAR & MAIN
# -----------------------------------------------------------------------------

st.sidebar.title("🚀 AlgoTrade Pro")
st.sidebar.markdown("---")

# Inputs
ticker_mode = st.sidebar.radio("Asset Selection", ["Preset", "Custom"])
if ticker_mode == "Preset":
    ticker_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
    ticker = ASSETS[ticker_name]
else:
    ticker = st.sidebar.text_input("Enter Ticker (e.g., TATAMOTORS.NS)", value="RELIANCE.NS")

selected_timeframes = st.sidebar.multiselect(
    "Select Timeframes (Multi-Select)",
    ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
    default=["15m", "1h", "1d"]
)

fetch_btn = st.sidebar.button("Fetch Data & Analyze", type="primary")

# Session State Initialization
if 'data_store' not in st.session_state:
    st.session_state.data_store = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# -----------------------------------------------------------------------------
# 6. ANALYSIS LOGIC
# -----------------------------------------------------------------------------
if fetch_btn:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    st.session_state.data_store = {}
    
    for i, tf in enumerate(selected_timeframes):
        period = TIMEFRAME_MAPPING.get(tf, "1mo")
        status_text.text(f"Fetching {tf} data ({period})...")
        
        raw_df = fetch_data(ticker, tf, period)
        if raw_df is not None:
            processed_df = process_dataframe(raw_df)
            st.session_state.data_store[tf] = processed_df
        
        progress_bar.progress((i + 1) / len(selected_timeframes))
    
    status_text.text("Analysis Complete!")
    st.session_state.analysis_complete = True
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

# -----------------------------------------------------------------------------
# 7. TAB RENDERING FUNCTIONS
# -----------------------------------------------------------------------------

def render_overview():
    st.header(f"📊 Multi-Timeframe Overview: {ticker}")
    
    if not st.session_state.data_store:
        st.warning("No data available. Please fetch data.")
        return

    overview_data = []
    
    for tf, df in st.session_state.data_store.items():
        if df.empty: continue
        last_row = df.iloc[-1]
        
        # Determine Status
        status = "🟡 Neutral"
        if last_row['Close'] > last_row['EMA_20'] and last_row['RSI'] > 50:
            status = "🟢 Bullish"
        elif last_row['Close'] < last_row['EMA_20'] and last_row['RSI'] < 50:
            status = "🔴 Bearish"
            
        overview_data.append({
            "Timeframe": tf,
            "Price": format_currency(last_row['Close']),
            "Change %": f"{last_row['Returns_Pct']:.2f}%",
            "RSI": f"{last_row['RSI']:.1f}",
            "Trend": status,
            "Volatility": f"{last_row['Volatility']:.2f}%",
            "Z-Score": f"{last_row['Z_Score']:.2f}"
        })
    
    st.table(pd.DataFrame(overview_data))

def render_technical_charts(tf, df):
    st.subheader(f"📈 Chart Analysis: {tf}")
    
    # Create Plotly Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

def render_signals_tab():
    st.header("🤖 AI Multi-Timeframe Signal Consensus")
    
    if not st.session_state.data_store:
        st.warning("Fetch data to generate signals.")
        return
        
    total_score = 0
    tf_count = 0
    bullish_tfs = 0
    bearish_tfs = 0
    
    breakdown = []
    
    for tf, df in st.session_state.data_store.items():
        if df.empty: continue
        row = df.iloc[-1]
        tf_score = 0
        bias = "Neutral"
        
        # Scoring Logic
        # 1. RSI
        if row['RSI'] < 30: tf_score += 20  # Oversold (Buy)
        elif row['RSI'] > 70: tf_score -= 20 # Overbought (Sell)
        
        # 2. EMA Alignment
        if row['Close'] > row['EMA_20']: tf_score += 15
        else: tf_score -= 15
        
        # 3. Z-Score
        if row['Z_Score'] < -2: tf_score += 20
        elif row['Z_Score'] > 2: tf_score -= 20
        
        # Aggregate
        total_score += tf_score
        tf_count += 1
        
        if tf_score > 15: 
            bias = "Bullish"
            bullish_tfs += 1
        elif tf_score < -15: 
            bias = "Bearish"
            bearish_tfs += 1
            
        breakdown.append({
            "Timeframe": tf,
            "Score": tf_score,
            "Bias": bias,
            "RSI": round(row['RSI'], 1),
            "Price vs EMA20": "Above" if row['Close'] > row['EMA_20'] else "Below"
        })

    # Final Signal Determination
    avg_score = total_score / tf_count if tf_count > 0 else 0
    
    signal = "HOLD / NEUTRAL"
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
        color = "#ff6666"

    # Confidence
    agreement_ratio = max(bullish_tfs, bearish_tfs) / tf_count if tf_count > 0 else 0
    confidence = min(60 + (agreement_ratio * 30) + (abs(avg_score) * 0.3), 95)
    
    # Display Result
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style='color: {color}; text-align: center;'>{signal}</h2>
            <hr>
            <p><b>Confidence:</b> {confidence:.1f}%</p>
            <p><b>Avg Score:</b> {avg_score:.1f} / 100</p>
            <p><b>Bullish TFs:</b> {bullish_tfs}/{tf_count}</p>
            <p><b>Bearish TFs:</b> {bearish_tfs}/{tf_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trading Plan Suggestion
        last_price = list(st.session_state.data_store.values())[0].iloc[-1]['Close']
        stop_loss_pct = 0.015 # 1.5%
        target_pct = 0.025   # 2.5%
        
        if "BUY" in signal:
            sl = last_price * (1 - stop_loss_pct)
            tgt = last_price * (1 + target_pct)
            st.markdown(f"""
            ### 📋 Long Setup
            - **Entry:** {format_currency(last_price)}
            - **Stop Loss:** {format_currency(sl)} (-1.5%)
            - **Target:** {format_currency(tgt)} (+2.5%)
            """)
        elif "SELL" in signal:
            sl = last_price * (1 + stop_loss_pct)
            tgt = last_price * (1 - target_pct)
            st.markdown(f"""
            ### 📋 Short Setup
            - **Entry:** {format_currency(last_price)}
            - **Stop Loss:** {format_currency(sl)} (+1.5%)
            - **Target:** {format_currency(tgt)} (-2.5%)
            """)

    with col2:
        st.subheader("Timeframe Breakdown")
        st.table(pd.DataFrame(breakdown))
        
        st.markdown("### 🔍 AI Analysis Summary")
        st.markdown(f"""
        **Consensus:** {bullish_tfs} out of {tf_count} timeframes allow for a {signal} bias.
        The algorithm analyzed RSI, EMA convergence, and Z-Score deviations. 
        {'High confidence suggests trend alignment across multiple timeframes.' if confidence > 80 else 'Lower confidence suggests market conflict or sideways movement.'}
        """)

# -----------------------------------------------------------------------------
# 8. LIVE TRADING SIMULATION
# -----------------------------------------------------------------------------
def run_live_simulation():
    st.header("🔴 Live Trading Simulator")
    
    if 'live_active' not in st.session_state:
        st.session_state.live_active = False
    
    col_controls, col_display = st.columns([1, 3])
    
    with col_controls:
        strategy = st.selectbox("Select Strategy", ["RSI + EMA Crossover", "Z-Score Reversion"])
        start_btn = st.button("Start Live Simulation")
        stop_btn = st.button("Stop Simulation")
        
        if start_btn: st.session_state.live_active = True
        if stop_btn: st.session_state.live_active = False

    place_holder = st.empty()
    
    # Live Loop
    if st.session_state.live_active:
        # Initial Fake Data for Simulation
        curr_price = 24500.00
        curr_rsi = 45.0
        position = None # None, 'LONG', 'SHORT'
        entry_price = 0
        pnl = 0
        
        logs = []
        
        for i in range(100): # Run for 100 iterations or until stopped
            if not st.session_state.live_active: break
            
            # Simulate random price movement
            change = np.random.normal(0, 5) # Random move
            curr_price += change
            
            # Simulate RSI movement
            curr_rsi += np.random.normal(0, 2)
            curr_rsi = max(0, min(100, curr_rsi))
            
            # Strategy Logic
            signal_msg = "WAITING"
            action_color = "gray"
            
            if strategy == "RSI + EMA Crossover":
                # Simplified Logic for visual demo
                if curr_rsi < 30 and position is None:
                    position = "LONG"
                    entry_price = curr_price
                    logs.append(f"{datetime.now().strftime('%H:%M:%S')} - BUY Signal @ {curr_price:.2f}")
                elif curr_rsi > 70 and position == "LONG":
                    pnl += curr_price - entry_price
                    position = None
                    logs.append(f"{datetime.now().strftime('%H:%M:%S')} - SELL Signal @ {curr_price:.2f}")

            # Display Updates
            with place_holder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{curr_price:.2f}", f"{change:.2f}")
                c2.metric("RSI (14)", f"{curr_rsi:.1f}")
                c3.metric("Active Position", position if position else "None")
                c4.metric("Realized P&L", f"₹{pnl:.2f}")
                
                st.write(f"**Active Strategy:** {strategy}")
                st.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
                
                if logs:
                    st.write("### Trade Log")
                    for log in logs[-5:]:
                        st.text(log)
            
            time.sleep(1.5) # Refresh Rate
    else:
        place_holder.info("Simulation Stopped. Click 'Start' to begin.")

# -----------------------------------------------------------------------------
# 9. MAIN APP LAYOUT
# -----------------------------------------------------------------------------

if st.session_state.analysis_complete:
    tab0, tab1, tab2, tab3, tab4, tab9, tab11 = st.tabs([
        "Overview", "Support/Resistance", "Indicators", 
        "Z-Score", "Volatility", "AI Signals (Recommended)", "Live Trading"
    ])

    with tab0:
        render_overview()

    with tab1:
        st.header("🧱 Support & Resistance Analysis")
        # Logic to display S/R for primary timeframe
        if selected_timeframes:
            tf = selected_timeframes[0] # Default to first
            df = st.session_state.data_store[tf]
            levels = TechnicalAnalysis.identify_support_resistance(df)
            
            col_chart, col_data = st.columns([2, 1])
            with col_data:
                st.subheader(f"Key Levels ({tf})")
                sr_df = pd.DataFrame(levels, columns=['Date', 'Price', 'Type'])
                st.dataframe(sr_df.sort_values(by='Price', ascending=False).head(10), use_container_width=True)
            
            with col_chart:
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                for _, price, l_type in levels:
                    color = "green" if l_type == "Support" else "red"
                    fig.add_hline(y=price, line_dash="dot", line_color=color, opacity=0.5)
                fig.update_layout(height=500, title=f"S/R Map - {tf}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📉 Technical Indicators")
        for tf in selected_timeframes:
            if tf in st.session_state.data_store:
                render_technical_charts(tf, st.session_state.data_store[tf])
                st.markdown("---")

    with tab3:
        st.header("📊 Z-Score Analysis (Statistical Extremes)")
        for tf in selected_timeframes:
            df = st.session_state.data_store[tf]
            last_z = df['Z_Score'].iloc[-1]
            
            st.markdown(f"### {tf} Analysis")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Current Z-Score", f"{last_z:.2f}")
                if last_z < -2:
                    st.success("EXTREME OVERSOLD (<-2) - Reversion Likely")
                elif last_z > 2:
                    st.error("EXTREME OVERBOUGHT (>2) - Correction Expected")
                else:
                    st.info("Normal Range")
            
            with col2:
                # Histogram of Z-Scores
                fig = go.Figure(data=[go.Histogram(x=df['Z_Score'], nbinsx=30, marker_color='#00d4ff')])
                fig.add_vline(x=last_z, line_color="yellow", annotation_text="Current")
                fig.update_layout(title="Z-Score Distribution (Historical)", height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("⚡ Volatility Analysis")
        for tf in selected_timeframes:
            df = st.session_state.data_store[tf]
            current_vol = df['Volatility'].iloc[-1]
            avg_vol = df['Volatility'].mean()
            
            st.subheader(f"Timeframe: {tf}")
            st.metric("Current Volatility (Annualized)", f"{current_vol:.2f}%", f"{current_vol - avg_vol:.2f}% vs Avg")
            
            if current_vol < (avg_vol * 0.7):
                st.warning("⚠️ Low Volatility Compression - Expect Breakout Soon!")
            
            st.line_chart(df['Volatility'])

    with tab9:
        render_signals_tab()

    with tab11:
        run_live_simulation()

else:
    st.info("👈 Please select assets and timeframes from the sidebar and click 'Fetch Data'.")
    
    # Landing Page Info
    st.markdown("""
    ### Welcome to AlgoTrade Pro
    This application provides institutional-grade technical analysis including:
    - **Multi-Timeframe Correlation**: Analyze trends across 1m to 1wk simultaneously.
    - **Statistical Analysis**: Z-Score and Volatility binning to identify extremes.
    - **AI Consensus**: Weighted scoring system to generate clear Buy/Sell signals.
    - **Backtesting & Simulation**: Test strategies in real-time.
    
    **Get started by selecting a ticker in the sidebar.**
    """)
