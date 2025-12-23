import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
import pytz
from datetime import datetime, timedelta
import time
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro AlgoTrader Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS & CONFIG ---
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

VALID_PERIODS = {
    '1m': '5d', '2m': '5d', '5m': '1mo', '15m': '1mo', '30m': '1mo',
    '60m': '3mo', '90m': '3mo', '1h': '3mo', 
    '1d': '1y', '5d': '2y', '1wk': '2y', '1mo': '5y', '3mo': 'max'
}

# --- HELPER FUNCTIONS ---

def get_ist_time():
    return datetime.now(IST)

def format_time_ago(dt_val):
    """Converts datetime to human-readable 'time ago' string."""
    if pd.isna(dt_val): return "N/A"
    
    # Ensure dt_val is timezone-aware/converted to IST for comparison
    now = get_ist_time()
    
    if isinstance(dt_val, str):
        try:
            dt_val = pd.to_datetime(dt_val).replace(tzinfo=IST)
        except:
            return str(dt_val)
            
    if dt_val.tzinfo is None:
        dt_val = IST.localize(dt_val)
    else:
        dt_val = dt_val.astimezone(IST)
        
    diff = now - dt_val
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    elif seconds < 2592000: # 30 days
        return f"{int(seconds // 86400)} days ago"
    else:
        return f"{int(seconds // 86400)} days ago ({dt_val.strftime('%Y-%m-%d %H:%M:%S IST')})"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_volatility(series, window=20):
    return series.pct_change().rolling(window=window).std() * np.sqrt(252) * 100

def identify_support_resistance(df, window=20):
    """Identify local mins and maxs."""
    df['Support'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    df['Resistance'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    return df

@st.cache_data(ttl=300)
def fetch_ticker_data(ticker, timeframe, period):
    """Robust data fetching with rate limiting."""
    try:
        time.sleep(0.5) # Rate limit
        df = yf.download(ticker, period=period, interval=timeframe, progress=False)
        
        if df.empty:
            return None
            
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Standardize columns
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'DateTime'}, inplace=True)
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'DateTime'}, inplace=True)
            
        # Timezone Handling
        if df['DateTime'].dt.tz is None:
            df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert(IST)
        else:
            df['DateTime'] = df['DateTime'].dt.tz_convert(IST)
            
        df['DateTime_IST'] = df['DateTime'] # Explicit column name as requested
        
        # Basic indicators
        df['Return_%'] = df['Close'].pct_change() * 100
        df['RSI'] = calculate_rsi(df['Close'])
        df['EMA_20'] = calculate_ema(df['Close'], 20)
        df['EMA_50'] = calculate_ema(df['Close'], 50)
        df['EMA_200'] = calculate_ema(df['Close'], 200)
        
        # Z-Score
        df['Z_Score'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).std()
        
        # Volatility
        df['Vol_%'] = calculate_volatility(df['Close'])
        
        # Drop initial NaNs
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching {timeframe}: {e}")
        return None

# --- SIDEBAR & INPUTS ---
st.sidebar.title("🛠️ Configuration")

ticker_mode = st.sidebar.radio("Asset Selection", ["Preset", "Custom Ticker"])
if ticker_mode == "Preset":
    asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
    ticker1 = ASSETS[asset_name]
else:
    ticker1 = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

# Multi-select timeframes
all_tfs = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '1wk', '1mo']
selected_tfs = st.sidebar.multiselect("Select Timeframes", all_tfs, default=['5m', '15m', '1h', '1d'])

enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
ticker2 = None
if enable_ratio:
    ticker2 = st.sidebar.text_input("Comparison Ticker", "^NSEBANK")

fetch_btn = st.sidebar.button("🚀 Analyze Market", type="primary")

# --- MAIN ANALYSIS LOGIC ---

if fetch_btn:
    st.session_state['data'] = {}
    st.session_state['analysis_complete'] = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data_store = {}
    
    for i, tf in enumerate(selected_tfs):
        # Determine best period for timeframe
        period = VALID_PERIODS.get(tf, '1mo')
        status_text.text(f"Fetching {tf} data ({period})...")
        
        df = fetch_ticker_data(ticker1, tf, period)
        if df is not None:
            data_store[tf] = df
            
        progress_bar.progress((i + 1) / len(selected_tfs))
        
    st.session_state['data'] = data_store
    st.session_state['analysis_complete'] = True
    st.session_state['ticker'] = ticker1
    status_text.empty()
    progress_bar.empty()

# --- DISPLAY LOGIC ---

if st.session_state.get('analysis_complete'):
    data_dict = st.session_state['data']
    ticker = st.session_state['ticker']
    
    if not data_dict:
        st.error("No data fetched. Please check ticker or try different timeframes.")
        st.stop()

    # --- HEADER ---
    last_price = list(data_dict.values())[-1]['Close'].iloc[-1]
    st.title(f"📊 Algorithmic Trading Analysis: {ticker}")
    st.markdown(f"**Current Price:** ₹{last_price:,.2f} | **Analysis Time:** {get_ist_time().strftime('%Y-%m-%d %H:%M IST')}")
    st.markdown("---")

    # --- TABS ---
    tabs = st.tabs([
        "Overview", "Support/Resistance", "Technical Indicators", "Z-Score Analysis", 
        "Volatility", "Elliott Waves", "Fibonacci", "RSI Divergence", "Ratio Analysis", 
        "🤖 AI Signals", "Backtesting"
    ])

    # ==============================================================================
    # TAB 0: MULTI-TIMEFRAME OVERVIEW
    # ==============================================================================
    with tabs[0]:
        st.subheader("Multi-Timeframe Market Status")
        
        overview_data = []
        for tf, df in data_dict.items():
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Trend Status
            ema_20 = last_row['EMA_20']
            ema_50 = last_row['EMA_50']
            trend = "🟢 Bullish" if last_row['Close'] > ema_20 else "🔴 Bearish"
            if (last_row['Close'] > ema_20 and last_row['Close'] < ema_50) or (last_row['Close'] < ema_20 and last_row['Close'] > ema_50):
                trend = "🟡 Neutral"
            
            overview_data.append({
                "Timeframe": tf,
                "Status": trend,
                "Price": f"₹{last_row['Close']:,.2f}",
                "Change %": f"{last_row['Return_%']:.2f}%",
                "RSI": f"{last_row['RSI']:.1f}",
                "Volatility": f"{last_row['Vol_%']:.2f}%",
                "Distance to 20EMA": f"{(last_row['Close'] - ema_20):.2f}"
            })
            
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

    # ==============================================================================
    # TAB 1: SUPPORT / RESISTANCE
    # ==============================================================================
    with tabs[1]:
        sr_consensus = {"Support": 0, "Resistance": 0}
        
        for tf, df in data_dict.items():
            st.markdown(f"## 📊 Support & Resistance: {tf}")
            
            # Find levels
            window = 5 # shorter window for local levels
            df['Min'] = df['Low'][(df['Low'].shift(window) > df['Low']) & (df['Low'].shift(-window) > df['Low'])]
            df['Max'] = df['High'][(df['High'].shift(window) < df['High']) & (df['High'].shift(-window) < df['High'])]
            
            supports = df['Min'].dropna().tolist()
            resistances = df['Max'].dropna().tolist()
            
            # Filter nearby levels (within 2%)
            current_price = df['Close'].iloc[-1]
            nearby_s = [s for s in supports if s < current_price and s > current_price * 0.9]
            nearby_r = [r for r in resistances if r > current_price and r < current_price * 1.1]
            
            # Sort and take closest
            nearby_s = sorted(nearby_s)[-3:] if nearby_s else []
            nearby_r = sorted(nearby_r)[:3] if nearby_r else []
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Nearest Supports:**")
                if nearby_s:
                    for s in nearby_s:
                        dist = current_price - s
                        dist_pct = (dist / current_price) * 100
                        st.markdown(f"- ₹{s:,.2f} (Dist: {dist:,.2f} pts / {dist_pct:.2f}%)")
                        if dist_pct < 0.5: sr_consensus["Support"] += 1
                else:
                    st.write("No nearby support found.")
                    
            with col2:
                st.markdown("**Nearest Resistances:**")
                if nearby_r:
                    for r in nearby_r:
                        dist = r - current_price
                        dist_pct = (dist / current_price) * 100
                        st.markdown(f"- ₹{r:,.2f} (Dist: {dist:,.2f} pts / {dist_pct:.2f}%)")
                        if dist_pct < 0.5: sr_consensus["Resistance"] += 1
                else:
                    st.write("No nearby resistance found.")
            
            # S/R Forecast
            if nearby_s and (current_price - nearby_s[-1]) / current_price < 0.005:
                st.info(f"💡 **Forecast ({tf}):** Price is testing Support at {nearby_s[-1]:.2f}. Expect a BOUNCE if level holds.")
            elif nearby_r and (nearby_r[0] - current_price) / current_price < 0.005:
                st.warning(f"💡 **Forecast ({tf}):** Price is testing Resistance at {nearby_r[0]:.2f}. Expect REJECTION.")
            else:
                st.write(f"💡 **Forecast ({tf}):** Price is in middle of range.")

            # Chart
            fig = go.Figure(data=[go.Candlestick(x=df['DateTime_IST'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            for s in nearby_s: fig.add_hline(y=s, line_color="green", line_dash="dash")
            for r in nearby_r: fig.add_hline(y=r, line_color="red", line_dash="dash")
            fig.update_layout(height=400, title=f"{tf} Price with S/R", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

        # Consensus
        st.subheader("🏆 S/R Consensus")
        st.write(f"Timeframes near Support: {sr_consensus['Support']}")
        st.write(f"Timeframes near Resistance: {sr_consensus['Resistance']}")
        if sr_consensus['Support'] > sr_consensus['Resistance']:
            st.success("Overall Bias: Potential Bounce from Support Zone")
        elif sr_consensus['Resistance'] > sr_consensus['Support']:
            st.error("Overall Bias: Resistance Pressure Detected")
        else:
            st.info("Overall Bias: Range Bound / No immediate S/R conflict")

    # ==============================================================================
    # TAB 2: TECHNICAL INDICATORS
    # ==============================================================================
    with tabs[2]:
        st.subheader("Multi-Timeframe EMA & RSI Analysis")
        
        ema_summary = []
        for tf, df in data_dict.items():
            last = df.iloc[-1]
            ema_summary.append({
                "Timeframe": tf,
                "Price": last['Close'],
                "EMA 20": last['EMA_20'],
                "EMA 50": last['EMA_50'],
                "Trend": "Bullish" if last['Close'] > last['EMA_20'] else "Bearish",
                "RSI": last['RSI'],
                "RSI Status": "Overbought" if last['RSI'] > 70 else ("Oversold" if last['RSI'] < 30 else "Neutral")
            })
            
        st.dataframe(pd.DataFrame(ema_summary).style.applymap(
            lambda x: 'color: green' if x == 'Bullish' or x == 'Oversold' else ('color: red' if x == 'Bearish' or x == 'Overbought' else ''),
            subset=['Trend', 'RSI Status']
        ), use_container_width=True)
        
        # Detailed Tables for Primary Timeframe (first selected)
        primary_tf = selected_tfs[0]
        st.markdown(f"### Detailed Data: {primary_tf}")
        st.dataframe(data_dict[primary_tf][['DateTime_IST', 'Close', 'EMA_20', 'EMA_50', 'RSI']].tail(20), use_container_width=True)

    # ==============================================================================
    # TAB 3: Z-SCORE ANALYSIS
    # ==============================================================================
    with tabs[3]:
        z_consensus = []
        
        for tf, df in data_dict.items():
            st.markdown(f"## 📊 Z-Score Analysis: {tf}")
            
            current_z = df['Z_Score'].iloc[-1]
            
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Z-Score", f"{current_z:.2f}")
            col2.metric("Mean Z-Score", f"{df['Z_Score'].mean():.2f}")
            col3.metric("Std Dev", f"{df['Z_Score'].std():.2f}")
            
            # Binning
            bins = [-np.inf, -2, -1, 1, 2, np.inf]
            labels = ['Extreme Negative (<-2)', 'Negative (-2 to -1)', 'Neutral (-1 to 1)', 'Positive (1 to 2)', 'Extreme Positive (>2)']
            df['Z_Bin'] = pd.cut(df['Z_Score'], bins=bins, labels=labels)
            
            bin_stats = df.groupby('Z_Bin', observed=False).agg(
                Count=('Close', 'count'),
                Min_Price=('Close', 'min'),
                Max_Price=('Close', 'max'),
                Avg_Return=('Return_%', 'mean')
            ).reset_index()
            
            bin_stats['%'] = (bin_stats['Count'] / len(df) * 100).round(2)
            bin_stats['Price Range'] = bin_stats.apply(lambda x: f"₹{x['Min_Price']:,.0f} - ₹{x['Max_Price']:,.0f}", axis=1)
            
            st.table(bin_stats[['Z_Bin', 'Count', '%', 'Price Range', 'Avg_Return']])
            
            # Historical Similarity & Forecast
            if current_z < -2:
                msg = "EXTREME OVERSOLD"
                bias = "Rally Expected"
                z_consensus.append(1) # Bullish
                st.success(f"**Condition: {msg}**")
                
                # Find past instances
                past_events = df[df['Z_Score'] < -2]
                st.write(f"**Historical Context:** Found {len(past_events)} similar events.")
                if not past_events.empty:
                    avg_rally = past_events['Return_%'].mean()
                    st.write(f"Average return during these periods: {avg_rally:.2f}%")
                    st.write(f"**Forecast:** High probability of mean reversion (Bounce). Target: EMA 20.")
            
            elif current_z > 2:
                msg = "EXTREME OVERBOUGHT"
                bias = "Correction Expected"
                z_consensus.append(-1) # Bearish
                st.error(f"**Condition: {msg}**")
                
                past_events = df[df['Z_Score'] > 2]
                st.write(f"**Historical Context:** Found {len(past_events)} similar events.")
                if not past_events.empty:
                    st.write(f"**Forecast:** Expect profit booking or pullback to mean.")
            else:
                st.info("Condition: NORMAL RANGE. No statistical extreme.")
                z_consensus.append(0)
            
            st.markdown("---")
            
        # Consensus
        st.subheader("🏆 Z-Score Consensus")
        bulls = z_consensus.count(1)
        bears = z_consensus.count(-1)
        st.write(f"Oversold Timeframes (Buy Signal): {bulls}")
        st.write(f"Overbought Timeframes (Sell Signal): {bears}")

    # ==============================================================================
    # TAB 4: VOLATILITY ANALYSIS
    # ==============================================================================
    with tabs[4]:
        for tf, df in data_dict.items():
            st.markdown(f"## 📊 Volatility: {tf}")
            
            curr_vol = df['Vol_%'].iloc[-1]
            avg_vol = df['Vol_%'].mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Current Volatility (Annualized)", f"{curr_vol:.2f}%")
            col2.metric("Average Volatility", f"{avg_vol:.2f}%")
            
            # Historical Context
            if curr_vol < avg_vol * 0.7:
                st.warning("⚠️ **Low Volatility Compression:** Expect a breakout soon. Large move imminent.")
            elif curr_vol > avg_vol * 1.5:
                st.info("⚠️ **High Volatility:** Expect large swings. Reduce position size.")
            else:
                st.write("Volatility is within normal range.")
                
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['DateTime_IST'], y=df['Vol_%'], name='Volatility'))
            fig.add_hline(y=avg_vol, line_dash="dash", annotation_text="Avg Vol")
            fig.update_layout(height=300, title="Volatility Regime")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

    # ==============================================================================
    # TAB 5: ELLIOTT WAVES (Simplified)
    # ==============================================================================
    with tabs[5]:
        st.markdown("## 🌊 Elliott Wave Detection (Primary Timeframe)")
        df_ew = data_dict[selected_tfs[0]].copy()
        
        # Simple Zig-Zag logic for wave visualization
        # Note: True Elliott Wave is subjective; this detects major pivots
        df_ew['Pivot'] = df_ew['Close'].rolling(5, center=True).apply(lambda x: 1 if x[2] == max(x) else (-1 if x[2] == min(x) else 0), raw=True)
        pivots = df_ew[df_ew['Pivot'] != 0]
        
        if not pivots.empty:
            st.write(f"Detected {len(pivots)} major pivots in {selected_tfs[0]}")
            
            last_pivot = pivots.iloc[-1]
            prev_pivot = pivots.iloc[-2]
            
            direction = "Impulse Up" if last_pivot['Close'] > prev_pivot['Close'] else "Correction/Impulse Down"
            st.write(f"**Current Wave Context:** Price moving from ₹{prev_pivot['Close']:.2f} to ₹{last_pivot['Close']:.2f}")
            st.info(f"Most recent identifiable move: **{direction}**")
            
            # Table of recent waves
            waves = []
            for i in range(len(pivots)-1, max(0, len(pivots)-6), -1):
                p1 = pivots.iloc[i-1]
                p2 = pivots.iloc[i]
                waves.append({
                    "Start Date": p1['DateTime_IST'],
                    "End Date": p2['DateTime_IST'],
                    "Start Price": p1['Close'],
                    "End Price": p2['Close'],
                    "Move %": ((p2['Close'] - p1['Close'])/p1['Close']*100)
                })
            
            st.table(pd.DataFrame(waves))
            
            # Chart
            fig = go.Figure(data=[go.Candlestick(x=df_ew['DateTime_IST'], open=df_ew['Open'], high=df_ew['High'], low=df_ew['Low'], close=df_ew['Close'])])
            fig.add_trace(go.Scatter(x=pivots['DateTime_IST'], y=pivots['Close'], mode='lines+markers', name='Wave Structure', line=dict(color='blue', width=2)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to detect clear waves.")

    # ==============================================================================
    # TAB 6: FIBONACCI
    # ==============================================================================
    with tabs[6]:
        df_fib = data_dict[selected_tfs[-1]] # Use longest timeframe for major levels
        st.markdown(f"## 🔢 Fibonacci Levels ({selected_tfs[-1]})")
        
        high_price = df_fib['High'].max()
        low_price = df_fib['Low'].min()
        curr_price = df_fib['Close'].iloc[-1]
        
        diff = high_price - low_price
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_data = []
        
        for level in levels:
            price_level = high_price - (diff * level)
            dist = curr_price - price_level
            status = "Resistance" if dist < 0 else "Support"
            fib_data.append({
                "Level": f"{level*100}%",
                "Price": price_level,
                "Distance": abs(dist),
                "Type": status
            })
            
        fib_df = pd.DataFrame(fib_data)
        st.table(fib_df)
        
        # Highlight nearest
        nearest = fib_df.loc[fib_df['Distance'].idxmin()]
        st.info(f"Price is currently closest to the **{nearest['Level']}** level at **₹{nearest['Price']:,.2f}** acting as **{nearest['Type']}**.")

    # ==============================================================================
    # TAB 7: RSI DIVERGENCE
    # ==============================================================================
    with tabs[7]:
        st.markdown("## 📉 RSI Divergence Detector")
        
        for tf, df in data_dict.items():
            # Simple divergence check on last 30 candles
            subset = df.tail(30).reset_index(drop=True)
            
            # Find peaks
            price_peaks = subset['Close'][(subset['Close'].shift(1) < subset['Close']) & (subset['Close'].shift(-1) < subset['Close'])]
            rsi_peaks = subset['RSI'][(subset['RSI'].shift(1) < subset['RSI']) & (subset['RSI'].shift(-1) < subset['RSI'])]
            
            div_found = False
            
            # Bearish Divergence Logic (Price Higher High, RSI Lower High)
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                if price_peaks.iloc[-1] > price_peaks.iloc[-2] and rsi_peaks.iloc[-1] < rsi_peaks.iloc[-2]:
                    st.error(f"**{tf}:** Bearish Divergence Detected! Price made Higher High, RSI made Lower High.")
                    div_found = True
            
            # Bullish Divergence Logic (Price Lower Low, RSI Higher Low)
            price_lows = subset['Close'][(subset['Close'].shift(1) > subset['Close']) & (subset['Close'].shift(-1) > subset['Close'])]
            rsi_lows = subset['RSI'][(subset['RSI'].shift(1) > subset['RSI']) & (subset['RSI'].shift(-1) > subset['RSI'])]
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                    st.success(f"**{tf}:** Bullish Divergence Detected! Price made Lower Low, RSI made Higher Low.")
                    div_found = True
            
            if not div_found:
                st.write(f"{tf}: No clear divergence in recent data.")

    # ==============================================================================
    # TAB 8: RATIO ANALYSIS
    # ==============================================================================
    with tabs[8]:
        if not enable_ratio:
            st.warning("Ratio analysis not enabled in sidebar.")
        else:
            st.markdown(f"## ⚖️ Ratio Analysis: {ticker1} / {ticker2}")
            # Fetch Ticker 2
            df1 = data_dict[selected_tfs[0]]
            df2 = fetch_ticker_data(ticker2, selected_tfs[0], VALID_PERIODS[selected_tfs[0]])
            
            if df2 is not None:
                # Merge
                merged = pd.merge(df1[['DateTime_IST', 'Close']], df2[['DateTime_IST', 'Close']], on='DateTime_IST', suffixes=('_1', '_2'))
                merged['Ratio'] = merged['Close_1'] / merged['Close_2']
                merged['Ratio_Mean'] = merged['Ratio'].rolling(50).mean()
                merged['Z_Score'] = zscore(merged['Ratio'].dropna())
                
                curr_ratio = merged['Ratio'].iloc[-1]
                curr_z = merged['Z_Score'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Ratio", f"{curr_ratio:.4f}")
                col2.metric("Ratio Z-Score", f"{curr_z:.2f}")
                
                if curr_z > 1.5:
                    col3.warning(f"{ticker1} Expensive vs {ticker2}")
                elif curr_z < -1.5:
                    col3.success(f"{ticker1} Cheap vs {ticker2}")
                else:
                    col3.info("Fair Value")
                
                # Plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=merged['DateTime_IST'], y=merged['Ratio'], name='Ratio'), row=1, col=1)
                fig.add_hline(y=merged['Ratio'].mean(), line_dash="dash", row=1, col=1)
                fig.add_trace(go.Scatter(x=merged['DateTime_IST'], y=merged['Z_Score'], name='Z-Score'), row=2, col=1)
                fig.add_hline(y=2, line_color='red', line_dash='dot', row=2, col=1)
                fig.add_hline(y=-2, line_color='green', line_dash='dot', row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

    # ==============================================================================
    # TAB 9: AI SIGNALS (THE HOLY GRAIL)
    # ==============================================================================
    with tabs[9]:
        st.markdown("# 🧠 AI Multi-Timeframe Signal Engine")
        
        # --- SCORING LOGIC ---
        total_score = 0
        score_details = []
        tf_count = len(data_dict)
        
        for tf, df in data_dict.items():
            tf_score = 0
            row = df.iloc[-1]
            reasons = []
            
            # RSI Scoring
            if row['RSI'] < 30: 
                tf_score += 20
                reasons.append("RSI Oversold (+20)")
            elif row['RSI'] > 70: 
                tf_score -= 20
                reasons.append("RSI Overbought (-20)")
            
            # EMA Trend
            if row['Close'] > row['EMA_20']:
                tf_score += 15
                reasons.append("Above EMA20 (+15)")
            else:
                tf_score -= 15
                reasons.append("Below EMA20 (-15)")
                
            # Z-Score
            if row['Z_Score'] < -2:
                tf_score += 20
                reasons.append("Z-Score Extreme Low (+20)")
            elif row['Z_Score'] > 2:
                tf_score -= 20
                reasons.append("Z-Score Extreme High (-20)")
                
            score_details.append({
                "Timeframe": tf,
                "Score": tf_score,
                "Bias": "🟢 Bullish" if tf_score > 0 else ("🔴 Bearish" if tf_score < 0 else "Neutral"),
                "Key Factors": ", ".join(reasons)
            })
            total_score += tf_score

        # Normalize score
        avg_score = total_score / tf_count
        
        # --- SIGNAL DETERMINATION ---
        signal = "NEUTRAL"
        color = "gray"
        action_text = "HOLD"
        
        if avg_score > 25:
            signal = "STRONG BUY"
            color = "green"
            action_text = "LONG"
        elif avg_score > 10:
            signal = "BUY"
            color = "lightgreen"
            action_text = "LONG"
        elif avg_score < -25:
            signal = "STRONG SELL"
            color = "red"
            action_text = "SHORT"
        elif avg_score < -10:
            signal = "SELL"
            color = "orange"
            action_text = "SHORT"
            
        # Confidence Calculation
        agreeing_tfs = sum(1 for s in score_details if (s['Score'] > 0 and avg_score > 0) or (s['Score'] < 0 and avg_score < 0))
        confidence = 60 + ((agreeing_tfs / tf_count) * 30) + (min(abs(avg_score), 50) * 0.2)
        confidence = min(confidence, 95.0)

        # --- TRADING PLAN ---
        curr_price = last_price
        
        # Define Targets based on Asset Type
        is_index = "NIFTY" in ticker or "SENSEX" in ticker
        sl_pct = 0.015 if is_index else 0.025
        tp_pct = 0.020 if is_index else 0.040
        
        if "BUY" in signal:
            sl_price = curr_price * (1 - sl_pct)
            tp_price = curr_price * (1 + tp_pct)
            sl_pts = curr_price - sl_price
            tp_pts = tp_price - curr_price
        else:
            sl_price = curr_price * (1 + sl_pct)
            tp_price = curr_price * (1 - tp_pct)
            sl_pts = sl_price - curr_price
            tp_pts = curr_price - tp_price

        # --- DISPLAY OUTPUT ---
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {color}'>
            <h1 style='color: {color}; margin:0;'>{signal}</h1>
            <h3>Confidence: {confidence:.1f}% | Multi-Timeframe Score: {avg_score:.1f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Recommended Trading Plan")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry Price", f"₹{curr_price:,.2f}")
        c2.metric("Stop Loss", f"₹{sl_price:,.2f}", f"-{sl_pts:.2f} pts")
        c3.metric("Target", f"₹{tp_price:,.2f}", f"+{tp_pts:.2f} pts")
        c4.metric("Risk:Reward", f"1 : {tp_pts/sl_pts:.2f}")
        
        st.markdown("### 🔍 Rationale")
        st.write(f"**Consensus:** {agreeing_tfs} out of {tf_count} timeframes agree with this direction.")
        st.write("**Detailed Breakdown:**")
        st.dataframe(pd.DataFrame(score_details), use_container_width=True)

    # ==============================================================================
    # TAB 10: BACKTESTING (Simplified Vectorized)
    # ==============================================================================
    with tabs[10]:
        st.header("⚡ Instant Strategy Backtest")
        st.write("Simulating trading strategies on the Primary Timeframe (Historical Data).")
        
        bt_df = data_dict[selected_tfs[0]].copy()
        
        strategy = st.selectbox("Select Strategy", ["RSI Reversal (30/70)", "EMA Crossover (20/50)", "Bollinger Breakout"])
        
        # Logic
        bt_df['Signal'] = 0
        
        if strategy == "RSI Reversal (30/70)":
            bt_df.loc[bt_df['RSI'] < 30, 'Signal'] = 1  # Buy
            bt_df.loc[bt_df['RSI'] > 70, 'Signal'] = -1 # Sell
            
        elif strategy == "EMA Crossover (20/50)":
            bt_df['Signal'] = np.where(bt_df['EMA_20'] > bt_df['EMA_50'], 1, -1)
            # Detect changes only
            bt_df['Signal'] = bt_df['Signal'].diff().fillna(0)
            bt_df.loc[bt_df['Signal'] > 0, 'Signal'] = 1
            bt_df.loc[bt_df['Signal'] < 0, 'Signal'] = -1
            
        elif strategy == "Bollinger Breakout":
            sma = bt_df['Close'].rolling(20).mean()
            std = bt_df['Close'].rolling(20).std()
            upper = sma + (2 * std)
            lower = sma - (2 * std)
            bt_df.loc[bt_df['Close'] < lower, 'Signal'] = 1
            bt_df.loc[bt_df['Close'] > upper, 'Signal'] = -1

        # Simulate Trades
        trades = []
        position = 0 # 0 none, 1 long, -1 short
        entry_price = 0
        
        for i, row in bt_df.iterrows():
            if row['Signal'] == 1 and position == 0:
                position = 1
                entry_price = row['Close']
                trades.append({'Type': 'Buy', 'Price': entry_price, 'Date': row['DateTime_IST']})
            elif row['Signal'] == -1 and position == 1:
                position = 0
                exit_price = row['Close']
                pnl = (exit_price - entry_price) / entry_price * 100
                trades[-1]['Exit Price'] = exit_price
                trades[-1]['Exit Date'] = row['DateTime_IST']
                trades[-1]['PnL%'] = pnl
        
        # Results
        if trades:
            trade_df = pd.DataFrame(trades).dropna()
            if not trade_df.empty:
                total_trades = len(trade_df)
                win_rate = len(trade_df[trade_df['PnL%'] > 0]) / total_trades * 100
                avg_pnl = trade_df['PnL%'].mean()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Trades", total_trades)
                m2.metric("Win Rate", f"{win_rate:.1f}%")
                m3.metric("Avg PnL per Trade", f"{avg_pnl:.2f}%")
                
                st.write("Recent Trades:")
                st.dataframe(trade_df.tail(10))
                
                # Equity Curve
                trade_df['Cum_PnL'] = trade_df['PnL%'].cumsum()
                st.line_chart(trade_df['Cum_PnL'])
            else:
                st.warning("Strategy generated signals but no closed trades.")
        else:
            st.warning("No trades triggered by this strategy in the selected period.")

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational and analytical purposes only. Algorithmic trading involves significant risk. Validate all signals manually.")
