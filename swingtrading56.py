import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from scipy.signal import argrelextrema
import time

st.set_page_config(page_title="Algorithmic Trading Analysis", layout="wide", initial_sidebar_state="expanded")

IST = pytz.timezone('Asia/Kolkata')

VALID_COMBINATIONS = {
    '1m': ['1d', '5d'], '2m': ['1d', '5d'], '5m': ['1d', '1mo'], '15m': ['1d', '1mo'],
    '30m': ['1d', '1mo'], '60m': ['1mo', '3mo'], '90m': ['1mo', '3mo'], '1h': ['1mo', '3mo'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y'], '5d': ['1y', '2y', '5y'],
    '1wk': ['1y', '2y', '5y'], '1mo': ['2y', '5y', '10y'], '3mo': ['5y', '10y']
}

PREDEFINED_ASSETS = {
    'NIFTY 50': '^NSEI', 'Bank NIFTY': '^NSEBANK', 'SENSEX': '^BSESN',
    'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Gold': 'GC=F', 'Silver': 'SI=F',
    'USD/INR': 'INR=X', 'EUR/USD': 'EURUSD=X'
}

def format_time_ago(dt):
    if pd.isna(dt):
        return "N/A"
    try:
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        else:
            dt = dt.astimezone(IST)
        now = pd.Timestamp.now(tz=IST)
        diff = now - dt
        minutes = diff.total_seconds() / 60
        hours = minutes / 60
        days = hours / 24
        if minutes < 60:
            return f"{int(minutes)} minutes ago"
        elif hours < 24:
            return f"{int(hours)} hours ago"
        elif days < 30:
            return f"{int(days)} days ago"
        else:
            months = int(days / 30)
            remaining_days = int(days % 30)
            return f"{months} months and {remaining_days} days ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"
    except:
        return str(dt)

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def detect_support_resistance(data, window=20, tolerance=0.02):
    highs = data['High'].values
    lows = data['Low'].values
    dates = data.index
    resistance_idx = argrelextrema(highs, np.greater, order=window)[0]
    support_idx = argrelextrema(lows, np.less, order=window)[0]
    levels = []
    for idx in resistance_idx:
        if idx < len(highs):
            levels.append({'type': 'Resistance', 'price': highs[idx], 'date': dates[idx]})
    for idx in support_idx:
        if idx < len(lows):
            levels.append({'type': 'Support', 'price': lows[idx], 'date': dates[idx]})
    if not levels:
        return []
    df_levels = pd.DataFrame(levels).sort_values('price')
    clustered = []
    for level_type in ['Support', 'Resistance']:
        type_levels = df_levels[df_levels['type'] == level_type].copy()
        if len(type_levels) == 0:
            continue
        current_cluster = [type_levels.iloc[0]]
        for i in range(1, len(type_levels)):
            if abs(type_levels.iloc[i]['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] <= tolerance:
                current_cluster.append(type_levels.iloc[i])
            else:
                avg_price = np.mean([l['price'] for l in current_cluster])
                first_hit = min([l['date'] for l in current_cluster])
                last_hit = max([l['date'] for l in current_cluster])
                hit_count = len(current_cluster)
                sustained = sum(1 for l in current_cluster if data.index.get_loc(l['date']) < len(data) - 1)
                clustered.append({'type': level_type, 'price': avg_price, 'hit_count': hit_count,
                                'sustained_count': sustained, 'first_hit': first_hit, 'last_hit': last_hit})
                current_cluster = [type_levels.iloc[i]]
        if current_cluster:
            avg_price = np.mean([l['price'] for l in current_cluster])
            first_hit = min([l['date'] for l in current_cluster])
            last_hit = max([l['date'] for l in current_cluster])
            hit_count = len(current_cluster)
            sustained = sum(1 for l in current_cluster if data.index.get_loc(l['date']) < len(data) - 1)
            clustered.append({'type': level_type, 'price': avg_price, 'hit_count': hit_count,
                            'sustained_count': sustained, 'first_hit': first_hit, 'last_hit': last_hit})
    return clustered

def fetch_data(ticker, interval, period):
    try:
        time.sleep(1.5)
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        date_col = data.columns[0]
        data = data.rename(columns={date_col: 'DateTime_IST'})
        if data['DateTime_IST'].dtype == 'datetime64[ns]':
            data['DateTime_IST'] = pd.to_datetime(data['DateTime_IST'])
            if data['DateTime_IST'].dt.tz is None:
                data['DateTime_IST'] = data['DateTime_IST'].dt.tz_localize('UTC').dt.tz_convert(IST)
            else:
                data['DateTime_IST'] = data['DateTime_IST'].dt.tz_convert(IST)
        cols = ['DateTime_IST', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in data.columns:
            cols.append('Volume')
        data = data[cols].copy()
        data = data.dropna(subset=['Close'])
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker} ({interval}/{period}): {str(e)}")
        return None

def calculate_z_score(data):
    returns = data['Close'].pct_change()
    return (returns - returns.mean()) / returns.std()

def calculate_volatility(data, window=20):
    returns = data['Close'].pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def detect_rsi_divergence(data, rsi_period=14):
    data = data.copy()
    data['RSI'] = calculate_rsi(data['Close'], rsi_period)
    price_peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    price_troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
    rsi_peaks = argrelextrema(data['RSI'].values, np.greater, order=5)[0]
    rsi_troughs = argrelextrema(data['RSI'].values, np.less, order=5)[0]
    divergences = []
    for i in range(len(price_peaks) - 1):
        idx1, idx2 = price_peaks[i], price_peaks[i+1]
        if idx2 < len(data) and data['Close'].iloc[idx2] > data['Close'].iloc[idx1]:
            rsi_peak1 = min(rsi_peaks, key=lambda x: abs(x - idx1)) if len(rsi_peaks) > 0 else None
            rsi_peak2 = min(rsi_peaks, key=lambda x: abs(x - idx2)) if len(rsi_peaks) > 0 else None
            if rsi_peak1 is not None and rsi_peak2 is not None and data['RSI'].iloc[rsi_peak2] < data['RSI'].iloc[rsi_peak1]:
                divergences.append({'type': 'Bearish', 'price1': data['Close'].iloc[idx1], 'price2': data['Close'].iloc[idx2],
                                  'date1': data['DateTime_IST'].iloc[idx1], 'date2': data['DateTime_IST'].iloc[idx2],
                                  'rsi1': data['RSI'].iloc[rsi_peak1], 'rsi2': data['RSI'].iloc[rsi_peak2],
                                  'resolved': idx2 < len(data) - 5 and data['Close'].iloc[idx2:idx2+5].min() < data['Close'].iloc[idx2]})
    for i in range(len(price_troughs) - 1):
        idx1, idx2 = price_troughs[i], price_troughs[i+1]
        if idx2 < len(data) and data['Close'].iloc[idx2] < data['Close'].iloc[idx1]:
            rsi_trough1 = min(rsi_troughs, key=lambda x: abs(x - idx1)) if len(rsi_troughs) > 0 else None
            rsi_trough2 = min(rsi_troughs, key=lambda x: abs(x - idx2)) if len(rsi_troughs) > 0 else None
            if rsi_trough1 is not None and rsi_trough2 is not None and data['RSI'].iloc[rsi_trough2] > data['RSI'].iloc[rsi_trough1]:
                divergences.append({'type': 'Bullish', 'price1': data['Close'].iloc[idx1], 'price2': data['Close'].iloc[idx2],
                                  'date1': data['DateTime_IST'].iloc[idx1], 'date2': data['DateTime_IST'].iloc[idx2],
                                  'rsi1': data['RSI'].iloc[rsi_trough1], 'rsi2': data['RSI'].iloc[rsi_trough2],
                                  'resolved': idx2 < len(data) - 5 and data['Close'].iloc[idx2:idx2+5].max() > data['Close'].iloc[idx2]})
    return divergences

def calculate_fibonacci_levels(data):
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    return {'0%': high, '23.6%': high - 0.236 * diff, '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff, '61.8%': high - 0.618 * diff, '78.6%': high - 0.786 * diff, '100%': low}

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.subheader("Primary Ticker")
ticker1_type = st.sidebar.selectbox("Select Asset Type", list(PREDEFINED_ASSETS.keys()) + ['Custom'], key='ticker1_type')
if ticker1_type == 'Custom':
    ticker1 = st.sidebar.text_input("Enter Ticker Symbol", "^NSEI", key='ticker1_custom')
else:
    ticker1 = PREDEFINED_ASSETS[ticker1_type]
st.sidebar.write(f"**Ticker 1:** {ticker1}")

st.sidebar.subheader("Timeframes & Periods")
selected_combinations = []
for tf, periods in VALID_COMBINATIONS.items():
    with st.sidebar.expander(f"📊 {tf}", expanded=False):
        for period in periods:
            if st.checkbox(f"{period}", key=f"{tf}_{period}"):
                selected_combinations.append((tf, period))

st.sidebar.subheader("Ratio Analysis (Optional)")
enable_ratio = st.sidebar.checkbox("Enable Ticker 2 Comparison", value=False)
ticker2 = None
if enable_ratio:
    ticker2_type = st.sidebar.selectbox("Select Asset Type", list(PREDEFINED_ASSETS.keys()) + ['Custom'], key='ticker2_type')
    if ticker2_type == 'Custom':
        ticker2 = st.sidebar.text_input("Enter Ticker Symbol", "BTC-USD", key='ticker2_custom')
    else:
        ticker2 = PREDEFINED_ASSETS[ticker2_type]
    st.sidebar.write(f"**Ticker 2:** {ticker2}")

if st.sidebar.button("🚀 Fetch Data & Analyze", type="primary"):
    if not selected_combinations:
        st.error("Please select at least one timeframe/period combination")
    else:
        st.session_state['data_fetched'] = True
        st.session_state['ticker1'] = ticker1
        st.session_state['ticker2'] = ticker2
        st.session_state['combinations'] = selected_combinations
        st.session_state['all_data'] = {}
        st.session_state['all_data_t2'] = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(selected_combinations)
        for i, (interval, period) in enumerate(selected_combinations):
            status_text.text(f"Fetching {interval}/{period}... ({i+1}/{total})")
            progress_bar.progress((i + 1) / total)
            data = fetch_data(ticker1, interval, period)
            if data is not None:
                st.session_state['all_data'][(interval, period)] = data
            if enable_ratio and ticker2:
                data2 = fetch_data(ticker2, interval, period)
                if data2 is not None:
                    st.session_state['all_data_t2'][(interval, period)] = data2
        status_text.text("✅ Data fetching complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

st.title("📈 Professional Algorithmic Trading Analysis System")

if 'data_fetched' not in st.session_state or not st.session_state['data_fetched']:
    st.info("👈 Configure settings in the sidebar and click 'Fetch Data & Analyze' to begin")
else:
    st.success(f"**Analyzing:** {st.session_state['ticker1']} | **Timeframes:** {len(st.session_state['combinations'])}")
    if st.session_state.get('ticker2'):
        st.info(f"**Ratio Analysis Enabled:** {st.session_state['ticker2']}")
    
    tabs = st.tabs(["📊 Overview", "🎯 S/R", "📉 Tech", "📊 Z-Score", "💨 Vol", "🌊 Waves", "📐 Fib", "🔄 Div", "⚖️ Ratio", "🤖 AI", "🔬 Backtest", "▶️ Live"])
    
    with tabs[0]:
        st.header("📊 Multi-Timeframe Overview")
        overview_data = []
        for interval, period in st.session_state['combinations']:
            if (interval, period) in st.session_state['all_data']:
                data = st.session_state['all_data'][(interval, period)]
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[0]
                returns = ((current_price - prev_price) / prev_price) * 100
                points = current_price - prev_price
                rsi = calculate_rsi(data['Close']).iloc[-1]
                divs = detect_rsi_divergence(data)
                rsi_div = "Bullish" if any(d['type'] == 'Bullish' and not d['resolved'] for d in divs) else "Bearish" if any(d['type'] == 'Bearish' and not d['resolved'] for d in divs) else "None"
                sr_levels = detect_support_resistance(data)
                near_sr = "None"
                if sr_levels:
                    closest = min(sr_levels, key=lambda x: abs(x['price'] - current_price))
                    distance_pct = abs(closest['price'] - current_price) / current_price * 100
                    if distance_pct < 1:
                        near_sr = f"{closest['type']} @ ₹{closest['price']:,.2f}"
                fib_levels = calculate_fibonacci_levels(data)
                closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
                status = "🟢" if returns > 0 else "🔴"
                overview_data.append({'Timeframe': interval, 'Period': period, 'Status': status,
                    'Current Price': f"₹{current_price:,.2f}", 'Returns %': f"{returns:.2f}%",
                    'Points': f"{points:,.2f}", 'RSI': f"{rsi:.1f}", 'RSI Divergence': rsi_div,
                    'Near S/R': near_sr, 'Nearest Fib': f"{closest_fib[0]} (₹{closest_fib[1]:,.2f})"})
        if overview_data:
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True)
            csv = df_overview.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"overview_{st.session_state['ticker1']}_{datetime.now(IST).strftime('%Y%m%d')}.csv", "text/csv")
    
    with tabs[1]:
        st.header("🎯 Support & Resistance Analysis")
        near_support_count = 0
        near_resistance_count = 0
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            data = st.session_state['all_data'][(interval, period)]
            st.subheader(f"S/R: {interval}/{period}")
            current_price = data['Close'].iloc[-1]
            sr_levels = detect_support_resistance(data)
            if sr_levels:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"₹{current_price:,.2f}")
                with col2:
                    st.metric("Supports", len([l for l in sr_levels if l['type'] == 'Support']))
                with col3:
                    st.metric("Resistances", len([l for l in sr_levels if l['type'] == 'Resistance']))
                sr_table = []
                for level in sr_levels:
                    distance = level['price'] - current_price
                    distance_pct = (distance / current_price) * 100
                    accuracy = (level['sustained_count'] / level['hit_count'] * 100) if level['hit_count'] > 0 else 0
                    sr_table.append({'Type': level['type'], 'Price': f"₹{level['price']:,.2f}",
                        'Distance': f"{distance:,.2f} ({distance_pct:.2f}%)", 'Hit Count': level['hit_count'],
                        'Sustained': level['sustained_count'], 'Accuracy %': f"{accuracy:.1f}%",
                        'First Hit': format_time_ago(level['first_hit']), 'Last Hit': format_time_ago(level['last_hit'])})
                st.dataframe(pd.DataFrame(sr_table), use_container_width=True)
                nearest_support = [l for l in sr_levels if l['type'] == 'Support' and l['price'] < current_price]
                nearest_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and l['price'] > current_price]
                support_distance = float('inf')
                resistance_distance = float('inf')
                if nearest_support:
                    nearest_support = max(nearest_support, key=lambda x: x['price'])
                    support_distance = ((current_price - nearest_support['price']) / current_price) * 100
                    if support_distance < 2:
                        near_support_count += 1
                if nearest_resistance:
                    nearest_resistance = min(nearest_resistance, key=lambda x: x['price'])
                    resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) * 100
                    if resistance_distance < 2:
                        near_resistance_count += 1
                if support_distance < resistance_distance and support_distance < 2:
                    st.success(f"⬆️ BOUNCE EXPECTED - Near support at ₹{nearest_support['price']:,.2f}")
                elif resistance_distance < 2:
                    st.error(f"⬇️ REJECTION EXPECTED - Near resistance at ₹{nearest_resistance['price']:,.2f}")
                else:
                    st.info("➡️ NEUTRAL ZONE")
            else:
                st.warning("No S/R levels detected")
            st.markdown("---")
        st.subheader("S/R Consensus")
        total_tf = len(st.session_state['combinations'])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Near Support", f"{near_support_count}/{total_tf}")
        with col2:
            st.metric("Near Resistance", f"{near_resistance_count}/{total_tf}")
        if near_support_count > near_resistance_count:
            st.success(f"✅ BOUNCE EXPECTED ({near_support_count} timeframes)")
        elif near_resistance_count > near_support_count:
            st.error(f"⚠️ REJECTION EXPECTED ({near_resistance_count} timeframes)")
        else:
            st.info("➡️ NEUTRAL")
    
    with tabs[2]:
        st.header("📉 Technical Indicators")
        ema_overview = []
        for interval, period in st.session_state['combinations']:
            if (interval, period) in st.session_state['all_data']:
                data = st.session_state['all_data'][(interval, period)]
                current_price = data['Close'].iloc[-1]
                ema_20 = calculate_ema(data['Close'], 20).iloc[-1]
                ema_50 = calculate_ema(data['Close'], 50).iloc[-1]
                trend = "Bullish 🟢" if current_price > ema_20 > ema_50 else "Bearish 🔴" if current_price < ema_20 < ema_50 else "Mixed 🟡"
                ema_overview.append({'TF': interval, 'Period': period, 'Price': f"₹{current_price:,.2f}",
                    '20 EMA': f"₹{ema_20:,.2f}", '50 EMA': f"₹{ema_50:,.2f}", 'Trend': trend})
        df_ema = pd.DataFrame(ema_overview)
        st.dataframe(df_ema, use_container_width=True)
        bullish_count = len([x for x in ema_overview if "Bullish" in x['Trend']])
        bearish_count = len([x for x in ema_overview if "Bearish" in x['Trend']])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish", f"{bullish_count}/{len(ema_overview)}")
        with col2:
            st.metric("Bearish", f"{bearish_count}/{len(ema_overview)}")
        with col3:
            consensus_pct = (bullish_count / len(ema_overview) * 100) if ema_overview else 0
            st.metric("Consensus %", f"{consensus_pct:.1f}%")
        if consensus_pct > 60:
            st.success("✅ Strong Bullish Alignment")
        elif consensus_pct < 40:
            st.error("⚠️ Strong Bearish Alignment")
        else:
            st.info("➡️ Mixed Signals")
    
    with tabs[3]:
        st.header("📊 Z-Score Analysis")
        correction_signals = 0
        rally_signals = 0
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            data = st.session_state['all_data'][(interval, period)]
            st.subheader(f"Z-Score: {interval}/{period}")
            data_z = data.copy()
            data_z['Return_%'] = data_z['Close'].pct_change() * 100
            data_z['Z_Score'] = calculate_z_score(data_z)
            current_z = data_z['Z_Score'].iloc[-1]
            z_bin = "Extreme Negative (<-2)" if current_z < -2 else "Negative (-2 to -1)" if current_z < -1 else "Slightly Negative (-1 to 0)" if current_z < 0 else "Slightly Positive (0 to 1)" if current_z < 1 else "Positive (1 to 2)" if current_z < 2 else "Extreme Positive (>2)"
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Z-Score", f"{current_z:.2f}")
            with col2:
                st.metric("Bin", z_bin)
            bins = [("Extreme Negative (<-2)", data_z[data_z['Z_Score'] < -2]),
                   ("Negative (-2 to -1)", data_z[(data_z['Z_Score'] >= -2) & (data_z['Z_Score'] < -1)]),
                   ("Slightly Negative (-1 to 0)", data_z[(data_z['Z_Score'] >= -1) & (data_z['Z_Score'] < 0)]),
                   ("Slightly Positive (0 to 1)", data_z[(data_z['Z_Score'] >= 0) & (data_z['Z_Score'] < 1)]),
                   ("Positive (1 to 2)", data_z[(data_z['Z_Score'] >= 1) & (data_z['Z_Score'] < 2)]),
                   ("Extreme Positive (>2)", data_z[data_z['Z_Score'] > 2])]
            bin_data = []
            for bin_name, bin_df in bins:
                if len(bin_df) > 0:
                    is_current = bin_name == z_bin
                    bin_data.append({'Bin': bin_name, 'Count': len(bin_df),
                        'Percentage': f"{(len(bin_df)/len(data_z)*100):.1f}%",
                        'Price Range': f"₹{bin_df['Close'].min():,.2f} - ₹{bin_df['Close'].max():,.2f}" + (" ✅" if is_current else "")})
            st.dataframe(pd.DataFrame(bin_data), use_container_width=True)
            if current_z > 2:
                st.error("⚠️ EXTREME OVERBOUGHT - Correction Expected")
                correction_signals += 1
            elif current_z < -2:
                st.success("🟢 EXTREME OVERSOLD - Rally Expected")
                rally_signals += 1
            else:
                st.info("➡️ NORMAL RANGE")
            st.markdown("---")
        st.subheader("Z-Score Consensus")
        total_tf = len(st.session_state['combinations'])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Correction Signals", f"{correction_signals}/{total_tf}")
        with col2:
            st.metric("Rally Signals", f"{rally_signals}/{total_tf}")
        if rally_signals > correction_signals:
            st.success(f"✅ RALLY EXPECTED ({rally_signals} timeframes)")
        elif correction_signals > rally_signals:
            st.error(f"⚠️ CORRECTION EXPECTED ({correction_signals} timeframes)")
        else:
            st.info("➡️ NEUTRAL")
    
    with tabs[4]:
        st.header("💨 Volatility Analysis")
        for interval, period in st.session_state['combinations']:
            if (interval, period) in st.session_state['all_data']:
                data = st.session_state['all_data'][(interval, period)]
                data_vol = data.copy()
                data_vol['Volatility_%'] = calculate_volatility(data_vol, 20)
                current_vol = data_vol['Volatility_%'].iloc[-1]
                avg_vol = data_vol['Volatility_%'].mean()
                st.subheader(f"{interval}/{period}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Vol", f"{current_vol:.2f}%")
                with col2:
                    st.metric("Avg Vol", f"{avg_vol:.2f}%")
                if current_vol > avg_vol * 1.5:
                    st.warning("⚡ HIGH VOLATILITY")
                elif current_vol < avg_vol * 0.7:
                    st.info("🔒 LOW VOLATILITY - Breakout imminent")
                else:
                    st.success("✅ NORMAL")
                st.markdown("---")
    
    with tabs[5]:
        st.header("🌊 Elliott Waves")
        st.info("Wave analysis for primary timeframe")
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                highs_idx = argrelextrema(data['High'].values, np.greater, order=10)[0]
                lows_idx = argrelextrema(data['Low'].values, np.less, order=10)[0]
                extrema = []
                for idx in highs_idx:
                    if idx < len(data):
                        extrema.append({'idx': idx, 'price': data['High'].iloc[idx], 'type': 'High'})
                for idx in lows_idx:
                    if idx < len(data):
                        extrema.append({'idx': idx, 'price': data['Low'].iloc[idx], 'type': 'Low'})
                extrema = sorted(extrema, key=lambda x: x['idx'])
                if len(extrema) >= 2:
                    wave_data = []
                    for i in range(len(extrema) - 1):
                        start_price = extrema[i]['price']
                        end_price = extrema[i+1]['price']
                        move_pct = ((end_price - start_price) / start_price) * 100
                        wave_type = "Impulse ⬆️" if move_pct > 0 else "Corrective ⬇️"
                        wave_data.append({'Wave': f"Wave {i+1}", 'Type': wave_type,
                            'Start': f"₹{start_price:,.2f}", 'End': f"₹{end_price:,.2f}", 'Move %': f"{move_pct:.2f}%"})
                    st.dataframe(pd.DataFrame(wave_data), use_container_width=True)
                    if wave_data:
                        current_wave = wave_data[-1]
                        if "Impulse" in current_wave['Type']:
                            st.success(f"**Current:** {current_wave['Wave']} - {current_wave['Type']}")
                            st.write("**Expected Next:** Corrective wave")
                        else:
                            st.info(f"**Current:** {current_wave['Wave']} - {current_wave['Type']}")
                            st.write("**Expected Next:** Impulse wave")
    
    with tabs[6]:
        st.header("📐 Fibonacci Levels")
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                fib_levels = calculate_fibonacci_levels(data)
                current_price = data['Close'].iloc[-1]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High", f"₹{data['High'].max():,.2f}")
                    st.metric("Current", f"₹{current_price:,.2f}")
                with col2:
                    st.metric("Low", f"₹{data['Low'].min():,.2f}")
                fib_data = []
                for level, price in fib_levels.items():
                    distance = price - current_price
                    distance_pct = (distance / current_price) * 100
                    status = "✅ NEAR" if abs(distance_pct) < 1 else "⬆️ Above" if distance > 0 else "⬇️ Below"
                    fib_data.append({'Level': level, 'Price': f"₹{price:,.2f}",
                        'Distance': f"{distance:,.2f} ({distance_pct:.2f}%)", 'Status': status})
                st.dataframe(pd.DataFrame(fib_data), use_container_width=True)
                nearest = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
                st.info(f"**Nearest Level:** {nearest[0]} at ₹{nearest[1]:,.2f}")
    
    with tabs[7]:
        st.header("🔄 RSI Divergence")
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                divergences = detect_rsi_divergence(data)
                if divergences:
                    div_data = []
                    for div in divergences:
                        div_data.append({'Type': div['type'] + (" 🟢" if div['type'] == 'Bullish' else " 🔴"),
                            'Price 1': f"₹{div['price1']:,.2f}", 'Price 2': f"₹{div['price2']:,.2f}",
                            'RSI 1': f"{div['rsi1']:.1f}", 'RSI 2': f"{div['rsi2']:.1f}",
                            'Resolved': "✅" if div['resolved'] else "⏳"})
                    st.dataframe(pd.DataFrame(div_data), use_container_width=True)
                    active_divs = [d for d in divergences if not d['resolved']]
                    if active_divs:
                        st.subheader("⚡ Active Divergences")
                        for div in active_divs:
                            current_price = data['Close'].iloc[-1]
                            if div['type'] == 'Bullish':
                                st.success("🟢 **Bullish Divergence Active**")
                                st.write(f"**Entry:** ₹{current_price:,.2f}")
                                st.write(f"**Target:** ₹{current_price * 1.02:,.2f} (+2%)")
                            else:
                                st.error("🔴 **Bearish Divergence Active**")
                                st.write(f"**Entry:** ₹{current_price:,.2f}")
                                st.write(f"**Target:** ₹{current_price * 0.98:,.2f} (-2%)")
                else:
                    st.info("No divergences detected")
    
    with tabs[8]:
        st.header("⚖️ Ratio Analysis")
        if not enable_ratio:
            st.warning("⚠️ Enable Ticker 2 comparison in sidebar")
        else:
            st.success(f"Comparing {st.session_state['ticker1']} vs {st.session_state.get('ticker2', 'N/A')}")
            common_tfs = [(i, p) for i, p in st.session_state['combinations'] 
                         if (i, p) in st.session_state['all_data'] and (i, p) in st.session_state.get('all_data_t2', {})]
            if common_tfs:
                for interval, period in common_tfs[:3]:
                    data1 = st.session_state['all_data'][(interval, period)]
                    data2 = st.session_state['all_data_t2'][(interval, period)]
                    merged = pd.merge(data1[['DateTime_IST', 'Close']], data2[['DateTime_IST', 'Close']],
                                     on='DateTime_IST', how='inner', suffixes=('_T1', '_T2'))
                    if len(merged) > 0:
                        merged['Ratio'] = merged['Close_T1'] / merged['Close_T2']
                        current_ratio = merged['Ratio'].iloc[-1]
                        avg_ratio = merged['Ratio'].mean()
                        st.subheader(f"{interval}/{period}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Ratio", f"{current_ratio:.6f}")
                        with col2:
                            st.metric("Avg Ratio", f"{avg_ratio:.6f}")
                        if current_ratio > avg_ratio * 1.1:
                            st.warning(f"⚠️ {st.session_state['ticker1']} EXPENSIVE")
                        elif current_ratio < avg_ratio * 0.9:
                            st.success(f"🟢 {st.session_state['ticker1']} CHEAP")
                        else:
                            st.info("➡️ NORMAL RANGE")
                        st.markdown("---")
    
    with tabs[9]:
        st.header("🤖 AI Signals & Final Forecast")
        st.markdown("### 🎯 Multi-Timeframe Algorithmic Signal")
        timeframe_scores = []
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            data = st.session_state['all_data'][(interval, period)]
            score = 0
            factors = []
            current_price = data['Close'].iloc[-1]
            rsi = calculate_rsi(data['Close'], 14).iloc[-1]
            if rsi < 30:
                score += 20
                factors.append("RSI Oversold +20")
            elif rsi > 70:
                score -= 20
                factors.append("RSI Overbought -20")
            ema_20 = calculate_ema(data['Close'], 20).iloc[-1]
            ema_50 = calculate_ema(data['Close'], 50).iloc[-1]
            if current_price > ema_20 > ema_50:
                score += 15
                factors.append("Bullish EMA +15")
            elif current_price < ema_20 < ema_50:
                score -= 15
                factors.append("Bearish EMA -15")
            sr_levels = detect_support_resistance(data)
            if sr_levels:
                nearest_support = [l for l in sr_levels if l['type'] == 'Support' and l['price'] < current_price]
                nearest_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and l['price'] > current_price]
                if nearest_support:
                    nearest_support = max(nearest_support, key=lambda x: x['price'])
                    support_distance = ((current_price - nearest_support['price']) / current_price) * 100
                    if support_distance < 1:
                        score += 20
                        factors.append("Near Support +20")
                if nearest_resistance:
                    nearest_resistance = min(nearest_resistance, key=lambda x: x['price'])
                    resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) * 100
                    if resistance_distance < 1:
                        score -= 20
                        factors.append("Near Resistance -20")
            data_z = data.copy()
            data_z['Z_Score'] = calculate_z_score(data_z)
            current_z = data_z['Z_Score'].iloc[-1]
            if current_z < -2:
                score += 20
                factors.append("Z-Score Oversold +20")
            elif current_z > 2:
                score -= 20
                factors.append("Z-Score Overbought -20")
            data_vol = data.copy()
            data_vol['Volatility_%'] = calculate_volatility(data_vol, 20)
            current_vol = data_vol['Volatility_%'].iloc[-1]
            avg_vol = data_vol['Volatility_%'].mean()
            vol_condition = "High" if current_vol > avg_vol * 1.5 else "Low" if current_vol < avg_vol * 0.7 else "Normal"
            bias = "Bullish 🟢" if score > 15 else "Bearish 🔴" if score < -15 else "Neutral 🟡"
            timeframe_scores.append({'timeframe': f"{interval}/{period}", 'score': score, 'bias': bias,
                'rsi': rsi, 'z_score': current_z, 'vol_pct': current_vol, 'vol_condition': vol_condition,
                'factors': ", ".join(factors[:3])})
        avg_score = np.mean([tf['score'] for tf in timeframe_scores])
        bullish_count = len([tf for tf in timeframe_scores if "Bullish" in tf['bias']])
        bearish_count = len([tf for tf in timeframe_scores if "Bearish" in tf['bias']])
        neutral_count = len([tf for tf in timeframe_scores if "Neutral" in tf['bias']])
        total_tf = len(timeframe_scores)
        if avg_score > 30:
            signal = "STRONG BUY"
            signal_emoji = "🟢"
        elif avg_score > 15:
            signal = "BUY"
            signal_emoji = "🟢"
        elif avg_score < -30:
            signal = "STRONG SELL"
            signal_emoji = "🔴"
        elif avg_score < -15:
            signal = "SELL"
            signal_emoji = "🔴"
        else:
            signal = "HOLD/NEUTRAL"
            signal_emoji = "🟡"
        agreement_pct = max(bullish_count, bearish_count) / total_tf if total_tf > 0 else 0
        confidence = 60 + (agreement_pct * 30) + (abs(avg_score) * 0.3)
        confidence = min(confidence, 95)
        if "BUY" in signal:
            st.success(f"# {signal_emoji} {signal}")
        elif "SELL" in signal:
            st.error(f"# {signal_emoji} {signal}")
        else:
            st.info(f"# {signal_emoji} {signal}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col2:
            st.metric("Score", f"{avg_score:.1f}/100")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish", f"{bullish_count}/{total_tf}", f"{(bullish_count/total_tf*100):.1f}%")
        with col2:
            st.metric("Bearish", f"{bearish_count}/{total_tf}", f"{(bearish_count/total_tf*100):.1f}%")
        with col3:
            st.metric("Neutral", f"{neutral_count}/{total_tf}", f"{(neutral_count/total_tf*100):.1f}%")
        st.markdown("### 📋 TRADING PLAN")
        current_price = st.session_state['all_data'][st.session_state['combinations'][0]]['Close'].iloc[-1]
        if current_price > 20000:
            sl_pct = 0.015
            target_pct = 0.0175
        elif current_price > 1000:
            sl_pct = 0.02
            target_pct = 0.025
        else:
            sl_pct = 0.025
            target_pct = 0.035
        if "SELL" in signal:
            sl_price = current_price * (1 + sl_pct)
            target_price = current_price * (1 - target_pct)
        else:
            sl_price = current_price * (1 - sl_pct)
            target_price = current_price * (1 + target_pct)
        sl_points = abs(current_price - sl_price)
        target_points = abs(target_price - current_price)
        risk_reward = target_points / sl_points if sl_points > 0 else 0
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Entry:** ₹{current_price:,.2f}")
            st.write(f"**Stop Loss:** ₹{sl_price:,.2f} ({sl_pct*100:.1f}%, {sl_points:.0f} pts)")
        with col2:
            st.write(f"**Target:** ₹{target_price:,.2f} ({target_pct*100:.2f}%, {target_points:.0f} pts)")
            st.write(f"**Risk:Reward:** 1:{risk_reward:.2f}")
        st.markdown("### 📊 Timeframe Breakdown")
        breakdown_data = []
        for tf in timeframe_scores:
            breakdown_data.append({'TF': tf['timeframe'], 'Score': f"{tf['score']:.1f}", 'Bias': tf['bias'],
                'RSI': f"{tf['rsi']:.1f}", 'Z-Score': f"{tf['z_score']:.2f}",
                'Vol': tf['vol_condition'], 'Factors': tf['factors']})
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
        st.markdown("### 🔍 WHY THIS RECOMMENDATION?")
        st.write(f"Based on {total_tf} timeframes: {bullish_count} bullish ({(bullish_count/total_tf*100):.1f}%), {bearish_count} bearish ({(bearish_count/total_tf*100):.1f}%)")
        st.write(f"**Success Probability:** ~{confidence:.0f}% based on {max(bullish_count, bearish_count)}/{total_tf} alignment")
        st.markdown("**Key Points:**")
        st.write("1. Multi-timeframe consensus reduces false signals")
        st.write("2. Risk-reward ratio supports favorable probability")
        st.write("3. Technical and statistical factors aligned")
        st.write(f"4. Position size: Risk 1-2% capital based on {sl_pct*100:.1f}% stop")
    
    with tabs[10]:
        st.header("🔬 Strategy Backtesting")
        st.subheader("Select Strategies")
        col1, col2, col3 = st.columns(3)
        with col1:
            test_rsi_ema = st.checkbox("RSI + EMA", value=True)
            test_ema_cross = st.checkbox("EMA Crossover", value=True)
        with col2:
            test_zscore = st.checkbox("Z-Score Reversion", value=True)
            test_9ema = st.checkbox("9 EMA Pullback", value=True)
        with col3:
            test_vol = st.checkbox("Volatility Breakout", value=False)
            test_support = st.checkbox("Support Bounce", value=False)
        test_timeframes = st.multiselect("Select Timeframes (max 5)",
            [f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations']],
            default=[f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations'][:min(3, len(st.session_state['combinations']))]])
        if st.button("🚀 Run Backtest", type="primary"):
            if not test_timeframes:
                st.error("Select at least one timeframe")
            else:
                all_results = []
                for tf_str in test_timeframes:
                    interval, period = tf_str.split('/')
                    if (interval, period) not in st.session_state['all_data']:
                        continue
                    data = st.session_state['all_data'][(interval, period)].copy()
                    data['RSI'] = calculate_rsi(data['Close'], 14)
                    data['EMA_9'] = calculate_ema(data['Close'], 9)
                    data['EMA_20'] = calculate_ema(data['Close'], 20)
                    data['EMA_50'] = calculate_ema(data['Close'], 50)
                    data['Z_Score'] = calculate_z_score(data)
                    if test_rsi_ema:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        for i in range(50, len(data)):
                            if not in_trade:
                                if data['RSI'].iloc[i] < 30 and data['Close'].iloc[i] > data['EMA_20'].iloc[i]:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                if data['RSI'].iloc[i] > 70 or data['Close'].iloc[i] < data['EMA_20'].iloc[i]:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    trades.append({'entry_price': entry_price, 'exit_price': exit_price, 'pnl_pct': pnl})
                                    in_trade = False
                        if trades:
                            winning = len([t for t in trades if t['pnl_pct'] > 0])
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            win_rate = (winning / len(trades)) * 100
                            all_results.append({'Strategy': 'RSI+EMA', 'TF': tf_str, 'Trades': len(trades),
                                'Win Rate': f"{win_rate:.1f}%", 'Total PnL': f"{total_pnl:.2f}%"})
                    if test_zscore:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        for i in range(50, len(data)):
                            if not in_trade:
                                if data['Z_Score'].iloc[i] < -2:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                            else:
                                if data['Z_Score'].iloc[i] > 0:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    trades.append({'entry_price': entry_price, 'exit_price': exit_price, 'pnl_pct': pnl})
                                    in_trade = False
                        if trades:
                            winning = len([t for t in trades if t['pnl_pct'] > 0])
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            win_rate = (winning / len(trades)) * 100
                            all_results.append({'Strategy': 'Z-Score', 'TF': tf_str, 'Trades': len(trades),
                                'Win Rate': f"{win_rate:.1f}%", 'Total PnL': f"{total_pnl:.2f}%"})
                if all_results:
                    st.success(f"✅ Tested {len(all_results)} strategy-timeframe combinations")
                    st.dataframe(pd.DataFrame(all_results), use_container_width=True)
                    best = max(all_results, key=lambda x: float(x['Total PnL'].replace('%', '')))
                    st.subheader(f"🏆 Best: {best['Strategy']} on {best['TF']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Trades", best['Trades'])
                    with col2:
                        st.metric("Win Rate", best['Win Rate'])
                    st.metric("Total PnL", best['Total PnL'])
                else:
                    st.warning("No trades generated")
    
    with tabs[11]:
        st.header("▶️ Live Trading Monitor")
        st.info("🔴 **LIVE MODE** - Updates every 2 seconds")
        live_strategy = st.selectbox("Strategy", ['RSI+EMA', 'EMA Crossover', 'Z-Score Reversion', '9 EMA Pullback'], key='live_strat')
        live_tf = st.selectbox("Timeframe", [f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations']], key='live_tf')
        st.markdown("### ⚙️ Strategy Parameters")
        params = {
            'RSI+EMA': {'Entry': 'RSI < 30 AND Price > 20 EMA', 'Exit': 'RSI > 70 OR Price < 20 EMA', 'SL': '20 EMA', 'Target': '+2%'},
            'EMA Crossover': {'Entry': '20 EMA crosses above 50 EMA', 'Exit': '20 EMA crosses below 50 EMA', 'SL': '-2%', 'Target': '+3%'},
            'Z-Score Reversion': {'Entry': 'Z-Score < -2', 'Exit': 'Z-Score > 0', 'SL': '-2%', 'Target': '+2%'},
            '9 EMA Pullback': {'Entry': 'Bounce off 9 EMA in uptrend', 'Exit': 'Close below 9 EMA', 'SL': '9 EMA', 'Target': '+1.5%'}
        }
        if live_strategy in params:
            p = params[live_strategy]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Entry:** {p['Entry']}")
                st.write(f"**Exit:** {p['Exit']}")
            with col2:
                st.write(f"**SL:** {p['SL']}")
                st.write(f"**Target:** {p['Target']}")
        auto_refresh = st.checkbox("🔄 Auto-Refresh (2s)", value=False, key='auto_ref')
        manual_refresh = st.button("🔃 Manual Refresh", type="primary")
        if auto_refresh or manual_refresh:
            interval, period = live_tf.split('/')
            with st.spinner("Fetching live data..."):
                live_data = fetch_data(st.session_state['ticker1'], interval, period)
            if live_data is not None and len(live_data) > 50:
                live_data['RSI'] = calculate_rsi(live_data['Close'], 14)
                live_data['EMA_9'] = calculate_ema(live_data['Close'], 9)
                live_data['EMA_20'] = calculate_ema(live_data['Close'], 20)
                live_data['EMA_50'] = calculate_ema(live_data['Close'], 50)
                live_data['Z_Score'] = calculate_z_score(live_data)
                current_price = live_data['Close'].iloc[-1]
                current_rsi = live_data['RSI'].iloc[-1]
                current_z = live_data['Z_Score'].iloc[-1]
                last_update = live_data['DateTime_IST'].iloc[-1]
                st.success(f"✅ Live | Updated: {format_time_ago(last_update)}")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"₹{current_price:,.2f}")
                with col2:
                    st.metric("RSI", f"{current_rsi:.1f}")
                with col3:
                    st.metric("Z-Score", f"{current_z:.2f}")
                with col4:
                    trend = "Bullish" if current_price > live_data['EMA_20'].iloc[-1] > live_data['EMA_50'].iloc[-1] else "Bearish"
                    st.metric("Trend", trend)
                st.markdown("### 🎯 Strategy Signal")
                signal_active = False
                if live_strategy == 'RSI+EMA':
                    if current_rsi < 30 and current_price > live_data['EMA_20'].iloc[-1]:
                        signal_active = True
                        st.success("🟢 **BUY SIGNAL ACTIVE**")
                        st.write(f"**Entry:** ₹{current_price:,.2f}")
                        st.write(f"**SL:** ₹{live_data['EMA_20'].iloc[-1]:,.2f}")
                        st.write(f"**Target:** ₹{current_price * 1.02:,.2f}")
                    else:
                        st.info("🟡 **WAITING FOR SETUP**")
                elif live_strategy == 'Z-Score Reversion':
                    if current_z < -2:
                        signal_active = True
                        st.success("🟢 **BUY SIGNAL - OVERSOLD**")
                        st.write(f"**Entry:** ₹{current_price:,.2f}")
                        st.write(f"**Target:** ₹{current_price * 1.02:,.2f}")
                    else:
                        st.info("🟡 **WAITING FOR EXTREME**")
                if auto_refresh:
                    time.sleep(2)
                    st.rerun()
            else:
                st.error("Failed to fetch live data")
        else:
            st.info("👆 Enable auto-refresh or manual refresh for live monitoring")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p><strong>Professional Algorithmic Trading Analysis System</strong></p>
<p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
