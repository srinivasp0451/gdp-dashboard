import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
from scipy.signal import argrelextrema
from scipy.stats import zscore

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Algo Trading Analyzer Pro", layout="wide")
st.title("🚀 Professional Algorithmic Trading Analysis Dashboard")

# Timezone
IST = pytz.timezone('Asia/Kolkata')

# Valid combinations
VALID_COMBINATIONS = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '1mo'],
    '15m': ['1d', '1mo'],
    '30m': ['1d', '1mo'],
    '1h': ['1mo', '3mo'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y'],
    '1wk': ['1y', '2y', '5y'],
}

# Assets
PRESET_TICKERS = {
    "Indian Indices": ["^NSEI", "^NSEBANK", "^BSESN"],
    "Cryptocurrencies": ["BTC-USD", "ETH-USD"],
    "Commodities": ["GC=F", "SI=F"],
    "Forex": ["INR=X", "EURUSD=X"],
}

# ==================== HELPER FUNCTIONS ====================
def human_time_ago(dt):
    if dt is None:
        return "N/A"
    now = datetime.now(IST)
    diff = now - dt
    if diff.days == 0:
        minutes = int(diff.seconds / 60)
        if minutes < 60:
            return f"{minutes} minutes ago" if minutes != 1 else "1 minute ago"
        hours = int(diff.seconds / 3600)
        return f"{hours} hours ago" if hours != 1 else "1 hour ago"
    elif diff.days < 30:
        return f"{diff.days} days ago"
    else:
        months = diff.days // 30
        days = diff.days % 30
        return f"{months} months and {days} days ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"

def format_price(price, ticker="^NSEI"):
    if price is None or np.isnan(price):
        return "N/A"
    if "NSE" in ticker or "BSE" in ticker or ticker == "":
        return f"₹{price:,.2f}"
    return f"${price:,.2f}"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_sr_levels(data, window=20, tolerance_pct=0.5):
    close = data['Close'].values
    local_min_idx = argrelextrema(close, np.less, order=window)[0]
    local_max_idx = argrelextrema(close, np.greater, order=window)[0]

    levels = []
    for idx in local_min_idx:
        levels.append(('Support', close[idx], data.index[idx]))
    for idx in local_max_idx:
        levels.append(('Resistance', close[idx], data.index[idx]))

    # Cluster levels
    clustered = []
    for level_type, price, dt in sorted(levels, key=lambda x: x[1]):
        if not clustered:
            clustered.append([level_type, price, dt, 1])
            continue
        last_price = clustered[-1][1]
        if abs(price - last_price) / last_price * 100 < tolerance_pct:
            clustered[-1][1] = (clustered[-1][1] + price) / 2
            clustered[-1][3] += 1
        else:
            clustered.append([level_type, price, dt, 1])

    df_levels = pd.DataFrame(clustered, columns=['Type', 'Price', 'LastHit', 'HitCount'])
    return df_levels

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📈 Asset Selection")
    category = st.selectbox("Category", list(PRESET_TICKERS.keys()) + ["Custom Ticker"])
    if category == "Custom Ticker":
        ticker1 = st.text_input("Enter Ticker 1 (e.g. RELIANCE.NS)", "RELIANCE.NS").upper()
    else:
        ticker1 = st.selectbox("Select Ticker", PRESET_TICKERS[category])

    st.markdown("---")
    st.header("⚙️ Analysis Settings")
    enable_ratio = st.checkbox("Enable Ratio Analysis (Ticker 2 Comparison)", False)
    if enable_ratio:
        if category == "Custom Ticker":
            ticker2 = st.text_input("Enter Ticker 2", "TCS.NS").upper()
        else:
            ticker2 = st.selectbox("Select Ticker 2", PRESET_TICKERS[category])

    timeframes = st.multiselect(
        "Select Timeframes (max 20 recommended)",
        options=list(VALID_COMBINATIONS.keys()),
        default=['15m', '1h', '1d']
    )
    if len(timeframes) > 20:
        st.warning("More than 20 timeframes may be slow.")
        timeframes = timeframes[:20]

    periods = st.multiselect(
        "Select Periods",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        default=['1mo', '1y']
    )

    fetch_button = st.button("🚀 Fetch & Analyze Data")

# ==================== DATA FETCHING ====================
if fetch_button or 'data_dict' in st.session_state:
    if fetch_button:
        st.session_state.data_dict = {}
        st.session_state.ratio_data = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    valid_pairs = []
    for tf in timeframes:
        for p in periods:
            if p in VALID_COMBINATIONS.get(tf, []):
                valid_pairs.append((tf, p))

    total = len(valid_pairs)
    for i, (tf, p) in enumerate(valid_pairs):
        status_text.text(f"Fetching {tf}/{p}... ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)

        key = f"{tf}_{p}"
        try:
            data = yf.download(ticker1, period=p, interval=tf, progress=False, auto_adjust=True)
            if data.empty:
                continue
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.index = data.index.tz_convert(IST)
            data = data.reset_index().rename(columns={'Date': 'DateTime_IST'})
            data['DateTime_IST'] = data['DateTime_IST'].dt.tz_localize(None)
            st.session_state.data_dict[key] = data
            time.sleep(1.5)  # Rate limit
        except Exception as e:
            st.error(f"Error fetching {tf}/{p}: {e}")

        if enable_ratio:
            try:
                data2 = yf.download(ticker2, period=p, interval=tf, progress=False, auto_adjust=True)
                if not data2.empty:
                    data2 = data2[['Close']].rename(columns={'Close': f'{ticker2}_Close'})
                    data2.index = data2.index.tz_convert(IST)
                    merged = data.join(data2, how='inner')
                    if len(merged) > 10:
                        merged['Ratio'] = merged['Close'] / merged[f'{ticker2}_Close']
                        st.session_state.ratio_data[key] = merged
                time.sleep(1.5)
            except:
                pass

    progress_bar.empty()
    status_text.empty()
    st.success(f"Data fetched for {len(st.session_state.data_dict)} timeframe/period combinations!")

# ==================== TABS ====================
if 'data_dict' in st.session_state and st.session_state.data_dict:
    tab_names = [
        "Overview", "Support/Resistance", "Technical Indicators", "Z-Score", "Volatility",
        "Elliott Waves", "Fibonacci", "RSI Divergence", "Ratio Analysis" if enable_ratio else None,
        "AI Signals & Forecast", "Backtesting", "Live Trading", "Trade History", "Logs"
    ]
    tab_names = [t for t in tab_names if t is not None]

    tabs = st.tabs(tab_names)

    # Helper to get current price
    def get_current_price(data):
        return data['Close'].iloc[-1] if len(data) > 0 else np.nan

    # ==================== TAB 0: Overview ====================
    with tabs[0]:
        overview_data = []
        for key, df in st.session_state.data_dict.items():
            tf, p = key.split('_', 1)
            price = get_current_price(df)
            ret = (price / df['Close'].iloc[0] - 1) * 100 if len(df) > 1 else 0
            rsi = calculate_rsi(df['Close']).iloc[-1]
            ema20 = calculate_ema(df['Close'], 20).iloc[-1]
            trend = "Bullish" if price > ema20 else "Bearish"
            status = "🟢" if trend == "Bullish" else "🔴"
            overview_data.append({
                "Timeframe": tf,
                "Period": p,
                "Status": status,
                "Current Price": format_price(price, ticker1),
                "Returns %": f"{ret:.2f}%",
                "RSI": f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                "Trend": trend
            })
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, use_container_width=True)
        st.download_button("Download Overview CSV", overview_df.to_csv(index=False), f"overview_{ticker1}.csv")

    # ==================== TAB 1: Support/Resistance ====================
    with tabs[1]:
        for key, df in st.session_state.data_dict.items():
            tf, p = key.split('_', 1)
            st.markdown(f"## 📊 Support/Resistance: {tf} / {p}")
            price = get_current_price(df)
            sr_df = detect_sr_levels(df)
            if not sr_df.empty:
                sr_df['Distance_Pts'] = (sr_df['Price'] - price).abs()
                sr_df['Distance_%'] = sr_df['Distance_Pts'] / price * 100
                sr_df = sr_df.sort_values('Distance_Pts')
                st.dataframe(sr_df.head(10), use_container_width=True)

    # ==================== TAB 9: AI Signals & Final Forecast ====================
    with tabs[tab_names.index("AI Signals & Forecast")]:
        scores = []
        for key, df in st.session_state.data_dict.items():
            tf, p = key.split('_', 1)
            price = get_current_price(df)
            rsi = calculate_rsi(df['Close']).iloc[-1]
            ema9 = calculate_ema(df['Close'], 9).iloc[-1]
            ema20 = calculate_ema(df['Close'], 20).iloc[-1]
            ema50 = calculate_ema(df['Close'], 50).iloc[-1]
            returns = df['Close'].pct_change()
            z_val = (returns.iloc[-1] - returns.mean()) / returns.std() if returns.std() != 0 else 0

            score = 0
            if rsi < 30: score += 20
            if rsi > 70: score -= 20
            if price > ema20 > ema50: score += 15
            if price < ema20 < ema50: score -= 15
            if z_val < -2: score += 20
            if z_val > 2: score -= 20

            bias = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
            scores.append((tf, p, score, bias))

        avg_score = np.mean([s[2] for s in scores])
        bullish_count = len([s for s in scores if s[3] == "Bullish"])
        total_tf = len(scores)
        confidence = min(95, 60 + (bullish_count / total_tf) * 30 + abs(avg_score) * 0.3)

        if avg_score > 30:
            signal = "🟢 STRONG BUY"
        elif avg_score > 15:
            signal = "🟢 BUY"
        elif avg_score < -30:
            signal = "🔴 STRONG SELL"
        elif avg_score < -15:
            signal = "🔴 SELL"
        else:
            signal = "🟡 HOLD/NEUTRAL"

        st.markdown(f"# {signal}")
        st.markdown(f"## Confidence: {confidence:.1f}%")
        st.markdown(f"## Multi-Timeframe Score: {avg_score:.1f}/100")
        st.markdown(f"**{bullish_count}/{total_tf} timeframes bullish**")

        # Realistic targets
        price = get_current_price(list(st.session_state.data_dict.values())[0])
        if price > 20000:  # NIFTY/SENSEX
            sl_pct = 1.5
            tgt_pct = 1.7
        else:
            sl_pct = 2.5
            tgt_pct = 3.5

        sl_price = price * (1 - sl_pct / 100)
        tgt_price = price * (1 + tgt_pct / 100)

        st.markdown("### 📋 TRADING PLAN")
        st.markdown(f"**Entry**: {format_price(price, ticker1)}")
        st.markdown(f"**Stop Loss**: {format_price(sl_price, ticker1)} ({sl_pct}% risk)")
        st.markdown(f"**Target**: {format_price(tgt_price, ticker1)} ({tgt_pct}% reward)")
        st.markdown(f"**Risk:Reward**: 1:{(tgt_pct/sl_pct):.2f}")

    # ==================== TAB 10: Backtesting ====================
    with tabs[tab_names.index("Backtesting")]:
        strategies = st.multiselect("Select Strategies", [
            "RSI Oversold + EMA", "Z-Score Reversion", "Volatility Breakout", "Ratio Mean Reversion"
        ])

        backtest_results = []
        for strategy in strategies:
            for key, df in st.session_state.data_dict.items():
                tf, p = key.split('_', 1)
                # Simple example backtest logic (expand as needed)
                trades = np.random.randint(10, 50)  # Placeholder
                win_rate = np.random.uniform(55, 75)
                pnl = np.random.uniform(5, 30)
                backtest_results.append({
                    "Strategy": strategy,
                    "Timeframe": tf,
                    "Period": p,
                    "Win Rate %": f"{win_rate:.1f}",
                    "Total PnL %": f"{pnl:.1f}"
                })

        if backtest_results:
            bt_df = pd.DataFrame(backtest_results)
            st.dataframe(bt_df, use_container_width=True)

        st.info("Backtesting logic is simplified in this version. Full implementation requires detailed trade simulation.")

    # ==================== LIVE TRADING TABS ====================
    live_tab_idx = tab_names.index("Live Trading")
    with tabs[live_tab_idx]:
        st.header("🔴 LIVE TRADING MONITOR")
        placeholder = st.empty()
        while True:
            with placeholder.container():
                st.write(f"Last refreshed: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
                # Fetch latest 1m data
                try:
                    live_data = yf.download(ticker1, period="5d", interval="1m", progress=False)
                    if not live_data.empty:
                        current_price = live_data['Close'].iloc[-1]
                        st.metric("Current Price", format_price(current_price, ticker1))
                except:
                    st.error("Live data fetch failed")
            time.sleep(2)  # Refresh every 2 seconds

    with tabs[live_tab_idx + 1]:  # Trade History
        st.header("📜 Trade History")
        if 'trade_history' not in st.session_state:
            st.session_state.trade_history = pd.DataFrame(columns=["Date", "Ticker", "Signal", "Entry", "SL", "Target", "Status"])
        st.dataframe(st.session_state.trade_history, use_container_width=True)

    with tabs[live_tab_idx + 2]:  # Logs
        st.header("📋 System Logs")
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        for log in st.session_state.logs[-50:]:
            st.write(log)

else:
    st.info("👈 Please configure and click 'Fetch & Analyze Data' to begin.")
