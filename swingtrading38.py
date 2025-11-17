# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import io
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration & CSS
# ---------------------------
st.set_page_config(page_title="Advanced Trading Pattern Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; font-weight: 700; color: #ff7f0e; margin-top: 0.8rem;}
    .metric-card {background-color: #f0f2f6; padding: 0.6rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .positive {color: #008000; font-weight: bold;}
    .negative {color: #c00000; font-weight: bold;}
    .neutral {color: #b8860b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Constants & ticker mapping
# ---------------------------
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom": "CUSTOM"
}

# ---------------------------
# Session state defaults
# ---------------------------
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None
if 'ratio_df' not in st.session_state:
    st.session_state.ratio_df = None

# ---------------------------
# Helper functions
# ---------------------------
def convert_to_ist(df):
    """Convert dataframe index to IST timezone (Asia/Kolkata)."""
    if df is None or df.empty:
        return df
    try:
        # If index is tz-naive, assume UTC then convert
        if df.index.tz is None:
            df = df.tz_localize('UTC')
        df = df.tz_convert('Asia/Kolkata')
    except Exception:
        # best-effort fallback: leave as-is
        pass
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when not enough data
    return rsi

def find_pattern_matches(df, window_size=10, top_n=20):
    """
    Find similar historical patterns to the most recent 'window_size' close values.
    Returns DataFrame of matches with readable fields.
    """
    if df is None or len(df) < window_size * 2:
        return pd.DataFrame()

    closes = df['Close'].values
    current_pattern = closes[-window_size:]
    # normalized
    cur_min, cur_max = current_pattern.min(), current_pattern.max()
    current_normalized = (current_pattern - cur_min) / (cur_max - cur_min + 1e-10)
    current_volatility = df['Close'].tail(window_size).pct_change().std()

    matches = []
    # iterate historic windows (exclude very recent to avoid overlap)
    for i in range(window_size, len(df) - window_size - 1):
        hist_window = closes[i - window_size:i]
        hmin, hmax = hist_window.min(), hist_window.max()
        hist_norm = (hist_window - hmin) / (hmax - hmin + 1e-10)

        # similarity measures
        distance = float(euclidean(current_normalized, hist_norm))
        correlation, _ = pearsonr(current_normalized, hist_norm)

        hist_vol = df['Close'].iloc[i - window_size:i].pct_change().std()

        # future movements expressed as absolute points and percent
        future_points = {}
        for j in [1, 3, 5, 10]:
            idx_future = i + j
            if idx_future < len(df):
                points = closes[idx_future] - closes[i]
                pct = (points / closes[i]) * 100
                future_points[f'future_{j}d_points'] = points
                future_points[f'future_{j}d_pct'] = pct
            else:
                future_points[f'future_{j}d_points'] = np.nan
                future_points[f'future_{j}d_pct'] = np.nan

        matches.append({
            'match_index': i,
            'match_datetime': df.index[i],
            'distance': distance,
            'correlation': correlation,
            'hist_volatility': hist_vol,
            'current_volatility': current_volatility,
            'volatility_diff': abs(current_volatility - hist_vol),
            'price_at_match': closes[i],
            **future_points
        })

    matches_df = pd.DataFrame(matches)
    if matches_df.empty:
        return matches_df

    # Score: prefer small distance, small volatility diff, higher correlation
    matches_df['score'] = (1 / (1 + matches_df['distance'])) * (1 / (1 + matches_df['volatility_diff'])) * (matches_df['correlation'].fillna(0) + 0.0001)
    matches_df = matches_df.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)
    return matches_df

def create_pattern_summary(matches_df, current_price):
    """Create human-readable summary for pattern matches."""
    if matches_df is None or matches_df.empty:
        return "No good historical matches found (not enough data)."

    lines = []
    lines.append("### 📊 Pattern-match summary (plain English)\n")
    lines.append(f"I found **{len(matches_df)}** historical moments that looked similar to what's happening now.\n")

    # when & how many times:
    times = matches_df['match_datetime'].dt.tz_convert('Asia/Kolkata').dt.strftime("%Y-%m-%d %H:%M").tolist()
    lines.append("**Times when similar patterns occurred (IST):**")
    for t in times[:10]:
        lines.append(f"- {t}")
    if len(times) > 10:
        lines.append(f"... and {len(times)-10} more matches.\n")

    # average future movement (points & percent)
    for k in [1, 3, 5, 10]:
        pts_col = f'future_{k}d_points'
        pct_col = f'future_{k}d_pct'
        mean_pts = matches_df[pts_col].mean()
        mean_pct = matches_df[pct_col].mean()
        lines.append(f"- On average, after {k} period(s) the market moved **{mean_pts:.2f} points** ({mean_pct:+.2f}%).")

    # How many times moved up vs down in 1 period
    pos = (matches_df['future_1d_pct'] > 0).sum()
    neg = (matches_df['future_1d_pct'] < 0).sum()
    lines.append(f"\n**Of the {len(matches_df)} matches:** {pos} times the price went up next period, {neg} times it went down.")
    lines.append(f"**Current price:** ₹{current_price:.2f}\n")

    # simple recommendation
    avg_1d_pct = matches_df['future_1d_pct'].mean()
    bullish_prob = pos / len(matches_df) * 100
    lines.append(f"**Average next-period percent change:** {avg_1d_pct:+.2f}%")
    lines.append(f"**Historical chance of an up-move (next period):** {bullish_prob:.1f}%")

    if avg_1d_pct > 1.5 and bullish_prob > 65:
        lines.append("\n✅ **Recommendation (simple):** Historically this pattern led to an upward move — consider a buy/long bias (but use risk limits).")
    elif avg_1d_pct < -1.5 and bullish_prob < 35:
        lines.append("\n⚠️ **Recommendation (simple):** Historically this pattern led to a downward move — consider reduce/short bias (but use risk limits).")
    else:
        lines.append("\nℹ️ **Recommendation (simple):** History is mixed — hold or wait for clearer signals.")

    # tabulated small summary of matches
    small = matches_df[['match_datetime', 'price_at_match', 'future_1d_points', 'future_1d_pct']].copy()
    small['match_datetime'] = small['match_datetime'].dt.tz_convert('Asia/Kolkata').dt.strftime("%Y-%m-%d %H:%M")
    small = small.rename(columns={
        'match_datetime': 'Time (IST)',
        'price_at_match': 'Price at match',
        'future_1d_points': 'Next 1 period (pts)',
        'future_1d_pct': 'Next 1 period (%)'
    }).round(3)
    return "\n".join(lines), small

def analyze_returns_and_volatility(df):
    df2 = df.copy()
    df2['Returns'] = df2['Close'].pct_change() * 100
    df2['Volatility'] = df2['Returns'].rolling(window=20, min_periods=1).std()
    total_return = ((df2['Close'].iloc[-1] - df2['Close'].iloc[0]) / df2['Close'].iloc[0]) * 100 if len(df2) > 1 else 0
    avg_daily_return = df2['Returns'].mean()
    volatility = df2['Returns'].std()
    max_gain = df2['Returns'].max()
    max_loss = df2['Returns'].min()
    return {
        'total_return': total_return,
        'avg_daily_return': avg_daily_return,
        'volatility': volatility,
        'max_gain': max_gain,
        'max_loss': max_loss
    }, df2

def create_ratio_analysis(df1, df2, ticker1, ticker2):
    """
    Build ratio_df and bin analysis table. Also create human-friendly explanation.
    """
    # align
    common = df1.index.intersection(df2.index)
    df1a = df1.loc[common].copy()
    df2a = df2.loc[common].copy()
    if df1a.empty or df2a.empty:
        return "Not enough overlapping data for ratio analysis.", pd.DataFrame(), pd.DataFrame()

    ratio = df1a['Close'] / df2a['Close']
    ratio_df = pd.DataFrame({
        'Ticker1_Close': df1a['Close'],
        'Ticker2_Close': df2a['Close'],
        'Ratio': ratio
    }, index=common)
    ratio_df['Returns'] = ratio_df['Ratio'].pct_change() * 100
    ratio_df['Volatility'] = ratio_df['Returns'].rolling(window=20, min_periods=1).std()

    # ratio bins
    ratio_df = ratio_df.dropna()
    try:
        ratio_df['Ratio_Bin'] = pd.qcut(ratio_df['Ratio'], q=6, labels=False, duplicates='drop')
    except Exception:
        ratio_df['Ratio_Bin'] = pd.cut(ratio_df['Ratio'], bins=6, labels=False)

    # analyze each bin: how often next 3 periods rally (positive returns)
    bin_stats = []
    for b in sorted(ratio_df['Ratio_Bin'].unique()):
        sub = ratio_df[ratio_df['Ratio_Bin'] == b]
        # compute next-3-period percent on ratio as example
        next3 = sub['Ratio'].shift(-3)
        pct_next3 = (next3 - sub['Ratio']) / sub['Ratio'] * 100
        # drop NA pairs
        valid = pct_next3.dropna()
        if valid.empty:
            continue
        avg_next3 = valid.mean()
        up_rate = (valid > 0).sum() / len(valid) * 100
        avg_vol = sub['Volatility'].mean()
        ratio_min, ratio_max = sub['Ratio'].min(), sub['Ratio'].max()
        sample_size = len(sub)
        # record an example row with high volatility info
        # find highest volatility row within this bin
        high_vol_row = sub['Volatility'].idxmax()
        bin_stats.append({
            'Ratio_Bin': int(b),
            'Ratio_range': f"{ratio_min:.4f} - {ratio_max:.4f}",
            'Avg_next3_pct': avg_next3,
            'Up_rate_next3_pct': up_rate,
            'Avg_volatility': avg_vol,
            'Sample_size': sample_size,
            'High_vol_time': high_vol_row,
            'High_vol_value': sub.loc[high_vol_row, 'Volatility'],
            'Ticker1_at_high_vol': sub.loc[high_vol_row, 'Ticker1_Close'],
            'Ticker2_at_high_vol': sub.loc[high_vol_row, 'Ticker2_Close']
        })

    bin_stats_df = pd.DataFrame(bin_stats).sort_values('Up_rate_next3_pct', ascending=False).reset_index(drop=True)

    # explanation text
    expl = []
    expl.append(f"### 🔄 Ratio Analysis: {ticker1} / {ticker2}\n")
    expl.append(f"- Data points analyzed: {len(ratio_df)}")
    expl.append(f"- Current ratio: {ratio_df['Ratio'].iloc[-1]:.4f}")
    expl.append("\nWhat we check: we grouped historical ratio levels into bins and measured *how often* the ratio went up (rallied) in the next 3 periods from that ratio level, and what typical volatility looked like.\n")

    if not bin_stats_df.empty:
        best = bin_stats_df.iloc[0]
        expl.append(f"- Top bin (by historical up-rate) is bin {best['Ratio_Bin']} covering ratios {best['Ratio_range']}. In this bin the ratio rose in the next 3 periods {best['Up_rate_next3_pct']:.1f}% of the time (sample size {int(best['Sample_size'])}).")
    else:
        expl.append("- Not enough data to identify ratio bin behavior.")

    return "\n".join(expl), ratio_df, bin_stats_df

def detect_divergences(df):
    """
    Simple divergence detection:
    - bullish divergence: price makes lower low but RSI makes higher low
    - bearish divergence: price makes higher high but RSI makes lower high
    Returns list of annotations to draw and a friendly summary.
    """
    res = []
    summary = []
    if df is None or len(df) < 30:
        return res, "Not enough data to check divergences."

    df2 = df.copy()
    df2['RSI'] = calculate_rsi(df2['Close'])
    # find local peaks/troughs using simple window
    window = 5
    highs = (df2['Close'][(df2['Close'].shift(window) < df2['Close']) & (df2['Close'].shift(-window) < df2['Close'])])
    lows = (df2['Close'][(df2['Close'].shift(window) > df2['Close']) & (df2['Close'].shift(-window) > df2['Close'])])

    # use last two lows for bullish divergence
    low_idx = lows.index
    if len(low_idx) >= 2:
        last_two = low_idx[-2:]
        p1, p2 = df2.loc[last_two[0], 'Close'], df2.loc[last_two[1], 'Close']
        r1, r2 = df2.loc[last_two[0], 'RSI'], df2.loc[last_two[1], 'RSI']
        # price lower low & rsi higher low -> bullish divergence
        if p2 < p1 and r2 > r1:
            res.append({
                'type': 'bullish',
                'p1_idx': last_two[0],
                'p2_idx': last_two[1],
                'p1_price': p1,
                'p2_price': p2,
                'p1_rsi': r1,
                'p2_rsi': r2
            })
            summary.append(f"✅ Bullish divergence found between {last_two[0]} and {last_two[1]} (price lower-low but RSI higher-low) — often a reversal signal to the upside.")
    # last two highs for bearish divergence
    high_idx = highs.index
    if len(high_idx) >= 2:
        last_two_h = high_idx[-2:]
        p1, p2 = df2.loc[last_two_h[0], 'Close'], df2.loc[last_two_h[1], 'Close']
        r1, r2 = df2.loc[last_two_h[0], 'RSI'], df2.loc[last_two_h[1], 'RSI']
        # price higher high & rsi lower high -> bearish divergence
        if p2 > p1 and r2 < r1:
            res.append({
                'type': 'bearish',
                'p1_idx': last_two_h[0],
                'p2_idx': last_two_h[1],
                'p1_price': p1,
                'p2_price': p2,
                'p1_rsi': r1,
                'p2_rsi': r2
            })
            summary.append(f"⚠️ Bearish divergence found between {last_two_h[0]} and {last_two_h[1]} (price higher-high but RSI lower-high) — often a correction-following signal.")
    if not summary:
        summary_text = "No clear bullish/bearish divergences detected in the recent data."
    else:
        summary_text = "\n".join(summary)
    return res, summary_text

def create_comprehensive_figure(df, ticker_name, matches_df=None, show_divergences=None):
    """Create a Plotly figure with candlesticks, volume (bottom), RSI, and returns."""
    dfp = df.copy()
    dfp['RSI'] = calculate_rsi(dfp['Close'])
    returns = dfp['Close'].pct_change() * 100

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.45, 0.12, 0.15, 0.18],
                        subplot_titles=(f'{ticker_name} - Candles', 'Volume', 'RSI (14)', 'Returns %'))

    # Candlestick
    fig.add_trace(go.Candlestick(x=dfp.index, open=dfp['Open'], high=dfp['High'], low=dfp['Low'], close=dfp['Close'],
                                 name='Price',
                                 increasing_line_color='#00ff00',
                                 decreasing_line_color='#ff0000'), row=1, col=1)

    # Mark pattern matches
    if matches_df is not None and not matches_df.empty:
        for _, match in matches_df.head(6).iterrows():
            match_dt = match['match_datetime']
            if match_dt in dfp.index:
                fig.add_trace(go.Scatter(x=[match_dt], y=[dfp.loc[match_dt, 'High'] * 1.01],
                                         text=[f"Similar ({match['future_1d_pct']:+.2f}%)"],
                                         mode='markers+text',
                                         marker=dict(size=10, color='cyan'),
                                         textposition='top center',
                                         showlegend=False), row=1, col=1)

    # Volume (colored)
    colors = ['#00ff00' if r['Close'] >= r['Open'] else '#ff0000' for _, r in dfp.iterrows()]
    fig.add_trace(go.Bar(x=dfp.index, y=dfp['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp['RSI'], name='RSI', line=dict(width=2, dash='solid')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    # Returns
    fig.add_trace(go.Scatter(x=dfp.index, y=returns, name='Returns %', fill='tozeroy', line=dict(width=1)), row=4, col=1)

    # Divergence annotations (if provided)
    if show_divergences:
        for d in show_divergences:
            if d['type'] == 'bullish':
                fig.add_annotation(x=d['p2_idx'], y=dfp.loc[d['p2_idx'], 'Low'],
                                   text="Bullish Divergence",
                                   showarrow=True, arrowhead=2, ax=0, ay=30,
                                   bgcolor="#1f772c", font=dict(color="white"))
            elif d['type'] == 'bearish':
                fig.add_annotation(x=d['p2_idx'], y=dfp.loc[d['p2_idx'], 'High'],
                                   text="Bearish Divergence",
                                   showarrow=True, arrowhead=2, ax=0, ay=-30,
                                   bgcolor="#8b1f1f", font=dict(color="white"))

    fig.update_layout(height=1100, hovermode='x unified', template='plotly_dark', showlegend=True)
    fig.update_xaxes(rangeslider_visible=False)

    return fig

# ---------------------------
# UI: main app
# ---------------------------
st.markdown('<p class="main-header">📊 Advanced Trading Pattern Analyzer — Human-friendly</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_choice = st.selectbox("Select Asset", list(TICKER_MAP.keys()))
    if ticker_choice == "Custom":
        custom_ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA)")
        ticker_symbol = custom_ticker.strip().upper() if custom_ticker else ""
    else:
        ticker_symbol = TICKER_MAP[ticker_choice]

    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Timeframe", ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'], index=7)
    with col2:
        period = st.selectbox("Period", ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'], index=5)

    fetch_button = st.button("🔄 Fetch Data & Analyze", type="primary")
    st.divider()
    st.subheader("🔄 Ratio Analysis")
    enable_ratio = st.checkbox("Enable Ratio Analysis")
    if enable_ratio:
        ticker2_choice = st.selectbox("Select Second Asset", list(TICKER_MAP.keys()), key='ticker2')
        if ticker2_choice == "Custom":
            custom_ticker2 = st.text_input("Enter Second Ticker", key='custom2')
            ticker2_symbol = custom_ticker2.strip().upper() if custom_ticker2 else ""
        else:
            ticker2_symbol = TICKER_MAP[ticker2_choice]
    st.divider()
    st.info("💡 This tool uses simple pattern-matching + ratio and RSI checks. It's for research and idea generation — not financial advice.")

# Main area
if fetch_button:
    if not ticker_symbol:
        st.error("Please enter a valid ticker symbol!")
    else:
        with st.spinner(f"Fetching {ticker_symbol}..."):
            try:
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period, interval=interval)
                if df.empty:
                    st.error("No data returned. Please check the ticker symbol and try again.")
                else:
                    df = convert_to_ist(df)
                    st.session_state.df = df
                    st.session_state.ticker_symbol = ticker_symbol
                    st.session_state.data_fetched = True
                    st.success(f"✅ Data fetched successfully! {len(df)} rows retrieved.")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df.copy()
    ticker_symbol = st.session_state.ticker_symbol

    # small top metrics
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    points_change = last['Close'] - prev['Close']
    pct_change = (points_change / prev['Close']) * 100 if prev['Close'] != 0 else 0
    col1, col2, col3 = st.columns([1.2, 1, 1])
    col1.markdown(f"**Ticker:** {ticker_symbol}")
    col2.metric("Last Close", f"₹{last['Close']:.2f}", delta=f"{pct_change:+.2f}%")
    col3.markdown(f"**Volume:** {int(last['Volume']) if not np.isnan(last['Volume']) else 'N/A'}")

    # 1) Pattern matching
    st.subheader("🔎 Similar Pattern Search (how many times, when, and outcomes)")
    window_size = st.slider("Pattern window size (how many periods to compare)", min_value=5, max_value=40, value=10)
    matches_df = find_pattern_matches(df, window_size=window_size, top_n=30)
    if matches_df.empty:
        st.info("Not enough history or no patterns found.")
    else:
        # pattern summary text + table
        pattern_summary_text, pattern_small_table = create_pattern_summary(matches_df, last['Close'])
        if isinstance(pattern_summary_text, tuple):
            # older returned shape - adjust
            pattern_summary_text = pattern_summary_text[0]
        st.markdown(pattern_summary_text)
        st.dataframe(pattern_small_table.style.format({"Next 1 period (%)": "{:+.2f}"}), height=300)

    # 2) Explanations and chart
    st.subheader("📈 Charts & Visuals")
    divergences, divergence_summary = detect_divergences(df)
    st.markdown("**Divergence check (plain English):**")
    st.write(divergence_summary)

    fig = create_comprehensive_figure(df, ticker_symbol, matches_df=matches_df, show_divergences=divergences)
    st.plotly_chart(fig, use_container_width=True)

    # 3) Price table with changes
    st.subheader("📋 Recent Price Table (points and percent vs previous point)")
    recent_n = st.number_input("Show last N rows", min_value=5, max_value=200, value=20, step=1)
    price_tbl = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    price_tbl['Points_change'] = price_tbl['Close'].diff().round(3)
    price_tbl['Pct_change'] = (price_tbl['Close'].pct_change() * 100).round(3)
    price_tbl_display = price_tbl.tail(int(recent_n)).copy()
    price_tbl_display.index = price_tbl_display.index.tz_convert('Asia/Kolkata')
    price_tbl_display = price_tbl_display.reset_index().rename(columns={'index': 'DateTime(IST)'})
    st.dataframe(price_tbl_display.style.format({
        'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
        'Points_change': '{:+.2f}', 'Pct_change': '{:+.3f}%', 'Volume': '{:.0f}'
    }), height=300)

    # 4) Volatility table: where volatility was high and price at that time
    st.subheader("⚡ Volatility table (higher volatility moments)")
    returns_vol = df['Close'].pct_change() * 100
    vol_series = returns_vol.rolling(window=20, min_periods=1).std()
    vol_df = pd.DataFrame({
        'Close': df['Close'],
        'Volatility': vol_series
    })
    # show top volatility moments
    top_vol = vol_df.sort_values('Volatility', ascending=False).head(20).copy()
    top_vol.index = top_vol.index.tz_convert('Asia/Kolkata')
    top_vol = top_vol.reset_index().rename(columns={'index': 'DateTime(IST)'})
    st.dataframe(top_vol.style.format({'Close': '{:.2f}', 'Volatility': '{:.4f}'}), height=300)
    st.markdown("**Plain English:** The table above shows times when the price moved more wildly (higher short-term volatility). Check these times to see whether large moves favoured buyers or sellers.")

    # 5) Ratio analysis (if enabled)
    if enable_ratio:
        st.subheader("🔁 Ratio analysis and bins")
        try:
            ticker2_symbol  # may not be defined
        except NameError:
            ticker2_symbol = None

        if not ticker2_symbol:
            st.error("Please select/enter the second ticker for ratio analysis.")
        else:
            with st.spinner("Fetching second ticker data..."):
                t2 = yf.Ticker(ticker2_symbol)
                df2 = t2.history(period=period, interval=interval)
                if df2.empty:
                    st.error("No data for second ticker.")
                else:
                    df2 = convert_to_ist(df2)
                    expl_text, ratio_df, bin_stats_df = create_ratio_analysis(df, df2, ticker_symbol, ticker2_symbol)
                    st.markdown(expl_text)
                    if not ratio_df.empty:
                        # store ratio_df for potential download
                        st.session_state.ratio_df = ratio_df
                        # show ratio plot: ticker1, ticker2 and ratio on same x axis
                        ratio_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                                 row_heights=[0.33, 0.33, 0.34],
                                                 subplot_titles=(f'{ticker_symbol} Price', f'{ticker2_symbol} Price', f'Ratio: {ticker_symbol}/{ticker2_symbol}'))
                        ratio_fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ticker1_Close'], name=ticker_symbol), row=1, col=1)
                        ratio_fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ticker2_Close'], name=ticker2_symbol), row=2, col=1)
                        ratio_fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ratio'], name='Ratio', fill='tozeroy'), row=3, col=1)
                        ratio_fig.update_layout(height=900, hovermode='x unified', template='plotly_dark')
                        st.plotly_chart(ratio_fig, use_container_width=True)

                        # show bin_stats table
                        if not bin_stats_df.empty:
                            st.markdown("**Ratio bins summary (which ratio levels historically rallied the most in next 3 periods):**")
                            display_bin = bin_stats_df.copy()
                            # format the high_vol_time and others
                            display_bin['High_vol_time'] = display_bin['High_vol_time'].dt.tz_convert('Asia/Kolkata')
                            st.dataframe(display_bin[['Ratio_Bin', 'Ratio_range', 'Avg_next3_pct', 'Up_rate_next3_pct', 'Avg_volatility', 'Sample_size', 'High_vol_time', 'Ticker1_at_high_vol', 'Ticker2_at_high_vol']].style.format({
                                'Avg_next3_pct': '{:+.3f}%', 'Up_rate_next3_pct': '{:.1f}%', 'Avg_volatility': '{:.4f}', 'Ticker1_at_high_vol': '{:.2f}', 'Ticker2_at_high_vol': '{:.2f}'
                            }), height=300)
                        else:
                            st.info("Not enough data to build ratio bins.")

    # 6) Export (only OHLCV columns)
    st.subheader("📥 Export data (only Open, High, Low, Close, Volume will be exported)")
    export_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    export_df.index = export_df.index.tz_convert('Asia/Kolkata')
    export_choice = st.radio("Export format", options=["CSV", "Excel"], horizontal=True)

    if export_choice == "CSV":
        csv_buf = export_df.to_csv(index=True).encode('utf-8')
        st.download_button("Download CSV (OHLCV)", data=csv_buf, file_name=f"{ticker_symbol.replace('^','')}_ohlcv.csv", mime="text/csv")
    else:
        # excel
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='OHLCV')
            writer.save()
        towrite.seek(0)
        st.download_button("Download Excel (OHLCV)", data=towrite, file_name=f"{ticker_symbol.replace('^','')}_ohlcv.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # 7) Short human-friendly summaries of charts & signals
    st.subheader("🗒️ Plain-English Summary (what the charts are telling you)")
    summary_lines = []
    # pattern-based summary (reuse earlier)
    if matches_df is not None and not matches_df.empty:
        pos = (matches_df['future_1d_pct'] > 0).sum()
        neg = (matches_df['future_1d_pct'] < 0).sum()
        avg1 = matches_df['future_1d_pct'].mean()
        summary_lines.append(f"- Pattern check: Found {len(matches_df)} historical matches. Historically, {pos} times it went up next period and {neg} times it went down. Average next-period change: {avg1:+.2f}%.")
    else:
        summary_lines.append("- Pattern check: Not enough data for pattern matching.")

    # divergence summary
    summary_lines.append(f"- Divergence check: {divergence_summary}")

    # volatility quick comment
    recent_vol = vol_series.iloc[-1]
    if recent_vol > vol_series.mean() * 1.2:
        summary_lines.append(f"- Volatility: Recent volatility ({recent_vol:.4f}) is higher than typical — expect choppiness and larger moves.")
    else:
        summary_lines.append(f"- Volatility: Recent volatility ({recent_vol:.4f}) is within normal range.")

    # ratio quick note
    if enable_ratio and 'bin_stats_df' in locals() and not bin_stats_df.empty:
        top_bin = bin_stats_df.iloc[0]
        summary_lines.append(f"- Ratio: Top historical ratio bin {top_bin['Ratio_Bin']} showed an up-rate of {top_bin['Up_rate_next3_pct']:.1f}% over next 3 periods (sample {int(top_bin['Sample_size'])}).")

    # final
    for ln in summary_lines:
        st.write(ln)

    st.success("Analysis complete — remember this is research only, not financial advice.")

else:
    st.info("Pick an asset from the sidebar and click 'Fetch Data & Analyze' to start.")
