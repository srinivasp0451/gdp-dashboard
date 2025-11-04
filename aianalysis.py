# streamlit_ai_trader.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
import time
from datetime import datetime, timedelta
from functools import partial

# Optional: find_peaks for support/resistance detection
from scipy.signal import find_peaks

# ---------------------------
# ------------- CONFIG -------
# ---------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
MODEL = st.secrets.get("LLAMA_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"  # groq endpoint

# Acceptable intervals list (yfinance supported)
ALLOWED_INTERVALS = [
    "1m","3m","5m","10m","15m","30m","60m","120m","240m","1d","5d","7d",
    "1mo","3mo","6mo","1y","2y","3y","5y","6y","10y","15y","20y","25y","30y"
]
# Map friendly interval names to yfinance accepted format for hourly intervals like '1h' vs '60m'
INTERVAL_MAP = {
    "1h":"60m", "2h":"120m", "4h":"240m", "30min":"30m"
}
# ---------------------------
# ------------- HELPERS ------
# ---------------------------

def to_ist_index(df):
    # converts index (DatetimeIndex) to IST timezone gracefully
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df

def safe_interval(interval):
    # allow synonyms in UI map to yfinance style
    return INTERVAL_MAP.get(interval, interval)

def retry_request(func, retries=3, backoff=1.0):
    # simple retry wrapper for network calls
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise

# -------------- Technical indicator calculations (pandas only) --------------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_prev_close = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (df['Volume'] * direction).cumsum()

# ---------------------------
# -------------- SR / Zones / Vol profile --------------
# ---------------------------
def find_support_resistance(close_series, distance=3, prominence=0.5):
    # use scipy find_peaks on reversed series for troughs (support)
    close = close_series.values
    # peaks -> resistance
    peaks, _ = find_peaks(close, distance=distance, prominence=prominence)
    # troughs -> support (peaks on -close)
    troughs, _ = find_peaks(-close, distance=distance, prominence=prominence)
    resistances = close[peaks]
    supports = close[troughs]
    # Return sorted unique levels
    return sorted(set(np.round(supports, 2))), sorted(set(np.round(resistances, 2)))

def high_volume_zones(df, bins=5):
    # naive high-volume price zones using quantiles
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return []
    vol_by_price = df.groupby(pd.cut(df['Close'], bins=bins))['Volume'].sum().sort_values(ascending=False)
    zones = []
    for interval in vol_by_price.index[:min(3, len(vol_by_price))]:
        zones.append((interval.left, interval.right))
    return zones

def fibonacci_levels(high, low):
    # return dictionary of fib levels
    diff = high - low
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100%': low
    }
    return levels

# -------------- Chart pattern heuristics --------------
def detect_flag(df):
    # flag: sharp pole then small consolidation
    # heuristic: recent big move followed by small slope consolidation
    close = df['Close']
    window = min(30, len(close))
    if window < 10:
        return False
    recent = close[-window:]
    # pole = big percent move within the first 1/3 of window
    third = window // 3
    pole_move = (recent.iloc[third] - recent.iloc[0]) / recent.iloc[0]
    consolidation = recent.iloc[third:].pct_change().abs().mean()
    return abs(pole_move) > 0.03 and consolidation < 0.01

def detect_inside_bars(df):
    # inside bar: candle fully within previous candle
    if len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return (last['High'] <= prev['High']) and (last['Low'] >= prev['Low'])

def detect_triangle(df):
    # detect ascending/descending triangle via slope of highs and lows
    if len(df) < 20:
        return None
    highs = df['High'][-20:]
    lows = df['Low'][-20:]
    # linear fit
    x = np.arange(len(highs))
    high_coef = np.polyfit(x, highs.values, 1)[0]
    low_coef = np.polyfit(x, lows.values, 1)[0]
    # ascending triangle: lows slope up (low_coef>0), highs roughly flat (~0)
    if low_coef > 0.000 and abs(high_coef) < 0.0005:
        return "Ascending Triangle"
    if high_coef < 0.000 and abs(low_coef) < 0.0005:
        return "Descending Triangle"
    return None

def detect_cup_handle(df):
    # crude heuristic: look for long U-shape then small consolidation to right
    close = df['Close']
    n = min(60, len(close))
    if n < 40:
        return False
    s = close[-n:]
    mid = n // 2
    left_mean = s[:mid].mean()
    right_mean = s[mid+int(mid*0.2):].mean() if mid+int(mid*0.2) < n else s[mid:].mean()
    valley = s[mid-3:mid+3].min()
    # left and right should be higher than valley
    return (left_mean - valley) > (0.02 * valley) and (right_mean - valley) > (0.02 * valley)

# ---------------------------
# --------------- LLaMA (Groq) call for final polish -------------------
# ---------------------------
def call_llama_groq(system_prompt: str, user_prompt: str, max_tokens=600, temp=0.2):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        st.error(f"LLaMA/Groq API error {resp.status_code}: {resp.text}")
        return None
    data = resp.json()
    # robustly parse content
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message") or {}
    return msg.get("content") or choice.get("text") or str(choice)

# ---------------------------
# --------------- CACHING / FETCH CONTROL ---------------------
# ---------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes by default
def fetch_data_cached(ticker, period, interval):
    # yfinance has some quirks with intraday history length; user controls period.
    # Use retry wrapper to avoid transient network errors.
    def fetch():
        return yf.download(ticker, period=period, interval=interval, progress=False)
    df = retry_request(fetch, retries=3, backoff=1.0)
    if df is None or df.empty:
        return pd.DataFrame()
    return df

# ---------------------------
# ---------------- STREAMLIT UI -------------------------
# ---------------------------
st.set_page_config(layout="wide", page_title="AI Market Analyst (LLaMA)")

st.title("AI Market Analyst — Professional Grade (LLaMA)")
st.markdown("Select ticker, timeframe and press **Fetch & Run**. Data is fetched only on button click and cached to avoid rate limits.")

# Left panel: inputs
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker (yfinance)", value="^NSEI")
    period = st.selectbox("Period", options=[
        "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","15y","20y","25y","30y"
    ], index=1)
    interval = st.selectbox("Interval", options=ALLOWED_INTERVALS, index=4)
    rows_to_analyze = st.number_input("Rows to analyze (last N candles)", min_value=5, max_value=500, value=60)
    show_chart = st.checkbox("Show interactive chart", value=True)
    api_key_input = st.text_input("Groq API key", value=GROQ_API_KEY)
    model_input = st.text_input("LLaMA model id", value=MODEL)
    st.markdown("---")
    st.write("⚠️ Notes:")
    st.write("- Data is fetched from Yahoo Finance only when you click **Fetch & Run**.")
    st.write("- Cached for 5 minutes to reduce rate limit problems.")
    run_btn = st.button("Fetch & Run")

# persist keys if user provided them
if api_key_input:
    GROQ_API_KEY = api_key_input
if model_input:
    MODEL = model_input

# maintain session state for last fetched data
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# On button click fetch data (and cache) — prevents fetch on every rerun
if run_btn:
    st.session_state.last_fetch = {"ticker": ticker, "period": period, "interval": interval, "time": datetime.now().isoformat()}
    safe_int = safe_interval(interval)
    with st.spinner(f"Fetching {ticker} {period} @ {safe_int} ..."):
        try:
            df = fetch_data_cached(ticker, period, safe_int)
            if df.empty:
                st.error("No data returned from yfinance. Try different period/interval.")
            else:
                df = to_ist_index(df)
                df = df.reset_index().rename(columns={"index": "Datetime_IST", "Datetime": "Datetime_IST"})
                st.session_state.df = df
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# If we have cached df, analyze
df = st.session_state.df
if df is None or df.empty:
    st.info("No data loaded yet. Click 'Fetch & Run' to fetch data from yfinance.")
    st.stop()

# Prepare analysis window
n = int(min(rows_to_analyze, len(df)))
recent_df = df.tail(n).copy()

# compute indicators
recent_df['EMA20'] = ema(recent_df['Close'], 20)
recent_df['EMA50'] = ema(recent_df['Close'], 50)
recent_df['SMA200'] = sma(recent_df['Close'], 200)
recent_df['RSI14'] = rsi(recent_df['Close'], 14)
recent_df['ATR14'] = atr(recent_df, 14)
recent_df['OBV'] = obv(recent_df)

# support/resistance and zones
supports, resistances = find_support_resistance(recent_df['Close'])
hv_zones = high_volume_zones(recent_df)
fib_high = recent_df['High'].max()
fib_low = recent_df['Low'].min()
fib_levels = fibonacci_levels(fib_high, fib_low)

# pattern detections
patterns = []
if detect_flag(recent_df):
    patterns.append("Flag/Pennant")
if detect_inside_bars(recent_df):
    patterns.append("Inside Bar")
tri = detect_triangle(recent_df)
if tri:
    patterns.append(tri)
if detect_cup_handle(recent_df):
    patterns.append("Cup & Handle (heuristic)")

# compose a structured analytic payload for LLaMA
analysis_payload = {
    "ticker": ticker,
    "period": period,
    "interval": interval,
    "last_time_ist": str(recent_df['Datetime_IST'].iloc[-1]),
    "close": float(recent_df['Close'].iloc[-1]),
    "open": float(recent_df['Open'].iloc[-1]),
    "high": float(recent_df['High'].iloc[-1]),
    "low": float(recent_df['Low'].iloc[-1]),
    "volume": int(recent_df['Volume'].iloc[-1]) if 'Volume' in recent_df.columns else None,
    "rsi": float(recent_df['RSI14'].iloc[-1]) if not np.isnan(recent_df['RSI14'].iloc[-1]) else None,
    "ema20": float(recent_df['EMA20'].iloc[-1]),
    "ema50": float(recent_df['EMA50'].iloc[-1]),
    "sma200": float(recent_df['SMA200'].iloc[-1]) if 'SMA200' in recent_df.columns else None,
    "atr": float(recent_df['ATR14'].iloc[-1]),
    "obv": float(recent_df['OBV'].iloc[-1]),
    "supports": supports,
    "resistances": resistances,
    "high_volume_zones": hv_zones,
    "fibonacci": fib_levels,
    "patterns": patterns
}

# Show raw technical snapshot on UI
with st.expander("Technical Snapshot (raw values)"):
    st.json(analysis_payload)

# show chart if requested (simple)
if show_chart:
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(
        x=recent_df['Datetime_IST'],
        open=recent_df['Open'],
        high=recent_df['High'],
        low=recent_df['Low'],
        close=recent_df['Close'],
        name=ticker
    )])
    # overlay EMA20 and EMA50
    fig.add_trace(go.Scatter(x=recent_df['Datetime_IST'], y=recent_df['EMA20'], name="EMA20"))
    fig.add_trace(go.Scatter(x=recent_df['Datetime_IST'], y=recent_df['EMA50'], name="EMA50"))
    # show support/resistance lines
    for s in supports:
        fig.add_hline(y=s, line=dict(color="green", width=1), annotation_text=f"Support {s}", annotation_position="bottom right")
    for r in resistances:
        fig.add_hline(y=r, line=dict(color="red", width=1), annotation_text=f"Res {r}", annotation_position="top right")
    # show fib levels
    for name, level in fib_levels.items():
        fig.add_hline(y=level, line=dict(color="rgba(128,128,128,0.3)"), annotation_text=f"Fib {name}:{level:.2f}", annotation_position="top left")
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Compose the prompt for the LLaMA model
system_prompt = """
You are a professional market analyst for institutional traders. Produce a crisp, professional-grade analysis suitable for a trading desk.
Output must include:
- A one-line Label: BULLISH / BEARISH / NEUTRAL
- A short bullet list (3 bullets) explaining reasoning and strength/confidence
- Key actionable levels: 2 support levels, 2 resistance levels, and suggested stop-loss and target ranges if recommending trade
- Identify major zones: high-volume zones, demand/supply zones (if any), trap zones
- List detected chart patterns (if any) and your confidence in them
- Keep language concise, use no jargon, and include explicit IST timestamp of last candle

Return only plain text (no JSON).
"""

# build pretty user prompt with extracted metrics
user_prompt_lines = [
    f"Ticker: {analysis_payload['ticker']}",
    f"Interval: {analysis_payload['interval']}  |  Last candle IST: {analysis_payload['last_time_ist']}",
    f"Close: {analysis_payload['close']:.2f}  Open: {analysis_payload['open']:.2f}  High: {analysis_payload['high']:.2f}  Low: {analysis_payload['low']:.2f}",
    f"RSI(14): {analysis_payload['rsi']:.2f}  EMA20: {analysis_payload['ema20']:.2f}  EMA50: {analysis_payload['ema50']:.2f}  ATR: {analysis_payload['atr']:.2f}",
    f"OBV (recent): {analysis_payload['obv']:.2f}",
    "",
    "Support levels (auto): " + (", ".join(map(str, analysis_payload['supports'])) if analysis_payload['supports'] else "None"),
    "Resistance levels (auto): " + (", ".join(map(str, analysis_payload['resistances'])) if analysis_payload['resistances'] else "None"),
    "High volume zones: " + (", ".join([f'{a:.2f}-{b:.2f}' for (a,b) in analysis_payload['high_volume_zones']]) if analysis_payload['high_volume_zones'] else "None"),
    "Fibonacci levels: " + ", ".join([f"{k}:{v:.2f}" for k,v in analysis_payload['fibonacci'].items()]),
    "Detected patterns: " + (", ".join(analysis_payload['patterns']) if analysis_payload['patterns'] else "None"),
    "",
    "Important: Provide a crisp actionable recommendation: BUY / SELL / HOLD, with clear entry, stop loss, target (if BUY/SELL), and confidence level (Low/Medium/High). Keep explanation short and professional."
]
user_prompt = "\n".join(user_prompt_lines)

# Call LLaMA to create final professional-grade analysis
with st.spinner("Running AI analysis (LLaMA) — producing professional report..."):
    llm_result = call_llama_groq(system_prompt, user_prompt, max_tokens=450, temp=0.15)

if llm_result:
    st.header("AI Professional Summary (LLaMA)")
    st.markdown(llm_result.replace("\n", "\n\n"))
else:
    st.error("Failed to get result from LLaMA model. Check API key, model id and endpoint.")

st.markdown("---")
st.caption("Design notes: data is fetched only when you click 'Fetch & Run'. Results cached 5 minutes to reduce yfinance rate-limit issues.")
