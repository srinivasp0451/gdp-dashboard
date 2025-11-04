# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from functools import lru_cache

# -------------------------
# ========== CONFIG ========
# -------------------------
# Replace with your Groq / LLaMA4 provider values
GROQ_API_KEY = "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"  # <-- set your real key here
LLAMA_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Supported intervals and periods (includes your long list)
ALLOWED_INTERVALS = [
    "1m","3m","5m","10m","15m","30m","60m","120m","240m",
    "1d","5d","7d","1mo","3mo","6mo","1y","2y","3y","5y","6y","10y","15y","20y","25y","30y"
]
# Convert some user-friendly names to yfinance-valid intervals
# yfinance accepts '1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1mo', etc.
# We will attempt to use the numeric versions and map 60m -> '60m', 120m->'120m', 240m->'240m'
# For very long periods yfinance supports period argument like '5y','10y', etc.

# -------------------------
# ========== HELPERS =======
# -------------------------

def call_llama_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.0):
    """Call Groq/OpenAI compatible endpoint for a professional polished output."""
    if not GROQ_API_KEY or GROQ_API_KEY == "abcd":
        return "[LLM output disabled — set GROQ_API_KEY to enable LLaMA summaries]"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        # support either {message: {content: ...}} or {text: ...}
        if isinstance(choice.get("message"), dict):
            return choice["message"]["content"]
        return choice.get("text") or str(choice)
    except Exception as e:
        return f"[LLM error: {e} | {resp.text if 'resp' in locals() else ''}]"

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def pivot_points(high, low, close):
    """Basic pivot point (classic) and S/R"""
    P = (high + low + close) / 3
    R1 = (2 * P) - low
    S1 = (2 * P) - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

def find_local_extrema(series: pd.Series, order=3):
    """Find local minima and maxima indexes (simple)"""
    from scipy.signal import argrelextrema
    arr = series.values
    if len(arr) < order*2+1:
        return [], []
    max_idx = argrelextrema(arr, np.greater_equal, order=order)[0].tolist()
    min_idx = argrelextrema(arr, np.less_equal, order=order)[0].tolist()
    return min_idx, max_idx

def support_resistance_from_levels(series: pd.Series, n_levels=5):
    """Derive S/R by clustering local extrema (simple quantized approach)"""
    # Use quantiles of price as candidate levels
    qs = np.linspace(0.05, 0.95, 19)
    levels = sorted(list({round(np.quantile(series.dropna(), q), 2) for q in qs}))
    # reduce to n_levels by selecting spaced ones
    if len(levels) <= n_levels: 
        return levels
    step = max(1, len(levels) // n_levels)
    return [levels[i] for i in range(0, len(levels), step)][:n_levels]

def detect_inside_bar(df):
    """Detect inside bar on last candle: last candle high < prev high and last low > prev low"""
    if len(df) < 2: 
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return (last['High'] < prev['High']) and (last['Low'] > prev['Low'])

def detect_flag_pole(df):
    """Simple heuristic: large single-candle move (flagpole) followed by consolidation"""
    if len(df) < 6:
        return False
    # large candle threshold
    last5 = df['Close'].iloc[-6:-1]
    last6 = df['Close'].iloc[-6:]
    returns = df['Close'].pct_change().abs()
    large_moves = returns.iloc[-10:-1]
    if len(large_moves) == 0:
        return False
    # find any candle with >2% move in last 10
    if (returns.iloc[-10:-1] > 0.02).any():
        # check subsequent small range candles
        small_range = (df['High'] - df['Low']).iloc[-5:] / df['Close'].iloc[-5:]
        if (small_range < 0.01).sum() >= 3:
            return True
    return False

def detect_ascending_descending_triangle(df):
    """Very simple heuristic for triangle: series of highs trending down and lows trending up"""
    if len(df) < 10:
        return None
    highs = df['High'].rolling(window=5).max().dropna()
    lows = df['Low'].rolling(window=5).min().dropna()
    # estimate slope of last N highs and lows
    N = min(8, len(highs))
    x = np.arange(N)
    y_high = highs.values[-N:]
    y_low = lows.values[-N:]
    # fit linear
    m_high = np.polyfit(x, y_high, 1)[0]
    m_low = np.polyfit(x, y_low, 1)[0]
    if m_high < 0 and m_low > 0:
        return "ascending_triangle"
    if m_high > 0 and m_low < 0:
        return "descending_triangle"
    return None

def fibonacci_levels(df):
    """Compute Fibonacci retracement levels from recent swing high/low in visible window"""
    if len(df) < 2:
        return {}
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    levels = {
        "0.0": round(high,2),
        "0.236": round(high - 0.236*diff,2),
        "0.382": round(high - 0.382*diff,2),
        "0.5": round(high - 0.5*diff,2),
        "0.618": round(high - 0.618*diff,2),
        "1.0": round(low,2),
    }
    return levels

# -------------------------
# ========== CACHING =======
# -------------------------
# Cache yfinance fetches to avoid repeated calls across Streamlit reruns.
# Use st.cache_data for data caching (Streamlit >=1.9). If unavailable, fallback to lru_cache.

try:
    cache_data = st.cache_data  # Streamlit cache decorator
except Exception:
    cache_data = None

if cache_data:
    @cache_data(ttl=60*5, show_spinner=False)  # cache for 5 minutes by default; adjust as needed
    def fetch_data_cached(ticker, period, interval):
        return yf.download(ticker, period=period, interval=interval, progress=False)
else:
    # fallback caching in-memory for the session
    @lru_cache(maxsize=32)
    def fetch_data_cached(ticker, period, interval):
        return yf.download(ticker, period=period, interval=interval, progress=False)

# -------------------------
# ========== STREAMLIT UI ==
# -------------------------
st.set_page_config(layout="wide", page_title="Professional AI Market Analyst")

st.title("Professional AI Market Analyst — Multi-Agent (LLaMA / Groq compatible)")
st.markdown("Fetch intraday / historical data from yfinance only when you click **Fetch & Run**. Timezone converted to IST automatically.")

# Left column: inputs
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Configuration")
    ticker = st.text_input("Ticker (yfinance)", value=st.session_state.get("ticker","^NSEI"))
    period = st.selectbox("Period (yfinance)", options=[
        "1d","5d","1mo","3mo","6mo","1y","2y","3y","5y","10y","20y","30y"], index=1)
    interval = st.selectbox("Interval", options=ALLOWED_INTERVALS, index=4)
    rows_to_analyze = st.number_input("Rows to analyze (recent candles)", value=20, min_value=5, max_value=200, step=1)
    use_llm = st.checkbox("Enable LLaMA professional writeup (Groq API)", value=False)
    if use_llm and (not GROQ_API_KEY or GROQ_API_KEY=="abcd"):
        st.warning("Enable LLM requires setting GROQ_API_KEY variable in the script.")
    # Keep inputs saved in session state
    st.session_state["ticker"] = ticker
    st.session_state["period"] = period
    st.session_state["interval"] = interval
    st.session_state["rows_to_analyze"] = rows_to_analyze

    st.markdown("---")
    if "last_fetch_time" in st.session_state:
        st.info(f"Last fetch: {st.session_state['last_fetch_time']} IST (cached)")

    fetch_btn = st.button("Fetch & Run Analysis")

# Right column: output
with col2:
    st.subheader("Analysis / Output")
    output_area = st.empty()

# Button logic (only fetch when clicked)
if fetch_btn:
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        # Fetch data (cached)
        try:
            raw_df = fetch_data_cached(ticker, period, interval)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            raw_df = pd.DataFrame()

        if raw_df.empty:
            st.error("No data returned. Try a different ticker/period/interval.")
        else:
            # Timezone handling: yfinance returns naive datetimes in UTC; convert to IST
            idx = raw_df.index
            try:
                if idx.tz is None:
                    idx = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
                else:
                    idx = idx.tz_convert("Asia/Kolkata")
            except Exception:
                # fallback: assume UTC
                idx = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
            raw_df.index = idx

            df = raw_df.copy()
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]:"Datetime_IST"}, inplace=True)

            # Compute indicators
            df['EMA20'] = ema(df['Close'], 20)
            df['EMA50'] = ema(df['Close'], 50)
            df['RSI14'] = rsi(df['Close'], 14)
            df['Change%'] = df['Close'].pct_change()*100

            # Basic stats
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest
            change_pct = float(((latest['Close'] - prev['Close'])/prev['Close'])*100) if prev['Close'] != 0 else 0.0

            # Zones and levels
            sr_levels = support_resistance_from_levels(df['Close'], n_levels=6)
            fibs = fibonacci_levels(df.tail(min(len(df), 200)))  # fibonacci on last 200 candles
            high_vol_zone = df.sort_values('Volume', ascending=False).head(max(1,int(len(df)*0.05)))
            avg_vol = float(df['Volume'].tail(50).mean()) if 'Volume' in df.columns else None
            hv_low = float(high_vol_zone['Low'].min()) if not high_vol_zone.empty else None
            hv_high = float(high_vol_zone['High'].max()) if not high_vol_zone.empty else None

            # Patterns
            inside = detect_inside_bar(df)
            flag = detect_flag_pole(df)
            triangle = detect_ascending_descending_triangle(df)
            # small support/resistance via rolling minima/maxima
            try:
                local_mins, local_maxs = find_local_extrema(df['Close'], order=3)
            except Exception:
                local_mins, local_maxs = [], []

            # Build concise professional analysis (numeric)
            numeric_report = {
                "ticker": ticker,
                "last_datetime_ist": str(latest['Datetime_IST']),
                "last_close": round(float(latest['Close']),2),
                "change_pct": round(change_pct,3),
                "rsi": round(float(latest['RSI14']) if pd.notna(latest['RSI14']) else np.nan,2),
                "ema20": round(float(latest['EMA20']) if pd.notna(latest['EMA20']) else np.nan,2),
                "ema50": round(float(latest['EMA50']) if pd.notna(latest['EMA50']) else np.nan,2),
                "avg_vol": round(avg_vol,2) if avg_vol is not None else None,
                "high_vol_zone": (hv_low, hv_high),
                "sr_levels": sr_levels,
                "fibonacci": fibs,
                "patterns": {
                    "inside_bar": inside,
                    "flag_pole": flag,
                    "triangle": triangle
                },
                "local_mins_idx": local_mins,
                "local_maxs_idx": local_maxs
            }

            # Multi-agent local analysis (trend, momentum, zones, pattern)
            def trend_agent(numeric):
                # Trend rules: EMA cross, price vs EMA, RSI
                p = numeric
                trend = "Neutral"
                reasons = []
                price = p['last_close']
                if p['rsi'] is not None:
                    if p['rsi'] > 65:
                        reasons.append("RSI very high indicates strong bullish momentum.")
                    elif p['rsi'] < 35:
                        reasons.append("RSI low indicates strong bearish momentum.")
                if p['ema20'] and p['ema50']:
                    if p['ema20'] > p['ema50']:
                        trend = "Uptrend"
                        reasons.append("EMA20 above EMA50 — short-term uptrend.")
                    elif p['ema20'] < p['ema50']:
                        trend = "Downtrend"
                        reasons.append("EMA20 below EMA50 — short-term downtrend.")
                # price vs ema
                if price > p['ema20']:
                    reasons.append("Price trading above EMA20.")
                else:
                    reasons.append("Price trading below EMA20.")
                return {"trend": trend, "reasons": reasons}

            def momentum_agent(numeric):
                p = numeric
                r = []
                if p['change_pct'] is not None:
                    if abs(p['change_pct']) > 0.7:
                        r.append(f"Last interval move {p['change_pct']:.2f}% — notable momentum.")
                if p['avg_vol'] and p['high_vol_zone'][0] is not None:
                    r.append(f"High volume zone between {p['high_vol_zone'][0]} and {p['high_vol_zone'][1]}" if p['high_vol_zone'][0] is not None else "")
                # simplify
                return {"momentum_notes": [x for x in r if x]}

            def zones_agent(numeric):
                z = []
                if numeric['high_vol_zone'][0] is not None:
                    z.append(f"High-volume zone: {numeric['high_vol_zone'][0]} — {numeric['high_vol_zone'][1]}")
                z += [f"Support/Resistance Level: {lvl}" for lvl in numeric['sr_levels']]
                return {"zones": z, "fibonacci": numeric['fibonacci']}

            def pattern_agent(numeric):
                pats = numeric['patterns']
                found = []
                if pats['inside_bar']:
                    found.append("Inside bar (last candle inside previous).")
                if pats['flag_pole']:
                    found.append("Flag/pole-like structure detected (sharp move then consolidation).")
                if pats['triangle']:
                    found.append(f"Triangle pattern: {pats['triangle']}")
                return {"patterns": found}

            # compute agents
            trend_res = trend_agent(numeric_report)
            momentum_res = momentum_agent(numeric_report)
            zones_res = zones_agent(numeric_report)
            pattern_res = pattern_agent(numeric_report)

            # Build professional prompt for LLM (if enabled)
            combined_prompt = f"""
You are an experienced market analyst for professional traders.
Provide a concise professional report for {ticker} using the supplied numeric data and findings.
1) A one-line market call (BULLISH/BEARISH/NEUTRAL).
2) Short rationale (2-3 lines) referencing RSI, EMA, volume zones, S/R levels, patterns, and Fibonacci.
3) Clear trading actions: exact suggestion (BUY/SELL/HOLD), entry guidance, stop-loss suggestion, and target(s).
4) Highlight key support/resistance and trap zones in plain language.
5) Keep the overall output crisp, professional, and suitable for live trading decisions.

Numeric summary:
{numeric_report}

Agent outputs:
Trend: {trend_res}
Momentum: {momentum_res}
Zones: {zones_res}
Patterns: {pattern_res}

Produce the report in bullet points or short paragraphs only (no long essays).
"""
            # LLM professional writeup
            llm_text = call_llama_groq(combined_prompt, max_tokens=400, temperature=0.0) if use_llm else "[LLM disabled]"

            # Render results in Streamlit (persisting UI)
            st.session_state['last_fetch_time'] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            with output_area.container():
                st.markdown("### Numeric Summary")
                cols = st.columns(3)
                cols[0].metric("Last Close", numeric_report['last_close'])
                cols[1].metric("Change % (last interval)", numeric_report['change_pct'])
                cols[2].metric("RSI(14)", numeric_report['rsi'])
                st.write("EMA20:", numeric_report['ema20'], "EMA50:", numeric_report['ema50'])
                st.write("Average Volume (recent):", numeric_report['avg_vol'])

                st.markdown("### Zones & Levels")
                st.write("Support/Resistance levels (derived):", numeric_report['sr_levels'])
                st.write("Fibonacci (from recent window):", numeric_report['fibonacci'])
                if numeric_report['high_vol_zone'][0] is not None:
                    st.write("High-volume zone (big player activity):", numeric_report['high_vol_zone'])

                st.markdown("### Patterns detected (simple heuristics)")
                if pattern_res['patterns']:
                    for p in pattern_res['patterns']:
                        st.write("- ", p)
                else:
                    st.write("None detected by heuristics.")

                st.markdown("### Agent Analysis (concise)")
                st.write("Trend Agent:", trend_res['trend'])
                for r in trend_res['reasons']:
                    st.write("- ", r)
                if momentum_res['momentum_notes']:
                    st.write("Momentum notes:")
                    for m in momentum_res['momentum_notes']:
                        st.write("- ", m)
                st.write("Zones:")
                for z in zones_res['zones']:
                    st.write("- ", z)

                st.markdown("### Professional LLaMA Summary (if enabled)")
                st.write(llm_text)

                st.markdown("---")
                st.write("### Raw data preview (last rows, timestamps IST)")
                st.dataframe(df.tail(rows_to_analyze).set_index('Datetime_IST'))

# Keep the interface active (do not hide after click)
st.markdown("---")
st.caption("App: data fetched from yfinance only when 'Fetch & Run' is clicked. Cached to prevent rate limits. LLaMA integration optional (set GROQ_API_KEY).")
