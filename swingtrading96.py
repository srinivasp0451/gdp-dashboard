#!/usr/bin/env python3
"""
NSE Intraday Rally Screener -- single-file Streamlit app
==========================================================
Scans a universe of liquid NSE stocks at/near market open and returns the
SINGLE best long (bullish) intraday candidate, ranked by a composite score
of momentum, relative volume, trend, relative strength vs Nifty, and
breakout proximity -- filtered so ONLY setups with acceptable risk:reward
survive.

No talib / pandas-ta / ta. All indicators (ATR, RSI, EMA, VWAP) are hand
rolled with pandas/numpy so you can see and tweak every formula.

HOW TO RUN
----------
Local:
    pip install -r requirements.txt
    streamlit run app_single_file.py

Streamlit Cloud:
    Push this file + requirements.txt to a repo, set the main file path
    to this file's name, deploy.

Note: this is a Streamlit app now, not a plain CLI script -- it must be
launched with `streamlit run`, not `python app_single_file.py`. If you
also want a terminal/cron-friendly version (prints to console, no
browser needed), ask for the two-file version back and keep this one
for the browser.

Best run between 9:20 AM and 9:45 AM IST (after the opening 5-10 minutes
of noise settle, before the move you want to catch is over).

IMPORTANT / HONEST DISCLAIMER
------------------------------
- This is a screening/research tool, NOT financial advice. It does not
  know your capital, risk tolerance, or existing positions.
- Data comes from Yahoo Finance via yfinance, which is FREE but can be
  delayed or occasionally miss NSE bars intraday. For live/production
  trading, wire this into a broker feed (Zerodha Kite Connect, Upstox,
  Angel One SmartAPI, etc.) instead of yfinance -- swap out the
  `run_stage1` / `fetch_intraday_one` functions only, the scoring logic
  below stays the same.
- Past price behaviour (momentum, volume, breakout proximity) is
  correlative, not predictive. A high score means "this fits the profile
  of setups that have historically had a good chance of continuing" --
  it is not a guarantee the stock rallies today. Always use the
  stop-loss this script gives you.
- Never risk more than you're prepared to lose on any single idea, no
  matter how good the score looks.
"""

import sys
import time
import warnings
import datetime as dt
from io import StringIO
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run:  pip install yfinance pandas numpy requests")

# ============================================================================
# CONFIG -- tune these to your style
# ============================================================================

INDEX_TICKER = "^NSEI"                 # Nifty 50 index, used for relative strength

NSE_NIFTY50_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_NIFTY200_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"

# Liquidity / quality filters -- these protect you from garbage setups
MIN_PRICE = 30.0                       # skip ultra low-price / penny stocks
MIN_AVG_TURNOVER_CR = 5.0              # min 20-day avg daily turnover (Rs. crore)
MIN_RR = 1.5                           # HARD floor -- reject any setup below this R:R

# How many stocks survive the cheap Stage-1 (daily-data) filter and get the
# expensive Stage-2 (intraday-data) deep dive
SHORTLIST_SIZE = 10

# Composite score weights (must sum to 1.0)
WEIGHTS = {
    "rel_volume": 0.25,      # volume leads price -- weighted highest
    "gap_momentum": 0.20,    # how strongly it's already moving today
    "trend": 0.15,           # is the daily trend actually up
    "rel_strength": 0.15,    # outperforming the index, or just riding it
    "breakout": 0.15,        # proximity to / break of recent highs
    "volatility_fit": 0.10,  # enough range to move, not so much it's chaotic
}

DAILY_PERIOD = "9mo"
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "1d"

# ---- RATE-LIMIT SAFETY -----------------------------------------------------
# yfinance is NOT an official API -- it's Yahoo Finance's internal website
# endpoint. Yahoo has no published rate limit; it silently blocks an IP that
# it thinks looks automated (a burst of requests, especially concurrent
# ones). These settings trade a bit of speed for a much lower chance of
# getting your IP temporarily blocked. Don't remove the delays just to make
# this faster -- that's exactly what causes multi-day blocks.
CHUNK_SIZE = 15                # tickers per batched daily-data request
DELAY_BETWEEN_CHUNKS_SEC = 2.0
DELAY_BETWEEN_INTRADAY_CALLS_SEC = 0.6
MAX_RETRIES = 2
RETRY_BACKOFF_BASE_SEC = 3.0
# -----------------------------------------------------------------------------

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
TRADING_MINUTES_PER_DAY = 375

# Curated, liquid fallback universe used only if the live NSE
# index-constituents CSV can't be fetched (blocked network, NSE site
# change, etc). Kept intentionally short by default (see UNIVERSE_MODE in
# the UI) -- a smaller universe means far fewer requests to Yahoo, which is
# the single biggest lever against getting rate-limited.
FALLBACK_UNIVERSE_NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "KOTAKBANK", "AXISBANK",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "M&M", "HCLTECH", "ADANIENT",
    "ADANIPORTS", "TATASTEEL", "TATAMOTORS", "JSWSTEEL", "BAJAJFINSV",
    "HDFCLIFE", "SBILIFE", "INDUSINDBK", "GRASIM", "DRREDDY", "CIPLA",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "BRITANNIA",
]
FALLBACK_UNIVERSE_EXTRA = [
    "DIVISLAB", "DABUR", "GODREJCP", "PIDILITIND", "HAVELLS", "SIEMENS",
    "DLF", "VEDANTA", "COALINDIA", "IOC", "BPCL", "GAIL", "HINDALCO",
    "SHREECEM", "AMBUJACEM", "ACC", "UPL", "BOSCHLTD", "MOTHERSON",
    "TVSMOTOR", "PAGEIND", "MARICO", "COLPAL", "TATACONSUM", "TECHM",
    "LTIM", "MPHASIS", "PERSISTENT", "COFORGE", "PIIND", "SRF", "TATAPOWER",
    "TORNTPHARM", "LUPIN", "AUROPHARMA", "ALKEM", "BIOCON", "ZYDUSLIFE",
    "ICICIPRULI", "ICICIGI", "BAJAJHLDNG", "CHOLAFIN", "MUTHOOTFIN",
    "SHRIRAMFIN", "PFC", "RECLTD", "IRFC", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "AUBANK", "NAUKRI", "DMART", "TRENT",
    "ZOMATO", "NYKAA", "POLICYBZR", "IRCTC", "INDIGO", "INDHOTEL",
    "JUBLFOOD", "PVRINOX",
]


class RateLimited(Exception):
    """Raised when Yahoo Finance appears to be throttling/blocking us."""
    pass


def _looks_rate_limited(err: Exception) -> bool:
    msg = str(err).lower()
    signals = ["429", "too many requests", "rate limit", "expecting value",
               "not in allowlist", "connection reset", "temporarily"]
    return any(s in msg for s in signals)


def safe_download(**kwargs):
    """Wrapper around yf.download with retry/backoff. Raises RateLimited
    (instead of silently returning empty data) if Yahoo appears to be
    throttling us, so the caller can stop immediately instead of hammering
    a block into a longer one."""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            df = yf.download(**kwargs, progress=False)
            if df is None or df.empty:
                raise ValueError("empty response")
            return df
        except Exception as e:
            last_err = e
            if _looks_rate_limited(e):
                raise RateLimited(str(e))
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE_SEC * (attempt + 1))
    raise RateLimited(f"repeated failures, likely rate-limited: {last_err}")


# ============================================================================
# UNIVERSE
# ============================================================================

def get_universe(mode: str = "safe") -> list:
    """Try to pull the live Nifty50 (mode='safe') or Nifty200 (mode='full')
    constituent list from NSE. Fall back to a curated liquid-stock list if
    that fails for any reason.

    mode='safe' (default, ~40 stocks) keeps the number of requests to Yahoo
    low -- this is the main defense against getting rate-limited. mode='full'
    (~150-200 stocks) scans more of the market but sends far more requests
    and carries real risk of a temporary Yahoo IP block.
    """
    url = NSE_NIFTY50_CSV_URL if mode == "safe" else NSE_NIFTY200_CSV_URL
    fallback = (FALLBACK_UNIVERSE_NIFTY50 if mode == "safe"
                else FALLBACK_UNIVERSE_NIFTY50 + FALLBACK_UNIVERSE_EXTRA)
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/124.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        symbols = [f"{s.strip()}.NS" for s in df["Symbol"].tolist() if isinstance(s, str)]
        min_expected = 30 if mode == "safe" else 100
        if len(symbols) >= min_expected:
            print(f"[Universe] Loaded {len(symbols)} live symbols from NSE ({mode} mode).")
            return symbols
        raise ValueError("CSV parsed but too few symbols")
    except Exception as e:
        print(f"[Universe] Could not fetch live NSE list ({e}). Using fallback list "
              f"of {len(fallback)} liquid stocks.")
        return [f"{s}.NS" for s in fallback]


# ============================================================================
# HAND-ROLLED INDICATORS (no talib / pandas-ta / ta)
# ============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    # avg_loss == 0: no losses in the window.
    #   - if there were gains -> RSI is 100 (pure uptrend)
    #   - if there were no gains either (flat) -> RSI is 50 (neutral)
    no_loss = avg_loss == 0
    out = out.where(~no_loss, np.where(avg_gain > 0, 100.0, 50.0))
    return pd.Series(out, index=series.index).fillna(50)


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (typical * df["Volume"]).cumsum() / cum_vol


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Candidate:
    ticker: str
    price: float = np.nan
    prev_close: float = np.nan
    today_open: float = np.nan
    gap_pct: float = np.nan
    intraday_change_pct: float = np.nan
    rel_volume: float = np.nan
    atr14: float = np.nan
    atr_pct: float = np.nan
    trend_ok: bool = False
    ema20: float = np.nan
    ema50: float = np.nan
    high20: float = np.nan
    breakout_pct: float = np.nan
    rel_strength_5d: float = np.nan
    avg_turnover_cr: float = np.nan
    opening_range_high: float = np.nan
    opening_range_low: float = np.nan
    above_vwap: bool = False
    orb_breakout: bool = False
    entry: float = np.nan
    stop: float = np.nan
    target: float = np.nan
    rr: float = np.nan
    scores: dict = field(default_factory=dict)
    composite_score: float = np.nan
    rejected_reason: str = ""


# ============================================================================
# STAGE 1: cheap, batched, universe-wide daily-data filter
# ============================================================================

def minutes_elapsed_today(now: dt.datetime) -> int:
    open_dt = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                           second=0, microsecond=0)
    elapsed = (now - open_dt).total_seconds() / 60.0
    return int(min(max(elapsed, 1), TRADING_MINUTES_PER_DAY))


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_stage1(universe: list, now: dt.datetime):
    """Returns (candidates: dict, warning: str|None). Downloads daily data
    in small chunks with a delay between each, so we never fire a single
    burst of 100+ simultaneous requests at Yahoo. If a rate-limit signal is
    detected, stops immediately and returns whatever was already gathered
    rather than continuing to hammer a block into a longer one."""
    all_tickers = universe + [INDEX_TICKER]
    chunks = list(_chunked(all_tickers, CHUNK_SIZE))
    print(f"[Stage 1] Downloading daily data for {len(all_tickers)} symbols "
          f"in {len(chunks)} chunks of ~{CHUNK_SIZE}...")

    raw_parts = []
    warning = None
    for i, chunk in enumerate(chunks):
        try:
            part = safe_download(tickers=chunk, period=DAILY_PERIOD, interval="1d",
                                  group_by="ticker", threads=False, auto_adjust=False)
            if not isinstance(part.columns, pd.MultiIndex):
                # yfinance flattens columns when a chunk has only 1 ticker
                part = pd.concat({chunk[0]: part}, axis=1)
            raw_parts.append(part)
        except RateLimited as e:
            warning = (
                f"Yahoo Finance appears to be rate-limiting requests "
                f"(stopped after {i}/{len(chunks)} chunks). Using the "
                f"{i * CHUNK_SIZE} stocks already fetched instead of "
                f"pushing further and risking a longer block. "
                f"Try again later, or use a smaller universe next time."
            )
            print(f"[Stage 1] {warning}")
            break
        if i < len(chunks) - 1:
            time.sleep(DELAY_BETWEEN_CHUNKS_SEC)

    if not raw_parts:
        return {}, (warning or "Could not fetch any data from Yahoo Finance.")

    raw = pd.concat(raw_parts, axis=1)

    elapsed_min = minutes_elapsed_today(now)
    day_fraction = elapsed_min / TRADING_MINUTES_PER_DAY

    # Index reference numbers for relative strength
    idx_change_5d = np.nan
    try:
        idx_df = raw[INDEX_TICKER].dropna(how="all")
        if len(idx_df) >= 6:
            idx_change_5d = (idx_df["Close"].iloc[-1] / idx_df["Close"].iloc[-6] - 1) * 100
    except Exception:
        pass

    candidates = {}
    for t in universe:
        try:
            if t not in raw.columns.get_level_values(0):
                continue  # not fetched, e.g. chunk was skipped after a rate-limit stop
            df = raw[t].dropna(how="all")
            if len(df) < 40:
                continue
            hist = df.iloc[:-1]   # fully closed days
            today_row = df.iloc[-1]

            prev_close = hist["Close"].iloc[-1]
            today_open = today_row["Open"]
            price_now = today_row["Close"]  # live-forming "last traded" proxy
            if any(pd.isna(x) for x in [prev_close, today_open, price_now]) or prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close * 100
            intraday_change_pct = (price_now - today_open) / today_open * 100

            atr_series = atr(hist, 14)
            atr14 = atr_series.iloc[-1]
            if pd.isna(atr14) or atr14 <= 0:
                continue
            atr_pct = atr14 / price_now * 100

            ema20_s = ema(hist["Close"], 20)
            ema50_s = ema(hist["Close"], 50)
            ema20, ema50 = ema20_s.iloc[-1], ema50_s.iloc[-1]
            trend_ok = bool(price_now > ema20 > ema50)

            high20 = hist["High"].rolling(20).max().iloc[-1]
            breakout_pct = (price_now - high20) / high20 * 100 if high20 else np.nan

            avg_vol20 = hist["Volume"].rolling(20).mean().iloc[-1]
            avg_turnover_cr = (avg_vol20 * price_now) / 1e7 if avg_vol20 else 0

            expected_vol_so_far = avg_vol20 * day_fraction if avg_vol20 else np.nan
            today_vol = today_row["Volume"]
            rel_volume = (today_vol / expected_vol_so_far
                          if expected_vol_so_far and expected_vol_so_far > 0 else np.nan)

            rel_strength_5d = np.nan
            if len(hist) >= 5 and not pd.isna(idx_change_5d):
                stock_change_5d = (price_now / hist["Close"].iloc[-5] - 1) * 100
                rel_strength_5d = stock_change_5d - idx_change_5d

            if price_now < MIN_PRICE or avg_turnover_cr < MIN_AVG_TURNOVER_CR:
                continue

            c = Candidate(
                ticker=t, price=price_now, prev_close=prev_close,
                today_open=today_open, gap_pct=gap_pct,
                intraday_change_pct=intraday_change_pct, rel_volume=rel_volume,
                atr14=atr14, atr_pct=atr_pct, trend_ok=trend_ok, ema20=ema20,
                ema50=ema50, high20=high20, breakout_pct=breakout_pct,
                rel_strength_5d=rel_strength_5d, avg_turnover_cr=avg_turnover_cr,
            )
            candidates[t] = c
        except Exception:
            continue

    print(f"[Stage 1] {len(candidates)} stocks passed liquidity/price filters.")
    return candidates, warning


def prescore_and_shortlist(candidates: dict, n: int) -> list:
    """Cheap rank on Stage-1 metrics only, to decide who gets the expensive
    intraday deep-dive in Stage 2."""
    if not candidates:
        return []
    df = pd.DataFrame([{
        "ticker": c.ticker,
        "gap_pct": c.gap_pct,
        "rel_volume": c.rel_volume,
        "breakout_pct": c.breakout_pct,
        "trend_ok": 1.0 if c.trend_ok else 0.0,
        "rel_strength_5d": c.rel_strength_5d,
    } for c in candidates.values()])

    df = df.fillna(df.median(numeric_only=True))
    for col in ["gap_pct", "rel_volume", "breakout_pct", "rel_strength_5d"]:
        df[col + "_pct"] = df[col].rank(pct=True)

    df["prescore"] = (
        0.30 * df["rel_volume_pct"] +
        0.25 * df["gap_pct_pct"] +
        0.20 * df["breakout_pct_pct"] +
        0.15 * df["rel_strength_5d_pct"] +
        0.10 * df["trend_ok"]
    )
    top = df.sort_values("prescore", ascending=False).head(n)
    return top["ticker"].tolist()


# ============================================================================
# STAGE 2: intraday deep-dive on the shortlist only
# ============================================================================

def fetch_intraday_one(ticker: str):
    df = safe_download(tickers=ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                        auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def run_stage2(candidates: dict, shortlist: list):
    """Sequential, throttled -- deliberately NOT multi-threaded. Hitting
    Yahoo with several simultaneous connections is one of the most common
    ways this unofficial endpoint flags an IP as a bot and blocks it.
    Returns a warning string if it had to stop early due to rate-limiting."""
    print(f"[Stage 2] Pulling 5-min intraday data for shortlisted "
          f"{len(shortlist)} stocks (sequential, throttled)...")
    warning = None
    for i, ticker in enumerate(shortlist):
        c = candidates.get(ticker)
        if c is None:
            continue
        try:
            idf = fetch_intraday_one(ticker)
        except RateLimited as e:
            warning = (
                f"Yahoo Finance rate-limited us during Stage 2 (after "
                f"{i}/{len(shortlist)} stocks). Ranking is based on the "
                f"{i} stocks already fetched. Try again later if you want "
                f"the full shortlist re-checked."
            )
            print(f"[Stage 2] {warning}")
            break
        if idf is None or len(idf) < 2:
            c.rejected_reason = "no usable intraday data"
        else:
            try:
                opening_bars = idf.iloc[:3]  # first ~15 minutes
                c.opening_range_high = opening_bars["High"].max()
                c.opening_range_low = opening_bars["Low"].min()

                last_price = idf["Close"].iloc[-1]
                c.price = last_price  # refine with more current intraday price

                vwap_series = intraday_vwap(idf)
                c.above_vwap = bool(last_price > vwap_series.iloc[-1])
                c.orb_breakout = bool(last_price > c.opening_range_high)
            except Exception as e:
                c.rejected_reason = f"intraday calc failed: {e}"

        if i < len(shortlist) - 1:
            time.sleep(DELAY_BETWEEN_INTRADAY_CALLS_SEC)

    return warning


# ============================================================================
# TRADE PLAN + FINAL SCORING
# ============================================================================

def build_trade_plan(c: Candidate) -> None:
    entry = c.price
    if pd.isna(entry) or pd.isna(c.atr14) or c.atr14 <= 0:
        c.rejected_reason = c.rejected_reason or "missing price/ATR"
        return

    atr_stop = entry - 1.5 * c.atr14
    if not pd.isna(c.opening_range_low) and c.opening_range_low > 0:
        structural_stop = c.opening_range_low - 0.1 * c.atr14
        # use the tighter of the two, but never above entry
        stop = max(atr_stop, structural_stop) if structural_stop < entry else atr_stop
    else:
        stop = atr_stop

    # Floor: never let the stop sit closer than 0.6x ATR to entry. A very
    # tight opening range can otherwise produce a stop so close it gets hit
    # by ordinary noise -- and it also artificially inflates R:R, which
    # would be misleading rather than genuinely favorable.
    min_stop_distance = 0.6 * c.atr14
    if (entry - stop) < min_stop_distance:
        stop = entry - min_stop_distance

    if stop >= entry:
        c.rejected_reason = "invalid stop (>= entry)"
        return

    target_resistance = c.high20 if (not pd.isna(c.high20) and c.high20 > entry) else None
    target_atr = entry + 2.5 * c.atr14
    if target_resistance is not None:
        # Use the nearer of "resistance" and "2.5x ATR reach" -- don't let a
        # far-away historical high produce an unrealistically distant target,
        # and don't artificially stretch the target just to hit a minimum
        # R:R. If the real target doesn't clear MIN_RR, that's a valid reject.
        target = min(target_resistance, entry + 4 * c.atr14, target_atr * 1.6)
    else:
        target = target_atr

    rr = (target - entry) / (entry - stop)

    c.entry, c.stop, c.target, c.rr = entry, stop, target, rr

    if rr < MIN_RR:
        c.rejected_reason = f"R:R {rr:.2f} below minimum {MIN_RR}"


def score_candidates(candidates: dict) -> pd.DataFrame:
    rows = []
    for c in candidates.values():
        build_trade_plan(c)
        rows.append(c)

    valid = [c for c in rows if not c.rejected_reason]
    if not valid:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "ticker": c.ticker,
        "price": c.price,
        "gap_pct": c.gap_pct,
        "intraday_change_pct": c.intraday_change_pct,
        "rel_volume": c.rel_volume,
        "breakout_pct": c.breakout_pct,
        "rel_strength_5d": c.rel_strength_5d,
        "atr_pct": c.atr_pct,
        "trend_ok": 1.0 if c.trend_ok else 0.0,
        "orb_breakout": 1.0 if c.orb_breakout else 0.0,
        "above_vwap": 1.0 if c.above_vwap else 0.0,
        "entry": c.entry, "stop": c.stop, "target": c.target, "rr": c.rr,
    } for c in valid])

    df = df.fillna(df.median(numeric_only=True))

    df["rel_volume_score"] = df["rel_volume"].rank(pct=True)
    df["gap_momentum_score"] = (0.5 * df["gap_pct"].rank(pct=True) +
                                 0.5 * df["intraday_change_pct"].rank(pct=True))
    df["trend_score"] = 0.7 * df["trend_ok"] + 0.3 * df["above_vwap"]
    df["rel_strength_score"] = df["rel_strength_5d"].rank(pct=True)
    df["breakout_score"] = (0.6 * df["breakout_pct"].rank(pct=True) +
                             0.4 * df["orb_breakout"])
    ideal_atr_pct = 2.2
    df["volatility_fit_score"] = (1 - (df["atr_pct"] - ideal_atr_pct).abs() /
                                   df["atr_pct"].clip(lower=1).max()).clip(0, 1)

    df["composite_score"] = (
        WEIGHTS["rel_volume"] * df["rel_volume_score"] +
        WEIGHTS["gap_momentum"] * df["gap_momentum_score"] +
        WEIGHTS["trend"] * df["trend_score"] +
        WEIGHTS["rel_strength"] * df["rel_strength_score"] +
        WEIGHTS["breakout"] * df["breakout_score"] +
        WEIGHTS["volatility_fit"] * df["volatility_fit_score"]
    ) * 100

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ============================================================================
# STREAMLIT UI (single-file build: run with  streamlit run app.py)
# ============================================================================

import streamlit as st

st.set_page_config(page_title="NSE Intraday Rally Screener", page_icon="📈",
                    layout="wide")

st.title("📈 NSE Intraday Rally Screener")
st.caption(
    "Scans a liquid NSE universe and returns the single best long setup "
    "by composite score, filtered to a minimum risk:reward. "
    "**Research tool, not investment advice.**"
)

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    universe_mode_label = st.radio(
        "Universe size",
        ["Safe (~40 stocks, low rate-limit risk)", "Full (~150-200 stocks, higher risk)"],
        index=0,
    )
    universe_mode = "safe" if universe_mode_label.startswith("Safe") else "full"
    if universe_mode == "full":
        st.warning(
            "⚠️ Full mode sends far more requests to Yahoo Finance and can "
            "get your IP temporarily rate-limited (sometimes for a day or "
            "more). Only use this if Safe mode has been working fine for "
            "you and you understand the risk.",
            icon="⚠️",
        )
    min_rr = st.slider("Minimum Risk:Reward to accept", 1.0, 3.0,
                        float(MIN_RR), 0.1)
    min_price = st.number_input("Minimum price (Rs.)", value=float(MIN_PRICE),
                                 step=5.0)
    min_turnover = st.number_input("Minimum avg daily turnover (Rs. crore)",
                                    value=float(MIN_AVG_TURNOVER_CR), step=1.0)
    shortlist_size = st.slider("Stage-2 shortlist size", 5, 25,
                                SHORTLIST_SIZE, 1)
    st.markdown("---")
    st.caption(
        "Best run between 9:20-9:45 AM IST, after the opening 5-10 minutes "
        "of noise settle. Avoid clicking Run repeatedly within the same "
        "few minutes -- each run makes real requests to Yahoo Finance, "
        "and running it back-to-back is what triggers rate limits."
    )
    run_clicked = st.button("🔍 Run Screener Now", type="primary",
                             use_container_width=True)

# Push sidebar overrides into the module-level config the core logic reads
MIN_RR = min_rr
MIN_PRICE = min_price
MIN_AVG_TURNOVER_CR = min_turnover
SHORTLIST_SIZE = shortlist_size

if "ranked" not in st.session_state:
    st.session_state.ranked = None
    st.session_state.run_time = None

# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
if run_clicked:
    run_time = dt.datetime.now()

    if run_time.weekday() >= 5:
        st.warning("Today is a weekend — NSE is closed. This is fine to "
                   "test with, but live results won't mean much until "
                   "a trading day.")

    status = st.status("Running screener...", expanded=True)

    status.write("Loading stock universe...")
    universe = get_universe(mode=universe_mode)
    status.write(f"Universe size: {len(universe)} symbols")

    status.write("Stage 1 — chunked, throttled daily-data scan (liquidity, "
                 "trend, gap, relative volume, relative strength)...")
    candidates, stage1_warning = run_stage1(universe, run_time)
    status.write(f"{len(candidates)} stocks passed liquidity/price filters.")
    if stage1_warning:
        st.warning(stage1_warning, icon="⏳")

    if not candidates:
        status.update(label="No data returned", state="error")
        st.error(
            "No data came back from Yahoo Finance at all. This almost "
            "always means your IP is currently rate-limited. Try again "
            "later, or from a different network (mobile hotspot/VPN)."
        )
        st.session_state.ranked = None
    else:
        shortlist = prescore_and_shortlist(candidates, shortlist_size)
        status.write(f"Stage 2 — pulling live 5-min intraday bars for "
                     f"{len(shortlist)} shortlisted stocks (one at a time, "
                     f"throttled)...")
        stage2_warning = run_stage2(candidates, shortlist)
        if stage2_warning:
            st.warning(stage2_warning, icon="⏳")

        shortlisted = {t: candidates[t] for t in shortlist if t in candidates}
        ranked = score_candidates(shortlisted)

        status.update(label="Done", state="complete")
        st.session_state.ranked = ranked
        st.session_state.run_time = run_time


# ----------------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------------
ranked = st.session_state.ranked
run_time = st.session_state.run_time

if ranked is None:
    st.info("Click **Run Screener Now** in the sidebar to scan the market.")
elif ranked.empty:
    st.warning(
        f"No stock cleared all filters as of {run_time:%H:%M} IST "
        f"(liquidity, trend, and minimum {MIN_RR}:1 risk-reward). "
        "That's a valid outcome — no trade beats a forced, bad-R:R trade. "
        "Try again closer to 9:30-9:45 once the open settles."
    )
else:
    top = ranked.iloc[0]
    ticker_clean = top["ticker"].replace(".NS", "")

    st.success(f"### 🏆 Today's Pick: **{ticker_clean}**  "
               f"(score {top['composite_score']:.1f}/100)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry (~market)", f"₹{top['entry']:.2f}")
    c2.metric("Stop Loss", f"₹{top['stop']:.2f}",
              f"{(top['stop']/top['entry']-1)*100:+.2f}%")
    c3.metric("Target", f"₹{top['target']:.2f}",
              f"{(top['target']/top['entry']-1)*100:+.2f}%")
    c4.metric("Risk : Reward", f"1 : {top['rr']:.2f}")

    st.markdown("#### Why this one")
    st.write(
        f"- Gap from previous close: **{top['gap_pct']:+.2f}%**\n"
        f"- Move since open: **{top['intraday_change_pct']:+.2f}%**\n"
        f"- Relative volume: **{top['rel_volume']:.2f}x** expected for time of day\n"
        f"- Trend (EMA20 > EMA50): **{'Yes' if top['trend_ok'] else 'No'}**\n"
        f"- Above VWAP today: **{'Yes' if top['above_vwap'] else 'No'}**\n"
        f"- Opening-range breakout: **{'Yes' if top['orb_breakout'] else 'No'}**\n"
        f"- Relative strength vs Nifty (5d): **{top['rel_strength_5d']:+.2f}%**"
    )

    st.markdown("#### Full ranked shortlist")
    show = ranked.head(10)[[
        "ticker", "composite_score", "gap_pct", "rel_volume", "rr",
        "entry", "stop", "target"
    ]].copy()
    show["ticker"] = show["ticker"].str.replace(".NS", "", regex=False)
    show.columns = ["Ticker", "Score", "Gap %", "Rel. Volume", "R:R",
                    "Entry", "Stop", "Target"]
    st.dataframe(show.style.format({
        "Score": "{:.1f}", "Gap %": "{:+.2f}", "Rel. Volume": "{:.2f}x",
        "R:R": "{:.2f}", "Entry": "₹{:.2f}", "Stop": "₹{:.2f}",
        "Target": "₹{:.2f}",
    }), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download full results (CSV)",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"screener_{run_time:%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.caption(f"Scanned as of {run_time:%Y-%m-%d %H:%M} IST")

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Research tool output, not investment advice. "
    "Data via Yahoo Finance (yfinance) can be delayed intraday — confirm "
    "price/volume on your broker terminal before executing. Always honor "
    "the stop-loss. Size the position so a stop-out costs you an amount "
    "you're fully comfortable losing."
)
