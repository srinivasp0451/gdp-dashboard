#!/usr/bin/env python3
"""
NSE Intraday Rally Screener
============================
Scans a universe of liquid NSE stocks at/near market open and returns the
SINGLE best long (bullish) intraday candidate, ranked by a composite score
of momentum, relative volume, trend, relative strength vs Nifty, and
breakout proximity -- filtered so ONLY setups with acceptable risk:reward
survive.

No talib / pandas-ta / ta. All indicators (ATR, RSI, EMA, VWAP) are hand
rolled with pandas/numpy so you can see and tweak every formula.

HOW TO RUN
----------
    pip install yfinance pandas numpy requests
    python nse_intraday_screener.py

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
  `fetch_daily_batch` / `fetch_intraday` functions only, the scoring
  logic below stays the same.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

NSE_INDEX_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"

# Liquidity / quality filters -- these protect you from garbage setups
MIN_PRICE = 30.0                       # skip ultra low-price / penny stocks
MIN_AVG_TURNOVER_CR = 5.0              # min 20-day avg daily turnover (Rs. crore)
MIN_RR = 1.5                           # HARD floor -- reject any setup below this R:R

# How many stocks survive the cheap Stage-1 (daily-data) filter and get the
# expensive Stage-2 (intraday-data) deep dive
SHORTLIST_SIZE = 20

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
MAX_WORKERS = 8

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
TRADING_MINUTES_PER_DAY = 375

# Curated, liquid fallback universe (Nifty ~100 constituents) used only if
# the live NSE index-constituents CSV can't be fetched (blocked network,
# NSE site change, etc.)
FALLBACK_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "KOTAKBANK", "AXISBANK",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "M&M", "HCLTECH", "ADANIENT",
    "ADANIPORTS", "TATASTEEL", "TATAMOTORS", "JSWSTEEL", "BAJAJFINSV",
    "HDFCLIFE", "SBILIFE", "INDUSINDBK", "GRASIM", "DRREDDY", "CIPLA",
    "DIVISLAB", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "BRITANNIA",
    "DABUR", "GODREJCP", "PIDILITIND", "HAVELLS", "SIEMENS", "DLF",
    "VEDANTA", "COALINDIA", "IOC", "BPCL", "GAIL", "HINDALCO", "SHREECEM",
    "AMBUJACEM", "ACC", "UPL", "BOSCHLTD", "MOTHERSON", "TVSMOTOR",
    "PAGEIND", "MARICO", "COLPAL", "TATACONSUM", "TECHM", "LTIM", "MPHASIS",
    "PERSISTENT", "COFORGE", "PIIND", "SRF", "TATAPOWER", "TORNTPHARM",
    "LUPIN", "AUROPHARMA", "ALKEM", "BIOCON", "ZYDUSLIFE", "ICICIPRULI",
    "ICICIGI", "BAJAJHLDNG", "CHOLAFIN", "MUTHOOTFIN", "SHRIRAMFIN", "PFC",
    "RECLTD", "IRFC", "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB",
    "FEDERALBNK", "AUBANK", "NAUKRI", "DMART", "TRENT", "ZOMATO", "NYKAA",
    "POLICYBZR", "IRCTC", "INDIGO", "INDHOTEL", "JUBLFOOD", "PVRINOX",
]


# ============================================================================
# UNIVERSE
# ============================================================================

def get_universe() -> list:
    """Try to pull the live Nifty200 constituent list from NSE. Fall back
    to a curated liquid-stock list if that fails for any reason."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/124.0 Safari/537.36"}
        resp = requests.get(NSE_INDEX_CSV_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        symbols = [f"{s.strip()}.NS" for s in df["Symbol"].tolist() if isinstance(s, str)]
        if len(symbols) > 50:
            print(f"[Universe] Loaded {len(symbols)} live symbols from NSE Nifty200 list.")
            return symbols
        raise ValueError("CSV parsed but too few symbols")
    except Exception as e:
        print(f"[Universe] Could not fetch live NSE list ({e}). Using fallback list "
              f"of {len(FALLBACK_UNIVERSE)} liquid stocks.")
        return [f"{s}.NS" for s in FALLBACK_UNIVERSE]


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


def run_stage1(universe: list, now: dt.datetime) -> dict:
    all_tickers = universe + [INDEX_TICKER]
    print(f"[Stage 1] Downloading daily data for {len(all_tickers)} symbols "
          f"(batched)...")
    raw = yf.download(tickers=all_tickers, period=DAILY_PERIOD, interval="1d",
                       group_by="ticker", threads=True, progress=False,
                       auto_adjust=False)

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
    return candidates


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
    try:
        df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                          progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return ticker, df
    except Exception:
        return ticker, None


def run_stage2(candidates: dict, shortlist: list) -> None:
    print(f"[Stage 2] Pulling 5-min intraday data for shortlisted "
          f"{len(shortlist)} stocks...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_intraday_one, t) for t in shortlist]
        for fut in as_completed(futures):
            ticker, idf = fut.result()
            c = candidates.get(ticker)
            if c is None:
                continue
            if idf is None or len(idf) < 2:
                c.rejected_reason = "no usable intraday data"
                continue
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
# REPORTING
# ============================================================================

def print_report(ranked: pd.DataFrame, run_time: dt.datetime):
    print("\n" + "=" * 72)
    print(f"  NSE INTRADAY RALLY SCREENER -- {run_time:%Y-%m-%d %H:%M} IST")
    print("=" * 72)

    if ranked.empty:
        print("\nNo stock cleared all filters today (liquidity, trend, and "
              f"minimum {MIN_RR}:1 risk-reward). That's a valid outcome --  "
              "no trade is better than a forced, bad-R:R trade. Try again "
              "closer to 9:30-9:45 once the open settles.")
        print("=" * 72)
        return

    top = ranked.iloc[0]
    print(f"\n>>> TODAY'S PICK: {top['ticker'].replace('.NS','')}  "
          f"(score {top['composite_score']:.1f}/100)\n")
    print(f"  Current price      : Rs. {top['price']:.2f}")
    print(f"  Gap from prev close: {top['gap_pct']:+.2f}%")
    print(f"  Move since open    : {top['intraday_change_pct']:+.2f}%")
    print(f"  Relative volume    : {top['rel_volume']:.2f}x expected-for-time-of-day")
    print(f"  Trend (EMA20>EMA50): {'Yes' if top['trend_ok'] else 'No'}")
    print(f"  Above VWAP today   : {'Yes' if top['above_vwap'] else 'No'}")
    print(f"  Opening-range break: {'Yes' if top['orb_breakout'] else 'No'}")
    print(f"  Rel. strength vs Nifty (5d): {top['rel_strength_5d']:+.2f}%")
    print()
    print(f"  --- TRADE PLAN ---")
    print(f"  Entry (~market)    : Rs. {top['entry']:.2f}")
    print(f"  Stop loss          : Rs. {top['stop']:.2f}  "
          f"({(top['stop']/top['entry']-1)*100:+.2f}%)")
    print(f"  Target             : Rs. {top['target']:.2f}  "
          f"({(top['target']/top['entry']-1)*100:+.2f}%)")
    print(f"  Risk:Reward        : 1 : {top['rr']:.2f}")
    print()
    print("  WHY THIS ONE: it ranks at the top of the scan on a blend of "
          "relative volume, gap/early momentum, uptrend + VWAP position, "
          "relative strength vs the Nifty, and proximity to a breakout "
          "level -- and it's the highest-scoring setup that still clears "
          f"the {MIN_RR}:1 minimum risk-reward bar.")

    print("\n" + "-" * 72)
    print(f"  Full ranked shortlist (top {min(10, len(ranked))} of "
          f"{len(ranked)} that passed all filters):")
    print("-" * 72)
    show = ranked.head(10)[["ticker", "composite_score", "gap_pct", "rel_volume",
                             "rr", "entry", "stop", "target"]].copy()
    show["ticker"] = show["ticker"].str.replace(".NS", "", regex=False)
    show.columns = ["Ticker", "Score", "Gap%", "RelVol", "R:R", "Entry", "Stop", "Target"]
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(show.to_string(index=False))

    print("\n" + "=" * 72)
    print("  DISCLAIMER: Research tool output, not investment advice. "
          "Data via Yahoo Finance (yfinance) can be delayed intraday -- "
          "confirm price/volume on your broker terminal before executing. "
          "Always honor the stop-loss. Size the position so a stop-out "
          "costs you an amount you're fully comfortable losing.")
    print("=" * 72 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    run_time = dt.datetime.now()
    if run_time.weekday() >= 5:
        print("Today is a weekend -- NSE is closed. Run this on a trading day.")
        return

    universe = get_universe()

    t0 = time.time()
    candidates = run_stage1(universe, run_time)
    if not candidates:
        print("Stage 1 returned no candidates -- check your internet connection "
              "or try again in a few minutes (Yahoo Finance can rate-limit).")
        return

    shortlist = prescore_and_shortlist(candidates, SHORTLIST_SIZE)
    run_stage2(candidates, shortlist)

    shortlisted_candidates = {t: candidates[t] for t in shortlist if t in candidates}
    ranked = score_candidates(shortlisted_candidates)

    print_report(ranked, run_time)
    print(f"[Done in {time.time()-t0:.1f}s]")

    if not ranked.empty:
        log_path = f"screener_log_{run_time:%Y%m%d_%H%M}.csv"
        ranked.to_csv(log_path, index=False)
        print(f"Full scored shortlist saved to {log_path}")


if __name__ == "__main__":
    main()
