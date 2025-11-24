# app.py
# Professional Algo Trading Dashboard (Streamlit + yfinance)
# - Manual indicators (RSI, EMA, SMA, ATR, volatility, Fibonacci, Z-score)
# - Multi-timeframe analysis
# - Ratio analysis
# - Statistical distributions and pattern recognition
# - Final trading recommendation with consistent multi-timeframe logic

import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime
from scipy.stats import skew, kurtosis, norm

# =========================
# Global Config & Constants
# =========================

st.set_page_config(
    page_title="Professional Algo Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

IST_TZ = "Asia/Kolkata"

# Map friendly names to yfinance tickers
DEFAULT_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "GOLD (XAUUSD)": "GC=F",
    "SILVER (XAGUSD)": "SI=F",
    "USD/INR": "USDINR=X",
}

INTERVAL_OPTIONS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "2h", "4h", "1d"
]

PERIOD_OPTIONS = [
    "1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y",
    "5y", "6y", "10y", "15y", "20y", "25y", "30y"
]

# Multi-timeframe sets (interval, period)
MTF_CONFIG = [
    ("1m", "1d"),
    ("5m", "5d"),
    ("15m", "5d"),
    ("30m", "1mo"),
    ("1h", "1mo"),
    ("2h", "3mo"),
    ("4h", "6mo"),
    ("1d", "1y"),
    ("1wk", "5y"),
    ("1mo", "10y"),
]

# ======================
# Helper Utility Methods
# ======================

def convert_to_ist(df: pd.DataFrame, datetime_col: str = "Datetime") -> pd.DataFrame:
    """Ensure datetime column is timezone-aware IST and sorted."""
    if datetime_col not in df.columns:
        return df
    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    # If tz-naive, localize to UTC first then convert; adjust as needed
    if df[datetime_col].dt.tz is None:
        df[datetime_col] = df[datetime_col].dt.tz_localize("UTC")
    df[datetime_col] = df[datetime_col].dt.tz_convert(IST_TZ)
    df = df.sort_values(datetime_col)
    df = df.reset_index(drop=True)
    return df


def clean_yfinance_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Handle yfinance multi-index or regular index.
    Always return DataFrame with columns:
    ['Datetime','Open','High','Low','Close','Adj Close','Volume']
    in IST timezone.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

    df = raw.copy()

    # If multi-index (ticker level), flatten
    if isinstance(df.index, pd.MultiIndex):
        # Index levels typically: (Datetime, Ticker)
        df = df.reset_index()
        if "index" in df.columns and "Date" in df.columns:
            # Old style
            pass
        if "Datetime" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
    else:
        # Normal DateTime index
        df = df.reset_index()
        # yfinance uses 'Date' or 'Datetime' depending on interval
        if "Date" in df.columns and "Datetime" not in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        if "index" in df.columns and "Datetime" not in df.columns:
            df = df.rename(columns={"index": "Datetime"})

    # Keep only OHLCV-related columns
    cols_map = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    }
    # Sometimes yahoo returns lowercase or slightly different names
    for col in list(df.columns):
        lc = col.lower()
        if lc == "open":
            df.rename(columns={col: "Open"}, inplace=True)
        elif lc == "high":
            df.rename(columns={col: "High"}, inplace=True)
        elif lc == "low":
            df.rename(columns={col: "Low"}, inplace=True)
        elif lc in ["close", "closing"]:
            df.rename(columns={col: "Close"}, inplace=True)
        elif lc in ["adj close", "adjclose", "adjusted close"]:
            df.rename(columns={col: "Adj Close"}, inplace=True)
        elif lc == "volume":
            df.rename(columns={col: "Volume"}, inplace=True)

    required_cols = ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[required_cols]
    df = convert_to_ist(df, "Datetime")
    return df


def fetch_yfinance_data(ticker: str, interval: str, period: str, delay_sec: float = 2.0) -> pd.DataFrame:
    """Safe wrapper to fetch data from yfinance with delay and error handling."""
    time.sleep(delay_sec)
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    df = clean_yfinance_df(data)
    return df


# ======================
# Indicator Calculations
# ======================

def calc_returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    return df[col].pct_change().replace([np.inf, -np.inf], np.nan)


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calc_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = pd.Series(rsi, index=series.index)
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr


def calc_volatility(df: pd.DataFrame, period: int = 20, col: str = "Close") -> pd.Series:
    rets = calc_returns(df, col)
    vol = rets.rolling(window=period, min_periods=period).std() * np.sqrt(period)
    return vol


def fib_levels(df: pd.DataFrame, col: str = "Close"):
    if df.empty:
        return {}
    high = df[col].max()
    low = df[col].min()
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low,
    }
    return levels


def detect_support_resistance(df: pd.DataFrame, col: str = "Close", window: int = 5):
    """Very basic pivot-based S/R detection."""
    prices = df[col]
    supports = []
    resistances = []
    for i in range(window, len(prices) - window):
        local_min = prices[i-window:i+window+1].min()
        local_max = prices[i-window:i+window+1].max()
        if prices.iloc[i] == local_min:
            supports.append(prices.iloc[i])
        if prices.iloc[i] == local_max:
            resistances.append(prices.iloc[i])
    # Deduplicate levels roughly
    def cluster_levels(levels, tol=0.003):
        clusters = []
        for lvl in sorted(levels):
            if not clusters:
                clusters.append(lvl)
            else:
                if abs(lvl - clusters[-1]) / clusters[-1] > tol:
                    clusters.append(lvl)
        return clusters
    return cluster_levels(supports), cluster_levels(resistances)


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


# ============================
# Ratio & Volatility Binning
# ============================

def create_ratio_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1.empty or df2.empty:
        return pd.DataFrame()
    merged = pd.merge_asof(
        df1.sort_values("Datetime"),
        df2.sort_values("Datetime"),
        on="Datetime",
        suffixes=("1", "2")
    )
    merged["Ratio"] = merged["Close1"] / merged["Close2"]
    merged["RSI1"] = calc_rsi(merged["Close1"])
    merged["RSI2"] = calc_rsi(merged["Close2"])
    merged["RSI_Ratio"] = calc_rsi(merged["Ratio"])
    return merged


def create_bins(series: pd.Series, bins_count: int = 5, labels_prefix: str = ""):
    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series_clean) < bins_count:
        quantiles = np.linspace(0, 1, len(series_clean))
    quantiles = np.linspace(0, 1, bins_count + 1)
    edges = series_clean.quantile(quantiles).values
    # Force monotonic
    edges = np.unique(edges)
    if len(edges) <= 1:
        return None, None
    # Build label strings with ranges
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{labels_prefix}{i+1} ({edges[i]:.4f}-{edges[i+1]:.4f})")
    return edges, labels


def assign_bins(series: pd.Series, edges, labels):
    if edges is None or labels is None or len(edges) <= 1:
        return pd.Series(["N/A"] * len(series), index=series.index)
    binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return binned.astype(str)


def volatility_bins(df: pd.DataFrame, vol_col: str = "Volatility") -> pd.DataFrame:
    if df.empty or vol_col not in df.columns:
        return pd.DataFrame()
    edges, labels = create_bins(df[vol_col], bins_count=5, labels_prefix="Bin ")
    df = df.copy()
    df["Vol_Bin"] = assign_bins(df[vol_col], edges, labels)
    return df, edges, labels


# ====================
# Pattern Recognition
# ====================

def detect_patterns(df: pd.DataFrame, move_threshold_points: float = 30.0, lookback_candles: int = 10):
    """
    Detect significant moves (absolute move >= threshold)
    and analyze preceding N candles for multiple pattern flags.
    """
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame(), {}

    closes = df["Close"]
    moves = closes.diff()
    results = []

    for i in range(1, len(df)):
        move_size = moves.iloc[i]
        if pd.isna(move_size):
            continue
        if abs(move_size) >= move_threshold_points:
            idx = df.index[i]
            ts = df.loc[idx, "Datetime"]
            direction = "Up" if move_size > 0 else "Down"
            start_idx = max(0, i - lookback_candles)
            window_df = df.iloc[start_idx:i]

            # Volatility burst: compare last candle range vs window mean range
            window_range = (window_df["High"] - window_df["Low"]).mean()
            last_range = df["High"].iloc[i] - df["Low"].iloc[i]
            vol_burst = last_range > 1.5 * window_range if not np.isnan(window_range) else False

            # Volume spike
            vol_spike = False
            if "Volume" in df.columns:
                window_vol = window_df["Volume"].mean()
                last_vol = df["Volume"].iloc[i]
                if not np.isnan(window_vol) and window_vol > 0:
                    vol_spike = last_vol > 1.5 * window_vol

            # RSI divergence (simplified)
            rsi_series = calc_rsi(df["Close"])
            rsi_before = rsi_series.iloc[start_idx:i].iloc[-1] if i - start_idx > 1 else np.nan
            rsi_at = rsi_series.iloc[i]
            price_before = df["Close"].iloc[start_idx:i].iloc[-1] if i - start_idx > 1 else np.nan
            price_at = df["Close"].iloc[i]
            rsi_div = False
            if not any(np.isnan([rsi_before, rsi_at, price_before, price_at])):
                # Bullish divergence: price lower low, RSI higher low
                bullish = (price_at < price_before) and (rsi_at > rsi_before)
                # Bearish divergence: price higher high, RSI lower high
                bearish = (price_at > price_before) and (rsi_at < rsi_before)
                rsi_div = bullish or bearish

            # EMA crossovers (20/50)
            ema20 = calc_ema(df["Close"], 20)
            ema50 = calc_ema(df["Close"], 50)
            ema_cross = False
            if i >= 1:
                prev20, prev50 = ema20.iloc[i-1], ema50.iloc[i-1]
                cur20, cur50 = ema20.iloc[i], ema50.iloc[i]
                if not any(np.isnan([prev20, prev50, cur20, cur50])):
                    if (prev20 < prev50 and cur20 > cur50) or (prev20 > prev50 and cur20 < cur50):
                        ema_cross = True

            # Support/Resistance breakout (rough)
            sup_levels, res_levels = detect_support_resistance(window_df)
            price = df["Close"].iloc[i]
            sr_break = False
            for s in sup_levels:
                if price < s * 0.995:
                    sr_break = True
            for r in res_levels:
                if price > r * 1.005:
                    sr_break = True

            # Large body candles
            body = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
            wick = (df["High"].iloc[i] - df["Low"].iloc[i]) - body
            large_body = body > 1.5 * (df["Close"] - df["Open"]).abs().rolling(lookback_candles).mean().iloc[i]

            # Consecutive moves
            consec_up = False
            consec_down = False
            if i >= 3:
                wins = closes.iloc[i-3:i].diff().dropna()
                consec_up = all(wins > 0)
                consec_down = all(wins < 0)

            # Liquidity sweeps / smart money / VWAP activity -> placeholders (need tick/order-book in real life)
            liquidity_sweep = vol_spike and vol_burst
            smart_money = liquidity_sweep and large_body
            vwap_activity = False  # placeholder without intrabar data

            window_rets = calc_returns(window_df).dropna()
            corr = window_rets.corr(window_rets.shift(1)) if len(window_rets) > 2 else np.nan

            results.append({
                "Datetime": ts,
                "MovePoints": float(move_size),
                "MovePct": float(move_size / df["Close"].iloc[i-1] * 100 if df["Close"].iloc[i-1] != 0 else np.nan),
                "Direction": direction,
                "VolatilityBurst": "Yes" if vol_burst else "No",
                "VolumeSpike": "Yes" if vol_spike else "No",
                "RSIDivergence": "Yes" if rsi_div else "No",
                "EMACrossover20_50": "Yes" if ema_cross else "No",
                "SRBreakout": "Yes" if sr_break else "No",
                "LargeBody": "Yes" if large_body else "No",
                "ConsecutiveUp": "Yes" if consec_up else "No",
                "ConsecutiveDown": "Yes" if consec_down else "No",
                "LiquiditySweep": "Yes" if liquidity_sweep else "No",
                "SmartMoneyActivity": "Yes" if smart_money else "No",
                "VWAPActivity": "Yes" if vwap_activity else "No",
                "RSI_Before": float(rsi_before) if not np.isnan(rsi_before) else np.nan,
                "RSI_AtMove": float(rsi_at) if not np.isnan(rsi_at) else np.nan,
                "CorrelationCoeff": float(corr) if not np.isnan(corr) else np.nan,
            })

    pattern_df = pd.DataFrame(results)
    summary = {}
    if not pattern_df.empty:
        summary["TotalPatterns"] = len(pattern_df)
        freq_cols = [
            "VolatilityBurst",
            "VolumeSpike",
            "RSIDivergence",
            "EMACrossover20_50",
            "SRBreakout",
            "LargeBody",
            "ConsecutiveUp",
            "ConsecutiveDown",
            "LiquiditySweep",
            "SmartMoneyActivity",
        ]
        for col in freq_cols:
            summary[f"{col}_Count"] = (pattern_df[col] == "Yes").sum()
    return pattern_df, summary


# =========================
# Multi-Timeframe Analysis
# =========================

def analyze_single_timeframe(df: pd.DataFrame, label: str) -> dict:
    """
    Compute all required metrics for one timeframe.
    """
    if df.empty:
        return {
            "Timeframe": label,
            "Trend": "N/A",
            "MaxClose": np.nan,
            "MinClose": np.nan,
            "FibonacciLevels": {},
            "Volatility": np.nan,
            "%Change": np.nan,
            "PointsChange": np.nan,
            "RSI": np.nan,
            "RSIStatus": "N/A",
            "EMA9": np.nan,
            "EMA20": np.nan,
            "EMA21": np.nan,
            "EMA33": np.nan,
            "EMA50": np.nan,
            "EMA100": np.nan,
            "EMA150": np.nan,
            "EMA200": np.nan,
            "SMA20": np.nan,
            "SMA50": np.nan,
            "SMA100": np.nan,
            "SMA150": np.nan,
            "SMA200": np.nan,
            "PriceVsEMAs": {},
            "PriceVsSMAs": {},
            "Supports": [],
            "Resistances": [],
        }

    close = df["Close"]
    max_close = close.max()
    min_close = close.min()
    fib = fib_levels(df, "Close")
    vol = calc_volatility(df, 20, "Close").iloc[-1]
    pct_change = (close.iloc[-1] / close.iloc[0] - 1) * 100 if close.iloc[0] != 0 else np.nan
    pts_change = close.iloc[-1] - close.iloc[0]
    rsi_val = calc_rsi(close).iloc[-1]

    def rsi_status(val):
        if np.isnan(val):
            return "N/A"
        if val < 30:
            return "Oversold"
        if val > 70:
            return "Overbought"
        return "Neutral"

    ema_dict = {}
    for w in [9, 20, 21, 33, 50, 100, 150, 200]:
        ema_dict[w] = calc_ema(close, w).iloc[-1]

    sma_dict = {}
    for w in [20, 50, 100, 150, 200]:
        sma_dict[w] = calc_sma(close, w).iloc[-1]

    last_price = close.iloc[-1]
    price_vs_emas = {w: "Above" if last_price > v else "Below" for w, v in ema_dict.items() if not np.isnan(v)}
    price_vs_smas = {w: "Above" if last_price > v else "Below" for w, v in sma_dict.items() if not np.isnan(v)}

    sups, ress = detect_support_resistance(df, "Close")

    trend = "Up" if pts_change > 0 else "Down" if pts_change < 0 else "Neutral"

    return {
        "Timeframe": label,
        "Trend": trend,
        "MaxClose": float(max_close),
        "MinClose": float(min_close),
        "FibonacciLevels": fib,
        "Volatility": float(vol) if not np.isnan(vol) else np.nan,
        "%Change": float(pct_change),
        "PointsChange": float(pts_change),
        "RSI": float(rsi_val),
        "RSIStatus": rsi_status(rsi_val),
        "EMA9": float(ema_dict[9]),
        "EMA20": float(ema_dict[20]),
        "EMA21": float(ema_dict[21]),
        "EMA33": float(ema_dict[33]),
        "EMA50": float(ema_dict[50]),
        "EMA100": float(ema_dict[100]),
        "EMA150": float(ema_dict[150]),
        "EMA200": float(ema_dict[200]),
        "SMA20": float(sma_dict[20]),
        "SMA50": float(sma_dict[50]),
        "SMA100": float(sma_dict[100]),
        "SMA150": float(sma_dict[150]),
        "SMA200": float(sma_dict[200]),
        "PriceVsEMAs": price_vs_emas,
        "PriceVsSMAs": price_vs_smas,
        "Supports": sups,
        "Resistances": ress,
    }


def multi_timeframe_analysis(ticker: str, delay_sec: float = 1.5):
    """
    Run across MTF_CONFIG for a ticker and return list of dicts.
    """
    mtf_results = []
    progress = st.progress(0.0, text=f"Multi-timeframe analysis for {ticker} ...")
    for idx, (interval, period) in enumerate(MTF_CONFIG):
        df = fetch_yfinance_data(ticker, interval, period, delay_sec=delay_sec)
        label = f"{interval}/{period}"
        res = analyze_single_timeframe(df, label)
        mtf_results.append(res)
        progress.progress((idx+1)/len(MTF_CONFIG), text=f"{ticker}: {label}")
    progress.empty()
    return mtf_results


# ==============================
# Statistical & Z-score Section
# ==============================

def stats_distribution(df: pd.DataFrame, col: str = "Close"):
    if df.empty:
        return None
    rets = calc_returns(df, col).dropna()
    if rets.empty:
        return None
    rets_points = df[col].diff().dropna()
    z = zscore(rets)
    mean_val = rets.mean()
    std_val = rets.std(ddof=1)
    sk = skew(rets)
    ku = kurtosis(rets)
    current_ret = rets.iloc[-1]
    current_z = z.iloc[-1]
    pct_rank = (rets <= current_ret).mean() * 100

    stats = {
        "returns_pct": rets,
        "returns_points": rets_points,
        "zscore_series": z,
        "mean": mean_val,
        "std": std_val,
        "skew": sk,
        "kurtosis": ku,
        "current_z": current_z,
        "percentile_rank": pct_rank,
    }
    return stats


# =========================
# Final Recommendation Logic
# =========================

def rsi_signal(value: float) -> int:
    """Return +1 (bullish), -1 (bearish), 0 (neutral)."""
    if np.isnan(value):
        return 0
    if value < 30:
        return +1
    if value > 70:
        return -1
    return 0


def zscore_signal(value: float) -> int:
    """Mean-reversion style: >2 short, <-2 long, else neutral."""
    if np.isnan(value):
        return 0
    if value > 2:
        return -1
    if value < -2:
        return +1
    return 0


def ema_alignment_signal(price_vs_emas: dict) -> int:
    """
    If majority of EMAs are below price -> bullish (+1),
    if majority above -> bearish (-1),
    else neutral (0).
    """
    if not price_vs_emas:
        return 0
    ups = sum(1 for v in price_vs_emas.values() if v == "Above")
    downs = sum(1 for v in price_vs_emas.values() if v == "Below")
    if ups > downs:
        return +1
    if downs > ups:
        return -1
    return 0


def aggregate_mtf_trend(mtf_results: list) -> int:
    """
    Resolve conflicting multi-timeframe directions.
    - Short-term (intraday) frames have higher weight for intraday trades.
    - Daily/weekly used as regime filter.
    Returns +1 (bull), -1 (bear), 0 (neutral).
    """
    if not mtf_results:
        return 0
    # Weighting by timeframe importance
    weights = {}
    for res in mtf_results:
        tf = res["Timeframe"]
        if tf.startswith("1m") or tf.startswith("5m") or tf.startswith("15m"):
            weights[tf] = 3
        elif tf.startswith("30m") or tf.startswith("1h") or tf.startswith("2h") or tf.startswith("4h"):
            weights[tf] = 2
        else:  # 1d+, swing frames
            weights[tf] = 1
    score = 0
    for res in mtf_results:
        tf = res["Timeframe"]
        w = weights.get(tf, 1)
        if res["Trend"] == "Up":
            score += w
        elif res["Trend"] == "Down":
            score -= w
    if score > 0:
        return +1
    if score < 0:
        return -1
    return 0


def build_trade_recommendation(
    df: pd.DataFrame,
    mtf_results: list,
    rsi_val: float,
    zstats: dict,
    ema_map: dict,
    risk_pct: float = 0.01,
    capital: float = 100000.0,
):
    """
    Combine signals with weights:
    - MTF trend: 30%
    - RSI: 20%
    - Z-score: 20%
    - EMA alignment: 30%
    And build a single clear action: Buy / Sell / Hold.
    """
    if df.empty:
        return None

    last_price = df["Close"].iloc[-1]

    # Signals
    mtf_dir = aggregate_mtf_trend(mtf_results)
    sig_rsi = rsi_signal(rsi_val)
    sig_z = zscore_signal(zstats["current_z"] if zstats else np.nan)
    sig_ema = ema_alignment_signal(ema_map)

    # Weights
    w_mtf, w_rsi, w_z, w_ema = 0.3, 0.2, 0.2, 0.3
    total_score = (
        w_mtf * mtf_dir +
        w_rsi * sig_rsi +
        w_z * sig_z +
        w_ema * sig_ema
    )

    if total_score > 0.25:
        action = "Buy"
    elif total_score < -0.25:
        action = "Sell"
    else:
        action = "Hold"

    # Confidence: map |score| -> Low/Moderate/High
    abs_score = abs(total_score)
    if abs_score >= 0.75:
        confidence = "High"
    elif abs_score >= 0.4:
        confidence = "Moderate"
    else:
        confidence = "Low"

    # ATR-based target & SL
    atr_series = calc_atr(df, period=14)
    atr_now = atr_series.iloc[-1] if not atr_series.empty else last_price * 0.02
    atr_now = atr_now if not np.isnan(atr_now) else last_price * 0.02

    if action == "Buy":
        entry = last_price
        stop = entry - 1.5 * atr_now
        target = entry + 2 * atr_now
    elif action == "Sell":
        entry = last_price
        stop = entry + 1.5 * atr_now
        target = entry - 2 * atr_now
    else:  # Hold, but still present symmetric hypothetical
        entry = last_price
        stop = entry - 1.5 * atr_now
        target = entry + 2 * atr_now

    # Risk per trade
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry - stop)
    position_size = math.floor(risk_amount / risk_per_unit) if risk_per_unit > 0 else 0

    rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else np.nan
    pct_gain_target = (target / entry - 1) * 100 if entry != 0 else np.nan
    pct_loss_stop = (stop / entry - 1) * 100 if entry != 0 else np.nan

    components = {
        "MTFTrendSignal": mtf_dir,
        "RSISignal": sig_rsi,
        "ZScoreSignal": sig_z,
        "EMAAlignSignal": sig_ema,
        "WeightedScore": total_score,
    }

    recommendation = {
        "Action": action,
        "EntryPrice": float(entry),
        "TargetPrice": float(target),
        "StopLoss": float(stop),
        "RiskReward": float(rr),
        "PositionSize": int(position_size),
        "PctGainTarget": float(pct_gain_target),
        "PctLossStop": float(pct_loss_stop),
        "Confidence": confidence,
        "RiskPctCapital": risk_pct * 100,
        "Components": components,
    }
    return recommendation


# ===========
# Streamlit UI
# ===========

def style_css():
    st.markdown(
        """
        <style>
        .metric-green {color: #00c853; font-weight: 600;}
        .metric-red {color: #d50000; font-weight: 600;}
        .metric-yellow {color: #ffab00; font-weight: 600;}
        .table-fixed-col th:first-child, .table-fixed-col td:first-child {position: sticky; left: 0; background: #0e1117;}
        .section-title {font-size: 1.35rem; font-weight: 700; margin-top: 1.2rem; margin-bottom: 0.4rem;}
        .sub-title {font-size: 1.05rem; font-weight: 600; margin-top: 0.8rem; margin-bottom: 0.3rem;}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    style_css()

    st.sidebar.title("Algo Trading Dashboard")

    # ----- Inputs -----
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ticker1_label = st.selectbox(
            "Ticker 1",
            options=list(DEFAULT_TICKERS.keys()) + ["Custom"],
            index=0
        )
    with col2:
        custom1 = st.text_input("Custom Ticker 1", value="", placeholder="e.g., RELIANCE.NS")

    if ticker1_label == "Custom":
        ticker1 = custom1.strip()
    else:
        ticker1 = DEFAULT_TICKERS[ticker1_label]

    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)", value=False)
    if enable_ratio:
        col3, col4 = st.sidebar.columns(2)
        with col3:
            ticker2_label = st.selectbox(
                "Ticker 2",
                options=list(DEFAULT_TICKERS.keys()) + ["Custom"],
                index=1
            )
        with col4:
            custom2 = st.text_input("Custom Ticker 2", value="", placeholder="e.g., TCS.NS")
        if ticker2_label == "Custom":
            ticker2 = custom2.strip()
        else:
            ticker2 = DEFAULT_TICKERS[ticker2_label]
    else:
        ticker2 = None

    interval = st.sidebar.selectbox("Interval", INTERVAL_OPTIONS, index=2)
    period = st.sidebar.selectbox("Period", PERIOD_OPTIONS, index=3)

    delay_sec = st.sidebar.slider("API delay (sec)", 1.5, 3.0, 2.0, 0.1)

    st.sidebar.markdown("### Risk Settings")
    capital = st.sidebar.number_input("Account capital", value=100000.0, min_value=1000.0, step=1000.0)
    risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.1) / 100.0

    st.sidebar.markdown("### Actions")
    fetch_btn = st.sidebar.button("Fetch Data & Analyze", type="primary")

    # Session state: preserve analysis
    if "analysis" not in st.session_state:
        st.session_state.analysis = {}

    if fetch_btn:
        st.session_state.analysis.clear()
        with st.spinner("Fetching data and running analysis..."):
            df1 = fetch_yfinance_data(ticker1, interval, period, delay_sec)
            st.session_state.analysis["df1"] = df1
            if enable_ratio and ticker2:
                df2 = fetch_yfinance_data(ticker2, interval, period, delay_sec)
                st.session_state.analysis["df2"] = df2
            else:
                st.session_state.analysis["df2"] = None

            # Multi-timeframe
            st.session_state.analysis["mtf1"] = multi_timeframe_analysis(ticker1, delay_sec)
            if enable_ratio and ticker2:
                st.session_state.analysis["mtf2"] = multi_timeframe_analysis(ticker2, delay_sec)
            else:
                st.session_state.analysis["mtf2"] = []

            # Ratio
            if enable_ratio and ticker2 and not st.session_state.analysis["df2"].empty:
                ratio_df = create_ratio_df(
                    st.session_state.analysis["df1"].rename(columns={"Open": "Open1", "High": "High1",
                                                                     "Low": "Low1", "Close": "Close1",
                                                                     "Adj Close": "Adj Close1", "Volume": "Volume1"}),
                    st.session_state.analysis["df2"].rename(columns={"Open": "Open2", "High": "High2",
                                                                     "Low": "Low2", "Close": "Close2",
                                                                     "Adj Close": "Adj Close2", "Volume": "Volume2"})
                )
                st.session_state.analysis["ratio_df"] = ratio_df
            else:
                st.session_state.analysis["ratio_df"] = None

            # Volatility & pattern analysis on primary df1
            if not df1.empty:
                df1["Returns"] = calc_returns(df1)
                df1["Volatility"] = calc_volatility(df1)
                st.session_state.analysis["stats1"] = stats_distribution(df1)
                patt_df, patt_summary = detect_patterns(df1)
                st.session_state.analysis["patterns1"] = patt_df
                st.session_state.analysis["patterns1_summary"] = patt_summary
            else:
                st.session_state.analysis["stats1"] = None
                st.session_state.analysis["patterns1"] = pd.DataFrame()
                st.session_state.analysis["patterns1_summary"] = {}

    # ------------- DISPLAY -------------
    st.title("Professional Algo Trading Dashboard")

    df1 = st.session_state.analysis.get("df1", pd.DataFrame())
    df2 = st.session_state.analysis.get("df2", None)
    ratio_df = st.session_state.analysis.get("ratio_df", None)
    mtf1 = st.session_state.analysis.get("mtf1", [])
    mtf2 = st.session_state.analysis.get("mtf2", [])
    stats1 = st.session_state.analysis.get("stats1", None)
    patt_df = st.session_state.analysis.get("patterns1", pd.DataFrame())
    patt_summary = st.session_state.analysis.get("patterns1_summary", {})

    if df1 is None or df1.empty:
        st.info("Select tickers, interval, period and click 'Fetch Data & Analyze' to start.")
        return

    # ==========
    # 1. Overview
    # ==========
    st.markdown("<div class='section-title'>Market Overview</div>", unsafe_allow_html=True)

    last_row = df1.iloc[-1]
    first_row = df1.iloc[0]
    price1 = last_row["Close"]
    prev_price1 = first_row["Close"]
    change_pts1 = price1 - prev_price1
    change_pct1 = (price1 / prev_price1 - 1) * 100 if prev_price1 != 0 else 0

    if enable_ratio and df2 is not None and not df2.empty:
        last_row2 = df2.iloc[-1]
        first_row2 = df2.iloc[0]
        price2 = last_row2["Close"]
        prev_price2 = first_row2["Close"]
        change_pts2 = price2 - prev_price2
        change_pct2 = (price2 / prev_price2 - 1) * 100 if prev_price2 != 0 else 0
    else:
        price2 = prev_price2 = change_pts2 = change_pct2 = None

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        cls = "metric-green" if change_pts1 > 0 else "metric-red" if change_pts1 < 0 else "metric-yellow"
        st.markdown(
            f"<span class='{cls}'>Ticker 1 ({ticker1})</span><br>"
            f"Price: {price1:.2f} | {change_pts1:+.2f} pts ({change_pct1:+.2f}%)",
            unsafe_allow_html=True
        )
        if enable_ratio and price2:
            ratio_val = price1 / price2 if price2 != 0 else np.nan
            st.markdown(f"Ratio (T1/T2): <b>{ratio_val:.4f}</b>", unsafe_allow_html=True)

    if enable_ratio and price2:
        with col_b:
            cls2 = "metric-green" if change_pts2 > 0 else "metric-red" if change_pts2 < 0 else "metric-yellow"
            st.markdown(
                f"<span class='{cls2}'>Ticker 2 ({ticker2})</span><br>"
                f"Price: {price2:.2f} | {change_pts2:+.2f} pts ({change_pct2:+.2f}%)",
                unsafe_allow_html=True
            )

    with col_c:
        st.markdown("Data Preview (Ticker 1)", unsafe_allow_html=True)
        st.dataframe(df1.tail(20), use_container_width=True)

    # Export OHLCV
    st.download_button(
        "Export Ticker 1 OHLCV to CSV",
        data=df1.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker1}_ohlcv.csv",
        mime="text/csv"
    )

    if enable_ratio and df2 is not None and not df2.empty:
        st.download_button(
            "Export Ticker 2 OHLCV to CSV",
            data=df2.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker2}_ohlcv.csv",
            mime="text/csv"
        )

    # ========================
    # 2. Ratio Analysis Section
    # ========================
    if enable_ratio and ratio_df is not None and not ratio_df.empty:
        st.markdown("<div class='section-title'>Ratio Analysis</div>", unsafe_allow_html=True)

        display_cols = [
            "Datetime", "Close1", "Close2", "Ratio", "RSI1", "RSI2", "RSI_Ratio"
        ]
        ratio_display = ratio_df[display_cols].copy()
        st.dataframe(ratio_display.tail(100), use_container_width=True)

        st.download_button(
            "Export Ratio Analysis to CSV",
            data=ratio_display.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker1}_{ticker2}_ratio.csv",
            mime="text/csv"
        )

        # Ratio binning
        edges, labels = create_bins(ratio_df["Ratio"], 5, labels_prefix="R")
        ratio_df["RatioBin"] = assign_bins(ratio_df["Ratio"], edges, labels)

        # Bin analysis table
        rows = []
        for b in ratio_df["RatioBin"].unique():
            subset = ratio_df[ratio_df["RatioBin"] == b]
            if subset.empty or b == "N/A":
                continue
            ret1_pts = subset["Close1"].iloc[-1] - subset["Close1"].iloc[0]
            ret2_pts = subset["Close2"].iloc[-1] - subset["Close2"].iloc[0]
            ret1_pct = (subset["Close1"].iloc[-1] / subset["Close1"].iloc[0] - 1) * 100 if subset["Close1"].iloc[0] != 0 else np.nan
            ret2_pct = (subset["Close2"].iloc[-1] / subset["Close2"].iloc[0] - 1) * 100 if subset["Close2"].iloc[0] != 0 else np.nan
            rows.append({
                "RatioBin": b,
                "T1_ReturnPts": ret1_pts,
                "T1_ReturnPct": ret1_pct,
                "T2_ReturnPts": ret2_pts,
                "T2_ReturnPct": ret2_pct,
            })
        if rows:
            bin_df = pd.DataFrame(rows)
            st.markdown("<div class='sub-title'>Ratio Binning Performance</div>", unsafe_allow_html=True)
            st.dataframe(bin_df, use_container_width=True)

            current_ratio = ratio_df["Ratio"].iloc[-1]
            current_bin = assign_bins(pd.Series([current_ratio]), edges, labels).iloc[0] if edges is not None else "N/A"
            st.markdown(
                f"Current ratio {current_ratio:.4f} is in bin <b>{current_bin}</b>, "
                f"historically showing these return characteristics above.",
                unsafe_allow_html=True
            )

    # ======================
    # 3. Multi-Timeframe Tables
    # ======================
    st.markdown("<div class='section-title'>Multi-Timeframe Analysis</div>", unsafe_allow_html=True)

    def mtf_to_table(mtf_results):
        if not mtf_results:
            return pd.DataFrame()
        rows = []
        for r in mtf_results:
            row = {
                "Timeframe": r["Timeframe"],
                "Trend": r["Trend"],
                "MaxClose": r["MaxClose"],
                "MinClose": r["MinClose"],
                "Volatility": r["Volatility"],
                "%Change": r["%Change"],
                "PointsChange": r["PointsChange"],
                "RSI": r["RSI"],
                "RSIStatus": r["RSIStatus"],
                "EMA9": r["EMA9"],
                "EMA20": r["EMA20"],
                "EMA21": r["EMA21"],
                "EMA33": r["EMA33"],
                "EMA50": r["EMA50"],
                "EMA100": r["EMA100"],
                "EMA150": r["EMA150"],
                "EMA200": r["EMA200"],
                "SMA20": r["SMA20"],
                "SMA50": r["SMA50"],
                "SMA100": r["SMA100"],
                "SMA150": r["SMA150"],
                "SMA200": r["SMA200"],
                "PriceVsEMAs": r["PriceVsEMAs"],
                "PriceVsSMAs": r["PriceVsSMAs"],
                "Supports": r["Supports"],
                "Resistances": r["Resistances"],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    mtf1_df = mtf_to_table(mtf1)
    st.markdown(f"<div class='sub-title'>Ticker 1 ({ticker1})</div>", unsafe_allow_html=True)
    st.dataframe(mtf1_df, use_container_width=True)

    if enable_ratio and mtf2:
        mtf2_df = mtf_to_table(mtf2)
        st.markdown(f"<div class='sub-title'>Ticker 2 ({ticker2})</div>", unsafe_allow_html=True)
        st.dataframe(mtf2_df, use_container_width=True)

    # Simple textual summary for ticker 1
    mtf_dir1 = aggregate_mtf_trend(mtf1)
    mtf_trend_text = "Bullish" if mtf_dir1 == 1 else "Bearish" if mtf_dir1 == -1 else "Neutral"
    st.markdown(
        f"Overall multi-timeframe trend for Ticker 1 appears <b>{mtf_trend_text}</b> "
        f"after weighting intraday and higher frames.", unsafe_allow_html=True
    )

    # ========================
    # 4. Volatility Bins (T1)
    # ========================
    st.markdown("<div class='section-title'>Volatility Bins (Ticker 1)</div>", unsafe_allow_html=True)
    if not df1.empty and "Volatility" in df1.columns:
        vol_df, vol_edges, vol_labels = volatility_bins(df1, "Volatility")
        if not vol_df.empty:
            show = vol_df[["Datetime", "Vol_Bin", "Volatility", "Close", "Returns"]].tail(200)
            show = show.rename(columns={
                "Volatility": "VolatilityPct",
                "Close": "Price",
                "Returns": "ReturnsPct"
            })
            st.dataframe(show, use_container_width=True)

            stats = {
                "HighVol": vol_df["Volatility"].max(),
                "LowVol": vol_df["Volatility"].min(),
                "MeanVol": vol_df["Volatility"].mean(),
                "MaxRetPts": df1["Close"].diff().max(),
                "MinRetPts": df1["Close"].diff().min(),
                "MaxRetPct": df1["Returns"].max(),
                "MinRetPct": df1["Returns"].min(),
            }
            current_bin = vol_df["Vol_Bin"].iloc[-1]
            st.markdown(
                f"Current volatility bin: <b>{current_bin}</b>. "
                f"Historical volatility ranged from {stats['LowVol']:.4f} to {stats['HighVol']:.4f} "
                f"with mean {stats['MeanVol']:.4f}.",
                unsafe_allow_html=True
            )

    # ==========================
    # 5. Pattern Recognition (T1)
    # ==========================
    st.markdown("<div class='section-title'>Advanced Pattern Recognition (Ticker 1)</div>", unsafe_allow_html=True)

    if patt_df is not None and not patt_df.empty:
        st.dataframe(patt_df.tail(100), use_container_width=True)
        st.markdown(
            f"Total significant moves detected: <b>{patt_summary.get('TotalPatterns', 0)}</b>. "
            f"Check table above for RSI divergences, EMA crossovers, and potential smart money activity.",
            unsafe_allow_html=True
        )
    else:
        st.markdown("No significant moves above threshold detected in this sample.")

    # =======================
    # 6. Statistical Analysis
    # =======================
    st.markdown("<div class='section-title'>Statistical & Z-Score Analysis (Ticker 1)</div>", unsafe_allow_html=True)
    if stats1:
        z_series = stats1["zscore_series"]
        z_table = pd.DataFrame({
            "Datetime": df1["Datetime"].iloc[-len(z_series):].values,
            "ReturnPts": stats1["returns_points"].values[-len(z_series):],
            "ReturnPct": stats1["returns_pct"].values[-len(z_series):] * 100,
            "ZScore": z_series.values
        })
        st.dataframe(z_table.tail(100), use_container_width=True)

        st.markdown(
            f"Mean return: {stats1['mean']*100:.4f}% | Std dev: {stats1['std']*100:.4f}% | "
            f"Skewness: {stats1['skew']:.3f} | Kurtosis: {stats1['kurtosis']:.3f}.",
            unsafe_allow_html=True
        )
        st.markdown(
            f"Current Z-score: <b>{stats1['current_z']:.3f}</b>, "
            f"percentile rank ≈ <b>{stats1['percentile_rank']:.1f}%</b>. "
            f"Within ±1σ covers ~68%, ±2σ ~95%, ±3σ ~99.7% under normality assumption.",
            unsafe_allow_html=True
        )

    # =========================
    # 7. Final Trade Suggestion
    # =========================
    st.markdown("<div class='section-title'>Final Trading Recommendation (Ticker 1)</div>", unsafe_allow_html=True)

    # Use latest timeframe MTF result for EMA alignment & RSI
    if mtf1:
        latest_frame = mtf1[0]  # take first (fastest) config 1m/1d as main trading TF
        ema_map = latest_frame["PriceVsEMAs"]
        rsi_val = latest_frame["RSI"]
    else:
        ema_map = {}
        rsi_val = calc_rsi(df1["Close"]).iloc[-1]

    rec = build_trade_recommendation(
        df=df1,
        mtf_results=mtf1,
        rsi_val=rsi_val,
        zstats=stats1,
        ema_map=ema_map,
        risk_pct=risk_pct,
        capital=capital,
    )

    if rec:
        colr1, colr2, colr3, colr4 = st.columns(4)
        with colr1:
            st.metric("Action", rec["Action"])
            st.metric("Confidence", rec["Confidence"])
        with colr2:
            st.metric("Entry Price", f"{rec['EntryPrice']:.2f}")
            st.metric("Target Price", f"{rec['TargetPrice']:.2f}")
        with colr3:
            st.metric("Stop Loss", f"{rec['StopLoss']:.2f}")
            st.metric("Risk/Reward", f"{rec['RiskReward']:.2f}")
        with colr4:
            st.metric("Position Size", f"{rec['PositionSize']} units")
            st.metric("%Gain @ Target", f"{rec['PctGainTarget']:.2f}%")

        st.markdown(
            f"Risk per trade: <b>{rec['RiskPctCapital']:.2f}%</b> of capital, "
            f"approx position size {rec['PositionSize']} units with stop at {rec['StopLoss']:.2f}.",
            unsafe_allow_html=True
        )
        comp = rec["Components"]
        st.markdown(
            f"- Multi-timeframe trend signal: {comp['MTFTrendSignal']} (30% weight)"
            f"- RSI signal: {comp['RSISignal']} (20% weight)"
            f"- Z-score signal: {comp['ZScoreSignal']} (20% weight)"
            f"- EMA alignment signal: {comp['EMAAlignSignal']} (30% weight)"
            f"- Combined weighted score: {comp['WeightedScore']:.3f}")

        st.markdown(
            "The final action intentionally resolves conflicting signals by weighting faster intraday timeframes "
            "more heavily for timing, while using higher timeframes as regime filters, and combining RSI, "
            "mean-reversion from Z-score, and EMA trend alignment into one unified score.",
            unsafe_allow_html=True
        )

    # =================
    # 8. Charts (Basic)
    # =================
    st.markdown("<div class='section-title'>Interactive Charts (Ticker 1)</div>", unsafe_allow_html=True)

    # Simple stacked charts: price + EMAs, and RSI
    plot_df = df1.set_index("Datetime")
    plot_df["EMA20"] = calc_ema(plot_df["Close"], 20)
    plot_df["EMA50"] = calc_ema(plot_df["Close"], 50)
    plot_df["EMA200"] = calc_ema(plot_df["Close"], 200)
    plot_df["RSI"] = calc_rsi(plot_df["Close"])

    st.line_chart(plot_df[["Close", "EMA20", "EMA50", "EMA200"]])
    st.line_chart(plot_df[["RSI"]])

    if enable_ratio and ratio_df is not None and not ratio_df.empty:
        st.markdown("<div class='section-title'>Ratio Charts</div>", unsafe_allow_html=True)
        rplot = ratio_df.set_index("Datetime")
        st.line_chart(rplot[["Close1", "Close2"]])
        st.line_chart(rplot[["Ratio"]])
        st.line_chart(rplot[["RSI1", "RSI2", "RSI_Ratio"]])


if __name__ == "__main__":
    main()
