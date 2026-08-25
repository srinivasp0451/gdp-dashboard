"""
Swing / Intraday Price-Action Trader — leakage-free rebuild.

Key differences vs the original script:
  * Every pattern is registered at the bar on which it could FIRST be known
    (pivot index + right-window), never at the pivot bar itself.
  * Support/resistance zones are built incrementally from confirmed pivots only.
  * Volume filter uses a trailing rolling median, not the full-sample median.
  * Patterns require a neckline / trigger break before they score.
  * Backtest: signal on bar N -> entry at OPEN of bar N+1. Stop loss is checked
    BEFORE target inside every bar (conservative). Gaps fill at the open.
  * Costs and slippage are charged on both legs.
  * Optimiser fits on a train slice and reports a purged out-of-sample slice.
  * A synthetic random-walk self-test flags any residual look-ahead bias.
"""

import time
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


# ============================ Ticker catalogue ============================

TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "USDINR=X",
    "GOLD (COMEX)": "GC=F",
    "SILVER (COMEX)": "SI=F",
    "FOREX (EURUSD)": "EURUSD=X",
    "Custom": "KAYNES.NS",
}

# yfinance history limits, in calendar days, per interval
INTERVAL_MAX_DAYS = {
    "1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59,
    "60m": 729, "1d": 20000, "1wk": 20000, "1mo": 20000,
}

INTERVAL_LABELS = {
    "1m": "1 minute", "2m": "2 minutes", "5m": "5 minutes", "15m": "15 minutes",
    "30m": "30 minutes", "60m": "1 hour", "1d": "1 day", "1wk": "1 week", "1mo": "1 month",
}

# Human periods instead of typing calendar days. Clamped to what Yahoo will serve.
PERIOD_DAYS = {
    "1 week": 7, "1 month": 30, "3 months": 90, "6 months": 180,
    "1 year": 365, "2 years": 730, "5 years": 1825, "10 years": 3650,
    "20 years": 7300, "30 years": 10950, "Max available": 20000,
}


def resolve_period(period_label: str, interval: str) -> tuple[int, bool]:
    """Return (days, was_clamped) for the chosen period at this interval."""
    want = PERIOD_DAYS[period_label]
    cap = INTERVAL_MAX_DAYS[interval]
    return min(want, cap), want > cap

POINT_VALUE_HINT = {"^NSEI": 75, "^NSEBANK": 30, "^BSESN": 20}


# ========================= yfinance rate limiting =========================

_YF_LOCK = threading.Lock()
_YF_LAST = {"t": 0.0}
YF_MIN_GAP = 0.3  # seconds between yfinance calls


def yf_throttle(min_gap: float = YF_MIN_GAP):
    """Block until at least `min_gap` seconds have passed since the last call."""
    with _YF_LOCK:
        wait = min_gap - (time.time() - _YF_LAST["t"])
        if wait > 0:
            time.sleep(wait)
        _YF_LAST["t"] = time.time()


def _yf_download(symbol, interval, days, tries=3):
    last_err = None
    for attempt in range(tries):
        yf_throttle()
        try:
            df = yf.download(
                symbol,
                period=f"{int(days)}d",
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is not None and len(df):
                return df
            last_err = "empty response"
        except Exception as e:  # rate limit, network, symbol errors
            last_err = str(e)
        time.sleep(0.6 * (attempt + 1))  # back off, then retry
    raise RuntimeError(f"yfinance returned no data for {symbol} ({interval}): {last_err}")


@st.cache_data(show_spinner=False, ttl=900)
def fetch_history(symbol: str, interval: str, days: int, tz: str) -> pd.DataFrame:
    """Historical bars for backtesting. Cached for 15 minutes."""
    raw = _yf_download(symbol, interval, days)
    return normalise_yf(raw, tz)


@st.cache_data(show_spinner=False, ttl=20)
def fetch_live(symbol: str, interval: str, days: int, tz: str) -> pd.DataFrame:
    """Recent bars for the live tab. Cached for 20 seconds so auto-refresh
    cannot hammer the API even if the refresh interval is set very low."""
    raw = _yf_download(symbol, interval, days)
    return normalise_yf(raw, tz)


def normalise_yf(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance returns MultiIndex for 1 ticker too
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(tz)
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df.get("Volume", np.nan), errors="coerce")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def standardise_upload(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Map an uploaded file's columns onto Date/OHLCV. Exact-token matching only,
    so 'Open Interest' can never be mistaken for 'Open'."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    wanted = {
        "date": ["datetime", "date", "timestamp", "time", "trade_date"],
        "open": ["open", "o", "openprice", "open_price"],
        "high": ["high", "h", "highprice", "high_price"],
        "low": ["low", "l", "lowprice", "low_price"],
        "close": ["close", "adj close", "adj_close", "c", "last", "ltp", "closeprice"],
        "volume": ["volume", "vol", "qty", "quantity"],
    }
    lower = {c.lower(): c for c in df.columns}
    found = {}
    for key, aliases in wanted.items():
        for a in aliases:
            if a in lower:
                found[key] = lower[a]
                break
    if "date" not in found:
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                found["date"] = c
                break
    if "date" not in found or "close" not in found:
        raise ValueError(
            "Need at least a date column and a close column. "
            f"Found: {sorted(found)} in {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[found["date"]], errors="coerce", dayfirst=True)
    out["Close"] = pd.to_numeric(df[found["close"]], errors="coerce")
    for k, col in (("open", "Open"), ("high", "High"), ("low", "Low")):
        out[col] = pd.to_numeric(df[found[k]], errors="coerce") if k in found else out["Close"]
    out["Volume"] = pd.to_numeric(df[found["volume"]], errors="coerce") if "volume" in found else np.nan

    out = out.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    if out["Date"].dt.tz is None:
        out["Date"] = out["Date"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        out["Date"] = out["Date"].dt.tz_convert(tz)
    out = out.dropna(subset=["Date"])
    return out[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)


# ======================= Indicators (all causal) ==========================

def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Every indicator here reads only bars <= i. No centring, no negative shift.

    min_periods is set to the FULL window, unlike the original which used
    min_periods=1 everywhere. With min_periods=1 a "120-period moving average"
    equals the first close on bar 0, so the first ~120 bars produce confident
    looking signals computed from almost no data.
    """
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    v = pd.to_numeric(d.get("Volume"), errors="coerce")

    d["ema_fast"] = c.ewm(span=p["ema_fast"], adjust=False, min_periods=p["ema_fast"]).mean()
    d["ema_slow"] = c.ewm(span=p["ema_slow"], adjust=False, min_periods=p["ema_slow"]).mean()
    d["ma_short"] = c.rolling(p["short_ma"], min_periods=p["short_ma"]).mean()
    d["ma_long"] = c.rolling(p["long_ma"], min_periods=p["long_ma"]).mean()

    delta = c.diff()
    ru = wilder(delta.clip(lower=0), p["rsi_period"])
    rd = wilder(-delta.clip(upper=0), p["rsi_period"])
    d["rsi"] = 100 - 100 / (1 + ru / rd.replace(0, np.nan))
    d["rsi"] = d["rsi"].fillna(50.0)          # assign back; .fillna(inplace=True)
                                              # on a column is a no-op under
                                              # copy-on-write and silently did nothing
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean() - \
        c.ewm(span=26, adjust=False, min_periods=26).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    d["macd_hist_prev"] = d["macd_hist"].shift(1)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = wilder(tr, p["atr_period"])
    d["atr_pct"] = d["atr"] / c * 100

    # Proper Wilder ADX. The original zeroed negatives but never suppressed the
    # smaller of the two directional moves, so +DM and -DM both fired on the
    # same bar and ADX was meaningless.
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr14 = wilder(tr, 14)
    pdi = 100 * wilder(pd.Series(plus_dm, index=d.index), 14) / atr14.replace(0, np.nan)
    mdi = 100 * wilder(pd.Series(minus_dm, index=d.index), 14) / atr14.replace(0, np.nan)
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    d["adx"] = wilder(dx.fillna(0), 14)

    mid = c.rolling(p["bb_period"], min_periods=p["bb_period"]).mean()
    sd = c.rolling(p["bb_period"], min_periods=p["bb_period"]).std()
    d["bb_upper"] = mid + p["bb_k"] * sd
    d["bb_lower"] = mid - p["bb_k"] * sd

    d["vol_avg"] = v.rolling(p["vol_period"], min_periods=p["vol_period"]).mean()
    d["volume_num"] = v
    return d


# ========================== Confluence engine =============================

ALL_CONFLUENCES = ["ma_cross", "ema_trend", "macd_hist", "bb_breakout",
                   "rsi_zone", "adx_trend", "vol_spike", "atr_ok"]

DEFAULT_PARAMS = {
    "short_ma": 10, "long_ma": 50, "ema_fast": 9, "ema_slow": 21,
    "rsi_period": 14, "rsi_long_min": 50.0, "atr_period": 14,
    "bb_period": 20, "bb_k": 2.0, "vol_period": 20, "vol_spike_mult": 1.5,
    "adx_min": 20.0, "atr_min_pct": 0.10,
    "rule_mode": "any_k", "any_k": 3,
    "confluences": ["ma_cross", "ema_trend", "macd_hist", "rsi_zone", "adx_trend"],
    "sl_mode": "pct", "sl_pct": 0.008, "sl_atr_mult": 1.5,
    "tp_mode": "pct", "tp_pct": 0.016, "tp_atr_mult": 3.0,
    "max_hold": 12, "allowed_dirs": ["long", "short"],
}


def confluence_frame(d: pd.DataFrame, p: dict, direction: int) -> dict:
    """Boolean Series per confluence for one direction. Vectorised, causal."""
    s = direction  # +1 long, -1 short
    has_vol = d["volume_num"].notna().any() and d["vol_avg"].notna().any()
    f = {
        "ma_cross": (d["ma_short"] - d["ma_long"]) * s > 0,
        "ema_trend": (d["ema_fast"] - d["ema_slow"]) * s > 0,
        "macd_hist": (d["macd_hist"] * s > 0) & ((d["macd_hist"] - d["macd_hist_prev"]) * s > 0),
        "bb_breakout": (d["Close"] > d["bb_upper"]) if s > 0 else (d["Close"] < d["bb_lower"]),
        "rsi_zone": (d["rsi"] > p["rsi_long_min"]) if s > 0 else (d["rsi"] < 100 - p["rsi_long_min"]),
        "adx_trend": d["adx"] >= p["adx_min"],
        "atr_ok": d["atr_pct"] >= p["atr_min_pct"],
    }
    # If the feed carries no volume (most Yahoo index symbols), vol_spike is
    # dropped rather than evaluated as permanently False. The original left it
    # False, so strict mode could never fire a single trade on an index.
    if has_vol:
        f["vol_spike"] = d["volume_num"] >= p["vol_spike_mult"] * d["vol_avg"]
    return {k: v.fillna(False) for k, v in f.items()}


def generate_signals(df: pd.DataFrame, params: dict):
    """Score each bar from indicator confluences. Interface matches the rest of
    the app: adds score / signal / reason / sl_ref / tp_ref columns."""
    p = {**DEFAULT_PARAMS, **params}
    d = add_indicators(df, p)
    n = len(d)

    active = [k for k in p["confluences"] if k in ALL_CONFLUENCES]
    fl = confluence_frame(d, p, +1)
    fs = confluence_frame(d, p, -1)
    active = [k for k in active if k in fl]
    if not active:
        active = ["ma_cross"]

    hits_l = sum(fl[k].astype(int) for k in active)
    hits_s = sum(fs[k].astype(int) for k in active)
    need = len(active) if p["rule_mode"] == "strict" else min(p["any_k"], len(active))

    allow_l = "long" in p["allowed_dirs"]
    allow_s = "short" in p["allowed_dirs"]
    ok_l = (hits_l >= need) & allow_l
    ok_s = (hits_s >= need) & allow_s

    # Warm-up guard: no signal until every active indicator has a full window.
    warm = max(p["long_ma"], p["short_ma"], p["ema_slow"], p["bb_period"],
               p["vol_period"], 26 + 9, p["atr_period"] + 14)
    valid = np.zeros(n, dtype=bool)
    valid[warm:] = True

    sig = np.where(ok_l & ~ok_s & valid, 1, np.where(ok_s & ~ok_l & valid, -1, 0))
    tie = (ok_l & ok_s & valid).to_numpy()
    if tie.any():  # both sides qualify: take the side with more confluences
        sig = np.where(tie, np.sign((hits_l - hits_s).to_numpy()), sig)

    reasons = []
    hl, hs = hits_l.to_numpy(), hits_s.to_numpy()
    fl_np = {k: fl[k].to_numpy() for k in active}
    fs_np = {k: fs[k].to_numpy() for k in active}
    for i in range(n):
        if sig[i] == 0:
            reasons.append(f"{max(hl[i], hs[i])}/{len(active)} confluences, need {need}")
        else:
            src_ = fl_np if sig[i] == 1 else fs_np
            on = [k for k in active if src_[k][i]]
            reasons.append(f"{len(on)}/{len(active)}: " + ", ".join(on))

    out = df.copy()
    out["score"] = np.where(sig >= 0, hl, -hs).astype(float)
    out["signal"] = sig.astype(int)
    out["reason"] = reasons
    out["atr_ref"] = d["atr"].to_numpy()          # ATR of the SIGNAL bar, for SL/TP
    out["hits"] = np.where(sig >= 0, hl, hs)
    out["n_confluences"] = len(active)

    meta = {"active_confluences": active, "need_hits": int(need),
            "rule_mode": p["rule_mode"], "warmup_bars": int(warm),
            "volume_available": "vol_spike" in fl,
            "long_signals": int((sig == 1).sum()), "short_signals": int((sig == -1).sum())}
    return out, meta


# ================================ Backtest ================================

def backtest(df_sig: pd.DataFrame, params: dict, cost_bps: float = 3.0,
             slippage_pts: float = 0.0, intraday_squareoff: bool = True):
    """Bar-by-bar simulation.

    Rules, stated explicitly because they drive every number reported:
      1. A signal on bar N is acted on at the OPEN of bar N+1. Never bar N's close.
      2. Inside each bar the OPEN is checked first (a gap through a level fills
         at the open, not at the level).
      3. Then the STOP is checked before the TARGET. If a bar's range spans
         both, the loss is booked. This is the conservative assumption: without
         tick data you cannot know which came first.
      4. Position is flat-or-one. New signals during an open trade are ignored.
      5. Costs are charged on entry and exit.
    """
    p = {**DEFAULT_PARAMS, **params}
    o = df_sig["Open"].to_numpy(float)
    h = df_sig["High"].to_numpy(float)
    l = df_sig["Low"].to_numpy(float)
    c = df_sig["Close"].to_numpy(float)
    sig = df_sig["signal"].to_numpy(int)
    dates = df_sig["Date"].to_numpy()
    days = pd.Series(df_sig["Date"]).dt.date.to_numpy()
    reasons = df_sig["reason"].tolist()
    atr_ref = df_sig["atr_ref"].to_numpy(float) if "atr_ref" in df_sig else np.full(len(df_sig), np.nan)
    hits = df_sig["hits"].to_numpy() if "hits" in df_sig else np.zeros(len(df_sig))
    n_conf = int(df_sig["n_confluences"].iloc[0]) if "n_confluences" in df_sig else 0
    n = len(df_sig)

    trades = []
    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        d = int(sig[i])
        e = i + 1
        entry = o[e] + (slippage_pts * d)
        # ATR of the SIGNAL bar (i), not the entry bar. The entry bar's ATR is
        # only known at its close, which is after the fill.
        sl, tp = compute_levels(entry, d, p, atr_ref[i])

        last_allowed = min(n - 1, e + p["max_hold"])
        if intraday_squareoff:
            same_day = np.where(days[e:last_allowed + 1] == days[e])[0]
            if len(same_day):
                last_allowed = e + int(same_day[-1])

        exit_idx, exit_px, exit_reason = None, None, None
        for j in range(e, last_allowed + 1):
            if d == 1:
                if o[j] <= sl:                      # gapped through the stop
                    exit_idx, exit_px, exit_reason = j, o[j], "sl_gap"
                elif o[j] >= tp:
                    exit_idx, exit_px, exit_reason = j, o[j], "tp_gap"
                elif l[j] <= sl:                    # stop checked BEFORE target
                    exit_idx, exit_px, exit_reason = j, sl, "sl"
                elif h[j] >= tp:
                    exit_idx, exit_px, exit_reason = j, tp, "tp"
            else:
                if o[j] >= sl:
                    exit_idx, exit_px, exit_reason = j, o[j], "sl_gap"
                elif o[j] <= tp:
                    exit_idx, exit_px, exit_reason = j, o[j], "tp_gap"
                elif h[j] >= sl:
                    exit_idx, exit_px, exit_reason = j, sl, "sl"
                elif l[j] <= tp:
                    exit_idx, exit_px, exit_reason = j, tp, "tp"
            if exit_idx is not None:
                break

        if exit_idx is None:
            exit_idx, exit_px = last_allowed, c[last_allowed]
            exit_reason = "eod_squareoff" if intraday_squareoff and days[last_allowed] == days[e] \
                and last_allowed < e + p["max_hold"] else "time_exit"

        exit_px -= slippage_pts * d
        gross_pts = (exit_px - entry) * d
        cost_pts = (entry + exit_px) * (cost_bps / 10000.0)
        net_pts = gross_pts - cost_pts

        trades.append({
            "entry_time": dates[e], "exit_time": dates[exit_idx],
            "direction": "long" if d == 1 else "short",
            "entry": round(float(entry), 4), "sl": round(float(sl), 4), "tp": round(float(tp), 4),
            "exit": round(float(exit_px), 4), "exit_reason": exit_reason,
            "gross_points": round(float(gross_pts), 4),
            "cost_points": round(float(cost_pts), 4),
            "net_points": round(float(net_pts), 4),
            "net_pct": round(float(net_pts / entry * 100), 5),
            "bars_held": int(exit_idx - e),
            "confluences_hit": int(hits[i]), "confluences_total": n_conf,
            "signal_reason": reasons[i][:160],
        })
        i = exit_idx + 1

    return pd.DataFrame(trades), summarise(pd.DataFrame(trades))


def compute_levels(entry: float, d: int, p: dict, atr: float | None = None):
    """Stop and target from percent or ATR. Both are forced onto the correct
    side of entry; a zero or NaN ATR falls back to the percent rule instead of
    silently placing the stop on top of the entry."""
    a = float(atr) if atr is not None and np.isfinite(atr) and atr > 0 else None

    if p.get("sl_mode") == "atr" and a:
        sl = entry - a * p["sl_atr_mult"] * d
    else:
        sl = entry * (1 - p["sl_pct"] * d)

    if p.get("tp_mode") == "atr" and a:
        tp = entry + a * p["tp_atr_mult"] * d
    else:
        tp = entry * (1 + p["tp_pct"] * d)

    if (sl - entry) * d >= 0:
        sl = entry * (1 - max(p["sl_pct"], 1e-4) * d)
    if (tp - entry) * d <= 0:
        tp = entry * (1 + max(p["tp_pct"], 1e-4) * d)
    return float(sl), float(tp)


def summarise(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "net_points": 0.0, "expectancy_points": 0.0,
                "gross_points": 0.0, "costs": 0.0, "profit_factor": 0.0,
                "max_drawdown_points": 0.0, "avg_bars_held": 0.0, "sharpe_per_trade": 0.0}
    wins = t["net_points"] > 0
    gp = t.loc[wins, "net_points"].sum()
    gl = -t.loc[~wins, "net_points"].sum()
    eq = t["net_points"].cumsum()
    dd = (eq - eq.cummax()).min()
    sd = t["net_points"].std(ddof=1)
    return {
        "trades": int(len(t)),
        "win_rate": round(float(wins.mean()), 4),
        "gross_points": round(float(t["gross_points"].sum()), 2),
        "costs": round(float(t["cost_points"].sum()), 2),
        "net_points": round(float(t["net_points"].sum()), 2),
        "expectancy_points": round(float(t["net_points"].mean()), 4),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else float("inf"),
        "max_drawdown_points": round(float(dd), 2),
        "avg_bars_held": round(float(t["bars_held"].mean()), 2),
        "sharpe_per_trade": round(float(t["net_points"].mean() / sd), 3) if sd and sd > 0 else 0.0,
    }


# ===================== Optimiser with a real holdout ======================

PARAM_SPACE = {
    "short_ma": [5, 10, 15, 20, 30],
    "long_ma": [30, 50, 80, 100, 120],
    "ema_fast": [5, 9, 12],
    "ema_slow": [21, 26, 34],
    "rsi_period": [14],
    "rsi_long_min": [45.0, 50.0, 55.0, 60.0],
    "atr_period": [14],
    "bb_period": [14, 20],
    "bb_k": [2.0, 2.5],
    "vol_period": [10, 20],
    "vol_spike_mult": [1.2, 1.5, 2.0],
    "adx_min": [15.0, 20.0, 25.0],
    "atr_min_pct": [0.05, 0.10, 0.20],
    "rule_mode": ["strict", "any_k"],
    "sl_mode": ["pct", "atr"],
    "tp_mode": ["pct", "atr"],
    "sl_pct": [0.004, 0.006, 0.010, 0.015],
    "tp_pct": [0.008, 0.012, 0.020, 0.030],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, 4.0],
    "max_hold": [5, 10, 20, 40],
}

CONFLUENCE_POOL = [
    ["ma_cross", "ema_trend", "macd_hist"],
    ["ma_cross", "ema_trend", "macd_hist", "rsi_zone"],
    ["ma_cross", "ema_trend", "macd_hist", "adx_trend"],
    ["ma_cross", "ema_trend", "macd_hist", "rsi_zone", "adx_trend"],
    ["ma_cross", "ema_trend", "macd_hist", "rsi_zone", "adx_trend", "atr_ok"],
    ["ma_cross", "ema_trend", "macd_hist", "bb_breakout", "rsi_zone", "adx_trend"],
    ["ema_trend", "macd_hist", "rsi_zone", "adx_trend", "vol_spike"],
    ["ma_cross", "ema_trend", "macd_hist", "bb_breakout", "rsi_zone", "adx_trend",
     "vol_spike", "atr_ok"],
]


def sample_params(rng, allowed_dirs):
    p = {k: rng.choice(v).item() for k, v in PARAM_SPACE.items()}
    p["confluences"] = list(CONFLUENCE_POOL[int(rng.integers(len(CONFLUENCE_POOL)))])
    p["any_k"] = int(rng.integers(2, len(p["confluences"]) + 1))
    p["allowed_dirs"] = allowed_dirs
    if p["short_ma"] >= p["long_ma"]:          # a "short" MA slower than the long
        p["short_ma"] = max(5, p["long_ma"] // 4)   # one makes the cross meaningless
    if p["ema_fast"] >= p["ema_slow"]:
        p["ema_fast"] = max(3, p["ema_slow"] // 2)
    return p


def score_params(stats: dict, min_trades: int) -> float:
    """Reject outright instead of scaling. The original multiplied the score by
    0.4 when trade count was too low, which for a NEGATIVE net PnL makes the
    number LARGER: -1000 over 2 trades scored -600 and beat -500 over 50 trades,
    so the optimiser actively preferred the worse, under-traded parameter set.
    """
    if stats["trades"] < min_trades:
        return -1e9
    if stats["expectancy_points"] <= 0:
        return -1e9
    return stats["sharpe_per_trade"] * np.sqrt(stats["trades"])


def optimise(df: pd.DataFrame, n_iter: int, allowed_dirs: list, min_trades: int,
             cost_bps: float, slippage_pts: float, intraday: bool,
             train_frac: float = 0.7, seed: int = 7, progress=None):
    """Random search fitted on the train slice, then scored once on a purged
    holdout the search never touched."""
    rng = np.random.default_rng(seed)
    n = len(df)
    cut = int(n * train_frac)
    purge = max(PARAM_SPACE["max_hold"]) + max(PARAM_SPACE["long_ma"]) + 5
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[min(cut + purge, n - 1):].reset_index(drop=True)

    best, results = None, []
    for it in range(n_iter):
        p = sample_params(rng, allowed_dirs)
        if p["tp_mode"] == "pct" and p["sl_mode"] == "pct" and p["tp_pct"] <= p["sl_pct"] * 0.6:
            continue  # reward:risk below 0.6 is a win-rate trap, skip it
        if p["tp_mode"] == "atr" and p["sl_mode"] == "atr" and \
                p["tp_atr_mult"] <= p["sl_atr_mult"] * 0.6:
            continue
        try:
            sig, _ = generate_signals(train, p)
            _, stats = backtest(sig, p, cost_bps, slippage_pts, intraday)
        except Exception:
            continue
        sc = score_params(stats, min_trades)
        results.append({"score": round(sc, 3), **{k: stats[k] for k in
                        ("trades", "win_rate", "net_points", "expectancy_points", "profit_factor")},
                        "params": p})
        if best is None or sc > best["score"]:
            best = {"score": sc, "params": p, "train_stats": stats}
        if progress:
            progress(it + 1, n_iter)

    if best is None or best["score"] <= -1e8:
        return None, pd.DataFrame(results)

    if len(test) > 50:
        sig_t, _ = generate_signals(test, best["params"])
        tt, ts = backtest(sig_t, best["params"], cost_bps, slippage_pts, intraday)
        best["test_stats"], best["test_trades"] = ts, tt
    else:
        best["test_stats"], best["test_trades"] = None, pd.DataFrame()

    sig_f, meta = generate_signals(df, best["params"])
    ft, fs = backtest(sig_f, best["params"], cost_bps, slippage_pts, intraday)
    best.update({"full_stats": fs, "full_trades": ft, "meta": meta, "signals": sig_f,
                 "train_rows": cut, "test_rows": len(test)})
    return best, pd.DataFrame(results).sort_values("score", ascending=False)


def leakage_self_test(df: pd.DataFrame, params: dict, cost_bps: float,
                      slippage_pts: float, intraday: bool, n_runs: int = 6):
    """Run the winning parameters over synthetic random walks calibrated to the
    real series' volatility. A random walk has no edge by construction, so any
    win rate materially above 50% means future information is leaking in."""
    r = np.log(df["Close"]).diff().dropna()
    sigma = float(r.std()) or 1e-4
    start = float(df["Close"].iloc[0])
    n = len(df)
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n_runs):
        ret = rng.normal(0, sigma, n)
        close = start * np.exp(np.cumsum(ret))
        openp = np.concatenate([[start], close[:-1]])
        wig = np.abs(rng.normal(0, sigma * 0.6, (n, 2))) * close[:, None]
        fake = pd.DataFrame({
            "Date": df["Date"].reset_index(drop=True),
            "Open": openp,
            "High": np.maximum(openp, close) + wig[:, 0],
            "Low": np.minimum(openp, close) - wig[:, 1],
            "Close": close,
            "Volume": rng.integers(1000, 20000, n).astype(float),
        })
        s, _ = generate_signals(fake, params)
        _, st_ = backtest(s, params, cost_bps, slippage_pts, intraday)
        if st_["trades"] > 0:
            out.append((st_["win_rate"], st_["expectancy_points"], st_["trades"]))
    if not out:
        return None
    arr = np.array(out)
    return {"mean_win_rate": float(arr[:, 0].mean()),
            "mean_expectancy_points": float(arr[:, 1].mean()),
            "mean_trades": float(arr[:, 2].mean()),
            "runs": len(out)}


# ===================== Reporting: EDA, summary, recommendation ============

def detailed_stats(t: pd.DataFrame, qty: int) -> pd.DataFrame:
    """Full performance table for the backtest tab."""
    if t.empty:
        return pd.DataFrame([{"Metric": "Trades", "Value": 0}])
    w = t[t["net_points"] > 0]
    l = t[t["net_points"] <= 0]
    eq = t["net_points"].cumsum()
    dd = (eq - eq.cummax())
    sd = t["net_points"].std(ddof=1)
    gp, gl = w["net_points"].sum(), -l["net_points"].sum()
    avg_w = w["net_points"].mean() if len(w) else 0.0
    avg_l = l["net_points"].mean() if len(l) else 0.0
    streak = cur = best = worst = 0
    for x in t["net_points"] > 0:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    cur = 0
    for x in t["net_points"] <= 0:
        cur = cur + 1 if x else 0
        worst = max(worst, cur)
    rows = [
        ("Total trades", f"{len(t):,}"),
        ("Winning trades", f"{len(w):,}"),
        ("Losing trades", f"{len(l):,}"),
        ("Accuracy (win rate)", f"{len(w)/len(t)*100:.2f}%"),
        ("Points won (sum of winners)", f"{gp:+,.2f}"),
        ("Points lost (sum of losers)", f"{-gl:+,.2f}"),
        ("Net points BEFORE charges", f"{t['gross_points'].sum():+,.2f}"),
        ("Charges paid (cost + slippage)", f"{-t['cost_points'].sum():,.2f}"),
        ("Net points AFTER charges", f"{t['net_points'].sum():+,.2f}"),
        (f"Net cash (qty {qty})", f"{t['net_points'].sum()*qty:+,.2f}"),
        ("Expectancy per trade", f"{t['net_points'].mean():+,.3f} pts"),
        ("Average win", f"{avg_w:+,.2f} pts"),
        ("Average loss", f"{avg_l:+,.2f} pts"),
        ("Reward:risk realised", f"{abs(avg_w/avg_l):.2f}" if avg_l else "n/a"),
        ("Profit factor", f"{gp/gl:.3f}" if gl > 0 else "inf"),
        ("Sharpe per trade", f"{t['net_points'].mean()/sd:.3f}" if sd else "n/a"),
        ("Sharpe annualised (approx)", annual_sharpe(t)),
        ("Max drawdown", f"{dd.min():,.2f} pts"),
        ("Longest win streak", f"{best}"),
        ("Longest loss streak", f"{worst}"),
        ("Average bars held", f"{t['bars_held'].mean():.2f}"),
        ("Exit mix", ", ".join(f"{k} {v}" for k, v in t["exit_reason"].value_counts().items())),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def annual_sharpe(t: pd.DataFrame) -> str:
    """Scale per-trade Sharpe by the real trade frequency, not a hardcoded 252.
    The original script annualised 1-minute returns with sqrt(252), which is
    why it reported nonsense like -0.04% annualised."""
    try:
        span_days = (pd.Timestamp(t["exit_time"].iloc[-1]) - pd.Timestamp(t["entry_time"].iloc[0])).days
        if span_days < 1:
            return "span too short"
        per_year = len(t) / span_days * 365.0
        sd = t["net_points"].std(ddof=1)
        if not sd:
            return "n/a"
        return f"{t['net_points'].mean()/sd*np.sqrt(per_year):.2f}"
    except Exception:
        return "n/a"


def returns_heatmap(df: pd.DataFrame):
    """Month-by-year returns when the data spans months, otherwise weekday-by-hour.
    A year/month heatmap over 8 days of intraday data is a single meaningless cell,
    which is what the original always drew."""
    d = df.copy()
    d["ret"] = d["Close"].pct_change()
    span_days = (d["Date"].iloc[-1] - d["Date"].iloc[0]).days
    if span_days > 75:
        d["Y"], d["M"] = d["Date"].dt.year, d["Date"].dt.month
        piv = d.groupby(["Y", "M"])["ret"].apply(lambda s: (1 + s).prod() - 1).unstack() * 100
        return piv, "Monthly return %", "Month", "Year"
    d["W"] = d["Date"].dt.day_name().str[:3]
    d["H"] = d["Date"].dt.hour
    piv = d.groupby(["W", "H"])["ret"].mean().unstack() * 100
    order = [x for x in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if x in piv.index]
    return piv.loc[order], "Mean return % per bar", "Hour of day", "Weekday"


def draw_heatmap(piv, title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(11, max(2.0, 0.45 * len(piv) + 1.2)))
    vals = piv.to_numpy(float)
    lim = np.nanmax(np.abs(vals)) or 1.0
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    return fig


def written_summary(df: pd.DataFrame, interval: str) -> str:
    c = df["Close"]
    r = c.pct_change().dropna()
    bars_per_year = {"1m": 98280, "2m": 49140, "5m": 19656, "15m": 6552, "30m": 3276,
                     "60m": 1638, "1d": 252, "1wk": 52, "1mo": 12}.get(interval, 252)
    ann_vol = r.std() * np.sqrt(bars_per_year) * 100
    trend = "up" if c.iloc[-1] > c.rolling(min(50, len(c))).mean().iloc[-1] else "down"
    hi, lo = c.max(), c.min()
    pos = (c.iloc[-1] - lo) / (hi - lo) * 100 if hi > lo else 50
    net = c.iloc[-1] - c.iloc[0]
    return (
        f"{len(df):,} bars of {INTERVAL_LABELS.get(interval, interval)} data from "
        f"{df['Date'].iloc[0]:%d %b %Y} to {df['Date'].iloc[-1]:%d %b %Y}. "
        f"Last close {c.iloc[-1]:,.2f}, which sits {pos:.0f}% of the way up the "
        f"{lo:,.2f}–{hi:,.2f} range. Price is trading {trend} relative to its moving average, "
        f"and buy-and-hold over this window returned {net:+,.2f} points "
        f"({net/c.iloc[0]*100:+.2f}%). Annualised volatility is roughly {ann_vol:.1f}%, "
        f"scaled using {bars_per_year:,} bars per year for this interval. "
        f"Wider ranges favour breakout entries; tighter ones favour fading the edges of "
        f"support and resistance. Size positions off the stop distance, not conviction."
    )


def benchmark_table(df: pd.DataFrame, trades: pd.DataFrame, qty: int, label: str) -> pd.DataFrame:
    """Strategy vs buy-and-hold over the SAME bars the strategy was measured on.

    Deliberately reports the difference in points, never a ratio. The original
    script divided by the absolute buy-and-hold points and printed things like
    "1393% more points" — when buy-and-hold is near zero that ratio explodes and
    means nothing.
    """
    first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    bh_pts = last - first
    bh_pct = bh_pts / first * 100

    gross = float(trades["gross_points"].sum()) if not trades.empty else 0.0
    charges = float(trades["cost_points"].sum()) if not trades.empty else 0.0
    net = float(trades["net_points"].sum()) if not trades.empty else 0.0
    bars_in = int(trades["bars_held"].sum()) if not trades.empty else 0
    exposure = bars_in / len(df) * 100 if len(df) else 0.0

    if not trades.empty:
        eq = trades["net_points"].cumsum()
        s_dd = float((eq - eq.cummax()).min())
    else:
        s_dd = 0.0
    curve = df["Close"] - first
    bh_dd = float((curve - curve.cummax()).min())

    rows = [
        ("Period", f"{df['Date'].iloc[0]:%d %b %Y} → {df['Date'].iloc[-1]:%d %b %Y}",
         f"{df['Date'].iloc[0]:%d %b %Y} → {df['Date'].iloc[-1]:%d %b %Y}"),
        ("Bars", f"{len(df):,}", f"{len(df):,}"),
        ("Trades", f"{len(trades):,}", "1"),
        ("Time in market", f"{exposure:.1f}%", "100.0%"),
        ("Points before charges", f"{gross:+,.2f}", f"{bh_pts:+,.2f}"),
        ("Charges", f"{-charges:,.2f}", "0.00"),
        ("Points after charges", f"{net:+,.2f}", f"{bh_pts:+,.2f}"),
        ("Return on first close", f"{net/first*100:+.2f}%", f"{bh_pct:+.2f}%"),
        (f"Cash at qty {qty}", f"{net*qty:+,.2f}", f"{bh_pts*qty:+,.2f}"),
        ("Max drawdown (points)", f"{s_dd:,.2f}", f"{bh_dd:,.2f}"),
    ]
    out = pd.DataFrame(rows, columns=["Metric", f"Strategy ({label})", "Buy and hold"])
    edge = net - bh_pts
    out.loc[len(out)] = ["Strategy minus buy-and-hold",
                         f"{edge:+,.2f} pts", f"{edge*qty:+,.2f} cash"]
    out.columns = ["Metric", f"Strategy ({label})", "Buy and hold"]
    return out



def recommendation(df_sig: pd.DataFrame, params: dict, hold_stats: dict,
                   capital: float, risk_pct: float, qty: int) -> dict:
    """Live call from the last CLOSED bar, using the fitted rule set."""
    last = df_sig.iloc[-1]
    d = int(last["signal"])
    if d == 0:
        return {"direction": "flat", "reason": last["reason"],
                "confluences_hit": int(last["hits"]),
                "confluences_total": int(last["n_confluences"])}
    entry = float(last["Close"])
    sl, tp = compute_levels(entry, d, params, float(last["atr_ref"]))
    risk = abs(entry - sl)
    return {
        "direction": "long" if d == 1 else "short",
        "signal_bar": f"{pd.Timestamp(last['Date']):%d %b %Y %H:%M}",
        "reference_price": round(entry, 2),
        "note": "Backtest fills at the NEXT bar's open, so treat this as a next-bar order.",
        "stop_loss": round(sl, 2), "target": round(tp, 2),
        "sl_basis": f"{params['sl_atr_mult']}x ATR({params['atr_period']})"
                    if params.get("sl_mode") == "atr" else f"{params['sl_pct']*100:.2f}% of entry",
        "tp_basis": f"{params['tp_atr_mult']}x ATR({params['atr_period']})"
                    if params.get("tp_mode") == "atr" else f"{params['tp_pct']*100:.2f}% of entry",
        "risk_points": round(risk, 2), "reward_points": round(abs(tp - entry), 2),
        "reward_risk": round(abs(tp - entry) / risk, 2) if risk else None,
        "confluences_hit": int(last["hits"]), "confluences_total": int(last["n_confluences"]),
        "rule": f"{params['rule_mode']} (need {params.get('any_k')})",
        "units_by_risk": int((capital * risk_pct / 100) // risk) if risk else 0,
        "sidebar_qty": qty,
        "reason": last["reason"][:300],
        "holdout_win_rate": hold_stats["win_rate"],
        "holdout_expectancy_points": hold_stats["expectancy_points"],
    }


# ============================== Live engine ===============================

def blank_live_state():
    return {
        "running": False,
        "position": None,      # open trade dict
        "pending": None,       # signal waiting for the next bar's open
        "trades": [],          # closed live trades
        "last_bar": None,      # timestamp of the last bar processed
        "log": [],
    }


def live_state():
    if "live" not in st.session_state:
        st.session_state.live = blank_live_state()
    return st.session_state.live


def log(msg: str):
    ls = live_state()
    ls["log"].insert(0, f"{pd.Timestamp.now(tz='Asia/Kolkata'):%H:%M:%S} — {msg}")
    ls["log"] = ls["log"][:60]


def step_live(df: pd.DataFrame, params: dict, cost_bps: float, slippage_pts: float,
              intraday: bool, drop_forming_bar: bool = True):
    """Advance the paper-trading state machine by whatever bars have closed
    since the last call. Mirrors the backtest exactly:
      signal on a CLOSED bar -> order pending -> filled at the NEXT bar's open
      -> stop checked before target on every subsequent closed bar.
    """
    ls = live_state()
    if len(df) < 60:
        return
    work = df.iloc[:-1].reset_index(drop=True) if drop_forming_bar else df.reset_index(drop=True)
    sig_df, _ = generate_signals(work, params)

    ts = pd.Series(sig_df["Date"])

    # First call after Start: mark where we are and trade FORWARD only.
    # Without this the engine replays the entire history buffer in one tick and
    # reports hundreds of instant "live" trades that never actually happened.
    if ls["last_bar"] is None:
        ls["last_bar"] = sig_df["Date"].iloc[-1]
        log(f"Armed at bar {pd.Timestamp(ls['last_bar']):%d %b %H:%M}. "
            f"Trading forward from the next closed bar.")
        return

    newer = np.flatnonzero((ts > pd.Timestamp(ls["last_bar"])).to_numpy())
    if len(newer) == 0:
        return
    start = int(newer[0])

    for i in range(start, len(sig_df)):
        row = sig_df.iloc[i]
        bar_t = row["Date"]
        ls["last_bar"] = bar_t

        # 1) fill any pending order at this bar's open
        if ls["pending"] is not None and ls["position"] is None:
            d = ls["pending"]["dir"]
            entry = float(row["Open"]) + slippage_pts * d
            sl, tp = compute_levels(entry, d, params, ls["pending"].get("atr"))
            ls["position"] = {
                "direction": "long" if d == 1 else "short", "dir": d,
                "entry_time": bar_t, "entry": entry, "sl": sl, "tp": tp,
                "reason": ls["pending"]["reason"], "bars": 0,
                "hits": ls["pending"].get("hits", 0),
                "params_used": {k: params.get(k) for k in
                                ("rule_mode", "any_k", "confluences", "sl_mode", "sl_pct",
                                 "sl_atr_mult", "tp_mode", "tp_pct", "tp_atr_mult",
                                 "max_hold", "short_ma", "long_ma", "adx_min")},
            }
            ls["pending"] = None
            log(f"ENTRY {ls['position']['direction']} @ {entry:.2f} | SL {sl:.2f} | TP {tp:.2f}")

        # 2) manage an open position on this closed bar
        elif ls["position"] is not None:
            pos = ls["position"]
            d, sl, tp = pos["dir"], pos["sl"], pos["tp"]
            o, hi, lo, cl = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            px = rsn = None
            if d == 1:
                if o <= sl: px, rsn = o, "sl_gap"
                elif o >= tp: px, rsn = o, "tp_gap"
                elif lo <= sl: px, rsn = sl, "sl"
                elif hi >= tp: px, rsn = tp, "tp"
            else:
                if o >= sl: px, rsn = o, "sl_gap"
                elif o <= tp: px, rsn = o, "tp_gap"
                elif hi >= sl: px, rsn = sl, "sl"
                elif lo <= tp: px, rsn = tp, "tp"

            pos["bars"] += 1
            new_day = pd.Timestamp(bar_t).date() != pd.Timestamp(pos["entry_time"]).date()
            if px is None and pos["bars"] >= params["max_hold"]:
                px, rsn = cl, "time_exit"
            if px is None and intraday and new_day:
                px, rsn = o, "eod_squareoff"
            if px is not None:
                close_position(px, rsn, bar_t, cost_bps, slippage_pts)

        # 3) look for a fresh signal on this closed bar
        if ls["position"] is None and ls["pending"] is None and int(row["signal"]) != 0:
            ls["pending"] = {"dir": int(row["signal"]), "reason": row["reason"],
                             "signal_time": bar_t, "atr": float(row.get("atr_ref", np.nan)),
                             "hits": int(row.get("hits", 0))}
            log(f"SIGNAL {'long' if row['signal'] == 1 else 'short'} on bar {pd.Timestamp(bar_t):%H:%M} "
                f"— fills at next bar's open")


def close_position(px: float, reason: str, when, cost_bps: float, slippage_pts: float):
    ls = live_state()
    pos = ls["position"]
    if pos is None:
        return
    d = pos["dir"]
    exit_px = px - slippage_pts * d
    gross = (exit_px - pos["entry"]) * d
    cost = (pos["entry"] + exit_px) * (cost_bps / 10000.0)
    ls["trades"].append({
        "entry_time": pos["entry_time"], "exit_time": when,
        "direction": pos["direction"], "entry": round(pos["entry"], 2),
        "sl": round(pos["sl"], 2), "tp": round(pos["tp"], 2),
        "exit": round(exit_px, 2), "exit_reason": reason,
        "gross_points": round(gross, 2), "cost_points": round(cost, 2),
        "net_points": round(gross - cost, 2),
        "net_pct": round((gross - cost) / pos["entry"] * 100, 4),
        "bars_held": pos["bars"], "confluences_hit": pos.get("hits", 0),
        "signal_reason": pos["reason"][:160],
    })
    log(f"EXIT {reason} @ {exit_px:.2f} | net {gross - cost:+.2f} pts")
    ls["position"] = None


def live_stats() -> dict:
    ls = live_state()
    return summarise(pd.DataFrame(ls["trades"]))


# ================================== UI ====================================

st.set_page_config(page_title="Confluence Trader", layout="wide")

SL_CHOICES = [0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]
TP_CHOICES = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]


def nearest(choices, value):
    return int(np.argmin([abs(c - value) for c in choices]))


def load_data(sidebar_cfg):
    src, tz = sidebar_cfg["source"], sidebar_cfg["tz"]
    if src == "yfinance":
        return fetch_history(sidebar_cfg["symbol"], sidebar_cfg["interval"],
                             sidebar_cfg["days"], tz), sidebar_cfg["symbol"]
    up = sidebar_cfg["upload"]
    if up is None:
        return None, None
    raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    return standardise_upload(raw, tz), up.name


def sidebar():
    with st.sidebar:
        st.header("Data")
        use_yf = st.checkbox("Use yfinance", value=True,
                             help="Uncheck to backtest an uploaded file instead.")
        if use_yf and not YF_AVAILABLE:
            st.error("yfinance is not installed. Run: pip install yfinance")
            use_yf = False

        symbol, upload = None, None
        if use_yf:
            name = st.selectbox("Instrument", list(TICKERS.keys()), index=0)
            symbol = TICKERS[name]
            if name == "Custom":
                symbol = st.text_input("Ticker symbol", value="KAYNES.NS",
                                       help="Yahoo format. NSE cash: SYMBOL.NS · BSE: SYMBOL.BO")
            st.caption(f"Requesting `{symbol}` · 0.3s enforced between API calls")
        else:
            upload = st.file_uploader("OHLC file", type=["csv", "xlsx", "xls"])

        interval = st.selectbox("Bar interval", list(INTERVAL_MAX_DAYS.keys()), index=3,
                                format_func=lambda k: INTERVAL_LABELS[k])
        period = st.selectbox("History", list(PERIOD_DAYS.keys()), index=4)
        days, clamped = resolve_period(period, interval)
        if clamped:
            st.caption(f"Yahoo serves at most {days} days of {INTERVAL_LABELS[interval]} bars. "
                       f"Using {days} days. Pick a longer bar interval for more history.")
        else:
            st.caption(f"{days} calendar days of {INTERVAL_LABELS[interval]} bars.")

        st.header("Execution")
        side = st.selectbox("Allowed side", ["both", "long only", "short only"])
        dirs = {"both": ["long", "short"], "long only": ["long"], "short only": ["short"]}[side]
        qty = st.number_input("Quantity per trade", 1, 1_000_000, 1, 1,
                              help="Lots or shares. Points are multiplied by this for cash P&L.")
        capital = st.number_input("Capital", 0.0, 1e12, 500000.0, 10000.0)
        risk_pct = st.number_input("Risk per trade (% of capital)", 0.1, 20.0, 1.0, 0.1)
        cost_bps = st.number_input("Cost per leg (bps of turnover)", 0.0, 100.0, 3.0, 0.5,
                                   help="Charged on entry AND exit, so 3 here means 6 bps "
                                        "round trip, about 14.5 points on NIFTY at 24,000. "
                                        "On Indian index futures STT alone is 2 bps on the sell leg.")
        slippage = st.number_input("Slippage (points per leg)", 0.0, 100.0, 0.5, 0.25)
        intraday = st.checkbox("Square off at session end", value=True,
                               help="Turn this OFF for crypto and forex, which trade 24/7.")

        st.header("Optimiser")
        n_iter = st.number_input("Random search iterations", 20, 2000, 150, 10)
        min_trades = st.number_input("Minimum trades to accept", 5, 1000, 30, 5)
        train_frac = st.slider("Train fraction", 0.5, 0.9, 0.7, 0.05,
                               help="The rest is a purged holdout the search never sees.")
        tz = "Asia/Kolkata"
    return dict(source="yfinance" if use_yf else "upload", symbol=symbol, upload=upload,
                interval=interval, period=period, days=int(days), dirs=dirs,
                qty=int(qty), capital=float(capital), risk_pct=float(risk_pct),
                cost_bps=cost_bps, slippage=slippage, intraday=intraday,
                n_iter=int(n_iter), min_trades=int(min_trades),
                train_frac=train_frac, tz=tz)


def tab_backtest(cfg, df, label):
    st.subheader(f"{label} · {len(df):,} bars · {df['Date'].min():%d %b %Y %H:%M} → {df['Date'].max():%d %b %Y %H:%M}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Last close", f"{df['Close'].iloc[-1]:,.2f}")
    c2.metric("Range", f"{df['Close'].min():,.2f} – {df['Close'].max():,.2f}")
    c3.metric("Buy & hold", f"{df['Close'].iloc[-1] - df['Close'].iloc[0]:+,.2f} pts")

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(df["Date"], df["Close"], lw=0.9)
    ax.set_title("Close"); ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Exploratory data analysis", expanded=False):
        st.markdown(written_summary(df, cfg["interval"]))
        try:
            piv, title, xlab, ylab = returns_heatmap(df)
            if piv.size and not piv.isna().all().all():
                st.pyplot(draw_heatmap(piv, title, xlab, ylab), clear_figure=True)
            else:
                st.caption("Not enough data spread to draw a meaningful heatmap.")
        except Exception as e:
            st.caption(f"Heatmap unavailable: {e}")
        r = df["Close"].pct_change().dropna() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean bar return", f"{r.mean():+.4f}%")
        c2.metric("Bar volatility", f"{r.std():.4f}%")
        c3.metric("Best bar", f"{r.max():+.2f}%")
        c4.metric("Worst bar", f"{r.min():+.2f}%")

    if st.button("Run optimisation", type="primary"):
        bar, txt = st.progress(0), st.empty()
        best, table = optimise(
            df, cfg["n_iter"], cfg["dirs"], cfg["min_trades"], cfg["cost_bps"],
            cfg["slippage"], cfg["intraday"], cfg["train_frac"],
            progress=lambda a, b: (bar.progress(a / b), txt.text(f"Tested {a}/{b}")))
        bar.empty(); txt.empty()
        if best is None:
            st.warning("No parameter set cleared the minimum-trades and positive-expectancy "
                       "filters. Widen the history, lower the minimum trades, or accept that "
                       "this instrument and interval has no edge for this rule set.")
            st.session_state.pop("best", None)
            return
        st.session_state.best = best
        st.session_state.best_table = table

    best = st.session_state.get("best")
    if not best:
        st.info("Run the optimisation to fit parameters and unlock live trading.")
        return

    st.markdown("### Train vs holdout")
    tr, te = best["train_stats"], best["test_stats"]
    cols = st.columns(4)
    cols[0].metric("Train win rate", f"{tr['win_rate']*100:.1f}%")
    cols[1].metric("Train net", f"{tr['net_points']:+,.0f} pts", f"{tr['trades']} trades")
    if te and te["trades"]:
        cols[2].metric("Holdout win rate", f"{te['win_rate']*100:.1f}%",
                       f"{(te['win_rate']-tr['win_rate'])*100:+.1f} pp")
        cols[3].metric("Holdout net", f"{te['net_points']:+,.0f} pts", f"{te['trades']} trades")
        if te["net_points"] <= 0:
            st.error("Holdout is unprofitable. The train numbers are curve fit — do not trade this.")
        elif te["win_rate"] < tr["win_rate"] - 0.15:
            st.warning("Holdout win rate is far below train. Treat the edge as unproven.")
    else:
        cols[2].metric("Holdout", "too short")

    st.markdown("### Leakage self-test")
    st.caption("The same parameters run on random walks. Anything meaningfully above 50% "
               "means the strategy is reading the future, not predicting it.")
    lk = leakage_self_test(df, best["params"], cfg["cost_bps"], cfg["slippage"], cfg["intraday"])
    if lk:
        a, b = st.columns(2)
        a.metric("Win rate on random data", f"{lk['mean_win_rate']*100:.1f}%")
        b.metric("Expectancy on random data", f"{lk['mean_expectancy_points']:+.3f} pts")
        if lk["mean_win_rate"] > 0.56:
            st.error("Look-ahead bias detected. Do not trust any result on this page.")
        else:
            st.success("No look-ahead bias detected — random data scores like a coin flip.")

    st.markdown("### Backtest summary")
    scope = st.radio("Scope", ["Holdout only", "Full sample"], horizontal=True,
                     index=0 if (te and te["trades"]) else 1, label_visibility="collapsed")
    tbl = best["test_trades"] if scope == "Holdout only" and not best["test_trades"].empty \
        else best["full_trades"]
    st.dataframe(detailed_stats(tbl, cfg["qty"]), use_container_width=True, hide_index=True)
    if scope == "Full sample":
        st.caption("Full sample includes the fitted window, so it flatters the strategy. "
                   "Judge it on the holdout.")

    st.markdown("### Live recommendation")
    st.caption("Read from the last closed bar using the fitted rule set.")
    rec = recommendation(best["signals"], best["params"],
                         best["test_stats"] or best["train_stats"],
                         cfg["capital"], cfg["risk_pct"], cfg["qty"])
    if rec and rec["direction"] == "flat":
        st.info(f"No trade on the last closed bar — {rec['reason']}.")
    elif rec:
        r1 = st.columns(5)
        r1[0].metric("Direction", rec["direction"].upper())
        r1[1].metric("Reference price", f"{rec['reference_price']:,.2f}")
        r1[2].metric("Stop loss", f"{rec['stop_loss']:,.2f}", f"-{rec['risk_points']:,.2f} pts")
        r1[3].metric("Target", f"{rec['target']:,.2f}", f"+{rec['reward_points']:,.2f} pts")
        r1[4].metric("Reward:risk", f"{rec['reward_risk']}")
        st.caption(f"Stop basis: {rec['sl_basis']} · Target basis: {rec['tp_basis']} · "
                   f"Rule: {rec['rule']} · "
                   f"{rec['confluences_hit']}/{rec['confluences_total']} confluences")
        r2 = st.columns(3)
        r2[0].metric("Units by risk budget", f"{rec['units_by_risk']:,}",
                     help=f"{cfg['risk_pct']}% of {cfg['capital']:,.0f} divided by the stop distance.")
        r2[1].metric("Sidebar quantity", f"{cfg['qty']:,}")
        r2[2].metric("Holdout win rate", f"{rec['holdout_win_rate']*100:.1f}%")
        st.caption(rec["note"])
        with st.expander("Full recommendation"):
            st.json(rec)

    st.markdown("### Backtest trades")
    show = best["test_trades"] if scope == "Holdout only" and not best["test_trades"].empty \
        else best["full_trades"]
    if show.empty:
        st.caption("No trades in this scope.")
    else:
        sc = show.copy()
        sc["net_cash"] = (sc["net_points"] * cfg["qty"]).round(2)
        st.dataframe(sc.iloc[::-1], use_container_width=True)
        st.download_button("Download backtest trades (CSV)", sc.to_csv(index=False),
                           "backtest_trades.csv", "text/csv")
        st.bar_chart(show["exit_reason"].value_counts())

    st.markdown("### Strategy vs buy and hold")
    if scope == "Holdout only" and best.get("test_rows", 0) > 1:
        bench_df = df.iloc[-int(best["test_rows"]):].reset_index(drop=True)
        bench_label = "holdout"
    else:
        bench_df, bench_label = df, "full sample"
    st.dataframe(benchmark_table(bench_df, show, cfg["qty"], bench_label),
                 use_container_width=True, hide_index=True)
    st.caption("Compared over the same bars. Time in market matters: a strategy that is "
               "flat 90% of the time carries far less risk than buy-and-hold for the "
               "same points, which is why drawdown and exposure are shown alongside.")

    with st.expander("Fitted parameters"):
        st.json({k: v for k, v in best["params"].items()})
    with st.expander("Active confluences and rule"):
        st.json(best["meta"])
        if not best["meta"].get("volume_available", True):
            st.caption("This feed carries no volume, so the vol_spike confluence was dropped "
                       "rather than evaluated as permanently False.")
    with st.expander("Search results (top 25)"):
        t = st.session_state.get("best_table", pd.DataFrame())
        st.dataframe(t.drop(columns=["params"], errors="ignore").head(25), use_container_width=True)

    eq = best["full_trades"]["net_points"].cumsum() if not best["full_trades"].empty else pd.Series()
    if len(eq):
        fig2, ax2 = plt.subplots(figsize=(11, 3))
        ax2.plot(eq.values, lw=1.2)
        ax2.axvline(len(best["full_trades"]) * cfg["train_frac"], color="crimson", ls="--", lw=1)
        ax2.set_title("Cumulative net points (red line ≈ train/holdout boundary)")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2, clear_figure=True)


def tab_live(cfg):
    ls = live_state()
    best = st.session_state.get("best")
    if not best:
        st.info("Fit parameters on the Backtesting tab first. Live trading uses that exact rule set.")
        return
    if cfg["source"] != "yfinance":
        st.warning("Live trading needs yfinance. Tick 'Use yfinance' in the sidebar.")
        return

    p = dict(best["params"])

    st.markdown("### Risk levels for this session")
    st.caption("Both dropdowns open on the optimised value. Overriding them changes live "
               "behaviour only — the accuracy shown below was measured at the optimised levels.")
    c1, c2, c3 = st.columns(3)
    sl_opts = sorted(set(SL_CHOICES + [round(p["sl_pct"], 4)]))
    tp_opts = sorted(set(TP_CHOICES + [round(p["tp_pct"], 4)]))
    sl = c1.selectbox("Stop loss", sl_opts, index=nearest(sl_opts, p["sl_pct"]),
                      format_func=lambda x: f"{x*100:.2f}%")
    tp = c2.selectbox("Target", tp_opts, index=nearest(tp_opts, p["tp_pct"]),
                      format_func=lambda x: f"{x*100:.2f}%")
    hold = c3.number_input("Max bars held", 1, 500, int(p["max_hold"]))
    p.update({"sl_pct": float(sl), "tp_pct": float(tp), "max_hold": int(hold)})
    if abs(sl - best["params"]["sl_pct"]) > 1e-9 or abs(tp - best["params"]["tp_pct"]) > 1e-9:
        st.caption(f"Overridden. Optimised values were SL {best['params']['sl_pct']*100:.2f}% / "
                   f"TP {best['params']['tp_pct']*100:.2f}%. Reward:risk now {tp/sl:.2f}.")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Start", type="primary", disabled=ls["running"]):
        ls["running"] = True
        log("Live paper trading started")
    if b2.button("Stop", disabled=not ls["running"]):
        ls["running"] = False
        log("Live paper trading stopped — open position left untouched")
    if b3.button("Square off now", disabled=ls["position"] is None):
        try:
            latest = fetch_live(cfg["symbol"], cfg["interval"], 2, cfg["tz"])
            px = float(latest["Close"].iloc[-1])
            close_position(px, "manual_squareoff", latest["Date"].iloc[-1],
                           cfg["cost_bps"], cfg["slippage"])
        except Exception as e:
            st.error(f"Could not fetch a price to square off: {e}")
    if b4.button("Reset session"):
        st.session_state.live = blank_live_state()
        st.rerun()

    refresh = st.select_slider("Auto-refresh every", [5, 10, 15, 30, 60, 120], value=30,
                               format_func=lambda s: f"{s}s")

    def render():
        ls = live_state()
        status = "RUNNING" if ls["running"] else "STOPPED"
        try:
            days_needed = min(INTERVAL_MAX_DAYS[cfg["interval"]], max(5, cfg["days"]))
            df = fetch_live(cfg["symbol"], cfg["interval"], days_needed, cfg["tz"])
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

        if ls["running"]:
            step_live(df, p, cfg["cost_bps"], cfg["slippage"], cfg["intraday"])

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last

        st.markdown("#### Last traded price")
        t = st.columns(4)
        t[0].metric("LTP", f"{last:,.2f}", f"{last - prev:+,.2f} ({(last/prev - 1)*100:+.2f}%)")
        t[1].metric("Bar open", f"{float(df['Open'].iloc[-1]):,.2f}")
        t[2].metric("Bar high / low",
                    f"{float(df['High'].iloc[-1]):,.2f} / {float(df['Low'].iloc[-1]):,.2f}")
        t[3].metric("Bar", f"{pd.Timestamp(df['Date'].iloc[-1]):%d %b %H:%M}",
                    help="Newest bar, still forming. Signals only read closed bars.")
        st.caption(f"{status} · {cfg['symbol']} · {INTERVAL_LABELS[cfg['interval']]} bars · "
                   f"refreshed {pd.Timestamp.now(tz=cfg['tz']):%H:%M:%S}")

        st.markdown("#### Session performance")
        s = live_stats()
        m = st.columns(6)
        m[0].metric("Live trades", s["trades"])
        m[1].metric("Live accuracy", f"{s['win_rate']*100:.1f}%" if s["trades"] else "—")
        m[2].metric("Net points", f"{s['net_points']:+,.2f}")
        m[3].metric(f"Net cash (qty {cfg['qty']})", f"{s['net_points'] * cfg['qty']:+,.2f}")
        m[4].metric("Expectancy", f"{s['expectancy_points']:+.2f} pts")
        hold_stats = best["test_stats"] or best["train_stats"]
        m[5].metric("Backtest accuracy", f"{hold_stats['win_rate']*100:.1f}%",
                    help="From the holdout slice, not the fitted window.")

        if ls["pending"]:
            pend = ls["pending"]
            st.info(f"Pending {('LONG' if pend['dir'] == 1 else 'SHORT')} order — fills at the "
                    f"open of the next bar. Signal: {pend['reason'][:120]}")

        if ls["position"]:
            pos = ls["position"]
            d = pos["dir"]
            open_pts = (last - pos["entry"]) * d
            st.markdown("#### Open position")
            k = st.columns(6)
            k[0].metric("Side", pos["direction"].upper())
            k[1].metric("Entry", f"{pos['entry']:,.2f}")
            k[2].metric("Stop loss", f"{pos['sl']:,.2f}", f"{(pos['sl']-pos['entry'])*d:+.1f} pts")
            k[3].metric("Target", f"{pos['tp']:,.2f}", f"{(pos['tp']-pos['entry'])*d:+.1f} pts")
            k[4].metric("Open P&L", f"{open_pts:+,.2f} pts",
                        f"{open_pts * cfg['qty']:+,.2f} cash")
            k[5].metric("Bars held", pos["bars"])
            st.caption(f"Signal: {pos['reason'][:200]}")
            with st.expander("Parameters selected for this trade"):
                st.json(pos["params_used"])
        elif not ls["pending"]:
            if ls["running"] and ls["last_bar"] is not None:
                st.markdown(f"#### Flat — armed at "
                            f"{pd.Timestamp(ls['last_bar']):%d %b %H:%M}, waiting for a signal")
            else:
                st.markdown("#### Flat — press Start to arm the engine")

        if ls["trades"]:
            st.markdown("#### Closed this session")
            st.dataframe(pd.DataFrame(ls["trades"]).iloc[::-1].head(15), use_container_width=True)
        with st.expander("Activity log"):
            st.code("\n".join(ls["log"]) or "nothing yet")

    if hasattr(st, "fragment"):
        frag = st.fragment(run_every=refresh if ls["running"] else None)(render)
        frag()
    else:
        render()
        if ls["running"]:
            time.sleep(refresh)
            st.rerun()


def tab_history(cfg):
    """Live paper trades only. Backtest trades live on the Backtesting tab."""
    ls = live_state()
    st.markdown("### Live paper trades")
    live_df = pd.DataFrame(ls["trades"])
    if live_df.empty:
        st.info("No live trades yet. Arm the engine on the Live trading tab; it trades "
                "forward from the moment you press Start and never backfills history.")
        return

    live_df["net_cash"] = (live_df["net_points"] * cfg["qty"]).round(2)
    st.dataframe(detailed_stats(live_df, cfg["qty"]), use_container_width=True, hide_index=True)
    st.markdown("#### Trades")
    st.dataframe(live_df.iloc[::-1], use_container_width=True)
    st.download_button("Download live trades (CSV)", live_df.to_csv(index=False),
                       "live_trades.csv", "text/csv")
    eq = live_df["net_points"].cumsum()
    st.line_chart(eq, height=200)


def main():
    st.title("Confluence Trader")
    st.caption("MA · EMA · MACD · RSI · ADX · Bollinger · ATR · Volume confluences · purged holdout · charges applied · paper trading only")
    cfg = sidebar()

    try:
        df, label = load_data(cfg)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return
    if df is None or len(df) < 120:
        st.info("Upload a file or pick an instrument. At least 120 bars are needed.")
        return

    t1, t2, t3 = st.tabs(["Backtesting", "Live trading", "Trade history"])
    with t1:
        tab_backtest(cfg, df, label)
    with t2:
        tab_live(cfg)
    with t3:
        tab_history(cfg)


if __name__ == "__main__":
    main()
