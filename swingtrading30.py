# pro_trading_app_fixed.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import itertools
import math

st.set_page_config(layout="wide", page_title="Pro Trading Strategy Lab — Fixed yfinance flattening")

# -------------------------
# Utilities: flatten yfinance DataFrame gracefully
# -------------------------
def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has MultiIndex columns (common when yf.download returns grouped columns),
    flatten to single-level names by joining tuple parts and then map to standard OHLCV names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            # col is a tuple; ignore empty parts
            parts = [str(c) for c in col if (c is not None and str(c).strip() != "")]
            # join with underscore
            flat_name = "_".join(parts).strip()
            if flat_name == "":
                flat_name = "unnamed"
            flat_cols.append(flat_name)
        df = df.copy()
        df.columns = flat_cols
    return df

def _find_best_col(cols, keywords):
    """
    find first column name in cols that contains any keyword (order matters).
    Comparison is relaxed: remove spaces/underscores and lowercase.
    """
    norm_map = {c: c.lower().replace(" ", "").replace("_", "") for c in cols}
    for kw in keywords:
        kw_norm = kw.lower().replace(" ", "").replace("_", "")
        for orig_c, cnorm in norm_map.items():
            if kw_norm in cnorm:
                return orig_c
    return None

def normalize_yf_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw yfinance df (possibly MultiIndex columns) and return a df with standard columns:
    ['Open','High','Low','Close','Volume'] with datetime index.
    Use 'Adj Close' for 'Close' if present (preferred).
    """
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame()

    df = _flatten_multiindex_columns(raw_df)

    cols = list(df.columns)

    # Prefer adjusted close if present
    close_col = _find_best_col(cols, ["adjclose", "adjclose", "adjclose", "adjclose", "adj close", "adjclose", "close"])
    # find open/high/low/volume
    open_col = _find_best_col(cols, ["open"])
    high_col = _find_best_col(cols, ["high"])
    low_col = _find_best_col(cols, ["low"])
    volume_col = _find_best_col(cols, ["volume", "vol"])

    # If no close found, take the first numeric column
    if close_col is None:
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                close_col = c
                break

    # Build normalized DF
    new = pd.DataFrame(index=df.index)
    if open_col is not None:
        new["Open"] = df[open_col]
    else:
        new["Open"] = np.nan
    if high_col is not None:
        new["High"] = df[high_col]
    else:
        new["High"] = np.nan
    if low_col is not None:
        new["Low"] = df[low_col]
    else:
        new["Low"] = np.nan
    if close_col is not None:
        new["Close"] = df[close_col]
    else:
        new["Close"] = np.nan
    if volume_col is not None:
        new["Volume"] = df[volume_col]
    else:
        # fill zeros if no volume present
        new["Volume"] = 0

    # Ensure index is datetime and sorted ascending
    if not isinstance(new.index, pd.DatetimeIndex):
        try:
            new.index = pd.to_datetime(new.index)
        except Exception:
            pass
    new = new.sort_index(ascending=True)
    # Drop rows where Close is NaN
    new = new.dropna(subset=["Close"])
    return new

# -------------------------
# Cached fetcher (graceful)
# -------------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_ticker_data(ticker: str, interval: str, period: str):
    """
    Fetch ticker with yfinance and normalize dataframe to columns Open/High/Low/Close/Volume.
    Uses caching to reduce yfinance rate hits.
    """
    try:
        # use yf.download which is robust; group_by default is 'column' except multiple tickers
        raw = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, threads=True, progress=False)
        norm = normalize_yf_df(raw)
        return norm
    except Exception as e:
        # return empty df on error; UI will show a message
        return pd.DataFrame()

# -------------------------
# Indicators & helpers (manual)
# -------------------------
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    # use exponential for responsive RSI
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def detect_swings(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """
    Return local peaks and troughs as DataFrame with columns ['type','price'] indexed by date.
    order controls sensitivity.
    """
    if len(df) < order*2 + 1:
        return pd.DataFrame(columns=['type', 'price'])
    close = df['Close'].values
    maxima_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    minima_idx = argrelextrema(close, np.less_equal, order=order)[0]
    rows = []
    for i in maxima_idx:
        rows.append((df.index[i], 'peak', df['Close'].iat[i]))
    for i in minima_idx:
        rows.append((df.index[i], 'trough', df['Close'].iat[i]))
    swings = pd.DataFrame(rows, columns=['datetime', 'type', 'price']).set_index('datetime').sort_index()
    return swings

def fib_levels(low, high):
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low
    }

def detect_rsi_divergence(df: pd.DataFrame, rsi_col='RSI', window=30):
    # simplified divergence detection (returns list of ('bullish'/'bearish', index))
    res = []
    if len(df) < window + 2 or rsi_col not in df.columns:
        return res
    for i in range(window, len(df)):
        sl = df.iloc[i-window:i+1]
        # bullish: price lower-low while rsi higher-low
        try:
            first_low_idx = sl['Close'].idxmin()
            first_low = sl.loc[first_low_idx, 'Close']
            first_rsi = sl.loc[first_low_idx, rsi_col]
            earlier = sl.loc[:first_low_idx]
            if len(earlier) >= 2:
                earlier_low_idx = earlier['Close'].idxmin()
                earlier_low = earlier.loc[earlier_low_idx, 'Close']
                earlier_rsi = earlier.loc[earlier_low_idx, rsi_col]
                if first_low < earlier_low and first_rsi > earlier_rsi:
                    res.append(('bullish', first_low_idx))
            # bearish
            first_high_idx = sl['Close'].idxmax()
            first_high = sl.loc[first_high_idx, 'Close']
            first_high_rsi = sl.loc[first_high_idx, rsi_col]
            earlier2 = sl.loc[:first_high_idx]
            if len(earlier2) >= 2:
                earlier_high_idx = earlier2['Close'].idxmax()
                earlier_high = earlier2.loc[earlier_high_idx, 'Close']
                earlier_high_rsi = earlier2.loc[earlier_high_idx, rsi_col]
                if first_high > earlier_high and first_high_rsi < earlier_high_rsi:
                    res.append(('bearish', first_high_idx))
        except Exception:
            continue
    return res

# -------------------------
# Signals & Backtester (simple, deterministic)
# -------------------------
def generate_signals(df: pd.DataFrame, params: dict):
    """
    Return df copy with columns 'RSI','ATR','signal'. signal: 1 long entry at bar close, -1 short entry, 0 none.
    Uses simple RSI oversold/overbought cross + recent swing check + optional MTF (not implemented here).
    """
    df = df.copy()
    df['RSI'] = rsi(df['Close'], length=params.get('rsi_len', 14))
    df['ATR'] = atr(df, length=params.get('atr_len', 14))
    swings = detect_swings(df, order=params.get('zig_order', 5))
    df['signal'] = 0

    # mark swing types in window for quick check
    swing_types_by_index = swings['type'].to_dict()

    for i in range(1, len(df)):
        prev_rsi = df['RSI'].iat[i-1]
        curr_rsi = df['RSI'].iat[i]
        # long signal: rsi crosses up oversold
        if (prev_rsi < params.get('rsi_oversold', 30)) and (curr_rsi >= params.get('rsi_oversold', 30)):
            # check if there is a recent trough within confirm_window bars
            w = params.get('confirm_window', 30)
            start = max(0, i - w)
            recent_indexes = df.index[start:i+1]
            has_trough = any(idx in swing_types_by_index and swing_types_by_index[idx] == 'trough' for idx in recent_indexes)
            if has_trough:
                df.at[df.index[i], 'signal'] = 1
        # short signal: rsi crosses down overbought
        if (prev_rsi > params.get('rsi_overbought', 70)) and (curr_rsi <= params.get('rsi_overbought', 70)):
            w = params.get('confirm_window', 30)
            start = max(0, i - w)
            recent_indexes = df.index[start:i+1]
            has_peak = any(idx in swing_types_by_index and swing_types_by_index[idx] == 'peak' for idx in recent_indexes)
            if has_peak:
                df.at[df.index[i], 'signal'] = -1

    # attach a fib from last major swing if possible
    if not swings.empty:
        peaks = swings[swings['type'] == 'peak']
        troughs = swings[swings['type'] == 'trough']
        if len(peaks) > 0 and len(troughs) > 0:
            # choose last trough and last peak
            last_trough_time = troughs.index[-1]
            last_peak_time = peaks.index[-1]
            # ensure low before high for uptrend; else invert
            if last_trough_time < last_peak_time:
                low = troughs.loc[last_trough_time, 'price']
                high = peaks.loc[last_peak_time, 'price']
            else:
                low = peaks.loc[last_peak_time, 'price']
                high = troughs.loc[last_trough_time, 'price']
            df.attrs['fib_levels'] = fib_levels(low, high)
        else:
            df.attrs['fib_levels'] = {}
    else:
        df.attrs['fib_levels'] = {}

    return df

def backtest(df: pd.DataFrame, params: dict, capital: float = 100000, max_holding_bars: int = 200):
    """
    Very simple backtester:
    - entries at close where signal != 0
    - TP/SL determined as ATR * multiplier and RR
    - flat 1 position at a time
    - returns trades dataframe and equity curve
    """
    df = df.copy()
    trades = []
    position = None
    cash = capital
    equity_list = []

    for i in range(len(df)):
        date = df.index[i]
        price = df['Close'].iat[i]

        # check exit if position open
        if position is not None:
            # check tp/sl on close price
            if position['side'] == 1:
                if price >= position['tp'] or price <= position['sl'] or (i - position['i']) >= max_holding_bars:
                    exit_price = price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    cash += pnl  # margin ignored (simplified)
                    trades.append({
                        'entry_dt': position['entry_dt'],
                        'exit_dt': date,
                        'side': 'Long',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'tp': position['tp'],
                        'sl': position['sl'],
                        'size': position['size'],
                        'pnl': pnl,
                        'holding_bars': i - position['i'],
                        'reason': position.get('reason', '')
                    })
                    position = None
            else:
                # short
                if price <= position['tp'] or price >= position['sl'] or (i - position['i']) >= max_holding_bars:
                    exit_price = price
                    pnl = (position['entry_price'] - exit_price) * position['size']
                    cash += pnl
                    trades.append({
                        'entry_dt': position['entry_dt'],
                        'exit_dt': date,
                        'side': 'Short',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'tp': position['tp'],
                        'sl': position['sl'],
                        'size': position['size'],
                        'pnl': pnl,
                        'holding_bars': i - position['i'],
                        'reason': position.get('reason', '')
                    })
                    position = None

        # if no position and signal -> open
        if position is None and df['signal'].iat[i] != 0:
            side = int(df['signal'].iat[i])
            entry_price = price
            atr_val = df['ATR'].iat[i] if 'ATR' in df.columns and not np.isnan(df['ATR'].iat[i]) else 1.0
            sl_mult = params.get('atr_sl_mult', 1.5)
            rr = params.get('rr', 2.0)
            if side == 1:
                sl = entry_price - sl_mult * atr_val
                tp = entry_price + rr * (entry_price - sl)
            else:
                sl = entry_price + sl_mult * atr_val
                tp = entry_price - rr * (sl - entry_price)

            # sizing: fixed risk percent per trade
            risk_pct = params.get('risk_pct', 0.01)
            risk_amount = capital * risk_pct
            per_unit_risk = abs(entry_price - sl)
            if per_unit_risk <= 0:
                continue
            size = math.floor(risk_amount / per_unit_risk)
            if size <= 0:
                continue
            position = {
                'i': i,
                'entry_dt': date,
                'entry_price': entry_price,
                'side': side,
                'size': size,
                'sl': sl,
                'tp': tp,
                'reason': 'RSI+Swing'
            }
            # cash reserved/margin ignored (simplified)
        # equity compute
        if position is None:
            equity = cash
        else:
            if position['side'] == 1:
                unreal = (price - position['entry_price']) * position['size']
            else:
                unreal = (position['entry_price'] - price) * position['size']
            equity = cash + unreal
        equity_list.append({'dt': date, 'equity': equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_list).set_index('dt') if len(equity_list) > 0 else pd.DataFrame()
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0.0
    total_pnl = trades_df['pnl'].sum() if len(trades_df) > 0 else 0.0
    buy_hold_pnl = df['Close'].iat[-1] - df['Close'].iat[0] if len(df) > 1 else 0.0

    return {
        'trades': trades_df,
        'equity': equity_df,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'buy_hold_pnl': buy_hold_pnl
    }

# -------------------------
# Simple auto-optimize (small grid)
# -------------------------
def auto_optimize(df: pd.DataFrame, param_grid: dict, capital: float):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))
    best_score = -1e12
    best_combo = None
    best_result = None
    for combo in combos:
        p = dict(zip(keys, combo))
        p['mtf_df'] = None
        sig_df = generate_signals(df, p)
        res = backtest(sig_df, p, capital=capital)
        # scoring: prefer positive pnl and higher winrate
        score = res['total_pnl'] + res['win_rate'] * 1000
        if score > best_score:
            best_score = score
            best_combo = p
            best_result = res
    return best_combo, best_result, best_score

# -------------------------
# Streamlit UI
# -------------------------
st.title("Pro Trading Strategy Lab — Fixed data flattening")
st.markdown("**Note:** This version fixes yfinance MultiIndex flattening and ensures the app runs. Strategy is illustrative — extend/validate before trading live.")

# Sidebar controls
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (yfinance)", value="^NSEI")
    interval = st.selectbox("Interval", options=[
        '1m','2m','5m','15m','30m','60m','90m','1h','1d','1wk'
    ], index=8)
    period = st.selectbox("Period", options=[
        '1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','10y','20y','30y'
    ], index=6)
    capital = st.number_input("Capital", value=100000.0, step=1000.0)
    rsi_len = st.number_input("RSI length", value=14, min_value=2, max_value=100)
    rsi_oversold = st.number_input("RSI oversold level", value=30)
    rsi_overbought = st.number_input("RSI overbought level", value=70)
    atr_len = st.number_input("ATR length", value=14, min_value=2, max_value=100)
    run = st.button("Run Analysis")
    st.markdown("---")
    st.markdown("Optimizer")
    auto_opt = st.checkbox("Auto optimize small grid", value=False)
    max_grid = st.selectbox("Grid aggressiveness (bigger = slower)", options=[10, 30, 60], index=0)

if not run:
    st.info("Set your inputs and click **Run Analysis**")
    st.stop()

# Fetch data
with st.spinner("Fetching & normalizing data from yfinance (cached)..."):
    raw_df = fetch_ticker_data(ticker, interval, period)

if raw_df is None or raw_df.empty:
    st.error("No data returned. Common causes:\n - invalid ticker\n - invalid interval/period combo (e.g., '1m' requires short periods)\nTry different inputs.")
    st.stop()

st.subheader("Fetched data (head & tail)")
st.dataframe(raw_df.head(20))
st.dataframe(raw_df.tail(20))

st.markdown(f"Date range: **{raw_df.index.min()}** to **{raw_df.index.max()}**  |  Min close: **{raw_df['Close'].min():.6f}**  |  Max close: **{raw_df['Close'].max():.6f}**")

# Build parameter grid or single param
param_template = {
    'rsi_len': [int(rsi_len)],
    'rsi_oversold': [int(rsi_oversold)],
    'rsi_overbought': [int(rsi_overbought)],
    'zig_order': [3,5,8],
    'atr_len': [int(atr_len)],
    'atr_sl_mult': [1.0, 1.5, 2.0],
    'rr': [1.5, 2.0],
    'risk_pct': [0.005, 0.01],
    'confirm_window': [20, 40]
}

if not auto_opt:
    params = {
        'rsi_len': int(rsi_len),
        'rsi_oversold': int(rsi_oversold),
        'rsi_overbought': int(rsi_overbought),
        'zig_order': 5,
        'atr_len': int(atr_len),
        'atr_sl_mult': 1.5,
        'rr': 2.0,
        'risk_pct': 0.01,
        'confirm_window': 30
    }
    sig_df = generate_signals(raw_df, params)
    bt = backtest(sig_df, params, capital=capital)
else:
    # reduce grid size if huge
    # compute total combos
    total_combos = 1
    for v in param_template.values():
        total_combos *= len(v)
    # sample by reducing long lists
    if total_combos > max_grid:
        # keep at most first and last values for each parameter
        small = {}
        for k, v in param_template.items():
            if len(v) > 2:
                small[k] = [v[0], v[-1]]
            else:
                small[k] = v
        input_grid = small
    else:
        input_grid = param_template
    best_params, bt, score = auto_optimize(raw_df, input_grid, capital)
    params = best_params
    st.markdown("### Auto-optimizer result (selected params)")
    st.json(params)
    st.write(f"Optimizer score: {score:.2f}")

# Ensure signals present
if 'RSI' not in sig_df.columns:
    sig_df['RSI'] = rsi(sig_df['Close'], length=params.get('rsi_len',14))
if 'ATR' not in sig_df.columns:
    sig_df['ATR'] = atr(sig_df, length=params.get('atr_len',14))

st.subheader("Signals (sample)")
st.dataframe(sig_df[['Close','RSI','ATR','signal']].tail(200))

# Backtest trades table
st.subheader("Backtest summary")
st.write(f"Strategy total PnL: {bt['total_pnl']:.4f}   |   Win rate: {bt['win_rate']*100:.2f}%   |   Buy & Hold points: {bt['buy_hold_pnl']:.4f}")
trades = bt['trades']
if trades.empty:
    st.warning("No trades generated with current params on this dataset.")
else:
    trades_display = trades.copy()
    trades_display = trades_display.rename(columns={
        'entry_dt': 'Entry DateTime',
        'exit_dt': 'Exit DateTime',
        'entry_price': 'Entry Level',
        'tp': 'Target',
        'sl': 'Stop Loss',
        'pnl': 'Total PnL',
        'reason': 'Reason/Logic'
    })
    trades_display['Probability of Profit (est)'] = f"{bt['win_rate']*100:.2f}%"
    trades_display = trades_display.sort_values('Entry DateTime', ascending=True)
    st.dataframe(trades_display)

# Live recommendation based on last candle
st.subheader("Live recommendation (based on last close)")
last_idx = sig_df.index[-1]
last_row = sig_df.iloc[-1]
signal = int(last_row['signal'])
if signal == 0:
    st.info("No trade signal on last candle.")
else:
    side = "Long" if signal == 1 else "Short"
    entry_price = float(last_row['Close'])
    atr_val = float(last_row['ATR']) if not np.isnan(last_row['ATR']) else float(atr(sig_df, length=params.get('atr_len',14)).iat[-1])
    sl_mult = params.get('atr_sl_mult', 1.5)
    rr = params.get('rr', 2.0)
    if signal == 1:
        sl = entry_price - sl_mult * atr_val
        tp = entry_price + rr * (entry_price - sl)
    else:
        sl = entry_price + sl_mult * atr_val
        tp = entry_price - rr * (sl - entry_price)

    risk_pct = params.get('risk_pct', 0.01)
    risk_amount = capital * risk_pct
    per_unit_risk = abs(entry_price - sl)
    size = math.floor(risk_amount / per_unit_risk) if per_unit_risk > 0 else 0

    st.write({
        'side': side,
        'entry_dt': str(last_idx),
        'entry_level': round(entry_price, 6),
        'target': round(tp, 6),
        'stop_loss': round(sl, 6),
        'size_units': size,
        'probability_of_profit_est': f"{bt['win_rate']*100:.2f}%",
        'reason_logic': 'RSI crossover near swing (RSI+ZigZag)'
    })

# Equity curve
if not bt['equity'].empty:
    st.subheader("Equity curve")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(bt['equity'].index, bt['equity']['equity'])
    ax.set_ylabel("Equity")
    ax.set_xlabel("Datetime")
    st.pyplot(fig)

# Fibonacci levels
if sig_df.attrs.get('fib_levels'):
    st.subheader("Detected Fibonacci levels (from last major swing)")
    fib = sig_df.attrs['fib_levels']
    fdf = pd.DataFrame(list(fib.items()), columns=['level','price']).set_index('level')
    st.table(fdf)

# Heatmap (monthly returns) - simple
st.subheader("Monthly returns heatmap (sum of pct returns)")
rets = sig_df['Close'].pct_change().dropna()
if rets.empty:
    st.warning("Not enough data to compute returns heatmap.")
else:
    tmp = rets.to_frame('r')
    tmp['year'] = tmp.index.year
    tmp['month'] = tmp.index.month
    monthly = tmp.groupby(['year','month']).sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='r').fillna(0)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    c = ax2.imshow(pivot.values, aspect='auto')
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels(pivot.index)
    ax2.set_xticks(np.arange(12))
    ax2.set_xticklabels(np.arange(1,13))
    ax2.set_title("Monthly sum returns")
    st.pyplot(fig2)

# Validation check
st.subheader("Automated validation (basic checks)")
msgs = []
wr = bt.get('win_rate', 0.0)
tpnl = bt.get('total_pnl', 0.0)
bh = bt.get('buy_hold_pnl', 0.0)
msgs.append(f"Win rate: {wr*100:.2f}% (target 80%)")
if tpnl > bh and tpnl > 0:
    msgs.append(f"Strategy beats buy & hold on this dataset (strategy pnl {tpnl:.4f} > buy-hold {bh:.4f})")
else:
    msgs.append(f"Strategy does NOT beat buy & hold (strategy pnl {tpnl:.4f} vs buy-hold {bh:.4f})")
st.write("\n".join(msgs))

# Offer CSV download for trades
if not trades.empty:
    csv = trades.to_csv(index=False)
    st.download_button("Download trades CSV", data=csv, file_name=f"{ticker}_trades.csv", mime="text/csv")

st.success("Finished.")
