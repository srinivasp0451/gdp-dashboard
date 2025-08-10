# streamlit_walkforward_continuation.py
# Full corrected Streamlit script:
# - Expanded parameter grid
# - Walk-forward optimization (profit-factor objective)
# - Real P/L simulation for profit factor
# - Continuation check (yesterday -> today)
# - Guards against empty splits / empty concat
#
# Paste this into a .py file and run with `streamlit run streamlit_walkforward_continuation.py`

import pandas as pd
import numpy as np
import streamlit as st
from itertools import product
from datetime import timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Swing Trading — Walk-Forward + Continuation", layout="wide")
st.title("📊 Swing Trading Screener — Walk-Forward Optimizer + Continuation Check")

# -----------------------
# Helpers: Column mapping
# -----------------------
def map_columns(df, col_map):
    col_lower = {c.lower(): c for c in df.columns}
    res = {}
    for target, choices in col_map.items():
        for choice in choices:
            if choice in df.columns:
                res[target] = choice
                break
            elif choice.lower() in col_lower:
                res[target] = col_lower[choice.lower()]
                break
    return res

# -----------------------
# Compute Indicators
# -----------------------
def compute_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] > (1.5 * df['AvgVol20'])

    # keep Date column as datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# -----------------------
# Strategy - produce signals (entry rows)
# -----------------------
def produce_signals(df, params):
    """
    Returns DataFrame of signal rows with Entry, SL, TP, Type, EntryIdx
    (no exit simulation here)
    """
    atr_sl = params['atr_sl']
    atr_tp = params['atr_tp']
    ema_period = params['ema_period']
    min_conf = params['min_conf']
    trade_mode = params['trade_mode']

    df = df.copy()
    df['EMA_TREND'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    signals = []
    # start index where needed indicators are available
    start_idx = max(60, ema_period)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        long_conf = []
        short_conf = []

        if pd.notna(row['SMA20']) and pd.notna(row['SMA50']) and row['SMA20'] > row['SMA50']:
            long_conf.append("SMA Bullish")
        if pd.notna(row['RSI']) and 30 < row['RSI'] < 70:
            long_conf.append("RSI Healthy")
            short_conf.append("RSI Healthy")  # RSI healthy used on both sides
        if pd.notna(row['MACD']) and pd.notna(row['Signal']) and row['MACD'] > row['Signal']:
            long_conf.append("MACD Bullish")
        if pd.notna(row['BB_Mid']) and row['Close'] > row['BB_Mid']:
            long_conf.append("BB Breakout")
        if row.get('Vol_Surge', False):
            long_conf.append("Volume Surge"); short_conf.append("Volume Surge")

        if pd.notna(row['SMA20']) and pd.notna(row['SMA50']) and row['SMA20'] < row['SMA50']:
            short_conf.append("SMA Bearish")
        if pd.notna(row['MACD']) and pd.notna(row['Signal']) and row['MACD'] < row['Signal']:
            short_conf.append("MACD Bearish")
        if pd.notna(row['BB_Mid']) and row['Close'] < row['BB_Mid']:
            short_conf.append("BB Breakdown")

        # Long entry
        if (row['Close'] > row['EMA_TREND']) and (len(long_conf) >= min_conf) and (trade_mode in ["Both","Long Only"]):
            entry = float(row['Close'])
            sl = entry - atr_sl * float(row['ATR']) if pd.notna(row['ATR']) else np.nan
            tp = entry + atr_tp * float(row['ATR']) if pd.notna(row['ATR']) else np.nan
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type': 'Long', 'Entry': entry, 'SL': sl, 'TP': tp, 'Reasons': ", ".join(long_conf)})
        # Short entry
        elif (row['Close'] < row['EMA_TREND']) and (len(short_conf) >= min_conf) and (trade_mode in ["Both","Short Only"]):
            entry = float(row['Close'])
            sl = entry + atr_sl * float(row['ATR']) if pd.notna(row['ATR']) else np.nan
            tp = entry - atr_tp * float(row['ATR']) if pd.notna(row['ATR']) else np.nan
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type': 'Short', 'Entry': entry, 'SL': sl, 'TP': tp, 'Reasons': ", ".join(short_conf)})

    return pd.DataFrame(signals)

# -----------------------
# Simulate outcomes to compute P/L and Profit Factor
# -----------------------
def simulate_outcomes(df, signals, max_holding_days=10):
    """
    For each signal, scan forward upto max_holding_days and check H/L for TP/SL
    Returns trades DataFrame with P/L column and gross profit/loss summary
    """
    trades = []
    gross_profit = 0.0
    gross_loss = 0.0

    if signals.empty:
        return pd.DataFrame(), gross_profit, gross_loss

    for _, s in signals.iterrows():
        idx = int(s['EntryIdx'])
        entry_price = s['Entry']
        sl = s['SL']
        tp = s['TP']
        direction = s['Type']
        exit_price = None
        exit_date = None
        pnl = None
        hit = None

        # scan next days
        for j in range(idx+1, min(idx+1+max_holding_days, len(df))):
            high = df.iloc[j]['High']
            low = df.iloc[j]['Low']
            date = df.iloc[j]['Date']

            if direction == 'Long':
                # TP first if high >= TP
                if pd.notna(tp) and high >= tp:
                    exit_price = tp
                    exit_date = date
                    pnl = exit_price - entry_price
                    gross_profit += max(0.0, pnl)
                    hit = 'TP'
                    break
                if pd.notna(sl) and low <= sl:
                    exit_price = sl
                    exit_date = date
                    pnl = exit_price - entry_price
                    gross_loss += min(0.0, pnl)
                    hit = 'SL'
                    break
            else:  # Short
                if pd.notna(tp) and low <= tp:
                    exit_price = tp
                    exit_date = date
                    pnl = entry_price - exit_price
                    gross_profit += max(0.0, pnl)
                    hit = 'TP'
                    break
                if pd.notna(sl) and high >= sl:
                    exit_price = sl
                    exit_date = date
                    pnl = entry_price - exit_price
                    gross_loss += min(0.0, pnl)
                    hit = 'SL'
                    break

        # if neither hit within holding window, close at last available close
        if exit_price is None:
            last_close = df.iloc[min(idx+max_holding_days, len(df)-1)]['Close']
            exit_price = float(last_close)
            exit_date = df.iloc[min(idx+max_holding_days, len(df)-1)]['Date']
            if direction == 'Long':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += pnl  # negative

            hit = 'TimeExit'

        trades.append({
            'EntryIdx': idx,
            'EntryDate': s['EntryDate'],
            'ExitDate': exit_date,
            'Type': direction,
            'Entry': entry_price,
            'Exit': exit_price,
            'SL': sl,
            'TP': tp,
            'P/L': pnl,
            'Hit': hit,
            'Reasons': s.get('Reasons', '')
        })

    trades_df = pd.DataFrame(trades)
    # gross_loss may be negative; convert to absolute for PF calculation
    gross_loss_abs = abs(gross_loss) if gross_loss < 0 else gross_loss
    return trades_df, gross_profit, gross_loss_abs

# -----------------------
# Backtest wrapper (returns trades_df and stats)
# -----------------------
def run_backtest_with_stats(df, params):
    # produce signals with given params
    signals = produce_signals(df, params)
    if signals.empty:
        return pd.DataFrame(), {'total_profit': 0.0, 'profit_factor': None, 'win_rate': 0.0, 'total_trades': 0}

    trades_df, gross_profit, gross_loss = simulate_outcomes(df, signals)
    if trades_df.empty:
        return pd.DataFrame(), {'total_profit': 0.0, 'profit_factor': None, 'win_rate': 0.0, 'total_trades': 0}

    total_profit = trades_df['P/L'].sum()
    total_trades = len(trades_df)
    win_rate = (trades_df['P/L'] > 0).mean() * 100 if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if (gross_loss is not None and gross_loss > 0) else (np.inf if gross_profit > 0 else None)

    stats = {
        'total_profit': total_profit,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }
    return trades_df, stats

# -----------------------
# Optimization over grid
# -----------------------
def run_optimization(train_df, param_grid):
    best_pf = -np.inf
    best_params = None
    best_stats = None
    best_trades = None

    # Precompute EMA trend per candidate? We'll set ema per param inside produce_signals
    for params in param_grid:
        atr_sl, atr_tp, ema_period, min_conf, trade_mode = params
        param_obj = {
            'atr_sl': atr_sl,
            'atr_tp': atr_tp,
            'ema_period': ema_period,
            'min_conf': min_conf,
            'trade_mode': trade_mode
        }
        trades_df, stats = run_backtest_with_stats(train_df, param_obj)
        pf = stats['profit_factor']
        # filter sets with no PF or no trades
        if trades_df.empty or pf is None:
            continue
        # choose by profit factor; smaller tie-breaker: more trades
        if (pf > best_pf) or (pf == best_pf and stats['total_trades'] > (best_stats['total_trades'] if best_stats else 0)):
            best_pf = pf
            best_params = param_obj
            best_stats = stats
            best_trades = trades_df

    return best_params, best_stats, best_trades

# -----------------------
# Walk-forward driver
# -----------------------
def walk_forward_driver(df, param_grid, train_frac=0.7, test_days=None, step_days=None):
    """
    df: full daily df
    param_grid: list of tuples (atr_sl, atr_tp, ema, min_conf, trade_mode)
    train_frac: fraction for train size
    test_days: if provided, fixed test window size in days (rows)
    step_days: how many rows to move window forward each iteration
    Returns: wf_results list and params_history list
    """
    n = len(df)
    train_size = int(train_frac * n)
    if test_days is None:
        test_size = max(int(0.3 * n), 10)
    else:
        test_size = test_days
    if step_days is None:
        step = test_size
    else:
        step = step_days

    wf_results = []
    params_history = []
    overall_test_trades = []

    # iterate windows
    for start in range(0, max(1, n - train_size - test_size + 1), step):
        train_df = df.iloc[start:start + train_size].reset_index(drop=True)
        test_df = df.iloc[start + train_size:start + train_size + test_size].reset_index(drop=True)

        if train_df.empty or test_df.empty:
            continue

        best_params, best_stats, best_train_trades = run_optimization(train_df, param_grid)
        if best_params is None:
            # no valid params found for this window
            continue

        params_history.append(best_params)
        # test on test_df
        test_trades, test_stats = run_backtest_with_stats(test_df, best_params)
        wf_results.append({
            'window_start_idx': start,
            'train_from': train_df['Date'].iloc[0],
            'train_to': train_df['Date'].iloc[-1],
            'test_from': test_df['Date'].iloc[0],
            'test_to': test_df['Date'].iloc[-1],
            'best_params': best_params,
            'train_stats': best_stats,
            'test_stats': test_stats,
            'test_trades': test_trades
        })
        if not test_trades.empty:
            overall_test_trades.append(test_trades)

    # concat safely
    combined_test_trades = pd.concat(overall_test_trades, ignore_index=True) if overall_test_trades else pd.DataFrame()
    return wf_results, params_history, combined_test_trades

# -----------------------
# Continuation check
# -----------------------
def continuation_check(df, best_params):
    """
    Re-evaluates yesterday's signal and checks if today's close has breached SL/TP.
    """
    if len(df) < 2:
        st.info("Not enough rows to check continuation (need at least 2 rows).")
        return

    # produce yesterday's signal only
    # We call produce_signals on last N rows but ensure indices align
    window = df.iloc[-30:].reset_index(drop=True)  # small slice to find yesterday index in slice
    signals = produce_signals(window, best_params)
    if signals.empty:
        st.warning("⛔ No valid trade was signaled yesterday (given most-recent params).")
        return

    # take signal whose EntryDate equals yesterday (or nearest previous)
    yesterday_date = df.iloc[-2]['Date']
    # find signal with EntryDate == yesterday_date; fallback to last signal if not exact match
    sig = None
    for _, s in signals.iterrows():
        if pd.to_datetime(s['EntryDate']).date() == pd.to_datetime(yesterday_date).date():
            sig = s
            break
    if sig is None:
        # fallback to latest signal in the slice
        sig = signals.iloc[-1]

    # map slice index to global index
    # slice start index in global df
    slice_start_global_idx = len(df) - len(window)
    entry_global_idx = slice_start_global_idx + int(sig['EntryIdx'])

    yesterday_entry = sig
    today = df.iloc[-1]
    entry_price = yesterday_entry['Entry']
    sl = yesterday_entry['SL']
    tp = yesterday_entry['TP']
    direction = yesterday_entry['Type']

    still_active = False
    reason_txt = yesterday_entry.get('Reasons', '')

    if direction == 'Long':
        # if today's close hasn't breached SL or TP
        if (pd.notna(sl) and pd.notna(tp)) and (today['Close'] > sl) and (today['Close'] < tp):
            still_active = True
    else:
        if (pd.notna(sl) and pd.notna(tp)) and (today['Close'] < sl) and (today['Close'] > tp):
            still_active = True

    if still_active:
        st.info(f"✅ Yesterday's {direction} at {entry_price:.2f} is STILL ACTIVE | SL: {sl:.2f} | TP: {tp:.2f} | Reasons: {reason_txt}")
    else:
        st.warning("⛔ Yesterday's recommendation is no longer valid (SL/TP hit or conditions invalidated).")

# -----------------------
# UI: File upload and run
# -----------------------
uploaded_file = st.file_uploader("Upload daily OHLCV CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV with Date, Open, High, Low, Close, Volume (daily).")
    st.stop()

# read CSV - handle BOM and whitespace columns
raw = pd.read_csv(uploaded_file)
raw.columns = [c.encode('utf-8').decode('utf-8-sig') for c in raw.columns]
raw.columns = [c.strip() for c in raw.columns]

col_map = {
    'Date': ['Date', 'date'],
    'Open': ['Open', 'OPEN', 'open'],
    'High': ['High', 'HIGH', 'high'],
    'Low': ['Low', 'LOW', 'low'],
    'Close': ['Close', 'CLOSE', 'close', 'LTP', 'ltp'],
    'Volume': ['Volume', 'VOLUME', 'volume', 'Shares Traded']
}
mapping = map_columns(raw, col_map)
missing = [k for k in col_map.keys() if k not in mapping]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = pd.DataFrame()
for tgt, src in mapping.items():
    df[tgt] = raw[src]

# parse and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# numeric conversions
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# compute indicators
df = compute_indicators(df)
st.success("Indicators computed ✅")
st.write(f"Rows: {len(df)} | From {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

# parameter grid (expanded)
atr_sl_choices = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
atr_tp_choices = [1.5, 2.0, 2.5, 3.0, 3.5]
ema_choices = [50, 100, 150, 200, 250]
min_conf_choices = [1, 2, 3, 4]
trade_modes = ["Both", "Long Only", "Short Only"]
param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

st.write(f"Parameter combinations to try: {len(param_grid)} (this may be slow).")

# walk-forward settings (UI options)
train_frac = st.sidebar.slider("Train fraction", 0.5, 0.85, 0.7)
test_days = st.sidebar.number_input("Test window (rows/days)", min_value=5, max_value=max(10, int(len(df) // 4)), value=max(30, int(0.3 * len(df))), step=1)
step_days = st.sidebar.number_input("Step (rows) for sliding window", min_value=1, max_value=max(1, test_days), value=test_days, step=1)
max_windows = st.sidebar.number_input("Max windows (0 = all)", min_value=0, value=0)

run_btn = st.button("Run Walk-Forward Optimization")
if run_btn:
    st.info("Running walk-forward optimization — this may take time depending on CSV size and grid size.")
    progress_bar = st.progress(0.0)

    # optionally limit param grid for faster iteration during debugging
    max_param_combos = st.sidebar.number_input("Max param combos (0 = all)", min_value=0, value=0)
    if max_param_combos > 0:
        param_grid_used = param_grid[:max_param_combos]
    else:
        param_grid_used = param_grid

    # run driver
    wf_results, params_history, combined_test_trades = walk_forward_driver(df, param_grid_used, train_frac=train_frac, test_days=test_days, step_days=step_days)

    # update progress to done
    progress_bar.progress(1.0)
    progress_bar.empty()

    if not wf_results:
        st.error("Walk-forward produced no valid windows / params. Try lowering windows/test_size or reduce param grid.")
        st.stop()

    st.success("Walk-forward completed ✅")
    st.write(f"Windows evaluated: {len(wf_results)}")
    # aggregate test results
    agg_profit = sum([w['test_stats']['total_profit'] for w in wf_results if w.get('test_stats')])
    agg_trades = sum([w['test_stats']['total_trades'] for w in wf_results if w.get('test_stats')])
    agg_win_sum = sum([w['test_stats']['win_rate'] * w['test_stats']['total_trades']/100 for w in wf_results if w.get('test_stats') and w['test_stats']['total_trades']>0])
    agg_win_rate = (agg_win_sum / agg_trades * 100) if agg_trades>0 else 0.0

    st.header("📈 Walk-Forward Summary (Aggregate Test Results)")
    st.write(f"Aggregate Test Profit (points): {agg_profit:.2f}")
    st.write(f"Aggregate Test Trades: {agg_trades}")
    st.write(f"Aggregate Test Win Rate (weighted): {agg_win_rate:.2f}%")

    # show recent windows and best params
    recent = wf_results[-10:]
    show_table = []
    for w in recent:
        bp = w['best_params']
        ts = w['test_stats'] or {}
        show_table.append({
            'window_start': w['train_from'].date(),
            'train_to': w['train_to'].date(),
            'test_from': w['test_from'].date(),
            'test_to': w['test_to'].date(),
            'atr_sl': bp['atr_sl'], 'atr_tp': bp['atr_tp'], 'ema': bp['ema_period'],
            'min_conf': bp['min_conf'], 'mode': bp['trade_mode'],
            'test_profit': ts.get('total_profit', None),
            'test_trades': ts.get('total_trades', None),
            'test_pf': ts.get('profit_factor', None)
        })
    st.subheader("Recent windows — best parameters")
    st.dataframe(pd.DataFrame(show_table))

    # combined test trades display
    if not combined_test_trades.empty:
        combined_test_trades['Cum_PnL'] = combined_test_trades['P/L'].cumsum()
        st.subheader("📜 Combined Test Trades (all windows)")
        st.dataframe(combined_test_trades)
        fig, ax = plt.subplots()
        ax.plot(combined_test_trades['ExitDate'] if 'ExitDate' in combined_test_trades.columns else combined_test_trades['ExitDate'], combined_test_trades['Cum_PnL'], marker='o')
        ax.set_title("Walk-Forward Equity Curve (combined test trades)")
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # pick most recent best params for live recommendation
    most_recent_best = params_history[-1]
    st.success(f"Using most-recent window's best params for live recommendation: {most_recent_best}")

    # LIVE recommendation using most recent params
    df['EMA_TREND'] = df['Close'].ewm(span=most_recent_best['ema_period'], adjust=False).mean()
    last = df.iloc[-1]
    live_long, live_short = [], []
    if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_long.append("RSI Healthy"); live_short.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
    if last['Vol_Surge']: live_long.append("Volume Surge"); live_short.append("Volume Surge")
    st.header("📢 Live Recommendation (Most Recent Best Parameters)")
    if last['Close'] > last['EMA_TREND'] and len(live_long) >= most_recent_best['min_conf'] and (most_recent_best['trade_mode'] in ["Both","Long Only"]):
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - most_recent_best['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] + most_recent_best['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_long)}")
    elif last['Close'] < last['EMA_TREND'] and len(live_short) >= most_recent_best['min_conf'] and (most_recent_best['trade_mode'] in ["Both","Short Only"]):
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + most_recent_best['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] - most_recent_best['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_short)}")
    else:
        st.error("❌ No strong trade setup currently (most recent best params).")

    # Continuation check
    st.subheader("🔁 Continuation Check (yesterday -> today)")
    continuation_check(df, most_recent_best)

    st.info("Completed. Tip: if runtime is long, reduce param grid or max param combos in sidebar.")
