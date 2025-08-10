# streamlit_walkforward_fixed.py
# Safe, robust Streamlit walk-forward optimizer + continuation check
# - auto-maps OHLCV column name variants
# - dynamic slider defaults (no StreamlitValueAboveMaxError)
# - profit-factor optimization with P/L simulation
# - guards against empty windows / concat errors
# - no persistent storage required (works off uploaded CSV)

import pandas as pd
import numpy as np
import itertools
import streamlit as st
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(page_title="Swing Trading — Walk-Forward (Fixed)", layout="wide")
st.title("📊 Swing Trading Screener — Walk-Forward + Continuation (Fixed)")

# -----------------------------
# Column mapping helper
# -----------------------------
def auto_map_columns(df):
    # produce a lowercase->original mapping
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}

    # candidate names for each required field
    candidates = {
        'Date': ['date', 'timestamp', 'datetime'],
        'Open': ['open', 'o'],
        'High': ['high', 'h'],
        'Low': ['low', 'l'],
        'Close': ['close', 'adj close', 'adj_close', 'close_price', 'ltp', 'last'],
        'Volume': ['volume', 'vol', 'totaltradedqty', 'shares traded', 'tradevolume']
    }

    mapping = {}
    for standard, cands in candidates.items():
        found = None
        for c in cands:
            if c in lower_map:
                found = lower_map[c]
                break
        # also accept exact same capitalization presence
        if not found and standard in cols:
            found = standard
        if found:
            mapping[found] = standard  # rename original -> standard

    return mapping

# -----------------------------
# indicator computations
# -----------------------------
def compute_indicators(df):
    df = df.copy()
    # SMA
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # ATR (True Range based)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # MACD
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger mid and volume surge
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] > (1.5 * df['AvgVol20'])

    return df

# -----------------------------
# produce signals (entry rows)
# -----------------------------
def produce_signals(df, params):
    atr_sl = params['atr_sl']
    atr_tp = params['atr_tp']
    ema_period = params['ema_period']
    min_conf = params['min_conf']
    trade_mode = params['trade_mode']

    df = df.copy()
    df['EMA_TREND'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    signals = []
    start_idx = max(60, ema_period)  # require enough history for indicators
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        long_conf = []
        short_conf = []

        if pd.notna(row['SMA20']) and pd.notna(row['SMA50']) and row['SMA20'] > row['SMA50']:
            long_conf.append("SMA Bullish")
        if pd.notna(row['RSI']) and 30 < row['RSI'] < 70:
            long_conf.append("RSI Healthy"); short_conf.append("RSI Healthy")
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

# -----------------------------
# simulate outcomes and compute gross_profit/gross_loss
# -----------------------------
def simulate_outcomes(df, signals, max_holding_days=10):
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

        for j in range(idx+1, min(idx+1+max_holding_days, len(df))):
            high = df.iloc[j]['High']
            low = df.iloc[j]['Low']
            date = df.iloc[j]['Date']

            if direction == 'Long':
                if pd.notna(tp) and high >= tp:
                    exit_price = tp; exit_date = date; pnl = exit_price - entry_price
                    gross_profit += max(0.0, pnl); hit = 'TP'; break
                if pd.notna(sl) and low <= sl:
                    exit_price = sl; exit_date = date; pnl = exit_price - entry_price
                    gross_loss += min(0.0, pnl); hit = 'SL'; break
            else:
                if pd.notna(tp) and low <= tp:
                    exit_price = tp; exit_date = date; pnl = entry_price - exit_price
                    gross_profit += max(0.0, pnl); hit = 'TP'; break
                if pd.notna(sl) and high >= sl:
                    exit_price = sl; exit_date = date; pnl = entry_price - exit_price
                    gross_loss += min(0.0, pnl); hit = 'SL'; break

        if exit_price is None:
            last_close = df.iloc[min(idx+max_holding_days, len(df)-1)]['Close']
            exit_price = float(last_close)
            exit_date = df.iloc[min(idx+max_holding_days, len(df)-1)]['Date']
            pnl = (exit_price - entry_price) if direction == 'Long' else (entry_price - exit_price)
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
    gross_loss_abs = abs(gross_loss) if gross_loss < 0 else gross_loss
    return trades_df, gross_profit, gross_loss_abs

# -----------------------------
# backtest wrapper -> trades + stats
# -----------------------------
def run_backtest_with_stats(df, params):
    signals = produce_signals(df, params)
    if signals.empty:
        return pd.DataFrame(), {'total_profit':0.0, 'profit_factor':None, 'win_rate':0.0, 'total_trades':0}

    trades_df, gross_profit, gross_loss = simulate_outcomes(df, signals)
    if trades_df.empty:
        return pd.DataFrame(), {'total_profit':0.0, 'profit_factor':None, 'win_rate':0.0, 'total_trades':0}

    total_profit = trades_df['P/L'].sum()
    total_trades = len(trades_df)
    win_rate = (trades_df['P/L'] > 0).mean() * 100 if total_trades>0 else 0.0
    profit_factor = (gross_profit / gross_loss) if (gross_loss and gross_loss>0) else (np.inf if gross_profit>0 else None)

    stats = {
        'total_profit': total_profit,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }
    return trades_df, stats

# -----------------------------
# optimization (grid search)
# -----------------------------
def run_optimization(train_df, param_grid, max_param_combos=0):
    best_pf = -np.inf
    best = None
    tried = 0
    for p in param_grid:
        if max_param_combos and tried >= max_param_combos:
            break
        atr_sl, atr_tp, ema_period, min_conf, trade_mode = p
        params = {'atr_sl': atr_sl, 'atr_tp': atr_tp, 'ema_period': ema_period, 'min_conf': min_conf, 'trade_mode': trade_mode}
        trades, stats = run_backtest_with_stats(train_df, params)
        tried += 1
        if trades.empty or stats['profit_factor'] is None:
            continue
        pf = stats['profit_factor']
        # choose highest PF, tiebreaker = more trades
        if (pf > best_pf) or (pf == best_pf and stats['total_trades'] > (best[1]['total_trades'] if best else 0)):
            best_pf = pf
            best = (params, stats, trades)
    if best:
        return best[0], best[1], best[2]
    return None, None, None

# -----------------------------
# walk-forward driver (safe)
# -----------------------------
def walk_forward_driver(df, param_grid, train_frac=0.7, test_days=None, step_days=None, max_param_combos=0):
    n = len(df)
    if n < 30:
        # still attempt but warn user
        st.warning("Small dataset (<30 rows). Walk-forward results may be noisy.")
    train_size = int(train_frac * n)
    test_size = test_days if test_days is not None else max(int(0.3 * n), 10)
    step = step_days if step_days is not None else test_size
    wf_results = []
    params_history = []
    overall_test_trades = []

    # compute valid start range
    max_start = n - train_size - test_size
    if max_start < 0:
        return wf_results, params_history, pd.DataFrame()

    starts = list(range(0, max_start+1, step))
    for start in starts:
        train_df = df.iloc[start:start+train_size].reset_index(drop=True)
        test_df = df.iloc[start+train_size:start+train_size+test_size].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        best_params, best_stats, best_train_trades = run_optimization(train_df, param_grid, max_param_combos=max_param_combos)
        if best_params is None:
            continue
        params_history.append(best_params)
        test_trades, test_stats = run_backtest_with_stats(test_df, best_params)
        wf_results.append({
            'start': start,
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

    combined = pd.concat(overall_test_trades, ignore_index=True) if overall_test_trades else pd.DataFrame()
    return wf_results, params_history, combined

# -----------------------------
# continuation check
# -----------------------------
def continuation_check(df, best_params):
    if len(df) < 2:
        return "Not enough data to check continuation"
    yesterday = df.iloc[-2]
    today = df.iloc[-1]
    # produce yesterday signal within a small window so EntryIdx aligns
    slice_df = df.iloc[-40:].reset_index(drop=True)
    signals = produce_signals(slice_df, best_params)
    if signals.empty:
        return "No trade signaled yesterday (given params)"
    # try to find an entry matching yesterday date
    y_date = pd.to_datetime(yesterday['Date']).date()
    sig = None
    for _, s in signals.iterrows():
        if pd.to_datetime(s['EntryDate']).date() == y_date:
            sig = s
            break
    if sig is None:
        sig = signals.iloc[-1]  # fallback to latest signal in slice

    entry_price = sig['Entry']; sl = sig['SL']; tp = sig['TP']; direction = sig['Type']
    if direction == 'Long':
        if (pd.notna(sl) and pd.notna(tp)) and (today['Close'] > sl) and (today['Close'] < tp):
            return f"✅ Yesterday's LONG at {entry_price:.2f} still active (SL {sl:.2f}, TP {tp:.2f})"
        else:
            return "⛔ Yesterday's LONG closed or invalidated"
    else:
        if (pd.notna(sl) and pd.notna(tp)) and (today['Close'] < sl) and (today['Close'] > tp):
            return f"✅ Yesterday's SHORT at {entry_price:.2f} still active (SL {sl:.2f}, TP {tp:.2f})"
        else:
            return "⛔ Yesterday's SHORT closed or invalidated"

# -----------------------------
# UI: file upload & running
# -----------------------------
uploaded_file = st.file_uploader("Upload your daily OHLCV CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV file with daily OHLCV (Date, Open, High, Low, Close, Volume).")
    st.stop()

# read CSV and auto-map columns
raw = pd.read_csv(uploaded_file, dtype=str)  # read as str first to handle messy files
raw_columns = raw.columns.tolist()

# standardize whitespace in column names for mapping
raw.columns = [c.strip() for c in raw.columns]
mapping = auto_map_columns(raw)

# rename found columns to standard names
if mapping:
    raw = raw.rename(columns=mapping)
# lower-case fallback mapping too
lower_map = {c.lower(): c for c in raw.columns}
# if standard names aren't present yet, try matching lower-case keys
expected = ['Date','Open','High','Low','Close','Volume']
missing_expected = [e for e in expected if e not in raw.columns]
if missing_expected:
    # attempt additional case-insensitive matches
    rename_map = {}
    for col in raw.columns:
        lc = col.lower().strip()
        if 'date' == lc and 'Date' not in raw.columns: rename_map[col] = 'Date'
        if lc in ('open','o') and 'Open' not in raw.columns: rename_map[col] = 'Open'
        if lc in ('high','h') and 'High' not in raw.columns: rename_map[col] = 'High'
        if lc in ('low','l') and 'Low' not in raw.columns: rename_map[col] = 'Low'
        if lc in ('close','adj close','adj_close','ltp','last') and 'Close' not in raw.columns: rename_map[col] = 'Close'
        if lc in ('volume','vol','totaltradedqty','shares traded') and 'Volume' not in raw.columns: rename_map[col] = 'Volume'
    if rename_map:
        raw = raw.rename(columns=rename_map)

# final check
missing = [c for c in ['Date','Open','High','Low','Close','Volume'] if c not in raw.columns]
if missing:
    st.error(f"Missing required columns after mapping: {missing}\nColumns found: {list(raw.columns)}")
    st.stop()

# coerce types
df = raw.copy()
# parse date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)  # drop rows with unparseable date

# numeric conversion: remove commas and coerce
for c in ['Open','High','Low','Close','Volume']:
    df[c] = df[c].astype(str).str.replace(',', '').str.strip()
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)
n = len(df)
if n < 10:
    st.error(f"Not enough usable rows after parsing: {n}. Need more historical rows.")
    st.stop()

# compute indicators
df = compute_indicators(df)
st.success("Indicators computed ✅")
st.write(f"Rows: {n}  |  From {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

# param grid (expanded)
atr_sl_choices = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
atr_tp_choices = [1.5, 2.0, 2.5, 3.0, 3.5]
ema_choices = [50, 100, 150, 200, 250]
min_conf_choices = [1, 2, 3, 4]
trade_modes = ["Both", "Long Only", "Short Only"]
param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

# UI: safe slider defaults / ranges
max_len = max(2, n-1)
train_min = 10 if n > 20 else 2
train_max = max_len
train_default = min(max(int(0.7 * n), train_min), train_max)

test_min = 5
test_default = min(max(int(0.2 * n), test_min), max_len)
test_max = max_len

step_default = min(test_default, max(1, int(max(1, n*0.05))))
step_max = max_len

st.sidebar.header("Walk-Forward settings")
train_frac_slider = st.sidebar.slider("Train window size (rows)", min_value=train_min, max_value=train_max, value=train_default, step=1)
test_window = st.sidebar.slider("Test window size (rows)", min_value=test_min, max_value=test_max, value=test_default, step=1)
step_days = st.sidebar.slider("Step (rows) for sliding window", min_value=1, max_value=step_max, value=step_default, step=1)

st.sidebar.write(f"Parameter combos to try: {len(param_grid)}")
max_param_combos = st.sidebar.number_input("Limit param combos (0 = all, >0 speeds run)", min_value=0, value=0, step=1)

run_btn = st.button("Run Walk-Forward Optimization")
if run_btn:
    st.info("Running walk-forward optimization. This may take a while depending on dataset & grid size.")
    wf_results, params_hist, combined_test_trades = walk_forward_driver(df, param_grid, train_frac=None, test_days=test_window, step_days=step_days, max_param_combos=max_param_combos)
    # NOTE: in driver we ignored train_frac param to accept explicit train size; adjust by using train size directly:
    # But for simplicity, our driver used train_frac; to get exact train size pass as proportional train_frac
    # We will compute train_frac from train_frac_slider as absolute rows:
    # (To keep code simple, re-run driver with computed train_frac)
    train_frac = train_frac_slider / n
    wf_results, params_hist, combined_test_trades = walk_forward_driver(df, param_grid, train_frac=train_frac, test_days=test_window, step_days=step_days, max_param_combos=max_param_combos)

    if not wf_results:
        st.error("Walk-forward produced no valid windows/params. Try reducing test size or limiting param combos.")
        st.stop()

    st.success("Walk-forward completed ✅")
    st.write(f"Windows evaluated: {len(wf_results)}")

    # aggregate
    agg_profit = sum([w['test_stats']['total_profit'] for w in wf_results if w.get('test_stats')])
    agg_trades = sum([w['test_stats']['total_trades'] for w in wf_results if w.get('test_stats')])
    agg_win_sum = sum([w['test_stats']['win_rate'] * w['test_stats']['total_trades']/100 for w in wf_results if w.get('test_stats') and w['test_stats']['total_trades']>0])
    agg_win_rate = (agg_win_sum / agg_trades * 100) if agg_trades>0 else 0.0

    st.header("📈 Walk-Forward Summary (Aggregate Test Results)")
    st.write(f"Aggregate Test Profit (points): {agg_profit:.2f}")
    st.write(f"Aggregate Test Trades: {agg_trades}")
    st.write(f"Aggregate Test Win Rate (weighted): {agg_win_rate:.2f}%")

    # show recent windows
    recent = wf_results[-8:]
    rows = []
    for w in recent:
        bp = w['best_params']
        ts = w.get('test_stats') or {}
        rows.append({
            'train_from': w['train_from'].date(),
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
    st.dataframe(pd.DataFrame(rows))

    # combined trades
    if not combined_test_trades.empty:
        combined_test_trades['Cum_PnL'] = combined_test_trades['P/L'].cumsum()
        st.subheader("📜 Combined Test Trades (all windows)")
        st.dataframe(combined_test_trades[['EntryDate','ExitDate','Type','Entry','Exit','P/L','Hit']].head(200))
        fig, ax = plt.subplots()
        ax.plot(combined_test_trades['ExitDate'], combined_test_trades['Cum_PnL'], marker='o')
        ax.set_title("Walk-Forward Equity Curve (combined test trades)")
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # pick most recent best params
    most_recent_best = params_hist[-1]
    st.success(f"Using most-recent window's best params for live recommendation: {most_recent_best}")

    # LIVE recommendation
    df['EMA_TREND'] = df['Close'].ewm(span=most_recent_best['ema_period'], adjust=False).mean()
    last = df.iloc[-1]
    live_long, live_short = [], []
    if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_long.append("RSI Healthy"); live_short.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
    if last['Vol_Surge']: live_long.append("Volume Surge"); live_short.append("Volume Surge")

    st.header("📢 Live Recommendation (Most Recent Best Parameters)")
    if (last['Close'] > last['EMA_TREND']) and (len(live_long) >= most_recent_best['min_conf']) and (most_recent_best['trade_mode'] in ["Both","Long Only"]):
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - most_recent_best['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] + most_recent_best['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_long)}")
    elif (last['Close'] < last['EMA_TREND']) and (len(live_short) >= most_recent_best['min_conf']) and (most_recent_best['trade_mode'] in ["Both","Short Only"]):
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + most_recent_best['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] - most_recent_best['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_short)}")
    else:
        st.error("❌ No strong trade setup currently (most recent best params).")

    # Continuation check
    st.subheader("🔁 Continuation Check (yesterday -> today)")
    cont = continuation_check(df, most_recent_best)
    st.write(cont)
