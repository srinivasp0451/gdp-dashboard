# swing_walkforward_safe.py
# Single-file Streamlit app — no ta-lib required.
# - Walk-forward optimizer (profit factor)
# - Real P/L simulation
# - Continuation check (yesterday -> today)
# - Robust column mapping and NaN guards

import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

st.set_page_config(page_title="Swing Walk-Forward (Safe)", layout="wide")
st.title("Swing Trading — Walk-Forward + Continuation (No TA-Lib)")

# -------------------------
# Helpers: column mapping
# -------------------------
def auto_map_columns(df):
    # map many common variants to standard names
    cols = list(df.columns)
    lower = {c.lower().strip(): c for c in cols}
    mapping = {}
    candidates = {
        'Date': ['date', 'datetime', 'timestamp'],
        'Open': ['open', 'o'],
        'High': ['high', 'h'],
        'Low': ['low', 'l'],
        'Close': ['close', 'adj close', 'adj_close', 'ltp', 'last', 'close_price'],
        'Volume': ['volume', 'vol', 'totaltradedqty', 'shares traded']
    }
    for std, cands in candidates.items():
        found = None
        for c in cands:
            if c in lower:
                found = lower[c]
                break
        # if original had exact std name, prefer that
        if not found and std in cols:
            found = std
        if found:
            mapping[found] = std
    return mapping

# -------------------------
# Indicator calculations (no ta-lib)
# -------------------------
def compute_indicators(df):
    df = df.copy()
    # SMA
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (14) using True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # MACD and Signal (12,26,9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger mid (20) and volume surge
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] > (1.5 * df['AvgVol20'])

    # Keep Date as datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# -------------------------
# Produce signals (entry rows)
# -------------------------
def produce_signals(df, params):
    atr_sl = params['atr_sl']
    atr_tp = params['atr_tp']
    ema_period = params['ema_period']
    min_conf = params['min_conf']
    trade_mode = params['trade_mode']

    df = df.copy()
    df['EMA_TREND'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    signals = []
    start_idx = max(60, ema_period)  # ensure indicators available
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        if pd.isna(row['ATR']):
            continue  # skip entries without ATR

        long_conf = []
        short_conf = []
        # long confluences
        if pd.notna(row['SMA20']) and pd.notna(row['SMA50']) and row['SMA20'] > row['SMA50']:
            long_conf.append("SMA Bullish")
        if pd.notna(row['RSI']) and 30 < row['RSI'] < 70:
            long_conf.append("RSI Healthy")
            short_conf.append("RSI Healthy")
        if pd.notna(row['MACD']) and pd.notna(row['Signal']) and row['MACD'] > row['Signal']:
            long_conf.append("MACD Bullish")
        if pd.notna(row['BB_Mid']) and row['Close'] > row['BB_Mid']:
            long_conf.append("BB Breakout")
        if row.get('Vol_Surge', False):
            long_conf.append("Volume Surge"); short_conf.append("Volume Surge")
        # short confluences
        if pd.notna(row['SMA20']) and pd.notna(row['SMA50']) and row['SMA20'] < row['SMA50']:
            short_conf.append("SMA Bearish")
        if pd.notna(row['MACD']) and pd.notna(row['Signal']) and row['MACD'] < row['Signal']:
            short_conf.append("MACD Bearish")
        if pd.notna(row['BB_Mid']) and row['Close'] < row['BB_Mid']:
            short_conf.append("BB Breakdown")

        # entry
        if (row['Close'] > row['EMA_TREND']) and (len(long_conf) >= min_conf) and (trade_mode in ["Both", "Long Only"]):
            entry = float(row['Close'])
            sl = entry - atr_sl * float(row['ATR'])
            tp = entry + atr_tp * float(row['ATR'])
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type': 'Long', 'Entry': entry, 'SL': sl, 'TP': tp, 'Reasons': ", ".join(long_conf)})
        elif (row['Close'] < row['EMA_TREND']) and (len(short_conf) >= min_conf) and (trade_mode in ["Both", "Short Only"]):
            entry = float(row['Close'])
            sl = entry + atr_sl * float(row['ATR'])
            tp = entry - atr_tp * float(row['ATR'])
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type': 'Short', 'Entry': entry, 'SL': sl, 'TP': tp, 'Reasons': ", ".join(short_conf)})

    return pd.DataFrame(signals)

# -------------------------
# Simulate outcomes & PF
# -------------------------
def simulate_outcomes(df, signals, max_holding_days=10):
    if signals.empty:
        return pd.DataFrame(), 0.0, 0.0

    trades = []
    gross_profit = 0.0
    gross_loss = 0.0

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

        # scan forward up to max_holding_days
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
            if pnl >= 0: gross_profit += pnl
            else: gross_loss += pnl
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
    stats = {'total_profit': total_profit, 'profit_factor': profit_factor, 'win_rate': win_rate, 'total_trades': total_trades, 'gross_profit': gross_profit, 'gross_loss': gross_loss}
    return trades_df, stats

# -------------------------
# Grid optimization (train)
# -------------------------
def run_optimization(train_df, param_grid, max_param_combos=0):
    best_pf = -np.inf
    best_params = None
    best_stats = None
    best_trades = None
    tried = 0
    for params in param_grid:
        if max_param_combos and tried >= max_param_combos:
            break
        atr_sl, atr_tp, ema_period, min_conf, trade_mode = params
        param_obj = {'atr_sl': atr_sl, 'atr_tp': atr_tp, 'ema_period': ema_period, 'min_conf': min_conf, 'trade_mode': trade_mode}
        trades, stats = run_backtest_with_stats(train_df, param_obj)
        tried += 1
        if trades.empty or stats['profit_factor'] is None:
            continue
        pf = stats['profit_factor']
        if (pf > best_pf) or (pf == best_pf and stats['total_trades'] > (best_stats['total_trades'] if best_stats else 0)):
            best_pf = pf
            best_params = param_obj
            best_stats = stats
            best_trades = trades
    return best_params, best_stats, best_trades

# -------------------------
# Walk-Forward driver
# -------------------------
def walk_forward_driver(df, param_grid, train_frac=0.7, test_days=None, step_days=None, max_param_combos=0):
    n = len(df)
    train_size = int(train_frac * n)
    test_size = test_days if test_days else max(int(0.3 * n), 10)
    step = step_days if step_days else test_size
    wf_results = []
    params_history = []
    overall_test_trades = []

    max_start = n - train_size - test_size
    if max_start < 0:
        return wf_results, params_history, pd.DataFrame()

    for start in range(0, max_start+1, step):
        train_df = df.iloc[start:start+train_size].reset_index(drop=True)
        test_df = df.iloc[start+train_size:start+train_size+test_size].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        best_params, best_stats, best_train_trades = run_optimization(train_df, param_grid, max_param_combos=max_param_combos)
        if best_params is None:
            continue
        params_history.append(best_params)
        test_trades, test_stats = run_backtest_with_stats(test_df, best_params)
        wf_results.append({'start': start, 'train_from': train_df['Date'].iloc[0], 'train_to': train_df['Date'].iloc[-1], 'test_from': test_df['Date'].iloc[0], 'test_to': test_df['Date'].iloc[-1], 'best_params': best_params, 'train_stats': best_stats, 'test_stats': test_stats, 'test_trades': test_trades})
        if not test_trades.empty:
            overall_test_trades.append(test_trades)

    combined = pd.concat(overall_test_trades, ignore_index=True) if overall_test_trades else pd.DataFrame()
    return wf_results, params_history, combined

# -------------------------
# Continuation check
# -------------------------
def continuation_check(df, best_params):
    if best_params is None:
        return "No params available"
    if len(df) < 2:
        return "Not enough rows (need 2)"
    # examine signals in a small tail slice so EntryIdx aligns
    slice_df = df.iloc[-40:].reset_index(drop=True)
    signals = produce_signals(slice_df, best_params)
    if signals.empty:
        return "No signal yesterday"
    # try match yesterday's date
    y_date = pd.to_datetime(df.iloc[-2]['Date']).date()
    sig = None
    for _, s in signals.iterrows():
        if pd.to_datetime(s['EntryDate']).date() == y_date:
            sig = s; break
    if sig is None:
        sig = signals.iloc[-1]
    entry = sig['Entry']; sl = sig['SL']; tp = sig['TP']; direction = sig['Type']
    today_close = df.iloc[-1]['Close']
    if direction == 'Long':
        if (pd.notna(sl) and pd.notna(tp)) and (today_close > sl) and (today_close < tp):
            return f"✅ Yesterday's LONG at {entry:.2f} still active (SL {sl:.2f}, TP {tp:.2f})"
        else:
            return "⛔ Yesterday's LONG closed/invalidated"
    else:
        if (pd.notna(sl) and pd.notna(tp)) and (today_close < sl) and (today_close > tp):
            return f"✅ Yesterday's SHORT at {entry:.2f} still active (SL {sl:.2f}, TP {tp:.2f})"
        else:
            return "⛔ Yesterday's SHORT closed/invalidated"

# -------------------------
# UI: upload and run
# -------------------------
uploaded_file = st.file_uploader("Upload daily OHLCV CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
if not uploaded_file:
    st.info("Upload your CSV file (daily OHLCV). Column name variants (close, ltp, adj close) are supported.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file, dtype=str)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Normalize column names, map to standard names
raw.columns = [c.strip() for c in raw.columns]
mapping = auto_map_columns(raw)
if mapping:
    raw = raw.rename(columns=mapping)
# additional case-insensitive tries
rename_map = {}
for col in raw.columns:
    lc = col.lower().strip()
    if lc in ('date','datetime','timestamp') and 'Date' not in raw.columns: rename_map[col] = 'Date'
    if lc in ('open','o') and 'Open' not in raw.columns: rename_map[col] = 'Open'
    if lc in ('high','h') and 'High' not in raw.columns: rename_map[col] = 'High'
    if lc in ('low','l') and 'Low' not in raw.columns: rename_map[col] = 'Low'
    if lc in ('close','adj close','adj_close','ltp','last','close_price') and 'Close' not in raw.columns: rename_map[col] = 'Close'
    if lc in ('volume','vol','totaltradedqty','shares traded') and 'Volume' not in raw.columns: rename_map[col] = 'Volume'
if rename_map:
    raw = raw.rename(columns=rename_map)

# final required check
missing = [c for c in ['Date','Open','High','Low','Close','Volume'] if c not in raw.columns]
if missing:
    st.error(f"Missing required columns after mapping: {missing}\nFound columns: {list(raw.columns)}")
    st.stop()

# coerce types
df = raw.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)
for c in ['Open','High','Low','Close','Volume']:
    df[c] = df[c].astype(str).str.replace(',', '').str.strip()
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)
n = len(df)
if n < 30:
    st.warning(f"Small dataset: {n} rows. Walk-forward may be noisy but will still run.")

# compute indicators
df = compute_indicators(df)
st.success("Indicators computed")

# param grid (expanded)
atr_sl_choices = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
atr_tp_choices = [1.5, 2.0, 2.5, 3.0, 3.5]
ema_choices = [50, 100, 150, 200, 250]
min_conf_choices = [1, 2, 3, 4]
trade_modes = ["Both", "Long Only", "Short Only"]
param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

# UI controls with safe defaults
st.sidebar.header("Walk-forward settings")
train_frac = st.sidebar.slider("Train fraction (of data)", 0.5, 0.85, 0.7)
test_window = st.sidebar.number_input("Test window (rows)", min_value=5, max_value=max(5, n//2), value=max(10, min(30, max(5, n//4))), step=1)
step_days = st.sidebar.number_input("Step (rows) per window", min_value=1, max_value=max(1, test_window), value=max(1, min(test_window, max(1, n//10))), step=1)
max_param_combos = st.sidebar.number_input("Limit param combos (0 = all)", min_value=0, value=0, step=1)

st.sidebar.write(f"Param combos total: {len(param_grid)} (limit with 'Limit param combos' if run is slow)")

if st.button("Run Walk-Forward + Live Recommendation"):
    with st.spinner("Running walk-forward (may take time)..."):
        wf_results, params_hist, combined_test_trades = walk_forward_driver(df, param_grid, train_frac=train_frac, test_days=test_window, step_days=step_days, max_param_combos=max_param_combos)

    if not wf_results:
        st.error("Walk-forward produced no valid windows or params. Try smaller test window or reduce grid.")
        st.stop()

    # summary
    st.success("Walk-forward completed")
    st.write(f"Windows evaluated: {len(wf_results)}")
    agg_profit = sum([w['test_stats']['total_profit'] for w in wf_results if w.get('test_stats')])
    agg_trades = sum([w['test_stats']['total_trades'] for w in wf_results if w.get('test_stats')])
    agg_win_sum = sum([w['test_stats']['win_rate'] * w['test_stats']['total_trades']/100 for w in wf_results if w.get('test_stats') and w['test_stats']['total_trades']>0])
    agg_win_rate = (agg_win_sum / agg_trades * 100) if agg_trades>0 else 0.0
    st.header("Walk-Forward Aggregate")
    st.write(f"Aggregate Test Profit: {agg_profit:.2f}")
    st.write(f"Aggregate Test Trades: {agg_trades}")
    st.write(f"Aggregate Win Rate (weighted): {agg_win_rate:.2f}%")

    # show recent windows
    recent = wf_results[-8:]
    rows = []
    for w in recent:
        bp = w['best_params']
        ts = w.get('test_stats') or {}
        rows.append({'train_from': w['train_from'].date(), 'train_to': w['train_to'].date(), 'test_from': w['test_from'].date(), 'test_to': w['test_to'].date(), 'atr_sl': bp['atr_sl'], 'atr_tp': bp['atr_tp'], 'ema': bp['ema_period'], 'min_conf': bp['min_conf'], 'mode': bp['trade_mode'], 'test_profit': ts.get('total_profit', None), 'test_trades': ts.get('total_trades', None), 'test_pf': ts.get('profit_factor', None)})
    st.subheader("Recent windows — best parameters")
    st.dataframe(pd.DataFrame(rows))

    # combined trades & equity
    if not combined_test_trades.empty:
        if 'P/L' in combined_test_trades.columns:
            combined_test_trades['Cum_PnL'] = combined_test_trades['P/L'].cumsum()
        st.subheader("Combined test trades (sample)")
        st.dataframe(combined_test_trades.head(200))
        if 'Cum_PnL' in combined_test_trades.columns:
            fig, ax = plt.subplots()
            ax.plot(combined_test_trades['ExitDate'], combined_test_trades['Cum_PnL'], marker='o')
            ax.set_title("Equity Curve (combined test trades)")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # pick most recent best params for live recommendation
    most_recent_best = params_hist[-1]
    st.success(f"Most-recent best params: {most_recent_best}")

    # Live recommendation
    df['EMA_TREND'] = df['Close'].ewm(span=most_recent_best['ema_period'], adjust=False).mean()
    last = df.iloc[-1]
    live_long, live_short = [], []
    if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_long.append("RSI Healthy"); live_short.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
    if last['Vol_Surge']: live_long.append("Volume Surge"); live_short.append("Volume Surge")

    st.header("Live Recommendation (most recent params)")
    if (last['Close'] > last['EMA_TREND']) and (len(live_long) >= most_recent_best['min_conf']) and (most_recent_best['trade_mode'] in ["Both","Long Only"]):
        sl = last['Close'] - most_recent_best['atr_sl']*last['ATR']
        tp = last['Close'] + most_recent_best['atr_tp']*last['ATR']
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Reasons: {', '.join(live_long)}")
    elif (last['Close'] < last['EMA_TREND']) and (len(live_short) >= most_recent_best['min_conf']) and (most_recent_best['trade_mode'] in ["Both","Short Only"]):
        sl = last['Close'] + most_recent_best['atr_sl']*last['ATR']
        tp = last['Close'] - most_recent_best['atr_tp']*last['ATR']
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Reasons: {', '.join(live_short)}")
    else:
        st.error("No strong trade setup currently")

    # continuation check
    st.subheader("Continuation Check (yesterday → today)")
    cont = continuation_check(df, most_recent_best)
    st.write(cont)

st.markdown("---")
st.caption("If you see any error, paste the exact traceback and the CSV header (first row) here and I'll fix it immediately. Again — sorry for all the broken fragments earlier.")
