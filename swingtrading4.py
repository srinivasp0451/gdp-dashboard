# streamlit_swing_walkforward_final_with_export.py
import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import io, base64
from datetime import datetime

st.set_page_config(page_title="Swing Walk-Forward + Export", layout="wide")
st.title("Swing Trading — Walk-Forward Backtest + Export")

# ---------------- utilities ----------------
def auto_map_columns(df):
    cols = list(df.columns)
    lower = {c.lower().strip(): c for c in cols}
    mapping = {}
    candidates = {
        'Date': ['date', 'datetime', 'timestamp'],
        'Open': ['open', 'o'],
        'High': ['high', 'h'],
        'Low': ['low', 'l'],
        'Close': ['close', 'adj close', 'adj_close', 'close_price', 'ltp', 'last'],
        'Volume': ['volume', 'vol', 'totaltradedqty', 'shares traded']
    }
    for std, cands in candidates.items():
        found = None
        for c in cands:
            if c in lower:
                found = lower[c]
                break
        if not found and std in cols:
            found = std
        if found:
            mapping[found] = std
    return mapping

def compute_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # RSI(14)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # True Range and ATR(14)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # MACD (12,26) and Signal (9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger mid and volume surge
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] > (1.5 * df['AvgVol20'])

    # ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

# produce entry signals but allow passing a window offset to only keep signals inside test portion
def produce_signals_window(df_window, params, test_start_idx_in_window):
    """
    df_window includes train+test (so indicators are available).
    We will only keep signals whose EntryIdx >= test_start_idx_in_window (i.e., signals that occur in test period).
    """
    atr_sl = params['atr_sl']; atr_tp = params['atr_tp']; ema_period = params['ema_period']
    min_conf = params['min_conf']; trade_mode = params['trade_mode']

    df = df_window.copy()
    df['EMA_TREND'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    signals = []
    start_idx = max(60, ema_period)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        if pd.isna(row['ATR']):
            continue
        long_conf = []; short_conf = []
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

        # entry rules
        if (row['Close'] > row['EMA_TREND']) and (len(long_conf) >= min_conf) and (trade_mode in ["Both","Long Only"]):
            entry = float(row['Close'])
            sl = entry - atr_sl * float(row['ATR'])
            tp = entry + atr_tp * float(row['ATR'])
            reasons = ", ".join(long_conf)
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type':'Long', 'Entry':entry, 'SL':sl, 'TP':tp, 'Reasons':reasons})
        elif (row['Close'] < row['EMA_TREND']) and (len(short_conf) >= min_conf) and (trade_mode in ["Both","Short Only"]):
            entry = float(row['Close'])
            sl = entry + atr_sl * float(row['ATR'])
            tp = entry - atr_tp * float(row['ATR'])
            reasons = ", ".join(short_conf)
            signals.append({'EntryIdx': i, 'EntryDate': row['Date'], 'Type':'Short', 'Entry':entry, 'SL':sl, 'TP':tp, 'Reasons':reasons})

    signals_df = pd.DataFrame(signals)
    if signals_df.empty:
        return signals_df
    # Keep only signals that happen in test portion
    sigs_in_test = signals_df[signals_df['EntryIdx'] >= test_start_idx_in_window].reset_index(drop=True)
    return sigs_in_test

# simulate outcomes scanning forward candle-by-candle
def simulate_outcomes_window(df_window, signals, max_holding_days=20):
    """
    df_window: train+test window (pandas dataframe)
    signals: produced by produce_signals_window and their EntryIdx are indices in df_window
    returns trades_df (EntryDate, ExitDate, Entry, Exit, SL, TP, Type, P/L, Hit, Reasons)
    """
    if signals.empty:
        return pd.DataFrame(), 0.0, 0.0
    trades = []
    gross_profit = 0.0
    gross_loss = 0.0

    for _, s in signals.iterrows():
        idx = int(s['EntryIdx'])
        entry_price = s['Entry']
        sl = s['SL']; tp = s['TP']; direction = s['Type']; reasons = s.get('Reasons','')
        exit_price = None; exit_date = None; pnl = None; hit = None

        for j in range(idx+1, min(idx+1+max_holding_days, len(df_window))):
            high = df_window.iloc[j]['High']; low = df_window.iloc[j]['Low']; close = df_window.iloc[j]['Close']; date = df_window.iloc[j]['Date']
            sma20 = df_window.iloc[j]['SMA20']; sma50 = df_window.iloc[j]['SMA50']

            if direction == 'Long':
                if pd.notna(tp) and high >= tp:
                    exit_price = tp; exit_date = date; pnl = exit_price - entry_price; hit = 'TP'; gross_profit += max(0.0,pnl); break
                if pd.notna(sl) and low <= sl:
                    exit_price = sl; exit_date = date; pnl = exit_price - entry_price; hit = 'SL'; gross_loss += min(0.0,pnl); break
                if pd.notna(sma20) and pd.notna(sma50) and sma20 < sma50:
                    exit_price = close; 
                    exit_date = date; 
                    pnl = exit_price - entry_price; 
                    hit = 'Invalidation'; 
                    if pnl >= 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)# we handle adding profit/loss below to keep consistent types
                    if pnl >=0: 
                        gross_profit += pnl
                    else: 
                        gross_loss += pnl
                    break
            else: # Short
                if pd.notna(tp) and low <= tp:
                    exit_price = tp; exit_date = date; pnl = entry_price - exit_price; hit = 'TP'; gross_profit += max(0.0,pnl); break
                if pd.notna(sl) and high >= sl:
                    exit_price = sl; exit_date = date; pnl = entry_price - exit_price; hit = 'SL'; gross_loss += min(0.0,pnl); break
                if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50:
                    exit_price = close; exit_date = date; pnl = entry_price - exit_price; hit = 'Invalidation'
                    if pnl >=0: gross_profit += pnl
                    else: gross_loss += pnl
                    break

        if exit_price is None:
            idx_close = min(idx + max_holding_days, len(df_window)-1)
            exit_price = float(df_window.iloc[idx_close]['Close'])
            exit_date = df_window.iloc[idx_close]['Date']
            pnl = (exit_price - entry_price) if direction == 'Long' else (entry_price - exit_price)
            if pnl >=0: 
                gross_profit += pnl
            else: 
                gross_loss += pnl
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
            'Reasons': reasons
        })

    trades_df = pd.DataFrame(trades)
    gross_loss_abs = abs(gross_loss) if gross_loss < 0 else gross_loss
    return trades_df, gross_profit, gross_loss_abs

# wrapper to run backtest and return trades+stats for a provided (train+test) window (but only trades inside test)
def backtest_window_and_stats(df_window, test_start_idx_in_window, params, max_holding_days=20):
    signals = produce_signals_window(df_window, params, test_start_idx_in_window)
    if signals.empty:
        return pd.DataFrame(), {'total_profit':0.0,'profit_factor':None,'win_rate':0.0,'total_trades':0}
    trades_df, gross_profit, gross_loss = simulate_outcomes_window(df_window, signals, max_holding_days=max_holding_days)
    if trades_df.empty:
        return pd.DataFrame(), {'total_profit':0.0,'profit_factor':None,'win_rate':0.0,'total_trades':0}
    total_profit = trades_df['P/L'].sum()
    total_trades = len(trades_df)
    win_rate = (trades_df['P/L'] > 0).mean() * 100 if total_trades>0 else 0.0
    profit_factor = (gross_profit / gross_loss) if (gross_loss and gross_loss>0) else (np.inf if gross_profit>0 else None)
    stats = {'total_profit': total_profit, 'profit_factor':profit_factor, 'win_rate':win_rate, 'total_trades':total_trades, 'gross_profit':gross_profit, 'gross_loss':gross_loss}
    return trades_df, stats

# optimization on train portion (grid)
def optimize_on_train(train_df, param_grid, max_param_combos=0):
    best_pf = -np.inf
    best = (None,None,None)
    tried = 0
    for p in param_grid:
        if max_param_combos and tried >= max_param_combos:
            break
        atr_sl, atr_tp, ema_period, min_conf, trade_mode = p
        params = {'atr_sl':atr_sl,'atr_tp':atr_tp,'ema_period':ema_period,'min_conf':min_conf,'trade_mode':trade_mode}
        # Use train_df as full frame for signals (we will only take trades inside train when analyzing)
        trades, stats = backtest_window_and_stats(train_df, test_start_idx_in_window= len(train_df), params=params)  # empty test idx -> get train trades if any
        tried += 1
        if trades.empty or stats['profit_factor'] is None:
            continue
        pf = stats['profit_factor']
        if (pf > best_pf) or (pf == best_pf and stats['total_trades'] > (best[1]['total_trades'] if best[1] else 0)):
            best_pf = pf
            best = (params, stats, trades)
    return best  # (params, stats, trades) or (None,None,None)

# Walk-forward driver (carry lookback)
def walk_forward(df, param_grid, train_frac=0.7, test_size=None, step=None, lookback_buffer=300, max_param_combos=0):
    """
    df: full dataframe with indicators
    param_grid: list of tuples (atr_sl,atr_tp,ema,min_conf,mode)
    train_frac: fraction to use for training size (proportional)
    test_size: rows for test window (if None, uses ~30% of data)
    step: slide step (if None, equals test_size)
    lookback_buffer: rows to keep before train start so test window includes sufficient history
    """
    n = len(df)
    train_size = int(train_frac * n)
    if test_size is None:
        test_size = max(int(0.3 * n), 10)
    if step is None:
        step = test_size
    wf_results = []
    params_history = []
    combined_trades = []

    max_start = n - train_size - test_size
    if max_start < 0:
        return wf_results, params_history, pd.DataFrame()

    for start in range(0, max_start+1, step):
        train_start = start
        train_end = start + train_size
        test_end = train_end + test_size
        # to ensure indicators available at test start, include lookback rows before train_start (but not <0)
        window_start = max(0, train_start - lookback_buffer)
        df_window = df.iloc[window_start:test_end].reset_index(drop=True)
        # compute index of test start within df_window
        test_start_idx_in_window = train_start - window_start

        # optimize on train (we want params that do well *on train part*); pass train subframe with lookback too
        train_subframe = df.iloc[window_start:train_end].reset_index(drop=True)
        best_params, best_stats, best_train_trades = optimize_on_train(train_subframe, param_grid, max_param_combos=max_param_combos)
        if best_params is None:
            # no valid params found
            continue
        params_history.append(best_params)

        # test on test portion using df_window but only keep signals with EntryIdx >= test_start_idx_in_window
        test_trades, test_stats = backtest_window_and_stats(df_window, test_start_idx_in_window=test_start_idx_in_window, params=best_params)
        wf_results.append({'start':start, 'train_from': df.iloc[train_start]['Date'], 'train_to': df.iloc[train_end-1]['Date'],
                           'test_from': df.iloc[train_end]['Date'], 'test_to': df.iloc[test_end-1]['Date'],
                           'best_params': best_params, 'train_stats': best_stats, 'test_stats': test_stats, 'test_trades': test_trades})
        if not test_trades.empty:
            # Map EntryIdx/ExitIdx (which are window indices) to actual dates already are EntryDate/ExitDate in returned trades
            combined_trades.append(test_trades)

    combined = pd.concat(combined_trades, ignore_index=True) if combined_trades else pd.DataFrame()
    return wf_results, params_history, combined

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload daily OHLCV CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
if not uploaded_file:
    st.info("Upload CSV. Common variants like 'close', 'ltp', 'adj close' are accepted.")
    st.stop()

raw = pd.read_csv(uploaded_file, dtype=str)
raw.columns = [c.strip() for c in raw.columns]
mapping = auto_map_columns(raw)
if mapping:
    raw = raw.rename(columns=mapping)
# fallback renames (case-insensitive)
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

missing = [c for c in ['Date','Open','High','Low','Close','Volume'] if c not in raw.columns]
if missing:
    st.error(f"Missing required columns after mapping: {missing}\nFound columns: {list(raw.columns)}")
    st.stop()

# coerce types
df = raw.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)
for c in ['Open','High','Low','Close','Volume']:
    df[c] = df[c].astype(str).str.replace(',','').str.strip()
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)
n = len(df)
if n < 30:
    st.warning(f"Small dataset ({n} rows). Walk-forward may be noisy but will run.")

# compute indicators (global)
df = compute_indicators(df)
st.success("Indicators computed ✅")
st.write(f"Dataset: {n} rows | {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

# param grid (expanded)
atr_sl_choices = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
atr_tp_choices = [1.5, 2.0, 2.5, 3.0, 3.5]
ema_choices = [50, 100, 150, 200, 250]
min_conf_choices = [1, 2, 3, 4]
trade_modes = ["Both", "Long Only", "Short Only"]
param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

st.sidebar.header("Walk-Forward Settings")
train_frac = st.sidebar.slider("Train fraction", 0.5, 0.85, 0.7)
test_window = st.sidebar.number_input("Test window (rows)", min_value=5, max_value=max(5, n//2), value=max(10, min(30, max(5, n//4))), step=1)
step_days = st.sidebar.number_input("Step (rows)", min_value=1, max_value=max(1,test_window), value=max(1, min(test_window, max(1, n//10))), step=1)
lookback_buffer = st.sidebar.number_input("Lookback buffer rows (for indicators)", min_value=50, max_value=1000, value=300, step=10)
max_param_combos = st.sidebar.number_input("Limit param combos (0=all)", min_value=0, value=0, step=1)

if st.button("Run Walk-Forward + Export"):
    with st.spinner("Running walk-forward (this can be slow)..."):
        wf_results, params_hist, combined_trades = walk_forward(df, param_grid, train_frac=train_frac, test_size=test_window, step=step_days, lookback_buffer=lookback_buffer, max_param_combos=max_param_combos)

    if not wf_results:
        st.error("No valid windows/params found. Try reducing test size or limiting param combos.")
        st.stop()

    # show aggregate metrics
    if combined_trades.empty:
        st.warning("Walk-forward completed but no trades were generated in test windows. Check parameters (EMA period vs test window size) or increase dataset length.")
    else:
        combined_trades = combined_trades.copy()
        total_pnl = combined_trades['P/L'].sum()
        win_rate = (combined_trades['P/L'] > 0).mean() * 100
        gross_profit = combined_trades.loc[combined_trades['P/L']>0, 'P/L'].sum()
        gross_loss = combined_trades.loc[combined_trades['P/L']<0, 'P/L'].sum()
        profit_factor = (gross_profit / abs(gross_loss)) if (gross_loss != 0) else (np.inf if gross_profit>0 else None)

        st.header("Aggregate Walk-Forward Test Results")
        st.write(f"Net Profit (points): {total_pnl:.2f}")
        st.write(f"Weighted Win Rate: {win_rate:.2f}%")
        st.write(f"Profit Factor: {profit_factor:.2f}" if (profit_factor is not None and not np.isinf(profit_factor)) else ("Profit Factor: inf" if profit_factor==np.inf else "Profit Factor: N/A"))

        # show sample trades
        display_cols = ['EntryDate','ExitDate','Type','Entry','Exit','SL','TP','P/L','Hit','Reasons']
        available = [c for c in display_cols if c in combined_trades.columns]
        st.subheader("Sample Combined Test Trades")
        st.dataframe(combined_trades[available].head(500))

        # equity curve
        if 'P/L' in combined_trades.columns:
            combined_trades['Cum_PnL'] = combined_trades['P/L'].cumsum()
            fig, ax = plt.subplots()
            ax.plot(combined_trades['ExitDate'], combined_trades['Cum_PnL'], marker='o')
            ax.set_title("Equity Curve (combined test trades)")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # CSV export (download)
        csv_buf = combined_trades.to_csv(index=False).encode()
        st.download_button("Download combined test trades CSV", data=csv_buf, file_name=f"combined_test_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv')

    # show most recent best params
    most_recent = params_hist[-1]
    st.success(f"Most-recent best params: {most_recent}")

    # live recommendation using most recent params
    df['EMA_TREND'] = df['Close'].ewm(span=most_recent['ema_period'], adjust=False).mean()
    last = df.iloc[-1]
    live_long, live_short = [], []
    if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_long.append("RSI Healthy"); live_short.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
    if last['Vol_Surge']: live_long.append("Volume Surge"); live_short.append("Volume Surge")

    st.header("Live Recommendation (Most Recent Best Params)")
    if (last['Close'] > last['EMA_TREND']) and (len(live_long) >= most_recent['min_conf']) and (most_recent['trade_mode'] in ["Both","Long Only"]):
        sl = last['Close'] - most_recent['atr_sl'] * last['ATR']; tp = last['Close'] + most_recent['atr_tp'] * last['ATR']
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Reasons: {', '.join(live_long)}")
    elif (last['Close'] < last['EMA_TREND']) and (len(live_short) >= most_recent['min_conf']) and (most_recent['trade_mode'] in ["Both","Short Only"]):
        sl = last['Close'] + most_recent['atr_sl'] * last['ATR']; tp = last['Close'] - most_recent['atr_tp'] * last['ATR']
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Reasons: {', '.join(live_short)}")
    else:
        st.error("No strong trade setup currently (most recent best params)")

    # continuation check
    st.subheader("Continuation Check (yesterday -> today)")
    cont = continuation_check = None
    try:
        cont = (lambda df_, params_: (lambda: continuation_check_fn(df_, params_)) )()(df, most_recent)  # call helper below
    except Exception:
        # fallback simple check (reproduce earlier logic inline)
        cont = continuation_check(df, most_recent)
    st.write(cont)

# simple helper continuation_check_fn used above (kept external for readability)
def continuation_check_fn(df, best_params):
    if best_params is None:
        return "No params available"
    if len(df) < 2:
        return "Need at least 2 rows"
    # use a small slice for alignment
    slice_df = df.iloc[-40:].reset_index(drop=True)
    signals = produce_signals_window(slice_df, best_params, test_start_idx_in_window=0)
    if signals.empty:
        return "No signal recently"
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

st.caption("If anything looks off, paste the full traceback plus the CSV header (column names) and I'll patch it immediately.")
