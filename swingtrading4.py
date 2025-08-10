import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from itertools import product
from datetime import timedelta

st.set_page_config(page_title="Swing Trading Screener — Walk-Forward + Continuation", layout="wide")
st.title("📊 Swing Trading Screener — Walk-Forward Optimizer + Continuation Check")

# ------------ helpers ------------
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

def compute_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] > (1.5 * df['AvgVol20'])
    return df

def generate_trades(df, params):
    """
    Run the strategy on df using params.
    Returns trades list of dicts and statistics.
    """
    atr_sl, atr_tp, ema_trend_period, min_conf, trade_mode = params
    df = df.copy()
    df['EMA_TREND'] = df['Close'].ewm(span=ema_trend_period, adjust=False).mean()

    trades = []
    position = None
    direction = None

    # start from index that likely has computed indicators
    start_idx = max(ema_trend_period, 50, 60)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        long_conf, short_conf = [], []

        if row['SMA20'] > row['SMA50']: long_conf.append("SMA Bullish")
        if 30 < row['RSI'] < 70: long_conf.append("RSI Healthy")
        if row['MACD'] > row['Signal']: long_conf.append("MACD Bullish")
        if row['Close'] > row['BB_Mid']: long_conf.append("BB Breakout")
        if row['Vol_Surge']: long_conf.append("Volume Surge")

        if row['SMA20'] < row['SMA50']: short_conf.append("SMA Bearish")
        if 30 < row['RSI'] < 70: short_conf.append("RSI Healthy")
        if row['MACD'] < row['Signal']: short_conf.append("MACD Bearish")
        if row['Close'] < row['BB_Mid']: short_conf.append("BB Breakdown")
        if row['Vol_Surge']: short_conf.append("Volume Surge")

        # entry
        if position is None and row['Close'] > row['EMA_TREND'] and len(long_conf) >= min_conf and trade_mode in ["Both", "Long Only"]:
            position, direction = "Open", "Long"
            entry_price = row['Close']
            sl = entry_price - atr_sl * row['ATR']
            target = entry_price + atr_tp * row['ATR']
            entry_date, reason = row['Date'], ", ".join(long_conf)
        elif position is None and row['Close'] < row['EMA_TREND'] and len(short_conf) >= min_conf and trade_mode in ["Both", "Short Only"]:
            position, direction = "Open", "Short"
            entry_price = row['Close']
            sl = entry_price + atr_sl * row['ATR']
            target = entry_price - atr_tp * row['ATR']
            entry_date, reason = row['Date'], ", ".join(short_conf)

        # exit
        if position == "Open" and direction == "Long":
            if (row['Close'] <= sl) or (row['Close'] >= target) or (row['SMA20'] < row['SMA50']):
                pnl = row['Close'] - entry_price
                trades.append({"Entry Date": entry_date, "Entry Price": entry_price, "Stop Loss": sl,
                               "Target": target, "Exit Date": row['Date'], "Exit Price": row['Close'],
                               "Direction": direction, "Reason": reason, "P/L": pnl})
                position = None
        elif position == "Open" and direction == "Short":
            if (row['Close'] >= sl) or (row['Close'] <= target) or (row['SMA20'] > row['SMA50']):
                pnl = entry_price - row['Close']
                trades.append({"Entry Date": entry_date, "Entry Price": entry_price, "Stop Loss": sl,
                               "Target": target, "Exit Date": row['Date'], "Exit Price": row['Close'],
                               "Direction": direction, "Reason": reason, "P/L": pnl})
                position = None

    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['P/L'].sum() if not trades_df.empty else 0.0
    gross_profit = trades_df[trades_df['P/L'] > 0]['P/L'].sum() if not trades_df.empty else 0.0
    gross_loss = -trades_df[trades_df['P/L'] < 0]['P/L'].sum() if not trades_df.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    win_rate = (trades_df['P/L'] > 0).mean() * 100 if not trades_df.empty else 0.0

    stats = dict(total_profit=total_profit, profit_factor=profit_factor, win_rate=win_rate, total_trades=len(trades_df))
    return trades_df, stats

# ------------ UI: file upload & column mapping ------------
uploaded_file = st.file_uploader("Upload your CSV file (daily OHLCV)", type=["csv"])
if not uploaded_file:
    st.info("Upload a daily CSV (Date, Open, High, Low, Close, Volume).")
    st.stop()

df = pd.read_csv(uploaded_file)
# fix BOM / whitespace
df.columns = [col.encode('utf-8').decode('utf-8-sig') for col in df.columns]
df.columns = [col.strip() for col in df.columns]
df.rename(columns={'ï»¿Date': 'Date'}, inplace=True)
st.write("Columns detected:", df.columns.tolist())

col_map = {
    'Date': ['Date'],
    'Open': ['Open', 'OPEN'],
    'High': ['High', 'HIGH'],
    'Low': ['Low', 'LOW'],
    'Close': ['Close', 'CLOSE', 'ltp', 'LTP', 'close'],
    'Volume': ['Shares Traded', 'VOLUME', 'Volume', 'volume']
}
mapping = map_columns(df, col_map)
missing = [k for k in col_map.keys() if k not in mapping]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

df_std = pd.DataFrame()
for k, v in mapping.items():
    df_std[k] = df[v]

df_std['Date'] = pd.to_datetime(df_std['Date'])
df_std.sort_values('Date', inplace=True)

# numeric conversion
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df_std.columns:
        df_std[col] = pd.to_numeric(df_std[col].astype(str).str.replace(',', ''), errors='coerce')

# compute indicators
df_std = compute_indicators(df_std)
st.success("Indicators computed ✅")

# ------------ parameter grid (expanded) ------------
atr_sl_choices = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
atr_tp_choices = [1.5, 2.0, 2.5, 3.0, 3.5]
ema_choices = [50, 100, 150, 200, 250]
min_conf_choices = [1, 2, 3, 4]
trade_modes = ["Both", "Long Only", "Short Only"]
param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

# optional: limit combination count for very long runs (uncomment to use)
# max_combinations = 800
# param_grid = param_grid[:max_combinations]

st.write(f"Parameter combinations to try: {len(param_grid)} (this may be slow for large CSVs)")

# ------------ walk-forward parameters ------------
n = len(df_std)
if n < 200:
    st.warning("Dataset small (<200 rows). Walk-forward may be noisy but will run anyway.")

train_frac = 0.7
train_size = int(train_frac * n)
test_size = max(int(0.3 * n), 10)  # ensure at least some test points
step = test_size  # slide forward by one test window
if train_size + test_size > n:
    st.error("Not enough data for the chosen train/test split. Please provide more historical rows.")
    st.stop()

# ------------ walk-forward loop ------------
wf_results = []
overall_test_trades = []
progress = st.progress(0)
num_windows = max(1, (n - train_size) // step)
window_count = 0

for start in range(0, n - train_size - test_size + 1, step):
    train_start = start
    train_end = start + train_size  # exclusive
    test_start = train_end
    test_end = train_end + test_size

    train_df = df_std.iloc[train_start:train_end].reset_index(drop=True)
    test_df = df_std.iloc[test_start:test_end].reset_index(drop=True)

    # find best params on train using profit factor
    best_pf = None
    best_param = None
    best_train_trades = None
    for params in param_grid:
        trades_df_train, stats_train = generate_trades(train_df, params)
        pf = stats_train['profit_factor']
        # require atleast 3 trades on train to consider (tweakable)
        if (best_pf is None) or (pf > best_pf):
            # prefer combos that produce at least some trades
            best_pf = pf
            best_param = params
            best_train_trades = (trades_df_train, stats_train)

    # test the best_param on test_df
    test_trades_df, test_stats = generate_trades(test_df, best_param)
    wf_results.append({
        'window': window_count,
        'train_range': (train_df['Date'].iloc[0], train_df['Date'].iloc[-1]),
        'test_range': (test_df['Date'].iloc[0], test_df['Date'].iloc[-1]),
        'best_param': best_param,
        'train_stats': best_train_trades[1] if best_train_trades else None,
        'test_stats': test_stats,
        'test_trades': test_trades_df
    })
    overall_test_trades.append(test_trades_df)
    window_count += 1
    progress.progress(min(1.0, window_count / (num_windows if num_windows>0 else 1)))

progress.empty()
st.success("Walk-forward completed ✅")

# ------------ aggregate walk-forward test stats ------------
agg_profit = sum([w['test_stats']['total_profit'] for w in wf_results if w['test_stats']])
agg_trades = sum([w['test_stats']['total_trades'] for w in wf_results if w['test_stats']])
agg_win_sum = sum([w['test_stats']['win_rate'] * w['test_stats']['total_trades']/100 for w in wf_results if w['test_stats'] and w['test_stats']['total_trades']>0])
agg_win_rate = (agg_win_sum / agg_trades * 100) if agg_trades>0 else 0.0

st.header("📈 Walk-Forward Summary (Aggregate Test Results)")
st.write(f"Windows evaluated: {len(wf_results)}")
st.write(f"Aggregate Test Profit (points): {agg_profit:.2f}")
st.write(f"Aggregate Test Trades: {agg_trades}")
st.write(f"Aggregate Test Win Rate (weighted): {agg_win_rate:.2f}%")

# show last few window best params
st.subheader("Recent windows — best parameters")
recent = wf_results[-5:] if len(wf_results) >= 1 else wf_results
show_table = []
for w in recent:
    bp = w['best_param']
    show_table.append({
        'window': w['window'],
        'train_from': w['train_range'][0].date(),
        'train_to': w['train_range'][1].date(),
        'test_from': w['test_range'][0].date(),
        'test_to': w['test_range'][1].date(),
        'atr_sl': bp[0], 'atr_tp': bp[1], 'ema': bp[2], 'min_conf': bp[3], 'mode': bp[4],
        'test_profit': w['test_stats']['total_profit'], 'test_trades': w['test_stats']['total_trades'],
        'test_pf': w['test_stats']['profit_factor']
    })
st.dataframe(pd.DataFrame(show_table))

# ------------ pick most recent best params for live recommendation ------------
most_recent_best = wf_results[-1]['best_param']
best_stats_live = {
    'atr_sl': most_recent_best[0],
    'atr_tp': most_recent_best[1],
    'ema_trend_period': most_recent_best[2],
    'min_conf': most_recent_best[3],
    'trade_mode': most_recent_best[4]
}
st.success(f"Using most-recent window's best params for live recommendation: {best_stats_live}")

# ------------ create overall best trades display (concatenate last test trades) ------------
combined_test_trades = pd.concat([t for t in overall_test_trades if not t.empty], ignore_index=True) if overall_test_trades else pd.DataFrame()
if not combined_test_trades.empty:
    st.subheader("📜 Combined Test Trades (all windows)")
    combined_test_trades['Cum_PnL'] = combined_test_trades['P/L'].cumsum()
    st.dataframe(combined_test_trades)
    fig, ax = plt.subplots()
    ax.plot(combined_test_trades['Exit Date'], combined_test_trades['Cum_PnL'], marker='o')
    ax.set_title("Walk-Forward Equity Curve (combined test trades)")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ------------ LIVE recommendation using most recent train window best params ------------
df_std['EMA_TREND'] = df_std['Close'].ewm(span=best_stats_live['ema_trend_period'], adjust=False).mean()
last = df_std.iloc[-1]
live_long, live_short = [], []

if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
if 30 < last['RSI'] < 70: live_long.append("RSI Healthy")
if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
if last['Vol_Surge']: live_long.append("Volume Surge")

if last['SMA20'] < last['SMA50']: live_short.append("SMA Bearish")
if 30 < last['RSI'] < 70: live_short.append("RSI Healthy")
if last['MACD'] < last['Signal']: live_short.append("MACD Bearish")
if last['Close'] < last['BB_Mid']: live_short.append("BB Breakdown")
if last['Vol_Surge']: live_short.append("Volume Surge")

st.header("📢 Live Recommendation (Most Recent Best Parameters)")
if last['Close'] > last['EMA_TREND'] and len(live_long) >= best_stats_live['min_conf'] and (best_stats_live['trade_mode'] in ["Both","Long Only"]):
    st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - best_stats_live['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] + best_stats_live['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_long)}")
elif last['Close'] < last['EMA_TREND'] and len(live_short) >= best_stats_live['min_conf'] and (best_stats_live['trade_mode'] in ["Both","Short Only"]):
    st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + best_stats_live['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] - best_stats_live['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_short)}")
else:
    st.error("❌ No strong trade setup currently (most recent best params).")

# ------------ CONTINUATION CHECK (yesterday -> today) ------------
st.subheader("🔁 Continuation Check (yesterday -> today)")
if len(df_std) >= 2:
    yesterday = df_std.iloc[-2]
    today = df_std.iloc[-1]

    # reconstruct yesterday's signal using same most recent params
    y_long, y_short = [], []
    if yesterday['SMA20'] > yesterday['SMA50']: y_long.append("SMA Bullish")
    if 30 < yesterday['RSI'] < 70: y_long.append("RSI Healthy")
    if yesterday['MACD'] > yesterday['Signal']: y_long.append("MACD Bullish")
    if yesterday['Close'] > yesterday['BB_Mid']: y_long.append("BB Breakout")
    if yesterday['Vol_Surge']: y_long.append("Volume Surge")

    if yesterday['SMA20'] < yesterday['SMA50']: y_short.append("SMA Bearish")
    if 30 < yesterday['RSI'] < 70: y_short.append("RSI Healthy")
    if yesterday['MACD'] < yesterday['Signal']: y_short.append("MACD Bearish")
    if yesterday['Close'] < yesterday['BB_Mid']: y_short.append("BB Breakdown")
    if yesterday['Vol_Surge']: y_short.append("Volume Surge")

    still_active = False
    entry_price = None
    if yesterday['Close'] > yesterday['EMA_TREND'] and len(y_long) >= best_stats_live['min_conf'] and (best_stats_live['trade_mode'] in ["Both","Long Only"]):
        y_sl = yesterday['Close'] - best_stats_live['atr_sl'] * yesterday['ATR']
        y_tp = yesterday['Close'] + best_stats_live['atr_tp'] * yesterday['ATR']
        entry_price = yesterday['Close']
        # if today's close hasn't breached SL or TP
        if (today['Close'] > y_sl) and (today['Close'] < y_tp):
            still_active = True
            st.info(f"✅ Yesterday's LONG at {entry_price:.2f} is STILL ACTIVE | SL: {y_sl:.2f} | TP: {y_tp:.2f} | Reasons: {', '.join(y_long)}")
    elif yesterday['Close'] < yesterday['EMA_TREND'] and len(y_short) >= best_stats_live['min_conf'] and (best_stats_live['trade_mode'] in ["Both","Short Only"]):
        y_sl = yesterday['Close'] + best_stats_live['atr_sl'] * yesterday['ATR']
        y_tp = yesterday['Close'] - best_stats_live['atr_tp'] * yesterday['ATR']
        entry_price = yesterday['Close']
        if (today['Close'] < y_sl) and (today['Close'] > y_tp):
            still_active = True
            st.info(f"✅ Yesterday's SHORT at {entry_price:.2f} is STILL ACTIVE | SL: {y_sl:.2f} | TP: {y_tp:.2f} | Reasons: {', '.join(y_short)}")
    if not still_active:
        st.warning("⛔ Yesterday's recommendation is no longer valid (hit SL/TP or conditions invalidated).")
else:
    st.info("Not enough rows to check continuation (need at least 2 rows).")
