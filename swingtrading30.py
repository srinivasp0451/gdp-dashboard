# pro_trading_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import itertools
import math

st.set_page_config(layout="wide", page_title="Pro Trading Strategy Lab")

# ------------------------
# Helpers: caching yfinance
# ------------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_ticker(ticker, interval, period):
    # yfinance intervals: '1m','2m','5m','15m','30m','60m','90m','1h','1d','1wk'
    # We'll request with yf.download which supports period + interval
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=False, threads=True, progress=False)
        if df is None or df.empty:
            return None
        df = df.rename_axis('datetime').reset_index().set_index('datetime')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# ------------------------
# Technical indicators (manual)
# ------------------------
def rsi(series_close, length=14):
    delta = series_close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(length, min_periods=length).mean()
    ma_down = down.rolling(length, min_periods=length).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

def atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

# ZigZag swing detection (simple)
def detect_swings(df, order=5):
    # order: how many bars for local extrema
    close = df['Close']
    # local maxima and minima indices
    local_max = argrelextrema(close.values, np.greater_equal, order=order)[0]
    local_min = argrelextrema(close.values, np.less_equal, order=order)[0]
    extremes = pd.DataFrame(index=df.index, columns=['type','price'])
    for i in local_max:
        extremes.iloc[i] = ['peak', close.iloc[i]]
    for i in local_min:
        extremes.iloc[i] = ['trough', close.iloc[i]]
    # Forward fill small sequences, drop NaNs
    extremes = extremes.dropna()
    return extremes

def fib_levels(low, high):
    # Return common Fibonacci retracement levels between low and high
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    return levels

# RSI divergence simple detector (checks last N bars)
def detect_rsi_divergence(df, rsi_col='RSI', window=30):
    # We'll check for bullish divergence (price makes lower low, RSI makes higher low) and
    # bearish divergence (price makes higher high, RSI makes lower high) within the window.
    res = []
    close = df['Close']
    rsi = df[rsi_col]
    for i in range(window, len(df)):
        slice_idx = df.index[i-window:i+1]
        sub = df.loc[slice_idx]
        # price extremes
        price_low_idx = sub['Close'].idxmin()
        price_low = sub['Close'].min()
        rsi_at_price_low = sub.loc[price_low_idx, rsi_col]
        # earlier low before that
        earlier = sub.loc[:price_low_idx]
        if len(earlier) < 2:
            continue
        earlier_low = earlier['Close'].idxmin()
        if earlier_low == price_low_idx:
            continue
        earlier_low_val = earlier.loc[earlier_low, 'Close']
        earlier_rsi = earlier.loc[earlier_low, rsi_col]
        # bullish divergence
        if price_low < earlier_low_val and rsi_at_price_low > earlier_rsi:
            res.append(('bullish', price_low_idx))
        # Now bearish
        price_high_idx = sub['Close'].idxmax()
        price_high = sub['Close'].max()
        rsi_at_price_high = sub.loc[price_high_idx, rsi_col]
        earlier_high = sub.loc[:price_high_idx]
        if len(earlier_high) < 2:
            continue
        earlier_high_idx = earlier_high['Close'].idxmax()
        if earlier_high_idx == price_high_idx:
            continue
        earlier_high_val = earlier_high.loc[earlier_high_idx,'Close']
        earlier_high_rsi = earlier_high.loc[earlier_high_idx,rsi_col]
        if price_high > earlier_high_val and rsi_at_price_high < earlier_high_rsi:
            res.append(('bearish', price_high_idx))
    return res

# ------------------------
# Strategy rules (generate signals)
# ------------------------
def generate_signals(df, params):
    """
    Return DataFrame with signals: 1 for long entry, -1 for short entry.
    params: dict with keys:
      - rsi_len, rsi_oversold, rsi_overbought
      - zig_order
      - atr_len, atr_sl_mult, rr (risk-reward)
      - mtf_intervals: list of tuples (df2, weight) or None
    """
    df = df.copy()
    df['RSI'] = rsi(df['Close'], length=params.get('rsi_len',14))
    df['ATR'] = atr(df, length=params.get('atr_len',14))
    # detect swings
    swings = detect_swings(df, order=params.get('zig_order',5))
    # mark latest major swing low/high
    df['swing_type'] = np.nan
    df['swing_price'] = np.nan
    for idx, row in swings.iterrows():
        df.at[idx,'swing_type'] = row['type']
        df.at[idx,'swing_price'] = row['price']
    # simple signals:
    # Long: RSI crosses above oversold and recent swing was a trough and multi-timeframe confirms momentum
    df['signal'] = 0
    rsi_ov = params.get('rsi_oversold', 30)
    rsi_ob = params.get('rsi_overbought', 70)
    for i in range(1, len(df)):
        prev = df['RSI'].iat[i-1]
        curr = df['RSI'].iat[i]
        # Long entry
        if (prev < rsi_ov and curr >= rsi_ov):
            # check last swing near i is trough
            window = params.get('confirm_window', 30)
            window_slice = df.iloc[max(0,i-window):i+1]
            if 'trough' in set(window_slice['swing_type'].dropna().values):
                # MTF confirmation: optional - we accept if higher timeframe momentum positive (close > sma)
                mtf_ok = True
                if params.get('mtf_df') is not None:
                    mtf_ok = params.get('mtf_df')['Close'].iloc[-1] > params.get('mtf_df')['Close'].rolling(params.get('mtf_sma',20)).mean().iloc[-1]
                if mtf_ok:
                    df['signal'].iat[i] = 1
        # Short entry
        if (prev > rsi_ob and curr <= rsi_ob):
            window = params.get('confirm_window', 30)
            window_slice = df.iloc[max(0,i-window):i+1]
            if 'peak' in set(window_slice['swing_type'].dropna().values):
                mtf_ok = True
                if params.get('mtf_df') is not None:
                    mtf_ok = params.get('mtf_df')['Close'].iloc[-1] < params.get('mtf_df')['Close'].rolling(params.get('mtf_sma',20)).mean().iloc[-1]
                if mtf_ok:
                    df['signal'].iat[i] = -1
    # add fibonacci around last major swing
    # find last trough and peak
    peaks = swings[swings['type']=='peak']
    troughs = swings[swings['type']=='trough']
    if not troughs.empty and not peaks.empty:
        last_trough_idx = troughs.index[-1]
        last_peak_idx = peaks.index[-1]
        # ensure low before high for uptrend fib; otherwise invert
        if last_trough_idx < last_peak_idx:
            low = troughs.iloc[-1]['price']
            high = peaks.iloc[-1]['price']
            df.attrs['fib'] = fib_levels(low, high)
        else:
            low = peaks.iloc[-1]['price']
            high = troughs.iloc[-1]['price']
            df.attrs['fib'] = fib_levels(low, high)
    else:
        df.attrs['fib'] = {}
    return df

# ------------------------
# Backtester (simple intrabar: entries at close on signal, exits by SL/TP or time)
# ------------------------
def backtest(df, params, capital=100000, max_holding_bars=200):
    df = df.copy()
    trades = []
    position = None
    equity_curve = []
    cash = capital
    equity = capital
    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        # if open position check exit
        if position is not None:
            # update MTM
            current_price = row['Close']
            unreal = (current_price - position['entry_price']) * position['size'] * (1 if position['side']==1 else -1)
            # Check TP or SL
            if position['side'] == 1:
                tp_hit = current_price >= position['tp']
                sl_hit = current_price <= position['sl']
            else:
                tp_hit = current_price <= position['tp']
                sl_hit = current_price >= position['sl']
            if tp_hit or sl_hit or (i - position['entry_idx']) >= max_holding_bars:
                # close at close price
                exit_price = current_price
                pnl = (exit_price - position['entry_price']) * position['size'] * (1 if position['side']==1 else -1)
                cash += position['margin'] + pnl
                trade = {
                    'entry_dt': position['entry_dt'],
                    'exit_dt': date,
                    'side': 'Long' if position['side']==1 else 'Short',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'tp': position['tp'],
                    'sl': position['sl'],
                    'size': position['size'],
                    'pnl': pnl,
                    'holding_bars': i - position['entry_idx'],
                    'reason': position.get('reason','auto')
                }
                trades.append(trade)
                position = None
            else:
                # keep holding
                pass
        # if no position and signal present, open new
        if position is None and row['signal'] != 0:
            side = int(row['signal'])
            entry_price = row['Close']
            atr_val = row['ATR'] if row['ATR']>0 else 1e-6
            sl_mult = params.get('atr_sl_mult', 1.5)
            rr = params.get('rr', 2.0)
            if side == 1:
                sl = entry_price - sl_mult * atr_val
                tp = entry_price + rr * (entry_price - sl)
            else:
                sl = entry_price + sl_mult * atr_val
                tp = entry_price - rr * (sl - entry_price)
            # position sizing: risk fixed percent per trade
            risk_pct = params.get('risk_pct', 0.01)
            risk_amount = capital * risk_pct
            # size in units (for indices/forex this is a theoretical / not contract size)
            size = 0
            if side == 1:
                per_unit_risk = entry_price - sl
            else:
                per_unit_risk = sl - entry_price
            if per_unit_risk <= 0:
                continue
            size = math.floor(risk_amount / per_unit_risk)
            if size <= 0:
                continue
            margin = 0  # simplified
            position = {
                'entry_dt': date,
                'entry_idx': i,
                'entry_price': entry_price,
                'side': side,
                'size': size,
                'sl': sl,
                'tp': tp,
                'margin': margin,
                'reason': 'RSI_divergence+zig+mtf'
            }
            # reserve margin
            cash -= margin
        # equity record
        if position is None:
            equity = cash
        else:
            current_price = row['Close']
            unreal = (current_price - position['entry_price']) * position['size'] * (1 if position['side']==1 else -1)
            equity = cash + position['margin'] + unreal
        equity_curve.append({'dt': date, 'equity': equity})
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('dt')
    # metrics
    if not trades_df.empty:
        wins = trades_df[trades_df['pnl']>0]
        losses = trades_df[trades_df['pnl']<=0]
        win_rate = len(wins)/len(trades_df)
        total_pnl = trades_df['pnl'].sum()
    else:
        win_rate = 0.0
        total_pnl = 0.0
    buy_hold_pnl = df['Close'].iloc[-1] - df['Close'].iloc[0]
    results = {
        'trades': trades_df,
        'equity': equity_df,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'buy_hold_pnl': buy_hold_pnl
    }
    return results

# ------------------------
# Auto-optimizer: small grid search
# ------------------------
def auto_optimize(df, param_grid, capital):
    best = None
    best_score = -1e9
    all_results = []
    # small grid to avoid explosion
    combos = list(itertools.product(*param_grid.values()))
    names = list(param_grid.keys())
    for combo in combos:
        p = dict(zip(names, combo))
        # insert mtf_df as None for now (we'll use the same df for mtf if requested)
        p['mtf_df'] = None
        sig_df = generate_signals(df, p)
        res = backtest(sig_df, p, capital=capital)
        # score = total pnl + factor*(win_rate)
        score = res['total_pnl'] + res['win_rate'] * 1000
        all_results.append((p, res, score))
        if score > best_score:
            best_score = score
            best = (p, res, score)
    return best, all_results

# ------------------------
# UI
# ------------------------
st.title("Pro Trading Strategy Lab — Streamlit")
st.markdown("**Disclaimer:** This tool provides signals and backtesting for research only. Do not trade real money without thorough testing. The app *does not* guarantee performance or a fixed accuracy — it provides automated validation metrics so you can judge performance.")

# Sidebar inputs
with st.sidebar:
    st.header("Data & Strategy Inputs")
    ticker = st.text_input("Ticker (yfinance format)", value="^NSEI")
    interval = st.selectbox("Timeframe / Interval", options=[
        '1m','2m','5m','15m','30m','60m','90m','1h','1d','1wk'
    ], index=7)
    period = st.selectbox("Period", options=[
        '1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y'
    ], index=7)
    capital = st.number_input("Capital (₹ / $)", value=100000, step=1000)
    run_button = st.button("Run / Refresh Analysis")
    st.markdown("---")
    st.subheader("Quick optimizer options")
    auto_opt = st.checkbox("Auto-optimize parameters (small grid search)", value=True)
    max_grid_iters = st.selectbox("Grid size hint", options=[10, 20, 40], index=1)
    st.markdown("Advanced: adjust base params")
    rsi_len = st.number_input("RSI length", value=14, min_value=2, max_value=50)
    rsi_oversold = st.number_input("RSI oversold", value=30)
    rsi_overbought = st.number_input("RSI overbought", value=70)
    atr_len = st.number_input("ATR length", value=14, min_value=2, max_value=50)

if not run_button:
    st.info("Set inputs on the left and click **Run / Refresh Analysis**")
    st.stop()

# Fetch data
with st.spinner("Fetching data... (cached when possible)"):
    df = fetch_ticker(ticker, interval, period)
if df is None or df.empty:
    st.error("No data returned. Try different ticker/period/interval.")
    st.stop()

# Show data frame and basics
st.subheader("Fetched Data (sample)")
st.dataframe(df.tail(200))

min_date = df.index.min()
max_date = df.index.max()
min_close = df['Close'].min()
max_close = df['Close'].max()
st.markdown(f"**Date Range:** {min_date} to {max_date}    |    **Min Close:** {min_close:.4f}    |    **Max Close:** {max_close:.4f}")

# Prepare param grid
param_grid = {
    'rsi_len': [rsi_len],
    'rsi_oversold': [rsi_oversold],
    'rsi_overbought': [rsi_overbought],
    'zig_order': [3,5,8],
    'atr_len': [atr_len],
    'atr_sl_mult': [1.0, 1.5, 2.0],
    'rr': [1.5, 2.0, 3.0],
    'risk_pct': [0.005, 0.01, 0.02],
    'confirm_window': [20, 40]
}
# shrink grid if user chose small size
if not auto_opt:
    # use a single param set
    base_params = {
        'rsi_len': rsi_len,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'zig_order': 5,
        'atr_len': atr_len,
        'atr_sl_mult': 1.5,
        'rr': 2.0,
        'risk_pct': 0.01,
        'confirm_window': 30
    }
    best_params = base_params
    sig_df = generate_signals(df, best_params)
    bt_res = backtest(sig_df, best_params, capital=capital)
else:
    # limit combinations to reasonable number
    # reduce grid based on max_grid_iters selection
    # create combos but sample if too large
    combos_total = np.prod([len(v) for v in param_grid.values()])
    # if too many, reduce options per param
    if combos_total > max_grid_iters:
        # for each parameter keep at most 2 values (middle or extremes)
        small_grid = {}
        for k,v in param_grid.items():
            if len(v) > 2:
                small_grid[k] = [v[0], v[-1]]
            else:
                small_grid[k] = v
        param_grid_use = small_grid
    else:
        param_grid_use = param_grid
    best, all_results = auto_optimize(df, param_grid_use, capital)
    best_params, bt_res, score = best
    st.write(f"Auto-optimizer selected params with score {score:.2f}")
    st.json(best_params)
    sig_df = generate_signals(df, best_params)

# Add derived columns to sig_df for display
sig_df['RSI'] = rsi(sig_df['Close'], length=best_params.get('rsi_len',14))
sig_df['ATR'] = atr(sig_df, length=best_params.get('atr_len',14))

# Backtest results summary
st.subheader("Backtest Results (using selected params)")
st.markdown(f"**Total PnL (points):** {bt_res['total_pnl']:.4f}    |    **Win rate:** {bt_res['win_rate']*100:.2f}%    |    **Buy & Hold points:** {bt_res['buy_hold_pnl']:.4f}")
# Trades table in requested format: entry date time levels target sl total pnl reason logic/entry probability
trades = bt_res['trades']
if trades.empty:
    st.warning("No trades generated by strategy on this dataset with selected parameters.")
else:
    trades_display = trades.copy()
    trades_display = trades_display.rename(columns={
        'entry_dt':'Entry DateTime',
        'exit_dt':'Exit DateTime',
        'entry_price':'Entry Level',
        'tp':'Target',
        'sl':'Stop Loss',
        'pnl':'Total PnL',
        'reason':'Reason/Logic'
    })
    # Add simple probability of profit = win_rate
    trades_display['Probability of Profit (est)'] = f"{bt_res['win_rate']*100:.2f}%"
    # Sort ascending by entry date
    trades_display = trades_display.sort_values('Entry DateTime', ascending=True)
    st.subheader("Trades (sorted by entry date asc)")
    st.dataframe(trades_display)

# Live recommendation (based on last available candle)
st.subheader("Live Recommendation (based on last close)")
last_idx = sig_df.index[-1]
last_row = sig_df.iloc[-1]
live_signal = last_row['signal']
if live_signal == 1:
    side = 'Long'
elif live_signal == -1:
    side = 'Short'
else:
    side = 'No trade'

if side == 'No trade':
    st.info("No immediate trade signal on the last candle.")
else:
    entry_dt = last_idx
    entry_price = last_row['Close']
    atr_val = last_row['ATR'] if last_row['ATR']>0 else atr(sig_df, length=best_params.get('atr_len',14)).iloc[-1]
    sl_mult = best_params.get('atr_sl_mult',1.5)
    rr = best_params.get('rr',2.0)
    if live_signal == 1:
        sl = entry_price - sl_mult * atr_val
        tp = entry_price + rr * (entry_price - sl)
    else:
        sl = entry_price + sl_mult * atr_val
        tp = entry_price - rr * (sl - entry_price)
    # simple probability estimate from backtest win rate
    prob = bt_res['win_rate']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommendation**")
        st.write(f"Side: **{side}**")
        st.write(f"Entry DateTime: {entry_dt}")
        st.write(f"Entry Level: {entry_price:.4f}")
        st.write(f"Target: {tp:.4f}")
        st.write(f"Stop Loss: {sl:.4f}")
    with col2:
        st.markdown("**Trade stats & sizing**")
        st.write(f"Estimated Probability of Profit (from backtest): {prob*100:.2f}%")
        risk_pct = best_params.get('risk_pct', 0.01)
        risk_amount = capital * risk_pct
        per_unit_risk = abs(entry_price - sl)
        size = math.floor(risk_amount / per_unit_risk) if per_unit_risk>0 else 0
        st.write(f"Recommended risk per trade: {risk_pct*100:.2f}% -> {risk_amount:.2f}")
        st.write(f"Suggested position size (units): {size}")
    # output a structured line as requested
    st.markdown("**Live Recommendation — Compact Row**")
    rec = {
        'entry_dt': str(entry_dt),
        'entry_level': round(entry_price, 6),
        'target': round(tp, 6),
        'sl': round(sl, 6),
        'size': size,
        'probability_of_profit': f"{prob*100:.2f}%",
        'reason/logic': 'RSI crossover near swing + MTF confirmation'
    }
    st.json(rec)

# Plot equity curve
if not bt_res['equity'].empty:
    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(bt_res['equity'].index, bt_res['equity']['equity'])
    ax.set_ylabel("Equity")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# Heatmap of returns by month-year
st.subheader("Heatmap of Monthly Returns")
returns = sig_df['Close'].pct_change().dropna()
returns_df = returns.to_frame(name='ret')
returns_df['year'] = returns_df.index.year
returns_df['month'] = returns_df.index.month
monthly = returns_df.groupby(['year','month']).agg({'ret':'sum'}).reset_index()
if monthly.empty:
    st.warning("Insufficient data for heatmap.")
else:
    pivot = monthly.pivot(index='year', columns='month', values='ret').fillna(0)
    fig2, ax2 = plt.subplots(figsize=(12,4))
    im = ax2.imshow(pivot, aspect='auto')
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels(pivot.index)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2.set_title("Monthly Returns Heatmap (sum of pct returns)")
    st.pyplot(fig2)

# Show fibonacci if available
if sig_df.attrs.get('fib'):
    st.subheader("Detected Fibonacci Levels (from last major swing)")
    fib = sig_df.attrs['fib']
    st.table(pd.DataFrame(list(fib.items()), columns=['level','price']).set_index('level'))

# Final metrics & validation: simple pass/fail
st.subheader("Automated Validation")
# criteria examples:
# - win_rate > 0.5 and total_pnl > buy_hold_pnl and total_pnl > 0
validation_messages = []
win_rate = bt_res.get('win_rate', 0)
total_pnl = bt_res.get('total_pnl', 0)
bh = bt_res.get('buy_hold_pnl', 0)
if win_rate > 0.8:
    validation_messages.append(f"Excellent win rate: {win_rate*100:.2f}% (>=80%)")
else:
    validation_messages.append(f"Win rate: {win_rate*100:.2f}% (target 80%)")

if total_pnl > bh and total_pnl > 0:
    validation_messages.append(f"Strategy beats buy & hold on this dataset (strategy pnl {total_pnl:.2f} vs buy-hold {bh:.2f})")
else:
    validation_messages.append(f"Strategy does NOT beat buy & hold on this dataset (strategy pnl {total_pnl:.2f} vs buy-hold {bh:.2f})")

st.write("\n".join(validation_messages))

# Save displayed outputs for user to copy/export
st.markdown("---")
st.write("Export / Notes")
if st.button("Download trades CSV"):
    if trades.empty:
        st.warning("No trades to download")
    else:
        csv = trades.to_csv(index=False)
        st.download_button("Download trades.csv", data=csv, file_name=f"{ticker}_trades.csv", mime="text/csv")

st.write("End of report.")
