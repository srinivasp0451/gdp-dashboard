# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- Indicator & helpers ---
def compute_indicators(df, atr_period=14, ema_fast=8, rsi_period=7):
    df = df.copy()
    df['hl'] = df['High'] - df['Low']
    df['hc'] = (df['High'] - df['Close'].shift(1)).abs()
    df['lc'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['tr'] = df[['hl','hc','lc']].max(axis=1)
    df['ATR'] = df['tr'].rolling(atr_period, min_periods=1).mean()
    df['EMA_fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()

    # RSI (fast)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['RSI_fast'] = 100 - (100 / (1 + rs))

    # Momentum: percent change over small window
    df['mom3'] = df['Close'].pct_change(3)

    # Price channels for breakout baseline
    df['prev_high_n'] = df['High'].rolling(20).max().shift(1)  # breakout level reference
    df['prev_low_n'] = df['Low'].rolling(20).min().shift(1)

    # Volume spike detection (if volume available)
    if 'Volume' in df.columns:
        df['vol_ma'] = df['Volume'].rolling(20).mean()
        df['vol_spike'] = df['Volume'] > (1.5 * df['vol_ma'])
    else:
        df['vol_spike'] = False

    return df.drop(columns=['hl','hc','lc','tr'])

def generate_signals(df, atr_multiplier=0.9, rsi_threshold=55, min_mom=0.002):
    """
    ABM entry logic:
      - Long entry when:
         close > prev_high_n + ATR * atr_multiplier
         AND RSI_fast > rsi_threshold
         AND mom3 > min_mom
         (volume spike strengthens signal)
      - Short symmetrical
    Exit:
      - Target = entry + k*ATR
      - SL = entry - s*ATR (for long)
      - Exit also on momentum reversal: close crosses EMA_fast opposite side
    """
    df = df.copy()
    df['signal'] = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if np.isnan(row['prev_high_n']) or np.isnan(row['ATR']):
            continue
        # Long
        if (row['Close'] > row['prev_high_n'] + atr_multiplier * row['ATR']
            and row['RSI_fast'] >= rsi_threshold
            and row['mom3'] >= min_mom):
            df.at[df.index[i], 'signal'] = 1
        # Short
        elif (row['Close'] < row['prev_low_n'] - atr_multiplier * row['ATR']
            and row['RSI_fast'] <= (100 - rsi_threshold)
            and row['mom3'] <= -min_mom):
            df.at[df.index[i], 'signal'] = -1
    return df

# --- Backtester using last-close entries ---
def backtest(df, atr_multiplier=0.9, target_atr=2.0, sl_atr=1.0, max_holding_days=14, capital=100000):
    """
    Backtest iterating row-by-row. When signal appears at index i, we assume entry at close_i (last-candle-close).
    Target and SL are set from entry using ATR at that bar.
    We then step forward until either target or SL or holding time reached, or opposite signal triggers exit at close.
    Returns trades dataframe and summary stats.
    """
    df = df.copy().reset_index(drop=True)
    trades = []
    position = None
    for i in range(len(df)):
        row = df.loc[i]
        # open new position if none
        if position is None and row['signal'] != 0:
            entry_idx = i
            entry_price = row['Close']
            atr = row['ATR'] if not np.isnan(row['ATR']) and row['ATR']>0 else 1e-6
            if row['signal'] == 1:
                target = entry_price + target_atr * atr
                sl = entry_price - sl_atr * atr
                side = 'LONG'
            else:
                target = entry_price - target_atr * atr
                sl = entry_price + sl_atr * atr
                side = 'SHORT'
            reason = f"Signal {'breakout' if abs(row['mom3'])>0 else 'momentum'} at close; RSI={row['RSI_fast']:.1f}"
            position = dict(entry_idx=entry_idx, entry_date=row.name if isinstance(row.name, (pd.Timestamp, str)) else df.at[i, 'Date'] if 'Date' in df.columns else i,
                            entry_price=entry_price, target=target, sl=sl, side=side, reason_entry=reason, entry_time=row.name)
            # continue to next bar to simulate intrabar? We stick to close-based
            continue

        # manage open position
        if position is not None:
            holding = i - position['entry_idx']
            price = row['Close']
            exited = False
            exit_reason = None
            exit_price = price
            # check target/SL hit using CLOSE-based check (conservative)
            if position['side'] == 'LONG':
                if price >= position['target']:
                    exit_price = position['target']  # assume target met at close
                    exit_reason = 'Target Hit'
                    exited = True
                elif price <= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL Hit'
                    exited = True
                # momentum reversal: close below EMA_fast
                elif price < row['EMA_fast']:
                    exit_price = price
                    exit_reason = 'Momentum Reversal (EMA)'
                    exited = True
            else:  # SHORT
                if price <= position['target']:
                    exit_price = position['target']
                    exit_reason = 'Target Hit'
                    exited = True
                elif price >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'SL Hit'
                    exited = True
                elif price > row['EMA_fast']:
                    exit_price = price
                    exit_reason = 'Momentum Reversal (EMA)'
                    exited = True

            # forced exit on max holding
            if not exited and holding >= max_holding_days:
                exit_price = price
                exit_reason = 'Max Holding Days'
                exited = True

            # exit on opposite signal (close-based)
            if not exited and row['signal'] != 0 and ((row['signal']==1 and position['side']=='SHORT') or (row['signal']==-1 and position['side']=='LONG')):
                exit_price = price
                exit_reason = 'Opposite Signal'
                exited = True

            if exited:
                pnl = (exit_price - position['entry_price']) if position['side']=='LONG' else (position['entry_price'] - exit_price)
                pnl_pct = (pnl / position['entry_price']) * 100
                trades.append({
                    'entry_idx': position['entry_idx'],
                    'entry_datetime': df.at[position['entry_idx'],'Date'] if 'Date' in df.columns else position['entry_idx'],
                    'entry_price': position['entry_price'],
                    'exit_idx': i,
                    'exit_datetime': df.at[i,'Date'] if 'Date' in df.columns else i,
                    'exit_price': exit_price,
                    'side': position['side'],
                    'target': position['target'],
                    'sl': position['sl'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_days': holding,
                    'entry_reason': position['reason_entry'],
                    'exit_reason': exit_reason
                })
                position = None

    trades_df = pd.DataFrame(trades)
    # summary
    if trades_df.empty:
        summary = {
            'total_trades':0, 'positive_trades':0, 'negative_trades':0,
            'accuracy':0.0, 'total_pnl':0.0, 'total_pnl_pct':0.0
        }
    else:
        positive = trades_df[trades_df['pnl']>0]
        negative = trades_df[trades_df['pnl']<=0]
        total_pnl = trades_df['pnl'].sum()
        # buy & hold comparison: use first close -> last close
        buy_hold_points = df['Close'].iloc[-1] - df['Close'].iloc[0]
        summary = {
            'total_trades':len(trades_df),
            'positive_trades':len(positive),
            'negative_trades':len(negative),
            'accuracy': len(positive)/len(trades_df) if len(trades_df)>0 else 0.0,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / capital) * 100,
            'total_points_strategy': trades_df['pnl'].sum(),
            'total_points_buy_hold': buy_hold_points
        }
    return trades_df, summary

# --- Data fetch / load ---
@st.cache_data(ttl=300)
def download_data_yf(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        return data
    data = data[['Open','High','Low','Close','Volume']].reset_index().rename(columns={'index':'Date'})
    return data

# --- UI ---
st.set_page_config(layout='wide', page_title='ABM Swing - Streamlit Algo')
st.title("ABM (Adaptive BreakMomentum) — Swing Trading Indicator & Backtester")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Data input")
    ticker = st.text_input("Ticker (yfinance, e.g. AAPL, ^NSEI)", value="^NSEI")
    use_csv = st.checkbox("Upload CSV instead of yfinance", value=False)
    uploaded_file = None
    if use_csv:
        uploaded_file = st.file_uploader("Upload OHLCV CSV", type=['csv'])
    interval = st.selectbox("Interval", options=["1d","60m","15m"], index=0)
    period = st.selectbox("Period (yfinance) or ignore if CSV", options=["6mo","1y","2y","5y"], index=1)
    start_date = st.date_input("Start date (optional)", value=None)
    end_date = st.date_input("End date (optional)", value=None)

    st.subheader("Strategy params")
    atr_p = st.number_input("ATR period", value=14, min_value=1)
    atr_multiplier = st.number_input("Break ATR multiplier", value=0.9, step=0.1)
    target_atr = st.number_input("Target (x ATR)", value=2.0, step=0.1)
    sl_atr = st.number_input("SL (x ATR)", value=1.0, step=0.1)
    rsi_threshold = st.number_input("RSI threshold (fast)", value=55, min_value=40, max_value=80)
    max_hold = st.number_input("Max holding bars (days for daily)", value=10, min_value=1)

    capital = st.number_input("Capital for %calc", value=100000)
    run_btn = st.button("Run Backtest")
    live_btn = st.button("Get Live Recommendation (last close)")

with col2:
    st.subheader("Plots & Results")
    chart_area = st.empty()
    trades_area = st.empty()
    summary_area = st.empty()

# Load data
df = pd.DataFrame()
if use_csv and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # try to standardize column names
    lower_cols = {c.lower():c for c in df.columns}
    for name in ['date','open','high','low','close','volume']:
        if name not in lower_cols:
            continue
    # ensure Date column is datetime
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date':'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date','Open','High','Low','Close','Volume']]
else:
    with st.spinner("Downloading..."):
        df = download_data_yf(ticker, period=period, interval=interval)
        if not df.empty and start_date is not None:
            try:
                if isinstance(start_date, datetime):
                    s = start_date
                else:
                    s = datetime.combine(start_date, datetime.min.time())
                df = df[df['Date'] >= s]
            except:
                pass
        if not df.empty and end_date is not None:
            try:
                if isinstance(end_date, datetime):
                    e = end_date
                else:
                    e = datetime.combine(end_date, datetime.max.time())
                df = df[df['Date'] <= e]
            except:
                pass

if df.empty:
    st.warning("No data loaded. Check ticker or upload CSV.")
    st.stop()

# Compute indicators and signals
df_ind = compute_indicators(df, atr_period=int(atr_p), ema_fast=8, rsi_period=7)
df_sig = generate_signals(df_ind, atr_multiplier=float(atr_multiplier), rsi_threshold=float(rsi_threshold), min_mom=0.002)

# Run backtest if requested
if run_btn:
    trades_df, summary = backtest(df_sig, atr_multiplier=atr_multiplier, target_atr=target_atr, sl_atr=sl_atr, max_holding_days=int(max_hold), capital=capital)
    # display chart
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df_sig['Date'], df_sig['Close'], label='Close')
    ax.plot(df_sig['Date'], df_sig['EMA_fast'], label='EMA_fast')
    ax.set_title(f'Price & EMA - {ticker}')
    ax.legend()
    chart_area.pyplot(fig)

    if trades_df.empty:
        trades_area.info("No trades found with current parameters.")
    else:
        # Format output as requested per trade
        trades_show = trades_df.copy()
        trades_show['entry_date_time'] = trades_show['entry_datetime']
        trades_show['levels'] = trades_show['entry_price']
        trades_show['target'] = trades_show['target']
        trades_show['sl'] = trades_show['sl']
        trades_show['total_pnl'] = trades_show['pnl']
        trades_show['total_pnl_percentage'] = trades_show['pnl_pct']
        trades_show['reason_of_entry'] = trades_show['entry_reason']
        trades_show['reason_of_exit'] = trades_show['exit_reason']
        trades_show = trades_show[['entry_date_time','levels','target','sl','total_pnl','total_pnl_percentage','reason_of_entry','reason_of_exit','holding_days']]
        trades_area.dataframe(trades_show)

        # Summary
        s = summary
        summary_md = f"""
        **Backtest Summary**
        - Total trades: {s['total_trades']}
        - Positive trades: {s['positive_trades']}
        - Negative trades: {s['negative_trades']}
        - Accuracy: {s['accuracy']*100:.2f}%
        - Total PnL (points): {s['total_points_strategy']:.2f}
        - Total PnL (% of capital): {s['total_pnl_pct']:.2f}%
        - Buy & Hold points (first->last): {s['total_points_buy_hold']:.2f}
        """
        summary_area.markdown(summary_md)

# Live recommendation based on last candle close
if live_btn:
    last = df_sig.iloc[-1]
    if last['signal'] == 0:
        st.info("No signal on last close.")
    else:
        side = 'LONG' if last['signal']==1 else 'SHORT'
        entry = last['Close']
        atr = last['ATR'] if last['ATR']>0 else 1e-6
        target = entry + target_atr * atr if side=='LONG' else entry - target_atr * atr
        sl = entry - sl_atr * atr if side=='LONG' else entry + sl_atr * atr
        # simple probability estimator: based on historical success of similar signals (quick heuristic)
        # count last 100 signals and compute proportion positive in similar context
        hist = df_sig.copy()
        # For speed: reuse backtester on past data up to penultimate bar
        hist_trades, hist_summary = backtest(hist.iloc[:-1], atr_multiplier=atr_multiplier, target_atr=target_atr, sl_atr=sl_atr, max_holding_days=int(max_hold), capital=capital)
        prob = hist_summary['accuracy'] if hist_summary['total_trades']>0 else 0.0

        live = {
            'entry_date_time': last['Date'],
            'levels': entry,
            'target': target,
            'sl': sl,
            'total_pnl': None,
            'total_pnl_percentage': None,
            'reason_of_entry': f"Signal on close: side={side}; RSI={last['RSI_fast']:.1f}; mom3={last['mom3']:.4f}",
            'probability_of_profit': f"{prob*100:.1f}%"
        }
        st.subheader("Live Recommendation (based on last close)")
        st.json(live)

st.markdown("""
---

**Notes & suggestions**
- This app uses *close-of-bar* entries (no peeking at future bars).
- Tune ATR multipliers, RSI threshold, and timeframes per instrument.
- For production, add position-sizing, slippage, commission, and walk-forward validation.
- For indices where Volume is zero or missing, volume-spike checks are disabled.
- Do NOT assume a single indicator universally beats buy-and-hold without robust cross-market validation.
""")
