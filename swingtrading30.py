# app_updated.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import functools

st.set_page_config(layout="wide", page_title="ABM+Squeeze Swing Algo (Updated)")

# ---------------------------
# Utilities & caching
# ---------------------------
@st.cache_data(ttl=300)
def download_data_yf_safe(ticker, interval="1d", start=None, end=None, period=None):
    """
    Safe wrapper around yfinance download:
      - prefer start/end when provided (more control)
      - otherwise use period
      - always reset index and ensure Date is datetime
      - returns standardized columns Date, Open, High, Low, Close, Volume
    """
    # choose method
    try:
        if start is not None or end is not None:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            # safe usage of period. yfinance supports 'max' for long history (subject to ticker availability)
            data = yf.download(ticker, period=period or "2y", interval=interval, progress=False)
    except Exception as e:
        raise e

    if data is None or data.empty:
        return pd.DataFrame()
    # ensure columns exist
    for col in ['Open','High','Low','Close']:
        if col not in data.columns:
            return pd.DataFrame()

    data = data.reset_index()
    # standardize column name for Date
    if 'Date' not in data.columns and 'Datetime' in data.columns:
        data.rename(columns={'Datetime':'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    # ensure numeric
    data[['Open','High','Low','Close']] = data[['Open','High','Low','Close']].apply(pd.to_numeric, errors='coerce')
    # Volume may be missing or zero on some indices; keep as-is
    if 'Volume' not in data.columns:
        data['Volume'] = 0
    # drop rows with NaNs in price columns
    data = data.dropna(subset=['Open','High','Low','Close']).reset_index(drop=True)
    return data[['Date','Open','High','Low','Close','Volume']]

# ---------------------------
# Indicators
# ---------------------------
def compute_indicators(df, atr_period=14, ema_fast=8, rsi_period=7, bb_n=20, bb_k=2, kc_n=20, kc_mult=1.5):
    """
    Computes:
     - ATR
     - EMA_fast
     - RSI (fast)
     - Momentum (3 bar)
     - Prev high/low for breakout baseline
     - Bollinger Bands & Keltner (squeeze)
    Careful about alignment: operate on same dataframe indices.
    """
    df = df.copy().reset_index(drop=True)
    # True range
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_period, min_periods=1).mean()

    df['EMA_fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()

    # RSI (fast, EWMA)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['RSI_fast'] = 100 - (100 / (1 + rs))

    df['mom3'] = df['Close'].pct_change(3)

    df['prev_high_n'] = df['High'].rolling(20).max().shift(1)
    df['prev_low_n'] = df['Low'].rolling(20).min().shift(1)

    # Bollinger Bands
    ma = df['Close'].rolling(bb_n).mean()
    std = df['Close'].rolling(bb_n).std()
    df['bb_upper'] = ma + bb_k * std
    df['bb_lower'] = ma - bb_k * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma

    # Keltner Channels (EMA + ATR)
    ema_kc = df['Close'].ewm(span=kc_n, adjust=False).mean()
    df['kc_upper'] = ema_kc + kc_mult * df['ATR']
    df['kc_lower'] = ema_kc - kc_mult * df['ATR']
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / ema_kc

    # Squeeze detection: when BB width < KC width -> squeeze (contraction)
    df['squeeze'] = df['bb_width'] < df['kc_width']

    # volume spike
    if 'Volume' in df.columns:
        df['vol_ma'] = df['Volume'].rolling(20).mean().fillna(0)
        df['vol_spike'] = df['Volume'] > (1.5 * df['vol_ma'])
    else:
        df['vol_spike'] = False

    return df

# ---------------------------
# Signal generation (setup + trigger)
# ---------------------------
def generate_signals_setup_trigger(df, atr_multiplier=0.8, rsi_threshold=55, min_mom=0.0015):
    """
    Two-stage system:
      - Setup: squeeze detected (contraction). We mark a 'setup' so user can see early.
      - Trigger (actual trade signal, executed at CLOSE of trigger bar): breakout from prior 20 high/low + ATR
      - Attempt to be earlier by also creating 'aggressive trigger' if close crosses mid of prior-range + rising momentum.
    Returns df with columns: setup (bool), signal (1 long, -1 short, 0 none), reason
    """
    df = df.copy().reset_index(drop=True)
    df['setup'] = False
    df['signal'] = 0
    df['reason'] = ''
    # mark setup when squeeze True and not previously in squeeze
    df['squeeze_start'] = (~df['squeeze'].shift(1).fillna(False)) & df['squeeze']
    # mark setup as recent squeeze (we'll mark setup if squeeze happened within last 10 bars)
    for i in range(len(df)):
        if df.loc[max(0, i-10):i, 'squeeze'].any():
            df.at[i, 'setup'] = True

    for i in range(len(df)):
        row = df.loc[i]
        # skip if insufficient history
        if pd.isna(row['prev_high_n']) or pd.isna(row['ATR']):
            continue

        # strict breakout trigger
        if (row['Close'] > row['prev_high_n'] + atr_multiplier * row['ATR'] and
            row['RSI_fast'] >= rsi_threshold and
            row['mom3'] >= min_mom):
            df.at[i, 'signal'] = 1
            df.at[i, 'reason'] = f"Breakout long: close>{row['prev_high_n']:.3f}+{atr_multiplier:.2f}*ATR; RSI {row['RSI_fast']:.1f}"
            continue

        # aggressive long: cross above prev_high - 0.5*ATR while squeeze previously present + rising mom
        if (row['Close'] > row['prev_high_n'] - 0.5 * row['ATR'] and
            df.loc[max(0, i-5):i, 'squeeze'].any() and
            row['mom3'] > (min_mom/2) and
            row['RSI_fast'] > (rsi_threshold - 5)):
            df.at[i, 'signal'] = 1
            df.at[i, 'reason'] = "Aggressive long trigger (early)"
            continue

        # short triggers symmetrical
        if (row['Close'] < row['prev_low_n'] - atr_multiplier * row['ATR'] and
            row['RSI_fast'] <= (100 - rsi_threshold) and
            row['mom3'] <= -min_mom):
            df.at[i, 'signal'] = -1
            df.at[i, 'reason'] = f"Breakout short"
            continue

        if (row['Close'] < row['prev_low_n'] + 0.5 * row['ATR'] and
            df.loc[max(0, i-5):i, 'squeeze'].any() and
            row['mom3'] < -(min_mom/2) and
            row['RSI_fast'] < (100 - rsi_threshold + 5)):
            df.at[i, 'signal'] = -1
            df.at[i, 'reason'] = "Aggressive short trigger (early)"
            continue

    return df

# ---------------------------
# Backtester (last-close entries)
# ---------------------------
def backtest_last_close(df, target_atr=2.0, sl_atr=1.0, max_holding_bars=14, capital=100000):
    df = df.copy().reset_index(drop=True)
    trades = []
    pos = None
    for i in range(len(df)):
        row = df.loc[i]
        if pos is None and row['signal'] != 0:
            # open position at close (last-close entry)
            pos = {
                'entry_idx': i,
                'entry_datetime': row['Date'],
                'entry_price': row['Close'],
                'side': 'LONG' if row['signal']==1 else 'SHORT',
                'atr': row['ATR'],
                'entry_reason': row['reason']
            }
            if pos['atr'] <= 0 or pd.isna(pos['atr']):
                pos['atr'] = 1e-6
            # set target/sl
            if pos['side']=='LONG':
                pos['target'] = pos['entry_price'] + target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] - sl_atr * pos['atr']
            else:
                pos['target'] = pos['entry_price'] - target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] + sl_atr * pos['atr']
            continue

        # manage open
        if pos is not None:
            price = row['Close']
            exited = False
            exit_price = price
            exit_reason = None
            holding = i - pos['entry_idx']

            # check target/sl using CLOSE-based criteria
            if pos['side']=='LONG':
                if price >= pos['target']:
                    exit_price = pos['target']; exit_reason = 'Target Hit'; exited=True
                elif price <= pos['sl']:
                    exit_price = pos['sl']; exit_reason = 'SL Hit'; exited=True
                elif price < row['EMA_fast']:
                    exit_price = price; exit_reason = 'Momentum Reversal (EMA)'; exited=True
            else:
                if price <= pos['target']:
                    exit_price = pos['target']; exit_reason = 'Target Hit'; exited=True
                elif price >= pos['sl']:
                    exit_price = pos['sl']; exit_reason = 'SL Hit'; exited=True
                elif price > row['EMA_fast']:
                    exit_price = price; exit_reason = 'Momentum Reversal (EMA)'; exited=True

            if not exited and holding >= max_holding_bars:
                exit_price = price; exit_reason='Max Holding'; exited=True

            # exit on opposite signal
            if not exited and row['signal'] != 0:
                if (pos['side']=='LONG' and row['signal'] == -1) or (pos['side']=='SHORT' and row['signal']==1):
                    exit_price = price; exit_reason='Opposite Signal'; exited=True

            if exited:
                pnl = (exit_price - pos['entry_price']) if pos['side']=='LONG' else (pos['entry_price'] - exit_price)
                pnl_pct = pnl / pos['entry_price'] * 100
                trades.append({
                    'entry_datetime': pos['entry_datetime'],
                    'entry_price': pos['entry_price'],
                    'exit_datetime': row['Date'],
                    'exit_price': exit_price,
                    'side': pos['side'],
                    'target': pos['target'],
                    'sl': pos['sl'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_bars': holding,
                    'entry_reason': pos['entry_reason'],
                    'exit_reason': exit_reason
                })
                pos = None

    trades_df = pd.DataFrame(trades)
    # summary stats
    if trades_df.empty:
        summary = {'total_trades':0,'positive_trades':0,'negative_trades':0,'accuracy':0.0,'total_pnl':0.0,'total_pnl_pct':0.0,'total_points_strategy':0.0,'total_points_buy_hold':0.0}
    else:
        pos_ct = (trades_df['pnl']>0).sum()
        neg_ct = (trades_df['pnl']<=0).sum()
        total_pnl = trades_df['pnl'].sum()
        buy_hold = df['Close'].iloc[-1] - df['Close'].iloc[0]
        summary = {
            'total_trades': len(trades_df),
            'positive_trades': int(pos_ct),
            'negative_trades': int(neg_ct),
            'accuracy': float(pos_ct / len(trades_df)),
            'total_pnl': float(total_pnl),
            'total_pnl_pct': float((total_pnl / capital) * 100),
            'total_points_strategy': float(trades_df['pnl'].sum()),
            'total_points_buy_hold': float(buy_hold)
        }
    return trades_df, summary

# ---------------------------
# Heatmaps: monthly and yearly returns
# ---------------------------
def returns_heatmaps(df):
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    # monthly returns: pct change month-end close
    month_close = df.set_index('Date').resample('M')['Close'].last().reset_index()
    month_close['Year'] = month_close['Date'].dt.year
    month_close['Month'] = month_close['Date'].dt.month
    month_close['ret'] = month_close['Close'].pct_change() * 100
    pivot_m = month_close.pivot(index='Year', columns='Month', values='ret')
    # yearly returns
    year_close = df.set_index('Date').resample('Y')['Close'].last().reset_index()
    year_close['ret'] = year_close['Close'].pct_change() * 100
    pivot_y = year_close[['Date','ret']].set_index(year_close['Date'].dt.year)['ret']
    return pivot_m, pivot_y

# ---------------------------
# UI & flow
# ---------------------------
st.title("ABM + Squeeze Swing Algo — Updated")

# left panel: input
left, right = st.columns([1,2])
with left:
    st.subheader("Data selection")
    ticker = st.text_input("Ticker (yfinance)", value="^NSEI")
    use_csv = st.checkbox("Upload CSV instead of yfinance", value=False)
    uploaded_file = st.file_uploader("Upload OHLCV CSV", type=['csv']) if use_csv else None

    interval = st.selectbox("Interval", options=["1d","60m","15m"], index=0)
    # advanced: allow start/end or period
    use_dates = st.checkbox("Use start/end dates (preferred for very long history)", value=False)
    if use_dates:
        start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365*5))
        end_date = st.date_input("End date", value=datetime.now().date())
        start = start_date.strftime("%Y-%m-%d")
        end = (datetime.combine(end_date, datetime.max.time())).strftime("%Y-%m-%d")
        period = None
    else:
        period = st.selectbox("Period (yfinance)", options=["6mo","1y","2y","5y","10y","max"], index=1)
        start = end = None

    st.markdown("**Strategy params**")
    atr_p = st.number_input("ATR period", value=14, min_value=1)
    atr_multiplier = st.number_input("Break ATR multiplier", value=0.8, step=0.1)
    target_atr = st.number_input("Target (x ATR)", value=2.0, step=0.1)
    sl_atr = st.number_input("SL (x ATR)", value=1.0, step=0.1)
    rsi_threshold = st.number_input("RSI threshold (fast)", value=55)
    max_hold = st.number_input("Max holding bars", value=10, min_value=1)
    capital = st.number_input("Capital (for % calc)", value=100000)

    # fetch control (important to avoid rate limits)
    if 'fetched' not in st.session_state:
        st.session_state.fetched = False
    fetch_button = st.button("Fetch Data (safe, single-click)")

    # optional: user wants to force refresh
    force_refresh = st.checkbox("Force refresh cached data", value=False)

with right:
    st.subheader("Results")
    results_area = st.empty()

# Data loading controlled by Fetch button (prevents auto requests)
data_df = pd.DataFrame()
if fetch_button:
    st.session_state.fetched = True
    try:
        if use_csv and uploaded_file is not None:
            raw = pd.read_csv(uploaded_file)
            # attempt to standardize
            cols_lower = {c.lower(): c for c in raw.columns}
            # map common names
            mapping = {}
            for need in ['date','open','high','low','close','volume']:
                if need in cols_lower:
                    mapping[cols_lower[need]] = need.capitalize()
            raw = raw.rename(columns=mapping)
            if 'Date' not in raw.columns and 'date' in raw.columns:
                raw.rename(columns={'date':'Date'}, inplace=True)
            raw['Date'] = pd.to_datetime(raw['Date'])
            # try to ensure column names exist
            for c in ['Open','High','Low','Close']:
                if c not in raw.columns:
                    st.error(f"CSV missing required column: {c}")
                    st.stop()
            raw = raw[['Date','Open','High','Low','Close'] + ([ 'Volume' ] if 'Volume' in raw.columns else [])]
            data_df = raw.copy().reset_index(drop=True)
        else:
            # small sleep to be kind to yfinance & avoid bursts
            time.sleep(0.5)
            data_df = download_data_yf_safe(ticker, interval=interval, start=start, end=end, period=period)
            if data_df.empty:
                st.error("No data returned from yfinance for this ticker/interval/period. Try a different period or upload CSV.")
                st.session_state.fetched = False
    except Exception as e:
        st.session_state.fetched = False
        st.error(f"Data download failed: {e}")

# If previously fetched, keep cached version
if st.session_state.get('fetched', False) and data_df.empty:
    # try to load from cache function (download_data_yf_safe caches by args) by calling with same args if available
    try:
        # attempt silent fetch via cache if possible (won't run network if cached)
        data_df = download_data_yf_safe(ticker, interval=interval, start=start, end=end, period=period)
    except:
        pass

if not data_df.empty:
    # compute indicators & signals
    df_ind = compute_indicators(data_df, atr_period=int(atr_p), ema_fast=8, rsi_period=7)
    df_sig = generate_signals_setup_trigger(df_ind, atr_multiplier=float(atr_multiplier), rsi_threshold=float(rsi_threshold), min_mom=0.0015)

    # backtest and live recommendation (simultaneous)
    trades_df, summary = backtest_last_close(df_sig, target_atr=float(target_atr), sl_atr=float(sl_atr), max_holding_bars=int(max_hold), capital=float(capital))

    # Live rec based on last close (no further button)
    last = df_sig.iloc[-1]
    live_reco = None
    if last['signal'] != 0:
        side = 'LONG' if last['signal']==1 else 'SHORT'
        entry = last['Close']
        atr = last['ATR'] if last['ATR']>0 else 1e-6
        target = entry + target_atr * atr if side=='LONG' else entry - target_atr * atr
        sl = entry - sl_atr * atr if side=='LONG' else entry + sl_atr * atr
        # simple estimator: use historical backtest accuracy
        prob = summary['accuracy'] if summary['total_trades']>0 else 0.0
        live_reco = {
            'entry_date_time': last['Date'],
            'side': side,
            'levels': float(entry),
            'target': float(target),
            'sl': float(sl),
            'reason_of_entry': last['reason'],
            'probability_of_profit': f"{prob*100:.1f}%"
        }

    # Produce heatmaps
    pivot_monthly, pivot_yearly = returns_heatmaps(data_df)

    # Display results nicely
    with results_area.container():
        st.markdown("### Backtest Summary")
        st.write(f"- Total trades: **{summary['total_trades']}**")
        st.write(f"- Positive trades: **{summary['positive_trades']}**, Negative trades: **{summary['negative_trades']}**")
        st.write(f"- Accuracy: **{summary['accuracy']*100:.2f}%**")
        st.write(f"- Total PnL (points): **{summary['total_points_strategy']:.2f}**")
        st.write(f"- Total PnL (% of capital): **{summary['total_pnl_pct']:.4f}%**")
        st.write(f"- Buy & Hold points (first->last): **{summary['total_points_buy_hold']:.2f}**")

        st.markdown("### Trades (each row = last-close entry & exit)")
        if trades_df.empty:
            st.info("No trades generated with current parameters.")
        else:
            # add fields required by you
            trades_display = trades_df.copy()
            trades_display['entry_date_time'] = trades_display['entry_datetime']
            trades_display['levels'] = trades_display['entry_price']
            trades_display['total_pnl'] = trades_display['pnl']
            trades_display['total_pnl_percentage'] = trades_display['pnl_pct']
            trades_display = trades_display[['entry_date_time','levels','target','sl','total_pnl','total_pnl_percentage','entry_reason','exit_reason','holding_bars','side']]
            st.data_editor(trades_display, use_container_width=True)

        st.markdown("### Live Recommendation (based on last close)")
        if live_reco is None:
            st.info("No signal on last close.")
        else:
            st.json(live_reco)

        st.markdown("### Squeeze (Setup) Alerts — earliest warning before trigger")
        # show last few bars with setup True
        setups = df_sig.loc[df_sig['setup'] | (df_sig['squeeze_start']) , ['Date','Close','squeeze','squeeze_start','signal','reason']].tail(20)
        if setups.empty:
            st.write("No recent setups detected.")
        else:
            st.dataframe(setups)

        st.markdown("### Price Chart with EMA_fast & signals")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_sig['Date'], df_sig['Close'], label='Close')
        ax.plot(df_sig['Date'], df_sig['EMA_fast'], label='EMA_fast', linewidth=0.9)
        # plot setup bars
        ax.scatter(df_sig.loc[df_sig['squeeze_start'],'Date'], df_sig.loc[df_sig['squeeze_start'],'Close'], marker='v', label='Squeeze Start', s=40)
        # plot signals
        ax.scatter(df_sig.loc[df_sig['signal']==1,'Date'], df_sig.loc[df_sig['signal']==1,'Close'], marker='^', label='Long Trigger', s=50)
        ax.scatter(df_sig.loc[df_sig['signal']==-1,'Date'], df_sig.loc[df_sig['signal']==-1,'Close'], marker='v', label='Short Trigger', s=50)
        ax.set_title(f"{ticker} Price & Signals")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Monthly Returns Heatmap")
        if not pivot_monthly.empty:
            fig2, ax2 = plt.subplots(figsize=(10,4))
            # heatmap via imshow; keep labels
            im = ax2.imshow(pivot_monthly.fillna(0).values, aspect='auto', cmap='RdYlGn', vmin=-20, vmax=20)
            ax2.set_yticks(np.arange(pivot_monthly.shape[0])); ax2.set_yticklabels(pivot_monthly.index)
            ax2.set_xticks(np.arange(12)); ax2.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
            ax2.set_title("Monthly returns (%)")
            plt.colorbar(im, ax=ax2, label='%')
            st.pyplot(fig2)
        else:
            st.write("Not enough data for monthly heatmap.")

        st.markdown("### Yearly Returns")
        if not pivot_yearly.empty:
            st.dataframe(pivot_yearly.rename('Yearly % Return').to_frame())
        else:
            st.write("Not enough data for yearly returns.")

        st.markdown("---")
        st.markdown("""
        **Notes / fixes made**
        - Fixed the yfinance alignment error by ensuring we work on the same index and computing TR via aligned shifts and per-row max.
        - Data fetch is only executed when you press **Fetch Data**. This reduces yfinance calls and prevents accidental rate limits.
        - `st.session_state` preserves fetched data & UI state so results don't vanish after clicks.
        - Added squeeze (setup) detection so you get advanced warnings before breakouts; an "aggressive trigger" attempts earlier detection (still last-close entry).
        - Live recommendation is shown immediately after data fetch/backtest (no extra button).
        - CSV upload fallback for very long history or alternative data sources.
        - Monthly & yearly heatmaps added.
        - Long/Short side is shown per trade.
        """)

else:
    st.info("Click **Fetch Data** to download data from yfinance or upload CSV. This prevents repeated automatic calls and reduces the chance of yfinance rate-limits.")

st.sidebar.markdown("## Tips")
