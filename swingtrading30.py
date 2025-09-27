# hpm2_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import itertools
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="HPM 2.0 — Hybrid Predictive Momentum")

st.title("HPM 2.0 — Hybrid Predictive Momentum (Universal)")

# ---------------------------
# Helper: safe yfinance download (cached)
# ---------------------------
@st.cache_data(ttl=600)
def download_yf(ticker, interval="1d", start=None, end=None, period=None):
    try:
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            df = yf.download(ticker, period=period or "2y", interval=interval, progress=False)
    except Exception as e:
        raise e
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # standardize
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df.rename(columns={'Datetime':'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    for c in ['Open','High','Low','Close']:
        if c not in df.columns:
            return pd.DataFrame()
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df = df[['Date','Open','High','Low','Close','Volume']].dropna().reset_index(drop=True)
    return df

# ---------------------------
# Indicators: squeeze, momentum, regime
# ---------------------------
def compute_hpm_indicators(df, bb_n=20, bb_k=2, kc_n=20, kc_mult=1.5, ema_regime=200, atr_n=14):
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # ATR (rolling)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_n, min_periods=1).mean()

    # Bollinger Bands
    ma = df['Close'].rolling(bb_n).mean()
    std = df['Close'].rolling(bb_n).std()
    df['BB_up'] = ma + bb_k * std
    df['BB_dn'] = ma - bb_k * std
    df['BB_width'] = (df['BB_up'] - df['BB_dn']) / ma

    # Keltner
    ema_kc = df['Close'].ewm(span=kc_n, adjust=False).mean()
    df['KC_up'] = ema_kc + kc_mult * df['ATR']
    df['KC_dn'] = ema_kc - kc_mult * df['ATR']
    df['KC_width'] = (df['KC_up'] - df['KC_dn']) / ema_kc

    # Squeeze: BB narrower than KC (contraction)
    df['squeeze'] = df['BB_width'] < df['KC_width']

    # Momentum: short-term momentum (non-lagging)
    df['mom_3'] = df['Close'].pct_change(3)
    df['mom_5'] = df['Close'].pct_change(5)
    df['mom_accel'] = df['mom_3'] - df['mom_5']  # acceleration

    # Volume pressure (if available)
    if 'Volume' in df.columns:
        df['vol_ma20'] = df['Volume'].rolling(20).mean().fillna(0)
        df['vol_spike'] = df['Volume'] > 1.5 * df['vol_ma20']
    else:
        df['vol_spike'] = False

    # Regime filter: long-term EMA direction
    df['EMA_regime'] = df['Close'].ewm(span=ema_regime, adjust=False).mean()
    df['regime'] = np.where(df['Close'] >= df['EMA_regime'], 1, -1)

    return df

# ---------------------------
# Signal generator: setup + trigger (last-close based)
# ---------------------------
def generate_hpm_signals(df, atr_mul=0.8, rsi_thresh=55, min_mom=0.002):
    df = df.copy().reset_index(drop=True)
    df['setup'] = False
    df['signal'] = 0
    df['reason'] = ""
    # Mark setup when squeeze seen in last N bars (early warning)
    for i in range(len(df)):
        lookback = 12
        if i - lookback < 0:
            window = df.loc[:i,'squeeze']
        else:
            window = df.loc[i-lookback:i,'squeeze']
        if window.any():
            df.at[i,'setup'] = True

    # Trigger: breakout beyond prior 20-bar high/low + ATR + momentum acceleration + regime alignment
    df['prev_high20'] = df['High'].rolling(20).max().shift(1)
    df['prev_low20'] = df['Low'].rolling(20).min().shift(1)
    for i in range(len(df)):
        r = df.loc[i]
        if pd.isna(r['prev_high20']) or pd.isna(r['ATR']):
            continue
        # Long trigger: price closes above prev_high + atr_mul*ATR, momentum accel positive, regime bullish
        if (r['Close'] > r['prev_high20'] + atr_mul * r['ATR'] and
            r['mom_accel'] > min_mom and
            r['regime'] == 1):
            df.at[i,'signal'] = 1
            df.at[i,'reason'] = f"Long trigger: breakout + accel + regime"
            continue
        # Aggressive early long: close above prev_high - 0.4*ATR if setup present and mom positive
        if (r['Close'] > r['prev_high20'] - 0.4 * r['ATR'] and
            r['setup'] and r['mom_3']>0 and r['regime']==1):
            df.at[i,'signal'] = 1
            df.at[i,'reason'] = f"Aggressive Early Long (setup present)"
            continue
        # Short symmetrical
        if (r['Close'] < r['prev_low20'] - atr_mul * r['ATR'] and
            r['mom_accel'] < -min_mom and
            r['regime'] == -1):
            df.at[i,'signal'] = -1
            df.at[i,'reason'] = f"Short trigger"
            continue
        if (r['Close'] < r['prev_low20'] + 0.4 * r['ATR'] and
            r['setup'] and r['mom_3']<0 and r['regime'] == -1):
            df.at[i,'signal'] = -1
            df.at[i,'reason'] = f"Aggressive Early Short (setup present)"
            continue
    return df

# ---------------------------
# Backtester: last-close entry, ATR-based target/sl, EMA momentum exit, max holding
# ---------------------------
def backtest_last_close(df, target_atr=2.0, sl_atr=1.0, max_hold=14, capital=100000):
    df = df.copy().reset_index(drop=True)
    trades = []
    pos = None
    for i in range(len(df)):
        r = df.loc[i]
        if pos is None and r['signal'] != 0:
            # open at close (last-close entry)
            pos = {
                'entry_idx': i,
                'entry_date': r['Date'],
                'entry_price': r['Close'],
                'side': 'LONG' if r['signal']==1 else 'SHORT',
                'atr': r['ATR'],
                'entry_reason': r['reason']
            }
            if pos['atr'] <= 0 or np.isnan(pos['atr']):
                pos['atr'] = 1e-6
            if pos['side']=='LONG':
                pos['target'] = pos['entry_price'] + target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] - sl_atr * pos['atr']
            else:
                pos['target'] = pos['entry_price'] - target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] + sl_atr * pos['atr']
            continue

        if pos is not None:
            price = r['Close']
            exited = False
            exit_price = price
            exit_reason = None
            holding = i - pos['entry_idx']
            # check target/sl
            if pos['side']=='LONG':
                if price >= pos['target']:
                    exit_price = pos['target']; exit_reason = 'Target Hit'; exited=True
                elif price <= pos['sl']:
                    exit_price = pos['sl']; exit_reason = 'SL Hit'; exited=True
                elif price < r['EMA_regime']:  # momentum reversal vs long-term EMA
                    exit_price = price; exit_reason = 'EMA Reversal'; exited=True
            else:
                if price <= pos['target']:
                    exit_price = pos['target']; exit_reason = 'Target Hit'; exited=True
                elif price >= pos['sl']:
                    exit_price = pos['sl']; exit_reason = 'SL Hit'; exited=True
                elif price > r['EMA_regime']:
                    exit_price = price; exit_reason = 'EMA Reversal'; exited=True
            # forced exit
            if not exited and holding >= max_hold:
                exit_price = price; exit_reason = 'Max Hold'; exited=True
            # exit on opposite signal
            if not exited and r['signal'] != 0:
                if (pos['side']=='LONG' and r['signal']==-1) or (pos['side']=='SHORT' and r['signal']==1):
                    exit_price = price; exit_reason='Opposite Signal'; exited=True

            if exited:
                pnl = (exit_price - pos['entry_price']) if pos['side']=='LONG' else (pos['entry_price'] - exit_price)
                pnl_pct = (pnl / pos['entry_price']) * 100
                trades.append({
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'exit_date': r['Date'],
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
    if trades_df.empty:
        summary = {'total_trades':0,'positive_trades':0,'negative_trades':0,'accuracy':0.0,'total_pnl':0.0,'total_pnl_pct':0.0,'strategy_points':0.0,'bh_points':0.0}
    else:
        pos_ct = (trades_df['pnl']>0).sum()
        neg_ct = (trades_df['pnl']<=0).sum()
        total_pnl = trades_df['pnl'].sum()
        buyhold = df['Close'].iloc[-1] - df['Close'].iloc[0]
        summary = {
            'total_trades': int(len(trades_df)),
            'positive_trades': int(pos_ct),
            'negative_trades': int(neg_ct),
            'accuracy': float(pos_ct / len(trades_df)),
            'total_pnl': float(total_pnl),
            'total_pnl_pct': float((total_pnl / float(capital)) * 100),
            'strategy_points': float(trades_df['pnl'].sum()),
            'bh_points': float(buyhold)
        }
    return trades_df, summary

# ---------------------------
# Walk-forward cross validation:
# - Split the time series into N folds (chronological)
# - For each fold: use earlier portion as 'train' to tune a small param grid, then test on next chunk (OOS)
# - Collect OOS metrics across folds
# ---------------------------
def walk_forward_validation(df, n_splits=5, param_grid=None, train_frac=0.6):
    df = df.copy().reset_index(drop=True)
    if param_grid is None:
        param_grid = {
            'atr_mul': [0.6, 0.8, 1.0],
            'target_atr': [1.5, 2.0, 2.5],
            'sl_atr': [0.8, 1.0, 1.2],
            'max_hold': [8, 12, 18]
        }
    # create list of parameter combinations
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(param_grid[k] for k in keys))]
    n = len(df)
    if n < 50:
        return pd.DataFrame(), {}
    # create chronological folds: for n_splits, define train_end and test ranges
    fold_size = n // (n_splits + 1)  # leave last chunk maybe
    results = []
    for k in range(n_splits):
        train_end = (k+1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if len(train_df) < 30 or len(test_df) < 10:
            continue
        # tune on train: brute force small grid, pick param with best total_pnl on train
        best = None
        best_metric = -np.inf
        for c in combos:
            # compute indicators & signals using train params where relevant
            train_ind = compute_hpm_indicators(train_df)
            train_sig = generate_hpm_signals(train_ind, atr_mul=c['atr_mul'], min_mom=0.0015)
            tr_trades, tr_summary = backtest_last_close(train_sig, target_atr=c['target_atr'], sl_atr=c['sl_atr'], max_hold=c['max_hold'])
            metric = tr_summary['total_pnl'] if tr_summary['total_pnl'] is not None else -99999
            if metric > best_metric:
                best_metric = metric
                best = c
        # test with best params on test_df
        test_ind = compute_hpm_indicators(test_df)
        test_sig = generate_hpm_signals(test_ind, atr_mul=best['atr_mul'], min_mom=0.0015)
        test_trades, test_summary = backtest_last_close(test_sig, target_atr=best['target_atr'], sl_atr=best['sl_atr'], max_hold=best['max_hold'])
        results.append({
            'fold': k+1,
            'train_end_idx': train_end,
            'test_start_idx': test_start,
            'test_end_idx': test_end,
            'best_params': best,
            'train_metric': best_metric,
            'test_total_pnl': test_summary['total_pnl'],
            'test_accuracy': test_summary['accuracy'],
            'test_trades': test_summary['total_trades']
        })
    results_df = pd.DataFrame(results)
    # aggregate summary
    if results_df.empty:
        agg = {}
    else:
        agg = {
            'folds': len(results_df),
            'avg_test_pnl': float(results_df['test_total_pnl'].mean()),
            'avg_test_accuracy': float(results_df['test_accuracy'].mean())
        }
    return results_df, agg

# ---------------------------
# Heatmaps helper
# ---------------------------
def monthly_yearly_heatmap(df):
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    month_close = df.set_index('Date').resample('M')['Close'].last().reset_index()
    month_close['Year'] = month_close['Date'].dt.year
    month_close['Month'] = month_close['Date'].dt.month
    month_close['ret'] = month_close['Close'].pct_change() * 100
    pivot_month = month_close.pivot(index='Year', columns='Month', values='ret')
    year_close = df.set_index('Date').resample('Y')['Close'].last().reset_index()
    year_close['ret'] = year_close['Close'].pct_change() * 100
    pivot_year = year_close[['Date','ret']].set_index(year_close['Date'].dt.year)['ret']
    return pivot_month, pivot_year

# ---------------------------
# UI: Inputs & fetch control
# ---------------------------
with st.sidebar:
    st.header("Data & Parameters")
    source = st.selectbox("Data source", ["yfinance", "CSV upload"])
    interval = st.selectbox("Interval", ["1d","60m","15m"], index=0)
    ticker = st.text_input("Ticker (for yfinance)", value="^NSEI")
    use_dates = st.checkbox("Use Start/End dates (prefer for long history)", value=True)
    if use_dates:
        start = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365*5))
        end = st.date_input("End date", value=datetime.now().date())
        start_s = start.strftime("%Y-%m-%d"); end_s = end.strftime("%Y-%m-%d")
    else:
        period = st.selectbox("Period (yfinance)", ["1y","2y","5y","10y","max"], index=1)
        start_s = end_s = None
    uploaded = st.file_uploader("Upload CSV (Date, Open, High, Low, Close, Volume)", type=['csv']) if source=='CSV upload' else None

    st.markdown("### Strategy base params")
    bb_n = st.number_input("BB/KC window (n)", value=20, min_value=5)
    kc_mult = st.number_input("KC ATR multiplier", value=1.5)
    atr_n = st.number_input("ATR period", value=14, min_value=1)
    ema_regime = st.number_input("EMA regime length", value=200, min_value=20)
    # backtest params
    target_atr_def = st.number_input("Default Target (xATR)", value=2.0, step=0.1)
    sl_atr_def = st.number_input("Default SL (xATR)", value=1.0, step=0.1)
    max_hold = st.number_input("Max holding bars", value=12, min_value=1)
    capital = st.number_input("Capital (for %)", value=100000)
    # walk-forward params
    st.markdown("### Walk-forward")
    wf_splits = st.slider("WF splits (folds)", 3, 8, value=4)
    run_fetch = st.button("Fetch Data & Run (safe single click)")

# session state storage for fetched data
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = pd.DataFrame()
    st.session_state.last_fetch_ts = None

if run_fetch:
    try:
        if source == 'yfinance':
            raw = download_yf(ticker, interval=interval, start=start_s, end=end_s, period=None)
        else:
            if uploaded is None:
                st.error("Please upload CSV")
                st.stop()
            raw = pd.read_csv(uploaded)
            if 'Date' not in raw.columns and 'date' in raw.columns:
                raw.rename(columns={'date':'Date'}, inplace=True)
            raw['Date'] = pd.to_datetime(raw['Date'])
            # ensure required columns
            for c in ['Open','High','Low','Close']:
                if c not in raw.columns:
                    st.error(f"CSV missing column: {c}")
                    st.stop()
            if 'Volume' not in raw.columns:
                raw['Volume'] = 0
            raw = raw[['Date','Open','High','Low','Close','Volume']].dropna().reset_index(drop=True)
        if raw.empty:
            st.error("No data returned. Check ticker/period or CSV format.")
        else:
            st.session_state.raw_df = raw.copy()
            st.session_state.last_fetch_ts = datetime.now().isoformat()
    except Exception as e:
        st.error(f"Data fetch failed: {e}")

# If data present in session, use it
data = st.session_state.raw_df.copy()
if data.empty:
    st.info("No data loaded. Click 'Fetch Data & Run' after setting inputs, or upload CSV.")
    st.stop()

# ---------------------------
# Compute indicators & signals
# ---------------------------
with st.spinner("Computing indicators and signals..."):
    df_ind = compute_hpm_indicators(data, bb_n=bb_n, bb_k=2, kc_n=bb_n, kc_mult=kc_mult, ema_regime=ema_regime, atr_n=atr_n)
    df_sig = generate_hpm_signals(df_ind, atr_mul=0.8, min_mom=0.0015)

# ---------------------------
# Run backtest on full data with default params
# ---------------------------
with st.spinner("Running backtest..."):
    trades_all, summary_all = backtest_last_close(df_sig, target_atr=target_atr_def, sl_atr=sl_atr_def, max_hold=max_hold, capital=capital)

# ---------------------------
# Walk-forward validation
# ---------------------------
with st.spinner("Running walk-forward validation (may take a moment)..."):
    param_grid = {
        'atr_mul': [0.6, 0.8, 1.0],
        'target_atr': [1.5, 2.0, 2.5],
        'sl_atr': [0.8, 1.0, 1.2],
        'max_hold': [8, 12, 18]
    }
    wf_df, wf_agg = walk_forward_validation(df_ind, n_splits=wf_splits, param_grid=param_grid)

# ---------------------------
# Heatmaps
# ---------------------------
pivot_month, pivot_year = monthly_yearly_heatmap(data)

# ---------------------------
# UI: display results
# ---------------------------
# Top row: summaries + live recommendation
col1, col2, col3 = st.columns([1.2,1,1])
with col1:
    st.subheader("Backtest Summary (full history)")
    st.metric("Total trades", summary_all['total_trades'])
    st.metric("Positive trades", summary_all['positive_trades'])
    st.metric("Negative trades", summary_all['negative_trades'])
    st.metric("Accuracy", f"{summary_all['accuracy']*100:.2f}%")
with col2:
    st.subheader("PnL & Comparison")
    st.write(f"- Strategy total PnL (points): **{summary_all['strategy_points']:.2f}**")
    st.write(f"- Strategy PnL (% of capital): **{summary_all['total_pnl_pct']:.4f}%**")
    st.write(f"- Buy & Hold (first->last points): **{summary_all['bh_points']:.2f}**")
with col3:
    st.subheader("Walk-Forward Summary")
    if wf_df.empty:
        st.write("Not enough data for WF or no folds produced.")
    else:
        st.write(f"- WF folds: {wf_agg.get('folds',0)}")
        st.write(f"- Avg OOS pnl per fold: {wf_agg.get('avg_test_pnl',0):.2f}")
        st.write(f"- Avg OOS accuracy: {wf_agg.get('avg_test_accuracy',0)*100 if wf_agg else 0:.2f}%")

# Live recommendation
st.subheader("Live Recommendation (last close)")
last = df_sig.iloc[-1]
if last['signal'] == 0:
    st.info("No signal on last close.")
else:
    side = 'LONG' if last['signal']==1 else 'SHORT'
    entry = float(last['Close'])
    atr = last['ATR'] if last['ATR']>0 else 1e-6
    target = entry + target_atr_def * atr if side=='LONG' else entry - target_atr_def * atr
    sl = entry - sl_atr_def * atr if side=='LONG' else entry + sl_atr_def * atr
    # simple probability from historical accuracy
    prob = summary_all['accuracy'] if summary_all['total_trades']>0 else 0.0
    st.json({
        'entry_date_time': str(last['Date']),
        'side': side,
        'levels': entry,
        'target': float(target),
        'sl': float(sl),
        'reason_of_entry': last['reason'],
        'probability_of_profit': f"{prob*100:.1f}%"
    })

# Chart with entry/exit markers
st.subheader("Price chart with signals")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df_sig['Date'], df_sig['Close'], label='Close')
ax.plot(df_sig['Date'], df_sig['EMA_regime'], label=f'EMA_regime({ema_regime})', linewidth=0.8)
ax.scatter(df_sig.loc[df_sig['signal']==1,'Date'], df_sig.loc[df_sig['signal']==1,'Close'], marker='^', label='Long Trigger', s=40)
ax.scatter(df_sig.loc[df_sig['signal']==-1,'Date'], df_sig.loc[df_sig['signal']==-1,'Close'], marker='v', label='Short Trigger', s=40)
ax.set_title(f"{ticker} — Price & Signals")
ax.legend()
st.pyplot(fig)

# Show top 5 / bottom 5 trades
st.subheader("Top 5 / Bottom 5 trades")
if trades_all.empty:
    st.info("No trades generated.")
else:
    top5 = trades_all.sort_values(by='pnl', ascending=False).head(5)
    bot5 = trades_all.sort_values(by='pnl', ascending=True).head(5)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top 5 profitable trades")
        st.dataframe(top5, height=220, use_container_width=True)
    with c2:
        st.write("Bottom 5 losing trades")
        st.dataframe(bot5, height=220, use_container_width=True)

# Full trade log (scrollable)
st.subheader("Full Trade Log (scrollable)")
if trades_all.empty:
    st.write("No trades to display.")
else:
    st.dataframe(trades_all.reset_index(drop=True), height=300, use_container_width=True)

# Original raw data display (scrollable)
st.subheader("Original Price Data (scrollable)")
st.dataframe(data.reset_index(drop=True), height=300, use_container_width=True)

# Heatmaps
st.subheader("Monthly Returns Heatmap")
if not pivot_month.empty:
    fig2, ax2 = plt.subplots(figsize=(10,4))
    im = ax2.imshow(pivot_month.fillna(0).values, aspect='auto', cmap='RdYlGn', vmin=-30, vmax=30)
    ax2.set_yticks(np.arange(pivot_month.shape[0])); ax2.set_yticklabels(pivot_month.index)
    ax2.set_xticks(np.arange(12)); ax2.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2.set_title("Monthly returns (%)")
    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)
else:
    st.write("Not enough data for monthly heatmap.")

st.subheader("Yearly Returns")
if not pivot_year.empty:
    st.dataframe(pivot_year.rename("Yearly % Return").to_frame(), use_container_width=True)
else:
    st.write("Not enough data for yearly returns.")

# Diagnostics / notes
st.markdown("---")
