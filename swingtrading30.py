"""
Streamlit app: Psychological Momentum Swing Strategy
- Fetch data via yfinance or upload CSV
- Compute PsychMomentum + Liquidity Sweep signals
- Backtest (entry at signal-bar close) and show trade-by-trade results
- Live recommendation uses last-candle close

How to run:
1. pip install streamlit yfinance pandas numpy matplotlib
2. streamlit run psych_swing_streamlit.py

Notes:
- No TA-lib dependency (pure pandas/numpy)
- Strategy is illustrative; backtesting on your data is strongly recommended.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Psych Swing Strategy", layout="wide")

# ---------- Utilities ----------
@st.cache_data
def fetch_data_yf(ticker, period="2y", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError("No data fetched for %s" % ticker)
    df = df.rename_axis('datetime').reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.columns = [c.capitalize() for c in df.columns]
    return df


def compute_indicators(df, params):
    df = df.copy()
    # Rolling return
    n = params['momentum_lookback']
    df['rtn'] = df['Close'] / df['Close'].shift(n) - 1
    # ATR simple
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(params['atr_lookback']).mean().fillna(method='bfill')
    # Volume zscore
    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(params['vol_lookback']).mean()) / (df['Volume'].rolling(params['vol_lookback']).std()+1e-9)
    # Candle strength
    df['candle_strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)
    # Psych Score: weighted
    df['pscore_raw'] = (df['rtn'].fillna(0) / (df['rtn'].rolling(params['momentum_lookback']).std()+1e-9)) * params['w_rtn'] \
                       + df['vol_z'].fillna(0) * params['w_vol'] \
                       + df['candle_strength'].fillna(0) * params['w_candle']
    # Normalize score
    df['pscore'] = (df['pscore_raw'] - df['pscore_raw'].rolling(params['pscore_norm']).mean()) / (df['pscore_raw'].rolling(params['pscore_norm']).std()+1e-9)

    # Liquidity sweep detection: last candle extended lower than recent lows but closed back up (long sweep)
    lookback = params['sweep_lookback']
    df['sweep_long'] = False
    df['sweep_short'] = False
    for i in range(lookback, len(df)):
        window = df.loc[i-lookback:i-1]
        prev_low = window['Low'].min()
        prev_high = window['High'].max()
        cur = df.loc[i]
        prev = df.loc[i-1]
        # Long sweep: previous candle pierced below prev_low but closed near/above prev range
        if prev['Low'] < prev_low and prev['Close'] > prev['Open'] and prev['Close'] > (window['Close'].mean()):
            df.at[i, 'sweep_long'] = True
        # Short sweep: previous candle pierced above prev_high but closed near/below prev range
        if prev['High'] > prev_high and prev['Close'] < prev['Open'] and prev['Close'] < (window['Close'].mean()):
            df.at[i, 'sweep_short'] = True

    return df


def backtest(df, params, capital=10000, risk_per_trade=0.01, verbose=False):
    df = df.copy().reset_index(drop=True)
    trades = []
    eq = capital
    peak = capital
    equity_curve = []

    for i in range(params['start_index'], len(df)-1):
        row = df.loc[i]
        # Long signal
        if row['pscore'] > params['pscore_entry'] and (row['sweep_long'] or row['Close'] > df['Close'].rolling(params['consolidation']).max().iloc[i-1]):
            entry_price = row['Close']
            sl = min(df['Low'].rolling(params['stop_lookback']).min().iloc[i-1], entry_price - params['min_stop_buffer'] * row['atr'])
            # Ensure SL < entry
            sl = min(sl, entry_price - 0.001)
            risk_per_share = entry_price - sl
            if risk_per_share <= 0:
                continue
            risk_amount = eq * risk_per_trade
            qty = max(1, int(risk_amount / (risk_per_share)))
            target = entry_price + params['reward_ratio'] * risk_per_share
            # Simulate forward
            hit = None
            exit_price = None
            exit_idx = None
            for j in range(i+1, min(len(df), i+params['max_holding'])):
                low = df.loc[j, 'Low']
                high = df.loc[j, 'High']
                if low <= sl:
                    hit = 'SL'
                    exit_price = sl
                    exit_idx = j
                    break
                if high >= target:
                    hit = 'TP'
                    exit_price = target
                    exit_idx = j
                    break
            if hit is None:
                # Exit on close of last allowed holding
                exit_idx = min(len(df)-1, i+params['max_holding'])
                exit_price = df.loc[exit_idx, 'Close']
                hit = 'TimeExit'
            pnl = (exit_price - entry_price) * qty
            eq += pnl
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            trades.append({
                'entry_idx': i,
                'entry_dt': row['datetime'],
                'entry_price': entry_price,
                'target': target,
                'stop': sl,
                'exit_idx': exit_idx,
                'exit_dt': df.loc[exit_idx, 'datetime'],
                'exit_price': exit_price,
                'pnl': pnl,
                'trade_type': 'LONG',
                'reason_entry': 'PsychMomentum+Sweep/Breakout',
                'reason_exit': hit,
                'pscore': row['pscore']
            })
        # Short signal (mirror)
        if row['pscore'] < -params['pscore_entry'] and (row['sweep_short'] or row['Close'] < df['Close'].rolling(params['consolidation']).min().iloc[i-1]):
            entry_price = row['Close']
            sl = max(df['High'].rolling(params['stop_lookback']).max().iloc[i-1], entry_price + params['min_stop_buffer'] * row['atr'])
            sl = max(sl, entry_price + 0.001)
            risk_per_share = sl - entry_price
            if risk_per_share <= 0:
                continue
            risk_amount = eq * risk_per_trade
            qty = max(1, int(risk_amount / (risk_per_share)))
            target = entry_price - params['reward_ratio'] * risk_per_share
            hit = None
            exit_price = None
            exit_idx = None
            for j in range(i+1, min(len(df), i+params['max_holding'])):
                low = df.loc[j, 'Low']
                high = df.loc[j, 'High']
                if high >= sl:
                    hit = 'SL'
                    exit_price = sl
                    exit_idx = j
                    break
                if low <= target:
                    hit = 'TP'
                    exit_price = target
                    exit_idx = j
                    break
            if hit is None:
                exit_idx = min(len(df)-1, i+params['max_holding'])
                exit_price = df.loc[exit_idx, 'Close']
                hit = 'TimeExit'
            pnl = (entry_price - exit_price) * qty
            eq += pnl
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            trades.append({
                'entry_idx': i,
                'entry_dt': row['datetime'],
                'entry_price': entry_price,
                'target': target,
                'stop': sl,
                'exit_idx': exit_idx,
                'exit_dt': df.loc[exit_idx, 'datetime'],
                'exit_price': exit_price,
                'pnl': pnl,
                'trade_type': 'SHORT',
                'reason_entry': 'PsychMomentum+Sweep/Breakout',
                'reason_exit': hit,
                'pscore': row['pscore']
            })
        equity_curve.append(eq)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.Series(equity_curve)
    # Compute statistics
    trades_df['win'] = trades_df['pnl'] > 0
    total = len(trades_df)
    wins = trades_df['win'].sum()
    winrate = wins / total
    gross_profit = trades_df[trades_df['pnl']>0]['pnl'].sum()
    gross_loss = trades_df[trades_df['pnl']<=0]['pnl'].sum()
    avg_win = trades_df[trades_df['pnl']>0]['pnl'].mean() if wins>0 else 0
    avg_loss = trades_df[trades_df['pnl']<=0]['pnl'].mean() if (total-wins)>0 else 0

    stats = {
        'total_trades': total,
        'wins': int(wins),
        'winrate': winrate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_equity': eq
    }
    return trades_df, pd.Series(equity_curve), stats

# ---------- Default params ----------
DEFAULT_PARAMS = {
    'momentum_lookback': 3,
    'atr_lookback': 14,
    'vol_lookback': 20,
    'pscore_norm': 50,
    'w_rtn': 1.0,
    'w_vol': 0.6,
    'w_candle': 0.6,
    'sweep_lookback': 10,
    'pscore_entry': 1.2,
    'consolidation': 20,
    'stop_lookback': 5,
    'min_stop_buffer': 0.5,
    'reward_ratio': 2.5,
    'max_holding': 20,
    'start_index': 60
}

# ---------- UI ----------
st.title("PsychSwing — Psychology-based Swing Strategy")
st.markdown("""
This strategy detects **psychological momentum** + **liquidity sweeps (stop-hunts)** and enters at the close of the signal candle.
It is designed as a framework — tweak parameters on your market/instrument.
""")

with st.sidebar:
    st.header("Data & Params")
    ticker = st.text_input("Ticker (yfinance)", value="SPY")
    period = st.selectbox("Period (yfinance)", options=["6mo","1y","2y","5y","10y"], index=2)
    interval = st.selectbox("Interval", options=["1d","1wk","1mo"], index=0)
    # strategy params
    p = DEFAULT_PARAMS.copy()
    p['pscore_entry'] = st.slider("Psych score threshold", 0.5, 3.0, float(p['pscore_entry']), step=0.1)
    p['reward_ratio'] = st.slider("Reward ratio (Target/Risk)", 1.0, 5.0, float(p['reward_ratio']), step=0.1)
    p['risk_per_trade'] = st.slider("Risk per trade (fraction of capital)", 0.005, 0.05, 0.01, step=0.005)
    capital = st.number_input("Starting capital", value=10000)
    run_btn = st.button("Fetch & Run Backtest")

if run_btn:
    try:
        df = fetch_data_yf(ticker, period=period, interval=interval)
        st.write(f"Fetched {len(df)} rows for {ticker}")
        df = compute_indicators(df, p)
        trades_df, eq_series, stats = backtest(df, p, capital=capital, risk_per_trade=p['risk_per_trade'])
        if trades_df.empty:
            st.warning("No trades found with current parameters — try lowering threshold or changing instrument/timeframe.")
        else:
            st.subheader("Backtest Summary")
            st.write(stats)
            st.subheader("Trade Log")
            st.dataframe(trades_df[['entry_dt','entry_price','target','stop','exit_dt','exit_price','pnl','trade_type','reason_entry','reason_exit']])

            # Equity curve
            fig, ax = plt.subplots()
            ax.plot(eq_series)
            ax.set_title('Equity Curve')
            ax.set_ylabel('Equity')
            st.pyplot(fig)

            # Live signal (last candle close)
            last_row = df.iloc[-1]
            st.subheader("Live Recommendation (last candle close)")
            if last_row['pscore'] > p['pscore_entry']:
                entry = last_row['Close']
                sl = min(df['Low'].rolling(p['stop_lookback']).min().iloc[-2], entry - p['min_stop_buffer'] * last_row['atr'])
                target = entry + p['reward_ratio'] * (entry-sl)
                reason = 'PsychMomentum+' + ('Sweep' if last_row['sweep_long'] else 'Breakout')
                st.success(f"LONG: Entry@{entry:.2f} Target@{target:.2f} SL@{sl:.2f} — Reason: {reason}")
            elif last_row['pscore'] < -p['pscore_entry']:
                entry = last_row['Close']
                sl = max(df['High'].rolling(p['stop_lookback']).max().iloc[-2], entry + p['min_stop_buffer'] * last_row['atr'])
                target = entry - p['reward_ratio'] * (sl-entry)
                reason = 'PsychMomentum+' + ('Sweep' if last_row['sweep_short'] else 'Breakout')
                st.error(f"SHORT: Entry@{entry:.2f} Target@{target:.2f} SL@{sl:.2f} — Reason: {reason}")
            else:
                st.info("No strong signal on last candle.")

            st.markdown("### How to use this app\n- Tweak thresholds to fit your instrument/timeframe.\n- Backtest on large out-of-sample periods before trading live.\n- This is a framework — no guarantees. Always use proper risk management.")

    except Exception as e:
        st.error(str(e))

st.markdown("---\n**References & notes:** This app pulls data via yfinance (open-source helper) and uses price+volume+psychology-based rules such as liquidity sweeps/stop-hunts to detect possible big moves. Test thoroughly before trading live.")
