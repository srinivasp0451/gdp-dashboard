import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------
# Utility: compute ATR
# -------------------------------
def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = ranges.max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# -------------------------------
# Strategy signal generator
# -------------------------------
def generate_signals(df):
    df['ema20'] = df['Close'].ewm(span=20).mean()
    df['ema200'] = df['Close'].ewm(span=200).mean()
    df['atr14'] = compute_atr(df, 14)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()

    signals = []
    reasons = []

    for i in range(len(df)):
        sig = 0
        reason = ""
        if df['Close'].iloc[i] > df['ema200'].iloc[i]:
            if df['Close'].iloc[i] > df['ema20'].iloc[i] and df['Close'].iloc[i-1] < df['ema20'].iloc[i-1]:
                sig = 1
                reason = "Bullish pullback entry"
        elif df['Close'].iloc[i] < df['ema200'].iloc[i]:
            if df['Close'].iloc[i] < df['ema20'].iloc[i] and df['Close'].iloc[i-1] > df['ema20'].iloc[i-1]:
                sig = -1
                reason = "Bearish pullback entry"
        signals.append(sig)
        reasons.append(reason)

    df['signal'] = signals
    df['signal_reason'] = reasons
    return df

# -------------------------------
# Backtest points-only
# -------------------------------
def backtest(df, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    equity_curve = [0]  # start at 0 points

    for i in range(len(df)):
        equity_curve.append(equity_curve[-1])
        sig = int(df['signal'].iloc[i])
        if sig == 0 or i+1 >= len(df):
            continue

        entry_idx = i+1
        entry_price = df['Open'].iloc[entry_idx]
        atr = df['atr14'].iloc[entry_idx]
        if np.isnan(atr) or atr == 0:
            atr = df['Close'].diff().abs().rolling(14).mean().iloc[entry_idx]

        if sig == 1:  # long
            stop = entry_price - atr_mult*atr
            target = entry_price + rr_target*(entry_price-stop)
            exit_price, exit_idx, exit_reason = None, None, ''
            for j in range(entry_idx, min(len(df), entry_idx+max_hold)):
                if df['Low'].iloc[j] <= stop:
                    exit_price, exit_idx, exit_reason = stop, j, 'SL'
                    break
                elif df['High'].iloc[j] >= target:
                    exit_price, exit_idx, exit_reason = target, j, 'TP'
                    break
            if exit_price is None:
                exit_idx = min(len(df)-1, entry_idx+max_hold-1)
                exit_price = df['Close'].iloc[exit_idx]
                exit_reason = 'TimeExit'
            pnl_points = exit_price - entry_price
            equity_curve[-1] += pnl_points
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'LONG',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'exit': exit_price,
                'pnl_points': pnl_points,
                'reason': df['signal_reason'].iloc[i],
                'exit_reason': exit_reason
            })

        elif sig == -1:  # short
            stop = entry_price + atr_mult*atr
            target = entry_price - rr_target*(stop-entry_price)
            exit_price, exit_idx, exit_reason = None, None, ''
            for j in range(entry_idx, min(len(df), entry_idx+max_hold)):
                if df['High'].iloc[j] >= stop:
                    exit_price, exit_idx, exit_reason = stop, j, 'SL'
                    break
                elif df['Low'].iloc[j] <= target:
                    exit_price, exit_idx, exit_reason = target, j, 'TP'
                    break
            if exit_price is None:
                exit_idx = min(len(df)-1, entry_idx+max_hold-1)
                exit_price = df['Close'].iloc[exit_idx]
                exit_reason = 'TimeExit'
            pnl_points = entry_price - exit_price
            equity_curve[-1] += pnl_points
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'SHORT',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'exit': exit_price,
                'pnl_points': pnl_points,
                'reason': df['signal_reason'].iloc[i],
                'exit_reason': exit_reason
            })

    if len(equity_curve) == len(df)+1:
        equity_series = pd.Series(equity_curve[1:], index=df.index)
    else:
        equity_series = pd.Series(equity_curve[-len(df):], index=df.index)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_trades = len(trades_df)
        wins = (trades_df['pnl_points']>0).sum()
        losses = total_trades - wins
        win_rate = wins/total_trades if total_trades else 0
        total_points = trades_df['pnl_points'].sum()
        avg_points = trades_df['pnl_points'].mean()
        cum_max = equity_series.cummax()
        drawdown = (equity_series-cum_max)
        max_dd = drawdown.min()
        # buy and hold
        buy_hold_points = df['Close'].iloc[-1]-df['Close'].iloc[0]
        trades_df.attrs['summary'] = {
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'total_points': float(total_points),
            'avg_points': float(avg_points),
            'max_drawdown': float(max_dd),
            'buy_hold_points': float(buy_hold_points)
        }
    else:
        trades_df.attrs['summary'] = {}

    return trades_df, equity_series

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Swing Trading Strategy (Points-based, with Buy & Hold Comparison)")

uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv","xlsx"])

if uploaded:
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # map columns
    cols = st.multiselect("Map columns (Date,Open,High,Low,Close,Volume)", df.columns.tolist(), default=df.columns.tolist()[:6])
    if len(cols)>=6:
        df = df[cols]
        df.columns = ["Date","Open","High","Low","Close","Volume"]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = generate_signals(df)
        trades_df, equity_series = backtest(df)

        st.subheader("Backtest Results")
        if 'summary' in trades_df.attrs:
            summary = trades_df.attrs['summary']
            st.write(f"**Total Trades:** {summary['total_trades']}")
            st.write(f"**Win Rate:** {summary['win_rate']*100:.2f}%")
            st.write(f"**Total Points:** {summary['total_points']:.2f}")
            st.write(f"**Average Points/Trade:** {summary['avg_points']:.2f}")
            st.write(f"**Max Drawdown (points):** {summary['max_drawdown']:.2f}")
            st.write(f"**Buy & Hold Points:** {summary['buy_hold_points']:.2f}")

        st.subheader("Trade Log")
        st.dataframe(trades_df)

        st.subheader("Equity Curve (Points)")
        fig, ax = plt.subplots(figsize=(10,4))
        equity_series.plot(ax=ax)
        ax.set_title("Equity Curve (Points)")
        st.pyplot(fig)
