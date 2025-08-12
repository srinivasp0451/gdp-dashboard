import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import product
from datetime import datetime

# =========================
# Strategy & Optimization
# =========================

def compute_indicators(df, short=10, long=50):
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['rsi'] = compute_rsi(df['close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_strategy(df, short_ma, long_ma, rsi_buy, rsi_sell, target_pct, sl_pct, trade_type):
    df = compute_indicators(df.copy(), short_ma, long_ma)
    df['position'] = 0
    in_trade = False
    entry_price = 0
    trades = []

    for i in range(len(df)):
        if not in_trade:
            if trade_type in ["All", "Long"]:
                if df['ema_short'].iloc[i] > df['ema_long'].iloc[i] and df['rsi'].iloc[i] < rsi_buy:
                    entry_price = df['close'].iloc[i]
                    target_price = entry_price * (1 + target_pct / 100)
                    sl_price = entry_price * (1 - sl_pct / 100)
                    in_trade = True
                    trades.append({
                        "entry_date": df.index[i],
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "sl_price": sl_price,
                        "type": "BUY"
                    })
            if trade_type in ["All", "Short"]:
                if df['ema_short'].iloc[i] < df['ema_long'].iloc[i] and df['rsi'].iloc[i] > rsi_sell:
                    entry_price = df['close'].iloc[i]
                    target_price = entry_price * (1 - target_pct / 100)
                    sl_price = entry_price * (1 + sl_pct / 100)
                    in_trade = True
                    trades.append({
                        "entry_date": df.index[i],
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "sl_price": sl_price,
                        "type": "SELL"
                    })
        else:
            last_trade = trades[-1]
            if last_trade['type'] == "BUY":
                if df['high'].iloc[i] >= last_trade['target_price'] or df['low'].iloc[i] <= last_trade['sl_price']:
                    in_trade = False
            elif last_trade['type'] == "SELL":
                if df['low'].iloc[i] <= last_trade['target_price'] or df['high'].iloc[i] >= last_trade['sl_price']:
                    in_trade = False

    net_pnl = sum(
        (t['target_price'] - t['entry_price']) if t['type'] == "BUY" else (t['entry_price'] - t['target_price'])
        for t in trades
    )
    win_trades = sum(
        1 for t in trades
        if (t['type'] == "BUY" and t['target_price'] > t['entry_price']) or
           (t['type'] == "SELL" and t['target_price'] < t['entry_price'])
    )
    accuracy = (win_trades / len(trades)) * 100 if trades else 0

    return {
        "trades": trades,
        "net_pnl": net_pnl,
        "n_trades": len(trades),
        "win_rate": accuracy
    }

def optimize_parameters(df, trade_type):
    best_params = None
    best_score = -np.inf
    for short_ma, long_ma, rsi_buy, rsi_sell, target, sl in product(
        range(5, 20, 5), range(20, 60, 10),
        range(20, 40, 5), range(60, 80, 5),
        [1, 2, 3], [1, 2]
    ):
        results = backtest_strategy(df, short_ma, long_ma, rsi_buy, rsi_sell, target, sl, trade_type)
        score = results['net_pnl']
        if score > best_score:
            best_score = score
            best_params = (short_ma, long_ma, rsi_buy, rsi_sell, target, sl)
    return best_params

# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload Stock CSV", type=["csv"])
    trade_type = st.selectbox("Trade Type", ["All", "Long", "Short"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.set_index('date', inplace=True)

    # Optimize
    params = optimize_parameters(df, trade_type)
    short_ma, long_ma, rsi_buy, rsi_sell, target, sl = params
    results = backtest_strategy(df, short_ma, long_ma, rsi_buy, rsi_sell, target, sl, trade_type)

    st.subheader("Backtest Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Net PnL", f"{results['net_pnl']:.2f}")
    col2.metric("Trades", results['n_trades'])
    col3.metric("Win Rate", f"{results['win_rate']:.1f}%")

    if results['trades']:
        st.subheader("Latest Recommendation")
        last_trade = results['trades'][-1]
        st.table(pd.DataFrame([{
            "Signal": last_trade['type'],
            "Entry Price": last_trade['entry_price'],
            "Target Price": last_trade['target_price'],
            "Stop Loss": last_trade['sl_price'],
            "Confidence": f"{results['win_rate']:.1f}%",
            "Params": f"EMA({short_ma},{long_ma}), RSI({rsi_buy},{rsi_sell}), Target {target}%, SL {sl}%"
        }]))
    else:
        st.info("No trades found with optimized parameters.")
else:
    st.info("Upload a CSV file to start.")
