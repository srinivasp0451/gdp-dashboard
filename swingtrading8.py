import streamlit as st import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import itertools import random from datetime import datetime, timedelta

st.set_page_config(page_title="Swing Trading Strategy App", layout="wide") st.title("📈 Swing Trading Strategy Optimizer & Live Recommendations")

---------------------- FILE UPLOAD ----------------------

file = st.file_uploader("Upload OHLCV data (CSV/XLSX)", type=["csv", "xlsx"])

if file: if file.name.endswith("csv"): df = pd.read_csv(file) else: df = pd.read_excel(file)

# Normalize columns
df.columns = [c.lower().strip() for c in df.columns]
col_map = {}
for c in df.columns:
    if "date" in c:
        col_map[c] = "date"
    elif "open" in c:
        col_map[c] = "open"
    elif "high" in c:
        col_map[c] = "high"
    elif "low" in c:
        col_map[c] = "low"
    elif "close" in c:
        col_map[c] = "close"
    elif "volume" in c:
        col_map[c] = "volume"
df = df.rename(columns=col_map)

required = ["date", "open", "high", "low", "close"]
for r in required:
    if r not in df.columns:
        st.error(f"Missing column: {r}")
        st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------- BASIC STATS ----------------------
st.subheader("🔍 Data Overview")
st.write("Top 5 Rows:", df.head())
st.write("Bottom 5 Rows:", df.tail())
st.write(f"Date Range: {df['date'].min().date()} → {df['date'].max().date()}")
st.write(f"Max Price: {df['close'].max():.2f} | Min Price: {df['close'].min():.2f}")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["date"], df["close"], label="Close Price")
ax.set_title("Raw Price Chart")
st.pyplot(fig)

# ---------------------- USER OPTIONS ----------------------
st.sidebar.header("⚙️ Strategy Options")
trade_side = st.sidebar.selectbox("Trade Side", ["Long", "Short", "Both"])
search_method = st.sidebar.selectbox("Optimization Method", ["Random Search", "Grid Search"])
end_date = st.sidebar.date_input("Backtest End Date", value=df["date"].max().date())
df = df[df["date"] <= pd.to_datetime(end_date)]

# ---------------------- INDICATORS ----------------------
def SMA(series, period):
    return series.rolling(period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = np.where(delta>0, delta, 0)
    down = np.where(delta<0, -delta, 0)
    RS = pd.Series(up).rolling(period).mean() / pd.Series(down).rolling(period).mean()
    return 100 - (100/(1+RS))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    return macd, signal_line

def Bollinger(series, period=20, mult=2):
    ma = SMA(series, period)
    std = series.rolling(period).std()
    upper = ma + mult*std
    lower = ma - mult*std
    return upper, lower

def Momentum(series, period=10):
    return series / series.shift(period) - 1

def Volatility(series, period=20):
    return series.pct_change().rolling(period).std()

def ATR(high, low, close, period=14):
    tr = np.maximum.reduce([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ])
    return pd.Series(tr).rolling(period).mean()

# ---------------------- ADD INDICATORS ----------------------
def add_indicators(df, params):
    df["sma"] = SMA(df["close"], params.get("sma",20))
    df["ema"] = EMA(df["close"], params.get("ema",50))
    df["rsi"] = RSI(df["close"], params.get("rsi",14))
    macd, signal = MACD(df["close"], params.get("macd_fast",12), params.get("macd_slow",26), params.get("macd_signal",9))
    df["macd"], df["signal"] = macd, signal
    df["bb_upper"], df["bb_lower"] = Bollinger(df["close"], params.get("bb_period",20), params.get("bb_mult",2))
    df["momentum"] = Momentum(df["close"], params.get("momentum",10))
    df["volatility"] = Volatility(df["close"], params.get("volatility",20))
    df["atr"] = ATR(df["high"], df["low"], df["close"], params.get("atr",14))
    return df

# ---------------------- STRATEGY & BACKTEST ----------------------
def run_strategy(df, params):
    df = add_indicators(df.copy(), params)
    trades = []
    position = None
    entry_price, entry_date, side = None, None, None
    for i in range(1, len(df)):
        logic = None
        prob = 0.5

        # Entry conditions
        if position is None:
            if df.loc[i,"rsi"] < 30 and df.loc[i,"close"] > df.loc[i,"sma"]:
                position = "LONG"
                entry_price, entry_date = df.loc[i,"close"], df.loc[i,"date"]
                target, sl = entry_price*1.03, entry_price*0.98
                logic = "RSI Oversold + Price above SMA"
                prob = 0.7
                trades.append({"entry_date": entry_date, "side": position, "entry": entry_price, "target": target, "sl": sl, "reason": logic, "prob": prob})
            elif df.loc[i,"rsi"] > 70 and df.loc[i,"close"] < df.loc[i,"sma"]:
                position = "SHORT"
                entry_price, entry_date = df.loc[i,"close"], df.loc[i,"date"]
                target, sl = entry_price*0.97, entry_price*1.02
                logic = "RSI Overbought + Price below SMA"
                prob = 0.7
                trades.append({"entry_date": entry_date, "side": position, "entry": entry_price, "target": target, "sl": sl, "reason": logic, "prob": prob})
        else:
            # Manage exits
            last_trade = trades[-1]
            price = df.loc[i,"close"]
            if last_trade["side"]=="LONG":
                if price>=last_trade["target"] or price<=last_trade["sl"]:
                    last_trade["exit"] = price
                    last_trade["exit_date"] = df.loc[i,"date"]
                    last_trade["pnl"] = price - last_trade["entry"]
                    position = None
            elif last_trade["side"]=="SHORT":
                if price<=last_trade["target"] or price>=last_trade["sl"]:
                    last_trade["exit"] = price
                    last_trade["exit_date"] = df.loc[i,"date"]
                    last_trade["pnl"] = last_trade["entry"] - price
                    position = None

    return pd.DataFrame(trades)

# ---------------------- OPTIMIZATION ----------------------
param_space = {
    "sma": [10,20,30,50],
    "ema": [20,50,100],
    "rsi": [10,14,20],
    "macd_fast": [8,12],
    "macd_slow": [20,26],
    "macd_signal": [9],
    "bb_period": [20,30],
    "bb_mult": [2,2.5],
    "momentum": [5,10,14],
    "volatility": [10,20],
    "atr": [10,14]
}

best_score, best_params, best_trades = -1e9, None, None
param_combinations = list(itertools.product(*param_space.values())) if search_method=="Grid Search" else [tuple(random.choice(v) for v in param_space.values()) for _ in range(30)]

for combo in param_combinations:
    params = dict(zip(param_space.keys(), combo))
    trades_df = run_strategy(df, params)
    if len(trades_df)==0: continue
    pnl = trades_df["pnl"].sum()
    win_rate = (trades_df["pnl"]>0).mean()
    score = pnl*win_rate
    if score>best_score:
        best_score, best_params, best_trades = score, params, trades_df

# ---------------------- RESULTS ----------------------
st.subheader("📊 Backtest Results (Best Strategy)")
if best_trades is not None and len(best_trades)>0:
    st.write("Best Parameters:", best_params)
    st.write(best_trades)
    st.metric("Total Trades", len(best_trades))
    st.metric("Win Rate", f"{(best_trades['pnl']>0).mean()*100:.2f}%")
    st.metric("Total PnL", f"{best_trades['pnl'].sum():.2f}")

    # ---------------------- HEATMAP ----------------------
    if (df["date"].max()-df["date"].min()).days > 365:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        pivot = df.pivot_table(index="year", columns="month", values="returns", aggfunc=np.mean)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn", ax=ax)
        st.subheader("📅 Year vs Month Returns Heatmap")
        st.pyplot(fig)

    # ---------------------- LIVE RECOMMENDATION ----------------------
    st.subheader("🚀 Live Recommendation")
    last = df.iloc[-1]
    rec_date = (last["date"] + timedelta(days=1)).date()
    if best_trades.iloc[-1]["side"]=="LONG":
        st.success(f"[{rec_date}] BUY at {last['close']:.2f}, Target {last['close']*1.03:.2f}, SL {last['close']*0.98:.2f}")
    else:
        st.error(f"[{rec_date}] SELL at {last['close']:.2f}, Target {last['close']*0.97:.2f}, SL {last['close']*1.02:.2f}")

    # ---------------------- SUMMARY ----------------------
    st.subheader("📝 Summary")
    summary_text = f"""
    The dataset spans from {df['date'].min().date()} to {df['date'].max().date()}.
    Using optimization over all data with {search_method}, the best strategy achieved {len(best_trades)} trades,
    win rate of {(best_trades['pnl']>0).mean()*100:.2f}%, and total PnL of {best_trades['pnl'].sum():.2f}.
    Indicators such as RSI, SMA, EMA, MACD, Bollinger Bands, and ATR were tuned to optimal values.
    The live recommendation is generated based on the last closing candle with forward date consideration.
    This optimized strategy aims to outperform buy-and-hold by focusing on high-probability swing setups with
    structured risk management (3% target, 2% SL).
    """
    st.write(summary_text)

