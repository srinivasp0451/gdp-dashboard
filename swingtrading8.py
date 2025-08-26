import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
from datetime import timedelta

st.set_page_config(page_title="Swing Trading Strategy App", layout="wide")
st.title("📈 Swing Trading Strategy Optimizer & Live Recommendations")

# ---------------------- FILE UPLOAD ----------------------
file = st.file_uploader("Upload OHLCV data (CSV/XLSX)", type=["csv", "xlsx"])

if file:
    # Read file
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

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

    # Ensure required cols
    required = ["date", "open", "high", "low", "close"]
    for r in required:
        if r not in df.columns:
            st.error(f"Missing column: {r}")
            st.stop()

    # Convert date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---------------------- BASIC STATS ----------------------
    st.subheader("🔍 Data Overview")
    st.write("Top 5 Rows:", df.head())
    st.write("Bottom 5 Rows:", df.tail())
    st.write(f"Date Range: {df['date'].min().date()} → {df['date'].max().date()}")
    st.write(f"Max Price: {df['close'].max():.2f} | Min Price: {df['close'].min():.2f}")

    # Plot raw close
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

    # ---------------------- STRATEGY + BACKTEST ENGINE ----------------------
    def backtest(data, params):
        df_bt = data.copy()

        # Indicators with param tuning
        df_bt["sma"] = SMA(df_bt["close"], params["sma"])
        df_bt["ema"] = EMA(df_bt["close"], params["ema"])
        df_bt["rsi"] = RSI(df_bt["close"], params["rsi"])
        macd, signal = MACD(df_bt["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"])
        df_bt["macd"], df_bt["signal"] = macd, signal
        df_bt["bb_upper"], df_bt["bb_lower"] = Bollinger(df_bt["close"], params["bb_period"], params["bb_mult"])

        trades = []
        position = None

        for i in range(1,len(df_bt)):
            row = df_bt.iloc[i]

            # Entry logic
            if position is None:
                if row["rsi"] < 30 and row["close"] > row["sma"]:
                    position = {"side":"BUY","entry":row["close"],"date":row["date"],"reason":"RSI Oversold + Price>SMA"}
                    position["target"] = row["close"]*1.03
                    position["sl"] = row["close"]*0.98
                elif row["rsi"] > 70 and row["close"] < row["sma"]:
                    position = {"side":"SELL","entry":row["close"],"date":row["date"],"reason":"RSI Overbought + Price<SMA"}
                    position["target"] = row["close"]*0.97
                    position["sl"] = row["close"]*1.02

            # Exit logic
            elif position is not None:
                if position["side"]=="BUY":
                    if row["high"]>=position["target"]:
                        pnl = position["target"]-position["entry"]
                        trades.append({**position,"exit":position["target"],"exit_date":row["date"],"pnl":pnl})
                        position=None
                    elif row["low"]<=position["sl"]:
                        pnl = position["sl"]-position["entry"]
                        trades.append({**position,"exit":position["sl"],"exit_date":row["date"],"pnl":pnl})
                        position=None
                elif position["side"]=="SELL":
                    if row["low"]<=position["target"]:
                        pnl = position["entry"]-position["target"]
                        trades.append({**position,"exit":position["target"],"exit_date":row["date"],"pnl":pnl})
                        position=None
                    elif row["high"]>=position["sl"]:
                        pnl = position["entry"]-position["sl"]
                        trades.append({**position,"exit":position["sl"],"exit_date":row["date"],"pnl":pnl})
                        position=None

        return pd.DataFrame(trades)

    # ---------------------- OPTIMIZATION ----------------------
    param_grid = {
        "sma":[10,20,30],
        "ema":[30,50],
        "rsi":[10,14,20],
        "macd_fast":[8,12],
        "macd_slow":[20,26],
        "macd_signal":[9,12],
        "bb_period":[20],
        "bb_mult":[2,2.5]
    }

    param_sets = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    if search_method=="Random Search":
        param_sets = random.sample(param_sets,min(20,len(param_sets)))

    best_pnl=-1e9
    best_params=None
    best_trades=None

    for ps in param_sets:
        params=dict(zip(keys,ps))
        trades=backtest(df,params)
        pnl=trades["pnl"].sum() if not trades.empty else -1e9
        if pnl>best_pnl:
            best_pnl=pnl
            best_params=params
            best_trades=trades

    # ---------------------- RESULTS ----------------------
    st.subheader("📊 Optimized Backtest Results")
    if best_trades is not None and not best_trades.empty:
        st.write("Best Parameters:", best_params)
        st.write(best_trades)

        total_trades=len(best_trades)
        positive=(best_trades["pnl"]>0).sum()
        negative=(best_trades["pnl"]<0).sum()
        accuracy=positive/total_trades*100 if total_trades>0 else 0
        total_pnl=best_trades["pnl"].sum()

        st.metric("Total Trades", total_trades)
        st.metric("Accuracy %", f"{accuracy:.2f}")
        st.metric("Total PnL", f"{total_pnl:.2f}")

        # ---------------------- BUY & HOLD BENCHMARK ----------------------
        st.subheader("📈 Buy & Hold Benchmark")
        start_price = df["close"].iloc[0]
        end_price = df["close"].iloc[-1]
        buy_hold_return = (end_price - start_price)
        st.write(f"Buy & Hold from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} = {buy_hold_return:.2f}")
        if total_pnl > buy_hold_return:
            st.success("✅ Strategy outperformed Buy & Hold")
        else:
            st.warning("⚠️ Strategy underperformed Buy & Hold")

        # ---------------------- LIVE RECOMMENDATION ----------------------
        st.subheader("🚀 Live Recommendation")
        last=df.iloc[-1]
        tomorrow=last["date"]+timedelta(days=1)
        rec="HOLD"
        reason="No strong signal"
        if last["rsi"]<30 and last["close"]>last["sma"]:
            rec="BUY"
            reason="RSI Oversold + Above SMA"
        elif last["rsi"]>70 and last["close"]<last["sma"]:
            rec="SELL"
            reason="RSI Overbought + Below SMA"
        st.write(f"Date: {tomorrow.date()} | Recommendation: {rec} | Close: {last['close']:.2f} | Reason: {reason}")

        # ---------------------- SUMMARY ----------------------
        st.subheader("📝 Summary")
        st.write(f"This dataset from {df['date'].min().date()} to {df['date'].max().date()} was optimized using {search_method}. ")
        st.write(f"The best strategy uses parameters {best_params} and generated {total_trades} trades with accuracy of {accuracy:.1f}% and total PnL {total_pnl:.2f}.")
        st.write(f"Buy & Hold return was {buy_hold_return:.2f}. The strategy {'outperformed' if total_pnl>buy_hold_return else 'underperformed'} buy & hold.")
        st.write(f"The last candle close at {last['close']:.2f} suggests a {rec} signal for {tomorrow.date()} with reason: {reason}.")
