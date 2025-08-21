import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

##############################
# LOAD AND CLEAN DATA
##############################
def load_data(file):
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df = df.copy()
    # Handle flexible column names
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if "date" in c: rename_map[col] = "Date"
        elif "open" in c: rename_map[col] = "Open"
        elif "high" in c: rename_map[col] = "High"
        elif "low" in c: rename_map[col] = "Low"
        elif "close" in c: rename_map[col] = "Close"
        elif "volume" in c: rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").dropna()
    df.reset_index(drop=True, inplace=True)
    return df

##############################
# INDICATORS (MANUAL)
##############################
def SMA(series, period):
    return series.rolling(period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    macd_line = EMA(series, fast) - EMA(series, slow)
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def Bollinger(series, period=20, stddev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + stddev * std
    lower = mid - stddev * std
    return upper, mid, lower

def ATR(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def Stochastic(df, k=14, d=3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    k_percent = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent

def CCI(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

##############################
# STRATEGY BACKTEST
##############################
def backtest(df, params):
    trades = []
    position = 0
    entry_price, sl, target = 0, 0, 0
    returns = []
    
    # Rules
    for i in range(max(params["ema_fast"], params["ema_slow"]), len(df)):
        price = df["Close"].iloc[i]
        signal = 0

        # ===== ENTRY LOGIC based on hybrid indicators =====
        if position == 0:
            trend_up = df["EMA_fast"].iloc[i] > df["EMA_slow"].iloc[i]
            rsi = df["RSI"].iloc[i]
            close = df["Close"].iloc[i]

            if trend_up and rsi < params["rsi_buy"] and close < df["BB_lower"].iloc[i]:
                signal = 1   # Long entry
                entry_price = price
                sl = entry_price * (1 - params["sl"])
                target = entry_price * (1 + params["tp"])
                position = 1
                trades.append({"EntryDate": df["Date"].iloc[i],
                               "Type": "LONG", "Entry": entry_price})
            elif not trend_up and rsi > params["rsi_sell"] and close > df["BB_upper"].iloc[i]:
                signal = -1  # Short entry
                entry_price = price
                sl = entry_price * (1 + params["sl"])
                target = entry_price * (1 - params["tp"])
                position = -1
                trades.append({"EntryDate": df["Date"].iloc[i],
                               "Type": "SHORT", "Entry": entry_price})

        elif position == 1:  # Manage Long
            if price <= sl or price >= target or price >= df["BB_mid"].iloc[i]:
                pnl = price - entry_price
                trades[-1].update({"ExitDate": df["Date"].iloc[i],
                                   "Exit": price, "PnL": pnl})
                returns.append(pnl/entry_price)
                position = 0

        elif position == -1: # Manage Short
            if price >= sl or price <= target or price <= df["BB_mid"].iloc[i]:
                pnl = entry_price - price
                trades[-1].update({"ExitDate": df["Date"].iloc[i],
                                   "Exit": price, "PnL": pnl})
                returns.append(pnl/entry_price)
                position = 0

    return trades, np.mean(returns) if returns else 0, np.sum(returns) if returns else 0

##############################
# OPTIMIZATION
##############################
def optimize_strategy(df):
    # Indicator calculations
    df["EMA_fast"] = EMA(df["Close"], 20)
    df["EMA_slow"] = EMA(df["Close"], 200)
    df["RSI"] = RSI(df["Close"], 14)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = Bollinger(df["Close"])
    df["ATR"] = ATR(df)
    df["OBV"] = OBV(df)
    df["CCI"] = CCI(df)

    # Parameter grid
    param_grid = {
        "ema_fast": [10, 20, 50],
        "ema_slow": [100, 150, 200],
        "rsi_buy": [25, 30, 35],
        "rsi_sell": [65, 70, 75],
        "sl": [0.01, 0.02],
        "tp": [0.02, 0.03, 0.05]
    }

    best_return = -1e9
    best_trades = []
    best_params = None

    for ema_f, ema_s, rsi_b, rsi_s, sl, tp in itertools.product(
        param_grid["ema_fast"], param_grid["ema_slow"],
        param_grid["rsi_buy"], param_grid["rsi_sell"],
        param_grid["sl"], param_grid["tp"]):

        params = {"ema_fast": ema_f, "ema_slow": ema_s,
                  "rsi_buy": rsi_b, "rsi_sell": rsi_s,
                  "sl": sl, "tp": tp}

        trades, avg_ret, total_ret = backtest(df, params)
        if total_ret > best_return:
            best_return = total_ret
            best_trades = trades
            best_params = params

    return best_trades, best_params, best_return

##############################
# STREAMLIT UI
##############################
st.title("📊 Ultra-Robust Swing Trading Strategy Optimizer")
file = st.file_uploader("Upload CSV or Excel (OHLCV format)", type=["csv","xlsx"])

if file:
    df = load_data(file)
    st.write("✅ Data Loaded:", df.head())

    trades, best_params, best_return = optimize_strategy(df)
    st.subheader("Best Strategy Parameters")
    st.json(best_params)

    st.subheader("Backtest Trades")
    st.write(pd.DataFrame(trades))

    st.subheader("Performance")
    st.write(f"Total Backtest Return: {round(best_return*100,2)} %")

    # Live Recommendation using last candle
    last_row = df.iloc[-1]
    signal_df = pd.DataFrame([last_row])
    signal_df["EMA_fast"] = EMA(df["Close"], best_params["ema_fast"]).iloc[-1]
    signal_df["EMA_slow"] = EMA(df["Close"], best_params["ema_slow"]).iloc[-1]
    signal_df["RSI"] = RSI(df["Close"],14).iloc[-1]
    signal_df["BB_upper"], signal_df["BB_mid"], signal_df["BB_lower"] = Bollinger(df["Close"])
    signal_df["BB_upper"] = signal_df["BB_upper"].iloc[-1]
    signal_df["BB_mid"] = signal_df["BB_mid"].iloc[-1]
    signal_df["BB_lower"] = signal_df["BB_lower"].iloc[-1]

    live_signal = None
    entry_price = last_row["Close"]
    if signal_df["EMA_fast"].iloc[0] > signal_df["EMA_slow"].iloc and signal_df["RSI"].iloc < best_params["rsi_buy"] and entry_price < signal_df["BB_lower"].iloc:
        live_signal = "BUY"
        sl = entry_price*(1-best_params["sl"])
        tp = entry_price*(1+best_params["tp"])
    elif signal_df["EMA_fast"].iloc < signal_df["EMA_slow"].iloc and signal_df["RSI"].iloc > best_params["rsi_sell"] and entry_price > signal_df["BB_upper"].iloc:
        live_signal = "SELL"
        sl = entry_price*(1+best_params["sl"])
        tp = entry_price*(1-best_params["tp"])

    st.subheader("Live Recommendation (Based on Last Candle Close)")
    if live_signal:
        st.write(f"Signal: **{live_signal}** | Entry: {entry_price:.2f} | SL: {sl:.2f} | Target: {tp:.2f}")
    else:
        st.write("No trade signal today.")
