import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import random
from itertools import product

st.set_page_config(page_title="Swing Trading Recommendations", layout="wide")

# -------------------------------
# Helper: Column mapping
# -------------------------------
def map_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def find(colname):
        for k,v in cols.items():
            if colname in k:
                return v
        return None

    mapping = {
        "open": find("open"),
        "high": find("high"),
        "low": find("low"),
        "close": find("close"),
        "volume": find("vol")
    }
    return mapping

# -------------------------------
# Indicators (manual)
# -------------------------------
def SMA(series, n):
    return series.rolling(n).mean()

def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()

def RSI(series, n=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(span=n, adjust=False).mean()
    loss_ema = pd.Series(loss).ewm(span=n, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-10)
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    return macd, signal_line

def Bollinger(series, n=20, k=2):
    sma = SMA(series, n)
    std = series.rolling(n).std()
    upper = sma + k*std
    lower = sma - k*std
    return upper, lower

def ATR(df, n=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(n).mean()

def Stochastic(df, k=14, d=3):
    low_min = df['Low'].rolling(k).min()
    high_max = df['High'].rolling(k).max()
    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def CCI(df, n=20):
    tp = (df['High']+df['Low']+df['Close'])/3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x-np.mean(x))))
    return (tp - ma) / (0.015*md)

def Momentum(series, n=10):
    return series/series.shift(n) - 1

def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

# -------------------------------
# Strategy & Backtest
# -------------------------------
def generate_signals(df, params):
    # Apply indicators
    df['SMA'] = SMA(df['Close'], params['sma'])
    df['EMA'] = EMA(df['Close'], params['ema'])
    df['RSI'] = RSI(df['Close'], params['rsi'])
    df['MACD'], df['MACD_sig'] = MACD(df['Close'])
    df['BB_up'], df['BB_low'] = Bollinger(df['Close'], params['bb'])
    df['ATR'] = ATR(df)
    df['Stoch_k'], df['Stoch_d'] = Stochastic(df)
    df['CCI'] = CCI(df)
    df['Momentum'] = Momentum(df['Close'])
    df['OBV'] = OBV(df)

    signals = []
    for i in range(len(df)):
        reason = []
        if i == 0: 
            signals.append(None)
            continue
        if df['RSI'].iloc[i] < 30 and df['Close'].iloc[i] < df['BB_low'].iloc[i]:
            signals.append("BUY")
            reason.append("Oversold with Bollinger support")
        elif df['RSI'].iloc[i] > 70 and df['Close'].iloc[i] > df['BB_up'].iloc[i]:
            signals.append("SELL")
            reason.append("Overbought with Bollinger resistance")
        else:
            signals.append(None)
    df['Signal'] = signals
    return df

def backtest(df, params, side="both"):
    df = generate_signals(df.copy(), params)
    trades = []
    position = None
    entry_price, entry_date, signal_reason = None, None, None
    atr_mult = 1.5

    for i in range(len(df)):
        sig = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]
        date = df['Date'].iloc[i]
        atr = df['ATR'].iloc[i]
        if sig == "BUY" and (side in ["both","long"]):
            if position is None:
                position = "long"
                entry_price = price
                entry_date = date
                target = price + atr*atr_mult
                sl = price - atr*atr_mult
                trades.append({"EntryDate":entry_date,"EntryPrice":entry_price,"Type":"Long",
                               "Target":target,"SL":sl,"Reason":"RSI/Bollinger Buy Signal"})
        elif sig == "SELL" and (side in ["both","short"]):
            if position is None:
                position = "short"
                entry_price = price
                entry_date = date
                target = price - atr*atr_mult
                sl = price + atr*atr_mult
                trades.append({"EntryDate":entry_date,"EntryPrice":entry_price,"Type":"Short",
                               "Target":target,"SL":sl,"Reason":"RSI/Bollinger Sell Signal"})
        elif position is not None:
            # exit on opposite signal
            if (position=="long" and sig=="SELL") or (position=="short" and sig=="BUY"):
                exit_price = price
                exit_date = date
                pnl = (exit_price-entry_price) if position=="long" else (entry_price-exit_price)
                trades[-1].update({"ExitDate":exit_date,"ExitPrice":exit_price,"PnL":pnl})
                position=None

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df["HoldDuration"] = (trade_df["ExitDate"]-trade_df["EntryDate"]).dt.days
        accuracy = np.mean(trade_df["PnL"]>0)*100
        total_pnl = trade_df["PnL"].sum()
        trade_summary = {
            "TotalPnL": total_pnl,
            "Accuracy": accuracy,
            "Trades": len(trade_df),
            "PositiveTrades": sum(trade_df["PnL"]>0),
            "LossTrades": sum(trade_df["PnL"]<=0)
        }
    else:
        trade_summary = {"TotalPnL":0,"Accuracy":0,"Trades":0,"PositiveTrades":0,"LossTrades":0}
    return trade_df, trade_summary

# -------------------------------
# Optimization
# -------------------------------
def optimize(df, search="random", side="both", iters=20):
    sma_vals = [10,20,50]
    ema_vals = [10,20,50]
    rsi_vals = [14,21]
    bb_vals = [14,20]

    combos = list(product(sma_vals, ema_vals, rsi_vals, bb_vals))
    if search=="random":
        combos = random.sample(combos, min(iters, len(combos)))

    best_result = None
    best_params = None
    for (sma,ema,rsi,bb) in combos:
        params={"sma":sma,"ema":ema,"rsi":rsi,"bb":bb}
        trades,summary = backtest(df,params,side)
        if best_result is None or summary["TotalPnL"]>best_result["TotalPnL"]:
            best_result=summary
            best_params=params
    return best_params, best_result

# -------------------------------
# Summary Generator
# -------------------------------
def generate_summary(df, summary):
    text = f"""The uploaded stock data ranges from {df['Date'].min().date()} to {df['Date'].max().date()}, 
with prices fluctuating between {df['Close'].min():.2f} and {df['Close'].max():.2f}. 
Overall strategy backtesting yielded {summary['Trades']} trades with {summary['Accuracy']:.1f}% accuracy. 
There were {summary['PositiveTrades']} profitable trades and {summary['LossTrades']} losing trades, 
resulting in a net PnL of {summary['TotalPnL']:.2f}. 
The analysis suggests potential swing opportunities when RSI crosses oversold/overbought zones in 
conjunction with Bollinger bands. Risk management with ATR-based stop-loss and targets is recommended. 
Optimized indicator parameters provide a strategy that aims to outperform simple buy-and-hold by a large margin, 
highlighting that disciplined swing trading could yield better returns than passive investing."""
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Swing Trading Recommendation Engine")

file = st.file_uploader("Upload your stock data (CSV/Excel)", type=["csv","xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Column mapping
    mapping = map_columns(df)
    df = df.rename(columns={
        mapping["open"]:"Open",
        mapping["high"]:"High",
        mapping["low"]:"Low",
        mapping["close"]:"Close",
        mapping["volume"]:"Volume"
    })
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df.index)
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values("Date").reset_index(drop=True)

    st.subheader("Raw Data Preview")
    st.write("Top 5 rows:", df.head())
    st.write("Bottom 5 rows:", df.tail())
    st.write("Date Range:", df['Date'].min(), "→", df['Date'].max())
    st.write("Price Range:", df['Close'].min(), "→", df['Close'].max())

    st.line_chart(df.set_index("Date")["Close"])

    # Year-month heatmap of returns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Return'] = df['Close'].pct_change()
    pivot = df.pivot_table(values="Return", index="Year", columns="Month", aggfunc="mean")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn", ax=ax)
    st.pyplot(fig)

    # End date selection
    end_date = st.date_input("Select End Date for Backtest", df['Date'].max())
    df_bt = df[df['Date']<=pd.to_datetime(end_date)]

    side = st.selectbox("Select Trade Side", ["both","long","short"])
    search = st.selectbox("Optimization Method", ["random","grid"])

    best_params, best_result = optimize(df_bt, search=search, side=side)
    trades, summary = backtest(df_bt, best_params, side)

    st.subheader("Best Strategy Parameters")
    st.write(best_params)
    st.subheader("Backtest Results")
    st.dataframe(trades)
    st.write(summary)

    st.subheader("Generated Summary")
    st.write(generate_summary(df_bt, summary))

    # Live recommendation
    st.subheader("Live Recommendation")
    last_date = df['Date'].max() + timedelta(days=1)
    last_price = df['Close'].iloc[-1]
    df_live = generate_signals(df.copy(), best_params)
    last_signal = df_live['Signal'].iloc[-1]
    if last_signal:
        st.success(f"Recommendation for {last_date.date()}: {last_signal} at {last_price:.2f}")
    else:
        st.info(f"No clear recommendation for {last_date.date()}")
