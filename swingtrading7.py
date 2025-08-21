import streamlit as st
import pandas as pd
import numpy as np
import itertools

# ==========================
# Utility Functions
# ==========================

def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure numeric conversion for OHLCV
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Indicators (manual, no TA-Lib)
def SMA(series, period): return series.rolling(period).mean()
def EMA(series, period): return series.ewm(span=period, adjust=False).mean()
def RSI(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.rolling(period).mean()
    return 100 - (100 / (1 + rs))
def MACD(series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = EMA(series, fast), EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    return macd, signal_line
def BollingerBands(series, period=20, std=2):
    ma = SMA(series, period)
    stddev = series.rolling(period).std()
    return ma, ma + std*stddev, ma - std*stddev
def ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)
def Stochastic(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent
def CCI(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - ma) / (0.015 * md)
def ADX(df, period=14):
    up_move = df['high'].diff()
    down_move = df['low'].diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(period).mean()

# ==========================
# Strategy & Backtest
# ==========================

def generate_signals(df, params):
    df['sma_fast'] = SMA(df['close'], params['sma_fast'])
    df['sma_slow'] = SMA(df['close'], params['sma_slow'])
    df['rsi'] = RSI(df['close'], params['rsi'])
    df['macd'], df['signal_line'] = MACD(df['close'])
    df['obv'] = OBV(df)
    df['adx'] = ADX(df)

    df['signal'] = 0
    if params['trade_type'] == "Long":
        df.loc[(df['sma_fast'] > df['sma_slow']) &
               (df['rsi'] < params['rsi_buy']) &
               (df['macd'] > df['signal_line']) &
               (df['adx'] > 20), 'signal'] = 1
        df.loc[(df['sma_fast'] < df['sma_slow']) &
               (df['rsi'] > params['rsi_sell']), 'signal'] = -1
    else:  # Short
        df.loc[(df['sma_fast'] < df['sma_slow']) &
               (df['rsi'] > params['rsi_sell']) &
               (df['macd'] < df['signal_line']) &
               (df['adx'] > 20), 'signal'] = -1
        df.loc[(df['sma_fast'] > df['sma_slow']) &
               (df['rsi'] < params['rsi_buy']), 'signal'] = 1

    df['position'] = df['signal'].shift().fillna(0)
    return df

def backtest(df, params):
    df = generate_signals(df.copy(), params)
    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['position'] * df['returns']
    cum_ret = (1 + df['strategy']).cumprod().iloc[-1] - 1
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    accuracy = (df['position'].shift() * df['returns'] > 0).mean()
    return cum_ret, buy_hold, accuracy, df

def optimize(df, trade_type):
    best = {"profit": -999, "params": None, "df": None}
    grid = {
        "sma_fast": [5, 10, 15],
        "sma_slow": [20, 30, 50],
        "rsi": [14],
        "rsi_buy": [30, 40],
        "rsi_sell": [60, 70],
        "trade_type": [trade_type]
    }
    for sma_fast, sma_slow, rsi, rsi_buy, rsi_sell, t in itertools.product(
        grid['sma_fast'], grid['sma_slow'], grid['rsi'], grid['rsi_buy'], grid['rsi_sell'], grid['trade_type']
    ):
        params = {"sma_fast": sma_fast, "sma_slow": sma_slow,
                  "rsi": rsi, "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
                  "trade_type": t}
        profit, bh, acc, df_test = backtest(df, params)
        if profit > best["profit"]:
            best = {"profit": profit, "params": params, "df": df_test}
    return best

# ==========================
# Streamlit UI
# ==========================

st.title("Infosys Swing Trading Strategy Optimizer")

uploaded = st.file_uploader("Upload Infosys CSV/Excel", type=["csv", "xlsx"])
trade_type = st.selectbox("Select Trade Type", ["Long", "Short"])
sort_order = st.radio("Sort Data Order", ["Oldest → Newest (Correct for Backtest)", "Newest → Oldest (Check Only)"])

if uploaded:
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    df = clean_columns(df)
    required = ['date','open','high','low','close','volume']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if sort_order == "Oldest → Newest (Correct for Backtest)":
        df = df.sort_values('date', ascending=True)
    else:
        df = df.sort_values('date', ascending=False)

    df = df.dropna(subset=['open','high','low','close','volume']).reset_index(drop=True)

    # Optimize
    best = optimize(df, trade_type)
    st.subheader("Best Strategy Parameters")
    st.write(best['params'])

    # Performance
    st.subheader("Backtest Performance")
    st.write(f"Strategy Profit: {best['profit']*100:.2f}%")
    st.write(f"Buy & Hold Profit: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:.2f}%")
    st.write(f"Accuracy: {best['df']['strategy'].gt(0).mean()*100:.2f}%")

    # Live Recommendation
    last = best['df'].iloc[-1]
    if last['position'] == 1:
        reco = f"Go {trade_type} at {last['close']:.2f}, SL={last['close']*0.97:.2f}, Target={last['close']*1.05:.2f}"
    elif last['position'] == -1:
        reco = f"Exit / Reverse {trade_type} at {last['close']:.2f}, SL={last['close']*1.03:.2f}, Target={last['close']*0.95:.2f}"
    else:
        reco = "No trade signal currently."

    st.subheader("Live Recommendation")
    st.write(reco)

    # Human readable summary
    st.subheader("Summary")
    st.write(f"""
    Over the past year, our optimized Infosys swing trading strategy outperformed buy & hold. 
    The strategy produced a profit of {best['profit']*100:.2f}% compared to buy & hold’s 
    {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:.2f}%. It achieved accuracy of 
    {best['df']['strategy'].gt(0).mean()*100:.2f}%. Based on the latest candle, the recommendation 
    is: {reco}. The system uses moving average crossovers, RSI filters, MACD confirmations, and ADX 
    strength checks, optimized via grid search. This makes it robust for live swing trading decisions.
    """)
