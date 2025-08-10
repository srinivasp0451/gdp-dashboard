import pandas as pd
import numpy as np
import itertools
import streamlit as st

# ================= INDICATOR FUNCTIONS =================
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA_TREND'] = df['Close'].ewm(span=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['ATR'] = compute_atr(df, 14)
    df['Vol_Surge'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_atr(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ================= BACKTEST =================
def run_backtest(df, params):
    trades = []
    gross_profit = 0
    gross_loss = 0

    for i in range(len(df)):
        row = df.iloc[i]
        long_conf, short_conf = [], []

        if row['SMA20'] > row['SMA50']: long_conf.append("SMA Bullish")
        if 30 < row['RSI'] < 70: long_conf.append("RSI Healthy")
        if row['MACD'] > row['Signal']: long_conf.append("MACD Bullish")
        if row['Close'] > row['BB_Mid']: long_conf.append("BB Breakout")
        if row['Vol_Surge']: long_conf.append("Volume Surge")

        if row['SMA20'] < row['SMA50']: short_conf.append("SMA Bearish")
        if 30 < row['RSI'] < 70: short_conf.append("RSI Healthy")
        if row['MACD'] < row['Signal']: short_conf.append("MACD Bearish")
        if row['Close'] < row['BB_Mid']: short_conf.append("BB Breakdown")
        if row['Vol_Surge']: short_conf.append("Volume Surge")

        trade_mode = params["trade_mode"]
        min_conf = params["min_conf"]
        atr_sl = params["atr_sl"]
        atr_tp = params["atr_tp"]

        if row['Close'] > row['EMA_TREND'] and len(long_conf) >= min_conf and trade_mode in ["Both", "Long Only"]:
            sl = row['Close'] - atr_sl * row['ATR']
            tp = row['Close'] + atr_tp * row['ATR']
            trades.append({"Type": "Long", "Entry": row['Close'], "SL": sl, "TP": tp})

        elif row['Close'] < row['EMA_TREND'] and len(short_conf) >= min_conf and trade_mode in ["Both", "Short Only"]:
            sl = row['Close'] + atr_sl * row['ATR']
            tp = row['Close'] - atr_tp * row['ATR']
            trades.append({"Type": "Short", "Entry": row['Close'], "SL": sl, "TP": tp})

    for t in trades:
        entry_index = df.index[df['Close'] == t["Entry"]]
        if entry_index.empty: continue
        idx = entry_index[0]

        for j in range(idx+1, min(idx+6, len(df))):
            high, low = df.iloc[j]['High'], df.iloc[j]['Low']
            if t["Type"] == "Long":
                if high >= t["TP"]:
                    gross_profit += (t["TP"] - t["Entry"])
                    break
                elif low <= t["SL"]:
                    gross_loss += (t["SL"] - t["Entry"])
                    break
            else:
                if low <= t["TP"]:
                    gross_profit += (t["Entry"] - t["TP"])
                    break
                elif high >= t["SL"]:
                    gross_loss += (t["Entry"] - t["SL"])
                    break

    profit_factor = None
    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)

    return pd.DataFrame(trades), profit_factor

# ================= OPTIMIZATION =================
def run_optimization(train_df):
    atr_sl_vals = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    atr_tp_vals = [1.5, 2.0, 2.5, 3.0, 3.5]
    ema_vals = [50, 100, 150, 200, 250]
    min_conf_vals = [1, 2, 3, 4]
    trade_modes = ["Both", "Long Only", "Short Only"]

    best_pf = -np.inf
    best_params = None

    for atr_sl, atr_tp, ema, min_conf, trade_mode in itertools.product(
        atr_sl_vals, atr_tp_vals, ema_vals, min_conf_vals, trade_modes):

        params = {"atr_sl": atr_sl, "atr_tp": atr_tp, "ema_period": ema,
                  "min_conf": min_conf, "trade_mode": trade_mode}
        trades, pf = run_backtest(train_df, params)
        if trades.empty or pf is None:
            continue
        if pf > best_pf:
            best_pf = pf
            best_params = params

    return best_params

# ================= WALK FORWARD =================
def walk_forward_optimization(df, train_size, test_size, step_size):
    results = []
    for start in range(0, len(df) - train_size - test_size, step_size):
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]
        if train_df.empty or test_df.empty: continue
        best_params = run_optimization(train_df)
        trades, pf = run_backtest(test_df, best_params)
        if trades.empty: continue
        trades["ProfitFactor"] = pf
        results.append(trades)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ================= CONTINUATION CHECK =================
def check_continuation(df, last_params):
    if df.shape[0] < 2: return "Not enough data"
    prev_day = df.iloc[-2]
    today = df.iloc[-1]
    prev_trade, _ = run_backtest(df.iloc[-2:-1], last_params)
    if prev_trade.empty: return "No previous trade to continue"

    trade = prev_trade.iloc[0]
    if trade["Type"] == "Long":
        if today["Low"] > trade["SL"] and today["High"] < trade["TP"]:
            return "Long trade still active"
    elif trade["Type"] == "Short":
        if today["High"] < trade["SL"] and today["Low"] > trade["TP"]:
            return "Short trade still active"
    return "No continuation"

# ================= STREAMLIT APP =================
st.title("Walk-Forward ATR Strategy Optimizer")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df = add_indicators(df.dropna().reset_index(drop=True))

    max_len = len(df) - 1
    train_window = st.slider("Training window size", min_value=10, max_value=max_len, value=min(75, max_len))
    test_window = st.slider("Testing window size", min_value=5, max_value=max_len, value=min(20, max_len))
    step_size = st.slider("Step size", min_value=1, max_value=max_len, value=min(5, max_len))

    if st.button("Run Walk-Forward Optimization"):
        wf_results = walk_forward_optimization(df, train_window, test_window, step_size)
        st.write(wf_results)

    if st.button("Get Live Recommendation"):
        last_params = run_optimization(df.iloc[-train_window:])
        trades, pf = run_backtest(df, last_params)
        st.write("Best Params:", last_params)
        st.write("Live Trades:", trades.tail(5))
        st.write("Continuation Status:", check_continuation(df, last_params))
