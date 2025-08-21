import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

# ======================
# Data Preprocessing
# ======================
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Standardize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Try mapping columns
    col_map = {}
    for key in ["date", "open", "high", "low", "close", "volume"]:
        for c in df.columns:
            if key in c:
                col_map[key] = c
                break

    # Mandatory fields
    df = df.rename(columns=col_map)

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    return df

# ======================
# Indicator Calculations
# ======================
def SMA(series, period):
    return series.rolling(period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    RS = roll_up / roll_down
    return 100 - (100 / (1 + RS))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def Bollinger(series, period=20, dev=2):
    ma = SMA(series, period)
    std = series.rolling(period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    return upper, ma, lower

def ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def OBV(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

# ======================
# Backtesting Core
# ======================
def backtest(df, params):
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']

    # Indicators
    df['ema_fast'] = EMA(close, params['ema_fast'])
    df['ema_slow'] = EMA(close, params['ema_slow'])
    df['rsi'] = RSI(close, params['rsi_period'])
    df['bb_up'], df['bb_mid'], df['bb_low'] = Bollinger(close, params['bb_period'], params['bb_dev'])
    df['macd'], df['macd_signal'], df['macd_hist'] = MACD(close, params['macd_fast'], params['macd_slow'], params['macd_signal'])
    df['atr'] = ATR(df, 14)

    position, entry_price = 0, 0
    trades = []

    for i in range(len(df)-1):
        # Entry signals
        long_cond = (df['ema_fast'][i] > df['ema_slow'][i]) and (df['rsi'][i] < params['rsi_buy']) and (df['close'][i] < df['bb_low'][i])
        short_cond = (df['ema_fast'][i] < df['ema_slow'][i]) and (df['rsi'][i] > params['rsi_sell']) and (df['close'][i] > df['bb_up'][i])

        if position == 0:
            if long_cond:
                position = 1
                entry_price = df['open'][i+1]
                trades.append({"entry_date": df['date'][i+1], "type": "LONG", "entry": entry_price})
            elif short_cond:
                position = -1
                entry_price = df['open'][i+1]
                trades.append({"entry_date": df['date'][i+1], "type": "SHORT", "entry": entry_price})

        elif position != 0:
            stop = entry_price * (1 - params['sl']*position)
            target = entry_price * (1 + params['tp']*position)
            trail_sl = entry_price

            exit_flag, exit_price = False, df['close'][i]

            if position == 1:  # Long
                trail_sl = max(trail_sl, df['close'][i] - df['atr'][i])
                if df['low'][i] <= stop: exit_flag, exit_price = True, stop
                elif df['high'][i] >= target: exit_flag, exit_price = True, target
                elif df['close'][i] >= df['bb_mid'][i]: exit_flag, exit_price = True, df['close'][i]

            if position == -1:  # Short
                trail_sl = min(trail_sl, df['close'][i] + df['atr'][i])
                if df['high'][i] >= stop: exit_flag, exit_price = True, stop
                elif df['low'][i] <= target: exit_flag, exit_price = True, target
                elif df['close'][i] <= df['bb_mid'][i]: exit_flag, exit_price = True, df['close'][i]

            if exit_flag:
                trades[-1].update({
                    "exit_date": df['date'][i],
                    "exit": exit_price,
                    "pnl": round((exit_price-entry_price)*position,2)
                })
                position = 0

    return pd.DataFrame(trades)

def evaluate(trades):
    if trades.empty: return None
    trades['return_%'] = trades['pnl'] / trades['entry'] * 100
    win_rate = (trades['pnl']>0).mean()*100
    total_return = trades['return_%'].sum()
    profit_factor = (trades[trades['pnl']>0]['pnl'].sum())/abs(trades[trades['pnl']<0]['pnl'].sum()+1e-9)
    return {"Trades":len(trades),"Win%":round(win_rate,2),"TotalRet%":round(total_return,2),"ProfitFactor":round(profit_factor,2)}, trades

# ======================
# Grid Search Optimizer
# ======================
def optimize(df):
    param_grid = {
        "ema_fast":[10,20,50],
        "ema_slow":[50,100,200],
        "rsi_period":[14],
        "rsi_buy":[25,30,35],
        "rsi_sell":[65,70,75],
        "bb_period":,
        "bb_dev":,
        "macd_fast":,
        "macd_slow":,
        "macd_signal":,
        "sl":[0.01],
        "tp":[0.02]
    }

    keys = list(param_grid.keys())
    best_score, best_params, best_trades = -999, None, None

    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        trades = backtest(df, params)
        stats = evaluate(trades)
        if stats:
            score = stats['TotalRet%']
            if score > best_score:
                best_score, best_params, best_trades = score, params, trades

    return best_params, best_trades, evaluate(best_trades)

# ======================
# Live Recommendation
# ======================
def live_signal(df, params):
    trades = backtest(df, params)
    last_trade = trades.iloc[-1] if not trades.empty else None

    # If last trade still open opportunity
    signal = {}
    close = df['close'].iloc[-1]
    date = df['date'].iloc[-1]

    long_cond = (EMA(df['close'], params['ema_fast']).iloc[-1] > EMA(df['close'], params['ema_slow']).iloc[-1]) and \
                (RSI(df['close'], params['rsi_period']).iloc[-1] < params['rsi_buy']) and \
                (close < Bollinger(df['close'], params['bb_period'],params['bb_dev']).iloc[-1])

    short_cond = (EMA(df['close'], params['ema_fast']).iloc[-1] < EMA(df['close'], params['ema_slow']).iloc[-1]) and \
                 (RSI(df['close'], params['rsi_period']).iloc[-1] > params['rsi_sell']) and \
                 (close > Bollinger(df['close'], params['bb_period'],params['bb_dev']).iloc[-1])

    if long_cond:
        signal = {"date":date,"signal":"LONG","entry":close,"sl":close*(1-params['sl']),"target":close*(1+params['tp'])}
    elif short_cond:
        signal = {"date":date,"signal":"SHORT","entry":close,"sl":close*(1+params['sl']),"target":close*(1-params['tp'])}
    else:
        signal = {"date":date,"signal":"NO TRADE"}

    return signal

# ======================
# Streamlit UI
# ======================
st.title("📊 Ultra-Robust Swing Trading Optimizer")

file = st.file_uploader("Upload CSV OHLCV")

if file:
    df = load_and_clean_data(file)
    st.write("Data preview:",df.head())

    best_params, best_trades, stats = optimize(df)

    st.subheader("✅ Best Strategy Parameters")
    st.json(best_params)

    st.subheader("📈 Backtest Performance")
    st.json(stats)
    st.write(stats[1])

    st.subheader("🔮 Live Recommendation (based on last candle)")
    signal = live_signal(df,best_params)
    st.json(signal)
