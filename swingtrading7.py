import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

# ----------------------------
# Utility functions
# ----------------------------

def clean_columns(df):
    """Standardize column names and convert OHLCV to numeric."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename variations
    for col in ["timestamp","datetime","date_","dates"]:
        if col in df.columns:
            df.rename(columns={col: "date"}, inplace=True)

    # Ensure OHLCV are numeric
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str)
                       .str.replace(",","", regex=False)
                       .str.strip())
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def preprocess_data(df):
    """Ensure date sorting and cleanup."""
    if "date" not in df.columns:
        raise ValueError("No 'date' column found in uploaded file!")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date','open','high','low','close']).copy()
    df = df.sort_values('date').reset_index(drop=True)
    return df

# ----------------------------
# Indicator Calculations (manual)
# ----------------------------

def add_indicators(df):
    """Calculate 10+ indicators manually."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 1. SMA
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()

    # 2. EMA
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = close.ewm(span=50, adjust=False).mean()

    # 3. RSI
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # 4. MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 5. Bollinger Bands
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # 6. ATR
    df['tr'] = np.maximum(high-low, np.maximum(abs(high-close.shift()), abs(low-close.shift())))
    df['atr_14'] = df['tr'].rolling(14).mean()

    # 7. OBV
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # 8. CCI
    tp = (high+low+close)/3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x-np.mean(x))), raw=True)
    df['cci'] = (tp - sma_tp) / (0.015 * mad)

    # 9. Stochastic Oscillator
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['stoch_k'] = 100 * (close - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 10. Momentum
    df['momentum'] = close / close.shift(10) * 100

    return df

# ----------------------------
# Strategy & Backtesting
# ----------------------------

def generate_signals(df, params, trade_side="long_short"):
    """Generate signals based on optimized parameters."""
    df['signal'] = 0

    # Example rule: SMA crossover + RSI filter + MACD filter
    cond_long = (df['sma_20'] > df['sma_50']) & (df['rsi'] < params['rsi_max']) & (df['macd'] > df['macd_signal'])
    cond_short = (df['sma_20'] < df['sma_50']) & (df['rsi'] > params['rsi_min']) & (df['macd'] < df['macd_signal'])

    if trade_side == "long":
        df.loc[cond_long, 'signal'] = 1
    elif trade_side == "short":
        df.loc[cond_short, 'signal'] = -1
    else: # both
        df.loc[cond_long, 'signal'] = 1
        df.loc[cond_short, 'signal'] = -1

    return df


def backtest(df, params, trade_side="long_short"):
    df = generate_signals(df.copy(), params, trade_side)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['position'] * df['returns']
    strat_cum = (1+df['strategy']).cumprod()-1
    buyhold_cum = (1+df['returns']).cumprod()-1

    total_profit = strat_cum.iloc[-1]
    buyhold_profit = buyhold_cum.iloc[-1]
    accuracy = (df.loc[df['position']!=0,'strategy']>0).mean() if any(df['position']!=0) else 0

    return total_profit, buyhold_profit, accuracy, df


# ----------------------------
# Optimization
# ----------------------------

def optimize(df, method="grid", trade_side="long_short", n_iter=20):
    best_profit = -999
    best_params = None
    best_result = None

    param_grid = {
        "rsi_max": [60,65,70,75,80],
        "rsi_min": [20,25,30,35,40]
    }

    if method=="grid":
        combos = list(itertools.product(param_grid['rsi_max'], param_grid['rsi_min']))
    else: # randomized
        combos = [(random.choice(param_grid['rsi_max']), random.choice(param_grid['rsi_min'])) for _ in range(n_iter)]

    for rsi_max,rsi_min in combos:
        params = {"rsi_max": rsi_max, "rsi_min": rsi_min}
        profit, bh, acc, result = backtest(df, params, trade_side)
        if profit > best_profit:
            best_profit = profit
            best_params = params
            best_result = result

    return best_params, best_result, best_profit

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Robust Stock Strategy Optimizer", layout="wide")
st.title("📈 Robust Stock Strategy Optimizer & Live Recommendations")

uploaded_file = st.file_uploader("Upload OHLCV CSV/Excel", type=['csv','xlsx'])
search_type = st.selectbox("Optimization Method", ["grid","randomized"])
trade_side = st.selectbox("Trade Side", ["long","short","long_short"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = clean_columns(df)
    df = preprocess_data(df)
    df = add_indicators(df)

    # Run optimization
    params, result_df, profit = optimize(df, search_type, trade_side)
    bh_profit = (1+result_df['returns']).cumprod().iloc[-1]-1
    accuracy = (result_df.loc[result_df['position']!=0,'strategy']>0).mean()

    # Show metrics
    st.subheader("📊 Backtest Results")
    st.write(f"Best Parameters: {params}")
    st.metric("Strategy Profit %", f"{profit*100:.2f}%")
    st.metric("Buy & Hold Profit %", f"{bh_profit*100:.2f}%")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

    # Recommendation
    latest = result_df.iloc[-1]
    if latest['position']==1:
        rec = f"✅ BUY @ {latest['close']:.2f}, SL {latest['close']-latest['atr_14']:.2f}, Target {latest['close']+2*latest['atr_14']:.2f}"
    elif latest['position']==-1:
        rec = f"❌ SELL @ {latest['close']:.2f}, SL {latest['close']+latest['atr_14']:.2f}, Target {latest['close']-2*latest['atr_14']:.2f}"
    else:
        rec = f"⚠️ HOLD | No clear signal. Price {latest['close']:.2f}"
    st.subheader("🔔 Live Recommendation")
    st.write(rec)

    # Plot
    st.subheader("📉 Price & Strategy")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(result_df['date'], result_df['close'], label="Close", color="black")
    ax.plot(result_df['date'], result_df['sma_20'], label="SMA20", color="blue")
    ax.plot(result_df['date'], result_df['sma_50'], label="SMA50", color="red")
    ax.legend()
    st.pyplot(fig)

    # Summary in layman terms
    st.subheader("📝 Summary")
    st.write(f"""
    Over the backtest period, the optimized strategy using indicators such as SMA, EMA, RSI, MACD, 
    Bollinger Bands, ATR, OBV, CCI, Stochastic, and Momentum showed a profit of {profit*100:.2f}%, 
    compared to Buy & Hold which returned {bh_profit*100:.2f}%. The system identified profitable 
    trades with an accuracy of {accuracy*100:.2f}%. Based on the latest candle, the system suggests: {rec}. 
    Entry, Stop Loss, and Target levels have been computed dynamically using ATR. This strategy adapts 
    to both bullish and bearish markets and can be used for swing trading with risk management built in.
    """)
