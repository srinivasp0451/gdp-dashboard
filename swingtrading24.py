import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Indicators (no pandas_ta / ta-lib)
# -------------------------
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def BollingerBands(series, period=20, num_std=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

# -------------------------
# Signal generation
# -------------------------
def generate_signals(df):
    df["signal"] = 0
    df["EMA20"] = EMA(df['Close'], 20)
    df["EMA200"] = EMA(df['Close'], 200)
    df["RSI"] = RSI(df['Close'])
    df["atr14"] = ATR(df)
    ma, upper, lower = BollingerBands(df['Close'])
    df['BB_mid'], df['BB_upper'], df['BB_lower'] = ma, upper, lower
    df['BB_width'] = (upper - lower) / ma
    df['vol_ma20'] = df['Volume'].rolling(20).mean()

    uptrend = df['Close'] > df['EMA200']
    downtrend = df['Close'] < df['EMA200']
    pullback_long = df['Close'] > df['EMA20']
    pullback_short = df['Close'] < df['EMA20']
    rsi_long = df['RSI'] > 55
    rsi_short = df['RSI'] < 45
    momentum = (df['BB_width'].pct_change() > 0.05) & (df['Volume'] > df['vol_ma20'])
    big_player = df['Volume'] > 1.5 * df['vol_ma20']

    df.loc[uptrend & pullback_long & rsi_long & (momentum | big_player), 'signal'] = 1
    df.loc[downtrend & pullback_short & rsi_short & (momentum | big_player), 'signal'] = -1

    df['above_ema200'] = uptrend
    df['below_ema200'] = downtrend
    df['to_ema20_pullback_long'] = pullback_long
    df['to_ema20_pullback_short'] = pullback_short
    df['momentum_imminent'] = momentum
    df['big_player'] = big_player

    return df

# -------------------------
# Backtest (fixed equity alignment)
# -------------------------
def backtest(df, capital=100000, risk_pct=1.0, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    cash = capital
    equity_curve = [capital]

    for i in range(len(df)):
        equity_curve.append(equity_curve[-1])
        sig = int(df['signal'].iat[i])
        if sig == 0:
            continue
        if i + 1 >= len(df):
            continue
        entry_idx = i + 1
        entry_price = df['Open'].iat[entry_idx]
        atr = df['atr14'].iat[entry_idx] if not np.isnan(df['atr14'].iat[entry_idx]) else 0
        if atr == 0:
            atr = df['Close'].diff().abs().rolling(14).mean().iat[entry_idx]

        if sig == 1:
            stop = entry_price - atr_mult * atr
            risk_per_share = entry_price - stop
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share)) if risk_per_share > 0 else 1
            target = entry_price + rr_target * (entry_price - stop)
            exit_price, exit_idx = None, None
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                if df['Low'].iat[j] <= stop:
                    exit_price, exit_idx = stop, j
                    break
                elif df['High'].iat[j] >= target:
                    exit_price, exit_idx = target, j
                    break
            if exit_price is None:
                exit_idx = min(len(df)-1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]
            profit = (exit_price - entry_price) * qty
            cash += profit
            trades.append({'entry_time': df.index[entry_idx], 'exit_time': df.index[exit_idx], 'side': 'LONG', 'entry': entry_price, 'exit': exit_price, 'qty': qty, 'pnl': profit})
            equity_curve[-1] = cash

        elif sig == -1:
            stop = entry_price + atr_mult * atr
            risk_per_share = stop - entry_price
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share)) if risk_per_share > 0 else 1
            target = entry_price - rr_target * (stop - entry_price)
            exit_price, exit_idx = None, None
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                if df['High'].iat[j] >= stop:
                    exit_price, exit_idx = stop, j
                    break
                elif df['Low'].iat[j] <= target:
                    exit_price, exit_idx = target, j
                    break
            if exit_price is None:
                exit_idx = min(len(df)-1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]
            profit = (entry_price - exit_price) * qty
            cash += profit
            trades.append({'entry_time': df.index[entry_idx], 'exit_time': df.index[exit_idx], 'side': 'SHORT', 'entry': entry_price, 'exit': exit_price, 'qty': qty, 'pnl': profit})
            equity_curve[-1] = cash

    # Align equity series length with df
    if len(equity_curve) == len(df) + 1:
        equity_series = pd.Series(equity_curve[1:], index=df.index)
    elif len(equity_curve) == len(df):
        equity_series = pd.Series(equity_curve, index=df.index)
    else:
        equity_series = pd.Series(equity_curve[-len(df):], index=df.index)

    trades_df = pd.DataFrame(trades)
    return trades_df, equity_series

# -------------------------
# Streamlit UI
# -------------------------
st.title("Swing Trading Strategy (File Upload)")

uploaded_file = st.file_uploader("Upload OHLCV file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Columns detected:", df.columns.tolist())
    col_date = st.selectbox("Date/Time column", df.columns)
    col_open = st.selectbox("Open column", df.columns)
    col_high = st.selectbox("High column", df.columns)
    col_low = st.selectbox("Low column", df.columns)
    col_close = st.selectbox("Close column", df.columns)
    col_volume = st.selectbox("Volume column", df.columns)

    df = df.rename(columns={col_date: "Date", col_open: "Open", col_high: "High", col_low: "Low", col_close: "Close", col_volume: "Volume"})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    df = generate_signals(df)
    trades_df, equity = backtest(df)

    st.subheader("Backtest Results")
    st.write(trades_df.tail())
    st.line_chart(equity)

    if st.button("Get Live Recommendation"):
        last = df.iloc[-1].copy()
        reasons = []
        score = 0.0
        if last.get('above_ema200', False):
            reasons.append('Trend: LONG (Close > EMA200)')
            score += 0.4
        elif last.get('below_ema200', False):
            reasons.append('Trend: SHORT (Close < EMA200)')
            score += 0.4
        else:
            reasons.append('Trend: Neutral')
        if last.get('to_ema20_pullback_long', False) and last.get('above_ema200', False):
            reasons.append('Price near/below EMA20 (pullback) — possible entry on bounce')
            score += 0.15
        if last.get('to_ema20_pullback_short', False) and last.get('below_ema200', False):
            reasons.append('Price near/above EMA20 (pullback) — possible short on rejection')
            score += 0.15
        if last.get('momentum_imminent', False):
            reasons.append('Momentum imminent (squeeze release + vol pickup)')
            score += 0.2
        if last.get('big_player', False):
            reasons.append('Big-player volume spike detected')
            score += 0.2
        if last.get('Volume', 0) > last.get('vol_ma20', 0):
            reasons.append('Volume above 20-period average')
            score += 0.1
        conf = min(score, 1.0)
        rec = 'HOLD'
        if int(last.get('signal', 0)) == 1:
            rec = 'BUY'
        elif int(last.get('signal', 0)) == -1:
            rec = 'SELL'
        else:
            if last.get('above_ema200', False) and (last.get('momentum_imminent', False) or last.get('big_player', False)):
                rec = 'WATCH/CONSIDER BUY'
            elif last.get('below_ema200', False) and (last.get('momentum_imminent', False) or last.get('big_player', False)):
                rec = 'WATCH/CONSIDER SELL'
        st.success(f"Recommendation: {rec} — Confidence: {conf*100:.0f}%")
        st.write("Reasons:")
        for r in reasons:
            st.write(f"- {r}")
