streamlit_swing_strategy.py

Enhanced Streamlit app: Swing trading strategy backtester + live signal recommender (long only)

Added indicators: MACD, ATR-based SL, ADX

Usage: streamlit run streamlit_swing_strategy.py

import streamlit as st import pandas as pd import numpy as np import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Swing Trading Backtester & Live Recommender")

------------------ Utility indicators ------------------

def compute_indicators(df, short=20, long=50, rsi_period=14): df = df.copy() df.rename(columns={c.capitalize(): c.capitalize() for c in df.columns}, inplace=True) df['ma_short'] = df['Close'].rolling(short, min_periods=1).mean() df['ma_long'] = df['Close'].rolling(long, min_periods=1).mean()

delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
rs = ma_up / ma_down.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))
df['rsi'].fillna(50, inplace=True)

exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = exp1 - exp2
df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

tr1 = df['High'] - df['Low']
tr2 = abs(df['High'] - df['Close'].shift())
tr3 = abs(df['Low'] - df['Close'].shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

plus_dm = df['High'].diff()
minus_dm = df['Low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm > 0] = 0
tr_smooth = tr.rolling(14).sum()
plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
minus_di = abs(100 * (minus_dm.rolling(14).sum() / tr_smooth))
dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
df['adx'] = dx.rolling(14).mean()

return df

------------------ Backtester ------------------

def backtest_strategy(df, params): df = compute_indicators(df, short=params['short_ma'], long=params['long_ma'], rsi_period=params['rsi_period']) trades = [] position = None

for i in range(1, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if position is None:
        ma_cond = prev['ma_short'] <= prev['ma_long'] and row['ma_short'] > row['ma_long']
        rsi_cond = row['rsi'] <= params['rsi_entry']
        macd_cond = row['macd'] > row['signal']
        adx_cond = row['adx'] >= 20

        if ma_cond and rsi_cond and macd_cond and adx_cond:
            atr_sl = row['Close'] - (row['atr'] * params['atr_mult'])
            entry_price = row['Open']
            position = {
                'entry_date': row.name,
                'entry_price': entry_price,
                'target': entry_price * (1 + params['target_pct'] / 100),
                'sl': atr_sl if params['use_atr_sl'] else entry_price * (1 - params['sl_pct'] / 100),
                'hold_days': 0
            }
    else:
        position['hold_days'] += 1
        if row['High'] >= position['target']:
            trades.append({**position, 'exit_date': row.name, 'exit_price': position['target'], 'pnl': position['target'] - position['entry_price'], 'reason': 'Target'})
            position = None
        elif row['Low'] <= position['sl']:
            trades.append({**position, 'exit_date': row.name, 'exit_price': position['sl'], 'pnl': position['sl'] - position['entry_price'], 'reason': 'StopLoss'})
            position = None
        elif position['hold_days'] >= params['max_hold']:
            trades.append({**position, 'exit_date': row.name, 'exit_price': row['Close'], 'pnl': row['Close'] - position['entry_price'], 'reason': 'MaxHold'})
            position = None

trades_df = pd.DataFrame(trades)
metrics = {
    'net_pnl': trades_df['pnl'].sum() if not trades_df.empty else 0,
    'n_trades': len(trades_df),
    'win_rate': (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
}
return trades_df, metrics

------------------ UI ------------------

st.title("Swing Trading Backtester & Live Recommender (Enhanced)") uploaded = st.file_uploader("Upload OHLC CSV", type=['csv']) if uploaded: df = pd.read_csv(uploaded, parse_dates=True, index_col=0) st.dataframe(df.tail()) short_ma = st.number_input('Short MA', 5, 50, 20) long_ma = st.number_input('Long MA', 10, 200, 50) rsi_entry = st.slider('RSI Entry <=', 10, 60, 40) target_pct = st.number_input('Target %', 0.1, 10.0, 2.0) sl_pct = st.number_input('Stoploss %', 0.1, 10.0, 2.0) max_hold = st.number_input('Max hold days', 1, 30, 5) use_atr_sl = st.checkbox('Use ATR-based SL', value=False) atr_mult = st.number_input('ATR SL Multiplier', 0.5, 5.0, 1.5)

if st.button('Run Backtest'):
    params = {
        'short_ma': short_ma,
        'long_ma': long_ma,
        'rsi_entry': rsi_entry,
        'target_pct': target_pct,
        'sl_pct': sl_pct,
        'max_hold': max_hold,
        'rsi_period': 14,
        'use_atr_sl': use_atr_sl,
        'atr_mult': atr_mult
    }
    trades_df, metrics = backtest_strategy(df, params)
    st.subheader('Metrics')
    st.write(metrics)
    st.subheader('Trades')
    st.dataframe(trades_df)
    st.subheader('Chart')
    df_ind = compute_indicators(df, short=short_ma, long=long_ma)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_ind['Close'], label='Close')
    ax.plot(df_ind['ma_short'], label=f'MA{short_ma}')
    ax.plot(df_ind['ma_long'], label=f'MA{long_ma}')
    ax.legend()
    st.pyplot(fig)

