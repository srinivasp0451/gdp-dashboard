streamlit_swing_strategy.py

Streamlit app: Swing trading strategy backtester + live signal recommender (long only)

Usage: streamlit run streamlit_swing_strategy.py

import streamlit as st import pandas as pd import numpy as np import matplotlib.pyplot as plt from datetime import timedelta

st.set_page_config(layout="wide", page_title="Swing Trading Backtester & Live Recommender")

------------------ Utility indicators ------------------

@st.cache_data def compute_indicators(df, short=20, long=50, rsi_period=14): df = df.copy() df['close'] = df['Close'] if 'Close' in df.columns else df['close'] df['open']  = df['Open']  if 'Open'  in df.columns else df['open'] df['high']  = df['High']  if 'High'  in df.columns else df['high'] df['low']   = df['Low']   if 'Low'   in df.columns else df['low'] df['volume'] = df['Volume'] if 'Volume' in df.columns else (df['volume'] if 'volume' in df.columns else 0)

df['ma_short'] = df['close'].rolling(short, min_periods=1).mean()
df['ma_long']  = df['close'].rolling(long,  min_periods=1).mean()

# RSI implementation
delta = df['close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
rs = ma_up / (ma_down.replace(0, np.nan))
df['rsi'] = 100 - (100 / (1 + rs))
df['rsi'].fillna(50, inplace=True)

return df

------------------ Backtester ------------------

def backtest_strategy(df, params): # params: dict with short_ma, long_ma, rsi_entry, target_pct, sl_pct, max_hold df = compute_indicators(df, short=params['short_ma'], long=params['long_ma'], rsi_period=params.get('rsi_period',14)) trades = [] position = None

for i in range(1, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i-1]
    date = row.name if hasattr(row, 'name') else row['Date']

    # Check for open position
    if position is None:
        # Entry condition (long only): MA bullish crossover + RSI below threshold
        ma_cond = (prev['ma_short'] <= prev['ma_long']) and (row['ma_short'] > row['ma_long'])
        rsi_cond = row['rsi'] <= params['rsi_entry']
        if ma_cond and rsi_cond:
            entry_price = row['open'] if not np.isnan(row['open']) else row['close']
            position = {
                'entry_date': date,
                'entry_price': entry_price,
                'target': entry_price * (1 + params['target_pct']/100),
                'sl': entry_price * (1 - params['sl_pct']/100),
                'max_hold': params['max_hold'],
                'hold_days': 0,
            }
    else:
        # Manage open position
        position['hold_days'] += 1
        high = row['high'] if not np.isnan(row['high']) else row['close']
        low  = row['low']  if not np.isnan(row['low'])  else row['close']
        exit_price = None
        exit_reason = None

        # target hit
        if high >= position['target']:
            exit_price = position['target']
            exit_reason = 'Target'
        # sl hit
        elif low <= position['sl']:
            exit_price = position['sl']
            exit_reason = 'StopLoss'
        # max hold
        elif position['hold_days'] >= position['max_hold']:
            exit_price = row['close']
            exit_reason = 'MaxHold'

        if exit_price is not None:
            pnl = exit_price - position['entry_price']
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': date,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'points': pnl, # for price based assets
                'reason': exit_reason,
                'hold_days': position['hold_days']
            })
            position = None

# if position still open at end
if position is not None:
    last = df.iloc[-1]
    last_date = last.name if hasattr(last, 'name') else last['Date']
    exit_price = last['close']
    pnl = exit_price - position['entry_price']
    trades.append({
        'entry_date': position['entry_date'],
        'exit_date': last_date,
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'pnl': pnl,
        'points': pnl,
        'reason': 'Open',
        'hold_days': position['hold_days']
    })

trades_df = pd.DataFrame(trades)
if trades_df.empty:
    metrics = {
        'net_pnl': 0,
        'n_trades': 0,
        'win_rate': 0.0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0
    }
else:
    wins = trades_df[trades_df['pnl']>0]
    losses = trades_df[trades_df['pnl']<=0]
    gross_win = wins['pnl'].sum() if not wins.empty else 0
    gross_loss = -losses['pnl'].sum() if not losses.empty else 0
    metrics = {
        'net_pnl': trades_df['pnl'].sum(),
        'n_trades': len(trades_df),
        'win_rate': len(wins)/len(trades_df),
        'avg_win': wins['pnl'].mean() if not wins.empty else 0,
        'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
        'profit_factor': (gross_win / gross_loss) if gross_loss>0 else np.inf
    }

return trades_df, metrics

------------------ Optimizer ------------------

def optimize_params(df, search_space, max_evals=50): # search_space: dict of lists; we'll run random search sampling combinations import random best = None results = [] all_combinations = [] for _ in range(max_evals): params = {k: random.choice(v) for k, v in search_space.items()} trades, metrics = backtest_strategy(df, params) results.append((params, metrics)) if best is None or metrics['net_pnl'] > best[1]['net_pnl']: best = (params, metrics) return best, results

------------------ Frontend ------------------

st.title("Swing Trading Backtester & Live Recommender (Long Only)")

col1, col2 = st.columns([2,1])

with col1: st.subheader("Data input") uploaded = st.file_uploader("Upload OHLC CSV (Date, Open, High, Low, Close, Volume). We'll try default file if not uploaded.", type=['csv']) if uploaded is None: try: df = pd.read_csv('/mnt/data/Quote-Equity-INFY.csv', parse_dates=True, index_col=0) st.caption("Using default sample: /mnt/data/Quote-Equity-INFY.csv") except Exception as e: st.warning("No default sample found. Please upload a CSV.") st.stop() else: df = pd.read_csv(uploaded, parse_dates=True, index_col=0)

# ensure required columns
df.rename(columns={c:c.capitalize() for c in df.columns}, inplace=True)
required = set(['Open','High','Low','Close'])
if not required.issubset(set(df.columns)):
    st.error(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
    st.stop()

st.write("Data preview")
st.dataframe(df.tail(10))

with col2: st.subheader("Strategy defaults") short_ma = st.number_input('Short MA (fast)', value=20, min_value=3) long_ma  = st.number_input('Long MA (slow)', value=50, min_value=5) rsi_entry = st.slider('RSI entry threshold (<=)', 10, 60, 40) target_pct = st.number_input('Target %', value=2.0, step=0.1) sl_pct = st.number_input('Stoploss %', value=2.0, step=0.1) max_hold = st.number_input('Max hold days', value=5, min_value=1) rsi_period = st.number_input('RSI period', value=14, min_value=5)

st.markdown('---')

Optimization controls

st.subheader('Optimization') with st.expander('Optimizer (random search)'): do_opt = st.checkbox('Run optimizer to tune params (random search)', value=False) if do_opt: short_choices = st.multiselect('Short MA choices', [10,15,20,30,50], default=[20,30]) long_choices  = st.multiselect('Long MA choices', [50,100,150,200], default=[50,100]) rsi_choices   = st.multiselect('RSI entry choices', [20,30,35,40,45], default=[35,40]) target_choices = st.multiselect('Target % choices', [0.5,1,1.5,2,3], default=[1,2]) sl_choices     = st.multiselect('SL % choices', [0.5,1,1.5,2,3,5], default=[1,2]) maxhold_choices = st.multiselect('Max hold choices', [3,5,7,10], default=[5,7]) max_evals = st.number_input('Max evaluations', value=30, min_value=5)

run_button = st.button('Run backtest & get live recommendation')

if run_button: params = { 'short_ma': int(short_ma), 'long_ma': int(long_ma), 'rsi_entry': int(rsi_entry), 'target_pct': float(target_pct), 'sl_pct': float(sl_pct), 'max_hold': int(max_hold), 'rsi_period': int(rsi_period) }

if do_opt:
    search_space = {
        'short_ma': short_choices or [params['short_ma']],
        'long_ma': long_choices or [params['long_ma']],
        'rsi_entry': rsi_choices or [params['rsi_entry']],
        'target_pct': target_choices or [params['target_pct']],
        'sl_pct': sl_choices or [params['sl_pct']],
        'max_hold': maxhold_choices or [params['max_hold']],
        'rsi_period': [params['rsi_period']]
    }
    st.info('Running optimizer (random search). This may take a while depending on evaluations).')
    best, results = optimize_params(df, search_space, max_evals=int(max_evals))
    best_params, best_metrics = best
    st.success('Optimization completed. Best params selected based on net_pnl.')
    st.write('Best params:')
    st.json(best_params)
    st.write('Best metrics:')
    st.json(best_metrics)
    params = best_params

trades_df, metrics = backtest_strategy(df, params)

st.subheader('Backtest Metrics')
st.write(metrics)

st.subheader('Trades Log')
if trades_df.empty:
    st.write('No trades found for these params on given data.')
else:
    st.dataframe(trades_df.sort_values('entry_date', ascending=False))

# Confidence & logic: derive from historical win-rate and rationale
confidence = float(metrics['win_rate']) if metrics['n_trades']>0 else 0.0
logic = (
    f"Long when {params['short_ma']}-MA crosses above {params['long_ma']}-MA and RSI <= {params['rsi_entry']}. "
    f"Target {params['target_pct']}% and SL {params['sl_pct']}%. Max hold {params['max_hold']} days."
)

# Live recommendation logic: check if last trade is still open (reason == 'Open') -> reuse
live_reco = None
if not trades_df.empty and trades_df.iloc[-1]['reason'] == 'Open':
    last_trade = trades_df.iloc[-1]
    live_reco = {
        'type': 'ExistingOpenTrade',
        'entry_date': str(last_trade['entry_date']),
        'entry_price': float(last_trade['entry_price']),
        'current_price': float(df.iloc[-1]['close']),
        'pnl': float(last_trade['pnl']),
        'logic': logic,
        'confidence': confidence
    }
else:
    # Evaluate signal on last row
    latest = compute_indicators(df, short=params['short_ma'], long=params['long_ma'], rsi_period=params['rsi_period']).iloc[-1]
    prev = compute_indicators(df, short=params['short_ma'], long=params['long_ma'], rsi_period=params['rsi_period']).iloc[-2]
    ma_cond = (prev['ma_short'] <= prev['ma_long']) and (latest['ma_short'] > latest['ma_long'])
    rsi_cond = latest['rsi'] <= params['rsi_entry']
    if ma_cond and rsi_cond:
        entry_price = latest['open'] if not np.isnan(latest['open']) else latest['close']
        live_reco = {
            'type': 'NewEntrySignal',
            'entry_price': float(entry_price),
            'target': float(entry_price * (1 + params['target_pct']/100)),
            'sl': float(entry_price * (1 - params['sl_pct']/100)),
            'logic': logic,
            'confidence': confidence
        }

st.subheader('Live Recommendation')
if live_reco is None:
    st.write('No live long recommendation at this moment based on the strategy and data.')
else:
    st.write(live_reco)

# Plot price and MA
st.subheader('Price Chart (last 200 bars)')
plot_df = compute_indicators(df, short=params['short_ma'], long=params['long_ma'])
chart_df = plot_df.tail(200)[['close','ma_short','ma_long']]
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(chart_df['close'], label='Close')
ax.plot(chart_df['ma_short'], label=f"MA{params['short_ma']}")
ax.plot(chart_df['ma_long'], label=f"MA{params['long_ma']}")
ax.legend()
st.pyplot(fig)

# Summary box
st.markdown('---')
st.subheader('Summary')
st.write(f"Net PNL: {metrics['net_pnl']:.2f}, Trades: {metrics['n_trades']}, Win rate: {metrics['win_rate']*100:.1f}%")
st.write('Logic:')
st.write(logic)
st.write('Confidence (historical win-rate for this setup):')
st.write(f"{confidence*100:.1f}%")

st.info('Notes:\n- This app runs entirely in memory; recommendations are NOT persisted.\n- If the backtest produces an open trade on the latest date, the same open trade will be shown as the current live recommendation (so repeated runs on the same data are idempotent).\n- Optimizer uses random search; if you want a deeper grid search increase evaluations.')

End of app

