# streamlit_swing_trader.py
# Robust Swing Trading + Smart Trading Facility — File Upload + Live Recommendation + Detailed Backtest Journal
# No pandas_ta or talib. Indicators implemented from scratch.
# Requirements: streamlit, pandas, numpy, matplotlib, openpyxl
# Run: streamlit run streamlit_swing_trader.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Robust Swing Trader (File Upload)")

st.title("Robust Swing Trading + Smart Trading Facility — File Upload + Live Recommendation")
st.markdown(
    """
    Upload OHLCV data (CSV / Excel). Map columns, compute indicators and signals, run a detailed backtest and get live trade suggestions that match backtest trade structure.

    The backtester now records a full trade journal (entry, target, stop, qty, exit, pnl, points, reason, confidence) and computes summary stats.
    """
)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload CSV / Excel with OHLCV data (Date/Time, Open, High, Low, Close, Volume)", type=["csv", "xls", "xlsx"])    
    st.markdown("---")

    st.subheader("Strategy Params")
    risk_percent = st.number_input("Risk per trade (% of capital)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    starting_capital = st.number_input("Starting capital (INR)", value=100000.0, step=1000.0)
    max_hold_bars = st.number_input("Max holding bars (if TP/SL not hit)", value=20, min_value=1, step=1)
    spike_mult = st.number_input("Volume spike multiplier (big-player detector)", value=3.0, step=0.1)
    atr_mult_sl = st.number_input("Stop-loss ATR multiplier", value=2.0, step=0.1)
    rr_target = st.number_input("Target R:R (e.g., 2 means 2R)", value=2.0, step=0.1)
    show_indicators = st.checkbox("Show indicators on chart", value=True)

    st.markdown("---")
    run_backtest = st.button("Compute & Run")

# ------------------------------
# Helpers
# ------------------------------

def guess_column(cols, candidates):
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand.lower() == c:
                return cols[i]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand.lower() in c:
                return cols[i]
    return None


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

# ------------------------------
# Indicators
# ------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = upper - lower
    return ma, upper, lower, width


def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume']
    df2 = df.copy()
    df2['typ_vol'] = tp * vol
    df2['date_only'] = df2.index.normalize()
    df2['cum_tp_vol'] = df2.groupby('date_only')['typ_vol'].cumsum()
    df2['cum_vol'] = df2.groupby('date_only')['Volume'].cumsum()
    df2['vwap'] = df2['cum_tp_vol'] / df2['cum_vol']
    return df2['vwap']


def obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iat[i] > df['Close'].iat[i - 1]:
            obv.append(obv[-1] + df['Volume'].iat[i])
        elif df['Close'].iat[i] < df['Close'].iat[i - 1]:
            obv.append(obv[-1] - df['Volume'].iat[i])
        else:
            obv.append(obv[-1])
    s = pd.Series(obv, index=df.index)
    return s


def vpt(df):
    pct = df['Close'].pct_change().fillna(0)
    vpt = (pct * df['Volume']).cumsum()
    return vpt


def volume_profile(df, bins=24):
    prices = df['Close']
    vols = df['Volume']
    low = prices.min()
    high = prices.max()
    if high == low:
        edges = np.linspace(low - 1, high + 1, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)
    inds = np.digitize(prices, edges) - 1
    vol_by_bin = np.zeros(bins)
    for i, v in zip(inds, vols):
        if 0 <= i < bins:
            vol_by_bin[i] += v
    top_idx = np.argsort(vol_by_bin)[-5:][::-1]
    zones = [(edges[i], edges[i + 1], vol_by_bin[i]) for i in top_idx]
    return zones, edges, vol_by_bin

# ------------------------------
# Compute indicators and signals
# ------------------------------

def compute_indicators(df):
    df = df.copy()
    df['ema20'] = ema(df['Close'], 20)
    df['ema200'] = ema(df['Close'], 200)
    df['rsi14'] = rsi(df['Close'], 14)
    df['atr14'] = atr(df, 14)
    ma20, bb_upper, bb_lower, bb_width = bollinger_bands(df['Close'], 20, 2)
    df['bb_mid'] = ma20
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['kc_mid'] = tp.ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['kc_mid'] + 1.5 * df['atr14']
    df['kc_lower'] = df['kc_mid'] - 1.5 * df['atr14']
    df['vwap'] = vwap(df)
    df['obv'] = obv(df)
    df['vpt'] = vpt(df)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['squeeze_on'] = (df['bb_width'] < (df['kc_upper'] - df['kc_lower']))
    return df


def generate_signals(df, spike_mult=3.0, atr_mult_sl=2.0, rr_target=2.0):
    df = df.copy()
    bb_w_ma = df['bb_width'].rolling(20).mean()
    df['squeeze_release'] = df['squeeze_on'].shift(1).fillna(False) & (df['bb_width'] > bb_w_ma * 1.5)
    df['vol_pickup'] = df['Volume'] > 1.2 * df['vol_ma20']
    df['momentum_imminent'] = df['squeeze_release'] & df['vol_pickup']
    df['vol_spike'] = df['Volume'] > spike_mult * df['vol_ma20']
    df['strong_move'] = (df['Close'] - df['Close'].shift(1)).abs() > 0.8 * df['atr14']
    df['big_player'] = df['vol_spike'] & df['strong_move']
    df['above_ema200'] = df['Close'] > df['ema200']
    df['below_ema200'] = df['Close'] < df['ema200']
    df['to_ema20_pullback_long'] = df['Close'] <= df['ema20']
    df['to_ema20_pullback_short'] = df['Close'] >= df['ema20']
    df['rsi_rise'] = df['rsi14'] > df['rsi14'].shift(1)
    df['close_up'] = df['Close'] > df['Open']
    df['close_down'] = df['Close'] < df['Open']
    long_cond = (
        df['above_ema200'] & df['to_ema20_pullback_long'] & (df['rsi14'] > 45) & df['rsi_rise'] & df['close_up']
    )
    short_cond = (
        df['below_ema200'] & df['to_ema20_pullback_short'] & (df['rsi14'] < 55) & (~df['rsi_rise']) & df['close_down']
    )
    extra_ok = df['momentum_imminent'] | (~df['squeeze_on']) | (df['Volume'] > df['vol_ma20'] * 0.6)
    df['signal'] = 0
    df['signal_reason'] = ''
    df.loc[long_cond & extra_ok, 'signal'] = 1
    df.loc[short_cond & extra_ok, 'signal'] = -1
    df.loc[long_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'
    df.loc[short_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'
    return df

# ------------------------------
# Detailed backtest with trade journal
# ------------------------------

def backtest(df, capital=100000, risk_pct=1.0, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    cash = capital
    equity_curve = [capital]

    def entry_confidence_at(idx):
        row = df.iloc[idx]
        score = 0.0
        if row.get('above_ema200', False):
            score += 0.4
        elif row.get('below_ema200', False):
            score += 0.4
        if row.get('to_ema20_pullback_long', False) and row.get('above_ema200', False):
            score += 0.15
        if row.get('to_ema20_pullback_short', False) and row.get('below_ema200', False):
            score += 0.15
        if row.get('momentum_imminent', False):
            score += 0.2
        if row.get('big_player', False):
            score += 0.2
        if row.get('Volume', 0) > row.get('vol_ma20', 0):
            score += 0.1
        return min(score, 1.0)

    for i in range(len(df)):
        equity_curve.append(equity_curve[-1])
        sig = int(df['signal'].iat[i])
        if sig == 0:
            continue
        if i + 1 >= len(df):
            continue
        entry_idx = i + 1
        entry_price = df['Open'].iat[entry_idx]
        atr_val = df['atr14'].iat[entry_idx] if not np.isnan(df['atr14'].iat[entry_idx]) else 0
        if atr_val == 0:
            atr_val = df['Close'].diff().abs().rolling(14).mean().iat[entry_idx]

        conf = entry_confidence_at(i)
        reason = df['signal_reason'].iat[i] if 'signal_reason' in df.columns else ''

        if sig == 1:
            stop = entry_price - atr_mult * atr_val
            if stop <= 0:
                stop = entry_price * 0.98
            risk_per_share = entry_price - stop
            if risk_per_share <= 0:
                continue
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share))
            target = entry_price + rr_target * (entry_price - stop)

            exit_price = None
            exit_idx = None
            exit_reason = ''
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                if df['Low'].iat[j] <= stop:
                    exit_price = stop
                    exit_idx = j
                    exit_reason = 'SL'
                    break
                elif df['High'].iat[j] >= target:
                    exit_price = target
                    exit_idx = j
                    exit_reason = 'TP'
                    break
            if exit_price is None:
                exit_idx = min(len(df) - 1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]
                exit_reason = 'TimeExit'

            pnl = (exit_price - entry_price) * qty
            pnl_points = (exit_price - entry_price)
            cash += pnl
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'LONG',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'qty': qty,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_points': pnl_points,
                'pnl_pct_of_capital': pnl / capital if capital != 0 else 0,
                'reason': reason,
                'exit_reason': exit_reason,
                'confidence': conf
            })
            equity_curve[-1] = cash

        elif sig == -1:
            stop = entry_price + atr_mult * atr_val
            risk_per_share = stop - entry_price
            if risk_per_share <= 0:
                continue
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share))
            target = entry_price - rr_target * (stop - entry_price)

            exit_price = None
            exit_idx = None
            exit_reason = ''
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                if df['High'].iat[j] >= stop:
                    exit_price = stop
                    exit_idx = j
                    exit_reason = 'SL'
                    break
                elif df['Low'].iat[j] <= target:
                    exit_price = target
                    exit_idx = j
                    exit_reason = 'TP'
                    break
            if exit_price is None:
                exit_idx = min(len(df) - 1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]
                exit_reason = 'TimeExit'

            pnl = (entry_price - exit_price) * qty
            pnl_points = (entry_price - exit_price)
            cash += pnl
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'SHORT',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'qty': qty,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_points': pnl_points,
                'pnl_pct_of_capital': pnl / capital if capital != 0 else 0,
                'reason': reason,
                'exit_reason': exit_reason,
                'confidence': conf
            })
            equity_curve[-1] = cash

    # Align equity series with df index
    if len(equity_curve) == len(df) + 1:
        equity_series = pd.Series(equity_curve[1:], index=df.index)
    elif len(equity_curve) == len(df):
        equity_series = pd.Series(equity_curve, index=df.index)
    else:
        equity_series = pd.Series(equity_curve[-len(df):], index=df.index)

    trades_df = pd.DataFrame(trades)
    # Summary
    if not trades_df.empty:
        trades_df['win'] = trades_df['pnl'] > 0
        total_trades = len(trades_df)
        wins = trades_df['win'].sum()
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        total_points = trades_df['pnl_points'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_points = trades_df['pnl_points'].mean()
        cum_max = equity_series.cummax()
        drawdown = (equity_series - cum_max) / cum_max
        max_dd = drawdown.min()
        expectancy = ((trades_df.loc[trades_df['win'], 'pnl'].mean() if wins>0 else 0) * win_rate + (trades_df.loc[~trades_df['win'], 'pnl'].mean() if losses>0 else 0) * (1 - win_rate)) if total_trades>0 else 0
        trades_df.attrs['summary'] = {
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'total_points': float(total_points),
            'avg_pnl': float(avg_pnl),
            'avg_points': float(avg_points),
            'max_drawdown': float(max_dd),
            'expectancy': float(expectancy)
        }
    else:
        trades_df.attrs['summary'] = {}

    return trades_df, equity_series

# ------------------------------
# Main app flow
# ------------------------------

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file with OHLCV data in the sidebar.")
    st.stop()

try:
    raw = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

st.subheader("File preview (first 10 rows)")
st.write(raw.head(10))

cols = list(raw.columns)
guessed_date = guess_column(cols, ["date", "datetime", "timestamp", "time"]) 
guessed_open = guess_column(cols, ["open", "opn"]) 
guessed_high = guess_column(cols, ["high", "hi"]) 
guessed_low = guess_column(cols, ["low", "lo"]) 
guessed_close = guess_column(cols, ["close", "adj close", "last"]) 
guessed_vol = guess_column(cols, ["volume", "vol"]) 

st.subheader("Map columns")
col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Date/Time column", options=[None] + cols, index=cols.index(guessed_date)+1 if guessed_date in cols else 0)
    open_col = st.selectbox("Open column", options=[None] + cols, index=cols.index(guessed_open)+1 if guessed_open in cols else 0)
    high_col = st.selectbox("High column", options=[None] + cols, index=cols.index(guessed_high)+1 if guessed_high in cols else 0)
with col2:
    low_col = st.selectbox("Low column", options=[None] + cols, index=cols.index(guessed_low)+1 if guessed_low in cols else 0)
    close_col = st.selectbox("Close column", options=[None] + cols, index=cols.index(guessed_close)+1 if guessed_close in cols else 0)
    vol_col = st.selectbox("Volume column", options=[None] + cols, index=cols.index(guessed_vol)+1 if guessed_vol in cols else 0)

if not date_col or not open_col or not high_col or not low_col or not close_col:
    st.warning("Please map Date, Open, High, Low and Close columns.")
    st.stop()

working = raw[[date_col, open_col, high_col, low_col, close_col]].copy()
if vol_col:
    working['Volume'] = raw[vol_col].fillna(0)
else:
    working['Volume'] = 0

# parse date
try:
    working[date_col] = pd.to_datetime(working[date_col], infer_datetime_format=True, errors='coerce')
except Exception as e:
    st.error(f"Failed to parse date column: {e}")
    st.stop()

if working[date_col].isnull().any():
    st.warning("Some date values could not be parsed and will be dropped.")
    working = working.dropna(subset=[date_col])

working = working.rename(columns={date_col: 'DateTime', open_col: 'Open', high_col: 'High', low_col: 'Low', close_col: 'Close'})
working = working[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
working = working.sort_values('DateTime').set_index('DateTime')

df = working.copy()

if run_backtest:
    with st.spinner("Computing indicators and signals..."):
        df = compute_indicators(df)
        df = generate_signals(df, spike_mult=spike_mult, atr_mult_sl=atr_mult_sl, rr_target=rr_target)
        zones, edges, vol_by_bin = volume_profile(df, bins=24)
    st.success("Indicators & signals computed.")

    st.subheader("Latest data (last 10 rows)")
    st.write(df.tail(10))

    # chart
    st.subheader("Price chart (indicators + signals)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Close'], label='Close')
    if show_indicators:
        ax.plot(df.index, df['ema20'], label='EMA20', linewidth=0.8)
        ax.plot(df.index, df['ema200'], label='EMA200', linewidth=0.8)
        ax.plot(df.index, df['bb_upper'], label='BB Upper', linewidth=0.5, alpha=0.6)
        ax.plot(df.index, df['bb_lower'], label='BB Lower', linewidth=0.5, alpha=0.6)
        ax.plot(df.index, df['vwap'], label='VWAP', linewidth=0.8)
    longs = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    ax.scatter(longs.index, longs['Close'], marker='^', color='green', s=60, label='Long Signal')
    ax.scatter(shorts.index, shorts['Close'], marker='v', color='red', s=60, label='Short Signal')
    bigp = df[df['big_player']]
    ax.scatter(bigp.index, bigp['Close'], marker='o', facecolors='none', edgecolors='black', s=80, label='Big-player vol')
    ax.legend(loc='upper left')
    ax.set_title(f"Price with indicators & signals")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # volume profile
    st.subheader("Volume Profile (approx)")
    fig2, ax2 = plt.subplots(figsize=(4, 6))
    bins_center = (edges[:-1] + edges[1:]) / 2
    ax2.barh(bins_center, vol_by_bin)
    ax2.set_xlabel('Volume')
    ax2.set_ylabel('Price')
    st.pyplot(fig2)

    # backtest
    st.subheader("Backtest (detailed trade journal)")
    trades_df, equity = backtest(df, capital=starting_capital, risk_pct=risk_percent, max_hold=max_hold_bars, atr_mult=atr_mult_sl, rr_target=rr_target)

    if trades_df.empty:
        st.warning("No trades generated with these parameters. Consider loosening conditions or changing timeframe/data sampling.")
    else:
        # store trades in session for live PoP estimation
        st.session_state['last_backtest_trades'] = trades_df

        # show trade journal
        st.write(trades_df[['entry_time','exit_time','side','entry','target','stop','qty','exit','pnl','pnl_points','reason','exit_reason','confidence']].sort_values(by='entry_time', ascending=False))

        # summary metrics
        summary = trades_df.attrs.get('summary', {})
        st.metric("Total Trades", summary.get('total_trades', 0))
        st.metric("Win Rate", f"{summary.get('win_rate', 0)*100:.2f}%")
        st.metric("Total P&L (INR)", f"{summary.get('total_pnl', 0):.2f}")
        st.metric("Total Points", f"{summary.get('total_points', 0):.2f}")
        st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0)*100:.2f}%")
        st.metric("Expectancy (avg P&L)", f"{summary.get('expectancy', 0):.2f}")

        # equity chart
        st.subheader("Equity Curve")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        equity.plot(ax=ax3)
        ax3.set_ylabel('Equity (INR)')
        st.pyplot(fig3)

    # smart insights
    st.subheader("Smart Trading Insights (recent)")
    insights = []
    recent = df.iloc[-30:]
    if recent['momentum_imminent'].any():
        insights.append("Momentum signals detected recently (squeeze release + vol pickup). Watch for directional breakout.")
    if recent['big_player'].any():
        insights.append("Big-player volume spikes observed recently — these often precede extended moves or mark liquidity events.")
    if zones:
        top_zone = zones[0]
        insights.append(f"Top high-volume price area near {top_zone[0]:.2f} to {top_zone[1]:.2f}. Treat as strong support/resistance.")
    if len(insights) == 0:
        st.write("No immediate smart signals flagged in the recent window. Continue monitoring squeeze/volume.")
    else:
        for it in insights:
            st.info(it)

    st.markdown("---")
    st.subheader("Live Recommendation (based on last closed candle)")
    if st.button("Get Live Recommendation"):
        last_idx = len(df) - 1
        last = df.iloc[last_idx].copy()
        # compute confidence
        conf = 0.0
        if last.get('above_ema200', False):
            conf += 0.4
        elif last.get('below_ema200', False):
            conf += 0.4
        if last.get('to_ema20_pullback_long', False) and last.get('above_ema200', False):
            conf += 0.15
        if last.get('to_ema20_pullback_short', False) and last.get('below_ema200', False):
            conf += 0.15
        if last.get('momentum_imminent', False):
            conf += 0.2
        if last.get('big_player', False):
            conf += 0.2
        if last.get('Volume', 0) > last.get('vol_ma20', 0):
            conf += 0.1
        conf = min(conf, 1.0)

        # build suggested trade structure
        suggested = {
            'side': 'HOLD',
            'entry_time': df.index[last_idx],
            'entry': last['Close'],
            'stop': None,
            'target': None,
            'qty': 0,
            'confidence': conf,
            'reason': last.get('signal_reason', '')
        }
        # determine side
        if int(last.get('signal', 0)) == 1:
            side = 1
        elif int(last.get('signal', 0)) == -1:
            side = -1
        else:
            if last.get('above_ema200', False):
                side = 1
            elif last.get('below_ema200', False):
                side = -1
            else:
                side = 0

        if side == 1:
            suggested['side'] = 'BUY'
            entry_price = last['Close']
            atrval = last.get('atr14', 0)
            if atrval == 0:
                atrval = df['Close'].diff().abs().rolling(14).mean().iat[last_idx]
            stop = entry_price - atr_mult_sl * atrval
            if stop <= 0:
                stop = entry_price * 0.98
            target = entry_price + rr_target * (entry_price - stop)
            risk_per_share = entry_price - stop
            risk_amount = starting_capital * (risk_percent / 100.0)
            qty = max(1, int(risk_amount / risk_per_share)) if risk_per_share>0 else 1
            suggested.update({'entry': entry_price, 'stop': stop, 'target': target, 'qty': qty})
        elif side == -1:
            suggested['side'] = 'SELL'
            entry_price = last['Close']
            atrval = last.get('atr14', 0)
            if atrval == 0:
                atrval = df['Close'].diff().abs().rolling(14).mean().iat[last_idx]
            stop = entry_price + atr_mult_sl * atrval
            target = entry_price - rr_target * (stop - entry_price)
            risk_per_share = stop - entry_price
            risk_amount = starting_capital * (risk_percent / 100.0)
            qty = max(1, int(risk_amount / risk_per_share)) if risk_per_share>0 else 1
            suggested.update({'entry': entry_price, 'stop': stop, 'target': target, 'qty': qty})

        # estimate PoP from backtest stored in session
        pop = None
        back_trades = st.session_state.get('last_backtest_trades', None)
        if back_trades is not None and not back_trades.empty:
            pop = back_trades.attrs.get('summary', {}).get('win_rate', None)

        st.subheader('Live Trade Suggestion (structured)')
        st.write(pd.DataFrame([suggested]))
        st.success(f"Suggested: {suggested['side']} — Confidence: {conf*100:.0f}% — Estimated PoP: {((pop*100) if pop is not None else 'N/A')}%")
        st.write('Reason:', suggested['reason'])
        if pop is not None:
            st.info(f"Backtest Win Rate (used as PoP estimate): {pop*100:.2f}%")

    st.markdown("---")
    st.write("**How to use**: Upload data, map columns, click 'Compute & Run'. Use the detailed trade journal for review and 'Get Live Recommendation' to get a structured trade suggestion consistent with backtest rules.")
else:
    st.info("Map columns and click 'Compute & Run' in the sidebar to compute indicators, signals, and run backtest.")

# EOF
