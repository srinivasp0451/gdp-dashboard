# swing_algo_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Tuple, Dict, List

st.set_page_config(layout="wide", page_title="Swing Algo Trading Platform", initial_sidebar_state="expanded")

# ----------------------
# Utilities & Helpers
# ----------------------
def infer_datetime(col: pd.Series):
    try:
        return pd.to_datetime(col, infer_datetime_format=True, errors='coerce')
    except:
        return pd.to_datetime(col, errors='coerce')

def flexible_column_mapping(df: pd.DataFrame) -> Dict[str,str]:
    """
    Map arbitrary column names (case/format) to standard 'date','open','high','low','close','volume'.
    Returns mapping dict.
    """
    lower_map = {c.lower(): c for c in df.columns}
    mapping = {}
    # candidates
    candidates = {
        'date': ['date', 'time', 'datetime', 'timestamp'],
        'open': ['open', 'openprice', 'o', 'op'],
        'high': ['high', 'highprice', 'h'],
        'low': ['low', 'lowprice', 'l'],
        'close': ['close', 'closeprice', 'c', 'last'],
        'volume': ['volume', 'vol', 'v']
    }
    for target, keys in candidates.items():
        for k in keys:
            if k in lower_map:
                mapping[target] = lower_map[k]
                break
    return mapping

def auto_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    mapping = flexible_column_mapping(df)
    st.write("Detected column mapping (you can override):")
    cols = df.columns.tolist()
    cols_display = {}
    # allow user to fix mapping
    for target in ['date','open','high','low','close','volume']:
        default = mapping.get(target, None)
        sel = st.selectbox(f"{target.upper()} column", options=[None] + cols, index=(0 if default is None else cols.index(default)+1), key=f"map_{target}")
        if sel:
            mapping[target] = sel
    # rename
    rename_map = {mapping[k]: k for k in mapping if k in mapping and mapping[k] in df.columns}
    df = df.rename(columns=rename_map)
    # parse date
    if 'date' in df.columns:
        df['date'] = infer_datetime(df['date'])
        if df['date'].isna().sum() > 0:
            st.warning("Some date values couldn't be parsed - rows with missing dates will be dropped.")
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        st.error("No date column selected - cannot proceed.")
        st.stop()
    # ensure numeric
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, mapping

# ----------------------
# Price-action primitives
# ----------------------
def compute_swing_points(df: pd.DataFrame, left=3, right=3):
    """
    Simple swing high/low detection.
    left/right define how many candles to compare.
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(left, n-right):
        if highs[i] == max(highs[i-left:i+right+1]):
            swing_high[i] = True
        if lows[i] == min(lows[i-left:i+right+1]):
            swing_low[i] = True
    df['swing_high'] = swing_high
    df['swing_low'] = swing_low
    return df

def detect_trend(df: pd.DataFrame, lookback=50):
    """
    Determine if market is uptrend/downtrend/range by comparing recent higher highs/lower lows.
    """
    recent = df.tail(lookback)
    hh = recent['high'].cummax().iloc[-1]
    ll = recent['low'].cummin().iloc[-1]
    start_h = recent['high'].iloc[0]
    start_l = recent['low'].iloc[0]
    # simple heuristic
    up_strength = (hh - start_h) / start_h
    down_strength = (start_l - ll) / start_l
    if up_strength > 0.02 and up_strength > down_strength:
        return 'Uptrend'
    elif down_strength > 0.02 and down_strength > up_strength:
        return 'Downtrend'
    else:
        return 'Range'

def detect_support_resistance(df: pd.DataFrame, window=20, thresh=0.0025):
    """
    Find horizontal zones where price repeatedly bounces (simple pivot clustering).
    Returns list of dicts with zone center and count.
    """
    pivots = []
    highs = df['high'].rolling(window, center=True).max().dropna()
    lows = df['low'].rolling(window, center=True).min().dropna()
    levels = pd.concat([highs.rename('level'), lows.rename('level')]).dropna().values
    # cluster levels
    levels = np.array(levels)
    clusters = []
    used = np.zeros(len(levels), dtype=bool)
    for i, lv in enumerate(levels):
        if used[i]: continue
        cluster_vals = [lv]
        used[i] = True
        for j in range(i+1, len(levels)):
            if not used[j] and abs(levels[j]-lv)/lv < thresh:
                cluster_vals.append(levels[j]); used[j]=True
        clusters.append({'center': np.mean(cluster_vals), 'count': len(cluster_vals)})
    # sort by count
    clusters = sorted(clusters, key=lambda x: x['count'], reverse=True)
    return clusters[:10]

def detect_order_blocks(df: pd.DataFrame, lookback=50):
    """
    Very simplified order block detection:
    - A bullish order block: a bearish candle followed by strong upward move (several green candles).
    - A bearish order block: a bullish candle followed by strong downward move.
    This returns recent order blocks with their zone.
    """
    obs = []
    for i in range(1, len(df)-3):
        prev = df.iloc[i-1]
        cur = df.iloc[i]
        # bearish candle then rally
        if cur['close'] < cur['open'] and df['close'].iloc[i+1] > cur['open'] and df['close'].iloc[i+2] > df['close'].iloc[i+1]:
            # bullish order block (demand)
            zone_high = cur['high']
            zone_low = cur['low']
            obs.append({'type':'bull', 'index':i, 'zone':(zone_low, zone_high), 'strength':(df['close'].iloc[i+2]-cur['close'])})
        # bullish candle then drop
        if cur['close'] > cur['open'] and df['close'].iloc[i+1] < cur['open'] and df['close'].iloc[i+2] < df['close'].iloc[i+1]:
            zone_high = cur['high']
            zone_low = cur['low']
            obs.append({'type':'bear', 'index':i, 'zone':(zone_low, zone_high), 'strength':(cur['close']-df['close'].iloc[i+2])})
    # sort by recency and strength
    obs = sorted(obs, key=lambda x: (x['index'], x['strength']), reverse=True)
    return obs[:20]

def detect_fair_value_gaps(df: pd.DataFrame, lookback=200):
    """
    Fair value gap (FVG) = an imbalance where a candle's body doesn't overlap with prior candle bodies.
    We'll detect simple 3-candle FVGs: e.g., for bullish FVG: prior candle high < next candle low.
    """
    fvg = []
    for i in range(1, len(df)-1):
        a = df.iloc[i-1]; b = df.iloc[i]; c = df.iloc[i+1]
        # bullish FVG: gap between a.high and b.low (rare), simplified
        if a['high'] < c['low']:
            fvg.append({'type':'bull', 'index':i, 'zone':(a['high'], c['low'])})
        if a['low'] > c['high']:
            fvg.append({'type':'bear', 'index':i, 'zone':(c['high'], a['low'])})
    return fvg

def detect_liquidity_zones(df: pd.DataFrame, lookback=200, wick_percentile=90):
    """
    Liquidity zones: price levels where many wicks cluster (frequent highs or lows).
    We'll detect levels from top wick extremes.
    """
    upper_wicks = df['high'] - df[['open','close']].max(axis=1)
    lower_wicks = df[['open','close']].min(axis=1) - df['low']
    up_thresh = np.nanpercentile(upper_wicks.dropna(), wick_percentile)
    low_thresh = np.nanpercentile(lower_wicks.dropna(), wick_percentile)
    zones = []
    for i in range(len(df)):
        if upper_wicks.iloc[i] >= up_thresh:
            zones.append({'type':'upper_wick','price': df['high'].iloc[i], 'index':i})
        if lower_wicks.iloc[i] >= low_thresh:
            zones.append({'type':'lower_wick','price': df['low'].iloc[i], 'index':i})
    return zones[:50]

# ----------------------
# Signal generation (pure PA + SMC heuristics)
# ----------------------
def generate_signals(df: pd.DataFrame, order_blocks: List[Dict], fvgs: List[Dict], sw_zones: List[Dict], params: Dict) -> pd.DataFrame:
    """
    Combining multiple heuristics to create signals. Returns df with 'signal' column and
    a 'reason' text.
    Signals: 1 (long), -1 (short), 0 (none). We aim to make high-prob setups only.
    """
    df = df.copy()
    df['signal'] = 0
    df['reason'] = ''
    lookback = params.get('lookback', 100)
    rr = params.get('risk_reward', 2.0)
    for i in range(len(df)-1):
        # Check recent structure
        recent = df.iloc[max(0, i-lookback):i+1]
        trend = detect_trend(recent, lookback=min(len(recent), 50))
        # Check order block near price
        price = df['close'].iloc[i]
        long_score = 0; short_score = 0; reasons = []
        # 1) order block proximity
        for ob in order_blocks[:10]:
            low, high = ob['zone']
            dist = 0
            if low <= price <= high:
                dist = 0
            else:
                dist = min(abs(price-low), abs(price-high))
            # scale
            if dist <= params.get('block_zone_buffer', 0.5*(df['high'].mean()-df['low'].mean())):
                if ob['type']=='bull':
                    long_score += 2
                    reasons.append('price inside bullish order block')
                else:
                    short_score += 2
                    reasons.append('price inside bearish order block')
        # 2) fair value gaps
        for f in fvgs[:8]:
            a,b = f['zone']
            if a <= price <= b:
                if f['type']=='bull':
                    long_score += 1.5; reasons.append('in bullish FVG')
                else:
                    short_score += 1.5; reasons.append('in bearish FVG')
        # 3) liquidity zone (wick clusters) - if price near lower wick -> long bias
        for z in sw_zones[:10]:
            if abs(z['center'] - price) / price < 0.004:
                # small boost
                reasons.append('near support/resistance cluster')
                if z.get('type') == 'support':
                    long_score += 0.8
                else:
                    short_score += 0.8
        # 4) Trend context
        if trend == 'Uptrend':
            long_score += 0.5
        elif trend == 'Downtrend':
            short_score += 0.5
        # 5) Simple candle confirmation: bullish engulfing or strong bullish candle for long
        if i-1 >= 0:
            prev = df.iloc[i-1]; cur = df.iloc[i]
            # bullish engulfing
            if (cur['close'] > cur['open']) and (prev['close'] < prev['open']) and (cur['close'] > prev['open']):
                long_score += 1; reasons.append('bullish engulfing')
            if (cur['close'] < cur['open']) and (prev['close'] > prev['open']) and (cur['close'] < prev['open']):
                short_score += 1; reasons.append('bearish engulfing')
        # rule to signal
        if (long_score >= params.get('long_threshold', 3.0)) and (long_score > short_score):
            df.at[df.index[i], 'signal'] = 1
            df.at[df.index[i], 'reason'] = '; '.join(reasons) or 'long_score'
        elif (short_score >= params.get('short_threshold', 3.0)) and (short_score > long_score):
            df.at[df.index[i], 'signal'] = -1
            df.at[df.index[i], 'reason'] = '; '.join(reasons) or 'short_score'
    return df

# ----------------------
# Backtester
# ----------------------
def backtest_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    For each signal row, simulate entry at next candle open (or next candle close),
    set stoploss based on orderblock/structure or fixed pct, take profit using RR.
    Returns logs DataFrame with details and stats.
    """
    logs = []
    risk_pct = params.get('risk_pct', 0.01)  # 1% default risk relative to price if no ob
    rr = params.get('risk_reward', 2.0)
    max_holding = params.get('max_holding_days', 20)
    for idx in df.index:
        row = df.loc[idx]
        if row['signal'] == 0: continue
        # entry at next candle's open if exists
        try:
            nxt = df.loc[idx+1]
        except KeyError:
            continue
        entry_price = float(nxt['open'])
        # default stoploss: recent swing low/high depending on direction
        lookback = params.get('sl_lookback', 10)
        recent = df.loc[max(df.index[0], idx-lookback):idx]
        if row['signal'] == 1:
            sl = float(recent['low'].min())
            if np.isnan(sl) or sl >= entry_price:
                sl = entry_price * (1 - risk_pct)
            target = entry_price + (entry_price - sl) * rr
        else:
            sl = float(recent['high'].max())
            if np.isnan(sl) or sl <= entry_price:
                sl = entry_price * (1 + risk_pct)
            target = entry_price - (sl - entry_price) * rr
        # simulate through next candles until hit SL or TP or max holding
        outcome = None
        exit_price = None
        exit_index = None
        for j in range(idx+1, min(idx+1+max_holding, df.index[-1]+1)):
            candle = df.loc[j]
            # for long: if low <= sl -> SL hit, if high >= target -> target hit
            if row['signal'] == 1:
                if candle['low'] <= sl:
                    exit_price = sl; outcome='SL'; exit_index=j; break
                if candle['high'] >= target:
                    exit_price = target; outcome='TP'; exit_index=j; break
            else:
                if candle['high'] >= sl:
                    exit_price = sl; outcome='SL'; exit_index=j; break
                if candle['low'] <= target:
                    exit_price = target; outcome='TP'; exit_index=j; break
        # if none hit until max_holding, exit at last close
        if outcome is None:
            last = df.loc[min(idx+max_holding, df.index[-1])]
            exit_price = last['close']; exit_index = last.name; outcome='TimedExit'
        pnl = (exit_price - entry_price) if row['signal']==1 else (entry_price - exit_price)
        points = pnl
        pct = pnl/entry_price*100
        holding_days = exit_index - (idx+1) + 1
        logs.append({
            'signal_index': idx,
            'signal_date': df.loc[idx,'date'],
            'direction': 'LONG' if row['signal']==1 else 'SHORT',
            'entry_date': df.loc[idx+1,'date'] if (idx+1 in df.index) else np.nan,
            'entry_price': entry_price,
            'exit_date': df.loc[exit_index,'date'] if exit_index in df.index else np.nan,
            'exit_price': exit_price,
            'outcome': outcome,
            'points': points,
            'pct': pct,
            'holding_bars': holding_days,
            'reason': row.get('reason','')
        })
    logs_df = pd.DataFrame(logs)
    if logs_df.empty:
        return logs_df
    # aggregate stats
    logs_df['win'] = logs_df['outcome'].apply(lambda x: 1 if x=='TP' else 0)
    return logs_df

# ----------------------
# UI Layout
# ----------------------
st.title("🧠 Swing Algo Trading Platform — Price Action & Smart Money Concepts")
st.markdown("Upload OHLC(V) historical data. The app auto-maps columns, runs EDA, detects price-action constructs, generates signals, and backtests them. Everything shown and downloadable.")

with st.sidebar:
    st.header("Run Controls")
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx','xls'], accept_multiple_files=False)
    st.write("Or load sample built-in dataset for quick test:")
    if st.button("Load sample (synthetic)"):
        # create synthetic dataset
        rng = pd.date_range(end=datetime.now(), periods=500, freq='D')
        price = np.cumsum(np.random.randn(len(rng))*2)+100
        df_sample = pd.DataFrame({'Date':rng, 'Open':price + np.random.randn(len(rng))*0.5, 'High':price + np.random.randn(len(rng))*1.5 + 1, 'Low':price - np.random.randn(len(rng))*1.5 - 1, 'Close':price + np.random.randn(len(rng))*0.5, 'Volume':np.random.randint(100,1000,len(rng))})
        uploaded = io.BytesIO()
        df_sample.to_csv(uploaded, index=False)
        uploaded.seek(0)
    st.header("Backtest / Signal params")
    risk_reward = st.number_input("Risk:Reward", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    risk_pct = st.number_input("Default risk pct (if no structure) e.g., 0.01 = 1%", min_value=0.0001, max_value=0.1, value=0.01, step=0.001, format="%.4f")
    long_thresh = st.number_input("Long score threshold", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    short_thresh = st.number_input("Short score threshold", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    max_holding = st.number_input("Max holding bars (days/candles)", min_value=1, max_value=1000, value=20)

# Main area
if uploaded is not None:
    try:
        # read file
        st.info("Loading file and auto-mapping columns...")
        if isinstance(uploaded, io.BytesIO) or hasattr(uploaded, 'read'):
            uploaded.seek(0)
        if hasattr(uploaded, "name") and uploaded.name.endswith(('xls','xlsx')):
            df_raw = pd.read_excel(uploaded)
        else:
            df_raw = pd.read_csv(uploaded)
        st.success(f"File loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    except Exception as e:
        st.exception(e)
        st.stop()

    # mapping
    df_mapped, mapping = auto_map_columns(df_raw)

    # quick EDA
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Summary")
        st.write(f"Date range: {df_mapped['date'].min()} → {df_mapped['date'].max()}")
        st.write("Columns after mapping:", list(df_mapped.columns))
        st.write(df_mapped[['open','high','low','close','volume']].describe().T)
        missing = df_mapped.isna().sum()
        st.write("Missing values:", missing[missing>0].to_dict())
    with col2:
        st.subheader("Price chart (interactive)")
        fig = go.Figure(data=[go.Candlestick(x=df_mapped['date'], open=df_mapped['open'], high=df_mapped['high'], low=df_mapped['low'], close=df_mapped['close'])])
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # compute primitives
    st.header("Detecting Price-Action Constructs")
    with st.spinner("Computing swing points, support/resistance, order blocks..."):
        df_pa = compute_swing_points(df_mapped.copy(), left=3, right=3)
        trend_now = detect_trend(df_pa, lookback=100)
        sr_zones = detect_support_resistance(df_pa, window=20)
        order_blocks = detect_order_blocks(df_pa)
        fvgs = detect_fair_value_gaps(df_pa)
        wick_zones = detect_liquidity_zones(df_pa)
    st.success("Detection complete.")
    st.write(f"Market context (last 100 bars): **{trend_now}**")

    st.subheader("Top Support/Resistance clusters (center, count)")
    if sr_zones:
        st.write(pd.DataFrame(sr_zones))
    else:
        st.write("No strong clusters detected.")

    st.subheader("Recent Order Blocks (top 10)")
    st.write(pd.DataFrame(order_blocks[:10]))

    st.subheader("Fair Value Gaps (sample)")
    st.write(pd.DataFrame(fvgs[:8]))

    st.subheader("Liquidity (wick) zones sample")
    st.write(pd.DataFrame(wick_zones[:12]))

    # signal generation
    st.header("Signal Generation")
    params = {
        'risk_reward': risk_reward,
        'risk_pct': risk_pct,
        'long_threshold': long_thresh,
        'short_threshold': short_thresh,
        'max_holding_days': int(max_holding),
        'sl_lookback': 10,
        'block_zone_buffer': 0.5*(df_pa['high'].mean()-df_pa['low'].mean())
    }

    with st.spinner("Generating signals using combined heuristics..."):
        # create a simplified support/resistance structure list for proximity checks
        sw_zones = [{'center': z['center'], 'type': 'support' if i%2==0 else 'resistance'} for i,z in enumerate(sr_zones)]
        df_signals = generate_signals(df_pa, order_blocks, fvgs, sw_zones, params)
    st.success("Signals generated.")

    # show signals
    sigs = df_signals[df_signals['signal']!=0].copy()
    st.subheader(f"Signals detected: {len(sigs)}")
    if not sigs.empty:
        st.dataframe(sigs[['date','signal','reason']].tail(20))
        # visualization of last signals on chart
        fig2 = go.Figure(data=[go.Candlestick(x=df_signals['date'], open=df_signals['open'], high=df_signals['high'], low=df_signals['low'], close=df_signals['close'])])
        sig_markers = df_signals[df_signals['signal']!=0]
        fig2.add_trace(go.Scatter(x=sig_markers['date'], y=sig_markers['close'], mode='markers', marker=dict(size=10, color=sig_markers['signal'].map({1:'green',-1:'red'})), name='signals'))
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No high-confidence signals found with current parameters.")

    # backtest
    st.header("Backtest Signals (historical simulation)")
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            logs_df = backtest_signals(df_signals, params)
        if logs_df.empty:
            st.warning("No trades simulated from signals. Try relaxing thresholds.")
        else:
            st.success(f"Backtest complete: {len(logs_df)} trades")
            # metrics
            wins = logs_df['win'].sum()
            total = len(logs_df)
            win_rate = wins/total*100
            avg_points = logs_df['points'].mean()
            total_points = logs_df['points'].sum()
            pnl = total_points
            st.metric("Total Trades", total)
            st.metric("Win Rate (%)", f"{win_rate:.2f}")
            st.metric("Total Points (sum)", f"{total_points:.4f}")
            st.metric("Avg Points per Trade", f"{avg_points:.4f}")

            st.subheader("Backtest trade logs (latest 200)")
            st.dataframe(logs_df.sort_values('signal_date', ascending=False).head(200))

            # download options
            csv = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download backtest logs CSV", data=csv, file_name='backtest_logs.csv', mime='text/csv')

            # show equity curve (cumulative points)
            logs_df = logs_df.sort_values('signal_date')
            logs_df['cum_points'] = logs_df['points'].cumsum()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=logs_df['signal_date'], y=logs_df['cum_points'], mode='lines+markers', name='Equity'))
            fig3.update_layout(height=300, title="Equity Curve (points)")
            st.plotly_chart(fig3, use_container_width=True)

            # Save logs to session state for downloads and live recommendation formatting
            st.session_state['last_backtest_logs'] = logs_df

    # Live recommendation (based on last close)
    st.header("Live Recommendation (based on latest candle close)")
    last_idx = df_signals.index[-1]
    last_row = df_signals.loc[last_idx]
    # We'll evaluate signal for the latest closed candle (index last_idx-1)
    latest_eval_index = last_idx-1 if last_idx-1 in df_signals.index else last_idx
    latest_row = df_signals.loc[latest_eval_index]
    if latest_row['signal'] != 0:
        st.success(f"Live Signal: {'LONG' if latest_row['signal']==1 else 'SHORT'}")
        st.write("Signal reason:", latest_row.get('reason',''))
        # build same format as backtest log but hypothetical execution next open
        try:
            hypothetical = backtest_signals(df_signals.loc[[latest_eval_index]].append(df_signals.loc[latest_eval_index+1:latest_eval_index+params.get('max_holding_days',20)+1]), params)
            if not hypothetical.empty:
                st.write(hypothetical.T.to_dict())
        except Exception:
            st.info("Hypothetical execution preview not available for this index (edge-case).")
    else:
        st.info("No live high-confidence signal on last candle with current parameters.")

    # Optimization skeleton
    st.header("Optimization (optional — random search skeleton)")
    if st.checkbox("Run quick random search (light)"):
        n_iters = st.number_input("Iterations", min_value=10, max_value=200, value=30)
        with st.spinner("Running random search..."):
            best = None
            pbar = st.progress(0)
            for i in range(n_iters):
                # randomize thresholds/risk_reward slightly
                rnd_params = params.copy()
                rnd_params['risk_reward'] = float(np.round(np.random.uniform(1.0,3.0),2))
                rnd_params['long_threshold'] = float(np.round(np.random.uniform(1.0,5.0),2))
                rnd_params['short_threshold'] = float(np.round(np.random.uniform(1.0,5.0),2))
                logs = backtest_signals(df_signals, rnd_params)
                if logs.empty:
                    score = -9999
                else:
                    score = logs['points'].sum()  # simple objective: maximize sum of points
                if best is None or score > best['score']:
                    best = {'score':score,'params':rnd_params}
                pbar.progress(int((i+1)/n_iters*100))
            st.write("Best random search result (simple objective: total points):")
            st.write(best)
    # debugging logs & download raw mapped data
    st.header("Debug / Download data")
    if st.checkbox("Show processed dataset (first 200 rows)"):
        st.dataframe(df_pa.head(200))
    csv_all = df_pa.to_csv(index=False).encode('utf-8')
    st.download_button("Download processed OHLCV CSV", data=csv_all, file_name='processed_ohlcv.csv', mime='text/csv')

    # final notes & limitations
    st.markdown("""
    ### Notes & Limitations
    - This is a **pure price-action / heuristic** engine intended as a strong foundation.  
    - Order blocks, FVGs, liquidity hunts are implemented with **simplified** but explainable heuristics — professional production systems often refine these with tick/orderflow data and more advanced filters.
    - Backtest assumptions: entry at **next candle open**, fixed SL based on recent swings or fallback risk_pct, TP via RR. You may want to modify to use intrabar fills or partial exits.
    - This app purposely avoids indicator libraries (as requested). You can extend it with volume profile, DOM/orderflow, or tick data for higher fidelity.
    - Always forward-test (paper) before live money. Markets change; historic outperformance is not guaranteed.
    """)

else:
    st.info("Upload a CSV or Excel file to start (or use the sample).")
