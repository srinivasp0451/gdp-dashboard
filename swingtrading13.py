"""
Streamlit app: Enhanced swing backtester + optimizer
- Keeps original backtest logic (entry on next bar open, exit via next bar H/L/close)
- Adds: volume-based indicator, indicator names with params included in indicator lists,
  input box for required accuracy and expected returns, random-search optimizer,
  top5/bottom5 trades, monthly-year heatmap of returns, and keeps logs format intact.

Usage:
1. Save this file and run: `streamlit run swing_backtester_with_optimizer.py`
2. Upload a CSV/XLSX with OHLCV columns (automatic column name detection: Open/High/Low/Close/Volume).

Notes:
- No external TA libs required.
- The optimizer performs random-search over reasonable parameter ranges and tries to find
  parameter sets that meet the user's expected accuracy (%) and expected returns (% vs buy-hold).

"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- column name constants (keeps compatibility with your previous functions) ---
O = "Open"
H = "High"
L = "Low"
C = "Close"
V = "Volume"

# ------------------------- helper utilities -------------------------

def standardize_columns(df):
    """Map common column name variants to Open/High/Low/Close/Volume and set datetime index if possible."""
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # open
    for s in ["open", "o"]:
        if s in cols:
            mapping[cols[s]] = O
            break
    for s in ["high", "h"]:
        if s in cols:
            mapping[cols[s]] = H
            break
    for s in ["low", "l"]:
        if s in cols:
            mapping[cols[s]] = L
            break
    for s in ["close", "c", "adj close", "closeprice"]:
        if s in cols:
            mapping[cols[s]] = C
            break
    for s in ["volume", "v"]:
        if s in cols:
            mapping[cols[s]] = V
            break
    df = df.rename(columns=mapping)
    # Force numeric conversion
    for col in [O, H, L, C, V]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # If there's a column that looks like date, try to set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            try:
                df.index = pd.to_datetime(df[date_cols[0]])
                df = df.drop(columns=[date_cols[0]])
            except Exception:
                pass
    return df


# ------------------------- indicator computations -------------------------

def compute_indicators(df, params):
    df = df.copy()
    # simple moving averages
    df[f"sma_{params['sma_fast']}"] = df[C].rolling(window=params['sma_fast']).mean()
    df[f"sma_{params['sma_slow']}"] = df[C].rolling(window=params['sma_slow']).mean()
    # exponential moving averages
    df[f"ema_{params['ema_fast']}"] = df[C].ewm(span=params['ema_fast'], adjust=False).mean()
    df[f"ema_{params['ema_slow']}"] = df[C].ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD
    ema_fast = df[C].ewm(span=params['macd_fast'], adjust=False).mean()
    ema_slow = df[C].ewm(span=params['macd_slow'], adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=params['macd_signal'], adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal

    # RSI
    delta = df[C].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/params['rsi_period'], adjust=False).mean()
    roll_down = down.ewm(alpha=1/params['rsi_period'], adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df[f"rsi_{params['rsi_period']}"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (on sma of bb_period)
    df['bb_middle'] = df[C].rolling(window=params['bb_period']).mean()
    df['bb_std'] = df[C].rolling(window=params['bb_period']).std()
    df['bb_upper'] = df['bb_middle'] + params['bb_k'] * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - params['bb_k'] * df['bb_std']

    # Momentum
    df[f"mom_{params['mom_period']}"] = df[C] - df[C].shift(params['mom_period'])

    # Stochastic %K/%D (fast)
    low_min = df[L].rolling(window=params['stoch_k']).min()
    high_max = df[H].rolling(window=params['stoch_k']).max()
    df['stoch_k'] = 100 * (df[C] - low_min) / (high_max - low_min + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(window=params['stoch_d']).mean()

    # ATR
    prev_close = df[C].shift(1)
    tr1 = df[H] - df[L]
    tr2 = (df[H] - prev_close).abs()
    tr3 = (df[L] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{params['atr_period']}"] = tr.rolling(window=params['atr_period']).mean()

    # OBV
    df['obv'] = ((df[C] - df[C].shift(1)).fillna(0) > 0).astype(int) * df[V].fillna(0)
    df['obv'] = df['obv'].cumsum()

    # VWAP-like (daily typical price weighted by volume over a rolling window)
    tp = (df[H] + df[L] + df[C]) / 3
    df['vwap'] = (tp * df[V]).rolling(window=params.get('vwap_window', 5)).sum() / (df[V].rolling(window=params.get('vwap_window', 5)).sum() + 1e-9)

    # CCI
    tp = (df[H] + df[L] + df[C]) / 3
    sma_tp = tp.rolling(window=params['cci_period']).mean()
    mad = tp.rolling(window=params['cci_period']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-9)

    # Volume SMA (new volume indicator)
    if V in df.columns:
        df[f"vol_sma_{params['vol_sma']}"] = df[V].rolling(window=params['vol_sma']).mean()
    else:
        df[f"vol_sma_{params['vol_sma']}"] = np.nan

    return df


# ------------------------- signal generation (confluence voting) -------------------------

def generate_confluence_signals(df_local, params, side="Both"):
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        # 1) SMA crossover (fast > slow -> long)
        sma_fast_name = f"sma_{params['sma_fast']}"
        sma_slow_name = f"sma_{params['sma_slow']}"
        sma_tag = f"SMA_{params['sma_fast']}_{params['sma_slow']}"
        sma_fast = row.get(sma_fast_name, np.nan)
        sma_slow = row.get(sma_slow_name, np.nan)
        if not np.isnan(sma_fast) and not np.isnan(sma_slow):
            if sma_fast > sma_slow:
                indicators_that_long.append(sma_tag)
            elif sma_fast < sma_slow:
                indicators_that_short.append(sma_tag)

        # 2) EMA crossover
        ema_f_name = f"ema_{params['ema_fast']}"
        ema_s_name = f"ema_{params['ema_slow']}"
        ema_tag = f"EMA_{params['ema_fast']}_{params['ema_slow']}"
        ema_f = row.get(ema_f_name, np.nan)
        ema_s = row.get(ema_s_name, np.nan)
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if ema_f > ema_s:
                indicators_that_long.append(ema_tag)
            elif ema_f < ema_s:
                indicators_that_short.append(ema_tag)

        # 3) MACD histogram positive -> long
        macd_tag = f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"
        if not np.isnan(row.get('macd_hist', np.nan)):
            if row['macd_hist'] > 0:
                indicators_that_long.append(macd_tag)
            elif row['macd_hist'] < 0:
                indicators_that_short.append(macd_tag)

        # 4) RSI strength
        rsi_tag = f"RSI_{params['rsi_period']}"
        rsi_val = row.get(f"rsi_{params['rsi_period']}", np.nan)
        if not np.isnan(rsi_val):
            if rsi_val < params.get('rsi_long_thresh', 35):
                indicators_that_long.append(rsi_tag)
            elif rsi_val > params.get('rsi_short_thresh', 65):
                indicators_that_short.append(rsi_tag)

        # 5) Bollinger bands
        bb_tag = f"BB_{params['bb_period']}_{params['bb_k']}"
        if not np.isnan(row.get('bb_upper', np.nan)) and not np.isnan(row.get('bb_lower', np.nan)):
            price = row.get(C, np.nan)
            if not np.isnan(price):
                if price < row['bb_lower']:
                    indicators_that_long.append(bb_tag)
                elif price > row['bb_upper']:
                    indicators_that_short.append(bb_tag)

        # 6) Momentum
        mom_tag = f"MOM_{params['mom_period']}"
        mom = row.get(f"mom_{params['mom_period']}", np.nan)
        if not np.isnan(mom):
            if mom > 0:
                indicators_that_long.append(mom_tag)
            elif mom < 0:
                indicators_that_short.append(mom_tag)

        # 7) Stochastic
        stoch_tag = f"STOCH_{params['stoch_k']}_{params['stoch_d']}"
        if not np.isnan(row.get('stoch_k', np.nan)) and not np.isnan(row.get('stoch_d', np.nan)):
            if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < params.get('stoch_long_thresh', 30):
                indicators_that_long.append(stoch_tag)
            elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > params.get('stoch_short_thresh', 70):
                indicators_that_short.append(stoch_tag)

        # 8) ADX direction
        adx_tag = f"ADX_{params['adx_period']}"
        if not np.isnan(row.get('adx', np.nan)) and not np.isnan(row.get('pdi', np.nan)) and not np.isnan(row.get('mdi', np.nan)):
            if row['adx'] > params.get('adx_strength', 20) and row['pdi'] > row['mdi']:
                indicators_that_long.append(adx_tag)
            elif row['adx'] > params.get('adx_strength', 20) and row['mdi'] > row['pdi']:
                indicators_that_short.append(adx_tag)

        # 9) OBV rising vs previous 3 bars
        i = df_calc.index.get_indexer([idx])[0]
        obv_tag = "OBV"
        obv_vote_long = False; obv_vote_short = False
        if i >= 3:
            recent = df_calc['obv'].iloc[i-2:i+1].mean()
            prev = df_calc['obv'].iloc[i-5:i-2].mean() if i>=5 else df_calc['obv'].iloc[max(0, i-5):i-2].mean()
            if recent > prev:
                obv_vote_long = True
            elif recent < prev:
                obv_vote_short = True
            if obv_vote_long: indicators_that_long.append(obv_tag)
            if obv_vote_short: indicators_that_short.append(obv_tag)

        # 10) VWAP
        vwap_tag = f"VWAP_{params.get('vwap_window',5)}"
        if not np.isnan(row.get('vwap', np.nan)) and not np.isnan(row.get(C, np.nan)):
            if row[C] > row['vwap']:
                indicators_that_long.append(vwap_tag)
            elif row[C] < row['vwap']:
                indicators_that_short.append(vwap_tag)

        # 11) CCI extremes
        cci_tag = f"CCI_{params['cci_period']}"
        if not np.isnan(row.get('cci', np.nan)):
            if row['cci'] < -100:
                indicators_that_long.append(cci_tag)
            elif row['cci'] > 100:
                indicators_that_short.append(cci_tag)

        # 12) Volume spike / above SMA (NEW)
        vol_tag = f"VOL_SMA_{params['vol_sma']}"
        if not np.isnan(row.get(f"vol_sma_{params['vol_sma']}", np.nan)) and not np.isnan(row.get(V, np.nan)):
            # if today's volume > vol_sma and price is up -> long vote; if price down -> short vote
            if row[V] > row.get(f"vol_sma_{params['vol_sma']}"):
                if row.get(C, np.nan) >= row.get(O, np.nan):
                    indicators_that_long.append(vol_tag)
                else:
                    indicators_that_short.append(vol_tag)

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # side filter
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        final_sig = vote_direction if total_votes >= params['min_confluence'] else 0

        votes.append({
            "index": idx,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "total_votes": total_votes,
            "direction": vote_direction,
            "signal": final_sig,
            "indicators_long": indicators_that_long,
            "indicators_short": indicators_that_short
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index("index")
    result = df_calc.join(votes_df)
    result["Signal"] = sig_series
    return result


# ------------------------- backtester (mostly original, minimal changes) -------------------------

def backtest_point_strategy(df_signals, params):
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    # buy & hold baseline (points)
    first_price = df_signals[C].iloc[0]
    last_price = df_signals[C].iloc[-1]
    buy_hold_points = last_price - first_price

    # iterate rows in order
    for i in range(len(df_signals)-1):
        row = df_signals.iloc[i]
        next_row = df_signals.iloc[i+1]
        sig = row["Signal"]

        # If not in position and there's a signal -> enter next day's open
        if (not in_pos) and sig != 0:
            entry_date = next_row.name  # next day's timestamp (index)
            entry_price = next_row[O] if not pd.isna(next_row[O]) else next_row[C]
            pos_side = sig  # 1 long, -1 short
            # set sl/target based on ATR
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params['target_atr_mult'] * atr_val
                sl = entry_price - params['sl_atr_mult'] * atr_val
            else:
                target = entry_price - params['target_atr_mult'] * atr_val
                sl = entry_price + params['sl_atr_mult'] * atr_val

            entry_details = {
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                # Keep original 'Indicators' field but include parameterized indicator names
                "Indicators": (row["indicators_long"] if pos_side==1 else row["indicators_short"]),
                "Confluences": row["total_votes"],
                # new: store full detail list for logs
                "Indicator_Details": (row["indicators_long"] if pos_side==1 else row["indicators_short"]) 
            }
            in_pos = True
            continue

        # If in position, check exit conditions for next_row
        if in_pos:
            h = next_row[H]; l = next_row[L]; closep = next_row[C]
            exit_price = None; exit_date = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row["Signal"] == -1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"
            else:  # short
                if not pd.isna(l) and l <= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row["Signal"] == 1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"

            # final day fallback
            if (i+1) == (len(df_signals)-1) and in_pos and exit_price is None:
                exit_price = closep
                reason = "End of data"

            if exit_price is not None:
                exit_date = next_row.name
                points = (exit_price - entry_price) if pos_side == 1 else (entry_price - exit_price)
                trades.append({
                    **entry_details,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Reason Exit": reason,
                    "Points": points,
                    "Hold Days": (exit_date.date() - entry_details["Entry Date"].date()).days
                })
                # reset
                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df["Points"] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades>0 else 0.0

    buy_hold_points = last_price - first_price if 'first_price' in locals() else (df_signals[C].iloc[-1] - df_signals[C].iloc[0])
    percent_vs_buyhold = (total_points / (abs(buy_hold_points)+1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        "total_points": total_points,
        "num_trades": num_trades,
        "wins": int(wins),
        "prob_of_profit": float(prob_of_profit),
        "buy_hold_points": buy_hold_points,
        "pct_vs_buyhold": float(percent_vs_buyhold)
    }
    return summary, trades_df


# ------------------------- optimizer (random search) -------------------------

def random_search_optimizer(df, user_inputs, params, n_iter=100):
    """Random search over a limited parameter grid to find candidates that meet user thresholds.
    user_inputs: dict with 'expected_accuracy_pct' and 'expected_returns_pct' (vs buy-hold)
    Returns: list of candidate dicts (with params and metrics)
    """
    candidates = []
    # parameter search spaces (tuned small for speed)
    param_space = {
        'sma_fast': [5,9,10,12],
        'sma_slow': [20,50,100],
        'ema_fast': [5,9,12],
        'ema_slow': [21,26,50],
        'rsi_period': [7,14,21],
        'mom_period': [5,10,14],
        'atr_period': [10,14,21],
        'min_confluence': [2,3,4,5],
        'target_atr_mult': [0.5,1.0,1.5,2.0,3.0],
        'sl_atr_mult': [0.5,1.0,1.5,2.0,3.0],
        'vol_sma': [5,10,20,50]
    }

    best_by_metric = []
    for it in range(n_iter):
        # sample params
        sampled = params.copy()
        for k, vals in param_space.items():
            sampled[k] = random.choice(vals)
        # generate signals and backtest
        df_sig = generate_confluence_signals(df, sampled, side=user_inputs.get('side', 'Both'))
        summary, trades_df = backtest_point_strategy(df_sig, sampled)
        # treat prob_of_profit*100 as accuracy
        acc_pct = summary['prob_of_profit'] * 100
        returns_pct = summary.get('pct_vs_buyhold', 0.0)
        candidate = {
            'params': sampled,
            'summary': summary,
            'trades': trades_df,
            'accuracy_pct': acc_pct,
            'returns_pct': returns_pct
        }
        # check if meets user thresholds
        if acc_pct >= user_inputs.get('expected_accuracy_pct', 0.0) and returns_pct >= user_inputs.get('expected_returns_pct', -1e9):
            candidates.append(candidate)
        best_by_metric.append(candidate)

    # sort candidates by returns then accuracy
    candidates_sorted = sorted(candidates, key=lambda x: (x['returns_pct'], x['accuracy_pct']), reverse=True)
    # also return best overall candidates even if thresholds not met
    best_overall = sorted(best_by_metric, key=lambda x: (x['returns_pct'], x['accuracy_pct']), reverse=True)[:10]
    return candidates_sorted, best_overall


# ------------------------- visualization helpers -------------------------

def plot_monthly_heatmap(trades_df):
    if trades_df is None or trades_df.empty:
        st.write("No trades to show for heatmap")
        return
    df = trades_df.copy()
    df['Exit Date'] = pd.to_datetime(df['Exit Date'])
    df['Year'] = df['Exit Date'].dt.year
    df['Month'] = df['Exit Date'].dt.month
    pivot = df.groupby(['Year', 'Month'])['Points'].sum().unstack(fill_value=0)
    # reindex months to 1..12
    all_months = list(range(1,13))
    pivot = pivot.reindex(columns=all_months, fill_value=0)
    plt.figure(figsize=(12, max(2, pivot.shape[0]*0.6)))
    sns.heatmap(pivot, annot=True, fmt='.1f', linewidths=0.5)
    plt.title('Monthly returns heatmap (Points)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    st.pyplot(plt.gcf())
    plt.clf()


# ------------------------- Streamlit app UI -------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced Swing Backtester + Optimizer")

    st.sidebar.header("Inputs")
    uploaded = st.sidebar.file_uploader("Upload OHLCV file (CSV or XLSX)", type=["csv","xlsx"]) 
    side = st.sidebar.selectbox("Trade Side", options=["Both","Long","Short"], index=0)
    random_iterations = st.sidebar.number_input("Random iterations (optimizer)", min_value=1, max_value=2000, value=200, step=10)
    expected_returns = st.sidebar.number_input("Expected returns vs buy-hold (%)", value=10.0)
    expected_accuracy = st.sidebar.number_input("Expected accuracy (%)", value=55.0)
    run_opt = st.sidebar.button("Run Optimize")
    run_backtest = st.sidebar.button("Run Backtest (current params)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Defaults used for indicators (you can change in code or via optimizer)")

    # default params
    default_params = {
        'sma_fast': 9,
        'sma_slow': 21,
        'ema_fast': 9,
        'ema_slow': 21,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14,
        'mom_period': 10,
        'bb_period': 20,
        'bb_k': 2,
        'stoch_k': 14,
        'stoch_d': 3,
        'atr_period': 14,
        'cci_period': 20,
        'vwap_window': 5,
        'min_confluence': 3,
        'target_atr_mult': 1.0,
        'sl_atr_mult': 1.0,
        'vol_sma': 20
    }

    params = default_params.copy()

    if uploaded is not None:
        try:
            if uploaded.type == 'text/csv' or str(uploaded.name).lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

        df = standardize_columns(df)
        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
                df = df.drop(columns=['Date'])
            else:
                # try to parse index
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    st.error('Could not parse dates - ensure first column is Date or index is datetime')
                    return

        st.success('File loaded and standardized')

        st.subheader('Preview data (first 5 rows)')
        st.dataframe(df.head())

        # run backtest or optimizer
        user_inputs = {
            'expected_accuracy_pct': expected_accuracy,
            'expected_returns_pct': expected_returns,
            'side': side
        }

        if run_opt:
            with st.spinner('Running optimizer (random search)...'):
                candidates, best_overall = random_search_optimizer(df, user_inputs, params, n_iter=int(random_iterations))
            if candidates:
                st.success(f"Found {len(candidates)} candidate(s) meeting thresholds. Showing top 5")
                for i, c in enumerate(candidates[:5]):
                    st.markdown(f"**Candidate {i+1}** — Accuracy: {c['accuracy_pct']:.1f}%, Returns vs BH: {c['returns_pct']:.1f}%")
                    st.json(c['params'])
                    st.write(c['summary'])
                    st.dataframe(c['trades'].head(20))
                    plot_monthly_heatmap(c['trades'])
            else:
                st.warning('No candidate met thresholds. Showing top overall results from random search:')
                for i, c in enumerate(best_overall[:5]):
                    st.markdown(f"**Candidate {i+1} (best)** — Accuracy: {c['accuracy_pct']:.1f}%, Returns vs BH: {c['returns_pct']:.1f}%")
                    st.json(c['params'])
                    st.write(c['summary'])

        if run_backtest:
            st.info('Running backtest with default params...')
            df_sig = generate_confluence_signals(df, params, side=side)
            summary, trades_df = backtest_point_strategy(df_sig, params)
            st.subheader('Backtest Summary')
            st.json(summary)
            if not trades_df.empty:
                st.subheader('Trades (all)')
                st.dataframe(trades_df)

                st.subheader('Top 5 winning trades')
                st.dataframe(trades_df.sort_values('Points', ascending=False).head(5))

                st.subheader('Bottom 5 losing trades')
                st.dataframe(trades_df.sort_values('Points', ascending=True).head(5))

                st.subheader('Monthly returns heatmap')
                plot_monthly_heatmap(trades_df)
            else:
                st.write('No trades generated with current params.')

        # show latest live-signal for last row
        df_sig_live = generate_confluence_signals(df.tail(200), params, side=side)
        last_row = df_sig_live.iloc[-1]
        st.subheader('Live signal (latest row)')
        st.write({'date': df_sig_live.index[-1], 'signal': int(last_row['Signal']), 'indicators': last_row['indicators_long'] if last_row['Signal']==1 else last_row['indicators_short'] if last_row['Signal']==-1 else []})

    else:
        st.info('Upload a file to begin. The app will keep your original backtest logic intact and only apply the requested enhancements.')


if __name__ == '__main__':
    main()
