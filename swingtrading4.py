import pandas as pd
import streamlit as st
import talib as ta
from itertools import product

# =========================
# Utility Functions
# =========================

def validate_ohlcv(df):
    """Ensure required columns exist."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

def calc_indicators(df, atr_period=14):
    """Calculate ATR with NaN protection."""
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=atr_period)
    df.dropna(inplace=True)  # Ensure no NaN ATR values
    return df

def profit_factor(trades):
    """Calculate profit factor = gross profit / abs(gross loss)."""
    gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_loss = trades[trades['PnL'] < 0]['PnL'].sum()
    return gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

def run_backtest(df, atr_sl, atr_tp, ema_period, min_conf, mode):
    """Very simplified backtest logic placeholder."""
    trades = []
    for i, row in df.iterrows():
        # Example signal: Close > EMA
        ema = ta.EMA(df["Close"], timeperiod=ema_period)
        if pd.isna(row["ATR"]):
            continue  # Skip if ATR is missing

        if row["Close"] > ema[i]:
            entry = row["Close"]
            sl = entry - atr_sl * row["ATR"]
            tp = entry + atr_tp * row["ATR"]
            trades.append({"Entry": entry, "SL": sl, "TP": tp, "PnL": tp-entry})  # placeholder
    return pd.DataFrame(trades)

def optimize_params(train_df, param_grid):
    """Test all parameter combos and return best by profit factor."""
    best_pf = -float("inf")
    best_params = None
    for params in product(*param_grid.values()):
        combo = dict(zip(param_grid.keys(), params))
        trades = run_backtest(train_df, **combo)
        if trades.empty:
            continue
        pf = profit_factor(trades)
        if pf > best_pf:
            best_pf = pf
            best_params = combo
    return best_params

def continuation_check(df, last_params):
    """Check if yesterday's signal is still active today."""
    if last_params is None:
        return "No prior params to check."
    trades = run_backtest(df.tail(2), **last_params)
    if trades.empty:
        return "No trade from yesterday."
    return "Still Active"  # Placeholder — here you'd check SL/TP hit logic

# =========================
# Streamlit App
# =========================

st.title("Walk-Forward Backtest with Continuation Logic")

uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    validate_ohlcv(df)

    # Sliders auto-adjust to data length
    max_train = max(10, len(df) - 2)
    train_size = st.slider("Train window size (days)", 10, max_train, min(75, max_train))
    max_step = max(1, len(df) - 2)
    step_size = st.slider("Step size (days)", 1, max_step, min(5, max_step))

    df = calc_indicators(df, atr_period=14)

    # Parameter grid
    param_grid = {
        "atr_sl": [0.8, 1.0, 1.2, 1.5, 2.0],
        "atr_tp": [1.5, 2.0, 2.5, 3.0],
        "ema_period": [50, 100, 150, 200],
        "min_conf": [1, 2, 3],
        "mode": ["Both", "Long Only", "Short Only"]
    }

    # Walk-forward loop
    results = []
    last_params = None
    for start in range(0, len(df) - train_size, step_size):
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + step_size]
        if train_df.empty or test_df.empty:
            continue

        best_params = optimize_params(train_df, param_grid)
        last_params = best_params

        trades = run_backtest(test_df, **best_params)
        if not trades.empty:
            results.append(trades)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        st.write("Walk-forward results", all_results)
    else:
        st.warning("No trades generated in walk-forward.")

    # Continuation logic
    cont_status = continuation_check(df, last_params)
    st.info(f"Continuation Status: {cont_status}")
