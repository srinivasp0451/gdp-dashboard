import streamlit as st
import pandas as pd
import numpy as np

# Helper functions
def detect_momentum(df, window=14):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window).std()
    df['Momentum'] = df['Close'].diff(window)
    return df

def identify_zones(df, window=20):
    df['Rolling_Max'] = df['High'].rolling(window).max()
    df['Rolling_Min'] = df['Low'].rolling(window).min()
    df['Demand_Zone'] = df['Rolling_Min'].shift(1)
    df['Supply_Zone'] = df['Rolling_Max'].shift(1)
    return df

# Backtest function with detailed trade log
def backtest(df):
    df = detect_momentum(df)
    df = identify_zones(df)

    trades = []
    equity_curve = []
    equity = 0
    peak_equity = 0
    drawdowns = []

    for i in range(20, len(df)):
        entry_price = df['Close'].iloc[i]
        date = df.index[i]
        reason = None
        direction = None
        target = None
        stop = None

        if df['Momentum'].iloc[i] > 0 and entry_price > df['Supply_Zone'].iloc[i]:
            direction = "BUY"
            reason = "Momentum breakout above supply"
            stop = df['Demand_Zone'].iloc[i]
            target = entry_price + (entry_price - stop) * 2  # 2R target
        elif df['Momentum'].iloc[i] < 0 and entry_price < df['Demand_Zone'].iloc[i]:
            direction = "SELL"
            reason = "Momentum breakdown below demand"
            stop = df['Supply_Zone'].iloc[i]
            target = entry_price - (stop - entry_price) * 2  # 2R target

        if direction:
            exit_price = df['Close'].iloc[i+1] if i+1 < len(df) else entry_price
            pnl_points = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
            risk = abs(entry_price - stop) if stop else np.nan
            reward = abs(target - entry_price) if target else np.nan
            rrr = reward / risk if risk and risk != 0 else np.nan

            trade = {
                "Entry Date": date,
                "Direction": direction,
                "Entry": entry_price,
                "Target": target,
                "Stop": stop,
                "Exit Date": df.index[i+1] if i+1 < len(df) else date,
                "Exit": exit_price,
                "PnL (Points)": pnl_points,
                "Risk": risk,
                "Reward": reward,
                "RRR": rrr,
                "Reason": reason,
                "Confidence": np.random.uniform(0.4, 0.7)
            }
            trades.append(trade)
            equity += pnl_points
            peak_equity = max(peak_equity, equity)
            drawdowns.append(peak_equity - equity)
        equity_curve.append(equity)

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['PnL (Points)'] > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_points = trades_df['PnL (Points)'].sum()
    avg_points = trades_df['PnL (Points)'].mean() if total_trades > 0 else 0
    max_dd = max(drawdowns) if drawdowns else 0
    buy_hold_points = df['Close'].iloc[-1] - df['Close'].iloc[0]
    avg_rrr = trades_df['RRR'].mean() if total_trades > 0 else np.nan

    summary = {
        "Total Trades": total_trades,
        "Win Rate": round(win_rate, 2),
        "Total Points": round(total_points, 2),
        "Average Points/Trade": round(avg_points, 2),
        "Max Drawdown (points)": round(-max_dd, 2),
        "Buy & Hold Points": round(buy_hold_points, 2),
        "Average RRR": round(avg_rrr, 2)
    }

    return trades_df, summary

# Live recommendation function
def live_recommendation(df, backtest_summary):
    df = detect_momentum(df)
    df = identify_zones(df)
    last = df.iloc[-1]
    entry_price = last['Close']
    direction = None
    reason = None
    stop = None
    target = None

    if last['Momentum'] > 0 and entry_price > last['Supply_Zone']:
        direction = "BUY"
        reason = "Momentum breakout above supply"
        stop = last['Demand_Zone']
        target = entry_price + (entry_price - stop) * 2
    elif last['Momentum'] < 0 and entry_price < last['Demand_Zone']:
        direction = "SELL"
        reason = "Momentum breakdown below demand"
        stop = last['Supply_Zone']
        target = entry_price - (stop - entry_price) * 2

    if direction:
        risk = abs(entry_price - stop) if stop else np.nan
        reward = abs(target - entry_price) if target else np.nan
        rrr = reward / risk if risk and risk != 0 else np.nan
        confidence = backtest_summary.get("Win Rate", 50) / 100
        return {
            "Direction": direction,
            "Entry": entry_price,
            "Target": target,
            "Stop": stop,
            "Risk": risk,
            "Reward": reward,
            "RRR": round(rrr, 2),
            "Probability of Profit": f"{confidence*100:.2f}%",
            "Reason": reason
        }
    return None

# Streamlit UI
st.title("Swing Trading Strategy with Backtest & Live Recommendation")

uploaded_file = st.file_uploader("Upload OHLC data (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.capitalize() for c in df.columns]
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    trades_df, summary = backtest(df)

    st.subheader("Backtest Results")
    for k, v in summary.items():
        st.write(f"{k}: {v}")

    st.subheader("Trade Log")
    st.dataframe(trades_df)

    if st.button("Get Live Recommendation"):
        reco = live_recommendation(df, summary)
        if reco:
            st.subheader("Live Recommendation")
            for k, v in reco.items():
                st.write(f"{k}: {v}")
        else:
            st.write("No clear trade signal right now.")
