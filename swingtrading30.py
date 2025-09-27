import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# ------------------ UTILITY FUNCTIONS ------------------

def download_data(ticker, period="5y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.reset_index(inplace=True)
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            df['Volume'] = df['Close'].pct_change().abs() * 100000  # surrogate volume
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_features(df, ema_short=10, ema_long=50, atr_period=14, rsi_period=14):
    df['EMA_short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    df['ATR'] = df['High'].rolling(atr_period).max() - df['Low'].rolling(atr_period).min()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum'] = df['Close'].pct_change() * 100
    df.dropna(inplace=True)
    return df

def label_data(df, horizon=3, percentile=20):
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    up_thr = np.percentile(df['Future_Return'], 100 - percentile)
    down_thr = np.percentile(df['Future_Return'], percentile)
    df['Label'] = 0
    df.loc[df['Future_Return'] >= up_thr, 'Label'] = 1
    df.loc[df['Future_Return'] <= down_thr, 'Label'] = -1
    df.dropna(inplace=True)
    return df

def optimize_params(df):
    # simple grid search over EMA_short/long
    best_acc = -1
    best_params = {'ema_short':10, 'ema_long':50, 'atr_period':14, 'rsi_period':14}
    for ema_s in [5,10,15]:
        for ema_l in [30,50,100]:
            for atr_p in [10,14,20]:
                for rsi_p in [10,14,20]:
                    df_feat = generate_features(df.copy(), ema_s, ema_l, atr_p, rsi_p)
                    df_lab = label_data(df_feat)
                    if len(df_lab) < 50:
                        continue
                    X = df_lab[['EMA_short','EMA_long','ATR','RSI','Momentum']]
                    y = df_lab['Label']
                    model = RandomForestClassifier(n_estimators=50)
                    tscv = TimeSeriesSplit(n_splits=3)
                    accs = []
                    for train_idx, test_idx in tscv.split(X):
                        model.fit(X.iloc[train_idx], y.iloc[train_idx])
                        pred = model.predict(X.iloc[test_idx])
                        accs.append(accuracy_score(y.iloc[test_idx], pred))
                    avg_acc = np.mean(accs)
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_params = {'ema_short':ema_s, 'ema_long':ema_l, 'atr_period':atr_p, 'rsi_period':rsi_p}
    return best_params

def run_backtest(df, params, horizon=3):
    df_feat = generate_features(df.copy(), **params)
    df_lab = label_data(df_feat, horizon=horizon)
    X = df_lab[['EMA_short','EMA_long','ATR','RSI','Momentum']]
    y = df_lab['Label']
    model = RandomForestClassifier(n_estimators=100)
    tscv = TimeSeriesSplit(n_splits=4)
    all_trades = []
    fold_count = 1
    fold_summary = []

    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_df = df_lab.iloc[test_idx].copy()
        test_df['Pred'] = model.predict(X.iloc[test_idx])
        pnl = 0
        trades = []
        for idx, row in test_df.iterrows():
            if row['Pred'] != 0:
                entry = row['Close']
                if row['Pred'] == 1:
                    target = entry * 1.01  # +1% target
                    sl = entry * 0.995     # -0.5% stop
                else:
                    target = entry * 0.99  # -1% target
                    sl = entry * 1.005     # +0.5% stop
                exit_price = row['Close'].shift(-horizon)
                trade_pnl = (exit_price - entry) if row['Pred']==1 else (entry - exit_price)
                trades.append({
                    'Date': row['Date'], 'Side': 'Long' if row['Pred']==1 else 'Short',
                    'Entry': entry, 'Target': target, 'SL': sl,
                    'Exit': exit_price, 'PnL': trade_pnl
                })
                pnl += trade_pnl
        all_trades.extend(trades)
        fold_summary.append({'Fold':fold_count, 'PnL':pnl, 'Accuracy':accuracy_score(test_df['Label'], test_df['Pred'])})
        fold_count += 1
    trades_df = pd.DataFrame(all_trades)
    return trades_df, fold_summary

def plot_heatmap(trades_df, title="Monthly Returns Heatmap"):
    trades_df['Month'] = pd.to_datetime(trades_df['Date']).dt.to_period('M')
    monthly = trades_df.groupby('Month')['PnL'].sum().reset_index()
    heatmap_data = monthly.pivot_table(values='PnL', index=monthly['Month'].dt.year,
                                       columns=monthly['Month'].dt.month, fill_value=0)
    plt.figure(figsize=(12,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn', cbar_kws={'label':'PnL'})
    plt.title(title)
    st.pyplot(plt.gcf())

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="Universal Swing Algo", layout="wide")
st.title("Universal Swing Trading Algo System")

st.sidebar.header("Data Options")
data_source = st.sidebar.radio("Data source:", ['Upload CSV/Excel', 'YFinance Ticker'])
if data_source == 'Upload CSV/Excel':
    uploaded_file = st.sidebar.file_uploader("Upload file", type=['csv','xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
elif data_source == 'YFinance Ticker':
    ticker = st.sidebar.text_input("Ticker (e.g., ^NSEI)", value="^NSEI")
    period = st.sidebar.selectbox("Period", ['1y','2y','5y','10y'])
    interval = st.sidebar.selectbox("Interval", ['1d','1wk','1mo'])
    df = download_data(ticker, period=period, interval=interval)

if 'df' in locals() and not df.empty:
    st.subheader("Original Data")
    st.dataframe(df.head(5).append(df.tail(5)), height=250)

    if st.button("Run Swing Algo"):
        progress_bar = st.progress(0)
        stage = st.empty()

        stage.text("Optimizing Parameters...")
        best_params = optimize_params(df)
        progress_bar.progress(25)

        stage.text("Running Backtest + Walkforward...")
        trades_df, fold_summary = run_backtest(df, best_params)
        progress_bar.progress(75)

        stage.text("Generating Results & Heatmap...")
        st.subheader("Trade Results")
        st.dataframe(trades_df, height=400)

        st.subheader("Walkforward Summary")
        st.table(fold_summary)

        st.subheader("Monthly PnL Heatmap")
        plot_heatmap(trades_df)

        # Backtest summary
        total_trades = len(trades_df)
        positive_trades = len(trades_df[trades_df['PnL']>0])
        negative_trades = len(trades_df[trades_df['PnL']<=0])
        accuracy = positive_trades/total_trades*100 if total_trades>0 else 0
        total_pnl = trades_df['PnL'].sum()
        total_pnl_pct = total_pnl/df['Close'].iloc[0]*100

        st.subheader("Backtest Summary")
        st.write(f"Total trades: {total_trades}")
        st.write(f"Positive trades: {positive_trades}")
        st.write(f"Negative trades: {negative_trades}")
        st.write(f"Accuracy: {accuracy:.2f}%")
        st.write(f"Strategy total PnL (points): {total_pnl:.2f}")
        st.write(f"Strategy PnL (% of capital): {total_pnl_pct:.4f}%")
        st.write(f"Buy & Hold (first->last points): {df['Close'].iloc[-1]-df['Close'].iloc[0]:.2f}")

        progress_bar.progress(100)
        stage.text("Completed ✅")
