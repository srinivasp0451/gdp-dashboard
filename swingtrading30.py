import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# ----------------------------- #
# Utility functions
# ----------------------------- #
def load_data(upload_file):
    df = pd.read_csv(upload_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        # surrogate volume
        df['Volume'] = df['Close'].pct_change().abs()*100000
    return df

def add_features(df, ema_short=10, ema_long=50, atr_period=14, rsi_period=14):
    df['EMA_short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    df['TR'] = df['High'] - df['Low']
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    delta = df['Close'].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.fillna(0, inplace=True)
    return df

def generate_labels(df, horizon=5, up_pct=1.0, down_pct=1.0):
    df['Future_Close'] = df['Close'].shift(-horizon)
    df['Return'] = ((df['Future_Close'] - df['Close']) / df['Close'])*100
    df['Label'] = np.where(df['Return']>=up_pct,1,
                           np.where(df['Return']<=-down_pct,-1,0))
    df.drop(['Future_Close','Return'],axis=1,inplace=True)
    return df

def prepare_features_labels(df):
    X = df[['Close','Open','High','Low','Volume','EMA_short','EMA_long','ATR','RSI']].values
    y = df['Label'].values
    return X, y

def walkforward_split(X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    return splits

def backtest_model(df, preds, horizon=5, atr_multiplier=1.5):
    trades = []
    for i in range(len(df)-horizon):
        row = df.iloc[i]
        if preds[i]==1:
            entry = row['Close']
            sl = entry - atr_multiplier*row['ATR']
            target = entry + atr_multiplier*row['ATR']
            exit_price = min(df['Low'].iloc[i:i+horizon].min(), sl)
            exit_price = target if df['High'].iloc[i:i+horizon].max()>=target else exit_price
            pnl = exit_price - entry
            trades.append({'Date':row['Date'], 'Side':'Long','Entry':entry,'Target':target,
                           'SL':sl,'Exit':exit_price,'PnL':pnl,'Reason':'Signal'})
        elif preds[i]==-1:
            entry = row['Close']
            sl = entry + atr_multiplier*row['ATR']
            target = entry - atr_multiplier*row['ATR']
            exit_price = max(df['High'].iloc[i:i+horizon].max(), sl)
            exit_price = target if df['Low'].iloc[i:i+horizon].min()<=target else exit_price
            pnl = entry - exit_price
            trades.append({'Date':row['Date'], 'Side':'Short','Entry':entry,'Target':target,
                           'SL':sl,'Exit':exit_price,'PnL':pnl,'Reason':'Signal'})
    trades_df = pd.DataFrame(trades)
    return trades_df

def summarize_backtest(trades_df, df):
    total_trades = len(trades_df)
    pos_trades = len(trades_df[trades_df['PnL']>0])
    neg_trades = total_trades - pos_trades
    acc = (pos_trades/total_trades*100) if total_trades>0 else 0
    total_pnl = trades_df['PnL'].sum()
    buy_hold = df['Close'].iloc[-1] - df['Close'].iloc[0]
    summary = {'Total trades': total_trades,
               'Positive trades': pos_trades,
               'Negative trades': neg_trades,
               'Accuracy': acc,
               'Strategy total PnL (points)': total_pnl,
               'Strategy PnL (% capital)': total_pnl/1000,
               'Buy & Hold points': buy_hold}
    return summary

def plot_heatmap(df):
    df['Month'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.to_period('Y')
    monthly = df.groupby('Month')['Close'].last().pct_change().fillna(0)*100
    yearly = df.groupby('Year')['Close'].last().pct_change().fillna(0)*100
    fig, axes = plt.subplots(1,2,figsize=(14,4))
    sns.heatmap(monthly.values.reshape(-1,1), annot=True, fmt=".2f", cmap='RdYlGn', ax=axes[0])
    axes[0].set_title('Monthly Returns (%)')
    axes[0].set_yticks(range(len(monthly)))
    axes[0].set_yticklabels(monthly.index.astype(str))
    sns.heatmap(yearly.values.reshape(-1,1), annot=True, fmt=".2f", cmap='RdYlGn', ax=axes[1])
    axes[1].set_title('Yearly Returns (%)')
    axes[1].set_yticks(range(len(yearly)))
    axes[1].set_yticklabels(yearly.index.astype(str))
    st.pyplot(fig)

# ----------------------------- #
# Streamlit UI
# ----------------------------- #
st.title("Professional Swing Trading Backtest & Live Signals")

st.sidebar.header("Input Options")
upload_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
horizon = st.sidebar.number_input("Prediction Horizon (bars)", value=5, min_value=1)
run_btn = st.sidebar.button("Run Backtest + Live Recommendation")

if run_btn and upload_file:
    progress_bar = st.progress(0)
    st.info("Step 1/5: Loading data...")
    df = load_data(upload_file)
    progress_bar.progress(10)

    st.info("Step 2/5: Feature engineering...")
    df = add_features(df)
    progress_bar.progress(30)

    st.info("Step 3/5: Label generation...")
    df = generate_labels(df,horizon=horizon)
    progress_bar.progress(40)

    st.info("Step 4/5: Walkforward training & prediction...")
    X, y = prepare_features_labels(df)
    splits = walkforward_split(X,y)
    preds = np.zeros(len(df))
    fold_num = 1
    for train_idx, test_idx in splits:
        st.info(f"Training fold {fold_num}/{len(splits)}")
        model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
        model.fit(X[train_idx],y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        fold_num += 1
    df['Pred'] = preds
    progress_bar.progress(70)

    st.info("Step 5/5: Backtesting...")
    trades_df = backtest_model(df, preds, horizon=horizon)
    summary = summarize_backtest(trades_df, df)
    progress_bar.progress(100)
    st.success("Completed!")

    # ---------------- UI Display ---------------- #
    st.subheader("Backtest Summary")
    for k,v in summary.items():
        st.write(f"{k}: {v}")

    st.subheader("Backtest Trades (Top 5 / Bottom 5)")
    st.dataframe(pd.concat([trades_df.head(), trades_df.tail()]))

    st.subheader("Full Trades Data")
    st.dataframe(trades_df)

    st.subheader("Original Data")
    st.dataframe(df)

    st.subheader("Monthly & Yearly Return Heatmaps")
    plot_heatmap(df)
