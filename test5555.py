import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Fetching Nifty 50 data from Yahoo Finance (adjust period and interval as needed)
st.title("Options scalping")
nifty_data = yf.download('^NSEI', period='1mo', interval='5m')
nifty_data.dropna(inplace=True)

# Calculate Bollinger Bands without TA-Lib
def bollinger_bands(df, n=20, std_dev=2):
    df['Middle Band'] = df['Close'].rolling(window=n).mean()
    df['Upper Band'] = df['Middle Band'] + (df['Close'].rolling(window=n).std() * std_dev)
    df['Lower Band'] = df['Middle Band'] - (df['Close'].rolling(window=n).std() * std_dev)
    return df

# Calculate RSI manually
def rsi(df, periods=14):
    close_delta = df['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -close_delta.clip(upper=0)

    avg_gain = up.rolling(window=periods).mean()
    avg_loss = down.rolling(window=periods).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Calculate MACD manually
def macd(df, fast=12, slow=26, signal=9):
    df['12_EMA'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['26_EMA'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['MACD Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

# Calculate VWAP manually
def vwap(df):
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    return df

# Apply all indicators
nifty_data = bollinger_bands(nifty_data)
nifty_data = rsi(nifty_data)
nifty_data = macd(nifty_data)
nifty_data = vwap(nifty_data)

# Backtesting Strategy with Loosened Criteria
class Backtest:
    def __init__(self, df):
        self.df = df
        self.trades = []  # Stores all trades
        self.balance = 100000  # Starting balance
        self.position = None  # None = no position, 'CE' = call option, 'PE' = put option

    def scalping_strategy(self):
        for i in range(1, len(self.df)):
            price = self.df['Close'].iloc[i]
            date = self.df.index[i]

            # Loosened Bullish signal (Buy CE)
            if (self.df['Close'].iloc[i] < self.df['Lower Band'].iloc[i] and
                self.df['RSI'].iloc[i] < 40 and  # Adjusted RSI from 30 to 40
                self.df['MACD'].iloc[i] > self.df['MACD Signal'].iloc[i]):

                if self.position is None:  # Enter position
                    self.position = 'CE'
                    self.entry_price = price
                    self.entry_date = date
                    print(f'Bought CE at {price} on {date}')

            # Loosened Bearish signal (Buy PE)
            elif (self.df['Close'].iloc[i] > self.df['Upper Band'].iloc[i] and
                  self.df['RSI'].iloc[i] > 60 and  # Adjusted RSI from 70 to 60
                  self.df['MACD'].iloc[i] < self.df['MACD Signal'].iloc[i]):

                if self.position is None:  # Enter position
                    self.position = 'PE'
                    self.entry_price = price
                    self.entry_date = date
                    print(f'Bought PE at {price} on {date}')

            # Exit logic (for both CE and PE)
            if self.position == 'CE' and (self.df['RSI'].iloc[i] > 70 or self.df['MACD'].iloc[i] < self.df['MACD Signal'].iloc[i]):
                self.exit_trade(price, date)
            elif self.position == 'PE' and (self.df['RSI'].iloc[i] < 30 or self.df['MACD'].iloc[i] > self.df['MACD Signal'].iloc[i]):
                self.exit_trade(price, date)

    def exit_trade(self, price, date):
        profit_loss = price - self.entry_price if self.position == 'CE' else self.entry_price - price
        self.balance += profit_loss
        self.trades.append({
            'Position': self.position,
            'Entry Price': self.entry_price,
            'Exit Price': price,
            'Entry Date': self.entry_date,
            'Exit Date': date,
            'Profit/Loss': profit_loss
        })
        print(f'Sold {self.position} at {price} on {date} | Profit/Loss: {profit_loss}')
        self.position = None  # Clear position

    def results(self):
        # Displaying the trades as a DataFrame
        trades_df = pd.DataFrame(self.trades)
        print("\nFinal Trades:\n", trades_df)
        print("\nFinal Balance: ₹", self.balance)
        return trades_df

# Backtesting instance
backtest = Backtest(nifty_data)
backtest.scalping_strategy()
trades_df = backtest.results()
st.dataframe(trades_df)
pnl = trades_df["Profit/Loss"].sum()
st.write("Net profit or Loss",pnl)
