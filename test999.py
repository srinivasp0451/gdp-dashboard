import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import streamlit as st

# Function to calculate EMA
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = calculate_ema(data, short_window)
    ema_long = calculate_ema(data, long_window)
    macd_line = ema_short - ema_long
    signal_line = calculate_ema(pd.DataFrame(macd_line), signal_window)
    return macd_line, signal_line

# Function to calculate ADX
def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_dm = (high.diff() > low.diff()).astype(float) * (high.diff()).clip(lower=0)
    minus_dm = (low.diff() > high.diff()).astype(float) * (low.diff()).clip(lower=0)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    adx = 100 * (plus_di - minus_di).rolling(window=period).mean()

    return adx, atr

# Function to calculate VWAP
def calculate_vwap(data):
    return (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

# Function to calculate support and resistance levels
def calculate_support_resistance(data, n=3):
    high = data['High']
    low = data['Low']

    resistance_levels = high.rolling(window=20).max().dropna().unique()[-n:]
    support_levels = low.rolling(window=20).min().dropna().unique()[-n:]

    return support_levels, resistance_levels

# Function to calculate trendlines using linear regression
def calculate_trendline(data, period=20):
    x = np.arange(len(data))
    trendline = np.polyfit(x[-period:], data['Close'].iloc[-period:], 1)
    return np.polyval(trendline, x[-1])

# Function to calculate buyer/seller strength
def calculate_buyer_seller_strength(data):
    buyers = (data['Close'] > data['Open']).sum()
    sellers = (data['Close'] < data['Open']).sum()
    return buyers, sellers

# Function to analyze selected index
def analyze_index(ticker):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)  # Last day for 5min data

    # Fetching 5-min data
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')

    # Calculating indicators
    data['EMA_9'] = calculate_ema(data, 9)
    data['EMA_21'] = calculate_ema(data, 21)
    data['RSI'] = calculate_rsi(data)
    data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['ADX'], data['ATR'] = calculate_adx(data)
    data['VWAP'] = calculate_vwap(data)

    # Calculate trendline
    trendline_value = calculate_trendline(data)

    # Calculate buyer/seller strength
    buyers, sellers = calculate_buyer_seller_strength(data)
    strength_ratio = buyers / (buyers + sellers) if (buyers + sellers) > 0 else 0

    # Current values
    last_close = data['Close'].iloc[-1]
    last_support = calculate_support_resistance(data)[0][-1]  # Most recent support level
    last_resistance = calculate_support_resistance(data)[1][-1]  # Most recent resistance level
    last_ema_9 = data['EMA_9'].iloc[-1]
    last_ema_21 = data['EMA_21'].iloc[-1]
    rsi_value = data['RSI'].iloc[-1]
    adx_value = data['ADX'].iloc[-1]
    atr_value = data['ATR'].iloc[-1]
    vwap_value = data['VWAP'].iloc[-1]
    macd_value = data['MACD'].iloc[-1]
    signal_value = data['Signal_Line'].iloc[-1]

    # Initialize signals
    buy_signal = False
    sell_signal = False

    # Strategy Logic
    # Buy conditions
    if (last_close > last_support) and (strength_ratio > 0.6) and (last_close > trendline_value) and (last_ema_9 > last_ema_21) and (rsi_value < 30) and (macd_value > signal_value) and (adx_value > 20):
        buy_signal = True

    # Sell conditions
    if (last_close < last_resistance) and (strength_ratio < 0.4) and (last_close < trendline_value) and (last_ema_9 < last_ema_21) and (rsi_value > 70) and (macd_value < signal_value) and (adx_value > 20):
        sell_signal = True

    # Recommendations
    recommendations = {
        'Index': ticker,
        'Current Index Trading': last_close,
        'Support Level': last_support,
        'Resistance Level': last_resistance,
        'Buy Signal': buy_signal,
        'Sell Signal': sell_signal,
        'Target (Buy)': last_close + (2 * atr_value),  # Example based on ATR
        'Target (Sell)': last_close - (2 * atr_value),  # Example based on ATR
        'Stop Loss (Buy)': last_close - (atr_value * 1.5),  # Example based on ATR
        'Stop Loss (Sell)': last_close + (atr_value * 1.5)   # Example based on ATR
    }

    return recommendations

# Streamlit app
def main():
    st.title("Index Analysis Tool")
    
    index_options = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "Midcap Nifty": "^NSEMDCP",
        "Bankex": "^NSEBANKEX",
        "All": "All"
    }

    selected_index = st.selectbox("Select an index to analyze:", list(index_options.keys()))

    if selected_index == "All":
        for index_name, ticker in index_options.items():
            if index_name != "All":
                recommendations = analyze_index(ticker)
                st.subheader(f"{recommendations['Index']} Analysis")
                st.write(f"Current Index Trading: {recommendations['Current Index Trading']}")
                st.write(f"Support Level: {recommendations['Support Level']}")
                st.write(f"Resistance Level: {recommendations['Resistance Level']}")

                if recommendations['Buy Signal']:
                    st.write("**Buy Recommendation:**")
                    st.write(f"Entry Price: {recommendations['Current Index Trading']}")
                    st.write(f"Target: {recommendations['Target (Buy)']}")
                    st.write(f"Stop Loss: {recommendations['Stop Loss (Buy)']}")
                else:
                    st.write("No clear buy recommendation.")

                if recommendations['Sell Signal']:
                    st.write("**Sell Recommendation:**")
                    st.write(f"Entry Price: {recommendations['Current Index Trading']}")
                    st.write(f"Target: {recommendations['Target (Sell)']}")
                    st.write(f"Stop Loss: {recommendations['Stop Loss (Sell)']}")
                else:
                    st.write("No clear sell recommendation.")
                st.markdown("---")
    else:
        selected_ticker = index_options[selected_index]
        recommendations = analyze_index(selected_ticker)

        st.subheader(f"{recommendations['Index']} Analysis")
        st.write(f"Current Index Trading: {recommendations['Current Index Trading']}")
        st.write(f"Support Level: {recommendations['Support Level']}")
        st.write(f"Resistance Level: {recommendations['Resistance Level']}")

        if recommendations['Buy Signal']:
            st.write("**Buy Recommendation:**")
            st.write(f"Entry Price: {recommendations['Current Index Trading']}")
            st.write(f"Target: {recommendations['Target (Buy)']}")
            st.write(f"Stop Loss: {recommendations['Stop Loss (Buy)']}")
        else:
            st.write("No clear buy recommendation.")

        if recommendations['Sell Signal']:
            st.write("**Sell Recommendation:**")
            st.write(f"Entry Price: {recommendations['Current Index Trading']}")
            st.write(f"Target: {recommendations['Target (Sell)']}")
            st.write(f"Stop Loss: {recommendations['Stop Loss (Sell)']}")
        else:
            st.write("No clear sell recommendation.")

if __name__ == "__main__":
    main()