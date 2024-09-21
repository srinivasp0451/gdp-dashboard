import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Function to calculate error metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, mse, rmse, mape

# Function to plot actual vs forecast
def plot_forecast(actual, forecast, title):
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual, label="Actual", marker='o')
    plt.plot(forecast.index, forecast, label="Forecast", marker='x')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(plt)

# Function to fetch historical data
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['Volatility_5'] = data['Returns'].rolling(window=5).std()
    return data.dropna()

# Function for similar day forecast
def similar_day_forecast(data, pattern_length=5):
    recent_pattern = data[['Returns', 'MA_5', 'MA_10', 'Volatility_5']].iloc[-pattern_length:].values
    historical_patterns = data[['Returns', 'MA_5', 'MA_10', 'Volatility_5']].rolling(window=pattern_length).apply(lambda x: np.linalg.norm(x - recent_pattern))
    most_similar_day_idx = historical_patterns.idxmin()
    predicted_day = data.iloc[most_similar_day_idx + pattern_length]
    return predicted_day

# Streamlit UI
st.title("Similar Day Forecast for Index")

# Dropdown for index selection
index_options = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'MIDCAP NIFTY': 'NIFTY_MID_SELECT.NS',
    'FIN NIFTY': 'NIFTY_FIN_SERVICE.NS',
    'BANKEX': 'BSE-BANK.BO'
}
selected_index = st.selectbox("Select Index", options=list(index_options.keys()))

# Date inputs for start and end date
end_date = st.date_input("End Date", pd.to_datetime('today'))
start_date = st.date_input("Start Date", pd.to_datetime('today') - pd.DateOffset(years=1))

# Button to forecast
if st.button("Run Forecast"):
    with st.spinner('Fetching data and forecasting...'):
        # Fetch historical data
        ticker = index_options[selected_index]
        data = fetch_data(ticker, start=start_date, end=end_date)

        # Check if forecast date is in the past or future
        # Convert end_date to pd.Timestamp for comparison
        if pd.Timestamp(end_date) < pd.Timestamp('today'):
            # Perform forecast on the selected end date
            predicted_day = similar_day_forecast(data)
            forecasted_close = predicted_day['Close']

            # Display results and evaluation metrics if forecast is in the past
            actual_close = data.loc[end_date, 'Close']
            mae, mse, rmse, mape = calculate_metrics([actual_close], [forecasted_close])
            
            # Show the results in a dataframe
            result_df = pd.DataFrame({
                'Date': [end_date],
                'Actual Close': [actual_close],
                'Forecasted Close': [forecasted_close],
                'MAE': [mae],
                'MSE': [mse],
                'RMSE': [rmse],
                'MAPE (%)': [mape]
            })
            st.write("Forecast Results:")
            st.write(result_df)
            
            # Plot Actual vs Forecast
            plot_forecast(data['Close'], pd.Series([forecasted_close], index=[end_date]), "Actual vs Forecast")

        else:
            # Perform forecast for the next day
            predicted_day = similar_day_forecast(data)
            forecasted_close = predicted_day['Close']

            # Show the forecast for the future date in a table
            forecast_df = pd.DataFrame({
                'Date': [end_date + pd.DateOffset(days=1)],
                'Forecasted Close': [forecasted_close]
            })
            st.write("Forecast for the next day:")
            st.write(forecast_df)
