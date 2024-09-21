import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
import yfinance as yf

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

def load_data(index, start_date, end_date):
    ticker_map = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "MIDCAP NIFTY": "NIFTY_MID_SELECT.NS",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "BANKEX": "BSE-BANK.BO"
    }
    
    ticker = ticker_map.get(index)
    if ticker:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data[['Close']].rename(columns={'Close': 'Close'})
    else:
        raise ValueError("Index not recognized.")

# Function to fit ARIMA model and forecast the next day
def arima_forecast(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))  # Adjust order as needed
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast[0]

# Function to calculate error metrics
def calculate_metrics(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if np.any(actual) else float('inf')
    return mae, mse, rmse, mape

# Function to plot actual vs forecast
def plot_forecast(actual, forecast, title):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Main Streamlit app code
st.title("Index Forecasting with ARIMA (Box-Jenkins) Method")

# User input for index selection
index_option = st.selectbox("Select Index", options=["NIFTY 50", "BANK NIFTY", "SENSEX", "MIDCAP NIFTY", "FINNIFTY", "BANKEX"])
start_date = st.date_input("Start Date", value=date.today().replace(year=date.today().year-1))
end_date = st.date_input("End Date", value=date.today())

# Button to trigger the forecast
if st.button("Forecast"):
    # Load data based on selected index and date range
    data = load_data(index_option, start_date, end_date)

    # Ensure end_date is properly formatted as pd.Timestamp
    end_date_ts = pd.Timestamp(end_date)

    if pd.Timestamp(end_date) < pd.Timestamp('today'):
        # Perform ARIMA forecast for the next day
        forecasted_close = arima_forecast(data)

        # Check if end_date exists in data
        if end_date_ts in data.index:
            actual_close = data.loc[end_date_ts, 'Close']
        else:
            # Handle missing date by choosing the closest available date using asof()
            closest_date = data.index.asof(end_date_ts)
            actual_close = data.loc[closest_date, 'Close']
            st.warning(f"No data available for {end_date_ts}. Using closest available date: {closest_date}.")

        # Compute evaluation metrics
        mae, mse, rmse, mape = calculate_metrics([actual_close], [forecasted_close])

        # Show the results in a dataframe
        result_df = pd.DataFrame({
            'Date': [closest_date],
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
        plot_forecast(data['Close'], pd.Series([forecasted_close], index=[closest_date]), "Actual vs Forecast")

    else:
        # Handle future forecast
        forecasted_close = arima_forecast(data)
        
        # Display forecasted values as a table
        st.write(f"Forecasted Close for {end_date_ts}:")
        st.write(pd.DataFrame({'Date': [end_date_ts], 'Forecasted Close': [forecasted_close]}))
