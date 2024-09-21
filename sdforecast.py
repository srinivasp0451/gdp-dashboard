import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dummy function to load data based on selected index (replace with real data loading code)
def load_data(index, start_date, end_date):
    # Here, you'd normally load data from an API or CSV file
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    data = pd.DataFrame({
        'Close': np.random.randn(len(dates)) * 100 + 15000,
        'Returns': np.random.randn(len(dates)),
        'MA_5': np.random.randn(len(dates)),
        'MA_10': np.random.randn(len(dates)),
        'Volatility_5': np.random.randn(len(dates))
    }, index=dates)
    return data

# Function to forecast the next day's value based on similar day patterns
def similar_day_forecast(data, pattern_length=5):
    recent_pattern = data[['Returns', 'MA_5', 'MA_10', 'Volatility_5']].iloc[-pattern_length:].values
    distances = []
    
    for i in range(len(data) - pattern_length):
        historical_pattern = data[['Returns', 'MA_5', 'MA_10', 'Volatility_5']].iloc[i:i + pattern_length].values
        distance = np.linalg.norm(recent_pattern - historical_pattern)
        distances.append(distance)
    
    most_similar_day_idx = np.argmin(distances)
    predicted_day = data.iloc[most_similar_day_idx + pattern_length]
    
    return predicted_day

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
    plt.plot(actual, label="Actual")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Main Streamlit app code
st.title("Index Forecasting with Similar Day Analysis")

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
        # Perform forecast on the selected end date
        predicted_day = similar_day_forecast(data)
        forecasted_close = predicted_day['Close']

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
        predicted_day = similar_day_forecast(data)
        forecasted_close = predicted_day['Close']
        
        # Display forecasted values as a table
        st.write(f"Forecasted Close for {end_date_ts}:")
        st.write(pd.DataFrame({'Date': [end_date_ts], 'Forecasted Close': [forecasted_close]}))
