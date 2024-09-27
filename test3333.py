import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import timedelta
import io

# Function to fetch real historical data from Yahoo Finance
def fetch_index_data(ticker, start_date, end_date):
    index_data = yf.download(ticker, start=start_date, end=end_date)
    if index_data.empty:
        raise ValueError(f"No data available for {ticker} from {start_date} to {end_date}.")
    index_data.reset_index(inplace=True)
    return index_data

# Method implementations (for illustration, using simple predictions)
def monte_carlo_simulation(data, num_simulations=1000):
    # Perform a simple Monte Carlo simulation as placeholder
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.02))

def bayesian_inference(data):
    # Placeholder Bayesian method
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.015))

def markov_chain(data):
    # Placeholder Markov chain method
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.01))

def statistical_confidence_intervals(data, confidence_level=0.95):
    # Placeholder for confidence intervals method
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.012))

def option_pricing_model(data, risk_free_rate=0.05):
    # Placeholder for option pricing model
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.02))

def poisson_distribution(data, avg_event_rate=0.01):
    # Placeholder Poisson distribution method
    return data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.01))

# Function to generate prediction based on method
def generate_prediction(index_name, method_name, index_data):
    try:
        if method_name == "Monte Carlo Simulation":
            return monte_carlo_simulation(index_data)
        elif method_name == "Bayesian Inference":
            return bayesian_inference(index_data)
        elif method_name == "Markov Chain":
            return markov_chain(index_data)
        elif method_name == "Statistical Confidence Intervals":
            return statistical_confidence_intervals(index_data)
        elif method_name == "Option Pricing Model":
            return option_pricing_model(index_data)
        elif method_name == "Poisson Distribution":
            return poisson_distribution(index_data)
    except Exception as e:
        st.warning(f"Error processing {method_name} for {index_name}: {str(e)}")
        return None

# Function to explain trends with more details
def detailed_reasons(trend, index_name, last_close_price, prediction):
    if trend == "Upward":
        return (f"Favorable buying sentiment is observed in {index_name} based on historical price movements, "
                f"market momentum, and potential economic optimism. The predicted price of {prediction} exceeds "
                f"the last close price of {last_close_price}, indicating more buying interest.")
    else:
        return (f"Potential selling pressure is noticed for {index_name} as the predicted price of {prediction} is "
                f"lower than the last close price of {last_close_price}. This may be driven by negative market sentiment, "
                f"profit booking, or external economic factors.")

# Dictionary of indices with tickers
indices = {
    "NIFTY 50": "^NSEI",
    "Midcap Nifty": "NIFTY_MID_SELECT.NS",
    "Bank Nifty": "^NSEBANK",
    "Fin Nifty": "NIFTY_FIN_SERVICE.NS",
    "Sensex": "^BSESN",
    "Bankex": "BSE-BANK.BO"
}

# Available methods
methods = [
    "Monte Carlo Simulation",
    "Bayesian Inference",
    "Markov Chain",
    "Statistical Confidence Intervals",
    "Option Pricing Model",
    "Poisson Distribution"
]

# Streamlit layout
st.title("Index Prediction using Various Methods")

# Date input for testing models on past dates
prediction_date = st.date_input("Select a prediction date", datetime.date.today())
end_date = prediction_date + timedelta(days=1)  # Ensure data fetch covers business day for prediction

# Dropdowns for indices and methods
index_selection = st.multiselect("Select Index", options=["All"] + list(indices.keys()), default="All")
method_selection = st.multiselect("Select Methods", options=["All"] + methods, default="All")

# Button to predict
if st.button("Predict"):
    # Handle "All" selection for indices and methods
    if "All" in index_selection:
        index_names = list(indices.keys())
    else:
        index_names = index_selection
    
    if "All" in method_selection:
        method_names = methods
    else:
        method_names = method_selection

    summary_data = []
    
    # Iterate through selected indices and methods
    for index_name in index_names:
        start_date = prediction_date - timedelta(days=370)  # Request more than a year of data to cover business days
        
        try:
            index_data = fetch_index_data(indices[index_name], start_date, end_date)
        except ValueError as e:
            st.warning(f"Data unavailable for {index_name}: {str(e)}")
            continue
        
        for method_name in method_names:
            prediction = generate_prediction(index_name, method_name, index_data)
            if prediction is not None:
                last_close_price = index_data['Close'].iloc[-1]
                trend = "Upward" if prediction > last_close_price else "Downward"
                reasons = detailed_reasons(trend, index_name, last_close_price, prediction)
                
                summary_data.append({
                    "Index": index_name,
                    "Method": method_name,
                    "Prediction Date": prediction_date.strftime("%Y-%m-%d"),
                    "Last Close Price": last_close_price,
                    "Predicted Price": prediction,
                    "Trend": trend,
                    "Reasons": reasons
                })
    
    # Display the results in a readable format
    st.subheader("Prediction Summary")
    for entry in summary_data:
        st.markdown(f"**Index**: {entry['Index']}")
        st.markdown(f"**Method**: {entry['Method']}")
        st.markdown(f"**Prediction Date**: {entry['Prediction Date']}")
        st.markdown(f"**Last Close Price**: {entry['Last Close Price']}")
        st.markdown(f"**Predicted Price**: {entry['Predicted Price']}")
        st.markdown(f"**Trend**: {entry['Trend']}")
        st.markdown(f"**Reasons**: {entry['Reasons']}")
        st.markdown("---")

    # Download summary as Excel file
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        st.download_button(label="Download Summary", data=output.getvalue(), file_name="index_summary.xlsx")
