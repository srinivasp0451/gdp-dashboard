import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, poisson
from datetime import datetime, timedelta
import io

# Function to dynamically fetch available index symbols from Yahoo Finance
def get_available_indices():
    return {
        "Nifty 50": "^NSEI",
        "Midcap Nifty": "NIFTY_MID_SELECT.NS",
        "Bank Nifty": "^NSEBANK",
        "Finnifty": "NIFTY_FIN_SERVICE.NS",
        "Sensex": "^BSESN",
        "Bankex": "BSE-BANK.BO"
    }

# Function to fetch real historical data from Yahoo Finance
def fetch_index_data(ticker, start_date, end_date):
    index_data = yf.download(ticker, start=start_date, end=end_date)
    index_data.reset_index(inplace=True)
    return index_data

# Monte Carlo Simulation for next-day prediction
def monte_carlo_simulation(index_data, num_simulations):
    last_price = index_data['Close'].iloc[-1]
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    simulated_prices = []
    for _ in range(num_simulations):
        simulated_price = last_price * np.exp(np.random.normal(mean_return, std_return))
        simulated_prices.append(simulated_price)
    
    return np.mean(simulated_prices)

# Bayesian Inference for next-day prediction
def bayesian_inference(index_data):
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    next_day_return = np.random.normal(mean_return, std_return)
    next_day_price = index_data['Close'].iloc[-1] * (1 + next_day_return)
    
    return next_day_price

# Markov Chain for next-day prediction
def markov_chain(index_data):
    states = ['bearish', 'neutral', 'bullish']
    transition_matrix = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    
    current_return = index_data['Close'].pct_change().iloc[-1]
    if current_return < -0.01:
        current_state = 0  # bearish
    elif current_return < 0.01:
        current_state = 1  # neutral
    else:
        current_state = 2  # bullish
    
    next_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
    if next_state == 0:
        next_return = -np.abs(np.random.normal(0, 0.01))
    elif next_state == 1:
        next_return = np.random.normal(0, 0.01)
    else:
        next_return = np.abs(np.random.normal(0, 0.01))
    
    next_day_price = index_data['Close'].iloc[-1] * (1 + next_return)
    return next_day_price

# Statistical Confidence Intervals for next-day prediction
def statistical_confidence_intervals(index_data, confidence_level):
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = mean_return - z_score * std_return
    upper_bound = mean_return + z_score * std_return
    
    predicted_return = np.random.uniform(lower_bound, upper_bound)
    next_day_price = index_data['Close'].iloc[-1] * (1 + predicted_return)
    
    return next_day_price

# Option Pricing Model (Black-Scholes) for next-day prediction
def option_pricing_model(index_data, risk_free_rate, volatility=None):
    S = index_data['Close'].iloc[-1]
    K = S
    T = 1/252
    r = risk_free_rate
    if volatility is None:
        returns = index_data['Close'].pct_change().dropna()
        volatility = np.std(returns) * np.sqrt(252)
    
    d1 = (np.log(S/K) + (r + (volatility**2)/2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return S + call_price

# Poisson Distribution for rare event prediction
def poisson_distribution(index_data, avg_event_rate):
    rare_event_count = poisson.rvs(mu=avg_event_rate)
    if rare_event_count > 0:
        event_magnitude = np.random.uniform(-0.05, 0.05)
    else:
        event_magnitude = 0
    
    next_day_price = index_data['Close'].iloc[-1] * (1 + event_magnitude)
    
    return next_day_price

# Adjust date for business days
def get_next_business_day(start_date):
    next_day = start_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day

# Streamlit UI
st.title("Dynamic Index Prediction App")

# Dropdown for selecting the index dynamically
indices = get_available_indices()
indices["All"] = "All"
selected_index = st.selectbox("Select Index", list(indices.keys()))

# Input for custom date
prediction_date = st.date_input("Select Prediction Date", datetime.now().date())
end_date = prediction_date
start_date = prediction_date - timedelta(days=365)  # Use 1 year of data before the prediction date

# Dropdown for selecting the prediction method dynamically
methods = {
    "Monte Carlo Simulation": monte_carlo_simulation,
    "Bayesian Inference": bayesian_inference,
    "Markov Chain": markov_chain,
    "Statistical Confidence Intervals": statistical_confidence_intervals,
    "Option Pricing Model": option_pricing_model,
    "Poisson Distribution": poisson_distribution
}
methods["All"] = "All"
selected_method = st.selectbox("Select Prediction Method", list(methods.keys()))

# Dynamic input fields for specific methods
num_simulations = st.number_input("Enter number of simulations", min_value=100, max_value=10000, value=1000) if selected_method == "Monte Carlo Simulation" or selected_method == "All" else None
confidence_level = st.slider("Select confidence level", min_value=0.8, max_value=0.99, value=0.95) if selected_method == "Statistical Confidence Intervals" or selected_method == "All" else None
risk_free_rate = st.number_input("Enter risk-free rate", min_value=0.0, max_value=0.2, value=0.01) if selected_method == "Option Pricing Model" or selected_method == "All" else None
avg_event_rate = st.number_input("Enter average event rate", min_value=0.0, max_value=1.0, value=0.01) if selected_method == "Poisson Distribution" or selected_method == "All" else None

# Button to trigger prediction
if st.button("Predict"):
    # Dataframe to store summary results
    summary_data = []

    # Function to generate prediction for all selected options
    def generate_prediction(index_name, method_name, index_data):
        try:
            if method_name == "Monte Carlo Simulation":
                return monte_carlo_simulation(index_data, num_simulations)
            elif method_name == "Bayesian Inference":
                return bayesian_inference(index_data)
            elif method_name == "Markov Chain":
                return markov_chain(index_data)
            elif method_name == "Statistical Confidence Intervals":
                return statistical_confidence_intervals(index_data, confidence_level)
            elif method_name == "Option Pricing Model":
                return option_pricing_model(index_data, risk_free_rate)
            elif method_name == "Poisson Distribution":
                return poisson_distribution(index_data, avg_event_rate)
        except Exception as e:
            st.warning(f"Error processing {method_name} for {index_name}: {str(e)}")
            return None

    # Process "All" options for indices
    index_names = [selected_index] if selected_index != "All" else list(indices.keys())[:-1]  # exclude "All"
    
    # Process "All" options for methods
    method_names = [selected_method] if selected_method != "All" else list(methods.keys())[:-1]  # exclude "All"

    # Loop through all index-method combinations
    for index_name in index_names:
        try:
            index_data = fetch_index_data(indices[index_name], start_date, end_date)
        except Exception as e:
            st.warning(f"Data unavailable for {index_name}: {str(e)}")
            continue
        
        for method_name in method_names:
            prediction = generate_prediction(index_name, method_name, index_data)
            last_close_price = index_data['Close'].iloc[-1]
            trend = "Upward" if prediction > last_close_price else "Downward"
            reasons = "Favorable buying sentiment and market momentum." if trend == "Upward" else "Possible selling pressure and negative sentiment."

            summary_data.append({
                "Index": index_name,
                "Method": method_name,
                "Prediction Date": prediction_date.strftime("%Y-%m-%d"),
                "Last Close Price": last_close_price,
                "Predicted Price": prediction,
                "Trend": trend,
                "Reasons": reasons
            })
    
    # Display the results in a more readable format
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
