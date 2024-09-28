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
        "Bankex": "BSE-BANK.BO",
        "All" : "All"
    }

# Function to fetch real historical data from Yahoo Finance with error handling
def fetch_index_data(ticker, start_date, end_date):
    try:
        index_data = yf.download(ticker, start=start_date, end=end_date)
        if index_data.empty:
            raise ValueError("No data available for the selected index.")
        index_data.reset_index(inplace=True)
        return index_data
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return None

# Monte Carlo Simulation for prediction
def monte_carlo_simulation(index_data, num_simulations):
    last_price = index_data['Close'].iloc[-1]
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    simulated_prices = []
    for _ in range(num_simulations):
        simulated_price = last_price * np.exp(np.random.normal(mean_return, std_return))
        simulated_prices.append(simulated_price)
    
    return np.mean(simulated_prices), mean_return, std_return

# Bayesian Inference for prediction
def bayesian_inference(index_data):
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    next_day_return = np.random.normal(mean_return, std_return)
    next_day_price = index_data['Close'].iloc[-1] * (1 + next_day_return)
    
    return next_day_price, mean_return, std_return

# Markov Chain for prediction
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
    return next_day_price, current_state, next_state

# Statistical Confidence Intervals for prediction
def statistical_confidence_intervals(index_data, confidence_level):
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = mean_return - z_score * std_return
    upper_bound = mean_return + z_score * std_return
    
    predicted_return = np.random.uniform(lower_bound, upper_bound)
    next_day_price = index_data['Close'].iloc[-1] * (1 + predicted_return)
    
    return next_day_price, lower_bound, upper_bound

# Option Pricing Model (Black-Scholes) for prediction
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
    
    return S + call_price, volatility

# Poisson Distribution for rare event prediction
def poisson_distribution(index_data, avg_event_rate):
    rare_event_count = poisson.rvs(mu=avg_event_rate)
    if rare_event_count > 0:
        event_magnitude = np.random.uniform(-0.05, 0.05)
    else:
        event_magnitude = 0
    
    next_day_price = index_data['Close'].iloc[-1] * (1 + event_magnitude)
    
    return next_day_price, event_magnitude

# Adjust date to next trading day if it's a holiday or weekend
def adjust_to_next_trading_day(selected_date):
    while selected_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        selected_date += timedelta(days=1)
    return selected_date

# Streamlit UI
st.title("Dynamic Index Prediction App")

# Dropdown for selecting the index dynamically (default to "All")
indices = get_available_indices()
selected_index = st.selectbox("Select Index", list(indices.keys()), index=len(indices)-1)  # Default to "All"

# Input for selecting the prediction date
prediction_date = st.date_input("Select Prediction Date", value=datetime.now())

# Adjust for non-trading days
adjusted_prediction_date = adjust_to_next_trading_day(prediction_date)

# Input for the period of data
period_days = st.slider("Select data period in days", min_value=30, max_value=365, value=365)

# Calculate start date based on the period
start_date = adjusted_prediction_date - timedelta(days=period_days)

# Dropdown for selecting the prediction method dynamically (default to "All")
methods = {
    "Monte Carlo Simulation": monte_carlo_simulation,
    "Bayesian Inference": bayesian_inference,
    "Markov Chain": markov_chain,
    "Statistical Confidence Intervals": statistical_confidence_intervals,
    "Option Pricing Model": option_pricing_model,
    "Poisson Distribution": poisson_distribution,
    "All" : "All"
}
selected_method = st.selectbox("Select Prediction Method", list(methods.keys()), index=len(methods)-1)  # Default to "All"

# Dynamic input fields for specific methods
num_simulations = st.number_input("Enter number of simulations", min_value=100, max_value=10000, value=1000) if selected_method == "Monte Carlo Simulation" else None
confidence_level = st.slider("Select confidence level", min_value=0.8, max_value=0.99, value=0.95) if selected_method == "Statistical Confidence Intervals" else None
risk_free_rate = st.number_input("Enter risk-free rate", min_value=0.0, max_value=0.2, value=0.01) if selected_method == "Option Pricing Model" else None
avg_event_rate = st.number_input("Enter average event rate", min_value=0.0, max_value=1.0, value=0.01) if selected_method == "Poisson Distribution" else None

# Button to trigger prediction
if st.button("Predict"):
    insights = []
    
    def generate_prediction(index_name, method_name, index_data):
        try:
            if method_name == "Monte Carlo Simulation":
                pred, mean_return, std_return = monte_carlo_simulation(index_data, num_simulations)
                insights.append(f"Using Monte Carlo Simulation, we expect a predicted price of {pred:.2f}. Mean return: {mean_return:.4f}, Standard deviation: {std_return:.4f}.")
            elif method_name == "Bayesian Inference":
                pred, mean_return, std_return = bayesian_inference(index_data)
                insights.append(f"Bayesian Inference suggests a predicted price of {pred:.2f}. Mean return: {mean_return:.4f}, Standard deviation: {std_return:.4f}.")
            elif method_name == "Markov Chain":
                pred, current_state, next_state = markov_chain(index_data)
                state_labels = ['Bearish', 'Neutral', 'Bullish']
                insights.append(f"Markov Chain model predicts {pred:.2f}. Current state: {state_labels[current_state]}, Next state: {state_labels[next_state]}.")
            elif method_name == "Statistical Confidence Intervals":
                pred, lower, upper = statistical_confidence_intervals(index_data, confidence_level)
                insights.append(f"Confidence Interval prediction: {pred:.2f} with range [{lower:.4f}, {upper:.4f}].")
            elif method_name == "Option Pricing Model":
                pred, volatility = option_pricing_model(index_data, risk_free_rate)
                insights.append(f"Option Pricing Model predicts {pred:.2f} with volatility: {volatility:.4f}.")
            elif method_name == "Poisson Distribution":
                pred, magnitude = poisson_distribution(index_data, avg_event_rate)
                insights.append(f"Poisson Distribution predicts {pred:.2f} with event magnitude: {magnitude:.4f}.")
        except Exception as e:
            st.warning(f"Failed to predict for {index_name} using {method_name}. Reason: {str(e)}")
    
    # Fetch data for the selected index
    if selected_index == "All":
        for index_name, ticker in indices.items():
            index_data = fetch_index_data(ticker, start_date, adjusted_prediction_date)
            if index_data is not None:
                for method_name in methods.keys():
                    generate_prediction(index_name, method_name, index_data)
    else:
        ticker = indices[selected_index]
        index_data = fetch_index_data(ticker, start_date, adjusted_prediction_date)
        if index_data is not None:
            if selected_method == "All":
                for method_name in methods.keys():
                    generate_prediction(selected_index, method_name, index_data)
            else:
                generate_prediction(selected_index, selected_method, index_data)
    
    # Display the insights
    for insight in insights:
        st.write(insight)
