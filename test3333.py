import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, poisson
from datetime import datetime, timedelta

# Function to dynamically fetch available index symbols from Yahoo Finance or a CSV file
def get_available_indices():
    available_indices = {
        "Nifty 50": "^NSEI",
        "Midcap Nifty": "NIFTY_MID_SELECT.NS",
        "Bank Nifty": "^NSEBANK",
        "Finnifty": "NIFTY_FIN_SERVICE.NS",
        "Sensex": "^BSESN",
        "Bankex": "BSE-BANK.BO"
    }
    return available_indices

# Function to fetch real historical data from Yahoo Finance dynamically based on selection
def fetch_index_data(ticker, period="1y"):
    index_data = yf.download(ticker, period=period)  # Fetch historical data for the selected period
    index_data.reset_index(inplace=True)
    return index_data

# Monte Carlo Simulation for next-day prediction
def monte_carlo_simulation(index_data, num_simulations):
    last_price = index_data['Close'].iloc[-1]
    returns = index_data['Close'].pct_change().dropna()  # Daily returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    simulated_prices = []
    for _ in range(num_simulations):
        simulated_price = last_price * np.exp(np.random.normal(mean_return, std_return))
        simulated_prices.append(simulated_price)
    
    return np.mean(simulated_prices)  # Return the average of all simulated prices

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
        next_return = -np.abs(np.random.normal(0, 0.01))  # Simulate a negative return
    elif next_state == 1:
        next_return = np.random.normal(0, 0.01)  # Simulate a neutral return
    else:
        next_return = np.abs(np.random.normal(0, 0.01))  # Simulate a positive return
    
    next_day_price = index_data['Close'].iloc[-1] * (1 + next_return)
    return next_day_price

# Statistical Confidence Intervals for next-day prediction
def statistical_confidence_intervals(index_data, confidence_level):
    returns = index_data['Close'].pct_change().dropna()
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)  # Calculate Z-score based on confidence level
    lower_bound = mean_return - z_score * std_return
    upper_bound = mean_return + z_score * std_return
    
    predicted_return = np.random.uniform(lower_bound, upper_bound)
    next_day_price = index_data['Close'].iloc[-1] * (1 + predicted_return)
    
    return next_day_price

# Option Pricing Model (Black-Scholes) for next-day prediction
def option_pricing_model(index_data, risk_free_rate, volatility=None):
    S = index_data['Close'].iloc[-1]  # Current stock price
    K = S  # Assume strike price = current price
    T = 1/252  # 1-day to expiry
    r = risk_free_rate
    if volatility is None:
        returns = index_data['Close'].pct_change().dropna()
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    d1 = (np.log(S/K) + (r + (volatility**2)/2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return S + call_price  # Simplified prediction

# Poisson Distribution for rare event prediction (like sudden rallies or drops)
def poisson_distribution(index_data, avg_event_rate):
    rare_event_count = poisson.rvs(mu=avg_event_rate)
    if rare_event_count > 0:
        event_magnitude = np.random.uniform(-0.05, 0.05)  # Up to ±5% change
    else:
        event_magnitude = 0
    
    next_day_price = index_data['Close'].iloc[-1] * (1 + event_magnitude)
    
    return next_day_price

# Streamlit UI
st.title("Dynamic Index Prediction App")

# Dropdown for selecting the index dynamically
indices = get_available_indices()
selected_index = st.selectbox("Select Index", list(indices.keys()))

# Input for the period of data (dynamic input)
period = st.text_input("Enter the data period (e.g., '1y', '6mo', '1mo')", "1y")

# Dropdown for selecting the prediction method dynamically
methods = {
    "Monte Carlo Simulation": monte_carlo_simulation,
    "Bayesian Inference": bayesian_inference,
    "Markov Chain": markov_chain,
    "Statistical Confidence Intervals": statistical_confidence_intervals,
    "Option Pricing Model": option_pricing_model,
    "Poisson Distribution": poisson_distribution
}
selected_method = st.selectbox("Select Prediction Method", list(methods.keys()))

# Additional dynamic inputs for specific methods
if selected_method == "Monte Carlo Simulation":
    num_simulations = st.number_input("Enter number of simulations", min_value=100, max_value=10000, value=1000)
elif selected_method == "Statistical Confidence Intervals":
    confidence_level = st.slider("Select confidence level", min_value=0.8, max_value=0.99, value=0.95)
elif selected_method == "Option Pricing Model":
    risk_free_rate = st.number_input("Enter risk-free rate", min_value=0.0, max_value=0.2, value=0.01)
elif selected_method == "Poisson Distribution":
    avg_event_rate = st.number_input("Enter average event rate", min_value=0.0, max_value=1.0, value=0.01)

# Button to trigger prediction
if st.button("Predict"):
    # Fetch data for the selected index dynamically
    index_data = fetch_index_data(indices[selected_index], period)
    
    # Based on the selected method, run the prediction dynamically
    if selected_method == "Monte Carlo Simulation":
        prediction = monte_carlo_simulation(index_data, num_simulations)
    elif selected_method == "Bayesian Inference":
        prediction = bayesian_inference(index_data)
    elif selected_method == "Markov Chain":
        prediction = markov_chain(index_data)
    elif selected_method == "Statistical Confidence Intervals":
        prediction = statistical_confidence_intervals(index_data, confidence_level)
    elif selected_method == "Option Pricing Model":
        prediction = option_pricing_model(index_data, risk_free_rate)
    elif selected_method == "Poisson Distribution":
        prediction = poisson_distribution(index_data, avg_event_rate)

    # Get the date for which the prediction is made
    prediction_date = (index_data['Date'].iloc[-1] + timedelta(days=1)).strftime('%Y-%m-%d')

    # Display the prediction and prediction date
    st.write(f"The predicted price for {selected_index} on {prediction_date} using {selected_method}: {prediction:.2f}")

    # Add insights based on prediction
    if prediction > index_data['Close'].iloc[-1]:
        insight = "The model suggests a potential upward movement for the next trading day."
    else:
        insight = "The model suggests a potential downward movement for the next trading day."

    st.write(insight)
