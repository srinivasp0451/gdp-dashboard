import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Import technical indicators from ta-lib or pandas_ta
import pandas_ta as ta

# Streamlit App Title
st.title("Enhanced NIFTY 50 Pre-Open Data Analysis with Clustering Insights")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Clean up the column names (fixing issues in headers)
    df.columns = ['DateTime', 'Pre Open NIFTY 50', 'NIFTY 50']  # Rename to proper names
    
    # Data Pre-processing
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Fill missing values in 'Pre Open NIFTY 50' from 'NIFTY 50'
    df['Pre Open NIFTY 50'].fillna(df['NIFTY 50'], inplace=True)

    # Drop rows where both columns are NaN (just in case)
    df.dropna(subset=['Pre Open NIFTY 50'], inplace=True)

    # Calculate percentage change (returns)
    df['Returns'] = df['Pre Open NIFTY 50'].pct_change() * 100

    # Fill missing values that arise from pct_change (first row becomes NaN)
    df['Returns'].fillna(0, inplace=True)

    # Calculate moving average (60-second)
    df['MA_60'] = df['Pre Open NIFTY 50'].rolling(window=60).mean()

    # Fill missing values in moving average
    df['MA_60'].fillna(df['Pre Open NIFTY 50'], inplace=True)

    # Display Descriptive Statistics
    st.subheader("Descriptive Statistics of Pre Open NIFTY 50 Prices")
    st.write(df['Pre Open NIFTY 50'].describe())

    # Box Plot for Pre Open NIFTY 50
    st.subheader("Box Plot of Pre Open NIFTY 50 Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(df['Pre Open NIFTY 50'], ax=ax)
    ax.set_xlabel('Pre Open NIFTY 50')
    st.pyplot(fig)

    # Volatility Clustering using K-Means

    df['Volatility'] = df['Returns'].abs()

    # Find the optimal number of clusters using the Elbow Method
    st.subheader("Optimal Clusters - Elbow Method")

    X = df[['Volatility']].dropna()
    
    if len(X) > 0:
        distortions = []
        K_range = range(2, 10)

        for k in K_range:
            kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
            distortions.append(kmeans_model.inertia_)

        # Plot Elbow Method
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K_range, distortions, 'bx-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method showing optimal k')
        st.pyplot(fig)

        # User selects optimal number of clusters
        k_opt = st.slider("Select the optimal number of clusters (k)", min_value=2, max_value=10, value=4)

        # Apply KMeans clustering with the selected number of clusters
        kmeans = KMeans(n_clusters=k_opt, random_state=0)
        df['Cluster'] = kmeans.fit_predict(X)

        # Scatter Plot with Clusters (Volatility)
        st.subheader(f"Volatility Clustering with {k_opt} Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df.index, df['Volatility'], c=df['Cluster'], cmap='viridis', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility (Returns)')
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

        # Cluster Insights with index levels, times, support/resistance
        st.subheader("Cluster Insights: Index Levels and Key Zones")
        
        for i in range(k_opt):
            cluster_data = df[df['Cluster'] == i]
            mean_volatility = cluster_data['Volatility'].mean()
            mean_price = cluster_data['Pre Open NIFTY 50'].mean()
            min_price = cluster_data['Pre Open NIFTY 50'].min()
            max_price = cluster_data['Pre Open NIFTY 50'].max()
            support_level = min_price
            resistance_level = max_price

            st.write(f"Cluster {i+1}:")
            st.write(f" - Average Price: {mean_price:.2f}")
            st.write(f" - Average Volatility: {mean_volatility:.2f}%")
            st.write(f" - Support Level: {support_level:.2f}")
            st.write(f" - Resistance Level: {resistance_level:.2f}")
            st.write(f" - **Key Time of Market Movement:** {cluster_data.index[0]}")

            # Display technical insights
            st.write("---")
        
        # Technical Indicators: Test Moving Averages, RSI, MACD
        st.subheader("Technical Indicators Evaluation")

        # Calculate RSI and MACD
        df['RSI'] = df['Pre Open NIFTY 50'].ta.rsi(length=14)
        df['MACD'] = df['Pre Open NIFTY 50'].ta.macd(fast=12, slow=26)['MACD_12_26_9']

        # Indicator Insights: Show for each cluster if RSI or MACD could predict a movement
        for i in range(k_opt):
            cluster_data = df[df['Cluster'] == i]
            rsi_signal = "Buy" if cluster_data['RSI'].iloc[-1] < 30 else "Sell" if cluster_data['RSI'].iloc[-1] > 70 else "Hold"
            macd_signal = "Buy" if cluster_data['MACD'].iloc[-1] > 0 else "Sell" if cluster_data['MACD'].iloc[-1] < 0 else "Hold"

            st.write(f"Cluster {i+1} Indicator Insights:")
            st.write(f" - RSI Suggests: **{rsi_signal}**")
            st.write(f" - MACD Suggests: **{macd_signal}**")
            st.write("---")
        
    else:
        st.write("Not enough data for clustering.")

else:
    st.write("Please upload a CSV file to begin analysis.")
