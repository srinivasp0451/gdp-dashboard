import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Streamlit App Title
st.title("Enhanced NIFTY 50 Pre-Open Data Analysis")

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

    # Volatility Clustering using K-Means with 7 Clusters
    df['Volatility'] = df['Returns'].abs()

    # Check if there is enough data for clustering
    kmeans_data = df[['Volatility']].dropna()
    if len(kmeans_data) > 0:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=7, random_state=0)
        df['Cluster'] = kmeans.fit_predict(kmeans_data)
        
        # Scatter Plot with Clusters (Volatility) with 7 Clusters
        st.subheader("Volatility Clustering with 7 Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df.index, df['Volatility'], c=df['Cluster'], cmap='viridis', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility (Returns)')
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

        # Display Cluster Insights
        st.subheader("Cluster Insights")
        cluster_centers = kmeans.cluster_centers_

        for i in range(7):
            cluster_data = df[df['Cluster'] == i]
            mean_volatility = cluster_data['Volatility'].mean()
            mean_price = cluster_data['Pre Open NIFTY 50'].mean()
            count = len(cluster_data)

            st.write(f"Cluster {i+1}:")
            st.write(f" - Average Price: {mean_price:.2f}")
            st.write(f" - Average Volatility: {mean_volatility:.2f}%")
            st.write(f" - Number of Instances: {count}")
            
            # Display specific patterns found in this cluster
            if mean_volatility > df['Volatility'].mean() + df['Volatility'].std():
                st.write(" - **High volatility cluster**: Market may be reacting to strong events.")
            else:
                st.write(" - **Low volatility cluster**: Stable movement, consolidation likely.")
            
            st.write("---")
            
        # Institutional Buying/Selling Approximation (large volatility spikes)
        st.subheader("Institutional Buying/Selling Detection")
        large_volatility = df[df['Volatility'] > df['Volatility'].mean() + df['Volatility'].std()]
        st.write("Possible Institutional Activity Detected at these Times:")
        st.dataframe(large_volatility[['Pre Open NIFTY 50', 'Volatility']])

        # Provide Insights on Institutional Activity
        st.subheader("Insights on Institutional Activity")
        if len(large_volatility) > 0:
            st.write(f"There were {len(large_volatility)} instances where high volatility (potential institutional buying/selling) occurred.")
            st.write("Institutional activity is often indicated by large spikes in volatility. These could represent large orders being executed, potentially signaling market-moving events.")
        else:
            st.write("No significant institutional activity detected based on volatility spikes.")
        
        # Trade Recommendations Based on Cluster Analysis
        st.subheader("Trade Recommendations for Next Day")
        st.write("Based on cluster behavior and observed patterns, here are some recommendations:")
        
        # Sample trade recommendations based on cluster insights:
        high_vol_cluster = df.groupby('Cluster').mean()['Volatility'].idxmax()
        low_vol_cluster = df.groupby('Cluster').mean()['Volatility'].idxmin()
        
        st.write(f" - Clusters with high volatility (e.g., Cluster {high_vol_cluster + 1}) may indicate potential **trend reversal** or market-moving news. Monitor these closely for breakout or breakdown.")
        st.write(f" - Clusters with low volatility (e.g., Cluster {low_vol_cluster + 1}) typically show **consolidation**; watch for potential breakouts after consolidation periods.")
        st.write(f" - If institutional activity is detected in a high-volatility cluster, it might suggest **large buying or selling pressure**. Consider trading in the direction of the trend.")
        
    else:
        st.write("Not enough data for clustering.")

else:
    st.write("Please upload a CSV file to begin analysis.")
