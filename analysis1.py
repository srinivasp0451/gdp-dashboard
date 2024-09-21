import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Streamlit App Title
st.title("NIFTY 50 Pre-Open Data Analysis")

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
    
    # Drop any rows with missing values to avoid mismatches
    df.dropna(subset=['Pre Open NIFTY 50'], inplace=True)

    # Calculate percentage change (returns)
    df['Returns'] = df['Pre Open NIFTY 50'].pct_change() * 100

    # Fill missing values that arise from pct_change (first row becomes NaN)
    df['Returns'].fillna(0, inplace=True)

    # Calculate moving average (60-second)
    df['MA_60'] = df['Pre Open NIFTY 50'].rolling(window=60).mean()

    # Fill missing values in moving average
    df['MA_60'].fillna(df['Pre Open NIFTY 50'], inplace=True)

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
    else:
        st.write("Not enough data for clustering.")
    
else:
    st.write("Please upload a CSV file to begin analysis.")
