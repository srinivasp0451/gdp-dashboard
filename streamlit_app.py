import streamlit as st
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
import yfinance as yf
import pandas as pd
import io

# Function to calculate pivot points
def calculate_pivot_points(data):
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Support1'] = 2 * data['Pivot'] - data['High']
    data['Resistance1'] = 2 * data['Pivot'] - data['Low']
    data['Support2'] = data['Pivot'] - (data['High'] - data['Low'])
    data['Resistance2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['Support3'] = data['Support1'] - (data['High'] - data['Low'])
    data['Resistance3'] = data['Resistance1'] + (data['High'] - data['Low'])
    return data[['Pivot', 'Support1', 'Resistance1', 'Support2', 'Resistance2', 'Support3', 'Resistance3']]

# Function to get support and resistance levels for today
def get_support_resistance_today(ticker):
    data = yf.download(ticker, period='1d', interval='1d')
    
    if len(data) < 1:
        raise ValueError(f"Not enough data to calculate support and resistance levels for {ticker}.")
    
    previous_day_data = data.iloc[0]
    pivot_points = calculate_pivot_points(pd.DataFrame([previous_day_data]))
    return pivot_points.iloc[0]

# Function to create a downloadable Excel file
def create_download_link_excel(data, filename="support_resistance_levels.xlsx"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name='Support_Resistance_Levels')
    return output.getvalue()

# Streamlit app title
st.title("Support and Resistance Levels Calculator")

# List of indices and their Yahoo Finance tickers (updated)
indices = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCAP NIFTY": "NIFTY_MID_SELECT.NS",
    "SENSEX": "^BSESN",
    "BANKEX": "BSE-BANK.BO",
    "ALL": "ALL"
}

# Dropdown menu for selecting the index
selected_index = st.selectbox("Select an Index", list(indices.keys()))

# Button to calculate support and resistance levels
if st.button("Get Support and Resistance Levels"):
    if selected_index == "ALL":
        all_levels = pd.DataFrame()
        text_buffer = ""
        for index, ticker in indices.items():
            if ticker == "ALL":
                continue
            try:
                levels = get_support_resistance_today(ticker)
                levels.name = index
                all_levels = pd.concat([all_levels, pd.DataFrame(levels).T])
                text_buffer += f"**{index} Support and Resistance Levels for Today**\n{levels.to_string()}\n\n"
            except ValueError as e:
                st.error(f"Error fetching data for {index}: {e}")
        
        st.write("**Support and Resistance Levels for All Indices**")
        st.dataframe(all_levels)
        
        # Add download button for Excel file
        excel_data = create_download_link_excel(all_levels)
        st.download_button(
            label="Download",
            data=excel_data,
            file_name="all_indices_support_resistance_levels.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Add copy to clipboard option with index names
        st.text_area(
            label="Copy to Clipboard",
            value=text_buffer,
            height=300
        )
    
    else:
        try:
            ticker = indices[selected_index]
            levels = get_support_resistance_today(ticker)
            st.write(f"**{selected_index} Support and Resistance Levels for Today**")
            st.dataframe(levels)
            
            # Create a text buffer for the download and copy options
            text_buffer = f"**{selected_index} Support and Resistance Levels for Today**\n{levels.to_string()}"
            
            # Add download button for Excel file
            excel_data = create_download_link_excel(pd.DataFrame(levels).T)
            st.download_button(
                label="Download",
                data=excel_data,
                file_name=f"{selected_index}_support_resistance_levels.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Add copy to clipboard option with index names
            st.text_area(
                label="Copy to Clipboard",
                value=text_buffer,
                height=300
            )
        except ValueError as e:
            st.error(e)
