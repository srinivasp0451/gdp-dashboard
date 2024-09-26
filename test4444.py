import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to process extracted text into a usable DataFrame (adjust based on your OCR output structure)
def process_extracted_text(text):
    # This function will process the raw OCR text and convert it into a usable format (DataFrame)
    # Here, the assumption is that OCR gives us data in a tabular format.
    
    # Example processing logic (replace this with actual parsing logic based on your OCR output):
    lines = text.split('\n')  # Split text by lines
    data = []
    
    for line in lines:
        # Assuming each line contains: Strike Price, CE Premium %, PE Premium %, CE OI, PE OI (tab-separated or space-separated)
        # Adjust this part based on how your actual OCR text looks!
        columns = line.split()  # Adjust splitting logic (e.g., using spaces, tabs, etc.)
        
        if len(columns) >= 5:  # Ensure that we have enough columns (Strike Price, CE/PE premiums, OI, etc.)
            try:
                data.append({
                    'Strike Price': float(columns[0]),
                    'CE Premium Change %': float(columns[1]),
                    'PE Premium Change %': float(columns[2]),
                    'CE OI': int(columns[3]),
                    'PE OI': int(columns[4])
                })
            except ValueError:
                continue  # Skip lines with incorrect data format
    
    return pd.DataFrame(data)

# Function to recommend whether to buy CE or PE based on premium % and OI
def recommend_strike_price(option_chain_df):
    recommendations = []
    
    for index, row in option_chain_df.iterrows():
        if row['CE Premium Change %'] > 0 and row['CE OI'] > 10000:
            recommendations.append(f"Buy CE at Strike Price {row['Strike Price']} (Premium Change {row['CE Premium Change %']}%)")
        elif row['PE Premium Change %'] > 0 and row['PE OI'] > 10000:
            recommendations.append(f"Buy PE at Strike Price {row['Strike Price']} (Premium Change {row['PE Premium Change %']}%)")
    
    return recommendations

# Streamlit app layout
st.title("Dynamic Option Chain Analysis and Recommendations")

# Upload Option Chain Image for Premiums
premium_image = st.file_uploader("Upload Option Chain Image (Premium Data)", type=["png", "jpg", "jpeg"])

# Upload Option Chain Image for Open Interest (OI Data)
oi_image = st.file_uploader("Upload Option Chain Image (OI Data)", type=["png", "jpg", "jpeg"])

# Button to analyze the images and make recommendations
if st.button("Analyze and Recommend"):
    if premium_image and oi_image:
        # Open the uploaded images
        premium_img = Image.open(premium_image)
        oi_img = Image.open(oi_image)
        
        # Extract text from the premium and OI images using OCR
        premium_text = extract_text_from_image(premium_img)
        oi_text = extract_text_from_image(oi_img)
        
        st.subheader("Extracted Premium Data (Raw Text)")
        st.write(premium_text)  # Display the raw OCR text for review
        
        st.subheader("Extracted OI Data (Raw Text)")
        st.write(oi_text)  # Display the raw OCR text for review
        
        # Process the extracted text into DataFrames
        premium_df = process_extracted_text(premium_text)
        oi_df = process_extracted_text(oi_text)
        
        # Combine the dataframes (assuming same structure and same strike prices)
        combined_df = premium_df.merge(oi_df, on='Strike Price', suffixes=('_premium', '_oi'))
        
        st.write("Processed Option Chain Data:")
        st.dataframe(combined_df)  # Display the processed data
        
        # Generate recommendations
        recommendations = recommend_strike_price(combined_df)
        
        st.subheader("Recommendations:")
        for rec in recommendations:
            st.write(rec)
    else:
        st.error("Please upload both premium and OI images.")
