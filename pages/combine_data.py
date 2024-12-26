# Upload data page
import streamlit as st
import pandas as pd

st.title("Read Data")
st.write("This page directly reads and combines predefined datasets.")

# Define file paths or URLs to the datasets
url1 = "https://github.com/ashl3y-ds/drought-prediction-app/blob/main/data/cleaned_drought_data_part1.csv"
url2 = "https://github.com/ashl3y-ds/drought-prediction-app/blob/main/data/cleaned_drought_data_part2.csv"

@st.cache_data
def load_and_combine_data():
    # Read CSV files
    df1 = pd.read_csv(url1, na_values=['NA', ''], encoding='utf-8')  # Replace with the raw URL to data_part1.csv
    df2 = pd.read_csv(url2, na_values=['NA', ''], encoding='utf-8')  # Replace with the raw URL to data_part2.csv

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load combined dataset
combined_df = load_and_combine_data()
st.success("Datasets loaded and combined successfully!")

# Display the combined dataframe
st.write(combined_df)

# Save combined data to session state for later use
st.session_state["combined_df"] = combined_df
 
