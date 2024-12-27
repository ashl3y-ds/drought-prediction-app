import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and combine datasets
@st.cache_data
def load_and_combine_data():
    df1 = pd.read_csv('data/cleaned_drought_data_part1.csv')
    df2 = pd.read_csv('data/cleaned_drought_data_part2.csv')  # Replace with the raw URL to data_part2.csv
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load and preprocess data
combined_df = load_and_combine_data()

st.write("Columns in the DataFrame:", combined_df.columns)
