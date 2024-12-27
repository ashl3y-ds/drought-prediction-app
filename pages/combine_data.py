# Upload data page
import streamlit as st
import pandas as pd

st.title("Read Data")
st.write("This page directly reads and combines predefined datasets.")

@st.cache_data
def load_and_combine_data():
    # Read CSV files
    df1 = pd.read_csv('data/cleaned_drought_data_part1.csv')
    df2 = pd.read_csv('data/cleaned_drought_data_part2.csv')  # Replace with the raw URL to data_part2.csv

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load combined dataset
combined_df = load_and_combine_data()
st.success("Datasets loaded and combined successfully!")

# Convert the 'score' column to integer, filling NaNs with 0
combined_df['score'] = combined_df['score'].fillna(0).astype(int)

# Show the value counts for rounded 'score'
st.write("### Value Counts of Rounded 'score' Column")
rounded_score_counts = combined_df['score'].round().value_counts()
st.write(rounded_score_counts)

# Show descriptive statistics for numeric columns
st.write("### Descriptive Statistics (Numeric Columns)")
st.write(combined_df.describe())

# Show descriptive statistics for categorical columns
st.write("### Descriptive Statistics (Categorical Columns)")
st.write(combined_df.describe(include=['object']))

# Display the combined dataframe
st.write("### Combined Data")
st.write(combined_df)

# Save combined data to session state for later use
st.session_state["combined_df"] = combined_df

 
