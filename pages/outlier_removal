import streamlit as st
import pandas as pd

# Load and combine the datasets
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")  # Replace with your file
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")  # Replace with your file
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Outlier Removal Function
def remove_outliers(df, features):
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        df = df[(df[feature] <= mean + 3 * std) & (df[feature] >= mean - 3 * std)]
    return df

# Load the dataset
drought_df = load_data()

# Streamlit UI
st.title("Outlier Removal for Dataset Features")
st.write("This page allows you to remove outliers using the 3-sigma rule for selected features.")

# Dropdown for selecting features to clean
all_features = drought_df.columns.tolist()
selected_features = st.multiselect(
    "Select Features for Outlier Removal (Default: All Features):", 
    options=all_features, 
    default=all_features
)

# Remove outliers on button click
if st.button("Remove Outliers"):
    cleaned_df = remove_outliers(drought_df.copy(), selected_features)
    st.success(f"Outliers removed! Total rows after cleaning: {len(cleaned_df)}")
    st.write(cleaned_df)

    # Save cleaned data in session state for other app sections
    st.session_state["cleaned_df"] = cleaned_df
else:
    st.write("No outliers removed yet.")
