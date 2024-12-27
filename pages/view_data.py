import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
@st.cache_data
def load_and_combine_data():
    df1 = pd.read_csv('data/cleaned_drought_data_part1.csv')
    df2 = pd.read_csv('data/cleaned_drought_data_part2.csv')
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load combined dataset
combined_df = load_and_combine_data()

# Clean column names
combined_df.columns = combined_df.columns.str.strip()

# Columns to drop
columns_to_drop = ['FIPS', 'DATE', 'PRECTOT', 'WS10M', 'WS10M_MIN', 'WS50M_MIN', 'YEAR']

# Check which columns are present and drop them if they exist
existing_columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
dataset = combined_df.drop(columns=existing_columns_to_drop)

# Ensure all values are numeric
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Remove rows with NaN values
dataset = dataset.dropna()

# Extract features and target
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# Check shape and handle NaNs or Inf values
print("X shape:", X.shape)
X = np.nan_to_num(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Function to predict drought level
def predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day):
    input_features = [ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day]
    scaled_features = scaler.transform([input_features])
    prediction = model.predict(scaled_features)[0]
    return prediction

# Streamlit app
st.title("Drought Prediction Application")
st.header("Enter Input Parameters")

# Input fields for prediction (same as before)

if st.button("Predict Drought Level"):
    prediction = predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day)
    drought_levels = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought"]
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{drought_levels[prediction]}**")
