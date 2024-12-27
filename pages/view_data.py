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

# Load and preprocess the dataset
combined_df = load_and_combine_data()
columns_to_drop = ['FIPS','DATE','PRECTOT','WS10M','WS10M_MIN','WS50M_MIN','YEAR']
dataset = combined_df.drop(columns=columns_to_drop)

# Ensure all values in the dataset are numeric, coercing errors to NaN
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.dropna()  # Drop rows with NaN values

# Extract features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# Check if X is 2D and print the shape
print("X shape:", X.shape)

# Check for NaN or Inf values
print("NaN values in X:", np.isnan(X).sum())
print("Inf values in X:", np.isinf(X).sum())

# Handle NaNs and Infinities if present
X = np.nan_to_num(X)

# Ensure X is 2D
if X.ndim == 1:
    X = X.reshape(-1, 1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Drought prediction function
def predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day):
    input_features = [ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day]
    scaled_features = scaler.transform([input_features])  # Scale input features
    prediction = model.predict(scaled_features)[0]  # Predict drought level
    return prediction

# Streamlit App
st.title("Drought Prediction Application")
st.header("Enter Input Parameters")

# Input fields (same as before)

# Predict drought level
if st.button("Predict Drought Level"):
    prediction = predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day)
    drought_levels = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought"]
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{drought_levels[prediction]}**")

