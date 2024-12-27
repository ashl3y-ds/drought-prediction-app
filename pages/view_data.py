import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
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

columns_to_drop = ['FIPS', 'DATE', 'PRECTOT', 'WS10M', 'WS10M_MIN', 'WS50M_MIN', 'YEAR']
dataset = dataset.drop(columns=columns_to_drop)

# Extract training features and target
X = dataset.iloc[:, :-1].values  # Features (exclude the target column)
y = dataset.iloc[:, -1].values   # Target (drought level)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Drought level predictor
def predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day):
    # Input features
    input_features = [ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day]
    # Scale the input
    scaled_features = scaler.transform([input_features])
    # Predict drought level
    prediction = model.predict(scaled_features)[0]
    return prediction

# Streamlit app
st.title("Drought Prediction Application")
st.header("Input Parameters")

ps = st.number_input("Pressure (PS, 66 to 103):", min_value=66.0, max_value=103.0)
qv2m = st.number_input("Specific Humidity at 2m (QV2M, 0.10 to 22.50):", min_value=0.1, max_value=22.5)
t2m = st.number_input("Temperature at 2m (T2M, -38.50 to 40.30):", min_value=-38.50, max_value=40.30)
t2mdew = st.number_input("Dew Temperature at 2m (T2MDEW, -41.50 to 27.05):", min_value=-41.50, max_value=27.05)
t2mwet = st.number_input("Wet Bulb Temperature at 2m (T2MWET, -38.50 to 27.00):", min_value=-38.50, max_value=27.00)
t2m_max = st.number_input("Max Temperature at 2m (T2M_MAX, -31.30 to 48.30):", min_value=-31.30, max_value=48.30)
t2m_min = st.number_input("Min Temperature at 2m (T2M_MIN, -45.40 to 32.30):", min_value=-45.40, max_value=32.30)
t2m_range = st.number_input("Temperature Range at 2m (T2M_RANGE, 0.12 to 29.65):", min_value=0.12, max_value=29.65)
ts = st.number_input("Surface Temperature (TS, 41.23 to 43.47):", min_value=-41.23, max_value=43.47)
ws10m_max = st.number_input("Max Wind Speed at 10m (WS10M_MAX, 0.60 to 24.90):", min_value=0.60, max_value=24.90)
ws10m_range = st.number_input("Wind Speed Range at 10m (WS10M_RANGE, 0.23 to 22.00):", min_value=0.23, max_value=22.00)
ws50m = st.number_input("Average Wind Speed at 50m (WS50M, 0.50 to 20.58):", min_value=0.50, max_value=20.58)
ws50m_max = st.number_input("Max Wind Speed at 50m (WS50M_MAX, 1.04 to 29.90):", min_value=1.04, max_value=29.90)
ws50m_range = st.number_input("Wind Speed Range at 50m (WS50M_RANGE, 0.45 to 26.30):", min_value=0.45, max_value=26.30)
month = st.number_input("Month (1 to 12):", min_value=1, max_value=12)
day = st.number_input("Day of the Month (1 to 31):", min_value=1, max_value=31)

# Predict drought level
if st.button("Predict Drought Level"):
    prediction = predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day)
    drought_levels = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought"]
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{drought_levels[prediction]}**")
    # Make prediction
    prediction = make_prediction(features, model, scaler)
    
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{prediction}**")
