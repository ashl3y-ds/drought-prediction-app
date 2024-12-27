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

columns_to_drop = ['FIPS', 'DATE', 'PRECTOT', 'WS10M', 'WS10M_MIN', 'WS50M_MIN', 'YEAR']
dataset = combined_df.drop(columns=columns_to_drop)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# StandardScaler for normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Streamlit App for Multiple Input Predictions
st.title("Drought Prediction Application")
st.header("Input Parameters (Multiple Records)")

# Input the feature values in batch (multiple inputs)
with st.form(key="batch_predict_form"):
    num_inputs = st.number_input("Number of data rows to predict:", min_value=1, max_value=10, value=1)
    
    input_data = []
    for i in range(num_inputs):
        st.subheader(f"Record {i+1}")
        ps = st.number_input(f"Pressure (PS) for Record {i+1}:", min_value=66.0, max_value=103.0)
        qv2m = st.number_input(f"Specific Humidity at 2m (QV2M) for Record {i+1}:", min_value=0.1, max_value=22.5)
        t2m = st.number_input(f"Temperature at 2m (T2M) for Record {i+1}:", min_value=-38.50, max_value=40.30)
        t2mdew = st.number_input(f"Dew Temperature at 2m (T2MDEW) for Record {i+1}:", min_value=-41.50, max_value=27.05)
        t2mwet = st.number_input(f"Wet Bulb Temperature at 2m (T2MWET) for Record {i+1}:", min_value=-38.50, max_value=27.00)
        t2m_max = st.number_input(f"Max Temperature at 2m (T2M_MAX) for Record {i+1}:", min_value=-31.30, max_value=48.30)
        t2m_min = st.number_input(f"Min Temperature at 2m (T2M_MIN) for Record {i+1}:", min_value=-45.40, max_value=32.30)
        t2m_range = st.number_input(f"Temperature Range at 2m (T2M_RANGE) for Record {i+1}:", min_value=0.12, max_value=29.65)
        ts = st.number_input(f"Surface Temperature (TS) for Record {i+1}:", min_value=-41.23, max_value=43.47)
        ws10m_max = st.number_input(f"Max Wind Speed at 10m (WS10M_MAX) for Record {i+1}:", min_value=0.60, max_value=24.90)
        ws10m_range = st.number_input(f"Wind Speed Range at 10m (WS10M_RANGE) for Record {i+1}:", min_value=0.23, max_value=22.00)
        ws50m = st.number_input(f"Average Wind Speed at 50m (WS50M) for Record {i+1}:", min_value=0.50, max_value=20.58)
        ws50m_max = st.number_input(f"Max Wind Speed at 50m (WS50M_MAX) for Record {i+1}:", min_value=1.04, max_value=29.90)
        ws50m_range = st.number_input(f"Wind Speed Range at 50m (WS50M_RANGE) for Record {i+1}:", min_value=0.45, max_value=26.30)
        month = st.number_input(f"Month (1 to 12) for Record {i+1}:", min_value=1, max_value=12)
        day = st.number_input(f"Day of the Month (1 to 31) for Record {i+1}:", min_value=1, max_value=31)
        
        # Store all feature values into input_data list
        input_data.append([ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day])

    submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        # Scale the input data for predictions
        input_data_scaled = scaler.transform(input_data)
        
        # Get predictions for all inputs
        predictions = model.predict(input_data_scaled)
        
        # Show the prediction results
        drought_levels = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought"]
        st.subheader("Prediction Results")
        for i, prediction in enumerate(predictions):
            st.write(f"Record {i+1}: {drought_levels[prediction]}")
