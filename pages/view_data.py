import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('your_dataset.csv')  # Replace with your actual file path
    return df

# Function to train model and scale features
def train_model():
    # Load dataset
    df = load_data()

    # Select relevant columns for training (modify according to your dataset)
    features = df[['PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 
                   'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M_MAX', 'WS10M_RANGE', 
                   'WS50M', 'WS50M_MAX', 'WS50M_RANGE', 'month', 'day']].values
    target = df['drought_label'].values  # 'drought_label' is the column containing your labels

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Return both model and scaler in-memory
    return model, scaler

# Function to predict drought level
def make_prediction(features, model, scaler):
    # Scale the features using the trained scaler
    scaled_features = scaler.transform([features])

    # Make the prediction
    prediction = model.predict(scaled_features)

    # Map prediction to drought level
    drought_levels = {0: "No drought", 1: "Mild drought", 2: "Moderate drought", 3: "Severe drought"}
    
    return drought_levels.get(prediction[0], "Unknown")

# Streamlit app interface
st.title("Drought Prediction Application")

st.header("Input Parameters")
ps = st.number_input("Pressure (PS):", min_value=900.0, max_value=1100.0, value=1012.0)
qv2m = st.number_input("Specific Humidity at 2m (QV2M):", min_value=0.0, max_value=20.0, value=5.0)
t2m = st.number_input("Temperature at 2m (T2M, °C):", min_value=-50.0, max_value=50.0, value=25.0)
t2mdew = st.number_input("Dew Temperature at 2m (T2MDEW, °C):", min_value=-50.0, max_value=50.0, value=15.0)
t2mwet = st.number_input("Wet Bulb Temperature at 2m (T2MWET, °C):", min_value=-50.0, max_value=50.0, value=20.0)
t2m_max = st.number_input("Max Temperature at 2m (T2M_MAX, °C):", min_value=-50.0, max_value=60.0, value=30.0)
t2m_min = st.number_input("Min Temperature at 2m (T2M_MIN, °C):", min_value=-50.0, max_value=50.0, value=10.0)
t2m_range = st.number_input("Temperature Range at 2m (T2M_RANGE, °C):", min_value=0.0, max_value=50.0, value=5.0)
ts = st.number_input("Surface Temperature (TS, °C):", min_value=-50.0, max_value=50.0, value=25.0)
ws10m_max = st.number_input("Max Wind Speed at 10m (WS10M_MAX, m/s):", min_value=0.0, max_value=50.0, value=10.0)
ws10m_range = st.number_input("Wind Speed Range at 10m (WS10M_RANGE, m/s):", min_value=0.0, max_value=50.0, value=5.0)
ws50m = st.number_input("Average Wind Speed at 50m (WS50M, m/s):", min_value=0.0, max_value=50.0, value=8.0)
ws50m_max = st.number_input("Max Wind Speed at 50m (WS50M_MAX, m/s):", min_value=0.0, max_value=50.0, value=12.0)
ws50m_range = st.number_input("Wind Speed Range at 50m (WS50M_RANGE, m/s):", min_value=0.0, max_value=50.0, value=4.0)
month = st.number_input("Month:", min_value=1, max_value=12, value=6)
day = st.number_input("Day of the Month:", min_value=1, max_value=31, value=15)

# Train the model on first visit
if 'model' not in st.session_state:
    st.session_state['model'], st.session_state['scaler'] = train_model()

# Predict button
if st.button("Predict Drought Level"):
    model = st.session_state['model']
    scaler = st.session_state['scaler']

    # Features collected from user input
    features = [
        ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, 
        t2m_min, t2m_range, ts, ws10m_max, ws10m_range, 
        ws50m, ws50m_max, ws50m_range, month, day
    ]
    
    # Make prediction
    prediction = make_prediction(features, model, scaler)
    
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{prediction}**")
