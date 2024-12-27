import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_and_combine_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load and preprocess the dataset
combined_df = load_and_combine_data()

# Drop unnecessary columns and define features explicitly
columns_to_drop = ['fips', 'date', 'PRECTOT', 'WS10M', 'WS10M_MIN', 'WS50M_MIN', 'year']
columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
dataset = combined_df.drop(columns=columns_to_drop)

# Define features and target explicitly
features = [
    "ps", "qv2m", "t2m", "t2mdew", "t2mwet", "t2m_max", "t2m_min",
    "t2m_range", "ts", "ws10m_max", "ws10m_range", "ws50m",
    "ws50m_max", "ws50m_range", "month", "day"
]
target = "score"

# Ensure only valid columns exist in the dataset
features = [f for f in features if f in dataset.columns]

X = dataset[features].values  # Features
y = dataset[target].values   # Target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Prediction function
def predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day):
    input_features = [ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day]
    scaled_features = scaler.transform([input_features])  # Scale input features
    prediction = model.predict(scaled_features)[0]  # Predict drought level
    print(f"Input Features: {input_features}")  # Debugging step
    print(f"Predicted Level: {prediction}")  # Debugging step
    return prediction

# Streamlit App
st.title("Drought Prediction Application")
st.header("Enter Input Parameters")

with st.form(key='drought_prediction_form'):
    # Input Fields
    ps = st.number_input("Pressure (66 to 103):", min_value=66.0, max_value=103.0)
    qv2m = st.number_input("Specific Humidity (0.10 to 22.50):", min_value=0.1, max_value=22.5)
    t2m = st.number_input("Temperature (T2M, -38.50 to 40.30):", min_value=-38.50, max_value=40.30)
    t2mdew = st.number_input("Dew Temperature (T2MDEW, -41.50 to 27.05):", min_value=-41.50, max_value=27.05)
    t2mwet = st.number_input("Wet Bulb Temperature (T2MWET, -38.50 to 27.00):", min_value=-38.50, max_value=27.00)
    t2m_max = st.number_input("Max Temperature (T2M_MAX, -31.30 to 48.30):", min_value=-31.30, max_value=48.30)
    t2m_min = st.number_input("Min Temperature (T2M_MIN, -45.40 to 32.30):", min_value=-45.40, max_value=32.30)
    t2m_range = st.number_input("Temperature Range (T2M_RANGE, 0.12 to 29.65):", min_value=0.12, max_value=29.65)
    ts = st.number_input("Surface Temperature (TS, 41.23 to 43.47):", min_value=-41.23, max_value=43.47)
    ws10m_max = st.number_input("Max Wind Speed at 10m (0.60 to 24.90):", min_value=0.60, max_value=24.90)
    ws10m_range = st.number_input("Wind Speed Range (0.23 to 22.00):", min_value=0.23, max_value=22.00)
    ws50m = st.number_input("Avg Wind Speed at 50m (0.50 to 20.58):", min_value=0.50, max_value=20.58)
    ws50m_max = st.number_input("Max Wind Speed at 50m (1.04 to 29.90):", min_value=1.04, max_value=29.90)
    ws50m_range = st.number_input("Wind Speed Range (0.45 to 26.30):", min_value=0.45, max_value=26.30)
    month = st.number_input("Month (1-12):", min_value=1, max_value=12)
    day = st.number_input("Day (1-31):", min_value=1, max_value=31)

    submit_button = st.form_submit_button(label='Predict Drought Level')

# Define drought levels (explicit levels)
drought_levels = [1, 2, 3, 4, 5]

# Predict and display result in the Streamlit app
if submit_button:
    prediction = predict_drought_level(
        ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts,
        ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day
    )
    
    # Validate prediction
    if prediction not in drought_levels:  # Compare against `drought_levels`
        st.error(f"Prediction out of bounds. Got '{prediction}', which is not a valid drought level.")
    else:
        st.subheader("Prediction Result")
        st.write(f"Drought Level: **{prediction}**")
