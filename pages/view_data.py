# Preprocess target labels to integers
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert drought levels to integer indices

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# Prediction function
def predict_drought_level(ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day):
    input_features = [ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day]
    scaled_features = scaler.transform([input_features])  # Scale input features
    prediction = int(model.predict(scaled_features)[0])  # Predict drought level and cast to int
    return prediction

# Streamlit App
st.title("Drought Prediction Application")
st.header("Enter Input Parameters")

drought_levels = ["No Drought", "Mild Drought", "Moderate Drought", "Severe Drought"]

with st.form(key='drought_prediction_form'):
    # Input Fields (Same as before)
    ...

    submit_button = st.form_submit_button(label='Predict Drought Level')

# Predict drought level
if submit_button:
    prediction = predict_drought_level(
        ps, qv2m, t2m, t2mdew, t2mwet, t2m_max, t2m_min, t2m_range, ts, 
        ws10m_max, ws10m_range, ws50m, ws50m_max, ws50m_range, month, day
    )
    st.subheader("Prediction Result")
    st.write(f"Drought Level: **{drought_levels[prediction]}**")

