import streamlit as st

# Set the title for your app's home page
st.title("Welcome to the Drought Prediction App 🌦️")

# Provide navigation information to the user
st.write("Use the sidebar to navigate to different sections of this app.")
st.write(
    """
    ### Available Sections:
    - **Display Data:** Upload drought-related datasets.
    - **Data Visualization:** Display diagrams to show information.
    - **Data Modeling:** Train the model with dataset.
    - **Predict New Data:** Predict new data drought level.
    """
)

# Add an optional image or further details
st.image("image/climate-change-and-droug.jpeg__800x500_q80_crop_subject_location-570,250_subsampling-2.jpg", caption="Example drought monitoring")  # Replace with your image URL or file path.
