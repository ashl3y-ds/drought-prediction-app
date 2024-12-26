import streamlit as st

# Set the title for your app's home page
st.title("Welcome to the Drought Prediction App 🌦️")

# Provide navigation information to the user
st.write("Use the sidebar to navigate to different sections of this app.")
st.write(
    """
    ### Available Sections:
    - **Upload Data:** Upload drought-related datasets.
    - **Preprocess Data:** Clean and preprocess your data.
    - **View Data:** Explore your datasets interactively.
    """
)

# Add an optional image or further details
st.image("https://example.com/some-image.jpg", caption="Example drought monitoring")  # Replace with your image URL or file path.
