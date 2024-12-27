import streamlit as st

st.write("Drought Prediction")

if "cleaned_df" in st.session_state:
    st.dataframe(st.session_state["combined_df"])
else:
    st.warning("No dataset found! Please upload and combine data on the Upload Data page.")
