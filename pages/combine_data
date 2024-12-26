import streamlit as st
import pandas as pd

st.title("Upload Data")
st.write("This page allows you to upload datasets.")

uploaded_file1 = st.file_uploader("Upload the first CSV file", type="csv")
uploaded_file2 = st.file_uploader("Upload the second CSV file", type="csv")

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    st.success("Datasets combined successfully!")
    st.write(combined_df)

    # Save combined data to session state
    st.session_state["combined_df"] = combined_df
