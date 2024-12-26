import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure the combined data is loaded into session state
if "combined_df" in st.session_state:
    # Access the combined dataframe from session state
    combined_df = st.session_state["combined_df"]
    
    st.write("### Histogram of Numeric Columns")

    # Get numeric columns
    numeric_cols = combined_df.select_dtypes(include=['float64']).columns

    # Loop over numeric columns and plot histograms
    for col in numeric_cols:
        # Create a figure for the histogram
        fig, ax = plt.subplots()

        # Plot histogram
        ax.hist(combined_df[col], bins=20, density=True)

        # Set the labels and title
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {col}')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
else:
    st.error("No combined data available. Please upload the data first.")

