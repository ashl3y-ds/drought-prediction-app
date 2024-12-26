import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to show histograms
def show_histograms():
    # Load the data (Replace with actual file path or data)
    # Plot histograms of numeric columns
    st.write("### Histogram of Numeric Columns")

    # Get numeric columns
    numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns

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

