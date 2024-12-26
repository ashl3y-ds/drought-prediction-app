import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to show histograms
def show_histograms():
    # Load the data (Replace with actual file path or data)
    @st.cache_data
    def load_data():
        df = pd.read_csv('data/cleaned_drought_data_part1.csv')  # Example path to your dataset
        return df

    df = load_data()

    # Define the columns to visualize (choose relevant columns)
    measures_column_list = ['T2M', 'PRECTOT', 'PS', 'T2M_MAX']  # Replace with actual columns

    # Plot each column as a histogram
    for col_name in measures_column_list:
        fig, ax = plt.subplots()  # Create a figure

        # Create a histogram
        ax.hist(df[col_name], bins=20, density=True)
        
        # Set labels and title
        ax.set_xlabel(col_name)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {col_name}')

        # Display the plot
        st.pyplot(fig)
