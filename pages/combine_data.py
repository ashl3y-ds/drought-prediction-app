# Upload data page
import streamlit as st
import pandas as pd

st.title("Upload Data")
st.write("This page allows you to upload datasets.")

uploaded_file1 = st.file_uploader("Upload the first CSV file", type="csv")
uploaded_file2 = st.file_uploader("Upload the second CSV file", type="csv")

if uploaded_file1 and uploaded_file2:
    # Read the uploaded CSV files
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    st.success("Datasets combined successfully!")

    # Save combined data to session state for later use
    st.session_state["combined_df"] = combined_df

    # Display the combined dataframe
    st.write(combined_df)

    # Save combined data to session state for later use
    st.session_state["combined_df"] = combined_df

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
